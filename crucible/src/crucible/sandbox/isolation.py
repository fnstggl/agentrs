"""Isolation for native tools that process agent-supplied bytes outside the Python sandbox (Part 16).

* `tool_command(argv, workdir)` - FFmpeg, ffprobe and SoX parse agent-uploaded media. They run under `jail.py`: an
  own unprivileged uid (not root, not the sandbox tenants' uids), ``no_new_privs``, Landlock (read-only system
  libraries and fonts, execute only the tool and the dynamic loader, read-write only the job's private temporary
  directory, no TCP) and seccomp (no sockets, no new processes, signals only to itself, no ptrace/mount/namespaces/
  BPF/io_uring/keyrings/uid changes). The caller keeps its rlimits and timeouts.
* `browser_executable()` - the Chromium that renders agent HTML (JavaScript disabled, `crucible.semantic.media`) and
  opens agent-authored pages with JavaScript (`crucible.web.browser`) is started through a launcher that applies the
  same jail in browser mode (AF_UNIX sockets only, for its own IPC; no TCP or UDP; processes allowed, because
  Chromium is multi-process) as its own uid. Chromium's own sandbox needs user namespaces or a setuid helper,
  neither of which this unprivileged container provides, so the jail is the outer boundary and Chromium runs with
  ``--no-sandbox`` inside it.

`available()` reports whether the jail can be applied here (root to change uid, Landlock, libseccomp). Callers fail
closed when it cannot, unless ``CRUCIBLE_ALLOW_UNJAILED_TOOLS=1`` is set explicitly (then the tools run with rlimits
only, as before, and `jailed()` is False).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile

JAIL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jail.py")
BASE = os.environ.get("CRUCIBLE_JAIL_DIR", "/var/tmp/crucible-jail")
UID_BASE = 0x5D000000
RO_SYSTEM = ["/usr", "/lib", "/lib64", "/etc/ld.so.cache", "/etc/fonts", "/etc/localtime", "/usr/share/zoneinfo",
             "/etc/alternatives"]


class JailUnavailable(RuntimeError):
    pass


def uid_for(kind: str) -> int:
    return UID_BASE + int(hashlib.blake2b(kind.encode(), digest_size=2).hexdigest(), 16)


def available() -> tuple[bool, str]:
    from .runner import capabilities
    c = capabilities()
    if not c["root"]:
        return False, "not running as root (cannot switch to the jail uid)"
    if c["landlock_abi"] < 1 or not c["seccomp"]:
        return False, f"Landlock/seccomp unavailable ({c})"
    return True, f"landlock ABI {c['landlock_abi']}, seccomp"


def allow_unjailed() -> bool:
    return os.environ.get("CRUCIBLE_ALLOW_UNJAILED_TOOLS") == "1"


def _loaders() -> list[str]:
    out = []
    for pat in ("/lib64/ld-linux*.so*", "/lib/x86_64-linux-gnu/ld-linux*.so*", "/usr/lib/x86_64-linux-gnu/ld-linux*.so*",
                "/lib/ld-linux*.so*", "/usr/lib64/ld-linux*.so*"):
        out.extend(os.path.realpath(p) for p in glob.glob(pat))
    return sorted(set(out))


def _base() -> str:
    os.makedirs(BASE, exist_ok=True)
    os.chmod(BASE, 0o711)
    return BASE


def _write_cfg(cfg: dict) -> str:
    fd, path = tempfile.mkstemp(prefix="jail-", suffix=".json", dir=_base())
    with os.fdopen(fd, "w") as fh:
        json.dump(cfg, fh)
    os.chmod(path, 0o600)
    return path


def tool_command(argv: list[str], workdir: str) -> tuple[list[str], str]:
    """(command line running `argv` jailed with read-write access to `workdir` only, config file to delete after)."""
    ok, why = available()
    if not ok:
        raise JailUnavailable(why)
    prog = shutil.which(argv[0]) or argv[0]
    real = os.path.realpath(prog)
    uid = uid_for("media-tools")
    cfg = {"uid": uid, "gid": uid, "mode": "tool", "ro": RO_SYSTEM + [os.path.dirname(real)],
           "exec": [real] + _loaders(), "rw": [workdir], "chown": [workdir]}
    path = _write_cfg(cfg)
    return [sys.executable, "-I", "-S", "-B", JAIL, path, real, *argv[1:]], path


# ---------------------------------------------------------------------------------------------- browser
def _browser_root(src_exe: str) -> str:
    """A copy of the Playwright Chromium directory the jail uid can reach (the cache under /root is mode 700). Files
    are hard links where the filesystem allows (no extra disk), copies otherwise; the copy is root-owned and
    read-only for others."""
    src_dir = os.path.dirname(os.path.realpath(src_exe))
    tag = hashlib.blake2b(src_dir.encode(), digest_size=6).hexdigest()
    dst = os.path.join(_base(), f"chromium-{tag}")
    done = os.path.join(dst, ".complete")
    if os.path.exists(done):
        return dst
    tmp = dst + f".tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)

    def link(s, d):
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
    shutil.copytree(src_dir, tmp, copy_function=link, symlinks=True)
    for root, dirs, files in os.walk(tmp):
        os.chmod(root, 0o755)
    try:
        os.rename(tmp, dst)
    except OSError:                       # another process finished first
        shutil.rmtree(tmp, ignore_errors=True)
    open(done, "w").close()
    return dst


def browser_executable(kind: str = "browser", tcp_ports: tuple[int, ...] = ()) -> tuple[str, dict]:
    """(executable path to pass to Playwright's `launch(executable_path=...)`, extra launch kwargs). The launcher
    jails Chromium as its own uid with no network sockets - or, with `tcp_ports` (the world's own web app on
    loopback), IPv4 TCP connect() to those ports only and no UDP/IPv6/raw sockets; the profile directory Playwright
    creates becomes the jail's."""
    ok, why = available()
    if not ok:
        raise JailUnavailable(why)
    from playwright.sync_api import sync_playwright  # noqa: F401 - the browser must be installed
    src = _playwright_chromium()
    root = _browser_root(src)
    exe = os.path.join(root, os.path.basename(src))
    uid = uid_for(kind)
    ports = sorted({int(p) for p in tcp_ports})
    tag = f"{uid}" + (("-" + "-".join(map(str, ports))) if ports else "")
    home = os.path.join(_base(), f"home-{uid}")
    os.makedirs(home, exist_ok=True)
    cfg = {"uid": uid, "gid": uid, "mode": "browser",
           "ro": RO_SYSTEM + ["/proc", "/sys", "/usr/share/fonts", "/var/cache/fontconfig"],
           "exec": [root] + _loaders(), "rw": [home], "chown": [home], "rw_from_argv": ["--user-data-dir="],
           "append_args": ["--disable-dev-shm-usage", "--no-sandbox"], "tcp_connect_ports": ports,
           "env": {"HOME": home, "TMPDIR": home, "XDG_CONFIG_HOME": home, "XDG_CACHE_HOME": home}}
    path = os.path.join(_base(), f"browser-{tag}.json")
    if not os.path.exists(path) or json.load(open(path)) != cfg:
        with open(path + ".tmp", "w") as fh:
            json.dump(cfg, fh)
        os.chmod(path + ".tmp", 0o600)
        os.replace(path + ".tmp", path)
    launcher = os.path.join(_base(), f"browser-{tag}.sh")
    body = f"#!/bin/sh\nexec {sys.executable} -I -S -B {JAIL} {path} {exe} \"$@\"\n"
    if not os.path.exists(launcher) or open(launcher).read() != body:
        with open(launcher + ".tmp", "w") as fh:
            fh.write(body)
        os.chmod(launcher + ".tmp", 0o755)
        os.replace(launcher + ".tmp", launcher)
    return launcher, {"chromium_sandbox": False}


_CHROMIUM: list[str] = []


def _playwright_chromium() -> str:
    """Path of Playwright's Chromium, without starting Playwright when the browser directory can be found (a caller
    may already be inside a running sync Playwright, which cannot be nested)."""
    if _CHROMIUM:
        return _CHROMIUM[0]
    base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or os.path.expanduser("~/.cache/ms-playwright")
    found = sorted(glob.glob(os.path.join(base, "chromium-*", "chrome-linux", "chrome")))
    if found:
        _CHROMIUM.append(found[-1])
        return found[-1]
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        _CHROMIUM.append(pw.chromium.executable_path)
    return _CHROMIUM[0]
