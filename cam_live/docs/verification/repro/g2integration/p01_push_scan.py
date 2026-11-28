"""G2-integration probe p01 — does tools/push_milestone.sh keep secrets out of what it pushes? (OP-08)

The script's header says it "secret-scans every commit not yet on the remote". This probe runs a copy of the script
against throwaway repositories with a *local bare* origin (no network) and fake credentials built at run time. The
only changes to the copy: the paths of this machine's credential files (/root/.config/...) are redirected to fake
files in the scratch directory (the real files are never read), the credential helper is /bin/true and the commit
range base falls back to the throwaway repository's first commit instead of 3968583.

Usage (from /workspace/camlive):
    .venv/bin/python cam_live/docs/verification/repro/g2integration/p01_push_scan.py [--script PATH]
Prints [HOLDS ] / [BROKEN] per check; exit 1 if any property is broken.
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCRIPT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "tools", "push_milestone.sh"))
SCRATCH = os.environ.get("G2I_SCRATCH", "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/"
                                        "G2INTEGRATION/p01")
broken = 0
SCANNERS: list = []          # redirected copies of tools/secret_scan.py, installed untracked in each work tree


def check(name: str, ok: bool, detail: str = "") -> None:
    global broken
    broken += 0 if ok else 1
    print(f"[{'HOLDS ' if ok else 'BROKEN'}] {name}" + (f" :: {detail}" if detail else ""))


def fake_pat() -> str:          # credential-shaped, built at run time (never a literal in a file)
    return "".join(["gh", "p_", "Q" * 6, "7" * 6, "zZ" * 12])


def fake_real_secret() -> str:  # a value standing in for "a real credential of this machine"
    return "".join(["machine", "-secret-", "4f1c", "9e2b", "77aa"])


def sh(cmd, cwd, env=None, check_rc=True):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if check_rc and r.returncode != 0:
        raise RuntimeError(f"{cmd} failed: {r.stdout}{r.stderr}")
    return r


def make_script(src: str, root: str, base: str) -> str:
    text = open(src).read()
    cfg = os.path.join(root, "fakecfg")
    os.makedirs(os.path.join(cfg, "camlive"), exist_ok=True)
    os.makedirs(os.path.join(cfg, "agentrs"), exist_ok=True)
    text = text.replace("/root/.config/camlive", os.path.join(cfg, "camlive"))
    text = text.replace("/root/.config/agentrs", os.path.join(cfg, "agentrs"))
    text = text.replace("3968583", base)
    assert "/root/.config" not in text
    out = os.path.join(root, "push_copy.sh")
    with open(out, "w") as f:
        f.write(text)
    os.chmod(out, 0o700)
    scanner = os.path.join(os.path.dirname(os.path.abspath(src)), "secret_scan.py")
    if "secret_scan.py" in text and os.path.exists(scanner):      # round 2: the script delegates to the scanner
        stext = open(scanner).read().replace("/root/.config/camlive", os.path.join(cfg, "camlive"))
        stext = stext.replace("/root/.config/agentrs", os.path.join(cfg, "agentrs"))
        assert "/root/.config" not in stext
        SCANNERS.append(stext)
    with open(os.path.join(cfg, "camlive", "gh_token"), "w") as f:
        f.write(fake_real_secret() + "\n")
    return out


def fresh(name: str, script_src: str):
    root = tempfile.mkdtemp(prefix=name + "_", dir=SCRATCH)
    origin, work = os.path.join(root, "origin.git"), os.path.join(root, "work")
    env = dict(os.environ, GIT_AUTHOR_NAME="probe", GIT_AUTHOR_EMAIL="probe@localhost", GIT_COMMITTER_NAME="probe",
               GIT_COMMITTER_EMAIL="probe@localhost", CAM_LIVE_CRED_HELPER="/bin/true", GIT_TERMINAL_PROMPT="0")
    sh(["git", "init", "-q", "--bare", origin], root, env)
    sh(["git", "init", "-q", "-b", "feat/cam-live", work], root, env)
    sh(["git", "remote", "add", "origin", origin], work, env)
    os.makedirs(os.path.join(work, "cam_live", "tools"))
    with open(os.path.join(work, "README"), "w") as f:
        f.write("base\n")
    sh(["git", "add", "README"], work, env)
    sh(["git", "commit", "-q", "-m", "base"], work, env)
    base = sh(["git", "rev-parse", "HEAD"], work, env).stdout.strip()
    sh(["git", "push", "-q", "origin", "HEAD:refs/heads/feat/cam-live"], work, env)
    script = make_script(script_src, root, base)
    if SCANNERS:
        with open(os.path.join(work, "cam_live", "tools", "secret_scan.py"), "w") as f:
            f.write(SCANNERS[-1])
        with open(os.path.join(work, "cam_live", "tools", "secret_scan_allow.txt"), "w") as f:
            f.write("# probe allow list\ncam_live/tests/fixture_allowed.py\n")
    return root, origin, work, env, script


def commit(work, env, path, data: bytes, msg):
    p = os.path.join(work, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    sh(["git", "add", path], work, env)
    sh(["git", "commit", "-q", "-m", msg], work, env)


def remove(work, env, path, msg):
    sh(["git", "rm", "-q", path], work, env)
    sh(["git", "commit", "-q", "-m", msg], work, env)


def run_push(script, work, env):
    return sh(["bash", script], work, env, check_rc=False)


def origin_history_contains(origin, needle: str) -> bool:
    """Any blob reachable from the pushed branch (all commits, decompressed .gz too) containing ``needle``."""
    r = subprocess.run(["git", "--git-dir", origin, "rev-list", "--objects", "refs/heads/feat/cam-live"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        oid = line.split()[0]
        if subprocess.run(["git", "--git-dir", origin, "cat-file", "-t", oid], capture_output=True,
                          text=True).stdout.strip() != "blob":
            continue
        data = subprocess.run(["git", "--git-dir", origin, "cat-file", "blob", oid], capture_output=True).stdout
        if data[:2] == b"\x1f\x8b":
            try:
                data = gzip.decompress(data)
            except OSError:
                pass
        if needle.encode() in data:
            return True
    return False


def origin_history_contains_zip(origin, needle: str) -> bool:
    """Like origin_history_contains, also opening zip archives."""
    import io
    import zipfile
    if origin_history_contains(origin, needle):
        return True
    r = subprocess.run(["git", "--git-dir", origin, "rev-list", "--objects", "refs/heads/feat/cam-live"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        oid = line.split()[0]
        data = subprocess.run(["git", "--git-dir", origin, "cat-file", "-p", oid], capture_output=True).stdout
        if data[:2] == b"PK":
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    if any(needle.encode() in z.read(n) for n in z.namelist()):
                        return True
            except (zipfile.BadZipFile, OSError):
                pass
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=DEFAULT_SCRIPT)
    a = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    print(f"script under test: {a.script}")
    pat = fake_pat()

    # A (control) credential-shaped text added in a plain file of the pushed range: refused
    root, origin, work, env, script = fresh("A", a.script)
    commit(work, env, "cam_live/src/x.py", f"KEY = '{pat}'\n".encode(), "add key")
    r = run_push(script, work, env)
    check("A control: credential-shaped text added in the range is refused (non-zero exit) and not pushed",
          r.returncode != 0 and not origin_history_contains(origin, pat), f"exit {r.returncode}; {r.stdout.strip()[-120:]}")

    # B credential-shaped text added in one unpushed commit and removed in the next
    root, origin, work, env, script = fresh("B", a.script)
    commit(work, env, "cam_live/src/x.py", f"KEY = '{pat}'\n".encode(), "add key")
    remove(work, env, "cam_live/src/x.py", "remove key")
    r = run_push(script, work, env)
    check("B a credential added and removed again inside the unpushed commits is not pushed",
          not origin_history_contains(origin, pat), f"exit {r.returncode}; pushed history contains it: "
          f"{origin_history_contains(origin, pat)}; {r.stdout.strip()[-80:]}")

    # C credential-shaped text inside a compressed (binary) evidence file
    root, origin, work, env, script = fresh("C", a.script)
    commit(work, env, "cam_live/docs/evidence/report.json.gz", gzip.compress(f'{{"k": "{pat}"}}'.encode()), "add gz")
    r = run_push(script, work, env)
    check("C a credential inside a compressed / binary file is not pushed",
          not origin_history_contains(origin, pat), f"exit {r.returncode}; {r.stdout.strip()[-120:]}")

    # D (control) an exact credential of 'this machine' (fake file) in a plain added line: refused (exit 4)
    real = fake_real_secret()
    root, origin, work, env, script = fresh("D", a.script)
    commit(work, env, "cam_live/notes.txt", f"token {real}\n".encode(), "add note")
    r = run_push(script, work, env)
    check("D control: the machine's own credential in the range is refused (non-zero exit) and not pushed",
          r.returncode != 0 and not origin_history_contains(origin, real), f"exit {r.returncode}")

    # E the machine's own credential added and removed again inside the unpushed commits
    root, origin, work, env, script = fresh("E", a.script)
    commit(work, env, "cam_live/notes.txt", f"token {real}\n".encode(), "add note")
    remove(work, env, "cam_live/notes.txt", "remove note")
    r = run_push(script, work, env)
    check("E the machine's own credential added and removed inside the unpushed commits is not pushed",
          not origin_history_contains(origin, real), f"exit {r.returncode}")

    # F the machine's own credential inside a gzip file
    root, origin, work, env, script = fresh("F", a.script)
    commit(work, env, "cam_live/docs/evidence/run.json.gz", gzip.compress(f'{{"t": "{real}"}}'.encode()), "add gz")
    r = run_push(script, work, env)
    check("F the machine's own credential inside a compressed file is not pushed",
          not origin_history_contains(origin, real), f"exit {r.returncode}")

    # G (round 2) credential-shaped text inside a zip archive (deflate, no gzip header)
    import io
    import zipfile
    root, origin, work, env, script = fresh("G", a.script)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("notes.txt", ("padding " * 50) + f"key {pat}\n")
    commit(work, env, "cam_live/docs/evidence/bundle.zip", buf.getvalue(), "add zip")
    r = run_push(script, work, env)
    check("G a credential inside a zip archive is not pushed", not origin_history_contains_zip(origin, pat),
          f"exit {r.returncode}")

    # H (round 2) the same blob at an allow-listed path and at an ordinary path in one commit
    root, origin, work, env, script = fresh("H", a.script)
    data = f"KEY = '{pat}'\n".encode()
    for path in ("cam_live/tests/fixture_allowed.py", "cam_live/zz_leak.py"):   # the allowed path sorts first
        p = os.path.join(work, path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
    sh(["git", "add", "cam_live/tests/fixture_allowed.py", "cam_live/zz_leak.py"], work, env)
    sh(["git", "commit", "-q", "-m", "fixture and copy"], work, env)
    r = run_push(script, work, env)
    check("H a credential at an ordinary path is refused although the same content is also at an allow-listed path",
          r.returncode != 0, f"exit {r.returncode}; {r.stdout.strip()[-140:]}")

    print(f"== p01_push_scan: {broken} broken of 8 checks (throwaway repositories under {SCRATCH})")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
