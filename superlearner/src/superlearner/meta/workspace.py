"""Candidate workspaces: separate source trees, verified hashes, and isolated execution. PROTECTED.

A *candidate* is a complete copy of the ``superlearner`` package in its own directory with some of the
candidate-editable files replaced. Editable: ``cam/`` (controller, features, decision procedure, memory),
``algo/`` (learning rules and algorithms) and ``train/curriculum.py``. Everything else is protected; a candidate
whose protected files differ from the champion's is rejected before anything runs (`verify`).

`run_isolated` executes a module from a candidate tree through `meta/isolate.py` (unprivileged uid, resource
limits, Landlock filesystem allowlist, no TCP, seccomp). The process can read the Python runtime, its own tree and
the Crucible library; write only its own work directory; and talk to exactly two unix sockets: the frozen-LLM
server and the trusted env hub. It never holds evaluation worlds, evaluator code that counts, reward
definitions that count, or other candidates' files. If the host cannot provide this isolation, `run_isolated`
raises `IsolationUnavailable` - it never falls back to running candidate code unconfined.

`probe_isolation` actually starts a confined process and checks, from the inside, that reading a file outside the
allowlist, writing outside the work directory, and opening a TCP socket all fail, and that the permitted unix
socket works.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import pwd
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                       # .../src/superlearner (the champion / protected tree)
SRC = os.path.dirname(PKG)
ISOLATE = os.path.join(HERE, "isolate.py")
EDITABLE_DIRS = ("cam", "algo")
EDITABLE_FILES = ("train/curriculum.py",)
_PROBE: dict | None = None


class IsolationUnavailable(RuntimeError):
    pass


def is_editable(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    if ".." in rel.split("/") or rel.startswith("/") or not rel.endswith(".py"):
        return False
    return rel in EDITABLE_FILES or any(rel.startswith(d + "/") for d in EDITABLE_DIRS)


def _files(pkg: str) -> list[str]:
    out = []
    for dp, dns, fns in os.walk(pkg):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for fn in sorted(fns):
            if fn.endswith(".py"):
                out.append(os.path.relpath(os.path.join(dp, fn), pkg).replace("\\", "/"))
    return sorted(out)


def tree_hashes(pkg: str) -> dict:
    """``{"editable": h, "protected": h, "files": {rel: sha}}`` of a package tree."""
    he, hp, files = hashlib.sha256(), hashlib.sha256(), {}
    for rel in _files(pkg):
        data = open(os.path.join(pkg, rel), "rb").read()
        files[rel] = hashlib.sha256(data).hexdigest()[:16]
        h = he if is_editable(rel) else hp
        h.update(rel.encode())
        h.update(data)
    return {"editable": he.hexdigest()[:16], "protected": hp.hexdigest()[:16], "files": files}


def create_candidate(dest: str, changes: dict[str, str], base_pkg: str = PKG) -> dict:
    """Copy `base_pkg` to ``dest/src/superlearner`` and replace the given editable files. Returns the record."""
    for rel in changes:
        if not is_editable(rel):
            raise PermissionError(f"{rel} is not candidate-editable")
    pkg = os.path.join(dest, "src", "superlearner")
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    shutil.copytree(base_pkg, pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for rel, content in changes.items():
        full = os.path.join(pkg, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
    h = tree_hashes(pkg)
    base = tree_hashes(base_pkg)
    rec = {"path": dest, "pkg": pkg, "editable_hash": h["editable"], "protected_hash": h["protected"],
           "base_editable_hash": base["editable"], "changed": sorted(r for r in h["files"] if h["files"][r] != base["files"].get(r)),
           "protected_ok": h["protected"] == base["protected"], "created": time.time()}
    json.dump(rec, open(os.path.join(dest, "candidate.json"), "w"), indent=1)
    return rec


def verify(dest: str, base_pkg: str = PKG) -> tuple[bool, str]:
    """A candidate may run only if its protected files are byte-identical to the evaluator's tree (`base_pkg`)."""
    pkg = os.path.join(dest, "src", "superlearner")
    if not os.path.isdir(pkg):
        return False, "no source tree"
    h, base = tree_hashes(pkg), tree_hashes(base_pkg)
    if h["protected"] != base["protected"]:
        bad = sorted(r for r in set(h["files"]) | set(base["files"])
                     if not is_editable(r) and h["files"].get(r) != base["files"].get(r))
        return False, f"protected files differ from the evaluator's tree: {bad[:5]}"
    return True, "ok"


def _unprivileged() -> tuple[int, int]:
    pw = pwd.getpwnam("nobody")
    return pw.pw_uid, pw.pw_gid


def _chown_tree(path: str, uid: int, gid: int) -> None:
    if os.getuid() != 0:
        return
    for dp, dns, fns in os.walk(path):
        os.chown(dp, uid, gid)
        for fn in fns:
            try:
                os.chown(os.path.join(dp, fn), uid, gid)
            except OSError:
                pass


def crucible_src() -> str:
    from .. import CRUCIBLE_SRC
    return CRUCIBLE_SRC


def runtime_read_paths() -> list[str]:
    paths = [sys.prefix, sys.base_prefix, "/usr/lib", "/lib", "/lib64", "/usr/lib64", "/usr/local/lib",
             "/etc/ld.so.cache", "/etc/passwd", "/etc/nsswitch.conf", "/etc/localtime", "/usr/share/zoneinfo",
             "/proc", "/sys/devices/system/cpu", "/usr/bin/env", os.path.realpath(sys.executable), ISOLATE]
    return [p for p in dict.fromkeys(paths) if os.path.exists(p)]


def run_isolated(module: str, args: list[str], cand_src: str, workdir: str, timeout_s: float = 3600.0,
                 gpu: bool = False, env: dict | None = None, limits: dict | None = None, extra_read: tuple = (),
                 log_path: str | None = None) -> dict:
    """Run ``python -m module args`` from the candidate tree `cand_src` (its ``src`` directory), confined."""
    if not os.path.exists(ISOLATE):
        raise IsolationUnavailable("isolate.py is missing")
    uid, gid = _unprivileged() if os.getuid() == 0 else (os.getuid(), os.getgid())
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "tmp"), exist_ok=True)
    _chown_tree(workdir, uid, gid)
    lim = dict({"nproc": 1024, "fsize": 8 * 2**30, "nofile": 4096, "cpu_s": int(timeout_s * 64)}, **(limits or {}))
    devices = ["/dev/null", "/dev/urandom", "/dev/zero", "/dev/random"]
    gpu_read: list[str] = []
    gpu_env: dict = {}
    if gpu:
        devices += sorted(glob.glob("/dev/nvidia*")) + sorted(glob.glob("/dev/nvidia-caps/*"))
        gpu_read = [p for p in ("/sys", "/usr/local/nvidia", "/usr/local/cuda", "/etc/OpenCL", "/usr/share/nvidia") if os.path.exists(p)]
        gpu_env = {k: os.environ[k] for k in ("LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES")
                   if k in os.environ}
    cfg = {"uid": uid, "gid": gid, "cwd": workdir,
           "read": runtime_read_paths() + gpu_read + [cand_src, crucible_src()] + list(extra_read),
           "write": [workdir], "devices": devices, "sys_path": [cand_src, crucible_src()],
           "env": dict({"HOME": workdir, "TMPDIR": os.path.join(workdir, "tmp"), "PYTHONHASHSEED": "0", "LANG": "C.UTF-8",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "PATH": os.path.dirname(sys.executable) + ":/usr/bin:/bin", "SUPERLEARNER_ISOLATED": "1",
                        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0") if gpu else ""},
                       **gpu_env, **(env or {})),
           "env_keep": [], **lim}
    argv = [sys.executable, "-I", "-B", ISOLATE, json.dumps(cfg), module] + list(args)
    t0 = time.time()
    out = open(log_path, "ab") if log_path else subprocess.PIPE
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT if log_path else subprocess.PIPE,
                            cwd="/", start_new_session=True, close_fds=True, env={"PATH": "/usr/bin:/bin"})
    timed_out = False
    try:
        so, se = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        so, se = b"", b""
    finally:
        try:                                   # process cleanup: the whole session, whatever it spawned
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        if log_path:
            out.close()
    tail = ""
    if log_path and os.path.exists(log_path):
        with open(log_path, "rb") as f:
            f.seek(max(0, os.path.getsize(log_path) - 4000))
            tail = f.read().decode("utf-8", "replace")
    return {"returncode": proc.returncode, "timed_out": timed_out, "wall_s": round(time.time() - t0, 2),
            "stdout": (so or b"").decode("utf-8", "replace")[-6000:] if not log_path else tail,
            "stderr": (se or b"").decode("utf-8", "replace")[-6000:] if not log_path else ""}


PROBE_SRC = r'''
import json, os, socket, sys
r = {"uid": os.getuid()}
canary, sockp, outside = sys.argv[1], sys.argv[2], sys.argv[3]
def tryit(name, fn):
    try:
        fn(); r[name] = "ALLOWED"
    except Exception as e:
        r[name] = f"denied ({type(e).__name__})"
tryit("read_outside_allowlist", lambda: open(canary).read())
tryit("read_home", lambda: os.listdir(os.path.expanduser("/root")))
tryit("write_outside_workdir", lambda: open(outside, "w").write("x"))
tryit("write_own_source", lambda: open(os.path.join(sys.path[0], "probe_write.txt"), "w").write("x"))
tryit("tcp_socket", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("1.1.1.1", 80)))
tryit("udp_socket", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
def unix():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(sockp); s.sendall(b"hi"); s.close()
tryit("unix_socket_to_trusted_service", unix)
tryit("write_workdir", lambda: open("ok.txt", "w").write("x"))
tryit("setuid_root", lambda: os.setuid(0))
if len(sys.argv) > 4 and sys.argv[4] == "gpu":
    try:
        import torch
        r["torch"] = torch.__version__; r["cuda"] = bool(torch.cuda.is_available())
        if r["cuda"]:
            r["cuda_op"] = float((torch.ones(4, device="cuda") * 2).sum())
    except Exception as e:
        r["torch_error"] = f"{type(e).__name__}: {e}"[:300]
print("PROBE " + json.dumps(r))
'''


def probe_isolation(gpu: bool = True, force: bool = False) -> dict:
    """Start a confined process and check the confinement from inside. Cached per process."""
    global _PROBE
    if _PROBE is not None and not force:
        return _PROBE
    base = tempfile.mkdtemp(prefix="sl-probe-", dir="/var/tmp")
    os.chmod(base, 0o755)
    rep: dict = {"level": "none", "usable": False}
    srv = None
    try:
        src = os.path.join(base, "src")
        os.makedirs(src)
        open(os.path.join(src, "probe.py"), "w").write(PROBE_SRC)
        canary = os.path.join(base, "canary_world_readable.txt")
        open(canary, "w").write("secret")
        os.chmod(canary, 0o644)
        sockp = os.path.join(base, "trusted.sock")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(sockp)
        os.chmod(sockp, 0o666)
        srv.listen(1)
        work = os.path.join(base, "work")
        res = run_isolated(os.path.join(src, "probe.py"), [canary, sockp, os.path.join(base, "outside.txt")] + (["gpu"] if gpu else []),
                           src, work, timeout_s=120, gpu=gpu)
        line = next((l for l in res["stdout"].splitlines() if l.startswith("PROBE ")), None)
        if line is None:
            rep["error"] = (res["stderr"] or res["stdout"])[-600:]
            _PROBE = rep
            return rep
        r = json.loads(line[6:])
        rep["inside"] = r
        must_deny = ("read_outside_allowlist", "read_home", "write_outside_workdir", "write_own_source", "tcp_socket",
                     "udp_socket", "setuid_root")
        ok = (all(str(r.get(k, "")).startswith("denied") for k in must_deny) and r.get("write_workdir") == "ALLOWED"
              and r.get("unix_socket_to_trusted_service") == "ALLOWED" and r.get("uid") != 0)
        rep.update(usable=bool(ok), level="uid+rlimits+landlock+seccomp" if ok else "insufficient",
                   gpu_inside=bool(r.get("cuda")) and "cuda_op" in r, torch_inside=r.get("torch"),
                   missing=["VM or container-grade isolation (no gVisor/Firecracker/user namespaces in this container)"])
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
    finally:
        if srv is not None:
            srv.close()
        shutil.rmtree(base, ignore_errors=True)
    _PROBE = rep
    return rep


def require_isolation() -> dict:
    rep = probe_isolation()
    if not rep.get("usable"):
        raise IsolationUnavailable(f"this host cannot confine candidate code: {rep}")
    return rep
