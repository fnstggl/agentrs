"""Jailed learner processes: evaluate a learner whose code is not trusted (MASTER_SPEC Parts 15 and 16; SP-15.8,
SP-16b "learners can be evaluated out of process").

The research instrument keeps worlds, verifiers, partitions and the sealed salt in an evaluator process
(`crucible.research.v5.evaluator`) and gives a learner only a *participant channel*. That separation means nothing
if the learner's code runs in the harness process that also holds the evaluator channel, or as the same user as
the evaluator: an adversarial learner then finds the evaluator handle through the garbage collector and reads its
own verified score mid-episode (a privileged progress signal) and its episode's partition, reads the evaluator
process's environment (``/proc/<pid>/environ`` names the sealed directory) and the sealed salt, and reads the
evaluator's memory (``/proc/<pid>/mem``; the harness is the evaluator's parent, which Yama ptrace scope 1 allows).
All four were measured on the integrated tree (SP-15.8 gate, control ``evaluator_internals``).

`LearnerHost` runs the learner in its own process under the native-tool jail (`crucible.sandbox.jail`, as for
FFmpeg and Chromium): its own unprivileged uid (not root, not a sandbox tenant's), ``no_new_privs``, Landlock
(read-only: system libraries, the Python standard library and packages, the crucible sources and the learner's
own code directories; read-write: one private working directory; no ``/proc``, no TCP), seccomp (no new sockets,
no new processes, no ptrace, signals only to itself) and rlimits. It inherits exactly one descriptor beyond its
pipes: the participant channel. The harness talks to it over a JSON-lines control pipe and never deserialises
anything else from it:

    host = LearnerHost("crucible.research.v5.learners:make", {"arch": "weights", "seed": 0},
                       participant=ev.part.sock)           # the evaluator's participant socket
    rec = host.episode(ticket, learn=True, seed=7)          # the learner's own episode record (diagnostics only)
    man = host.checkpoint()                                 # files copied out and hashed by the HOST
    out = fresh.weights_only_eval(man, worlds)              # scored in a fresh process from the parameters alone
    host.close()

What the learner reports (its episode record) is diagnostics; outcomes come from the evaluator channel. A
checkpoint is the two files the weights-only protocol loads (``arch.json``, ``params.npz``), opened without
following links, size-capped and hashed by the host; the learner's own manifest is never read.

Limits: process-level isolation on a shared kernel (no VM); the learner may use the CPU, memory and time its
rlimits allow and may write within its private directory (RLIMIT_FSIZE per file, a watchdog-enforced quota for the
directory); it reads the crucible sources (public code, no secrets). Learners that need a GPU or extra system paths are out of scope here.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import select
import shutil
import socket
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Any

from . import isolation

MAX_REPLY = 16 * 2**20           # a control reply larger than this is a protocol violation
MAX_CKPT_FILE = 256 * 2**20      # largest arch.json / params.npz the host copies out of the jail
NPROC = 512                      # tasks (threads) a learner process's own uid may hold
DISK_POLL_S = 0.25               # how often the host measures the learner's private directory

_BOOT = r"""
import json, os, sys
work, src, site, entry, kw, pfd, proto, code = sys.argv[1:9]
sys.path[:0] = [src, site] + json.loads(code)          # the learner's own code directories are importable too
os.chdir(work)
_proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)                                 # anything the learner prints goes to stderr, never into the protocol
sys.stdout = sys.stderr
import numpy as np


def _jd(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _reply(obj):
    _proto.write(json.dumps(obj, default=_jd) + "\n")
    _proto.flush()


def _target(spec):
    mod, _, fn = str(spec).rpartition(":")
    if not mod or not fn:
        raise ValueError("a target is named 'module:function', got %r" % (spec,))
    return getattr(__import__(mod, fromlist=["_"]), fn)


try:
    learner = _target(entry)(**json.loads(kw)) if entry else None
    chan = None
    if int(pfd) >= 0 and proto == "research-evaluator":
        import socket
        from crucible.research.v5.evaluator import Channel
        chan = Channel(socket.socket(fileno=int(pfd)))
    elif int(pfd) >= 0:                        # crucible.experience.remote: JSON frames on a multiprocessing pipe
        from multiprocessing.connection import Connection
        from crucible.experience.remote import PARTICIPANT_OPS, RemoteParticipant
        chan = RemoteParticipant(Connection(int(pfd)), PARTICIPANT_OPS)
    _reply({"ok": True, "ready": True, "pid": os.getpid(), "uid": os.getuid()})
except BaseException as e:
    _reply({"ok": False, "ready": False, "error": "%s: %s" % (type(e).__name__, str(e)[:500])})
    raise SystemExit(1)
n_ckpt = 0
for line in sys.stdin:
    try:
        req = json.loads(line)
        op = req.get("op")
        if op == "episode":
            from crucible.research.v5.evaluator import ParticipantEnv
            from crucible.research.v5.episodes import run_episode
            env = ParticipantEnv(chan, str(req["ticket"]))
            rec = run_episode(env, learner, learn=bool(req.get("learn")), rng=np.random.default_rng(int(req["seed"])),
                              budget=req.get("budget"), verify=False)
            out = {"ok": True, "record": rec}
        elif op == "checkpoint":
            from crucible.research.v5.fresh import save_checkpoint
            n_ckpt += 1
            d = os.path.join(work, "ckpt-%d" % n_ckpt)
            save_checkpoint(learner, d)
            out = {"ok": True, "dir": d}
        elif op == "function":                  # fn(participant, **kw): the experience.remote learner contract
            out = {"ok": True, "value": _target(req["target"])(chan, **dict(req.get("kw") or {}))}
        elif op == "call":
            name = str(req.get("method"))
            if name.startswith("_"):
                raise ValueError("private methods are not callable")
            out = {"ok": True, "value": getattr(learner, name)(*list(req.get("args") or []))}
        elif op == "close":
            _reply({"ok": True})
            break
        else:
            out = {"ok": False, "error": "unknown op %r" % (op,)}
    except BaseException as e:
        out = {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:500])}
    _reply(out)
"""


class LearnerHostError(RuntimeError):
    pass


def _site_packages() -> str:
    import numpy
    return os.path.dirname(os.path.dirname(os.path.abspath(numpy.__file__)))


def _src_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _traversable(path: str, uid: int) -> bool:
    """Whether `uid` (no supplementary groups, gid = uid) may search every directory down to and including `path`."""
    d = os.path.realpath(path)
    while True:
        try:
            st = os.stat(d)
        except OSError:
            return False
        if not (st.st_mode & 0o001 or (st.st_uid == uid and st.st_mode & 0o100)
                or (st.st_gid == uid and st.st_mode & 0o010)):
            return False
        if d == "/":
            return True
        d = os.path.dirname(d)


def _reachable_copy(src: str, uid: int) -> str:
    """`src` itself when the jail uid can reach it; otherwise a copy of the tree (hard links where the filesystem
    allows, no extra disk) under the jail base, keyed by the files' paths, sizes and modification times - a checkout
    below a root-only directory (an agent scratchpad, a fresh clone in a private temporary directory) is otherwise
    unreadable to the jail (the same DAC limit as DEFECTS DEF-P-16)."""
    if _traversable(src, uid):
        return src
    h = hashlib.sha256(os.path.realpath(src).encode())
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            st = os.stat(os.path.join(root, f))
            h.update(f"{os.path.relpath(os.path.join(root, f), src)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    dst = os.path.join(isolation._base(), f"src-{h.hexdigest()[:16]}")
    if os.path.exists(os.path.join(dst, ".complete")):
        return dst
    for old in os.listdir(isolation._base()):          # copies of earlier source states, a day old
        p = os.path.join(isolation._base(), old)
        try:
            if old.startswith("src-") and time.time() - os.stat(p).st_mtime > 86400:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass
    tmp = f"{dst}.tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)

    def link(s, d):
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
    shutil.copytree(src, tmp, copy_function=link, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for root, dirs, files in os.walk(tmp):
        os.chmod(root, 0o755)
        for f in files:
            p = os.path.join(root, f)
            if not os.stat(p).st_mode & 0o004:
                shutil.copy2(p, p + ".r")
                os.replace(p + ".r", p)
                os.chmod(p, 0o644)
    try:
        os.rename(tmp, dst)
    except OSError:                       # another process finished first
        shutil.rmtree(tmp, ignore_errors=True)
    open(os.path.join(dst, ".complete"), "w").close()
    return dst


_INSTANCES = [0]
UID_SPAN = 0x10000                # instance uids: isolation.UID_BASE + 0x10000 + [0, UID_SPAN)


def _uid_busy(uid: int) -> bool:
    """Whether any live process runs as `uid` (a learner process whose harness died keeps its uid busy)."""
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                if os.stat(f"/proc/{d}").st_uid == uid:
                    return True
            except OSError:
                continue
    return False


def _claim_dir() -> str:
    d = os.path.join(isolation._base(), "uid-claims")
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


def _claim_alive(path: str) -> bool:
    try:
        pid, start = open(path).read().split()[:2]
        with open(f"/proc/{int(pid)}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[19] == start
    except (OSError, ValueError, IndexError):
        return False


def _instance_uid(kind: str) -> int:
    """A uid of this learner process alone, claimed machine-wide: learner processes of one kind used to share a uid,
    so one learner held the thread budget (RLIMIT_NPROC counts per uid) of every other, and the kernel's same-uid
    checks (signals by descriptor ownership, System V IPC, prlimit, setpriority) let it reach them (wave-3b review).
    The first version hashed (kind, harness pid, instance number) into 16 bits, so concurrent learner processes
    shared a uid with probability ~n^2/2^17, and a forced collision let one set the other's RLIMIT_CPU to 1 s and
    kill it (wave-3b verifier). Now a candidate is taken only when no live claim names it and no live process runs
    as it; the claim (an O_EXCL file holding the harness pid and its start time) is released by `close`, and a
    claim whose harness is gone is reclaimed. The jail's seccomp filter, independently, restricts prlimit,
    scheduling and signals to the process itself."""
    _INSTANCES[0] += 1
    nonce = f"{kind}|{os.getpid()}|{_INSTANCES[0]}"
    start = int(hashlib.blake2b(nonce.encode(), digest_size=2).hexdigest(), 16)
    d = _claim_dir()
    with open(f"/proc/{os.getpid()}/stat") as fh:
        me = f"{os.getpid()} {fh.read().rsplit(')', 1)[1].split()[19]}"
    for i in range(UID_SPAN):
        uid = isolation.UID_BASE + 0x10000 + (start + i) % UID_SPAN
        path = os.path.join(d, str(uid))
        if os.path.exists(path):
            if _claim_alive(path):
                continue
            try:
                os.unlink(path)                    # its harness is gone
            except OSError:
                continue
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(me)
        if _uid_busy(uid):                         # a learner process left running by a harness that died
            os.unlink(path)
            continue
        return uid
    raise LearnerHostError("no free learner uid")


def _release_uid(uid: int | None) -> None:
    if uid is None:
        return
    path = os.path.join(_claim_dir(), str(int(uid)))
    try:
        if open(path).read().split()[0] == str(os.getpid()):
            os.unlink(path)
    except (OSError, IndexError):
        pass


class LearnerHost:
    """One learner in its own jailed process (module docstring). `entry` is ``"module:factory"`` (importable from the
    crucible sources or `code_dirs`), called as ``factory(**kw)`` inside the jail to build a lever learner (or None:
    only `function` is used). `participant` is the participant endpoint the jail inherits: the research evaluator's
    participant socket (``protocol="research-evaluator"``) or the participant pipe of `crucible.experience.remote.host`
    (``protocol="experience-remote"``, anything with a ``fileno()``)."""

    def __init__(self, entry: str | None, kw: dict | None = None, *, participant=None,
                 protocol: str = "research-evaluator", code_dirs: tuple[str, ...] = (), kind: str = "learner-host",
                 mem_mb: int = 3072, cpu_s: int = 3600, start_timeout: float = 120.0, op_timeout: float = 900.0,
                 disk_quota_mb: int = 2048, max_entries: int = 10000):
        ok, why = isolation.available()
        if not ok:
            raise isolation.JailUnavailable(why)
        if protocol not in ("research-evaluator", "experience-remote"):
            raise ValueError(f"unknown participant protocol {protocol!r}")
        self.entry, self.kw, self.protocol = str(entry or ""), dict(kw or {}), protocol
        self.op_timeout = float(op_timeout)
        self.uid = _instance_uid(kind)
        self.work = tempfile.mkdtemp(prefix="learner-", dir=isolation._base())
        self.out_dir = tempfile.mkdtemp(prefix="learner-out-")          # host side, root only
        py = os.path.realpath(sys.executable)
        paths = sysconfig.get_paths()
        site = _reachable_copy(_site_packages(), self.uid)
        src = _reachable_copy(_src_root(), self.uid)
        code = [_reachable_copy(os.path.realpath(d), self.uid) for d in code_dirs]
        ro = isolation.RO_SYSTEM + [paths["stdlib"], paths["platstdlib"], os.path.join(paths["stdlib"], "lib-dynload"),
                                    site, src, os.path.dirname(py), os.path.join(sys.base_prefix, "lib"),
                                    *code]
        self._cfg = isolation._write_cfg({"uid": self.uid, "gid": self.uid, "mode": "tool", "ro": sorted(set(ro)),
                                          "exec": [py] + isolation._loaders(), "rw": [self.work],
                                          "chown": [self.work],
                                          "env": {"HOME": self.work, "TMPDIR": self.work, "OMP_NUM_THREADS": "1",
                                                  "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                                  "PYTHONHASHSEED": "0"}})
        pfd = -1
        fds: tuple = ()
        if participant is not None:
            pfd = participant.fileno()
            os.set_inheritable(pfd, True)
            fds = (pfd,)
        cmd = [sys.executable, "-I", "-S", "-B", isolation.JAIL, self._cfg, py, "-I", "-S", "-B", "-c", _BOOT,
               self.work, src, site, self.entry, json.dumps(self.kw), str(pfd), self.protocol, json.dumps(code)]
        lim = (int(mem_mb) * 2**20, int(cpu_s))
        self._err = open(os.path.join(self.out_dir, "stderr.log"), "wb")

        def pre():
            os.setsid()
            resource.setrlimit(resource.RLIMIT_AS, (lim[0], lim[0]))
            resource.setrlimit(resource.RLIMIT_CPU, (lim[1], lim[1] + 5))
            resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_CKPT_FILE, MAX_CKPT_FILE))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_NPROC, (NPROC, NPROC))   # threads of the jail uid (no new processes)
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
                                     pass_fds=fds, preexec_fn=pre, cwd=self.work,
                                     env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
        self._buf = b""
        hello = self._read(start_timeout)
        if not hello.get("ready"):
            err = hello.get("error") or self.stderr_tail()
            self.close()
            raise LearnerHostError(f"the learner process did not start: {err}")
        self.pid, self.jail_uid = int(hello["pid"]), int(hello["uid"])
        # The private directory has no per-directory rlimit (RLIMIT_FSIZE bounds one file): a watchdog measures it
        # while the process lives - also between operations, when a learner's own threads may keep writing - and
        # stops the learner beyond its quota (wave-3b review: a learner could fill the machine's disk).
        self.disk_quota = int(disk_quota_mb) * 2**20
        self.max_entries = int(max_entries)
        self.dead_reason: str | None = None
        import threading
        self._watch = threading.Thread(target=self._watchdog, name="learner-disk-watchdog", daemon=True)
        self._watch.start()

    def disk_held(self) -> tuple[int, int, list[str]]:
        """(bytes, entries, unmeasured) the learner holds on disk: its private directory's listing, the files it
        holds open after unlinking them (metering.held_bytes - the listing alone missed 8x the quota held this way;
        wave-3b verifier) and its stderr log (written by the learner, outside its directory)."""
        from . import metering as M
        b, n = M.dir_usage(self.work, self.max_entries)
        pid = getattr(self, "pid", None)
        hidden, unmeasured, _ok = M.held_bytes(pid, self.jail_uid, self.jail_uid, self.work) if pid else (0, [], False)
        try:
            log = os.path.getsize(os.path.join(self.out_dir, "stderr.log"))
        except OSError:
            log = 0
        return b + hidden + log, n, unmeasured

    def _watchdog(self) -> None:
        while self.proc.poll() is None:
            b, n, unmeasured = self.disk_held()
            if b > self.disk_quota or n > self.max_entries or unmeasured:
                self.dead_reason = (f"the learner exceeded its disk quota ({b} bytes incl. deleted files held open "
                                    f"and its log, {n} entries; quota {self.disk_quota} bytes, {self.max_entries} "
                                    "entries)" if not unmeasured else
                                    "the learner keeps a deleted file of its directory alive by a memory mapping "
                                    "alone (its size cannot be measured)")
                self.kill()
                shutil.rmtree(self.work, ignore_errors=True)
                return
            time.sleep(DISK_POLL_S)

    # ------------------------------------------------------------------ protocol
    def _read(self, timeout: float) -> dict:
        end = time.monotonic() + timeout
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buf:
            left = end - time.monotonic()
            if left <= 0:
                self.kill()
                raise LearnerHostError(f"the learner process did not answer within {timeout:.0f}s")
            r, _, _ = select.select([fd], [], [], left)
            if not r:
                continue
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                why = getattr(self, "dead_reason", None)
                raise LearnerHostError(f"the learner process exited: {why or self.stderr_tail()[-600:]}")
            self._buf += chunk
            if len(self._buf) > MAX_REPLY:
                self.kill()
                raise LearnerHostError("the learner process sent an oversized reply")
        line, _, self._buf = self._buf.partition(b"\n")
        try:
            m = json.loads(line.decode("utf-8"), parse_constant=lambda c: None)
        except (ValueError, UnicodeDecodeError, RecursionError):
            self.kill()
            raise LearnerHostError("the learner process broke the control protocol")
        if not isinstance(m, dict):
            self.kill()
            raise LearnerHostError("the learner process broke the control protocol")
        return m

    def _call(self, req: dict, timeout: float | None = None) -> dict:
        if self.proc.poll() is not None:
            why = getattr(self, "dead_reason", None)
            raise LearnerHostError(f"the learner process is not running: {why or self.stderr_tail()[-600:]}")
        self.proc.stdin.write((json.dumps(req) + "\n").encode())
        self.proc.stdin.flush()
        return self._read(timeout or self.op_timeout)

    # ------------------------------------------------------------------ operations
    def episode(self, ticket: str, *, learn: bool, seed: int, budget: int | None = None) -> dict:
        """Run one episode of the evaluator-held `ticket` in the learner process (`episodes.run_episode` over the
        participant channel, ``verify=False``). Returns the learner-side record (diagnostics; the verified outcome
        is read from the evaluator channel)."""
        r = self._call({"op": "episode", "ticket": str(ticket), "learn": bool(learn), "seed": int(seed),
                        "budget": None if budget is None else int(budget)})
        if not r.get("ok"):
            raise LearnerHostError(f"episode failed in the learner process: {r.get('error')}")
        return r.get("record") or {}

    def function(self, target: str, kw: dict | None = None, timeout: float | None = None) -> Any:
        """Call ``target(participant, **kw)`` inside the learner process (``"module:function"``): the learner contract
        of `crucible.experience.remote.evaluate`, with a `RemoteParticipant` over the inherited participant pipe
        (protocol ``experience-remote``). Returns the function's JSON result."""
        r = self._call({"op": "function", "target": str(target), "kw": dict(kw or {})}, timeout=timeout)
        if not r.get("ok"):
            raise LearnerHostError(f"{target} failed in the learner process: {r.get('error')}")
        return r.get("value")

    def call(self, method: str, *args) -> Any:
        """Call a public method of the learner inside its process (JSON arguments and result)."""
        r = self._call({"op": "call", "method": str(method), "args": list(args)})
        if not r.get("ok"):
            raise LearnerHostError(f"{method} failed in the learner process: {r.get('error')}")
        return r.get("value")

    def checkpoint(self) -> dict:
        """Have the learner write its checkpoint (`fresh.save_checkpoint`) inside its private directory, copy the two
        files weights-only evaluation loads out of the jail (no links followed, size-capped) and compute the manifest
        here: the SHA-256 of arch.json and the parameter hash of params.npz loaded without pickle."""
        r = self._call({"op": "checkpoint"})
        if not r.get("ok"):
            raise LearnerHostError(f"checkpoint failed in the learner process: {r.get('error')}")
        src = os.path.realpath(os.path.join(self.work, os.path.basename(str(r.get("dir", "")))))
        if os.path.dirname(src) != os.path.realpath(self.work):
            raise LearnerHostError("the learner named a checkpoint outside its directory")
        dst = tempfile.mkdtemp(prefix="ckpt-", dir=self.out_dir)
        for name in ("arch.json", "params.npz"):
            try:
                fd = os.open(os.path.join(src, name), os.O_RDONLY | os.O_NOFOLLOW)
            except OSError as e:
                raise LearnerHostError(f"checkpoint file {name}: {e.strerror}")
            try:
                st = os.fstat(fd)
                if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_CKPT_FILE:
                    raise LearnerHostError(f"checkpoint file {name} is not a regular file within the size limit")
                with os.fdopen(os.dup(fd), "rb") as fh, open(os.path.join(dst, name), "wb") as out:
                    shutil.copyfileobj(fh, out, 1 << 20)
            finally:
                os.close(fd)
        import numpy as np
        from ..research.v5.learners import params_hash
        arch_bytes = open(os.path.join(dst, "arch.json"), "rb").read()
        try:
            arch = json.loads(arch_bytes.decode("utf-8"))
            with np.load(os.path.join(dst, "params.npz"), allow_pickle=False) as z:
                params = {k: np.array(z[k]) for k in z.files}
        except (ValueError, OSError, UnicodeDecodeError) as e:
            raise LearnerHostError(f"the checkpoint cannot be read safely: {type(e).__name__}")
        shutil.rmtree(src, ignore_errors=True)        # copied out: it no longer counts against the learner's quota
        return {"dir": dst, "arch_sha256": hashlib.sha256(arch_bytes).hexdigest(), "params_hash": params_hash(params),
                "arch": arch.get("arch") if isinstance(arch, dict) else None,
                "n_params": int(sum(a.size for a in params.values())), "hashed_by": "host"}

    def stderr_tail(self) -> str:
        try:
            self._err.flush()
            with open(os.path.join(self.out_dir, "stderr.log"), "rb") as fh:
                fh.seek(max(0, os.path.getsize(fh.name) - 4000))
                return fh.read().decode(errors="replace").replace(self.work, "<learner>")
        except OSError:
            return ""

    def kill(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 9)
            except OSError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        if getattr(self, "proc", None) is not None and self.proc.poll() is None:
            try:
                self._call({"op": "close"}, timeout=10)
            except (LearnerHostError, OSError, ValueError):
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.kill()
        try:
            self._err.close()
        except Exception:  # noqa: BLE001
            pass
        _release_uid(getattr(self, "uid", None))
        self.uid = None
        for p in (getattr(self, "_cfg", None),):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        shutil.rmtree(getattr(self, "work", ""), ignore_errors=True)
        shutil.rmtree(getattr(self, "out_dir", ""), ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
