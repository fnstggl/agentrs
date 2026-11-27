"""Cross-world contamination probes (Gate W.5): two or more worlds in one process share the artifact byte pool, the
sandbox's measurement memo and the host's uids; nothing may pass between them, and a world's memory must be
released when it closes.

The artifact pool (`crucible.substrate.artifacts._POOL`) belongs to workstream A, which is adding eviction; the
eviction probe exercises whatever API exists (`World.close`, an `evict`/`release` function if present) and reports
honestly when bytes of closed worlds stay in the process."""
from __future__ import annotations

import base64
import gc
import json
import time

from ..framework import HELD, INTEGRITY_BREAK, LEAK, Attacker, Evaluator, NotApplicable, Operator, Verdict
from .base import V5Probe, b64, repo, run


def _second_world(kind: str, seed: int, world_id: str):
    from ...core.kernel import World
    from .worlds import v5_spec
    return World.create(v5_spec(kind, seed), world_id=world_id)


class ArtifactsDoNotCrossWorlds(V5Probe):
    name = "v5.crossworld.artifact_bytes_do_not_cross_worlds"
    attack_class = "cross_world/artifact_pool"
    gates = ("W.5", "W.9")
    target, owner = "artifacts", "A"
    world = {"v5": "substrate"}
    description = ("A victim in another world of the same process (memory stores: one process-wide byte pool) stores "
                   "private bytes. The attacker in this world tries to read them by REF, asks for their info, and "
                   "stores the same bytes and fresh bytes to look for any difference (an existence oracle through "
                   "the shared pool). Checked: nothing distinguishes bytes stored in the other world from bytes "
                   "stored nowhere, and none are readable.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("xw-attacker", 1000)
        self.other = _second_world("substrate", 12, "xw-victim-world")
        va = self.other.system_call("admin.join", {"handle": "victim", "endowment": 1000})["principal"]
        self.secret = b"world V private: launch code 4417-2291-XQ"
        self.ref = self.other.system_call("artifact.put", {"data_b64": b64(self.secret), "mime": "text/plain"},
                                          as_principal=va)["ref"]

    def attack(self, a: Attacker) -> None:
        missing = "sha256:" + "cd" * 32
        self.reads = {}
        for tool in ("artifact.get", "artifact.info"):
            rs = [a.act(self.att, tool, ref=r) for r in (self.ref, missing)]
            self.reads[tool] = [{"ok": r.ok, "code": (r.error or {}).get("code"),
                                 "msg": str((r.error or {}).get("message", "")).replace(self.ref, "<R>")
                                 .replace(missing, "<R>")[:120]} for r in rs]
        r1 = a.act(self.att, "artifact.put", data_b64=b64(self.secret), mime="text/plain")
        r2 = a.act(self.att, "artifact.put", data_b64=b64(self.secret.replace(b"4417-2291-XQ", b"0000-0000-ZZ")),
                   mime="text/plain")         # same size, stored nowhere
        strip = lambda r: {k: v for k, v in (r.result or {}).items() if k != "ref"} if r.ok else r.error  # noqa: E731
        self.put_existing, self.put_fresh = strip(r1), strip(r2)

    def verdict(self, ev: Evaluator) -> Verdict:
        try:
            got = [x for x in self.reads["artifact.get"][:1] if x["ok"]]
            same = self.reads["artifact.get"][0] == self.reads["artifact.get"][1] and \
                self.reads["artifact.info"][0] == self.reads["artifact.info"][1]
            differs = json.dumps(self.put_existing, sort_keys=True, default=str) != \
                json.dumps(self.put_fresh, sort_keys=True, default=str)
            evd = {"reads": self.reads, "put_existing_elsewhere": self.put_existing, "put_fresh": self.put_fresh,
                   "cross_world_leaks": int(bool(got)) + int(not same) + int(differs)}
        finally:
            self.other.close()
        if got:
            return Verdict(LEAK, "the attacker read another world's private bytes by REF", evd, severity="critical")
        if not same or differs:
            return Verdict(LEAK, "responses differ for bytes held by another world (existence oracle)", evd,
                           severity="medium")
        return Verdict(HELD, "another world's bytes look exactly like bytes stored nowhere", evd)


class ClosedWorldBytesAreReleased(V5Probe):
    name = "v5.crossworld.closed_worlds_release_their_bytes"
    attack_class = "unbounded_resources/memory_of_closed_worlds"
    gates = ("W.7",)
    target, owner = "artifacts", "A"
    world = {"v5": "substrate"}
    description = ("Long multi-world processes (experiments, leagues, curricula) create and close many store-less "
                   "worlds. Each of 12 short-lived worlds stores 256 KiB of distinct bytes and is closed. Checked: "
                   "the process-wide artifact pool does not keep the bytes of closed worlds (audit O 3.12: a "
                   "host-RAM exhaustion path with a 512 MiB per-owner quota).")

    def setup(self, op: Operator) -> None:
        self.att = op.join("xw-filler", 1000)

    def attack(self, a: Attacker) -> None:
        a.note("operator-level lifecycle probe: worlds are created and closed by the experiment harness")
        from ...substrate import artifacts as art
        self.refs = []
        for i in range(12):
            w = _second_world("substrate", 100 + i, f"xw-short-{i}")
            p = w.system_call("admin.join", {"handle": f"u{i}", "endowment": 10})["principal"]
            data = (f"world {i} payload ".encode() * 20000)[:256 * 1024]
            self.refs.append(w.system_call("artifact.put", {"data_b64": base64.b64encode(data).decode(),
                                                            "mime": "application/octet-stream"}, as_principal=p)["ref"])
            w.close()
            del w
        gc.collect()
        for name in ("evict_closed", "evict", "release_world", "gc_pool", "collect"):
            fn = getattr(art, name, None)
            if callable(fn):
                try:
                    fn()
                except TypeError:
                    pass
        self.kept = [r for r in self.refs if art.parse_ref(r) in getattr(art, "_POOL", {})]
        self.api = [n for n in dir(art) if any(k in n.lower() for k in ("evict", "release", "retention", "compact"))]

    def verdict(self, ev: Evaluator) -> Verdict:
        evd = {"worlds": len(self.refs), "bytes_each": 256 * 1024, "still_in_pool": len(self.kept),
               "eviction_api_found": self.api, "cross_world_leaks": 0, "resource_limit_escapes": int(bool(self.kept))}
        if self.kept:
            return Verdict(INTEGRITY_BREAK, f"{len(self.kept)} of {len(self.refs)} closed worlds' bytes stay in the "
                           "process-wide artifact pool", evd, severity="medium",
                           fix="substrate/artifacts.py: evict a store-less world's bytes from _POOL when it closes "
                               "(reference counts per world), workstream A")
        return Verdict(HELD, "closed worlds' bytes were released from the process pool", evd)


class SandboxStateIsPerWorld(V5Probe):
    name = "v5.crossworld.sandbox_pools_and_meter_memo_are_per_world"
    attack_class = "cross_world/sandbox_state"
    gates = ("W.5",)
    target, owner = "sandbox", "O"
    needs_sandbox = True
    world = {"v5": "commerce"}
    description = ("A service program keeps a module-level counter (state of its long-lived worker) and the "
                   "deterministic cost model remembers measurements per computation. An identical program runs in a "
                   "second world of the same process. Checked: the second world's program does not see the first's "
                   "worker state, the measurement memo is keyed by world lineage, and the worker pools are separate "
                   "objects.")

    SRC = "N = [0]\ndef tick(a, c):\n    N[0] += 1\n    return N[0]\n"

    def setup(self, op: Operator) -> None:
        self.att = op.join("xw-state", 2_000_000)
        self.other = _second_world("commerce", 9, "xw-state-other")
        p = self.other.system_call("admin.join", {"handle": "other", "endowment": 2_000_000})["principal"]
        rid = self.other.system_call("code.create_repo", {"name": "t"}, as_principal=p)["repo"]
        self.other.system_call("code.commit", {"repo": rid, "files": {"t.py": self.SRC}}, as_principal=p)
        self.other_runs = [self.other.system_call("code.run", {"repo": rid, "target": "t.py:tick"},
                                                  as_principal=p).get("result") for _ in range(3)]

    def attack(self, a: Attacker) -> None:
        rid = repo(a, self.att, "t", {"t.py": self.SRC})
        self.mine = [run(a, self.att, rid, "t.py:tick").get("result") for _ in range(2)]

    def verdict(self, ev: Evaluator) -> Verdict:
        pa, pb = ev.world.runtime.get("sandbox_pool"), self.other.runtime.get("sandbox_pool")
        scopes = (getattr(pa, "cfg", None) and pa.cfg.meter_scope, getattr(pb, "cfg", None) and pb.cfg.meter_scope)
        tenants = (getattr(pa, "cfg", None) and pa.cfg.tenant, getattr(pb, "cfg", None) and pb.cfg.tenant)
        self.other.close()
        evd = {"this_world_results": self.mine, "other_world_results": self.other_runs, "meter_scopes": scopes,
               "tenants": tenants, "same_pool_object": pa is pb}
        bad = []
        if self.mine[:1] != [1]:
            bad.append("this world's first run saw another world's worker state")
        if pa is pb or scopes[0] == scopes[1] or tenants[0] == tenants[1]:
            bad.append("worlds share a sandbox pool, a meter scope or a tenant")
        evd["cross_world_leaks"] = len(bad)
        if bad:
            return Verdict(LEAK, "; ".join(bad), evd, severity="high")
        return Verdict(HELD, "separate pools, meter scopes and tenants; no worker state crossed", evd)


class ReplayIgnoresForeignStore(V5Probe):
    name = "v5.crossworld.replay_is_independent_of_other_worlds"
    attack_class = "cross_world/replay"
    gates = ("W.5",)
    target, owner = "kernel", "A"
    world = {"v5": "substrate"}
    description = ("The attacker's world is replayed from its event log while a second world with different content "
                   "lives in the same process and holds its own artifacts. Checked: the replay reproduces exactly "
                   "the attacker world's digest and hash, and the second world's bytes never appear in it.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("xw-replay", 1000)
        self.other = _second_world("substrate", 31, "xw-replay-other")
        q = self.other.system_call("admin.join", {"handle": "qq", "endowment": 10})["principal"]
        self.other_ref = self.other.system_call("artifact.put", {"data_b64": b64(b"other world only 55"),
                                                                 "mime": "text/plain"}, as_principal=q)["ref"]

    def attack(self, a: Attacker) -> None:
        for i in range(3):
            a.act(self.att, "artifact.put", data_b64=b64(f"mine {i}".encode()), mime="text/plain")

    def verdict(self, ev: Evaluator) -> Verdict:
        from ...core.kernel import World
        w = ev.world
        r = World.replay(w.spec, [e.to_dict() for e in w.log], verify=True)
        exact = r.digest() == w.digest() and r.last_hash == w.last_hash
        mention = any(self.other_ref in json.dumps(e.payload, default=str) for e in r.log)
        r.close()
        self.other.close()
        evd = {"replay_exact": exact, "other_world_ref_in_replay": mention,
               "cross_world_leaks": int(not exact) + int(mention)}
        if not exact or mention:
            return Verdict(INTEGRITY_BREAK, "replay depends on another world in the process", evd, severity="high")
        return Verdict(HELD, "replay is exact and contains nothing of the other world", evd)


class MemoEvictionIsPerLineage(V5Probe):
    name = "v5.crossworld.meter_memo_eviction_is_per_lineage"
    attack_class = "cross_world/sandbox_state"
    gates = ("W.5",)
    target, owner = "sandbox", "O"
    world = {"v5": "substrate"}
    description = ("The deterministic cost model's anchors live in one machine-level memo file shared by every world "
                   "and process (metering.DiskMemo), so that a replication in a new process bills exactly. The file is "
                   "bounded. A victim world's program runs a few computations; an attacker world then runs far more "
                   "distinct computations than the file keeps (in this probe the real class with its bounds scaled "
                   "down 1000x: 2,000 rows, 200 per lineage; in a world: many distinct cheap jobs). Checked: every "
                   "victim anchor survives, so the victim's replication still bills its first measurements.")

    def setup(self, op: Operator) -> None:
        import tempfile
        self.dir = tempfile.mkdtemp(prefix="crucible-rt-memo-")

    def attack(self, a: Attacker) -> None:
        import os
        import sqlite3
        from ...sandbox import metering as M
        path = os.path.join(self.dir, "memo.sqlite")
        victim = M.DiskMemo(path, max_rows=2_000, scope_max=200, check_every=100)
        attacker = M.DiskMemo(path, max_rows=2_000, scope_max=200, check_every=100)   # another process, same file
        self.vkeys = [M.computation_key("victim-lineage", "f" * 8, {"kind": "call", "args": i}, [], {"ok": True})
                      for i in range(50)]
        for k in self.vkeys:
            victim.first(k, 7_000_000, "victim-lineage")
        for i in range(20_000):
            attacker.first(M.computation_key("attacker-lineage", "a" * 8, {"kind": "call", "args": i}, [],
                                             {"ok": True}), 1, "attacker-lineage")
        check = M.DiskMemo(path, max_rows=2_000, scope_max=200)
        self.kept = sum(1 for k in self.vkeys if check.first(k, 1, "victim-lineage") == (7_000_000, False))
        con = sqlite3.connect(path)
        self.rows = con.execute("SELECT scope, count(*) FROM memo GROUP BY scope").fetchall()
        con.close()

    def verdict(self, ev: Evaluator) -> Verdict:
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)
        evd = {"victim_anchors": len(self.vkeys), "victim_anchors_kept": self.kept, "rows_by_lineage": self.rows,
               "cross_world_leaks": len(self.vkeys) - self.kept}
        if self.kept < len(self.vkeys):
            return Verdict(INTEGRITY_BREAK, f"another lineage's computations evicted {len(self.vkeys) - self.kept} of "
                           f"{len(self.vkeys)} victim anchors from the shared memo file", evd, severity="low",
                           fix="sandbox/metering.py: DiskMemo evicts per lineage")
        return Verdict(HELD, "every victim anchor survived another lineage's flood; the flood evicted only its own "
                             "rows", evd)


def _ipc_key(tag: str) -> int:
    """A System V key unique to this run of a probe (process id + time), so concurrent runs never meet."""
    import os
    import time
    import zlib
    return 0x5E000000 | (zlib.crc32(f"{tag}|{os.getpid()}|{time.monotonic_ns()}".encode()) & 0x00FFFFF0)


_IPC_RM = r"""
import ctypes, json, sys
libc = ctypes.CDLL(None, use_errno=True)
key, out = int(sys.argv[1]), {}
for name, get, rm in (("shm", lambda: libc.shmget(key, 0, 0), lambda i: libc.shmctl(i, 0, None)),
                      ("msg", lambda: libc.msgget(key + 1, 0), lambda i: libc.msgctl(i, 0, None)),
                      ("sem", lambda: libc.semget(key + 2, 0, 0), lambda i: libc.semctl(i, 0, 0))):
    i = get()
    if i >= 0:
        out[name] = rm(i) == 0
try:
    out["mq"] = libc.mq_unlink(("/crucible-rt-%x" % key).encode()) == 0
except AttributeError:
    pass
print(json.dumps(out))
"""


def _ipc_cleanup(key: int, uids) -> dict:
    """Remove whatever kernel IPC objects a run under `key` left behind; returns what was removed. Root in this
    container has no CAP_IPC_OWNER/CAP_SYS_ADMIN, so the removal runs as each uid that may own an object."""
    import subprocess
    import sys
    gone = {}
    for uid in sorted({int(u) for u in uids if isinstance(u, int) and u > 0}):
        try:
            r = subprocess.run([sys.executable, "-I", "-S", "-c", _IPC_RM, str(int(key))], user=uid, group=uid,
                               extra_groups=[], cwd="/", capture_output=True, text=True, timeout=60)
            gone.update({k: v for k, v in json.loads(r.stdout or "{}").items() if v})
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return gone


_IPC_ENTRY = ("\n\ndef stash(a, c):\n    return ipc_stash(None, a['key'], a['canary'], 0o666)\n"
              "\n\ndef recall(a, c):\n    return ipc_recall(None, a['key'])\n")


def _ipc_files() -> dict:
    """The attack program: the red team's IPC functions (`adversaries.ipc_stash` / `ipc_recall`, plain ctypes) copied
    into the agent's own program file (a sandboxed program carries its own code; it cannot import crucible)."""
    import inspect
    from . import adversaries as ADV
    lib = "import ctypes\nimport os\n\n\n" + "\n\n".join(inspect.getsource(f) for f in
                                                           (ADV._ipc_libc, ADV.ipc_stash, ADV.ipc_recall))
    return {"ipc.py": lib + _IPC_ENTRY}


class KernelIpcDoesNotCrossWorlds(V5Probe):
    name = "v5.crossworld.kernel_ipc_does_not_cross_worlds"
    attack_class = "cross_world/kernel_ipc"
    gates = ("W.5",)
    target, owner = "sandbox", "O"
    needs_sandbox = True
    world = {"v5": "commerce"}
    description = ("System V shared memory, message queues and semaphores (and POSIX message queues) are kernel "
                   "objects named by a key, not by a path: Landlock does not mediate them on this kernel and they "
                   "outlive the process that made them. An agent's program in a second world (another tenant uid) "
                   "creates them world-readable under a key both agents' code knows and writes a canary; then this "
                   "world's program, in its own sandbox, looks the key up and reads. Checked: the writer cannot "
                   "create them and the reader receives nothing - no channel between worlds, and no state outside "
                   "the world that a later run (a replay, a regenerated world) could read.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("xw-ipc", 2_000_000)
        self.key = _ipc_key(self.name)
        self.canary = f"xw-ipc-canary-{self.key:x}"
        self.other = _second_world("commerce", 11, "xw-ipc-other")
        p = self.other.system_call("admin.join", {"handle": "other", "endowment": 2_000_000})["principal"]
        rid = self.other.system_call("code.create_repo", {"name": "ipc"}, as_principal=p)["repo"]
        self.other.system_call("code.commit", {"repo": rid, "files": _ipc_files()}, as_principal=p)
        r = self.other.system_call("code.run", {"repo": rid, "target": "ipc.py:stash",
                                                "input": {"key": self.key, "canary": self.canary}}, as_principal=p)
        self.writer = r.get("result") if isinstance(r.get("result"), dict) else {"error": str(r)[:400]}

    def attack(self, a: Attacker) -> None:
        rid = repo(a, self.att, "ipc", _ipc_files())
        self.reader = run(a, self.att, rid, "ipc.py:recall", input={"key": self.key})

    def verdict(self, ev: Evaluator) -> Verdict:
        reader = self.reader.get("result") if isinstance(self.reader.get("result"), dict) else self.reader
        try:
            self.other.close()
        finally:
            removed = _ipc_cleanup(self.key, [(self.writer or {}).get("uid"), (reader or {}).get("uid")])
        read = dict((reader or {}).get("read") or {})
        evd = {"writer": self.writer, "reader": reader, "left_in_kernel": removed}
        if not isinstance(self.writer, dict) or "shm" not in self.writer or not isinstance(reader, dict) \
                or "shm" not in reader:
            raise NotApplicable(f"the IPC programs did not run: {json.dumps(evd, default=str)[:400]}")
        crossed = sorted(k for k, v in read.items() if self.canary in str(v) or (k == "sem_value" and
                                                                                 v == len(self.canary)))
        created = sorted(k for k in ("shm", "msg", "sem", "mq") if self.writer.get(k) == "created")
        evd.update({"crossed": crossed, "created_by_writer": created, "cross_world_leaks": len(crossed)})
        if crossed:
            return Verdict(LEAK, f"another world's program passed the canary through kernel IPC: {crossed}", evd,
                           severity="high", fix="sandbox/boot.py seccomp: deny SysV IPC and POSIX mq syscalls")
        if created or removed:
            return Verdict(INTEGRITY_BREAK, f"a sandboxed program left kernel IPC objects outside the world: "
                                            f"{created or sorted(removed)}", evd, severity="medium",
                           fix="sandbox/boot.py seccomp: deny SysV IPC and POSIX mq syscalls")
        return Verdict(HELD, "no SysV or POSIX IPC object could be created, and nothing crossed between the worlds",
                       evd)


_LOCK_FILES = ("abc.py", "ast.py", "base64.py", "bisect.py", "calendar.py", "codecs.py", "copy.py", "csv.py",
               "datetime.py", "enum.py", "fnmatch.py", "functools.py", "gettext.py", "glob.py", "heapq.py", "hmac.py")
_LOCK_LIB = ("import fcntl, os, struct\nFILES = " + repr(_LOCK_FILES) + "\nHELD = []\n"
             "def _path(n):\n    return os.path.join(os.path.dirname(os.__file__), n)\n"
             "def _try(fn):\n    try:\n        fn()\n        return 'ok'\n"
             "    except OSError as e:\n        return 'errno %d' % e.errno\n")
# three encodings of the same secret, one per lock kind: flock() on read-only standard-library files, POSIX write
# locks on byte ranges of /dev/null (writable for every tenant), POSIX read locks on read-only files (seen by F_GETLK)
LOCK_HOLDER = _LOCK_LIB + (
    "def take(args, caller):\n    out = {'flock': {}, 'devnull': {}, 'rdlock': {}}\n"
    "    null = open('/dev/null', 'r+b')\n    HELD.append(null)\n"
    "    for i, n in enumerate(FILES):\n        if not int(args['bits']) >> i & 1:\n            continue\n"
    "        f = open(_path(n), 'rb')\n        g = open(_path(n), 'rb')\n        HELD.extend((f, g))\n"
    "        out['flock'][n] = _try(lambda: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB))\n"
    "        out['devnull'][n] = _try(lambda: fcntl.lockf(null, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, i))\n"
    "        out['rdlock'][n] = _try(lambda: fcntl.fcntl(g, fcntl.F_SETLK, struct.pack('hhqqi', fcntl.F_RDLCK, 0, 0, 0, 0)))\n"
    "    return out\n")
LOCK_READER = _LOCK_LIB + (
    "def read(args, caller):\n    bits, out = {'flock': 0, 'devnull': 0, 'rdlock': 0}, {}\n"
    "    null = open('/dev/null', 'r+b')\n"
    "    for i, n in enumerate(FILES):\n        f = open(_path(n), 'rb')\n        r = {}\n"
    "        r['flock'] = _try(lambda: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB))\n"
    "        r['devnull'] = _try(lambda: fcntl.lockf(null, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, i))\n"
    "        try:\n            q = fcntl.fcntl(f, fcntl.F_GETLK, struct.pack('hhqqi', fcntl.F_WRLCK, 0, 0, 0, 0))\n"
    "            t, _w, _s, _l, pid = struct.unpack('hhqqi', q)\n"
    "            r['rdlock'] = 'busy pid %d' % pid if t != fcntl.F_UNLCK else 'ok'\n"
    "        except OSError as e:\n            r['rdlock'] = 'errno %d' % e.errno\n"
    "        for k, v in r.items():\n            if v.startswith('errno 11') or v.startswith('errno 13') or v.startswith('busy'):\n"
    "                bits[k] |= 1 << i\n"
    "        try:\n            fcntl.flock(f, fcntl.LOCK_UN)\n            fcntl.lockf(null, fcntl.LOCK_UN, 1, i)\n"
    "        except OSError:\n            pass\n"
    "        f.close()\n        out[n] = r\n"
    "    return {'bits': bits, 'files': out}\n")


class FileLocksDoNotCrossWorlds(V5Probe):
    name = "v5.crossworld.file_locks_do_not_cross_worlds"
    attack_class = "cross_world/file_locks"
    gates = ("W.5",)
    target, owner = "sandbox", "O"
    needs_sandbox = True
    world = {"v5": "commerce"}
    description = ("Every world's sandboxed programs may read the same system files (the Python standard library), "
                   "and flock() works on a file opened read-only. A service deployed in this world (a long-lived "
                   "worker) takes exclusive locks on a pattern of 16 standard-library files - 16 bits of a secret - "
                   "and returns; a program in a second world (another tenant) tries a non-blocking lock on each and "
                   "reads the bits from which ones are busy; the same secret also as POSIX write locks on byte "
                   "ranges of /dev/null (writable for every tenant) and as POSIX read locks that F_GETLK would "
                   "report (with the holder's pid). Checked: no bit crosses.")

    SECRET = 0b1011001110001011

    def setup(self, op: Operator) -> None:
        self.att = op.join("xw-locks", 20_000_000)
        self.other = _second_world("commerce", 13, "xw-locks-other")

    def attack(self, a: Attacker) -> None:
        from .base import deploy, manifest, org_of
        org, acct = org_of(a, self.att, "lockco", company=True)
        deploy(a, self.att, org, acct, "locker", {"crucible.json": manifest(take=0), "service.py": LOCK_HOLDER})
        r = a.act(self.att, "locker.take", bits=self.SECRET)
        self.holder = r.result if r.ok else {"error": r.error}
        o = self.other
        p = o.system_call("admin.join", {"handle": "reader", "endowment": 2_000_000})["principal"]
        rid = o.system_call("code.create_repo", {"name": "r"}, as_principal=p)["repo"]
        o.system_call("code.commit", {"repo": rid, "files": {"r.py": LOCK_READER}}, as_principal=p)
        rr = o.system_call("code.run", {"repo": rid, "target": "r.py:read"}, as_principal=p)
        self.reader = rr.get("result") if isinstance(rr.get("result"), dict) else {"error": str(rr)[:400]}

    def verdict(self, ev: Evaluator) -> Verdict:
        self.other.close()
        got = (self.reader or {}).get("bits")
        body = (self.holder or {}).get("body", self.holder) if isinstance(self.holder, dict) else None
        evd = {"holder": body, "reader": self.reader, "secret": self.SECRET}
        if not isinstance(got, dict) or not isinstance(body, dict) or "flock" not in body:
            raise NotApplicable(f"the lock programs did not run: {json.dumps(evd, default=str)[:400]}")
        n = bin(self.SECRET).count("1")
        crossed = {k: bin(int(v) & self.SECRET).count("1") for k, v in got.items()}
        false_bits = {k: bin(int(v) & ~self.SECRET).count("1") for k, v in got.items()}
        evd.update({"bits_crossed": crossed, "bits_set_but_not_secret": false_bits,
                    "cross_world_leaks": sum(1 for v in crossed.values() if v)})
        if any(crossed.values()):
            return Verdict(LEAK, f"secret bits (of {n}) crossed to another world through advisory locks on shared "
                                 f"files: {crossed}", evd, severity="high",
                           fix="sandbox/boot.py seccomp: locks private to the program (no-op), lock queries refused")
        return Verdict(HELD, f"no bit crossed through flock, /dev/null record locks or lock queries ({n} set)", evd)

_WATCH_FILES = ("aifc.py", "antigravity.py", "cgi.py", "cgitb.py", "chunk.py", "crypt.py", "imghdr.py", "mailcap.py",
                "nntplib.py", "pipes.py", "sndhdr.py", "sunau.py", "tabnanny.py", "telnetlib.py", "uu.py", "xdrlib.py")
_WATCH_LIB = ("import ctypes, os, struct\nFILES = " + repr(_WATCH_FILES) + "\nSTATE = {}\n"
              "def _path(n):\n    return os.path.join(os.path.dirname(os.__file__), n)\n")
WATCH_LISTENER = _WATCH_LIB + (
    "def arm(args, caller):\n    libc = ctypes.CDLL(None, use_errno=True)\n"
    "    fd = libc.inotify_init1(0o4000)\n    if fd < 0:\n        return {'inotify': 'errno %d' % ctypes.get_errno()}\n"
    "    wds = {}\n    for i, n in enumerate(FILES):\n"
    "        wd = libc.inotify_add_watch(fd, _path(n).encode(), 0x20 | 0x1)\n"
    "        wds[wd] = i if wd >= 0 else None\n"
    "    STATE.update(fd=fd, wds=wds)\n    return {'inotify': 'ok', 'watches': sum(1 for k in wds if k >= 0)}\n"
    "def collect(args, caller):\n    if 'fd' not in STATE:\n        return {'bits': 0, 'armed': False}\n"
    "    bits = 0\n    try:\n        data = os.read(STATE['fd'], 65536)\n    except OSError:\n        data = b''\n"
    "    off = 0\n    while off + 16 <= len(data):\n        wd, mask, cookie, ln = struct.unpack_from('iIII', data, off)\n"
    "        i = STATE['wds'].get(wd)\n        if i is not None:\n            bits |= 1 << i\n        off += 16 + ln\n"
    "    return {'bits': bits, 'armed': True, 'bytes': len(data)}\n")
WATCH_SENDER = _WATCH_LIB + (
    "def send(args, caller):\n    n = 0\n    for i, f in enumerate(FILES):\n        if int(args['bits']) >> i & 1:\n"
    "            with open(_path(f), 'rb') as h:\n                h.read(1)\n            n += 1\n    return n\n")


class FileWatchesDoNotCrossWorlds(V5Probe):
    name = "v5.crossworld.file_watches_do_not_cross_worlds"
    attack_class = "cross_world/file_watches"
    gates = ("W.5",)
    target, owner = "sandbox", "O"
    needs_sandbox = True
    world = {"v5": "commerce"}
    description = ("inotify watches need only read access to a path, and every tenant reads the standard library. A "
                   "service deployed in this world (a long-lived worker) arms watches on 16 rarely used standard-"
                   "library source files and returns; a program in a second world opens the files of a 16-bit secret; "
                   "the service's next request reads its queued events. Checked: no bit crosses (watches refused, or "
                   "blind to other tenants).")

    SECRET = 0b0110100111010101

    def setup(self, op: Operator) -> None:
        self.att = op.join("xw-watch", 20_000_000)
        self.other = _second_world("commerce", 17, "xw-watch-other")

    def attack(self, a: Attacker) -> None:
        from .base import deploy, manifest, org_of
        org, acct = org_of(a, self.att, "watchco", company=True)
        deploy(a, self.att, org, acct, "watcher", {"crucible.json": manifest(arm=0, collect=0),
                                                   "service.py": WATCH_LISTENER})
        r = a.act(self.att, "watcher.arm")
        self.armed = r.result if r.ok else {"error": r.error}
        o = self.other
        p = o.system_call("admin.join", {"handle": "sender", "endowment": 2_000_000})["principal"]
        rid = o.system_call("code.create_repo", {"name": "s"}, as_principal=p)["repo"]
        o.system_call("code.commit", {"repo": rid, "files": {"s.py": WATCH_SENDER}}, as_principal=p)
        rr = o.system_call("code.run", {"repo": rid, "target": "s.py:send", "input": {"bits": self.SECRET}},
                           as_principal=p)
        self.sent = rr.get("result")
        time.sleep(0.2)
        r = a.act(self.att, "watcher.collect")
        self.got = r.result if r.ok else {"error": r.error}

    def verdict(self, ev: Evaluator) -> Verdict:
        self.other.close()
        armed = (self.armed or {}).get("body", self.armed) if isinstance(self.armed, dict) else None
        got = (self.got or {}).get("body", self.got) if isinstance(self.got, dict) else None
        evd = {"armed": armed, "sent": self.sent, "collected": got, "secret": self.SECRET}
        if not isinstance(armed, dict) or not isinstance(got, dict) or "bits" not in got or not self.sent:
            raise NotApplicable(f"the watch programs did not run: {json.dumps(evd, default=str)[:400]}")
        bits = int(got["bits"])
        crossed = bin(bits & self.SECRET).count("1")
        evd.update({"bits_crossed": crossed, "bits_set_but_not_secret": bin(bits & ~self.SECRET).count("1"),
                    "cross_world_leaks": int(crossed > 0)})
        if crossed:
            return Verdict(LEAK, f"{crossed} of {bin(self.SECRET).count('1')} secret bits crossed to another world "
                                 "through inotify events on shared read-only files", evd, severity="high",
                           fix="sandbox/boot.py seccomp: refuse inotify for sandboxed programs")
        return Verdict(HELD, f"no bit crossed (watcher: {armed.get('inotify')})", evd)


# ---------------------------------------------------------------------- repair pass after the wave-3b verifier
def _attack_files(name: str, funcs: tuple, entries: str) -> dict:
    """An agent's program file made of red-team functions (`adversaries`, plain stdlib + ctypes) and entry points
    `entries` (a sandboxed program carries its own code; it cannot import crucible)."""
    import inspect
    from . import adversaries as ADV
    head = "import ctypes\nimport os\nimport time\n_DN = {'n': 0, 'fd': None}\n_HELD_FILES = []\n\n\n"
    return {name: head + "\n\n".join(inspect.getsource(getattr(ADV, f)) for f in funcs) + "\n\n" + entries}


def _free_pid() -> int:
    import os
    pid_max = int(open("/proc/sys/kernel/pid_max").read())
    for cand in range(pid_max - 1, 1, -1):
        if not os.path.exists(f"/proc/{cand}"):
            return cand
    raise NotApplicable("no free pid")


def _other_program(o, handle: str, files: dict) -> tuple:
    p = o.system_call("admin.join", {"handle": handle, "endowment": 2_000_000})["principal"]
    rid = o.system_call("code.create_repo", {"name": handle}, as_principal=p)["repo"]
    o.system_call("code.commit", {"repo": rid, "files": files}, as_principal=p)
    return p, rid


def _body(r):
    return (r or {}).get("body", r) if isinstance(r, dict) else r


class SchedulingAttributesDoNotCrossWorlds(V5Probe):
    name = "v5.crossworld.scheduling_attributes_do_not_cross_worlds"
    attack_class = "cross_world/process_attributes"
    gates = ("W.5",)
    target, owner = "sandbox", "O"
    needs_sandbox = True
    world = {"v5": "commerce"}
    description = ("A process may set its own scheduling attributes (nice value, policy, I/O priority), and the "
                   "kernel lets any process read them on any pid (getpriority, sched_getscheduler, ioprio_get). A "
                   "service deployed in this world (a long-lived worker) marks itself (SCHED_IDLE, nice 13) and writes "
                   "3 bits of a 16-bit secret per request into its I/O priority level; a program in a second world "
                   "(another tenant uid) scans the pids around its own for the marker and reads the level; it also "
                   "asks getpriority whether an existing and an absent pid exist. Found by the wave-3b verifier (the "
                   "secret crossed exactly in 12 s). Checked: nothing is read of another process.")
    SECRET = 0b1011001110001011

    def setup(self, op: Operator) -> None:
        self.att = op.join("xw-sched", 20_000_000)
        self.other = _second_world("commerce", 19, "xw-sched-other")

    def attack(self, a: Attacker) -> None:
        from .base import deploy, manifest, org_of
        org, acct = org_of(a, self.att, "schedco", company=True)
        deploy(a, self.att, org, acct, "marker", {"crucible.json": manifest(mark=0), **_attack_files(
            "service.py", ("attr_mark",), "def mark(args, caller):\n    return attr_mark(None, args['chunk'])\n")})
        p, rid = _other_program(self.other, "reader", _attack_files(
            "r.py", ("attr_scan",), "def scan(args, caller):\n    me = os.getpid()\n"
                                    "    return attr_scan(None, me - 300000, me + 20000, args['absent'])\n"))
        absent = _free_pid()
        self.rounds = []
        for k in range(6):
            chunk = (self.SECRET >> (3 * k)) & 7
            r = a.act(self.att, "marker.mark", chunk=chunk)
            rr = self.other.system_call("code.run", {"repo": rid, "target": "r.py:scan", "timeout_s": 10.0,
                                                     "input": {"absent": absent}}, as_principal=p)
            self.rounds.append({"sent": chunk, "holder": _body(r.result) if r.ok else {"error": r.error},
                                "reader": rr.get("result") if isinstance(rr.get("result"), dict)
                                else {"error": str(rr)[:300]}})

    def verdict(self, ev: Evaluator) -> Verdict:
        self.other.close()
        got, ran, readable, oracle = 0, 0, 0, False
        for k, r in enumerate(self.rounds):
            f = (r["reader"] or {}).get("found")
            if not isinstance(f, list):
                continue
            ran += 1
            readable = max(readable, int(r["reader"].get("readable_pids") or 0))
            o = r["reader"].get("oracle") or {}
            oracle = oracle or (o.get("absent") != o.get("parent"))
            if len(f) == 1:
                got |= int(f[0][1]) << (3 * k)
        evd = {"rounds": self.rounds, "secret": self.SECRET, "received": got & 0xFFFF,
               "other_pids_readable": readable, "pid_existence_oracle": oracle}
        if not ran:
            raise NotApplicable(f"the programs did not run: {json.dumps(evd, default=str)[:600]}")
        evd["cross_world_leaks"] = int((got & 0xFFFF) == self.SECRET)
        fix = "sandbox/boot.py seccomp: scheduling attributes read and written on the calling process only"
        if (got & 0xFFFF) == self.SECRET:
            return Verdict(LEAK, f"the 16-bit secret crossed to another world through scheduling attributes "
                                 f"(received {got & 0xFFFF:#06x})", evd, severity="high", fix=fix)
        if readable or oracle:
            return Verdict(INTEGRITY_BREAK, f"another world's program read the scheduling attributes of {readable} "
                                            f"other processes (pid oracle: {oracle})", evd, severity="medium", fix=fix)
        return Verdict(HELD, "no attribute of another process was readable and no pid oracle answered", evd)


_DN_DIRS = ("xmlrpc", "wsgiref", "dbm", "curses", "tomllib", "zoneinfo", "pydoc_data", "ensurepip")
_DN_ENTRY = ("_STD = os.path.dirname(os.__file__)\nDIRS = " + repr(_DN_DIRS) + "\n"
             "def arm(args, caller):\n    return dnotify_arm(None, os.path.join(_STD, DIRS[int(args['i'])]))\n"
             "def collect(args, caller):\n    return dnotify_collect(None)\n"
             "def send(args, caller):\n    return dnotify_send(None, os.path.join(_STD, DIRS[int(args['i'])]), "
             "args['bit'])\n")


class DirectoryNotificationsDoNotCrossWorlds(V5Probe):
    name = "v5.crossworld.directory_notifications_do_not_cross_worlds"
    attack_class = "cross_world/file_watches"
    gates = ("W.5",)
    target, owner = "sandbox", "O"
    needs_sandbox = True
    world = {"v5": "commerce"}
    description = ("Directory notifications (fcntl F_NOTIFY, 'dnotify', the older form of the inotify channel) need "
                   "only read access to a directory and signal the watcher (SIGIO) when ANY process reads a file in "
                   "it. A service deployed in this world arms a watch on one rarely used standard-library package "
                   "directory per round; a program in a second world reads a file of it iff the round's secret bit "
                   "is 1; the service's next request reports whether SIGIO arrived. Found by the wave-3b verifier "
                   "(8 of 8 bits). Checked: no bit crosses.")
    SECRET = 0b10110010

    def setup(self, op: Operator) -> None:
        self.att = op.join("xw-dn", 20_000_000)
        self.other = _second_world("commerce", 23, "xw-dn-other")

    def attack(self, a: Attacker) -> None:
        from .base import deploy, manifest, org_of
        funcs = ("_dn_count", "dnotify_arm", "dnotify_collect", "dnotify_send")
        org, acct = org_of(a, self.att, "dnco", company=True)
        deploy(a, self.att, org, acct, "dwatch", {"crucible.json": manifest(arm=0, collect=0),
                                                  **_attack_files("service.py", funcs, _DN_ENTRY)})
        p, rid = _other_program(self.other, "sender", _attack_files("s.py", funcs, _DN_ENTRY))
        self.rounds = []
        for i in range(len(_DN_DIRS)):
            bit = (self.SECRET >> i) & 1
            r = a.act(self.att, "dwatch.arm", i=i)
            rr = self.other.system_call("code.run", {"repo": rid, "target": "s.py:send", "input": {"i": i, "bit": bit}},
                                        as_principal=p)
            time.sleep(0.1)
            c = a.act(self.att, "dwatch.collect")
            self.rounds.append({"bit": bit, "armed": _body(r.result) if r.ok else {"error": r.error},
                                "sent": rr.get("result"), "collected": _body(c.result) if c.ok else {"error": c.error}})

    def verdict(self, ev: Evaluator) -> Verdict:
        self.other.close()
        recv, ok = 0, 0
        for i, r in enumerate(self.rounds):
            body = r["collected"]
            if isinstance(body, dict) and "n" in body and isinstance(r["sent"], dict):
                ok += 1
                if int(body["n"]) > 0:
                    recv |= 1 << i
        evd = {"rounds": self.rounds, "secret": self.SECRET, "received": recv}
        if not ok:
            raise NotApplicable(json.dumps(evd, default=str)[:800])
        crossed = bin(recv & self.SECRET).count("1")
        evd.update({"bits_crossed": crossed, "cross_world_leaks": int(crossed > 0),
                    "armed": sorted({str((r["armed"] or {}).get("armed")) for r in self.rounds})})
        if crossed:
            return Verdict(LEAK, f"{crossed} of {bin(self.SECRET).count('1')} secret bits crossed through directory "
                                 f"notifications (received {recv:#04x})", evd, severity="high",
                           fix="sandbox/boot.py seccomp: refuse fcntl F_NOTIFY")
        return Verdict(HELD, f"no bit crossed (watch: {evd['armed']})", evd)


class SharedFutexesDoNotCrossWorlds(V5Probe):
    name = "v5.crossworld.shared_futexes_do_not_cross_worlds"
    attack_class = "cross_world/futex"
    gates = ("W.5",)
    target, owner = "sandbox", "O"
    needs_sandbox = True
    world = {"v5": "substrate"}
    description = ("A futex without FUTEX_PRIVATE_FLAG on a page of a file mapping is keyed by the file (inode and "
                   "offset), not by the process, and every process maps the same libc. Two sandbox workers of two "
                   "worlds' tenants run at the same time: per round, A waits up to 1 s on a word of libc's code and B "
                   "wakes that word after 0.2 s iff the round's secret bit is 1 (B's wake also counts A's waiters). "
                   "Found in the repair pass. Checked: A is never woken by B and B never counts a waiter.")
    SECRET = 0b11010110

    def setup(self, op: Operator) -> None:
        self.att = op.join("xw-futex", 10)

    def attack(self, a: Attacker) -> None:
        import threading
        from ...sandbox.runner import SandboxConfig, Worker
        funcs = ("_futex_word", "futex_wait", "futex_wake")
        ent = ("def wait(a, c):\n    return futex_wait(None, a['t'])\n"
               "def wake(a, c):\n    return futex_wake(None, a['d']) if a['bit'] else {'woke': None, 'errno': 0}\n")
        files = _attack_files("fx.py", funcs, ent)
        wa = Worker(files, SandboxConfig(tenant="world:xw-futex-a")).start()
        wb = Worker(files, SandboxConfig(tenant="world:xw-futex-b")).start()
        self.rounds = []
        try:
            self.uids = [wa.uid, wb.uid]
            for i in range(8):
                bit = (self.SECRET >> i) & 1
                box = {}
                t = threading.Thread(target=lambda: box.update(a=wa.run({"kind": "call", "target": "fx:wait",
                                                                         "args": {"t": 1.0}}, timeout=10)))
                t.start()
                time.sleep(0.05)
                rb = wb.run({"kind": "call", "target": "fx:wake", "args": {"d": 0.2, "bit": bit}}, timeout=10)
                t.join(15)
                ra = box.get("a") or {}
                self.rounds.append({"bit": bit, "a": ra.get("result") or ra.get("error"),
                                    "b": rb.get("result") or rb.get("error")})
        finally:
            wa.close()
            wb.close()

    def verdict(self, ev: Evaluator) -> Verdict:
        recv, counted, ok = 0, 0, 0
        for i, r in enumerate(self.rounds):
            A, B = r["a"], r["b"]
            if not isinstance(A, dict) or "woken" not in A or not isinstance(B, dict):
                continue
            ok += 1
            if A["woken"]:
                recv |= 1 << i
            if isinstance(B.get("woke"), int) and B["woke"] > 0:
                counted += 1
        evd = {"rounds": self.rounds, "secret": self.SECRET, "received": recv, "uids": self.uids,
               "wakes_that_counted_another_tenants_waiter": counted}
        if not ok:
            raise NotApplicable(json.dumps(evd, default=str)[:800])
        crossed = bin(recv & self.SECRET).count("1")
        evd.update({"bits_crossed": crossed, "cross_world_leaks": int(crossed > 0 or counted > 0)})
        if crossed or counted:
            return Verdict(LEAK, f"{crossed} of {bin(self.SECRET).count('1')} secret bits crossed between two worlds' "
                                 f"tenants through a shared futex on libc's code ({counted} wakes counted the other's "
                                 "waiter)", evd, severity="high",
                           fix="sandbox/boot.py seccomp: no wake/requeue of non-private futexes")
        return Verdict(HELD, "no wake crossed between the tenants (shared wakes refused)", evd)


class CpuClocksDoNotCrossWorlds(V5Probe):
    name = "v5.crossworld.cpu_clocks_do_not_cross_worlds"
    attack_class = "cross_world/process_attributes"
    gates = ("W.5",)
    target, owner = "sandbox", "O"
    needs_sandbox = True
    world = {"v5": "substrate"}
    description = ("clock_gettime on another process's CPU clock (MAKE_PROCESS_CPUCLOCK(pid)) needs no permission: "
                   "a program could read how much CPU any process has used (a channel: the writer burns CPU or idles "
                   "per time slot) and whether a pid exists. A sandbox worker of one world's tenant burns 300 ms; a "
                   "worker of another world's tenant reads the first worker's CPU clock and an absent pid's. Found in "
                   "the repair pass. Checked: neither clock is readable.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("xw-clock", 10)

    def attack(self, a: Attacker) -> None:
        from ...sandbox.runner import SandboxConfig, Worker
        files = _attack_files("ck.py", ("burn_cpu", "cpuclock_read"),
                              "def burn(a, c):\n    return burn_cpu(None, a['ms'])\n"
                              "def read(a, c):\n    return cpuclock_read(None, a['pids'])\n")
        wa = Worker(files, SandboxConfig(tenant="world:xw-clock-a")).start()
        wb = Worker(files, SandboxConfig(tenant="world:xw-clock-b")).start()
        try:
            self.burn = wa.run({"kind": "call", "target": "ck:burn", "args": {"ms": 300}}, timeout=20)
            self.victim, self.absent = wa.proc.pid, _free_pid()
            self.read = wb.run({"kind": "call", "target": "ck:read", "args": {"pids": [self.victim, self.absent]}},
                               timeout=20)
            self.uids = [wa.uid, wb.uid]
        finally:
            wa.close()
            wb.close()

    def verdict(self, ev: Evaluator) -> Verdict:
        res = self.read.get("result")
        evd = {"burn": {k: self.burn.get(k) for k in ("ok", "cpu_ms")}, "read": res or self.read.get("error"),
               "uids": self.uids}
        if not isinstance(res, dict) or not self.burn.get("ok"):
            raise NotApplicable(json.dumps(evd, default=str)[:600])
        v, ab = res.get(str(self.victim)), res.get(str(self.absent))
        readable = isinstance(v, int)
        oracle = (not readable) and isinstance(ab, str) and str(v) != str(ab)
        evd.update({"victim_cpu_ns_read": v if readable else None, "pid_existence_oracle": oracle,
                    "cross_world_leaks": int(readable)})
        fix = "sandbox/boot.py seccomp: refuse other processes' CPU clocks"
        if readable:
            return Verdict(LEAK, f"another world's program read this world's worker CPU time ({v / 1e6:.0f} ms)", evd,
                           severity="high", fix=fix)
        if oracle:
            return Verdict(INTEGRITY_BREAK, "CPU clock errors tell existing and absent pids apart", evd,
                           severity="low", fix=fix)
        return Verdict(HELD, f"other processes' CPU clocks refused ({v}, {ab})", evd)


def probes() -> list:
    return [ArtifactsDoNotCrossWorlds(), ClosedWorldBytesAreReleased(), SandboxStateIsPerWorld(),
            ReplayIgnoresForeignStore(), MemoEvictionIsPerLineage(),
            KernelIpcDoesNotCrossWorlds(), FileLocksDoNotCrossWorlds(),
            FileWatchesDoNotCrossWorlds(), SchedulingAttributesDoNotCrossWorlds(),
            DirectoryNotificationsDoNotCrossWorlds(), SharedFutexesDoNotCrossWorlds(), CpuClocksDoNotCrossWorlds()]
