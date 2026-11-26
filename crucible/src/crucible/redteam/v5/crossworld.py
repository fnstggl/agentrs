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


def probes() -> list:
    return [ArtifactsDoNotCrossWorlds(), ClosedWorldBytesAreReleased(), SandboxStateIsPerWorld(),
            ReplayIgnoresForeignStore(), MemoEvictionIsPerLineage(),
            KernelIpcDoesNotCrossWorlds()]
