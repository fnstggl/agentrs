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
    attack_class = "cross_world/memory_exhaustion"
    gates = ("W.5", "W.7")
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


def probes() -> list:
    return [ArtifactsDoNotCrossWorlds(), ClosedWorldBytesAreReleased(), SandboxStateIsPerWorld(),
            ReplayIgnoresForeignStore()]
