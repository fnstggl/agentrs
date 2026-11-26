"""Gate SP-2.3 (MASTER_SPEC Part 2.3): shared world substrate - common typed primitives, no duplicate shared facts,
cross-module transactional integrity, recorded neural inference replayable without rerunning it.

Metrics (docs/v5/ACCEPTANCE_MANIFEST.json): shared_primitive_kinds >= 19 (D-23), cross_module_rollback_failures == 0,
inference_replay_without_rerun == true. Measured on the real system (D-21; audit A6/A9 found the 5.0 version measured
test-only modules and a vacuous, opt-in duplicate check):

1. shared_primitive_kinds - of the 19 primitives MASTER_SPEC Part 2.3 lists (primitives.SPEC_PRIMITIVES, checked
   against the spec text), those whose kind (primitives.OWNERS) has an owner that is a shipped kernel module really
   creating the owner table, according to the MANDATORY scan of every table, column table and array of every shipped
   module (crucible.substrate.facts.audit), and that have no copy elsewhere - a kind with a known duplicate on the
   reviewed allow-list does NOT count (it is reported in spec_primitives_with_known_duplicates; facts.py holds its
   migration). The refining kinds OWNERS adds are reported apart (additional_kinds_single_owner) and never counted
   in place of a listed one. The test records the number honestly; the acceptance checker compares it with the
   criterion.
2. cross_module_rollback_failures - (a) fault injection into REAL multi-module transactions of a world with every
   shipped module: each scheduled job (and agent command) whose events touch two or more modules is re-run on a fork
   with a fault raised after a random one of its events; the fork must commit only the kernel's failure record and
   leave every module table and column exactly as before; (b) randomised commands that compose operations of six
   real modules (bank, artifacts with a typed subscriber, assets, columns, attention, identity) through a test
   driver and fail in seven real ways.
3. inference_replay_without_rerun - (a) the population's neural classes (pop_adaptive GRU policies, pop_strategic):
   a SQLite-stored run in one process, then replay and regeneration in NEW processes from the store alone, counting
   inference executions at the recorded-inference boundary (must be 0, results exact), and a cross-process resume;
   (b) a torch policy with OS-entropy sampling (a test module) whose rerun differs, regenerated and replayed exactly.
"""
import base64
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import crucible
from crucible.acceptance.v5 import evidence
from crucible.core import schema as S
from crucible.core.errors import CrucibleError, Forbidden, InvalidAction
from crucible.core.kernel import MODULES, Module, World, WorldSpec, applier, endpoint, register_module
from crucible.substrate import artifacts as A
from crucible.substrate import attention as AT
from crucible.substrate import bus
from crucible.substrate import columns as C
from crucible.substrate import facts
from crucible.substrate import ids
from crucible.substrate import inference as I
from crucible.substrate import primitives as P
from crucible.worldgen.generator import FAMILIES, generate

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_2_3.py"
TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def with_deps(names):
    order = []

    def add(m):
        if m in order:
            return
        for d in MODULES[m].depends:
            add(d)
        order.append(m)
    for m in names:
        add(m)
    return order


# ---------------------------------------------------------------------------------------------- 1. primitives
def measure_primitives() -> dict:
    audit = facts.audit()                                    # every shipped module, each installed on its own
    mods = with_deps(["identity", "bank", "admin", *P.owner_modules()])
    w = World.create(WorldSpec("sp23-prim", 1, [{"name": m, "config": {}} for m in mods]))
    probs = P.problems(w)
    rep = P.check_no_duplicates(w) if not probs else {"kinds": {}}
    dup_kinds = sorted({facts.REVIEWED[k]["kind"] for k in audit["known_duplicates"]})
    owned, single = [], []
    for kind in P.kinds():
        o = P.owner(kind)
        if o["module"] == P.KERNEL:
            ok = o["table"].startswith("@") or o["table"] in w.state.tables
        else:
            ok = f"{o['module']}.{o['table']}" in audit["owners"] and rep["kinds"].get(kind, {}).get("present")
        if ok:
            owned.append(kind)
            if kind not in dup_kinds:
                single.append(kind)
    fam_dups = {f: P.problems(World(generate({"family": f}, 0), f"sp23-{f}")) for f in sorted(FAMILIES)}
    # D-23: the criterion counts the 19 primitives MASTER_SPEC Part 2.3 lists (in the spec's words, checked against
    # the spec text); the refining kinds OWNERS adds (account, listing, ...) are reported apart, never in their place
    listed = P.spec_primitives_from_text(open(os.path.join(TESTS, os.pardir, "docs", "v5", "MASTER_SPEC.md"),
                                              encoding="utf-8").read())
    assert listed == list(P.SPEC_PRIMITIVES), (listed, list(P.SPEC_PRIMITIVES))
    spec_kinds = [P.SPEC_PRIMITIVES[x] for x in listed]
    spec_single = [k for k in spec_kinds if k in single]
    return {"shared_primitive_kinds": len(spec_single), "spec_primitives_listed": len(listed),
            "spec_primitives_owned": sum(1 for k in spec_kinds if k in owned),
            "spec_primitives_single_owner": ",".join(spec_single),
            "spec_primitives_with_known_duplicates": ",".join(k for k in spec_kinds if k in owned and k not in single),
            "spec_primitives_without_owner": ",".join(k for k in spec_kinds if k not in owned),
            "additional_kinds_single_owner": ",".join(k for k in single if k not in spec_kinds),
            "all_kinds_single_owner": len(single), "shared_primitive_kinds_owned": len(owned),
            "primitive_kinds_declared": len(P.kinds()), "kinds_with_known_duplicates": ",".join(dup_kinds),
            "known_duplicate_tables": len(audit["known_duplicates"]),
            "unexplained_duplicate_facts": len(audit["problems"]),
            "fact_scan_modules": audit["modules"], "fact_scan_items": audit["items"],
            "fact_scan_uninstallable": len(audit["uninstallable"]),
            "owners_world_modules": len(w.modules), "owners_world_problems": len(probs),
            "families_checked_for_duplicates": len(fam_dups),
            "families_with_duplicate_facts": sum(1 for v in fam_dups.values() if v),
            "typed_event_types": len(bus.types()), "_audit": audit}


# ---------------------------------------------------------------------------------------------- 2a. fault injection
KERNEL_T = {"cmd", "cmd.rejected", "proc", "proc.failed", "time", "sched.add", "sched.del", "ctr", "idem", "ext",
            "note", "know", "principal.add", "principal.update"}
KERNEL_TABLES = ("sched", "principals", "idem", "knowledge")


def all_modules_spec(seed=7) -> WorldSpec:
    first = ["identity", "bank", "admin", "artifacts", "cols", "code", "search", "forum", "cloud", "net", "appdb",
             "popsites", "pop", "popweb"]
    excl = {"calendar", "crowd", "feed"}       # alias of cal; the stand-in population; feed needs register_activity
    names = facts._with_deps(first + [m for m in facts.shipped_modules() if m not in excl and m not in first])
    return WorldSpec("sp23-all", seed, [{"name": m, "config": {}} for m in names], {"cost_model": "deterministic"})


def _modules_of(world, evs) -> set:
    out = set()
    for e in evs:
        if e.type in KERNEL_T:
            continue
        if e.type in ("cols.batch", "cols.grow"):
            by = C._registry(world).get(e.payload["table"], {}).get("by", {})
            out.add(next(iter(by.values()), "cols"))
        else:
            out.add(getattr(getattr(world.appliers.get(e.type), "__self__", None), "name", "?"))
    return out


def _faulted(blob, k, run):
    """Fork from `blob`, arm a fault after the k-th applied event, run the same operation; the fork and the
    committed tail."""
    f = World.from_blob(blob, world_id="sp23-fault")
    orig = f._apply
    state = {"n": 0, "armed": True}

    def ap(ev):
        orig(ev)
        if state["armed"]:
            state["n"] += 1
            if state["n"] == k:
                state["armed"] = False
                raise RuntimeError("injected fault (gate SP-2.3)")
    f._apply = ap
    s0 = f.seq
    run(f)
    return f, f.log[s0 - f.log_base:]


def _rolled_back(blob, f, tail, record_type) -> tuple[bool, str]:
    types = [e.type for e in tail]
    leaked = [t for t in types if t not in KERNEL_T]
    if leaked or record_type not in types:
        return False, f"leaked {leaked[:4]} / records {types[:5]}"
    pre = World.from_blob(blob, world_id="sp23-pre")
    if World.replay(pre.spec, [e.to_dict() for e in tail], base_blob=blob).digest() != f.digest():
        return False, "state differs from the replay of its committed log"
    a = {x: d for x, d in pre.state.table_digests().items() if x not in KERNEL_TABLES}
    b = {x: d for x, d in f.state.table_digests().items() if x not in KERNEL_TABLES}
    if a != b:
        return False, f"module tables changed: {sorted(x for x in set(a) | set(b) if a.get(x) != b.get(x))[:4]}"
    if set(pre.state.arrays) != set(f.state.arrays) or not all(
            np.array_equal(pre.state.arrays[x], f.state.arrays[x]) for x in pre.state.arrays):
        return False, "columns changed"
    return True, ""


def measure_fault_injection(n_targets: int = 60, seed: int = 5) -> dict:
    from crucible.agents.scripted import _random_args
    w = World.create(all_modules_spec(), world_id="sp23-all")
    w.advance_time(3 * 3600)
    rng = np.random.default_rng(seed)
    agents = [w.system_call("admin.join", {"handle": f"sp23-f{i}", "endowment": 200_000})["principal"]
              for i in range(3)]
    eps = sorted(e for e in w.endpoints if not e.startswith(("admin.", "world.")))
    st = {"jobs": 0, "commands": 0, "targets": 0, "job_targets": 0, "command_targets": 0, "failures": 0,
          "max_modules": 0, "subscriber_targets": 0, "v5_module_targets": 0}
    v5 = {n for n in w.modules if MODULES[n].__module__.split(".")[1] in crucible.V5_PACKAGES}
    fails, touched = [], set()
    end = w.now + 48 * 3600           # until n_targets multi-module transactions were found (or two days)
    while st["targets"] < n_targets and w.now < end:
        # alternate: one agent command (random, schema-valid) and one scheduled job
        blob, s0 = w.snapshot_blob(), w.seq
        if st["commands"] <= st["jobs"]:
            ep = w.endpoints[eps[int(rng.integers(0, len(eps)))]]
            actor = agents[st["commands"] % 3]
            act = {"tool": ep.name, "args": _random_args(ep.params, rng, actor)}
            r = w.execute(actor, act)
            st["commands"] += 1
            kind, run, rec = "command", (lambda x, a=act, p=actor: x.execute(p, a)), "cmd.rejected"
            if not r.ok:
                continue
        else:
            nxt = w.next_job_time()
            if nxt is None:
                break
            w.advance_time(nxt, max_jobs=1)
            st["jobs"] += 1
            kind, run, rec = "job", (lambda x, t=nxt: x.advance_time(t, max_jobs=1)), "proc.failed"
        evs = w.log[s0 - w.log_base:]
        ms = _modules_of(w, evs)
        if len(ms) < 2:
            continue
        k = int(rng.integers(1, len(evs) + 1))
        f, tail = _faulted(blob, k, run)
        ok, why = _rolled_back(blob, f, tail, rec)
        st["targets"] += 1
        st[f"{kind}_targets"] += 1
        st["max_modules"] = max(st["max_modules"], len(ms))
        st["subscriber_targets"] += any(e.via for e in evs)
        st["v5_module_targets"] += bool(ms & v5)
        touched |= ms
        if not ok:
            st["failures"] += 1
            fails.append((kind, evs[0].type, evs[0].payload.get("proc") or evs[0].payload.get("tool"), k, why))
    st["modules_touched"] = len(touched)
    st["world_modules"] = len(w.modules)
    return {"stats": st, "fails": fails, "touched": sorted(touched)}


# ---------------------------------------------------------------------------------------------- 2b. composed failures
bus.event_type("sp23.indexed", S.obj({"ref": {"type": "string"}, "owner": {"type": "string"},
                                      "fee": {"type": "integer", "minimum": 0}}, ["ref", "owner", "fee"]),
               owner="sp23_index", doc="a content index entry (gate SP-2.3 coupling)")


@register_module
class Sp23Index(Module):
    """A content index coupled to the artifact store only through the typed `artifact.stored` event: it indexes every
    participant artifact in the same transaction, charges the owner an indexing fee through `bank`, and refuses
    content flagged as banned (which must undo the whole command, the stored artifact included)."""
    name = "sp23_index"
    depends = ("artifacts", "bank")

    def install(self, world):
        world.state.declare_table("sp23_index", index=("owner",))
        bus.subscribe(world, "artifact.stored", self.on_stored, owner=self.name)

    def on_stored(self, ctx, ev):
        p = ev.payload
        if p["owner"] == "sys":
            return
        if p["meta"].get("banned"):
            raise Forbidden("the index refuses banned content")
        bank = ctx.world.modules["bank"]
        acct = bank.primary_account(ctx.state, p["owner"])
        bank.move(ctx, acct, "acct_fees", 3, "fee", "content indexing")
        bus.emit(ctx, "sp23.indexed", {"ref": p["ref"], "owner": p["owner"], "fee": 3}, vis=(p["owner"],))

    @applier("sp23.indexed")
    def ap_indexed(self, ev):
        self.world.state.put("sp23_index", ev.payload["ref"], {"owner": ev.payload["owner"], "fee": ev.payload["fee"]})


OPS = ("pay", "put", "asset", "batch", "attention", "grow", "principal", "contained")
FAILS = ("none", "funds", "quota", "subscriber", "batch", "handler", "bug", "assets_class")
EXPECTED_CODE = {"funds": "insufficient_funds", "quota": "capacity_exceeded", "subscriber": "forbidden",
                 "batch": "invalid_action", "handler": "invalid_action", "bug": "error", "assets_class": "invalid_action"}


@register_module
class Sp23Driver(Module):
    """Composes real operations of six modules in ONE command (bank transfer, artifact put with its index coupling,
    asset registration, column batch, attention spend, column growth, principal creation, a contained failing
    sub-call), then fails in one of several real ways - or not. Data are derived from the arguments only."""
    name = "sp23_driver"
    host = "sp23"
    depends = ("identity", "bank", "artifacts", "cols", "assets", "sp23_index")

    def install(self, world):
        n = 400
        C.declare(world, "pop", n, {"score": ("float32", (), 0.0), "seen": ("int32", (), 0)})
        AT.declare(world, n)

    def genesis(self, ctx):
        AT.reset(ctx, budget=np.full(400, 60, np.float32))

    def _data(self, seed, i, size=64):
        return hashlib.sha256(f"{seed}:{i}".encode()).digest() * (size // 32)

    def _op(self, ctx, op, seed, i, to):
        rng = np.random.default_rng([seed, i])
        n = C.rows(ctx.world, "pop")
        bank = ctx.world.modules["bank"]
        if op == "pay":
            ctx.call("bank.transfer", {"from_account": bank.primary_account(ctx.state, ctx.actor), "to": to,
                                       "amount": 7, "memo": "sp23"})
        elif op == "put":
            ctx.call("artifact.put", {"data_b64": base64.b64encode(self._data(seed, i)).decode(),
                                      "mime": "application/octet-stream", "meta": {"i": i}})
        elif op == "asset":
            ctx.call("asset.register", {"class": "licence", "title": f"licence {seed}-{i}"})
        elif op == "batch":
            k = 50
            C.batch(ctx, "pop", "sp23.score", rng.integers(0, n, k), {"score": rng.random(k, dtype=np.float32),
                                                                      "seen": np.ones(k, np.int32)}, mode="add")
        elif op == "attention":
            AT.spend(ctx, rng.integers(0, n, 40), rng.random(40) * 20, activity="sp23")
        elif op == "grow":
            C.grow(ctx, "pop", 3)
        elif op == "principal":
            ids.ensure_principal(ctx, int(rng.integers(0, n)), account=True)
        elif op == "contained":
            try:
                ctx.call("sp23.doomed", {"seed": seed, "i": i})
            except CrucibleError:
                pass               # the caller handles the refusal; the sub-call's partial effects must be gone

    @endpoint("compose", S.obj({"ops": S.s_arr(S.s_enum(list(OPS)), maxItems=16), "fail": S.s_enum(list(FAILS)),
                                "seed": S.s_int(minimum=0), "to": S.s_str(maxLength=40)},
                               ["ops", "fail", "seed", "to"]), doc="compose cross-module operations, then fail or not")
    def ep_compose(self, ctx, ops, fail, seed, to):
        for i, op in enumerate(ops):
            self._op(ctx, op, seed, i, to)
        bank = ctx.world.modules["bank"]
        mine = bank.primary_account(ctx.state, ctx.actor)
        if fail == "funds":
            ctx.call("bank.transfer", {"from_account": mine, "to": to, "amount": bank.balance(ctx.state, mine) + 1})
        elif fail == "quota":
            ctx.call("artifact.put", {"data_b64": base64.b64encode(b"q" * 5000).decode(), "mime": "text/plain"})
        elif fail == "subscriber":
            ctx.call("artifact.put", {"data_b64": base64.b64encode(self._data(seed, 99)).decode(),
                                      "mime": "text/plain", "meta": {"banned": True}})
        elif fail == "batch":
            C.batch(ctx, "pop", "sp23.bad", np.array([0, C.rows(ctx.world, "pop")]), {"seen": np.ones(2, np.int32)})
        elif fail == "handler":
            raise InvalidAction("the composed operation is refused at the end")
        elif fail == "bug":
            raise ZeroDivisionError("a bug after all side effects")
        elif fail == "assets_class":
            ctx.call("asset.register", {"class": "equity", "title": "shares are owned by business"})
        return {"ops": len(ops)}

    @endpoint("doomed", S.obj({"seed": S.s_int(minimum=0), "i": S.s_int(minimum=0), "fail": S.s_bool()},
                              ["seed", "i"]), doc="a sub-operation that changes three modules and then fails")
    def ep_doomed(self, ctx, seed, i, fail=True):
        self._op(ctx, "batch", seed, 1000 + i, None)
        self._op(ctx, "put", seed, 1000 + i, None)
        self._op(ctx, "asset", seed, 1000 + i, None)
        if fail:
            raise InvalidAction("doomed sub-operation")
        return {"done": True}


def _fingerprint(w) -> str:
    h = hashlib.blake2b(digest_size=16)
    for name, d in sorted(w.state.table_digests().items()):
        h.update(name.encode() + d.encode())
    for name in sorted(w.state.arrays):
        a = np.ascontiguousarray(w.state.arrays[name])
        h.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    from crucible.core.canonical import canonical_bytes
    h.update(canonical_bytes({k: v for k, v in w.state.meta.items() if k not in ("seq", "hash", "cmd_counter")}))
    return h.hexdigest()


def measure_integrity(n_scenarios: int = 240, seed: int = 23) -> dict:
    mods = with_deps(["identity", "bank", "admin", "contracts", "sp23_driver"])
    cfg = {"artifacts": {"quota_bytes": 4000, "fee_cents_per_mib": 50}}
    spec = WorldSpec("sp23-integrity", seed, [{"name": m, "config": cfg.get(m, {})} for m in mods],
                     {"typed_events": True})
    w = World.create(spec)                       # not strict: a handler bug is a recorded rejection, as in a run
    actors = [w.system_call("admin.join", {"handle": f"x{i}", "endowment": 5_000})["principal"] for i in range(4)]
    rng = np.random.default_rng(seed)
    bank = w.modules["bank"]
    stats = {"scenarios": 0, "failing": 0, "rollback_failures": 0, "controls": 0, "control_noops": 0,
             "contained": 0, "contained_failures": 0, "modules_touched_max": 0, "failed_as_intended": 0,
             "doomed_effective": 0}
    fails = []
    for s in range(n_scenarios):
        actor = actors[s % len(actors)]
        to = actors[(s + 1) % len(actors)]
        ops = [OPS[int(k)] for k in rng.integers(0, len(OPS), int(rng.integers(2, 8)))]
        fail = FAILS[int(rng.integers(0, len(FAILS)))]
        before = _fingerprint(w)
        n_art = w.state.count("artifacts")
        n_idx = w.state.count("sp23_index")
        seq0 = w.seq
        r = w.execute(actor, {"tool": "sp23.compose", "args": {"ops": ops, "fail": fail, "seed": s, "to": to}})
        stats["scenarios"] += 1
        inv = w.check_invariants() + bank.check_ledger_consistency(w)
        total = sum(a["balance"] for a in w.state.tables["accounts"].values())
        if fail == "none" or (fail == "quota" and r.ok):
            stats["controls"] += 1
            if not r.ok:
                # a control can be refused for a real reason (an actor that ran out of money or quota): it must
                # then leave no trace either
                if _fingerprint(w) != before:
                    stats["rollback_failures"] += 1
                    fails.append((s, ops, fail, "refused control left state"))
            elif _fingerprint(w) == before:
                stats["control_noops"] += 1
            else:
                touched = {e.type.split(".")[0] for e in w.log[seq0 - w.log_base:]}
                stats["modules_touched_max"] = max(stats["modules_touched_max"], len(touched))
                if "contained" in ops:
                    stats["contained"] += 1
        else:
            stats["failing"] += 1
            if not r.ok and r.error["code"] == EXPECTED_CODE[fail]:
                stats["failed_as_intended"] += 1
            ok = (not r.ok and _fingerprint(w) == before and w.state.count("artifacts") == n_art
                  and w.state.count("sp23_index") == n_idx
                  and [e.type for e in w.log[seq0 - w.log_base:]] == ["cmd.rejected"])
            if not ok:
                stats["rollback_failures"] += 1
                fails.append((s, ops, fail, r.error))
        if inv or total != 0:
            stats["rollback_failures"] += 1
            fails.append((s, ops, fail, inv[:3], total))
        if s % 40 == 39:
            w.advance_time(w.now + 86400)
            for a in actors:
                if bank.balance(w.state, bank.primary_account(w.state, a)) < 500:
                    w.system_call("admin.endow", {"principal": a, "amount": 3_000})
    # contained failures: a composed command with a failing sub-call equals the same command without it
    for s in range(20):
        ops = ["batch", "put", "pay", "attention", "asset"]
        a = w.fork(f"sp23-c{s}-a")
        b = w.fork(f"sp23-c{s}-b")
        ra = a.execute(actors[0], {"tool": "sp23.compose", "args": {"ops": ops[:2] + ["contained"] + ops[2:],
                                                                    "fail": "none", "seed": 10_000 + s,
                                                                    "to": actors[1]}})
        rb = b.execute(actors[0], {"tool": "sp23.compose", "args": {"ops": ops, "fail": "none", "seed": 10_000 + s,
                                                                    "to": actors[1]}})
        stats["contained"] += 1
        # the doomed sub-operation is effective when it does not fail (three modules change)
        c = w.fork(f"sp23-c{s}-c")
        fc = _fingerprint(c)
        def counts(x):
            return (len(x.state.find_ids("artifacts", "owner", actors[0])), x.state.count("assets"),
                    x.state.count("sp23_index"), int(C.col(x, "pop", "seen").sum()))
        n3 = counts(c)
        c.system_call("sp23.doomed", {"seed": 10_000 + s, "i": 2, "fail": False}, as_principal=actors[0])
        n4 = counts(c)
        if _fingerprint(c) != fc and n4[:3] == tuple(x + 1 for x in n3[:3]) and n4[3] == n3[3] + 50:
            stats["doomed_effective"] += 1
        # the ops after the contained call use index i+1 in `a`; compare the state effects that do not depend on
        # the op index: no artifact, asset, batch or index entry of the doomed sub-call survives
        doomed_ref = "sha256:" + hashlib.sha256(Sp23Driver._data(None, 10_000 + s, 1000 + 2)).hexdigest()
        assets_a = sorted(x["title"] for x in a.state.tables["assets"].values())
        if not (ra.ok and rb.ok) or A.exists(a, doomed_ref) or any("-1002" in t for t in assets_a) or \
                a.state.count("artifacts") != b.state.count("artifacts") or \
                a.state.count("assets") != b.state.count("assets") or \
                a.state.count("sp23_index") != b.state.count("sp23_index") or \
                np.array_equal(C.col(a, "pop", "seen"), C.col(b, "pop", "seen")) is False:
            stats["contained_failures"] += 1
            fails.append(("contained", s, ra.error, rb.error))
    evs = [e.to_dict() for e in w.log]
    replay_exact = World.replay(w.spec, evs).digest() == w.digest()
    g = World.regenerate(w.spec, evs)
    regen_exact = g.last_hash == w.last_hash
    return {"stats": stats, "fails": fails, "replay_exact": replay_exact, "regeneration_exact": regen_exact,
            "events": w.seq, "modules": len(w.modules)}


# ---------------------------------------------------------------------------------------------- 3b. nondeterministic policy
CALLS = {"n": 0}


@register_module
class Sp23Policy(Module):
    """Individuals choose how long to browse with a shared small neural policy whose sampling is nondeterministic;
    the decision is recorded (crucible.substrate.inference) and drives real attention spending (one cols.batch)."""
    name = "sp23_policy"
    host = "sp23p"
    depends = ("cols",)

    def __init__(self, config=None):
        super().__init__(config)
        import torch
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(5)
        self.net = torch.nn.Sequential(torch.nn.Linear(3, 32), torch.nn.ReLU(), torch.nn.Linear(32, 5))
        with torch.no_grad():
            for p in self.net.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.4)
        self.whash = I.weights_hash(self.net)

    def install(self, world):
        C.declare(world, "pop", 2000, {"mood": ("float32", (), 0.0)})
        AT.declare(world, 2000)

    def genesis(self, ctx):
        AT.reset(ctx, budget=np.full(2000, 120, np.float32))
        ctx.schedule(3600, "sp23_policy.session", {}, period=3600, key="sp23_policy.session")
        ctx.schedule(86400, "sp23_policy.day", {}, period=86400, key="sp23_policy.day")

    def process_session(self, ctx):
        import torch
        rng = ctx.rng("cohort")
        idx = np.sort(rng.choice(2000, size=256, replace=False))
        left = C.col(ctx.world, "pop", "attention_left_min")[idx]
        feats = np.stack([left / 120.0, C.col(ctx.world, "pop", "mood")[idx], np.full(256, (ctx.now // 3600) % 24 / 24)],
                         axis=1).astype(np.float32)

        def run():
            CALLS["n"] += 1
            with torch.no_grad():
                p = torch.softmax(self.net(torch.from_numpy(feats)), -1)
            gen = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "little"))
            return torch.multinomial(p, 1, generator=gen).squeeze(1).numpy().astype(np.int64)
        choice = I.recorded(ctx, "browse_policy", "sp23-browse-mlp", self.whash, {"features": feats}, run,
                            sampling={"method": "multinomial", "temperature": 1.0})
        minutes = choice.astype(np.float64) * 10.0
        AT.spend(ctx, idx, minutes, activity="browse")
        C.batch(ctx, "pop", "sp23.mood", idx, {"mood": (choice.astype(np.float32) - 2) / 10}, mode="add")


from crucible.core.kernel import process as _process  # noqa: E402

Sp23Policy.p_session = _process("session")(lambda self, ctx: self.process_session(ctx))
Sp23Policy.p_day = _process("day")(lambda self, ctx: AT.reset(ctx))


def measure_inference(days: int = 2) -> dict:
    spec = WorldSpec("sp23-infer", 4, [{"name": "artifacts", "config": {}}, {"name": "cols", "config": {}},
                                       {"name": "sp23_policy", "config": {}}])
    CALLS["n"] = 0
    w = World.create(spec, strict=True)
    w.advance_time(days * 86400)
    live_calls = CALLS["n"]
    evs = [e.to_dict() for e in w.log]
    recs = [r for r in I.records(evs) if r["kind"] == "browse_policy"]
    g = World.regenerate(spec, evs)
    regen_calls = CALLS["n"] - live_calls
    r = World.replay(spec, evs)
    replay_calls = CALLS["n"] - live_calls - regen_calls
    # rerunning the policy on recorded inputs gives different decisions: replay works only because of the record
    m = w.modules["sp23_policy"]
    import torch
    feats = np.random.default_rng(1).random((256, 3)).astype(np.float32)
    outs = set()
    for _ in range(4):
        with torch.no_grad():
            p = torch.softmax(m.net(torch.from_numpy(feats)), -1)
        gen = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "little"))
        outs.add(torch.multinomial(p, 1, generator=gen).squeeze(1).numpy().tobytes())
    ok = (live_calls == len(recs) and live_calls >= 24 * days - 1 and regen_calls == 0 and replay_calls == 0
          and g.last_hash == w.last_hash and r.digest() == w.digest() and len(outs) > 1
          and all(x["model_hash"] == m.whash for x in recs) and AT.check(w) == [])
    return {"inference_replay_without_rerun": bool(ok), "inference_records": len(recs), "inference_live_calls": live_calls,
            "inference_calls_during_regeneration": regen_calls, "inference_calls_during_replay": replay_calls,
            "inference_regeneration_exact": g.last_hash == w.last_hash, "inference_replay_exact": r.digest() == w.digest(),
            "inference_rerun_differs": len(outs) > 1}


# ---------------------------------------------------------------------------------------------- 3a. across processes
def _child(*args, timeout=1200) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONPATH=os.path.join(os.path.dirname(TESTS), "src"))
    p = subprocess.run([sys.executable, os.path.join(TESTS, "v5_resume_child.py"), *map(str, args)],
                       capture_output=True, text=True, timeout=timeout, env=env)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads([ln for ln in p.stdout.splitlines() if ln.startswith("{")][-1])


def measure_inference_across_processes(tmp) -> dict:
    from crucible.population.worlds import population_world
    spec = population_world("development", 3, n=300, sites=8)
    sp = os.path.join(tmp, "spec.json")
    with open(sp, "w") as fh:
        json.dump(spec.to_dict(), fh)
    db = os.path.join(tmp, "sp23.sqlite")
    part = _child("part", sp, db, 1.0, 1.5)              # writer process: dies without closing anything
    rep = _child("replay", sp, db)                       # new process: replay from genesis, store only
    reg = _child("regen", sp, db)                        # new process: regeneration, recorded outcomes injected
    res = _child("resume", sp, db, 2.0)                  # new process: snapshot + replay, then live again
    ref = _child("ref", sp, 2.0)                         # uninterrupted reference
    exact_replay = (rep["seq"], rep["hash"], rep["digest"]) == (part["seq"], part["hash"], part["digest"])
    exact_regen = (reg["seq"], reg["hash"], reg["digest"]) == (part["seq"], part["hash"], part["digest"])
    exact_resume = (res["hash"], res["digest"]) == (ref["hash"], ref["digest"])
    ok = (part["inference_runs"] > 0 and rep["inference_runs"] == 0 and reg["inference_runs"] == 0 and exact_replay
          and exact_regen and exact_resume and "pop_adaptive" in part["modules"] and "pop_strategic" in part["modules"])
    return {"ok": bool(ok), "population_inferences_recorded": part["inference_runs"],
            "population_inference_runs_during_replay": rep["inference_runs"],
            "population_inference_runs_during_regeneration": reg["inference_runs"],
            "population_cross_process_replay_exact": exact_replay,
            "population_cross_process_regeneration_exact": exact_regen,
            "population_cross_process_resume_exact": exact_resume,
            "population_resume_inference_runs": res["inference_runs"], "population_events": part["seq"],
            # skeptic S2: rebuilding neural weights on load re-runs the training steps logged after the last
            # checkpoint (deterministic on one machine, ISA-dependent across machines) - reported, not hidden
            "population_training_backward_passes_live": part["backward_passes"],
            "population_training_backward_passes_during_replay": rep["backward_passes"],
            "population_training_backward_passes_during_regeneration": reg["backward_passes"]}


@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_SP_2_3_shared_world_substrate(tmp_path):
    prim = measure_primitives()
    audit = prim.pop("_audit")
    fault = measure_fault_injection()
    integ = measure_integrity()
    xproc = measure_inference_across_processes(str(tmp_path))
    infer = measure_inference()
    st, fs = integ["stats"], fault["stats"]
    metrics = dict(prim)
    metrics.update({
        "cross_module_rollback_failures": st["rollback_failures"] + st["contained_failures"] + fs["failures"],
        "fault_injection_targets": fs["targets"], "fault_injection_job_targets": fs["job_targets"],
        "fault_injection_command_targets": fs["command_targets"], "fault_injection_failures": fs["failures"],
        "fault_injection_max_modules_per_transaction": fs["max_modules"],
        "fault_injection_modules_touched": fs["modules_touched"], "fault_injection_world_modules": fs["world_modules"],
        "fault_injection_subscriber_targets": fs["subscriber_targets"],
        "fault_injection_v5_module_targets": fs["v5_module_targets"],
        "integrity_scenarios": st["scenarios"], "integrity_failing_scenarios": st["failing"],
        "integrity_controls": st["controls"], "integrity_control_noops": st["control_noops"],
        "integrity_contained_subcalls": st["contained"], "integrity_failed_as_intended": st["failed_as_intended"],
        "integrity_doomed_subcall_effective": st["doomed_effective"],
        "integrity_max_event_hosts_per_command": st["modules_touched_max"],
        "integrity_world_modules": integ["modules"], "integrity_events": integ["events"],
        "integrity_replay_exact": integ["replay_exact"], "integrity_regeneration_exact": integ["regeneration_exact"],
    })
    metrics.update({k: v for k, v in xproc.items() if k != "ok"})
    metrics.update({f"testpolicy_{k}" if k != "inference_replay_without_rerun" else "testpolicy_replay_without_rerun":
                    v for k, v in infer.items()})
    metrics["inference_replay_without_rerun"] = bool(xproc["ok"] and infer["inference_replay_without_rerun"])
    evidence.record("SP-2.3", metrics, command=CMD, notes=(
        "shared_primitive_kinds counts the MASTER_SPEC Part 2.3 primitives (19, checked against the spec text) whose "
        "kind has one real owner and NO copy anywhere in the mandatory scan of every shipped module (facts.audit); "
        "kinds with known duplicates (allow-list with migrations and expiry, substrate/facts.py) are listed and do "
        "not count, nor do the refining kinds outside the spec's list. Rollback: fault injection into real multi-module jobs and "
        "commands of a world with every shipped module (a fault after a random event; the fork must commit only the "
        "failure record and leave every module table and column unchanged), plus driver-composed failures of six "
        "real modules. Inference: pop_adaptive/pop_strategic replayed and regenerated in NEW processes from the "
        "SQLite store alone (inference executions counted at the recorded-inference boundary), cross-process resume "
        "exact; a test torch policy with OS-entropy sampling regenerated and replayed in process."))
    # structural facts that must hold whatever the criterion says (the checker compares the criterion itself)
    assert audit["problems"] == [] and audit["uninstallable"] == {}
    assert prim["shared_primitive_kinds_owned"] == len(P.kinds()) and prim["families_with_duplicate_facts"] == 0
    assert metrics["cross_module_rollback_failures"] == 0, (fault["fails"][:5], integ["fails"][:5])
    assert fs["targets"] >= 40 and fs["max_modules"] >= 4 and fs["v5_module_targets"] >= 10
    assert st["failing"] >= 100 and st["controls"] >= 20 and st["control_noops"] == 0
    assert st["failed_as_intended"] >= 0.8 * st["failing"] and st["doomed_effective"] == 20
    assert integ["replay_exact"] and integ["regeneration_exact"]
    assert xproc["ok"], xproc
    assert infer["inference_replay_without_rerun"], infer
