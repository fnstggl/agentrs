"""Deterministic unit tests of the world kernel (core/kernel.py, core/state.py) with identity/bank/admin.

A small test module (`_ktest`, host `kt`) provides endpoints that emit events and then fail, nested calls,
dense arrays, scheduled processes (also failing ones) and private/public/internal events.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from crucible.core import schema as S
from crucible.core.errors import Conflict, CrucibleError, InvalidAction, InvariantViolation, ProtocolError
from crucible.core.kernel import SYSTEM, Module, World, WorldSpec, applier, endpoint, process, register_module
from crucible.services import admin as _admin, bank as _bank, identity as _identity  # noqa: F401  (registers modules)
from crucible.storage.memory import MemoryStore


# ====================================================================== test module
@register_module
class KTest(Module):
    name = "_ktest"
    host = "kt"
    depends = ("identity", "bank")

    def install(self, world):
        world.state.declare_table("kt_items", index=("owner",))
        world.state.declare_table("kt_runs", index=("label",))

    # ---------------------------------------------------------------- plain effects
    @endpoint("add", S.obj({"label": S.s_str(maxLength=40)}, ["label"]))
    def ep_add(self, ctx, label):
        iid = ctx.state.new_id("kt")
        ctx.emit("kt.add", {"id": iid, "owner": ctx.actor, "label": label})
        return {"id": iid}

    @endpoint("add_then_fail", S.obj({"label": S.s_str(maxLength=40), "crash": S.s_bool()}, ["label"]))
    def ep_add_then_fail(self, ctx, label, crash=False):
        """Emits events (an item, a payment, a job, an array bump) and then fails."""
        iid = ctx.state.new_id("kt")
        ctx.emit("kt.add", {"id": iid, "owner": ctx.actor, "label": label})
        bank = ctx.world.modules["bank"]
        bank.move(ctx, bank.primary_account(ctx.state, ctx.actor), "acct_fees", 7, "fee", "fee before failure")
        ctx.schedule(ctx.now + 10, "_ktest.job", {"label": label})
        if "kt_pop" in ctx.state.arrays:
            ctx.emit("kt.bump", {"i": 0, "x": 5.0})
        if crash:
            raise RuntimeError("handler bug")
        raise InvalidAction("deliberate failure")

    # ---------------------------------------------------------------- nested calls
    @endpoint("outer", S.obj({"fail_inner": S.s_bool(), "catch": S.s_bool(), "second": S.s_bool()}))
    def ep_outer(self, ctx, fail_inner=False, catch=True, second=True):
        first = ctx.state.new_id("kt")
        ctx.emit("kt.add", {"id": first, "owner": ctx.actor, "label": "outer-1"})
        inner_error = None
        try:
            ctx.call("kt.inner", {"fail": fail_inner})
        except CrucibleError as e:
            if not catch:
                raise
            inner_error = e.code
        out = {"first": first, "inner_error": inner_error}
        if second:
            out["second"] = ctx.state.new_id("kt")  # re-allocates the counter value the failed callee had taken
            ctx.emit("kt.add", {"id": out["second"], "owner": ctx.actor, "label": "outer-2"})
        return out

    @endpoint("inner", S.obj({"fail": S.s_bool()}))
    def ep_inner(self, ctx, fail=False):
        iid = ctx.state.new_id("kt")
        ctx.emit("kt.add", {"id": iid, "owner": ctx.actor, "label": "inner"})
        ctx.schedule(ctx.now + 5, "_ktest.job", {"label": "from-inner"})
        if "kt_pop" in ctx.state.arrays:
            ctx.emit("kt.bump", {"i": 0, "x": 100.0})
        if fail:
            raise InvalidAction("inner failed")
        return {"id": iid}

    @endpoint("pay_or_note", S.obj({"to": S.s_str(maxLength=40), "amount": S.s_int(minimum=1)}, ["to", "amount"]))
    def ep_pay_or_note(self, ctx, to, amount):
        """Try a payment through the public bank tool; on failure record a note instead."""
        ctx.emit("kt.add", {"id": ctx.state.new_id("kt"), "owner": ctx.actor, "label": "before-pay"})
        src = ctx.world.modules["bank"].primary_account(ctx.state, ctx.actor)
        try:
            r = ctx.call("bank.transfer", {"from_account": src, "to": to, "amount": amount})
            paid = r["txn"]
        except CrucibleError as e:
            paid = None
            ctx.emit("kt.add", {"id": ctx.state.new_id("kt"), "owner": ctx.actor, "label": f"failed:{e.code}"})
        return {"paid": paid}

    @endpoint("call_admin", S.obj())
    def ep_call_admin(self, ctx):
        return ctx.call("admin.endow", {"principal": ctx.actor, "amount": 1_000_000})

    # ---------------------------------------------------------------- arrays
    @endpoint("arr_init", S.obj({"n": S.s_int(minimum=1, maximum=1000)}, ["n"]))
    def ep_arr_init(self, ctx, n):
        ctx.emit("kt.arr_init", {"n": n}, vis="internal")

    @endpoint("arr_bump", S.obj({"i": S.s_int(minimum=0), "x": S.s_num()}, ["i", "x"]))
    def ep_arr_bump(self, ctx, i, x):
        ctx.emit("kt.bump", {"i": i, "x": x}, vis="internal")

    @endpoint("arr_outer", S.obj())
    def ep_arr_outer(self, ctx):
        ctx.emit("kt.bump", {"i": 0, "x": 1.0}, vis="internal")
        try:
            ctx.call("kt.inner", {"fail": True})
        except CrucibleError:
            pass
        ctx.emit("kt.bump", {"i": 1, "x": 2.0}, vis="internal")

    # ---------------------------------------------------------------- scheduling
    @endpoint("schedule", S.obj({"at": S.s_int(), "prio": S.s_int(), "period": S.s_int(minimum=1),
                                 "label": S.s_str(maxLength=40), "fail": S.s_enum(["crucible", "crash"])},
                                ["at", "label"]))
    def ep_schedule(self, ctx, at, label, prio=50, period=None, fail=None):
        args = {"label": label} if fail is None else {"label": label, "fail": fail}
        return {"job": ctx.schedule(at, "_ktest.job", args, prio=prio, period=period)}

    @process("job")
    def pr_job(self, ctx, label, fail=None):
        ctx.emit("kt.ran", {"label": label}, vis="internal")
        if fail == "crucible":
            raise InvalidAction(f"job {label} failed")
        if fail == "crash":
            raise RuntimeError(f"job {label} crashed")

    # ---------------------------------------------------------------- visibility
    @endpoint("secret", S.obj({"text": S.s_str(maxLength=100)}, ["text"]))
    def ep_secret(self, ctx, text):
        ctx.emit("kt.say", {"text": text})  # default visibility: the acting principal only

    @endpoint("tell", S.obj({"to": S.s_str(maxLength=40), "text": S.s_str(maxLength=100)}, ["to", "text"]))
    def ep_tell(self, ctx, to, text):
        ctx.emit("kt.say", {"text": text}, vis=[ctx.actor, to])

    @endpoint("shout", S.obj({"text": S.s_str(maxLength=100)}, ["text"]))
    def ep_shout(self, ctx, text):
        ctx.emit("kt.say", {"text": text}, vis="public")

    @endpoint("audit", S.obj({"text": S.s_str(maxLength=100)}, ["text"]))
    def ep_audit(self, ctx, text):
        ctx.emit("kt.say", {"text": text}, vis="internal")

    # ---------------------------------------------------------------- appliers
    @applier("kt.add")
    def ap_add(self, ev):
        p = ev.payload
        self.world.state.insert("kt_items", p["id"], {"owner": p["owner"], "label": p["label"], "t": ev.t})

    @applier("kt.ran")
    def ap_ran(self, ev):
        self.world.state.insert("kt_runs", f"{ev.seq:08d}", {"label": ev.payload["label"], "t": ev.t})

    @applier("kt.say")
    def ap_say(self, ev):
        return None

    @applier("kt.arr_init")
    def ap_arr_init(self, ev):
        self.world.state.set_array("kt_pop", np.zeros(ev.payload["n"], dtype=np.float64))

    @applier("kt.bump")
    def ap_bump(self, ev):
        self.world.state.array_mut("kt_pop")[ev.payload["i"]] += ev.payload["x"]


# ====================================================================== helpers
MODULES = [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}, {"name": "_ktest"}]
ALICE, BOB, CAROL = "p:alice", "p:bob", "p:carol"
A_ACCT, B_ACCT, C_ACCT = "acct_000001", "acct_000002", "acct_000003"


def make_world(params: dict | None = None, store=None, strict: bool = False, world_id: str = "w-kt") -> World:
    spec = WorldSpec("kernel-test", 11, MODULES, params=dict(params or {}))
    w = World.create(spec, world_id, store=store, strict=strict)
    assert w.system_call("admin.join", {"handle": "alice", "endowment": 10_000})["account"] == A_ACCT
    assert w.system_call("admin.join", {"handle": "bob", "endowment": 5_000})["account"] == B_ACCT
    assert w.system_call("admin.join", {"handle": "carol"})["account"] == C_ACCT
    return w


def act(w: World, actor: str, tool: str, key=None, **args):
    a = {"tool": tool, "args": args}
    if key is not None:
        a["idempotency_key"] = key
    return w.execute(actor, a)


def log_dicts(w: World) -> list[dict]:
    """Full event log (genesis included) as independent dicts."""
    return copy.deepcopy([e.to_dict() for e in w.events_range(0)])


def balance(w: World, acct: str) -> int:
    return w.state.tables["accounts"][acct]["balance"]


def assert_consistent(w: World) -> None:
    """Replay (appliers) and regeneration (handlers) reproduce the world exactly; invariants hold."""
    assert w.check_invariants() == []
    assert w.modules["bank"].check_ledger_consistency(w) == []
    evs = log_dicts(w)
    assert [e["seq"] for e in evs] == list(range(1, w.seq + 1))
    r = World.replay(w.spec, evs)
    assert (r.digest(), r.last_hash, r.seq) == (w.digest(), w.last_hash, w.seq)
    g = World.regenerate(w.spec, evs)
    assert (g.last_hash, g.seq) == (w.last_hash, w.seq)
    assert g.digest() == w.digest()


def labels(w: World, table: str = "kt_items") -> list[str]:
    return [r["label"] for _, r in w.state.scan(table)]


def meta_without_chain(w: World) -> dict:
    return {k: v for k, v in w.state.meta.items() if k not in ("seq", "hash", "cmd_counter")}


# ====================================================================== transactions
def test_failing_command_rolls_back_completely():
    w = make_world()
    act(w, ALICE, "kt.arr_init", n=4)
    tables0, meta0 = w.state.table_digests(), meta_without_chain(w)
    pop0 = w.state.arrays["kt_pop"].copy()
    seq0, bal0, heap0 = w.seq, balance(w, A_ACCT), len(w.state.tables["sched"])

    r = act(w, ALICE, "kt.add_then_fail", label="doomed")
    assert not r.ok and r.error["code"] == "invalid_action"
    # the only trace is the rejection record itself
    new = w.events_range(seq0)
    assert [e.type for e in new] == ["cmd.rejected"] and w.seq == seq0 + 1
    assert new[0].payload["error"]["message"] == "deliberate failure"
    assert w.state.table_digests() == tables0
    assert meta_without_chain(w) == meta0
    np.testing.assert_array_equal(w.state.arrays["kt_pop"], pop0)
    assert balance(w, A_ACCT) == bal0 and len(w.state.tables["sched"]) == heap0
    assert "doomed" not in labels(w)
    assert not w.state.in_tx
    assert_consistent(w)


def test_handler_crash_rolls_back_and_strict_mode_raises():
    w = make_world()
    seq0, tables0 = w.seq, w.state.table_digests()
    r = act(w, ALICE, "kt.add_then_fail", label="bug", crash=True)
    assert not r.ok and "internal error: RuntimeError" in r.error["message"]
    assert [e.type for e in w.events_range(seq0)] == ["cmd.rejected"]
    assert w.state.table_digests() == tables0

    ws = make_world(strict=True)
    d0, seq0, h0 = ws.digest(), ws.seq, ws.last_hash
    with pytest.raises(RuntimeError):
        act(ws, ALICE, "kt.add_then_fail", label="bug", crash=True)
    # strict mode: nothing at all is recorded, state digest and seq are unchanged
    assert (ws.digest(), ws.seq, ws.last_hash) == (d0, seq0, h0)
    assert not ws.state.in_tx
    assert act(ws, ALICE, "kt.add", label="after").ok


def test_failing_system_call_leaves_no_trace():
    w = make_world()
    d0, seq0, h0, n_log = w.digest(), w.seq, w.last_hash, len(w.log)
    with pytest.raises(Conflict):
        w.system_call("admin.join", {"handle": "alice", "endowment": 99})  # handle taken
    with pytest.raises(InvalidAction):
        w.system_call("kt.add_then_fail", {"label": "x"}, as_principal=ALICE)
    assert (w.digest(), w.seq, w.last_hash, len(w.log)) == (d0, seq0, h0, n_log)
    assert not w.state.in_tx
    assert_consistent(w)


def test_nested_call_failure_caught_keeps_caller_events_and_drops_callee():
    w = make_world()
    seq0 = w.seq
    r = act(w, ALICE, "kt.outer", fail_inner=True)
    assert r.ok and r.result["inner_error"] == "invalid_action"
    evs = w.events_range(seq0)
    types = [e.type for e in evs]
    assert types.count("kt.add") == 2 and "sched.add" not in types
    assert [e.seq for e in evs] == list(range(seq0 + 1, w.seq + 1))  # no hole where the callee's events were
    assert labels(w) == ["outer-1", "outer-2"]
    # the counter value taken by the failed callee was reused by the caller and recorded (ctr) for replay
    assert r.result == {"first": "kt_000001", "second": "kt_000002", "inner_error": "invalid_action"}
    assert w.state.meta["ctr:id:kt"] == 2
    assert w.state.tables["sched"] == {}
    assert_consistent(w)

    r = act(w, ALICE, "kt.outer", fail_inner=False)
    assert r.ok and labels(w) == ["outer-1", "outer-2", "outer-1", "inner", "outer-2"]
    assert len(w.state.tables["sched"]) == 1
    assert_consistent(w)


def test_nested_call_failure_uncaught_rejects_whole_command():
    w = make_world()
    tables0 = w.state.table_digests()
    r = act(w, ALICE, "kt.outer", fail_inner=True, catch=False)
    assert not r.ok and r.error["message"] == "inner failed"
    assert w.state.table_digests() == tables0
    assert_consistent(w)


def test_nested_bank_call_failure_with_real_service():
    w = make_world()
    seq0 = w.seq
    r = act(w, CAROL, "kt.pay_or_note", to="alice", amount=50)  # carol has no money
    assert r.ok and r.result == {"paid": None}
    assert labels(w) == ["before-pay", "failed:insufficient_funds"]
    assert balance(w, C_ACCT) == 0 and balance(w, A_ACCT) == 10_000
    assert "bank.txn" not in [e.type for e in w.events_range(seq0)]
    r = act(w, ALICE, "kt.pay_or_note", to="carol", amount=50)
    assert r.ok and r.result["paid"] and balance(w, C_ACCT) == 50
    assert_consistent(w)


def test_regression_nested_failure_restores_arrays_saved_before_savepoint():
    """Before the fix, an array first written by the caller and then by a failing callee kept the callee's
    in-place change (the savepoint journal only saved each array once per transaction)."""
    w = make_world()
    act(w, ALICE, "kt.arr_init", n=3)
    assert act(w, ALICE, "kt.arr_outer").ok
    np.testing.assert_array_equal(w.state.arrays["kt_pop"], [1.0, 2.0, 0.0])
    # a failing top-level command restores the array, also after an inner savepoint
    assert not act(w, ALICE, "kt.add_then_fail", label="x").ok
    np.testing.assert_array_equal(w.state.arrays["kt_pop"], [1.0, 2.0, 0.0])
    assert_consistent(w)


def test_regression_counter_reallocated_after_failed_nested_call_is_replayed():
    """Before the fix the `ctr` event omitted a counter whose last allocation was rolled back, so replay produced
    a different meta counter (and a reloaded world would hand out a duplicate id)."""
    w = make_world()
    r = act(w, ALICE, "kt.outer", fail_inner=True, second=False)   # caller allocates kt_000001 only
    assert r.ok and r.result == {"first": "kt_000001", "inner_error": "invalid_action"}
    assert w.state.meta["ctr:id:kt"] == 1
    ctr = [e for e in w.events_range(r.events[0] - 1) if e.type == "ctr"]
    assert len(ctr) == 1 and ctr[0].payload.get("ctr:id:kt") == 1
    rp = World.replay(w.spec, log_dicts(w))
    assert rp.state.meta["ctr:id:kt"] == 1 and rp.digest() == w.digest()
    # the replayed world continues with fresh ids instead of colliding with kt_000001
    assert act(rp, ALICE, "kt.add", label="next").result == {"id": "kt_000002"}
    assert_consistent(w)


# ====================================================================== idempotency, rate limits, admin
def test_idempotency_keys():
    w = make_world()
    args = {"from_account": A_ACCT, "to": "bob", "amount": 300}
    r1 = act(w, ALICE, "bank.transfer", key="pay-1", **args)
    assert r1.ok and not r1.replayed
    seq1 = w.seq
    r2 = act(w, ALICE, "bank.transfer", key="pay-1", **args)
    assert r2.ok and r2.replayed and r2.result == r1.result and r2.events == []
    assert w.seq == seq1 and balance(w, A_ACCT) == 9_700 and balance(w, B_ACCT) == 5_300
    # the key wins over different arguments (the first outcome is the answer)
    r3 = act(w, ALICE, "bank.transfer", key="pay-1", **dict(args, amount=1))
    assert r3.replayed and r3.result == r1.result and balance(w, A_ACCT) == 9_700
    # keys are per actor
    r4 = act(w, BOB, "bank.transfer", key="pay-1", from_account=B_ACCT, to="alice", amount=5)
    assert r4.ok and not r4.replayed
    # a failure under a key is remembered too
    f1 = act(w, CAROL, "bank.transfer", key="pay-2", from_account=C_ACCT, to="alice", amount=10)
    assert not f1.ok and f1.error["code"] == "insufficient_funds"
    seq2 = w.seq
    f2 = act(w, CAROL, "bank.transfer", key="pay-2", from_account=C_ACCT, to="alice", amount=10)
    assert not f2.ok and f2.replayed and f2.error == f1.error and w.seq == seq2
    # a request rejected before reaching the handler (schema error) does not consume the key
    bad = act(w, ALICE, "bank.transfer", key="pay-3", from_account=A_ACCT, to="bob", amount=-1)
    assert not bad.ok and bad.error["code"] == "invalid_action"
    good = act(w, ALICE, "bank.transfer", key="pay-3", from_account=A_ACCT, to="bob", amount=1)
    assert good.ok and not good.replayed
    # falsy keys are keys
    e1 = act(w, ALICE, "bank.transfer", key="", from_account=A_ACCT, to="bob", amount=2)
    e2 = act(w, ALICE, "bank.transfer", key="", from_account=A_ACCT, to="bob", amount=2)
    assert e1.ok and e2.replayed
    assert_consistent(w)


def test_regression_regenerate_rejected_command_with_idempotency_key():
    """Before the fix regeneration re-ran a rejected command without its key, so the `idem` record was missing
    from the regenerated log (different hash)."""
    w = make_world()
    assert not act(w, CAROL, "bank.transfer", key="k", from_account=C_ACCT, to="bob", amount=1).ok
    assert not act(w, ALICE, "kt.add_then_fail", key=0, label="x").ok   # falsy, non-string key
    assert act(w, ALICE, "kt.add", key=0, label="y").replayed is True
    g = World.regenerate(w.spec, log_dicts(w))
    assert g.last_hash == w.last_hash and g.digest() == w.digest()
    assert g.state.has("idem", "p:carol|k") and g.state.has("idem", "p:alice|0")


def test_rate_limit_max_actions_per_hour():
    w = make_world(params={"max_actions_per_hour": 3})
    for i in range(3):
        assert act(w, ALICE, "kt.add", label=f"a{i}").ok
    r = act(w, ALICE, "kt.add", label="a3")
    assert not r.ok and r.error["code"] == "rate_limited"
    assert "a3" not in labels(w)
    assert act(w, BOB, "kt.add", label="b0").ok          # per actor
    assert not act(w, ALICE, "world.clock").ok            # read-only tools count as actions too
    w.advance_time(3600)                                  # next simulated hour
    assert act(w, ALICE, "kt.add", label="a4").ok
    assert w.principal(ALICE)["hour_actions"] == 1
    assert_consistent(w)


def test_admin_tools_rejected_for_actors():
    w = make_world()
    admin_tools = sorted(n for n in w.endpoints if n.startswith("admin."))
    assert admin_tools == ["admin.broadcast", "admin.endow", "admin.join", "admin.set_status"]
    tables0 = w.state.table_digests()
    calls = {"admin.join": {"handle": "mallory", "endowment": 10**9},
             "admin.endow": {"principal": ALICE, "amount": 10**6},
             "admin.set_status": {"principal": BOB, "status": "suspended"},
             "admin.broadcast": {"text": "free money"}}
    for tool, args in calls.items():
        r = act(w, ALICE, tool, **args)
        assert not r.ok and r.error["code"] == "forbidden", tool
    # also through a nested call made by an in-world actor
    r = act(w, ALICE, "kt.call_admin")
    assert not r.ok and r.error["code"] == "forbidden"
    assert w.state.table_digests() == tables0 and balance(w, A_ACCT) == 10_000
    # the operator can
    assert w.system_call("admin.endow", {"principal": ALICE, "amount": 5})["txn"]
    assert balance(w, A_ACCT) == 10_005
    # and a suspended actor cannot act at all
    w.system_call("admin.set_status", {"principal": BOB, "status": "suspended"})
    assert act(w, BOB, "kt.add", label="x").error["code"] == "unauthenticated"
    assert_consistent(w)


# ====================================================================== time & scheduler
def test_time_cannot_go_backwards():
    w = make_world()
    assert w.advance_time(100) == 0 and w.now == 100
    seq0, d0 = w.seq, w.digest()
    with pytest.raises(ProtocolError):
        w.advance_time(99)
    assert (w.now, w.seq, w.digest()) == (100, seq0, d0)
    w.advance_time(100)                       # same time: no-op, no event
    assert w.seq == seq0
    # jobs requested in the past are clamped to now
    job = act(w, ALICE, "kt.schedule", at=10, label="late").result["job"]
    assert w.state.tables["sched"][job]["at"] == 100
    assert w.advance_time(100) == 1 and w.now == 100
    assert [r["t"] for _, r in w.state.scan("kt_runs")] == [100]
    assert_consistent(w)


def test_scheduler_runs_in_time_priority_sequence_order():
    w = make_world()
    plan = [(50, 50, "a"), (20, 90, "b"), (20, 10, "c"), (20, 10, "d"), (50, 1, "e"), (70, 50, "f"), (500, 0, "late")]
    for at, prio, label in plan:
        assert act(w, ALICE, "kt.schedule", at=at, prio=prio, label=label).ok
    assert w.next_job_time() == 20
    assert w.advance_time(100) == 6
    runs = [(r["t"], r["label"]) for _, r in w.state.scan("kt_runs")]
    assert runs == [(20, "c"), (20, "d"), (20, "b"), (50, "e"), (50, "a"), (70, "f")]
    procs = [e for e in w.events_range(0) if e.type == "proc"]
    assert [e.t for e in procs] == [20, 20, 20, 50, 50, 70]
    assert w.now == 100 and w.next_job_time() == 500
    assert w.check_invariants() == []
    assert_consistent(w)


def test_periodic_jobs_and_advance_exactly_to_job_time():
    w = make_world()
    job = act(w, ALICE, "kt.schedule", at=100, period=50, label="tick").result["job"]
    assert w.advance_time(260) == 4
    assert [r["t"] for _, r in w.state.scan("kt_runs")] == [100, 150, 200, 250]
    assert w.state.tables["sched"][job]["at"] == 300 and w.now == 260
    # landing exactly on a job time runs it (regression: regenerate used to miss these runs)
    assert w.advance_time(300) == 1 and w.now == 300
    assert w.advance_time(300) == 0
    assert_consistent(w)


def test_regression_regenerate_with_max_jobs():
    w = make_world()
    act(w, ALICE, "kt.schedule", at=100, label="j1")
    act(w, ALICE, "kt.schedule", at=100, label="j2")
    assert w.advance_time(200, max_jobs=1) == 1 and w.now == 100
    assert act(w, ALICE, "kt.add", label="between").ok
    assert w.state.tables["kt_items"]["kt_000001"]["t"] == 100
    assert w.advance_time(200) == 1 and w.now == 200
    assert_consistent(w)


def test_process_failures_are_recorded_and_periodic_jobs_rescheduled():
    w = make_world()
    once = act(w, ALICE, "kt.schedule", at=10, label="once", fail="crucible").result["job"]
    per = act(w, ALICE, "kt.schedule", at=20, period=30, label="per", fail="crash").result["job"]
    ok = act(w, ALICE, "kt.schedule", at=25, period=30, label="ok").result["job"]
    assert w.advance_time(90) == 7   # once@10, per@20, ok@25, per@50, ok@55, per@80, ok@85
    failed = [e for e in w.events_range(0) if e.type == "proc.failed"]
    assert [(e.payload["id"], e.t) for e in failed] == [(once, 10), (per, 20), (per, 50), (per, 80)]
    assert failed[0].payload["error"] == "InvalidAction: job once failed"
    assert failed[1].payload["error"] == "RuntimeError: job per crashed"
    # the failing runs' own effects were rolled back; the healthy job ran normally
    assert labels(w, "kt_runs") == ["ok", "ok", "ok"]
    assert once not in w.state.tables["sched"]
    assert w.state.tables["sched"][per]["at"] == 110 and w.state.tables["sched"][ok]["at"] == 115
    assert w.check_invariants() == []
    assert_consistent(w)


def test_regression_strict_process_crash_keeps_job_scheduled():
    w = make_world(strict=True)
    job = act(w, ALICE, "kt.schedule", at=10, label="boom", fail="crash").result["job"]
    d0, seq0 = w.digest(), w.seq
    with pytest.raises(RuntimeError):
        w.advance_time(20)
    assert (w.digest(), w.seq, w.now) == (d0, seq0, 0)
    assert job in w.state.tables["sched"]
    assert w.next_job_time() == 10          # still in the heap: it is retried, not silently lost
    with pytest.raises(RuntimeError):
        w.advance_time(20)


# ====================================================================== visibility
def test_actors_never_see_other_actors_private_events():
    w = make_world()
    act(w, ALICE, "kt.secret", text="alice-secret")
    act(w, BOB, "kt.secret", text="bob-secret")
    act(w, ALICE, "kt.tell", to=BOB, text="alice-to-bob")
    act(w, ALICE, "kt.shout", text="hello-all")
    act(w, ALICE, "kt.audit", text="audit-only")
    act(w, ALICE, "bank.transfer", from_account=A_ACCT, to="bob", amount=10, memo="a->b")
    act(w, CAROL, "bank.transfer", from_account=C_ACCT, to="bob", amount=10**6)      # carol's rejection
    act(w, ALICE, "id.create_org", handle="acme", name="Acme")
    inv = act(w, ALICE, "id.invite", org="acme", person="bob").result["invite"]
    act(w, BOB, "id.respond_invite", invite=inv, accept=True)

    def seen(actor):
        evs, dropped = w.visible_events(actor, 0, limit=10_000)
        assert dropped == 0
        ps = w.principals_of(actor)
        for e in evs:
            assert e.vis != "internal"
            assert e.vis == "public" or set(e.vis) & ps, (actor, e)
        return evs

    def texts(evs):
        return {e.payload["text"] for e in evs if e.type == "kt.say"}

    sa, sb, sc = seen(ALICE), seen(BOB), seen(CAROL)
    assert texts(sa) == {"alice-secret", "alice-to-bob", "hello-all"}
    assert texts(sb) == {"bob-secret", "alice-to-bob", "hello-all"}
    assert texts(sc) == {"hello-all"}
    memo = lambda evs: [e for e in evs if e.type == "bank.txn" and e.payload["memo"] == "a->b"]  # noqa: E731
    assert len(memo(sa)) == len(memo(sb)) == 1 and memo(sc) == []
    # commands (with their arguments) and rejections are private to the actor
    assert all(e.actor == ALICE for e in sa if e.type == "cmd")
    assert all(e.actor == BOB for e in sb if e.type == "cmd")
    assert all(e.payload["actor"] == CAROL for e in sc if e.type == "cmd.rejected")
    assert not any(e.type == "cmd.rejected" for e in sa + sb)
    # alice administers acme and therefore sees its private events; carol sees none of them
    acme = [e for e in w.events_range(0) if isinstance(e.vis, tuple) and "o:acme" in e.vis]
    assert acme and all(e in sa for e in acme)
    assert not any(e in sc for e in acme)
    # limit keeps the most recent events and reports how many were dropped
    last2, dropped = w.visible_events(ALICE, 0, limit=2)
    assert last2 == sa[-2:] and dropped == len(sa) - 2


# ====================================================================== integrity, snapshots, forks
def _populated_world(**kw) -> World:
    w = make_world(**kw)
    act(w, ALICE, "kt.arr_init", n=5)
    act(w, ALICE, "bank.transfer", from_account=A_ACCT, to="bob", amount=1234)
    act(w, BOB, "kt.add", label="b1")
    act(w, ALICE, "kt.schedule", at=40, period=40, label="tick")
    w.advance_time(100)
    act(w, ALICE, "kt.arr_bump", i=2, x=2.5)
    return w


def test_hash_chain_tamper_detection_in_replay():
    w = _populated_world()
    evs = log_dicts(w)
    assert World.replay(w.spec, evs).digest() == w.digest()
    i = next(k for k, e in enumerate(evs) if e["type"] == "bank.txn" and e["payload"]["kind"] == "transfer")

    tampered = copy.deepcopy(evs)
    tampered[i]["payload"]["postings"][0][1] -= 1000     # steal 10 dollars
    with pytest.raises(InvariantViolation, match="hash chain broken"):
        World.replay(w.spec, tampered)
    # without verification the edit goes through and the state differs (this is what the chain guards)
    assert World.replay(w.spec, tampered, verify=False).digest() != w.digest()

    for mutate in (lambda l: l[i].__setitem__("t", l[i]["t"] + 1),
                   lambda l: l[i].__setitem__("vis", "public"),
                   lambda l: l[-1].__setitem__("hash", "f" * 32),
                   lambda l: l.__setitem__(slice(i, i + 2), [l[i + 1], l[i]])):
        bad = copy.deepcopy(evs)
        mutate(bad)
        with pytest.raises(InvariantViolation):
            World.replay(w.spec, bad)
    with pytest.raises(InvariantViolation, match="gap"):
        World.replay(w.spec, evs[:i] + evs[i + 1:])


def test_tampered_store_is_detected_on_load():
    store = MemoryStore()
    w = _populated_world(store=store)
    w.flush()
    assert World.load(store, w.world_id).digest() == w.digest()
    store.events[w.world_id][20]["payload"]["tampered"] = True
    with pytest.raises(InvariantViolation, match="hash chain broken"):
        World.load(store, w.world_id)


def _array_shares(a: World, b: World) -> bool:
    return any(np.shares_memory(a.state.arrays[k], b.state.arrays[k]) for k in a.state.arrays if k in b.state.arrays)


def test_snapshot_and_fork_isolation():
    w = _populated_world()
    d0, pop0 = w.digest(), w.state.arrays["kt_pop"].copy()
    f = w.fork("w-kt-fork")
    assert f.digest() == d0 and f.seq == w.seq and f.last_hash == w.last_hash and f.now == w.now
    assert f.state.tables["accounts"] is not w.state.tables["accounts"]
    assert not _array_shares(w, f)

    # fork diverges, parent untouched (tables, arrays, scheduler, clock)
    assert act(f, ALICE, "bank.transfer", from_account=A_ACCT, to="carol", amount=500).ok
    assert act(f, ALICE, "kt.arr_bump", i=0, x=9.0).ok
    f.state.array_mut("kt_pop")  # even a raw handle on the fork's array is not the parent's
    f.advance_time(200)
    assert w.digest() == d0 and balance(w, C_ACCT) == 0 and w.now == 100
    np.testing.assert_array_equal(w.state.arrays["kt_pop"], pop0)
    assert labels(w, "kt_runs") == ["tick", "tick"] and labels(f, "kt_runs") == ["tick"] * 5

    # parent diverges, fork untouched
    fd = f.digest()
    assert act(w, ALICE, "kt.arr_bump", i=1, x=1.0).ok
    assert act(w, BOB, "kt.add", label="parent-only").ok
    assert f.digest() == fd and "parent-only" not in labels(f)
    assert f.state.arrays["kt_pop"][1] == 0.0 and f.state.arrays["kt_pop"][0] == 9.0
    # forking again does not disturb either of them
    d1 = w.digest()
    g = w.fork("w-kt-fork-2")
    assert (w.digest(), f.digest(), g.digest()) == (d1, fd, d1)


def test_snapshot_blob_roundtrip_and_fork_replay():
    w = _populated_world()
    blob = w.snapshot_blob()
    seq0 = w.seq
    f = World.from_blob(blob, world_id="w-kt-f", fork=True)
    assert f.digest() == w.digest() and f.log_base == seq0
    act(f, ALICE, "kt.add", label="in-fork")
    f.advance_time(300)
    own = copy.deepcopy([e.to_dict() for e in f.events_range(seq0)])
    assert own and own[0]["seq"] == seq0 + 1
    r = World.replay(f.spec, own, base_blob=blob)
    assert (r.digest(), r.last_hash) == (f.digest(), f.last_hash)
    # the parent is unaffected by the blob being loaded and extended
    assert "in-fork" not in labels(w)


def test_fork_of_a_fork():
    store = MemoryStore()
    w = _populated_world(store=store)
    f1 = w.fork("f1", store=store)
    act(f1, ALICE, "kt.add", label="f1-only")
    f2 = f1.fork("f2", store=store)
    act(f2, BOB, "kt.add", label="f2-only")
    act(w, CAROL, "kt.add", label="root-only")
    assert "f1-only" in labels(f2) and "f2-only" not in labels(f1) and "root-only" not in labels(f2)
    assert len({w.digest(), f1.digest(), f2.digest()}) == 3
    for x in (w, f1, f2):
        x.flush()
    assert [i["id"] for i in store.lineage("f2")] == ["f2", "f1", w.world_id]
    assert store.world_info("f2")["parent"] == {"world": "f1", "seq": f1.seq, "hash": f1.last_hash}
    for x in (w, f1, f2):
        y = World.load(store, x.world_id)
        assert (y.digest(), y.last_hash, y.seq) == (x.digest(), x.last_hash, x.seq)
    assert_consistent(w)


# ====================================================================== tools & discovery
def test_available_tools_lists_builtins():
    w = make_world()
    tools = w.available_tools(ALICE)
    names = [t["name"] for t in tools]
    assert names == sorted(names)
    for n in ("world.whoami", "world.clock", "world.describe", "bank.transfer", "bank.accounts", "id.create_org",
              "id.grant", "id.invite", "kt.add"):
        assert n in names
    for t in tools:
        assert t["description"] is not None and t["inputSchema"]["type"] == "object"
        assert t["annotations"]["builtin"] is True and t["annotations"]["host"] == t["name"].split(".")[0]
    assert next(t for t in tools if t["name"] == "bank.balance")["annotations"]["readOnlyHint"] is True
    assert w.available_tools(BOB) == tools   # nothing discovered: same list for everyone
    r = act(w, ALICE, "world.whoami")
    assert r.ok and r.result == {"principal": ALICE, "kind": "person", "time": 0}


def test_describe_endpoint():
    w = make_world()
    seq0 = w.seq
    r = act(w, ALICE, "world.describe", host="bank")
    assert r.ok and r.result["host"] == "bank"
    eps = [e["name"] for e in r.result["endpoints"]]
    assert eps == sorted(eps) and "bank.transfer" in eps and all(n.startswith("bank.") for n in eps)
    know = [e for e in w.events_range(seq0) if e.type == "know"]
    assert len(know) == 1 and know[0].vis == (ALICE,)
    assert w.knows(ALICE, "host", "bank") and not w.knows(BOB, "host", "bank")
    seq1 = w.seq
    act(w, ALICE, "world.describe", host="bank")
    assert [e.type for e in w.events_range(seq1)] == ["cmd"]   # already known: no second `know`
    bad = act(w, ALICE, "world.describe", host="nowhere")
    assert not bad.ok and bad.error["code"] == "unknown_tool"
    assert act(w, ALICE, "no.such_tool").error["code"] == "unknown_tool"
    assert_consistent(w)


# ====================================================================== malformed input
def test_regression_malformed_actions_never_wedge_the_world():
    """Before the fix, NaN/Infinity arguments or unpaired surrogates made the rejection record unhashable: the
    exception escaped `execute` with the transaction still open, and every later command failed."""
    w = make_world()
    cases = [
        {"tool": "bank.transfer", "args": {"from_account": A_ACCT, "to": "bob", "amount": float("nan")}},
        {"tool": "kt.add", "args": {"label": float("inf")}},
        {"tool": "kt.add", "args": {"label": object()}},
        {"tool": "kt.add", "args": {"label": "lone \ud800 surrogate"}},
        {"tool": "kt.secret", "args": {"text": "nul \x00 byte"}},
        {"tool": "kt.\udcff", "args": {}},
        {"tool": 5, "args": {}},
        {"tool": None},
        {"tool": "kt.add", "args": ["label"]},
        "not-an-action",
    ]
    for a in cases:
        r = w.execute(ALICE, a)
        assert not w.state.in_tx
        assert isinstance(r.ok, bool)
    assert act(w, ALICE, "kt.add", label="still works").ok
    assert act(w, ALICE, "kt.add", label="lone \ud800 surrogate").ok
    assert w.execute("p:nobody", {"tool": "kt.add", "args": {"label": "x"}}).error["code"] == "unauthenticated"
    assert_consistent(w)
    store = MemoryStore()
    w2 = make_world(store=store, world_id="w-malformed")
    for a in cases:
        w2.execute(ALICE, a)
    w2.flush()
    assert World.load(store, "w-malformed").digest() == w2.digest()


def test_stats_and_whoami_for_system():
    w = make_world()
    assert w.principal(SYSTEM)["kind"] == "system"
    assert w.execute(SYSTEM, {"tool": "world.whoami", "args": {}}).error["code"] == "unauthenticated"


def test_regeneration_independent_of_argument_key_order():
    """Args read back from sorted JSON must regenerate the same log as the live (insertion-ordered) call."""
    import crucible
    crucible.load_all_modules()
    from crucible.core.kernel import World
    from crucible.worldgen import generate
    spec = generate("tiny", 4)
    w = World.create(spec)
    w.system_call("admin.join", {"handle": "zed", "endowment": 1000})
    w.execute("p:zed", {"tool": "contracts.propose", "args": {"title": "t", "text": "x", "parties": [], "clauses": []}})
    w.execute("p:zed", {"tool": "bank.transfer", "args": {"to": "nobody", "amount": -5, "from_account": "acct_x"}})
    import json
    evs = [json.loads(json.dumps(e.to_dict(), sort_keys=True)) for e in w.log]
    assert World.regenerate(spec, evs).last_hash == w.last_hash
