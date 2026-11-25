"""Crucible 5.0 substrate: columnar state with batch events (1.2), identity indices (1.3), attention ledger (1.6)."""
import numpy as np
import pytest

import crucible
from crucible.core import schema as S
from crucible.core.errors import InvalidAction
from crucible.core.kernel import Module, World, WorldSpec, endpoint, process, register_module
from crucible.storage.memory import MemoryStore
from crucible.storage.sqlite import SQLiteStore
from crucible.substrate import artifacts as A
from crucible.substrate import attention as AT
from crucible.substrate import columns as C
from crucible.substrate import ids
from crucible.worldgen.admission import admit

crucible.load_all_modules()


@register_module
class ColTestPop(Module):
    """A small real population: hourly cohort decisions recorded as batches, a daily attention reset."""
    name = "coltest_pop"
    host = "coltest"
    depends = ("cols",)

    def install(self, world):
        n = int(self.config.get("n", 500))
        C.declare(world, "pop", n, {"wealth": ("int64", (), 0), "taste": ("float16", (4,), 0.5),
                                    "best": ("float32", (), 0.0), "flag": ("bool", (), False)})
        AT.declare(world, n)

    def genesis(self, ctx):
        ctx.schedule(0, "coltest_pop.day", {}, period=86400, key="coltest_pop.day")
        ctx.schedule(1800, "coltest_pop.hour", {}, period=3600, key="coltest_pop.hour")

    @process("day")
    def day(self, ctx):
        n = C.rows(ctx.world, "pop")
        AT.reset(ctx, budget=np.full(n, 90, np.float32) if ctx.now == 0 else None)

    @process("hour")
    def hour(self, ctx):
        rng = ctx.rng("hour")
        n = C.rows(ctx.world, "pop")
        k = int(self.config.get("k", 200))
        idx = rng.integers(0, n, k)                          # unsorted, with repeats
        C.batch(ctx, "pop", "coltest.earn", idx, {"wealth": rng.integers(0, 100, k)}, mode="add")
        C.batch(ctx, "pop", "coltest.best", idx, {"best": rng.random(k, dtype=np.float32)}, mode="max")
        C.batch(ctx, "pop", "coltest.taste", idx, {"taste": rng.random((k, 4)).astype(np.float16),
                                                   "flag": rng.random(k) < 0.5}, mode="set")
        AT.spend(ctx, idx, rng.random(k) * 25, activity="browse")

    @endpoint("grow", S.obj({"n": S.s_int(minimum=1, maximum=10_000)}, ["n"]), doc="operator: add individuals")
    def ep_grow(self, ctx, n):
        C.grow(ctx, "pop", n)
        return {"rows": C.rows(ctx.world, "pop") + 0}

    @endpoint("join", S.obj({"i": S.s_int(minimum=0)}, ["i"]), doc="materialise individual i with a bank account")
    def ep_join(self, ctx, i):
        if i >= C.rows(ctx.world, "pop"):
            raise InvalidAction("no such individual")
        return {"principal": ids.ensure_principal(ctx, i, account=True)}

    @endpoint("fail_after_batch", S.obj({"k": S.s_int(minimum=1, maximum=1000)}, ["k"]),
              doc="apply a batch, then fail (rollback test)")
    def ep_fail(self, ctx, k):
        C.batch(ctx, "pop", "coltest.doomed", np.arange(k), {"wealth": np.full(k, 7)}, mode="add")
        AT.spend(ctx, np.arange(k), np.full(k, 5.0), activity="doomed")
        raise InvalidAction("deliberate failure after the batch")


def make(n=500, k=200, seed=3, store=None, strict=True):
    sp = WorldSpec("col-test", seed, [{"name": "identity", "config": {}}, {"name": "bank", "config": {}},
                                      {"name": "admin", "config": {}}, {"name": "artifacts", "config": {}},
                                      {"name": "cols", "config": {}},
                                      {"name": "coltest_pop", "config": {"n": n, "k": k}}])
    return World.create(sp, store=store, strict=strict)


def test_declare_col_and_readonly():
    w = make()
    assert C.rows(w, "pop") == 500
    t = C.tables(w)["pop"]
    assert t["columns"]["taste"] == (np.dtype("float16").str, (4,), 0.5)
    taste = C.col(w, "pop", "taste")
    assert taste.shape == (500, 4) and taste.dtype == np.float16 and float(taste[0, 0]) == 0.5
    with pytest.raises(ValueError):
        taste[0, 0] = 1.0
    assert "pop.taste" in w.state.arrays           # part of the kernel's dense arrays (snapshot + digest)


def test_batch_modes_are_exact_with_repeated_rows():
    w = make(n=50, k=10)
    w.system_call("admin.join", {"handle": "op", "endowment": 0})
    idx = np.array([5, 3, 5, 9, 3, 3])
    vals = np.array([1, 2, 3, 4, 5, 6])

    def run(fn):
        out = {}

        def body(ctx):
            out["ref"] = fn(ctx)
        w._run_system("test", body)
        return out["ref"]
    base = C.col(w, "pop", "wealth").copy()
    run(lambda ctx: C.batch(ctx, "pop", "t.add", idx, {"wealth": vals}, mode="add"))
    exp = base.copy()
    np.add.at(exp, idx, vals)
    assert np.array_equal(C.col(w, "pop", "wealth"), exp)
    run(lambda ctx: C.batch(ctx, "pop", "t.set", idx, {"wealth": vals}, mode="set"))
    exp[[5, 3, 9]] = [3, 6, 4]                       # last write wins in the order given
    assert np.array_equal(C.col(w, "pop", "wealth"), exp)
    run(lambda ctx: C.batch(ctx, "pop", "t.max", idx, {"best": vals.astype(np.float32)}, mode="max"))
    assert C.col(w, "pop", "best")[3] == 6 and C.col(w, "pop", "best")[5] == 3
    run(lambda ctx: C.batch(ctx, "pop", "t.min", idx, {"best": -vals.astype(np.float32)}, mode="min"))
    assert C.col(w, "pop", "best")[3] == -6
    ev = [e for e in w.log if e.type == "cols.batch"][-1]
    assert set(ev.payload) == {"table", "kind", "ref", "n", "cols", "mode", "extra"}
    assert ev.payload["ref"].startswith("sha256:") and ev.payload["n"] == 6
    arrs = A.get_array_blob(w, ev.payload["ref"])
    assert list(arrs["idx"]) == sorted(idx.tolist())  # stored sorted (stable), so replay applies in one pass


def test_bad_batches_are_refused():
    w = make(n=20)
    errs = []
    for kw in ({"idx": [0, 20], "values": {"wealth": [1, 2]}}, {"idx": [0, 1], "values": {"nope": [1, 2]}},
               {"idx": [0, 1], "values": {"best": [np.nan, 1.0]}}, {"idx": [0, 1], "values": {"wealth": [1]}},
               {"idx": [[0, 1]], "values": {"wealth": [1, 2]}}):
        def body(ctx, kw=kw):
            C.batch(ctx, "pop", "t.bad", np.asarray(kw["idx"]), {k: np.asarray(v) for k, v in kw["values"].items()})
        try:
            w._run_system("bad", body)
        except Exception as e:  # noqa: BLE001
            errs.append(type(e).__name__)
    assert len(errs) == 5 and set(errs) <= {"InvalidAction", "NotFound"}


def test_same_spec_and_seed_give_identical_hashes_and_replay_regen_fork_exact():
    a, b = make(seed=11), make(seed=11)
    for w in (a, b):
        w.advance_time(2 * 86400 + 7200)
    assert a.last_hash == b.last_hash and a.digest() == b.digest()
    c = make(seed=12)
    c.advance_time(2 * 86400 + 7200)
    assert c.last_hash != a.last_hash
    assert a.check_invariants() == []
    evs = [e.to_dict() for e in a.log]
    assert sum(1 for e in evs if e["type"] == "cols.batch") > 150
    r = World.replay(a.spec, evs)
    assert r.digest() == a.digest()
    assert np.array_equal(C.col(r, "pop", "taste"), C.col(a, "pop", "taste"))
    g = World.regenerate(a.spec, evs)
    assert g.last_hash == a.last_hash and g.digest() == a.digest()
    # a fork after batches equals its parent and evolves identically
    f = a.fork("col-fork")
    assert f.digest() == a.digest()
    a.advance_time(a.now + 86400)
    f.advance_time(f.now + 86400)
    assert f.digest() == a.digest() and f.last_hash == a.last_hash
    s = World.from_blob(a.snapshot_blob())
    assert s.digest() == a.digest()


def test_failed_command_rolls_columns_and_attention_back_exactly():
    w = make(n=100)
    w.advance_time(3600 * 5)
    w.system_call("admin.join", {"handle": "op", "endowment": 0})
    before = {k: v.copy() for k, v in w.state.arrays.items()}
    n_art = w.state.count("artifacts")
    r = w.execute("p:op", {"tool": "coltest.fail_after_batch", "args": {"k": 50}})
    assert not r.ok and "deliberate" in r.error["message"]
    for k, v in before.items():
        assert np.array_equal(w.state.arrays[k], v), k
    assert w.state.count("artifacts") == n_art


def test_grow_appends_fill_rows_and_replays():
    w = make(n=10)
    w.advance_time(3600 * 3)
    w.system_call("coltest.grow", {"n": 5})
    assert C.rows(w, "pop") == 15
    assert float(C.col(w, "pop", "taste")[14, 2]) == 0.5 and C.col(w, "pop", "attention_day")[14] == -1
    w.advance_time(w.now + 3600 * 3)
    evs = [e.to_dict() for e in w.log]
    assert World.replay(w.spec, evs).digest() == w.digest()
    assert World.regenerate(w.spec, evs).last_hash == w.last_hash


def test_batches_replay_from_sqlite_in_fresh_world(tmp_path):
    st = SQLiteStore(str(tmp_path / "c.db"))
    w = make(store=st)
    w.advance_time(86400)
    w.flush()
    wid, d = w.world_id, w.digest()
    w.close()
    st.close()
    A._POOL.clear()
    st2 = SQLiteStore(str(tmp_path / "c.db"))
    w2 = World.load(st2, wid)
    assert w2.digest() == d
    st2.close()


def test_attention_spend_grants_at_most_what_is_left_and_records_one_batch():
    w = make(n=10, k=1)
    w.advance_time(1)                                    # daily reset at t=0 set budget 90
    assert np.all(C.col(w, "pop", "attention_left_min") == 90)
    out = {}

    def body(ctx):
        out["g"] = AT.spend(ctx, np.array([2, 2, 2, 7]), np.array([50.0, 30.0, 40.0, 100.0]), activity="watch")
    n0 = w.seq
    w._run_system("spend", body)
    g = out["g"]
    assert g.tolist() == [50.0, 30.0, 10.0, 90.0]        # requests of one individual served in order
    assert C.col(w, "pop", "attention_left_min")[2] == 0 and C.col(w, "pop", "attention_left_min")[7] == 0
    assert C.col(w, "pop", "attention_spent_min")[2] == 90
    batches = [e for e in w.log[n0 - w.log_base:] if e.type == "cols.batch"]
    assert len(batches) == 1 and batches[0].payload["kind"] == "attention.watch"
    assert AT.check(w) == []
    # pure function agrees
    left = np.full(10, 90, np.float32)
    assert AT.granted_minutes(left, np.array([2, 2, 2]), np.array([50.0, 30.0, 40.0])).tolist() == [50, 30, 10]


def test_attention_invariant_holds_over_days_and_detects_violations():
    w = make(n=300, k=400)
    w.advance_time(3 * 86400)
    assert AT.check(w) == [] and w.check_invariants() == []
    left = C.col(w, "pop", "attention_left_min")
    assert float(left.min()) >= 0
    # an illegal direct write (not through spend) is caught, by kind, as an InvariantMonitor-compatible record
    w.state.arrays["pop.attention_spent_min"][3] = 1000.0
    found = AT.check(w)
    assert [f["kind"] for f in found] == ["attention.overspent"] and found[0]["where"] == "pop[3]"
    assert any("attention.overspent" in m for m in w.check_invariants())
    from crucible.evaluation.invariants import InvariantMonitor
    kinds = {v.kind for v in InvariantMonitor(w).check()}
    assert "module.cols" in kinds
    assert AT.violations(w)[0].kind == "attention.overspent"


def test_ids_pid_index_and_ensure_principal_shares_column_state():
    assert ids.pid(0) == "p0" and ids.pid(12345) == "p12345"
    assert ids.index("p12345") == 12345 and ids.index("p:alice") is None and ids.index("p01") is None
    assert ids.index("o:acme") is None and ids.index("p") is None
    w = make(n=50)
    w.advance_time(7200)
    cols_before = {k: v.copy() for k, v in w.state.arrays.items()}
    r = w.system_call("coltest.join", {"i": 7})
    assert r == {"principal": "p7"}
    n = w.seq
    assert w.system_call("coltest.join", {"i": 7}) == {"principal": "p7"}     # idempotent
    assert not any(e.type in ("principal.add", "id.person", "bank.account") for e in w.log[n - w.log_base:])
    p = w.state.get("principals", "p7")
    assert p["kind"] == "person" and p["info"]["pop_index"] == 7
    person = w.state.get("persons", "p7")
    assert person["kind"] == ids.PERSON_KIND and person["info"] == {"pop_index": 7}
    # no column data is copied into the records; columns are untouched
    for k, v in cols_before.items():
        assert np.array_equal(w.state.arrays[k], v)
    assert ids.materialised(w) == [7] and ids.principal_of(w, 7) == "p7" and ids.principal_of(w, 8) is None
    # the individual is a real actor with a real account: bank resolves "p7"
    w.system_call("admin.endow", {"principal": "p7", "amount": 500})
    bank = w.modules["bank"]
    assert bank.balance(w.state, bank.primary_account(w.state, "p7")) == 500
    w.system_call("admin.join", {"handle": "shop", "endowment": 0})
    acct7 = bank.primary_account(w.state, "p7")
    r = w.execute("p7", {"tool": "bank.transfer", "args": {"from_account": acct7, "to": "p:shop", "amount": 120,
                                                           "memo": "purchase"}})
    assert r.ok, r.error
    assert bank.balance(w.state, bank.primary_account(w.state, "p:shop")) == 120
    # and other participants can address the individual by its id ("p7" resolves to itself, not to "p:p7")
    shop_acct = bank.primary_account(w.state, "p:shop")
    r = w.execute("p:shop", {"tool": "bank.transfer", "args": {"from_account": shop_acct, "to": "p7", "amount": 20}})
    assert r.ok, r.error
    assert bank.balance(w.state, acct7) == 400
    evs = [e.to_dict() for e in w.log]
    assert World.regenerate(w.spec, evs).last_hash == w.last_hash


def test_cols_and_population_modules_pass_admission():
    rep = admit(["artifacts", "cols", "coltest_pop"], steps=200, configs={"coltest_pop": {"n": 200, "k": 50}})
    assert rep.passed, (rep.checks, rep.errors[:5])
