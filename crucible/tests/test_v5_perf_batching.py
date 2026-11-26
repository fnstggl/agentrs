"""Shared batched inference across participant classes (crucible.substrate.inference_batch; MASTER_SPEC 13.1, SP-13.1).

Real models of the tree: class-2 `AdaptivePolicy` (shared adaptive network), class-3 `StrategicGroup` (GRU and MLP
strategic policies) and the class-4 `SemanticService` (Qwen3-0.6B). Decisions requested by several call sites in
one transaction are grouped by compatibility, evaluated in one call per shared model and recorded once; outputs
equal the separate evaluations; regeneration re-forms the same groups from the record without running a model."""
import numpy as np
import pytest

import crucible
from crucible.core import schema as S
from crucible.core.errors import ProtocolError
from crucible.core.kernel import Module, World, WorldSpec, applier, endpoint, register_module
from crucible.substrate import inference as I
from crucible.substrate import inference_batch as IB

crucible.load_all_modules()
torch = pytest.importorskip("torch")       # part of the environment; a missing torch is a reported failure
torch.set_num_threads(1)

OBS, CAND, MEM, K = 12, 9, 8, 5


def _policies():
    from crucible.population.adaptive import AdaptivePolicy
    from crucible.population.strategic import StrategicGroup
    a = AdaptivePolicy(0, OBS, CAND, MEM, hidden=32, seed=1)
    b = AdaptivePolicy(1, OBS, CAND, MEM, hidden=32, seed=2)
    g = StrategicGroup({"group": "gru", "arch": "gru", "hidden": 48, "learning": False, "code": 0}, seed=3)
    m = StrategicGroup({"group": "mlp", "arch": "mlp", "hidden": 64, "learning": False, "code": 1}, seed=4)
    return a, b, g, m


RUNS = {"adaptive": 0, "strategic": 0}


def _counting(obj, meth, key):
    f = getattr(obj, meth)

    def wrapped(*a, **k):
        RUNS[key] += 1
        return f(*a, **k)
    setattr(obj, meth, wrapped)


@register_module
class IBTest(Module):
    """Several call sites deciding in one transaction: three class-2 cohorts of one policy group, one cohort of
    another group, two class-3 cohorts of the GRU group and one of the MLP group."""
    name = "ibtest"
    host = "ibt"

    def __init__(self, config=None):
        super().__init__(config)
        self.a, self.b, self.g, self.m = _policies()
        _counting(self.a, "scores", "adaptive")
        _counting(self.b, "scores", "adaptive")
        _counting(self.g, "logits", "strategic")
        _counting(self.m, "logits", "strategic")

    def install(self, world):
        world.state.declare_table("ib_decisions")

    def _class2(self, ctx, pol, n, tag):
        rng = ctx.rng(f"obs.{tag}")
        obs = rng.normal(size=(n, OBS)).astype(np.float32)
        mem = rng.normal(size=(n, MEM)).astype(np.float32)
        cand = rng.normal(size=(n, K, CAND)).astype(np.float32)
        out = yield IB.adaptive_request(pol, obs, mem, cand)
        pick = np.argmax(out["scores"], axis=1)
        ctx.emit("ibt.decision", {"tag": tag, "picks": [int(x) for x in pick],
                                  "mem0": float(out["memory"][0, 0])}, vis="internal")
        return pick

    def _class3(self, ctx, grp, n, tag):
        from crucible.population.strategic import HIST, OBS_DIM
        rng = ctx.rng(f"obs.{tag}")
        obs = rng.normal(size=(n, OBS_DIM)).astype(np.float32)
        hist = rng.normal(size=(n, HIST, OBS_DIM)).astype(np.float32)
        out = yield IB.strategic_request(grp, obs, hist)
        acts = {k: [int(x) for x in np.argmax(out[k], axis=1)] for k in ("topic", "price")}
        ctx.emit("ibt.decision", {"tag": tag, "picks": acts["topic"] + acts["price"], "mem0": 0.0}, vis="internal")
        return acts

    @endpoint("round", S.obj({"n": S.s_int(minimum=1, maximum=64)}, ["n"]), doc="one decision round")
    def ep_round(self, ctx, n):
        entered = []

        class Owner:                                  # stands in for population.core.running_as(ctx, owner)
            def __enter__(self):
                entered.append(1)

            def __exit__(self, *a):
                return False
        tasks = [IB.Task(self._class2(ctx, self.a, n, "a1"), Owner), self._class2(ctx, self.a, n + 3, "a2"),
                 self._class3(ctx, self.g, n, "g1"), self._class2(ctx, self.b, n, "b1"),
                 self._class2(ctx, self.a, 1, "a3"), self._class3(ctx, self.g, 2, "g2"),
                 self._class3(ctx, self.m, n, "m1"), "not a generator"]
        res = IB.gather(ctx, tasks)
        assert res[-1] == "not a generator" and len(entered) == 2     # the wrapped task's two steps
        assert res[0].shape == (n,)
        return {"tasks": len(res)}

    @applier("ibt.decision")
    def ap_decision(self, ev):
        self.world.state.put("ib_decisions", str(ev.seq), dict(ev.payload))


def make():
    sp = WorldSpec("ib-test", 1, [{"name": "identity", "config": {}}, {"name": "bank", "config": {}},
                                  {"name": "admin", "config": {}}, {"name": "artifacts", "config": {}},
                                  {"name": "ibtest", "config": {}}])
    w = World.create(sp, strict=True)
    w.system_call("admin.join", {"handle": "eve", "endowment": 0})
    return w


def test_compatible_requests_share_one_call_per_model_and_regenerate_exactly():
    w = make()
    RUNS.update(adaptive=0, strategic=0)
    for n in (4, 7, 16):
        assert w.execute("p:eve", {"tool": "ibt.round", "args": {"n": n}}).ok
    st = IB.report(IB.stats(w))
    # per round: policy a (3 call sites merged), policy b, strategic gru (2 merged), strategic mlp -> 4 calls
    assert st["calls"] == 12 and st["computed_calls"] == 12 and st["requests"] == 21
    assert st["merged_calls"] == 6
    assert RUNS == {"adaptive": 6, "strategic": 6}
    c2, c3 = st["by_class"]["2"], st["by_class"]["3"]
    assert c2["calls"] == 6 and c2["requests"] == 12 and c3["calls"] == 6 and c3["requests"] == 9
    assert c2["rows_per_call"] > 1 and c3["rows_per_call"] > 1
    assert st["by_op"]["adaptive.scores"]["max_rows"] == 16 + 19 + 1
    recs = [r for r in I.records([e.to_dict() for e in w.log]) if r["kind"].startswith("batch.")]
    assert len(recs) == 12
    assert {r["model_hash"] for r in recs if r["kind"] == "batch.adaptive.scores"} == \
        {w.modules["ibtest"].a.weights_hash(), w.modules["ibtest"].b.weights_hash()}
    # regeneration: the same groups are formed again and served from the record (no model runs)
    before = dict(RUNS)
    g = World.regenerate(w.spec, [e.to_dict() for e in w.log])
    assert g.last_hash == w.last_hash and g.digest() == w.digest()
    gm = g.modules["ibtest"]
    assert RUNS == before                                  # no model ran during regeneration
    assert IB.report(IB.stats(g))["served_calls"] == 12 and IB.report(IB.stats(g))["computed_calls"] == 0
    assert gm is not w.modules["ibtest"]


def test_merged_outputs_equal_separate_evaluation():
    a, b, g, m = _policies()
    rng = np.random.default_rng(0)
    parts = []
    for n in (3, 8, 1, 13):
        parts.append((rng.normal(size=(n, OBS)).astype(np.float32), rng.normal(size=(n, MEM)).astype(np.float32),
                      rng.normal(size=(n, K, CAND)).astype(np.float32)))
    merged = IB.ADAPTIVE_SCORES.run(a, {"obs": np.concatenate([p[0] for p in parts]),
                                        "mem": np.concatenate([p[1] for p in parts]),
                                        "cand": np.concatenate([p[2] for p in parts])}, {"use_memory": True})
    off = 0
    agree = total = 0
    for o, mm, c in parts:
        s, m2 = a.scores(o, mm, c)
        n = len(o)
        np.testing.assert_allclose(merged["scores"][off:off + n], s, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(merged["memory"][off:off + n], m2, rtol=1e-5, atol=1e-5)
        agree += int((np.argmax(merged["scores"][off:off + n], 1) == np.argmax(s, 1)).sum())
        total += n
        off += n
    assert agree == total


def test_incompatible_requests_are_never_merged():
    a, b, g, m = _policies()
    x = np.zeros((2, OBS), np.float32), np.zeros((2, MEM), np.float32), np.zeros((2, K, CAND), np.float32)
    ra = IB.adaptive_request(a, *x)
    rb = IB.adaptive_request(b, *x)                       # other weights
    a_nomem = IB.adaptive_request(a, *x)
    a_nomem.static = {"use_memory": False}               # other static parameter
    wider = IB.adaptive_request(a, *x[:2], np.zeros((2, K + 1, CAND), np.float32))   # other row shape
    keys = {IB.compat_key(r) for r in (ra, rb, a_nomem, wider)}
    assert len(keys) == 4
    assert IB.compat_key(ra) == IB.compat_key(IB.adaptive_request(a, *x))
    with pytest.raises(ProtocolError):
        IB.Request(IB.ADAPTIVE_SCORES, a, "x", "h", {"obs": np.zeros((2, 3)), "mem": np.zeros((3, 3))}).n_rows()


def test_max_rows_splits_groups_without_splitting_a_request():
    reqs = []
    op = IB.Op("t.sum", lambda mdl, x, s: x.sum(axis=1, keepdims=True), max_rows=10)
    for n in (4, 4, 4, 12, 1):
        reqs.append(IB.Request(op, None, "t", "h1", np.ones((n, 3), np.float32), cls=1))
    assert [len(c) for c in IB._chunks(list(range(5)), reqs, 10)] == [2, 1, 1, 1]


SEM_PROMPTS = [("Which is a fruit?", ["apple", "hammer"]), ("Which is a colour?", ["seven", "blue"]),
               ("Which can fly?", ["eagle", "stone"])]


@register_module
class IBSem(Module):
    """Class-4 semantic generalists of two groups that share the semantic service: their choose calls merge."""
    name = "ibsem"
    host = "ibs"

    def install(self, world):
        world.state.declare_table("ib_sem")

    def _participant(self, ctx, svc, prompts, tag):
        reqs = [{"messages": [{"role": "user", "content": q}], "choices": ch} for q, ch in prompts]
        out = yield IB.semantic_request(svc, "choose", reqs)
        ctx.emit("ibs.choice", {"tag": tag, "choices": [int(o["choice"]) for o in out]}, vis="internal")
        return out

    @endpoint("round", S.obj({}, []), doc="one round of two semantic groups")
    def ep_round(self, ctx):
        from crucible.semantic.service import service_for
        svc = service_for()
        IB.gather(ctx, [self._participant(ctx, svc, SEM_PROMPTS[:2], "g1"),
                        self._participant(ctx, svc, SEM_PROMPTS[2:], "g2")])
        return {}

    @applier("ibs.choice")
    def ap_choice(self, ev):
        self.world.state.put("ib_sem", str(ev.seq), dict(ev.payload))


def test_semantic_requests_of_several_groups_share_the_service_call():
    sp = WorldSpec("ib-sem", 1, [{"name": "identity", "config": {}}, {"name": "bank", "config": {}},
                                 {"name": "admin", "config": {}}, {"name": "artifacts", "config": {}},
                                 {"name": "ibsem", "config": {}}])
    w = World.create(sp, strict=True)
    w.system_call("admin.join", {"handle": "eve", "endowment": 0})
    assert w.execute("p:eve", {"tool": "ibs.round", "args": {}}).ok
    st = IB.report(IB.stats(w))
    assert st["calls"] == 1 and st["requests"] == 2 and st["by_class"]["4"]["rows"] == 3
    got = [v["choices"] for v in w.state.tables["ib_sem"].values()]
    assert sorted(sum(got, [])) == sorted([0, 1, 0])          # apple, blue, eagle
    g = World.regenerate(w.spec, [e.to_dict() for e in w.log])
    assert g.last_hash == w.last_hash
    assert IB.report(IB.stats(g))["served_calls"] == 1


@register_module
class IBRound(Module):
    """The submit/flush form: two call sites submit class-2 requests of one policy group; one call."""
    name = "ibround"
    host = "ibr"

    def __init__(self, config=None):
        super().__init__(config)
        self.a = _policies()[0]

    def install(self, world):
        world.state.declare_table("ib_round")

    @endpoint("go", S.obj({}, []), doc="submit/flush")
    def ep_go(self, ctx):
        rng = ctx.rng("x")
        hs = []
        with IB.batched(ctx) as rnd:
            for n in (2, 5):
                hs.append(rnd.submit(IB.adaptive_request(self.a, rng.normal(size=(n, OBS)).astype(np.float32),
                                                         np.zeros((n, MEM), np.float32),
                                                         rng.normal(size=(n, K, CAND)).astype(np.float32))))
            assert not any(h.done for h in hs)
        assert all(h.done for h in hs) and [len(h.value["scores"]) for h in hs] == [2, 5]
        ctx.emit("ibr.done", {"n": 7}, vis="internal")
        return {}

    @applier("ibr.done")
    def ap_done(self, ev):
        self.world.state.put("ib_round", str(ev.seq), dict(ev.payload))


def test_submit_flush_round_groups_requests():
    sp = WorldSpec("ib-round", 1, [{"name": "identity", "config": {}}, {"name": "bank", "config": {}},
                                   {"name": "admin", "config": {}}, {"name": "artifacts", "config": {}},
                                   {"name": "ibround", "config": {}}])
    w = World.create(sp, strict=True)
    w.system_call("admin.join", {"handle": "eve", "endowment": 0})
    assert w.execute("p:eve", {"tool": "ibr.go", "args": {}}).ok
    st = IB.report(IB.stats(w))
    assert st["calls"] == 1 and st["requests"] == 2 and st["rows"] == 7 and st["by_class"]["2"]["merged_calls"] == 1
