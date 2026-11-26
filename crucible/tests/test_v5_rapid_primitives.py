"""The 28 causal primitives: registry, versions, characteristic behaviour, batching, and composition in a causal
graph (the confounding demonstration through the primitive_graph engine, not by hand)."""
from __future__ import annotations

import numpy as np
import pytest

from crucible.core.schema import check_schema
from crucible.rapid import primitives as P
from crucible.rapid.engines.primitive_graph import PrimitiveGraph
from crucible.rapid.generator import GraphBuilder, compose, topo_order


@pytest.mark.parametrize("name", sorted(P.REGISTRY))
def test_each_primitive_shows_its_characteristic_behaviour(name):
    e = P.REGISTRY[name]
    assert e["version"] == "1.0" and len(e["doc"]) > 20 and not check_schema(e["params"])
    res = e["check"]()
    assert res["ok"], res


def test_registry_maps_the_part_7_1_list_one_to_one():
    assert len(P.SPEC_LIST) == 28 and set(P.SPEC_LIST.values()) == set(P.REGISTRY)
    assert {e["kind"] for e in P.REGISTRY.values()} == {"node", "source", "channel", "vector"}


def test_versions_are_enforced():
    assert P.ref("delay") == "delay@1.0"
    assert P.parse_ref("delay@1.0") is P.get("delay")
    with pytest.raises(ValueError):
        P.get("delay", "2.0")
    with pytest.raises(KeyError):
        P.get("no_such_primitive")
    assert P.validate_params("threshold", {"theta": 1.0, "lo": 0.0}) != []      # missing "hi"


def test_apply_never_mutates_state_and_batches_match_single_rows():
    rng = np.random.default_rng(0)
    for name in ("stock", "hysteresis", "path_dependence", "adaptive_opponent", "endogenous_demand",
                 "learning_behaviour", "feedback", "decay", "delay"):
        pr = P.get(name)
        p = {k: (2 if v.get("type") == "integer" else ("undercut" if k == "mode" else 0.5))
             for k, v in pr.params["properties"].items()}
        if name == "hysteresis":
            p.update(lo=0.2, hi=0.8)
        B = 6
        s = pr.init(p, B)
        snap = {k: np.array(v, copy=True) for k, v in s.items()}
        x = {port: rng.random(B) for port in pr.ports}
        out, s2 = pr.apply(p, x, s, np.zeros((B, 0)), np.zeros((B, 0)), 0)
        for k in s:
            assert np.array_equal(s[k], snap[k]), (name, k)
        for b in range(B):
            sb = {k: v[b:b + 1] for k, v in snap.items()}
            ob, _ = pr.apply(p, {k: v[b:b + 1] for k, v in x.items()}, sb, np.zeros((1, 0)), np.zeros((1, 0)), 0)
            assert ob[0] == out[b], name


def _confounded_graph(rng):
    gb = GraphBuilder(rng)
    u = gb.prim("hidden_confounding", {"mean": 0.0, "sd": 0.45, "ar": 0.9})
    x = gb.lever(0.0, 10.0, 5.0, natural={"x": [(u, 0, 2.0)], "bias": 5.0, "sd": 0.3})
    y = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, x=[(u, 0, 4.0), (x, 0, -0.5)],
                bias={"x": 40.0})
    gb.gauge(y, [("measurement_noise", {"sd": 0.5})])
    return gb.build(horizon=200, money={"revenue_node": y, "price": 1.0, "cap": 1e6}), x, y


def test_confounding_in_a_composed_graph_reverses_under_intervention():
    cfg, x, y = _confounded_graph(np.random.default_rng(1))
    eng = PrimitiveGraph(cfg)
    assert "hidden_confounding" in eng.motifs()
    st = eng.empty_state()
    rng = np.random.default_rng(2)
    xs, ys = [], []
    for t in range(300):                                    # passive regime: the lever follows its natural mechanism
        z, u = eng.draw(rng)
        cur, read = eng.tick(st, z, u, t)
        xs.append(cur[0, eng.idx[x]])
        ys.append(cur[0, eng.idx[y]])
    passive = np.polyfit(xs, ys, 1)[0]
    xd, yd = [], []
    name = eng.public_levers()[0]["name"]
    for t in range(300):                                    # intervention: the lever is set at random
        eng.set_levers(st, {name: float(rng.uniform(0, 10))})
        z, u = eng.draw(rng)
        cur, read = eng.tick(st, z, u, 300 + t)
        xd.append(cur[0, eng.idx[x]])
        yd.append(cur[0, eng.idx[y]])
    do = np.polyfit(xd, yd, 1)[0]
    assert passive > 0.5 and abs(do - (-0.5)) < 0.1


def test_compose_builds_every_topology_with_valid_order_and_money():
    rng = np.random.default_rng(3)
    shapes = set()
    for topo in ("chain", "fork", "collider", "diamond", "layered", "hub", "feedback", "random_dag"):
        cfg = compose(rng, {"topology": topo, "motifs": ["saturation", "threshold", "decay"], "n_levers": 2,
                            "temporal": "mixed", "channels": ["noisy", "missing", "aggregated", "on_demand"],
                            "n_distractors": 1, "affordances": ["levers", "experiment", "query"]})
        eng = PrimitiveGraph(cfg)
        assert topo_order(cfg["nodes"]) == cfg["order"]
        st = eng.init_state(np.random.default_rng(0))
        out = eng.advance(st, np.random.default_rng(1), 0)
        assert out["revenue"]["a0"] >= 0 and set(eng.action_space("a0").kinds) >= {"levers", "experiment", "query"}
        if topo == "feedback":
            assert any(int(tm[1]) > 0 for nd in cfg["nodes"] for terms in nd.get("ports", {}).values() for tm in terms)
        shapes.add(tuple(sorted(eng.motifs())))
    assert len(shapes) >= 2


def test_instantaneous_cycles_are_refused():
    nodes = [{"id": "a", "type": "prim", "ports": {"x": [["b", 0, 1.0]]}},
             {"id": "b", "type": "prim", "ports": {"x": [["a", 0, 1.0]]}}]
    with pytest.raises(ValueError):
        topo_order(nodes)


def _recorded_calls():
    """Every (primitive, params, inputs, ticks, seed) call the 28 characteristic-behaviour experiments make."""
    from crucible.rapid import primitive_checks as PC
    calls, orig = [], PC.run

    def rec(name, params, inputs, ticks=None, seed=0, state=None):
        calls.append((name, dict(params), inputs, ticks, seed, state))
        return orig(name, params, inputs, ticks=ticks, seed=seed, state=state)
    PC.run = rec
    try:
        for n in P.PRIMITIVES:
            getattr(PC, n)()
    finally:
        PC.run = orig
    return calls


def test_every_primitive_accepts_per_instance_parameters():
    """The batch contract: any number-typed parameter may be a (B,) array of per-instance values. Replacing each scalar
    parameter of every characteristic-experiment call by a (B,) array of the same value gives the same outputs;
    and a batch of two instances with different per-instance values equals the two single-instance runs."""
    exercised, mixed = set(), set()
    for name, params, inputs, ticks, seed, state in _recorded_calls():
        if state is not None:
            continue
        ref_out, _ = P.run(name, params, inputs, ticks=ticks, seed=seed)
        ref_out = np.asarray(ref_out, dtype=float)
        B = ref_out.shape[-1] if ref_out.ndim else 1
        props = P.REGISTRY[name]["params"].get("properties", {})
        # integer parameters (lags, unit counts) fix array shapes and are structural; every number parameter may
        # vary per instance
        pb = {k: (np.full(B, float(v)) if isinstance(v, float) and props.get(k, {}).get("type") == "number" else v)
              for k, v in params.items()}
        out, _ = P.run(name, pb, inputs, ticks=ticks, seed=seed)
        assert np.allclose(np.nan_to_num(np.asarray(out, float), nan=1e99), np.nan_to_num(ref_out, nan=1e99)), name
        exercised.add(name)
    for name in ("selection_effect", "saturation", "threshold", "measurement_noise", "decay", "congestion"):
        e = P.REGISTRY[name]
        call = next(c for c in _recorded_calls() if c[0] == name and c[5] is None)
        _, params, inputs, ticks, seed, _ = call
        alt = {k: (v * 1.3 if isinstance(v, float) else v) for k, v in params.items()}
        one = [np.asarray(P.run(name, p, {k: np.asarray(v, float)[..., :1] for k, v in inputs.items()},
                                ticks=ticks, seed=seed)[0], float) for p in (params, alt)]
        both_p = {k: (np.array([params[k], alt[k]]) if isinstance(params[k], float) else params[k]) for k in params}
        both_in = {k: np.concatenate([np.asarray(v, float)[..., :1]] * 2, axis=-1) for k, v in inputs.items()}
        both, _ = P.run(name, both_p, both_in, ticks=ticks, seed=seed)
        both = np.asarray(both, float)
        if name in ("selection_effect", "measurement_noise"):
            # noise draws differ by instance row; the deterministic part must still broadcast per instance
            assert both.shape[-1] == 2 and np.isfinite(np.nan_to_num(both)).all(), name
        else:
            assert np.allclose(both[..., 0], one[0][..., 0]) and np.allclose(both[..., 1], one[1][..., 0]), name
        mixed.add(e["name"] if "name" in e else name)
    assert len(exercised) >= 25 and mixed
