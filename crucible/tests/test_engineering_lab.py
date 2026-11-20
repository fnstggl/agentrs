"""Engineering lab world: flow physics, controller validation, the hidden traffic process, paid experiments,
gated deployment, paired production metrics, hidden-parameter secrecy, and kernel determinism with sandboxed
controllers."""
from __future__ import annotations

import copy
import json
import math
import re

import pytest

import crucible.services.admin  # noqa: F401  (registers modules)
import crucible.services.bank  # noqa: F401
import crucible.services.code  # noqa: F401
import crucible.services.docs  # noqa: F401
import crucible.services.forum  # noqa: F401
import crucible.services.identity  # noqa: F401
import crucible.services.mail  # noqa: F401
import crucible.services.search  # noqa: F401
from crucible.core.kernel import World, WorldSpec
from crucible.goals.metrics import METRICS, evaluate_metric
from crucible.worlds import engineering_lab as L
from crucible.worldgen.generator import FAMILIES

DAY, HOUR = 86400, 3600
MODS = ["identity", "bank", "admin", "code", "search", "forum", "mail", "docs", "lab"]
LAB_CFG = {"topology": "grid", "n_nodes": 16, "shortcuts": 3, "n_demands": 30, "hotspots": 1, "peak_util": 1.08,
           "shock_p": 0.0, "engineer_seat": "eng", "shocks": [{"day": 2, "kind": "hotspot"}]}
PARAMS = {"free_cpu_ms_per_day": 0, "dev_compute_cents_per_cpu_s": 5000.0}


def make_world(seed=3, cfg=None, params=None) -> World:
    mods = [{"name": m, "config": dict(LAB_CFG, **(cfg or {}))} if m == "lab" else {"name": m} for m in MODS]
    return World.create(WorldSpec("lab-test", seed, mods, dict(PARAMS, **(params or {}))), strict=True)


def log_dicts(w: World) -> list[dict]:
    return copy.deepcopy([e.to_dict() for e in w.events_range(0)])


def act(w, actor, tool, **args):
    return w.execute(actor, {"tool": tool, "args": args})


def ok(w, actor, tool, **args):
    r = act(w, actor, tool, **args)
    assert r.ok, (tool, r.error)
    return r.result


def bal(w, acct):
    return w.state.tables["accounts"][acct]["balance"]


# ====================================================================== physics
def line_net(cap=10.0, lat=2.0):
    return L.Network({"nodes": ["a", "b", "c"], "links": [
        {"id": "l0", "u": "a", "v": "b", "capacity": cap, "latency": lat},
        {"id": "l1", "u": "b", "v": "c", "capacity": cap, "latency": lat}]})


def tri_net():
    return L.Network({"nodes": ["a", "b", "c"], "links": [
        {"id": "ab", "u": "a", "v": "b", "capacity": 10.0, "latency": 1.0},
        {"id": "bc", "u": "b", "v": "c", "capacity": 10.0, "latency": 1.0},
        {"id": "ac", "u": "a", "v": "c", "capacity": 10.0, "latency": 1.5}]})


def test_link_delay_formula():
    assert L.link_delay(5.0, 0.0, 100.0) == 5.0
    assert L.link_delay(5.0, 50.0, 100.0) == pytest.approx(5.0 + 50.0 / 50.0)
    assert L.link_delay(3.0, 75.0, 100.0) == pytest.approx(3.0 + 75.0 / 25.0)
    assert L.link_delay(3.0, 75.0, 100.0, queue_scale=2.0) == pytest.approx(3.0 + 2 * 3.0)
    cap = 5.0 + L.RHO_MAX / (1 - L.RHO_MAX)
    assert L.link_delay(5.0, 99.5, 100.0) == pytest.approx(cap)   # queue saturates at rho = 0.99
    assert L.link_delay(5.0, 250.0, 100.0) == pytest.approx(cap)
    assert L.link_drop(80.0, 100.0) == 0.0
    assert L.link_drop(150.0, 100.0) == pytest.approx(50.0 / 150.0)


def test_overload_drops_excess_along_the_path():
    net = line_net()
    dem = [{"id": "a>c", "src": "a", "dst": "c", "volume": 15.0}]
    r = L.evaluate(net, dem, {"routes": {"a>c": [[["a", "b", "c"], 1.0]]}}, drop_penalty_ms=250.0)
    keep = (1 - 5.0 / 15.0) ** 2
    path_delay = 2 * (2.0 + 99.0)
    assert r["util"] == {"l0": 1.5, "l1": 1.5} and r["max_util"] == 1.5
    assert r["dropped_fraction"] == pytest.approx(1 - keep)
    assert r["mean_delay"] == pytest.approx(keep * path_delay + (1 - keep) * 250.0)
    assert r["delivered_delay"] == pytest.approx(path_delay)
    # below capacity: nothing dropped, delay = sum of link delays
    r2 = L.evaluate(net, [dict(dem[0], volume=5.0)], {"routes": {"a>c": [[["a", "b", "c"], 1.0]]}})
    assert r2["dropped"] == 0.0 and r2["mean_delay"] == pytest.approx(2 * (2.0 + 5.0 / 5.0))


def test_split_routing_loads_and_delays():
    net = tri_net()
    dem = [{"id": "a>c", "src": "a", "dst": "c", "volume": 8.0}]
    r = L.evaluate(net, dem, {"routes": {"a>c": [[["a", "c"], 0.25], [["a", "b", "c"], 0.75]]}})
    assert r["util"] == {"ab": 0.6, "ac": 0.2, "bc": 0.6} and r["invalid"] == 0
    d_direct = 1.5 + 2.0 / 8.0
    d_two = 2 * (1.0 + 6.0 / 4.0)
    assert r["mean_delay"] == pytest.approx(0.25 * d_direct + 0.75 * d_two)


@pytest.mark.parametrize("entry,reason", [
    (None, "no route"),
    ([], "non-empty"),
    ([[["a", "c"], 0.5]], "do not sum to 1"),
    ([[["a", "c"], 0.6], [["a", "b", "c"], 0.6]], "do not sum to 1"),
    ([[["a", "c"], True]], "bad fraction"),
    ([[["a", "c"], -0.5], [["a", "b", "c"], 1.5]], "bad fraction"),
    ([[["a", "c"], float("inf")]], "bad fraction"),
    ([[["b", "c"], 1.0]], "start at src"),
    ([[["a", "b"], 1.0]], "start at src"),
    ([[["a", "x", "c"], 1.0]], "unknown node"),
    ([[["a", "b", "c"], 0.5], [["a", "c", "c"], 0.5]], "not simple"),
    ([[["a", "b", "a", "c"], 1.0]], "not simple"),
    ([[["a", "c"], 0.5], [["a", "c"], 0.5]], "duplicate"),
    ([[["a", "c"], 0.25]] * 4, "more than 3"),
    ([["a", "c", 1.0]], "[path, fraction]"),
    ([[["a"], 1.0]], "at least two"),
    ([[[1, 2], 1.0]], "at least two"),
])
def test_invalid_routes_drop_the_whole_demand(entry, reason):
    net = tri_net()
    dem = [{"id": "a>c", "src": "a", "dst": "c", "volume": 4.0}, {"id": "b>c", "src": "b", "dst": "c", "volume": 2.0}]
    out = {"routes": {"a>c": entry, "b>c": [[["b", "c"], 1.0]]}}
    routes, errors = L.validate_routes(net, dem, out)
    assert reason in errors["a>c"] and "b>c" in routes
    r = L.evaluate(net, dem, out, drop_penalty_ms=100.0)
    assert r["invalid"] == 1 and r["dropped"] == pytest.approx(4.0)
    assert r["mean_delay"] == pytest.approx((4.0 * 100.0 + 2.0 * (1.0 + 2.0 / 8.0)) / 6.0)


def test_hop_without_link_is_invalid():
    dem = [{"id": "a>c", "src": "a", "dst": "c", "volume": 1.0}]
    assert L.validate_routes(line_net(), dem, {"routes": {"a>c": [[["a", "c"], 1.0]]}})[1] == {"a>c": "no link a-c"}


@pytest.mark.parametrize("output", [None, [], {"route": {}}, {"routes": []}, "x", {"routes": None}])
def test_malformed_output_drops_everything(output):
    dem = [{"id": "a>c", "src": "a", "dst": "c", "volume": 4.0}]
    r = L.evaluate(tri_net(), dem, output)
    assert r["invalid"] == 1 and r["dropped_fraction"] == 1.0


def test_fractions_within_tolerance_are_normalised():
    dem = [{"id": "a>c", "src": "a", "dst": "c", "volume": 3.0}]
    zero = {"routes": {"a>c": [[["a", "c"], 1.0], [["a", "b", "c"], 0]]}}
    assert "bad fraction" in L.validate_routes(tri_net(), dem, zero)[1]["a>c"]
    near = {"routes": {"a>c": [[["a", "c"], 0.5 + 4e-7], [["a", "b", "c"], 0.5]]}}
    routes, errors = L.validate_routes(tri_net(), dem, near)
    assert not errors and sum(f for _, f in routes["a>c"]) == pytest.approx(1.0, abs=1e-12)
    far = {"routes": {"a>c": [[["a", "c"], 0.5 + 4e-6], [["a", "b", "c"], 0.5]]}}
    assert "do not sum to 1" in L.validate_routes(tri_net(), dem, far)[1]["a>c"]


# ====================================================================== generators and reference controllers
def _offline(cfg, seed=0):
    import numpy as np
    c = dict(L.DEFAULTS, **cfg)
    rng = np.random.default_rng(seed)
    topo = L.make_topology(rng, c)
    net = L.Network(topo)
    reg = L.make_regime(rng, topo, c)
    reg["volume_scale"], sla = L.calibrate(net, reg, c)
    return topo, net, reg, sla


@pytest.mark.parametrize("cfg", [{"topology": "grid", "n_nodes": 16}, {"topology": "scale_free", "n_nodes": 30,
                                                                       "traffic": "hub", "n_demands": 60},
                                 {"topology": "waxman", "n_nodes": 20}, {"topology": "grid", "n_nodes": 5}])
def test_generators_and_greedy_beats_baseline_offline(cfg):
    topo, net, reg, sla = _offline(cfg)
    assert net.connected() and 12 <= len(topo["nodes"]) <= 40
    assert all(l["capacity"] > 0 and l["latency"] > 0 for l in topo["links"])
    greedy = L.compile_controller(L.GREEDY_SOURCE)
    tb = tg = 0.0
    for h in range(0, 24, 3):
        dem = L.sample_demands(reg, h, 100 + h)
        assert dem == L.sample_demands(reg, h, 100 + h)            # snapshots reproduce from their seed
        b = L.evaluate(net, dem, L.baseline_route(topo, dem))
        g = L.evaluate(net, dem, greedy(topo, dem))
        assert g["invalid"] == 0 and b["invalid"] == 0
        tb += b["cost"]
        tg += g["cost"]
    assert tg < tb and sla > 0


# ====================================================================== the world scenario (shared, sequential)
@pytest.fixture(scope="module")
def scenario():
    w = make_world()
    s = {"w": w}
    lab = w.modules["lab"]
    conf = lab.conf(w.state)
    s["conf"] = dict(conf)
    s["op_bal0"] = bal(w, conf["account"])
    eng = w.system_call("admin.join", {"handle": "eng", "endowment": 10_000_000})
    mal = w.system_call("admin.join", {"handle": "mallory", "endowment": 1_000_000})
    s["eng_acct"], s["mal_acct"] = eng["account"], mal["account"]
    s["eng_me"] = ok(w, "p:eng", "id.me")
    s["topology"] = ok(w, "p:eng", "lab.topology")
    s["repo"] = ok(w, "p:eng", "code.create_repo", name="routing")["repo"]
    ok(w, "p:eng", "code.commit", repo=s["repo"], files={"greedy.py": L.GREEDY_SOURCE, "broken.py": L.INVALID_SOURCE,
                                                         "crash.py": "def route(t, d):\n    raise ValueError('x')\n"},
       message="controllers")
    # paid experiments
    b0, fees0 = bal(w, s["eng_acct"]), bal(w, "acct_fees")
    s["exp_greedy"] = ok(w, "p:eng", "lab.experiment", repo=s["repo"], target="greedy.py:route",
                         hours=[0, 6, 12, 15, 17, 18, 20, 22])
    s["exp_cost"] = (b0 - bal(w, s["eng_acct"]), bal(w, "acct_fees") - fees0)
    s["exp_broken"] = ok(w, "p:eng", "lab.experiment", repo=s["repo"], target="broken.py:route", scenarios=3)
    s["exp_crash"] = ok(w, "p:eng", "lab.experiment", repo=s["repo"], target="crash.py:route", scenarios=2)
    s["exp_missing"] = act(w, "p:eng", "lab.experiment", repo=s["repo"], target="nope.py:route")
    s["exp_poor"] = act(w, "p:mallory", "lab.experiment", repo="netops/controllers", target="baseline:route",
                        scenarios=24, account=s["mal_acct"], hours=list(range(24)) * 2)
    s["mal_exp_private"] = act(w, "p:mallory", "lab.experiment", repo=s["repo"], target="greedy.py:route")
    s["mal_propose"] = act(w, "p:mallory", "lab.propose_change", repo="netops/controllers", target="baseline:route")
    s["mal_status"] = ok(w, "p:mallory", "lab.status")
    # day 0: baseline in production
    w.advance_time(DAY)
    s["day0"] = {m: evaluate_metric(w, m, {"window_h": 24}) for m in
                 ("lab_mean_delay", "lab_dropped_fraction", "lab_improvement_vs_baseline")}
    s["op_bal_day0"] = bal(w, conf["account"])
    s["prop_greedy"] = ok(w, "p:eng", "lab.propose_change", repo=s["repo"], target="greedy.py:route")
    s["status_pending"] = ok(w, "p:eng", "lab.status")
    w.advance_time(2 * DAY + HOUR)             # greedy deployed at day 1 02:00 -> hours 26..48
    s["greedy"] = {m: evaluate_metric(w, m, {"window_h": 23}) for m in
                   ("lab_mean_delay", "lab_dropped_fraction", "lab_improvement_vs_baseline")}
    s["pen_greedy"] = w.state.get("lab", "totals")["penalties"]
    s["prop_broken"] = ok(w, "p:eng", "lab.propose_change", repo=s["repo"], target="broken.py:route")
    w.advance_time(3 * DAY + HOUR)             # broken deployed at day 2 02:00 -> hours 50..72
    s["broken"] = {m: evaluate_metric(w, m, {"window_h": 23}) for m in
                   ("lab_mean_delay", "lab_dropped_fraction", "lab_improvement_vs_baseline")}
    s["pen_total"] = w.state.get("lab", "totals")["penalties"]
    s["measure"] = ok(w, "p:mallory", "lab.measure", window_h=30)
    s["status_end"] = ok(w, "p:eng", "lab.status")
    yield s
    w.close()


def test_genesis_operator_repo_and_public_topology(scenario):
    w, s = scenario["w"], scenario
    conf = s["conf"]
    assert conf["operator"] == "o:netops" and conf["repo"] == "netops/controllers" and conf["sla_ms"] > 0
    repo = w.state.get("repos", "netops/controllers")
    assert repo["visibility"] == "public" and repo["license"] == "open"
    assert ok(w, "p:mallory", "code.read", repo="netops/controllers", path="baseline.py")["content"] == \
        L.BASELINE_SOURCE
    topo = s["topology"]
    assert len(topo["nodes"]) == 16 and topo["kind"] == "grid"
    assert {"id", "u", "v", "capacity", "latency"} <= set(topo["links"][0])
    # engineer seat: member of the operator + lab.operate grant; mallory has neither
    assert {"org": "o:netops", "role": "member", "title": "network engineer"} in s["eng_me"]["orgs"]
    assert any(g["cap"] == "lab.operate" and g["owner"] == "o:netops" for g in s["eng_me"]["grants_held"])


def test_experiments_cost_money_and_compute(scenario):
    s = scenario
    e = s["exp_greedy"]
    conf = s["conf"]
    assert e["fee_cents"] == conf["experiment_fee_cents"] + 8 * conf["scenario_fee_cents"]
    assert e["cpu_ms"] > 0 and e["compute_cents"] > 0
    assert s["exp_cost"] == (e["fee_cents"] + e["compute_cents"], e["compute_cents"])
    assert e["metrics"]["invalid_demands"] == 0 and e["controller_error"] is None
    assert e["improvement_vs_baseline"] > 0.05
    assert e["metrics"]["mean_delay"] < e["baseline"]["mean_delay"]
    assert [p["hour_of_day"] for p in e["per_scenario"]] == [0, 6, 12, 15, 17, 18, 20, 22]
    b = s["exp_broken"]
    assert b["metrics"]["dropped_fraction"] > 0.5 and b["improvement_vs_baseline"] < 0 and b["errors"]
    c = s["exp_crash"]
    assert c["metrics"]["dropped_fraction"] == 1.0
    assert c["errors"][0] == "controller raised: ValueError at crash.py:2"     # message withheld (side channel)
    assert s["exp_missing"].error["code"] == "not_found"
    assert s["exp_poor"].error["code"] == "invalid_action"          # more scenarios than allowed
    assert s["mal_exp_private"].error["code"] == "forbidden"
    w = s["w"]
    totals = w.state.get("lab", "totals")
    assert totals["experiments"] == 3 and totals["fees"] == sum(x["fee_cents"] for x in (e, b, c))


def test_deployment_needs_the_operate_grant(scenario):
    s = scenario
    assert s["mal_propose"].error["code"] == "forbidden"
    assert s["mal_status"]["you_operate"] is False and s["mal_status"]["pending"] is False
    p = s["prop_greedy"]
    assert p["deploy_at"] == DAY + 2 * HOUR
    assert s["status_pending"]["pending"]["target"] == "greedy.py:route"
    end = s["status_end"]
    assert end["deployed"]["target"] == "broken.py:route" and end["deployed"]["by"] == "p:eng"
    assert end["deployed"]["since"] == 2 * DAY + 2 * HOUR and end["pending"] is None


def test_paired_improvement_tracks_the_deployed_controller(scenario):
    s = scenario
    d0, g, b = s["day0"], s["greedy"], s["broken"]
    assert d0["lab_improvement_vs_baseline"] == 0.0           # the baseline is deployed: paired difference is exact
    assert d0["lab_mean_delay"] > 0
    assert g["lab_improvement_vs_baseline"] > 0.1
    assert g["lab_dropped_fraction"] <= d0["lab_dropped_fraction"] + 1e-9
    assert b["lab_improvement_vs_baseline"] < -1.0
    assert b["lab_dropped_fraction"] > 0.5
    assert b["lab_mean_delay"] > s["conf"]["sla_ms"]
    print("\nlab metrics", {"baseline_day": d0, "greedy_23h": g, "broken_23h": b, "sla_ms": s["conf"]["sla_ms"],
                            "penalties_cents": s["pen_total"]})


def test_sla_penalties_hit_the_operator(scenario):
    s, w = scenario, scenario["w"]
    assert s["pen_total"] > s["pen_greedy"]                   # the broken controller breaches the SLA every hour
    pens = [t for _, t in w.state.scan("txns") if t["kind"] == "sla_penalty"]
    assert sum(-dict(t["postings"])[s["conf"]["account"]] for t in pens) == s["pen_total"]
    fees = sum(x["fee_cents"] for x in (s["exp_greedy"], s["exp_broken"], s["exp_crash"]))
    assert bal(w, s["conf"]["account"]) == s["op_bal0"] + fees - s["pen_total"]


def test_dashboard_is_delayed_sampled_and_noisy(scenario):
    s, w = scenario, scenario["w"]
    m = s["measure"]
    last = w.now // HOUR - 1
    hours = [r["hour"] for r in m["rows"]]
    assert hours == list(range(last - 30 + 1, last - 1)) and max(hours) == last - m["delay_h"]
    exact = {r["hour"]: r["exact"] for _, r in w.state.scan("lab_hours")}
    diffs = [abs(r["mean_delay"] - exact[r["hour"]]["mean_delay"]) for r in m["rows"]]
    assert max(diffs) > 0                                       # measurement is not the ground truth
    rel = [abs(r["max_util"] / exact[r["hour"]]["max_util"] - 1) for r in m["rows"]]
    assert max(rel) < 0.2
    pub = [e for e in w.events_range(0) if e.type == "lab.dashboard"]
    assert pub and all(e.vis == "public" for e in pub)
    assert pub[-1].payload["hour"] == last - m["delay_h"]


def test_drift_and_scheduled_shock_change_the_hidden_process(scenario):
    w = scenario["w"]
    regimes = [r for _, r in w.state.scan("lab_regimes")]
    assert [r["day"] for r in regimes] == [0, 1, 2, 3]
    assert regimes[1]["cause"] == "drift" and regimes[2]["cause"] == "drift+hotspot"
    assert len(regimes[2]["params"]["hotspots"]) == len(regimes[1]["params"]["hotspots"]) + 1
    assert regimes[1]["params"]["base_rates"] != regimes[0]["params"]["base_rates"]


def test_hidden_traffic_parameters_never_visible(scenario):
    w = scenario["w"]
    secrets = set()
    for _, r in w.state.scan("lab_regimes"):
        p = r["params"]
        secrets |= {repr(p["diurnal"]["amp"]), repr(p["volume_scale"])} | {repr(h[2]) for h in p["hotspots"]}
        secrets |= {repr(x) for x in p["diurnal"]["peaks"]}
    secrets |= {str(r["seed"]) for _, r in w.state.scan("lab_hours")}
    for who in ("p:eng", "p:mallory"):
        evs, dropped = w.visible_events(who, 0, limit=10**9)
        assert dropped == 0 and evs
        assert not [e.type for e in evs if e.type in ("lab.regime", "lab.hour", "lab.experiment_samples", "ext")]
        for e in evs:
            text = json.dumps(e.payload)
            assert not any(f'"{k}"' in text for k in L.HIDDEN_KEYS), e.type
            nums = set(re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", text))
            assert not (secrets & nums), (e.type, secrets & nums)
    # public observation of the lab carries no hidden data either
    assert not any(f'"{k}"' in json.dumps(w.observe("p:eng")) for k in L.HIDDEN_KEYS)


def test_invariants_replay_and_regeneration(scenario):
    w = scenario["w"]
    assert w.check_invariants() == []
    assert w.modules["bank"].check_ledger_consistency(w) == []
    evs = log_dicts(w)
    r = World.replay(w.spec, evs)
    assert (r.digest(), r.last_hash, r.seq) == (w.digest(), w.last_hash, w.seq)
    assert evaluate_metric(r, "lab_improvement_vs_baseline", {"window_h": 23}) == \
        scenario["broken"]["lab_improvement_vs_baseline"]
    g = World.regenerate(w.spec, evs)
    try:
        assert (g.last_hash, g.seq) == (w.last_hash, w.seq)
        assert g.digest() == w.digest()
    finally:
        g.close()
        r.close()


# ====================================================================== families and registry
def test_metrics_registered():
    assert {"lab_mean_delay", "lab_dropped_fraction", "lab_improvement_vs_baseline"} <= set(METRICS)


@pytest.mark.parametrize("family,split,kind,traffic", [("engineering_lab", "train", "grid", "gravity"),
                                                       ("engineering_lab_heldout", "test", "scale_free", "hub")])
def test_families(family, split, kind, traffic):
    assert FAMILIES[family].split == split
    spec = L.lab_spec(family, seed=5)
    assert [m["name"] for m in spec.modules] == MODS and spec.params["family_split"] == split
    w = World.create(spec, strict=True)
    try:
        topo = w.state.get("lab", "topology")["topology"]
        lab_cfg = w.modules["lab"].cfg
        assert topo["kind"] == kind and lab_cfg["traffic"] == traffic and lab_cfg["engineer_seat"] == "eng"
        assert 12 <= len(topo["nodes"]) <= 40
        if split == "test":
            assert len(topo["nodes"]) >= 28
        eng = w.system_call("admin.join", {"handle": "eng", "endowment": 1_000_000})
        assert ok(w, "p:eng", "lab.status")["you_operate"] is True
        w.advance_time(6 * HOUR)
        assert len(w.state.ids("lab_hours")) == 6 and w.check_invariants() == []
        assert evaluate_metric(w, "lab_improvement_vs_baseline", {"window_h": 6}) == 0.0
        assert eng["principal"] == "p:eng"
    finally:
        w.close()


def test_insufficient_funds_rejects_experiment_without_side_effects():
    w = make_world(seed=7, cfg={"engineer_seat": None})
    try:
        poor = w.system_call("admin.join", {"handle": "poor", "endowment": 500})
        ok(w, "p:poor", "code.create_repo", name="r")
        ok(w, "p:poor", "code.commit", repo="poor/r", files={"c.py": L.GREEDY_SOURCE})
        seq = w.seq
        r = act(w, "p:poor", "lab.experiment", repo="poor/r", target="c.py:route", scenarios=1)
        assert r.error["code"] == "insufficient_funds" and bal(w, poor["account"]) == 500
        assert [e.type for e in w.events_range(seq)] == ["cmd.rejected"]
        # no seat configured: nobody but the operator may deploy
        assert act(w, "p:poor", "lab.propose_change", repo="poor/r", target="c.py:route").error["code"] == "forbidden"
        assert math.isfinite(w.modules["lab"].conf(w.state)["sla_ms"])
    finally:
        w.close()
