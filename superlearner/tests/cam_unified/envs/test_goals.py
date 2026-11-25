"""Operator-side metrics and predicates for OW-1 (operating profit), OW-4 (recovery after a shock), OW-6
(anti-trivial self-goal filter) and OW-7 (joint utility), on real Crucible worlds."""
from __future__ import annotations

import json

import pytest

import crucible
crucible.load_all_modules()
from crucible.goals.contracts import GoalContract, GoalVerifier      # noqa: E402
from crucible.goals.metrics import METRICS, check_predicate, evaluate_metric   # noqa: E402
from crucible.research import llm_eval as L                         # noqa: E402
from crucible.usefulness.agents import ExampleInductor             # noqa: E402

from superlearner.cam_unified.interfaces import Action                # noqa: E402
from superlearner.cam_unified.envs import goals as G                  # noqa: E402
from superlearner.cam_unified.envs.agent_protocol import PolicyAgent  # noqa: E402
from superlearner.cam_unified.envs.harness import run_lifetime        # noqa: E402
from superlearner.cam_unified.envs.openworld import open_world        # noqa: E402
from superlearner.cam_unified.envs.openworld_eval import OpenWorldEvaluator   # noqa: E402

DAY = 86400


def tiny(days=8, extra=()):
    g = L.open_economy_goal(world_family="ow_tiny", seed=1, overrides={"n_consumers": 800}, days=days)
    if extra:
        g.world = dict(g.world, extra_modules=list(extra))
    return g


@pytest.fixture(scope="module")
def business():
    """An ow_tiny world where the ExampleInductor has run a business for 6 days (through the adapter)."""
    ad, op = open_world([tiny(days=6)], seed=0)
    run_lifetime(PolicyAgent([ExampleInductor()]), ad, OpenWorldEvaluator(op), think_s=600)
    yield op
    ad.close()


def test_operating_profit_metric(business):
    op = business
    w, me = op.world(0), op.principal(0)
    orgs = [o for o in w.principals_of(me) if o.startswith("o:")]
    assert orgs
    eco = G.operating_economics(w, orgs[0], 30 * DAY)
    assert eco["independent_revenue"] > 0 and eco["operating_costs"] > 0
    assert eco["operating_profit"] == eco["independent_revenue"] - eco["operating_costs"]
    assert "compute" in eco["costs_by_kind"] and "fee" in eco["costs_by_kind"]
    # financing / owner capital is not operating: the founder's capital transfer and the endowment never count
    assert not {"endowment", "transfer"} & set(eco["costs_by_kind"])
    # registered with Crucible's metric registry: usable in a GoalContract predicate and by GoalVerifier
    assert "cam_operating_profit" in METRICS
    pred = {"all": [{"metric": "independent_revenue", "args": {"principal": "$org", "window_s": 30 * DAY}, "op": ">=",
                     "value": 1}, {"metric": "cam_operating_profit", "args": {"principal": "$org", "window_s": 30 * DAY},
                                   "op": ">", "value": 0}]}
    assert check_predicate(pred) == []
    g = GoalContract("ow1_profit", "x", {"family": "ow_tiny", "seed": 1}, "$self", pred,
                     {"metric": "cam_operating_profit", "args": {"principal": "$org", "window_s": 30 * DAY},
                      "target": 1}, 90 * DAY, 1)
    assert g.validate() == []
    v = GoalVerifier(g, me)
    assert v.predicate(w) == (eco["operating_profit"] > 0)
    assert evaluate_metric(w, "cam_operating_profit", {"principal": orgs[0]}) == eco["operating_profit"]


def test_recovery_metrics_unit():
    series = [(d, 100.0) for d in range(10)] + [(10, 20.0), (11, 30.0), (12, 85.0), (13, 90.0), (14, 95.0)]
    m = G.recovery_metrics(series, 10.0, baseline_days=7, recover_frac=0.8, sustain=3)
    assert m["baseline"] == 100.0 and m["trough"] == 20.0 and m["drop_frac"] == pytest.approx(0.8)
    assert m["dipped"] and m["recovered"] and m["days_to_recover"] == pytest.approx(4.0)
    never = G.recovery_metrics(series[:12], 10.0)
    assert never["recovered"] is False
    flat = G.recovery_metrics([(d, 50.0) for d in range(20)], 10.0)
    assert flat["dipped"] is False and flat["recovered"] and flat["days_to_recover"] == pytest.approx(2.0)


def test_recovery_objective_injects_hidden_shock_and_measures():
    task = G.RecoveryTask(world=dict(tiny(extra=("disturb",)).world), shock={"kind": "provider_outage", "hours": 48},
                          shock_day=5.0, horizon_s=10 * DAY, endowment=3_000_000, min_baseline=50.0)
    ad, op = open_world([task], seed=1)
    ev = OpenWorldEvaluator(op)
    res = run_lifetime(PolicyAgent([ExampleInductor()]), ad, ev, think_s=600)
    st = ev.objectives[0].status(op.world(0), op.principal(0))
    ad.close()
    assert st["shock"]["done"] and st["shock"]["kind"] == "provider_outage"
    rec = st["recovery"]
    assert rec["n_pre"] >= 3 and rec["baseline"] > 50 and rec["n_post"] >= 4
    assert rec["trough"] < rec["baseline"]           # the outage bit: revenue fell
    assert res.final[0]["details"]["recovery"]["baseline"] == rec["baseline"]
    # the agent's records never mention the shock schedule (it is evaluator-only)
    assert "shock_day" not in json.dumps([g.to_dict() for g in ad.goals()])


def test_anti_trivial_self_goal_filter(business):
    op = business
    w, me = op.world(0), op.principal(0)
    org = [o for o in w.principals_of(me) if o.startswith("o:")][0]
    f = G.AntiTrivialFilter()
    cash = evaluate_metric(w, "cash", {"principal": me})
    v = f.assess(w, me, {"metric": "cash", "args": {"principal": me}, "op": ">=", "value": 0}, 3 * DAY)
    assert not v.accepted and "already_true" in v.reasons
    v = f.assess(w, me, {"metric": "cash", "args": {"principal": me}, "op": ">=", "value": cash + 1}, 3 * DAY)
    assert not v.accepted and any(r.startswith("trivial_gain") for r in v.reasons)
    v = f.assess(w, me, {"metric": "rig_net", "args": {"principal": me, "rig": "x"}, "op": ">=", "value": 5}, DAY)
    assert not v.accepted
    v = f.assess(w, me, {"all": [{"metric": "cash", "args": {"principal": me}, "op": ">=", "value": 1}]}, DAY)
    assert not v.accepted
    # the running business keeps earning: a modest revenue target is met by waiting (idle fork), so it is trivial
    rev7 = evaluate_metric(w, "independent_revenue", {"principal": org, "window_s": 7 * DAY})
    rev14 = evaluate_metric(w, "independent_revenue", {"principal": org, "window_s": 14 * DAY})
    p_wait = {"metric": "independent_revenue", "args": {"principal": org, "window_s": 14 * DAY}, "op": ">=",
              "value": int(rev14 * 1.3) + 1}
    v = f.assess(w, me, p_wait, 7 * DAY)
    assert not v.accepted and any(r.startswith("achievable_by_waiting") for r in v.reasons), v
    assert v.idle_value is not None and v.idle_value >= rev14 * 1.3
    # an ambitious target an idle world does not reach is accepted
    p_hard = {"metric": "independent_revenue", "args": {"principal": org, "window_s": 7 * DAY}, "op": ">=",
              "value": rev7 * 10 + 1_000_000}
    v = f.assess(w, me, p_hard, 7 * DAY)
    assert v.accepted, v.reasons
    # once met, the same target is not new
    f.record_met(me, p_hard)
    assert "repeats an already met goal" in f.assess(w, me, p_hard, 7 * DAY).reasons
    assert w.now == op.world(0).now                 # the counterfactual fork did not move the real world


def test_self_directed_objective_screens_kernel_self_goals():
    task = G.SelfDirectedTask(world=dict(tiny().world), horizon_s=2 * DAY, self_goal_bonus=1.0,
                              probe_goals=[tiny(days=8)], probe_every_s=DAY)
    ad, op = open_world([task], seed=2)
    ev = OpenWorldEvaluator(op)
    b = ad.reset()
    me = op.principal(0)
    r1 = ad.execute([Action("goals.propose", {"title": "stay solvent", "due_in_s": 7200,
                                              "predicate": {"metric": "cash", "args": {"principal": me}, "op": ">=",
                                                            "value": 0}})])[0]
    r2 = ad.execute([Action("goals.propose", {"title": "big", "due_in_s": 7200,
                                              "predicate": {"metric": "cash", "args": {"principal": me}, "op": ">=",
                                                            "value": 10 ** 9}})])[0]
    assert r1.ok and r2.ok
    ad.advance_time([0.2])
    assert ev.take_rewards() == [0.0]               # the trivial goal was met by the kernel but earns nothing
    st = ev.objectives[0].status(op.world(0), me)
    assert st["self_goals"] == 2 and st["accepted"] == 1 and st["accepted_met"] == 0
    assert any("already_true" in r for r in st["rejected"].values())
    assert st["probes"] and "progress" in next(iter(st["probes"][0]["probe"].values()))
    ad.close()


def test_joint_utility_requires_verified_dependency():
    ad, op = open_world([tiny()], seed=1, participants=[("partner-01", ExampleInductor(), 3_000_000)])
    try:
        ad.reset()
        ad.advance_time([1.0])                        # the independent partner launches its service
        w, me = op.world(0), op.principal(0)
        before = G.joint_utility(w, me, ["p:partner-01"], 14 * DAY)
        assert before["dependency"] is False and before["joint"] == 0.0 and before["partners_external"] >= 0
        rows = json.loads(ad.execute([Action("market.browse", {"limit": 30})])[0].observations[0].content)
        lst = next(x for x in rows if (x.get("host") or "").startswith("partner-01"))
        assert ad.execute([Action("world.describe", {"host": lst["host"]})])[0].ok
        ex = json.loads(ad.execute([Action("use.examples", {"category": lst["need"], "n": 2})])[0]
                        .observations[0].content)["examples"]
        for i in range(4):
            r = ad.execute([Action(lst["host"] + ".solve", ex[i % 2]["input"])])[0]
            assert r.ok and r.cost == lst["price_cents"]
        ad.advance_time([0.3])
        after = G.joint_utility(w, me, ["p:partner-01"], 14 * DAY)
        assert after["dependency"] and after["paid_to_partners"] == 4 * lst["price_cents"]
        assert after["joint"] == pytest.approx(max(0, after["actor_external"]) + max(0, after["partners_external"]))
        assert after["partners_external"] > 0           # the partner's third-party revenue, excluding our payments
        assert evaluate_metric(w, "cam_joint_utility", {"principal": me, "partners": ["p:partner-01"]}) == after["joint"]
    finally:
        ad.close()
