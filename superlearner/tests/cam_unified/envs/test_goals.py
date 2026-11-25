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


def test_joint_utility_gives_nothing_for_payments_and_independent_business():
    """The earlier loophole: the partner's whole independent revenue counted once the actor had paid it. Direct paid
    calls by the actor are an exchange (paid_to > 0) but produce no joint third-party value, and neither side's
    independent business counts."""
    ad, op = open_world([tiny()], seed=1, participants=[("partner-01", ExampleInductor(), 3_000_000)])
    try:
        ad.reset()
        ad.advance_time([1.0])                        # the independent partner launches its service
        w, me = op.world(0), op.principal(0)
        before = G.joint_utility(w, me, ["p:partner-01"], 14 * DAY)
        assert before["exchange"] is False and before["joint"] == 0.0
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
        assert after["exchange"] and after["paid_to_partners"] == 4 * lst["price_cents"]
        assert after["partners_external"] > 0           # the partner earns on its own ... and it does not count
        assert after["joint"] == 0.0 and after["dependency"] is False and after["actor_dependent_revenue"] == 0.0
        # the index booked the four direct calls as the partner's revenue from the actor (a transfer), call-linked
        # to nothing
        idx = G.coop_index(w)
        mine = [(k, c) for k, c in idx.rev.items() if k[3] == me]          # rows aggregate per (t, ..., callees)
        assert mine and sum(c for _, c in mine) == 4 * lst["price_cents"]
        assert all(k[1] == lst["host"] and k[4] == frozenset() for k, _ in mine)
        assert evaluate_metric(w, "cam_joint_utility", {"principal": me, "partners": ["p:partner-01"]}) == 0.0
        assert evaluate_metric(w, "cam_partner_dependence", {"principal": me, "partners": ["p:partner-01"]}) == 0.0
    finally:
        ad.close()


@pytest.fixture(scope="module")
def relay():
    """ow_tiny with the OW-7 partner; the actor runs the RelayIntegrator control (its service calls the partner's
    service inside every request) for 9 days."""
    from superlearner.cam_unified.envs import protocols as P
    p = P.load("ow7")
    task = p.make_task("ow_tiny", 1)                 # carries its participants (the partner)
    ad, op = open_world([task], seed=0)
    run_lifetime(PolicyAgent([p.controls[0].factory()]), ad, None, think_s=600, days=9)
    yield ad, op
    ad.close()


def test_joint_utility_is_call_linked_relay_value(relay):
    ad, op = relay
    w, me = op.world(0), op.principal(0)
    org = [o for o in w.principals_of(me) if o.startswith("o:")][0]
    ju = G.joint_utility(w, org, ["p:partner-01"], 14 * DAY)
    assert ju["actor_hosts"] and ju["partner_hosts"] and ju["exchange"]
    relay_host = ju["actor_hosts"][0]
    # every relay request calls the partner: its third-party revenue is all call-linked
    assert ju["actor_dependent_revenue"] > 10_000
    assert ju["actor_dependent_revenue"] == pytest.approx(ju["actor_service_revenue"])
    assert ju["joint"] == pytest.approx(min(ju["actor_dependent_revenue"], ju["actor_external"]))
    assert ju["joint_partner_side"] == 0.0           # the partner's own business never used the actor's service
    assert ju["partners_external"] > 0 and ju["joint"] < ju["actor_dependent_revenue"] + ju["partners_external"]
    assert ju["partner_call_reliability"] == 1.0 and ju["partner_calls_ok"] > 100
    # cross-checks of the index against the kernel's own tables / ledger (independent bookkeeping):
    st = w.state
    idx = G.coop_index(w)
    # (1) executed relay requests that called the partner == successful calls the partner recorded from the relay
    svc = next(d["svc"] for d in st.tables["deployments"].values() if d["host"] == relay_host)
    used = sum(r["ok"] for r in st.tables["use_calls"].values() if r["caller"] == svc)
    assert used == ju["partner_calls_ok"]
    # (2) household money booked for the relay host == the ledger's household sales to it
    ledger = sum(-d for _, tx in st.tables["txns"].items() if tx.get("ref") == f"svc:{relay_host}"
                 and tx["kind"] == "sale" for a, d in tx["postings"] if a == "acct_households")
    booked = sum(c for k, c in idx.rev.items() if k[1] == relay_host and k[3] == "households")
    assert booked == pytest.approx(ledger) and ledger > 0
    # (3) the relay's payments to the partner are booked as the partner's revenue from the relay service
    paid = sum(c for k, c in idx.rev.items() if k[3] == svc)
    assert paid == pytest.approx(ju["paid_to_partners"])
    assert evaluate_metric(w, "cam_joint_utility", {"principal": org, "partners": ["p:partner-01"]}) == ju["joint"]


def test_partner_dependence_audit_in_forks(relay):
    ad, op = relay
    w, me = op.world(0), op.principal(0)
    org = [o for o in w.principals_of(me) if o.startswith("o:")][0]
    now, seq = w.now, w.seq
    pd = G.partner_dependence(w, org, ["p:partner-01"])
    assert w.now == now and w.seq == seq                  # the forks never touched the real world
    assert pd["audited_hosts"] and pd["partner_hosts_stopped"]
    assert pd["with_partners"] > 0 and pd["without_partners"] < 0.1 * pd["with_partners"]
    assert pd["dependence"] > 0.9
    assert G.partner_dependence(w, org, ["p:partner-01"]) is pd          # cached for this state
    # the partner's own business does not depend on the actor
    rev = G.partner_dependence(w, "o:partner-01-labs", [me])
    assert rev["dependence"] == 0.0 and rev["audited_hosts"] == []


def test_coop_index_survives_checkpoint_restore(tmp_path):
    """A restored world's event log starts at the snapshot: the call-linked index is carried by the operator
    checkpoint and continues from there (operator checkpoints do not support scripted participants, so this runs an
    own business without partners)."""
    ad, op = open_world([tiny(days=5)], seed=0)
    try:
        run_lifetime(PolicyAgent([ExampleInductor()]), ad, None, think_s=600, days=3)
        w = op.world(0)
        idx = G.coop_index(w)
        rows = dict(idx.rev)
        assert rows and any(k[3] == "households" for k in rows)        # the business's household billing
        path = str(tmp_path / "op.pkl")
        op.checkpoint(path)
        op.restore(path)
        w2 = op.world(0)
        assert w2 is not w and w2.log_base == w2.seq          # a new kernel world whose log starts at the snapshot
        idx2 = G.coop_index(w2)
        assert idx2.rev == rows and idx2.seq == w2.seq and idx2.start_t == 0
        ad.advance_time([0.5])                                 # ... and it continues on the restored world
        idx3 = G.coop_index(op.world(0))
        assert idx3.seq == op.world(0).seq > w2.log_base and len(idx3.rev) > len(rows)
        assert all(idx3.rev[k] == v for k, v in rows.items())
        # without the persisted index the history before the snapshot is invisible (rebuilt from the log base)
        del op.world(0).runtime[G.COOP_KEY]
        blind = G.coop_index(op.world(0))
        assert blind.start_seq == op.world(0).log_base and not set(rows) & set(blind.rev)
    finally:
        ad.close()
