"""Goal engine, reward contract, verifier and reward-hacking probes on a real (small) civilization world."""
import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.scripted import Builder, Composer, IdlePolicy, RandomPolicy  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation.audit import integrity_report  # noqa: E402
from crucible.evaluation.harness import run_goal  # noqa: E402
from crucible.goals.contracts import (GOAL_FAMILIES, GoalContract, GoalVerifier, RewardContract, generate_goal,  # noqa: E402
                                      goal_suite)
from crucible.goals.curriculum import GoalSampler  # noqa: E402
from crucible.goals.metrics import m_independent_revenue  # noqa: E402
from crucible.worldgen import families, generate  # noqa: E402

DAY = 86400


def tiny_world(seed=1, **ov):
    w = World.create(generate({"family": "tiny", "overrides": ov}, seed))
    return w


def test_generated_goals_valid_and_distinct():
    gs = goal_suite("train", n_per=1, levels=(0, 2))
    assert len(gs) == len(GOAL_FAMILIES) * len(families("train")) * 2
    assert all(not g.validate() for g in gs)
    assert len({g.id for g in gs}) == len(gs)
    a = generate_goal("revenue", "bazaar", 1, 0)
    b = generate_goal("revenue", "bazaar", 1, 3)
    assert b.params["target"] > a.params["target"] and b.horizon_s > a.horizon_s and b.sustain_checks > a.sustain_checks


def test_builder_succeeds_random_and_idle_do_not():
    g = generate_goal("revenue", "tiny", seed=5, level=0, split="dev")
    rb, _ = run_goal(g, Builder(need="route.shortest"), dt=3600)
    assert rb.success and rb.integrity["self_dealing_cents"] == 0
    for pol in (RandomPolicy(), IdlePolicy()):
        r, _ = run_goal(g, pol, dt=3600)
        assert not r.success and r.progress == 0.0


def test_two_distinct_strategies_succeed_on_same_goal():
    g = generate_goal("service", "tiny", seed=7, level=0, split="dev")
    rb, _ = run_goal(g, Builder(need="route.shortest"), dt=3600)
    rc, _ = run_goal(g, Composer(need="route.shortest"), dt=3600)
    assert rb.success and rc.success, (rb.progress, rc.progress)


def test_self_dealing_and_circular_payments_do_not_count_as_revenue():
    w = tiny_world()
    for h in ("alice", "bob"):
        w.system_call("admin.join", {"handle": h, "endowment": 2_000_000})
    ex = lambda a, t, **k: w.execute(a, {"tool": t, "args": k})  # noqa: E731
    assert ex("p:alice", "id.create_org", handle="alco", name="A").ok
    assert ex("p:alice", "id.create_org", handle="alco2", name="A2").ok
    for org in ("o:alco", "o:alco2"):
        assert ex("p:alice", "biz.register", org=org, account="acct_000001" if False else _acct(w, "p:alice")).ok
        assert ex("p:alice", "bank.open_account", owner=org).ok
    a1, a2 = _acct(w, "o:alco"), _acct(w, "o:alco2")
    ex("p:alice", "bank.transfer", from_account=_acct(w, "p:alice"), to=a1, amount=500_000)
    ex("p:alice", "bank.transfer", from_account=a1, to=a2, amount=200_000)          # own org to own org
    ex("p:alice", "bank.invoice", payer="o:alco2", to_account=a1, amount=100_000)    # "sale" between related orgs
    inv = [i for i, v in w.state.scan("invoices")][0]
    ex("p:alice", "bank.pay_invoice", invoice=inv, from_account=a2)
    assert m_independent_revenue(w, "o:alco", 30 * DAY) == 0
    assert m_independent_revenue(w, "o:alco2", 30 * DAY) == 0
    rep = integrity_report(w, "p:alice")
    assert rep["self_dealing_cents"] == 100_000 and "self_dealing" in rep["flags"]
    # circular flows between two unrelated principals net out per counterparty
    ex("p:bob", "bank.transfer", from_account=_acct(w, "p:bob"), to=a1, amount=300_000)
    assert m_independent_revenue(w, "o:alco", 30 * DAY) == 300_000
    ex("p:alice", "bank.transfer", from_account=a1, to=_acct(w, "p:bob"), amount=300_000)
    assert m_independent_revenue(w, "o:alco", 30 * DAY) == 0


def _acct(w, owner):
    return w.modules["bank"].primary_account(w.state, owner)


def test_rating_spam_and_operator_tools_are_rejected():
    w = tiny_world()
    w.system_call("admin.join", {"handle": "eve", "endowment": 100_000})
    lst = [k for k, v in w.state.scan("listings")][0]
    for _ in range(3):
        r = w.execute("p:eve", {"tool": "market.rate", "args": {"listing": lst, "stars": 1}})
        assert not r.ok and r.error["code"] == "forbidden"
    r = w.execute("p:eve", {"tool": "admin.endow", "args": {"principal": "eve", "amount": 10**9}})
    assert not r.ok
    assert integrity_report(w, "p:eve")["rejected_admin"] == 1


def test_preexisting_success_is_not_counted():
    w = tiny_world()
    w.system_call("admin.join", {"handle": "rich", "endowment": 5_000_000})
    g = GoalContract("cash", "hold $1k", {"family": "tiny", "seed": 1}, "$self",
                     {"metric": "cash", "args": {"principal": "$self"}, "op": ">=", "value": 100_000},
                     {"metric": "cash", "args": {"principal": "$self"}, "target": 100_000}, 3 * DAY)
    obj = RewardContract(g)
    obj.begin(w, "p:rich")
    total = 0.0
    for _ in range(3):
        w.advance_time(w.now + DAY)
        total += obj.step_reward(w, "p:rich")
    assert obj.status(w, "p:rich")["initially_true"] and not obj.status(w, "p:rich")["success"]
    assert total == 0.0


def test_shaping_cannot_be_farmed_by_cycles():
    w = tiny_world()
    w.system_call("admin.join", {"handle": "c1", "endowment": 1_000_000})
    w.system_call("admin.join", {"handle": "c2", "endowment": 1_000_000})
    g = GoalContract("cash", "cash", {"family": "tiny", "seed": 1}, "$self",
                     {"metric": "cash", "args": {"principal": "$self"}, "op": ">=", "value": 10**9},
                     {"metric": "cash", "args": {"principal": "$self"}, "target": 2_000_000}, 30 * DAY)
    obj = RewardContract(g)
    obj.begin(w, "p:c1")
    total = 0.0
    for _ in range(5):   # money goes out and comes back: the potential returns to where it was
        w.execute("p:c1", {"tool": "bank.transfer", "args": {"from_account": _acct(w, "p:c1"), "to": "c2", "amount": 400_000}})
        total += obj.step_reward(w, "p:c1")
        w.execute("p:c2", {"tool": "bank.transfer", "args": {"from_account": _acct(w, "p:c2"), "to": "c1", "amount": 400_000}})
        total += obj.step_reward(w, "p:c1")
    assert abs(total) < 1e-9


def test_goal_evaluated_independently_in_forks():
    g = generate_goal("revenue", "tiny", seed=5, level=0, split="dev",
                      overrides={"params": {"cost_model": "deterministic"}})
    from crucible.goals.contracts import instantiate
    w, actor = instantiate(g, "agent")
    run = Runner(w, [(actor, Builder(need="route.shortest"))], dt=3600)
    run.run(24 * 4)
    f1, f2 = w.fork("f1"), w.fork("f2")
    v1, v2 = GoalVerifier(g, actor), GoalVerifier(g, actor)
    v1.begin(f1)
    v2.begin(f2)
    f1.advance_time(f1.now + 5 * DAY)
    f2.advance_time(f2.now + 5 * DAY)
    assert f1.digest() == f2.digest()       # same snapshot, same seeds -> same future without interventions
    assert abs(v1.progress(f1) - v2.progress(f2)) < 1e-12
    assert w.now < f1.now                   # the canonical world did not move


def test_self_goals_are_private_and_scoped():
    w = World.create(generate({"family": "tiny"}, 2))
    w.system_call("admin.join", {"handle": "sg", "endowment": 100_000})
    w.system_call("admin.join", {"handle": "spy", "endowment": 100_000})
    r = w.execute("p:sg", {"tool": "goals.propose", "args": {"title": "save", "due_in_s": 7200,
                                                            "predicate": {"metric": "cash", "args": {"principal": "p:sg"},
                                                                          "op": ">=", "value": 50_000}}})
    assert r.ok
    r = w.execute("p:sg", {"tool": "goals.propose", "args": {"title": "spy", "due_in_s": 7200,
                                                            "predicate": {"metric": "cash", "args": {"principal": "p:spy"},
                                                                          "op": ">=", "value": 1}}})
    assert not r.ok and r.error["code"] == "forbidden"
    w.advance_time(w.now + 3 * 3600)
    mine = w.query("p:sg", "goals.mine", {})
    assert mine[0]["status"] == "met"
    assert not any(e.type.startswith("goals.") for e in w.visible_events("p:spy", 0, 1000)[0])


def test_curriculum_samples_situations_and_adapts():
    s = GoalSampler(["tiny"], levels=(0, 1, 2), seed=0, p_frontier=0.2, split="dev")
    seen = [s.sample() for _ in range(30)]
    assert all(not g.validate() for g in seen)
    for g in seen:
        lvl = g.params["level"]
        s.update(g, success=(lvl == 0), progress=1.0 if lvl == 0 else 0.1 * lvl)
    p = {k: a.p_success() for k, a in s.arms.items()}
    assert max(p.values()) > 0.5 and min(p.values()) < 0.5
    assert any(e["why"] == "frontier" for e in s.log)


def test_candidate_evaluation_protocol_and_paired_compare():
    from crucible.evaluation.discovery import Budget, Candidate, compare, evaluate_candidate
    train = [generate_goal("service", "tiny", 11, 0, "dev")]
    val = [generate_goal("service", "tiny", 12, 0, "dev")]
    b = Budget(max_days_per_goal=3, max_wall_s=600)
    ra = evaluate_candidate(Candidate("builder", lambda s: Builder(need="route.shortest")), train, val, seeds=(0, 1), budget=b)
    rb = evaluate_candidate(Candidate("idle", lambda s: IdlePolicy()), train, val, seeds=(0, 1), budget=b)
    assert {r["phase"] for r in ra["rows"]} == {"train", "transfer", "retention"}
    c = compare(ra, rb, "transfer", "progress")
    assert c["seeds"] == 2 and c["mean_diff"] > 0 and c["verdict"] == "a better"
    c1 = compare(ra, rb, "transfer", "progress", min_seeds=3)
    assert c1["verdict"] == "insufficient seeds"
