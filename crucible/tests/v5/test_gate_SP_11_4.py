"""SP-11.4: self-generated experiments - a testable prediction, a measurable outcome, a deadline, an authorised
experiment, retrieval of the actual result - with evidence returned and no reward for easy subgoals.

The `exp` module (crucible.experience.experiments) measures any quantity the agent may read through a read-only tool,
as the agent. The same action sequence is run with and without self-experiments (an easy one, an informative one, a
wrong one, and a 4.0 `goals.propose` self-goal) in all three reward modes: rewards must not differ, and nothing in
the experiments' transactions may move money."""
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.experience import REWARD_MODES, leakage
from crucible.experience.kernel import KernelExperienceEnv

pytestmark = pytest.mark.timeout(900)
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_11_4.py"
BAL = {"tool": "bank.accounts", "where": {"owner": "$self"}, "field": "balance", "aggregate": "sum"}


def _run(mode: str, experiments: bool) -> dict:
    env = KernelExperienceEnv({"family": "tiny", "overrides": {"n_consumers": 100}}, 12, agents=["learner"],
                              endowment=50_000, reward_mode=mode, objectives=leakage.cash_goal(target=40_000))
    w = env.world
    a = env.agents()[0]
    other = w.system_call("admin.join", {"handle": "shop", "endowment": 0})["principal"]
    acct = w.modules["bank"].primary_account(w.state, a)
    ids, rewards = [], []
    if experiments:
        for hyp, pred, due in (("my balance stays non-negative for an hour (easy)", {"op": ">=", "value": 0}, 3600),
                               ("two 40-dollar orders in two days lower my balance by more than 50 dollars",
                                {"op": "decrease", "min_change": 5_000}, 2 * 86400),
                               ("my balance will grow (it will not: I only pay)", {"op": "increase"}, 3 * 86400)):
            r = env.execute(a, {"tool": "exp.register", "args": {"hypothesis": hyp, "measure": BAL,
                                                                 "prediction": pred, "deadline_in_s": due,
                                                                 "samples": 1}})
            assert r["ok"], r
            ids.append(r["result"]["experiment"])
        r = env.execute(a, {"tool": "goals.propose", "args": {
            "title": "keep some cash", "predicate": {"metric": "cash", "args": {"principal": a}, "op": ">=", "value": 1},
            "due_in_s": 86400}})
        assert r["ok"], r
    rewards.append(env.reward(a))
    for day in range(4):
        r = env.execute(a, {"tool": "bank.transfer", "args": {"from_account": acct, "to": other, "amount": 4000,
                                                              "memo": f"order {day}"}})
        assert r["ok"], r
        env.advance(86400)
        rewards.append(env.reward(a))
    evidence = [env.execute(a, {"tool": "exp.result", "args": {"experiment": i}}) for i in ids]
    obs = env.observe(a)
    # money moved inside experiment transactions (must be none): bank events in exp.* transactions
    exp_money = 0
    tx_is_exp = False
    for e in w.events_range(0):
        if e.type in ("cmd", "proc", "time", "cmd.rejected"):
            tool = e.payload.get("tool") if e.type == "cmd" else e.payload.get("proc") if e.type == "proc" else ""
            tx_is_exp = str(tool or "").startswith("exp.")
        elif tx_is_exp and e.type.startswith("bank."):
            exp_money += 1
    return {"rewards": rewards, "evidence": evidence, "obs": obs, "exp_money_events": exp_money,
            "status": env.verify()["agents"][a]}


def test_SP_11_4_self_generated_experiments():
    runs = {(m, x): _run(m, x) for m in REWARD_MODES for x in (False, True)}
    roundtrips = 0
    details = []
    for m in REWARD_MODES:
        for ev in runs[(m, True)]["evidence"]:
            assert ev["ok"], ev
            g = ev["result"]
            complete = g["status"] == "resolved" and g["measured"] is not None and g["measured_t"] >= g["due"] \
                and g["held"] is not None and g["baseline"] is not None and len(g["samples"]) == 1
            roundtrips += int(complete)
            if m == "observation_only":
                details.append((g["hypothesis"][:40], g["baseline"], g["measured"], g["held"], g["actions_during"]))
    paid = 0
    for m in REWARD_MODES:
        with_e, without = runs[(m, True)]["rewards"], runs[(m, False)]["rewards"]
        paid += sum(1 for a, b in zip(with_e, without) if a != b)
    paid += sum(r["exp_money_events"] for r in runs.values())
    held = [d[3] for d in details]
    informative = held == [True, True, False]
    nonzero_legacy = sum(1 for r in runs[("legacy_shaped_reward", False)]["rewards"] if r != 0)
    metrics = {"self_experiment_roundtrips": roundtrips, "self_goal_rewards_paid": paid,
               "outcomes_match_world": bool(informative), "legacy_reward_steps_same_actions": nonzero_legacy,
               "modes_compared": len(REWARD_MODES)}
    record("SP-11.4", metrics, command=CMD, notes=f"evidence (hypothesis, baseline, measured, held, actions during): "
           f"{details}; rewards per step with/without experiments identical in every mode; the legacy reward is "
           "nonzero on the same actions (the cash objective moves), so equality is not trivially 0 = 0")
    assert roundtrips >= 1 and paid == 0 and informative, metrics
