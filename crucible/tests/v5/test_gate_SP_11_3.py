"""SP-11.3: the reward modes legacy_shaped_reward, verified_terminal_outcome and observation_only.

A mode counts when its defining behaviour is verified on at least two different adapters with the same inputs:

* legacy: byte-identical to the 4.0 implementation (git c07098b) and nonzero shaped rewards;
* verified terminal outcome: nothing before the independent verifier confirms the objective, exactly 1.0 once
  when it does, only `success`/`failed` labels in infos;
* observation-only: no reward on any path (the leakage battery), verifier still active.
"""
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.experience import REWARD_MODES, leakage
from crucible.experience.kernel import KernelExperienceEnv

pytestmark = pytest.mark.timeout(900)
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_11_3.py"
TARGET = 30_000


def _income_kernel(mode: str, days: int = 4) -> dict:
    """The learner starts with $100 and is paid $50 a day by another participant; its goal (evaluator side: cash >=
    $300 at a daily check) becomes true on day 4. Rewards per day in `mode`."""
    env = KernelExperienceEnv({"family": "tiny", "overrides": {"n_consumers": 100}}, 9, agents=["learner"],
                              endowment=10_000, reward_mode=mode,
                              objectives=leakage.cash_goal(target=TARGET))
    w = env.world
    payer = w.system_call("admin.join", {"handle": "payer", "endowment": 1_000_000})["principal"]
    acct = w.modules["bank"].primary_account(w.state, payer)
    a = env.agents()[0]
    rewards = []
    for _ in range(days + 2):
        r = w.execute(payer, {"tool": "bank.transfer", "args": {"from_account": acct, "to": a, "amount": 5000,
                                                               "memo": "daily fee"}})
        assert r.ok, r.error
        env.advance(86400)
        rewards.append(env.reward(a))
    return {"rewards": rewards, "status": env.verify()["agents"][a]}


def _income_gym(mode: str, steps: int = 8):
    from crucible.api.gym_env import CrucibleEnv
    from crucible.goals.contracts import RewardContract
    payer = {}

    def driver(server, wid):
        w = server.world(wid)
        w.execute(payer["p"], {"tool": "bank.transfer", "args": {"from_account": payer["a"], "to": "p:agent",
                                                                 "amount": 5000, "memo": "daily fee"}})
    env = CrucibleEnv(spec={"family": "tiny", "overrides": {"n_consumers": 100}}, seed=9, endowment=10_000,
                      objective=RewardContract(leakage.cash_goal(target=TARGET)), dt=86400, max_steps=steps,
                      driver=driver, reward_mode=mode)
    ident = env.server.join_world(env.world_id, "payer", endowment=1_000_000)
    w = env.server.world(env.world_id)
    payer.update(p=ident.principal, a=w.modules["bank"].primary_account(w.state, ident.principal))
    env.reset()
    rewards, infos, terms = [], [], []
    for _ in range(steps):
        _, r, term, trunc, info = env.step({"tool": "bank.accounts", "args": {}})
        rewards.append(r)
        infos.append(info)
        terms.append(term)
        if term or trunc:
            break
    out = {"rewards": rewards, "infos": infos, "terms": terms, "verify": env.verify()}
    env.close()
    return out


def test_SP_11_3_reward_modes():
    verified_modes = {}
    # legacy: the 4.0 contract unchanged, and it pays
    legacy = leakage.legacy_check(n_steps=20)
    k_leg, g_leg = _income_kernel("legacy_shaped_reward"), _income_gym("legacy_shaped_reward")
    legacy_ok = [all(o["identical"] for o in legacy), any(r != 0 for r in k_leg["rewards"]),
                 any(r != 0 for r in g_leg["rewards"])]
    verified_modes["legacy_shaped_reward"] = all(legacy_ok)
    # verified terminal outcome
    k_ver, g_ver = _income_kernel("verified_terminal_outcome"), _income_gym("verified_terminal_outcome")
    mech = leakage.test_verified_terminal()
    k_ok = sorted(k_ver["rewards"]).count(1.0) == 1 and sum(k_ver["rewards"]) == 1.0 and k_ver["status"]["success"]
    g_labels = all(set((i.get("objective") or {}).keys()) <= {"success", "failed"} for i in g_ver["infos"])
    g_ok = sum(g_ver["rewards"]) == 1.0 and g_ver["rewards"][-1] == 1.0 and g_ver["terms"][-1] and g_labels
    verified_modes["verified_terminal_outcome"] = bool(k_ok and g_ok and mech["passed"])
    # observation-only
    k_obs, g_obs = _income_kernel("observation_only"), _income_gym("observation_only")
    battery = leakage.run_battery((leakage.test_mechvec, leakage.test_bridge, leakage.test_selfplay,
                                   leakage.test_gym, leakage.test_pettingzoo, leakage.test_worldenv))
    o_ok = all(r == 0.0 for r in k_obs["rewards"] + g_obs["rewards"]) and k_obs["status"]["success"] \
        and g_obs["verify"].get("success") and all("objective" not in i for i in g_obs["infos"]) \
        and all(r["passed"] for r in battery)
    verified_modes["observation_only"] = bool(o_ok)
    metrics = {"reward_modes": sum(1 for m in REWARD_MODES if verified_modes.get(m)),
               "legacy_runs_identical_to_4_0": sum(1 for o in legacy if o["identical"]),
               "kernel_rewards_legacy_sum": round(sum(k_leg["rewards"]), 6),
               "kernel_rewards_verified_sum": sum(k_ver["rewards"]), "kernel_rewards_observation_only_sum":
               sum(k_obs["rewards"]), "gym_rewards_verified_sum": sum(g_ver["rewards"]),
               "gym_rewards_observation_only_sum": sum(g_obs["rewards"])}
    record("SP-11.3", metrics, command=CMD, notes=f"modes verified: {verified_modes}; same inputs in every mode "
           "(a participant pays the learner $50 a day; the evaluator's goal is cash >= $300); kernel rewards per day: "
           f"legacy {k_leg['rewards']}, verified {k_ver['rewards']}, observation-only {k_obs['rewards']}")
    assert metrics["reward_modes"] >= 3, verified_modes
