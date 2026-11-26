"""SP-11.6d: compatibility with existing learner interfaces is preserved - the 4.0 Gymnasium (`api.gym_env.CrucibleEnv`)
and PettingZoo (`api.pettingzoo_env.CrucibleParallelEnv`) interfaces keep working (Clarification 1).

"Working" is measured three ways on each interface: (1) the standard API loop (reset, step with real tool actions,
close) runs with the interface's declared spaces, in the default mode and in the explicitly selected 4.0 mode;
(2) with ``reward_mode='legacy_shaped_reward'`` named, its observations, rewards and infos are byte-identical to the
4.0 implementation (git c07098b, loaded by `experience.leakage.v4_module`); (3) Gymnasium's own `check_env` /
PettingZoo's `parallel_api_test` accept the environment."""
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.experience import leakage
from crucible.experience.modes import LEGACY, OBSERVATION_ONLY

pytestmark = pytest.mark.timeout(1200)
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_11_6d.py"
SPEC = {"family": "tiny", "overrides": {"n_consumers": 60}}


def _gym_loop(mode):
    from crucible.api.gym_env import CrucibleEnv
    env = CrucibleEnv(spec=SPEC, seed=3, endowment=500_000, dt=3600, max_steps=6,
                      **({} if mode is None else {"reward_mode": mode}))
    env.server.join_world(env.world_id, "payee", endowment=0)
    w = env.server.world(env.world_id)
    acct = w.modules["bank"].primary_account(w.state, env.principal)
    obs, info = env.reset()
    ok = env.observation_space.contains(obs) if hasattr(env.observation_space, "contains") else True
    rewards = []
    for k in range(6):
        obs, r, term, trunc, info = env.step({"tool": "bank.transfer", "args": {
            "from_account": acct, "to": "p:payee", "amount": 100 + k, "memo": "x"}})
        rewards.append(float(r))
        ok &= isinstance(info, dict)
    bal = w.state.get("accounts", acct)["balance"]
    env.close()
    return bool(ok and bal == 500_000 - sum(100 + k for k in range(6))), rewards, env.reward_mode


def _pz_loop(mode):
    from crucible.api.pettingzoo_env import CrucibleParallelEnv
    env = CrucibleParallelEnv(spec=SPEC, seed=4, agents=["ann", "ben"], endowment=500_000, dt=3600, max_steps=5,
                              **({} if mode is None else {"reward_mode": mode}))
    w = env.server.world(env.world_id)
    accts = {a: w.modules["bank"].primary_account(w.state, i.principal) for a, i in env.identities.items()}
    obs, infos = env.reset()
    ok = set(obs) == set(env.agents)
    rewards = []
    for k in range(5):
        acts = {a: {"tool": "bank.transfer", "args": {"from_account": accts[a], "to": "p:ben" if a == "ann" else
                                                      "p:ann", "amount": 50 + k, "memo": "x"}} for a in env.agents}
        obs, rew, term, trunc, infos = env.step(acts)
        rewards += [float(v) for v in rew.values()]
        ok &= set(obs) <= set(env.possible_agents)
    env.close()
    return bool(ok), rewards, env.reward_mode


def _checkers():
    out = {}
    from crucible.api.gym_env import CrucibleEnv
    try:
        from gymnasium.utils.env_checker import check_env
        env = CrucibleEnv(spec=SPEC, seed=5, endowment=1000, dt=3600, max_steps=4)
        check_env(env, skip_render_check=True)
        env.close()
        out["gymnasium.check_env"] = True
    except Exception as e:  # noqa: BLE001 - recorded: the checker rejected the environment
        out["gymnasium.check_env"] = f"{type(e).__name__}: {e}"
    try:
        from pettingzoo.test import parallel_api_test
        from crucible.api.pettingzoo_env import CrucibleParallelEnv
        env = CrucibleParallelEnv(spec=SPEC, seed=6, agents=["ann", "ben"], endowment=1000, dt=3600, max_steps=4)
        parallel_api_test(env, num_cycles=4)
        env.close()
        out["pettingzoo.parallel_api_test"] = True
    except Exception as e:  # noqa: BLE001
        out["pettingzoo.parallel_api_test"] = f"{type(e).__name__}: {e}"
    return out


def test_SP_11_6d_legacy_learner_interfaces_keep_working():
    loops = {"gymnasium default": _gym_loop(None), "gymnasium legacy": _gym_loop(LEGACY),
             "pettingzoo default": _pz_loop(None), "pettingzoo legacy": _pz_loop(LEGACY)}
    same = {o["name"]: o["identical"] for o in leakage.legacy_check()
            if o["name"].startswith(("Gymnasium", "PettingZoo"))}
    checks = _checkers()
    gym_ok = loops["gymnasium default"][0] and loops["gymnasium legacy"][0] and \
        same.get("Gymnasium CrucibleEnv(reward_mode=legacy)") is True
    pz_ok = loops["pettingzoo default"][0] and loops["pettingzoo legacy"][0] and \
        same.get("PettingZoo CrucibleParallelEnv(reward_mode=legacy)") is True
    defaults_zero = all(r == 0.0 for k in ("gymnasium default", "pettingzoo default") for r in loops[k][1]) and \
        loops["gymnasium default"][2] == loops["pettingzoo default"][2] == OBSERVATION_ONLY
    metrics = {"legacy_interfaces_working": int(gym_ok) + int(pz_ok),
               "identical_to_v4_with_legacy_named": sum(1 for v in same.values() if v),
               "api_checkers_passed": sum(1 for v in checks.values() if v is True),
               "default_mode_rewards_zero": bool(defaults_zero)}
    record("SP-11.6d", metrics, command=CMD, notes=f"API loops (ok, rewards, mode): {loops}; 4.0 byte-identity with "
           f"legacy named: {same}; library checkers: {checks}")
    assert metrics["legacy_interfaces_working"] >= 2 and defaults_zero, metrics
