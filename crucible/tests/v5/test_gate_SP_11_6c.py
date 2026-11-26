"""SP-11.6c: environment capabilities validated with four kinds of learner (Clarification 1) - scripted reference
agents, a small neural RL agent, a statistical learner and a lightweight language-model agent - and a reference
learner that improves over a frozen copy of itself in at least one world per tier.

Learner kinds (each must act through the interface and achieve the capability it is used for):

* scripted: `crucible.experience.scripted` founds a company and deploys a service another participant pays for;
* small neural RL: `ReferenceLearner` (recurrent network, PPO-clip) acts and updates its parameters;
* statistical: the external curiosity learner (`examples/external_learner/curiosity_learner.py`, numpy linear
  prediction models with its own prediction-error reward) acts and updates its parameters;
* lightweight language model: the frozen `agents.llm.LLMPolicy` on Qwen3-0.6B (local weights, greedy decoding) emits
  tool calls that the world carries out. Needs the model in the local Hugging Face cache; runs on the GPU when the
  slot scheduler grants one, else on the CPU.

Improvement over a frozen copy (prespecified before running): a `ReferenceLearner` is trained in one world of the tier
(observation-only, its own signal); then the trained weights and the frozen initial weights act with their own stochastic policies, without
learning, on held-out seeds; the outcome is the evaluator's (Tier 1: the task's episode score against its private
reference; kernel tiers: the evaluator's cash measure of the learner). Improvement = the trained copy beats the frozen
copy on the mean AND on at least 3 of 4 held-out seeds (both copies follow their stochastic policy with the same
sampling seed). The do-nothing outcome is reported beside it, so that "learned to stop doing harm" is not mistaken
for learning the mechanism. A tier that cannot be built fails."""
import importlib.util
import os

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.experience.loop import run_learner
from crucible.experience.reference import ReferenceLearner
from crucible.experience.registry import make

pytestmark = [pytest.mark.slow, pytest.mark.timeout(5400)]
CMD = ("PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 4000 -- python -m pytest "
       "tests/v5/test_gate_SP_11_6c.py")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOUR = 3600
CIV = "balanced"
EVAL_SEEDS = (11, 12, 13, 14)


def _attempt(fn):
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001 - recorded and asserted on
        return None, f"{type(e).__name__}: {str(e)[:400]}"


def _scripted() -> dict:
    from crucible.experience.scripted import call, deploy_service, discover, need_input, personal_account
    env = make(0, "tiny", 2, agents=["builder", "client"], endowment=3_000_000, overrides={"n_consumers": 60})
    b, c = env.agents()
    svc = deploy_service(env, b, handle="refco", host="refsvc", price=6)
    discover(env, c, "refsvc")
    acct = personal_account(env, c)
    before = [a for a in call(env, c, "bank.accounts") if a["account"] == acct][0]["balance"]
    r = env.execute(c, {"tool": "refsvc.solve", "args": need_input("sort.records", 2)})
    after = [a for a in call(env, c, "bank.accounts") if a["account"] == acct][0]["balance"]
    return {"validated": bool(r.get("ok") and before - after == svc["price"]), "paid_cents": int(before - after)}


def _neural() -> dict:
    env = make(1, "rapid:net_routing", 2)
    L = ReferenceLearner(seed=1, reward_mode=env.reward_mode)
    s = run_learner(env, L, 160, update_every=40)
    return {"validated": s["actions"] >= 100 and s["params_changed_updates"] >= 1, "actions": s["actions"],
            "param_changes": s["params_changed_updates"]}


def _statistical() -> dict:
    spec = importlib.util.spec_from_file_location("curiosity_learner", os.path.join(
        ROOT, "examples", "external_learner", "curiosity_learner.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = make(1, "mech:scm_additive", 2, validate=False)
    r = mod.run(env, mod.CuriosityLearner(seed=2), 120, update_every=20)
    return {"validated": r["actions"] >= 100 and r["param_updates_changed"] >= 1, "actions": r["actions"],
            "param_changes": r["param_updates_changed"], "own_reward": round(r["own_reward_sum"], 4)}


def _language_model() -> dict:
    import torch
    from crucible.agents.llm import LLMPolicy
    env = make(0, "tiny", 3, agents=["writer", "reader"], endowment=500_000, overrides={"n_consumers": 40})
    a, b = env.agents()
    pol = LLMPolicy(model="Qwen/Qwen3-0.6B", device="cuda" if torch.cuda.is_available() else "cpu",
                    max_new_tokens=96, interface="v4",
                    goal=f"Tell {b} by mail that the market opens at nine, then post a short note on the forum.")
    pol.reset(a, np.random.default_rng(0))
    ok, calls = 0, 0
    for _ in range(6):
        act = pol.act(env.world.observe(a), env.tools(a))
        if act is not None and act.get("tool") != "wait":
            calls += 1
            r = env.execute(a, act)
            ok += int(bool(r.get("ok")))
            pol.observe_result(act, r)
        env.advance(HOUR)
    return {"validated": ok >= 2, "tool_calls": calls, "ok": ok, "device": pol.backend.device,
            "parse_failures": pol.stats["parse_failures"], "model": "Qwen/Qwen3-0.6B"}


class _Wait:
    """The do-nothing reference (every decision a wait): separates 'learned to stop doing harm' from improvement
    beyond inaction (reported, not part of the prespecified criterion)."""

    def act(self, obs, tools=None):
        return None


def _improvement(make_env, train_steps: int, eval_steps: int, think_s, outcome) -> dict:
    """Both copies act with their own stochastic policy (the policy the learner actually follows), the same sampling
    seed, no learning; only the weights differ."""
    L = ReferenceLearner(seed=7, reward_mode="observation_only")
    init = L.params_blob()
    run_learner(make_env(5), L, train_steps, update_every=32, think_s=think_s)
    trained = L.params_blob()
    per = []
    for s in EVAL_SEEDS:
        vals = []
        for blob in (trained, init):
            X = ReferenceLearner(seed=7, reward_mode="observation_only")
            X.load_params(blob)
            env = make_env(s)
            run_learner(env, X, eval_steps, learn=False, explore=True, think_s=think_s)
            vals.append(float(outcome(env)))
        env = make_env(s)
        run_learner(env, _Wait(), eval_steps, learn=False, think_s=think_s)
        per.append({"seed": s, "trained": vals[0], "frozen": vals[1], "wait_only": float(outcome(env))})
    wins = sum(1 for p in per if p["trained"] > p["frozen"])
    mean_t = float(np.mean([p["trained"] for p in per]))
    mean_f = float(np.mean([p["frozen"] for p in per]))
    mean_w = float(np.mean([p["wait_only"] for p in per]))
    return {"improved": bool(mean_t > mean_f and wins >= 3), "wins": wins, "trained_mean": mean_t,
            "frozen_mean": mean_f, "wait_only_mean": mean_w, "beats_wait_only": bool(mean_t > mean_w),
            "per_seed": per}


def _tier1():
    def mk(seed):
        return make(1, "mech:scm_additive", seed, validate=True)

    def outcome(env):
        eps = env.verify()["episodes"]
        return float(np.mean([e["score"] for e in eps])) if eps else float("-inf")
    return _improvement(mk, 3000, 300, None, outcome)


def _kernel_outcome(env):
    a = env.agents()[0]
    return env.verify()["measures"][a]["cash_cents"]


def _tier2():
    from crucible.experience.probes_tiers import tier2_names
    name = tier2_names()[0]
    return _improvement(lambda s: make(2, name, s), 400, 120, HOUR, _kernel_outcome)


def _tier3():
    # a civilisation day costs minutes of CPU: 120 training hours, 24 evaluation hours per copy and seed
    return _improvement(lambda s: make(3, CIV, s, agents=["learner"]), 120, 24, HOUR, _kernel_outcome)


def test_SP_11_6c_learner_kinds_and_improvement_over_frozen_copies():
    kinds, kind_err = {}, {}
    for name, fn in (("scripted", _scripted), ("small_neural_rl", _neural), ("statistical", _statistical),
                     ("language_model", _language_model)):
        r, err = _attempt(fn)
        if err:
            kind_err[name] = err
        else:
            kinds[name] = r
    imp, imp_err = {}, {}
    for t, fn in ((1, _tier1), (2, _tier2), (3, _tier3)):
        r, err = _attempt(fn)
        if err:
            imp_err[t] = err
        else:
            imp[t] = r
    metrics = {"learner_kinds_validated": sum(1 for v in kinds.values() if v["validated"]),
               "tiers_with_learning_improvement": sum(1 for v in imp.values() if v["improved"]),
               "tiers_measured_for_improvement": len(imp),
               "tiers_beating_wait_only": sum(1 for v in imp.values() if v["beats_wait_only"])}
    record("SP-11.6c", metrics, command=CMD, labels={"scripted": True, "learned": True},
           notes=f"learner kinds: {kinds}; could not run: {kind_err}; improvement over a frozen copy (trained vs "
                 f"initial weights, greedy, no learning, held-out seeds {EVAL_SEEDS}; prespecified: mean better and "
                 f">= 3 of 4 seeds): {imp}; tiers that could not be measured: {imp_err}")
    assert metrics["learner_kinds_validated"] == 4, (kinds, kind_err)
    assert metrics["tiers_with_learning_improvement"] == 3, (imp, imp_err)
