"""SP-11.6c: environment capabilities validated with four kinds of learner (Clarification 1) - scripted reference
agents, a small neural RL agent, a statistical learner and a lightweight language-model agent - and a reference
learner that improves over a frozen copy of itself in at least one world per tier.

Learner kinds (each must act through the interface and achieve the capability it is used for):

* scripted: `crucible.experience.scripted` founds a company and deploys a service another participant pays for;
* small neural RL: `ReferenceLearner` (recurrent network, PPO-clip) acts and updates its parameters;
* statistical: the external curiosity learner (`examples/external_learner/curiosity_learner.py`, numpy linear
  prediction models with its own prediction-error reward) acts and updates its parameters;
* lightweight language model: the frozen `agents.llm.LLMPolicy` on Qwen3-1.7B (local weights, greedy decoding) emits
  tool calls that the world carries out and whose consequence another participant observes. Needs the model in the local Hugging Face cache; runs on the GPU when the
  slot scheduler grants one, else on the CPU.

Improvement over a frozen copy (the rule of wave 3a, made non-degenerate in the wave-3b repair pass): a reference
learner is trained in worlds of the tier (observation-only, its own signal: the change of its own money as its
observation shows it); then the trained parameters and the frozen initial parameters act with their own stochastic
policies, without learning, on held-out seeds (11-14 in Tier 1; 21-24 in the kernel tiers, whose design was
developed on 1-5 and 11-14); the outcome is the evaluator's (Tier 1: the task's episode score against its private
reference; kernel tiers: the money the learner EARNED over the evaluation - the net flow of its own accounts from
authoritative ledger entries, unconditional transfers excluded: endowments and grants such as the crisis
coordinator's emergency fund - with its cash reported beside it). The trained copy must beat the frozen copy on the
mean AND on at least 3 of 4 held-out seeds (both copies follow their stochastic policy with the same sampling seed)
AND beat the do-nothing outcome the same way. Learned inaction ("stop doing harm": at 507adb24 the trained Tier 1
copy was bit-identical to doing nothing on all four seeds) and claiming a one-time grant and holding it (the Tier 2
crisis copy's $70,000 was the $10,000 endowment plus the $60,000 fund, the crisis domain's own verdict equal for the
trained, frozen and do-nothing copies) are therefore not counted as learning improvement. The learned values and the
domain's own verdict (Tier 2) are reported. A tier that cannot be built fails.

Learner and world per tier (D-68 addendum, wave 3b; the rule above is the same in every tier):

* Tier 1: `ReferenceLearner` (neural, PPO) in `mech:scm_additive`, 3000 training decisions in one world.
* Tier 2: `ToolValueLearner` (statistical: per-tool value of the observed change of its own money) in
  `crisis:storm_radial`, trained 300 decisions in each of the training seeds 1-4, evaluated 120 hourly decisions.
* Tier 3: `ToolValueLearner` in the `balanced` civilisation, decisions a simulated minute apart, trained 4000
  decisions in each of the training seeds 1-2 (about 14 tries of each of its ~580 tools), evaluated 3000 decisions
  (about two simulated days). Measured beforehand on the civilisation restriction `startup:two_formats` (190 tools,
  CPU): a frozen copy paid for compute rentals or subscriptions on 4 of 4 held-out seeds in 1000 decisions, the
  trained copy (2 x 2000 training decisions) on none.

Why not the neural learner in the kernel tiers (measured on this branch, observation-only, held-out seeds): its
near-uniform policy over 60-580 tools rarely repeats a tool, so a few thousand decisions move its tool preferences
little; in crisis it still spent the coordinator's fund (below its frozen copy on 3 of 4 held-out seeds after 1200
decisions; after 2400 better on 2 of 4 with a lower mean, $21,700 against $30,700), and
in the first wave-3b run it never moved its money at all in commercialization or the civilisation (both copies at their
endowment; part of that was a learner defect fixed since: integer arguments were sent as 2.0 and rejected). That is
an algorithmic limitation of the neural reference learner, not an environment failure. Why crisis: it is the Part 9
world whose seat's own money moves within a hundred decisions of a random policy (the coordinator's emergency fund,
paid purchases of parts, fuel, mutual aid and resilience investments); in a random 150-decision scan of the other
fourteen first configurations the seat's money moved at most twice (a $5 listing fee, $4 inquiries, a $59 sample, the
operator's accounts disappearing from view when the seat left the operator), so there is nothing for a learner of
its own money to learn from within the budget."""
import importlib.util
import json
import os

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.experience.loop import run_learner
from crucible.experience.reference import ReferenceLearner
from crucible.experience.registry import make
from crucible.experience.statistical import ToolValueLearner, describe_values

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7000)]
CMD = ("PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 7000 -- python -m pytest "
       "tests/v5/test_gate_SP_11_6c.py")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOUR = 3600
CIV = "balanced"
T2_WORLD = "crisis:storm_radial"
EVAL_SEEDS = (11, 12, 13, 14)
# the kernel tiers' design (learner, world, horizons, temperature) was developed on seeds 1-5 and 11-14 on this branch;
# their improvement is measured on seeds never used in that development
KERNEL_EVAL_SEEDS = (21, 22, 23, 24)
# Qwen3-0.6B only replied "wait" in six turns (measured on this branch); Qwen3-1.7B (LLMPolicy's default) acts
LM_MODEL = "Qwen/Qwen3-1.7B"


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
    pol = LLMPolicy(model=LM_MODEL, device="cuda" if torch.cuda.is_available() else "cpu",
                    max_new_tokens=96, interface="v4",
                    goal=f"Tell {b} by mail that the market opens at nine, then post a short note on the forum.")
    pol.reset(a, np.random.default_rng(0))
    ok, calls, tools_ok = 0, 0, []
    for _ in range(6):
        act = pol.act(env.world.observe(a), env.tools(a))
        if act is not None and act.get("tool") != "wait":
            calls += 1
            r = env.execute(a, act)
            ok += int(bool(r.get("ok")))
            if r.get("ok"):
                tools_ok.append(act["tool"])
            pol.observe_result(act, r)
        env.advance(HOUR)
    # the capability: the model's tool calls are carried out and their consequence reaches the other participant
    got = env.execute(b, {"tool": "mail.inbox", "args": {}})
    received = bool(got.get("ok")) and any(m.get("from") == a or m.get("sender") == a or a in json.dumps(m)
                                           for m in (got.get("result") or []))
    return {"validated": ok >= 1 and received, "tool_calls": calls, "ok": ok, "tools_ok": tools_ok,
            "consequence_seen_by_other": received, "device": pol.backend.device,
            "parse_failures": pol.stats["parse_failures"], "waits": pol.stats["waits"], "model": LM_MODEL}


class _Wait:
    """The do-nothing reference (every decision a wait): separates 'learned to stop doing harm' from improvement
    beyond inaction (reported, not part of the prespecified criterion)."""

    def act(self, obs, tools=None):
        return None


def _improvement(make_learner, make_env, train_seeds, train_steps: int, eval_steps: int, think_s, outcome,
                 eval_seeds=EVAL_SEEDS) -> dict:
    """Both copies act with their own stochastic policy (the policy the learner actually follows), the same sampling
    seed, no learning; only the learned parameters differ."""
    L = make_learner()
    init = L.params_blob()
    for ts in train_seeds:
        env = make_env(ts)
        L.reset(env.agents()[0])
        run_learner(env, L, train_steps, update_every=32, think_s=think_s)
        _close(env)
    trained = L.params_blob()
    per = []
    for s in eval_seeds:
        vals, extra = [], {}
        for lab, blob in (("trained", trained), ("frozen", init)):
            X = make_learner()
            X.load_params(blob)
            env = make_env(s)
            start = _start(env)
            run_learner(env, X, eval_steps, learn=False, explore=True, think_s=think_s)
            vals.append(float(outcome(env, start)))
            extra[lab] = _report(env)
            _close(env)
        env = make_env(s)
        start = _start(env)
        run_learner(env, _Wait(), eval_steps, learn=False, think_s=think_s)
        per.append({"seed": s, "trained": vals[0], "frozen": vals[1], "wait_only": float(outcome(env, start)),
                    "reported": {**extra, "wait_only": _report(env)}})
        _close(env)
    wins = sum(1 for p in per if p["trained"] > p["frozen"])
    wins_w = sum(1 for p in per if p["trained"] > p["wait_only"])
    mean_t = float(np.mean([p["trained"] for p in per]))
    mean_f = float(np.mean([p["frozen"] for p in per]))
    mean_w = float(np.mean([p["wait_only"] for p in per]))
    over_frozen = bool(mean_t > mean_f and wins >= 3)
    over_wait = bool(mean_t > mean_w and wins_w >= 3)
    out = {"improved": over_frozen and over_wait, "improved_over_frozen": over_frozen, "wins": wins,
           "wins_over_wait_only": wins_w, "trained_mean": mean_t, "frozen_mean": mean_f, "wait_only_mean": mean_w,
           "beats_wait_only": over_wait, "identical_to_wait_only": sum(1 for p in per if p["trained"] == p["wait_only"]),
           "per_seed": per, "learner": f"{L.name} ({L.kind})", "train_seeds": list(train_seeds),
           "train_steps_per_seed": train_steps, "eval_steps": eval_steps, "eval_seeds": list(eval_seeds)}
    if isinstance(L, ToolValueLearner):
        out["learned_values"] = describe_values(L, 6)
    return out


def _close(env):
    close = getattr(env, "close", None)
    if callable(close):
        close()


# ledger kinds that are unconditional transfers to the learner, not money it earned: its endowment and grants (the
# crisis coordinator's emergency fund is paid as a grant on the first call to any coordinator tool)
UNEARNED_KINDS = ("endowment", "grant")


def _ledger_seq(env) -> int:
    st = getattr(getattr(env, "world", None), "state", None)
    return len(st.tables["acct_txns"]) if st is not None and "acct_txns" in st.tables else 0


def _start(env):
    """Evaluator-side marker taken before the evaluation run (the number of ledger entries so far)."""
    return {"ledger_ids": set(env.world.state.tables["acct_txns"]) if _ledger_seq(env) else set()} \
        if getattr(env, "world", None) is not None else {}


def _earned_cents(env, start) -> float:
    """The money the learner earned over the evaluation (evaluator side, authoritative ledger): the sum of the
    ledger entries of its own accounts written during the run, except unconditional transfers (`UNEARNED_KINDS`)."""
    st = env.world.state
    a = env.agents()[0]
    if "acct_txns" not in st.tables:
        return 0.0
    mine = set(st.find_ids("accounts", "owner", a))
    seen = start.get("ledger_ids", set())
    total = 0
    for lid, e in st.tables["acct_txns"].items():
        if lid in seen or e.get("acct") not in mine:
            continue
        if st.tables["txns"][e["txn"]]["kind"] in UNEARNED_KINDS:
            continue
        total += int(e["amount"])
    return float(total)


def _report(env) -> dict:
    """Reported beside the outcome (not part of the criterion): the learner's cash and, in a Tier 2 world, the
    domain's own verdict (score, success)."""
    out = {}
    v = env.verify()
    a = env.agents()[0] if hasattr(env, "agents") else None
    m = (v.get("measures") or {}).get(a) or {}
    if "cash_cents" in m:
        out["cash_cents"] = m["cash_cents"]
    ev = (v.get("evaluation") or {}).get(a)
    if isinstance(ev, dict):
        out["domain_verdict"] = {"score": ev.get("score"), "success": ev.get("success")}
    return out


def _tier1():
    def mk(seed):
        return make(1, "mech:scm_additive", seed, validate=True)

    def outcome(env, start=None):
        eps = env.verify()["episodes"]
        return float(np.mean([e["score"] for e in eps])) if eps else float("-inf")
    return _improvement(lambda: ReferenceLearner(seed=7, reward_mode="observation_only"), mk, (5,), 3000, 300, None,
                        outcome)


def _kernel_outcome(env, start):
    return _earned_cents(env, start)


def _statistical_learner():
    return ToolValueLearner(seed=7, reward_mode="observation_only")


def _tier2():
    return _improvement(_statistical_learner, lambda s: make(2, T2_WORLD, s), (1, 2, 3, 4), 300, 120, HOUR,
                        _kernel_outcome, KERNEL_EVAL_SEEDS)


def _tier3():
    # decisions a simulated minute apart: a person's money in a civilisation moves rarely (a purchase, a fee), so the
    # copies need many decisions to show it; a civilisation day costs ~20 s on the GPU text path, a decision little
    return _improvement(_statistical_learner, lambda s: make(3, CIV, s, agents=["learner"]), (1, 2), 4000, 3000,
                        60, _kernel_outcome, KERNEL_EVAL_SEEDS)


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
               "tiers_beating_wait_only": sum(1 for v in imp.values() if v["beats_wait_only"]),
               "tiers_improved_over_frozen_only": sum(1 for v in imp.values()
                                                      if v["improved_over_frozen"] and not v["beats_wait_only"])}
    record("SP-11.6c", metrics, command=CMD, labels={"scripted": True, "learned": True},
           notes=f"learner kinds: {kinds}; could not run: {kind_err}; improvement over a frozen copy (trained vs "
                 f"initial parameters - Tier 1 the neural ReferenceLearner, Tiers 2 and 3 the statistical "
                 f"ToolValueLearner -, stochastic policies with one sampling seed, no learning, held-out seeds "
                 f"{EVAL_SEEDS} (Tier 1) / {KERNEL_EVAL_SEEDS} (Tiers 2-3); kernel-tier outcome: money earned over "
                 f"the evaluation, endowments and grants excluded; a tier counts when the trained copy beats the "
                 f"frozen copy AND the do-nothing outcome on the mean and on >= 3 of 4 seeds): "
                 f"{imp}; tiers that could not be measured: {imp_err}")
    assert metrics["learner_kinds_validated"] == 4, (kinds, kind_err)
    assert metrics["tiers_with_learning_improvement"] == 3, (imp, imp_err)
