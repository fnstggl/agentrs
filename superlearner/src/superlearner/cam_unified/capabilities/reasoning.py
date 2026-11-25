"""A03 through the integrated agent: do latent reasoning iterations change and improve actual decisions? (spec Part 4)

Protocol (`run_reasoning`): `ChainEnv` (follow arrows k times through a random colour permutation; k in `hops`),
one decision per episode, reward 1 for the right button. The full agent (real tokenizer over Qwen3-0.6B, core,
Reasoner, BeliefModule, action generator, value head) is trained with its own policy objective (PPO on verified
returns through `runtime.run_steps` + D's CombinedTrainer) under a reasoning condition:

* "adaptive" - B's ACT halting (cfg.adaptive_reasoning, compute penalty cfg.reason_compute_cost);
* "fixed<J>" - J iterations forced in training and acting (`CamUnifiedModel(reason_iters=J)`);
* "none" - cfg.use_reasoning = False (R = Z).

Evaluation on development seeds (offset 100,000+), greedy decisions of `CamUnifiedModel.decide` (the decisions the
agent executes): accuracy per hop count under the trained condition and with the iteration count forced at test time
(J in 1, 2, 4, 8, 16 on the same weights), the fraction of decisions that change when J is forced, and the iterations
the adaptive halting policy actually computes per hop count. The tool pointer reads R, so any change of the decision
under a forced J is caused by the reasoning state."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from .envs import ChainEnv, ScoreEvaluator
from .loop import Episode, mean, train_agent
from .stack import build_stack, capability_config


@dataclass
class ReasoningSettings:
    seed: int = 0
    condition: str = "adaptive"          # adaptive | fixed<J> | none
    B: int = 64
    train_rounds: int = 200
    lr: float = 1e-3
    n_colours: int = 4
    hops: tuple = (1, 2)
    eval_rows: int = 256
    eval_seed0: int = 100_000
    test_iters: tuple = (1, 2, 4, 8, 16)
    cfg_overrides: dict = field(default_factory=dict)


def _condition(settings: ReasoningSettings) -> tuple:
    c = settings.condition
    if c == "adaptive":
        return {}, None
    if c == "none":
        return {"use_reasoning": False}, None
    if c.startswith("fixed"):
        return {"adaptive_reasoning": False}, int(c[5:])
    raise ValueError(f"unknown reasoning condition {c!r}")


def evaluate_chain(model, settings: ReasoningSettings, reason_iters="trained") -> dict:
    """Greedy decisions on `eval_rows` development episodes. reason_iters: "trained" (the model's own setting) or
    an int J forced at test time. Returns accuracy per hop, iterations computed per hop and the chosen tools."""
    env = ChainEnv(settings.eval_rows, n_colours=settings.n_colours, hops=settings.hops)
    ev = ScoreEvaluator(env)
    old = model.reason_iters
    if reason_iters != "trained":
        model.reason_iters = int(reason_iters)
    try:
        ep = Episode(model, env, ev, [settings.eval_seed0 + b for b in range(settings.eval_rows)], greedy=True)
        rec = ep.step()
    finally:
        model.reason_iters = old
    hops = [h["k"] for h in env.hidden]
    iters = [r["iters"] for r in rec.decision.reasoning]
    out = {"tools": [a.tool for a in rec.decision.actions], "acc": mean(rec.rewards), "per_hop": {}}
    for k in sorted(set(hops)):
        rows = [b for b, h in enumerate(hops) if h == k]
        out["per_hop"][k] = {"n": len(rows), "acc": mean(rec.rewards[b] for b in rows),
                             "iters": mean(iters[b] for b in rows)}
    return out


def run_reasoning(settings: Optional[ReasoningSettings] = None, device=None, log_every: int = 0) -> dict:
    st = settings or ReasoningSettings()
    t0 = time.time()
    over, J = _condition(st)
    cfg = capability_config(use_imagination=False, use_memory=False, lr=st.lr, **{**over, **st.cfg_overrides})
    stack = build_stack(cfg, device=device, seed=st.seed, model_kwargs={"reason_iters": J})
    model = stack.model
    env = ChainEnv(st.B, n_colours=st.n_colours, hops=st.hops)
    ev = ScoreEvaluator(env)
    before = evaluate_chain(model, st)
    train = train_agent(stack, env, ev, st.train_rounds, update_every=1,
                        seeds=[st.seed * 10_007 + b for b in range(st.B)], log_every=log_every)
    after = evaluate_chain(model, st)
    forced = {}
    if cfg.use_reasoning:
        for j in st.test_iters:
            r = evaluate_chain(model, st, j)
            r["changed_vs_trained"] = mean(float(a != b) for a, b in zip(r["tools"], after["tools"]))
            forced[j] = r
    chance = 1.0 / st.n_colours
    rpr = train["reward_per_round"]
    out = {"settings": asdict(st), "components": stack.real_component_report(),
           "train": {"n_updates": train["n_updates"], "compute": train["compute"],
                     "return_first10": mean(rpr[:10]), "return_last10": mean(rpr[-10:])},
           "untrained": {k: v for k, v in before.items() if k != "tools"},
           "trained": {k: v for k, v in after.items() if k != "tools"},
           "forced_iters": {str(j): {k: v for k, v in r.items() if k != "tools"} for j, r in forced.items()},
           "chance": chance, "wall_s": time.time() - t0}
    return out
