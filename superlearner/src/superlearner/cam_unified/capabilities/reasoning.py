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
from .loop import Episode, mean, rebatch, train_agent
from .readout import discard_readout_buffer
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
    standardized_readout: bool = False   # experimental condition (capabilities/readout.py)
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
    discard_readout_buffer(model)
    hops = [h["k"] for h in env.hidden]
    iters = [r["iters"] for r in rec.decision.reasoning]
    out = {"tools": [a.tool for a in rec.decision.actions], "acc": mean(rec.rewards), "per_hop": {}}
    for k in sorted(set(hops)):
        rows = [b for b, h in enumerate(hops) if h == k]
        out["per_hop"][k] = {"n": len(rows), "acc": mean(rec.rewards[b] for b in rows),
                             "iters": mean(iters[b] for b in rows)}
    return out


@torch.no_grad()
def tool_distribution(model, dec, tools: list) -> torch.Tensor:
    """[B, T] probabilities of every offered tool under the action generator for the decision's inputs (the
    learner-path recomputation `policy_evaluate`, with the model's current reasoning setting)."""
    from ..interfaces import Action
    names = [t.name for t in tools[0]]
    cols = [model.policy_evaluate(dec.inputs, [Action(tool=n) for _ in range(dec.inputs.B)])["logp"].float().exp()
            for n in names]
    return torch.stack(cols, 1).cpu()


@torch.no_grad()
def decision_sensitivity(model, settings: ReasoningSettings, iters=(1, 16), seed0: Optional[int] = None) -> dict:
    """How strongly the reasoning state reaches the decision: KL between the tool distributions computed with J
    forced to iters[0] vs iters[1] (same weights and inputs), the mean absolute change of the probabilities, and the
    KL when the rows' final R are permuted (another episode's reasoning state). Measured on `eval_rows`
    development episodes."""
    env = ChainEnv(settings.eval_rows, n_colours=settings.n_colours, hops=settings.hops)
    seeds = [(settings.eval_seed0 if seed0 is None else seed0) + b for b in range(settings.eval_rows)]
    dists = {}
    old = model.reason_iters
    try:
        for j in iters:
            model.reason_iters = int(j)
            ep = Episode(model, env, ScoreEvaluator(env), seeds, greedy=True)
            dec, tools = ep.decide()
            dists[j] = tool_distribution(model, dec, tools)
        model.reason_iters = old
        ep = Episode(model, env, ScoreEvaluator(env), seeds, greedy=True)
        dec, tools = ep.decide()
        p_own = tool_distribution(model, dec, tools)
        perm = torch.roll(torch.arange(dec.inputs.B), 1)
        with model.reasoner.intervention(lambda j, R: R[perm.to(R.device)] if R.shape[0] == len(perm) else R):
            p_swap = tool_distribution(model, dec, tools)
    finally:
        model.reason_iters = old
        discard_readout_buffer(model)

    def kl(p, q):
        p, q = p.clamp_min(1e-12), q.clamp_min(1e-12)
        return float((p * (p.log() - q.log())).sum(-1).mean())
    a, b = dists[iters[0]], dists[iters[1]]
    return {"kl_J%d_vs_J%d" % tuple(iters): kl(a, b), "mean_abs_prob_change": float((a - b).abs().mean()),
            "kl_R_permuted": kl(p_own, p_swap), "max_prob_mean": float(p_own.max(-1).values.mean()),
            "argmax_changed_J": float((a.argmax(-1) != b.argmax(-1)).float().mean())}


def actor_gradient_split(stack, env, seeds) -> dict:
    """Where the policy objective's gradient goes: one episode of the agent on `env` (fresh on-policy buffer), then
    the agent's own `objective_policy` losses back-propagated separately; gradient L2 norm per component group
    (no optimiser step; the buffer is restored). Shows whether decision learning reaches the latent reasoning state
    (core / reasoner) or acts on the tool embeddings and pointer."""
    from ..agent import UpdateBatch, objective_policy
    from ..runtime import run_steps
    agent, model = stack.agent, stack.model
    saved = list(agent.fresh)
    if agent.state is not None and list(agent.env_ids) != list(env.env_ids):
        rebatch(agent)
        saved = []
    run_steps(agent, env, int(getattr(env, "horizon", 1)), evaluator=ScoreEvaluator(env), update_every=None,
              seeds=seeds)                              # one whole episode per row (every decision gets a target)
    model.train()
    try:
        out = objective_policy(agent, UpdateBatch(None, None, list(agent.fresh), model))
        if not out:
            return {}
        groups = {"action_generator": [p for n, p in model.named_parameters()
                                       if n.startswith("action_generator.") and p.requires_grad],
                  "tokenizer": [p for p in model.tokenizer.parameters() if p.requires_grad],
                  "core": list(model.core.parameters()),
                  "reasoner": list(model.reasoner.parameters()) if model.reasoner is not None else [],
                  "value_head": list(model.value_head.parameters())}
        res = {}
        for name, loss in out["losses"].items():
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            res[name] = {g: float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in ps if p.grad is not None)))
                         if any(p.grad is not None for p in ps) else 0.0 for g, ps in groups.items()}
        model.zero_grad(set_to_none=True)
    finally:
        model.eval()
        agent.fresh = saved
        discard_readout_buffer(model)
    return res


def run_reasoning(settings: Optional[ReasoningSettings] = None, device=None, log_every: int = 0) -> dict:
    st = settings or ReasoningSettings()
    t0 = time.time()
    over, J = _condition(st)
    cfg = capability_config(use_imagination=False, use_memory=False, lr=st.lr, **{**over, **st.cfg_overrides})
    stack = build_stack(cfg, device=device, seed=st.seed, standardized_readout=st.standardized_readout,
                        model_kwargs={"reason_iters": J})
    model = stack.model
    env = ChainEnv(st.B, n_colours=st.n_colours, hops=st.hops)
    ev = ScoreEvaluator(env)
    before = evaluate_chain(model, st)
    sens_before = decision_sensitivity(model, st) if cfg.use_reasoning else None
    grad_env = ChainEnv(st.B, n_colours=st.n_colours, hops=st.hops)
    grads_before = actor_gradient_split(stack, grad_env, [st.eval_seed0 + 90_000 + b for b in range(st.B)])
    train = train_agent(stack, env, ev, st.train_rounds, update_every=1,
                        seeds=[st.seed * 10_007 + b for b in range(st.B)], log_every=log_every)
    after = evaluate_chain(model, st)
    sens_after = decision_sensitivity(model, st) if cfg.use_reasoning else None
    grads_after = actor_gradient_split(stack, grad_env, [st.eval_seed0 + 91_000 + b for b in range(st.B)])
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
           "sensitivity_untrained": sens_before, "sensitivity_trained": sens_after,
           "policy_grad_norms_untrained": grads_before, "policy_grad_norms_trained": grads_after,
           "chance": chance, "wall_s": time.time() - t0}
    return out
