"""A07 / A08 / C05 through the integrated agent: action-conditioned predictions and belief-aware multistep imagination
that changes a real action (spec 6.6(1), 7.1, 7.4).

Protocol (`run_imagination`):
1. Train the full agent on `ProbeEnv` (spec 7.4 diagnostic: a costly test reveals whether a risky launch pays) with
   its own objectives - world model, support/query adaptation, PPO policy/value on verified returns - acting with its
   policy (cfg.use_imagination off while training; the planner's argmax selection would otherwise turn the policy
   objective into advantage-weighted regression). E's archive + replay feed the replay objectives.
2. At checkpoints, on development seeds (offset 100,000+), evaluate the FIRST decision of fresh episodes:
   * `policy`: the decision of the action generator alone (imagination switched off) and its tool probabilities;
   * `planner`: CamUnifiedModel's LatentImaginationPlanner (cfg.use_imagination on): CAM-generated candidate first
     actions, each rolled forward `imag_horizon` decisions over stochastic futures with imagined outcomes, an
     independent FastState clone updated with them (particles + Kalman), the reasoner re-run and the next action
     sampled from the policy on the revised belief, bootstrapped with V;
   * ablations: `planner_frozen_beliefs` (identical planner, imagined belief revision disabled: the clone keeps the
     belief of the real state) and `planner_h1` (one imagined decision + V).
   Reported: the chosen-tool distribution, the mean imagined score per candidate tool, the fraction of rows whose
   candidates contained the test, the change of the real action against the policy-only decision, and the return
   of the whole episode when the first action is the planner's (later decisions by the policy).
3. A07 (spec 6.6(1)): from the same real state, the world model's predictions of the reward and of the test signal
   for each contemplated tool (prior transition with that action's tokens) - different actions, different predictions
   - and after a real test, the prediction of the launch reward conditioned on the observed signal.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from ..interfaces import Action
from ..outcome_spec import CORE_INDEX
from ..runtime import SEED_STRIDE
from .envs import ProbeEnv, ScoreEvaluator
from .loop import Episode, mean, train_agent
from .stack import build_stack, capability_config

R_IDX = CORE_INDEX["reward"]
TOOLS = ("run_test", "launch", "play_safe")


@dataclass
class ImaginationSettings:
    seed: int = 0
    B: int = 32
    env_horizon: int = 3
    train_rounds: int = 240
    checkpoints: tuple = (60, 120, 240)
    update_every: int = 1
    lr: float = 1e-3
    eval_rows: int = 64
    eval_seed0: int = 100_000
    imag_candidates: int = 12
    imag_samples: int = 4
    imag_horizon: int = 2
    probe_cost: float = 1.0
    win: float = 10.0
    loss: float = 10.0
    safe: float = 1.0
    cfg_overrides: dict = field(default_factory=dict)


@contextlib.contextmanager
def config_override(model, **kw):
    """Temporarily replace fields of the model's (frozen) config, e.g. the use_imagination switch."""
    old = model.cfg
    model.cfg = dataclasses.replace(old, **kw)
    try:
        yield model
    finally:
        model.cfg = old


@contextlib.contextmanager
def frozen_imagined_beliefs(model):
    """Ablation: inside the block the belief module's and the Kalman layer's updates return their input state, so an
    imagined branch keeps the belief of the real state (no imagined belief revision). Used only around planning."""
    model.beliefs.update = lambda fast, *a, **k: (fast, {})
    model.kalman_update = lambda fast, *a, **k: fast
    try:
        yield model
    finally:
        del model.beliefs.update
        del model.kalman_update


@contextlib.contextmanager
def imagination_settings(model, horizon: int, candidates: int, samples: int):
    old = (model.imag_horizon, model.imag_candidates, model.imag_samples)
    model.imag_horizon, model.imag_candidates, model.imag_samples = horizon, candidates, samples
    try:
        yield model
    finally:
        model.imag_horizon, model.imag_candidates, model.imag_samples = old


@torch.no_grad()
def policy_tool_probs(model, dec) -> torch.Tensor:
    """[B, len(TOOLS)] probabilities pi(tool | R, G, B) of the action generator for the decision's inputs."""
    cols = []
    for t in TOOLS:
        acts = [Action(tool=t) for _ in range(dec.inputs.B)]
        out = model.policy_evaluate(dec.inputs, acts)
        cols.append(out["logp"].float().exp().cpu())
    return torch.stack(cols, 1)


@torch.no_grad()
def action_conditioned_predictions(model, dec, state, tools) -> dict:
    """A07: from the same Z_t and FastState, the predicted reward and test-signal change for each contemplated tool."""
    specs, prosp = model.prospective_batch(state)
    keys = prosp.keys[0] if prosp.keys else []
    sig = [q for q, k in enumerate(keys) if k is not None and "test signal" in k]
    out = {}
    B = dec.Z.shape[0]
    for t in TOOLS:
        acts = [Action(tool=t) for _ in range(B)]
        with model._autocast():
            A = model.tokenizer.action(acts, tools)
            dt = model.expected_dt(dec.Z, A, acts, tools)
            belief = model.beliefs.tokens(state.fast)
            Zh, _ = model.world_model.prior(dec.Z, A, belief, dt, sample=False)
            m, v, _ = model.predict_outcomes(Zh, state.fast, prosp)
        out[t] = {"reward_mean": mean(m[:, R_IDX].tolist()), "reward_var": mean(v[:, R_IDX].tolist()),
                  "signal_mean": mean(m[:, sig[0]].tolist()) if sig else None}
    rew = [out[t]["reward_mean"] for t in TOOLS]
    out["reward_spread"] = max(rew) - min(rew)
    return out


def first_decision(model, st: ImaginationSettings, mode: str, seed0: int) -> dict:
    """Evaluate the first decision of `eval_rows` fresh episodes in `mode`, then finish the episodes with the policy."""
    env = ProbeEnv(st.eval_rows, horizon=st.env_horizon, probe_cost=st.probe_cost, win=st.win, loss=st.loss,
                   safe=st.safe)
    ev = ScoreEvaluator(env)
    gen = torch.Generator().manual_seed(seed0)
    ep = Episode(model, env, ev, [seed0 + b for b in range(st.eval_rows)], greedy=True, generator=gen)
    plan_cfg = {"use_imagination": mode != "policy"}
    h = 1 if mode == "planner_h1" else st.imag_horizon
    ctx = frozen_imagined_beliefs(model) if mode == "planner_frozen_beliefs" else contextlib.nullcontext()
    with config_override(model, **plan_cfg), imagination_settings(model, h, st.imag_candidates, st.imag_samples), ctx:
        dec, tools = ep.decide()
    probs = policy_tool_probs(model, dec)
    chosen = [a.tool for a in dec.actions]
    res = {"mode": mode, "frac": {t: mean(float(c == t) for c in chosen) for t in TOOLS},
           "policy_probs": {t: mean(probs[:, i].tolist()) for i, t in enumerate(TOOLS)}}
    if dec.imagined and dec.imagined[0]:
        per_tool = {t: [] for t in TOOLS}
        cover = []
        for b in range(len(chosen)):
            cands = [c["action"]["tool"] for c in dec.candidates[b]]
            cover.append(float("run_test" in cands))
            for item, c in zip(dec.imagined[b], cands):
                per_tool[c].append(item["value"])
        res["imagined_value"] = {t: mean(v) for t, v in per_tool.items()}
        res["candidates_include_test"] = mean(cover)
        res["imagined_next_tools_after_test"] = _next_tools(dec, "run_test")
    if mode == "policy":
        res["a07"] = action_conditioned_predictions(model, dec, ep.state, tools)
    ep.execute(dec, tools)
    with config_override(model, use_imagination=False):
        while not all(ep.done):
            ep.step()
    res["return"] = mean(ep.scores)
    res["chosen"] = chosen
    return res


def _next_tools(dec, first: str) -> dict:
    """Distribution of the imagined second actions sampled by the policy after an imagined `first` action."""
    cnt: dict = {}
    for b in range(len(dec.candidates)):
        for item, c in zip(dec.imagined[b], dec.candidates[b]):
            if c["action"]["tool"] == first:
                for t in item.get("next_tools", []):
                    cnt[t] = cnt.get(t, 0) + 1
    n = sum(cnt.values())
    return {t: v / n for t, v in cnt.items()} if n else {}


@torch.no_grad()
def after_forced_test(model, st: ImaginationSettings, seed0: int) -> dict:
    """The first action is forced to the test (only `run_test` offered); at the second decision, from the real
    state after the observed signal: the predicted reward of launching and of playing safe, and the policy's
    probability of launching, split by the observed signal (new observations change the predictive distribution
    and the decision)."""
    env = ProbeEnv(st.eval_rows, horizon=st.env_horizon, probe_cost=st.probe_cost, win=st.win, loss=st.loss,
                   safe=st.safe)
    ev = ScoreEvaluator(env)
    ep = Episode(model, env, ev, [seed0 + b for b in range(st.eval_rows)], greedy=True)
    full = env.available_tools()
    only_test = [[t for t in lst if t.name == "run_test"] for lst in full]
    with config_override(model, use_imagination=False):
        dec0, _ = ep.decide(tools=only_test)
        ep.execute(dec0, only_test)
        dec, tools = ep.decide()
        pred = action_conditioned_predictions_rows(model, dec, ep.state, tools)
        probs = policy_tool_probs(model, dec)
    out = {}
    for name, val in (("signal_pos", True), ("signal_neg", False)):
        rows = [b for b in range(ep.B) if env.hidden[b]["signal"] == val]
        if rows:
            out[name] = {"n": len(rows), "pred_launch_reward": mean(pred["launch"][b] for b in rows),
                         "pred_safe_reward": mean(pred["play_safe"][b] for b in rows),
                         "p_launch": mean(float(probs[b, 1]) for b in rows),
                         "greedy_launch": mean(float(dec.actions[b].tool == "launch") for b in rows)}
    return out


@torch.no_grad()
def action_conditioned_predictions_rows(model, dec, state, tools) -> dict:
    """Per-row predicted reward [B] for each contemplated tool from the decision's real Z_t and FastState."""
    specs, prosp = model.prospective_batch(state)
    out = {}
    B = dec.Z.shape[0]
    for t in TOOLS:
        acts = [Action(tool=t) for _ in range(B)]
        with model._autocast():
            A = model.tokenizer.action(acts, tools)
            dt = model.expected_dt(dec.Z, A, acts, tools)
            Zh, _ = model.world_model.prior(dec.Z, A, model.beliefs.tokens(state.fast), dt, sample=False)
            m, _, _ = model.predict_outcomes(Zh, state.fast, prosp)
        out[t] = m[:, R_IDX].float().cpu().tolist()
    return out


def evaluate_all(model, st: ImaginationSettings, modes=("policy", "planner", "planner_frozen_beliefs", "planner_h1")
                 ) -> dict:
    out = {}
    for i, mode in enumerate(modes):
        out[mode] = first_decision(model, st, mode, st.eval_seed0)     # same development episodes for every mode
    pol = out["policy"]["chosen"]
    for mode in modes:
        out[mode]["changed_vs_policy"] = mean(float(a != b) for a, b in zip(out[mode]["chosen"], pol))
    for mode in modes:
        out[mode].pop("chosen")
    out["after_forced_test"] = after_forced_test(model, st, st.eval_seed0 + 50_000)
    return out


def run_imagination(settings: Optional[ImaginationSettings] = None, root: Optional[str] = None, device=None,
                    log_every: int = 0) -> dict:
    st = settings or ImaginationSettings()
    t0 = time.time()
    cfg = capability_config(use_imagination=False, use_memory=False, lr=st.lr, imag_branches=st.imag_candidates,
                            imag_samples=st.imag_samples, imag_horizons=(st.imag_horizon,), **st.cfg_overrides)
    tmp = root or tempfile.mkdtemp(prefix="cam_y_imag_")
    stack = build_stack(cfg, device=device, seed=st.seed, memory_root=os.path.join(tmp, f"mem_s{st.seed}"),
                        agent_kwargs=dict(seq_len=st.env_horizon, n_seqs=8, adapt_seq_len=st.env_horizon,
                                          adapt_n_seqs=8))
    env = ProbeEnv(st.B, horizon=st.env_horizon, probe_cost=st.probe_cost, win=st.win, loss=st.loss, safe=st.safe)
    ev = ScoreEvaluator(env)
    base = [st.seed * 10_007 + b for b in range(st.B)]
    evals = {"0": evaluate_all(stack.model, st)}
    done, episodes = 0, 0
    train_log = []
    for cp in st.checkpoints:
        n = cp - done
        seeds = [s + SEED_STRIDE * episodes for s in base]        # continue the run's episode-seed sequence
        tr = train_agent(stack, env, ev, n, update_every=st.update_every, seeds=seeds, log_every=log_every)
        episodes += tr["compute"]["episodes"]
        done = cp
        rpr = tr["reward_per_round"]
        train_log.append({"rounds": cp, "return_per_row_per_round": mean(rpr), "final_losses": tr["final_losses"],
                          "compute": tr["compute"]})
        evals[str(cp)] = evaluate_all(stack.model, st)
    return {"settings": asdict(st), "components": stack.real_component_report(), "train": train_log,
            "evals": evals, "value_bounds": env.value_bounds(), "wall_s": time.time() - t0}
