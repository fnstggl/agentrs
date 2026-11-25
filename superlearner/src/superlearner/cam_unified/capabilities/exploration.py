"""A09 through the integrated agent: learned exploration with learning-progress reward (spec Part 8).

Protocol (`run_exploration`): `ProbeEnv` with `env_horizon` decisions (the test costs now; its signal lets later
decisions launch only when it pays). The full agent acts with its own action generator (sampling, temperature 1) and
learns with its own objectives through `runtime.run_steps`: world model and support/query adaptation from E's
archive + replay, and the policy objective, whose per-decision reward is the verified return plus
`cfg.lp_coef` x the hindsight learning progress of that decision's outcome (CamUnifiedModel.observe_outcomes:
LP_{t-1} = log p(o_t | B_t) - log p(o_t | B_{t-1}), o_t disjoint from the support o_{t-1}; GAE over the semi-Markov
discount). Conditions at the same architecture and experience budget:

* "lp": cfg.lp_coef > 0 (learned exploration with the learning-progress reward);
* "extrinsic": cfg.lp_coef = 0 (verified returns only).

The exploratory decision is an ordinary action of the ActionGenerator (the `run_test` tool), so its probability
is trained by the same policy-gradient path as every other decision - no scalar exploration coefficient and no
knowledge-gradient score is an input. Evaluation on development seeds: the probability and greedy frequency of
testing first, the episode return under greedy decisions, the launch decision after each signal, and the same agent
on an uninformative variant (the signal is a coin flip, so testing is worthless and should not be preferred).
LP statistics by action are read back from the archived records (`extra["lp_prev"]` of the following decision).
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from ..runtime import SEED_STRIDE
from .envs import ProbeEnv, ScoreEvaluator
from .imagination import TOOLS, config_override, policy_tool_probs
from .loop import Episode, mean, train_agent
from .stack import build_stack, capability_config


@dataclass
class ExplorationSettings:
    seed: int = 0
    condition: str = "lp"              # lp | extrinsic
    B: int = 32
    env_horizon: int = 3
    train_rounds: int = 300
    checkpoints: tuple = (100, 200, 300)
    lr: float = 1e-3
    lp_coef: float = 0.1                # cfg.lp_coef default
    eval_rows: int = 64
    eval_seed0: int = 100_000
    probe_cost: float = 1.0
    win: float = 10.0
    loss: float = 10.0
    safe: float = 1.0
    cfg_overrides: dict = field(default_factory=dict)


def evaluate_probe(model, st: ExplorationSettings, informative: bool, seed0: int) -> dict:
    env = ProbeEnv(st.eval_rows, horizon=st.env_horizon, probe_cost=st.probe_cost, win=st.win, loss=st.loss,
                   safe=st.safe, informative=informative)
    ev = ScoreEvaluator(env)
    ep = Episode(model, env, ev, [seed0 + b for b in range(st.eval_rows)], greedy=True)
    with config_override(model, use_imagination=False):
        dec, tools = ep.decide()
        probs = policy_tool_probs(model, dec)
        first = [a.tool for a in dec.actions]
        ep.execute(dec, tools)
        second = None
        while not all(ep.done):
            rec = ep.step()
            if second is None:
                second = [a.tool for a in rec.decision.actions]
    pos = [b for b in range(ep.B) if first[b] == "run_test" and env.hidden[b]["signal"]]
    neg = [b for b in range(ep.B) if first[b] == "run_test" and not env.hidden[b]["signal"]]
    return {"informative": informative, "p_test_first": mean(probs[:, 0].tolist()),
            "p_first": {t: mean(probs[:, i].tolist()) for i, t in enumerate(TOOLS)},
            "greedy_test_first": mean(float(t == "run_test") for t in first),
            "return": mean(ep.scores),
            "launch_after_pos": mean(float(second[b] == "launch") for b in pos) if pos else None,
            "launch_after_neg": mean(float(second[b] == "launch") for b in neg) if neg else None,
            "value_bounds": env.value_bounds()}


def lp_by_action(archive) -> dict:
    """Hindsight learning progress credited to each action (from the archived decision records)."""
    if archive is None:
        return {}
    eps: dict = {}
    for batch in archive.scan():
        for r in batch:
            if r.action is None:
                continue
            eps.setdefault((r.env_id, r.episode_id), []).append(r)
    by: dict = {}
    for recs in eps.values():
        recs.sort(key=lambda r: r.step)
        for prev, nxt in zip(recs[:-1], recs[1:]):
            by.setdefault(prev.action.tool, []).append(float(nxt.extra.get("lp_prev", 0.0)))
    return {t: {"n": len(v), "mean_lp": mean(v)} for t, v in by.items()}


def run_exploration(settings: Optional[ExplorationSettings] = None, root: Optional[str] = None, device=None,
                    log_every: int = 0) -> dict:
    st = settings or ExplorationSettings()
    t0 = time.time()
    if st.condition not in ("lp", "extrinsic"):
        raise ValueError(st.condition)
    lp = st.lp_coef if st.condition == "lp" else 0.0
    cfg = capability_config(use_imagination=False, use_memory=False, lr=st.lr, lp_coef=lp, **st.cfg_overrides)
    tmp = root or tempfile.mkdtemp(prefix="cam_y_explore_")
    stack = build_stack(cfg, device=device, seed=st.seed,
                        memory_root=os.path.join(tmp, f"mem_{st.condition}_s{st.seed}"),
                        agent_kwargs=dict(seq_len=st.env_horizon, n_seqs=8, adapt_seq_len=st.env_horizon,
                                          adapt_n_seqs=8))
    env = ProbeEnv(st.B, horizon=st.env_horizon, probe_cost=st.probe_cost, win=st.win, loss=st.loss, safe=st.safe)
    ev = ScoreEvaluator(env)
    base = [st.seed * 10_007 + b for b in range(st.B)]
    evals = {"0": {"informative": evaluate_probe(stack.model, st, True, st.eval_seed0),
                   "uninformative": evaluate_probe(stack.model, st, False, st.eval_seed0 + 10_000)}}
    done, episodes = 0, 0
    log = []
    for cp in st.checkpoints:
        seeds = [s + SEED_STRIDE * episodes for s in base]
        tr = train_agent(stack, env, ev, cp - done, update_every=1, seeds=seeds, log_every=log_every)
        episodes += tr["compute"]["episodes"]
        done = cp
        log.append({"rounds": cp, "return_per_row_per_round": mean(tr["reward_per_round"]),
                    "final_losses": tr["final_losses"], "compute": tr["compute"]})
        evals[str(cp)] = {"informative": evaluate_probe(stack.model, st, True, st.eval_seed0),
                          "uninformative": evaluate_probe(stack.model, st, False, st.eval_seed0 + 10_000)}
    return {"settings": asdict(st), "lp_coef": lp, "components": stack.real_component_report(), "train": log,
            "evals": evals, "lp_by_action": lp_by_action(stack.archive), "wall_s": time.time() - t0}
