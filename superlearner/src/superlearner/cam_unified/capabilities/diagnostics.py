"""Integration diagnostics of the assembled stack (workstream Y): the measurements behind the interface change requests.

`run_diagnostics` (all on the real stack, untrained unless stated):

* `state_signal_effect` - counterfactual pairs that differ only in the observed test signal (ProbeEnv, same seeds and
  rows): the size of the signal's effect on Z_t and R_t relative to their norms and to the across-row variation, and
  the change of the action generator's launch probability;
* `policy_gradient_split` - where the policy objective's gradient goes (tool embeddings / pointer vs core / reasoner)
  on ProbeEnv and ChainEnv;
* `latent_scale` - the world model's latent term at initialisation with B's RMS-normalised core, with the
  `latent_floor_rms` floor (world/dynamics.py) and without it (floor 0 = the previous behaviour);
* `memory_serving` - fraction of retrievals that return any memory after consolidation, through E's Consolidator as
  the agent calls it vs with the serving adapter (ensure_current);
* `ema_target` - whether the world model's EMA target backbone changes over agent updates with the plain trainer vs
  the target-sync adapter;
* `replay_sync` - replay draws available to the agent with E's ReplaySampler as given (the agent calls `add`, which
  the sampler does not have) vs with the syncing adapter;
* `prospective_first_decision` - outcome keys queried at an episode's first decision with and without the
  prospective-ingest adapter.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import torch

from .envs import ChainEnv, MechanismEnv, ProbeEnv, ScoreEvaluator
from .imagination import policy_tool_probs
from .loop import Episode, train_agent
from .reasoning import actor_gradient_split
from .stack import build_stack, capability_config


@torch.no_grad()
def state_signal_effect(stack, n: int = 32, seed0: int = 300_000) -> dict:
    model = stack.model
    res = {}
    for flip in (False, True):
        env = ProbeEnv(n, horizon=3)
        ep = Episode(model, env, ScoreEvaluator(env), [seed0 + b for b in range(n)], greedy=True)
        if flip:
            for h in env.hidden:
                h["signal"] = not h["signal"]
        only = [[t for t in lst if t.name == "run_test"] for lst in env.available_tools()]
        d0, _ = ep.decide(tools=only)
        ep.execute(d0, only)
        d1, _ = ep.decide()
        res[flip] = (d1.Z.float().cpu(), d1.R.float().cpu(), policy_tool_probs(model, d1))
    (Z0, R0, P0), (Z1, R1, P1) = res[False], res[True]
    fl = lambda x: x.reshape(n, -1)
    return {"Z_norm": float(fl(Z0).norm(dim=1).mean()), "Z_signal_effect_norm": float(fl(Z1 - Z0).norm(dim=1).mean()),
            "Z_across_row_deviation_norm": float(fl(Z0 - Z0.mean(0)).norm(dim=1).mean()),
            "R_norm": float(fl(R0).norm(dim=1).mean()), "R_signal_effect_norm": float(fl(R1 - R0).norm(dim=1).mean()),
            "launch_prob": float(P0[:, 1].mean()), "launch_prob_change": float((P1[:, 1] - P0[:, 1]).abs().mean())}


def latent_scale(device=None, seed: int = 0) -> dict:
    """World-model terms at init on MechanismEnv replay with the RMS floor and without it."""
    out = {}
    for floor in (0.1, 0.0):
        cfg = capability_config(use_imagination=False, use_memory=False)
        tmp = tempfile.mkdtemp(prefix="cam_y_diag_")
        st = build_stack(cfg, device=device, seed=seed, memory_root=tmp,
                         agent_kwargs=dict(seq_len=8, n_seqs=4, adapt_seq_len=8, adapt_n_seqs=4))
        st.model.world_model.latent_floor_rms = floor
        env = MechanismEnv(16, horizon=8)
        train_agent(st, env, ScoreEvaluator(env), 8, update_every=0, seeds=list(range(16)))
        seqs = st.agent._sample(4, 8)
        seq, aux = st.agent.tokenise_sequences(seqs)
        st.model.train()
        with torch.no_grad():
            filt = st.model.filter_sequence(seq, aux)
            o = st.model.world_loss(seq, filt)
        out[f"floor_{floor}"] = {k: float(o[k]) for k in ("latent", "open_loop", "outcome", "reward", "obs", "world")}
        Z = torch.stack(filt.Zs).float()
        out[f"floor_{floor}"]["Z_across_sample_std_over_rms"] = float(
            Z.reshape(-1, *Z.shape[2:]).std(0).mean() / Z.pow(2).mean().sqrt())
    return out


def memory_serving(device=None, seed: int = 0) -> dict:
    from .memory import EpisodeScopedRetrieval, MemorySettings, evaluate_cue
    out = {}
    for serve in (False, True):
        cfg = capability_config(use_imagination=False, use_memory=True, mem_segment=8)
        st = build_stack(cfg, device=device, seed=seed, memory_root=tempfile.mkdtemp(prefix="cam_y_diag_"),
                         agent_kwargs=dict(seq_len=12, n_seqs=2, adapt_seq_len=12, adapt_n_seqs=2))
        if not serve:
            st.consolidator = st.consolidator.consolidator        # E's Consolidator exactly as the agent calls it
            st.agent.consolidator = st.consolidator
        scope = EpisodeScopedRetrieval(st.agent)
        r = evaluate_cue(st, MemorySettings(eval_rows=4, eval_episodes=1, delay=10))
        scope.remove()
        out["with_serving_adapter" if serve else "consolidator_as_called_by_agent"] = {
            k: r[k] for k in ("retrieved_any_rate", "retrieval_hit_rate", "cue_memory_in_context_rate",
                              "index_segments")}
        out[("with" if serve else "without") + "_current_version"] = st.index.current_version
    return out


def ema_target(device=None, seed: int = 0, rounds: int = 8) -> dict:
    out = {}
    for sync in (False, True):
        cfg = capability_config(use_imagination=False, use_memory=False)
        st = build_stack(cfg, device=device, seed=seed, memory_root=tempfile.mkdtemp(prefix="cam_y_diag_"),
                         agent_kwargs=dict(seq_len=8, n_seqs=4, adapt_seq_len=8, adapt_n_seqs=4))
        if not sync:
            st.agent.trainer = st.trainer.trainer                 # D's CombinedTrainer exactly as given
        wm = st.model.world_model
        t0 = [p.detach().clone() for p in wm.target_core.parameters()]
        env = MechanismEnv(16, horizon=8)
        train_agent(st, env, ScoreEvaluator(env), rounds, update_every=1, seeds=list(range(16)))
        moved = max(float((p.detach() - q).abs().max()) for p, q in zip(wm.target_core.parameters(), t0))
        out["with_target_sync_adapter" if sync else "agent_with_plain_trainer"] = {"target_max_abs_change": moved}
    return out


def replay_sync(device=None, seed: int = 0) -> dict:
    out = {}
    for adapter in (False, True):
        cfg = capability_config(use_imagination=False, use_memory=False)
        st = build_stack(cfg, device=device, seed=seed, memory_root=tempfile.mkdtemp(prefix="cam_y_diag_"),
                         agent_kwargs=dict(seq_len=8, n_seqs=4, adapt_seq_len=8, adapt_n_seqs=4))
        if not adapter:
            st.agent.replay = st.replay.sampler                   # E's ReplaySampler exactly as given
        env = MechanismEnv(8, horizon=8)
        train_agent(st, env, ScoreEvaluator(env), 8, update_every=0, seeds=list(range(8)))
        out["with_sync_adapter" if adapter else "sampler_as_given"] = {
            "archived": st.archive.count(), "replay_draws": len(st.agent._sample(4, 8))}
    return out


@torch.no_grad()
def prospective_first_decision(device=None, seed: int = 0) -> dict:
    cfg = capability_config(use_imagination=False, use_memory=False)
    out = {}
    for ingest in (False, True):
        st = build_stack(cfg, device=device, seed=seed, prospective_ingest=ingest)
        env = ProbeEnv(2, horizon=3)
        ep = Episode(st.model, env, ScoreEvaluator(env), [1, 2], greedy=True)
        dec, _ = ep.decide()
        keys = [s.key for s in dec.prospective_specs[0]]
        out["with_ingest_adapter" if ingest else "model_as_given"] = {
            "queried_keys": keys, "test_signal_queried": any("test signal" in k for k in keys)}
    return out


def run_diagnostics(settings=None, root: Optional[str] = None, device=None, log_every: int = 0) -> dict:
    t0 = time.time()
    seed = getattr(settings, "seed", 0)
    cfg = capability_config(use_imagination=False, use_memory=False)
    st = build_stack(cfg, device=device, seed=seed)
    out = {"state_signal_effect_untrained": state_signal_effect(st)}
    probe = ProbeEnv(32, horizon=3)
    out["policy_gradient_split_probe_untrained"] = actor_gradient_split(st, probe, list(range(500_000, 500_032)))
    chain = ChainEnv(64, hops=(1, 2))
    out["policy_gradient_split_chain_untrained"] = actor_gradient_split(st, chain, list(range(510_000, 510_064)))
    out["latent_scale"] = latent_scale(device, seed)
    out["memory_serving"] = memory_serving(device, seed)
    out["ema_target"] = ema_target(device, seed)
    out["replay_sync"] = replay_sync(device, seed)
    out["prospective_first_decision"] = prospective_first_decision(device, seed)
    out["wall_s"] = time.time() - t0
    return out


@dataclass
class DiagnosticsSettings:
    """Settings of capabilities.run diagnostics (only the seed is used)."""
    seed: int = 0
