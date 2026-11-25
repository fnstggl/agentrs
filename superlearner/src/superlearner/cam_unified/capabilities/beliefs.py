"""A05 / A06 through the integrated agent: fast beliefs and competing mechanism hypotheses (spec Part 5).

Protocol (`run_beliefs`):
1. Assemble the real stack with E's archive + replay; train `CamUnifiedAgent` on `MechanismEnv` (two competing payout
   mechanisms, uniform per episode) with its own objectives (world model, support/query adaptation, policy/value)
   through `runtime.run_steps`.
2. Evaluate on fresh development seeds (offset 100,000+) with the trained weights, step by step through
   `CamUnifiedModel.decide` / `observe_outcomes` (the agent's calls), one fixed mechanism per evaluation batch:
   * the predictive reward distribution of the next pull (mean/variance reported in the DecisionTrace) after
     k = 0..H-1 observed pulls;
   * the part of that change caused by the fast state alone: the same action-conditioned latent Z_hat scored with the
     current FastState versus the episode's initial FastState (`belief_effect`), split into the particle mixture and
     the Kalman/NIG last layer;
   * particle weights: ESS, weight entropy and the weight mass on particles whose own conditional prediction of the
     reward is closer to the active mechanism's mean than to any competing mechanism's mean (`mass_on_active`);
     a particle has no name, so "concentration on the active mechanism" is defined by what each particle predicts;
   * misspecification: an unfamiliar mechanism never seen in training; surprise z-scores, CUSUM alarms, uncertainty
     expansions (B's BeliefModule) and the predictive variance compared with familiar mechanisms.
"""
from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from ..outcome_spec import CORE_INDEX
from .envs import MechanismEnv, ScoreEvaluator
from .loop import Episode, mean, train_agent
from .stack import build_stack, capability_config

R_IDX = CORE_INDEX["reward"]


@dataclass
class BeliefSettings:
    seed: int = 0
    B: int = 16
    horizon: int = 8
    train_rounds: int = 160
    update_every: int = 2
    lr: float = 1e-3
    eval_rows: int = 32
    eval_seed0: int = 100_000          # development offsets (experiment register)
    mechanisms: dict = field(default_factory=lambda: {"low": (-1.0, 0.25), "high": (1.0, 0.25)})
    unfamiliar: dict = field(default_factory=lambda: {"extreme": (4.0, 0.25)})
    cfg_overrides: dict = field(default_factory=dict)


def belief_step_stats(model, state, dec, fast0) -> dict:
    """Per-row statistics of the current belief before executing `dec` (state.fast is B_t)."""
    fast = state.fast
    specs, prosp = model.prospective_batch(state)
    with torch.no_grad(), model._autocast():
        feat = model.outcome_head.features(dec.Z_hat.to(model.device), prosp)
        d = model.outcome_head.dist(feat, prosp, cond=fast.particles)
        mk = d.mean().float()[:, :, R_IDX]                                                     # [B,K]
        w = torch.softmax(fast.logw.float(), -1)
        mix_now, var_now, _ = model.base_moments(dec.Z_hat, fast, prosp)
        mix_0, var_0, _ = model.base_moments(dec.Z_hat, fast0, prosp)
        full_now, fvar_now, _ = model.predict_outcomes(dec.Z_hat, fast, prosp)
        full_0, fvar_0, _ = model.predict_outcomes(dec.Z_hat, fast0, prosp)
    ess = 1.0 / (w * w).sum(-1)
    ent = -(w * torch.log(w.clamp_min(1e-30))).sum(-1)
    return {"particle_mean": mk.cpu(), "w": w.cpu(), "ess": ess.cpu(), "entropy": ent.cpu(),
            "mixture_mean": mix_now[:, R_IDX].cpu(), "mixture_mean_initial_belief": mix_0[:, R_IDX].cpu(),
            "full_mean": full_now[:, R_IDX].cpu(), "full_mean_initial_belief": full_0[:, R_IDX].cpu(),
            "full_var": fvar_now[:, R_IDX].cpu(), "full_var_initial_belief": fvar_0[:, R_IDX].cpu(),
            "mixture_var": var_now[:, R_IDX].cpu()}


def evaluate_mechanism(stack, settings: BeliefSettings, mechanism: str, seed0: int) -> dict:
    """One evaluation batch with every row on `mechanism`. Returns per-step (k = observations so far) means."""
    env = MechanismEnv(settings.eval_rows, horizon=settings.horizon, mechanisms=settings.mechanisms,
                       fixed=mechanism, unfamiliar=settings.unfamiliar)
    ev = ScoreEvaluator(env)
    ep = Episode(stack.model, env, ev, [seed0 + b for b in range(settings.eval_rows)], greedy=True)
    fast0 = ep.state.fast.clone()
    mu = env.all_mechanisms[mechanism][0]
    competitors = [v[0] for k, v in settings.mechanisms.items() if k != mechanism]
    if mechanism not in settings.mechanisms:          # unfamiliar: compared with every training mechanism
        competitors = [v[0] for v in settings.mechanisms.values()]
    per_k = []
    for k in range(settings.horizon):
        dec, tools = ep.decide()
        s = belief_step_stats(stack.model, ep.state, dec, fast0)
        pk = s["particle_mean"]
        near_active = torch.ones_like(pk, dtype=torch.bool)
        for c in competitors:
            near_active &= (pk - mu).abs() < (pk - c).abs()
        mass = (s["w"] * near_active.float()).sum(-1)
        rec = ep.execute(dec, tools)
        bi = rec.info.get("belief_info", {})
        per_k.append({
            "k": k,
            "pred_mean": mean(d["reward"] for d in dec.predicted),
            "pred_var": mean(u["reward"] for u in dec.uncertainty),
            "belief_effect_full": mean((s["full_mean"] - s["full_mean_initial_belief"]).tolist()),
            "belief_effect_mixture": mean((s["mixture_mean"] - s["mixture_mean_initial_belief"]).tolist()),
            "full_var_initial_belief": mean(s["full_var_initial_belief"].tolist()),
            "mixture_var": mean(s["mixture_var"].tolist()),
            "ess": mean(s["ess"].tolist()), "weight_entropy": mean(s["entropy"].tolist()),
            "mass_on_active": mean(mass.tolist()),
            "frac_particles_near_active": mean(near_active.float().mean(-1).tolist()),
            "particle_pred_spread": mean(pk.std(-1).tolist()),
            "abs_err": mean((torch.tensor([d["reward"] for d in dec.predicted]) - mu).abs().tolist()),
            "realised_mean": mean(rec.rewards),
            "surprise_z": mean(bi["surprise_z"].float().cpu().tolist()) if "surprise_z" in bi else None,
            "alarm_rate": mean(bi["alarm"].float().cpu().tolist()) if "alarm" in bi else None,
            "expanded_rate": mean(bi["expanded"].float().cpu().tolist()) if "expanded" in bi else None,
            "resampled_rate": mean(bi["resampled"].float().cpu().tolist()) if "resampled" in bi else None,
        })
    return {"mechanism": mechanism, "mu": mu, "steps": per_k,
            "expansions_total": int(sum(r["expanded_rate"] or 0 for r in per_k) * settings.eval_rows)}


def run_beliefs(settings: Optional[BeliefSettings] = None, root: Optional[str] = None, device=None,
                log_every: int = 0) -> dict:
    """Train + evaluate (see the module docstring). Returns a JSON-serialisable result dict."""
    st = settings or BeliefSettings()
    t0 = time.time()
    cfg = capability_config(use_imagination=False, use_memory=False, lr=st.lr, **st.cfg_overrides)
    own_tmp = root is None
    tmp = tempfile.mkdtemp(prefix="cam_y_beliefs_") if own_tmp else root
    stack = build_stack(cfg, device=device, seed=st.seed, memory_root=os.path.join(tmp, f"mem_s{st.seed}"),
                        agent_kwargs=dict(seq_len=st.horizon, n_seqs=4, adapt_seq_len=st.horizon, adapt_n_seqs=4))
    env = MechanismEnv(st.B, horizon=st.horizon, mechanisms=st.mechanisms)
    ev = ScoreEvaluator(env)
    seeds = [st.seed * 10_007 + b for b in range(st.B)]                     # training seeds (not 100k+)

    def evaluate(tag):
        return {m: evaluate_mechanism(stack, st, m, st.eval_seed0 + 1000 * i)
                for i, m in enumerate(list(st.mechanisms) + list(st.unfamiliar))}

    before = evaluate("untrained")
    train = train_agent(stack, env, ev, st.train_rounds, update_every=st.update_every, seeds=seeds,
                        log_every=log_every)
    after = evaluate("trained")
    out = {"settings": asdict(st), "cfg": {k: v for k, v in cfg.to_dict().items() if not isinstance(v, dict)},
           "components": stack.real_component_report(), "train": {k: v for k, v in train.items()
                                                                  if k != "reward_per_round"},
           "untrained": before, "trained": after, "wall_s": time.time() - t0}
    out["summary"] = summarise(out)
    return out


def summarise(res: dict) -> dict:
    tr = res["trained"]
    mechs = res["settings"]["mechanisms"]
    unf = res["settings"]["unfamiliar"]
    H = len(next(iter(tr.values()))["steps"])
    s = {}
    for m, r in tr.items():
        st = r["steps"]
        s[m] = {"pred_mean_k0": st[0]["pred_mean"], "pred_mean_last": st[-1]["pred_mean"],
                "abs_err_k0": st[0]["abs_err"], "abs_err_last": st[-1]["abs_err"],
                "mass_on_active_k0": st[0]["mass_on_active"], "mass_on_active_last": st[-1]["mass_on_active"],
                "ess_k0": st[0]["ess"], "ess_last": st[-1]["ess"],
                "belief_effect_full_last": st[-1]["belief_effect_full"],
                "belief_effect_mixture_last": st[-1]["belief_effect_mixture"],
                "pred_var_k0": st[0]["pred_var"], "pred_var_last": st[-1]["pred_var"],
                "max_pred_var": max(x["pred_var"] for x in st),
                "expansions_total": r["expansions_total"],
                "mean_surprise_z": mean(x["surprise_z"] for x in st[1:])}
    fam = [s[m] for m in mechs]
    s["familiar_mean_pred_var_last"] = mean(x["pred_var_last"] for x in fam)
    s["familiar_expansions_total"] = sum(x["expansions_total"] for x in fam)
    for m in unf:
        s[m]["var_ratio_vs_familiar"] = s[m]["max_pred_var"] / max(1e-12, s["familiar_mean_pred_var_last"])
    s["horizon"] = H
    return s
