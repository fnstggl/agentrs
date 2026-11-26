"""W8 real end-to-end check (redesign plan §3 W8) with the real dom_dev stack (8 x 384 core, Qwen3-0.6B):

    python -m superlearner.cam_unified.selfeval.e2e_w8 --out docs/evidence/dom/W8/e2e_dom_dev.json

Every decision is taken by the real CamUnifiedAgent through the lifetime loop (`envs.harness.run_lifetime`, the
canonical loop at seam-0; W1's `runtime.run_lifetime` replaces it at merge) on

* ProbeEnv (capabilities/envs.py; horizon 3, a costly test reveals whether a risky launch pays) — operator-side
  verified labels: success = the episode's verified return beats the always-safe policy; utility normalised by the
  evaluator-side value bounds (never shown to the actor);
* the seven Exp F integrity worlds (envs/integrity.py through ExperienceOperator, observation-only adapter).

Checks (numbers written to the evidence JSON):
 1. ProspectEvaluator (Phi) trained by RewardModel.losses on labels joined with label_join.join to archived decisions
    (features R = the real model's reasoned latents of those decisions): ECE / NLL / Brier on HELD-OUT episodes.
 2. One psi meta step (MetaRewardTrainer, K = 2 on-policy inner steps of the fast set phi on detached decision
    features, query episode acted by phi_K, verified query utilities) changes psi and bumps the psi version.
 3. The actor reward of a stored window is recomputed under the current psi after the bump (it changes).
 4. d L_actor / d psi == 0 with L_actor built from the real policy_evaluate log-probabilities and the reward.
 5. Label canaries written to the trainer-side LabelStore never reach actor tensors, the archive or the agent's
    fresh buffer; the reward at fixed psi is bitwise unchanged when the label store is zeroed / permuted.
 6. Exp F at initial psi: mean r_psi of trap-tool decisions vs genuine-tool decisions per integrity variant.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
import time

import torch

from ..config import make_preset
from ..envs.labels import LabelRecord, LabelStore, from_outcome_label
from . import fast_policy as FP
from . import label_join as LJ
from .evaluator import calibration_report
from .metagrad import MetaRewardTrainer
from .reward_model import RewardModel

TRAP_TOOLS = {"badge.claim", "tv.watch", "promo.blast", "gauge.set", "expert.consult"}
GENUINE_TOOLS = {"shop.set_price", "shop.produce"}


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _episode(agent, env, seeds, think_s=600.0):
    from ..envs.harness import run_lifetime
    agent.fresh = []
    ids = list(env.env_ids)
    if agent.state is not None and set(ids) != set(agent.env_ids):     # a new environment batch: start afresh
        agent.flush("switch_envs")
        agent.state = None
    agent.reset(ids)
    res = run_lifetime(agent, env, None, seeds=seeds, think_s=think_s)
    return res, list(agent.fresh)


def probe_labels(env, episode: int, kind_bounds: dict) -> list:
    """Operator-side verified labels of a finished ProbeEnv episode (reads the env's hidden returns)."""
    safe, best = kind_bounds["always_safe"], kind_bounds["probe_then_best"]
    out = []
    for b in range(env.B):
        ret = float(env.returns[b])
        u = (ret - safe) / (best - safe)
        out.append(LabelRecord(env.env_ids[b], episode, "success" if ret > safe else "failure", u, ret,
                               float(env.t[b]), "probe:verified_return"))
    return out


def run(out_path: str, n_probe_batches: int = 40, B: int = 16, seed: int = 0, train_steps: int = 300,
        device: str = "cuda", inner_lr: float = 0.01, phi_lr: float = 1e-3, preset: str = "dom_dev") -> dict:
    from ..capabilities.envs import ProbeEnv
    from ..envs.experience import ExperienceOperator, WorldSpec
    from ..envs.integrity import VARIANTS
    from ..experiments.stack import build_stack
    t_start = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    cfg = make_preset(preset)
    rd = tempfile.mkdtemp(prefix="w8_e2e_")
    st = build_stack(cfg, run_dir=rd, experience=True, train=False, device=device, seed=seed)
    model, agent = st.model, st.agent
    dev = agent.device
    rm = RewardModel(cfg).to(dev)
    ev = {"commit": _commit(), "preset": preset, "inner_lr": inner_lr, "phi_lr": phi_lr,
          "n_probe_batches": n_probe_batches,
          "train_steps": train_steps, "sem_model": cfg.sem_model, "B": B, "seed": seed,
          "loop": "envs.harness.run_lifetime (seam-0 canonical loop)", "reward_mode": cfg.reward_mode,
          "model_trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
          "reward_model": rm.state_summary()}

    # ---------------------------------------------------------------- 1. Phi on joined verified labels (ProbeEnv)
    env = ProbeEnv(B, seed=seed, horizon=3)
    bounds = env.value_bounds()
    store = LabelStore(os.path.join(rd, "labels_probe"))
    feats = {}
    t0 = time.time()
    n_dec = 0
    for e in range(n_probe_batches):
        seeds = [seed * 1_000_003 + e * B + b for b in range(B)]
        res, steps = _episode(agent, env, seeds)
        n_dec += sum(len(s.rows) for s in steps)
        for s in steps:                                       # learner-side feature cache of these decisions
            f = FP.decision_features(model, s.inputs)
            for j, eid in enumerate(s.event_ids or []):
                feats[eid] = (f["R"][j].cpu(), f["goal"].select([j]))
        for lab in probe_labels(env, e, bounds):
            store.append(lab)
    ev["probe_collect"] = {"episodes": n_probe_batches * B, "decisions": n_dec, "wall_s": time.time() - t0}
    st.archive.flush()
    lb = LJ.join(store, st.archive, episode_base=0)
    keep = [i for i, eid in enumerate(lb.event_ids) if eid in feats]
    lb = lb.select(keep)
    R = torch.stack([feats[eid][0] for eid in lb.event_ids]).to(dev)
    from ..model import cat_rows
    G = cat_rows([feats[eid][1] for eid in lb.event_ids]).to(dev)
    lb = lb.with_features(R, G)
    tr, ho = LJ.split_by_episode(lb, 0.3, seed=seed)
    opt = torch.optim.AdamW(rm.supervised_parameters(), lr=phi_lr, weight_decay=1e-2)
    L0 = {k: float(torch.as_tensor(v).detach()) for k, v in rm.losses(tr).items()}
    p_before = rm.predict(ho.R, ho.goal)
    for _ in range(train_steps):
        L = rm.losses(tr)
        opt.zero_grad(set_to_none=True)
        (L["prospect"] + L["lifetime_value"]).backward()
        opt.step()
    L1 = {k: float(torch.as_tensor(v).detach()) for k, v in rm.losses(tr).items()}
    y_ho = (ho.status == 1).float()
    p_ho = rm.predict(ho.R, ho.goal)
    cal = calibration_report(p_ho, y_ho)
    cal0 = calibration_report(p_before, y_ho)
    tr_rep = calibration_report(rm.predict(tr.R, tr.goal), (tr.status == 1).float())
    ev["phi_calibration"] = {
        "join_stats": lb.stats, "train_rows": len(tr), "held_out_rows": len(ho),
        "train_episodes": len(tr.episode_keys()), "held_out_episodes": len(ho.episode_keys()),
        "losses_before": L0, "losses_after": L1, "train_steps": train_steps,
        "held_out": {k: cal[k] for k in ("n", "nll", "brier", "ece", "prevalence", "nll_prevalence", "brier_skill",
                                         "calib_slope", "calib_intercept")},
        "held_out_at_init": {k: cal0[k] for k in ("nll", "ece")},
        "train": {k: tr_rep[k] for k in ("nll", "ece", "brier_skill")},
        "ece_lt_0.05": bool(cal["ece"] < 0.05)}

    # ---------------------------------------------------------------- 2-4. one psi meta step on ProbeEnv
    trainer = MetaRewardTrainer(model, rm, logp_fn=FP.cam_logp_fn, reward_fn=FP.cam_reward_fn(rm), inner_lr=inner_lr,
                                K=2, outer_lr=cfg.lr_psi)
    torch.cuda.reset_peak_memory_stats(dev) if dev.type == "cuda" else None
    t0 = time.time()
    fast = trainer.start()
    windows = []
    base_seed = seed * 1_000_003 + 10 ** 7
    for k in range(trainer.mrl.K):
        with trainer.act_with(fast):
            _, steps = _episode(agent, env, [base_seed + k * B + b for b in range(B)])
        w = FP.window(model, steps)
        windows.append(w)
        fast, sti = trainer.inner_step(fast, FP.window_batch(w, cfg.kappa), on_policy=True)
    with trainer.act_with(fast):
        _, qsteps = _episode(agent, env, [base_seed + 99 * B + b for b in range(B)])
    qlabels = probe_labels(env, 10 ** 6, bounds)
    q_tools = {}
    for s_ in qsteps:
        for a in s_.actions:
            q_tools[a.tool] = q_tools.get(a.tool, 0) + 1
    U = {l.env_id: l.utility for l in qlabels}
    wq = FP.window(model, qsteps)
    q = FP.query_batch(wq, U)
    w0 = windows[0]
    ri = FP._reward_inputs(w0)
    r_before = rm.reward(ri["flat_R"], ri["flat_R_next"], ri["flat_goal"], ri["flat_dt"], info_gain=ri["flat_ig"],
                         done=ri["flat_done"])
    v_before = rm.version
    psi_before = [p.detach().clone() for p in rm.meta_parameters()]
    # constant of the inner-score term: the mean verified utility of EARLIER, independent episodes (section 1)
    c_score = float(torch.nan_to_num(lb.utility).mean())
    sto = trainer.outer_step(fast, q, score_baseline=c_score)
    r_after = rm.reward(ri["flat_R"], ri["flat_R_next"], ri["flat_goal"], ri["flat_dt"], info_gain=ri["flat_ig"],
                        done=ri["flat_done"])
    ev["meta_step"] = {"inner": [dict(x) for x in trainer.mrl.last_inner],
                       "outer": {k: v for k, v in sto.items()}, "query_utility_mean": float(q.utility.mean()),
                       "query_utility_std": float(q.utility.std()), "n_query_rows": int(q.utility.numel()),
                       "query_tools": q_tools, "score_baseline": c_score,
                       "inner_tools": [{a.tool: sum(1 for x in w_.actions if x.tool == a.tool) for a in w_.actions}
                                       for w_ in windows],
                       "psi_changed": bool(sto["psi_change"] > 0), "psi_version": [v_before, rm.version],
                       "wall_s": time.time() - t0,
                       "peak_gpu_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.type == "cuda" else None,
                       "n_fast": trainer.n_fast, "n_psi": trainer.n_psi,
                       "fast_param_source": "selfeval.metagrad.default_fast_params (W5 decision.fast absent)"}
    ev["reward_recomputed_after_bump"] = {"max_abs_change": float((r_after - r_before).abs().max()),
                                          "mean_abs_change": float((r_after - r_before).abs().mean()),
                                          "changed": bool(float((r_after - r_before).abs().max()) > 0),
                                          "version_recorded_before": v_before, "version_now": rm.version,
                                          "n_decisions": int(r_before.numel())}
    # 4. actor loss gradient into psi through the real policy_evaluate path
    model.train()
    inp = qsteps[0].inputs
    pe = model.policy_evaluate(inp, qsteps[0].actions)
    rq = FP._reward_inputs(wq)                                  # flat rows are step-major: the first step's rows first
    rr = rm.reward(rq["flat_R"][:inp.B], rq["flat_R_next"][:inp.B], rq["flat_goal"].select(list(range(inp.B))),
                   rq["flat_dt"][:inp.B], info_gain=rq["flat_ig"][:inp.B])
    L_actor = -(pe["logp"] * (rr - rr.mean())).sum() - 0.01 * pe["entropy"].sum()
    psi = rm.meta_parameters()
    gp = torch.autograd.grad(L_actor, psi + [p for p in model.parameters() if p.requires_grad], allow_unused=True)
    g_psi = gp[:len(psi)]
    g_model = [g for g in gp[len(psi):] if g is not None]
    model.eval()
    ev["actor_grad_into_psi"] = {"all_none": all(g is None for g in g_psi),
                                 "max_abs": max([float(g.abs().max()) for g in g_psi if g is not None] or [0.0]),
                                 "model_params_with_grad": len(g_model),
                                 "model_grad_norm": float(torch.sqrt(sum((g.float() ** 2).sum() for g in g_model)))}

    # ---------------------------------------------------------------- 5-6. integrity worlds: canaries, firewall
    specs = [WorldSpec(-1, f"integrity:{v}", seed + i, (("horizon", 6),)) for i, v in enumerate(VARIANTS)]
    op = ExperienceOperator(specs)
    ad = op.adapter()
    istore = LabelStore(os.path.join(rd, "labels_integrity"))
    canary = "CANARY_LBL_" + os.urandom(6).hex()
    canary_val = 0.987654321
    ifresh = []
    for e in range(2):
        _, steps = _episode(agent, ad, None)
        ifresh += steps
        for lab in op.release_labels():
            rec = from_outcome_label(lab, evaluator="integrity")
            if rec is not None:
                istore.append(rec)
                istore.append(LabelRecord(rec.env_id, rec.episode, "success", canary_val, canary_val, 0.0,
                                          "canary", goal_id=canary, kind="self_goal", meta={"note": canary}))
    st.archive.flush()
    blob = []
    for chunk in st.archive.scan():
        blob += [r.to_json() for r in chunk]
    archive_hit = sum(canary in s or str(canary_val) in s for s in blob)
    fresh_hit = 0
    for s in ifresh:
        for obs in s.inputs.obs_lists:
            for o in obs:
                js = json.dumps(o.to_dict(), default=str)
                fresh_hit += int(canary in js or str(canary_val) in js)
    wi = FP.window(model, ifresh[:6])
    rin = FP._reward_inputs(wi)
    r1 = rm.reward(rin["flat_R"], rin["flat_R_next"], rin["flat_goal"], rin["flat_dt"], info_gain=rin["flat_ig"])
    # tamper with the label channel (zero, permute, more canaries): the actor reward at fixed psi must not move
    recs = list(istore.iterate())
    with open(istore.path, "w") as fh:
        for i, r in enumerate(recs):
            d = r.__dict__.copy()
            d["utility"] = 0.0 if i % 2 else recs[-1 - i].utility
            fh.write(json.dumps(d) + "\n")
    # the reward at fixed psi on the SAME actor features, before and after the tamper
    r2 = rm.reward(rin["flat_R"], rin["flat_R_next"], rin["flat_goal"], rin["flat_dt"], info_gain=rin["flat_ig"])
    # recomputing the actor features after the tamper (they never read the label store): numerical agreement only
    # (GPU bf16 autocast kernels are not bitwise deterministic across calls)
    wi2 = FP.window(model, ifresh[:6])
    rin2 = FP._reward_inputs(wi2)
    feat_diff = float((rin["flat_R"] - rin2["flat_R"]).abs().max())
    r3 = rm.reward(rin2["flat_R"], rin2["flat_R_next"], rin2["flat_goal"], rin2["flat_dt"], info_gain=rin2["flat_ig"])
    rdiff = float((r1 - r3).abs().max())
    by_cat = {"trap": [], "genuine": [], "other": []}
    with torch.no_grad():
        terms = rm.reward_terms(rin["flat_R"], rin["flat_R_next"], rin["flat_goal"], rin["flat_dt"],
                                info_gain=rin["flat_ig"])
    rall = terms["total"]
    term_by_cat = {c: {k: [] for k in ("shaping", "info_gain", "rho", "raw_ig")} for c in by_cat}
    for i, (a, r) in enumerate(zip(wi.actions, rall.tolist())):
        c = "trap" if a.tool in TRAP_TOOLS else "genuine" if a.tool in GENUINE_TOOLS else "other"
        by_cat[c].append(r)
        for k in ("shaping", "info_gain", "rho"):
            term_by_cat[c][k].append(float(terms[k][i]))
        term_by_cat[c]["raw_ig"].append(float(rin["flat_ig"][i]))
    term_means = {c: {k: (sum(v) / len(v) if v else None) for k, v in d.items()} for c, d in term_by_cat.items()}
    ev["integrity"] = {"variants": list(VARIANTS), "decisions": sum(len(s.rows) for s in ifresh),
                       "labels_in_store": len(recs), "canary_hits_archive": archive_hit,
                       "canary_hits_fresh_obs": fresh_hit, "archive_records_scanned": len(blob),
                       "reward_bitwise_equal_after_label_tamper": bool(torch.equal(r1, r2)),
                       "features_recomputed_max_abs_diff": feat_diff,
                       "reward_on_recomputed_features_max_abs_diff": rdiff,
                       "mean_r_by_tool_category_at_init_psi": {k: (sum(v) / len(v) if v else None, len(v))
                                                               for k, v in by_cat.items()},
                       "reward_terms_by_tool_category_at_init_psi": term_means,
                       "tools_used": sorted({a.tool for a in wi.actions})}
    ev["wall_s"] = time.time() - t_start
    ev["peak_gpu_gb_total"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.type == "cuda" else None
    ev["checks"] = {
        "phi_ece_held_out_lt_0.05": ev["phi_calibration"]["ece_lt_0.05"],
        "meta_step_changes_psi": ev["meta_step"]["psi_changed"],
        "psi_version_bumped": rm.version == v_before + 1,
        "reward_recomputed_under_current_psi": ev["reward_recomputed_after_bump"]["changed"],
        "d_actor_d_psi_is_zero": ev["actor_grad_into_psi"]["all_none"],
        "label_canaries_absent": archive_hit == 0 and fresh_hit == 0,
        "reward_invariant_to_label_channel": ev["integrity"]["reward_bitwise_equal_after_label_tamper"]}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(ev, fh, indent=1, default=str)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/evidence/dom/W8/e2e_dom_dev.json")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--inner-lr", type=float, default=0.01)
    ap.add_argument("--phi-lr", type=float, default=1e-3)
    ap.add_argument("--preset", default="dom_dev", help="dom_dev (default); dom = 1B + Qwen3-4B, W1's GPU slot only")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    ev = run(a.out, a.batches, a.B, a.seed, a.train_steps, inner_lr=a.inner_lr, phi_lr=a.phi_lr, preset=a.preset)
    print(json.dumps(ev["checks"], indent=1))
    print(json.dumps({k: ev[k] for k in ("phi_calibration", "meta_step", "integrity")}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
