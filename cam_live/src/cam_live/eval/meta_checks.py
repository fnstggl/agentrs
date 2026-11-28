"""cam_live.eval.meta_checks — the six reward meta-learning checks of spec 16.5 on the integrated agent.

Inputs: a run directory written by ``experiments.run_condition(..., save_dir=...)`` (its experience store, the
pre-experience export and the experienced checkpoint) and the run's result JSON (its meta-update records).
The experienced agent is rebuilt from the checkpoint over a *copy* of the run's learner directory; nothing is
written back. Per run:

1. reward parameters affect the inner update: theta_K(psi_post) vs theta_K(psi_pre) from the same theta_0 on the
   same support batch, both at the live bootstrap mix and at mix 0 (the differentiated update); noise floor =
   the same update repeated with the same psi
2. the inner update changes policy behaviour: total variation of pi_theta0 vs pi_thetaK on held-out rows
3. verified outcomes drive reward optimisation: every applied meta-update had >= min_independent independent
   verified outcome groups; the run-level ablation (``E_unverified``: signatures refused) has no applied update
4. nonzero valid meta-gradients reach psi: applied updates with |dJ/dpsi| > 0 and a changed reward version; the
   reward group hash of the experienced export differs from the pre-experience export
5. removing the meta-gradient removes the pathway: ``D`` (meta-updates off) and ``E_unverified`` keep psi
   bit-identical to the pre-experience export while their other groups change
6. changes beyond magnitude: rank correlation of learned rewards (psi_pre vs psi_post) over the run's records,
   and the policy update with every reward it sees multiplied by 3 (a pure magnitude change: must give the same
   update, returns are standardised) against the update under psi_pre (can differ only through the reward's
   *shape*)
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence

import torch

from .stats import spearman


def _tv(p: torch.Tensor, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((p - q).abs() * mask).sum(-1)


def _norm(theta: Dict[str, torch.Tensor], other: Dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(((theta[k].double() - other[k].double()) ** 2).sum()) for k in theta))


def load_experienced(root: str, work: str) -> Any:
    from ..agent import CamLiveAgent
    from ..config import CamLiveConfig
    from ..validation import OffsetClock
    with open(os.path.join(root, "export_post", "config.json")) as f:
        meta = json.load(f)
    cfg = CamLiveConfig.from_dict(meta["config"])
    cfg.device = "cpu" if not torch.cuda.is_available() or cfg.backbone.name == "tiny" else cfg.device
    shutil.copytree(os.path.join(root, "agent"), os.path.join(work, "agent"))
    agent = CamLiveAgent(cfg, os.path.join(work, "agent"), clock=OffsetClock())
    import functools
    from .. import agent as agent_mod
    from ..credit import checkpoint as ck
    orig = agent_mod.load_checkpoint
    # retention removed the optimizer states of evaluation checkpoints (storage rules); the checks never use them
    pruned = _optim_pruned(root)
    src = os.path.join(root, "ckpt_post")
    if pruned:                                   # a copy with an empty optim/ directory (load_checkpoint lists it)
        shutil.copytree(src, os.path.join(work, "ckpt_post"))
        src = os.path.join(work, "ckpt_post")
        os.makedirs(os.path.join(src, "optim"), exist_ok=True)
    agent_mod.load_checkpoint = functools.partial(ck.load_checkpoint, verify=not pruned)
    try:
        agent.load(src)
    finally:
        agent_mod.load_checkpoint = orig
    return agent


def _optim_pruned(root: str) -> bool:
    d = os.path.join(root, "ckpt_post", "optim")
    return not os.path.isdir(d) or not os.listdir(d)


def reward_group(export_dir: str) -> Dict[str, torch.Tensor]:
    from ..credit.checkpoint import load_weights_only
    w, _ = load_weights_only(export_dir)
    return dict(w.get("reward", {}))


def group_hashes(export_dir: str) -> Dict[str, str]:
    with open(os.path.join(export_dir, "config.json")) as f:
        return json.load(f).get("group_hashes", {})


def _psi_from(agent: Any, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """psi (as ``meta.reward.psi()`` names it) with the reward-group tensors ``tensors`` (model parameter names)."""
    params = dict(agent.model.named_parameters())
    saved = {n: params[n].detach().clone() for n in tensors if n in params}
    with torch.no_grad():
        for n, t in tensors.items():
            if n in params:
                params[n].copy_(t.to(params[n].device, params[n].dtype))
        psi = {k: v.detach().clone() for k, v in agent.meta.reward.psi().items()}
        for n, t in saved.items():
            params[n].copy_(t)
    return psi


def run_checks(root: str, result: Dict[str, Any], *, max_support: int = 16) -> Dict[str, Any]:
    work = tempfile.mkdtemp(prefix="camlive_meta_checks_")
    try:
        agent = load_experienced(root, work)
        meta = agent.meta
        recs = [r for r in sorted(agent.store.all(), key=lambda r: r.seq) if meta.inner.eligible(r)]
        out: Dict[str, Any] = {"condition": result["condition"], "seed": result["seed"], "n_eligible": len(recs),
                               "live_mix": float(meta.mixer.mix)}
        if len(recs) < 6:
            out["skipped"] = "fewer than 6 eligible records"
            return out
        half = max(3, min(max_support, len(recs) // 2))
        sup, held = recs[-2 * half:-half] if len(recs) >= 2 * half else recs[:half], recs[-half:]
        ib = meta.inner.build_batch(agent.store, sup)
        ibh = meta.inner.build_batch(agent.store, held)
        theta0 = meta.subset.snapshot()
        psi_post = {k: v.detach().clone() for k, v in meta.reward.psi().items()}
        psi_pre = _psi_from(agent, reward_group(os.path.join(root, "export_pre")))
        lr, K = float(meta.mc.inner_lr), int(meta.mc.inner_steps)

        def thetaK(psi: Dict[str, torch.Tensor], mix: float) -> Dict[str, torch.Tensor]:
            tr = meta.engine.inner_update(theta0, psi, [ib.with_mix(mix)], lr, K, create_graph=False)
            return {k: v.detach() for k, v in tr.detached().items()}

        res1: Dict[str, Any] = {}
        for label, mix in (("live_mix", float(meta.mixer.mix)), ("mix0_differentiated", 0.0)):
            a = thetaK(psi_post, mix)
            a2 = thetaK(psi_post, mix)
            b = thetaK(psi_pre, mix)
            res1[label] = {"mix": mix, "delta_theta_psi_post_vs_psi_pre": _norm(a, b),
                           "noise_floor_same_psi": _norm(a, a2), "update_norm": _norm(a, theta0)}
        out["1_reward_params_affect_inner_update"] = res1
        # 2: behaviour change of the inner update on held-out rows
        a = thetaK(psi_post, float(meta.mixer.mix))
        with torch.no_grad():
            p0 = meta.subset.forward(theta0, ibh).probs.double()
            p1 = meta.subset.forward(a, ibh).probs.double()
            m = ibh.a_mask.double()
            tv = _tv(p0, p1, m)
        out["2_inner_update_changes_policy"] = {"heldout_rows": int(tv.numel()), "tv_mean": float(tv.mean()),
                                                "tv_max": float(tv.max())}
        # 6: beyond magnitude
        with torch.no_grad():
            _, aux_post = meta.inner.functional(theta0, psi_post, ibh.with_mix(0.0), detach_reward=True,
                                                return_aux=True)
            _, aux_pre = meta.inner.functional(theta0, psi_pre, ibh.with_mix(0.0), detach_reward=True,
                                               return_aux=True)
        lp, lq = [float(x) for x in aux_pre["learned"].flatten()], [float(x) for x in aux_post["learned"].flatten()]
        a0 = thetaK(psi_post, 0.0)
        orig = meta.inner.intrinsic                      # pure magnitude: every reward the update sees x 3

        def tripled(psi: Any, ib_: Any, detach_reward: bool = False) -> Any:
            r, rl = orig(psi, ib_, detach_reward=detach_reward)
            return 3.0 * r, 3.0 * rl
        meta.inner.intrinsic = tripled
        try:
            a_scaled = thetaK(psi_post, 0.0)
        finally:
            del meta.inner.intrinsic
        a_pre = thetaK(psi_pre, 0.0)
        out["6_beyond_magnitude"] = {
            "spearman_learned_reward_pre_vs_post": spearman(lp, lq), "n_rows": len(lp),
            "mean_abs_learned_pre": sum(abs(x) for x in lp) / len(lp),
            "mean_abs_learned_post": sum(abs(x) for x in lq) / len(lq),
            "delta_theta_output_scaled_x3_vs_post": _norm(a_scaled, a0),
            "delta_theta_psi_pre_vs_post": _norm(a_pre, a0)}
        # 3 / 4 from the run's meta-update records
        metas = result.get("extract", {}).get("meta_updates", [])
        applied = [x for x in metas if x.get("applied")]
        valid = [x for x in applied if (x.get("meta_grad_norm") or 0.0) > 0 and
                 x.get("reward_version_before") != x.get("reward_version_after")]
        out["3_4_meta_updates"] = {
            "n_records": len(metas), "applied": len(applied), "applied_nonzero_changed_version": len(valid),
            "min_independent_required": int(meta.mc.min_independent),
            "applied_with_fewer_independent_outcomes": sum(1 for x in applied if (x.get("n_independent_outcomes") or 0)
                                                           < meta.mc.min_independent),
            "meta_grad_norms": [x.get("meta_grad_norm") for x in valid][:20],
            "skip_reasons": _count(str(x.get("skipped_reason", "")).split(":")[0] for x in metas
                                   if not x.get("applied"))}
        hp, hq = group_hashes(os.path.join(root, "export_pre")), group_hashes(os.path.join(root, "export_post"))
        out["group_changed_pre_to_post"] = {g: hp.get(g) != hq.get(g) for g in sorted(set(hp) | set(hq))}
        try:
            out["shadow_checkpoints"] = shadow_checkpoints(agent, root, result)
        except Exception as e:  # noqa: BLE001
            out["shadow_checkpoints"] = {"error": f"{type(e).__name__}: {e}"}
        return out
    finally:
        try:
            agent.close()  # type: ignore[possibly-undefined]
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(work, ignore_errors=True)


def shadow_checkpoints(agent: Any, root: str, result: Dict[str, Any], *, eps: float = 1e-12) -> Dict[str, Any]:
    """Spec 11.2 for checkpoints: the trainable policy of the pre-experience and of the experienced checkpoint
    evaluated on the *recorded* decisions (their stored contexts and candidate sets) without acting. Columns kept
    apart: importance-weighted estimates from the observed consequences (IPS, SNIPS, ESS), the model-based
    prediction (sum pi * predicted goal mean) and, diagnostic only, the ground truth on the same contexts."""
    from ..contracts import ItemKind
    from ..credit.checkpoint import load_weights, load_weights_only
    truth = {d["decision_id"]: d for rows in result.get("decision_rows", []) for d in rows}
    recs = [r for r in sorted(agent.store.all(), key=lambda r: r.seq)
            if r.decision is not None and r.decision.decision_id in truth and r.decision.probabilities]
    if not recs:
        return {"n": 0}
    rows = []
    for r in recs:
        refs = list(r.decision.diagnostics.get("context_item_refs") or [])
        items = agent.store.items(refs)
        goal = next((it.text for it in items if it.kind == ItemKind.GOAL), None)
        cands = [c.action for c in r.decision.candidates]
        if not items or len(cands) != len(truth[r.decision.decision_id]["candidates"]):
            continue
        ch = r.decision.chosen
        obs = [o.value for o in r.outcomes if o.value is not None]
        rew = float(obs[0]) if (ch.operation == "lab.act" and obs) else 0.0
        rows.append((items, cands, goal, list(r.decision.probabilities), r.decision.chosen_index, rew,
                     truth[r.decision.decision_id]["candidates"]))
    post = [agent.policy_distribution(it, c, goal_text=g) for it, c, g, *_ in rows]
    saved = {n: p.detach().clone() for n, p in agent.model.named_parameters() if p.requires_grad}
    w_pre, _ = load_weights_only(os.path.join(root, "export_pre"))
    load_weights(agent.model, w_pre, strict=False)
    agent.model.mark_adapters_updated()
    pre = [agent.policy_distribution(it, c, goal_text=g) for it, c, g, *_ in rows]
    with torch.no_grad():
        params = dict(agent.model.named_parameters())
        for n, t in saved.items():
            params[n].copy_(t)
    agent.model.mark_adapters_updated()
    out: Dict[str, Any] = {"n_decisions": len(rows)}
    for name, pis in (("pre_experience_checkpoint", pre), ("experienced_checkpoint", post)):
        ws, num, mod, tru = [], 0.0, 0.0, 0.0
        for pi, (_, _, _, mu, i, rew, tc) in zip(pis, rows):
            pi = [float(x) for x in pi]
            if mu[i] <= eps:
                continue
            w = pi[i] / mu[i]
            ws.append(w)
            num += w * rew
            mod += sum(p * (c.get("goal_mean") or 0.0) for p, c in zip(pi, tc))
            tru += sum(p * c["truth"] for p, c in zip(pi, tc))
        n = len(ws)
        out[name] = {"ips_observed": num / n if n else None,
                     "snips_observed": (num / sum(ws)) if ws and sum(ws) > 0 else None,
                     "ess_fraction": ((sum(ws) ** 2) / sum(w * w for w in ws) / n) if ws else None,
                     "max_weight": max(ws) if ws else None, "model_prediction": mod / n if n else None,
                     "ground_truth_same_contexts": tru / n if n else None}
    beh = [rew for *_, rew, _ in rows]
    out["behaviour_observed_mean"] = sum(beh) / len(beh)
    out["tv_pre_vs_post_mean"] = sum(0.5 * float((a - b).abs().sum()) for a, b in zip(pre, post)) / len(rows)
    return out


def _count(xs: Any) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def pathway_ablation(root: str) -> Dict[str, Any]:
    """Check 5 for a run without the meta pathway (D / E_unverified): which groups changed pre -> post."""
    hp, hq = group_hashes(os.path.join(root, "export_pre")), group_hashes(os.path.join(root, "export_post"))
    return {g: hp.get(g) != hq.get(g) for g in sorted(set(hp) | set(hq))}


# ------------------------------------------------------------------------------------------------ Protocol v3 (X6)
def oe_result_for_checks(res: Dict[str, Any]) -> Dict[str, Any]:
    """A result of ``eval.oe_runner sequence`` in the form :func:`run_checks` reads: its meta-update records are
    the concatenation of every objective's records; there are no ground-truth decision rows (the open-ended
    families have no per-candidate truth), so the shadow evaluation reports ``n = 0``."""
    metas = [m for o in res.get("objectives", []) for m in (o.get("meta_updates") or [])]
    return {"condition": res.get("condition"), "seed": res.get("seed"), "extract": {"meta_updates": metas},
            "decision_rows": []}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m cam_live.eval.meta_checks --root RUN_ROOT --result RESULT.json --out OUT.json``: the 16.5 checks
    on one X1 run root (export_pre / export_post / ckpt_post / agent of ``oe_runner --save-exports``) under the
    shared GPU lock (the experienced agent is rebuilt with the real backbone)."""
    import argparse
    import contextlib
    import sys
    import time
    ap = argparse.ArgumentParser(prog="python -m cam_live.eval.meta_checks")
    ap.add_argument("--root", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-support", dest="max_support", type=int, default=16)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--lock-timeout-s", dest="lock_timeout_s", type=float, default=6 * 3600.0)
    ap.add_argument("--min-free-gb", dest="min_free_gb", type=float, default=12.0)
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    torch.set_num_threads(int(a.threads))
    from .experiments import code_identity
    with open(a.result) as f:
        res = json.load(f)
    with open(os.path.join(a.root, "export_post", "config.json")) as f:
        ecfg = json.load(f)["config"]
    # the lock only when the experienced agent is rebuilt on the GPU (a debug / tiny-backbone run stays on the CPU)
    on_gpu = torch.cuda.is_available() and str(ecfg.get("device", "")).startswith("cuda") and \
        (ecfg.get("backbone") or {}).get("name") != "tiny"
    lock: Any = contextlib.nullcontext()
    if on_gpu:
        from ..nn.gpu import free_gpu_memory, gpu_lock
        lock = gpu_lock(timeout_s=a.lock_timeout_s, min_free_gb=a.min_free_gb)
    t0 = time.time()
    with lock:
        t_lock = time.time()
        out = run_checks(os.path.abspath(a.root), oe_result_for_checks(res), max_support=a.max_support)
        held = time.time() - t_lock
        if on_gpu:
            free_gpu_memory()
    out.update({"root": os.path.abspath(a.root), "result": os.path.abspath(a.result), "gpu": on_gpu,
                "lock_waited_s": t_lock - t0,
                "lock_held_s": held, "code": code_identity(),
                "run_code": res.get("code"), "plan": res.get("plan")})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: out.get(k) for k in ("condition", "seed", "n_eligible", "skipped", "lock_held_s")}))
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
