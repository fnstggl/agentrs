"""The reward meta-learning pathway inside the integrated agent, end to end, on the real configuration (workstream
LRN, task item 4): Qwen3-4B (frozen, bf16) + LoRA + bridge + 50m core on the GPU under the shared lock.

    reward params psi -> internal reward R_psi on the support decisions -> inner policy update (live, at the live mix,
    and the differentiated one) -> changed policy -> decisions taken afterwards and their verifier-signed outcomes
    -> outer meta-gradient -> psi delta -> the *next live* inner update, recomputed from its own snapshot with psi
    before vs after the meta-step (did the meta-step change the update that was actually applied next?)

One ``CamLiveAgent`` (selection rule ``thompson_policy``) runs LabWorld F1 lifetimes through the real runtime
(``cam_live.eval.runner.run_lifetime``: ground-truth tap, the verifier's signed measurements of every due outcome and
its final verdict after the decision budget). ``meta_update`` and ``policy_update`` of the agent's meta-learner are
wrapped (nothing else changes) to record every applied outer step and every live inner update; the bootstrap mix,
every promotion decision and every skipped meta-update with its reason are kept. Output: one JSON evidence file.

    cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/C/real_pathway.py \\
        --out cam_live/docs/evidence/C/real_pathway_<ts>.json [--budget-s 780] [--debug]

``--debug`` runs the same on the tiny CPU configuration (plumbing check only; never reported as the real model).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time


def _tv(p, q):
    return 0.5 * (p - q).abs().sum(-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="")
    ap.add_argument("--budget-s", type=float, default=780.0)
    ap.add_argument("--decisions", type=int, default=12)
    ap.add_argument("--lifetimes", type=int, default=12)
    ap.add_argument("--agent", default='{"selection_rule": "thompson_policy"}')
    ap.add_argument("--min-independent", type=int, default=4)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--meta", default="{}", help="JSON overrides of reward.meta (e.g. a fixed mix for a pathway check)")
    a = ap.parse_args(argv)
    import torch
    from cam_live.agent import CamLiveAgent
    from cam_live.contracts import RunFiles, SimClock
    from cam_live.eval.runner import run_lifetime
    from cam_live.runtime.logs import iter_jsonl
    from cam_live.storage import DiskGuard
    from cam_live.validation import _git_head, staging_config
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    work = os.path.abspath(a.work or os.path.splitext(a.out)[0] + "_work")
    os.makedirs(work, exist_ok=True)
    DiskGuard(work).check()
    real = not a.debug
    cfg = staging_config(None, real, 0)
    cfg.optim.meta_min_independent_outcomes = int(a.min_independent)
    cfg.sections["agent"].update(json.loads(a.agent))
    rw = dict(cfg.sections.get("reward", {}))
    rw["meta"] = {**rw.get("meta", {}), **json.loads(a.meta)}
    cfg.sections["reward"] = rw
    report = {"kind": "real_model_reward_meta_pathway" if real else "debug_reward_meta_pathway", "executed": False,
              "started_utc": stamp, "git_head": _git_head(), "backbone": cfg.backbone.name,
              "core_scale": cfg.core.scale, "agent_settings": cfg.section("agent"),
              "optim": {k: getattr(cfg.optim, k) for k in ("inner_lr", "inner_steps", "lr_reward",
                                                           "meta_min_independent_outcomes")},
              "reward_meta_overrides": json.loads(a.meta),
              "lifetimes": [], "meta_updates": [], "applied": [], "live_updates": [], "promotions": [],
              "checkpoint_rollbacks": []}
    lock = None
    t_start = time.time()
    if real:
        from cam_live.nn.gpu import gpu_lock
        lock = gpu_lock(min_free_gb=18.0)
        lock.__enter__()
        torch.cuda.reset_peak_memory_stats()
    t_lock = time.time()
    clock = SimClock()
    agent = CamLiveAgent(cfg, os.path.join(work, "agent"), clock=clock)
    meta = agent.meta
    report["build_seconds"] = time.time() - t_lock
    report["selection_rule"] = agent.settings.selection_rule
    report["meta_config"] = {k: getattr(meta.mc, k) for k in ("inner_lr", "target_kl", "max_inner_lr",
                                                              "replay_matured", "mixer_step", "mixer_min_independent",
                                                              "mixer_rel_margin", "meta_mix", "heldout_fraction")}
    pending_next = []                                  # applied meta-steps waiting for the next live update

    orig_meta, orig_pol = meta.meta_update, meta.policy_update

    def psi_now():
        return {n: p.detach().clone() for n, p in meta.reward.psi().items()}

    def wrapped_meta(now=None):
        psi0 = psi_now()
        rec = orig_meta(now)
        row = {k: getattr(rec, k) for k in ("applied", "skipped_reason", "meta_grad_norm", "n_independent_outcomes",
                                            "n_query", "estimator", "bootstrap_mix", "reward_version_before",
                                            "reward_version_after")}
        report["meta_updates"].append(row)
        if rec.applied:
            try:
                psi1 = psi_now()
                delta = math.sqrt(sum(float((psi1[n] - psi0[n]).double().pow(2).sum()) for n in psi0))
                ev = next((e for e in reversed(meta.events) if e.theta0 and not e.external), None)
                out = {"meta_id": rec.meta_id, "meta_grad_norm": rec.meta_grad_norm, "estimator": rec.estimator,
                       "n_independent_outcomes": rec.n_independent_outcomes, "n_query": rec.n_query,
                       "reward_param_delta_norm": delta, "reward_version_before": rec.reward_version_before,
                       "reward_version_after": rec.reward_version_after, "mix": meta.mixer.mix,
                       "query_evidence": {k: rec.diagnostics.get(k) for k in (
                           "utility_mean", "n_distinct_utilities", "max_abs_advantage", "ess", "n_episodes",
                           "policy_prob_change_on_query", "psi_tensors_with_gradient", "promotion",
                           "promotion_mean_diff", "promotion_lower_bound", "promotion_upper_bound")}}
                if ev is not None:
                    dev = next(iter(psi1.values())).device
                    th0 = {n: v.to(dev) for n, v in ev.theta0.items()}
                    ib = ev.ib if ev.ib is not None else meta.inner.build_batch(meta.store, [
                        r for r in (meta.store.get(i) for i in ev.exp_ids) if r is not None])
                    with torch.no_grad():
                        rb = meta.inner.learned_reward({n: v.to(dev) for n, v in psi0.items()}, ib).double()
                        ra = meta.inner.learned_reward(psi1, ib).double()
                    out["internal_reward_on_support"] = {"rows": int(rb.numel()),
                                                         "mean_abs_change": float((ra - rb).abs().mean()),
                                                         "std_before": float(rb.std()) if rb.numel() > 1 else 0.0}
                    for name, mix in (("live_mix", ev.mix), ("differentiated_mix", meta.mc.meta_mix
                                                             if meta.mc.meta_mix is not None else ev.mix)):
                        b_ = ib.with_mix(mix)
                        tb = meta.engine.inner_update(th0, {n: v.to(dev) for n, v in psi0.items()}, [b_],
                                                      ev.inner_lr, ev.steps, create_graph=False).detached()
                        ta = meta.engine.inner_update(th0, psi1, [b_], ev.inner_lr, ev.steps,
                                                      create_graph=False).detached()
                        with torch.no_grad():
                            p0 = meta.inner.probs(th0, b_).double()
                            pb, pa = meta.inner.probs(tb, b_).double(), meta.inner.probs(ta, b_).double()
                        out[f"inner_update_at_{name}"] = {
                            "mix": float(mix), "step": ev.inner_lr,
                            "policy_change_theta0_to_thetaK_mean_tv": float(_tv(pb, p0).mean()),
                            "change_caused_by_meta_step_mean_tv": float(_tv(pa, pb).mean()),
                            "change_caused_by_meta_step_max_abs_prob": float((pa - pb).abs().max())}
                report["applied"].append(out)
                pending_next.append((len(report["applied"]) - 1, psi0, psi1))
            except Exception as e:  # noqa: BLE001 — evidence, never a reason to stop
                report["applied"].append({"meta_id": rec.meta_id, "error": f"{type(e).__name__}: {e}"})
        return rec

    def wrapped_pol(records, **kw):
        psi_used = psi_now()
        mix = meta.mixer.mix
        up = orig_pol(records, **kw)
        if up is None:
            return up
        ev = meta.events[-1]
        row = {"event_id": ev.event_id, "mix": mix, "step": ev.inner_lr, "kl": up.prediction_change.get("policy_kl"),
               "weight_delta": up.weight_delta, "reward_version": ev.reward_version,
               "diagnostics": {k: (up.diagnostics or {}).get(k) for k in (
                   "n_replayed_matured", "goal_verified_rows", "goal_predicted_rows")}}
        # this live update depends on psi with weight (1 - mix): recompute it with psi perturbed in shape
        try:
            dev = next(iter(psi_used.values())).device
            th0 = {n: v.to(dev) for n, v in ev.theta0.items()}
            g = torch.Generator(device="cpu").manual_seed(ev.event_id)
            pert = {n: (v + 0.5 * v.float().std(unbiased=False).to(v.dtype)
                        * torch.randn(v.shape, generator=g).to(dev, v.dtype) if n.startswith("out.") else v)
                    for n, v in psi_used.items()}
            ib = ev.ib
            ta = meta.engine.inner_update(th0, psi_used, [ib], ev.inner_lr, ev.steps, create_graph=False).detached()
            tp = meta.engine.inner_update(th0, pert, [ib], ev.inner_lr, ev.steps, create_graph=False).detached()
            with torch.no_grad():
                pa, pp = meta.inner.probs(ta, ib).double(), meta.inner.probs(tp, ib).double()
            row["dependence_on_R_psi_shape_mean_tv"] = float(_tv(pa, pp).mean())
            # the next live update after an applied meta-step: recomputed with the psi before that step
            while pending_next:
                k, psi0, psi1 = pending_next.pop(0)
                tb = meta.engine.inner_update(th0, {n: v.to(dev) for n, v in psi0.items()}, [ib], ev.inner_lr,
                                              ev.steps, create_graph=False).detached()
                with torch.no_grad():
                    pb = meta.inner.probs(tb, ib).double()
                report["applied"][k]["next_live_update"] = {
                    "event_id": ev.event_id, "mix": mix, "step": ev.inner_lr,
                    "psi_used_is_psi_after_meta_step": all(torch.equal(psi_used[n], psi1[n].to(psi_used[n].device))
                                                           for n in psi_used),
                    "psi_used_minus_psi_after_meta_step_norm": math.sqrt(sum(
                        float((psi_used[n] - psi1[n].to(psi_used[n].device)).double().pow(2).sum()) for n in psi_used)),
                    "change_caused_by_meta_step_mean_tv": float(_tv(pa, pb).mean()),
                    "change_caused_by_meta_step_max_abs_prob": float((pa - pb).abs().max())}
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        report["live_updates"].append(row)
        return up

    meta.meta_update, meta.policy_update = wrapped_meta, wrapped_pol
    try:
        longest = 0.0
        for i in range(a.lifetimes):
            if time.time() - t_lock + longest > a.budget_s:      # never start a lifetime that would overrun the hold
                report["stopped"] = (f"lock budget {a.budget_s:.0f} s: {time.time() - t_lock:.0f} s used, the longest "
                                     f"lifetime took {longest:.0f} s, not starting lifetime {i}")
                break
            t_lt = time.time()
            n_rb = int(agent.counters.get("rollbacks", 0))
            n_dec = len(meta.mixer.decisions)
            lt = run_lifetime(agent, "F1", 10_000 + i, os.path.join(work, f"lt{i:02d}"), clock,
                              max_decisions=a.decisions, k=16)
            report["promotions"] += [{"lifetime": i, "action": x.action, "reason": x.reason,
                                      "mix_before": x.mix_before, "mix_after": x.mix_after, "n": x.n_independent,
                                      "mean_diff": x.mean_diff, "lower": x.lower_bound, "upper": x.upper_bound,
                                      "margin": getattr(x, "margin", None)} for x in meta.mixer.decisions[n_dec:]]
            longest = max(longest, time.time() - t_lt)
            if int(agent.counters.get("rollbacks", 0)) > n_rb:
                report["checkpoint_rollbacks"].append({"lifetime": i, "n": int(agent.counters["rollbacks"]) - n_rb})
            rows = lt["decision_rows"]
            q = []
            for r in rows:
                tr = [c["truth"] for c in r["candidates"]]
                if max(tr) > 0:
                    q.append(sum(p * t for p, t in zip(r["probabilities"], tr)) / max(tr))
            report["lifetimes"].append({"index": i, "progress": lt["progress"], "decisions": lt["decisions"],
                                        "verified_status": lt["verified_status"], "wall_seconds": lt["wall_seconds"],
                                        "expected_choice_quality": sum(q) / len(q) if q else None,
                                        "mix_end": meta.mixer.mix,
                                        "calibration": agent.decider.calibration.summary()})
            print(f"lt {i}: progress {lt['progress']:.2f} E[q] {report['lifetimes'][-1]['expected_choice_quality']} "
                  f"mix {meta.mixer.mix:.2f} applied {len(report['applied'])} wall {lt['wall_seconds']:.0f}s "
                  f"lock {time.time() - t_lock:.0f}s", flush=True)
            with open(a.out, "w") as f:
                json.dump(report, f, indent=1, default=str)
    finally:
        if real:
            report["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        report["lock_seconds"] = time.time() - t_lock
        report["agent_status"] = agent.status()
        agent.close()
        del agent
        if lock is not None:
            lock.__exit__(None, None, None)
            torch.cuda.empty_cache()
    ok = [r for r in report["applied"] if "error" not in r]
    nxt = [r["next_live_update"] for r in ok if "next_live_update" in r]
    report["executed"] = True
    report["wall_seconds"] = time.time() - t_start
    report["result"] = {
        "meta_update_calls": len(report["meta_updates"]),
        "applied": len(ok), "applied_with_nonzero_gradient": sum(1 for r in ok if (r["meta_grad_norm"] or 0) > 0),
        "skipped_reasons": {r: sum(1 for m in report["meta_updates"] if not m["applied"]
                                   and (m["skipped_reason"] or "").split(":")[0] == r)
                            for r in sorted({(m["skipped_reason"] or "").split(":")[0]
                                             for m in report["meta_updates"] if not m["applied"]})},
        "reward_param_delta_norms": [r["reward_param_delta_norm"] for r in ok],
        "internal_reward_change_on_support": [(r.get("internal_reward_on_support") or {}).get("mean_abs_change")
                                              for r in ok],
        "next_live_update_change_by_meta_step_mean_tv": [x["change_caused_by_meta_step_mean_tv"] for x in nxt],
        "mix_trajectory": [lt["mix_end"] for lt in report["lifetimes"]],
        "promotion_actions": [p["action"] for p in report["promotions"]],
        "checkpoint_rollbacks": sum(r["n"] for r in report["checkpoint_rollbacks"]),
        "live_updates": len(report["live_updates"]),
        "live_update_dependence_on_R_psi_shape_mean_tv": [u.get("dependence_on_R_psi_shape_mean_tv")
                                                          for u in report["live_updates"]],
        "pathway_complete": bool(nxt) and any(x["change_caused_by_meta_step_mean_tv"] > 0 for x in nxt)}
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report["result"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
