"""RW (amendment item 1): experience -> meta-update of R_psi -> evidence-weighted R_psi -> changed live policy update ->
changed decisions, inside ONE long objective, on the integrated agent (``CamLiveAgent`` through ``LiveRuntime``).

    cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/RW/rw_pathway.py --arm learned \\
        --out cam_live/docs/evidence/RW/<name>.json [--debug] [--decisions 60] [--budget-s 780]

Arms (same seeds, same world): ``learned`` — the configuration's defaults (RW: ``promotion_rule="validity"``,
``estimator="auto"``, ``utility="direct"``); ``frozen`` — identical, but the reward optimizer's step is 0 (R_psi stays at its
initialisation; every meta-update is computed and refused as ``psi_unchanged_by_the_step``).

World: one LabWorld objective (the controlled staging world of ``tasks/lab_staging.yaml``: verifier-signed adopter
counts attributed to the interventions that caused them) made long — a target that is not reached within the run,
budget and deadline far beyond it — so all learning happens inside one objective, never across lifetimes.

Recorded (nothing of it feeds back into the run):
* every live policy update: the mix in force; the update recomputed from its own snapshot at mix 1 (bootstrap only)
  and the mean TV on its support between the applied and the mix-1 policy ("R_psi changed this live update");
  a shadow chain of the same updates at mix 1 and, on the decisions taken since the previous update, the TV between
  the policy that took them and the shadow policy ("the policy that decided was not the bootstrap-trained one");
* every meta-update record (applied / reason / estimator / gradient norm);
* the validity evidence and every mix decision; per decision: chosen operation and arguments, the selection
  propensities, the policy's share of the selection (``policy_share_tv``), the ground truth of every candidate
  (``TruthTap``, never visible to the agent) and the mix in force.
``--debug`` runs the tiny CPU configuration (plumbing; never reported as the real model).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

DAY = 86400.0


def _tv(p, q):
    return 0.5 * (p - q).abs().sum(-1)


def _collect(report, meta, agent, task, dtap, verified_progress) -> None:
    report["mix_decisions"] = [{"action": d.action, "reason": d.reason, "mix_before": d.mix_before,
                                "mix_after": d.mix_after, "n_pairs": d.n_independent, "mean": d.mean_diff,
                                "lower": d.lower_bound, "upper": d.upper_bound} for d in meta.mixer.decisions]
    report["validity"] = meta.validity.summary()
    report["validity_recent_pairs"] = list(meta.validity.recent)[-40:]
    report["mix_end"] = meta.mixer.mix
    report["label_end"] = meta.mixer.label
    report["meta_stats"] = dict(meta.stats)
    recs = sorted(agent.store.trajectory(task.objective_id), key=lambda r: r.seq)
    by_dec = {r.decision.decision_id: r for r in recs}
    for k, row in enumerate(dtap.rows):
        r = by_dec.get(row["decision_id"])
        sel = ((r.decision.diagnostics or {}).get("selection") or {}) if r is not None else {}
        tr = [c["truth"] for c in row["candidates"]]
        ch = row["candidates"][row["chosen_index"]]
        report["decisions"].append({
            "i": k, "operation": ch["operation"], "arguments": ch["arguments"], "chosen_index": row["chosen_index"],
            "probabilities": row["probabilities"], "truth": tr, "truth_chosen": ch["truth"],
            "expected_quality": (sum(p * t for p, t in zip(row["probabilities"], tr)) / max(tr)) if max(tr) > 0
            else None, "policy_share_tv": sel.get("policy_share_tv"),
            "policy_weight_mean": (sel.get("policy_weight") or {}).get("mean"),
            "verified_progress": verified_progress(r) if r is not None else None,
            "reward_version": r.reward_version if r is not None else None})


def _result(report) -> None:
    mu, lu, dec = report["meta_updates"], report["live_updates"], report["decisions"]
    val = report.get("validity") or {}
    report["result"] = {
        "decisions": len(dec), "meta_update_calls": len(mu), "meta_applied": sum(1 for m in mu if m["applied"]),
        "skipped": {r: sum(1 for m in mu if not m["applied"] and (m["reason"] or "").split(":")[0] == r)
                    for r in sorted({(m["reason"] or "").split(":")[0] for m in mu if not m["applied"]})},
        "mix_trajectory": [(d["mix_before"], d["mix_after"], d["action"], d["n_pairs"])
                           for d in report.get("mix_decisions", []) if d["action"] != "hold"],
        "mix_end": report.get("mix_end"), "validity_pairs": val.get("pairs"),
        "concordance_learned": val.get("concordance_learned"), "concordance_bootstrap": val.get("concordance_bootstrap"),
        "live_updates": len(lu),
        "live_updates_changed_by_R_psi": [round(u.get("R_psi_changed_this_update_mean_tv", 0.0), 5) for u in lu],
        "deciding_policy_vs_bootstrap_trained": [round(u["deciding_policy_vs_bootstrap_trained_mean_tv"], 5)
                                                 for u in lu if "deciding_policy_vs_bootstrap_trained_mean_tv" in u],
        "verified_adopters": sum(d["verified_progress"] or 0.0 for d in dec),
        "truth_chosen_mean": (sum(d["truth_chosen"] for d in dec) / len(dec)) if dec else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=("learned", "frozen"), default="learned")
    ap.add_argument("--estimator", default="", help="outer estimator override (default: the configuration's)")
    ap.add_argument("--decisions", type=int, default=60)
    ap.add_argument("--budget-s", type=float, default=780.0, help="real model: wall seconds of the GPU hold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world-seed", type=int, default=31_000)
    ap.add_argument("--min-independent", type=int, default=4)
    ap.add_argument("--agent", default="{}")
    ap.add_argument("--meta", default="{}")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(argv)
    import torch
    from cam_live.agent import CamLiveAgent
    from cam_live.contracts import SimClock
    from cam_live.diagnostics.world import AllowAllPolicy, LabConfig, LabVerifier, LabWorld, lab_task
    from cam_live.eval.conditions import Timed
    from cam_live.eval.envs import TruthTap
    from cam_live.eval.runner import DecisionTap, close_out, runtime_config
    from cam_live.reward.inner import verified_progress
    from cam_live.runtime.loop import LiveRuntime
    from cam_live.validation import _git_head, staging_config
    real = not a.debug
    t_start = time.time()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    work = os.path.abspath(os.path.splitext(a.out)[0] + "_work")
    os.makedirs(work, exist_ok=True)
    cfg = staging_config(None, real, a.seed, meta_min_independent=a.min_independent)
    cfg.sections["agent"].update(json.loads(a.agent))
    meta_over = {**({"estimator": a.estimator} if a.estimator else {}), **json.loads(a.meta)}
    if a.arm == "frozen":
        meta_over["lr_reward"] = 0.0
    rw = dict(cfg.sections.get("reward", {}))
    rw["meta"] = {**rw.get("meta", {}), **meta_over}
    cfg.sections["reward"] = rw
    report = {"kind": ("real_model" if real else "debug") + "_rw_pathway", "arm": a.arm, "executed": False,
              "started_utc": stamp, "git_head": _git_head(), "backbone": cfg.backbone.name,
              "core_scale": cfg.core.scale, "seed": a.seed, "world_seed": a.world_seed,
              "agent_settings": cfg.section("agent"), "reward_meta_overrides": meta_over,
              "live_updates": [], "meta_updates": [], "mix_decisions": [], "decisions": []}
    lock = None
    if real:
        from cam_live.nn.gpu import gpu_lock
        lock = gpu_lock(timeout_s=float(os.environ.get("RW_LOCK_WAIT_S", "1800")), min_free_gb=18.0)
        lock.__enter__()
        torch.cuda.reset_peak_memory_stats()
    t_lock = time.time()
    clock = SimClock()
    agent = CamLiveAgent(cfg, os.path.join(work, "agent"), clock=clock)
    meta = agent.meta
    report["build_seconds"] = time.time() - t_lock
    report["selection_rule"] = agent.settings.selection_rule
    report["use_backbone"] = bool(agent.settings.use_backbone)
    report["meta_config"] = {k: getattr(meta.mc, k) for k in ("estimator", "utility", "promotion_rule", "validity_alpha",
                                                              "mixer_step", "lr_reward", "min_independent",
                                                              "target_kl", "meta_mix", "pool_size")}
    report["meta_config"]["heldout"] = meta.heldout
    lc = LabConfig(seed=a.world_seed, n_channels=4, target=10_000, budget=1000.0, horizon_s=60 * DAY,
                   outcome_horizon_s=6 * 3600.0, mean_delay_s=(1800.0, 4 * 3600.0), at_most_once=True,
                   family="lab/first")
    world = LabWorld(lc, clock)
    task = lab_task(world, f"obj_rw_long_{a.world_seed}")
    task.limits.max_decisions = int(a.decisions)
    verifier = LabVerifier(world)
    tap = TruthTap(world)
    agent.attach_verifier(verifier)
    timed = Timed(agent)
    dtap = DecisionTap(timed, tap)
    rt = LiveRuntime(dtap, tap, AllowAllPolicy(), verifier, task, os.path.join(work, "rt"), clock, runtime_config(16))

    orig_pol, orig_meta = meta.policy_update, meta.meta_update
    shadow = {"theta": None}            # policy + trunk trained at mix 1 on the same supports (exists once mix < 1)

    def wrapped_pol(records, **kw):
        mix = meta.mixer.mix
        psi = {n: p.detach().clone() for n, p in meta.reward.psi().items()}
        up = orig_pol(records, **kw)
        if up is None:
            return up
        ev = meta.events[-1]
        row = {"event_id": ev.event_id, "decision_index": len(dtap.rows), "mix": mix, "step": ev.inner_lr,
               "kl": up.prediction_change.get("policy_kl"), "n_support": len(ev.exp_ids)}
        try:
            dev = next(iter(psi.values())).device
            th0 = {n: v.to(dev) for n, v in ev.theta0.items()}
            ib = ev.ib
            new_rows = [k for k, i in enumerate(ib.exp_ids) if i in set(r.exp_id for r in records)]
            with meta.reward.meta_window():
                t_live = meta.engine.inner_update(th0, psi, [ib], ev.inner_lr, ev.steps, create_graph=False).detached()
                t_b = meta.engine.inner_update(th0, psi, [ib.with_mix(1.0)], ev.inner_lr, ev.steps,
                                               create_graph=False).detached()
            with torch.no_grad():
                p_live, p_b = meta.inner.probs(t_live, ib).double(), meta.inner.probs(t_b, ib).double()
            row["R_psi_changed_this_update_mean_tv"] = float(_tv(p_live, p_b).mean())
            row["R_psi_changed_this_update_max_tv"] = float(_tv(p_live, p_b).max())
            if shadow["theta"] is None and mix < 1.0:
                shadow["theta"] = {n: v.detach().clone() for n, v in th0.items()}
            if shadow["theta"] is not None:
                sub = ib.select(new_rows) if new_rows else None
                if sub is not None:
                    with torch.no_grad():
                        p_dec = meta.inner.probs(th0, sub).double()
                        p_sh = meta.inner.probs(shadow["theta"], sub).double()
                    row["deciding_policy_vs_bootstrap_trained_mean_tv"] = float(_tv(p_dec, p_sh).mean())
                with meta.reward.meta_window():
                    shadow["theta"] = meta.engine.inner_update(shadow["theta"], psi, [ib.with_mix(1.0)], ev.inner_lr,
                                                               ev.steps, create_graph=False).detached()
        except Exception as e:  # noqa: BLE001 — evidence only
            row["error"] = f"{type(e).__name__}: {e}"
        report["live_updates"].append(row)
        return up

    def wrapped_meta(now=None):
        rec = orig_meta(now)
        report["meta_updates"].append({"decision_index": len(dtap.rows), "applied": rec.applied,
                                       "reason": rec.skipped_reason, "estimator": rec.estimator,
                                       "meta_grad_norm": rec.meta_grad_norm, "n_groups": rec.n_independent_outcomes,
                                       "n_query": rec.n_query, "mix": rec.bootstrap_mix,
                                       "ess": (rec.diagnostics or {}).get("ess")})
        return rec

    meta.policy_update, meta.meta_update = wrapped_pol, wrapped_meta
    n_dec_seen = 0
    longest = 0.0
    try:
        rt.start()
        steps, t_prev = 0, time.time()
        while rt.status == "running" and steps < 6 * a.decisions:
            if real and time.time() - t_lock + 1.5 * longest + 150.0 > a.budget_s:
                rt._finish("stopped", f"GPU hold budget {a.budget_s:.0f} s")
                report["stopped"] = f"hold budget: {time.time() - t_lock:.0f} s used, longest step {longest:.0f} s"
                break
            rt.step()
            steps += 1
            longest = max(longest, time.time() - t_prev)
            t_prev = time.time()
            if len(dtap.rows) > n_dec_seen:
                n_dec_seen = len(dtap.rows)
                print(f"decision {n_dec_seen} mix {meta.mixer.mix:.2f} meta applied "
                      f"{int(meta.stats['meta_applied'])} pairs {meta.validity.n_pairs} "
                      f"wall {time.time() - t_lock:.0f}s", flush=True)
        if rt.status == "running":
            rt._finish("stopped", "step cap")
        rt.close()
        report["closing"] = close_out(timed, world, verifier, task)
    except Exception as e:  # noqa: BLE001 — the run's error is evidence too; the report is still written
        import traceback
        report["run_error"] = f"{type(e).__name__}: {e}"
        report["run_traceback"] = traceback.format_exc()[-4000:]
    finally:
        try:
            _collect(report, meta, agent, task, dtap, verified_progress)
        except Exception as e:  # noqa: BLE001 — evidence only; the hold is released below in any case
            report["report_error"] = f"{type(e).__name__}: {e}"
        if real:
            report["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        report["lock_seconds"] = time.time() - t_lock
        try:
            report["agent_status"] = agent.status()
        except Exception as e:  # noqa: BLE001
            report["agent_status"] = f"{type(e).__name__}: {e}"
        try:
            agent.close()
        finally:
            del agent
            if lock is not None:
                lock.__exit__(None, None, None)
                torch.cuda.empty_cache()
        report["executed"] = True
        report["wall_seconds"] = time.time() - t_start
        try:
            _result(report)
        except Exception as e:  # noqa: BLE001
            report["result_error"] = f"{type(e).__name__}: {e}"
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1, default=str)
    print(json.dumps(report.get("result"), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
