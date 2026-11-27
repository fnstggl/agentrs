"""A reward meta-update with a non-zero meta-gradient inside the integrated agent on the real configuration.

Qwen3-4B (frozen) + LoRA + bridge + 50m core on the GPU under the shared lock. The integrated `CamLiveAgent` runs
LabWorld F1 lifetimes through the real runtime (`cam_live.eval.runner.run_lifetime`: ground-truth tap, the
verifier's signed measurements of every due outcome and its final verdict delivered after the decision budget).
`validation.MetaStepProbe` wraps the agent's meta-learner and records, for every *applied* outer step, the full
pathway:

    reward params psi -> internal reward on the support decisions (before / after the step)
    -> inner update theta_0 -> theta_K (policy change on the support decisions)
    -> changed policy -> later verified outcomes of the query decisions (utility statistics, ESS)
    -> outer meta-gradient norm -> reward-parameter delta
    -> effect on the next inner update (theta_K recomputed from the same snapshot with psi before vs after)

Every meta-update record (applied or skipped, with its reason) is kept. Output: one JSON evidence file.

    cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/validation/meta_real/run_meta_real.py \\
        --out cam_live/docs/evidence/validation/meta_real/meta_real_<ts>.json [--budget-s 780]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="")
    ap.add_argument("--budget-s", type=float, default=780.0)
    ap.add_argument("--decisions", type=int, default=12)
    ap.add_argument("--lifetimes", type=int, default=12)
    ap.add_argument("--agent", default="{}")
    ap.add_argument("--optim", default="{}")
    ap.add_argument("--min-independent", type=int, default=4)
    a = ap.parse_args(argv)
    import torch
    from cam_live.agent import CamLiveAgent
    from cam_live.contracts import RunFiles, SimClock
    from cam_live.eval.runner import run_lifetime
    from cam_live.nn.gpu import gpu_lock
    from cam_live.runtime.logs import iter_jsonl
    from cam_live.validation import MetaStepProbe, _git_head, staging_config
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    work = os.path.abspath(a.work or os.path.splitext(a.out)[0] + "_work")
    os.makedirs(work, exist_ok=True)
    cfg = staging_config(None, True, 0)
    cfg.optim.meta_min_independent_outcomes = int(a.min_independent)
    for k, v in json.loads(a.optim).items():
        setattr(cfg.optim, k, v)
    cfg.sections["agent"].update(json.loads(a.agent))
    report = {"kind": "real_model_meta_update", "executed": False, "started_utc": stamp, "git_head": _git_head(),
              "backbone": cfg.backbone.name, "core_scale": cfg.core.scale, "agent_settings": cfg.section("agent"),
              "optim": {k: getattr(cfg.optim, k) for k in ("inner_lr", "inner_steps", "lr_reward",
                                                           "meta_min_independent_outcomes")},
              "lifetimes": [], "meta_updates": [], "applied_pathway": []}
    t_start = time.time()
    with gpu_lock(min_free_gb=18.0):
        t_lock = time.time()
        torch.cuda.reset_peak_memory_stats()
        clock = SimClock()
        agent = CamLiveAgent(cfg, os.path.join(work, "agent"), clock=clock)
        probe = MetaStepProbe(agent)
        report["build_seconds"] = time.time() - t_lock
        report["parameters"] = {k: v for k, v in agent.model.count_parameters().items()
                                if k != "experiential_breakdown"}
        try:
            for i in range(a.lifetimes):
                if time.time() - t_lock > a.budget_s:
                    report["stopped"] = f"lock budget {a.budget_s:.0f} s reached before lifetime {i}"
                    break
                h0 = agent.model.group_hashes()["reward"]
                lt = run_lifetime(agent, "F1", 10_000 + i, os.path.join(work, f"lt{i:02d}"), clock,
                                  max_decisions=a.decisions, k=16)
                metas = [e["data"] for e in iter_jsonl(os.path.join(lt["runtime_dir"], RunFiles.META))]
                close = (lt.get("close_out") or {}).get("meta_updates") or []
                report["meta_updates"] += [{k: m.get(k) for k in ("applied", "skipped_reason", "meta_grad_norm",
                                                                    "n_independent_outcomes", "n_query", "estimator",
                                                                    "reward_version_before", "reward_version_after")}
                                           for m in metas] + [dict(m, at="close_out") for m in close]
                report["lifetimes"].append({"index": i, "progress": lt["progress"], "quality": lt["quality"],
                                            "decisions": lt["decisions"], "verified_status": lt["verified_status"],
                                            "reward_hash_changed": agent.model.group_hashes()["reward"] != h0,
                                            "wall_seconds": lt["wall_seconds"],
                                            "proposal": (agent.last.get("proposal") or {}).get("stats", {})})
                report["applied_pathway"] = list(probe.rows)
                print(f"lt {i}: progress {lt['progress']:.2f} decisions {lt['decisions']} applied "
                      f"{sum(1 for m in report['meta_updates'] if m.get('applied'))} wall {lt['wall_seconds']:.0f}s "
                      f"lock {time.time() - t_lock:.0f}s", flush=True)
                with open(a.out, "w") as f:
                    json.dump(report, f, indent=1, default=str)
        finally:
            report["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            report["lock_seconds"] = time.time() - t_lock
            report["agent_status"] = agent.status()
            agent.close()
            del agent
            torch.cuda.empty_cache()
    applied = [r for r in report["applied_pathway"] if (r.get("meta_grad_norm") or 0) > 0]
    report["executed"] = True
    report["wall_seconds"] = time.time() - t_start
    report["result"] = {
        "meta_updates": len(report["meta_updates"]),
        "applied_with_nonzero_gradient": len(applied),
        "skipped_reasons": {r: sum(1 for m in report["meta_updates"] if not m.get("applied")
                                   and (m.get("skipped_reason") or "").split(":")[0] == r)
                            for r in sorted({(m.get("skipped_reason") or "").split(":")[0]
                                             for m in report["meta_updates"] if not m.get("applied")})},
        "reward_param_delta_norms": [r.get("reward_param_delta_norm") for r in applied],
        "policy_change_by_meta_step_max_abs_prob": [(r.get("policy_change_by_meta_step") or {})
                                                    .get("max_abs_prob_diff") for r in applied]}
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report["result"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
