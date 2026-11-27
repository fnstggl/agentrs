"""Why does the integrated agent choose near random in LabWorld F1? Measured on the agent's own decisions.

One agent (``--real``: Qwen3-4B + 50m under the GPU lock; default: the debug configuration, an engineering
diagnostic only) runs ``--lifetimes`` objectives of the S evaluation family F1 through ``cam_live.eval.runner
.run_lifetime`` (real runtime, ground-truth tap on every candidate, close-out to the deadline). Recorded per
decision and per lifetime:

* choice quality against the ground truth: truth(chosen) / max truth(C_t), and the same for a uniform choice over
  the offered set C_t; whether the best channel's intervention was offered at all; Spearman rank correlation between
  the policy's probabilities and the truth over C_t;
* the reward the inner policy update learns from: intrinsic / bootstrap reward of every executed decision and its
  components, against the truth of the chosen action (and separately for waits);
* the policy updates (``meta.policy_update``): weight-delta norm of the heads, losses; the reward meta-updates:
  applied / skipped reasons, gradient norms, reward-parameter hash before / after;
* verified progress and money efficiency per lifetime.

    .venv/bin/python cam_live/docs/evidence/validation/diagnosis/diagnose_f1.py --out OUT.json [--real]
        [--lifetimes 8] [--decisions 16] [--agent '{"selection_rule": "thompson"}'] [--label NAME]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time


def spearman(x, y):
    n = len(x)
    if n < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--lifetimes", type=int, default=8)
    ap.add_argument("--decisions", type=int, default=16)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent", default="{}")
    ap.add_argument("--label", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--budget-s", type=float, default=0.0)
    a = ap.parse_args(argv)
    import torch
    from cam_live.agent import CamLiveAgent
    from cam_live.contracts import RunFiles
    from cam_live.eval.runner import run_lifetime
    from cam_live.runtime.logs import iter_jsonl
    from cam_live.validation import OffsetClock, staging_config
    from cam_live.contracts import SimClock
    cfg = staging_config(None, a.real, a.seed)
    over = json.loads(a.agent)
    cfg.sections["agent"].update(over)
    work = os.path.abspath(a.work or os.path.splitext(a.out)[0] + "_work")
    os.makedirs(work, exist_ok=True)
    lock = None
    if a.real:
        from cam_live.nn.gpu import gpu_lock
        lock = gpu_lock(min_free_gb=18.0)
        lock.__enter__()
    clock = SimClock()
    agent = CamLiveAgent(cfg, os.path.join(work, "agent"), clock=clock)
    report = {"label": a.label, "real_model": a.real, "agent_settings": cfg.section("agent"),
              "credit_default": agent.assigner.default, "lifetimes": [], "started": time.time()}
    t0 = time.time()
    try:
        for i in range(a.lifetimes):
            if a.budget_s and time.time() - t0 > a.budget_s:
                report["stopped"] = f"budget {a.budget_s:.0f} s reached before lifetime {i}"
                break
            psi0 = agent.model.group_hashes().get("reward")
            lt = run_lifetime(agent, "F1", 10_000 + 100 * a.seed + i, os.path.join(work, f"lt{i:02d}"), clock,
                              max_decisions=a.decisions, k=a.k)
            psi1 = agent.model.group_hashes().get("reward")
            dec = []
            for row in lt["decision_rows"]:
                tr = [c["truth"] for c in row["candidates"]]
                p = row["probabilities"]
                ci = row["chosen_index"]
                best = max(tr) if tr else 0.0
                rec = agent.store.by_decision(row["decision_id"])
                chosen = row["candidates"][ci]
                dec.append({"op": chosen["operation"], "truth": tr[ci], "best_offered": best,
                            "uniform_mean": statistics.fmean(tr) if tr else 0.0,
                            "quality": (tr[ci] / best) if best > 0 else None,
                            "uniform_quality": (statistics.fmean(tr) / best) if best > 0 else None,
                            "n_act_offered": sum(1 for c in row["candidates"] if c["operation"] == "lab.act"),
                            "p_of_best": p[tr.index(best)] if best > 0 else None, "p_uniform": 1.0 / len(p),
                            "spearman_p_truth": spearman(p, tr), "rule": row["exploration"],
                            "intrinsic": getattr(rec, "intrinsic_reward", None) if rec else None,
                            "bootstrap": getattr(rec, "bootstrap_reward", None) if rec else None,
                            "verified_delta": ((rec.decision.diagnostics.get("reward_signals") or {})
                                               .get("verified_progress_delta")) if rec else None})
            ups = [e["data"] for e in iter_jsonl(os.path.join(lt["runtime_dir"], RunFiles.LEARNING))]
            metas = [e["data"] for e in iter_jsonl(os.path.join(lt["runtime_dir"], RunFiles.META))]
            pol = [u for u in ups if str(u.get("trigger", "")).endswith("newly rewarded decisions")]
            report["lifetimes"].append({
                "index": i, "progress": lt["progress"], "quality": lt["quality"],
                "expected_adopters": lt.get("expected_adopters"), "oracle": lt.get("oracle_expected_adopters"),
                "decisions": lt["decisions"], "verified_status": lt["verified_status"],
                "reward_hash_changed": psi0 != psi1, "decision_rows": dec,
                "policy_updates": [{"weight_delta": u.get("weight_delta"), "losses": u.get("losses"),
                                    "grad_norms": u.get("grad_norms")} for u in pol],
                "frequent_updates": sum(1 for u in ups if u.get("timescale") == "frequent"),
                "meta": [{"applied": m.get("applied"), "skipped": (m.get("skipped_reason") or "")[:80],
                          "grad": m.get("meta_grad_norm")} for m in metas],
                "close_out": lt.get("close_out")})
            q = [d["quality"] for d in dec if d["quality"] is not None]
            u = [d["uniform_quality"] for d in dec if d["uniform_quality"] is not None]
            print(f"[{a.label}] lt {i}: progress {lt['progress']:.2f} quality {lt['quality']:.2f} "
                  f"choice-quality {statistics.fmean(q) if q else float('nan'):.2f} (uniform "
                  f"{statistics.fmean(u) if u else float('nan'):.2f}) waits {sum(d['op'] == 'wait' for d in dec)} "
                  f"psi changed {psi0 != psi1} {time.time() - t0:.0f}s", flush=True)
            with open(a.out, "w") as f:
                json.dump(report, f, indent=1, default=str)
        # reward vs truth over all executed decisions
        rows = [d for lt in report["lifetimes"] for d in lt["decision_rows"] if d["bootstrap"] is not None]
        report["reward_vs_truth"] = {
            "n": len(rows),
            "spearman_bootstrap_truth": spearman([d["bootstrap"] for d in rows], [d["truth"] for d in rows]),
            "spearman_intrinsic_truth": spearman([d["intrinsic"] for d in rows], [d["truth"] for d in rows]),
            "mean_bootstrap_by_op": {op: statistics.fmean(d["bootstrap"] for d in rows if d["op"] == op)
                                     for op in sorted({d["op"] for d in rows})},
            "mean_verified_delta_by_op": {op: statistics.fmean([d["verified_delta"] for d in rows if d["op"] == op
                                                                and d["verified_delta"] is not None] or [0.0])
                                          for op in sorted({d["op"] for d in rows})}}
        report["status"] = agent.status()
    finally:
        agent.close()
        if lock is not None:
            lock.__exit__(None, None, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report["seconds"] = time.time() - t0
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
