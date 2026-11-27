"""Does the integrated agent's choice track its own predictions, and does what it learns reach the choices?
Engineering diagnostic (debug scale unless --real), workstream LRN (B/C), change request S-5.

One CamLiveAgent built exactly like evaluation condition ``--condition`` (``cam_live.eval.conditions``: base_config +
eval_config, schema-walker candidates) runs ``--lifetimes`` LabWorld F1 objectives through the real runtime
(``cam_live.eval.runner.run_lifetime``: ground-truth tap on every candidate, close-out to the deadline). Recorded per
decision, against the ground truth of every offered candidate (expected adopters; 0 for probe / gauge / wait):

* choice quality: truth(chosen) / max truth(C_t); expected choice quality sum_i p_i truth_i / max truth under the
  recorded selection probabilities (less noisy); the same for a uniform choice, for greedy-on-the-agent's-own
  predicted goal value (``extras["q_goal"]``) and for the policy head alone (``extras["policy_prob"]``);
* prediction quality: Spearman(q_goal, truth) and Spearman(policy_prob, truth) over C_t;
* the selection rule's diagnostics (policy share, calibration of the learned action value);
* the reward the agent stored at reward time and the reward the inner policy update uses at the end of the
  lifetime (bootstrap, learned, goal term, mixed total), by whether the chosen action had any value;
* policy updates (step chosen by the trust region, KL), meta-updates, the bootstrap mix and promotion decisions.

    .venv/bin/python cam_live/docs/evidence/B/choice_diagnostic.py --out OUT.json [--condition E] [--seed 0]
        [--lifetimes 8] [--decisions 16] [--agent '{"selection_rule": "thompson_policy"}'] [--label NAME] [--real]
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


def fmean(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return statistics.fmean(v) if v else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", default="E")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--lifetimes", type=int, default=8)
    ap.add_argument("--decisions", type=int, default=16)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent", default="{}", help="JSON overrides of the agent section")
    ap.add_argument("--meta", default="{}", help="JSON overrides of reward.meta")
    ap.add_argument("--label", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--budget-s", type=float, default=0.0)
    a = ap.parse_args(argv)
    import torch
    from cam_live.agent import CamLiveAgent
    from cam_live.contracts import RunFiles, SimClock
    from cam_live.eval.conditions import base_config, eval_config
    from cam_live.eval.runner import run_lifetime
    from cam_live.runtime.logs import iter_jsonl
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    cfg = eval_config(base_config(real=a.real, seed=a.seed), a.condition, seed=a.seed)
    cfg.sections["agent"].update(json.loads(a.agent))
    rw = dict(cfg.sections.get("reward", {}))
    rw["meta"] = {**rw.get("meta", {}), **json.loads(a.meta)}
    cfg.sections["reward"] = rw
    work = os.path.abspath(a.work or os.path.splitext(a.out)[0] + "_work")
    os.makedirs(work, exist_ok=True)
    lock = None
    if a.real:
        from cam_live.nn.gpu import gpu_lock
        lock = gpu_lock(min_free_gb=18.0)
        lock.__enter__()
    clock = SimClock()
    agent = CamLiveAgent(cfg, os.path.join(work, "agent"), clock=clock)
    report = {"label": a.label, "condition": a.condition, "seed": a.seed, "real_model": a.real,
              "agent_settings": cfg.section("agent"), "reward_meta": cfg.section("reward").get("meta", {}),
              "selection_rule": agent.settings.selection_rule, "lifetimes": [], "started": time.time()}
    t0 = time.time()
    try:
        for i in range(a.lifetimes):
            if a.budget_s and time.time() - t0 > a.budget_s:
                report["stopped"] = f"budget {a.budget_s:.0f} s reached before lifetime {i}"
                break
            n_mix_dec = len(agent.meta.mixer.decisions)
            lt = run_lifetime(agent, "F1", 10_000 + 100 * a.seed + i, os.path.join(work, f"lt{i:02d}"), clock,
                              max_decisions=a.decisions, k=a.k)
            dec = []
            for row in lt["decision_rows"]:
                tr = [c["truth"] for c in row["candidates"]]
                p = row["probabilities"]
                ci = row["chosen_index"]
                best = max(tr) if tr else 0.0
                rec = agent.store.by_decision(row["decision_id"])
                ex = [dict(c.extras) for c in rec.decision.candidates] if rec else [{} for _ in tr]
                q = [e.get("q_goal") for e in ex]
                pol = [e.get("policy_prob") for e in ex]
                d = {"op": row["candidates"][ci]["operation"], "truth": tr[ci], "best": best,
                     "zero_value": tr[ci] <= 0.0, "rule": row["exploration"], "experiment": bool(row["experiment_id"])}
                if best > 0:
                    d["quality"] = tr[ci] / best
                    d["expected_quality"] = sum(pi * t for pi, t in zip(p, tr)) / best
                    d["uniform_quality"] = statistics.fmean(tr) / best
                    if all(x is not None for x in q):
                        d["greedy_pred_quality"] = tr[max(range(len(q)), key=lambda j: q[j])] / best
                    if all(x is not None for x in pol):
                        s = sum(pol)
                        d["policy_quality"] = sum(pi * t for pi, t in zip(pol, tr)) / best / s if s > 0 else None
                if all(x is not None for x in q):
                    d["spearman_pred_truth"] = spearman(q, tr)
                if all(x is not None for x in pol):
                    d["spearman_policy_truth"] = spearman(pol, tr)
                if rec is not None:
                    sel = rec.decision.diagnostics.get("selection") or {}
                    d["policy_share_tv"] = sel.get("policy_share_tv")
                    d["calibration"] = sel.get("calibration")
                    d["stored_intrinsic"] = rec.intrinsic_reward
                    d["stored_bootstrap"] = rec.bootstrap_reward
                    d["verified_delta"] = ((rec.decision.diagnostics.get("reward_signals") or {})
                                           .get("verified_progress_delta"))
                dec.append(d)
            # the reward the inner policy update uses now (rebuilt from the store with every annotation that arrived)
            inner = {}
            try:
                recs = [r for r in agent.store.trajectory(lt["objective_id"]) if r.result is not None]
                ib = agent.meta.inner.build_batch(agent.store, recs)
                psi = {k: v.detach() for k, v in agent.meta.reward.psi().items()}
                with torch.no_grad():
                    r_tot, r_l = agent.meta.inner.intrinsic(psi, ib, detach_reward=True)
                    g = agent.meta.inner.goal_term(ib, r_l.dtype) if hasattr(agent.meta.inner, "goal_term") else None
                pos = {e: k for k, e in enumerate(ib.exp_ids)}
                by_dec = {r.decision.decision_id: pos[r.exp_id] for r in recs if r.exp_id in pos}
                for row, d in zip(lt["decision_rows"], dec):
                    k = by_dec.get(row["decision_id"])
                    if k is None:
                        continue
                    d["inner_total"] = float(r_tot[k])
                    d["inner_bootstrap"] = float(ib.r_boot[k])
                    d["inner_learned"] = float(r_l[k])
                    d["inner_goal"] = None if g is None else float(g[k])
                    d["inner_goal_source"] = ib.goal_source[k] if getattr(ib, "goal_source", None) else ""
                inner = {"mix": ib.mix, "n": ib.n}
            except Exception as ex:  # noqa: BLE001
                report.setdefault("errors", []).append(f"inner batch: {type(ex).__name__}: {ex}")
            ups = [e["data"] for e in iter_jsonl(os.path.join(lt["runtime_dir"], RunFiles.LEARNING))]
            metas = [e["data"] for e in iter_jsonl(os.path.join(lt["runtime_dir"], RunFiles.META))]
            pol_ups = [u for u in ups if str(u.get("trigger", "")).endswith("newly rewarded decisions")]
            row = {"index": i, "progress": lt["progress"], "quality": lt["quality"],
                   "expected_adopters": lt.get("expected_adopters"), "oracle": lt.get("oracle_expected_adopters"),
                   "decisions": lt["decisions"], "verified_status": lt["verified_status"],
                   "mean": {k: fmean([d.get(k) for d in dec]) for k in (
                       "quality", "expected_quality", "uniform_quality", "greedy_pred_quality", "policy_quality",
                       "spearman_pred_truth", "spearman_policy_truth", "policy_share_tv")},
                   "zero_value_share": fmean([1.0 if d["zero_value"] else 0.0 for d in dec]),
                   "decision_rows": dec, "inner": inner,
                   "policy_updates": [{"weight_delta": u.get("weight_delta"),
                                       "kl": (u.get("prediction_change") or {}).get("policy_kl"),
                                       "notes": u.get("notes"), "diagnostics": u.get("diagnostics")} for u in pol_ups],
                   "meta": [{"applied": m.get("applied"), "skipped": (m.get("skipped_reason") or "")[:80],
                             "grad": m.get("meta_grad_norm"), "mix": m.get("bootstrap_mix"),
                             "promotion": (m.get("diagnostics") or {}).get("promotion")} for m in metas],
                   "mix_end": agent.meta.mixer.mix,
                   "mix_decisions": [{"action": x.action, "reason": x.reason, "mix_before": x.mix_before,
                                      "mix_after": x.mix_after, "n": x.n_independent, "mean_diff": x.mean_diff,
                                      "lower": x.lower_bound, "upper": x.upper_bound,
                                      "margin": getattr(x, "margin", None)}
                                     for x in agent.meta.mixer.decisions[n_mix_dec:]],
                   "calibration": agent.decider.calibration.summary() if hasattr(agent.decider, "calibration")
                   else None}
            report["lifetimes"].append(row)
            m = row["mean"]
            print(f"[{a.label}] lt {i}: progress {lt['progress']:.2f} E[q] {m['expected_quality'] or float('nan'):.2f} "
                  f"(uniform {m['uniform_quality'] or float('nan'):.2f}, greedy-pred "
                  f"{m['greedy_pred_quality'] or float('nan'):.2f}, policy {m['policy_quality'] or float('nan'):.2f}) "
                  f"rho(pred) {m['spearman_pred_truth'] or float('nan'):.2f} zero-share {row['zero_value_share']:.2f} "
                  f"mix {row['mix_end']:.2f} {time.time() - t0:.0f}s", flush=True)
            with open(a.out, "w") as f:
                json.dump(report, f, indent=1, default=str)
        rows = [d for lt in report["lifetimes"] for d in lt["decision_rows"]]

        def pos_share(key):
            pos = [(d[key], d["zero_value"]) for d in rows if d.get(key) is not None and d[key] > 0]
            tot = sum(v for v, _ in pos)
            return None if tot <= 0 else sum(v for v, z in pos if z) / tot
        lts = report["lifetimes"]
        report["summary"] = {
            "n_decisions": len(rows),
            "expected_quality": fmean([d.get("expected_quality") for d in rows]),
            "quality": fmean([d.get("quality") for d in rows]),
            "uniform_quality": fmean([d.get("uniform_quality") for d in rows]),
            "greedy_pred_quality": fmean([d.get("greedy_pred_quality") for d in rows]),
            "policy_quality": fmean([d.get("policy_quality") for d in rows]),
            "spearman_pred_truth": fmean([d.get("spearman_pred_truth") for d in rows]),
            "zero_value_share": fmean([1.0 if d["zero_value"] else 0.0 for d in rows]),
            "share_positive_stored_reward_to_zero_value": pos_share("stored_intrinsic"),
            "share_positive_inner_reward_to_zero_value": pos_share("inner_total"),
            "spearman_inner_reward_truth": spearman([d["inner_total"] for d in rows if d.get("inner_total") is not None],
                                                    [d["truth"] for d in rows if d.get("inner_total") is not None]),
            "per_lifetime_spearman_pred_vs_expected_quality": spearman(
                [lt["mean"]["spearman_pred_truth"] or 0.0 for lt in lts],
                [lt["mean"]["expected_quality"] or 0.0 for lt in lts]),
            "expected_adopters_per_decision": (sum(lt["expected_adopters"] or 0.0 for lt in lts)
                                               / max(1, sum(lt["decisions"] for lt in lts))),
            "progress": fmean([lt["progress"] for lt in lts]),
            "mix_trajectory": [lt["mix_end"] for lt in lts],
            "mix_decisions": [x for lt in lts for x in lt["mix_decisions"]],
            "meta_applied": sum(1 for lt in lts for m in lt["meta"] if m["applied"]),
            "policy_updates": sum(len(lt["policy_updates"]) for lt in lts)}
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
