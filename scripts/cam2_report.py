"""Compile the CAM-v2 controlled experiments (and the PPO baselines) into ``runs/cam2/report.md``.

    PYTHONPATH=src python scripts/cam2_report.py [--runs /workspace/agentrs/runs]

Only numbers present in run directories are reported. Validation/test tasks are identical across all
runs (same environment code paths and seeds; verified by hashing hidden parameters, goals and first
observations). Success always means verified success at the unchanged full-difficulty goal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(d: Path):
    try:
        return json.loads((d / "state.json").read_text())
    except (OSError, ValueError):
        return None


def ppo_compute(run_dir: Path, st) -> dict:
    """PPO runs record per-iteration collect/update seconds and per-eval wall seconds: cumulative compute per iteration."""
    cum, tot = {}, 0.0
    try:
        for line in (run_dir / "iters.jsonl").read_text().splitlines():
            r = json.loads(line)
            tot += r.get("collect_s", 0) + r.get("update_s", 0)
            cum[r["iter"]] = tot
    except (OSError, ValueError):
        pass
    ev = {}
    acc = 0.0
    for e in st.get("evals", []):
        acc += e.get("wall_s", 0)
        ev[e["iter"]] = acc
    return {i: v + max([a for k, a in ev.items() if k <= i] or [0]) for i, v in cum.items()}


def val_curve(st, fam, greedy: bool = True):
    pc = st.get("_ppo_compute") or {}
    return [(e["iter"], e["env_interactions_so_far"],
             sum((e.get("compute_s_so_far") or {}).values()) or pc.get(e["iter"], 0.0),
             e["by_family"][fam]["rate"], e["by_family"][fam]["final_progress"], e["by_family"][fam].get("violation_rate"))
            for e in st.get("evals", []) if e["split"] == "val" and fam in e["by_family"] and not e.get("planning")
            and e.get("greedy", True) == greedy and not e["tag"].endswith("-monitor")]


def at_budget(curve, budget):
    pts = [c for c in curve if c[1] <= budget]
    return pts[-1] if pts else None


def first_reach(curve, thr):
    for c in curve:
        if c[3] >= thr:
            return c[1]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/workspace/agentrs/runs")
    a = ap.parse_args()
    R = Path(a.runs)
    runs = {"PPO original (cam)": R / "cam", "PPO no-WM (cam_nowm)": R / "cam_nowm", "PPO + product fix (cam_pfix)": R / "cam_pfix"}
    for root in ("cam2", "cam2_v21", "cam2_v22"):
        for d in sorted((R / root).glob("c*")):
            if (d / "state.json").exists():
                runs[d.name] = d
    S = {k: load(v) for k, v in runs.items()}
    S = {k: v for k, v in S.items() if v}
    for k, st in S.items():
        if "grad_updates" not in st.get("counters", {}):
            st["_ppo_compute"] = ppo_compute(runs[k], st)
    md = ["# CAM-v2 experiment report", "",
          "Success = verified success on the unchanged full-difficulty validation goals (200 episodes per family, greedy). "
          "`viol` = episodes ended by a hard-constraint violation (product: bankruptcy; negotiation: counterparty walk-out; "
          "PPO runs do not record it). Interactions = real environment actions taken by the agent.", ""]
    budgets = [100_000, 200_000, 400_000]
    for fam in ("product", "negotiation", "bizdev"):
        rows = []
        for name, st in S.items():
            cv = val_curve(st, fam)
            if not cv:
                continue
            cells = []
            for b in budgets:
                p = at_budget(cv, b)
                cells.append(f"{p[3]:.3f}" + (f" / viol {p[5]:.2f}" if p and p[5] is not None else "") if p else "–")
            best = max(cv, key=lambda c: c[3])
            f10, f30 = first_reach(cv, 0.10), first_reach(cv, 0.30)
            rows.append(f"| {name} | " + " | ".join(cells) + f" | {best[3]:.3f} @ {best[1]:,} | "
                        f"{f'{f10:,}' if f10 else '–'} | {f'{f30:,}' if f30 else '–'} | {cv[-1][1]:,} | {cv[-1][2] / 3600:.2f} |")
        if rows:
            md += [f"## {fam}: validation success vs real interactions", "",
                   "| run | ≤100k | ≤200k | ≤400k | best (at interactions) | interactions to ≥10% | to ≥30% | interactions so far | compute h so far |",
                   "|---|---|---|---|---|---|---|---|---|"] + rows + [""]
    md += ["## Compute-matched view (wall-clock seconds of collect + learn + dream + eval phases on the shared GPU; PPO runs: from their iteration logs)", ""]
    cbud = [0.5, 1.0, 2.0]
    for fam in ("product", "negotiation", "bizdev"):
        rows = []
        for name, st in S.items():
            cv = [c for c in val_curve(st, fam) if c[2] > 0]
            if not cv:
                continue
            cells = []
            for b in cbud:
                pts = [c for c in cv if c[2] <= b * 3600]
                cells.append(f"{pts[-1][3]:.3f} ({pts[-1][1] // 1000}k inter.)" if pts else "–")
            rows.append(f"| {name} | " + " | ".join(cells) + " |")
        if rows:
            md += [f"**{fam}**", "", "| run | ≤0.5 h | ≤1 h | ≤2 h |", "|---|---|---|---|"] + rows + [""]
    md += ["## Sampled (stochastic) policy validation success, CAM-v2 runs (secondary; same tasks)", ""]
    for name, st in S.items():
        for fam in ("product", "negotiation", "bizdev"):
            cv = val_curve(st, fam, greedy=False)
            if cv:
                md.append(f"- {name} {fam}: " + " → ".join(f"{c[1] // 1000}k: {c[3]:.3f}" for c in cv))
    md += ["", "## First verified successes during training (real interactions needed)", ""]
    for name, st in S.items():
        fs = st.get("first_success")
        if fs:
            md.append(f"- {name}: " + ", ".join(f"{k} at {v['env_interactions']:,}" for k, v in sorted(fs.items())))
    md += ["", "## Sealed tests (held-out markets / counterparties; bizdev never trained except in bizdev runs)", ""]
    for name, st in S.items():
        for e in st.get("evals", []):
            if e["split"] == "test":
                md.append(f"- {name} `{e['tag']}`: " + ", ".join(
                    f"{k} {v['success']}/{v['n']} = {v['rate']:.3f}" + (f" (viol {v['violation_rate']:.2f}, time left {v['time_left_at_end']:.2f}, "
                                                                        f"resources {v['resources_at_end']:.2f})" if 'violation_rate' in v else "")
                    for k, v in e["by_family"].items()))
    md += ["", "## World model: open-loop prediction on held-out validation episodes (latest gate)", ""]
    for name, st in S.items():
        ev = [e for e in st.get("evals", []) if e.get("world_model") and "ret_abs_err_k1" in e["world_model"]]
        if ev:
            w = ev[-1]["world_model"]
            md.append(f"- {name} ({ev[-1]['tag']}): return abs err k1 {w['ret_abs_err_k1']:.3f}, k2 {w['ret_abs_err_k2']:.3f}, "
                      f"k4 {w['ret_abs_err_k4']:.3f}, k8 {w['ret_abs_err_k8']:.3f}; violation ECE k1 {w['viol_ece_k1']:.3f}, "
                      f"k4 {w['viol_ece_k4']:.3f}, k8 {w['viol_ece_k8']:.3f}; continuation acc {w['cont_acc_k1']:.3f}; "
                      f"disagreement-vs-error Spearman {w['u_err_spearman']:.2f}; reward err on rare actions {w['rew_abs_err_rare_actions_k1']:.3f}"
                      + (f"; dream gate {ev[-1].get('dream_gate')}" if ev[-1].get("dream_gate") else ""))
    md += ["", "## Counterfactual action ranking vs. actual simulator outcomes (final checkpoint; 4-step returns)", ""]
    for name, st in S.items():
        cf = st.get("counterfactual")
        if cf:
            for f, v in cf.items():
                md.append(f"- {name} {f}: Spearman {v['spearman']:.2f}, top-1 {v['top1_agreement']:.2f} (chance {v['chance_top1']:.2f}), "
                          f"pairwise {v['pairwise_accuracy']:.2f}, regret of model's choice {v['regret_of_model_choice']:.3f}, "
                          f"simulator-oracle best-minus-mean {v['oracle_best_minus_mean (simulator-oracle reference)']:.3f} ({v['states']} states)")
        wv = st.get("wm_validation_final")
        if wv:
            md.append(f"- {name} corrupted-model control: reliability gate clean {wv['gate_clean']['reliability']:.2f} "
                      f"(horizon {wv['gate_clean']['max_h']}) vs corrupted {wv['gate_corrupted']['reliability']:.2f}; "
                      f"corrupted return err k1 {wv['corrupted'].get('ret_abs_err_k1', float('nan')):.3f} vs clean "
                      f"{wv['clean'].get('ret_abs_err_k1', float('nan')):.3f}")
    md += ["", "## Planning diagnosis on validation tasks: CAM actor vs world-model planner vs SIMULATOR-ORACLE planner", ""]
    for name, st in S.items():
        pdg = st.get("planning_diagnosis")
        if pdg:
            for f, d in pdg.items():
                md.append(f"- {name} {f}: success actor {d['actor']['success']:.2f} (bankrupt/walk-out {d['actor']['violation']:.2f}), "
                          f"WM planner {d['model']['success']:.2f} ({d['model']['violation']:.2f}), "
                          f"simulator-oracle planner [oracle information] {d['oracle']['success']:.2f} ({d['oracle']['violation']:.2f}); "
                          f"n={d['actor']['n']}, lookahead {d['actor']['k']}")
    md += ["", "## Learning efficiency, replay and imagination", ""]
    for name, st in S.items():
        c = st.get("counters", {})
        if "grad_updates" not in c:
            continue
        last = (st.get("iters") or [{}])[-1]
        rp = last.get("replay", {})
        cs = st.get("compute_s", {})
        md.append(f"- {name}: {c['transitions']:,} real interactions, {c['grad_updates']:,} gradient updates, "
                  f"{c.get('dream_updates', 0):,} dream updates / {c.get('imagined_transitions', 0):,} imagined transitions, "
                  f"replay {rp.get('steps', 0):,} steps (protected {rp.get('protected_episodes', 0)} episodes, "
                  f"{rp.get('successes', 0)} successes), replay ratio {rp.get('replay_ratio', 0):.1f}; compute h: "
                  + ", ".join(f"{k} {v / 3600:.2f}" for k, v in cs.items())
                  + (f"; dream fallbacks {len(st.get('dream_fallbacks', []))}" if st.get("dream_fallbacks") is not None else "")
                  + f"; phase {st.get('phase')}")
    md += ["", "## Policy stability", ""]
    for name, st in S.items():
        parts = []
        for fam in ("product", "negotiation", "bizdev"):
            cv = val_curve(st, fam)
            if len(cv) >= 3:
                d = np.abs(np.diff([c[3] for c in cv]))
                parts.append(f"{fam} mean |Δ val success| between gates {d.mean():.3f} (max {d.max():.3f})")
        lc = [r.get("learn", {}).get("logp_change_vs_behaviour") for r in st.get("iters", [])]
        lc = [x for x in lc if x is not None]
        if lc:
            parts.append(f"mean log-prob change of replayed decisions vs behaviour policy {np.mean(lc):+.3f}")
        rb = st.get("rollbacks") or st.get("dream_fallbacks")
        if rb:
            parts.append(f"rollbacks/fallbacks {len(rb)}")
        if parts:
            md.append(f"- {name}: " + "; ".join(parts))
    md += ["", "## Reward-contract audits", ""]
    for name, st in S.items():
        au = st.get("reward_audit")
        if au:
            md.append(f"- {name}: {'PASSED' if au['passed'] else 'FAILED'} (lam probe {au['lam_probe']})")
    out = R / "cam2" / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
