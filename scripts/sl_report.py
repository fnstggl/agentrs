"""Compile the superlearner experiments into a report of the pre-registered measures (docs/superlearner.md).

    PYTHONPATH=src python scripts/sl_report.py --root /workspace/agentrs/runs/sl [--metric-for ext/vending=score]

Only numbers present in run directories are reported. Confidence intervals: across-seed variation combined with
the evaluation-task sampling error (normal approximation over the 48 validation tasks); stated as 90% intervals.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


def load(d: Path):
    try:
        return json.loads((d / "state.json").read_text())
    except (OSError, ValueError):
        return None


def metric_of(fam: str, overrides: dict) -> str:
    if fam in overrides:
        return overrides[fam]
    # cam/bizdev: every learner so far (original PPO CAM, CAM-v2 PPO arms, superlearner) has 0% success within these
    # budgets, so its primary metric is verified progress (fraction of required partners committed); amendment 3
    return "score" if fam.startswith(("ext/", "mp/", "mpsc/")) or fam == "cam/bizdev" else "success"


def ev_val(st, fam, mode="full", split="val"):
    """[(interactions, value-dict)] of milestone evaluations (tags m<N>) for a family."""
    out = []
    for e in st.get("evals", []):
        if e["split"] == split and e["mode"] == mode and re.fullmatch(r"m\d+", e["tag"]) and fam in e["families"]:
            out.append((int(e["tag"][1:]), e["families"][fam]))
    return sorted(out, key=lambda x: x[0])


def auc_value(v, metric):
    return v["auc_success"] if metric == "success" else v["auc_score"]


def learning_auc(curve, metric, budget):
    """Normalised area under the validation learning curve over [0, budget] interactions (trapezoid)."""
    pts = [(x, auc_value(v, metric)) for x, v in curve if x <= budget]
    if len(pts) < 2:
        return None
    xs, ys = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
    return float(np.trapezoid(ys, xs) / max(1.0, xs[-1] - xs[0]))


def ci90(vals, extra_se=0.0):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = float(np.mean(vals))
    se = math.sqrt((np.var(vals, ddof=1) / len(vals) if len(vals) > 1 else 0.0) + extra_se ** 2)
    return m, m - 1.645 * se, m + 1.645 * se, len(vals)


def fmt(c, nd=3):
    return "–" if c is None else f"{c[0]:+.{nd}f} [{c[1]:+.{nd}f}, {c[2]:+.{nd}f}] (n={c[3]})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/agentrs/runs/sl")
    ap.add_argument("--metric-for", nargs="*", default=[])
    a = ap.parse_args()
    root = Path(a.root)
    ov = dict(x.split("=", 1) for x in a.metric_for)
    runs = {d.name: s for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith("_") and (s := load(d))}
    md = [f"# Superlearner report: `{root}`", "", f"{len(runs)} runs. Phases: " +
          ", ".join(f"{k} {v}" for k, v in sorted(_count(runs).items())), ""]

    # --------------------------------------------------------------- 1. fast adaptation with frozen weights
    md += ["## 1. Fast adaptation inside a task with frozen weights (validation, latest milestone of each lifetime)", "",
           "Each task is one trial of K episodes with the same hidden parameters. The table gives the episode-1 → "
           "episode-K value of the success rate (score for external worlds) per adaptation mode. Gain = last − first "
           "episode. DiD = gain(full) − gain(episode): improvement attributable to memory carried across episodes.", "",
           "| run | family | metric | full | episode (no cross-episode memory) | none (no adaptation) | DiD |", "|---|---|---|---|---|---|---|"]
    for name, st in runs.items():
        if not name.startswith(("P_", "A_")):
            continue
        tags = sorted({int(e["tag"][1:]) for e in st.get("evals", []) if re.fullmatch(r"m\d+", e["tag"])})
        if not tags:
            continue
        last = f"m{tags[-1]}"
        fams = sorted({f for e in st["evals"] if e["tag"] == last for f in e["families"]})
        for fam in fams:
            met = metric_of(fam, ov)
            cells, gains = {}, {}
            for mode in ("full", "episode", "none"):
                e = next((e for e in st["evals"] if e["tag"] == last and e["mode"] == mode and fam in e["families"]), None)
                if e is None:
                    continue
                cv = e["families"][fam]["curve"]
                if cv:
                    f0, f1 = cv[0][met], cv[-1][met]
                    cells[mode] = f"{f0:.3f} → {f1:.3f}"
                    gains[mode] = (f1 - f0, cv[0]["n"])
            if not cells:
                continue
            did = "–"
            if "full" in gains and "episode" in gains:
                n = gains["full"][1]
                se = math.sqrt(2 * 0.25 / max(1, n)) * (1 if met == "success" else 0) if met == "success" else 0.0
                d = gains["full"][0] - gains["episode"][0]
                did = f"{d:+.3f}" + (f" (±{1.645 * math.sqrt(2) * se:.3f})" if se else "")
            held = " (held out)" if fam in st["config"].get("heldout_families", []) else ""
            md.append(f"| {name} @ {last} | {fam}{held} | {met} | {cells.get('full', '–')} | {cells.get('episode', '–')} | "
                      f"{cells.get('none', '–')} | {did} |")
    md.append("")

    # --------------------------------------------------------------- 2. transfer / learning speed on held-out targets
    md += ["## 2. Learning a held-out world: experienced inits vs scratch (validation learning curves, mode full)", ""]
    groups: dict = {}
    for name, st in runs.items():
        m = re.fullmatch(r"T_(\w+?)_(scratch|from\w+?)_s(\d+)", name)
        if m:
            groups.setdefault(m.group(1), {}).setdefault(m.group(2), []).append(name)
    for tgt, inits in sorted(groups.items()):
        any_run = runs[next(iter(inits.values()))[0]]
        fam = list(any_run["config"]["families"])[0]
        met = metric_of(fam, ov)
        budget = any_run["config"]["max_env_steps"]
        md += [f"### target {fam} (metric: {met}; budget {budget:,} interactions)", ""]
        ms = sorted({x for n_ in sum(inits.values(), []) for x, _ in ev_val(runs[n_], fam)})
        md += ["| init | seeds | " + " | ".join(f"{x // 1000}k" for x in ms) + " | learning AUC | ΔAUC vs scratch (90% CI) | final test |",
               "|---|---|" + "---|" * len(ms) + "---|---|---|"]
        auc = {}
        for init, names in sorted(inits.items(), key=lambda kv: (kv[0] != "scratch", kv[0])):
            vals = {x: [] for x in ms}
            aucs, tests = [], []
            for n_ in names:
                cv = ev_val(runs[n_], fam)
                for x, v in cv:
                    vals[x].append(auc_value(v, met))
                aucs.append(learning_auc(cv, met, budget))
                te = next((e for e in runs[n_].get("evals", []) if e["tag"] == "final" and e["split"] == "test" and fam in e["families"]), None)
                if te:
                    tests.append(auc_value(te["families"][fam], met))
            auc[init] = [x for x in aucs if x is not None]
            cells = [f"{np.mean(vals[x]):.3f}" if vals[x] else "–" for x in ms]
            d = "–"
            if init != "scratch" and auc.get("scratch") and auc[init]:
                k = min(len(auc[init]), len(auc["scratch"]))
                diffs = [auc[init][i] - auc["scratch"][i] for i in range(k)]
                d = fmt(ci90(diffs, extra_se=0.01 if met == "success" else 0.0))
            md.append(f"| {init} | {len(names)} | " + " | ".join(cells) + f" | {np.mean(auc[init]):.3f} | {d} | "
                      + (f"{np.mean(tests):.3f} (n={len(tests)})" if tests else "–") + " |" if auc[init] else
                      f"| {init} | {len(names)} | " + " | ".join(cells) + " | – | – | – |")
        md.append("")
        # interactions to threshold
        thr = {"success": 0.10}.get(met) or {"cam/bizdev": 0.5}.get(fam, 1.5)
        md.append(f"Interactions until validation {met} ≥ {thr} (first milestone; per seed): " + "; ".join(
            f"{init}: " + ", ".join(str(next((x for x, v in ev_val(runs[n_], fam) if auc_value(v, met) >= thr), "not reached")) for n_ in names)
            for init, names in sorted(inits.items())))
        md.append("")
        # retention of source families during transfer
        rows = []
        for init, names in sorted(inits.items()):
            for n_ in names:
                st = runs[n_]
                for mf in st["config"].get("heldout_families", []):
                    cv = ev_val(st, mf)
                    if len(cv) >= 2:
                        mm = metric_of(mf, ov)
                        rows.append(f"- {n_}: {mf} {auc_value(cv[0][1], mm):.3f} → {auc_value(cv[-1][1], mm):.3f} ({mm}, {cv[0][0] // 1000}k → {cv[-1][0] // 1000}k)")
        if rows:
            md += ["Retention of source families while learning the target (validation, before → after):", ""] + rows + [""]

    # --------------------------------------------------------------- 3. lifetimes: learning on training families
    md += ["## 3. Lifetimes: validation on training and held-out families across lifetime experience (mode full)", ""]
    for name, st in runs.items():
        if not name.startswith(("P_", "A_")):
            continue
        fams = list(st["config"]["families"]) + list(st["config"].get("heldout_families", []))
        parts = []
        for fam in fams:
            met = metric_of(fam, ov)
            cv = ev_val(st, fam)
            if cv:
                parts.append(f"{fam}{' (held out)' if fam not in st['config']['families'] else ''}: "
                             + " → ".join(f"{x // 1000}k {auc_value(v, met):.3f}" for x, v in cv))
        md.append(f"- **{name}** ({st['phase']}, {st['interactions']:,} interactions, mode {st['config'].get('mode')}, "
                  f"skills {st['config'].get('model', {}).get('skills', 8)}, wm {st['config'].get('wm_coef')}): " + "; ".join(parts))
    md.append("")

    # --------------------------------------------------------------- 4. diagnostics, audits, compute
    md += ["## 4. Memory ablation (final, frozen weights), world model, reward audits, compute", ""]
    for name, st in runs.items():
        d = st.get("diagnostics") or {}
        if "memory_ablation" in d:
            md.append(f"- {name} memory ablation (validation trial AUC by mode): " + "; ".join(
                f"{md_}: " + ", ".join(f"{f} {v:.3f}" for f, v in fams.items()) for md_, fams in d["memory_ablation"].items()))
        if "world_model" in d:
            md.append(f"- {name} world model (one-step loss on fresh validation transitions): " + ", ".join(
                f"{f} {v['wm_loss']:.3f}" for f, v in d["world_model"].items()))
        if st.get("diagnostic_errors"):
            md.append(f"- {name} diagnostic errors: {sorted(st['diagnostic_errors'])}")
    md.append("")
    md += ["| run | phase | interactions | collect h | learn h | eval h | GPU peak GB | audit |", "|---|---|---|---|---|---|---|---|"]
    for name, st in runs.items():
        cs = st.get("compute_s", {})
        peak = None
        try:
            peak = max(json.loads(l).get("gpu_peak_gb") or 0 for l in (root / name / "iters.jsonl").read_text().splitlines())
        except (OSError, ValueError):
            pass
        au = st.get("reward_audit") or {}
        md.append(f"| {name} | {st['phase']} | {st['interactions']:,} | {cs.get('collect', 0) / 3600:.2f} | {cs.get('learn', 0) / 3600:.2f} | "
                  f"{cs.get('eval', 0) / 3600:.2f} | {peak if peak is not None else '–'} | "
                  f"{'pass' if au and all(v['passed'] for v in au.values()) else ('FAIL' if au else '–')} |")
    out = root / "report.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def _count(runs):
    c = {}
    for st in runs.values():
        c[st["phase"]] = c.get(st["phase"], 0) + 1
    return c


if __name__ == "__main__":
    raise SystemExit(main())
