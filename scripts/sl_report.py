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
    global ROOT
    ROOT = root
    ov = dict(x.split("=", 1) for x in a.metric_for)
    runs = {d.name: s for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith("_") and (s := load(d))}
    md = [f"# Superlearner report: `{root}`", "", f"{len(runs)} runs. Phases: " +
          ", ".join(f"{k} {v}" for k, v in sorted(_count(runs).items())), ""]

    md += prereg_tests(runs)
    md += inaction_check(runs)
    md += paired_adaptation(root)

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
        md += [f"### T_{tgt}_*: target {fam} (metric: {met}; budget {budget:,} interactions)", ""]
        ms = sorted({x for n_ in sum(inits.values(), []) for x, _ in ev_val(runs[n_], fam)})
        md += ["| init | seeds | " + " | ".join(f"{x // 1000}k" for x in ms) + " | learning AUC | ΔAUC vs scratch (90% CI) | final test |",
               "|---|---|" + "---|" * len(ms) + "---|---|---|"]
        auc = {}
        # Learning AUC only for runs whose curve reaches the budget: a curve normalised over a shorter range is not
        # comparable (curves rise, so a run still in progress looks worse). Differences pair runs by seed.
        for init, names in sorted(inits.items(), key=lambda kv: (kv[0] != "scratch", kv[0])):
            vals = {x: [] for x in ms}
            aucs, tests = {}, []
            for n_ in names:
                cv = ev_val(runs[n_], fam)
                for x, v in cv:
                    vals[x].append(auc_value(v, met))
                if cv and max(x for x, _ in cv) >= budget:
                    aucs[n_.rsplit("_s", 1)[1]] = learning_auc(cv, met, budget)
                te = next((e for e in runs[n_].get("evals", []) if e["tag"] == "final" and e["split"] == "test" and fam in e["families"]), None)
                if te:
                    tests.append(auc_value(te["families"][fam], met))
            auc[init] = {s: x for s, x in aucs.items() if x is not None}
            cells = [f"{np.mean(vals[x]):.3f}" if vals[x] else "–" for x in ms]
            d = "–"
            if init != "scratch" and auc.get("scratch") and auc[init]:
                both = sorted(set(auc[init]) & set(auc["scratch"]))
                if both:
                    d = fmt(ci90([auc[init][s] - auc["scratch"][s] for s in both], extra_se=0.01 if met == "success" else 0.0))
            seeds = f"{len(auc[init])}/{len(names)}"
            md.append(f"| {init} | {seeds} | " + " | ".join(cells) + f" | {np.mean(list(auc[init].values())):.3f} | {d} | "
                      + (f"{np.mean(tests):.3f} (n={len(tests)})" if tests else "–") + " |" if auc[init] else
                      f"| {init} | {seeds} | " + " | ".join(cells) + " | – | – | – |")
        md += ["", "Seeds = runs that reached the budget / runs; milestone columns average every run that reached the "
               "milestone; ΔAUC pairs complete runs by seed.", ""]
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
                rehearsed = [f for f in st["config"]["families"] if f != fam]  # source worlds trained alongside (amendment 12)
                for mf in rehearsed + list(st["config"].get("heldout_families", [])):
                    cv = ev_val(st, mf)
                    if len(cv) >= 2:
                        mm = metric_of(mf, ov)
                        rows.append(f"- {n_}: {mf}{' (rehearsed)' if mf in rehearsed else ''} {auc_value(cv[0][1], mm):.3f} → {auc_value(cv[-1][1], mm):.3f} ({mm}, {cv[0][0] // 1000}k → {cv[-1][0] // 1000}k)")
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
            # The training-side diagnostic stores success AUC. Families scored by `score` (Melting Pot, Vending, bizdev)
            # have no success signal there, so their zeros say nothing about memory.
            uninf = {f for fams in d["memory_ablation"].values() for f in fams if metric_of(f, {}) != "success"}
            info = "; ".join(f"{md_}: " + ", ".join(f"{f} {v:.3f}" for f, v in fams.items() if f not in uninf)
                             for md_, fams in d["memory_ablation"].items() if any(f not in uninf for f in fams))
            note = f"not informative for {len(uninf)} score-metric families (success AUC only)" if uninf else ""
            md.append(f"- {name} memory ablation (validation trial success AUC by mode): "
                      + (info + (f" ({note})" if note else "") if info else note))
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


def _val_at(st, fam, tag, key):
    e = next((e for e in st.get("evals", []) if e["tag"] == tag and e["split"] == "val" and e["mode"] == "full"
              and fam in e["families"]), None)
    return None if e is None else e["families"][fam][key]


def prereg_tests(runs) -> list:
    """Pre-registered hypothesis tests of the protocol amendments (docs/superlearner.md), evaluated mechanically."""
    md = []
    old = {n: s for n, s in runs.items() if re.fullmatch(r"T_B_scratch_s\d", n)}
    new = {n: s for n, s in runs.items() if re.fullmatch(r"T_B11_scratch_s\d", n)}
    if new or old:
        md += ["## 0. Pre-registered tests", "",
               "**H6 (amendment 6): the shaping-discount defect causes the bizdev collapse.** Validation progress "
               "(cam/bizdev, mode full). Confirmed if every v11 seed is ≥ 0.20 at 100k and at 300k; disproved if at "
               "least 2 of 3 v11 seeds are < 0.10 at 300k.", "",
               "| run | code | 0k | 100k | 300k |", "|---|---|---|---|---|"]
        for n, s in sorted(old.items()) + sorted(new.items()):
            v = [_val_at(s, "cam/bizdev", tg, "auc_score") for tg in ("m0", "m100000", "m300000")]
            md.append(f"| {n} | {'v11' if n in new else 'v10'} | " + " | ".join("–" if x is None else f"{x:.3f}" for x in v) + " |")
        at = {n: (_val_at(s, "cam/bizdev", "m100000", "auc_score"), _val_at(s, "cam/bizdev", "m300000", "auc_score"))
              for n, s in new.items()}
        low300 = sum(1 for a, b in at.values() if b is not None and b < 0.10)
        complete = len(at) == 3 and all(a is not None and b is not None for a, b in at.values())
        if low300 >= 2:
            verdict = "DISPROVED"
        elif complete and all(a >= 0.20 and b >= 0.20 for a, b in at.values()):
            verdict = "CONFIRMED"
        elif complete:
            verdict = "neither confirmed nor disproved (mixed)"
        else:
            verdict = f"pending ({sum(b is not None for a, b in at.values())}/3 v11 seeds at 300k)"
        md += ["", f"H6 verdict: **{verdict}**", ""]
    dp = {n: runs[n] for n in ("D_P10_s1", "D_P11_s1") if n in runs}
    if dp:
        md += ["**Product (amendment 6, secondary):** product-only learning, identical configuration, v10 vs v11 "
               "contract. Prediction: v11 validation progress at 300k exceeds v10 by ≥ 0.10.", "",
               "| run | 0k progress | 100k | 300k | 300k success |", "|---|---|---|---|---|"]
        for n, s in dp.items():
            v = [_val_at(s, "cam/product", tg, "auc_score") for tg in ("m0", "m100000", "m300000")]
            sc = _val_at(s, "cam/product", "m300000", "auc_success")
            md.append(f"| {n} | " + " | ".join("–" if x is None else f"{x:.3f}" for x in v) + f" | {'–' if sc is None else f'{sc:.3f}'} |")
        a, b = (_val_at(dp[n], "cam/product", "m300000", "auc_score") if n in dp else None for n in ("D_P11_s1", "D_P10_s1"))
        md += ["", "Product verdict: **" + ("pending" if a is None or b is None else
                                             ("CONFIRMED" if a - b >= 0.10 else "NOT CONFIRMED") + f" (Δ = {a - b:+.3f})") + "**", ""]
    dc = {n: s for n, s in runs.items() if re.fullmatch(r"D_P12c_s\d", n)}
    if dc:
        md += ["**H7 (amendment 7): product fails because training almost never succeeds; a training-only goal curriculum "
               "fixes it.** Validation at the full, unchanged targets (cam/product, mode full). Confirmed if every "
               "curriculum seed has progress ≥ 0.20 and success > 0 at 300k; disproved if every curriculum seed has "
               "progress < 0.05 at 300k, or if no seed's training difficulty ever rises (success never reached at the "
               "easiest goals).", "",
               "| run | curriculum | 0k progress | 100k | 300k | 300k success | training difficulty now (max, rises) |",
               "|---|---|---|---|---|---|---|"]
        res = {}
        for n, s in [(k, runs[k]) for k in ("D_P11_s1",) if k in runs] + sorted(dc.items()):
            v = [_val_at(s, "cam/product", tg, "auc_score") for tg in ("m0", "m100000", "m300000")]
            sc = _val_at(s, "cam/product", "m300000", "auc_success")
            cur = (s.get("curriculum") or {}).get("cam/product")
            dtxt = "–" if not cur else f"{cur['d']:.3f} ({cur.get('d_max', cur['d']):.3f}, {cur.get('ups', 0)})"
            md.append(f"| {n} | {'yes' if n in dc else 'no'} | " + " | ".join("–" if x is None else f"{x:.3f}" for x in v)
                      + f" | {'–' if sc is None else f'{sc:.3f}'} | {dtxt} |")
            if n in dc:
                res[n] = (v[2], sc, (cur or {}).get("ups", 0))
        done = [r for r in res.values() if r[0] is not None]
        if len(done) < len(res) or not res:
            verdict = f"pending ({len(done)}/{len(res)} curriculum seeds at 300k)"
        elif all(p < 0.05 for p, _, _ in done) or all(u == 0 for _, _, u in done):
            verdict = "DISPROVED"
        elif all(p >= 0.20 and (sc or 0) > 0 for p, sc, _ in done):
            verdict = "CONFIRMED"
        else:
            verdict = "neither confirmed nor disproved (mixed)"
        md += ["", f"H7 verdict: **{verdict}**", ""]
    for hyp, ver, amend, what in (("H8", "13", 8, "the adapter's cash-timed orders and per-call discounting"),
                                  ("H9", "15", 9, "refused orders (minimum order not observed, invalid quantities offered); tested with v15 (amendment 10: v14 never trained, its audit found closure cheaper than paying)")):
        md += _vending_test(runs, hyp, ver, amend, what)
    md += _familiarity_test(runs)
    md += _rehearsal_test(runs)
    md += _head_reset_test(runs)
    md += _head_bank_test(runs)
    md += _early_transfer_test(runs)
    md += _bank_breadth_test(runs)
    md += _anchor_test(runs)
    md += _core_lr_test(runs)
    mp = _mp_breadth_test(runs) + _mp_dilution_test(runs)
    if mp and not md:
        md += ["## 0. Pre-registered tests", ""]
    md += mp
    return md


ROOT = Path(".")


def _iter_stats(name, lo=3, hi=40):
    """Mean policy entropy and advantage std over training iterations lo..hi (≈ 8k–165k interactions at seg 4096):
    the exploration diagnostic of amendment 13. None when the run has not reached iteration hi."""
    p = ROOT / name / "iters.jsonl"
    try:
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return None
    sel = [r["learn"] for r in rows if lo <= r.get("iter", 0) <= hi and "learn" in r]
    if len(sel) < hi - lo + 1:
        return None
    return float(np.mean([s["ent"] for s in sel])), float(np.mean([s["adv_std"] for s in sel]))


def _head_reset_test(runs) -> list:
    """H12 (amendment 13): the negative transfer to bizdev lives in the transferred action-selection heads (pointer
    query, slot decoder, skill selector, termination), whose source preferences collapse exploration in the new world;
    re-initialising exactly those heads at transfer, keeping perception, belief, memory, skill embeddings, value and
    world model, removes it."""
    if not any(re.fullmatch(r"T_B16_from\w+_s\d", n) for n in runs):
        return []
    fam, budget = "cam/bizdev", 300_000

    def auc(n):
        cv = ev_val(runs[n], fam) if n in runs else []
        return learning_auc(cv, "score", budget) if cv and max(x for x, _ in cv) >= budget else None

    def f(x, sg=False):
        return "–" if x is None else (f"{x:+.3f}" if sg else f"{x:.3f}")

    md = ["**Exploration diagnostic (amendment 13).** Mean policy entropy and advantage std over training iterations "
          "3–40 (≈ 8k–165k interactions) of every v16 bizdev transfer.", "",
          "| run | entropy | advantage std | learning AUC |", "|---|---|---|---|"]
    for n in sorted(x for x in runs if re.fullmatch(r"T_B1[67]_\w+_s\d", x)):
        es = _iter_stats(n)
        md.append(f"| {n} | {f(es[0] if es else None)} | {f(es[1] if es else None)} | {f(auc(n))} |")
    if not any(re.fullmatch(r"T_B16_from(N|NP)hr_s\d", n) for n in runs):
        return md + [""]
    md += ["", "**H12 (amendment 13): the negative transfer to bizdev lives in the transferred action-selection heads, whose "
           "source preferences collapse exploration in the new world; re-initialising exactly those heads (`query.`, "
           "`slot.`, `hi.`, `term.`) at transfer removes it.** Learning AUC of validation score (cam/bizdev, mode full, "
           "0-300k), complete runs paired by seed against `T_B16_scratch` (same v16 code and gate). Confirmed if both "
           "head-reset groups (fromNhr, fromNPhr) have mean ΔAUC ≥ −0.02; disproved if fromNPhr mean ΔAUC ≤ −0.10 (the "
           "deficit survives the reset, so it lives in the shared representation, value or world model). Mechanism check: "
           "head-reset runs' mean entropy over iterations 3–40 ≥ 0.9 × the same seed's scratch.", "",
           "| init | seed | 0k score | AUC | Δ vs scratch | Δ vs same source, heads kept | entropy / scratch |",
           "|---|---|---|---|---|---|---|"]
    res, mech = {}, []
    for init, kept in (("fromNhr", "fromN"), ("fromNPhr", "fromNP")):
        for s in (1, 2, 3):
            n = f"T_B16_{init}_s{s}"
            if n not in runs:
                continue
            a, sc, k = auc(n), auc(f"T_B16_scratch_s{s}"), auc(f"T_B16_{kept}_s{s}")
            z = _val_at(runs[n], fam, "m0", "auc_score")
            e, es = _iter_stats(n), _iter_stats(f"T_B16_scratch_s{s}")
            er = None if not e or not es else e[0] / es[0]
            d1 = None if a is None or sc is None else a - sc
            d2 = None if a is None or k is None else a - k
            md.append(f"| {init} | {s} | {f(z)} | {f(a)} | {f(d1, True)} | {f(d2, True)} | {f(er)} |")
            res.setdefault(init, []).append(d1)
            mech.append(er)
    done = {k: [x for x in v if x is not None] for k, v in res.items()}
    if set(done) != {"fromNhr", "fromNPhr"} or not all(done.values()) or any(len(v) < 2 for v in done.values()):
        verdict = "pending (" + ", ".join(f"{k}: {len(v)} complete" for k, v in sorted(done.items())) + ")"
    else:
        means = {k: float(np.mean(v)) for k, v in done.items()}
        if all(m >= -0.02 for m in means.values()):
            verdict = "CONFIRMED"
        elif means["fromNPhr"] <= -0.10:
            verdict = "DISPROVED"
        else:
            verdict = "neither confirmed nor disproved (mixed)"
        verdict += " (" + ", ".join(f"{k} mean Δ {m:+.3f}" for k, m in sorted(means.items())) + ")"
        mm = [x for x in mech if x is not None]
        verdict += f"; mechanism check {'passed' if mm and all(x >= 0.9 for x in mm) else 'failed'}"
    return md + ["", f"H12 verdict: **{verdict}**", ""]


def _familiarity_test(runs) -> list:
    """H10 (amendment 11): negative transfer to bizdev is caused by confident action preferences not earned in the new
    world; the v16 familiarity gate removes it."""
    g = {n: s for n, s in runs.items() if re.fullmatch(r"T_B16_(scratch|fromN|fromNP)_s\d", n)}
    if not g:
        return []
    fam, budget = "cam/bizdev", 300_000

    def auc(n):
        cv = ev_val(runs[n], fam) if n in runs else []
        return learning_auc(cv, "score", budget) if cv and max(x for x, _ in cv) >= budget else None

    def f(x, sg=False):
        return "–" if x is None else (f"{x:+.3f}" if sg else f"{x:.3f}")

    md = ["**H10 (amendment 11): experienced inits learn bizdev slower than scratch because their action preferences, "
          "not earned in bizdev, are confident and wrong; the v16 familiarity gate removes the negative transfer.** "
          "Learning AUC of validation score (cam/bizdev, mode full, 0-300k), complete runs, paired by seed. Confirmed if "
          "both gated transfer groups (fromN, fromNP) have mean ΔAUC ≥ −0.02 against gated scratch; disproved if gated "
          "fromNP (the significant ungated deficit, −0.156) stays ≤ −0.10 (criteria revised 15:30 UTC, before any v16 "
          "run). Mechanism check: gated transfers start at validation score ≥ 0.20 (0k).", "",
          "| init | seed | 0k score | AUC v16 (gated) | AUC v11 (no gate) | Δ vs v16 scratch | Δ vs v11 same init |",
          "|---|---|---|---|---|---|---|"]
    res = {}
    for init in ("scratch", "fromN", "fromNP"):
        for s in (1, 2, 3):
            n16, n11 = f"T_B16_{init}_s{s}", f"T_B11_{init}_s{s}"
            if n16 not in runs:
                continue
            a16, a11, sc = auc(n16), auc(n11), auc(f"T_B16_scratch_s{s}")
            z = _val_at(runs[n16], fam, "m0", "auc_score")
            d1 = None if a16 is None or sc is None else a16 - sc
            d2 = None if a16 is None or a11 is None else a16 - a11
            md.append(f"| {init} | {s} | {f(z)} | {f(a16)} | {f(a11)} | {f(d1, True) if init != 'scratch' else '–'} | {f(d2, True)} |")
            if init != "scratch":
                res.setdefault(init, []).append(d1)
    done = {k: [x for x in v if x is not None] for k, v in res.items()}
    if set(done) != {"fromN", "fromNP"} or not all(done.values()):
        verdict = "pending (" + ", ".join(f"{k}: {len(v)} complete" for k, v in sorted(done.items())) + ")"
    else:
        means = {k: float(np.mean(v)) for k, v in done.items()}
        if all(m >= -0.02 for m in means.values()):
            verdict = "CONFIRMED"
        elif means["fromNP"] <= -0.10:
            verdict = "DISPROVED"
        else:
            verdict = "neither confirmed nor disproved (mixed)"
        verdict += " (" + ", ".join(f"{k} mean Δ {m:+.3f}" for k, m in sorted(means.items())) + ")"
    return md + ["", f"H10 verdict: **{verdict}**", ""]



def _win_auc(cv, lo, hi, metric="score"):
    """Normalised trapezoid AUC of a validation curve over [lo, hi] (None unless milestones lo and hi both exist)."""
    pts = [(x, auc_value(v, metric)) for x, v in cv if lo <= x <= hi]
    if len(pts) < 2 or pts[0][0] != lo or pts[-1][0] != hi:
        return None
    xs, ys = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
    return float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))


def _fmt(x, sg=False):
    return "–" if x is None else (f"{x:+.3f}" if sg else f"{x:.3f}")


def _head_bank_test(runs) -> list:
    """H13 (amendment 14): forgetting of negotiation while learning bizdev, and the bizdev failure under negotiation
    rehearsal (H11), both come from the two worlds sharing action-selection heads; v17's per-world heads remove both."""
    if not any(re.fullmatch(r"T_B17_fromNbank(rh)?_s\d", n) for n in runs):
        return []
    fam, src, budget = "cam/bizdev", "cam/negotiation", 300_000

    def auc(n):
        cv = ev_val(runs[n], fam) if n in runs else []
        return learning_auc(cv, "score", budget) if cv and max(x for x, _ in cv) >= budget else None

    def neg(n):  # negotiation validation success (start, at the budget milestone); None until reached
        cv = ev_val(runs[n], src) if n in runs else []
        return (auc_value(cv[0][1], "success") if cv else None), next((auc_value(v, "success") for x, v in cv if x >= budget), None)

    md = ["**H13 (amendment 14): forgetting of negotiation while learning bizdev, and the bizdev failure under negotiation "
          "rehearsal (H11), both come from the two worlds sharing action-selection heads; giving bizdev its own heads (v17 "
          "head bank) while negotiation keeps the transferred ones removes both.** H13a: confirmed if `T_B17_fromNbank` "
          "keeps negotiation validation success ≥ 0.40 at 300k in both seeds; disproved if ≤ 0.20 in both. H13b: confirmed "
          "if `T_B17_fromNbankrh` keeps negotiation ≥ 0.50 at 300k in both seeds and its mean bizdev ΔAUC (0-300k, paired "
          "by seed against `T_B16_scratch`) ≥ −0.05; disproved if mean bizdev ΔAUC ≤ −0.15. Mechanism check: "
          "`T_B17_fromNbank` entropy (iterations 3–40) ≥ 0.9 × same-seed scratch. Shared-head references: `T_B16_fromN` "
          "(no rehearsal) and `T_B16_fromNrh` (rehearsal).", "",
          "| run | heads | negotiation start → 300k | bizdev AUC | Δ vs scratch | entropy / scratch |",
          "|---|---|---|---|---|---|"]
    res = {}
    for arm, ref, heads in (("fromNbank", "fromN", "bizdev own"), ("fromNbankrh", "fromNrh", "bizdev own")):
        for s in (1, 2):
            sc = f"T_B16_scratch_s{s}"
            es = _iter_stats(sc)
            for name, h in ((f"T_B17_{arm}_s{s}", heads), (f"T_B16_{ref}_s{s}", "shared")):
                if name not in runs:
                    continue
                n0, n1 = neg(name)
                a, b = auc(name), auc(sc)
                d = None if a is None or b is None else a - b
                e = _iter_stats(name)
                er = None if e is None or es is None else e[0] / es[0]
                md.append(f"| {name} | {h} | {_fmt(n0)} → {_fmt(n1)} | {_fmt(a)} | {_fmt(d, True)} | {_fmt(er)} |")
                if name.startswith("T_B17_"):
                    res.setdefault(arm, []).append((n1, d, er))
    md.append("")
    bank, bankrh = res.get("fromNbank", []), res.get("fromNbankrh", [])
    if len(bank) < 2 or any(k is None for k, _, _ in bank):
        va = f"pending ({sum(k is not None for k, _, _ in bank)} of 2 runs at 300k)"
    elif all(k >= 0.40 for k, _, _ in bank):
        va = "CONFIRMED"
    elif all(k <= 0.20 for k, _, _ in bank):
        va = "DISPROVED"
    else:
        va = "neither confirmed nor disproved (mixed)"
    mech = ""
    if len(bank) == 2 and all(er is not None for _, _, er in bank):
        mech = "; mechanism check " + ("passed" if all(er >= 0.9 for _, _, er in bank) else "failed")
    if len(bankrh) < 2 or any(k is None or d is None for k, d, _ in bankrh):
        vb = f"pending ({sum(d is not None for _, d, _ in bankrh)} of 2 runs complete)"
    else:
        md_ = float(np.mean([d for _, d, _ in bankrh]))
        if all(k >= 0.50 for k, _, _ in bankrh) and md_ >= -0.05:
            vb = "CONFIRMED"
        elif md_ <= -0.15:
            vb = "DISPROVED"
        else:
            vb = "neither confirmed nor disproved (mixed)"
        vb += f" (mean bizdev ΔAUC {md_:+.3f}; negotiation at 300k " + ", ".join(f"{k:.3f}" for k, _, _ in bankrh) + ")"
    if bank and all(k is not None for k, _, _ in bank):
        va += " (negotiation at 300k " + ", ".join(f"{k:.3f}" for k, _, _ in bank) + ")"
    md += [f"H13a verdict: **{va}**{mech}", "", f"H13b verdict: **{vb}**", ""]
    return md


def _early_transfer_test(runs) -> list:
    """H14 (amendment 14): with fresh action heads, experience in other CAM worlds speeds early learning of bizdev (0-50k
    AUC), and broader experience (negotiation + product) more than negotiation alone. New seeds 3-4 only."""
    if not any(re.fullmatch(r"T_B16_(scratch|fromNhr|fromNPhr)_s[34]", n) for n in runs):
        return []
    fam = "cam/bizdev"

    def w(n, lo, hi):
        return _win_auc(ev_val(runs[n], fam), lo, hi) if n in runs else None

    md = ["**H14 (amendment 14): with fresh action heads, experience in other CAM worlds speeds early learning of bizdev, "
          "and broader experience speeds it more.** Early AUC = normalised area under bizdev validation score over 0-50k. "
          "New seeds 3-4 only (seeds 1-2 produced the hypothesis and are shown for reference). H14a: confirmed if fromNPhr "
          "> scratch in both new seeds with mean difference ≥ +0.03, disproved if the mean difference ≤ 0. H14b: confirmed "
          "if fromNPhr > fromNhr in both new seeds, disproved if the mean difference ≤ 0.", "",
          "| seed | scratch 0-50k / 50-300k | fromNhr 0-50k / 50-300k | fromNPhr 0-50k / 50-300k | fromNPhr − scratch (0-50k) | fromNPhr − fromNhr (0-50k) |",
          "|---|---|---|---|---|---|"]
    da, db = [], []
    for s in (1, 2, 3, 4):
        e = {i: w(f"T_B16_{i}_s{s}", 0, 50_000) for i in ("scratch", "fromNhr", "fromNPhr")}
        l = {i: w(f"T_B16_{i}_s{s}", 50_000, 300_000) for i in ("scratch", "fromNhr", "fromNPhr")}
        if all(v is None for v in e.values()):
            continue
        a = None if e["fromNPhr"] is None or e["scratch"] is None else e["fromNPhr"] - e["scratch"]
        b = None if e["fromNPhr"] is None or e["fromNhr"] is None else e["fromNPhr"] - e["fromNhr"]
        tag = f"{s}" + (" (reference)" if s < 3 else "")
        md.append(f"| {tag} | " + " | ".join(f"{_fmt(e[i])} / {_fmt(l[i])}" for i in ("scratch", "fromNhr", "fromNPhr"))
                  + f" | {_fmt(a, True)} | {_fmt(b, True)} |")
        if s >= 3:
            da.append(a)
            db.append(b)
    md.append("")

    def verdict(d, both_pos_extra):
        if len(d) < 2 or any(x is None for x in d):
            return f"pending ({sum(x is not None for x in d)} of 2 new seeds)"
        m = float(np.mean(d))
        if all(x > 0 for x in d) and both_pos_extra(m):
            return f"CONFIRMED (mean {m:+.3f})"
        if m <= 0:
            return f"DISPROVED (mean {m:+.3f})"
        return f"neither confirmed nor disproved (mixed) (mean {m:+.3f})"

    md += [f"H14a verdict (experience): **{verdict(da, lambda m: m >= 0.03)}**", "",
           f"H14b verdict (breadth): **{verdict(db, lambda m: True)}**", ""]
    return md


def _bank_breadth_test(runs) -> list:
    """Amendment 16: H13c (H13b on new seeds 3-4) and H16 (breadth with per-world heads: fromNPbankrh vs fromNbankrh,
    seeds 1-4, bizdev AUC 0-300k)."""
    if not any(re.fullmatch(r"T_B17_(fromNPbankrh_s\d|fromNbankrh_s[34])", n) for n in runs):
        return []
    fam, src, budget = "cam/bizdev", "cam/negotiation", 300_000

    def auc(n):
        cv = ev_val(runs[n], fam) if n in runs else []
        return learning_auc(cv, "score", budget) if cv and max(x for x, _ in cv) >= budget else None

    def win(n, lo, hi):
        return _win_auc(ev_val(runs[n], fam), lo, hi) if n in runs else None

    def neg(n):
        cv = ev_val(runs[n], src) if n in runs else []
        return next((auc_value(v, "success") for x, v in cv if x >= budget), None)

    md = ["**H13c and H16 (amendment 16).** H13c: `T_B17_fromNbankrh` on new seeds 3-4 replicates H13b if negotiation "
          "≥ 0.50 at 300k in both and mean bizdev ΔAUC vs `T_B16_scratch_s3/s4` ≥ −0.05; fails if mean ΔAUC ≤ −0.15 or "
          "negotiation < 0.50 in both. H16: paired by seed (1-4), bizdev AUC 0-300k of `T_B17_fromNPbankrh` − "
          "`T_B17_fromNbankrh`; confirmed if mean ≥ +0.02 and positive in ≥ 3 of 4 seeds, disproved if mean ≤ 0.", "",
          "| seed | scratch AUC | fromNbankrh AUC (Δ vs scratch) | fromNPbankrh AUC (Δ vs scratch) | NP − N | 0-50k scratch / N / NP | "
          "50k-300k scratch / N / NP | negotiation at 300k N / NP |",
          "|---|---|---|---|---|---|---|---|"]
    rep, br = [], []
    for s_ in (1, 2, 3, 4):
        sc, nb, npb = f"T_B16_scratch_s{s_}", f"T_B17_fromNbankrh_s{s_}", f"T_B17_fromNPbankrh_s{s_}"
        if not any(x in runs for x in (nb, npb)):
            continue
        a_s, a_n, a_p = auc(sc), auc(nb), auc(npb)
        dn = None if a_n is None or a_s is None else a_n - a_s
        dp = None if a_p is None or a_s is None else a_p - a_s
        d = None if a_n is None or a_p is None else a_p - a_n
        e = [win(x, 0, 50_000) for x in (sc, nb, npb)]
        l = [win(x, 50_000, budget) for x in (sc, nb, npb)]
        md.append(f"| {s_} | {_fmt(a_s)} | {_fmt(a_n)} ({_fmt(dn, True)}) | {_fmt(a_p)} ({_fmt(dp, True)}) | {_fmt(d, True)} | "
                  + " / ".join(_fmt(x) for x in e) + " | " + " / ".join(_fmt(x) for x in l)
                  + f" | {_fmt(neg(nb))} / {_fmt(neg(npb))} |")
        if s_ >= 3:
            rep.append((neg(nb), dn))
        br.append(d)
    md.append("")
    if len(rep) < 2 or any(k is None or d is None for k, d in rep):
        vc = f"pending ({sum(d is not None for _, d in rep)} of 2 new seeds complete)"
    else:
        m = float(np.mean([d for _, d in rep]))
        if all(k >= 0.50 for k, _ in rep) and m >= -0.05:
            vc = "REPLICATES"
        elif m <= -0.15 or all(k < 0.50 for k, _ in rep):
            vc = "FAILS TO REPLICATE"
        else:
            vc = "neither (mixed)"
        vc += f" (mean bizdev ΔAUC {m:+.3f}; negotiation at 300k " + ", ".join(f"{k:.3f}" for k, _ in rep) + ")"
    done = [d for d in br if d is not None]
    if len(br) < 4 or len(done) < 4:
        v16 = f"pending ({len(done)} of 4 seed pairs complete)"
    else:
        m = float(np.mean(done))
        if m >= 0.02 and sum(d > 0 for d in done) >= 3:
            v16 = "CONFIRMED"
        elif m <= 0:
            v16 = "DISPROVED"
        else:
            v16 = "neither confirmed nor disproved (mixed)"
        v16 += f" (mean NP − N {m:+.3f}; positive in {sum(d > 0 for d in done)} of 4)"
    return md + [f"H13c verdict (replication): **{vc}**", "", f"H16 verdict (breadth with per-world heads): **{v16}**", ""]

def _iters_mean(name, key, lo, hi):
    """Mean of a learner statistic over training iterations with lo < interactions <= hi (None if not reached)."""
    try:
        rows = [json.loads(x) for x in (ROOT / name / "iters.jsonl").read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return None
    if not rows or rows[-1].get("interactions", 0) < hi:
        return None
    sel = [r["learn"][key] for r in rows if lo < r.get("interactions", 0) <= hi and key in r.get("learn", {})]
    return float(np.mean(sel)) if sel else None


def _anchor_test(runs) -> list:
    """H17 (amendment 17): rehearsal collapses when the rehearsed world's policy degenerates (all decisions alike, all
    failures equally costly, zero advantage); a KL anchor to the frozen source policy restores and prevents it."""
    if not any(n.startswith("T_B18_rescue_") or n.startswith("T_B18_fromNbankanc_") for n in runs):
        return []
    fam, src = "cam/bizdev", "cam/negotiation"

    def neg_at(n, x):
        cv = ev_val(runs[n], src) if n in runs else []
        return next((auc_value(v, "success") for t, v in cv if t == x), None)

    def neg_curve(n):
        cv = ev_val(runs[n], src) if n in runs else []
        return " → ".join(f"{auc_value(v, 'success'):.3f}" for _, v in cv) or "–"

    def auc(n, budget=300_000):
        cv = ev_val(runs[n], fam) if n in runs else []
        return learning_auc(cv, "score", budget) if cv and max(x for x, _ in cv) >= budget else None

    md = ["**H17 (amendment 17): on-policy rehearsal cannot recover a degenerate source-world policy (identical returns, "
          "zero advantage); a KL anchor to the frozen source policy (coefficient 0.05, shared-head decisions) restores and "
          "prevents it.** H17a (rescue, valid only if `T_B17_fromNbankrh_s4` ends with negotiation ≤ 0.20 at 300k): "
          "continuing its final weights for 100k, confirmed if the anchor arm reaches negotiation ≥ 0.40 at 100k in both "
          "seeds while the control stays ≤ 0.20 in both; disproved if the anchor arm is ≤ 0.20 in both. H17b (prevention, "
          "seeds 1-4): confirmed if `T_B18_fromNbankanc` keeps negotiation ≥ 0.50 at 300k in all four seeds and its mean "
          "bizdev ΔAUC (0-300k, paired against `T_B16_scratch`) ≥ −0.05; disproved if negotiation ≤ 0.20 at 300k in any "
          "seed or the mean ΔAUC ≤ −0.15.", "",
          "| run | negotiation (validation success per milestone) | bizdev score at the last milestone | mean anchor KL |",
          "|---|---|---|---|"]
    for n in sorted(x for x in runs if x.startswith("T_B18_rescue_")):
        cv = ev_val(runs[n], fam)
        md.append(f"| {n} | {neg_curve(n)} | {_fmt(auc_value(cv[-1][1], 'score') if cv else None)} | "
                  f"{_fmt(_iters_mean(n, 'anchor_kl', 0, 100_000))} |")
    md.append("")
    base = neg_at("T_B17_fromNbankrh_s4", 300_000)
    resc = {a: [neg_at(f"T_B18_rescue_{a}_s{s}", 100_000) for s in (1, 2)] for a in ("anc", "ctl")}
    if base is None:
        va = "pending (T_B17_fromNbankrh_s4 not at 300k)"
    elif base > 0.20:
        va = f"not applicable (T_B17_fromNbankrh_s4 ended at negotiation {base:.3f} > 0.20)"
    elif any(x is None for v in resc.values() for x in v):
        va = f"pending ({sum(x is not None for v in resc.values() for x in v)} of 4 runs at 100k)"
    elif all(x >= 0.40 for x in resc["anc"]) and all(x <= 0.20 for x in resc["ctl"]):
        va = "CONFIRMED"
    elif all(x <= 0.20 for x in resc["anc"]):
        va = "DISPROVED"
    else:
        va = "neither confirmed nor disproved (mixed)"
    if base is not None and base <= 0.20 and all(x is not None for v in resc.values() for x in v):
        va += " (anchor " + ", ".join(f"{x:.3f}" for x in resc["anc"]) + "; control " + ", ".join(f"{x:.3f}" for x in resc["ctl"]) + ")"
    md += ["| seed | negotiation anchor (per milestone) | no anchor, `T_B17_fromNbankrh` | bizdev AUC anchor (Δ vs scratch) | "
           "no anchor (Δ) | mean anchor KL 50k-300k |", "|---|---|---|---|---|---|"]
    prev = []
    for s_ in (1, 2, 3, 4):
        a, b, sc = f"T_B18_fromNbankanc_s{s_}", f"T_B17_fromNbankrh_s{s_}", f"T_B16_scratch_s{s_}"
        if a not in runs:
            continue
        aa, ab, asc = auc(a), auc(b), auc(sc)
        da = None if aa is None or asc is None else aa - asc
        db = None if ab is None or asc is None else ab - asc
        md.append(f"| {s_} | {neg_curve(a)} | {neg_curve(b)} | {_fmt(aa)} ({_fmt(da, True)}) | {_fmt(ab)} ({_fmt(db, True)}) | "
                  f"{_fmt(_iters_mean(a, 'anchor_kl', 50_000, 300_000))} |")
        prev.append((neg_at(a, 300_000), da))
    md.append("")
    if len(prev) < 4 or any(k is None or d is None for k, d in prev):
        vb = f"pending ({sum(d is not None for _, d in prev)} of 4 runs complete)"
    else:
        m = float(np.mean([d for _, d in prev]))
        if all(k >= 0.50 for k, _ in prev) and m >= -0.05:
            vb = "CONFIRMED"
        elif any(k <= 0.20 for k, _ in prev) or m <= -0.15:
            vb = "DISPROVED"
        else:
            vb = "neither confirmed nor disproved (mixed)"
        vb += f" (mean bizdev ΔAUC {m:+.3f}; negotiation at 300k " + ", ".join(f"{k:.3f}" for k, _ in prev) + ")"
    return md + [f"H17a verdict (rescue): **{va}**", "", f"H17b verdict (prevention): **{vb}**", ""]


def _mp_breadth_test(runs) -> list:
    """H15 (amendment 15): at equal total experience (1M agent-steps), six Melting Pot substrates transfer to the
    held-out commons_harvest__closed at least as well as its nearest training substrate, commons_harvest__open, alone."""
    if not any(n.startswith("T_closed_fromMP1open_") for n in runs):
        return []
    fam, bos, budget = "mp/commons_harvest__closed", "mp/bach_or_stravinsky_in_the_matrix__repeated", 1_000_000

    def curve(n, f=fam):
        return ev_val(runs[n], f) if n in runs else []

    def auc(n):
        cv = curve(n)
        return learning_auc(cv, "score", budget) if cv and max(x for x, _ in cv) >= budget else None

    def at(n, x, f=fam):
        return next((auc_value(v, "score") for t, v in curve(n, f) if t == x), None)

    def g(x, sg=False):
        return "–" if x is None else (f"{x:+.2f}" if sg else f"{x:.2f}")

    md = ["**H15 (amendment 15): at equal total experience (1M agent-steps), experience spread over six Melting Pot "
          "substrates transfers to the held-out commons_harvest__closed at least as well as experience on its nearest "
          "training substrate, commons_harvest__open, alone.** Paired by transfer seed (1-3), fromMP6 − fromMP1open. "
          "Primary: learning AUC of closed validation score, 0-1M; secondary: zero-shot (0k validation score in the "
          "transfer runs). Confirmed if the mean AUC difference is ≥ 0 and the mean zero-shot difference is ≥ −1.0; "
          "disproved if both are ≤ −1.0; anything else mixed.", "",
          "| seed | AUC fromMP6 | AUC fromMP1open | AUC scratch | Δ AUC (MP6 − MP1open) | fromMP6 − scratch | "
          "fromMP1open − scratch | zero-shot fromMP6 | zero-shot fromMP1open | Δ zero-shot |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    dA, dZ = [], []
    for s in (1, 2, 3):
        a6, a1, a0 = (auc(f"T_closed_{arm}_s{s}") for arm in ("fromMP6", "fromMP1open", "scratch"))
        z6, z1 = (at(f"T_closed_{arm}_s{s}", 0) for arm in ("fromMP6", "fromMP1open"))
        da = None if a6 is None or a1 is None else a6 - a1
        dz = None if z6 is None or z1 is None else z6 - z1
        if da is not None and dz is not None:
            dA.append(da)
            dZ.append(dz)
        md.append(f"| {s} | {g(a6)} | {g(a1)} | {g(a0)} | {g(da, True)} | "
                  f"{g(None if a6 is None or a0 is None else a6 - a0, True)} | "
                  f"{g(None if a1 is None or a0 is None else a1 - a0, True)} | {g(z6)} | {g(z1)} | {g(dz, True)} |")
    md += ["", "Lifetime held-out monitors (validation score, 0 → 1M):", ""]
    for n in ("P_MP6_s1", "P_MP6_s2", "P_MP1open_s1", "P_MP1open_s2"):
        if n in runs:
            md.append(f"- {n}: closed {g(at(n, 0))} → {g(at(n, budget))}; bos {g(at(n, 0, bos))} → {g(at(n, budget, bos))}")
    if len(dA) < 3:
        verdict = (f"pending ({len(dA)}/3 seeds complete in both arms"
                   + (f"; so far mean Δ AUC {np.mean(dA):+.2f}, Δ zero-shot {np.mean(dZ):+.2f}" if dA else "") + ")")
    else:
        mA, mZ = float(np.mean(dA)), float(np.mean(dZ))
        if mA >= 0 and mZ >= -1.0:
            verdict = "CONFIRMED"
        elif mA <= -1.0 and mZ <= -1.0:
            verdict = "DISPROVED"
        else:
            verdict = "neither confirmed nor disproved (mixed)"
        verdict += f" (mean Δ AUC {mA:+.2f}, mean Δ zero-shot {mZ:+.2f})"
    return md + ["", f"H15 verdict (breadth vs nearest single world): **{verdict}**", ""]


def _mp_dilution_test(runs) -> list:
    """H19 (amendment 18): the six-world lifetime's lack of transfer to commons_harvest__closed is interference, not
    dilution: open-only experience equal to its open share at 1M (172,032 agent-steps) transfers better."""
    if not any(n.startswith("T_closed_fromMP1openD_") or n.startswith("P_MP1openD_") for n in runs):
        return []
    fam, opn, budget = "mp/commons_harvest__closed", "mp/commons_harvest__open", 1_000_000

    def auc(n):
        cv = ev_val(runs[n], fam) if n in runs else []
        return learning_auc(cv, "score", budget) if cv and max(x for x, _ in cv) >= budget else None

    def zs(n):
        cv = ev_val(runs[n], fam) if n in runs else []
        return next((auc_value(v, "score") for t, v in cv if t == 0), None)

    def g(x, sg=False):
        return "–" if x is None else (f"{x:+.2f}" if sg else f"{x:.2f}")

    def d(a, b):
        return None if a is None or b is None else a - b

    md = ["**H19 (amendment 18): the six-world lifetime's lack of transfer to commons_harvest__closed is interference, "
          "not dilution: commons_harvest__open experience equal to its open share at 1M (172,032 agent-steps), without "
          "the other five substrates, transfers better.** Paired by transfer seed (1-3), Δ = learning AUC (closed "
          "validation score, 0-1M) of fromMP1openD minus fromMP6. Confirmed if the mean Δ ≥ +5.0; disproved if the mean "
          "Δ ≤ +1.0; anything else mixed.", "",
          "| seed | AUC open 172k / six worlds / open 1M / scratch | Δ (open 172k − six worlds) | open 172k − scratch | "
          "open 172k − open 1M | zero-shot open 172k / six worlds / scratch |", "|---|---|---|---|---|---|"]
    ds = []
    for sd in (1, 2, 3):
        aD, a6, a1, a0 = (auc(f"T_closed_{arm}_s{sd}") for arm in ("fromMP1openD", "fromMP6", "fromMP1open", "scratch"))
        zD, z6, z0 = (zs(f"T_closed_{arm}_s{sd}") for arm in ("fromMP1openD", "fromMP6", "scratch"))
        if d(aD, a6) is not None:
            ds.append(aD - a6)
        md.append(f"| {sd} | {g(aD)} / {g(a6)} / {g(a1)} / {g(a0)} | {g(d(aD, a6), True)} | {g(d(aD, a0), True)} | "
                  f"{g(d(aD, a1), True)} | {g(zD)} / {g(z6)} / {g(z0)} |")
    md += ["", "Source lifetimes (open only, so every interaction is an open agent-step):", ""]
    for n in ("P_MP1openD_s1", "P_MP1openD_s2"):
        st = runs.get(n)
        if st:
            md.append(f"- {n}: {st.get('interactions', 0):,} interactions, phase {st.get('phase')}, "
                      f"saved {st.get('lifetime_saved')}")
    if len(ds) < 3:
        verdict = (f"pending ({len(ds)}/3 seeds complete" + (f"; so far mean Δ {np.mean(ds):+.2f}" if ds else "") + ")")
    else:
        m = float(np.mean(ds))
        verdict = ("CONFIRMED" if m >= 5.0 else "DISPROVED" if m <= 1.0 else "neither confirmed nor disproved (mixed)") + f" (mean Δ {m:+.2f})"
    return md + ["", f"H19 verdict (interference vs dilution): **{verdict}**", ""]


def _core_lr_test(runs) -> list:
    """H18 (amendment 17): the late deficit of fresh-head transfers comes from oversized per-update policy changes of
    the transferred core at scratch's learning rate; halving the core's learning rate removes it."""
    if not any(n.startswith("T_B18_fromNPhrcore_") or n.startswith("T_B18_scratchcore_") for n in runs):
        return []
    fam, budget = "cam/bizdev", 300_000

    def win(n, lo, hi):
        return _win_auc(ev_val(runs[n], fam), lo, hi) if n in runs else None

    md = ["**H18 (amendment 17): the late (50k-300k) deficit of fresh-head transfers comes from oversized per-update "
          "policy changes of the transferred core at scratch's learning rate; halving the learning rate of everything but "
          "the fresh action heads removes it.** `T_B18_fromNPhrcore` (= `T_B16_fromNPhr`, core lr × 0.5) against "
          "`T_B16_scratch`, paired by seed 1-4, AUC of bizdev validation score over 50k-300k. Mechanism check: mean "
          "logged per-update KL over 50k-300k, fromNPhrcore / same-seed scratch ≤ 1.1. Confirmed if the mean late Δ ≥ "
          "−0.005 and the check passes; disproved if the mean late Δ ≤ −0.027 (no better than `T_B16_fromNPhr`, whose "
          "mean was −0.027) while the check passes; otherwise mixed. `T_B18_scratchcore` (scratch with the same lr "
          "scale) is reported to show whether a smaller core step helps any learner.", "",
          "| seed | late AUC scratch / fromNPhr / fromNPhrcore / scratchcore | fromNPhrcore − scratch (late) | "
          "fromNPhr − scratch (late) | scratchcore − scratch (late) | early 0-50k scratch / fromNPhrcore | KL ratio core / scratch |",
          "|---|---|---|---|---|---|---|"]
    res = []
    for s_ in (1, 2, 3, 4):
        sc, fp, fc, scc = (f"T_B16_scratch_s{s_}", f"T_B16_fromNPhr_s{s_}", f"T_B18_fromNPhrcore_s{s_}", f"T_B18_scratchcore_s{s_}")
        if fc not in runs and scc not in runs:
            continue
        L = [win(x, 50_000, budget) for x in (sc, fp, fc, scc)]
        d = None if L[2] is None or L[0] is None else L[2] - L[0]
        dp = None if L[1] is None or L[0] is None else L[1] - L[0]
        ds = None if L[3] is None or L[0] is None else L[3] - L[0]
        k1, k0 = _iters_mean(fc, "kl", 50_000, budget), _iters_mean(sc, "kl", 50_000, budget)
        kr = None if k1 is None or not k0 else k1 / k0
        md.append(f"| {s_} | " + " / ".join(_fmt(x) for x in L) + f" | {_fmt(d, True)} | {_fmt(dp, True)} | {_fmt(ds, True)} | "
                  f"{_fmt(win(sc, 0, 50_000))} / {_fmt(win(fc, 0, 50_000))} | {_fmt(kr)} |")
        res.append((d, kr))
    md.append("")
    if len(res) < 4 or any(d is None or k is None for d, k in res):
        v = f"pending ({sum(d is not None for d, _ in res)} of 4 pairs complete)"
    else:
        m = float(np.mean([d for d, _ in res]))
        ok = all(k <= 1.1 for _, k in res)
        if m >= -0.005 and ok:
            v = "CONFIRMED"
        elif m <= -0.027 and ok:
            v = "DISPROVED"
        else:
            v = "neither confirmed nor disproved (mixed)" + ("" if ok else "; mechanism check failed (KL not matched)")
        v += f" (mean late Δ {m:+.3f}; KL ratios " + ", ".join(f"{k:.2f}" for _, k in res) + ")"
    return md + [f"H18 verdict: **{v}**", ""]

def _rehearsal_test(runs) -> list:
    """H11 (amendment 12): forgetting during transfer comes from training on the new world alone; rehearsing the source
    world on 10% of interactions keeps it at little cost to learning the new one."""
    g = {n for n in runs if re.fullmatch(r"T_B16_fromNrh_s\d", n)}
    if not g:
        return []
    fam, src, budget = "cam/bizdev", "cam/negotiation", 300_000

    def auc(n):
        cv = ev_val(runs[n], fam) if n in runs else []
        return learning_auc(cv, "score", budget) if cv and max(x for x, _ in cv) >= budget else None

    def kept(n):  # negotiation validation success at the budget milestone (None until reached)
        cv = ev_val(runs[n], src) if n in runs else []
        return next((auc_value(v, "success") for x, v in cv if x >= budget), None), (auc_value(cv[0][1], "success") if cv else None)

    def f(x, sg=False):
        return "–" if x is None else (f"{x:+.3f}" if sg else f"{x:.3f}")

    md = ["**H11 (amendment 12): forgetting during transfer comes from training on the new world alone, not from a capacity "
          "conflict; rehearsing negotiation on 10% of interactions keeps it.** Negotiation validation success (start → "
          "300k) and bizdev learning AUC (0-300k, total interactions) of `T_B16_fromNrh` against `T_B16_fromN` (same "
          "seed, v16, no rehearsal). Confirmed if negotiation success at 300k ≥ 0.50 in both seeds and mean bizdev ΔAUC "
          "≥ −0.03; disproved if negotiation success at 300k ≤ 0.30 in both seeds.", "",
          "| seed | negotiation, rehearsal | negotiation, no rehearsal | bizdev AUC, rehearsal | bizdev AUC, no rehearsal | ΔAUC |",
          "|---|---|---|---|---|---|"]
    keep, dd = [], []
    for s in (1, 2):
        a, b = f"T_B16_fromNrh_s{s}", f"T_B16_fromN_s{s}"
        if a not in runs:
            continue
        (ka, k0a), (kb, k0b) = kept(a), kept(b)
        aa, ab = auc(a), auc(b)
        d = None if aa is None or ab is None else aa - ab
        md.append(f"| {s} | {f(k0a)} → {f(ka)} | {f(k0b)} → {f(kb)} | {f(aa)} | {f(ab)} | {f(d, True)} |")
        keep.append(ka)
        dd.append(d)
    if len(keep) < 2 or any(x is None for x in keep + dd):
        verdict = f"pending ({sum(x is not None for x in dd)} of 2 seed pairs complete)"
    elif all(k >= 0.50 for k in keep) and float(np.mean(dd)) >= -0.03:
        verdict = "CONFIRMED"
    elif all(k <= 0.30 for k in keep):
        verdict = "DISPROVED"
    else:
        verdict = "neither confirmed nor disproved (mixed)"
    if dd and all(x is not None for x in dd):
        verdict += f" (mean ΔAUC {float(np.mean(dd)):+.3f}; negotiation at 300k " + ", ".join(f(k) for k in keep) + ")"
    return md + ["", f"H11 verdict: **{verdict}**", ""]


def _vending_test(runs, hyp, ver, amend, what) -> list:
    md = []
    vv = {n: s for n, s in runs.items() if re.fullmatch(rf"T_V{ver}_scratch_s\d", n)}
    if vv:
        md += [f"**{hyp} (amendment {amend}): Vending's failure is caused by {what}.** Validation score (ext/vending, mode "
               f"full; do-nothing = 0.760). Confirmed if both v{ver} seeds reach ≥ 0.85 at 300k with a return different from "
               "the do-nothing return; disproved if both are ≤ 0.78.", "",
               "| run | adapter | 0k | 100k | 300k | return at 300k (do-nothing) |", "|---|---|---|---|---|---|"]
        res = {}
        olds = {n: runs[n] for n in runs if re.fullmatch(r"T_V_scratch_s\d", n)}
        for n, s in sorted(olds.items()) + sorted(vv.items()):
            v = [_val_at(s, "ext/vending", tg, "auc_score") for tg in ("m0", "m100000", "m300000")]
            e = next((e for e in s.get("evals", []) if e["tag"] == "m300000" and e["split"] == "val" and e["mode"] == "full"), None)
            ret = None if e is None else float(np.mean([c["return"] for c in e["families"]["ext/vending"]["curve"]]))
            stall = ((s.get("reward_audit") or {}).get("ext/vending") or {}).get("stall", {}).get("return")
            md.append(f"| {n} | {('v' + ver) if n in vv else 'v1 (sl-exp-v10)'} | " + " | ".join("–" if x is None else f"{x:.3f}" for x in v)
                      + f" | {'–' if ret is None else f'{ret:.3f}'} ({'–' if stall is None else f'{stall:.3f}'}) |")
            if n in vv:
                res[n] = (v[2], ret, stall)
        done = [r for r in res.values() if r[0] is not None]
        if len(done) < 2:
            verdict = f"pending ({len(done)}/2 v{ver} seeds at 300k)"
        elif all(sc <= 0.78 for sc, _, _ in done):
            verdict = "DISPROVED"
        elif all(sc >= 0.85 and (st is None or abs(r - st) > 0.05 * max(1.0, abs(st))) for sc, r, st in done):
            verdict = "CONFIRMED"
        else:
            verdict = "neither confirmed nor disproved (mixed)"
        md += ["", f"{hyp} verdict: **{verdict}**", ""]
    return md


def paired_adaptation(root: Path) -> list:
    """Offline paired fast-adaptation evaluations (scripts/sl_adapt_eval.py) stored in <root>/_analysis/adapt_paired."""
    d = root / "_analysis" / "adapt_paired"
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    if not files:
        return []
    md = ["## 1b. Fast adaptation, paired by task (offline, frozen weights, scripts/sl_adapt_eval.py)", "",
          "Many validation tasks, every mode on the same tasks. DiD = mean over tasks of [gain(full) − gain(episode)], "
          "gain = episode K − episode 1 (memory carried across episodes); last columns compare episode K between modes. "
          "90% intervals from per-task paired differences.", "",
          "| checkpoint | family | tasks | full ep1 → epK | episode ep1 → epK | none ep1 → epK | DiD | epK full − episode | epK full − none |",
          "|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "–" if not x else f"{x['mean']:+.3f} ± {x['ci90']:.3f}"
    for fp in files:
        r = json.loads(fp.read_text())
        m = r["modes"]
        cell = lambda k: f"{m[k]['first']:.3f} → {m[k]['last']:.3f}" if k in m else "–"
        ck = f"{Path(r['run']).name} @ {Path(r['ckpt']).name}"
        md.append(f"| {ck} | {r['family']} ({r['metric']}) | {r['tasks']} | {cell('full')} | {cell('episode')} | {cell('none')} | "
                  f"{f(r.get('did_full_vs_episode'))} | {f(r.get('lastK_full_minus_episode'))} | {f(r.get('lastK_full_minus_none'))} |")
    return md + [""]


def inaction_check(runs) -> list:
    """Does a learned policy merely reproduce the reward audit's do-nothing ("stall") baseline? A learning curve that
    ends at the inaction return is not competence (found 2026-11-19 for product and Vending-Bench)."""
    rows = []
    for n, s in sorted(runs.items()):
        aud = s.get("reward_audit") or {}
        last = next((e for e in reversed(s.get("evals", [])) if e["split"] == "val" and e["mode"] == "full"), None)
        if last is None:
            continue
        cfg = s.get("config") or {}
        mine = set(cfg.get("families") or {}) | set(cfg.get("heldout_families") or [])
        for fam, f in sorted(last["families"].items()):
            if fam not in mine:  # retention monitors of transfer runs are reported in section 2
                continue
            st = (aud.get(fam) or {}).get("stall")
            if not st or not f.get("curve"):
                continue
            ret = float(np.mean([c["return"] for c in f["curve"]]))
            ref = float(st["return"])
            same = abs(ret - ref) <= 0.05 * max(1.0, abs(ref))
            kind = "trained" if fam in (cfg.get("families") or {}) else "held out"
            rows.append(f"| {n} @ {last['tag']} | {fam} ({kind}) | {ret:.3f} | {ref:.3f} | {'**yes**' if same else 'no'} |")
    if not rows:
        return []
    return ["## 0b. Inaction check (validation return vs the audit's do-nothing policy)", "",
            "A family whose learned return equals the stall policy's return has learned inaction, not the task; its "
            "score gains (e.g. Vending 0.64 → 0.76 = no longer losing money on random orders) are not competence.", "",
            "| run | family | learned return | do-nothing return | ≈ inaction |", "|---|---|---|---|---|", *rows, ""]


def _count(runs):
    c = {}
    for st in runs.values():
        c[st["phase"]] = c.get(st["phase"], 0) + 1
    return c


if __name__ == "__main__":
    raise SystemExit(main())
