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
    return md


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
