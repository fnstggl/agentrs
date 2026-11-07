"""Aggregate results/temporal/*.json into markdown tables (printed; pasted into docs)."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2] / "results" / "temporal"


def load(pat):
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(R / pat)))]


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def exp_a():
    rs = load("expA_c*_s*.json")
    if not rs:
        return
    arms = ["marginal_km", "single_step", "discrete", "temporal", "no_content", "temporal_tuned", "no_content_tuned"]
    print(f"\n### Experiment A ({len(rs)} corpora; mean over 5 events)\n")
    print("| model | test Brier (profile h) | test AUROC | test Brier (between h) | unseen-env Brier | unseen-env AUROC |")
    print("|---|---|---|---|---|---|")
    for arm in arms:
        vals = [r["arms"][arm]["eval"] for r in rs if arm in r["arms"]]
        if not vals:
            continue
        def m(split, key, k2):
            xs = [v[split][key][k2] for v in vals if v[split][key][k2] is not None]
            return (np.mean(xs), np.std(xs)) if xs else (None, None)
        cells = []
        for split, key, k2 in (("test", "observed_horizons", "mean_brier"), ("test", "observed_horizons", "mean_auroc"),
                               ("test", "between_horizons", "mean_brier"), ("test_env", "observed_horizons", "mean_brier"),
                               ("test_env", "observed_horizons", "mean_auroc")):
            mu, sd = m(split, key, k2)
            cells.append(f"{f(mu)} ± {f(sd)}" if mu is not None else "—")
        print(f"| {arm} (n={len(vals)}) | " + " | ".join(cells) + " |")


def exp_c():
    rs = load("expC_c*_s*.json")
    if not rs:
        return
    print(f"\n### Experiment C ({len(rs)} corpora; mean Brier / AUROC)\n")
    sets = ["self_test", "third_test", "test_env"]
    print("| training data | " + " | ".join(sets) + " |")
    print("|---|" + "---|" * len(sets))
    for arm in ["marginal_km_self", "marginal_km_all", "self_only", "self_plus_third_matched", "third_only", "self_plus_third"]:
        row = []
        for s in sets:
            b = [r["arms"][arm]["eval"][s]["mean_brier"] for r in rs if arm in r["arms"] and s in r["arms"][arm]["eval"]]
            a = [r["arms"][arm]["eval"][s]["mean_auroc"] for r in rs if arm in r["arms"] and s in r["arms"][arm]["eval"]
                 and r["arms"][arm]["eval"][s]["mean_auroc"] is not None]
            row.append(f"{f(np.mean(b))} / {f(np.mean(a)) if a else '—'}" if b else "—")
        n = [r["arms"][arm].get("n_train") for r in rs if arm in r["arms"]]
        print(f"| {arm}" + (f" (n≈{int(np.mean([x for x in n if x]))})" if any(n) else "") + " | " + " | ".join(row) + " |")


def exp_d():
    rs = load("expD_c*_s*.json")
    if not rs:
        return
    print(f"\n### Experiment D ({len(rs)} corpora)\n")
    print("| selection | examples | strata | test Brier | test AUROC | unseen-env Brier |")
    print("|---|---|---|---|---|---|")
    for arm in ["marginal_km", "repetitive_volume", "uniform", "stratified", "novelty", "uncertainty", "all"]:
        v = [r["arms"][arm] for r in rs if arm in r["arms"]]
        if not v:
            continue
        print(f"| {arm} | {f(np.mean([x.get('n_train') or 0 for x in v]), 0)} | {f(np.mean([x.get('n_groups') or 0 for x in v]), 1)} | "
              f"{f(np.mean([x['eval']['test']['mean_brier'] for x in v]))} | "
              f"{f(np.mean([x['eval']['test']['mean_auroc'] or 0 for x in v]))} | "
              f"{f(np.mean([x['eval']['test_env']['mean_brier'] for x in v]))} |")


def exp_b():
    for r in load("expB_s*.json"):
        print(f"\n### Experiment B (seed {r['seed']}; marginal Brier {f(r['marginal_km']['mean_brier'])})\n")
        print("| scheduling | observation requests | exact / interval / censored labels | median detection latency (h) | "
              "test Brier | Brier gain per 1000 requests |")
        print("|---|---|---|---|---|---|")
        for m, v in r["modes"].items():
            k = v["label_kinds"]
            print(f"| {m} | {v['observation_requests']} | {k['exact']} / {k['interval']} / {k['right_censored']} | "
                  f"{f(v['detection_latency_h_median'], 2)} | {f(v['eval']['mean_brier'])} | {f(v['brier_gain_per_1000_requests'], 4)} |")


def exp_e():
    for r in load("expE_s*.json"):
        print(f"\n### Experiment E (lab seed {r['seed']}, {r['n_experiment']} randomized posts, "
              f"{r['n_observational']} observational)\n")
        ra = r["randomized"]["answered"]
        o = r["observational"]
        print("| estimate of the effect on P(answered within 24 h) | content (detailed+polite) | audience (30 vs 0 followers) | timing (19:00 vs 04:00) |")
        print("|---|---|---|---|")
        print(f"| randomized experiment | {f(ra['detailed']['effect'])} [{f(ra['detailed']['ci95'][0])}, {f(ra['detailed']['ci95'][1])}] | "
              f"{f(ra['high']['effect'])} [{f(ra['high']['ci95'][0])}, {f(ra['high']['ci95'][1])}] | "
              f"{f(ra['peak']['effect'])} [{f(ra['peak']['ci95'][0])}, {f(ra['peak']['ci95'][1])}] |")
        print(f"| naive observational | {f(o['naive_detail_and_polite']['effect'])} | {f(o['followers_effect_naive']['effect'])} (≥10 followers) | — |")
        print(f"| observational, follower-stratified | {f(o['adjusted_by_followers'])} | — | — |")
        for mname in ("model_obs", "model_nocov"):
            if mname in r:
                mm = r[mname]
                print(f"| world model ({mname}) | {f(mm['content']['mean'])} | {f(mm['audience']['mean'])} | {f(mm['timing']['mean'])} |")


def exp_f():
    rs = load("expF_s*.json")
    rs = [r for r in rs if r.get("seed", 0) < 50]
    if not rs:
        return
    arms = list(rs[0].get("eval", {}).keys())
    print(f"\n### Experiment F ({len(rs)} seeds; helpful answer within 24 h, environment-verified)\n")
    print("| agent | held-out problems | unseen spaces | all | right space | detail | shouting/clickbait | views |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in arms:
        v = [r["eval"][arm]["summary"] for r in rs if arm in r.get("eval", {})]
        def mm(split, k):
            return np.mean([x[split][k] for x in v])
        print(f"| {arm} | {f(mm('test', 'success'))} | {f(mm('test_env', 'success'))} | {f(mm('all', 'success'))} | "
              f"{f(mm('all', 'right_space'), 2)} | {f(mm('all', 'detail'), 2)} | {f(mm('all', 'shouting_or_clickbait'), 2)} | "
              f"{f(mm('all', 'views'), 1)} |")
    for r in rs:
        c = r["phases"]["rl"]["curve"]
        print(f"\nseed {r['seed']} RL matured reward by iteration: " +
              ", ".join(f(x["mean_matured_reward"], 2) for x in c if x["mean_matured_reward"] is not None))


if __name__ == "__main__":
    for fn in (exp_a, exp_b, exp_c, exp_d, exp_e, exp_f):
        fn()
