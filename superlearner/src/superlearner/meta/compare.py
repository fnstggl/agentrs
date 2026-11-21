"""Trusted comparison of learning runs. PROTECTED.

Everything here is computed from the env hub's own results file (one line per finished episode, written by the
trusted process, stamped with the number of training decisions served so far). Nothing a candidate process printed
or wrote is used for a score.

Measures of one run (`run_measures`):
    aulc_eval     area under the held-out evaluation success curve over the experience budget, normalised to [0, 1]
                  (the primary sample-efficiency measure)
    aulc_train    mean verified success of training episodes over ten equal slices of the experience budget
    final_eval    success at the last evaluation round;  best_eval the best round
    retention / transfer are measured by separate evaluation lists (see experiments)
    experience    training decisions and episodes actually served

Comparison of two algorithms (`compare`): difference of means with a bootstrap confidence interval over seeds
(paired by seed when both sides ran the same seeds) and the bootstrap probability that the difference is positive.
`pareto` marks which candidates are not dominated on several measures at once; nothing is collapsed into one score.
"""
from __future__ import annotations

import json
import os

import numpy as np


def load_trusted(path: str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def run_measures(recs: list[dict], budget: int) -> dict:
    train = [r for r in recs if r["kind"] == "train"]
    evals = [r for r in recs if r["kind"] == "eval"]
    rounds: dict[int, list] = {}
    for r in evals:
        rounds.setdefault(int(r["stamp"]), []).append(float(r["summary"]["success"]))
    xs = sorted(rounds)
    ys = [float(np.mean(rounds[x])) for x in xs]
    curve = [{"decisions": x, "success": round(y, 4), "n": len(rounds[x])} for x, y in zip(xs, ys)]
    aulc = 0.0
    if xs:
        px = [min(x, budget) for x in xs]
        py = list(ys)
        if px[0] > 0:
            px, py = [0] + px, [py[0]] + py
        if px[-1] < budget:
            px, py = px + [budget], py + [py[-1]]
        px, py = np.asarray(px, float), np.asarray(py, float)
        aulc = float(np.sum((py[1:] + py[:-1]) * 0.5 * np.diff(px)) / max(1, budget))
    bins = [[] for _ in range(10)]
    for r in train:
        i = min(9, int(10 * max(0, r["stamp"] - 1) / max(1, budget)))
        bins[i].append(float(r["summary"]["success"]))
    filled = [float(np.mean(b)) for b in bins if b]
    calls = sum(r["summary"]["commands"] for r in train)
    return {"aulc_eval": round(aulc, 5), "aulc_train": round(float(np.mean(filled)), 5) if filled else 0.0,
            "final_eval": ys[-1] if ys else 0.0, "best_eval": max(ys) if ys else 0.0, "first_eval": ys[0] if ys else 0.0,
            "eval_rounds": len(xs), "eval_curve": curve, "train_episodes": len(train),
            "train_decisions": int(sum(r["summary"]["decisions"] for r in train)),
            "train_valid_rate": (sum(r["summary"]["commands_ok"] for r in train) / calls) if calls else 0.0,
            "train_success_late": float(np.mean(bins[-1] or bins[-2] or [0.0])),
            "sim_wall_s": round(float(sum(r["summary"].get("sim_wall_s", 0) for r in recs)), 1)}


def compare(cand: dict[int, float], base: dict[int, float], n_boot: int = 4000, seed: int = 0) -> dict:
    """`cand`, `base`: seed -> measure. Bootstrap over seeds; paired when the seed sets are equal."""
    rng = np.random.default_rng(seed)
    paired = sorted(cand) == sorted(base) and len(cand) > 0
    c = np.array([cand[s] for s in sorted(cand)], float)
    b = np.array([base[s] for s in sorted(base)], float)
    if len(c) == 0 or len(b) == 0:
        return {"n_cand": len(c), "n_base": len(b), "diff": 0.0, "ci": [0.0, 0.0], "p_improve": 0.5, "paired": False}
    if paired:
        d = c - b
        boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1) if len(d) > 1 else np.full(n_boot, d[0])
    else:
        boots = (c[rng.integers(0, len(c), size=(n_boot, len(c)))].mean(1)
                 - b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_cand": int(len(c)), "n_base": int(len(b)), "mean_cand": float(c.mean()), "mean_base": float(b.mean()),
            "diff": float(c.mean() - b.mean()), "ci": [float(lo), float(hi)], "p_improve": float((boots > 0).mean()),
            "paired": bool(paired), "established": bool(len(c) >= 3 and (lo > 0 or hi < 0))}


def pareto(points: dict[str, dict], maximise: tuple = ("aulc_eval", "final_eval"), minimise: tuple = ("wall_s",)) -> dict:
    """name -> True if no other point is at least as good on every measure and better on one."""
    def vec(p):
        return [p.get(k, 0.0) for k in maximise] + [-p.get(k, 0.0) for k in minimise]
    out = {}
    for a, pa in points.items():
        va = vec(pa)
        dominated = False
        for b, pb in points.items():
            if a == b:
                continue
            vb = vec(pb)
            if all(y >= x for x, y in zip(va, vb)) and any(y > x for x, y in zip(va, vb)):
                dominated = True
                break
        out[a] = not dominated
    return out


def verdict(cmp_primary: dict, cmp_final: dict, margin: float = 0.0, max_final_drop: float = 0.05) -> str:
    """Predeclared decision rule. ``improved``: the primary difference is positive with a 95% interval above
    `margin` on at least three seeds and final success did not drop by more than `max_final_drop`.
    ``harmful``: the interval is entirely below zero. Otherwise ``inconclusive``."""
    if cmp_primary.get("n_cand", 0) >= 3 and cmp_primary["ci"][0] > margin and cmp_final.get("diff", 0.0) >= -max_final_drop:
        return "improved"
    if cmp_primary.get("n_cand", 0) >= 3 and cmp_primary["ci"][1] < 0:
        return "harmful"
    return "inconclusive"
