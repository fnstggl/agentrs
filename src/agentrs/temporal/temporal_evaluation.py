"""Held-out evaluation of temporal predictions, with censoring-aware ground truth.

For an event and a horizon h, an example is scored only when its outcome by h is known:
positive if the event happened at or before h; negative if it was known absent at a time >= h
(right-censored later than h, or it happened later than h). Pending outcomes before h are skipped,
never counted as negatives. Counts are scored at the observation times (cumulative = sum of
increments). Baselines: Kaplan–Meier marginal survival per event (a censoring-aware lookup
table), the training-set marginal count curve, and any other predict function with the same
signature.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from agentrs.data.temporal_transition import TemporalExample
from agentrs.evaluation.metrics import (
    auroc,
    bernoulli_nll,
    brier,
    expected_calibration_error,
)
from agentrs.temporal.time_encoding import describe_duration

# predict(examples, horizons) -> {"event": {name: array (N, Q)}, "count": {name: array (N, Q)}}
PredictFn = Callable[[Sequence[TemporalExample], Sequence[float]], dict]


def event_truth(ex: TemporalExample, name: str, h: float) -> int | None:
    lab = ex.events.get(name)
    if lab is None or lab.kind == "none":
        return None
    if lab.kind == "exact":
        return 1 if lab.b_s <= h + 1e-6 else 0
    if lab.kind == "interval":
        if lab.b_s <= h + 1e-6:
            return 1
        if lab.a_s >= h - 1e-6:
            return 0
        return None
    return 0 if lab.b_s >= h - 1e-6 else None


def count_truth(ex: TemporalExample, name: str) -> list[tuple[float, float]]:
    out, tot = [], 0.0
    for inc in sorted(ex.counts.get(name, []), key=lambda i: i.b_s):
        tot += inc.delta
        out.append((inc.b_s, tot))
    return out


def model_predict_fn(model, batch: int = 64) -> PredictFn:
    from agentrs.temporal.temporal_world_model import TemporalQuery

    def f(examples, horizons):
        preds = model.predict_batch([TemporalQuery.from_example(e) for e in examples], list(horizons), chunk=batch)
        spec = model.cfg.spec
        ev = {n: np.array([[hp.event_prob[n] for hp in p.horizons] for p in preds]) for n in spec.events}
        ct = {n: np.array([[hp.count_mean[n] for hp in p.horizons] for p in preds]) for n in spec.counts}
        sd = {n: np.array([[hp.event_std[n] for hp in p.horizons] for p in preds]) for n in spec.events}
        return {"event": ev, "count": ct, "event_std": sd}

    return f


def kaplan_meier(train: Sequence[TemporalExample], name: str) -> Callable[[float], float]:
    """Censoring-aware marginal P(event by h). Interval labels use their midpoint (log time)."""
    times = []
    for ex in train:
        lab = ex.events.get(name)
        if lab is None or lab.kind == "none":
            continue
        if lab.kind == "exact":
            times.append((lab.b_s, 1))
        elif lab.kind == "interval":
            t = math.expm1(0.5 * (math.log1p(lab.a_s) + math.log1p(lab.b_s)))
            times.append((t, 1))
        else:
            times.append((lab.b_s, 0))
    times.sort()
    surv, curve, at_risk, i = 1.0, [(0.0, 0.0)], len(times), 0
    while i < len(times):
        t = times[i][0]
        d = c = 0
        while i < len(times) and times[i][0] == t:
            d += times[i][1]
            c += 1 - times[i][1]
            i += 1
        if at_risk > 0 and d:
            surv *= 1.0 - d / at_risk
        curve.append((t, 1.0 - surv))
        at_risk -= d + c

    def F(h: float) -> float:
        v = 0.0
        for t, p in curve:
            if t <= h:
                v = p
            else:
                break
        return v

    return F


def marginal_predict_fn(train: Sequence[TemporalExample], events: Sequence[str], counts: Sequence[str]) -> PredictFn:
    kms = {n: kaplan_meier(train, n) for n in events}
    cpts = {n: sorted(p for ex in train for p in count_truth(ex, n)) for n in counts}

    def mean_count(n, h):
        pts = [v for t, v in cpts[n] if abs(math.log1p(t) - math.log1p(h)) < 0.35]
        return float(np.mean(pts)) if pts else 0.0

    def f(examples, horizons):
        N = len(examples)
        return {"event": {n: np.tile([kms[n](h) for h in horizons], (N, 1)) for n in events},
                "count": {n: np.tile([mean_count(n, h) for h in horizons], (N, 1)) for n in counts}}

    return f


def evaluate_temporal(predict: PredictFn, examples: Sequence[TemporalExample], horizons: Sequence[float],
                      events: Sequence[str], counts: Sequence[str] = ()) -> dict:
    """Per event and horizon: n, positives, Brier, NLL, AUROC, ECE; per count: NB-free MAE and
    log-MAE at observation times (interpolating predictions in log time)."""
    horizons = sorted(horizons)
    examples = list(examples)
    if not examples:
        return {}
    P = predict(examples, horizons)
    out: dict = {"events": {}, "counts": {}, "n_examples": len(examples)}
    agg_p, agg_y = [], []
    for name in events:
        per_h = {}
        for j, h in enumerate(horizons):
            ys, ps = [], []
            for i, ex in enumerate(examples):
                y = event_truth(ex, name, h)
                if y is not None:
                    ys.append(y)
                    ps.append(float(P["event"][name][i, j]))
            if not ys:
                continue
            agg_p += ps
            agg_y += ys
            per_h[describe_duration(h)] = {"n": len(ys), "pos": int(sum(ys)), "brier": brier(ps, ys),
                                           "nll": bernoulli_nll(ps, ys), "auroc": auroc(ps, ys),
                                           "ece": expected_calibration_error(ps, ys)}
        out["events"][name] = per_h
    if agg_y:
        out["overall"] = {"n": len(agg_y), "brier": brier(agg_p, agg_y), "nll": bernoulli_nll(agg_p, agg_y),
                          "auroc": auroc(agg_p, agg_y), "ece": expected_calibration_error(agg_p, agg_y)}
    lh = np.log1p(np.asarray(horizons))
    for name in counts:
        errs, lerrs = [], []
        for i, ex in enumerate(examples):
            for t, v in count_truth(ex, name):
                pred = float(np.interp(math.log1p(t), lh, P["count"][name][i]))
                errs.append(abs(pred - v))
                lerrs.append(abs(math.log1p(max(pred, 0)) - math.log1p(v)))
        if errs:
            out["counts"][name] = {"n": len(errs), "mae": float(np.mean(errs)), "log_mae": float(np.mean(lerrs))}
    return out


def summarize(res: dict, events: Sequence[str]) -> dict:
    """Mean Brier / AUROC over events and horizons (unweighted), for compact tables."""
    b, a = [], []
    for n in events:
        for v in res.get("events", {}).get(n, {}).values():
            b.append(v["brier"])
            if v["auroc"] is not None:
                a.append(v["auroc"])
    return {"mean_brier": float(np.mean(b)) if b else None, "mean_auroc": float(np.mean(a)) if a else None,
            "overall": res.get("overall"), "counts": res.get("counts")}
