"""Private evaluation of hidden mechanisms. OPERATOR SIDE ONLY: nothing in a world code path imports this module,
and nothing here is reachable through a tool.

Everything works on any engine through the `Engine` contract, by simulation with common random numbers (the same
world seed in both arms of a comparison, so the noise cancels and what remains is the effect):

* `effect_matrix`           the true average effect of every lever on every gauge and on revenue, cost, net and
                            score, as a step response at several lags, with a standard error and a significance flag;
* `score_claims`            sign accuracy, lag accuracy, coverage and Brier score of filed hypotheses
                            (`rig.hypothesize`), and the error of their predictions;
* `prediction_error`        error of predicted mean outcomes of given settings;
* `discriminate`            log score of a posterior over candidate mechanisms;
* `oracle`                  the best constant settings found by a seeded cross-entropy search and, when the
                            mechanism has regimes, the best piecewise-constant schedule. It is used only to normalise
                            regret: it is a strong reference, not a proof of optimality;
* `regret`                  regret of a trajectory against that reference;
* `information_efficiency`  identification as a function of ticks of experience;
* `implied_signs`           the signs and lags that follow from `truth()` alone (used to test the simulation-based
                            measures against the declared structure).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .base import Engine, build_engine
from .sim import MechSim, ensure_engines

SCORE_TARGETS = ("revenue", "cost", "net", "score")
RIG = "eval"                        # facility id used for evaluation rollouts (any fixed string)


def _engine(mech: dict, engine: Engine | None = None) -> Engine:
    if engine is not None:
        return engine
    ensure_engines()
    return build_engine(mech)


def _targets(eng: Engine, seat: int) -> list[str]:
    return [g.name for g in eng.gauges(seat)] + list(SCORE_TARGETS)


def _row(res, seat: int, gauges: list[str]) -> list[float]:
    rd = res.readings[seat]
    return [rd[g] for g in gauges] + [res.revenue[seat], res.cost[seat], res.revenue[seat] - res.cost[seat],
                                      res.score[seat]]


def _u_to_settings(eng: Engine, seat: int, u: Sequence[float]) -> dict[str, float]:
    return {lv.name: lv.clip(lv.lo + float(x) * (lv.hi - lv.lo)) for lv, x in zip(eng.levers(seat), u)}


def _per_seat(eng: Engine, seat: int, settings: dict | None) -> list:
    return [settings if s == seat else None for s in range(eng.n_seats)]


def run_schedule(mech: dict, schedule: Sequence[dict], world_seed: int, ticks: int, seat: int = 0,
                 engine: Engine | None = None) -> dict:
    """Simulate `ticks` ticks from a fresh facility where the seat follows `schedule`
    (`[{"from": tick, "settings": {...}}, ...]`, other seats at defaults). Returns `MechSim.run` arrays."""
    sim = MechSim(mech, world_seed, RIG, engine=_engine(mech, engine))
    plan = {int(e["from"]): e["settings"] for e in schedule}
    n = sim.n_seats
    return sim.run(lambda t, last: [plan[t] if s == seat else None for s in range(n)] if t in plan else None, ticks)


def mean_net(mech: dict, schedule: Sequence[dict], seeds: Sequence[int], ticks: int, seat: int = 0,
             window: tuple[int, int] | None = None, engine: Engine | None = None) -> tuple[float, float]:
    """(mean, standard error over seeds) of the seat's net cents per tick under a schedule, over the ticks in
    `window` (default: all)."""
    engine = _engine(mech, engine)
    a, b = window or (0, ticks)
    vals = [float(run_schedule(mech, schedule, s, ticks, seat, engine)["net"][a:b, seat].mean()) for s in seeds]
    se = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return float(np.mean(vals)), se


# ====================================================================== effects
def effect_matrix(mech: dict, seat: int = 0, lags: Sequence[int] = (0, 1, 2, 4, 8), n_base: int = 16,
                  t0_range: tuple[int, int] = (8, 24), levels: tuple[float, float] = (0.25, 0.75), z: float = 3.0,
                  seed: int = 0, engine: Engine | None = None) -> dict:
    """True average effects by intervention.

    For each of `n_base` base conditions (the seat's other levers at uniformly random levels, a fresh world seed, a
    start tick drawn from `t0_range`) and each lever, two arms are run from the same state with the same random
    numbers: the lever stepped to a low level and to a high level (`levels`, as fractions of its range) and held.
    The effect at lag l is the mean over base conditions of (target in the high arm - target in the low arm) l
    ticks after the step, i.e. a step response averaged over the settings of the other levers.

    Returns a dict with `levers`, `targets` (gauge names then revenue, cost, net, score), `lags`,
    arrays `effect`, `se`, `significant` of shape (levers, targets, lags), and per pair: `sign` (-1, 0, 1: the
    sign of the most significant lag, 0 when no lag is significant), `first_lag` (first lag with a significant
    effect among 0..max(lags), -1 when none) and `scale` (per target: spread of the target across base conditions,
    for normalising errors). Regime mechanisms: choose `t0_range` inside the regime you want described.
    """
    eng = _engine(mech, engine)
    levers = eng.levers(seat)
    gauges = [g.name for g in eng.gauges(seat)]
    targets = gauges + list(SCORE_TARGETS)
    lmax = int(max(lags))
    nl, nt = len(levers), len(targets)
    diff = np.zeros((n_base, nl, nt, lmax + 1))
    base_rows = np.zeros((n_base, nt))
    for b in range(n_base):
        rng = np.random.default_rng([int(seed), 7, b])
        sim = MechSim(mech, 1_000_003 * (int(seed) + 1) + b, RIG, engine=eng)
        base = _u_to_settings(eng, seat, rng.random(nl))
        t0 = int(rng.integers(t0_range[0], max(t0_range[0] + 1, t0_range[1])))
        sim._rng.prefetch(0, t0 + lmax + 1)            # every arm below re-uses these ticks' generators
        sim.step(_per_seat(eng, seat, base))
        for _ in range(t0 - 1):
            sim.step(None)
        snap = sim.snapshot()
        for i, lv in enumerate(levers):
            arms = []
            for frac in levels:
                sim.restore(snap)
                r = sim.step(_per_seat(eng, seat, {lv.name: lv.lo + frac * (lv.hi - lv.lo)}))
                rows = [_row(r, seat, gauges)]
                for _ in range(lmax):
                    rows.append(_row(sim.step(None), seat, gauges))
                arms.append(np.asarray(rows))
            diff[b, i] = (arms[1] - arms[0]).T
            if i == 0:
                base_rows[b] = arms[0][-1]
    mean = diff.mean(axis=0)
    se = diff.std(axis=0, ddof=1) / math.sqrt(n_base) if n_base > 1 else np.zeros_like(mean)
    scale = np.maximum(base_rows.std(axis=0), 1e-9)
    sig = (np.abs(mean) > z * se) & (np.abs(mean) > 1e-9 * scale[None, :, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(sig, np.abs(mean) / np.maximum(se, 1e-300), 0.0)
    best = tstat.argmax(axis=2)
    any_sig = sig.any(axis=2)
    sign = np.where(any_sig, np.sign(np.take_along_axis(mean, best[..., None], axis=2)[..., 0]), 0).astype(int)
    first = np.where(any_sig, sig.argmax(axis=2), -1).astype(int)
    li = [int(x) for x in lags]
    return {"levers": [lv.name for lv in levers], "targets": targets, "lags": li, "effect": mean[:, :, li],
            "se": se[:, :, li], "significant": sig[:, :, li], "sign": sign, "first_lag": first, "scale": scale,
            "n_base": n_base, "seat": seat, "levels": list(levels), "t0_range": list(t0_range),
            "response": mean, "response_se": se}


def effect_rows(em: dict) -> list[dict]:
    """The effect matrix as one record per (lever, target)."""
    out = []
    for i, lv in enumerate(em["levers"]):
        for j, tg in enumerate(em["targets"]):
            out.append({"lever": lv, "target": tg, "sign": int(em["sign"][i, j]), "first_lag": int(em["first_lag"][i, j]),
                        "effect": {int(l): float(em["effect"][i, j, k]) for k, l in enumerate(em["lags"])},
                        "se": {int(l): float(em["se"][i, j, k]) for k, l in enumerate(em["lags"])}})
    return out


def true_claims(em: dict, with_lag: bool = True) -> dict:
    """Claims that state exactly what the effect matrix says (the reference a perfect analyst would file)."""
    eff = []
    for r in effect_rows(em):
        c = {"lever": r["lever"], "target": r["target"], "sign": r["sign"]}
        if with_lag and r["sign"] != 0:
            c["lag"] = r["first_lag"]
        eff.append(c)
    return {"effects": eff}


def implied_signs(truth: dict, seat: int = 0, max_len: int = 8) -> dict[tuple[str, str], tuple[int | None, int]]:
    """What the declared causal graph alone implies: for every lever of the seat and every gauge of the seat (by
    public name) plus "score" (the seat's outcome): (sign, shortest lag). The sign is the common sign of all
    directed simple paths, or None when a path contains a non-monotone or gated edge (sign 0) or paths disagree.
    Pairs with no directed path are returned with sign 0."""
    nodes = {n["id"]: n for n in truth["nodes"]}
    out_edges: dict[str, list[dict]] = {}
    for e in truth["edges"]:
        out_edges.setdefault(e["src"], []).append(e)
    res = {}
    targets = {n["id"]: n["public"] for n in truth["nodes"] if n["role"] == "gauge" and n["seat"] == seat}
    targets.update({n["id"]: "score" for n in truth["nodes"] if n["role"] == "outcome" and n["seat"] == seat})
    for lv in (n for n in truth["nodes"] if n["role"] == "lever" and n["seat"] == seat):
        found: dict[str, list[tuple[int, int]]] = {}
        stack = [(lv["id"], 1, 0, (lv["id"],))]
        while stack:
            node, sgn, lag, path = stack.pop()
            if node in targets and len(path) > 1:
                found.setdefault(node, []).append((sgn, lag))
            if len(path) > max_len:
                continue
            for e in out_edges.get(node, []):
                if e["dst"] not in path:
                    stack.append((e["dst"], sgn * int(e["sign"]), lag + int(e.get("lag", 0)), path + (e["dst"],)))
        for tid, name in targets.items():
            paths = found.get(tid, [])
            if not paths:
                res[(lv["public"], name)] = (0, 0)
            else:
                signs = {p[0] for p in paths}
                res[(lv["public"], name)] = (signs.pop() if len(signs) == 1 and 0 not in signs else None,
                                             min(p[1] for p in paths))
    assert all(nodes[t]["seat"] == seat for t in targets)
    return res


# ====================================================================== claims
def prediction_error(mech: dict, predictions: Sequence[dict], seat: int = 0, seeds: int = 3, ticks: int | None = None,
                     n_ref: int = 12, seed: int = 0, engine: Engine | None = None) -> dict:
    """How wrong predicted mean outcomes are. Each prediction `{"settings", "target", "value"}` is compared with
    the simulated mean of the target per tick under those constant settings (unmentioned levers at default) over
    the horizon after a burn-in of a quarter of it. `norm_err` divides the absolute error by the spread of that
    target's mean across `n_ref` uniformly random settings, so 1.0 is the error of someone who knows nothing."""
    eng = _engine(mech, engine)
    h = int(ticks or eng.horizon)
    burn = min(h // 4, 24)
    gauges = [g.name for g in eng.gauges(seat)]
    targets = gauges + list(SCORE_TARGETS)
    names = {lv.name for lv in eng.levers(seat)}

    def means(settings: dict, ws: int) -> np.ndarray:
        sim = MechSim(mech, ws, RIG, engine=eng)
        rows = [_row(sim.step(_per_seat(eng, seat, settings) if t == 0 else None), seat, gauges) for t in range(h)]
        return np.asarray(rows)[burn:].mean(axis=0)

    rng = np.random.default_rng([int(seed), 11])
    ref = np.asarray([means(_u_to_settings(eng, seat, rng.random(len(names))), 5_000_011 + k) for k in range(n_ref)])
    scale = np.maximum(ref.std(axis=0), 1e-9)
    items = []
    for p in predictions:
        if p.get("target") not in targets or any(k not in names for k in p.get("settings", {})):
            items.append({"target": p.get("target"), "valid": False})
            continue
        j = targets.index(p["target"])
        actual = float(np.mean([means(dict(p["settings"]), 6_000_017 + int(seed) * 31 + k)[j] for k in range(seeds)]))
        err = abs(float(p["value"]) - actual)
        items.append({"target": p["target"], "valid": True, "predicted": float(p["value"]), "actual": actual,
                      "abs_err": err, "norm_err": err / float(scale[j])})
    valid = [x for x in items if x["valid"]]
    return {"n": len(items), "n_valid": len(valid), "items": items,
            "mae": float(np.mean([x["abs_err"] for x in valid])) if valid else None,
            "nmae": float(np.mean([x["norm_err"] for x in valid])) if valid else None}


def score_claims(mech: dict, claims: dict, seat: int = 0, effects: dict | None = None, lag_tolerance: int = 0,
                 engine: Engine | None = None, **em_kw: Any) -> dict:
    """Score filed hypotheses against the truth measured by `effect_matrix` (pass `effects` to reuse one).

    * `sign_accuracy`   fraction of claimed (lever, target) pairs whose claimed sign equals the true sign;
    * `lag_accuracy`    among claims that give a lag for a pair with a true effect: fraction within
                        `lag_tolerance` ticks of the true first lag (`lag_mae`: mean absolute error in ticks);
    * `coverage`        fraction of the pairs that truly have an effect for which a non-zero sign was claimed;
    * `identification`  fraction of ALL pairs claimed with the correct sign (an unclaimed pair is not identified);
    * `brier`           mean (confidence - [sign correct])^2 over claims that give a confidence;
    * `prediction`      `prediction_error` of the claims' predictions.

    A later claim about the same pair replaces an earlier one. Claims naming unknown levers or targets are
    counted in `invalid` and otherwise ignored."""
    eng = _engine(mech, engine)
    em = effects if effects is not None else effect_matrix(mech, seat, engine=eng, **em_kw)
    li = {n: i for i, n in enumerate(em["levers"])}
    tj = {n: j for j, n in enumerate(em["targets"])}
    latest: dict[tuple[int, int], dict] = {}
    invalid = 0
    for c in claims.get("effects", []) or []:
        if c.get("lever") in li and c.get("target") in tj and c.get("sign") in (-1, 0, 1):
            latest[(li[c["lever"]], tj[c["target"]])] = c
        else:
            invalid += 1
    correct, lag_ok, lag_err, brier = [], [], [], []
    for (i, j), c in latest.items():
        ok = int(c["sign"]) == int(em["sign"][i, j])
        correct.append(ok)
        if c.get("lag") is not None and em["sign"][i, j] != 0:
            e = abs(int(c["lag"]) - int(em["first_lag"][i, j]))
            lag_err.append(e)
            lag_ok.append(e <= lag_tolerance)
        if c.get("confidence") is not None:
            brier.append((float(c["confidence"]) - float(ok)) ** 2)
    true_nz = {(i, j) for i in range(len(li)) for j in range(len(tj)) if em["sign"][i, j] != 0}
    n_pairs = len(li) * len(tj)
    preds = claims.get("predictions") or []
    return {"n_claims": len(latest), "invalid": invalid, "n_pairs": n_pairs, "n_true_effects": len(true_nz),
            "sign_accuracy": float(np.mean(correct)) if correct else None,
            "lag_accuracy": float(np.mean(lag_ok)) if lag_ok else None,
            "lag_mae": float(np.mean(lag_err)) if lag_err else None,
            "coverage": (sum(1 for k in true_nz if k in latest and int(latest[k]["sign"]) != 0) / len(true_nz))
            if true_nz else None,
            "identification": sum(correct) / n_pairs if n_pairs else None,
            "brier": float(np.mean(brier)) if brier else None,
            "prediction": prediction_error(mech, preds, seat, engine=eng) if preds else None}


def discriminate(candidates: Sequence[Any], posterior: Sequence[float], true_index: int) -> dict:
    """Log score of a posterior over candidate mechanisms, one of which (`true_index`) is the real one.
    `log_score` is ln p(true) after normalising the posterior (probabilities floored at 1e-12);
    `normalized` = 1 + ln p(true) / ln n is 1 for certainty on the truth, 0 for a uniform posterior and negative
    when worse than uniform."""
    n = len(candidates)
    p = np.asarray(posterior, dtype=float)
    if n < 2 or p.shape != (n,) or not 0 <= int(true_index) < n:
        raise ValueError("need >= 2 candidates, one probability per candidate, and a valid true_index")
    if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("posterior must be finite, non-negative and not all zero")
    p = np.maximum(p / p.sum(), 1e-12)
    p = p / p.sum()
    onehot = np.zeros(n)
    onehot[int(true_index)] = 1.0
    ls = float(np.log(p[int(true_index)]))
    return {"log_score": ls, "normalized": 1.0 + ls / math.log(n), "prob_true": float(p[int(true_index)]),
            "rank": int((p > p[int(true_index)]).sum()) + 1, "top1": bool(p.argmax() == int(true_index)),
            "brier": float(((p - onehot) ** 2).sum()), "n": n}


# ====================================================================== the reference policy
def _cem(objective, d: int, starts: list[np.ndarray], evals: int, rng: np.random.Generator) -> tuple[np.ndarray, float, int]:
    """Cross-entropy search over [0, 1]^d. Returns (best point, its value, evaluations used)."""
    best_u, best_v, used = None, -math.inf, 0
    seen = []
    for u in starts:
        v = objective(u)
        used += 1
        seen.append((v, u))
        if v > best_v:
            best_u, best_v = u, v
    pop = int(max(8, min(24, (evals - used) // 4)))
    elite = max(3, pop // 4)
    mu, sd = best_u.copy(), np.full(d, 0.35)
    while used + pop <= evals:
        xs = np.clip(mu + sd * rng.standard_normal((pop, d)), 0.0, 1.0)
        for u in xs:
            v = objective(u)
            used += 1
            seen.append((v, u))
            if v > best_v:
                best_u, best_v = u.copy(), v
        seen.sort(key=lambda x: -x[0])
        top = np.asarray([u for _, u in seen[:elite]])
        mu = top.mean(axis=0)
        sd = np.maximum(top.std(axis=0), 0.03)
        seen = seen[:elite]
    return best_u, best_v, used


def oracle(mech: dict, seat: int = 0, budget: int = 160, seeds: int = 2, seed: int = 0, ticks: int | None = None,
           piecewise: bool | None = None, eval_seeds: int = 4, engine: Engine | None = None) -> dict:
    """A strong reference policy for one seat (other seats at defaults), found by seeded search on the simulator.

    Stage 1: the best constant settings (cross-entropy search over the lever box, started from the defaults, both
    corners and, when the engine can state one, its own constructive optimum). Stage 2, when the mechanism declares
    regimes (or `piecewise=True`): for each segment between regime ticks in turn, the best settings for that
    segment given the earlier ones. `budget` is the total number of rollouts searched (each `ticks` long on
    `seeds` common world seeds). The winners are then re-evaluated on `eval_seeds` fresh seeds, so
    `expected_net` is not biased by the search.

    Returns `settings` (best constant), `net_constant`, `schedule` (`[{"from", "settings"}]`, one entry without
    regimes), `segments` (`[{"from", "to", "net"}]`: the schedule's net per tick inside each segment),
    `expected_net` (net cents per tick of the better of the two), `default_net`, `evals`, `ticks`.
    Used only to normalise regret."""
    eng = _engine(mech, engine)
    h = int(ticks or eng.horizon)
    levers = eng.levers(seat)
    d = len(levers)
    rng = np.random.default_rng([int(seed), 13, seat])
    search = [9_000_001 + 7919 * int(seed) + k for k in range(seeds)]
    fresh = [9_500_003 + 104_729 * int(seed) + k for k in range(eval_seeds)]
    cuts = [0]
    if piecewise is not False:
        try:
            cuts += sorted({int(r["at"]) for r in eng.truth().get("regimes", []) if 0 < int(r["at"]) < h})
        except Exception:                                       # noqa: BLE001 - an engine without regimes
            pass
    if piecewise is True and len(cuts) == 1:
        cuts.append(h // 2)
    bounds = list(zip(cuts, cuts[1:] + [h]))

    def to_u(settings: dict) -> np.ndarray:
        return np.asarray([(lv.clip(settings.get(lv.name, lv.default)) - lv.lo) / (lv.hi - lv.lo) for lv in levers])

    default_u = to_u(eng.defaults(seat))
    hints: list[list[np.ndarray]] = [[] for _ in bounds]
    con = getattr(eng, "constructive_optimum", None)
    if con is not None:
        try:
            for k, e in enumerate((con(seat) or {}).get("schedule", [])[:len(bounds)]):
                hints[k].append(to_u(e["settings"]))
        except Exception:                                       # noqa: BLE001
            pass
    all_hints = [u for hs in hints for u in hs]
    starts = [default_u, np.zeros(d), np.ones(d)] + all_hints
    evals = 0
    share = budget if len(bounds) == 1 else int(budget * 0.4)
    u_const, _, used = _cem(lambda u: mean_net(mech, [{"from": 0, "settings": _u_to_settings(eng, seat, u)}],
                                               search, h, seat, engine=eng)[0], d, starts, share, rng)
    evals += used
    const = _u_to_settings(eng, seat, u_const)
    schedule = [{"from": 0, "settings": const}]
    if len(bounds) > 1:
        per = max(12, (budget - evals) // len(bounds))
        schedule = []
        for k, (a, b) in enumerate(bounds):
            prev = list(schedule)

            def obj(u, a=a, b=b, prev=prev):
                sch = prev + [{"from": a, "settings": _u_to_settings(eng, seat, u)}]
                return mean_net(mech, sch, search, b, seat, window=(a, b), engine=eng)[0]
            u_k, _, used = _cem(obj, d, [u_const, default_u] + hints[k], per, rng)
            evals += used
            schedule.append({"from": a, "settings": _u_to_settings(eng, seat, u_k)})
    net_const, se_const = mean_net(mech, [{"from": 0, "settings": const}], fresh, h, seat, engine=eng)
    net_default, se_default = mean_net(mech, [{"from": 0, "settings": eng.defaults(seat)}], fresh, h, seat,
                                       engine=eng)
    if len(bounds) > 1:
        net_sched, se_sched = mean_net(mech, schedule, fresh, h, seat, engine=eng)
        if net_sched < net_const:                               # the schedule must never be worse than constant
            schedule, net_sched, se_sched = [{"from": 0, "settings": const}], net_const, se_const
    else:
        net_sched, se_sched = net_const, se_const
    full = [{"from": a, "settings": next(e["settings"] for e in reversed(schedule) if e["from"] <= a)} for a, _ in bounds]
    segments = [{"from": a, "to": b, "net": mean_net(mech, full, fresh, h, seat, window=(a, b), engine=eng)[0]}
                for a, b in bounds]
    return {"seat": seat, "settings": const, "net_constant": net_const, "schedule": full, "segments": segments,
            "expected_net": max(net_sched, net_const), "expected_net_se": se_sched, "default_net": net_default,
            "default_net_se": se_default, "evals": evals, "ticks": h, "seeds": seeds, "eval_seeds": eval_seeds}


def regret(mech: dict, trajectory: Sequence[Any], seat: int = 0, oracle_result: dict | None = None, t0: int = 0,
           engine: Engine | None = None, **oracle_kw: Any) -> dict:
    """Regret of what a seat actually earned against the reference policy.

    `trajectory` is one entry per tick starting at tick `t0`: a number (net cents) or a record with
    `net_cents`, or `revenue_cents` and `cost_cents` (a `rig.report`). The reference for tick t is the oracle
    schedule's mean net in the segment containing t. Returns the totals, `regret_total` = sum(reference - net),
    `regret_mean`, and `normalized` = regret divided by the regret of staying at defaults (0: as good as the
    reference, 1: no better than never touching a lever, above 1: worse than that)."""
    orc = oracle_result or oracle(mech, seat, engine=engine, **oracle_kw)
    nets = []
    for x in trajectory:
        if isinstance(x, dict):
            nets.append(float(x["net_cents"]) if "net_cents" in x else float(x["revenue_cents"]) - float(x["cost_cents"]))
        else:
            nets.append(float(x))
    ref = []
    for k in range(len(nets)):
        t = t0 + k
        seg = next((s for s in orc["segments"] if s["from"] <= t < s["to"]), orc["segments"][-1])
        ref.append(seg["net"])
    nets_a, ref_a = np.asarray(nets), np.asarray(ref)
    total = float((ref_a - nets_a).sum())
    base = float((ref_a - orc["default_net"]).sum())
    return {"ticks": len(nets), "net_total": float(nets_a.sum()), "net_mean": float(nets_a.mean()) if nets else 0.0,
            "reference_mean": float(ref_a.mean()) if nets else 0.0, "default_net": orc["default_net"],
            "regret_total": total, "regret_mean": total / len(nets) if nets else 0.0,
            "normalized": total / base if base > 0 else None}


def information_efficiency(mech: dict, checkpoints: Sequence[dict], seat: int = 0, effects: dict | None = None,
                           threshold: float = 0.8, engine: Engine | None = None, **em_kw: Any) -> dict:
    """Identification as a function of experience. `checkpoints` are `[{"ticks": n, "claims": {...}}, ...]`: what
    the learner believed after n ticks of experience (for a hosted facility: the filings of `rig.hypothesize`
    with their `t`). Each is scored with `score_claims`; the curve is `identification` (fraction of all
    lever-target pairs with the right sign) against ticks. Returns the curve, its normalised area (`auc`, the mean
    identification over the tick span, counting 0 before the first checkpoint), the first tick count at which
    identification reached `threshold` (None if never) and the final value."""
    eng = _engine(mech, engine)
    em = effects if effects is not None else effect_matrix(mech, seat, engine=eng, **em_kw)
    pts = sorted(checkpoints, key=lambda c: int(c["ticks"]))
    curve = []
    for c in pts:
        s = score_claims(mech, c["claims"], seat, effects=em, engine=eng)
        curve.append({"ticks": int(c["ticks"]), "identification": s["identification"] or 0.0,
                      "sign_accuracy": s["sign_accuracy"], "coverage": s["coverage"], "n_claims": s["n_claims"]})
    auc, reach = None, None
    if curve:
        span = curve[-1]["ticks"]
        area, prev_t, prev_v = 0.0, 0, 0.0
        for p in curve:                         # step function: a belief holds until the next checkpoint
            area += prev_v * (p["ticks"] - prev_t)
            prev_t, prev_v = p["ticks"], p["identification"]
        auc = area / span if span > 0 else curve[-1]["identification"]
        reach = next((p["ticks"] for p in curve if p["identification"] >= threshold), None)
    return {"curve": curve, "auc": auc, "ticks_to_threshold": reach, "threshold": threshold,
            "final": curve[-1]["identification"] if curve else None}
