"""Validity checks for generated mechanisms (operator side; engine-agnostic: only the `Engine` contract is used).

`check_mechanism(mech)` runs seeded simulations and reports, each with its evidence:

    purity           two independently built engines agree bit for bit; a step repeated from the same state gives
                     the same result; a step never mutates its input state
    bounded          state, readings, revenue, cost and score stay finite and bounded under random settings,
                     both corners and settings that flip between the corners every tick
    conservation     revenue is never negative and never exceeds the demand income; cost is never negative
    exploits         no setting has a negative cost; revenue does not grow without bound at extreme settings
    consequences     at least two levers have a significant effect on net (interventions that move a lever across
                     nearly its whole range, common random numbers)
    discoverability  at least two levers are detectable on net from ONE single-lever experiment of three arms
                     (low, middle, high; `ARM_TICKS` ticks each, everything else at defaults) at a
                     signal-to-noise ratio of `SNR_MIN`, the noise being measured across independent replicates
    difficulty       the reference policy beats defaults and beats uniformly random settings by margins, and
                     neither corner (everything max / everything min) comes close to it
    dynamism         (mechanisms that declare a lasting regime change) the settings that are best before the
                     switch are clearly worse after it than the settings that are best after it; changes an
                     engine marks `transient` are hazards, not shifts, and are exempt
    claims           properties the engine itself declares (`engine.claims`): `default_zero` (defaults earn about
                     nothing), `random_loses` (random settings lose money on average), `corners_bad`

Solvability is a separate verdict, `"yes" | "unknown" | "no"`, combining independent signals: the engine's own
constructive optimum where it can state one, a bounded search (the evaluator's oracle), and two scripted baseline
operators that use only the public interface. Any positive signal gives "yes". "no" needs positive evidence that
there is nothing to find (no lever moves net at all); a mechanism is never rejected merely because the scripted
operators failed on it, and "unknown" does not reject.

`sample_valid(family, seed)` redraws deterministically until a draw passes and records every rejected draw with
its reasons; `acceptance(families, n)` gives the acceptance rate per family.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any

import numpy as np

from ..core.canonical import digest
from . import evaluator as E
from .base import MECH_FAMILIES, Engine, build_engine, make_mechanism
from .sim import MechSim, TickRng, ensure_engines, suggest_income

STATE_BOUND = 1e9
ARM_TICKS = 12                 # ticks per arm of a single-lever experiment
REPLICATES = 6                 # independent replicates used to measure the experiment's noise
SNR_MIN = 2.0
M_DEFAULT = 0.02               # reference - default >= M_DEFAULT x demand income per seat
M_RANDOM = 0.04                # reference - random  >= M_RANDOM  x demand income per seat
CORNER_MAX = 0.7               # (corner - default) <= CORNER_MAX x (reference - default)
ZERO_TOL = 0.2                 # |default net| <= ZERO_TOL x (reference - default) (+ 3 standard errors)
RIG = "valid"


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _settings(eng: Engine, kind: str, rng: np.random.Generator | None = None) -> list[dict[str, float]]:
    out = []
    for s in range(eng.n_seats):
        lv = eng.levers(s)
        if kind == "default":
            out.append(eng.defaults(s))
        elif kind == "hi":
            out.append({x.name: x.clip(x.hi) for x in lv})
        elif kind == "lo":
            out.append({x.name: x.clip(x.lo) for x in lv})
        else:
            out.append({x.name: x.clip(rng.uniform(x.lo, x.hi)) for x in lv})
    return out


def _raw_run(eng: Engine, world_seed: int, policy, ticks: int) -> dict:
    """Step the engine directly (no settlement) and record what it reports before any rounding or clamping."""
    rngf = TickRng(world_seed, RIG)
    state = eng.init_state(rngf("init"))
    out = {"finite": True, "max_abs": 0.0, "min_revenue": math.inf, "min_cost": math.inf, "max_total_revenue": 0.0,
           "revenue": [], "ticks": ticks}
    for t in range(ticks):
        res = eng.step(state, policy(t), rngf(t), t)
        state = res.state
        for a in state.values():
            a = np.asarray(a)
            if a.dtype.kind in "fc" and not np.isfinite(a).all():
                out["finite"] = False
            elif a.size and a.dtype.kind in "fiu":
                out["max_abs"] = max(out["max_abs"], float(np.abs(a).max()))
        vals = list(res.revenue) + list(res.cost) + list(res.score) + [v for rd in res.readings for v in rd.values()]
        if not all(_finite(v) for v in vals):
            out["finite"] = False
            break
        out["min_revenue"] = min(out["min_revenue"], min(res.revenue))
        out["min_cost"] = min(out["min_cost"], min(res.cost))
        tot = float(sum(res.revenue))
        out["max_total_revenue"] = max(out["max_total_revenue"], tot)
        out["revenue"].append(tot)
    return out


def _purity(mech: dict, seed: int, ticks: int = 40) -> dict:
    a, b = build_engine(mech), build_engine(mech)
    ra, rb = TickRng(seed, RIG), TickRng(seed, RIG)
    sa, sb = a.init_state(ra("init")), b.init_state(rb("init"))
    rng = np.random.default_rng([seed, 3])
    same = all(np.array_equal(sa[k], sb[k]) for k in sa) and sorted(sa) == sorted(sb)
    repeat = mutated = True
    for t in range(ticks):
        x = _settings(a, "random", rng) if t % 2 else _settings(a, "default")
        before = {k: np.array(v, copy=True) for k, v in sa.items()}
        r1 = a.step(sa, x, ra(t), t)
        mutated = mutated and all(np.array_equal(before[k], sa[k]) for k in before)
        r1b = a.step(sa, x, ra(t), t)                      # again from the same state: no hidden memory
        r2 = b.step(sb, x, rb(t), t)
        for u, v in ((r1, r1b), (r1, r2)):
            eq = (sorted(u.state) == sorted(v.state)
                  and all(np.asarray(u.state[k]).tobytes() == np.asarray(v.state[k]).tobytes() for k in u.state)
                  and u.readings == v.readings and list(u.revenue) == list(v.revenue)
                  and list(u.cost) == list(v.cost) and list(u.score) == list(v.score))
            if v is r1b:
                repeat = repeat and eq
            else:
                same = same and eq
        sa, sb = r1.state, r2.state
    return {"ok": bool(same and repeat and mutated), "two_engines_identical": bool(same),
            "repeat_identical": bool(repeat), "input_not_mutated": bool(mutated), "ticks": ticks}


def _nets(mech: dict, eng: Engine, settings: dict, seeds: list[int], ticks: int, seat: int,
          window: tuple[int, int] | None = None) -> np.ndarray:
    a, b = window or (0, ticks)
    return np.asarray([float(E.run_schedule(mech, [{"from": 0, "settings": settings}], s, ticks, seat, eng)
                             ["net"][a:b, seat].mean()) for s in seeds])


def _experiments(mech: dict, eng: Engine, seat: int, seed: int) -> dict[str, float]:
    """Signal-to-noise of one single-lever experiment per lever on net: arms low, middle, high in sequence."""
    out = {}
    for lv in eng.levers(seat):
        d_lin, d_curv = [], []
        for r in range(REPLICATES):
            sim = MechSim(mech, 31_000_003 + 977 * seed + r, RIG, engine=eng)
            means = []
            for frac in (0.0, 0.5, 1.0):
                x = E._per_seat(eng, seat, {lv.name: lv.lo + frac * (lv.hi - lv.lo)})
                net = 0.0
                for k in range(ARM_TICKS):
                    res = sim.step(x if k == 0 else None)
                    net += res.revenue[seat] - res.cost[seat]
                means.append(net / ARM_TICKS)
            d_lin.append(means[2] - means[0])
            d_curv.append(means[1] - 0.5 * (means[0] + means[2]))
        snr = 0.0
        for d in (d_lin, d_curv):
            m, s = abs(float(np.mean(d))), float(np.std(d, ddof=1))
            snr = max(snr, m / s if s > 0 else (math.inf if m > 0 else 0.0))
        out[lv.name] = snr
    return out


def _baseline_explore_commit(mech: dict, eng: Engine, seat: int, world_seed: int, ticks: int) -> tuple[float, int]:
    """A scripted operator: try a handful of random settings (and the defaults) for a few ticks each, then keep
    the one that earned most. Returns (mean net per tick while committed, first committed tick)."""
    levers = eng.levers(seat)
    rng = np.random.default_rng([world_seed, 17])
    k = max(4, ticks // 12)
    hold = max(3, ticks // (3 * k))
    cands = [eng.defaults(seat)] + [{lv.name: lv.clip(rng.uniform(lv.lo, lv.hi)) for lv in levers} for _ in range(k - 1)]
    sim = MechSim(mech, world_seed, RIG, engine=eng)
    scores = []
    for c in cands:
        tot = 0.0
        for j in range(hold):
            r = sim.step(E._per_seat(eng, seat, c) if j == 0 else None)
            tot += r.revenue[seat] - r.cost[seat]
        scores.append(tot / hold)
    start = sim.t
    best = cands[int(np.argmax(scores))]
    tot = 0.0
    for j in range(ticks - start):
        r = sim.step(E._per_seat(eng, seat, best) if j == 0 else None)
        tot += r.revenue[seat] - r.cost[seat]
    return tot / max(1, ticks - start), start


def _baseline_coordinate(mech: dict, eng: Engine, seat: int, world_seed: int, ticks: int) -> tuple[float, int]:
    """A scripted operator: one lever at a time, try its low and high end for a few ticks, keep the better of
    (current, low, high); then stay. Returns (mean net per tick while committed, first committed tick)."""
    levers = eng.levers(seat)
    hold = max(2, (ticks // 2) // (3 * max(1, len(levers))))
    sim = MechSim(mech, world_seed, RIG, engine=eng)
    cur = eng.defaults(seat)

    def trial(x: dict) -> float:
        tot = 0.0
        for j in range(hold):
            r = sim.step(E._per_seat(eng, seat, x) if j == 0 else None)
            tot += r.revenue[seat] - r.cost[seat]
        return tot / hold

    for lv in levers:
        if sim.t + 3 * hold > ticks // 2 + hold:
            break
        opts = [dict(cur), dict(cur, **{lv.name: lv.clip(lv.lo)}), dict(cur, **{lv.name: lv.clip(lv.hi)})]
        vals = [trial(o) for o in opts]
        cur = opts[int(np.argmax(vals))]
    start = sim.t
    tot = 0.0
    for j in range(ticks - start):
        r = sim.step(E._per_seat(eng, seat, cur) if j == 0 else None)
        tot += r.revenue[seat] - r.cost[seat]
    return tot / max(1, ticks - start), start


def check_mechanism(mech: dict, n_seeds: int = 6, budget: int = 96, seed: int = 0, n_random: int = 160) -> dict:
    """Check one mechanism. `n_seeds` world seeds per estimate, `budget` rollouts for the reference search,
    `n_random` uniformly random constant settings for the random baseline (the spread of net across settings is
    of the order of the default revenue, so a powered estimate of its mean needs many). Deterministic in its
    arguments."""
    t_start = time.process_time()
    ensure_engines()
    eng = build_engine(mech)
    h = eng.horizon
    n_seats = eng.n_seats
    checks: dict[str, dict] = {}
    income = mech["config"].get("income_cents_per_tick")
    declared = income is not None
    if not declared:
        income = suggest_income(mech)
    scale = float(income) / n_seats
    seeds = [41_000_009 + 7919 * seed + k for k in range(n_seeds)]

    # ---- purity
    checks["purity"] = _purity(mech, 43 + seed)

    # ---- bounded state, conservation, exploits (raw engine output)
    rng = np.random.default_rng([seed, 5])
    hi, lo = _settings(eng, "hi"), _settings(eng, "lo")
    runs = {"random": _raw_run(eng, 47 + seed, lambda t: _settings(eng, "random", rng), 2 * h),
            "hi": _raw_run(eng, 53 + seed, lambda t: hi, 2 * h),
            "lo": _raw_run(eng, 59 + seed, lambda t: lo, 2 * h),
            "flip": _raw_run(eng, 61 + seed, lambda t: hi if t % 2 else lo, 2 * h),
            "default": _raw_run(eng, 67 + seed, lambda t: _settings(eng, "default"), h)}
    finite = all(r["finite"] for r in runs.values())
    max_abs = max(r["max_abs"] for r in runs.values())
    checks["bounded"] = {"ok": bool(finite and max_abs <= STATE_BOUND), "finite": bool(finite),
                         "max_abs_state": max_abs, "ticks": sum(r["ticks"] for r in runs.values())}
    min_rev = min(r["min_revenue"] for r in runs.values())
    min_cost = min(r["min_cost"] for r in runs.values())
    max_rev = max(r["max_total_revenue"] for r in runs.values())
    checks["conservation"] = {"ok": bool(finite and min_rev >= 0 and min_cost >= 0 and max_rev <= income),
                              "min_revenue": min_rev, "min_cost": min_cost, "max_total_revenue": max_rev,
                              "income_cents_per_tick": int(income), "income_declared": declared}
    growth = {}
    for k in ("hi", "flip", "random"):
        rev = np.asarray(runs[k]["revenue"]) if runs[k]["revenue"] else np.zeros(4)
        q = max(1, len(rev) // 4)
        growth[k] = float(rev[-q:].mean() - rev[:q].mean())
    checks["exploits"] = {"ok": bool(finite and min_cost >= 0 and max_rev <= income
                                     and all(g <= 0.5 * income for g in growth.values())),
                          "negative_cost": bool(min_cost < 0), "revenue_above_income": bool(max_rev > income),
                          "revenue_growth_late_minus_early": growth}

    # ---- per seat: economics, effects, experiments, solvability
    try:
        declared = [r for r in eng.truth().get("regimes", []) if 0 < int(r["at"]) < h]
    except Exception:                                           # noqa: BLE001
        declared = []
    # the optimum must move only for changes that stay: a disturbance that is undone later (marked `transient`
    # by the engine) is a hazard to survive, and constant robust settings may be the best response to it
    regimes = sorted({int(r["at"]) for r in declared if not r.get("transient")})
    transient_only = bool(declared) and not regimes
    claims = tuple(getattr(eng, "claims", ()) or ())
    econ, per_seat = [], []
    for seat in range(n_seats):
        levers = eng.levers(seat)
        dflt = eng.defaults(seat)
        orc = E.oracle(mech, seat, budget=budget, seeds=1, seed=seed, engine=eng)
        ref = [{"from": e["from"], "settings": e["settings"]} for e in orc["schedule"]]
        ref_nets = np.asarray([float(E.run_schedule(mech, ref, s, h, seat, eng)["net"][:, seat].mean()) for s in seeds])
        if orc["net_constant"] >= orc["expected_net"] - 1e-9:
            ref_nets = _nets(mech, eng, orc["settings"], seeds, h, seat)
        d_nets = _nets(mech, eng, dflt, seeds, h, seat)
        hi_nets = _nets(mech, eng, {lv.name: lv.clip(lv.hi) for lv in levers}, seeds, h, seat)
        lo_nets = _nets(mech, eng, {lv.name: lv.clip(lv.lo) for lv in levers}, seeds, h, seat)
        rr = np.random.default_rng([seed, 19, seat])
        h_r = h if regimes else max(32, h // 2)        # many settings matter more than long runs of each
        r_nets = np.asarray([_nets(mech, eng, {lv.name: lv.clip(rr.uniform(lv.lo, lv.hi)) for lv in levers},
                                   [seeds[k % n_seeds]], h_r, seat)[0] for k in range(n_random)])

        def se(x: np.ndarray) -> float:
            return float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0

        gap = ref_nets - d_nets                                   # paired: same world seeds
        gap_m, gap_se = float(gap.mean()), se(gap)
        r_m, r_se = float(r_nets.mean()), se(r_nets)
        c_hi, c_lo = float((hi_nets - d_nets).mean()), float((lo_nets - d_nets).mean())
        e = {"seat": seat, "default_net": float(d_nets.mean()), "default_net_se": se(d_nets),
             "reference_net": float(ref_nets.mean()), "random_net": r_m, "random_net_se": r_se,
             "corner_max_net": float(hi_nets.mean()), "corner_min_net": float(lo_nets.mean()),
             "gap_reference_default": gap_m, "gap_se": gap_se, "scale": scale, "oracle_evals": orc["evals"],
             "reference": ref}
        econ.append(e)
        corners_ok = max(c_hi, c_lo) <= CORNER_MAX * gap_m
        diff = {"ok": bool(gap_m >= max(M_DEFAULT * scale, 3 * gap_se)
                           and float(ref_nets.mean()) - r_m >= max(M_RANDOM * scale, 3 * r_se) and corners_ok),
                "reference_minus_default": gap_m, "reference_minus_random": float(ref_nets.mean()) - r_m,
                "corner_max_minus_default": c_hi, "corner_min_minus_default": c_lo, "corners_ok": bool(corners_ok),
                "margins": {"default": M_DEFAULT * scale, "random": M_RANDOM * scale, "corner_fraction": CORNER_MAX}}
        cl = {}
        if "default_zero" in claims:
            tol = ZERO_TOL * max(gap_m, 0.0) + 3 * se(d_nets)
            cl["default_zero"] = {"ok": bool(abs(float(d_nets.mean())) <= tol), "default_net": float(d_nets.mean()),
                                  "tolerance": tol}
        if "random_loses" in claims:
            cl["random_loses"] = {"ok": bool(r_m + 2 * r_se < 0), "random_net": r_m, "se": r_se, "n": n_random}
        if "corners_bad" in claims:
            cl["corners_bad"] = {"ok": bool(corners_ok)}
        # consequences: interventions
        em = E.effect_matrix(mech, seat, n_base=12, levels=(0.05, 0.95), seed=seed, engine=eng)
        j = em["targets"].index("net")
        moved = [lv for i, lv in enumerate(em["levers"]) if em["sign"][i, j] != 0 or em["significant"][i, j].any()]
        cons = {"ok": len(moved) >= 2, "levers_with_effect_on_net": moved, "n_levers": len(levers)}
        snr = _experiments(mech, eng, seat, seed)
        det = sorted(k for k, v in snr.items() if v >= SNR_MIN)
        disc = {"ok": len(det) >= 2, "detectable": det, "snr": {k: (None if math.isinf(v) else round(v, 3))
                                                               for k, v in snr.items()},
                "arms": 3, "ticks_per_arm": ARM_TICKS, "replicates": REPLICATES, "snr_required": SNR_MIN}
        dyn = None
        if regimes:
            cut = regimes[0]
            if len(orc["schedule"]) > 1:
                before, after = orc["schedule"][0]["settings"], orc["schedule"][1]["settings"]
                a_after = _nets(mech, eng, before, seeds, h, seat, (cut, h))
                b_after = _nets(mech, eng, after, seeds, h, seat, (cut, h))
                a_before = _nets(mech, eng, before, seeds, h, seat, (0, cut))
                b_before = _nets(mech, eng, after, seeds, h, seat, (0, cut))
                g_after, g_before = float((b_after - a_after).mean()), float((a_before - b_before).mean())
                dyn = {"ok": bool(g_after >= max(M_DEFAULT * scale, 3 * se(b_after - a_after))), "switch_at": cut,
                       "after_minus_before_settings_after_switch": g_after,
                       "before_minus_after_settings_before_switch": g_before}
            else:
                dyn = {"ok": False, "switch_at": cut, "why": "the search found no schedule better than constant settings"}
        # solvability: independent signals
        margin = max(M_DEFAULT * scale, 3 * gap_se)
        ev: dict[str, Any] = {"margin": margin, "search": {"gap": gap_m, "positive": bool(gap_m >= margin)}}
        con = getattr(eng, "constructive_optimum", None)
        ev["constructive"] = None
        if con is not None:
            try:
                c = con(seat)
            except Exception:                                   # noqa: BLE001
                c = None
            if c and c.get("schedule"):
                c_nets = np.asarray([float(E.run_schedule(mech, c["schedule"], s, h, seat, eng)["net"][:, seat].mean())
                                     for s in seeds])
                g = float((c_nets - d_nets).mean())
                ev["constructive"] = {"gap": g, "positive": bool(g >= margin)}
        base = {}
        for name, fn in (("explore_commit", _baseline_explore_commit), ("coordinate", _baseline_coordinate)):
            gains = []
            for s in seeds[:3]:
                net, start = fn(mech, eng, seat, s, h)
                ref_d = float(E.run_schedule(mech, [{"from": 0, "settings": dflt}], s, h, seat, eng)
                              ["net"][start:, seat].mean()) if start < h else 0.0
                gains.append(net - ref_d)
            base[name] = {"gain_over_default": float(np.mean(gains)), "positive": bool(np.mean(gains) >= margin)}
        ev["baselines"] = base
        signals = [ev["search"]["positive"], bool(ev["constructive"] and ev["constructive"]["positive"])] + \
                  [b["positive"] for b in base.values()]
        verdict = "yes" if any(signals) else ("no" if not moved else "unknown")
        per_seat.append({"difficulty": diff, "claims": cl, "consequences": cons, "discoverability": disc,
                         "dynamism": dyn, "solvable": verdict, "evidence": ev})

    def merge(name: str) -> dict:
        parts = [p[name] for p in per_seat]
        return dict(parts[0], ok=all(p["ok"] for p in parts), seats=parts) if n_seats > 1 else parts[0]

    for name in ("consequences", "discoverability", "difficulty"):
        checks[name] = merge(name)
    if regimes:
        checks["dynamism"] = merge("dynamism")
    elif transient_only:
        checks["dynamism"] = {"ok": True, "applies": False,
                              "why": "every declared change is transient; the optimum is not required to move"}
    if claims:
        names = sorted(per_seat[0]["claims"])
        checks["claims"] = {"ok": all(p["claims"][n]["ok"] for p in per_seat for n in names), "declared": list(claims),
                            **{n: (per_seat[0]["claims"][n] if n_seats == 1 else [p["claims"][n] for p in per_seat])
                               for n in names}}
    order = {"no": 0, "unknown": 1, "yes": 2}
    solvable = min((p["solvable"] for p in per_seat), key=lambda v: order[v])
    reasons = [k for k, v in checks.items() if not v["ok"]]
    if solvable == "no":
        reasons.append("unsolvable")
    return {"engine": mech.get("engine"), "family": mech.get("family"), "seed": mech.get("seed"), "ok": not reasons,
            "reasons": reasons, "checks": checks, "solvable": solvable,
            "evidence": per_seat[0]["evidence"] if n_seats == 1 else [p["evidence"] for p in per_seat],
            "economics": econ, "horizon": h, "n_seeds": n_seeds, "cpu_s": round(time.process_time() - t_start, 2)}


def redraw_seed(family: str, seed: int, attempt: int) -> int:
    """The mechanism seed of the k-th draw for (family, seed): the seed itself first, then a fixed derived series."""
    return int(seed) if attempt == 0 else int(digest(["mech-redraw", family, int(seed), int(attempt)])[:8], 16)


def sample_valid(family: str, seed: int, max_tries: int = 8, overrides: dict | None = None, **check_kw: Any) -> dict:
    """Draw mechanisms of a family until one passes `check_mechanism`. Deterministic: the k-th candidate is
    `make_mechanism(family, redraw_seed(family, seed, k), overrides)`. Returns `{"mech", "seed", "tries",
    "rejected": [{"seed", "reasons", "solvable"}], "report"}`; `mech` is None if `max_tries` draws all failed."""
    ensure_engines()
    rejected = []
    for k in range(max_tries):
        s = redraw_seed(family, seed, k)
        mech = make_mechanism(family, s, overrides)
        rep = check_mechanism(mech, **check_kw)
        if rep["ok"]:
            return {"mech": mech, "seed": s, "tries": k + 1, "rejected": rejected, "report": rep}
        rejected.append({"seed": s, "reasons": rep["reasons"], "solvable": rep["solvable"]})
    return {"mech": None, "seed": None, "tries": max_tries, "rejected": rejected, "report": None}


def acceptance(families: list[str] | None = None, n: int = 30, first_seed: int = 0, **check_kw: Any) -> dict:
    """Acceptance rate of first draws per family over seeds first_seed .. first_seed + n - 1, with the reasons
    of the rejections and the solvability verdicts."""
    ensure_engines()
    out = {}
    for fam in families or sorted(MECH_FAMILIES):
        ok, why, solv, cpu = 0, Counter(), Counter(), 0.0
        for s in range(first_seed, first_seed + n):
            rep = check_mechanism(make_mechanism(fam, s), **check_kw)
            ok += rep["ok"]
            why.update(rep["reasons"])
            solv[rep["solvable"]] += 1
            cpu += rep["cpu_s"]
        out[fam] = {"n": n, "accepted": ok, "rate": ok / n, "reasons": dict(why), "solvable": dict(solv),
                    "cpu_s_per_mechanism": round(cpu / n, 2)}
    return out


if __name__ == "__main__":                                     # python -m crucible.mechanisms.validity [n] [family ...]
    import json
    import sys
    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    fams = sys.argv[2:] or None
    ensure_engines()
    for fam_name in fams or sorted(MECH_FAMILIES):
        print(json.dumps({fam_name: acceptance([fam_name], n_arg)[fam_name]}), flush=True)
