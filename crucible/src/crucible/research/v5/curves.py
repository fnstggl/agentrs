"""Learning curves, censored N_q and learning acceleration (MASTER_SPEC Part 15.3; U.6, SP-15.3, SP-15.3b).

A *run* is one learner (one learner seed, possibly pre-trained) practising one unfamiliar task (a world: family,
seed). The curve records the task's verified, normalised score before any adaptation (n = 0) and after exactly
n = 1, 2, 4, 8, 16, 32, 64 informative interactions (`SCHEDULE`), then after longer budgets (`LONG`). An
informative interaction is an executed action that changed the world (counted operator side by the adapter); the
driver grants practice episodes on the task until the next budget is reached exactly (setting changes beyond it
are not executed), then scores the learner in exploit mode on the same task (frozen: no update, no practice
counted).

N_q is the first budget at which a run's score reaches the competence threshold q. A run that never reaches it is
right-censored at its largest budget - never given a finite value. Across runs:

* `kaplan_meier` - the survival function of N_q (probability of not yet being competent) on the budget grid, with
  censoring; its median (None when more than half the runs are censored) and the restricted mean
  RMST(tau) = sum S(t_i) (t_{i+1} - t_i) up to the largest budget tau (always defined, censoring-aware);
* `acceleration` - N_q(untrained) / N_q(experienced) as a ratio of Kaplan-Meier medians when both are finite, a
  *lower bound* tau / median(experienced) when only the untrained arm is censored, *undefined* when the experienced
  arm is censored; plus the RMST ratio with a crossed bootstrap interval (tasks x learner seeds).

Efficiency quantities recorded per run (SP-15.3b): performance before adaptation and on every budget, time to
competence (interactions and simulated seconds), total action cost, wall-clock and CPU compute, data usage (ticks
and observation bytes seen), retention (score on a previously practised task after practising another) and
transfer (score at n = 0 after training elsewhere versus untrained), and sample efficiency (area under the curve).
"""
from __future__ import annotations

import math
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from .distribution import Registry
from .episodes import references, normalised, run_episode
from .sampler import build_env

SCHEDULE = (0, 1, 2, 4, 8, 16, 32, 64)
LONG = (128, 256)
MAX_EPISODES = 400


def _score(reg: Registry, task: Mapping, learner, rng) -> tuple[float, dict]:
    e = reg[task["entry"]]
    env = build_env(e, int(task["seed"]), dict(task.get("difficulty") or {}), {"mode": "assigned"})
    was = learner.frozen
    learner.frozen = True
    rec = run_episode(env, learner, learn=False, rng=rng)
    learner.frozen = was
    refs = references(e, int(task["seed"]), task.get("difficulty"))
    return normalised(rec["value"], refs), rec


def practice_curve(reg: Registry, task: Mapping, learner, *, budgets: Sequence[int] = SCHEDULE + LONG,
                   rng: np.random.Generator, q: float | None = None) -> dict:
    """Practise `learner` on `task` ({entry, seed, difficulty?}) and score it at every budget of informative
    interactions. Returns the run record (curve, N_q if `q` given, accounting)."""
    e = reg[task["entry"]]
    budgets = sorted(set(int(b) for b in budgets))
    t0, c0 = time.perf_counter(), time.process_time()
    used, episodes, ticks, cost_s, obs_bytes = 0, 0, 0, 0, 0
    curve = []
    for b in budgets:
        stalled = 0
        while used < b and episodes < MAX_EPISODES:
            env = build_env(e, int(task["seed"]), dict(task.get("difficulty") or {}), {"mode": "assigned"})
            rec = run_episode(env, learner, learn=True, rng=rng, budget=b - used)
            used += rec["interactions"]
            episodes += 1
            ticks += rec["ticks"]
            cost_s += rec["action_cost_s"]
            obs_bytes += rec["ticks"] * 64 * rec["n_levers"]
            stalled = stalled + 1 if rec["interactions"] == 0 else 0
            if stalled >= 3:                  # the learner stopped acting: it gets no further budget
                break
        s, _ = _score(reg, task, learner, rng)
        curve.append({"budget": b, "interactions_used": int(used), "score": float(s), "episodes": episodes,
                      "ticks": ticks, "action_cost_s": cost_s})
    out = {"entry": task["entry"], "seed": int(task["seed"]), "curve": curve, "practice_episodes": episodes,
           "ticks": ticks, "action_cost_s": cost_s, "obs_bytes_est": obs_bytes,
           "cpu_s": time.process_time() - c0, "wall_s": time.perf_counter() - t0,
           "auc": auc(curve)}
    if q is not None:
        out.update(n_q(curve, q))
    return out


def n_q(curve: Sequence[Mapping], q: float) -> dict:
    """First budget whose score reaches q; censored (None) at the largest budget otherwise."""
    for p in curve:
        if p["score"] >= q:
            return {"q": q, "N_q": int(p["budget"]), "censored": False, "tau": int(curve[-1]["budget"])}
    return {"q": q, "N_q": None, "censored": True, "tau": int(curve[-1]["budget"])}


def auc(curve: Sequence[Mapping]) -> float:
    """Mean score over the budget points (log-spaced grid: equal weight per doubling)."""
    return float(np.mean([p["score"] for p in curve])) if curve else 0.0


def kaplan_meier(runs: Sequence[Mapping], grid: Sequence[int] | None = None) -> dict:
    """Survival of 'not yet competent' over budgets; runs are {N_q, censored, tau}."""
    if not runs:
        raise ValueError("no runs")
    times = sorted({int(r["N_q"]) for r in runs if not r["censored"]} | {int(r["tau"]) for r in runs})
    grid = sorted(set(grid or times))
    S, s = {}, 1.0
    at_risk_events = []
    for t in sorted(set(times) | set(grid)):
        n_risk = sum(1 for r in runs if (r["N_q"] if not r["censored"] else r["tau"]) >= t and
                     (r["censored"] is False or r["tau"] >= t))
        d = sum(1 for r in runs if not r["censored"] and int(r["N_q"]) == t)
        if n_risk > 0 and d > 0:
            s *= 1.0 - d / n_risk
        at_risk_events.append((t, n_risk, d))
        S[t] = s
    med = next((t for t in sorted(S) if S[t] <= 0.5), None)
    tau = max(int(r["tau"]) for r in runs)
    pts = sorted(t for t in S if t <= tau)
    rm, prev_t, prev_s = 0.0, 0, 1.0
    for t in pts:
        rm += prev_s * (t - prev_t)
        prev_t, prev_s = t, S[t]
    rm += prev_s * (tau - prev_t)
    return {"survival": {str(t): float(S[t]) for t in sorted(S)}, "median": med, "tau": tau, "rmst": float(rm),
            "runs": len(runs), "censored": sum(1 for r in runs if r["censored"]),
            "reached": sum(1 for r in runs if not r["censored"])}


def acceleration(untrained: Sequence[Mapping], experienced: Sequence[Mapping], *, n_boot: int = 1000,
                 seed: int = 0) -> dict:
    """N_q(untrained) / N_q(experienced), censoring-aware (module doc). Runs carry 'task' and 'learner_seed'."""
    ku, ke = kaplan_meier(untrained), kaplan_meier(experienced)
    tau = max(ku["tau"], ke["tau"])
    mu, me = ku["median"], ke["median"]
    if me is None:
        med = {"finite": False, "kind": "undefined", "why": "experienced arm's median N_q is censored"}
    elif mu is None:
        med = {"finite": False, "kind": "lower_bound", "lower_bound": tau / max(me, 1),
               "why": "untrained arm's median N_q is censored at tau"}
    elif me == 0:
        med = {"finite": False, "kind": "lower_bound", "lower_bound": float(max(mu, 1)),
               "why": "experienced arm is competent before any interaction"}
    else:
        med = {"finite": True, "kind": "ratio", "value": mu / me}
    rng = np.random.default_rng(seed)
    tasks = sorted({r["task"] for r in list(untrained) + list(experienced)})
    vals = []
    for _ in range(n_boot):
        pick = [tasks[i] for i in rng.integers(len(tasks), size=len(tasks))]
        U = [r for t in pick for r in untrained if r["task"] == t]
        X = [r for t in pick for r in experienced if r["task"] == t]
        if not U or not X:
            continue
        su = [U[i] for i in rng.integers(len(U), size=len(U))]
        sx = [X[i] for i in rng.integers(len(X), size=len(X))]
        a, b = kaplan_meier(su)["rmst"], kaplan_meier(sx)["rmst"]
        if b > 0:
            vals.append(a / b)
    rr = ku["rmst"] / ke["rmst"] if ke["rmst"] > 0 else None
    lo, hi = (np.quantile(vals, [0.025, 0.975]) if vals else (None, None))
    return {"median_ratio": med, "rmst_untrained": ku["rmst"], "rmst_experienced": ke["rmst"], "rmst_ratio": rr,
            "rmst_ratio_ci": [None if lo is None else float(lo), None if hi is None else float(hi)],
            "km_untrained": ku, "km_experienced": ke, "tau": tau}


def efficiency_record(run: Mapping, *, q: float, tick_s: int = 3600, retention: float | None = None,
                      transfer: float | None = None) -> dict:
    """The Part 15.3 quantities of one run."""
    nq = n_q(run["curve"], q)
    c0 = run["curve"][0]["score"]
    reached_at = nq["N_q"]
    return {"performance_before_adaptation": c0,
            "performance_by_budget": {str(p["budget"]): p["score"] for p in run["curve"]},
            "time_to_competence_interactions": reached_at, "time_to_competence_censored": nq["censored"],
            "time_to_competence_sim_s": None if reached_at is None else
            next(p["ticks"] for p in run["curve"] if p["budget"] == reached_at) * tick_s,
            "total_action_cost_s": run["action_cost_s"], "wall_clock_s": run["wall_s"], "cpu_s": run["cpu_s"],
            "data_ticks": run["ticks"], "data_obs_bytes_est": run["obs_bytes_est"], "retention": retention,
            "transfer": transfer, "sample_efficiency_auc": run["auc"]}
