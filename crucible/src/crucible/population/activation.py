"""Class 1 activation: whether an individual is active on a simulated day and how many sessions it takes, from
workstream E's fitted runtime decision models (crucible.calibration.runtime, contracts in
docs/v5/calibration/RUNTIME.md), plus the session length and the scheduling of sessions inside the waking day.

Decisions and their models (every one is recorded with `population.record_decisions` inside the population's own
session, so `population.check_provenance` and `calibration.inworld.audit` count them as organic):

* **active today** - `daily_return@1` (fitted `return_daily@1`, KuaiRand-Pure app activity, served K). Evaluated once
  per individual and simulated day, at its first activation of the day (its waking time or a triggered wake-up).
  Inputs from the individual's own recorded history (an 8-day ring of sessions and items per day): active days in
  the previous 7, days since the last active day (capped 14), items reached in the previous 7 days (the contract's
  feed exposures; `count_scale` is the runtime's declared factor, 1.0), sessions in the previous 7 days, active
  yesterday, weekend (epoch calendar: day 0 = Thursday 1970-01-01, the calendar E's fixtures use). Batch kind
  `pop.day_active`, column `active` (bool), read by `calibration.inworld.world_activity`.
* **sessions on an active day** - `extra_sessions@1` (fitted `session_rate@1`; the E3 review serves contender S, a
  constant rate, and the runtime says so in the provenance). sessions = 1 + the model's draw for the individual's
  observed history (extra sessions and active days observed so far). Batch kind `pop.day_sessions`, column
  `sessions` (int32).
* **when the sessions happen** (`pop.decision.schedule`, plausible): the planned sessions are spread uniformly over
  the rest of the individual's waking day (the next one is the minimum of the remaining uniform draws), at least
  30 simulated minutes apart (KuaiRand's session gap); a triggered wake-up (a message, a notification) starts a
  session at once and does not use up a planned one on an inactive day.
* **session length** (`pop.decision.session_length`, plausible): the minutes the session may take, drawn from the
  individual's own log-normal session-length distribution (median `session_min`, sd 0.6 in log space) and capped by
  the attention left today. The fitted `session_minutes@1` is NOT used: its minutes count plays of KuaiRand's 7,583
  pool videos only (median 0.06-0.2 min per session for typical inputs) and its contract declares no scale factor,
  so using it as the length of a whole session would make sessions last seconds (rule 2 of RUNTIME.md forbids
  rescaling it from the batch); a declared scale factor from workstream E would let it replace this rule.

Exposures. The contract's `feed_exposures_7d` counts items reached in the feed. The population counts the items an
individual reached in its sessions: what the activity reported (`note_items`) or, for an activity that reports
nothing, one item per `ITEM_MIN` minutes of attention it spent (a documented approximation).

Initial history (genesis, `pop.decision.genesis_history`, plausible): a world starts with individuals that already
have a 7-day history, drawn from each individual's latent daily activity propensity (segment-dependent Beta) and
extra-session rate (the served rate x a Gamma(2, 1/2) individual factor) - otherwise the fitted return model would
treat every individual as a dormant user (P(active) 0.31 for no history).
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

DAY = 86400
HIST = 8
ITEM_MIN = 0.5                 # minutes of attention per item reached, for activities that report no items
SESSION_GAP_S = 1800           # KuaiRand: a session ends after 30 idle minutes
SESSION_LEN_SD = 0.6
DAILY = "daily_return@1"
SESSIONS = "extra_sessions@1"
SCHEDULE_MODEL = ("pop.schedule", "1", "pop.decision.schedule")
LENGTH_MODEL = ("pop.session_length", "1", "pop.decision.session_length")
LEARN_MODEL = ("pop.habit", "1", "pop.decision.habit_learning")

COLUMNS = {
    "active": ("bool", (), False),            # today's daily_return@1 decision (same spec as E's in-world fixture)
    "sessions": ("int32", (), 0),             # sessions planned today (extra_sessions@1)
    "act_day": ("int32", (), -1),             # simulated day of the last daily decision
    "day_left": ("int16", (), 0),             # planned sessions left today
    "last_active_day": ("int32", (), -1),
    "hist_day": ("int32", (HIST,), -1),       # 8-day ring (slot = day % 8): which day the slot holds
    "hist_sess": ("uint8", (HIST,), 0),       # sessions that day
    "hist_exp": ("uint16", (HIST,), 0),     # items reached that day
    "extra_obs": ("int32", (), 0),            # sum over observed active days of (sessions - 1)
    "days_obs": ("int32", (), 0),             # observed active days
    "sl_log_sum": ("float32", (), 0.0),       # sum of log1p(minutes) of sessions on earlier days
    "sl_n": ("int32", (), 0),                 # sessions on earlier days
    "sl_exp": ("float32", (), 0.0),         # items reached in sessions on earlier days
    "td_log_sum": ("float32", (), 0.0),       # the same for the last day with sessions (rolled at the day decision)
    "td_n": ("int16", (), 0),
    "td_exp": ("float32", (), 0.0),
    "session_budget_min": ("float32", (), 0.0),   # minutes the current session may take
    "session_end_at": ("int64", (), -1),
}


def weekend(day: int) -> float:
    return float(((int(day) + 4) % 7) >= 5)


def month(day: int) -> int:
    return (_dt.date(1970, 1, 1) + _dt.timedelta(days=int(day))).month


def _runtime():
    from ..calibration import runtime
    return runtime


def window(hist_day: np.ndarray, today: int, days: int = 7) -> np.ndarray:
    """bool[n, HIST]: ring slots holding one of the `days` days before `today`."""
    return (hist_day >= today - days) & (hist_day <= today - 1)


def history_stats(world, idx: np.ndarray, today: int) -> dict[str, np.ndarray]:
    from ..substrate import columns
    c = lambda name: columns.col(world, "pop", name)[idx]  # noqa: E731
    hd = c("hist_day").astype(np.int64)
    hs = c("hist_sess").astype(np.float64)
    hi = c("hist_exp").astype(np.float64)
    win = window(hd, today)
    act = win & (hs > 0)
    last = c("last_active_day").astype(np.int64)
    since = np.where(last >= 0, np.minimum(14, today - last), 14).astype(np.float64)
    yday = ((hd == today - 1) & (hs > 0)).any(axis=1)
    return {"active_days_7d": act.sum(axis=1).astype(np.float64), "days_since_last_active": np.maximum(since, 0.0),
            "feed_exposures_7d": (hi * win).sum(axis=1), "sessions_7d": (hs * win).sum(axis=1),
            "active_yesterday": yday.astype(np.float64)}


def daily_inputs(world, idx: np.ndarray, today: int) -> dict[str, np.ndarray]:
    """Inputs of daily_return@1 for individuals `idx` on simulated day `today` (row-wise, own history only)."""
    h = history_stats(world, idx, today)
    h["weekend"] = np.full(idx.size, weekend(today))
    return h


def decide_day(ctx, idx: np.ndarray, today: int, rng) -> tuple[np.ndarray, np.ndarray, dict, dict | None]:
    """The day's decisions of individuals `idx` (first activation today): (active bool, planned sessions int,
    daily_return provenance, extra_sessions provenance or None)."""
    rt = _runtime()
    dm = rt.get(DAILY)
    res = dm.evaluate(daily_inputs(ctx.world, idx, today))
    active = rng.random(idx.size) < res.values
    planned = np.zeros(idx.size, np.int64)
    pv2 = None
    a = np.flatnonzero(active)
    if a.size:
        from ..substrate import columns
        ses = rt.get(SESSIONS)
        inp = {"extra_sessions_observed": columns.col(ctx.world, "pop", "extra_obs")[idx[a]].astype(np.float64),
               "active_days_observed": columns.col(ctx.world, "pop", "days_obs")[idx[a]].astype(np.float64)}
        extra = np.asarray(ses.sample(inp, rng), np.int64)
        planned[a] = np.clip(1 + extra, 1, 24)
        r2 = ses.evaluate(inp)          # provenance of the served contender (the draw above evaluated it too)
        pv2 = r2.provenance()
    return active, planned, res.provenance(), pv2


def waking_left_s(now: int, day_start_min: np.ndarray, day_len_min: np.ndarray) -> np.ndarray:
    """Seconds of the current waking period left at `now` (0 when asleep)."""
    ds = np.asarray(day_start_min, np.int64) * 60
    dl = np.asarray(day_len_min, np.int64) * 60
    since = (now % DAY - ds) % DAY
    return np.where(since < dl, dl - since, 0).astype(np.float64)


def next_wake_s(now: int, day_start_min: np.ndarray, day_len_min: np.ndarray, rng) -> np.ndarray:
    """Absolute time of the start of the individual's next waking period (+ up to 45 min), int64."""
    ds = np.asarray(day_start_min, np.int64) * 60
    since = (now % DAY - ds) % DAY
    return (now + (DAY - since) + rng.integers(0, 45 * 60, since.shape[0])).astype(np.int64)


def next_session_gap_s(left_s: np.ndarray, remaining: np.ndarray, rng) -> np.ndarray:
    """Gap to the next of `remaining` sessions spread uniformly over `left_s` seconds: the minimum of `remaining`
    uniform draws (inf where nothing remains)."""
    r = np.maximum(np.asarray(remaining, np.float64), 1.0)
    u = rng.random(r.shape[0])
    g = np.asarray(left_s, np.float64) * (1.0 - u ** (1.0 / r))
    return np.where(np.asarray(remaining) > 0, g, np.inf)


def session_budget(rng, session_min: np.ndarray, left_min: np.ndarray) -> np.ndarray:
    """Minutes a session may take: the individual's log-normal session length, capped by the attention left."""
    m = np.exp(np.log(np.maximum(np.asarray(session_min, np.float64), 0.5)) +
               SESSION_LEN_SD * rng.standard_normal(np.asarray(session_min).shape[0]))
    return np.clip(np.minimum(m, np.asarray(left_min, np.float64)), 0.0, 600.0)


def ring_update(hist_day, hist_sess, hist_exp, rows: np.ndarray, today: int, items: np.ndarray):
    """Add one session with `items` items to today's ring slot of `rows` (in-place on copies the caller owns)."""
    s = today % HIST
    fresh = hist_day[rows, s] != today
    hist_sess[rows[fresh], s] = 0
    hist_exp[rows[fresh], s] = 0
    hist_day[rows, s] = today
    hist_sess[rows, s] = np.minimum(hist_sess[rows, s].astype(np.int64) + 1, 255)
    hist_exp[rows, s] = np.minimum(hist_exp[rows, s].astype(np.int64) + np.asarray(items, np.int64), 65535)


def genesis_history(rng, seg: np.ndarray, session_min: np.ndarray, d0: int) -> dict[str, np.ndarray]:
    """A 7-day history before day `d0` for new individuals (see the module docstring)."""
    m = seg.shape[0]
    heavy = (seg == 1) | (seg == 3)
    q = np.where(heavy, rng.beta(4.0, 1.2, m), rng.beta(1.6, 1.4, m))
    rate = float(_runtime().get(SESSIONS).spec.params.get("served_rate", 1.0))
    lam = rate * rng.gamma(2.0, 0.5, m)
    hd = np.full((m, HIST), -1, np.int32)
    hs = np.zeros((m, HIST), np.uint8)
    hi = np.zeros((m, HIST), np.uint16)
    extra = np.zeros(m, np.int64)
    days = np.zeros(m, np.int64)
    sl_n = np.zeros(m, np.int64)
    sl_log = np.zeros(m, np.float64)
    sl_exp = np.zeros(m, np.float64)
    last = np.full(m, -1, np.int64)
    for k in range(1, 8):
        d = d0 - k
        act = rng.random(m) < q
        s = np.where(act, 1 + rng.poisson(lam), 0)
        mins = np.exp(np.log(np.maximum(session_min, 0.5))[:, None] + SESSION_LEN_SD *
                      rng.standard_normal((m, 3))).mean(axis=1) * 0.5
        it = np.where(act, np.maximum(1, np.round(s * mins / ITEM_MIN)), 0)
        slot = d % HIST
        hd[:, slot] = d
        hs[:, slot] = np.minimum(s, 255)
        hi[:, slot] = np.minimum(it, 65535)
        extra += np.where(act, s - 1, 0)
        days += act
        sl_n += s
        sl_log += s * np.log1p(mins)
        sl_exp += it
        last = np.where(act & (last < 0), d, last)
    return {"hist_day": hd, "hist_sess": hs, "hist_exp": hi, "extra_obs": extra.astype(np.int32),
            "days_obs": days.astype(np.int32), "sl_n": sl_n.astype(np.int32), "sl_log_sum": sl_log.astype(np.float32),
            "sl_exp": sl_exp.astype(np.float32), "last_active_day": last.astype(np.int32)}


def provenance_args(pv: dict) -> dict:
    """record_decisions keyword arguments from a runtime Result.provenance()."""
    return {"model_id": str(pv["model"]).split("@")[0], "model_version": str(pv["version"]),
            "fidelity_id": str(pv["fidelity"])}


def served_extra(pv: dict) -> dict:
    return {"served": pv.get("served"), "extrapolated": int(pv.get("extrapolated", 0)), "rows": int(pv.get("rows", 0))}


def items_from_minutes(minutes: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(minutes, np.float64) / ITEM_MIN).astype(np.int64)


def log1p_minutes(minutes: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(minutes, np.float64), 0.0))


__all__ = ["COLUMNS", "DAILY", "SESSIONS", "decide_day", "daily_inputs", "history_stats", "genesis_history",
           "waking_left_s", "next_wake_s", "next_session_gap_s", "session_budget", "ring_update", "weekend", "month"]
