"""Persistent spending, resource and rate accounting.

Usage is committed by ``PolicyService.record`` (actual cost reported by the executor); between a grant and its
record the estimate is *reserved* so that concurrent grants cannot jointly exceed a limit. A granted action whose
record never arrives is charged its estimate once its execution window has passed.

Limits are plain numbers per task: ``money`` (total), ``money_per_action``, ``money_per_day``, ``compute_s``,
``max_external_actions``, ``max_decisions``, ``duration_s`` and an absolute time window (live authorization
record). A limit of 0 means "not configured": in live mode that is a denial, never "unlimited".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .store import Store, uj

DAY_S = 86400.0
USAGE_KEYS = ("money", "compute_s", "external_actions", "decisions", "primitive_ops", "seconds")
#: limit name -> (usage key, policy id)
TOTALS = {"money": ("money", "L-MONEY-TOTAL"), "compute_s": ("compute_s", "L-COMPUTE"),
          "max_external_actions": ("external_actions", "L-EXTERNAL-ACTIONS"), "max_decisions": ("decisions", "L-DECISIONS")}
RESOURCE_PREFIX = "resource: "


def min_nonzero(*vals: Optional[float]) -> float:
    v = [float(x) for x in vals if x is not None and float(x) > 0]
    return min(v) if v else 0.0


class LimitLedger:
    def __init__(self, store: Store):
        self.store = store

    # -- committed usage -------------------------------------------------------------------------------------
    def usage(self, task_id: str) -> Dict[str, float]:
        out = {k: 0.0 for k in USAGE_KEYS}
        for r in self.store.q("SELECT k, v FROM usage WHERE task_id=?", (task_id,)):
            out[r["k"]] = r["v"]
        return out

    def day_usage(self, task_id: str, key: str, now: float) -> float:
        r = self.store.one("SELECT v FROM usage_daily WHERE task_id=? AND day=? AND k=?",
                           (task_id, int(now // DAY_S), key))
        return 0.0 if r is None else r["v"]

    def add(self, task_id: str, amounts: Dict[str, float], now: float) -> None:
        for k, v in amounts.items():
            if not v:
                continue
            self.store.x("INSERT INTO usage(task_id, k, v) VALUES(?,?,?) ON CONFLICT(task_id, k) DO UPDATE SET v=v+excluded.v",
                         (task_id, k, float(v)))
            self.store.x("INSERT INTO usage_daily(task_id, day, k, v) VALUES(?,?,?,?) "
                         "ON CONFLICT(task_id, day, k) DO UPDATE SET v=v+excluded.v",
                         (task_id, int(now // DAY_S), k, float(v)))

    # -- reservations (derived from live grants) ---------------------------------------------------------------
    def settle_stale(self, now: float) -> int:
        """Charge the estimate of granted actions that were started but never recorded, and drop unused grants
        that expired. Returns the number of grants charged."""
        n = 0
        with self.store.tx():
            self.store.x("UPDATE grants SET status='expired' WHERE status='issued' AND expires_at < ?", (now,))
            for r in self.store.q("SELECT nonce, task_id, reserve_json FROM grants WHERE status='used' AND exec_until < ?",
                                  (now,)):
                self.add(r["task_id"], uj(r["reserve_json"], {}), now)
                self.store.x("UPDATE grants SET status='unsettled', closed_at=? WHERE nonce=?", (now, r["nonce"]))
                n += 1
        return n

    def reserved(self, task_id: str, now: float) -> Dict[str, float]:
        out = {k: 0.0 for k in USAGE_KEYS}
        for r in self.store.q("SELECT reserve_json FROM grants WHERE task_id=? AND "
                              "((status='issued' AND expires_at >= ?) OR status='used')", (task_id, now)):
            for k, v in uj(r["reserve_json"], {}).items():
                out[k] = out.get(k, 0.0) + float(v)
        return out

    # -- checks ----------------------------------------------------------------------------------------------
    def check(self, task_id: str, limits: Dict[str, float], want: Dict[str, float], now: float, started_at: float,
              *, strict_unset: bool, window: Optional[Tuple[float, float]] = None) -> List[Tuple[str, str]]:
        """``[(policy_id, reason)]`` for every limit the request would exceed. ``want`` holds what the action
        needs (``money``, ``compute_s``, ``external_actions``, ``decisions``). With ``strict_unset`` a limit that
        is not configured counts as exceeded for any use of it."""
        self.settle_stale(now)
        used, res = self.usage(task_id), self.reserved(task_id, now)
        out: List[Tuple[str, str]] = []

        def over(name: str, key: str, pid: str, spent: float) -> None:
            lim, need = float(limits.get(name, 0.0)), float(want.get(key, 0.0))
            if lim <= 0:
                if strict_unset and need > 0:
                    out.append(("L-UNSET", f"{RESOURCE_PREFIX}limit {name!r} is not configured; nothing is assumed "
                                           f"to be unlimited"))
                return
            if spent + need > lim + 1e-9:
                state = "exhausted" if spent >= lim - 1e-9 else "would be exceeded"
                out.append((pid, f"{RESOURCE_PREFIX}{name} {state} (used {spent:g} incl. reserved, "
                                 f"requested {need:g}, limit {lim:g})"))

        for name, (key, pid) in TOTALS.items():
            over(name, key, pid, used[key] + res[key])
        need_money = float(want.get("money", 0.0))
        pa = float(limits.get("money_per_action", 0.0))
        if need_money > 0 and pa > 0 and need_money > pa + 1e-9:
            out.append(("L-MONEY-ACTION", f"{RESOURCE_PREFIX}money per action would be exceeded "
                                          f"(requested {need_money:g}, limit {pa:g})"))
        pd = float(limits.get("money_per_day", 0.0))
        if need_money > 0 and pd > 0:
            spent = self.day_usage(task_id, "money", now) + res["money"]
            if spent + need_money > pd + 1e-9:
                out.append(("L-MONEY-DAY", f"{RESOURCE_PREFIX}money per day would be exceeded (used {spent:g} "
                                           f"incl. reserved, requested {need_money:g}, limit {pd:g})"))
        dur = float(limits.get("duration_s", 0.0))
        if dur > 0 and now > started_at + dur:
            out.append(("L-DURATION", f"{RESOURCE_PREFIX}experiment duration exhausted "
                                      f"({now - started_at:.0f} s elapsed, limit {dur:g} s)"))
        elif dur <= 0 and strict_unset:
            out.append(("L-UNSET", f"{RESOURCE_PREFIX}limit 'duration_s' is not configured; nothing is assumed "
                                   f"to be unlimited"))
        if window is not None:
            nb, na = window
            if now < nb:
                out.append(("L-WINDOW", f"{RESOURCE_PREFIX}the authorized time window has not started"))
            elif na > 0 and now > na:
                out.append(("L-WINDOW", f"{RESOURCE_PREFIX}the authorized time window has ended"))
        return out

    # -- rate limits -----------------------------------------------------------------------------------------
    def rate_count(self, key: str, window_s: float, now: float) -> int:
        r = self.store.one("SELECT COUNT(*) AS n FROM rate_events WHERE k=? AND ts > ?", (key, now - window_s))
        return int(r["n"])

    def rate_oldest(self, key: str, window_s: float, now: float) -> float:
        r = self.store.one("SELECT MIN(ts) AS t FROM rate_events WHERE k=? AND ts > ?", (key, now - window_s))
        return float(r["t"]) if r is not None and r["t"] is not None else now

    def rate_hit(self, key: str, now: float) -> None:
        self.store.x("INSERT INTO rate_events(k, ts) VALUES(?, ?)", (key, now))

    def rate_gc(self, now: float, horizon_s: float = 8 * DAY_S) -> None:
        self.store.x("DELETE FROM rate_events WHERE ts < ?", (now - horizon_s,))

    # -- reporting -------------------------------------------------------------------------------------------
    def state(self, task_id: str, limits: Dict[str, float], now: float, started_at: float) -> Dict[str, Any]:
        used, res = self.usage(task_id), self.reserved(task_id, now)
        remaining, exhausted = {}, []
        for name, (key, _pid) in TOTALS.items():
            lim = float(limits.get(name, 0.0))
            if lim > 0:
                remaining[name] = max(0.0, lim - used[key] - res[key])
                if remaining[name] <= 1e-9:
                    exhausted.append(name)
        dur = float(limits.get("duration_s", 0.0))
        if dur > 0:
            remaining["duration_s"] = max(0.0, started_at + dur - now)
            if remaining["duration_s"] <= 0:
                exhausted.append("duration_s")
        pd = float(limits.get("money_per_day", 0.0))
        if pd > 0:
            remaining["money_per_day"] = max(0.0, pd - self.day_usage(task_id, "money", now) - res["money"])
        return {"limits": dict(limits), "used": used, "reserved": res, "remaining": remaining,
                "exhausted": exhausted, "started_at": started_at}
