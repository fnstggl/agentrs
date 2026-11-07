"""Delayed, verified rewards: maturity, increments, revisions and duration-aware discounting.

* ``event_by_horizon``: reward 1 if the objective event (e.g. ``resolved``) happened within ``h``
  of the action. It *matures* when the event is observed (value 1) or when an observation at or
  after ``h`` shows it did not happen (value 0). Until then it is ``pending`` and is not used; if
  observation stopped before ``h`` it is ``censored`` and is never used as a 0.
* ``count_stream``: an ongoing reward per new unit of a count (e.g. accepted answers). Rewards are
  the *increments* between observations, so the sum always equals the latest cumulative value
  (10 replies at 1 h and 20 at 24 h give 10 + 10, not 30). Decreases are stored as explicit
  ``revision`` records.
* ``semi_markov_return``: sum_i gamma ** ((t_i - t_0) / tau) * r_i — discounting by elapsed time,
  not by step count.

Every reward record keeps the policy checkpoint that generated the action, so policy learning can
apply the correct off-policy correction however late the reward arrives.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger
from agentrs.temporal.outcome_labels import OutcomeMapping, _mapped


class RewardSpec(BaseModel):
    kind: Literal["event_by_horizon", "count_stream"] = "event_by_horizon"
    objective: str = "resolved"
    horizon_s: float = 86400.0
    value: float = 1.0
    gamma: float = 1.0  # per tau seconds
    tau_s: float = 86400.0


class MaturedReward(BaseModel):
    status: Literal["pending", "matured", "censored"]
    value: float | None = None
    maturity_time: float | None = None
    increments: list[tuple[float, float]] = []  # (time, increment) for streams
    checkpoint: str | None = None


def mature_reward(ledger: DelayedOutcomeLedger, action_id: str, spec: RewardSpec, as_of: float,
                  mapping: OutcomeMapping | None = None, record: bool = True) -> MaturedReward:
    mapping = mapping or OutcomeMapping()
    a = ledger.action(action_id)
    if a is None:
        return MaturedReward(status="censored")
    t0, ck = a["action_time"], a["policy_checkpoint"]
    obs = [o for o in ledger.observations(action_id, observed_before=as_of) if o["completeness"] != "missing"]
    evs = ledger.events(action_id, observed_before=as_of)
    if spec.kind == "event_by_horizon":
        hit = None
        for e in evs:
            if _mapped(e, mapping) == spec.objective:
                t = e["env_event_time"] if e["env_event_time"] is not None else e["observed_time"]
                if t - t0 <= spec.horizon_s + 1e-6:
                    hit = e["observed_time"]
                break
        if hit is None:
            for o in obs:
                if (o["payload"].get("occurred") or {}).get(spec.objective):
                    at = (o["payload"].get("occurred_at") or {}).get(spec.objective, o["actual_time"])
                    if at - t0 <= spec.horizon_s + 1e-6:
                        hit = o["actual_time"]
                    break
        if hit is not None:
            res = MaturedReward(status="matured", value=spec.value, maturity_time=hit, checkpoint=ck)
        else:
            late = [o["actual_time"] for o in obs if o["elapsed"] >= spec.horizon_s - 1e-6]
            if late:
                res = MaturedReward(status="matured", value=0.0, maturity_time=min(late), checkpoint=ck)
            elif a["status"] in ("closed", "purged") and a["close_reason"] != "window":
                res = MaturedReward(status="censored", checkpoint=ck)
            else:
                res = MaturedReward(status="pending", checkpoint=ck)
        if record and res.status == "matured":
            ledger.record_reward(action_id, "delayed" if spec.horizon_s > 0 else "immediate", res.value,
                                 env_time=res.maturity_time, maturity_time=res.maturity_time, checkpoint=ck,
                                 objective=f"{spec.objective}@{spec.horizon_s:.0f}",
                                 idem=f"{action_id}:{spec.objective}@{spec.horizon_s:.0f}")
        return res
    # count stream
    incs, prev = [], 0.0
    for o in obs:
        v = (o["payload"].get("counts") or {}).get(spec.objective)
        if v is None:
            continue
        d = float(v) - prev
        if d != 0:
            incs.append((o["actual_time"], d * spec.value))
            if record:
                ledger.record_reward(action_id, "stream" if d > 0 else "revision", d * spec.value, env_time=o["actual_time"],
                                     maturity_time=o["actual_time"], checkpoint=ck, objective=spec.objective,
                                     idem=f"{action_id}:{spec.objective}:{o['obs_id']}")
        prev = float(v)
    return MaturedReward(status="matured" if obs else "pending", value=sum(d for _, d in incs),
                         maturity_time=obs[-1]["actual_time"] if obs else None, increments=incs, checkpoint=ck)


def semi_markov_return(rewards: Sequence[tuple[float, float]], t0: float, gamma: float, tau_s: float) -> float:
    return float(sum((gamma ** max(0.0, (t - t0) / tau_s)) * r for t, r in rewards))
