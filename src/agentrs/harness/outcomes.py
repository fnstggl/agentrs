"""Temporal follow-up of real actions, using the existing delayed-outcome ledger and the durable
observation scheduler (``agentrs.temporal``).

Every executed mutating action becomes a ``self`` action in the ledger, with the forecast the agent
made *before* executing it, and is followed up on its horizon profile (a post: 1 min … 7 d; a pull
request: creation, CI, reviews, merge). Webhook / stream events are attached as follow-up events
with their provider timestamps (exact event times); polling is used only where no event source
exists, and always through the same account, connector and rate limits. Pending outcomes stay
pending (right-censored), never failures; counts are increments, never re-added cumulatives.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

from agentrs.temporal.delayed_outcome_ledger import ActionRecord, DelayedOutcomeLedger, EventRecord
from agentrs.temporal.horizon_config import HorizonConfig, HorizonProfile, parse_duration
from agentrs.temporal.observation_scheduler import TemporalObservationScheduler

EXTRA_PROFILES = {
    "github:create_pull_request": HorizonProfile(name="pull_request", horizons=[parse_duration(x) for x in
                                                                               ("1m", "15m", "1h", "6h", "1d", "3d")],
                                                 window_s=parse_duration("14d")),
    "email:send": HorizonProfile(name="email", horizons=[parse_duration(x) for x in ("1h", "6h", "1d", "3d")],
                                 window_s=parse_duration("7d")),
}


class OutcomeTracker:
    def __init__(self, workdir: str | Path, horizon_cfg: Optional[HorizonConfig] = None):
        d = Path(workdir)
        d.mkdir(parents=True, exist_ok=True)
        self.ledger = DelayedOutcomeLedger(d / "ledger.sqlite")
        self.scheduler = TemporalObservationScheduler(d / "followups.sqlite")
        self.hcfg = horizon_cfg or HorizonConfig()

    def profile(self, provider: str, operation: str) -> HorizonProfile:
        return EXTRA_PROFILES.get(f"{provider}:{operation}") or self.hcfg.profile_for(operation)

    def record_action(self, *, action_id: str, provider: str, operation: str, agent_id: str, account_id: str,
                      object_ref: Optional[str], content: Optional[str], action_str: str, context_text: str,
                      goal: Optional[str], time_base: str, action_time: float, checkpoint: Optional[str],
                      generation: Optional[dict], forecast: Optional[dict], split: str = "train",
                      task_id: Optional[str] = None, group_key: Optional[str] = None) -> str:
        prof = self.profile(provider, operation)
        rec = ActionRecord(action_id=action_id, source_id=f"harness:{provider}", source_type="self", actor_type="agent",
                           actor_ref=agent_id, env_id=provider, time_base=time_base, action_time=action_time,
                           observed_time=action_time, goal=goal, context_text=context_text,
                           context={"object_ref": object_ref, "account_id": account_id,
                                    "task_id": group_key or task_id, "harness_task_id": task_id,
                                    "operation": operation}, action_str=action_str,
                           action={"operation": operation, "content": content}, semantic=generation,
                           policy_checkpoint=checkpoint, split=split, group=f"{provider}:{operation}",
                           profile=prof.name, window_end=action_time + prof.window_s)
        self.ledger.record_action(rec)
        if forecast is not None:  # recorded before execution by the caller (created_time <= action_time)
            self.ledger.record_forecast(action_id, forecast, forecast.get("checkpoint", checkpoint or ""),
                                        info_cutoff_time=action_time, created_time=action_time,
                                        horizons=[h["horizon_s"] for h in forecast.get("horizons", [])])
        if object_ref is not None:
            self.scheduler.schedule_horizons(action_id, f"harness:{provider}", action_time, prof.horizons, time_base)
        return action_id

    def record_event(self, action_id: str, type_: str, env_time: Optional[float], observed_time: float,
                     provider_event_id: str, source: str, by_original_actor: Optional[bool] = None,
                     payload: Optional[dict] = None) -> bool:
        if self.ledger.action(action_id) is None:
            return False
        fresh = self.ledger.record_event(EventRecord(action_id=action_id, type=type_, observed_time=observed_time,
                                                     env_event_time=env_time, by_original_actor=by_original_actor,
                                                     payload=payload or {}, source_id=source,
                                                     provider_event_id=provider_event_id))
        if fresh:
            a = self.ledger.action(action_id)
            self.scheduler.on_event(action_id, a["source_id"], observed_time, a["time_base"])
        return fresh

    async def run_due(self, fetchers: dict[str, Callable[[str], Any]], now_by_base: dict[str, float],
                      owner: str = "harness-outcomes") -> dict[str, int]:
        """Execute due follow-up observations. ``fetchers``: source id -> async fn(object_ref) -> payload."""
        from agentrs.temporal.delayed_outcome_ledger import ObservationRecord

        stats = {"done": 0, "retry": 0, "failed": 0}
        while True:
            jobs = self.scheduler.claim_due(now_by_base, owner, limit=128)
            if not jobs:
                return stats
            for j in jobs:
                now = now_by_base[j["time_base"]]
                a = self.ledger.action(j["action_id"])
                ref = ((a or {}).get("context") or {}).get("object_ref")
                fn = fetchers.get(j["source_id"])
                try:
                    if a is None or ref is None or fn is None:
                        raise RuntimeError("nothing to observe (action purged, no object, or connector unavailable)")
                    payload = await fn(str(ref))
                except Exception as e:  # noqa: BLE001 - retried with backoff; a final failure is a missing observation
                    st = self.scheduler.fail(j["job_id"], f"{type(e).__name__}: {e}", now)
                    stats["retry" if st == "retry" else "failed"] += 1
                    if st == "failed" and a is not None:
                        self.ledger.record_observation(ObservationRecord(
                            action_id=j["action_id"], actual_time=now, payload={}, horizon_label=j["horizon_label"],
                            intended_time=j["due_time"], completeness="missing", source_id=j["source_id"],
                            job_id=j["job_id"], idem=f"job:{j['job_id']}"))
                    continue
                oid = self.ledger.record_observation(ObservationRecord(
                    action_id=j["action_id"], actual_time=now, payload=payload, horizon_label=j["horizon_label"],
                    intended_time=j["due_time"], completeness="complete", source_id=j["source_id"],
                    job_id=j["job_id"], idem=f"job:{j['job_id']}"))
                self.scheduler.complete(j["job_id"], oid)
                stats["done"] += 1

    def close_windows(self, now_by_base: dict[str, float]) -> int:
        n = 0
        for base, now in now_by_base.items():
            for r in self.ledger._q("SELECT action_id FROM actions WHERE status='open' AND window_end<=? AND time_base=?",
                                    (now, base)):
                self.ledger.close_action(r["action_id"], now, "window")
                self.scheduler.cancel(r["action_id"], "outcome window closed")
                n += 1
        return n


def wall_now() -> dict[str, float]:
    return {"real": time.time()}
