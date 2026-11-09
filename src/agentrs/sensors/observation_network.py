"""ObservationSensorNetwork: an ingestion subsystem that runs independently of the acting agent.

For every authorised source it repeatedly:

1. polls new public events (bounded pages, respecting the declared rate limits);
2. normalises them; checks record categories against the declaration; pseudonymises actors;
   deduplicates (same source, across sources, and our own actions);
3. records each new observed action in the ``DelayedOutcomeLedger`` **and immediately forecasts its
   consequences** at the profile's horizons with the shared temporal world model (batched through a
   bounded ``ForecastService``); the forecast is stored before any outcome is observed;
4. schedules follow-up observations (fixed / event-driven / adaptive) in the durable scheduler;
5. attaches follow-up events (replies, edits, moderation, deletion) to the right action, applying
   the source's deletion policy;
6. executes due observation jobs through the source's public interface and (optionally) records an
   updated conditional forecast after each observation, with its own information cutoff;
7. closes outcome windows and cancels remaining jobs.

Our own agent's actions enter through ``register_self_action`` / ``link_self_action`` and are then
followed through exactly the same sources and ledger, labelled ``source_type="self"``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from agentrs.data.observed_actor_action import ObservedActorAction, canonical_json
from agentrs.sensors.deduplication import Deduplicator, stable_action_id
from agentrs.sensors.permissions import check_record, check_source, redact_pii
from agentrs.sensors.source_adapter import FollowUp, SourceError
from agentrs.sensors.source_registry import SourceRegistry
from agentrs.temporal.delayed_outcome_ledger import (
    ActionRecord,
    DelayedOutcomeLedger,
    EventRecord,
)
from agentrs.temporal.horizon_config import HorizonConfig
from agentrs.temporal.observation_scheduler import (
    AdaptivePolicy,
    ObservationWorker,
    SchedulerConfig,
    TemporalObservationScheduler,
)


class NetworkConfig(BaseModel):
    scheduling: str = "fixed"  # fixed | event | adaptive
    event_mode_horizons: list[float] = [3600.0, 86400.0]  # polls kept in event mode (counts are not in the feed)
    updated_forecasts: bool = False
    max_updated_per_action: int = 3
    forecast_batch: int = 64
    forecast_queue: int = 4096
    max_pages_per_poll: int = 20
    forecast_horizons: list[float] | None = None  # default: the action's profile horizons
    forecasts_enabled: bool = True
    use_feed_events: bool = True  # False: learn only from scheduled polls (ablation: no event feed)
    event_coalesce_s: float = 60.0  # event-driven checks of one action are merged within this window


class QueueFull(RuntimeError):
    pass


class ForecastService:
    """Shared, batched inference over many observed actions and horizons (one model copy)."""

    def __init__(self, model, ledger: DelayedOutcomeLedger, batch: int = 64, max_queue: int = 4096):
        self.model = model
        self.ledger = ledger
        self.batch = batch
        self.max_queue = max_queue
        self.q: deque = deque()
        self.lock = threading.Lock()
        self.stats = {"forecasts": 0, "batches": 0, "seconds": 0.0, "rejected": 0}

    def submit(self, action_id: str, query, horizons: list[float], created_time: float, cutoff_time: float,
               kind: str = "initial") -> None:
        with self.lock:
            if len(self.q) >= self.max_queue:
                self.stats["rejected"] += 1
                raise QueueFull("forecast queue full")
            self.q.append((action_id, query, list(horizons), created_time, cutoff_time, kind))

    def flush(self) -> list[tuple[str, dict]]:
        out = []
        while True:
            with self.lock:
                if not self.q:
                    break
                chunk = [self.q.popleft() for _ in range(min(self.batch, len(self.q)))]
            t = time.perf_counter()
            by_h: dict[tuple, list] = {}
            for c in chunk:
                by_h.setdefault(tuple(c[2]), []).append(c)
            for hs, items in by_h.items():
                preds = self.model.predict_batch([c[1] for c in items], list(hs))
                for c, p in zip(items, preds):
                    d = p.to_dict()
                    hist = hashlib.sha1(canonical_json(c[1].history).encode()).hexdigest()[:12]
                    self.ledger.record_forecast(c[0], d, p.checkpoint, info_cutoff_time=c[4], created_time=c[3],
                                                horizons=list(hs), kind=c[5], history_hash=hist)
                    out.append((c[0], d))
            self.stats["forecasts"] += len(chunk)
            self.stats["batches"] += 1
            self.stats["seconds"] += time.perf_counter() - t
        return out


class ObservationSensorNetwork:
    def __init__(self, workdir: str | Path, registry: SourceRegistry, forecaster: ForecastService | None = None,
                 cfg: NetworkConfig | None = None, horizon_cfg: HorizonConfig | None = None,
                 split_fn: Callable[[ObservedActorAction], str] | None = None,
                 ledger: DelayedOutcomeLedger | None = None,
                 scheduler: TemporalObservationScheduler | None = None):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.ledger = ledger or DelayedOutcomeLedger(self.dir / "ledger.sqlite")
        self.scheduler = scheduler or TemporalObservationScheduler(
            self.dir / "jobs.sqlite", SchedulerConfig(event_coalesce_s=(cfg or NetworkConfig()).event_coalesce_s))
        self.dedup = Deduplicator(self.dir / "dedup.sqlite")
        self.forecaster = forecaster
        self.cfg = cfg or NetworkConfig()
        self.hcfg = horizon_cfg or HorizonConfig()
        self.split_fn = split_fn or (lambda o: "train")
        self.adaptive = AdaptivePolicy()
        self._cursor_path = self.dir / "cursors.json"
        self.cursors: dict[str, int] = json.loads(self._cursor_path.read_text()) if self._cursor_path.exists() else {}
        self._lock = threading.RLock()
        self._updated: dict[str, int] = {}
        self.stats = {"polled_events": 0, "new_actions": 0, "follow_ups": 0, "refused": 0, "unknown_object": 0,
                      "duplicates": 0, "observations": 0, "updated_forecasts": 0, "redacted": 0, "wall_ingest": 0.0,
                      "wall_observe": 0.0}
        for sid in self.registry.active():
            rpm = self.registry.declaration(sid).collection_limits.requests_per_minute
            self.scheduler.set_budget(sid, capacity=max(1.0, rpm), refill_per_s=rpm / 60.0,
                                      now=self.registry.adapters[sid].now())

    # ------------------------------------------------------------------ helpers
    def _save_cursors(self) -> None:
        self._cursor_path.write_text(json.dumps(self.cursors))

    def _query_for(self, action: dict, cutoff_elapsed: float = 0.0, history=(), known=None):
        from agentrs.temporal.temporal_world_model import TemporalQuery

        return TemporalQuery(context=action["context_text"] or "", action=action["action_str"],
                             source_type=action["source_type"], actor_type=action["actor_type"], goal=action["goal"],
                             cutoff_elapsed_s=cutoff_elapsed, history=list(history), known_now=dict(known or {}))

    def _horizons(self, operation: str) -> list[float]:
        return list(self.cfg.forecast_horizons or self.hcfg.horizons_for(operation))

    # ------------------------------------------------------------------ ingestion
    def ingest(self, source_id: str) -> dict:
        t0 = time.perf_counter()
        adapter = self.registry.adapters.get(source_id)
        if adapter is None or source_id not in self.registry.active():
            raise PermissionError(f"source {source_id} is not active")
        decl = adapter.declaration
        ok, why = check_source(decl)
        if not ok:
            raise PermissionError("; ".join(why))
        new_ids = []
        pages = 0
        while pages < self.cfg.max_pages_per_poll:
            try:
                raws, cur = adapter.poll(self.cursors.get(source_id, 0))
            except SourceError:
                break
            pages += 1
            if not raws:
                if cur != self.cursors.get(source_id, 0):  # e.g. a live source's starting point
                    self.cursors[source_id] = cur
                    self._save_cursors()
                break
            self.stats["polled_events"] += len(raws)
            for raw in raws:
                for item in adapter.normalize(raw):
                    aid = self._handle(adapter, item)
                    if aid:
                        new_ids.append(aid)
            self.cursors[source_id] = cur
            self._save_cursors()
        if self.forecaster is not None and self.cfg.forecasts_enabled:
            done = dict(self.forecaster.flush())
        else:
            done = {}
        for aid in new_ids:
            a = self.ledger.action(aid)
            hs = self.hcfg.profile_for(a["action"].get("operation")).horizons
            self._schedule(aid, a, source_id, hs, done.get(aid))
        self.stats["wall_ingest"] += time.perf_counter() - t0
        return {"new_actions": len(new_ids), "pages": pages}

    def _schedule(self, aid: str, a: dict, source_id: str, hs: list[float], forecast: dict | None) -> None:
        mode = self.cfg.scheduling
        if mode == "adaptive" and forecast is not None:
            sub = self.model_forecast_at(forecast, hs)
            self.scheduler.schedule_adaptive(aid, source_id, a["action_time"], hs, a["time_base"], sub, self.adaptive)
        elif mode == "event":
            keep = sorted(set([h for h in self.cfg.event_mode_horizons if h <= hs[-1]] + [hs[-1]]))
            self.scheduler.schedule_horizons(aid, source_id, a["action_time"], keep, a["time_base"])
        else:
            self.scheduler.schedule_horizons(aid, source_id, a["action_time"], hs, a["time_base"])

    @staticmethod
    def model_forecast_at(forecast: dict, hs: list[float]) -> dict:
        return forecast  # forecasts are made at the profile horizons already

    def _handle(self, adapter, item) -> str | None:
        decl = adapter.declaration
        sid = decl.source_id
        cats = adapter.categories_of(item)
        ok, why = check_record(decl, cats)
        if not ok:
            self.stats["refused"] += 1
            return None
        if isinstance(item, FollowUp):
            aid = self.dedup.alias(adapter.env_key(item.object_ref)) if hasattr(adapter, "env_key") else None
            aid = aid or stable_action_id(sid, item.object_ref)
            if self.ledger.action(aid) is None:
                self.stats["unknown_object"] += 1
                return None
            if not self.cfg.use_feed_events and item.type != "item_deleted":
                return None
            fresh = self.ledger.record_event(EventRecord(
                action_id=aid, type=item.type, observed_time=adapter.now(), env_event_time=item.env_event_time,
                actor_ref=item.actor_ref, by_original_actor=item.by_original_actor, payload=item.payload,
                source_id=sid, provider_event_id=item.provider_event_id))
            if not fresh:
                self.stats["duplicates"] += 1
                return None
            self.stats["follow_ups"] += 1
            if self.cfg.scheduling in ("event", "adaptive"):
                self.scheduler.on_event(aid, sid, adapter.now(), self.ledger.action(aid)["time_base"])
            if item.type == "item_deleted" and decl.deletion_policy != "retain":
                self.scheduler.cancel(aid, "deleted at source")
                if decl.deletion_policy == "purge":
                    self.ledger.purge_actions([aid], "deleted at source", source_id=sid)
                    self.dedup.forget([aid])
                else:
                    self.ledger.redact_action(aid, "deleted at source")
                self.stats["redacted"] += 1
            return None
        obs: ObservedActorAction = item
        if decl.pii_redaction and obs.action.content:
            obs.action.content = redact_pii(obs.action.content)
        aid = stable_action_id(sid, obs.provider_object_id or obs.observation_id)
        env_key = adapter.env_key(obs.provider_object_id) if hasattr(adapter, "env_key") and obs.provider_object_id else None
        verdict, _ = self.dedup.check(aid, obs.content_hash, obs.actor_ref, obs.effective_action_time(), env_key)
        if verdict != "new":
            self.stats["duplicates"] += verdict != "own_action"
            return None
        prof = self.hcfg.profile_for(obs.action.operation)
        t_act = obs.effective_action_time()
        rec = ActionRecord(
            action_id=aid, source_id=sid, source_type="third_party", actor_type=obs.actor_type, actor_ref=obs.actor_ref,
            env_id=obs.env_id, env_version=obs.env_version, time_base=obs.time_base, action_time=t_act,
            observed_time=max(t_act, obs.observation_time), goal=None, context_text=obs.context_text(),
            context={**obs.context_before, "object_ref": obs.provider_object_id}, action_str=obs.action.to_action_str(),
            action=obs.action.model_dump(), provenance=obs.provenance,
            permission={"categories": adapter.categories_of(obs), "source_declaration": decl.source_id,
                        "training_authorized": decl.training_authorized},
            split=self.split_fn(obs), group=obs.group, profile=prof.name, window_end=t_act + prof.window_s)
        self.ledger.record_action(rec)
        self.stats["new_actions"] += 1
        if self.forecaster is not None and self.cfg.forecasts_enabled:
            a = self.ledger.action(aid)
            lag = max(0.0, rec.observed_time - t_act)
            self.forecaster.submit(aid, self._query_for(a, cutoff_elapsed=lag if lag > 60 else 0.0),
                                   self._horizons(obs.action.operation), created_time=rec.observed_time,
                                   cutoff_time=rec.observed_time)
        return aid

    # ------------------------------------------------------------------ our own actions
    def register_self_action(self, rec: ActionRecord, horizons: list[float] | None = None) -> str:
        """Record our own action and forecast it BEFORE it is executed (information cutoff = now)."""
        rec.source_type = "self"
        prof = self.hcfg.profile_for(rec.action.get("operation"))
        rec.profile = rec.profile or prof.name
        rec.window_end = rec.window_end or rec.action_time + prof.window_s
        self.ledger.record_action(rec)
        if self.forecaster is not None and self.cfg.forecasts_enabled:
            a = self.ledger.action(rec.action_id)
            self.forecaster.submit(rec.action_id, self._query_for(a), horizons or self._horizons(rec.action.get("operation")),
                                   created_time=rec.action_time, cutoff_time=rec.action_time)
            self.forecaster.flush()
        return rec.action_id

    def link_self_action(self, action_id: str, source_id: str, object_ref: str) -> None:
        """After execution: tie the environment object (e.g. the created post) to our action so the
        same sources follow it, and the feed's report of it is not taken for a third-party action."""
        adapter = self.registry.adapters[source_id]
        self.dedup.register_alias(adapter.env_key(object_ref), action_id)
        a = self.ledger.action(action_id)
        ctx = dict(a["context"] or {})
        ctx["object_ref"] = object_ref
        self.ledger._x("UPDATE actions SET context_json=? WHERE action_id=?", (json.dumps(ctx), action_id))
        hs = self.hcfg.profile_for(a["action"].get("operation")).horizons
        self._schedule(action_id, a, source_id, hs, (self.ledger.forecasts(action_id) or [{}])[0].get("prediction"))

    # ------------------------------------------------------------------ follow-up observations
    def _fetcher(self, source_id: str):
        adapter = self.registry.adapters[source_id]

        def f(job):
            if source_id not in self.registry.active():
                raise PermissionError("source revoked")
            a = self.ledger.action(job["action_id"])
            if a is None:
                raise RuntimeError("action purged")
            ref = (a["context"] or {}).get("object_ref")
            if ref is None:
                raise RuntimeError("action has no environment object yet")
            return adapter.fetch_state(str(ref))

        return f

    def observe(self) -> int:
        t0 = time.perf_counter()
        fetchers = {sid: self._fetcher(sid) for sid in self.registry.active()}
        now_by_base = {}
        for sid in fetchers:
            ad = self.registry.adapters[sid]
            now_by_base[ad.declaration.time_base] = ad.now()
        worker = ObservationWorker(self.scheduler, self.ledger, fetchers, owner="sensor-net")
        n = 0
        while True:
            k = worker.run_due(now_by_base, limit=512)
            n += k
            if k == 0:
                break
        self.stats["observations"] += worker.stats["done"]
        if self.cfg.updated_forecasts and self.forecaster is not None and worker.stats["done"]:
            self._updated_forecasts(now_by_base)
        self.stats["wall_observe"] += time.perf_counter() - t0
        return n

    def _updated_forecasts(self, now_by_base: dict[str, float]) -> None:
        from agentrs.temporal.outcome_labels import (
            OutcomeMapping,
            _mapped,
            default_event_text,
        )

        mapping = OutcomeMapping()
        recent = self.ledger._q("SELECT DISTINCT o.action_id FROM observations o JOIN actions a ON a.action_id=o.action_id "
                                "WHERE a.status='open' AND o.completeness='complete' AND o.ingest_wall>?",
                                (time.time() - 3600,))
        for r in recent:
            aid = r["action_id"]
            if self._updated.get(aid, 0) >= self.cfg.max_updated_per_action:
                continue
            a = self.ledger.action(aid)
            now = now_by_base.get(a["time_base"])
            if now is None:
                continue
            obs = self.ledger.observations(aid, observed_before=now)
            evs = self.ledger.events(aid, observed_before=now)
            if not obs:
                continue
            last = obs[-1]
            if any(f["info_cutoff_time"] >= last["actual_time"] for f in self.ledger.forecasts(aid)):
                continue
            occurred = sorted({n for n in (_mapped(e, mapping) for e in evs) if n} |
                              {k for k, v in (last["payload"].get("occurred") or {}).items() if v})
            known = {**(last["payload"].get("counts") or {}), "state": last["payload"].get("state"), "occurred": occurred}
            hist = [((e["env_event_time"] or e["observed_time"]) - a["action_time"], default_event_text(e)) for e in evs]
            c = last["actual_time"] - a["action_time"]
            hs = [h for h in self._horizons(a["action"].get("operation")) if h > c]
            if not hs:
                continue
            self.forecaster.submit(aid, self._query_for(a, c, hist, known), hs, created_time=last["actual_time"],
                                   cutoff_time=last["actual_time"], kind="updated")
            self._updated[aid] = self._updated.get(aid, 0) + 1
            self.stats["updated_forecasts"] += 1
        self.forecaster.flush()

    def close_windows(self) -> int:
        n = 0
        for sid in self.registry.active():
            ad = self.registry.adapters[sid]
            now = ad.now()
            rows = self.ledger._q("SELECT action_id FROM actions WHERE status='open' AND window_end<=? AND time_base=?",
                                  (now, ad.declaration.time_base))
            for r in rows:
                self.ledger.close_action(r["action_id"], now, "window")
                self.scheduler.cancel(r["action_id"], "outcome window closed")
                n += 1
        return n

    def step(self, parallel: bool = False) -> dict:
        """One collection cycle over all active sources: ingest, observe due follow-ups, close windows."""
        sids = self.registry.active()
        if parallel and len(sids) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(sids))) as ex:
                res = dict(zip(sids, ex.map(self.ingest, sids)))
        else:
            res = {s: self.ingest(s) for s in sids}
        n_obs = self.observe()
        closed = self.close_windows()
        return {"ingest": res, "observations": n_obs, "closed": closed}

    # ------------------------------------------------------------------ governance
    def revoke_source(self, source_id: str, reason: str) -> dict:
        self.registry.revoke(source_id, reason)
        self.scheduler.cancel_source(source_id, reason)
        ids = [r["action_id"] for r in self.ledger._q("SELECT action_id FROM actions WHERE source_id=?", (source_id,))]
        counts = self.ledger.purge_source(source_id, reason)
        self.dedup.forget(ids)
        return counts

    def enforce_retention(self, now_wall: float | None = None) -> int:
        now_wall = now_wall or time.time()
        n = 0
        for sid, e in self.registry.entries.items():
            days = e.declaration.retention_days
            if days is None:
                continue
            ids = [r["action_id"] for r in self.ledger._q("SELECT action_id FROM actions WHERE source_id=? AND "
                                                          "created_wall<?", (sid, now_wall - days * 86400))]
            if ids:
                self.ledger.purge_actions(ids, f"retention {days} days", source_id=sid)
                self.dedup.forget(ids)
                n += len(ids)
        return n
