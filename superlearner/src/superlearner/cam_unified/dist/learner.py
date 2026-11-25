"""The central learner: the only process that optimises the shared weights (spec 14.1/14.4, manifest A16/F03).

Loop (`CentralLearner.step`):

1. ingest committed queue segments (dedup, exactly-once into the `ArchiveP` sink; DecisionTraces into the trace
   recorder; delayed outcomes fill the traces of the decisions they are attributed to);
2. assemble trajectories per (actor, episode) with `TrajectoryAssembler` (decision records in step order, delayed
   outcome records attached to the decisions they are attributed to — also when the outcome arrives first —, closed
   by the final ``done`` decision or an explicit ``episode_end`` record; trajectories left open longer than
   ``stale_s`` are emitted as truncated);
3. route each ready trajectory by policy lag (`offpolicy.LagRouter`): lag = learner version − behaviour version;
   lag 0 → on-policy, 0 < lag ≤ ``max_lag`` → off-policy (V-trace in the model), beyond → discarded from training
   (kept in the archive, counted);
4. call ``model.update(LearnerBatch)`` — one optimiser step — and increment the version;
5. append the update to ``learner/updates.jsonl`` (update id, version, event ids consumed, lag summary, metrics),
   link the consumed decisions' traces to the update (``update_refs``), and publish a validated weights-only checkpoint
   every ``publish_every`` updates.

Versions count optimiser updates (a published version is the parameters after that many updates), so policy lag is
exactly the number of updates a sample's behaviour policy is behind the parameters being trained.

``train_actors`` restricts which actors' experience is trained on (everything is still archived and traced) — used to
prove that one actor's experience changes another actor's behaviour.

A restarted learner resumes from LATEST (weights and version; optimiser moments restart from zero — weights-only
checkpoints by design) and rebuilds its open trajectories from the archive: every archived record whose event id
does not appear in ``updates.jsonl`` is re-assembled.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol, runtime_checkable

from ..observe.accounting import ComputeMeter, atomic_write_json
from ..observe.trace import TraceRecorder
from .actor import resolve, run_paths
from .offpolicy import LagRouter, lag_summary
from .publish import CheckpointPublisher, load_weights, read_latest
from .queue import ExperienceQueue


@dataclass
class Trajectory:
    actor_id: str
    episode_id: str
    env_id: str
    events: list = field(default_factory=list)          # decision EventRecords, step order
    outcomes: dict = field(default_factory=dict)        # decision event_id -> [delayed outcome EventRecords]
    world_events: list = field(default_factory=list)    # unattributed records of the episode
    complete: bool = False
    truncated: bool = False
    t_first: float = field(default_factory=time.time)
    lag: int = 0
    route: str = ""

    @property
    def behavior_version(self) -> int:
        return max((int(e.model_version) for e in self.events), default=0)

    @property
    def n(self) -> int:
        return len(self.events)


@dataclass
class LearnerBatch:
    learner_version: int
    trajectories: list                                  # on-policy and off-policy Trajectory objects (lag set)
    lag: dict                                           # LagReport.summary()
    late_outcomes: list = field(default_factory=list)   # delayed outcomes whose decision was already trained on

    @property
    def on_policy(self) -> list:
        return [t for t in self.trajectories if t.lag == 0]

    @property
    def off_policy(self) -> list:
        return [t for t in self.trajectories if t.lag > 0]


@runtime_checkable
class LearnerModelP(Protocol):
    def state_dict(self) -> dict: ...                   # the weights actors load (tensors only)
    def load_state_dict(self, sd: dict) -> None: ...
    def update(self, batch: LearnerBatch) -> dict: ...  # one optimiser step -> metrics


class TrajectoryAssembler:
    def __init__(self, stale_s: float = 600.0):
        self.stale_s = stale_s
        self.open: dict = {}
        self.index: dict = {}                   # decision event_id -> trajectory key
        self.waiting: dict = defaultdict(list)  # attributed_to -> outcome records that arrived before the decision
        self.trained: set = set()               # decision event ids already emitted (for late-outcome detection)
        self.late: list = []
        self.counts = Counter()

    def add(self, records: list) -> None:
        for r in records:
            key = (r.actor_id, r.episode_id)
            if r.action is not None:
                tr = self.open.get(key)
                if tr is None:
                    tr = self.open[key] = Trajectory(r.actor_id, r.episode_id, r.env_id)
                tr.events.append(r)
                self.index[r.event_id] = key
                if r.event_id in self.waiting:
                    tr.outcomes.setdefault(r.event_id, []).extend(self.waiting.pop(r.event_id))
                if r.done:
                    tr.complete = True
                self.counts["decisions"] += 1
            elif r.attributed_to:
                self.counts["delayed"] += 1
                k = self.index.get(r.attributed_to)
                if k is not None and k in self.open:
                    self.open[k].outcomes.setdefault(r.attributed_to, []).append(r)
                elif r.attributed_to in self.trained:
                    self.late.append(r)
                    self.counts["late_outcomes"] += 1
                else:
                    self.waiting[r.attributed_to].append(r)
            else:
                tr = self.open.get(key)
                if (r.extra or {}).get("kind") == "episode_end":
                    if tr is not None:
                        tr.complete = True
                        tr.truncated = bool(r.extra.get("truncated"))
                    self.counts["episode_end"] += 1
                elif tr is not None:
                    tr.world_events.append(r)

    def pop_ready(self, now: Optional[float] = None) -> list:
        now = time.time() if now is None else now
        out = []
        for key in list(self.open):
            tr = self.open[key]
            if tr.complete or now - tr.t_first > self.stale_s:
                if not tr.complete:
                    tr.truncated = True
                    self.counts["stale_truncated"] += 1
                tr.events.sort(key=lambda e: e.step)
                out.append(self.open.pop(key))
                for e in tr.events:
                    self.index.pop(e.event_id, None)
                    self.trained.add(e.event_id)
        if len(self.trained) > 2_000_000:
            self.trained = set(list(self.trained)[-1_000_000:])
        return out

    def take_late(self) -> list:
        out, self.late = self.late, []
        return out


class CentralLearner:
    def __init__(self, run_dir: str, model, sink=None, max_lag: int = 8, lag_mode: str = "vtrace",
                 min_decisions: int = 64, max_batch_decisions: Optional[int] = None,
                 max_batch_trajectories: Optional[int] = None, publish_every: int = 1,
                 train_actors: Optional[list] = None, ingest_segments: int = 8, stale_s: float = 600.0,
                 max_backlog: Optional[int] = None,
                 queue_kwargs: Optional[dict] = None, keep_versions: int = 5, validate_fn=None,
                 record_traces: bool = True, meter: Optional[ComputeMeter] = None, resume: bool = True):
        self.run_dir = os.path.abspath(run_dir)
        self.paths = run_paths(self.run_dir)
        for d in self.paths.values():
            os.makedirs(d, exist_ok=True)
        self.model = model
        self.sink = sink
        self.queue = ExperienceQueue(self.paths["spool"], **(queue_kwargs or {}))
        self.consumer = self.queue.consumer()
        self.publisher = CheckpointPublisher(self.paths["ckpt"], keep_last=keep_versions, validate_fn=validate_fn)
        self.traces = TraceRecorder(self.paths["traces"], name="learner") if record_traces else None
        self.router = LagRouter(max_lag=max_lag, mode=lag_mode)
        self.assembler = TrajectoryAssembler(stale_s=stale_s)
        self.min_decisions = min_decisions
        self.max_batch_trajectories = max_batch_trajectories
        self.max_batch_decisions = max_batch_decisions
        self.max_backlog = max_backlog or 8 * max(min_decisions, max_batch_decisions or 0)
        self.publish_every = max(1, publish_every)
        self.train_actors = set(train_actors) if train_actors else None
        self.ingest_segments = ingest_segments
        self.meter = meter or ComputeMeter("learner", role="learner")
        self.updates_path = os.path.join(self.paths["learner"], "updates.jsonl")
        self.ready: list = []
        self.counts = Counter()
        self.version = 0
        self.t_start = time.time()
        lt = read_latest(self.paths["ckpt"]) if resume else None
        if lt is not None:
            payload = load_weights(os.path.join(self.paths["ckpt"], lt["dir"], "weights.pt"), lt["sha256"])
            self.model.load_state_dict(payload["state_dict"])
            self.version = int(lt["version"])
            self.counts["resumed_from"] = self.version
            self._recover_open()
        else:
            self.publish({"initial": True})

    # ---------------------------------------------------------------- recovery
    def _recover_open(self) -> None:
        if self.sink is None or not os.path.exists(self.updates_path):
            return
        trained = set()
        with open(self.updates_path) as fh:
            for line in fh:
                if line.strip():
                    trained.update(json.loads(line).get("event_ids", []))
        n = 0
        for chunk in self.sink.scan():
            fresh = [r for r in chunk if r.event_id not in trained]
            n += len(fresh)
            self.assembler.add(fresh)
        self.counts["recovered_records"] = n

    # ---------------------------------------------------------------- publication
    def publish(self, meta: Optional[dict] = None) -> dict:
        cfg = self.model.config() if hasattr(self.model, "config") else {}
        with self.meter.gpu():
            m = self.publisher.publish(self.model.state_dict(), self.version, meta=meta or {}, config=cfg)
        self.counts["published"] += 1
        if not m["validated"]:
            self.counts["publish_rejected"] += 1
        return m

    # ---------------------------------------------------------------- one iteration
    def ingest(self) -> int:
        res = self.consumer.ingest(self.sink, self.traces, max_segments=self.ingest_segments)
        if res.empty:
            return 0
        self.assembler.add(res.events)
        if self.traces is not None:
            for d in res.delayed:
                vals = [{"event_id": d.event_id, "key": (o.meta or {}).get("key") or o.source or o.kind.value,
                         "value": o.value, "unit": o.unit, "t": o.t, "text": o.content[:200]}
                        for o in d.observations]
                patch = {"delayed": vals}
                for v in vals:
                    if v["value"] is not None:
                        patch[v["key"]] = v["value"]
                self.traces.fill_outcome(patch, event_id=d.attributed_to)
        self.meter.add(records=len(res.events))
        self.counts["ingested_events"] += len(res.events)
        self.counts["ingested_traces"] += len(res.traces)
        self.counts["dup"] += res.n_dup
        self.counts["quarantined"] += len(res.quarantined)
        return len(res.events)

    def backlog(self) -> int:
        """Decisions ingested but not yet trained (ready + open trajectories)."""
        return sum(t.n for t in self.ready) + sum(t.n for t in self.assembler.open.values())

    def step(self) -> Optional[dict]:
        # End-to-end backpressure: stop consuming while the trainable backlog is full, so unconsumed segments stay in
        # the spool and the spool byte limit blocks (or sheds at) the actors instead of the learner's memory growing.
        if self.backlog() < self.max_backlog:
            self.ingest()
        else:
            self.counts["ingest_paused"] += 1
        for tr in self.assembler.pop_ready():
            if self.train_actors is not None and tr.actor_id not in self.train_actors:
                self.counts["not_trained_actor_filter"] += tr.n
                continue
            if not tr.events:
                continue
            self.ready.append(tr)
        n_ready = sum(t.n for t in self.ready)
        if n_ready < self.min_decisions:
            return None
        take, n = [], 0
        cap = self.max_batch_decisions or self.min_decisions
        while self.ready and n < cap and (self.max_batch_trajectories is None
                                          or len(take) < self.max_batch_trajectories):
            tr = self.ready.pop(0)
            take.append(tr)
            n += tr.n
        rep = self.router.route_trajectories(take, self.version)
        for tr in rep.on_policy + rep.off_policy:
            tr.lag = self.version - tr.behavior_version
            tr.route = "on_policy" if tr.lag == 0 else "off_policy"
        for tr in rep.discarded:
            tr.route = "discarded"
            self.counts["discarded_trajectories"] += 1
            self.counts["discarded_decisions"] += tr.n
        trainable = rep.on_policy + rep.off_policy
        lagsum = rep.summary()
        lagsum["decision_lags"] = lag_summary([t.lag for t in trainable for _ in range(t.n)])
        if not trainable:
            self._log_update(None, take, rep, lagsum, trained=False)
            return None
        batch = LearnerBatch(self.version, trainable, lagsum, self.assembler.take_late())
        t0 = time.perf_counter()
        with self.meter.gpu():
            metrics = self.model.update(batch)
        dt = time.perf_counter() - t0
        self.version += 1
        n_dec = sum(t.n for t in trainable)
        self.meter.add(updates=1, samples_trained=n_dec)
        self.counts["updates"] += 1
        self.counts["trained_decisions"] += n_dec
        self.counts["trained_on_policy"] += sum(t.n for t in rep.on_policy)
        self.counts["trained_off_policy"] += sum(t.n for t in rep.off_policy)
        published = None
        if self.version % self.publish_every == 0:
            published = self.publish({"update": self.version, "metrics": metrics})
        rec = self._log_update(metrics, trainable, rep, lagsum, trained=True, update_s=dt, published=published)
        return rec

    def _log_update(self, metrics, trajs, rep, lagsum, trained: bool, update_s: float = 0.0, published=None):
        uid = f"u{self.version:08d}" if trained else f"skip{int(time.time() * 1e6)}"
        ids = [e.event_id for t in trajs for e in t.events] if trained else []
        by_actor = Counter()
        for t in trajs:
            by_actor[t.actor_id] += t.n
        rec = {"update_id": uid, "version": self.version, "trained": trained, "t": time.time(),
               "t_rel": time.time() - self.t_start, "update_s": update_s, "n_traj": len(trajs),
               "n_decisions": sum(t.n for t in trajs), "by_actor": dict(by_actor), "lag": lagsum,
               "metrics": metrics or {}, "published": bool(published and published.get("validated")),
               "event_ids": ids}
        with open(self.updates_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        if trained and self.traces is not None:
            self.traces.add_update_refs(ids, uid)
            self.traces.flush()
        return rec

    # ---------------------------------------------------------------- driving
    def run(self, stop_event=None, max_updates: Optional[int] = None, max_seconds: Optional[float] = None,
            idle_sleep: float = 0.005, status_every_s: float = 0.5, drain: bool = True) -> dict:
        t0 = time.time()
        last = 0.0
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if max_updates is not None and self.counts["updates"] >= max_updates:
                break
            if max_seconds is not None and time.time() - t0 >= max_seconds:
                break
            r = self.step()
            if r is None:
                time.sleep(idle_sleep)
            if time.time() - last >= status_every_s:
                self.write_status()
                last = time.time()
        if drain:
            while self.ingest():
                pass
        self.write_status(final=True)
        return self.stats()

    def stats(self) -> dict:
        return {"version": self.version, "counts": dict(self.counts), "router": self.router.totals(),
                "queue": self.consumer.stats(), "assembler": dict(self.assembler.counts),
                "open_trajectories": len(self.assembler.open), "ready_trajectories": len(self.ready),
                "archive_count": self.sink.count() if self.sink is not None else None}

    def write_status(self, final: bool = False) -> None:
        st = self.stats()
        st.update({"t": time.time(), "pid": os.getpid(), "state": "finished" if final else "running",
                   "t_start": self.t_start})
        atomic_write_json(os.path.join(self.paths["status"], "learner.json"), st, fsync=False)
        self.meter.write(os.path.join(self.paths["compute"], f"learner.{os.getpid()}.json"))

    def close(self) -> None:
        if self.traces is not None:
            self.traces.close()
        self.consumer.close()
        if self.sink is not None and hasattr(self.sink, "close"):
            self.sink.close()


@dataclass
class LearnerSpec:
    run_dir: str
    model_factory: str = "superlearner.cam_unified.dist.toyworld:make_tiny_learner"
    model_kwargs: dict = field(default_factory=dict)
    sink_factory: Optional[str] = "superlearner.cam_unified.dist.sinks:make_jsonl_archive"
    sink_kwargs: dict = field(default_factory=dict)
    max_lag: int = 8
    lag_mode: str = "vtrace"
    min_decisions: int = 64
    max_batch_decisions: Optional[int] = None
    max_batch_trajectories: Optional[int] = None
    publish_every: int = 1
    train_actors: Optional[list] = None
    max_updates: Optional[int] = None
    max_seconds: Optional[float] = None
    ingest_segments: int = 8
    max_backlog: Optional[int] = None
    stale_s: float = 600.0
    queue_kwargs: dict = field(default_factory=dict)
    torch_threads: int = 2
    validate_factory: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def learner_main(spec: dict, stop_event=None) -> None:
    """Process entry point (spawn)."""
    import torch
    s = LearnerSpec(**spec)
    torch.set_num_threads(s.torch_threads)
    model = resolve(s.model_factory)(**s.model_kwargs)
    sink = None
    if s.sink_factory:
        sink = resolve(s.sink_factory)(run_paths(s.run_dir)["archive"], **s.sink_kwargs)
    validate = resolve(s.validate_factory)() if s.validate_factory else None
    L = CentralLearner(s.run_dir, model, sink=sink, max_lag=s.max_lag, lag_mode=s.lag_mode,
                       min_decisions=s.min_decisions, max_batch_decisions=s.max_batch_decisions,
                       max_batch_trajectories=s.max_batch_trajectories,
                       publish_every=s.publish_every, train_actors=s.train_actors, ingest_segments=s.ingest_segments,
                       stale_s=s.stale_s, max_backlog=s.max_backlog, queue_kwargs=s.queue_kwargs,
                       validate_fn=validate)
    try:
        L.run(stop_event=stop_event, max_updates=s.max_updates, max_seconds=s.max_seconds)
    finally:
        L.close()
