"""The central learner: the only process that optimises the shared weights (spec 14.1/14.4, manifest A16/F03).

Loop (`CentralLearner.step`):

1. ingest committed queue segments (dedup, exactly-once into the `ArchiveP` sink; DecisionTraces into the trace
   recorder; delayed outcomes fill the traces of the decisions they are attributed to, numeric values summed per key);
2. assemble trajectories per (actor, episode) with `TrajectoryAssembler`: decision records in step order, delayed
   outcome records attached to the decisions they are attributed to (also when the outcome arrives first). Long
   episodes are cut into **segments** of ``segment_len`` decisions while they are still running: a segment carries a
   ``burn_in`` prefix of earlier (already trained) decisions as recurrent context, its trained decisions, and one
   following decision whose value bootstraps the V-trace targets at the cut; the most recent ``settle`` decisions are
   never trained before they are followed by later decisions (their delayed outcomes may still be in flight). An
   episode is closed by its final ``done`` decision (terminal), an explicit ``episode_end`` record, or ``stale_s``
   seconds without any new record of the episode (truncated: the unsettled tail is not trained, the decision after
   the trained part bootstraps). So a lifetime of any length trains continuously with bounded learner memory;
3. route each segment's decisions by policy lag (`offpolicy.LagRouter`, per decision): lag 0 → on-policy, 0 < lag ≤
   ``max_lag`` → off-policy (V-trace), beyond → not trained (kept as recurrent context, logged as skipped);
4. call ``model.update(LearnerBatch)`` — one optimiser step — and increment the version (not when the model reports
   ``optimizer_step: False``, e.g. nothing in the batch was usable; decisions a model reports in
   ``skipped_event_ids`` are logged as skipped, never as trained);
5. record what each update consumed — ``learner/updates.jsonl`` (audit log) and ``learner/consumed.sqlite`` (the
   durable index of every record the learner has consumed: trained / skipped decisions, their attached outcomes,
   world events, episode ends, late and orphan outcomes, with the learner version) — link the trained decisions'
   traces (``update_refs``), and publish a validated weights-only checkpoint every ``publish_every`` updates.

Versions count optimiser updates (a published version is the parameters after that many updates), so policy lag is
exactly the number of updates a sample's behaviour policy is behind the parameters being trained.

``train_actors`` restricts which actors' experience is trained on (everything is still archived and traced) — used to
prove that one actor's experience changes another actor's behaviour.

**Restart.** A restarted learner resumes from LATEST (weights and version; optimiser moments restart from zero —
weights-only checkpoints by design). Consumption recorded after LATEST (updates whose weights were never published)
is rolled back, and the open trajectories are rebuilt by streaming the archive and keeping every record that
``consumed.sqlite`` does not name; episodes that were partly trained get their last ``burn_in`` trained decisions back
as context (indexed archive query per episode). Memory is bounded by the untrained backlog, not the lifetime.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Protocol, runtime_checkable

from ..observe.accounting import ComputeMeter, atomic_write_json
from ..observe.trace import TraceRecorder
from .actor import resolve, run_paths
from .offpolicy import LagRouter, lag_summary
from .publish import CheckpointPublisher, load_weights, read_latest
from .queue import ExperienceQueue


@dataclass
class Trajectory:
    """A trainable piece of one episode: ``events`` = context prefix + offered decisions + (optional) bootstrap
    decision, in step order. ``train_mask[i]`` — the assembler offers decision i for training (the router narrows it
    to trainable decisions); ``terminal`` — the last event is the episode's final decision (no value after it);
    ``bootstrap`` — the last event only supplies the value estimate at the cut."""
    actor_id: str
    episode_id: str
    env_id: str
    events: list = field(default_factory=list)          # decision EventRecords, step order
    outcomes: dict = field(default_factory=dict)        # decision event_id -> [delayed outcome EventRecords]
    world_events: list = field(default_factory=list)    # unattributed records of the episode
    train_mask: list = field(default_factory=list)
    terminal: bool = False
    bootstrap: bool = False
    truncated: bool = False
    from_start: Optional[bool] = None                   # events[0] is the episode's first decision (derived)
    segment: int = 0
    aux_ids: list = field(default_factory=list)         # non-decision records consumed with this piece
    lags: list = field(default_factory=list)            # set by the router
    exact: list = field(default_factory=list)
    dropped: list = field(default_factory=list)         # [(event_id, reason)] offered but not trainable
    lag: int = 0
    route: str = ""

    def __post_init__(self):
        if not self.train_mask:
            self.train_mask = [True] * len(self.events)
        if self.from_start is None:
            self.from_start = bool(self.events) and int(self.events[0].step) == 0

    @property
    def behavior_version(self) -> int:
        """Oldest behaviour version among the offered decisions (the one with the largest lag)."""
        vs = [int(e.model_version) for e, m in zip(self.events, self.train_mask) if m]
        return min(vs) if vs else min((int(e.model_version) for e in self.events), default=0)

    @property
    def n(self) -> int:
        """Decisions to be trained (offered, and after routing: trainable)."""
        return sum(1 for m in self.train_mask if m)

    @property
    def n_events(self) -> int:
        return len(self.events)

    def trained_ids(self) -> list:
        return [e.event_id for e, m in zip(self.events, self.train_mask) if m]


@dataclass
class LearnerBatch:
    learner_version: int
    trajectories: list                                  # Trajectory objects with lags/train_mask/exact set
    lag: dict                                           # LagReport.summary()
    late_outcomes: list = field(default_factory=list)   # delayed outcomes whose decision was already trained on

    @property
    def on_policy(self) -> list:
        return [t for t in self.trajectories if t.route == "on_policy"]

    @property
    def off_policy(self) -> list:
        return [t for t in self.trajectories if t.route == "off_policy"]


@runtime_checkable
class LearnerModelP(Protocol):
    """``update`` performs one optimiser step over the batch and returns metrics. Reserved metric keys:
    ``optimizer_step`` (False: nothing was optimised — the learner then neither increments its version nor publishes)
    and ``skipped_event_ids`` (decisions of the batch the model did not train on; logged as skipped)."""
    def state_dict(self) -> dict: ...                   # the weights actors load (tensors only)
    def load_state_dict(self, sd: dict) -> None: ...
    def update(self, batch: LearnerBatch) -> dict: ...


class _Open:
    __slots__ = ("key", "env_id", "events", "outcomes", "world", "aux", "k", "complete", "truncated", "t_last",
                 "segments", "dirty")

    def __init__(self, key, env_id):
        self.key, self.env_id = key, env_id
        self.events: list = []
        self.outcomes: dict = {}
        self.world: list = []
        self.aux: list = []             # ids of episode_end / world records consumed with the next emission
        self.k = 0                      # index of the first decision not yet emitted for training
        self.complete = self.truncated = False
        self.t_last = time.time()
        self.segments = 0
        self.dirty = False              # events appended out of step order


class TrajectoryAssembler:
    """Builds trainable trajectory pieces from ingested records (see the module docstring).

    segment_len: decisions trained per segment of a running episode (None: whole episodes only — memory then grows
    with the episode); burn_in: context decisions kept before a segment; settle: most recent decisions never trained
    until followed by later ones (>= 1; the decision after a cut supplies the bootstrap value); stale_s: seconds
    without a new record before an open episode is closed as truncated; waiting_ttl_s / max_waiting: bound on delayed
    outcomes whose decision is unknown (expired ones are returned by `expire_waiting`, never dropped silently);
    trained_cap: FIFO bound on the ids remembered for late-outcome detection (older ones fall back to the learner's
    durable consumed index)."""

    def __init__(self, stale_s: float = 600.0, segment_len: Optional[int] = 64, burn_in: int = 16, settle: int = 2,
                 waiting_ttl_s: float = 3600.0, max_waiting: int = 200_000, trained_cap: int = 1_000_000):
        if segment_len is not None and segment_len < 1:
            raise ValueError("segment_len must be >= 1 or None")
        self.stale_s = stale_s
        self.segment_len = segment_len
        self.burn_in = max(0, int(burn_in))
        self.settle = max(1, int(settle))
        self.waiting_ttl_s, self.max_waiting, self.trained_cap = waiting_ttl_s, max_waiting, trained_cap
        self.open: dict = {}
        self.index: dict = {}                   # decision event_id -> trajectory key (decisions held in memory)
        self.waiting: OrderedDict = OrderedDict()   # attributed_to -> [(outcome record, t_arrived)]
        self.n_waiting = 0
        self.trained: OrderedDict = OrderedDict()   # decision event ids already emitted for training (FIFO bound)
        self.late: list = []
        self.untrainable: list = []             # [(event_id, reason, actor_id)] decisions that will never be trained
        self.counts = Counter()

    # ------------------------------------------------------------ input
    def _mark_trained(self, ids: Iterable[str]) -> None:
        for i in ids:
            self.trained[i] = None
        while len(self.trained) > self.trained_cap:
            self.trained.popitem(last=False)
            self.counts["trained_ids_pruned"] += 1

    def add(self, records: list, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        for r in records:
            key = (r.actor_id, r.episode_id)
            if r.action is not None:
                o = self.open.get(key)
                if o is None:
                    o = self.open[key] = _Open(key, r.env_id)
                if o.events and r.step < o.events[-1].step:
                    o.dirty = True
                o.events.append(r)
                o.t_last = now
                self.index[r.event_id] = key
                w = self.waiting.pop(r.event_id, None)
                if w is not None:
                    self.n_waiting -= len(w)
                    o.outcomes.setdefault(r.event_id, []).extend(x for x, _ in w)
                if r.done:
                    o.complete = True
                self.counts["decisions"] += 1
            elif r.attributed_to:
                self.counts["delayed"] += 1
                a = r.attributed_to
                k = self.index.get(a)
                if a in self.trained:
                    self.late.append(r)
                    self.counts["late_outcomes"] += 1
                elif k is not None and k in self.open:
                    self.open[k].outcomes.setdefault(a, []).append(r)
                    self.open[k].t_last = now
                else:
                    self.waiting.setdefault(a, []).append((r, now))
                    self.n_waiting += 1
            else:
                o = self.open.get(key)
                if (r.extra or {}).get("kind") == "episode_end":
                    self.counts["episode_end"] += 1
                    if o is not None:
                        o.complete = True
                        o.truncated = bool(r.extra.get("truncated"))
                        o.aux.append(r.event_id)
                    else:
                        self.counts["episode_end_without_open"] += 1
                elif o is not None:
                    o.world.append(r)
                    o.t_last = now

    def restore(self, records: list, context: dict, consumed_attr: set) -> None:
        """Rebuild open episodes after a learner restart. ``records``: unconsumed archive records in archive order;
        ``context``: (actor, episode) -> already-trained decisions (step order) to keep as burn-in context;
        ``consumed_attr``: decision ids already consumed (delayed outcomes attributed to them are late outcomes)."""
        for key, ctx in context.items():
            ctx = sorted(ctx, key=lambda e: e.step)[-self.burn_in:] if self.burn_in else []
            if ctx:
                o = self.open[key] = _Open(key, ctx[0].env_id)
                o.events = list(ctx)
                o.k = len(ctx)
                for e in ctx:
                    self.index[e.event_id] = key
                self._mark_trained(e.event_id for e in ctx)
        self._mark_trained(consumed_attr)
        self.add(records)

    # ------------------------------------------------------------ output
    def _emit(self, o: _Open, end: int, terminal: bool, truncated: bool, final: bool) -> Optional[Trajectory]:
        """Emit decisions [o.k, end) for training (+ the decision at ``end`` as bootstrap when not terminal)."""
        start = o.k
        if end <= start:
            return None
        lo = max(0, start - self.burn_in)
        boot = (not terminal) and end < len(o.events)
        hi = end + 1 if boot else end
        evs = o.events[lo:hi]
        mask = [lo + i >= start and lo + i < end for i in range(len(evs))]
        offered = o.events[start:end]
        outs = {e.event_id: o.outcomes.pop(e.event_id) for e in offered if e.event_id in o.outcomes}
        aux = [r.event_id for rs in outs.values() for r in rs] + [r.event_id for r in o.world] + \
            (list(o.aux) if final else [])
        tr = Trajectory(o.key[0], o.key[1], o.env_id, events=evs, outcomes=outs, world_events=list(o.world),
                        train_mask=mask, terminal=terminal, bootstrap=boot, truncated=truncated,
                        segment=o.segments, aux_ids=aux)
        o.world = []
        if final:
            o.aux = []
        o.segments += 1
        self._mark_trained(e.event_id for e in offered)
        o.k = end
        # forget decisions that can no longer be context (bounded memory for long lifetimes)
        cut = max(0, o.k - self.burn_in)
        if cut > 0 and not final:
            for e in o.events[:cut]:
                self.index.pop(e.event_id, None)
                o.outcomes.pop(e.event_id, None)
            o.events = o.events[cut:]
            o.k -= cut
        self.counts["segments"] += 1
        return tr

    def pop_ready(self, now: Optional[float] = None) -> list:
        now = time.time() if now is None else now
        out = []
        for key in list(self.open):
            o = self.open[key]
            if o.dirty:
                o.events.sort(key=lambda e: e.step)
                o.dirty = False
            if self.segment_len is not None:
                while len(o.events) - o.k >= self.segment_len + self.settle:
                    tr = self._emit(o, o.k + self.segment_len, terminal=False, truncated=False, final=False)
                    out.append(tr)
            stale = (not o.complete) and now - o.t_last > self.stale_s
            if not (o.complete or stale):
                continue
            if stale:
                o.truncated = True
                self.counts["stale_truncated"] += 1
            n = len(o.events)
            if o.truncated:
                end = n - self.settle                       # unsettled tail: delayed outcomes may be missing
                tr = self._emit(o, end, terminal=False, truncated=True, final=True)
                tail = o.events[max(end, o.k):]
                for e in tail:
                    self.untrainable.append((e.event_id, "truncated_unsettled", e.actor_id))
                self.counts["untrainable_truncated"] += len(tail)
            else:
                tr = self._emit(o, n, terminal=True, truncated=False, final=True)
            if tr is not None:
                out.append(tr)
            else:                                           # nothing trainable: still consume its other records
                self.untrainable.extend((i, "aux", o.key[0]) for i in o.aux + [r.event_id for r in o.world])
            for e in o.events:
                self.index.pop(e.event_id, None)
            # outcomes of the bootstrap / unsettled-tail decisions (never trained): consumed, counted, not lost
            leftover = [r.event_id for rs in o.outcomes.values() for r in rs]
            self.untrainable.extend((i, "aux", o.key[0]) for i in leftover)
            self.counts["outcomes_of_untrained_decisions"] += len(leftover)
            del self.open[key]
        return out

    def expire_waiting(self, now: Optional[float] = None) -> list:
        """Delayed outcomes whose decision never arrived (decision lost, shed, quarantined, or its id pruned from
        `trained`) older than ``waiting_ttl_s``, or the oldest beyond ``max_waiting``. Returned for the learner to
        resolve against its durable consumed index (late outcome) or log as orphans."""
        now = time.time() if now is None else now
        out = []
        while self.waiting:
            a, lst = next(iter(self.waiting.items()))
            if self.n_waiting <= self.max_waiting and now - lst[0][1] <= self.waiting_ttl_s:
                break
            self.waiting.popitem(last=False)
            self.n_waiting -= len(lst)
            out.extend(r for r, _ in lst)
        self.counts["waiting_expired"] += len(out)
        return out

    def take_late(self) -> list:
        out, self.late = self.late, []
        return out

    def take_untrainable(self) -> list:
        out, self.untrainable = self.untrainable, []
        return out

    def held_events(self) -> int:
        return sum(len(o.events) for o in self.open.values())


_CONSUMED_SCHEMA = """
CREATE TABLE IF NOT EXISTS consumed(event_id TEXT PRIMARY KEY, status TEXT, version INTEGER, ref TEXT);
CREATE INDEX IF NOT EXISTS consumed_version ON consumed(version);
"""


class CentralLearner:
    def __init__(self, run_dir: str, model, sink=None, max_lag: Optional[int] = None, lag_mode: str = "vtrace",
                 min_decisions: int = 64, max_batch_decisions: Optional[int] = None,
                 max_batch_trajectories: Optional[int] = None, publish_every: int = 1,
                 train_actors: Optional[list] = None, ingest_segments: int = 8, stale_s: float = 600.0,
                 max_backlog: Optional[int] = None, segment_len: Optional[int] = 64, burn_in: int = 16,
                 settle: int = 2, waiting_ttl_s: float = 3600.0,
                 queue_kwargs: Optional[dict] = None, keep_versions: int = 5, validate_fn=None,
                 record_traces: bool = True, meter: Optional[ComputeMeter] = None, resume: bool = True,
                 fsync: bool = True):
        self.run_dir = os.path.abspath(run_dir)
        self.paths = run_paths(self.run_dir)
        for d in self.paths.values():
            os.makedirs(d, exist_ok=True)
        self.model = model
        self.sink = sink
        self.queue = ExperienceQueue(self.paths["spool"], **(queue_kwargs or {}))
        self.consumer = self.queue.consumer()
        self.publisher = CheckpointPublisher(self.paths["ckpt"], keep_last=keep_versions, validate_fn=validate_fn,
                                             fsync=fsync)
        self.traces = TraceRecorder(self.paths["traces"], name="learner") if record_traces else None
        self.router = LagRouter(max_lag=max_lag, mode=lag_mode)
        self.assembler = TrajectoryAssembler(stale_s=stale_s, segment_len=segment_len, burn_in=burn_in,
                                             settle=settle, waiting_ttl_s=waiting_ttl_s)
        self.min_decisions = min_decisions
        self.max_batch_trajectories = max_batch_trajectories
        self.max_batch_decisions = max_batch_decisions
        self.max_backlog = max_backlog or 8 * max(min_decisions, max_batch_decisions or 0)
        self.publish_every = max(1, publish_every)
        self.train_actors = set(train_actors) if train_actors else None
        self.ingest_segments = ingest_segments
        self.meter = meter or ComputeMeter("learner", role="learner")
        self.updates_path = os.path.join(self.paths["learner"], "updates.jsonl")
        self.skipped_path = os.path.join(self.paths["learner"], "skipped.jsonl")
        self.orphans_path = os.path.join(self.paths["learner"], "orphans.jsonl")
        self.db = sqlite3.connect(os.path.join(self.paths["learner"], "consumed.sqlite"), isolation_level=None,
                                  timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(f"PRAGMA synchronous={'FULL' if fsync else 'OFF'}")
        self.db.executescript(_CONSUMED_SCHEMA)
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
            if self.publisher.last_version > self.version:       # cannot happen after the publisher's cleanup
                raise RuntimeError(f"publisher at {self.publisher.last_version} > LATEST {self.version}")
            self._recover_open()
        else:
            self.publish({"initial": True})

    # ---------------------------------------------------------------- consumed index
    def _consume(self, ids: Iterable[str], status: str, ref: str = "", version: Optional[int] = None) -> int:
        rows = [(i, status, self.version if version is None else version, ref) for i in ids]
        if rows:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.executemany("INSERT OR REPLACE INTO consumed VALUES (?,?,?,?)", rows)
            self.db.execute("COMMIT")
        return len(rows)

    def consumed(self, ids: list) -> set:
        """The subset of ``ids`` the learner has consumed (trained, skipped, or auxiliary)."""
        out = set()
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            q = "SELECT event_id FROM consumed WHERE event_id IN (%s)" % ",".join("?" * len(chunk))
            out.update(r[0] for r in self.db.execute(q, chunk))
        return out

    # ---------------------------------------------------------------- recovery
    def _recover_open(self) -> None:
        """Roll back consumption recorded after LATEST, then re-assemble every archived record the consumed index
        does not name (streamed chunk by chunk; see the module docstring)."""
        rb = self.db.execute("SELECT COUNT(*) FROM consumed WHERE version > ?", (self.version,)).fetchone()[0]
        if rb:
            vs = [r[0] for r in self.db.execute("SELECT DISTINCT version FROM consumed WHERE version > ?",
                                                (self.version,))]
            self.db.execute("DELETE FROM consumed WHERE version > ?", (self.version,))
            with open(self.updates_path, "a") as fh:
                fh.write(json.dumps({"rollback": True, "t": time.time(), "resumed_version": self.version,
                                     "rolled_back_versions": sorted(vs), "n_records": rb}) + "\n")
            self.counts["rolled_back_records"] = rb
        if self.sink is None:
            return
        fresh = []
        for chunk in self.sink.scan():
            cons = self.consumed([r.event_id for r in chunk])
            fresh.extend(r for r in chunk if r.event_id not in cons)
        fresh_ids = {r.event_id for r in fresh}
        consumed_attr = self.consumed(sorted({r.attributed_to for r in fresh if r.attributed_to}))
        context = {}
        for key in {(r.actor_id, r.episode_id) for r in fresh if r.action is not None}:
            prev = []
            try:
                it = self.sink.scan(actor_id=key[0], episode_id=key[1])
            except TypeError:
                it = self.sink.scan()
            for chunk in it:
                prev.extend(e for e in chunk if e.action is not None and (e.actor_id, e.episode_id) == key
                            and e.event_id not in fresh_ids)
            if prev:
                context[key] = prev
        self.assembler.restore(fresh, context, consumed_attr)
        self.counts["recovered_records"] = len(fresh)
        self.counts["recovered_context_decisions"] = sum(min(len(v), self.assembler.burn_in) for v in context.values())

    def _log_skipped(self, items: list, reason: Optional[str] = None) -> None:
        """items: [(event_id, reason, actor_id)] (or plain event ids with ``reason``)."""
        rows = [(x, reason, None) if isinstance(x, str) else x for x in items]
        if not rows:
            return
        by = defaultdict(list)
        for eid, why, actor in rows:
            by[why].append((eid, actor))
        with open(self.skipped_path, "a") as fh:
            for why, lst in by.items():
                fh.write(json.dumps({"t": time.time(), "version": self.version, "reason": why,
                                     "actors": sorted({a for _, a in lst if a}),
                                     "event_ids": [e for e, _ in lst]}) + "\n")
                self.counts[f"skipped:{why}"] += len(lst)
        for why, lst in by.items():
            self._consume([e for e, _ in lst], "aux" if why == "aux" else "skipped", why)

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
        """Consume up to ``ingest_segments`` segments. Returns the number of segments consumed or quarantined
        (0 when the spool is empty), so ``while ingest(): ...`` drains everything, including segments that held
        only traces or duplicates."""
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
                    if isinstance(v["value"], (int, float)) and not isinstance(v["value"], bool):
                        patch[v["key"]] = patch.get(v["key"], 0.0) + float(v["value"])
                self.traces.fill_outcome(patch, event_id=d.attributed_to, env_id=d.env_id, accumulate=True)
        self.meter.add(records=len(res.events))
        self.counts["ingested_events"] += len(res.events)
        self.counts["ingested_traces"] += len(res.traces)
        self.counts["dup"] += res.n_dup
        self.counts["quarantined"] += len(res.quarantined)
        return len(res.segments) + len(res.quarantined)

    def backlog(self) -> int:
        """Decisions ready for training. Open trajectories do not count: they are cut into segments as they grow,
        so their memory is bounded by (open episodes) x (segment_len + burn_in + settle)."""
        return sum(t.n for t in self.ready)

    def _expire_waiting(self) -> None:
        exp = self.assembler.expire_waiting()
        if not exp:
            return
        known = self.consumed(sorted({r.attributed_to for r in exp}))
        late = [r for r in exp if r.attributed_to in known]
        orphans = [r for r in exp if r.attributed_to not in known]
        self.assembler.late.extend(late)
        self.counts["late_outcomes_resolved"] += len(late)
        if orphans:
            with open(self.orphans_path, "a") as fh:
                for r in orphans:
                    fh.write(json.dumps({"t": time.time(), "event_id": r.event_id, "attributed_to": r.attributed_to,
                                         "actor_id": r.actor_id, "episode_id": r.episode_id,
                                         "reason": "decision never ingested"}) + "\n")
            self._consume([r.event_id for r in orphans], "orphan", "decision never ingested")
            self.counts["orphan_outcomes"] += len(orphans)

    def step(self) -> Optional[dict]:
        # End-to-end backpressure: stop consuming while the trainable backlog is full, so unconsumed segments stay in
        # the spool and the spool byte limit blocks (or sheds at) the actors instead of the learner's memory growing.
        if self.backlog() < self.max_backlog:
            self.ingest()
        else:
            self.counts["ingest_paused"] += 1
        self._expire_waiting()
        filtered = []
        for tr in self.assembler.pop_ready():
            if self.train_actors is not None and tr.actor_id not in self.train_actors:
                self.counts["not_trained_actor_filter"] += tr.n
                filtered.append(tr)
                continue
            self.ready.append(tr)
        un = self.assembler.take_untrainable()
        if un:
            self._log_skipped([x for x in un if x[1] != "aux"])
            self._consume([x[0] for x in un if x[1] == "aux"], "aux", "untrained episode records")
        if filtered:
            self._log_skipped([(i, "actor_filter", t.actor_id) for t in filtered for i in t.trained_ids()])
            self._consume([i for t in filtered for i in t.aux_ids], "aux", "actor_filter")
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
        dropped = [(eid, why, t.actor_id) for t in take for eid, why in t.dropped]
        if dropped:
            self._log_skipped(dropped)
        self.counts["discarded_trajectories"] += len(rep.discarded)
        self.counts["discarded_decisions"] += rep.n_dropped
        trainable = rep.on_policy + rep.off_policy
        lagsum = rep.summary()
        lagsum["decision_lags"] = lag_summary([l for t in trainable for l, m in zip(t.lags, t.train_mask) if m])
        if not trainable:
            self._consume([i for t in take for i in t.aux_ids], "aux", "discarded")
            self._log_update(None, take, lagsum, trained=False, reason="no trainable decision")
            return None
        late = self.assembler.take_late()
        batch = LearnerBatch(self.version, trainable, lagsum, late)
        t0 = time.perf_counter()
        with self.meter.gpu():
            metrics = dict(self.model.update(batch) or {})
        dt = time.perf_counter() - t0
        self._consume([i for t in take for i in t.aux_ids] + [r.event_id for r in late], "aux",
                      version=self.version + (0 if metrics.get("optimizer_step") is False else 1))
        if metrics.get("optimizer_step") is False:
            ids = [(i, "model_no_step", t.actor_id) for t in trainable for i in t.trained_ids()]
            self._log_skipped(ids)
            self.counts["model_no_step"] += 1
            self._log_update(metrics, trainable, lagsum, trained=False, reason="model made no optimiser step")
            return None
        model_skipped = set(metrics.pop("skipped_event_ids", None) or [])
        self.version += 1
        for t in trainable:
            if model_skipped:
                t.train_mask = [m and e.event_id not in model_skipped for e, m in zip(t.events, t.train_mask)]
        if model_skipped:
            self._log_skipped([(i, "model_skipped", None) for i in sorted(model_skipped)])
        n_dec = sum(t.n for t in trainable)
        n_on = sum(1 for t in trainable for l, m in zip(t.lags, t.train_mask) if m and l == 0)
        self.meter.add(updates=1, samples_trained=n_dec)
        self.counts["updates"] += 1
        self.counts["trained_decisions"] += n_dec
        self.counts["trained_on_policy"] += n_on
        self.counts["trained_off_policy"] += n_dec - n_on
        uid = f"u{self.version:08d}"
        ids = [i for t in trainable for i in t.trained_ids()]
        # durable consumption first, then publication: a crash after publishing never re-trains these decisions,
        # and a crash before it rolls their consumption back on restart (their version is above LATEST)
        self._consume(ids, "trained", uid)
        published = None
        if self.version % self.publish_every == 0:
            published = self.publish({"update": self.version, "metrics": metrics})
        return self._log_update(metrics, trainable, lagsum, trained=True, update_s=dt, uid=uid, ids=ids,
                                published=bool(published and published.get("validated")))

    def _log_update(self, metrics, trajs, lagsum, trained: bool, update_s: float = 0.0, reason: str = "",
                    uid: Optional[str] = None, ids: Optional[list] = None, published: bool = False):
        uid = uid or (f"u{self.version:08d}" if trained else f"skip{int(time.time() * 1e6)}")
        ids = ids if ids is not None else []
        by_actor = Counter()
        for t in trajs:
            by_actor[t.actor_id] += t.n
        rec = {"update_id": uid, "version": self.version, "trained": trained, "t": time.time(),
               "t_rel": time.time() - self.t_start, "update_s": update_s, "n_traj": len(trajs),
               "n_decisions": len(ids) if trained else sum(t.n for t in trajs), "by_actor": dict(by_actor),
               "lag": lagsum, "metrics": metrics or {}, "published": published, "reason": reason,
               "n_segments_mid_episode": sum(1 for t in trajs if t.bootstrap and not t.truncated),
               "event_ids": ids}
        with open(self.updates_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        if trained and self.traces is not None:
            by_env = defaultdict(list)
            for t in trajs:
                by_env[t.env_id].extend(t.trained_ids())
            for env_id, eids in by_env.items():
                self.traces.add_update_refs(eids, uid, env_id=env_id)
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
                "open_trajectories": len(self.assembler.open), "held_events": self.assembler.held_events(),
                "waiting_outcomes": self.assembler.n_waiting, "ready_trajectories": len(self.ready),
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
        self.db.close()
        if self.sink is not None and hasattr(self.sink, "close"):
            self.sink.close()


@dataclass
class LearnerSpec:
    run_dir: str
    model_factory: str = "superlearner.cam_unified.dist.toyworld:make_tiny_learner"
    model_kwargs: dict = field(default_factory=dict)
    sink_factory: Optional[str] = "superlearner.cam_unified.dist.sinks:make_jsonl_archive"
    sink_kwargs: dict = field(default_factory=dict)
    max_lag: Optional[int] = None       # None: the validated lag (offpolicy.validated_max_lag)
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
    segment_len: Optional[int] = 64
    burn_in: int = 16
    settle: int = 2
    queue_kwargs: dict = field(default_factory=dict)
    torch_threads: int = 2
    validate_factory: Optional[str] = None
    cuda: bool = False              # False: the process never sees the GPU (CUDA_VISIBLE_DEVICES=""), so a CPU-only
                                    # learner holds no CUDA context on the shared card (torch.optim's step otherwise
                                    # initialises one through its CUDA-graph capture check)

    def to_dict(self) -> dict:
        return asdict(self)


def learner_main(spec: dict, stop_event=None) -> None:
    """Process entry point (spawn)."""
    s = LearnerSpec(**spec)
    if not s.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # before any CUDA query in this process
    import torch
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
                       stale_s=s.stale_s, max_backlog=s.max_backlog, segment_len=s.segment_len, burn_in=s.burn_in,
                       settle=s.settle, queue_kwargs=s.queue_kwargs, validate_fn=validate)
    try:
        L.run(stop_event=stop_event, max_updates=s.max_updates, max_seconds=s.max_seconds)
    finally:
        L.close()
