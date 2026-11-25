"""Durable, versioned experience queue between actor processes and the central learner (spec 14.2, manifest F01).

Layout of a spool directory::

    spool/
      incoming/<commit_ns:020d>-<actor_id>-<seq:010d>.seg   committed segments waiting for the learner
      incoming/.<actor_id>-<seq:010d>.tmp                   a segment being written (never read by the learner)
      done/            segments the learner has ingested (audit trail; `keep_done=False` deletes instead)
      quarantine/      segments that failed their checksum or could not be decoded (+ ``.reason.json``)
      shed/<actor_id>.jsonl   ledger of segments an actor shed under the "shed" backpressure policy
      writers/<actor_id>.json writer state (next sequence number, last commit time)
      writers/<actor_id>.wal  write-ahead log: every buffered, not yet committed item (see below)
      writers/<actor_id>.open.json  episodes the actor has open (closed explicitly by its next incarnation)
      seen.sqlite      learner-side seen-set (event-id dedup), segment status and per-actor sequence accounting

**Segment format** (``cam-unified-seg/1``, JSON lines): a header line ``{"format", "actor_id", "seq",
"commit_ns", "n", "model_versions": [min, max], "kinds": {...}}``, one line per item ``{"k": "event"|"trace",
"d": <record dict>}`` and a footer line ``{"footer": true, "n": n, "sha256": <hex of the body bytes>}``. A segment
is written to a hidden ``.tmp`` file, flushed, ``fsync``-ed, then atomically renamed to ``.seg`` and the directory is
``fsync``-ed: the learner only ever sees complete segments, and the checksum catches anything else (copy errors,
disk corruption).

**Write-ahead log.** `SpoolWriter.put` encodes each item immediately and appends its line to the actor's ``.wal``
with one unbuffered ``write`` (no fsync), so an actor process killed at any point (SIGKILL included) loses nothing it
had buffered: the kernel keeps the written bytes, and the next writer for that actor id commits the WAL's complete
lines as a segment before anything else (a line torn by the kill itself — the kernel interrupted inside ``write`` — is
dropped and counted, ``wal_torn_lines``). The WAL is truncated after each committed (or shed) segment; a crash between
the commit and the truncation re-commits those items, which event-id dedup absorbs. A power loss can still lose the
WAL tail written since the last fsync-ed segment.

**Ordering.** File names sort by commit time, then actor and sequence, so the learner consumes segments in commit
order and each actor's segments in sequence order. Per actor the order is exact (commit times strictly increase per
writer). Across actors it is approximate: the commit time is taken after backpressure admission, immediately before
the segment is written and renamed, so two writers committing within the same few milliseconds may be consumed in
the opposite order of their renames. The consumer records the highest sequence seen per actor and reports gaps (e.g.
a quarantined or shed segment) instead of hiding them.

**Exactly-once ingestion into the sink.** For each ingest call the consumer (1) marks the segments ``claimed`` in
SQLite, (2) drops event ids already in the seen-set (and duplicates within the batch), (3) appends the new records to
the sink and flushes it, (4) in one transaction inserts the ids into the seen-set and marks the segments ``done``,
(5) moves the files to ``done/``. After a crash, ``claimed`` segments are reconciled against the sink (``sink.get``)
before anything is re-appended, ``done`` segments still in ``incoming/`` are just moved, and leftover ``.tmp`` files
of a writer are committed if their checksum is valid (the crash happened after ``fsync``) or quarantined otherwise.

**Backpressure** (``max_spool_bytes`` = bytes of committed, unconsumed segments): policy ``"block"`` makes the
writer wait (the actor stops stepping its environment) until the learner catches up, raising `SpoolFull` after
``block_timeout_s``; ``"raise"`` raises immediately; ``"shed"`` does not write the segment and appends its event ids,
size and reason to ``shed/<actor>.jsonl`` and to the writer's counters. Nothing is ever dropped silently. The limit is
checked without a cross-process lock, so N writers admitted at the same moment can overshoot it by at most N
segments (each at most ``batch_records`` items); size ``max_spool_bytes`` with that margin.

**Delayed outcomes** travel as ordinary `EventRecord` items whose ``attributed_to`` names the event id of the
responsible action; the consumer returns them separately (`IngestResult.delayed`). Every record carries the
``model_version`` of the policy that produced it, and the header carries the range.

**Beyond one machine.** The protocol only needs atomic rename within a directory and fsync: actors on other hosts can
write into a shared filesystem spool (NFS/Lustre rename is atomic within a directory), or into a local spool whose
``.seg`` files a shipper moves (copy to ``incoming/.x.tmp`` + rename) to the learner host; event-id dedup makes
re-shipping safe. Multi-GPU learners shard by consuming disjoint actor-id sets (`ingest(actors=...)`).
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..interfaces import (Action, ActionSource, DecisionTrace, EventRecord, EvidenceKind, ExecutionResult,
                          Observation, ObsKind, Provenance)
from ..observe.accounting import atomic_write_json

SEG_FORMAT = "cam-unified-seg/1"
_ID_RE = re.compile(r"^[A-Za-z0-9_.]+$")
_SEG_RE = re.compile(r"^(\d{20})-([A-Za-z0-9_.]+)-(\d{10})\.seg$")


class SpoolFull(RuntimeError):
    """The spool exceeded its byte limit and the writer's policy is "raise" (or "block" timed out)."""


class CorruptSegment(ValueError):
    """A segment failed its checksum or could not be decoded."""


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _defaults(cls) -> dict:
    out = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:      # type: ignore[misc]
            out[f.name] = f.default_factory()                    # type: ignore[misc]
    return out


_OBS_DEFAULTS = {k: (v.value if isinstance(v, enum.Enum) else v) for k, v in _defaults(Observation).items()}


_OBS_ITEMS = [(k, v) for k, v in _OBS_DEFAULTS.items() if k not in ("kind", "evidence")]
_OBSERVATIONAL = EvidenceKind.OBSERVATIONAL


def _obs_compact(o) -> dict:
    """Observation -> dict without default-valued fields (``Observation.from_dict`` restores them)."""
    od = o.__dict__
    d = {"kind": o.kind.value}
    for k, dv in _OBS_ITEMS:
        v = od[k]
        if v != dv:
            d[k] = v
    if o.evidence is not _OBSERVATIONAL:
        d["evidence"] = o.evidence.value if isinstance(o.evidence, enum.Enum) else o.evidence
    return d


def action_dict(a) -> dict:
    """Action -> JSON-ready dict (str-valued enums serialise as their values)."""
    d = {k: v for k, v in a.__dict__.items() if k[0] != "_"}
    if isinstance(a.source, enum.Enum):
        d["source"] = a.source.value
    return d


def encode_record(rec) -> dict:
    """Fast, compact JSON-ready dict of an EventRecord / DecisionTrace (a dict is returned as is). Equivalent to
    ``rec.to_dict()`` after ``from_dict`` (tested) without the recursive ``dataclasses.fields`` walk: observations
    drop default-valued fields, and nested values are left to ``json.dumps`` (the contract's enums are ``str``
    enums, which serialise as their values)."""
    if isinstance(rec, dict):
        return rec
    if isinstance(rec, EventRecord):
        d = {k: v for k, v in rec.__dict__.items() if k[0] != "_"}
        d["observations"] = [_obs_compact(o) for o in rec.observations]
        if rec.action is not None:
            d["action"] = action_dict(rec.action)
        if rec.result is not None:
            r = {k: v for k, v in rec.result.__dict__.items() if k[0] != "_"}
            r["observations"] = [_obs_compact(o) for o in rec.result.observations]
            d["result"] = r
        d["provenance"] = rec.provenance.value if isinstance(rec.provenance, enum.Enum) else rec.provenance
        d["evidence"] = rec.evidence.value if isinstance(rec.evidence, enum.Enum) else rec.evidence
        return d
    if isinstance(rec, DecisionTrace):
        d = {k: v for k, v in rec.__dict__.items() if k[0] != "_"}
        for k, v in d.items():
            if hasattr(v, "to_dict"):
                d[k] = v.to_dict()
        return d
    if hasattr(rec, "to_dict"):
        return rec.to_dict()
    raise TypeError(f"cannot encode {type(rec)}")


class _Item:
    """One encoded queue item: its JSON line (bytes, with the trailing newline) and what the header/ledger need."""
    __slots__ = ("kind", "line", "version", "rid")

    def __init__(self, kind: str, line: bytes, version: int, rid: str):
        self.kind, self.line, self.version, self.rid = kind, line, version, rid


def encode_item(k: str, rec) -> _Item:
    d = encode_record(rec)
    if k == "event":
        v, rid = int(d.get("model_version", 0)), d.get("event_id")
    elif k == "trace":
        v, rid = int(d.get("policy_version", 0)), d.get("decision_id")
    else:
        raise ValueError(f"unknown item kind {k!r}")
    return _Item(k, (json.dumps({"k": k, "d": d}, separators=(",", ":"), default=str) + "\n").encode(), v, rid)


def _item_from_line(line: bytes) -> _Item:
    o = json.loads(line)
    k, d = o["k"], o["d"]
    if k == "event":
        return _Item(k, line, int(d.get("model_version", 0)), d["event_id"])
    if k == "trace":
        return _Item(k, line, int(d.get("policy_version", 0)), d["decision_id"])
    raise ValueError(f"unknown item kind {k!r}")


def encode_items(actor_id: str, seq: int, commit_ns: int, items: list, extra_header: Optional[dict] = None) -> bytes:
    """Pre-encoded `_Item` list -> segment bytes."""
    kinds: dict = {}
    for it in items:
        kinds[it.kind] = kinds.get(it.kind, 0) + 1
    body = b"".join(it.line for it in items)
    versions = [it.version for it in items]
    header = {"format": SEG_FORMAT, "actor_id": actor_id, "seq": seq, "commit_ns": commit_ns, "n": len(items),
              "model_versions": [min(versions), max(versions)] if versions else None, "kinds": kinds}
    if extra_header:
        header.update(extra_header)
    footer = {"footer": True, "n": len(items), "sha256": hashlib.sha256(body).hexdigest()}
    return (json.dumps(header, separators=(",", ":")) + "\n").encode() + body + \
        (json.dumps(footer, separators=(",", ":")) + "\n").encode()


def encode_segment(actor_id: str, seq: int, commit_ns: int, items: list) -> bytes:
    """items: list of ("event", EventRecord | dict) / ("trace", DecisionTrace | dict) -> segment bytes."""
    return encode_items(actor_id, seq, commit_ns, [encode_item(k, rec) for k, rec in items])


_TRACE_PRE = b'{"k":"trace","d":'
_TRACE_HEAD = re.compile(rb'^\{"k":"trace","d":\{"decision_id":"([^"\\]+)","event_id":"([^"\\]+)",'
                         rb'"env_id":"((?:[^"\\]|\\.)*)","policy_version":(-?\d+),')


class WireTrace:
    """A DecisionTrace kept as the exact JSON it arrived as (the learner forwards traces to the recorder without
    decoding them); `load` returns the full DecisionTrace."""
    __slots__ = ("decision_id", "event_id", "env_id", "policy_version", "__dict__")

    def __init__(self, decision_id: str, event_id: str, env_id: str, policy_version: int, wire: str):
        self.decision_id, self.event_id, self.env_id = decision_id, event_id, env_id
        self.policy_version = policy_version
        self.__dict__["_wire"] = wire

    def load(self) -> DecisionTrace:
        return DecisionTrace(**json.loads(self.__dict__["_wire"]))


def decode_segment(data: bytes, lazy_traces: bool = False) -> tuple:
    """-> (header dict, [(kind, record dict | WireTrace, wire json | None)]). Raises CorruptSegment on any
    inconsistency. With ``lazy_traces`` trace lines are not JSON-decoded (their checksum is still verified)."""
    if not data.endswith(b"\n"):
        raise CorruptSegment("truncated (no final newline)")
    first = data.find(b"\n")
    last = data.rfind(b"\n", 0, len(data) - 1)
    if first < 0 or last < first:
        raise CorruptSegment("missing header or footer")
    try:
        header = json.loads(data[:first])
        footer = json.loads(data[last + 1:])
    except json.JSONDecodeError as e:
        raise CorruptSegment(f"bad header/footer json: {e}") from None
    if header.get("format") != SEG_FORMAT:
        raise CorruptSegment(f"unknown format {header.get('format')!r}")
    if not footer.get("footer"):
        raise CorruptSegment("missing footer")
    body = data[first + 1:last + 1]
    if hashlib.sha256(body).hexdigest() != footer.get("sha256"):
        raise CorruptSegment("checksum mismatch")
    items = []
    for line in body.splitlines():
        if not line:
            continue
        if lazy_traces and line.startswith(_TRACE_PRE):
            m = _TRACE_HEAD.match(line)
            if m is not None and line.endswith(b"}"):
                env_id = json.loads(b'"' + m.group(3) + b'"')
                items.append(("trace", WireTrace(m.group(1).decode(), m.group(2).decode(), env_id,
                                                 int(m.group(4)), line[len(_TRACE_PRE):-1].decode()), None))
                continue
        o = json.loads(line)
        items.append((o["k"], o["d"], _wire_of(line, o["k"])))
    if len(items) != footer.get("n") or len(items) != header.get("n"):
        raise CorruptSegment(f"item count {len(items)} != {footer.get('n')}")
    return header, items


def _wire_of(line: bytes, kind: str) -> Optional[str]:
    """The exact JSON text of the record inside an item line written by `encode_segment`
    (``{"k":"<kind>","d":<record>}``), or None if the line has another shape."""
    pre = b'{"k":"' + kind.encode() + b'","d":'
    if line.startswith(pre) and line.endswith(b"}"):
        return line[len(pre):-1].decode()
    return None


_OBSKIND = {e.value: e for e in ObsKind}
_EVIDENCE = {e.value: e for e in EvidenceKind}
_SOURCE = {e.value: e for e in ActionSource}
_PROV = {e.value: e for e in Provenance}


def _obs_from(d: dict) -> Observation:
    d["kind"] = _OBSKIND[d["kind"]]
    ev = d.get("evidence")
    d["evidence"] = _EVIDENCE[ev] if ev is not None else EvidenceKind.OBSERVATIONAL
    return Observation(**d)


def decode_event(d: dict) -> EventRecord:
    """Fast equivalent of ``EventRecord.from_dict`` for freshly parsed JSON (mutates ``d``; tested to produce
    identical records). Enum lookups go through value dicts instead of Enum calls."""
    d["observations"] = [_obs_from(o) for o in d.get("observations", [])]
    a = d.get("action")
    if a:
        a["source"] = _SOURCE[a.get("source", "cam_decoder")]
        d["action"] = Action(**a)
    else:
        d["action"] = None
    r = d.get("result")
    if r:
        r["observations"] = [_obs_from(o) for o in r.get("observations", [])]
        d["result"] = ExecutionResult(**r)
    else:
        d["result"] = None
    d["provenance"] = _PROV[d.get("provenance", "real")]
    d["evidence"] = _EVIDENCE[d.get("evidence", "controlled")]
    return EventRecord(**d)


def record_from_dict(kind: str, d, wire: Optional[str] = None):
    """Rebuild the record. ``wire`` (its exact JSON as it arrived) is kept on the object as ``_wire`` so sinks and
    trace recorders can store it without re-serialising; code that mutates a record must delete ``_wire``."""
    if isinstance(d, WireTrace):
        return d
    if kind == "event":
        r = decode_event(d)
    elif kind == "trace":
        r = DecisionTrace(**d)
    else:
        raise CorruptSegment(f"unknown item kind {kind!r}")
    if wire is not None:
        r.__dict__["_wire"] = wire
    return r


@dataclass
class IngestResult:
    events: list = field(default_factory=list)       # new EventRecords (deduplicated), in commit order
    traces: list = field(default_factory=list)       # new DecisionTraces (WireTrace objects when lazy)
    delayed: list = field(default_factory=list)      # subset of events with attributed_to set
    segments: list = field(default_factory=list)     # names of the segments ingested by this call
    n_items: int = 0
    n_dup: int = 0
    n_recovered_in_sink: int = 0                     # records found already in the sink during crash reconciliation
    quarantined: list = field(default_factory=list)  # [(name, reason)]

    @property
    def empty(self) -> bool:
        return not self.segments and not self.quarantined


class ExperienceQueue:
    """Handle on one spool directory; create `writer`s in actor processes and one `consumer` in the learner."""

    def __init__(self, root: str, max_spool_bytes: int = 1 << 30, policy: str = "block",
                 block_timeout_s: Optional[float] = None, fsync: bool = True, keep_done: bool = True,
                 poll_s: float = 0.01):
        if policy not in ("block", "shed", "raise"):
            raise ValueError(f"unknown backpressure policy {policy!r}")
        self.root = os.path.abspath(root)
        self.max_spool_bytes = int(max_spool_bytes)
        self.policy = policy
        self.block_timeout_s = block_timeout_s
        self.fsync = fsync
        self.keep_done = keep_done
        self.poll_s = poll_s
        self.incoming = os.path.join(self.root, "incoming")
        self.done = os.path.join(self.root, "done")
        self.quarantine = os.path.join(self.root, "quarantine")
        self.shed_dir = os.path.join(self.root, "shed")
        self.writers_dir = os.path.join(self.root, "writers")
        for d in (self.incoming, self.done, self.quarantine, self.shed_dir, self.writers_dir):
            os.makedirs(d, exist_ok=True)
        meta = os.path.join(self.root, "meta.json")
        if not os.path.exists(meta):
            atomic_write_json(meta, {"format": SEG_FORMAT, "created": time.time()})

    # ------------------------------------------------------------ views
    def pending(self) -> list:
        """Committed, unconsumed segment names in consumption order."""
        return sorted(n for n in os.listdir(self.incoming) if _SEG_RE.match(n))

    def spool_bytes(self) -> int:
        tot = 0
        with os.scandir(self.incoming) as it:
            for e in it:
                if e.name.endswith(".seg"):
                    try:
                        tot += e.stat().st_size
                    except FileNotFoundError:
                        pass
        return tot

    def writer(self, actor_id: str, batch_records: int = 256, policy: Optional[str] = None) -> "SpoolWriter":
        return SpoolWriter(self, actor_id, batch_records, policy or self.policy)

    def consumer(self, lazy_traces: bool = True) -> "SpoolConsumer":
        return SpoolConsumer(self, lazy_traces=lazy_traces)

    def shed_ledger(self, actor_id: Optional[str] = None) -> list:
        out = []
        for fn in sorted(os.listdir(self.shed_dir)):
            if actor_id is not None and fn != f"{actor_id}.jsonl":
                continue
            with open(os.path.join(self.shed_dir, fn)) as fh:
                out.extend(json.loads(l) for l in fh if l.strip())
        return out


class SpoolWriter:
    """Actor-side writer. `put` encodes and logs records (write-ahead log); a segment is committed every
    ``batch_records`` items and on `flush` (call it at trajectory boundaries). Not thread-safe; one writer per actor
    id at a time."""

    def __init__(self, q: ExperienceQueue, actor_id: str, batch_records: int, policy: str, wal: bool = True):
        if not _ID_RE.match(actor_id):
            raise ValueError(f"actor id must match {_ID_RE.pattern}: {actor_id!r}")
        self.q = q
        self.actor_id = actor_id
        self.batch_records = max(1, int(batch_records))
        self.policy = policy
        self.buf: list = []
        self.state_path = os.path.join(q.writers_dir, f"{actor_id}.json")
        self.wal_path = os.path.join(q.writers_dir, f"{actor_id}.wal")
        self.open_path = os.path.join(q.writers_dir, f"{actor_id}.open.json")
        st = {"next_seq": 0, "last_commit_ns": 0}
        if os.path.exists(self.state_path):
            with open(self.state_path) as fh:
                st.update(json.load(fh))
        self.next_seq = int(st["next_seq"])
        self.last_commit_ns = int(st["last_commit_ns"])
        self.stats = {"segments": 0, "items": 0, "bytes": 0, "blocked_s": 0.0, "shed_segments": 0, "shed_items": 0,
                      "recovered_tmp": 0, "quarantined_tmp": 0, "wal_recovered_items": 0, "wal_torn_lines": 0}
        self._prev_open: list = []
        try:
            with open(self.open_path) as fh:
                self._prev_open = [tuple(x) for x in json.load(fh).get("episodes", [])]
        except (OSError, json.JSONDecodeError):
            self._prev_open = []
        self._recover()
        self.wal_fd = os.open(self.wal_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644) if wal else None
        self._recover_wal()

    # ------------------------------------------------------------ crash recovery
    def _recover(self) -> None:
        prefix = f".{self.actor_id}-"
        for fn in sorted(os.listdir(self.q.incoming)):
            if not (fn.startswith(prefix) and fn.endswith(".tmp")):
                continue
            path = os.path.join(self.q.incoming, fn)
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
                header, _ = decode_segment(data)
                seq, cns = int(header["seq"]), int(header["commit_ns"])
                final = os.path.join(self.q.incoming, f"{cns:020d}-{self.actor_id}-{seq:010d}.seg")
                os.replace(path, final)
                self.next_seq = max(self.next_seq, seq + 1)
                self.last_commit_ns = max(self.last_commit_ns, cns)
                self.stats["recovered_tmp"] += 1
            except (CorruptSegment, json.JSONDecodeError, KeyError, ValueError) as e:
                dst = os.path.join(self.q.quarantine, fn)
                os.replace(path, dst)
                atomic_write_json(dst + ".reason.json", {"reason": f"incomplete tmp segment: {e}",
                                                        "actor_id": self.actor_id, "t": time.time()})
                self.stats["quarantined_tmp"] += 1
        if self.q.fsync:
            _fsync_dir(self.q.incoming)
        self._save_state()

    def _recover_wal(self) -> None:
        """Commit the items a previous writer for this actor had logged but not committed."""
        try:
            with open(self.wal_path, "rb") as fh:
                data = fh.read()
        except FileNotFoundError:
            return
        if not data:
            return
        lines = data.split(b"\n")
        tail = lines.pop()                       # bytes after the last newline: a line torn by the kill
        items = []
        for ln in lines:
            if not ln:
                continue
            try:
                items.append(_item_from_line(ln + b"\n"))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                self.stats["wal_torn_lines"] += 1
        if tail:
            self.stats["wal_torn_lines"] += 1
        if self.stats["wal_torn_lines"]:
            with open(os.path.join(self.q.shed_dir, f"{self.actor_id}.jsonl"), "a") as fh:
                fh.write(json.dumps({"actor_id": self.actor_id, "t": time.time(), "n": self.stats["wal_torn_lines"],
                                     "reason": "write-ahead log line torn by a process kill"}) + "\n")
        if items:
            self.buf = items
            self.stats["wal_recovered_items"] += len(items)
            self.flush(extra_header={"recovered_from_wal": True})
        elif self.wal_fd is not None:
            os.ftruncate(self.wal_fd, 0)

    def _save_state(self) -> None:
        atomic_write_json(self.state_path, {"next_seq": self.next_seq, "last_commit_ns": self.last_commit_ns},
                          fsync=self.q.fsync)

    # ------------------------------------------------------------ episodes left open
    def note_open_episodes(self, episodes: list) -> None:
        """Persist the actor's open episodes [(episode_id, env_id, incarnation)] (called at episode start/end)."""
        atomic_write_json(self.open_path, {"t": time.time(), "episodes": [list(e) for e in episodes]}, fsync=False)

    def previous_open_episodes(self) -> list:
        """Episodes the previous process for this actor id left open when it died: [(episode_id, env_id, inc)]."""
        out, self._prev_open = self._prev_open, []
        return out

    # ------------------------------------------------------------ writing
    def put(self, event: Optional[EventRecord] = None, trace: Optional[DecisionTrace] = None) -> Optional[str]:
        """Encode and log one event and/or trace; commits a segment when the buffer is full. Returns the committed
        segment name (or None)."""
        for k, rec in (("event", event), ("trace", trace)):
            if rec is None:
                continue
            it = encode_item(k, rec)
            if self.wal_fd is not None:
                os.write(self.wal_fd, it.line)
            self.buf.append(it)
        if len(self.buf) >= self.batch_records:
            return self.flush()
        return None

    def flush(self, extra_header: Optional[dict] = None) -> Optional[str]:
        """Commit the buffered items as one segment (no-op on an empty buffer). Applies backpressure."""
        if not self.buf:
            return None
        seq = self.next_seq
        nbytes = sum(len(it.line) for it in self.buf) + 512
        if not self._admit(nbytes, seq):
            return None
        cns = max(time.time_ns(), self.last_commit_ns + 1)   # taken after admission (see "Ordering")
        data = encode_items(self.actor_id, seq, cns, self.buf, extra_header)
        tmp = os.path.join(self.q.incoming, f".{self.actor_id}-{seq:010d}.tmp")
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            if self.q.fsync:
                os.fsync(fh.fileno())
        self.next_seq = seq + 1
        self.last_commit_ns = cns
        self._save_state()                 # a crash from here until the rename is recovered from the .tmp
        name = f"{cns:020d}-{self.actor_id}-{seq:010d}.seg"
        os.replace(tmp, os.path.join(self.q.incoming, name))
        if self.q.fsync:
            _fsync_dir(self.q.incoming)
        if self.wal_fd is not None:
            os.ftruncate(self.wal_fd, 0)   # committed: a crash before this line re-commits (dedup absorbs it)
        self.stats["segments"] += 1
        self.stats["items"] += len(self.buf)
        self.stats["bytes"] += len(data)
        self.buf = []
        return name

    def _admit(self, nbytes: int, seq: int) -> bool:
        lim = self.q.max_spool_bytes
        if self.q.spool_bytes() + nbytes <= lim:
            return True
        if self.policy == "raise":
            raise SpoolFull(f"spool {self.q.root} holds more than {lim} bytes")
        if self.policy == "shed":
            ids = [it.rid for it in self.buf]
            entry = {"actor_id": self.actor_id, "seq": seq, "n": len(self.buf), "bytes": nbytes, "t": time.time(),
                     "reason": f"spool over {lim} bytes", "ids": ids}
            with open(os.path.join(self.q.shed_dir, f"{self.actor_id}.jsonl"), "a") as fh:
                fh.write(json.dumps(entry) + "\n")
                fh.flush()
                if self.q.fsync:
                    os.fsync(fh.fileno())
            self.next_seq = seq + 1          # the consumer sees the gap and the ledger explains it
            self._save_state()
            if self.wal_fd is not None:
                os.ftruncate(self.wal_fd, 0)
            self.stats["shed_segments"] += 1
            self.stats["shed_items"] += len(self.buf)
            self.buf = []
            return False
        t0 = time.monotonic()                # "block"
        while self.q.spool_bytes() + nbytes > lim:
            if self.q.block_timeout_s is not None and time.monotonic() - t0 > self.q.block_timeout_s:
                self.stats["blocked_s"] += time.monotonic() - t0
                raise SpoolFull(f"blocked {self.q.block_timeout_s}s on a full spool {self.q.root}")
            time.sleep(self.q.poll_s)
        self.stats["blocked_s"] += time.monotonic() - t0
        return True

    def close(self) -> Optional[str]:
        name = self.flush()
        if self.wal_fd is not None:
            os.close(self.wal_fd)
            self.wal_fd = None
        return name


_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen(id TEXT PRIMARY KEY, segment TEXT, actor_id TEXT, model_version INTEGER);
CREATE TABLE IF NOT EXISTS segments(name TEXT PRIMARY KEY, actor_id TEXT, seq INTEGER, status TEXT,
    n_items INTEGER, n_new INTEGER, n_dup INTEGER, t_claim REAL, t_done REAL, reason TEXT);
CREATE TABLE IF NOT EXISTS actors(actor_id TEXT PRIMARY KEY, max_seq INTEGER, n_segments INTEGER, missing TEXT);
"""


class SpoolConsumer:
    """Learner-side consumer with a durable SQLite seen-set. One consumer per spool."""

    def __init__(self, q: ExperienceQueue, lazy_traces: bool = True):
        self.q = q
        self.lazy_traces = lazy_traces
        self.db = sqlite3.connect(os.path.join(q.root, "seen.sqlite"), isolation_level=None, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(f"PRAGMA synchronous={'FULL' if q.fsync else 'OFF'}")
        self.db.executescript(_SCHEMA)
        self.totals = {"segments": 0, "items": 0, "events": 0, "traces": 0, "dup": 0, "delayed": 0,
                       "quarantined": 0, "recovered_in_sink": 0}

    # ------------------------------------------------------------ helpers
    def _status(self, name: str) -> Optional[str]:
        r = self.db.execute("SELECT status FROM segments WHERE name=?", (name,)).fetchone()
        return r[0] if r else None

    def _seen(self, ids: list) -> set:
        out = set()
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            q = "SELECT id FROM seen WHERE id IN (%s)" % ",".join("?" * len(chunk))
            out.update(r[0] for r in self.db.execute(q, chunk))
        return out

    def _move_done(self, name: str) -> None:
        src = os.path.join(self.q.incoming, name)
        if not os.path.exists(src):
            return
        if self.q.keep_done:
            os.replace(src, os.path.join(self.q.done, name))
        else:
            os.remove(src)

    def _quarantine(self, name: str, reason: str) -> None:
        src = os.path.join(self.q.incoming, name)
        dst = os.path.join(self.q.quarantine, name)
        if os.path.exists(src):
            os.replace(src, dst)
        atomic_write_json(dst + ".reason.json", {"reason": reason, "t": time.time()})
        m = _SEG_RE.match(name)
        self.db.execute("INSERT OR REPLACE INTO segments(name, actor_id, seq, status, reason) VALUES (?,?,?,?,?)",
                        (name, m.group(2) if m else None, int(m.group(3)) if m else None, "quarantined", reason))
        self.totals["quarantined"] += 1

    # ------------------------------------------------------------ ingestion
    def ingest(self, sink=None, trace_sink=None, max_segments: int = 16, actors: Optional[Iterable[str]] = None,
               admit=None) -> IngestResult:
        """Consume up to ``max_segments`` committed segments in order.

        sink: ArchiveP (``append``/``flush``/``get``) receiving the new, deduplicated EventRecords;
        trace_sink: object with ``record(trace)`` (e.g. observe.trace.TraceRecorder) receiving new DecisionTraces;
        actors: restrict to these actor ids (learner sharding); admit: optional predicate on EventRecord deciding
        which records go to the sink (all records are still marked seen and returned).
        """
        res = IngestResult()
        names = self.q.pending()
        if actors is not None:
            allowed = set(actors)
            names = [n for n in names if _SEG_RE.match(n).group(2) in allowed]
        names = names[:max_segments]
        batch, claimed, claim_rows = [], [], []
        for name in names:
            st = self._status(name)
            if st == "done":                      # crash after commit, before the move
                self._move_done(name)
                continue
            try:
                with open(os.path.join(self.q.incoming, name), "rb") as fh:
                    header, items = decode_segment(fh.read(), lazy_traces=self.lazy_traces)
                recs = [(k, record_from_dict(k, d, w)) for k, d, w in items]
            except FileNotFoundError:
                continue
            except (CorruptSegment, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                self._quarantine(name, f"{type(e).__name__}: {e}")
                res.quarantined.append((name, str(e)))
                continue
            m = _SEG_RE.match(name)
            if st is None:
                claim_rows.append((name, m.group(2), int(m.group(3)), "claimed", len(recs), time.time()))
            batch.append((name, header, recs, st == "claimed"))
            claimed.append(name)
        if not batch:
            return res
        if claim_rows:                                  # one durable transaction claims the whole batch
            self.db.execute("BEGIN IMMEDIATE")
            self.db.executemany("INSERT INTO segments(name, actor_id, seq, status, n_items, t_claim) "
                                "VALUES (?,?,?,?,?,?)", claim_rows)
            self.db.execute("COMMIT")

        # ---- dedup against the seen-set and within the batch
        keys = []
        for _, _, recs, _ in batch:
            for k, r in recs:
                keys.append(r.event_id if k == "event" else "trace:" + r.decision_id)
        already = self._seen(keys)
        fresh_keys: set = set()
        new_events, new_traces, rows = [], [], []
        in_sink: set = set()
        per_seg = {}
        for name, header, recs, was_claimed in batch:
            n_new = n_dup = 0
            seg_events = []
            for k, r in recs:
                key = r.event_id if k == "event" else "trace:" + r.decision_id
                if key in already or key in fresh_keys:
                    n_dup += 1
                    continue
                fresh_keys.add(key)
                n_new += 1
                if k == "event":
                    seg_events.append(r)
                    rows.append((key, name, r.actor_id, int(r.model_version)))
                else:
                    new_traces.append(r)
                    rows.append((key, name, header["actor_id"], int(r.policy_version)))
            if was_claimed and sink is not None and seg_events:
                # crash reconciliation: records that reached the sink before the crash are not appended again
                present = {e.event_id for e in sink.get([e.event_id for e in seg_events]) if e is not None}
                res.n_recovered_in_sink += len(present)
                in_sink.update(present)
            new_events.extend(seg_events)
            per_seg[name] = (n_new, n_dup)
            res.n_items += len(recs)
            res.n_dup += n_dup

        # ---- sink first (durable), then the seen-set and segment status in one transaction
        to_sink = [e for e in new_events if e.event_id not in in_sink]
        if admit is not None:
            to_sink = [e for e in to_sink if admit(e)]
        if sink is not None and to_sink:
            sink.append(to_sink)
            sink.flush()
        if trace_sink is not None:
            for t in new_traces:
                trace_sink.record(t)
            if hasattr(trace_sink, "flush"):
                trace_sink.flush()
        now = time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.executemany("INSERT OR IGNORE INTO seen(id, segment, actor_id, model_version) VALUES (?,?,?,?)",
                                rows)
            for name, header, _, _ in batch:
                n_new, n_dup = per_seg[name]
                self.db.execute("UPDATE segments SET status='done', n_new=?, n_dup=?, t_done=? WHERE name=?",
                                (n_new, n_dup, now, name))
                self._note_seq(header["actor_id"], int(header["seq"]))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        for name in claimed:
            self._move_done(name)
        res.events = new_events
        res.traces = new_traces
        res.delayed = [e for e in new_events if e.attributed_to]
        res.segments = claimed
        t = self.totals
        t["segments"] += len(claimed)
        t["items"] += res.n_items
        t["events"] += len(new_events)
        t["traces"] += len(new_traces)
        t["dup"] += res.n_dup
        t["delayed"] += len(res.delayed)
        t["recovered_in_sink"] += res.n_recovered_in_sink
        return res

    def _note_seq(self, actor_id: str, seq: int) -> None:
        r = self.db.execute("SELECT max_seq, n_segments, missing FROM actors WHERE actor_id=?", (actor_id,)).fetchone()
        if r is None:
            missing = list(range(0, seq))
            self.db.execute("INSERT INTO actors VALUES (?,?,?,?)", (actor_id, seq, 1, json.dumps(missing)))
            return
        max_seq, n, missing = r[0], r[1], json.loads(r[2])
        if seq > max_seq:
            missing.extend(range(max_seq + 1, seq))
            max_seq = seq
        elif seq in missing:
            missing.remove(seq)
        self.db.execute("UPDATE actors SET max_seq=?, n_segments=?, missing=? WHERE actor_id=?",
                        (max_seq, n + 1, json.dumps(missing), actor_id))

    # ------------------------------------------------------------ accounting
    def gaps(self) -> dict:
        """actor id -> sequence numbers never ingested (shed, quarantined or still in flight)."""
        return {a: json.loads(m) for a, m in self.db.execute("SELECT actor_id, missing FROM actors") if json.loads(m)}

    def seen_count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM seen WHERE id NOT LIKE 'trace:%'").fetchone()[0])

    def stats(self) -> dict:
        by_status = dict(self.db.execute("SELECT status, COUNT(*) FROM segments GROUP BY status").fetchall())
        actors = {a: {"max_seq": s, "segments": n} for a, s, n, _ in self.db.execute("SELECT * FROM actors")}
        return {"totals": dict(self.totals), "segments_by_status": by_status, "actors": actors, "gaps": self.gaps(),
                "pending": len(self.q.pending()), "spool_bytes": self.q.spool_bytes(), "seen_events": self.seen_count()}

    def close(self) -> None:
        self.db.close()
