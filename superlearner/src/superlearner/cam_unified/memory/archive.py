"""Durable append-only lifetime event archive (implements ``ArchiveP``; spec Part 10.2, architecture §8).

There is no lifetime ingestion cap anywhere in this module: the only bound is an explicit byte budget
(``archive_max_bytes``), and reaching it raises ``StorageLimitExceeded`` after the cold-archive policy ran. Records
are never dropped silently, never evicted, and never truncated to a fixed number of rows.

Budget: ``max_bytes`` bounds everything under ``root`` (sealed chunks, WAL including its unflushed buffer, tag logs,
SQLite index and journal, manifest, and any other file placed there, e.g. the replay sampler's database). The default
cold policy only recompresses (``("compress",)``). Moving chunks ("move") is allowed only to an explicitly configured
``cold_dir`` outside ``root`` with its own explicit ``cold_max_bytes``; chunks there count against that budget instead.
Transient extra: one temporary chunk file while a chunk is being sealed or recompressed.

On-disk layout (``root/``)::

    manifest.json              committed chunk list (the commit record): number, file, rows, bytes, sha256, first gseq,
                               tier ("hot" = fast codec, lz4 by default; "z" = recompressed with strong zstd),
                               location ("root" | "cold")
    chunks/chunk_<n>.<tier>.parquet   sealed, immutable Parquet chunks of at most ``chunk_rows`` events and about
                               ``chunk_bytes`` of WAL bytes (row groups of ``row_group_rows``): flat indexed columns +
                               JSON-encoded nested fields; "z" tier = zstd level ``cold_compression_level``
    wal/wal_<gseq0>.jsonl      write-ahead log of the not-yet-sealed tail (durable after ``flush()``); the tail held in
                               memory is bounded by ``chunk_rows`` rows and ``chunk_bytes`` bytes
    tags/tags_<k>.jsonl        append-only log of later tag updates and training-use marks (durable after ``flush()``)
    index.sqlite               secondary index (rebuildable from chunks + WAL + tag logs)
    cold/                      internal cold directory (counted in ``max_bytes``); an external ``cold_dir`` lives
                               on another volume

Write protocol of a chunk: Parquet to a temp file -> fsync -> atomic rename -> fsync(dir) -> index transaction ->
manifest (temp -> fsync -> rename) -> WAL rotation. Restart recovery: temp files are removed; a renamed chunk the
manifest does not yet list is validated and adopted (it was complete before its rename); missing index entries are
rebuilt from the chunk; the WAL tail is replayed (a torn last line from a crash mid-write is ignored); if the SQLite
index is lost or corrupt it is rebuilt from chunks, tag logs and WAL.

Single writer per archive (an ``flock`` on ``root/.writer.lock``); any number of ``readonly=True`` readers.

Producer tag contract (flat, indexed, used by the replay sampler and the memory filters):

    reward       EventRecord.reward
    failed       EventRecord.extra["failure"] truthy, or result present and ``result.ok`` False
    success      EventRecord.extra["success"] (bool; None = unknown)
    surprise     EventRecord.extra["surprise"] or EventRecord.uncertainty["surprise"] (float; None = unknown)
    novelty      EventRecord.extra["novelty"] (float)
    innovation   EventRecord.extra["innovation"] (bool)
    family       EventRecord.extra["family"] (task / mechanism family; never a model input)
    skill        EventRecord.extra["skill"] (name of a skill / option exercised successfully)
    entities     EventRecord.entity_links (str or {"id"/"entity": str}) + Observation.entity / Observation.target

Tags known only later (episode success, hindsight surprise) are set with ``tag()``; the learner records which updates
consumed an event with ``mark_used()`` (merged into ``EventRecord.training_provenance["used_in_updates"]``).
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import sqlite3
import time
from collections import OrderedDict
from typing import Iterable, Iterator, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ..interfaces import EventRecord, Provenance
from ._fs import atomic_write_json, fsync_dir, fsync_file, sha256_file
from .errors import ArchiveCorrupted, ArchiveLocked, StorageLimitExceeded

ARCHIVE_FORMAT = 1

_FLAT = [
    ("gseq", pa.int64()), ("event_id", pa.string()), ("env_id", pa.string()), ("actor_id", pa.string()),
    ("episode_id", pa.string()), ("step", pa.int64()), ("model_version", pa.int64()), ("t_sim", pa.float64()),
    ("t_wall", pa.float64()), ("provenance", pa.string()), ("evidence", pa.string()), ("goal_id", pa.string()),
    ("reward", pa.float64()), ("cost", pa.float64()), ("done", pa.bool_()), ("behavior_logp", pa.float64()),
    ("randomization_p", pa.float64()), ("attributed_to", pa.string()), ("tool", pa.string()), ("ok", pa.bool_()),
    ("failed", pa.bool_()), ("success", pa.int8()), ("surprise", pa.float64()), ("novelty", pa.float64()),
    ("family", pa.string()), ("innovation", pa.bool_()), ("skill", pa.string()), ("entities", pa.list_(pa.string())),
]
_JSON = ["observations", "action", "result", "predicted", "uncertainty", "entity_links", "simultaneous",
         "training_provenance", "extra"]
SCHEMA = pa.schema(_FLAT + [(c, pa.string()) for c in _JSON])
COLUMNS = [n for n, _ in _FLAT] + _JSON

# columns of the SQLite index (everything flat except evidence / behaviour / attribution / entities)
_IDX = ["gseq", "event_id", "chunk", "row", "env_id", "actor_id", "episode_id", "step", "model_version", "t_sim",
        "t_wall", "provenance", "goal_id", "reward", "cost", "done", "tool", "ok", "failed", "success", "surprise",
        "novelty", "family", "innovation", "skill"]
TAG_KEYS = ("surprise", "success", "novelty", "failed", "innovation", "skill", "family")

_EQ_FILTERS = {"env_id", "actor_id", "episode_id", "goal_id", "family", "tool", "provenance", "skill", "model_version",
               "chunk", "event_id"}
_BOOL_FILTERS = {"success", "failed", "done", "innovation", "ok"}
_RANGE_FILTERS = {"t_min": ("t_sim", ">="), "t_max": ("t_sim", "<="), "gseq_min": ("gseq", ">="),
                  "gseq_max": ("gseq", "<="), "step_min": ("step", ">="), "step_max": ("step", "<="),
                  "model_version_min": ("model_version", ">="), "model_version_max": ("model_version", "<="),
                  "reward_min": ("reward", ">="), "reward_max": ("reward", "<="), "surprise_min": ("surprise", ">="),
                  "novelty_min": ("novelty", ">="), "t_wall_min": ("t_wall", ">="), "t_wall_max": ("t_wall", "<=")}
FILTER_KEYS = frozenset(_EQ_FILTERS | _BOOL_FILTERS | set(_RANGE_FILTERS) | {"entity", "event_ids"})


def _jd(v) -> str:
    return json.dumps(v, separators=(",", ":"), default=str)


def _enum_val(v):
    return getattr(v, "value", v)


def event_entities(rec: EventRecord) -> list:
    """Entity ids an event is about (entity links + the entities of its observations), order-preserving, unique."""
    out, seen = [], set()

    def add(e):
        if isinstance(e, dict):
            e = e.get("id", e.get("entity"))
        if e is None:
            return
        e = str(e)
        if e not in seen:
            seen.add(e)
            out.append(e)
    for e in rec.entity_links or []:
        add(e)
    for o in rec.observations or []:
        add(getattr(o, "entity", None) if not isinstance(o, dict) else o.get("entity"))
        add(getattr(o, "target", None) if not isinstance(o, dict) else o.get("target"))
    return out


def event_tags(rec: EventRecord) -> dict:
    """Flat replay/filter tags of a record (see the module docstring for the producer contract)."""
    ex = rec.extra or {}
    unc = rec.uncertainty or {}
    res = rec.result
    ok = None if res is None else bool(res.ok)
    failed = bool(ex.get("failure")) or (ok is False)
    succ = ex.get("success")
    surprise = ex.get("surprise", unc.get("surprise"))
    novelty = ex.get("novelty")
    return {"ok": ok, "failed": failed, "success": None if succ is None else int(bool(succ)),
            "surprise": None if surprise is None else float(surprise),
            "novelty": None if novelty is None else float(novelty),
            "family": str(ex.get("family") or ""), "innovation": bool(ex.get("innovation", False)),
            "skill": str(ex.get("skill") or "")}


class EventArchive:
    """Append-only, chunked, restart-safe event archive over Parquet + SQLite (see module docstring).

    Args:
        root: archive directory.
        cfg: ``CamConfig`` supplying ``archive_chunk_rows`` and ``archive_max_bytes`` (overridable below).
        chunk_rows: events per sealed Parquet chunk.
        chunk_bytes: also seal once the buffered tail holds this many WAL bytes (bounds the writer's memory for large
            records; chunks then hold fewer than ``chunk_rows`` rows).
        max_bytes: explicit budget of everything under ``root`` (see the module docstring).
        cold_policy: ordered actions tried when a write would exceed ``max_bytes``: ``"compress"`` (rewrite the oldest
            hot chunks with ``cold_compression_level`` zstd, in place) and/or ``"move"`` (move the oldest chunks to
            the external ``cold_dir``). ``()`` = raise immediately. Data is never deleted.
        cold_dir: cold volume (default ``root/cold``, internal and inside the budget). "move" requires a ``cold_dir``
            outside ``root`` and an explicit ``cold_max_bytes`` (its own budget).
        artifacts: optional ``ArtifactStore``; observation content longer than ``inline_limit`` characters is stored
            there verbatim and the archived observation keeps a truncated copy plus ``artifact_ref``.
        allow_counterfactual: accept ``Provenance.COUNTERFACTUAL`` (explicitly declared experimental conditions only).
            ``Provenance.IMAGINED`` is always rejected.
        autoflush: fsync the WAL after every ``append`` (actors); otherwise durability point = ``flush()``.
        readonly: open without the writer lock and without any recovery writes.
    """

    def __init__(self, root: str, cfg=None, *, chunk_rows: Optional[int] = None, max_bytes: Optional[int] = None,
                 cold_policy: Sequence[str] = ("compress",), cold_dir: Optional[str] = None,
                 cold_max_bytes: Optional[int] = None, artifacts=None, inline_limit: int = 16384,
                 allow_counterfactual: bool = False, autoflush: bool = False, readonly: bool = False,
                 row_group_rows: int = 2048, compression: str = "lz4", compression_level: Optional[int] = None,
                 cold_compression_level: int = 19, index_reserve_bytes: int = 1_500_000,
                 fsync: bool = True, chunk_bytes: int = 32 << 20):
        self.root = os.path.abspath(root)
        self._chunk_rows_arg = chunk_rows or (cfg.archive_chunk_rows if cfg is not None else None)
        self.chunk_rows = int(self._chunk_rows_arg or 65536)
        self.chunk_bytes = int(chunk_bytes)
        self.max_bytes = max_bytes if max_bytes is not None else (cfg.archive_max_bytes if cfg is not None else None)
        if isinstance(cold_policy, str):
            cold_policy = (cold_policy,)
        for p in cold_policy:
            if p not in ("compress", "move"):
                raise ValueError(f"unknown cold policy {p!r}")
        self.cold_policy = tuple(cold_policy)
        self.cold_max_bytes = cold_max_bytes
        self.artifacts = artifacts
        self.inline_limit = int(inline_limit)
        self.allow_counterfactual = allow_counterfactual
        self.autoflush = autoflush
        self.readonly = readonly
        self.row_group_rows = int(row_group_rows)
        self.compression = compression
        self.compression_level = compression_level
        self.cold_compression_level = cold_compression_level
        self.index_reserve_bytes = int(index_reserve_bytes)
        self.do_fsync = fsync
        self.stats = {"appended": 0, "duplicates": 0, "rejected": 0, "sealed_chunks": 0, "compressed_chunks": 0,
                      "moved_chunks": 0, "freed_bytes": 0, "seal_deferred": 0}
        self.recovery = {"removed_tmp": 0, "adopted_chunks": 0, "quarantined": 0, "reindexed_chunks": 0,
                         "wal_rows": 0, "wal_torn_lines": 0, "index_rebuilt": False, "dropped_duplicate_files": 0}
        self._lock_fh = None
        self._wal_fh = None
        self._wal_written = 0
        self._tag_written = 0
        self._tags_closed_bytes = 0
        self._rebuild_tags = False
        self._tag_fh = None
        self._pf_cache: "OrderedDict[int, pq.ParquetFile]" = OrderedDict()
        self._buf = {c: [] for c in COLUMNS}
        self._buf_sizes: list = []
        self._buf_bytes = 0
        self._buf_g0 = 0
        for d in ("chunks", "wal", "tags"):
            if not readonly:
                os.makedirs(os.path.join(self.root, d), exist_ok=True)
        if not readonly:
            self._lock_fh = open(os.path.join(self.root, ".writer.lock"), "a+")
            try:
                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self._lock_fh.close()
                self._lock_fh = None
                raise ArchiveLocked(f"another process writes to {self.root}")
        self._manifest = self._load_manifest()
        if self._chunk_rows_arg is None and self._manifest.get("chunk_rows"):
            self.chunk_rows = int(self._manifest["chunk_rows"])      # reopened without explicit sizing
        self._manifest.setdefault("chunk_rows", self.chunk_rows)
        self.cold_dir = os.path.abspath(cold_dir or self._manifest.get("cold_dir") or os.path.join(self.root, "cold"))
        self.cold_external = not (self.cold_dir == self.root or self.cold_dir.startswith(self.root + os.sep))
        if "move" in self.cold_policy:
            if not self.cold_external:
                raise ValueError("cold policy 'move' needs cold_dir on a separate volume outside the archive root "
                                 "(moving inside root frees nothing from the budget)")
            if self.cold_max_bytes is None:
                raise ValueError("cold policy 'move' needs an explicit cold_max_bytes budget for the cold volume")
        if not readonly:
            os.makedirs(self.cold_dir, exist_ok=True)
            self._manifest["cold_dir"] = self.cold_dir
        self._recover_files()
        self._open_index()
        self._replay_wal()
        if not readonly:
            self._open_logs()

    # ================================================================== public API (ArchiveP)
    def append(self, records: Iterable[EventRecord]) -> None:
        """Append records (all-or-nothing per call). Rejects non-REAL provenance with ``ValueError``; skips event ids
        already archived (idempotent re-delivery, counted in ``stats['duplicates']``); raises
        ``StorageLimitExceeded`` (nothing accepted) when the budget is exhausted even after the cold policy."""
        self._writable()
        records = list(records)
        for r in records:
            if not isinstance(r, EventRecord):
                raise TypeError(f"EventArchive.append takes EventRecord, got {type(r).__name__}")
            prov = Provenance(_enum_val(r.provenance))
            if prov is Provenance.IMAGINED or (prov is Provenance.COUNTERFACTUAL and not self.allow_counterfactual):
                self.stats["rejected"] += len(records)
                raise ValueError(f"archive accepts only REAL experience; event {r.event_id} has provenance {prov.value}")
        seen, fresh = set(), []
        known = self._known_ids([r.event_id for r in records])
        for r in records:
            if r.event_id in seen or r.event_id in known:
                self.stats["duplicates"] += 1
                continue
            seen.add(r.event_id)
            fresh.append(r)
        if not fresh:
            return
        g = self._next
        rows = [self._row(r, g + i) for i, r in enumerate(fresh)]
        lines = [_jd(row) + "\n" for row in rows]
        need = sum(len(x) for x in lines)
        self._ensure_room(need + len(rows) * self._index_row_bytes())
        self._wal_fh.write("".join(lines))
        self._wal_written += need
        for row in rows:
            for c in COLUMNS:
                self._buf[c].append(row[c])
        self._buf_sizes.extend(len(x) for x in lines)
        self._buf_bytes += need
        self._index_rows(rows, chunk=-1, g0=None)
        self._db.commit()
        self._next += len(rows)
        self.stats["appended"] += len(rows)
        if self.autoflush:
            self.flush()
        while self._buf_len() and (self._buf_len() >= self.chunk_rows or self._buf_bytes >= self.chunk_bytes):
            try:
                self._seal(min(self.chunk_rows, self._buf_len()))
            except StorageLimitExceeded:
                # the rows of this call are accepted and durable in the WAL (they fitted the budget); sealing is
                # retried on the next append, and the next append that does not fit raises
                self.stats["seal_deferred"] += 1
                break

    def flush(self) -> None:
        """Durability point: fsync the WAL and the tag log (everything appended so far survives a crash)."""
        if self.readonly:
            return
        for fh in (self._wal_fh, self._tag_fh):
            if fh is not None:
                fh.flush()
                if self.do_fsync:
                    os.fsync(fh.fileno())

    def seal(self) -> None:
        """Write the whole buffered tail as a (possibly short) Parquet chunk."""
        self._writable()
        if self._buf_len():
            self._seal(self._buf_len())

    def count(self) -> int:
        """Total archived events (sealed + WAL tail)."""
        return self._next

    def get(self, event_ids: Sequence[str]) -> list:
        """Records in the order of ``event_ids``; ``KeyError`` lists ids that were never archived."""
        event_ids = list(event_ids)
        loc = {}
        for part in _batches(event_ids, 900):
            q = f"SELECT event_id, gseq, chunk, row FROM events WHERE event_id IN ({','.join('?' * len(part))})"
            for eid, g, c, r in self._db.execute(q, part):
                loc[eid] = (g, c, r)
        missing = [e for e in event_ids if e not in loc]
        if missing:
            raise KeyError(f"{len(missing)} event ids not in the archive, e.g. {missing[:3]}")
        rows = self._fetch([loc[e] for e in event_ids], OrderedDict())
        return self._records(rows)

    def scan(self, batch_size: int = 1024, **filters) -> Iterator[list]:
        """Stream matching records in archive order (gseq) in lists of at most ``batch_size``.

        Memory is bounded by ``batch_size`` records plus at most two decoded row groups; the lifetime is never loaded.
        Filters (combined with AND; list values mean IN): env_id, actor_id, episode_id, goal_id, family, tool,
        provenance, skill, model_version, chunk, event_id; booleans success, failed, done, innovation, ok; ranges
        t_min/t_max (t_sim), gseq_min/gseq_max, step_min/step_max, model_version_min/max, reward_min/max,
        surprise_min, novelty_min, t_wall_min/max; entity (str or list); event_ids (list). Unknown keys raise."""
        where, params = self._where(filters)
        conn = self._reader()
        rg_cache: "OrderedDict" = OrderedDict()
        try:
            cur = conn.execute(f"SELECT gseq, chunk, row FROM events {where} ORDER BY gseq", params)
            while True:
                part = cur.fetchmany(batch_size)
                if not part:
                    break
                yield self._records(self._fetch(part, rg_cache))
        finally:
            conn.close()

    def scan_meta(self, batch_size: int = 4096, columns: Optional[Sequence[str]] = None, order: str = "gseq",
                  **filters) -> Iterator[list]:
        """Stream index rows (dicts of the flat columns) without reading any chunk; ``order="step"`` orders by
        (episode, env, actor, step, gseq), i.e. stream by stream."""
        cols = list(columns or _IDX)
        for c in cols:
            if c not in _IDX:
                raise ValueError(f"unknown index column {c!r}")
        if order not in ("gseq", "step"):
            raise ValueError("order must be 'gseq' or 'step'")
        where, params = self._where(filters)
        conn = self._reader()
        try:
            cur = conn.execute(f"SELECT {','.join(cols)} FROM events {where} ORDER BY "
                               f"{'gseq' if order == 'gseq' else 'episode_id, env_id, actor_id, step, gseq'}", params)
            while True:
                part = cur.fetchmany(batch_size)
                if not part:
                    break
                yield [dict(zip(cols, r)) for r in part]
        finally:
            conn.close()

    def meta(self, event_ids: Sequence[str]) -> list:
        """Index rows (current tags) of ``event_ids`` in order."""
        out = {}
        for part in _batches(list(event_ids), 900):
            q = f"SELECT {','.join(_IDX)} FROM events WHERE event_id IN ({','.join('?' * len(part))})"
            for r in self._db.execute(q, part):
                out[r[1]] = dict(zip(_IDX, r))
        return [out[e] for e in event_ids]

    def count_where(self, **filters) -> int:
        where, params = self._where(filters)
        return int(self._db.execute(f"SELECT COUNT(*) FROM events {where}", params).fetchone()[0])

    def episode_steps(self, episode_id: str, step_min: Optional[int] = None, step_max: Optional[int] = None,
                      env_id: Optional[str] = None, actor_id: Optional[str] = None) -> list:
        """[(step, event_id, gseq)] of one episode ordered by (step, gseq). Episode ids need not be unique across
        environments / actors: pass ``env_id`` and ``actor_id`` to select one (env, actor, episode) stream."""
        f = {"episode_id": episode_id}
        if env_id is not None:
            f["env_id"] = env_id
        if actor_id is not None:
            f["actor_id"] = actor_id
        if step_min is not None:
            f["step_min"] = step_min
        if step_max is not None:
            f["step_max"] = step_max
        where, params = self._where(f)
        return [tuple(r) for r in self._db.execute(
            f"SELECT step, event_id, gseq FROM events {where} ORDER BY step, gseq", params)]

    def stream_window(self, env_id: str, actor_id: str, episode_id: str, step: int, gseq: int, before: int,
                      after: int) -> tuple:
        """The event ``gseq`` with up to ``before`` preceding and ``after`` following events of its own
        (env, actor, episode) stream in (step, gseq) order -> ([(step, event_id, gseq)], index of the event).
        Two indexed range lookups: the cost is independent of the episode's length."""
        base = ("SELECT step, event_id, gseq FROM events INDEXED BY ev_stream WHERE episode_id=? AND env_id=? AND "
                "actor_id=? AND ")
        key = [episode_id, env_id, actor_id]
        b = self._db.execute(base + "step <= ? AND (step < ? OR gseq < ?) ORDER BY step DESC, gseq DESC LIMIT ?",
                             key + [step, step, gseq, int(before)]).fetchall()
        me = self._db.execute("SELECT step, event_id, gseq FROM events WHERE gseq=?", (gseq,)).fetchone()
        if me is None:
            raise KeyError(f"gseq {gseq} not in the archive")
        f = self._db.execute(base + "step >= ? AND (step > ? OR gseq > ?) ORDER BY step, gseq LIMIT ?",
                             key + [step, step, gseq, int(after)]).fetchall()
        return [tuple(r) for r in reversed(b)] + [tuple(me)] + [tuple(r) for r in f], len(b)

    def resolve(self, obs) -> str:
        """Full original content of an archived observation (follows ``artifact_ref`` when it was offloaded)."""
        ref = obs.get("artifact_ref") if isinstance(obs, dict) else obs.artifact_ref
        content = obs.get("content", "") if isinstance(obs, dict) else obs.content
        if ref and self.artifacts is not None:
            return self.artifacts.get_text(ref)
        return content

    # ------------------------------------------------------------------ later metadata
    def tag(self, event_ids: Sequence[str], **tags) -> int:
        """Set flat tags known only later (keys in ``TAG_KEYS``). Logged durably (after ``flush``) and indexed."""
        self._writable()
        for k in tags:
            if k not in TAG_KEYS:
                raise ValueError(f"unknown tag {k!r}; allowed {TAG_KEYS}")
        vals = {k: _tag_value(k, v) for k, v in tags.items()}
        locs = self._gseqs(event_ids)
        for eid, g in locs:
            line = _jd({"g": g, "e": eid, "t": vals}) + "\n"
            self._tag_fh.write(line)
            self._tag_written += len(line)
        self._apply_tags([(g, vals) for _, g in locs])
        self._db.commit()
        return len(locs)

    def mark_used(self, event_ids: Sequence[str], update_id: str) -> int:
        """Record that learner update ``update_id`` consumed these events (training provenance)."""
        self._writable()
        locs = self._gseqs(event_ids)
        for eid, g in locs:
            line = _jd({"g": g, "e": eid, "u": str(update_id)}) + "\n"
            self._tag_fh.write(line)
            self._tag_written += len(line)
        self._db.executemany("INSERT INTO used(event_id, update_id) VALUES (?,?)", [(e, str(update_id)) for e, _ in locs])
        self._db.commit()
        return len(locs)

    def tag_changes(self, since: int = 0, limit: int = 100_000, with_keys: bool = False) -> list:
        """[(seq, gseq)] of tag updates after change number ``since`` (change-data capture for the replay sampler);
        ``with_keys``: [(seq, gseq, [changed tag keys] or None if unknown)]."""
        rows = self._db.execute("SELECT seq, gseq, keys FROM tag_log WHERE seq > ? ORDER BY seq LIMIT ?",
                                (int(since), int(limit))).fetchall()
        if with_keys:
            return [(q, g, None if k is None else k.split(",")) for q, g, k in rows]
        return [(q, g) for q, g, _ in rows]

    @property
    def index_epoch(self) -> Optional[str]:
        """Identity of this archive's SQLite index; changes when the index is rebuilt (change numbers restart)."""
        try:
            r = self._db.execute("SELECT value FROM akv WHERE key='epoch'").fetchone()
        except sqlite3.OperationalError:
            return None
        return None if r is None else r[0]

    # ------------------------------------------------------------------ storage management
    def used_bytes(self) -> int:
        """Bytes counted against ``max_bytes``: everything under ``root`` (committed chunks from the manifest, the
        active WAL incl. its unflushed buffer, tag logs, SQLite files, manifest and any other file) plus chunks in an
        internal cold directory. Chunks on an external cold volume are counted by ``cold_bytes`` instead."""
        tot = sum(c["bytes"] for c in self._manifest["chunks"] if c["loc"] == "root" or not self.cold_external)
        wd, td = os.path.join(self.root, "wal"), os.path.join(self.root, "tags")
        tot += self._wal_written if self._wal_fh is not None else _dir_bytes(wd)
        tot += (self._tags_closed_bytes + self._tag_written) if self._tag_fh is not None else _dir_bytes(td)
        skip = {os.path.join(self.root, d) for d in ("chunks", "wal", "tags")}
        if not self.cold_external:
            skip.add(self.cold_dir)                 # its chunks are counted from the manifest above
        try:
            entries = list(os.scandir(self.root))
        except OSError:
            entries = []
        for e in entries:
            if e.path in skip:
                continue
            try:
                tot += _dir_bytes(e.path, skip) if e.is_dir(follow_symlinks=False) else \
                    e.stat(follow_symlinks=False).st_size
            except OSError:
                pass
        return tot

    def cold_bytes(self) -> int:
        """Bytes of chunks on the cold volume (external: its own budget ``cold_max_bytes``)."""
        return sum(c["bytes"] for c in self._manifest["chunks"] if c["loc"] == "cold")

    def chunks(self) -> list:
        return [dict(c) for c in self._manifest["chunks"]]

    def info(self) -> dict:
        return {"root": self.root, "count": self.count(), "sealed": self._sealed_rows(), "wal_rows": self._buf_len(),
                "wal_bytes": self._buf_bytes, "chunks": len(self._manifest["chunks"]), "used_bytes": self.used_bytes(),
                "cold_bytes": self.cold_bytes(), "cold_external": self.cold_external,
                "max_bytes": self.max_bytes, "tiers": {t: sum(1 for c in self._manifest["chunks"] if c["tier"] == t)
                                                       for t in ("hot", "z")},
                "stats": dict(self.stats), "recovery": dict(self.recovery)}

    def verify(self) -> dict:
        """Recompute the sha256 of every committed chunk and compare it with the manifest."""
        bad = []
        for c in self._manifest["chunks"]:
            p = self._chunk_path(c)
            if not os.path.exists(p) or sha256_file(p) != c["sha256"]:
                bad.append(c["n"])
        return {"ok": not bad, "bad_chunks": bad, "chunks": len(self._manifest["chunks"])}

    def backup(self, dest: str) -> dict:
        """Verified backup: copies every chunk (re-hashing each copy against the manifest), the WAL tail, the tag logs
        and a consistent SQLite snapshot into ``dest``; ``EventArchive(dest)`` opens it."""
        self.flush()
        dest = os.path.abspath(dest)
        for d in ("chunks", "cold", "wal", "tags"):
            os.makedirs(os.path.join(dest, d), exist_ok=True)
        verified, t0 = [], time.time()
        man = json.loads(json.dumps(self._manifest))
        for c in man["chunks"]:
            src = self._chunk_path(c)
            dst = os.path.join(dest, "chunks" if c["loc"] == "root" else "cold", c["file"])
            tmp = dst + ".tmp-copy"
            shutil.copyfile(src, tmp)
            fsync_file(tmp)
            if sha256_file(tmp) != c["sha256"]:
                os.unlink(tmp)
                raise ArchiveCorrupted(f"backup of chunk {c['n']} does not match its manifest digest")
            os.replace(tmp, dst)
            verified.append(c["n"])
        for d in ("wal", "tags"):
            for f in os.listdir(os.path.join(self.root, d)):
                shutil.copyfile(os.path.join(self.root, d, f), os.path.join(dest, d, f))
        dst_db = sqlite3.connect(os.path.join(dest, "index.sqlite"))
        self._db.commit()
        self._db.backup(dst_db)
        dst_db.close()
        man["cold_dir"] = os.path.join(dest, "cold")
        atomic_write_json(os.path.join(dest, "manifest.json"), man)
        rep = {"source": self.root, "dest": dest, "chunks_verified": verified, "count": self.count(),
               "seconds": time.time() - t0, "created": time.time()}
        atomic_write_json(os.path.join(dest, "BACKUP.json"), rep)
        fsync_dir(dest)
        return rep

    def pointer(self) -> dict:
        """What a full-lifetime checkpoint records about the archive (it is never copied into a checkpoint)."""
        return {"root": self.root, "count": self.count(), "sealed": self._sealed_rows(),
                "chunks": [(c["n"], c["rows"], c["sha256"]) for c in self._manifest["chunks"]],
                "cold_dir": self.cold_dir, "chunk_rows": self.chunk_rows}

    def refresh(self) -> None:
        """Read-only readers: pick up chunks and WAL rows the writer committed since this reader opened."""
        if not self.readonly:
            return
        self._manifest = self._load_manifest()
        self._recover_files()
        self._pf_cache.clear()
        self._replay_wal()

    def close(self) -> None:
        if getattr(self, "_db", None) is None:
            return
        try:
            self.flush()
        finally:
            for fh in (self._wal_fh, self._tag_fh):
                if fh is not None:
                    fh.close()
            self._wal_fh = self._tag_fh = None
            self._db.close()
            self._db = None
            if self._lock_fh is not None:
                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
                self._lock_fh.close()
                self._lock_fh = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __len__(self) -> int:
        return self.count()

    # ================================================================== rows
    def _row(self, r: EventRecord, gseq: int) -> dict:
        obs = [self._offload(o.to_dict() if hasattr(o, "to_dict") else dict(o)) for o in (r.observations or [])]
        res = None
        if r.result is not None:
            res = r.result.to_dict()
            res["observations"] = [self._offload(o) for o in res.get("observations", [])]
        tags = event_tags(r)
        return {"gseq": gseq, "event_id": str(r.event_id), "env_id": str(r.env_id), "actor_id": str(r.actor_id),
                "episode_id": str(r.episode_id), "step": int(r.step), "model_version": int(r.model_version),
                "t_sim": float(r.t_sim), "t_wall": float(r.t_wall), "provenance": _enum_val(r.provenance),
                "evidence": _enum_val(r.evidence), "goal_id": r.goal_id, "reward": float(r.reward),
                "cost": float(r.cost), "done": bool(r.done),
                "behavior_logp": None if r.behavior_logp is None else float(r.behavior_logp),
                "randomization_p": None if r.randomization_p is None else float(r.randomization_p),
                "attributed_to": r.attributed_to, "tool": None if r.action is None else str(r.action.tool),
                "ok": tags["ok"], "failed": tags["failed"], "success": tags["success"], "surprise": tags["surprise"],
                "novelty": tags["novelty"], "family": tags["family"], "innovation": tags["innovation"],
                "skill": tags["skill"], "entities": event_entities(r),
                "observations": _jd(obs), "action": _jd(None if r.action is None else r.action.to_dict()),
                "result": _jd(res), "predicted": _jd(r.predicted), "uncertainty": _jd(r.uncertainty),
                "entity_links": _jd(r.entity_links), "simultaneous": _jd(r.simultaneous),
                "training_provenance": _jd(r.training_provenance), "extra": _jd(r.extra)}

    def _offload(self, od: dict) -> dict:
        c = od.get("content") or ""
        if self.artifacts is not None and len(c) > self.inline_limit and not od.get("artifact_ref"):
            od = dict(od)
            od["artifact_ref"] = self.artifacts.put(c)
            od["content"] = c[: self.inline_limit]
            od["meta"] = dict(od.get("meta") or {}, truncated=True, content_chars=len(c))
        return od

    def _records(self, rows: list) -> list:
        ids = [r["event_id"] for r in rows]
        used = {}
        for part in _batches(ids, 900):
            q = f"SELECT event_id, update_id FROM used WHERE event_id IN ({','.join('?' * len(part))}) ORDER BY rowid"
            for e, u in self._db.execute(q, part):
                used.setdefault(e, []).append(u)
        out = []
        for d in rows:
            tp = json.loads(d["training_provenance"]) or {}
            if d["event_id"] in used:
                tp = dict(tp)
                tp["used_in_updates"] = list(tp.get("used_in_updates", [])) + used[d["event_id"]]
            out.append(EventRecord.from_dict({
                "event_id": d["event_id"], "env_id": d["env_id"], "actor_id": d["actor_id"],
                "episode_id": d["episode_id"], "step": d["step"], "model_version": d["model_version"],
                "t_sim": d["t_sim"], "observations": json.loads(d["observations"]), "action": json.loads(d["action"]),
                "result": json.loads(d["result"]), "predicted": json.loads(d["predicted"]),
                "uncertainty": json.loads(d["uncertainty"]), "goal_id": d["goal_id"], "reward": d["reward"],
                "cost": d["cost"], "done": d["done"], "entity_links": json.loads(d["entity_links"]),
                "behavior_logp": d["behavior_logp"], "provenance": d["provenance"], "evidence": d["evidence"],
                "randomization_p": d["randomization_p"], "simultaneous": json.loads(d["simultaneous"]),
                "attributed_to": d["attributed_to"], "training_provenance": tp, "t_wall": d["t_wall"],
                "extra": json.loads(d["extra"])}))
        return out

    def _fetch(self, locs: list, rg_cache: "OrderedDict") -> list:
        """Row dicts for [(gseq, chunk, row)] in the given order. Rows still in the WAL tail come from the buffer."""
        out = [None] * len(locs)
        by_chunk: dict = {}
        for i, (g, c, r) in enumerate(locs):
            if c == -1:
                k = g - self._buf_g0
                if not 0 <= k < self._buf_len() and self.readonly:
                    self.refresh()                   # a reader older than the writer's latest rows
                    k = g - self._buf_g0
                if 0 <= k < self._buf_len():
                    out[i] = {col: self._buf[col][k] for col in COLUMNS}
                    continue
                # sealed after the reader's snapshot: locate it again
                c, r = self._db.execute("SELECT chunk, row FROM events WHERE gseq=?", (g,)).fetchone()
            by_chunk.setdefault(c, []).append((r, i))
        for c, items in by_chunk.items():
            pf = self._pf(c)
            bounds = _rg_bounds(pf)
            items.sort()
            j = 0
            while j < len(items):
                rg = _rg_of(bounds, items[j][0])
                lo, hi = bounds[rg], bounds[rg + 1]
                key = (c, rg)
                tbl = rg_cache.get(key)
                if tbl is None:
                    tbl = pf.read_row_group(rg)
                    rg_cache[key] = tbl
                    while len(rg_cache) > 2:
                        rg_cache.popitem(last=False)
                take, dst = [], []
                while j < len(items) and items[j][0] < hi:
                    take.append(items[j][0] - lo)
                    dst.append(items[j][1])
                    j += 1
                for d, row in zip(dst, tbl.take(take).to_pylist()):
                    out[d] = row
        return out

    # ================================================================== index
    def _open_index(self) -> None:
        path = os.path.join(self.root, "index.sqlite")
        fresh = not os.path.exists(path)
        try:
            self._db = self._connect(path)
            ok = self._db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            have = {r[0] for r in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not fresh and not {"events", "chunks", "entities", "used", "tag_log"} <= have:
                ok = False
        except sqlite3.DatabaseError:
            ok = False
        if not ok:
            if self.readonly:
                raise ArchiveCorrupted(f"index of {self.root} is unreadable; open once read-write to rebuild it")
            try:
                self._db.close()
            except Exception:
                pass
            for suf in ("", "-wal", "-shm"):
                if os.path.exists(path + suf):
                    os.replace(path + suf, f"{path}{suf}.corrupt-{int(time.time())}")
            self._db = self._connect(path)
            fresh = True
        if not self.readonly:
            self._create_schema()
        if fresh and not self.readonly:
            self.recovery["index_rebuilt"] = bool(self._manifest["chunks"]) or self.recovery["index_rebuilt"]
            for c in self._manifest["chunks"]:
                self._reindex_chunk(c)
            self._rebuild_tags = True          # all tag logs are replayed once the WAL tail is indexed too
        elif not self.readonly:
            have = {int(r[0]): int(r[1]) for r in self._db.execute("SELECT n, rows FROM chunks")}
            man = {c["n"]: c for c in self._manifest["chunks"]}
            for n in set(have) - set(man):          # index rows of a chunk that is not committed
                self._db.execute("DELETE FROM entities WHERE gseq IN (SELECT gseq FROM events WHERE chunk=?)", (n,))
                self._db.execute("DELETE FROM events WHERE chunk=?", (n,))
                self._db.execute("DELETE FROM chunks WHERE n=?", (n,))
            for n, c in sorted(man.items()):
                if have.get(n) != c["rows"]:
                    self._reindex_chunk(c)
            self._db.commit()
        self._db.commit()

    def _connect(self, path: str) -> sqlite3.Connection:
        if self.readonly:
            db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        else:
            db = sqlite3.connect(path, check_same_thread=False)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("PRAGMA wal_autocheckpoint=256")           # bound the SQLite -wal file (~1 MB)
            db.execute("PRAGMA journal_size_limit=1048576")
        return db

    def _reader(self) -> sqlite3.Connection:
        path = os.path.join(self.root, "index.sqlite")
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)

    def _create_schema(self) -> None:
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            gseq INTEGER PRIMARY KEY, event_id TEXT NOT NULL UNIQUE, chunk INTEGER NOT NULL, row INTEGER NOT NULL,
            env_id TEXT, actor_id TEXT, episode_id TEXT, step INTEGER, model_version INTEGER, t_sim REAL, t_wall REAL,
            provenance TEXT, goal_id TEXT, reward REAL, cost REAL, done INTEGER, tool TEXT, ok INTEGER,
            failed INTEGER, success INTEGER, surprise REAL, novelty REAL, family TEXT, innovation INTEGER, skill TEXT);
        DROP INDEX IF EXISTS ev_episode;
        CREATE INDEX IF NOT EXISTS ev_stream ON events(episode_id, env_id, actor_id, step);
        CREATE INDEX IF NOT EXISTS ev_env_t ON events(env_id, t_sim);
        CREATE INDEX IF NOT EXISTS ev_t ON events(t_sim);
        CREATE INDEX IF NOT EXISTS ev_chunk ON events(chunk);
        CREATE TABLE IF NOT EXISTS entities (entity TEXT NOT NULL, gseq INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS ent_entity ON entities(entity, gseq);
        CREATE INDEX IF NOT EXISTS ent_gseq ON entities(gseq);
        CREATE TABLE IF NOT EXISTS used (event_id TEXT NOT NULL, update_id TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS used_event ON used(event_id);
        CREATE TABLE IF NOT EXISTS chunks (n INTEGER PRIMARY KEY, rows INTEGER, gseq0 INTEGER, sha256 TEXT);
        CREATE TABLE IF NOT EXISTS tag_log (seq INTEGER PRIMARY KEY AUTOINCREMENT, gseq INTEGER NOT NULL, keys TEXT);
        CREATE TABLE IF NOT EXISTS akv (key TEXT PRIMARY KEY, value TEXT);
        """)
        if "keys" not in {r[1] for r in self._db.execute("PRAGMA table_info(tag_log)")}:
            self._db.execute("ALTER TABLE tag_log ADD COLUMN keys TEXT")
        import uuid
        self._db.execute("INSERT OR IGNORE INTO akv(key, value) VALUES ('epoch', ?)", (uuid.uuid4().hex,))
        self._db.commit()

    def _index_rows(self, rows: list, chunk: int, g0: Optional[int]) -> None:
        def irow(d):
            return (d["gseq"], d["event_id"], chunk, -1 if g0 is None else d["gseq"] - g0, d["env_id"], d["actor_id"],
                    d["episode_id"], d["step"], d["model_version"], d["t_sim"], d["t_wall"], d["provenance"],
                    d["goal_id"], d["reward"], d["cost"], int(bool(d["done"])), d["tool"],
                    None if d["ok"] is None else int(bool(d["ok"])), int(bool(d["failed"])), d["success"],
                    d["surprise"], d["novelty"], d["family"], int(bool(d["innovation"])), d["skill"])
        self._db.executemany(f"INSERT OR REPLACE INTO events({','.join(_IDX)}) VALUES ({','.join('?' * len(_IDX))})",
                             [irow(d) for d in rows])
        self._db.executemany("INSERT INTO entities(entity, gseq) VALUES (?,?)",
                             [(e, d["gseq"]) for d in rows for e in (d["entities"] or [])])

    def _reindex_chunk(self, c: dict) -> None:
        """(Re)build the index rows of one committed chunk from its Parquet file (flat columns only)."""
        g0, n = c["gseq0"], c["rows"]
        self._db.execute("DELETE FROM entities WHERE gseq BETWEEN ? AND ?", (g0, g0 + n - 1))
        self._db.execute("DELETE FROM events WHERE gseq BETWEEN ? AND ?", (g0, g0 + n - 1))
        pf = pq.ParquetFile(self._chunk_path(c))
        flat = [x for x, _ in _FLAT]
        for b in pf.iter_batches(batch_size=8192, columns=flat):
            self._index_rows(b.to_pylist(), chunk=c["n"], g0=g0)
        self._db.execute("INSERT OR REPLACE INTO chunks(n, rows, gseq0, sha256) VALUES (?,?,?,?)",
                         (c["n"], n, g0, c["sha256"]))
        self.recovery["reindexed_chunks"] += 1

    def _known_ids(self, ids: list) -> set:
        out = set()
        for part in _batches(ids, 900):
            q = f"SELECT event_id FROM events WHERE event_id IN ({','.join('?' * len(part))})"
            out.update(r[0] for r in self._db.execute(q, part))
        return out

    def _gseqs(self, event_ids) -> list:
        event_ids = list(event_ids)
        found = {}
        for part in _batches(event_ids, 900):
            q = f"SELECT event_id, gseq FROM events WHERE event_id IN ({','.join('?' * len(part))})"
            found.update({e: g for e, g in self._db.execute(q, part)})
        missing = [e for e in event_ids if e not in found]
        if missing:
            raise KeyError(f"{len(missing)} event ids not in the archive, e.g. {missing[:3]}")
        return [(e, found[e]) for e in event_ids]

    def _apply_tags(self, items: list, only_changed: bool = False) -> None:
        """Apply tag dicts to index rows (+ buffered tail) and log each change for the replay sampler's CDC.
        ``only_changed``: skip rows that already hold these values (re-applying a tag log after a restart)."""
        for g, vals in items:
            if not vals:
                continue
            if only_changed:
                cur = self._db.execute(f"SELECT {','.join(vals)} FROM events WHERE gseq=?", (g,)).fetchone()
                if cur is not None and all(c == v for c, v in zip(cur, vals.values())):
                    continue
            sets = ",".join(f"{k}=?" for k in vals)
            self._db.execute(f"UPDATE events SET {sets} WHERE gseq=?", [*vals.values(), g])
            self._db.execute("INSERT INTO tag_log(gseq, keys) VALUES (?,?)", (g, ",".join(sorted(vals))))
            k = g - self._buf_g0
            if 0 <= k < self._buf_len():
                for key, v in vals.items():
                    self._buf[key][k] = v if key not in ("failed", "innovation") else bool(v)

    def _where(self, filters: dict) -> tuple:
        clauses, params = [], []
        for k, v in filters.items():
            if k not in FILTER_KEYS:
                raise ValueError(f"unknown archive filter {k!r}")
            if v is None:
                continue
            if k in _RANGE_FILTERS:
                col, op = _RANGE_FILTERS[k]
                clauses.append(f"{col} {op} ?")
                params.append(v)
            elif k in _BOOL_FILTERS:
                clauses.append(f"{k} = ?")
                params.append(int(bool(v)))
            elif k == "entity":
                vs = [v] if isinstance(v, str) else list(v)
                clauses.append(f"gseq IN (SELECT gseq FROM entities WHERE entity IN ({','.join('?' * len(vs))}))")
                params.extend(vs)
            else:
                col = "event_id" if k == "event_ids" else k
                vs = list(v) if isinstance(v, (list, tuple, set, frozenset)) else [v]
                vs = [_enum_val(x) for x in vs]
                if not vs:
                    clauses.append("0")
                else:
                    clauses.append(f"{col} IN ({','.join('?' * len(vs))})")
                    params.extend(vs)
        return ("WHERE " + " AND ".join(clauses)) if clauses else "", params

    # ================================================================== chunks
    def _chunk_path(self, c: dict) -> str:
        return os.path.join(self.root, "chunks", c["file"]) if c["loc"] == "root" else os.path.join(self.cold_dir, c["file"])

    def _pf(self, n: int) -> pq.ParquetFile:
        pf = self._pf_cache.get(n)
        if pf is None:
            c = self._chunk_by_n(n)
            pf = pq.ParquetFile(self._chunk_path(c))
            self._pf_cache[n] = pf
            while len(self._pf_cache) > 16:
                self._pf_cache.popitem(last=False)
        else:
            self._pf_cache.move_to_end(n)
        return pf

    def _chunk_by_n(self, n: int, _retry: bool = True) -> dict:
        for c in self._manifest["chunks"]:
            if c["n"] == n:
                return c
        if self.readonly and _retry:                 # committed by the writer after this reader opened
            self.refresh()
            return self._chunk_by_n(n, _retry=False)
        raise ArchiveCorrupted(f"chunk {n} is not in the manifest")

    def _sealed_rows(self) -> int:
        return sum(c["rows"] for c in self._manifest["chunks"])

    def _buf_len(self) -> int:
        return len(self._buf["gseq"])

    def _seal(self, n: int) -> None:
        """Write the first ``n`` buffered rows as chunk number ``next_chunk`` (see the module write protocol)."""
        num = self._manifest["next_chunk"]
        g0 = self._buf_g0
        tbl = pa.Table.from_pydict({c: self._buf[c][:n] for c in COLUMNS}, schema=SCHEMA)
        name = f"chunk_{num:08d}.hot.parquet"
        final = os.path.join(self.root, "chunks", name)
        tmp = os.path.join(self.root, "chunks", f".tmp-{name}")
        pq.write_table(tbl, tmp, compression=self.compression, compression_level=self.compression_level,
                       row_group_size=self.row_group_rows)
        if self.do_fsync:
            fsync_file(tmp)
        size = os.path.getsize(tmp)
        if self.max_bytes is not None:
            remaining = sum(self._buf_sizes[n:])                              # the WAL is rewritten to these rows
            over = self.used_bytes() + size - max(0, self._wal_written - remaining) - self.max_bytes   # after seal
            if over > 0 and self._make_room(over) < over:
                os.unlink(tmp)
                raise StorageLimitExceeded(
                    f"sealing chunk {num} of {self.root} needs {size} bytes beyond the limit {self.max_bytes}; the "
                    f"{n} rows stay durable in the write-ahead log", used=self.used_bytes(), limit=self.max_bytes,
                    needed=size)
        sha = sha256_file(tmp)
        self._rename_chunk(tmp, final)
        entry = {"n": num, "file": name, "rows": n, "bytes": size, "sha256": sha, "gseq0": g0, "tier": "hot",
                 "loc": "root", "sealed": time.time()}
        self._commit_chunk_index(entry)
        self._manifest["chunks"].append(entry)
        self._manifest["next_chunk"] = num + 1
        self._write_manifest()
        for c in COLUMNS:
            del self._buf[c][:n]
        self._buf_bytes -= sum(self._buf_sizes[:n])
        del self._buf_sizes[:n]
        self._buf_g0 = g0 + n
        self._rotate_logs()
        self.stats["sealed_chunks"] += 1

    def _rename_chunk(self, tmp: str, final: str) -> None:
        os.replace(tmp, final)
        if self.do_fsync:
            fsync_dir(os.path.dirname(final))

    def _commit_chunk_index(self, entry: dict) -> None:
        g0, n = entry["gseq0"], entry["rows"]
        self._db.execute("UPDATE events SET chunk=?, row=gseq-? WHERE gseq BETWEEN ? AND ?",
                         (entry["n"], g0, g0, g0 + n - 1))
        self._db.execute("INSERT OR REPLACE INTO chunks(n, rows, gseq0, sha256) VALUES (?,?,?,?)",
                         (entry["n"], n, g0, entry["sha256"]))
        self._db.commit()

    def _write_manifest(self) -> None:
        self._manifest["updated"] = time.time()
        atomic_write_json(os.path.join(self.root, "manifest.json"), self._manifest, fsync=self.do_fsync)

    def _load_manifest(self) -> dict:
        p = os.path.join(self.root, "manifest.json")
        if os.path.exists(p):
            with open(p) as fh:
                m = json.load(fh)
            if m.get("format") != ARCHIVE_FORMAT:
                raise ArchiveCorrupted(f"unknown archive format {m.get('format')}")
            return m
        return {"format": ARCHIVE_FORMAT, "chunks": [], "next_chunk": 0, "created": time.time()}

    def _wal_bytes(self) -> int:
        d = os.path.join(self.root, "wal")
        return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) if os.path.isdir(d) else 0

    # ------------------------------------------------------------------ cold policy
    def _index_row_bytes(self) -> int:
        """Running estimate of SQLite index bytes per event (main file size / events), clipped to [160, 4096]."""
        p = os.path.join(self.root, "index.sqlite")
        if self._next < 1000 or not os.path.exists(p):
            return 512
        return int(min(4096, max(160, 1.25 * os.path.getsize(p) / self._next)))

    def _ensure_room(self, need: int) -> None:
        """``need`` = WAL bytes + estimated index growth of the call; ``index_reserve_bytes`` keeps head-room for
        SQLite's own journal, whose size the archive does not control exactly."""
        if self.max_bytes is None:
            return
        j = os.path.join(self.root, "index.sqlite-wal")
        reserve = max(0, self.index_reserve_bytes - (os.path.getsize(j) if os.path.exists(j) else 0))
        over = self.used_bytes() + need + reserve - self.max_bytes
        if over <= 0:
            return
        freed = self._make_room(over)
        if freed < over:
            raise StorageLimitExceeded(
                f"archive {self.root}: {self.used_bytes()} + {need} bytes exceeds the limit {self.max_bytes} after the "
                f"cold policy {self.cold_policy} freed {freed} bytes; no record of this call was accepted",
                used=self.used_bytes(), limit=self.max_bytes, needed=need)

    def _make_room(self, need: float) -> int:
        freed = 0
        for pol in self.cold_policy:
            for c in list(self._manifest["chunks"]):
                if freed >= need:
                    return freed
                if pol == "compress" and c["tier"] == "hot" and c["loc"] == "root":
                    freed += self._recompress(c)
                elif pol == "move" and c["loc"] == "root":
                    freed += self._move_cold(c)
        return freed

    def _recompress(self, c: dict) -> int:
        """Rewrite a hot chunk with strong zstd under a new name; rows and order are unchanged (index stays valid)."""
        src = self._chunk_path(c)
        tbl = pq.read_table(src)
        name = f"chunk_{c['n']:08d}.z.parquet"
        dst = os.path.join(os.path.dirname(src), name)
        tmp = os.path.join(os.path.dirname(src), f".tmp-{name}")
        pq.write_table(tbl, tmp, compression="zstd", compression_level=self.cold_compression_level,
                       row_group_size=self.row_group_rows)
        del tbl
        if self.do_fsync:
            fsync_file(tmp)
        size = os.path.getsize(tmp)
        if size >= c["bytes"]:
            os.unlink(tmp)
            c["tier"] = "z"                     # nothing to gain; do not try again
            self._write_manifest()
            return 0
        sha = sha256_file(tmp)
        self._rename_chunk(tmp, dst)
        old = c["bytes"]
        c.update(file=name, bytes=size, sha256=sha, tier="z")
        self._write_manifest()
        self._pf_cache.pop(c["n"], None)
        os.unlink(src)
        self.stats["compressed_chunks"] += 1
        self.stats["freed_bytes"] += old - size
        return old - size

    def _move_cold(self, c: dict) -> int:
        src = self._chunk_path(c)
        dst = os.path.join(self.cold_dir, c["file"])
        if self.cold_max_bytes is not None and self.cold_bytes() + c["bytes"] > self.cold_max_bytes:
            return 0
        tmp = dst + ".tmp-move"
        shutil.copyfile(src, tmp)
        if self.do_fsync:
            fsync_file(tmp)
        if sha256_file(tmp) != c["sha256"]:
            os.unlink(tmp)
            raise ArchiveCorrupted(f"copy of chunk {c['n']} to the cold volume failed verification")
        self._rename_chunk(tmp, dst)
        c["loc"] = "cold"
        self._write_manifest()
        self._pf_cache.pop(c["n"], None)
        os.unlink(src)
        self.stats["moved_chunks"] += 1
        self.stats["freed_bytes"] += c["bytes"]
        return c["bytes"]

    # ================================================================== recovery
    def _recover_files(self) -> None:
        """Bring files and manifest into agreement after a crash (see module docstring)."""
        dirs = {"root": os.path.join(self.root, "chunks"), "cold": self.cold_dir}
        on_disk: dict = {}
        for loc, d in dirs.items():
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                if f.startswith(".tmp-") or f.endswith(".tmp-move") or f.endswith(".tmp-copy"):
                    if not self.readonly:
                        os.unlink(p)
                        self.recovery["removed_tmp"] += 1
                    continue
                if f.startswith("chunk_") and f.endswith(".parquet"):
                    n = int(f.split("_")[1].split(".")[0])
                    on_disk.setdefault(n, []).append((loc, f))
        man = {c["n"]: c for c in self._manifest["chunks"]}
        changed = False
        for n, c in man.items():
            files = on_disk.get(n, [])
            if (c["loc"], c["file"]) in files:
                for loc, f in files:
                    if (loc, f) != (c["loc"], c["file"]) and not self.readonly:
                        os.unlink(os.path.join(dirs[loc], f))       # complete redundant copy (compress/move crash)
                        self.recovery["dropped_duplicate_files"] += 1
                continue
            alt = [x for x in files if self._valid_chunk(os.path.join(dirs[x[0]], x[1]), c["rows"], c["gseq0"])]
            if not alt:
                raise ArchiveCorrupted(f"committed chunk {n} ({c['file']}) of {self.root} is missing")
            loc, f = alt[0]
            p = os.path.join(dirs[loc], f)
            c.update(loc=loc, file=f, bytes=os.path.getsize(p), sha256=sha256_file(p),
                     tier="z" if ".z." in f else "hot")
            changed = True
        end = self._sealed_rows()
        for n in sorted(set(on_disk) - set(man)):
            for loc, f in on_disk[n]:
                p = os.path.join(dirs[loc], f)
                meta = self._chunk_range(p)
                if meta is not None and meta[0] == end and n >= self._manifest["next_chunk"]:
                    entry = {"n": n, "file": f, "rows": meta[1], "bytes": os.path.getsize(p), "sha256": sha256_file(p),
                             "gseq0": end, "tier": "z" if ".z." in f else "hot", "loc": loc, "sealed": time.time()}
                    self._manifest["chunks"].append(entry)
                    self._manifest["next_chunk"] = n + 1
                    end += meta[1]
                    changed = True
                    self.recovery["adopted_chunks"] += 1
                    break
                if not self.readonly:
                    q = os.path.join(self.root, "quarantine")
                    os.makedirs(q, exist_ok=True)
                    os.replace(p, os.path.join(q, f))
                    self.recovery["quarantined"] += 1
        self._manifest["chunks"].sort(key=lambda c: c["n"])
        if changed and not self.readonly:
            self._write_manifest()

    @staticmethod
    def _chunk_range(path: str):
        try:
            pf = pq.ParquetFile(path)
            n = pf.metadata.num_rows
            if n == 0:
                return None
            g = pf.read_row_group(0, columns=["gseq"]).column(0)
            return int(g[0].as_py()), int(n)
        except Exception:
            return None

    def _valid_chunk(self, path: str, rows: int, g0: int) -> bool:
        r = self._chunk_range(path)
        return r is not None and r == (g0, rows)

    def _replay_wal(self) -> None:
        """Load the unsealed tail from the WAL files (torn last lines ignored) and index it."""
        end = self._sealed_rows()
        self._buf = {c: [] for c in COLUMNS}
        self._buf_sizes, self._buf_bytes = [], 0
        self._buf_g0 = end
        if not self.readonly:
            # tag values already committed (and change-logged) for tail rows survive their re-indexing from the WAL
            saved = {r[0]: (r[1], dict(zip(TAG_KEYS, r[2:]))) for r in self._db.execute(
                f"SELECT gseq, event_id, {','.join(TAG_KEYS)} FROM events WHERE chunk=-1")}
            self._db.execute("DELETE FROM entities WHERE gseq IN (SELECT gseq FROM events WHERE chunk=-1)")
            self._db.execute("DELETE FROM events WHERE chunk=-1")
        else:
            saved = {}
        d = os.path.join(self.root, "wal")
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        nxt, rows = end, []
        for f in files:
            with open(os.path.join(d, f), "rb") as fh:
                data = fh.read()
            for line in data.split(b"\n"):
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    self.recovery["wal_torn_lines"] += 1
                    break
                if row["gseq"] < nxt:
                    continue
                if row["gseq"] != nxt:
                    break
                rows.append(row)
                self._buf_sizes.append(len(line) + 1)
                nxt += 1
        for row in rows:
            for c in COLUMNS:
                self._buf[c].append(row[c])
        self._buf_bytes = sum(self._buf_sizes)
        if not self.readonly and rows:
            self._index_rows(rows, chunk=-1, g0=None)
            for k, row in enumerate(rows):
                eid, tags = saved.get(row["gseq"], (None, None))
                if eid != row["event_id"]:
                    continue
                diff = {t: v for t, v in tags.items() if v != _tag_value(t, row[t]) and not (
                    t in ("family", "skill") and v in (None, "") and row[t] in (None, ""))}
                if diff:
                    self._db.execute(f"UPDATE events SET {','.join(f'{t}=?' for t in diff)} WHERE gseq=?",
                                     [*diff.values(), row["gseq"]])
                    for t, v in diff.items():
                        self._buf[t][k] = bool(v) if t in ("failed", "innovation") else v
        self.recovery["wal_rows"] = len(rows)
        self._next = nxt
        if not self.readonly:
            self._replay_tag_logs(all_files=self._rebuild_tags)
            self._rebuild_tags = False
            self._db.commit()

    def _replay_tag_logs(self, all_files: bool) -> None:
        """Re-apply logged tags (the newest log segment; all of them after an index rebuild). Only the net effect per
        event is applied, and only where it differs from the index, so restarts add no change-log rows."""
        d = os.path.join(self.root, "tags")
        if not os.path.isdir(d):
            return
        files = sorted(f for f in os.listdir(d) if f.endswith(".jsonl"))
        if not all_files:
            files = files[-1:]
        net: "OrderedDict[int, dict]" = OrderedDict()
        for f in files:
            with open(os.path.join(d, f), "rb") as fh:
                data = fh.read()
            for line in data.split(b"\n"):
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    break
                g = e["g"]
                if self._db.execute("SELECT 1 FROM events WHERE gseq=? AND event_id=?", (g, e["e"])).fetchone() is None:
                    continue                                  # tagged event did not survive (was never durable)
                if "t" in e:
                    net.setdefault(g, {}).update(e["t"])
                elif "u" in e and all_files:
                    self._db.execute("INSERT INTO used(event_id, update_id) VALUES (?,?)", (e["e"], e["u"]))
        self._apply_tags(list(net.items()), only_changed=True)

    def _open_logs(self) -> None:
        wd = os.path.join(self.root, "wal")
        name = os.path.join(wd, f"wal_{self._buf_g0:012d}.jsonl")
        existing = sorted(os.listdir(wd))
        # rewrite the tail into one fresh WAL file so the active file starts exactly at the buffer start
        tmp = name + ".new"
        with open(tmp, "w") as fh:
            sizes = []
            for k in range(self._buf_len()):
                line = _jd({c: self._buf[c][k] for c in COLUMNS}) + "\n"
                fh.write(line)
                sizes.append(len(line))
            fh.flush()
            if self.do_fsync:
                os.fsync(fh.fileno())
        self._buf_sizes, self._buf_bytes = sizes, sum(sizes)
        os.replace(tmp, name)
        for f in existing:
            if os.path.join(wd, f) != name:
                os.unlink(os.path.join(wd, f))
        if self.do_fsync:
            fsync_dir(wd)
        self._wal_fh = open(name, "a", buffering=1 << 20)
        self._wal_written = os.path.getsize(name)
        td = os.path.join(self.root, "tags")
        tags = sorted(f for f in os.listdir(td) if f.endswith(".jsonl"))
        tname = os.path.join(td, tags[-1]) if tags else os.path.join(td, f"tags_{self._manifest['next_chunk']:08d}.jsonl")
        self._tags_closed_bytes = _dir_bytes(td) - (os.path.getsize(tname) if os.path.exists(tname) else 0)
        self._tag_fh = open(tname, "a", buffering=1 << 16)
        self._tag_written = os.path.getsize(tname)

    def _rotate_logs(self) -> None:
        """After a seal: the new WAL holds only the remaining tail; a new tag-log segment starts."""
        old = self._wal_fh
        old.flush()
        old_name = old.name
        old.close()
        wd = os.path.join(self.root, "wal")
        name = os.path.join(wd, f"wal_{self._buf_g0:012d}.jsonl")
        tmp = name + ".new"
        with open(tmp, "w") as fh:
            sizes = []
            for k in range(self._buf_len()):
                line = _jd({c: self._buf[c][k] for c in COLUMNS}) + "\n"
                fh.write(line)
                sizes.append(len(line))
            fh.flush()
            if self.do_fsync:
                os.fsync(fh.fileno())
        self._buf_sizes, self._buf_bytes = sizes, sum(sizes)
        os.replace(tmp, name)
        if old_name != name:
            os.unlink(old_name)
        if self.do_fsync:
            fsync_dir(wd)
        self._wal_fh = open(name, "a", buffering=1 << 20)
        self._wal_written = os.path.getsize(name)
        self._tag_fh.flush()
        if self.do_fsync:
            os.fsync(self._tag_fh.fileno())
        self._tag_fh.close()
        self._tags_closed_bytes += self._tag_written
        tname = os.path.join(self.root, "tags", f"tags_{self._manifest['next_chunk']:08d}.jsonl")
        self._tag_fh = open(tname, "a", buffering=1 << 16)
        self._tag_written = os.path.getsize(tname)

    def _writable(self) -> None:
        if self.readonly:
            raise PermissionError("archive opened read-only")
        if getattr(self, "_db", None) is None:
            raise ValueError("archive is closed")


def _tag_value(k: str, v):
    if v is None:
        return None
    if k in ("failed", "innovation"):
        return int(bool(v))
    if k == "success":
        return int(bool(v))
    if k in ("surprise", "novelty"):
        return float(v)
    return str(v)


def _dir_bytes(path: str, skip: Optional[set] = None) -> int:
    """Bytes of the files under ``path`` (not following symlinks), excluding the directories in ``skip``."""
    tot = 0
    for root, dirs, files in os.walk(path):
        if skip:
            dirs[:] = [d for d in dirs if os.path.join(root, d) not in skip]
        for f in files:
            try:
                tot += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return tot


def _batches(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _rg_bounds(pf: pq.ParquetFile) -> list:
    b = [0]
    for i in range(pf.metadata.num_row_groups):
        b.append(b[-1] + pf.metadata.row_group(i).num_rows)
    return b


def _rg_of(bounds: list, row: int) -> int:
    import bisect
    return bisect.bisect_right(bounds, row) - 1
