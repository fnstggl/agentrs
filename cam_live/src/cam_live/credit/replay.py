"""cam_live.credit.replay — durable experience storage and stratified replay sampling (spec 9.1, 10.1, 15.2).

:class:`SqliteExperienceStore` implements ``contracts.ExperienceStore`` on SQLite (WAL) plus a content-addressed
blob directory for large artifacts.

Design decisions (each answers a failure of an earlier project):

* **Durability.** Every public write is one ``BEGIN IMMEDIATE .. COMMIT`` with ``synchronous=FULL``: when ``append``
  returns the record is on disk, so a writer killed with SIGKILL loses nothing it acknowledged. Large parts are
  written to the blob directory (fsync, atomic rename) *before* the row that references them is committed.
* **Single writer.** A writer holds ``flock(<db>.lock)``; a second writer gets :class:`StoreLockedError`. Readers
  open with ``readonly=True`` and take no lock.
* **Cheap updates.** A record is stored in three parts (static: decision and context; result; dynamic: outcomes,
  credit, eligibility ...). ``update`` rewrites only the parts whose hash changed, so adding an outcome or a credit
  value does not copy the decision and its provenance again.
* **Nothing is evicted.** There is no reservoir: every record stays, and balance comes from the sampler, which draws
  from strata (recent / informative older / successes / failures / delayed consequences / objective families).
* **Provenance strata.** ``live`` / ``staging`` / ``diagnostic`` records are never mixed in a sample unless the
  caller names several provenances explicitly.
* **Eligibility is per loss.** ``sample(loss=name)`` honours every explicit ``eligible[name]`` flag. A record with no
  entry for a loss follows ``eligible_default`` — True, as in the contract's reference ``MemoryStore``; construct the
  store with ``eligible_default=False`` to make missing entries ineligible (fail-closed).
* **History is immutable.** A context item, once stored under a ref, is never replaced: rewriting it would change
  ``H_t`` of every earlier decision that references it (future leakage). A stored record's history is append-only
  too: ``update`` refuses (:class:`ImmutableHistoryError`) any change of the decision (candidates, behaviour
  probabilities, time, how it was made), its context refs, goal, family, provenance or lifetime position, of an
  executed result, of a stored final outcome or of a query definition; it may add outcomes, queries and decision
  annotations and change derived data (credit, eligibility, labels, intrinsic rewards, window). See
  :func:`history_changes`.
* **The index never contradicts the content.** Index columns are written from the record in the same transaction
  and every read checks them against the decoded body (:class:`StoreIntegrityError`; ``verify_index``).
"""
from __future__ import annotations

import collections.abc
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import threading
import typing
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..contracts import (ExperienceItem, ExperienceRecord, ExperimentRecord, OutcomeKind, OutcomeObservation,
                         OutcomeQuery, OutcomeStatus, PredictionRecord, ResultStatus)
from ._fs import atomic_write_bytes, fsync_dir

SCHEMA_VERSION = 1
PROVENANCES = ("live", "staging", "diagnostic")
STRATA = ("recent", "informative", "success", "failure", "delayed", "family")
#: default mixture; a caller's ``strata`` argument replaces it (weights are renormalised over non-empty strata)
DEFAULT_STRATA: Dict[str, float] = {"recent": 0.30, "informative": 0.20, "success": 0.125, "failure": 0.125,
                                    "delayed": 0.15, "family": 0.10}
ANY_LOSS = "*"                      # ``sample(loss="*")``: no eligibility filter (inspection, anchors)
_FINAL = (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED, OutcomeStatus.UNOBSERVABLE)
_MATURED = (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED)
_STATIC = ("exp_id", "objective_id", "decision", "context_item_refs", "goal_text", "objective_family",
           "provenance", "seq")
_RESULT = ("result", "post_item_refs", "duration_s")
#: fields that are history (fixed at decision time) although they are stored in the dynamic part for layout reasons
_IMMUTABLE_DYN = ("lifetime",)
#: ``decision.diagnostics`` keys that describe how the decision was made: never added or changed after the record
#: was stored (other keys are annotations that may be *added*, never changed or removed)
PROTECTED_DIAGNOSTICS = ("arm", "arms", "open_ended", "tracked_proposals", "proposal_probs", "context_item_refs",
                         "lifetime")
#: a stored result in one of these states is a placeholder (the action has not been executed): it may be replaced
_PLACEHOLDER_RESULTS = (ResultStatus.NEEDS_APPROVAL,)
#: index columns checked against the decoded body on every read (column, accessor)
_INDEX_CHECK = (("exp_id", lambda r: r.exp_id), ("seq", lambda r: int(r.seq)), ("provenance", lambda r: r.provenance),
                ("objective_id", lambda r: r.objective_id), ("decision_id", lambda r: r.decision.decision_id),
                ("action_id", lambda r: r.decision.chosen.action_id), ("ts", lambda r: float(r.decision.ts)),
                ("family", lambda r: r.objective_family), ("experiment_id", lambda r: r.decision.experiment_id),
                ("window_id", lambda r: r.window_id))
_COLS = ", ".join(c for c, _ in _INDEX_CHECK) + ", static_body, result_body, dyn_body"


class StoreLockedError(RuntimeError):
    """Another process holds the writer lock of this store."""


class ImmutableHistoryError(ValueError):
    """``update`` tried to rewrite history: the decision, its probabilities, context, provenance or another field
    fixed when the record was stored, or a stored outcome / query definition. Nothing was written."""

    def __init__(self, exp_id: str, problems: Sequence[str]):
        self.exp_id, self.problems = exp_id, list(problems)
        super().__init__(f"record {exp_id}: refused to rewrite history ({'; '.join(self.problems[:6])})")


class StoreIntegrityError(RuntimeError):
    """An index column of a stored record disagrees with the record's stored content."""


def _diff_paths(a: Any, b: Any, path: str, out: List[str], limit: int = 8) -> None:
    """Paths at which two JSON values differ (at most ``limit``)."""
    if len(out) >= limit:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}" + (" added" if k not in a else " removed"))
            else:
                _diff_paths(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _diff_paths(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(path)


def _query_defs(part: Dict[str, Any]) -> Dict[str, Any]:
    """query_id -> definition over pending_queries, queries and result.expected_outcomes of a record dict."""
    out: Dict[str, Any] = {}
    res = part.get("result") or {}
    for q in list(part.get("pending_queries") or []) + list(part.get("queries") or []) + \
            list(res.get("expected_outcomes") or []):
        out.setdefault(q.get("query_id"), q)
    return out


def history_changes(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    """What an update from record dict ``old`` to ``new`` would rewrite of the record's history ([] = allowed).

    Immutable: every static field (decision with its candidates, probabilities, time and diagnostics that describe
    how it was made; context refs; goal; family; provenance; seq) and ``lifetime``. ``decision.diagnostics`` may
    gain *new* annotation keys (not :data:`PROTECTED_DIAGNOSTICS`). The result is write-once (a missing result or a
    ``needs_approval`` placeholder may be replaced by the executed one); afterwards only queries may be appended to
    ``result.expected_outcomes``, ``post_item_refs`` may only grow and ``duration_s`` may only be set from 0. Stored
    final outcomes are append-only (an unverified one may be replaced by the verified measurement of the same query),
    and so are query definitions (``pending_queries`` / ``queries`` / ``result.expected_outcomes`` by query id).
    Credit, eligibility, labels, intrinsic rewards and the meta-learning window are derived data and may change."""
    problems: List[str] = []
    for k in _STATIC + _IMMUTABLE_DYN:
        if k not in old and k not in new:
            continue
        a, b = old.get(k), new.get(k)
        if k == "decision" and isinstance(a, dict) and isinstance(b, dict):
            da, db = dict(a.get("diagnostics") or {}), dict(b.get("diagnostics") or {})
            for key in sorted(set(da) | set(db)):
                if key not in db:
                    problems.append(f"decision.diagnostics.{key} removed")
                elif key not in da:
                    if key in PROTECTED_DIAGNOSTICS:
                        problems.append(f"decision.diagnostics.{key} added after the decision was stored")
                elif _dumps(da[key]) != _dumps(db[key]):
                    problems.append(f"decision.diagnostics.{key} changed")
            a = {x: y for x, y in a.items() if x != "diagnostics"}
            b = {x: y for x, y in b.items() if x != "diagnostics"}
        if _dumps(a) != _dumps(b):
            sub: List[str] = []
            _diff_paths(a, b, k, sub)
            problems += sub or [k]
    ra, rb = old.get("result"), new.get("result")
    placeholder = ra is None or (isinstance(ra, dict) and ra.get("status") in _PLACEHOLDER_RESULTS)
    if not placeholder:
        if rb is None:
            problems.append("result removed")
        else:
            xa = {k: v for k, v in ra.items() if k != "expected_outcomes"}
            xb = {k: v for k, v in rb.items() if k != "expected_outcomes"}
            if _dumps(xa) != _dumps(xb):
                sub = []
                _diff_paths(xa, xb, "result", sub)
                problems += sub or ["result"]
        pa, pb = list(old.get("post_item_refs") or []), list(new.get("post_item_refs") or [])
        if pb[:len(pa)] != pa:
            problems.append("post_item_refs rewritten (only appending is allowed)")
        da_, db_ = float(old.get("duration_s") or 0.0), float(new.get("duration_s") or 0.0)
        if da_ != 0.0 and db_ != da_:
            problems.append("duration_s changed")
        qa, qb = _query_defs(old), _query_defs(new)
        for qid, q in qa.items():
            if qid not in qb:
                problems.append(f"query {qid} removed")
            elif _dumps(q) != _dumps(qb[qid]):
                problems.append(f"query {qid} redefined")
    new_out = {}
    for o in new.get("outcomes") or []:
        new_out.setdefault(o.get("query_id"), []).append(o)
    for o in old.get("outcomes") or []:
        if o.get("status") not in _FINAL:
            continue
        cands = new_out.get(o.get("query_id"), [])
        if any(_dumps(o) == _dumps(c) for c in cands):
            continue
        if not o.get("verified") and any(c.get("verified") and c.get("status") in _FINAL for c in cands):
            continue                                   # the verifier's measurement replaces an unverified one
        problems.append(f"stored outcome of query {o.get('query_id')} removed or rewritten")
    return problems


@dataclass
class AddOutcomeResult:
    accepted: bool
    exp_id: str = ""
    duplicate: bool = False
    upgraded: bool = False          # an unverified stored observation was replaced by a verified one
    final: bool = False
    reason: str = ""
    dedup_key: str = ""


@dataclass
class _Index:
    exp_id: np.ndarray
    seq: np.ndarray
    family: np.ndarray
    success: np.ndarray             # 1 / 0 / -1 (unknown)
    delayed: np.ndarray
    score: np.ndarray               # informativeness: stored surprise + credit magnitude
    cache: Dict[str, Any] = field(default_factory=dict)      # per-index derived arrays (strata members, weights)


# ---- fast record decoding --------------------------------------------------------------------------------------
# ``contracts.Record.from_dict`` resolves type hints on every call (about 6 ms for one ExperienceRecord with eight
# candidates), which dominated replay sampling. The same conversion with the per-class plan built once:
_PLANS: Dict[type, Dict[str, Any]] = {}


def _converter(tp: Any):
    """Converter for a value of annotated type ``tp``, or None when the JSON value can be used as it is."""
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return _converter(args[0]) if len(args) == 1 else None
    if origin in (list, collections.abc.Sequence):
        c = _converter((typing.get_args(tp) or (Any,))[0])
        return None if c is None else (lambda v: [None if x is None else c(x) for x in v])
    if origin is tuple:
        args = typing.get_args(tp)
        c = _converter(args[0]) if len(args) == 2 and args[1] is Ellipsis else None
        return (lambda v: tuple(v)) if c is None else (lambda v: tuple(None if x is None else c(x) for x in v))
    if origin is dict:
        args = typing.get_args(tp)
        c = _converter(args[1]) if len(args) == 2 else None
        return None if c is None else (lambda v: {k: None if x is None else c(x) for k, x in v.items()})
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return lambda v: record_from_dict(tp, v) if isinstance(v, dict) else v
    return None


def record_from_dict(cls: type, d: Dict[str, Any]):
    """Equivalent of ``cls.from_dict(d)`` for contract records, with cached type plans."""
    plan = _PLANS.get(cls)
    if plan is None:
        hints = typing.get_type_hints(cls)
        plan = {f.name: _converter(hints.get(f.name, Any)) for f in dataclasses.fields(cls)}
        _PLANS[cls] = plan
    kw = {}
    for k, v in d.items():
        if k in plan:
            c = plan[k]
            kw[k] = v if (c is None or v is None) else c(v)
    return cls(**kw)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS items(ref TEXT PRIMARY KEY, kind TEXT, ts REAL, sha TEXT, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiences(
  exp_id TEXT PRIMARY KEY, seq INTEGER NOT NULL UNIQUE, decision_id TEXT NOT NULL, action_id TEXT, objective_id TEXT,
  family TEXT,
  provenance TEXT NOT NULL, window_id TEXT, ts REAL, level TEXT, model_version TEXT, reward_version TEXT,
  checkpoint_id TEXT, experiment_id TEXT, success INTEGER, success_explicit INTEGER NOT NULL DEFAULT 0,
  delayed INTEGER NOT NULL DEFAULT 0, n_pending INTEGER NOT NULL DEFAULT 0, n_outcomes INTEGER NOT NULL DEFAULT 0,
  surprise REAL NOT NULL DEFAULT 0, surprise_explicit INTEGER NOT NULL DEFAULT 0, credit_mag REAL NOT NULL DEFAULT 0,
  static_body TEXT NOT NULL, static_sha TEXT NOT NULL, result_body TEXT NOT NULL, result_sha TEXT NOT NULL,
  dyn_body TEXT NOT NULL, dyn_sha TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS exp_decision ON experiences(decision_id);
CREATE INDEX IF NOT EXISTS exp_action ON experiences(action_id);
CREATE INDEX IF NOT EXISTS exp_window ON experiences(window_id, seq);
CREATE INDEX IF NOT EXISTS exp_prov ON experiences(provenance, seq);
CREATE INDEX IF NOT EXISTS exp_objective ON experiences(objective_id, seq);
CREATE TABLE IF NOT EXISTS eligibility(loss TEXT NOT NULL, exp_id TEXT NOT NULL, flag INTEGER NOT NULL,
  PRIMARY KEY(loss, exp_id)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS queries(query_id TEXT PRIMARY KEY, exp_id TEXT NOT NULL, due_at REAL NOT NULL,
  metric TEXT, resolved INTEGER NOT NULL DEFAULT 0, body TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS queries_due ON queries(resolved, due_at);
CREATE INDEX IF NOT EXISTS queries_exp ON queries(exp_id);
CREATE TABLE IF NOT EXISTS outcomes(dedup_key TEXT NOT NULL, query_id TEXT NOT NULL, exp_id TEXT NOT NULL, status TEXT,
  value REAL, observed_at REAL, verified INTEGER, body TEXT NOT NULL, PRIMARY KEY(dedup_key, query_id));
CREATE INDEX IF NOT EXISTS outcomes_exp ON outcomes(exp_id);
CREATE TABLE IF NOT EXISTS windows(window_id TEXT PRIMARY KEY, kind TEXT, opened_at REAL, closed_at REAL, meta TEXT);
CREATE TABLE IF NOT EXISTS versions(kind TEXT NOT NULL, version TEXT NOT NULL, first_seq INTEGER, first_ts REAL,
  meta TEXT, PRIMARY KEY(kind, version));
CREATE TABLE IF NOT EXISTS predictions(prediction_id TEXT PRIMARY KEY, decision_id TEXT, query_id TEXT, made_at REAL,
  resolved INTEGER NOT NULL DEFAULT 0, body TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS predictions_query ON predictions(query_id);
CREATE TABLE IF NOT EXISTS experiments(experiment_id TEXT PRIMARY KEY, created_at REAL, status TEXT, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS links(link_id TEXT PRIMARY KEY, outcome_key TEXT NOT NULL, exp_id TEXT NOT NULL,
  weight REAL NOT NULL, uncertainty REAL NOT NULL, evidence TEXT, created_at REAL, body TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS links_outcome ON links(outcome_key);
CREATE INDEX IF NOT EXISTS links_exp ON links(exp_id);
CREATE TABLE IF NOT EXISTS retrain_queue(exp_id TEXT NOT NULL, reason TEXT NOT NULL, outcome_key TEXT NOT NULL DEFAULT '',
  enqueued_at REAL, PRIMARY KEY(exp_id, reason, outcome_key));
CREATE TABLE IF NOT EXISTS credit_ledger(outcome_key TEXT NOT NULL, exp_id TEXT NOT NULL, level TEXT NOT NULL,
  weight REAL NOT NULL, counted INTEGER NOT NULL, note TEXT, PRIMARY KEY(outcome_key, exp_id));
CREATE TABLE IF NOT EXISTS objective_outcomes(objective_id TEXT PRIMARY KEY, status TEXT, value REAL, evaluated_at REAL,
  verified INTEGER NOT NULL DEFAULT 0, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS credit_history(exp_id TEXT NOT NULL, estimator TEXT NOT NULL, version INTEGER NOT NULL,
  value REAL, uncertainty REAL, ts REAL, reason TEXT, PRIMARY KEY(exp_id, estimator, version));
"""


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def outcome_key(o: OutcomeObservation) -> str:
    """The identity of an observed event: its ``dedup_key``, else one final answer per query."""
    return o.dedup_key or f"q:{o.query_id}"


class SqliteExperienceStore:
    """Durable ``contracts.ExperienceStore``. See the module docstring for the guarantees.

    path                 SQLite file (``contracts.RunFiles.STORE`` inside a run directory)
    provenance           provenance(s) ``sample`` draws from when the caller names none; ``None`` = the provenance of
                         the newest record (one stratum, never a mixture)
    durable              ``True``: ``synchronous=FULL`` (power-loss safe). ``False``: ``NORMAL`` (still safe against a
                         killed process, not against power loss) for diagnostics
    blob_threshold       record parts whose JSON exceeds this many bytes are stored by reference in ``<path>.blobs/``
    eligible_default     eligibility of a record that has no entry for the sampled loss (see module docstring)
    item_conflict        what ``put_item`` does when a ref is already stored with different content: "keep_first"
                         (default: history stays as it was, a warning is emitted and counted), "raise", or "replace"
                         (only for stores whose items are not used as decision context)
    strict_items         ``items`` raises ``KeyError`` for refs that were never stored instead of skipping them with a
                         warning (the reference ``MemoryStore`` skips silently)
    """

    def __init__(self, path: str, *, provenance: Union[None, str, Sequence[str]] = None, seed: int = 0,
                 recent_window: int = 256, recent_fraction: float = 0.25, priority_alpha: float = 0.6,
                 uniform_floor: float = 0.1, delayed_min_s: float = 1.0, blob_threshold: int = 64 * 1024,
                 durable: bool = True, readonly: bool = False, eligible_default: bool = True,
                 item_conflict: str = "keep_first", strict_items: bool = False,
                 strata: Optional[Dict[str, float]] = None):
        self.path = os.path.abspath(path)
        self.blob_dir = self.path + ".blobs"
        self.readonly = bool(readonly)
        self.default_provenance = (provenance,) if isinstance(provenance, str) else \
            (tuple(provenance) if provenance else None)
        self.recent_window, self.recent_fraction = int(recent_window), float(recent_fraction)
        self.priority_alpha, self.uniform_floor = float(priority_alpha), float(uniform_floor)
        self.delayed_min_s = float(delayed_min_s)
        self.blob_threshold = int(blob_threshold)
        if item_conflict not in ("keep_first", "raise", "replace"):
            raise ValueError(item_conflict)
        self.item_conflict, self.strict_items = item_conflict, bool(strict_items)
        self.eligible_default = bool(eligible_default)
        self.stats: Dict[str, int] = {"item_conflicts": 0, "missing_item_refs": 0}
        self.default_strata = dict(strata or DEFAULT_STRATA)
        self.last_sample_info: Dict[str, Any] = {}
        self._seed, self._draws = int(seed), 0
        self._lock = threading.RLock()
        self._depth, self._poisoned = 0, False
        self._gen = 0
        self._index_cache: Dict[Tuple, Tuple[int, _Index]] = {}
        self._lockfile = None
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not self.readonly:
            self._lockfile = open(self.path + ".lock", "a+")
            try:
                fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._lockfile.close()
                self._lockfile = None
                raise StoreLockedError(f"experience store {self.path} already has a writer")
            os.makedirs(self.blob_dir, exist_ok=True)
        uri = f"file:{self.path}" + ("?mode=ro" if self.readonly else "")
        self._db = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False, timeout=30.0)
        self._db.execute("PRAGMA busy_timeout=30000")
        if not self.readonly:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(f"PRAGMA synchronous={'FULL' if durable else 'NORMAL'}")
            self._db.executescript(_SCHEMA)
            row = self._db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
            if row is None:
                self._db.execute("INSERT INTO meta VALUES('schema', ?)", (str(SCHEMA_VERSION),))
            elif int(row[0]) != SCHEMA_VERSION:
                raise RuntimeError(f"store schema {row[0]} != {SCHEMA_VERSION}")
            st = self._db.execute("SELECT value FROM meta WHERE key='sampler'").fetchone()
            if st is not None:
                d = json.loads(st[0])
                self._seed, self._draws = int(d.get("seed", self._seed)), int(d.get("draws", 0))

    # ------------------------------------------------------------------------------------------ infrastructure
    @contextlib.contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        if self.readonly:
            raise PermissionError("store opened read-only")
        with self._lock:
            if self._depth == 0:
                self._db.execute("BEGIN IMMEDIATE")
                self._poisoned = False
            self._depth += 1
            try:
                yield self._db
            except BaseException:
                self._poisoned = True
                raise
            finally:
                self._depth -= 1
                if self._depth == 0:
                    if self._poisoned:
                        self._db.execute("ROLLBACK")
                    else:
                        self._db.execute("COMMIT")
                    self._gen += 1

    def batch(self):
        """Group several writes into one transaction (one fsync). Nothing inside is acknowledged before exit."""
        return self._tx()

    def _q(self, sql: str, args: Sequence[Any] = ()) -> List[tuple]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                if not self.readonly:
                    self._save_sampler()
                self._db.close()
                self._db = None
            if self._lockfile is not None:
                fcntl.flock(self._lockfile, fcntl.LOCK_UN)
                self._lockfile.close()
                self._lockfile = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def flush(self) -> None:
        """Checkpoint the WAL into the main file and persist the sampler counters."""
        if self.readonly:
            return
        with self._lock:
            self._save_sampler()
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            fsync_dir(os.path.dirname(self.path))

    def integrity_ok(self) -> bool:
        return self._q("PRAGMA integrity_check")[0][0] == "ok"

    # ------------------------------------------------------------------------------------------ blobs
    def _blob_path(self, sha: str) -> str:
        return os.path.join(self.blob_dir, sha[:2], sha)

    def put_blob(self, data: bytes) -> str:
        """Store ``data`` content-addressed (durably) and return its reference ``blob:<sha256>``."""
        sha = hashlib.sha256(data).hexdigest()
        p = self._blob_path(sha)
        if not os.path.exists(p):
            atomic_write_bytes(p, zlib.compress(data, 3))
        return "blob:" + sha

    def get_blob(self, ref: str) -> bytes:
        sha = ref[5:] if ref.startswith("blob:") else ref
        with open(self._blob_path(sha), "rb") as fh:
            data = zlib.decompress(fh.read())
        if hashlib.sha256(data).hexdigest() != sha:
            raise IOError(f"blob {sha} is corrupt")
        return data

    def _pack(self, text: str) -> str:
        if len(text) > self.blob_threshold:
            return self.put_blob(text.encode())
        return text

    def _unpack(self, s: str) -> str:
        return self.get_blob(s).decode() if s.startswith("blob:") else s

    # ------------------------------------------------------------------------------------------ items
    def put_item(self, item: ExperienceItem) -> None:
        """Store one context item under ``item.ref`` (assigned from the content hash when empty). Re-putting identical
        content is a no-op. Different content under a stored ref does not replace it (``item_conflict``): rewriting
        an item would change the history ``H_t`` of every earlier decision that references it. Use a new ref for a
        new version of something."""
        body = _dumps(item.to_dict())
        sha = _sha(body)
        if not item.ref:
            item.ref = "item_" + sha[:20]
            body = _dumps(item.to_dict())
            sha = _sha(body)
        with self._tx() as db:
            row = db.execute("SELECT sha FROM items WHERE ref=?", (item.ref,)).fetchone()
            if row is not None:
                if row[0] == sha:
                    return
                self.stats["item_conflicts"] += 1
                if self.item_conflict == "raise":
                    raise ValueError(f"item {item.ref!r} is already stored with different content (items are immutable)")
                if self.item_conflict == "keep_first":
                    if self.stats["item_conflicts"] == 1:
                        warnings.warn(f"item {item.ref!r} is already stored with different content; the stored "
                                      "version is kept (history is immutable). Further conflicts are only counted "
                                      "in store.stats['item_conflicts'].", stacklevel=2)
                    return
            db.execute("INSERT OR REPLACE INTO items VALUES(?,?,?,?,?)",
                       (item.ref, item.kind, float(item.ts), sha, self._pack(body)))

    def put_items(self, items: Iterable[ExperienceItem]) -> None:
        with self._tx():
            for it in items:
                self.put_item(it)

    def items(self, refs: Sequence[str], *, missing_ok: Optional[bool] = None) -> List[ExperienceItem]:
        """Items in the order of ``refs``. Refs that were never stored are skipped, counted in
        ``stats['missing_item_refs']`` and warned about once (a shortened context changes what the model is trained
        on); with ``strict_items`` / ``missing_ok=False`` they raise ``KeyError`` instead."""
        refs = list(refs)
        found: Dict[str, str] = {}
        for i in range(0, len(refs), 500):
            chunk = refs[i:i + 500]
            for ref, body in self._q(f"SELECT ref, body FROM items WHERE ref IN ({','.join('?' * len(chunk))})", chunk):
                found[ref] = body
        missing = [r for r in refs if r not in found]
        if missing:
            if missing_ok is False or (missing_ok is None and self.strict_items):
                raise KeyError(f"{len(missing)} item refs are not in the store, first: {missing[:3]}")
            if missing_ok is None:
                if self.stats["missing_item_refs"] == 0:
                    warnings.warn(f"{len(missing)} context item refs are not in the store (first: {missing[:3]}); "
                                  "they are skipped. Further misses are only counted in "
                                  "store.stats['missing_item_refs'].", stacklevel=2)
                self.stats["missing_item_refs"] += len(missing)
        return [record_from_dict(ExperienceItem, json.loads(self._unpack(found[r]))) for r in refs if r in found]

    def item_times(self, refs: Sequence[str]) -> Dict[str, float]:
        refs, out = list(refs), {}
        for i in range(0, len(refs), 500):
            chunk = refs[i:i + 500]
            out.update(self._q(f"SELECT ref, ts FROM items WHERE ref IN ({','.join('?' * len(chunk))})", chunk))
        return out

    # ------------------------------------------------------------------------------------------ records: write
    def _derive(self, rec: ExperienceRecord, queries: Dict[str, OutcomeQuery]) -> Dict[str, Any]:
        d = rec.decision
        res = rec.result
        end = res.ended if (res is not None and res.ended) else d.ts + rec.duration_s
        resolved = {o.query_id for o in rec.outcomes if o.status in _FINAL}
        matured = [o for o in rec.outcomes if o.status in _MATURED]
        delayed = any((o.observed_at or 0.0) > end + self.delayed_min_s for o in matured)
        success: Optional[int] = None
        if res is not None and (res.status != ResultStatus.OK or res.verified is False):
            success = 0
        else:
            for o in matured:
                q = queries.get(o.query_id)
                if q is None or not q.goal_relevant or o.value is None or o.status != OutcomeStatus.OBSERVED:
                    continue
                if q.kind == OutcomeKind.BINARY:
                    success = max(success or 0, int(o.value >= 0.5))
                elif q.kind == OutcomeKind.COUNT:
                    success = max(success or 0, int(o.value > 0))
        surprise = 0.0
        for ev in d.candidates:
            if ev.action.action_id == d.chosen.action_id:
                for p in ev.predictions:
                    if p.nll is not None and math.isfinite(p.nll):
                        surprise = max(surprise, float(p.nll))
        mags = [abs(v) for v in rec.credit.values() if v is not None and math.isfinite(v)]
        return {"success": success, "delayed": int(delayed), "surprise": surprise,
                "credit_mag": max(mags) if mags else 0.0,
                "n_pending": sum(1 for q in queries if q not in resolved), "n_outcomes": len(matured),
                "resolved": resolved}

    @staticmethod
    def _queries_of(rec: ExperienceRecord) -> Dict[str, OutcomeQuery]:
        qs: Dict[str, OutcomeQuery] = {q.query_id: q for q in rec.pending_queries}
        if rec.result is not None:
            for q in rec.result.expected_outcomes:
                qs.setdefault(q.query_id, q)
        return qs

    @staticmethod
    def _dedup_outcomes(rec: ExperienceRecord) -> None:
        """Within a record each query keeps one final answer (the first; a verified one wins)."""
        best: Dict[str, OutcomeObservation] = {}
        order: List[Any] = []
        for i, o in enumerate(rec.outcomes):
            if o.status not in _FINAL:
                order.append(o)
            elif o.query_id not in best:
                best[o.query_id] = o
                order.append(o.query_id)
            elif o.verified and not best[o.query_id].verified:
                best[o.query_id] = o
        if len(order) != len(rec.outcomes):
            rec.outcomes = [best[x] if isinstance(x, str) else x for x in order]

    def _parts(self, rec: ExperienceRecord) -> Tuple[str, str, str]:
        d = rec.to_dict()
        static = _dumps({k: d[k] for k in _STATIC})
        result = _dumps({k: d[k] for k in _RESULT})
        dyn = _dumps({k: v for k, v in d.items() if k not in _STATIC and k not in _RESULT})
        return static, result, dyn

    def _write(self, rec: ExperienceRecord, *, new: bool) -> None:
        if rec.provenance not in PROVENANCES:
            raise ValueError(f"provenance {rec.provenance!r} is not one of {PROVENANCES}")
        with self._tx() as db:
            old = db.execute("SELECT seq, static_sha, result_sha, dyn_sha, success, success_explicit, surprise, "
                             "surprise_explicit, static_body, result_body, dyn_body FROM experiences WHERE exp_id=?",
                             (rec.exp_id,)).fetchone()
            if new and old is not None:
                raise KeyError(f"duplicate experience {rec.exp_id} (use update)")
            if not new and old is None:
                raise KeyError(f"experience {rec.exp_id} is not in the store (use append)")
            if new:
                head = db.execute("SELECT COALESCE(MAX(seq), 0) FROM experiences").fetchone()[0]
                if rec.seq <= 0:
                    rec.seq = head + 1
                elif rec.seq <= head:
                    raise ValueError(f"seq {rec.seq} is not after the lifetime head {head}")
            else:
                if rec.seq not in (0, old[0]):
                    self.stats["refused_updates"] = self.stats.get("refused_updates", 0) + 1
                    raise ImmutableHistoryError(rec.exp_id, [f"seq {old[0]} -> {rec.seq}"])
                rec.seq = old[0]                      # the lifetime position never changes
            self._dedup_outcomes(rec)
            queries = self._queries_of(rec)
            ix = self._derive(rec, queries)
            static, result, dyn = self._parts(rec)
            shas = (_sha(static), _sha(result), _sha(dyn))
            if not new:
                # history is immutable: only derived data and appended outcomes / queries may change
                # both sides through the current contract (fields added to the contract later get their defaults)
                prev = json.loads(_dumps(self._decode(old[8], old[9], old[10]).to_dict()))
                cur = json.loads(_dumps(rec.to_dict()))
                if shas[0] == old[1]:                 # identical static part: nothing to compare there
                    for k in _STATIC:
                        prev.pop(k, None)
                        cur.pop(k, None)
                problems = history_changes(prev, cur)
                if problems:
                    self.stats["refused_updates"] = self.stats.get("refused_updates", 0) + 1
                    raise ImmutableHistoryError(rec.exp_id, problems)
            d = rec.decision
            if new:
                db.execute(
                    "INSERT INTO experiences(exp_id, seq, decision_id, action_id, objective_id, family, provenance, window_id, ts, "
                    "level, model_version, reward_version, checkpoint_id, experiment_id, success, delayed, n_pending, "
                    "n_outcomes, surprise, credit_mag, static_body, static_sha, result_body, result_sha, dyn_body, "
                    "dyn_sha) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rec.exp_id, rec.seq, d.decision_id, d.chosen.action_id, rec.objective_id, rec.objective_family, rec.provenance,
                     rec.window_id, float(d.ts), d.level, d.model_version, rec.reward_version or d.reward_version,
                     d.checkpoint_id, d.experiment_id, ix["success"], ix["delayed"], ix["n_pending"], ix["n_outcomes"],
                     ix["surprise"], ix["credit_mag"], self._pack(static), shas[0], self._pack(result), shas[1],
                     self._pack(dyn), shas[2]))
            else:
                sets, args = [], []
                for col, text, sha, old_sha in (("static", static, shas[0], old[1]), ("result", result, shas[1], old[2]),
                                                ("dyn", dyn, shas[2], old[3])):
                    if sha != old_sha:           # only changed parts are rewritten
                        sets += [f"{col}_body=?", f"{col}_sha=?"]
                        args += [self._pack(text), sha]
                success = old[4] if old[5] else ix["success"]
                surprise = old[6] if old[7] else ix["surprise"]
                sets += ["window_id=?", "family=?", "reward_version=?", "success=?", "delayed=?", "n_pending=?",
                         "n_outcomes=?", "surprise=?", "credit_mag=?"]
                args += [rec.window_id, rec.objective_family, rec.reward_version or d.reward_version, success,
                         ix["delayed"], ix["n_pending"], ix["n_outcomes"], surprise, ix["credit_mag"]]
                db.execute(f"UPDATE experiences SET {', '.join(sets)} WHERE exp_id=?", args + [rec.exp_id])
            for kind, v in (("model", d.model_version), ("reward", rec.reward_version or d.reward_version)):
                if v:
                    db.execute("INSERT OR IGNORE INTO versions VALUES(?,?,?,?,?)", (kind, v, rec.seq, float(d.ts), "{}"))
            # eligibility: every explicit flag is indexed
            db.execute("DELETE FROM eligibility WHERE exp_id=?", (rec.exp_id,))
            db.executemany("INSERT INTO eligibility VALUES(?,?,?)",
                           [(loss, rec.exp_id, int(bool(ok))) for loss, ok in rec.eligible.items()])
            for q in queries.values():
                db.execute("INSERT INTO queries(query_id, exp_id, due_at, metric, resolved, body) VALUES(?,?,?,?,?,?) "
                           "ON CONFLICT(query_id) DO UPDATE SET resolved=MAX(resolved, excluded.resolved)",
                           (q.query_id, rec.exp_id, float(q.due_at), q.metric, int(q.query_id in ix["resolved"]),
                            _dumps(q.to_dict())))
            for o in rec.outcomes:
                if o.status in _FINAL:
                    db.execute("INSERT INTO outcomes VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(dedup_key, query_id) DO UPDATE "
                               "SET status=excluded.status, value=excluded.value, observed_at=excluded.observed_at, "
                               "verified=excluded.verified, body=excluded.body WHERE excluded.verified > outcomes.verified",
                               (outcome_key(o), o.query_id, rec.exp_id, o.status, o.value, float(o.observed_at or 0.0),
                                int(o.verified), _dumps(o.to_dict())))

    def append(self, rec: ExperienceRecord) -> None:
        """Add a new record. ``rec.seq`` is assigned (lifetime head + 1) when it is 0. Durable on return."""
        self._write(rec, new=True)

    def update(self, rec: ExperienceRecord) -> None:
        """Store the new state of a record (same ``exp_id``); unchanged parts are not rewritten. Only appended
        outcomes / queries / annotations and derived data may differ from the stored record; anything else raises
        :class:`ImmutableHistoryError` and nothing is written (:func:`history_changes` lists the rules)."""
        self._write(rec, new=False)

    def add_outcome(self, obs: OutcomeObservation, exp_id: str = "") -> AddOutcomeResult:
        """Attach a measured outcome to the record that registered its query (or to ``exp_id``).

        De-duplication: a second observation with the same (dedup key, query) is the same event and is dropped; a
        verified observation replaces an unverified stored one. ``PENDING`` observations are not outcomes."""
        key = outcome_key(obs)
        if obs.status == OutcomeStatus.PENDING:
            return AddOutcomeResult(False, reason="pending is not an outcome", dedup_key=key)
        with self._tx() as db:
            qrow = db.execute("SELECT exp_id, body FROM queries WHERE query_id=?", (obs.query_id,)).fetchone()
            target = exp_id or (qrow[0] if qrow else "")
            if not target:
                return AddOutcomeResult(False, reason="no record registered this query", dedup_key=key)
            if qrow is not None and obs.status in _MATURED:
                q = record_from_dict(OutcomeQuery, json.loads(qrow[1]))
                if obs.observed_at and obs.observed_at + 1e-9 < q.issued_at:
                    return AddOutcomeResult(False, target, reason="observed before the query was issued", dedup_key=key)
            # one final answer per query: a repeated event (same dedup key) or a second answer to an answered query
            # is dropped, unless it is the verifier's measurement replacing an unverified one
            old = db.execute("SELECT MAX(verified) FROM outcomes WHERE query_id=?", (obs.query_id,)).fetchone()
            upgraded = False
            if old is not None and old[0] is not None:
                if not (obs.verified and not old[0]):
                    return AddOutcomeResult(False, target, duplicate=True, reason="duplicate event", dedup_key=key)
                upgraded = True
            rec = self.get(target)
            if rec is None:
                return AddOutcomeResult(False, reason=f"record {target} not found", dedup_key=key)
            if upgraded:
                rec.outcomes = [o for o in rec.outcomes if o.query_id != obs.query_id]
                db.execute("DELETE FROM outcomes WHERE query_id=?", (obs.query_id,))
            # the query definition stays in ``pending_queries`` / ``result.expected_outcomes``: it is what makes
            # the outcome interpretable later (``learning_api.matured_outcomes``); 'pending' is derived from the
            # absence of a final outcome, not from list membership.
            rec.outcomes.append(obs)
            self._write(rec, new=False)
            db.execute("UPDATE queries SET resolved=1 WHERE query_id=?", (obs.query_id,))
            return AddOutcomeResult(True, target, upgraded=upgraded, final=obs.status in _FINAL, dedup_key=key)

    # ------------------------------------------------------------------------------------------ records: read
    def _decode(self, static: str, result: str, dyn: str) -> ExperienceRecord:
        """Raw decoder of the three stored parts (no index check; reads go through :meth:`_row`)."""
        d = json.loads(self._unpack(static))
        d.update(json.loads(self._unpack(result)))
        d.update(json.loads(self._unpack(dyn)))
        return record_from_dict(ExperienceRecord, d)

    def _row(self, row: Sequence[Any]) -> ExperienceRecord:
        """Decode a row selected with :data:`_COLS` and verify that every index column agrees with the stored
        content (the sampler, provenance strata and lookups read the columns): a disagreement raises
        :class:`StoreIntegrityError` instead of returning a record that the index misdescribes."""
        n = len(_INDEX_CHECK)
        rec = self._decode(*row[n:n + 3])
        bad = []
        for (col, get), v in zip(_INDEX_CHECK, row[:n]):
            want = get(rec)
            if (v if v is not None else "") != want:
                bad.append(f"{col}: index {v!r} != body {want!r}")
        if bad:
            raise StoreIntegrityError(f"record {row[0]}: index columns disagree with the stored record ({'; '.join(bad)})")
        return rec

    def verify_index(self) -> List[str]:
        """Every record whose index columns disagree with its stored content ([] = consistent)."""
        out = []
        for row in self._q(f"SELECT {_COLS} FROM experiences ORDER BY seq"):
            try:
                self._row(row)
            except StoreIntegrityError as e:
                out.append(str(e))
        return out

    def get(self, exp_id: str) -> Optional[ExperienceRecord]:
        rows = self._q(f"SELECT {_COLS} FROM experiences WHERE exp_id=?", (exp_id,))
        return self._row(rows[0]) if rows else None

    def get_many(self, exp_ids: Sequence[str]) -> List[ExperienceRecord]:
        """Records in the order of ``exp_ids`` (unknown ids are skipped)."""
        ids, found = list(exp_ids), {}
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            for row in self._q(f"SELECT {_COLS} FROM experiences WHERE exp_id IN ({','.join('?' * len(chunk))})",
                               chunk):
                found[row[0]] = row
        return [self._row(found[e]) for e in ids if e in found]

    def by_decision(self, decision_id: str) -> Optional[ExperienceRecord]:
        rows = self._q(f"SELECT {_COLS} FROM experiences WHERE decision_id=? ORDER BY seq LIMIT 1", (decision_id,))
        return self._row(rows[0]) if rows else None

    def by_action(self, action_id: str) -> Optional[ExperienceRecord]:
        """The record whose executed action has ``action_id``."""
        rows = self._q(f"SELECT {_COLS} FROM experiences WHERE action_id=? ORDER BY seq LIMIT 1", (action_id,))
        return self._row(rows[0]) if rows else None

    def count(self, *, provenance: Union[None, str, Sequence[str]] = None, loss: str = "") -> int:
        sql, args = "SELECT COUNT(*) FROM experiences e", []
        where = []
        if loss and loss != ANY_LOSS:
            sql += " LEFT JOIN eligibility g ON g.exp_id=e.exp_id AND g.loss=?"
            args.append(loss)
            where.append("COALESCE(g.flag, ?)=1")
            args.append(int(self.eligible_default))
        if provenance:
            prov = (provenance,) if isinstance(provenance, str) else tuple(provenance)
            where.append(f"e.provenance IN ({','.join('?' * len(prov))})")
            args += list(prov)
        if where:
            sql += " WHERE " + " AND ".join(where)
        return int(self._q(sql, args)[0][0])

    def all(self) -> List[ExperienceRecord]:
        """Every record in lifetime order (same as the reference ``MemoryStore.all``)."""
        return self.iter_records()

    def head_seq(self) -> int:
        return int(self._q("SELECT COALESCE(MAX(seq), 0) FROM experiences")[0][0])

    def iter_records(self, *, objective_id: str = "", provenance: Union[None, str, Sequence[str]] = None,
                     since_seq: int = 0, until_seq: Optional[int] = None, limit: Optional[int] = None,
                     ) -> List[ExperienceRecord]:
        """Records in lifetime order, optionally restricted to one objective / provenance / seq range."""
        sql, args = f"SELECT {_COLS} FROM experiences WHERE seq>?", [int(since_seq)]
        if until_seq is not None:
            sql += " AND seq<=?"
            args.append(int(until_seq))
        if objective_id:
            sql += " AND objective_id=?"
            args.append(objective_id)
        if provenance:
            prov = (provenance,) if isinstance(provenance, str) else tuple(provenance)
            sql += f" AND provenance IN ({','.join('?' * len(prov))})"
            args += list(prov)
        sql += " ORDER BY seq"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [self._row(r) for r in self._q(sql, args)]

    def trajectory(self, objective_id: str) -> List[ExperienceRecord]:
        """All decisions of one objective in order (the sequence a return decomposition runs over)."""
        return self.iter_records(objective_id=objective_id)

    # ------------------------------------------------------------------------------------------ pending outcomes
    def pending(self, now: float) -> List[Tuple[ExperienceRecord, OutcomeQuery]]:
        """Unresolved queries that are due at ``now`` with the record that registered them, oldest due first."""
        rows = self._q("SELECT exp_id, body FROM queries WHERE resolved=0 AND due_at<=? ORDER BY due_at, query_id",
                       (float(now),))
        recs = {r.exp_id: r for r in self.get_many(list({e for e, _ in rows}))}
        return [(recs[e], record_from_dict(OutcomeQuery, json.loads(b))) for e, b in rows if e in recs]

    def pending_all(self) -> List[OutcomeQuery]:
        return [record_from_dict(OutcomeQuery, json.loads(b)) for (b,) in
                self._q("SELECT body FROM queries WHERE resolved=0 ORDER BY due_at, query_id")]

    def next_due(self) -> Optional[float]:
        """Earliest due time among unresolved queries (what the runtime sleeps until), or ``None``."""
        v = self._q("SELECT MIN(due_at) FROM queries WHERE resolved=0")[0][0]
        return None if v is None else float(v)

    def pending_pointer(self) -> Dict[str, Any]:
        """Small summary stored in checkpoints: the queue itself lives durably in this store."""
        n = int(self._q("SELECT COUNT(*) FROM queries WHERE resolved=0")[0][0])
        return {"store": self.path, "head_seq": self.head_seq(), "n_pending": n, "next_due": self.next_due(),
                "n_retrain": self.retrain_count()}

    def query(self, query_id: str) -> Optional[OutcomeQuery]:
        rows = self._q("SELECT body FROM queries WHERE query_id=?", (query_id,))
        return record_from_dict(OutcomeQuery, json.loads(rows[0][0])) if rows else None

    def queries_for_metric(self, metric: str, at: float) -> List[Tuple[str, OutcomeQuery]]:
        """(exp_id, query) of every registered query on ``metric`` whose declared window [issued_at, due_at] contains
        ``at`` — the attribution windows that decisions declared for that metric."""
        out = []
        for e, b in self._q("SELECT exp_id, body FROM queries WHERE metric=? AND due_at>=? ORDER BY due_at, query_id",
                            (metric, float(at))):
            q = record_from_dict(OutcomeQuery, json.loads(b))
            if q.issued_at <= at:
                out.append((e, q))
        return out

    def predecessor_ids(self, exp_id: str, k: int) -> List[str]:
        """Ids of the ``k`` decisions of the same objective that precede ``exp_id`` (nearest first)."""
        row = self._q("SELECT objective_id, seq FROM experiences WHERE exp_id=?", (exp_id,))
        if not row or k <= 0:
            return []
        return [e for (e,) in self._q("SELECT exp_id FROM experiences WHERE objective_id=? AND seq<? ORDER BY seq DESC "
                                      "LIMIT ?", (row[0][0], row[0][1], int(k)))]

    def outcome_events(self, *, verified_only: bool = False) -> List[Dict[str, Any]]:
        """Distinct observed events (one per dedup key) with the records they are attached to. Independent-outcome
        counts must use this, not row counts."""
        sql = ("SELECT dedup_key, GROUP_CONCAT(exp_id), MAX(verified), MIN(observed_at), MAX(value), COUNT(*) FROM "
               "outcomes WHERE status IN ('observed','censored')")
        if verified_only:
            sql += " AND verified=1"
        return [{"dedup_key": k, "exp_ids": sorted(set(e.split(","))), "verified": bool(v), "observed_at": t,
                 "value": val, "n_rows": n} for k, e, v, t, val, n in self._q(sql + " GROUP BY dedup_key ORDER BY 4")]

    # ------------------------------------------------------------------------------------------ windows / versions
    def window(self, window_id: str) -> List[ExperienceRecord]:
        return [self._row(r) for r in self._q(
            f"SELECT {_COLS} FROM experiences WHERE window_id=? ORDER BY seq", (window_id,))]

    def assign_window(self, exp_ids: Sequence[str], window_id: str, *, kind: str = "", opened_at: float = 0.0,
                      closed_at: float = 0.0, meta: Optional[Dict[str, Any]] = None) -> int:
        """Put records into a (meta-learning) window. Returns the number of records moved."""
        n = 0
        with self._tx() as db:
            db.execute("INSERT INTO windows VALUES(?,?,?,?,?) ON CONFLICT(window_id) DO UPDATE SET "
                       "kind=excluded.kind, closed_at=excluded.closed_at, meta=excluded.meta",
                       (window_id, kind, float(opened_at), float(closed_at), _dumps(meta or {})))
            for rec in self.get_many(exp_ids):
                if rec.window_id != window_id:
                    rec.window_id = window_id
                    self._write(rec, new=False)
                    n += 1
        return n

    def windows(self) -> List[Dict[str, Any]]:
        rows = self._q("SELECT e.window_id, COUNT(*), MIN(e.seq), MAX(e.seq), w.kind, w.opened_at, w.closed_at FROM "
                       "experiences e LEFT JOIN windows w ON w.window_id=e.window_id WHERE e.window_id<>'' "
                       "GROUP BY e.window_id ORDER BY MIN(e.seq)")
        return [{"window_id": w, "n": n, "first_seq": a, "last_seq": b, "kind": k or "", "opened_at": o, "closed_at": c}
                for w, n, a, b, k, o, c in rows]

    def versions(self, kind: str = "") -> List[Dict[str, Any]]:
        """Model / reward versions seen, with the number of records decided under each."""
        col = {"model": "model_version", "reward": "reward_version"}
        out = []
        for k in ([kind] if kind else ["model", "reward"]):
            counts = dict(self._q(f"SELECT {col[k]}, COUNT(*) FROM experiences GROUP BY 1"))
            for v, first_seq, first_ts in self._q("SELECT version, first_seq, first_ts FROM versions WHERE kind=? "
                                                  "ORDER BY first_seq", (k,)):
                out.append({"kind": k, "version": v, "first_seq": first_seq, "first_ts": first_ts,
                            "n_records": int(counts.get(v, 0))})
        return out

    def provenance_counts(self) -> Dict[str, int]:
        return {p: int(n) for p, n in self._q("SELECT provenance, COUNT(*) FROM experiences GROUP BY 1")}

    # ------------------------------------------------------------------------------------------ labels / eligibility
    def set_eligible(self, exp_ids: Sequence[str], loss: str, flag: bool = True) -> int:
        """Set ``eligible[loss]`` on several records (rewrites only their dynamic part)."""
        n = 0
        with self._tx():
            for rec in self.get_many(exp_ids):
                if rec.eligible.get(loss) is not flag:
                    rec.eligible[loss] = bool(flag)
                    self._write(rec, new=False)
                    n += 1
        return n

    def refresh_eligibility(self, loss: str, predicate: Callable[[ExperienceRecord], bool], *, since_seq: int = 0,
                            exp_ids: Optional[Sequence[str]] = None) -> Dict[str, int]:
        """Re-evaluate ``predicate`` (a ``LossProvider.eligible``) on records and store the flags."""
        recs = self.get_many(exp_ids) if exp_ids is not None else self.iter_records(since_seq=since_seq)
        changed = n_true = 0
        with self._tx():
            for rec in recs:
                ok = bool(predicate(rec))
                n_true += ok
                if rec.eligible.get(loss) is not ok:
                    rec.eligible[loss] = ok
                    self._write(rec, new=False)
                    changed += 1
        return {"checked": len(recs), "eligible": n_true, "changed": changed}

    def set_success(self, exp_id: str, success: Optional[bool]) -> None:
        """Explicit success / failure label of a record (e.g. from a verified objective outcome). It overrides the
        label derived from execution status and goal-relevant binary / count outcomes; ``None`` returns to derived."""
        with self._tx() as db:
            if success is None:
                db.execute("UPDATE experiences SET success_explicit=0 WHERE exp_id=?", (exp_id,))
                rec = self.get(exp_id)
                if rec is not None:
                    self._write(rec, new=False)
            else:
                db.execute("UPDATE experiences SET success=?, success_explicit=1 WHERE exp_id=?", (int(success), exp_id))

    def set_surprise(self, exp_id: str, surprise: float) -> None:
        """Stored informativeness of a record (prediction NLL, update magnitude ...) used by the ``informative``
        stratum. Without it the largest resolved prediction NLL of the executed candidate is used."""
        with self._tx() as db:
            db.execute("UPDATE experiences SET surprise=?, surprise_explicit=1 WHERE exp_id=?", (float(surprise), exp_id))

    def set_credit(self, exp_id: str, estimator: str, value: Optional[float], uncertainty: Optional[float] = None) -> None:
        """Write one estimator's retrospective credit for a record (``None`` removes it: no credit is not zero)."""
        with self._tx():
            rec = self.get(exp_id)
            if rec is None:
                raise KeyError(exp_id)
            if value is None:
                rec.credit.pop(estimator, None)
                rec.credit_uncertainty.pop(estimator, None)
            else:
                rec.credit[estimator] = float(value)
                if uncertainty is not None:
                    rec.credit_uncertainty[estimator] = float(uncertainty)
            self._write(rec, new=False)

    def set_credits(self, exp_id: str, values: Dict[str, Optional[Tuple[float, Optional[float]]]], *, version: int = 0,
                    reason: str = "", ts: float = 0.0) -> Dict[str, float]:
        """Write several estimators' credit for one record in one transaction and append each *changed* value to
        the credit history under ``version`` (``credit.providers.CreditAssigner`` supplies a monotone version).
        ``values[name] = (value, uncertainty)``; ``None`` removes the entry (no credit is not zero credit).
        Returns estimator -> change of the stored value (new entries count with their full value)."""
        changes: Dict[str, float] = {}
        with self._tx() as db:
            rec = self.get(exp_id)
            if rec is None:
                raise KeyError(exp_id)
            for name, v in values.items():
                old = rec.credit.get(name)
                if v is None:
                    if old is None:
                        continue
                    rec.credit.pop(name, None)
                    rec.credit_uncertainty.pop(name, None)
                    changes[name] = -float(old)
                    db.execute("INSERT OR REPLACE INTO credit_history VALUES(?,?,?,?,?,?,?)",
                               (exp_id, name, int(version), None, None, float(ts), reason))
                    continue
                val, unc = float(v[0]), (None if v[1] is None else float(v[1]))
                if not math.isfinite(val):
                    raise ValueError(f"credit {name} for {exp_id} is not finite")
                old_unc = rec.credit_uncertainty.get(name)
                if old is not None and old == val and (unc is None or old_unc == unc):
                    continue
                rec.credit[name] = val
                if unc is not None:
                    rec.credit_uncertainty[name] = unc
                changes[name] = val - (old or 0.0)
                db.execute("INSERT OR REPLACE INTO credit_history VALUES(?,?,?,?,?,?,?)",
                           (exp_id, name, int(version), val, unc, float(ts), reason))
            if changes:
                self._write(rec, new=False)
        return changes

    def credit_history(self, exp_id: str = "", estimator: str = "") -> List[Dict[str, Any]]:
        """Every stored version of a record's credit, oldest first (``value`` None = the entry was withdrawn)."""
        sql, args = "SELECT exp_id, estimator, version, value, uncertainty, ts, reason FROM credit_history WHERE 1=1", []
        if exp_id:
            sql += " AND exp_id=?"
            args.append(exp_id)
        if estimator:
            sql += " AND estimator=?"
            args.append(estimator)
        try:
            rows = self._q(sql + " ORDER BY version, exp_id, estimator", args)
        except sqlite3.OperationalError:          # a read-only store written before this table existed
            return []
        return [dict(zip(("exp_id", "estimator", "version", "value", "uncertainty", "ts", "reason"), r)) for r in rows]

    # ------------------------------------------------------------------------------------------ objectives / small state
    def put_objective_outcome(self, objective_id: str, *, status: str, value: float = 0.0, evaluated_at: float = 0.0,
                              verified: bool = False, body: Optional[Dict[str, Any]] = None) -> None:
        """Record that an objective was closed with this (verified) result. A closed objective is what makes a
        trajectory's return final for return decomposition; a verified statement is never replaced by an unverified
        one."""
        with self._tx() as db:
            old = db.execute("SELECT verified FROM objective_outcomes WHERE objective_id=?", (objective_id,)).fetchone()
            if old is not None and old[0] and not verified:
                return
            db.execute("INSERT OR REPLACE INTO objective_outcomes VALUES(?,?,?,?,?,?)",
                       (objective_id, status, float(value), float(evaluated_at), int(bool(verified)), _dumps(body or {})))

    def objective_outcome(self, objective_id: str) -> Optional[Dict[str, Any]]:
        try:
            rows = self._q("SELECT status, value, evaluated_at, verified, body FROM objective_outcomes WHERE "
                           "objective_id=?", (objective_id,))
        except sqlite3.OperationalError:
            return None
        if not rows:
            return None
        st, v, t, ver, body = rows[0]
        return {"objective_id": objective_id, "status": st, "value": v, "evaluated_at": t, "verified": bool(ver),
                "body": json.loads(body)}

    def objective_ids(self, *, closed: Optional[bool] = None, family: str = "",
                      provenance: Union[None, str, Sequence[str]] = None) -> List[str]:
        """Objective ids in order of their first decision; ``closed`` filters on a recorded objective outcome."""
        sql, args = "SELECT e.objective_id, MIN(e.seq) FROM experiences e", []
        where = []
        if closed is not None:
            sql += " LEFT JOIN objective_outcomes o ON o.objective_id=e.objective_id"
            where.append("o.objective_id IS NOT NULL" if closed else "o.objective_id IS NULL")
        if family:
            where.append("e.family=?")
            args.append(family)
        if provenance:
            prov = (provenance,) if isinstance(provenance, str) else tuple(provenance)
            where.append(f"e.provenance IN ({','.join('?' * len(prov))})")
            args += list(prov)
        if where:
            sql += " WHERE " + " AND ".join(where)
        return [o for o, _ in self._q(sql + " GROUP BY e.objective_id ORDER BY 2", args)]

    def objective_of(self, exp_ids: Sequence[str]) -> Dict[str, str]:
        """exp_id -> objective_id without decoding the records."""
        out: Dict[str, str] = {}
        ids = list(exp_ids)
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            out.update(self._q(f"SELECT exp_id, objective_id FROM experiences WHERE exp_id IN ({','.join('?' * len(chunk))})",
                               chunk))
        return out

    def kv_get(self, key: str, default: Any = None) -> Any:
        """Small JSON-native state kept durably next to the experience (credit version counter, trigger state)."""
        rows = self._q("SELECT value FROM meta WHERE key=?", ("kv:" + key,))
        return json.loads(rows[0][0]) if rows else default

    def kv_put(self, key: str, value: Any) -> None:
        with self._tx() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", ("kv:" + key, _dumps(value)))

    def labels(self, exp_id: str) -> Dict[str, Any]:
        r = self._q("SELECT seq, provenance, family, success, success_explicit, delayed, n_pending, n_outcomes, surprise, "
                    "credit_mag FROM experiences WHERE exp_id=?", (exp_id,))
        if not r:
            raise KeyError(exp_id)
        k = ("seq", "provenance", "family", "success", "success_explicit", "delayed", "n_pending", "n_outcomes",
             "surprise", "credit_mag")
        return dict(zip(k, r[0]))

    # ------------------------------------------------------------------------------------------ predictions / experiments
    def put_prediction(self, p: PredictionRecord) -> None:
        with self._tx() as db:
            db.execute("INSERT OR REPLACE INTO predictions VALUES(?,?,?,?,?,?)",
                       (p.prediction_id, p.decision_id, p.query.query_id, float(p.made_at),
                        int(p.resolved_status != OutcomeStatus.PENDING), self._pack(_dumps(p.to_dict()))))

    def predictions(self, *, query_id: str = "", decision_id: str = "", unresolved_only: bool = False
                    ) -> List[PredictionRecord]:
        sql, args = "SELECT body FROM predictions WHERE 1=1", []
        for col, v in (("query_id", query_id), ("decision_id", decision_id)):
            if v:
                sql += f" AND {col}=?"
                args.append(v)
        if unresolved_only:
            sql += " AND resolved=0"
        return [record_from_dict(PredictionRecord, json.loads(self._unpack(b))) for (b,) in self._q(sql + " ORDER BY made_at", args)]

    def put_experiment(self, e: ExperimentRecord) -> None:
        with self._tx() as db:
            db.execute("INSERT OR REPLACE INTO experiments VALUES(?,?,?,?)",
                       (e.experiment_id, float(e.created_at), e.status, self._pack(_dumps(e.to_dict()))))

    def experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        rows = self._q("SELECT body FROM experiments WHERE experiment_id=?", (experiment_id,))
        return record_from_dict(ExperimentRecord, json.loads(self._unpack(rows[0][0]))) if rows else None

    def experiments(self) -> List[ExperimentRecord]:
        return [record_from_dict(ExperimentRecord, json.loads(self._unpack(b))) for (b,) in
                self._q("SELECT body FROM experiments ORDER BY created_at")]

    def experiment_objectives(self, *, randomized_only: bool = True) -> Dict[str, List[str]]:
        """objective_id -> ids of the registered experiments that assigned one of its decisions (the lifetimes that
        experiment evidence is measured on). ``randomized_only``: only experiments registered as randomised."""
        exps = {e.experiment_id: e for e in self.experiments()}
        out: Dict[str, List[str]] = {}
        for oid, xid in self._q("SELECT DISTINCT objective_id, experiment_id FROM experiences WHERE experiment_id<>'' "
                                "ORDER BY objective_id, experiment_id"):
            e = exps.get(xid)
            if e is None or (randomized_only and e.assignment != "randomized"):
                continue
            out.setdefault(oid or "", []).append(xid)
        return out

    def experiment_records(self, experiment_id: str) -> List[ExperienceRecord]:
        return [self._row(r) for r in self._q(
            f"SELECT {_COLS} FROM experiences WHERE experiment_id=? ORDER BY seq", (experiment_id,))]

    # ------------------------------------------------------------------------------------------ links / retrain queue
    def put_links(self, links: Sequence[Dict[str, Any]]) -> None:
        """Persist outcome->decision links (dicts with link_id, outcome_key, exp_id, weight, uncertainty, evidence,
        created_at). Used by ``credit.links.OutcomeLinker``."""
        with self._tx() as db:
            for l in links:
                db.execute("INSERT OR REPLACE INTO links VALUES(?,?,?,?,?,?,?,?)",
                           (l["link_id"], l["outcome_key"], l["exp_id"], float(l["weight"]), float(l["uncertainty"]),
                            l.get("evidence", ""), float(l.get("created_at", 0.0)), _dumps(l)))

    def links(self, *, outcome_key: str = "", exp_id: str = "") -> List[Dict[str, Any]]:
        sql, args = "SELECT body FROM links WHERE 1=1", []
        if outcome_key:
            sql += " AND outcome_key=?"
            args.append(outcome_key)
        if exp_id:
            sql += " AND exp_id=?"
            args.append(exp_id)
        return [json.loads(b) for (b,) in self._q(sql + " ORDER BY created_at, link_id", args)]

    def mark_for_retraining(self, exp_ids: Sequence[str], reason: str, *, outcome_key: str = "", now: float = 0.0) -> int:
        """Queue earlier records whose targets changed (an outcome matured, credit was revised)."""
        with self._tx() as db:
            before = db.total_changes
            db.executemany("INSERT OR IGNORE INTO retrain_queue VALUES(?,?,?,?)",
                           [(e, reason, outcome_key, float(now)) for e in exp_ids])
            return db.total_changes - before

    def retrain_queue(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Queued records, oldest first: ``[{"exp_id", "reasons": [...], "outcome_keys": [...], "enqueued_at"}]``."""
        out: Dict[str, Dict[str, Any]] = {}
        for e, reason, key, t in self._q("SELECT exp_id, reason, outcome_key, enqueued_at FROM retrain_queue ORDER BY "
                                         "enqueued_at, exp_id"):
            d = out.setdefault(e, {"exp_id": e, "reasons": [], "outcome_keys": [], "enqueued_at": t})
            if reason not in d["reasons"]:
                d["reasons"].append(reason)
            if key and key not in d["outcome_keys"]:
                d["outcome_keys"].append(key)
        res = list(out.values())
        return res if limit is None else res[:limit]

    def retrain_count(self) -> int:
        return int(self._q("SELECT COUNT(DISTINCT exp_id) FROM retrain_queue")[0][0])

    def pop_retrain(self, exp_ids: Sequence[str]) -> int:
        """Acknowledge records as re-trained (call after the update that used them has been applied)."""
        with self._tx() as db:
            before = db.total_changes
            db.executemany("DELETE FROM retrain_queue WHERE exp_id=?", [(e,) for e in exp_ids])
            return db.total_changes - before

    # ------------------------------------------------------------------------------------------ credit ledger
    def ledger_assign(self, outcome_key: str, exp_id: str, level: str, weight: float, *, counted: bool = True,
                      note: str = "", tol: float = 1e-9) -> None:
        """Record that ``weight`` of one outcome event is credited to ``exp_id``. The counted weights of an event can
        never exceed 1: an outcome is credited once (``credit.smdp``)."""
        with self._tx() as db:
            other = db.execute("SELECT COALESCE(SUM(weight), 0) FROM credit_ledger WHERE outcome_key=? AND counted=1 AND "
                               "exp_id<>?", (outcome_key, exp_id)).fetchone()[0]
            if counted and other + weight > 1.0 + tol:
                raise ValueError(f"outcome {outcome_key!r} would be credited {other + weight:.6f} > 1 times")
            db.execute("INSERT OR REPLACE INTO credit_ledger VALUES(?,?,?,?,?,?)",
                       (outcome_key, exp_id, level, float(weight), int(counted), note))

    def ledger(self, outcome_key: str = "") -> List[Dict[str, Any]]:
        sql, args = "SELECT outcome_key, exp_id, level, weight, counted, note FROM credit_ledger", []
        if outcome_key:
            sql += " WHERE outcome_key=?"
            args.append(outcome_key)
        return [dict(zip(("outcome_key", "exp_id", "level", "weight", "counted", "note"), r)) | {"counted": bool(r[4])}
                for r in self._q(sql + " ORDER BY outcome_key, exp_id", args)]

    # ------------------------------------------------------------------------------------------ sampling
    def _resolve_provenance(self, provenance: Union[None, str, Sequence[str]]) -> Tuple[str, ...]:
        if provenance:
            return (provenance,) if isinstance(provenance, str) else tuple(provenance)
        if self.default_provenance:
            return self.default_provenance
        rows = self._q("SELECT provenance FROM experiences ORDER BY seq DESC LIMIT 1")
        return (rows[0][0],) if rows else ("live",)

    def _index(self, loss: str, prov: Tuple[str, ...]) -> _Index:
        key = (loss, prov)
        hit = self._index_cache.get(key)
        if hit is not None and hit[0] == self._gen and not self.readonly:
            return hit[1]
        ph = ",".join("?" * len(prov))
        if loss == ANY_LOSS:
            rows = self._q("SELECT exp_id, seq, family, success, delayed, surprise, credit_mag FROM experiences WHERE "
                           f"provenance IN ({ph}) ORDER BY seq", prov)
        else:
            rows = self._q("SELECT e.exp_id, e.seq, e.family, e.success, e.delayed, e.surprise, e.credit_mag FROM "
                           "experiences e LEFT JOIN eligibility g ON g.exp_id=e.exp_id AND g.loss=? WHERE "
                           f"COALESCE(g.flag, ?)=1 AND e.provenance IN ({ph}) ORDER BY e.seq",
                           (loss, int(self.eligible_default)) + prov)
        ix = _Index(exp_id=np.array([r[0] for r in rows], dtype=object),
                    seq=np.array([r[1] for r in rows], dtype=np.int64),
                    family=np.array([r[2] or "" for r in rows], dtype=object),
                    success=np.array([-1 if r[3] is None else r[3] for r in rows], dtype=np.int64),
                    delayed=np.array([r[4] for r in rows], dtype=np.int64),
                    score=np.array([max(r[5], 0.0) + r[6] for r in rows], dtype=np.float64))
        if len(self._index_cache) > 16:
            self._index_cache.clear()
        self._index_cache[key] = (self._gen, ix)
        return ix

    @staticmethod
    def _quotas(n: int, weights: Dict[str, float]) -> Dict[str, int]:
        tot = sum(weights.values())
        raw = {k: n * w / tot for k, w in weights.items()}
        q = {k: int(math.floor(v)) for k, v in raw.items()}
        rest = n - sum(q.values())
        for k in sorted(raw, key=lambda k: (-(raw[k] - q[k]), STRATA.index(k)))[:rest]:
            q[k] += 1
        return q

    def sample_ids(self, n: int, *, loss: str, strata: Optional[Dict[str, float]] = None, seed: Optional[int] = None,
                   provenance: Union[None, str, Sequence[str]] = None) -> List[str]:
        """Ids of up to ``n`` distinct eligible records, balanced across strata. Deterministic given the store
        content and ``seed``; with ``seed=None`` the store's own counter-based stream is used (see ``state_dict``)."""
        with self._lock:
            prov = self._resolve_provenance(provenance)
            ix = self._index(loss, prov)
            N = len(ix.seq)
            tag = zlib.crc32(("|".join((loss,) + prov)).encode())
            if seed is None:
                rng = np.random.default_rng([abs(self._seed), self._draws, tag])
                self._draws += 1
            else:
                rng = np.random.default_rng([int(seed) & 0xFFFFFFFFFFFF, tag])
            weights = {k: float(v) for k, v in (strata or self.default_strata).items() if v > 0}
            unknown = set(weights) - set(STRATA)
            if unknown:
                raise KeyError(f"unknown strata {sorted(unknown)}; choose from {STRATA}")
            self.last_sample_info = {"provenance": prov, "pool": N, "strata": {}, "requested": n}
            if N == 0 or n <= 0:
                return []
            n = min(int(n), N)
            ckey = (self.recent_window, self.recent_fraction, self.priority_alpha, self.uniform_floor)
            if ix.cache.get("key") != ckey:
                k_recent = max(1, min(self.recent_window, int(math.ceil(N * self.recent_fraction))))
                is_recent = np.zeros(N, dtype=bool)
                is_recent[N - k_recent:] = True
                older = np.flatnonzero(~is_recent)
                inf = older if len(older) else np.arange(N)
                sc = ix.score[inf]
                s0 = float(sc.mean()) or 1.0          # smoothing: unscored records keep a real share
                sp = (sc + s0) ** self.priority_alpha
                _, inv, cnt = np.unique(ix.family.astype(str), return_inverse=True, return_counts=True)
                ix.cache = {"key": ckey, "members": {
                    "recent": np.flatnonzero(is_recent), "informative": inf,
                    "success": np.flatnonzero(ix.success == 1), "failure": np.flatnonzero(ix.success == 0),
                    "delayed": np.flatnonzero(ix.delayed == 1), "family": np.arange(N)},
                    "p": {"informative": self.uniform_floor / len(inf) + (1 - self.uniform_floor) * sp / sp.sum(),
                          "family": 1.0 / (len(cnt) * cnt[inv])}}
            members, probs = ix.cache["members"], ix.cache["p"]
            weights = {k: w for k, w in weights.items() if len(members[k])}
            if not weights:
                weights = {"family": 1.0}
            quotas = self._quotas(n, weights)
            taken = np.zeros(N, dtype=bool)
            picked: List[int] = []
            origin: Dict[str, int] = {}

            def draw(cand: np.ndarray, k: int, p: Optional[np.ndarray] = None) -> np.ndarray:
                cand = cand[~taken[cand]] if p is None else cand
                if p is not None:
                    keep = ~taken[cand]
                    cand, p = cand[keep], p[keep]
                k = min(k, len(cand))
                if k <= 0:
                    return cand[:0]
                if p is None:
                    return rng.choice(cand, size=k, replace=False)
                # Efraimidis-Spirakis weighted sampling without replacement
                keys = np.log(rng.random(len(cand))) / np.maximum(p, 1e-300)
                return cand[np.argsort(-keys)[:k]]

            for name in STRATA:
                k = quotas.get(name, 0)
                if k <= 0:
                    continue
                got = draw(members[name], k, probs.get(name))
                taken[got] = True
                picked += got.tolist()
                origin[name] = len(got)
            if len(picked) < n:                      # strata that ran dry: fill uniformly over the whole lifetime
                got = draw(np.flatnonzero(~taken), n - len(picked))
                taken[got] = True
                picked += got.tolist()
                origin["fill"] = len(got)
            order = rng.permutation(len(picked))
            self.last_sample_info["strata"] = origin
            return [ix.exp_id[picked[i]] for i in order]

    def sample(self, n: int, *, loss: str, strata: Optional[Dict[str, float]] = None, seed: Optional[int] = None,
               provenance: Union[None, str, Sequence[str]] = None) -> List[ExperienceRecord]:
        """Replay sample (``contracts.ExperienceStore.sample``): up to ``n`` distinct records eligible for ``loss``,
        drawn from one provenance stratum unless several are named, balanced over ``strata`` (default
        ``DEFAULT_STRATA``: recent, informative older, success, failure, delayed, family)."""
        return self.get_many(self.sample_ids(n, loss=loss, strata=strata, seed=seed, provenance=provenance))

    # ------------------------------------------------------------------------------------------ sampler state
    def _save_sampler(self) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta VALUES('sampler', ?)",
                         (_dumps({"seed": self._seed, "draws": self._draws}),))

    def state_dict(self) -> Dict[str, Any]:
        """Replay indices for a checkpoint: the sampler stream position and the lifetime head it was taken at."""
        return {"version": 1, "seed": self._seed, "draws": self._draws, "head_seq": self.head_seq(),
                "count": self.count(), "strata": dict(self.default_strata), "recent_window": self.recent_window,
                "recent_fraction": self.recent_fraction, "priority_alpha": self.priority_alpha,
                "uniform_floor": self.uniform_floor,
                "default_provenance": list(self.default_provenance) if self.default_provenance else None}

    def load_state_dict(self, st: Dict[str, Any]) -> Dict[str, Any]:
        """Restore the sampler stream. Returns what differs between the checkpoint and the store (records appended
        after the checkpoint are kept: experience is never rolled back with the weights)."""
        with self._lock:
            self._seed, self._draws = int(st["seed"]), int(st["draws"])
            self.default_strata = dict(st.get("strata", self.default_strata))
            for k in ("recent_window", "recent_fraction", "priority_alpha", "uniform_floor"):
                if k in st:
                    setattr(self, k, type(getattr(self, k))(st[k]))
            if not self.readonly:
                self._save_sampler()
            head = self.head_seq()
            return {"records_after_checkpoint": head - int(st.get("head_seq", head)), "head_seq": head}
