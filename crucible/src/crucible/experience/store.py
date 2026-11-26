"""`ExperienceStore`: persistent, append-only experience with content hashes.

Layout of a store directory::

    manifest.jsonl          one line per sealed shard: {shard, records, first, last, bytes, sha256, prev, entry}
                            (`entry` = sha256 over the line without it and `prev` = the previous line's entry: a
                            hash chain over every sealed shard)
    shards/000000.jsonl     sealed shards, never modified again; one record per line:
                            {"i": index, "h": sha256 of the canonical record, "r": record}
    shards/000123.jsonl.part   the open shard (appended line by line, sealed on `seal`/`close` or when full)
    blobs/<sha256>.json     content-addressed side objects shared by many records (tool lists), written once
    HEAD                    {records, shards, head}: the chain head after the last seal (detects a truncated tail)
    writer.lock             held (flock) by the one writer; a second writer is refused (`StoreLocked`)

`append(record)` validates that the record is JSON-native, replaces large repeated sub-objects by blob
references (``{"$blob": "<sha256>"}``, only at the dedupe paths, which each line lists in ``"b"`` - content of the
same shape anywhere else, e.g. a message a participant wrote, is stored and returned verbatim), writes one line and
returns the record hash. `iterate(...)` yields records in order, verifying every line hash, every sealed shard's file
hash, the manifest chain and the HEAD anchor, resolving blob references (blob content is verified against its name).
A reopened store continues the index; an open shard left by a crashed writer is recovered by keeping its valid prefix
and sealing it; a seal interrupted between its manifest line and its rename is completed. A store has one writer at
a time (``writer.lock``); readers open it with ``readonly=True``. An experimenter who needs tamper evidence beyond the
store itself keeps `digest()` elsewhere.

`make_record(...)` builds the standard experience record (format ``crucible-exp/1``) from one interaction.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterator

from .observation import _json_native

FORMAT = "crucible-exp/1"
SHARD_DIR = "shards"
BLOB_DIR = "blobs"
MANIFEST = "manifest.jsonl"
DEDUPE = (("observation", "tools"),)


def canonical(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class StoreCorrupt(RuntimeError):
    pass


class StoreLocked(RuntimeError):
    """Another writer holds the store."""


HEAD = "HEAD"
LOCK = "writer.lock"


class ExperienceStore:
    def __init__(self, root: str, *, shard_records: int = 1000, dedupe: tuple = DEDUPE, fsync: bool = False,
                 readonly: bool = False):
        self.root = os.path.abspath(root)
        self.shard_records = int(shard_records)
        self.dedupe = tuple(tuple(p) for p in dedupe)
        self.fsync = bool(fsync)
        self.readonly = bool(readonly)
        self._lock_fh = None
        if not self.readonly:
            os.makedirs(os.path.join(self.root, SHARD_DIR), exist_ok=True)
            os.makedirs(os.path.join(self.root, BLOB_DIR), exist_ok=True)
            self._lock()
        self._manifest = self._read_manifest()
        self._next = self._manifest[-1]["last"] + 1 if self._manifest else 0
        self._shard = (self._manifest[-1]["shard"] + 1) if self._manifest else 0
        self._fh = None
        self._open_n = 0
        self._open_first = self._next
        self._blobs_known: set[str] = set()
        if not self.readonly:
            self._recover()

    def _lock(self) -> None:
        try:
            import fcntl
        except ImportError:                        # pragma: no cover - non-POSIX: no inter-process lock
            return
        fh = open(os.path.join(self.root, LOCK), "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            fh.close()
            raise StoreLocked(f"{self.root} has another writer (open it with readonly=True to read)") from e
        self._lock_fh = fh

    def _unlock(self) -> None:
        if self._lock_fh is not None:
            try:
                import fcntl
                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_fh.close()
                self._lock_fh = None

    # ------------------------------------------------------------------ manifest
    def _mpath(self) -> str:
        return os.path.join(self.root, MANIFEST)

    def _read_manifest(self) -> list[dict]:
        out = []
        try:
            with open(self._mpath(), encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        out.append(json.loads(line))
        except FileNotFoundError:
            pass
        return out

    def _spath(self, k: int, part: bool = False) -> str:
        return os.path.join(self.root, SHARD_DIR, f"{k:06d}.jsonl" + (".part" if part else ""))

    def _recover(self) -> None:
        # a seal that wrote its manifest line but died before renaming the open shard: finish it
        if self._manifest:
            last = self._manifest[-1]
            final, lpart = self._spath(last["shard"]), self._spath(last["shard"], True)
            if not os.path.exists(final) and os.path.exists(lpart):
                with open(lpart, "rb") as fh:
                    if sha256(fh.read()) == last["sha256"]:
                        os.replace(lpart, final)
        # a shard file of the next index without a manifest line (a writer that predates this order): adopt it
        orphan = self._spath(self._shard)
        if os.path.exists(orphan) and not os.path.exists(self._spath(self._shard, True)):
            os.replace(orphan, self._spath(self._shard, True))
        part = self._spath(self._shard, True)
        if not os.path.exists(part):
            return
        good, n = 0, 0
        with open(part, "rb") as fh:
            data = fh.read()
        pos = 0
        for line in data.split(b"\n"):
            end = pos + len(line) + 1
            if not line:
                break
            try:
                d = json.loads(line)
                ok = d.get("i") == self._next + n and d.get("h") == sha256(canonical(d["r"]))
            except (ValueError, KeyError, TypeError):
                ok = False
            if not ok or end > len(data):
                break
            n += 1
            good = end
            pos = end
        with open(part, "r+b") as fh:          # drop a torn tail left by a crashed writer
            fh.truncate(good)
        self._open_n, self._open_first = n, self._next
        self._next += n
        if n:
            self._seal_part()
        else:
            os.remove(part)

    # ------------------------------------------------------------------ writing
    def _blob(self, value: Any) -> dict:
        b = canonical(value)
        h = sha256(b)
        if h not in self._blobs_known:
            p = os.path.join(self.root, BLOB_DIR, f"{h}.json")
            if not os.path.exists(p):
                tmp = f"{p}.{os.getpid()}.tmp"
                with open(tmp, "wb") as fh:
                    fh.write(b)
                os.replace(tmp, p)
            self._blobs_known.add(h)
        return {"$blob": h}

    def _dedupe(self, rec: dict) -> tuple[dict, list]:
        out = dict(rec)
        used = []
        for path in self.dedupe:
            node, parents = out, []
            for k in path[:-1]:
                if not isinstance(node, dict) or not isinstance(node.get(k), dict):
                    node = None
                    break
                parents.append((node, k))
                node[k] = dict(node[k])
                node = node[k]
            if isinstance(node, dict) and isinstance(node.get(path[-1]), list):
                node[path[-1]] = self._blob(node[path[-1]])
                used.append(list(path))
        return out, used

    def append(self, record: dict) -> str:
        """Append one JSON-native record; returns its content hash (of the record as given, before dedupe)."""
        errs: list[str] = []
        _json_native(record, "record", errs)
        if errs or not isinstance(record, dict):
            raise TypeError(f"experience records must be JSON-native dicts: {errs[:3]}")
        if self.readonly:
            raise StoreLocked("this store was opened readonly")
        h = sha256(canonical(record))
        stored, used = self._dedupe(record)
        line = canonical({"i": self._next, "h": sha256(canonical(stored)), "r": stored, "b": used})
        if self._fh is None:
            self._fh = open(self._spath(self._shard, True), "ab")
            self._open_first = self._next
        self._fh.write(line + b"\n")
        self._next += 1
        self._open_n += 1
        if self._open_n >= self.shard_records:
            self.seal()
        return h

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())

    def _seal_part(self) -> None:
        """Seal the open shard: the manifest line first, then the rename (a crash in between is completed by
        `_recover`), then the HEAD anchor."""
        part, final = self._spath(self._shard, True), self._spath(self._shard)
        with open(part, "rb") as fh:
            data = fh.read()
        prev = self._manifest[-1]["entry"] if self._manifest else "0" * 64
        ent = {"shard": self._shard, "records": self._open_n, "first": self._open_first,
               "last": self._open_first + self._open_n - 1, "bytes": len(data), "sha256": sha256(data), "prev": prev}
        ent["entry"] = sha256(canonical(ent))
        with open(self._mpath(), "ab") as fh:
            fh.write(canonical(ent) + b"\n")
            fh.flush()
            if self.fsync:
                os.fsync(fh.fileno())
        os.replace(part, final)
        self._manifest.append(ent)
        self._write_head()
        self._shard += 1
        self._open_n = 0
        self._open_first = self._next

    def seal(self) -> None:
        """Close the open shard (if it has records) and record it in the manifest."""
        if self._fh is not None:
            self.flush()
            self._fh.close()
            self._fh = None
        if self._open_n:
            self._seal_part()

    def _write_head(self) -> None:
        p = os.path.join(self.root, HEAD)
        tmp = f"{p}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(canonical({"records": self._manifest[-1]["last"] + 1 if self._manifest else 0,
                                "shards": len(self._manifest),
                                "head": self._manifest[-1]["entry"] if self._manifest else "0" * 64}))
        os.replace(tmp, p)

    def close(self) -> None:
        if not self.readonly:
            self.seal()
        self._unlock()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ reading
    def __len__(self) -> int:
        return self._next

    def _load_blob(self, ref: Any) -> Any:
        if not (isinstance(ref, dict) and set(ref) == {"$blob"} and isinstance(ref["$blob"], str)):
            raise StoreCorrupt(f"a dedupe path holds no blob reference: {str(ref)[:80]}")
        h = ref["$blob"]
        try:
            with open(os.path.join(self.root, BLOB_DIR, f"{h}.json"), "rb") as fh:
                b = fh.read()
        except FileNotFoundError as e:
            raise StoreCorrupt(f"blob {h} is missing") from e
        if sha256(b) != h:
            raise StoreCorrupt(f"blob {h} does not match its content")
        return json.loads(b)

    def _resolve(self, r: dict, used: list) -> dict:
        """Replace the blob references at the dedupe paths a line lists (nothing else is ever interpreted)."""
        if not used:
            return r
        out = dict(r)
        for path in used:
            node = out
            for k in path[:-1]:
                node[k] = dict(node[k])
                node = node[k]
            node[path[-1]] = self._load_blob(node[path[-1]])
        return out

    def verify(self) -> dict:
        """Check the manifest chain, every sealed shard's file hash and the HEAD anchor. Returns counts; raises
        StoreCorrupt."""
        prev = "0" * 64
        for ent in self._manifest:
            body = {k: v for k, v in ent.items() if k != "entry"}
            if ent["prev"] != prev or sha256(canonical(body)) != ent["entry"]:
                raise StoreCorrupt(f"manifest chain broken at shard {ent['shard']}")
            with open(self._spath(ent["shard"]), "rb") as fh:
                if sha256(fh.read()) != ent["sha256"]:
                    raise StoreCorrupt(f"shard {ent['shard']} does not match its manifest hash")
            prev = ent["entry"]
        try:
            with open(os.path.join(self.root, HEAD), "rb") as fh:
                anchor = json.loads(fh.read())
        except FileNotFoundError:
            anchor = None
        if anchor is not None and (anchor.get("shards") != len(self._manifest) or anchor.get("head") != prev):
            raise StoreCorrupt(f"store truncated or rewritten: HEAD names {anchor.get('shards')} sealed shards "
                               f"(head {str(anchor.get('head'))[:12]}), the manifest has {len(self._manifest)}")
        return {"shards": len(self._manifest), "records": self._next, "head": prev}

    def iterate(self, *, start: int = 0, agent: str | None = None, env: dict | None = None,
                reward_mode: str | None = None, resolve: bool = True, verify: bool = True) -> Iterator[dict]:
        """Records in order from index `start`, optionally filtered by agent, by env fields (``{"tier": 1}``) or
        reward mode. Every line's hash is checked; sealed shards and the manifest chain when `verify`."""
        self.flush()
        if verify:
            self.verify()
        files = [(e["shard"], self._spath(e["shard"])) for e in self._manifest]
        part = self._spath(self._shard, True)
        if os.path.exists(part):
            files.append((self._shard, part))
        expect = None
        for _, path in files:
            with open(path, "rb") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if expect is not None and d["i"] != expect:
                        raise StoreCorrupt(f"record index gap at {expect}")
                    expect = d["i"] + 1
                    if d["i"] < start:
                        continue
                    if sha256(canonical(d["r"])) != d["h"]:
                        raise StoreCorrupt(f"record {d['i']} does not match its hash")
                    r = d["r"]
                    if agent is not None and r.get("agent") != agent:
                        continue
                    if reward_mode is not None and r.get("reward_mode") != reward_mode:
                        continue
                    if env is not None and any((r.get("env") or {}).get(k) != v for k, v in env.items()):
                        continue
                    if not resolve:
                        yield r
                        continue
                    used = d["b"] if "b" in d else [list(p) for p in self.dedupe if _blob_at(r, p)]   # older lines
                    yield self._resolve(r, used)

    def digest(self) -> str:
        """A content hash of everything appended so far (sealed shards' chain head plus the open shard's bytes)."""
        self.flush()
        head = self._manifest[-1]["entry"] if self._manifest else "0" * 64
        part = self._spath(self._shard, True)
        tail = b""
        if os.path.exists(part):
            with open(part, "rb") as fh:
                tail = fh.read()
        return sha256(head.encode() + tail)

    def stats(self) -> dict:
        size = 0
        for d in (SHARD_DIR, BLOB_DIR):
            for f in os.listdir(os.path.join(self.root, d)):
                size += os.path.getsize(os.path.join(self.root, d, f))
        return {"records": self._next, "sealed_shards": len(self._manifest), "open_records": self._open_n,
                "bytes": size, "blobs": len(os.listdir(os.path.join(self.root, BLOB_DIR)))}


def _blob_at(r: dict, path) -> bool:
    node = r
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return False
        node = node[k]
    return isinstance(node, dict) and set(node) == {"$blob"}


def make_record(env, agent: str, step: int, observation: dict, action: dict | None, result: dict | None,
                reward: float, *, learner: dict | None = None, next_cursor: int | None = None,
                extra: dict | None = None, done: bool = False, episode: int | None = None) -> dict:
    """The standard experience record of one interaction (format ``crucible-exp/1``). `reward` is what the
    environment's reward mode gave (0.0 in observation-only mode); nothing evaluator-side is stored. `done` marks
    the last interaction of an episode (a Tier 1 horizon; the next record starts a new episode) and `episode` numbers
    episodes, so offline learners never chain transitions across an episode reset."""
    d = env.describe()
    rec = {"format": FORMAT, "env": {k: d.get(k) for k in ("tier", "kind", "name", "seed", "world_id", "spec_hash")},
           "reward_mode": env.reward_mode, "agent": agent, "step": int(step), "t": int(observation.get("t", 0)),
           "observation": observation, "action": action, "result": result, "reward": float(reward),
           "next_cursor": None if next_cursor is None else int(next_cursor), "learner": learner or {},
           "done": bool(done), "episode": int(getattr(env, "episode", 0) if episode is None else episode)}
    if extra:
        rec["extra"] = extra
    return rec
