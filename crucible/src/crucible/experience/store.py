"""`ExperienceStore`: persistent, append-only experience with content hashes.

Layout of a store directory::

    manifest.jsonl          one line per sealed shard: {shard, records, first, last, bytes, sha256, prev, entry}
                            (`entry` = sha256 over the line without it and `prev` = the previous line's entry: a
                            hash chain over every sealed shard)
    shards/000000.jsonl     sealed shards, never modified again; one record per line:
                            {"i": index, "h": sha256 of the canonical record, "r": record}
    shards/000123.jsonl.part   the open shard (appended line by line, sealed on `seal`/`close` or when full)
    blobs/<sha256>.json     content-addressed side objects shared by many records (tool lists), written once

`append(record)` validates that the record is JSON-native, replaces large repeated sub-objects by blob
references (``{"$blob": "<sha256>"}``), writes one line and returns the record hash. `iterate(...)` yields records
in order, verifying every line hash, every sealed shard's file hash and the manifest chain, resolving blob
references (blob content is verified against its name). A reopened store continues the index; an open shard left
by a crashed writer is recovered by keeping its valid prefix and sealing it.

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


class ExperienceStore:
    def __init__(self, root: str, *, shard_records: int = 1000, dedupe: tuple = DEDUPE, fsync: bool = False):
        self.root = os.path.abspath(root)
        self.shard_records = int(shard_records)
        self.dedupe = tuple(tuple(p) for p in dedupe)
        self.fsync = bool(fsync)
        os.makedirs(os.path.join(self.root, SHARD_DIR), exist_ok=True)
        os.makedirs(os.path.join(self.root, BLOB_DIR), exist_ok=True)
        self._manifest = self._read_manifest()
        self._next = self._manifest[-1]["last"] + 1 if self._manifest else 0
        self._shard = (self._manifest[-1]["shard"] + 1) if self._manifest else 0
        self._fh = None
        self._open_n = 0
        self._open_first = self._next
        self._blobs_known: set[str] = set()
        self._recover()

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

    def _dedupe(self, rec: dict) -> dict:
        out = dict(rec)
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
        return out

    def append(self, record: dict) -> str:
        """Append one JSON-native record; returns its content hash (of the record as given, before dedupe)."""
        errs: list[str] = []
        _json_native(record, "record", errs)
        if errs or not isinstance(record, dict):
            raise TypeError(f"experience records must be JSON-native dicts: {errs[:3]}")
        h = sha256(canonical(record))
        stored = self._dedupe(record)
        line = canonical({"i": self._next, "h": sha256(canonical(stored)), "r": stored})
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
        part, final = self._spath(self._shard, True), self._spath(self._shard)
        with open(part, "rb") as fh:
            data = fh.read()
        os.replace(part, final)
        prev = self._manifest[-1]["entry"] if self._manifest else "0" * 64
        ent = {"shard": self._shard, "records": self._open_n, "first": self._open_first,
               "last": self._open_first + self._open_n - 1, "bytes": len(data), "sha256": sha256(data), "prev": prev}
        ent["entry"] = sha256(canonical(ent))
        with open(self._mpath(), "ab") as fh:
            fh.write(canonical(ent) + b"\n")
            fh.flush()
            if self.fsync:
                os.fsync(fh.fileno())
        self._manifest.append(ent)
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

    def close(self) -> None:
        self.seal()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ reading
    def __len__(self) -> int:
        return self._next

    def _resolve(self, x: Any) -> Any:
        if isinstance(x, dict):
            if set(x) == {"$blob"}:
                h = x["$blob"]
                with open(os.path.join(self.root, BLOB_DIR, f"{h}.json"), "rb") as fh:
                    b = fh.read()
                if sha256(b) != h:
                    raise StoreCorrupt(f"blob {h} does not match its content")
                return json.loads(b)
            return {k: self._resolve(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self._resolve(v) for v in x]
        return x

    def verify(self) -> dict:
        """Check the manifest chain and every sealed shard's file hash. Returns counts; raises StoreCorrupt."""
        prev = "0" * 64
        for ent in self._manifest:
            body = {k: v for k, v in ent.items() if k != "entry"}
            if ent["prev"] != prev or sha256(canonical(body)) != ent["entry"]:
                raise StoreCorrupt(f"manifest chain broken at shard {ent['shard']}")
            with open(self._spath(ent["shard"]), "rb") as fh:
                if sha256(fh.read()) != ent["sha256"]:
                    raise StoreCorrupt(f"shard {ent['shard']} does not match its manifest hash")
            prev = ent["entry"]
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
                    yield self._resolve(r) if resolve else r

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


def make_record(env, agent: str, step: int, observation: dict, action: dict | None, result: dict | None,
                reward: float, *, learner: dict | None = None, next_cursor: int | None = None,
                extra: dict | None = None) -> dict:
    """The standard experience record of one interaction (format ``crucible-exp/1``). `reward` is what the
    environment's reward mode gave (0.0 in observation-only mode); nothing evaluator-side is stored."""
    d = env.describe()
    rec = {"format": FORMAT, "env": {k: d.get(k) for k in ("tier", "kind", "name", "seed", "world_id", "spec_hash")},
           "reward_mode": env.reward_mode, "agent": agent, "step": int(step), "t": int(observation.get("t", 0)),
           "observation": observation, "action": action, "result": result, "reward": float(reward),
           "next_cursor": None if next_cursor is None else int(next_cursor), "learner": learner or {}}
    if extra:
        rec["extra"] = extra
    return rec
