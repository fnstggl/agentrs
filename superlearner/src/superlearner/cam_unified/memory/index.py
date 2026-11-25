"""MemoryIndex: version-partitioned retrieval keys and memory-token values with exact (brute-force) GPU top-k.

Spec 10.3 / 10.5, architecture §8. Every stored key belongs to exactly one *embedding version* (the identity of the
encoder snapshot that produced it). Searches name a version and only ever touch that version's partition, so keys of
different versions are never compared. When the encoder changes, ``missing(new_version)`` lists the segments without
a key in the new version and ``reencode(new_version, encode_fn)`` recomputes them from raw events through a caller
callback; ``set_current`` promotes a version once it covers every segment.

Storage (``root/``)::

    index.sqlite                       segments (metadata + source links), per-version key rows, entities, state
    versions/<vdir>/keys_<s>.f16       append-only fp16 rows [rows, key_dim]   (read via np.memmap)
    versions/<vdir>/vals_<s>.f16       append-only fp16 rows [rows, value_tokens * value_dim]

Key ids (``kid``) are dense per version (0..n-1); shard s holds kids [s*shard_rows, (s+1)*shard_rows). Rows are
appended + fsynced before their SQLite rows commit; bytes beyond the committed row count (a crash between the two) are
truncated on open. Search = chunked ``q @ K^T`` on the GPU in float32 from fp16 keys with a running top-k merge;
metadata prefilters (time range, entity, env, episode, actor, level, gseq bound) come from SQLite and are applied
either as a mask (broad filters) or by gathering only the candidate keys (narrow filters). Results are exact.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from collections import OrderedDict, namedtuple
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from ._fs import fsync_dir

SearchResult = namedtuple("SearchResult", ["scores", "kids", "seg_ids"])
SEGMENT_FIELDS = ("seg_id", "level", "env_id", "actor_id", "episode_id", "t_start", "t_end", "gseq_start", "gseq_end",
                  "n_events", "event_ids", "child_ids", "artifact_refs", "entities")
FILTER_KEYS = frozenset({"t_min", "t_max", "env_id", "actor_id", "episode_id", "exclude_episode", "entity", "level",
                         "gseq_max", "exclude_seg_ids"})


def _vdir(version: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", version)[:48]
    return f"{safe}-{hashlib.sha256(version.encode()).hexdigest()[:10]}"


class _fp32_matmul:
    """Exact fp32 matmul inside the block (TF32 disabled), restoring the global setting afterwards."""

    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        return self

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.prev


class MemoryIndex:
    """Args:
        root: directory.
        key_dim: retrieval key width (``cfg.mem_key_dim``).
        value_tokens, value_dim: memory tokens per segment and their width (``cfg.mem_segment_tokens``, ``d_model``).
        shard_rows: rows per shard file.
        device: search device (default CUDA when available).
        gpu_cache_bytes: budget of key shards kept resident on the device between searches (LRU).
        search_chunk: keys per matmul chunk.
    """

    def __init__(self, root: str, key_dim: int, value_tokens: int, value_dim: int, shard_rows: int = 65536,
                 device=None, gpu_cache_bytes: int = 1 << 30, search_chunk: int = 65536, readonly: bool = False,
                 fsync: bool = True):
        self.root = os.path.abspath(root)
        self.key_dim, self.value_tokens, self.value_dim = int(key_dim), int(value_tokens), int(value_dim)
        self.shard_rows = int(shard_rows)
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gpu_cache_bytes = int(gpu_cache_bytes)
        self.search_chunk = int(search_chunk)
        self.readonly = readonly
        self.do_fsync = fsync
        self._gpu: "OrderedDict" = OrderedDict()
        self._mm: dict = {}
        self.stats = {"searches": 0, "search_s": 0.0, "added": 0, "reencoded": 0}
        if not readonly:
            os.makedirs(os.path.join(self.root, "versions"), exist_ok=True)
        path = os.path.join(self.root, "index.sqlite")
        if readonly:
            self._db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        else:
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._schema()
            self._check_dims()
            self._truncate_uncommitted()

    # ================================================================== schema
    def _schema(self) -> None:
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS segments (
            seg_id TEXT PRIMARY KEY, level INTEGER, env_id TEXT, actor_id TEXT, episode_id TEXT, t_start REAL,
            t_end REAL, gseq_start INTEGER, gseq_end INTEGER, n_events INTEGER, event_ids TEXT, child_ids TEXT,
            artifact_refs TEXT, created REAL);
        CREATE INDEX IF NOT EXISTS seg_level ON segments(level, gseq_start);
        CREATE INDEX IF NOT EXISTS seg_env ON segments(env_id, t_end);
        CREATE INDEX IF NOT EXISTS seg_episode ON segments(episode_id);
        CREATE TABLE IF NOT EXISTS seg_entities (seg_id TEXT NOT NULL, entity TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS sent_entity ON seg_entities(entity);
        CREATE TABLE IF NOT EXISTS versions (version TEXT PRIMARY KEY, vdir TEXT, n_rows INTEGER, created REAL,
                                             meta TEXT);
        CREATE TABLE IF NOT EXISTS keys (version TEXT NOT NULL, kid INTEGER NOT NULL, seg_id TEXT NOT NULL,
                                         PRIMARY KEY (version, kid));
        CREATE UNIQUE INDEX IF NOT EXISTS keys_seg ON keys(version, seg_id);
        CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS weight_links (seg_id TEXT, update_id TEXT, model_version INTEGER, t REAL);
        """)
        self._db.commit()

    def _check_dims(self) -> None:
        dims = self.get_state("dims")
        mine = {"key_dim": self.key_dim, "value_tokens": self.value_tokens, "value_dim": self.value_dim,
                "shard_rows": self.shard_rows}
        if dims is None:
            self.set_state("dims", mine)
        elif dims != mine:
            raise ValueError(f"index {self.root} was created with {dims}, opened with {mine}")

    def _truncate_uncommitted(self) -> None:
        kw, vw = self.key_dim * 2, self.value_tokens * self.value_dim * 2
        for v, vdir, n in self._db.execute("SELECT version, vdir, n_rows FROM versions").fetchall():
            d = os.path.join(self.root, "versions", vdir)
            for s in range(max(1, -(-n // self.shard_rows)) + 1):
                rows = max(0, min(self.shard_rows, n - s * self.shard_rows))
                for pre, w in (("keys", kw), ("vals", vw)):
                    p = os.path.join(d, f"{pre}_{s:05d}.f16")
                    if os.path.exists(p) and os.path.getsize(p) > rows * w:
                        with open(p, "r+b") as fh:
                            fh.truncate(rows * w)

    # ================================================================== state
    def get_state(self, key: str, default=None):
        r = self._db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return default if r is None else json.loads(r[0])

    def set_state(self, key: str, value, commit: bool = True) -> None:
        self._db.execute("INSERT OR REPLACE INTO state(key, value) VALUES (?,?)", (key, json.dumps(value)))
        if commit:
            self._db.commit()

    @property
    def current_version(self) -> Optional[str]:
        return self.get_state("current_version")

    def set_current(self, version: str) -> None:
        """Promote ``version`` (searches by default go there); refuses a version that does not cover every segment."""
        cov = self.coverage(version)
        if cov < 1.0:
            raise ValueError(f"version {version} covers only {cov:.3f} of the segments; re-encode first")
        self.set_state("current_version", version)

    def versions(self) -> dict:
        return {v: n for v, n in self._db.execute("SELECT version, n_rows FROM versions ORDER BY created")}

    def stale_versions(self) -> list:
        """Versions other than the current one (their keys must not be searched as if current)."""
        cur = self.current_version
        return [v for v in self.versions() if v != cur]

    def count(self, version: Optional[str] = None) -> int:
        if version is None:
            return int(self._db.execute("SELECT COUNT(*) FROM segments").fetchone()[0])
        r = self._db.execute("SELECT n_rows FROM versions WHERE version=?", (version,)).fetchone()
        return 0 if r is None else int(r[0])

    def coverage(self, version: str) -> float:
        n = self.count()
        return 1.0 if n == 0 else self.count(version) / n

    # ================================================================== segments (metadata + source links)
    def add_segments(self, metas: Sequence[dict]) -> int:
        """Register segment metadata (idempotent by ``seg_id``). Fields: ``SEGMENT_FIELDS``."""
        rows, ents = [], []
        now = time.time()
        for m in metas:
            rows.append((m["seg_id"], int(m.get("level", 1)), m.get("env_id"), m.get("actor_id"), m.get("episode_id"),
                         float(m.get("t_start", 0.0)), float(m.get("t_end", 0.0)), m.get("gseq_start"),
                         m.get("gseq_end"), int(m.get("n_events", len(m.get("event_ids") or []))),
                         json.dumps(list(m.get("event_ids") or [])), json.dumps(list(m.get("child_ids") or [])),
                         json.dumps(list(m.get("artifact_refs") or [])), now))
            ents.extend((m["seg_id"], str(e)) for e in (m.get("entities") or []))
        before = self._db.total_changes
        self._db.executemany(f"INSERT OR IGNORE INTO segments VALUES ({','.join('?' * 14)})", rows)
        added = self._db.total_changes - before
        if added:
            new = {r[0] for r in rows}
            have = {r[0] for r in self._db.execute(
                f"SELECT DISTINCT seg_id FROM seg_entities WHERE seg_id IN ({','.join('?' * len(new))})", list(new))}
            self._db.executemany("INSERT INTO seg_entities(seg_id, entity) VALUES (?,?)",
                                 [e for e in ents if e[0] not in have])
        self._db.commit()
        return added

    def segment(self, seg_id: str) -> dict:
        return self.segments([seg_id])[0]

    def segments(self, seg_ids: Sequence[str]) -> list:
        out = {}
        cols = "seg_id, level, env_id, actor_id, episode_id, t_start, t_end, gseq_start, gseq_end, n_events, " \
               "event_ids, child_ids, artifact_refs"
        for i in range(0, len(seg_ids), 900):
            part = list(seg_ids[i:i + 900])
            for r in self._db.execute(f"SELECT {cols} FROM segments WHERE seg_id IN ({','.join('?' * len(part))})",
                                      part):
                d = dict(zip(SEGMENT_FIELDS[:-1], r))
                for k in ("event_ids", "child_ids", "artifact_refs"):
                    d[k] = json.loads(d[k])
                out[d["seg_id"]] = d
            for s, e in self._db.execute(
                    f"SELECT seg_id, entity FROM seg_entities WHERE seg_id IN ({','.join('?' * len(part))})", part):
                out[s].setdefault("entities", []).append(e)
        missing = [s for s in seg_ids if s not in out]
        if missing:
            raise KeyError(f"unknown segments {missing[:3]}")
        for d in out.values():
            d.setdefault("entities", [])
        return [out[s] for s in seg_ids]

    def source_events(self, seg_id: str) -> list:
        """Raw event ids under a segment of any level (follows child links down to level 1)."""
        d = self.segment(seg_id)
        if d["event_ids"]:
            return list(d["event_ids"])
        out = []
        for c in d["child_ids"]:
            out.extend(self.source_events(c))
        return out

    def link_weights(self, seg_ids: Sequence[str], update_id: str, model_version: int) -> None:
        """Parametric level: record that a weight update consolidated these segments (source links into weights)."""
        t = time.time()
        self._db.executemany("INSERT INTO weight_links VALUES (?,?,?,?)",
                             [(s, str(update_id), int(model_version), t) for s in seg_ids])
        self._db.commit()

    def weight_links(self, seg_id: str) -> list:
        return [{"update_id": u, "model_version": v} for u, v in self._db.execute(
            "SELECT update_id, model_version FROM weight_links WHERE seg_id=? ORDER BY rowid", (seg_id,))]

    # ================================================================== keys / values
    def add(self, version: str, seg_ids: Sequence[str], keys, values) -> np.ndarray:
        """Store keys [n, key_dim] and values [n, value_tokens, value_dim] of registered segments under ``version``.
        Segments that already have a key in this version are skipped. Returns the kids (dense per version)."""
        if self.readonly:
            raise PermissionError("index opened read-only")
        seg_ids = list(seg_ids)
        keys = _np16(keys).reshape(len(seg_ids), self.key_dim)
        values = _np16(values).reshape(len(seg_ids), self.value_tokens * self.value_dim)
        known = {r[0] for r in self._db.execute(
            f"SELECT seg_id FROM segments WHERE seg_id IN ({','.join('?' * len(seg_ids))})", seg_ids)} if seg_ids else set()
        unknown = [s for s in seg_ids if s not in known]
        if unknown:
            raise KeyError(f"segments not registered: {unknown[:3]}")
        have = self.kids_of(version, seg_ids)
        sel = np.nonzero(have < 0)[0]
        if len(sel) != len(set(seg_ids[i] for i in sel)):
            raise ValueError("duplicate seg_ids in one add() call")
        r = self._db.execute("SELECT vdir, n_rows FROM versions WHERE version=?", (version,)).fetchone()
        if r is None:
            vdir, n0 = _vdir(version), 0
            self._db.execute("INSERT INTO versions VALUES (?,?,?,?,?)", (version, vdir, 0, time.time(), "{}"))
        else:
            vdir, n0 = r
        if len(sel) == 0:
            self._db.commit()
            return have
        d = os.path.join(self.root, "versions", vdir)
        os.makedirs(d, exist_ok=True)
        K, V = keys[sel], values[sel]
        pos = 0
        while pos < len(sel):
            g = n0 + pos
            s, off = divmod(g, self.shard_rows)
            take = min(len(sel) - pos, self.shard_rows - off)
            for pre, arr in (("keys", K), ("vals", V)):
                p = os.path.join(d, f"{pre}_{s:05d}.f16")
                with open(p, "ab") as fh:
                    if fh.tell() != off * arr.shape[1] * 2:
                        raise IOError(f"shard {p} has {fh.tell()} bytes, expected {off * arr.shape[1] * 2}")
                    fh.write(np.ascontiguousarray(arr[pos:pos + take]).tobytes())
                    fh.flush()
                    if self.do_fsync:
                        os.fsync(fh.fileno())
            self._mm.pop((version, s), None)
            pos += take
        if self.do_fsync:
            fsync_dir(d)
        kids = np.arange(n0, n0 + len(sel), dtype=np.int64)
        self._db.executemany("INSERT INTO keys(version, kid, seg_id) VALUES (?,?,?)",
                             [(version, int(k), seg_ids[i]) for k, i in zip(kids, sel)])
        self._db.execute("UPDATE versions SET n_rows=? WHERE version=?", (n0 + len(sel), version))
        self._db.commit()
        have[sel] = kids
        self.stats["added"] += len(sel)
        return have

    def kids_of(self, version: str, seg_ids: Sequence[str]) -> np.ndarray:
        out = {}
        seg_ids = list(seg_ids)
        for i in range(0, len(seg_ids), 900):
            part = seg_ids[i:i + 900]
            out.update({s: k for s, k in self._db.execute(
                f"SELECT seg_id, kid FROM keys WHERE version=? AND seg_id IN ({','.join('?' * len(part))})",
                [version, *part])})
        return np.array([out.get(s, -1) for s in seg_ids], dtype=np.int64)

    def seg_ids_of(self, version: str, kids) -> list:
        kids = [int(k) for k in np.asarray(kids).reshape(-1)]
        out = {}
        valid = sorted({k for k in kids if k >= 0})
        for i in range(0, len(valid), 900):
            part = valid[i:i + 900]
            out.update({k: s for k, s in self._db.execute(
                f"SELECT kid, seg_id FROM keys WHERE version=? AND kid IN ({','.join('?' * len(part))})",
                [version, *part])})
        return [out.get(k) if k >= 0 else None for k in kids]

    def _mmap(self, version: str, shard: int, pre: str, width: int):
        n = self.count(version)
        rows = max(0, min(self.shard_rows, n - shard * self.shard_rows))
        key = (version, shard, pre)
        hit = self._mm.get(key)
        if hit is not None and hit[0] == rows:
            return hit[1]
        vdir = self._db.execute("SELECT vdir FROM versions WHERE version=?", (version,)).fetchone()[0]
        p = os.path.join(self.root, "versions", vdir, f"{pre}_{shard:05d}.f16")
        mm = np.memmap(p, dtype=np.float16, mode="r", shape=(rows, width)) if rows else np.zeros((0, width), np.float16)
        self._mm[key] = (rows, mm)
        return mm

    def _rows(self, version: str, kids: np.ndarray, pre: str, width: int) -> np.ndarray:
        kids = np.asarray(kids, dtype=np.int64)
        out = np.empty((len(kids), width), dtype=np.float16)
        sh = kids // self.shard_rows
        for s in np.unique(sh):
            sel = np.nonzero(sh == s)[0]
            mm = self._mmap(version, int(s), pre, width)
            out[sel] = mm[kids[sel] - s * self.shard_rows]
        return out

    def keys(self, version: str, kids) -> torch.Tensor:
        """[n, key_dim] float32 keys of ``kids`` (exactly what search compares against)."""
        return torch.from_numpy(self._rows(version, kids, "keys", self.key_dim).astype(np.float32))

    def values(self, version: str, kids) -> torch.Tensor:
        """[n, value_tokens, value_dim] float32 memory tokens of ``kids``."""
        v = self._rows(version, kids, "vals", self.value_tokens * self.value_dim).astype(np.float32)
        return torch.from_numpy(v).view(len(v), self.value_tokens, self.value_dim)

    def _shard_tensor(self, version: str, shard: int) -> torch.Tensor:
        mm = self._mmap(version, shard, "keys", self.key_dim)
        key = (version, shard)
        hit = self._gpu.get(key)
        if hit is not None and hit.shape[0] == mm.shape[0]:
            self._gpu.move_to_end(key)
            return hit
        if hit is not None and hit.shape[0] < mm.shape[0]:
            t = torch.cat([hit, torch.from_numpy(np.array(mm[hit.shape[0]:])).to(self.device)], 0)
        else:
            t = torch.from_numpy(np.array(mm)).to(self.device)
        self._gpu[key] = t
        self._gpu.move_to_end(key)
        while sum(x.numel() * 2 for x in self._gpu.values()) > self.gpu_cache_bytes and len(self._gpu) > 1:
            self._gpu.popitem(last=False)
        return t

    # ================================================================== search
    def candidates(self, version: str, f: dict) -> np.ndarray:
        """Sorted kids of ``version`` whose segment satisfies filter dict ``f`` (SQLite prefilter)."""
        clauses, params = ["k.version = ?"], [version]
        for k, v in f.items():
            if k not in FILTER_KEYS:
                raise ValueError(f"unknown memory filter {k!r}")
            if v is None:
                continue
            if k == "t_min":
                clauses.append("s.t_end >= ?")
                params.append(float(v))
            elif k == "t_max":
                clauses.append("s.t_start <= ?")
                params.append(float(v))
            elif k == "gseq_max":
                clauses.append("s.gseq_end <= ?")
                params.append(int(v))
            elif k == "entity":
                vs = [v] if isinstance(v, str) else list(v)
                clauses.append(f"s.seg_id IN (SELECT seg_id FROM seg_entities WHERE entity IN ({','.join('?' * len(vs))}))")
                params.extend(vs)
            elif k in ("exclude_episode", "exclude_seg_ids"):
                vs = [v] if isinstance(v, str) else list(v)
                col = "s.episode_id" if k == "exclude_episode" else "s.seg_id"
                if vs:
                    clauses.append(f"{col} NOT IN ({','.join('?' * len(vs))})")
                    params.extend(vs)
            else:
                vs = list(v) if isinstance(v, (list, tuple, set)) else [v]
                clauses.append(f"s.{k} IN ({','.join('?' * len(vs))})")
                params.extend(vs)
        q = f"SELECT k.kid FROM keys k JOIN segments s ON s.seg_id = k.seg_id WHERE {' AND '.join(clauses)} ORDER BY k.kid"
        return np.fromiter((r[0] for r in self._db.execute(q, params)), dtype=np.int64)

    def search(self, version: str, q, k: int, filters=None) -> SearchResult:
        """Exact top-k by dot product within one version.

        q: [B, key_dim] (any device/dtype); filters: None, one dict for all rows, or a list of B (dict | None).
        Returns SearchResult(scores [B,k] float32 on ``q``'s device (-inf = no result), kids np.int64 [B,k] (-1),
        seg_ids [B][k] (None))."""
        t0 = time.time()
        if version not in self.versions():
            raise KeyError(f"no keys of version {version!r} (have {list(self.versions())})")
        out_dev = q.device if isinstance(q, torch.Tensor) else torch.device("cpu")
        qd = torch.as_tensor(q, dtype=torch.float32).to(self.device)
        if qd.dim() != 2 or qd.shape[1] != self.key_dim:
            raise ValueError(f"query shape {tuple(qd.shape)} != [B, {self.key_dim}]")
        B, n = qd.shape[0], self.count(version)
        best_s = torch.full((B, k), float("-inf"), device=self.device)
        best_i = torch.full((B, k), -1, dtype=torch.long, device=self.device)
        if n == 0 or k == 0:
            return self._result(version, best_s, best_i, out_dev, t0)
        if filters is None or isinstance(filters, dict):
            filters = [filters] * B
        if len(filters) != B:
            raise ValueError("filters must be one dict or a list with one entry per query row")
        groups: dict = {}
        for b, f in enumerate(filters):
            groups.setdefault(json.dumps(f, sort_keys=True, default=list) if f else "", []).append(b)
        with _fp32_matmul():
            for gkey, rows in groups.items():
                rows_t = torch.tensor(rows, device=self.device)
                qg = qd[rows_t]
                if gkey == "":
                    s, i = self._scan(version, qg, k, None, n)
                else:
                    cand = self.candidates(version, json.loads(gkey))
                    if len(cand) == 0:
                        continue
                    if len(cand) <= max(4096, n // 4):
                        s, i = self._gather_topk(version, qg, k, cand)
                    else:
                        s, i = self._scan(version, qg, k, cand, n)
                best_s[rows_t], best_i[rows_t] = s, i
        return self._result(version, best_s, best_i, out_dev, t0)

    def _result(self, version, best_s, best_i, out_dev, t0) -> SearchResult:
        kids = best_i.cpu().numpy()
        seg = self.seg_ids_of(version, kids.reshape(-1))
        B, k = kids.shape
        self.stats["searches"] += 1
        self.stats["search_s"] += time.time() - t0
        return SearchResult(best_s.to(out_dev), kids, [seg[b * k:(b + 1) * k] for b in range(B)])

    def _scan(self, version: str, q: torch.Tensor, k: int, cand: Optional[np.ndarray], n: int):
        B = q.shape[0]
        bs = torch.full((B, k), float("-inf"), device=self.device)
        bi = torch.full((B, k), -1, dtype=torch.long, device=self.device)
        for s in range(-(-n // self.shard_rows)):
            K = self._shard_tensor(version, s)
            base = s * self.shard_rows
            for c0 in range(0, K.shape[0], self.search_chunk):
                Kc = K[c0:c0 + self.search_chunk].float()
                sc = q @ Kc.T                                     # [B, c]
                g0 = base + c0
                if cand is not None:
                    lo, hi = np.searchsorted(cand, [g0, g0 + Kc.shape[0]])
                    m = torch.zeros(Kc.shape[0], dtype=torch.bool, device=self.device)
                    if hi > lo:
                        m[torch.from_numpy(cand[lo:hi] - g0).to(self.device)] = True
                    sc = sc.masked_fill(~m, float("-inf"))
                kk = min(k, sc.shape[1])
                ts, ti = sc.topk(kk, dim=1)
                ti = torch.where(torch.isfinite(ts), ti + g0, torch.full_like(ti, -1))
                bs, idx = torch.cat([bs, ts], 1).topk(k, dim=1)
                bi = torch.cat([bi, ti], 1).gather(1, idx)
        return bs, bi

    def _gather_topk(self, version: str, q: torch.Tensor, k: int, cand: np.ndarray):
        K = self.keys(version, cand).to(self.device)              # [C, key_dim]
        sc = q @ K.T
        kk = min(k, sc.shape[1])
        ts, ti = sc.topk(kk, dim=1)
        B = q.shape[0]
        bs = torch.full((B, k), float("-inf"), device=self.device)
        bi = torch.full((B, k), -1, dtype=torch.long, device=self.device)
        bs[:, :kk] = ts
        bi[:, :kk] = torch.from_numpy(cand).to(self.device)[ti]
        return bs, bi

    # ================================================================== versions / re-encoding
    def missing(self, version: str, level: Optional[int] = None, limit: Optional[int] = None) -> list:
        """Segments (oldest first, lower levels first) that have no key in ``version`` (stale or never encoded)."""
        q = "SELECT s.seg_id FROM segments s WHERE NOT EXISTS (SELECT 1 FROM keys k WHERE k.version=? AND " \
            "k.seg_id=s.seg_id)"
        params: list = [version]
        if level is not None:
            q += " AND s.level=?"
            params.append(int(level))
        q += " ORDER BY s.level, s.gseq_start, s.seg_id"
        if limit is not None:
            q += " LIMIT ?"
            params.append(int(limit))
        return [r[0] for r in self._db.execute(q, params)]

    def levels(self) -> list:
        return [r[0] for r in self._db.execute("SELECT DISTINCT level FROM segments ORDER BY level")]

    def reencode(self, version: str, encode_fn: Callable, batch: int = 256, max_items: Optional[int] = None) -> int:
        """Recompute keys/values of segments missing in ``version`` via ``encode_fn(metas) -> (keys, values)``,
        level by level (a level starts only once the lower levels are complete, since higher levels are built from
        the lower levels' values of the same version). Returns the number of segments encoded."""
        done = 0
        for lvl in self.levels():
            while max_items is None or done < max_items:
                todo = self.missing(version, level=lvl, limit=batch if max_items is None else min(batch, max_items - done))
                if not todo:
                    break
                metas = self.segments(todo)
                keys, vals = encode_fn(metas)
                self.add(version, todo, keys, vals)
                done += len(todo)
                self.stats["reencoded"] += len(todo)
            if self.missing(version, level=lvl, limit=1):
                break
        return done

    def drop_version(self, version: str) -> None:
        """Delete a stale version's keys and files (refuses the current version)."""
        if version == self.current_version:
            raise ValueError("cannot drop the current version")
        r = self._db.execute("SELECT vdir FROM versions WHERE version=?", (version,)).fetchone()
        if r is None:
            return
        self._db.execute("DELETE FROM keys WHERE version=?", (version,))
        self._db.execute("DELETE FROM versions WHERE version=?", (version,))
        self._db.commit()
        d = os.path.join(self.root, "versions", r[0])
        for f in os.listdir(d) if os.path.isdir(d) else []:
            os.unlink(os.path.join(d, f))
        if os.path.isdir(d):
            os.rmdir(d)
        for key in [k for k in self._gpu if k[0] == version]:
            self._gpu.pop(key)
        for key in [k for k in self._mm if k[0] == version]:
            self._mm.pop(key)

    def pointer(self) -> dict:
        return {"root": self.root, "key_dim": self.key_dim, "value_tokens": self.value_tokens,
                "value_dim": self.value_dim, "shard_rows": self.shard_rows, "current_version": self.current_version,
                "versions": self.versions(), "segments": self.count()}

    def storage_bytes(self) -> int:
        from ._fs import dir_bytes
        return dir_bytes(self.root)

    def close(self) -> None:
        if self._db is not None:
            self._db.commit()
            self._db.close()
            self._db = None
        self._gpu.clear()
        self._mm.clear()


def _np16(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float16)
