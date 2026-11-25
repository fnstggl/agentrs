"""Persistent stratified replay over the durable archive (spec 11.2, architecture §8). Not a FIFO buffer.

The source of every sample is the ``EventArchive``. The sampler keeps, in its own SQLite file, append-only
*membership tables* per stratum (dense positions -> archive gseq) that it updates incrementally by change-data capture
(new events after its cursor, and later tag updates via ``EventArchive.tag_changes``). Drawing uniformly from a
stratum is then one indexed lookup at a random position, exact over the whole lifetime and independent of its length.

Strata (``STRATA``):

    recent            uniform over the last ``recent_window`` archived events
    uniform           uniform over every archived event ever
    domain            a domain (``family`` tag, else ``env_id``) uniformly, then uniform within it
    failures          execution failures or unsuccessful outcomes (``failed`` or ``success == 0``)
    innovations       ``innovation`` tag or novelty z-score > ``novelty_z`` (running statistics, prequential)
    rare              surprise z-score > ``rare_z`` (running statistics, prequential)
    skills            successful skills: ``success == 1`` or a ``skill`` tag
    underrepresented  a domain with probability proportional to (n_domain)^-alpha, then uniform within it

Each draw returns a contiguous, step-ordered slice of the anchor event's episode (``ReplaySlice``) for sequence
training. Empty strata pass their mass proportionally to the non-empty ones (reported in ``last_mix``). A bounded LRU
of decoded records (``cache_events``) and the optional ``ActiveCache`` of slices keep the working set bounded.
"""
from __future__ import annotations

import math
import os
import random
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

STRATA = ("recent", "uniform", "domain", "failures", "innovations", "rare", "skills", "underrepresented")
DEFAULT_MIX = {"recent": 0.20, "uniform": 0.20, "domain": 0.15, "failures": 0.15, "innovations": 0.075, "rare": 0.10,
               "skills": 0.075, "underrepresented": 0.05}


@dataclass
class ReplaySlice:
    stratum: str
    anchor_gseq: int
    episode_id: str
    steps: list                       # step numbers, ascending
    event_ids: list
    records: Optional[list] = None    # EventRecords (when requested)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.event_ids)


class _Welford:
    def __init__(self, n=0, mean=0.0, m2=0.0):
        self.n, self.mean, self.m2 = n, mean, m2

    def z(self, x: float) -> float:
        if self.n < 8:
            return 0.0
        sd = math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0
        return 0.0 if sd <= 1e-12 else (x - self.mean) / sd

    def add(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)


class ReplaySampler:
    """Args:
        archive: the ``EventArchive``.
        path: sampler database (default ``<archive.root>/replay.sqlite``).
        mix: stratum -> probability (normalised; missing strata get 0).
        recent_window: events of the ``recent`` stratum.
        rare_z / novelty_z: z-score thresholds of ``rare`` / ``innovations``.
        alpha_under: exponent of the ``underrepresented`` domain weighting.
        cache_events: LRU capacity of decoded records.
    """

    def __init__(self, archive, path: Optional[str] = None, mix: Optional[dict] = None, recent_window: int = 4096,
                 rare_z: float = 2.0, novelty_z: float = 2.0, alpha_under: float = 1.0, seed: int = 0,
                 cache_events: int = 20_000):
        self.archive = archive
        self.path = os.path.abspath(path or os.path.join(archive.root, "replay.sqlite"))
        mix = dict(DEFAULT_MIX if mix is None else mix)
        for k in mix:
            if k not in STRATA:
                raise ValueError(f"unknown stratum {k!r}")
        tot = sum(mix.values())
        self.mix = {k: mix.get(k, 0.0) / tot for k in STRATA}
        self.recent_window = int(recent_window)
        self.rare_z, self.novelty_z, self.alpha_under = float(rare_z), float(novelty_z), float(alpha_under)
        self.rng = random.Random(seed)
        self.cache_events = int(cache_events)
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self.draws = {k: 0 for k in STRATA}
        self.last_mix: dict = {}
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS members (stratum TEXT NOT NULL, pos INTEGER NOT NULL, gseq INTEGER NOT NULL,
                                            PRIMARY KEY (stratum, pos));
        CREATE UNIQUE INDEX IF NOT EXISTS members_g ON members(stratum, gseq);
        CREATE TABLE IF NOT EXISTS counts (stratum TEXT PRIMARY KEY, n INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value REAL);
        """)
        self._db.commit()
        self._n = {s: n for s, n in self._db.execute("SELECT stratum, n FROM counts")}
        kv = {k: v for k, v in self._db.execute("SELECT key, value FROM kv")}
        self.cursor = int(kv.get("cursor", -1))
        self.tag_cursor = int(kv.get("tag_cursor", 0))
        self._sur = _Welford(int(kv.get("sur_n", 0)), kv.get("sur_mean", 0.0), kv.get("sur_m2", 0.0))
        self._nov = _Welford(int(kv.get("nov_n", 0)), kv.get("nov_mean", 0.0), kv.get("nov_m2", 0.0))

    # ------------------------------------------------------------------ membership (change-data capture)
    def _strata_of(self, r: dict, update_stats: bool) -> list:
        out = []
        if r["failed"] or r["success"] == 0:
            out.append("failures")
        if r["success"] == 1 or r["skill"]:
            out.append("skills")
        nov = r["novelty"]
        if r["innovation"] or (nov is not None and self._nov.z(nov) > self.novelty_z):
            out.append("innovations")
        sur = r["surprise"]
        if sur is not None and self._sur.z(sur) > self.rare_z:
            out.append("rare")
        if update_stats:
            if nov is not None:
                self._nov.add(nov)
            if sur is not None:
                self._sur.add(sur)
        out.append("dom:" + (r["family"] or r["env_id"] or ""))
        return out

    def _add(self, items: list) -> None:
        rows = []
        for stratum, g in items:
            rows.append((stratum, g))
        for stratum, g in rows:
            n = self._n.get(stratum, 0)
            cur = self._db.execute("INSERT OR IGNORE INTO members(stratum, pos, gseq) VALUES (?,?,?)", (stratum, n, g))
            if cur.rowcount:
                self._n[stratum] = n + 1

    def sync(self, max_events: Optional[int] = None) -> int:
        """Bring membership tables up to date with the archive (new events + tag changes). Returns events consumed."""
        cols = ["gseq", "env_id", "failed", "success", "novelty", "surprise", "innovation", "skill", "family"]
        n = 0
        for rows in self.archive.scan_meta(batch_size=8192, columns=cols, gseq_min=self.cursor + 1):
            items = []
            for r in rows:
                for s in self._strata_of(r, update_stats=True):
                    items.append((s, r["gseq"]))
                self.cursor = r["gseq"]
                n += 1
                if max_events is not None and n >= max_events:
                    break
            self._add(items)
            self._persist()
            if max_events is not None and n >= max_events:
                break
        while True:
            ch = self.archive.tag_changes(self.tag_cursor, limit=8192)
            if not ch:
                break
            gs = sorted({g for _, g in ch if g <= self.cursor})
            items = []
            for meta in self._meta_by_gseq(gs, cols):
                for s in self._strata_of(meta, update_stats=False):
                    if not s.startswith("dom:"):
                        items.append((s, meta["gseq"]))
            self._add(items)
            self.tag_cursor = ch[-1][0]
            self._persist()
        return n

    def _persist(self) -> None:
        self._db.executemany("INSERT OR REPLACE INTO counts(stratum, n) VALUES (?,?)", list(self._n.items()))
        kv = {"cursor": self.cursor, "tag_cursor": self.tag_cursor, "sur_n": self._sur.n, "sur_mean": self._sur.mean,
              "sur_m2": self._sur.m2, "nov_n": self._nov.n, "nov_mean": self._nov.mean, "nov_m2": self._nov.m2}
        self._db.executemany("INSERT OR REPLACE INTO kv(key, value) VALUES (?,?)", list(kv.items()))
        self._db.commit()

    def _meta_by_gseq(self, gseqs: list, cols: list) -> list:
        out = []
        for i in range(0, len(gseqs), 900):
            part = gseqs[i:i + 900]
            q = f"SELECT {','.join(cols)} FROM events WHERE gseq IN ({','.join('?' * len(part))})"
            out.extend(dict(zip(cols, r)) for r in self.archive._db.execute(q, part))
        return out

    # ------------------------------------------------------------------ sizes
    def stratum_sizes(self) -> dict:
        N = self.cursor + 1
        doms = self.domains()
        return {"recent": min(N, self.recent_window), "uniform": N, "domain": sum(doms.values()),
                "failures": self._n.get("failures", 0), "innovations": self._n.get("innovations", 0),
                "rare": self._n.get("rare", 0), "skills": self._n.get("skills", 0),
                "underrepresented": sum(doms.values())}

    def domains(self) -> dict:
        return {s[4:]: n for s, n in self._n.items() if s.startswith("dom:") and n > 0}

    # ------------------------------------------------------------------ sampling
    def _pick(self, stratum: str, n: int) -> list:
        N = self.cursor + 1
        if stratum == "uniform":
            return [self.rng.randrange(N) for _ in range(n)]
        if stratum == "recent":
            lo = max(0, N - self.recent_window)
            return [self.rng.randrange(lo, N) for _ in range(n)]
        if stratum in ("domain", "underrepresented"):
            doms = self.domains()
            names = sorted(doms)
            if stratum == "domain":
                w = [1.0] * len(names)
            else:
                w = [doms[d] ** (-self.alpha_under) for d in names]
            out = []
            for _ in range(n):
                d = self.rng.choices(names, weights=w)[0]
                out.append(self._at("dom:" + d, self.rng.randrange(doms[d])))
            return out
        m = self._n.get(stratum, 0)
        return [self._at(stratum, self.rng.randrange(m)) for _ in range(n)]

    def _at(self, stratum: str, pos: int) -> int:
        return int(self._db.execute("SELECT gseq FROM members WHERE stratum=? AND pos=?", (stratum, pos)).fetchone()[0])

    def sample_anchors(self, n: int) -> list:
        """[(stratum, gseq)] for n draws following the mix (empty strata redistributed)."""
        sizes = self.stratum_sizes()
        avail = {s: p for s, p in self.mix.items() if p > 0 and sizes[s] > 0}
        if not avail:
            return []
        tot = sum(avail.values())
        mix = {s: p / tot for s, p in avail.items()}
        self.last_mix = mix
        names = sorted(mix)
        chosen = self.rng.choices(names, weights=[mix[s] for s in names], k=n)
        out = []
        for s in names:
            k = chosen.count(s)
            if k:
                out.extend((s, g) for g in self._pick(s, k))
                self.draws[s] += k
        self.rng.shuffle(out)
        return out

    def sample(self, n: int, length: int, with_records: bool = True) -> list:
        """n contiguous episode slices of up to ``length`` steps, each containing its stratum's anchor event."""
        anchors = self.sample_anchors(n)
        if not anchors:
            return []
        metas = {m["gseq"]: m for m in self._meta_by_gseq(sorted({g for _, g in anchors}),
                                                          ["gseq", "episode_id", "step", "event_id"])}
        out = []
        for stratum, g in anchors:
            m = metas[g]
            ep = self.archive.episode_steps(m["episode_id"])
            i = next(j for j, (_s, e, gg) in enumerate(ep) if gg == g)
            lo = max(0, i - length + 1)
            hi = min(i, max(0, len(ep) - length))
            start = self.rng.randint(lo, hi) if hi >= lo else lo
            part = ep[start:start + length]
            sl = ReplaySlice(stratum, g, m["episode_id"], [s for s, _, _ in part], [e for _, e, _ in part])
            out.append(sl)
        if with_records:
            self._attach_records(out)
        return out

    def _attach_records(self, slices: list) -> None:
        need = [e for s in slices for e in s.event_ids if e not in self._cache]
        if need:
            for r in self.archive.get(list(dict.fromkeys(need))):
                self._cache[r.event_id] = r
                self._cache.move_to_end(r.event_id)
        for s in slices:
            recs = []
            for e in s.event_ids:
                r = self._cache[e]
                self._cache.move_to_end(e)
                recs.append(r)
            s.records = recs
        while len(self._cache) > self.cache_events:
            self._cache.popitem(last=False)

    # ------------------------------------------------------------------ checkpoint state
    def state_dict(self) -> dict:
        """Sampler state for a full-lifetime checkpoint: cursors, per-stratum membership counts, running statistics,
        RNG and draw counters. Membership tables are append-only with dense positions, so the counts identify the
        exact membership at checkpoint time."""
        st = self.rng.getstate()
        return {"path": self.path, "cursor": self.cursor, "tag_cursor": self.tag_cursor, "counts": dict(self._n),
                "surprise_stats": [self._sur.n, self._sur.mean, self._sur.m2],
                "novelty_stats": [self._nov.n, self._nov.mean, self._nov.m2], "draws": dict(self.draws),
                "rng": [st[0], list(st[1]), st[2]], "mix": dict(self.mix), "recent_window": self.recent_window,
                "rare_z": self.rare_z, "novelty_z": self.novelty_z, "alpha_under": self.alpha_under}

    def load_state_dict(self, st: dict) -> None:
        """Restore a checkpoint's sampler state exactly, also when the persistent tables have since moved on (the
        archive is append-only and may hold later experience): membership beyond the checkpoint's counts is
        truncated and re-derived from the archive by the next ``sync``."""
        counts = {k: int(v) for k, v in st["counts"].items()}
        for (stratum,) in self._db.execute("SELECT DISTINCT stratum FROM members").fetchall():
            self._db.execute("DELETE FROM members WHERE stratum=? AND pos>=?", (stratum, counts.get(stratum, 0)))
        self._db.execute("DELETE FROM counts")
        self._n = counts
        self.cursor, self.tag_cursor = int(st["cursor"]), int(st["tag_cursor"])
        self._sur = _Welford(*st["surprise_stats"])
        self._nov = _Welford(*st["novelty_stats"])
        self.draws = dict(st["draws"])
        r = st["rng"]
        self.rng.setstate((r[0], tuple(r[1]), r[2]))
        self.mix = dict(st["mix"])
        self.recent_window = int(st["recent_window"])
        self.rare_z, self.novelty_z, self.alpha_under = float(st["rare_z"]), float(st["novelty_z"]), float(
            st["alpha_under"])
        self._cache.clear()
        self._persist()

    def close(self) -> None:
        if self._db is not None:
            self._persist()
            self._db.close()
            self._db = None


class ActiveCache:
    """Bounded active working set of replay slices; ``refresh`` replaces the oldest with fresh stratified draws."""

    def __init__(self, sampler: ReplaySampler, capacity: int = 256, length: int = 32, seed: int = 0):
        self.sampler, self.capacity, self.length = sampler, int(capacity), int(length)
        self.slices: list = []
        self.rng = random.Random(seed)

    def refresh(self, n: Optional[int] = None) -> int:
        n = self.capacity - len(self.slices) if n is None else n
        new = self.sampler.sample(max(0, n), self.length)
        self.slices.extend(new)
        if len(self.slices) > self.capacity:
            del self.slices[: len(self.slices) - self.capacity]
        return len(new)

    def batch(self, n: int) -> list:
        if not self.slices:
            self.refresh()
        return [self.slices[self.rng.randrange(len(self.slices))] for _ in range(n)]
