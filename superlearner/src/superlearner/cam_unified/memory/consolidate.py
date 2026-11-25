"""Hierarchical memory consolidation (spec 10.4, architecture §8).

Levels, each keeping source links to the one below:

    0  raw events            EventArchive (exact records; artifacts for long originals)
    1  segments              <= ``segment_len`` consecutive events of one (env, actor, episode) stream ->
                             ``mem_segment_tokens`` learned latent memory tokens + key  (Retriever.segment_encoder)
    2  longer representations ``fanout`` consecutive level-1 segments of one (env, actor) stream -> memory tokens + key
                             (Retriever.long_encoder over the children's memory tokens)
    -  durable artifacts      content-addressed originals referenced by the segment's observations (``artifact_refs``)
    -  parametric knowledge   weight updates that consumed a segment (``link_weights`` -> MemoryIndex.weight_links and
                             EventArchive.mark_used: the events' training provenance)

Keys/values are the *versioned values store*: they live in the ``MemoryIndex`` partition of the retriever's
``key_version``. ``reencode_step`` migrates every segment to a new version by recomputing from the raw events (level 1)
and from the new version's level-1 values (level 2), then promotes it; obsolete, unpinned versions are dropped.

Ingestion during a migration (``step()`` after ``Retriever.sync_target()`` and before the new version is promoted):
new segments are encoded in the new (target) version only, because the previous version's encoders no longer exist;
they become searchable when the new version is promoted (the served version keeps answering queries meanwhile). A
level-2 group whose level-1 children were encoded under the old version gets those children re-encoded from their raw
events on demand (exactly what ``reencode_step`` would compute for them), so groups are never lost or left open.

``step()`` is atomic: it works on a copy of the consolidation state and writes the segments, their keys/values and the
new state (cursor = last archive gseq consumed, open segments / groups) in ONE ``MemoryIndex.transaction()``; on any
exception nothing is written and the in-memory state is unchanged, so a retry (or a restart) neither loses nor
duplicates segments (segment ids are deterministic: ``s1_<first gseq>``, ``s2_<first child gseq>``).
"""
from __future__ import annotations

import json
import time
from typing import Optional, Sequence

import torch

from .archive import event_entities

_STATE_KEY = "consolidator"


class Consolidator:
    """Args:
        archive: ``EventArchive`` (level 0).
        index: ``MemoryIndex`` (levels 1-2 keys/values + source links).
        retriever: ``Retriever`` (encoders; ``key_version``).
        segment_len: events per level-1 segment (default ``cfg.mem_segment``).
        fanout: level-1 segments per level-2 representation (default ``retriever.fanout``).
        batch_segments: segments encoded per forward pass.
        device: encoding device (default: the retriever's parameters' device).
    """

    def __init__(self, archive, index, retriever, segment_len: Optional[int] = None, fanout: Optional[int] = None,
                 batch_segments: int = 64, device=None):
        self.archive, self.index, self.retriever = archive, index, retriever
        self.segment_len = int(segment_len or retriever.segment_len)
        self.fanout = int(fanout or retriever.fanout)
        if self.fanout * retriever.n_mem > retriever.long_encoder.max_len:
            raise ValueError("fanout * mem_segment_tokens exceeds the long encoder's max_len")
        if self.segment_len > retriever.segment_encoder.max_len:
            raise ValueError("segment_len exceeds the segment encoder's max_len")
        self.batch_segments = int(batch_segments)
        self.device = device if device is not None else next(retriever.parameters()).device
        st = index.get_state(_STATE_KEY)
        self.state = st if st is not None else {"cursor": -1, "open1": {}, "open2": {}, "n1": 0, "n2": 0}
        self.stats = {"events": 0, "segments1": 0, "segments2": 0, "encode_s": 0.0, "children_reencoded": 0}

    # ------------------------------------------------------------------ ingestion
    def step(self, max_events: int = 100_000, close_all: bool = False) -> dict:
        """Consume archived events after the cursor, emit full (or episode-final) segments, encode and index them.
        ``close_all`` also closes every open segment / group (e.g. at the end of a lifetime phase). Atomic (see the
        module docstring): on an exception neither the index nor ``self.state`` changes."""
        version = self.retriever.key_version
        if not self.archive.readonly:
            self.archive.flush()             # only durable events become segments (their gseqs are final)
        st = json.loads(json.dumps(self.state))          # work on a copy; committed together with the segments
        emitted1: list = []
        n_ev = 0
        for rows in self.archive.scan_meta(batch_size=4096, gseq_min=st["cursor"] + 1,
                                           columns=["gseq", "event_id", "env_id", "actor_id", "episode_id", "done"]):
            for r in rows:
                stream = f"{r['env_id']}|{r['actor_id']}|{r['episode_id']}"
                cur = st["open1"].setdefault(stream, [])
                cur.append([r["event_id"], r["gseq"]])
                if len(cur) >= self.segment_len or r["done"]:
                    emitted1.append((stream, st["open1"].pop(stream)))
                st["cursor"] = r["gseq"]
                n_ev += 1
                if n_ev >= max_events:
                    break
            if n_ev >= max_events:
                break
        if close_all:
            emitted1.extend(st["open1"].items())
            st["open1"] = {}
        counts = {"segments1": 0, "segments2": 0, "children_reencoded": 0}
        t_enc = self.stats["encode_s"]
        try:
            with self.index.transaction():
                metas1 = self._encode_level1(emitted1, version, counts) if emitted1 else []
                metas2 = []
                for m in metas1:
                    s2 = f"{m['env_id']}|{m['actor_id']}"
                    grp = st["open2"].setdefault(s2, [])
                    grp.append(m["seg_id"])
                    if len(grp) >= self.fanout:
                        metas2.append(st["open2"].pop(s2))
                if close_all:
                    metas2.extend(g for g in st["open2"].values() if g)
                    st["open2"] = {}
                made2 = self._encode_level2(metas2, version, counts) if metas2 else []
                self.index.set_state(_STATE_KEY, st)
        except BaseException:
            self.stats["encode_s"] = t_enc
            raise
        self.state = st
        self.stats["events"] += n_ev
        for k, v in counts.items():
            self.stats[k] += v
        cur = self.index.current_version
        return {"events": n_ev, "segments1": len(metas1), "segments2": len(made2), "version": version,
                "open_streams": len(st["open1"]), "children_reencoded": counts["children_reencoded"],
                "served_version": cur, "migration_pending": cur is not None and cur != version}

    def _encode_level1(self, emitted: list, version: str, counts: dict) -> list:
        """emitted: [(stream, [[event_id, gseq], ...])] -> registered + keyed level-1 segment metas."""
        out = []
        for i in range(0, len(emitted), self.batch_segments):
            part = emitted[i:i + self.batch_segments]
            ids = [e for _, evs in part for e, _ in evs]
            recs = self.archive.get(ids)
            by_id = {r.event_id: r for r in recs}
            metas, groups = [], []
            for _stream, evs in part:
                rs = [by_id[e] for e, _ in evs]
                refs = sorted({o.artifact_ref for r in rs for o in (r.observations or []) if o.artifact_ref} |
                              {o.artifact_ref for r in rs if r.result is not None
                               for o in (r.result.observations or []) if o.artifact_ref})
                ents = []
                for r in rs:
                    for e in event_entities(r):
                        if e not in ents:
                            ents.append(e)
                metas.append({"seg_id": f"s1_{evs[0][1]:012d}", "level": 1, "env_id": rs[0].env_id,
                              "actor_id": rs[0].actor_id, "episode_id": rs[0].episode_id,
                              "t_start": min(r.t_sim for r in rs), "t_end": max(r.t_sim for r in rs),
                              "gseq_start": evs[0][1], "gseq_end": evs[-1][1], "n_events": len(rs),
                              "event_ids": [e for e, _ in evs], "child_ids": [], "artifact_refs": refs,
                              "entities": ents})
                groups.append(rs)
            keys, vals = self._encode1(groups, metas, target=True)
            self.index.add_segments(metas)
            self.index.add(version, [m["seg_id"] for m in metas], keys, vals)
            out.extend(metas)
            counts["segments1"] += len(metas)
        return out

    def _encode_level2(self, groups: list, version: str, counts: dict) -> list:
        metas = []
        for g in groups:
            ch = self.index.segments(g)
            ents = []
            for c in ch:
                for e in c["entities"]:
                    if e not in ents:
                        ents.append(e)
            metas.append({"seg_id": f"s2_{ch[0]['gseq_start']:012d}", "level": 2, "env_id": ch[0]["env_id"],
                          "actor_id": ch[0]["actor_id"], "episode_id": ch[0]["episode_id"] if len(
                              {c["episode_id"] for c in ch}) == 1 else None,
                          "t_start": min(c["t_start"] for c in ch), "t_end": max(c["t_end"] for c in ch),
                          "gseq_start": ch[0]["gseq_start"], "gseq_end": ch[-1]["gseq_end"],
                          "n_events": sum(c["n_events"] for c in ch), "event_ids": [], "child_ids": list(g),
                          "artifact_refs": sorted({a for c in ch for a in c["artifact_refs"]}), "entities": ents})
        for i in range(0, len(metas), self.batch_segments):
            part = metas[i:i + self.batch_segments]
            counts["children_reencoded"] += self._ensure_child_keys(version, [c for m in part for c in m["child_ids"]])
            keys, vals = self._encode2(part, version, target=True)
            self.index.add_segments(part)
            self.index.add(version, [m["seg_id"] for m in part], keys, vals)
            counts["segments2"] += len(part)
        return metas

    def _ensure_child_keys(self, version: str, seg_ids: list) -> int:
        """Level-1 children without a key in ``version`` (encoded before a ``sync_target``) are re-encoded from their
        raw events with the current target encoders, i.e. exactly what the migration computes for them."""
        if version != self.retriever.key_version:
            raise RuntimeError(f"cannot encode for {version}: the retriever's target version is "
                               f"{self.retriever.key_version}")
        kids = self.index.kids_of(version, seg_ids)
        miss = [s for s, k in zip(seg_ids, kids) if k < 0]
        if not miss:
            return 0
        done = 0
        for i in range(0, len(miss), self.batch_segments):
            part = miss[i:i + self.batch_segments]
            metas = self.index.segments(part)
            if any(m["level"] != 1 for m in metas):
                raise RuntimeError("level-2 children must be level-1 segments")
            groups = [self.archive.get(m["event_ids"]) for m in metas]
            keys, vals = self._encode1(groups, metas, target=True)
            self.index.add(version, part, keys, vals)
            done += len(part)
        return done

    # ------------------------------------------------------------------ encoders (shared by ingestion and migration)
    @torch.no_grad()
    def _encode1(self, groups: list, metas: list, target: bool = True):
        t0 = time.time()
        L = max(len(g) for g in groups)
        ev = self.retriever.encode_records([r for g in groups for r in g], target=target).to(self.device)
        n, D = len(groups), ev.shape[-1]
        x = torch.zeros(n, L, D, device=self.device)
        mask = torch.zeros(n, L, dtype=torch.bool, device=self.device)
        dt = torch.zeros(n, L, device=self.device)
        p = 0
        for i, g in enumerate(groups):
            x[i, :len(g)] = ev[p:p + len(g)]
            mask[i, :len(g)] = True
            dt[i, :len(g)] = torch.tensor([r.t_sim - metas[i]["t_end"] for r in g], device=self.device)
            p += len(g)
        mem, key = self.retriever.encode_segments(x, mask, dt, target=target)
        self.stats["encode_s"] += time.time() - t0
        return key, mem

    @torch.no_grad()
    def _encode2(self, metas: list, version: str, target: bool = True):
        T = self.retriever.n_mem
        Fm = max(len(m["child_ids"]) for m in metas)
        n, D = len(metas), self.retriever.d
        x = torch.zeros(n, Fm * T, D, device=self.device)
        mask = torch.zeros(n, Fm * T, dtype=torch.bool, device=self.device)
        dt = torch.zeros(n, Fm * T, device=self.device)
        for i, m in enumerate(metas):
            kids = self.index.kids_of(version, m["child_ids"])
            if (kids < 0).any():
                raise RuntimeError(f"level-2 segment {m['seg_id']} has children without keys in {version}")
            v = self.index.values(version, kids).to(self.device)          # [F, T, D]
            ch = self.index.segments(m["child_ids"])
            x[i, :len(kids) * T] = v.reshape(-1, D)
            mask[i, :len(kids) * T] = True
            dt[i, :len(kids) * T] = torch.tensor([c["t_end"] - m["t_end"] for c in ch],
                                                 device=self.device).repeat_interleave(T)
        mem, key = self.retriever.encode_long(x, mask, dt, target=target)
        return key, mem

    # ------------------------------------------------------------------ versions
    def reencode_step(self, max_items: Optional[int] = 512, drop_old: bool = True) -> dict:
        """Migrate segments to the retriever's current ``key_version`` (recomputed from raw events / new level-1
        values); promotes the version once it covers every segment and drops versions the retriever cannot use."""
        version = self.retriever.key_version

        def encode_fn(metas):
            if metas[0]["level"] == 1:
                groups = [self.archive.get(m["event_ids"]) for m in metas]
                return self._encode1(groups, metas, target=True)
            return self._encode2(metas, version, target=True)
        n = self.index.reencode(version, encode_fn, batch=self.batch_segments, max_items=max_items)
        cov = self.index.coverage(version)
        promoted = False
        if cov >= 1.0 and self.index.current_version != version:
            self.index.set_current(version)
            promoted = True
        dropped, kept = [], []
        if drop_old and self.index.current_version == version:
            pinned = self.index.pinned_versions()
            for v in self.index.stale_versions():          # obsolete vectors are deleted, never searched again
                if v in pinned:                            # ... unless a full-lifetime checkpoint still needs them
                    kept.append(v)
                    continue
                self.index.drop_version(v)
                dropped.append(v)
        return {"version": version, "reencoded": n, "coverage": cov, "promoted": promoted, "dropped": dropped,
                "kept_pinned": kept}

    def ensure_current(self) -> None:
        """Make the index's current version the retriever's version when it already covers everything (first use)."""
        v = self.retriever.key_version
        if self.index.current_version is None and self.index.coverage(v) >= 1.0 and self.index.count(v) > 0:
            self.index.set_current(v)

    # ------------------------------------------------------------------ source links
    def source_events(self, seg_id: str) -> list:
        return self.index.source_events(seg_id)

    def source_records(self, seg_id: str) -> list:
        return self.archive.get(self.index.source_events(seg_id))

    def artifacts_of(self, seg_id: str) -> list:
        return list(self.index.segment(seg_id)["artifact_refs"])

    def link_weights(self, seg_ids: Sequence[str], update_id: str, model_version: int) -> None:
        """Parametric consolidation bookkeeping: a weight update consumed these segments."""
        self.index.link_weights(seg_ids, update_id, model_version)
        ev = [e for s in seg_ids for e in self.index.source_events(s)]
        if ev and not self.archive.readonly:
            self.archive.mark_used(ev, update_id)

    def state_dict(self) -> dict:
        """Consolidation state (cursor, open segments / groups) for a full-lifetime checkpoint."""
        return json.loads(json.dumps(self.state))

    def load_state_dict(self, st: dict) -> None:
        self.state = json.loads(json.dumps(st))
        self.index.set_state(_STATE_KEY, self.state)

    def pointer(self) -> dict:
        """Everything needed to rebuild this consolidator over the same archive / index / retriever."""
        return {"state": self.state_dict(), "segment_len": self.segment_len, "fanout": self.fanout,
                "batch_segments": self.batch_segments}
