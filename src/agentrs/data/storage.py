"""Parquet-backed experience storage, migration and a content-addressed blob store.

Layout::

    <root>/
      real/<shard>.parquet        real transitions (agent, teacher, human, dataset, inferred)
      imagined/<shard>.parquet    world-model rollouts, physically separated
      private/<shard>.parquet     human demonstrations marked privacy="private"
      blobs/<sha256[:2]>/<sha256> large binary objects (screenshots, recordings)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from agentrs.data.schemas import SCHEMA_VERSION, EpisodeRecord, Source, Transition

# Flat columns for fast filtering; the full record is kept as JSON for lossless reload.
_COLUMNS = [
    ("episode_id", pa.string()),
    ("step", pa.int32()),
    ("task_id", pa.string()),
    ("env_id", pa.string()),
    ("family", pa.string()),
    ("split", pa.string()),
    ("source", pa.string()),
    ("action", pa.string()),
    ("action_type", pa.string()),
    ("action_valid", pa.bool_()),
    ("reward", pa.float32()),
    ("terminated", pa.bool_()),
    ("truncated", pa.bool_()),
    ("episode_success", pa.bool_()),
    ("timestamp", pa.float64()),
    ("schema_version", pa.int32()),
    ("record", pa.string()),
]
SCHEMA = pa.schema(_COLUMNS)


def _partition(t: Transition) -> str:
    if t.provenance.source == Source.IMAGINED:
        return "imagined"
    if t.provenance.privacy == "private":
        return "private"
    return "real"


def transitions_to_table(ts: list[Transition]) -> pa.Table:
    cols: dict[str, list] = {name: [] for name, _ in _COLUMNS}
    for t in ts:
        cols["episode_id"].append(t.episode_id)
        cols["step"].append(t.step)
        cols["task_id"].append(t.task_id)
        cols["env_id"].append(t.env_id)
        cols["family"].append(t.family)
        cols["split"].append(t.split)
        cols["source"].append(t.provenance.source.value)
        cols["action"].append(t.action)
        cols["action_type"].append(t.action_type)
        cols["action_valid"].append(t.action_valid)
        cols["reward"].append(t.reward)
        cols["terminated"].append(t.terminated)
        cols["truncated"].append(t.truncated)
        cols["episode_success"].append(t.episode_success)
        cols["timestamp"].append(t.timestamp)
        cols["schema_version"].append(t.schema_version)
        cols["record"].append(t.model_dump_json())
    return pa.table(cols, schema=SCHEMA)


def migrate_record(d: dict) -> dict:
    """Upgrade older serialized transitions to the current schema."""
    v = d.get("schema_version", 1)
    if v < 2:
        # v1 had no 'selection'/'behavior_logprob'; behaviour == policy by construction
        d.setdefault("selection", "policy")
        d.setdefault("behavior_logprob", d.get("policy_logprob"))
        d["schema_version"] = 2
    return d


class ExperienceStore:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        for sub in ("real", "imagined", "private", "blobs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ write
    def write_transitions(self, ts: list[Transition], tag: str = "") -> list[Path]:
        by_part: dict[str, list[Transition]] = {}
        for t in ts:
            by_part.setdefault(_partition(t), []).append(t)
        paths = []
        for part, items in by_part.items():
            name = f"{int(time.time() * 1000)}_{tag + '_' if tag else ''}{uuid.uuid4().hex[:8]}.parquet"
            path = self.root / part / name
            tmp = path.with_suffix(".tmp")
            pq.write_table(transitions_to_table(items), tmp)
            os.replace(tmp, path)  # atomic: readers never see partial shards
            paths.append(path)
        return paths

    def write_episodes(self, eps: Iterable[EpisodeRecord], tag: str = "") -> list[Path]:
        ts = [t for ep in eps for t in ep.transitions]
        return self.write_transitions(ts, tag=tag) if ts else []

    # ------------------------------------------------------------------ read
    def shards(self, partition: str = "real") -> list[Path]:
        return sorted((self.root / partition).glob("*.parquet"))

    def iter_transitions(self, partition: str = "real", shards: Optional[list[Path]] = None,
                         filters: Optional[dict] = None) -> Iterator[Transition]:
        for path in shards if shards is not None else self.shards(partition):
            table = pq.read_table(path)
            if filters:
                import pyarrow.compute as pc

                mask = None
                for k, v in filters.items():
                    m = pc.is_in(table[k], value_set=pa.array(v)) if isinstance(v, (list, tuple, set)) \
                        else pc.equal(table[k], v)
                    mask = m if mask is None else pc.and_(mask, m)
                table = table.filter(mask)
            for rec in table.column("record").to_pylist():
                yield Transition.model_validate(migrate_record(json.loads(rec)))

    def load_transitions(self, partition: str = "real", **kw) -> list[Transition]:
        return list(self.iter_transitions(partition, **kw))

    def load_episodes(self, partition: str = "real", **kw) -> list[EpisodeRecord]:
        eps: dict[str, list[Transition]] = {}
        for t in self.iter_transitions(partition, **kw):
            eps.setdefault(t.episode_id, []).append(t)
        out = []
        for eid, ts in eps.items():
            ts.sort(key=lambda t: t.step)
            last = ts[-1]
            out.append(EpisodeRecord(
                episode_id=eid, task_id=last.task_id, env_id=last.env_id, family=last.family,
                split=last.split, goal=last.goal, transitions=ts, success=last.episode_success,
                termination_reason=last.termination_reason, num_steps=len(ts),
                num_invalid=sum(not t.action_valid for t in ts), provenance=last.provenance,
                checkpoints=last.checkpoints))
        return out

    def summary(self) -> dict:
        out = {}
        for part in ("real", "imagined", "private"):
            n = 0
            for p in self.shards(part):
                n += pq.ParquetFile(p).metadata.num_rows
            out[part] = {"shards": len(self.shards(part)), "transitions": n}
        out["schema_version"] = SCHEMA_VERSION
        return out

    # ------------------------------------------------------------------ blobs
    def put_blob(self, data: bytes) -> str:
        h = hashlib.sha256(data).hexdigest()
        p = self.root / "blobs" / h[:2] / h
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return h

    def get_blob(self, h: str) -> bytes:
        return (self.root / "blobs" / h[:2] / h).read_bytes()
