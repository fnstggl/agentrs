"""Released training outcome labels: the trainer-only channel (meta-reward spec 2.4, 4.1, 11.2; manifest OB-03, OB-05,
SE-03, CR-01).

`LabelStore` is an append-only JSONL file (one record per line, fsynced) written only by the operator / harness side
(`ExperienceOperator.release_labels`, OpenWorld evaluators) and read only by the learner. A label is released only
after the outcome occurred; PENDING records are never written (pending is the absence of a label). Each label names
the episode and goal it resolves, the evaluator that produced it, the verified objective quantity, the normalised
verified utility, resolution time and release time. Nothing private (hidden mechanism parameters, oracle values,
verifier progress/streak, family names, seeds) is ever written: `LabelStore.append` rejects records carrying such
keys.

The learner joins labels to archived decisions by (env_id, episode) and uses them for: success / outcome predictor
targets (every prefix of the resolved episode), RUDDER return targets, COCOA/R-HCA rewarding outcomes, the outer
meta-objective and lifetime value.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional

FORBIDDEN_KEYS = frozenset({"family", "seed", "mech_seed", "oracle", "null", "references", "progress", "streak",
                            "split", "engine", "difficulty", "reference_net", "random_net", "private", "hidden",
                            "world_info", "spec"})
STATUSES = ("success", "failure", "uncertain")


@dataclass
class LabelRecord:
    env_id: str
    episode: int
    status: str                       # success / failure / uncertain (pending labels are never released)
    utility: Optional[float]          # normalised verified utility of the episode objective
    value: Optional[float]            # verified objective quantity in the world's unit
    t_resolved_days: Optional[float]
    evaluator: str
    goal_id: str = ""
    actor_id: str = ""
    t_release_wall: float = field(default_factory=time.time)
    kind: str = "terminal"            # terminal | self_goal | hindsight_relabel
    meta: dict = field(default_factory=dict)


def _scan(obj, path="") -> list:
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in FORBIDDEN_KEYS:
                bad.append(f"{path}.{k}")
            bad += _scan(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += _scan(v, f"{path}[{i}]")
    return bad


class LabelStore:
    def __init__(self, root: str):
        os.makedirs(root, exist_ok=True)
        self.path = os.path.join(root, "labels.jsonl")
        self._seen: set = set()
        for r in self.iterate():
            self._seen.add((r.env_id, r.episode, r.goal_id, r.kind))

    def append(self, rec: LabelRecord) -> bool:
        """Append once per (env, episode, goal, kind); returns False for a duplicate (never counted twice)."""
        if rec.status not in STATUSES:
            raise ValueError(f"label status {rec.status!r}: pending outcomes are not labels")
        d = asdict(rec)
        bad = _scan(d.get("meta", {}))
        if bad:
            raise ValueError(f"label carries private keys {bad}")
        key = (rec.env_id, rec.episode, rec.goal_id, rec.kind)
        if key in self._seen:
            return False
        with open(self.path, "a") as fh:
            fh.write(json.dumps(d, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._seen.add(key)
        return True

    def iterate(self) -> Iterator[LabelRecord]:
        if not os.path.exists(self.path):
            return
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield LabelRecord(**json.loads(line))

    def by_episode(self) -> dict:
        out = {}
        for r in self.iterate():
            out.setdefault((r.env_id, r.episode), []).append(r)
        return out

    def __len__(self) -> int:
        return len(self._seen)


def from_outcome_label(lab, evaluator: str = "", actor_id: str = "") -> Optional[LabelRecord]:
    """envs.experience.OutcomeLabel -> LabelRecord (None while pending)."""
    if lab.status == "pending":
        return None
    return LabelRecord(env_id=lab.env_id, episode=int(lab.episode), status=lab.status, utility=lab.utility,
                       value=lab.value, t_resolved_days=lab.t_resolved_days, evaluator=evaluator or lab.verifier,
                       actor_id=actor_id, goal_id=getattr(lab, "objective", "") or "")
