"""Released training outcome labels: the trainer-only channel (meta-reward spec 2.4, 4.1, 11.2; manifest OB-03, OB-05,
SE-03, CR-01; redesign plan 2026-11-26 §2 step 7, W2).

`LabelStore` is an append-only JSONL file (one record per line, fsynced) written only by the operator / harness side
and read only by the learner. Labels come from a `LabelSource` (`interfaces.LabelSource`: ``take_labels() ->
[LabelRecord]``, each label exactly once): `experience.ExperienceOperator`, `openworld_eval.OpenWorldEvaluator`
(and its process-isolated proxy), `capabilities.envs.ScoreEvaluator`. The canonical loop drains the source into the
store after every round (`drain`); the acting agent never holds a reference to either.

A label is released only after the outcome occurred; PENDING records are never written (pending is the absence of a
label). Each label names the (actor-visible, opaque) environment id and the episode it resolves, the goal id the
agent saw, the evaluator that produced it, the verified objective quantity, the normalised verified utility, the
resolution time (simulated days since the episode start), the episode's simulated time span (``t_start_days`` /
``t_end_days``, on the clock of `Observation.t` / `EventRecord.t_sim`, so the learner can join labels to archived
decisions by (env_id, time window)) and the wall-clock release time. Nothing private (hidden mechanism parameters,
oracle values, verifier progress/streak, family names, seeds) is ever written: `LabelStore.append` rejects records
whose meta carries such keys.

Streams: labels of an evaluation stream (`factory.make_env(..., split="val" | "test")`) carry
``meta["stream"] = "eval"``; a training `LabelStore` refuses them (``accept_eval=False``, the default), so a held-out
episode can never become a training target by accident.

The learner joins labels to archived decisions and uses them for: success / outcome predictor targets (every prefix
of the resolved episode), RUDDER return targets, COCOA/R-HCA rewarding outcomes, the outer meta-objective and
lifetime value. A verified label never enters an actor reward or an actor input.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Iterable, Iterator, Optional

FORBIDDEN_KEYS = frozenset({"family", "seed", "mech_seed", "oracle", "null", "references", "progress", "streak",
                            "split", "engine", "difficulty", "reference_net", "random_net", "private", "hidden",
                            "world_info", "spec"})
STATUSES = ("success", "failure", "uncertain")
KINDS = ("terminal", "self_goal", "hindsight_relabel", "check")
TRAIN_STREAM, EVAL_STREAM = "train", "eval"


@dataclass
class LabelRecord:
    env_id: str
    episode: int
    status: str                       # success / failure / uncertain (pending labels are never released)
    utility: Optional[float]          # normalised verified utility of the episode objective
    value: Optional[float]            # verified objective quantity in the world's unit
    t_resolved_days: Optional[float]  # simulated days since the episode start at which the outcome was verified
    evaluator: str
    goal_id: str = ""
    actor_id: str = ""
    t_release_wall: float = field(default_factory=time.time)
    kind: str = "terminal"            # terminal | self_goal | hindsight_relabel | check (off by default)
    meta: dict = field(default_factory=dict)
    t_start_days: Optional[float] = None   # simulated time (days, Observation.t clock) of the episode start
    t_end_days: Optional[float] = None     # simulated time at release (the label covers [t_start, t_end])

    @property
    def stream(self) -> str:
        return str((self.meta or {}).get("stream", TRAIN_STREAM))

    @staticmethod
    def from_dict(d: dict) -> "LabelRecord":
        names = {f.name for f in fields(LabelRecord)}
        return LabelRecord(**{k: v for k, v in d.items() if k in names})


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


def check_label(rec: LabelRecord) -> None:
    """Raise ValueError for a record that may not be released (pending status, unknown kind, private meta keys)."""
    if rec.status not in STATUSES:
        raise ValueError(f"label status {rec.status!r}: pending outcomes are not labels")
    if rec.kind not in KINDS:
        raise ValueError(f"label kind {rec.kind!r} not in {KINDS}")
    bad = _scan(rec.meta or {})
    if bad:
        raise ValueError(f"label carries private keys {bad}")


class LabelStore:
    def __init__(self, root: str, accept_eval: bool = False):
        os.makedirs(root, exist_ok=True)
        self.path = os.path.join(root, "labels.jsonl")
        self.accept_eval = bool(accept_eval)
        self._seen: set = set()
        for r in self.iterate():
            self._seen.add((r.env_id, r.episode, r.goal_id, r.kind))

    def append(self, rec: LabelRecord) -> bool:
        """Append once per (env, episode, goal, kind); returns False for a duplicate (never counted twice)."""
        check_label(rec)
        if rec.stream != TRAIN_STREAM and not self.accept_eval:
            raise ValueError(f"label of an evaluation stream ({rec.env_id}, episode {rec.episode}) refused by a "
                             f"training LabelStore (open it with accept_eval=True for evaluation records)")
        d = asdict(rec)
        key = (rec.env_id, rec.episode, rec.goal_id, rec.kind)
        if key in self._seen:
            return False
        with open(self.path, "a") as fh:
            fh.write(json.dumps(d, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._seen.add(key)
        return True

    def extend(self, recs: Iterable[LabelRecord]) -> int:
        return sum(1 for r in recs if self.append(r))

    def iterate(self) -> Iterator[LabelRecord]:
        if not os.path.exists(self.path):
            return
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield LabelRecord.from_dict(json.loads(line))

    def by_episode(self) -> dict:
        out = {}
        for r in self.iterate():
            out.setdefault((r.env_id, r.episode), []).append(r)
        return out

    def __len__(self) -> int:
        return len(self._seen)


def drain(source, store: Optional[LabelStore]) -> list:
    """One canonical-loop label step (plan §2 step 7): ``source.take_labels()`` -> ``store.append`` (when a store is
    given). Returns the labels taken (operator / learner side only; never hand them to the agent)."""
    if source is None:
        return []
    labs = list(source.take_labels())
    for r in labs:
        check_label(r)
        if store is not None:
            store.append(r)
    return labs


def from_outcome_label(lab, evaluator: str = "", actor_id: str = "", goal_id: str = "",
                       t_start_days: Optional[float] = None, t_end_days: Optional[float] = None,
                       meta: Optional[dict] = None) -> Optional[LabelRecord]:
    """envs.experience.OutcomeLabel -> LabelRecord (None while pending)."""
    if lab.status == "pending":
        return None
    return LabelRecord(env_id=lab.env_id, episode=int(lab.episode), status=lab.status, utility=lab.utility,
                       value=lab.value, t_resolved_days=lab.t_resolved_days, evaluator=evaluator or lab.verifier,
                       actor_id=actor_id, goal_id=goal_id or getattr(lab, "objective", "") or "",
                       t_start_days=t_start_days, t_end_days=t_end_days, meta=dict(meta or {}))
