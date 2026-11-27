"""cam_live.predict.registry — prospective registration of predictions and experiments (spec 4.2, 6.4, 11.1).

Both registries are append-only, hash-chained JSONL logs (``RunFiles.PREDICTIONS`` / ``RunFiles.EXPERIMENTS``):
state is only ever changed by appending an event, every event carries the hash of the previous one, and reopening
a log replays and verifies it.

``PredictionRegistry``
    * a prediction is registered *before* its outcome exists (unresolved, not dated in the future);
    * it can never be edited — re-registering an id is an error;
    * it is resolved exactly once, by a real ``OutcomeObservation`` (never an imagined value) whose
      ``observed_at`` is strictly later than both the prediction's ``made_at`` and the time it was registered;
    * the negative log-likelihood of the outcome under the registered distribution is computed at resolution.

``ExperimentRegistry``
    * question, hypothesis, arms and predictions are registered before acting;
    * randomised assignment draws an arm for each unit with recorded probabilities;
    * outcomes are attached per arm; the experiment matures after its horizon;
    * the conclusion records supported / not supported, the test that was used and alternative explanations.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..contracts import (Clock, Cost, ExperimentRecord, OutcomeKind, OutcomeObservation, OutcomeStatus,
                         PredictionRecord, WallClock, new_id)
from . import calibration as C
from . import dists as D
from .imagination import refuse_imagined


class RegistryError(ValueError):
    """A registry rule was violated; nothing was written."""


class DuplicatePredictionError(RegistryError):
    """An id that is already registered was registered again (predictions cannot be edited)."""


class AlreadyResolvedError(RegistryError):
    """A prediction is resolved exactly once."""


class ResolutionBeforePredictionError(RegistryError):
    """The outcome is not strictly later than the prediction."""


class UnknownRecordError(RegistryError, KeyError):
    """No record with that id."""


class TamperedRegistryError(RegistryError):
    """The log on disk does not match its hash chain."""


GENESIS = "0" * 64


class _ChainedLog:
    """Append-only JSONL with a sha256 chain. One writer process at a time is enforced with ``flock``."""

    def __init__(self, path: str):
        self.path = path
        self.head = GENESIS
        self.seq = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    @staticmethod
    def _digest(prev: str, seq: int, op: str, data: Dict[str, Any]) -> str:
        body = json.dumps({"prev": prev, "seq": seq, "op": op, "data": data}, sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()

    def replay(self) -> List[Tuple[str, Dict[str, Any]]]:
        events: List[Tuple[str, Dict[str, Any]]] = []
        if not os.path.exists(self.path):
            return events
        with open(self.path, "r", encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError as e:
                    raise TamperedRegistryError(f"{self.path}:{n}: not valid JSON ({e})") from e
                if ev.get("prev") != self.head or ev.get("seq") != self.seq + 1 or \
                        ev.get("hash") != self._digest(ev["prev"], ev["seq"], ev["op"], ev["data"]):
                    raise TamperedRegistryError(f"{self.path}:{n}: hash chain broken")
                self.head, self.seq = ev["hash"], ev["seq"]
                events.append((ev["op"], ev["data"]))
        return events

    def append(self, op: str, data: Dict[str, Any]) -> None:
        data = json.loads(json.dumps(data, sort_keys=True, default=str))          # exactly what will be stored
        with open(self.path, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                seq = self.seq + 1
                h = self._digest(self.head, seq, op, data)
                f.write(json.dumps({"seq": seq, "op": op, "data": data, "prev": self.head, "hash": h},
                                   sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
                self.head, self.seq = h, seq
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def verify_log(path: str) -> int:
    """Re-read a registry file and check its chain; returns the number of events."""
    log = _ChainedLog(path)
    return len(log.replay())


# ----------------------------------------------------------------------------------------------------------------
# predictions
# ----------------------------------------------------------------------------------------------------------------
class PredictionRegistry:
    def __init__(self, path: str, clock: Optional[Clock] = None, future_tolerance_s: float = 1e-6):
        self.clock: Clock = clock or WallClock()
        self.future_tolerance_s = future_tolerance_s
        self._log = _ChainedLog(path)
        self._recs: Dict[str, PredictionRecord] = {}
        self._registered_at: Dict[str, float] = {}
        for op, data in self._log.replay():
            self._apply(op, data)

    # ---- state transitions (also used for replay) ----
    def _apply(self, op: str, data: Dict[str, Any]) -> None:
        if op == "register":
            rec = PredictionRecord.from_dict(data["record"])
            self._recs[rec.prediction_id] = rec
            self._registered_at[rec.prediction_id] = float(data["registered_at"])
        elif op == "resolve":
            rec = self._recs[data["prediction_id"]]
            rec.resolved_at, rec.resolved_value = data["resolved_at"], data["resolved_value"]
            rec.resolved_status, rec.nll = data["resolved_status"], data["nll"]
        else:
            raise TamperedRegistryError(f"unknown event {op!r} in a prediction registry")

    # ---- API ----
    def register(self, rec: PredictionRecord) -> PredictionRecord:
        """Record a prediction before its outcome is known. Returns the stored copy."""
        if not isinstance(rec, PredictionRecord):
            raise RegistryError("register() takes a contracts.PredictionRecord")
        if rec.prediction_id in self._recs:
            raise DuplicatePredictionError(f"prediction {rec.prediction_id!r} is already registered and cannot be "
                                           f"edited")
        if rec.resolved_at is not None or rec.resolved_value is not None or rec.nll is not None or \
                rec.resolved_status != OutcomeStatus.PENDING:
            raise RegistryError("a prediction must be registered unresolved")
        now = self.clock.now()
        if rec.made_at > now + self.future_tolerance_s:
            raise RegistryError(f"made_at {rec.made_at} is in the future of the registry clock {now}")
        try:
            D.from_record(rec.dist)
        except (KeyError, ValueError, TypeError) as e:
            raise RegistryError(f"the predicted distribution cannot be evaluated: {e}") from e
        if D.FAMILY_KIND[rec.dist.family] != rec.query.kind:
            raise RegistryError(f"a {rec.dist.family!r} distribution cannot predict a {rec.query.kind!r} outcome")
        self._log.append("register", {"record": rec.to_dict(), "registered_at": now})
        self._apply("register", {"record": rec.to_dict(), "registered_at": now})
        return self.get(rec.prediction_id)

    def resolve(self, prediction_id: str, outcome: OutcomeObservation) -> PredictionRecord:
        """Attach the real outcome, exactly once, and score it."""
        refuse_imagined(outcome, "outcome of a prediction")
        if not isinstance(outcome, OutcomeObservation):
            raise RegistryError("resolve() takes a contracts.OutcomeObservation")
        if prediction_id not in self._recs:
            raise UnknownRecordError(f"no prediction {prediction_id!r}")
        rec = self._recs[prediction_id]
        if rec.resolved_at is not None or rec.resolved_status != OutcomeStatus.PENDING:
            raise AlreadyResolvedError(f"prediction {prediction_id!r} was already resolved")
        if outcome.query_id != rec.query.query_id:
            raise RegistryError(f"outcome answers query {outcome.query_id!r}, the prediction is about "
                                f"{rec.query.query_id!r}")
        if outcome.status not in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED, OutcomeStatus.UNOBSERVABLE):
            raise RegistryError(f"status {outcome.status!r} is not a resolution; pending predictions stay pending")
        made = max(rec.made_at, self._registered_at[prediction_id])
        if not (outcome.observed_at > made):
            raise ResolutionBeforePredictionError(
                f"outcome observed at {outcome.observed_at} is not strictly after the prediction "
                f"(made {rec.made_at}, registered {self._registered_at[prediction_id]})")
        nll: Optional[float] = None
        value: Optional[float] = None
        if outcome.status != OutcomeStatus.UNOBSERVABLE:
            if outcome.value is None or not math.isfinite(float(outcome.value)):
                raise RegistryError("an observed or censored outcome needs a finite value")
            value = float(outcome.value)
            try:
                nll = -D.record_log_prob(rec.dist, value, outcome.status)
            except ValueError as e:
                raise RegistryError(str(e)) from e
        data = {"prediction_id": prediction_id, "resolved_at": float(outcome.observed_at), "resolved_value": value,
                "resolved_status": outcome.status, "nll": nll, "verified": bool(outcome.verified),
                "source": outcome.source, "dedup_key": outcome.dedup_key}
        self._log.append("resolve", data)
        self._apply("resolve", data)
        return self.get(prediction_id)

    def get(self, prediction_id: str) -> PredictionRecord:
        if prediction_id not in self._recs:
            raise UnknownRecordError(f"no prediction {prediction_id!r}")
        return copy.deepcopy(self._recs[prediction_id])

    def __contains__(self, prediction_id: str) -> bool:
        return prediction_id in self._recs

    def __len__(self) -> int:
        return len(self._recs)

    def registered_at(self, prediction_id: str) -> float:
        return self._registered_at[prediction_id]

    def records(self) -> List[PredictionRecord]:
        return [copy.deepcopy(r) for r in self._recs.values()]

    def pending(self, now: Optional[float] = None) -> List[PredictionRecord]:
        """Unresolved predictions; with ``now`` only those whose query is due."""
        out = [r for r in self._recs.values() if r.resolved_status == OutcomeStatus.PENDING]
        if now is not None:
            out = [r for r in out if r.query.due_at <= now]
        return [copy.deepcopy(r) for r in out]

    def scoreable(self) -> List[PredictionRecord]:
        """Resolved by a real observed or censored outcome (unobservable ones are never scored)."""
        return [copy.deepcopy(r) for r in self._recs.values()
                if r.resolved_status in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED)]

    def by_decision(self, decision_id: str) -> List[PredictionRecord]:
        return [copy.deepcopy(r) for r in self._recs.values() if r.decision_id == decision_id]

    def by_query(self, query_id: str) -> List[PredictionRecord]:
        return [copy.deepcopy(r) for r in self._recs.values() if r.query.query_id == query_id]

    def score(self, family_of: Optional[Callable[[PredictionRecord], str]] = None, **kw: Any) -> Dict[str, Any]:
        """Calibration report over everything scoreable, with the prospective guard on registration time."""
        return C.score_predictions(self.scoreable(), family_of=family_of,
                                   registered_at=lambda r: self._registered_at[r.prediction_id], **kw)

    def verify_chain(self) -> int:
        return verify_log(self._log.path)


# ----------------------------------------------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Assignment:
    experiment_id: str
    unit_id: str
    arm: str
    arm_index: int
    probability: float               # probability with which this arm was assigned
    probabilities: List[float]       # the full assignment distribution over arms


def welch_test(a: Sequence[float], b: Sequence[float], level: float = 0.95) -> Dict[str, Any]:
    """Two-sample Welch t-test of mean(a) - mean(b) with a studentised interval."""
    na, nb = len(a), len(b)
    out: Dict[str, Any] = {"test": "welch_t", "n_a": na, "n_b": nb, "level": level}
    if na < 2 or nb < 2:
        out.update(valid=False, reason="each arm needs at least two outcomes")
        return out
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se2 = va / na + vb / nb
    diff = ma - mb
    if se2 <= 0:
        out.update(valid=False, reason="no variation in either arm", diff=diff)
        return out
    se = math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t = diff / se
    half = C.t_quantile(0.5 + level / 2, df) * se
    out.update(valid=True, diff=diff, se=se, t=t, df=df, p_value=C.t_sf2(t, df), lo=diff - half, hi=diff + half,
               mean_a=ma, mean_b=mb)
    return out


class ExperimentRegistry:
    STATUSES = ("registered", "running", "matured", "closed")
    ASSIGNMENTS = ("randomized", "sequential", "none")

    def __init__(self, path: str, predictions: PredictionRegistry, clock: Optional[Clock] = None, seed: int = 0):
        self.clock: Clock = clock or predictions.clock
        self.predictions = predictions
        self.seed = seed
        self._log = _ChainedLog(path)
        self._recs: Dict[str, ExperimentRecord] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}
        for op, data in self._log.replay():
            self._apply(op, data)

    def _apply(self, op: str, data: Dict[str, Any]) -> None:
        eid = data["experiment_id"]
        if op == "register":
            self._recs[eid] = ExperimentRecord.from_dict(data["record"])
            self._meta[eid] = {"arm_probs": list(data["arm_probs"]), "assignments": [], "outcome_arms": {},
                               "started_at": None, "test": None, "concluded_at": None,
                               "ordinal": len(self._recs) - 1}
            return
        rec, meta = self._recs[eid], self._meta[eid]
        if op == "start":
            rec.status = "running"
            meta["started_at"] = data["ts"]
        elif op == "assign":
            meta["assignments"].append({k: data[k] for k in ("unit_id", "arm", "arm_index", "probability", "ts")})
        elif op == "decision":
            rec.decision_ids.append(data["decision_id"])
        elif op == "outcome":
            rec.actual.append(OutcomeObservation.from_dict(data["outcome"]))
            meta["outcome_arms"][str(len(rec.actual) - 1)] = data["arm"]
        elif op == "mature":
            rec.status = "matured"
        elif op == "conclude":
            rec.status, rec.supported = "closed", data["supported"]
            rec.alternative_explanations = list(data["alternative_explanations"])
            rec.measurement_uncertainty, rec.next_action = data["measurement_uncertainty"], data["next_action"]
            meta["test"], meta["concluded_at"] = data["test"], data["ts"]
        else:
            raise TamperedRegistryError(f"unknown event {op!r} in an experiment registry")

    def _event(self, op: str, data: Dict[str, Any]) -> None:
        self._log.append(op, data)
        self._apply(op, json.loads(json.dumps(data, sort_keys=True, default=str)))

    def _need(self, experiment_id: str) -> ExperimentRecord:
        if experiment_id not in self._recs:
            raise UnknownRecordError(f"no experiment {experiment_id!r}")
        return self._recs[experiment_id]

    # ---- registration (before acting) ----
    def register(self, *, question: str, hypothesis: str, predictions: Sequence[PredictionRecord],
                 arms: Sequence[str] = (), assignment: str = "none", arm_probs: Optional[Sequence[float]] = None,
                 intervention: str = "", comparison: str = "", horizon_s: float = 0.0, cost: Optional[Cost] = None,
                 evidence_refs: Sequence[str] = (), experiment_id: Optional[str] = None) -> ExperimentRecord:
        if not question.strip() or not hypothesis.strip():
            raise RegistryError("an experiment needs a question and a hypothesis before it starts")
        if not predictions:
            raise RegistryError("an experiment needs at least one registered prediction before acting")
        if assignment not in self.ASSIGNMENTS:
            raise RegistryError(f"assignment must be one of {self.ASSIGNMENTS}")
        arms = list(arms)
        if len(set(arms)) != len(arms):
            raise RegistryError("arm names must be unique")
        if assignment == "randomized" and len(arms) < 2:
            raise RegistryError("randomised assignment needs at least two arms")
        probs = [1.0 / len(arms)] * len(arms) if arm_probs is None and arms else list(arm_probs or [])
        if arms and (len(probs) != len(arms) or any(p <= 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-9):
            raise RegistryError("arm_probs must be positive, one per arm, and sum to 1")
        eid = experiment_id or new_id("exp")
        if eid in self._recs:
            raise DuplicatePredictionError(f"experiment {eid!r} is already registered and cannot be edited")
        stored: List[PredictionRecord] = []
        for p in predictions:
            if p.query.experiment_id not in ("", eid):
                raise RegistryError("a prediction belongs to another experiment")
            if p.prediction_id in self.predictions:
                raise DuplicatePredictionError(f"prediction {p.prediction_id!r} is already registered")
        for p in predictions:
            q = copy.deepcopy(p)
            q.query.experiment_id = eid
            stored.append(self.predictions.register(q))
        rec = ExperimentRecord(experiment_id=eid, created_at=self.clock.now(), question=question,
                               hypothesis=hypothesis, evidence_refs=list(evidence_refs), intervention=intervention,
                               comparison=comparison, arms=arms, assignment=assignment, predicted=stored,
                               cost=cost or Cost(), horizon_s=float(horizon_s), status="registered")
        self._event("register", {"experiment_id": eid, "record": rec.to_dict(), "arm_probs": probs})
        return self.get(eid)

    # ---- running ----
    def start(self, experiment_id: str) -> ExperimentRecord:
        rec = self._need(experiment_id)
        if rec.status != "registered":
            raise RegistryError(f"experiment is {rec.status}, it cannot be started again")
        self._event("start", {"experiment_id": experiment_id, "ts": self.clock.now()})
        return self.get(experiment_id)

    def assign(self, experiment_id: str, unit_id: str) -> Assignment:
        """Randomly assign ``unit_id`` to an arm. The draw is a deterministic function of (registry seed, the
        experiment's registration ordinal and arms, unit) — reproducible and auditable, also by a process restored from
        the registry file whose new experiments get different (random) ids — and its probability is recorded."""
        rec, meta = self._need(experiment_id), self._meta[experiment_id]
        if rec.assignment != "randomized":
            raise RegistryError("this experiment does not use randomised assignment")
        if rec.status not in ("registered", "running"):
            raise RegistryError(f"experiment is {rec.status}; no further assignment")
        if any(a["unit_id"] == unit_id for a in meta["assignments"]):
            raise RegistryError(f"unit {unit_id!r} was already assigned")
        if rec.status == "registered":
            self.start(experiment_id)
        probs = meta["arm_probs"]
        key = f"{meta.get('ordinal', experiment_id)}|{'|'.join(rec.arms)}"
        h = hashlib.sha256(f"{self.seed}|{key}|{unit_id}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2.0 ** 64
        acc, idx = 0.0, len(probs) - 1
        for i, p in enumerate(probs):
            acc += p
            if u < acc:
                idx = i
                break
        data = {"experiment_id": experiment_id, "unit_id": unit_id, "arm": rec.arms[idx], "arm_index": idx,
                "probability": probs[idx], "ts": self.clock.now()}
        self._event("assign", data)
        return Assignment(experiment_id, unit_id, rec.arms[idx], idx, probs[idx], list(probs))

    def link_decision(self, experiment_id: str, decision_id: str) -> None:
        rec = self._need(experiment_id)
        if rec.status == "closed":
            raise RegistryError("the experiment is closed")
        self._event("decision", {"experiment_id": experiment_id, "decision_id": decision_id})

    def add_outcome(self, experiment_id: str, outcome: OutcomeObservation, arm: str = "") -> None:
        refuse_imagined(outcome, "outcome of an experiment")
        if not isinstance(outcome, OutcomeObservation):
            raise RegistryError("add_outcome() takes a contracts.OutcomeObservation")
        rec = self._need(experiment_id)
        if rec.status not in ("running", "matured"):
            raise RegistryError(f"experiment is {rec.status}; outcomes are accepted while running or matured")
        if arm and arm not in rec.arms:
            raise RegistryError(f"unknown arm {arm!r}")
        if not (outcome.observed_at > rec.created_at):
            raise ResolutionBeforePredictionError("the outcome predates the registration of the experiment")
        if outcome.dedup_key and any(o.dedup_key == outcome.dedup_key for o in rec.actual):
            return                                                     # the same event reported twice counts once
        self._event("outcome", {"experiment_id": experiment_id, "outcome": outcome.to_dict(), "arm": arm})

    def mature(self, experiment_id: str, now: Optional[float] = None) -> bool:
        """Mark the experiment matured once its horizon has elapsed since it started. Returns the new state."""
        rec, meta = self._need(experiment_id), self._meta[experiment_id]
        if rec.status == "matured":
            return True
        if rec.status != "running":
            return False
        now = self.clock.now() if now is None else now
        if now < meta["started_at"] + rec.horizon_s:
            return False
        self._event("mature", {"experiment_id": experiment_id, "ts": now})
        return True

    # ---- analysis and conclusion ----
    def outcomes_by_arm(self, experiment_id: str, query_ids: Optional[Iterable[str]] = None
                        ) -> Dict[str, List[float]]:
        rec, meta = self._need(experiment_id), self._meta[experiment_id]
        wanted = None if query_ids is None else set(query_ids)
        out: Dict[str, List[float]] = {a: [] for a in rec.arms}
        for i, o in enumerate(rec.actual):
            arm = meta["outcome_arms"].get(str(i), "")
            if o.status == OutcomeStatus.OBSERVED and o.value is not None and (wanted is None or o.query_id in wanted):
                out.setdefault(arm, []).append(float(o.value))
        return out

    def analyse(self, experiment_id: str, arm_a: Optional[str] = None, arm_b: Optional[str] = None,
                query_ids: Optional[Iterable[str]] = None, level: float = 0.95) -> Dict[str, Any]:
        """Welch comparison of two arms on the observed outcomes (defaults: the first two arms)."""
        rec = self._need(experiment_id)
        if len(rec.arms) < 2:
            return {"test": "welch_t", "valid": False, "reason": "fewer than two arms"}
        a, b = arm_a or rec.arms[0], arm_b or rec.arms[1]
        groups = self.outcomes_by_arm(experiment_id, query_ids)
        res = welch_test(groups.get(a, []), groups.get(b, []), level)
        res.update(arm_a=a, arm_b=b, randomized=rec.assignment == "randomized")
        return res

    def conclude(self, experiment_id: str, *, supported: Optional[bool], test: Dict[str, Any],
                 alternative_explanations: Sequence[str], measurement_uncertainty: str = "",
                 next_action: str = "") -> ExperimentRecord:
        """Close a matured experiment. ``supported`` is True / False, or None when the evidence is inconclusive.
        ``test`` names the test that was used (e.g. the result of :meth:`analyse`); the alternative explanations
        considered must be stated explicitly (an empty list is a statement too)."""
        rec = self._need(experiment_id)
        if rec.status != "matured":
            raise RegistryError(f"experiment is {rec.status}; it can only be concluded once matured")
        if not isinstance(test, dict) or not test.get("test"):
            raise RegistryError("the conclusion must name the test that was used")
        if alternative_explanations is None:
            raise RegistryError("state the alternative explanations that were considered")
        self._event("conclude", {"experiment_id": experiment_id, "supported": supported, "test": test,
                                 "alternative_explanations": list(alternative_explanations),
                                 "measurement_uncertainty": measurement_uncertainty, "next_action": next_action,
                                 "ts": self.clock.now()})
        return self.get(experiment_id)

    # ---- reads ----
    def get(self, experiment_id: str) -> ExperimentRecord:
        rec = copy.deepcopy(self._need(experiment_id))
        rec.predicted = [self.predictions.get(p.prediction_id) for p in rec.predicted]     # current resolution state
        return rec

    def meta(self, experiment_id: str) -> Dict[str, Any]:
        """Registry-side facts the contract record has no field for: arm probabilities, the assignments with
        their probabilities, the arm of each outcome, start time and the test used."""
        self._need(experiment_id)
        return copy.deepcopy(self._meta[experiment_id])

    def records(self) -> List[ExperimentRecord]:
        return [self.get(e) for e in self._recs]

    def open(self) -> List[ExperimentRecord]:
        return [self.get(e) for e, r in self._recs.items() if r.status != "closed"]

    def verify_chain(self) -> int:
        return verify_log(self._log.path)
