"""cam_live.reward.integrity — guards that keep the learned reward honest (spec 7.2, 7.8, 16.6).

    psi isolation        the inner loss can never change psi: optimizer overlap check, gradient-path check, PsiGuard
    scale invariance     bounded output + within-batch standardisation: inflating reward magnitude is a no-op
    outcome screening    de-duplication by dedup_key; verifier signature required; imagined, self-authored and
                         unverified outcomes rejected; stale (non-prospective) predictions rejected
    self-goal discount   repeated or trivial self-generated goals lose weight and never supervise the outer objective
    noisy-TV gate        curiosity-type reward decays once epistemic uncertainty has stopped falling
    outcome ledger       append-only, hash-chained record of every outcome admitted to meta-learning

None of these guards is learned and none takes the policy or the reward network as an authority: the verifier's
signature check is injected (``OutcomeVerifier.check_signature``) and lives outside the learner.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from ..contracts import OutcomeObservation, OutcomeStatus, PredictionRecord, Trust, VerifiedOutcome

__all__ = ["IntegrityError", "scale_normalise", "bound_reward", "params_hash", "optimizer_overlap",
           "assert_optimizers_disjoint", "assert_reward_optimizer_exclusive", "depends_on", "assert_no_grad_path",
           "PsiGuard", "dedup_outcomes", "screen_observation", "screen_verified", "is_prospective", "LedgerEntry",
           "AdmitResult", "OutcomeLedger", "SelfGoalDiscount", "NoisyTVGate", "noisy_tv_audit", "SELF_SOURCES"]


class IntegrityError(RuntimeError):
    """A reward-integrity invariant was violated. Never caught and ignored by the learner."""


# ============================================================================================ scale and bounds
def bound_reward(raw: Tensor, bound: float) -> Tensor:
    """bound * tanh(raw / bound): identity near zero, never exceeds ``bound`` in magnitude."""
    return bound * torch.tanh(raw / bound)


def scale_normalise(r: Tensor, group: Optional[Tensor] = None, eps: float = 1e-12) -> Tensor:
    """(r - mean) / sqrt(var + eps) within the batch (within each ``group`` id when given). Differentiable.
    The result is unchanged by r -> c * r + d for any c > 0, so the policy update it drives cannot be changed by
    inflating or shifting the reward output; only *which* rows are rewarded more than others matters."""
    if group is None:
        d = r - r.mean()
        return d / torch.sqrt((d ** 2).mean() + eps)
    n = int(group.max()) + 1
    cnt = torch.zeros(n, dtype=r.dtype, device=r.device).index_add(0, group, torch.ones_like(r))
    mean = torch.zeros(n, dtype=r.dtype, device=r.device).index_add(0, group, r) / cnt
    d = r - mean[group]
    var = torch.zeros(n, dtype=r.dtype, device=r.device).index_add(0, group, d ** 2) / cnt
    return d / torch.sqrt(var[group] + eps)


# =============================================================================================== psi isolation
def _param_list(params: Union[nn.Module, Mapping[str, Tensor], Iterable[Tensor]]) -> List[Tensor]:
    if isinstance(params, nn.Module):
        return list(params.parameters())
    if isinstance(params, Mapping):
        return list(params.values())
    return list(params)


def params_hash(params: Union[nn.Module, Mapping[str, Tensor], Iterable[Tensor]]) -> str:
    """sha256 over the raw bytes of the tensors, in order (shape and dtype included)."""
    h = hashlib.sha256()
    for p in _param_list(params):
        t = p.detach().cpu().contiguous()
        h.update(f"{tuple(t.shape)}|{t.dtype}|".encode())
        h.update(t.view(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def optimizer_overlap(optimizer: torch.optim.Optimizer, params) -> List[int]:
    """Indices (into ``params``) of tensors that ``optimizer`` would update."""
    owned = {id(p) for g in optimizer.param_groups for p in g["params"]}
    return [i for i, p in enumerate(_param_list(params)) if id(p) in owned]


def assert_optimizers_disjoint(inner_optimizers: Sequence[torch.optim.Optimizer], reward_params) -> None:
    """No optimizer of the learner (inner / continual losses) may hold a reward parameter."""
    for i, opt in enumerate(inner_optimizers):
        hit = optimizer_overlap(opt, reward_params)
        if hit:
            raise IntegrityError(f"learner optimizer #{i} holds {len(hit)} reward parameter tensor(s)")


def assert_reward_optimizer_exclusive(reward_optimizer: torch.optim.Optimizer, reward_params) -> None:
    """The reward optimizer holds the reward parameters and nothing else."""
    owned = {id(p) for g in reward_optimizer.param_groups for p in g["params"]}
    want = {id(p) for p in _param_list(reward_params)}
    if owned != want:
        raise IntegrityError(f"reward optimizer mismatch: {len(owned - want)} foreign, {len(want - owned)} missing")


def depends_on(value: Tensor, params) -> bool:
    """True iff autograd has a path from any of ``params`` to ``value``."""
    ps = [p for p in _param_list(params) if p.requires_grad]
    if not ps or not value.requires_grad:
        return False
    grads = torch.autograd.grad(value.sum(), ps, allow_unused=True, retain_graph=True)
    return any(g is not None for g in grads)


def assert_no_grad_path(value: Tensor, params, what: str = "loss") -> None:
    """E.g. world-model and evaluator losses, and every first-order learner loss, must not depend on psi."""
    if depends_on(value, params):
        raise IntegrityError(f"{what} has a gradient path into the reward parameters")


class PsiGuard:
    """Wrap any learner step that is *not* the outer meta-update:

        with PsiGuard(reward_module):
            loss.backward(); learner_optimizer.step()

    On exit the reward parameters must be bit-identical and must not have received a gradient."""

    def __init__(self, reward_params, name: str = "learner step"):
        self.params, self.name = _param_list(reward_params), name
        self._hash = ""

    def __enter__(self) -> "PsiGuard":
        for p in self.params:
            p.grad = None
        self._hash = params_hash(self.params)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            return
        if any(p.grad is not None and bool((p.grad != 0).any()) for p in self.params):
            raise IntegrityError(f"{self.name}: a gradient reached the reward parameters")
        if params_hash(self.params) != self._hash:
            raise IntegrityError(f"{self.name}: the reward parameters changed")


# =========================================================================================== outcome screening
#: ``OutcomeObservation.source`` values that mark something the learner produced itself
SELF_SOURCES = frozenset({"self", "learner", "agent", "model", "policy", "prediction", "imagination", "imagined",
                          "planner", "world_model", "evaluator"})


def _obs_key(o: OutcomeObservation) -> str:
    return o.dedup_key or f"{o.query_id}|{o.observed_at!r}|{o.value!r}"


def dedup_outcomes(outcomes: Sequence[OutcomeObservation]) -> Tuple[List[OutcomeObservation], List[OutcomeObservation]]:
    """(kept, dropped): the first observation of each ``dedup_key`` is kept; the same event is never counted twice."""
    seen, kept, dropped = set(), [], []
    for o in outcomes:
        k = _obs_key(o)
        (dropped if k in seen else kept).append(o)
        seen.add(k)
    return kept, dropped


def screen_observation(o: OutcomeObservation, *, imagined: bool = False,
                       check_measurement: Optional[Callable[[OutcomeObservation], bool]] = None
                       ) -> Tuple[bool, str]:
    """May this measurement supervise the outer objective? ``check_measurement`` is the verifier-side
    authentication of the observation; without it no measurement is admitted (fail closed)."""
    if imagined:
        return False, "imagined"
    if (o.source or "").strip().lower() in SELF_SOURCES:
        return False, "self_authored"
    if not o.verified:
        return False, "not_verified"
    if o.status not in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED):
        return False, f"status_{o.status}"
    if o.value is None or o.value != o.value or abs(o.value) == float("inf"):
        return False, "no_finite_value"
    if check_measurement is None:
        return False, "no_measurement_authentication"
    try:
        ok = bool(check_measurement(o))
    except Exception:
        ok = False
    return (True, "") if ok else (False, "bad_signature")


def screen_verified(v: VerifiedOutcome, check_signature: Callable[[VerifiedOutcome], bool], *,
                    objective_origin: str = Trust.OPERATOR) -> Tuple[bool, str]:
    """May this objective outcome supervise the outer objective? Needs a valid verifier signature and an objective
    that was assigned independently of the learner (self-generated goals never qualify)."""
    if objective_origin == Trust.SELF:
        return False, "self_authored_objective"
    if objective_origin not in (Trust.OPERATOR, Trust.VERIFIER):
        return False, f"objective_origin_{objective_origin}"
    if not v.signature:
        return False, "unsigned"
    try:
        ok = bool(check_signature(v))
    except Exception:
        ok = False
    if not ok:
        return False, "bad_signature"
    if v.value != v.value or abs(v.value) == float("inf"):
        return False, "no_finite_value"
    return True, ""


def is_prospective(pred: PredictionRecord, obs: OutcomeObservation) -> bool:
    """A prediction counts only if it was registered before its outcome was observed (no stale or hindsight
    predictions in learning-progress or calibration signals) and was a real, not imagined-rollout, registration."""
    return pred.made_at < obs.observed_at and pred.query.query_id == obs.query_id \
        and pred.query.issued_at <= obs.observed_at


# ============================================================================================== outcome ledger
@dataclass(frozen=True)
class LedgerEntry:
    index: int
    prev: str
    key: str                  # de-duplication key of the event
    kind: str                 # "objective" | "measurement"
    objective_id: str
    group: str                # independent-outcome group (experiment / parent decision)
    utility: float
    observed_at: float
    signature: str
    content_sha256: str
    exp_ids: Tuple[str, ...]
    hash: str

    def body(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in ("index", "prev", "key", "kind", "objective_id", "group", "utility",
                                             "observed_at", "signature", "content_sha256")}
        d["exp_ids"] = list(self.exp_ids)
        return d


@dataclass
class AdmitResult:
    accepted: bool
    reason: str
    entry: Optional[LedgerEntry] = None


def _entry_hash(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class OutcomeLedger:
    """The only source of utilities for the outer objective. Append-only and hash-chained: every entry commits to
    its predecessor, so editing, removing or reordering an admitted outcome is detected by :meth:`verify_chain`
    (and by comparing :attr:`head` with the head recorded in an earlier ``MetaUpdateRecord``)."""

    GENESIS = "0" * 64

    def __init__(self, check_signature: Callable[[VerifiedOutcome], bool],
                 check_measurement: Optional[Callable[[OutcomeObservation], bool]] = None,
                 path: Optional[str] = None):
        self._check_signature, self._check_measurement = check_signature, check_measurement
        self.path = path
        self._entries: List[LedgerEntry] = []
        self._keys: Dict[str, int] = {}
        self.rejected: List[Tuple[str, str]] = []            # (reason, key)
        if path and os.path.exists(path):
            self._load(path)

    # ------------------------------------------------------------------------------------------- admission
    def _append(self, key: str, kind: str, objective_id: str, group: str, utility: float, observed_at: float,
                signature: str, content: str, exp_ids: Sequence[str]) -> AdmitResult:
        if key in self._keys:
            self.rejected.append(("duplicate", key))
            return AdmitResult(False, "duplicate", self._entries[self._keys[key]])
        body = {"index": len(self._entries), "prev": self.head, "key": key, "kind": kind,
                "objective_id": objective_id, "group": group or key, "utility": float(utility),
                "observed_at": float(observed_at), "signature": signature, "content_sha256": content,
                "exp_ids": list(exp_ids)}
        e = LedgerEntry(**{**body, "exp_ids": tuple(exp_ids)}, hash=_entry_hash(body))
        self._entries.append(e)
        self._keys[key] = e.index
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps({**body, "hash": e.hash}, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return AdmitResult(True, "", e)

    def admit_verified(self, v: VerifiedOutcome, *, utility: Optional[float] = None, group: str = "",
                       exp_ids: Sequence[str] = (), objective_origin: str = Trust.OPERATOR) -> AdmitResult:
        """Admit a signed objective outcome. ``utility`` defaults to ``v.value``."""
        key = "obj:" + hashlib.sha256(json.dumps(
            [v.objective_id, v.status, v.value, sorted(v.evidence_refs) or v.evaluated_at]).encode()).hexdigest()[:24]
        ok, reason = screen_verified(v, self._check_signature, objective_origin=objective_origin)
        if not ok:
            self.rejected.append((reason, key))
            return AdmitResult(False, reason)
        return self._append(key, "objective", v.objective_id, group, v.value if utility is None else utility,
                            v.evaluated_at, v.signature, v.content_hash(), exp_ids)

    def admit_measurement(self, o: OutcomeObservation, *, objective_id: str = "", utility: Optional[float] = None,
                          group: str = "", exp_ids: Sequence[str] = (), imagined: bool = False,
                          signature: str = "") -> AdmitResult:
        """Admit a verifier measurement (``OutcomeVerifier.measure``) authenticated by ``check_measurement``."""
        key = "obs:" + _obs_key(o)
        ok, reason = screen_observation(o, imagined=imagined, check_measurement=self._check_measurement)
        if not ok:
            self.rejected.append((reason, key))
            return AdmitResult(False, reason)
        return self._append(key, "measurement", objective_id, group, o.value if utility is None else utility,
                            o.observed_at, signature, o.content_hash(), exp_ids)

    # ------------------------------------------------------------------------------------------- reading
    @property
    def head(self) -> str:
        return self._entries[-1].hash if self._entries else self.GENESIS

    def entries(self) -> List[LedgerEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def verify_chain(self) -> Tuple[bool, int]:
        """(ok, index of the first inconsistent entry or -1)."""
        prev = self.GENESIS
        for i, e in enumerate(self._entries):
            if e.index != i or e.prev != prev or _entry_hash(e.body()) != e.hash:
                return False, i
            prev = e.hash
        return True, -1

    def _load(self, path: str) -> None:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                d["exp_ids"] = tuple(d.get("exp_ids", ()))
                e = LedgerEntry(**d)
                self._entries.append(e)
                self._keys[e.key] = e.index
        ok, bad = self.verify_chain()
        if not ok:
            raise IntegrityError(f"outcome ledger {path} fails its hash chain at entry {bad}")

    def independent_groups(self, since_index: int = 0) -> int:
        return len({e.group for e in self._entries[since_index:]})


# ========================================================================================== self-generated goals
class SelfGoalDiscount:
    """Weight in [0, 1] for anything earned on a self-generated goal. Repeats of the same goal decay
    geometrically; goals whose success was already predicted with near certainty (nothing to achieve) get no
    weight. Outcomes of self-generated goals never enter the outer objective at all (``screen_verified``)."""

    def __init__(self, decay: float = 0.5, trivial_above: float = 0.9):
        self.decay, self.trivial_above = float(decay), float(trivial_above)
        self.counts: Dict[str, int] = {}

    @staticmethod
    def key(goal_text: str) -> str:
        """Same goal up to case, whitespace, punctuation and embedded numbers."""
        norm = re.sub(r"[^a-z ]+", " ", re.sub(r"\d+(\.\d+)?", "#", goal_text.casefold()))
        return hashlib.sha256(" ".join(norm.split()).encode()).hexdigest()[:16]

    def weight(self, goal_text: str, p_success_before: float, *, register: bool = True) -> float:
        k = self.key(goal_text)
        n = self.counts.get(k, 0)
        if register:
            self.counts[k] = n + 1
        p = min(max(float(p_success_before), 0.0), 1.0)
        difficulty = min(1.0, max(0.0, (1.0 - p) / max(1e-9, 1.0 - self.trivial_above))) if p > self.trivial_above \
            else 1.0
        return (self.decay ** n) * difficulty

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "trivial_above": self.trivial_above, "counts": dict(self.counts)}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        self.decay, self.trivial_above = float(d["decay"]), float(d["trivial_above"])
        self.counts = dict(d["counts"])


# ====================================================================================================== noisy TV
class NoisyTVGate:
    """Per-context multiplier in [0, 1] for curiosity-type reward. While epistemic uncertainty about a context
    keeps falling the gate stays open; once it has stopped falling (irreducible randomness, or nothing left to
    learn) the gate closes geometrically, whatever the prediction error is."""

    def __init__(self, window: int = 6, min_rel_drop: float = 0.05, decay: float = 0.6, recover: float = 0.5):
        self.window, self.min_rel_drop, self.decay, self.recover = int(window), min_rel_drop, decay, recover
        self._hist: Dict[str, Deque[float]] = {}
        self._gate: Dict[str, float] = {}

    def update(self, key: str, epistemic: float) -> float:
        h = self._hist.setdefault(key, collections.deque(maxlen=self.window))
        h.append(float(epistemic))
        g = self._gate.get(key, 1.0)
        if len(h) == self.window:
            first, last = h[0], h[-1]
            falling = first > 0 and (first - last) / first >= self.min_rel_drop
            g = min(1.0, g + (1.0 - g) * self.recover) if falling else g * self.decay
        self._gate[key] = g
        return g

    def gate(self, key: str) -> float:
        return self._gate.get(key, 1.0)

    def state_dict(self) -> Dict[str, Any]:
        return {"hist": {k: list(v) for k, v in self._hist.items()}, "gate": dict(self._gate)}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        self._hist = {k: collections.deque(v, maxlen=self.window) for k, v in d["hist"].items()}
        self._gate = dict(d["gate"])


def noisy_tv_audit(rewards: Sequence[float], epistemic: Sequence[float], *, tail: float = 0.25,
                   max_tail_ratio: float = 0.2, flat_rel_drop: float = 0.05) -> Dict[str, Any]:
    """Audit a reward stream paid on one context: if epistemic uncertainty is flat over the last ``tail`` fraction
    of the stream, the mean |reward| there must be at most ``max_tail_ratio`` of the mean |reward| at the start.
    ``passed`` is False when reward keeps being paid for randomness that no longer teaches anything."""
    n = len(rewards)
    if n != len(epistemic) or n < 8:
        raise ValueError("need aligned streams of at least 8 steps")
    k = max(2, int(n * tail))
    head = sum(abs(x) for x in rewards[:k]) / k
    tail_r = sum(abs(x) for x in rewards[-k:]) / k
    e0, e1 = epistemic[-k], epistemic[-1]
    flat = e0 <= 0 or (e0 - e1) / e0 < flat_rel_drop
    ratio = tail_r / head if head > 0 else (0.0 if tail_r == 0 else float("inf"))
    return {"epistemic_flat_in_tail": flat, "head_reward": head, "tail_reward": tail_r, "tail_ratio": ratio,
            "passed": (not flat) or ratio <= max_tail_ratio}
