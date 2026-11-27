"""cam_live.reward.integrity — guards that keep the learned reward honest (spec 7.2, 7.8, 16.6).

    psi isolation        the inner loss can never change psi: optimizer overlap check, gradient-path check, PsiGuard
    scale invariance     bounded output + within-batch standardisation: inflating reward magnitude is a no-op
    outcome screening    de-duplication by dedup_key; verifier signature required; imagined, self-authored and
                         unverified outcomes rejected; stale (non-prospective) predictions rejected
    utility mapping      the utility that supervises psi is derived here from the signed outcome itself
                         (:func:`objective_utility`, :func:`measurement_utility`); a caller cannot choose it
    objective origin     only objectives whose *recorded* origin is the operator (or the verifier) supervise; the
                         origin is registered from the ``TaskSpec`` (``register_objective``); unknown -> excluded
    self-goal discount   repeated (also reworded / semantically near-duplicate) or trivial self-generated goals lose
                         weight and never supervise the outer objective
    noisy-TV gate        curiosity-type reward decays once epistemic uncertainty has stopped falling
    outcome ledger       append-only, hash-chained record of every outcome admitted to meta-learning (and of every
                         validly signed outcome that was refused), anchored by a head file next to it

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

from ..contracts import GoalStatus, OutcomeObservation, OutcomeStatus, PredictionRecord, Trust, VerifiedOutcome

__all__ = ["IntegrityError", "scale_normalise", "bound_reward", "params_hash", "optimizer_overlap",
           "assert_optimizers_disjoint", "assert_reward_optimizer_exclusive", "depends_on", "assert_no_grad_path",
           "PsiGuard", "dedup_outcomes", "screen_observation", "screen_verified", "is_prospective", "LedgerEntry",
           "AdmitResult", "OutcomeLedger", "SelfGoalDiscount", "NoisyTVGate", "noisy_tv_audit", "SELF_SOURCES",
           "objective_utility", "measurement_utility", "OBJECTIVE_UTILITY_MAPPING", "MEASUREMENT_UTILITY_MAPPING",
           "SUPERVISING_ORIGINS", "utility_matches"]


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


# ============================================================================================ utility mapping
#: name of the fixed mapping from a signed objective outcome to the utility that supervises psi (stored in the ledger)
OBJECTIVE_UTILITY_MAPPING = "goal_progress_v1"
#: name of the fixed mapping from an authenticated verifier measurement to its utility
MEASUREMENT_UTILITY_MAPPING = "measured_value_v1"
#: objective origins that may supervise the outer objective
SUPERVISING_ORIGINS = (Trust.OPERATOR, Trust.VERIFIER)


def objective_utility(v: VerifiedOutcome) -> float:
    """The *level* of verified goal achievement of a signed outcome, in [0, 1] (mapping ``goal_progress_v1``)::

        1.0                                         status == ACHIEVED
        0.5 * clip(value / max(target, 1), 0, 1)    otherwise (PENDING, FAILED, ...)

    It reads nothing but the signed fields ``status``, ``value`` and ``target``. The utility admitted for an
    evaluation is the change of this level since the previous admitted evaluation of the same objective (the first
    evaluation counts from 0), so progress is never counted twice and the sum over an objective's evaluations is its
    final level. (The same scalar as ``diagnostics.collect.verified_return`` for LabWorld lifetimes.)"""
    if v.status == GoalStatus.ACHIEVED:
        return 1.0
    return max(0.0, min(1.0, float(v.value) / max(float(v.target), 1.0))) * 0.5


def measurement_utility(o: OutcomeObservation) -> float:
    """Utility of an authenticated verifier measurement (mapping ``measured_value_v1``): its measured value."""
    return float(o.value)


def utility_matches(claimed: float, derived: float, rtol: float = 1e-9) -> bool:
    """Does a caller-stated utility agree with the derived one (to rounding)?"""
    c, d = float(claimed), float(derived)
    return c == c and abs(c - d) <= rtol * max(1.0, abs(d))


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
    note: str = ""            # kind "refused": why a validly signed outcome was not admitted (and what was claimed)

    def body(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in ("index", "prev", "key", "kind", "objective_id", "group", "utility",
                                             "observed_at", "signature", "content_sha256")}
        d["exp_ids"] = list(self.exp_ids)
        if self.note:
            d["note"] = self.note
        return d

    @property
    def admitted(self) -> bool:
        """An outcome that supervises the outer objective (kinds "objective" and "measurement")."""
        return self.kind in ("objective", "measurement")


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
    (and by comparing :attr:`head` with the head recorded in an earlier ``MetaUpdateRecord``).

    * **Utility.** The number stored for an admitted outcome is derived here from the signed record
      (:func:`objective_utility` as progress since the previous admitted evaluation of the objective,
      :func:`measurement_utility`). A caller may state the utility it expects; if it disagrees the outcome is
      refused (``utility_mismatch``).
    * **Origin.** An objective outcome supervises only if the objective's *recorded* origin is the operator or the
      verifier (:meth:`register_objective`, from ``TaskSpec.origin``). Unknown origin -> refused
      (``objective_origin_unknown``); a self-generated goal -> ``self_authored_objective``. The caller's
      ``objective_origin`` can only exclude, never admit.
    * **Refusals are logged.** Every refusal goes to :attr:`rejected` (reason, key) and :attr:`rejected_details`; a
      refusal of a *validly signed / authenticated* outcome (wrong utility, unknown or self origin, crediting
      decisions of another objective) is also appended to the chain as an entry of kind ``"refused"`` (utility 0,
      no decisions credited, reason in ``note``), so attempted manipulation stays on the tamper-evident record.
    * **Anchor.** With a ``path`` the head is also written to ``<path>.head`` after every append. On load the file
      must reach that head: a file truncated at an entry boundary (or one without its anchor) loads, but
      :meth:`verify_chain` fails (``anchor_error``), nothing more can be appended and ``MetaLearner.meta_update``
      refuses to run. ``expected_head`` (e.g. from a checkpoint) adds the same requirement for a head kept
      elsewhere — the only protection against someone who rewrites both files."""

    GENESIS = "0" * 64

    def __init__(self, check_signature: Callable[[VerifiedOutcome], bool],
                 check_measurement: Optional[Callable[[OutcomeObservation], bool]] = None,
                 path: Optional[str] = None, *, expected_head: Optional[str] = None):
        self._check_signature, self._check_measurement = check_signature, check_measurement
        self.path = path
        self._entries: List[LedgerEntry] = []
        self._keys: Dict[str, int] = {}
        self._by_exp: Dict[str, List[int]] = {}
        self._level: Dict[str, float] = {}                    # objective -> admitted achievement level so far
        self._origins: Dict[str, str] = {}                    # objective -> recorded origin
        self._refused: Dict[Tuple[str, str], int] = {}        # (reason, key) -> index of the "refused" entry
        self.origin_conflicts: List[Tuple[str, str, str]] = []
        self.rejected: List[Tuple[str, str]] = []            # (reason, key)
        self.rejected_details: List[Dict[str, Any]] = []
        self.anchor_error = ""
        if path and os.path.exists(path):
            self._load(path)
        if path:
            self._check_anchor(path)
        if expected_head and expected_head != self.GENESIS and expected_head not in {e.hash for e in self._entries}:
            self.anchor_error = self.anchor_error or "expected_head_missing"

    @property
    def check_signature(self) -> Callable[[VerifiedOutcome], bool]:
        """The verifier's signature check this ledger admits with (injected; never learned)."""
        return self._check_signature

    # ------------------------------------------------------------------------------------------- objective origin
    def register_objective(self, task: Any, origin: Optional[str] = None) -> Optional[str]:
        """Record where an objective comes from. ``task`` is a ``TaskSpec`` (its ``origin`` is used) or an
        objective id (then ``origin`` must be given: an explicit operator registration). Returns the recorded
        origin (``None``: nothing known, the objective stays excluded). Once an objective was recorded as
        self-generated it stays excluded whatever is registered later."""
        if isinstance(task, str):
            oid = task
            if origin is None:
                raise ValueError("registering an objective id needs an explicit origin (e.g. Trust.OPERATOR)")
        else:
            oid = str(task.objective_id)
            origin = origin if origin is not None else getattr(task, "origin", None)
        if not oid or not origin:
            return self._origins.get(oid)
        origin = str(origin)
        old = self._origins.get(oid)
        if old is not None and old != origin:
            self.origin_conflicts.append((oid, old, origin))
            if Trust.SELF in (old, origin):
                origin = Trust.SELF
            elif old in SUPERVISING_ORIGINS and origin not in SUPERVISING_ORIGINS:
                pass                                          # the more restrictive one wins
            else:
                origin = old
        self._origins[oid] = origin
        return origin

    def origin_of(self, objective_id: str) -> Optional[str]:
        return self._origins.get(objective_id)

    def origins(self) -> Dict[str, str]:
        return dict(self._origins)

    def load_origins(self, origins: Mapping[str, str]) -> None:
        for oid, o in origins.items():
            self.register_objective(oid, origin=o)

    # ------------------------------------------------------------------------------------------- admission
    def _write(self, line: Dict[str, Any], head: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp = self.path + ".head.tmp"
        with open(tmp, "w") as f:
            json.dump({"n": len(self._entries), "head": head}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path + ".head")

    def _chain(self, key: str, kind: str, objective_id: str, group: str, utility: float, observed_at: float,
               signature: str, content: str, exp_ids: Sequence[str], note: str = "") -> LedgerEntry:
        if self.anchor_error:
            raise IntegrityError(f"the outcome ledger failed verification ({self.anchor_error}); nothing is appended")
        body = {"index": len(self._entries), "prev": self.head, "key": key, "kind": kind,
                "objective_id": objective_id, "group": group or key, "utility": float(utility),
                "observed_at": float(observed_at), "signature": signature, "content_sha256": content,
                "exp_ids": list(exp_ids)}
        if note:
            body["note"] = note
        e = LedgerEntry(**{**body, "exp_ids": tuple(exp_ids)}, hash=_entry_hash(body))
        self._entries.append(e)
        if self.path:
            self._write({**body, "hash": e.hash}, e.hash)
        return e

    def _append(self, key: str, kind: str, objective_id: str, group: str, utility: float, observed_at: float,
                signature: str, content: str, exp_ids: Sequence[str]) -> AdmitResult:
        if key in self._keys:
            self._reject("duplicate", key)
            return AdmitResult(False, "duplicate", self._entries[self._keys[key]])
        e = self._chain(key, kind, objective_id, group, utility, observed_at, signature, content, exp_ids)
        self._index(e)
        return AdmitResult(True, "", e)

    def _index(self, e: LedgerEntry) -> None:
        if e.kind == "refused":
            try:
                reason = json.loads(e.note).get("reason", "")
            except ValueError:
                reason = ""
            self._refused[(reason, e.key.split(":", 1)[1].rsplit(":", 1)[0])] = e.index
        if not e.admitted:
            return
        self._keys[e.key] = e.index
        for x in e.exp_ids:
            self._by_exp.setdefault(x, []).append(e.index)
        if e.kind == "objective":
            self._level[e.objective_id] = self._level.get(e.objective_id, 0.0) + e.utility

    def _reject(self, reason: str, key: str, **detail: Any) -> None:
        self.rejected.append((reason, key))
        self.rejected_details.append({"reason": reason, "key": key, **detail})

    def refuse(self, reason: str, key: str, *, kind: str, objective_id: str, observed_at: float, signature: str,
               content: str, claimed: Optional[float] = None, derived: Optional[float] = None,
               exp_ids: Sequence[str] = ()) -> AdmitResult:
        """Refuse an authenticated outcome and keep the refusal on the chain (kind ``"refused"``, utility 0,
        nothing credited). Used by the admission methods and by ``MetaLearner`` for checks it owns."""
        detail = {"refused_kind": kind, "objective_id": objective_id, "claimed_utility": claimed,
                  "derived_utility": derived, "claimed_exp_ids": list(exp_ids)}
        self._reject(reason, key, **detail)
        if (reason, key) in self._refused:                    # the same refusal is chained once
            return AdmitResult(False, reason, self._entries[self._refused[(reason, key)]])
        note = json.dumps({"reason": reason, **detail}, sort_keys=True)
        e = self._chain(f"refused:{key}:{len(self._entries)}", "refused", objective_id, "", 0.0, observed_at,
                        signature, content, (), note=note)
        self._index(e)
        return AdmitResult(False, reason, e)

    def derived_objective_utility(self, v: VerifiedOutcome) -> float:
        """The utility an admission of ``v`` would store: level of ``v`` minus the level admitted so far."""
        return objective_utility(v) - self._level.get(v.objective_id, 0.0)

    def objective_key(self, v: VerifiedOutcome) -> str:
        return "obj:" + hashlib.sha256(json.dumps(
            [v.objective_id, v.status, v.value, sorted(v.evidence_refs) or v.evaluated_at]).encode()).hexdigest()[:24]

    def admit_verified(self, v: VerifiedOutcome, *, utility: Optional[float] = None, group: str = "",
                       exp_ids: Sequence[str] = (), objective_origin: str = Trust.OPERATOR) -> AdmitResult:
        """Admit a signed objective outcome of an objective with a recorded operator / verifier origin. The stored
        utility is :meth:`derived_objective_utility`; ``utility`` (optional) is the caller's expectation and must
        agree. ``objective_origin`` other than OPERATOR / VERIFIER excludes the outcome (it cannot admit one)."""
        key = self.objective_key(v)
        caller_excludes = objective_origin not in SUPERVISING_ORIGINS
        ok, reason = screen_verified(v, self._check_signature,
                                     objective_origin=objective_origin if caller_excludes else Trust.OPERATOR)
        if not ok:
            self._reject(reason, key, objective_id=v.objective_id)
            return AdmitResult(False, reason)
        refuse = dict(kind="objective", objective_id=v.objective_id, observed_at=v.evaluated_at,
                      signature=v.signature, content=v.content_hash(), claimed=utility, exp_ids=exp_ids)
        origin = self._origins.get(v.objective_id)
        if origin is None:
            return self.refuse("objective_origin_unknown", key, **refuse)
        if origin == Trust.SELF:
            return self.refuse("self_authored_objective", key, **refuse)
        if origin not in SUPERVISING_ORIGINS:
            return self.refuse(f"objective_origin_{origin}", key, **refuse)
        if key in self._keys:
            self._reject("duplicate", key, objective_id=v.objective_id)
            return AdmitResult(False, "duplicate", self._entries[self._keys[key]])
        derived = self.derived_objective_utility(v)
        if utility is not None and not utility_matches(utility, derived):
            return self.refuse("utility_mismatch", key, **{**refuse, "derived": derived})
        return self._append(key, "objective", v.objective_id, group, derived, v.evaluated_at, v.signature,
                            v.content_hash(), exp_ids)

    def admit_measurement(self, o: OutcomeObservation, *, objective_id: str = "", utility: Optional[float] = None,
                          group: str = "", exp_ids: Sequence[str] = (), imagined: bool = False,
                          signature: str = "") -> AdmitResult:
        """Admit a verifier measurement (``OutcomeVerifier.measure``) authenticated by ``check_measurement``. The
        stored utility is :func:`measurement_utility`; a disagreeing ``utility`` is refused."""
        key = "obs:" + _obs_key(o)
        ok, reason = screen_observation(o, imagined=imagined or bool(getattr(o, "imagined", False)),
                                        check_measurement=self._check_measurement)
        if not ok:
            self._reject(reason, key, objective_id=objective_id)
            return AdmitResult(False, reason)
        if key in self._keys:
            self._reject("duplicate", key, objective_id=objective_id)
            return AdmitResult(False, "duplicate", self._entries[self._keys[key]])
        derived = measurement_utility(o)
        if utility is not None and not utility_matches(utility, derived):
            return self.refuse("utility_mismatch", key, kind="measurement", objective_id=objective_id,
                               observed_at=o.observed_at, signature=signature or getattr(o, "signature", ""),
                               content=o.content_hash(), claimed=utility, derived=derived, exp_ids=exp_ids)
        return self._append(key, "measurement", objective_id, group, derived, o.observed_at,
                            signature or getattr(o, "signature", ""), o.content_hash(), exp_ids)

    # ------------------------------------------------------------------------------------------- reading
    @property
    def head(self) -> str:
        return self._entries[-1].hash if self._entries else self.GENESIS

    def entries(self) -> List[LedgerEntry]:
        return list(self._entries)

    def admitted(self) -> List[LedgerEntry]:
        return [e for e in self._entries if e.admitted]

    def __len__(self) -> int:
        return len(self._entries)

    def verify_chain(self) -> Tuple[bool, int]:
        """(ok, index of the first inconsistent entry or -1). A missing or unmatched anchor (truncated file) or a
        missing expected head fails with the index of the first entry that should be there."""
        prev = self.GENESIS
        for i, e in enumerate(self._entries):
            if e.index != i or e.prev != prev or _entry_hash(e.body()) != e.hash:
                return False, i
            prev = e.hash
        if self.anchor_error:
            return False, len(self._entries)
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
        ok, bad = self.verify_chain()
        if not ok:
            raise IntegrityError(f"outcome ledger {path} fails its hash chain at entry {bad}")
        for e in self._entries:
            self._index(e)

    def _check_anchor(self, path: str) -> None:
        anchor = path + ".head"
        if not os.path.exists(anchor):
            if self._entries:
                self.anchor_error = "missing_anchor"
            return
        try:
            with open(anchor) as f:
                a = json.load(f)
            n, head = int(a["n"]), str(a["head"])
        except (OSError, ValueError, KeyError, TypeError):
            self.anchor_error = "unreadable_anchor"
            return
        if n == 0 and head == self.GENESIS:
            return
        if len(self._entries) < n:
            self.anchor_error = f"truncated:{len(self._entries)}<{n}"
        elif self._entries[n - 1].hash != head:
            self.anchor_error = "anchor_mismatch"
        # a file longer than its anchor (crash between the two writes) is accepted: the chain covers the rest

    def confirm_head(self, head: str) -> bool:
        """Adopt a ledger file written before anchors existed: if the only problem is a missing anchor and ``head``
        (kept elsewhere, e.g. the checkpoint's) is in the chain, the anchor is written and the ledger verifies.
        A truncated or mismatched anchor is never cleared. Returns whether the ledger verifies afterwards."""
        if self.anchor_error == "missing_anchor" and (head == self.GENESIS or head in {e.hash for e in self._entries}):
            self.anchor_error = ""
            if self.path and self._entries:
                tmp = self.path + ".head.tmp"
                with open(tmp, "w") as f:
                    json.dump({"n": len(self._entries), "head": self.head}, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path + ".head")
        return self.verify_chain()[0]

    def by_exp(self, exp_id: str) -> List[LedgerEntry]:
        """Admitted entries credited to an experience record."""
        return [self._entries[i] for i in self._by_exp.get(exp_id, ())]

    def has_observation(self, o: OutcomeObservation) -> bool:
        """Was this measurement admitted (authenticated and not a duplicate)?"""
        return ("obs:" + _obs_key(o)) in self._keys

    def independent_groups(self, since_index: int = 0) -> int:
        return len({e.group for e in self._entries[since_index:] if e.admitted})


# ========================================================================================== self-generated goals
#: number words folded to "#" like digits ("ten members" == "10 members")
_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion dozen "
    "first second third fourth fifth tenth hundredth".split())
#: generic attainment verbs: "reach / get / obtain / achieve ... N members" all state the same goal
_ATTAIN = frozenset("reach get got obtain achiev attain gain hit acquir earn secur grow grew accumulat collect "
                    "arriv land score win won".split())
_STOP = frozenset("a an the to of at in on for by with and or our my your its this that these those some least"
                  .split())


def _stem(w: str) -> str:
    for suf, keep in (("ing", 5), ("ed", 4), ("es", 4), ("s", 3), ("e", 4)):
        if w.endswith(suf) and len(w) > keep:
            return w[: -len(suf)]
    return w


class SelfGoalDiscount:
    """Weight in [0, 1] for anything earned on a self-generated goal. Repeats of the same goal decay
    geometrically; goals whose success was already predicted with near certainty (nothing to achieve) get no
    weight. Outcomes of self-generated goals never enter the outer objective at all (``OutcomeLedger``).

    *Same goal* means any of: equal canonical text (:meth:`key`: case, punctuation, digits and number words,
    plural / tense endings, articles and generic attainment verbs folded — "Reach 10 members", "get ten members"
    and "obtain 10 members" are one goal), or, when ``embed`` is given (text -> embedding vector, e.g. the
    backbone's semantic embedding of the goal text), a cosine similarity of at least ``similar_above`` with an
    earlier goal: semantic near-duplicates that share no words still count as repeats. The count of a new goal is
    the number of earlier registered goals it is the same as."""

    def __init__(self, decay: float = 0.5, trivial_above: float = 0.9, *,
                 embed: Optional[Callable[[str], Any]] = None, similar_above: float = 0.9, max_remembered: int = 4096):
        self.decay, self.trivial_above = float(decay), float(trivial_above)
        self.embed, self.similar_above, self.max_remembered = embed, float(similar_above), int(max_remembered)
        self.counts: Dict[str, int] = {}
        self._vecs: List[Tuple[str, Tensor]] = []          # (key, unit embedding) of registered goals, in order

    @staticmethod
    def key(goal_text: str) -> str:
        """Same goal up to case, whitespace, punctuation, numbers (digits and number words), word endings,
        articles and generic attainment verbs."""
        t = re.sub(r"\d+([.,]\d+)?", " # ", goal_text.casefold())
        words = []
        for w in re.sub(r"[^a-z# ]+", " ", t).split():
            w = "#" if w in _NUMBER_WORDS else w
            if w in _STOP:
                continue
            st = _stem(w)
            if st in _ATTAIN or w in _ATTAIN:
                continue
            if st == "#" and words and words[-1] == "#":
                continue                                     # "twenty five" -> one number
            words.append(st)
        return hashlib.sha256(" ".join(words).encode()).hexdigest()[:16]

    def _unit(self, goal_text: str) -> Optional[Tensor]:
        if self.embed is None:
            return None
        v = torch.as_tensor(self.embed(goal_text)).detach().to("cpu", torch.float64).reshape(-1)
        n = float(v.norm())
        return v / n if n > 0 and n == n else None

    def repeats(self, goal_text: str) -> int:
        """How many earlier registered goals this one repeats (lexically or semantically)."""
        k = self.key(goal_text)
        n_lex = self.counts.get(k, 0)
        u = self._unit(goal_text)
        if u is None or not self._vecs:
            return n_lex
        keys = [kk for kk, _ in self._vecs]
        sims = torch.stack([v for _, v in self._vecs]) @ u
        n_sem = sum(1 for kk, s in zip(keys, sims.tolist()) if s >= self.similar_above and kk != k)
        return n_lex + n_sem

    def weight(self, goal_text: str, p_success_before: float, *, register: bool = True) -> float:
        n = self.repeats(goal_text)
        if register:
            k = self.key(goal_text)
            self.counts[k] = self.counts.get(k, 0) + 1
            u = self._unit(goal_text)
            if u is not None:
                self._vecs.append((k, u))
                del self._vecs[:-self.max_remembered]
        p = min(max(float(p_success_before), 0.0), 1.0)
        difficulty = min(1.0, max(0.0, (1.0 - p) / max(1e-9, 1.0 - self.trivial_above))) if p > self.trivial_above \
            else 1.0
        return (self.decay ** n) * difficulty

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "trivial_above": self.trivial_above, "counts": dict(self.counts),
                "similar_above": self.similar_above,
                "vecs": [(k, v.tolist()) for k, v in self._vecs]}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        self.decay, self.trivial_above = float(d["decay"]), float(d["trivial_above"])
        self.counts = dict(d["counts"])
        self.similar_above = float(d.get("similar_above", self.similar_above))
        self._vecs = [(k, torch.tensor(v, dtype=torch.float64)) for k, v in d.get("vecs", [])]


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
                   max_tail_ratio: float = 0.2, flat_rel_drop_per_step: float = 0.01) -> Dict[str, Any]:
    """Audit a reward stream paid on one context: if epistemic uncertainty has (nearly) stopped falling over the
    last ``tail`` fraction of the stream — a relative drop below ``flat_rel_drop_per_step`` per step — the mean
    |reward| there must be at most ``max_tail_ratio`` of the mean |reward| at the start. ``passed`` is False when
    reward keeps being paid for randomness that no longer teaches anything."""
    n = len(rewards)
    if n != len(epistemic) or n < 8:
        raise ValueError("need aligned streams of at least 8 steps")
    k = max(2, int(n * tail))
    head = sum(abs(x) for x in rewards[:k]) / k
    tail_r = sum(abs(x) for x in rewards[-k:]) / k
    e0, e1 = epistemic[-k], epistemic[-1]
    rate = 0.0 if e0 <= 0 else (e0 - e1) / e0 / (k - 1)
    flat = rate < flat_rel_drop_per_step
    ratio = tail_r / head if head > 0 else (0.0 if tail_r == 0 else float("inf"))
    return {"epistemic_flat_in_tail": flat, "epistemic_rel_drop_per_step": rate, "head_reward": head,
            "tail_reward": tail_r, "tail_ratio": ratio, "passed": (not flat) or ratio <= max_tail_ratio}
