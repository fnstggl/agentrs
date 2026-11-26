"""cam_live.harness.failures — execution-failure taxonomy, bounded retries and failure observations (spec 12A.12).

Owned by workstream ES.

* :func:`classify` maps *observable evidence* (an exception, an HTTP status, a timeout, a stale observation id, a
  verification mismatch, an authorization verdict, page-structure hashes) to one ``contracts.FailureClass``.
* :class:`RetryPolicy` bounds retries for the transient classes and refuses, by construction, to carry a rule for
  the ``FailureClass.HARD`` classes: a denial or a platform restriction is never retried.
* :func:`failure_observation` turns a classified failure into the ``Observation`` the learner ingests.

There is deliberately no recovery decision tree here. Beyond "wait and repeat the identical operation a bounded
number of times for a transient fault", what to do after a failure is the learner's decision.
"""
from __future__ import annotations

import errno as _errno
import random
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..contracts import (ActionResult, ActionSpec, AuthorizationDecision, AuthVerdict, Clock, FailureClass, Level,
                         Modality, Observation, OutcomeKind, PrimitiveOp, Provenance, Quantity, ResultStatus, Trust,
                         new_id)

#: obstacle categories of the autonomous-obstacle section of the specification (``ObstacleRecord.category``)
CATEGORY = {
    FailureClass.NONE: "",
    FailureClass.INVALID_ARGUMENTS: "technical",
    FailureClass.STALE_OBSERVATION: "technical",
    FailureClass.VISUAL_GROUNDING: "technical",
    FailureClass.PAGE_CHANGED: "technical",
    FailureClass.TRANSIENT_NETWORK: "technical",
    FailureClass.WRONG_SITE_ASSUMPTION: "informational",
    FailureClass.MISSING_PERMISSION: "permission",
    FailureClass.PLATFORM_RESTRICTION: "permission",
    FailureClass.RESOURCE_LIMIT: "resource",
    FailureClass.STRATEGIC: "strategic",
    FailureClass.UNKNOWN: "informational",
}

_HTTP = {
    400: FailureClass.INVALID_ARGUMENTS, 422: FailureClass.INVALID_ARGUMENTS,
    401: FailureClass.MISSING_PERMISSION, 403: FailureClass.MISSING_PERMISSION, 407: FailureClass.MISSING_PERMISSION,
    451: FailureClass.PLATFORM_RESTRICTION,
    404: FailureClass.WRONG_SITE_ASSUMPTION, 405: FailureClass.WRONG_SITE_ASSUMPTION,
    406: FailureClass.WRONG_SITE_ASSUMPTION, 409: FailureClass.WRONG_SITE_ASSUMPTION,
    410: FailureClass.WRONG_SITE_ASSUMPTION, 415: FailureClass.WRONG_SITE_ASSUMPTION,
    501: FailureClass.WRONG_SITE_ASSUMPTION,
    408: FailureClass.TRANSIENT_NETWORK, 425: FailureClass.TRANSIENT_NETWORK, 500: FailureClass.TRANSIENT_NETWORK,
    502: FailureClass.TRANSIENT_NETWORK, 503: FailureClass.TRANSIENT_NETWORK, 504: FailureClass.TRANSIENT_NETWORK,
    429: FailureClass.RESOURCE_LIMIT, 402: FailureClass.RESOURCE_LIMIT, 413: FailureClass.RESOURCE_LIMIT,
    507: FailureClass.RESOURCE_LIMIT, 509: FailureClass.RESOURCE_LIMIT,
}
_TIMEOUT_NAMES = ("Timeout",)
_NETWORK_NAMES = ("ConnectError", "ConnectionError", "NetworkError", "RemoteProtocolError", "ReadError",
                  "WriteError", "ProtocolError", "gaierror", "IncompleteRead", "TargetClosedError")
_STALE_NAMES = ("Stale",)
_LIMIT_NAMES = ("MemoryError", "RecursionError", "ResourceExhausted", "QuotaExceeded", "LimitExceeded")
_ARG_NAMES = ("ValueError", "TypeError", "KeyError", "ValidationError", "JSONDecodeError", "IndexError")
_NET_ERRNO = {_errno.ECONNRESET, _errno.ECONNREFUSED, _errno.ECONNABORTED, _errno.EHOSTUNREACH, _errno.ENETUNREACH,
              _errno.ENETDOWN, _errno.EPIPE, _errno.ETIMEDOUT}
_LIMIT_ERRNO = {_errno.ENOSPC, _errno.EDQUOT, _errno.EMFILE, _errno.ENOMEM}


@dataclass
class FailureEvidence:
    """Everything observable about one failed (or surprising) execution. Unknown fields stay at their defaults."""
    exception: Optional[BaseException] = None
    http_status: Optional[int] = None
    retry_after_s: Optional[float] = None
    timed_out: bool = False
    arg_errors: Sequence[str] = ()
    stale_ref: bool = False                 # an element / observation reference was rejected as stale
    obs_id_expected: str = ""               # observation id the action was built against
    obs_id_current: str = ""                # observation id current at execution time
    structure_hash_expected: str = ""       # page / response structure the caller relied on
    structure_hash_actual: str = ""         # structure actually found
    target_missing: bool = False            # the addressed element / endpoint / field was not there
    verified: Optional[bool] = None         # ActionSpec.verify evaluated on the resulting state
    executed_ok: bool = False               # every primitive operation itself reported success
    outcome_mismatch: bool = False          # executed and verified, yet the expected consequence did not follow
    auth_verdict: str = ""                  # AuthVerdict.* when the authorization layer answered
    auth_reasons: Sequence[str] = ()
    platform_rule: bool = False             # an explicit platform rule was stated (policy id, error code, notice)
    limit_exceeded: bool = False            # a budget, quota, deadline or rate limit was reported exhausted
    grounding_confidence: Optional[float] = None
    grounding_hit: Optional[bool] = None    # did the proposed coordinates hit the intended target
    modality: str = ""
    level: str = Level.PRIMITIVE
    detail: str = ""


@dataclass
class FailureReport:
    failure_class: str
    category: str
    hard: bool
    retryable: bool
    signals: List[str] = field(default_factory=list)
    candidates: Dict[str, float] = field(default_factory=dict)   # class -> normalised weight of the evidence
    detail: str = ""
    http_status: Optional[int] = None
    retry_after_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"failure_class": self.failure_class, "category": self.category, "hard": self.hard,
                "retryable": self.retryable, "signals": list(self.signals), "candidates": dict(self.candidates),
                "detail": self.detail, "http_status": self.http_status, "retry_after_s": self.retry_after_s}


def _names(exc: BaseException) -> List[str]:
    return [c.__name__ for c in type(exc).__mro__]


def _has(names: Sequence[str], needles: Sequence[str]) -> bool:
    return any(n in name for name in names for n in needles)


def classify(ev: FailureEvidence) -> FailureReport:
    """Classify a failure from evidence. Every matching rule votes; the strongest vote is the class and the
    normalised votes are returned as ``candidates`` so the learner sees the ambiguity instead of a false certainty.

    A policy denial or an explicit platform rule always wins: no other evidence may reinterpret a refusal as
    something that could be retried or worked around.
    """
    votes: Dict[str, float] = {}
    signals: List[str] = []

    def vote(cls: str, weight: float, signal: str) -> None:
        votes[cls] = votes.get(cls, 0.0) + weight
        signals.append(signal)

    # ---- authorization and platform rules (dominant) --------------------------------------------------------
    if ev.auth_verdict == AuthVerdict.DENY:
        vote(FailureClass.PLATFORM_RESTRICTION if ev.platform_rule else FailureClass.MISSING_PERMISSION, 1000.0,
             "authorization_denied")
    elif ev.auth_verdict == AuthVerdict.NEEDS_APPROVAL:
        vote(FailureClass.MISSING_PERMISSION, 1000.0, "authorization_needs_approval")
    if ev.platform_rule and ev.auth_verdict != AuthVerdict.DENY:
        vote(FailureClass.PLATFORM_RESTRICTION, 500.0, "platform_rule_stated")
    exc_names = _names(ev.exception) if ev.exception is not None else []
    if "PermissionError" in exc_names:
        vote(FailureClass.MISSING_PERMISSION, 300.0, "exception_permission")

    # ---- HTTP status ----------------------------------------------------------------------------------------
    if ev.http_status is not None and ev.http_status >= 400:
        cls = _HTTP.get(ev.http_status)
        if cls is None:
            cls = FailureClass.TRANSIENT_NETWORK if ev.http_status >= 500 else FailureClass.UNKNOWN
        hard = cls in FailureClass.HARD
        vote(cls, 300.0 if hard else 6.0, f"http_{ev.http_status}")

    # ---- invalid arguments ----------------------------------------------------------------------------------
    if ev.arg_errors:
        vote(FailureClass.INVALID_ARGUMENTS, 8.0, "argument_validation")
    if exc_names and _has(exc_names, _ARG_NAMES) and not _has(exc_names, _TIMEOUT_NAMES + _NETWORK_NAMES):
        vote(FailureClass.INVALID_ARGUMENTS, 3.0, "exception_argument")

    # ---- staleness and changed structure --------------------------------------------------------------------
    stale = ev.stale_ref or (bool(ev.obs_id_expected) and bool(ev.obs_id_current)
                             and ev.obs_id_expected != ev.obs_id_current) or _has(exc_names, _STALE_NAMES)
    structure_changed = (bool(ev.structure_hash_expected) and bool(ev.structure_hash_actual)
                         and ev.structure_hash_expected != ev.structure_hash_actual)
    if structure_changed:
        vote(FailureClass.PAGE_CHANGED, 7.0, "structure_hash_changed")
    if stale:
        vote(FailureClass.STALE_OBSERVATION, 5.0 if structure_changed else 7.5, "stale_observation")

    # ---- visual grounding -----------------------------------------------------------------------------------
    if ev.grounding_hit is False:
        vote(FailureClass.VISUAL_GROUNDING, 7.0, "grounding_missed_target")
    elif ev.modality == Modality.VISUAL and ev.verified is False and not structure_changed:
        vote(FailureClass.VISUAL_GROUNDING, 4.0, "visual_action_not_verified")
    if ev.grounding_confidence is not None and ev.grounding_confidence < 0.5 and ev.verified is not True:
        vote(FailureClass.VISUAL_GROUNDING, 2.0, "grounding_low_confidence")

    # ---- network and timeouts -------------------------------------------------------------------------------
    os_errno = getattr(ev.exception, "errno", None) if ev.exception is not None else None
    if ev.timed_out or _has(exc_names, _TIMEOUT_NAMES):
        vote(FailureClass.TRANSIENT_NETWORK, 6.0, "timeout")
    if _has(exc_names, _NETWORK_NAMES) or os_errno in _NET_ERRNO:
        vote(FailureClass.TRANSIENT_NETWORK, 6.0, "network_exception")

    # ---- resource limits ------------------------------------------------------------------------------------
    if ev.limit_exceeded:
        vote(FailureClass.RESOURCE_LIMIT, 7.0, "limit_exceeded")
    if _has(exc_names, _LIMIT_NAMES) or os_errno in _LIMIT_ERRNO:
        vote(FailureClass.RESOURCE_LIMIT, 6.0, "exception_resource")

    # ---- wrong assumption about the site / mismatch after a clean execution ---------------------------------
    if ev.target_missing and not structure_changed and not stale:
        vote(FailureClass.WRONG_SITE_ASSUMPTION, 6.0, "target_missing")
    elif ev.target_missing:
        vote(FailureClass.WRONG_SITE_ASSUMPTION, 1.5, "target_missing")
    if ev.verified is False and ev.modality != Modality.VISUAL:
        vote(FailureClass.PAGE_CHANGED if structure_changed else FailureClass.WRONG_SITE_ASSUMPTION, 4.0,
             "verification_mismatch")
    if ev.outcome_mismatch and ev.executed_ok and ev.verified is not False:
        vote(FailureClass.STRATEGIC, 6.0, "outcome_mismatch_after_clean_execution")

    if not votes:
        vote(FailureClass.UNKNOWN, 1.0, "no_recognised_evidence")
    primary = max(votes.items(), key=lambda kv: (kv[1], -FailureClass.ALL.index(kv[0])))[0]
    total = sum(votes.values())
    hard = primary in FailureClass.HARD
    retryable = (not hard) and primary in RetryPolicy.TRANSIENT and not any(c in FailureClass.HARD for c in votes)
    detail = ev.detail or (f"{type(ev.exception).__name__}: {ev.exception}" if ev.exception is not None else "")
    if not detail and ev.arg_errors:
        detail = "; ".join(ev.arg_errors)
    if not detail and ev.auth_reasons:
        detail = "; ".join(ev.auth_reasons)
    return FailureReport(failure_class=primary, category=CATEGORY[primary], hard=hard, retryable=retryable,
                         signals=signals, candidates={k: round(v / total, 6) for k, v in votes.items()},
                         detail=detail[:600], http_status=ev.http_status, retry_after_s=ev.retry_after_s)


# --------------------------------------------------------------------------------------------------------------
# Bounded retries
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RetryRule:
    max_retries: int
    base_delay_s: float = 0.0
    factor: float = 2.0
    max_delay_s: float = 30.0
    jitter: float = 0.1                 # fraction of the delay, drawn from the policy's seeded generator
    needs_retry_after: bool = False     # retry only when the other side said how long to wait


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    delay_s: float = 0.0
    reason: str = ""


class RetryPolicy:
    """Bounded retries for transient failure classes. HARD classes cannot be given a rule (``ValueError``)."""

    TRANSIENT = (FailureClass.TRANSIENT_NETWORK, FailureClass.RESOURCE_LIMIT, FailureClass.STALE_OBSERVATION)
    MAX_RETRIES_CEILING = 5
    DEFAULT_RULES = {
        FailureClass.TRANSIENT_NETWORK: RetryRule(max_retries=3, base_delay_s=0.5, max_delay_s=30.0),
        FailureClass.RESOURCE_LIMIT: RetryRule(max_retries=2, max_delay_s=60.0, jitter=0.0, needs_retry_after=True),
        FailureClass.STALE_OBSERVATION: RetryRule(max_retries=1, jitter=0.0),
    }

    def __init__(self, rules: Optional[Dict[str, RetryRule]] = None, *, seed: int = 0,
                 max_total_wait_s: float = 120.0):
        rules = dict(self.DEFAULT_RULES if rules is None else rules)
        for cls, rule in rules.items():
            if cls not in FailureClass.ALL:
                raise ValueError(f"unknown failure class {cls!r}")
            if cls in FailureClass.HARD:
                raise ValueError(f"{cls} is a hard failure class and can never be retried")
            if cls not in self.TRANSIENT:
                raise ValueError(f"{cls} is not a transient failure class; what to do next is the learner's decision")
            if not 0 <= rule.max_retries <= self.MAX_RETRIES_CEILING:
                raise ValueError(f"max_retries for {cls} must be within 0..{self.MAX_RETRIES_CEILING}")
        self.rules, self.max_total_wait_s = rules, float(max_total_wait_s)
        self._rng = random.Random(seed)

    def decide(self, failure_class: str, retries_done: int, *, retry_after_s: Optional[float] = None,
               waited_s: float = 0.0, hard_evidence: bool = False) -> RetryDecision:
        if failure_class in FailureClass.HARD or hard_evidence:
            return RetryDecision(False, 0.0, "hard failure class: never retried")
        rule = self.rules.get(failure_class)
        if rule is None:
            return RetryDecision(False, 0.0, "not a transient failure class")
        if retries_done >= rule.max_retries:
            return RetryDecision(False, 0.0, f"retry bound reached ({rule.max_retries})")
        if retry_after_s is not None and retry_after_s >= 0:
            delay = float(retry_after_s)
            if delay > rule.max_delay_s:
                return RetryDecision(False, delay, f"stated wait {delay:.1f}s exceeds the bound {rule.max_delay_s:.1f}s")
        elif rule.needs_retry_after:
            return RetryDecision(False, 0.0, "no stated wait for a limit: not retried blindly")
        else:
            delay = min(rule.max_delay_s, rule.base_delay_s * (rule.factor ** retries_done))
            delay *= 1.0 + rule.jitter * self._rng.random()
        if waited_s + delay > self.max_total_wait_s:
            return RetryDecision(False, delay, "total wait bound reached")
        return RetryDecision(True, delay, "transient failure: bounded retry")


def wait(clock: Clock, seconds: float, sleep: Optional[Callable[[float], None]] = None) -> None:
    """Wait on the given clock: an injected ``sleep``, a simulated clock's ``advance`` or real time."""
    if seconds <= 0:
        return
    if sleep is not None:
        sleep(seconds)
    elif hasattr(clock, "advance"):
        clock.advance(seconds)
    else:
        _time.sleep(seconds)


def failure_of(result: ActionResult) -> Dict[str, Any]:
    """The structured failure payload attached to a failed result by :func:`failed_result` ({} if none)."""
    for o in result.observations:
        if o.kind == "error" and isinstance(o.data.get("failure"), dict):
            return o.data["failure"]
    return {}


def run_with_retries(attempt: Callable[[int], ActionResult], policy: RetryPolicy, clock: Clock, *,
                     sleep: Optional[Callable[[float], None]] = None, deadline: float = 0.0,
                     cancel: Any = None) -> ActionResult:
    """Call ``attempt(retries_done)`` until it succeeds, fails non-transiently or the bounds are reached.

    DENIED / NEEDS_APPROVAL results and HARD failure classes return immediately. The returned result carries the
    number of retries and the primitive operations of every attempt.
    """
    retries, waited, prior_ops = 0, 0.0, []
    while True:
        res = attempt(retries)
        res.ops = prior_ops + list(res.ops)
        res.retries = retries
        if res.status == ResultStatus.OK or res.status in (ResultStatus.DENIED, ResultStatus.NEEDS_APPROVAL,
                                                           ResultStatus.CANCELLED):
            return res
        info = failure_of(res)
        d = policy.decide(res.failure_class, retries, retry_after_s=info.get("retry_after_s"), waited_s=waited,
                          hard_evidence=any(c in FailureClass.HARD for c in info.get("candidates", {})))
        if not d.retry:
            return res
        if deadline and clock.now() + d.delay_s > deadline:
            return res
        if cancel is not None and cancel.is_set():
            res.status = ResultStatus.CANCELLED
            return res
        wait(clock, d.delay_s, sleep)
        waited += d.delay_s
        retries += 1
        prior_ops = list(res.ops)


# --------------------------------------------------------------------------------------------------------------
# Results and observations
# --------------------------------------------------------------------------------------------------------------
def failure_observation(report: FailureReport, action: ActionSpec, clock: Clock, *, retries: int = 0,
                        elapsed_s: float = 0.0, source: str = "harness.failures") -> Observation:
    """The structured observation of a failure. Its text is a bounded, untrusted digest; the exact numbers travel
    as quantities; the class, category, signals and competing explanations are in ``data["failure"]``."""
    q = [Quantity("exec.failure.retries", float(retries), kind=OutcomeKind.COUNT),
         Quantity("exec.failure.elapsed_s", float(elapsed_s), unit="s")]
    if report.http_status is not None:
        q.append(Quantity("exec.failure.http_status", float(report.http_status)))
    if report.retry_after_s is not None:
        q.append(Quantity("exec.failure.retry_after_s", float(report.retry_after_s), unit="s"))
    text = (f"execution failure [{report.failure_class}; {report.category}] in {action.operation} "
            f"({action.modality}): {report.detail}")[:800]
    return Observation(obs_id=new_id("obs"), ts=clock.now(), kind="error", text=text, quantities=q,
                       provenance=Provenance(source=source, uri=action.target, retrieved_at=clock.now(),
                                             trust=Trust.UNTRUSTED),
                       parent_action_id=action.action_id,
                       data={"failure": report.to_dict(), "operation": action.operation,
                             "modality": action.modality, "level": action.level})


def failed_result(action: ActionSpec, report: FailureReport, clock: Clock, started: float, *,
                  ops: Optional[List[PrimitiveOp]] = None, observations: Optional[List[Observation]] = None,
                  retries: int = 0, status: str = "") -> ActionResult:
    """Build the ``ActionResult`` of a classified failure, with the failure observation appended."""
    now = clock.now()
    if not status:
        # DENIED is reserved for a refusal by the authorization layer (see authorization_result); a refusal by the
        # remote side is a FAILED execution whose failure class is still HARD.
        status = ResultStatus.FAILED
        if "timeout" in report.signals and report.failure_class == FailureClass.TRANSIENT_NETWORK:
            status = ResultStatus.TIMEOUT
    obs = list(observations or [])
    obs.append(failure_observation(report, action, clock, retries=retries, elapsed_s=now - started))
    res = ActionResult(action_id=action.action_id, status=status, failure_class=report.failure_class,
                       failure_detail=report.detail, started=started, ended=now, observations=obs,
                       ops=list(ops or []), retries=retries)
    res.cost.seconds = now - started
    res.cost.primitive_ops = len(res.ops)
    return res


def authorization_result(action: ActionSpec, decision: AuthorizationDecision, clock: Clock, started: float, *,
                         ops: Optional[List[PrimitiveOp]] = None) -> Optional[ActionResult]:
    """``None`` when the primitive operation is allowed; otherwise the DENIED / NEEDS_APPROVAL result. Nothing is
    retried and no other modality is tried here: the refusal is returned as it is."""
    verdict = getattr(decision, "verdict", AuthVerdict.DENY)
    if verdict == AuthVerdict.ALLOW:
        return None
    reasons = list(getattr(decision, "reasons", []) or [])
    policy_ids = list(getattr(decision, "policy_ids", []) or [])
    platform = any(str(p).startswith("platform") for p in policy_ids)
    report = classify(FailureEvidence(auth_verdict=verdict, auth_reasons=reasons, platform_rule=platform,
                                      modality=action.modality, level=action.level,
                                      detail="; ".join(reasons) or f"authorization verdict: {verdict}"))
    status = ResultStatus.NEEDS_APPROVAL if verdict == AuthVerdict.NEEDS_APPROVAL else ResultStatus.DENIED
    for op in ops or []:
        if op.status == ResultStatus.OK and not op.ended:
            op.status, op.failure_class, op.ended = status, report.failure_class, clock.now()
    res = failed_result(action, report, clock, started, ops=ops, status=status)
    if verdict == AuthVerdict.NEEDS_APPROVAL:
        res.failure_class = FailureClass.NONE
        res.artifacts["approval_id"] = getattr(decision, "approval_id", "") or ""
    return res
