"""cam_live.contracts — the shared, versioned data and interface contracts of CAM Live.

MASTER-OWNED. Workstreams import from here and must not edit this file; request a change in
``cam_live/docs/change_requests/<workstream>.md`` instead.

Everything that crosses a workstream boundary (records written to disk, messages between the learner, the
computer environment, the authorization service and the independent verifier, and the method signatures those
services expose) is defined here. Tensor-level contracts live in ``cam_live.nn.api``.

Design rules that every implementation must respect:

* **Time** is UNIX seconds (float). Nothing calls ``time.time()`` directly; everything takes a :class:`Clock`.
* **Exact evidence** travels as :class:`Quantity` objects next to text. Numbers are never re-derived from an
  embedding.
* **Environment content is data.** ``Observation.text`` and tool output are untrusted (``Trust.UNTRUSTED``) and can
  never change the objective, permissions or execution boundary.
* **Predictions are not outcomes.** A :class:`PredictionRecord` is written before the outcome exists; an
  :class:`OutcomeObservation` only ever comes from the environment or the verifier and says which
  (``verified``); imagined values carry ``imagined=True`` and are not accepted by outcome APIs.
* **The environment gives no reward.** There is no scalar reward field anywhere in the environment interface.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import time as _time
import typing
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

CONTRACT_VERSION = "1.1.0"              # 1.1: additive fields / optional protocol members (change requests, APPLIED.md)
EXEC_INTERFACE_VERSION = "1.0"          # learner <-> computer environment execution interface


# --------------------------------------------------------------------------------------------------------------
# Vocabularies (plain strings so records stay JSON-native)
# --------------------------------------------------------------------------------------------------------------
class Trust:
    UNTRUSTED = "untrusted"      # anything read from the world: pages, documents, messages, tool output
    OPERATOR = "operator"        # task configuration and operator commands
    VERIFIER = "verifier"        # the independent outcome verifier
    SELF = "self"                # produced by CAM (predictions, drafts, imagined values)


class Modality:
    STRUCTURED_BROWSER = "structured_browser"
    VISUAL = "visual"
    CODE = "code"
    API = "api"
    SKILL = "skill"
    INTERNAL = "internal"        # wait / think / inspect own records: no external effect
    ALL = (STRUCTURED_BROWSER, VISUAL, CODE, API, SKILL, INTERNAL)


class Level:
    PRIMITIVE = "primitive"          # one browser/computer operation
    PROCEDURE = "procedure"          # a multi-operation skill or program with one intermediate purpose
    INTERVENTION = "intervention"    # a decision meant to change something in the external world
    ALL = (PRIMITIVE, PROCEDURE, INTERVENTION)


#: ``ActionSpec.operation`` of a multi-step program: its content is ``steps``, executed in order, every step
#: authorized individually (a program never has more permission than its steps issued one by one) [B-CR-1]
PROGRAM_OPERATION = "program"


class ResultStatus:
    OK = "ok"
    FAILED = "failed"
    DENIED = "denied"                    # the authorization layer refused; never retried
    NEEDS_APPROVAL = "needs_approval"    # queued for the operator
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class FailureClass:
    NONE = "none"
    INVALID_ARGUMENTS = "invalid_arguments"
    STALE_OBSERVATION = "stale_observation"
    VISUAL_GROUNDING = "visual_grounding"
    PAGE_CHANGED = "page_changed"
    TRANSIENT_NETWORK = "transient_network"
    WRONG_SITE_ASSUMPTION = "wrong_site_assumption"
    MISSING_PERMISSION = "missing_permission"
    PLATFORM_RESTRICTION = "platform_restriction"
    RESOURCE_LIMIT = "resource_limit"
    STRATEGIC = "strategic"              # executed fine, the world did not respond as assumed
    UNKNOWN = "unknown"
    ALL = (NONE, INVALID_ARGUMENTS, STALE_OBSERVATION, VISUAL_GROUNDING, PAGE_CHANGED, TRANSIENT_NETWORK,
           WRONG_SITE_ASSUMPTION, MISSING_PERMISSION, PLATFORM_RESTRICTION, RESOURCE_LIMIT, STRATEGIC, UNKNOWN)
    #: classes that must never be retried or worked around by switching modality
    HARD = (MISSING_PERMISSION, PLATFORM_RESTRICTION)


class OutcomeKind:
    REAL = "real"                    # continuous quantity
    COUNT = "count"                  # non-negative integer over an exposure
    BINARY = "binary"
    TIME_TO_EVENT = "time_to_event"  # may be right-censored
    CATEGORICAL = "categorical"
    ALL = (REAL, COUNT, BINARY, TIME_TO_EVENT, CATEGORICAL)


class OutcomeStatus:
    PENDING = "pending"
    OBSERVED = "observed"
    CENSORED = "censored"            # horizon ended without the event (time-to-event) — a real observation
    UNOBSERVABLE = "unobservable"    # the measurement could not be taken; never used as a training target


class ItemKind:
    """Kinds of experience token (spec Part 3.1)."""
    OBSERVATION = "observation"; ACTION = "action"; GOAL = "goal"; DOCUMENT = "document"; SOFTWARE = "software"
    ENTITY = "entity"; RELATION = "relation"; EVENT = "event"; TIME = "time"; COST = "cost"
    RESOURCE = "resource"; OUTCOME = "outcome"; UNCERTAINTY = "uncertainty"; TOOL = "tool"
    ATTEMPT = "attempt"; PREDICTION = "prediction"; FAILURE = "failure"
    ALL = (OBSERVATION, ACTION, GOAL, DOCUMENT, SOFTWARE, ENTITY, RELATION, EVENT, TIME, COST, RESOURCE,
           OUTCOME, UNCERTAINTY, TOOL, ATTEMPT, PREDICTION, FAILURE)


class AuthVerdict:
    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"


class GoalStatus:
    PENDING = "pending"
    ACHIEVED = "achieved"
    FAILED = "failed"


# --------------------------------------------------------------------------------------------------------------
# Record base: JSON round-trip for nested dataclasses
# --------------------------------------------------------------------------------------------------------------
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _from(tp: Any, v: Any) -> Any:
    if v is None:
        return None
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return _from(args[0], v) if len(args) == 1 else v
    if origin in (list, List, typing.Sequence):
        (a,) = typing.get_args(tp) or (Any,)
        return [_from(a, x) for x in v]
    if origin in (tuple, Tuple):
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_from(args[0], x) for x in v)
        return tuple(v)
    if origin in (dict, Dict):
        args = typing.get_args(tp)
        if len(args) == 2:
            return {k: _from(args[1], x) for k, x in v.items()}
        return dict(v)
    if isinstance(tp, type) and dataclasses.is_dataclass(tp) and isinstance(v, dict):
        return tp.from_dict(v)
    return v


@functools.lru_cache(maxsize=None)
def _hints(cls) -> Tuple[Dict[str, Any], frozenset]:
    """Resolved type hints and field names of a record class, computed once per class [D-5]."""
    return typing.get_type_hints(cls), frozenset(f.name for f in dataclasses.fields(cls))


@dataclass
class Record:
    """Base for every contract record: ``to_dict`` / ``from_dict`` / ``to_json`` with nested dataclasses."""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        hints, known = _hints(cls)
        return cls(**{k: _from(hints.get(k, Any), v) for k, v in d.items() if k in known})

    def content_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


# --------------------------------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------------------------------
@dataclass
class Quantity(Record):
    """One exact measured number. ``name`` is a free-form dotted metric key chosen by whoever measured it."""
    name: str
    value: float
    unit: str = ""
    kind: str = OutcomeKind.REAL
    exposure: float = 1.0            # for counts: the exposure (seconds, trials, visitors) the count refers to


@dataclass
class Provenance(Record):
    source: str = ""                 # adapter / tool / service that produced it
    uri: str = ""                    # URL, file path or record id
    retrieved_at: float = 0.0
    trust: str = Trust.UNTRUSTED
    content_sha256: str = ""
    imagined: bool = False           # True only for model-generated (never observed) content


@dataclass
class Observation(Record):
    obs_id: str
    ts: float
    kind: str                        # "page" | "tool_result" | "event" | "metric" | "document" | "image" | "error" | ...
    text: str = ""                   # bounded, untrusted digest of what was observed
    quantities: List[Quantity] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    refs: Dict[str, str] = field(default_factory=dict)       # artifact / blob / element references
    provenance: Provenance = field(default_factory=Provenance)
    parent_action_id: str = ""       # the action whose execution produced it, if any
    data: Dict[str, Any] = field(default_factory=dict)       # structured payload (e.g. a11y tree digest, diff)


@dataclass
class Cost(Record):
    """Resources consumed or estimated. All fields are exact when measured, estimates when predicted."""
    seconds: float = 0.0
    money: float = 0.0               # in the task's currency
    compute_s: float = 0.0           # GPU/CPU seconds spent on inference for the operation
    primitive_ops: int = 0
    tokens: int = 0


@dataclass
class ExperienceItem(Record):
    """One token source for the experiential core (spec 3.1). Built from records by ``cam_live.nn.tokens``."""
    kind: str                        # ItemKind.*
    text: str = ""
    ts: float = 0.0
    quantities: List[Quantity] = field(default_factory=list)
    ref: str = ""                    # id of the record it came from
    trust: str = Trust.UNTRUSTED


# --------------------------------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------------------------------
@dataclass
class VerificationCondition(Record):
    """How execution success is checked from the *resulting world state* (never assumed).

    ``spec`` keys by modality and kind, as the executors evaluate them (anything else yields ``verified=None``,
    never ``True``) [EC-3]:

    ============  ===============  =================================================================================
    visual        element_state    ``{"color_at": [x, y], "rgb": [r, g, b], "tolerance": 60}`` or
                                   ``{"changed": bool, "region": [x, y, w, h], "min_fraction": f}``
    visual        url_matches      ``{"pattern": regex}`` (only when the screen provider has ``url()``)
    code          text_present     ``{"text": str, "stream": "stdout" | "stderr"}``
    code          program          ``{"command": shell text, "expect_exit": 0}`` (run in the sandbox afterwards)
    ============  ===============  =================================================================================
    """
    description: str = ""
    kind: str = "none"               # "none" | "text_present" | "url_matches" | "element_state" | "metric" | "program"
    spec: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionSpec(Record):
    """A typed, compositional executable action (spec 12A.6). ``steps`` makes it a program."""
    action_id: str
    operation: str                   # tool name from discover_tools(), e.g. "browser.click", "sandbox.python", "skill.run"
    modality: str = Modality.INTERNAL
    level: str = Level.PRIMITIVE
    target: str = ""                 # resource acted on: URL, element ref, file, channel, skill id
    arguments: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    expected: str = ""               # expected observable result, in words
    est_cost: Cost = field(default_factory=Cost)
    auth_scope: str = ""             # permission scope the action claims to need
    timeout_s: float = 60.0
    verify: VerificationCondition = field(default_factory=VerificationCondition)
    parent_decision_id: str = ""
    experiment_id: str = ""
    steps: List["ActionSpec"] = field(default_factory=list)
    provenance: str = ""             # candidate generator: "backbone" | "skill_library" | "mutation" | "builtin" | ...
    rationale: str = ""              # short generator-side note; not a faithful trace of neural computation

    def render(self) -> str:
        """Canonical text used for semantic encoding and logs (no secrets ever appear in arguments)."""
        args = json.dumps(self.arguments, sort_keys=True, default=str)
        head = f"{self.level}:{self.modality}:{self.operation}({args}) -> {self.target}"
        if self.steps:
            head += " {" + "; ".join(s.render() for s in self.steps) + "}"
        if self.expected:
            head += f" expect: {self.expected}"
        return head


@dataclass
class ToolDescriptor(Record):
    name: str                        # operation name
    modality: str
    description: str
    arg_schema: Dict[str, Any] = field(default_factory=dict)     # JSON Schema of ``arguments``
    level: str = Level.PRIMITIVE
    auth_scope: str = ""
    side_effects: str = "none"       # "none" | "local" | "external_read" | "external_write"
    est_cost: Cost = field(default_factory=Cost)
    available: bool = True
    unavailable_reason: str = ""     # e.g. "blocked: no Discord credentials provided"


@dataclass
class PrimitiveOp(Record):
    """One executed low-level operation inside an action (kept for procedural learning and audit)."""
    op: str
    modality: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    started: float = 0.0
    ended: float = 0.0
    status: str = ResultStatus.OK
    failure_class: str = FailureClass.NONE
    obs_before: str = ""
    obs_after: str = ""
    note: str = ""
    step_id: str = ""                # action_id of the ActionSpec.steps entry that issued this operation ("" = top) [D-4]


@dataclass
class AuthorizationDecision(Record):
    """Issued only by the independent authorization service. ``grant`` is an opaque signed token bound to the
    action's content hash; the executor verifies it and refuses anything else."""
    verdict: str                     # AuthVerdict.*
    action_hash: str = ""
    reasons: List[str] = field(default_factory=list)
    policy_ids: List[str] = field(default_factory=list)
    grant: str = ""
    expires_at: float = 0.0
    approval_id: str = ""            # set when verdict == NEEDS_APPROVAL


@dataclass
class ActionResult(Record):
    action_id: str
    status: str                      # ResultStatus.*
    failure_class: str = FailureClass.NONE
    failure_detail: str = ""
    started: float = 0.0
    ended: float = 0.0
    observations: List[Observation] = field(default_factory=list)
    ops: List[PrimitiveOp] = field(default_factory=list)
    cost: Cost = field(default_factory=Cost)
    verified: Optional[bool] = None  # result of ActionSpec.verify against the resulting state; None = not checked
    artifacts: Dict[str, str] = field(default_factory=dict)      # name -> reference
    expected_outcomes: List["OutcomeQuery"] = field(default_factory=list)   # measurements that will mature later
    retries: int = 0


# --------------------------------------------------------------------------------------------------------------
# Outcomes and predictions
# --------------------------------------------------------------------------------------------------------------
@dataclass
class OutcomeQuery(Record):
    """A question about a future observable quantity: *what will ``metric`` be ``horizon_s`` after ``issued_at``?*"""
    query_id: str
    metric: str                      # free-form metric key; described by ``description`` for semantic encoding
    kind: str = OutcomeKind.REAL
    horizon_s: float = 0.0
    issued_at: float = 0.0
    description: str = ""
    decision_id: str = ""
    experiment_id: str = ""
    exposure: float = 1.0
    categories: List[str] = field(default_factory=list)
    goal_relevant: bool = False

    @property
    def due_at(self) -> float:
        return self.issued_at + self.horizon_s


@dataclass
class OutcomeObservation(Record):
    query_id: str
    status: str                      # OutcomeStatus.*
    value: Optional[float] = None    # category index for CATEGORICAL; elapsed seconds for TIME_TO_EVENT
    observed_at: float = 0.0
    verified: bool = False           # True only when produced by the independent verifier
    source: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    uncertainty: float = 0.0         # measurement standard error where known
    dedup_key: str = ""              # two observations with the same key are the same event
    signature: str = ""              # verifier signature over the record with signature="" (measure() only) [C-1, F-5]


@dataclass
class Distribution(Record):
    """A predictive distribution. ``family`` in {"normal","student_t","neg_binomial","poisson","bernoulli",
    "discrete_hazard","categorical"}; ``params`` holds plain floats / lists."""
    family: str
    params: Dict[str, Any] = field(default_factory=dict)
    mean: Optional[float] = None
    std: Optional[float] = None
    epistemic_std: Optional[float] = None    # the part of ``std`` attributed to reducible uncertainty


@dataclass
class PredictionRecord(Record):
    """Registered *before* the outcome is known. ``resolved_*`` fields are filled once, later, by reconciliation."""
    prediction_id: str
    query: OutcomeQuery
    dist: Distribution
    made_at: float
    decision_id: str = ""
    action_id: str = ""
    model_version: str = ""
    checkpoint_id: str = ""
    imagined_horizon_steps: int = 0
    registered_at: Optional[float] = None    # set by the prediction registry; outcomes must be strictly later [B-CR-3]
    resolved_at: Optional[float] = None
    resolved_value: Optional[float] = None
    resolved_status: str = OutcomeStatus.PENDING
    nll: Optional[float] = None


@dataclass
class CandidateEvaluation(Record):
    action: ActionSpec
    predictions: List[PredictionRecord] = field(default_factory=list)
    p_success: Optional[float] = None        # goal-conditioned success probability if this action is taken
    value: Optional[float] = None            # policy's action value / logit
    voi: Optional[float] = None              # decision-relevant value of information
    cost: Cost = field(default_factory=Cost)
    uncertainty: Optional[float] = None
    feasible: bool = True
    infeasible_reason: str = ""
    extras: Dict[str, float] = field(default_factory=dict)


@dataclass
class Decision(Record):
    decision_id: str
    ts: float
    objective_id: str
    chosen: ActionSpec
    candidates: List[CandidateEvaluation] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)     # behaviour-policy probability of each candidate
    chosen_index: int = 0
    exploration: str = ""            # predict.policy.RULES: "policy" | "thompson" | "ucb" | "greedy" | "voi" | "randomized" [B-CR-4]
    rationale: str = ""              # concise and faithful: built from the numbers above, not invented
    context_obs_ids: List[str] = field(default_factory=list)
    model_version: str = ""
    checkpoint_id: str = ""
    reward_version: str = ""
    experiment_id: str = ""
    level: str = Level.INTERVENTION
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def propensity(self) -> float:
        return self.probabilities[self.chosen_index] if self.probabilities else 1.0


@dataclass
class ExperimentRecord(Record):
    experiment_id: str
    created_at: float
    question: str
    hypothesis: str
    evidence_refs: List[str] = field(default_factory=list)
    intervention: str = ""
    comparison: str = ""             # comparison condition, "" if none
    arms: List[str] = field(default_factory=list)
    assignment: str = "none"         # "randomized" | "sequential" | "none"
    arm_probs: List[float] = field(default_factory=list)               # assignment probability of each arm [B-CR-2]
    assignments: List[Dict[str, Any]] = field(default_factory=list)    # [{"unit_id", "arm", "probability", "ts"}]
    test: Dict[str, Any] = field(default_factory=dict)                 # the test used for ``supported``
    predicted: List[PredictionRecord] = field(default_factory=list)
    cost: Cost = field(default_factory=Cost)
    horizon_s: float = 0.0
    status: str = "registered"       # registered -> running -> matured -> closed
    actual: List[OutcomeObservation] = field(default_factory=list)
    measurement_uncertainty: str = ""
    supported: Optional[bool] = None
    alternative_explanations: List[str] = field(default_factory=list)
    next_action: str = ""
    decision_ids: List[str] = field(default_factory=list)


@dataclass
class VerifiedOutcome(Record):
    """The independent verifier's statement about an objective. The learner cannot construct a valid one: the
    verifier signs it (``signature``) and the runtime checks the signature before any use as supervision."""
    objective_id: str
    status: str                      # GoalStatus.*
    value: float = 0.0               # e.g. verified member count
    target: float = 0.0
    uncertainty: float = 0.0
    evaluated_at: float = 0.0
    evidence_refs: List[str] = field(default_factory=list)
    excluded: Dict[str, int] = field(default_factory=dict)       # e.g. {"bots": 3, "test_accounts": 2}
    notes: str = ""
    secondary: List[Quantity] = field(default_factory=list)
    signature: str = ""
    origin: str = ""                 # TaskSpec.origin of the verified task, covered by the signature [C-12] ("" = not
                                     # stated by the verifier)


# --------------------------------------------------------------------------------------------------------------
# Learning records
# --------------------------------------------------------------------------------------------------------------
@dataclass
class ExperienceRecord(Record):
    """The unit of replay: one consequential decision with everything known about it so far. Low-level traces
    stay in ``result.ops``; credit is assigned at this (decision) level so an outcome is counted once."""
    exp_id: str
    objective_id: str
    decision: Decision
    result: Optional[ActionResult] = None
    context_item_refs: List[str] = field(default_factory=list)   # ids of the items that formed H_t
    post_item_refs: List[str] = field(default_factory=list)      # H_{t+1}: context after the result was observed
    goal_text: str = ""              # objective text at decision time (for goal encoding during replay)
    outcomes: List[OutcomeObservation] = field(default_factory=list)
    pending_queries: List[OutcomeQuery] = field(default_factory=list)
    queries: List[OutcomeQuery] = field(default_factory=list)    # every question asked about this decision [B-CR-6]
    intrinsic_reward: Optional[float] = None
    reward_version: str = ""
    bootstrap_reward: Optional[float] = None
    reward_inputs: Dict[str, Any] = field(default_factory=dict)  # RewardSignals.to_dict() + {"life_step": int} [C-4]
    credit: Dict[str, float] = field(default_factory=dict)       # estimator name -> redistributed credit
    credit_uncertainty: Dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0
    eligible: Dict[str, bool] = field(default_factory=dict)      # loss name -> may be trained on (no entry = eligible) [D-2]
    window_id: str = ""              # meta-learning window (support/query bookkeeping)
    objective_family: str = ""
    provenance: str = "live"         # "live" | "staging" | "diagnostic" — never mixed silently
    seq: int = 0                     # monotone position in the lifetime
    lifetime: int = 0                # index of the lifetime (0 for the single live lifetime) [C-7]


@dataclass
class LearningUpdateRecord(Record):
    update_id: str
    ts: float
    trigger: str                     # what new evidence triggered learning
    timescale: str                   # "fast" | "frequent" | "slow_adapters" | "meta_reward"
    losses: Dict[str, float] = field(default_factory=dict)
    param_groups: List[str] = field(default_factory=list)
    grad_norms: Dict[str, float] = field(default_factory=dict)
    weight_delta: Dict[str, float] = field(default_factory=dict)     # group -> L2 norm of the change
    hash_before: Dict[str, str] = field(default_factory=dict)
    hash_after: Dict[str, str] = field(default_factory=dict)
    n_experiences: int = 0
    exp_ids: List[str] = field(default_factory=list)
    intrinsic_reward_stats: Dict[str, float] = field(default_factory=dict)
    credit_changes: Dict[str, float] = field(default_factory=dict)
    prediction_change: Dict[str, float] = field(default_factory=dict)  # probe-set predictions before -> after
    checkpoint_id: str = ""
    model_version: str = ""
    notes: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)   # loss shares, balance scale, strata, flags [D-11]


@dataclass
class MetaUpdateRecord(Record):
    meta_id: str
    ts: float
    reward_version_before: str
    reward_version_after: str
    support_window: str = ""
    query_window: str = ""
    n_support: int = 0
    n_query: int = 0
    n_independent_outcomes: int = 0  # independent verified outcomes supervising this update (not row count)
    inner_steps: int = 0
    inner_lr: float = 0.0
    outer_objective: float = 0.0
    meta_grad_norm: float = 0.0
    estimator: str = ""              # "on_policy_query" | "importance_weighted" | ...
    max_importance_weight: float = 1.0
    bootstrap_mix: float = 1.0       # 1.0 = pure bootstrap reward in use, 0.0 = pure learned reward
    applied: bool = False
    skipped_reason: str = ""
    diagnostics: Dict[str, float] = field(default_factory=dict)
    ledger_head: str = ""            # reward.integrity.OutcomeLedger.head when the update was computed [C-3]
    window_pair: str = ""            # reward.windows.WindowPair.pair_id


@dataclass
class ObstacleRecord(Record):
    obstacle_id: str
    ts: float
    action_id: str
    failure_class: str
    description: str
    explanations: List[Dict[str, Any]] = field(default_factory=list)   # [{"hypothesis": str, "p": float}, ...]
    category: str = ""               # technical | informational | strategic | resource | permission
    alternatives: List[str] = field(default_factory=list)             # action ids considered
    chosen_alternative: str = ""
    outcome: str = ""
    strategy_changed: bool = False
    learning_update_ids: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------------------------
# Task configuration (loaded from tasks/*.yaml). Contains NO solution strategy.
# --------------------------------------------------------------------------------------------------------------
@dataclass
class Limits(Record):
    duration_s: float = 0.0          # 0 = must be set explicitly before a live run
    money: float = 0.0
    compute_s: float = 0.0
    max_external_actions: int = 0
    max_decisions: int = 0


@dataclass
class TaskSpec(Record):
    objective_id: str
    objective: str                   # the semantic objective given to CAM, verbatim
    success: Dict[str, Any] = field(default_factory=dict)        # verifier configuration (independent of CAM)
    family: str = ""                 # objective family label for replay balance / transfer bookkeeping
    resources: Dict[str, Any] = field(default_factory=dict)      # authorized accounts / destinations / sites
    interfaces: List[str] = field(default_factory=list)          # permitted environment interfaces (modalities/tools)
    limits: Limits = field(default_factory=Limits)
    measurement: Dict[str, Any] = field(default_factory=dict)    # outcome measurement protocol
    data_permissions: Dict[str, bool] = field(default_factory=dict)
    stop_conditions: List[str] = field(default_factory=list)
    secondary_metrics: List[str] = field(default_factory=list)
    mode: str = "staging"            # "diagnostic" | "staging" | "live"
    origin: str = Trust.OPERATOR     # Trust.OPERATOR | Trust.SELF (goal-free mode); set by the runtime only [C-2]

    FORBIDDEN_KEYS = ("strategy", "plan", "steps", "niche", "schedule", "playbook", "actions")


# --------------------------------------------------------------------------------------------------------------
# Service interfaces
# --------------------------------------------------------------------------------------------------------------
@runtime_checkable
class Clock(Protocol):
    def now(self) -> float: ...


class WallClock:
    def now(self) -> float:
        return _time.time()


class SimClock:
    """Deterministic clock for diagnostics and staging; delayed outcomes can mature in milliseconds."""

    def __init__(self, start: float = 1_800_000_000.0):
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> float:
        if dt < 0:
            raise ValueError("time does not go backwards")
        self._t += float(dt)
        return self._t


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """The versioned environment contract (spec 12.1). Observations and consequences only — no reward."""
    interface_version: str

    def observe(self) -> List[Observation]: ...
    def discover_tools(self) -> List[ToolDescriptor]: ...
    def validate_action(self, action: ActionSpec) -> List[str]: ...            # [] = structurally valid
    def execute(self, action: ActionSpec, authorization: AuthorizationDecision) -> ActionResult: ...
    def poll_events(self, cursor: str) -> Tuple[List[Observation], str]: ...   # (events, next cursor)
    def get_resource_state(self) -> Dict[str, Any]: ...                         # budgets consumed / remaining
    def get_pending_outcomes(self) -> List[OutcomeQuery]: ...
    def get_observable_metrics(self) -> List[Quantity]: ...
    def get_capabilities(self) -> Dict[str, Any]: ...
    # ``get_capabilities()["features"]["at_most_once_by_action_id"] = True`` declares that executing an action id a
    # second time returns the recorded result instead of acting again; the runtime re-issues an action that was in
    # flight at a crash only when this is declared [F-4]. ``"provenance"`` is "diagnostic" | "staging" | "live".

    # outcome maturation: environment-side measurements for due queries (unverified unless stated)
    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]: ...


@runtime_checkable
class ModalityExecutor(Protocol):
    """One interchangeable execution modality inside the computer environment (browser, visual, code, api, skill)."""
    modality: str

    def tools(self) -> List[ToolDescriptor]: ...
    def validate(self, action: ActionSpec) -> List[str]: ...
    def execute(self, action: ActionSpec, ctx: "ExecContext") -> ActionResult: ...
    def close(self) -> None: ...


@runtime_checkable
class SecretHandle(Protocol):
    """Executor-side secret handle placed in ``ExecContext.secrets`` (``harness.auth.vault.ExecutorSecrets``):
    releases a secret only for a granted action that is being executed, and scrubs every known secret value from
    text that is about to become an observation. Optional: ``has(name) -> bool`` (whether a non-revoked secret of
    that name exists; releases nothing) and ``bind(action, decision)`` returning a handle with
    ``get(name, destination)`` [EB-2, EC-2, ES-2]."""

    def get(self, name: str, action: "ActionSpec", decision: "AuthorizationDecision", destination: str) -> str: ...
    def scrub(self, text: str) -> str: ...


@dataclass
class ExecContext:
    """What an executor gets besides the action. ``check`` MUST be called for every primitive operation that
    touches anything outside the executor's own process (navigation, request, file write, publish, spend)."""
    clock: Clock
    check: Any                       # Callable[[PrimitiveOp, ActionSpec], AuthorizationDecision] — PolicyEngine.check_primitive
    obs_sink: Any                    # Callable[[Observation], None]
    run_dir: str = ""
    deadline: float = 0.0
    cancel: Any = None               # threading.Event-like; executors poll it
    secrets: Any = None              # SecretHandle (executor side only; never the learner)
    authorization: Any = None        # AuthorizationDecision of the top-level action being executed [EA-1, EB-1, ES-1]
    granted_action: Any = None       # the top-level ActionSpec whose grant covers the operations being executed [ES-1]
    dispatch: Any = None             # Callable[[ActionSpec], ActionResult]: run a sub-step through the environment [EB-1]
    workspace: str = ""              # persistent workspace directory of the environment [EB-1]


@runtime_checkable
class PolicyEngine(Protocol):
    """The independent authorization layer (spec 12.6 / 12A.14). Runs outside the learner; the learned policy and
    reward cannot change its configuration or its audit log."""

    def authorize(self, action: ActionSpec, task: TaskSpec, context: Dict[str, Any]) -> AuthorizationDecision: ...
    # check_grant: called exactly once, by the executor, immediately before execution: verifies binding / expiry and
    # USES UP the grant (implementations may offer consume=False to only look) [EA-2]
    def check_grant(self, action: ActionSpec, decision: AuthorizationDecision) -> bool: ...
    def check_primitive(self, op: PrimitiveOp, action: ActionSpec) -> AuthorizationDecision: ...
    # record: audit + limit accounting. Called exactly once per executed action by the computer environment /
    # executor after execution (``ComputerEnvironment(record_results=True)``); the runtime does not call it [EB-3, F-3]
    def record(self, action: ActionSpec, result: ActionResult) -> None: ...
    def approvals(self) -> List[Dict[str, Any]]: ...
    def resolve_approval(self, approval_id: str, approve: bool, by: str, note: str = "") -> AuthorizationDecision: ...
    def emergency_stop(self, reason: str, by: str) -> None: ...
    def stopped(self) -> bool: ...
    def revoke(self, scope: str, by: str) -> None: ...
    def limits_state(self) -> Dict[str, Any]: ...


@runtime_checkable
class OutcomeVerifier(Protocol):
    """Independent objective verification (spec 13.2). Not a model; reads permitted evidence directly."""

    def verify(self, task: TaskSpec) -> VerifiedOutcome: ...
    def check_signature(self, outcome: VerifiedOutcome) -> bool: ...
    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]: ...   # verified=True measurements
    # check_measurement: True only for an observation this verifier produced and signed (``OutcomeObservation.
    # signature`` over the record with signature=""). Callers must use getattr: verifiers written before contract
    # 1.1 do not have it, and their measurements then never supervise the outer (meta) objective [C-1, F-5]
    def check_measurement(self, obs: OutcomeObservation) -> bool: ...


@runtime_checkable
class ExperienceStore(Protocol):
    """Durable experience storage (spec 10.1). Implemented by ``cam_live.credit.replay``."""

    def append(self, rec: ExperienceRecord) -> None: ...
    def update(self, rec: ExperienceRecord) -> None: ...
    def get(self, exp_id: str) -> Optional[ExperienceRecord]: ...
    def by_decision(self, decision_id: str) -> Optional[ExperienceRecord]: ...
    def put_item(self, item: ExperienceItem) -> None: ...
    def items(self, refs: Sequence[str]) -> List[ExperienceItem]: ...
    def pending(self, now: float) -> List[Tuple[ExperienceRecord, OutcomeQuery]]: ...
    def sample(self, n: int, *, loss: str, strata: Optional[Dict[str, float]] = None,
               seed: Optional[int] = None, provenance: Optional[Sequence[str]] = None) -> List[ExperienceRecord]: ...
    def window(self, window_id: str) -> List[ExperienceRecord]: ...
    def count(self) -> int: ...
    def flush(self) -> None: ...
    # [D-1] outcomes with de-duplication (result has .accepted / .duplicate), the earliest due pending query, and
    # the replay-sampler state for checkpoints (experience itself is never rolled back)
    def add_outcome(self, obs: OutcomeObservation, exp_id: str = "") -> Any: ...
    def next_due(self) -> Optional[float]: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state: Dict[str, Any]) -> Dict[str, Any]: ...
    # [C-5] the records of a lifetime in order
    def iter_records(self, *, since_seq: int = 0, objective_id: str = "") -> Iterator[ExperienceRecord]: ...
    # [D-10] a closed objective and its verified outcome (``verified`` = the verifier's signature was checked)
    def put_objective_outcome(self, objective_id: str, *, status: str, value: float = 0.0, evaluated_at: float = 0.0,
                              verified: bool = False, body: Optional[Dict[str, Any]] = None) -> None: ...
    def objective_outcome(self, objective_id: str) -> Optional[Dict[str, Any]]: ...


@runtime_checkable
class AgentAPI(Protocol):
    """What the persistent runtime calls on the learner; one method per neural step of the loop (spec 15.1)."""

    def set_objective(self, task: TaskSpec) -> None: ...
    def ingest(self, observations: Sequence[Observation]) -> None: ...                       # steps 2-3
    def reconcile(self, outcomes: Sequence[OutcomeObservation],
                  verified: Optional[VerifiedOutcome] = None) -> Dict[str, Any]: ...        # step 4
    def update_beliefs(self) -> Dict[str, Any]: ...                                          # step 5
    def propose(self, tools: Sequence[ToolDescriptor], k: int) -> List[ActionSpec]: ...      # step 6
    def evaluate(self, candidates: Sequence[ActionSpec]) -> List[CandidateEvaluation]: ...   # steps 7-8
    def select(self, evaluations: Sequence[CandidateEvaluation]) -> Decision: ...            # step 9
    # record_result: also called after ``load()`` for decisions journalled after that checkpoint (crash recovery);
    # it must accept a decision it has not seen and must not double-count one it has [F-6]
    def record_result(self, decision: Decision, result: ActionResult) -> None: ...           # step 12
    def internal_rewards(self) -> Dict[str, Any]: ...                                        # step 13
    def learn(self, budget_s: float) -> List[LearningUpdateRecord]: ...                      # step 14
    def meta_learn(self, budget_s: float) -> List[MetaUpdateRecord]: ...                     # step 15
    def save(self, path: str) -> str: ...                                                    # step 16 -> checkpoint id
    # load: restores the complete learner state of a checkpoint written by ``save``. Afterwards the runtime may
    # re-deliver results journalled after it (``record_result``), events after the committed cursor (``ingest``)
    # and outcomes that were pending at the commit (``reconcile``); all of them must be accepted idempotently [F-6]
    def load(self, path: str) -> None: ...
    def next_wake(self) -> Optional[float]: ...                                              # step 17
    def status(self) -> Dict[str, Any]: ...
    # [F-1] experiment records registered or updated since the last call (logged to experiments.jsonl), and the
    # learner's explanations of an obstacle: [{"hypothesis": str, "p": float}, ...]
    def drain_experiments(self) -> List[ExperimentRecord]: ...
    def explain_obstacle(self, obstacle: ObstacleRecord) -> List[Dict[str, Any]]: ...


@runtime_checkable
class VisualGrounder(Protocol):
    """Replaceable visual-grounding model. It only *proposes* coordinates; the harness validates and executes.
    Optional members read by the visual executor [EC-3]: attribute ``heavy: bool`` (needs the shared GPU lock) and
    method ``close()`` (release model memory)."""
    name: str
    revision: str

    def ground(self, image_png: bytes, instruction: str) -> Dict[str, Any]: ...   # {"x": int, "y": int, "confidence": float}
    def info(self) -> Dict[str, Any]: ...                                         # license, params, memory, latency


# --------------------------------------------------------------------------------------------------------------
# Canonical paths inside a run directory (spec 14.4)
# --------------------------------------------------------------------------------------------------------------
class RunFiles:
    EVENTS = "events.jsonl"
    DECISIONS = "decisions.jsonl"
    EXPERIMENTS = "experiments.jsonl"
    PREDICTIONS = "predictions.jsonl"
    LEARNING = "learning_updates.jsonl"
    META = "meta_updates.jsonl"
    OBSTACLES = "obstacles.jsonl"
    JOURNAL = "journal.md"
    CHECKPOINTS = "checkpoints"
    STATE = "state.json"             # runtime state for resume
    CONTROL = "control.json"         # operator commands (pause/resume/stop)
    AUDIT = "audit.jsonl"            # written only by the authorization service
    VERIFIER = "verifier.jsonl"      # written only by the independent verifier
    STORE = "experience.sqlite"
