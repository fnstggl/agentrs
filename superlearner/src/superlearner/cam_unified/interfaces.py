"""CAM Unified shared interface contract (version 1). Owned by the master architect.

Every workstream codes against these types and signatures; none edits this file. Change requests go to the master,
who versions them here (`CONTRACT_VERSION`). Shapes use B = batch of environments/actors, S = n_state latent tokens,
N = context tokens, D = d_model, K = belief particles, Q = outcome queries, T = tools.

Three layers:
1. Records crossing process / environment boundaries (plain dataclasses, JSON-serialisable): Observation, ToolSchema,
   Action, ExecutionResult, EventRecord, DecisionTrace, GoalSpec, IndependentGoalResult.
2. The environment boundary: EnvAdapter (agent side; partially observable) and GoalEvaluator (evaluator side;
   may read hidden state; its results reach the learner only as verified returns).
3. Tensor-side component protocols (TokenSet, OutcomeBatch, FastState and the nn.Module protocols).
"""
from __future__ import annotations

import enum
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Iterator, Optional, Protocol, Sequence, runtime_checkable

CONTRACT_VERSION = 2   # 1.1: TokenSet.entity2; 2: seq cursors, time/money/duration semantics (stage-0 critic)
OBS_SCHEMA_VERSION = 1


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex}"


def sha256_text(s: str | bytes) -> str:
    b = s.encode() if isinstance(s, str) else s
    return hashlib.sha256(b).hexdigest()


class _Rec:
    """JSON round-trip for nested dataclass records (enums by value)."""

    def to_dict(self) -> dict:
        def conv(v):
            if isinstance(v, enum.Enum):
                return v.value
            if isinstance(v, _Rec):
                return v.to_dict()
            if isinstance(v, (list, tuple)):
                return [conv(x) for x in v]
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            return v
        return {f.name: conv(getattr(self, f.name)) for f in fields(self)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), default=str)


# ============================================================================ 1. records

class ObsKind(str, enum.Enum):
    TEXT = "text"                    # natural language (pages, messages, docs, descriptions)
    TOOL_SCHEMA = "tool_schema"      # a tool's description / JSON schema (also delivered via available_tools)
    TOOL_RESULT = "tool_result"      # a tool's structured response
    HTML = "html"
    A11Y = "a11y"                    # accessibility tree / structured browser view
    CODE = "code"                    # source code
    FILE = "file"                    # file contents
    EXEC_RESULT = "exec_result"      # program/build/deploy execution output
    ERROR = "error"                  # error message / failure
    IMAGE = "image"                  # screenshot (artifact ref); optional
    NUMERIC = "numeric"              # one measurement (value + unit + entity + time + source)
    RELATION = "relation"            # observed relationship between two entities
    MESSAGE = "message"              # communication from/to another participant
    TRANSACTION = "transaction"      # economic transaction
    EVENT = "event"                  # delayed or asynchronous world event
    GOAL = "goal"                    # objective statement
    CONSTRAINT = "constraint"        # environmental constraint / rule / budget


class EvidenceKind(str, enum.Enum):
    CONTROLLED = "controlled"        # direct consequence of the agent's own intervention
    OBSERVATIONAL = "observational"  # observed without intervention
    INFERRED = "inferred"            # derived by the agent (never ground truth)


@dataclass
class Observation(_Rec):
    kind: ObsKind
    content: str = ""                # original text/code/html/json (may be truncated; full copy at artifact_ref)
    artifact_ref: Optional[str] = None   # content-addressed ref into the ArtifactStore for exact retrieval
    value: Optional[float] = None    # NUMERIC: the number exactly as observed (no unit conversion by adapters)
    unit: Optional[str] = None       # NUMERIC: as the environment states it, e.g. "cents" (Crucible money is integer
                                     # cents), "cents/core-hour", "ms", "count", "fraction"
    entity: Optional[str] = None     # persistent id of the primary entity the observation is about
    entity_type: Optional[str] = None
    target: Optional[str] = None     # RELATION/MESSAGE/TRANSACTION: the second entity
    relation: Optional[str] = None   # RELATION: observed relation label (free text, may be absent)
    t: float = 0.0                   # simulated time of the observation in DAYS since world start (= seconds / 86400;
                                     # adapters convert exactly; Crucible's native clock is integer seconds)
    seq: Optional[int] = None        # environment event sequence number (cursor), when the environment has one
    source: str = ""                 # evidence source: tool name / page url / sensor / participant id
    evidence: EvidenceKind = EvidenceKind.OBSERVATIONAL
    caused_by: Optional[str] = None  # event_id of the action that produced it, when known (delayed attribution)
    meta: dict = field(default_factory=dict)
    schema_version: int = OBS_SCHEMA_VERSION

    @staticmethod
    def from_dict(d: dict) -> "Observation":
        d = dict(d)
        d["kind"] = ObsKind(d["kind"])
        d["evidence"] = EvidenceKind(d.get("evidence", "observational"))
        return Observation(**d)


@dataclass
class ToolSchema(_Rec):
    name: str                        # environment-given name (may be randomised in tests; never a model input id)
    description: str
    parameters: dict                 # JSON schema of the arguments ({"type":"object","properties":{...},"required":[...]})
    returns: str = ""                # description of the result
    cost_hint: Optional[float] = None
    duration_hint: Optional[float] = None    # simulated days
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return sha256_text(self.name + "\x00" + json.dumps(self.parameters, sort_keys=True))[:16]

    @staticmethod
    def from_dict(d: dict) -> "ToolSchema":
        return ToolSchema(**d)


class ActionSource(str, enum.Enum):
    CAM_DECODER = "cam_decoder"          # CAM's learned policy + latent-conditioned decoder (default)
    CAM_PLANNER = "cam_planner"          # chosen by CAM's imagination/planning among CAM-generated candidates
    FROZEN_PROPOSAL = "frozen_proposal"  # proposed by a frozen semantic model (must be logged with probabilities)
    BASELINE = "baseline"                # an explicit baseline learner
    SCRIPTED = "scripted"                # scripted builder (feasibility calibration only)
    SYSTEM = "system"                    # harness-issued (e.g. advance_time at episode boundaries)


@dataclass
class Action(_Rec):
    tool: str                            # ToolSchema.name, or "advance_time" / "noop"
    args: dict = field(default_factory=dict)
    purpose: str = ""                    # optional semantic purpose (decoded or empty; not required for execution)
    entities: list = field(default_factory=list)
    code: Optional[str] = None           # executable code when the tool runs code (also inside args)
    expected_duration: Optional[float] = None
    expected_cost: Optional[float] = None
    option_id: Optional[int] = None      # active temporally-extended option (hierarchical policy)
    source: ActionSource = ActionSource.CAM_DECODER
    logp: Optional[float] = None         # behaviour-policy log-probability of the whole action
    logp_parts: dict = field(default_factory=dict)   # {"tool":..., "evidence":..., "args":..., "option":...}
    proposal: Optional[dict] = None      # when candidates were proposed: {"n": int, "probs": [...], "chosen": int}
    action_id: str = field(default_factory=lambda: new_id("a_"))

    @staticmethod
    def from_dict(d: dict) -> "Action":
        d = dict(d)
        d["source"] = ActionSource(d.get("source", "cam_decoder"))
        return Action(**d)


@dataclass
class ExecutionResult(_Rec):
    ok: bool
    observations: list = field(default_factory=list)     # list[Observation]: the tool's direct output
    error: Optional[str] = None
    cost: float = 0.0                    # observable resource cost in the world's money unit (Crucible: cents)
    duration: float = 0.0                # simulated days the environment charges for the action (Crucible cost_s/86400).
                                         # Executing never moves the clock: only the harness's advance_time does (SMDP).
    started_t: float = 0.0
    finished_t: float = 0.0
    meta: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "ExecutionResult":
        d = dict(d)
        d["observations"] = [Observation.from_dict(o) for o in d.get("observations", [])]
        return ExecutionResult(**d)


@dataclass
class GoalSpec(_Rec):
    goal_id: str
    text: str                            # semantic statement shown to the agent
    metrics: dict = field(default_factory=dict)       # numeric targets visible to the agent, e.g. {"mrr_usd": 10000}
    deadline_days: Optional[float] = None
    family: str = ""                     # task family (for curriculum and held-out bookkeeping; not a model input)
    meta: dict = field(default_factory=dict)


@dataclass
class IndependentGoalResult(_Rec):
    goal_id: str
    achieved: bool
    score: float                         # normalised progress in [0, 1] (or task-defined)
    metrics: dict = field(default_factory=dict)       # independently verified values
    t: float = 0.0
    evaluator: str = ""                  # which evaluator produced it
    details: dict = field(default_factory=dict)       # evaluator-only details: NEVER fed to the learner's inputs


class Provenance(str, enum.Enum):
    REAL = "real"                        # actually executed in an environment
    IMAGINED = "imagined"                # produced by imagination; never enters the real archive
    COUNTERFACTUAL = "counterfactual"    # evaluator fork; only in explicitly declared experimental conditions


@dataclass
class EventRecord(_Rec):
    """One consequential event (an executed action and its direct result, or an asynchronous world event)."""
    event_id: str
    env_id: str
    actor_id: str
    episode_id: str
    step: int
    model_version: int
    t_sim: float
    observations: list                   # list[Observation] available before/at the action (the agent's view)
    action: Optional[Action] = None
    result: Optional[ExecutionResult] = None
    predicted: dict = field(default_factory=dict)     # model predictions made before acting (by outcome key)
    uncertainty: dict = field(default_factory=dict)
    goal_id: Optional[str] = None
    reward: float = 0.0                  # verified return increment attributed at this step (from the harness)
    cost: float = 0.0
    done: bool = False
    entity_links: list = field(default_factory=list)
    behavior_logp: Optional[float] = None
    provenance: Provenance = Provenance.REAL
    evidence: EvidenceKind = EvidenceKind.CONTROLLED
    randomization_p: Optional[float] = None   # known assignment probability of the chosen intervention, if randomised
    simultaneous: list = field(default_factory=list)  # other interventions active at the same time (action ids)
    attributed_to: Optional[str] = None      # for delayed outcomes: event_id of the responsible action, if known
    training_provenance: dict = field(default_factory=dict)  # e.g. {"used_in_updates": [...]} appended by learner
    t_wall: float = field(default_factory=time.time)
    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "EventRecord":
        d = dict(d)
        d["observations"] = [Observation.from_dict(o) for o in d.get("observations", [])]
        d["action"] = Action.from_dict(d["action"]) if d.get("action") else None
        d["result"] = ExecutionResult.from_dict(d["result"]) if d.get("result") else None
        d["provenance"] = Provenance(d.get("provenance", "real"))
        d["evidence"] = EvidenceKind(d.get("evidence", "controlled"))
        return EventRecord(**d)


@dataclass
class DecisionTrace(_Rec):
    """Machine-readable record of one consequential decision (spec 18.1)."""
    decision_id: str
    event_id: str
    env_id: str
    policy_version: int
    t_sim: float
    objective: str
    context_refs: list = field(default_factory=list)        # event ids / artifact refs of the observable context used
    retrieved: list = field(default_factory=list)           # [{"id":..., "score":...}] memory retrieved
    candidates: list = field(default_factory=list)          # [{"action": {...}, "score":..., "source":...}]
    predicted: dict = field(default_factory=dict)           # predicted outcomes of the selected action
    uncertainty: dict = field(default_factory=dict)
    imagined: list = field(default_factory=list)            # [{"candidate": i, "horizon": h, "value":..., "outcomes": {...}}]
    selected: Optional[dict] = None
    reasoning: dict = field(default_factory=dict)           # {"iters": n, "halt_p": [...]}
    belief_before: dict = field(default_factory=dict)       # {"ess":..., "entropy":..., "top_w": [...]}
    belief_after: dict = field(default_factory=dict)
    result_ref: Optional[str] = None
    outcome: dict = field(default_factory=dict)             # filled when consequences are observed
    update_refs: list = field(default_factory=list)         # learner update ids that consumed this decision
    source: str = ActionSource.CAM_DECODER.value


# ============================================================================ 2. environment boundary

@dataclass
class ObservationBatch:
    env_ids: list                         # [B] str
    observations: list                    # [B] list[Observation] new since the previous call
    t: list                               # [B] current simulated time (days)
    done: list                            # [B] bool


@runtime_checkable
class EnvAdapter(Protocol):
    """Agent-side view of a batch of environments (MechWorld, OpenWorld, ...). Partially observable: nothing here
    may expose evaluator-only state. Adapters translate their environment into the shared record types."""

    def reset(self, seeds: Sequence[int] | None = None) -> ObservationBatch: ...
    def observe(self) -> ObservationBatch: ...
    def available_tools(self) -> list:                        # [B] list[ToolSchema]
        ...
    def execute(self, actions: Sequence[Optional[Action]]) -> list:   # [B] ExecutionResult (None action = skip)
        ...
    def collect_events(self, since: Sequence[int]) -> tuple:    # since = [B] opaque event cursors (Crucible: seq)
        ...                                                     # -> ([B] list[Observation], [B] new cursors); lossless
    def advance_time(self, duration: Sequence[float]) -> ObservationBatch: ...   # days; harness/operator side only
    def goals(self) -> list:                                  # [B] GoalSpec
        ...
    def close(self) -> None: ...


@runtime_checkable
class GoalEvaluator(Protocol):
    """Evaluator side. May inspect hidden state. The harness converts its output into verified returns; the
    agent never receives `details` or hidden mechanism parameters."""

    def evaluate_goal(self, env_index: int) -> IndependentGoalResult: ...


# ============================================================================ 3. tensor side
# (torch is imported lazily so record-only users, e.g. actor processes serialising events, stay light)

class TokenKind(enum.IntEnum):
    PAD = 0
    STATE = 1          # Z_t latent state tokens
    OBS_SEM = 2        # pooled semantic observation token
    OBS_NUM = 3        # numerical measurement token
    ENTITY = 4         # entity token e_i
    RELATION = 5       # relationship token r_ij
    EVENT = 6          # event token (action + outcome summary)
    ACTION = 7         # actual or contemplated action token(s)
    BELIEF = 8         # mechanism particle token / belief summary
    MEMORY = 9         # retrieved memory token
    GOAL = 10          # goal token(s)
    REASON = 11        # reasoning scratch tokens
    TIME = 12          # elapsed-time token (dynamics)
    NOISE = 13         # stochastic latent token (dynamics)
    QUERY = 14         # outcome / retrieval query token
    OPTION = 15        # active option token
    TOOL = 16          # available-tool token (pointer target)
    BUDGET = 17        # remaining budget / horizon token
    N_KINDS = 18


@dataclass
class TokenSet:
    """A padded batch of variable-length token sequences.
    x [B,N,D] float; mask [B,N] bool (True = real); kind [B,N] long (TokenKind); time [B,N] float (token time minus
    now, simulated days); entity [B,N] long (episode-local entity slot or -1); refs: optional [B][N] source ids
    (event ids / artifact refs / observation indices) used for evidence pointers and traces; entity2: optional [B,N]
    long second entity slot (RELATION / MESSAGE / TRANSACTION targets), -1 = none."""
    x: Any
    mask: Any
    kind: Any
    time: Any
    entity: Any
    refs: Optional[list] = None
    entity2: Any = None

    def _e2(self):
        import torch
        return self.entity2 if self.entity2 is not None else torch.full_like(self.entity, -1)

    @property
    def B(self) -> int:
        return self.x.shape[0]

    @property
    def N(self) -> int:
        return self.x.shape[1]

    def to(self, device) -> "TokenSet":
        return TokenSet(self.x.to(device), self.mask.to(device), self.kind.to(device), self.time.to(device),
                        self.entity.to(device), self.refs, None if self.entity2 is None else self.entity2.to(device))

    @staticmethod
    def empty(B: int, D: int, device=None, dtype=None) -> "TokenSet":
        import torch
        return TokenSet(torch.zeros(B, 0, D, device=device, dtype=dtype), torch.zeros(B, 0, dtype=torch.bool, device=device),
                        torch.zeros(B, 0, dtype=torch.long, device=device), torch.zeros(B, 0, device=device),
                        torch.full((B, 0), -1, dtype=torch.long, device=device), [[] for _ in range(B)])

    @staticmethod
    def from_dense(x, kind: int, time=None, entity=None, mask=None) -> "TokenSet":
        import torch
        B, N, _ = x.shape
        dev = x.device
        return TokenSet(x, torch.ones(B, N, dtype=torch.bool, device=dev) if mask is None else mask,
                        torch.full((B, N), int(kind), dtype=torch.long, device=dev),
                        torch.zeros(B, N, device=dev) if time is None else time,
                        torch.full((B, N), -1, dtype=torch.long, device=dev) if entity is None else entity, None)

    @staticmethod
    def cat(*sets: "TokenSet") -> "TokenSet":
        import torch
        sets = [s for s in sets if s is not None]
        refs = None
        if any(s.refs is not None for s in sets):
            B = sets[0].B
            refs = [sum(((s.refs[b] if s.refs is not None else [None] * s.N) for s in sets), []) for b in range(B)]
        e2 = None if all(s.entity2 is None for s in sets) else torch.cat([s._e2() for s in sets], 1)
        return TokenSet(torch.cat([s.x for s in sets], 1), torch.cat([s.mask for s in sets], 1),
                        torch.cat([s.kind for s in sets], 1), torch.cat([s.time for s in sets], 1),
                        torch.cat([s.entity for s in sets], 1), refs, e2)

    def select(self, idx) -> "TokenSet":
        """Batch rows `idx` (LongTensor or list)."""
        refs = None if self.refs is None else [self.refs[int(i)] for i in idx]
        return TokenSet(self.x[idx], self.mask[idx], self.kind[idx], self.time[idx], self.entity[idx], refs,
                        None if self.entity2 is None else self.entity2[idx])


class Likelihood(enum.IntEnum):
    GAUSSIAN = 0
    STUDENT_T = 1
    BERNOULLI = 2
    CATEGORICAL = 3
    COUNT = 4          # negative binomial
    SURVIVAL = 5       # time-to-event, discrete hazard over horizon bins, censoring-aware


@dataclass
class OutcomeBatch:
    """Typed outcome queries and (optionally) their observed values.
    q: TokenSet [B,Q,D] query tokens (encode outcome type, unit, entity, horizon); lik [B,Q] long (Likelihood);
    n_classes [B,Q] long (CATEGORICAL); y [B,Q] float (value; class index; event time for SURVIVAL);
    censored [B,Q] bool (SURVIVAL); y_mask [B,Q] bool (observed); channel [B,Q] long (fast-adaptation channel slot,
    -1 = none); keys: [B][Q] str outcome keys (e.g. "revenue@prod_3", "build_ok@repo_1") for traces."""
    q: TokenSet
    lik: Any
    n_classes: Any
    y: Any
    censored: Any
    y_mask: Any
    channel: Any
    keys: Optional[list] = None


class OutcomeDist(Protocol):
    """Distribution over Q typed outcomes; batch shape [B,Q] or [B,K,Q] (particle-conditioned)."""
    def log_prob(self, y, censored=None) -> Any: ...     # same batch shape; masked entries arbitrary
    def mean(self) -> Any: ...
    def variance(self) -> Any: ...
    def sample(self, generator=None) -> Any: ...
    def entropy(self) -> Any: ...


@dataclass
class FastState:
    """Per-environment fast adaptive state. Never part of weights; reset by reset_fast_state(); cloned per imagined
    branch. particles [B,K,Dp]; logw [B,K] (normalised log-weights); pscale [B,K,Dp] (diagonal uncertainty);
    bayes_mu [B,C,F]; bayes_P [B,C,F,F] (Kalman covariance); bayes_noise [B,C]; bayes_n [B,C];
    surprise [B,W] (recent predictive z-scores); steps [B]; channels: [B] dict outcome-key -> channel slot."""
    particles: Any
    logw: Any
    pscale: Any
    bayes_mu: Any
    bayes_P: Any
    bayes_noise: Any
    bayes_n: Any
    surprise: Any
    steps: Any
    channels: list

    def clone(self) -> "FastState":
        import copy
        return FastState(*(t.clone() for t in (self.particles, self.logw, self.pscale, self.bayes_mu, self.bayes_P,
                                               self.bayes_noise, self.bayes_n, self.surprise, self.steps)),
                         copy.deepcopy(self.channels))

    def select(self, idx) -> "FastState":
        idx_l = [int(i) for i in idx]
        return FastState(self.particles[idx], self.logw[idx], self.pscale[idx], self.bayes_mu[idx], self.bayes_P[idx],
                         self.bayes_noise[idx], self.bayes_n[idx], self.surprise[idx], self.steps[idx],
                         [dict(self.channels[i]) for i in idx_l])


class EntityRegistry:
    """Episode-local binding of persistent entity ids to slots 0..max-1. Slots get random per-episode identity codes
    (so no slot or id carries meaning across worlds); persistent ids are kept for the archive and traces only.
    When the working budget is full the least-recently-seen entity's slot is recycled (the archive keeps all)."""

    def __init__(self, max_entities: int, seed: int = 0):
        import random
        self.max = max_entities
        self.rng = random.Random(seed)
        self.slot: dict[str, int] = {}
        self.etype: dict[str, str] = {}
        self.last_seen: dict[str, float] = {}
        self.free = list(range(max_entities))
        self.rng.shuffle(self.free)

    def bind(self, entity_id: Optional[str], t: float = 0.0, etype: Optional[str] = None) -> int:
        if entity_id is None:
            return -1
        if entity_id not in self.slot:
            if not self.free:
                victim = min(self.last_seen, key=self.last_seen.get)
                self.free.append(self.slot.pop(victim))
                self.last_seen.pop(victim)
            self.slot[entity_id] = self.free.pop()
        if etype:
            self.etype[entity_id] = etype
        self.last_seen[entity_id] = t
        return self.slot[entity_id]


# ---- component protocols (implementations live in the owning workstream's package) -------------------------

class SemanticEncoder(Protocol):              # workstream A: cam_unified/semantic/encoder.py
    sem_dim: int
    version: str                               # changes whenever trainable semantic weights change (cache key)
    def hidden(self, texts: list, max_tokens: int) -> tuple: ...   # -> (h [N,T,sem_dim], mask [N,T]) token-level
    def embed(self, texts: list) -> Any: ...                         # -> [N, sem_dim] pooled document/code embedding


class ObservationTokenizer(Protocol):         # workstream A: cam_unified/semantic/tokenizer.py (nn.Module)
    def observations(self, obs: list, registries: list, now: list) -> TokenSet: ...   # [B] list[Observation] -> [B,N,D]
    def tools(self, tools: list) -> TokenSet: ...          # [B] list[ToolSchema] -> one TOOL token per tool [B,T,D]
    def goal(self, goals: list) -> TokenSet: ...           # [B] GoalSpec -> GOAL tokens
    def action(self, actions: list) -> TokenSet: ...       # [B] Action|None -> ACTION tokens (EncAction_theta)


class UnifiedCoreP(Protocol):                 # workstream B: cam_unified/core/backbone.py (nn.Module)
    def init_state(self, B: int, device=None) -> Any: ...               # [B,S,D]
    def forward(self, Z_prev, ctx: TokenSet) -> tuple: ...              # -> (Z [B,S,D], ctx_out [B,N,D])


class ReasonerP(Protocol):                    # workstream B: cam_unified/core/reasoning.py
    def forward(self, Z, ctx: TokenSet, n_iters: Optional[int] = None) -> tuple: ...
    # -> (R [B,S,D], info {"halt_p": [B,J], "iters": [B] float, "ponder": [B]})


class EntityEncoderP(Protocol):               # workstream B: cam_unified/core/entities.py
    def forward(self, obs_tokens: TokenSet) -> TokenSet: ...            # -> ENTITY + RELATION tokens


class BeliefModuleP(Protocol):                # workstream B: cam_unified/core/beliefs.py
    def init_fast(self, B: int, device=None) -> FastState: ...
    def tokens(self, fast: FastState) -> TokenSet: ...                  # B_t tokens [B, K+1, D]
    def update(self, fast: FastState, Z_prev, act: TokenSet, outcomes: OutcomeBatch, outcome_head) -> tuple: ...
    # -> (fast', info {"ess": [B], "loglik": [B,K], "surprise_z": [B], "resampled": [B], "expanded": [B]})


class BayesLastLayerP(Protocol):              # workstream B: cam_unified/core/bayes.py
    def features(self, out_feat, outcomes: OutcomeBatch) -> Any: ...    # -> phi [B,Q,F]
    def predict(self, fast: FastState, phi, outcomes: OutcomeBatch) -> tuple: ...   # -> (mean [B,Q], var [B,Q])
    def update(self, fast: FastState, phi, outcomes: OutcomeBatch) -> FastState: ...


class OutcomeHeadP(Protocol):                 # workstream C: cam_unified/world/outcomes.py
    def features(self, Z, outcomes: OutcomeBatch) -> Any: ...           # -> [B,Q,D]
    def dist(self, feat, outcomes: OutcomeBatch, cond=None) -> OutcomeDist: ...   # cond [B,K,Dc] -> batch [B,K,Q]


class WorldModelP(Protocol):                  # workstream C: cam_unified/world/dynamics.py
    def prior(self, Z, act: TokenSet, belief: TokenSet, dt, sample: bool = True) -> tuple: ...
    # -> (Z_hat [B,S,D], {"logits": [B,G,C], "z": [B,G*C]})
    def loss(self, batch: "SequenceBatch") -> dict: ...                 # world, kl, outcome, obs, reward, duration, horizon errors


@dataclass
class StepBatch:
    """One aligned decision step of a training sequence (tokenised on the fly from archived EventRecords)."""
    obs: TokenSet                 # observation/entity tokens available at the step
    act: TokenSet                 # executed action tokens
    outcomes: OutcomeBatch        # observed consequences of the action (y filled)
    dt: Any                       # [B] simulated days until the next step
    reward: Any                   # [B] verified return increment
    done: Any                     # [B] bool
    behavior_logp: Any            # [B]
    model_version: Any            # [B] long (policy lag)
    goal: Optional[TokenSet] = None
    event_ids: Optional[list] = None


@dataclass
class SequenceBatch:
    steps: list                   # list[StepBatch], length T (aligned; padded rows masked via done/obs.mask)


@dataclass
class ImaginedStep:
    Z: Any                        # [Bb,S,D]
    action: list                  # [Bb] Action or action-token summaries
    act_tokens: TokenSet
    outcome_mean: Any             # [Bb,Q]
    outcome_var: Any              # [Bb,Q]
    value: Any                    # [Bb]
    reward: Any                   # [Bb] predicted
    belief_ess: Any               # [Bb]
    model_error: Any              # [Bb] predicted model-error / epistemic score


@dataclass
class ImaginedTrajectory:
    steps: list                   # list[ImaginedStep]
    returns: Any                  # [Bb] predicted return incl. bootstrap value at the horizon
    branch_of: Any                # [Bb] index of the real environment each branch belongs to
    first_candidate: Any          # [Bb] index of the contemplated first action
    provenance: Provenance = Provenance.IMAGINED


class ArchiveP(Protocol):                     # workstream E: cam_unified/memory/archive.py
    def append(self, records: list) -> None: ...      # list[EventRecord]; rejects non-REAL provenance
    def get(self, event_ids: list) -> list: ...
    def scan(self, **filters) -> Iterator[list]: ...  # chunked, never loads the whole lifetime
    def count(self) -> int: ...
    def flush(self) -> None: ...


class ArtifactStoreP(Protocol):               # workstream E: cam_unified/memory/artifacts.py
    def put(self, data) -> str: ...                   # -> "sha256:<hex>"
    def get(self, ref: str) -> bytes: ...


class RetrieverP(Protocol):                   # workstream E: cam_unified/memory/retrieval.py (nn.Module)
    def query(self, Z, goal: TokenSet, belief: TokenSet) -> Any: ...     # -> q [B, key_dim]
    def retrieve(self, q, k: int, filters: Optional[list] = None) -> tuple: ...   # -> (M_t TokenSet, ids [B][k], scores)


class ActionGeneratorP(Protocol):             # workstream A: cam_unified/semantic/actions.py
    def sample(self, R, goal: TokenSet, belief: TokenSet, tools: TokenSet, obs: TokenSet, tool_lists: list,
               obs_lists: list, temperature: float = 1.0, greedy: bool = False) -> list: ...   # -> [B] Action (logp filled)
    def log_prob(self, R, goal: TokenSet, belief: TokenSet, tools: TokenSet, obs: TokenSet, tool_lists: list,
                 obs_lists: list, actions: list) -> tuple: ...   # -> (logp [B] differentiable, entropy [B])


class CheckpointOpsP(Protocol):               # workstream E: cam_unified/memory/checkpoint.py
    def save_weights(self, path: str) -> None: ...
    def load_weights_only(self, path: str) -> None: ...
    def save_full_lifetime(self, path: str) -> None: ...
    def load_full_lifetime(self, path: str) -> None: ...
    def reset_fast_state(self) -> None: ...
    def reset_optimizer(self) -> None: ...
    def reset_experience_access(self) -> None: ...
