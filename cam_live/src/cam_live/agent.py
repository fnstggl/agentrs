"""cam_live.agent — the integrated CAM Live learner: one neural intelligence behind ``contracts.AgentAPI``.

:class:`CamLiveAgent` composes the components of the workstreams; it adds no learning mathematics of its own.

    model         nn.model.build_model(cfg)            frozen backbone + LoRA + latent bridge, experiential core
    heads         predict.heads.attach_heads(model)    world model, transition, evaluator, policy, downstream VOI
    fast state    ContinualLearner.fast_state           Bayesian last-layer beliefs (one object, shared by the predictor)
    store         credit.replay.SqliteExperienceStore   <run_dir>/learner/experience.sqlite
    registries    predict.registry                      hash-chained prediction and experiment registries
    predictor     predict.predictor.Predictor           predictions registered before anything is executed
    decider       predict.policy.DecisionMaker          exact probabilities of the full candidate set
    proposer      predict.candidates.BackboneProposer   backbone text conditioned on the belief b through the bridge,
                                                        merged with the schema walker over env.discover_tools()
    planner       predict.planner.LatentPlanner         latent imagination, only up to the horizon measured accurate
    meta          reward.metalearner.MetaLearner        R_psi, the real inner policy update, the outer meta-update
    learner       credit.learner.ContinualLearner       world / fast-prior / evaluator / adapter / RUDDER / COCOA /
                                                        bridge-grounding / retention providers, three timescales
    credit        credit.providers.CreditAssigner + credit.links.OutcomeLinker
    goal-free     reward.selfgoals.GoalFreeMode         budgeted self-chosen learning goals when no objective is given

Nothing in here knows about Discord, LabWorld or any task. Candidate actions come from the tools the environment
describes; the questions the agent asks about its actions come from (a) the measurements the environment itself
reported after earlier executions of the same operation (learned query templates), (b) the immediate execution
outcome and (c) later readings of the metrics the environment reports. Verified objective outcomes reach the
evaluator, the store and the meta-learner only after the verifier's signature has been checked here; verifier
measurements supervise the meta objective only when the verifier can authenticate them (``check_measurement``).
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import torch

from .config import CamLiveConfig
from .contracts import (ActionResult, ActionSpec, CandidateEvaluation, Clock, Cost, Decision, ExperienceItem,
                        ExperienceRecord, ExperimentRecord, FailureClass, GoalStatus, ItemKind, LearningUpdateRecord,
                        MetaUpdateRecord, Modality, ObstacleRecord, Observation, OutcomeKind, OutcomeObservation,
                        OutcomeQuery, OutcomeStatus, PredictionRecord, Quantity, ResultStatus, RunFiles, TaskSpec,
                        ToolDescriptor, Trust, VerifiedOutcome, WallClock, new_id)
from .credit.checkpoint import capture_learner, load_checkpoint, restore_learner, save_checkpoint_at
from .credit.learner import ContinualLearner, LearnerSettings
from .credit.links import OutcomeLinker
from .credit.providers import (BridgeGroundingProvider, CreditAssigner, HindsightProvider,
                               ReturnDecompositionProvider)
from .credit.replay import SqliteExperienceStore
from .credit.retention import AnchorSet, RetentionProvider
from .credit.select import EstimatorSelector
from .learning_api import DecisionBatch
from .nn.model import build_model
from .predict.candidates import BackboneProposer
from .predict.heads import attach_heads, canonical_query
from .predict.imagination import is_imagined
from .predict.planner import LatentPlanner
from .predict.policy import RULES, DecisionMaker
from .predict.predictor import Predictor
from .predict.providers import default_providers
from .predict.registry import AlreadyResolvedError, ExperimentRegistry, PredictionRegistry, RegistryError
from .reward.metalearner import MetaLearner
from .reward.selfgoals import GoalFreeBudget, GoalFreeMode

AGENT_STATE_VERSION = 1
LEARNER_DIR = "learner"                      # <run_dir>/learner: everything the learner process owns
QUERY_EXEC_OK = "action.succeeded"           # immediate execution outcome the agent asks about every action
OBS_PREFIX = "observed."                     # later reading of a metric the environment reports


@dataclass
class AgentSettings:
    """Section ``agent`` of the configuration. Engineering knobs only: no strategy, no task knowledge."""
    selection_rule: str = "policy"           # predict.policy.RULES except "randomized"
    goal_free_rule: str = "voi"              # rule among self-chosen learning goals
    experiment_rate: float = 0.1             # chance to run a randomised comparison when the model is unsure
    experiment_min_flip: float = 0.2         # ... of the order of two candidates (DecisionMaker.propose_experiment)
    max_candidates: int = 16
    max_context: int = 0                     # 0: cfg.core.max_context
    use_backbone: bool = True                # backbone proposals (BackboneProposer) besides the schema walker
    proposer_k: int = 6
    proposer_max_new_tokens: int = 256
    proposer_temperature: float = 0.7
    proposer_per_tool: int = 4
    proposer_max_obs_chars: int = 3000
    templates_per_operation: int = 4
    observed_metric_queries: int = 3         # metrics whose next reading is predicted for every candidate
    policy_update_every: int = 4             # newly rewarded decisions between real inner policy updates
    policy_update_batch: int = 16
    planner_measure_every: int = 8           # decisions between multi-step accuracy measurements of the planner
    planner_sequence_len: int = 12
    goal_free_seconds: float = 7 * 86400.0
    goal_free_decisions: int = 200
    goal_free_k: int = 3
    adapters: bool = True                    # slow-timescale adapter providers
    bridge_grounding: bool = True
    retention: bool = True
    credit_beta: float = 0.0                 # duration discount exp(-beta * dt) of credit and returns
    durable: bool = True                     # store synchronous=FULL (False only for diagnostics / tests)
    max_obs_text: int = 2000

    @classmethod
    def from_cfg(cls, cfg: CamLiveConfig, **over: Any) -> "AgentSettings":
        sec = dict(cfg.section("agent"))
        sec.update(over)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(sec) - known)
        if unknown:
            raise KeyError(f"unknown keys in the 'agent' configuration section: {unknown}")
        s = cls(**sec)
        if s.selection_rule not in RULES or s.selection_rule == "randomized":
            raise ValueError(f"agent.selection_rule must be one of {[r for r in RULES if r != 'randomized']}")
        return s


# ----------------------------------------------------------------------------------------------------------------
# experience items (one token source each; exact numbers stay in quantities)
# ----------------------------------------------------------------------------------------------------------------
_OBS_KIND = {"event": ItemKind.EVENT, "document": ItemKind.DOCUMENT, "error": ItemKind.FAILURE,
             "metric": ItemKind.OBSERVATION, "page": ItemKind.OBSERVATION, "tool_result": ItemKind.OBSERVATION}


def observation_item(o: Observation, max_text: int = 2000) -> ExperienceItem:
    return ExperienceItem(kind=_OBS_KIND.get(o.kind, ItemKind.OBSERVATION), text=(o.text or "")[:max_text], ts=o.ts,
                          quantities=list(o.quantities), ref=o.obs_id, trust=o.provenance.trust or Trust.UNTRUSTED)


def action_item(a: ActionSpec, ts: float, result: Optional[ActionResult] = None) -> ExperienceItem:
    qs = []
    if result is not None:
        qs = [Quantity("action.ok", 1.0 if result.status == ResultStatus.OK else 0.0, kind=OutcomeKind.BINARY),
              Quantity("action.seconds", float(result.cost.seconds), "s"),
              Quantity("action.money", float(result.cost.money))]
    text = a.render() if result is None else f"{a.render()} => {result.status}/{result.failure_class}"
    return ExperienceItem(kind=ItemKind.ACTION if result is None or result.status == ResultStatus.OK
                          else ItemKind.FAILURE, text=text[:2000], ts=ts, quantities=qs, ref=f"{a.action_id}:act",
                          trust=Trust.SELF)


def outcome_item(q: OutcomeQuery, o: OutcomeObservation, action_id: str, trusted: bool) -> ExperienceItem:
    cq = canonical_query(q, [action_id])
    return ExperienceItem(kind=ItemKind.OUTCOME, text=(q.description or cq.metric)[:500], ts=o.observed_at,
                          quantities=[Quantity(cq.metric, float(o.value or 0.0), kind=q.kind, exposure=q.exposure)],
                          ref=f"{o.query_id}:out", trust=Trust.VERIFIER if trusted else Trust.UNTRUSTED)


def tools_item(tools: Sequence[ToolDescriptor], ts: float, objective_id: str) -> ExperienceItem:
    names = sorted(t.name for t in tools if t.available)
    digest = hashlib.sha256("|".join(names).encode()).hexdigest()[:12]
    return ExperienceItem(kind=ItemKind.TOOL, text=("available tools: " + ", ".join(names))[:1500], ts=ts,
                          quantities=[Quantity("tools.available", float(len(names)), kind=OutcomeKind.COUNT)],
                          ref=f"tools:{objective_id}:{digest}", trust=Trust.OPERATOR)


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


# ----------------------------------------------------------------------------------------------------------------
# the agent
# ----------------------------------------------------------------------------------------------------------------
class CamLiveAgent:
    """``contracts.AgentAPI`` + ``drain_experiments`` / ``explain_obstacle`` for any ``EnvironmentAdapter``.

    ``verifier`` (an ``OutcomeVerifier``, or anything with ``check_signature`` / optional ``check_measurement``)
    is used only to *check* signatures; without one every verified outcome is refused for supervision and the
    refusal is reported (fail closed). It can be attached later with :meth:`attach_verifier`."""

    def __init__(self, cfg: CamLiveConfig, run_dir: str, *, clock: Optional[Clock] = None,
                 task: Optional[TaskSpec] = None, verifier: Any = None, model: Any = None,
                 settings: Optional[AgentSettings] = None):
        self.cfg = cfg
        self.run_dir = os.path.abspath(run_dir)
        self.dir = os.path.join(self.run_dir, LEARNER_DIR)
        os.makedirs(self.dir, exist_ok=True)
        self.clock = clock or WallClock()
        self.settings = settings or AgentSettings.from_cfg(cfg)
        self.verifier = verifier
        s = self.settings
        torch.manual_seed(int(cfg.seed))
        # ---- neural model and heads ----
        self.model = model if model is not None else build_model(cfg)
        self.heads = attach_heads(self.model)
        # ---- durable experience, registries ----
        self.store = SqliteExperienceStore(os.path.join(self.dir, RunFiles.STORE), seed=cfg.seed, durable=s.durable)
        self.predictions = PredictionRegistry(os.path.join(self.dir, "prediction_registry.jsonl"), self.clock)
        self.experiments = ExperimentRegistry(os.path.join(self.dir, "experiment_registry.jsonl"), self.predictions,
                                              self.clock, seed=cfg.seed)
        # ---- reward / meta (registers R_psi as head "reward", group "reward") ----
        self.meta = MetaLearner(self.model, self.store, cfg, check_signature=self._check_signature,
                                check_measurement=self._check_measurement,
                                ledger_path=os.path.join(self.dir, "outcome_ledger.jsonl"), clock=self.clock)
        # ---- credit ----
        self.linker = OutcomeLinker(self.store, clock=self.clock, retro_predecessors=64)
        self.rudder = ReturnDecompositionProvider(self.model, self.store, beta=s.credit_beta, seed=cfg.seed)
        self.cocoa = HindsightProvider(self.model, self.store, beta=s.credit_beta, seed=cfg.seed)
        self.selector = EstimatorSelector(log_path=os.path.join(self.dir, "credit_selection.jsonl"), clock=self.clock)
        self.assigner = CreditAssigner(self.model, self.store, rudder=self.rudder, cocoa=self.cocoa,
                                       selector=self.selector, beta=s.credit_beta, clock=self.clock)
        # ---- continual learner and its providers ----
        lsec = cfg.section("credit").get("learner", {})
        lover = {} if "val_split" in lsec else {"val_split": "record"}   # a single live objective must train
        self.learner = ContinualLearner(self.model, self.store, cfg, clock=self.clock,
                                        settings=LearnerSettings.from_cfg(cfg, **lover),
                                        log_path=os.path.join(self.dir, "learner_updates.jsonl"), credit=self.assigner)
        self.providers = default_providers(self.heads, self._check_signature, adapters=s.adapters)
        for p in self.providers:
            self.learner.register(p)
        self.learner.register(self.rudder)
        self.learner.register(self.cocoa)
        self.anchors = AnchorSet()
        if s.retention:
            self.learner.register(RetentionProvider(self.anchors, seed=cfg.seed))
        if s.adapters and s.bridge_grounding:
            self.learner.register(BridgeGroundingProvider(self.model))
        self.learner.add_probe("policy", self._policy_probe)
        self.meta.check_learner_optimizers(list(self.learner.optimizers.values()))
        self.evaluator_loss = next((p for p in self.providers if p.name == "evaluator"), None)
        # ---- prediction, decision, generation, imagination (one fast state, shared) ----
        self.predictor = Predictor(self.model, self.learner.fast_state, self.predictions, self.clock, cfg,
                                   heads=self.heads)
        self.decider = DecisionMaker(self.model, cfg, heads=self.heads, clock=self.clock,
                                     experiments=self.experiments, seed=cfg.seed)
        self.planner = LatentPlanner(self.model, cfg, heads=self.heads)
        self.proposer = BackboneProposer(self.model, k=s.proposer_k, max_new_tokens=s.proposer_max_new_tokens,
                                         temperature=s.proposer_temperature, per_tool=s.proposer_per_tool,
                                         seed=cfg.seed)
        self.goal_free = GoalFreeMode(self.model, self.model.new_fast_state(),
                                      GoalFreeBudget(seconds=s.goal_free_seconds, decisions=s.goal_free_decisions),
                                      clock=self.clock)
        # ---- randomness (all checkpointed) ----
        self.gen = torch.Generator().manual_seed(int(cfg.seed) + 17)
        self.rng = random.Random(int(cfg.seed) + 23)
        self.uid = os.urandom(3).hex()                    # per agent (checkpointed): ids of two agents never collide
        self.max_context = int(s.max_context or cfg.core.max_context)
        # ---- per-lifetime state ----
        self.task: Optional[TaskSpec] = None
        self.objective_started_at = 0.0
        self.lifetime = 0
        self.ctx: List[str] = []                          # item refs of the current objective, in order
        self.goal_ref = self.tools_ref = ""
        self.recent_obs: Deque[Dict[str, Any]] = collections.deque(maxlen=12)
        self.metric_kinds: Dict[str, str] = {}             # metric name -> outcome kind, in order of last reading
        self.templates: Dict[str, List[Dict[str, Any]]] = {}
        self.self_pending: Dict[str, Dict[str, Any]] = {}  # self-asked query id -> {exp_id, metric, due, made_at}
        self.query_owner: Dict[str, Dict[str, str]] = {}  # query id -> {exp_id, action_id, experiment_id, arm}
        self.exp_outcome_keys: set = set()
        self.unrewarded: List[str] = []
        self.rewarded_since_update: List[str] = []
        self.rewarded: List[str] = []
        self.buffer: List[OutcomeObservation] = []        # matured outcomes waiting for the fast update (step 5)
        self.fast_records: List[LearningUpdateRecord] = []
        self.exp_out: Dict[str, ExperimentRecord] = {}
        self.pending_eval: Dict[str, Any] = {}
        self.verified_log: List[Dict[str, Any]] = []      # signed objective outcomes that supervise (re-added on load)
        self.failure_counts: Dict[str, Dict[str, int]] = {}
        self.active_goals: Dict[str, Any] = {}            # action id -> SelfGoal
        self.n_decisions = 0
        self.n_since_planner = 0
        self.meta_state_key: Tuple = ()
        self.checkpoint_id = ""
        self.last: Dict[str, Any] = {}
        self.counters: Dict[str, int] = collections.Counter()
        if task is not None:
            self.set_objective(task)

    # ============================================================================================ verification
    def attach_verifier(self, verifier: Any) -> None:
        """The independent verifier whose signatures this agent checks. Re-admits signed outcomes restored from a
        checkpoint that could not be checked before."""
        self.verifier = verifier
        for d in list(self.verified_log):
            if not d.get("admitted_to_evaluator") and self.evaluator_loss is not None:
                self._evaluator_add(VerifiedOutcome.from_dict(d["outcome"]), d)

    def _check_signature(self, v: VerifiedOutcome) -> bool:
        if self.verifier is None or not isinstance(v, VerifiedOutcome) or not v.signature:
            return False
        try:
            return bool(self.verifier.check_signature(v))
        except Exception:  # noqa: BLE001 — fail closed
            return False

    def _check_measurement(self, o: OutcomeObservation) -> bool:
        fn = getattr(self.verifier, "check_measurement", None)
        if fn is None or not isinstance(o, OutcomeObservation) or not o.signature:
            return False
        try:
            return bool(fn(o))
        except Exception:  # noqa: BLE001
            return False

    # ============================================================================================ context
    def _window(self) -> List[str]:
        head = [r for r in (self.goal_ref, self.tools_ref) if r]
        rest = [r for r in self.ctx if r not in head]
        keep = max(1, self.max_context - len(head))
        return head + rest[-keep:]

    def _put(self, item: ExperienceItem) -> None:
        self.store.put_item(item)
        if item.ref not in self.ctx[-64:]:
            self.ctx.append(item.ref)
        if len(self.ctx) > 8 * self.max_context:
            self.ctx = self.ctx[-4 * self.max_context:]

    def _items(self, refs: Optional[Sequence[str]] = None) -> List[ExperienceItem]:
        return self.store.items(list(refs if refs is not None else self._window()))

    @torch.no_grad()
    def _belief(self, items: Sequence[ExperienceItem]) -> torch.Tensor:
        return self.model.core_forward(self.model.tokenize([list(items)]), None).b

    def _goal_text(self) -> str:
        return (self.task.objective if self.task is not None else "") or ""

    def _goal_free_active(self) -> bool:
        t = self.task
        return t is not None and (not t.objective.strip() or not t.success)

    def _deadline(self) -> Optional[float]:
        t = self.task
        if t is None:
            return None
        d = t.measurement.get("deadline_ts") if isinstance(t.measurement, dict) else None
        if isinstance(d, (int, float)) and d > 0:
            return float(d)
        if t.limits.duration_s > 0:
            return float(self.objective_started_at + t.limits.duration_s)
        return None

    def _scale(self) -> float:
        t = self.task
        try:
            return max(1.0, float((t.success or {}).get("target", 1.0))) if t is not None else 1.0
        except (TypeError, ValueError):
            return 1.0

    # ============================================================================================ step 1
    def set_objective(self, task: TaskSpec) -> None:
        """A new objective keeps every weight, the reward parameters and the learned priors; task-specific fast
        beliefs return to the learned prior and the meta-learner starts a new lifetime. The same objective id only
        updates the task's boundaries."""
        if self.task is not None and task.objective_id == self.task.objective_id:
            self.task = task
            reg = getattr(self.meta, "register_objective", None)
            if callable(reg):
                reg(task)
            return
        now = float(self.clock.now())
        if self.task is not None:
            self._close_objective_bookkeeping()
            self.meta.new_lifetime()
            self.lifetime += 1
        self.learner.reset_fast()
        self.predictor.set_fast_state(self.learner.fast_state)
        self.task = task
        reg = getattr(self.meta, "register_objective", None)
        if callable(reg):                                  # the ledger admits outcomes of registered objectives only
            reg(task)
        self.objective_started_at = now
        self.ctx = []
        self.tools_ref = ""
        self.recent_obs.clear()
        self.active_goals = {}
        goal = ExperienceItem(kind=ItemKind.GOAL, text=task.objective or "no assigned objective", ts=now,
                              ref=f"goal:{task.objective_id}", trust=Trust.OPERATOR,
                              quantities=[Quantity("goal.target", self._scale())])
        self.goal_ref = goal.ref
        self._put(goal)
        self.counters["objectives"] += 1

    def _close_objective_bookkeeping(self) -> None:
        """Retention anchors from the finished objective; open experiments of it are concluded as inconclusive."""
        t = self.task
        recs = [r for r in self.store.trajectory(t.objective_id) if r.result is not None]
        if self.settings.retention and recs:
            try:
                self.anchors.add(self.model, self.store, recs[-64:], family=t.family or t.objective_id)
            except Exception as e:  # noqa: BLE001 — anchors are an aid; their absence is reported in status
                self.last["anchor_error"] = f"{type(e).__name__}: {e}"
        for rec in self.experiments.open():
            if rec.status in ("running", "matured"):
                try:
                    self.experiments.conclude(rec.experiment_id, supported=None,
                                              test={"test": "none", "reason": "objective ended"},
                                              alternative_explanations=["the objective ended before the experiment "
                                                                        "had enough outcomes"],
                                              measurement_uncertainty="insufficient outcomes")
                    self.exp_out[rec.experiment_id] = self.experiments.get(rec.experiment_id)
                except RegistryError:
                    pass

    # ============================================================================================ steps 2-3
    def ingest(self, observations: Sequence[Observation]) -> None:
        events = []
        for o in observations:
            self._put(observation_item(o, self.settings.max_obs_text))
            self.recent_obs.append({"obs_id": o.obs_id, "ts": o.ts, "kind": o.kind, "text": (o.text or "")[:600],
                                    "quantities": [[q.name, q.value] for q in o.quantities[:12]],
                                    "vocab": self._vocab_of(o)})
            for q in o.quantities:
                self.metric_kinds.pop(q.name, None)
                self.metric_kinds[q.name] = q.kind
            if o.kind == "event":
                events.append(o)
            self._resolve_observed(o)
        while len(self.metric_kinds) > 64:
            self.metric_kinds.pop(next(iter(self.metric_kinds)))
        if events:
            rep = self.linker.reconcile(events=events)
            self.counters["events_linked"] += len(rep.links)

    @staticmethod
    def _vocab_of(o: Observation) -> Dict[str, List[str]]:
        """Strings an action may refer to (element references, URLs) read from an observation's structured data."""
        out: Dict[str, List[str]] = {}
        data = o.data or {}
        for el in (data.get("elements") or [])[:24]:
            if isinstance(el, dict) and el.get("ref"):
                out.setdefault("ref", []).append(str(el["ref"]))
        for key in ("url",):
            if isinstance(data.get(key), str):
                out.setdefault("url", []).append(data[key])
        for ln in (data.get("links") or [])[:12]:
            if isinstance(ln, dict) and isinstance(ln.get("href"), str):
                out.setdefault("url", []).append(ln["href"])
        if out:
            out.setdefault("obs_id", []).append(o.obs_id)
        return out

    def _resolve_observed(self, o: Observation) -> None:
        """A reading of a metric the agent predicted resolves that prediction (an unverified environment reading)."""
        if not self.self_pending:
            return
        vals = {q.name: q.value for q in o.quantities}
        for qid, info in list(self.self_pending.items()):
            m = info["metric"]
            if m not in vals or o.ts < info["due"] or not o.ts > info["made_at"]:
                continue
            obs = OutcomeObservation(query_id=qid, status=OutcomeStatus.OBSERVED, value=float(vals[m]),
                                     observed_at=float(o.ts), verified=False, source=f"environment:{o.obs_id}",
                                     dedup_key=f"self:{qid}")
            self._attach_self_outcome(info["exp_id"], obs)
            del self.self_pending[qid]

    def _attach_self_outcome(self, exp_id: str, obs: OutcomeObservation) -> None:
        rec = self.store.get(exp_id)
        if rec is None or any(x.query_id == obs.query_id for x in rec.outcomes):
            return
        q = next((x for x in rec.queries if x.query_id == obs.query_id), None)
        if q is not None and rec.result is not None:
            self._goal_outcome(rec.decision.chosen.action_id, q.metric, float(obs.value), rec.result.cost)
        rec.outcomes.append(obs)
        self.store.update(rec)
        self.store.mark_for_retraining([exp_id], "outcome_matured", outcome_key=obs.dedup_key,
                                       now=float(self.clock.now()))
        try:
            self.predictor.resolve([obs])
        except Exception as e:  # noqa: BLE001 — e.g. not strictly after the prediction: kept for training only
            self.last["self_resolve_error"] = f"{type(e).__name__}: {e}"
        self.buffer.append(obs)

    # ============================================================================================ step 4
    def reconcile(self, outcomes: Sequence[OutcomeObservation],
                  verified: Optional[VerifiedOutcome] = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {"n_outcomes": len(outcomes)}
        outs = [o for o in outcomes if not is_imagined(o) and not str(o.source).lower().startswith("imagin")]
        info["refused_imagined"] = len(outcomes) - len(outs)
        rep = self.linker.reconcile(outcomes=outs)
        info.update(accepted=rep.outcomes_accepted, duplicates=rep.duplicates, links=len(rep.links),
                    retrain_marked=len(rep.retrain_marked))
        resolved = 0
        try:
            resolved = len(self.predictor.resolve(outs))
        except Exception as e:  # noqa: BLE001
            info["resolve_error"] = f"{type(e).__name__}: {e}"
        info["predictions_resolved"] = resolved
        for o in outs:
            if o.status not in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED) or o.value is None:
                continue
            q = self.store.query(o.query_id)
            owner = self.query_owner.get(o.query_id, {})
            if q is not None:
                trusted = bool(o.verified) and self._check_measurement(o)
                self._put(outcome_item(q, o, owner.get("action_id", ""), trusted))
            self.buffer.append(o)
            self._experiment_outcome(o, owner)
            self._goal_free_outcome(o, owner, q)
        if verified is not None:
            info["verified"] = self._verified(verified)
        return info

    def _experiment_outcome(self, o: OutcomeObservation, owner: Dict[str, str]) -> None:
        eid = owner.get("experiment_id", "")
        if not eid:
            return
        key = (eid, o.dedup_key or f"{o.query_id}@{o.observed_at}")
        if key in self.exp_outcome_keys:
            return
        self.exp_outcome_keys.add(key)
        try:
            self.experiments.add_outcome(eid, o, owner.get("arm", ""))
            if self.experiments.mature(eid):
                by_arm = self.experiments.outcomes_by_arm(eid)
                if len(by_arm) >= 2 and all(len(v) >= 2 for v in by_arm.values()):
                    res = self.experiments.analyse(eid)
                    lo, hi = res.get("ci_low", res.get("lo")), res.get("ci_high", res.get("hi"))
                    supported = None
                    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                        supported = bool(lo > 0.0) if not (lo <= 0.0 <= hi) else None
                    self.experiments.conclude(eid, supported=supported, test={"name": "welch", **_jsonable(res)},
                                              alternative_explanations=["sampling noise", "changed conditions "
                                                                        "between units"],
                                              measurement_uncertainty="Welch interval over the randomised units")
            self.exp_out[eid] = self.experiments.get(eid)
        except RegistryError as e:
            self.last["experiment_error"] = f"{type(e).__name__}: {e}"

    def _verified(self, v: VerifiedOutcome) -> Dict[str, Any]:
        """A verifier outcome supervises only after its signature checks here (and only operator objectives)."""
        if not self._check_signature(v):
            self.counters["verified_refused"] += 1
            return {"accepted": False, "reason": "signature not checked" if self.verifier is None else "bad signature"}
        origin = getattr(self.task, "origin", Trust.OPERATOR) if self.task is not None else Trust.OPERATOR
        adm = self.meta.admit_objective(v, objective_origin=origin)
        out: Dict[str, Any] = {"accepted": True, "status": v.status, "value": v.value,
                               "meta_ledger": {"accepted": adm.accepted, "reason": adm.reason}}
        entry = {"outcome": v.to_dict(), "deadline": self._deadline() or v.evaluated_at,
                 "admitted_to_evaluator": False}
        self._evaluator_add(v, entry)
        out["evaluator"] = entry["admitted_to_evaluator"]
        if v.status in (GoalStatus.ACHIEVED, GoalStatus.FAILED):
            self.store.put_objective_outcome(v.objective_id, status=v.status, value=v.value,
                                             evaluated_at=v.evaluated_at, verified=True, body=v.to_dict())
            ids = [r.exp_id for r in self.store.trajectory(v.objective_id)]
            self.store.mark_for_retraining(ids, "outcome_matured", outcome_key=f"objective:{v.objective_id}",
                                           now=float(self.clock.now()))
            try:
                out["success_predictions_resolved"] = len(self.predictor.resolve_objective(v, self._check_signature))
            except Exception as e:  # noqa: BLE001
                out["success_resolve_error"] = f"{type(e).__name__}: {e}"
            out["closed"] = True
        self.verified_log = [d for d in self.verified_log if d["outcome"]["objective_id"] != v.objective_id]
        self.verified_log.append(entry)
        return out

    def _evaluator_add(self, v: VerifiedOutcome, entry: Dict[str, Any]) -> None:
        if self.evaluator_loss is None:
            return
        try:
            traj = self.store.trajectory(v.objective_id)
            self.evaluator_loss.add_outcome(v, deadline=float(entry["deadline"]), trajectory=traj)
            entry["admitted_to_evaluator"] = True
        except Exception as e:  # noqa: BLE001 — UnverifiedOutcomeError: not admitted
            entry["evaluator_error"] = f"{type(e).__name__}: {e}"

    # ============================================================================================ step 5
    def update_beliefs(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"n_outcomes": len(self.buffer)}
        if self.buffer:
            buf, self.buffer = self.buffer, []
            res = self.predictor.observe_outcomes(buf)
            out.update(fast_updates=res["n"], surprise=res["surprise"], belief_kl=res["kl"],
                       change_detected=res["change_detected"], skipped=res["skipped"])
            if res["n"]:
                self.learner.counters["fast"] += res["n"]
                rec = LearningUpdateRecord(
                    update_id=new_id("upd"), ts=float(self.clock.now()), trigger=f"{res['n']} matured outcome(s)",
                    timescale="fast", losses={"surprise": float(res["surprise"])}, param_groups=[],
                    n_experiences=res["n"], exp_ids=sorted({d.get("action_id", "") for d in res["details"]}),
                    prediction_change={"belief_kl": float(res["kl"])}, model_version=self.learner.model_version,
                    notes=json.dumps({"change_detected": res["change_detected"], "skipped": res["skipped"],
                                      "errors": [d.get("error") for d in res["details"]]}, sort_keys=True))
                self.fast_records.append(rec)
        items = self._items()
        if self.task is not None and self.task.objective.strip():
            bv = self.predictor.belief_value(items, self._goal_text(), deadline=self._deadline(),
                                             objective_scale=self._scale())
            out.update(p_success=bv["p_success"], b_norm=bv["b_norm"])
        return out

    # ============================================================================================ step 6
    def propose(self, tools: Sequence[ToolDescriptor], k: int) -> List[ActionSpec]:
        s = self.settings
        now = float(self.clock.now())
        tools = [t for t in tools if t.available]
        if self.task is not None:
            ti = tools_item(tools, now, self.task.objective_id)
            if ti.ref != self.tools_ref:
                self.tools_ref = ti.ref
                self._put(ti)
        items = self._items()
        b = self._belief(items)
        self._reserve_decision_number()
        prefix = f"a{self.uid}{self.n_decisions:06d}"
        digest = self._digest()
        vocab = self._vocab()
        k_max = max(1, min(int(k), s.max_candidates))
        if s.use_backbone:
            res = self.proposer.propose(tools, digest, objective=self._goal_text(), latent=b, vocab=vocab,
                                        k=s.proposer_k, id_prefix=prefix, seed=self.rng.randrange(1 << 30))
            cands, stats = res.actions, res.stats
        else:
            from .predict.candidates import schema_candidates
            cands = schema_candidates(tools, vocab=vocab, per_tool=s.proposer_per_tool, id_prefix=prefix)
            stats = {"n_schema": len(cands), "latent_conditioned": False}
        for a in cands:
            if a.est_cost.seconds <= 0 and a.est_cost.money <= 0:
                t = next((t for t in tools if t.name == a.operation), None)
                if t is not None:
                    a.est_cost = Cost.from_dict(t.est_cost.to_dict())
            if a.operation.endswith("wait") and "seconds" in a.arguments:
                a.est_cost.seconds = float(a.arguments["seconds"])
        cands = self._order(cands, k_max)
        if self._goal_free_active():
            cands = self._goal_free_candidates(b, cands)
        self.last["proposal"] = {"n": len(cands), "stats": _jsonable(stats), "prefix": prefix}
        return cands

    def _decision_id(self, n: Optional[int] = None) -> str:
        return f"dec_{self.uid}{self.n_decisions if n is None else n:06d}"

    def _reserve_decision_number(self) -> None:
        """Ids are a deterministic function of the agent and its decision counter (a restored agent decides exactly
        like the uninterrupted one). After a crash the counter restored from the checkpoint may name decisions that
        were journalled or predicted after that checkpoint; such numbers are skipped, so no id is ever reused."""
        n = self.n_decisions
        while self.store.by_decision(self._decision_id(n)) is not None or self.predictions.by_decision(
                self._decision_id(n)):
            n += 1
        self.n_decisions = n

    def _order(self, cands: List[ActionSpec], k: int) -> List[ActionSpec]:
        """At most ``k`` candidates: backbone proposals first, then an even spread over the operations (so one tool
        with many argument settings cannot crowd out the others); the order within each tool is kept."""
        lm = [a for a in cands if "backbone" in a.provenance.split("+")]
        rest = [a for a in cands if a not in lm]
        by_op: Dict[str, List[ActionSpec]] = collections.OrderedDict()
        for a in rest:
            by_op.setdefault(a.operation, []).append(a)
        out = lm[:max(1, k // 2)]
        while len(out) < k and any(by_op.values()):
            for op in list(by_op):
                if by_op[op] and len(out) < k:
                    out.append(by_op[op].pop(0))
        return out

    def _digest(self) -> str:
        lines, n = [], 0
        for o in reversed(self.recent_obs):
            q = ", ".join(f"{a}={b:g}" for a, b in o["quantities"][:8])
            line = f"[{o['kind']}] {o['text']}" + (f" ({q})" if q else "")
            n += len(line)
            if n > self.settings.proposer_max_obs_chars:
                break
            lines.append(line)
        return "\n".join(reversed(lines))

    def _resource_vocab(self) -> Dict[str, List[str]]:
        """Identifiers of the resources the operator authorized in the task (URLs, account / channel ids), as
        values for the schema walker. Operator configuration, not environment content; no strategy."""
        out: Dict[str, List[str]] = {}

        def add(key: str, v: str) -> None:
            if v and v not in out.setdefault(key, []):
                out[key].append(v)

        def walk(key: str, x: Any, depth: int = 0) -> None:
            if depth > 4:
                return
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(str(k), v, depth + 1)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    walk(key, v, depth + 1)
            elif isinstance(x, (str, int)) and not isinstance(x, bool):
                v = str(x)
                if "://" in v:
                    add("url", v.rstrip("/") + "/")
                elif v.count(":") == 2 and v.split(":")[0] in ("channel", "account"):
                    add(f"{v.split(':')[0]}_id", v.split(":")[2])
                elif key.endswith("_id") or key.endswith("id"):
                    add(key, v)
        if self.task is not None:
            walk("", self.task.resources)
        return out

    def _vocab(self) -> Dict[str, List[str]]:
        v: Dict[str, List[str]] = {k: list(x) for k, x in self._resource_vocab().items()}
        for o in list(self.recent_obs)[-3:]:
            for key, vals in (o.get("vocab") or {}).items():
                for x in vals:
                    if x not in v.setdefault(key, []):
                        v[key].append(x)
        return {k: vals[-16:] for k, vals in v.items()}

    def _goal_free_candidates(self, b: torch.Tensor, cands: List[ActionSpec]) -> List[ActionSpec]:
        pairs = []
        for a in cands:
            for q in self._queries_for(a):
                if q.kind in (OutcomeKind.REAL, OutcomeKind.COUNT, OutcomeKind.BINARY):
                    pairs.append((a, q))
        goals = self.goal_free.propose(b, pairs, k=self.settings.goal_free_k, objective_assigned=False) if pairs else []
        if not goals:
            return cands
        picked, seen = [], set()
        for g in goals:
            if g.action.action_id not in seen:
                seen.add(g.action.action_id)
                g.action.provenance = (g.action.provenance + "+goal_free").strip("+")
                picked.append(g.action)
                self.active_goals[g.action.action_id] = g
            else:
                self._abandon_goal(g)                      # one learning goal per candidate action
        return picked

    def _abandon_goal(self, g: Any) -> None:
        fn = getattr(self.goal_free, "abandon", None)    # gives the goal's budget reservation back
        if callable(fn):
            fn(g)

    def _goal_outcome(self, action_id: str, metric: str, value: Optional[float], cost: Cost,
                      censored: bool = False) -> None:
        """Resolve the learning goal of an executed action with what was observed for the question it is about."""
        g = self.active_goals.get(action_id)
        if g is None or canonical_query(g.query, [action_id]).metric != metric:
            return
        self.goal_free.resolve(g, value, cost, censored=censored)
        del self.active_goals[action_id]

    # ============================================================================================ steps 7-8
    def _queries_for(self, a: ActionSpec) -> List[OutcomeQuery]:
        """The questions asked about candidate ``a`` before it is executed (templates; ids assigned on use)."""
        now = float(self.clock.now())
        out: List[OutcomeQuery] = [OutcomeQuery(
            query_id="", metric=QUERY_EXEC_OK, kind=OutcomeKind.BINARY, horizon_s=0.0, issued_at=now,
            description="whether the action executes successfully")]
        for t in self.templates.get(a.operation, []):
            ex = t["exposure"]
            if t["exposure_arg"] and isinstance(a.arguments.get(t["exposure_arg"]), (int, float)):
                ex = float(a.arguments[t["exposure_arg"]])
            out.append(OutcomeQuery(query_id="", metric=t["metric"], kind=t["kind"], horizon_s=t["horizon_s"],
                                    issued_at=0.0, description=t["description"], goal_relevant=t["goal_relevant"],
                                    exposure=float(ex), categories=list(t["categories"])))
        horizon = max(float(a.est_cost.seconds), 1.0)
        for m in self._watched_metrics():
            kind = self.metric_kinds.get(m, OutcomeKind.REAL)
            if kind not in (OutcomeKind.REAL, OutcomeKind.COUNT):
                kind = OutcomeKind.REAL
            out.append(OutcomeQuery(query_id="", metric=OBS_PREFIX + m, kind=kind, horizon_s=horizon, issued_at=now,
                                    description=f"next reading of {m} after the action"))
        return out

    def _watched_metrics(self) -> List[str]:
        n = self.settings.observed_metric_queries
        if n <= 0:
            return []
        goal = str((self.task.success or {}).get("metric", "")) if self.task is not None else ""
        names = [m for m in reversed(list(self.metric_kinds)) if not m.startswith(("tools.", "action.", "goal."))]
        out = [goal] if goal and goal in self.metric_kinds else []
        for m in names:
            if m not in out:
                out.append(m)
        return out[:n]

    def evaluate(self, candidates: Sequence[ActionSpec]) -> List[CandidateEvaluation]:
        cands = list(candidates)
        if not cands:
            return []
        refs = self._window()
        items = self._items(refs)
        did = self._decision_id()
        objective = self.task.objective_id if (self.task is not None and self.task.objective.strip()) else ""
        evals = self.predictor.evaluate_candidates(
            items, self._goal_text(), cands, queries=self._queries_for, decision_id=did, objective_id=objective,
            deadline=self._deadline(), objective_scale=self._scale(), compute_voi=True, generator=self.gen)
        imagined = None
        allowed = self.planner.controller.allowed()
        if allowed >= 1:                                   # only where multi-step accuracy has been measured
            try:
                imagined = self.planner.plan(items, self._goal_text(), cands, now=float(self.clock.now()),
                                             deadline=self._deadline(), objective_scale=self._scale(),
                                             horizon=allowed, generator=self.gen)
                for ev, im in zip(evals, imagined):
                    ev.extras.update({"imagined_value_mean": float(im.value_mean),
                                      "imagined_value_std": float(im.value_std),
                                      "imagined_horizon_steps": float(im.horizon_steps)})
            except Exception as e:  # noqa: BLE001
                self.last["planner_error"] = f"{type(e).__name__}: {e}"
        self.pending_eval = {"decision_id": did, "refs": refs, "imagined": imagined is not None,
                             "planner_horizon": allowed}
        return evals

    # ============================================================================================ step 9
    def select(self, evaluations: Sequence[CandidateEvaluation]) -> Decision:
        evals = list(evaluations)
        did = self.pending_eval.get("decision_id") or self._decision_id()
        refs = self.pending_eval.get("refs") or self._window()
        now = float(self.clock.now())
        objective_id = self.task.objective_id if self.task is not None else ""
        fields = dict(context_obs_ids=[r for r in refs if r.startswith("obs_")][-32:],
                      model_version=self.learner.model_version, checkpoint_id=self.checkpoint_id,
                      reward_version=self.meta.reward.version)
        decision, rec = None, None
        rule = self.settings.goal_free_rule if self._goal_free_active() else self.settings.selection_rule
        if self.settings.experiment_rate > 0 and len(evals) >= 2 and self.rng.random() < self.settings.experiment_rate:
            prop = self.decider.propose_experiment(evals)
            if prop is not None and prop.get("expected_flip_probability", 0.0) >= self.settings.experiment_min_flip:
                decision, rec = self._experiment(evals, prop, objective_id, did, now, fields)
        if decision is None:
            decision = self.decider.select(evals, mode=rule, objective_id=objective_id, decision_id=did, ts=now,
                                           generator=self.gen, **fields)
        decision.diagnostics["context_item_refs"] = list(refs)
        decision.diagnostics["lifetime"] = self.lifetime
        decision.diagnostics["planner_horizon"] = self.pending_eval.get("planner_horizon", 0)
        if rec is not None:
            self.exp_out[rec.experiment_id] = rec
        self.n_decisions += 1
        self.pending_eval = {}
        return decision

    def _experiment(self, evals, prop, objective_id, did, now, fields):
        names = set(prop["arms"])
        unit = f"{objective_id}:{self.n_decisions}"         # deterministic: a restored agent assigns identically
        running = next((r for r in self.experiments.open()
                        if r.assignment == "randomized" and set(r.arms) == names and r.status == "running"), None)
        try:
            if running is not None:
                arms = {n: prop["arms"][n] for n in running.arms}
                return self.decider.select_experiment(evals, arms, experiment_id=running.experiment_id,
                                                      objective_id=objective_id, decision_id=did, ts=now,
                                                      unit_id=unit, **fields)
            return self.decider.select_experiment(
                evals, prop["arms"], question=prop["question"], hypothesis=prop["hypothesis"],
                objective_id=objective_id, decision_id=did, ts=now, unit_id=unit,
                expected_flip_probability=prop.get("expected_flip_probability"), **fields)
        except (RegistryError, ValueError) as e:
            self.last["experiment_error"] = f"{type(e).__name__}: {e}"
            return None, None

    # ============================================================================================ step 12
    def record_result(self, decision: Decision, result: ActionResult) -> None:
        """Idempotent. A repeated delivery of a result the agent already holds changes nothing. A result the store
        holds but the restored agent state does not know (a decision journalled after the checkpoint the runtime
        resumed from, F-6) is re-entered into the agent's bookkeeping without touching the stored record. A later
        result for a decision that was queued for approval replaces the queued one."""
        existing = self.store.by_decision(decision.decision_id)
        replay = False
        if existing is not None and existing.result is not None:
            same = (existing.result.status == result.status and existing.result.ended == result.ended
                    and existing.result.action_id == result.action_id)
            known = existing.exp_id in self.unrewarded or existing.exp_id in self.rewarded
            if same and known:
                return
            if not same and existing.result.status != ResultStatus.NEEDS_APPROVAL:
                return
            replay = same
        a = decision.chosen
        refs = list(decision.diagnostics.get("context_item_refs") or self._window())
        self._put(action_item(a, decision.ts, result))
        for o in result.observations:
            self._put(observation_item(o, self.settings.max_obs_text))
            self.recent_obs.append({"obs_id": o.obs_id, "ts": o.ts, "kind": o.kind, "text": (o.text or "")[:600],
                                    "quantities": [[q.name, q.value] for q in o.quantities[:12]],
                                    "vocab": self._vocab_of(o)})
            for q in o.quantities:
                self.metric_kinds.pop(q.name, None)
                self.metric_kinds[q.name] = q.kind
        post = self._window()
        self._learn_templates(a, result)
        linked = self.predictor.adopt_queries(a.action_id, result.expected_outcomes)
        self.predictor.commit(decision.decision_id, a.action_id)
        chosen_ev = next((e for e in decision.candidates if e.action.action_id == a.action_id), None)
        own = [p.query for p in (chosen_ev.predictions if chosen_ev else [])]
        selfq = [q for q in own if q.metric == QUERY_EXEC_OK or q.metric.startswith(OBS_PREFIX)]
        exp_id = existing.exp_id if existing is not None else f"exp_{decision.decision_id[4:]}"
        arm = str(decision.diagnostics.get("arm", ""))
        for q in list(result.expected_outcomes) + own:
            self.query_owner[q.query_id] = {"exp_id": exp_id, "action_id": a.action_id,
                                            "experiment_id": decision.experiment_id or a.experiment_id, "arm": arm}
        for eq, pq in linked.items():
            self.query_owner[eq] = self.query_owner.get(pq, self.query_owner[eq])
        outcomes: List[OutcomeObservation] = []
        immediate: List[OutcomeObservation] = []
        for q in selfq:
            if q.metric == QUERY_EXEC_OK:
                ok = result.status == ResultStatus.OK and result.verified is not False
                o = OutcomeObservation(query_id=q.query_id, status=OutcomeStatus.OBSERVED, value=1.0 if ok else 0.0,
                                       observed_at=float(result.ended or decision.ts), verified=False,
                                       source="environment:action_result", dedup_key=f"self:{q.query_id}")
                outcomes.append(o)
                immediate.append(o)
            elif result.status == ResultStatus.OK:
                self.self_pending[q.query_id] = {"exp_id": exp_id, "metric": q.metric[len(OBS_PREFIX):],
                                                 "due": q.due_at, "made_at": decision.ts}
        if replay:
            immediate = []                                 # already attached to the stored record
        elif existing is None:
            self.store.append(ExperienceRecord(
                exp_id=exp_id, objective_id=decision.objective_id, decision=decision, result=result,
                context_item_refs=refs, post_item_refs=post, goal_text=self._goal_text(), outcomes=outcomes,
                pending_queries=list(result.expected_outcomes), queries=selfq,
                duration_s=max(0.0, float(result.ended - result.started)) if result.ended else 0.0,
                objective_family=(self.task.family if self.task is not None else ""),
                provenance=self._provenance(), lifetime=self.lifetime))
        else:                                              # the executed result of a decision queued for approval
            existing.result, existing.post_item_refs = result, post
            existing.pending_queries = list(result.expected_outcomes)
            existing.queries = selfq
            existing.outcomes = [o for o in existing.outcomes if o.query_id not in {x.query_id for x in outcomes}]
            existing.outcomes += outcomes
            existing.duration_s = max(0.0, float(result.ended - result.started)) if result.ended else 0.0
            self.store.update(existing)
        if immediate:
            self.store.mark_for_retraining([exp_id], "outcome_matured", outcome_key=immediate[0].dedup_key,
                                           now=float(self.clock.now()))
            resolvable = [o for o in immediate if o.observed_at > decision.ts]
            if resolvable:
                try:
                    self.predictor.resolve(resolvable)
                except Exception as e:  # noqa: BLE001
                    self.last["self_resolve_error"] = f"{type(e).__name__}: {e}"
                self.buffer.extend(resolvable)
        if result.status != ResultStatus.OK:
            fc = self.failure_counts.setdefault(a.operation, {})
            fc[result.failure_class] = fc.get(result.failure_class, 0) + 1
        for k in [k for k in self.active_goals if k != a.action_id]:      # proposed, not chosen
            self._abandon_goal(self.active_goals.pop(k))
        g = self.active_goals.get(a.action_id)
        if g is not None and result.status != ResultStatus.OK:
            self.goal_free.resolve(g, None, result.cost)
            del self.active_goals[a.action_id]
        for o in immediate:
            self._goal_outcome(a.action_id, QUERY_EXEC_OK, float(o.value), result.cost)
        if exp_id not in self.unrewarded and exp_id not in self.rewarded:
            self.unrewarded.append(exp_id)
        self.counters["results"] += 1

    def _provenance(self) -> str:
        mode = self.task.mode if self.task is not None else "diagnostic"
        return mode if mode in ("live", "staging", "diagnostic") else "diagnostic"

    def _learn_templates(self, a: ActionSpec, result: ActionResult) -> None:
        """What the environment reported it will measure after this operation becomes a question asked about the
        same operation's future candidates (before they are executed)."""
        lst = self.templates.setdefault(a.operation, [])
        for q in result.expected_outcomes:
            cq = canonical_query(q, [a.action_id])
            exposure_arg = ""
            if q.exposure != 1.0:
                exposure_arg = next((k for k, v in a.arguments.items() if isinstance(v, (int, float))
                                     and not isinstance(v, bool) and float(v) == float(q.exposure)), "")
            t = {"metric": cq.metric, "kind": q.kind, "horizon_s": float(q.horizon_s), "description": q.description,
                 "goal_relevant": bool(q.goal_relevant), "categories": list(q.categories),
                 "exposure": float(q.exposure), "exposure_arg": exposure_arg}
            key = (t["metric"], t["kind"], round(t["horizon_s"]))
            if not any((x["metric"], x["kind"], round(x["horizon_s"])) == key for x in lst):
                lst.append(t)
        del lst[self.settings.templates_per_operation:]

    def _goal_free_outcome(self, o: OutcomeObservation, owner: Dict[str, str], q: Optional[OutcomeQuery]) -> None:
        aid = owner.get("action_id", "")
        if q is None or aid not in self.active_goals:
            return
        rec = self.store.get(owner.get("exp_id", ""))
        cost = rec.result.cost if (rec is not None and rec.result is not None) else Cost()
        self._goal_outcome(aid, canonical_query(q, [aid]).metric, float(o.value), cost,
                           censored=o.status == OutcomeStatus.CENSORED)

    # ============================================================================================ step 13
    def internal_rewards(self) -> Dict[str, Any]:
        if not self.unrewarded:
            return {}
        recs = [r for r in self.store.get_many(self.unrewarded) if r is not None]
        recs.sort(key=lambda r: r.seq)
        out = self.meta.internal_rewards(recs)
        ids = [r.exp_id for r in recs]
        self.rewarded.extend(ids)
        self.rewarded = self.rewarded[-4096:]
        self.rewarded_since_update.extend(ids)
        self.unrewarded = []
        out["exp_ids"] = ids
        return _jsonable(out)

    # ============================================================================================ step 14
    def learn(self, budget_s: float) -> List[LearningUpdateRecord]:
        t0 = time.perf_counter()
        out: List[LearningUpdateRecord] = list(self.fast_records)
        self.fast_records = []
        plan = self.learner.poll()
        if plan is not None:
            out += self.learner.learn(dataclasses.replace(plan, budget_s=float(budget_s)))
        s = self.settings
        if len(self.rewarded_since_update) >= s.policy_update_every:
            ids = self.rewarded[-s.policy_update_batch:]
            recs = sorted((r for r in self.store.get_many(ids) if r is not None), key=lambda r: r.seq)
            rec = self.meta.policy_update(recs, trigger=f"{len(self.rewarded_since_update)} newly rewarded decisions")
            if rec is not None:
                rec.model_version = self.learner.model_version
                out.append(rec)
            self.rewarded_since_update = []
        self.n_since_planner += 1
        if self.n_since_planner >= s.planner_measure_every and time.perf_counter() - t0 < budget_s:
            self.n_since_planner = 0
            self._measure_planner()
        return out

    def _measure_planner(self) -> None:
        if self.task is None:
            return
        recs = [r for r in self.store.trajectory(self.task.objective_id) if r.post_item_refs and r.result is not None]
        seq = recs[-self.settings.planner_sequence_len:]
        if len(seq) >= 3:
            try:
                self.last["planner"] = _jsonable(self.planner.measure(self.store, [seq]))
            except Exception as e:  # noqa: BLE001
                self.last["planner_error"] = f"{type(e).__name__}: {e}"

    # ============================================================================================ step 15
    def meta_learn(self, budget_s: float) -> List[MetaUpdateRecord]:
        """One outer update of R_psi when the verified query evidence allows it, else a record of the exact reason it
        was skipped. Asked again only when something it depends on changed (events, ledger, decisions)."""
        key = (len(self.meta.events), len(self.meta.ledger), self.counters["results"])
        if key == self.meta_state_key:
            return []
        self.meta_state_key = key
        return [self.meta.meta_update(float(self.clock.now()))]

    # ============================================================================================ step 16
    def save(self, path: str) -> str:
        self.store.flush()
        state = capture_learner(self.learner, extra_optimizers={"reward_meta": self.meta.optimizer},
                                generators={"agent": self.gen, "decider": self.decider.generator},
                                lifetime_context=self.meta.reward.lifetime.state.detach().clone(),
                                current_task=self.task.to_dict() if self.task is not None else None,
                                reward_version=self.meta.reward.version,
                                extra={"agent": self._json_state(), "scaler": self.heads.scaler.state(),
                                       "planner": self.planner.controller.state()})
        state.aux_state = dict(state.aux_state or {})
        state.aux_state["agent"] = {"meta": self.meta.state_dict(), "meta_batches": self._meta_batches(),
                                    "predictor": self._predictor_state(),
                                    "goal_free": self.goal_free.state_dict(),
                                    "goal_free_fast": self.goal_free.fast.state_dict(),
                                    "py_rng": _rng_state(self.rng), "proposer_rng": _rng_state(self.proposer.rng)}
        info = save_checkpoint_at(path, state, notes=f"agent decisions={self.n_decisions}", clock=self.clock)
        self.checkpoint_id = info.checkpoint_id
        return info.checkpoint_id

    def load(self, path: str) -> None:
        st = load_checkpoint(path, map_location=self.model.device)
        restore_learner(self.learner, st, extra_optimizers={"reward_meta": self.meta.optimizer},
                        generators={"agent": self.gen, "decider": self.decider.generator})
        aux = (st.aux_state or {}).get("agent", {})
        self.meta.load_state_dict(aux["meta"])
        self._load_meta_batches(aux.get("meta_batches", {}))
        self.predictor.set_fast_state(self.learner.fast_state)
        self._load_predictor_state(aux.get("predictor", {}))
        if aux.get("goal_free"):
            self.goal_free.load_state_dict(aux["goal_free"])
        if aux.get("goal_free_fast"):
            self.goal_free.fast.load_state_dict(aux["goal_free_fast"])
        if aux.get("py_rng"):
            self.rng.setstate(_rng_from(aux["py_rng"]))
        if aux.get("proposer_rng"):
            self.proposer.rng.setstate(_rng_from(aux["proposer_rng"]))
        ex = st.extra or {}
        self.heads.scaler.load_state(ex.get("scaler", {}))
        if ex.get("planner"):
            self.planner.controller.load_state(ex["planner"])
        self._load_json_state(ex.get("agent", {}))
        if st.current_task:
            self.task = TaskSpec.from_dict(st.current_task)
        self.checkpoint_id = ex.get("checkpoint_id", "")
        for d in self.verified_log:
            d["admitted_to_evaluator"] = False
            self._evaluator_add(VerifiedOutcome.from_dict(d["outcome"]), d)

    def _json_state(self) -> Dict[str, Any]:
        return _jsonable({
            "version": AGENT_STATE_VERSION, "lifetime": self.lifetime, "ctx": self.ctx, "goal_ref": self.goal_ref,
            "tools_ref": self.tools_ref, "recent_obs": list(self.recent_obs),
            "metric_kinds": list(self.metric_kinds.items()), "templates": self.templates,
            "self_pending": self.self_pending, "query_owner": self.query_owner,
            "exp_outcome_keys": sorted([list(k) for k in self.exp_outcome_keys]), "unrewarded": self.unrewarded,
            "rewarded_since_update": self.rewarded_since_update, "rewarded": self.rewarded,
            "verified_log": self.verified_log, "failure_counts": self.failure_counts,
            "n_decisions": self.n_decisions, "n_since_planner": self.n_since_planner, "uid": self.uid,
            "objective_started_at": getattr(self, "objective_started_at", 0.0), "counters": dict(self.counters),
            "proposer_calls": self.proposer._calls, "settings": dataclasses.asdict(self.settings),
            "buffer": [o.to_dict() for o in self.buffer]})

    def _load_json_state(self, d: Dict[str, Any]) -> None:
        if not d:
            return
        self.lifetime = int(d["lifetime"])
        self.ctx, self.goal_ref, self.tools_ref = list(d["ctx"]), d["goal_ref"], d["tools_ref"]
        self.recent_obs = collections.deque(d["recent_obs"], maxlen=12)
        self.metric_kinds = collections.OrderedDict((k, v) for k, v in d["metric_kinds"])
        self.templates = {k: list(v) for k, v in d["templates"].items()}
        self.self_pending = dict(d["self_pending"])
        self.query_owner = dict(d["query_owner"])
        self.exp_outcome_keys = {tuple(k) for k in d["exp_outcome_keys"]}
        self.unrewarded, self.rewarded = list(d["unrewarded"]), list(d["rewarded"])
        self.rewarded_since_update = list(d["rewarded_since_update"])
        self.verified_log = list(d["verified_log"])
        self.failure_counts = {k: dict(v) for k, v in d["failure_counts"].items()}
        self.n_decisions, self.n_since_planner = int(d["n_decisions"]), int(d["n_since_planner"])
        self.uid = str(d.get("uid", self.uid))
        self.objective_started_at = float(d["objective_started_at"])
        self.counters = collections.Counter(d["counters"])
        self.proposer._calls = int(d["proposer_calls"])
        self.buffer = [OutcomeObservation.from_dict(x) for x in d.get("buffer", [])]

    def _meta_batches(self) -> Dict[str, Any]:
        """The support latents each policy-update event cached when it happened. ``MetaLearner.state_dict`` leaves
        them out (it would rebuild them from the store with the *current* model); keeping them makes a restored
        agent's meta-gradient the one the uninterrupted agent computes."""
        out = {}
        for e in self.meta.events:
            if e.ib is None:
                continue
            d = {}
            for f in dataclasses.fields(e.ib):
                v = getattr(e.ib, f.name)
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu()
                elif f.name == "h0":
                    v = [h.detach().cpu() for h in v]
                elif f.name in ("norm_state", "boot_components"):
                    v = {k: (x.detach().cpu() if isinstance(x, torch.Tensor) else x) for k, x in v.items()}
                elif f.name == "segments":
                    v = [list(x) for x in v]
                d[f.name] = v
            out[str(e.event_id)] = d
        return out

    def _load_meta_batches(self, d: Dict[str, Any]) -> None:
        from .reward.inner import InnerBatch
        dev = self.model.device
        for e in self.meta.events:
            b = d.get(str(e.event_id))
            if b is None:
                continue
            kw = dict(b)
            kw["segments"] = [tuple(x) for x in kw.get("segments", [])]
            e.ib = InnerBatch(**kw).to(device=dev)

    def _predictor_state(self) -> Dict[str, Any]:
        p = self.predictor
        cache = []
        for c in p._cache.values():
            cache.append({"prediction_id": c.prediction_id, "query": c.query.to_dict(), "metric": c.metric,
                          "action_id": c.action_id, "decision_id": c.decision_id, "phi": c.phi.detach().cpu(),
                          "nums": [c.head_count, c.head_loc, c.head_time, c.real_loc, c.real_scale, c.mean,
                                   c.aleatoric_std, c.epistemic_std]})
        return {"cache": cache, "alias": dict(p._alias), "absorbed": sorted(p._absorbed),
                "detectors": {k: {a: v for a, v in vars(d).items() if isinstance(v, (int, float, bool))}
                              for k, d in p._detectors.items()},
                "success": {k: list(v) for k, v in p._success.items()}}

    def _load_predictor_state(self, d: Dict[str, Any]) -> None:
        if not d:
            return
        from .predict.errors import ChangeDetector
        from .predict.predictor import _Cached
        p = self.predictor
        p._cache.clear()
        for c in d.get("cache", []):
            n = c["nums"]
            p._cache[c["query"]["query_id"]] = _Cached(c["prediction_id"], OutcomeQuery.from_dict(c["query"]),
                                                       c["metric"], c["action_id"], c["decision_id"], c["phi"],
                                                       *[float(x) for x in n])
        p._alias = dict(d.get("alias", {}))
        p._absorbed = set(d.get("absorbed", []))
        p._detectors = {}
        for k, v in d.get("detectors", {}).items():
            det = ChangeDetector()
            for a, x in v.items():
                setattr(det, a, x)
            p._detectors[k] = det
        p._success = {k: list(v) for k, v in d.get("success", {}).items()}

    # ============================================================================================ step 17 / status
    def next_wake(self) -> Optional[float]:
        """The agent itself needs no wake-up: outcomes the environment will report are scheduled by the runtime and
        the readings it predicts arrive with the next observation of meaningful work."""
        return None

    def status(self) -> Dict[str, Any]:
        cp = self.model.count_parameters()
        return _jsonable({
            "objective": self.task.objective_id if self.task is not None else "",
            "goal_free": self._goal_free_active(), "lifetime": self.lifetime, "decisions": self.n_decisions,
            "model_version": self.learner.model_version, "reward_version": self.meta.reward.version,
            "bootstrap_mix": self.meta.mixer.mix, "learner": self.learner.status(), "meta": dict(self.meta.stats),
            "meta_events": len(self.meta.events), "ledger_entries": len(self.meta.ledger),
            "credit_default": self.assigner.default, "store_records": self.store.count(),
            "predictions": len(self.predictions), "planner_horizon": self.planner.controller.allowed(),
            "parameters": {k: cp[k] for k in ("frozen", "experiential", "adapters", "adapters_lora",
                                               "adapters_bridge", "reward", "heads_total",
                                               "total_persistent_trainable")},
            "backbone": cp["backbone"], "scale": cp["scale"], "counters": dict(self.counters),
            "verifier_attached": self.verifier is not None, "checkpoint": self.checkpoint_id})

    # ============================================================================================ F-1 hooks
    def drain_experiments(self) -> List[ExperimentRecord]:
        out = list(self.exp_out.values())
        self.exp_out = {}
        return out

    def explain_obstacle(self, obstacle: ObstacleRecord) -> List[Dict[str, Any]]:
        """Possible explanations with probabilities: a Dirichlet(1) posterior over the failure classes this
        operation has shown so far (counted from executed results), so explanations follow experience rather than a
        fixed decision tree. The class the environment reported is evidence, not certainty."""
        rec = self.store.by_action(obstacle.action_id)
        op = rec.decision.chosen.operation if rec is not None else ""
        counts = dict(self.failure_counts.get(op, {}))
        classes = [c for c in FailureClass.ALL if c != FailureClass.NONE]
        tot = sum(counts.get(c, 0) + 1.0 for c in classes)
        out = sorted(({"hypothesis": c, "p": (counts.get(c, 0) + 1.0) / tot, "observed": counts.get(c, 0),
                       "operation": op} for c in classes), key=lambda d: -d["p"])
        return out[:4]

    # ============================================================================================ probes / misc
    def _policy_probe(self, model: Any, batch: DecisionBatch, store: Any) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            return {"policy_probs": self.heads.policy(batch.z, batch.a_mask, batch.b).probs.detach()}

    def policy_distribution(self, items: Sequence[ExperienceItem], candidates: Sequence[ActionSpec],
                            goal_text: Optional[str] = None) -> torch.Tensor:
        """[N] probabilities of the trainable policy over ``candidates`` in context ``items`` (no registration, no
        side effects): what ``select`` samples from under ``selection_rule="policy"`` before the exploration floor."""
        with torch.no_grad():
            b = self._belief(items)
            e_a, _ = self.model.encode_actions([list(candidates)])
            e_g = self.model.encode_goals([self._goal_text() if goal_text is None else goal_text])
            z = self.model.decision_latent(b, e_a, e_g)
            mask = torch.ones(1, len(candidates), dtype=torch.bool, device=z.device)
            return self.heads.policy(z, mask, b).probs[0].detach().cpu().double()

    def close(self) -> None:
        try:
            self.store.close()
        except Exception:  # noqa: BLE001
            pass


def _rng_state(r: random.Random) -> List[Any]:
    v, internal, gauss = r.getstate()
    return [v, list(internal), gauss]


def _rng_from(s: Sequence[Any]) -> Tuple:
    return (int(s[0]), tuple(int(x) for x in s[1]), s[2])


def build_agent(cfg: Optional[CamLiveConfig] = None, run_dir: str = "", clock: Optional[Clock] = None,
                task: Optional[TaskSpec] = None, verifier: Any = None, **settings: Any) -> CamLiveAgent:
    """Factory used by ``cam-live`` (``runtime.loop_service``): the full configuration of ``cfg`` (Qwen3-4B + 50m
    core by default; ``CamLiveConfig.debug()`` gives the tiny CPU configuration)."""
    cfg = cfg or CamLiveConfig.load()
    run_dir = run_dir or os.path.join(cfg.resolved_home(), "runs", new_id("run"))
    s = AgentSettings.from_cfg(cfg, **settings) if settings else None
    return CamLiveAgent(cfg, run_dir, clock=clock, task=task, verifier=verifier, settings=s)
