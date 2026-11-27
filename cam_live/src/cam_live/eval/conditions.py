"""cam_live.eval.conditions — the compared conditions of spec 16.1, all built on the integrated agent.

Every condition is a :class:`cam_live.agent.CamLiveAgent` (one model: same backbone, same experiential network
initialisation for a given seed, same candidate generator, same observations and tools, same runtime, same
authorization, same verifier). They differ only in which learning pathways are active and in who chooses:

    A  frozen_llm        the frozen pretrained backbone chooses among the agent's candidates (letter scoring with
                         ``generation_logprob``); no update of any kind (no fast beliefs, no weights, no memory)
    B  llm_memory        A plus a written experience memory: after every action / matured outcome a text note is
                         appended and the most recent notes are placed in the backbone's prompt (persist across
                         lifetimes of the run; the only thing that changes)
    C  bayes_fast        the fast Bayesian last layer over the *frozen initial* semantic features (Thompson sampling
                         over its posterior predictions); no weight update of any group, no learned reward
    D  no_learned_reward the full continual learner (fast + frequent + slow updates, policy updates, credit) with
                         the learned intrinsic reward switched off: meta-updates never run, the bootstrap mix stays 1
    E  full              the full CAM Live agent, default configuration
    E_no_credit          E without retrospective credit (RUDDER / COCOA providers and credit re-assignment removed)
                         — the ablation of spec 16.4 ("does credit assignment improve learning efficiency?")

Matched by construction: candidates (``agent.use_backbone=False`` everywhere: the schema walker over the tools the
environment describes; the backbone-written candidates of E are evaluated separately in the real-backbone runs and
named as such), permissions (``AllowAllPolicy`` for diagnostic tasks), starting information, experience budget
(decisions per lifetime, lifetimes per run), verifier. Compute is *measured* per condition (:class:`Timed`), not
equalised: A/B spend inference compute only, D/E spend training compute as well — reported side by side.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..agent import CamLiveAgent
from ..config import CamLiveConfig
from ..contracts import (ActionResult, CandidateEvaluation, Decision, LearningUpdateRecord, MetaUpdateRecord,
                         OutcomeObservation, OutcomeStatus, VerifiedOutcome)

CONDITIONS = ("A", "B", "C", "D", "E", "E_no_credit", "E_mix0", "E_unverified")
NAMES = {"A": "frozen pretrained backbone, no persistent learning",
         "B": "frozen pretrained backbone + written experience memory",
         "C": "Bayesian adaptive learner on frozen semantic features (fast layer only)",
         "D": "continual learner without learned intrinsic reward (bootstrap only)",
         "E": "full CAM Live",
         "E_no_credit": "full CAM Live without retrospective credit assignment",
         "E_mix0": "full CAM Live with the learned reward driving the live policy from the start (bootstrap mix 0)",
         "E_unverified": "full CAM Live whose verifier signatures are refused (no verified supervision)"}
LETTERS = "ABCDEFGHIJKLMNOP"


def base_config(*, real: bool, scale: str = "", seed: int = 0) -> CamLiveConfig:
    """The evaluation configuration shared by every condition. ``real``: Qwen3-4B (frozen, bf16) on the GPU with the
    50m core (or ``scale``); otherwise the tiny random CPU backbone with the debug core (or ``scale``). Learning
    cadence as in the staging validation (``validation.staging_config``), so a short lifetime trains at all."""
    if real:
        cfg = CamLiveConfig.load()
        cfg.device = "cuda"
        agent = dict(proposer_max_new_tokens=160, proposer_k=4, policy_update_every=3, policy_update_batch=12,
                     planner_measure_every=4, experiment_rate=0.1, durable=False)
        learner = {"min_independent_outcomes": 2, "warmup_steps": 2, "batch_size": 8, "slow_batch_size": 3,
                   "grad_last": 2, "adapter_every": 3}
    else:
        cfg = CamLiveConfig.debug()
        cfg.backbone.tiny_vocab = 4096
        agent = dict(proposer_max_new_tokens=16, durable=False, policy_update_every=3, policy_update_batch=12,
                     planner_measure_every=4, experiment_rate=0.1)
        learner = {"min_independent_outcomes": 2, "warmup_steps": 0, "slow_batch_size": 3, "grad_last": 2,
                   "adapter_every": 3}
    if scale:
        import dataclasses
        mc = cfg.core.max_context
        cfg.core = dataclasses.replace(cfg.core.apply_scale(scale), max_context=mc)
    cfg.seed = int(seed)
    cfg.optim.meta_min_independent_outcomes = 4
    sec = dict(cfg.sections.get("agent", {}))
    sec.update(agent)
    cfg.sections["agent"] = sec
    cred = dict(cfg.sections.get("credit", {}))
    cred["learner"] = {**cred.get("learner", {}), **learner}
    cfg.sections["credit"] = cred
    return cfg


def eval_config(base: CamLiveConfig, condition: str, *, seed: int) -> CamLiveConfig:
    """The configuration of ``condition`` (a deep copy of ``base``): only the switches named in the module doc."""
    cfg = copy.deepcopy(base)
    cfg.seed = int(seed)
    agent = dict(cfg.sections.get("agent", {}))
    agent["use_backbone"] = False
    if condition == "C":
        agent["selection_rule"] = "thompson"
        agent["experiment_rate"] = 0.0
    if condition in ("A", "B"):
        agent["experiment_rate"] = 0.0
    cfg.sections["agent"] = agent
    if condition in ("D", "E_mix0"):
        reward = copy.deepcopy(cfg.sections.get("reward", {}))
        meta = dict(reward.get("meta", {}))
        if condition == "D":
            meta.update(min_independent=10 ** 9, mixer_step=0.0, mix=1.0)
        else:
            meta.update(mixer_step=0.0, mix=0.0)
        reward["meta"] = meta
        cfg.sections["reward"] = reward
    return cfg


# ------------------------------------------------------------------------------------------------ A / B
class FrozenBackboneChooser(CamLiveAgent):
    """Baselines A and B. The integrated agent's bookkeeping (observations, candidates, journal records) is kept so
    the runtime and the evaluator see the same interface; every learning step is a no-op."""

    memory: bool = False
    max_note_chars: int = 2400
    temperature: float = 1.0
    eps_floor: float = 0.02

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self.notes: List[str] = []
        self.q_action: Dict[str, str] = {}
        self.llm_seconds = 0.0

    # no fast update, no weights, no reward
    def update_beliefs(self) -> Dict[str, Any]:
        n, self.buffer = len(self.buffer), []
        return {"n_outcomes": n, "fast_updates": 0, "note": "baseline without persistent learning"}

    def internal_rewards(self) -> Dict[str, Any]:
        self.unrewarded = []
        return {}

    def learn(self, budget_s: float) -> List[LearningUpdateRecord]:
        self.fast_records = []
        return []

    def meta_learn(self, budget_s: float) -> List[MetaUpdateRecord]:
        return []

    def evaluate(self, candidates):
        did = self._decision_id()
        self.pending_eval = {"decision_id": did, "refs": self._window(), "imagined": False, "planner_horizon": 0}
        return [CandidateEvaluation(action=a, cost=a.est_cost) for a in candidates]

    def _prompt(self, evals: Sequence[CandidateEvaluation]) -> str:
        tools = {}
        lines = []
        for i, ev in enumerate(evals):
            a = ev.action
            lines.append(f"{LETTERS[i]}. {a.operation} {_args(a.arguments)}")
            tools.setdefault(a.operation, "")
        digest = self._digest() or "(nothing observed yet)"
        parts = [f"Objective: {self._goal_text()}", "", "Recent observations (most recent first):", digest]
        if self.memory:
            notes = []
            n = 0
            for t in reversed(self.notes):
                n += len(t) + 1
                if n > self.max_note_chars:
                    break
                notes.append(t)
            parts += ["", "Your notes from earlier experience (oldest first):",
                      "\n".join(reversed(notes)) if notes else "(none yet)"]
        parts += ["", "Possible next actions:", "\n".join(lines), "",
                  "Which action brings the objective closest? Answer with the letter only."]
        user = "\n".join(parts)
        bb = self.model.backbone
        if hasattr(bb, "chat_prompt"):
            return bb.chat_prompt(user, system="You choose the next action of an autonomous agent.")
        return user + "\nAnswer: "

    def select(self, evaluations: Sequence[CandidateEvaluation]) -> Decision:
        evals = list(evaluations)[:len(LETTERS)]
        did = self.pending_eval.get("decision_id") or self._decision_id()
        refs = self.pending_eval.get("refs") or self._window()
        now = float(self.clock.now())
        t0 = time.perf_counter()
        prompt = self._prompt(evals)
        with torch.no_grad():
            lp = self.model.generation_logprob([prompt] * len(evals), [LETTERS[i] for i in range(len(evals))])
        self.llm_seconds += time.perf_counter() - t0
        lp = lp.detach().to(torch.float64).cpu() / self.temperature
        mask = torch.tensor([bool(e.feasible) for e in evals])
        lp = torch.where(mask, lp, torch.full_like(lp, -math.inf))
        p = torch.softmax(lp, -1)
        p = (1 - self.eps_floor) * p + self.eps_floor * mask.to(torch.float64) / max(1, int(mask.sum()))
        idx = int(torch.multinomial(p, 1, generator=self.gen))
        objective_id = self.task.objective_id if self.task is not None else ""
        d = Decision(decision_id=did, ts=now, objective_id=objective_id, chosen=evals[idx].action,
                     candidates=evals, probabilities=[float(x) for x in p], chosen_index=idx,
                     exploration="policy",
                     rationale=f"frozen backbone letter scoring: p={float(p[idx]):.3f} for {LETTERS[idx]}",
                     context_obs_ids=[r for r in refs if r.startswith("obs_")][-32:],
                     model_version=self.learner.model_version, checkpoint_id=self.checkpoint_id,
                     reward_version=self.meta.reward.version,
                     diagnostics={"chooser": "frozen_backbone", "memory_notes": len(self.notes) if self.memory else 0,
                                  "letter_logprob": [float(x) for x in lp], "context_item_refs": list(refs),
                                  "lifetime": self.lifetime})
        self.n_decisions += 1
        self.pending_eval = {}
        return d

    def record_result(self, decision: Decision, result: ActionResult) -> None:
        super().record_result(decision, result)
        if not self.memory:
            return
        a = decision.chosen
        qs = []
        for o in result.observations[:2]:
            q = ", ".join(f"{x.name}={x.value:g}" for x in o.quantities[:4])
            qs.append(o.text[:100] + (f" ({q})" if q else ""))
        self.notes.append(f"- [{self.task.objective_id if self.task else ''}] {a.operation} {_args(a.arguments)} -> "
                          f"{result.status}{(': ' + '; '.join(qs)) if qs else ''}")
        for q in result.expected_outcomes:
            self.q_action[q.query_id] = f"{a.operation} {_args(a.arguments)}"

    def reconcile(self, outcomes: Sequence[OutcomeObservation],
                  verified: Optional[VerifiedOutcome] = None) -> Dict[str, Any]:
        info = super().reconcile(outcomes, verified)
        if self.memory:
            for o in outcomes:
                if o.status == OutcomeStatus.OBSERVED and o.value is not None and o.query_id in self.q_action:
                    self.notes.append(f"- later: {self.q_action.pop(o.query_id)} produced {o.value:g} "
                                      f"(measured outcome)")
            if verified is not None and verified.status in ("achieved", "failed"):
                self.notes.append(f"- objective {verified.objective_id}: {verified.status} "
                                  f"(value {verified.value:g} of target {verified.target:g})")
        return info


class MemoryChooser(FrozenBackboneChooser):
    memory = True


def _args(a: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(a.items())) or "(no arguments)"


# ------------------------------------------------------------------------------------------------ C
class FastOnlyAgent(CamLiveAgent):
    """Baseline C: Thompson sampling over the fast Bayesian layer's posterior predictions; the experiential network,
    the adapters, the heads and the reward stay at their initial values (only ``FastBeliefState`` changes)."""

    def internal_rewards(self) -> Dict[str, Any]:
        self.unrewarded = []
        return {}

    def learn(self, budget_s: float) -> List[LearningUpdateRecord]:
        out, self.fast_records = list(self.fast_records), []
        return out

    def meta_learn(self, budget_s: float) -> List[MetaUpdateRecord]:
        return []


class FrozenEvalAgent(CamLiveAgent):
    """A checkpoint evaluated as it is (spec 16.7): its own selection rule and predictions, no learning of any kind
    (no fast belief update, no weight update, no reward update). Predictions are still registered before and
    resolved after the outcomes, so prediction accuracy of the frozen weights is measured."""

    def update_beliefs(self) -> Dict[str, Any]:
        n, self.buffer = len(self.buffer), []
        return {"n_outcomes": n, "fast_updates": 0, "note": "frozen evaluation"}

    def internal_rewards(self) -> Dict[str, Any]:
        self.unrewarded = []
        return {}

    def learn(self, budget_s: float) -> List[LearningUpdateRecord]:
        self.fast_records = []
        return []

    def meta_learn(self, budget_s: float) -> List[MetaUpdateRecord]:
        return []


# ------------------------------------------------------------------------------------------------ E without credit
class NoCreditAgent(CamLiveAgent):
    """E with retrospective credit removed: the RUDDER / COCOA loss providers are unregistered and the learner gets
    no credit assigner, so delayed outcomes reach the learner only through the simple delayed-outcome attribution
    of the outcome linker (the same records, no re-assigned credit)."""

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self.learner.credit = None
        self.learner.providers = [p for p in self.learner.providers
                                  if p is not self.rudder and p is not self.cocoa]


class UnverifiedAgent(CamLiveAgent):
    """E whose verifier signatures and measurements are never accepted (fail closed): the verified-supervision
    pathway of the meta-learner, the evaluator and the objective outcomes is removed; everything else runs."""

    def _check_signature(self, v: Any) -> bool:
        return False

    def _check_measurement(self, o: Any) -> bool:
        return False


_CLASSES = {"A": FrozenBackboneChooser, "B": MemoryChooser, "C": FastOnlyAgent, "D": CamLiveAgent,
            "E": CamLiveAgent, "E_no_credit": NoCreditAgent, "E_mix0": CamLiveAgent, "E_unverified": UnverifiedAgent}


def build_condition(condition: str, base: CamLiveConfig, run_dir: str, *, seed: int, clock: Any = None,
                    model: Any = None) -> CamLiveAgent:
    if condition not in _CLASSES:
        raise KeyError(f"unknown condition {condition!r}; choose from {CONDITIONS}")
    cfg = eval_config(base, condition, seed=seed)
    return _CLASSES[condition](cfg, run_dir, clock=clock, model=model)


# ------------------------------------------------------------------------------------------------ compute
INFERENCE = ("ingest", "reconcile", "update_beliefs", "propose", "evaluate", "select", "record_result",
             "internal_rewards")
TRAINING = ("learn", "meta_learn")


class Timed:
    """Transparent proxy that measures the agent's compute per AgentAPI call (process CPU time and wall time).
    ``update_beliefs`` (the fast Bayesian update) is counted as inference: it is a closed-form posterior update."""

    def __init__(self, agent: Any):
        object.__setattr__(self, "_a", agent)
        object.__setattr__(self, "seconds", {"inference": 0.0, "training": 0.0, "save": 0.0})
        object.__setattr__(self, "cpu_seconds", {"inference": 0.0, "training": 0.0, "save": 0.0})

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._a, name)
        cat = ("inference" if name in INFERENCE else "training" if name in TRAINING else
               "save" if name in ("save", "load") else None)
        if cat is None or not callable(attr):
            return attr

        def wrapped(*a: Any, **kw: Any) -> Any:
            t0, c0 = time.perf_counter(), time.process_time()
            try:
                return attr(*a, **kw)
            finally:
                self.seconds[cat] += time.perf_counter() - t0
                self.cpu_seconds[cat] += time.process_time() - c0
        return wrapped

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._a, name, value)
