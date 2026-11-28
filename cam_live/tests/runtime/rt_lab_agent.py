"""A small real agent for the runtime tests (not the integrated CAM learner).

It implements ``contracts.AgentAPI`` honestly for LabWorld: schema-valid candidates enumerated from the discovered
tool descriptors, a uniform random behaviour policy with recorded propensities, a Gamma-Poisson belief per channel
behind every registered prediction, and a ridge regression over channel features as its only trainable weights.
It has no reward model, so it reports no intrinsic rewards and its meta-updates are recorded as skipped.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cam_live.contracts import (ActionSpec, CandidateEvaluation, Cost, Decision, Distribution, ExperimentRecord,
                                LearningUpdateRecord, MetaUpdateRecord, Observation, OutcomeKind, OutcomeObservation,
                                OutcomeQuery, OutcomeStatus, PredictionRecord, TaskSpec, ToolDescriptor,
                                VerifiedOutcome, new_id)
from cam_live.runtime.logs import atomic_write_json, read_json


def schema_values(prop: Dict[str, Any]) -> List[Any]:
    """A few valid values for one JSON-Schema property (enum, boolean, integer, number, string with a default)."""
    if "enum" in prop:
        return list(prop["enum"])[:6]
    tp = prop.get("type")
    if tp == "boolean":
        return [False, True]
    if tp == "string":
        return [prop["default"]] if isinstance(prop.get("default"), str) else []
    if tp not in ("integer", "number"):
        return []
    lo, hi = prop.get("minimum", 0), prop.get("maximum", 1)
    if tp == "integer":
        vals = list(range(int(lo), int(hi) + 1))
        return vals if len(vals) <= 8 else [vals[0], vals[len(vals) // 2], vals[-1]]
    return [float(lo), float(math.sqrt(max(lo, 1e-9) * max(hi, 1e-9))), float(hi)]


def schema_walk(tool: ToolDescriptor) -> List[Dict[str, Any]]:
    """Argument settings that satisfy the tool's schema: every combination of values of the required properties
    (optional properties are left out). A tool with a required property this walker cannot fill yields nothing."""
    schema = tool.arg_schema or {}
    props, required = schema.get("properties", {}), schema.get("required", [])
    combos: List[Dict[str, Any]] = [{}]
    for name in required:
        vals = schema_values(props.get(name, {}))
        combos = [{**c, name: v} for c in combos for v in vals]
    return combos


class LabAgent:
    def __init__(self, clock: Any, seed: int = 0, only_ops: Optional[Sequence[str]] = None,
                 wait_seconds: Optional[float] = None, durable: bool = False):
        self.clock, self.seed, self.durable = clock, seed, durable
        self.rng = random.Random(seed)
        self.only_ops = set(only_ops) if only_ops else None       # test knob: restrict the candidate set
        self.wait_seconds = wait_seconds                          # test knob: the only wait length proposed
        self.task: Optional[TaskSpec] = None
        self.objectives: List[str] = []
        self.features: Dict[int, List[float]] = {}
        self.belief: Dict[int, List[float]] = {}                  # channel -> [a, b] of Gamma(a, b) over the rate
        self.queries: Dict[str, List[Any]] = {}                   # env query id -> [channel, effort, action id]
        self.samples: List[List[float]] = []                      # [channel, effort, count] matured outcomes
        self.new_samples = 0
        self.sample_actions: List[str] = []                       # action id of each sample (learning provenance)
        self.w: List[float] = []
        self.version = 0
        self.shared_version = 0
        self.shared_w: List[float] = []
        self.experience: List[Dict[str, Any]] = []
        self.reconciled: List[str] = []
        self.verified_seen: List[Tuple[str, float]] = []
        self.results: List[Tuple[str, str]] = []                  # (action id, status) as delivered
        self.n_ingested = 0
        self.calls: Dict[str, int] = {}
        self.last_obs_ids: List[str] = []
        self.checkpoint_id = ""
        self._next_wake: Optional[float] = None
        self._experiments: Dict[str, ExperimentRecord] = {}
        self._exp_by_action: Dict[str, str] = {}
        self._exp_out: List[ExperimentRecord] = []
        self._exported = 0

    def _count(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    # ---- AgentAPI ------------------------------------------------------------------------------------------
    def set_objective(self, task: TaskSpec) -> None:
        self._count("set_objective")
        self.task = task
        self.objectives.append(task.objective_id)

    def ingest(self, observations: Sequence[Observation]) -> None:
        self._count("ingest")
        for o in observations:
            self.n_ingested += 1
            for q in o.quantities:
                parts = q.name.split(".")
                if len(parts) == 3 and parts[0] == "channel" and parts[2].startswith("f"):
                    f = self.features.setdefault(int(parts[1]), [])
                    j = int(parts[2][1:])
                    f.extend([0.0] * (j + 1 - len(f)))
                    f[j] = float(q.value)
        self.last_obs_ids = [o.obs_id for o in observations][-8:]

    def reconcile(self, outcomes: Sequence[OutcomeObservation], verified: Optional[VerifiedOutcome] = None) -> Dict[str, Any]:
        self._count("reconcile")
        matched = 0
        for o in outcomes:
            self.reconciled.append(o.query_id)
            info = self.queries.get(o.query_id)
            if info is None or o.status != OutcomeStatus.OBSERVED or o.value is None:
                continue
            ch, effort, action_id = int(info[0]), float(info[1]), info[2]
            a, b = self._ab(ch)
            self.belief[ch] = [a + float(o.value), b + effort]
            self.samples.append([ch, effort, float(o.value)])
            self.sample_actions.append(str(action_id))          # which experience each sample came from
            self.new_samples += 1
            matched += 1
            exp = self._experiments.get(self._exp_by_action.get(action_id, ""))
            if exp is not None and exp.predicted:
                mean, std = exp.predicted[0].dist.mean or 0.0, exp.predicted[0].dist.std or 0.0
                exp.actual.append(o)
                exp.status = "matured"
                exp.supported = abs(float(o.value) - mean) <= 2.0 * std
                exp.measurement_uncertainty = "count measured by the environment at the end of the attribution window"
                self._exp_out.append(exp)
        if verified is not None:
            self.verified_seen.append((verified.status, verified.value))
        return {"matched": matched, "unmatched": len(outcomes) - matched}

    def update_beliefs(self) -> Dict[str, Any]:
        self._count("update_beliefs")
        return {"channels": {str(ch): self._ab(ch)[0] / self._ab(ch)[1] for ch in sorted(self.features)}}

    def propose(self, tools: Sequence[ToolDescriptor], k: int) -> List[ActionSpec]:
        self._count("propose")
        out: List[ActionSpec] = []
        for t in tools:
            if not t.available or (self.only_ops is not None and t.name not in self.only_ops):
                continue
            for args in schema_walk(t):
                if t.name == "wait" and self.wait_seconds is not None:
                    args = {"seconds": self.wait_seconds}
                money = t.est_cost.money * (int(args.get("effort", 1)) if t.name == "lab.act" else 1)
                out.append(ActionSpec(action_id=new_id("act"), operation=t.name, modality=t.modality, level=t.level,
                                      arguments=args, est_cost=Cost(seconds=t.est_cost.seconds, money=money),
                                      auth_scope=t.auth_scope, provenance="schema_walk"))
                if t.name == "wait" and self.wait_seconds is not None:
                    break
        if len(out) > k:
            out = self.rng.sample(out, k)
        return out

    def evaluate(self, candidates: Sequence[ActionSpec]) -> List[CandidateEvaluation]:
        self._count("evaluate")
        now, evals = self.clock.now(), []
        for a in candidates:
            preds, value, unc = [], None, None
            if a.operation == "lab.act":
                ch, effort = int(a.arguments["channel"]), float(a.arguments["effort"])
                al, be = self._ab(ch)
                mean = al / be * effort
                std = math.sqrt(al * effort / be * (1.0 + effort / be))      # negative-binomial predictive
                q = OutcomeQuery(query_id=new_id("pq"), metric=f"lab.adopters.from.{a.action_id}",
                                 kind=OutcomeKind.COUNT, issued_at=now, exposure=effort, goal_relevant=True,
                                 description="adopters attributed to this intervention")
                preds.append(PredictionRecord(prediction_id=new_id("pred"), query=q,
                                              dist=Distribution("neg_binomial", {"a": al, "b": be, "exposure": effort},
                                                                mean=mean, std=std, epistemic_std=math.sqrt(al) / be * effort),
                                              made_at=now, action_id=a.action_id,
                                              model_version=f"lab-agent-v{self.version}"))
                value, unc = mean, std
            evals.append(CandidateEvaluation(action=a, predictions=preds, value=value, cost=a.est_cost, uncertainty=unc))
        return evals

    def select(self, evaluations: Sequence[CandidateEvaluation]) -> Decision:
        self._count("select")
        n, now = len(evaluations), self.clock.now()
        idx = self.rng.randrange(n)
        chosen = evaluations[idx].action
        d = Decision(decision_id=new_id("dec"), ts=now, objective_id=self.task.objective_id if self.task else "",
                     chosen=chosen, candidates=list(evaluations), probabilities=[1.0 / n] * n, chosen_index=idx,
                     exploration="randomized", rationale=f"uniform random choice among {n} schema-valid candidates",
                     context_obs_ids=list(self.last_obs_ids), model_version=f"lab-agent-v{self.version}",
                     checkpoint_id=self.checkpoint_id)
        if chosen.operation == "lab.act" and evaluations[idx].predictions:
            p = evaluations[idx].predictions[0]
            exp = ExperimentRecord(
                experiment_id=new_id("exp"), created_at=now,
                question=f"How many adopters follow effort {chosen.arguments['effort']} on channel "
                         f"{chosen.arguments['channel']} within the attribution window?",
                hypothesis=f"the count is near {p.dist.mean:.2f} (predictive std {p.dist.std:.2f})",
                evidence_refs=list(self.last_obs_ids), intervention=chosen.render(), arms=[chosen.render()],
                assignment="randomized", predicted=[p], cost=chosen.est_cost, decision_ids=[d.decision_id])
            d.experiment_id = chosen.experiment_id = exp.experiment_id
            self._experiments[exp.experiment_id] = exp
            self._exp_by_action[chosen.action_id] = exp.experiment_id
            self._exp_out.append(exp)
        return d

    def record_result(self, decision: Decision, result: Any) -> None:
        self._count("record_result")
        a = decision.chosen
        self.results.append((a.action_id, result.status))
        self.experience.append({"decision_id": decision.decision_id, "action_id": a.action_id,
                                "operation": a.operation, "arguments": dict(a.arguments), "status": result.status,
                                "propensity": decision.propensity, "ts": decision.ts})
        for q in result.expected_outcomes:
            self.queries[q.query_id] = [a.arguments.get("channel"), a.arguments.get("effort", 1), a.action_id]

    def internal_rewards(self) -> Dict[str, Any]:
        self._count("internal_rewards")
        return {}

    def learn(self, budget_s: float) -> List[LearningUpdateRecord]:
        self._count("learn")
        if not self.new_samples or not self.features:
            return []
        n_new, self.new_samples = self.new_samples, 0
        X = np.array([self.features[int(ch)] for ch, _, _ in self.samples], dtype=float)
        y = np.array([math.log1p(c / max(e, 1e-9)) for _, e, c in self.samples], dtype=float)
        old = np.array(self.w or [0.0] * X.shape[1], dtype=float)
        new = np.linalg.solve(X.T @ X + 1.0 * np.eye(X.shape[1]), X.T @ y)
        probe = np.array([self.features[ch] for ch in sorted(self.features)], dtype=float)
        change = float(np.mean(np.abs(probe @ new - probe @ old)))
        rec = LearningUpdateRecord(
            update_id=new_id("upd"), ts=self.clock.now(), trigger=f"{n_new} newly matured outcome(s)",
            timescale="frequent", losses={"mse_before": float(np.mean((X @ old - y) ** 2)),
                                          "mse_after": float(np.mean((X @ new - y) ** 2))},
            param_groups=["heads"], weight_delta={"heads": float(np.linalg.norm(new - old))},
            hash_before={"heads": self._hash(old)}, hash_after={"heads": self._hash(new)},
            n_experiences=len(self.samples), prediction_change={"probe_mean_abs_change": change},
            exp_ids=list(dict.fromkeys(self.sample_actions)),   # the experiences this regression used
            model_version=f"lab-agent-v{self.version + 1}", notes="ridge regression over channel features")
        self.w, self.version = [float(v) for v in new], self.version + 1
        self._pending_meta = True
        return [rec]

    def meta_learn(self, budget_s: float) -> List[MetaUpdateRecord]:
        self._count("meta_learn")
        if not getattr(self, "_pending_meta", False):
            return []
        self._pending_meta = False
        return [MetaUpdateRecord(meta_id=new_id("meta"), ts=self.clock.now(), reward_version_before="none",
                                 reward_version_after="none", applied=False, bootstrap_mix=1.0,
                                 skipped_reason="this agent has no learned reward function")]

    def save(self, path: str) -> str:
        self._count("save")
        state = {k: getattr(self, k) for k in ("seed", "objectives", "features", "belief", "queries", "samples",
                                                "new_samples", "w", "version", "shared_version", "shared_w",
                                                "experience", "reconciled", "verified_seen", "results", "n_ingested")}
        state["rng"] = self.rng.getstate()
        state["task"] = self.task.to_dict() if self.task else None
        body = json.dumps(state, sort_keys=True, default=str)
        self.checkpoint_id = "lab-" + hashlib.sha256(body.encode()).hexdigest()[:12]
        os.makedirs(path, exist_ok=True)
        atomic_write_json(os.path.join(path, "agent.json"), {"checkpoint_id": self.checkpoint_id, "state": state},
                          durable=self.durable)
        return self.checkpoint_id

    def load(self, path: str) -> None:
        self._count("load")
        doc = read_json(os.path.join(path, "agent.json"))
        st = doc["state"]
        for k in ("seed", "objectives", "queries", "samples", "new_samples", "w", "version", "shared_version",
                  "shared_w", "experience", "reconciled", "n_ingested"):
            setattr(self, k, st[k])
        self.features = {int(k): v for k, v in st["features"].items()}
        self.belief = {int(k): v for k, v in st["belief"].items()}
        self.verified_seen = [tuple(v) for v in st["verified_seen"]]
        self.results = [tuple(v) for v in st["results"]]
        r = st["rng"]
        self.rng.setstate((r[0], tuple(r[1]), r[2]))
        self.task = TaskSpec.from_dict(st["task"]) if st["task"] else None
        self.checkpoint_id = doc["checkpoint_id"]

    def next_wake(self) -> Optional[float]:
        self._count("next_wake")
        return self._next_wake

    def status(self) -> Dict[str, Any]:
        return {"kind": "lab test agent", "version": self.version, "shared_version": self.shared_version,
                "experience": len(self.experience), "matured_outcomes": len(self.samples),
                "checkpoint_id": self.checkpoint_id, "calls": dict(self.calls)}

    # ---- optional runtime hooks ------------------------------------------------------------------------------
    def drain_experiments(self) -> List[ExperimentRecord]:
        out, self._exp_out = self._exp_out, []
        return out

    def explain_obstacle(self, obstacle: Any) -> List[Dict[str, Any]]:
        return []                                         # this agent forms no explanations

    # ---- multi-actor plumbing (runtime.multi.ActorLearnerLink) -----------------------------------------------
    def export_experience(self) -> List[Dict[str, Any]]:
        new = self.samples[self._exported:]
        self._exported = len(self.samples)
        return [{"features": self.features[int(ch)], "effort": e, "count": c} for ch, e, c in new]

    def weights_version(self) -> int:
        return self.shared_version

    def set_weights(self, weights: Any, version: int) -> None:
        self.shared_w, self.shared_version = [float(v) for v in weights], int(version)

    # ---- internals ---------------------------------------------------------------------------------------------
    def _ab(self, ch: int) -> List[float]:
        if ch in self.belief:
            return self.belief[ch]
        prior_rate = 1.0
        if self.shared_w and ch in self.features:            # shared weights give the prior of an untried channel
            prior_rate = max(0.05, math.expm1(float(np.dot(self.shared_w, self.features[ch]))))
        return [prior_rate, 1.0]

    @staticmethod
    def _hash(w: Any) -> str:
        return hashlib.sha256(np.asarray(w, dtype=float).round(12).tobytes()).hexdigest()[:16]


def build_agent(cfg: Any, run_dir: str, clock: Any) -> LabAgent:
    """Agent factory for ``--agent-factory rt_lab_agent:build_agent`` (settings in the ``lab_agent`` config section)."""
    sec = cfg.section("lab_agent")
    return LabAgent(clock, seed=int(cfg.seed), only_ops=sec.get("only_ops"), wait_seconds=sec.get("wait_seconds"),
                    durable=True)
