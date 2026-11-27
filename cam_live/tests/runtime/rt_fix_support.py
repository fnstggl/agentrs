"""Small scripted AgentAPI and environment for the runtime regression tests (test doubles of workstream F; the real
learner is ``cam_live.agent``). The scripted agent proposes one action per decision from ``script``."""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from cam_live.contracts import (ActionResult, ActionSpec, CandidateEvaluation, Cost, Decision, Distribution, Level,
                                Limits, Modality, Observation, OutcomeKind, OutcomeObservation, OutcomeQuery,
                                OutcomeStatus, PredictionRecord, Provenance, ResultStatus, TaskSpec, ToolDescriptor,
                                Trust, new_id)


class ScriptAgent:
    """``script``: list of operation names (cycled). ``args_of(n)`` gives the arguments of the n-th decision;
    ``preds_of(action)`` the predictions registered for a candidate; ``experiments`` is drained once."""

    def __init__(self, clock: Any, script: Sequence[str] = ("x.act",), wait_seconds: float = 3600.0,
                 args_of: Optional[Callable[[int], Dict[str, Any]]] = None,
                 preds_of: Optional[Callable[[ActionSpec], List[PredictionRecord]]] = None):
        self.clock, self.script, self.wait_seconds = clock, list(script), wait_seconds
        self.args_of, self.preds_of = args_of, preds_of
        self.n = 0
        self.task_seen: List[TaskSpec] = []
        self.results: List[List[str]] = []
        self.reconciled: List[str] = []
        self.experiments: List[Any] = []
        self.calls: Dict[str, int] = {}

    def _c(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def set_objective(self, task):
        self._c("set_objective")
        self.task_seen.append(task)

    def ingest(self, obs):
        self._c("ingest")

    def reconcile(self, outcomes, verified=None):
        self._c("reconcile")
        self.reconciled += [o.query_id for o in outcomes]
        return {"n": len(outcomes)}

    def update_beliefs(self):
        self._c("update_beliefs")
        return {"beliefs": "updated", "n": self.n}

    def propose(self, tools, k):
        self._c("propose")
        by = {t.name: t for t in tools}
        op = self.script[self.n % len(self.script)]
        t = by[op]
        args = {"seconds": self.wait_seconds} if op == "wait" else (self.args_of(self.n) if self.args_of else {})
        return [ActionSpec(action_id=new_id("act"), operation=op, modality=t.modality, level=t.level, arguments=args,
                           est_cost=Cost(money=t.est_cost.money), provenance="test")]

    def evaluate(self, cands):
        self._c("evaluate")
        return [CandidateEvaluation(action=a, predictions=list(self.preds_of(a)) if self.preds_of else [])
                for a in cands]

    def select(self, evals):
        self._c("select")
        self.n += 1
        return Decision(decision_id=new_id("dec"), ts=self.clock.now(), objective_id="", chosen=evals[0].action,
                        candidates=list(evals), probabilities=[1.0], exploration="scripted", rationale="test")

    def record_result(self, decision, result):
        self._c("record_result")
        self.results.append([decision.chosen.action_id, result.status])

    def internal_rewards(self):
        return {}

    def learn(self, budget_s):
        return []

    def meta_learn(self, budget_s):
        return []

    def drain_experiments(self):
        out, self.experiments = self.experiments, []
        return out

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "agent.json"), "w") as f:
            json.dump({"n": self.n, "results": self.results, "reconciled": self.reconciled}, f)
        return f"ck-{self.n}"

    def load(self, path):
        d = json.load(open(os.path.join(path, "agent.json")))
        self.n, self.results, self.reconciled = d["n"], d["results"], d["reconciled"]

    def next_wake(self):
        return None

    def status(self):
        return {"kind": "test double"}


class OneToolEnv:
    """Tool ``x.act`` (external_write, estimated and charged ``money``) and ``wait``. Each execution issues one query
    ``x.result.<action id>`` maturing ``horizon`` seconds later; ``edit(result)`` may tamper with a result."""
    interface_version = "1.0"

    def __init__(self, clock: Any, money: float = 1.0, horizon: float = 3600.0, text: str = "page",
                 caps: Optional[Dict[str, Any]] = None):
        self.clock, self.money, self.horizon, self.text, self.caps = clock, money, horizon, text, caps or {}
        self.executed: List[str] = []
        self.edit: Optional[Callable[[ActionResult], Any]] = None

    def _obs(self, text, parent=""):
        now = self.clock.now()
        return Observation(obs_id=new_id("obs"), ts=now, kind="tool_result", text=text, parent_action_id=parent,
                           provenance=Provenance(source="onetool", retrieved_at=now, trust=Trust.UNTRUSTED))

    def observe(self):
        return [self._obs(self.text)]

    def discover_tools(self):
        return [ToolDescriptor("x.act", Modality.API, "Run the operation.", {}, Level.INTERVENTION, "x.write",
                               "external_write", Cost(money=self.money)),
                ToolDescriptor("wait", Modality.INTERNAL, "Do nothing until later.", {}, Level.INTERVENTION, "", "none")]

    def validate_action(self, action):
        return []

    def execute(self, action, authorization):
        now = self.clock.now()
        self.executed.append(action.action_id)
        q = OutcomeQuery(query_id=new_id("q"), metric=f"x.result.{action.action_id}", horizon_s=self.horizon,
                         issued_at=now)
        res = ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=now, ended=now,
                           observations=[self._obs("done", action.action_id)], cost=Cost(money=self.money),
                           expected_outcomes=[q], verified=True)
        if self.edit is not None:
            out = self.edit(res)
            return res if out is None else out
        return res

    def poll_events(self, cursor):
        return [], cursor

    def get_resource_state(self):
        return {}

    def get_pending_outcomes(self):
        return []

    def get_observable_metrics(self):
        return []

    def get_capabilities(self):
        return {"name": "onetool", "provenance": "diagnostic", "live": False, **self.caps}

    def measure(self, queries):
        now = self.clock.now()
        return [OutcomeObservation(q.query_id, OutcomeStatus.PENDING) if now < q.due_at else
                OutcomeObservation(q.query_id, OutcomeStatus.OBSERVED, 1.0, observed_at=now, source="onetool")
                for q in queries]


class QuietVerifier:
    """Always pending, signed with a key of its own (no measurements)."""

    def __init__(self, clock):
        import hmac, hashlib  # noqa: E401
        self.clock, self._k, self._h = clock, os.urandom(16), (hmac, hashlib)

    def _sig(self, o):
        hmac, hashlib = self._h
        d = o.to_dict()
        d.pop("signature", None)
        return hmac.new(self._k, json.dumps(d, sort_keys=True).encode(), hashlib.sha256).hexdigest()

    def verify(self, task):
        from cam_live.contracts import VerifiedOutcome
        o = VerifiedOutcome(objective_id=task.objective_id, status="pending", target=1.0, evaluated_at=self.clock.now())
        o.signature = self._sig(o)
        return o

    def check_signature(self, o):
        return bool(o.signature) and o.signature == self._sig(o)

    def measure(self, queries):
        return []


def small_task(objective_id: str = "obj_t", mode: str = "diagnostic", **limits) -> TaskSpec:
    lim = {"duration_s": 30 * 86400.0, "max_decisions": 20, **limits}
    return TaskSpec(objective_id=objective_id, objective="Reach the target of the test environment.",
                    success={"verifier": "membership", "metric": "x.total", "target": 10 ** 6,
                             "definition": "VERIFIER-ONLY-DEFINITION counts only what the verifier reads",
                             "exclusions": {"test_account_ids": ["111111111111111111"]},
                             "evidence_source": "jsonl:/nonexistent"},
                    family="test/x", interfaces=["api", "internal"], limits=Limits(**lim), mode=mode,
                    measurement={"protocol": "VERIFIER-ONLY-PROTOCOL read by the verifier"},
                    resources={"discord_bot_credential_name": "bot_secret_name", "authorized_tools": ["x.act"]},
                    stop_conditions=["verified_failure"])
