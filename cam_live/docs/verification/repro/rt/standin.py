"""G-rt stand-in agents and helpers for probing workstream F's runtime (independent of F's test agent).

These are verification doubles written by the verifier, NOT the CAM learner: ``cam_live.agent`` does not exist yet.
Everything a probe concludes with them is a statement about the runtime's handling of the ``AgentAPI`` contract.

Run probes from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/rt/<probe>.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cam_live.contracts import (ActionSpec, CandidateEvaluation, Cost, Decision, Distribution,  # noqa: E402
                                LearningUpdateRecord, MetaUpdateRecord, Observation, OutcomeKind, OutcomeObservation,
                                OutcomeQuery, PredictionRecord, RunFiles, SimClock, TaskSpec, ToolDescriptor,
                                VerifiedOutcome, new_id)

SCRATCH = os.environ.get("G_RT_SCRATCH",
                         "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/g_rt")


def scratch(name: str) -> str:
    os.makedirs(SCRATCH, exist_ok=True)
    return tempfile.mkdtemp(prefix=name + "_", dir=SCRATCH)


class StandInAgent:
    """Deterministic AgentAPI implementation. ``script`` is a list of operation names chosen in order (cycling);
    each entry may be ``"op"`` or ``("op", {args})``. Every lab.act candidate gets one registered prediction."""

    def __init__(self, clock: Any, script: Sequence[Any] = ("lab.act", "wait"), wait_seconds: float = 86400.0,
                 durable: bool = True, name: str = "g-standin"):
        self.clock, self.script, self.wait_seconds, self.durable, self.name = clock, list(script), wait_seconds, durable, name
        self.task: Optional[TaskSpec] = None
        self.trace: List[str] = []                 # method names in call order (not persisted)
        self.n_select = 0
        self.results: List[List[str]] = []         # [action_id, status] in delivery order
        self.reconciled: List[str] = []            # query ids in delivery order
        self.verified: List[List[Any]] = []
        self.ingested: List[str] = []
        self.query_of: Dict[str, str] = {}         # env query id -> action id
        self.new_outcomes = 0
        self.version = 0
        self.objectives: List[str] = []
        self.hooks: Dict[str, Callable[..., None]] = {}     # method name -> callable run at entry (probe instrumentation)

    def _t(self, name: str, *a: Any) -> None:
        self.trace.append(name)
        h = self.hooks.get(name)
        if h:
            h(*a)

    # ---- AgentAPI ----
    def set_objective(self, task: TaskSpec) -> None:
        self._t("set_objective", task)
        self.task = task
        self.objectives.append(task.objective_id)

    def ingest(self, observations: Sequence[Observation]) -> None:
        self._t("ingest", observations)
        self.ingested += [o.obs_id for o in observations]

    def reconcile(self, outcomes: Sequence[OutcomeObservation], verified: Optional[VerifiedOutcome] = None) -> Dict[str, Any]:
        self._t("reconcile", outcomes, verified)
        for o in outcomes:
            self.reconciled.append(o.query_id)
            self.new_outcomes += 1
        if verified is not None:
            self.verified.append([verified.status, verified.value])
        return {"n": len(outcomes)}

    def update_beliefs(self) -> Dict[str, Any]:
        self._t("update_beliefs")
        return {}

    def propose(self, tools: Sequence[ToolDescriptor], k: int) -> List[ActionSpec]:
        self._t("propose", tools, k)
        out: List[ActionSpec] = []
        by = {t.name: t for t in tools}
        want = self.script[self.n_select % len(self.script)] if self.script else "wait"
        op, args = (want, None) if isinstance(want, str) else (want[0], dict(want[1]))
        for name in dict.fromkeys([op, "wait"]):
            t = by.get(name)
            if t is None:
                continue
            if name == op and args is not None:
                a = args
            elif name == "wait":
                a = {"seconds": self.wait_seconds}
            elif name == "lab.act":
                a = {"channel": self.n_select % 3, "effort": 1}
            elif name == "lab.probe":
                a = {"channel": self.n_select % 3}
            else:
                a = {}
            out.append(ActionSpec(action_id=new_id("act"), operation=name, modality=t.modality, level=t.level,
                                  arguments=a, est_cost=Cost(seconds=t.est_cost.seconds, money=t.est_cost.money),
                                  auth_scope=t.auth_scope, provenance="g-standin"))
        return out[:k]

    def evaluate(self, candidates: Sequence[ActionSpec]) -> List[CandidateEvaluation]:
        self._t("evaluate", candidates)
        now, evals = self.clock.now(), []
        for a in candidates:
            preds = []
            if a.operation == "lab.act":
                q = OutcomeQuery(query_id=new_id("pq"), metric=f"lab.adopters.from.{a.action_id}",
                                 kind=OutcomeKind.COUNT, issued_at=now, goal_relevant=True)
                preds.append(PredictionRecord(prediction_id=new_id("pred"), query=q,
                                              dist=Distribution("poisson", {"rate": 1.5}, mean=1.5, std=1.22),
                                              made_at=now, action_id=a.action_id, model_version=f"{self.name}-v{self.version}"))
            evals.append(CandidateEvaluation(action=a, predictions=preds, value=0.0, cost=a.est_cost))
        return evals

    def select(self, evaluations: Sequence[CandidateEvaluation]) -> Decision:
        self._t("select", evaluations)
        idx = 0                                         # the scripted op is always first
        self.n_select += 1
        n = len(evaluations)
        return Decision(decision_id=new_id("dec"), ts=self.clock.now(),
                        objective_id=self.task.objective_id if self.task else "", chosen=evaluations[idx].action,
                        candidates=list(evaluations), probabilities=[1.0 / n] * n, chosen_index=idx,
                        exploration="scripted", rationale="scripted stand-in", model_version=f"{self.name}-v{self.version}")

    def record_result(self, decision: Decision, result: Any) -> None:
        self._t("record_result", decision, result)
        self.results.append([decision.chosen.action_id, result.status])
        for q in result.expected_outcomes:
            self.query_of[q.query_id] = decision.chosen.action_id

    def internal_rewards(self) -> Dict[str, Any]:
        self._t("internal_rewards")
        return {"n": float(len(self.results))}

    def learn(self, budget_s: float) -> List[LearningUpdateRecord]:
        self._t("learn", budget_s)
        if not self.new_outcomes:
            return []
        n, self.new_outcomes = self.new_outcomes, 0
        self.version += 1
        return [LearningUpdateRecord(update_id=new_id("upd"), ts=self.clock.now(), trigger=f"{n} matured outcome(s)",
                                     timescale="frequent", param_groups=["heads"], weight_delta={"heads": 0.01},
                                     prediction_change={"probe": 0.01}, n_experiences=n,
                                     model_version=f"{self.name}-v{self.version}")]

    def meta_learn(self, budget_s: float) -> List[MetaUpdateRecord]:
        self._t("meta_learn", budget_s)
        return []

    def save(self, path: str) -> str:
        self._t("save", path)
        st = {k: getattr(self, k) for k in ("n_select", "results", "reconciled", "verified", "query_of",
                                            "new_outcomes", "version", "objectives")}
        st["task"] = self.task.to_dict() if self.task else None
        body = json.dumps(st, sort_keys=True)
        cid = "g-" + hashlib.sha256(body.encode()).hexdigest()[:12]
        os.makedirs(path, exist_ok=True)
        tmp = os.path.join(path, "agent.json.tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps({"id": cid, "state": st}))
            f.flush()
            if self.durable:
                os.fsync(f.fileno())
        os.replace(tmp, os.path.join(path, "agent.json"))
        return cid

    def load(self, path: str) -> None:
        self._t("load", path)
        with open(os.path.join(path, "agent.json")) as f:
            st = json.load(f)["state"]
        for k in ("n_select", "results", "reconciled", "verified", "query_of", "new_outcomes", "version", "objectives"):
            setattr(self, k, st[k])
        self.task = TaskSpec.from_dict(st["task"]) if st["task"] else None

    def next_wake(self) -> Optional[float]:
        self._t("next_wake")
        return None

    def status(self) -> Dict[str, Any]:
        return {"kind": "G-rt stand-in (not the CAM learner)", "version": self.version, "results": len(self.results)}


# ----------------------------------------------------------------------------------------------------------------
# log helpers
# ----------------------------------------------------------------------------------------------------------------
LOGS = (RunFiles.EVENTS, RunFiles.DECISIONS, RunFiles.EXPERIMENTS, RunFiles.PREDICTIONS, RunFiles.LEARNING,
        RunFiles.META, RunFiles.OBSTACLES)


def read_log(run_dir: str, name: str) -> List[Dict[str, Any]]:
    out = []
    p = os.path.join(run_dir, name)
    if not os.path.exists(p):
        return out
    with open(p, "rb") as f:
        for raw in f:
            if not raw.endswith(b"\n"):
                out.append({"__torn__": raw.decode(errors="replace")})
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                out.append({"__corrupt__": raw.decode(errors="replace")})
    return out


def all_records(run_dir: str) -> List[Dict[str, Any]]:
    recs = []
    for n in LOGS:
        for r in read_log(run_dir, n):
            r["__file__"] = n
            recs.append(r)
    return recs


def check_seq(run_dir: str) -> Dict[str, Any]:
    """Global seq: unique across files, strictly increasing inside each file, no torn / corrupt line."""
    recs = all_records(run_dir)
    bad = [r for r in recs if "__torn__" in r or "__corrupt__" in r]
    good = [r for r in recs if "seq" in r]
    seqs = [r["seq"] for r in good]
    dup = sorted({s for s in seqs if seqs.count(s) > 1}) if len(set(seqs)) != len(seqs) else []
    nonmono = []
    for n in LOGS:
        last = -1
        for r in good:
            if r["__file__"] != n:
                continue
            if r["seq"] <= last:
                nonmono.append((n, last, r["seq"]))
            last = r["seq"]
    rids = [r["rid"] for r in good]
    gaps = sorted(set(range(min(seqs), max(seqs) + 1)) - set(seqs)) if seqs else []
    return {"n": len(good), "torn_or_corrupt": len(bad), "duplicate_seq": dup, "non_monotonic": nonmono,
            "duplicate_rid": len(rids) - len(set(rids)), "gaps": gaps[:10], "n_gaps": len(gaps)}


class Report:
    """Collects named checks of one probe and prints a JSON summary; exit code 1 if any check failed
    (a *failed check* means the property under attack does not hold = a defect was reproduced)."""

    def __init__(self, probe: str):
        self.probe, self.checks = probe, []

    def check(self, name: str, ok: bool, detail: Any = "") -> bool:
        self.checks.append({"check": name, "holds": bool(ok), "detail": detail})
        print(f"[{'HOLDS ' if ok else 'BROKEN'}] {name}" + (f" :: {detail}" if detail != "" else ""), flush=True)
        return bool(ok)

    def note(self, name: str, detail: Any) -> None:
        self.checks.append({"note": name, "detail": detail})
        print(f"[NOTE  ] {name} :: {detail}", flush=True)

    def finish(self) -> int:
        broken = [c["check"] for c in self.checks if c.get("holds") is False]
        out = os.path.join(SCRATCH, f"{self.probe}.result.json")
        os.makedirs(SCRATCH, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"probe": self.probe, "broken": broken, "checks": self.checks}, f, indent=1, default=str)
        print(f"== {self.probe}: {len(broken)} broken of {sum(1 for c in self.checks if 'check' in c)} checks; "
              f"details: {out}")
        return 1 if broken else 0


def make_lab_run(name: str, *, lab_kw: Optional[Dict[str, Any]] = None, script: Sequence[Any] = ("lab.act", "wait"),
                 rt_kw: Optional[Dict[str, Any]] = None, task_edit: Any = None, policy: Any = None, verifier: Any = None,
                 wait_seconds: float = 86400.0):
    """A LabWorld run driven by the stand-in agent. Returns a namespace (rt, agent, world, policy, verifier, clock, run_dir)."""
    from types import SimpleNamespace
    from cam_live.diagnostics.world import AllowAllPolicy, LabConfig, LabVerifier, LabWorld, lab_task
    from cam_live.runtime.loop import LiveRuntime, RuntimeConfig
    clock = SimClock()
    world = LabWorld(LabConfig(**{"seed": 3, "target": 5, **(lab_kw or {})}), clock)
    task = lab_task(world, "obj_g")
    if task_edit:
        task_edit(task)
    agent = StandInAgent(clock, script=script, wait_seconds=wait_seconds, durable=False)
    policy = policy or AllowAllPolicy()
    verifier = verifier or LabVerifier(world)
    run_dir = os.path.join(scratch(name), "run")
    rt = LiveRuntime(agent, world, policy, verifier, task, run_dir, clock, RuntimeConfig(**{"fsync": False, **(rt_kw or {})}))
    return SimpleNamespace(rt=rt, agent=agent, world=world, policy=policy, verifier=verifier, clock=clock,
                           run_dir=run_dir, task=task)


# ----------------------------------------------------------------------------------------------------------------
# a tiny EnvironmentAdapter stand-in that works with a real clock (LabWorld needs a simulated one)
# ----------------------------------------------------------------------------------------------------------------
class SlowEnv:
    """One tool ``slow.act`` (external_write) whose execution takes ``exec_seconds`` of the given clock and whose
    outcome matures ``outcome_horizon`` seconds later; plus ``wait``. Diagnostic provenance, never live."""
    interface_version = "1.0"

    def __init__(self, clock: Any, exec_seconds: float = 0.0, outcome_horizon: float = 3600.0, money: float = 1.0,
                 est_money: Optional[float] = None, text: str = "state of the slow environment", caps: Optional[Dict[str, Any]] = None):
        import time as _t
        from cam_live.contracts import ActionResult, Level, Modality, Provenance, ResultStatus, Trust
        self._c = dict(ActionResult=ActionResult, Level=Level, Modality=Modality, Provenance=Provenance,
                       ResultStatus=ResultStatus, Trust=Trust, time=_t)
        self.clock, self.exec_seconds, self.outcome_horizon, self.money = clock, exec_seconds, outcome_horizon, money
        self.est_money = money if est_money is None else est_money
        self.text, self.caps = text, caps or {}
        self.executed: List[Dict[str, Any]] = []       # ground truth: every execution with its start time
        self.calls: Dict[str, int] = {}
        self.measured: Dict[str, float] = {}
        self.on_execute: Optional[Callable[[Any], None]] = None
        self.result_edit: Optional[Callable[[Any], None]] = None

    def _n(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def _obs(self, text: str, parent: str = "", kind: str = "metric", data: Optional[Dict[str, Any]] = None) -> Observation:
        c, now = self._c, self.clock.now()
        return Observation(obs_id=new_id("obs"), ts=now, kind=kind, text=text, parent_action_id=parent, data=data or {},
                           provenance=c["Provenance"](source="slowenv", retrieved_at=now, trust=c["Trust"].UNTRUSTED))

    def observe(self) -> List[Observation]:
        self._n("observe")
        return [self._obs(self.text)]

    def discover_tools(self) -> List[ToolDescriptor]:
        c = self._c
        return [ToolDescriptor("slow.act", c["Modality"].API, "Run the slow operation.", {"type": "object", "properties": {}},
                               c["Level"].INTERVENTION, "slow.write", "external_write", Cost(money=self.est_money)),
                ToolDescriptor("wait", c["Modality"].INTERNAL, "Do nothing until later.", {"type": "object", "properties": {
                    "seconds": {"type": "number"}}, "required": ["seconds"]}, c["Level"].INTERVENTION, "", "none", Cost())]

    def validate_action(self, action: ActionSpec) -> List[str]:
        return [] if action.operation in ("slow.act", "wait") else [f"unknown operation {action.operation!r}"]

    def execute(self, action: ActionSpec, authorization: Any) -> Any:
        c, now = self._c, self.clock.now()
        self._n("execute")
        self.executed.append({"action_id": action.action_id, "at": now})
        if self.on_execute:
            self.on_execute(action)
        if self.exec_seconds:
            if hasattr(self.clock, "advance"):
                self.clock.advance(self.exec_seconds)
            else:
                c["time"].sleep(self.exec_seconds)
        end = self.clock.now()
        q = OutcomeQuery(query_id=new_id("q"), metric=f"slow.result.{action.action_id}", kind=OutcomeKind.REAL,
                         horizon_s=self.outcome_horizon, issued_at=end, description="delayed result of the slow operation")
        res = c["ActionResult"](action_id=action.action_id, status=c["ResultStatus"].OK, started=now, ended=end,
                                observations=[self._obs("slow operation finished", action.action_id, "tool_result")],
                                cost=Cost(seconds=end - now, money=self.money, primitive_ops=1), verified=True,
                                expected_outcomes=[q])
        if self.result_edit:
            self.result_edit(res)
        return res

    def poll_events(self, cursor: str):
        self._n("poll_events")
        return [], cursor

    def get_resource_state(self) -> Dict[str, Any]:
        return {}

    def get_pending_outcomes(self) -> List[OutcomeQuery]:
        return []

    def get_observable_metrics(self) -> List[Any]:
        return []

    def get_capabilities(self) -> Dict[str, Any]:
        return {"name": "slowenv", "provenance": "diagnostic", "live": False, **self.caps}

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        from cam_live.contracts import OutcomeStatus
        self._n("measure")
        now, out = self.clock.now(), []
        for q in queries:
            if now < q.due_at:
                out.append(OutcomeObservation(q.query_id, OutcomeStatus.PENDING, source="slowenv"))
            else:
                self.measured[q.query_id] = now
                out.append(OutcomeObservation(q.query_id, OutcomeStatus.OBSERVED, 1.0, observed_at=now, source="slowenv"))
        return out


class PendingVerifier:
    """OutcomeVerifier stand-in: always reports the objective as pending, with an HMAC signature it can check."""

    def __init__(self, clock: Any):
        self.clock, self._key, self.calls = clock, os.urandom(32), 0

    def _sig(self, o: VerifiedOutcome) -> str:
        import hmac
        d = o.to_dict()
        d.pop("signature", None)
        return hmac.new(self._key, json.dumps(d, sort_keys=True).encode(), hashlib.sha256).hexdigest()

    def verify(self, task: TaskSpec) -> VerifiedOutcome:
        self.calls += 1
        o = VerifiedOutcome(objective_id=task.objective_id, status="pending", value=0.0,
                            target=float(task.success.get("target", 1)), evaluated_at=self.clock.now())
        o.signature = self._sig(o)
        return o

    def check_signature(self, outcome: VerifiedOutcome) -> bool:
        import hmac
        return bool(outcome.signature) and hmac.compare_digest(outcome.signature, self._sig(outcome))

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        return []


def slow_task(objective_id: str = "obj_slow", **limits: Any) -> TaskSpec:
    from cam_live.contracts import Limits
    lim = {"duration_s": 10 * 86400.0, "max_decisions": 10_000, **limits}
    return TaskSpec(objective_id=objective_id, objective="Reach the target of the slow diagnostic environment.",
                    success={"metric": "slow.total", "target": 10 ** 9}, family="diag/slow", interfaces=["api", "internal"],
                    limits=Limits(**lim), mode="diagnostic", stop_conditions=["verified_failure"])


def make_slow_run(name: str, clock: Any, *, env_kw: Optional[Dict[str, Any]] = None, script: Sequence[Any] = ("slow.act",),
                  rt_kw: Optional[Dict[str, Any]] = None, limits: Optional[Dict[str, Any]] = None, policy: Any = None,
                  agent: Any = None, wait_seconds: float = 3600.0, task: Optional[TaskSpec] = None):
    from types import SimpleNamespace
    from cam_live.diagnostics.world import AllowAllPolicy
    from cam_live.runtime.loop import LiveRuntime, RuntimeConfig
    env = SlowEnv(clock, **(env_kw or {}))
    task = task or slow_task(**(limits or {}))
    agent = agent or StandInAgent(clock, script=script, wait_seconds=wait_seconds, durable=False)
    policy = policy or AllowAllPolicy()
    verifier = PendingVerifier(clock)
    run_dir = os.path.join(scratch(name), "run")
    rt = LiveRuntime(agent, env, policy, verifier, task, run_dir, clock, RuntimeConfig(**{"fsync": False, **(rt_kw or {})}))
    return SimpleNamespace(rt=rt, agent=agent, env=env, policy=policy, verifier=verifier, clock=clock, run_dir=run_dir,
                           task=task)
