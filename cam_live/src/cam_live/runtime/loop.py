"""cam_live.runtime.loop — the persistent event-driven runtime (spec 15.1, 15.2, 13.4).

:class:`LiveRuntime` drives any :class:`~cam_live.contracts.AgentAPI` against any
:class:`~cam_live.contracts.EnvironmentAdapter`, :class:`~cam_live.contracts.PolicyEngine` and
:class:`~cam_live.contracts.OutcomeVerifier`. One call of :meth:`LiveRuntime.step` is one pass over the 17 steps:

     1 receive the objective            7-8 agent.evaluate (predictions written to disk before execution)
     2 env.observe                      9  agent.select
     3 env.poll_events (cursor)         10 env.validate_action, then policy.authorize
     4 reconcile matured outcomes       11 execute, or queue for approval and carry on
       (+ signed objective verification) 12 record observations
     5 agent.update_beliefs             13 agent.internal_rewards   14 agent.learn   15 agent.meta_learn
     6 agent.propose                    16 save state               17 continue only if there is work, else sleep

Things this module deliberately does not do: it computes no reward, it contains no recovery logic (failures flow
back to the learner as results, observations and obstacle records), and it never acts to fill time.

Crash safety. ``state.json`` is replaced atomically. Evidence delivery (event cursor, pending-outcome queue) is
committed together with a learner checkpoint; everything decided since that commit is in a write-ahead journal
(``wal.jsonl``), so a restarted process loads the checkpoint, re-delivers the journalled results, re-reads the
evidence after the committed cursor, and resolves an action that was in flight through the environment's
de-duplication by action id — it is never executed a second time.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..contracts import (ActionResult, ActionSpec, AgentAPI, AuthorizationDecision, AuthVerdict, Clock, Cost, Decision,
                         EnvironmentAdapter, FailureClass, GoalStatus, Modality, Observation, OutcomeObservation,
                         OutcomeStatus, OutcomeVerifier, PolicyEngine, Provenance, ResultStatus, RunFiles, TaskSpec,
                         Trust, VerifiedOutcome, new_id)
from .control import PID_FILE, ControlChannel, file_sha256
from .logs import Redactor, RunLog, atomic_write_json, atomic_write_text, iter_jsonl, load_salt, read_json
from .obstacles import ObstacleTracker, is_obstacle
from .scheduler import PendingItem, PendingOutcomes, SimSleeper, WallSleeper, make_sleeper, plan_wake
from .tasks import LIMIT_FIELDS, TaskValidationError, check_runnable, load_task

STATE_VERSION = 1
RUNNING, PAUSED, STOPPED = "running", "paused", "stopped"
COMPLETED, FAILED, EXHAUSTED = "completed", "failed", "exhausted"
TERMINAL = (COMPLETED, FAILED, EXHAUSTED)
WAL_FILE, TASK_FILE = "wal.jsonl", "task.json"
_COUNTERS = {"decisions": 0, "external_actions": 0, "actions_executed": 0, "waits": 0, "failures": 0,
             "money": 0.0, "compute_s": 0.0}
_LIMIT_COUNTER = {"money": "money", "compute_s": "compute_s", "max_external_actions": "external_actions",
                  "max_decisions": "decisions"}


@dataclass
class RuntimeConfig:
    """Settings of the loop itself; read from the ``runtime`` section of the root configuration."""
    candidates_k: int = 16               # candidates requested from agent.propose
    learn_budget_s: float = 30.0
    meta_budget_s: float = 30.0
    checkpoint_every_s: float = 0.0      # clock seconds between learner checkpoints; 0 = after every cycle with new experience
    keep_checkpoints: int = 5            # 0 = keep all
    verify_every_s: float = 0.0          # minimum clock seconds between objective verifications
    evidence_poll_s: float = 0.0         # wake to poll the event stream this often while idle; 0 = only scheduled wakes
    approval_poll_s: float = 0.0         # re-check pending approvals at the policy this often while idle; 0 = on notice only
    control_poll_s: float = 1.0          # how often a real-time sleep stats the control file
    min_idle_sleep_s: float = 1.0        # lower bound of one idle sleep (no busy loop when the learner returns at once)
    min_cycle_interval_s: float = 0.0    # pause between cycles whose decision did not execute successfully
    pending_retry_s: float = 300.0       # re-ask for an outcome the environment says is not available yet
    error_backoff_s: float = 30.0
    max_consecutive_errors: int = 5      # service or learner errors in a row before the run pauses itself
    journal_every: int = 200             # cycles between journal regenerations (always at pause / finish / stop) ...
    journal_min_interval_s: float = 600.0   # ... and not more often than this (clock seconds): it re-reads the logs
    fsync: bool = True                   # False keeps atomic renames but skips fsync (fast tests; never for live runs)
    record_at_policy: bool = False       # call policy.record after execute; only for an environment that does not
                                         # record its results at the authorization service itself (the computer
                                         # environment does, under the executor's identity)
    sleeper: str = "auto"                # "auto": simulated sleeping for advanceable clocks, real otherwise; "wall" | "sim"
    redaction: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, cfg: Any) -> "RuntimeConfig":
        if isinstance(cfg, cls):
            return cfg
        if cfg is None:
            return cls()
        d = cfg.section("runtime") if hasattr(cfg, "section") else dict(cfg)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def action_signature(a: ActionSpec) -> str:
    """Identity of *what an action does* (independent of its id), used to recognise a re-proposal."""
    def core(x: ActionSpec) -> Dict[str, Any]:
        return {"op": x.operation, "mod": x.modality, "target": x.target, "args": x.arguments,
                "steps": [core(s) for s in x.steps]}
    return hashlib.sha256(json.dumps(core(a), sort_keys=True, default=str).encode()).hexdigest()


def is_wait(a: ActionSpec) -> bool:
    return a.modality == Modality.INTERNAL and a.operation.split(".")[-1] in ("wait", "sleep") and not a.steps


class LiveRuntime:
    def __init__(self, agent: AgentAPI, env: EnvironmentAdapter, policy: PolicyEngine, verifier: OutcomeVerifier,
                 task: TaskSpec, run_dir: str, clock: Clock, cfg: Any = None, *,
                 wake: Optional[threading.Event] = None, actor_id: str = ""):
        self.agent, self.env, self.policy, self.verifier = agent, env, policy, verifier
        self.task, self.clock, self.actor_id = task, clock, actor_id
        self.run_dir = os.path.abspath(run_dir)
        self._base_cfg = RuntimeConfig.from_any(cfg)
        self.cfg = self._with_measurement(task)
        os.makedirs(self.run_dir, exist_ok=True)
        self._wake = wake or threading.Event()
        self.sleeper = (WallSleeper(clock, self._wake, self.cfg.control_poll_s) if self.cfg.sleeper == "wall" else
                        SimSleeper(clock) if self.cfg.sleeper == "sim" else
                        make_sleeper(clock, self._wake, self.cfg.control_poll_s))
        self.control = ControlChannel(self.run_dir, clock)
        if task.mode == "live" and not self.cfg.fsync:
            raise ValueError("runtime.fsync=false is not allowed for a live task")
        self._durable = bool(self.cfg.fsync)
        self.log = RunLog(self.run_dir, clock, provenance=task.mode, durable=self._durable,
                          redactor=Redactor.from_config(self.cfg.redaction, load_salt(self.run_dir),
                                                        task.data_permissions))
        self.obstacles = ObstacleTracker(self.log, clock)
        self.pending = PendingOutcomes()
        self.cursor = ""
        self.status, self.status_reason = "new", ""
        self.started_at = self.paused_at = 0.0
        self.paused_total_s = 0.0
        self.iterations = self.cycles = self.generation = 0
        self.counters: Dict[str, Any] = dict(_COUNTERS)
        self.lifetime: Dict[str, Any] = {"decisions": 0, "actions_executed": 0, "objectives": []}
        self.approvals: Dict[str, Dict[str, Any]] = {}
        self.hooks: Dict[str, List[Callable[["LiveRuntime"], None]]] = {"after_cycle": []}
        self._committed: Dict[str, Any] = {}
        self._ckpts: List[Dict[str, Any]] = []
        self._exports: List[Dict[str, Any]] = []
        self._next_tasks: List[Dict[str, Any]] = []
        self._open_preds: List[Dict[str, Any]] = []
        self._control_applied = 0
        self._control_results: Dict[str, Dict[str, Any]] = {}
        self._control_mtime = -1.0
        self._activity: Dict[str, Any] = {"phase": "new", "detail": "", "since": clock.now()}
        self._wake_info: Dict[str, Any] = {}
        self._last_verified: Dict[str, Any] = {}
        self._last_verify_at: Optional[float] = None
        self._refused_verified = 0
        self._agent_wake_at: Optional[float] = None
        self._waiting: Optional[Dict[str, Any]] = None
        self._triggers: List[str] = []
        self._errors: List[Dict[str, Any]] = []
        self._consecutive_errors = 0
        self._error_total = 0
        self._compute_unbooked = 0.0
        self._dirty = False
        self._last_ckpt_ts: Optional[float] = None
        self._stop_requested = ""
        self._started = False
        self._tool_effects: Dict[str, str] = {}
        self._cycle_obs: List[Observation] = []
        self._metrics: Dict[str, float] = {}
        self._interrupt_checks = 0
        self._last_result_ok = True
        self._last_journal_at = float("-inf")
        self._wal_fh = None
        self._wal_path = os.path.join(self.run_dir, WAL_FILE)
        self._state_path = os.path.join(self.run_dir, RunFiles.STATE)

    def _with_measurement(self, task: TaskSpec) -> RuntimeConfig:
        """The operator's measurement protocol (task file) sets the verification and evidence-poll intervals."""
        m, over = task.measurement or {}, {}
        for key, field_name in (("verify_interval_s", "verify_every_s"), ("evidence_poll_interval_s", "evidence_poll_s")):
            if isinstance(m.get(key), (int, float)) and not isinstance(m.get(key), bool) and m[key] >= 0:
                over[field_name] = float(m[key])
        return dataclasses.replace(self._base_cfg, **over)

    # ==========================================================================================================
    # start / resume
    # ==========================================================================================================
    def start(self) -> None:
        if self._started:
            return
        st = read_json(self._state_path)
        resume = isinstance(st, dict) and st.get("version") == STATE_VERSION and bool(st.get("status"))
        self._guard_live(TaskSpec.from_dict(st["task"]) if resume else self.task)
        atomic_write_text(os.path.join(self.run_dir, PID_FILE), str(os.getpid()), durable=self._durable)
        self._started = True
        if resume:
            self._restore(st)
        else:
            self._fresh()

    def _guard_live(self, task: TaskSpec) -> None:
        problems = check_runnable(task)
        if task.mode == "live":
            caps = self.env.get_capabilities() or {}
            if not caps.get("live", False) or caps.get("provenance") in ("diagnostic", "staging"):
                problems.append("the environment is not a live environment "
                                f"(capabilities: live={caps.get('live')}, provenance={caps.get('provenance')})")
        if problems:
            raise TaskValidationError(problems)

    def _fresh(self) -> None:
        now = self.clock.now()
        self.status, self.started_at = RUNNING, now
        self._agent("set_objective", self.task)
        atomic_write_json(os.path.join(self.run_dir, TASK_FILE), self.task.to_dict(), durable=self._durable)
        unset = [f for f in LIMIT_FIELDS if getattr(self.task.limits, f) <= 0]
        self.log.event("run_started", {"task": self.task.to_dict(), "limits_not_enforced": unset,
                                       "actor_id": self.actor_id, "environment": self._caps(),
                                       "config": dataclasses.asdict(self.cfg)}, consequential=True,
                       trust=Trust.OPERATOR)
        self._committed = self._snapshot(None)
        self._wal_reset()
        self._triggers.append("start")
        self._save_state()

    def _caps(self) -> Dict[str, Any]:
        try:
            return dict(self.env.get_capabilities() or {})
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    def _restore(self, st: Dict[str, Any]) -> None:
        self.status, self.status_reason = st["status"], st.get("status_reason", "")
        self.task = TaskSpec.from_dict(st["task"])
        self.cfg = self._with_measurement(self.task)
        self.log.provenance = self.task.mode
        self.log.redact.set_permissions(self.task.data_permissions)
        self.started_at, self.paused_at = float(st.get("started_at", 0.0)), float(st.get("paused_at", 0.0))
        self.paused_total_s = float(st.get("paused_total_s", 0.0))
        self.iterations, self.cycles = int(st.get("iterations", 0)), int(st.get("cycles", 0))
        self.lifetime = dict(st.get("lifetime", self.lifetime))
        self.approvals = dict(st.get("approvals", {}))
        self._ckpts, self._exports = list(st.get("checkpoints", [])), list(st.get("exports", []))
        self._next_tasks = list(st.get("next_tasks", []))
        self._control_applied = int(st.get("control_applied", 0))
        self._control_results = dict(st.get("control_results", {}))
        self._last_verified = dict(st.get("last_verified", {}))
        self._refused_verified = int(st.get("refused_verified", 0))
        self.obstacles.load_state(st.get("obstacles_open"))
        c = self._committed = dict(st.get("committed", {}))
        self.generation = int(c.get("generation", 0))
        self.cursor = c.get("cursor", "")
        self.pending = PendingOutcomes.from_state(c.get("pending"))
        self.counters = {**_COUNTERS, **c.get("counters", {})}
        self._open_preds = list(c.get("open_predictions", []))
        self._agent_wake_at = c.get("agent_wake_at")
        ckpt = c.get("ckpt")
        if ckpt:
            self._agent("load", ckpt["path"])
            self._last_ckpt_ts = ckpt.get("ts")
        else:
            self._agent("set_objective", self.task)
        replay = self._replay_wal()
        self.log.event("run_resumed", {"status": self.status, "checkpoint": (ckpt or {}).get("id", ""),
                                       "generation": self.generation, "log_repairs": self.log.repaired, **replay},
                       consequential=True)
        self._triggers.append("resume")
        self._save_state()

    # ==========================================================================================================
    # write-ahead journal
    # ==========================================================================================================
    def _wal_write(self, entry: Dict[str, Any]) -> None:
        if self._wal_fh is None:
            fd = os.open(self._wal_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            self._wal_fh = os.fdopen(fd, "a")
        entry["g"] = self.generation
        self._wal_fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        self._wal_fh.flush()
        if self._durable:
            os.fsync(self._wal_fh.fileno())

    def _wal_reset(self) -> None:
        if self._wal_fh is not None:
            self._wal_fh.close()
            self._wal_fh = None
        atomic_write_text(self._wal_path, "", 0o600, self._durable)

    def _replay_wal(self) -> Dict[str, Any]:
        begins: Dict[str, Dict[str, Any]] = {}
        done: List[str] = []
        for e in iter_jsonl(self._wal_path):
            if e.get("g") != self.generation:
                continue
            if e.get("t") == "begin":
                begins[e["action_id"]] = e
                self._apply_delta(e.get("delta", {}))
            elif e.get("t") == "end":
                b = begins.pop(e["action_id"], None)
                if b is not None:
                    self._absorb(Decision.from_dict(b["decision"]), ActionResult.from_dict(e["result"]),
                                 e.get("delta", {}), b.get("preds", []), e.get("approval_id", ""), e.get("kind", ""))
                    done.append(e["action_id"])
        inflight = list(begins)
        for b in begins.values():
            self._recover_inflight(b)
        return {"replayed_results": done, "inflight_recovered": inflight}

    def _recover_inflight(self, b: Dict[str, Any]) -> None:
        """An action was started and the process died before its result was journalled."""
        decision = Decision.from_dict(b["decision"])
        action, now = decision.chosen, self.clock.now()
        kind, approval_id = b.get("kind", "execute"), b.get("approval_id", "")
        if kind == "wait":
            self._waiting = {"decision": decision, "started": float(b.get("started", now)),
                             "until": float(b.get("until", now)), "preds": b.get("preds", [])}
            self._finish_wait()
            return
        caps = self._caps()
        how = ""
        if (caps.get("idempotent_actions") or caps.get("dedup_by_action_id")
                or (caps.get("features") or {}).get("at_most_once_by_action_id")):
            try:
                auth = self.policy.authorize(action, self.task, self._auth_context(decision))
                if auth.verdict == AuthVerdict.ALLOW and self.policy.check_grant(action, auth):
                    result = self.env.execute(action, auth)     # the environment returns the stored result, or runs it once
                    how = "re-issued under the same action id (environment de-duplicates)"
                else:
                    result = self._synthetic(action, ResultStatus.CANCELLED, FailureClass.UNKNOWN,
                                             "interrupted by a process restart; not re-authorized afterwards "
                                             f"({'; '.join(auth.reasons)}); whether it took effect is unknown")
            except Exception as e:  # noqa: BLE001
                result = self._synthetic(action, ResultStatus.CANCELLED, FailureClass.UNKNOWN,
                                         f"interrupted by a process restart; recovery failed: {type(e).__name__}: {e}")
        else:
            result = self._synthetic(action, ResultStatus.CANCELLED, FailureClass.UNKNOWN,
                                     "interrupted by a process restart; the environment does not de-duplicate by "
                                     "action id, so it was not issued again; whether it took effect is unknown")
        self.log.event("inflight_recovered", {"action_id": action.action_id, "status": result.status, "how": how},
                       consequential=True)
        self._complete(decision, result, kind, b.get("preds", []), approval_id, journalled=True)

    # ==========================================================================================================
    # state
    # ==========================================================================================================
    def _snapshot(self, ckpt: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {"generation": self.generation, "cursor": self.cursor, "pending": self.pending.to_state(),
                "counters": dict(self.counters), "open_predictions": list(self._open_preds), "ckpt": ckpt,
                "agent_wake_at": self._agent_wake_at, "ts": self.clock.now(), "seq": self.log.next_seq}

    def _limit_state(self) -> Dict[str, Dict[str, float]]:
        lim, out = self.task.limits, {}
        out["duration_s"] = {"used": self._elapsed(), "limit": lim.duration_s}
        for f, c in _LIMIT_COUNTER.items():
            out[f] = {"used": float(self.counters[c]), "limit": float(getattr(lim, f))}
        return out

    def _state(self) -> Dict[str, Any]:
        w = self._waiting
        return {
            "version": STATE_VERSION, "run_id": self.log.run_id, "actor_id": self.actor_id, "pid": os.getpid(),
            "status": self.status, "status_reason": self.status_reason, "task": self.task.to_dict(),
            "awaiting_operator": self.status in (PAUSED,) + TERMINAL,
            "next_tasks": self._next_tasks, "started_at": self.started_at, "paused_at": self.paused_at,
            "paused_total_s": self.paused_total_s, "updated_at": self.clock.now(), "iterations": self.iterations,
            "cycles": self.cycles, "counters": dict(self.counters), "lifetime": self.lifetime,
            "limits": self._limit_state(), "approvals": self.approvals, "committed": self._committed,
            "live": {"cursor": self.cursor, "pending": self.pending.to_state(), "generation": self.generation},
            "checkpoints": self._ckpts, "exports": self._exports, "control_applied": self._control_applied,
            "control_results": self._control_results, "activity": self._activity, "wake": self._wake_info,
            "last_verified": self._last_verified, "refused_verified": self._refused_verified,
            "obstacles_open": self.obstacles.to_state(), "errors": self._errors[-20:],
            "waiting": None if w is None else {"until": w["until"], "started": w["started"],
                                                "action_id": w["decision"].chosen.action_id},
        }

    def _save_state(self) -> None:
        atomic_write_json(self._state_path, self._state(), durable=self._durable)

    def _set_activity(self, phase: str, detail: str = "") -> None:
        self._activity = {"phase": phase, "detail": self.log.redact.text(detail)[:300], "since": self.clock.now()}

    def _elapsed(self) -> float:
        now = self.clock.now()
        paused = self.paused_total_s + ((now - self.paused_at) if self.status == PAUSED and self.paused_at else 0.0)
        return max(0.0, now - self.started_at - paused)

    def _deadline(self) -> Optional[float]:
        d = self.task.limits.duration_s
        return None if d <= 0 else self.started_at + self.paused_total_s + d

    def _exhausted(self) -> str:
        """Name of the first exhausted limit, or ''. A limit of 0 is "not set": allowed only outside live mode,
        and reported at run start rather than silently treated as unlimited."""
        for name, s in self._limit_state().items():
            if s["limit"] > 0 and s["used"] >= s["limit"] - 1e-9:
                return name
        return ""

    def status_report(self) -> Dict[str, Any]:
        st = self._state()
        try:
            st["agent"] = self.agent.status()
        except Exception as e:  # noqa: BLE001
            st["agent"] = {"error": f"{type(e).__name__}: {e}"}
        return st

    # ==========================================================================================================
    # learner calls (timed for compute accounting)
    # ==========================================================================================================
    def _agent(self, name: str, *args: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return getattr(self.agent, name)(*args)
        finally:
            self._compute_unbooked += time.perf_counter() - t0

    def _agent_try(self, name: str, *args: Any, default: Any = None) -> Any:
        try:
            out = self._agent(name, *args)
            return default if out is None else out
        except Exception as e:  # noqa: BLE001 — recorded; repeated errors pause the run (see _note_error)
            self._note_error("learner", name, e)
            return default

    def _note_error(self, who: str, what: str, e: BaseException) -> None:
        self._consecutive_errors += 1
        self._error_total += 1
        rec = {"ts": self.clock.now(), "who": who, "what": what, "error": f"{type(e).__name__}: {e}"[:500]}
        self._errors.append(rec)
        self._errors = self._errors[-50:]
        self.log.event("error", {**rec, "trace": traceback.format_exc(limit=8)}, consequential=True)

    # ==========================================================================================================
    # operator commands
    # ==========================================================================================================
    def request_stop(self, reason: str = "process signal") -> None:
        self._stop_requested = reason
        self._wake.set()

    def _interrupted(self) -> bool:
        if self._stop_requested:
            return True
        m = self.control.mtime()
        if m != self._control_mtime and len(self.control.read()) > self._control_applied:
            return True
        self._interrupt_checks += 1
        if self._interrupt_checks % 5 == 0:
            try:
                return bool(self.policy.stopped()) and self.status != STOPPED
            except Exception:  # noqa: BLE001 — an unreachable policy service is handled at the next cycle
                return False
        return False

    def _apply_commands(self) -> int:
        cmds = self.control.read()
        self._control_mtime = self.control.mtime()
        new = cmds[self._control_applied:]
        for c in new:
            try:
                ok, detail = self._apply_command(c.get("cmd", ""), c.get("args", {}) or {}, c.get("by", "operator"))
            except Exception as e:  # noqa: BLE001
                ok, detail = False, f"{type(e).__name__}: {e}"
            self._control_applied += 1
            self._control_results[c.get("id", "")] = {"ok": ok, "detail": detail, "ts": self.clock.now(),
                                                      "cmd": c.get("cmd"), "status": self.status}
            for k in list(self._control_results)[:-200]:
                del self._control_results[k]
            self.log.event("operator_command", {"id": c.get("id"), "cmd": c.get("cmd"), "by": c.get("by"),
                                                "args": c.get("args"), "ok": ok, "detail": detail},
                           consequential=True, trust=Trust.OPERATOR)
        if new:
            self._save_state()
        return len(new)

    def _apply_command(self, cmd: str, args: Dict[str, Any], by: str) -> tuple:
        if cmd == "pause":
            if self.status != RUNNING:
                return False, f"not running (status {self.status})"
            self._pause(args.get("reason") or f"paused by {by}")
            return True, "paused; learner state checkpointed"
        if cmd == "resume":
            if self.status != PAUSED:
                return False, (f"status is {self.status}; a finished objective continues only through a newly "
                               "assigned task" if self.status in TERMINAL else f"not paused (status {self.status})")
            self.paused_total_s += max(0.0, self.clock.now() - self.paused_at)
            self.status, self.status_reason, self.paused_at = RUNNING, "", 0.0
            self._consecutive_errors = 0
            self._triggers.append("resume")
            return True, "resumed"
        if cmd == "stop":
            self._halt(STOPPED, args.get("reason") or f"stopped by {by}")
            return True, "stopped; learner state checkpointed"
        if cmd in ("approve", "reject"):
            return self._relay_approval(args.get("approval_id", ""), cmd == "approve", by, args.get("note", ""))
        if cmd == "approval_notice":
            self._triggers.append("approval_notice")
            return True, "approvals will be re-read from the authorization service"
        if cmd == "wake":
            self._triggers.append("operator_wake")
            return True, "woken"
        if cmd == "export_checkpoint":
            return self._export(args.get("dest", ""))
        if cmd == "assign_objective":
            return self._assign(args.get("path", ""), args.get("sha256", ""), by)
        return False, f"unknown command {cmd!r}"

    def _pause(self, reason: str) -> None:
        self._finish_wait()
        self._commit(force=True)
        self.status, self.status_reason, self.paused_at = PAUSED, reason, self.clock.now()
        self._set_activity("paused", reason)
        self.log.event("paused", {"reason": reason, "checkpoint": (self._committed.get("ckpt") or {}).get("id", "")},
                       consequential=True)
        self._journal()
        self._save_state()

    def _halt(self, status: str, reason: str) -> None:
        if self.status in (RUNNING, PAUSED):
            self._finish_wait()
            self._commit(force=True)
        self.status, self.status_reason = status, reason
        self._set_activity(status, reason)
        self.log.event("halted", {"status": status, "reason": reason,
                                  "checkpoint": (self._committed.get("ckpt") or {}).get("id", "")}, consequential=True)
        self._journal()
        self._save_state()

    def _export(self, dest: str) -> tuple:
        dest = dest or os.path.join(self.run_dir, "exports", f"export_{len(self._exports) + 1:04d}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        cid = self._agent("save", dest)
        meta = {"checkpoint_id": cid, "path": dest, "ts": self.clock.now(), "objective_id": self.task.objective_id,
                "provenance": self.task.mode, "counters": dict(self.counters)}
        atomic_write_json(dest.rstrip("/") + ".export.json", {**meta, "task": self.task.to_dict()})
        self._exports.append(meta)
        return True, f"exported checkpoint {cid} to {dest}"

    def _assign(self, path: str, sha256: str, by: str) -> tuple:
        if sha256 and file_sha256(path) != sha256:
            return False, "task file changed after the command was issued"
        task = load_task(path, for_run=True)           # same validation as any task (no strategy, explicit limits)
        self._guard_live(task)
        entry = {"task": task.to_dict(), "path": path, "by": by, "assigned_at": self.clock.now()}
        if task.objective_id == self.task.objective_id:
            self.task = task                            # updated boundaries for the same objective
            self.cfg = self._with_measurement(task)
            self.log.redact.set_permissions(task.data_permissions)
            atomic_write_json(os.path.join(self.run_dir, TASK_FILE), task.to_dict(), durable=self._durable)
            if self.status in TERMINAL and not self._exhausted():
                self.status, self.status_reason = RUNNING, ""
                self._triggers.append("limits_updated")
            return True, "task boundaries updated for the current objective"
        if self.status in TERMINAL:
            self._switch_objective(entry)
            return True, f"objective {task.objective_id} started"
        self._next_tasks.append(entry)
        return True, f"objective {task.objective_id} queued; it starts when the current objective ends"

    def _switch_objective(self, entry: Dict[str, Any]) -> None:
        self.task = TaskSpec.from_dict(entry["task"])
        self.cfg = self._with_measurement(self.task)
        self.log.provenance = self.task.mode
        self.log.redact.set_permissions(self.task.data_permissions)
        self.counters = dict(_COUNTERS)
        self.started_at, self.paused_total_s, self.paused_at = self.clock.now(), 0.0, 0.0
        self.status, self.status_reason = RUNNING, ""
        self._last_verified, self._last_verify_at, self._agent_wake_at = {}, None, None
        self._agent("set_objective", self.task)        # the learner's weights and lifetime context carry over
        atomic_write_json(os.path.join(self.run_dir, TASK_FILE), self.task.to_dict(), durable=self._durable)
        self.log.event("objective_assigned", {"task": self.task.to_dict(), "by": entry.get("by", ""),
                                              "path": entry.get("path", "")}, consequential=True, trust=Trust.OPERATOR)
        self._dirty = True
        self._commit(force=True)
        self._triggers.append("objective_assigned")

    # ==========================================================================================================
    # approvals
    # ==========================================================================================================
    def _relay_approval(self, approval_id: str, approve: bool, by: str, note: str) -> tuple:
        a = self.approvals.get(approval_id)
        if a is None or a["status"] != "pending":
            return False, f"no pending approval {approval_id!r}"
        auth = self.policy.resolve_approval(approval_id, approve, by, note)
        if approve and auth.verdict == AuthVerdict.ALLOW:
            a.update(status="approved", resolved_at=self.clock.now(), by=by, note=note, auth=auth.to_dict())
        elif not approve and auth.verdict != AuthVerdict.ALLOW:
            a.update(status="rejected", resolved_at=self.clock.now(), by=by, note=note,
                     reasons=list(auth.reasons) or a.get("reasons", []))
        else:
            return False, f"the authorization service answered {auth.verdict}: {'; '.join(auth.reasons)}"
        self._triggers.append("approval_resolved")
        return True, f"approval {approval_id} {'approved' if approve else 'rejected'}"

    def _refresh_approvals(self) -> bool:
        """Notice approvals resolved at the authorization service by the operator's own tools."""
        pend = {k: a for k, a in self.approvals.items() if a["status"] == "pending"}
        if not pend:
            return False
        try:
            listed = {str(e.get("approval_id") or e.get("id") or ""): e for e in (self.policy.approvals() or [])}
        except Exception as e:  # noqa: BLE001
            self._note_error("authorization", "approvals", e)
            return False
        changed = False
        for aid, a in pend.items():
            e = listed.get(aid)
            state = str((e or {}).get("status", "pending" if e is not None else "")).lower()
            if e is not None and state in ("pending", "queued", "waiting", ""):
                continue
            decision = Decision.from_dict(a["decision"])
            if state in ("rejected", "denied", "deny"):
                a.update(status="rejected", resolved_at=self.clock.now(), note=str((e or {}).get("note", "")))
                changed = True
                continue
            # approved there, or no longer listed: the authorization service is asked again and is the only judge
            try:
                auth = self.policy.authorize(decision.chosen, self.task, self._auth_context(decision))
            except Exception as ex:  # noqa: BLE001
                self._note_error("authorization", "authorize", ex)
                continue
            if auth.verdict == AuthVerdict.ALLOW:
                a.update(status="approved", resolved_at=self.clock.now(), auth=auth.to_dict())
                changed = True
            elif auth.verdict == AuthVerdict.DENY:
                a.update(status="rejected", resolved_at=self.clock.now(), reasons=list(auth.reasons))
                changed = True
            elif auth.approval_id and auth.approval_id != aid:
                a["approval_id"] = auth.approval_id
                self.approvals[auth.approval_id] = self.approvals.pop(aid)
        return changed

    def _process_approvals(self) -> bool:
        """Execute each approved action exactly once; report each rejected one exactly once."""
        acted = False
        for aid in list(self.approvals):
            a = self.approvals[aid]
            if a["status"] not in ("approved", "rejected"):
                continue
            decision = Decision.from_dict(a["decision"])
            action = decision.chosen
            if a["status"] == "approved":
                auth = AuthorizationDecision.from_dict(a.get("auth") or {"verdict": AuthVerdict.DENY})
                if auth.verdict != AuthVerdict.ALLOW or not self.policy.check_grant(action, auth):
                    result = self._synthetic(action, ResultStatus.DENIED, FailureClass.MISSING_PERMISSION,
                                             "the approval did not yield a grant valid for this action")
                    self._complete(decision, result, "denied", [], aid, new_decision=False)
                else:
                    self._execute(decision, auth, [], approval_id=aid, new_decision=False)
            else:
                # the operator's note stays in the audit trail; no free text from the operator reaches the learner
                result = self._synthetic(action, ResultStatus.DENIED, FailureClass.MISSING_PERMISSION,
                                         f"the operator rejected approval {aid}")
                self._complete(decision, result, "rejected", [], aid, new_decision=False)
            acted = True
        return acted

    # ==========================================================================================================
    # evidence: steps 2-4
    # ==========================================================================================================
    def _ingest(self, obs: Sequence[Observation], via: str) -> None:
        if not obs:
            return
        for o in obs:
            self.log.observation(o, via)
            for q in o.quantities:
                self._metrics[q.name] = q.value
        self._cycle_obs.extend(obs)
        self._agent_try("ingest", list(obs))
        self._dirty = True

    def _measure(self, due: List[PendingItem]) -> List[OutcomeObservation]:
        """Step 4 for matured queries. Goal-relevant queries are asked of the independent verifier first; anything
        the environment measures is unverified whatever it claims."""
        if not due:
            return []
        now = self.clock.now()
        got: Dict[str, OutcomeObservation] = {}
        goal = [i.query for i in due if i.query.goal_relevant]
        if goal:
            try:
                for o in self.verifier.measure(goal) or []:
                    if o.status in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED):
                        o.verified = True
                        got[o.query_id] = o
            except Exception as e:  # noqa: BLE001
                self._note_error("verifier", "measure", e)
        rest = [i.query for i in due if i.query.query_id not in got]
        if rest:
            try:
                for o in self.env.measure(rest) or []:
                    o.verified = False
                    got.setdefault(o.query_id, o)
            except Exception as e:  # noqa: BLE001
                self._note_error("environment", "measure", e)
        out: List[OutcomeObservation] = []
        for item in due:
            o = got.get(item.query.query_id)
            if o is None or o.status == OutcomeStatus.PENDING:
                self.pending.defer(item.query.query_id, now + self.cfg.pending_retry_s)
                continue
            if self.pending.resolve(o) is None:
                continue                                       # duplicate of an event already reconciled
            rid = self.log.event("outcome", {"query": item.query.to_dict(), "observation": o.to_dict(),
                                             "action_id": item.action_id, "decision_id": item.decision_id,
                                             "experiment_id": item.experiment_id}, consequential=True,
                                 trust=Trust.VERIFIER if o.verified else Trust.UNTRUSTED)
            self._resolve_predictions(item, o, rid)
            if item.experiment_id:
                self.log.experiment_outcome(item.experiment_id, item.decision_id, {**o.to_dict(), "outcome_rid": rid})
            out.append(o)
        return out

    def _resolve_predictions(self, item: PendingItem, o: OutcomeObservation, outcome_rid: str) -> None:
        keep = []
        for p in self._open_preds:
            if p["query_id"] == o.query_id or (p["action_id"] and p["action_id"] == item.action_id
                                                 and p["metric"] == item.query.metric):
                self.log.resolution(p["prediction_id"], o.query_id,
                                    {"status": o.status, "value": o.value, "observed_at": o.observed_at,
                                     "verified": o.verified, "outcome_rid": outcome_rid, "metric": item.query.metric},
                                    registered_rid=p.get("rid", ""), predicted_mean=p.get("mean"),
                                    predicted_std=p.get("std"))
            else:
                keep.append(p)
        self._open_preds = keep[-5000:]

    def _verify(self, force: bool = False) -> Optional[VerifiedOutcome]:
        """Objective verification. Only an outcome whose signature the verifier confirms is ever returned."""
        if not self.task.objective.strip() or not self.task.success:
            return None                                        # autonomous learning: no objective to verify
        now = self.clock.now()
        if not force and self._last_verify_at is not None and now - self._last_verify_at < self.cfg.verify_every_s:
            return None
        self._last_verify_at = now
        try:
            vo = self.verifier.verify(self.task)
            ok = (isinstance(vo, VerifiedOutcome) and vo.objective_id == self.task.objective_id
                  and bool(vo.signature) and bool(self.verifier.check_signature(vo)))
        except Exception as e:  # noqa: BLE001
            self._note_error("verifier", "verify", e)
            return None
        if not ok:
            self._refused_verified += 1
            self.log.event("verified_outcome_refused",
                           {"reason": "signature check failed or objective mismatch",
                            "claimed": vo.to_dict() if isinstance(vo, VerifiedOutcome) else repr(vo)[:300]},
                           consequential=True)
            return None
        rid = self.log.event("verified_outcome", {"outcome": vo.to_dict(), "signature_checked": True},
                             consequential=True, trust=Trust.VERIFIER)
        self._last_verified = {"outcome": vo.to_dict(), "rid": rid, "at": now}
        return vo

    # ==========================================================================================================
    # decision and execution: steps 6-12
    # ==========================================================================================================
    def _auth_context(self, decision: Decision) -> Dict[str, Any]:
        return {"run_id": self.log.run_id, "actor_id": self.actor_id, "decision_id": decision.decision_id,
                "objective_id": self.task.objective_id, "mode": self.task.mode, "now": self.clock.now(),
                "counters": dict(self.counters)}

    def _synthetic(self, action: ActionSpec, status: str, failure_class: str, detail: str,
                   observations: Optional[List[Observation]] = None, cost: Optional[Cost] = None) -> ActionResult:
        """A result produced by the runtime for an action that was not handed to an executor."""
        now = self.clock.now()
        return ActionResult(action_id=action.action_id, status=status, failure_class=failure_class,
                            failure_detail=detail, started=now, ended=now, observations=observations or [],
                            cost=cost or Cost())

    def _situation(self) -> Dict[str, Any]:
        lim = self._limit_state()
        return {"observation_ids": [o.obs_id for o in self._cycle_obs][-32:],
                "metrics": dict(list(self._metrics.items())[-64:]),
                "pending_outcomes": len(self.pending),
                "pending_approvals": sum(1 for a in self.approvals.values() if a["status"] == "pending"),
                "verified": {k: (self._last_verified.get("outcome") or {}).get(k) for k in ("status", "value", "target",
                                                                                           "uncertainty")},
                "remaining": {k: (v["limit"] - v["used"]) for k, v in lim.items() if v["limit"] > 0}}

    def _decide_and_act(self) -> bool:
        tools = list(self.env.discover_tools() or [])
        self._tool_effects = {t.name: t.side_effects for t in tools}
        self._set_activity("deciding")
        cands = self._agent_try("propose", tools, self.cfg.candidates_k, default=[])
        queued = {a["signature"]: k for k, a in self.approvals.items() if a["status"] in ("pending", "approved")}
        kept = [c for c in cands if action_signature(c) not in queued]
        if len(kept) != len(cands):
            self.log.event("candidates_withheld", {"n": len(cands) - len(kept),
                                                   "reason": "identical to an action awaiting operator approval",
                                                   "approval_ids": sorted({queued[action_signature(c)] for c in cands
                                                                           if action_signature(c) in queued})})
        if not kept:
            self.log.event("no_candidates", {"proposed": len(cands)})
            return False
        evals = self._agent_try("evaluate", kept, default=[])
        self._drain_agent_records()
        if not evals:
            return False
        decision = self._agent_try("select", evals)
        if decision is None:
            return False
        action = decision.chosen
        # ---- registered before anything is executed (steps 7-8 -> disk) ----
        all_preds = [p for e in evals for p in e.predictions]
        rids = self.log.predictions(all_preds, decision_id=decision.decision_id, chosen_action_id=action.action_id)
        preds = [{"prediction_id": p.prediction_id, "query_id": p.query.query_id, "metric": p.query.metric,
                  "action_id": p.action_id, "decision_id": decision.decision_id, "mean": p.dist.mean,
                  "std": p.dist.std, "rid": rid}
                 for p, rid in zip(all_preds, rids) if p.action_id == action.action_id]
        self.log.decision(decision, objective=self.task.objective, situation=self._situation())
        self.obstacles.on_decision(decision)
        # ---- step 10: structure, then authorization ----
        try:
            errs = list(self.env.validate_action(action) or [])
        except Exception as e:  # noqa: BLE001
            errs = [f"validation failed: {type(e).__name__}: {e}"]
        if errs:
            self._complete(decision, self._synthetic(action, ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS,
                                                     "; ".join(errs)), "invalid", preds)
            return True
        if is_wait(action):
            self._begin_wait(decision, preds)
            return False
        over = self._would_exceed(action)
        if over:
            self._complete(decision, self._synthetic(action, ResultStatus.FAILED, FailureClass.RESOURCE_LIMIT, over),
                           "over_limit", preds)
            return True
        try:
            auth = self.policy.authorize(action, self.task, self._auth_context(decision))
        except Exception as e:  # noqa: BLE001 — fail closed
            self._note_error("authorization", "authorize", e)
            auth = AuthorizationDecision(AuthVerdict.DENY, action.content_hash(),
                                         [f"authorization service error: {type(e).__name__}"])
        self.log.event("authorization", {"decision_id": decision.decision_id, "action_id": action.action_id,
                                         "verdict": auth.verdict, "reasons": auth.reasons,
                                         "policy_ids": auth.policy_ids, "approval_id": auth.approval_id,
                                         "expires_at": auth.expires_at}, consequential=True)
        if auth.verdict == AuthVerdict.NEEDS_APPROVAL:
            aid = auth.approval_id or new_id("apr")
            self._complete(decision, self._synthetic(action, ResultStatus.NEEDS_APPROVAL, FailureClass.NONE,
                                                     "queued for operator approval: " + "; ".join(auth.reasons)),
                           "queued", preds, aid)
            return True
        if auth.verdict != AuthVerdict.ALLOW or not self.policy.check_grant(action, auth):
            detail = "; ".join(auth.reasons) or ("authorization refused" if auth.verdict != AuthVerdict.ALLOW
                                                 else "the grant does not match this action")
            self._complete(decision, self._synthetic(action, ResultStatus.DENIED, FailureClass.MISSING_PERMISSION,
                                                     detail), "denied", preds)
            return True
        self._execute(decision, auth, preds, new_decision=True)
        return True

    def _would_exceed(self, action: ActionSpec) -> str:
        lim = self.task.limits
        est = action.est_cost.money
        if lim.money > 0 and est > 0 and self.counters["money"] + est > lim.money + 1e-9:
            return (f"estimated cost {est:g} exceeds the remaining budget "
                    f"{lim.money - self.counters['money']:g}; not executed")
        if (lim.max_external_actions > 0 and self._is_external(action)
                and self.counters["external_actions"] >= lim.max_external_actions):
            return "the limit on external actions is reached; not executed"
        return ""

    def _is_external(self, action: ActionSpec) -> bool:
        if action.modality == Modality.INTERNAL:
            return False
        return self._tool_effects.get(action.operation, "external_write") not in ("none", "local")

    def _execute(self, decision: Decision, auth: AuthorizationDecision, preds: List[Dict[str, Any]], *,
                 approval_id: str = "", new_decision: bool = True) -> None:
        action = decision.chosen
        # the journal entry is durable before the executor is called; the grant itself is never written to disk
        self._begin(decision, "execute", preds, approval_id, new_decision)
        self._set_activity("executing", action.render())
        self.log.event("action_start", {"decision_id": decision.decision_id, "action_id": action.action_id,
                                        "operation": action.operation, "modality": action.modality,
                                        "approval_id": approval_id}, consequential=True)
        try:
            result = self.env.execute(action, auth)
        except Exception as e:  # noqa: BLE001 — an executor error is an observation for the learner, not a crash
            self._note_error("environment", "execute", e)
            result = self._synthetic(action, ResultStatus.FAILED, FailureClass.UNKNOWN,
                                     f"executor error: {type(e).__name__}: {e}")
        self._complete(decision, result, "execute", preds, approval_id, journalled=True)

    def _begin(self, decision: Decision, kind: str, preds: List[Dict[str, Any]], approval_id: str = "",
               new_decision: bool = True, **extra: Any) -> None:
        """Journal the start of a decision's handling; the decision is counted from this moment."""
        delta = {"decisions": int(new_decision), "waits": int(kind == "wait")}
        self._wal_write({"t": "begin", "kind": kind, "action_id": decision.chosen.action_id,
                         "decision": decision.to_dict(), "preds": preds, "approval_id": approval_id,
                         "started": self.clock.now(), "delta": delta, **extra})
        self._apply_delta(delta)

    def _apply_delta(self, delta: Dict[str, Any]) -> None:
        for k, v in delta.items():
            self.counters[k] = self.counters.get(k, 0) + v
        self.lifetime["decisions"] += delta.get("decisions", 0)
        self.lifetime["actions_executed"] += delta.get("actions_executed", 0)

    def _begin_wait(self, decision: Decision, preds: List[Dict[str, Any]]) -> None:
        """A wait is a decision to do nothing: the runtime sleeps; no executor is involved and nothing is spent."""
        a, now = decision.chosen.arguments, self.clock.now()
        seconds = float(a.get("seconds", a.get("duration_s", 0.0)) or 0.0)
        self._begin(decision, "wait", preds, until=now + seconds)
        self._waiting = {"decision": decision, "started": now, "until": now + seconds, "preds": preds}
        self._agent_wake_at = now + seconds

    def _finish_wait(self) -> None:
        w, self._waiting = self._waiting, None
        if w is None:
            return
        now = self.clock.now()
        elapsed = max(0.0, now - w["started"])
        action = w["decision"].chosen
        obs = Observation(obs_id=new_id("obs"), ts=now, kind="tool_result", text=f"waited {elapsed:.0f} seconds",
                          provenance=Provenance(source="runtime", retrieved_at=now, trust=Trust.SELF),
                          parent_action_id=action.action_id)
        result = ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=w["started"], ended=now,
                              observations=[obs], cost=Cost(seconds=elapsed), verified=True)
        self._complete(w["decision"], result, "wait", w["preds"], journalled=True)

    def _complete(self, decision: Decision, result: ActionResult, kind: str, preds: List[Dict[str, Any]],
                  approval_id: str = "", *, journalled: bool = False, new_decision: bool = True) -> None:
        """Step 12 for every decision, executed or not: journal, account, log, hand the result to the learner."""
        action = decision.chosen
        if not journalled:
            self._begin(decision, kind, preds, approval_id, new_decision)
        delta: Dict[str, Any] = {"compute_s": float(result.cost.compute_s) + self._compute_unbooked}
        self._compute_unbooked = 0.0
        if kind == "execute":
            ran = result.status not in (ResultStatus.DENIED, ResultStatus.NEEDS_APPROVAL)
            delta["actions_executed"] = int(ran)
            delta["external_actions"] = int(ran and self._is_external(action))
            delta["money"] = float(result.cost.money)
        if is_obstacle(result):
            delta["failures"] = 1
        self._wal_write({"t": "end", "kind": kind, "action_id": action.action_id, "result": result.to_dict(),
                         "delta": delta, "approval_id": approval_id})
        if kind == "execute" and self.cfg.record_at_policy:
            try:
                self.policy.record(action, result)
            except Exception as e:  # noqa: BLE001
                self._note_error("authorization", "record", e)
        self.log.event("action_result",
                       {"decision_id": decision.decision_id, "action_id": action.action_id, "kind": kind,
                        "operation": action.operation, "status": result.status,
                        "failure_class": result.failure_class, "failure_detail": result.failure_detail,
                        "started": result.started, "ended": result.ended, "cost": result.cost.to_dict(),
                        "verified": result.verified, "retries": result.retries, "artifacts": result.artifacts,
                        "n_ops": len(result.ops), "observation_ids": [o.obs_id for o in result.observations],
                        "expected_outcomes": [q.to_dict() for q in result.expected_outcomes],
                        "approval_id": approval_id}, consequential=True)
        for o in result.observations:
            self.log.observation(o, "action")
            for q in o.quantities:
                self._metrics[q.name] = q.value
        self._absorb(decision, result, delta, preds, approval_id, kind)
        self._last_result_ok = result.status == ResultStatus.OK
        if is_obstacle(result):
            self.obstacles.open(action, result, self.agent)
        self.obstacles.on_result(action, result)
        self._save_state()

    def _absorb(self, decision: Decision, result: ActionResult, delta: Dict[str, Any], preds: List[Dict[str, Any]],
                approval_id: str, kind: str) -> None:
        """The part of step 12 that is repeated when the journal is replayed after a restart."""
        self._apply_delta(delta)
        for q in result.expected_outcomes:
            self.pending.add(q, action_id=result.action_id, decision_id=decision.decision_id,
                             experiment_id=decision.experiment_id or decision.chosen.experiment_id)
        known = {p["prediction_id"] for p in self._open_preds}
        self._open_preds.extend(p for p in preds if p["prediction_id"] not in known)
        if kind == "queued" and approval_id and approval_id not in self.approvals:
            action = decision.chosen
            self.approvals[approval_id] = {
                "approval_id": approval_id, "status": "pending", "decision": decision.to_dict(),
                "signature": action_signature(action), "queued_at": result.started,
                "preview": self.log.redact.text(action.render())[:4000], "operation": action.operation,
                "reasons": [result.failure_detail]}
        elif approval_id and approval_id in self.approvals and kind != "queued":
            self.approvals[approval_id]["status"] = "reported" if kind in ("rejected", "denied") else "executed"
            self.approvals[approval_id]["result_status"] = result.status
            self.approvals[approval_id].pop("auth", None)
        self._agent_try("record_result", decision, result)
        self._dirty = True

    def _drain_agent_records(self) -> None:
        """Experiment records registered by the learner (optional ``drain_experiments`` hook)."""
        fn = getattr(self.agent, "drain_experiments", None)
        if callable(fn):
            try:
                for rec in fn() or []:
                    self.log.experiment(rec)
            except Exception as e:  # noqa: BLE001
                self._note_error("learner", "drain_experiments", e)

    # ==========================================================================================================
    # step 16: commit
    # ==========================================================================================================
    def _commit(self, force: bool = False) -> bool:
        """Checkpoint the learner and commit the evidence position with it. Returns True when a checkpoint was taken."""
        self.counters["compute_s"] += self._compute_unbooked
        self._compute_unbooked = 0.0
        now = self.clock.now()
        due = force or (self._dirty and (self.cfg.checkpoint_every_s <= 0 or self._last_ckpt_ts is None
                                         or now - self._last_ckpt_ts >= self.cfg.checkpoint_every_s))
        if not due:
            self._save_state()
            return False
        path = os.path.join(self.run_dir, RunFiles.CHECKPOINTS, f"ckpt_{self.generation + 1:08d}")
        try:
            if os.path.isdir(path):                          # left behind by a commit that never completed
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
            cid = self._agent("save", path)
        except Exception as e:  # noqa: BLE001
            self._note_error("learner", "save", e)
            self._save_state()
            return False
        self.counters["compute_s"] += self._compute_unbooked
        self._compute_unbooked = 0.0
        self.generation += 1
        ckpt = {"id": cid, "path": path, "ts": now, "generation": self.generation, "cycle": self.cycles}
        self._committed = self._snapshot(ckpt)
        self._ckpts.append(ckpt)
        self._save_state()                                   # the commit point
        self._wal_reset()
        self._dirty, self._last_ckpt_ts = False, now
        self.log.event("checkpoint", ckpt, consequential=True)
        keep = self.cfg.keep_checkpoints
        while keep > 0 and len(self._ckpts) > keep:
            old = self._ckpts.pop(0)
            for p in (old["path"],):
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.exists(p):
                    os.remove(p)
        return True

    def _journal(self) -> None:
        self._last_journal_at = self.clock.now()
        try:
            from .journal import write_journal
            self.log.sync()
            write_journal(self.run_dir, state=self._state())
        except Exception as e:  # noqa: BLE001 — the journal is derived; failing to render it must not stop the run
            self._note_error("runtime", "journal", e)

    def _finish(self, status: str, reason: str) -> None:
        self._finish_wait()
        self._commit(force=True)
        self.status, self.status_reason = status, reason
        self._set_activity("finished", reason)
        self.lifetime["objectives"].append({"objective_id": self.task.objective_id, "status": status,
                                            "reason": reason, "at": self.clock.now(),
                                            "decisions": self.counters["decisions"],
                                            "verified": (self._last_verified.get("outcome") or {}).get("value")})
        self.log.event("objective_finished", {"objective_id": self.task.objective_id, "status": status,
                                              "reason": reason, "counters": dict(self.counters),
                                              "limits": self._limit_state(),
                                              "checkpoint": (self._committed.get("ckpt") or {}).get("id", ""),
                                              "verified_rid": self._last_verified.get("rid", "")}, consequential=True)
        self._journal()
        self._save_state()
        if self._next_tasks:
            self._switch_objective(self._next_tasks.pop(0))
            self._save_state()

    # ==========================================================================================================
    # the loop
    # ==========================================================================================================
    def step(self) -> Dict[str, Any]:
        self.start()
        self.iterations += 1
        rep: Dict[str, Any] = {"iteration": self.iterations, "cycle": False, "reasons": [], "acted": False,
                               "slept": "", "idle": False}
        self._apply_commands()
        if self._stop_requested and self.status != STOPPED:
            self._halt(STOPPED, self._stop_requested)
        elif self.status in (RUNNING, PAUSED) and self._policy_stopped():
            self._halt(STOPPED, "emergency stop raised at the authorization service")
        if self.status != RUNNING:
            rep["status"] = self.status
            if self.status != STOPPED and self.sleeper.blocking:
                rep["slept"] = self.sleeper.sleep_until(None, self._interrupted)
            return rep
        if self._consecutive_errors >= self.cfg.max_consecutive_errors:
            self._pause(f"paused after {self._consecutive_errors} consecutive errors; last: "
                        f"{(self._errors[-1] if self._errors else {}).get('error', '')}")
            rep["status"] = self.status
            return rep

        now = self.clock.now()
        reasons, self._triggers = list(self._triggers), []
        if self._agent_wake_at is not None and now >= self._agent_wake_at:
            reasons.append("agent_wake")
        try:
            events, cursor = self.env.poll_events(self.cursor)                     # step 3 (read)
        except Exception as e:  # noqa: BLE001
            self._note_error("environment", "poll_events", e)
            self._backoff(rep)
            return rep
        if events:
            reasons.append("events")
        due = self.pending.due(now)
        if due:
            reasons.append("outcomes_due")
        if self._refresh_approvals() or any(a["status"] in ("approved", "rejected") for a in self.approvals.values()):
            reasons.append("approval_resolved")
        exhausted = self._exhausted()
        if exhausted:
            reasons.append(f"limit:{exhausted}")
        if not reasons:
            self._sleep(rep)                                                       # step 17: nothing to do
            return rep

        # -------------------------------------------------------------------------------- a cycle with work
        self.cycles += 1
        errors_before = self._error_total
        rep.update(cycle=True, reasons=reasons)
        self._finish_wait()
        self._agent_wake_at = None
        self._cycle_obs = []
        self._set_activity("observing", ", ".join(reasons))
        try:
            self._ingest(list(self.env.observe() or []), "observe")                # step 2
        except Exception as e:  # noqa: BLE001
            self._note_error("environment", "observe", e)
        self._ingest(list(events), "poll")                                         # step 3 (deliver)
        self.cursor = cursor
        outcomes = self._measure(due)                                              # step 4
        verified = self._verify(force=bool(exhausted))
        if outcomes or verified is not None:
            info = self._agent_try("reconcile", outcomes, verified, default={})
            self.log.event("reconciled", {"n_outcomes": len(outcomes), "query_ids": [o.query_id for o in outcomes],
                                          "with_verified_outcome": verified is not None,
                                          "learner_report": info if isinstance(info, dict) else {}})
            self._dirty = True
        self._drain_agent_records()
        stops = set(self.task.stop_conditions) or {"verified_failure"}
        terminal = ""
        if verified is not None and verified.status == GoalStatus.ACHIEVED:
            terminal = COMPLETED
        elif verified is not None and verified.status == GoalStatus.FAILED and "verified_failure" in stops:
            terminal = FAILED
        self._agent_try("update_beliefs")                                          # step 5
        acted = False
        if not terminal and not exhausted:
            try:
                acted = self._process_approvals()
                if not self._exhausted() and not self._policy_stopped():
                    acted = self._decide_and_act() or acted                        # steps 6-12
            except Exception as e:  # noqa: BLE001
                self._note_error("runtime", "decide_and_act", e)
        rep["acted"] = acted
        self._set_activity("learning")
        rewards = self._agent_try("internal_rewards", default={})                  # step 13
        if rewards:
            self.log.event("internal_rewards", rewards if isinstance(rewards, dict) else {"value": rewards})
        updates = self._agent_try("learn", self.cfg.learn_budget_s, default=[])    # step 14
        for u in updates:
            self.log.learning_update(u)
        self.obstacles.on_learning(updates)
        metas = self._agent_try("meta_learn", self.cfg.meta_budget_s, default=[])  # step 15
        for m in metas:
            self.log.meta_update(m)
        if updates or metas:
            self._dirty = True
        for hook in self.hooks["after_cycle"]:
            hook(self)
        if terminal:                                                               # step 16
            vo = self._last_verified.get("outcome", {})
            self._finish(terminal, f"verifier status {vo.get('status')}: value {vo.get('value')} "
                                   f"against target {vo.get('target')}")
        elif exhausted:
            self._finish(EXHAUSTED, f"limit {exhausted} exhausted")
        else:
            self._commit()
            if (self.cycles % max(1, self.cfg.journal_every) == 0
                    and self.clock.now() - self._last_journal_at >= self.cfg.journal_min_interval_s):
                self._journal()
        rep["status"] = self.status
        if self._error_total == errors_before:
            self._consecutive_errors = 0                     # a clean cycle ends an error streak
        if self.status != RUNNING:                                                 # step 17
            return rep
        if acted:
            self._triggers.append("action_completed")
            if self.cfg.min_cycle_interval_s > 0 and not self._last_result_ok:
                self.sleeper.sleep_until(self.clock.now() + self.cfg.min_cycle_interval_s, self._interrupted)
        elif self._exhausted():
            self._triggers.append("limit_check")
        if self._triggers:
            return rep
        if self._waiting is None:
            nw = self._agent_try("next_wake")
            self._agent_wake_at = float(nw) if nw is not None else None
        self._sleep(rep)
        return rep

    def _policy_stopped(self) -> bool:
        try:
            return bool(self.policy.stopped())
        except Exception as e:  # noqa: BLE001 — an unreachable authorization service authorizes nothing anyway
            self._note_error("authorization", "stopped", e)
            return False

    def _backoff(self, rep: Dict[str, Any]) -> None:
        self._set_activity("backoff", (self._errors[-1] if self._errors else {}).get("error", ""))
        self._save_state()
        rep["slept"] = self.sleeper.sleep_until(self.clock.now() + self.cfg.error_backoff_s, self._interrupted)

    def _sleep(self, rep: Dict[str, Any]) -> None:
        now = self.clock.now()
        pend_approvals = any(a["status"] == "pending" for a in self.approvals.values())
        t, why = plan_wake({
            "agent": self._agent_wake_at,
            "outcome": self.pending.next_ready(),
            "poll": now + self.cfg.evidence_poll_s if self.cfg.evidence_poll_s > 0 else None,
            "approval_check": now + self.cfg.approval_poll_s if pend_approvals and self.cfg.approval_poll_s > 0 else None,
            "deadline": self._deadline(),
        })
        if t is not None:
            t = max(t, now + self.cfg.min_idle_sleep_s)
        elif not self.sleeper.blocking:
            rep["idle"] = True                               # nothing is scheduled and nothing can interrupt a simulated sleep
            self._set_activity("idle", "nothing scheduled")
            self._save_state()
            return
        self._wake_info = {"at": t, "reason": why or "operator command", "since": now}
        self._set_activity("sleeping", f"until {t if t is not None else 'an operator command'} ({why or 'command'})")
        self._save_state()
        self.log.event("sleep", {"until": t, "reason": why or "operator command",
                                 "pending_outcomes": len(self.pending)})
        rep["slept"] = self.sleeper.sleep_until(t, self._interrupted)
        rep["wake_reason"] = why

    def run(self, max_iterations: Optional[int] = None, block: Optional[bool] = None) -> str:
        """Run until stopped. With ``block=False`` (the default for simulated clocks) also return when the run is
        paused, finished, or has nothing scheduled; a real-time run stays alive and waits for the operator."""
        self.start()
        block = self.sleeper.blocking if block is None else block
        n = 0
        while True:
            rep = self.step()
            n += 1
            if self.status == STOPPED:
                break
            if not block and (self.status != RUNNING or rep.get("idle")):
                break
            if max_iterations is not None and n >= max_iterations:
                break
        self.log.sync()
        return "idle" if self.status == RUNNING and rep.get("idle") else self.status

    def close(self) -> None:
        if self._wal_fh is not None:
            self._wal_fh.close()
            self._wal_fh = None
        self.log.close()
