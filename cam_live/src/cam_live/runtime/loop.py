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
committed together with a learner checkpoint; everything since that commit is in a write-ahead journal
(``wal.jsonl``): a decision is journalled before its decision record is written, an execution before the executor
is called, a result and a matured outcome before they are logged, and a wait in progress is carried into the journal
of the next commit. A restarted process repairs a torn journal line, loads the checkpoint, re-delivers the journalled
results and outcomes (writing any record the crash prevented, never a second one), closes a decision that was never
handled, re-reads the evidence after the committed cursor, and resolves an action that was in flight through the
environment's de-duplication by action id — it is never executed a second time. Without de-duplication the action
is reported with unknown effect and booked at its estimated cost.

Private records. The journal, the raw decision of every pending approval and the exact resume state are sealed
(``runtime.private``); ``state.json`` itself holds only redacted, human-readable fields next to the sealed part.

What the learner receives. ``agent.set_objective`` gets :func:`runtime.tasks.learner_view` of the task — never the
success definition, exclusion rules, evidence source or measurement protocol, which only the verifier uses.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import math
import os
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..storage import DEFAULT_MIN_FREE_GB, DEFAULT_WARN_FREE_GB, DiskGuard, DiskSpaceLow, dir_bytes
from ..contracts import (ActionResult, ActionSpec, AgentAPI, AuthorizationDecision, AuthVerdict, Clock, Cost, Decision,
                         EnvironmentAdapter, ExperimentRecord, FailureClass, GoalStatus, Modality, Observation,
                         OutcomeObservation, OutcomeQuery, OutcomeStatus, OutcomeVerifier, PolicyEngine,
                         PredictionRecord, Provenance, ResultStatus, RunFiles, TaskSpec, Trust, VerifiedOutcome,
                         new_id)
from .control import PID_FILE, PRIVILEGED, ControlChannel
from .logs import (Redactor, RunLog, RunReader, _repair_torn_tail, atomic_write_json, atomic_write_text, iter_jsonl,
                   iter_since, load_salt, read_json)
from .obstacles import ObstacleTracker, is_obstacle
from .private import PrivateBox, SealError
from .scheduler import PendingItem, PendingOutcomes, SimSleeper, WallSleeper, make_sleeper, plan_wake
from .tasks import (LIMIT_FIELDS, TaskFileChanged, TaskValidationError, check_runnable, learner_view, limit_problems,
                    load_task, load_task_bytes, read_task_file)

STATE_VERSION = 2
RUNNING, PAUSED, STOPPED = "running", "paused", "stopped"
COMPLETED, FAILED, EXHAUSTED = "completed", "failed", "exhausted"
TERMINAL = (COMPLETED, FAILED, EXHAUSTED)
WAL_FILE, TASK_FILE = "wal.jsonl", "task.json"
_COUNTERS = {"decisions": 0, "external_actions": 0, "actions_executed": 0, "waits": 0, "failures": 0,
             "money": 0.0, "compute_s": 0.0}
_LIMIT_COUNTER = {"money": "money", "compute_s": "compute_s", "max_external_actions": "external_actions",
                  "max_decisions": "decisions"}
#: the parts of a task the learner never receives (``tasks.learner_view``); named in ``run_started``
WITHHELD_FROM_LEARNER = ("success (except metric and target)", "measurement", "resources: credential names, test "
                         "accounts, operator authorization record, evidence settings")
#: learner calls that do no heavy (GPU) work: no shared-lock guard around them
_NO_GPU_CALLS = ("status", "next_wake", "drain_experiments", "set_action_validator")
MAX_LEARNER_QUERIES_PER_DECISION = 16
MAX_LEARNER_QUERIES_PENDING = 2000
MAX_LEARNER_HORIZON_S = 10 * 365 * 86400.0
#: results of these kinds mean the chosen action (or the wait) actually happened
_RAN = (ResultStatus.OK, ResultStatus.FAILED, ResultStatus.TIMEOUT)


def _finite(v: Any) -> Optional[float]:
    """float(v) when v is a real finite number (bool excluded), else None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


@dataclass
class RuntimeConfig:
    """Settings of the loop itself; read from the ``runtime`` section of the root configuration."""
    candidates_k: int = 16               # candidates requested from agent.propose
    learn_budget_s: float = 30.0
    meta_budget_s: float = 30.0
    checkpoint_every_s: float = 0.0      # clock seconds between learner checkpoints; 0 = after every cycle with new experience
    keep_checkpoints: int = 2            # 0 = keep all; optimizer state is kept in the newest one only
    min_free_gb: float = DEFAULT_MIN_FREE_GB    # storage guard (config section ``storage``): below this no checkpoint
    warn_free_gb: float = DEFAULT_WARN_FREE_GB  # is written and the run pauses with a resource obstacle; below this
                                                # a warning is logged
    drain_after_exhaustion: bool = False # S-1: when the decision budget is used up with the objective pending, keep
                                         # observing (no proposal, no execution) until the verifier's verdict is
                                         # terminal or the deadline passes, so the experience gets its outcomes
    settle_after_finish_s: float = 7 * 86400.0  # INT3B: when an objective ends (signed verdict achieved / failed, or a
                                         # limit other than the duration used up), outcomes still pending are
                                         # measured for at most this many clock seconds — never past the deadline;
                                         # nothing is proposed or executed meanwhile — so the objective's last
                                         # decisions get their consequences in the same lifetime. Outcomes that
                                         # cannot mature in that window (and those of a stopped run) are carried:
                                         # they stay in the persisted pending-outcome queue (state.json) and are
                                         # measured if the run continues. 0 = finish at once (carry everything).
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
        d = dict(cfg.section("runtime")) if hasattr(cfg, "section") else dict(cfg)
        if hasattr(cfg, "section"):                    # the storage thresholds live in their own section
            from .doctor import storage_thresholds     # (the environment can only make them stricter)
            mn, wn = storage_thresholds(cfg)
            d.setdefault("min_free_gb", mn)
            d.setdefault("warn_free_gb", wn)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def look_grant(policy: Any, action: ActionSpec, auth: AuthorizationDecision) -> bool:
    """Verify a grant *without using it up* (M-int-6). The executor that runs the action consumes the grant
    immediately before execution (``harness.auth.PolicyService.check_grant``: single use, one defined point of
    consumption); a runtime-side check that consumed it would make the environment refuse the action ("the grant
    does not verify"). A ``PolicyEngine`` whose ``check_grant`` takes no ``consume`` keyword (the contract's
    two-argument form, e.g. a remote client whose service already forces look-only for the learner uid, or a
    diagnostic policy without single use) is asked with the plain call."""
    fn = policy.check_grant
    try:                              # only a signature that names ``consume`` (a generic ``**kwargs`` proxy may
        takes = "consume" in inspect.signature(fn).parameters   # forward to a two-argument engine)
    except (TypeError, ValueError):
        takes = False
    return bool(fn(action, auth, consume=False)) if takes else bool(fn(action, auth))


def _logged_args(args: Any) -> Any:
    """A command's arguments as the run log records them: a relayed grant (G2I-15) is named, not copied."""
    if isinstance(args, dict) and "grant" in args:
        g = args.get("grant") if isinstance(args.get("grant"), dict) else {}
        return {**args, "grant": {"verdict": g.get("verdict", ""), "approval_id": g.get("approval_id", ""),
                                  "token": "(signed grant of the authorization service; not logged)"}}
    return args


def action_signature(a: ActionSpec) -> str:
    """Identity of *what an action does* (independent of its id), used to recognise a re-proposal."""
    def core(x: ActionSpec) -> Dict[str, Any]:
        return {"op": x.operation, "mod": x.modality, "target": x.target, "args": x.arguments,
                "steps": [core(s) for s in x.steps]}
    return hashlib.sha256(json.dumps(core(a), sort_keys=True, default=str).encode()).hexdigest()


def refusal_failure_class(auth: AuthorizationDecision) -> str:
    """G2I-14: the failure class of a refusal by the authorization service — a budget / rate-window refusal (every
    reason a ``resource: ...`` limit, or only limit rules ``L-...``) is ``RESOURCE_LIMIT``, a platform rule
    ``PLATFORM_RESTRICTION``, anything else ``MISSING_PERMISSION`` (``harness.failures.refusal_class``)."""
    from ..harness.failures import refusal_class
    if auth.verdict == AuthVerdict.ALLOW:
        return FailureClass.MISSING_PERMISSION          # an ALLOW whose grant does not verify
    pids = [str(p) for p in (auth.policy_ids or [])]
    if pids and all(p.startswith("L-") for p in pids):
        return FailureClass.RESOURCE_LIMIT
    return refusal_class(list(auth.reasons or []), pids)


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
        self.guard = DiskGuard(self.run_dir, float(self.cfg.min_free_gb), float(self.cfg.warn_free_gb))
        self._storage_warned = False
        self._draining = False
        self._settle: Optional[Dict[str, Any]] = None          # the ended objective whose outcomes are being measured
        self._cmd_in_progress: Optional[Dict[str, Any]] = None
        self.learner_guard: Optional[Callable[[str], None]] = None    # called before every learner call (G2R-9)
        self.learner_timing: Optional[Callable[[str, float], None]] = None
        self.box = PrivateBox(self.run_dir)
        self.log = RunLog(self.run_dir, clock, provenance=task.mode, durable=self._durable,
                          redactor=Redactor.from_config(self.cfg.redaction, load_salt(self.run_dir),
                                                        task.data_permissions, keep_ids=self._keep_ids(task)))
        self.obstacles = ObstacleTracker(self.log, clock)
        self.pending = PendingOutcomes()
        self.cursor = ""
        self.status, self.status_reason = "new", ""
        self.started_at = self.paused_at = 0.0
        self.paused_total_s = 0.0
        self.iterations = self.cycles = self.generation = 0
        self.counters: Dict[str, Any] = dict(_COUNTERS)
        self.lifetime: Dict[str, Any] = {"decisions": 0, "actions_executed": 0, "objectives": []}
        self.approvals: Dict[str, Dict[str, Any]] = {}           # redacted rows (state.json)
        self._approval_raw: Dict[str, Dict[str, Any]] = {}       # approval id -> exact decision / preds / auth (sealed)
        # before_cycle hooks run at a decision boundary, before a cycle with work; one returning False skips the
        # cycle (e.g. runtime.gpu_cycle: the shared GPU lock could not be taken because a command stopped the run)
        self.hooks: Dict[str, List[Callable[["LiveRuntime"], Any]]] = {"after_cycle": [], "before_cycle": []}
        self.gpu: Optional[Dict[str, Any]] = None                # runtime.gpu_cycle lease status (state.json)
        self._committed: Dict[str, Any] = {}
        self._ckpts: List[Dict[str, Any]] = []
        self._exports: List[Dict[str, Any]] = []
        self._next_tasks: List[Dict[str, Any]] = []
        self._open_preds: List[Dict[str, Any]] = []
        self._control_applied = 0
        self._control_results: Dict[str, Dict[str, Any]] = {}
        self._control_mtime = -1.0
        self._seen_cmds: List[str] = []
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
        self._tools: Dict[str, Any] = {}
        self._tool_effects: Dict[str, str] = {}
        self._cycle_obs: List[Observation] = []
        self._metrics: Dict[str, float] = {}
        self._interrupt_checks = 0
        self._last_result_ok = True
        self._last_journal_at = float("-inf")
        self._redeliver: List[OutcomeObservation] = []          # outcomes restored from the journal after a restart
        self._wal_known: set = set()                             # action ids with a begin entry in this generation
        self._red_cache: Dict[str, str] = {}
        self._wal_fh = None
        self._wal_path = os.path.join(self.run_dir, WAL_FILE)
        self._state_path = os.path.join(self.run_dir, RunFiles.STATE)

    @staticmethod
    def _keep_ids(task: TaskSpec) -> List[str]:
        """Identifiers the operator configured (guild, channels, application): they stay readable in the logs."""
        r = task.resources if isinstance(task.resources, dict) else {}
        d = r.get("discord") if isinstance(r.get("discord"), dict) else {}
        out = [r.get("discord_guild_id"), d.get("guild_id"), d.get("application_id"), d.get("metric_channel_id")]
        out += list((d.get("channels") or {}).values()) if isinstance(d.get("channels"), dict) else []
        return [str(x) for x in out if x not in (None, "") and not str(x).upper().startswith("REQUIRED")]

    def _with_measurement(self, task: TaskSpec) -> RuntimeConfig:
        """The operator's measurement protocol (task file) sets the verification and evidence-poll intervals."""
        m, over = task.measurement or {}, {}
        for key, field_name in (("verify_interval_s", "verify_every_s"), ("evidence_poll_interval_s", "evidence_poll_s")):
            v = _finite(m.get(key))
            if v is not None and v >= 0:
                over[field_name] = v
        return dataclasses.replace(self._base_cfg, **over)

    def _use_task(self, task: TaskSpec) -> None:
        """Make ``task`` the current task of the runtime (configuration, provenance, redaction)."""
        self.task = task
        self.cfg = self._with_measurement(task)
        self.log.provenance = task.mode
        self.log.redact.set_permissions(task.data_permissions)
        self.log.redact.set_keep_ids(self._keep_ids(task))

    @staticmethod
    def _view(task: TaskSpec) -> Dict[str, Any]:
        """``learner_view(task)`` as plain JSON (a limit that is not a finite number is written as text)."""
        d = learner_view(task).to_dict()
        for k, v in list((d.get("limits") or {}).items()):
            if isinstance(v, float) and not math.isfinite(v):
                d["limits"][k] = f"invalid: {v!r}"
        return d

    def _write_task_file(self) -> None:
        """``task.json``: what the learner received and the hash of the operator's task (the task file itself is
        ``task.yaml``; the full task is kept in the sealed state)."""
        atomic_write_json(os.path.join(self.run_dir, TASK_FILE),
                          {"learner_view": self._view(self.task), "task_sha256": self.task.content_hash(),
                           "withheld_from_learner": list(WITHHELD_FROM_LEARNER)}, durable=self._durable)

    # ==========================================================================================================
    # start / resume
    # ==========================================================================================================
    def start(self) -> None:
        if self._started:
            return
        st = read_json(self._state_path)
        resume = isinstance(st, dict) and st.get("version") in (1, STATE_VERSION) and bool(st.get("status"))
        inner = self._inner(st) if resume else {}
        self._guard_live(TaskSpec.from_dict(inner["task"]) if resume else self.task)
        atomic_write_text(os.path.join(self.run_dir, PID_FILE), str(os.getpid()), durable=self._durable)
        self._started = True
        hook = getattr(self.agent, "set_action_validator", None)
        if callable(hook):                    # the learner may drop candidates the environment reports as invalid
            try:
                hook(self.env.validate_action)
            except Exception as e:  # noqa: BLE001 — optional; the chosen action is validated here regardless
                self._note_error("agent", "set_action_validator", e)
        if resume:
            self._restore(st, inner)
        else:
            self._fresh()

    def _inner(self, st: Dict[str, Any]) -> Dict[str, Any]:
        """The exact resume state: the sealed part of state.json (or the plain fields of a version-1 file)."""
        if st.get("sealed"):
            try:
                return self.box.open(st["sealed"])
            except SealError as e:
                raise RuntimeError(f"cannot resume {self.run_dir}: {e}; the runtime does not continue on a state it "
                                   "cannot read exactly") from e
        raw = {aid: {"decision": a.get("decision"), "auth": a.get("auth"), "preds": []}
               for aid, a in (st.get("approvals") or {}).items() if a.get("decision")}
        return {"task": st["task"], "committed": st.get("committed", {}), "approvals_raw": raw,
                "next_tasks": st.get("next_tasks", []), "obstacles_open": st.get("obstacles_open"), "seen_cmds": []}

    def _guard_live(self, task: TaskSpec) -> None:
        problems = check_runnable(task)
        if task.mode != "live":
            # outside live mode an invalid limit does not prevent the start: the run ends at once instead (fail
            # closed, reported in run_started) — it is never treated as unlimited
            bad = limit_problems(task)
            problems = [p for p in problems if not any(p.startswith(f"limits.{f} is invalid") for f in bad)]
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
        self._use_task(self.task)
        self._agent("set_objective", learner_view(self.task))
        self._write_task_file()
        invalid = limit_problems(self.task)
        unset = [f for f in LIMIT_FIELDS if f not in invalid and getattr(self.task.limits, f) <= 0]
        self.log.event("run_started", {"task": self._view(self.task), "task_sha256": self.task.content_hash(),
                                       "withheld_from_learner": list(WITHHELD_FROM_LEARNER),
                                       "limits_not_enforced": unset + sorted(invalid), "limits_invalid": invalid,
                                       "invalid_limit_policy": "an invalid limit is never treated as unlimited: the "
                                                               "run ends at once with status exhausted",
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

    def _restore(self, st: Dict[str, Any], inner: Dict[str, Any]) -> None:
        self.status, self.status_reason = st["status"], st.get("status_reason", "")
        self._use_task(TaskSpec.from_dict(inner["task"]))
        self.started_at, self.paused_at = float(st.get("started_at", 0.0)), float(st.get("paused_at", 0.0))
        self.paused_total_s = float(st.get("paused_total_s", 0.0))
        self.iterations, self.cycles = int(st.get("iterations", 0)), int(st.get("cycles", 0))
        self.lifetime = dict(st.get("lifetime", self.lifetime))
        self.approvals = {aid: {k: v for k, v in a.items() if k not in ("decision", "auth")}
                          for aid, a in (st.get("approvals") or {}).items()}
        self._approval_raw = dict(inner.get("approvals_raw") or {})
        self._ckpts, self._exports = list(st.get("checkpoints", [])), list(st.get("exports", []))
        self._next_tasks = list(inner.get("next_tasks", []))
        self._control_applied = int(st.get("control_applied", 0))
        self._control_results = dict(st.get("control_results", {}))
        self._seen_cmds = list(inner.get("seen_cmds") or [])
        self._last_verified = dict(st.get("last_verified", {}))
        self._refused_verified = int(st.get("refused_verified", 0))
        self._settle = dict(inner["settle"]) if isinstance(inner.get("settle"), dict) else None
        self.obstacles.load_state(inner.get("obstacles_open"))
        c = self._committed = dict(inner.get("committed", {}))
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
            self._agent("set_objective", learner_view(self.task))
        replay = self._replay_wal()
        self.log.event("run_resumed", {"status": self.status, "checkpoint": (ckpt or {}).get("id", ""),
                                       "generation": self.generation, "log_repairs": self.log.repaired, **replay},
                       consequential=True)
        self._triggers.append("resume")
        self._save_state()

    # ==========================================================================================================
    # write-ahead journal (sealed lines: {"g", "t", "kind", "action_id", "x": <sealed payload>})
    # ==========================================================================================================
    def _wal_line(self, entry: Dict[str, Any]) -> str:
        entry["g"] = self.generation
        head = {k: entry[k] for k in ("g", "t", "kind", "action_id") if k in entry}
        head["x"] = self.box.seal({k: v for k, v in entry.items() if k not in head})
        return json.dumps(head, sort_keys=True) + "\n"

    def _wal_write(self, entry: Dict[str, Any]) -> None:
        if self._wal_fh is None:
            _repair_torn_tail(self._wal_path)            # a torn last line would swallow the next entry
            fd = os.open(self._wal_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            self._wal_fh = os.fdopen(fd, "a")
        line = self._wal_line(entry)
        self._wal_fh.write(line)
        self._wal_fh.flush()
        if self._durable:
            os.fsync(self._wal_fh.fileno())
        if entry.get("t") == "begin":
            self._wal_known.add(entry.get("action_id", ""))

    def _wal_carry(self) -> List[Dict[str, Any]]:
        """Entries that outlive a commit: a wait in progress (its decision was counted in the committed counters)."""
        w = self._waiting
        if w is None:
            return []
        return [{"t": "begin", "kind": "wait", "action_id": w["decision"].chosen.action_id,
                 "decision": w["decision"].to_dict(), "preds": w["preds"], "approval_id": "", "started": w["started"],
                 "until": w["until"], "delta": {}, "carried": True}]

    def _wal_reset(self) -> None:
        """Empty the journal after a commit, atomically keeping the carried entries (same generation)."""
        if self._wal_fh is not None:
            self._wal_fh.close()
            self._wal_fh = None
        carry = self._wal_carry()
        atomic_write_text(self._wal_path, "".join(self._wal_line(e) for e in carry), 0o600, self._durable)
        self._wal_known = {e["action_id"] for e in carry}

    def _wal_entries(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for e in iter_jsonl(self._wal_path):
            if "x" in e:
                try:
                    body = self.box.open(e["x"])
                except SealError as ex:
                    self.log.event("error", {"who": "runtime", "what": "journal",
                                             "error": f"a journal line does not decrypt and is ignored: {ex}"},
                                   consequential=True)
                    continue
                out.append({**body, **{k: v for k, v in e.items() if k != "x"}})
            else:
                out.append(e)                            # version-1 journal line
        return out

    def _logged_since_commit(self) -> Dict[str, set]:
        """What the logs already hold since the committed position (read backwards from the end of each file)."""
        seq = int(self._committed.get("seq", 0) or 0)
        got: Dict[str, Any] = {k: set() for k in ("results", "observations", "outcomes", "decisions", "resolutions",
                                                  "exp_outcomes")}
        got["outcome_rids"] = {}
        d = self.log.log_dir
        for r in iter_since(os.path.join(d, RunFiles.EVENTS), seq):
            data = r.get("data") or {}
            if r.get("type") == "action_result":
                got["results"].add(data.get("action_id", ""))
            elif r.get("type") == "observation":
                got["observations"].add((data.get("observation") or {}).get("obs_id", ""))
            elif r.get("type") == "outcome":
                qid = (data.get("query") or {}).get("query_id", "")
                got["outcomes"].add(qid)
                got["outcome_rids"][qid] = r.get("rid", "")
        for r in iter_since(os.path.join(d, RunFiles.DECISIONS), seq):
            got["decisions"].add(((r.get("data") or {}).get("decision") or {}).get("decision_id", ""))
        for r in iter_since(os.path.join(d, RunFiles.PREDICTIONS), seq):
            if r.get("type") == "resolution":
                got["resolutions"].add((r.get("data") or {}).get("prediction_id", ""))
        for r in iter_since(os.path.join(d, RunFiles.EXPERIMENTS), seq):
            if r.get("type") == "experiment_outcome":
                data = r.get("data") or {}
                got["exp_outcomes"].add((data.get("experiment_id", ""), (data.get("outcome") or {}).get("query_id", "")))
        return got

    def _replay_wal(self) -> Dict[str, Any]:
        logged = self._logged_since_commit()
        open_: Dict[str, Dict[str, Any]] = {}
        done: List[str] = []
        outcomes = 0
        self._wal_known = set()
        for e in self._wal_entries():
            if e.get("g") != self.generation:
                continue
            t, aid = e.get("t"), e.get("action_id", "")
            if t == "begin":
                self._apply_delta(e.get("delta", {}))
                self._wal_known.add(aid)
                slot = open_.setdefault(aid, {})
                slot["decided" if e.get("kind") == "decided" else "begin"] = e
            elif t == "end":
                slot = open_.pop(aid, {})
                b = slot.get("begin") or slot.get("decided") or e
                if not b.get("decision"):
                    continue                             # an end of an earlier generation's decision without its copy
                decision = Decision.from_dict(b["decision"])
                result = ActionResult.from_dict(e["result"])
                self._ensure_decision_record(decision, slot.get("decided"), logged)
                self._ensure_result_records(decision, result, e, logged)
                self._absorb(decision, result, e.get("delta", {}), (slot.get("decided") or b).get("preds", []),
                             e.get("approval_id", ""), e.get("kind", ""))
                done.append(aid)
            elif t == "outcome":
                self._replay_outcome(e, logged)
                outcomes += 1
        inflight, interrupted = [], []
        for aid, slot in open_.items():
            if slot.get("begin") is not None:
                self._ensure_decision_record(Decision.from_dict(slot["begin"]["decision"]), slot.get("decided"), logged)
                self._recover_inflight(slot["begin"], (slot.get("decided") or slot["begin"]).get("preds", []))
                inflight.append(aid)
            elif slot.get("decided") is not None:
                d = slot["decided"]
                decision = Decision.from_dict(d["decision"])
                self._ensure_decision_record(decision, d, logged)
                self._complete(decision, self._synthetic(decision.chosen, ResultStatus.CANCELLED, FailureClass.UNKNOWN,
                                                         "interrupted by a process restart before the decision was "
                                                         "handled; nothing was executed"),
                               "interrupted", d.get("preds", []), d.get("approval_id", ""))
                interrupted.append(aid)
        return {"replayed_results": done, "inflight_recovered": inflight, "decisions_closed_unhandled": interrupted,
                "replayed_outcomes": outcomes, "redelivered_outcomes": len(self._redeliver)}

    def _ensure_decision_record(self, decision: Decision, decided: Optional[Dict[str, Any]],
                                logged: Dict[str, set]) -> None:
        """The decision record of a decision journalled in this generation, if the crash came before it."""
        if decided is None or decision.decision_id in logged["decisions"]:
            return
        self.log.decision(self._clean_decision(decision), objective=self.task.objective,
                          situation={**(decided.get("situation") or {}), "written_after_restart": True})
        logged["decisions"].add(decision.decision_id)

    def _ensure_result_records(self, decision: Decision, result: ActionResult, end: Dict[str, Any],
                               logged: Dict[str, set]) -> None:
        """The action_result and observation records of a journalled result, if the crash came before them."""
        if result.action_id not in logged["results"]:
            self._log_result(decision, result, end.get("kind", ""), end.get("approval_id", ""),
                             end.get("normalised") or [], replayed=True)
            logged["results"].add(result.action_id)
        else:
            for o in result.observations:
                if o.obs_id not in logged["observations"]:
                    self.log.observation(o, "action")

    def _replay_outcome(self, e: Dict[str, Any], logged: Dict[str, set]) -> None:
        o = OutcomeObservation.from_dict(e["observation"])
        item = PendingItem.from_dict(e["item"])
        self.pending.resolve(o)
        resolved = set(e.get("preds") or [])
        matched = [p for p in self._open_preds if p["prediction_id"] in resolved]
        self._open_preds = [p for p in self._open_preds if p["prediction_id"] not in resolved]
        if o.query_id not in logged["outcomes"]:
            self._log_outcome(item, o, matched, replayed=True)
        else:
            rid = logged["outcome_rids"].get(o.query_id, "")
            for p in matched:
                if p["prediction_id"] not in logged["resolutions"]:
                    self._log_resolution(p, item, o, rid)
            if item.experiment_id and (item.experiment_id, o.query_id) not in logged["exp_outcomes"]:
                self.log.experiment_outcome(item.experiment_id, item.decision_id, {**o.to_dict(), "outcome_rid": rid})
        self._redeliver.append(o)

    def _recover_inflight(self, b: Dict[str, Any], preds: List[Dict[str, Any]]) -> None:
        """An action (or a wait) was started and the process died before its result was journalled."""
        decision = Decision.from_dict(b["decision"])
        action, now = decision.chosen, self.clock.now()
        kind, approval_id = b.get("kind", "execute"), b.get("approval_id", "")
        if kind == "wait":
            self._waiting = {"decision": decision, "started": float(b.get("started", now)),
                             "until": float(b.get("until", now)), "preds": preds}
            self._finish_wait()
            return
        caps = self._caps()
        how, est = "", self._estimate(action)
        unknown = (f"whether it took effect is unknown; its estimated cost {est:g} is booked against the limits")
        if (caps.get("idempotent_actions") or caps.get("dedup_by_action_id")
                or (caps.get("features") or {}).get("at_most_once_by_action_id")):
            try:
                auth = self.policy.authorize(action, self.task, self._auth_context(decision))
                if auth.verdict == AuthVerdict.ALLOW and look_grant(self.policy, action, auth):
                    result = self.env.execute(action, auth)     # the environment returns the stored result, or runs it once
                    how = "re-issued under the same action id (environment de-duplicates)"
                else:
                    result = self._synthetic(action, ResultStatus.CANCELLED, FailureClass.UNKNOWN,
                                             "interrupted by a process restart; not re-authorized afterwards "
                                             f"({'; '.join(auth.reasons)}); {unknown}", cost=Cost(money=est))
            except Exception as e:  # noqa: BLE001
                result = self._synthetic(action, ResultStatus.CANCELLED, FailureClass.UNKNOWN,
                                         f"interrupted by a process restart; recovery failed: {type(e).__name__}: {e}; "
                                         f"{unknown}", cost=Cost(money=est))
        else:
            result = self._synthetic(action, ResultStatus.CANCELLED, FailureClass.UNKNOWN,
                                     "interrupted by a process restart; the environment does not de-duplicate by "
                                     f"action id, so it was not issued again; {unknown}", cost=Cost(money=est))
        status = result.status if isinstance(result, ActionResult) else "malformed"
        self.log.event("inflight_recovered", {"action_id": action.action_id, "status": status, "how": how},
                       consequential=True)
        self._complete(decision, result, kind, preds, approval_id)

    # ==========================================================================================================
    # state
    # ==========================================================================================================
    def _snapshot(self, ckpt: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {"generation": self.generation, "cursor": self.cursor, "pending": self.pending.to_state(),
                "counters": dict(self.counters), "open_predictions": list(self._open_preds), "ckpt": ckpt,
                "agent_wake_at": self._agent_wake_at, "ts": self.clock.now(), "seq": self.log.next_seq}

    def _limit_state(self) -> Dict[str, Dict[str, Any]]:
        lim, out = self.task.limits, {}
        bad = limit_problems(self.task)
        used = {"duration_s": self._elapsed(), **{f: float(self.counters[c]) for f, c in _LIMIT_COUNTER.items()}}
        used["compute_s"] += self._compute_unbooked
        for f in ("duration_s",) + tuple(_LIMIT_COUNTER):
            row: Dict[str, Any] = {"used": used[f], "limit": 0.0 if f in bad else float(getattr(lim, f))}
            if f in bad:
                row["invalid"] = bad[f]
            out[f] = row
        return out

    def _red(self, text: Any) -> str:
        """Redacted text for state.json (cached: the same metric names recur in every save)."""
        t = str(text)
        r = self._red_cache.get(t)
        if r is None:
            r = self.log.redact.text(t)
            if len(self._red_cache) > 20000:
                self._red_cache.clear()
            self._red_cache[t] = r
        return r

    def _pending_rows(self) -> List[Dict[str, Any]]:
        return [{"query_id": i.query.query_id, "metric": self._red(i.query.metric), "due_at": i.query.due_at,
                 "ready_at": i.ready_at, "action_id": i.action_id, "decision_id": i.decision_id, "origin": i.origin,
                 "attempts": i.attempts} for i in self.pending.items()]

    def _state(self) -> Dict[str, Any]:
        """state.json: redacted, human-readable fields plus the sealed exact state used to resume."""
        w = self._waiting
        c = self._committed or {}
        red = self.log.redact
        inner = {"task": self.task.to_dict(), "committed": c, "approvals_raw": self._approval_raw,
                 "next_tasks": self._next_tasks, "obstacles_open": self.obstacles.to_state(),
                 "seen_cmds": self._seen_cmds[-100000:], "settle": self._settle}
        return {
            "version": STATE_VERSION, "run_id": self.log.run_id, "actor_id": self.actor_id, "pid": os.getpid(),
            "status": self.status, "status_reason": self._red(self.status_reason),
            "task": self._view(self.task), "task_sha256": self.task.content_hash(),
            "awaiting_operator": self.status in (PAUSED,) + TERMINAL,
            "next_tasks": [{"task": {"objective_id": (t.get("task") or {}).get("objective_id", ""),
                                     "mode": (t.get("task") or {}).get("mode", "")},
                            "path": t.get("path", ""), "by": t.get("by", ""), "assigned_at": t.get("assigned_at")}
                           for t in self._next_tasks],
            "started_at": self.started_at, "paused_at": self.paused_at,
            "paused_total_s": self.paused_total_s, "updated_at": self.clock.now(), "iterations": self.iterations,
            "cycles": self.cycles, "counters": dict(self.counters), "lifetime": self.lifetime,
            "limits": self._limit_state(), "approvals": red(self.approvals),
            "committed": {k: c.get(k) for k in ("generation", "ckpt", "counters", "ts", "seq")},
            "live": {"pending": {"items": self._pending_rows()}, "generation": self.generation},
            "checkpoints": self._ckpts, "exports": self._exports, "control_applied": self._control_applied,
            "control_results": red(self._control_results), "activity": self._activity, "wake": self._wake_info,
            "last_verified": red(self._last_verified), "refused_verified": self._refused_verified,
            "obstacles_open": self.obstacles.open_count(), "errors": red(self._errors[-20:]),
            "settling": None if self._settle is None else {**self._settle, "reason": self._red(self._settle["reason"])},
            "gpu": self.gpu,
            "waiting": None if w is None else {"until": w["until"], "started": w["started"],
                                                "action_id": w["decision"].chosen.action_id},
            "sealed": self.box.seal(inner),
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
        d = _finite(self.task.limits.duration_s)
        return None if d is None or d <= 0 else self.started_at + self.paused_total_s + d

    def _exhausted(self) -> str:
        """Name of the first exhausted limit, or ''. A limit of 0 is "not set": allowed only outside live mode,
        and reported at run start rather than silently treated as unlimited. A limit that is not a finite
        non-negative number counts as exhausted (fail closed)."""
        bad = limit_problems(self.task)
        if bad:
            return sorted(bad)[0]
        for name, s in self._limit_state().items():
            if s["limit"] > 0 and s["used"] >= s["limit"] - 1e-9:
                return name
        return ""

    def _may_drain(self, exhausted: str) -> bool:
        """S-1: with ``drain_after_exhaustion``, a run whose *decision* budget is used up while its objective is
        pending keeps observing — measuring due outcomes, verifying, learning — but decides nothing more, until the
        verifier's verdict is terminal or the deadline passes (then it finishes as exhausted)."""
        if not (exhausted == "max_decisions" and self.cfg.drain_after_exhaustion and self.task.objective.strip()
                and self.task.success):
            return False
        d = self._deadline()
        return d is not None and self.clock.now() < d

    def _exhaust_text(self, name: str) -> str:
        why = limit_problems(self.task).get(name)
        return (f"limit {name} is invalid ({why}): the run does not proceed with a limit that cannot be enforced"
                if why else f"limit {name} exhausted")

    def status_report(self) -> Dict[str, Any]:
        st = self._state()
        st.pop("sealed", None)
        try:
            st["agent"] = self.agent.status()
        except Exception as e:  # noqa: BLE001
            st["agent"] = {"error": self._red(f"{type(e).__name__}: {e}")}
        return st

    # ==========================================================================================================
    # learner calls (timed for compute accounting)
    # ==========================================================================================================
    def _agent(self, name: str, *args: Any) -> Any:
        guard = self.learner_guard
        if guard is not None and name not in _NO_GPU_CALLS:
            guard(name)                 # G2R-9: e.g. runtime.gpu_cycle: the shared GPU lock is held for every
                                        # learner call; waiting for it is not charged to the compute limit
        t0 = time.perf_counter()
        try:
            return getattr(self.agent, name)(*args)
        finally:
            dt = time.perf_counter() - t0
            self._compute_unbooked += dt
            if self.learner_timing is not None:
                self.learner_timing(name, dt)

    def _agent_try(self, name: str, *args: Any, default: Any = None) -> Any:
        try:
            out = self._agent(name, *args)
            return default if out is None else out
        except DiskSpaceLow as e:                         # the learner refused to write: pause, do not retry
            self._storage_low(e, f"learner.{name}")
            return default
        except Exception as e:  # noqa: BLE001 — recorded; repeated errors pause the run (see _note_error)
            self._note_error("learner", name, e)
            return default

    def _note_error(self, who: str, what: str, e: BaseException) -> None:
        self._consecutive_errors += 1
        self._error_total += 1
        rec = {"ts": self.clock.now(), "who": who, "what": what, "error": f"{type(e).__name__}: {e}"[:500]}
        self._errors.append({**rec, "error": self.log.redact.text(rec["error"])})
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
            cmd, cid = c.get("cmd", ""), str(c.get("id", ""))
            signed, why = self.control.authenticate(c)
            by = str(c.get("by", "operator")) if signed else "unauthenticated"
            if signed and cid in self._seen_cmds:
                ok, detail = False, "refused: a signed command with this id was already applied (replayed command)"
            elif cmd in PRIVILEGED and not signed:
                ok, detail = False, (f"refused: {cmd} needs a command signed with the operator's key ({why}); "
                                     "use the cam-live CLI or the dashboard as the operator")
            else:
                self._cmd_in_progress = c if signed else None       # the signed command, for the verifier (G2R-4)
                try:
                    ok, detail = self._apply_command(cmd, c.get("args", {}) or {}, by)
                except Exception as e:  # noqa: BLE001
                    ok, detail = False, f"{type(e).__name__}: {e}"
                finally:
                    self._cmd_in_progress = None
            if signed:
                self._seen_cmds.append(cid)
            self._control_applied += 1
            self._control_results[cid] = {"ok": ok, "detail": detail, "ts": self.clock.now(), "cmd": cmd,
                                          "status": self.status, "authenticated": signed}
            for k in list(self._control_results)[:-200]:
                del self._control_results[k]
            self.log.event("operator_command", {"id": cid, "cmd": cmd, "by": by, "declared_by": c.get("by"),
                                                "authenticated": signed, "args": _logged_args(c.get("args")), "ok": ok,
                                                "detail": detail},
                           consequential=True, trust=Trust.OPERATOR if signed else Trust.UNTRUSTED)
        if new:
            self._save_state()
        return len(new)

    def _apply_command(self, cmd: str, args: Dict[str, Any], by: str) -> tuple:
        if cmd == "pause":
            if self.status != RUNNING:
                return False, f"not running (status {self.status})"
            self._pause(str(args.get("reason") or "")[:200] or f"paused by {by}")
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
            self._halt(STOPPED, str(args.get("reason") or "")[:200] or f"stopped by {by}")
            return True, "stopped; learner state checkpointed"
        if cmd in ("approve", "reject"):
            return self._relay_approval(str(args.get("approval_id", "")), cmd == "approve", by,
                                        str(args.get("note", ""))[:200])
        if cmd == "approval_notice":
            self._triggers.append("approval_notice")
            aid, grant = str(args.get("approval_id", "")), args.get("grant")
            a = self.approvals.get(aid)
            if (isinstance(grant, dict) and self._cmd_in_progress is not None and a is not None
                    and a.get("status") == "pending"):
                # G2I-15: the grant issued when the operator approved (signed notice only); checked at refresh
                self._approval_raw.setdefault(aid, {})["notice_auth"] = dict(grant)
            return True, "approvals will be re-read from the authorization service"
        if cmd == "wake":
            self._triggers.append("operator_wake")
            return True, "woken"
        if cmd == "export_checkpoint":
            return self._export(str(args.get("dest", "")))
        if cmd == "assign_objective":
            return self._assign(str(args.get("path", "")), str(args.get("sha256", "")), by)
        return False, f"unknown command {cmd!r}"

    def _pause(self, reason: str, commit: bool = True) -> None:
        self._finish_wait()
        if commit:
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
        settle, self._settle = self._settle, None
        self.log.event("halted", {"status": status, "reason": reason,
                                  "checkpoint": (self._committed.get("ckpt") or {}).get("id", ""),
                                  "pending_outcomes_carried": len(self.pending),
                                  "interrupted_settle": settle}, consequential=True)
        self._journal()
        self._save_state()

    def _export(self, dest: str) -> tuple:
        dest = dest or os.path.join(self.run_dir, "exports", f"export_{len(self._exports) + 1:04d}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # storage rule: an export must leave the configured minimum free (about the size of the last checkpoint)
        last = (self._committed.get("ckpt") or {}).get("path", "")
        try:
            DiskGuard(os.path.dirname(dest), float(self.cfg.min_free_gb), float(self.cfg.warn_free_gb)).check(
                dir_bytes(last) if last and os.path.isdir(last) else 0, "export")
        except DiskSpaceLow as e:
            return False, f"not exported: {e}"
        if callable(getattr(self.agent, "export_weights", None)):
            # weights-only transfer artefact (spec 15.3): weights, adapters and the configuration that rebuilds the
            # model — no experience, optimizer or task state (credit.checkpoint.load_exported_model reads it)
            info = self._agent("export_weights", dest) or {}
            cid = str(info.get("model_version", "")) or "weights"
            how = "weights-only export (agent.export_weights)"
        else:
            cid = self._agent("save", dest)
            how = "complete learner checkpoint (agent.save)"
        meta = {"checkpoint_id": cid, "path": dest, "ts": self.clock.now(), "objective_id": self.task.objective_id,
                "provenance": self.task.mode, "counters": dict(self.counters), "method": how}
        atomic_write_json(dest.rstrip("/") + ".export.json", {**meta, "task": self.task.to_dict()})
        self._exports.append(meta)
        return True, f"exported {cid} to {dest} ({how})"

    def _assign(self, path: str, sha256: str, by: str) -> tuple:
        # G2R-18: the file is read once; the bytes whose hash the command names are the bytes parsed
        try:
            task = load_task_bytes(read_task_file(path), sha256=sha256, for_run=True, path=path)
        except TaskFileChanged:
            return False, "task file changed after the command was issued"
        self._guard_live(task)
        # G2R-4: the run's verifier must be able to judge the new objective: it accepts it only with the operator's
        # signed command (it checks the signature and the task file itself); if it refuses, so does the runtime.
        # G2R-15: also for the objective in force (new deadline, target, ...): the verifier then judges the changed
        # objective, not the one it was pinned to
        accept = getattr(self.verifier, "assign_objective", None)
        if callable(accept):
            try:
                accept(dict(self._cmd_in_progress or {}))
            except Exception as e:  # noqa: BLE001
                return False, (f"refused: the run's verifier cannot judge objective {task.objective_id} "
                               f"({type(e).__name__}: {str(e)[:200]})")
        entry = {"task": task.to_dict(), "path": path, "by": by, "assigned_at": self.clock.now()}
        if task.objective_id == self.task.objective_id:
            self._use_task(task)                       # updated boundaries for the same objective
            self._write_task_file()
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
        self._settle = None
        self._use_task(TaskSpec.from_dict(entry["task"]))
        self.counters = dict(_COUNTERS)
        self.started_at, self.paused_total_s, self.paused_at = self.clock.now(), 0.0, 0.0
        self.status, self.status_reason = RUNNING, ""
        self._last_verified, self._last_verify_at, self._agent_wake_at = {}, None, None
        self._agent("set_objective", learner_view(self.task))   # the learner's weights and lifetime context carry over
        self._write_task_file()
        self.log.event("objective_assigned", {"task": self._view(self.task),
                                              "task_sha256": self.task.content_hash(), "by": entry.get("by", ""),
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
            a.update(status="approved", resolved_at=self.clock.now(), by=by, note=note)
            self._approval_raw.setdefault(approval_id, {})["auth"] = auth.to_dict()
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
            raw = self._approval_raw.get(aid) or {}
            if not raw.get("decision"):
                a.update(status="rejected", resolved_at=self.clock.now(),
                         reasons=["the exact decision of this approval could not be restored; not executed"])
                changed = True
                continue
            decision = Decision.from_dict(raw["decision"])
            if state in ("rejected", "denied", "deny"):
                a.update(status="rejected", resolved_at=self.clock.now(), note=str((e or {}).get("note", ""))[:200])
                changed = True
                continue
            # approved there, or no longer listed. G2I-15: the grant the service issued at the approval (relayed by
            # the operator's signed notice) is used when the service confirms it for this exact action — asking
            # again would issue a second grant and reserve the action's budget a second time; otherwise the
            # authorization service is asked again and is the only judge
            notice = raw.pop("notice_auth", None)
            auth = None
            if isinstance(notice, dict) and state in ("approved", "allow", "allowed", ""):
                try:
                    cand = AuthorizationDecision.from_dict(notice)
                    if (cand.verdict == AuthVerdict.ALLOW and cand.action_hash == decision.chosen.content_hash()
                            and look_grant(self.policy, decision.chosen, cand)):
                        auth = cand
                except Exception as ex:  # noqa: BLE001 — not usable: ask the service again
                    self._note_error("authorization", "check_grant", ex)
            try:
                auth = auth or self.policy.authorize(decision.chosen, self.task, self._auth_context(decision))
            except Exception as ex:  # noqa: BLE001
                self._note_error("authorization", "authorize", ex)
                continue
            if auth.verdict == AuthVerdict.ALLOW:
                a.update(status="approved", resolved_at=self.clock.now())
                raw["auth"] = auth.to_dict()
                changed = True
            elif auth.verdict == AuthVerdict.DENY:
                # approved by the operator (no longer pending there), then refused by a rule of the service
                a.update(status="refused_after_approval", resolved_at=self.clock.now(), reasons=list(auth.reasons),
                         policy_ids=list(auth.policy_ids or []), failure_class=refusal_failure_class(auth))
                changed = True
            elif auth.approval_id and auth.approval_id != aid:
                a["approval_id"] = auth.approval_id
                self.approvals[auth.approval_id] = self.approvals.pop(aid)
                self._approval_raw[auth.approval_id] = self._approval_raw.pop(aid, raw)
        return changed

    def _process_approvals(self) -> bool:
        """Execute each approved action exactly once — if the limits still allow it at that moment — and report
        each rejected one exactly once."""
        acted = False
        for aid in list(self.approvals):
            a = self.approvals.get(aid)
            if a is None or a["status"] not in ("approved", "rejected", "refused_after_approval"):
                continue
            raw = self._approval_raw.get(aid) or {}
            if not raw.get("decision"):
                a.update(status="reported", result_status="not_restorable")
                self.log.event("approval_not_restorable", {"approval_id": aid}, consequential=True)
                continue
            decision = Decision.from_dict(raw["decision"])
            action, preds = decision.chosen, list(raw.get("preds") or [])
            if a["status"] == "approved":
                over = self._exhausted()
                why = self._exhaust_text(over) if over else self._would_exceed(action)
                if not why and self._policy_stopped():
                    why = "the emergency stop is engaged at the authorization service"
                auth = AuthorizationDecision.from_dict(raw.get("auth") or {"verdict": AuthVerdict.DENY})
                if why:
                    self._complete(decision, self._synthetic(action, ResultStatus.FAILED, FailureClass.RESOURCE_LIMIT,
                                                             f"approved by the operator, but {why}; not executed"),
                                   "over_limit", preds, aid, new_decision=False)
                elif auth.verdict != AuthVerdict.ALLOW or not look_grant(self.policy, action, auth):
                    self._complete(decision, self._synthetic(action, ResultStatus.DENIED, FailureClass.MISSING_PERMISSION,
                                                             "the approval did not yield a grant valid for this action"),
                                   "denied", preds, aid, new_decision=False)
                else:
                    self._execute(decision, auth, preds, approval_id=aid, new_decision=False)
            elif a["status"] == "refused_after_approval":
                why = "; ".join(str(r) for r in a.get("reasons") or []) or "refused"
                rules = ", ".join(a.get("policy_ids") or []) or "a rule of the authorization service"
                self._complete(decision, self._synthetic(action, ResultStatus.DENIED,
                                                         a.get("failure_class") or FailureClass.MISSING_PERMISSION,
                                                         f"approved by the operator ({aid}), then refused by the "
                                                         f"authorization service ({rules}): {why}; not executed"),
                               "denied", preds, aid, new_decision=False)
            else:
                # the operator's note stays in the audit trail; no free text from the operator reaches the learner
                self._complete(decision, self._synthetic(action, ResultStatus.DENIED, FailureClass.MISSING_PERMISSION,
                                                         f"the operator rejected approval {aid}"),
                               "rejected", preds, aid, new_decision=False)
            acted = True
        return acted

    # ==========================================================================================================
    # evidence: steps 2-4
    # ==========================================================================================================
    def _ingest(self, obs: Sequence[Observation], via: str) -> None:
        obs = [o for o in obs if isinstance(o, Observation)]
        if not obs:
            return
        for o in obs:
            self.log.observation(o, via)
            for q in o.quantities:
                v = _finite(getattr(q, "value", None))
                if v is not None:
                    self._metrics[str(q.name)] = v
        self._cycle_obs.extend(obs)
        self._agent_try("ingest", list(obs))
        self._dirty = True

    def _verified_measurement(self, o: Any) -> bool:
        """True only for an observation the verifier signed and confirms (``check_measurement``)."""
        fn = getattr(self.verifier, "check_measurement", None)
        if not callable(fn):
            return False
        try:
            return bool(fn(o))
        except Exception as e:  # noqa: BLE001
            self._note_error("verifier", "check_measurement", e)
            return False

    def _measure(self, due: List[PendingItem]) -> List[OutcomeObservation]:
        """Step 4 for matured queries. Goal-relevant queries are asked of the independent verifier first; an
        observation counts as verified only when the verifier's signature on it checks. Anything the environment
        measures is unverified whatever it claims."""
        if not due:
            return []
        now = self.clock.now()
        got: Dict[str, OutcomeObservation] = {}
        goal = [i.query for i in due if i.query.goal_relevant]
        if goal:
            try:
                for o in self.verifier.measure(goal) or []:
                    if isinstance(o, OutcomeObservation) and o.status in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED):
                        o.verified = self._verified_measurement(o)
                        if not o.verified:
                            o.source = f"unsigned:{o.source}"
                        got[o.query_id] = o
            except Exception as e:  # noqa: BLE001
                self._note_error("verifier", "measure", e)
        rest = [i.query for i in due if i.query.query_id not in got]
        if rest:
            try:
                for o in self.env.measure(rest) or []:
                    if isinstance(o, OutcomeObservation):
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
            if o.value is not None and _finite(o.value) is None:
                o.value, o.status = None, OutcomeStatus.UNOBSERVABLE
            if self.pending.resolve(o) is None:
                continue                                       # duplicate of an event already reconciled
            matched = self._matching_preds(item, o)
            # journalled before it is logged: a restart neither measures it again nor loses it
            self._wal_write({"t": "outcome", "kind": item.origin, "action_id": item.action_id,
                             "query_id": o.query_id, "item": item.to_dict(), "observation": o.to_dict(),
                             "preds": [p["prediction_id"] for p in matched]})
            ids = {p["prediction_id"] for p in matched}
            self._open_preds = [p for p in self._open_preds if p["prediction_id"] not in ids]
            self._log_outcome(item, o, matched)
            out.append(o)
        return out

    def _matching_preds(self, item: PendingItem, o: OutcomeObservation) -> List[Dict[str, Any]]:
        return [p for p in self._open_preds
                if p["query_id"] == o.query_id or (p["action_id"] and p["action_id"] == item.action_id
                                                    and p["metric"] == item.query.metric)]

    def _log_outcome(self, item: PendingItem, o: OutcomeObservation, matched: List[Dict[str, Any]],
                     replayed: bool = False) -> str:
        data = {"query": item.query.to_dict(), "observation": o.to_dict(), "action_id": item.action_id,
                "decision_id": item.decision_id, "experiment_id": item.experiment_id, "query_origin": item.origin}
        if replayed:
            data["written_after_restart"] = True
        rid = self.log.event("outcome", data, consequential=True,
                             trust=Trust.VERIFIER if o.verified else Trust.UNTRUSTED)
        for p in matched:
            self._log_resolution(p, item, o, rid)
        if item.experiment_id:
            self.log.experiment_outcome(item.experiment_id, item.decision_id, {**o.to_dict(), "outcome_rid": rid})
        return rid

    def _log_resolution(self, p: Dict[str, Any], item: PendingItem, o: OutcomeObservation, outcome_rid: str) -> None:
        self.log.resolution(p["prediction_id"], o.query_id,
                            {"status": o.status, "value": o.value, "observed_at": o.observed_at,
                             "verified": o.verified, "outcome_rid": outcome_rid, "metric": item.query.metric},
                            registered_rid=p.get("rid", ""), predicted_mean=p.get("mean"), predicted_std=p.get("std"))

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

    @staticmethod
    def _clean_prediction(p: PredictionRecord) -> Tuple[PredictionRecord, bool]:
        """A registered prediction carries no outcome: resolved_* fields filled in by the learner are cleared."""
        if (p.resolved_at is None and p.resolved_value is None and p.nll is None
                and p.resolved_status == OutcomeStatus.PENDING):
            return p, False
        c = copy.deepcopy(p)
        c.resolved_at, c.resolved_value, c.nll, c.resolved_status = None, None, None, OutcomeStatus.PENDING
        return c, True

    def _clean_decision(self, decision: Decision) -> Decision:
        if not any(self._clean_prediction(p)[1] for c in decision.candidates for p in c.predictions):
            return decision
        d = Decision.from_dict(decision.to_dict())
        for c in d.candidates:
            c.predictions = [self._clean_prediction(p)[0] for p in c.predictions]
        return d

    def _decide_and_act(self) -> bool:
        tools = list(self.env.discover_tools() or [])
        self._tools = {t.name: t for t in tools}
        self._tool_effects = {t.name: t.side_effects for t in tools}
        self._set_activity("deciding")
        cands = [c for c in (self._agent_try("propose", tools, self.cfg.candidates_k, default=[]) or [])
                 if isinstance(c, ActionSpec)]
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
        if not isinstance(decision, Decision) or not isinstance(decision.chosen, ActionSpec):
            return False
        action = decision.chosen
        # ---- registered before anything is executed (steps 7-8 -> disk); learner-filled outcomes are cleared ----
        all_preds, cleared = [], 0
        for e in evals:
            for p in getattr(e, "predictions", []) or []:
                if isinstance(p, PredictionRecord):
                    c, changed = self._clean_prediction(p)
                    all_preds.append(c)
                    cleared += changed
        rids = self.log.predictions(all_preds, decision_id=decision.decision_id, chosen_action_id=action.action_id)
        if cleared:
            self.log.event("prediction_fields_cleared",
                           {"n": cleared, "decision_id": decision.decision_id,
                            "reason": "a registered prediction carries no outcome; resolved_* fields supplied by the "
                                      "learner were removed before registration"})
        preds = [{"prediction_id": p.prediction_id, "query_id": p.query.query_id, "metric": p.query.metric,
                  "action_id": p.action_id, "decision_id": decision.decision_id, "mean": p.dist.mean,
                  "std": p.dist.std, "rid": rid, "query": p.query.to_dict()}
                 for p, rid in zip(all_preds, rids) if p.action_id == action.action_id]
        situation = self._situation()
        # the decision exists from this journal entry on: a crash after it closes the decision at restart
        self._begin(decision, "decided", preds, situation=situation)
        self.log.decision(self._clean_decision(decision), objective=self.task.objective, situation=situation)
        self.obstacles.on_decision(decision)
        # ---- step 10: structure, then authorization ----
        try:
            errs = list(self.env.validate_action(action) or [])
        except Exception as e:  # noqa: BLE001
            errs = [f"validation failed: {type(e).__name__}: {e}"]
        if errs:
            self._complete(decision, self._synthetic(action, ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS,
                                                     "; ".join(str(x) for x in errs)), "invalid", preds)
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
        if auth.verdict != AuthVerdict.ALLOW or not look_grant(self.policy, action, auth):
            detail = "; ".join(auth.reasons) or ("authorization refused" if auth.verdict != AuthVerdict.ALLOW
                                                 else "the grant does not match this action")
            self._complete(decision, self._synthetic(action, ResultStatus.DENIED, refusal_failure_class(auth),
                                                     detail), "denied", preds)
            return True
        self._execute(decision, auth, preds, new_decision=True)
        return True

    def _estimate(self, action: ActionSpec) -> float:
        """Money an action may spend: the larger of the learner's estimate and the tool's declared estimate."""
        vals = [_finite(getattr(getattr(action, "est_cost", None), "money", None))]
        tool = self._tools.get(action.operation)
        if tool is not None:
            vals.append(_finite(getattr(getattr(tool, "est_cost", None), "money", None)))
        return max([v for v in vals if v is not None and v > 0] or [0.0])

    def _would_exceed(self, action: ActionSpec) -> str:
        """Why executing ``action`` now would exceed a limit ('' = it would not). Checked before every execution,
        including actions the operator approved earlier."""
        lim = self.task.limits
        est = self._estimate(action)
        money = _finite(lim.money) or 0.0
        if money > 0 and self.counters["money"] >= money - 1e-9:
            return "the money limit is reached; not executed"
        if money > 0 and est > 0 and self.counters["money"] + est > money + 1e-9:
            return (f"estimated cost {est:g} exceeds the remaining budget "
                    f"{money - self.counters['money']:g}; not executed")
        ext = _finite(lim.max_external_actions) or 0.0
        if ext > 0 and self._external_steps(action) > 0 and self.counters["external_actions"] >= ext:
            return "the limit on external actions is reached; not executed"
        return ""

    def _is_external(self, action: ActionSpec) -> bool:
        if action.modality == Modality.INTERNAL:
            return False
        return self._tool_effects.get(action.operation, "external_write") not in ("none", "local")

    def _external_steps(self, action: ActionSpec) -> int:
        """G2R-5: externally effective steps of an action — a program's (composition's) steps counted one by one,
        recursively, whatever the program tool itself declares; a single action counts 1 when it is external."""
        steps = list(getattr(action, "steps", None) or [])
        if steps:
            return sum(self._external_steps(s) for s in steps)
        return int(self._is_external(action))

    def _external_steps_ran(self, action: ActionSpec, result: ActionResult) -> int:
        """External steps of an executed action that ran: from the executor's primitive record when it lists the
        steps' operations, else every external step (an action that ran is booked in full, like an in-flight one)."""
        steps = list(getattr(action, "steps", None) or [])
        if not steps:
            return int(self._is_external(action))
        ops = [o for o in (result.ops or []) if getattr(o, "op", "")]
        if ops:
            ran = [o for o in ops if getattr(o, "status", ResultStatus.OK) in _RAN]
            n = sum(1 for o in ran if self._tool_effects.get(o.op, "external_write") not in ("none", "local")
                    and getattr(o, "modality", "") != Modality.INTERNAL)
            return min(n, self._external_steps(action)) if n else 0
        return self._external_steps(action)

    def _execute(self, decision: Decision, auth: AuthorizationDecision, preds: List[Dict[str, Any]], *,
                 approval_id: str = "", new_decision: bool = True) -> None:
        action = decision.chosen
        # the journal entry is durable before the executor is called; the grant itself is never written to disk
        self._begin(decision, "execute", preds, approval_id, new_decision=False)
        self._set_activity("executing", action.render())
        self.log.event("action_start", {"decision_id": decision.decision_id, "action_id": action.action_id,
                                        "operation": action.operation, "modality": action.modality,
                                        "approval_id": approval_id}, consequential=True)
        try:
            result = self.env.execute(action, auth)
        except Exception as e:  # noqa: BLE001 — an executor error is an observation for the learner, not a crash
            self._note_error("environment", "execute", e)
            est = self._estimate(action)
            result = self._synthetic(action, ResultStatus.FAILED, FailureClass.UNKNOWN,
                                     f"executor error: {type(e).__name__}: {e}; whether the action took effect is "
                                     f"unknown, its estimated cost {est:g} is booked", cost=Cost(money=est))
        self._complete(decision, result, "execute", preds, approval_id)

    def _begin(self, decision: Decision, kind: str, preds: List[Dict[str, Any]], approval_id: str = "",
               new_decision: bool = True, **extra: Any) -> None:
        """Journal the start of a decision's handling. ``decided`` counts the decision; ``wait`` counts a wait."""
        delta = {"decisions": int(new_decision and kind == "decided"), "waits": int(kind == "wait")}
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
        a, now = decision.chosen.arguments or {}, self.clock.now()
        seconds = _finite(a.get("seconds", a.get("duration_s", 0.0))) if isinstance(a, dict) else None
        seconds = min(max(seconds or 0.0, 0.0), MAX_LEARNER_HORIZON_S)
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
        self._complete(w["decision"], result, "wait", w["preds"])

    def _normalise_result(self, action: ActionSpec, result: Any, kind: str) -> Tuple[ActionResult, List[str]]:
        """Whatever the executor returned becomes a well-formed ActionResult; nothing here can raise. A cost that is
        not a finite non-negative number is booked as the action's estimate (money) or 0."""
        notes: List[str] = []
        if not isinstance(result, ActionResult):
            est = self._estimate(action) if kind == "execute" else 0.0
            return self._synthetic(action, ResultStatus.FAILED, FailureClass.UNKNOWN,
                                   f"the executor returned {type(result).__name__}, not an ActionResult; whether the "
                                   f"action took effect is unknown", cost=Cost(money=est)), ["not an ActionResult"]
        r = copy.copy(result)
        if r.action_id != action.action_id:
            notes.append(f"result named action {str(r.action_id)[:80]!r}")
            r.action_id = action.action_id
        if not isinstance(r.status, str) or not r.status:
            notes.append(f"status {r.status!r} is not a status")
            r.status = ResultStatus.FAILED
        for f in ("failure_class", "failure_detail"):
            if not isinstance(getattr(r, f), str):
                setattr(r, f, str(getattr(r, f)))
        c = r.cost if isinstance(r.cost, Cost) else Cost()
        c = copy.copy(c)
        for f in ("seconds", "money", "compute_s"):
            v = _finite(getattr(c, f, 0.0))
            if v is None or v < 0:
                repl = self._estimate(action) if (f == "money" and kind == "execute") else 0.0
                notes.append(f"cost.{f} {getattr(c, f, None)!r} is not a finite non-negative number; {repl:g} booked")
                setattr(c, f, repl)
            else:
                setattr(c, f, v)
        for f in ("primitive_ops", "tokens"):
            v = _finite(getattr(c, f, 0))
            setattr(c, f, int(v) if v is not None and v >= 0 else 0)
        r.cost = c
        for f in ("started", "ended"):
            if _finite(getattr(r, f)) is None:
                setattr(r, f, self.clock.now())
        r.observations = [o for o in (r.observations or []) if isinstance(o, Observation)]
        r.ops = [o for o in (r.ops or []) if not isinstance(o, (str, bytes))]
        qs = []
        for q in r.expected_outcomes or []:
            if (isinstance(q, OutcomeQuery) and isinstance(q.query_id, str) and q.query_id
                    and _finite(q.horizon_s) is not None and q.horizon_s >= 0 and _finite(q.issued_at) is not None):
                qs.append(q)
            else:
                notes.append("a malformed expected outcome was dropped")
        r.expected_outcomes = qs
        if not isinstance(r.artifacts, dict):
            r.artifacts = {}
        return r, notes

    def _complete(self, decision: Decision, result: Any, kind: str, preds: List[Dict[str, Any]],
                  approval_id: str = "", *, new_decision: bool = True) -> None:
        """Step 12 for every decision, executed or not: journal, account, log, hand the result to the learner.
        The journal entry is written first, so a result that exists anywhere reaches the logs and the learner."""
        action = decision.chosen
        result, notes = self._normalise_result(action, result, kind)
        delta: Dict[str, Any] = {"compute_s": float(result.cost.compute_s) + self._compute_unbooked}
        self._compute_unbooked = 0.0
        if kind == "execute":
            ran = result.status not in (ResultStatus.DENIED, ResultStatus.NEEDS_APPROVAL)
            delta["actions_executed"] = int(ran)
            delta["external_actions"] = self._external_steps_ran(action, result) if ran else 0   # G2R-5
            delta["money"] = float(result.cost.money)
        if is_obstacle(result):
            delta["failures"] = 1
        end = {"t": "end", "kind": kind, "action_id": action.action_id, "result": result.to_dict(), "delta": delta,
               "approval_id": approval_id, "normalised": notes}
        if action.action_id not in self._wal_known:
            end["decision"], end["preds"] = decision.to_dict(), preds   # no begin of it in this generation
        self._wal_write(end)
        if kind == "execute" and self.cfg.record_at_policy:
            try:
                self.policy.record(action, result)
            except Exception as e:  # noqa: BLE001
                self._note_error("authorization", "record", e)
        self._log_result(decision, result, kind, approval_id, notes)
        self._absorb(decision, result, delta, preds, approval_id, kind)
        self._last_result_ok = result.status == ResultStatus.OK
        if is_obstacle(result):
            self.obstacles.open(action, result, self.agent, decision_id=decision.decision_id)
        self.obstacles.on_result(action, result)
        self._save_state()

    def _log_result(self, decision: Decision, result: ActionResult, kind: str, approval_id: str, notes: List[str],
                    replayed: bool = False) -> None:
        action = decision.chosen
        data = {"decision_id": decision.decision_id, "action_id": action.action_id, "kind": kind,
                "operation": action.operation, "status": result.status,
                "failure_class": result.failure_class, "failure_detail": result.failure_detail,
                "started": result.started, "ended": result.ended, "cost": result.cost.to_dict(),
                "verified": result.verified, "retries": result.retries, "artifacts": result.artifacts,
                "n_ops": len(result.ops), "observation_ids": [o.obs_id for o in result.observations],
                "expected_outcomes": [q.to_dict() for q in result.expected_outcomes],
                "approval_id": approval_id}
        if notes:
            data["normalised"] = notes
        if replayed:
            data["written_after_restart"] = True
        self.log.event("action_result", data, consequential=True)
        for o in result.observations:
            self.log.observation(o, "action")
            for q in o.quantities:
                v = _finite(getattr(q, "value", None))
                if v is not None:
                    self._metrics[str(q.name)] = v

    def _absorb(self, decision: Decision, result: ActionResult, delta: Dict[str, Any], preds: List[Dict[str, Any]],
                approval_id: str, kind: str) -> None:
        """The part of step 12 that is repeated when the journal is replayed after a restart."""
        self._apply_delta(delta)
        action = decision.chosen
        exp_id = decision.experiment_id or action.experiment_id
        for q in result.expected_outcomes:
            self.pending.add(q, action_id=result.action_id, decision_id=decision.decision_id, experiment_id=exp_id)
        happened = kind == "wait" or (kind == "execute" and result.status in _RAN)
        if kind in ("execute", "wait", "queued"):
            known = {p["prediction_id"] for p in self._open_preds}
            self._open_preds.extend(p for p in preds if p["prediction_id"] not in known)
            self._open_preds = self._open_preds[-5000:]
        if happened:
            mine = preds or [p for p in self._open_preds if p.get("action_id") == action.action_id]
            self._add_learner_queries(decision, result, mine, exp_id)
        if kind == "queued" and approval_id and approval_id not in self.approvals:
            self.approvals[approval_id] = {
                "approval_id": approval_id, "status": "pending", "signature": action_signature(action),
                "queued_at": result.started, "preview": self.log.redact.text(action.render())[:4000],
                "operation": action.operation, "reasons": [result.failure_detail]}
            self._approval_raw[approval_id] = {"decision": decision.to_dict(), "preds": preds, "auth": None}
        elif approval_id and approval_id in self.approvals and kind != "queued":
            row = self.approvals[approval_id]
            row["status"] = "executed" if kind == "execute" else "reported"
            row["result_status"] = result.status
            self._approval_raw.pop(approval_id, None)
        self._agent_try("record_result", decision, result)
        self._dirty = True

    def _add_learner_queries(self, decision: Decision, result: ActionResult, preds: List[Dict[str, Any]],
                             exp_id: str) -> None:
        """Step 4 also measures what the learner itself predicted for an action that happened: a registered
        prediction whose question no query of the environment covers is queued as a query of the learner's."""
        covered = {q.metric for q in result.expected_outcomes} | {q.query_id for q in result.expected_outcomes}
        n = 0
        for p in preds:
            qd = p.get("query")
            if not qd or p.get("action_id") != decision.chosen.action_id:
                continue
            try:
                q = OutcomeQuery.from_dict(qd)
            except Exception:  # noqa: BLE001
                continue
            if q.metric in covered or q.query_id in covered or self.pending.has(q.query_id):
                continue
            horizon, issued = _finite(q.horizon_s), _finite(q.issued_at)
            if (not isinstance(q.metric, str) or not q.metric or len(q.metric) > 200 or not isinstance(q.query_id, str)
                    or not q.query_id or horizon is None or horizon < 0 or horizon > MAX_LEARNER_HORIZON_S):
                continue
            if (n >= MAX_LEARNER_QUERIES_PER_DECISION
                    or self.pending.count_origin("learner") >= MAX_LEARNER_QUERIES_PENDING):
                self.log.event("learner_queries_capped", {"decision_id": decision.decision_id})
                break
            # the question is asked from the moment the action started, never earlier
            q.issued_at = max(issued if issued is not None else result.started, result.started)
            q.decision_id = q.decision_id or decision.decision_id
            self.pending.add(q, action_id=decision.chosen.action_id, decision_id=decision.decision_id,
                             experiment_id=exp_id or q.experiment_id, origin="learner")
            covered.add(q.metric)
            n += 1

    def _drain_agent_records(self) -> None:
        """Experiment records registered by the learner (``AgentAPI.drain_experiments``). They are the learner's
        registrations and assessments: an ``actual`` the learner reports is kept apart as unverified
        (``learner_reported_actual``) — actual outcomes are only the runtime's own ``experiment_outcome`` records —
        and predictions carry no resolution."""
        fn = getattr(self.agent, "drain_experiments", None)
        if not callable(fn):
            return
        try:
            recs = list(fn() or [])
        except Exception as e:  # noqa: BLE001
            self._note_error("learner", "drain_experiments", e)
            return
        for rec in recs:
            if not isinstance(rec, ExperimentRecord):
                continue
            d = rec.to_dict()
            claimed = []
            for o in d.get("actual") or []:
                o = dict(o, verified=False, source=f"learner-reported:{o.get('source', '')}")
                claimed.append(o)
            d["actual"] = []
            if claimed:
                d["learner_reported_actual"] = claimed
            for p in d.get("predicted") or []:
                p.update(resolved_at=None, resolved_value=None, nll=None, resolved_status=OutcomeStatus.PENDING)
            d["assessed_by"] = "learner"
            self.log.append(RunFiles.EXPERIMENTS, "experiment", d, consequential=True)

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
            need = dir_bytes(self._ckpts[-1]["path"]) if self._ckpts and os.path.isdir(self._ckpts[-1]["path"]) else 0
            g = self.guard.check(need, "runtime checkpoint")
            if g["level"] == "warn" and not self._storage_warned:
                self._storage_warned = True
                self.log.event("storage_warning", g, consequential=True)
        except DiskSpaceLow as e:
            self._storage_low(e, "runtime checkpoint")
            return False
        try:
            if os.path.isdir(path):                          # left behind by a commit that never completed
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
            cid = self._agent("save", path)
        except DiskSpaceLow as e:
            self._storage_low(e, "learner.save")
            self._save_state()
            return False
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
        for e in self._wal_carry():                          # journalled under the new generation before the commit
            self._wal_write(e)
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
        for old in self._ckpts[:-1]:                         # optimizer state only in the newest checkpoint
            optim = os.path.join(old["path"], "optim")
            if os.path.isdir(optim):
                shutil.rmtree(optim, ignore_errors=True)
                old["optimizer_state"] = "removed (retention: only the newest checkpoint keeps it)"
        return True

    def _storage_low(self, e: DiskSpaceLow, what: str) -> None:
        """Free disk space is below the minimum: nothing was written. The run pauses with a resource obstacle (logged
        and journalled); the operator frees space and resumes. Experience since the last commit stays in the
        write-ahead journal."""
        detail = {"what": what, "free_gb": round(e.free_gb, 2), "need_gb": round(e.need_gb, 3),
                  "min_free_gb": e.minimum_gb, "path": e.path, "class": "resource", "failure_class": "resource_limit",
                  "explanation": "free disk space at the run directory is below the configured minimum"}
        self.log.event("obstacle", {"kind": "storage_low", **detail}, consequential=True)
        self.log.event("storage_low", detail, consequential=True)
        if self.status == RUNNING:
            self._pause(f"storage: {e}", commit=False)
    def _journal(self) -> None:
        self._last_journal_at = self.clock.now()
        try:
            from .journal import write_journal
            self.log.sync()
            st = self._state()
            st.pop("sealed", None)
            write_journal(self.run_dir, state=st)
        except Exception as e:  # noqa: BLE001 — the journal is derived; failing to render it must not stop the run
            self._note_error("runtime", "journal", e)

    def _settle_until(self) -> Optional[float]:
        """End of the settle window of an objective that ends now (None: no window — finish at once)."""
        w = _finite(self.cfg.settle_after_finish_s) or 0.0
        if w <= 0:
            return None
        until = self.clock.now() + w
        d = self._deadline()
        if d is not None:
            until = min(until, d)
        return until if until > self.clock.now() else None

    def _maturing_by(self, until: float) -> int:
        return sum(1 for i in self.pending.items() if i.ready_at <= until)

    def _end_objective(self, status: str, reason: str) -> None:
        """The objective has ended (``status``). If outcomes still pending can mature within the settle window, keep
        measuring them first (no decision, no execution); otherwise finish now and carry them."""
        until = self._settle_until() if self._settle is None else None
        n = self._maturing_by(until) if until is not None else 0
        if until is None or n == 0:
            self._finish(status, reason)
            return
        now = self.clock.now()
        self._settle = {"status": status, "reason": reason, "since": now, "until": until,
                        "pending_at_end": len(self.pending), "maturing_in_window": n, "measured": 0, "cycles": 0}
        self.log.event("settling", {"objective_id": self.task.objective_id, "status": status, "reason": reason,
                                    "until": until, "pending_outcomes": len(self.pending), "maturing_in_window": n,
                                    "why": "the objective ended with outcomes of its decisions still pending: they "
                                           "are measured (nothing is proposed or executed) before it is closed"},
                       consequential=True)
        self.status_reason = f"settling after '{reason}': measuring {n} pending outcomes until {until:.0f}"
        self._set_activity("settling", f"{n} pending outcomes; no new decisions")
        self._commit(force=True)

    def _settle_over(self) -> bool:
        s = self._settle
        if s is None:
            return False
        return self.clock.now() >= float(s["until"]) or self._maturing_by(float(s["until"])) == 0

    def _close_resolved_approvals(self, status: str, reason: str) -> int:
        """An objective that ends while the operator has already resolved some of its approvals: each is reported
        to the learner once — an approved one as not executed (the objective ended first; nothing runs after the
        end of an objective), a rejected one as denied — so none stays 'approved' in the record forever. Approvals
        still pending at the authorization service are left as they are (counted in ``objective_finished``)."""
        self._refresh_approvals()
        n = 0
        for aid in list(self.approvals):
            a = self.approvals.get(aid)
            if a is None or a["status"] not in ("approved", "rejected"):
                continue
            raw = self._approval_raw.get(aid) or {}
            if not raw.get("decision"):
                continue
            decision = Decision.from_dict(raw["decision"])
            action, preds = decision.chosen, list(raw.get("preds") or [])
            if a["status"] == "approved":
                limit = status == EXHAUSTED
                res = self._synthetic(action, ResultStatus.FAILED if limit else ResultStatus.CANCELLED,
                                      FailureClass.RESOURCE_LIMIT if limit else FailureClass.NONE,
                                      f"approved by the operator, but the objective ended first ({reason}); "
                                      "not executed")
                self._complete(decision, res, "over_limit" if limit else "cancelled", preds, aid, new_decision=False)
            else:
                self._complete(decision, self._synthetic(action, ResultStatus.DENIED, FailureClass.MISSING_PERMISSION,
                                                         f"the operator rejected approval {aid}"),
                               "rejected", preds, aid, new_decision=False)
            n += 1
        return n

    def _finished_logged(self) -> bool:
        """An ``objective_finished`` record of the current objective exists after its start (written before a
        crash that came before the next state save)."""
        try:
            for e in reversed(RunReader(self.run_dir).records(RunFiles.EVENTS, types=["objective_finished",
                                                                                     "objective_assigned",
                                                                                     "run_started"])):
                if e["type"] != "objective_finished":
                    return False
                d = e.get("data") or {}
                if d.get("objective_id") == self.task.objective_id and float(e.get("ts", 0.0)) >= self.started_at:
                    return True
                return False
        except Exception:  # noqa: BLE001 — unreadable log: record normally
            return False
        return False

    def _finish(self, status: str, reason: str) -> None:
        self._finish_wait()
        settle, self._settle = self._settle, None
        closed = self._close_resolved_approvals(status, reason)
        self._commit(force=True)
        self.status, self.status_reason = status, reason
        self._set_activity("finished", reason)
        carried = len(self.pending)
        # G2R-8: after a crash between this record and the next state save, the restarted runtime finishes the
        # objective again: the record exists already and is not written twice
        already = self._finished_logged()
        self.lifetime["objectives"].append({"objective_id": self.task.objective_id, "status": status,
                                            "reason": reason, "at": self.clock.now(),
                                            "decisions": self.counters["decisions"],
                                            "verified": (self._last_verified.get("outcome") or {}).get("value"),
                                            "pending_outcomes_carried": carried})
        if not already:
            self.log.event("objective_finished", {"objective_id": self.task.objective_id, "status": status,
                                                  "reason": reason, "counters": dict(self.counters),
                                                  "limits": self._limit_state(),
                                                  "checkpoint": (self._committed.get("ckpt") or {}).get("id", ""),
                                                  "verified_rid": self._last_verified.get("rid", ""),
                                                  "settle": None if settle is None else
                                                  {k: settle[k] for k in ("since", "until", "pending_at_end",
                                                                          "maturing_in_window", "measured", "cycles")},
                                                  "pending_outcomes_carried": carried,
                                                  "approvals_closed_unexecuted": closed,
                                                  "approvals_pending": sum(1 for a in self.approvals.values()
                                                                           if a.get("status") == "pending")},
                           consequential=True)
        self._journal()
        self._save_state()
        if self._next_tasks:
            self._switch_objective(self._next_tasks.pop(0))
            self._save_state()

    # ==========================================================================================================
    # the loop
    # ==========================================================================================================
    def _compute_budget(self, wanted: float) -> float:
        """The learner's time budget for one learning call: never more than what the compute limit has left (a limit
        of 0 is not set). 0 = the limit is used up: the call is skipped and recorded."""
        lim = _finite(self.task.limits.compute_s) or 0.0
        if lim <= 0:
            return wanted
        left = lim - (float(self.counters["compute_s"]) + self._compute_unbooked)
        if left <= 0:
            self.log.event("learning_skipped", {"reason": "the compute limit is used up", "limit_s": lim})
            return 0.0
        return min(wanted, left)

    def _authorization_revoked(self) -> str:
        """Why the authorization service says access for this task was revoked ('' = it does not)."""
        try:
            st = self.policy.limits_state() or {}
        except Exception as e:  # noqa: BLE001
            self._note_error("authorization", "limits_state", e)
            return ""
        t = (st.get("tasks") or {}).get(self.task.objective_id) if isinstance(st, dict) else None
        reasons = [str(r) for r in ((t or {}).get("unavailable") or [])] if isinstance(t, dict) else []
        hits = [r for r in reasons if "revok" in r.lower()]
        return "; ".join(hits)[:300]

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
        elif self.status in (RUNNING, PAUSED) and "authorization_revoked" in self.task.stop_conditions:
            why = self._authorization_revoked()
            if why:
                self._halt(STOPPED, f"stop condition authorization_revoked: {why}")
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
        if self._redeliver:
            reasons.append("outcomes_restored")
        if self._refresh_approvals() or any(a["status"] in ("approved", "rejected") for a in self.approvals.values()):
            reasons.append("approval_resolved")
        exhausted = self._exhausted()
        settling = self._settle is not None
        draining = not settling and self._may_drain(exhausted)
        if exhausted and not draining and not settling:
            reasons.append(f"limit:{exhausted}")
        if settling and self._settle_over():
            reasons.append("settle_end")
        if draining and not self._draining:
            self._draining = True
            reasons.append("draining")
            self.log.event("draining", {"limit": exhausted, "deadline": self._deadline(),
                                        "pending_outcomes": len(self.pending),
                                        "reason": "decision budget used up with the objective pending: observing "
                                                  "until the verifier's verdict is terminal or the deadline passes"},
                           consequential=True)
            self._set_activity("draining", "observing outcomes; no new decisions")
        if not reasons:
            self._sleep(rep)                                                       # step 17: nothing to do
            return rep

        # -------------------------------------------------------------------------------- a cycle with work
        for hook in list(self.hooks.get("before_cycle", [])):
            try:
                go = hook(self)
            except Exception as e:  # noqa: BLE001
                self._note_error("runtime", f"before_cycle hook {getattr(hook, '__name__', type(hook).__name__)}", e)
                go = True
            if go is False:
                self._triggers = reasons + self._triggers          # the work is still there for the next cycle
                rep["status"] = self.status
                return rep
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
        self._ingest(list(events or []), "poll")                                   # step 3 (deliver)
        self.cursor = cursor
        outcomes = list(self._redeliver) + self._measure(due)                      # step 4
        self._redeliver = []
        # while settling, a verdict that already ended the objective is final; after an exhausted limit the
        # verifier is still asked (the last decisions' consequences may complete the objective)
        verified = (None if settling and self._settle["status"] != EXHAUSTED
                    else self._verify(force=bool(exhausted) and not settling))
        if settling:
            self._settle["measured"] += len(outcomes)
            self._settle["cycles"] += 1
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
        beliefs = self._agent_try("update_beliefs", default={})                    # step 5
        self.log.event("beliefs_updated", {"learner_report": beliefs if isinstance(beliefs, dict) else
                                           {"value": str(beliefs)[:500]}})
        acted = False
        if not terminal and not exhausted and not settling:
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
        budget = self._compute_budget(self.cfg.learn_budget_s)
        updates = [u for u in (self._agent_try("learn", budget, default=[]) or []) if u is not None] \
            if budget > 0 else []                                                  # step 14
        for u in updates:
            self.log.learning_update(u)
        self.obstacles.on_learning([u for u in updates if hasattr(u, "update_id")])
        budget = self._compute_budget(self.cfg.meta_budget_s)
        metas = [m for m in (self._agent_try("meta_learn", budget, default=[]) or []) if m is not None] \
            if budget > 0 else []                                                  # step 15
        for m in metas:
            self.log.meta_update(m)
        if updates or metas:
            self._dirty = True
        for hook in list(self.hooks["after_cycle"]):
            try:
                hook(self)
            except Exception as e:  # noqa: BLE001 — e.g. an unreachable central learner: recorded, the loop goes on
                self._note_error("runtime", f"after_cycle hook {getattr(hook, '__name__', type(hook).__name__)}", e)
        ended = False
        if terminal:                                                               # step 16
            vo = self._last_verified.get("outcome", {})
            why = f"verifier status {vo.get('status')}: value {vo.get('value')} against target {vo.get('target')}"
            if settling:                     # the settled objective's own decisions completed it
                self._settle.update(status=terminal, reason=why)
            else:
                self._end_objective(terminal, why)
                ended = True
        elif exhausted and not draining and not settling:
            self._end_objective(EXHAUSTED, self._exhaust_text(exhausted))
            ended = True
        if self._settle is not None and self.status == RUNNING and self._settle_over():
            self._finish(self._settle["status"], self._settle["reason"])
        elif not ended:
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
        elif self._settle is None and self._exhausted() and not self._may_drain(self._exhausted()):
            self._triggers.append("limit_check")
        if self._triggers:
            return rep
        if self._waiting is None:
            nw = _finite(self._agent_try("next_wake"))
            self._agent_wake_at = nw
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
            "settle_end": None if self._settle is None else float(self._settle["until"]),
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
        rep: Dict[str, Any] = {}
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
