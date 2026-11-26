"""cam_live.harness.env — ``ComputerEnvironment``: the general-purpose computer environment (spec 12.1, 12A).
Workstream EB.

Implements ``contracts.EnvironmentAdapter`` by composing registered ``contracts.ModalityExecutor`` objects (browser,
visual, code, api, skill, internal). The environment itself owns everything that must be identical for every
modality:

* **authorization** — the ``AuthorizationDecision`` is verified with ``PolicyEngine.check_grant`` before anything
  runs, and ``PolicyEngine.check_primitive`` is what executors receive as ``ExecContext.check`` (always bound to the
  granted top-level action, so a program or skill never has more permission than its primitives);
* **observation ids and staleness** — an action that refers to an observation the environment no longer considers
  current fails with ``FailureClass.STALE_OBSERVATION`` before any executor is called;
* **coordination** — an action id is executed at most once (duplicate submissions and crash retries get the recorded
  result; an interrupted state-changing action is never re-run), state-changing actions are serialised, read-only
  work runs concurrently;
* **verification** — ``ActionSpec.verify`` is evaluated against the resulting state, never assumed;
* the **event stream** and its cursor, **pending outcome** bookkeeping and ``measure()``, **resource accounting**,
  the persistent **workspace**, the complete **execution log** (``events.jsonl``; secrets scrubbed) and **restart
  recovery** (journal + state file under ``<run_dir>/env``).

It returns observations and consequences only. There is no reward anywhere in this interface.

Optional executor hooks (duck-typed; see docs/HARNESS_ADAPTER_GUIDE.md): ``bind(env)``, ``observe()``,
``capabilities()``, ``metrics()``, ``pending()``, ``measure(query, now)``, ``poll()``, ``next_wake()``,
``verify(cond, action, ctx)``, ``snapshot_state()`` / ``restore_state(d)``, ``handles_steps``.
"""
from __future__ import annotations

import collections
import fcntl
import json
import os
import re
import threading
import time as _time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..contracts import (CONTRACT_VERSION, EXEC_INTERFACE_VERSION, ActionResult, ActionSpec, AuthorizationDecision,
                         AuthVerdict, Clock, Cost, ExecContext, FailureClass, Level, Limits, Modality, ModalityExecutor,
                         Observation, OutcomeObservation, OutcomeQuery, OutcomeStatus, PolicyEngine, PrimitiveOp,
                         Quantity, ResultStatus, RunFiles, ToolDescriptor, WallClock)
from .perception import Perception

_WRITE = ("external_write",)
_RETRY_SAFE = ("none", "local", "external_read")
_SIDE_RANK = {"none": 0, "local": 1, "external_read": 2, "external_write": 3}
PROGRAM = "program"
MAX_PROGRAM_STEPS, MAX_PROGRAM_DEPTH = 64, 3


def validate_args(schema: Dict[str, Any], value: Any, path: str = "arguments") -> List[str]:
    """Check a value against the JSON-Schema subset used by tool descriptors (type, properties, required,
    additionalProperties, enum, minimum, maximum, items, minLength, maxLength)."""
    errs: List[str] = []
    if not schema:
        return errs
    t = schema.get("type")
    ok = {"object": isinstance(value, dict), "array": isinstance(value, (list, tuple)),
          "string": isinstance(value, str), "boolean": isinstance(value, bool),
          "integer": isinstance(value, int) and not isinstance(value, bool),
          "number": isinstance(value, (int, float)) and not isinstance(value, bool), "null": value is None}
    types = t if isinstance(t, list) else [t] if t else []
    if types and not any(ok.get(x, True) for x in types):
        return [f"{path} must be of type {'/'.join(types)}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path} must be one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path} must be <= {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path} is too long")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}.{k} is required")
        for k, v in value.items():
            if k in props:
                errs += validate_args(props[k], v, f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}.{k} is not a known argument")
    if isinstance(value, (list, tuple)) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value):
            errs += validate_args(schema["items"], v, f"{path}[{i}]")
    return errs


def scrub_with(secrets: Any, text: str) -> str:
    """Remove known secret values from text using whatever the broker handle offers (``scrub`` / ``redact`` /
    an iterable of known values). Without a broker the text is returned unchanged."""
    if secrets is None or not text:
        return text
    for meth in ("scrub", "redact", "scrub_text"):
        fn = getattr(secrets, meth, None)
        if callable(fn):
            out = fn(text)
            return out if isinstance(out, str) else text
    for meth in ("known_values", "values"):
        fn = getattr(secrets, meth, None)
        if callable(fn):
            for v in fn():
                if isinstance(v, (bytes, bytearray)):
                    v = bytes(v).decode("utf-8", "replace")
                if isinstance(v, str) and len(v) >= 4:
                    text = text.replace(v, "[secret]")
    return text


def _scrub_obj(secrets: Any, o: Any) -> Any:
    if isinstance(o, str):
        return scrub_with(secrets, o)
    if isinstance(o, dict):
        return {k: _scrub_obj(secrets, v) for k, v in o.items()}
    if isinstance(o, list):
        return [_scrub_obj(secrets, v) for v in o]
    return o


class _Run:
    """Book-keeping of one action while it executes."""

    def __init__(self, action: ActionSpec):
        self.action = action
        self.done = threading.Event()
        self.cancel = threading.Event()
        self.result: Optional[ActionResult] = None
        self.sunk: List[Observation] = []
        self.checks = 0
        self.finished = False


class ComputerEnvironment:
    """``contracts.EnvironmentAdapter`` over registered modality executors."""
    interface_version = EXEC_INTERFACE_VERSION

    def __init__(self, run_dir: str, policy: PolicyEngine, *, clock: Optional[Clock] = None, secrets: Any = None,
                 limits: Optional[Limits] = None, provenance: str = "staging", record_results: bool = True,
                 perception: Optional[Perception] = None, events_log: Optional[str] = None, name: str = "computer"):
        if provenance not in ("live", "staging", "diagnostic"):
            raise ValueError("provenance must be live, staging or diagnostic")
        self.run_dir = os.path.abspath(run_dir)
        self.policy, self.clock, self.secrets = policy, clock or WallClock(), secrets
        self.limits, self.provenance, self.record_results, self.name = limits, provenance, record_results, name
        self.perception = perception or Perception(self.clock)
        self.workspace = os.path.join(self.run_dir, "workspace")
        self._dir = os.path.join(self.run_dir, "env")
        for d in (self.workspace, self._dir):
            os.makedirs(d, exist_ok=True)
        self._lockf = open(os.path.join(self._dir, "lock"), "a+")
        try:
            fcntl.flock(self._lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lockf.close()
            raise RuntimeError(f"another computer environment already owns {self.run_dir}")
        self._events_path = events_log or os.path.join(self.run_dir, RunFiles.EVENTS)
        self._state_path = os.path.join(self._dir, "state.json")
        self._journal_path = os.path.join(self._dir, "actions.jsonl")
        self._stream_path = os.path.join(self._dir, "stream.jsonl")
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._executors: Dict[str, ModalityExecutor] = {}
        self._tools: Dict[str, Tuple[ToolDescriptor, str]] = {}
        self._sources: List[Any] = []
        self._annotations: Dict[str, Any] = {}
        self._ledger: "collections.OrderedDict[str, Dict[str, Any]]" = collections.OrderedDict()
        self._surface_gen: Dict[str, str] = {}
        self._surface_ts: Dict[str, float] = {}
        self._pending: Dict[str, OutcomeQuery] = {}
        self._journal: Dict[str, Dict[str, Any]] = {}
        self._results: "collections.OrderedDict[str, ActionResult]" = collections.OrderedDict()
        self._inflight: Dict[str, _Run] = {}
        self._stream_offsets: List[int] = []
        self._seq = 0
        self._closed = False
        self._saved_exec_state: Dict[str, Any] = {}
        self._res: Dict[str, Any] = {"actions": 0, "ok": 0, "failed": 0, "denied": 0, "external_actions": 0,
                                     "duplicates_suppressed": 0, "primitive_ops": 0, "primitive_checks": 0,
                                     "seconds": 0.0, "money": 0.0, "compute_s": 0.0, "tokens": 0,
                                     "by_modality": {}, "started_at": self.clock.now()}
        self._load()
        self._log("env.started", provenance=provenance, interface_version=self.interface_version,
                  recovered_interrupted=sorted(a for a, e in self._journal.items() if e["phase"] == "started"),
                  pending_outcomes=len(self._pending), stream_events=len(self._stream_offsets))

    # ---- persistence ------------------------------------------------------------------------------------------
    def _load(self) -> None:
        try:
            with open(self._state_path) as f:
                st = json.load(f)
        except (OSError, ValueError):
            st = {}
        for d in st.get("pending", []):
            q = OutcomeQuery.from_dict(d)
            self._pending[q.query_id] = q
        if st.get("resources"):
            self._res.update(st["resources"])
        self._seq = int(st.get("seq", 0))
        self._saved_exec_state = st.get("executors", {})
        # every observation issued by an earlier process is outdated: its surfaces no longer exist
        self._surface_gen = {}
        if os.path.exists(self._journal_path):
            with open(self._journal_path, "rb") as f:
                while True:
                    off = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue                    # a torn last line from a crash: the action counts as interrupted
                    e = self._journal.setdefault(rec["action_id"], {})
                    e.update(hash=rec["hash"], phase=rec["phase"], side_effects=rec.get("side_effects", "external_write"))
                    if rec["phase"] == "finished":
                        e["offset"] = off
        if os.path.exists(self._stream_path):
            with open(self._stream_path, "rb") as f:
                while True:
                    off = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.endswith(b"\n"):
                        self._stream_offsets.append(off)

    def _save_state(self) -> None:
        with self._lock:
            ex_state = dict(self._saved_exec_state)
            for m, ex in self._executors.items():
                fn = getattr(ex, "snapshot_state", None)
                if callable(fn):
                    try:
                        ex_state[m] = fn()
                    except Exception as e:  # noqa: BLE001 — keep the previous snapshot of that executor
                        self._log("env.error", where=f"snapshot_state:{m}", error=repr(e))
            st = {"version": 1, "saved_at": self.clock.now(), "seq": self._seq,
                  "pending": [q.to_dict() for q in self._pending.values()], "resources": self._res,
                  "executors": ex_state, "stream_events": len(self._stream_offsets)}
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._state_path)

    def _append(self, path: str, rec: Dict[str, Any], sync: bool = False) -> int:
        line = (json.dumps(rec, sort_keys=True, default=str) + "\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            off = os.lseek(fd, 0, os.SEEK_END)
            os.write(fd, line)
            if sync:
                os.fsync(fd)
            return off
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _log(self, kind: str, **fields: Any) -> None:
        """Append one record to the run's events log. Known secret values are removed before it is written."""
        with self._log_lock:
            self._seq += 1
            rec = {"type": kind, "src": "computer_environment", "seq": self._seq, "ts": self.clock.now(),
                   "wall": _time.time(), **fields}
            if self.secrets is not None:
                rec = _scrub_obj(self.secrets, json.loads(json.dumps(rec, default=str)))
            self._append(self._events_path, rec)

    # ---- composition ------------------------------------------------------------------------------------------
    def register(self, executor: ModalityExecutor) -> None:
        """Add a modality executor. One executor per modality; its saved state (if any) is handed back to it."""
        m = executor.modality
        if m not in Modality.ALL:
            raise ValueError(f"unknown modality {m!r}")
        with self._lock:
            if m in self._executors:
                raise ValueError(f"an executor for modality {m!r} is already registered")
            self._executors[m] = executor
            bind = getattr(executor, "bind", None)
            if callable(bind):
                bind(self)
            restore = getattr(executor, "restore_state", None)
            if callable(restore) and m in self._saved_exec_state:
                restore(self._saved_exec_state[m])
            self._refresh_tools()
        self._log("env.executor_registered", modality=m, tools=[t.name for t in executor.tools()])

    def add_source(self, source: Any) -> None:
        """Add a measurement source: any object with some of ``pending()``, ``measure(query, now)``, ``metrics()``."""
        self._sources.append(source)

    def annotate(self, key: str, value: Any) -> None:
        """Add an operator annotation to ``get_capabilities()`` (for example the staging-site labels)."""
        self._annotations[key] = value

    def executor(self, modality: str) -> Optional[ModalityExecutor]:
        return self._executors.get(modality)

    def _refresh_tools(self) -> None:
        tools: Dict[str, Tuple[ToolDescriptor, str]] = {}
        for m, ex in self._executors.items():
            for t in ex.tools():
                tools.setdefault(t.name, (t, m))
        tools[PROGRAM] = (ToolDescriptor(
            name=PROGRAM, modality=Modality.INTERNAL, level=Level.PROCEDURE, side_effects="none",
            description="Run the given steps in order, each through the executor of its own modality, stopping at "
                        "the first step that does not succeed.",
            arg_schema={"type": "object", "properties": {}, "additionalProperties": False}), "")
        self._tools = tools

    def _tool(self, name: str) -> Optional[Tuple[ToolDescriptor, str]]:
        if name not in self._tools:
            with self._lock:
                self._refresh_tools()
        return self._tools.get(name)

    def _hooks(self, name: str) -> List[Tuple[str, Callable[..., Any]]]:
        out = []
        for label, obj in [(m, e) for m, e in self._executors.items()] + \
                          [(getattr(s, "name", type(s).__name__), s) for s in self._sources]:
            fn = getattr(obj, name, None)
            if callable(fn):
                out.append((label, fn))
        return out

    # ---- observations -----------------------------------------------------------------------------------------
    def _register_obs(self, obs: Observation) -> Observation:
        """Scrub, and enter the observation in the staleness ledger."""
        if self.secrets is not None:
            obs.text = scrub_with(self.secrets, obs.text)
            obs.data = _scrub_obj(self.secrets, obs.data)
            obs.refs = _scrub_obj(self.secrets, obs.refs)
            obs.entities = _scrub_obj(self.secrets, obs.entities)
            obs.provenance.uri = scrub_with(self.secrets, obs.provenance.uri)
        surface, gen = obs.data.get("surface"), obs.data.get("generation")
        with self._lock:
            if obs.obs_id in self._ledger:
                return obs
            self._ledger[obs.obs_id] = {"surface": surface, "generation": gen, "ts": obs.ts}
            if surface is not None and obs.ts >= self._surface_ts.get(str(surface), float("-inf")):
                self._surface_gen[str(surface)], self._surface_ts[str(surface)] = gen, obs.ts
            while len(self._ledger) > 4000:
                self._ledger.popitem(last=False)
        return obs

    def is_stale(self, obs_id: str) -> Optional[str]:
        """Why an observation is outdated, or ``None`` if the environment still considers it current."""
        with self._lock:
            e = self._ledger.get(obs_id)
            if e is None:
                return f"observation {obs_id} is unknown to this environment process (expired or from before a restart)"
            if e["surface"] is not None and self._surface_gen.get(str(e["surface"])) != e["generation"]:
                return f"observation {obs_id} describes an earlier state of {e['surface']}"
        return None

    def _stale_reason(self, action: ActionSpec) -> Optional[str]:
        a = action.arguments or {}
        ids = []
        if a.get("obs_id"):
            ids.append(str(a["obs_id"]))
        ref = a.get("ref")
        if isinstance(ref, str) and "@" in ref:
            ids.append(ref.split("@", 1)[1])
        for oid in ids:
            why = self.is_stale(oid)
            if why:
                return why
        return None

    def observe(self) -> List[Observation]:
        out: List[Observation] = []
        for label, fn in self._hooks("observe"):
            try:
                out += list(fn() or [])
            except Exception as e:  # noqa: BLE001 — one unreadable surface does not hide the others
                out.append(self.perception.error(f"{label} could not be observed: {type(e).__name__}",
                                                 source=str(label)))
        m = self.perception.metrics(self.get_observable_metrics(), source=self.name)
        m.data["resources"] = self.get_resource_state()
        out.append(m)
        for o in out:
            self._register_obs(o)
            self._log("env.observation", via="observe", observation=o.to_dict())
        return out

    # ---- tools and validation ---------------------------------------------------------------------------------
    def discover_tools(self) -> List[ToolDescriptor]:
        with self._lock:
            self._refresh_tools()
            return [t for t, _ in self._tools.values()]

    def _side_effects(self, action: ActionSpec, depth: int = 0) -> str:
        if action.operation == PROGRAM or (action.steps and depth < MAX_PROGRAM_DEPTH):
            ranks = [self._side_effects(s, depth + 1) for s in action.steps] or ["none"]
            own = "none"
            if action.operation != PROGRAM:
                t = self._tool(action.operation)
                own = t[0].side_effects if t else "external_write"
            return max(ranks + [own], key=lambda s: _SIDE_RANK.get(s, 3))
        t = self._tool(action.operation)
        return t[0].side_effects if t else "external_write"

    def validate_action(self, action: ActionSpec, _depth: int = 0) -> List[str]:
        errs: List[str] = []
        if not action.action_id or not isinstance(action.action_id, str):
            errs.append("action_id is required")
        if not isinstance(action.arguments, dict):
            return errs + ["arguments must be an object"]
        if not (isinstance(action.timeout_s, (int, float)) and action.timeout_s > 0):
            errs.append("timeout_s must be positive")
        if action.operation == PROGRAM:
            if not action.steps:
                errs.append("a program needs at least one step")
            if _depth >= MAX_PROGRAM_DEPTH:
                errs.append("programs are nested too deeply")
            if self._count_steps(action) > MAX_PROGRAM_STEPS:
                errs.append(f"a program may have at most {MAX_PROGRAM_STEPS} steps")
            seen = {action.action_id}
            for i, s in enumerate(action.steps):
                if s.action_id in seen:
                    errs.append(f"step {i}: duplicate action_id {s.action_id!r}")
                seen.add(s.action_id)
                errs += [f"step {i} ({s.operation}): {e}" for e in self.validate_action(s, _depth + 1)]
            return errs
        t = self._tool(action.operation)
        if t is None:
            return errs + [f"unknown operation {action.operation!r}"]
        tool, modality = t
        if action.modality != modality:
            errs.append(f"operation {action.operation!r} belongs to modality {modality!r}, not {action.modality!r}")
        if not tool.available:
            errs.append(f"operation {action.operation!r} is unavailable: {tool.unavailable_reason}")
        ex = self._executors[modality]
        if action.steps and not getattr(ex, "handles_steps", False):
            errs.append(f"operation {action.operation!r} does not take steps; use {PROGRAM!r}")
        errs += validate_args(tool.arg_schema, action.arguments)
        try:
            errs += [e for e in ex.validate(action) if e not in errs]
        except Exception as e:  # noqa: BLE001
            errs.append(f"validator error: {type(e).__name__}: {e}")
        return errs

    def _count_steps(self, action: ActionSpec) -> int:
        return sum(1 + self._count_steps(s) for s in action.steps)

    # ---- execution --------------------------------------------------------------------------------------------
    def _refuse(self, action: ActionSpec, status: str, fc: str, detail: str, **log: Any) -> ActionResult:
        now = self.clock.now()
        res = ActionResult(action_id=action.action_id, status=status, failure_class=fc, failure_detail=detail,
                           started=now, ended=now)
        obs = self._register_obs(self.perception.error(f"{action.operation} {status}: {detail}", failure_class=fc,
                                                       parent_action_id=action.action_id, source=self.name))
        res.observations.append(obs)
        with self._lock:
            self._res["denied" if status in (ResultStatus.DENIED, ResultStatus.NEEDS_APPROVAL) else "failed"] += 1
        self._log("env.action_refused", action_id=action.action_id, status=status, failure_class=fc, detail=detail, **log)
        self._log("env.observation", via="refusal", action_id=action.action_id, observation=obs.to_dict())
        return res

    def _stored_result(self, action_id: str) -> Optional[ActionResult]:
        with self._lock:
            if action_id in self._results:
                return self._results[action_id]
            e = self._journal.get(action_id)
            if not e or "offset" not in e:
                return None
            with open(self._journal_path, "rb") as f:
                f.seek(e["offset"])
                rec = json.loads(f.readline())
            return ActionResult.from_dict(rec["result"])

    def execute(self, action: ActionSpec, authorization: AuthorizationDecision) -> ActionResult:
        if self._closed:
            return self._refuse(action, ResultStatus.FAILED, FailureClass.UNKNOWN, "the environment is closed")
        ahash = action.content_hash()
        self._log("env.action_submitted", action=action.to_dict(), action_hash=ahash, authorization={
            "verdict": getattr(authorization, "verdict", None), "action_hash": getattr(authorization, "action_hash", ""),
            "policy_ids": getattr(authorization, "policy_ids", []), "expires_at": getattr(authorization, "expires_at", 0.0),
            "approval_id": getattr(authorization, "approval_id", ""), "has_grant": bool(getattr(authorization, "grant", ""))})
        # -- 1. at most once per action id ---------------------------------------------------------------------
        interrupted = False
        with self._lock:
            entry = self._journal.get(action.action_id)
            waiter = self._inflight.get(action.action_id)
            if entry is not None and entry["hash"] != ahash:
                return self._refuse(action, ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS,
                                    "this action_id was already used for a different action")
            if entry is not None and waiter is None:
                if entry["phase"] == "finished":
                    self._res["duplicates_suppressed"] += 1
                    self._log("env.duplicate_suppressed", action_id=action.action_id, how="recorded result returned")
                    return self._stored_result(action.action_id)
                interrupted = True
            run = _Run(action)
        if waiter is not None:
            self._log("env.duplicate_suppressed", action_id=action.action_id, how="joined the execution in progress")
            with self._lock:
                self._res["duplicates_suppressed"] += 1
            waiter.done.wait()
            return waiter.result  # type: ignore[return-value]
        # -- 2. authorization: nothing runs without a verified grant -------------------------------------------
        try:
            if self.policy.stopped():
                return self._refuse(action, ResultStatus.DENIED, FailureClass.MISSING_PERMISSION,
                                    "the authorization service is stopped (emergency stop)")
            if authorization is None or not isinstance(authorization, AuthorizationDecision):
                return self._refuse(action, ResultStatus.DENIED, FailureClass.MISSING_PERMISSION, "no authorization")
            if authorization.verdict == AuthVerdict.NEEDS_APPROVAL:
                return self._refuse(action, ResultStatus.NEEDS_APPROVAL, FailureClass.NONE,
                                    "; ".join(authorization.reasons) or "waiting for operator approval",
                                    approval_id=authorization.approval_id)
            if authorization.verdict != AuthVerdict.ALLOW:
                words = " ".join(authorization.reasons + authorization.policy_ids).lower()
                return self._refuse(action, ResultStatus.DENIED, FailureClass.PLATFORM_RESTRICTION
                                    if "platform" in words else FailureClass.MISSING_PERMISSION,
                                    "; ".join(authorization.reasons) or "denied by the authorization layer")
            if authorization.expires_at and self.clock.now() > authorization.expires_at:
                return self._refuse(action, ResultStatus.DENIED, FailureClass.MISSING_PERMISSION, "the grant has expired")
            if not self.policy.check_grant(action, authorization):
                return self._refuse(action, ResultStatus.DENIED, FailureClass.MISSING_PERMISSION,
                                    "the grant does not verify for this action")
        except Exception as e:  # noqa: BLE001 — fail closed when the authorization service cannot be asked
            return self._refuse(action, ResultStatus.DENIED, FailureClass.MISSING_PERMISSION,
                                f"authorization could not be verified: {type(e).__name__}")
        # -- 3. structure, limits, staleness -----------------------------------------------------------------
        errs = self.validate_action(action)
        if errs:
            return self._refuse(action, ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS, "; ".join(errs))
        side = self._side_effects(action)
        if interrupted:
            with self._lock:
                safe = self._journal[action.action_id].get("side_effects", "external_write") in _RETRY_SAFE
            if not safe:
                now = self.clock.now()
                res = ActionResult(action_id=action.action_id, status=ResultStatus.FAILED,
                                   failure_class=FailureClass.UNKNOWN, started=now, ended=now,
                                   failure_detail="this action was interrupted by a restart before its result was "
                                                  "recorded; its effect is unknown and it is not executed again")
                res.observations.append(self._register_obs(self.perception.error(
                    res.failure_detail, failure_class=res.failure_class, parent_action_id=action.action_id,
                    source=self.name)))
                self._finish(run, action, res, ahash, side)
                return res
            self._log("env.recovery", action_id=action.action_id, how="interrupted read-only action is run again")
        lim = self._limit_reason(action, side)
        if lim:
            return self._refuse(action, ResultStatus.FAILED, FailureClass.RESOURCE_LIMIT, lim)
        for a in [action] + self._flatten(action):
            why = self._stale_reason(a)
            if why:
                return self._refuse(action, ResultStatus.FAILED, FailureClass.STALE_OBSERVATION, why)
        # -- 4. run ---------------------------------------------------------------------------------------------
        with self._lock:
            again = self._inflight.get(action.action_id)
            if again is not None:
                run = again
            else:
                self._inflight[action.action_id] = run
                self._journal[action.action_id] = {"hash": ahash, "phase": "started", "side_effects": side}
        if again is not None:
            run.done.wait()
            return run.result  # type: ignore[return-value]
        self._append(self._journal_path, {"action_id": action.action_id, "hash": ahash, "phase": "started",
                                          "side_effects": side, "ts": self.clock.now()}, sync=True)
        self._log("env.action_started", action_id=action.action_id, operation=action.operation,
                  modality=action.modality, side_effects=side, serialised=side in _WRITE)
        try:
            if side in _WRITE:
                with self._write_lock:
                    res = self._run_top(run, action, authorization)
            else:
                res = self._run_top(run, action, authorization)
        except BaseException as e:  # noqa: BLE001 — never leave waiters hanging; report and re-raise fatal ones
            now = self.clock.now()
            res = ActionResult(action_id=action.action_id, status=ResultStatus.FAILED, started=now, ended=now,
                               failure_class=FailureClass.UNKNOWN, failure_detail=f"environment error: {e!r}")
            self._finish(run, action, res, ahash, side)
            if not isinstance(e, Exception):
                raise
            return res
        self._finish(run, action, res, ahash, side)
        return res

    def _flatten(self, action: ActionSpec) -> List[ActionSpec]:
        out: List[ActionSpec] = []
        for s in action.steps:
            out.append(s)
            out += self._flatten(s)
        return out

    def _limit_reason(self, action: ActionSpec, side: str) -> Optional[str]:
        lim = self.limits
        if lim is None:
            return None
        with self._lock:
            r = self._res
            if lim.max_external_actions and side in _WRITE and r["external_actions"] >= lim.max_external_actions:
                return f"external action limit reached ({lim.max_external_actions})"
            if lim.money and r["money"] + action.est_cost.money > lim.money + 1e-9:
                return f"money limit reached ({lim.money})"
            if lim.compute_s and r["compute_s"] >= lim.compute_s:
                return f"compute limit reached ({lim.compute_s} s)"
            if lim.duration_s and self.clock.now() - r["started_at"] >= lim.duration_s:
                return f"duration limit reached ({lim.duration_s} s)"
        return None

    def _context(self, run: _Run, top: ActionSpec, authorization: Optional[AuthorizationDecision] = None) -> ExecContext:
        def check(op: PrimitiveOp, _action: Optional[ActionSpec] = None) -> AuthorizationDecision:
            try:
                dec = self.policy.check_primitive(op, top)
            except Exception as e:  # noqa: BLE001 — fail closed
                dec = AuthorizationDecision(AuthVerdict.DENY, reasons=[f"authorization check failed: {type(e).__name__}"])
            if not isinstance(dec, AuthorizationDecision):
                dec = AuthorizationDecision(AuthVerdict.DENY, reasons=["authorization check returned no decision"])
            run.checks += 1
            self._log("env.primitive_check", action_id=top.action_id, op=op.to_dict(), verdict=dec.verdict,
                      reasons=dec.reasons, policy_ids=dec.policy_ids)
            return dec

        def sink(obs: Observation) -> None:
            if run.finished:
                self.emit_event(obs)            # late delivery (e.g. a background job): it is an event now
            else:
                run.sunk.append(obs)

        now = self.clock.now()
        ctx = ExecContext(clock=self.clock, check=check, obs_sink=sink, run_dir=self.run_dir,
                          deadline=now + float(top.timeout_s or 0.0), cancel=run.cancel, secrets=self.secrets)
        # extensions (see docs/change_requests/EB.md): sub-step dispatch for skill executors, the workspace path and
        # the action's grant (secret brokers release a secret only against it)
        ctx.dispatch = lambda step: self._dispatch(run, top, step, ctx)   # type: ignore[attr-defined]
        ctx.workspace = self.workspace                                    # type: ignore[attr-defined]
        ctx.authorization = authorization                                 # type: ignore[attr-defined]
        return ctx

    def _dispatch(self, run: _Run, top: ActionSpec, action: ActionSpec, ctx: ExecContext, depth: int = 0) -> ActionResult:
        """Run one action (or step) through the executor of its modality; programs recurse."""
        now = self.clock.now()
        if action.operation == PROGRAM:
            agg = ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=now, ended=now)
            for i, step in enumerate(action.steps):
                if run.cancel.is_set():
                    agg.status, agg.failure_detail = ResultStatus.CANCELLED, f"cancelled before step {i}"
                    break
                why = self._stale_reason(step)
                if why:
                    r = ActionResult(action_id=step.action_id, status=ResultStatus.FAILED,
                                     failure_class=FailureClass.STALE_OBSERVATION, failure_detail=why, started=self.clock.now())
                else:
                    r = self._dispatch(run, top, step, ctx, depth + 1)
                for o in r.observations:
                    self._register_obs(o)       # so later steps may refer to observations of earlier ones
                agg.observations += r.observations
                agg.ops += r.ops
                agg.artifacts.update(r.artifacts)
                agg.expected_outcomes += r.expected_outcomes
                agg.retries += r.retries
                for f in ("money", "compute_s", "tokens"):
                    setattr(agg.cost, f, getattr(agg.cost, f) + getattr(r.cost, f))
                if r.status != ResultStatus.OK:
                    agg.status, agg.failure_class = r.status, r.failure_class
                    agg.failure_detail = f"step {i} ({step.operation}): {r.failure_detail}"
                    break
                if step.verify.kind != "none":
                    v = self._verify(run, top, step, r, ctx)
                    if v is False:
                        agg.status = ResultStatus.FAILED
                        agg.failure_class = FailureClass.WRONG_SITE_ASSUMPTION
                        agg.failure_detail = f"step {i} ({step.operation}): verification condition not met"
                        break
            agg.ended = self.clock.now()
            return agg
        t = self._tool(action.operation)
        if t is None:
            return ActionResult(action_id=action.action_id, status=ResultStatus.FAILED, started=now, ended=now,
                                failure_class=FailureClass.INVALID_ARGUMENTS,
                                failure_detail=f"unknown operation {action.operation!r}")
        ex = self._executors[t[1]]
        try:
            res = ex.execute(action, ctx)
        except Exception as e:  # noqa: BLE001 — an executor bug is a failed action, not a dead environment
            res = ActionResult(action_id=action.action_id, status=ResultStatus.FAILED, started=now,
                               failure_class=FailureClass.UNKNOWN, failure_detail=f"executor error: {type(e).__name__}: {e}")
        if not isinstance(res, ActionResult):
            res = ActionResult(action_id=action.action_id, status=ResultStatus.FAILED, started=now,
                               failure_class=FailureClass.UNKNOWN, failure_detail="executor returned no result")
        res.action_id = action.action_id
        res.started = res.started or now
        res.ended = res.ended or self.clock.now()
        if res.status == ResultStatus.DENIED and res.failure_class not in FailureClass.HARD:
            res.failure_class = FailureClass.MISSING_PERMISSION
        return res

    def _run_top(self, run: _Run, action: ActionSpec, authorization: AuthorizationDecision) -> ActionResult:
        ctx = self._context(run, action, authorization)
        t0 = self.clock.now()
        res = self._dispatch(run, action, action, ctx)
        res.action_id = action.action_id
        have = {o.obs_id for o in res.observations}
        res.observations += [o for o in run.sunk if o.obs_id not in have]
        side = self._side_effects(action)
        if res.status == ResultStatus.OK and side in _WRITE and not any(o.kind != "error" for o in res.observations):
            # consequential action without a view of the outcome: look at the resulting state
            t = self._tool(action.operation)
            fn = getattr(self._executors.get(t[1]) if t else None, "observe", None)
            if callable(fn):
                try:
                    res.observations += list(fn() or [])
                except Exception as e:  # noqa: BLE001
                    self._log("env.error", where="post-action observe", error=repr(e))
        if res.status == ResultStatus.OK and action.verify.kind != "none":
            res.verified = self._verify(run, action, action, res, ctx)
            if res.verified is False:
                res.status = ResultStatus.FAILED
                res.failure_class = (FailureClass.STRATEGIC if action.level == Level.INTERVENTION
                                     else FailureClass.WRONG_SITE_ASSUMPTION)
                res.failure_detail = ("the action ran but its verification condition is not met in the resulting "
                                      f"state: {action.verify.description or action.verify.kind}")
        if res.status != ResultStatus.OK and not res.observations:
            res.observations.append(self.perception.error(
                f"{action.operation} {res.status}: {res.failure_detail}", failure_class=res.failure_class,
                parent_action_id=action.action_id, source=self.name))
        res.started = res.started or t0
        res.ended = self.clock.now()
        res.cost.seconds = max(res.cost.seconds, res.ended - res.started)
        res.cost.primitive_ops = max(res.cost.primitive_ops, len(res.ops))
        return res

    # ---- verification -----------------------------------------------------------------------------------------
    def _verify(self, run: _Run, top: ActionSpec, action: ActionSpec, res: ActionResult, ctx: ExecContext) -> Optional[bool]:
        cond, spec = action.verify, action.verify.spec or {}
        try:
            if cond.kind == "program":
                steps = [ActionSpec.from_dict(s) if isinstance(s, dict) else s for s in spec.get("steps", [])]
                if not steps:
                    return None
                for s in steps:
                    if self.validate_action(s) or self._side_effects(s) not in _RETRY_SAFE:
                        self._log("env.verify", action_id=action.action_id, note="verification program rejected: "
                                  "it must consist of valid read-only steps")
                        return None
                    r = self._dispatch(run, top, s, ctx)
                    res.ops += r.ops
                    res.observations += r.observations
                    if r.status != ResultStatus.OK:
                        return False
                    if s.verify.kind != "none" and self._verify(run, top, s, r, ctx) is False:
                        return False
                return True
            target = action
            if action.operation == PROGRAM and action.steps:
                target = action.steps[-1]
            t = self._tool(target.operation)
            fn = getattr(self._executors.get(t[1]) if t else None, "verify", None)
            if callable(fn):
                v = fn(cond, target, ctx)
                if v is not None:
                    return bool(v)
            obs = [o for o in res.observations if o.kind != "error"]
            if cond.kind == "text_present":
                hay, want = "\n".join(o.text for o in obs), str(spec.get("text", ""))
                if not spec.get("case_sensitive", True):
                    hay, want = hay.lower(), want.lower()
                return (want in hay) != bool(spec.get("absent", False))
            if cond.kind == "url_matches":
                urls = [o.data.get("url") or o.provenance.uri for o in obs if o.data.get("url") or o.provenance.uri]
                return bool(re.search(str(spec.get("pattern", "")), urls[-1])) if urls else None
            if cond.kind == "element_state":
                pages = [o for o in obs if o.kind == "page"]
                if not pages:
                    return None
                els = [e for e in pages[-1].data.get("elements", [])
                       if (not spec.get("role") or e.get("role") == spec["role"])
                       and (spec.get("name") is None or e.get("name") == spec["name"])]
                state = spec.get("state", "visible")
                if state in ("hidden", "absent", "detached"):
                    return not els
                if not els:
                    return False
                e = els[0]
                return {"visible": True, "checked": e.get("checked") is True, "unchecked": e.get("checked") is False,
                        "enabled": not e.get("disabled"), "disabled": bool(e.get("disabled")),
                        "value": str(e.get("value", "")) == str(spec.get("value", ""))}.get(state)
            if cond.kind == "metric":
                vals = {q.name: q.value for q in self.get_observable_metrics()}
                for o in obs:
                    vals.update({q.name: q.value for q in o.quantities})
                if spec.get("name") not in vals:
                    return None
                v, ref = vals[spec["name"]], float(spec.get("value", 0.0))
                return {">=": v >= ref, "<=": v <= ref, ">": v > ref, "<": v < ref, "==": v == ref,
                        "!=": v != ref}.get(str(spec.get("op", "==")))
        except Exception as e:  # noqa: BLE001 — an unevaluable condition is "not checked", never "passed"
            self._log("env.error", where="verify", action_id=action.action_id, error=repr(e))
        return None

    # ---- completion -------------------------------------------------------------------------------------------
    def _finish(self, run: _Run, action: ActionSpec, res: ActionResult, ahash: str, side: str) -> None:
        now = self.clock.now()
        res.observations = [self._register_obs(o) for o in res.observations]
        for o in res.observations:
            o.parent_action_id = o.parent_action_id or action.action_id
        for q in res.expected_outcomes:
            q.issued_at = q.issued_at or now
            q.decision_id = q.decision_id or action.parent_decision_id
            q.experiment_id = q.experiment_id or action.experiment_id
        if self.secrets is not None:
            res.failure_detail = scrub_with(self.secrets, res.failure_detail)
            res.artifacts = _scrub_obj(self.secrets, res.artifacts)
            for op in res.ops:
                op.arguments, op.note = _scrub_obj(self.secrets, op.arguments), scrub_with(self.secrets, op.note)
        with self._lock:
            for q in res.expected_outcomes:
                self._pending[q.query_id] = q
            r = self._res
            r["actions"] += 1
            r["ok" if res.status == ResultStatus.OK else
              "denied" if res.status in (ResultStatus.DENIED, ResultStatus.NEEDS_APPROVAL) else "failed"] += 1
            if side in _WRITE and res.status != ResultStatus.DENIED and (res.ops or res.status == ResultStatus.OK):
                r["external_actions"] += 1
            r["primitive_ops"] += len(res.ops)
            r["primitive_checks"] += run.checks
            for f in ("seconds", "money", "compute_s", "tokens"):
                r[f] += getattr(res.cost, f)
            bm = r["by_modality"].setdefault(action.modality, {"actions": 0, "ok": 0, "seconds": 0.0, "primitive_ops": 0})
            bm["actions"] += 1
            bm["ok"] += int(res.status == ResultStatus.OK)
            bm["seconds"] += res.cost.seconds
            bm["primitive_ops"] += len(res.ops)
        for op in res.ops:
            self._log("env.primitive_op", action_id=action.action_id, op=op.to_dict())
        for o in res.observations:
            self._log("env.observation", via="action", action_id=action.action_id, observation=o.to_dict())
        self._log("env.action_result", action_id=action.action_id, status=res.status, failure_class=res.failure_class,
                  failure_detail=res.failure_detail, verified=res.verified, cost=res.cost.to_dict(),
                  artifacts=res.artifacts, retries=res.retries, started=res.started, ended=res.ended,
                  observation_ids=[o.obs_id for o in res.observations], primitive_ops=len(res.ops),
                  primitive_checks=run.checks, expected_outcomes=[q.to_dict() for q in res.expected_outcomes])
        off = self._append(self._journal_path, {"action_id": action.action_id, "hash": ahash, "phase": "finished",
                                                "side_effects": side, "ts": now, "result": res.to_dict()}, sync=True)
        with self._lock:
            self._journal[action.action_id] = {"hash": ahash, "phase": "finished", "side_effects": side, "offset": off}
            self._results[action.action_id] = res
            while len(self._results) > 256:
                self._results.popitem(last=False)
            run.result, run.finished = res, True
            self._inflight.pop(action.action_id, None)
        run.done.set()
        if self.record_results:
            try:
                self.policy.record(action, res)
            except Exception as e:  # noqa: BLE001 — the audit failure is logged here; the action already happened
                self._log("env.error", where="policy.record", action_id=action.action_id, error=repr(e))
        self._save_state()

    def cancel_action(self, action_id: str) -> bool:
        """Ask a running action to stop (executors poll ``ExecContext.cancel``)."""
        with self._lock:
            run = self._inflight.get(action_id)
        if run is None:
            return False
        run.cancel.set()
        self._log("env.cancel_requested", action_id=action_id)
        return True

    # ---- events -----------------------------------------------------------------------------------------------
    def emit_event(self, obs: Observation) -> str:
        """Append an observation to the event stream; returns the cursor that points just past it."""
        self._register_obs(obs)
        with self._lock:
            off = self._append(self._stream_path, obs.to_dict(), sync=True)
            self._stream_offsets.append(off)
            cur = str(len(self._stream_offsets))
        self._log("env.event", observation=obs.to_dict(), cursor=cur)
        return cur

    def poll_events(self, cursor: str) -> Tuple[List[Observation], str]:
        for label, fn in self._hooks("poll"):
            try:
                for o in fn() or []:
                    self.emit_event(o)
            except Exception as e:  # noqa: BLE001
                self._log("env.error", where=f"poll:{label}", error=repr(e))
        try:
            start = max(0, int(cursor or 0))
        except ValueError:
            raise ValueError(f"bad event cursor {cursor!r}")
        with self._lock:
            n = len(self._stream_offsets)
            start = min(start, n)
            end = min(n, start + 500)
            out: List[Observation] = []
            if end > start:
                with open(self._stream_path, "rb") as f:
                    f.seek(self._stream_offsets[start])
                    for _ in range(end - start):
                        out.append(Observation.from_dict(json.loads(f.readline())))
        return out, str(end)

    # ---- resources, outcomes, metrics ---------------------------------------------------------------------------
    def get_resource_state(self) -> Dict[str, Any]:
        with self._lock:
            r = json.loads(json.dumps(self._res))
        now = self.clock.now()
        r.update(time_elapsed_s=now - r["started_at"], pending_outcomes=len(self._pending),
                 in_flight=len(self._inflight), workspace=self.workspace)
        wakes = []
        for _, fn in self._hooks("next_wake"):
            try:
                w = fn()
                if w is not None:
                    wakes.append(float(w))
            except Exception:  # noqa: BLE001
                continue
        due = [q.due_at for q in self._pending.values()]
        r["next_wake_at"] = min(wakes) if wakes else None
        r["next_outcome_due_at"] = min(due) if due else None
        lim = self.limits
        if lim is not None:
            r["limits"] = lim.to_dict()
            r["remaining"] = {
                "money": (lim.money - r["money"]) if lim.money else None,
                "compute_s": (lim.compute_s - r["compute_s"]) if lim.compute_s else None,
                "time_s": (lim.duration_s - r["time_elapsed_s"]) if lim.duration_s else None,
                "external_actions": (lim.max_external_actions - r["external_actions"]) if lim.max_external_actions else None}
        try:
            r["policy_limits"] = self.policy.limits_state()
        except Exception as e:  # noqa: BLE001
            r["policy_limits"] = {"unavailable": type(e).__name__}
        return r

    def add_pending(self, query: OutcomeQuery) -> None:
        with self._lock:
            self._pending[query.query_id] = query
        self._save_state()

    def drop_pending(self, query_id: str) -> bool:
        with self._lock:
            found = self._pending.pop(query_id, None) is not None
        if found:
            self._save_state()
        return found

    def get_pending_outcomes(self) -> List[OutcomeQuery]:
        new = False
        for label, fn in self._hooks("pending"):
            try:
                for q in fn() or []:
                    with self._lock:
                        if q.query_id not in self._pending:
                            self._pending[q.query_id], new = q, True
            except Exception as e:  # noqa: BLE001
                self._log("env.error", where=f"pending:{label}", error=repr(e))
        if new:
            self._save_state()
        with self._lock:
            return sorted(self._pending.values(), key=lambda q: q.due_at)

    def get_observable_metrics(self) -> List[Quantity]:
        out: List[Quantity] = []
        seen = set()
        for label, fn in self._hooks("metrics"):
            try:
                for q in fn() or []:
                    if q.name not in seen:
                        seen.add(q.name)
                        out.append(q)
            except Exception as e:  # noqa: BLE001
                self._log("env.error", where=f"metrics:{label}", error=repr(e))
        with self._lock:
            r = self._res
            out += [Quantity("env.actions", float(r["actions"]), kind="count"),
                    Quantity("env.external_actions", float(r["external_actions"]), kind="count"),
                    Quantity("env.primitive_ops", float(r["primitive_ops"]), kind="count"),
                    Quantity("env.pending_outcomes", float(len(self._pending)), kind="count")]
        return out

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        """Environment-side (unverified) measurements for due queries. Never invents a value: a query nobody can
        measure is reported ``unobservable``."""
        now, out = self.clock.now(), []
        measurers = self._hooks("measure")
        for q in queries:
            if now < q.due_at:
                out.append(OutcomeObservation(q.query_id, OutcomeStatus.PENDING, source=self.name))
                continue
            got: Optional[OutcomeObservation] = None
            for label, fn in measurers:
                try:
                    got = fn(q, now)
                except Exception as e:  # noqa: BLE001
                    self._log("env.error", where=f"measure:{label}", error=repr(e))
                    got = None
                if got is not None:
                    got.source = got.source or str(label)
                    break
            if got is None:
                vals = {m.name: m for m in self.get_observable_metrics()}
                if q.metric in vals:
                    got = OutcomeObservation(q.query_id, OutcomeStatus.OBSERVED, float(vals[q.metric].value),
                                             observed_at=now, source=f"{self.name}.metric_at_measurement",
                                             dedup_key=f"{q.metric}@{q.due_at:.0f}")
                else:
                    got = OutcomeObservation(q.query_id, OutcomeStatus.UNOBSERVABLE, observed_at=now, source=self.name)
            got.query_id, got.verified = q.query_id, False
            if got.status in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED):
                got.observed_at = got.observed_at or now
                with self._lock:
                    self._pending.pop(q.query_id, None)
            out.append(got)
        self._log("env.measure", outcomes=[o.to_dict() for o in out])
        self._save_state()
        return out

    def get_capabilities(self) -> Dict[str, Any]:
        caps: Dict[str, Any] = {
            "name": self.name, "interface_version": self.interface_version, "contract_version": CONTRACT_VERSION,
            "provenance": self.provenance, "live": self.provenance == "live",
            "modalities": sorted(self._executors), "tools": len(self.discover_tools()),
            "run_dir": self.run_dir, "workspace": self.workspace, "simulated_time": hasattr(self.clock, "advance"),
            "features": {"at_most_once_by_action_id": True, "serialised_external_writes": True,
                         "stale_observation_detection": True, "programs": True, "restart_recovery": True,
                         "execution_log": self._events_path, "secret_scrubbing": self.secrets is not None,
                         "reward": False},
            "sources": [getattr(s, "name", type(s).__name__) for s in self._sources], "executors": {}}
        for m, ex in self._executors.items():
            fn = getattr(ex, "capabilities", None)
            try:
                caps["executors"][m] = fn() if callable(fn) else {}
            except Exception as e:  # noqa: BLE001
                caps["executors"][m] = {"error": type(e).__name__}
        caps.update(self._annotations)
        return caps

    # ---- own records (used by the internal executor) -------------------------------------------------------------
    def records(self, kind: str = "", limit: int = 50, action_id: str = "") -> List[Dict[str, Any]]:
        """The newest ``limit`` records of this environment's execution log, optionally filtered."""
        out: "collections.deque[Dict[str, Any]]" = collections.deque(maxlen=max(1, min(int(limit), 500)))
        try:
            with open(self._events_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec.get("src") != "computer_environment":
                        continue
                    if kind and rec.get("type") != kind:
                        continue
                    if action_id and rec.get("action_id") != action_id and \
                            (rec.get("action") or {}).get("action_id") != action_id:
                        continue
                    out.append(rec)
        except OSError:
            pass
        return list(out)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._save_state()
        finally:
            for m, ex in list(self._executors.items()):
                try:
                    ex.close()
                except Exception as e:  # noqa: BLE001
                    self._log("env.error", where=f"close:{m}", error=repr(e))
            self._log("env.closed")
            try:
                fcntl.flock(self._lockf, fcntl.LOCK_UN)
            finally:
                self._lockf.close()

    def __enter__(self) -> "ComputerEnvironment":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
