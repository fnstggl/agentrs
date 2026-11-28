"""cam_live.harness.internal — the ``internal`` executor: wait, inspect own records, note, ask the operator.
Workstream EB (``internal.request_operator`` added by the master, see ``harness/operator_requests.py``).

None of these touch the outside world. ``wait`` *schedules* a wake-up and returns at once: waiting for delayed
consequences costs no inference and produces no public activity (spec 12A.11). The scheduled time is exposed
through ``ComputerEnvironment.get_resource_state()["next_wake_at"]`` and an event is emitted on the stream when it
passes, so the runtime can sleep until then. With a simulated clock (``SimClock``) the wait advances that clock
instead, so staging runs mature delayed outcomes in milliseconds.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional

from ..contracts import (ActionResult, ActionSpec, AuthVerdict, Clock, Cost, ExecContext, FailureClass, Level, Modality,
                         Observation, PrimitiveOp, Quantity, ResultStatus, ToolDescriptor, Trust, WallClock, new_id)
from .operator_requests import KINDS as REQUEST_KINDS, OperatorRequests
from .perception import Perception

DAY = 86400.0
_KINDS = ("actions", "action", "results", "events", "pending", "resources", "notes", "log", "tools", "blob")


class InternalExecutor:
    """``contracts.ModalityExecutor`` for ``Modality.INTERNAL``."""
    modality = Modality.INTERNAL

    def __init__(self, run_dir: str, *, clock: Optional[Clock] = None, advance_sim_clock: bool = True,
                 max_block_s: float = 30.0, max_wait_s: float = 30 * DAY, perception: Optional[Perception] = None):
        self.run_dir = os.path.abspath(run_dir)
        self.clock = clock or WallClock()
        self.advance_sim_clock, self.max_block_s, self.max_wait_s = advance_sim_clock, max_block_s, max_wait_s
        self.perception = perception or Perception(self.clock)
        self._env: Any = None
        self._lock = threading.Lock()
        self._wakes: List[Dict[str, Any]] = []
        self._notes_path = os.path.join(self.run_dir, "workspace", "notes.jsonl")
        os.makedirs(os.path.dirname(self._notes_path), exist_ok=True)
        self.requests = OperatorRequests(self.run_dir, now=lambda: self.clock.now())

    # ---- hooks --------------------------------------------------------------------------------------------------
    def bind(self, env: Any) -> None:
        self._env = env
        self.clock = env.clock

    def snapshot_state(self) -> Dict[str, Any]:
        with self._lock:
            return {"wakes": [dict(w) for w in self._wakes]}

    def restore_state(self, state: Dict[str, Any]) -> None:
        with self._lock:
            self._wakes = [dict(w) for w in (state or {}).get("wakes", [])]

    def next_wake(self) -> Optional[float]:
        with self._lock:
            return min((w["wake_at"] for w in self._wakes), default=None)

    def poll(self) -> List[Observation]:
        """Wake-ups whose time has come, as events (each exactly once)."""
        now = self.clock.now()
        with self._lock:
            due = [w for w in self._wakes if w["wake_at"] <= now]
            self._wakes = [w for w in self._wakes if w["wake_at"] > now]
        out = [self.perception.event("wait.finished", f"scheduled wait finished ({w.get('reason') or 'no reason given'})",
                                     quantities=[Quantity("wait.seconds", float(w["seconds"]), unit="s")],
                                     data={"action_id": w["action_id"], "wake_at": w["wake_at"]}, source="internal",
                                     parent_action_id=w["action_id"]) for w in due]
        for r in self.requests.undelivered_answers():      # the operator answered a request made earlier
            out += self.answer_events(r, now)
        return out

    def answer_events(self, r: Dict[str, Any], now: float) -> List[Observation]:
        """An operator's answer as two observations: what the operator decided (status, the operator's own note)
        under ``trust=operator``, and — separately, under ``trust=self`` — the request as the learner wrote it
        (title, description), so nothing the learner wrote is ever returned under the operator's label."""
        note = str(r.get("resolution_note") or "")
        op = self.perception.event(
            "operator.answer", f"operator {r['status']} request {r['request_id']} ({r['kind']})"
            + (f"; operator note: {note}" if note else ""),
            quantities=[Quantity("operator.request.granted", 1.0 if r["status"] == "granted" else 0.0, kind="binary")],
            data={"request_id": r["request_id"], "status": r["status"], "kind": r["kind"], "operator_note": note,
                  "waited_s": float(r.get("resolved_at", now)) - float(r.get("ts", now))},
            source="operator", parent_action_id=r.get("action_id", ""))
        op.provenance.trust = Trust.OPERATOR
        own = self.perception.event(
            "operator.request.context", f"request {r['request_id']} as asked: \"{r.get('title', '')}\"",
            data={"request_id": r["request_id"], "title": r.get("title", ""),
                  "description": str(r.get("description", ""))[:4000], "answer_obs_id": op.obs_id},
            source="internal", parent_action_id=r.get("action_id", ""))
        own.provenance.trust = Trust.SELF
        op.refs["request_context"] = own.obs_id
        return [op, own]

    # ---- ModalityExecutor -----------------------------------------------------------------------------------------
    def tools(self) -> List[ToolDescriptor]:
        return [
            ToolDescriptor("wait", Modality.INTERNAL, "Schedule a wake-up after a number of seconds and return. "
                           "Nothing is sent anywhere while waiting.",
                           {"type": "object", "properties": {
                               "seconds": {"type": "number", "minimum": 0, "maximum": self.max_wait_s},
                               "reason": {"type": "string", "maxLength": 400},
                               "block": {"type": "boolean", "description": "sleep inside the call (short waits only)"}},
                            "required": ["seconds"], "additionalProperties": False},
                           Level.PRIMITIVE, "", "none", Cost()),
            ToolDescriptor("internal.inspect", Modality.INTERNAL, "Read this environment's own records: recent "
                           "actions and results, one action in full, events, pending outcomes, resources, notes, "
                           "tools, or a slice of the full payload of an observation that was too large (what=blob).",
                           {"type": "object", "properties": {
                               "what": {"type": "string", "enum": list(_KINDS)},
                               "action_id": {"type": "string"},
                               "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                               "blob": {"type": "string", "description": "refs.data_blob of a bounded observation"},
                               "offset": {"type": "integer", "minimum": 0},
                               "max_chars": {"type": "integer", "minimum": 1, "maximum": 60000}},
                            "required": ["what"], "additionalProperties": False},
                           Level.PRIMITIVE, "", "none", Cost()),
            ToolDescriptor("internal.note", Modality.INTERNAL, "Append a note to the persistent workspace notebook.",
                           {"type": "object", "properties": {
                               "text": {"type": "string", "minLength": 1, "maxLength": 8000},
                               "tags": {"type": "array", "items": {"type": "string", "maxLength": 60}}},
                            "required": ["text"], "additionalProperties": False},
                           Level.PRIMITIVE, "", "local", Cost()),
            ToolDescriptor("internal.request_operator", Modality.INTERNAL, "Ask the operator for something that "
                           "cannot be obtained otherwise: a capability, permission, credential, resource, account, "
                           "a verification step only a person may take, a piece of information or a decision. The "
                           "request is recorded for the operator; nothing is sent anywhere and nothing is granted by "
                           "asking. The operator's answer arrives later as an event.",
                           {"type": "object", "properties": {
                               "kind": {"type": "string", "enum": list(REQUEST_KINDS)},
                               "title": {"type": "string", "minLength": 3, "maxLength": 200},
                               "description": {"type": "string", "minLength": 3, "maxLength": 4000},
                               "justification": {"type": "string", "maxLength": 4000},
                               "blocking": {"type": "boolean"}},
                            "required": ["kind", "title", "description"], "additionalProperties": False},
                           Level.PRIMITIVE, "", "local", Cost()),
        ]

    def validate(self, action: ActionSpec) -> List[str]:
        from .env import validate_args
        for t in self.tools():
            if t.name == action.operation:
                errs = validate_args(t.arg_schema, action.arguments)
                if action.operation == "internal.inspect" and action.arguments.get("what") == "action" \
                        and not action.arguments.get("action_id"):
                    errs.append("action_id is required to inspect one action")
                if action.operation == "internal.inspect" and action.arguments.get("what") == "blob" \
                        and not re.fullmatch(r"(blob:)?[0-9a-f]{64}", str(action.arguments.get("blob", ""))):
                    errs.append("blob (a refs.data_blob reference) is required to read a stored payload")
                return errs
        return [f"unknown operation {action.operation!r}"]

    def execute(self, action: ActionSpec, ctx: ExecContext) -> ActionResult:
        now = ctx.clock.now()
        res = ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=now, ended=now)
        errs = self.validate(action)
        if errs:
            res.status, res.failure_class, res.failure_detail = (ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS,
                                                                 "; ".join(errs))
            return res
        a = action.arguments
        op = PrimitiveOp(op=action.operation, modality=self.modality, started=now, ended=now,
                         arguments={k: v for k, v in a.items() if k not in ("text", "description", "justification")})
        res.ops.append(op)
        dec = ctx.check(op, action)
        if dec.verdict != AuthVerdict.ALLOW:
            needs = dec.verdict == AuthVerdict.NEEDS_APPROVAL
            res.status = op.status = ResultStatus.NEEDS_APPROVAL if needs else ResultStatus.DENIED
            res.failure_class = op.failure_class = FailureClass.NONE if needs else FailureClass.MISSING_PERMISSION
            res.failure_detail = "; ".join(dec.reasons) or "denied by the authorization layer"
            return res
        if action.operation == "wait":
            obs = self._wait(action, ctx, op, res)
        elif action.operation == "internal.note":
            obs = self._note(action, ctx)
        elif action.operation == "internal.request_operator":
            obs = self._request(action, res, op)
        else:
            obs = self._inspect(action)
        obs.parent_action_id = action.action_id
        res.observations.append(obs)
        op.ended = res.ended = ctx.clock.now()
        op.obs_after = obs.obs_id
        res.cost = Cost(seconds=max(0.0, res.ended - res.started), primitive_ops=1)
        res.verified = None
        return res

    def close(self) -> None:
        return None

    # ---- operations -------------------------------------------------------------------------------------------------
    def _wait(self, action: ActionSpec, ctx: ExecContext, op: PrimitiveOp, res: ActionResult) -> Observation:
        seconds, clock = float(action.arguments["seconds"]), ctx.clock
        start = clock.now()
        wake_at = start + seconds
        how = "scheduled"
        if hasattr(clock, "advance") and self.advance_sim_clock:
            clock.advance(seconds)
            how = "simulated clock advanced"
        elif action.arguments.get("block") and seconds <= self.max_block_s:
            cancelled = ctx.cancel.wait(seconds) if ctx.cancel is not None else (threading.Event().wait(seconds))
            how = "cancelled while blocking" if cancelled else "slept"
            if cancelled:
                res.status = op.status = ResultStatus.CANCELLED
        else:
            with self._lock:
                self._wakes.append({"action_id": action.action_id, "wake_at": wake_at, "seconds": seconds,
                                    "reason": str(action.arguments.get("reason", ""))[:400]})
        op.note = how
        return self.perception.tool_result("wait", {"seconds": seconds, "wake_at": wake_at, "mode": how},
                                           source="internal")

    def _note(self, action: ActionSpec, ctx: ExecContext) -> Observation:
        rec = {"note_id": new_id("note"), "ts": ctx.clock.now(), "text": action.arguments["text"],
               "tags": list(action.arguments.get("tags", [])), "action_id": action.action_id}
        with self._lock, open(self._notes_path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        obs = self.perception.tool_result("internal.note", {"saved": rec["note_id"], "characters": len(rec["text"])},
                                          source="internal")
        obs.refs["note"] = rec["note_id"]
        return obs

    def _request(self, action: ActionSpec, res: ActionResult, op: PrimitiveOp) -> Observation:
        a = action.arguments
        try:
            rec = self.requests.request(a["kind"], a["title"], a["description"], a.get("justification", ""),
                                        blocking=bool(a.get("blocking", False)), action_id=action.action_id)
        except PermissionError as e:                 # too many requests are already waiting
            res.status = op.status = ResultStatus.FAILED
            res.failure_class = op.failure_class = FailureClass.RESOURCE_LIMIT
            res.failure_detail = str(e)
            return self.perception.tool_result("internal.request_operator", {"recorded": False, "reason": str(e)},
                                               source="internal")
        n_open = len(self.requests.open_requests())
        obs = self.perception.tool_result(
            "internal.request_operator",
            {"recorded": not rec.get("duplicate", False), "request_id": rec["request_id"],
             "already_open": bool(rec.get("duplicate", False)), "open_requests": n_open}, source="internal")
        obs.quantities.append(Quantity("operator.requests.open", float(n_open), kind="count"))
        obs.refs["request"] = rec["request_id"]
        return obs

    def notes(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            with open(self._notes_path) as f:
                rows = [json.loads(x) for x in f if x.strip()]
        except OSError:
            rows = []
        return rows[-limit:]

    def _inspect(self, action: ActionSpec) -> Observation:
        what, limit = action.arguments["what"], int(action.arguments.get("limit", 20))
        env = self._env
        if what == "notes":
            data: Any = self.notes(limit)
        elif env is None:
            data = {"error": "the internal executor is not attached to an environment"}
        elif what == "actions":
            data = [{"action_id": r.get("action_id"), "operation": r.get("operation"), "modality": r.get("modality"),
                     "ts": r.get("ts")} for r in env.records("env.action_started", limit)]
        elif what == "results":
            data = [{k: r.get(k) for k in ("action_id", "status", "failure_class", "failure_detail", "verified", "cost",
                                           "primitive_ops", "ts")} for r in env.records("env.action_result", limit)]
        elif what == "action":
            data = [{k: v for k, v in r.items() if k not in ("src", "wall", "seq")}
                    for r in env.records("", 200, action.arguments["action_id"])
                    if r.get("type") != "env.observation"]
        elif what == "events":
            evs, cur = env.poll_events(str(max(0, len(env._stream_offsets) - limit)))
            data = {"cursor": cur, "events": [{"obs_id": o.obs_id, "ts": o.ts, "text": o.text[:300],
                                               "quantities": [q.to_dict() for q in o.quantities]} for o in evs]}
        elif what == "pending":
            data = [q.to_dict() for q in env.get_pending_outcomes()[:limit]]
        elif what == "resources":
            data = env.get_resource_state()
        elif what == "tools":
            data = [{"name": t.name, "modality": t.modality, "available": t.available} for t in env.discover_tools()]
        elif what == "blob":
            try:
                data = env.read_blob(action.arguments["blob"], int(action.arguments.get("offset", 0)),
                                     int(action.arguments.get("max_chars", 3000)))
            except (OSError, ValueError) as e:
                data = {"error": f"no such stored payload: {type(e).__name__}"}
            obs = self.perception.document(data.get("text", ""), uri=str(action.arguments["blob"]),
                                           media_type="text/plain", source="internal")
            obs.data.update(what=what, **{k: v for k, v in data.items() if k != "text"})
            return obs
        else:
            data = [{"type": r.get("type"), "ts": r.get("ts"), "action_id": r.get("action_id")}
                    for r in env.records("", limit)]
        obs = self.perception.tool_result("internal.inspect", data, source="internal")
        obs.data.update(what=what, records=data)
        return obs
