"""Self-generated experiments in the Tier 1 adapters (MASTER_SPEC 11.4; Clarification 2: "support self-generated
experiments, hypotheses and learning objectives without automatically rewarding their completion"; OO.11).

Kernel worlds (Tiers 0, 2 and 3) have the kernel module `exp` (`crucible.experience.experiments`). The Tier 1
adapters (`MechExperienceEnv`, `RapidExperienceEnv`) are not kernel worlds, so they get the same three tools here,
with the same prediction semantics (`experiments.holds`) over the one quantity source a Tier 1 participant has: the
numbers of its own observation.

    exp.register {hypothesis, measure: {"number": "<a key of your observation's numbers>"}, prediction,
                  deadline_in_s, plan?}   -> {"experiment", "baseline", "resolves_at"}
    exp.result {experiment}              -> the evidence (baseline, measured value, change, held, times)
    exp.mine {status?}                   -> your experiments

At the first observation time at or after the deadline the adapter reads the same number from the agent's own
observation and records `exp.resolved`, visible only to the agent, caused by (linked to) the `exp.register` action.
Nothing is paid, scored or rewarded: `reward()` is untouched in every mode, and an experiment that held earns
exactly what one that failed earns. The measurement reads what the agent itself sees (never evaluator data), and a
registration is an action like any other (it costs no world time).
"""
from __future__ import annotations

import math
from typing import Any, Callable

from .experiments import OPS, PREDICTION, holds

MAX_OPEN = 32
TOOLS = [
    {"name": "exp.register",
     "description": "Register a self-generated experiment: a hypothesis about one number of your own observation, "
                    "a prediction and a deadline. The result is recorded at the deadline (exp.resolved, private). "
                    "Nothing is rewarded.",
     "inputSchema": {"type": "object", "properties": {
         "hypothesis": {"type": "string", "maxLength": 2000, "description": "what you expect and why, in words"},
         "plan": {"type": "string", "maxLength": 2000},
         "measure": {"type": "object", "properties": {
             "number": {"type": "string", "maxLength": 200,
                        "description": "a key of your observation's numbers, e.g. rig.seats.r1.revenue_cents"}},
             "required": ["number"], "additionalProperties": False},
         "prediction": PREDICTION,
         "deadline_in_s": {"type": "integer", "minimum": 1, "maximum": 400 * 86400}},
         "required": ["hypothesis", "measure", "prediction", "deadline_in_s"], "additionalProperties": False},
     "annotations": {"host": "exp", "readOnlyHint": False, "costSeconds": 0}},
    {"name": "exp.result", "description": "The evidence of one of your experiments.",
     "inputSchema": {"type": "object", "properties": {"experiment": {"type": "string", "maxLength": 40}},
                     "required": ["experiment"], "additionalProperties": False},
     "annotations": {"host": "exp", "readOnlyHint": True, "costSeconds": 0}},
    {"name": "exp.mine", "description": "Your experiments (all, or by status: open, resolved).",
     "inputSchema": {"type": "object", "properties": {"status": {"type": "string", "enum": ["open", "resolved"]}},
                     "additionalProperties": False},
     "annotations": {"host": "exp", "readOnlyHint": True, "costSeconds": 0}},
]
NAMES = frozenset(t["name"] for t in TOOLS)


class Refused(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(float(v)) else None


class AdapterExperiments:
    """The experiment table of one Tier 1 adapter. `numbers(agent)` returns the agent's current observation numbers
    (without consuming its event cursor); `emit(typ, payload, agent, cause)` records an event visible to the agent."""

    def __init__(self, numbers: Callable[[str], dict], emit: Callable[[str, dict, str, dict], int]):
        self._numbers, self._emit = numbers, emit
        self.rows: dict[str, dict] = {}
        self._n = 0

    def tools(self) -> list[dict]:
        import copy
        return copy.deepcopy(TOOLS)

    def execute(self, agent: str, tool: str, args: dict, seq: int, now: int) -> dict:
        if tool == "exp.register":
            return self._register(agent, args, seq, now)
        if tool == "exp.result":
            r = self.rows.get(str(args.get("experiment")))
            if r is None or r["owner"] != agent:
                raise Refused("not_found", f"no experiment {args.get('experiment')!r} of yours")
            return dict(r)
        if tool == "exp.mine":
            st = args.get("status")
            return [dict(r) for k, r in sorted(self.rows.items()) if r["owner"] == agent
                    and (st is None or r["status"] == st)]
        raise Refused("unknown_tool", tool)

    def _register(self, agent: str, args: dict, seq: int, now: int) -> dict:
        key = (args.get("measure") or {}).get("number")
        pred = dict(args.get("prediction") or {})
        if pred.get("op") not in OPS:
            raise Refused("invalid_action", f"prediction op must be one of {OPS}")
        if pred["op"] in ("increase", "decrease", "change"):
            pred.setdefault("min_change", 0.0)
        elif _num(pred.get("value")) is None:
            raise Refused("invalid_action", f"prediction {pred['op']!r} needs a numeric value")
        if sum(1 for r in self.rows.values() if r["owner"] == agent and r["status"] == "open") >= MAX_OPEN:
            raise Refused("rate_limited", f"at most {MAX_OPEN} open experiments")
        base = _num(self._numbers(agent).get(key))
        if base is None:
            raise Refused("invalid_action", f"{key!r} is not a number of your observation (see observe()['numbers'])")
        self._n += 1
        eid = f"x{self._n:06d}"
        dl = int(args["deadline_in_s"])
        self.rows[eid] = {"experiment": eid, "owner": agent, "hypothesis": str(args.get("hypothesis", ""))[:2000],
                          "plan": str(args.get("plan", ""))[:2000], "measure": {"number": key}, "prediction": pred,
                          "baseline": base, "registered_at": int(now), "resolves_at": int(now) + dl,
                          "status": "open", "action_seq": int(seq), "value": None, "change": None, "held": None,
                          "resolved_at": None}
        return {"experiment": eid, "baseline": base, "resolves_at": int(now) + dl}

    def poll(self, now: int) -> int:
        """Resolve every open experiment whose deadline has passed (called by the adapter as time advances)."""
        n = 0
        for eid, r in sorted(self.rows.items()):
            if r["status"] != "open" or r["resolves_at"] > now:
                continue
            v = _num(self._numbers(r["owner"]).get(r["measure"]["number"]))
            r.update(status="resolved", resolved_at=int(now), value=v,
                     change=None if v is None else v - r["baseline"],
                     held=None if v is None else bool(holds(r["prediction"], v, r["baseline"])))
            cause = {"agent": r["owner"], "action_seq": r["action_seq"], "tool": "exp.register",
                     "t": r["registered_at"], "via": "experiment"}
            self._emit("exp.resolved", {k: r[k] for k in ("experiment", "hypothesis", "measure", "prediction",
                                                          "baseline", "value", "change", "held", "registered_at",
                                                          "resolved_at")}, r["owner"], cause)
            n += 1
        return n

    def state(self) -> dict:
        return {"n": self._n, "rows": {k: dict(v) for k, v in self.rows.items()}}

    def load(self, st: dict) -> None:
        self._n = int(st.get("n", 0))
        self.rows = {k: dict(v) for k, v in (st.get("rows") or {}).items()}
