"""The Observation record shared by every tier (INTERFACE_CONTRACTS section 11) and its validation.

One `Observation` is what one agent perceives at one moment, in any world: a Tier 1 rapid environment, a Tier 2
domain or a Tier 3 civilisation. Every field is JSON-native, so an observation can be stored, hashed, sent over a
pipe or rendered as text without loss. Exact quantities are numbers, never text: money is integer cents, time is
integer seconds, readings keep their full float value.

    t            world time in simulated seconds (every tier and adapter; `TIME_UNIT`, stated in ``resources``
                 as ``time_unit`` and in each adapter's ``describe()``)
    agent        the observing agent (a principal id)
    numbers      flat ``{"path.to.quantity": number}`` of exact scalar quantities the agent may see
    arrays       ``{"name": [numbers...]}`` vectors (gauge vectors, semantic representations ``semantic.<ref>``)
    text         natural-language items (``"<source>: <text>"``)
    artifacts    references to content the agent can read: ``{"ref", "kind", "owner", ...}``
    tools        MCP-shaped tool specs (name, description, inputSchema, annotations incl. ``category``)
    events       visible events since the previous observation: ``{"seq", "t", "type", "actor", "payload",
                 "cause"}``; ``cause`` links a consequence to the agent's own earlier action (or is None)
    entities     ``{"id", "kind", "relations": [{"rel", "to"}], "attrs": {...}}``
    resources    the agent's own resource usage (actions, time used, money, quotas)
    analytics    public aggregate statistics anyone in the world could compute
    last_result  the result of the agent's previous action (or None)
    missing      what the agent does not (yet) know: pending results, hidden mechanisms, truncation notices
    cursor       event cursor up to which `events` were delivered (use `observe_since` to read further)
    objective    the agent's public objective, if it has one

Nothing in an observation is evaluator data: no reward, no goal progress, no hidden parameter. That is checked by
the reward-leakage tests (`tests/v5/test_gate_R.py`), not assumed.
"""
from __future__ import annotations

import math
from typing import Any, TypedDict

OBSERVATION_FORMAT = "crucible-obs/1"
TIME_UNIT = "s"          # `t`, event times, `cost_s`, `delay_s`: simulated seconds in every tier and adapter


class Observation(TypedDict):
    t: int
    agent: str
    numbers: dict[str, float]
    arrays: dict[str, list]
    text: list[str]
    artifacts: list[dict]
    tools: list[dict]
    events: list[dict]
    entities: list[dict]
    resources: dict
    analytics: dict
    last_result: dict | None
    missing: list[str]
    cursor: int
    objective: str | None


FIELDS: dict[str, tuple] = {
    "t": (int,), "agent": (str,), "numbers": (dict,), "arrays": (dict,), "text": (list,), "artifacts": (list,),
    "tools": (list,), "events": (list,), "entities": (list,), "resources": (dict,), "analytics": (dict,),
    "last_result": (dict, type(None)), "missing": (list,), "cursor": (int,), "objective": (str, type(None)),
}

# The thirteen observation kinds of MASTER_SPEC 11.1 and how each is recognised in an observation.
OBSERVATION_KINDS = (
    "structured_numbers", "text", "artifact_references", "semantic_representations", "tool_schemas", "events",
    "entity_relationships", "timestamps", "resource_usage", "public_analytics", "action_results",
    "missing_or_uncertain", "delayed_consequences",
)

SEMANTIC_PREFIX = "semantic."


def empty(agent: str, t: int = 0, cursor: int = 0, objective: str | None = None) -> Observation:
    return Observation(t=int(t), agent=str(agent), numbers={}, arrays={}, text=[], artifacts=[], tools=[], events=[],
                       entities=[], resources={}, analytics={}, last_result=None, missing=[], cursor=int(cursor),
                       objective=objective)


# ---------------------------------------------------------------------------------------------- validation
def _json_native(x: Any, path: str, errs: list[str], depth: int = 0) -> None:
    if depth > 64:
        errs.append(f"{path}: nested too deeply")
        return
    if x is None or isinstance(x, (bool, str)):
        return
    if isinstance(x, int):
        return
    if isinstance(x, float):
        if not math.isfinite(x):
            errs.append(f"{path}: non-finite number {x!r}")
        return
    if isinstance(x, dict):
        for k, v in x.items():
            if not isinstance(k, str):
                errs.append(f"{path}: non-string key {k!r}")
                continue
            _json_native(v, f"{path}.{k}", errs, depth + 1)
        return
    if isinstance(x, list):
        for i, v in enumerate(x):
            _json_native(v, f"{path}[{i}]", errs, depth + 1)
        return
    errs.append(f"{path}: not JSON-native ({type(x).__name__})")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _numeric_list(x: Any) -> bool:
    if isinstance(x, list):
        return all(_numeric_list(v) for v in x)
    return _is_number(x)


def validate_tool(t: Any, where: str = "tool") -> list[str]:
    errs = []
    if not isinstance(t, dict):
        return [f"{where}: not an object"]
    if not isinstance(t.get("name"), str) or "." not in t["name"]:
        errs.append(f"{where}: name must be '<host>.<endpoint>'")
    if not isinstance(t.get("description", ""), str):
        errs.append(f"{where}: description must be a string")
    sch = t.get("inputSchema")
    if not isinstance(sch, dict) or sch.get("type") not in ("object", None):
        errs.append(f"{where}: inputSchema must be a JSON object schema")
    ann = t.get("annotations", {})
    if not isinstance(ann, dict):
        errs.append(f"{where}: annotations must be an object")
    return errs


def validate_observation(obs: Any) -> list[str]:
    """Schema problems of an observation (empty list = valid). Checks the exact field set, field types,
    JSON-nativeness, that `numbers` holds finite numbers, `arrays` numeric lists, events/tools/artifacts/entities
    their required keys."""
    if not isinstance(obs, dict):
        return ["observation is not a dict"]
    errs: list[str] = []
    keys = set(obs)
    for k in sorted(set(FIELDS) - keys):
        errs.append(f"missing field {k!r}")
    for k in sorted(keys - set(FIELDS)):
        errs.append(f"unknown field {k!r}")
    for k, types in FIELDS.items():
        if k in obs and not isinstance(obs[k], types) or (k in obs and isinstance(obs[k], bool) and bool not in types):
            errs.append(f"field {k!r} has type {type(obs[k]).__name__}")
    if errs:
        return errs
    _json_native(obs, "obs", errs)
    for k, v in obs["numbers"].items():
        if not _is_number(v):
            errs.append(f"numbers[{k!r}] is not a finite number: {v!r}")
    for k, v in obs["arrays"].items():
        if not _numeric_list(v):
            errs.append(f"arrays[{k!r}] is not a numeric list")
    for i, s in enumerate(obs["text"]):
        if not isinstance(s, str):
            errs.append(f"text[{i}] is not a string")
    for i, s in enumerate(obs["missing"]):
        if not isinstance(s, str):
            errs.append(f"missing[{i}] is not a string")
    for i, e in enumerate(obs["events"]):
        if not isinstance(e, dict) or not {"seq", "t", "type", "actor", "payload", "cause"} <= set(e):
            errs.append(f"events[{i}] lacks seq/t/type/actor/payload/cause")
            continue
        if not isinstance(e["seq"], int) or not isinstance(e["t"], int) or not isinstance(e["type"], str):
            errs.append(f"events[{i}]: seq and t must be integers and type a string")
        c = e["cause"]
        if c is not None and (not isinstance(c, dict) or not {"action_seq", "t", "delay_s", "via"} <= set(c)):
            errs.append(f"events[{i}].cause lacks action_seq/t/delay_s/via")
    for i, t in enumerate(obs["tools"]):
        errs += validate_tool(t, f"tools[{i}]")
    for i, a in enumerate(obs["artifacts"]):
        if not isinstance(a, dict) or not isinstance(a.get("ref"), str) or not isinstance(a.get("kind"), str):
            errs.append(f"artifacts[{i}] needs string 'ref' and 'kind'")
    for i, en in enumerate(obs["entities"]):
        if not isinstance(en, dict) or not isinstance(en.get("id"), str) or not isinstance(en.get("kind"), str) \
                or not isinstance(en.get("relations", []), list):
            errs.append(f"entities[{i}] needs string 'id', 'kind' and a 'relations' list")
    lr = obs["last_result"]
    if lr is not None:
        errs += validate_result(lr, "last_result")
    if obs["cursor"] < 0:
        errs.append("cursor is negative")
    return errs


# ---------------------------------------------------------------------------------------------- actions and results
def validate_action(a: Any) -> list[str]:
    """The one action shape of every tier: ``{"tool": "<host>.<endpoint>", "args": {...}}`` (optionally
    ``"idempotency_key"``)."""
    if not isinstance(a, dict):
        return ["action is not a dict"]
    errs = []
    if not isinstance(a.get("tool"), str) or not a["tool"]:
        errs.append("action.tool must be a non-empty string")
    if not isinstance(a.get("args", {}), dict):
        errs.append("action.args must be an object")
    extra = set(a) - {"tool", "args", "idempotency_key"}
    if extra:
        errs.append(f"unknown action keys {sorted(extra)}")
    sub: list[str] = []
    _json_native(a.get("args", {}), "action.args", sub)
    return errs + sub


def validate_result(r: Any, where: str = "result") -> list[str]:
    """`execute` returns ``{"ok", "t", "tool", "cost_s", "result" | "error"}`` and, when the action was recorded,
    ``"action_seq"`` (the seq its consequences' `cause` names)."""
    if not isinstance(r, dict):
        return [f"{where} is not a dict"]
    errs = []
    if not isinstance(r.get("ok"), bool):
        errs.append(f"{where}.ok must be a bool")
    if not isinstance(r.get("t"), int) or not isinstance(r.get("cost_s"), int):
        errs.append(f"{where}.t and {where}.cost_s must be integers")
    if r.get("ok") is True and "result" not in r:
        errs.append(f"{where} ok without 'result'")
    if r.get("ok") is False and not isinstance(r.get("error"), dict):
        errs.append(f"{where} failed without an 'error' object")
    sub: list[str] = []
    _json_native(r, where, sub)
    return errs + sub


# ---------------------------------------------------------------------------------------------- kinds
def kinds_present(obs: Observation) -> set[str]:
    """Which of the 13 observation kinds this observation actually carries (non-empty content, not just a field)."""
    k: set[str] = set()
    if obs["numbers"] or any(not n.startswith(SEMANTIC_PREFIX) for n in obs["arrays"]):
        k.add("structured_numbers")
    if obs["text"]:
        k.add("text")
    if obs["artifacts"]:
        k.add("artifact_references")
    if any(n.startswith(SEMANTIC_PREFIX) and obs["arrays"][n] for n in obs["arrays"]):
        k.add("semantic_representations")
    if obs["tools"]:
        k.add("tool_schemas")
    if obs["events"]:
        k.add("events")
    if any(e.get("relations") for e in obs["entities"]):
        k.add("entity_relationships")
    if isinstance(obs["t"], int) and (obs["events"] and all(isinstance(e.get("t"), int) for e in obs["events"])):
        k.add("timestamps")
    if obs["resources"]:
        k.add("resource_usage")
    if obs["analytics"]:
        k.add("public_analytics")
    if obs["last_result"] is not None:
        k.add("action_results")
    if obs["missing"]:
        k.add("missing_or_uncertain")
    if any(isinstance(e.get("cause"), dict) and e["cause"].get("delay_s", 0) > 0 for e in obs["events"]):
        k.add("delayed_consequences")
    return k


# ---------------------------------------------------------------------------------------------- helpers for adapters
def json_copy(x: Any) -> Any:
    """A deep copy of a JSON-native value (event payloads are shared with the world log and must never be mutated
    through an observation)."""
    if isinstance(x, dict):
        return {k: json_copy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_copy(v) for v in x]
    if isinstance(x, tuple):
        return [json_copy(v) for v in x]
    return x


def exact(v: Any) -> int | float | None:
    """A JSON number for a scalar (ints stay ints, integral floats from int sources stay floats), None if it is not
    a finite number."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    try:                                   # numpy scalars
        import numpy as np
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            f = float(v)
            return f if math.isfinite(f) else None
    except ImportError:                    # pragma: no cover
        pass
    return None
