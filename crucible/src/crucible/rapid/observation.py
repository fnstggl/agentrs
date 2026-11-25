"""Tier 1 observations in the shape of the universal Observation dict (INTERFACE_CONTRACTS section 11).

Every field of section 11 is present in every observation. Exact numbers stay numbers (`numbers`, `arrays`,
`resources`, `analytics`); a reading that is missing this tick is not invented: it is left out of `numbers` and its
name is listed in `missing`. Nothing here carries a reward, shaped progress, objective satisfaction or evaluator
data: an observation holds what the participant could see in the world (its readings, its own money and budget,
the results of its own actions, messages addressed to it, public events).
"""
from __future__ import annotations

import math
from typing import Any

FIELDS = ("t", "agent", "numbers", "arrays", "text", "artifacts", "tools", "events", "entities", "resources",
          "analytics", "last_result", "missing", "cursor", "objective")

# words that would reveal evaluator-side quantities if they appeared as keys of an observation
FORBIDDEN_KEYS = ("reward", "progress", "success", "oracle", "regret", "truth", "hidden", "score_normalized",
                  "objective_value", "reference")


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def make(*, t: int, agent: str, numbers: dict | None = None, arrays: dict | None = None, text: list | None = None,
         artifacts: list | None = None, tools: list | None = None, events: list | None = None,
         entities: list | None = None, resources: dict | None = None, analytics: dict | None = None,
         last_result: dict | None = None, missing: list | None = None, cursor: int = 0,
         objective: str | None = None) -> dict:
    """Build an Observation. Non-finite numbers are moved to `missing` (they are never reported as values)."""
    nums, miss = {}, list(missing or [])
    for k, v in (numbers or {}).items():
        x = _num(v)
        if x is None:
            if k not in miss:
                miss.append(k)
        else:
            nums[str(k)] = x
    arrs = {}
    for k, v in (arrays or {}).items():
        arrs[str(k)] = [(_num(x) if not isinstance(x, (list, str, dict)) else x) for x in v]
    return {"t": int(t), "agent": str(agent), "numbers": nums, "arrays": arrs, "text": [str(s) for s in (text or [])],
            "artifacts": list(artifacts or []), "tools": list(tools or []), "events": list(events or []),
            "entities": list(entities or []), "resources": {str(k): v for k, v in (resources or {}).items()},
            "analytics": dict(analytics or {}), "last_result": last_result, "missing": sorted(set(miss)),
            "cursor": int(cursor), "objective": objective}


def check(obs: Any) -> list[str]:
    """Shape errors of an observation (empty when it is a valid section-11 Observation)."""
    import json
    errs = []
    if not isinstance(obs, dict):
        return ["an observation is a dict"]
    for f in FIELDS:
        if f not in obs:
            errs.append(f"missing field {f}")
    extra = set(obs) - set(FIELDS)
    if extra:
        errs.append(f"unexpected fields {sorted(extra)}")
    if errs:
        return errs
    if not isinstance(obs["t"], int) or not isinstance(obs["cursor"], int):
        errs.append("t and cursor are integers")
    for k, v in obs["numbers"].items():
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            errs.append(f"numbers.{k} is not a finite number")
    for f in ("text", "artifacts", "tools", "events", "entities", "missing"):
        if not isinstance(obs[f], list):
            errs.append(f"{f} is a list")
    for f in ("numbers", "arrays", "resources", "analytics"):
        if not isinstance(obs[f], dict):
            errs.append(f"{f} is a dict")
    try:
        json.dumps(obs, allow_nan=False)
    except (TypeError, ValueError) as e:
        errs.append(f"not JSON-native: {e}")
    for path, key in _keys(obs):
        low = key.lower()
        if any(w in low for w in FORBIDDEN_KEYS):
            errs.append(f"{path}: key {key!r} names an evaluator-side quantity")
    return errs


def _keys(x: Any, path: str = "obs"):
    if isinstance(x, dict):
        for k, v in x.items():
            if path in ("obs.tools",) or ".tools[" in path or ".inputSchema" in path:
                continue                      # tool schemas describe actions; their keys are not observations
            yield path, str(k)
            yield from _keys(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _keys(v, f"{path}[{i}]")
