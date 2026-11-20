"""A strict JSON-Schema subset for tool arguments.

Supported keywords: type (string, integer, number, boolean, object, array, null; or a list of these), properties,
required, additionalProperties (default False: unknown keys are errors), enum, minimum, maximum, minLength,
maxLength, pattern, items, minItems, maxItems, default, description. Tool schemas are published to agents
verbatim (MCP `inputSchema`, OpenAI-style function specs, Gymnasium text actions).
"""
from __future__ import annotations

import re
from typing import Any

_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}

MAX_STRING = 200_000


def obj(properties: dict[str, dict] | None = None, required: list[str] | None = None, **kw) -> dict:
    s = {"type": "object", "properties": properties or {}, "additionalProperties": False}
    if required:
        s["required"] = list(required)
    s.update(kw)
    return s


def s_str(desc: str = "", **kw) -> dict:
    return {"type": "string", "description": desc, "maxLength": kw.pop("maxLength", 4000), **kw}


def s_int(desc: str = "", **kw) -> dict:
    return {"type": "integer", "description": desc, **kw}


def s_num(desc: str = "", **kw) -> dict:
    return {"type": "number", "description": desc, **kw}


def s_bool(desc: str = "", **kw) -> dict:
    return {"type": "boolean", "description": desc, **kw}


def s_enum(values: list, desc: str = "") -> dict:
    return {"type": "string", "enum": list(values), "description": desc}


def s_arr(items: dict, desc: str = "", **kw) -> dict:
    return {"type": "array", "items": items, "description": desc, "maxItems": kw.pop("maxItems", 1000), **kw}


def s_any_obj(desc: str = "") -> dict:
    return {"type": "object", "description": desc, "additionalProperties": True}


def validate(schema: dict, value: Any, path: str = "args") -> list[str]:
    errs: list[str] = []
    _v(schema, value, path, errs)
    return errs


def _v(s: dict, v: Any, path: str, errs: list[str]) -> None:
    t = s.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(_TYPES[x](v) for x in ts):
            errs.append(f"{path}: expected {t}, got {type(v).__name__}")
            return
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{path}: {v!r} not one of {s['enum']}")
    if isinstance(v, str):
        if len(v) > s.get("maxLength", MAX_STRING):
            errs.append(f"{path}: longer than {s.get('maxLength', MAX_STRING)}")
        if len(v) < s.get("minLength", 0):
            errs.append(f"{path}: shorter than {s['minLength']}")
        if "pattern" in s and not re.fullmatch(s["pattern"], v):
            errs.append(f"{path}: does not match {s['pattern']}")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in s and v < s["minimum"]:
            errs.append(f"{path}: below minimum {s['minimum']}")
        if "maximum" in s and v > s["maximum"]:
            errs.append(f"{path}: above maximum {s['maximum']}")
        if isinstance(v, float) and v != v:
            errs.append(f"{path}: NaN")
    if isinstance(v, list):
        if len(v) > s.get("maxItems", 10_000):
            errs.append(f"{path}: more than {s.get('maxItems', 10_000)} items")
        if len(v) < s.get("minItems", 0):
            errs.append(f"{path}: fewer than {s['minItems']} items")
        if "items" in s:
            for i, x in enumerate(v):
                _v(s["items"], x, f"{path}[{i}]", errs)
    if isinstance(v, dict) and (s.get("type") == "object" or "properties" in s):
        props = s.get("properties", {})
        for k in s.get("required", []):
            if k not in v:
                errs.append(f"{path}.{k}: required")
        addl = s.get("additionalProperties", False)
        for k, x in v.items():
            if k in props:
                _v(props[k], x, f"{path}.{k}", errs)
            elif addl is False:
                errs.append(f"{path}.{k}: unexpected property")
            elif isinstance(addl, dict):
                _v(addl, x, f"{path}.{k}", errs)


def with_defaults(schema: dict, args: dict) -> dict:
    out = dict(args)
    for k, p in schema.get("properties", {}).items():
        if k not in out and "default" in p:
            out[k] = p["default"]
    return out


def check_schema(schema: dict) -> list[str]:
    """Admission check for schemas declared by agent-deployed services."""
    errs: list[str] = []
    allowed = {"type", "properties", "required", "additionalProperties", "enum", "minimum", "maximum", "minLength",
               "maxLength", "pattern", "items", "minItems", "maxItems", "default", "description"}

    def walk(s: Any, path: str, depth: int) -> None:
        if depth > 8:
            errs.append(f"{path}: nesting too deep")
            return
        if not isinstance(s, dict):
            errs.append(f"{path}: schema must be an object")
            return
        for k in s:
            if k not in allowed:
                errs.append(f"{path}: unsupported keyword {k!r}")
        t = s.get("type")
        if t is not None:
            for x in (t if isinstance(t, list) else [t]):
                if x not in _TYPES:
                    errs.append(f"{path}: unknown type {x!r}")
        if "pattern" in s:
            try:
                re.compile(s["pattern"])
            except re.error as e:
                errs.append(f"{path}: bad pattern ({e})")
        for k, p in (s.get("properties") or {}).items():
            walk(p, f"{path}.{k}", depth + 1)
        if "items" in s:
            walk(s["items"], f"{path}[]", depth + 1)
        if isinstance(s.get("additionalProperties"), dict):
            walk(s["additionalProperties"], f"{path}.*", depth + 1)

    walk(schema, "schema", 0)
    if isinstance(schema, dict) and schema.get("type") != "object":
        errs.append("schema: top level must be type object")
    return errs
