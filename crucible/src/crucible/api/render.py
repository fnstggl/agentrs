"""Text rendering of observations and tool lists for language-model agents, and tolerant action parsing.

Everything here is a pure function of its input: the same observation always renders to the same bytes (sorted
JSON keys, no wall-clock, no hash-seed dependence), so rendered transcripts are reproducible and cacheable.

* `render_observation(obs)`  - compact multi-line text of one `Observation` (or its dict form).
* `render_tools(tools)`      - one line per tool: ``name(arg: type, opt?: type) [Ns] - description``.
* `render_event(ev)`, `render_result(res)`, `fmt_time(t)`.
* `parse_action(text)`       - extract a ``{"tool": ..., "args": {...}}`` call from model output. Tolerates code
  fences, ``<tool_call>`` tags, ``<think>`` blocks, surrounding prose, OpenAI/MCP shapes (``name``/``arguments``,
  ``function``), arguments given as a JSON string, and ``tool.name {json}``. Never raises: on failure it returns
  ``{"error": {"code": "parse_error", "message": ...}}``.
"""
from __future__ import annotations

import ast
import json
import re
import warnings
from typing import Any, Mapping

__all__ = ["fmt_time", "compact_json", "render_event", "render_tool", "render_tools", "render_result",
           "render_observation", "action_format_help", "parse_action", "normalize_action", "sanitize_text"]


# ---------------------------------------------------------------------------------------------- formatting
def fmt_time(t: int) -> str:
    """Simulated seconds -> ``d3 14:05:09``."""
    t = int(t)
    d, r = divmod(t, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return f"d{d} {h:02d}:{m:02d}:{s:02d}"


def compact_json(x: Any, limit: int | None = 300) -> str:
    """Deterministic one-line JSON (sorted keys, ASCII), truncated with ``...`` to `limit` characters."""
    s = json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    if limit is not None and len(s) > limit:
        s = s[: max(0, limit - 3)] + "..."
    return s


def _get(o: Any, key: str, default: Any = None) -> Any:
    if isinstance(o, Mapping):
        return o.get(key, default)
    return getattr(o, key, default)


def render_event(ev: Mapping, payload_chars: int = 240) -> str:
    who = ev.get("actor", "")
    return f"[{fmt_time(ev.get('t', 0))}] #{ev.get('seq')} {ev.get('type')} by {who}: " \
           f"{compact_json(ev.get('payload', {}), payload_chars)}"


def _type_str(s: Mapping) -> str:
    if "enum" in s:
        return "|".join(str(v) for v in s["enum"])
    t = s.get("type", "any")
    if isinstance(t, list):
        t = "|".join(t)
    if t == "array" and isinstance(s.get("items"), Mapping):
        return f"{_type_str(s['items'])}[]"
    return {"string": "str", "integer": "int", "number": "num", "boolean": "bool", "object": "obj"}.get(t, t)


def render_tool(tool: Mapping, description_chars: int = 160) -> str:
    schema = tool.get("inputSchema") or {}
    props = schema.get("properties") or {}
    req = set(schema.get("required") or [])
    params = []
    for name in sorted(props, key=lambda n: (n not in req, n)):
        params.append(f"{name}{'' if name in req else '?'}: {_type_str(props[name])}")
    line = f"{tool.get('name')}({', '.join(params)})"
    cost = (tool.get("annotations") or {}).get("costSeconds")
    if cost:
        line += f" [{cost}s]"
    desc = (tool.get("description") or "").strip().splitlines()
    if desc and description_chars:
        d = desc[0].strip()
        if len(d) > description_chars:
            d = d[: description_chars - 3] + "..."
        line += f" - {d}"
    return line


def render_tools(tools: list[Mapping], description_chars: int = 160) -> str:
    return "\n".join(render_tool(t, description_chars) for t in sorted(tools, key=lambda t: t.get("name", "")))


def render_result(res: Mapping | None, chars: int = 600) -> str:
    if not res:
        return "none"
    tool = res.get("tool") or "?"
    if res.get("ok"):
        out = f"{tool} -> ok {compact_json(res.get('result'), chars)}"
    else:
        err = res.get("error") or {}
        out = f"{tool} -> FAILED {err.get('code', 'error')}: {err.get('message', '')}"[: chars + 40]
    if res.get("cost_s"):
        out += f" (took {res['cost_s']}s)"
    if res.get("replayed"):
        out += " (idempotent replay)"
    return out


def render_observation(obs: Any, *, max_events: int | None = None, payload_chars: int = 240,
                       show_tools: bool = True, description_chars: int = 160) -> str:
    """Render an `Observation` (or `Observation.to_dict()`) as compact text. Deterministic."""
    t = _get(obs, "time", 0)
    lines = [f"time {fmt_time(t)} (t={t}) | you are {_get(obs, 'principal')}"]
    lines.append("last action: " + render_result(_get(obs, "last_result")))
    summary = _get(obs, "summary") or {}
    if summary:
        lines.append("status:")
        for k in sorted(summary):
            lines.append(f"  {k}: {compact_json(summary[k], 400)}")
    events = list(_get(obs, "events") or [])
    dropped = int(_get(obs, "dropped", 0) or 0)
    if max_events is not None and len(events) > max_events:
        dropped += len(events) - max_events
        events = events[-max_events:] if max_events > 0 else []
    head = f"new events: {len(events)}"
    if dropped:
        head += f" (+{dropped} older not shown)"
    lines.append(head)
    lines.extend("  " + render_event(e, payload_chars) for e in events)
    if _get(obs, "tools_changed"):
        added = _get(obs, "new_tools") or []
        removed = _get(obs, "removed_tools") or []
        tools = _get(obs, "tools")
        n = len(tools) if tools is not None else "?"
        msg = f"tools changed: {n} available"
        if added and _get(obs, "observed", 1) > 1:
            msg += f"; new: {', '.join(added)}"
        if removed:
            msg += f"; removed: {', '.join(removed)}"
        lines.append(msg)
    tools = _get(obs, "tools")
    if show_tools and tools:
        lines.append("tools:")
        lines.extend("  " + ln for ln in render_tools(tools, description_chars).splitlines())
    return "\n".join(lines)


def action_format_help() -> str:
    """Instruction snippet for prompts: how to express an action."""
    return ('Act by replying with exactly one JSON object: {"tool": "<tool name>", "args": {...}}. '
            'Example: {"tool": "bank.balance", "args": {"account": "acct_000001"}}')


def sanitize_text(text: str, charset: frozenset | set | str, max_length: int | None = None) -> str:
    """Replace characters outside `charset` by ``?`` and truncate (Gymnasium ``Text`` spaces check both)."""
    cs = charset if isinstance(charset, (set, frozenset)) else frozenset(charset)
    out = "".join(c if c in cs else "?" for c in text)
    if max_length is not None and len(out) > max_length:
        out = out[: max(0, max_length - 3)] + "..."
    return out


# ---------------------------------------------------------------------------------------------- parsing
def _err(msg: str, raw: Any = None) -> dict:
    d = {"error": {"code": "parse_error", "message": msg}}
    if isinstance(raw, str):
        d["raw"] = raw[:200]
    return d


def normalize_action(obj: Any) -> dict:
    """Coerce the common tool-call shapes into ``{"tool": str, "args": dict[, "idempotency_key": str]}``."""
    for _ in range(4):  # unwrap {"tool_call": {...}}, {"function": {...}}, [call, ...]
        if isinstance(obj, list):
            if not obj:
                return _err("empty list")
            obj = obj[0]
            continue
        if not isinstance(obj, Mapping):
            return _err(f"expected a JSON object, got {type(obj).__name__}")
        inner = None
        if not isinstance(obj.get("tool"), str) and not isinstance(obj.get("name"), str):
            for k in ("tool_call", "function", "action", "call", "tool_calls", "tool"):
                if isinstance(obj.get(k), (Mapping, list)):
                    inner = obj[k]
                    break
        if inner is None:
            break
        obj = inner
    if not isinstance(obj, Mapping):
        return _err("no tool call found")
    tool = obj.get("tool") if isinstance(obj.get("tool"), str) else obj.get("name")
    if not isinstance(tool, str) or not tool.strip():
        return _err('missing "tool" (string)')
    args: Any = {}
    for k in ("args", "arguments", "parameters", "input"):
        if k in obj:
            args = obj[k]
            break
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError as e:
            return _err(f"arguments are not valid JSON: {e}")
    if not isinstance(args, Mapping):
        return _err('"args" must be an object')
    out = {"tool": tool.strip(), "args": dict(args)}
    key = obj.get("idempotency_key")
    if isinstance(key, str) and key:
        out["idempotency_key"] = key
    return out


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_TAG = re.compile(r"<(tool_call|action|function_call)>\s*(.*?)\s*</\1>", re.S | re.I)
_FENCE = re.compile(r"```[a-zA-Z_]*\s*\n?(.*?)```", re.S)
_NAME_JSON = re.compile(r"^\s*([A-Za-z][\w.\-]*)\s*(\{.*\})\s*$", re.S)


def _balanced(text: str, start: int) -> str | None:
    """Substring from `start` (a '{' or '[') to its matching bracket, aware of JSON/Python string literals."""
    open_c = text[start]
    close_c = "}" if open_c == "{" else "]"
    depth, i, q = 0, start, None
    while i < len(text):
        c = text[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if c == q:
                q = None
        elif c in "\"'":
            q = c
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return text[start: i + 1] if c == close_c else None
        i += 1
    return None


def _loads_lenient(s: str) -> Any:
    try:
        return json.loads(s)
    except ValueError:
        pass
    t = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(t)
    except ValueError:
        pass
    t = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", t)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = ast.literal_eval(t)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        raise ValueError("not JSON")
    return v


def _objects(text: str, limit: int = 64):
    n = 0
    for m in re.finditer(r"[{\[]", text):
        sub = _balanced(text, m.start())
        if sub is None:
            continue
        try:
            yield _loads_lenient(sub)
        except ValueError:
            continue
        n += 1
        if n >= limit:
            return


def parse_action(text: Any) -> dict:
    """Extract one tool call from model output; returns ``{"tool","args"}`` or ``{"error": {...}}``."""
    if isinstance(text, Mapping) or isinstance(text, list):
        return normalize_action(text)
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    if not isinstance(text, str):
        return _err(f"expected text, got {type(text).__name__}")
    raw = text
    text = _THINK.sub("", text).strip()
    if not text:
        return _err("empty action", raw)
    try:
        return normalize_action(json.loads(text))
    except ValueError:
        pass
    candidates: list[str] = [m.group(2) for m in _TAG.finditer(text)] + [m.group(1) for m in _FENCE.finditer(text)]
    candidates.append(text)
    last_err = None
    for cand in candidates:
        m = _NAME_JSON.match(cand)
        if m:
            try:
                args = _loads_lenient(m.group(2))
                if isinstance(args, Mapping) and not ("tool" in args or "name" in args):
                    return normalize_action({"tool": m.group(1), "args": args})
            except ValueError:
                pass
        for obj in _objects(cand):
            out = normalize_action(obj)
            if "error" not in out:
                return out
            last_err = out
    if last_err is not None:
        last_err["raw"] = raw[:200]
        return last_err
    return _err("no JSON tool call found", raw)
