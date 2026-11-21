"""Tools made by the agent: a library of reusable procedures it wrote or extracted from its own experience.

Two kinds, both stored as data the agent can inspect and call again in later plans and later worlds:

* ``macro`` - a program (see `superlearner.actions.program`) with parameters. `macro_from_trace` generalises a
  stretch of the agent's own successful calls: a value that came out of an earlier step becomes a reference to that
  step, an identifier that existed before the stretch becomes a parameter, everything else stays literal.
* ``code``  - Python source defining ``main(input, caller)``; it runs inside Crucible's sandbox through the world's
  own ``code.*`` tools (the agent pays the world's price for running it).

For every tool the library records: source (steps or code), parameter schema, provenance (who made it, from which
episode and world, by which mechanism), dependencies (primitive tools it calls), the hosts it touches (its
permission footprint), a version, and outcome statistics from real use. A tool is ``verified`` only after it has
run to completion successfully in an episode other than the one that created it; registering or describing a tool
earns nothing by itself.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from typing import Any

from ..actions.program import ProgramError, normalize_step, program_tools, strip_internal
from .knowledge import alias_kind

_NAME = re.compile(r"^[a-z][a-z0-9_]{2,39}$")


def _find_path(x: Any, v: Any, path: str = "", depth: int = 0) -> str | None:
    """JSON path (dot notation) at which value `v` occurs inside `x`, or None."""
    if depth > 5:
        return None
    if isinstance(x, (str, int)) and not isinstance(x, bool) and x == v:
        return path
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            p = _find_path(x[k], v, f"{path}.{k}" if path else str(k), depth + 1)
            if p is not None:
                return p
    elif isinstance(x, list):
        for i, y in enumerate(x[:8]):
            p = _find_path(y, v, f"{path}.{i}" if path else str(i), depth + 1)
            if p is not None:
                return p
    return None


def macro_from_trace(calls: list[tuple[str, dict, Any]], actor: str, name: str, description: str = "") -> dict:
    """Generalise successful calls ``[(tool, args, result)]`` into a parameterised macro tool."""
    steps, params, required = [], {}, []
    by_value: dict = {}
    for j, (tool, args, _res) in enumerate(calls):
        new_args = {}
        for k, v in args.items():
            expr: Any = v
            if isinstance(v, (str, int)) and not isinstance(v, bool) and not (isinstance(v, int) and abs(v) < 2):
                src = None
                for i in range(j - 1, -1, -1):
                    p = _find_path(calls[i][2], v)
                    if p is not None:
                        src = f"$s{i}.{p}" if p else f"$s{i}"
                        break
                if src is not None:
                    expr = src
                elif v == actor:
                    expr = "$me"
                elif isinstance(v, str) and alias_kind(v) is not None:
                    pname = by_value.get(v)
                    if pname is None:
                        pname = k if k not in params else f"{k}_{len(params)}"
                        by_value[v] = pname
                        params[pname] = {"type": "string", "description": f"`{k}` of {tool}"}
                        required.append(pname)
                    expr = f"${pname}"
            new_args[k] = expr
        steps.append({"call": tool, "args": new_args, "as": f"s{j}"})
    return {"name": name, "kind": "macro", "steps": steps,
            "description": description or ("Runs " + " -> ".join(t for t, _, _ in calls)),
            "params": {"type": "object", "properties": params, "required": required, "additionalProperties": False}}


class ToolLibrary:
    def __init__(self):
        self.tools: dict[str, dict] = {}
        self.history: list[dict] = []          # every registration, including rejected ones and old versions

    # ------------------------------------------------------------------ registration
    def register(self, spec: dict, provenance: dict | None = None) -> tuple[bool, str]:
        spec = copy.deepcopy(spec)
        name = spec.get("name")
        if not isinstance(name, str) or not _NAME.match(name):
            return self._reject(spec, "tool name must match [a-z][a-z0-9_]{2,39}")
        kind = spec.get("kind") or ("code" if "code" in spec else "macro")
        spec["kind"] = kind
        if kind == "macro":
            try:
                steps = [normalize_step(s) for s in spec.get("steps") or []]
            except ProgramError as e:
                return self._reject(spec, f"invalid steps: {e}")
            if not steps:
                return self._reject(spec, "a macro needs at least one step")
            spec["steps"] = strip_internal(steps)
            spec["deps"] = program_tools({"steps": steps})
        elif kind == "code":
            src = spec.get("code")
            if not isinstance(src, str) or len(src) > 60_000:
                return self._reject(spec, "code tool needs source text")
            try:
                tree = ast.parse(src)          # parsed only; it is executed in the world's sandbox, never here
            except SyntaxError as e:
                return self._reject(spec, f"code does not parse: {e.msg} (line {e.lineno})")
            if not any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body):
                return self._reject(spec, "code must define main(input, caller)")
            spec["deps"] = ["code.create_repo", "code.commit", "code.run"]
        else:
            return self._reject(spec, f"unknown tool kind {kind!r}")
        p = spec.get("params") or {"type": "object", "properties": {}}
        if not isinstance(p, dict) or p.get("type", "object") != "object":
            return self._reject(spec, "params must be a JSON object schema")
        p.setdefault("properties", {})
        spec["params"] = p
        spec["description"] = str(spec.get("description") or "")[:400]
        body = json.dumps([spec.get("steps"), spec.get("code"), p], sort_keys=True, default=str)
        spec["hash"] = hashlib.sha1(body.encode()).hexdigest()[:12]
        old = self.tools.get(name)
        if old is not None and old["hash"] == spec["hash"]:
            return True, f"tool {name} already exists (version {old['version']})"
        spec["version"] = (old["version"] + 1) if old else 1
        spec["provenance"] = dict(provenance or {}, parent=(old or {}).get("hash"))
        spec["effects"] = sorted({d.split(".")[0] for d in spec["deps"]})
        spec["stats"] = {"calls": 0, "ok": 0, "fail": 0, "episodes_ok": [], "errors": {}}
        spec["verified"] = False
        self.tools[name] = spec
        self.history.append({"event": "registered", "name": name, "version": spec["version"], "hash": spec["hash"],
                             "provenance": spec["provenance"]})
        return True, f"registered tool {name} (version {spec['version']})"

    def _reject(self, spec, msg) -> tuple[bool, str]:
        self.history.append({"event": "rejected", "name": str(spec.get("name"))[:60], "reason": msg})
        return False, msg

    # ------------------------------------------------------------------ use
    def get(self, name: str) -> dict | None:
        return self.tools.get(name)

    def names(self) -> list[str]:
        return sorted(self.tools)

    def record(self, name: str, ok: bool, fclass: str = "", episode: str = "") -> None:
        t = self.tools.get(name)
        if t is None:
            return
        s = t["stats"]
        s["calls"] += 1
        if ok:
            s["ok"] += 1
            created_in = (t.get("provenance") or {}).get("episode")
            if episode and episode != created_in:
                if episode not in s["episodes_ok"]:
                    s["episodes_ok"] = (s["episodes_ok"] + [episode])[-20:]
                t["verified"] = True
        else:
            s["fail"] += 1
            s["errors"][fclass or "error"] = s["errors"].get(fclass or "error", 0) + 1

    def describe(self, name: str) -> str:
        t = self.tools[name]
        ps = ", ".join(f"{k}{'' if k in (t['params'].get('required') or []) else '?'}" for k in t["params"]["properties"])
        s = t["stats"]
        return (f"{name}({ps}) [agent tool v{t['version']}, {t['kind']}, "
                f"{'verified' if t['verified'] else 'unverified'}, {s['ok']}/{s['calls']} ok] - {t['description']}")

    def inspect(self, name: str) -> dict:
        return copy.deepcopy(self.tools[name])

    def state_dict(self) -> dict:
        return {"tools": copy.deepcopy(self.tools), "history": list(self.history[-500:])}

    def load_state_dict(self, s: dict) -> None:
        self.tools = copy.deepcopy(s.get("tools", {}))
        self.history = list(s.get("history", []))
