"""Action programs: the open-ended action representation, and its interpreter.

An action is a *program* over the world's primitive operations, not a choice from a list of strategies. The
language is small JSON so that the frozen language model can write it, the trainable controller can build and edit
it, and the agent's own tools can be stored in it:

    {"plan": "free text", "steps": [STEP, ...]}

    STEP := {"call": "tool.name", "args": {name: EXPR}, "as": "x", "on_fail": "stop"|"continue"}
          | {"wait": seconds}
          | {"if": COND, "then": [STEP], "else": [STEP]}
          | {"repeat": n, "do": [STEP]}                      | {"while": COND, "do": [STEP], "max": n}
          | {"foreach": EXPR, "as": "item", "do": [STEP]}
          | {"set": "name", "value": EXPR}
          | {"code": "python source defining main(input, caller)", "input": EXPR, "as": "y"}
          | {"use": "tool_made_by_the_agent", "args": {...}, "as": "z"}
          | {"define_tool": {"name", "description", "params", "steps" | "code"}}
          | {"subgoal": "free text"} | {"note": "free text"} | {"stop": "reason"}

    EXPR := any JSON literal
          | "$x" | "$x.field.0.other"                         (value bound earlier; ``result`` may be omitted)
          | {"expr": [op, EXPR, EXPR]}   op in add sub mul div min max int round len
          | {"gen": "what to write"}                          (ask the language model for this value)

    COND := {"ok": "$x"} | {"exists": "$x.path"} | {"cmp": [EXPR, "<"|"<="|"=="|"!="|">="|">", EXPR]}
          | {"not": COND} | {"all": [COND]} | {"any": [COND]}

Every primitive operation goes through the grounded executor (so it is validated, recorded and timed like any
other action). ``code`` runs inside Crucible's sandbox through the world's own ``code.*`` tools. The interpreter
enforces a budget of primitive operations and bounded loops; a failed call stops the program unless the step says
``"on_fail": "continue"``, and the trace says exactly where and why it stopped.
"""
from __future__ import annotations

import copy
import operator
from dataclasses import dataclass, field
from typing import Any, Callable

from ..llm.base import extract_json

_OPS = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": lambda a, b: a / b if b else 0,
        "min": min, "max": max}
_CMP = {"<": operator.lt, "<=": operator.le, "==": operator.eq, "!=": operator.ne, ">=": operator.ge, ">": operator.gt}
STEP_KINDS = ("call", "wait", "if", "repeat", "while", "foreach", "set", "code", "use", "define_tool", "subgoal",
              "note", "stop")


class ProgramError(ValueError):
    """The program text is not a valid program (reported to the learner as a syntax failure)."""


def normalize_step(s: Any) -> dict:
    """Accept the documented forms plus two common variants (``{"tool": .., "args": ..}`` and ``{"op": "call", ..}``)."""
    if not isinstance(s, dict):
        raise ProgramError(f"a step must be an object, got {type(s).__name__}")
    s = dict(s)
    if "op" in s and s["op"] in STEP_KINDS and s["op"] not in s:
        op = s.pop("op")
        if op == "call":
            s["call"] = s.pop("tool", None)
        elif op == "wait":
            s["wait"] = s.pop("seconds", s.pop("s", 3600))
        elif op in ("subgoal", "note", "stop"):
            s[op] = s.pop("text", s.pop("reason", ""))
        elif op == "code":
            s["code"] = s.pop("source", "")
        elif op == "use":
            s["use"] = s.pop("name", s.pop("tool", None))
        else:
            s[op] = s.pop("value", s.get(op, True))
    if "tool" in s and "call" not in s:
        s["call"] = s.pop("tool")
    kinds = [k for k in STEP_KINDS if k in s]
    if len(kinds) != 1:
        raise ProgramError(f"a step needs exactly one of {', '.join(STEP_KINDS)}; got keys {sorted(s)}")
    k = kinds[0]
    if k == "call":
        if not isinstance(s["call"], str):
            raise ProgramError("call: tool name must be a string")
        if not isinstance(s.get("args", {}), dict):
            raise ProgramError(f"call {s['call']}: args must be an object")
        s.setdefault("args", {})
    for body in ("then", "else", "do"):
        if body in s:
            if not isinstance(s[body], list):
                raise ProgramError(f"{k}: `{body}` must be a list of steps")
            s[body] = [normalize_step(x) for x in s[body]]
    if k == "define_tool":
        d = s["define_tool"]
        if not isinstance(d, dict) or not isinstance(d.get("name"), str):
            raise ProgramError("define_tool needs an object with a name")
        if "steps" in d:
            d["steps"] = [normalize_step(x) for x in d["steps"]]
    s["_kind"] = k
    return s


def parse_program(x: Any) -> dict:
    """Program from model text or a JSON value. Accepts a program object, a bare list of steps or one step."""
    if isinstance(x, str):
        v = extract_json(x)
        if v is None:
            raise ProgramError("no JSON program found in the reply")
        x = v
    if isinstance(x, list):
        x = {"steps": x}
    if not isinstance(x, dict):
        raise ProgramError("a program must be a JSON object")
    if "steps" not in x:
        x = {"steps": [x]}
    if not isinstance(x["steps"], list):
        raise ProgramError("`steps` must be a list")
    return {"plan": str(x.get("plan", ""))[:1000], "steps": [normalize_step(s) for s in x["steps"]]}


def program_tools(prog: dict) -> list[str]:
    """Names of the primitive tools a program may call (static scan, for dependency records)."""
    out: list[str] = []

    def walk(steps):
        for s in steps:
            if s.get("_kind") == "call":
                out.append(s["call"])
            for b in ("then", "else", "do"):
                walk(s.get(b, []))
    walk(prog.get("steps", []))
    return sorted(set(out))


def strip_internal(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: strip_internal(v) for k, v in x.items() if k != "_kind"}
    if isinstance(x, list):
        return [strip_internal(v) for v in x]
    return x


@dataclass
class StepTrace:
    kind: str
    tool: str = ""
    args: dict = field(default_factory=dict)
    ok: bool = True
    fclass: str = ""
    feedback: str = ""
    result: Any = None
    cost_s: int = 0
    depth: int = 0


@dataclass
class ProgramTrace:
    steps: list = field(default_factory=list)
    ok: bool = True
    stopped: str = ""                 # why execution ended early ("" = ran to the end)
    env: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    subgoals: list = field(default_factory=list)
    defined: list = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind in ("call", "code", "use"))

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.steps if not s.ok)


class _Stop(Exception):
    pass


class Interpreter:
    """Runs a program. `call(tool, args) -> Outcome`, `wait(seconds)`, optional `gen(brief) -> value`,
    `code(source, input) -> Outcome`, `library` (agent-made tools: ``get(name)``, ``register(spec) -> (ok, msg)``)."""

    def __init__(self, call: Callable, wait: Callable, gen: Callable | None = None, code: Callable | None = None,
                 library=None, max_ops: int = 16, max_loop: int = 8, max_depth: int = 4):
        self.call, self.wait, self.gen, self.code, self.library = call, wait, gen, code, library
        self.max_ops, self.max_loop, self.max_depth = max_ops, max_loop, max_depth
        self._ops = 0

    # ------------------------------------------------------------------ expressions
    def _ref(self, path: str, env: dict):
        parts = path[1:].split(".")
        if parts[0] not in env:
            raise KeyError(f"${parts[0]} is not bound")
        v = env[parts[0]]
        for i, p in enumerate(parts[1:]):
            if isinstance(v, dict):
                if p in v:
                    v = v[p]
                elif i == 0 and isinstance(v.get("result"), (dict, list)):
                    r = v["result"]
                    if isinstance(r, dict) and p in r:
                        v = r[p]
                    elif isinstance(r, list) and p.lstrip("-").isdigit() and -len(r) <= int(p) < len(r):
                        v = r[int(p)]
                    else:
                        raise KeyError(f"{path}: no field {p!r}")
                else:
                    raise KeyError(f"{path}: no field {p!r}")
            elif isinstance(v, list) and p.lstrip("-").isdigit() and -len(v) <= int(p) < len(v):
                v = v[int(p)]
            else:
                raise KeyError(f"{path}: cannot take {p!r} of {type(v).__name__}")
        if len(parts) == 1 and isinstance(v, dict) and set(v) >= {"ok", "result"}:
            v = v["result"]
        return v

    def eval(self, x: Any, env: dict):
        if isinstance(x, str):
            if x.startswith("$") and len(x) > 1 and (x[1].isalpha() or x[1] == "_"):
                return self._ref(x, env)
            return x
        if isinstance(x, list):
            return [self.eval(v, env) for v in x]
        if isinstance(x, dict):
            if set(x) == {"expr"} and isinstance(x["expr"], list) and x["expr"]:
                op, *a = x["expr"]
                vals = [self.eval(v, env) for v in a]
                if op in ("int", "round"):
                    return int(round(float(vals[0])))
                if op == "len":
                    return len(vals[0])
                if op in _OPS and len(vals) == 2:
                    return _OPS[op](vals[0], vals[1])
                raise KeyError(f"unknown expression {op!r}")
            if set(x) == {"gen"}:
                if self.gen is None:
                    raise KeyError("no language model is available for {\"gen\": ...}")
                return self.gen(str(x["gen"]))
            return {k: self.eval(v, env) for k, v in x.items()}
        return x

    def cond(self, c: Any, env: dict) -> bool:
        if isinstance(c, bool):
            return c
        if not isinstance(c, dict):
            raise KeyError("condition must be an object")
        if "ok" in c:
            v = env.get(str(c["ok"]).lstrip("$").split(".")[0])
            return bool(isinstance(v, dict) and v.get("ok"))
        if "exists" in c:
            try:
                return self._ref(str(c["exists"]), env) not in (None, [], {}, "")
            except KeyError:
                return False
        if "cmp" in c:
            a, op, b = c["cmp"]
            return bool(_CMP[op](self.eval(a, env), self.eval(b, env)))
        if "not" in c:
            return not self.cond(c["not"], env)
        if "all" in c:
            return all(self.cond(x, env) for x in c["all"])
        if "any" in c:
            return any(self.cond(x, env) for x in c["any"])
        raise KeyError(f"unknown condition {sorted(c)}")

    # ------------------------------------------------------------------ execution
    def run(self, program: dict, env: dict | None = None) -> ProgramTrace:
        tr = ProgramTrace(env=dict(env or {}))
        self._ops = 0
        try:
            self._block(program.get("steps", []), tr, 0)
        except _Stop as e:
            tr.stopped = str(e)
        tr.ok = tr.n_failed == 0 and not tr.stopped.startswith(("error", "failed", "budget"))
        return tr

    def _bump(self) -> None:
        self._ops += 1
        if self._ops > self.max_ops:
            raise _Stop(f"budget: more than {self.max_ops} operations in one program")

    def _outcome(self, tr, kind, name, args, out, depth, bind, on_fail):
        st = StepTrace(kind, name, args, out.ok, out.fclass, out.feedback, out.result, out.cost_s, depth)
        tr.steps.append(st)
        if bind:
            tr.env[bind] = {"ok": out.ok, "result": out.result, "error": out.error}
        tr.env["_last"] = {"ok": out.ok, "result": out.result, "error": out.error}
        if not out.ok and on_fail != "continue":
            raise _Stop(f"failed: {name} ({out.fclass}): {out.feedback[:200]}")

    def _block(self, steps: list, tr: ProgramTrace, depth: int) -> None:
        if depth > self.max_depth:
            raise _Stop("error: program nested too deeply")
        for s in steps:
            k = s.get("_kind") or normalize_step(s)["_kind"]
            try:
                self._step(k, s, tr, depth)
            except KeyError as e:
                tr.steps.append(StepTrace(k, str(s.get("call", "")), {}, False, "syntax", f"program error: {e}", depth=depth))
                raise _Stop(f"error: {e}") from None

    def _step(self, k: str, s: dict, tr: ProgramTrace, depth: int) -> None:
        env = tr.env
        if k == "call":
            self._bump()
            args = self.eval(s.get("args", {}), env)
            self._outcome(tr, "call", s["call"], args, self.call(s["call"], args), depth, s.get("as"), s.get("on_fail"))
        elif k == "wait":
            sec = int(float(self.eval(s["wait"], env)))
            tr.steps.append(StepTrace("wait", "wait", {"seconds": self.wait(sec)}, depth=depth))
        elif k == "if":
            self._block(s.get("then", []) if self.cond(s["if"], env) else s.get("else", []), tr, depth + 1)
        elif k == "repeat":
            for _ in range(min(int(self.eval(s["repeat"], env)), self.max_loop)):
                self._block(s.get("do", []), tr, depth + 1)
        elif k == "while":
            for _ in range(min(int(s.get("max", self.max_loop)), self.max_loop)):
                if not self.cond(s["while"], env):
                    break
                self._block(s.get("do", []), tr, depth + 1)
        elif k == "foreach":
            items = self.eval(s["foreach"], env)
            if not isinstance(items, list):
                raise KeyError("foreach needs a list")
            for it in items[: self.max_loop]:
                env[str(s.get("as", "item"))] = it
                self._block(s.get("do", []), tr, depth + 1)
        elif k == "set":
            env[str(s["set"])] = self.eval(s.get("value"), env)
        elif k == "code":
            self._bump()
            if self.code is None:
                raise KeyError("code execution is not available")
            inp = self.eval(s.get("input"), env)
            self._outcome(tr, "code", "code", {"input": inp}, self.code(str(s["code"]), inp), depth, s.get("as"), s.get("on_fail"))
        elif k == "use":
            self._bump()
            tool = self.library.get(str(s["use"])) if self.library is not None else None
            if tool is None:
                raise KeyError(f"no tool named {s['use']!r} in your library")
            args = self.eval(s.get("args", {}), env)
            out = self.use(tool, args, tr, depth)
            self._outcome(tr, "use", tool["name"], args, out, depth, s.get("as"), s.get("on_fail"))
        elif k == "define_tool":
            if self.library is None:
                raise KeyError("no tool library")
            okv, msg = self.library.register(copy.deepcopy(strip_internal(s["define_tool"])))
            tr.steps.append(StepTrace("define_tool", str(s["define_tool"].get("name")), {}, okv, "" if okv else "syntax", msg, depth=depth))
            if okv:
                tr.defined.append(s["define_tool"]["name"])
        elif k == "subgoal":
            tr.subgoals.append(str(s["subgoal"])[:400])
        elif k == "note":
            tr.notes.append(str(s["note"])[:400])
        elif k == "stop":
            raise _Stop(f"stop: {str(s['stop'])[:200]}")

    def use(self, tool: dict, args: dict, tr: ProgramTrace, depth: int):
        """Run an agent-made tool: a macro (nested steps with its parameters bound) or a code tool."""
        from ..grounding.executor import Outcome
        missing = [p for p in (tool.get("params", {}).get("required") or []) if p not in args]
        if missing:
            return Outcome(False, tool["name"], args, None, {"code": "invalid_action", "message": f"missing {missing}"},
                           "syntax", f"tool {tool['name']} needs arguments {missing}", sent=False)
        if tool.get("kind") == "code":
            return self.code(tool["code"], args)
        sub = ProgramTrace(env={**{k: v for k, v in args.items()}, "args": args})
        n0 = len(tr.steps)
        try:
            saved, tr.env = tr.env, sub.env
            try:
                self._block([normalize_step(x) for x in tool.get("steps", [])], tr, depth + 1)
            finally:
                sub.env, tr.env = tr.env, saved
            stopped = ""
        except _Stop as e:
            stopped = str(e)
        inner = tr.steps[n0:]
        okv = not stopped and all(st.ok for st in inner)
        last = sub.env.get("_last", {})
        return Outcome(okv, tool["name"], args, last.get("result") if okv else None,
                       None if okv else {"code": "tool_failed", "message": stopped or "a step failed"},
                       "" if okv else (inner[-1].fclass if inner and not inner[-1].ok else "runtime"),
                       "" if okv else (stopped or "a step failed"), sum(st.cost_s for st in inner), True)
