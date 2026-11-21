"""Grounded execution of one tool call, with a failure taxonomy and feedback the learner can act on.

A call goes through, in order:

1. reference resolution - aliases shown to the language model (``ACCT1``) become real identifiers; an alias the
   actor never saw is an ``unresolved_reference`` and does not reach the world;
2. tool lookup          - a tool the actor cannot currently call is ``unknown_tool`` (similar names are suggested);
3. schema validation    - Crucible's own validator (`crucible.core.schema`); a violation is ``syntax``;
4. the retry guard      - an identical call that already failed `max_repeat` times, with no successful state-changing
   action of the actor since, is ``loop_blocked`` and is not sent again;
5. the world            - the session's ``execute`` (`World.execute`), which is the authority on everything else. Its error code is mapped to
   ``nonexistent_resource`` | ``permission`` | ``prerequisite`` | ``rate_limited`` | ``runtime`` | ``syntax``.

The executor never repairs or completes a call. It reports precisely what was wrong (the offending argument, the
schema fragment, identifiers of that kind the actor has already seen) and counts every failure by class.
Whether a run of valid actions was a *strategic* failure is decided at episode level by the evaluator.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .. import ensure_crucible
from .knowledge import CapabilityIndex, Knowledge

ensure_crucible()
from crucible.core import schema as S  # noqa: E402

FAILURE_CLASSES = ("syntax", "unknown_tool", "unresolved_reference", "nonexistent_resource", "permission",
                   "prerequisite", "rate_limited", "runtime", "loop_blocked", "budget")
CODE_CLASS = {"invalid_action": "syntax", "unknown_tool": "unknown_tool", "not_found": "nonexistent_resource",
              "forbidden": "permission", "unauthenticated": "permission", "insufficient_funds": "prerequisite",
              "conflict": "prerequisite", "capacity_exceeded": "prerequisite", "budget_exhausted": "prerequisite",
              "rate_limited": "rate_limited", "service_unavailable": "runtime", "timeout": "runtime",
              "sandbox_error": "runtime", "error": "runtime"}


@dataclass
class Outcome:
    ok: bool
    tool: str
    args: dict
    result: Any = None
    error: dict | None = None
    fclass: str = ""                 # "" when ok, else one of FAILURE_CLASSES
    feedback: str = ""               # what was wrong, for the learner
    cost_s: int = 0
    sent: bool = True                # False when the call was stopped before the world (syntax, loop, ...)
    read_only: bool = False
    events: list = field(default_factory=list)

    def to_result(self) -> dict:
        d = {"ok": self.ok, "tool": self.tool, "cost_s": self.cost_s}
        if self.ok:
            d["result"] = self.result
        else:
            d["error"] = dict(self.error or {}, **{"class": self.fclass})
        return d


def call_key(tool: str, args: dict) -> str:
    return hashlib.sha1(json.dumps([tool, args], sort_keys=True, default=str).encode()).hexdigest()[:16]


class GroundedExecutor:
    def __init__(self, exec_fn, caps: CapabilityIndex, know: Knowledge, max_repeat: int = 2):
        """`exec_fn(tool, args) -> dict` is the session protocol's ``execute`` (an `ActionResult` dict)."""
        self.exec_fn, self.caps, self.know = exec_fn, caps, know
        self.max_repeat = max_repeat
        self.failed: dict[str, tuple[int, str]] = {}     # call key -> (count, error code)
        self.counts = {c: 0 for c in FAILURE_CLASSES}
        self.n_calls = 0
        self.n_ok = 0
        self.n_sent = 0

    # ------------------------------------------------------------------ checks that never touch the world
    def precheck(self, tool: str, args: Any) -> Outcome | None:
        if not isinstance(tool, str) or not isinstance(args, dict):
            return self._fail(str(tool), {}, "syntax", "invalid_action",
                              'a call needs a tool name and an object of arguments: {"tool": "...", "args": {...}}')
        unresolved: list = []
        args2 = self.know.resolve(args, unresolved)
        if unresolved:
            return self._fail(tool, args, "unresolved_reference", "unresolved_reference",
                              f"unknown reference(s) {unresolved}: only identifiers you have observed can be used")
        if not self.caps.has(tool):
            near = self.caps.nearest_names(tool)
            return self._fail(tool, args2, "unknown_tool", "unknown_tool",
                              f"no tool named {tool!r} is available to you" + (f"; similar tools: {', '.join(near)}" if near else ""))
        spec = self.caps.tools[tool]
        sc = spec.get("inputSchema") or {"type": "object"}
        args2 = S.with_defaults(sc, args2)
        errs = S.validate(sc, args2)
        if errs:
            return self._fail(tool, args2, "syntax", "invalid_action", "; ".join(errs[:4]) + self._schema_hint(sc, errs))
        key = call_key(tool, args2)
        n, code = self.failed.get(key, (0, ""))
        if n >= self.max_repeat:
            return self._fail(tool, args2, "loop_blocked", "loop_blocked",
                              f"this exact call already failed {n} times ({code}) and nothing you did since could "
                              f"change that; it was not sent again. Change the arguments or do something else first.")
        return None

    def _schema_hint(self, sc: dict, errs: list[str]) -> str:
        hints = []
        props = sc.get("properties") or {}
        for e in errs[:3]:
            name = e.split(":", 1)[0].replace("args.", "").split(".")[0].split("[")[0]
            p = props.get(name)
            if p is not None:
                frag = {k: p[k] for k in ("type", "enum", "pattern", "minimum", "maximum", "maxLength", "description")
                        if k in p and p[k] != ""}
                hints.append(f"{name}: {json.dumps(frag)[:160]}")
            elif "unknown" in e or "additional" in e:
                hints.append("allowed arguments: " + ", ".join(sorted(props)))
        return (" | schema: " + " ; ".join(dict.fromkeys(hints))) if hints else ""

    def _fail(self, tool, args, fclass, code, message, cost=0, sent=False, error=None, events=None) -> Outcome:
        self.counts[fclass] += 1
        return Outcome(False, tool, args if isinstance(args, dict) else {}, None,
                       error or {"code": code, "message": message}, fclass, message, cost, sent, events=events or [])

    # ------------------------------------------------------------------ execution
    def execute(self, tool: str, args: Any) -> Outcome:
        self.n_calls += 1
        pre = self.precheck(tool, args)
        if pre is not None:
            return pre
        spec = self.caps.tools[tool]
        args = S.with_defaults(spec.get("inputSchema") or {"type": "object"}, self.know.resolve(args))
        ro = bool((spec.get("annotations") or {}).get("readOnlyHint"))
        r = self.exec_fn(tool, args)
        self.n_sent += 1
        key = call_key(tool, args)
        cost = int(r.get("cost_s", 0) or 0)
        if r.get("ok"):
            self.n_ok += 1
            if not ro:
                self.failed.clear()       # the actor changed something: earlier failures may no longer hold
            else:
                self.failed.pop(key, None)
            return Outcome(True, tool, args, r.get("result"), None, "", "", cost, True, ro)
        err = r.get("error") or {"code": "error", "message": ""}
        code = err.get("code", "error")
        fclass = CODE_CLASS.get(code, "runtime")
        n, _ = self.failed.get(key, (0, ""))
        self.failed[key] = (n + 1, code)
        fb = err.get("message", "")
        if fclass == "nonexistent_resource":
            for k, v in args.items():
                if isinstance(v, str) and v and v in fb:
                    known = [x for x in self.know.known_ids(k) if x != v]
                    if known:
                        fb += f" | `{k}` values you have seen: {', '.join(known[:5])}"
                    break
        return self._fail(tool, args, fclass, code, fb, cost, True, err)

    def stats(self) -> dict:
        return {"calls": self.n_calls, "sent": self.n_sent, "ok": self.n_ok,
                "valid_rate": (self.n_ok / self.n_calls) if self.n_calls else 0.0,
                "failures": {k: v for k, v in self.counts.items() if v}}
