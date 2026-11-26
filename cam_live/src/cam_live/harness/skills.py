"""cam_live.harness.skills — automatically acquired executable skills (spec 12A.8, 12A.9).

Owned by workstream ES.

A :class:`Skill` is a typed, parameterised program of ``ActionSpec`` steps abstracted from *successful* executions:

* :class:`SkillInducer` aligns two or more traces (or uses the declared inputs of a single trace), keeps the
  reusable segment, turns what varied into parameters, turns values that the environment produced in an earlier
  step into captures, and derives pre / postconditions from the observed before / after states.
* :class:`SkillExecutor` (modality ``skill``; tools ``skill.run``, ``skill.list``, ``skill.describe``,
  ``skill.induce``) expands a skill into its primitive steps and sends every one of them through
  ``ExecContext.check`` and the normal ``ModalityExecutor`` of its modality. A skill therefore never has more
  permission than its steps issued one by one, and its required scopes are exactly the union of its steps' scopes.
* A candidate is registered only after it has been *validated*: executed with fresh arguments in the authorized
  environment with every step succeeding and every postcondition holding.
* :class:`SkillLibrary` stores every version on disk and monitors use: each attributable run updates a Beta
  posterior of reliability; low reliability, or repeated postcondition failures after the interface changed,
  mark the skill *stale*. A stale skill is never executed; it is repaired (re-induced from new successful
  traces, version bump, re-validated) or retired.
* Induction and execution refuse any step that addresses the authorization system, the credential store or the
  independent verifier (:func:`protected_reasons`).

The result of ``skill.run`` is one PROCEDURE-level action: it records the start state, the underlying primitive
operations, the elapsed time, the resources used and the termination condition, so that a procedure counts as one
decision for the learner while its low-level trace is kept.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..contracts import (ActionResult, ActionSpec, AuthVerdict, Clock, Cost, ExecContext, FailureClass, Level,
                         Modality, Observation, OutcomeKind, PrimitiveOp, Provenance, Quantity, Record, ResultStatus,
                         RunFiles, ToolDescriptor, Trust, VerificationCondition, new_id)
from .failures import FailureEvidence, authorization_result, classify, failed_result
from .http_api import schema_errors, structure_hash

STATUS_CANDIDATE, STATUS_VALIDATED, STATUS_STALE, STATUS_RETIRED = "candidate", "validated", "stale", "retired"

#: nothing a skill does may address the authorization system, the credential store or the independent verifier
PROTECTED_OPERATION_PREFIXES = ("auth.", "authorization.", "policy.", "approval.", "approvals.", "verifier.",
                                "verify.", "secret.", "secrets.", "vault.", "audit.", "credential.", "skill.")
PROTECTED_SCOPE_PREFIXES = ("auth", "policy", "approval", "verifier", "secret", "vault", "audit", "credential")
PROTECTED_PATH_MARKERS = (RunFiles.AUDIT, RunFiles.VERIFIER, RunFiles.CONTROL, "/vault/", "/vault", "harness/auth",
                          "verifier_service", "auth.sock", "verify.sock", "verifier.key", "policy.yaml")
#: failure classes that say something about the skill itself (the others are not evidence about its reliability)
ATTRIBUTABLE = (FailureClass.PAGE_CHANGED, FailureClass.WRONG_SITE_ASSUMPTION, FailureClass.STALE_OBSERVATION,
                FailureClass.INVALID_ARGUMENTS, FailureClass.VISUAL_GROUNDING, FailureClass.UNKNOWN,
                FailureClass.STRATEGIC)
_VOLATILE = re.compile(r"(elapsed_s|_bytes|\.bytes|ratelimit|retry_after)|^d:(headers|truncated|structure_hash|redirects)(/|$)")
#: transport metadata of an observation is not a value "produced by the step" and is never captured
_TRANSPORT_KEYS = ("method", "status", "headers", "truncated", "structure_hash", "redirects")
_MISSING = object()


class SkillInductionError(ValueError):
    pass


class SkillBindError(LookupError):
    pass


# --------------------------------------------------------------------------------------------------------------
# Beta posterior
# --------------------------------------------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function I_x(a, b) (continued fraction, Lentz's method)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(ln) * _betacf(a, b, x) / a
    return 1.0 - math.exp(ln) * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def beta_interval(a: float, b: float, level: float = 0.9) -> Tuple[float, float]:
    """Equal-tailed credible interval of Beta(a, b)."""
    t = (1.0 - level) / 2.0
    return beta_ppf(t, a, b), beta_ppf(1.0 - t, a, b)


# --------------------------------------------------------------------------------------------------------------
# JSON helpers, templates, state
# --------------------------------------------------------------------------------------------------------------
def _same(a: Any, b: Any) -> bool:
    if a is _MISSING or b is _MISSING:
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if a == b:
        return True
    if isinstance(a, str) != isinstance(b, str) and isinstance(a, (str, int)) and isinstance(b, (str, int)):
        return str(a) == str(b)
    return False


def _flatten(v: Any, prefix: Tuple = (), out: Optional[Dict[Tuple, Any]] = None, depth: int = 7,
             max_list: int = 24) -> Dict[Tuple, Any]:
    out = {} if out is None else out
    if depth < 0 or len(out) > 4000:
        return out
    if isinstance(v, dict):
        for k, x in v.items():
            _flatten(x, prefix + (str(k),), out, depth - 1, max_list)
    elif isinstance(v, list):
        out[prefix + ("#",)] = len(v)
        for i, x in enumerate(v[:max_list]):
            _flatten(x, prefix + (i,), out, depth - 1, max_list)
    else:
        out[prefix] = v
    return out


def _get(data: Any, path: Sequence[Any]) -> Any:
    cur = data
    for p in path:
        if p == "#" and isinstance(cur, list):
            return len(cur)
        if isinstance(cur, dict):
            if p not in cur:
                raise KeyError(p)
            cur = cur[p]
        elif isinstance(cur, list):
            if not isinstance(p, int) or not -len(cur) <= p < len(cur):
                raise KeyError(p)
            cur = cur[p]
        else:
            raise KeyError(p)
    return cur


def _arg_leaves(v: Any, prefix: Tuple[str, ...] = ()) -> Dict[Tuple[str, ...], Any]:
    """Leaves of an argument tree; dictionaries are descended, everything else (lists included) is a leaf."""
    if isinstance(v, dict) and v:
        out: Dict[Tuple[str, ...], Any] = {}
        for k, x in v.items():
            out.update(_arg_leaves(x, prefix + (str(k),)))
        return out
    return {prefix: v}


def _set_leaf(tree: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur = tree
    for p in path[:-1]:
        cur = cur.setdefault(p, {})
    cur[path[-1]] = value


def _cast(v: Any, cast: str) -> Any:
    if cast == "str":
        return str(v)
    if cast == "int":
        return int(v)
    return v


def _is_dynamic(t: Any) -> bool:
    return isinstance(t, dict) and len(t) == 1 and next(iter(t)) in ("$param", "$tpl", "$capture", "$lookup")


def bind(template: Any, args: Dict[str, Any], step_data: Sequence[Any] = ()) -> Any:
    """Instantiate a template: ``$param`` from the call arguments, ``$capture`` / ``$lookup`` from the observation
    data of earlier steps, ``$tpl`` by string concatenation."""
    if isinstance(template, dict):
        if _is_dynamic(template):
            kind, spec = next(iter(template.items()))
            if kind == "$param":
                if spec not in args:
                    raise SkillBindError(f"missing argument {spec!r}")
                return args[spec]
            if kind == "$tpl":
                return "".join(str(bind(p, args, step_data)) for p in spec)
            if kind == "$capture":
                try:
                    return _cast(_get(step_data[spec["step"]], spec["path"]), spec.get("cast", ""))
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    raise SkillBindError(f"step {spec['step']} did not yield {'/'.join(map(str, spec['path']))}") from e
            try:
                records = _get(step_data[spec["step"]], spec["list"])
            except (KeyError, IndexError, TypeError) as e:
                raise SkillBindError(f"step {spec['step']} did not yield the list {'/'.join(map(str, spec['list']))}") from e
            where = {g: bind(t, args, step_data) for g, t in spec["where"].items()}
            hits = [r for r in records if isinstance(r, dict)
                    and all(_same(r.get(g, _MISSING), w) for g, w in where.items())] if isinstance(records, list) else []
            if len(hits) != 1 or spec["field"] not in hits[0]:
                raise SkillBindError(f"step {spec['step']}: {len(hits)} records match {sorted(where)} (need exactly 1)")
            return _cast(hits[0][spec["field"]], spec.get("cast", ""))
        return {k: bind(v, args, step_data) for k, v in template.items()}
    if isinstance(template, list):
        return [bind(v, args, step_data) for v in template]
    return template


def _obs_data(observations: Sequence[Observation]) -> Dict[str, Any]:
    """The structured payload of a step: the data of its last non-error observation."""
    for o in reversed(list(observations)):
        if o.kind != "error":
            return o.data
    return {}


def _obs_text(observations: Sequence[Observation]) -> str:
    parts = []
    for o in observations:
        parts.append(o.text)
        if o.data:
            parts.append(json.dumps(o.data, sort_keys=True, default=str, ensure_ascii=False))
    return "\n".join(parts)


def state_leaves(observations: Sequence[Observation]) -> Dict[str, Any]:
    """Flat view of observed state: ``q:<quantity name>`` and ``d:<path>`` leaves (later observations win)."""
    out: Dict[str, Any] = {}
    for o in observations:
        for q in o.quantities:
            out["q:" + q.name] = q.value
        for path, v in _flatten(o.data).items():
            out["d:" + "/".join(f"[{p}]" if isinstance(p, int) else str(p) for p in path)] = v
    return out


_OPS = {
    "eq": lambda a, b: _same(a, b), "ne": lambda a, b: not _same(a, b),
    "ge": lambda a, b: a >= b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "lt": lambda a, b: a < b,
    "contains": lambda a, b: str(b) in str(a),
}


def evaluate_condition(cond: VerificationCondition, observations: Sequence[Observation],
                       args: Optional[Dict[str, Any]] = None, before: Optional[Sequence[Observation]] = None,
                       info: Optional[Dict[str, Any]] = None) -> Optional[bool]:
    """Evaluate a verification condition on observed state. ``True`` / ``False`` when it can be decided from the
    observations, ``None`` when the needed evidence is not there (never assumed either way).

    Kinds: ``text_present`` {text, absent?}; ``url_matches`` {pattern}; ``metric`` / ``element_state``
    {name | path, op, value} with op in eq, ne, ge, le, gt, lt, contains, exists, delta_eq, delta_ge;
    ``program`` {check: "steps_ok" | "tools", operations?}.
    """
    kind = cond.kind
    if kind in ("", "none"):
        return None
    try:
        spec = bind(cond.spec, args or {})
    except SkillBindError:
        return None
    info = info or {}
    if kind == "text_present":
        if not observations:
            return None
        found = str(spec.get("text", "")) in _obs_text(observations)
        return (not found) if spec.get("absent") else found
    if kind == "url_matches":
        urls = [u for o in observations for u in (o.data.get("url"), o.provenance.uri) if u]
        if not urls:
            return None
        return any(re.search(spec.get("pattern", ""), str(u)) is not None for u in urls)
    if kind in ("metric", "element_state"):
        key = "q:" + spec["name"] if "name" in spec else "d:" + str(spec.get("path", ""))
        op, want = spec.get("op", "eq" if "equals" not in spec else "eq"), spec.get("value", spec.get("equals"))
        now = state_leaves(observations).get(key, _MISSING)
        if op == "exists":
            return now is not _MISSING if observations else None
        if now is _MISSING:
            return None
        if op in ("delta_eq", "delta_ge"):
            if before is None:
                return None
            was = state_leaves(before).get(key, _MISSING)
            if was is _MISSING or not isinstance(now, (int, float)) or not isinstance(was, (int, float)):
                return None
            d = float(now) - float(was)
            return abs(d - float(want)) <= 1e-9 * max(1.0, abs(float(want))) if op == "delta_eq" else d >= float(want)
        try:
            return bool(_OPS[op](now, want))
        except (KeyError, TypeError):
            return None
    if kind == "program":
        check = spec.get("check")
        if check == "steps_ok":
            return info.get("steps_ok")
        if check == "tools":
            have = info.get("operations")
            return None if have is None else set(spec.get("operations", [])) <= set(have)
    return None


def protected_reasons(step: ActionSpec) -> List[str]:
    """Why a step may not be part of a skill ([] = it may). Skill generation cannot touch the authorization
    system, the credential store or the independent verifier."""
    why = []
    op = step.operation.lower()
    if op.startswith(PROTECTED_OPERATION_PREFIXES):
        why.append(f"operation {step.operation!r} belongs to a protected service")
    if step.auth_scope and step.auth_scope.lower().startswith(PROTECTED_SCOPE_PREFIXES):
        why.append(f"scope {step.auth_scope!r} is a protected scope")
    blob = json.dumps([step.target, step.arguments], default=str).lower()
    for m in PROTECTED_PATH_MARKERS:
        if m.lower() in blob:
            why.append(f"step addresses a protected resource ({m})")
            break
    for sub in step.steps:
        why.extend(protected_reasons(sub))
    return why


# --------------------------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------------------------
@dataclass
class Skill(Record):
    skill_id: str
    name: str = ""
    description: str = ""
    version: int = 1
    status: str = STATUS_CANDIDATE
    steps: List[ActionSpec] = field(default_factory=list)          # templates: arguments may hold $param / $tpl / $capture / $lookup
    arg_schema: Dict[str, Any] = field(default_factory=dict)       # JSON Schema of the call arguments
    preconditions: List[VerificationCondition] = field(default_factory=list)
    postconditions: List[VerificationCondition] = field(default_factory=list)
    probe: Optional[ActionSpec] = None                             # read-only step that observes the state the conditions refer to
    required_scopes: List[str] = field(default_factory=list)       # exactly the union of the steps' scopes
    expected_cost: Cost = field(default_factory=Cost)
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    alpha: float = 1.0                                             # Beta posterior of reliability
    beta: float = 1.0
    runs: int = 0
    successes: int = 0
    interface_failures: int = 0                                    # consecutive attributable failures with a changed interface
    step_fingerprints: List[str] = field(default_factory=list)     # structure hash of each step's observation when validated
    failure_conditions: List[Dict[str, Any]] = field(default_factory=list)
    recovery: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)       # which successful traces it was abstracted from
    created_at: float = 0.0
    updated_at: float = 0.0
    parent_version: int = 0
    repair_attempts: int = 0
    status_reason: str = ""
    run_history: List[Dict[str, Any]] = field(default_factory=list)

    def reliability(self, level: float = 0.9) -> Dict[str, float]:
        lo, hi = beta_interval(self.alpha, self.beta, level)
        return {"mean": self.alpha / (self.alpha + self.beta), "lo": lo, "hi": hi, "level": level,
                "n": self.alpha + self.beta - 2.0}

    def operations(self) -> List[str]:
        return [s.operation for s in self.steps]


@dataclass
class TraceStep:
    action: ActionSpec
    result: ActionResult


@dataclass
class Trace:
    """One successful execution episode: the primitive actions issued and what each returned, the named inputs the
    actor started from, and the world state observed before and after."""
    steps: List[TraceStep]
    inputs: Dict[str, Any] = field(default_factory=dict)
    before: List[Observation] = field(default_factory=list)
    after: List[Observation] = field(default_factory=list)
    trace_id: str = ""

    @classmethod
    def from_pairs(cls, pairs: Sequence[Tuple[ActionSpec, ActionResult]], inputs: Optional[Dict[str, Any]] = None,
                   before: Optional[Sequence[Observation]] = None, after: Optional[Sequence[Observation]] = None,
                   trace_id: str = "") -> "Trace":
        return cls([TraceStep(a, r) for a, r in pairs], dict(inputs or {}), list(before or []), list(after or []),
                   trace_id or new_id("trace"))

    @classmethod
    def from_result(cls, action: ActionSpec, result: ActionResult, inputs: Optional[Dict[str, Any]] = None,
                    before: Optional[Sequence[Observation]] = None, after: Optional[Sequence[Observation]] = None) -> "Trace":
        """A trace from one ``ActionResult``: each ``PrimitiveOp`` (operation name = tool name) with its arguments
        and the observation it produced becomes a step."""
        by_id = {o.obs_id: o for o in result.observations}
        steps = []
        for i, op in enumerate(result.ops):
            a = ActionSpec(action_id=f"{action.action_id}.op{i}", operation=op.op, modality=op.modality,
                           arguments=dict(op.arguments), auth_scope=action.auth_scope if not action.steps else "",
                           timeout_s=action.timeout_s)
            if i < len(action.steps):
                a.auth_scope, a.target, a.verify = action.steps[i].auth_scope, action.steps[i].target, action.steps[i].verify
            obs = [by_id[op.obs_after]] if op.obs_after in by_id else []
            steps.append(TraceStep(a, ActionResult(action_id=a.action_id, status=op.status, failure_class=op.failure_class,
                                                   started=op.started, ended=op.ended, observations=obs, ops=[op],
                                                   cost=Cost(seconds=max(0.0, op.ended - op.started), primitive_ops=1))))
        return cls(steps, dict(inputs or action.arguments), list(before or []), list(after or result.observations[-1:]),
                   new_id("trace"))


# --------------------------------------------------------------------------------------------------------------
# Induction
# --------------------------------------------------------------------------------------------------------------
def _signature(a: ActionSpec) -> Tuple:
    return (a.operation, a.modality, tuple(sorted(a.arguments)))


def _common_segment(seqs: List[List[Tuple]]) -> Tuple[int, List[int]]:
    """Longest run of signatures that occurs contiguously in every sequence -> (length, start offset per sequence)."""
    first = seqs[0]
    for length in range(len(first), 0, -1):
        for s0 in range(0, len(first) - length + 1):
            window, starts = first[s0:s0 + length], [s0]
            for other in seqs[1:]:
                pos = next((i for i in range(0, len(other) - length + 1) if other[i:i + length] == window), -1)
                if pos < 0:
                    break
                starts.append(pos)
            if len(starts) == len(seqs):
                return length, starts
    return 0, []


def _json_type(values: Sequence[Any]) -> Dict[str, Any]:
    kinds = set()
    for v in values:
        kinds.add("boolean" if isinstance(v, bool) else "integer" if isinstance(v, int) else
                  "number" if isinstance(v, float) else "string" if isinstance(v, str) else
                  "array" if isinstance(v, list) else "object" if isinstance(v, dict) else "null")
    if kinds == {"integer", "number"}:
        kinds = {"number"}
    return {"type": kinds.pop()} if len(kinds) == 1 else {}


_TOKEN = re.compile(r"([/?&=#]+)")          # address-like strings are split here; free text never is


class _Abstraction:
    """State of one induction: the aligned steps of every trace and the parameters found so far."""

    def __init__(self, segs: List[List[TraceStep]], inputs: List[Dict[str, Any]]):
        self.T = len(segs)
        self.inputs = inputs
        self.data = [[_obs_data(s.result.observations) for s in seg] for seg in segs]
        self.leaves = [[_flatten(d) for d in row] for row in self.data]
        self.params: Dict[str, Tuple] = {}            # name -> value in each trace
        self.where: Dict[str, str] = {}               # name -> where it was first used

    # -- parameters ------------------------------------------------------------------------------------------
    def _unique(self, base: str) -> str:
        base = re.sub(r"[^A-Za-z0-9_]", "_", base).strip("_") or "arg"
        name, i = base, 2
        while name in self.params:
            name, i = f"{base}_{i}", i + 1
        return name

    def param_for(self, values: Sequence[Any]) -> Optional[str]:
        for k in self.inputs[0]:
            if all(k in self.inputs[t] and _same(self.inputs[t][k], values[t]) for t in range(self.T)):
                if k not in self.params:
                    self.params[k] = tuple(self.inputs[t][k] for t in range(self.T))
                return k
        for name, vals in self.params.items():
            if all(_same(vals[t], values[t]) for t in range(self.T)):
                return name
        return None

    def new_param(self, values: Sequence[Any], hint: str, where: str) -> str:
        name = self._unique(hint)
        self.params[name], self.where[name] = tuple(values), where
        return name

    # -- values produced by earlier steps --------------------------------------------------------------------
    def capture(self, values: Sequence[Any], j: int) -> Optional[Dict[str, Any]]:
        for i in range(j - 1, -1, -1):
            for path, v in sorted(self.leaves[0][i].items(), key=lambda kv: len(kv[0])):
                if not path or path[-1] == "#" or path[0] in _TRANSPORT_KEYS or not _same(v, values[0]):
                    continue
                if all(_same(self.leaves[t][i].get(path, _MISSING), values[t]) for t in range(1, self.T)):
                    cast = "" if type(v) is type(values[0]) else ("str" if isinstance(values[0], str) else "int")
                    out = {"step": i, "path": list(path)}
                    if cast:
                        out["cast"] = cast
                    return {"$capture": out}
        return None

    def lookup(self, values: Sequence[Any], j: int) -> Optional[Dict[str, Any]]:
        for i in range(j - 1, -1, -1):
            list_paths = [p[:-1] for p in self.leaves[0][i] if p and p[-1] == "#"]
            for lp in sorted(list_paths, key=len):
                try:
                    recs = [_get(self.data[t][i], lp) for t in range(self.T)]
                except KeyError:
                    continue
                if not all(isinstance(r, list) and r and all(isinstance(x, dict) for x in r) for r in recs):
                    continue
                for f in recs[0][0]:
                    hits = [[r for r in recs[t] if _same(r.get(f, _MISSING), values[t])] for t in range(self.T)]
                    if any(len(h) != 1 for h in hits):
                        continue
                    for g in recs[0][0]:
                        gv = [hits[t][0].get(g, _MISSING) for t in range(self.T)]
                        if g == f or any(v is _MISSING or isinstance(v, (dict, list)) for v in gv):
                            continue
                        if self.T >= 2 and all(_same(gv[0], v) for v in gv[1:]):
                            tmpl: Any = gv[0]
                        else:
                            name = self.param_for(gv)
                            if name is None:
                                continue
                            tmpl = {"$param": name}
                        if all(sum(1 for r in recs[t] if _same(r.get(g, _MISSING), gv[t])) == 1 for t in range(self.T)):
                            out = {"step": i, "list": list(lp), "where": {g: tmpl}, "field": f}
                            if not isinstance(hits[0][0][f], type(values[0])):
                                out["cast"] = "str" if isinstance(values[0], str) else "int"
                            return {"$lookup": out}
        return None

    # -- one leaf --------------------------------------------------------------------------------------------
    def _atom(self, values: Sequence[Any], j: int, single_ok: bool) -> Optional[Any]:
        name = self.param_for(values)
        if name is not None:
            return {"$param": name}
        v0 = values[0]
        if self.T == 1 and not single_ok:
            return None
        if self.T == 1 and (isinstance(v0, bool) or v0 is None or v0 in (0, 1, "") or
                            (isinstance(v0, str) and len(v0) < 2 and not v0.isdigit())):
            return None                       # with one trace, trivial values are not evidence of a dependency
        return self.capture(values, j) or self.lookup(values, j)

    def abstract(self, values: Sequence[Any], j: int, hint: str, where: str) -> Any:
        varies = any(not (_same(values[0], v) and type(values[0]) is type(v)) for v in values[1:])
        if self.T >= 2 and not varies:
            return values[0]
        if isinstance(values[0], (dict, list)):
            name = self.param_for(values)
            return {"$param": name or self.new_param(values, hint, where)} if (varies or name) else values[0]
        atom = self._atom(values, j, single_ok=True)
        if atom is not None:
            return atom
        if all(isinstance(v, str) and re.search(r"\s", v) is None for v in values):
            toks = [_TOKEN.split(v) for v in values]
            if len(toks[0]) > 1 and all(len(t) == len(toks[0]) for t in toks):
                parts, dynamic, constant = [], False, False
                for k in range(len(toks[0])):
                    col = [t[k] for t in toks]
                    if col[0] == "" and all(c == "" for c in col):
                        continue
                    piece_varies = any(c != col[0] for c in col[1:])
                    sub = None
                    if _TOKEN.fullmatch(col[0]) is None and (piece_varies or self.T == 1):
                        sub = self._atom(col, j, single_ok=True)
                        if sub is None and piece_varies:
                            sub = {"$param": self.new_param(col, hint, where)}
                    if sub is None:
                        if piece_varies:
                            parts = []
                            break
                        constant = constant or _TOKEN.fullmatch(col[0]) is None
                        if parts and isinstance(parts[-1], str):
                            parts[-1] += col[0]
                        else:
                            parts.append(col[0])
                    else:
                        dynamic = True
                        parts.append(sub)
                if parts and dynamic and (constant or len(parts) > 1):
                    return {"$tpl": parts}
        if self.T == 1:
            return values[0]
        return {"$param": self.new_param(values, hint, where)}


class SkillInducer:
    """Turns successful traces into a candidate :class:`Skill` (status ``candidate``; not yet usable)."""

    def __init__(self, clock: Clock, tools: Sequence[ToolDescriptor] = (), *, min_steps: int = 2, max_steps: int = 24):
        self.clock, self.min_steps, self.max_steps = clock, min_steps, max_steps
        self.tools = {t.name: t for t in tools}

    def set_tools(self, tools: Sequence[ToolDescriptor]) -> None:
        self.tools = {t.name: t for t in tools}

    def induce(self, traces: Sequence[Trace], *, name: str = "", description: str = "",
               probe: Optional[ActionSpec] = None, skill_id: str = "") -> Skill:
        traces = list(traces)
        if not traces:
            raise SkillInductionError("no traces")
        clean = [[s for s in t.steps if s.result.status == ResultStatus.OK and s.result.verified is not False]
                 for t in traces]
        for seg in clean:
            for s in seg:
                why = protected_reasons(s.action)
                if why:
                    raise SkillInductionError("protected: " + "; ".join(why))
                if any(str(key).startswith("$") for path in _arg_leaves(s.action.arguments) for key in path):
                    raise SkillInductionError("argument keys starting with '$' cannot be abstracted")
        if probe is not None and protected_reasons(probe):
            raise SkillInductionError("protected: " + "; ".join(protected_reasons(probe)))
        if len(clean) == 1:
            length, starts = len(clean[0]), [0]
        else:
            length, starts = _common_segment([[_signature(s.action) for s in seg] for seg in clean])
        if length < self.min_steps:
            raise SkillInductionError(f"no reusable segment of at least {self.min_steps} successful steps "
                                      f"common to the {len(clean)} trace(s)")
        length = min(length, self.max_steps)
        segs = [seg[st:st + length] for seg, st in zip(clean, starts)]
        ab = _Abstraction(segs, [t.inputs for t in traces])
        skill_id = skill_id or new_id("skill")
        steps: List[ActionSpec] = []
        fingerprints: List[str] = []
        for j in range(length):
            src = [seg[j].action for seg in segs]
            tree: Dict[str, Any] = {}
            paths = sorted({p for a in src for p in _arg_leaves(a.arguments)})
            for path in paths:
                if not path:
                    continue
                vals = [_arg_leaves(a.arguments).get(path, _MISSING) for a in src]
                if any(v is _MISSING for v in vals):
                    raise SkillInductionError(f"step {j}: argument {'.'.join(path)} is not present in every trace")
                _set_leaf(tree, path, ab.abstract(vals, j, path[-1], f"step {j} ({src[0].operation}) {'.'.join(path)}"))
            target = ab.abstract([a.target for a in src], j, "target", f"step {j} target") if src[0].target else ""
            tool = self.tools.get(src[0].operation)
            scope = src[0].auth_scope or (tool.auth_scope if tool else "")
            verify = src[0].verify if all(a.verify == src[0].verify for a in src) else VerificationCondition()
            steps.append(ActionSpec(action_id=f"{skill_id}.s{j}", operation=src[0].operation, modality=src[0].modality,
                                    level=Level.PRIMITIVE, target=target if isinstance(target, str) else "",
                                    arguments=tree, auth_scope=scope, timeout_s=max(a.timeout_s for a in src),
                                    verify=verify, provenance="skill_library"))
            hashes = {structure_hash(_obs_data(seg[j].result.observations)) for seg in segs}
            fingerprints.append(hashes.pop() if len(hashes) == 1 else "")
        bindings = [{n: vals[t] for n, vals in ab.params.items()} for t in range(len(traces))]
        props = {}
        for n, vals in ab.params.items():
            props[n] = dict(_json_type(vals), description=ab.where.get(n, "named input of the source trace"),
                            examples=[v for v in vals[:3]])
        schema = {"type": "object", "properties": props, "required": sorted(props), "additionalProperties": False}
        pre, post = self._conditions(traces, segs, bindings, steps)
        scopes = sorted({s.auth_scope for s in steps + ([probe] if probe else []) if s.auth_scope})
        n = float(len(segs))
        cost = Cost(seconds=sum(s.result.cost.seconds or max(0.0, s.result.ended - s.result.started)
                                for seg in segs for s in seg) / n,
                    money=sum(s.result.cost.money for seg in segs for s in seg) / n,
                    compute_s=sum(s.result.cost.compute_s for seg in segs for s in seg) / n,
                    primitive_ops=int(round(sum(max(1, len(s.result.ops)) for seg in segs for s in seg) / n)),
                    tokens=int(sum(s.result.cost.tokens for seg in segs for s in seg) / n))
        now = self.clock.now()
        ops = " -> ".join(s.operation for s in steps)
        return Skill(
            skill_id=skill_id, name=name or f"procedure of {length} steps",
            description=description or f"Parameterised procedure abstracted from {len(traces)} successful trace(s): {ops}.",
            steps=steps, arg_schema=schema, preconditions=pre, postconditions=post, probe=probe,
            required_scopes=scopes, expected_cost=cost, step_fingerprints=fingerprints,
            failure_conditions=[
                {"when": "step_failed", "meaning": "a step returned a status other than ok; execution stops there"},
                {"when": "postcondition_false", "meaning": "the resulting state does not satisfy a postcondition"},
                {"when": "binding_failed", "meaning": "an earlier step no longer yields a value a later step needs"},
                {"when": "interface_changed", "meaning": "the structure of a step's observation differs from the validated one"}],
            recovery=[
                {"on": "stale", "action": "repair", "how": "re-induce from new successful traces, bump the version, validate again"},
                {"on": "repair_failed", "action": "retire", "how": "the skill is kept on disk as retired and never executed"}],
            provenance={"method": "aligned_traces" if len(traces) > 1 else "single_trace_inputs",
                        "trace_ids": [t.trace_id for t in traces],
                        "action_ids": [[s.action.action_id for s in seg] for seg in segs],
                        "segment_offsets": starts, "source_args_sha256": [_args_sha(b) for b in bindings],
                        "induced_at": now},
            created_at=now, updated_at=now)

    # ---- conditions from the observed before / after states --------------------------------------------------
    def _conditions(self, traces: List[Trace], segs: List[List[TraceStep]], bindings: List[Dict[str, Any]],
                    steps: List[ActionSpec]) -> Tuple[List[VerificationCondition], List[VerificationCondition]]:
        T = len(traces)
        before = [t.before for t in traces]
        after = [t.after or seg[-1].result.observations for t, seg in zip(traces, segs)]
        B, A = [state_leaves(b) for b in before], [state_leaves(a) for a in after]
        pre = [VerificationCondition(description="the operations the procedure uses are available", kind="program",
                                     spec={"check": "tools", "operations": sorted({s.operation for s in steps})})]
        post: List[VerificationCondition] = []
        urls = [next((str(o.data.get("url") or o.provenance.uri) for o in b if o.data.get("url") or o.provenance.uri), "")
                for b in before]
        if all(urls):
            toks = [_TOKEN.split(u.split("?")[0]) for u in urls]
            if all(len(t) == len(toks[0]) for t in toks):
                pat = "".join(re.escape(toks[0][k]) if all(t[k] == toks[0][k] for t in toks) else r"[^/?#&=]+"
                              for k in range(len(toks[0])))
                pre.append(VerificationCondition(description="the starting location has the observed form",
                                                 kind="url_matches", spec={"pattern": "^" + pat}))
        observed_before = all(bool(b) for b in before)
        changed = [k for k in A[0] if all(k in A[t] for t in range(T)) and not _VOLATILE.search(k)
                   and all(k not in B[t] or not _same(B[t][k], A[t][k]) for t in range(T))]
        for k in sorted(changed):
            have_before = all(k in B[t] for t in range(T))
            positional = "/[" in k or k.startswith("d:[")
            numeric = all(isinstance(A[t][k], (int, float)) and not isinstance(A[t][k], bool) for t in range(T))
            target = {"name": k[2:]} if k.startswith("q:") else {"path": k[2:]}
            pname = next((n for n in bindings[0] if all(_same(bindings[t].get(n, _MISSING), A[t][k]) for t in range(T))
                          and not isinstance(bindings[0][n], (dict, list, bool))), None)
            if pname is not None and not positional:
                post.append(VerificationCondition(description=f"{k[2:]} equals the argument {pname}", kind="metric",
                                                  spec=dict(target, op="eq", value={"$param": pname})))
            elif numeric and have_before and all(isinstance(B[t][k], (int, float)) for t in range(T)):
                deltas = [float(A[t][k]) - float(B[t][k]) for t in range(T)]
                if all(abs(d - deltas[0]) < 1e-9 for d in deltas) and float(deltas[0]).is_integer():
                    post.append(VerificationCondition(description=f"{k[2:]} changes by {deltas[0]:+g}", kind="metric",
                                                      spec=dict(target, op="delta_eq", value=deltas[0])))
            elif (T >= 2 and observed_before and not positional and not isinstance(A[0][k], float)
                  and all(_same(A[0][k], A[t][k]) for t in range(1, T))):
                post.append(VerificationCondition(description=f"{k[2:]} becomes {A[0][k]!r}", kind="metric",
                                                  spec=dict(target, op="eq", value=A[0][k])))
                if have_before and all(_same(B[0][k], B[t][k]) for t in range(1, T)):
                    pre.append(VerificationCondition(description=f"{k[2:]} is {B[0][k]!r} beforehand", kind="metric",
                                                     spec=dict(target, op="eq", value=B[0][k])))
        texts_b, texts_a = [_obs_text(b) for b in before], [_obs_text(a) for a in after]
        for n in sorted(bindings[0]):
            vals = [bindings[t].get(n) for t in range(T)]
            if all(isinstance(v, (str, int)) and not isinstance(v, bool) and len(str(v)) >= 3 for v in vals) and \
                    all(str(vals[t]) in texts_a[t] and (not before[t] or str(vals[t]) not in texts_b[t]) for t in range(T)):
                post.append(VerificationCondition(description=f"the argument {n} appears in the resulting state",
                                                  kind="text_present", spec={"text": {"$param": n}}))
        post = post[:12]
        post.append(VerificationCondition(description="every step reported success", kind="program",
                                          spec={"check": "steps_ok"}))
        return pre[:8], post


def _args_sha(args: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()[:16]


def fresh_arguments(skill: Skill, rng: random.Random) -> Dict[str, Any]:
    """Arguments that differ from every example the skill was induced from (types from the schema)."""
    out: Dict[str, Any] = {}
    for name, sch in skill.arg_schema.get("properties", {}).items():
        ex = (sch.get("examples") or [None])[0]
        t = sch.get("type")
        if t == "string":
            out[name] = f"{str(ex or name)[:24]}-{rng.getrandbits(32):08x}"
        elif t == "integer":
            out[name] = int(ex or 0) + rng.randint(1000, 9999)
        elif t == "number":
            out[name] = float(ex or 0.0) + rng.uniform(1.0, 9.0)
        elif t == "boolean":
            out[name] = not bool(ex)
        else:
            out[name] = ex
    return out


# --------------------------------------------------------------------------------------------------------------
# Library (persistent, versioned, monitored)
# --------------------------------------------------------------------------------------------------------------
def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class SkillLibrary:
    """Versioned skill storage under ``<root>/skills`` and the monitoring rules that decide staleness."""

    def __init__(self, root: str, clock: Clock, *, min_reliability: float = 0.5, stale_confidence: float = 0.9,
                 stale_after: int = 2, max_repairs: int = 2, min_validations: int = 1, forgetting: float = 0.98):
        self.dir = os.path.join(root, "skills")
        os.makedirs(self.dir, exist_ok=True)
        self.clock = clock
        self.min_reliability, self.stale_confidence = min_reliability, stale_confidence
        self.stale_after, self.max_repairs, self.min_validations = stale_after, max_repairs, min_validations
        self.forgetting = forgetting          # older evidence about reliability is discounted: interfaces change
        self._skills: Dict[str, Skill] = {}
        self._index_text = ""
        self._load()

    # ---- storage -------------------------------------------------------------------------------------------
    def _index_path(self) -> str:
        return os.path.join(self.dir, "index.json")

    def _load(self) -> None:
        if not os.path.exists(self._index_path()):
            return
        with open(self._index_path(), encoding="utf-8") as f:
            index = json.load(f)
        for sid, meta in index.items():
            path = os.path.join(self.dir, sid, f"v{int(meta['version'])}.json")
            with open(path, encoding="utf-8") as f:
                self._skills[sid] = Skill.from_dict(json.load(f))

    def _save(self, skill: Skill) -> None:
        skill.updated_at = self.clock.now()
        d = os.path.join(self.dir, skill.skill_id)
        os.makedirs(d, exist_ok=True)
        _atomic_write(os.path.join(d, f"v{skill.version}.json"), skill.to_json())
        self._skills[skill.skill_id] = skill
        index = json.dumps({sid: {"version": s.version, "status": s.status, "name": s.name}
                            for sid, s in self._skills.items()}, sort_keys=True, indent=1)
        if index != self._index_text:                    # the index changes only with a version or a status
            _atomic_write(self._index_path(), index)
            self._index_text = index

    def log_rejected(self, skill: Skill, report: Dict[str, Any]) -> None:
        with open(os.path.join(self.dir, "rejected.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": self.clock.now(), "skill": skill.to_dict(), "report": report}, default=str) + "\n")

    # ---- access --------------------------------------------------------------------------------------------
    def get(self, skill_id: str, version: Optional[int] = None) -> Optional[Skill]:
        cur = self._skills.get(skill_id)
        if version is None or (cur is not None and cur.version == version):
            return cur
        path = os.path.join(self.dir, skill_id, f"v{int(version)}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return Skill.from_dict(json.load(f))

    def versions(self, skill_id: str) -> List[int]:
        d = os.path.join(self.dir, skill_id)
        if not os.path.isdir(d):
            return []
        return sorted(int(n[1:-5]) for n in os.listdir(d) if re.fullmatch(r"v\d+\.json", n))

    def list(self, status: Optional[str] = None) -> List[Skill]:
        return [s for s in sorted(self._skills.values(), key=lambda s: s.created_at) if status in (None, s.status)]

    def candidate_actions(self) -> List[ActionSpec]:
        """One ``skill.run`` action template per usable skill (arguments to be filled by the proposer)."""
        return [ActionSpec(action_id=new_id("act"), operation="skill.run", modality=Modality.SKILL,
                           level=Level.PROCEDURE, target=s.skill_id,
                           arguments={"skill_id": s.skill_id, "arguments": {}}, expected=s.description,
                           est_cost=s.expected_cost, auth_scope=" ".join(s.required_scopes), provenance="skill_library")
                for s in self.list(STATUS_VALIDATED)]

    # ---- lifecycle -----------------------------------------------------------------------------------------
    def register(self, skill: Skill) -> Skill:
        """Store a skill that has passed validation. Anything else is refused."""
        ok = [h for h in skill.validation_history if h.get("ok") and h.get("fresh") and h.get("version") == skill.version]
        if skill.status != STATUS_VALIDATED or len(ok) < self.min_validations:
            raise ValueError("only a skill validated by execution with fresh arguments can be registered")
        cur = self._skills.get(skill.skill_id)
        if cur is not None and cur.version >= skill.version:
            raise ValueError(f"version {skill.version} is not newer than the stored version {cur.version}")
        self._save(skill)
        return skill

    def record_run(self, skill_id: str, *, ok: bool, failure_class: str = FailureClass.NONE,
                   termination: str = "", interface_changed: bool = False, detail: str = "",
                   n_ops: int = 0, elapsed_s: float = 0.0) -> Skill:
        """Update monitoring after a run. Denials, platform restrictions, resource limits and transient network
        faults are logged but say nothing about the skill; every other failure counts against its reliability."""
        s = self._skills[skill_id]
        attributable = ok or failure_class in ATTRIBUTABLE
        s.runs += 1
        if attributable:
            s.alpha = 1.0 + self.forgetting * (s.alpha - 1.0)
            s.beta = 1.0 + self.forgetting * (s.beta - 1.0)
        if ok:
            s.successes += 1
            s.alpha += 1.0
            s.interface_failures = 0
        elif attributable:
            s.beta += 1.0
            interface = (interface_changed or termination in ("postcondition_failed", "binding_failed")
                         or failure_class in (FailureClass.PAGE_CHANGED, FailureClass.WRONG_SITE_ASSUMPTION))
            s.interface_failures = s.interface_failures + 1 if interface else 0
        s.run_history = (s.run_history + [{"ts": self.clock.now(), "ok": ok, "failure_class": failure_class,
                                           "termination": termination, "interface_changed": interface_changed,
                                           "attributable": attributable, "n_ops": n_ops, "elapsed_s": elapsed_s,
                                           "detail": detail[:200]}])[-50:]
        if s.status == STATUS_VALIDATED and not ok and attributable:
            p_low = beta_cdf(self.min_reliability, s.alpha, s.beta)
            if s.interface_failures >= self.stale_after:
                s.status, s.status_reason = STATUS_STALE, (f"{s.interface_failures} consecutive failures after the "
                                                           f"interface changed ({termination})")
            elif p_low >= self.stale_confidence:
                s.status, s.status_reason = STATUS_STALE, (f"P(reliability < {self.min_reliability}) = {p_low:.3f} "
                                                           f">= {self.stale_confidence}")
        self._save(s)
        return s

    def mark_stale(self, skill_id: str, reason: str) -> Skill:
        s = self._skills[skill_id]
        if s.status != STATUS_RETIRED:
            s.status, s.status_reason = STATUS_STALE, reason
            self._save(s)
        return s

    def retire(self, skill_id: str, reason: str) -> Skill:
        s = self._skills[skill_id]
        s.status, s.status_reason = STATUS_RETIRED, reason
        self._save(s)
        return s

    def note_repair_failed(self, skill_id: str, reason: str) -> Skill:
        s = self._skills[skill_id]
        s.repair_attempts += 1
        if s.repair_attempts >= self.max_repairs:
            return self.retire(skill_id, f"unrecoverable: {s.repair_attempts} repair attempts failed ({reason})")
        s.status_reason = f"stale; repair attempt {s.repair_attempts} failed ({reason})"
        self._save(s)
        return s


# --------------------------------------------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------------------------------------------
_RUN_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["skill_id"], "properties": {
    "skill_id": {"type": "string", "minLength": 1}, "arguments": {"type": "object"}}}
_LIST_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "status": {"type": "string", "enum": [STATUS_VALIDATED, STATUS_STALE, STATUS_RETIRED]}}}
_DESCRIBE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["skill_id"], "properties": {
    "skill_id": {"type": "string", "minLength": 1}, "version": {"type": "integer", "minimum": 1}}}
_STEP_SCHEMA = {"type": "object", "required": ["operation"], "properties": {
    "operation": {"type": "string"}, "arguments": {"type": "object"}}}
_INDUCE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["traces"], "properties": {
    "traces": {"type": "array", "minItems": 1, "maxItems": 8,
               "items": {"type": "array", "minItems": 1, "items": {"type": "string"}},
               "description": "for each trace, the ids of the executed actions that form it, in order"},
    "inputs": {"type": "array", "items": {"type": "object"},
               "description": "for each trace, the named values the actions were built from"},
    "name": {"type": "string", "maxLength": 80}, "description": {"type": "string", "maxLength": 400},
    "probe": _STEP_SCHEMA,
    "validation_arguments": {"type": "array", "items": {"type": "object"}, "maxItems": 4},
    "repair_skill_id": {"type": "string"}}}


class SkillExecutor:
    """``ModalityExecutor`` for the ``skill`` modality."""

    modality = Modality.SKILL

    def __init__(self, library: SkillLibrary, executors: Sequence[Any] = (), *,
                 dispatch: Optional[Callable[[ActionSpec, ExecContext], ActionResult]] = None,
                 inducer: Optional[SkillInducer] = None, trace_buffer: int = 512, seed: int = 0):
        self.library, self.executors = library, list(executors)
        self._dispatch_fn = dispatch
        self.inducer = inducer or SkillInducer(library.clock)
        if not self.inducer.tools:
            self.inducer.set_tools([t for e in self.executors for t in e.tools()])
        self._buffer: Dict[str, Tuple[ActionSpec, ActionResult]] = {}
        self._buffer_size, self._rng = trace_buffer, random.Random(seed)

    # ---- wiring --------------------------------------------------------------------------------------------
    def bind_dispatch(self, dispatch: Callable[[ActionSpec, ExecContext], ActionResult],
                      tools: Sequence[ToolDescriptor] = ()) -> None:
        """Let the environment route skill steps through its own executors (the normal execution path)."""
        self._dispatch_fn = dispatch
        if tools:
            self.inducer.set_tools(tools)

    def observe_execution(self, action: ActionSpec, result: ActionResult) -> None:
        """Called by the environment after each executed action; successful non-skill executions become material
        for induction (referenced by action id in ``skill.induce``)."""
        if result.status == ResultStatus.OK and action.modality != Modality.SKILL:
            self._buffer[action.action_id] = (action, result)
            while len(self._buffer) > self._buffer_size:
                self._buffer.pop(next(iter(self._buffer)))

    def _operations(self) -> Optional[List[str]]:
        ops = [t.name for e in self.executors for t in e.tools() if t.available] or sorted(self.inducer.tools)
        return ops or None

    def _dispatch(self, step: ActionSpec, ctx: ExecContext) -> ActionResult:
        if self._dispatch_fn is not None:
            return self._dispatch_fn(step, ctx)
        for e in self.executors:
            if any(t.name == step.operation for t in e.tools()):
                errs = e.validate(step)
                if errs:
                    return failed_result(step, classify(FailureEvidence(arg_errors=errs, modality=step.modality)),
                                         ctx.clock, ctx.clock.now())
                return e.execute(step, ctx)
        rep = classify(FailureEvidence(target_missing=True, modality=step.modality,
                                       detail=f"no executor offers the operation {step.operation!r}"))
        return failed_result(step, rep, ctx.clock, ctx.clock.now())

    # ---- ModalityExecutor ----------------------------------------------------------------------------------
    def tools(self) -> List[ToolDescriptor]:
        return [
            ToolDescriptor(name="skill.run", modality=Modality.SKILL, level=Level.PROCEDURE, arg_schema=_RUN_SCHEMA,
                           side_effects="external_write", auth_scope="",
                           description="Run a validated skill from the library with the given arguments. The skill is "
                                       "expanded into its primitive steps; each step is authorized and executed "
                                       "individually, and the skill's postconditions are checked on the resulting state."),
            ToolDescriptor(name="skill.list", modality=Modality.SKILL, level=Level.PRIMITIVE, arg_schema=_LIST_SCHEMA,
                           side_effects="none",
                           description="List the skills in the library: identifier, description, argument schema, "
                                       "required permission scopes, expected cost, status and reliability estimate."),
            ToolDescriptor(name="skill.describe", modality=Modality.SKILL, level=Level.PRIMITIVE,
                           arg_schema=_DESCRIBE_SCHEMA, side_effects="none",
                           description="Describe one skill: its steps, arguments, preconditions, postconditions, "
                                       "validation history, reliability, version and provenance."),
            ToolDescriptor(name="skill.induce", modality=Modality.SKILL, level=Level.PROCEDURE,
                           arg_schema=_INDUCE_SCHEMA, side_effects="external_write",
                           description="Abstract a reusable skill from earlier successful executions (given by their "
                                       "action ids), execute it with fresh arguments to validate it, and add it to "
                                       "the library if validation succeeds. With repair_skill_id, replaces a stale "
                                       "skill by a new version."),
        ]

    def validate(self, action: ActionSpec) -> List[str]:
        schema = {"skill.run": _RUN_SCHEMA, "skill.list": _LIST_SCHEMA, "skill.describe": _DESCRIBE_SCHEMA,
                  "skill.induce": _INDUCE_SCHEMA}.get(action.operation)
        if schema is None:
            return [f"unknown operation {action.operation!r} for the skill executor"]
        errs = schema_errors(schema, action.arguments)
        if errs or action.operation in ("skill.list", "skill.induce"):
            return errs
        s = self.library.get(action.arguments["skill_id"], action.arguments.get("version"))
        if s is None:
            return [f"arguments.skill_id: no skill {action.arguments['skill_id']!r} in the library"]
        if action.operation == "skill.run":
            errs = schema_errors(s.arg_schema, action.arguments.get("arguments", {}), "arguments.arguments")
        return errs

    def close(self) -> None:
        return None

    def execute(self, action: ActionSpec, ctx: ExecContext) -> ActionResult:
        clock, started = ctx.clock, ctx.clock.now()
        errs = self.validate(action)
        if errs:
            return failed_result(action, classify(FailureEvidence(arg_errors=errs, modality=Modality.SKILL,
                                                                  level=action.level)), clock, started)
        if action.operation == "skill.list":
            return self._info(action, ctx, [self._summary(s) for s in self.library.list(action.arguments.get("status"))])
        if action.operation == "skill.describe":
            s = self.library.get(action.arguments["skill_id"], action.arguments.get("version"))
            d = self._summary(s)
            d.update(steps=[{"operation": x.operation, "modality": x.modality, "arguments": x.arguments,
                             "auth_scope": x.auth_scope} for x in s.steps],
                     preconditions=[c.to_dict() for c in s.preconditions],
                     postconditions=[c.to_dict() for c in s.postconditions],
                     validation_history=s.validation_history[-8:], failure_conditions=s.failure_conditions,
                     recovery=s.recovery, provenance=s.provenance, versions=self.library.versions(s.skill_id))
            return self._info(action, ctx, [d])
        if action.operation == "skill.induce":
            return self._induce_action(action, ctx)
        skill = self.library.get(action.arguments["skill_id"])
        return self.run_skill(skill, action.arguments.get("arguments", {}), ctx, action=action)

    # ---- read-only tools -----------------------------------------------------------------------------------
    def _summary(self, s: Skill) -> Dict[str, Any]:
        return {"skill_id": s.skill_id, "name": s.name, "description": s.description, "version": s.version,
                "status": s.status, "status_reason": s.status_reason, "arg_schema": s.arg_schema,
                "required_scopes": s.required_scopes, "operations": s.operations(),
                "expected_cost": s.expected_cost.to_dict(), "reliability": s.reliability(), "runs": s.runs}

    def _info(self, action: ActionSpec, ctx: ExecContext, items: List[Dict[str, Any]]) -> ActionResult:
        now = ctx.clock.now()
        q = [Quantity("skill.library.count", float(len(self.library.list())), kind=OutcomeKind.COUNT),
             Quantity("skill.library.validated", float(len(self.library.list(STATUS_VALIDATED))), kind=OutcomeKind.COUNT)]
        for it in items[:8]:
            q.append(Quantity(f"skill.{it['skill_id']}.reliability", it["reliability"]["mean"]))
        text = "; ".join(f"{it['skill_id']} v{it['version']} [{it['status']}] {it['name']}: {it['description']}"
                         for it in items)[:4000] or "the skill library is empty"
        obs = Observation(obs_id=new_id("obs"), ts=now, kind="tool_result", text=text, quantities=q,
                          provenance=Provenance(source="skill_library", retrieved_at=now, trust=Trust.SELF),
                          parent_action_id=action.action_id, data={"skills": items})
        return ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=now, ended=ctx.clock.now(),
                            observations=[obs], verified=True)

    # ---- running a skill -----------------------------------------------------------------------------------
    def run_skill(self, skill: Skill, args: Dict[str, Any], ctx: ExecContext, *, action: Optional[ActionSpec] = None,
                  validating: bool = False) -> ActionResult:
        clock, started, t_wall = ctx.clock, ctx.clock.now(), _time.perf_counter()
        action = action or ActionSpec(action_id=new_id("act"), operation="skill.run", modality=Modality.SKILL,
                                      level=Level.PROCEDURE, target=skill.skill_id,
                                      arguments={"skill_id": skill.skill_id, "arguments": args},
                                      auth_scope=" ".join(skill.required_scopes), provenance="skill_library")
        ops: List[PrimitiveOp] = []
        observations: List[Observation] = []
        step_log: List[Dict[str, Any]] = []
        state = {"termination": "", "interface_changed": False, "start_obs": "", "end_obs": "", "post": []}

        def finish(base: Optional[ActionResult], status: str, failure_class: str, detail: str, verified: Optional[bool]) -> ActionResult:
            now, elapsed = clock.now(), _time.perf_counter() - t_wall
            rel = skill.reliability()
            done = sum(1 for s in step_log if s["status"] == ResultStatus.OK and not s["step"].startswith("probe"))
            option = {"level": Level.PROCEDURE, "skill_id": skill.skill_id, "version": skill.version,
                      "start_obs": state["start_obs"], "end_obs": state["end_obs"], "termination": state["termination"],
                      "n_primitive_ops": len(ops), "elapsed_s": max(now - started, elapsed), "steps": step_log,
                      "arguments_sha256": _args_sha(args)}
            summary = Observation(
                obs_id=new_id("obs"), ts=now, kind="tool_result",
                text=(f"skill {skill.skill_id} v{skill.version} ({skill.name}): {state['termination']}; "
                      f"{done}/{len(skill.steps)} steps, {len(ops)} primitive operations"),
                quantities=[Quantity("skill.primitive_ops", float(len(ops)), kind=OutcomeKind.COUNT),
                            Quantity("skill.steps_completed", float(done), kind=OutcomeKind.COUNT),
                            Quantity("skill.elapsed_s", option["elapsed_s"], unit="s"),
                            Quantity("skill.reliability.mean", rel["mean"]), Quantity("skill.reliability.lo", rel["lo"]),
                            Quantity("skill.reliability.hi", rel["hi"])],
                provenance=Provenance(source="skill_library", uri=skill.skill_id, retrieved_at=now, trust=Trust.SELF),
                parent_action_id=action.action_id,
                data={"option": option, "postconditions": state["post"], "interface_changed": state["interface_changed"]})
            res = base or ActionResult(action_id=action.action_id, status=status)
            res.action_id, res.status, res.failure_class, res.failure_detail = action.action_id, status, failure_class, detail
            res.started, res.ended, res.ops, res.verified = started, now, ops, verified
            res.observations = [summary] + observations[-16:] + [o for o in res.observations if o.kind == "error"]
            res.cost = Cost(seconds=option["elapsed_s"], primitive_ops=len(ops),
                            compute_s=sum(s.get("compute_s", 0.0) for s in step_log),
                            money=sum(s.get("money", 0.0) for s in step_log))
            res.artifacts.update(skill_id=skill.skill_id, skill_version=str(skill.version), termination=state["termination"])
            if not validating and skill.status == STATUS_VALIDATED and state["termination"] not in (
                    "refused_stale", "invalid_arguments", "precondition_failed", "protected"):
                self.library.record_run(skill.skill_id, ok=status == ResultStatus.OK, failure_class=failure_class,
                                        termination=state["termination"], interface_changed=state["interface_changed"],
                                        detail=detail, n_ops=len(ops), elapsed_s=option["elapsed_s"])
            return res

        def fail(ev: FailureEvidence, termination: str, base_obs: Optional[List[Observation]] = None) -> ActionResult:
            state["termination"] = termination
            ev.modality, ev.level = Modality.SKILL, Level.PROCEDURE
            rep = classify(ev)
            return finish(failed_result(action, rep, clock, started, observations=base_obs),
                          ResultStatus.DENIED if rep.hard else ResultStatus.FAILED,
                          rep.failure_class, rep.detail, False if ev.verified is False else None)

        # 1. only validated skills run (a candidate runs only inside validation); stale / retired never
        allowed = (STATUS_VALIDATED,) + ((STATUS_CANDIDATE,) if validating else ())
        if skill.status not in allowed:
            changed = "interface" in skill.status_reason or "postcondition" in skill.status_reason
            return fail(FailureEvidence(target_missing=True, structure_hash_expected="validated" if changed else "",
                                        structure_hash_actual="changed" if changed else "",
                                        detail=f"skill {skill.skill_id} v{skill.version} is {skill.status} and was not "
                                               f"executed: {skill.status_reason}"), "refused_stale")
        errs = schema_errors(skill.arg_schema, args, "arguments")
        if errs:
            return fail(FailureEvidence(arg_errors=errs), "invalid_arguments")
        for st in skill.steps + ([skill.probe] if skill.probe else []):
            why = protected_reasons(st)
            if why:
                return fail(FailureEvidence(exception=PermissionError("; ".join(why))), "protected")

        def run_step(tmpl: ActionSpec, label: str, data: List[Any]) -> Tuple[Optional[ActionResult], Optional[ActionResult]]:
            """-> (step result, terminal result). Every step is checked, then executed by its own modality."""
            try:
                bound = bind(tmpl.arguments, args, data)
                target = bind(tmpl.target, args, data) if tmpl.target else ""
            except SkillBindError as e:
                state["interface_changed"] = True
                return None, fail(FailureEvidence(target_missing=True, structure_hash_expected="validated",
                                                  structure_hash_actual="changed", detail=f"{label}: {e}"), "binding_failed")
            step = ActionSpec(action_id=f"{action.action_id}.{label}", operation=tmpl.operation, modality=tmpl.modality,
                              level=Level.PRIMITIVE, target=target, arguments=bound, auth_scope=tmpl.auth_scope,
                              timeout_s=tmpl.timeout_s, verify=tmpl.verify, parent_decision_id=action.parent_decision_id,
                              experiment_id=action.experiment_id, provenance="skill_library")
            op = PrimitiveOp(op=step.operation, modality=step.modality, arguments=bound, started=clock.now(),
                             note=f"skill:{skill.skill_id}:v{skill.version}:{label}")
            refused = authorization_result(action, ctx.check(op, step), clock, started, ops=[op])
            if refused is not None:
                ops.append(op)
                state["termination"] = "denied" if refused.status == ResultStatus.DENIED else "needs_approval"
                return None, finish(refused, refused.status, refused.failure_class, refused.failure_detail, None)
            sub = self._dispatch(step, ctx)
            if sub.ops:
                ops.extend(sub.ops)
            else:
                op.ended, op.status, op.failure_class = clock.now(), sub.status, sub.failure_class
                ops.append(op)
            step_log.append({"step": label, "operation": step.operation, "status": sub.status,
                             "failure_class": sub.failure_class, "seconds": sub.cost.seconds or (sub.ended - sub.started),
                             "compute_s": sub.cost.compute_s, "money": sub.cost.money})
            if sub.status in (ResultStatus.DENIED, ResultStatus.NEEDS_APPROVAL):
                state["termination"] = "denied" if sub.status == ResultStatus.DENIED else "needs_approval"
                observations.extend(sub.observations)
                return None, finish(None, sub.status, sub.failure_class, sub.failure_detail, None)
            return sub, None

        # 2. observe the start state and check the preconditions
        info = {"operations": self._operations(), "steps_ok": None}
        before: Optional[List[Observation]] = None
        if skill.probe is not None:
            sub, terminal = run_step(skill.probe, "probe0", [])
            if terminal is not None:
                return terminal
            before = [o for o in sub.observations if o.kind != "error"]
            state["start_obs"] = before[-1].obs_id if before else ""
            if sub.status != ResultStatus.OK:           # the fault is reported as it was classified, not re-read
                state["termination"] = "probe_failed"
                base = ActionResult(action_id=action.action_id, status=sub.status,
                                    observations=[o for o in sub.observations if o.kind == "error"])
                st = ResultStatus.TIMEOUT if sub.status == ResultStatus.TIMEOUT else ResultStatus.FAILED
                return finish(base, st, sub.failure_class if sub.failure_class != FailureClass.NONE else FailureClass.UNKNOWN,
                              f"the start state could not be observed: {sub.failure_detail}", None)
        for c in skill.preconditions:
            if evaluate_condition(c, before or [], args, info=info) is False:
                return fail(FailureEvidence(target_missing=True, detail=f"precondition not met: {c.description}"),
                            "precondition_failed")

        # 3. the steps
        data: List[Any] = []
        last: List[Observation] = []
        for j, tmpl in enumerate(skill.steps):
            if ctx.cancel is not None and ctx.cancel.is_set():
                state["termination"] = "cancelled"
                return finish(None, ResultStatus.CANCELLED, FailureClass.NONE, "cancelled", None)
            if ctx.deadline and clock.now() > ctx.deadline:
                state["termination"] = "deadline"
                rep = classify(FailureEvidence(timed_out=True, detail="deadline reached between steps"))
                return finish(failed_result(action, rep, clock, started), ResultStatus.TIMEOUT, rep.failure_class, rep.detail, None)
            sub, terminal = run_step(tmpl, f"s{j}", data)
            if terminal is not None:
                return terminal
            last = [o for o in sub.observations if o.kind != "error"]
            observations.extend(sub.observations[-2:])
            d = _obs_data(sub.observations)
            data.append(d)
            fp = skill.step_fingerprints[j] if j < len(skill.step_fingerprints) else ""
            actual = structure_hash(d)
            changed = bool(fp) and sub.status == ResultStatus.OK and actual != fp
            state["interface_changed"] = state["interface_changed"] or changed
            if sub.status != ResultStatus.OK:
                state["termination"] = f"step_failed:{j}"
                cls = sub.failure_class if sub.failure_class != FailureClass.NONE else FailureClass.UNKNOWN
                detail = f"step {j} ({tmpl.operation}) failed: {sub.failure_detail}"
                st = ResultStatus.TIMEOUT if sub.status == ResultStatus.TIMEOUT else ResultStatus.FAILED
                base = ActionResult(action_id=action.action_id, status=st,
                                    observations=[o for o in sub.observations if o.kind == "error"])
                return finish(base, st, cls, detail, False if sub.verified is False else None)

        # 4. inspect the resulting state; never assume success
        after = last
        if skill.probe is not None:
            sub, terminal = run_step(skill.probe, "probe1", [])
            if terminal is not None:
                return terminal
            if sub.status == ResultStatus.OK:
                after = [o for o in sub.observations if o.kind != "error"]
        state["end_obs"] = after[-1].obs_id if after else ""
        info["steps_ok"] = True
        results = [(c, evaluate_condition(c, after, args, before=before, info=info)) for c in skill.postconditions]
        state["post"] = [{"description": c.description, "kind": c.kind, "result": r} for c, r in results]
        failed = [c for c, r in results if r is False]
        if failed:
            return fail(FailureEvidence(verified=False, executed_ok=True,
                                        structure_hash_expected="validated" if state["interface_changed"] else "",
                                        structure_hash_actual="changed" if state["interface_changed"] else "",
                                        detail="postcondition not met: " + "; ".join(c.description for c in failed)),
                        "postcondition_failed", None)
        state["termination"] = "postconditions_met"
        checked = [r for c, r in results if r is True and c.spec.get("check") != "steps_ok"]
        return finish(None, ResultStatus.OK, FailureClass.NONE, "", True if checked else None)

    # ---- validation, registration, repair ------------------------------------------------------------------
    def validate_candidate(self, skill: Skill, ctx: ExecContext, arg_sets: Optional[Sequence[Dict[str, Any]]] = None,
                           *, environment: str = "staging") -> Dict[str, Any]:
        """Execute a candidate with fresh arguments through the normal path. Returns the validation report; the
        candidate becomes ``validated`` only if every run succeeded with its postconditions holding."""
        seen = set(skill.provenance.get("source_args_sha256", []))
        need = max(1, self.library.min_validations)
        arg_sets = list(arg_sets or [])
        if not arg_sets:
            arg_sets = [fresh_arguments(skill, self._rng) for _ in range(need)]
        runs = []
        nontrivial = [c for c in skill.postconditions if c.spec.get("check") != "steps_ok"]
        for a in arg_sets:
            fresh = _args_sha(a) not in seen or not skill.arg_schema.get("properties")
            if not fresh:
                runs.append({"ts": ctx.clock.now(), "ok": False, "fresh": False, "version": skill.version,
                             "args_sha256": _args_sha(a), "termination": "not_fresh", "failure_class": FailureClass.NONE})
                continue
            res = self.run_skill(skill, a, ctx, validating=True)
            confirmed = res.verified is True or not nontrivial
            runs.append({"ts": ctx.clock.now(), "ok": res.status == ResultStatus.OK and confirmed,
                         "fresh": True, "version": skill.version, "args_sha256": _args_sha(a),
                         "termination": res.artifacts.get("termination", ""), "failure_class": res.failure_class,
                         "detail": res.failure_detail[:300], "n_ops": len(res.ops), "elapsed_s": res.cost.seconds,
                         "verified": res.verified, "environment": environment})
        skill.validation_history.extend(runs)
        n_ok = sum(1 for r in runs if r["ok"])
        passed = n_ok >= need and all(r["ok"] for r in runs if r["fresh"])
        report = {"passed": passed, "n_ok": n_ok, "n_runs": len(runs), "runs": runs,
                  "reason": "" if passed else next((f"{r['termination']}: {r.get('detail', '')}" for r in runs if not r["ok"]),
                                                  "not enough validation runs")}
        if passed:
            skill.status, skill.status_reason = STATUS_VALIDATED, ""
            skill.alpha, skill.beta = 1.0 + n_ok, 1.0
        return report

    def induce_validate_register(self, traces: Sequence[Trace], ctx: ExecContext, *,
                                 validation_arguments: Optional[Sequence[Dict[str, Any]]] = None, name: str = "",
                                 description: str = "", probe: Optional[ActionSpec] = None,
                                 environment: str = "staging") -> Tuple[Optional[Skill], Dict[str, Any]]:
        """Induce -> validate by execution -> register. Returns (skill or None, report)."""
        try:
            skill = self.inducer.induce(traces, name=name, description=description, probe=probe)
        except SkillInductionError as e:
            return None, {"passed": False, "reason": f"induction failed: {e}", "runs": []}
        report = self.validate_candidate(skill, ctx, validation_arguments, environment=environment)
        if not report["passed"]:
            self.library.log_rejected(skill, report)
            return None, report
        self.library.register(skill)
        return skill, report

    def repair(self, skill_id: str, traces: Sequence[Trace], ctx: ExecContext, *,
               validation_arguments: Optional[Sequence[Dict[str, Any]]] = None, probe: Optional[ActionSpec] = None,
               environment: str = "staging") -> Tuple[Optional[Skill], Dict[str, Any]]:
        """Replace a stale skill by a new version induced from new successful traces. A failed repair is counted;
        after ``max_repairs`` failures the skill is retired."""
        old = self.library.get(skill_id)
        if old is None or old.status == STATUS_RETIRED:
            return None, {"passed": False, "reason": "no such repairable skill", "runs": []}
        try:
            new = self.inducer.induce(traces, name=old.name, description=old.description,
                                      probe=probe if probe is not None else old.probe, skill_id=skill_id)
        except SkillInductionError as e:
            self.library.note_repair_failed(skill_id, f"induction failed: {e}")
            return None, {"passed": False, "reason": f"induction failed: {e}", "runs": []}
        new.version, new.parent_version, new.created_at = old.version + 1, old.version, old.created_at
        new.provenance["repaired_from_version"] = old.version
        new.provenance["repair_reason"] = old.status_reason
        report = self.validate_candidate(new, ctx, validation_arguments, environment=environment)
        if not report["passed"]:
            self.library.log_rejected(new, report)
            self.library.note_repair_failed(skill_id, report["reason"])
            return None, report
        self.library.register(new)
        return new, report

    def _induce_action(self, action: ActionSpec, ctx: ExecContext) -> ActionResult:
        a, clock, started = action.arguments, ctx.clock, ctx.clock.now()
        missing = [i for ids in a["traces"] for i in ids if i not in self._buffer]
        if missing:
            rep = classify(FailureEvidence(arg_errors=[f"no successful execution recorded for action id {m!r}" for m in missing[:5]],
                                           modality=Modality.SKILL))
            return failed_result(action, rep, clock, started)
        inputs = a.get("inputs") or [{} for _ in a["traces"]]
        if len(inputs) != len(a["traces"]):
            rep = classify(FailureEvidence(arg_errors=["arguments.inputs: one entry per trace"], modality=Modality.SKILL))
            return failed_result(action, rep, clock, started)
        traces = [Trace.from_pairs([self._buffer[i] for i in ids], inputs=inp) for ids, inp in zip(a["traces"], inputs)]
        probe = None
        if a.get("probe"):
            tool = self.inducer.tools.get(a["probe"]["operation"])
            if tool is None or tool.side_effects not in ("none", "external_read"):
                rep = classify(FailureEvidence(arg_errors=["arguments.probe: must be an available read-only operation"],
                                               modality=Modality.SKILL))
                return failed_result(action, rep, clock, started)
            probe = ActionSpec(action_id="probe", operation=tool.name, modality=tool.modality,
                               arguments=dict(a["probe"].get("arguments") or {}), auth_scope=tool.auth_scope)
        kw = dict(validation_arguments=a.get("validation_arguments"), probe=probe)
        if a.get("repair_skill_id"):
            skill, report = self.repair(a["repair_skill_id"], traces, ctx, **kw)
        else:
            skill, report = self.induce_validate_register(traces, ctx, name=a.get("name", ""),
                                                          description=a.get("description", ""), **kw)
        now = clock.now()
        runs = report.get("runs", [])
        q = [Quantity("skill.validation.runs", float(len(runs)), kind=OutcomeKind.COUNT),
             Quantity("skill.validation.ok", float(sum(1 for r in runs if r.get("ok"))), kind=OutcomeKind.COUNT),
             Quantity("skill.registered", 1.0 if skill else 0.0, kind=OutcomeKind.BINARY)]
        data = {"registered": bool(skill), "report": report, "skill": self._summary(skill) if skill else None}
        text = (f"skill {skill.skill_id} v{skill.version} registered after {len(runs)} validation run(s)" if skill
                else f"no skill registered: {report.get('reason', '')}")[:600]
        obs = Observation(obs_id=new_id("obs"), ts=now, kind="tool_result", text=text, quantities=q,
                          provenance=Provenance(source="skill_library", retrieved_at=now, trust=Trust.SELF),
                          parent_action_id=action.action_id, data=data)
        if skill is None:
            denied = any(r.get("termination") in ("denied", "needs_approval") for r in runs)
            rep = classify(FailureEvidence(auth_verdict=AuthVerdict.DENY if denied else "", target_missing=not denied,
                                           modality=Modality.SKILL, level=Level.PROCEDURE, detail=text))
            return failed_result(action, rep, clock, started, observations=[obs],
                                 status=ResultStatus.DENIED if denied else ResultStatus.FAILED)
        return ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=started, ended=now,
                            observations=[obs], verified=True, artifacts={"skill_id": skill.skill_id,
                                                                          "skill_version": str(skill.version)})
