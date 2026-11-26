"""cam_live.runtime.tasks — external task configuration: YAML -> :class:`~cam_live.contracts.TaskSpec` (spec Part 13).

A task file states the objective, how success is verified independently of CAM, which resources and interfaces are
authorized, the experiment boundaries, the measurement protocol and the data permissions. It never contains a way
of achieving the objective: any forbidden key or strategy-like text anywhere in the file is rejected.

Unset items are written with the marker ``REQUIRED`` (optionally ``REQUIRED: what the operator must provide``).
A task in mode ``live`` with any unset item cannot be loaded for a run; resources are never assumed unlimited.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

from ..contracts import Limits, TaskSpec

UNSET_MARKER = "REQUIRED"
MODES = ("diagnostic", "staging", "live")
LIMIT_FIELDS = ("duration_s", "money", "compute_s", "max_external_actions", "max_decisions")
TOP_LEVEL_KEYS = ("version", "objective_id", "objective", "success", "family", "resources", "interfaces", "limits",
                  "measurement", "data_permissions", "stop_conditions", "secondary_metrics", "mode")
#: stop conditions the runtime knows how to enforce (anything else in a task file is rejected)
STOP_CONDITIONS = ("verified_success", "verified_failure", "duration_exhausted", "money_exhausted",
                   "compute_exhausted", "external_actions_exhausted", "decisions_exhausted", "operator_stop",
                   "emergency_stop", "authorization_revoked")
#: data permissions that must be stated explicitly (true or false) before a live run
REQUIRED_DATA_PERMISSIONS = ("store_message_content", "train_on_message_content", "store_member_identifiers",
                             "store_aggregate_metrics", "collect_feedback_with_consent")

# ---- rejection lists (this module only *refuses* such content; nothing here is ever shown to the learner) -----
#: mapping keys that would carry a solution; matched as whole keys and as key tokens
REJECTED_KEYS = tuple(TaskSpec.FORBIDDEN_KEYS) + (
    "tactic", "tactics", "approach", "method", "methods", "recipe", "instructions", "guidance", "hints", "tips",
    "roadmap", "campaign", "marketing", "promotion", "funnel", "audience", "persona", "topic", "theme", "calendar",
    "howto", "procedure", "workflow", "sequence", "framework", "suggestions", "ideas", "recommended", "examples")
#: schema keys that contain a rejected token but are part of the contract itself
_SCHEMA_SAFE_KEYS = ("max_external_actions",)
#: text patterns that read as a way of achieving the objective (label -> regex)
REJECTED_TEXT: Dict[str, str] = {
    "enumerated procedure": r"(?im)(?:^|\s)(?:step\s*\d+\b|\d+[.)]\s+[a-z])",
    "ordered instruction": r"(?i)\bfirst\b[^.]{0,80}\b(?:then|next|after that|finally)\b",
    "directive to the agent": r"(?i)\b(?:you|cam|the agent|the model)\s+(?:should|must|shall|ought to|needs? to|has to|is to)\b",
    "timing directive": r"(?i)\b(?:every|each)\s+(?:day|week|hour|morning|evening|night|weekday|weekend|"
                        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\b(?:daily|weekly|hourly|nightly)\b"
                        r"|\b\d+\s*(?:x|times)\s*(?:a|per)\s*(?:day|week|hour)\b|\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
    "tactic vocabulary": r"(?i)\b(?:niche|tactic|marketing|advertis\w*|promot(?:e|es|ing|ion|ions)|giveaway\w*|influencer\w*|"
                         r"viral\w*|seo|funnel\w*|call[- ]to[- ]action|target audience|persona\w*|outreach|cross[- ]?post\w*|"
                         r"growth hack\w*|engagement bait|scarcity|fomo|social proof|reciprocity|gamif\w*|onboarding flow)\b",
    "suggested solution": r"(?i)\b(?:for example|e\.g\.|such as|consider|try|recommend\w*|suggest\w*|best practice\w*|"
                          r"proven|playbook)\b[^.]{0,60}\b(?:post\w*|invit\w*|shar\w*|launch\w*|build\w*|publish\w*|messag\w*)\b",
}
_REJECTED_TEXT = [(label, re.compile(rx)) for label, rx in REJECTED_TEXT.items()]


class TaskValidationError(ValueError):
    def __init__(self, problems: List[str], path: str = ""):
        self.problems, self.path = list(problems), path
        super().__init__((f"{path}: " if path else "") + "; ".join(self.problems))


def is_unset(v: Any) -> bool:
    return v is None or (isinstance(v, str) and (not v.strip() or v.strip().upper().startswith(UNSET_MARKER)))


def _walk(node: Any, path: str = "") -> Iterator[Tuple[str, str, Any]]:
    """Yield (path, key, value) for every mapping entry and list element of a nested structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}" if path else str(k)
            yield p, str(k), v
            yield from _walk(v, p)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            p = f"{path}[{i}]"
            yield p, "", v
            yield from _walk(v, p)


def strategy_findings(raw: Any) -> List[str]:
    """Every place in a task document that carries (or looks like) a solution strategy. [] = clean."""
    out: List[str] = []
    rejected = {k.lower() for k in REJECTED_KEYS}
    for path, key, value in _walk(raw):
        k = key.lower()
        if k and k not in _SCHEMA_SAFE_KEYS:
            tokens = set(re.split(r"[^a-z0-9]+", k)) | {k}
            hit = sorted(tokens & rejected)
            if hit:
                out.append(f"{path}: key {key!r} is not allowed in a task file ({', '.join(hit)})")
        if isinstance(value, str) and not is_unset(value):
            for label, rx in _REJECTED_TEXT:
                m = rx.search(value)
                if m:
                    out.append(f"{path}: text reads as a solution strategy ({label}: {m.group(0).strip()!r})")
    return out


def unset_paths(raw: Any) -> List[str]:
    """Dotted paths of every value still carrying the ``REQUIRED`` marker (or left empty)."""
    return [p for p, _k, v in _walk(raw) if not isinstance(v, (dict, list)) and is_unset(v)]


def live_requirements(raw: Dict[str, Any]) -> List[Dict[str, str]]:
    """Everything a live run needs from the task file that is absent or unset: [{kind, item, detail}]."""
    miss: List[Dict[str, str]] = []

    def need(kind: str, item: str, detail: str) -> None:
        miss.append({"kind": kind, "item": item, "detail": detail})

    limits = raw.get("limits") or {}
    for f in LIMIT_FIELDS:
        v = limits.get(f)
        if is_unset(v) or not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
            need("limit", f"limits.{f}", "must be an explicit positive number; resources are never assumed unlimited")
    res = raw.get("resources") or {}
    tools = res.get("authorized_tools")
    if is_unset(tools) or not isinstance(tools, list) or not tools or any(is_unset(t) for t in tools):
        need("authorization", "resources.authorized_tools", "explicit list of authorized tools / tool families")
    pub = res.get("publishing")
    if not isinstance(pub, dict) or not isinstance(pub.get("authorized"), bool):
        need("authorization", "resources.publishing.authorized", "publishing authorization must be stated (true or false)")
    elif pub.get("authorized") and (is_unset(pub.get("destinations")) or not pub.get("destinations")):
        need("authorization", "resources.publishing.destinations", "authorized publishing destinations")
    perms = raw.get("data_permissions") or {}
    for k in REQUIRED_DATA_PERMISSIONS:
        if not isinstance(perms.get(k), bool):
            need("data_permission", f"data_permissions.{k}", "must be stated explicitly (true or false)")
    stops = raw.get("stop_conditions")
    if not isinstance(stops, list) or not stops:
        need("stop_condition", "stop_conditions", "explicit list of stop conditions")
    if not isinstance(raw.get("interfaces"), list) or not raw.get("interfaces"):
        need("authorization", "interfaces", "permitted environment interfaces")
    if not (raw.get("success") or {}):
        need("verification", "success", "independent success criteria (verifier configuration)")
    if not (raw.get("measurement") or {}):
        need("verification", "measurement", "outcome measurement protocol")
    seen = {m["item"] for m in miss}
    for p in unset_paths(raw):
        if p not in seen and not any(p.startswith(s + "[") or p.startswith(s + ".") for s in seen):
            kind = ("credential" if "credential" in p else "limit" if p.startswith("limits.") else
                    "data_permission" if p.startswith("data_permissions.") else "resource")
            need(kind, p, "unset (marked REQUIRED in the task file)")
    return miss


def structural_problems(raw: Any) -> List[str]:
    if not isinstance(raw, dict):
        return ["task file must be a mapping"]
    out = [f"unknown top-level key {k!r}" for k in raw if k not in TOP_LEVEL_KEYS]
    if is_unset(raw.get("objective_id")):
        out.append("objective_id is required")
    if not isinstance(raw.get("objective", ""), str):
        out.append("objective must be text")
    mode = raw.get("mode", "staging")
    if mode not in MODES:
        out.append(f"mode must be one of {MODES}")
    for name, tp in (("success", dict), ("resources", dict), ("limits", dict), ("measurement", dict),
                     ("data_permissions", dict), ("interfaces", list), ("stop_conditions", list),
                     ("secondary_metrics", list)):
        if raw.get(name) is not None and not isinstance(raw[name], tp):
            out.append(f"{name} must be a {tp.__name__}")
    for k in (raw.get("limits") or {}) if isinstance(raw.get("limits"), dict) else ():
        if k not in LIMIT_FIELDS:
            out.append(f"unknown limit {k!r}")
    for s in raw.get("stop_conditions") or [] if isinstance(raw.get("stop_conditions"), list) else []:
        if s not in STOP_CONDITIONS:
            out.append(f"unknown stop condition {s!r} (known: {', '.join(STOP_CONDITIONS)})")
    for k, v in ((raw.get("data_permissions") or {}).items() if isinstance(raw.get("data_permissions"), dict) else ()):
        if not isinstance(v, bool) and not is_unset(v):
            out.append(f"data_permissions.{k} must be true or false")
    return out


def run_problems(raw: Dict[str, Any]) -> List[str]:
    """Why this task cannot be *run* as configured (beyond structure and strategy content)."""
    mode = raw.get("mode", "staging")
    if mode == "live":
        return [f"{m['item']}: {m['detail']}" for m in live_requirements(raw)]
    out: List[str] = []
    limits = raw.get("limits") or {}
    for f in ("duration_s", "max_decisions"):
        v = limits.get(f)
        if is_unset(v) or not isinstance(v, (int, float)) or v <= 0:
            out.append(f"limits.{f}: must be an explicit positive number in every mode")
    if not str(raw.get("objective") or "").strip():
        # autonomous learning (spec 11.4): no objective, but a complete fixed budget
        for f in ("money", "compute_s", "max_external_actions"):
            v = limits.get(f)
            if is_unset(v) or not isinstance(v, (int, float)) or v < 0:
                out.append(f"limits.{f}: autonomous learning needs every budget stated explicitly")
    elif not (raw.get("success") or {}):
        out.append("success: independent success criteria are required")
    return out


def _num(v: Any) -> float:
    return 0.0 if is_unset(v) or isinstance(v, bool) or not isinstance(v, (int, float)) else float(v)


def parse_task(raw: Dict[str, Any]) -> TaskSpec:
    """Build the TaskSpec. Unset limits become 0 (= "must be set explicitly before a live run")."""
    lim = raw.get("limits") or {}
    perms = {k: bool(v) for k, v in (raw.get("data_permissions") or {}).items() if isinstance(v, bool)}
    return TaskSpec(
        objective_id=str(raw.get("objective_id", "")), objective=str(raw.get("objective") or ""),
        success=dict(raw.get("success") or {}), family=str(raw.get("family") or ""),
        resources=dict(raw.get("resources") or {}), interfaces=[str(x) for x in raw.get("interfaces") or []],
        limits=Limits(duration_s=_num(lim.get("duration_s")), money=_num(lim.get("money")),
                      compute_s=_num(lim.get("compute_s")),
                      max_external_actions=int(_num(lim.get("max_external_actions"))),
                      max_decisions=int(_num(lim.get("max_decisions")))),
        measurement=dict(raw.get("measurement") or {}), data_permissions=perms,
        stop_conditions=[str(s) for s in raw.get("stop_conditions") or []],
        secondary_metrics=[str(s) for s in raw.get("secondary_metrics") or []], mode=str(raw.get("mode", "staging")))


def load_task_raw(path: str) -> Dict[str, Any]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TaskValidationError(["task file must be a mapping"], path)
    return raw


def validate_task(raw: Dict[str, Any], *, for_run: bool = True) -> List[str]:
    problems = structural_problems(raw)
    problems += strategy_findings(raw)
    if for_run and not problems:
        problems += run_problems(raw)
    return problems


def load_task(path: str, *, for_run: bool = True) -> TaskSpec:
    """Load and validate a task file. ``for_run=False`` allows unset items (inspection / preflight) but still
    rejects strategy content and malformed files."""
    raw = load_task_raw(path)
    problems = validate_task(raw, for_run=for_run)
    if problems:
        raise TaskValidationError(problems, path)
    return parse_task(raw)


def task_from_dict(raw: Dict[str, Any], *, for_run: bool = True) -> TaskSpec:
    problems = validate_task(raw, for_run=for_run)
    if problems:
        raise TaskValidationError(problems)
    return parse_task(raw)


def check_runnable(task: TaskSpec) -> List[str]:
    """Runtime-side re-check of a TaskSpec object (a TaskSpec may be built in code, not only from YAML)."""
    out: List[str] = []
    lim = task.limits
    if task.mode == "live":
        for f in LIMIT_FIELDS:
            if getattr(lim, f) <= 0:
                out.append(f"limits.{f} is not set; a live run needs every limit")
        if not task.stop_conditions:
            out.append("stop_conditions are not set")
        if not task.interfaces:
            out.append("interfaces are not set")
        if unset_paths({"resources": task.resources, "success": task.success, "measurement": task.measurement}):
            out.append("unset items: " + ", ".join(unset_paths({"resources": task.resources, "success": task.success,
                                                               "measurement": task.measurement})))
    elif lim.duration_s <= 0 and lim.max_decisions <= 0:
        out.append("neither limits.duration_s nor limits.max_decisions is set; unbounded runs are refused")
    if not task.objective.strip():
        if lim.duration_s <= 0 or lim.max_decisions <= 0:
            out.append("autonomous learning (no objective) needs explicit duration and decision budgets")
    for k in tuple(task.success) + tuple(task.resources) + tuple(task.measurement):
        if str(k).lower() in {x.lower() for x in TaskSpec.FORBIDDEN_KEYS}:
            out.append(f"forbidden key {k!r}")
    return out


def autonomous_task(objective_id: str, limits: Limits, *, interfaces: Optional[List[str]] = None,
                    mode: str = "staging", family: str = "autonomous") -> TaskSpec:
    """Task for the optional learning mode without an assigned objective (spec 11.4): a fixed budget and nothing
    else. Self-chosen learning goals stay inside the learner; usefulness is judged on later assigned objectives."""
    t = TaskSpec(objective_id=objective_id, objective="", family=family, interfaces=list(interfaces or []),
                 limits=limits, mode=mode, stop_conditions=["duration_exhausted", "decisions_exhausted"])
    problems = check_runnable(t)
    if problems:
        raise TaskValidationError(problems)
    return t


def scan_file(path: str) -> List[str]:
    """Mechanical strategy scan of one task file (used by tests and ``cam-live``)."""
    return strategy_findings(load_task_raw(path))


def default_tasks_dir() -> str:
    from ..config import PROJECT_ROOT
    return os.path.join(PROJECT_ROOT, "tasks")
