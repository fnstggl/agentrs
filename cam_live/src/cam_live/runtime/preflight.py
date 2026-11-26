"""cam_live.runtime.preflight — the live-readiness report (spec Phase 5; ``cam-live run`` calls it before starting).

Live operation is BLOCKED unless every credential, authorization record, platform integration, limit and data
permission is present. Each missing item is listed explicitly. Staging readiness is reported separately, and no
answer from a staging, diagnostic or simulated component is ever counted towards live readiness: a handle whose
capabilities do not say ``live`` is itself a missing item.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..config import CamLiveConfig
from ..contracts import TaskSpec
from ..harness import users_exist
from .doctor import run_doctor
from .tasks import (TaskValidationError, live_requirements, load_task_raw, parse_task, strategy_findings,
                    structural_problems, unset_paths, is_unset)


def _item(kind: str, item: str, detail: str) -> Dict[str, str]:
    return {"kind": kind, "item": item, "detail": detail}


def _credential_names(raw: Dict[str, Any]) -> List[str]:
    return [str(v) for k, v in (raw.get("resources") or {}).items() if "credential" in str(k) and not is_unset(v)]


def task_items(raw: Dict[str, Any]) -> List[Dict[str, str]]:
    out = [_item("task", "task file", p) for p in structural_problems(raw) + strategy_findings(raw)]
    return out + live_requirements(raw)


def authorization_items(task: TaskSpec, raw: Dict[str, Any], policy: Any) -> List[Dict[str, str]]:
    """Operator authorization record and credentials, as the authorization service itself reports them."""
    out: List[Dict[str, str]] = []
    if policy is None:
        out.append(_item("authorization", "authorization service", "not reachable / not started: the operator "
                         "authorization record and credentials cannot be confirmed"))
        for name in _credential_names(raw):
            out.append(_item("credential", name, "presence in the vault not confirmed (no authorization service)"))
        return out
    if type(policy).__name__ == "AllowAllPolicy" or not callable(getattr(policy, "limits_state", None)):
        out.append(_item("authorization", "authorization service", f"{type(policy).__name__} is a diagnostic policy; "
                         "it can never authorize live operation"))
        return out
    try:
        state = policy.limits_state() or {}
    except Exception as e:  # noqa: BLE001
        return [_item("authorization", "authorization service", f"limits_state failed: {type(e).__name__}: {e}")]
    if not (state.get("policy") or {}).get("valid", False):
        out.append(_item("authorization", "policy", f"no valid policy loaded: {(state.get('policy') or {}).get('error', '')}"))
    if state.get("stopped"):
        out.append(_item("authorization", "emergency stop", "the emergency stop is engaged"))
    t = (state.get("tasks") or {}).get(task.objective_id)
    if t is None:
        out.append(_item("authorization", "task registration", f"task {task.objective_id!r} is not registered with "
                         "the authorization service by the operator"))
    elif t.get("unavailable"):
        for why in t["unavailable"]:
            out.append(_item("authorization", "operator authorization record", str(why)))
    elif str(t.get("mode", "")).lower() != "live":
        out.append(_item("authorization", "operator authorization record", f"the authorization service holds this "
                         f"task in mode {t.get('mode')!r}, not live"))
    names_fn = getattr(policy, "secret_names", None)
    try:
        have = {str(n.get("name") if isinstance(n, dict) else n) for n in (names_fn() if callable(names_fn) else [])}
    except Exception as e:  # noqa: BLE001
        have = set()
        out.append(_item("credential", "vault", f"could not list credential names: {type(e).__name__}: {e}"))
    for name in _credential_names(raw):
        if name not in have:
            out.append(_item("credential", name, "no credential of this name is stored in the vault"))
    return out


def integration_items(task: TaskSpec, env: Any) -> List[Dict[str, str]]:
    """Platform integrations, from the environment's own tool inventory and capabilities."""
    if env is None:
        return [_item("integration", "computer environment", "not reachable / not started: tools and platform "
                      "integrations cannot be confirmed")]
    out: List[Dict[str, str]] = []
    try:
        caps = env.get_capabilities() or {}
        tools = list(env.discover_tools() or [])
    except Exception as e:  # noqa: BLE001
        return [_item("integration", "computer environment", f"{type(e).__name__}: {e}")]
    if not caps.get("live", False) or caps.get("provenance") in ("staging", "diagnostic") or caps.get("simulated_time"):
        out.append(_item("integration", "computer environment",
                         f"this is not a live environment (name {caps.get('name')!r}, provenance "
                         f"{caps.get('provenance')!r}, live={caps.get('live')}, simulated_time="
                         f"{caps.get('simulated_time')}); its answers do not count towards live readiness"))
    for t in tools:
        if not t.available:
            out.append(_item("integration", t.name, t.unavailable_reason or "listed as unavailable"))
    have = {t.modality for t in tools if t.available}
    for m in task.interfaces:
        if m not in have:
            out.append(_item("integration", f"interface:{m}", "the task permits this interface but the environment "
                             "offers no available tool for it"))
    return out


def verifier_items(task: TaskSpec, verifier_info: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    if verifier_info is None:
        return [_item("verification", "independent verifier", "not reachable / not started")]
    out = []
    if not verifier_info.get("source_live", False):
        out.append(_item("verification", "evidence source", f"{verifier_info.get('source')!r} is not a live evidence "
                         "source"))
    if verifier_info.get("objective_id") != task.objective_id:
        out.append(_item("verification", "independent verifier", "it serves a different objective"))
    return out


def preflight(task_path: str, cfg: Optional[CamLiveConfig] = None, *, policy: Any = None, env: Any = None,
              verifier_info: Optional[Dict[str, Any]] = None, with_doctor: bool = True) -> Dict[str, Any]:
    """Readiness report for one task file. ``policy`` / ``env`` / ``verifier_info`` are the operator's handles on
    the running services when there are any; without them the corresponding items are reported as unconfirmed."""
    cfg = cfg or CamLiveConfig()
    try:
        raw = load_task_raw(task_path)
    except (OSError, TaskValidationError) as e:
        blocked = [_item("task", task_path, str(e))]
        return {"task": task_path, "mode": None, "live": {"ready": False, "blocked": True, "missing": blocked},
                "staging": {"ready": False, "missing": blocked}, "warnings": []}
    task = parse_task(raw)
    doc = run_doctor(cfg) if with_doctor else {"sections": {}, "summary": {"blocking_staging": [], "blocking_live": []}}
    flat = {c["name"]: c for s in doc["sections"].values() for c in s}

    live: List[Dict[str, str]] = task_items(raw)
    live += authorization_items(task, raw, policy)
    live += integration_items(task, env)
    live += verifier_items(task, verifier_info)
    if raw.get("mode") != "live":
        live.append(_item("task", "mode", f"the task file is in mode {raw.get('mode')!r}; a live run needs mode: live"))
    if not users_exist() or os.geteuid() != 0:
        live.append(_item("isolation", "separated services", "the service users do not exist or this is not root: "
                          "the learner, executor, authorization service and verifier cannot run under separate uids"))
    for name in doc["summary"]["blocking_live"]:
        c = flat[name]
        live.append(_item("system", name, c["detail"] if c["ok"] is False else f"not determined: {c['detail']}"))

    staging: List[Dict[str, str]] = [_item("task", "task file", p) for p in structural_problems(raw) + strategy_findings(raw)]
    if raw.get("mode") != "live":
        staging += [_item("task", p, "unset (marked REQUIRED in the task file)") for p in unset_paths(raw)]
    for name in doc["summary"]["blocking_staging"]:
        staging.append(_item("system", name, flat[name]["detail"]))

    warnings = [c["detail"] for n, c in flat.items() if n in ("kernel_network_filtering",) and c["ok"] is not True]
    return {"task": task_path, "objective_id": task.objective_id, "mode": raw.get("mode"),
            "live": {"ready": not live, "blocked": bool(live), "missing": live},
            "staging": {"ready": not staging, "missing": staging},
            "warnings": warnings,
            "note": "live readiness is established only from live services and operator records; nothing reported "
                    "by a staging, diagnostic or simulated component counts towards it"}


def render(report: Dict[str, Any]) -> str:
    lines = [f"preflight for {report['task']} (objective {report.get('objective_id')}, mode {report.get('mode')})"]
    lv = report["live"]
    lines.append("LIVE: " + ("READY" if lv["ready"] else f"BLOCKED — {len(lv['missing'])} item(s) missing"))
    for m in lv["missing"]:
        lines.append(f"  - [{m['kind']}] {m['item']}: {m['detail']}")
    sg = report["staging"]
    lines.append("STAGING: " + ("ready" if sg["ready"] else f"not ready — {len(sg['missing'])} item(s)"))
    for m in sg["missing"]:
        lines.append(f"  - [{m['kind']}] {m['item']}: {m['detail']}")
    for w in report.get("warnings", []):
        lines.append(f"warning: {w}")
    return "\n".join(lines)
