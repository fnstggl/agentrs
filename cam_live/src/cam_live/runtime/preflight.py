"""cam_live.runtime.preflight — the live-readiness report (spec Phase 5; ``cam-live run`` calls it before starting).

Live operation is BLOCKED unless every credential, authorization record, platform integration, limit and data
permission is present. Each missing item is listed explicitly. Staging readiness is reported separately, and no
answer from a staging, diagnostic or simulated component is ever counted towards live readiness: a handle whose
capabilities do not say ``live`` is itself a missing item.

Self-declared labels are not trusted on their own. The environment's top-level ``live`` label is cross-checked
against every component it reports (an executor that names a staging service or a loopback address blocks), the
authorization service must hold the task in mode live with a valid operator record, the verifier must report a live
evidence source, every interface the task permits needs an available tool of its own, and a task with Discord
resources needs the Discord tools plus a read-only identity and permission check executed through the computer
environment (bot in the authorized guild, the privileged intents the verifier and the data permissions need, the
operator's permission scope held), the bot credential in the vault and the verifier's own copy of it.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlsplit

from ..config import CamLiveConfig
from ..contracts import ActionSpec, AuthVerdict, Modality, TaskSpec, new_id
from ..harness import users_exist
from .doctor import run_doctor
from .tasks import (TaskValidationError, has_discord, live_requirements, load_task_raw, parse_task, strategy_findings,
                    structural_problems, unset_paths, is_unset)

#: read-only Discord tools the preflight itself needs for the identity / permission check
DISCORD_CHECK_TOOLS = ("discord.get_identity", "discord.get_permissions")
_NOT_LIVE = ("staging", "diagnostic", "simulated", "test", "local")


def _item(kind: str, item: str, detail: str) -> Dict[str, str]:
    return {"kind": kind, "item": item, "detail": detail}


def _credential_names(raw: Dict[str, Any]) -> List[str]:
    res = raw.get("resources") or {}
    out = [str(v) for k, v in res.items() if "credential" in str(k) and isinstance(v, (str, int)) and not is_unset(v)]
    d = res.get("discord") if isinstance(res.get("discord"), dict) else {}
    if d.get("token_secret") and not is_unset(d.get("token_secret")):
        out.append(str(d["token_secret"]))
    return list(dict.fromkeys(out))


def _discord_settings(task: TaskSpec) -> Dict[str, str]:
    r = task.resources if isinstance(task.resources, dict) else {}
    d = r.get("discord") if isinstance(r.get("discord"), dict) else {}
    guild = d.get("guild_id") or r.get("discord_guild_id") or ""
    secret = d.get("token_secret") or r.get("discord_bot_credential_name") or "discord_bot_token"
    return {"guild_id": "" if is_unset(guild) else str(guild), "token_secret": "" if is_unset(secret) else str(secret)}


def _is_diagnostic_policy(policy: Any) -> bool:
    try:
        from ..diagnostics.world import AllowAllPolicy
        if isinstance(policy, AllowAllPolicy):
            return True
    except ImportError:
        pass
    return not callable(getattr(policy, "limits_state", None)) or not callable(getattr(policy, "secret_names", None))


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
    if _is_diagnostic_policy(policy):
        out.append(_item("authorization", "authorization service", f"{type(policy).__name__} is a diagnostic policy "
                         "or not an authorization service (no limits_state / secret_names); it can never authorize "
                         "live operation"))
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


def _labels(caps: Any, path: str = "capabilities") -> List[str]:
    """Every component inside the environment's capabilities that says it is not live (recursively)."""
    out: List[str] = []
    if isinstance(caps, dict):
        for k, v in caps.items():
            p = f"{path}.{k}"
            if k in ("environment", "provenance", "mode") and isinstance(v, str) and v.lower() in _NOT_LIVE:
                out.append(f"{p} = {v!r}")
            elif k in ("live",) and v is False and path != "capabilities":
                out.append(f"{p} = False")
            elif k in ("staging", "simulated", "simulated_time") and v is True and path != "capabilities":
                out.append(f"{p} = True")
            elif k in ("base_url", "url", "endpoint") and isinstance(v, str):
                host = (urlsplit(v).hostname or "").lower()
                if host in ("localhost", "::1") or host.startswith("127.") or host.endswith(".local"):
                    out.append(f"{p} = {v!r} (a local service)")
            else:
                out += _labels(v, p)
    elif isinstance(caps, list):
        for i, v in enumerate(caps):
            out += _labels(v, f"{path}[{i}]")
    return out


def missing_integrations(task: Optional[TaskSpec], tools: Sequence[Any]) -> List[Dict[str, str]]:
    """Integrations the task needs that the environment's tool inventory lacks, named tool by tool (used by the
    preflight and by ``cam-live inspect-harness``). Without a task the Discord integration of the first live
    objective is still named when it is absent."""
    avail = [t for t in tools if getattr(t, "available", False)]
    names = {t.name for t in avail}
    out: List[Dict[str, str]] = []
    if task is not None:
        for m in task.interfaces:
            if m in Modality.ALL:
                if not any(t.modality == m for t in avail):
                    out.append({"tool": f"interface:{m}", "reason": "permitted by the task, but no available tool "
                                "of this interface is registered"})
            else:
                import fnmatch
                if not any(fnmatch.fnmatchcase(n, m) for n in names):
                    out.append({"tool": f"interface:{m}", "reason": "permitted by the task, but no available tool "
                                "matches it"})
    if task is None or has_discord(task):
        listed = [t for t in tools if t.name.startswith("discord.")]
        if not listed:
            why = ("the task authorizes a Discord guild, but" if task is not None else "the first live objective "
                   "needs the Discord integration, but")
            out.append({"tool": "discord", "reason": f"{why} no Discord tool is registered in this environment (no "
                        "Discord executor configured: runtime.executors / launcher --executor, and the bot "
                        "credential in the vault)"})
        else:
            for n in DISCORD_CHECK_TOOLS:
                if n not in names:
                    out.append({"tool": n, "reason": "needed for the readiness check of the bot (read-only), but not "
                                "available"})
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
    if not caps.get("live", False) or caps.get("provenance") != "live" or caps.get("simulated_time"):
        out.append(_item("integration", "computer environment",
                         f"this is not a live environment (name {caps.get('name')!r}, provenance "
                         f"{caps.get('provenance')!r}, live={caps.get('live')}, simulated_time="
                         f"{caps.get('simulated_time')}); its answers do not count towards live readiness"))
    for lab in _labels(caps):
        out.append(_item("integration", "environment component", f"{lab}: a component of the environment is not "
                         "live, whatever the environment's own label says"))
    for t in tools:
        if not t.available:
            out.append(_item("integration", t.name, t.unavailable_reason or "listed as unavailable"))
    for m in missing_integrations(task, tools):
        out.append(_item("integration", m["tool"], m["reason"]))
    return out


def discord_items(task: TaskSpec, raw: Dict[str, Any], policy: Any, env: Any, run_dir: str = "") -> List[Dict[str, str]]:
    """The Discord readiness of a task with Discord resources, item by item. With the operator's handles on the
    running services the bot is checked read-only through the computer environment (``discord.get_identity``,
    ``discord.get_permissions``, authorized like any action); without them every item is named as unconfirmed."""
    if not has_discord(task):
        return []
    s = _discord_settings(task)
    out: List[Dict[str, str]] = []
    if not s["guild_id"]:
        out.append(_item("discord", "guild", "no authorized guild is configured (resources.discord_guild_id)"))
    need_intents = ["GUILD_MEMBERS"]
    perms = task.data_permissions or {}
    if perms.get("store_message_content") or perms.get("read_message_content") or perms.get("train_on_message_content"):
        need_intents.append("MESSAGE_CONTENT")
    secret = s["token_secret"]
    if run_dir and secret:
        copy_path = os.path.join(run_dir, "verifier", "secrets", secret)
        try:
            present = os.path.isfile(copy_path)
        except OSError:
            present = False
        if not present:
            out.append(_item("discord", "verifier credential copy", f"the verifier's own copy of {secret!r} is not "
                             f"in {os.path.dirname(copy_path)} (cam_live.harness.credentials.store(..., verifier=True))"))
    else:
        out.append(_item("discord", "verifier credential copy", "not confirmed (no run directory of the running "
                         "services)"))
    if policy is None or env is None or _is_diagnostic_policy(policy) or not s["guild_id"]:
        for what in ("bot membership in the authorized guild", "privileged intents " + "/".join(need_intents),
                     "the bot's permissions against the operator's scope"):
            out.append(_item("discord", what, "not confirmed: needs the running computer environment and the "
                             "operator's handle on the authorization service (checked by the supervisor of "
                             "`cam-live run`)"))
        return out
    try:
        tools = {t.name: t for t in env.discover_tools() or []}
    except Exception as e:  # noqa: BLE001
        return out + [_item("discord", "identity check", f"tool inventory not readable: {type(e).__name__}: {e}")]
    data: Dict[str, Dict[str, Any]] = {}
    for name in DISCORD_CHECK_TOOLS:
        t = tools.get(name)
        if t is None or not t.available:
            out.append(_item("discord", name, "not available, so the bot cannot be checked"
                             + (f": {t.unavailable_reason}" if t is not None and t.unavailable_reason else "")))
            continue
        action = ActionSpec(action_id=new_id("act"), operation=name, modality=t.modality, level=t.level,
                            arguments={"guild_id": s["guild_id"]}, auth_scope=t.auth_scope, provenance="preflight",
                            expected="read-only readiness check of the bot")
        try:
            auth = policy.authorize(action, task, {"purpose": "preflight", "mode": task.mode, "objective_id":
                                                   task.objective_id})
            if auth.verdict != AuthVerdict.ALLOW:
                out.append(_item("discord", name, f"the authorization service does not allow the read-only check "
                                 f"({auth.verdict}: {'; '.join(auth.reasons)[:200]})"))
                continue
            res = env.execute(action, auth)
        except Exception as e:  # noqa: BLE001
            out.append(_item("discord", name, f"check failed: {type(e).__name__}: {e}"))
            continue
        if getattr(res, "status", "") != "ok":
            out.append(_item("discord", name, f"check did not succeed ({getattr(res, 'status', '?')}: "
                             f"{str(getattr(res, 'failure_detail', ''))[:200]})"))
            continue
        d: Dict[str, Any] = {}
        for o in getattr(res, "observations", []) or []:
            d.update(getattr(o, "data", {}) or {})
        data[name] = d
        if str(d.get("environment", "")).lower() != "live":
            out.append(_item("discord", name, f"the Discord executor answered from {d.get('environment')!r}, not "
                             "from the live platform"))
    ident = data.get("discord.get_identity")
    if ident is not None:
        if ident.get("in_guild") is not True:
            out.append(_item("discord", "bot membership in the authorized guild", "the bot is not a member of guild "
                             f"{s['guild_id']}; the server owner must add it"))
        intents = ((ident.get("application") or {}).get("intents") or {})
        for i in need_intents:
            if intents.get(i) is not True:
                out.append(_item("discord", f"privileged intent {i}", "not granted to the application (Discord "
                                 "developer portal)"))
    perms_d = data.get("discord.get_permissions")
    if perms_d is not None and perms_d.get("missing_from_scope"):
        out.append(_item("discord", "bot permissions", "the bot lacks permissions of the operator's scope: "
                         + ", ".join(str(x) for x in perms_d["missing_from_scope"])[:300]))
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


#: live-blocking system checks the operator may accept as a risk, by an explicit command-line acknowledgement
#: (``cam-live run --accept-ephemeral-storage``); nothing else can be accepted, and nothing is accepted by default
ACCEPTABLE_RISKS = {"persistent_storage": "--accept-ephemeral-storage"}


def preflight(task_path: str, cfg: Optional[CamLiveConfig] = None, *, policy: Any = None, env: Any = None,
              verifier_info: Optional[Dict[str, Any]] = None, with_doctor: bool = True,
              run_dir: str = "", accept_risks: Sequence[str] = ()) -> Dict[str, Any]:
    """Readiness report for one task file. ``policy`` / ``env`` / ``verifier_info`` are the operator's handles on
    the running services when there are any; without them the corresponding items are reported as unconfirmed.

    ``accept_risks``: names of ``ACCEPTABLE_RISKS`` the operator acknowledged on the command line. Only in mode live
    (outside it these checks do not block anything); an accepted check is still run and reported, under
    ``live.accepted_risks`` with its finding, instead of blocking."""
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
    live += discord_items(task, raw, policy, env, run_dir)
    live += verifier_items(task, verifier_info)
    if raw.get("mode") != "live":
        live.append(_item("task", "mode", f"the task file is in mode {raw.get('mode')!r}; a live run needs mode: live"))
    if not users_exist() or os.geteuid() != 0:
        live.append(_item("isolation", "separated services", "the service users do not exist or this is not root: "
                          "the learner, executor, authorization service and verifier cannot run under separate uids"))
    accepted: List[Dict[str, str]] = []
    for name in doc["summary"]["blocking_live"]:
        c = flat[name]
        it = _item("system", name, c["detail"] if c["ok"] is False else f"not determined: {c['detail']}")
        if raw.get("mode") == "live" and name in ACCEPTABLE_RISKS and name in set(accept_risks):
            accepted.append({**it, "accepted_by": f"operator (command line {ACCEPTABLE_RISKS[name]})",
                             "consequence": "the run proceeds although this check fails; the risk is the operator's"})
        else:
            live.append(it)

    staging: List[Dict[str, str]] = [_item("task", "task file", p) for p in structural_problems(raw) + strategy_findings(raw)]
    if raw.get("mode") != "live":
        staging += [_item("task", p, "unset (marked REQUIRED in the task file)") for p in unset_paths(raw)]
    for name in doc["summary"]["blocking_staging"]:
        staging.append(_item("system", name, flat[name]["detail"]))

    warnings = [c["detail"] for n, c in flat.items() if n in ("kernel_network_filtering",) and c["ok"] is not True]
    return {"task": task_path, "objective_id": task.objective_id, "mode": raw.get("mode"),
            "live": {"ready": not live, "blocked": bool(live), "missing": live, "accepted_risks": accepted},
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
    for m in lv.get("accepted_risks", []):
        lines.append(f"  ! ACCEPTED RISK [{m['kind']}] {m['item']} ({m['accepted_by']}): {m['detail']}")
    sg = report["staging"]
    lines.append("STAGING: " + ("ready" if sg["ready"] else f"not ready — {len(sg['missing'])} item(s)"))
    for m in sg["missing"]:
        lines.append(f"  - [{m['kind']}] {m['item']}: {m['detail']}")
    for w in report.get("warnings", []):
        lines.append(f"warning: {w}")
    return "\n".join(lines)
