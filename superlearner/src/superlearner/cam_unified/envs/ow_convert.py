"""Conversion of Crucible actor-visible data into contract records (`interfaces.Observation`, `ToolSchema`,
`ExecutionResult`). Pure functions over plain dicts: nothing here touches a world kernel.

Rules (contract v2, architecture decisions 6-7):

* **Time**: Crucible's clock is integer simulated seconds since world start; records carry float days
  (``t = seconds / 86400`` exactly, no rounding).
* **Units as given**: no value is converted. Every NUMERIC observation states the unit the environment uses
  (``cents`` for all Crucible money, ``s`` for ``*_s`` fields, ``count`` for counters ...) from an explicit table
  (`unit_for`) keyed by field name and module; unknown numbers are labelled ``number``.
* **Entities**: Crucible has no uniform id scheme (``p:<handle>``, ``o:<org>``, ``svc:<host>``, ``acct_000017``,
  ``lst_000001``, ``inst_..``, repos ``org/name``, bare hosts, ``gen.<family>.<hash>`` categories). Extraction is
  key-driven (`ENTITY_KEYS`) plus prefix patterns (`id_type`).
* **Lossless content**: an event observation's ``content`` is the canonical JSON of ``{type, actor, payload}`` and a
  tool result's ``content`` is the JSON of the complete result, so a consumer can reconstruct exactly what the actor
  saw (`event_dict`, `result_dict` invert the conversion; tests check the round trip against the server).
* **Bounded NUMERIC tokens, never silently**: a summary module or a tool result yields at most `max_numeric` NUMERIC
  records (breadth-first, top-level fields first); when more numeric leaves exist, the TEXT / TOOL_RESULT record that
  carries the complete JSON says how many were left out (``meta["numeric_truncated"]``, ``meta["numeric_total"]``).
* **No aliasing**: `tool_schema` deep-copies the kernel's schema dicts (`Endpoint.spec()` returns the endpoint's own
  parameter dict), so nothing a consumer does to a ToolSchema can change the world's validation rules.
"""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import replace
from typing import Any, Iterable, Optional

from ..interfaces import EvidenceKind, ExecutionResult, Observation, ObsKind, ToolSchema

DAY = 86400.0

# ---------------------------------------------------------------------------------------------- entities
_PREFIXED = re.compile(r"^(p|o|svc):[a-z0-9][A-Za-z0-9_.-]*$")
_NUMBERED = re.compile(r"^(acct|tx|lst|inst|dep|doc|fpost|probe|tkt|ctr|job|post|ord|app|camp|bty|req|msg|thr|cap|"
                       r"inv|loan|deal|wf|db|site|page|ad|sub|txn)_[A-Za-z0-9]+$")
_REPO = re.compile(r"^[a-z0-9][a-z0-9-]*/[A-Za-z0-9_.-]+$")
_CATEGORY = re.compile(r"^gen\.[a-z_]+\.[0-9a-f]{6,}$")

ID_PREFIX_TYPES = {"p": "person", "o": "organisation", "svc": "service", "acct": "account", "tx": "transaction",
                   "lst": "listing", "inst": "instance", "dep": "deployment", "doc": "document", "fpost": "post",
                   "probe": "probe", "tkt": "ticket", "ctr": "contract", "job": "job", "post": "posting",
                   "ord": "order", "app": "application", "camp": "campaign", "bty": "bounty", "req": "request",
                   "msg": "message", "thr": "thread", "cap": "capability", "inv": "invite", "loan": "loan",
                   "deal": "deal", "wf": "workflow", "db": "database", "site": "site", "page": "page", "ad": "ad",
                   "sub": "subscription", "txn": "transaction"}

# keys whose (string) value names an entity, in priority order for choosing the primary entity
ENTITY_KEYS = ("id", "listing", "host", "repo", "instance", "deployment", "account", "org", "owner", "seller",
               "buyer", "author", "holder", "founder", "person", "principal", "provider", "rig", "category", "need",
               "ticket", "contract", "posting", "order", "applicant", "employer", "recipient", "to", "from", "by",
               "poster", "winner", "member", "lender", "borrower", "service", "capability", "board", "thread")
_KEY_TYPES = {"host": "host", "repo": "repository", "rig": "facility", "category": "category", "need": "category",
              "board": "board", "service": "host"}


def id_type(value: Any, key: str | None = None) -> Optional[str]:
    """Entity type of an id-like string, or None when the string is not an identifier."""
    if not isinstance(value, str) or not value or len(value) > 96 or " " in value:
        return None
    m = _PREFIXED.match(value)
    if m:
        return ID_PREFIX_TYPES[m.group(1)]
    m = _NUMBERED.match(value)
    if m:
        return ID_PREFIX_TYPES.get(m.group(1), "object")
    if _CATEGORY.match(value):
        return "category"
    if key in ("repo",) and _REPO.match(value):
        return "repository"
    if key in _KEY_TYPES and re.match(r"^[a-z0-9][a-z0-9_.-]{0,62}$", value):
        return _KEY_TYPES[key]
    return None


def entities_in(obj: Any, limit: int = 8) -> list[tuple[str, str]]:
    """Distinct (entity id, type) pairs named by a payload, primary first (key priority, then document order)."""
    found: list[tuple[int, int, str, str]] = []
    seen: set[str] = set()
    order = [0]

    def visit(x: Any, key: str | None, depth: int) -> None:
        if depth > 4 or len(found) > 4 * limit:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                visit(v, k, depth + 1)
        elif isinstance(x, (list, tuple)):
            for v in x[:32]:
                visit(v, key, depth + 1)
        else:
            ty = id_type(x, key)
            if ty is not None and x not in seen:
                seen.add(x)
                pri = ENTITY_KEYS.index(key) if key in ENTITY_KEYS else len(ENTITY_KEYS)
                order[0] += 1
                found.append((pri, order[0], x, ty))
    visit(obj, None, 0)
    found.sort(key=lambda r: (r[0], r[1]))
    return [(x, ty) for _, _, x, ty in found[:limit]]


# ---------------------------------------------------------------------------------------------- units
_COUNT_KEYS = {"unread", "count", "n", "requests", "failures", "ticks", "cores", "cores_free", "seats", "members",
               "pending_invites", "commits", "size", "examples", "calls", "served", "dropped", "rejected", "ok",
               "errors", "stars_n", "reviews", "customers", "orders", "units", "instances", "deployments", "listings",
               "outages_30d", "overload", "attempts", "requests_ok", "retained", "independent_requests", "day",
               "hour", "seq", "tick", "applies_at_tick", "horizon_ticks", "lines", "ms_free"}
_FRACTION_KEYS = {"availability", "reliability", "share", "fraction", "rate", "quality", "accuracy", "utilisation",
                  "utilization", "p", "prob", "retention", "conversion", "mean_quality", "verified_quality"}


def unit_for(path: Iterable[str], module: str = "") -> str:
    """Unit label of a numeric field, from its name and context; values are never converted."""
    keys = [str(k) for k in path]
    k = keys[-1].lower() if keys else ""
    ctx = {x.lower() for x in keys[:-1]} | {module.lower()}
    if k.endswith("_cents") or k == "cents" or "balances" in ctx or k in ("balance", "amount", "fee", "net", "revenue",
                                                                        "cost", "charged", "shortfall"):
        return "cents"
    if k == "price":
        if "cloud" in ctx or "provider" in ctx or "instance" in ctx:
            return "cents/core-hour"
        return "cents/call"
    if k.endswith("_ms") or k == "ms" or k in ("cpu_ms", "latency_ms"):
        return "ms"
    if k.endswith("_s") or k in ("seconds", "cost_s", "time", "t", "tick_s", "elapsed_s"):
        return "s"
    if k.endswith("_days") or k == "days":
        return "days"
    if k.endswith("_h") or k == "hours":
        return "h"
    if k in ("rating", "stars"):
        return "stars"
    if k in _COUNT_KEYS or k.startswith("n_") or k.endswith("_count") or k.endswith("_n"):
        return "count"
    if k in _FRACTION_KEYS or k.endswith("_fraction") or k.endswith("_rate") or k.endswith("_share"):
        return "fraction"
    if k == "score":
        return "score"
    return "number"


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


DEFAULT_MAX_NUMERIC_SUMMARY = 48      # NUMERIC records per module summary
DEFAULT_MAX_NUMERIC_RESULT = 32       # NUMERIC records per tool result
NUMERIC_DEPTH = 3


def all_numeric_leaves(obj: Any, prefix: tuple = (), depth: int = NUMERIC_DEPTH) -> list[tuple[tuple, float]]:
    """(path, value) of every finite number in a nested dict down to `depth` levels (lists are not expanded: they
    are records, not measurements), breadth-first so top-level fields come first."""
    out: list[tuple[tuple, float]] = []
    frontier = [(prefix, obj)]
    for _ in range(depth):
        nxt = []
        for p, x in frontier:
            if not isinstance(x, dict):
                continue
            for k, v in x.items():
                if _is_num(v):
                    out.append((p + (str(k),), float(v)))
                elif isinstance(v, dict):
                    nxt.append((p + (str(k),), v))
        frontier = nxt
    return out


def numeric_leaves(obj: Any, prefix: tuple = (), limit: int = 32, depth: int = NUMERIC_DEPTH
                   ) -> list[tuple[tuple, float]]:
    """The first `limit` of `all_numeric_leaves` (callers that truncate must report it, see `_truncation_meta`)."""
    return all_numeric_leaves(obj, prefix, depth)[:max(0, int(limit))]


def _truncation_meta(n_total: int, limit: int) -> dict:
    """meta entries of the record carrying the full JSON when NUMERIC records were left out."""
    return {"numeric_total": int(n_total), "numeric_truncated": int(n_total - limit)} if n_total > limit else {}


def _entity_for_path(path: tuple, default: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    for k in reversed(path):
        ty = id_type(k)
        if ty is not None:
            return k, ty
    return default, None


def compact_json(x: Any) -> str:
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------------------------- events
MESSAGE_TYPES = ("mail.", "forum.post", "forum.reply", "market.rating", "support.", "chat.", "deal.message",
                 "contracts.message", "jobs.application", "ops.request", "ops.response")


def event_kind(etype: str) -> ObsKind:
    if etype == "bank.txn":
        return ObsKind.TRANSACTION
    if any(etype == m or (m.endswith(".") and etype.startswith(m)) for m in MESSAGE_TYPES):
        return ObsKind.MESSAGE
    return ObsKind.EVENT


def event_observation(ev: dict, own_accounts: set[str] | None = None, caused_by: Optional[str] = None,
                      own_principals: set[str] | None = None) -> Observation:
    """One actor-visible Crucible event ({seq, t, type, actor, payload}) as an Observation. Bank transactions carry
    the net cents moved into the actor's own accounts as ``value`` (unit cents) and the largest counterparty account
    as ``target``. ``caused_by`` is the action id of the actor's own command that emitted it (direct consequence,
    evidence CONTROLLED); delayed or third-party events are OBSERVATIONAL."""
    etype, payload = ev["type"], ev.get("payload")
    kind = event_kind(etype)
    ents = entities_in(payload)
    entity, etype_ent = (ents[0] if ents else (None, None))
    target = ents[1][0] if len(ents) > 1 else None
    value, unit = None, None
    if kind is ObsKind.TRANSACTION and isinstance(payload, dict) and own_accounts is not None:
        posts = [(a, int(d)) for a, d in payload.get("postings", []) if isinstance(d, (int, float))]
        mine = [(a, d) for a, d in posts if a in own_accounts]
        other = [(a, d) for a, d in posts if a not in own_accounts]
        if mine:
            value, unit = float(sum(d for _, d in mine)), "cents"
            entity, etype_ent = max(mine, key=lambda r: abs(r[1]))[0], "account"
        if other:
            target = max(other, key=lambda r: abs(r[1]))[0]
    meta = {"event_type": etype}
    if own_principals is not None and ev.get("actor") in own_principals:
        meta["own"] = True
    return Observation(kind=kind, content=compact_json({"type": etype, "actor": ev.get("actor"), "payload": payload}),
                       value=value, unit=unit, entity=entity, entity_type=etype_ent, target=target,
                       relation=etype if kind is ObsKind.TRANSACTION else None, t=float(ev["t"]) / DAY,
                       seq=int(ev["seq"]), source=str(ev.get("actor", "")),
                       evidence=EvidenceKind.CONTROLLED if caused_by else EvidenceKind.OBSERVATIONAL,
                       caused_by=caused_by, meta=meta)


def event_dict(obs: Observation) -> dict:
    """Inverse of `event_observation`: the compact event dict the actor saw."""
    d = json.loads(obs.content)
    return {"seq": obs.seq, "t": int(round(obs.t * DAY)), "type": d["type"], "actor": d["actor"],
            "payload": d["payload"]}


# ---------------------------------------------------------------------------------------------- summaries
def summary_observations(summary: dict, now_s: int, principal: str,
                         max_numeric: int = DEFAULT_MAX_NUMERIC_SUMMARY) -> list[Observation]:
    """Per-module summaries (`Module.observe`) as records: one TEXT observation per module carrying the module's
    canonical JSON (lossless), one NUMERIC per numeric leaf (balances in cents, unread counts, ...; at most
    `max_numeric` per module, the TEXT record's meta says how many more there were), and one RELATION per
    organisation the actor belongs to (role as the relation label)."""
    t = float(now_s) / DAY
    out: list[Observation] = []
    for mod in sorted(summary):
        val = summary[mod]
        leaves = all_numeric_leaves(val) if isinstance(val, dict) else []
        out.append(Observation(ObsKind.TEXT, content=compact_json(val), entity=principal, entity_type="person", t=t,
                               source=f"summary:{mod}",
                               meta={"summary_module": mod, **_truncation_meta(len(leaves), max_numeric)}))
        if not isinstance(val, dict):
            continue
        for path, v in leaves[:max(0, int(max_numeric))]:
            ent, ety = _entity_for_path(path, principal)
            out.append(Observation(ObsKind.NUMERIC, value=v, unit=unit_for(path, mod), entity=ent,
                                   entity_type=ety or ("person" if ent == principal else None), t=t,
                                   source=f"summary:{mod}", meta={"field": ".".join((mod,) + path)}))
        if mod == "identity" and isinstance(val.get("orgs"), dict):
            for org, role in sorted(val["orgs"].items()):
                out.append(Observation(ObsKind.RELATION, entity=principal, entity_type="person", target=org,
                                       relation=str(role), t=t, source="summary:identity",
                                       meta={"field": "identity.orgs"}))
    return out


def summary_dict(observations: Iterable[Observation]) -> dict:
    """Inverse of `summary_observations` (the module summaries, from the TEXT records)."""
    out = {}
    for o in observations:
        mod = o.meta.get("summary_module") if o.meta else None
        if mod and o.kind is ObsKind.TEXT:
            out[mod] = json.loads(o.content)
    return out


# ---------------------------------------------------------------------------------------------- tools
def tool_schema(spec: dict) -> ToolSchema:
    """An MCP-shaped Crucible tool spec as a ToolSchema (``costSeconds`` -> duration_hint in days). The schema and
    annotations are deep copies: the kernel hands out its endpoints' own dicts."""
    ann = copy.deepcopy(dict(spec.get("annotations") or {}))
    cost_s = ann.get("costSeconds")
    return ToolSchema(name=spec["name"], description=spec.get("description", ""),
                      parameters=copy.deepcopy(spec.get("inputSchema")) or {"type": "object", "properties": {}},
                      returns="JSON result {ok, result | error{code, message, details}}",
                      duration_hint=None if cost_s is None else float(cost_s) / DAY,
                      meta={"read_only": bool(ann.get("readOnlyHint", False)), "host": ann.get("host"),
                            "builtin": ann.get("builtin"), "version": ann.get("version"),
                            "cost_s": cost_s})


def tool_spec(ts: ToolSchema) -> dict:
    """Inverse of `tool_schema` (the MCP-shaped dict a Crucible `Policy` expects)."""
    ann = {"readOnlyHint": ts.meta.get("read_only", False), "costSeconds": ts.meta.get("cost_s"),
           "host": ts.meta.get("host"), "builtin": ts.meta.get("builtin"), "version": ts.meta.get("version")}
    return {"name": ts.name, "description": ts.description, "inputSchema": copy.deepcopy(ts.parameters),
            "annotations": {k: v for k, v in ann.items() if v is not None}}


# ---------------------------------------------------------------------------------------------- results
def result_observations(res: dict, tool: str, t_s: int, caused_by: Optional[str],
                        max_numeric: int = DEFAULT_MAX_NUMERIC_RESULT) -> list[Observation]:
    """A Crucible action result ({ok, t, tool, cost_s, result | error}) as TOOL_RESULT (complete JSON) + NUMERIC
    leaves of the result (at most `max_numeric`; the TOOL_RESULT meta reports any left out), or ERROR (code,
    message, details kept in meta for exact reconstruction)."""
    t = float(t_s) / DAY
    if res.get("ok"):
        body = res.get("result")
        ents = entities_in(body) if isinstance(body, (dict, list)) else []
        ent, ety = ents[0] if ents else (None, None)
        leaves = all_numeric_leaves(body) if isinstance(body, dict) else []
        out = [Observation(ObsKind.TOOL_RESULT, content=compact_json(body), entity=ent, entity_type=ety,
                           target=ents[1][0] if len(ents) > 1 else None, t=t, source=tool,
                           evidence=EvidenceKind.CONTROLLED, caused_by=caused_by,
                           meta={"tool": tool, "replayed": bool(res.get("replayed", False)),
                                 **_truncation_meta(len(leaves), max_numeric)})]
        if isinstance(body, dict):
            for path, v in leaves[:max(0, int(max_numeric))]:
                e2, ty2 = _entity_for_path(path, ent)
                out.append(Observation(ObsKind.NUMERIC, value=v, unit=unit_for(path, tool.split(".", 1)[0]),
                                       entity=e2, entity_type=ty2 or ety, t=t, source=tool,
                                       evidence=EvidenceKind.CONTROLLED, caused_by=caused_by,
                                       meta={"field": ".".join(path), "tool": tool}))
        return out
    err = res.get("error") or {}
    code = str(err.get("code", "error"))
    msg = str(err.get("message", ""))
    details = err.get("details")
    text = f"{code}: {msg}" + (f"\n{compact_json(details)}" if details else "")
    return [Observation(ObsKind.ERROR, content=text, t=t, source=tool or "parser", evidence=EvidenceKind.CONTROLLED,
                        caused_by=caused_by, meta={"tool": tool, "code": code, "message": msg, "details": details})]


RESULT_DELIVERY = "next_observation"


def delivered_copy(o: Observation) -> Observation:
    """The copy of a direct-result observation that an adapter delivers at the head of the next ObservationBatch
    (redesign plan 2026-11-26 §2 step 5, W2): same content and ``caused_by`` (the action id), marked with
    ``meta["result_of"]`` = the action id and ``meta["delivery"]`` so a loop that also merges
    `ExecutionResult.observations` can drop the duplicate (`is_delivered_result`, `result_key`)."""
    return replace(o, meta={**dict(o.meta or {}), "result_of": o.caused_by, "delivery": RESULT_DELIVERY})


def is_delivered_result(o: Observation) -> bool:
    return bool(o.meta) and o.meta.get("delivery") == RESULT_DELIVERY


def result_key(o: Observation) -> tuple:
    """Identity of a direct-result observation, equal for the ExecutionResult copy and the delivered copy."""
    return (o.kind.value, o.caused_by, o.source, o.t, o.value, (o.meta or {}).get("field"), o.content)


def result_dict(er: ExecutionResult, tool: str) -> dict:
    """Inverse of `result_observations` + ExecutionResult: the ActionResult dict a Crucible `Policy` expects."""
    d: dict[str, Any] = {"ok": bool(er.ok), "t": int(round(er.started_t * DAY)), "tool": er.meta.get("tool", tool),
                         "cost_s": int(round(er.duration * DAY))}
    if er.ok:
        tr = [o for o in er.observations if o.kind is ObsKind.TOOL_RESULT]
        d["result"] = json.loads(tr[0].content) if tr else None
        if tr and tr[0].meta.get("replayed"):
            d["replayed"] = True
    else:
        eo = [o for o in er.observations if o.kind is ObsKind.ERROR]
        if eo:
            m = eo[0].meta
            e = {"code": m.get("code"), "message": m.get("message", "")}
            if m.get("details") is not None:
                e["details"] = m["details"]
            d["error"] = e
        else:
            d["error"] = {"code": er.error or "error", "message": ""}
    return d
