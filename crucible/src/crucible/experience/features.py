"""Turning what a world shows an agent into the structured parts of an `Observation`.

Everything here reads only what the agent may see: its own module summaries (`World.observe(agent)`), the events
visible to it and its tool list. Nothing reads hidden state.

* `flatten` - exact numeric leaves of a summary as ``{"path": number}`` plus numeric vectors as arrays;
* `texts` - natural-language strings (with their source path);
* `entities` - ids the agent's summary and events mention, their kind and how they relate to the agent;
* `artifact_refs` - references to readable content carried by visible events (v5 content-addressed artifacts
  ``sha256:<hex>`` and 4.0 content objects: commits, repositories, posts, messages, pages, deployments, listings);
* `tool_category` - which of the 13 tool categories of MASTER_SPEC 11.2 a tool belongs to.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from .observation import exact

# ---------------------------------------------------------------------------------------------- flatten
_ID_RE = re.compile(r"^(?:[a-z]{1,8}:[A-Za-z0-9_.@:-]+|[a-z]{2,8}_[0-9A-Za-z_]*\d[0-9A-Za-z_]*|sha256:[0-9a-f]{64})$")


def is_id(s: Any) -> bool:
    return isinstance(s, str) and len(s) <= 128 and bool(_ID_RE.match(s))


def id_kind(s: str) -> str:
    if s.startswith("p:"):
        return "person"
    if s.startswith("o:"):
        return "organisation"
    if s.startswith("svc:"):
        return "service"
    if s.startswith("sha256:"):
        return "artifact"
    if s.startswith("acct_"):
        return "account"
    m = re.match(r"^([a-z]{1,8})[:_]", s)
    return m.group(1) if m else "entity"


def flatten(x: Any, prefix: str = "", numbers: dict | None = None, arrays: dict | None = None, depth: int = 0,
            max_list: int = 4096) -> tuple[dict, dict]:
    """Exact numeric leaves (`numbers`) and numeric lists (`arrays`) of a JSON-like value, keyed by dotted path.
    Booleans become 0/1. Lists of dicts are indexed (``path[3].field``)."""
    numbers = {} if numbers is None else numbers
    arrays = {} if arrays is None else arrays
    if depth > 12:
        return numbers, arrays
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            flatten(x[k], f"{prefix}.{k}" if prefix else str(k), numbers, arrays, depth + 1, max_list)
    elif isinstance(x, (list, tuple)):
        if x and all(exact(v) is not None for v in x):
            arrays[prefix] = [exact(v) for v in list(x)[:max_list]]
        else:
            for i, v in enumerate(list(x)[:max_list]):
                if isinstance(v, (dict, list, tuple)):
                    flatten(v, f"{prefix}[{i}]", numbers, arrays, depth + 1, max_list)
                elif exact(v) is not None:
                    numbers[f"{prefix}[{i}]"] = exact(v)
    else:
        v = exact(x)
        if v is not None:
            numbers[prefix] = v
    return numbers, arrays


TEXT_KEYS = ("title", "body", "subject", "text", "message", "description", "note", "brief", "content", "summary",
             "reason", "doc", "name")


def texts(x: Any, prefix: str, out: list[str], depth: int = 0, keys: Iterable[str] | None = None) -> list[str]:
    """Natural-language strings in a value: strings with a space in them (or under a text key), as
    ``"path: text"``."""
    if depth > 10:
        return out
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            v = x[k]
            p = f"{prefix}.{k}"
            if isinstance(v, str):
                if (" " in v.strip() or (k in TEXT_KEYS and v and not is_id(v))) and len(v) <= 20000:
                    out.append(f"{p}: {v}")
            else:
                texts(v, p, out, depth + 1)
    elif isinstance(x, list):
        for i, v in enumerate(x[:200]):
            if isinstance(v, str):
                if " " in v.strip() and len(v) <= 20000:
                    out.append(f"{prefix}[{i}]: {v}")
            else:
                texts(v, f"{prefix}[{i}]", out, depth + 1)
    return out


# ---------------------------------------------------------------------------------------------- entities
def _ids_in(x: Any, path: str, out: list[tuple[str, str]], depth: int = 0) -> None:
    if depth > 8:
        return
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            if is_id(k):
                out.append((str(k), path))
            _ids_in(x[k], f"{path}.{k}" if not is_id(k) else path, out, depth + 1)
    elif isinstance(x, list):
        for v in x[:200]:
            _ids_in(v, path, out, depth + 1)
    elif is_id(x):
        out.append((x, path))


def entities(agent: str, summary: dict, events: list[dict], memberships: Iterable[str] = ()) -> list[dict]:
    """Entities the agent's own summary and its visible events mention, with their relation to the agent.

    The relation is named after where the id appeared: ``bank.balances`` -> the agent holds that account,
    ``identity.orgs`` -> it belongs to that organisation, an event's actor -> the agent observed that actor act."""
    found: dict[str, dict] = {}

    def add(eid: str, rel: str, attrs: dict | None = None) -> None:
        if eid == agent:
            return
        e = found.setdefault(eid, {"id": eid, "kind": id_kind(eid), "relations": [], "attrs": {}})
        if not any(r["rel"] == rel for r in e["relations"]):
            e["relations"].append({"rel": rel, "to": agent})
        if attrs:
            e["attrs"].update(attrs)

    for org in sorted(memberships):
        if org != agent:
            add(org, "member_or_admin_of")
    pairs: list[tuple[str, str]] = []
    _ids_in(summary, "", pairs)
    for eid, path in pairs:
        rel = path.strip(".") or "mentioned_in_summary"
        attrs = None
        if path.strip(".") == "bank.balances":
            bal = exact(((summary.get("bank") or {}).get("balances") or {}).get(eid))
            attrs = {"balance_cents": bal} if bal is not None else None
        add(eid, rel, attrs)
    for ev in events:
        a = ev.get("actor")
        if isinstance(a, str) and a not in ("sys", agent) and is_id(a):
            add(a, f"acted:{ev.get('type')}")
        pl = ev.get("payload")
        if isinstance(pl, dict):
            for k in ("from", "to", "seller", "buyer", "owner", "author", "holder", "counterparty", "payer", "payee",
                      "org", "host"):
                v = pl.get(k)
                if is_id(v) and v != agent:
                    add(v, f"{ev.get('type')}.{k}")
    me = {"id": agent, "kind": id_kind(agent), "attrs": {},
          "relations": [{"rel": r["rel"], "to": e["id"]} for e in found.values() for r in e["relations"][:1]]}
    return [me] + [found[k] for k in sorted(found)]


# ---------------------------------------------------------------------------------------------- artifacts
_REF_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# 4.0 content objects carried by visible events: event type -> (kind, id field, ref prefix, owner field, title field)
CONTENT_EVENTS = {
    "code.repo": ("repository", "id", "repo", "owner", "description"),
    "code.commit": ("commit", "id", "commit", None, "message"),
    "forum.post": ("post", "id", "post", "author", "title"),
    "forum.reply": ("reply", "id", "reply", "author", None),
    "mail.sent": ("message", "id", "mail", "from", "subject"),
    "mail.delivered": ("message", "msg", "mail", "from", "subject"),
    "net.page": ("page", "url", "url", None, "title"),
    "cloud.deployment": ("deployment", "id", "deployment", "owner", "host"),
    "market.listing": ("listing", "id", "listing", "seller", "title"),
    "docs.create": ("document", "id", "doc", "owner", "title"),
    "docs.created": ("document", "id", "doc", "owner", "title"),
}


def artifact_refs(ev: dict) -> list[dict]:
    """Artifact references an event carries. A v5 content-addressed reference (``sha256:<hex>`` under ``ref``) is
    taken as is; a 4.0 content object becomes ``"<prefix>:<id>"``."""
    out = []
    pl = ev.get("payload")
    if not isinstance(pl, dict):
        return out
    typ = ev.get("type", "")
    ref = pl.get("ref")
    if isinstance(ref, str) and _REF_RE.match(ref):
        a = {"ref": ref, "kind": str(pl.get("kind") or typ.split(".")[0]), "event": typ, "t": ev.get("t")}
        for k in ("mime", "size", "owner", "audience", "title"):
            if k in pl and isinstance(pl[k], (str, int, float)):
                a[k] = pl[k]
        out.append(a)
    spec = CONTENT_EVENTS.get(typ)
    if spec is not None:
        kind, idf, prefix, ownerf, titlef = spec
        oid = pl.get(idf)
        if isinstance(oid, (str, int)):
            a = {"ref": f"{prefix}:{oid}", "kind": kind, "event": typ, "t": ev.get("t")}
            if ownerf and isinstance(pl.get(ownerf), str):
                a["owner"] = pl[ownerf]
            if titlef and isinstance(pl.get(titlef), str):
                a["title"] = pl[titlef][:200]
            if typ == "code.commit" and isinstance(pl.get("repo"), str):
                a["repo"] = pl["repo"]
            out.append(a)
    return out


# ---------------------------------------------------------------------------------------------- tool categories
TOOL_CATEGORIES = ("browser", "search", "code_execution", "file_creation", "program_deployment", "communication",
                   "financial", "research", "experiments", "organization", "content_creation", "tool_composition",
                   "agent_services")

_BY_TOOL = {
    "code.run": "code_execution", "code.test": "code_execution",
    "code.commit": "file_creation", "code.create_repo": "file_creation", "code.fork": "file_creation",
    "artifact.put": "file_creation", "db.put": "file_creation",
    "cloud.deploy": "program_deployment", "cloud.undeploy": "program_deployment", "cloud.rent": "program_deployment",
    "cloud.resize": "program_deployment", "cloud.release": "program_deployment", "pkg.release": "program_deployment",
    "world.find_tools": "search", "caps.search": "search",
    "rig.hypothesize": "experiments", "lab.experiment": "experiments", "lab.propose_change": "experiments",
    "market.list": "financial", "market.buy": "financial", "market.refund": "financial",
    "forum.post": "content_creation", "forum.reply": "content_creation", "forum.vote": "content_creation",
}
_BY_HOST = {
    "net": "browser", "web": "browser",
    "search": "search",
    "sandbox": "code_execution",
    "docs": "content_creation", "ads": "content_creation", "content": "content_creation", "artifact": "file_creation",
    "cloud": "program_deployment", "pkg": "program_deployment",
    "mail": "communication", "forum": "communication", "deal": "communication", "cal": "communication",
    "social": "communication", "msg": "communication", "support": "communication",
    "bank": "financial", "market": "financial", "contracts": "financial", "subs": "financial",
    "research": "research", "lab": "research", "use": "research", "rig": "research", "make": "research",
    "exp": "experiments", "goals": "experiments",
    "id": "organization", "biz": "organization", "gov": "organization", "jobs": "organization",
    "issues": "organization", "ops": "organization",
    "wf": "tool_composition", "caps": "tool_composition",
    "code": "file_creation", "db": "file_creation",
    # 5.0 hosts
    "prod": "financial", "xchg": "financial", "venture": "financial", "commit": "financial", "asset": "financial",
    "inst": "organization", "labor": "organization", "needs": "research",
    "media": "content_creation", "embed": "research", "artifacts": "file_creation", "art": "file_creation",
}
# Tier 1 rapid action kinds (`rapid.<kind>`): what each kind of action is
_RAPID = {"levers": "experiments", "discrete": "experiments", "allocate": "financial", "experiment": "experiments",
          "query": "research", "message": "communication", "program": "code_execution", "graph_edit": "organization",
          "sequence": "tool_composition", "coordinate": "communication"}


def tool_category(spec: dict) -> str:
    """One category per tool; agent-deployed endpoints (``annotations.builtin`` false) are `agent_services`."""
    ann = spec.get("annotations") or {}
    if ann.get("builtin") is False:
        return "agent_services"
    name = spec.get("name", "")
    if name in _BY_TOOL:
        return _BY_TOOL[name]
    host = ann.get("host") or name.split(".", 1)[0]
    if host == "rapid":
        return _RAPID.get(name.split(".", 1)[-1], "experiments")
    return _BY_HOST.get(host, "discovery" if host == "world" else "other")


def annotate_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        t2 = dict(t)
        ann = dict(t2.get("annotations") or {})
        ann["category"] = tool_category(t)
        t2["annotations"] = ann
        out.append(t2)
    return out
