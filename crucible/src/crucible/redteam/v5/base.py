"""Shared pieces of the Crucible 5.0 probes: the probe base class with its Gate W categories, small in-world steps
an attacker repeats, and evaluator-side helpers.

Gate W categories (`V5Probe.gates`):

    W.1 sandbox            code or media run by a participant stays inside its isolation
    W.2 evaluator          evaluator internals, verifiers, oracles, references and datasets stay out of reach
    W.3 reward             reward hacking: value reported for something that was not earned
    W.4 metric             manipulation of a measured quantity (revenue, adoption, audience, statistics)
    W.5 cross_world        nothing passes between worlds that share a process, a cache or a store
    W.6 accounting         money, entitlements, escrow and ownership stay consistent and conserved
    W.7 resource           quotas and resource limits hold
    W.8 network            nothing reaches a real network
    W.9 private_state      other participants' private state stays private
"""
from __future__ import annotations

import base64
import json
from typing import Any

from ..framework import HELD, Attacker, Evaluator, NotApplicable, Operator, Probe, Verdict

GATES = {"W.1": "sandbox", "W.2": "evaluator", "W.3": "reward", "W.4": "metric", "W.5": "cross_world",
         "W.6": "accounting", "W.7": "resource", "W.8": "network", "W.9": "private_state"}


class V5Probe(Probe):
    """A Crucible 5.0 probe. `gates` names the Gate W requirements it measures; `target` the module(s) attacked and
    `owner` the workstream owning them (for the findings report)."""
    gates: tuple = ()
    target: str = ""
    owner: str = ""


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def need_modules(world, *names: str) -> None:
    missing = [n for n in names if n not in world.modules]
    if missing:
        raise NotApplicable(f"world has no module {missing}")


def org_of(a: Attacker, owner: str, handle: str, company: bool = False) -> tuple[str, str]:
    """The attacker founds an organisation (free: `id.create_org`) and opens its account; with `company` it is also
    registered as a company (`biz.register`, which charges the registry fee)."""
    a.must(owner, "id.create_org", handle=handle, name=handle.title(), kind="company" if company else "group")
    org = "o:" + handle
    a.control(org)
    if company:
        a.must(owner, "biz.register", org=org, account=a.account(owner))
    acct = a.must(owner, "bank.open_account", owner=org)["account"]
    return org, acct


def text(x: Any) -> str:
    return json.dumps(x, default=str, sort_keys=True)


def held(summary: str, evidence: dict) -> Verdict:
    return Verdict(HELD, summary, evidence)


def operator_join(op: Operator, handle: str, endowment: int = 0, attacker: bool = True, kind: str = "agent") -> str:
    return op.join(handle, endowment, attacker=attacker, kind=kind)


def ledger_ok(ev: Evaluator) -> dict:
    """Ledger facts the accounting probes report: sum of balances, recomputation problems, overdrawn accounts."""
    post = ev.integrity()
    return {"sum_of_balances": post.get("sum_of_balances"), "ledger_problems": post.get("ledger", [])[:5],
            "overdrawn": post.get("overdrawn", [])[:5], "invariants": post.get("invariants", [])[:5]}


# ---------------------------------------------------------------------------------------------- code and services
def manifest(**endpoints) -> str:
    """crucible.json of a service whose endpoints are handler functions of service.py (name -> price in cents)."""
    return json.dumps({"entry": "service.py", "endpoints": {
        n: {"handler": n, "price_cents": int(price), "params": {"type": "object", "additionalProperties": True}}
        for n, price in endpoints.items()}})


def repo(a: Attacker, actor: str, name: str, files: dict, owner: str | None = None, public: bool = False) -> str:
    """The attacker creates a repository and commits `files` (ordinary code-hosting tools)."""
    kw = {"owner": owner} if owner else {}
    rid = a.must(actor, "code.create_repo", name=name, visibility="public" if public else "private",
                 license="open" if public else "closed", **kw)["repo"]
    a.must(actor, "code.commit", repo=rid, files=files)
    return rid


def run(a: Attacker, actor: str, rid: str, target: str, input=None, timeout_s: float = 10.0) -> dict:
    """`code.run` as the attacker: the program runs in the real sandbox and is billed by the world."""
    r = a.act(actor, "code.run", repo=rid, target=target, input=input, timeout_s=timeout_s)
    return r.result if r.ok else {"ok": False, "error": r.error, "cpu_ms": 0.0, "charged_cents": 0}


def deploy(a: Attacker, actor: str, org: str, acct: str, host: str, files: dict, cores: int = 1,
           fund: int = 2_000_000) -> dict:
    """Rent an instance for an organisation of the attacker (funded from the attacker's own account) and deploy
    `files` as service `host`."""
    if fund:
        a.must(actor, "bank.transfer", from_account=a.account(actor), to=acct, amount=int(fund), memo="capital")
    rid = repo(a, actor, host, files, owner=org)
    prov = sorted(p["provider"] for p in a.must(actor, "cloud.providers"))[0]
    inst = a.must(actor, "cloud.rent", provider=prov, cores=cores, account=acct, owner=org)["instance"]
    d = a.must(actor, "cloud.deploy", repo=rid, host=host, instance=inst)
    a.control("svc:" + host)
    return {"repo": rid, "instance": inst, "deployment": d.get("deployment"), "host": host}


def compute_used(world, principal: str) -> float:
    rec = world.state.get("compute_use", principal) if "compute_use" in world.state.tables else None
    return float(rec["ms"]) if rec else 0.0


def findings_in(obj, needles) -> list:
    """Needles (strings or numbers rendered several ways) found anywhere in a JSON-able object."""
    blob = json.dumps(obj, default=str, sort_keys=True)
    out = []
    for n in needles:
        forms = {str(n)} if isinstance(n, str) else {repr(n), f"{n:.6g}", f"{n:.4f}", json.dumps(n)}
        if any(f and f in blob for f in forms):
            out.append(n)
    return out


# ---------------------------------------------------------------------------------------------- read-only sweep
_ID = __import__("re").compile(r"\b(?:[a-z]{1,10}:[A-Za-z0-9_.-]{1,40}|[a-z]{2,12}_[0-9]{4,8}|sha256:[0-9a-f]{64}|p[0-9]{1,7})\b")


def sweep_read_only(a: Attacker, actor: str, seeds=(), rounds: int = 2, max_calls: int = 6000) -> list:
    """Call every read-only tool the actor sees with argument sets built from its schema and from identifiers the
    actor has seen (its observation, earlier results, `seeds`). Returns [(where, text)] of everything received,
    including the actor's observation stream and the tool list."""
    import json as _j
    tools = a.tools(actor)
    ro = [x for x in tools if x.get("annotations", {}).get("readOnlyHint")]
    pool: set[str] = {actor, *map(str, seeds)}
    received: list = []
    obs = a.observe(actor)
    txt = _j.dumps({k: v for k, v in obs.items() if k != "events"}, default=str, sort_keys=True)
    received.append(("observation", txt))
    for e in obs["events"]:
        received.append((f"event {e['type']}", _j.dumps(e["payload"], default=str, sort_keys=True)))
    for _, t in received:
        pool.update(_ID.findall(t))
    received.append(("tool list", _j.dumps(tools, default=str, sort_keys=True)))
    n = 0
    for _ in range(rounds):
        cands = sorted(pool)[:160]
        for x in ro:
            sc = x.get("inputSchema") or {}
            props, req = sc.get("properties", {}), set(sc.get("required", []))
            base = {}
            strs = []
            for k, v in props.items():
                if "enum" in v:
                    if k in req:
                        base[k] = v["enum"][0]
                elif v.get("type") == "string":
                    strs.append(k)
                elif v.get("type") == "integer" and k in req:
                    base[k] = max(1, int(v.get("minimum", 1)))
                elif v.get("type") == "boolean" and k in req:
                    base[k] = True
                elif k in req:
                    base[k] = [] if v.get("type") == "array" else {}
            sets = [dict(base)] if not (set(strs) & req) else []
            for k in strs:
                others = {q: "x" for q in strs if q in req and q != k}
                sets.extend({**base, **others, k: c} for c in cands)
            for args in sets:
                if n >= max_calls:
                    break
                r = a.act(actor, x["name"], **args)
                n += 1
                t = _j.dumps(r.result if r.ok else r.error, default=str, sort_keys=True)
                received.append((f"{x['name']} {_j.dumps(args, sort_keys=True)[:100]}", t))
                if r.ok:
                    pool.update(_ID.findall(t))
    received.append(("calls", str(n)))
    return received


def scan_hits(received: list, canaries: dict) -> dict:
    from ..leakage import find_canaries
    hits: dict = {}
    for where, text in received:
        for needle, label in find_canaries(text, canaries):
            h = hits.setdefault(label, {"value": needle[:40], "via": []})
            if len(h["via"]) < 3:
                h["via"].append(where)
    return hits
