"""Evaluator-side measurement for Gate J (open-ended digital creation), Gate L (endogenous opportunities), SP-6.3,
SP-6.4 and SP-6.5. Everything here reads authoritative state and the event log; nothing is a tool and nothing is
reachable by in-world actors.

The helpers are written for any world that contains the workstream-G modules (`caps` with the registry, `pkg`)
and the population's work activities (`economy.work`): participants are individuals of the one population.
"Independent" is computed here, at evaluation time, never in-world: not related to the creator (4.0 related
parties closed under shared controllers and money control, `commerce.integrity.related_set`) and not an unlabelled
admitted persona. Creators admitted with a controller label starting `scripted:` are scripted reference policies:
their products count for feasibility only and are reported as scripted (`creator_kind`).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.kernel import SYSTEM, World

DAY = 86400
SYSTEM_ACCOUNTS = ("acct_genesis", "acct_external", "acct_fees")


# ====================================================================== principals
def is_agent(world: World, principal: str) -> bool:
    """An in-world actor admitted as an agent or human (admin.join kind agent|human), or an organisation one of
    them founded or belongs to. World-generated participants (kind npc) and the system are not agents."""
    st = world.state
    if principal.startswith("p:"):
        p = st.get("persons", principal)
        return bool(p) and p.get("kind") in ("agent", "human")
    if principal.startswith("o:"):
        o = st.get("orgs", principal)
        if o and o.get("founder") and is_agent(world, o["founder"]):
            return True
        return any(is_agent(world, m["person"]) for m in st.find("members", "org", principal))
    return False


def creator_kind(world: World, principal: str) -> str:
    """"scripted" (controller label scripted:...), "agent" (admitted agent or human), "npc" or "population"."""
    st = world.state
    who = [principal]
    if principal.startswith("o:"):
        o = st.get("orgs", principal) or {}
        who = [o.get("founder")] + [m["person"] for m in st.find("members", "org", principal)]
    kinds = set()
    for p in who:
        rec = st.get("persons", p) if p else None
        if rec is None:
            continue
        if str((rec.get("info") or {}).get("controller") or "").startswith("scripted:"):
            kinds.add("scripted")
        elif rec.get("kind") in ("agent", "human"):
            kinds.add("agent")
        else:
            kinds.add(rec.get("kind") or "npc")
    for k in ("agent", "scripted", "population", "npc"):
        if k in kinds:
            return k
    return "npc"


def independent(world: World, creator: str, other: str) -> bool:
    from ..commerce.integrity import independent_customer, related_set
    if other in (SYSTEM, None) or other.startswith("svc:"):
        return False
    cache = world.runtime.setdefault("measure.rel", {})
    key = (creator, world.seq)
    if key not in cache:
        if len(cache) > 64:
            cache.clear()
        cache[key] = related_set(world, creator)
    return independent_customer(world, creator, other, cache[key])


# ====================================================================== products (Gate J)
def agent_products(world: World) -> list[dict]:
    """Products created by in-world agents: typed services (capability declarations), software packages (with a
    release) and web applications. `executable` means it actually executed successfully for someone: a service
    answered at least one request with status 200 after its declaration; a package's code ran inside a
    dependent's successful execution or its release imported and its tests ran in the sandbox."""
    st = world.state
    out = []
    first: dict[tuple, str] = {}
    if "cap_decls" in st.tables:
        for did, d in st.scan("cap_decls"):
            first.setdefault((d["host"], d["name"]), did)
        for (host, name), did in sorted(first.items()):
            d = st.tables["cap_decls"][did]
            if not is_agent(world, d["provider"]):
                continue
            ok_after = sum(r["ok"] for k, r in st.scan("svc_days") if r["host"] == host and int(k.split("|")[1]) >=
                           d["t"] // DAY)
            live = world.modules["cloud"].live_deployment(st, host) is not None
            active = any(x["status"] == "active" for _, x in st.scan("cap_decls") if x["host"] == host)
            out.append({"id": f"service:{host}:{name}", "kind": "service", "category": "api_service",
                        "subcategory": name, "creator": d["provider"], "created_t": d["t"], "host": host,
                        "declarations": sorted(i for i, x in st.scan("cap_decls") if x["host"] == host),
                        "published": bool(live and active), "executable": ok_after > 0, "ok_requests": int(ok_after),
                        "creator_kind": creator_kind(world, d["provider"])})
    if "pkg_packages" in st.tables:
        pkg = world.modules["pkg"]
        for name, p in st.scan("pkg_packages"):
            if not is_agent(world, p["owner"]):
                continue
            rels = pkg.releases(st, name)
            executed = any(a["executions"] > 0 for a in st.find("pkg_adoptions", "package", name))
            out.append({"id": f"package:{name}", "kind": "package", "category": "software_package",
                        "subcategory": name, "creator": p["owner"], "created_t": p["t"], "package": name,
                        "published": bool(rels), "executable": executed or any(r["api"] for r in rels),
                        "releases": len(rels), "creator_kind": creator_kind(world, p["owner"])})
    if "deployments" in st.tables:
        for did, dep in st.scan("deployments"):
            if dep["manifest"].get("web") and is_agent(world, dep["owner"]):
                ok = sum(r["ok"] for k, r in st.scan("svc_days") if r["host"] == dep["host"])
                out.append({"id": f"webapp:{dep['host']}", "kind": "webapp", "category": "web_app",
                            "subcategory": dep["host"], "creator": dep["owner"], "created_t": dep["created"],
                            "host": dep["host"], "published": dep["status"] == "live", "executable": ok > 0})
    return out


def _provider_ids(world: World, product: dict) -> set[str]:
    """Ids under which participants rely on the product: its declarations and any product selling its host."""
    st = world.state
    ids = set(product.get("declarations", []))
    host = product.get("host")
    if host and "products" in st.tables:
        ids |= {pid for pid, p in st.scan("products") if (p["delivery"] or {}).get("host") == host}
    return ids


def discoverers(world: World, product: dict) -> list[str]:
    """Independent participants that *considered* the product after it existed: individuals of the population
    who read it as a candidate for one of their tasks (`work.considered`, after the interface check), tried it,
    or principals that tried it through `caps.try`. A bare search hit does not count."""
    st = world.state
    who: set[str] = set()
    provs = _provider_ids(world, product)
    for ev in world.log:
        if ev.type == "work.considered" and ev.t >= product["created_t"]:
            who.update(f"p{i}" for i, prov in ev.payload["rows"] if prov in provs)
    if product["kind"] == "service":
        decls = set(product.get("declarations", []))
        who.update(t["by"] for _, t in st.scan("cap_trials") if t["decl"] in decls)
    elif product["kind"] == "package":
        who.update(r["by"] for r in st.find("pkg_seen", "package", product["package"]))
    return sorted(p for p in who if independent(world, product["creator"], p))


def users(world: World, product: dict) -> list[str]:
    """Independent participants for whom the product actually executed on their own input: work-activity uses,
    trials and adopted use of a declared service, product use, or owners of repositories whose own successful
    execution depended on the package (parent-side ablation)."""
    st = world.state
    who: set[str] = set()
    provs = _provider_ids(world, product)
    if "work_exp" in st.tables:
        who.update(f"p{r['i']}" for _, r in st.scan("work_exp") if r["prov"] in provs and r["uses"] > 0)
    if product["kind"] == "service":
        decls = set(product.get("declarations", []))
        who.update(t["by"] for _, t in st.scan("cap_trials") if t["decl"] in decls and t["status"] in (200, 500))
        who.update(a["adopter"] for _, a in st.scan("cap_adoptions") if a["decl"] in decls and a["uses"] > 0)
        if "prod_use_holders" in st.tables:
            who.update(h["holder"] for _, h in st.scan("prod_use_holders") if h["product"] in provs and h["uses"] > 0)
    elif product["kind"] == "package":
        who.update(a["owner"] for a in st.find("pkg_adoptions", "package", product["package"]))
    return sorted(p for p in who if independent(world, product["creator"], p))


def adopters(world: World, product: dict) -> list[str]:
    st = world.state
    who: set[str] = set()
    provs = _provider_ids(world, product)
    if "work_exp" in st.tables:
        who.update(f"p{r['i']}" for _, r in st.scan("work_exp") if r["prov"] in provs and r.get("since") is not None)
    if product["kind"] == "service":
        decls = set(product.get("declarations", []))
        who |= {a["adopter"] for _, a in st.scan("cap_adoptions") if a["decl"] in decls}
    elif product["kind"] == "package":
        who |= {a["owner"] for a in st.find("pkg_adoptions", "package", product["package"])}
    return sorted(p for p in who if independent(world, product["creator"], p))


def independent_transactions(world: World, product: dict) -> dict:
    """Genuine transactions paying the creator for the product: per-call sales of its host (failed calls are
    refunded by the cloud and netted here) and subscription invoices / purchases of products selling it, net of
    refunds, from payers independent of the creator, weighted by the share of each payer's money that is not
    traced or netted back to the creator (`commerce.integrity.payer_weights`). Returns {n, cents, payers}."""
    from ..commerce.integrity import payer_weights
    st = world.state
    if product["kind"] not in ("service", "webapp"):
        return {"n": 0, "cents": 0, "payers": 0}
    creator = product["creator"]
    weights, _ = payer_weights(world, creator)
    ref = f"svc:{product['host']}"
    provs = _provider_ids(world, product)
    net: dict[str, int] = {}
    count: dict[str, int] = {}
    mine = set(st.find_ids("accounts", "owner", creator))
    for tid, t in st.scan("txns"):
        r = t["ref"]
        is_call = r == ref and t["kind"] in ("sale", "refund")
        is_sub = False
        if not is_call and isinstance(r, str) and t["kind"] in ("subscription", "refund", "product_sale", "license"):
            inv = st.get("sub_invoices", r) if "sub_invoices" in st.tables else None
            pur = st.get("prod_purchases", r) if "prod_purchases" in st.tables else None
            is_sub = (inv is not None and inv["product"] in provs) or (pur is not None and pur["product"] in provs)
        if not (is_call or is_sub):
            continue
        for a, d in t["postings"]:
            if a in mine or a in SYSTEM_ACCOUNTS:
                continue
            payer = st.tables["accounts"][a]["owner"]
            if not independent(world, creator, payer):
                continue
            net[payer] = net.get(payer, 0) - d
            if d < 0 and t["kind"] != "refund":
                count[payer] = count.get(payer, 0) + 1
    payers = {p for p, c in net.items() if c > 0}
    cents = sum(float(net[p]) * float(weights.get(p, 1)) for p in payers)
    return {"n": int(sum(count.get(p, 0) for p in payers)), "cents": int(cents), "payers": len(payers)}


def compute_use(world: World, since_t: int = 0) -> dict:
    """World compute actually consumed: compute units metered on every service (svc_hours / svc_days), CPU
    milliseconds, and compute rent paid to providers (ledger kind `compute`) since `since_t`."""
    st = world.state
    cu = sum(r["cu"] for k, r in st.scan("svc_days") if int(k.rsplit("|", 1)[1]) >= since_t // DAY)
    cpu = sum(r["cpu_ms"] for k, r in st.scan("svc_days") if int(k.rsplit("|", 1)[1]) >= since_t // DAY)
    rent = 0
    for tid in st.find_ids("txns", "kind", "compute"):
        t = st.tables["txns"][tid]
        if t["t"] >= since_t:
            rent += sum(d for _, d in t["postings"] if d > 0)
    return {"cu": round(cu, 3), "cpu_ms": round(cpu, 3), "compute_rent_cents": int(rent)}


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test p-value for the 2x2 table [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2

    def logc(x, y):
        return math.lgamma(x + 1) - math.lgamma(y + 1) - math.lgamma(x - y + 1)

    def p(x):
        return math.exp(logc(n1, x) + logc(n2, k - x) - logc(n, k))
    lo, hi = max(0, k - n2), min(k, n1)
    p0 = p(a)
    return float(min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9))))


# ====================================================================== needs (Gate L, SP-6.4)
def first_update_t(world: World) -> int | None:
    t = world.state.meta_get("needs:first")
    if t is not None:
        return t
    for ev in world.log:
        if ev.type == "needs.ran":
            return ev.t
    return None


def emerged_needs(world: World) -> list[dict]:
    """Needs first derived after the needs graph's first update, i.e. that arose from a change of world state
    rather than being present when the world began."""
    st = world.state
    if "needs_open" not in st.tables:
        return []
    t0 = first_update_t(world)
    return [dict(n) for _, n in st.scan("needs_open") if t0 is not None and n["t"] > t0]


def needs_addressed_by_new_services(world: World) -> list[dict]:
    """Emerged needs addressed by something created after the need arose (a new declared service or a fixed
    release), where at least one holder then adopted the new service (or upgraded off the affected version)."""
    st = world.state
    out = []
    for n in emerged_needs(world):
        if not n["addressed_by"]:
            continue
        for a in n["addressed_by"]:
            if a.get("host"):
                ad = [x for x in st.find("cap_adoptions", "decl", a["by"]) if x["adopter"] in n["holders_ever"]
                      and x["t"] >= n["t"]]
                if ad:
                    out.append({"need": n["id"], "kind": n["kind"], "target": n["target"], "by": a["by"],
                                "host": a["host"], "holders_adopting": len({x["adopter"] for x in ad}),
                                "status": n["status"]})
                    break
            else:
                pkgn = n["target"].split(":", 1)[1]
                moved = [d for d in st.find("pkg_deps", "package", pkgn) if d["owner"] in n["holders_ever"]
                         and d["t"] >= n["t"]]
                if moved:
                    out.append({"need": n["id"], "kind": n["kind"], "target": n["target"], "by": a["by"],
                                "host": None, "holders_adopting": len(moved), "status": n["status"]})
                    break
    return out


# ====================================================================== software ecosystem (SP-6.5)
SOFTWARE_FEATURES = ("hosting", "package_discovery", "dependency_graph", "issue_tracker", "pull_requests",
                     "independent_review", "testing", "releases", "documentation", "deployment",
                     "developer_adoption", "maintenance", "forks", "compatibility", "security_reliability_failures")


def software_features(world: World) -> dict:
    """Which of the fifteen Part 6.5 mechanisms have actually happened in this world (each from recorded state)."""
    st = world.state
    pk = st.tables.get("pkg_packages", {})
    projects = {p["project"] for p in pk.values()}
    issues = [i for i in st.tables.get("iss_issues", {}).values() if i["project"] in projects]
    deps = [d for d in st.tables.get("pkg_deps", {}).values() if d["status"] == "active"]
    dep_repos = {d["repo"] for d in deps}
    pkg_repos = {p["repo"]: n for n, p in pk.items()}
    transitive = any(r in pkg_repos for r in dep_repos)       # a package that itself depends on a package
    deploys = st.tables.get("pkg_deploys", {})
    served = [d for d in deploys.values() if d["packages"] and any(
        r["ok"] > 0 for k, r in st.scan("svc_days") if r["host"] == d["host"])]
    rel = list(st.tables.get("pkg_releases", {}).values())
    advs = [a for a in st.tables.get("pkg_advisories", {}).values() if a["verified"]]
    closed_by_maint = [i for i in issues if i["status"] == "closed"]
    fixed = [a for a in advs if a["fixed_in"]]
    compat = list(st.tables.get("pkg_compat", {}).values())
    ev = {
        "hosting": len(pk),
        "package_discovery": len(st.tables.get("pkg_seen", {})),
        "dependency_graph": len(deps) + (1 if transitive else 0),
        "issue_tracker": len(issues),
        "pull_requests": len(st.tables.get("pkg_prs", {})),
        "independent_review": sum(1 for r in st.tables.get("pkg_reviews", {}).values() if r["independent"]),
        "testing": len(st.tables.get("pkg_execs", {})) + sum(1 for r in rel if r["tests"]["passed"] + r["tests"]["failed"]),
        "releases": len(rel),
        "documentation": sum(1 for r in rel if r["docs"]),
        "deployment": len(served),
        "developer_adoption": len(adoptions_by_execution(world)),
        "maintenance": len(closed_by_maint) + len(fixed) + sum(1 for p in st.tables.get("pkg_prs", {}).values()
                                                              if p["status"] == "merged"),
        "forks": int(st.meta_get("pkg:forks", 0)),
        "compatibility": len(compat) + sum(1 for r in rel if r["compat"]["previous"]),
        "security_reliability_failures": len(advs) + sum(1 for r in rel if r["tests"]["failed"]),
    }
    return {"evidence": ev, "present": sorted(k for k, v in ev.items() if v > 0),
            "count": sum(1 for v in ev.values() if v > 0), "transitive_dependencies": transitive,
            "fixed_advisories": len(fixed)}


def adoptions_by_execution(world: World) -> list[dict]:
    """(package, dependent repository) pairs where the dependent is independent of the package owner (evaluated
    here) and the dependent's own successful sandboxed execution depended on the package (parent-side ablation)."""
    st = world.state
    out = []
    for _, a in st.scan("pkg_adoptions"):
        owner = st.tables["pkg_packages"][a["package"]]["owner"]
        if a["executions"] > 0 and independent(world, owner, a["owner"]):
            out.append(dict(a, independent=True))
    return out


# ====================================================================== bonuses (Gate L.n1)
def special_case_bonuses(world: World, creators: list[str], since_t: int = 1) -> dict:
    """Money credited to creators (or principals related to them) from system accounts after their admission:
    every credit from a system account counts, whatever its ledger kind ("endowment" included), except each
    creator's own admission endowment (the single `admin.join` endowment). Revenue may come only from other
    participants' accounts. Must be zero."""
    from ..commerce.integrity import related_set
    st = world.state
    circle: set[str] = set()
    for c in creators:
        circle |= related_set(world, c)
    accts = {a for a, r in st.scan("accounts") if r["owner"] in circle}
    sys_accts = {a for a, r in st.scan("accounts") if r["owner"] == SYSTEM and r["kind"] != "escrow"}
    admission = set()
    for c in circle:
        first = None
        for a in st.find_ids("accounts", "owner", c):
            for lid in st.indexes[("acct_txns", "acct")].get(a, ()):
                e = st.tables["acct_txns"][lid]
                tx = st.tables["txns"][e["txn"]]
                if tx["kind"] == "endowment" and tx["memo"] == f"endowment for {c}" and e["amount"] > 0:
                    first = e["txn"] if first is None else min(first, e["txn"])
        if first:
            admission.add(first)
    bad = []
    for tid, t in st.scan("txns"):
        if t["t"] < since_t or tid in admission:
            continue
        src = {a for a, d in t["postings"] if d < 0}
        dst = {a for a, d in t["postings"] if d > 0}
        if src & sys_accts and dst & accts:
            bad.append(tid)
    return {"bonuses": len(bad), "txns": bad[:20], "creator_accounts": len(accts)}


# ====================================================================== causal chains and counterfactuals (Gate L)
@dataclass
class Stage:
    name: str
    subsystem: str
    t: int | None
    evidence: dict = field(default_factory=dict)

    @property
    def present(self) -> bool:
        return self.t is not None


def chain_record(name: str, stages: list[Stage], provenance: str) -> dict:
    ok = [s for s in stages if s.present]
    ordered = all(a.t <= b.t for a, b in zip(ok, ok[1:]))
    return {"chain": name, "stages": [{"name": s.name, "subsystem": s.subsystem, "t": s.t, "evidence": s.evidence}
                                      for s in stages],
            "n_stages": len(ok), "complete": len(ok) == len(stages) and ordered, "ordered": ordered,
            "subsystems": sorted({s.subsystem for s in ok}), "provenance": provenance}


def counterfactual(world: World, agents: list, removal: Callable[[World, list], Any], horizon_s: int,
                   probe: Callable[[World], dict], drive: Callable[[World, list, int], None]) -> dict:
    """Fork `world` twice from one snapshot: the factual arm continues unchanged, the counterfactual arm applies
    `removal(world, agents)` (recorded system calls and/or withdrawing an agent's action) first. Both run the same
    horizon with copies of the same agents; `probe` measures each. Exact under the deterministic cost model."""
    blob = world.snapshot_blob()
    t0 = world.now
    fa = World.from_blob(blob, world_id=f"{world.world_id}~factual")
    cf = World.from_blob(blob, world_id=f"{world.world_id}~removed")
    ag_f, ag_c = copy.deepcopy(agents), copy.deepcopy(agents)
    try:
        removed = removal(cf, ag_c)
        drive(fa, ag_f, t0 + horizon_s)
        drive(cf, ag_c, t0 + horizon_s)
        a, b = probe(fa), probe(cf)
        return {"fork_t": t0, "horizon_s": horizon_s, "removed": removed, "factual": a, "counterfactual": b,
                "exact": world.spec.params.get("cost_model") == "deterministic",
                "invariants": {"factual": fa.check_invariants(), "counterfactual": cf.check_invariants()}}
    finally:
        fa.close()
        cf.close()


def first_event(world: World, etype: str, pred: Callable[[Any], bool] = lambda e: True, after_t: int = 0):
    for ev in world.log:
        if ev.type == etype and ev.t >= after_t and pred(ev):
            return ev
    return None
