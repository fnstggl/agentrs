"""One reusable relation graph of activities, resources, capabilities, constraints and needs (D-18, Part 6.4).

EVALUATOR SIDE. Nothing here is a kernel module, a tool or an event: in-world actors never see derived needs (the
4.0/5.0 `needs` module's in-world events are off by default - audit G S.2). Participants and entrepreneurs react to
the *consequences* the world produces (their own time spent by hand, failures and overloads of the services they
use, rejected work, public requests for help on the forum, prices); this module is the instrument that labels which
needs those consequences constitute, for Gate L and SP-6.4.

Graph. Relation *providers* read world state and return relation records
`{"src", "src_kind", "rel", "dst", "dst_kind", "attrs"}`; node ids are opaque strings and node kinds are declared by
the provider (no prefix conventions). Relation types:

    requires       an activity (or service, repository) needs a capability, resource, skill or access
    consumes       a service or activity uses up a resource
    produces       a provider (service, release, course) offers a capability, resource or skill
    constrains     a constraint (capacity, advisory, time budget, distance) limits a node
    substitutes    two providers offer the same thing
    complements    one capability's outputs feed another's inputs
    lowers_cost    a capability makes a requirement cheaper (time, money)
    raises_outcome a capability or skill raises the outcome of an activity
    loads          using a provider loads a resource (compute, transport capacity)
    teaches        an educational product raises a skill

*Observables* are generic readers registered per node kind: `fn(world, node_ids) -> {node: {name: value}}`.
Standard observable names the rules understand: `time_share` (share of an activity's time spent on this
requirement by hand), `lost_share` (share of an activity's tasks lost for lack of time), `failure_share` (share of
delegated executions that failed), `reject_share` (share of delegated outputs rejected downstream), `utilisation`
(peak share of a resource's capacity used), `refused_share` (share of requests refused for capacity),
`holders` (principals whose own activity is concerned).

Rules (keyed only by relation type and observable names; none names a host, package, capability or participant):

* bottleneck  - `requires` edge a -> x that is the bottleneck of its holders: observed per individual, x is at least
                `bottleneck` of the hand work an individual still does, and such individuals hold at least
                `unserved` of x's work (`unserved_share`, holders `bottleneck_holders`); or, for relations given
                as data, `time_share` >= `bottleneck`: need `provide x` for those holders;
* unreliable  - `requires` edge a -> x relied on through producer p whose `failure_share` or `reject_share` >=
                `unreliable`: need `alternative x`;
* scarcity    - `consumes` (or `loads`) edge s -> r with r `constrains`-ed and `utilisation` >= `scarce` or
                `refused_share` >=
                `refused`: need `supply x` for everything s `produces`, held by s's users and operator;
* overflow    - an activity with `lost_share` >= `overflow`: need `capacity a` (more throughput for its dominant
                requirement);
* advisory    - `constrains` edge c -> x where x is `required` by holders on an affected version: need
                `alternative x`.

A need's record carries its rule and the path of edges that derived it; `special_case_triggers` re-checks that
every recorded path consists of edges of the current graph and that every rule is generic. Holders are only
principals whose own activity requires the target (audit G S.5: planted interfaces do not count).

Modules register providers and observables with `register_provider` / `register_observable`; the providers for the
world's standard modules (work activities, cloud, capability registry, packages, configured data edges) are
registered below. A world can add relations as data (`data_relations`), e.g. skills, transport or coordination,
without changing any rule.
"""
from __future__ import annotations

from typing import Callable

DAY = 86400
RELATIONS = ("requires", "consumes", "produces", "constrains", "substitutes", "complements", "lowers_cost",
             "raises_outcome", "loads", "teaches")
THRESHOLDS = {"bottleneck": 0.35, "unserved": 0.35, "unreliable": 0.25, "scarce": 0.95, "refused": 0.05, "overflow": 0.1,
              "min_holders": 2}
RULES = ("bottleneck", "unreliable", "scarcity", "overflow", "advisory")
_PROVIDERS: dict[str, Callable] = {}
_OBSERVABLES: dict[str, Callable] = {}


def register_provider(name: str, fn: Callable) -> None:
    """`fn(world) -> [relation records]`; idempotent by name."""
    _PROVIDERS[name] = fn


def register_observable(kind: str, fn: Callable) -> None:
    """`fn(world, node_ids) -> {node: {observable: value}}` for nodes of `kind`; several per kind are merged."""
    key = f"{kind}:{getattr(fn, '__name__', id(fn))}"
    _OBSERVABLES[key] = fn


def rel(src, src_kind, r, dst, dst_kind, **attrs) -> dict:
    if r not in RELATIONS:
        raise ValueError(f"unknown relation type {r!r}")
    return {"src": src, "src_kind": src_kind, "rel": r, "dst": dst, "dst_kind": dst_kind, "attrs": attrs}


def graph(world) -> list[dict]:
    out = []
    for name in sorted(_PROVIDERS):
        for e in _PROVIDERS[name](world) or ():
            out.append(dict(e, provider=name))
    return out


def observe(world, edges: list[dict]) -> dict[str, dict]:
    kinds: dict[str, set] = {}
    for e in edges:
        kinds.setdefault(e["src_kind"], set()).add(e["src"])
        kinds.setdefault(e["dst_kind"], set()).add(e["dst"])
    obs: dict[str, dict] = {}
    for key in sorted(_OBSERVABLES):
        kind = key.split(":", 1)[0]
        nodes = sorted(kinds.get(kind, ()))
        if not nodes:
            continue
        for n, vals in (_OBSERVABLES[key](world, nodes) or {}).items():
            obs.setdefault(n, {}).update(vals)
    for e in edges:                     # edge-level observables (e.g. time share of one requirement)
        for k, v in (e["attrs"].get("obs") or {}).items():
            obs.setdefault(_ekey(e), {})[k] = v
    for n, vals in (world.spec.params.get("relation_nodes") or {}).items():   # node observables given as data
        obs.setdefault(n, {}).update(vals)
    return obs


def _ekey(e: dict) -> str:
    return f"{e['src']}|{e['rel']}|{e['dst']}"


def label(world, thresholds: dict | None = None) -> list[dict]:
    """The needs the current world state constitutes (see module doc). Deterministic and read-only."""
    th = {**THRESHOLDS, **(thresholds or {})}
    E = graph(world)
    obs = observe(world, E)
    by_src: dict[str, list] = {}
    by_dst: dict[str, list] = {}
    for e in E:
        by_src.setdefault(e["src"], []).append(e)
        by_dst.setdefault(e["dst"], []).append(e)

    def producers(x):
        return [e for e in by_dst.get(x, []) if e["rel"] == "produces"]

    def path(*es):
        return [{"src": e["src"], "rel": e["rel"], "dst": e["dst"], "provider": e["provider"]} for e in es]
    needs: dict[str, dict] = {}

    def add(kind, target, rule, holders, p, evidence):
        holders = sorted(set(holders))
        if len(holders) < int(th["min_holders"]):
            return
        nid = f"{kind}:{target}"
        cur = needs.get(nid)
        if cur is None:
            needs[nid] = {"id": nid, "kind": kind, "target": target, "rule": rule, "holders": holders,
                          "path": p, "evidence": evidence}
        else:
            cur["holders"] = sorted(set(cur["holders"]) | set(holders))
    for e in E:
        o = obs.get(_ekey(e), {})
        if e["rel"] == "requires":
            hs = o.get("holders") or obs.get(e["src"], {}).get("holders") or []
            relied = o.get("relied_holders") or []
            unserved = sorted(set(hs) - set(relied))
            if "bottleneck_holders" in o:
                # observed per individual: enough of the step's work is in individuals' bottleneck state
                if float(o.get("unserved_share", 0.0)) >= th["unserved"]:
                    add("provide", e["dst"], "bottleneck", o["bottleneck_holders"], path(e),
                        {"time_share": o.get("time_share"), "unserved_share": o.get("unserved_share")})
            elif float(o.get("time_share", 0.0)) >= th["bottleneck"]:
                add("provide", e["dst"], "bottleneck", unserved, path(e), {"time_share": o.get("time_share")})
            fs = max(float(o.get("failure_share", 0.0)), float(o.get("reject_share", 0.0)))
            if fs >= th["unreliable"] and relied:
                add("alternative", e["dst"], "unreliable", relied, path(e), {"failure_share": o.get("failure_share"),
                                                                             "reject_share": o.get("reject_share")})
        elif e["rel"] in ("consumes", "loads"):
            ro = obs.get(e["dst"], {})
            cons = [c for c in by_dst.get(e["dst"], []) if c["rel"] == "constrains"]
            if cons and (float(ro.get("utilisation", 0.0)) >= th["scarce"] or
                         float(ro.get("refused_share", 0.0)) >= th["refused"]):
                users = obs.get(e["src"], {}).get("holders") or []
                for pe in by_src.get(e["src"], []):
                    if pe["rel"] == "produces":
                        add("supply", pe["dst"], "scarcity", users, path(e, cons[0], pe),
                            {"utilisation": ro.get("utilisation"), "refused_share": ro.get("refused_share")})
        elif e["rel"] == "constrains" and e["attrs"].get("affects"):
            hit, hs = [], []
            for req in by_dst.get(e["dst"], []):
                if req["rel"] == "requires" and req["src"] in set(e["attrs"]["affects"]):
                    hit.append(req)
                    hs += obs.get(_ekey(req), {}).get("holders") or [req["src"]]
            if hit:
                add("alternative", e["dst"], "advisory", hs, path(e, *hit), {"constraint": e["src"]})
    for n, o in sorted(obs.items()):
        if "|" not in n and float(o.get("lost_share", 0.0)) >= th["overflow"]:
            reqs = [e for e in by_src.get(n, []) if e["rel"] == "requires"]
            if reqs:
                top = max(reqs, key=lambda e: (float(obs.get(_ekey(e), {}).get("time_share", 0.0)), e["dst"]))
                add("capacity", top["dst"], "overflow", o.get("holders") or [], path(top),
                    {"lost_share": o.get("lost_share")})
    for n in needs.values():
        n["producers"] = sorted({p["src"] for p in producers(n["target"])})
    return [needs[k] for k in sorted(needs)]


def special_case_triggers(world, needs: list[dict]) -> int:
    """Needs whose rule is not one of the generic rules or whose recorded path is not made of graph edges."""
    keys = {_ekey(e) for e in graph(world)}
    bad = 0
    for n in needs:
        if n["rule"] not in RULES or any(f"{p['src']}|{p['rel']}|{p['dst']}" not in keys for p in n["path"]):
            bad += 1
    return bad


# ====================================================================== providers of the standard modules
def _work_relations(world) -> list[dict]:
    """Activities of the population's work: activity -> requires -> capability per step, with edge observables
    from the individuals' own outcomes (time share by hand, failures, rejections, who relies on whom)."""
    st = world.state
    w = world.modules.get("work")
    if w is None:
        return []
    import numpy as np
    from ..substrate import columns
    out = []
    fam = columns.col(world, "work", "fam")
    act = columns.col(world, "work", "active")
    diy = columns.col(world, "work", "diy")
    exp_by_i: dict[int, list] = {}
    for _, r in st.scan("work_exp"):
        exp_by_i.setdefault(int(r["i"]), []).append(r)
    for j, g in enumerate(w.fams):
        rows = np.flatnonzero((fam == j) & act)
        if rows.size == 0:
            continue
        a = f"activity/{g['need']}+{g['then']}" if g["then"] else f"activity/{g['need']}"
        steps = [(0, g["need"])] + ([(1, g["then"])] if g["then"] else [])
        holders = [f"p{int(i)}" for i in rows]
        relied_by: dict[str, set] = {need: set() for _, need in steps}
        stats = {need: [0, 0, 0] for _, need in steps}            # uses, fails, rejections
        for i in rows.tolist():
            for r in exp_by_i.get(i, []):
                if r["step_need"] in stats:
                    st_ = stats[r["step_need"]]
                    st_[0] += r["uses"]
                    st_[1] += r["fail"]
                    st_[2] += r["rej"]
                    if r["active"]:
                        relied_by[r["step_need"]].add(i)
        th = THRESHOLDS["bottleneck"]
        hand = {need: 0.0 for _, need in steps}
        stuck = {need: 0.0 for _, need in steps}
        stuck_holders = {need: [] for _, need in steps}
        for i in rows.tolist():
            mine = {need: (0.0 if i in relied_by[need] else float(diy[i, s])) for s, need in steps}
            left = sum(mine.values())
            for need, m in mine.items():
                hand[need] += m
                # the step is this individual's bottleneck: most of the hand work it still does
                if left > 0 and m > 0 and m / left >= th:
                    stuck[need] += m
                    stuck_holders[need].append(f"p{i}")
        total_hand = sum(hand.values())
        for s, need in steps:
            base = float(diy[rows, s].sum())
            uses, fails, rej = stats[need]
            out.append(rel(a, "activity", "requires", f"capability/{need}", "capability",
                           obs={"time_share": round(hand[need] / total_hand, 4) if total_hand else 0.0,
                                "unserved_share": round(stuck[need] / base, 4) if base else 0.0,
                                "bottleneck_holders": stuck_holders[need],
                                "holders": holders, "relied_holders": [f"p{i}" for i in sorted(relied_by[need])],
                                "failure_share": fails / uses if uses else 0.0,
                                "reject_share": rej / uses if uses else 0.0}))
    return out


def _work_activity_obs(world, nodes) -> dict:
    """lost_share and holders of work activities (tasks lost for lack of working time)."""
    import numpy as np
    from ..substrate import columns
    w = world.modules.get("work")
    if w is None:
        return {}
    fam = columns.col(world, "work", "fam")
    done, lost = columns.col(world, "work", "done"), columns.col(world, "work", "lost")
    out = {}
    for j, g in enumerate(w.fams):
        a = f"activity/{g['need']}+{g['then']}" if g["then"] else f"activity/{g['need']}"
        if a not in nodes:
            continue
        rows = np.flatnonzero(fam == j)
        d, l_ = float(done[rows].sum()), float(lost[rows].sum())
        out[a] = {"lost_share": l_ / (d + l_) if d + l_ else 0.0, "holders": [f"p{int(i)}" for i in rows]}
    return out


def _cloud_relations(world) -> list[dict]:
    st = world.state
    if "deployments" not in st.tables:
        return []
    out = []
    for did, dep in sorted(st.scan("deployments")):
        if dep["status"] != "live":
            continue
        s, r = f"service/{dep['host']}", f"compute/{dep['instance']}"
        out.append(rel(s, "service", "consumes", r, "resource"))
        out.append(rel(f"capacity/{dep['instance']}", "constraint", "constrains", r, "resource"))
        if "products" in st.tables:
            for pid, p in st.scan("products"):
                if (p["delivery"] or {}).get("host") == dep["host"] and p.get("need"):
                    out.append(rel(s, "service", "produces", f"capability/{p['need']}", "capability"))
    return out


def _cloud_obs(world, nodes) -> dict:
    """utilisation (peak hourly CU / capacity over the last day) and refused share of compute resources; users
    (holders) of services."""
    st = world.state
    out = {}
    now_h = world.now // 3600
    for n in nodes:
        if n.startswith("compute/"):
            iid = n.split("/", 1)[1]
            inst = st.get("instances", iid)
            if inst is None:
                continue
            cap = inst["cores"] * 100_000
            hosts = [d["host"] for d in st.find("deployments", "instance", iid) if d["status"] == "live"]
            peak, over, tot = 0.0, 0, 0
            for h in hosts:
                for hr in range(now_h - 23, now_h + 1):
                    r = st.get("svc_hours", f"{h}|{hr}")
                    if r:
                        peak = max(peak, float(r["cu"]))
                        over += int(r["overload"])
                        tot += int(r["ok"]) + int(r["errors"]) + int(r["overload"])
            out[n] = {"utilisation": peak / cap if cap else 0.0, "refused_share": over / tot if tot else 0.0}
        elif n.startswith("service/"):
            host = n.split("/", 1)[1]
            users = set()
            if "work_exp" in st.tables and "products" in st.tables:
                provs = {pid for pid, p in st.scan("products") if (p["delivery"] or {}).get("host") == host}
                provs |= {d for d in st.find_ids("cap_decls", "host", host)} if "cap_decls" in st.tables else set()
                users = {f"p{r['i']}" for _, r in st.scan("work_exp") if r["prov"] in provs and r["active"]}
            out[n] = {"holders": sorted(users)}
    return out


def _caps_relations(world) -> list[dict]:
    st = world.state
    if "cap_decls" not in st.tables:
        return []
    return [rel(f"service/{d['host']}", "service", "produces", f"capability/{d['name']}", "capability")
            for _, d in sorted(st.scan("cap_decls")) if d["status"] == "active"]


def _pkg_relations(world) -> list[dict]:
    pkg = world.modules.get("pkg")
    if pkg is None:
        return []
    st = world.state
    out = []
    for name, p in st.scan("pkg_packages"):
        for r in pkg.releases(st, name):
            out.append(rel(f"release/{name}@{r['version']}", "release", "produces", f"package/{name}", "package"))
    for _, d in st.scan("pkg_deps"):
        if d["status"] == "active":
            out.append(rel(f"repo/{d['repo']}", "repository", "requires", f"package/{d['package']}", "package",
                           obs={"holders": [d["owner"]], "relied_holders": [d["owner"]]}))
    for aid, a in st.scan("pkg_advisories"):
        if a["verified"]:
            affected = [f"repo/{d['repo']}" for _, d in st.scan("pkg_deps") if d["package"] == a["package"]
                        and d["status"] == "active" and d["version"] in a["affected"]]
            out.append(rel(f"advisory/{aid}", "constraint", "constrains", f"package/{a['package']}", "package",
                           affects=affected))
    return out


def data_relations(world) -> list[dict]:
    """Relations a world declares as data (world spec param `relations`: a list of relation records with optional
    edge observables `obs`) - skills, transport links, coordination load, content communities, ..."""
    out = []
    for r in (world.spec.params.get("relations") or []):
        out.append(rel(r["src"], r["src_kind"], r["rel"], r["dst"], r["dst_kind"], **dict(r.get("attrs") or {}),
                       obs=dict(r.get("obs") or {})))
    return out


def _compute_market_relations(world) -> list[dict]:
    """Compute providers' core capacity as a shared resource (a technology raising computing demand): every live
    service loads the market of the provider its instance runs on, the provider's capacity constrains it."""
    st = world.state
    if "providers" not in st.tables or "deployments" not in st.tables:
        return []
    provs = sorted(pid for pid, p in st.scan("providers") if p["status"] == "active")
    out = [rel(f"capacity/{pid}", "constraint", "constrains", f"compute-market/{pid}", "compute_market")
           for pid in provs]
    on = set(provs)
    for _, dep in sorted(st.scan("deployments")):
        if dep["status"] != "live":
            continue
        inst = st.get("instances", dep["instance"])
        if inst and inst["provider"] in on:
            out.append(rel(f"service/{dep['host']}", "service", "loads", f"compute-market/{inst['provider']}",
                           "compute_market"))
    return out


def _compute_market_obs(world, nodes) -> dict:
    """utilisation (running cores / the provider's cores) and refused share (rentals refused for capacity over the
    last 7 days, against rentals made) of compute providers."""
    st = world.state
    since = world.now - 7 * DAY
    refused: dict[str, int] = {}
    for e in world.log:
        if e.t >= since and e.type == "ops.loss" and e.payload.get("kind") == "capacity_unavailable":
            refused[e.payload.get("resource")] = refused.get(e.payload.get("resource"), 0) + int(e.payload["n"])
    out = {}
    for n in nodes:
        pid = n.split("/", 1)[1]
        prov = st.get("providers", pid)
        if prov is None:
            continue
        insts = st.find("instances", "provider", pid)
        used = sum(i["cores"] for i in insts if i["status"] == "running")
        made = sum(1 for i in insts if i["created"] >= since)
        r = refused.get(pid, 0)
        out[n] = {"utilisation": used / prov["cores"] if prov["cores"] else 0.0,
                  "refused_share": r / (r + made) if r + made else 0.0}
    return out


def _freight_relations(world) -> list[dict]:
    """The freight network (transport and market access): the carrier's service on each link loads the link,
    whose daily capacity constrains it, and gives access between the link's two regions."""
    st = world.state
    if "links" not in st.tables or world.modules.get("freight") is None:
        return []
    out = []
    for lid, l in sorted(st.scan("links")):
        s = f"carrier/{lid}"
        out.append(rel(s, "carrier", "loads", f"transport/{lid}", "transport_link"))
        out.append(rel(f"capacity/{lid}", "constraint", "constrains", f"transport/{lid}", "transport_link"))
        out.append(rel(s, "carrier", "produces", f"access/{l['a']}-{l['b']}", "capability"))
    return out


def _freight_obs(world, nodes) -> dict:
    """Links: utilisation (the most loaded of the last 3 days: units reserved / daily capacity) and refused share
    (shipments over the link in the last 7 days that waited for capacity). Carriers: holders (payers of those
    shipments and owners of the receiving facilities)."""
    st = world.state
    today = world.now // DAY
    since = world.now - 7 * DAY
    ships: dict[str, list] = {}
    for _, sh in st.scan("shipments"):
        if sh["t"] >= since:
            for lid in sh.get("route") or ():
                ships.setdefault(lid, []).append(sh)
    out = {}
    for n in nodes:
        lid = n.split("/", 1)[1]
        link = st.get("links", lid)
        if link is None:
            continue
        mine = ships.get(lid, [])
        if n.startswith("transport/"):
            peak = max((float((st.get("link_days", f"{lid}|{d}") or {}).get("used", 0)) for d in range(today - 2, today + 1)),
                       default=0.0)
            out[n] = {"utilisation": peak / max(1, link["capacity"]),
                      "refused_share": sum(1 for x in mine if x["delayed"]) / len(mine) if mine else 0.0}
        elif n.startswith("carrier/"):
            hs = set()
            for x in mine:
                hs.add(x["payer"])
                fac = st.get("facilities", x["to"]) if "facilities" in st.tables else None
                if fac:
                    hs.add(fac["owner"])
            out[n] = {"holders": sorted(h for h in hs if h)}
    return out


def _org_relations(world) -> list[dict]:
    """Organisations' coordination load (the civilisation's `org_coordination` coupling, crucible.civ.couplings):
    an organisation running production with operators requires coordination; the share of its operator hours lost
    to coordination overhead (which grows with headcount, halved by running workflows) is the requirement's time
    share; its members are the holders, and they rely on a coordination tool when it runs workflows."""
    st = world.state
    if world.modules.get("civlinks") is None or "facilities" not in st.tables:
        return []
    from ..civ import couplings as CP
    if not CP.active(world, "org_coordination"):
        return []
    out = []
    for org in sorted({f["owner"] for _, f in st.scan("facilities") if f["owner"]}):
        n = CP.headcount(world, st, org)
        if n < 2:
            continue
        share = 1.0 - CP.coordination_factor(world, st, org)
        members = sorted({m["person"] for m in st.find("members", "org", org)} | {org})
        flows = st.find_ids("wf_flows", "owner", org) if "wf_flows" in st.tables else []
        out.append(rel(f"activity/operations:{org}", "activity", "requires", "capability/coordination", "capability",
                       obs={"time_share": round(share, 4), "holders": members,
                            "relied_holders": members if flows else [], "operators": n}))
    return out


register_provider("work", _work_relations)
register_provider("cloud", _cloud_relations)
register_provider("caps", _caps_relations)
register_provider("pkg", _pkg_relations)
register_provider("data", data_relations)
register_observable("activity", _work_activity_obs)
register_observable("resource", _cloud_obs)
register_observable("service", _cloud_obs)
register_provider("compute_market", _compute_market_relations)
register_provider("freight", _freight_relations)
register_provider("orgs", _org_relations)
register_observable("compute_market", _compute_market_obs)
register_observable("transport_link", _freight_obs)
register_observable("carrier", _freight_obs)
