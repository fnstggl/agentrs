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

* bottleneck  - `requires` edge a -> x whose `time_share` >= `bottleneck` with no live producer of x relied on:
                need `provide x` for a's holders;
* unreliable  - `requires` edge a -> x relied on through producer p whose `failure_share` or `reject_share` >=
                `unreliable`: need `alternative x`;
* scarcity    - `consumes` edge s -> r with r `constrains`-ed and `utilisation` >= `scarce` or `refused_share` >=
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
THRESHOLDS = {"bottleneck": 0.35, "unreliable": 0.25, "scarce": 0.95, "refused": 0.05, "overflow": 0.1,
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
            if float(o.get("time_share", 0.0)) >= th["bottleneck"]:
                add("provide", e["dst"], "bottleneck", unserved, path(e), {"time_share": o.get("time_share")})
            fs = max(float(o.get("failure_share", 0.0)), float(o.get("reject_share", 0.0)))
            if fs >= th["unreliable"] and relied:
                add("alternative", e["dst"], "unreliable", relied, path(e), {"failure_share": o.get("failure_share"),
                                                                             "reject_share": o.get("reject_share")})
        elif e["rel"] == "consumes":
            ro = obs.get(e["dst"], {})
            cons = [c for c in by_dst.get(e["dst"], []) if c["rel"] == "constrains"]
            if cons and (float(ro.get("utilisation", 0.0)) >= th["scarce"] or
                         float(ro.get("refused_share", 0.0)) >= th["refused"]):
                users = obs.get(e["src"], {}).get("holders") or []
                for pe in by_src.get(e["src"], []):
                    if pe["rel"] == "produces":
                        add("supply", pe["dst"], "scarcity", users, path(e, cons[0], pe),
                            {"utilisation": ro.get("utilisation"), "refused_share": ro.get("refused_share")})
        elif e["rel"] == "constrains":
            for req in by_dst.get(e["dst"], []):
                if req["rel"] != "requires":
                    continue
                ro = obs.get(_ekey(req), {})
                if e["attrs"].get("affects") and req["src"] in set(e["attrs"]["affects"]):
                    add("alternative", e["dst"], "advisory", ro.get("holders") or [req["src"]], path(e, req),
                        {"constraint": e["src"]})
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
        total_hand = float(diy[rows].sum())
        for s, need in steps:
            relied, fails, rej, uses = [], 0, 0, 0
            hand = 0.0
            for i in rows.tolist():
                rs = [r for r in exp_by_i.get(i, []) if r["step_need"] == need]
                act_r = [r for r in rs if r["active"]]
                if act_r:
                    relied.append(f"p{i}")
                else:
                    hand += float(diy[i, s])
                for r in rs:
                    uses += r["uses"]
                    fails += r["fail"]
                    rej += r["rej"]
            share = hand / total_hand if total_hand > 0 else 0.0
            out.append(rel(a, "activity", "requires", f"capability/{need}", "capability",
                           obs={"time_share": round(share, 4), "holders": holders, "relied_holders": relied,
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


register_provider("work", _work_relations)
register_provider("cloud", _cloud_relations)
register_provider("caps", _caps_relations)
register_provider("pkg", _pkg_relations)
register_provider("data", data_relations)
register_observable("activity", _work_activity_obs)
register_observable("resource", _cloud_obs)
register_observable("service", _cloud_obs)
