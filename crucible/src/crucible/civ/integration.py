"""Measurements of integration on a running civilisation (Gate O: O.1-O.4, O.7, O.n1).

Everything here is measured on data flow in the world, not on which modules happen to be loaded (audit A_J,
"Why the criteria are weak"):

* `subsystems(world, pv)` - for each Mode C subsystem: present (its modules are in the world, or for energy its grids
  exist), active (state-changing events attributed to it in the window) and connected (a provenance edge between
  one of its events and an event of another subsystem). `strict` counts the subsystems that are all three.
* `cross_system(pv)` - provenance edges between subsystems: the event types that are consequences of another
  subsystem's events (O.4) and the subsystem pairs they connect.
* `modules_without_shared_state(world, pv)` - modules (other than kernel/storage infrastructure) none of whose
  events depends on another module's events or process runs and none of whose events another module's events
  depend on (provenance in the window): inert modules count, because a module nobody uses is a disconnected
  mini-environment (O.n1).
* `duplicate_facts(world)` - a data-level scan (not declarations): for each kind of shared fact, the tables of the
  world that hold facts of that kind; a kind held in more than one populated table, and every single fact present
  in two tables, is a duplicate (O.2).
* `identity_violations(world)` - dangling references to principals or accounts, population principals without a
  population row, organisation members that are not persons, accounts without an owner (O.2).
* `ledger_violations(world)` - bank invariants and the ledger recomputed from postings (O.3).
* `fingerprint(world)` / `structural_distance(a, b)` - the discrete structure of a BUILT world: module set, coupling
  set, supply-chain graph shape, freight topology, regime categories, institutions, social placement, science
  bindings (O.7, SP-10.1); two worlds whose fingerprints are equal differ only in parameters.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from . import chains as CH
from . import subsystems as SUB

# kernel/storage infrastructure and genesis-only installers (their output - firms, sites, providers - is the shared
# state other modules use; they have no events of their own after the first day)
INFRA_MODULES = frozenset({"admin", "artifacts", "cols", "assets", "civgen", "civlinks", "industrygen", "popsites"})


# --------------------------------------------------------------------------------------------- subsystems
def subsystems(world, pv: CH.Provenance, since_seq: int = 0) -> dict:
    """Activity and connections are counted only for events after `since_seq` (genesis and warm-up excluded)."""
    present = {}
    for s, d in SUB.SUBSYSTEMS.items():
        mods = [m for m in d["modules"] if m in world.modules]
        if s == "energy":
            st = world.state
            grids = [r for _, r in st.scan("regions") if r.get("grid")] if "regions" in st.tables else []
            present[s] = bool(grids)
        else:
            present[s] = bool(mods)
    active = Counter()
    for s_, sub in pv.subsystem.items():
        if sub in SUB.SUBSYSTEMS and s_ > since_seq:
            active[sub] += 1
    connected = Counter()
    for b, ps in pv.parents.items():
        if b <= since_seq:
            continue
        sb = pv.subsystem.get(b)
        for a in ps:
            sa = pv.subsystem.get(a)
            if sa and sb and sa != sb and sa not in CH.NOT_STAGES and sb not in CH.NOT_STAGES:
                connected[sa] += 1
                connected[sb] += 1
    rows = {s: {"present": present[s], "events": int(active.get(s, 0)), "cross_edges": int(connected.get(s, 0)),
                "modules": [m for m in SUB.SUBSYSTEMS[s]["modules"] if m in world.modules]}
            for s in SUB.SUBSYSTEMS}
    strict = [s for s, r in rows.items() if r["present"] and r["events"] > 0 and r["cross_edges"] > 0]
    return {"subsystems": rows, "present": sum(r["present"] for r in rows.values()),
            "active": sum(1 for r in rows.values() if r["present"] and r["events"] > 0),
            "strict": len(strict), "strict_names": strict,
            "missing": sorted(set(SUB.SUBSYSTEMS) - set(strict))}


def cross_system(pv: CH.Provenance, since_seq: int = 0) -> dict:
    pairs = Counter()
    types = Counter()
    for b, ps in pv.parents.items():
        if b <= since_seq:
            continue
        sb = pv.subsystem.get(b)
        if not sb or sb in CH.NOT_STAGES:
            continue
        for a in ps:
            sa = pv.subsystem.get(a)
            if sa and sa != sb and sa not in CH.NOT_STAGES:
                pairs[(sa, sb)] += 1
                types[pv.events[b].type] += 1
    return {"cross_system_event_types": len(types), "types": dict(types.most_common()),
            "subsystem_pairs": {f"{a}->{b}": n for (a, b), n in pairs.most_common()}}


def modules_without_shared_state(world, pv: CH.Provenance, since_seq: int = 0) -> dict:
    hm = SUB.host_module_map(world)

    def mod_of(s):
        ev = pv.events[s]
        if ev.type == "cols.batch":
            return hm.get(str(ev.payload.get("kind", "")).split(".", 1)[0])
        return hm.get(ev.type.split(".", 1)[0])
    linked: set[str] = set()
    seen: Counter = Counter()
    for s in pv.events:
        if pv.subsystem.get(s) is None or s <= since_seq:
            continue
        m = mod_of(s)
        if m:
            seen[m] += 1
    for b, ps in pv.parents.items():
        if b <= since_seq or pv.subsystem.get(b) is None:
            continue
        mb = mod_of(b)
        for a in ps:
            # the module whose event (or process run) this one depends on; a module is connected when its events
            # depend on another module's, or another module's depend on it
            ma = pv.proc_module.get(a) if pv.events[a].type == "proc" else \
                (mod_of(a) if pv.subsystem.get(a) else None)
            if ma and mb and ma != mb:
                linked.add(ma)
                linked.add(mb)
    mods = sorted(m for m in world.modules if m not in INFRA_MODULES)
    without = [m for m in mods if m not in linked]
    return {"modules": len(mods), "modules_without_shared_state": len(without), "without": without,
            "inert": [m for m in without if not seen.get(m)], "events_by_module": dict(seen.most_common())}


# --------------------------------------------------------------------------------------------- shared facts
def _rows(st, table):
    return st.tables.get(table, {}) if table in st.tables else {}


def duplicate_facts(world) -> dict:
    """Data-level duplicate scan. For each fact kind: the populated tables holding it, and facts found twice."""
    st = world.state
    from ..substrate import columns
    kinds: dict[str, dict[str, int]] = defaultdict(dict)
    dups: dict[str, list] = defaultdict(list)
    # relationship (one principal follows another)
    ff = {(r["follower"], r["target"]) for r in _rows(st, "forum_follows").values()
          if not str(r["target"]).startswith("board:")}
    if ff:
        kinds["follow"]["forum.forum_follows"] = len(ff)
    if "social" in world.modules:
        try:
            from ..social import graph as G
            edges = G.active_edges(world) if hasattr(G, "active_edges") else None
        except Exception:  # noqa: BLE001
            edges = None
        if edges:
            kinds["follow"]["social.follow"] = len(edges)
            dups["follow"] += sorted(ff & set(edges))[:20]
    # community
    for t, label in (("forum_boards", "forum.forum_boards"), ("communities", "community.communities")):
        n = len(_rows(st, t))
        if n:
            kinds["community"][label] = n
    # product offers to the population
    for t, label in (("listings", "market.listings"), ("products", "products.products"),
                     ("xchg_listings", "xchg.xchg_listings"), ("retail", "industry.retail")):
        n = len(_rows(st, t))
        if n:
            kinds["product_offer"][label] = n
    # employment of persons
    if "pop" in columns.tables(world):
        n = int((columns.col(world, "pop", "employer") >= 0).sum())
        if n:
            kinds["employment"]["pop.employer"] = n
    for t, label in (("labor_workers", "labor.labor_workers"), ("npc_workers", "jobs.npc_workers"),
                     ("employments", "labor.employments")):
        n = len(_rows(st, t))
        if n:
            kinds["employment"][label] = n
    own = sum(int(v) for v in (st.meta.get("own_operators") or {}).values())
    if own:
        kinds["employment"]["industry.own_operators"] = own
    # institutions
    for t, label in (("constitutions", "gov.constitutions"), ("inst_charters", "inst.inst_charters")):
        n = len(_rows(st, t))
        if n:
            kinds["institution"][label] = n
    orgs_c = set(_rows(st, "constitutions")) & set(_rows(st, "inst_charters"))
    dups["institution"] += sorted(orgs_c)[:20]
    # contracts and commitments
    for t, label in (("contracts", "contracts.contracts"), ("commitments", "commit.commitments"),
                     ("subscriptions", "subs.subscriptions")):
        n = len(_rows(st, t))
        if n:
            kinds["contract"][label] = n
    # energy price of the same region: grids (industry) and the datacenter fleet's own tariff (fleet)
    if "fleet" in world.modules and any(r.get("grid") for r in _rows(st, "regions").values()):
        kinds["energy_price"]["industry.grid"] = sum(1 for r in _rows(st, "regions").values() if r.get("grid"))
        kinds["energy_price"]["fleet.tariff"] = 1
    multi = {k: v for k, v in kinds.items() if len(v) > 1}
    n_dup = len(multi) + sum(len(v) for v in dups.values())
    return {"duplicate_shared_facts": n_dup, "kinds_in_several_tables": multi, "duplicate_facts": dict(dups),
            "kinds": {k: dict(v) for k, v in kinds.items()}}


def identity_violations(world) -> dict:
    st = world.state
    from ..substrate import columns, ids
    errs = []
    principals = st.tables["principals"]
    n_pop = columns.rows(world, "pop") if "pop" in columns.tables(world) else 0
    for p in principals:
        i = ids.index(p)
        if i is not None and i >= n_pop:
            errs.append(f"{p}: population principal without a population row")
    for aid, a in _rows(st, "accounts").items():
        if a["owner"] not in principals:
            errs.append(f"account {aid} owned by unknown {a['owner']}")
    for k, m in _rows(st, "members").items():
        if m["person"] not in principals or m["org"] not in principals:
            errs.append(f"member {k} refers to an unknown principal")
    for t, key in (("deployments", "owner"), ("instances", "owner"), ("hosts", "owner"), ("repos", "owner"),
                   ("sci_papers", "author"), ("forum_posts", "author"), ("minds", "owner")):
        for rid, r in _rows(st, t).items():
            v = r.get(key)
            if v and v not in principals:
                errs.append(f"{t}.{rid}.{key} = unknown principal {v}")
    persons_outside = sorted(p for p, r in principals.items() if r["kind"] == "person" and ids.index(p) is None)
    return {"identity_consistency_violations": len(errs), "detail": errs[:20],
            "persons_outside_population": len(persons_outside)}


def ledger_violations(world) -> dict:
    bank = world.modules["bank"]
    errs = bank.check_invariants(world) + bank.check_ledger_consistency(world)
    from ..substrate import columns
    outside = int(columns.col(world, "pop", "wallet_cents").astype(np.int64).sum()) \
        if "pop" in columns.tables(world) else 0
    return {"ledger_invariant_violations": len(errs), "detail": errs[:20], "money_supply": bank.money_supply(world),
            "population_undeposited_cents": outside}


# --------------------------------------------------------------------------------------------- structure
def _graph_sig(nodes: set, edges: set) -> dict:
    deg_in, deg_out = Counter(), Counter()
    for a, b in edges:
        deg_out[a] += 1
        deg_in[b] += 1
    depth = {}

    def d(n, seen=()):
        if n in depth:
            return depth[n]
        ps = [a for a, b in edges if b == n and a not in seen]
        depth[n] = 1 + max((d(a, seen + (n,)) for a in ps), default=0)
        return depth[n]
    for n in nodes:
        d(n)
    return {"nodes": len(nodes), "edges": len(edges), "depth": max(depth.values(), default=0),
            "in_degrees": sorted(deg_in[n] for n in nodes), "out_degrees": sorted(deg_out[n] for n in nodes)}


def _bucket(x, cuts, names):
    for c, n in zip(cuts, names):
        if x < c:
            return n
    return names[-1]


def fingerprint(world) -> dict:
    """Discrete structure of a built world (call after genesis or later)."""
    st = world.state
    from ..substrate import columns
    fp: dict = {"modules": sorted(world.modules)}
    fp["couplings"] = sorted(k for k, r in _rows(st, "civ_links").items() if r.get("enabled"))
    # supply chain: firm -> firm when a design's input is another firm's output
    produces = {}
    for did, d in _rows(st, "designs").items():
        produces[f"i:{did}"] = d["owner"]
        if d.get("material"):
            produces[d["material"]] = d["owner"]
    nodes, edges = set(), set()
    for did, d in _rows(st, "designs").items():
        nodes.add(d["owner"])
        for x in d.get("inputs") or []:
            src = produces.get(x["sku"])
            if src and src != d["owner"]:
                edges.add((src, d["owner"]))
    fp["supply_chain"] = _graph_sig(nodes, edges)
    regs = set(_rows(st, "regions"))
    links = {tuple(sorted((r["a"], r["b"]))) for r in _rows(st, "links").values()}
    fp["freight"] = {"regions": len(regs), "links": len(links),
                     "degrees": sorted(sum(1 for l in links if r in l) for r in regs)}
    caps = [r["grid"]["capacity"] for r in _rows(st, "regions").values() if r.get("grid")]
    fp["energy_regime"] = _bucket(float(np.median(caps)) if caps else 0, (2000, 20000),
                                  ("scarce", "normal", "abundant"))
    provs = list(_rows(st, "providers").values())
    fp["compute"] = {"providers": len(provs),
                     "regime": _bucket(float(np.median([p["cores"] for p in provs])) if provs else 0, (64, 1024),
                                       ("scarce", "normal", "abundant")),
                     "datacenter_regions": len({r["region"] for r in _rows(st, "civ_dc").values()})}
    fp["institutions"] = sorted(Counter((c.get("distribution") or {}).get("basis", "none")
                                        for c in _rows(st, "constitutions").values()).items())
    fp["public_orgs"] = sum(1 for o in _rows(st, "orgs").values() if o.get("kind") == "public")
    fp["boards"] = sorted(_rows(st, "forum_boards"))
    if "goods" in columns.tables(world) and "pop" in columns.tables(world):
        reg = columns.col(world, "goods", "region").astype(np.int64)
        con = columns.col(world, "pop", "contacts").astype(np.int64)
        ok = con >= 0
        same = (reg[np.where(ok, con, 0)] == reg[:, None]) & ok
        loc = float(same.sum() / max(1, ok.sum()))
        fp["social_locality"] = _bucket(loc, (0.45, 0.8), ("interconnected", "mixed", "fragmented"))
    fp["science_bound"] = len((world.modules["civlinks"].config.get("bindings") or [])) \
        if "civlinks" in world.modules else 0
    fp["phenomena"] = len(_rows(st, "sci_phen"))
    return fp


def structural_distance(a: dict, b: dict) -> float:
    """Share of fingerprint features that differ (0: the same structure)."""
    keys = sorted(set(a) | set(b))
    return sum(1 for k in keys if a.get(k) != b.get(k)) / max(1, len(keys))
