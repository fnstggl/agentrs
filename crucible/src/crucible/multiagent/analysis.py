"""Evaluator-side analysis of a self-play match: cooperation, competition, shared-world consequences, the Part 12
capabilities and behavioural learning evidence. Everything is computed from the world's authoritative state and
event log after the fact; no participant can read any of it.

Parties. Every admitted identity belongs to its *controller* (`admin.join`'s controller label); an organisation
belongs to the controller(s) of the identities that own or administer it; repositories, packages, products,
accounts and hosts belong to the party of their owner. Interactions are counted between parties: two identities
of one controller interacting is not cooperation (nor trade, nor competition) - it is one party.

`interactions(world, controllers, seq0)` returns typed interaction records:

cooperative  negotiation (a deal between parties answered: accepted, rejected or countered), agreement (an active or
             completed contract between parties), membership (an identity in an organisation of another party),
             contribution (a commit to, or merged pull request into, another party's repository), dependency (a
             repository of one party depending on another party's package), grant (a capability delegated across
             parties), trade (a commerce payment between parties: purchases, subscriptions, licences, contract and
             service payments), service_use (a call of another party's deployed service, executed and metered),
             venture (a shared project with partners of several parties, or an organisation with members of several
             parties); records of something one party provides to another (trade, service_use, dependency) name
             the ``provider`` (the payee, the service's owner, the package's owner) and the ``user``;
competitive  same_market (two parties selling products of one kind or listing in one need), scarce_resource (a
             request for compute capacity refused because the provider's finite capacity was taken, while another
             party held capacity there: the refused party and each holder, with the provider's utilisation at the
             refusal), contested_customer (one third-party customer paying two parties), competing_tech (two
             parties' released packages exposing a function of the same family).
shared_capacity (reported, not competition) two parties renting from one provider with no request refused there.

`consequences(world, controllers, seq0)` attributes every event to the party whose command (directly, or through
the processes the command scheduled) caused it and counts the events visible to identities of *other* parties, and
the response chains: a later command of another party that names an id first brought into being by the first.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

COMMERCE = frozenset({"subscription", "product_sale", "license", "take_rate", "sale", "invoice", "contract", "bounty",
                      "commitment", "api_call", "usage"})
BOOK = frozenset({"cmd", "cmd.rejected", "ext", "idem", "ctr", "know", "note", "sched.add", "sched.del", "time",
                  "proc", "proc.failed"})
CAPABILITIES = ("found_competing_businesses", "collaborate_on_software", "trade", "negotiate", "join_organizations",
                "develop_competing_technologies", "learn_from_independent_experiences",
                "create_services_used_by_others", "form_cooperative_ventures", "compete_for_scarce_resources",
                "respond_to_changing_strategies")
ID_TOKEN = re.compile(r"^[a-z]{2,8}_[0-9]{4,}$|^[a-z0-9-]+/[a-z0-9_-]+$")


class Parties:
    """Party (controller) of principals, organisations and the things they own."""

    def __init__(self, world, controllers: dict[str, str]):
        self.world = world
        self.of_principal = dict(controllers)
        st = world.state
        org_ctrl: dict[str, set] = defaultdict(set)
        if "members" in st.tables:
            for _, m in st.scan("members"):
                c = controllers.get(m["person"])
                if c and m.get("role") in ("owner", "admin"):
                    org_ctrl[m["org"]].add(c)
        for o, cs in org_ctrl.items():
            self.of_principal.setdefault(o, sorted(cs)[0] if len(cs) == 1 else "joint:" + "+".join(sorted(cs)))
        self.learner_parties = set(controllers.values())

    def __call__(self, principal: str | None) -> str | None:
        if not principal:
            return None
        return self.of_principal.get(principal)

    def account(self, acct: str) -> str | None:
        st = self.world.state
        a = st.get("accounts", acct) if "accounts" in st.tables else None
        return self(a["owner"]) if a else None

    def owner_of_account(self, acct: str) -> str | None:
        st = self.world.state
        a = st.get("accounts", acct) if "accounts" in st.tables else None
        return a["owner"] if a else None

    def repo(self, repo: str) -> str | None:
        r = self.world.state.get("repos", repo) if "repos" in self.world.state.tables else None
        return self(r["owner"]) if r else None

    def is_learner(self, party: str | None) -> bool:
        return party is not None and (party in self.learner_parties or party.startswith("joint:"))


def _rows(st, table: str):
    return st.scan(table) if table in st.tables else ()


def _t_ok(rec: dict, t0: int) -> bool:
    t = rec.get("t", rec.get("created", rec.get("since", 0)))
    return t is None or int(t) >= t0


def interactions(world, controllers: dict[str, str], t0: int = 0) -> dict[str, list[dict]]:
    P = Parties(world, controllers)
    st = world.state
    coop: list[dict] = []
    comp: list[dict] = []

    def two(a, b) -> bool:
        return a is not None and b is not None and a != b and (P.is_learner(a) or P.is_learner(b))

    # negotiation and agreements
    for did, d in _rows(st, "deals"):
        a, b = P(d.get("from")), P(d.get("to"))
        if two(a, b) and d.get("status") in ("accepted", "rejected", "countered") and _t_ok(d, t0):
            coop.append({"kind": "negotiation", "id": did, "parties": [a, b], "status": d["status"],
                         "both_learners": P.is_learner(a) and P.is_learner(b)})
    for cid, c in _rows(st, "contracts"):
        ps = sorted({P(x) or f"world:{x}" for x in c.get("parties", [])})
        if len(ps) >= 2 and any(P.is_learner(x) for x in ps) and c.get("status") in ("active", "completed") \
                and _t_ok(c, t0):
            coop.append({"kind": "agreement", "id": cid, "parties": ps,
                         "clauses": sorted({x.get("type") for x in c.get("clauses", [])}),
                         "both_learners": sum(P.is_learner(x) for x in ps) >= 2})
    # membership across parties
    for mid, m in _rows(st, "members"):
        a = P(m["person"])
        if a is None or not P.is_learner(a):
            continue
        owners = [x["person"] for x in st.find("members", "org", m["org"]) if x["role"] == "owner"]
        b = P(owners[0]) if owners else None
        b = b or f"world:{m['org']}"
        if b != a and m.get("role") in ("member", "admin") and _t_ok(m, t0):
            coop.append({"kind": "membership", "id": mid, "parties": [a, b], "org": m["org"],
                         "both_learners": P.is_learner(b)})
    # software collaboration
    for cid, c in _rows(st, "commits"):
        a, b = P(c.get("author")), P.repo(c.get("repo", ""))
        if two(a, b) and P.is_learner(a) and _t_ok(c, t0):
            coop.append({"kind": "contribution", "id": cid, "parties": [a, b], "repo": c["repo"],
                         "both_learners": P.is_learner(b)})
    for pid, pr in _rows(st, "pkg_prs"):
        a, b = P(pr.get("author")), P.repo(pr.get("repo", ""))
        if two(a, b) and pr.get("status") == "merged":
            coop.append({"kind": "contribution", "id": pid, "parties": [a, b], "repo": pr.get("repo"),
                         "merged_pr": True, "both_learners": P.is_learner(a) and P.is_learner(b)})
    pkg_owner = {k: P(v.get("owner")) for k, v in _rows(st, "pkg_packages")}
    for did, d in _rows(st, "pkg_deps"):
        a, b = P.repo(d.get("repo", "")), pkg_owner.get(d.get("package"))
        if two(a, b):
            coop.append({"kind": "dependency", "id": did, "parties": [a, b], "package": d.get("package"),
                         "provider": b, "user": a, "provider_is_learner": P.is_learner(b),
                         "both_learners": P.is_learner(a) and P.is_learner(b)})
    for gid, g in _rows(st, "grants"):
        a, b = P(g.get("owner")), P(g.get("grantee"))
        if two(a, b) and _t_ok(g, t0):
            coop.append({"kind": "grant", "id": gid, "parties": [a, b], "cap": g.get("cap"),
                         "both_learners": P.is_learner(a) and P.is_learner(b)})
    # trade: commerce payments between parties
    for tid, tx in _rows(st, "txns"):
        if tx.get("kind") not in COMMERCE or int(tx.get("t", 0)) < t0:
            continue
        pay = [(b, d) for b, d in tx["postings"]]
        src = [P.account(b) or f"world:{P.owner_of_account(b)}" for b, d in pay if d < 0]
        dst = [P.account(b) or f"world:{P.owner_of_account(b)}" for b, d in pay if d > 0]
        for a in set(src):
            for b in set(dst):
                if a != b and (P.is_learner(a) or P.is_learner(b)) and "world:sys" not in (a, b):
                    coop.append({"kind": "trade", "id": tid, "parties": [a, b], "ledger_kind": tx["kind"],
                                 "cents": int(sum(d for _, d in pay if d > 0)),
                                 "provider": b, "user": a, "provider_is_learner": P.is_learner(b),
                                 "both_learners": P.is_learner(a) and P.is_learner(b)})
    # shared projects / mixed organisations
    for pid, pj in _rows(st, "inst_projects"):
        partners = [x["partner"] for x in st.find("inst_partners", "project", pid)] if "inst_partners" in st.tables \
            else []
        ps = sorted({P(x) or f"world:{x}" for x in [pj.get("owner")] + partners if x})
        if len(ps) >= 2 and any(P.is_learner(x) for x in ps):
            coop.append({"kind": "venture", "id": pid, "parties": ps, "how": "shared project",
                         "both_learners": sum(P.is_learner(x) for x in ps) >= 2})
    orgs: dict[str, set] = defaultdict(set)
    for _, m in _rows(st, "members"):
        c = P(m["person"])
        if c is not None:
            orgs[m["org"]].add(c)
    for o, cs in orgs.items():
        if len(cs) >= 2 and sum(P.is_learner(x) for x in cs) >= 1 and any(P.is_learner(x) for x in cs):
            if len([x for x in cs if P.is_learner(x)]) >= 2:
                coop.append({"kind": "venture", "id": o, "parties": sorted(cs), "how": "organisation of several "
                             "parties", "both_learners": True})
    # one pass over the event log: services executed for another party (metered calls in the caller's
    # transaction) and compute capacity (instances held at each provider, replayed, and refused requests)
    dep_owner = {k: v.get("owner") for k, v in _rows(st, "deployments")}
    seen = set()
    cap_now = {k: int(v.get("cores", 0) or 0) for k, v in _rows(st, "providers")}
    delivered: list[tuple[int, str, int]] = []   # hardware added later (capacity at a refusal = now - later adds)
    inst: dict[str, dict] = {}                 # instance -> {owner, provider, cores, running}
    refusals: list[dict] = []
    for e in world.events_range(0, world.seq):
        typ = e.type
        pl = e.payload
        if typ == "cloud.usage":
            if e.t < t0:
                continue
            owner = dep_owner.get(pl.get("dep"))
            a, b = P(e.actor), P(owner)
            if two(a, b) and (a, b, pl.get("dep")) not in seen:
                seen.add((a, b, pl.get("dep")))
                coop.append({"kind": "service_use", "id": pl.get("dep"), "parties": [a, b], "caller": e.actor,
                             "provider": b, "user": a, "provider_is_learner": P.is_learner(b),
                             "both_learners": P.is_learner(a) and P.is_learner(b)})
        elif typ == "cloud.hw_delivered":
            delivered.append((int(e.t), pl.get("provider"), int(pl.get("cores", 0) or 0)))
        elif typ == "cloud.instance":
            inst[pl["id"]] = {"owner": pl.get("owner"), "provider": pl.get("provider"),
                              "cores": int(pl.get("cores", 0) or 0), "running": True}
        elif typ == "cloud.instance_status" and pl.get("id") in inst:
            inst[pl["id"]]["running"] = pl.get("status") == "running"
        elif typ == "cloud.resize" and pl.get("id") in inst:
            inst[pl["id"]]["cores"] = int(pl.get("cores", 0) or 0)
        elif typ == "cmd.rejected" and e.t >= t0 and pl.get("tool") in ("cloud.rent", "cloud.resize") \
                and (pl.get("error") or {}).get("code") == "capacity_exceeded":
            args = pl.get("args") or {}
            prov = args.get("provider") if pl["tool"] == "cloud.rent" else (inst.get(args.get("instance")) or {}
                                                                              ).get("provider")
            if prov not in cap_now and isinstance(prov, str):
                prov = next((k for k, v in _rows(st, "providers") if v.get("name") == prov), prov)
            held = defaultdict(int)
            for x in inst.values():
                if x["running"] and x["provider"] == prov:
                    held[P(x["owner"]) or f"world:{x['owner']}"] += x["cores"]
            refusals.append({"t": int(e.t), "provider": prov, "party": P(pl.get("actor")) or
                             f"world:{pl.get('actor')}", "tool": pl["tool"],
                             "cores_requested": int(args.get("cores", 0) or 0), "holders": dict(held),
                             "capacity_cores": None, "cores_held": int(sum(held.values()))})
    for rf in refusals:
        if rf["provider"] in cap_now:
            rf["capacity_cores"] = cap_now[rf["provider"]] - sum(c for t, pv, c in delivered
                                                                 if pv == rf["provider"] and t > rf["t"])
    # ------------------------------------------------------------------ competition
    by_kind: dict[str, set] = defaultdict(set)
    for _, p in _rows(st, "products"):
        a = P(p.get("owner"))
        if a is not None and P.is_learner(a):
            by_kind[f"product:{p.get('kind')}"].add(a)
    for _, l in _rows(st, "listings"):
        a = P(l.get("seller"))
        if a is not None and P.is_learner(a):
            by_kind[f"listing:{l.get('need') or l.get('kind')}"].add(a)
    for market, ps in sorted(by_kind.items()):
        ps = sorted(ps)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                comp.append({"kind": "same_market", "market": market, "parties": [ps[i], ps[j]],
                             "both_learners": True})
    # scarce compute: a request refused for capacity at a provider while another party held capacity there
    contests: dict[tuple, dict] = {}
    for i_rf, rf in enumerate(refusals):
        for holder, cores in rf["holders"].items():
            a, b = rf["party"], holder
            if a == b or not (P.is_learner(a) or P.is_learner(b)):
                continue
            key = (rf["provider"], a, b)
            c = contests.setdefault(key, {"kind": "scarce_resource", "resource": f"compute:{rf['provider']}",
                                          "parties": [a, b], "refused_party": a, "holder_party": b,
                                          "capacity_cores": rf["capacity_cores"], "refusals": 0,
                                          "holder_cores_max": 0, "utilisation_at_refusal_max": 0.0,
                                          "first_t": rf["t"], "both_learners": P.is_learner(a) and P.is_learner(b),
                                          "refusal_ids": []})
            c["refusals"] += 1
            c["refusal_ids"].append(i_rf)
            c["holder_cores_max"] = max(c["holder_cores_max"], int(cores))
            if rf["capacity_cores"]:
                c["utilisation_at_refusal_max"] = max(c["utilisation_at_refusal_max"],
                                                      round(rf["cores_held"] / rf["capacity_cores"], 4))
    comp += [contests[k] for k in sorted(contests, key=str)]
    shared: list[dict] = []
    rent: dict[str, set] = defaultdict(set)
    used: Counter = Counter()                 # cores currently rented at each provider, by anyone
    for _, row in _rows(st, "instances"):
        if row.get("status") == "running":
            used[row.get("provider")] += int(row.get("cores", 0) or 0)
        a = P(row.get("owner"))
        if a is not None and P.is_learner(a):
            rent[row.get("provider")].add(a)
    for prov, ps in sorted(rent.items()):
        if any(k[0] == prov for k in contests):
            continue
        ps = sorted(ps)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                shared.append({"kind": "shared_capacity", "resource": f"compute:{prov}",
                               "capacity_cores": cap_now.get(prov), "cores_rented": int(used[prov]),
                               "utilisation": round(used[prov] / cap_now[prov], 4) if cap_now.get(prov) else None,
                               "parties": [ps[i], ps[j]], "both_learners": True})
    payers: dict[str, set] = defaultdict(set)
    for r in coop:
        if r["kind"] == "trade":
            a, b = r["parties"]
            if not P.is_learner(a) and P.is_learner(b):
                payers[a].add(b)
    for cust, ps in sorted(payers.items()):
        ps = sorted(ps)
        if len(ps) >= 2:
            comp.append({"kind": "contested_customer", "customer": cust, "parties": ps[:2], "both_learners": True})
    fams: dict[str, set] = defaultdict(set)
    for pk, pkg in _rows(st, "pkg_packages"):
        a = P(pkg.get("owner"))
        if a is None or not P.is_learner(a):
            continue
        for rel in (st.find("pkg_releases", "package", pk) if "pkg_releases" in st.tables else ()):
            for fn in (rel.get("api") or {}):
                fams[re.sub(r"_[a-z0-9]{1,6}$", "", str(fn))].add(a)
    for fam, ps in sorted(fams.items()):
        ps = sorted(ps)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                comp.append({"kind": "competing_tech", "family": fam, "parties": [ps[i], ps[j]],
                             "both_learners": True})
    return {"cooperative": coop, "competitive": comp, "shared_capacity": shared,
            "capacity_refusals": [dict(r, holders=dict(r["holders"])) for r in refusals]}


# ====================================================================== consequences
def consequences(world, controllers: dict[str, str], seq0: int = 0) -> dict:
    """Shared-world consequences (see the module docstring)."""
    P = Parties(world, controllers)
    idents: dict[str, list[str]] = defaultdict(list)
    for p, c in controllers.items():
        idents[c].append(p)
    vis_sets = {c: [world.principals_of(p) for p in ps] for c, ps in idents.items()}
    job_party: dict[str, str] = {}
    tx_party: str | None = None
    created: dict[str, tuple[str, int]] = {}
    visible_to_other = 0
    by_pair: Counter = Counter()
    chains: list[dict] = []
    examples: list[dict] = []
    for e in world.events_range(seq0, world.seq):
        typ = e.type
        if typ == "cmd":
            tx_party = P(e.actor) if e.payload.get("n") is not None and e.actor in controllers else None
            if tx_party is not None:
                # a command naming an id another party's action created: a response chain
                for v in _ids_in(e.payload.get("args")):
                    src = created.get(v)
                    if src is not None and src[0] != tx_party:
                        chains.append({"from": src[0], "to": tx_party, "id": v, "created_seq": src[1],
                                       "response_seq": e.seq, "tool": e.payload.get("tool")})
            continue
        if typ == "proc":
            tx_party = job_party.get(e.payload.get("id"))
            continue
        if typ in ("time", "cmd.rejected"):
            tx_party = None
            continue
        if tx_party is None:
            continue
        if typ == "sched.add":
            job_party[e.payload.get("id")] = tx_party
            continue
        if typ in BOOK:
            continue
        for v in _ids_in(e.payload, top=True):
            created.setdefault(v, (tx_party, e.seq))
        for c, sets in vis_sets.items():
            if c == tx_party:
                continue
            if any(world.visible(e, s) for s in sets):
                visible_to_other += 1
                by_pair[(tx_party, c)] += 1
                if len(examples) < 12:
                    examples.append({"seq": e.seq, "type": typ, "caused_by": tx_party, "seen_by": c})
                break
    return {"events_visible_to_other_parties": visible_to_other,
            "by_pair": {f"{a}->{b}": n for (a, b), n in sorted(by_pair.items())},
            "response_chains": chains, "examples": examples}


def _ids_in(x: Any, top: bool = False, depth: int = 0) -> list[str]:
    out: list[str] = []
    if depth > 4:
        return out
    if isinstance(x, str):
        if ID_TOKEN.match(x) or x.startswith(("o:", "deal_", "prd_", "lst_", "plan_", "proj")):
            out.append(x)
    elif isinstance(x, dict):
        keys = ("id",) if top else tuple(x)
        for k in keys:
            if k in x:
                out += _ids_in(x[k], False, depth + 1)
        if top:
            for k in ("repo", "package", "product", "listing", "deal", "plan", "project", "org", "host"):
                if isinstance(x.get(k), str):
                    out.append(x[k])
    elif isinstance(x, list):
        for v in x[:20]:
            out += _ids_in(v, False, depth + 1)
    return out


# ====================================================================== capabilities (SP-12.0)
def capabilities(inter: dict, *, ventures: list[dict], groups_learning: dict[str, dict],
                 strategy_response: dict | None, parties: set[str] | None = None) -> dict[str, dict]:
    """Which Part 12 capabilities the learning participants exercised, each with its evidence. With `parties`
    (the learning parties), only interactions and ventures in which one of them takes part count."""
    co, cm = inter["cooperative"], inter["competitive"]
    if parties is not None:
        co = [r for r in co if set(r["parties"]) & set(parties)]
        cm = [r for r in cm if set(r["parties"]) & set(parties)]
        ventures = [v for v in ventures if v["party"] in parties]

    def kinds(rs, k, both=False):
        return [r for r in rs if r["kind"] == k and (not both or r.get("both_learners"))]
    out: dict[str, dict] = {}
    sectors = defaultdict(set)
    for v in ventures:
        sectors[v.get("sector") or "?"].add(v["party"])
    founders = {v["party"] for v in ventures}
    same = kinds(cm, "same_market")
    out["found_competing_businesses"] = {"ok": len(founders) >= 2 and (bool(same) or any(len(s) >= 2 for s in
                                                                                        sectors.values())),
                                         "founders": sorted(founders), "same_market_pairs": len(same)}
    collab = kinds(co, "contribution") + kinds(co, "dependency") + \
        [r for r in kinds(co, "grant") if str(r.get("cap", "")).startswith("code")]
    out["collaborate_on_software"] = {"ok": bool(collab), "n": len(collab),
                                      "kinds": sorted({r["kind"] for r in collab})}
    tr = kinds(co, "trade")
    out["trade"] = {"ok": bool(tr), "n": len(tr), "between_learners": sum(1 for r in tr if r["both_learners"])}
    neg = kinds(co, "negotiation")
    out["negotiate"] = {"ok": bool(neg), "n": len(neg), "statuses": dict(Counter(r["status"] for r in neg))}
    mem = kinds(co, "membership")
    out["join_organizations"] = {"ok": bool(mem), "n": len(mem)}
    tech = kinds(cm, "competing_tech")
    out["develop_competing_technologies"] = {"ok": bool(tech), "n": len(tech)}
    indep = [g for g, d in groups_learning.items() if d.get("changed_updates", 0) > 0]
    disjoint = True
    seen_ids: dict[str, str] = {}
    for g in indep:
        for ident in groups_learning[g].get("identities", []):
            if ident in seen_ids and seen_ids[ident] != g:
                disjoint = False
            seen_ids[ident] = g
    out["learn_from_independent_experiences"] = {"ok": len(indep) >= 2 and disjoint, "groups": indep,
                                                 "disjoint_experience": disjoint}
    def provided_by_learning_party(r):
        # the learning party is the one that PROVIDES (paid seller, service owner, depended-on package owner)
        return (r.get("provider") in parties) if parties is not None else bool(r.get("provider_is_learner"))
    used = [r for r in kinds(co, "service_use") + kinds(co, "dependency") +
            [r for r in tr if r.get("ledger_kind") in ("product_sale", "subscription", "license", "api_call",
                                                       "usage")] if provided_by_learning_party(r)]
    out["create_services_used_by_others"] = {"ok": bool(used), "n": len(used),
                                             "kinds": sorted({r["kind"] for r in used}),
                                             "providers": sorted({r["provider"] for r in used}),
                                             "users": sorted({r["user"] for r in used}),
                                             "cents": int(sum(r.get("cents", 0) for r in used))}
    ven = kinds(co, "venture") + [r for r in kinds(co, "agreement") if len(r.get("clauses", [])) >= 1]
    out["form_cooperative_ventures"] = {"ok": bool(ven), "n": len(ven),
                                        "kinds": sorted({r.get("how") or r["kind"] for r in ven})}
    # binding scarcity only: a learning party's request refused for capacity another party held, or a learning
    # party holding capacity another party was refused (a third-party customer paying two parties is reported but
    # is not a scarce resource: nothing shows that the customer's spending was the binding limit)
    sc = kinds(cm, "scarce_resource")
    out["compete_for_scarce_resources"] = {"ok": bool(sc), "n": len(sc), "kinds": sorted({r["kind"] for r in sc}),
                                           # distinct refused requests (one refusal is one record per holder)
                                           "refusals": len({i for r in sc for i in r.get("refusal_ids", [])}),
                                           "resources": sorted({r["resource"] for r in sc}),
                                           "utilisation_at_refusal_max": max(
                                               (r.get("utilisation_at_refusal_max") or 0.0 for r in sc),
                                               default=None),
                                           "contested_customers_reported": len(kinds(cm, "contested_customer")),
                                           # which side the learning parties were on: refused while another party
                                           # held the capacity, or holding capacity another party was refused
                                           "learning_party_refused": sum(1 for r in sc if (r["refused_party"] in parties
                                                                                          if parties is not None else
                                                                                          True)),
                                           "learning_party_holding": sum(1 for r in sc if parties is not None and
                                                                         r["holder_party"] in parties),
                                           "between_learning_parties": sum(1 for r in sc if r.get("both_learners"))}
    sr = strategy_response or {}
    out["respond_to_changing_strategies"] = {"ok": bool(sr.get("responded")), **{k: v for k, v in sr.items()
                                                                                  if k != "responded"}}
    return out


def ventures_of(world, controllers: dict[str, str], t0: int = 0) -> list[dict]:
    P = Parties(world, controllers)
    out = []
    for o, v in _rows(world.state, "ventures"):
        a = P(v.get("founder"))
        if a is not None and P.is_learner(a) and int(v.get("t", 0)) >= t0:
            out.append({"org": o, "party": a, "sector": (v.get("sector") or "")[:40]})
    return out


# ====================================================================== behavioural evidence (Q.7)
def tv_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * float(sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys))


def host_mass(dist: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for k, v in dist.items():
        out[k.split(".", 1)[0]] += v
    return dict(out)


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64)
    b = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64)
    a, b = a / max(a.sum(), 1e-12), b / max(b.sum(), 1e-12)
    m = 0.5 * (a + b)

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log(x[mask] / y[mask])))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def usage_distribution(history: list[dict], *, state_changing_only: bool = False) -> dict[str, float]:
    c = Counter(h["tool"].split(".", 1)[0] for h in history if h.get("tool") and
                (not state_changing_only or (h.get("ok") and not h.get("read_only"))))
    n = sum(c.values())
    return {k: v / n for k, v in c.items()} if n else {}


def bootstrap_ci(x: Iterable[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray(list(x), dtype=np.float64)
    if a.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n, a.size))].mean(axis=1)
    return float(a.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
