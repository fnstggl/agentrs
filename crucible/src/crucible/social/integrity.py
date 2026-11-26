"""Engagement integrity (evaluator side): which engagement a creator or organisation received is *organic*.

In-world counters (follower counts, votes, kudos, comments, fan memberships) stay raw, as on real platforms: they
are what participants see, and fake engagement can move them. Evaluator measurements of social success must not
be movable that way. `classify(world, target)` labels every principal that engaged with the target's group (the
target, the organisations it owns or administers and the services they operate) from authoritative state:

* ``related``  - a related party of the target's group (`goals.metrics.is_related` either way: the same controller,
                 an organisation one founded or administers, shared spending grants, a service's operator);
* ``funded``   - received money from the target's group (directly, or through one intermediary that the group paid)
                 before engaging: engagement bought with the target's own money;
* ``sybil``    - an admitted agent whose engagement is concentrated on the target's group (at least
                 ``CONCENTRATION`` of its engagements, ``MIN_EDGES`` or more) and that has no footprint of its own (no
                 content, no organisation, no paid purchase from anyone outside the group): an account that exists
                 to engage with one target;
* ``ring``     - an admitted agent in a reciprocal-follow ring with another engager of the target that is itself
                 not organic;
* ``population`` / ``organic`` - population individuals (their decisions are recorded batches of the decision
                 engine) and every other unrelated principal.

`organic_counts` returns the counts per engagement kind with and without the excluded classes. The two metrics
registered here have no subject, so no in-world agreement can read them (`check_predicate_scope`).

Limits (KNOWN_OPEN in `crucible.redteam.v5.suite`): identity relies on the operator's controller labels; a set of
colluding agents with distinct controllers, independent funding and a believable footprint of their own is not
distinguishable from an audience by state alone.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

CONCENTRATION = 0.6
MIN_EDGES = 2
EXCLUDED = ("related", "funded", "sybil", "ring")


def group_of(world, target: str) -> set[str]:
    st = world.state
    out = {target}
    if "members" in st.tables:
        for m in st.find("members", "person", target):
            if m["role"] in ("owner", "admin"):
                out.add(m["org"])
    if "orgs" in st.tables:
        for oid, o in st.scan("orgs"):
            if o.get("founder") == target:
                out.add(oid)
    for pid, rec in st.scan("principals"):
        if rec["kind"] == "service" and rec["info"].get("operator") in out:
            out.add(pid)
    return out


def engagements(world, group: set[str]) -> list[tuple[str, str, str]]:
    """(engager, kind, target member) of every current engagement with the group's members and their items."""
    st = world.state
    out: list[tuple[str, str, str]] = []
    if "social" in world.modules:
        from . import graph, nodes
        for g in sorted(group):
            n = nodes.node_of(world, g)
            if n is None:
                continue
            for p in nodes.principals_of_nodes(world, graph.followers(world, n)):
                if p:
                    out.append((p, "follow", g))
    items: dict[str, str] = {}
    if "content" in st.tables:
        for iid, c in st.scan("content"):
            if c.get("creator") in group:
                items[iid] = c["creator"]
    if "content_comments" in st.tables:
        for _, c in st.scan("content_comments"):
            if c.get("item") in items:
                out.append((c["author"], "comment", items[c["item"]]))
    for tname, vt in (("launches", "launch_votes"),):
        if tname in st.tables and vt in st.tables:
            mine = {lid: l.get("target") or l.get("by") for lid, l in st.scan(tname)}
            for _, v in st.scan(vt):
                owner = mine.get(v.get("item"))
                if owner in group:
                    out.append((v["voter"], "upvote", owner))
    for t in list(st.tables):
        if t.endswith("_votes") and t.replace("_votes", "_posts") in st.tables:
            posts = st.tables[t.replace("_votes", "_posts")]
            for _, v in st.scan(t):
                p = posts.get(v.get("post"))
                if p is not None and p.get("author") in group:
                    out.append((v["voter"], "vote", p["author"]))
    if "creator_members" in st.tables:
        for _, m in st.scan("creator_members"):
            if m.get("creator") in group and m.get("status") == "active":
                out.append((m["fan"], "membership", m["creator"]))
    return [e for e in out if e[0] not in group]


def _accounts(st, principals) -> set[str]:
    return {a for p in principals for a in st.find_ids("accounts", "owner", p)} if "accounts" in st.tables else set()


def _paid_by(world, payers: set[str]) -> set[str]:
    """Principals that received money from `payers`' accounts, directly or through one intermediary."""
    st = world.state
    if "txns" not in st.tables or "accounts" not in st.tables:
        return set()
    owner = {a: r["owner"] for a, r in st.tables["accounts"].items()}
    src = _accounts(st, payers)
    edges: dict[str, set[str]] = defaultdict(set)
    for _, t in st.scan("txns"):
        posts = t.get("postings") or []
        outs = {a for a, d in posts if d < 0}
        ins = {a for a, d in posts if d > 0}
        for a in outs:
            edges[a] |= ins
    first = {b for a in src for b in edges.get(a, ())} - src
    second = {c for b in first for c in edges.get(b, ())} - src
    return {owner[a] for a in first | second if a in owner}


def classify(world, target: str) -> dict[str, str]:
    from ..goals.metrics import is_related
    st = world.state
    group = group_of(world, target)
    eng = engagements(world, group)
    engagers = sorted({e for e, _, _ in eng})
    funded = _paid_by(world, group)
    kinds = {p: (st.get("principals", p) or {}).get("kind") for p in engagers}
    # concentration over the whole world: how much of an agent's engagement goes to this group
    edges_all: dict[str, int] = defaultdict(int)
    from . import graph, nodes
    fv = graph.FOLLOW.view(world) if "social" in world.modules else None
    if fv is not None:
        src = fv["src"][fv["active"]]
        u, c = np.unique(src, return_counts=True)
        for n_, k in zip(u.tolist(), c.tolist()):
            p = nodes.principal_of(world, int(n_))
            if p:
                edges_all[p] += int(k)
    for e, kind, _ in eng:
        if kind != "follow":
            edges_all[e] += 1
    on_group: dict[str, int] = defaultdict(int)
    for e, _, _ in eng:
        on_group[e] += 1
    label: dict[str, str] = {}
    for p in engagers:
        if p.startswith("p") and p[1:].isdigit():
            label[p] = "population"
        elif any(is_related(world, p, g) or is_related(world, g, p) for g in group):
            label[p] = "related"
        elif p in funded:
            label[p] = "funded"
        elif kinds.get(p) == "agent" and on_group[p] >= MIN_EDGES and \
                on_group[p] >= CONCENTRATION * max(1, edges_all.get(p, on_group[p])) and not _footprint(world, p, group):
            label[p] = "sybil"
        else:
            label[p] = "organic"
    if fv is not None:                                  # reciprocal rings with excluded engagers
        pairs = set(zip(fv["src"][fv["active"]].tolist(), fv["dst"][fv["active"]].tolist()))
        node = {p: nodes.node_of(world, p) for p in engagers}
        bad = {p for p, l in label.items() if l in EXCLUDED}
        for p in engagers:
            if label[p] != "organic" or node[p] is None or kinds.get(p) != "agent":
                continue
            for q in bad:
                if node.get(q) is not None and (node[p], node[q]) in pairs and (node[q], node[p]) in pairs:
                    label[p] = "ring"
                    break
    return label


def _footprint(world, p: str, group: set[str]) -> bool:
    st = world.state
    if "content" in st.tables and st.find_ids("content", "creator", p):
        return True
    if "members" in st.tables and any(m["role"] in ("owner", "admin") for m in st.find("members", "person", p)):
        return True
    if "txns" in st.tables and "accounts" in st.tables:
        mine = _accounts(st, {p})
        theirs = _accounts(st, group)
        for a in mine:
            for lid in st.indexes.get(("acct_txns", "acct"), {}).get(a, ()):
                t = st.tables["txns"].get(st.tables["acct_txns"][lid].get("txn")) if "acct_txns" in st.tables else None
                if not t:
                    continue
                ins = {x for x, d in t.get("postings", []) if d > 0}
                if t.get("kind") in ("sale", "purchase", "subscription") and ins and not ins & (theirs | mine):
                    return True
    return False


def organic_counts(world, target: str) -> dict:
    label = classify(world, target)
    eng = engagements(world, group_of(world, target))
    raw: dict[str, int] = defaultdict(int)
    org: dict[str, int] = defaultdict(int)
    for e, kind, _ in eng:
        raw[kind] += 1
        if label.get(e) not in EXCLUDED:
            org[kind] += 1
    by_class: dict[str, int] = defaultdict(int)
    for v in label.values():
        by_class[v] += 1
    return {"raw": dict(raw), "organic": dict(org), "engagers": dict(by_class)}


def _register():
    from ..goals.metrics import metric

    @metric("organic_followers", "EVALUATOR-ONLY. Followers of a principal's group that are neither related, funded "
            "by the group, sybil-like nor in a ring with such accounts (crucible.social.integrity).")
    def m_organic_followers(world, principal: str) -> float:
        return float(organic_counts(world, principal)["organic"].get("follow", 0))

    @metric("organic_engagement", "EVALUATOR-ONLY. All organic engagements (follows, comments, votes, upvotes, fan "
            "memberships) with a principal's group (crucible.social.integrity).")
    def m_organic_engagement(world, principal: str) -> float:
        return float(sum(organic_counts(world, principal)["organic"].values()))


_register()
