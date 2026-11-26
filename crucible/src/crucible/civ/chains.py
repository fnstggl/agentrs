"""Causal chains from event provenance (MASTER_SPEC Part 10.3; INTERFACE_CONTRACTS section 10).

Chains are mined from the event log, not from per-scenario extractors keyed to host names (audit G 5.5). The
provenance graph links an event to the events it depends on by rules that hold for every module:

* transaction: every event of a command or process transaction depends on the transaction's first event (the
  command or the process run);
* schedule: a process run depends on the event that scheduled it (`sched.add` with its job id);
* entity use: an event depends on the latest earlier event that mentioned the same entity (an id in its payload, a
  `<host>.crucible` address in its text, its acting service principal `svc:<host>`), so each entity's events form a
  timeline from its creation onwards; principals, accounts, regions, boards and kernel ids are not entities here
  (they would connect everything);
* column rows: a `cols.batch` on a table whose rows stand for entities (`ROW_ENTITY`, e.g. the population's web
  resource table `res`, row -> site host) mentions the entities of its rows (read from the batch blob);
* explicit references: `refs` in a payload or a batch `extra` (seqs of earlier events) - modules that know a cause
  (e.g. an energy draw names the deployments whose compute it powers) state it.

`trace(world, from_seq)` follows the graph forward from one event and groups what it reaches into stages: the
subsystems (crucible.civ.subsystems) in the order they are first reached. `mine(world)` does that for every
candidate root (an event that creates an entity, by an agent, an autonomous organisation or a population member -
never a scheduled shock) and keeps chains with at least three stages in different subsystems, labelled:

    agent_artifact   the root is an action of an admitted agent that created an artifact or capability
                     (scripted reference agents carry `scripted: true` in their identity and are labelled as such)
    social           a stage is social activity (posts, replies, follows, messages, communities, word of mouth)
    technical        a stage is scientific or technical activity (experiments, papers, replications, process
                     settings, production runs, code, packages, capabilities)

Provenance proposes chains; only counterfactual removal confirms them (crucible.civ.counterfactual).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

from . import subsystems as SUB

BOOKKEEPING = frozenset({"cmd", "cmd.rejected", "ext", "idem", "ctr", "know", "note", "sched.add", "sched.del",
                         "time", "proc", "proc.failed", "principal.add", "principal.update"})
ROOTS = frozenset({"cmd", "proc", "time", "cmd.rejected"})
NOT_ENTITY_TYPES = frozenset({"id.person", "id.org", "id.member", "bank.account", "make.region", "forum.board",
                              "artifact.stored", "cols.batch", "cols.grow", "popweb.page", "net.rate", "net.hit",
                              "society.mind", "society.add"})
HUB = re.compile(r"^(p\d+|p:.*|o:.*|svc:.*|acct_.*|sys|r\d+|board:.*|job_\d+|inv_\d+)$")
HOST_RE = re.compile(r"\b([a-z][a-z0-9-]{2,30})\.crucible\b")
ROW_ENTITY = {"res": ("popweb.resource", "row", "id")}       # table -> (event type, row field, entity field)
SOCIAL_TYPES = ("forum.", "content.", "social.", "community.", "msg.", "creators.", "goods.wom")
TECH_SUBSYSTEMS = {"scientific_research", "software_development", "technology_invention", "industry"}
TECH_TYPES = ("sci.", "make.run", "make.tuned", "make.design", "code.", "pkg.", "caps.", "links.rnd")
SCRIPTED_KEY = "scripted"
NOT_STAGES = frozenset({"substrate", "couplings", "other"})      # storage and bookkeeping are not stages
MAX_STR = 400


def _strings(x, out: list, depth=0):
    if len(out) > MAX_STR or depth > 6:
        return
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            _strings(v, out, depth + 1)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _strings(v, out, depth + 1)


@dataclass
class Provenance:
    events: dict = field(default_factory=dict)          # seq -> Event
    parents: dict = field(default_factory=lambda: defaultdict(set))
    children: dict = field(default_factory=lambda: defaultdict(set))
    created: dict = field(default_factory=dict)         # entity -> seq of the event that first mentioned it
    mentions: dict = field(default_factory=lambda: defaultdict(set))   # seq -> entities
    subsystem: dict = field(default_factory=dict)       # seq -> subsystem (None for bookkeeping)
    roots: dict = field(default_factory=dict)           # seq -> transaction root seq

    def link(self, a: int, b: int) -> None:
        if a != b and a in self.events and b in self.events:
            self.parents[b].add(a)
            self.children[a].add(b)


def _row_entities(world, ev, row_maps) -> list[str]:
    p = ev.payload
    t = p.get("table")
    if t not in row_maps or not row_maps[t]:
        return []
    from ..substrate import artifacts
    try:
        arr = artifacts.get_array_blob(world, p["ref"])
    except Exception:  # noqa: BLE001 - a blob that is gone (compacted) gives no row provenance
        return []
    m = row_maps[t]
    return sorted({m[int(i)] for i in arr["idx"].tolist() if int(i) in m})


def build(world, from_seq: int = 0, to_seq: int | None = None, *, events=None, rows: bool = True) -> Provenance:
    """The provenance graph of events (from_seq, to_seq] of `world` (or of the given event list)."""
    evs = list(events) if events is not None else world.events_range(from_seq, to_seq)
    pv = Provenance()
    hm = SUB.host_module_map(world)
    hosts = set(world.state.ids("hosts")) if "hosts" in world.state.tables else set()
    sched: dict[str, int] = {}
    last: dict[str, int] = {}                    # entity -> latest event seq mentioning it
    row_maps: dict[str, dict[int, str]] = {t: {} for t in ROW_ENTITY}
    for t, (etype, rf, ef) in ROW_ENTITY.items():          # rows created before the window
        st = world.state
        if t == "res" and "pop_res" in st.tables:
            row_maps[t] = {int(r["row"]): h for h, r in st.scan("pop_res")}
    root = None
    for ev in evs:
        s = ev.seq
        pv.events[s] = ev
        typ = ev.type
        if typ in ROOTS:
            root = s
            if typ == "proc":
                j = sched.get(ev.payload.get("id"))
                if j is not None:
                    pv.link(j, s)
        else:
            if root is not None:
                pv.link(root, s)
                pv.roots[s] = root
        if typ == "sched.add":
            sched[ev.payload["id"]] = s
            continue
        pv.subsystem[s] = None if typ in BOOKKEEPING else SUB.subsystem_of_event(world, typ, hm)
        if typ in BOOKKEEPING:
            continue
        p = ev.payload
        ents: set[str] = set()
        strs: list[str] = []
        _strings(p, strs)
        for v in strs:
            if not HUB.match(v) and (v in pv.created or v in hosts):
                ents.add(v)
            for h in HOST_RE.findall(v):
                ents.add(h)
        if isinstance(ev.actor, str) and ev.actor.startswith("svc:"):
            ents.add(ev.actor[4:])
        for t, (etype, rf, ef) in ROW_ENTITY.items():
            if typ == etype:
                row_maps[t][int(p[rf])] = p[ef]
        if typ == "cols.batch" and rows:
            ents.update(_row_entities(world, ev, row_maps))
        new_id = p.get("id") if isinstance(p.get("id"), str) else None
        if typ not in NOT_ENTITY_TYPES:
            for key in ("id", "host"):
                v = p.get(key)
                if isinstance(v, str) and v not in pv.created and not HUB.match(v):
                    pv.created[v] = s
                    if key == "host":
                        hosts.add(v)
        for ref in list(p.get("refs") or []) + list((p.get("extra") or {}).get("refs") or []):
            if isinstance(ref, int):
                pv.link(ref, s)
        for e in sorted(ents):
            if e == new_id and pv.created.get(e) == s:
                continue
            j = last.get(e)
            if j is not None:
                pv.link(j, s)
            elif e in pv.created and pv.created[e] != s:
                pv.link(pv.created[e], s)
            last[e] = s
        if new_id and pv.created.get(new_id) == s:
            last[new_id] = s
        pv.mentions[s] = ents
    return pv


# --------------------------------------------------------------------------------------------- tracing
def _is_social(ev) -> bool:
    return ev.type.startswith(SOCIAL_TYPES) or (ev.type == "cols.batch" and str(ev.payload.get("kind", "")).startswith(
        ("popweb.forum", "goods.wom")))


def _is_tech(ev, sub) -> bool:
    return sub in TECH_SUBSYSTEMS or ev.type.startswith(TECH_TYPES)


def trace(world, from_seq: int, *, pv: Provenance | None = None, horizon_s: int | None = None,
          max_nodes: int = 50_000) -> list[dict]:
    """Stages reached forward from event `from_seq`: [{subsystem, first_seq, first_t, n_events, types}] in order of
    first reach (the root's own subsystem first)."""
    pv = pv or build(world, max(0, from_seq - 1))
    if from_seq not in pv.events:
        raise KeyError(f"event {from_seq} is not in the provenance window")
    t0 = pv.events[from_seq].t
    seen = {from_seq}
    q = deque([from_seq])
    while q and len(seen) < max_nodes:
        u = q.popleft()
        for v in sorted(pv.children.get(u, ())):
            if v in seen:
                continue
            if horizon_s is not None and pv.events[v].t > t0 + horizon_s:
                continue
            seen.add(v)
            q.append(v)
    stages: dict[str, dict] = {}
    for s in sorted(seen):
        sub = pv.subsystem.get(s)
        if sub is None or sub in NOT_STAGES:
            continue
        ev = pv.events[s]
        st = stages.get(sub)
        if st is None:
            st = stages[sub] = {"subsystem": sub, "first_seq": s, "first_t": ev.t, "n_events": 0, "types": Counter(),
                                "social": False, "technical": False, "seqs": []}
        st["n_events"] += 1
        st["types"][ev.type] += 1
        st["social"] = st["social"] or _is_social(ev)
        st["technical"] = st["technical"] or _is_tech(ev, sub)
        if len(st["seqs"]) < 200:
            st["seqs"].append(s)
    out = sorted(stages.values(), key=lambda x: x["first_seq"])
    for st in out:
        st["types"] = dict(st["types"].most_common(8))
    return out


def _root_actor(world, pv: Provenance, s: int) -> tuple[str, str, bool]:
    """(actor, kind, scripted) of the transaction an event belongs to."""
    r = pv.roots.get(s, s)
    ev = pv.events[r]
    actor = ev.actor
    if ev.type == "proc":
        return actor, "process", False
    p = world.state.get("persons", actor) if "persons" in world.state.tables else None
    kind = (p or {}).get("kind") or (world.state.get("principals", actor) or {}).get("kind", "?")
    info = (p or {}).get("info") or {}
    from ..substrate import ids
    if ids.index(actor) is not None:
        kind = "population"
    return actor, kind, bool(info.get(SCRIPTED_KEY))


def candidate_roots(world, pv: Provenance, *, since_seq: int = 0) -> list[int]:
    """Events that create an entity (first mention), excluding bookkeeping and scheduled shocks."""
    out = []
    for e, s in pv.created.items():
        if s <= since_seq:
            continue
        ev = pv.events[s]
        if ev.type.startswith("shocks.") or pv.subsystem.get(s) is None:
            continue
        out.append(s)
    return sorted(set(out))


def mine(world, *, from_seq: int = 0, to_seq: int | None = None, min_stages: int = 3, horizon_s: int | None = None,
         pv: Provenance | None = None, max_roots: int = 400) -> list[dict]:
    """Chains (>= min_stages stages in distinct subsystems) from every candidate root of the window."""
    pv = pv or build(world, from_seq, to_seq)
    chains = []
    roots = candidate_roots(world, pv, since_seq=from_seq)
    # prefer roots by agents and population members, then the rest, deterministic order
    keyed = []
    for s in roots:
        actor, kind, scripted = _root_actor(world, pv, s)
        keyed.append(((0 if kind == "agent" else 1 if kind == "population" else 2), s, actor, kind, scripted))
    keyed.sort()
    for _, s, actor, kind, scripted in keyed[:max_roots]:
        stages = trace(world, s, pv=pv, horizon_s=horizon_s)
        subs = [st["subsystem"] for st in stages]
        if len(set(subs)) < min_stages:
            continue
        ev = pv.events[s]
        entity = next((e for e, c in pv.created.items() if c == s), None)
        chains.append({"root_seq": s, "root_type": ev.type, "root_t": ev.t, "root_entity": entity, "actor": actor,
                       "actor_kind": kind, "scripted": scripted, "stages": stages, "n_stages": len(set(subs)),
                       "agent_artifact": kind == "agent",
                       "social": any(st["social"] for st in stages),
                       "technical": any(st["technical"] for st in stages)})
    return chains


def summarize(chains: list[dict]) -> dict:
    return {"chains": len(chains), "min_stages": min((c["n_stages"] for c in chains), default=0),
            "agent_artifact": sum(c["agent_artifact"] for c in chains), "social": sum(c["social"] for c in chains),
            "technical": sum(c["technical"] for c in chains)}
