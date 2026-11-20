"""Researcher-side analytics over a world's event log and authoritative state (read-only).

* `trace`          - causal debugging: entity timelines, "why did X happen" explanations, causal chains.
* `metrics_export` - Prometheus text exposition and JSON snapshots of world metrics.

The helpers in this module are shared by the analytics, dashboard, training and evaluation tooling. None of them
mutates the world: they read `World.state`, the in-memory log and the durable store (following fork lineage).
"""
from __future__ import annotations

from typing import Iterable, Iterator

from ..core.kernel import Event

DAY = 86400
HOUR = 3600
_CHUNK = 20_000

# event types that are bookkeeping rather than world happenings (hidden from timelines by default)
NOISE_TYPES = frozenset({"ctr", "sched.add", "sched.del", "proc", "ext", "idem", "time", "cloud.usage",
                         "cloud.usage_bulk", "cloud.revenue", "consumers.voices_done", "search.index",
                         "search.unindex", "ads.reset", "support.stats"})


def _read_lineage(store, world_id: str, from_seq: int, to_seq: int | None) -> list[dict]:
    """Event dicts with from_seq < seq <= to_seq for a world, following fork parents for the shared prefix."""
    info = store.world_info(world_id)
    if info is None:
        return []
    par = info.get("parent")
    out: list[dict] = []
    if par and from_seq < par["seq"]:
        out = _read_lineage(store, par["world"], from_seq, par["seq"] if to_seq is None else min(to_seq, par["seq"]))
        from_seq = par["seq"]
    if to_seq is None or to_seq > from_seq:
        out.extend(store.read_events(world_id, from_seq, to_seq))
    return out


def world_events(world, from_seq: int = 0, to_seq: int | None = None,
                 types: Iterable[str] | None = None, prefixes: Iterable[str] | None = None) -> Iterator[Event]:
    """Events with from_seq < seq <= to_seq, oldest first: the durable store (following fork lineage) for the part
    before the in-memory window, then the in-memory log. Events that are neither in memory nor in a store (an
    in-memory fork's pre-fork history) are skipped. Never mutates the world (no flush)."""
    to_seq = world.seq if to_seq is None else min(int(to_seq), world.seq)
    tset = set(types) if types is not None else None
    pre = tuple(prefixes) if prefixes else None

    def keep(e: Event) -> bool:
        if tset is not None and e.type not in tset and not (pre and e.type.startswith(pre)):
            return False
        if tset is None and pre and not e.type.startswith(pre):
            return False
        return True

    lo = int(from_seq)
    if lo < world.log_base and world.store is not None:
        hi = min(to_seq, world.log_base)
        s = lo
        while s < hi:
            e_hi = min(hi, s + _CHUNK)
            for d in _read_lineage(world.store, world.world_id, s, e_hi):
                ev = Event.from_dict(d)
                if keep(ev):
                    yield ev
            s = e_hi
    lo = max(lo, world.log_base)
    if to_seq > lo:
        for ev in world.log[lo - world.log_base: to_seq - world.log_base]:
            if keep(ev):
                yield ev


def first_available_seq(world) -> int:
    """The lowest seq (exclusive) from which `world_events` can return the full history."""
    return 0 if (world.store is not None or world.log_base == 0) else world.log_base


def has_table(world, name: str) -> bool:
    return name in world.state.tables


def slot_map(world) -> dict[int, dict]:
    """Consumer-market slot -> {listing, host, seller, need} (every offering the consumer module ever registered)."""
    st = world.state
    out: dict[int, dict] = {}
    if not has_table(world, "cons_offers"):
        return out
    for lid, o in st.scan("cons_offers"):
        l = st.get("listings", lid) if has_table(world, "listings") else None
        out[int(o["slot"])] = {"listing": lid, "host": l["host"] if l else None, "seller": l["seller"] if l else None,
                               "need": o["need"], "since": o.get("t", 0)}
    return out


def host_owner(world, host: str) -> str | None:
    if not has_table(world, "hosts"):
        return None
    h = world.state.get("hosts", host)
    return h["owner"] if h else None


def hosts_of(world, owner: str) -> list[str]:
    """Hosts ever deployed by an owner (organisation or person)."""
    st = world.state
    if not has_table(world, "deployments"):
        return []
    return sorted({st.tables["deployments"][d]["host"] for d in st.find_ids("deployments", "owner", owner)})


def accounts_of(world, owner: str) -> list[str]:
    if not has_table(world, "accounts"):
        return []
    return world.state.find_ids("accounts", "owner", owner)


def owned_orgs(world, person: str, roles=("owner",)) -> list[str]:
    if not has_table(world, "members"):
        return []
    return sorted(m["org"] for m in world.state.find("members", "person", person) if m["role"] in roles)


def resolve_entity(world, entity: str) -> tuple[str, str]:
    """Normalise a user-supplied entity (principal id, handle, host, account, listing, ...) -> (id, kind).
    kind is one of: org, person, service, host, account, listing, contract, ticket, instance, deployment, repo,
    unknown."""
    st = world.state
    e = entity.strip()
    if st.has("principals", e):
        k = st.tables["principals"][e]["kind"]
        if e.startswith("svc:"):
            return e, "service"
        return e, {"organisation": "org", "org": "org", "person": "person", "system": "system"}.get(k, k)
    for pre, kind in (("o:", "org"), ("p:", "person")):
        if st.has("principals", pre + e):
            return pre + e, kind
    for table, kind in (("hosts", "host"), ("accounts", "account"), ("listings", "listing"),
                        ("contracts", "contract"), ("sup_tickets", "ticket"), ("instances", "instance"),
                        ("deployments", "deployment"), ("repos", "repo"), ("claims", "claim")):
        if has_table(world, table) and st.has(table, e):
            return e, kind
    return e, "unknown"


def money(cents: float | int | None) -> str:
    if cents is None:
        return "-"
    c = int(round(cents))
    sign = "-" if c < 0 else ""
    c = abs(c)
    return f"{sign}${c // 100:,}.{c % 100:02d}"


def day_of(t: int) -> int:
    return int(t) // DAY
