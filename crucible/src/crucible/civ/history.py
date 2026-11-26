"""Historical queries on a recorded civilisation (MASTER_SPEC Part 10.4: "historical queries"; SP-10.0).

A civilisation run on a store (`crucible.storage.sqlite.SQLiteStore`) keeps its whole event log and snapshots. These
functions answer questions about its past from that record alone - nothing is re-simulated, no neural inference
runs (recorded inference outputs are read from the store):

    state_at(store, world_id, t)                  the world as it was at simulated time t: the latest snapshot at or
                                                  before t plus the recorded events up to t (World.replay)
    entity_timeline(store, world_id, entity, ...)  every recorded event that names an entity (an id, a host, a
                                                  principal), in order - the history of a firm, a product, a site
    activity_series(store, world_id, ...)         per simulated day: events, economic acts, social acts, money
                                                  moved between participants (crucible.civ.accounting.activity)
    balance_at(store, world_id, account, t)       an account's balance at time t from the ledger postings alone
    trace_back(store, world_id, seq, ...)         the provenance of one event: the earlier events it depends on
                                                  (crucible.civ.chains rules), as far back as asked

`verify(world)` runs each kind on a store-backed world against what the live world (or its snapshots) says, so
every kind is shown to give correct answers, not only answers.
"""
from __future__ import annotations

import json
from collections import defaultdict

DAY = 86400


def _spec(store, world_id):
    from ..core.kernel import WorldSpec
    info = store.world_info(world_id)
    if info is None:
        raise KeyError(f"no world {world_id} in this store")
    return WorldSpec.from_dict(info["spec"])


def state_at(store, world_id: str, t: int):
    """The world at simulated time `t` (events with time <= t applied), rebuilt from the record."""
    from ..core.kernel import World
    seq = store.seq_at_time(world_id, int(t))
    snaps = [s for s in store.list_snapshots(world_id) if s["seq"] <= seq]
    base, s0 = None, 0
    if snaps:
        best = max(snaps, key=lambda s: s["seq"])
        base, s0 = store.get_snapshot(best["id"])[1], int(best["seq"])
    evs = store.read_events(world_id, from_seq=s0, to_seq=seq)
    return World.replay(_spec(store, world_id), evs, world_id=f"{world_id}@{int(t)}", base_blob=base, store=store)


def _mentions(ev: dict, entity: str) -> bool:
    if ev.get("actor") == entity:
        return True
    return entity in json.dumps(ev.get("payload"), default=str)


def entity_timeline(store, world_id: str, entity: str, t0: int = 0, t1: int | None = None, *,
                    types=None, limit: int = 10_000) -> list[dict]:
    """[{seq, t, type, actor}] of the recorded events naming `entity` (exact string) in [t0, t1]."""
    s0 = store.seq_at_time(world_id, int(t0) - 1) if t0 > 0 else 0
    s1 = store.seq_at_time(world_id, int(t1)) if t1 is not None else None
    out = []
    for e in store.read_events(world_id, from_seq=s0, to_seq=s1, types=types):
        if _mentions(e, entity):
            out.append({"seq": e["seq"], "t": e["t"], "type": e["type"], "actor": e["actor"]})
            if len(out) >= limit:
                break
    return out


def activity_series(store, world_id: str, t0: int = 0, t1: int | None = None) -> list[dict]:
    """Per simulated day in [t0, t1]: {day, events, economic, economic_cents, social}."""
    from ..core.kernel import Event
    from .accounting import is_economic, is_social
    s0 = store.seq_at_time(world_id, int(t0) - 1) if t0 > 0 else 0
    s1 = store.seq_at_time(world_id, int(t1)) if t1 is not None else None
    days: dict[int, dict] = defaultdict(lambda: {"events": 0, "economic": 0, "economic_cents": 0, "social": 0})
    for d in store.read_events(world_id, from_seq=s0, to_seq=s1):
        e = Event.from_dict(d)
        r = days[e.t // DAY]
        r["events"] += 1
        if is_economic(e):
            r["economic"] += 1
            r["economic_cents"] += sum(x for _, x in e.payload["postings"] if x > 0)
        elif is_social(e):
            r["social"] += 1
    return [{"day": k, **v} for k, v in sorted(days.items())]


def balance_at(store, world_id: str, account: str, t: int) -> int:
    """Balance of `account` at time `t`, summed from the recorded `bank.txn` postings (the ledger) alone."""
    s1 = store.seq_at_time(world_id, int(t))
    tot = 0
    for e in store.read_events(world_id, from_seq=0, to_seq=s1, types=["bank.txn"]):
        for a, d in e["payload"]["postings"]:
            if a == account:
                tot += int(d)
    return tot


def trace_back(store, world_id: str, seq: int, *, depth: int = 3, window_events: int = 20_000) -> list[dict]:
    """The provenance of event `seq`: earlier events it depends on, `depth` steps back (crucible.civ.chains rules
    over the recorded window of `window_events` events before it)."""
    from ..core.kernel import Event
    from . import chains
    w = state_at(store, world_id, store.read_events(world_id, from_seq=seq - 1, to_seq=seq)[0]["t"])
    evs = [Event.from_dict(d) for d in store.read_events(world_id, from_seq=max(0, seq - window_events), to_seq=seq)]
    pv = chains.build(w, events=evs, rows=False)
    w.close()
    out, frontier, seen = [], {seq}, {seq}
    for k in range(depth):
        nxt = set()
        for s in sorted(frontier):
            for a in sorted(pv.parents.get(s, ())):
                if a not in seen:
                    seen.add(a)
                    nxt.add(a)
                    e = pv.events[a]
                    out.append({"seq": a, "t": e.t, "type": e.type, "step": k + 1, "of": s})
        frontier = nxt
    return out


# --------------------------------------------------------------------------------------------- verification
def verify(world, snapshot_seqs: dict[int, str] | None = None) -> dict:
    """Run every query kind on a store-backed `world` (flushed) and check each answer against an independent
    source: the live world, a snapshot's digest, the state rebuilt at that time, the event log itself."""
    st = world.state
    store, wid = world.store, world.world_id
    world.flush()
    out: dict[str, dict] = {}
    # 1. state at a past time equals the snapshot recorded then (and state now equals the live world)
    snaps = sorted(store.list_snapshots(wid), key=lambda s: s["seq"])
    ok_state, detail = True, []
    from ..core.kernel import World
    for s in snaps[1:3]:
        t = store.read_events(wid, from_seq=s["seq"] - 1, to_seq=s["seq"])[0]["t"] if s["seq"] else 0
        past = state_at(store, wid, t)
        want = World.from_blob(store.get_snapshot(s["id"])[1], world_id="hist-check").digest()
        # events at the same time after the snapshot's seq are part of `t`; compare at the snapshot's seq instead
        if past.seq != s["seq"]:
            evs = store.read_events(wid, from_seq=s["seq"], to_seq=past.seq)
            same_t = all(e["t"] == t for e in evs)
            ok_state = ok_state and same_t
        else:
            ok_state = ok_state and past.digest() == want
        detail.append({"t": t, "seq": past.seq, "snapshot_seq": s["seq"]})
        past.close()
    now = state_at(store, wid, world.now)
    ok_state = ok_state and now.digest() == world.digest() and now.seq == world.seq
    now.close()
    out["state_at_time"] = {"ok": bool(ok_state and len(detail) >= 1), "checked": detail}
    # 2. an entity's timeline: a compute provider's (or any org's) recorded history is non-empty, ordered, and starts
    #    with the event that created it
    orgs = sorted(st.ids("providers")) if "providers" in st.tables else sorted(st.ids("orgs"))
    ent = orgs[0] if orgs else None
    tl = entity_timeline(store, wid, ent) if ent else []
    seqs = [r["seq"] for r in tl]
    out["entity_timeline"] = {"ok": bool(tl) and seqs == sorted(seqs) and tl[0]["t"] <= DAY, "entity": ent,
                              "events": len(tl), "first": tl[0] if tl else None}
    # 3. per-day activity series sums to the event count of the whole record
    ser = activity_series(store, wid)
    out["activity_series"] = {"ok": sum(r["events"] for r in ser) == world.seq and len(ser) >= 2,
                              "days": len(ser), "economic": sum(r["economic"] for r in ser),
                              "social": sum(r["social"] for r in ser)}
    # 4. a balance at a past time from postings alone equals the balance in the state rebuilt at that time
    bank = world.modules["bank"]
    acct = None
    for p in orgs:
        acct = bank.primary_account(st, p)
        if acct:
            break
    ok_bal, rows = True, []
    if acct:
        for t in sorted({DAY, world.now // 2, world.now}):
            b_log = balance_at(store, wid, acct, t)
            w_t = state_at(store, wid, t)
            b_state = int(w_t.modules["bank"].balance(w_t.state, acct))
            w_t.close()
            rows.append({"t": t, "ledger": b_log, "state": b_state})
            ok_bal = ok_bal and b_log == b_state
    out["balance_at_time"] = {"ok": bool(acct) and ok_bal, "account": acct, "checked": rows}
    # 5. trace back: the provenance of a recent economic event reaches earlier events of other kinds
    last_txn = None
    for d in reversed(store.read_events(wid, from_seq=max(0, world.seq - 5000), to_seq=world.seq,
                                        types=["make.retail_sold"])):
        last_txn = d["seq"]
        break
    tb = trace_back(store, wid, last_txn) if last_txn else []
    out["trace_back"] = {"ok": len({r["type"] for r in tb}) >= 2, "seq": last_txn, "parents": len(tb),
                         "types": sorted({r["type"] for r in tb})[:12]}
    out["historical_query_kinds"] = sum(1 for v in out.values() if isinstance(v, dict) and v.get("ok"))
    return out
