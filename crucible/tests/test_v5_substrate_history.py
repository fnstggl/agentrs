"""Historical queries (crucible.substrate.history): past state at a time or seq from snapshots + replay, the event
history of an entity (including a population individual's cohort decisions), record and column histories - on a
real SQLite-stored population world."""
import numpy as np
import pytest

import crucible
from crucible.core.errors import NotFound, ProtocolError
from crucible.core.kernel import World
from crucible.population.worlds import population_world
from crucible.storage.sqlite import SQLiteStore
from crucible.substrate import columns, history, ids

crucible.load_all_modules()
DAY = 86400


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    db = tmp_path_factory.mktemp("hist") / "w.sqlite"
    st = SQLiteStore(str(db))
    w = World.create(population_world("development", 4, n=240, sites=6, adaptive=False, strategic=False),
                     world_id="hist", store=st)
    a = w.system_call("admin.join", {"handle": "hist-a", "endowment": 90_000})["principal"]
    b = w.system_call("admin.join", {"handle": "hist-b", "endowment": 10_000})["principal"]
    bank = w.modules["bank"]
    marks = []
    for k in range(4):
        w.advance_time(w.now + 6 * 3600)
        r = w.execute(a, {"tool": "bank.transfer", "args": {"from_account": bank.primary_account(w.state, a),
                                                            "to": b, "amount": 1000 + k, "memo": f"h{k}"}})
        assert r.ok, r.error
        marks.append((w.seq, w.now, w.digest()))
        if k == 1:
            w.snapshot()
    w.flush()
    return {"world": w, "store": st, "a": a, "b": b, "marks": marks}


def _direct(w, seq):
    """Reference: replay the first `seq` events of the in-memory log, no snapshot involved."""
    return World.replay(w.spec, [e.to_dict() for e in w.log[:seq]], world_id="direct")


@pytest.mark.timeout(600)
def test_state_at_seq_and_time_from_snapshot_plus_replay(stored):
    w, st = stored["world"], stored["store"]
    snap_seq = st.list_snapshots("hist")[0]["seq"]
    for seq, t, dig in stored["marks"]:
        hw = history.state_at((st, "hist"), seq=seq)
        assert hw.seq == seq and hw.digest() == dig == _direct(w, seq).digest()
        info = hw.runtime["history"]
        assert info["base_snapshot_seq"] == (snap_seq if seq >= snap_seq else 0)
        assert history.state_at(w, seq=seq).digest() == dig          # the live world as source
        # at time t: every event with event time <= t
        assert history.seq_at_time((st, "hist"), t) >= seq and history.state_at(w, t=t).now <= t
    mid = (stored["marks"][0][0] + stored["marks"][1][0]) // 2
    assert history.state_at((st, "hist"), seq=mid).digest() == _direct(w, mid).digest()
    with pytest.raises(NotFound):
        history.state_at((st, "hist"), seq=w.seq + 5)


def test_historical_worlds_are_read_only(stored):
    hw = history.state_at((stored["store"], "hist"), seq=stored["marks"][0][0])
    with pytest.raises(ProtocolError):
        hw.store.append_events("hist", [])
    with pytest.raises(ProtocolError):
        hw.store.put_blob("art/x", b"x")


def test_record_history_follows_one_account_through_time(stored):
    w, a, b = stored["world"], stored["a"], stored["b"]
    bank = w.modules["bank"]
    acct = bank.primary_account(w.state, b)
    hist = history.record_history((stored["store"], "hist"), "accounts", acct)
    bals = [h["record"]["balance"] for h in hist if h["record"] is not None]
    # opened empty, the endowment, then the four incoming transfers - and nothing else ever touched it
    assert bals == [0, 10_000, 11_000, 12_001, 13_003, 14_006] and bals[-1] == bank.balance(w.state, acct)
    assert hist[0]["type"] == "bank.account" and all(h["type"] == "bank.txn" for h in hist[1:])


@pytest.mark.timeout(600)
def test_events_of_an_individual_include_its_cohort_decisions(stored):
    w, st = stored["world"], stored["store"]
    evs = history.events_of((st, "hist"), ids.pid(7), types=["cols.batch"])
    cohort = [e for e in evs if e["role"] == "cohort"]
    assert cohort and all(e["row"] == 7 and e["payload"]["table"] == "pop" for e in cohort)
    # the decisions agree with the column's recorded history: every change of the cell is a cohort event of row 7
    col = next(c for c in ("satisfaction", "interests", "attention_spent_min") if c in
               columns._registry(w)["pop"]["columns"])
    ch = history.column_history((st, "hist"), "pop", col, 7)
    changed = {h["seq"] for h in ch[1:]}
    assert changed and changed <= {e["seq"] for e in cohort}
    assert np.allclose(np.asarray(ch[-1]["value"]), np.asarray(columns.col(w, "pop", col)[7]))
    # an agent principal: its own commands and the events addressed to it
    mine = history.events_of(w, stored["a"], since_seq=stored["marks"][0][0] - 50)
    assert any(e["role"] == "actor" and e["type"] == "cmd" for e in mine)
    assert any(e["type"] == "bank.txn" for e in mine)
