"""crucible.civ.history: historical queries answered from the record alone, checked against the live world (a small
store-backed world, so this runs without the civilisation's language-model path; the civilisation-scale check is
crucible.civ.gates.persistence / tests/v5/test_gate_SP_10_0.py)."""
import pytest

import crucible

crucible.load_all_modules()

from crucible.civ import history  # noqa: E402
from crucible.core.kernel import World, WorldSpec  # noqa: E402
from crucible.storage.sqlite import SQLiteStore  # noqa: E402

DAY = 86400


@pytest.fixture()
def recorded(tmp_path):
    store = SQLiteStore(str(tmp_path / "h.sqlite"))
    spec = WorldSpec("hist", 1, [{"name": "identity", "config": {}}, {"name": "bank", "config": {}},
                                 {"name": "admin", "config": {}}], {})
    w = World.create(spec, world_id="h1", store=store)
    w.snapshot()
    bank = w.modules["bank"]
    w.system_call("admin.join", {"handle": "ann", "kind": "agent", "endowment": 10_000})
    w.system_call("admin.join", {"handle": "bob", "kind": "agent", "endowment": 5_000})
    w.advance_time(DAY)
    w.snapshot()
    acct_b = bank.primary_account(w.state, "p:bob")
    balances = {"day1": bank.balance(w.state, acct_b)}
    acct_a = bank.primary_account(w.state, "p:ann")
    w.advance_time(DAY + 3600)
    r = w.execute("p:ann", {"tool": "bank.transfer", "args": {"from_account": acct_a, "to": "bob", "amount": 1_234,
                                                              "memo": "rent"}})
    assert r.ok, r.error
    w.advance_time(2 * DAY)
    balances["day2"] = bank.balance(w.state, acct_b)
    w.flush()
    yield w, store, acct_b, balances
    w.close()


def test_state_and_balance_at_a_past_time(recorded):
    w, store, acct, bal = recorded
    past = history.state_at(store, "h1", DAY)
    assert past.modules["bank"].balance(past.state, acct) == bal["day1"]
    assert history.balance_at(store, "h1", acct, DAY) == bal["day1"]
    assert history.balance_at(store, "h1", acct, 2 * DAY) == bal["day2"] == bal["day1"] + 1_234
    now = history.state_at(store, "h1", w.now)
    assert now.digest() == w.digest() and now.seq == w.seq


def test_entity_timeline_and_activity_series(recorded):
    w, store, acct, bal = recorded
    tl = history.entity_timeline(store, "h1", acct)
    assert tl and [r["seq"] for r in tl] == sorted(r["seq"] for r in tl)
    assert tl[0]["t"] == 0 and any(r["type"] == "bank.txn" and r["t"] > DAY for r in tl)
    ser = history.activity_series(store, "h1")
    assert sum(r["events"] for r in ser) == w.seq
    assert any(r["economic"] >= 1 for r in ser if r["day"] == 1)         # the transfer between two participants
