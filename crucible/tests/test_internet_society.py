"""Optional LLM-driven simulated people: the LLM proposes, the person's persistent preferences and the authoritative
services decide; LLM outputs are recorded so worlds replay exactly."""
import crucible

crucible.load_all_modules()
from crucible.core.kernel import World  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def _w(backend, seed=1):
    w = World.create(generate({"family": "open_digital", "overrides": {
        "n_consumers": 200, "workers": 8, "params": {"always_reply": True, "llm_minds": {"backend": backend}}}}, seed))
    w.system_call("admin.join", {"handle": "ada", "endowment": 5_000_000})
    ex = lambda t, **k: w.execute("p:ada", {"tool": t, "args": k})  # noqa: E731
    me = w.modules["bank"].primary_account(w.state, "p:ada")
    ex("id.create_org", handle="adaco", name="Ada", kind="company")
    ex("biz.register", org="o:adaco", account=me)
    acct = ex("bank.open_account", owner="o:adaco").result["account"]
    ex("bank.transfer", from_account=me, to=acct, amount=4_000_000)
    return w, ex, acct


def _offer(w, ex, acct, mult):
    p = sorted(k for k, m in w.state.scan("minds") if m["kind"] == "person")[0]
    res = w.state.tables["minds"][p]["reservation"]
    terms = [{"type": "membership", "person": p, "org": "o:adaco", "role": "member"},
             {"type": "payment", "from": "o:adaco", "to": p, "amount": int(res * mult * 7 / 30), "every_s": 7 * DAY,
              "count": 3, "start_in_s": 0}, {"type": "notice", "seconds": 7 * DAY}]
    r = ex("deal.propose", to=p, as_org="o:adaco", terms=terms, title="offer", message="Please join, we pay well.",
           accounts={"1": acct})
    assert r.ok, r.error
    w.advance_time(w.now + 2 * DAY)
    return r.result["deal"]


def test_an_llm_cannot_make_a_person_accept_against_its_own_reservation():
    w, ex, acct = _w("test:accept")
    did = _offer(w, ex, acct, 0.05)
    d = w.state.tables["deals"][did]
    assert d["status"] in ("rejected", "countered")
    dec = [e.payload for e in w.log if e.type == "society.llm_decision"]
    assert dec and dec[0]["llm_action"] == "accept" and dec[0]["grounded_action"] != "accept" and dec[0]["overridden"]


def test_llm_decisions_within_preferences_are_executed_and_replay_exactly():
    w, ex, acct = _w("test:reject")
    did = _offer(w, ex, acct, 2.0)
    assert w.state.tables["deals"][did]["status"] == "rejected"          # a person may always decline
    replies = [d for _, d in w.state.scan("deals") if d["parent"] == did]
    w2 = World.regenerate(w.spec, [e.to_dict() for e in w.log])
    assert w2.last_hash == w.last_hash and w2.digest() == w.digest()
    del replies
