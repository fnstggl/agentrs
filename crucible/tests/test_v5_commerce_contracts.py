"""Crucible 5.0 escrowed commitments (module `commit`): negotiation threads, escrow deposits, the three deliverable
kinds, breaches, and dispute resolution by settlement, arbitration and default."""
import crucible

crucible.load_all_modules()
from crucible.commerce import orgworld as OW  # noqa: E402
from crucible.commerce.contracts import ESCROW  # noqa: E402

DAY = 86400


def _world(seed=1):
    w = OW.create(seed)
    client, c_acct = OW.join(w, "client", 2_000_000)
    dev, d_acct = OW.join(w, "dev", 100_000)
    judge, _ = OW.join(w, "judge", 0)
    return w, client, dev, judge, c_acct


def _ex(w, a, t, **k):
    return OW.act(w, a, t, **k)


def _terms(amounts=(300_000,), kind="acceptance", due=10 * DAY, **extra):
    ms = [{"amount_cents": a, "deliverable": {"kind": kind} if kind == "acceptance" else kind, "due_in_s": due}
          for a in amounts]
    return {"title": "build a site", "payer": "$me", "milestones": ms, **extra}


def _escrow(w):
    return w.state.tables["accounts"][ESCROW]["balance"]


def _agree(w, client, dev, terms):
    o = _ex(w, client, "commit.offer", to="dev", terms=terms)
    assert o.ok, o.error
    r = _ex(w, dev, "commit.respond", offer=o.result["offer"], action="accept")
    assert r.ok, r.error
    return r.result["commitment"]


def test_negotiation_thread_counter_and_escrow_at_acceptance():
    w, client, dev, judge, c_acct = _world(1)
    o1 = _ex(w, client, "commit.offer", to="dev", terms=_terms((200_000,)), message="200 for the site").result["offer"]
    assert _escrow(w) == 0                                          # an offer binds nothing
    r = _ex(w, dev, "commit.respond", offer=o1, action="counter", message="needs 350",
            terms={"title": "build a site", "payer": "$you", "milestones": [
                {"amount_cents": 350_000, "deliverable": {"kind": "acceptance"}, "due_in_s": 14 * DAY}]})
    assert r.ok, r.error
    o2 = r.result["offer"]
    assert not _ex(w, dev, "commit.respond", offer=o1, action="accept").ok      # superseded
    o3 = _ex(w, client, "commit.respond", offer=o2, action="counter",
             terms=_terms((300_000,), due=14 * DAY)).result["offer"]
    before = OW.balance(w, client)
    cid = _ex(w, dev, "commit.respond", offer=o3, action="accept").result["commitment"]
    assert OW.balance(w, client) == before - 300_000 and _escrow(w) == 300_000
    thread = _ex(w, client, "commit.thread", offer=o3).result
    assert [t["round"] for t in thread] == [0, 1, 2]
    assert not _ex(w, judge, "commit.thread", offer=o3).ok                       # private to the parties
    assert w.state.tables["commitments"][cid]["status"] == "active"
    assert w.check_invariants() == []


def test_delivery_acceptance_and_silence_release_the_money():
    w, client, dev, judge, c_acct = _world(2)
    cid = _agree(w, client, dev, _terms((100_000, 200_000)))
    assert not _ex(w, client, "commit.deliver", commitment=cid, milestone=0).ok   # only the payee delivers
    assert _ex(w, dev, "commit.deliver", commitment=cid, milestone=0, note="v1", ref="repo:dev/site").ok
    before = OW.balance(w, dev)
    assert _ex(w, client, "commit.accept", commitment=cid, milestone=0).ok
    assert OW.balance(w, dev) == before + 100_000
    assert _ex(w, dev, "commit.deliver", commitment=cid, milestone=1).ok
    w.advance_time(w.now + 3 * DAY + 3600)                                         # the client says nothing
    assert OW.balance(w, dev) == before + 300_000 and _escrow(w) == 0
    assert w.state.tables["commitments"][cid]["status"] == "completed"
    assert w.modules["commit"].record(w, dev)["completed"] == 1


def test_missed_due_refunds_the_payer_and_is_a_public_breach():
    w, client, dev, judge, c_acct = _world(3)
    before = OW.balance(w, client)
    cid = _agree(w, client, dev, _terms((150_000,), due=2 * DAY))
    w.advance_time(w.now + 2 * DAY + 3600)
    c = w.state.tables["commitments"][cid]
    assert c["status"] == "breached" and c["breaches"][0]["by"] == dev
    assert OW.balance(w, client) == before and _escrow(w) == 0
    assert any(e.type == "commit.breach" and e.vis == "public" for e in w.log)
    assert w.modules["commit"].record(w, dev)["breaches"] == 1


def test_predicate_deliverable_is_checked_by_the_world():
    w, client, dev, judge, c_acct = _world(4)
    pred = {"metric": "cash", "args": {"principal": dev}, "op": ">=", "value": 500_000}
    cid_ok = _agree(w, client, dev, {"title": "raise", "payer": "$me", "milestones": [
        {"amount_cents": 50_000, "deliverable": {"kind": "predicate", "predicate": pred}, "due_in_s": 5 * DAY}]})
    cid_no = _agree(w, client, dev, {"title": "raise2", "payer": "$me", "milestones": [
        {"amount_cents": 60_000, "deliverable": {"kind": "predicate", "predicate": dict(pred, value=10**9)},
         "due_in_s": 5 * DAY}]})
    w.system_call("admin.endow", {"principal": dev, "amount": 450_000})
    w.advance_time(w.now + 5 * DAY + 3600)
    assert w.state.tables["commitments"][cid_ok]["milestones"][0]["status"] == "released"
    assert w.state.tables["commitments"][cid_no]["milestones"][0]["status"] == "refunded"
    # a predicate about somebody outside the agreement cannot be signed
    r = _ex(w, client, "commit.offer", to="dev", terms={"title": "spy", "payer": "$me", "milestones": [
        {"amount_cents": 1_000, "deliverable": {"kind": "predicate", "predicate": dict(pred, args={"principal": judge})},
         "due_in_s": DAY}]})
    assert not r.ok and r.error["code"] == "forbidden"


def test_dispute_resolved_by_negotiated_settlement():
    w, client, dev, judge, c_acct = _world(5)
    cid = _agree(w, client, dev, _terms((200_000,)))
    _ex(w, dev, "commit.deliver", commitment=cid, milestone=0, note="done")
    assert _ex(w, client, "commit.dispute", commitment=cid, milestone=0, reason="half of it is missing").ok
    assert w.state.tables["commitments"][cid]["status"] == "disputed"
    w.advance_time(w.now + 4 * DAY)                                                # frozen: no silent release
    assert w.state.tables["commitments"][cid]["milestones"][0]["status"] == "disputed"
    c0, d0 = OW.balance(w, client), OW.balance(w, dev)
    assert _ex(w, client, "commit.settle", commitment=cid, milestone=0, payee_share=0.4).result["status"] == "proposed"
    assert _ex(w, dev, "commit.settle", commitment=cid, milestone=0, payee_share=0.7).result["status"] == "proposed"
    assert _ex(w, dev, "commit.settle", commitment=cid, milestone=0, payee_share=0.4).result["status"] == "resolved"
    assert OW.balance(w, dev) == d0 + 80_000 and OW.balance(w, client) == c0 + 120_000
    assert w.state.tables["commitments"][cid]["milestones"][0]["outcome"]["how"] == "settlement"


def test_dispute_resolved_by_an_independent_arbiter_who_is_paid_its_fee():
    w, client, dev, judge, c_acct = _world(6)
    r = _ex(w, client, "commit.offer", to="dev", terms=_terms((200_000,), arbiter="judge", arbiter_fee_cents=10_000))
    cid = _ex(w, dev, "commit.respond", offer=r.result["offer"], action="accept").result["commitment"]
    _ex(w, dev, "commit.deliver", commitment=cid, milestone=0)
    _ex(w, client, "commit.dispute", commitment=cid, milestone=0, reason="late and buggy")
    assert not _ex(w, dev, "commit.rule", commitment=cid, milestone=0, payee_share=1.0).ok     # parties don't rule
    j0, d0 = OW.balance(w, judge), OW.balance(w, dev)
    assert _ex(w, judge, "commit.rule", commitment=cid, milestone=0, payee_share=0.75, reason="mostly done").ok
    assert OW.balance(w, judge) == j0 + 10_000 and OW.balance(w, dev) == d0 + 142_500
    assert w.modules["commit"].record(w, judge)["arbitrations_served"] == 1
    assert _escrow(w) == 0 and w.check_invariants() == []


def test_unresolved_dispute_takes_the_default_split_and_arbiters_must_be_independent():
    w, client, dev, judge, c_acct = _world(7)
    cid = _agree(w, client, dev, _terms((100_000,), default_split=0.3, dispute_s=2 * DAY))
    _ex(w, dev, "commit.deliver", commitment=cid, milestone=0)
    _ex(w, client, "commit.dispute", commitment=cid, milestone=0, reason="no")
    d0 = OW.balance(w, dev)
    w.advance_time(w.now + 2 * DAY + 3600)
    m = w.state.tables["commitments"][cid]["milestones"][0]
    assert m["outcome"] == {"payee_share": 0.3, "how": "default"} and OW.balance(w, dev) == d0 + 30_000
    org, _ = OW.found_company(w, client, "clientco", 0)
    OW.add_member(w, org, client, judge)                         # the judge now works with the client
    r = _ex(w, client, "commit.offer", to="dev", terms=_terms((100_000,), arbiter="judge"))
    assert not r.ok and "independent" in r.error["message"]


def test_a_limited_spending_grant_cannot_commit_money_to_escrow_and_cancel_is_mutual():
    w, client, dev, judge, c_acct = _world(8)
    org, o_acct = OW.found_company(w, client, "buyer", 500_000)
    assert _ex(w, client, "id.grant", grantee="judge", cap="bank.spend", owner=org, limit=50_000).ok
    r = _ex(w, judge, "commit.offer", to="dev", terms=_terms((40_000,)), as_org=org, account=o_acct)
    assert not r.ok                                               # judge is no admin of the org
    cid = _agree(w, client, dev, _terms((100_000, 100_000)))
    _ex(w, dev, "commit.deliver", commitment=cid, milestone=0)
    _ex(w, client, "commit.accept", commitment=cid, milestone=0)
    c0 = OW.balance(w, client)
    assert _ex(w, client, "commit.cancel", commitment=cid).result["status"] == "cancellation requested"
    assert _ex(w, dev, "commit.cancel", commitment=cid).result["status"] == "completed"
    assert OW.balance(w, client) == c0 + 100_000 and _escrow(w) == 0


def test_task_deliverable_pays_when_the_payee_did_the_inst_task():
    w, client, dev, judge, c_acct = _world(9)
    org, o_acct = OW.found_company(w, client, "studio", 500_000)
    OW.add_member(w, org, client, dev)
    prj = _ex(w, client, "inst.project_create", title="site", as_org=org).result["project"]
    t = _ex(w, client, "inst.task_add", project=prj, title="frontend", effort_hours=10).result["task"]
    r = _ex(w, client, "commit.offer", to="dev", as_org=org, account=o_acct, terms={"title": "frontend", "payer": "$me",
            "milestones": [{"amount_cents": 90_000, "deliverable": {"kind": "task", "task": t}, "due_in_s": 7 * DAY}]})
    assert r.ok, r.error
    cid = _ex(w, dev, "commit.respond", offer=r.result["offer"], action="accept").result["commitment"]
    assert not _ex(w, dev, "commit.deliver", commitment=cid, milestone=0).ok      # the task is not done yet
    _ex(w, client, "inst.task_assign", task=t, assignee="dev")
    _ex(w, dev, "inst.task_submit", task=t)
    _ex(w, client, "inst.task_review", task=t, accept=True)
    d0 = OW.balance(w, dev)
    assert _ex(w, dev, "commit.deliver", commitment=cid, milestone=0).ok
    assert OW.balance(w, dev) == d0 + 90_000
    rep = OW.verify_reproducible(w)
    assert rep["replay_digest_equal"] and rep["regeneration_hash_equal"]


def test_every_commitment_is_one_contract_with_an_escrow_clause():
    """SP-2.3 contract migration: a commitment creates its `contracts` record (parties, an escrow clause naming it,
    signed by both, active), the record ends with the commitment (completed / breached by the payee that missed
    its due date), agents cannot write escrow clauses themselves, and no contract fact is kept twice."""
    from crucible.substrate import facts, primitives
    w, client, dev, judge, c_acct = _world(31)
    assert primitives.owner("contract")["table"] == "contracts" and facts.REVIEWED["commit.commitments"]["role"] == "part"
    cid = _agree(w, client, dev, _terms((100_000,)))
    k = w.state.tables["commitments"][cid]["contract"]
    c = w.state.tables["contracts"][k]
    assert c["status"] == "active" and sorted(c["parties"]) == sorted([client, dev])
    assert c["clauses"] == [{"type": "escrow", "instrument": "commit", "ref": cid}] and set(c["signed"]) == set(c["parties"])
    assert _ex(w, dev, "commit.deliver", commitment=cid, milestone=0, note="done").ok
    assert _ex(w, client, "commit.accept", commitment=cid, milestone=0).ok
    assert w.state.tables["commitments"][cid]["status"] == "completed"
    assert w.state.tables["contracts"][k]["status"] == "completed"
    # a payee that misses its due date breaches: the contract record says so, with who breached
    cid2 = _agree(w, client, dev, _terms((50_000,), due=2 * DAY))
    k2 = w.state.tables["commitments"][cid2]["contract"]
    w.advance_time(w.now + 3 * DAY)
    assert w.state.tables["contracts"][k2]["status"] == "breached"
    assert w.state.tables["contracts"][k2]["breached_by"] == dev
    # nobody but the instrument writes escrow clauses
    r = _ex(w, client, "contracts.propose", parties=[client, dev], title="fake",
            clauses=[{"type": "escrow", "instrument": "commit", "ref": cid}])
    assert not r.ok and "instrument" in r.error["message"]
    assert w.check_invariants() == []
