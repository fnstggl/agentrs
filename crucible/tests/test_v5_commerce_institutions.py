"""Crucible 5.0 institutions (module `inst`): charters, decision procedures, enforcement at authorisation,
restructuring, shared projects and ventures, and the stable constraints institutions cannot change."""
import crucible

crucible.load_all_modules()
from crucible.commerce import orgworld as OW  # noqa: E402

DAY = 86400
MAJ = {"type": "majority", "electorate": "members", "quorum": 0.5}


def _coop(seed=1, n=3, cash=1_000_000):
    w = OW.create(seed)
    people = [OW.join(w, h, max(2_000_000, cash + 1_000_000))[0] for h in ("ann", "bob", "cat", "dan", "eve")[:n]]
    org, acct = OW.found_company(w, people[0], "acme", cash)
    for p in people[1:]:
        OW.add_member(w, org, people[0], p)
    return w, people, org, acct


def _ex(w, a, t, **k):
    return OW.act(w, a, t, **k)


def _propose(w, who, org, action):
    r = _ex(w, who, "inst.propose", org=org, action=action)
    assert r.ok, r.error
    return r.result["decision"]


def _status(w, did):
    return w.state.tables["inst_decisions"][did]["status"]


def test_charter_rule_binds_owners_and_majority_vote_executes_as_the_org():
    w, (ann, bob, cat), org, acct = _coop()
    charter = {"procedures": {"maj": MAJ}, "rules": [{"match": {"kind": "tool", "tool": "bank.transfer",
                                                               "min_amount": 10_000}, "procedure": "maj"}]}
    assert _ex(w, ann, "inst.adopt", org=org, charter=charter).ok
    r = _ex(w, ann, "bank.transfer", from_account=acct, to=ann, amount=50_000)
    assert not r.ok and "charter requires a majority" in r.error["message"]       # the owner too
    assert _ex(w, ann, "bank.transfer", from_account=acct, to=bob, amount=5_000).ok   # below the rule's amount
    before = OW.balance(w, cat)
    did = _propose(w, bob, org, {"kind": "tool", "tool": "bank.transfer",
                                 "args": {"from_account": acct, "to": cat, "amount": 40_000}})
    assert _status(w, did) == "open"                                              # the proposer's own vote: 1 of 3
    assert _ex(w, cat, "inst.vote", decision=did, choice="yes").result["status"] == "executed"
    assert OW.balance(w, cat) == before + 40_000
    assert not _ex(w, ann, "inst.vote", decision=did, choice="no").ok            # decided: no more votes
    assert w.check_invariants() == []


def test_rejection_lapse_and_quorum():
    w, (ann, bob, cat), org, acct = _coop(2)
    assert _ex(w, ann, "inst.adopt", org=org, charter={
        "procedures": {"maj": dict(MAJ, quorum=0.67), "board": {"type": "majority", "electorate": "admins"}},
        "rules": [{"match": {"kind": "office"}, "procedure": "board"}], "default": "maj"}).ok
    d1 = _propose(w, ann, org, {"kind": "unit_create", "name": "ops"})
    _ex(w, bob, "inst.vote", decision=d1, choice="no")
    _ex(w, cat, "inst.vote", decision=d1, choice="no")
    assert _status(w, d1) == "rejected"
    d2 = _propose(w, ann, org, {"kind": "unit_create", "name": "ops"})          # only the proposer votes:
    w.advance_time(w.now + 3 * DAY)                                              # 1 of 3 is below the quorum
    assert _status(w, d2) == "rejected"
    d3 = _propose(w, bob, org, {"kind": "office", "title": "clerk", "holder": "p:bob"})   # bob is no admin
    assert dict(w.state.tables["inst_decisions"][d3]["electorate"]) == {ann: 1.0}
    w.advance_time(w.now + 3 * DAY)
    assert _status(w, d3) == "lapsed"                                            # nobody voted


def test_weighted_by_shares_and_electorate_fixed_at_proposal():
    w, (ann, bob, cat), org, acct = _coop(3)
    biz = w.modules["business"]
    w._run_system("cap table", lambda ctx: (biz.transfer_shares(ctx, org, ann, bob, 300_000),
                                            biz.transfer_shares(ctx, org, ann, cat, 100_000)))
    charter = {"procedures": {"w": {"type": "weighted", "weights": "shares", "electorate": "shareholders"}},
               "default": "w"}
    assert _ex(w, ann, "inst.adopt", org=org, charter=charter).ok
    did = _propose(w, bob, org, {"kind": "unit_create", "name": "lab"})
    el = dict(w.state.tables["inst_decisions"][did]["electorate"])
    assert el == {ann: 600_000.0, bob: 300_000.0, cat: 100_000.0}
    # buying shares after the proposal does not change who decides it
    w._run_system("late buy", lambda ctx: biz.transfer_shares(ctx, org, ann, cat, 500_000))
    _ex(w, cat, "inst.vote", decision=did, choice="yes")
    assert _status(w, did) == "open"                                              # 40% of the fixed weights
    _ex(w, ann, "inst.vote", decision=did, choice="yes")
    assert _status(w, did) == "executed"


def test_delegated_votes_flow_along_chains_direct_votes_override_cycles_abstain():
    w, people, org, acct = _coop(4, n=5)
    ann, bob, cat, dan, eve = people
    assert _ex(w, ann, "inst.adopt", org=org, charter={"procedures": {"liquid": {"type": "delegated"}},
                                                       "default": "liquid"}).ok
    assert _ex(w, cat, "inst.delegate_vote", org=org, to="bob").ok
    assert _ex(w, dan, "inst.delegate_vote", org=org, to="cat").ok                # dan -> cat -> bob
    did = _propose(w, bob, org, {"kind": "unit_create", "name": "design"})       # bob's own vote carries
    assert _status(w, did) == "executed"                                          # cat's and dan's: 3 of 5
    assert _ex(w, eve, "inst.delegate_vote", org=org, to="ann").ok
    assert _ex(w, ann, "inst.delegate_vote", org=org, to="eve").ok                # a cycle: both abstain
    d2 = _propose(w, dan, org, {"kind": "unit_create", "name": "sales"})        # dan votes directly (yes)
    counts = lambda: w.modules["inst"].counts(w.state, dict(w.state.tables["inst_decisions"][d2], id=d2))  # noqa
    assert counts() == (1.0, 0.0, 5.0)        # cat -> bob has not voted: abstains; ann <-> eve: a cycle abstains
    _ex(w, cat, "inst.vote", decision=d2, choice="no")
    _ex(w, bob, "inst.vote", decision=d2, choice="no")
    assert counts() == (1.0, 2.0, 5.0) and _status(w, d2) == "open"
    _ex(w, ann, "inst.vote", decision=d2, choice="no")                            # ann now votes: eve follows
    assert counts() == (1.0, 4.0, 5.0) and _status(w, d2) == "rejected"


def test_consensus_blocks_on_one_objection_and_passes_on_silence():
    w, (ann, bob, cat), org, acct = _coop(5)
    assert _ex(w, ann, "inst.adopt", org=org, charter={"procedures": {"c": {"type": "consensus", "window_s": DAY}},
                                                       "default": "c"}).ok
    d1 = _propose(w, ann, org, {"kind": "unit_create", "name": "a"})
    _ex(w, cat, "inst.vote", decision=d1, choice="no")
    assert _status(w, d1) == "rejected"
    d2 = _propose(w, ann, org, {"kind": "unit_create", "name": "b"})
    w.advance_time(w.now + DAY + 3600)
    assert _status(w, d2) == "executed"


def test_officer_acts_alone_within_monthly_budget():
    w, (ann, bob, cat), org, acct = _coop(6)
    charter = {"procedures": {"maj": MAJ, "treasurer": {"type": "officer", "office": "treasurer"}},
               "offices": {"treasurer": {"holder": "p:bob", "budget_cents_month": 60_000,
                                         "tools": ["bank.transfer"]}},
               "rules": [{"match": {"kind": "tool", "tool": "bank.transfer", "min_amount": 100_000}, "procedure": "maj"},
                         {"match": {"kind": "tool", "tool": "bank.transfer"}, "procedure": "treasurer"}]}
    assert _ex(w, ann, "inst.adopt", org=org, charter=charter).ok
    # a plain-member officer decides alone through inst.propose (executed at once, charged to the office)
    r = _ex(w, bob, "inst.propose", org=org, action={"kind": "tool", "tool": "bank.transfer",
                                                    "args": {"from_account": acct, "to": cat, "amount": 40_000}})
    assert r.ok and r.result["status"] == "executed"
    r = _ex(w, bob, "inst.propose", org=org, action={"kind": "tool", "tool": "bank.transfer",
                                                    "args": {"from_account": acct, "to": cat, "amount": 30_000}})
    assert r.ok and r.result["status"] == "failed"                                       # 70k > the 60k budget
    r = _ex(w, ann, "bank.transfer", from_account=acct, to=cat, amount=10_000)
    assert not r.ok and "treasurer" in r.error["message"]                                # the owner is not the officer
    # an officer who also holds identity authority (an admin) acts directly, within the same budget
    assert _ex(w, ann, "id.set_role", org=org, person="bob", role="admin").ok
    assert _ex(w, bob, "bank.transfer", from_account=acct, to=cat, amount=20_000).ok
    assert not _ex(w, bob, "bank.transfer", from_account=acct, to=cat, amount=1_000).ok   # budget used up
    w.advance_time(w.now + 31 * DAY)
    did = _propose(w, cat, org, {"kind": "tool", "tool": "bank.transfer",
                                 "args": {"from_account": acct, "to": cat, "amount": 50_000}})
    assert _ex(w, bob, "inst.vote", decision=did, choice="yes").result["status"] == "executed"   # new month, budget
    big = _propose(w, bob, org, {"kind": "tool", "tool": "bank.transfer",
                                 "args": {"from_account": acct, "to": cat, "amount": 150_000}})
    assert w.state.tables["inst_decisions"][big]["proc_name"] == "maj"                 # large sums go to a vote


def test_amendment_changes_the_procedure_through_the_amendment_rule():
    w, (ann, bob, cat), org, acct = _coop(7)
    charter = {"procedures": {"maj": MAJ, "all": {"type": "unanimous"}},
               "rules": [{"match": {"kind": "amend"}, "procedure": "all"}], "default": "maj"}
    assert _ex(w, ann, "inst.adopt", org=org, charter=charter).ok
    assert not _ex(w, ann, "inst.adopt", org=org, charter=charter).ok                    # only once
    new = {"procedures": {"tenure": {"type": "weighted", "weights": "tenure"}, "all": {"type": "unanimous"}},
           "rules": [{"match": {"kind": "amend"}, "procedure": "all"}], "default": "tenure"}
    did = _propose(w, bob, org, {"kind": "amend", "charter": new})
    _ex(w, ann, "inst.vote", decision=did, choice="yes")
    assert _status(w, did) == "open"                                                      # unanimity needs cat
    _ex(w, cat, "inst.vote", decision=did, choice="yes")
    assert _status(w, did) == "executed"
    rec = w.state.tables["inst_charters"][org]
    assert rec["version"] == 2 and rec["charter"]["default"] == "tenure"
    d2 = _propose(w, ann, org, {"kind": "unit_create", "name": "x"})
    assert w.state.tables["inst_decisions"][d2]["procedure"]["weights"] == "tenure"


def test_both_institutions_apply_and_a_gov_proposal_cannot_bypass_the_charter():
    w, (ann, bob, cat), org, acct = _coop(8)
    assert _ex(w, ann, "inst.adopt", org=org, charter={"procedures": {"maj": MAJ}, "rules": [
        {"match": {"kind": "tool", "tool": "bank.transfer"}, "procedure": "maj"}]}).ok
    # a 4.0 gov proposal approved by the owner alone executes as the organisation - and is refused by the charter
    p = _ex(w, ann, "gov.propose", org=org, tool="bank.transfer",
            args={"from_account": acct, "to": ann, "amount": 30_000}).result["proposal"]
    assert _ex(w, ann, "gov.vote", proposal=p, approve=True).ok
    assert w.state.tables["proposals"][p]["status"] == "failed"
    # and where a gov constitution governs a tool too, an inst decision alone cannot execute it
    assert _ex(w, ann, "gov.adopt", org=org, rules=[{"match": {"tool": "bank.transfer"},
                                                    "require": {"type": "vote", "electorate": "members"}}]).ok
    did = _propose(w, bob, org, {"kind": "tool", "tool": "bank.transfer",
                                 "args": {"from_account": acct, "to": bob, "amount": 30_000}})
    _ex(w, bob, "inst.vote", decision=did, choice="yes")
    _ex(w, cat, "inst.vote", decision=did, choice="yes")
    assert _status(w, did) == "failed"
    assert "constitution" in w.state.tables["inst_decisions"][did]["result"]["message"]


def test_membership_admission_rules_expulsion_and_offices_vacate_on_leaving():
    w, (ann, bob, cat), org, acct = _coop(9)
    dan, _ = OW.join(w, "dan", 0)
    eve, _ = OW.join(w, "eve", 0)
    assert not _ex(w, dan, "inst.join", org=org).ok                       # no charter: invitation only
    charter = {"procedures": {"maj": MAJ}, "default": "maj", "membership": {"admission": "decision", "max_members": 4},
               "offices": {"clerk": {"holder": "p:cat", "budget_cents_month": 0}}}
    assert _ex(w, ann, "inst.adopt", org=org, charter=charter).ok
    r = _ex(w, dan, "inst.join", org=org, note="I build things")
    assert r.ok and r.result["applied"]
    _ex(w, ann, "inst.vote", decision=r.result["decision"], choice="yes")
    _ex(w, bob, "inst.vote", decision=r.result["decision"], choice="yes")
    assert w.modules["identity"].role(w.state, org, dan) == "member"
    r = _ex(w, eve, "inst.join", org=org)                                 # the charter's limit of 4 members
    assert not r.ok and "limit" in r.error["message"]
    did = _propose(w, ann, org, {"kind": "expel", "person": "p:cat"})
    _ex(w, bob, "inst.vote", decision=did, choice="yes")
    _ex(w, dan, "inst.vote", decision=did, choice="yes")
    assert w.modules["identity"].role(w.state, org, cat) is None
    assert w.state.tables["inst_offices"][f"{org}|clerk"]["holder"] is None      # the office is vacant
    assert not _ex(w, ann, "inst.propose", org=org, action={"kind": "expel", "person": "p:ann"}).ok   # owners stay


def test_units_restructuring_and_spin_off_into_a_funded_company():
    w, people, org, acct = _coop(10, n=4, cash=3_000_000)
    ann, bob, cat, dan = people
    assert _ex(w, ann, "inst.adopt", org=org, charter={"procedures": {"maj": MAJ}, "default": "maj"}).ok

    def decide(who, action, voters):
        did = _propose(w, who, org, action)
        for v in voters:
            if _status(w, did) == "open":
                _ex(w, v, "inst.vote", decision=did, choice="yes")
        return w.state.tables["inst_decisions"][did]

    d = decide(ann, {"kind": "unit_create", "name": "research", "head": "p:bob", "budget_cents_month": 20_000},
               [bob, cat])
    unit = d["result"]["unit"]
    assert decide(ann, {"kind": "unit_assign", "unit": unit, "person": "p:cat"}, [bob, cat])["status"] == "executed"
    assert w.state.tables["inst_offices"][f"{org}|unit:{unit}"]["holder"] == bob
    assert decide(cat, {"kind": "spin_off", "unit": unit, "handle": "acme-labs"}, [ann, bob])["status"] == "failed"
    d = decide(bob, {"kind": "spin_off", "unit": unit, "handle": "acme-labs", "name": "Acme Labs",
                     "budget_cents": 500_000, "parent_stake": 0.2}, [ann, cat])
    assert d["status"] == "executed", d["result"]
    new = d["result"]["org"]
    assert w.modules["identity"].role(w.state, new, bob) == "owner"
    assert OW.balance(w, new) == 500_000
    assert w.state.tables["shares"][f"{new}|{org}"]["n"] == 200_000
    assert any(i["person"] == cat and i["org"] == new for i in w.state.tables["invites"].values())
    assert w.state.tables["inst_units"][unit]["status"] == "dissolved"
    assert w.check_invariants() == []


def test_merge_requires_both_decisions_and_moves_money_projects_and_people():
    w, (ann, bob, cat), a_org, a_acct = _coop(11, cash=400_000)
    zed, _ = OW.join(w, "zed", 2_000_000)
    b_org, b_acct = OW.found_company(w, zed, "bigco", 100_000)
    prj = _ex(w, ann, "inst.project_create", title="widget", as_org=a_org).result["project"]
    assert not _ex(w, zed, "inst.propose", org=b_org, action={"kind": "absorb", "org": a_org}).ok   # no offer yet
    for who in (ann,):
        did = _propose(w, who, a_org, {"kind": "merge", "into": b_org})        # no charter: owners unanimously
        _ex(w, ann, "inst.vote", decision=did, choice="yes")
    assert w.state.tables["inst_merge_offers"][a_org]["into"] == b_org
    did = _propose(w, zed, b_org, {"kind": "absorb", "org": a_org})
    _ex(w, zed, "inst.vote", decision=did, choice="yes")
    assert _status(w, did) == "executed", w.state.tables["inst_decisions"][did]["result"]
    assert w.state.tables["orgs"][a_org]["status"] == "merged"
    assert OW.balance(w, b_org) == 500_000 and OW.balance(w, a_org) == 0
    assert w.state.tables["inst_projects"][prj]["owner"] == b_org
    assert {i["person"] for i in w.state.tables["invites"].values() if i["org"] == b_org} == {ann, bob, cat}
    assert w.check_invariants() == []


def test_dissolution_pays_creditors_then_distributes_by_the_charter_basis():
    w, (ann, bob, cat), org, acct = _coop(12, cash=900_000)
    assert _ex(w, ann, "inst.adopt", org=org, charter={"procedures": {"all": {"type": "unanimous"}},
                                                       "default": "all"}).ok
    did = _propose(w, cat, org, {"kind": "dissolve"})
    for v in (ann, bob, cat):
        _ex(w, v, "inst.vote", decision=did, choice="yes")
    assert _status(w, did) == "executed"
    assert w.state.tables["companies"][org]["status"] == "dissolved"
    assert w.state.tables["orgs"][org]["status"] == "dissolved"
    assert OW.balance(w, org) == 0                                       # shareholders were paid (ann holds all)
    assert w.check_invariants() == []


def test_project_dependencies_allocation_and_rewards_on_review():
    w, (ann, bob, cat), org, acct = _coop(13, cash=1_000_000)
    prj = _ex(w, ann, "inst.project_create", title="app", as_org=org, deadline_days=30).result["project"]
    assert _ex(w, ann, "inst.project_fund", project=prj, account=acct, amount=300_000).ok
    t1 = _ex(w, ann, "inst.task_add", project=prj, title="api", effort_hours=40, skills={"software": 0.6}).result["task"]
    t2 = _ex(w, ann, "inst.task_add", project=prj, title="ui", effort_hours=20, skills={"design": 0.5},
             depends_on=[t1]).result["task"]
    assert not _ex(w, ann, "inst.task_add", project=prj, title="x", effort_hours=1, depends_on=["task_999999"]).ok
    assert _ex(w, ann, "inst.allocate", project=prj, task=t1, amount=100_000).ok
    assert not _ex(w, ann, "inst.allocate", project=prj, task=t2, amount=250_000).ok      # over the free pool
    assert _ex(w, ann, "inst.task_assign", task=t1, assignee="bob").ok
    assert _ex(w, ann, "inst.task_assign", task=t2, assignee="cat").ok
    assert not _ex(w, cat, "inst.task_submit", task=t2).ok                                 # blocked by t1
    assert not _ex(w, bob, "inst.task_assign", task=t1, assignee="cat").ok                 # bob does not lead
    before = OW.balance(w, bob)
    assert _ex(w, bob, "inst.task_submit", task=t1, note="done").ok
    assert _ex(w, ann, "inst.task_review", task=t1, accept=True).ok
    assert OW.balance(w, bob) == before + 100_000
    assert _ex(w, cat, "inst.task_submit", task=t2).ok
    p = w.state.tables["inst_projects"][prj]
    assert (p["pool"], p["allocated"]) == (200_000, 0)
    r = _ex(w, ann, "inst.project_close", project=prj).result
    assert r["refunded"] == 200_000 and OW.balance(w, org) == 900_000
    assert w.check_invariants() == []


def test_lead_delegation_through_an_identity_grant():
    w, (ann, bob, cat), org, acct = _coop(14)
    prj = _ex(w, ann, "inst.project_create", title="p", as_org=org).result["project"]
    t = _ex(w, ann, "inst.task_add", project=prj, title="t", effort_hours=5).result["task"]
    assert not _ex(w, bob, "inst.task_assign", task=t, assignee="cat").ok
    assert _ex(w, ann, "id.grant", grantee="bob", cap="inst.lead", owner=org, scope=prj).ok
    assert _ex(w, bob, "inst.task_assign", task=t, assignee="cat").ok
    assert _ex(w, bob, "inst.task_add", project=prj, title="t2", effort_hours=3).ok


def test_cooperative_venture_decides_by_contribution_and_returns_the_pool_pro_rata():
    w = OW.create(15)
    ann, _ = OW.join(w, "ann", 3_000_000)
    zed, _ = OW.join(w, "zed", 3_000_000)
    a, a_acct = OW.found_company(w, ann, "alpha", 1_000_000)
    b, b_acct = OW.found_company(w, zed, "beta", 1_000_000)
    r = _ex(w, ann, "inst.project_create", title="joint", as_org=a, partners=[b],
            procedure={"type": "weighted", "weights": "contribution", "threshold": 0.5})
    prj = r.result["project"]
    assert not _ex(w, zed, "inst.project_fund", project=prj, account=b_acct, amount=1_000).ok   # not joined yet
    assert _ex(w, zed, "inst.project_join", project=prj, as_org=b).ok
    assert _ex(w, ann, "inst.project_fund", project=prj, account=a_acct, amount=300_000).ok
    assert _ex(w, zed, "inst.project_fund", project=prj, account=b_acct, amount=100_000).ok
    t = _ex(w, ann, "inst.task_add", project=prj, title="build", effort_hours=10).result["task"]
    assert not _ex(w, ann, "inst.allocate", project=prj, task=t, amount=50_000).ok       # a venture decides
    did = _propose(w, zed, prj, {"kind": "allocate", "task": t, "amount": 50_000})
    assert dict(w.state.tables["inst_decisions"][did]["electorate"]) == {a: 300_000.0, b: 100_000.0}
    _ex(w, zed, "inst.vote", decision=did, choice="yes", as_org=b)
    assert _status(w, did) == "open"                                                       # beta has 25%
    _ex(w, ann, "inst.vote", decision=did, choice="yes", as_org=a)
    assert _status(w, did) == "executed"
    did = _propose(w, ann, prj, {"kind": "project_close"})
    _ex(w, ann, "inst.vote", decision=did, choice="yes", as_org=a)
    assert _status(w, did) == "executed"
    assert OW.balance(w, a) == 1_000_000 and OW.balance(w, b) == 1_000_000      # 3:1 pro rata refund of 400k
    assert w.check_invariants() == []


def test_stable_constraints_charters_and_decisions_cannot_reach_the_laws():
    w, (ann, bob, cat), org, acct = _coop(16)
    bad = [{"accounting": {"overdraft": 10**9}}, {"ledger": {}}, {"permissions": {"p:bob": "*"}},
           {"evaluation": {"independent": True}}, {"metrics": {}}, {"state": {"accounts": {}}}, {"clock": 0}]
    for extra in bad:
        r = _ex(w, ann, "inst.adopt", org=org, charter={"procedures": {"maj": MAJ}, **extra})
        assert not r.ok and r.error["code"] == "forbidden", extra
    r = _ex(w, ann, "inst.adopt", org=org, charter={"procedures": {"maj": MAJ}, "rules": [
        {"match": {"kind": "tool", "tool": "bank.transfer",
                   "when": {"metric": "quality_margin_vs_incumbents", "args": {"org": org, "need": "x"}, "op": ">",
                            "value": 0}}, "procedure": "maj"}]})
    assert not r.ok and "only condition on its own data" in r.error["message"]       # evaluator-only metric
    assert _ex(w, ann, "inst.adopt", org=org, charter={"procedures": {"maj": MAJ}, "default": "maj"}).ok
    for action in ({"kind": "tool", "tool": "admin.endow", "args": {"principal": org, "amount": 10**9}},
                   {"kind": "set_state", "table": "accounts"},
                   {"kind": "role", "person": "p:bob", "role": "owner"}):
        assert not _ex(w, ann, "inst.propose", org=org, action=action).ok, action
    zed, z_acct = OW.join(w, "zed", 1_000_000)
    did = _propose(w, ann, org, {"kind": "tool", "tool": "bank.transfer",
                                 "args": {"from_account": z_acct, "to": acct, "amount": 500_000}})
    for v in (ann, bob, cat):
        if _status(w, did) == "open":
            _ex(w, v, "inst.vote", decision=did, choice="yes")
    assert _status(w, did) == "failed" and OW.balance(w, zed) == 1_000_000            # others' money is not theirs


def test_institution_world_replays_and_regenerates_exactly():
    w, (ann, bob, cat), org, acct = _coop(17)
    assert _ex(w, ann, "inst.adopt", org=org, charter={"procedures": {"liquid": {"type": "delegated"}},
                                                       "default": "liquid"}).ok
    _ex(w, cat, "inst.delegate_vote", org=org, to="bob")
    did = _propose(w, bob, org, {"kind": "unit_create", "name": "ops", "head": "p:cat"})
    _ex(w, bob, "inst.vote", decision=did, choice="yes")
    prj = _ex(w, ann, "inst.project_create", title="p", as_org=org).result["project"]
    _ex(w, ann, "inst.project_fund", project=prj, account=acct, amount=10_000)
    w.advance_time(w.now + 3 * DAY)
    rep = OW.verify_reproducible(w)
    assert rep["replay_digest_equal"] and rep["regeneration_hash_equal"]


def test_repeated_applications_are_refused_while_one_is_pending():
    w, (ann, bob, cat), org, acct = _coop(18)
    dan, _ = OW.join(w, "dan", 0)
    assert _ex(w, ann, "inst.adopt", org=org, charter={"procedures": {"maj": MAJ}, "default": "maj",
                                                       "membership": {"admission": "decision"}}).ok
    assert _ex(w, dan, "inst.join", org=org).ok
    r = _ex(w, dan, "inst.join", org=org)
    assert not r.ok and r.error["code"] == "conflict"


def test_f2_modules_pass_the_admission_harness():
    from crucible.worldgen.admission import admit
    base = ["identity", "bank", "admin", "contracts", "business", "gov"]
    for mods, cfg in ((["inst"], {}), (["commit"], {}), (["inst", "commit", "labor"], {"labor": {"workers": 4}})):
        rep = admit(mods, base=base, steps=800, seed=3, configs=cfg)
        assert rep.passed, (mods, rep.checks, rep.errors[:5])
