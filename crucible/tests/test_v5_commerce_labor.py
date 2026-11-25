"""Crucible 5.0 labour market (module `labor`): recruitment, negotiation, employment contracts, private skills
and verified records, project dependencies, measured performance, retention and the independent agency of
simulated workers."""
import json

import crucible

crucible.load_all_modules()
from crucible.commerce import orgworld as OW  # noqa: E402
from crucible.commerce.labor import SKILLS  # noqa: E402
from crucible.goals.metrics import evaluate_metric  # noqa: E402

DAY = 86400


def _ex(w, a, t, **k):
    return OW.act(w, a, t, **k)


def _market(seed=5, workers=14, cash=20_000_000):
    w = OW.create(seed, workers=workers)
    boss, _ = OW.join(w, "boss", cash + 10_000_000)
    org, acct = OW.found_company(w, boss, "acme", cash)
    return w, boss, org, acct


def _hire_team(w, boss, org, skill="software", salary=650_000, days=3, close=True):
    """Post, let simulated workers apply, offer `salary` to every applicant, accept every counter-offer, and close
    the posting."""
    post = _ex(w, boss, "labor.post", org=org, title=f"{skill} role", skills={skill: 0.5}, salary_min=300_000,
               salary_max=900_000).result["post"]
    w.advance_time(w.now + days * DAY)
    apps = _ex(w, boss, "labor.applicants", post=post).result
    for a in apps:
        assert _ex(w, boss, "labor.offer", application=a["application"], salary_cents_month=salary).ok
    w.advance_time(w.now + 2 * DAY)
    for o in _ex(w, boss, "labor.offers", as_org=org).result:
        if o["from"] == "worker":
            assert _ex(w, boss, "labor.respond", offer=o["offer"], action="accept", as_org=org).ok
    if close:
        assert _ex(w, boss, "labor.close_post", post=post).ok
    assert _ex(w, boss, "labor.team", org=org).result, "nobody was hired"
    return post, apps


def test_recruitment_negotiation_and_employment_contracts():
    w, boss, org, acct = _market()
    post, apps = _hire_team(w, boss, org)
    assert len(apps) >= 3                                                    # simulated workers searched and applied
    offers = [w.state.tables["labor_offers"][k] for k in w.state.find_ids("labor_offers", "org", org)]
    first = [o for o in offers if o["round"] == 0]
    outcomes = {o["status"] for o in first}
    assert outcomes <= {"accepted", "countered", "rejected"} and len(outcomes) >= 2   # workers decide differently
    for o in first:
        wk = w.state.tables["labor_workers"][o["worker"]]
        if o["status"] == "accepted":
            assert o["salary"] >= 0.9 * wk["reservation"]
    for o in offers:
        if o["from"] == "worker":
            assert o["salary"] > 650_000                                      # a counter asks for more
    team = _ex(w, boss, "labor.team", org=org).result
    assert team
    for t in team:
        c = w.state.tables["contracts"][t["contract"]]
        assert c["status"] == "active" and "labor" in c["tags"] and c["employee"] == t["worker"]
        assert w.modules["identity"].role(w.state, org, t["worker"]) == "member"
    w.advance_time(w.now + 31 * DAY)
    still = [t["contract"] for t in team if w.state.tables["contracts"][t["contract"]]["status"] == "active"]
    assert still
    for cid in still:                                                         # salaries were paid by the contracts
        assert w.state.tables["contracts"][cid]["paid"].get("0", 0) >= 1
        assert any(tx["kind"] == "contract" and tx["ref"] == cid for _, tx in w.state.scan("txns"))
    assert evaluate_metric(w, "labor_employees", {"org": org}) == len(still)


def test_skill_levels_and_worker_state_stay_private():
    w, boss, org, acct = _market(6)
    post, apps = _hire_team(w, boss, org)
    hidden = ("reservation", "interests", "satisfaction", "patience", "wage_weight", "fatigue")
    blob = json.dumps([_ex(w, boss, "labor.applicants", post=post).result, _ex(w, boss, "labor.team", org=org).result,
                       [_ex(w, boss, "labor.profile", worker=a["profile"]["worker"]).result for a in apps]])
    for k in hidden:
        assert f'"{k}"' not in blob
    for a in apps:
        prof = a["profile"]
        assert set(prof["declared_skills"]) <= set(SKILLS) and isinstance(prof["declared_skills"], list)
        assert prof["verified_record"] == {}                                  # nothing done yet: nothing verified
    evs, _ = w.visible_events("p:boss", 0, limit=100_000)
    assert not [e for e in evs if e.type in ("labor.worker", "labor.worker_update")]


def test_dependencies_gate_work_and_quality_is_measured_from_skills():
    w, boss, org, acct = _market(7)
    _hire_team(w, boss, org)
    team = [t["worker"] for t in _ex(w, boss, "labor.team", org=org).result]
    assert len(team) >= 2
    prj = _ex(w, boss, "inst.project_create", title="product", as_org=org, deadline_days=60).result["project"]
    t1 = _ex(w, boss, "inst.task_add", project=prj, title="backend", effort_hours=50,
             skills={"software": 0.5}).result["task"]
    t2 = _ex(w, boss, "inst.task_add", project=prj, title="frontend", effort_hours=20, skills={"software": 0.4},
             depends_on=[t1]).result["task"]
    assert _ex(w, boss, "inst.task_assign", task=t1, assignee=team[0]).ok
    assert _ex(w, boss, "inst.task_assign", task=t2, assignee=team[1]).ok
    tasks = w.state.tables["inst_tasks"]
    while tasks[t1]["status"] != "done" and w.now < 50 * DAY:
        w.advance_time(w.now + DAY)
        if tasks[t1]["status"] != "done":
            assert tasks[t2]["progress"] == 0.0                              # blocked until its dependency is done
    assert tasks[t1]["status"] == "done"
    w.advance_time(w.now + 20 * DAY)
    assert tasks[t2]["status"] == "done"
    for tid, who in ((t1, team[0]), (t2, team[1])):
        wk = w.state.tables["labor_workers"][who]
        req = tasks[tid]["skills"]
        base = 0.65 + sum(wk["skills"][k] - v for k, v in req.items()) / len(req)
        assert abs(tasks[tid]["quality"] - min(1.0, max(0.0, base))) < 0.2  # skill match plus small task noise
    prof = _ex(w, boss, "labor.profile", worker=team[0]).result
    assert prof["verified_record"]["software"]["tasks"] >= 1
    assert evaluate_metric(w, "project_completion", {"org": org, "project": prj}) == 1.0
    assert evaluate_metric(w, "project_delivered", {"org": org, "project": prj}) == 1.0
    assert 0 < evaluate_metric(w, "project_quality", {"org": org, "project": prj}) <= 1
    assert evaluate_metric(w, "project_quality", {"org": "o:other", "project": prj}) == 0.0   # not its project


def test_workers_decline_assignments_they_cannot_fit():
    w, boss, org, acct = _market(8)
    _hire_team(w, boss, org)
    who = _ex(w, boss, "labor.team", org=org).result[0]["worker"]
    prj = _ex(w, boss, "inst.project_create", title="huge", as_org=org).result["project"]
    big = _ex(w, boss, "inst.task_add", project=prj, title="everything", effort_hours=80).result["task"]
    more = _ex(w, boss, "inst.task_add", project=prj, title="more", effort_hours=120).result["task"]
    assert _ex(w, boss, "inst.task_assign", task=big, assignee=who).ok
    assert w.state.tables["inst_tasks"][big]["assignee"] == who
    assert _ex(w, boss, "inst.task_assign", task=more, assignee=who).ok      # 200 hours > four weeks of its time
    t = w.state.tables["inst_tasks"][more]
    assert t["assignee"] is None and t["status"] == "open"
    assert any(e.type == "labor.declined" and e.payload["worker"] == who for e in w.log)


def test_unpaid_salary_is_a_breach_the_worker_leaves_and_does_not_return():
    w, boss, org, acct = _market(9, cash=3_000_000)
    _hire_team(w, boss, org)
    team = [t["worker"] for t in _ex(w, boss, "labor.team", org=org).result]
    assert team
    bal = w.state.tables["accounts"][acct]["balance"]
    assert _ex(w, boss, "bank.transfer", from_account=acct, to=boss, amount=bal).ok   # the owner empties it
    w.advance_time(w.now + 31 * DAY)
    for who in team:
        wk = w.state.tables["labor_workers"][who]
        assert wk["employer"] is None and org in wk["blacklist"] and wk["unpaid"] == 1
        assert w.modules["identity"].role(w.state, org, who) is None
    _ex(w, boss, "bank.transfer", from_account=w.modules["bank"].primary_account(w.state, boss), to=acct,
        amount=5_000_000)
    post = _ex(w, boss, "labor.post", org=org, title="again", skills={"software": 0.3},
               salary_max=2_000_000).result["post"]
    w.advance_time(w.now + 5 * DAY)
    applicants = {a["profile"]["worker"] for a in _ex(w, boss, "labor.applicants", post=post).result}
    assert not applicants & set(team)


def test_dissatisfied_workers_quit_for_better_pay_elsewhere():
    w, boss, org, acct = _market(10)
    _hire_team(w, boss, org, salary=480_000)
    _ex(w, boss, "labor.close_post", post=w.state.find_ids("labor_posts", "org", org)[0])
    rival, _ = OW.join(w, "rival", 30_000_000)
    r_org, _ = OW.found_company(w, rival, "rivalco", 20_000_000)
    _ex(w, rival, "labor.post", org=r_org, title="senior", skills={"software": 0.3}, salary_max=2_000_000)
    team0 = {t["worker"] for t in _ex(w, boss, "labor.team", org=org).result}
    w.advance_time(w.now + 60 * DAY)
    quits = [e.payload for e in w.log if e.type == "labor.quit"]
    assert quits and all(q["better_offer"] or q["satisfaction"] < 0.25 for q in quits)
    assert {q["worker"] for q in quits} <= team0
    assert evaluate_metric(w, "labor_retention", {"org": org, "window_s": 60 * DAY}) < 1.0


def test_pay_raise_replaces_the_contract_and_keeps_tenure():
    w, boss, org, acct = _market(11)
    _hire_team(w, boss, org)
    t = _ex(w, boss, "labor.team", org=org).result[0]
    who, old = t["worker"], t["contract"]
    joined = w.state.tables["labor_workers"][who]["joined"]
    w.advance_time(w.now + 5 * DAY)
    o = _ex(w, boss, "labor.adjust_pay", worker=who, org=org, salary_cents_month=t["salary"] + 100_000).result["offer"]
    w.advance_time(w.now + DAY + 3600)
    wk = w.state.tables["labor_workers"][who]
    assert w.state.tables["labor_offers"][o]["status"] == "accepted"
    assert wk["salary"] == t["salary"] + 100_000 and wk["contract"] != old and wk["joined"] == joined
    assert w.state.tables["contracts"][old]["status"] == "terminated"
    assert w.modules["identity"].role(w.state, org, who) == "member"


def test_team_lead_delegation_and_labour_contributions_weight_votes():
    w, boss, org, acct = _market(12)
    _hire_team(w, boss, org)
    team = [t["worker"] for t in _ex(w, boss, "labor.team", org=org).result]
    lead, _ = OW.join(w, "lead", 0)
    OW.add_member(w, org, boss, lead)
    prj = _ex(w, boss, "inst.project_create", title="p", as_org=org).result["project"]
    assert _ex(w, boss, "id.grant", grantee="lead", cap="inst.lead", owner=org, scope=prj).ok
    t = _ex(w, lead, "inst.task_add", project=prj, title="build", effort_hours=40,
            skills={"software": 0.4}).result["task"]
    assert _ex(w, lead, "inst.task_assign", task=t, assignee=team[0]).ok
    w.advance_time(w.now + 10 * DAY)
    assert w.state.tables["inst_tasks"][t]["status"] == "done"
    worked = w.modules["inst"].contribution_of(w.state, org, team[0])
    assert worked > 0 and w.state.meta[f"labor_hours:{org}"][team[0]] == worked     # gov profit sharing reads it
    assert _ex(w, boss, "inst.adopt", org=org, charter={
        "procedures": {"labor": {"type": "weighted", "weights": "contribution"}}, "default": "labor"}).ok
    r = _ex(w, boss, "inst.propose", org=org, action={"kind": "unit_create", "name": "core"})
    el = dict(w.state.tables["inst_decisions"][r.result["decision"]]["electorate"])
    assert el.get(team[0]) == float(worked) and boss not in el                 # weight = recorded labour


def test_labour_world_replays_and_regenerates():
    w, boss, org, acct = _market(13, workers=8)
    _hire_team(w, boss, org)
    team = [t["worker"] for t in _ex(w, boss, "labor.team", org=org).result]
    prj = _ex(w, boss, "inst.project_create", title="p", as_org=org).result["project"]
    t = _ex(w, boss, "inst.task_add", project=prj, title="t", effort_hours=30, skills={"software": 0.3}).result["task"]
    if team:
        _ex(w, boss, "inst.task_assign", task=t, assignee=team[0])
    w.advance_time(w.now + 12 * DAY)
    assert w.check_invariants() == []
    rep = OW.verify_reproducible(w)
    assert rep["replay_digest_equal"] and rep["regeneration_hash_equal"]
