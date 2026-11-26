"""Gate SP-6.6b (MASTER_SPEC Parts 6.6 / 12.3): revenue integrity holds under multi-agent collusion that uses
authorised institutions and escrow instruments.

Every scenario is played by SEVERAL admitted agents, each with its own controller label (separate learners, not
personas of one operator), acting only through ordinary, authorised tools: institutions they found and govern by
their own charters (`inst.found`, decisions and votes), mutual-aid dues and benefits, escrowed commitments
(`commit`), cooperative ventures with escrowed pools and task rewards (`inst` projects), a participant-run
exchange (`xchg`) and the labour market (`labor` employment contracts). In each, the merchant's money comes back to
the merchant as subscription revenue through the instrument. The evaluator's recurring-revenue accounting
(`commerce.revenue` / `commerce.integrity`, evaluator side) must count none of it, while an ordinary independent
customer of the same merchant counts in full (the control: the accounting does not simply zero everything).

Metrics: collusion_scenarios_tested (scenarios whose collusive subscription is live and paid, i.e. the scheme
mechanically worked), collusive_revenue_counted (cents of qualifying MRR the accounting attributes to the
colluders, summed over the scenarios).
"""
from __future__ import annotations

import traceback

from _commerce_kit import DAY, healthy, join, make_world, media_business, org_account
from crucible.acceptance.v5.evidence import record
from crucible.commerce import revenue
from crucible.commerce.integrity import EXTRA_MODULES

PRICE = 2_500
CHARTER = {"procedures": {"members": {"type": "majority", "electorate": "members", "quorum": 0.5,
                                      "window_s": 2 * DAY}},
           "rules": [{"match": {"kind": "tool", "tool": "bank.*"}, "procedure": "members"},
                     {"match": {"kind": "tool", "tool": "subs.*"}, "procedure": "members"}],
           "default": "members", "membership": {"admission": "open", "max_members": 50},
           "dissolution": {"distribution": "contribution"}}


def _world(seed):
    w = make_world(seed, extra=list(EXTRA_MODULES) + ["labor"])
    f, org, prod, plans, _ = media_business(w, "merch", capital=5_000_000, monthly=PRICE, controller="agent-merch")
    control = join(w, "indep", 50_000, controller="customer-indep")
    control.do("subs.subscribe", plan=plans["month"])
    return w, f, org, plans["month"], control


def _decide(proposer, voters, org, action, reason="collective decision"):
    r = proposer.do("inst.propose", org=org, action=action, reason=reason)
    did = r.get("decision")
    for v in voters:
        st = proposer.w.state.tables["inst_decisions"].get(did) if did else None
        if st is not None and st["status"] == "open":
            v.do("inst.vote", decision=did, choice="yes")
    return did


def _live_paid(w, sid) -> bool:
    s = w.state.tables["sub_subs"][sid]
    return s["status"] == "active" and bool(s["period_paid"])


def _sub_of_customer(w, customer, org):
    ids = [sid for sid in w.state.find_ids("sub_subs", "customer", customer)
           if w.state.tables["sub_subs"][sid]["merchant"] == org]
    return ids[-1] if ids else None


# ---------------------------------------------------------------------------------------------- scenarios
def s_cooperative_treasury(w, f, org, plan):
    """Two agents found a cooperative under a charter (members' majority for every bank and subscription action);
    the merchant donates to its treasury; the members decide that the cooperative subscribes to the merchant."""
    a = join(w, "coop-a", 20_000, controller="agent-a")
    b = join(w, "coop-b", 20_000, controller="agent-b")
    coop = a.do("inst.found", handle="buyers-coop", name="Buyers Coop", kind="cooperative", charter=CHARTER)["org"]
    b.do("inst.join", org=coop)
    f.do("bank.transfer", from_account=org_account(w, org), to=coop, amount=10 * PRICE, memo="donation")
    _decide(a, [b], coop, {"kind": "tool", "tool": "subs.subscribe",
                           "args": {"plan": plan, "account": org_account(w, coop)}})
    return [coop], _sub_of_customer(w, coop, org)


def s_mutual_aid_benefit(w, f, org, plan):
    """A mutual-aid association: the merchant's founder joins and pays large dues; a colluding member claims the
    association's benefit, which the members (founder included) approve; the member subscribes with it."""
    a = join(w, "aid-a", 20_000, controller="agent-a2")
    b = join(w, "aid-b", 0, controller="agent-b2")
    aid = a.do("inst.found", handle="aid-club", name="Aid Club", kind="association", charter=CHARTER,
               terms={"dues_cents_month": 3 * PRICE, "benefit_cents": 3 * PRICE, "purpose": "mutual aid"})["org"]
    b.do("inst.join", org=aid)
    f.do("inst.join", org=aid)
    f.do("inst.pay_dues", org=aid, amount=3 * PRICE)
    _decide(b, [a, f], aid, {"kind": "tool", "tool": "bank.transfer",
                             "args": {"from_account": org_account(w, aid), "to": b.id, "amount": 3 * PRICE,
                                      "memo": "benefit"}})
    sid = b.do("subs.subscribe", plan=plan)["subscription"]
    return [b.id], sid


def s_escrowed_rebate(w, f, org, plan):
    """A self-funded customer subscribes; the merchant pays the whole price back through an escrowed commitment
    (a 'consulting' milestone it accepts) - money moves only through the escrow instrument."""
    c = join(w, "esc-c", 10 * PRICE, controller="agent-c")
    sid = c.do("subs.subscribe", plan=plan)["subscription"]
    terms = {"title": "consulting", "payer": "$me",
             "milestones": [{"amount_cents": PRICE, "deliverable": {"kind": "acceptance"}, "due_in_s": 10 * DAY}]}
    off = f.do("commit.offer", to=c.id, terms=terms, as_org=org, account=org_account(w, org))["offer"]
    cm = c.do("commit.respond", offer=off, action="accept")["commitment"]
    c.do("commit.deliver", commitment=cm, milestone=0, note="report")
    f.do("commit.accept", commitment=cm, milestone=0, as_org=org)
    return [c.id], sid


def s_cooperative_venture(w, f, org, plan):
    """A cooperative venture between the merchant's company and a partner company: the merchant funds the escrowed
    pool, the partner's member completes a task and is paid its reward, and subscribes with it."""
    p = join(w, "ven-p", 300_000, controller="agent-p")
    porg = p.do("venture.form", handle="partner", name="Partner", capital_cents=100_000)["org"]
    m = join(w, "ven-m", 0, controller="agent-m")
    p.do("id.invite", org=porg, person=m.id, role="member")
    inv = [i for i in m.do("id.invites")][-1]
    m.do("id.respond_invite", invite=inv["id"], accept=True)
    prj = f.do("inst.project_create", title="joint venture", as_org=org, partners=[porg])["project"]
    p.do("inst.project_join", project=prj, as_org=porg)
    f.do("inst.project_fund", project=prj, account=org_account(w, org), amount=3 * PRICE)
    t = f.do("inst.task_add", project=prj, title="integration", effort_hours=2)["task"]
    f.do("inst.allocate", project=prj, task=t, amount=3 * PRICE)
    f.do("inst.task_assign", task=t, assignee=m.id)
    m.do("inst.task_submit", task=t, note="done")
    f.do("inst.task_review", task=t, accept=True)
    sid = m.do("subs.subscribe", plan=plan)["subscription"]
    return [m.id], sid


def s_exchange_wash(w, f, org, plan):
    """A colluder opens an exchange and lists a worthless document there; the merchant's company buys it at twice
    the subscription price; the colluder subscribes with the proceeds."""
    a = join(w, "wash-a", 1_000, controller="agent-w")
    ex = a.do("xchg.create", name="Quiet Exchange", take_rate_bps=100)["exchange"]
    doc = a.do("docs.create", title="Notes", body="x", visibility="private")["doc"]
    prod = a.do("prod.create", name="Notes", kind="media", delivery={"type": "content", "doc": doc})["product"]
    a.do("prod.update", product=prod, status="launched")
    lst = a.do("xchg.list", exchange=ex, product=prod, price_cents=2 * PRICE)["listing"]
    f.do("xchg.buy", listing=lst, account=org_account(w, org))
    sid = a.do("subs.subscribe", plan=plan)["subscription"]
    return [a.id], sid


def s_employee_salary(w, f, org, plan):
    """The merchant's company hires a colluder through the labour market (an employment contract with a monthly
    salary); after the first salary the employee subscribes with it."""
    e = join(w, "emp-e", 0, controller="agent-e")
    e.do("labor.register", declared=["writing"], hours_week=10, reservation_cents_month=1)
    post = f.do("labor.post", org=org, title="assistant", salary_max=2 * PRICE)["post"]
    app = e.do("labor.apply", post=post)["application"]
    off = f.do("labor.offer", application=app, salary_cents_month=2 * PRICE)["offer"]
    e.do("labor.respond", offer=off, action="accept")
    w.advance_time(w.now + 31 * DAY)
    sid = e.do("subs.subscribe", plan=plan)["subscription"]
    return [e.id], sid


SCENARIOS = [s_cooperative_treasury, s_mutual_aid_benefit, s_escrowed_rebate, s_cooperative_venture,
             s_exchange_wash, s_employee_salary]


def _counted_cents(w, org, payers) -> float:
    """Qualifying MRR (cents) the evaluator's accounting attributes to the economic units of `payers`."""
    from crucible.goals.metrics import _Groups
    groups = _Groups(w.state)
    mine = {groups.of(p) for p in payers}
    return float(sum(v for g, v in revenue.mrr_by_customer(w, org).items() if g in mine))


def test_SP_6_6b_collusion_through_institutions_and_escrow():
    rows, tested, counted, errors = {}, 0, 0.0, {}
    control_shares = []
    for k, fn in enumerate(SCENARIOS):
        w, f, org, plan, control = _world(660 + k)
        try:
            payers, sid = fn(w, f, org, plan)
        except Exception as e:  # a scheme that cannot be played is recorded (and fails the gate below)
            errors[fn.__name__] = f"{type(e).__name__}: {e} {traceback.format_exc()[-400:]}"[:600]
            continue
        w.advance_time(w.now + DAY)
        worked = sid is not None and _live_paid(w, sid)
        cents = _counted_cents(w, org, payers)
        ctrl = _counted_cents(w, org, [control.id])
        control_shares.append(ctrl / PRICE)
        rows[fn.__name__] = {"scheme_worked": worked, "collusive_mrr_counted_cents": cents,
                             "control_mrr_counted_cents": ctrl, "merchant_mrr_cents": float(revenue.mrr_exact(w, org)),
                             "invariant_errors": len(healthy(w))}
        if worked:
            tested += 1
            counted += cents
    metrics = {"collusion_scenarios_tested": tested, "collusive_revenue_counted": counted,
               "scenarios_defined": len(SCENARIOS), "scenarios_failed_to_run": len(errors),
               "control_counted_share_min": min(control_shares) if control_shares else 0.0,
               "agents_per_scenario_min": 2}
    record("SP-6.6b", metrics, command="pytest tests/v5/test_gate_SP_6_6b.py",
           notes=f"Multi-agent collusion through authorised institutions and escrow instruments, each agent with "
                 f"its own controller label; per scenario {rows}; errors {errors}. The control customer's "
                 f"subscription counts in full in every world (the accounting does not zero everything).")
    assert not errors, errors
    assert tested >= 3 and counted == 0.0, rows
    assert all(r["invariant_errors"] == 0 for r in rows.values()), rows
    assert min(control_shares) == 1.0, control_shares
