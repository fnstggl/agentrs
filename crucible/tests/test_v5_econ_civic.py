"""ECON wave 3b: the population takes part in marketplaces (exchange passes), the labour market (job search, hiring,
salaries, shifts on the employer's support queue) and institutions (mutual-aid associations), and sees display
campaigns with randomised eligibility. One small population startup world with exchanges, institutions and the
labour market, driven only by the population's own sessions and the incumbents' labelled rules."""
import collections

import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.commerce.worlds import startup_spec  # noqa: E402
from crucible.core.kernel import World  # noqa: E402

DAY = 86400


@pytest.fixture(scope="module")
def civic():
    spec = startup_spec(21, n=700, formats=["schedule.jobs"], extra=("xchg", "inst", "labor"),
                        rivals={"display_ads": True})
    w = World.create(spec)
    for d in range(1, 39):
        w.advance_time(d * DAY)
    return spec, w


def _decisions(w, prefix):
    return [e for e in w.log if e.type == "cols.batch" and str(e.payload.get("kind", "")).startswith(prefix)]


def test_population_buys_passes_on_exchanges_with_its_own_money(civic):
    _, w = civic
    st = w.state
    orders = [o for _, o in st.scan("xchg_orders")]
    assert orders, "individuals bought passes"
    for o in orders:
        assert o["buyer"][0] == "p" and o["buyer"][1:].isdigit()
        assert st.tables["accounts"][o["account"]]["owner"] == o["buyer"], "paid from its own account"
    venues = [e for _, e in st.scan("xchg_exchanges")]
    assert len(venues) == 2 and {v["policy"] for v in venues} == {"buyer_first", None}
    # passes expire: every exchange entitlement has a term
    ents = [e for _, e in st.scan("prod_entitlements") if e["source"] == "exchange"]
    assert ents and all(e["expires"] is not None for e in ents)
    # disputes are the buyers' own decisions after bad outcomes, decided by the venue's policy
    assert all(o["buyer"] in {f"p{k}" for k in range(700)} for o in orders if o["disputed"])
    dec = _decisions(w, "work.buy_pass")
    assert dec and all(e.payload["extra"]["decision"]["model_id"] == "work.value" for e in dec)
    # buyers rate sellers on the venue from their own experience (settled orders only, one rating per order)
    rated = list(st.scan("xchg_ratings"))
    assert rated, "pass buyers rated sellers on the venues"
    by_order = {oid: o for oid, o in st.scan("xchg_orders")}
    refunded_at = {}                       # when an order was refunded in full (a later dispute may do that)
    for e in w.log:
        if e.type == "xchg.order_status" and e.payload["refunded"] >= by_order[e.payload["id"]]["amount"]:
            refunded_at.setdefault(e.payload["id"], e.seq)
    rated_at = {e.payload["id"]: e.seq for e in w.log if e.type == "xchg.rating"}
    for oid, r in rated:
        o = by_order[oid]
        assert o["buyer"][1:].isdigit() and r["seller"] == o["seller"]
        assert rated_at[oid] < refunded_at.get(oid, float("inf")), "rated while the order stood"


def test_labour_market_hires_individuals_who_are_paid_and_work_shifts(civic):
    _, w = civic
    st = w.state
    hired = [e.payload for e in w.log if e.type == "labor.hired"]
    assert hired, "incumbents hired support staff from the population"
    for h in hired:
        assert h["worker"][0] == "p" and h["worker"][1:].isdigit() and h["org"].startswith("o:")
        c = st.tables["contracts"][h["contract"]]
        assert c["employee"] == h["worker"] and c["employer"] == h["org"]
    shifts = [e.payload for e in w.log if e.type == "labor.shift"]
    assert shifts and sum(len(s["tickets"]) for s in shifts) > 0, "employees resolved tickets in their shifts"
    # salaries are monthly contract payments from the employer to the individual's own account: every active
    # employment contract carries the salary clause and its scheduled payment; contracts that ran a month were paid
    paid = {e.payload["id"] for e in w.log if e.type == "contracts.paid"}
    for h in hired:
        c = st.tables["contracts"][h["contract"]]
        pay = [cl for cl in c["clauses"] if cl["type"] == "payment"]
        assert pay and pay[0]["from"] == h["org"] and pay[0]["to"] == h["worker"] and pay[0]["amount"] == h["salary"]
        if c["status"] == "active":
            assert st.has("sched", f"pay|{h['contract']}|0") or h["contract"] in paid
        if c["status"] == "active" and w.now - c["t"] > 31 * DAY:
            assert h["contract"] in paid
    # registrations keep no copy of the individual's skills
    pop_workers = [r for p, r in st.scan("labor_workers") if p[1:].isdigit()]
    assert pop_workers and all(r["skills"] == {} and not r["sim"] for r in pop_workers)
    kinds = collections.Counter(e.payload["kind"] for e in _decisions(w, "work.job_"))
    assert kinds["work.job_register"] and kinds["work.job_apply"] and kinds["work.job_accept"]
    # a labour-market job is work: the economy's employment view (what association members insure) includes it
    from crucible.economy import popbridge as pb
    from crucible.substrate import ids
    emp_idx = np.array(sorted(ids.index(p) for p, r in st.scan("labor_workers") if r.get("employer") and
                              p[1:].isdigit()), np.int64)
    assert emp_idx.size and pb.employment(w, emp_idx).all()
    own = pb.features(w, emp_idx)["employed"].astype(bool)
    assert not own.all(), "labour-market employees were out of work by the population's own record"


def test_mutual_aid_associations_run_on_member_decisions(civic):
    _, w = civic
    st = w.state
    from crucible.economy import assoc
    orgs = assoc.associations(st)
    assert orgs, "an individual founded an association"
    org = orgs[0][0]
    members = st.find_ids("members", "org", org)
    assert len(members) >= 10
    dues = [r for _, r in st.scan("inst_contributions") if r["kind"] == "dues"]
    assert dues and all(r["contributor"][1:].isdigit() for r in dues)
    for r in dues:
        assert st.tables["accounts"][r["account"]]["owner"] == r["contributor"]
    # claims are inst decisions under the charter (a members' majority); ballots come from members' own sessions
    claims = [d for _, d in st.scan("inst_decisions") if d["action"].get("tool") == "bank.transfer"]
    if claims:
        assert all(d["procedure"]["type"] == "majority" for d in claims)
    kinds = collections.Counter(e.payload["kind"] for e in _decisions(w, "work.assoc_"))
    assert kinds["work.assoc_found"] == len(orgs) and kinds["work.assoc_join"] and kinds["work.assoc_pay"]


def test_display_campaigns_randomise_eligibility_and_record_ad_response_decisions(civic):
    _, w = civic
    st = w.state
    draws = [r for _, r in st.scan("work_ad_draws")]
    assert draws, "incumbents ran display campaigns"
    arms = collections.Counter(r["arm"] for r in draws)
    assert arms["eligible"] and arms["holdout"]
    assert not any(r["displayed"] for r in draws if r["arm"] == "holdout"), "the holdout is never displayed"
    batches = _decisions(w, "work.ad_visit")
    assert batches and all(b.payload["table"] == "ads_seen" for b in batches)
    for b in batches:
        ex = b.payload["extra"]
        assert ex["experiment"]["id"] == "ad_holdout" and ex["experiment"]["arm"] in ("eligible", "holdout")
        assert ex["decision"]["model_id"] == "ad_response@2"
    # eligibility is the platform's random assignment: about the configured holdout share, independent of state
    share = arms["holdout"] / sum(arms.values())
    assert 0.1 < share < 0.32, share


def test_decisions_are_organic_population_decisions(civic):
    _, w = civic
    from crucible.calibration import inworld
    au = inworld.audit(w)
    mine = [b for b in au.batches if b["table"] in ("work", "jobs", "assoc", "ads_seen") and b.get("provenance")]
    assert mine and all(b["class"] in ("organic", "participant") for b in mine), \
        collections.Counter((b["class"], b["kind"]) for b in mine if b["class"] not in ("organic", "participant"))
    # every kind of decision these activities make is among them (none is recorded outside a population session)
    kinds = {b["kind"] for b in mine}
    for k in ("work.buy_pass", "work.job_register", "work.job_apply", "work.job_accept", "work.job_shift",
              "work.assoc_found", "work.assoc_join", "work.assoc_pay", "work.ad_visit"):
        assert k in kinds, (k, sorted(kinds))
    # decision kinds of these activities never appear without provenance (module bookkeeping has its own kinds)
    bare = [e.payload["kind"] for e in w.log if e.type == "cols.batch" and
            e.payload["table"] in ("jobs", "assoc", "ads_seen") and str(e.payload["kind"]).startswith("work.") and
            not isinstance((e.payload.get("extra") or {}).get("decision"), dict)]
    assert not bare, collections.Counter(bare)


def test_civic_world_replays_and_regenerates_exactly(civic):
    spec, w = civic
    evs = [e.to_dict() for e in w.log]
    assert World.replay(spec, evs).digest() == w.digest()
    assert World.regenerate(spec, evs).last_hash == w.last_hash
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []
    assert not [e for e in w.log if e.type in ("proc.failed", "rivals.failed")]


@pytest.fixture(scope="module")
def twelve_weeks():
    spec = startup_spec(21, n=700, formats=["schedule.jobs"], extra=("xchg", "inst", "labor"),
                        rivals={"display_ads": True})
    w = World.create(spec)
    for d in range(1, 85):
        w.advance_time(d * DAY)
    return w


def test_activity_continues_and_incumbent_days_never_fail(twelve_weeks):
    """Regression (wave 3b): a refused dismissal of an employee already on notice failed the incumbent's whole day
    (support, pricing, listings) from day 55 on; the activities must keep running, week after week."""
    w = twelve_weeks
    assert not [e.payload for e in w.log if e.type in ("rivals.failed", "proc.failed")]
    weeks = collections.defaultdict(collections.Counter)
    for e in w.log:
        wk = int(e.t // (7 * DAY))
        if e.type in ("labor.shift", "xchg.order", "inst.contribution", "inst.ballot"):
            weeks[wk][e.type] += 1
    # labour demand follows the incumbents' ticket backlog (it may fall to the founder alone in a given week), so
    # employment and pass sales are checked over the last four weeks, the members' decisions week by week
    last4 = [weeks[k] for k in range(8, 12)]
    assert sum(c["labor.shift"] for c in last4) > 0, last4        # employees still work shifts
    assert all(c["inst.ballot"] > 0 for c in last4), last4        # association members keep deciding claims
    assert sum(c["inst.contribution"] for c in last4) > 0, last4  # and paying dues
    assert sum(c["xchg.order"] for c in last4) > 0, last4         # passes still sell
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []
