"""ECON (F+G) unit and integration tests: population work activities as the demand side, platform trust rules,
seats, the charter money rule, and exact replay of a population startup world."""
import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.commerce.reference import ReferenceSaaS  # noqa: E402
from crucible.commerce.worlds import startup_spec  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.substrate import columns  # noqa: E402

DAY = 86400


@pytest.fixture(scope="module")
def startup():
    spec = startup_spec(21, n=260, formats=["schedule.jobs"])
    w = World.create(spec)
    ref = ReferenceSaaS(ads_budget_day=500)
    ref.setup(w, ["schedule.jobs"])
    for d in range(1, 9):
        w.advance_time(d * DAY)
        ref.day(w)
    return spec, w, ref


def test_demand_comes_from_population_individuals_paying_their_own_money(startup):
    spec, w, ref = startup
    st = w.state
    subs = [s for _, s in st.scan("sub_subs")]
    assert subs, "individuals of the population subscribed"
    for s in subs:
        assert s["customer"][0] == "p" and s["customer"][1:].isdigit(), s["customer"]
        assert st.tables["accounts"][s["account"]]["owner"] == s["customer"], "paid from its own account"
    assert not any(m["name"] in ("patrons", "consumers", "shoppers") for m in spec.modules)
    # decisions are recorded with their decision model (provenance)
    dec = [e for e in w.log if e.type == "cols.batch" and (e.payload.get("extra") or {}).get("decision")]
    assert dec and all(e.payload["extra"]["decision"]["model_id"] == "work.value" for e in dec)


def test_work_uses_real_executions_and_hidden_truth_only_as_world_physics(startup):
    _, w, _ = startup
    import inspect
    from crucible.economy import work
    src = inspect.getsource(work.Work)
    # the verifier is called in exactly one place: the employer's acceptance (world physics)
    assert src.count(".score(") == 1 and "def _accepted" in src
    assert int(columns.col(w, "work", "delegated").sum()) > 0
    uses = [e for e in w.log if e.type == "prod.used"]
    assert uses and all(e.payload["holder"][1:].isdigit() for e in uses)


def test_population_world_replays_and_regenerates_exactly(startup):
    spec, w, _ = startup
    evs = [e.to_dict() for e in w.log]
    assert World.replay(spec, evs).digest() == w.digest()
    assert World.regenerate(spec, evs).last_hash == w.last_hash
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []


def test_ratings_shown_publicly_are_verified_purchases_only():
    from _commerce_kit_path import kit
    K = kit()
    w = K.make_world(901)
    f, org, prod, plans, _ = K.media_business(w, "rateco", monthly=1_000)
    honest = [K.join(w, f"h{i}", 10_000, controller=f"h{i}") for i in range(3)]
    for c in honest:
        c.do("subs.subscribe", plan=plans["month"])
        c.do("prod.rate", product=prod, stars=2)
    socks = [K.join(w, f"sock{i}", 0, controller="sockfarm") for i in range(3)]
    for s in socks:
        f.do("bank.transfer", from_account=K.org_account(w, org), to=s.id, amount=2_000, memo="thanks")
        s.do("subs.subscribe", plan=plans["month"])
        s.do("prod.rate", product=prod, stars=5)
    view = honest[0].do("prod.view", product=prod)
    assert view["ratings"] == 3 and view["rating"] == 2.0, view


def test_seats_limit_organisation_access():
    from _commerce_kit_path import kit
    K = kit()
    w = K.make_world(902)
    f = K.join(w, "dev-f", 400_000)
    org = f.do("venture.form", handle="devco", name="DevCo", capital_cents=100_000)["org"]
    repo = f.do("code.create_repo", name="lib", owner=org)["repo"]
    f.do("code.commit", repo=repo, files={"lib.py": "def f():\n    return 1\n"}, message="lib")
    p = f.do("prod.create", name="Lib", kind="devtool", owner=org, delivery={"type": "repo", "repo": repo})["product"]
    f.do("prod.update", product=p, status="launched")
    off = f.do("prod.offer", product=p, kind="license", price_cents=5_000, seats=1)["offer"]
    boss = K.join(w, "boss", 300_000)
    borg = boss.do("venture.form", handle="buyer", name="Buyer", capital_cents=50_000)
    admins = [K.join(w, f"adm{i}", 1_000) for i in range(3)]
    for a in admins:
        inv = boss.do("id.invite", org=borg["org"], person=a.id, role="admin")["invite"]
        a.do("id.respond_invite", invite=inv, accept=True)
    ent = boss.do("prod.buy", offer=off, account=borg["account"])["entitlement"]
    assert not admins[0].attempt("code.read", repo=repo, path="lib.py").ok, "no seat, no access"
    boss.do("prod.assign_seat", entitlement=ent, person=admins[0].id)
    assert admins[0].attempt("code.read", repo=repo, path="lib.py").ok
    assert not boss.attempt("prod.assign_seat", entitlement=ent, person=admins[1].id).ok, "1 seat only"
    assert not admins[1].attempt("code.read", repo=repo, path="lib.py").ok
    boss.do("prod.release_seat", entitlement=ent, person=admins[0].id)
    boss.do("prod.assign_seat", entitlement=ent, person=admins[1].id)
    assert admins[1].attempt("code.read", repo=repo, path="lib.py").ok
    assert not admins[0].attempt("code.read", repo=repo, path="lib.py").ok
    assert w.check_invariants() == []


def test_exchange_statistics_do_not_count_operator_funded_shells():
    from _commerce_kit_path import kit
    K = kit()
    w = K.make_world(903)
    op = K.join(w, "opr", 2_000_000)
    ex = op.do("xchg.create", name="Mart", take_rate_bps=3000)["exchange"]
    seller = K.join(w, "sel", 100_000, controller="sel")
    K.mint(w, seller.id, "gear", 50)
    p = seller.do("prod.create", name="Gear", kind="industrial", delivery={"type": "goods", "sku": "gear"})["product"]
    seller.do("prod.update", product=p, status="launched")
    lst = seller.do("xchg.list", exchange=ex, product=p, price_cents=1)["listing"]
    for i in range(10):
        s = K.join(w, f"shell{i}", 0)
        op.do("bank.transfer", from_account=op.acct, to=s.id, amount=10)
        s.do("xchg.buy", listing=lst)
    real = K.join(w, "realbuyer", 1_000, controller="rb")
    real.do("xchg.buy", listing=lst)
    stats = w.modules["xchg"].stats(w.state, ex, w.now)
    assert stats["buyers_30d"] == 1, stats
