"""Gate SP-6.2 (MASTER_SPEC Part 6.2): genuine recurring revenue.

Metrics (docs/v5/ACCEPTANCE_MANIFEST.json):

* revenue_features - Part 6.2 distinctions each demonstrated on a real world driven through ordinary tools
  (customers pay from their own accounts, money moves only through the bank) and measured by the evaluator's
  accounting (`commerce.revenue`): one-time sales vs MRR, MRR/ARR normalisation of annual plans, active vs
  cancelled subscriptions, refunds net of what was kept, discounts, customer acquisition cost from real ad spend,
  churn, operating costs and profit, related-party exclusion, sustained subscription validity (past-due and
  scheduled-to-cancel subscriptions do not count), marketplace passes as one-time sales.
* rolling_sales_counted_as_recurring - violations, over controlled worlds and a population startup world with
  participant-run exchanges, of "rolling sales are never recurring revenue": MRR above what the live, paid-up,
  recurring subscriptions charge per month (it would then contain one-time sales, passes or 30-day sales), MRR
  that does not fall to zero when only rolling sales remain, or pass revenue counted in MRR.
* recurring_revenue_systems_of_record - modules of the shipped registry that charge recurring subscriptions to the
  ledger themselves (a `bank.move`/`bank.debit` of ledger kind "subscription", found in their syntax trees): the
  subscription module `subs` plus every other module that keeps its own recurring billing beside it.
* product_catalogues - tables that hold products: the owner and every reviewed duplicate of the shared kind
  `product` in the mandatory fact scan of every shipped module (substrate.facts.audit), plus, in the integrated
  population world, any non-empty table of sellable offers that does not reference a `products` product (a 4.0
  market listing outside the catalogue).
"""
from __future__ import annotations

import ast
import inspect
import traceback
from fractions import Fraction

import pytest

from _commerce_kit import DAY, balance, healthy, join, make_world, media_business, org_account
from crucible.acceptance.v5.evidence import record
from crucible.commerce import integrity, revenue

PRICE = 2_000


def _run(scenarios) -> tuple[int, dict]:
    out = {}
    for fn in scenarios:
        try:
            fn()
            out[fn.__name__] = "passed"
        except AssertionError as e:
            out[fn.__name__] = f"failed: {e}"[:300]
        except Exception as e:  # a crash is a failed feature, with its traceback in the notes
            out[fn.__name__] = f"error: {type(e).__name__}: {e} {traceback.format_exc()[-300:]}"[:500]
    return sum(1 for v in out.values() if v == "passed"), out


# ---------------------------------------------------------------------------------------------- features
def f_one_time_sales_are_not_recurring():
    w = make_world(201)
    f, org, prod, plans, _ = media_business(w, "one", monthly=PRICE)
    off = f.do("prod.offer", product=prod, kind="one_time", price_cents=10_000)["offer"]
    for i in range(3):
        join(w, f"b{i}", 20_000).do("prod.buy", offer=off)
    rep = revenue.revenue_report(w, org)
    assert rep["one_time_sales_cents"] == 30_000 and rep["mrr_cents"] == 0.0, rep
    join(w, "s0", 20_000).do("subs.subscribe", plan=plans["month"])
    assert revenue.mrr_exact(w, org) == PRICE


def f_mrr_arr_normalisation():
    w = make_world(202)
    f, org, prod, plans, _ = media_business(w, "norm", monthly=PRICE, annual=12 * 1_000 * 10)
    join(w, "monthly", 20_000).do("subs.subscribe", plan=plans["month"])
    join(w, "yearly", 200_000).do("subs.subscribe", plan=plans["year"])
    mrr = revenue.mrr_exact(w, org)
    assert mrr == PRICE + Fraction(120_000, 12) and revenue.m_arr(w, org) == float(12 * mrr)
    rep = revenue.revenue_report(w, org)
    assert rep["subscription_revenue_cents"] == PRICE + 120_000 != rep["mrr_cents"], "30-day sales are not MRR"


def f_active_vs_cancelled():
    w = make_world(203)
    f, org, prod, plans, _ = media_business(w, "act", monthly=PRICE)
    cs = [join(w, f"c{i}", 20_000) for i in range(3)]
    sids = [c.do("subs.subscribe", plan=plans["month"])["subscription"] for c in cs]
    cs[0].do("subs.cancel", subscription=sids[0], at_period_end=False)
    rep = revenue.revenue_report(w, org)
    assert rep["active_subscriptions"] == 2 and rep["cancelled_subscriptions"]["voluntary"] == 1, rep
    assert revenue.mrr_exact(w, org) == 2 * PRICE


def f_refunds_net_mrr():
    w = make_world(204)
    f, org, prod, plans, _ = media_business(w, "ref", monthly=PRICE)
    c = join(w, "cust", 20_000)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    inv = [i for i in c.do("subs.invoices", subscription=sid) if i["status"] == "paid"][0]
    f.do("subs.refund", invoice=inv["id"], amount_cents=500, reason="goodwill")
    assert revenue.mrr_exact(w, org) == PRICE - 500
    rep = revenue.revenue_report(w, org)
    assert rep["subscription_refunds_cents"] == 500 and rep["refunds_cents"] >= 500, rep
    f.do("subs.refund", invoice=inv["id"], amount_cents=PRICE - 500, reason="full")
    assert revenue.mrr_exact(w, org) == 0


def f_discounts():
    w = make_world(205)
    f, org, prod, plans, _ = media_business(w, "disc", monthly=4_000)
    cp = f.do("subs.create_coupon", plan=plans["month"], percent_off=50, periods=1, max_redemptions=5)["coupon"]
    join(w, "cust", 50_000).do("subs.subscribe", plan=plans["month"], coupon=cp)
    assert revenue.mrr_exact(w, org) == 2_000, "MRR is net of the recurring discount"
    rep = revenue.revenue_report(w, org)
    assert rep["discounts_cents"] == 2_000, rep
    w.advance_time(30 * DAY + 3600)
    assert revenue.mrr_exact(w, org) == 4_000, "the discount ends after its period"


def f_customer_acquisition_cost():
    w = make_world(206, extra=["ads"])
    f, org, prod, plans, _ = media_business(w, "cac", monthly=PRICE)
    f.do("ads.create_campaign", account=org_account(w, org), keywords=["premium", "handbook"], bid_cents=50,
         daily_budget_cents=10_000, title="Premium handbook", text="the best handbook")
    t0 = w.now
    for i in range(4):
        c = join(w, f"a{i}", 20_000)
        c.do("search.query", q="premium handbook")
        c.do("subs.subscribe", plan=plans["month"])
    rep = revenue.revenue_report(w, org, window_s=w.now - t0 + 3600)
    assert rep["acquisition_spend_cents"] == 200 and rep["new_customers"] == 4, rep
    assert rep["cac_cents"] == 50.0, rep


def f_churn():
    w = make_world(207)
    f, org, prod, plans, _ = media_business(w, "churn", monthly=PRICE)
    cs = [join(w, f"c{i}", 50_000) for i in range(4)]
    sids = [c.do("subs.subscribe", plan=plans["month"])["subscription"] for c in cs]
    w.advance_time(31 * DAY)
    cs[0].do("subs.cancel", subscription=sids[0], at_period_end=False)
    w.advance_time(40 * DAY)
    assert revenue.m_churn_rate(w, org, 30 * DAY) == pytest.approx(0.25)


def f_operating_costs_and_profit():
    w = make_world(208)
    f, org, prod, plans, _ = media_business(w, "cost", capital=500_000, monthly=PRICE)
    for i in range(3):
        join(w, f"c{i}", 20_000).do("subs.subscribe", plan=plans["month"])
    k = join(w, "contractor", 0)
    cid = f.do("contracts.propose", parties=[org, k.id], title="Writing", as_org=org,
               clauses=[{"type": "payment", "from": org, "to": k.id, "amount": 2_500, "every_s": None,
                         "count": None, "start_in_s": 0}])["contract"]
    k.do("contracts.sign", contract=cid)
    w.advance_time(w.now + DAY)
    p = revenue.profit(w, org, 30 * DAY)
    assert p["revenue"] == 3 * PRICE and p["operating_costs"] == 2_500 and p["profit"] == 3 * PRICE - 2_500, p


def f_related_party_excluded():
    w = make_world(209)
    f, org, prod, plans, _ = media_business(w, "rel", monthly=PRICE, controller="agent-rel")
    join(w, "indep", 20_000, controller="cust").do("subs.subscribe", plan=plans["month"])
    integrity.SCHEMES["sister_org"](w, "relsis", f.id, org, plans["month"], PRICE)
    integrity.SCHEMES["self_subscription"](w, "relself", f.id, org, plans["month"], PRICE)
    rep = revenue.revenue_report(w, org)
    assert revenue.mrr_exact(w, org) == PRICE and rep["independent_customers"] == 1, rep
    assert rep["related_party_cents"] >= 2 * PRICE, rep


def f_subscription_validity():
    w = make_world(210)
    f, org, prod, plans, _ = media_business(w, "valid", monthly=PRICE)
    poor = join(w, "poor", PRICE + 100)
    poor.do("subs.subscribe", plan=plans["month"])
    annual_f, aorg, aprod, aplans, _ = media_business(w, "valid2", monthly=None, annual=24_000)
    leaving = join(w, "leaving", 30_000)
    s2 = leaving.do("subs.subscribe", plan=aplans["year"])["subscription"]
    leaving.do("subs.cancel", subscription=s2, at_period_end=True)
    assert revenue.mrr_exact(w, aorg) == 0, "a subscription set to cancel is scheduled churn, not recurring"
    w.advance_time(30 * DAY + 3600)
    s = [x for _, x in w.state.scan("sub_subs") if x["customer"] == poor.id][0]
    assert s["status"] == "past_due" and revenue.mrr_exact(w, org) == 0, s["status"]


def f_marketplace_passes_are_one_time():
    w = make_world(211)
    f, org, prod, plans, doc = media_business(w, "pass", monthly=PRICE)
    op = join(w, "op", 200_000)
    ex = op.do("xchg.create", name="Mart", take_rate_bps=500)["exchange"]
    lst = f.do("xchg.list", exchange=ex, product=prod, price_cents=3_000, term_days=30)["listing"]
    buyers = [join(w, f"pb{i}", 20_000) for i in range(3)]
    for b in buyers:
        b.do("xchg.buy", listing=lst)
    assert buyers[0].attempt("docs.get", doc=doc).ok, "a pass gives access"
    assert revenue.mrr_exact(w, org) == 0, "passes are one-time sales"
    w.advance_time(31 * DAY)
    assert not buyers[0].attempt("docs.get", doc=doc).ok, "a pass expires"


FEATURES = [f_one_time_sales_are_not_recurring, f_mrr_arr_normalisation, f_active_vs_cancelled, f_refunds_net_mrr,
            f_discounts, f_customer_acquisition_cost, f_churn, f_operating_costs_and_profit, f_related_party_excluded,
            f_subscription_validity, f_marketplace_passes_are_one_time]


# ---------------------------------------------------------------------------------------------- rolling sales
def _subscription_bound(w, org) -> Fraction:
    """What the live, paid-up, recurring subscriptions of `org` charge per month (no weights): MRR can never exceed
    it; if it did, MRR would contain something other than recurring subscription charges."""
    sm = w.modules["subs"]
    tot = Fraction(0)
    for sid in w.state.find_ids("sub_subs", "merchant", org):
        s = w.state.tables["sub_subs"][sid]
        if sm.recurring_valid(s, w.now):
            tot += sm.monthly_recurring(s)
    return tot


def _ledger_bound(w, org) -> Fraction:
    """An independent upper bound on `org`'s MRR from the money it actually received (wave-3b repair: the check
    against `_subscription_bound` alone used the same subscription rows and normalisation as MRR, so it could only
    fail on a payer weight above 1). For every subscription whose latest paid invoice (`subs.invoice` event) covers
    now, the invoice's charge must be in the bank ledger - a `subscription` transaction crediting the merchant with
    exactly the invoiced cash - and the subscription then contributes that period's cash plus applied credit (a
    plan change in the period: the largest such amount of its invoices), normalised to a month. MRR that includes
    rolling sales, passes, unbacked or unpaid charges exceeds this bound."""
    from crucible.commerce.subscriptions import MONTHS
    st = w.state
    mine = {a for a, r in st.scan("accounts") if r["owner"] == org}
    refunds: dict[str, list] = {}
    for tid in st.find_ids("txns", "kind", "refund"):
        tx = st.tables["txns"][tid]
        if tx.get("ref"):
            refunds.setdefault(tx["ref"], []).append((tid, tx))
    per: dict[str, Fraction] = {}
    for e in w.log:
        if e.type != "subs.invoice" or e.payload["merchant"] != org or e.payload.get("status") != "paid":
            continue
        p = e.payload
        if int(p.get("period_end") or 0) <= w.now and p.get("kind") != "proration":
            continue
        cash = int(p["amount"])
        if cash > 0:
            tx = st.get("txns", p["txn"]) if p.get("txn") else None
            got = sum(d for a, d in (tx or {}).get("postings", []) if a in mine and d > 0)
            if tx is None or tx["kind"] != "subscription" or got != cash:
                continue                                   # not backed by the ledger: never counts
        # money returned on the invoice (bank `refund` transactions referencing it) is not revenue
        back = sum(-d for _, tx in refunds.get(p["id"], []) for a, d in tx["postings"] if a in mine and d < 0)
        amt = Fraction(max(0, cash - back) + int(p.get("credit_applied") or 0), MONTHS[p["interval"]])
        if p.get("kind") == "proration":
            amt = Fraction(int(p["list_price"]) - int(p.get("discount") or 0), MONTHS[p["interval"]])
            if p["subscription"] not in per:
                continue
        per[p["subscription"]] = max(per.get(p["subscription"], Fraction(0)), amt)
    return sum(per.values(), Fraction(0))


def rolling_violations() -> tuple[int, dict]:
    v, detail = 0, {}
    # (1) only rolling sales remain: renewals stopped, one-time sales and passes continue
    w = make_world(220)
    f, org, prod, plans, _ = media_business(w, "roll", monthly=PRICE)
    off = f.do("prod.offer", product=prod, kind="one_time", price_cents=5_000)["offer"]
    op = join(w, "op", 100_000)
    ex = op.do("xchg.create", name="Mart", take_rate_bps=300)["exchange"]
    lst = f.do("xchg.list", exchange=ex, product=prod, price_cents=3_000, term_days=30)["listing"]
    cs = [join(w, f"s{i}", 100_000) for i in range(3)]
    sids = [c.do("subs.subscribe", plan=plans["month"])["subscription"] for c in cs]
    w.advance_time(10 * DAY)
    for c, sid in zip(cs, sids):
        c.do("subs.cancel", subscription=sid, at_period_end=False)
    for i in range(4):
        b = join(w, f"r{i}", 50_000)
        b.do("prod.buy", offer=off)
        b.do("xchg.buy", listing=lst)
    rep = revenue.revenue_report(w, org)
    roll30 = rep["subscription_revenue_cents"] + rep["one_time_sales_cents"] + rep["other_revenue_cents"]
    detail["only_rolling"] = {"rolling_30d_cents": roll30, "mrr_cents": rep["mrr_cents"]}
    v += int(roll30 > 0 and rep["mrr_cents"] != 0.0)
    v += int(revenue.mrr_exact(w, org) > _subscription_bound(w, org))
    v += int(revenue.mrr_exact(w, org) > _ledger_bound(w, org))
    # (2) a population startup world with exchanges: incumbents sell subscriptions AND passes to individuals
    from crucible.commerce.worlds import startup_spec
    from crucible.core.kernel import World
    spec = startup_spec(23, n=400, formats=["schedule.jobs"], extra=("xchg",))
    pw = World.create(spec)
    for d in range(1, 19):
        pw.advance_time(d * DAY)
    st = pw.state
    merchants = sorted({s["merchant"] for _, s in st.scan("sub_subs")})
    pass_cents = sum(o["amount"] for _, o in st.scan("xchg_orders"))
    over = ledger_over = 0
    for m in merchants:
        over += int(revenue.mrr_exact(pw, m) > _subscription_bound(pw, m))
        ledger_over += int(revenue.mrr_exact(pw, m) > _ledger_bound(pw, m))
    v += over + ledger_over
    detail["ledger_bound_cents"] = {m: float(_ledger_bound(pw, m)) for m in merchants}
    detail["population_world"] = {"merchants": len(merchants), "pass_order_cents": int(pass_cents),
                                  "mrr_above_subscription_bound": over, "mrr_above_ledger_bound": ledger_over,
                                  "mrr_cents": {m: float(revenue.mrr_exact(pw, m)) for m in merchants},
                                  "invariant_errors": len(healthy(pw))}
    return v, {"detail": detail, "pass_cents": int(pass_cents), "merchants": len(merchants),
               "population_world": pw}


# ---------------------------------------------------------------------------------------------- systems / catalogues
def _subscription_memo(node) -> bool:
    """A payment memo that names a subscription: a string constant, or an f-string whose text starts with it."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower().startswith("subscription")
    if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
        return str(node.values[0].value).lower().startswith("subscription")
    return False


def _billing_calls(node) -> list[int]:
    """Line numbers of recurring-charge calls under `node`: a bank.move / bank.debit of ledger kind "subscription",
    or a payment (move, debit, or the population's own `pay`) whose memo names a subscription."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("move", "debit", "pay"):
            vals = [a.value for a in n.args if isinstance(a, ast.Constant)] + \
                [k.value.value for k in n.keywords if k.arg == "kind" and isinstance(k.value, ast.Constant)]
            memo = any(_subscription_memo(a) for a in n.args) or \
                any(_subscription_memo(k.value) for k in n.keywords if k.arg == "memo")
            if (n.func.attr != "pay" and "subscription" in vals) or memo:
                out.append(n.lineno)
    return out


def recurring_billing_modules() -> dict[str, list[str]]:
    """{module: [file:line]} of shipped kernel modules whose code posts a recurring charge itself (syntax-tree scan,
    `_billing_calls`). A module's code is every crucible class of its MRO (inherited billing counts: e.g. `caps`
    inherits `services.capabilities.Capabilities._bill`), each scanned in its own defining file, plus the top-level
    functions of those files (helpers the class calls). Wave-3b repair: the scan read only the registered class's
    own body, so a module billing through a base class or a module-level helper was not counted."""
    import crucible
    from crucible.core.kernel import MODULES, Module
    from crucible.substrate import facts
    crucible.load_all_modules()
    trees: dict[str, ast.Module] = {}

    def tree(path):
        if path not in trees:
            trees[path] = ast.parse(open(path, encoding="utf-8").read())
        return trees[path]
    out: dict[str, list[str]] = {}
    for name in sorted(facts.shipped_modules()):
        hits: set[str] = set()
        for cls in MODULES[name].__mro__:
            if cls in (Module, object) or not cls.__module__.startswith("crucible"):
                continue
            path = inspect.getsourcefile(cls)
            rel = path.split("/src/")[-1]
            t = tree(path)
            body = next((n for n in ast.walk(t) if isinstance(n, ast.ClassDef) and n.name == cls.__name__), None)
            if body is not None:
                hits.update(f"{rel}:{ln}" for ln in _billing_calls(body))
            for fn in t.body:
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    hits.update(f"{rel}:{ln}" for ln in _billing_calls(fn))
        if hits:
            out[name] = sorted(hits)
    return out


def product_catalogue_tables(pop_world) -> dict:
    from crucible.substrate import facts
    audit = facts.audit()
    owners = [k for k in audit["owners"] if k == "products.products"]
    dups = [k for k in audit["duplicates"] if facts.REVIEWED.get(k, {}).get("kind") == "product"]
    st = pop_world.state
    loose = 0
    if "listings" in st.tables:
        loose = sum(1 for _, x in st.scan("listings") if not x.get("product"))
    return {"owner_tables": owners, "duplicate_tables": dups, "unlinked_market_listings": int(loose),
            "count": len(owners) + len(dups) + (1 if loose else 0)}


def test_SP_6_2_genuine_recurring_revenue():
    n, res = _run(FEATURES)
    viol, roll = rolling_violations()
    billing = recurring_billing_modules()
    cats = product_catalogue_tables(roll.pop("population_world"))
    metrics = {"revenue_features": n, "revenue_features_defined": len(FEATURES),
               "rolling_sales_counted_as_recurring": int(viol),
               "recurring_revenue_systems_of_record": len(billing),
               "product_catalogues": int(cats["count"]),
               "population_world_pass_order_cents": roll["pass_cents"],
               "population_world_merchants": roll["merchants"]}
    record("SP-6.2", metrics, command="pytest tests/v5/test_gate_SP_6_2.py",
           notes=f"features {res}; rolling-sales checks {roll['detail']}; modules posting recurring 'subscription' "
                 f"charges themselves: {billing}; product catalogues {cats}.")
    assert n >= 6 and all(v == "passed" for v in res.values()), res
    assert viol == 0, roll["detail"]
    assert roll["pass_cents"] > 0, "the population bought passes on the exchanges (otherwise the check is vacuous)"
    assert cats["count"] == 1, cats
    assert len(billing) == 1 and "subs" in billing, billing
