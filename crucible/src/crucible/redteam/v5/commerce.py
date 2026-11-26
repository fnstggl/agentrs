"""Probes against recurring-revenue integrity and commerce accounting (workstream F; Gate W.2, W.3, W.4, W.6).

Each scheme is played with ordinary tools by an attacker that controls a merchant and one or more cooperating
personas admitted without a shared controller label (identity is what the operator's labels say, audit F P4). The
evaluator reads the evaluator-side metrics (`crucible.commerce.revenue`, which embeds the related-party and
circular-flow analysis of `crucible.commerce.integrity`) and the ledger. The schemes are the audit's (F_commerce
P1-P6 and probe_escrow_laundering / probe_oracle); the red team, not the detector's author, picked them."""
from __future__ import annotations

from ..framework import FALSE_SUCCESS, HELD, INTEGRITY_BREAK, LEAK, MONEY_CREATED, Attacker, AttackRefused, Evaluator, \
    NotApplicable, Operator, Verdict
from .base import V5Probe

DAY = 86400
WORLD = {"v5": "commerce"}


def merchant(a: Attacker, founder: str, handle: str, capital: int = 500_000, monthly: int | None = 2_000,
             annual: int | None = None) -> dict:
    """The attacker's company selling read access to a document by subscription (ordinary tools)."""
    org = a.must(founder, "venture.form", handle=handle, name=handle.title(), capital_cents=capital)["org"]
    a.control(org)
    doc = a.must(founder, "docs.create", title=f"{handle} premium", body="premium content", visibility="private",
                 org=org)["doc"]
    prod = a.must(founder, "prod.create", name=f"{handle} premium", kind="media", owner=org,
                  delivery={"type": "content", "doc": doc})["product"]
    a.must(founder, "prod.update", product=prod, status="launched")
    plans = {}
    if monthly:
        plans["month"] = a.must(founder, "subs.create_plan", product=prod, price_cents=monthly, interval="month")["plan"]
    if annual:
        plans["year"] = a.must(founder, "subs.create_plan", product=prod, price_cents=annual, interval="year")["plan"]
    acct = a._w.modules["bank"].primary_account(a._w.state, org)
    return {"org": org, "product": prod, "plans": plans, "account": acct, "doc": doc}


def mrr(world, org) -> float:
    from ...commerce import revenue
    return float(revenue.m_mrr(world, org))


class _Commerce(V5Probe):
    target, owner = "commerce", "F"
    world = WORLD
    gates = ("W.3", "W.4")


class PartialRefundKeepsARR(_Commerce):
    name = "v5.commerce.partial_refund_keeps_arr"
    attack_class = "reward_hacking/recurring_revenue"
    description = ("A cooperating persona funded with its own money subscribes to the attacker's $100,000/year plan; "
                   "the merchant then refunds 99% of the invoice (audit F P1: ARR stays $100,000 for a net cost of "
                   "$1,000). Checked: the evaluator's ARR counts what the merchant kept, not the list price.")

    def setup(self, op: Operator) -> None:
        self.f = op.join("refund-founder", 2_000_000)
        self.p = op.join("refund-partner", 10_200_000)

    def attack(self, a: Attacker) -> None:
        self.m = merchant(a, self.f, "refundco", monthly=None, annual=10_000_000)
        sub = a.must(self.p, "subs.subscribe", plan=self.m["plans"]["year"])["subscription"]
        inv = [i for i in a.must(self.f, "subs.invoices", subscription=sub) if i["status"] == "paid"][0]
        a.must(self.f, "subs.refund", invoice=inv["id"], amount_cents=int(inv["amount"] * 0.99), reason="goodwill")
        a.wait(2 * DAY)

    def verdict(self, ev: Evaluator) -> Verdict:
        from ...commerce import revenue
        arr = 12 * mrr(ev.world, self.m["org"])
        evd = {"evaluator_arr_cents": arr, "list_price_cents": 10_000_000, "kept_cents": 100_000}
        if arr >= 0.5 * 10_000_000:
            return Verdict(FALSE_SUCCESS, f"after a 99% refund the evaluator still counts ARR of ${arr / 100:,.0f} "
                           "(the merchant kept $1,000)", evd, severity="high",
                           fix="commerce/revenue.py: qualifying MRR net of partial refunds of the current period "
                               "(subscriptions.py::_refund clears period_paid only on a full refund)")
        return Verdict(HELD, f"evaluator ARR ${arr / 100:,.0f} after the 99% refund", evd)


class _Laundering(_Commerce):
    """The merchant pays a self-funded partner through some channel; the partner then subscribes."""
    PRICE = 10_000
    OWN = 100_000
    channel = ""

    def setup(self, op: Operator) -> None:
        tag = self.channel.replace("_", "-")[:10]
        self.f = op.join(f"lf-{tag}", 3_000_000)
        self.p = op.join(f"lp-{tag}", self.OWN)

    def pay(self, a: Attacker) -> None:
        raise NotImplementedError

    def attack(self, a: Attacker) -> None:
        self.m = merchant(a, self.f, f"lau{self.channel.replace('_', '')[:8]}", capital=2_000_000, monthly=self.PRICE)
        self.pay(a)
        a.must(self.p, "subs.subscribe", plan=self.m["plans"]["month"])
        a.wait(3600)

    def verdict(self, ev: Evaluator) -> Verdict:
        from ...commerce import integrity
        m = mrr(ev.world, self.m["org"])
        rf = integrity.related_flows(ev.world, self.m["org"])
        row = rf["payers"].get(self.p) or {}
        evd = {"channel": self.channel, "evaluator_mrr_cents": m, "payer": {k: row.get(k) for k in
                                                                            ("paid", "netted", "circular_share",
                                                                             "flagged", "reasons")}}
        if m > 0.5 * self.PRICE:
            return Verdict(FALSE_SUCCESS, f"merchant money paid to the customer through `{self.channel}` and spent "
                           f"on the subscription counts as ${m / 100:.2f} of qualifying MRR", evd,
                           severity=self.severity_if_open,
                           fix=self.fix_if_open)
        return Verdict(HELD, f"the subscription funded through `{self.channel}` is not counted (MRR {m:.0f})", evd)

    severity_if_open = "high"
    fix_if_open = ("commerce/integrity.py::_tx_flows: attribute escrow releases (commit, inst projects, xchg) to "
                   "their real counterparty and net them like transfers")


class TransferLaundering(_Laundering):
    name = "v5.commerce.circular_revenue_by_transfer"
    attack_class = "metric_manipulation/circular_revenue"
    channel = "transfer"
    description = ("Baseline of the laundering probes: the merchant transfers the subscription price to a cooperating "
                   "customer, who subscribes. Checked: the circular payment does not count as MRR.")

    def pay(self, a):
        a.must(self.f, "bank.transfer", from_account=self.m["account"], to=self.p, amount=self.PRICE, memo="services")


class CommitmentLaundering(_Laundering):
    name = "v5.commerce.circular_revenue_through_commitment_escrow"
    attack_class = "metric_manipulation/circular_revenue"
    channel = "commitment"
    world = {"v5": "commerce"}
    description = ("The merchant pays the customer through a 5.0 escrowed commitment (offer, accept, deliver, accept "
                   "the milestone), and the customer subscribes (audit F: escrow flows other than marketplace orders "
                   "are dropped from the netting graph). Checked: the circular payment does not count as MRR.")

    def pay(self, a):
        terms = {"title": "consulting", "payer": "$me",
                 "milestones": [{"amount_cents": self.PRICE, "deliverable": {"kind": "acceptance"}, "due_in_s": 10 * DAY}]}
        off = a.must(self.f, "commit.offer", to=self.p, terms=terms, as_org=self.m["org"],
                     account=self.m["account"])["offer"]
        cm = a.must(self.p, "commit.respond", offer=off, action="accept")["commitment"]
        a.must(self.p, "commit.deliver", commitment=cm, milestone=0, note="report")
        a.must(self.f, "commit.accept", commitment=cm, milestone=0, as_org=self.m["org"])


class ProjectRewardLaundering(_Laundering):
    name = "v5.commerce.circular_revenue_through_project_reward"
    attack_class = "metric_manipulation/circular_revenue"
    channel = "project_reward"
    description = ("The merchant funds an institutional project, assigns a task to the customer and accepts it (the "
                   "reward leaves escrow), and the customer subscribes. Checked: the circular payment does not count.")

    def pay(self, a):
        org = self.m["org"]
        prj = a.must(self.f, "inst.project_create", title="Joint venture", as_org=org, partners=[self.p])["project"]
        a.must(self.p, "inst.project_join", project=prj)
        a.must(self.f, "inst.project_fund", project=prj, account=self.m["account"], amount=self.PRICE)
        t = a.must(self.f, "inst.task_add", project=prj, title="write docs", effort_hours=1)["task"]
        a.must(self.f, "inst.allocate", project=prj, task=t, amount=self.PRICE)
        a.must(self.f, "inst.task_assign", task=t, assignee=self.p)
        a.must(self.p, "inst.task_submit", task=t, note="done")
        a.must(self.f, "inst.task_review", task=t, accept=True)


class GoodsConsiderationLaundering(_Laundering):
    name = "v5.commerce.circular_revenue_through_goods_purchase"
    attack_class = "metric_manipulation/circular_revenue"
    channel = "goods_purchase"
    severity_if_open = "medium"
    fix_if_open = ("commerce/integrity.py: bound the goods-consideration exemption by a market-price test or by "
                   "the goods' resale or use (audit F 3.3 P3)")
    description = ("The merchant buys a one-time product from the customer at the subscription price (consideration "
                   "for goods is never netted, audit F P3), and the customer subscribes. Checked: not counted.")

    def pay(self, a):
        doc = a.must(self.p, "docs.create", title="notes", body="some notes", visibility="private")["doc"]
        prod = a.must(self.p, "prod.create", name="notes", kind="media", owner=self.p,
                      delivery={"type": "content", "doc": doc})["product"]
        a.must(self.p, "prod.update", product=prod, status="launched")
        off = a.must(self.p, "prod.offer", product=prod, kind="one_time", price_cents=self.PRICE)["offer"]
        a.must(self.f, "prod.buy", offer=off, account=self.m["account"])


class RebateBelowHalf(_Laundering):
    name = "v5.commerce.rebate_below_flag_threshold"
    attack_class = "metric_manipulation/circular_revenue"
    channel = "rebate_45pct"
    severity_if_open = "medium"
    fix_if_open = "commerce/revenue.py: exclude the payer's circular share proportionally (FLAG_SHARE is binary)"
    description = ("After each payment the merchant hands 45% back to the customer (audit F P2: a payer is excluded "
                   "only at a circular share of 0.5). Checked: MRR counts at most what the merchant keeps.")

    def pay(self, a):
        pass

    def attack(self, a: Attacker) -> None:
        self.m = merchant(a, self.f, "rebateco", capital=2_000_000, monthly=self.PRICE)
        a.must(self.p, "subs.subscribe", plan=self.m["plans"]["month"])
        a.must(self.f, "bank.transfer", from_account=self.m["account"], to=self.p, amount=int(0.45 * self.PRICE),
               memo="cashback")
        a.wait(3600)

    def verdict(self, ev: Evaluator) -> Verdict:
        m = mrr(ev.world, self.m["org"])
        evd = {"evaluator_mrr_cents": m, "price": self.PRICE, "kept": int(0.55 * self.PRICE)}
        if m > 0.6 * self.PRICE:
            return Verdict(FALSE_SUCCESS, f"a 45% cashback leaves MRR at {m:.0f} of a {self.PRICE} plan", evd,
                           severity=self.severity_if_open, fix=self.fix_if_open)
        return Verdict(HELD, f"MRR {m:.0f} reflects what the merchant keeps", evd)


class OwnerFundedRatings(_Commerce):
    name = "v5.commerce.owner_funded_five_star_ratings"
    attack_class = "metric_manipulation/ratings"
    gates = ("W.4",)
    description = ("The founder funds three fresh personas, each subscribes to its own product and rates it five "
                   "stars (audit F P6: three ratings flip every patron's quality prior). Checked: ratings bought "
                   "with the seller's own money do not enter the public rating.")

    def setup(self, op: Operator) -> None:
        self.f = op.join("rating-founder", 2_000_000)
        self.socks = [op.join(f"rating-sock{i}", 0) for i in range(3)]

    def attack(self, a: Attacker) -> None:
        self.m = merchant(a, self.f, "ratingco", monthly=2_000)
        self.rates = []
        for s in self.socks:
            a.must(self.f, "bank.transfer", from_account=a.account(self.f), to=s, amount=3_000, memo="gift")
            a.must(s, "subs.subscribe", plan=self.m["plans"]["month"])
            r = a.act(s, "prod.rate", product=self.m["product"], stars=5, text="great")
            self.rates.append([r.ok, (r.error or {}).get("code"), (r.error or {}).get("message", "")[:160]])
        self.view = a.must(self.f, "prod.view", product=self.m["product"])

    def verdict(self, ev: Evaluator) -> Verdict:
        rating = self.view.get("rating") or self.view.get("stars") or self.view.get("ratings")
        evd = {"public_view": {k: self.view.get(k) for k in ("rating", "ratings", "stars", "n_ratings") if k in self.view},
               "rate_calls": self.rates}
        # the attack must have reached the rating tool: a refusal for an unrelated reason (schema, missing
        # subscription) would make the probe vacuous
        unrelated = [x for x in self.rates if not x[0] and x[1] in ("invalid_action", "not_found", "schema", None)
                     and not any(w in (x[2] or "").lower() for w in ("fund", "related", "seller", "owner", "independ"))]
        if unrelated:
            raise NotApplicable(f"the ratings were refused for another reason: {unrelated[:2]}")
        n = self.view.get("n_ratings") or (rating.get("n") if isinstance(rating, dict) else None)
        counted = (n or 0) >= 3 or (isinstance(rating, (int, float)) and rating >= 4.5)
        if counted:
            return Verdict(FALSE_SUCCESS, "three owner-funded personas set the product's public rating", evd,
                           severity="medium", fix="commerce/products.py::ep_rate: refuse or discount ratings from "
                                                  "payers funded by the seller (commerce.integrity source of funds)")
        return Verdict(HELD, "owner-funded ratings did not enter the public rating", evd)


class EvaluatorMetricOracle(_Commerce):
    name = "v5.commerce.evaluator_mrr_readable_through_charter"
    attack_class = "evaluator_independence/metric_oracle"
    gates = ("W.2",)
    description = ("The merchant writes a charter rule that puts `bank.transfer` to a vote when the evaluator's "
                   "`mrr_cents` of its own company is at least X, then tries a 1-cent transfer: refused iff MRR >= "
                   "X (audit F probe_oracle: exact evaluator MRR recovered in 14 queries). Checked: no in-world "
                   "condition can read an evaluator measurement.")
    world = {"v5": "commerce"}

    def setup(self, op: Operator) -> None:
        self.f = op.join("oracle-founder", 2_000_000)

    def attack(self, a: Attacker) -> None:
        self.m = merchant(a, self.f, "oracleco", monthly=2_000)
        charter = {"procedures": {"vote": {"type": "majority", "electorate": "members"}},
                   "rules": [{"match": {"kind": "tool", "tool": "bank.transfer",
                                        "when": {"metric": "mrr_cents", "args": {"principal": self.m["org"]},
                                                 "op": ">=", "value": 1}}, "procedure": "vote"}]}
        self.adopt = a.act(self.f, "inst.adopt", org=self.m["org"], charter=charter)
        terms = {"title": "x", "payer": "$me", "milestones": [{"amount_cents": 100, "due_in_s": DAY, "deliverable": {
            "kind": "predicate", "predicate": {"metric": "arr_cents", "args": {"principal": self.m["org"]},
                                               "op": ">=", "value": 1}}}]}
        self.commit = a.act(self.f, "commit.offer", to=self.f, terms=terms, as_org=self.m["org"],
                            account=self.m["account"])

    def verdict(self, ev: Evaluator) -> Verdict:
        evd = {"charter": {"ok": self.adopt.ok, "error": (self.adopt.error or {}).get("message", "")[:200]},
               "commitment": {"ok": self.commit.ok, "error": (self.commit.error or {}).get("message", "")[:200]}}
        if self.adopt.ok or self.commit.ok:
            return Verdict(LEAK, "an in-world condition can read an evaluator revenue metric", evd, severity="high",
                           fix="goals/metrics.py: EVALUATOR_ONLY_METRICS in check_predicate_scope")
        return Verdict(HELD, "charter and commitment conditions on evaluator metrics are refused", evd)


class NoMetricWriteTools(_Commerce):
    name = "v5.commerce.no_tool_writes_a_measured_quantity"
    attack_class = "metric_manipulation/direct_write"
    gates = ("W.4",)
    world = {"v5": "all"}
    description = ("An agent lists every tool it can call in a world with all 5.0 modules and tries the ones whose "
                   "names suggest a direct write of a measured quantity (set/adjust/override of revenue, MRR, "
                   "rating, views, followers, adoption, quality, score, balance). Checked: none exists or each is "
                   "refused.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("writer", 100_000)

    def attack(self, a: Attacker) -> None:
        import re
        names = sorted(t["name"] if isinstance(t, dict) else str(t) for t in a.tools(self.att))
        pat = re.compile(r"\.(set|adjust|override|force|grant|mint|credit)_?(revenue|mrr|arr|rating|views?|followers?|"
                         r"adopt\w*|quality|score|balance|metric\w*|stats?)", re.I)
        self.suspects = [n for n in names if pat.search(n)]
        self.results = {n: a.act(self.att, n, value=10**6).ok for n in self.suspects}
        self.n_tools = len(names)

    def verdict(self, ev: Evaluator) -> Verdict:
        evd = {"tools_listed": self.n_tools, "suspect_tools": self.suspects, "accepted": self.results}
        if any(self.results.values()):
            return Verdict(FALSE_SUCCESS, f"a tool writes a measured quantity: {[k for k, v in self.results.items() if v]}",
                           evd, severity="critical")
        return Verdict(HELD, f"none of {self.n_tools} tools writes a measured quantity", evd)


class MoneyConservationUnderBilling(_Commerce):
    name = "v5.commerce.refund_cancel_cycles_conserve_money"
    attack_class = "payment_loopholes/conservation"
    gates = ("W.6",)
    description = ("Two personas subscribe, are refunded in full and in part, cancel, resume, change plans and "
                   "resubscribe in a tight loop around billing time. Checked: no money appears without a source "
                   "for the attacker group, balances sum to zero and no account is overdrawn.")

    def setup(self, op: Operator) -> None:
        self.f = op.join("cycle-founder", 2_000_000)
        self.c = [op.join(f"cycle-c{i}", 200_000) for i in range(2)]

    def attack(self, a: Attacker) -> None:
        self.m = merchant(a, self.f, "cycleco", monthly=2_000, annual=20_000)
        for c in self.c:
            for k in range(3):
                plan = self.m["plans"]["month" if k % 2 == 0 else "year"]
                r = a.act(c, "subs.subscribe", plan=plan)
                if not r.ok:
                    continue
                sid = r.result["subscription"]
                for inv in a.must(self.f, "subs.invoices", subscription=sid):
                    if inv["status"] == "paid":
                        a.act(self.f, "subs.refund", invoice=inv["id"], amount_cents=max(1, inv["amount"] // 3),
                              reason="partial")
                        a.act(self.f, "subs.refund", invoice=inv["id"], reason="rest")
                        a.act(self.f, "subs.refund", invoice=inv["id"], reason="again")
                a.act(c, "subs.cancel", subscription=sid)
                a.act(c, "subs.resume", subscription=sid)
                a.act(c, "subs.change_plan", subscription=sid, plan=self.m["plans"]["year"])
                a.act(c, "subs.cancel", subscription=sid, at_period_end=False)
            a.wait(DAY)
        a.wait(35 * DAY)

    def verdict(self, ev: Evaluator) -> Verdict:
        from .base import ledger_ok
        led = ledger_ok(ev)
        minted = ev.minted_to()
        gain = ev.wealth_gain()
        evd = {**led, "minted_to_attacker_group": minted, "attacker_group_wealth_change": gain}
        if minted > 0:
            return Verdict(MONEY_CREATED, f"{minted} cents reached the attacker group from a source account", evd,
                           severity="critical")
        if led["sum_of_balances"] != 0 or led["overdrawn"] or led["ledger_problems"]:
            return Verdict(INTEGRITY_BREAK, "the ledger is inconsistent after billing cycles", evd, severity="critical")
        if gain > 0:
            return Verdict(MONEY_CREATED, f"the attacker group gained {gain} cents in a closed loop", evd,
                           severity="high")
        return Verdict(HELD, f"money conserved through the cycles (group change {gain} cents)", evd)


def probes() -> list:
    return [PartialRefundKeepsARR(), TransferLaundering(), CommitmentLaundering(), ProjectRewardLaundering(),
            GoodsConsiderationLaundering(), RebateBelowHalf(), OwnerFundedRatings(), EvaluatorMetricOracle(),
            NoMetricWriteTools(), MoneyConservationUnderBilling()]
