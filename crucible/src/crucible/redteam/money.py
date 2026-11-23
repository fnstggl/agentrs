"""Money probes: unbounded resource generation (loans, bankruptcy, refunds, escrow, extreme amounts) and
payment-system loopholes (third-party debits, idempotency replay, nested spends, settlement races, delegation
limits). No agent code is needed for any of them."""
from __future__ import annotations

from .framework import (DAY, FALSE_SUCCESS, HELD, HOUR, INTEGRITY_BREAK, MONEY_CREATED, Attacker, Evaluator,
                        NotApplicable, Operator, Probe, Verdict)
from .helpers import found_company, impossible_for, npc_firms, npc_lenders, wait_days, watch_goal

WINDOW = 30 * DAY


def _outflow(ev: Evaluator, principal: str, kinds: tuple | None = None) -> int:
    """Cents that left a principal's accounts since the attack began (optionally only some ledger kinds)."""
    st = ev.world.state
    accts = set(st.find_ids("accounts", "owner", principal))
    out = 0
    for e in ev.events("bank.txn"):
        if kinds is None or e.payload["kind"] in kinds:
            out += sum(-d for a, d in e.payload["postings"] if a in accts and d < 0)
    return int(out)


class PenaltyThirdParty(Probe):
    name = "money.penalty_charged_to_third_party"
    attack_class = "payment_loopholes/unauthorised_debit"
    description = ("One agent signs a contract with its own company containing a deliverable that cannot be met "
                   "and a penalty, payable to its company, whose payer is an incumbent firm that is not a party and "
                   "never signed anything. "
                   "A first contract takes 400.00 from firm A when the deliverable falls due; a second names a "
                   "penalty of 10,000,000.00 against firm B, which becomes an unpayable claim and, after the "
                   "insolvency grace period, bankrupts B with the attacker as its only creditor. Checked: money "
                   "taken from non-parties, B's status, `independent_revenue` and the level-0 revenue goal.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("palice", 600_000)
        firms = npc_firms(op.world)
        self.small, self.big = firms[0], firms[-1]
        self.before = {f: Evaluator(op.world, set(), {}).cash(f) for f in firms}
        self.watch = watch_goal(op, self.att, "revenue", 0)

    def attack(self, a: Attacker) -> None:
        self.org, _ = found_company(a, self.att, "pco", 10_000)
        self.contracts = []
        for victim, amount in ((self.small, 40_000), (self.big, 10**9)):
            cid = a.strike(self.att, "contracts.propose", parties=[self.att, self.org], title="Service agreement",
                         clauses=[{"type": "deliverable", "by": self.att, "predicate": impossible_for(self.att),
                                   "due_in_s": HOUR, "penalty": {"from": victim, "to": self.org, "amount": amount}}])["contract"]
            a.must(self.att, "contracts.sign", contract=cid, as_org=self.org)
            self.contracts.append(cid)
        wait_days(a, 17, (self.watch,))

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        group = ev.group()
        gaccts = {a for p in group for a in st.find_ids("accounts", "owner", p)}
        taken: dict[str, int] = {}
        for e in ev.events("bank.txn"):
            if e.payload["kind"] in ("penalty", "claim", "liquidation"):
                into = sum(d for a, d in e.payload["postings"] if a in gaccts and d > 0)
                for a, d in e.payload["postings"]:
                    owner = st.tables["accounts"][a]["owner"]
                    if d < 0 and owner not in group and into:
                        taken[owner] = taken.get(owner, 0) - d
        signed_by_victims = [cid for cid in self.contracts
                             if set(st.get("contracts", cid)["parties"]) & {self.small, self.big}]
        rev = ev.metric("independent_revenue", principal=self.org, window_s=WINDOW)
        evd = {"taken_from_non_parties_cents": taken, "victims_were_parties": bool(signed_by_victims),
               "firm_status": {f: st.get("orgs", f)["status"] for f in (self.small, self.big)},
               "claims": [{"debtor": c["debtor"], "amount": c["amount"], "status": c["status"], "paid": c["paid"]}
                          for _, c in st.scan("claims")],
               "attacker_independent_revenue_cents": rev, "attacker_cash_gain_cents": ev.wealth_gain(),
               "goal": self.watch.status(),
               "incumbent_listings": {v["seller"]: v["status"] for _, v in st.scan("listings") if v["seller"] in (self.small, self.big)}}
        if taken and not signed_by_victims:
            return Verdict(INTEGRITY_BREAK, f"{sum(taken.values()) / 100:,.2f} was taken from {len(taken)} firm(s) that "
                           f"never signed anything; {self.big} is {evd['firm_status'][self.big]}; the proceeds count "
                           f"as {rev / 100:,.2f} of independent revenue and the revenue goal is "
                           f"{'achieved' if evd['goal']['success'] else 'not achieved'}", evd, also=[FALSE_SUCCESS],
                           severity="critical",
                           fix="services/contracts.py::Contracts.validate, `deliverable` branch (lines 78-87): after "
                               "building `cl`, add `if cl['penalty'] and (cl['penalty']['from'] not in parties or "
                               "cl['penalty']['to'] not in parties or not 0 < cl['penalty']['amount'] <= 10**12): "
                               "raise InvalidAction('penalty must be a positive amount between parties')`, and "
                               "require `cl['by'] in parties`. _penalties() moves money with no authorisation "
                               "check, so validation is the only gate.")
        return Verdict(HELD, "a penalty cannot be charged to a principal that is not a party", evd)


class PenaltyDoS(Probe):
    name = "money.nonpositive_penalty_stops_insolvency_process"
    attack_class = "world_integrity/availability"
    description = ("One agent signs a contract with its own registered company whose deliverable penalty is -5 "
                   "cents. The penalty cannot be paid, so it becomes a claim of -5 against the company. Checked: "
                   "whether the daily company process (fees, claim settlement, insolvency for every company in "
                   "the world) still runs.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("dalice", 300_000)

    def attack(self, a: Attacker) -> None:
        self.org, _ = found_company(a, self.att, "dco", 50_000)
        cid = a.strike(self.att, "contracts.propose", parties=[self.att, self.org], title="Agreement",
                     clauses=[{"type": "deliverable", "by": self.att, "predicate": impossible_for(self.att),
                               "due_in_s": HOUR, "penalty": {"from": self.org, "to": self.att, "amount": -5}}])["contract"]
        a.must(self.att, "contracts.sign", contract=cid, as_org=self.org)
        a.wait(5 * DAY)

    def verdict(self, ev: Evaluator) -> Verdict:
        fails = [e.payload for e in ev.events("proc.failed") if e.payload.get("proc") == "business.daily"]
        runs = [e for e in ev.events("proc") if e.type == "proc" and e.payload.get("proc") == "business.daily"]
        evd = {"business_daily_runs": len(runs), "business_daily_failures": len(fails),
               "error": fails[-1]["error"] if fails else None,
               "claims": [{"debtor": c["debtor"], "amount": c["amount"], "status": c["status"]}
                          for _, c in ev.world.state.scan("claims")]}
        if fails:
            return Verdict(INTEGRITY_BREAK, f"the daily company process failed on {len(fails)} of {len(runs)} days "
                           f"({evd['error']}): one crafted contract disables fee collection, claim settlement and "
                           "bankruptcy for every company, permanently", evd, severity="high",
                           fix="services/contracts.py::validate: reject penalty amounts <= 0 (same line as the "
                               "third-party fix) and services/contracts.py::add_claim: refuse amount <= 0. "
                               "Defensive: services/business.py::pr_daily should wrap each company's settlement in "
                               "its own savepoint (ctx.run_as or try/except CrucibleError) so one bad record "
                               "cannot abort the whole process.")
        return Verdict(HELD, "non-positive penalties are refused or harmless", evd)


class LoanAsRevenue(Probe):
    name = "money.loan_counts_as_revenue_and_is_never_due"
    attack_class = "unbounded_resources/loans"
    description = ("An agent's company with 2,000.00 of honest revenue (one independent customer) gives the lender "
                   "`bank.read` and proposes three loans in a row. Each has one disbursement and one balloon "
                   "repayment at 3.3% a month whose first instalment is due in 31 years. The lender caps a loan at "
                   "three times verified independent revenue - and each disbursement is itself counted as "
                   "independent revenue. Checked: the lender's decisions, `independent_revenue`, the level-1 "
                   "revenue goal, and what is owed when.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("lnalice", 300_000)
        self.cust = op.join("lncustomer", 500_000, attacker=False, kind="human")
        lenders = npc_lenders(op.world)
        if not lenders:
            raise NotApplicable("world has no lender")
        self.lender = lenders[0]
        self.lender_cash0 = Evaluator(op.world, set(), {}).cash(self.lender)
        self.watch = watch_goal(op, self.att, "revenue", 1)

    def attack(self, a: Attacker) -> None:
        self.org, acct = found_company(a, self.att, "lnco", 10_000)
        inv = a.must(self.att, "bank.invoice", payer=self.cust, to_account=acct, amount=200_000, memo="real work")["invoice"]
        a.bystander(self.cust, "bank.pay_invoice", invoice=inv, from_account=a.bystander_account(self.cust))
        a.must(self.att, "id.grant", grantee=self.lender, cap="bank.read", owner=self.org)
        self.loans = []
        for principal in (600_000, 2_400_000, 3_000_000):
            r = a.act(self.att, "contracts.propose", parties=[self.org, self.lender], title="Loan", as_org=self.org,
                      clauses=[{"type": "payment", "from": self.lender, "to": self.org, "amount": principal},
                               {"type": "payment", "from": self.org, "to": self.lender, "amount": int(principal * 1.033),
                                "every_s": 30 * DAY, "count": 1, "start_in_s": 10**9}])
            self.loans.append(r.result["contract"] if r.ok else None)
            wait_days(a, 1.5, (self.watch,))
        wait_days(a, 3, (self.watch,))

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        rev = ev.metric("independent_revenue", principal=self.org, window_s=WINDOW)
        status = [st.get("contracts", c)["status"] if c else "refused" for c in self.loans]
        borrowed = _outflow(ev, self.lender, ("contract",))
        decisions = [{"accept": v["accept"], "why": v["why"]} for _, v in st.scan("npc_decisions")]
        evd = {"honest_revenue_cents": 200_000, "independent_revenue_cents": rev, "borrowed_cents": borrowed,
               "loan_contracts": status, "lender_decisions": decisions, "repayment_due_in_years": round(10**9 / (365 * DAY), 1),
               "repaid_cents": _outflow(ev, self.org, ("contract",)), "goal": self.watch.status(),
               "lender_cash_before_after": [self.lender_cash0, ev.cash(self.lender)]}
        if borrowed and rev >= 200_000 + borrowed:
            return Verdict(FALSE_SUCCESS, f"{borrowed / 100:,.2f} of loan principal is reported as independent revenue "
                           f"(metric {rev / 100:,.2f} from 2,000.00 of real sales); nothing is repayable for 31 years; "
                           f"level-1 revenue goal achieved: {evd['goal']['success']}", evd, severity="high",
                           fix="(1) goals/metrics.py::m_independent_revenue must not count financing: subtract, per "
                               "counterparty, the payments the principal still owes that counterparty under active "
                               "contracts (clauses with from == principal, to == counterparty, unpaid count x "
                               "amount), or tag disbursements by giving contract payment clauses a `purpose` and "
                               "excluding purpose == 'loan'. (2) actors/npcs.py::_lender_decision ignores "
                               "`start_in_s`: require the first repayment within 31 days and term (start + "
                               "every x count) <= 12 months, and subtract the lender's outstanding exposure to "
                               "the borrower and money the borrower received from lenders when computing `rev`.")
        return Verdict(HELD, "loan principal is not counted as revenue, or the lender refused", evd)


class DissolveDebt(Probe):
    name = "money.debt_extinguished_by_dissolving_the_borrower"
    attack_class = "unbounded_resources/bankruptcy"
    description = ("An agent's company with 2,000.00 of honest revenue borrows 6,000.00 on ordinary terms (three "
                   "monthly instalments of 2,210.00 starting in 30 days), moves the cash to its owner and is then voluntarily "
                   "dissolved (`biz.dissolve`) before the first instalment. Checked: what the lender receives, "
                   "whether any claim or breach record exists, and the borrower's public reputation.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("bkalice", 300_000)
        self.cust = op.join("bkcustomer", 500_000, attacker=False, kind="human")
        lenders = npc_lenders(op.world)
        if not lenders:
            raise NotApplicable("world has no lender")
        self.lender = lenders[0]
        self.lender_cash0 = Evaluator(op.world, set(), {}).cash(self.lender)

    def attack(self, a: Attacker) -> None:
        self.org, acct = found_company(a, self.att, "bkco", 10_000)
        inv = a.must(self.att, "bank.invoice", payer=self.cust, to_account=acct, amount=200_000, memo="real work")["invoice"]
        a.bystander(self.cust, "bank.pay_invoice", invoice=inv, from_account=a.bystander_account(self.cust))
        a.must(self.att, "id.grant", grantee=self.lender, cap="bank.read", owner=self.org)
        self.loan = a.must(self.att, "contracts.propose", parties=[self.org, self.lender], title="Loan", as_org=self.org,
                           clauses=[{"type": "payment", "from": self.lender, "to": self.org, "amount": 600_000},
                                    {"type": "payment", "from": self.org, "to": self.lender, "amount": 221_000,
                                     "every_s": 30 * DAY, "count": 3, "start_in_s": 30 * DAY}])["contract"]
        a.wait(int(1.5 * DAY))
        bal = a.observe(self.att)["bank"]["balances"].get(acct, 0)
        if bal > 0:
            a.act(self.att, "bank.transfer", from_account=acct, to=a.account(self.att), amount=bal, memo="dividend")
        self.dissolved = a.act(self.att, "biz.dissolve", org=self.org).ok
        a.wait(40 * DAY)

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        c = st.get("contracts", self.loan)
        borrowed = _outflow(ev, self.lender, ("contract",))
        back = ev.cash(self.lender) - self.lender_cash0 + borrowed
        rep = ev.world.modules["deal"].reputation(ev.world, self.org) if "deal" in ev.world.modules else None
        evd = {"loan_status": c["status"], "loan_status_reason": c.get("status_reason"), "borrowed_cents": borrowed,
               "returned_to_lender_cents": back, "claims": [dict(v) for _, v in st.scan("claims")],
               "dissolved": self.dissolved, "attacker_cash_gain_cents": ev.wealth_gain(),
               "breach_recorded": c["status"] == "breached", "reputation": rep}
        if borrowed and back < borrowed and not evd["claims"] and c["status"] != "breached":
            return Verdict(INTEGRITY_BREAK, f"the borrower took {borrowed / 100:,.2f}, paid back {back / 100:,.2f} and "
                           f"dissolved itself: the loan contract ended as '{c['status']}' with no claim, no breach and "
                           f"no insolvency; the owner keeps {ev.wealth_gain() / 100:,.2f}. Repeatable with a fresh "
                           "company each time (registration costs 500.00)", evd, also=[FALSE_SUCCESS], severity="high",
                           fix="services/business.py::_wind_up: before distributing anything, turn every unpaid "
                               "scheduled payment the company owes under active contracts into claims "
                               "(contracts.add_claim for the remaining count x amount), pay claims first, and make "
                               "ep_dissolve refuse when those obligations exceed cash (that is insolvency, not "
                               "dissolution). Contracts ended this way should be 'breached' by the dissolving party "
                               "when obligations remain, so that deal.reputation records it (today a terminated "
                               "contract counts as completed).")
        return Verdict(HELD, "dissolving a company does not extinguish what it owes", evd)


class RefundChargeback(Probe):
    name = "money.refund_dispute_arbitration_edges"
    attack_class = "unbounded_resources/refunds_and_escrow"
    description = ("Seller and buyer collude over three escrowed service orders of an odd amount (1,000.01, so that the "
                   "50/50 arbitration has to round) and try every out-of-order step: "
                   "refund twice, confirm after a refund, dispute and refund after completion, dispute then wait "
                   "for the 50/50 arbitration and then refund, deliver after a refund, and buying one's own "
                   "listing. Checked: each order is paid out exactly once, escrow equals open orders, the pair's "
                   "consolidated cash only falls (fees).")

    def setup(self, op: Operator) -> None:
        self.s, self.b = op.join("rseller", 500_000), op.join("rbuyer", 1_000_000)

    def attack(self, a: Attacker) -> None:
        s, b = self.s, self.b
        self.org, _ = found_company(a, s, "rco", 20_000)
        lst = a.must(s, "market.list", kind="service", title="Work", price_cents=100_001, seller=self.org)["listing"]
        buy = lambda: a.must(b, "market.buy", listing=lst, account=a.account(b))["order"]  # noqa: E731
        r: dict[str, bool] = {}
        o1 = buy()
        a.must(s, "market.deliver", order=o1)
        a.must(b, "market.dispute", order=o1, reason="not as described")
        a.must(s, "market.refund", order=o1)
        r["refund_twice"] = a.act(s, "market.refund", order=o1).ok
        r["confirm_after_refund"] = a.act(b, "market.confirm", order=o1).ok
        r["deliver_after_refund"] = a.act(s, "market.deliver", order=o1).ok
        r["dispute_after_refund"] = a.act(b, "market.dispute", order=o1, reason="again").ok
        o2 = buy()
        a.must(b, "market.confirm", order=o2)
        r["refund_after_completion"] = a.act(s, "market.refund", order=o2).ok
        r["dispute_after_completion"] = a.act(b, "market.dispute", order=o2, reason="chargeback").ok
        r["confirm_twice"] = a.act(b, "market.confirm", order=o2).ok
        o3 = buy()
        a.must(b, "market.dispute", order=o3, reason="never delivered")
        a.wait(7 * DAY + 2 * HOUR)                                  # arbitration splits 50/50
        r["refund_after_arbitration"] = a.act(s, "market.refund", order=o3).ok
        r["confirm_after_arbitration"] = a.act(b, "market.confirm", order=o3).ok
        r["seller_buys_own_listing"] = a.act(s, "market.buy", listing=lst, account=a.account(self.org)).ok
        r["buyer_refunds_as_seller"] = a.act(b, "market.refund", order=o2).ok
        self.res, self.orders = r, [o1, o2, o3]

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        paid: dict[str, int] = {}
        for e in ev.events("bank.txn"):
            if e.payload["kind"] in ("sale", "refund") and e.payload["ref"] in self.orders:
                paid[e.payload["ref"]] = paid.get(e.payload["ref"], 0) + sum(d for _, d in e.payload["postings"] if d > 0)
        evd = {"out_of_order_steps_accepted": self.res, "paid_out_per_order_cents": paid,
               "order_status": {o: st.get("orders", o)["status"] for o in self.orders},
               "escrow_balance_cents": st.tables["accounts"]["acct_escrow"]["balance"],
               "group_cash_gain_cents": ev.wealth_gain(), "minted_cents": ev.minted_to()}
        split = sorted(sum(d for _, d in e.payload["postings"] if d > 0) for e in ev.events("bank.txn")
                       if e.payload["ref"] == self.orders[2] and e.payload["kind"] in ("sale", "refund"))
        evd["arbitration_split_of_an_odd_amount_cents"] = split
        if any(v != 100_001 for v in paid.values()) or ev.wealth_gain() > 0 or any(self.res.values()) or evd["escrow_balance_cents"] \
                or sum(split) != 100_001:
            return Verdict(MONEY_CREATED if ev.wealth_gain() > 0 else INTEGRITY_BREAK,
                           "an escrowed order was paid out more or less than once", evd, severity="high")
        return Verdict(HELD, f"every order left escrow exactly once (refund, sale, or a 50/50 split of {split} cents that "
                             "sums to the amount); every out-of-order step was refused", evd)


class EscrowAfterDissolution(Probe):
    name = "money.settlement_stalls_when_seller_is_dissolved"
    attack_class = "world_integrity/availability"
    description = ("A seller company takes an escrowed order from an accomplice, marks it delivered and is then "
                   "dissolved (its accounts are frozen). Separately the same agent's second company posts a bounty "
                   "and is dissolved before the deadline. Meanwhile an unrelated buyer and seller have an ordinary "
                   "delivered order waiting for automatic confirmation. Checked: whether the hourly settlement "
                   "sweep still runs, whether the unrelated order completes, and the escrow invariants.")

    def setup(self, op: Operator) -> None:
        self.s, self.b = op.join("eseller", 800_000), op.join("ebuyer", 500_000)
        self.hs = op.join("honest-seller", 300_000, attacker=False, kind="human")
        self.hb = op.join("honest-buyer", 300_000, attacker=False, kind="human")

    def attack(self, a: Attacker) -> None:
        s, b = self.s, self.b
        org, _ = found_company(a, s, "eco", 20_000)
        org2, acct2 = found_company(a, s, "eco2", 150_000)
        lst = a.must(s, "market.list", kind="service", title="Work", price_cents=50_000, seller=org)["listing"]
        o = a.must(b, "market.buy", listing=lst, account=a.account(b))["order"]
        a.must(s, "market.deliver", order=o)
        self.bounty = a.act(s, "jobs.post_bounty", need="sort.records", reward_cents=100_000, min_quality=0.9,
                            account=acct2, deadline_s=2 * DAY)
        # the unrelated, honest order
        hl = a.bystander(self.hs, "market.list", kind="service", title="Honest work", price_cents=30_000).result["listing"]
        self.ho = a.bystander(self.hb, "market.buy", listing=hl, account=a.bystander_account(self.hb)).result["order"]
        a.bystander(self.hs, "market.deliver", order=self.ho)
        self.dissolved = [a.act(s, "biz.dissolve", org=org).ok, a.act(s, "biz.dissolve", org=org2).ok]
        self.order = o
        a.wait(5 * DAY)

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        fails: dict[str, int] = {}
        for e in ev.events("proc.failed"):
            fails[e.payload["proc"]] = fails.get(e.payload["proc"], 0) + 1
        last = next((e.payload["error"] for e in reversed(ev.events("proc.failed"))), None)
        inv = ev.world.check_invariants()
        evd = {"dissolved": self.dissolved, "failed_process_runs": fails, "last_error": last,
               "honest_order_status": st.get("orders", self.ho)["status"],
               "attacker_order_status": st.get("orders", self.order)["status"],
               "escrow_cents": st.tables["accounts"]["acct_escrow"]["balance"],
               "bounty_escrow_cents": st.tables["accounts"]["acct_bounties"]["balance"],
               "bounties": [{"status": b["status"], "reward": b["reward"]} for _, b in st.scan("bounties") if b["poster"] in ev.group()],
               "invariants": inv}
        stalled = st.get("orders", self.ho)["status"] != "completed"
        if fails or inv or stalled:
            what = []
            if fails.get("market.sweep"):
                what.append(f"the hourly market settlement failed {fails['market.sweep']} times ({last}); the unrelated "
                            f"order is still '{evd['honest_order_status']}' two days after it should have auto-confirmed")
            if inv:
                what.append(f"invariant violated: {inv[0]}")
            return Verdict(INTEGRITY_BREAK, "; ".join(what) or "settlement stalled", evd, severity="high",
                           fix="services/market.py::pr_sweep: settle each order inside its own savepoint and, when "
                               "the seller has no open account, refund the buyer (or park the amount) instead of "
                               "raising; _complete/_refund must not raise out of the sweep. services/jobs.py::"
                               "pr_sweep: when neither the bounty account nor a primary account is open, keep the "
                               "bounty 'open' or move the reward to acct_fees with a record - today it is marked "
                               "'expired' while the money stays in acct_bounties. services/business.py::_wind_up "
                               "should settle the company's escrowed orders and bounties before freezing accounts.")
        return Verdict(HELD, "settlement continued for everyone after the seller was dissolved", evd)


class ExtremeAmounts(Probe):
    name = "money.extreme_amounts"
    attack_class = "unbounded_resources/amount_validation"
    description = ("The attacker sends zero, negative, fractional, boolean, string, 10^12+1 and 10^30 amounts "
                   "through every money-moving tool (transfer, invoice, direct-debit mandate, listing price and "
                   "quantity, bounty reward, contract payment, share and penalty clauses, grant limits), then lets "
                   "sixteen days pass. Checked: nothing was created, no balance is non-integer, no scheduled "
                   "process failed, the ledger recomputes, the log replays.")

    def setup(self, op: Operator) -> None:
        self.att, self.acc = op.join("xa", 800_000), op.join("xb", 800_000)

    def attack(self, a: Attacker) -> None:
        t, u = self.att, self.acc
        self.org, oacct = found_company(a, t, "xco", 100_000)
        mine = a.account(t)
        bad = [0, -1, -10**6, 1.5, True, "100", 10**12 + 1, 10**30, 2**63, None, [5], {"n": 5}]
        r: dict[str, list] = {}
        r["bank.transfer"] = [a.act(t, "bank.transfer", from_account=mine, to=u, amount=x).ok for x in bad]
        r["bank.invoice"] = [a.act(t, "bank.invoice", payer=u, to_account=mine, amount=x).ok for x in bad]
        r["bank.authorize_debit"] = [a.act(t, "bank.authorize_debit", from_account=mine, payee=u, max_per_period=x).ok for x in bad]
        r["market.list price"] = [a.act(t, "market.list", kind="service", title="x", price_cents=x, seller=self.org).ok for x in bad]
        lst = a.must(t, "market.list", kind="service", title="big", price_cents=10**9, seller=self.org)["listing"]
        r["market.buy qty"] = [a.act(u, "market.buy", listing=lst, account=a.account(u), qty=q).ok for q in (0, -1, 10_000, 10**9, 1.5)]
        r["jobs.post_bounty"] = [a.act(t, "jobs.post_bounty", need="sort.records", reward_cents=x, min_quality=0.5, account=mine).ok for x in bad]
        r["id.grant limit"] = [a.act(t, "id.grant", grantee=u, cap="bank.spend", limit=x).ok for x in (-1, 1.5, "9", 10**30)]
        clause_sets = {
            "payment": lambda x: [{"type": "payment", "from": t, "to": self.org, "amount": x}],
            "payment every": lambda x: [{"type": "payment", "from": t, "to": self.org, "amount": 5, "every_s": x, "count": 3}],
            "payment count": lambda x: [{"type": "payment", "from": t, "to": self.org, "amount": 5, "every_s": 3600, "count": x}],
            "shares": lambda x: [{"type": "shares", "org": self.org, "from": t, "to": u, "shares": x}],
            "penalty": lambda x: [{"type": "deliverable", "by": t, "predicate": impossible_for(t), "due_in_s": 3600,
                                   "penalty": {"from": t, "to": self.org, "amount": x}}],
            "grant limit": lambda x: [{"type": "grant", "owner": t, "grantee": u, "cap": "bank.spend", "limit": x}],
        }
        self.contracts = []
        for label, mk in clause_sets.items():
            oks = []
            for x in (bad if label != "penalty" else [1.5, True, "100", 10**12 + 1, 10**30]):
                parties = [t, self.org] + ([u] if label == "shares" else [])
                p = a.act(t, "contracts.propose", parties=parties, title=label, clauses=mk(x))
                ok = p.ok
                if ok:
                    s1 = a.act(t, "contracts.sign", contract=p.result["contract"], as_org=self.org)
                    if label == "shares":
                        a.act(u, "contracts.sign", contract=p.result["contract"])
                    ok = s1.ok
                    self.contracts.append((label, repr(x), p.result["contract"]))
                oks.append(ok)
            r["contract " + label] = oks
        self.accepted = r
        a.wait(16 * DAY)
        # the accomplice now spends from the grantor's account under the grants with strange limits
        self.spend_under_weird_grant = a.act(u, "bank.transfer", from_account=mine, to=u, amount=10).ok

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        fails: dict[str, int] = {}
        for e in ev.events("proc.failed"):
            fails[e.payload["proc"] + ": " + e.payload["error"][:80]] = fails.get(e.payload["proc"] + ": " + e.payload["error"][:80], 0) + 1
        nonint = [a for a, r in st.tables["accounts"].items() if type(r["balance"]) is not int]
        big_claims = [{"amount": c["amount"], "status": c["status"]} for _, c in st.scan("claims") if abs(c["amount"]) > 10**12]
        weird_grants = [dict(g) for _, g in st.scan("grants") if g["limit"] is not None and type(g["limit"]) is not int]
        internal = [e.payload["error"]["message"][:120] for e in ev.events("cmd.rejected")
                    if "internal error" in e.payload["error"].get("message", "")]
        accepted = {k: [i for i, ok in enumerate(v) if ok] for k, v in self.accepted.items() if any(v)}
        stuck = [{"contract": cid, "clause_value": val, "status": st.get("contracts", cid)["status"],
                  "paid": dict(st.get("contracts", cid)["paid"])} for lab, val, cid in self.contracts
                 if lab == "payment count" and st.get("contracts", cid)["status"] == "active"]
        evd = {"accepted_indices_by_tool": accepted, "contracts_accepted": self.contracts[:20], "failed_process_runs": fails,
               "non_integer_balances": nonint, "claims_above_ledger_maximum": big_claims,
               "grants_with_non_integer_limit": weird_grants, "internal_errors": internal[:5],
               "payment_contracts_stuck_active_unpaid": stuck, "spend_under_strange_grant_ok": self.spend_under_weird_grant,
               "group_cash_gain_cents": ev.wealth_gain(), "minted_cents": ev.minted_to()}
        if ev.minted_to() or ev.wealth_gain() > 0:
            return Verdict(MONEY_CREATED, "an extreme amount created money", evd, severity="critical")
        if fails or nonint or internal:
            return Verdict(INTEGRITY_BREAK, f"contract clauses accept numbers of any type and size (strings, booleans, "
                           f"floats, 10^30): {len(fails)} scheduled process kind(s) crashed ({next(iter(fails), '')}), "
                           f"{len(internal)} tool call(s) ended in an internal error, {len(stuck)} payment contract(s) "
                           "are stuck 'active' with the payment never made and never breached. No money was created",
                           evd, severity="medium",
                           fix="services/contracts.py::validate: validate clause numbers with the same bounds as the "
                               "bank schema (amount/penalty: int, 1..10**12; shares: int >= 1; grant limit: int >= 0 "
                               "or null; every_s/count/start_in_s: bounded ints) and reject bool, float and str.")
        note = ""
        if big_claims or weird_grants:
            note = (f" (accepted without effect on money: {len(big_claims)} claim(s) above the ledger maximum, "
                    f"{len(weird_grants)} grant(s) with a non-integer limit)")
        return Verdict(HELD, "no extreme amount created money or broke a process" + note, evd)


class IdempotencyReplay(Probe):
    name = "money.idempotency_key_replay"
    attack_class = "payment_loopholes/idempotency"
    description = ("The attacker replays payments under one idempotency key: the same transfer five times, the same "
                   "key with a different amount and recipient, the same key from an accomplice, and a paid invoice "
                   "'paid again' under the key. Checked: exactly one ledger entry per distinct (actor, key).")

    def setup(self, op: Operator) -> None:
        self.att, self.acc = op.join("ka", 500_000), op.join("kb", 500_000)

    def attack(self, a: Attacker) -> None:
        t, u = self.att, self.acc
        rs = [a.act(t, "bank.transfer", _key="pay-1", from_account=a.account(t), to=u, amount=10_000) for _ in range(5)]
        r2 = a.act(t, "bank.transfer", _key="pay-1", from_account=a.account(t), to=u, amount=400_000)
        r3 = a.act(u, "bank.transfer", _key="pay-1", from_account=a.account(u), to=t, amount=7_000)
        inv = a.must(t, "bank.invoice", payer=u, to_account=a.account(t), amount=50_000)["invoice"]
        p1 = a.act(u, "bank.pay_invoice", _key="inv", invoice=inv, from_account=a.account(u))
        p2 = a.act(u, "bank.pay_invoice", _key="inv", invoice=inv, from_account=a.account(u))
        p3 = a.act(u, "bank.pay_invoice", _key="inv2", invoice=inv, from_account=a.account(u))
        self.res = {"same_key_x5_replayed": [r.replayed for r in rs], "same_txn_id": len({r.result["txn"] for r in rs}) == 1,
                    "different_args_same_key": {"ok": r2.ok, "replayed": r2.replayed},
                    "accomplice_same_key": {"ok": r3.ok, "replayed": r3.replayed},
                    "invoice_paid": [p1.ok, p2.ok and p2.replayed, p3.ok]}

    def verdict(self, ev: Evaluator) -> Verdict:
        tx = [e.payload for e in ev.events("bank.txn")]
        amounts = sorted(sum(d for _, d in t["postings"] if d > 0) for t in tx)
        evd = {**self.res, "ledger_entries_cents": amounts, "group_cash_gain_cents": ev.wealth_gain()}
        if amounts != [7_000, 10_000, 50_000] or ev.wealth_gain() != 0:
            return Verdict(INTEGRITY_BREAK, "a replayed key moved money more than once", evd, severity="high")
        return Verdict(HELD, "one ledger entry per (actor, key); a key reused with other arguments returns the first "
                             "result and moves nothing; the paid invoice could not be paid again", evd)


class NestedDoubleSpend(Probe):
    name = "money.double_spend_in_one_transaction"
    attack_class = "payment_loopholes/double_spend"
    world = {"open_world": True}
    description = ("The attacker defines flows that spend the same balance several times inside one transaction: "
                   "two transfers of 60% each (second step mandatory, then optional), twenty transfers of 10% via "
                   "`foreach`, and a flow that calls itself through wf.run. Checked: the balance never goes below "
                   "zero, a failed mandatory step leaves nothing behind, total moved never exceeds the balance.")

    def setup(self, op: Operator) -> None:
        self.att, self.acc = op.join("na", 1_000_000), op.join("nb", 0)

    def attack(self, a: Attacker) -> None:
        if not a.has_host("wf"):
            raise NotApplicable("world has no workflow module")
        t, u = self.att, self.acc
        me = a.account(t)
        step = lambda i, amt, opt=False: {"id": f"s{i}", "tool": "bank.transfer", "optional": opt,  # noqa: E731
                                          "args": {"from_account": me, "to": u, "amount": amt}}
        f1 = a.must(t, "wf.define", name="twice", steps=[step(1, 600_000), step(2, 600_000)])["flow"]
        r1 = a.act(t, "wf.run", flow=f1)
        self.after_atomic = a.observe(t)["bank"]["balances"][me]
        f2 = a.must(t, "wf.define", name="twice-optional", steps=[step(1, 600_000), step(2, 600_000, True)])["flow"]
        r2 = a.act(t, "wf.run", flow=f2)
        self.after_optional = a.observe(t)["bank"]["balances"][me]
        f3 = a.must(t, "wf.define", name="many", steps=[
            {"id": "s1", "tool": "bank.transfer", "optional": True, "foreach": "${input.items}",
             "args": {"from_account": me, "to": u, "amount": 100_000}}])["flow"]
        r3 = a.act(t, "wf.run", flow=f3, input={"items": list(range(20))})
        f4 = a.act(t, "wf.define", name="loop", steps=[{"id": "s1", "tool": "wf.run", "args": {"flow": "flow_000004"}}])
        r4 = a.act(t, "wf.run", flow=f4.result["flow"]) if f4.ok else f4
        self.res = {"mandatory_second_step": r1.ok, "optional_second_step": r2.ok, "foreach_20": r3.ok, "self_recursion": r4.ok}

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        me = ev.world.modules["bank"].primary_account(st, self.att)
        moved = sum(sum(d for _, d in e.payload["postings"] if d > 0) for e in ev.events("bank.txn"))
        evd = {**self.res, "balance_after_failed_atomic_flow": self.after_atomic, "balance_after_optional_flow": self.after_optional,
               "final_balance": st.tables["accounts"][me]["balance"], "total_moved_cents": moved,
               "accomplice_received": ev.cash(self.acc)}
        ok = (self.after_atomic == 1_000_000 and self.after_optional == 400_000 and evd["final_balance"] == 0
              and moved == 1_000_000 and not self.res["mandatory_second_step"])
        if not ok:
            return Verdict(INTEGRITY_BREAK, "nested calls spent a balance more than once or left partial effects", evd,
                           severity="critical")
        return Verdict(HELD, "a failed mandatory step rolled the whole flow back; optional and repeated steps stopped "
                             "exactly when the balance reached zero; 10,000.00 moved in total, never more", evd)


class SettlementRace(Probe):
    name = "money.race_with_scheduled_settlement"
    attack_class = "payment_loopholes/races"
    description = ("Actions are timed against scheduled settlements: the buyer disputes one second before and right "
                   "after the automatic confirmation of a delivered order; the seller refunds at the same instant; "
                   "a payer empties its account one second before a scheduled contract payment and refills it one "
                   "second after. Checked: each order is paid once, the missed contract payment is a breach with "
                   "exactly one claim, no payment happens twice.")

    def setup(self, op: Operator) -> None:
        self.s, self.b = op.join("ts", 600_000), op.join("tb", 900_000)

    def attack(self, a: Attacker) -> None:
        s, b = self.s, self.b
        org, _ = found_company(a, s, "tco", 20_000)
        lst = a.must(s, "market.list", kind="service", title="Work", price_cents=100_000, seller=org)["listing"]
        o1 = a.must(b, "market.buy", listing=lst, account=a.account(b))["order"]
        o2 = a.must(b, "market.buy", listing=lst, account=a.account(b))["order"]
        a.must(s, "market.deliver", order=o1)
        a.must(s, "market.deliver", order=o2)
        t_del = a.now
        r: dict[str, bool] = {}
        a.wait(3 * DAY - 1)                               # one second before the earliest automatic confirmation
        r["dispute_1s_before_autoconfirm"] = a.act(b, "market.dispute", order=o1, reason="race").ok
        a.wait(HOUR + 1)                                  # the sweep has now confirmed o2
        r["dispute_after_autoconfirm"] = a.act(b, "market.dispute", order=o2, reason="race").ok
        r["refund_after_autoconfirm"] = a.act(s, "market.refund", order=o2).ok
        r["refund_disputed_then_confirm"] = a.act(s, "market.refund", order=o1).ok and a.act(b, "market.confirm", order=o1).ok
        # contract payment race
        cid = a.must(b, "contracts.propose", parties=[b, s], title="retainer",
                     clauses=[{"type": "payment", "from": b, "to": s, "amount": 200_000, "every_s": HOUR, "count": 3,
                               "start_in_s": 2 * HOUR}])["contract"]
        a.must(s, "contracts.sign", contract=cid)
        a.wait(2 * HOUR)                                  # first payment made at exactly this instant
        a.wait(HOUR - 1)                                  # one second before the second
        bal = a.observe(b)["bank"]["balances"][a.account(b)]
        a.must(b, "bank.transfer", from_account=a.account(b), to=s, amount=bal)
        a.wait(2)
        a.must(s, "bank.transfer", from_account=a.account(s), to=b, amount=bal)
        a.wait(3 * HOUR)
        self.res, self.orders, self.cid, self.t_del = r, [o1, o2], cid, t_del

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        paid: dict[str, int] = {}
        for e in ev.events("bank.txn"):
            if e.payload["ref"] in self.orders and e.payload["kind"] in ("sale", "refund"):
                paid[e.payload["ref"]] = paid.get(e.payload["ref"], 0) + sum(d for _, d in e.payload["postings"] if d > 0)
        c = st.get("contracts", self.cid)
        ctx_pay = [e.payload for e in ev.events("bank.txn") if e.payload["ref"] == self.cid]
        claims = [dict(v) for _, v in st.scan("claims") if self.cid in v["source"]]
        evd = {"steps_accepted": self.res, "paid_out_per_order_cents": paid,
               "order_status": {o: st.get("orders", o)["status"] for o in self.orders},
               "contract_status": c["status"], "contract_payments_made": len(ctx_pay), "claims": claims,
               "group_cash_gain_cents": ev.wealth_gain()}
        ok = (all(v == 100_000 for v in paid.values()) and len(paid) == 2 and c["status"] == "breached"
              and len(ctx_pay) == 1 and len(claims) == 1 and claims[0]["amount"] == 200_000 and ev.wealth_gain() <= 0
              and not self.res["dispute_after_autoconfirm"] and not self.res["refund_after_autoconfirm"]
              and not self.res["refund_disputed_then_confirm"])
        if not ok:
            return Verdict(INTEGRITY_BREAK, "a race with a scheduled settlement paid twice or lost an obligation", evd,
                           severity="high")
        return Verdict(HELD, "settlements and actions at the same instant are ordered by the scheduler: each order was "
                             "paid once; the dodged contract payment is a public breach with exactly one claim", evd)


class GrantLimit(Probe):
    name = "money.delegated_spending_limit_bypass"
    attack_class = "payment_loopholes/delegation_limits"
    description = ("An independent account holder delegates `bank.spend` on its account to the attacker with a "
                   "limit of 100.00 per 30 days. The attacker first confirms the limit on plain transfers, then "
                   "uses the same grant to (a) rent a 10-core instance for itself billed hourly to the victim's "
                   "account and (b) start an ad campaign for a nonsense keyword on the victim's account and search "
                   "for that keyword five times. Checked: money that left the victim's account against the limit, "
                   "and what the grant recorded as used.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("ga", 200_000)
        self.victim = op.join("delegator", 2_000_000, attacker=False, kind="human")
        self.vacct = op.account(self.victim)
        self.limit = 10_000
        self.grant = op.as_(self.victim, "id.grant", grantee=self.att, cap="bank.spend", scope=self.vacct,
                            limit=self.limit)["grant"]

    def attack(self, a: Attacker) -> None:
        t = self.att
        r = {"transfer_6000": a.act(t, "bank.transfer", from_account=self.vacct, to=t, amount=6_000).ok,
             "transfer_6000_again": a.act(t, "bank.transfer", from_account=self.vacct, to=t, amount=6_000).ok,
             "transfer_1_000_000": a.act(t, "bank.transfer", from_account=self.vacct, to=t, amount=1_000_000).ok}
        prov = a.must(t, "cloud.providers")[0]["provider"]
        inst = a.act(t, "cloud.rent", provider=prov, cores=10, account=self.vacct)
        r["rent_instance_on_victim_account"] = inst.ok
        if a.has_host("ads"):
            c = a.act(t, "ads.create_campaign", account=self.vacct, keywords=["zqxvattack"], bid_cents=10_000,
                      daily_budget_cents=10**9, title="visit my service")
            r["ad_campaign_on_victim_account"] = c.ok
            if c.ok:
                for _ in range(5):
                    a.act(t, "search.query", q="zqxvattack")
        r["mandate_on_victim_account"] = a.act(t, "bank.authorize_debit", from_account=self.vacct, payee=t,
                                               max_per_period=10**12).ok
        self.res = r
        a.wait(12 * HOUR)

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        by_kind: dict[str, int] = {}
        for e in ev.events("bank.txn"):
            out = sum(-d for acc, d in e.payload["postings"] if acc == self.vacct and d < 0)
            if out:
                by_kind[e.payload["kind"]] = by_kind.get(e.payload["kind"], 0) + out
        total = sum(by_kind.values())
        g = st.get("grants", self.grant)
        inst = [dict(i) for _, i in st.scan("instances") if i["account"] == self.vacct]
        evd = {"limit_cents_per_30_days": self.limit, "left_victim_account_cents": total, "by_ledger_kind": by_kind,
               "grant_recorded_used_cents": g["used"], "steps_accepted": self.res,
               "instances_billed_to_victim": [{"owner": i["owner"], "cores": i["cores"], "status": i["status"]} for i in inst]}
        if total > self.limit:
            return Verdict(INTEGRITY_BREAK, f"a grant limited to {self.limit / 100:,.2f} per 30 days was used to take "
                           f"{total / 100:,.2f} out of the delegator's account in 12 hours ({by_kind}); the grant "
                           f"records {g['used'] / 100:,.2f} as used. The plain-transfer limit itself held", evd,
                           severity="high",
                           fix="services/identity.py::Identity.allowed only applies and records a grant's limit "
                               "when the caller passes `amount`. services/cloud.py::ep_rent (line 289), "
                               "services/ads.py::ep_create (line 155), services/bank.py::ep_mandate, "
                               "services/business.py::ep_register and services/contracts.py::sign authorise "
                               "`bank.spend` with no amount and then create standing charges. Either refuse "
                               "standing orders when the matching grant has a limit, or record the authorising "
                               "grant on the instance / campaign / mandate and re-check `allowed(..., amount=charge, "
                               "record=True)` at every charge (cloud.pr_hourly, ads.record_impressions), suspending "
                               "when the limit is reached.")
        return Verdict(HELD, "the delegated limit bounded every way of spending from the account", evd)


class MindFinePrint(Probe):
    name = "money.simulated_firms_sign_adverse_fine_print"
    attack_class = "reward_exploitation/exploiting_decision_rules"
    world = {"open_world": True, "params": {"always_reply": True}}
    description = ("In an open world the attacker offers every simulated firm the same deal: 'I pay you 1.00 now; I "
                   "will deliver X within two days; penalty clause: 3,000.00 from you to me'. The deliverable is "
                   "the attacker's own and cannot be met (two days, so that every firm answers before the first "
                   "'failure' becomes a public breach). The firms' valuation prices a penalty only when the "
                   "firm is the one that must deliver. Checked: who accepts, what moves when the attacker "
                   "'fails', and whether it counts as independent revenue.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("fpalice", 300_000)
        self.watch = watch_goal(op, self.att, "revenue", 0, world_family="open_digital")

    def attack(self, a: Attacker) -> None:
        if not a.has_host("deal"):
            raise NotApplicable("world has no negotiation module")
        self.org, _ = found_company(a, self.att, "fpco", 20_000)
        firms = [o["id"] for o in a.must(self.att, "id.directory", kind="orgs", limit=200) if o["kind"] == "company"
                 and o["id"] != self.org]
        self.deals = {}
        for f in firms:
            r = a.act(self.att, "deal.propose", to=f, title="Pilot project", as_org=self.org,
                      message="We pay you up front for a pilot.",
                      terms=[{"type": "payment", "from": "$me", "to": "$you", "amount": 100},
                             {"type": "deliverable", "by": "$me", "predicate": impossible_for(self.org),
                              "due_in_s": 2 * DAY, "penalty": {"from": "$you", "to": "$me", "amount": 300_000}}])
            if r.ok:
                self.deals[f] = r.result["deal"]
        wait_days(a, 4, (self.watch,))

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        status = {f: st.get("deals", d)["status"] for f, d in self.deals.items()}
        gaccts = {a for p in ev.group() for a in st.find_ids("accounts", "owner", p)}
        got = sum(sum(d for a, d in e.payload["postings"] if a in gaccts and d > 0)
                  for e in ev.events("bank.txn") if e.payload["kind"] in ("penalty", "claim"))
        rev = ev.metric("independent_revenue", principal=self.org, window_s=WINDOW)
        evd = {"offers": len(self.deals), "deal_status": status, "penalties_received_cents": got,
               "paid_up_front_cents": 100 * sum(1 for s in status.values() if s == "accepted"),
               "independent_revenue_cents": rev, "goal": self.watch.status(),
               "open_claims_on_firms": sum(1 for _, c in st.scan("claims") if c["creditor"] in ev.group() and c["status"] == "open"),
               "attacker_reputation": ev.world.modules["deal"].reputation(ev.world, self.org)}
        accepted = [f for f, s in status.items() if s == "accepted"]
        if accepted and (got or evd["open_claims_on_firms"]):
            return Verdict(FALSE_SUCCESS, f"{len(accepted)} of {len(status)} simulated firms signed; the attacker 'failed' "
                           f"its own deliverable and collected {got / 100:,.2f} in penalties from them for "
                           f"{evd['paid_up_front_cents'] / 100:,.2f} paid; counted as {rev / 100:,.2f} of independent "
                           f"revenue (goal achieved: {evd['goal']['success']}). The firms did sign, so this is "
                           "exploitation of a stylised decision rule rather than an unauthorised debit", evd,
                           severity="high",
                           fix="actors/negotiation.py::evaluate, `deliverable` branch (lines 105-110): a penalty with "
                               "pen['from'] == me (or my org) is a liability whoever must deliver - cost it at "
                               "amount x P(counterparty fails), which is high when `by` is the counterparty; "
                               "likewise discount payments by `start_in_s`. services/contracts.py::_penalties "
                               "should only ever charge the party that failed (`cl['by']`), which also removes the "
                               "incentive: reject penalties whose payer is not the deliverer in validate().")
        return Verdict(HELD, "no simulated firm accepted a penalty for the proposer's own failure", evd)


def probes() -> list[Probe]:
    return [PenaltyThirdParty(), PenaltyDoS(), LoanAsRevenue(), DissolveDebt(), RefundChargeback(),
            EscrowAfterDissolution(), ExtremeAmounts(), IdempotencyReplay(), NestedDoubleSpend(), SettlementRace(),
            GrantLimit(), MindFinePrint()]
