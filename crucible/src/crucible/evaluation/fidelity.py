"""The fidelity register: what the world's realism claims rest on, and what has (not) been checked.

Crucible is a model. This module states, mechanism by mechanism, which kind of realism each behavioural or
economic rule bears on and how far that realism has been established:

* ``calibrated``             - parameters were fitted to, and the output compared with, a named dataset or
                               publication, and the comparison code is in this repository. `validate_register`
                               refuses a calibrated entry without both.
* ``stylised-fact-checked``  - the simulated pattern is compared in code with a qualitative published regularity.
* ``plausible-hypothesis``   - a reasoned rule with chosen parameters; nothing in the repository compares it with
                               data.
* ``known-unrealistic``      - a simplification known to differ from reality (stated so that results that depend
                               on it are read accordingly).

**As of this version no entry is calibrated and none is stylised-fact-checked. Every entry is a plausible
hypothesis or a known simplification, and the repository contains no empirical reference data.** The qualitative
checks implemented here (e.g. whether an adoption curve has an interior inflection) are reported per world by
`fidelity_report`; they do not upgrade a register status.

Eight kinds of realism are reported separately: interface (do tools and protocols behave like their real
counterparts), causal (do outcomes follow from executed mechanisms an actor can influence), behavioural (do
simulated people and firms decide like real ones), economic (prices, costs, money, competition), temporal (delays,
rhythms, durations), strategic (do other actors respond to incentives and to each other), distributional
(heterogeneity and tails), transfer (does what works here work outside).

The measurement functions are deterministic functions of a finished world's state and event log (a world itself
is reproducible run-to-run only under `cost_model = "deterministic"`; with measured CPU time two runs of the same
seed differ slightly). They compute the simulated quantities one would compare with data:
adoption curves, retention and churn, price elasticity of demand, market concentration, entry and exit rates, the
service reliability distribution, compute utilisation and congestion (plus the daily demand profile and the
size distribution of offerings). They never contain reference values.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

DAY = 86400
KINDS = ("interface", "causal", "behavioural", "economic", "temporal", "strategic", "distributional", "transfer")
STATUSES = ("calibrated", "stylised-fact-checked", "plausible-hypothesis", "known-unrealistic")
NO_REFERENCE = "no reference data in repository"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # .../crucible


@dataclass(frozen=True)
class Mechanism:
    id: str
    modules: tuple            # kernel modules that implement it
    source: str               # file(s) under src/crucible
    kinds: tuple              # which kinds of realism it bears on
    status: str
    rule: str                 # how the outcome arises
    caveat: str = ""          # what is simplified, assumed or known to be wrong
    evidence: dict | None = None   # calibrated / stylised-fact-checked only: {"reference": ..., "comparison": path::function}

    def to_dict(self) -> dict:
        return {"id": self.id, "modules": list(self.modules), "source": self.source, "kinds": list(self.kinds),
                "status": self.status, "rule": self.rule, "caveat": self.caveat, "evidence": self.evidence}


def _m(id, modules, source, kinds, status, rule, caveat="", evidence=None) -> Mechanism:
    return Mechanism(id, tuple(modules), source, tuple(kinds), status, rule, caveat, evidence)


PH, KU = "plausible-hypothesis", "known-unrealistic"

REGISTER: list[Mechanism] = [
    # ------------------------------------------------------------------ money and payments
    _m("money.double_entry_ledger", ["bank"], "services/bank.py", ["causal", "economic", "interface"], PH,
       "integer cents on a double-entry ledger; every movement is one balanced transaction; balances sum to zero",
       "structural claim only (conservation, atomic transfers); no interest, inflation, taxes or foreign exchange"),
    _m("money.exogenous_sources", ["bank", "consumers", "admin"], "services/bank.py, economy/consumers.py",
       ["economic", "temporal"], KU,
       "money enters from declared source accounts: endowments at genesis, a fixed household income per day, the "
       "rest of the world buying nothing and selling hardware and energy",
       "household income is exogenous and constant; there is no credit creation beyond lenders' own funds, no "
       "macro feedback from firms' wages to household demand"),
    _m("money.instant_final_settlement", ["bank", "market", "contracts"], "services/bank.py", ["temporal", "interface"], KU,
       "transfers settle instantly and finally", "no clearing delay, float, chargeback or payment failure other "
       "than insufficient funds"),
    _m("money.delegated_authority", ["identity", "bank"], "services/identity.py", ["interface", "strategic"], PH,
       "capability grants with scope, 30-day spending limits and expiry; organisations act through roles",
       "limits are applied to one-off payments only (red-team finding money.delegated_spending_limit_bypass)"),
    # ------------------------------------------------------------------ demand
    _m("demand.request_arrivals", ["consumers"], "economy/consumers.py", ["behavioural", "temporal", "distributional"], PH,
       "requests per offering-hour ~ Poisson(sum of its customers' rates); rates drawn uniformly per consumer; an "
       "optional sinusoidal daily cycle (`diurnal`, `peak_hour`)",
       "rate distribution and cycle shape are parameters; no weekly or seasonal pattern, no bursts"),
    _m("demand.price_response", ["consumers"], "economy/consumers.py", ["economic", "behavioural"], PH,
       "each consumer has a willingness to pay (base price x uniform multiplier); utility falls with price relative "
       "to it and requests are reduced accordingly", "functional form and spread chosen, not estimated; see "
       "`price_elasticity` for the value the simulation actually produces"),
    _m("demand.household_budget", ["consumers", "bank"], "economy/consumers.py", ["economic"], PH,
       "consumer spending is paid from one budgeted pool that cannot go negative, so aggregate demand is "
       "budget-constrained", "one pooled budget for all households: no individual budgets, saving or borrowing"),
    _m("demand.choice_and_preferences", ["consumers"], "economy/consumers.py", ["behavioural", "distributional"], PH,
       "consumers shop among at most six offerings they know, by a weighted utility of experienced quality, "
       "latency and price with individual preference weights", "weights drawn from a fixed distribution per world"),
    _m("demand.discovery", ["consumers", "search", "ads", "forum"], "economy/consumers.py, services/search.py",
       ["behavioural", "strategic"], PH,
       "a consumer learns of an offering through organic search for its need, sponsored results, forum posts and "
       "word of mouth from six neighbours on a small-world graph",
       "search and word-of-mouth rates are per-family parameters; consumers do not browse web sites"),
    _m("demand.loyalty_and_switching", ["consumers"], "economy/consumers.py", ["behavioural", "temporal"], PH,
       "satisfaction is an exponential average of experienced utility; dissatisfied or unserved consumers switch, "
       "damped by individual loyalty", "see `retention_and_churn` for the simulated retention it produces"),
    _m("demand.turnover", ["consumers"], "economy/consumers.py", ["temporal", "distributional"], KU,
       "a constant share of consumers leaves each day and is replaced by newcomers with fresh preferences",
       "constant hazard independent of age, satisfaction or the economy; population size is fixed"),
    _m("demand.sampled_execution", ["consumers", "cloud"], "economy/consumers.py, services/cloud.py",
       ["causal", "distributional"], KU,
       "a few real requests per offering-hour are executed and verified; the rest are extrapolated from them",
       "per-request quality and cost are sampling estimates, not per-request truth"),
    _m("demand.voice", ["consumers", "market", "forum", "support"], "economy/consumers.py", ["behavioural"], PH,
       "customers leave ratings and reviews, and open support tickets when failure rates exceed 5%",
       "thresholds and propensities chosen"),
    _m("demand.statistical_population", ["consumers"], "economy/consumers.py", ["behavioural", "transfer"], KU,
       "thousands of consumers are rows of arrays updated by vectorised rules",
       "they do not reason, negotiate or read; they are a demand model, not agents"),
    _m("demand.behaviour_models_and_segments", ["consumers"], "economy/consumers.py, economy/behavior.py",
       ["behavioural", "distributional"], PH,
       "(4.0, off by default) structurally different consumer rule sets and heterogeneous segments selected by "
       "configuration", "which rule set resembles real customers is exactly the open question; none is fitted"),
    _m("demand.latent_and_derived", ["consumers"], "economy/consumers.py", ["economic", "temporal", "causal"], PH,
       "(4.0, off by default) latent needs become active as verified volume served in a source need grows "
       "(saturating response with a lag)", "functional form and lags chosen"),
    # ------------------------------------------------------------------ work, software, compute
    _m("work.verified_needs", ["consumers", "jobs", "civ"], "economy/needs.py", ["causal", "transfer", "interface"], KU,
       "value comes from a finite set of problem formats with exact verifiers holding ground truth",
       "eight formats; real demand for software is open-ended and rarely exactly verifiable"),
    _m("software.real_execution_and_cost", ["code", "cloud"], "services/code.py, services/cloud.py, sandbox/",
       ["causal", "economic", "transfer"], PH,
       "agent code really runs; measured CPU (or executed lines in the deterministic model) times a work scale "
       "consumes compute units", "the deterministic cost model undercounts native-code work; one machine type"),
    _m("compute.capacity_and_overload", ["cloud"], "services/cloud.py", ["causal", "economic"], PH,
       "an instance has cores x 100,000 compute units per hour; requests beyond that are refused (503)",
       "capacity resets on the hour; no queueing or latency degradation before refusal"),
    _m("compute.price_rule", ["cloud", "society", "civ"], "services/cloud.py, actors/dynamics.py", ["economic", "strategic"], KU,
       "providers post a price per core-hour that moves by a fixed rule with utilisation",
       "one region, no spot market, no negotiated or reserved pricing"),
    _m("compute.outages", ["cloud"], "services/cloud.py", ["temporal", "distributional"], PH,
       "each provider draws an outage each day with probability 1 - reliability, lasting 1 to 6 hours",
       "independent across providers and days; no correlated failures or degraded states"),
    _m("compute.billing_and_suspension", ["cloud", "bank"], "services/cloud.py", ["causal", "temporal"], PH,
       "instances are billed hourly; an unpaid bill suspends the instance and 72 hours later terminates it"),
    _m("compute.hardware_market", ["cloud"], "services/cloud.py", ["economic"], KU,
       "anyone may become a provider and buy cores at a fixed price with a fixed delivery lead time",
       "fixed price and lead time; no depreciation, energy price or supply constraint"),
    # ------------------------------------------------------------------ firms
    _m("firms.incumbent_policy", ["npcs"], "actors/npcs.py", ["strategic", "behavioural"], PH,
       "incumbents reprice toward observed competition and capacity, add capacity when overloaded, start R&D when "
       "out-rated, hire when queues grow", "fixed heuristics with a few drawn parameters; no learning"),
    _m("firms.minds", ["society"], "actors/minds.py, actors/producers.py", ["strategic", "behavioural", "causal"], PH,
       "firm minds keep their own books, publish problems, trial capabilities with randomised designs and adopt on "
       "their own measured effect", "decision rules are fixed heuristics with traits; no learning beyond beliefs"),
    _m("firms.entry", ["business", "society"], "services/business.py, actors/dynamics.py", ["strategic", "economic"], PH,
       "new companies register for a fee; simulated entrants appear by rule or by scheduled shock",
       "see `entry_exit` for realised rates; entry rules are not derived from expected profit"),
    _m("firms.insolvency", ["business", "contracts"], "services/business.py", ["economic", "temporal"], PH,
       "a company with claims it cannot pay for the grace period is wound up: accounts frozen, cash distributed "
       "pro rata, services stopped, contracts ended",
       "no restructuring, no secured creditors, no director liability; obligations not yet due vanish on "
       "voluntary dissolution (red-team finding money.debt_extinguished_by_dissolving_the_borrower)"),
    # ------------------------------------------------------------------ labour and finance
    _m("labour.reservation_wage_decision", ["npcs", "jobs", "contracts"], "actors/npcs.py", ["behavioural", "economic"], PH,
       "a worker accepts an offer when the monthly wage is at least its reservation wage times a risk factor from "
       "public signals (company age, breaches, reputation); unpaid workers leave and warn others",
       "the wage is taken at face value whatever its due date (red-team finding reward.fake_members.wage_never_due)"),
    _m("labour.productivity", ["npcs", "issues", "support"], "actors/npcs.py", ["behavioural", "distributional"], PH,
       "developers progress on assigned issues at a random rate that grows with skill and commit code of a tier "
       "set by skill; support staff resolve a skill-dependent number of tickets a day", "no effort choice, no teams"),
    _m("labour.effort_under_profit_sharing", ["gov", "society"], "services/governance.py", ["behavioural"], KU,
       "effort under a profit-sharing rule follows e = 0.55 + 2.5 s (clipped)", "an assumed formula, not an emergent "
       "behaviour; results about sharing rules restate it"),
    _m("finance.lender_rule", ["npcs", "contracts"], "actors/npcs.py", ["economic", "strategic"], PH,
       "a lender approves a loan of at most three times the borrower's verified 30-day independent revenue, at or "
       "above its rate, for at most 12 months",
       "repayment start dates and existing exposure are ignored and the metric it relies on counts earlier loans "
       "(red-team finding money.loan_counts_as_revenue_and_is_never_due)"),
    # ------------------------------------------------------------------ markets, contracts, institutions
    _m("market.escrow_and_arbitration", ["market"], "services/market.py", ["economic", "interface"], KU,
       "orders are escrowed; delivery is auto-confirmed after 3 days; unresolved disputes are split 50/50 after 7",
       "arbitration ignores evidence; no reputational or legal cost of disputing"),
    _m("market.ratings", ["market", "consumers", "search"], "services/market.py", ["behavioural", "strategic"], PH,
       "only buyers may rate, once each; ratings feed search rank and consumer choice",
       "one rating per buyer whatever was spent; a free order counts as a purchase (red-team finding)"),
    _m("market.bounties", ["jobs"], "services/jobs.py", ["economic", "interface"], PH,
       "a poster escrows a reward paid to the first submission clearing a quality bar on hidden inputs"),
    _m("contracts.clauses_and_breach", ["contracts"], "services/contracts.py", ["interface", "causal", "strategic"], PH,
       "agreements are compositions of payment, membership, grant, share, deliverable and notice clauses; a missed "
       "payment or deliverable is a public breach and a claim",
       "enforcement is automatic and free; no courts, delay or partial performance"),
    _m("negotiation.valuation_and_concession", ["deal", "society"], "actors/negotiation.py", ["strategic", "behavioural"], PH,
       "a mind values an offer from its own position (payments weighted by trust, time, shares, delegated authority, "
       "effort), accepts above an aspiration that decays over rounds, otherwise counters or rejects",
       "penalties are priced only when the mind itself must deliver and due dates are not discounted (red-team "
       "finding money.simulated_firms_sign_adverse_fine_print)"),
    _m("reputation.public_record_and_trust", ["deal", "society", "contracts"], "services/deals.py, actors/minds.py",
       ["strategic", "behavioural"], PH,
       "public record of completed and breached agreements; private trust is a Beta update per counterparty",
       "identities are free to create, so a record can be shed by founding a new organisation"),
    _m("governance.voting", ["gov", "society"], "services/governance.py, actors/negotiation.py", ["strategic"], KU,
       "constitutions are enforced at authorisation; simulated members vote from trust in the proposer",
       "votes do not come from modelled payoffs"),
    # ------------------------------------------------------------------ information
    _m("information.search_ranking", ["search", "net", "ads"], "services/search.py, services/net.py",
       ["interface", "strategic"], PH,
       "BM25 relevance times a rating signal times a count of linking hosts; ads are shown above organic results",
       "no spam detection; links from the page owner's own hosts count (red-team finding)"),
    _m("information.ad_auction", ["ads"], "services/ads.py", ["economic", "strategic"], KU,
       "highest bids first, one ad per advertiser, each impression charged at the bid up to a daily budget",
       "pay-per-impression at one's own bid; no quality score, no second-price rule, no click model"),
    _m("information.forum_and_mail", ["forum", "mail"], "services/forum.py, services/mail.py", ["interface", "temporal"], PH,
       "posts, votes and follows; mail with a per-family delivery latency and loss probability",
       "people read a bounded number of posts; no spam filtering"),
    _m("information.beliefs_and_evidence", ["society", "caps"], "actors/minds.py, services/capabilities.py",
       ["behavioural", "causal"], PH,
       "minds update beliefs by source and act on evidence records whose references the world can verify",
       "update rule and source weights chosen"),
    # ------------------------------------------------------------------ technology and operations
    _m("technology.trials_and_adoption", ["caps", "society"], "services/capabilities.py, actors/minds.py",
       ["causal", "behavioural", "temporal"], PH,
       "a firm installs a capability under a randomised switchback or washout trial on real days and keeps it if "
       "its own measured effect is positive", "see `adoption_curves`; trial lengths and thresholds chosen"),
    _m("technology.simulated_invention", ["society"], "actors/innovators.py", ["strategic", "transfer"], KU,
       "simulated innovators generate controller programs from one grammar, deploy them and iterate on trial evidence",
       "one grammar family; field trials are the only test"),
    _m("operations.books_and_losses", ["ops"], "economy/operations.py", ["causal", "economic"], PH,
       "daily profit and loss by ledger kind and a journal of operating losses (overload, outage, downtime) derived "
       "from recorded events", "losses are counted by rule, not valued by the firm"),
    _m("automation.workflows", ["wf"], "services/workflows.py", ["interface"], PH,
       "triggers and flows call tools as their owner on schedules or events", "no partial failure modes of real schedulers"),
    # ------------------------------------------------------------------ the web
    _m("web.http_semantics", ["net", "appdb", "webgen"], "services/net.py, worldgen/sitegen.py", ["interface", "transfer"], PH,
       "agent-deployed and generated sites answer requests with status codes, sessions, consent login, per-app "
       "storage, rate limits and outages; a crawler builds the index",
       "server-rendered only; no TLS, DNS hierarchy, cross-site cookies or third-party scripts"),
    _m("web.adversarial_content", ["adversary"], "worldgen/adversarial.py", ["strategic", "transfer"], KU,
       "injection pages, a fraudulent capability, spam and false documentation created at genesis",
       "static adversaries that do not adapt to defences"),
    # ------------------------------------------------------------------ exogenous structure
    _m("world.scheduled_shocks", ["shocks", "civ"], "worlds/digital_civilization.py", ["temporal", "strategic"], KU,
       "preference drift, demand surges, compute shortages, entrants and reliability storms happen on a schedule "
       "fixed by the world family", "scripted; not caused by anything inside the world"),
    _m("world.genesis_draws", ["civ"], "worlds/digital_civilization.py, worldgen/generator.py", ["distributional"], KU,
       "firms, prices, cash, wages, skills and consumer attributes are drawn from uniform ranges set per family",
       "uniform ranges, independent draws; real size and wage distributions are heavy-tailed and correlated"),
    _m("world.action_time_costs", ["identity", "admin", "goals", "docs", "cal", "issues", "support", "code"],
       "core/kernel.py and the service modules", ["temporal", "interface"], KU,
       "every tool call costs a fixed number of simulated seconds and an hourly action limit applies",
       "fixed per tool; real effort depends on content. These service modules are record-keeping interfaces "
       "(profiles, self-goals, documents, calendars, issues, tickets, repositories) with no behavioural rule of "
       "their own"),
    # ------------------------------------------------------------------ industrial families
    _m("industry.production_and_wear", ["industry", "industrygen"], "economy/industry.py", ["causal", "economic"], PH,
       "yields depend on machine settings on parametric response surfaces; machines wear and break down; energy "
       "grids curtail when demand exceeds capacity", "response surfaces and wear laws are chosen parametric forms"),
    _m("industry.freight", ["freight"], "economy/industry.py", ["temporal", "economic"], PH,
       "goods move on links with transit time, cost and daily capacity; congestion delays and reprices"),
    _m("industry.household_choice", ["shoppers"], "economy/shoppers.py", ["behavioural", "distributional"], PH,
       "households choose products by a linear utility in attributes, price and distance with Gumbel noise, within "
       "a budget, after discovering products by search and word of mouth",
       "a standard discrete-choice functional form with chosen coefficients; not estimated"),
    _m("lab.network_physics", ["lab"], "worlds/engineering_lab.py", ["causal", "transfer"], KU,
       "a static per-hour flow model on a capacitated network with queueing delay and drops",
       "no packet dynamics, no fixed point after drops"),
]

NOT_BEHAVIOURAL = {"admin"}       # operator tools only: no rule that an in-world outcome depends on


def register_for_module(name: str) -> list[Mechanism]:
    return [m for m in REGISTER if name in m.modules]


def validate_register(register: list[Mechanism] | None = None) -> list[str]:
    """Problems with the register itself. In particular: nothing may be `calibrated` (or `stylised-fact-checked`)
    without a reference and comparison code that exists in this repository."""
    errs = []
    reg = REGISTER if register is None else register
    seen = set()
    for m in reg:
        if m.id in seen:
            errs.append(f"{m.id}: duplicate id")
        seen.add(m.id)
        if m.status not in STATUSES:
            errs.append(f"{m.id}: unknown status {m.status!r}")
        if not m.kinds or any(k not in KINDS for k in m.kinds):
            errs.append(f"{m.id}: kinds must be among {KINDS}")
        if not m.modules or not m.rule:
            errs.append(f"{m.id}: needs modules and a rule")
        if m.status in ("calibrated", "stylised-fact-checked"):
            ev = m.evidence or {}
            if not ev.get("reference") or not ev.get("comparison"):
                errs.append(f"{m.id}: status {m.status} needs evidence {{reference, comparison}}")
                continue
            path, _, fn = str(ev["comparison"]).partition("::")
            full = os.path.join(REPO, path)
            if not os.path.isfile(full):
                errs.append(f"{m.id}: comparison code {path} is not in the repository")
            elif fn and f"def {fn}" not in open(full, encoding="utf-8").read():
                errs.append(f"{m.id}: {path} has no function {fn}")
            if m.status == "calibrated" and not ev.get("dataset"):
                errs.append(f"{m.id}: a calibrated entry must name the dataset (file in the repository) it was fitted to")
            elif m.status == "calibrated" and not os.path.isfile(os.path.join(REPO, str(ev["dataset"]))):
                errs.append(f"{m.id}: dataset {ev['dataset']} is not in the repository")
        elif m.evidence:
            errs.append(f"{m.id}: evidence given but status is {m.status}")
    return errs


def modules_with_behavioural_rules(world) -> list[str]:
    """Kernel modules of a world whose outcomes follow from a rule of their own: anything that runs a scheduled
    process, keeps population arrays, or lives in the economy / actors / worlds packages."""
    out = []
    procs = {p.split(".", 1)[0] for p in world.processes}
    for name, mod in world.modules.items():
        pkg = type(mod).__module__
        if name in NOT_BEHAVIOURAL:
            continue
        if name in procs or pkg.startswith(("crucible.economy", "crucible.actors", "crucible.worlds", "crucible.mechanisms",
                                           "crucible.usefulness")):
            out.append(name)
    return sorted(out)


def uncovered_modules(world) -> list[str]:
    """Modules with behavioural rules that have no register entry (must be empty: see tests/test_fidelity.py)."""
    return [m for m in modules_with_behavioural_rules(world) if not register_for_module(m)]


# ====================================================================== measurement
def _events(world, types: tuple) -> list:
    return [e for e in world.events_range(0) if e.type in types]


def _slots(world) -> dict[int, dict]:
    st = world.state
    out = {}
    if "cons_offers" not in st.tables:
        return out
    for lid, o in st.scan("cons_offers"):
        l = st.get("listings", lid)
        out[int(o["slot"])] = {"listing": lid, "need": o["need"], "host": l["host"] if l else None,
                               "seller": l["seller"] if l else None, "t": o["t"]}
    return out


def _daily(world) -> dict[int, dict[int, dict]]:
    """{day: {slot: {customers (end of day), requests, ok, errors, unavailable, overload, revenue, price}}}."""
    days: dict[int, dict[int, dict]] = defaultdict(dict)
    for e in _events(world, ("consumers.hour",)):
        p = e.payload
        d = (int(p["t"]) - 1) // DAY
        for k, sm in p["slots"].items():
            r = days[d].setdefault(int(k), {"customers": 0, "requests": 0, "ok": 0, "errors": 0, "unavailable": 0,
                                            "overload": 0, "revenue": 0, "price": None})
            for f in ("requests", "ok", "errors", "unavailable", "overload", "revenue"):
                r[f] += int(sm.get(f, 0))
            r["customers"] = int(sm.get("customers", 0))
        for k, pub in (p.get("public") or {}).items():
            if int(k) in days[d]:
                days[d][int(k)]["price"] = pub.get("price")
    return dict(days)


def _result(value, how: str, n: int, extra: dict | None = None) -> dict:
    return {"value": value, "how_measured": how, "n": int(n), "comparison": NO_REFERENCE, **(extra or {})}


def adoption_curves(world) -> dict:
    """Customers of each offering at the end of every simulated day, with the usual descriptors of a diffusion
    curve: peak, days to 10 / 50 / 90 % of the peak, the day of fastest growth, and whether growth first
    accelerates and then slows (an interior inflection: the qualitative S shape of diffusion curves)."""
    daily, slots = _daily(world), _slots(world)
    curves = {}
    for s, info in sorted(slots.items()):
        ser = [(d, daily[d][s]["customers"]) for d in sorted(daily) if s in daily[d]]
        if not ser:
            continue
        vals = [v for _, v in ser]
        peak = max(vals)
        d0 = ser[0][0]

        def reach(frac, vals=vals, ser=ser, peak=peak, d0=d0):
            return next((d - d0 for d, v in ser if peak and v >= frac * peak), None)
        inc = [b - a for a, b in zip([0] + vals[:-1], vals)]
        k = int(np.argmax(inc)) if inc else 0
        curves[info["host"] or info["listing"]] = {
            "need": info["need"], "series": [[int(d), int(v)] for d, v in ser], "peak": int(peak),
            "days_to_10pct": reach(0.1), "days_to_50pct": reach(0.5), "days_to_90pct": reach(0.9),
            "day_of_fastest_growth": int(ser[k][0] - d0), "interior_inflection": bool(0 < k < len(inc) - 1),
            "final_share_of_need": None}
    by_need: dict[str, int] = defaultdict(int)
    for c in curves.values():
        by_need[c["need"]] += c["series"][-1][1]
    for c in curves.values():
        tot = by_need[c["need"]]
        c["final_share_of_need"] = round(c["series"][-1][1] / tot, 4) if tot else 0.0
    t50 = [c["days_to_50pct"] for c in curves.values() if c["days_to_50pct"] is not None and c["peak"] > 0]
    s_shaped = [c["interior_inflection"] for c in curves.values() if c["peak"] > 0]
    return _result({"median_days_to_half_peak": float(np.median(t50)) if t50 else None,
                    "share_with_interior_inflection": round(float(np.mean(s_shaped)), 4) if s_shaped else None,
                    "offerings": curves},
                   "customers per offering at the last hourly market update of each simulated day (consumers.hour "
                   "events); days counted from the offering's first day in the market", len(curves),
                   {"qualitative_check": "diffusion curves are typically S-shaped (growth accelerates, then slows); "
                                         "`share_with_interior_inflection` is the share of offerings whose fastest "
                                         "growth is neither on the first nor on the last day observed. Reported, "
                                         "not used to set any register status"})


def retention_and_churn(world, horizons: tuple = (1, 7)) -> dict:
    """Exact cohort retention of consumers, from the event log: the world is replayed day by day and each
    consumer's provider at one day boundary is compared with its provider `h` days later."""
    from ..core.kernel import World
    if world.log_base != 0 or "consumers" not in world.modules:
        return _result(None, "needs the full event log of a world with consumers", 0)
    events = [e.to_dict() for e in world.log]
    bounds, last_day = [], None
    for i, e in enumerate(events):
        d = e["t"] // DAY
        if last_day is not None and d != last_day:
            bounds.append(i)
        last_day = d
    bounds.append(len(events))
    snaps, blob, start = [], None, 0
    for b in bounds:
        w = World.replay(world.spec, events[start:b], world_id="fidelity-replay", verify=False, base_blob=blob)
        snaps.append(np.array(w.state.arrays["c_prov"], copy=True))
        blob, start = w.snapshot_blob(), b
        w.close()
    out = {}
    for h in horizons:
        kept = switched = left = total = 0
        for i in range(len(snaps) - h):
            a, b2 = snaps[i], snaps[i + h]
            has = a >= 0
            total += int(has.sum())
            kept += int((has & (b2 == a)).sum())
            left += int((has & (b2 < 0)).sum())
            switched += int((has & (b2 >= 0) & (b2 != a)).sum())
        out[f"{h}d"] = {"retention": round(kept / total, 5) if total else None,
                        "switched": round(switched / total, 5) if total else None,
                        "left_unserved": round(left / total, 5) if total else None, "customer_days": total}
    r1 = out.get("1d", {}).get("retention")
    out["churn_per_day"] = round(1 - r1, 5) if r1 is not None else None
    return _result(out, "event log replayed to every day boundary; share of customers of an offering still with the "
                        "same offering h days later (pooled over offerings and days); consumer turnover counts as "
                        "leaving", len(snaps))


def price_elasticity(world, window_days: int = 2) -> dict:
    """Arc elasticity of an offering's requests with respect to its own price, from price changes that happened
    inside the world (`market.listing_update`). Also a difference-in-differences version that subtracts the
    change in the other offerings of the same need over the same windows."""
    daily, slots = _daily(world), _slots(world)
    by_listing = {v["listing"]: s for s, v in slots.items()}
    st = world.state
    rows = []
    price = {}
    for e in world.events_range(0):
        if e.type == "market.listing":
            price[e.payload["id"]] = e.payload["price"]
        elif e.type == "market.listing_update" and "price" in e.payload["changes"]:
            lid = e.payload["id"]
            p0, p1 = price.get(lid), e.payload["changes"]["price"]
            price[lid] = p1
            if lid not in by_listing or not p0 or not p1 or p0 == p1:
                continue
            s, d = by_listing[lid], e.t // DAY
            before = [daily[x][s]["requests"] for x in range(d - window_days, d) if x in daily and s in daily[x]]
            after = [daily[x][s]["requests"] for x in range(d + 1, d + 1 + window_days) if x in daily and s in daily[x]]
            if len(before) < window_days or len(after) < window_days or not sum(before):
                continue
            q0, q1 = float(np.mean(before)), float(np.mean(after))
            c0 = [daily[x][s]["customers"] for x in range(d - window_days, d) if x in daily and s in daily[x]]
            c1 = [daily[x][s]["customers"] for x in range(d + 1, d + 1 + window_days) if x in daily and s in daily[x]]
            pc0 = sum(before) / max(1, sum(c0))
            pc1 = sum(after) / max(1, sum(c1))
            dp = (p1 - p0) / ((p1 + p0) / 2)
            per_customer = ((pc1 - pc0) / ((pc1 + pc0) / 2)) / dp if (pc0 + pc1) > 0 else None
            need = slots[s]["need"]
            rivals = [x for x, v in slots.items() if v["need"] == need and x != s]
            r0 = sum(daily[x][r]["requests"] for x in range(d - window_days, d) for r in rivals if x in daily and r in daily[x])
            r1 = sum(daily[x][r]["requests"] for x in range(d + 1, d + 1 + window_days) for r in rivals if x in daily and r in daily[x])
            arc = ((q1 - q0) / ((q1 + q0) / 2)) / ((p1 - p0) / ((p1 + p0) / 2))
            did = None
            if r0 > 0 and r1 > 0 and q1 > 0:
                did = (math.log(q1 / q0) - math.log(r1 / r0)) / math.log(p1 / p0)
            rows.append({"listing": lid, "host": st.get("listings", lid)["host"], "day": int(d), "t": int(e.t), "price_before": p0,
                         "price_after": p1, "requests_per_day_before": round(q0, 2), "requests_per_day_after": round(q1, 2),
                         "customers_before": int(np.mean(c0)) if c0 else 0, "customers_after": int(np.mean(c1)) if c1 else 0,
                         "arc_elasticity": round(arc, 4),
                         "per_customer_elasticity": None if per_customer is None else round(per_customer, 4),
                         "did_elasticity": None if did is None else round(did, 4)})
    for r in rows:      # a window is clean when the same offering's price did not change again inside it
        r["clean_window"] = not any(o is not r and o["listing"] == r["listing"] and abs(o["day"] - r["day"]) <= window_days
                                    for o in rows)
    arcs = [r["arc_elasticity"] for r in rows]
    dids = [r["did_elasticity"] for r in rows if r["did_elasticity"] is not None]
    clean = [r["arc_elasticity"] for r in rows if r["clean_window"]]
    return _result({"median_arc_elasticity": float(np.median(arcs)) if arcs else None,
                    "median_arc_elasticity_clean_windows": float(np.median(clean)) if clean else None,
                    "median_did_elasticity": float(np.median(dids)) if dids else None, "changes": rows},
                   f"for every in-world price change of an API offering with {window_days} full days of demand on both "
                   "sides: (dQ/mean Q)/(dP/mean P) on consumer requests per day; the difference-in-differences "
                   "version subtracts the log change of the other offerings of the same need", len(rows),
                   {"note": "no in-world price change with enough demand on both sides was observed" if not rows else
                    "observational: in a market whose customer base is still growing (or shrinking) the before/after "
                    "figure mixes the price effect with that trend - as it would in field data. "
                    "`per_customer_elasticity` (requests per customer-day) removes the customer-count trend but "
                    "also the customers lost to the price; `price_elasticity_counterfactual` gives the causal figure"})


def price_elasticity_counterfactual(world, listing: str | None = None, factor: float = 1.2, days: int = 2) -> dict:
    """The causal own-price elasticity the simulator implies *now*: the finished world is forked twice, one fork
    changes the offering's price by `factor` (an ordinary `market.update` by its seller), both run `days` days,
    and requests are compared. The world itself is not touched. Exact only under the deterministic cost model;
    the two forks draw different random numbers after the change, so small differences are sampling noise."""
    st = world.state
    slots = _slots(world)
    cands = sorted((v["listing"] for v in slots.values() if st.get("listings", v["listing"])["status"] == "active"),
                   key=lambda lid: (-st.get("listings", lid)["price"], lid))
    lid = listing or (cands[0] if cands else None)
    if lid is None:
        return _result(None, "no active consumer offering", 0)
    l = st.get("listings", lid)
    p0 = l["price"]
    p1 = max(p0 + 1, int(round(p0 * factor))) if factor > 1 else max(1, min(p0 - 1, int(round(p0 * factor))))
    if p1 == p0:
        return _result(None, "price cannot be changed by one cent in that direction", 0)
    owners = [m["person"] for m in st.find("members", "org", l["seller"]) if m["role"] == "owner"] or [l["seller"]]
    slot = next(s for s, v in slots.items() if v["listing"] == lid)
    blob = world.snapshot_blob()
    from ..core.kernel import World
    out = {}
    for label, change in (("baseline", False), ("changed", True)):
        f = World.from_blob(blob, world_id=f"fidelity-cf-{label}")
        seq0 = f.seq
        if change:
            f.system_call("market.update", {"listing": lid, "price_cents": int(p1)}, as_principal=owners[0])
        f.advance_time(f.now + days * DAY)
        req = sum(int(e.payload["slots"].get(str(slot), {}).get("requests", 0)) for e in f.events_range(seq0)
                  if e.type == "consumers.hour")
        out[label] = {"requests": req, "customers_end": int(f.state.get("cons_summary", str(slot))["customers"])}
        f.close()
    q0, q1 = out["baseline"]["requests"], out["changed"]["requests"]
    arc = ((q1 - q0) / ((q1 + q0) / 2)) / ((p1 - p0) / ((p1 + p0) / 2)) if (q0 + q1) else None
    return _result({"listing": lid, "host": l["host"], "price_baseline": p0, "price_changed": int(p1), "days": days,
                    "forks": out, "arc_elasticity": None if arc is None else round(arc, 4)},
                   f"two forks of the finished world run {days} days, one after an ordinary price change by the seller; "
                   "arc elasticity of the offering's consumer requests between the forks", 2)


def _hhi(xs) -> float:
    tot = float(sum(xs))
    return float(sum((x / tot) ** 2 for x in xs)) if tot else 0.0


def concentration(world) -> dict:
    """Herfindahl-Hirschman index of customer shares per need, per simulated day."""
    daily, slots = _daily(world), _slots(world)
    series: dict[str, list] = defaultdict(list)
    for d in sorted(daily):
        by_need: dict[str, list] = defaultdict(list)
        for s, r in daily[d].items():
            if s in slots:
                by_need[slots[s]["need"]].append(r["customers"])
        for need, xs in sorted(by_need.items()):
            series[need].append([int(d), round(_hhi(xs), 4), sum(1 for x in xs if x > 0)])
    summ = {need: {"mean": round(float(np.mean([h for _, h, _ in ser])), 4), "first": ser[0][1], "last": ser[-1][1],
                   "min": min(h for _, h, _ in ser), "max": max(h for _, h, _ in ser), "active_offerings_last": ser[-1][2],
                   "series": ser} for need, ser in series.items()}
    means = [v["mean"] for v in summ.values()]
    return _result({"mean_hhi_over_needs": round(float(np.mean(means)), 4) if means else None, "needs": summ},
                   "sum of squared customer shares of the offerings of a need (0..1; 1 = monopoly) at the end of each "
                   "simulated day", sum(len(s) for s in series.values()))


def entry_exit(world) -> dict:
    """Entry and exit of offerings and of companies, per incumbent-month."""
    st = world.state
    days = max(1.0, world.now / DAY)
    offers_in = [e for e in _events(world, ("consumers.offer",))]
    genesis_offers = sum(1 for e in offers_in if e.t <= 3600)      # present at the first hourly market update
    closed = set()
    for e in _events(world, ("market.listing_update",)):
        if e.payload["changes"].get("status") == "closed":
            closed.add(e.payload["id"])
    offer_ids = {e.payload["id"] for e in offers_in}
    offer_exits = len(closed & offer_ids)
    reg = _events(world, ("biz.registered",))
    comp_entries = sum(1 for e in reg if e.t > 0)
    comp_exits = defaultdict(int)
    for e in _events(world, ("biz.status",)):
        comp_exits[e.payload["status"]] += 1
    n_comp0 = sum(1 for e in reg if e.t == 0)
    months = days / 30.0
    rate = lambda n, base: round(n / (max(1, base) * months), 5)  # noqa: E731
    return _result({"offering_entries": len(offers_in) - genesis_offers, "offering_exits": offer_exits,
                    "offerings_at_start": genesis_offers,
                    "offering_entry_rate_per_incumbent_month": rate(len(offers_in) - genesis_offers, genesis_offers),
                    "offering_exit_rate_per_incumbent_month": rate(offer_exits, genesis_offers),
                    "company_entries": comp_entries, "company_exits": dict(comp_exits), "companies_at_start": n_comp0,
                    "company_entry_rate_per_incumbent_month": rate(comp_entries, n_comp0),
                    "company_exit_rate_per_incumbent_month": rate(sum(comp_exits.values()), n_comp0),
                    "simulated_days": round(days, 2)},
                   "entries: offerings joining the consumer market after its first hourly update and company "
                   "registrations after genesis; exits: "
                   "closed listings of such offerings and companies ended as bankrupt or dissolved; rates per "
                   "incumbent at genesis per 30 simulated days", len(offers_in) + len(reg))


def reliability_distribution(world) -> dict:
    """Distribution over host-days of the share of requests served, plus provider outage time."""
    st = world.state
    av = []
    if "svc_days" in st.tables:
        for _, r in st.scan("svc_days"):
            tot = r["ok"] + r["errors"] + r["unavailable"] + r["overload"]
            if tot:
                av.append(r["ok"] / tot)
    outage_h = sum((e.payload["until"] - e.payload["start"]) / 3600.0 for e in _events(world, ("cloud.outage",)))
    providers = st.count("providers") if "providers" in st.tables else 0
    a = np.asarray(av, float)
    q = lambda p: round(float(np.quantile(a, p)), 5) if len(a) else None  # noqa: E731
    return _result({"mean": round(float(a.mean()), 5) if len(a) else None, "p05": q(0.05), "p50": q(0.5), "p95": q(0.95),
                    "min": round(float(a.min()), 5) if len(a) else None,
                    "share_of_host_days_below_0.99": round(float((a < 0.99).mean()), 5) if len(a) else None,
                    "share_of_host_days_below_0.95": round(float((a < 0.95).mean()), 5) if len(a) else None,
                    "provider_outage_hours_per_provider_month": round(outage_h / max(1, providers) / max(1e-9, world.now / (30 * DAY)), 4)},
                   "per host and simulated day: served requests / (served + errors + unavailable + overloaded), from "
                   "the service day counters; outage hours from scheduled provider outages", len(a))


def compute_utilisation(world) -> dict:
    """How full the compute is: cores rented out of cores owned (time-averaged from instance events), compute
    units used out of instance capacity per host-day, and how often demand was turned away."""
    from ..services.cloud import CU_PER_CORE_HOUR
    st = world.state
    if "providers" not in st.tables:
        return _result(None, "world has no compute module", 0)
    cores_total = sum(p["cores"] for _, p in st.scan("providers")) or 1
    now = max(1, world.now)
    cores: dict[str, int] = {}
    running: dict[str, bool] = {}
    area, t_prev, rented = 0.0, 0, 0
    peak = 0
    for e in _events(world, ("cloud.instance", "cloud.resize", "cloud.instance_status")):
        area += rented * (e.t - t_prev)
        t_prev = e.t
        p = e.payload
        if e.type == "cloud.instance":
            cores[p["id"]], running[p["id"]] = p["cores"], True
        elif e.type == "cloud.resize":
            cores[p["id"]] = p["cores"]
        else:
            running[p["id"]] = p["status"] == "running"
        rented = sum(c for i, c in cores.items() if running.get(i))
        peak = max(peak, rented)
    area += rented * (now - t_prev)
    host_cores = {}
    for _, d in st.scan("deployments"):
        inst = st.get("instances", d["instance"])
        if inst is not None:
            host_cores[d["host"]] = inst["cores"]
    util, req, over, unav = [], 0, 0, 0
    today = world.now // DAY
    for k, r in st.scan("svc_days"):
        host, day = k.rsplit("|", 1)
        req += r["requests"] + r["unavailable"] + r["overload"]
        over += r["overload"]
        unav += r["unavailable"]
        if int(day) < today and host in host_cores:           # complete days only
            util.append(r["cu"] / (host_cores[host] * CU_PER_CORE_HOUR * 24))
    u = np.asarray(util, float)
    refused = sum(1 for e in _events(world, ("cmd.rejected",)) if e.payload["error"].get("code") == "capacity_exceeded")
    return _result({"rented_share_of_cores_time_avg": round(area / now / cores_total, 5),
                    "rented_share_of_cores_peak": round(peak / cores_total, 5),
                    "instance_utilisation_mean": round(float(u.mean()), 5) if len(u) else None,
                    "instance_utilisation_p95": round(float(np.quantile(u, 0.95)), 5) if len(u) else None,
                    "share_of_host_days_above_0.8": round(float((u > 0.8).mean()), 5) if len(u) else None,
                    "overloaded_share_of_requests": round(over / req, 6) if req else None,
                    "unavailable_share_of_requests": round(unav / req, 6) if req else None,
                    "rentals_refused_for_capacity": refused},
                   "rented cores / provider cores, time-weighted from instance events (current provider capacity); "
                   "compute units used per complete host-day / (cores x 100,000 x 24) with the instance's current "
                   "cores; overloaded and unavailable requests from the service day counters", len(u))


def demand_timing(world) -> dict:
    """Requests by hour of day (peak-to-trough ratio and the peak hour)."""
    by_hour = np.zeros(24)
    n = 0
    for e in _events(world, ("consumers.hour",)):
        h = ((int(e.payload["t"]) - 1) // 3600) % 24
        by_hour[h] += sum(int(s.get("requests", 0)) for s in e.payload["slots"].values())
        n += 1
    if not n or by_hour.sum() == 0:
        return _result(None, "no consumer demand in this world", 0)
    prof = by_hour / by_hour.sum()
    return _result({"peak_hour": int(prof.argmax()), "peak_to_trough": round(float(prof.max() / max(prof.min(), 1e-12)), 4),
                    "share_by_hour": [round(float(x), 5) for x in prof]},
                   "consumer requests summed by simulated hour of day over the whole run", n)


def size_distribution(world) -> dict:
    """Heterogeneity of offering sizes at the end of the run (customers per offering): Gini and top share."""
    daily, slots = _daily(world), _slots(world)
    if not daily:
        return _result(None, "no consumer demand in this world", 0)
    last = daily[max(daily)]
    xs = np.sort(np.asarray([r["customers"] for s, r in last.items() if s in slots], float))
    if not len(xs) or xs.sum() == 0:
        return _result(None, "no customers", 0)
    n = len(xs)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(xs) / (n * xs.sum()))
    return _result({"gini": round(gini, 5), "top_share": round(float(xs[-1] / xs.sum()), 5), "offerings": int(n)},
                   "customers per offering after the last hourly update: Gini coefficient and the largest share", n)


MEASURES = {   # name -> (function, kinds of realism it bears on)
    "adoption_curves": (adoption_curves, ("behavioural", "temporal")),
    "retention_and_churn": (retention_and_churn, ("behavioural",)),
    "price_elasticity": (price_elasticity, ("economic", "causal")),
    "concentration": (concentration, ("economic", "strategic")),
    "entry_exit": (entry_exit, ("strategic", "economic")),
    "reliability_distribution": (reliability_distribution, ("distributional", "interface")),
    "compute_utilisation": (compute_utilisation, ("economic", "causal")),
    "demand_timing": (demand_timing, ("temporal",)),
    "size_distribution": (size_distribution, ("distributional",)),
}


def measure_world(world) -> dict:
    return {name: fn(world) for name, (fn, _) in MEASURES.items()}


def _headline(name: str, res: dict):
    v = res.get("value")
    if not isinstance(v, dict):
        return v
    keep = {k: x for k, x in v.items() if not isinstance(x, (dict, list))}
    if name == "retention_and_churn":
        keep = {"retention_1d": (v.get("1d") or {}).get("retention"), "retention_7d": (v.get("7d") or {}).get("retention"),
                "churn_per_day": v.get("churn_per_day")}
    return keep


def fidelity_report(worlds, counterfactual: bool = False) -> dict:
    """The register and the simulated quantities of the given finished worlds, reported separately for the eight
    kinds of realism. Every measure carries its value per world, how it was measured, and its comparison status -
    which is "no reference data in repository" for all of them, because that is the truth."""
    worlds = list(worlds)
    per_world = []
    for w in worlds:
        per_world.append({"world": w.world_id, "family": w.spec.family, "seed": w.spec.seed,
                          "simulated_days": round(w.now / DAY, 2), "measures": measure_world(w),
                          **({"price_elasticity_counterfactual": price_elasticity_counterfactual(w)}
                             if counterfactual and "consumers" in w.modules else {}),
                          "modules_without_register_entry": uncovered_modules(w)})
    by_status = {s: sum(1 for m in REGISTER if m.status == s) for s in STATUSES}
    kinds = {}
    for k in KINDS:
        mech = [m for m in REGISTER if k in m.kinds]
        meas = []
        for name, (fn, mk) in MEASURES.items():
            if k not in mk:
                continue
            vals = [{"world": pw["world"], "value": _headline(name, pw["measures"][name])} for pw in per_world]
            how = next((pw["measures"][name]["how_measured"] for pw in per_world), fn.__doc__.strip().split("\n")[0])
            meas.append({"measure": name, "values": vals, "how_measured": how, "comparison": NO_REFERENCE})
        kinds[k] = {"mechanisms": [{"id": m.id, "status": m.status, "caveat": m.caveat} for m in mech],
                    "status_counts": {s: sum(1 for m in mech if m.status == s) for s in STATUSES},
                    "measures": meas,
                    "note": ("no quantitative measure is implemented for this kind; transfer can only be established "
                             "by authorised real-world experiments (crucible.realworld), and none has been run"
                             if k == "transfer" else
                             ("no quantitative measure is implemented for this kind" if not meas else ""))}
    return {"schema": "crucible-fidelity/1",
            "statement": (f"{by_status['calibrated']} of {len(REGISTER)} mechanisms are calibrated and "
                          f"{by_status['stylised-fact-checked']} are checked against a published stylised fact; "
                          f"{by_status['plausible-hypothesis']} are plausible hypotheses and "
                          f"{by_status['known-unrealistic']} are known simplifications. The repository contains no "
                          "empirical reference data: every measure below is a simulated quantity with nothing to "
                          "compare it to."),
            "register_problems": validate_register(),
            "status_counts": by_status, "kinds": kinds,
            "register": [m.to_dict() for m in REGISTER], "worlds": per_world}
