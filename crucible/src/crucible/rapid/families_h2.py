"""Families of workstream H2: the `planning` engine (resource allocation, 8.4.1-8.4.6; complementary production
8.9.1 and multi-agent scheduling 8.9.4) and the `strategic` engine (strategic games 8.8.1-8.8.6; distributed
information 8.9.2, shared-resource coordination 8.9.3, collaborative discovery 8.9.5 and team formation 8.9.6).

Each builder draws one world from a seed: neutral names, the hidden mechanism parameters (lead times, shelf lives,
recipes, rival policy types and their private parameters, counterparts' issue weights, patch capacities, ingredient
types, worker skills...) and the public terms of the world. Evaluation families (split `val`, partition v5-eval-1)
use participant policy families that no training family uses (held-out opponents: tit-for-tat and Boulware
negotiators, a regret-matching player) or a structure no training family has. The evaluator side of every family
(oracle, scripted public reference, identification, intervention tests) is in `crucible.rapid.evaluator_h2`, which
no world module imports.
"""
from __future__ import annotations

import numpy as np

from ..mechanisms.base import neutral_names
from .families import family

E = "crucible.rapid.evaluator_h2:"


def u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def hooks(name: str) -> dict:
    return {"claims": (E + f"claim_{name}",), "identify": E + f"identify_{name}", "reference": E + f"reference_{name}",
            "oracle": E + f"oracle_{name}"}


# ====================================================================== planning engine: 8.4 resource allocation
@family("plan_distribution", engine="planning", category="8.4", item="8.4.1", split="train",
        doc="A depot replenished by a supplier after a hidden lead time ships stock to sites whose demand rates are "
            "hidden; unmet demand is lost and sales are censored by stock.",
        distinct_because="two-echelon distribution: an order lever with a hidden replenishment lag feeds a depot whose "
                         "stock is split over sites by allocation; demand is only seen through censored sales",
        axes={"graph_size": 3, "horizon": 50, "resources": 40}, objective="maximize_net", **hooks("distribution"))
def plan_distribution(rng, d):
    M = int(d["graph_size"])
    names = neutral_names(rng, M + 1)
    # sites differ in demand (an even ladder from a low to a high rate, in a random order): where stock goes matters
    lo_, hi_ = u(rng, 1.5, 2.5), u(rng, 5.5, 7.5)
    ladder = [lo_ * (hi_ / lo_) ** (k / max(1, M - 1)) for k in range(M)]
    lam = [float(ladder[i]) for i in rng.permutation(M)]
    lead = int(rng.integers(2, 6))
    return {"engine": "planning", "scenario": "distribution", "horizon": int(d["horizon"]), "site_names": names[:M],
            "order_name": names[M], "lam": lam, "lead": lead, "price": 50.0, "unit_cost": 20.0, "ship_cost": 2.0,
            "hold_depot": 0.4, "hold_site": 1.0, "max_order": float(20 * M), "ship_cap": float(d["resources"]),
            "init_depot": 20.0, "init_sites": [float(round(2 * x)) for x in lam], "endowment": 100_000,
            "objective": {"type": "maximize_net", "text": "Maximise sales revenue minus purchase, shipping and "
                                                          "holding costs."},
            "brief": f"You run a depot that supplies {M} sites. Order units from the supplier (lever "
                     f"{names[M]}; they reach the depot after a lead time you do not know) and ship depot stock to "
                     "the sites (allocate; shipments arrive next period). Customers buy at the sites; demand that "
                     "finds no stock is lost. Price 50, unit cost 20, shipping 2 per unit, holding 0.4 per unit at "
                     "the depot and 1 at a site per period."}


@family("plan_perishable", engine="planning", category="8.4", item="8.4.2", split="train",
        doc="Daily production of a good with a hidden shelf life; fresh-only, flexible and bargain customers with "
            "a weekly demand cycle; a markdown on older units and the issuing order (fifo or lifo).",
        distinct_because="age-structured lots with a hidden expiry, customer segments that substitute between fresh "
                         "and older units, a markdown that creates bargain demand, and a discrete issuing policy",
        axes={"horizon": 56, "resources": 40}, objective="maximize_net", **hooks("perishable"))
def plan_perishable(rng, d):
    names = neutral_names(rng, 2)
    return {"engine": "planning", "scenario": "perishable", "horizon": int(d["horizon"]), "bake_name": names[0],
            "markdown_name": names[1], "lam_bar": u(rng, 8.0, 13.0), "amp": u(rng, 0.3, 0.6),
            "phase": u(rng, 0.0, 7.0), "fresh_frac": u(rng, 0.25, 0.6), "bargain": u(rng, 0.6, 1.6),
            "shelf": int(rng.integers(2, 5)), "unit_price": 10.0, "unit_cost": 4.0, "disposal_cost": 1.0,
            "hold_cost": 0.2, "max_bake": float(d["resources"]), "init_stock": 0.0, "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise sales minus production, disposal and holding "
                                                          "costs."},
            "brief": f"Each period you make units of a perishable good (lever {names[0]}); they can be sold that "
                     "period and for a few more periods, then they must be thrown away. Some customers take only "
                     "units made the same period; others take any; a markdown (lever "
                     f"{names[1]}, fraction off) applies to older units and attracts bargain hunters. Decision "
                     "'issue' sets whether flexible customers get the oldest (fifo) or newest (lifo) units first. "
                     "Price 10, unit cost 4, disposal 1 per unit, holding 0.2 per unit per period."}


@family("plan_supply_chain", engine="planning", category="8.4", item="8.4.3", split="train",
        doc="A serial supply chain (retailer, intermediate stages, factory) with hidden shipping and production "
            "lags, backlogs, and customer demand that shifts at a hidden time.",
        distinct_because="multi-echelon serial chain: one order lever per stage, backlogged orders, lagged "
                         "shipments between stages and a hidden demand regime shift; controller programs allowed",
        axes={"graph_size": 3, "horizon": 50}, objective="maximize_net", **hooks("supply_chain"))
def plan_supply_chain(rng, d):
    K = int(d["graph_size"])
    H = int(d["horizon"])
    names = neutral_names(rng, 2 * K)
    mu0 = u(rng, 4.0, 8.0)
    mu1 = mu0 * (u(rng, 0.45, 0.65) if rng.random() < 0.5 else u(rng, 1.5, 1.9))
    return {"engine": "planning", "scenario": "supply_chain", "horizon": H, "stage_names": names[:K],
            "order_names": names[K:], "lags": [int(rng.integers(1, 4)) for _ in range(K)], "mu0": mu0, "mu1": mu1,
            "t_change": int(rng.integers(H // 3, 2 * H // 3)), "unit_price": 12.0, "raw_cost": 4.0,
            "hold": [0.5] + [0.3] * (K - 1), "lost_cost": 2.0, "max_order": 30.0, "init_stock": [12.0] * K,
            "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise sales minus raw material, holding and backlog "
                                                          "costs over the whole chain."},
            "brief": f"A chain of {K} stages: {names[0]} sells to customers; each stage orders from the next one up "
                     f"and {names[K - 1]} produces from raw material. Levers {', '.join(names[K:])} are the orders "
                     "each stage places this period (one-off quantities). Orders a stage cannot fill wait as its "
                     "backlog; customers who find no stock at the retailer are lost (2 per lost sale); shipments and "
                     "production take time to arrive. Price 12, raw material 4 per unit, holding 0.5 at the retailer "
                     "and 0.3 elsewhere."}


@family("plan_capacity", engine="planning", category="8.4", item="8.4.4", split="val",
        doc="Lumpy, irreversible capacity investment before the demand scenario (boom, bust or a late jump) is known; "
            "capacity arrives after a hidden build lag, costs upkeep, and can be retired for a salvage value.",
        distinct_because="real options: discrete irreversible investments with a build lag against a hidden demand "
                         "growth scenario, with fully observed demand (sales plus turned-away customers)",
        axes={"horizon": 60, "resources": 5}, objective="maximize_net", **hooks("capacity"))
def plan_capacity(rng, d):
    H = int(d["horizon"])
    d0 = u(rng, 15.0, 25.0)
    path = ["boom", "bust", "late"][int(rng.integers(3))]
    return {"engine": "planning", "scenario": "capacity", "horizon": H, "path": path, "d0": d0,
            "growth": u(rng, 1.0, 1.6), "t_jump": int(rng.integers(H // 4, H // 2)),
            "build_lag": int(rng.integers(2, 5)), "unit_block": float(d["resources"]), "cost_small": 60.0,
            "cost_large": 45.0, "salvage": 15.0, "upkeep": 1.0, "margin": 6.0, "cap0": float(round(1.05 * d0)),
            "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise margin on sales minus investment and upkeep, plus "
                                                          "salvage."},
            "brief": f"You own production capacity. Each period: build a small block ({d['resources']} units, 60 per "
                     f"unit), a large block (3 x {d['resources']} units, 45 per unit), retire a block (salvage 15 per "
                     "unit), or do nothing. New capacity comes on line after a build lag. Each unit of capacity "
                     "costs 1 per period; each unit sold earns a margin of 6. Customers you cannot serve are "
                     "turned away."}


@family("plan_complements", engine="planning", category="8.4", item="8.4.5", split="train",
        doc="Products made from several inputs in hidden fixed proportions; inputs are ordered with lead times and "
            "machine hours are allocated to products each period.",
        distinct_because="Leontief production: output needs every input of a hidden recipe at once; inputs are "
                         "shared between products, ordered separately (levers) and hours are allocated (allocate)",
        axes={"n_variables": 4, "action_complexity": 3, "resources": 12, "horizon": 50}, objective="maximize_net",
        **hooks("complements"))
def plan_complements(rng, d):
    R, P = max(3, int(d["n_variables"])), max(2, int(d["action_complexity"]))
    names = neutral_names(rng, R + P + R)
    rec = np.zeros((R, P), dtype=int)
    for j in range(P):
        k = int(rng.integers(2, min(3, R) + 1))
        for i in rng.choice(R, size=k, replace=False):
            rec[i, j] = int(rng.integers(1, 3))
    for i in range(R):
        if rec[i].sum() == 0:
            rec[i, int(rng.integers(P))] = 1
    res_cost = [u(rng, 1.0, 3.0) for _ in range(R)]
    lead = [int(rng.integers(1, 4)) for _ in range(R)]
    price = [float(round(sum(rec[i, j] * res_cost[i] for i in range(R)) * u(rng, 1.8, 2.6) + 2.0, 2))
             for j in range(P)]
    return {"engine": "planning", "scenario": "complements", "horizon": int(d["horizon"]), "res_names": names[:R],
            "prod_names": names[R:R + P], "order_names": names[R + P:], "recipe": rec.tolist(),
            "lead_time": lead, "res_cost": [round(x, 2) for x in res_cost],
            "price": price, "dem": [u(rng, 2.0, 6.0) for _ in range(P)], "hours": float(d["resources"]), "rate": 1.0,
            "hold_res": 0.05, "hold_fg": 0.3, "max_order": 40.0, "init_res": [12.0] * R, "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise product sales minus input purchases and holding "
                                                          "costs."},
            "brief": f"You make {P} products ({', '.join(names[R:R + P])}) from {R} inputs "
                     f"({', '.join(names[:R])}). Each product needs some of the inputs in fixed amounts you do not "
                     f"know. Order inputs with levers {', '.join(names[R + P:])} (unit costs "
                     f"{', '.join(f'{x:.2f}' for x in res_cost)}; lead times {', '.join(map(str, lead))} periods); "
                     f"allocate "
                     f"{d['resources']} machine hours per period over the products (one unit per hour, standing plan). "
                     f"Product prices {', '.join(f'{p:g}' for p in price)}; holding 0.05 per input and 0.3 per product "
                     "unit per period."}


@family("plan_portfolio", engine="planning", category="8.4", item="8.4.6", split="train",
        doc="Staff allocated over projects with hidden work content and productivity, diminishing returns to "
            "staffing, and duds revealed at a midpoint milestone; completed projects pay, less after their deadline.",
        distinct_because="a portfolio of competing projects: concave progress in staff, threshold completion "
                         "payments with deadlines, noisy progress reports, costly exact audits, abandoned duds",
        axes={"action_complexity": 5, "resources": 10, "horizon": 50, "observation_noise": 0.03,
              "experiment_cost": 30.0}, objective="maximize_net", **hooks("portfolio"))
def plan_portfolio(rng, d):
    N = int(d["action_complexity"])
    H = int(d["horizon"])
    names = neutral_names(rng, N)
    dud = [False] * N
    for j in rng.choice(N, size=max(1, N // 4), replace=False):
        dud[int(j)] = True
    return {"engine": "planning", "scenario": "portfolio", "horizon": H, "proj_names": names,
            "value": [float(round(u(rng, 2000, 6000), -1)) for _ in range(N)],
            "deadline": [int(rng.integers(H // 5, 3 * H // 4)) for _ in range(N)], "work": [u(rng, 30.0, 90.0) for _ in range(N)],
            "prod": [u(rng, 0.6, 1.4) for _ in range(N)], "beta": u(rng, 0.55, 0.75), "dud": dud,
            "staff": float(d["resources"]), "wage": 8.0, "late_window": 15.0, "report_sd": float(d["observation_noise"]),
            "audit_cost": float(d["experiment_cost"]), "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise payments for completed projects minus wages."},
            "brief": f"{N} projects compete for {d['resources']} staff (allocate; standing plan). Staff assigned cost "
                     "8 each per period. A project pays its value when its work is done, less for every period after "
                     "its deadline (nothing after 15 late periods). Progress reports are noisy; an audit (query) "
                     "gives the exact work done. Some projects turn out to be unworkable halfway and are abandoned."}


@family("team_assembly", engine="planning", category="8.9", item="8.9.1", split="train",
        doc="Two participants with private, complementary production costs make the two components of one product; "
            "output needs both components and is sold against fluctuating demand; revenue is shared.",
        distinct_because="complementary production capabilities: each participant's own costs are private, each is "
                         "cheap at a different component, and only matched pairs of components sell",
        axes={"resources": 10, "horizon": 40}, objective="maximize_net", participants=("teammate:shared_revenue",),
        **hooks("assembly"))
def team_assembly(rng, d):
    names = neutral_names(rng, 2)
    lo = [u(rng, 2.0, 4.0), u(rng, 2.0, 4.0)]
    hi = [u(rng, 9.0, 13.0), u(rng, 9.0, 13.0)]
    if rng.random() < 0.5:
        cost = [[lo[0], hi[0]], [hi[1], lo[1]]]
    else:
        cost = [[hi[0], lo[0]], [lo[1], hi[1]]]
    cap = float(d["resources"])
    return {"engine": "planning", "scenario": "assembly", "horizon": int(d["horizon"]), "agents": ["a0", "a1"],
            "comp_names": names, "make_names": [f"make_{n}" for n in names],
            "unit_cost": [[round(x, 2) for x in r] for r in cost], "cap": cap, "unit_price": 20.0, "hold": 0.5,
            "mu_mean": u(rng, 0.6, 0.9) * cap, "mu_amp": u(rng, 0.2, 0.5), "mu_phase": u(rng, 0.0, 12.0),
            "mu_period": 12.0, "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise the team's combined sales minus production and "
                                                          "holding costs."},
            "brief": f"You and a partner each can make components {names[0]} and {names[1]} (up to {cap:g} units per "
                     "period in total; standing rates). One unit of each makes a product that sells for 20 when a "
                     "customer orders it; revenue and holding costs (0.5 per component per period) are shared. Each "
                     "of you knows only your own unit costs; message your partner."}


@family("team_schedule", engine="planning", category="8.9", item="8.9.4", split="val",
        doc="Three participants with private job queues (values, due periods, estimated work) share two machines "
            "whose speed differs by participant; double bookings waste the period; joint plans reserve machines.",
        distinct_because="multi-agent scheduling on shared machines: private job information, conflicts from "
                         "simultaneous bookings, hidden machine speeds and true work content, joint reservations",
        axes={"horizon": 40, "action_complexity": 4}, objective="maximize_net",
        participants=("teammate:shared_machines",), **hooks("schedule"))
def team_schedule(rng, d):
    H = int(d["horizon"])
    J = int(d["action_complexity"])
    names = neutral_names(rng, 2 + 3 * J)
    jobs, true_time = [], []
    for i in range(3):
        js, tt = [], []
        for j in range(J):
            est = int(rng.integers(3, 7))
            rel = int(rng.integers(0, H // 2))
            js.append({"name": names[2 + i * J + j], "value": float(round(u(rng, 300, 900), -1)), "est": est,
                       "release": rel, "deadline": rel + est + int(rng.integers(2, 9))})
            tt.append(est * u(rng, 0.7, 1.4))
        jobs.append(js)
        true_time.append(tt)
    return {"engine": "planning", "scenario": "schedule", "horizon": H, "agents": ["a0", "a1", "a2"],
            "machine_names": names[:2], "jobs": jobs, "true_time": true_time,
            "speed": [[f, s_] if rng.random() < 0.5 else [s_, f] for f, s_ in
                      [(u(rng, 1.1, 1.5), u(rng, 0.5, 0.85)) for _ in range(3)]], "conflict_cost": 20.0, "work_sd": 0.15,
            "machine_cost": 10.0, "late_window": 8.0, "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise the value of the team's finished jobs minus machine "
                                                          "and conflict costs."},
            "brief": f"Three of you share machines {names[0]} and {names[1]}. Each period you may book one machine "
                     "for your next job (the released, unfinished one due first). A machine booked by two of you does "
                     "nothing that period (20 each). A joint plan (coordinate, all three) reserves machines. Running "
                     "a machine costs 10 per period; a job pays its value when its work is done, less for every "
                     "period late (nothing after 8)."}


# ====================================================================== strategic engine: 8.8 strategic games
@family("strat_price_war", engine="strategic", category="8.8", item="8.8.1", split="train",
        doc="Repeated price competition against two rivals: a propensity learner that credits undercutting with the "
            "profit of that period and the next (so retaliation teaches it to hold), and a price follower or a "
            "trigger strategy.",
        distinct_because="logit price competition with adaptive rival policies (propensity learning with two-period "
                         "credit, following, triggers): sustaining high prices needs a reactive strategy",
        axes={"opponent_capability": 0.3, "horizon": 50}, objective="maximize_net",
        participants=("rival:propensity_learner", "rival:follower_or_trigger"), **hooks("price_war"))
def strat_price_war(rng, d):
    names = neutral_names(rng, 3)
    kind1 = "follower" if rng.random() < 0.5 else "trigger"
    return {"engine": "strategic", "scenario": "price_war", "horizon": int(d["horizon"]), "price_name": names[0],
            "rival_names": names[1:], "cost": 10.0, "rival_cost": [u(rng, 8.5, 11.0), u(rng, 8.5, 11.0)],
            "rival_kind": ["propensity", kind1], "hi": [u(rng, 15.0, 18.0), u(rng, 15.0, 18.0)],
            "cut": u(rng, 0.6, 1.5), "follow": u(rng, 0.3, 0.7), "gap": u(rng, 0.3, 1.0), "trigger": u(rng, 11.5, 13.5),
            "punish": int(rng.integers(3, 6)), "eta": float(d["opponent_capability"]), "temp": u(rng, 0.03, 0.06),
            "v0": [100.0, 100.0], "appeal": 15.1, "sens": 1.0, "market": 100.0, "market_sd": 0.1, "p_min": 8.0,
            "p_max": 26.0, "p_default": 12.0, "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise your profit (price minus unit cost 10, times "
                                                          "units sold)."},
            "brief": f"You set your price (lever {names[0]}) every period against rivals {names[1]} and {names[2]}. "
                     "Customers choose between the three products and not buying (logit demand); you see the rivals' "
                     "posted prices and your own sales. Rivals adapt to what you do."}


@family("strat_entry", engine="strategic", category="8.8", item="8.8.2", split="train",
        doc="Entry into several markets held by incumbents of hidden type: tough ones fight, soft ones accommodate, "
            "and some fight until the entrant has persisted; pre-entry prices signal incumbents' costs.",
        distinct_because="discrete irreversible entry and exit decisions per market against incumbent response "
                         "policies (fight, accommodate, fight-until-persistent) with noisy cost signalling",
        axes={"graph_size": 5, "horizon": 40, "observation_noise": 0.8}, objective="maximize_net",
        participants=("incumbent:tough", "incumbent:soft", "incumbent:reputation_learner"), **hooks("entry"))
def strat_entry(rng, d):
    M = max(2, int(d["graph_size"]))
    names = neutral_names(rng, M)
    nt = M // 2
    kinds = ["tough"] * nt + ["soft"] * (M - nt)
    if M - nt >= 2 and rng.random() < 0.6:
        kinds[nt + 1] = "reputation"
    kinds = [kinds[int(i)] for i in rng.permutation(M)]
    inc_cost = [u(rng, 5.0, 6.5) if k == "tough" else (u(rng, 9.5, 12.0) if k == "soft" else u(rng, 7.5, 9.0))
                for k in kinds]
    op = [u(rng, 50.0, 70.0) for _ in range(M)]
    return {"engine": "strategic", "scenario": "entry", "horizon": int(d["horizon"]), "market_names": names,
            "inc_kind": kinds, "inc_cost": inc_cost, "markup": 3.0, "price_sd": float(d["observation_noise"]),
            "patience": [int(rng.integers(3, 7)) for _ in range(M)], "p_fight_tough": 0.85, "p_fight_soft": 0.1,
            "rev_acc": [x * u(rng, 1.7, 2.0) for x in op], "op_cost": op,
            "rev_fight": [x * u(rng, 0.1, 0.3) for x in op], "entry_cost": 150.0, "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise your profit across markets minus entry costs."},
            "brief": f"{M} markets ({', '.join(names)}) each have an incumbent. Entering a market costs 150; once in, "
                     "you earn or lose money each period depending on how the incumbent responds (a price war "
                     "shows as a lower incumbent price). You can exit (no refund) and re-enter. Incumbents' posted "
                     "prices are visible before you enter."}


@family("strat_bargain", engine="strategic", category="8.8", item="8.8.3", split="val",
        doc="Repeated three-issue negotiations with counterparts of private issue weights and reservation levels; "
            "held-out negotiator families: Boulware (time-dependent) and tit-for-tat (concedes as you concede).",
        distinct_because="alternating-offer multi-issue bargaining with private preferences (integrative trade-offs) "
                         "against held-out counterpart policy families; offers, acceptance and walking away",
        axes={"horizon": 48, "action_complexity": 3}, objective="maximize_net",
        participants=("counterpart:boulware", "counterpart:tit_for_tat"), **hooks("bargain"))
def strat_bargain(rng, d):
    nI = max(2, int(d["action_complexity"]))
    names = neutral_names(rng, nI + 3)

    def simplex(lo, hi):
        x = np.asarray([u(rng, lo, hi) for _ in range(nI)])
        return (x / x.sum()).tolist()
    my_w = simplex(0.15, 0.6)
    cps = []
    for _ in range(3):
        while True:
            v = simplex(0.1, 0.7)
            if int(np.argmax(v)) != int(np.argmax(my_w)):
                break
        cps.append(v)
    kinds = ["boulware", "tit_for_tat", ["boulware", "tit_for_tat"][int(rng.integers(2))]]
    kinds = [kinds[int(i)] for i in rng.permutation(3)]
    sched = [int(x) for x in rng.permutation(np.tile(np.arange(3), 6))]
    return {"engine": "strategic", "scenario": "bargain", "horizon": int(d["horizon"]), "issue_names": names[:nI],
            "cp_names": names[nI:], "my_w": [round(x, 4) for x in my_w], "cp_v": cps,
            "cp_r0": [u(rng, 0.3, 0.5) for _ in range(3)], "cp_kind": kinds, "cp_e": [u(rng, 0.2, 0.5) for _ in range(3)],
            "cp_zeta": [u(rng, 0.6, 1.0) for _ in range(3)], "asp0": 0.95, "rounds": 6, "deal_value": 1000.0,
            "round_cost": 10.0, "accept_sd": 0.02, "schedule": sched, "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise the value of your agreements (1000 times your "
                                                          "utility of the terms) minus 10 per round."},
            "brief": f"You negotiate deals one after another with counterparts {', '.join(names[nI:])} over issues "
                     f"{', '.join(names[:nI])}. Terms are your share of each issue (levers, 0-1). Each round the "
                     "counterpart has an offer on the table: accept it, propose your current terms, or walk away. "
                     "A deal not closed in 6 rounds lapses. Counterparts value the issues differently from you and "
                     "from each other."}


@family("strat_blotto", engine="strategic", category="8.8", item="8.8.4", split="train",
        doc="Contests for prizes at several sites (Tullock contest success) against an opponent with private "
            "valuations and budget that best-responds to a moving average of your past allocations.",
        distinct_because="resource competition by allocation: contest success functions over sites against a "
                         "fictitious-play opponent; static allocations are countered, the opponent's lag can be used",
        axes={"graph_size": 5, "resources": 10, "opponent_capability": 0.85, "horizon": 40},
        objective="maximize_net", participants=("opponent:fictitious_play",), **hooks("blotto"))
def strat_blotto(rng, d):
    K = int(d["graph_size"])
    names = neutral_names(rng, K)
    return {"engine": "strategic", "scenario": "blotto", "horizon": int(d["horizon"]), "site_names": names,
            "prize": [float(round(u(rng, 50.0, 140.0))) for _ in range(K)], "opp_value": [u(rng, 0.3, 1.9) for _ in range(K)],
            "budget": float(d["resources"]), "opp_budget": u(rng, 1.1, 1.6) * float(d["resources"]), "budget_sd": 0.05,
            "decisive": u(rng, 1.5, 3.0), "memory": min(0.95, float(d["opponent_capability"])), "unit_cost": 12.0,
            "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise the prizes you win minus 3 per unit of effort."},
            "brief": f"Each period you and an opponent put effort into {K} sites ({', '.join(names)}); your share of "
                     "a site's prize grows with your effort relative to the opponent's. You see the opponent's "
                     f"efforts after each contest. Effort costs 12 per unit; at most {d['resources']} units per period."}


@family("strat_interdiction", engine="strategic", category="8.8", item="8.8.5", split="train",
        doc="Inspection capacity over routes against an attacker that learns each route's value from its own "
            "outcomes and substitutes away from inspected routes.",
        distinct_because="strategic substitution: the attacker's route choice responds to your inspection through "
                         "experience-based learning; damages differ by route; catches and breaches are observed",
        axes={"graph_size": 4, "resources": 8, "opponent_capability": 0.3, "horizon": 50}, objective="maximize_net",
        participants=("attacker:experience_learner",), **hooks("interdiction"))
def strat_interdiction(rng, d):
    N = int(d["graph_size"])
    names = neutral_names(rng, N)
    return {"engine": "strategic", "scenario": "interdiction", "horizon": int(d["horizon"]), "route_names": names,
            "damage": [float(round(u(rng, 40.0, 120.0))) for _ in range(N)], "att_util": [u(rng, 0.5, 1.5) for _ in range(N)],
            "penalty": u(rng, 1.0, 2.0), "kappa": u(rng, 0.25, 0.45), "eta": float(d["opponent_capability"]),
            "att_beta": u(rng, 2.0, 4.0), "attempts": 5, "bounty": 30.0, "inspectors": float(d["resources"]),
            "deploy_cost": 1.0, "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise bounties for catches minus damage from breaches "
                                                          "and deployment costs."},
            "brief": f"An attacker makes 5 attempts per period over routes {', '.join(names)}. You deploy up to "
                     f"{d['resources']} inspectors (allocate; standing deployment, 1 per inspector per period); more "
                     "inspectors on a route make a catch there more likely. A catch pays 30; a breach costs that "
                     "route's damage. The attacker learns from what happens to it."}


@family("strat_learning_rival", engine="strategic", category="8.8", item="8.8.6", split="val",
        doc="A repeated three-action game against a regret-matching opponent with private payoffs and forgetting, "
            "whose strategy evolves with what you play (held-out opponent family).",
        distinct_because="an opponent whose mixed strategy is the output of its own learning (regret matching): "
                         "constant play is learned and countered; its adaptation can be predicted and led",
        axes={"opponent_capability": 0.5, "horizon": 60}, objective="maximize_net",
        participants=("opponent:regret_matching",), **hooks("learning"))
def strat_learning_rival(rng, d):
    names = neutral_names(rng, 6)
    A, B = [], []
    for i in range(3):
        row_a, row_b = [], []
        for j in range(3):
            if j == i:
                row_a.append(5.0)
                row_b.append(3.0)
            elif j == (i + 1) % 3:                      # your option i beats their option i+1
                row_a.append(float(round(u(rng, 8.0, 12.0), 1)))
                row_b.append(u(rng, 0.0, 1.0))
            else:                                       # their option i+2 beats your option i
                row_a.append(float(round(u(rng, 0.0, 2.0), 1)))
                row_b.append(u(rng, 6.0, 10.0))
        A.append(row_a)
        B.append(row_b)
    return {"engine": "strategic", "scenario": "learning", "horizon": int(d["horizon"]), "option_names": names[:3],
            "opp_names": names[3:], "A": A, "B": B, "forget": max(0.3, 1.0 - 0.15 * float(d["opponent_capability"])),
            "scale": 10.0, "fee": 20.0, "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise your payoff minus the fee of 20 per period."},
            "brief": f"Each period you play one of {', '.join(names[:3])} (decision 'play'; it stays until you "
                     f"change it) and an opponent plays one of {', '.join(names[3:])}. You see what it played."}


# ====================================================================== strategic engine: 8.9 cooperation
@family("team_profile", engine="strategic", category="8.9", item="8.9.2", split="train",
        doc="Three participants each see one attribute of every option in a slate; an option's value weights the "
            "three attributes with hidden weights; only a joint commitment of all three pays.",
        distinct_because="a hidden profile: information is split by attribute across three participants, value is "
                         "a hidden weighting of all attributes, decisions need unanimous joint commitment",
        axes={"horizon": 45, "action_complexity": 4}, objective="maximize_net",
        participants=("teammate:private_attribute",), **hooks("profile"))
def team_profile(rng, d):
    K = max(2, int(d["action_complexity"]))
    names = neutral_names(rng, K)
    # one attribute weighs most, but every attribute weighs enough that no participant's view alone ranks the
    # options well (interdependence, 8.9): the two lesser weights are at least 0.4 before the boost
    w = np.asarray([u(rng, 0.4, 0.8) for _ in range(3)])
    w[int(rng.integers(3))] += 0.8
    w = w / np.linalg.norm(w)
    return {"engine": "strategic", "scenario": "profile", "horizon": int(d["horizon"]), "agents": ["a0", "a1", "a2"],
            "slot_names": names, "w": w.tolist(), "value_sd": 0.4, "scale": 400.0, "decision_cost": 20.0, "period": 3,
            "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise the team's total payoff from joint choices."},
            "brief": f"Every 3 periods a new set of {K} options ({', '.join(names)}) appears. Each of you sees one "
                     "attribute of every option; an option's payoff depends on all three attributes. The team is "
                     "paid only when all three commit to the same option (coordinate); one choice per set, 20 per "
                     "decision. Message each other."}


@family("team_commons", engine="strategic", category="8.9", item="8.9.3", split="train",
        doc="Three participants harvest renewable patches with hidden capacities, regrowth and tipping points "
            "(depensation); boats on the same patch crowd each other; surveys are noisy.",
        distinct_because="shared renewable stocks with logistic regrowth and an Allee threshold, crowding between "
                         "harvesters, ground choice (discrete) plus effort (lever) per participant",
        axes={"graph_size": 3, "horizon": 50, "observation_noise": 0.1}, objective="maximize_net",
        participants=("teammate:harvester",), **hooks("commons"))
def team_commons(rng, d):
    P = int(d["graph_size"])
    names = neutral_names(rng, P)
    K = [u(rng, 800.0, 1500.0) for _ in range(P)]
    return {"engine": "strategic", "scenario": "commons", "horizon": int(d["horizon"]), "agents": ["a0", "a1", "a2"],
            "patch_names": names, "Kp": K, "rp": [u(rng, 0.25, 0.5) for _ in range(P)],
            "allee": [k * u(rng, 0.15, 0.4) for k in K], "s0_frac": [u(rng, 0.7, 0.95) for _ in range(P)],
            "crowd": u(rng, 0.6, 1.2), "catchability": 0.02, "e_max": 10.0, "unit_price": 5.0, "effort_cost": 6.0,
            "shock_sd": 0.15, "survey_sd": float(d["observation_noise"]), "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise the team's landings value minus effort costs."},
            "brief": f"Three of you fish {P} grounds ({', '.join(names)}). Each period choose a ground (or rest) and "
                     "an effort (0-10, 6 per unit). Each unit of effort catches about 2% of the ground's stock; "
                     "boats on the same ground get in each other's way. Stocks regrow, but a stock fished too low may not "
                     "recover. You see your catch and a noisy survey of your ground; landings sell for 5."}


@family("team_discovery", engine="strategic", category="8.9", item="8.9.5", split="train",
        doc="Two participants each own ingredients of hidden types; a joint test of one ingredient from each yields "
            "only if their types match; private assays label each side's ingredients in that side's own labels.",
        distinct_because="collaborative discovery: the matching between two privately labelled partitions can only "
                         "be found by joint experiments; assays, messages and joint tests are complementary",
        axes={"action_complexity": 5, "hidden_state_complexity": 3, "horizon": 40, "experiment_cost": 15.0},
        objective="maximize_net", participants=("teammate:ingredient_owner",), **hooks("discovery"))
def team_discovery(rng, d):
    n = int(d["action_complexity"])
    T = max(2, int(d["hidden_state_complexity"]))
    names = neutral_names(rng, 2 * n)

    def types():
        t = list(range(T)) + [int(x) for x in rng.integers(0, T, size=max(0, n - T))]
        return [t[int(i)] for i in rng.permutation(len(t))][:n]
    labels = [f"class_{k + 1}" for k in range(T)]
    return {"engine": "strategic", "scenario": "discovery", "horizon": int(d["horizon"]), "agents": ["a0", "a1"],
            "ingr_a": names[:n], "ingr_b": names[n:], "typeA": types(), "typeB": types(),
            "qualA": [u(rng, 0.6, 1.5) for _ in range(n)], "qualB": [u(rng, 0.6, 1.5) for _ in range(n)],
            "labelA": [labels[int(i)] for i in rng.permutation(T)], "labelB": [labels[int(i)] for i in rng.permutation(T)],
            "base_yield": 100.0, "yield_sd": 3.0, "unit_price": 2.0, "try_cost": 40.0,
            "assay_cost": float(d["experiment_cost"]), "purity_sd": 0.25, "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise the team's yield revenue minus test and assay "
                                                          "costs."},
            "brief": f"You and a partner each own {n} ingredients. A joint test (coordinate, both) of one ingredient "
                     "from each costs 40 and yields product (2 per unit) only if the two are compatible; a compatible "
                     "pair can be run again every period. An assay (query) gives the class of one of your own "
                     "ingredients in your own labelling; your partner's labels are not yours. Purity readings hint "
                     "at quality."}


@family("team_formation", engine="strategic", category="8.9", item="8.9.6", split="train",
        doc="A lead pays workers with hidden skills, reservation pay and reliability to form a team for each task; "
            "a task succeeds only if every required skill turns up; under-paid workers become less committed.",
        distinct_because="team formation: complementary skill coverage (all required skills at once), hidden "
                         "acceptance thresholds and turnout that depends on how workers are paid; claimed skills are "
                         "unreliable",
        axes={"action_complexity": 6, "resources": 100, "horizon": 50}, objective="maximize_net",
        participants=("worker:reservation_and_commitment",), **hooks("formation"))
def team_formation(rng, d):
    W = max(3, int(d["action_complexity"]))
    S = 3
    names = neutral_names(rng, W + S)
    while True:
        skills = []
        for _ in range(W):
            k = int(rng.integers(1, 3))
            row = [0] * S
            for s in rng.choice(S, size=k, replace=False):
                row[int(s)] = 1
            skills.append(row)
        if all(sum(r[s] for r in skills) >= 2 for s in range(S)):
            break
    claims = []
    for row in skills:
        cl = [names[W + s] for s in range(S) if row[s]]
        if rng.random() < 0.35:
            extra = [names[W + s] for s in range(S) if not row[s]]
            if extra:
                cl.append(extra[int(rng.integers(len(extra)))])
        claims.append(sorted(cl))
    rel = [u(rng, 0.75, 0.98) for _ in range(W)]
    rel[int(rng.integers(W))] = u(rng, 0.35, 0.5)
    cfg = {"engine": "strategic", "scenario": "formation", "horizon": int(d["horizon"]), "worker_names": names[:W],
           "skill_names": names[W:], "skills": skills, "claims": claims,
           "reservation": [u(rng, 15.0, 35.0) for _ in range(W)], "reliability": rel, "budget": float(d["resources"]),
           "task_value": 150.0, "p_need": 0.5, "trust_step": 0.05, "endowment": 50_000,
           "objective": {"type": "maximize_net", "text": "Maximise task payments minus pay to workers who turn up."},
           "brief": f"Each period a task arrives that needs some of the skills {', '.join(names[W:])}; it pays if "
                    "every needed skill is present. Offer pay to workers (allocate; a worker accepts only enough "
                    "pay, and is paid only if it turns up). Workers list the skills they claim; claims are not "
                    "always right. How you pay affects how reliably workers turn up."}
    need = [1 if rng.random() < 0.5 else 0 for _ in range(S)]
    if not any(need):
        need[0] = 1
    cfg["task0"] = need + [150.0 * (0.5 + 0.5 * sum(need))]
    return cfg
