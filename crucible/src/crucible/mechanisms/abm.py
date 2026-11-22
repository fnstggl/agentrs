"""Agent-based market engine (`engine = "abm"`): a hidden mechanism made of individual customers and firms.

Nothing in the simulation is a causal graph. A market is a population of N heterogeneous customers held as numpy
arrays (income and a budget that is never overspent, willingness to pay, weights on price, experienced quality and
wait, a memory of their own experiences with every firm, an awareness set, tenure with their current firm, a
position in a generated social network), a set of firms (the controllable *seats* plus scripted incumbents that
adapt price and capacity, exit when unprofitable and are replaced by an entrant after a delay), and one tick loop:

    income -> offers -> awareness -> choice -> service (capacity) -> experience, reviews, complaints -> payment
    -> loyalty / lapse -> incumbents adapt (price, capacity, exit, entry) -> shared attention pool

`truth()` is a *description* of that loop as nodes and signed, lagged edges; `tests/test_mech_abm.py` checks the
description against the simulation by intervention with common random numbers.

Public surface. Every lever is a dial from 0 to 100 with an opaque name (`base.neutral_names`); the engine maps
each dial linearly onto the natural range of the quantity it controls (a price in cents, an effort in [0, 1], a
number of service slots, a spend in cents per tick, a reward in cents). Which way a dial runs is seeded (a dial may
have the top of its range at 0), so neither a name, a range nor a direction tells a participant which lever is
which. Distractor levers do nothing at all. Rivals' prices are shown on the seat's own price dial (same direction),
lagged; -1 marks a rival that is not in the market. Left alone a seat rests at a high price with no effort, no
outreach, the least capacity and no reward, where next to nothing happens.

Money. All money is integer-valued cents. A customer pays the posted price out of their budget or does not buy;
revenue of a firm is exactly the sum of what its customers paid. `state["totals"]` keeps the running totals
`[income credited to customers, revenue of all firms, revenue of seats, ticks]`, and at every tick
`income credited - revenue of all firms == sum of budgets` holds exactly. `score` is the seat's net (revenue minus
cost) per tick in every family.

State. Arrays are firm-major: `aware` is (firms, N), `mem_q` (options, N), `traits` (N_TRAITS, N); `cur` holds each
customer's current option (the number of options means "none"). Arrays only some rule sets read exist only in
their states (`mem_w` with rationing, `tried` with the gate or word of mouth, `happy` with word of mouth, `pool`
with the commons). The static traits and the social network are regenerated from seeds stored in the config;
the traits are also carried in the state, which is what `step` reads. `step` is pure: it draws one block of
uniforms per tick from the generator it is given, in a pattern that depends on the config only, so two runs that
differ in settings share every random number (which is what makes paired interventions exact).

Structural families (they differ in causal structure, not only in parameters)
------------------------------------------------------------------------------
For each family: what produces awareness -> what produces trial -> what produces retention -> what feeds back.

abm_utility (train)      outreach spend (private hazard per customer, forgotten at a constant rate) -> a private
                         random-utility comparison of known firms against a neutral prior -> the customer's own
                         remembered experience and a mild habit -> no social feedback; only incumbents' price
                         rule reacts. The network exists but carries nothing.
abm_wom (train)          satisfied current customers telling their network neighbours (outreach is weak; a
                         referral reward raises the telling rate and is paid per referred first purchase) -> the
                         same utility comparison with an optimistic referred prior -> remembered experience ->
                         reinforcing loop: quality -> satisfied customers -> awareness -> customers.
abm_congestion (train)   outreach -> utility comparison -> service is rationed by capacity; unserved customers
                         pay nothing, complain, lapse and remember the wait -> balancing loop: demand -> queue ->
                         wait memory and churn -> demand. Growing demand without capacity is the trap.
abm_trust_gate (val)     outreach -> a customer will not try a firm they have no experience of until its public
                         rating passes their personal threshold with enough reviews behind it (a few pioneers have
                         no threshold: cold start) -> strong habit (switching cost) -> reinforcing, thresholded
                         loop: reviews -> rating -> gate opens -> trial -> reviews.
abm_network_fx (test)    outreach -> value of a firm rises with the share of the customer's neighbours using it ->
                         lock-in through neighbours and habit -> reinforcing, path-dependent loop: installed base
                         -> value -> adoption. The market starts open; early penetration pricing can pay, late
                         price cuts do not.
abm_segments (test)      two segments in one market: segment A obeys the abm_utility rule (reachable by outreach,
                         deaf to neighbours), segment B obeys the abm_wom rule (reachable only through satisfied
                         neighbours, of either segment). Compositional.
abm_duopoly (train, 2)   the abm_utility rule with two seats fighting for the same customers.
abm_commons (val, 2)     both seats draw on one congestible stock of customer attention: outreach by either
                         depletes it, and a depleted stock lowers the effectiveness of everyone's outreach and how
                         often customers shop at all; the seats still compete for the same customers.
abm_complement (val, 2)  customers need a compatible bundle of one component from each seat: they must know both,
                         the bundle is worth its weaker component less a mismatch penalty, and they pay the sum of
                         both prices. A component alone is worth little.
abm_duopoly_fx (test, 2) abm_duopoly with the abm_network_fx value rule.

None of the behavioural rules is calibrated against real consumers: see `CALIBRATION`.
"""
from __future__ import annotations

import math
from itertools import chain
from types import SimpleNamespace

import numpy as np

from .base import Engine, Gauge, Lever, StepResult, mech_family, neutral_names, register_engine

# ====================================================================== scientific status of every rule
_HYP = "plausible hypothesis"
CALIBRATION: dict[str, dict[str, str]] = {
    "_scope": {"status": "not calibrated",
               "note": "No rule and no parameter in this engine is fitted to data on real consumers or firms. "
                       "Parameter values are chosen so that the generated tasks have interior optima and "
                       "distinguishable causal structures; they are not empirical estimates."},
    "choice": {"status": "inspired by a published model",
               "reference": "random-utility discrete choice (McFadden 1974, conditional logit)",
               "note": "argmax of value minus weighted price plus an individual taste shock over known firms and "
                       "an outside option; shocks here are normal, not extreme-value; weights are invented."},
    "budget": {"status": _HYP, "note": "income per tick accrues to a capped budget; a purchase needs the full "
                                       "price in the budget. Hard accounting constraint, not a behavioural estimate."},
    "outreach_awareness": {"status": "inspired by a published model",
                           "reference": "Bass 1969 diffusion (external-influence term)",
                           "note": "per-customer hazard proportional to spend with exponential saturation; "
                                   "the per-customer form and the constant forgetting rate are hypotheses."},
    "word_of_mouth": {"status": "inspired by a published model",
                      "reference": "Bass 1969 diffusion (imitation term)",
                      "note": "moved from an aggregate equation to network edges: only satisfied current customers "
                              "tell neighbours. The edge-level hazard and the referral-reward response are "
                              "hypotheses."},
    "experience_memory": {"status": _HYP, "note": "exponentially weighted memory of experienced quality; satisfied "
                                                  "when an experience meets a personal standard."},
    "habit": {"status": "inspired by a published model",
              "reference": "consumer switching costs (Klemperer 1987)",
              "note": "a utility bonus for the current firm that grows with tenure; magnitude invented."},
    "congestion": {"status": _HYP, "note": "hard capacity with random rationing; unserved customers complain, lapse "
                                           "and carry a decaying memory of the wait."},
    "reviews_rating": {"status": _HYP, "note": "a random share of served customers posts their experienced quality; "
                                               "the public rating is a decayed mean with a weak neutral prior."},
    "trust_gate": {"status": "inspired by a published model",
                   "reference": "threshold models of collective behaviour (Granovetter 1978)",
                   "note": "heterogeneous personal thresholds, applied here to a public rating and a review count "
                           "instead of a count of adopters; that transposition is a hypothesis."},
    "network_effects": {"status": "inspired by a published model",
                        "reference": "network externalities (Katz and Shapiro 1985); lock-in (Arthur 1989)",
                        "note": "value rises linearly with the share of a customer's own neighbours using the firm "
                                "(a local version); the linear form and its size are hypotheses."},
    "segments": {"status": _HYP, "note": "two segments that differ in which awareness channel reaches them."},
    "attention_commons": {"status": "inspired by a published model",
                          "reference": "the tragedy of the commons (Hardin 1968)",
                          "note": "customer attention as a renewable stock depleted by total outreach; that "
                                  "attention behaves like a common-pool resource is a hypothesis."},
    "complements": {"status": "inspired by a published model",
                    "reference": "complementary monopoly (Cournot 1838)",
                    "note": "customers pay the sum of two component prices; the weakest-link quality with a "
                            "mismatch penalty is a hypothesis."},
    "incumbent_price_rule": {"status": _HYP, "note": "rule of thumb: cut price when share falls below its slow "
                                                     "moving average, drift back to a reference price otherwise."},
    "incumbent_capacity_rule": {"status": _HYP, "note": "add capacity after turning customers away, shed it when "
                                                        "utilisation is low."},
    "exit_entry": {"status": _HYP, "note": "exit when smoothed profit stays below a loss threshold; a replacement "
                                           "with freshly drawn price and quality enters after a fixed delay."},
    "social_network": {"status": "inspired by a published model",
                       "reference": "ring with rewiring (Watts and Strogatz 1998); preferential attachment "
                                    "(Barabasi and Albert 1999); community blocks (stochastic block model, "
                                    "Holland, Laskey and Leinhardt 1983)",
                       "note": "generators only; sizes and densities are not matched to any observed network."},
}

# ====================================================================== layout of the state arrays
# rows of state["traits"] (static, one column per customer)
(T_INC, T_BCAP, T_WTP, T_WP, T_WQ, T_WW, T_HAB, T_NEED, T_STD, T_RECV, T_SUS, T_NFX, T_GTH, T_GM,
 T_SEG, T_VA, T_VB, T_WPN, T_HABR, T_STDM, T_NFXD) = range(21)
N_TRAITS = 21
# The last six rows are the forms the tick loop reads, derived once from the rows before them: value of an option
# of believed quality q is value_base + value_slope * q (in reference prices), price_weight_n multiplies a price in
# cents, habit_rate is habit per tick of tenure, complaint_level is standard minus the complaint margin, and
# network_per_neighbour is the network weight divided by the customer's degree.
TRAIT_ROWS = ("income", "budget_cap", "wtp", "w_price", "w_quality", "w_wait", "habit", "need", "standard",
              "receptivity", "susceptibility", "network_weight", "gate_rating", "gate_reviews", "segment",
              "value_base", "value_slope", "price_weight_n", "habit_rate", "complaint_level",
              "network_per_neighbour")
# rows of state["firm"] (one column per firm; seats first, then incumbents)
(R_PRICE, R_PREF, R_QUAL, R_CAP, R_REACH, R_ALIVE, R_RS, R_RN, R_SHARE, R_PEMA, R_TIMER, R_AGE) = range(12)
N_FROWS = 12
FIRM_ROWS = ("price", "ref_price", "quality", "capacity", "reach", "alive", "review_sum", "review_n", "share_ema",
             "profit_ema", "reentry_timer", "age")
_NEG = float("-inf")
_U16 = 1.0 / 65536.0

_BASE_RULES = dict(
    sigma=0.25,        # sd of the individual taste shock (in units of the reference price)
    qref=0.5,          # quality at which a customer's value equals their willingness to pay
    q_prior=0.5,       # believed quality of a firm never tried
    alpha=0.4,         # weight of a new experience in memory
    sx=0.10,           # sd of experienced quality around the firm's level
    phi=0.04,          # per-tick probability of forgetting a firm one is not a customer of
    org=0.002,         # per-tick hazard of stumbling on a firm with no outreach at all
    omega=0.08,        # outreach unit: hazard = receptivity * spend / (N * p0 * omega)
    wom=False, beta0=0.03, ref_rho=2.5, ref_half=0.3,
    congest=False, alpha_w=0.5, w_decay=0.97, pc_wait=0.5,
    gate=False,
    nfx=False,
    bundle=False, solo_scale=0.35, mismatch=0.5,
    commons=False, pool_regen=0.10, pool_deplete=0.05, pool_need=0.5, pool_omega=0.02,
    rho=0.3, rating_decay=0.98, rating_n0=3.0, rating_r0=0.5,
    pc=0.4, margin=0.12,
    delta=0.02,        # per-tick probability that an idle customer drops their current firm
    tau=8.0,           # tenure (ticks) at which habit is fully formed
    price_lag=1, vol_noise=0.08,
)
_BASE_POP = dict(wtp_med=1.5, wtp_sig=0.30, inc_med=1.0, inc_sig=0.35, save_ticks=5.0, endow_ticks=2.0,
                 need_lo=0.45, need_hi=0.85,
                 wp_sig=0.2, wq_scale=0.7, ww_scale=0.0, h_scale=0.15, std_mu=0.5, std_sd=0.08, recv_scale=1.0,
                 sus_scale=0.0, nfx_scale=0.0, pioneer=1.0, th_lo=0.0, th_hi=0.0, m_lo=0.0, m_hi=0.0,
                 segments=[{"name": "all", "share": 1.0}])
_BASE_COST = dict(unit=0.60, quality=0.10, capacity=0.05, overhead=0.0, unit_inc=0.60)   # in p0 (quality: N * p0)
_BASE_SEAT = dict(price_lo=0.3, price_hi=2.8, price_default=1.6, q_lo=0.2, q_hi=0.9, q_gamma=0.5, cap_lo=0.02,
                  cap_hi=0.6, out_hi=0.08, ref_hi=2.0, aware0=0.02)
_BASE_ADAPT = dict(eta_loss=0.08, eta_gain=0.03, kappa=0.05, share_rate=0.03, floor=1.05, ceil=2.2, cap_up=0.5,
                   cap_down=0.03, util_lo=0.7, profit_rate=0.1, exit_frac=0.25, grace=20, delay=20,
                   entrant_price=(0.95, 1.35), entrant_quality=(0.40, 0.75), launch_aware=0.2)
_BASE_INC = dict(price=(1.0, 1.35), quality=(0.45, 0.65), cap=(0.20, 0.30), reach=(0.08, 0.20), fixed=0.02,
                 aware0=(0.60, 0.85), attach=0.7)

# What each family kind switches on. `net` lists the generators the seed may pick; `noop` the range of distractors.
_KINDS: dict[str, dict] = {
    "utility": dict(seats=1, levers=("price", "quality", "outreach"), noop=(1, 2), net=("ring", "pa", "blocks"),
                    n_inc=(2, 3), horizon=120, pop=dict(wq_scale=0.5),
                    motifs=("outreach_awareness", "private_utility_choice", "experience_memory",
                            "adaptive_incumbents")),
    "wom": dict(seats=1, levers=("price", "quality", "outreach", "referral"), noop=(0, 1), net=("pa",),
                n_inc=(2, 3), horizon=160,
                rules=dict(wom=True, q_prior=0.6), pop=dict(recv_scale=0.03, sus_scale=1.0),
                seat=dict(aware0=0.03),
                motifs=("network_word_of_mouth", "satisfaction_referral_loop", "weak_outreach",
                        "adaptive_incumbents")),
    "congestion": dict(seats=1, levers=("price", "quality", "capacity", "outreach"), noop=(0, 1), net=("ring",),
                       n_inc=(2, 3), horizon=120, rules=dict(congest=True), pop=dict(ww_scale=1.0),
                       motifs=("outreach_awareness", "capacity_rationing", "wait_churn_loop",
                               "adaptive_incumbents")),
    "trust_gate": dict(seats=1, levers=("price", "quality", "outreach"), noop=(1, 1), net=("blocks",),
                       n_inc=(2, 3), horizon=160, rules=dict(gate=True, delta=0.03),
                       pop=dict(h_scale=0.45, pioneer=0.06, th_lo=0.50, th_hi=0.72, m_lo=5.0, m_hi=60.0),
                       motifs=("outreach_awareness", "rating_threshold_gate", "cold_start", "switching_cost",
                               "adaptive_incumbents")),
    "network_fx": dict(seats=1, levers=("price", "quality", "outreach"), noop=(1, 1), net=("ring",),
                       n_inc=(2, 2), horizon=160, rules=dict(nfx=True),
                       pop=dict(nfx_scale=1.2, wtp_med=1.0),
                       inc=dict(aware0=(0.10, 0.20), attach=0.5, reach=(0.10, 0.20), price=(0.9, 1.1)),
                       motifs=("outreach_awareness", "local_network_effects", "lock_in", "path_dependence",
                               "adaptive_incumbents")),
    "segments": dict(seats=1, levers=("price", "quality", "outreach", "referral"), noop=(0, 1), net=("blocks",),
                     n_inc=(2, 3), horizon=160, rules=dict(wom=True),
                     pop=dict(segments=[{"name": "A_utility", "share": 0.5, "recv": 1.0, "sus": 0.0},
                                        {"name": "B_wom", "share": 0.5, "recv": 0.03, "sus": 1.0}],
                              recv_scale=1.0, sus_scale=1.0),
                     motifs=("outreach_awareness", "network_word_of_mouth", "two_segments", "adaptive_incumbents")),
    "duopoly": dict(seats=2, levers=("price", "quality", "outreach"), noop=(0, 1), net=("ring", "pa"),
                    n_inc=(1, 2), horizon=120, pop=dict(wq_scale=0.5),
                    motifs=("outreach_awareness", "private_utility_choice", "two_seat_rivalry")),
    "commons": dict(seats=2, levers=("price", "quality", "outreach"), noop=(0, 1), net=("pa",), n_inc=(1, 1),
                    horizon=120, rules=dict(commons=True),
                    motifs=("outreach_awareness", "shared_attention_pool", "two_seat_rivalry")),
    "complement": dict(seats=2, levers=("price", "quality", "outreach"), noop=(0, 1), net=("blocks",),
                       n_inc=(1, 1), horizon=120, rules=dict(bundle=True, omega=0.05),
                       cost=dict(unit=0.30, quality=0.05), seat=dict(price_lo=0.1, price_hi=1.5, price_default=0.8),
                       motifs=("outreach_awareness", "bundle_of_complements", "weakest_link_quality",
                               "sum_of_prices")),
    "duopoly_fx": dict(seats=2, levers=("price", "quality", "outreach"), noop=(0, 1), net=("ring",),
                       n_inc=(1, 1), horizon=160, rules=dict(nfx=True), pop=dict(nfx_scale=1.2, wtp_med=1.0),
                       inc=dict(aware0=(0.10, 0.20), attach=0.5, reach=(0.10, 0.20), price=(0.9, 1.1)),
                       motifs=("outreach_awareness", "local_network_effects", "lock_in", "two_seat_rivalry")),
}
# The causal chain of each kind, as stated to the operator in `truth()["chain"]`.
_CHAINS: dict[str, dict[str, str]] = {
    "utility": dict(
        awareness="own outreach spend (a hazard per customer); forgotten at a constant rate unless a customer",
        trial="a private random-utility comparison of known, affordable firms, with a neutral prior on quality",
        retention="the customer's own remembered experience, and a mild habit",
        feedback="none through other customers; incumbents answer lost share with their price"),
    "wom": dict(
        awareness="satisfied current customers telling network neighbours; outreach is weak; a referral reward "
                  "raises the telling rate and is paid per referred first purchase",
        trial="the same comparison, with an optimistic prior for a firm one was told about",
        retention="remembered experience and a mild habit",
        feedback="reinforcing: quality -> satisfied customers -> awareness along edges -> customers"),
    "congestion": dict(
        awareness="own outreach spend",
        trial="the same comparison, minus the remembered wait at each firm",
        retention="being served: capacity rations orders at random; the unserved pay nothing, complain and lapse",
        feedback="balancing: demand -> queue -> remembered wait and churn -> demand (incumbents add capacity)"),
    "trust_gate": dict(
        awareness="own outreach spend",
        trial="gated: a customer with no experience of a firm considers it only once its public rating has passed "
              "their personal threshold with enough reviews behind it; a few pioneers have no threshold",
        retention="a strong habit (switching cost) that builds with tenure",
        feedback="reinforcing with a threshold: reviews -> rating -> gate opens -> trial -> reviews (cold start)"),
    "network_fx": dict(
        awareness="own outreach spend",
        trial="value rises with the share of the customer's network neighbours currently using the firm",
        retention="lock-in: neighbours' usage plus habit",
        feedback="reinforcing and path dependent: installed base -> value -> adoption; the market starts open"),
    "segments": dict(
        awareness="segment A: own outreach spend only; segment B: satisfied neighbours only (of either segment)",
        trial="the same comparison in both segments",
        retention="remembered experience and a mild habit",
        feedback="segment B only: quality -> satisfied customers -> awareness along edges"),
    "duopoly": dict(
        awareness="each seat's own outreach spend",
        trial="a private comparison in which both seats (and incumbents) are options",
        retention="remembered experience and a mild habit",
        feedback="none through other customers; each seat's gain is taken from the other and from incumbents"),
    "commons": dict(
        awareness="each seat's outreach spend times the level of a shared attention stock",
        trial="a private comparison in which both seats are options; how often customers shop falls with the stock",
        retention="remembered experience and a mild habit",
        feedback="balancing and shared: total outreach of both seats -> attention stock down -> everyone's "
                 "outreach and all shopping down; the stock regenerates"),
    "complement": dict(
        awareness="each seat's own outreach spend; the bundle is considered only by customers who know both seats",
        trial="the bundle is valued at its weaker component less a mismatch penalty and costs the sum of both "
              "prices; a component alone is worth a fraction",
        retention="remembered experience of the bundle and a mild habit",
        feedback="none through other customers; an integrated incumbent sells a whole bundle"),
    "duopoly_fx": dict(
        awareness="each seat's own outreach spend",
        trial="value rises with the share of the customer's neighbours using that seat's firm",
        retention="lock-in: neighbours' usage plus habit",
        feedback="reinforcing and path dependent: an early lead in installed base decides the market"),
}
_LEVER_ROLES = ("price", "quality", "capacity", "outreach", "referral")
_GAUGE_ROLES = ("orders", "active", "rating", "complaints", "queue", "volume")


# ====================================================================== deterministic structure from the config
def _net_edges(net: dict, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Directed edge list (each undirected edge twice), sorted, of the generated social network; plus the block
    of every node for the community generator. Deterministic in (`net`, n)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(net["seed"]), int(n), 7]))
    kind = net["kind"]
    block = None
    if kind == "ring":                      # ring lattice, each tie rewired with probability `rewire`
        kk = max(1, int(net.get("k", 4)))
        i = np.repeat(np.arange(n), kk)
        j = (i + np.tile(np.arange(1, kk + 1), n)) % n
        rew = rng.random(i.size) < float(net.get("rewire", 0.1))
        j = np.where(rew, rng.integers(0, n, i.size), j)
    elif kind == "pa":                      # preferential attachment: each newcomer ties to m degree-drawn nodes
        m = max(1, int(net.get("m", 4)))
        ends: list[int] = list(range(m + 1)) * m
        src: list[int] = []
        dst: list[int] = []
        for a in range(m + 1):
            for b in range(a + 1, m + 1):
                src.append(a)
                dst.append(b)
        picks = rng.random((n, m))
        for v in range(m + 1, n):
            chosen = {ends[int(picks[v, q] * len(ends))] for q in range(m)}
            for w in sorted(chosen):
                src.append(v)
                dst.append(w)
                ends.append(w)
                ends.append(v)
        i = np.array(src, dtype=np.int64)
        j = np.array(dst, dtype=np.int64)
    elif kind == "blocks":                  # community blocks: dense inside a block, sparse across
        nb = max(2, int(net.get("blocks", 6)))
        kin, kout = max(1, int(net.get("k_in", 3))), int(net.get("k_out", 1))
        block = (np.arange(n) * nb) // n
        lo = np.searchsorted(block, np.arange(nb))
        hi = np.searchsorted(block, np.arange(nb), side="right")
        i_in = np.repeat(np.arange(n), kin)
        b = block[i_in]
        j_in = lo[b] + (rng.random(i_in.size) * (hi[b] - lo[b])).astype(np.int64)
        i_out = np.repeat(np.arange(n), kout)
        j_out = rng.integers(0, n, i_out.size)
        i = np.concatenate([i_in, i_out])
        j = np.concatenate([j_in, j_out])
    else:
        raise ValueError(f"unknown network kind {kind!r}")
    a, b = np.minimum(i, j), np.maximum(i, j)
    code = np.unique(a[a != b] * n + b[a != b])
    a, b = code // n, code % n
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    order = np.lexsort((dst, src))
    return src[order].astype(np.int64), dst[order].astype(np.int64), block


def _population(cfg: dict, block: np.ndarray | None, inv_deg: np.ndarray) -> np.ndarray:
    """The static traits of every customer, (N_TRAITS, N) float64. Deterministic in the config."""
    n, pop, p0 = int(cfg["n_customers"]), cfg["pop"], float(cfg["p0"])
    rng = np.random.default_rng(np.random.SeedSequence([int(pop["seed"]), n, 11]))
    z = rng.standard_normal((4, n))
    u = rng.random((10, n))
    segs = pop["segments"]
    cum = np.cumsum([float(s["share"]) for s in segs])
    cum = cum / cum[-1]
    if len(segs) == 1:
        seg = np.zeros(n, dtype=np.int64)
    elif block is not None:                 # whole communities belong to one segment
        nb = int(block.max()) + 1
        seg = np.searchsorted(cum, (block + 0.5) / nb)
    else:
        seg = np.searchsorted(cum, u[9])
    seg = np.minimum(seg, len(segs) - 1)

    def sv(key: str) -> np.ndarray:
        return np.array([float(s.get(key, 1.0)) for s in segs])[seg]

    tr = np.zeros((N_TRAITS, n))
    tr[T_INC] = np.maximum(1.0, np.round(p0 * pop["inc_med"] * np.exp(pop["inc_sig"] * (0.6 * z[0] + 0.8 * z[1]))))
    tr[T_BCAP] = np.round(tr[T_INC] * pop["save_ticks"])
    tr[T_WTP] = np.round(p0 * pop["wtp_med"] * np.exp(pop["wtp_sig"] * z[0]) * sv("wtp"))
    tr[T_WP] = np.exp(pop["wp_sig"] * z[2]) * sv("wp")
    tr[T_WQ] = (0.6 + 0.8 * u[0]) * pop["wq_scale"] * sv("wq")
    tr[T_WW] = (0.5 + 1.0 * u[1]) * pop["ww_scale"]
    tr[T_HAB] = (0.5 + 1.0 * u[2]) * pop["h_scale"]
    tr[T_NEED] = pop["need_lo"] + (pop["need_hi"] - pop["need_lo"]) * u[3]
    tr[T_STD] = np.clip(pop["std_mu"] + pop["std_sd"] * z[3], 0.2, 0.85)
    tr[T_RECV] = np.exp(0.3 * z[1]) * pop["recv_scale"] * sv("recv")
    tr[T_SUS] = (0.6 + 0.8 * u[4]) * pop["sus_scale"] * sv("sus")
    tr[T_NFX] = (0.7 + 0.6 * u[5]) * pop["nfx_scale"]
    pioneer = u[6] < pop["pioneer"]
    tr[T_GTH] = np.where(pioneer, 0.0, pop["th_lo"] + (pop["th_hi"] - pop["th_lo"]) * u[7])
    tr[T_GM] = np.where(pioneer, 0.0, pop["m_lo"] + (pop["m_hi"] - pop["m_lo"]) * u[8])
    tr[T_SEG] = seg
    rules = cfg["rules"]
    wtpn = tr[T_WTP] / p0
    tr[T_VB] = wtpn * tr[T_WQ]
    tr[T_VA] = wtpn - tr[T_VB] * float(rules["qref"])
    tr[T_WPN] = tr[T_WP] / p0
    tr[T_HABR] = tr[T_HAB] / float(rules["tau"])
    tr[T_STDM] = tr[T_STD] - float(rules["margin"])
    tr[T_NFXD] = tr[T_NFX] * inv_deg
    return tr


def _compile(cfg: dict) -> SimpleNamespace:
    """Everything `step` needs, derived deterministically from the config once per engine instance."""
    k = SimpleNamespace()
    r, cost, seat, ad = cfg["rules"], cfg["cost"], cfg["seat"], cfg["adapt"]
    n = k.N = int(cfg["n_customers"])
    s_ = k.S = int(cfg["n_seats"])
    j_ = k.J = len(cfg["inc"])
    f_ = k.F = s_ + j_
    p0 = k.p0 = float(cfg["p0"])
    for key in ("sigma", "qref", "q_prior", "alpha", "sx", "phi", "org", "beta0", "ref_rho", "alpha_w", "w_decay",
                "pc_wait", "solo_scale", "mismatch", "pool_regen", "pool_deplete", "pool_need", "rho",
                "rating_decay", "rating_n0", "rating_r0", "pc", "margin", "delta", "tau", "vol_noise"):
        setattr(k, key, float(r[key]))
    for key in ("wom", "congest", "gate", "nfx", "bundle", "commons"):
        setattr(k, key, bool(r[key]))
    k.L = int(r["price_lag"])
    k.ref_half = float(r["ref_half"]) * p0
    k.inv_out = 1.0 / (n * p0 * float(r["omega"]))          # spend (cents) -> hazard for receptivity 1
    k.inv_pool = 1.0 / (n * p0 * float(r["pool_omega"]))    # spend (cents) -> depletion pressure
    if k.bundle and (s_ != 2 or k.wom or k.nfx or k.congest or k.gate):
        raise ValueError("the bundle rule needs exactly two seats and none of wom/nfx/congest/gate")
    # ---- options a customer can choose (a firm, or the bundle of both seats)
    if k.bundle:
        oa = [0, 0, 1] + list(range(2, f_))
        ob = [1, 0, 1] + list(range(2, f_))
        oscale = [1.0, k.solo_scale, k.solo_scale] + [1.0] * j_
    else:
        oa = ob = list(range(f_))
        oscale = [1.0] * f_
    o_ = k.O = len(oa)
    k.oa, k.ob = oa, ob
    k.oa_arr, k.ob_arr = np.array(oa), np.array(ob)
    k.oscale = np.array(oscale)
    k.fo = [[o for o in range(o_) if oa[o] == f or ob[o] == f] for f in range(f_)]   # options a firm sells in
    k.opt_inc = [k.fo[s_ + j][0] for j in range(j_)]
    k.member = np.zeros((o_, f_), dtype=np.int8)
    for o in range(o_):
        k.member[o, oa[o]] = 1
        k.member[o, ob[o]] = 1
    k.arO_col = np.arange(o_)[:, None]
    k.arN = np.arange(n)
    k.oscale_col = k.oscale[:, None]
    k.member_t = np.ascontiguousarray(k.member.T)
    # ---- rows of the per-tick uniform block
    k.r_pri = 3 if k.congest else -1
    k.r_aw = 4 if k.congest else 3
    k.r_e = k.r_aw + f_
    k.n_rows = k.r_e + o_
    k.n_small = 2 * j_ + s_
    k.n_block = k.n_rows * n
    k.n_words = (k.n_block + k.n_small + 3) // 4
    k.noise_w = k.sigma * math.sqrt(12.0)       # taste shock: uniform with standard deviation sigma
    k.half_noise = 0.5 * k.noise_w
    k.sx_w = k.sx * math.sqrt(12.0)             # experience noise: uniform with standard deviation sx
    k.sx_half = 0.5 * k.sx_w
    k.vol_w = k.vol_noise * math.sqrt(12.0)
    k.pc_hi, k.pcw_hi, k.idle_hi = 1.0 - k.pc, 1.0 - k.pc_wait, 1.0 - k.delta
    k.need_tried = k.gate or k.wom
    # ---- network and population
    esrc, edst, block = _net_edges(cfg["net"], n)
    k.edst = edst
    k.esrc = esrc
    k.NF = n * f_
    k.NF1 = n * (f_ + 1)
    deg = np.bincount(esrc, minlength=n).astype(np.float64)
    k.inv_deg = 1.0 / np.maximum(deg, 1.0)
    k.n_edges = int(esrc.size)
    k.traits = _population(cfg, block, k.inv_deg)
    k.income_total = float(k.traits[T_INC].sum())
    k.endow = float(cfg["pop"]["endow_ticks"])
    # ---- seat levers (dial 0..100 -> natural range)
    k.lever_pairs = [(lv["name"], lv["role"]) for lv in cfg["levers"]]
    names = {lv["role"]: lv["name"] for lv in cfg["levers"]}
    dflt = {lv["role"]: float(lv["default"]) for lv in cfg["levers"]}
    flip = {lv["role"]: bool(lv.get("flip", False)) for lv in cfg["levers"]}

    def dial(role: str):                # (public name, default position, runs backwards?) or None if hidden
        return (names[role], dflt[role], flip[role]) if role in names else None
    k.l_price, k.l_quality, k.l_cap = dial("price"), dial("quality"), dial("capacity")
    k.l_out, k.l_ref = dial("outreach"), dial("referral")
    k.flip = flip
    k.flip_price = flip["price"]
    k.p_lo = round(float(seat["price_lo"]) * p0)
    k.p_span = round(float(seat["price_hi"]) * p0) - k.p_lo
    k.q_lo, k.q_span, k.q_gamma = float(seat["q_lo"]), float(seat["q_hi"]) - float(seat["q_lo"]), float(seat["q_gamma"])
    k.cap_lo = max(1, round(float(seat["cap_lo"]) * n))
    k.cap_span = max(1, round(float(seat["cap_hi"]) * n)) - k.cap_lo
    k.out_hi = round(float(seat["out_hi"]) * n * p0)
    k.ref_hi = round(float(seat["ref_hi"]) * p0)
    k.finite_cap = k.congest
    k.cap_ample = float(n)
    # ---- costs (cents)
    k.c_unit = float(cost["unit"]) * p0
    k.c_unit_inc = float(cost["unit_inc"]) * p0
    k.c_q = float(cost["quality"]) * n * p0
    k.c_cap = float(cost["capacity"]) * p0
    k.overhead = float(cost["overhead"]) * n * p0
    # ---- incumbents
    inc = cfg["inc"]
    k.inc_fixed = [float(i["fixed"]) * n * p0 for i in inc]
    k.inc_cap0 = [float(max(1, round(float(i["cap"]) * n))) if k.finite_cap else k.cap_ample for i in inc]
    k.inc_reach0 = [float(i["reach"]) for i in inc]
    k.eta_loss, k.eta_gain, k.kappa = float(ad["eta_loss"]), float(ad["eta_gain"]), float(ad["kappa"])
    k.share_rate, k.profit_rate = float(ad["share_rate"]), float(ad["profit_rate"])
    k.p_floor = float(round(float(ad["floor"]) * k.c_unit_inc))
    k.p_ceil = float(round(float(ad["ceil"]) * p0))
    k.cap_up, k.cap_down, k.util_lo = float(ad["cap_up"]), float(ad["cap_down"]), float(ad["util_lo"])
    k.cap_min, k.cap_max = float(max(1, round(0.02 * n))), float(n)
    k.exit_frac, k.grace, k.delay = float(ad["exit_frac"]), int(ad["grace"]), int(ad["delay"])
    k.ent_p, k.ent_q = [float(v) for v in ad["entrant_price"]], [float(v) for v in ad["entrant_quality"]]
    k.launch_aware = float(ad["launch_aware"])
    # ---- gauges
    k.gauge_pairs = [(g["name"], g["role"]) for g in cfg["gauges"]]
    k.rivals = [[f for f in range(f_) if f != s] for s in range(s_)]
    canon = list(_GAUGE_ROLES) + [f"rival_price_{i}" for i in range(f_ - 1)]
    k.gauge_plan = [(g["name"], canon.index(g["role"])) for g in cfg["gauges"]]
    k.inv_span = 100.0 / k.p_span
    return k


def _frac(st: dict, spec: tuple | None) -> float:
    """A dial as a fraction of its natural range in [0, 1] (0 for a hidden lever). A dial that runs backwards has
    100 at the bottom of the range; anything unusable falls back to the default position."""
    if spec is None:
        return 0.0
    name, dflt, flip = spec
    try:
        d = float(st.get(name, dflt)) * 0.01
    except (TypeError, ValueError):
        d = dflt * 0.01
    if d != d:
        d = dflt * 0.01
    d = 0.0 if d < 0.0 else (1.0 if d > 1.0 else d)
    return 1.0 - d if flip else d


def _merge(base: dict, *over: dict | None) -> dict:
    out = dict(base)
    for o in over:
        if o:
            out.update(o)
    return out


def _draw(rng: np.random.Generator, spec) -> float:
    """A number, or a uniform draw from a (lo, hi) pair. Always consumes exactly one draw."""
    u = float(rng.random())
    if isinstance(spec, (list, tuple)):
        return float(spec[0]) + (float(spec[1]) - float(spec[0])) * u
    return float(spec)


# ====================================================================== the engine
@register_engine
class AbmEngine(Engine):
    """Agent-based market. See the module docstring for the tick loop and the families."""
    engine = "abm"
    version = "1"
    state_in_event = False      # the state holds every customer: hosts recompute the step on replay

    def __init__(self, config: dict):
        super().__init__(config)
        self._k = _compile(config)
        self._levers = [Lever(lv["name"], 0.0, 100.0, float(lv["default"])) for lv in config["levers"]]
        self._gauges = [Gauge(g["name"]) for g in config["gauges"]]

    # ------------------------------------------------------------------ construction
    @classmethod
    def generate(cls, rng: np.random.Generator, params: dict) -> dict:
        """Draw a market. `params["kind"]` picks the structural rule set (see `_KINDS`); every other key overrides
        a drawn or default value: `n_customers`, `n_seats`, `n_inc`, `horizon`, `noop` (number of distractors),
        `net`, `p0`, `flip` (False: no dial runs backwards), and partial dicts `rules`, `pop` (e.g. `endow_ticks`),
        `cost`, `seat`, `adapt`, `inc` (applied to every incumbent)."""
        p = dict(params or {})
        kind = str(p.get("kind", "utility"))
        if kind not in _KINDS:
            raise ValueError(f"unknown abm kind {kind!r}")
        ks = _KINDS[kind]
        # every draw happens before any override is applied, so overriding one thing changes nothing else
        p0 = int(rng.integers(300, 901))
        n = int((400, 500, 600)[int(rng.integers(3))])
        net_seed, pop_seed = (int(v) for v in rng.integers(0, 2 ** 31 - 1, 2))
        net_kind = ks["net"][int(rng.integers(len(ks["net"])))]
        n_inc = int(rng.integers(ks["n_inc"][0], ks["n_inc"][1] + 1))
        n_noop = int(rng.integers(ks["noop"][0], ks["noop"][1] + 1))
        lag = int(rng.integers(1, 3))
        inc_spec = _merge(_BASE_INC, ks.get("inc"), p.get("inc"))
        inc_draws = [{key: _draw(rng, inc_spec[key]) for key in ("price", "quality", "cap", "reach", "aware0")}
                     for _ in range(3)]
        seat = _merge(_BASE_SEAT, ks.get("seat"), p.get("seat"))
        price_default = float(seat["price_default"]) * (0.95 + 0.15 * float(rng.random()))
        noop_defaults = [round(60.0 * float(rng.random()), 0) for _ in range(2)]
        net_draw = {"k": int(rng.integers(3, 6)), "rewire": round(0.05 + 0.15 * float(rng.random()), 3),
                    "m": int(rng.integers(3, 6)), "blocks": int(2 * rng.integers(2, 6)),
                    "k_in": int(rng.integers(3, 5)), "k_out": 1}
        # ---- apply overrides
        n = int(p.get("n_customers", n))
        if not 50 <= n <= 20000:
            raise ValueError("n_customers must be between 50 and 20000")
        seats = int(p.get("n_seats", ks["seats"]))
        n_inc = int(p.get("n_inc", n_inc))
        n_noop = int(p.get("noop", n_noop))
        p0 = int(p.get("p0", p0))
        rules = _merge(_BASE_RULES, ks.get("rules"), {"price_lag": lag}, p.get("rules"))
        pop = _merge(_BASE_POP, ks.get("pop"), {"seed": pop_seed}, p.get("pop"))
        pop["segments"] = [dict(s) for s in pop["segments"]]
        cost = _merge(_BASE_COST, ks.get("cost"), p.get("cost"))
        if rules["bundle"] and "unit_inc" not in (p.get("cost") or {}):
            cost["unit_inc"] = 2.0 * float(cost["unit"])       # an integrated rival makes both components
        adapt = _merge(_BASE_ADAPT, ks.get("adapt"), p.get("adapt"))
        adapt["entrant_price"] = [float(v) for v in adapt["entrant_price"]]
        adapt["entrant_quality"] = [float(v) for v in adapt["entrant_quality"]]
        net = _merge({"kind": net_kind, "seed": net_seed}, net_draw, p.get("net"))
        inc = []
        for j in range(n_inc):
            d = inc_draws[j % 3]
            inc.append({"price": int(round(d["price"] * p0)), "quality": round(d["quality"], 4),
                        "cap": round(d["cap"], 4), "reach": round(d["reach"], 4), "aware0": round(d["aware0"], 4),
                        "attach": float(inc_spec["attach"]), "fixed": float(inc_spec["fixed"])})
        # ---- public surface: opaque names, seeded order, dials 0..100
        roles = [ro for ro in _LEVER_ROLES if ro in ks["levers"]] + [f"noop{i + 1}" for i in range(n_noop)]
        g_roles = list(_GAUGE_ROLES) + [f"rival_price_{i}" for i in range(seats + n_inc - 1)]
        l_names = neutral_names(rng, len(roles))
        l_order = [int(i) for i in rng.permutation(len(roles))]
        names = l_names + neutral_names(rng, len(g_roles), used=set(l_names))
        g_order = [int(i) for i in rng.permutation(len(g_roles))]
        # Which way each dial runs is seeded too (a flipped dial has the top of its range at 0), so the direction
        # of a lever says nothing about what it is. In natural terms a seat rests at a high price with no effort,
        # no outreach, the least capacity and no reward: next to nothing happens until it is operated.
        flips = {ro: bool(f) for ro, f in zip(_LEVER_ROLES, rng.random(len(_LEVER_ROLES)) < 0.5)}
        if p.get("flip") is False:
            flips = {ro: False for ro in flips}
        p_lo, p_hi = round(float(seat["price_lo"]) * p0), round(float(seat["price_hi"]) * p0)
        rest = {"price": round((price_default * p0 - p_lo) / (p_hi - p_lo), 2)}
        levers = []
        for i in l_order:
            ro = roles[i]
            fl = bool(flips.get(ro, False))
            at = rest.get(ro, 0.0)
            dflt = noop_defaults[(int(ro[4:]) - 1) % 2] if ro.startswith("noop") else 100.0 * (1.0 - at if fl else at)
            levers.append({"name": names[i], "role": ro, "default": float(round(dflt, 0)), "flip": fl})
        gauges = [{"name": names[len(roles) + i], "role": g_roles[i]} for i in g_order]
        return {"engine": "abm", "abm": 1, "kind": kind, "n_seats": seats, "n_customers": n,
                "horizon": int(p.get("horizon", ks["horizon"])), "p0": p0, "net": net, "pop": pop, "rules": rules,
                "cost": cost, "seat": seat, "adapt": adapt, "inc": inc, "levers": levers, "gauges": gauges,
                "motifs": list(ks["motifs"])}

    # ------------------------------------------------------------------ public surface
    def levers(self, seat: int) -> list[Lever]:
        return list(self._levers)

    def gauges(self, seat: int) -> list[Gauge]:
        return list(self._gauges)

    def brief(self, seat: int) -> str:
        s = self._k.S
        who = "the only seat" if s == 1 else f"seat {int(seat) + 1} of {s}"
        return (f"You operate {who} of a facility. Every tick the facility pays your seat its receipts for that "
                "tick and charges the running cost of your current settings. Your score each tick is receipts "
                "minus running cost. Every setting is a dial from 0 to 100; every reading is reported once per "
                "tick.")

    # ------------------------------------------------------------------ private helpers (operator side only)
    def roles(self, seat: int = 0) -> dict:
        """PRIVATE. What every public name controls or shows, with the natural range behind each dial."""
        k, seat_cfg = self._k, self.config["seat"]
        nat = {"price": (k.p_lo, k.p_lo + k.p_span, "cents"), "quality": (0.0, 1.0, "effort"),
               "capacity": (k.cap_lo, k.cap_lo + k.cap_span, "slots"), "outreach": (0, k.out_hi, "cents per tick"),
               "referral": (0, k.ref_hi, "cents per referred first purchase")}
        lv = {}
        for name, role in k.lever_pairs:
            lo, hi, unit = nat.get(role, (0.0, 0.0, "nothing (distractor)"))
            lv[role] = {"public": name, "natural_lo": lo, "natural_hi": hi, "unit": unit, "flip": k.flip[role]}
        rv = self._k.rivals[seat]
        gg = {}
        for name, role in k.gauge_pairs:
            what = role
            if role.startswith("rival_price_"):
                f = rv[int(role.rsplit("_", 1)[1])]
                what = (f"price dial of seat {f}" if f < k.S else f"price dial of incumbent {f - k.S}") + \
                    f", lagged {k.L}"
            gg[role] = {"public": name, "shows": what}
        return {"levers": lv, "gauges": gg, "quality_levels": [seat_cfg["q_lo"], seat_cfg["q_hi"]]}

    @property
    def income_cents_per_tick(self) -> float:
        """PRIVATE. Total income of all customers per tick, in cents (exact: the population is in the config)."""
        return self._k.income_total

    def revenue_bound(self, ticks: int) -> float:
        """PRIVATE. The most all firms together can have been paid after `ticks` ticks, in cents: the customers'
        initial budgets (`pop.endow_ticks` ticks of income, 0 for none) plus their income since. A host that funds
        the demand side with this never falls short."""
        tr = self._k.traits
        return float(np.minimum(tr[T_INC] * self._k.endow, tr[T_BCAP]).sum() + int(ticks) * self._k.income_total)

    def encode(self, natural: dict, base: dict | None = None) -> dict[str, float]:
        """PRIVATE. Settings (public name -> dial) from natural values by role: `price` in cents, `quality` as
        effort in [0, 1], `capacity` in slots, `outreach` in cents per tick, `referral` in cents. Roles not given
        keep their default (or `base`)."""
        k = self._k
        out = dict(base) if base is not None else self.defaults(0)
        span = {"price": (k.p_lo, k.p_span), "quality": (0.0, 1.0), "capacity": (k.cap_lo, k.cap_span),
                "outreach": (0.0, k.out_hi), "referral": (0.0, k.ref_hi)}
        names = {role: name for name, role in k.lever_pairs}
        for role, v in natural.items():
            if role not in names:
                continue
            if role in span:
                lo, sp = span[role]
                d = min(100.0, max(0.0, 100.0 * (float(v) - lo) / max(sp, 1e-12)))
                out[names[role]] = 100.0 - d if k.flip[role] else d
            else:
                out[names[role]] = min(100.0, max(0.0, float(v)))
        return out

    # ------------------------------------------------------------------ dynamics
    def init_state(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """The market before the seats have done anything: incumbents are known and have customers, the seats'
        firms are almost unknown. Arrays are laid out firm-major: (firms or options, customers)."""
        k, cfg = self._k, self.config
        n, f_, o_, s_, j_ = k.N, k.F, k.O, k.S, k.J
        u = rng.random((3 + f_ + max(j_, 1), n))
        tr = k.traits.copy()
        inc = cfg["inc"]
        aware0 = np.array([float(cfg["seat"]["aware0"])] * s_ + [float(i["aware0"]) for i in inc])
        aware = u[3:3 + f_] < aware0[:, None]
        cur = np.full(n, o_, dtype=np.int64)
        mem_q = np.full((o_, n), k.q_prior)
        tried = np.zeros((o_, n), dtype=bool)
        happy = np.zeros(n, dtype=bool)
        tenure = np.zeros(n)
        if j_:
            score = np.where(aware[s_:], u[3 + f_:3 + f_ + j_], -1.0)
            best = score.argmax(0)
            att = (score.max(0) >= 0.0) & (u[0] < np.array([float(i["attach"]) for i in inc])[best])
            idx = np.flatnonzero(att)
            cur[idx] = np.array(k.opt_inc, dtype=np.int64)[best[idx]]
            q = np.clip(np.array([float(i["quality"]) for i in inc])[best[idx]] + k.sx_w * (u[2][idx] - 0.5),
                        0.0, 1.0)
            mem_q[cur[idx], idx] = q
            tried[cur[idx], idx] = True
            happy[idx] = q >= tr[T_STD][idx]
            tenure[idx] = 1.0 + np.floor(u[1][idx] * 12.0)
        budget = np.minimum(np.floor(tr[T_INC] * k.endow), tr[T_BCAP])
        firm = np.zeros((N_FROWS, f_))
        p_def = float(round(k.p_lo + k.p_span * _frac({}, k.l_price)))
        for s in range(s_):
            firm[R_PRICE, s] = firm[R_PREF, s] = p_def
            firm[R_QUAL, s] = k.q_lo
            firm[R_CAP, s] = k.cap_lo if k.finite_cap else k.cap_ample
            firm[R_ALIVE, s] = 1.0
        for j, i in enumerate(inc):
            f = s_ + j
            firm[R_PRICE, f] = firm[R_PREF, f] = float(i["price"])
            firm[R_QUAL, f] = float(i["quality"])
            firm[R_CAP, f] = k.inc_cap0[j]
            firm[R_REACH, f] = k.inc_reach0[j]
            firm[R_ALIVE, f] = 1.0
            firm[R_RN, f] = 60.0                         # an established firm has a review history
            firm[R_RS, f] = 60.0 * float(i["quality"])
            firm[R_SHARE, f] = 1.0 / max(1, j_)
            firm[R_AGE, f] = 1000.0
        state = {"traits": tr, "budget": budget, "aware": aware, "mem_q": mem_q, "cur": cur, "tenure": tenure,
                 "firm": firm, "price_hist": np.tile(firm[R_PRICE], (k.L + 1, 1)),
                 "totals": np.array([float(budget.sum()), 0.0, 0.0, 0.0])}
        # arrays that only some rule sets read are only kept by those rule sets
        if k.congest:
            state["mem_w"] = np.zeros((o_, n))
        if k.need_tried:
            state["tried"] = tried
        if k.wom:
            state["happy"] = happy
        if k.commons:
            state["pool"] = np.array([1.0])
        return state

    def step(self, state: dict[str, np.ndarray], settings: list[dict[str, float]], rng: np.random.Generator,
             t: int) -> StepResult:
        k = self._k
        n, f_, o_, s_ = k.N, k.F, k.O, k.S
        tr = state["traits"]
        fl = state["firm"].tolist()
        # ---- every random number of the tick, in a fixed pattern (so paired runs share them exactly): uniforms
        #      with 16-bit resolution, four from each raw 64-bit word of the generator. Rows: need / idle lapse,
        #      review / complaint, experience, [service priority], awareness per firm, taste per option; then a
        #      few firm-level draws.
        raw = rng.bit_generator.random_raw(k.n_words).astype("<u8", copy=False).view("<u2") * _U16
        u = raw[:k.n_block].reshape(k.n_rows, n)
        us = raw[k.n_block:k.n_block + k.n_small].tolist()
        u0, u1 = u[0], u[1]

        # ---- 1. income accrues to capped budgets
        b0 = state["budget"]
        budget = np.minimum(b0 + tr[T_INC], tr[T_BCAP])
        credited = float(budget.sum() - b0.sum())

        # ---- 2. offers: seats from their dials, incumbents from their own state
        price, qual, cap, reach, alive = fl[R_PRICE], fl[R_QUAL], fl[R_CAP], fl[R_REACH], fl[R_ALIVE]
        effort, spend, refr = [0.0] * s_, [0.0] * s_, [0.0] * s_
        for s in range(s_):
            st = settings[s]
            price[s] = float(round(k.p_lo + k.p_span * _frac(st, k.l_price)))
            eff = _frac(st, k.l_quality)
            effort[s] = eff
            qual[s] = k.q_lo + k.q_span * eff ** k.q_gamma
            if k.l_cap is not None:
                cap[s] = float(round(k.cap_lo + k.cap_span * _frac(st, k.l_cap)))
            if k.l_out is not None:
                spend[s] = float(round(k.out_hi * _frac(st, k.l_out)))
                reach[s] = spend[s] * k.inv_out
            if k.l_ref is not None:
                refr[s] = float(round(k.ref_hi * _frac(st, k.l_ref)))
        dead = [f for f in range(f_) if alive[f] <= 0.0]

        # ---- 3. awareness: outreach, neighbours' word of mouth, chance; forgetting of firms one does not use
        aware0, cur0 = state["aware"], state["cur"]
        pool = 1.0
        reach_col = np.array(reach)[:, None]
        if k.commons:
            pool = float(state["pool"][0])
            reach_col *= pool
        h = reach_col * tr[T_RECV]
        cnt = None
        if k.wom:                           # neighbours who are satisfied current customers of each firm
            adv = np.where(state["happy"], cur0, f_)
            cnt = np.bincount(k.esrc + (adv * n)[k.edst], minlength=k.NF1)
            beta = [k.beta0] * f_
            for s in range(s_):
                beta[s] = k.beta0 * (1.0 + k.ref_rho * refr[s] / (refr[s] + k.ref_half))
            h += (cnt[:k.NF].reshape(f_, n) * tr[T_SUS]) * np.array(beta)[:, None]
        np.subtract(-k.org, h, out=h)
        np.exp(h, out=h)                    # probability of staying unaware this tick
        np.putmask(h, aware0, k.phi)        # ... and of forgetting a firm already known
        iscur = k.arO_col == cur0
        aware = u[k.r_aw:k.r_aw + f_] >= h
        aware |= ((k.member_t @ iscur.view(np.int8)) > 0) if k.bundle else iscur
        if dead:
            aware[dead] = False

        # ---- 4. choice: value of each known, affordable option against the outside option (0)
        mq = state["mem_q"]
        v = mq * tr[T_VB]
        v += tr[T_VA]
        if k.bundle:
            v *= k.oscale_col
            popt = [price[k.oa[o]] + (price[k.ob[o]] if k.ob[o] != k.oa[o] else 0.0) for o in range(o_)]
            q0, q1 = qual[0], qual[1]
            qopt = [min(q0, q1) - k.mismatch * abs(q0 - q1)] + qual
            aw_opt = aware[k.oa_arr] & aware[k.ob_arr]
        else:
            popt, qopt, aw_opt = price, qual, aware
        popt_col = np.array(popt)[:, None]
        v -= popt_col * tr[T_WPN]
        ten0 = state["tenure"]
        v += iscur * (tr[T_HABR] * np.minimum(ten0, k.tau))
        if k.congest:
            v -= state["mem_w"] * tr[T_WW]
        if k.nfx:                           # share of the customer's neighbours who currently use each firm
            nb = np.bincount(k.esrc + (cur0 * n)[k.edst], minlength=k.NF1)
            v += nb[:k.NF].reshape(f_, n) * tr[T_NFXD]
        v += u[k.r_e:k.r_e + o_] * k.noise_w     # taste shock, centred by comparing with half its width below
        ok = aw_opt & (popt_col <= budget)
        rs, rn = fl[R_RS], fl[R_RN]
        if k.gate:
            rating0 = np.array([(rs[f] + k.rating_r0 * k.rating_n0) / (rn[f] + k.rating_n0) for f in range(f_)])
            ok &= state["tried"] | ((rating0[:, None] >= tr[T_GTH]) & (np.array(rn)[:, None] >= tr[T_GM]))
        if o_ <= 4:                         # best allowed option, first one on ties (a running maximum is faster
            best = np.where(ok[0], v[0], _NEG)   # than a masked argmax across so few options)
            ch = np.zeros(n, dtype=np.int64)
            for o in range(1, o_):
                vo = v[o]
                better = (vo > best) & ok[o]
                np.putmask(ch, better, o)
                np.putmask(best, better, vo)
            flat = ch * n + k.arN
        else:
            v = np.where(ok, v, _NEG)
            ch = v.T.argmax(1)
            flat = ch * n + k.arN
            best = v.ravel()[flat]
        need_p = tr[T_NEED]
        if k.commons:
            need_p = need_p * ((1.0 - k.pool_need) + k.pool_need * pool)
        need = u0 < need_p
        buy = need & (best > k.half_noise)

        # ---- 5. service: capacity rations orders at random; the unserved pay nothing
        chb = np.where(buy, ch, o_)
        ordc = np.bincount(chb, minlength=o_ + 1).tolist()
        served, rationed = buy, False
        uns = [0] * f_
        if k.finite_cap:
            for f in range(f_):
                c = int(cap[f])
                if ordc[f] > c:
                    if not rationed:
                        served, rationed = buy.copy(), True
                    idx_f = np.flatnonzero(chb == f)
                    drop = idx_f[np.argpartition(u[k.r_pri][idx_f], c)[c:]] if c > 0 else idx_f
                    served[drop] = False
                    uns[f] = ordc[f] - c
        if rationed:
            chs = np.where(served, ch, o_)
            sv = np.bincount(chs, minlength=o_ + 1).tolist()
        else:
            chs, sv = chb, ordc

        # ---- 6. experience, memory, reviews, complaints
        x = (np.array(qopt) - k.sx_half)[ch]
        x += u[2] * k.sx_w
        np.maximum(x, 0.0, out=x)
        np.minimum(x, 1.0, out=x)
        old = mq.ravel()[flat]
        mem_q = mq.copy()
        mem_q.ravel()[flat] = old + (x - old) * (served * k.alpha)
        std = tr[T_STD]
        rvw = served & (u1 < k.rho)
        rsum = np.bincount(ch, weights=x * rvw, minlength=o_).tolist()
        rcnt = np.bincount(ch, weights=rvw, minlength=o_).tolist()
        bad = served & (x < tr[T_STDM]) & (u1 > k.pc_hi)
        cmpl = np.bincount(ch, weights=bad, minlength=o_).tolist()
        mem_w = None
        if k.congest:
            mem_w = state["mem_w"] * k.w_decay
            oldw = mem_w.ravel()[flat]
            mem_w.ravel()[flat] = oldw + ((buy & ~served) - oldw) * (buy * k.alpha_w)
            if rationed:
                cw = np.bincount(ch, weights=buy & ~served & (u1 > k.pcw_hi), minlength=o_).tolist()
                cmpl = [cmpl[o] + cw[o] for o in range(o_)]
        tried = nref = None
        if k.need_tried:
            was = state["tried"].ravel()[flat]
            tried = state["tried"].copy()
            tried.ravel()[flat] = was | served
            if cnt is not None and any(refr):   # first purchases by customers with a satisfied neighbour there
                nref = np.bincount(ch, weights=served & ~was & (cnt[flat] > 0), minlength=o_).tolist()

        # ---- 7. payment: exactly the posted price, out of the budget (affordability was checked at choice)
        budget -= np.array(popt + [0.0])[chs]
        if k.bundle:
            units = [sum(sv[o] for o in k.fo[f]) for f in range(f_)]
            orders = [sum(ordc[o] for o in k.fo[f]) for f in range(f_)]
        else:
            units, orders = sv[:f_], ordc[:f_]
        revenue = [price[f] * units[f] for f in range(f_)]
        vol = len(chs) - sv[o_]

        # ---- 8. loyalty: served customers stay or switch; a customer who needed and was not served, lapses,
        #         and now and then an idle customer simply drops their firm
        moved = need | (u0 > k.idle_hi)
        cur = np.where(moved, chs, cur0)
        tenure = ten0 * (cur == cur0) + served
        happy = np.where(moved, served & (x >= std), state["happy"]) if k.wom else None

        # ---- 9. ratings, then incumbents adapt: price to share, capacity to queues, exit and entry
        events: list[dict] = []
        new_rs, new_rn = fl[R_RS], fl[R_RN]
        for f in range(f_):
            if alive[f] > 0.0:
                if k.bundle:
                    new_rs[f] = rs[f] * k.rating_decay + sum(rsum[o] for o in k.fo[f])
                    new_rn[f] = rn[f] * k.rating_decay + sum(rcnt[o] for o in k.fo[f])
                else:
                    new_rs[f] = rs[f] * k.rating_decay + rsum[f]
                    new_rn[f] = rn[f] * k.rating_decay + rcnt[f]
        price_now, qual_now, cap_now, alive_now = list(price), list(qual), list(cap), list(alive)
        inv_vol = 1.0 / max(1, vol)
        for j in range(k.J):
            f = s_ + j
            if alive[f] > 0.0:
                share = units[f] * inv_vol
                ema = fl[R_SHARE][f]
                dev = max(-1.0, min(1.0, (share - ema) / max(ema, 0.05)))
                pj = price[f]
                profit = (pj - k.c_unit_inc) * units[f] - k.inc_fixed[j]
                pj = pj * (1.0 + (k.eta_loss if dev < 0.0 else k.eta_gain) * dev) + k.kappa * (fl[R_PREF][f] - pj)
                price[f] = min(k.p_ceil, max(k.p_floor, float(round(pj))))
                fl[R_SHARE][f] = ema + k.share_rate * (share - ema)
                if k.finite_cap:
                    profit -= k.c_cap * cap[f]
                    if uns[f] > 0:
                        cap[f] = min(k.cap_max, cap[f] + math.ceil(k.cap_up * uns[f]))
                    elif units[f] < k.util_lo * cap[f]:
                        cap[f] = max(k.cap_min, cap[f] - max(1.0, float(round(k.cap_down * cap[f]))))
                pema = fl[R_PEMA][f] + k.profit_rate * (profit - fl[R_PEMA][f])
                fl[R_PEMA][f] = pema
                age = fl[R_AGE][f] + 1.0
                fl[R_AGE][f] = age
                if age >= k.grace and pema < -k.exit_frac * k.inc_fixed[j]:
                    o = k.opt_inc[j]            # exit: its customers are released and it is forgotten
                    gone = cur == o
                    cur[gone] = o_
                    tenure[gone] = 0.0
                    aware[f] = False
                    mem_q[o] = k.q_prior
                    if mem_w is not None:
                        mem_w[o] = 0.0
                    if tried is not None:
                        tried[o] = False
                    if happy is not None:
                        happy[gone] = False
                    alive[f] = 0.0
                    fl[R_TIMER][f] = float(k.delay)
                    new_rs[f] = new_rn[f] = 0.0
                    events.append({"t": int(t), "what": "exit", "firm": f})
            else:
                timer = fl[R_TIMER][f] - 1.0
                fl[R_TIMER][f] = timer
                if timer <= 0.0:                # a replacement enters with freshly drawn price and quality
                    pref = float(round(k.p0 * (k.ent_p[0] + (k.ent_p[1] - k.ent_p[0]) * us[j])))
                    price[f] = fl[R_PREF][f] = min(k.p_ceil, max(k.p_floor, pref))
                    qual[f] = k.ent_q[0] + (k.ent_q[1] - k.ent_q[0]) * us[k.J + j]
                    cap[f] = k.inc_cap0[j]
                    reach[f] = k.inc_reach0[j]
                    alive[f] = 1.0
                    fl[R_SHARE][f] = 1.0 / f_
                    fl[R_PEMA][f] = 0.0
                    fl[R_AGE][f] = 0.0
                    aware[f] = u[k.r_aw + f] < k.launch_aware
                    events.append({"t": int(t), "what": "entry", "firm": f})
        firm = np.fromiter(chain.from_iterable(fl), dtype=np.float64, count=N_FROWS * f_).reshape(N_FROWS, f_)

        # ---- 10. what each seat reads, earns and pays
        hist = np.array([[price_now[f] if f not in dead else -1.0 for f in range(f_)]]
                        + state["price_hist"].tolist()[:-1])
        lagged = hist[k.L].tolist()
        act = np.bincount(cur, minlength=o_ + 1).tolist()
        active = [sum(act[o] for o in k.fo[f]) for f in range(f_)] if k.bundle else act[:f_]
        rating = [(new_rs[f] + k.rating_r0 * k.rating_n0) / (new_rn[f] + k.rating_n0) for f in range(f_)]
        readings, rev_out, cost_out, score = [], [], [], []
        for s in range(s_):             # canonical order: _GAUGE_ROLES, then one lagged price per rival
            vals = [float(orders[s]), float(active[s]), round(100.0 * rating[s], 1),
                    float(sum(cmpl[o] for o in k.fo[s]) if k.bundle else cmpl[s]), float(uns[s]),
                    round(vol * math.exp(k.vol_w * (us[2 * k.J + s] - 0.5)), 1)]
            for f in k.rivals[s]:           # on the seat's own price dial; -1 when that rival was not in the market
                lp = lagged[f]
                d = min(100.0, max(0.0, (lp - k.p_lo) * k.inv_span))
                vals.append(round(100.0 - d if k.flip_price else d, 2) if lp >= 0.0 else -1.0)
            readings.append({name: vals[i] for name, i in k.gauge_plan})
            c = k.overhead + k.c_q * effort[s] + spend[s] + k.c_unit * units[s]
            if k.l_cap is not None:
                c += k.c_cap * cap[s]
            if nref is not None:
                c += refr[s] * sum(nref[o] for o in k.fo[s])
            c = float(round(c))
            rev_out.append(float(revenue[s]))
            cost_out.append(c)
            score.append(float(revenue[s]) - c)
        tot0 = state["totals"].tolist()
        totals = np.array([tot0[0] + credited, tot0[1] + sum(revenue), tot0[2] + sum(revenue[:s_]), tot0[3] + 1.0])
        new_state = {"traits": tr.copy(), "budget": budget, "aware": aware, "mem_q": mem_q, "cur": cur,
                     "tenure": tenure, "firm": firm, "price_hist": hist, "totals": totals}
        if mem_w is not None:
            new_state["mem_w"] = mem_w
        if tried is not None:
            new_state["tried"] = tried
        if happy is not None:
            new_state["happy"] = happy
        if k.commons:                       # the shared attention stock: regenerates, depleted by all outreach
            nxt = pool + k.pool_regen * (1.0 - pool) - k.pool_deplete * pool * sum(spend) * k.inv_pool
            new_state["pool"] = np.array([min(1.0, max(0.02, nxt))])
        info = {"t": int(t), "price": price_now, "quality": qual_now, "capacity": cap_now, "alive": alive_now,
                "orders": orders, "served": units, "unserved": uns, "active": active, "rating": rating,
                "revenue_all": revenue, "volume": vol, "pool": pool, "income_credited": credited,
                "events": events}
        return StepResult(new_state, readings, rev_out, cost_out, score, info)

    # ------------------------------------------------------------------ private description
    def truth(self) -> dict:
        return _truth(self)


# ====================================================================== the description of the mechanism
def _truth(eng: AbmEngine) -> dict:
    k, cfg = eng._k, eng.config
    nodes: list[dict] = []
    edges: list[dict] = []
    seen: set[str] = set()

    def node(i: str, role: str, seat: int | None = None, public: str | None = None) -> str:
        if i not in seen:
            seen.add(i)
            nodes.append({"id": i, "role": role, "seat": seat, "public": public})
        return i

    def edge(src: str, dst: str, sign: int, lag: int, kind: str) -> None:
        # signs are written below in natural terms; out of a lever they are reported in terms of its dial
        edges.append({"src": src, "dst": dst, "sign": int(sign) * turn.get(src, 1), "lag": int(lag), "kind": kind})

    turn = {f"s{s}.{role}": -1 for s in range(k.S) for role, fl in k.flip.items() if fl}
    dial_p = -1 if k.flip_price else 1      # rivals' prices are read on the seat's own price dial

    lever_roles = [role for _, role in k.lever_pairs]
    lname = {role: name for name, role in k.lever_pairs}
    gname = {role: name for name, role in k.gauge_pairs}
    segs = cfg["pop"]["segments"]
    two_rules = len(segs) > 1
    node("budget", "latent")
    inc_ids = [f"inc{j}." for j in range(k.J)]
    for j, q in enumerate(inc_ids):         # each incumbent: share of volume -> its price rule (-> its capacity rule)
        node(q + "share", "latent")
        node(q + "price", "latent")
    for j, q in enumerate(inc_ids):
        edge(q + "share", q + "price", +1, 1, "incumbent_price_rule")
        edge(q + "price", q + "share", -1, 0, "own_price")
        if k.finite_cap:
            node(q + "capacity", "latent")
            edge(q + "share", q + "capacity", +1, 1, "incumbent_capacity_rule")
            edge(q + "capacity", q + "share", +1, 1, "incumbent_service")
        for j2, q2 in enumerate(inc_ids):
            if j2 != j:
                edge(q + "price", q2 + "share", +1, 0, "substitution")
                if k.nfx:
                    edge(q + "share", q2 + "share", -1, 1, "rival_installed_base")
    if k.commons:
        node("pool", "latent")
    if k.bundle:
        node("bundle.awareness", "latent")
        node("bundle.quality", "latent")
        node("bundle.orders", "latent")
        edge("bundle.awareness", "bundle.orders", +1, 0, "consideration")
    for s in range(k.S):
        p = f"s{s}."
        for role in lever_roles:
            node(p + role, "lever", s, lname[role])
        aw_nodes = [p + "awareness"] if not two_rules else [p + "awareness_a", p + "awareness_b"]
        for a in aw_nodes:
            node(a, "latent", s)
        node(p + "satisfaction", "latent", s)
        node(p + "served", "latent", s)
        for role in _GAUGE_ROLES:
            node(p + "g." + role, "gauge", s, gname[role])
        for i, f in enumerate(k.rivals[s]):
            node(p + f"g.rival_price_{i}", "gauge", s, gname[f"rival_price_{i}"])
        node(p + "net", "outcome", s)
        g = p + "g."
        target = "bundle.orders" if k.bundle else g + "orders"
        # --- awareness
        if not two_rules:
            edge(p + "outreach", p + "awareness", +1, 0, "outreach_weak" if k.wom else "outreach")
        else:
            edge(p + "outreach", p + "awareness_a", +1, 0, "outreach")
            edge(p + "outreach", p + "awareness_b", +1, 0, "outreach_weak")
        for a in aw_nodes:
            if k.bundle:
                edge(a, "bundle.awareness", 0, 0, "and_gate")
                edge(a, g + "orders", +1, 0, "solo_consideration")
            else:
                edge(a, g + "orders", +1, 0, "consideration")
        if k.wom:
            node(p + "advocates", "latent", s)
            edge(p + "satisfaction", p + "advocates", +1, 0, "satisfied_customers")
            edge(p + "served", p + "advocates", +1, 0, "satisfied_customers")
            edge(p + "advocates", aw_nodes[-1], +1, 1, "network_tell")
            edge(p + "referral", aw_nodes[-1], +1, 0, "tell_rate")
            edge(p + "referral", p + "net", -1, 0, "cost_per_referred_first_purchase")
        # --- choice, service, experience
        edge(p + "price", target, -1, 0, "choice")
        if k.bundle:
            edge("bundle.orders", g + "orders", +1, 0, "bundle_sale")
            edge(p + "price", g + "orders", -1, 0, "solo_choice")
            edge(p + "quality", "bundle.quality", 0, 0, "weakest_link_with_mismatch_penalty")
            edge("bundle.quality", p + "satisfaction", +1, 0, "experience")
            edge(p + "satisfaction", "bundle.orders", +1, 1, "memory")
        edge(p + "quality", p + "satisfaction", +1, 0, "experience")
        edge(p + "satisfaction", g + "rating", +1, 0, "reviews")
        edge(p + "satisfaction", g + "complaints", -1, 0, "bad_experience")
        edge(p + "satisfaction", g + "orders", +1, 1, "memory")
        edge(g + "orders", p + "served", +1, 0, "service")
        edge(p + "served", g + "active", +1, 0, "customer_base")
        edge(g + "active", g + "orders", +1, 1, "habit")
        edge(p + "served", g + "volume", +1, 0, "market_total")
        edge(p + "served", g + "complaints", +1, 0, "exposure")
        edge(p + "served", g + "rating", 0, 0, "sample_size")
        edge(p + "served", "budget", -1, 0, "payment")
        edge(p + "price", "budget", -1, 0, "payment")
        edge("budget", g + "orders", +1, 1, "affordability")
        if k.finite_cap:
            node(p + "wait_memory", "latent", s)
            edge(p + "capacity", p + "served", +1, 0, "capacity")
            edge(p + "capacity", g + "queue", -1, 0, "capacity")
            edge(g + "orders", g + "queue", +1, 0, "rationing")
            edge(g + "queue", g + "complaints", +1, 0, "turned_away")
            edge(g + "queue", g + "active", -1, 0, "wait_churn")
            edge(g + "queue", p + "wait_memory", +1, 0, "remembered_wait")
            edge(p + "wait_memory", g + "orders", -1, 1, "remembered_wait")
            edge(p + "capacity", p + "net", -1, 0, "cost")
        if k.gate:
            node(p + "reviews", "latent", s)
            node(p + "trial_gate", "latent", s)
            edge(p + "served", p + "reviews", +1, 0, "review_count")
            edge(p + "reviews", p + "trial_gate", 0, 1, "count_threshold")
            edge(g + "rating", p + "trial_gate", 0, 1, "rating_threshold")
            edge(p + "trial_gate", g + "orders", +1, 0, "gated_trial")
        if k.nfx:
            node(p + "neighbour_share", "latent", s)
            edge(g + "active", p + "neighbour_share", +1, 1, "installed_base")
            edge(p + "neighbour_share", g + "orders", +1, 0, "network_value")
        if k.commons:
            edge(p + "outreach", "pool", -1, 1, "attention_depletion")
            for a in aw_nodes:
                edge("pool", a, +1, 0, "outreach_effectiveness")
            edge("pool", g + "orders", +1, 0, "engagement")
        # --- money
        edge(p + "price", p + "net", +1, 0, "unit_revenue")
        edge(p + "served", p + "net", 0, 0, "unit_margin")
        edge(p + "quality", p + "net", -1, 0, "cost")
        edge(p + "outreach", p + "net", -1, 0, "cost")
        # --- incumbents
        for j, q in enumerate(inc_ids):
            edge(p + "served", q + "share", -1, 0, "substitution")
            edge(q + "price", target, +1, 0, "substitution")
            edge(q + "price", g + f"rival_price_{k.rivals[s].index(k.S + j)}", dial_p, k.L, "public_lagged")
            if k.finite_cap:            # rationed orders are buyers taken from rivals but not served
                edge(g + "orders", q + "share", -1, 0, "diversion")
                edge(g + "queue", q + "share", +1, 0, "smaller_market_total")
                edge(q + "capacity", g + "orders", -1, 1, "substitution")
            if k.nfx:
                edge(q + "share", g + "orders", -1, 1, "rival_installed_base")
        if k.finite_cap and k.J:
            edge(g + "orders", g + "volume", -1, 0, "diversion_when_rationed")
        # --- the other seat
        for o in range(k.S):
            if o == s:
                continue
            q = f"s{o}."
            slot = k.rivals[o].index(s)
            edge(p + "price", q + f"g.rival_price_{slot}", dial_p, k.L, "public_lagged")
            if not k.bundle:
                edge(p + "price", q + "g.orders", +1, 0, "substitution")
                for a in aw_nodes:
                    edge(a, q + "g.orders", -1, 0, "substitution")
                edge(p + "satisfaction", q + "g.orders", -1, 1, "substitution")
                if k.nfx:
                    edge(g + "active", q + "g.orders", -1, 1, "rival_installed_base")
    notes = [
        "Dials: every lever is 0..100 and maps linearly to its natural range; a lever marked flip in `roles` runs "
        "backwards (100 is the bottom of the range). Signs of edges out of a lever node are in terms of its dial.",
        "Signs are direct effects with the other parents of the destination held fixed; sign 0 marks a gated or "
        "state-dependent relation (thresholds, and-gates, the unit margin, the review sample size).",
        "Lags are ticks between a change in the source and the first possible change in the destination.",
        "Incumbents exit when their smoothed profit stays below a loss threshold and a replacement with freshly "
        "drawn price and quality enters after a fixed delay; these regime changes are endogenous (state-triggered), "
        "so `regimes` lists no calendar ticks.",
        "Distractor levers (role noop*) have no outgoing edge: they change nothing, bit for bit.",
    ]
    if two_rules:
        notes.append("Segments: " + "; ".join(f"{sg.get('name', i)} share {sg['share']}"
                                              for i, sg in enumerate(segs)))
    return {"engine": "abm", "family": cfg.get("family"), "nodes": nodes, "edges": edges, "regimes": [],
            "motifs": list(cfg.get("motifs", [])), "notes": " ".join(notes), "chain": dict(_CHAINS[cfg["kind"]]),
            "roles": eng.roles(0), "calibration": CALIBRATION}


# ====================================================================== families
def _family(name: str, kind: str, split: str, transfer: str, doc: str) -> None:
    ks = _KINDS[kind]

    def build(rng: np.random.Generator, overrides: dict) -> dict:
        return {**(overrides or {}), "kind": kind}

    build.__name__ = f"build_{name}"
    mech_family(name, "abm", split, transfer, doc, motifs=tuple(ks["motifs"]), seats=int(ks["seats"]))(build)


_family("abm_utility", "utility", "train", "base",
        "Awareness from own outreach (forgotten at a constant rate); trial by a private utility comparison of "
        "known firms; retention by remembered own experience; no social feedback. Incumbents adapt price.")
_family("abm_wom", "wom", "train", "base",
        "Awareness spreads only along network edges from satisfied current customers (outreach is weak; a "
        "referral reward raises the telling rate); retention by experience; reinforcing loop quality -> "
        "satisfied customers -> awareness -> customers.")
_family("abm_congestion", "congestion", "train", "base",
        "Awareness from outreach; service rationed by capacity; unserved customers pay nothing, complain, lapse "
        "and remember the wait; balancing loop demand -> queue -> churn. Under-provisioning is the trap.")
_family("abm_trust_gate", "trust_gate", "val", "structural",
        "Customers will not try a firm until its public rating passes a personal threshold backed by enough "
        "reviews (few pioneers: cold start); then habit with switching costs. Thresholded reinforcing loop "
        "reviews -> rating -> trial -> reviews.")
_family("abm_network_fx", "network_fx", "test", "structural",
        "A customer's value from a firm rises with the share of their neighbours using it; lock-in and path "
        "dependence: the market starts open, early penetration pricing can pay and late price cuts do not.")
_family("abm_segments", "segments", "test", "compositional",
        "Two customer segments in one market: one obeys the abm_utility rule (outreach reaches it, neighbours "
        "do not), the other the abm_wom rule (only satisfied neighbours reach it).")
_family("abm_duopoly", "duopoly", "train", "base",
        "Two seats under the abm_utility rule fight for the same customers (competitive).")
_family("abm_commons", "commons", "val", "structural",
        "Two seats share one congestible stock of customer attention: outreach by either depletes it for both "
        "(less effective outreach, less frequent shopping) while they still compete for customers (mixed motive).")
_family("abm_complement", "complement", "val", "structural",
        "Customers need a compatible bundle of one component from each seat: they must know both, the bundle is "
        "worth its weaker component less a mismatch penalty, and they pay the sum of both prices (cooperative).")
_family("abm_duopoly_fx", "duopoly_fx", "test", "structural",
        "Two seats compete under local network effects: installed base raises value, so early share locks in.")
