"""Engine `strategic`: multi-agent strategic simulation (Part 7.3 "multi-agent strategic simulations"; Part 8.8 and
8.9).

The world is a repeated interaction between the participants and a population of simulated agents, each running a
policy from a named policy family with private parameters: a propensity learner that credits each choice with the
profit of the period it was made and the next (so retaliation teaches it), a price follower, a trigger strategy, a
fictitious-play contestant that best-responds to a moving average of what it has seen, an attacker that learns route
values from its own outcomes, a regret-matching player, time-dependent and tit-for-tat negotiators, incumbents that
fight or accommodate (one of them learns that fighting a persistent entrant does not pay), and workers whose turnout
depends on how they are paid. Each tick is one stage game: the simulated agents choose from their state, the
participants' actions resolve against theirs, payoffs are booked, and every simulated agent updates its state by its
own learning rule from what it observed. Some worlds are purely cooperative: several participants hold private
information or capabilities and are paid only for joint commitments (`coordinate`) or combined output.

This is a different transition structure from every other engine: the state that matters is other agents' policy
state (propensities, regrets, beliefs, aspiration levels, trust), and it changes in response to the participant.
Every period's random draws are taken up front with a fixed shape, so copies of a world see the same noise whatever
they do; joint-plan outcomes draw from the action stream.

Scenarios (one per family):

    price_war    repeated price competition against a propensity learner and a follower or trigger rival (8.8.1)
    entry        market entry against incumbents that fight, accommodate, or fight until the entrant has persisted;
                 pre-entry prices signal their cost (8.8.2)
    bargain      repeated multi-issue negotiations with counterparts of private issue weights and reservation
                 levels, time-dependent or tit-for-tat concession (8.8.3)
    blotto       contests for prizes at several sites against a fictitious-play opponent with private valuations
                 and budget (resource competition, 8.8.4)
    interdiction inspection capacity over routes against an attacker that learns from its outcomes and substitutes
                 between routes (8.8.5)
    learning     a repeated matrix game against a regret-matching opponent whose strategy evolves (8.8.6)
    profile      three participants each see one attribute of every option; only a joint commitment pays (8.9.2)
    commons      three participants harvest renewable patches with crowding and hidden tipping points (8.9.3)
    discovery    two participants own ingredients of hidden types; only joint tests of matching types yield, and
                 private assays label each side's ingredients in its own private labels (8.9.5)
    formation    a lead pays workers of hidden skills, reservation pay and reliability; a task succeeds only when
                 every required skill turns up (8.9.6)
"""
from __future__ import annotations

import math

import numpy as np

from .. import actions as A
from .base import ActionError, RapidEngine, register

SCENARIOS = ("price_war", "entry", "bargain", "blotto", "interdiction", "learning", "profile", "commons", "discovery",
             "formation")


def _softmax(v, beta=1.0):
    m = max(v)
    e = [math.exp(beta * (x - m)) for x in v]
    s = sum(e)
    return [x / s for x in e]


def _pick(p, u):
    c = 0.0
    for k, x in enumerate(p):
        c += x
        if u < c:
            return k
    return len(p) - 1


def contest_br(values, others, budget, r, steps=20):
    """Greedy best response in a Tullock contest: `budget` in `steps` increments, each to the site with the largest
    marginal expected prize values[k] * y^r / (y^r + others[k]^r)."""
    K = len(values)
    y = [0.0] * K
    inc = budget / steps

    def win(k, yk):
        if yk <= 0:
            return 0.0
        o = others[k]
        return values[k] * yk ** r / (yk ** r + o ** r) if o > 0 else values[k]
    for _ in range(steps):
        gains = [win(k, y[k] + inc) - win(k, y[k]) for k in range(K)]
        k = int(np.argmax(gains))
        y[k] += inc
    return y


def concede_offer(v, aspiration):
    """The offer a negotiator with issue weights v makes at aspiration level `aspiration` (its own utility): it
    gives the other side the issues it values least first. Returns the other side's shares per issue."""
    order = sorted(range(len(v)), key=lambda k: (v[k], k))
    give = max(0.0, 1.0 - aspiration)                  # utility the negotiator is willing to give up
    y = [0.0] * len(v)
    for k in order:
        if give <= 1e-12:
            break
        x = min(1.0, give / v[k]) if v[k] > 0 else 1.0
        y[k] = x
        give -= x * v[k]
    return y


@register
class Strategic(RapidEngine):
    kind = "strategic"
    SECRET_KEYS = ("rival_cost", "rival_kind", "hi", "cut", "follow", "gap", "trigger", "eta", "temp", "sens",
                   "inc_kind", "inc_cost", "patience", "cp_v", "cp_r0", "cp_kind", "cp_e", "cp_zeta", "opp_value",
                   "opp_budget", "decisive", "memory", "att_util", "penalty", "kappa", "B", "forget", "w",
                   "value_sd", "Kp", "rp", "allee", "crowd", "shock_sd", "typeA", "typeB", "qualA", "qualB", "labelA",
                   "labelB", "skills", "reservation", "reliability")

    def __init__(self, config: dict):
        super().__init__(config)
        self.sc = config["scenario"]
        if self.sc not in SCENARIOS:
            raise ValueError(f"unknown scenario {self.sc}")

    # ================================================================== public surface
    def action_space(self, agent: str, state: dict | None = None) -> A.ActionSpace:
        c = self.config
        sc = self.sc
        kinds: dict = {}
        limits: dict = {}
        q = {"tools": "the action kinds available to you", "history": "your last observations (args: n)",
             "spec": "the public description of this world"}
        others = [a for a in self.agents if a != agent]
        if sc == "price_war":
            kinds["levers"] = A.levers_schema([{"name": c["price_name"], "min": float(c["p_min"]),
                                                "max": float(c["p_max"]), "default": float(c["p_default"])}])
        elif sc == "entry":
            decs = {m: ["stay_out", "enter", "exit"] for m in c["market_names"]}
            kinds["discrete"] = A.discrete_schema(decs)
            limits["decisions"] = decs
        elif sc == "bargain":
            kinds["levers"] = A.levers_schema([{"name": n, "min": 0.0, "max": 1.0, "default": 1.0}
                                               for n in c["issue_names"]])
            kinds["discrete"] = A.discrete_schema({"respond": ["propose", "accept", "walk_away"]})
            limits["decisions"] = {"respond": ["propose", "accept", "walk_away"]}
        elif sc == "blotto":
            kinds["allocate"] = A.allocate_schema(c["site_names"], float(c["budget"]), "contest effort this period")
            limits["allocate_total"] = float(c["budget"])
        elif sc == "interdiction":
            kinds["allocate"] = A.allocate_schema(c["route_names"], float(c["inspectors"]), "inspectors deployed")
            limits["allocate_total"] = float(c["inspectors"])
        elif sc == "learning":
            kinds["discrete"] = A.discrete_schema({"play": list(c["option_names"])})
            limits["decisions"] = {"play": list(c["option_names"])}
        elif sc == "profile":
            kinds["message"] = A.message_schema(others)
            kinds["coordinate"] = A.coordinate_schema(others, {"option": A.S.s_enum(list(c["slot_names"]),
                                                                                    "the option chosen jointly")})
            limits["parties"] = others
        elif sc == "commons":
            opts = ["rest"] + list(c["patch_names"])
            kinds["discrete"] = A.discrete_schema({"ground": opts})
            limits["decisions"] = {"ground": opts}
            kinds["levers"] = A.levers_schema([{"name": "effort", "min": 0.0, "max": float(c["e_max"]),
                                                "default": 0.0}])
            kinds["message"] = A.message_schema(others)
        elif sc == "discovery":
            i = self.agents.index(agent)
            mine = c["ingr_a"] if i == 0 else c["ingr_b"]
            q["assay"] = (f"the private class label of one of your own ingredients (args: {{\"ingredient\": name}}); "
                          f"costs {c['assay_cost']}; your ingredients: {', '.join(mine)}")
            kinds["message"] = A.message_schema(others)
            kinds["coordinate"] = A.coordinate_schema(others, {
                "a": A.S.s_enum(list(c["ingr_a"]), "ingredient of the first participant"),
                "b": A.S.s_enum(list(c["ingr_b"]), "ingredient of the second participant")})
            limits["parties"] = others
        elif sc == "formation":
            kinds["allocate"] = A.allocate_schema(c["worker_names"], float(c["budget"]), "pay offered this period")
            limits["allocate_total"] = float(c["budget"])
        kinds["query"] = A.query_schema(q)
        if sc in ("bargain", "commons", "entry"):
            kinds["sequence"] = A.sequence_schema(sorted(k for k in kinds if k not in ("sequence", "program")))
        return A.ActionSpace(kinds, limits)

    def brief(self, agent: str) -> str:
        c = self.config
        b = str(c.get("brief", ""))
        if self.sc == "bargain":
            b += " Your weights on the issues: " + ", ".join(f"{n} {w:g}" for n, w in zip(c["issue_names"], c["my_w"]))
        if self.sc == "learning":
            rows = "; ".join(f"{o}: " + ", ".join(f"vs {p} {c['A'][i][j]:g}" for j, p in enumerate(c["opp_names"]))
                             for i, o in enumerate(c["option_names"]))
            b += f" Your payoff per period (cents x {c['scale']:g}): {rows}."
        return b

    # ================================================================== state
    def init_state(self, rng) -> dict:
        c = self.config
        sc = self.sc
        if sc == "price_war":
            p0 = float(c["p_default"])
            return {"p": [p0, float(c["hi"][0]), float(c["hi"][1])], "last": [p0, float(c["hi"][0]), float(c["hi"][1])],
                    "V": [list(map(float, c["v0"])), list(map(float, c["v0"]))], "prev_mode": [-1, -1],
                    "prev_profit": [0.0, 0.0], "punish_left": [0, 0], "modes": [0, 0], "cuts": [0, 0]}
        if sc == "entry":
            M = len(c["market_names"])
            return {"active": [0] * M, "since": [0] * M, "pend_cost": 0.0, "entries": [0] * M, "fought": [0] * M}
        if sc == "bargain":
            st = {"x": [1.0] * len(c["issue_names"]), "deal": 0, "round": 0, "resp": None, "deals_made": 0,
                  "pend_rev": 0.0, "last_prop_v": None, "util": []}
            self._new_deal(st)
            return st
        if sc == "blotto":
            K = len(c["site_names"])
            return {"x": [0.0] * K, "xbar": [float(c["budget"]) / K] * K, "y": [0.0] * K}
        if sc == "interdiction":
            N = len(c["route_names"])
            return {"x": [0.0] * N, "Q": [float(v) for v in c["att_util"]], "tries": [0] * N}
        if sc == "learning":
            return {"R": [0.0, 0.0, 0.0], "me": 0, "counts": [[0] * 3 for _ in range(3)]}
        if sc == "profile":
            K = len(c["slot_names"])
            return {"F": [[0.0] * 3 for _ in range(K)], "fresh": 0, "open": 0, "decisions": 0}
        if sc == "commons":
            return {"S": [float(k) * float(f) for k, f in zip(c["Kp"], c["s0_frac"])], "ground": [-1] * 3,
                    "effort": [0.0] * 3, "collapsed": [0] * len(c["patch_names"])}
        if sc == "discovery":
            return {"tests": 0, "pend_cost": [0.0, 0.0], "pend_rev": [0.0, 0.0], "found": []}
        # formation
        W = len(c["worker_names"])
        return {"pay": [0.0] * W, "trust": [1.0] * W, "task": list(c["task0"]), "done": 0}

    # ================================================================== actions
    def apply(self, state, agent, action, rng, t):
        c = self.config
        sc = self.sc
        k = action["kind"]
        i = self.agents.index(agent)
        if sc == "price_war" and k == "levers":
            if c["price_name"] in action["values"]:
                state["p"][0] = float(action["values"][c["price_name"]])
            return {"price": state["p"][0]}
        if sc == "entry" and k == "discrete":
            m = c["market_names"].index(action["decision"])
            ch = action["choice"]
            if ch == "enter" and not state["active"][m]:
                state["active"][m] = 1
                state["since"][m] = 0
                state["entries"][m] += 1
                state["pend_cost"] += float(c["entry_cost"])
            elif ch == "exit" and state["active"][m]:
                state["active"][m] = 0
            return {"market": action["decision"], "active": bool(state["active"][m])}
        if sc == "bargain":
            if k == "levers":
                for j, n in enumerate(c["issue_names"]):
                    if n in action["values"]:
                        state["x"][j] = float(action["values"][n])
                return {"terms": {n: x for n, x in zip(c["issue_names"], state["x"])}}
            if k == "discrete":
                state["resp"] = action["choice"]
                return {"response": action["choice"]}
        if sc in ("blotto", "interdiction", "formation") and k == "allocate":
            names = c["site_names"] if sc == "blotto" else (c["route_names"] if sc == "interdiction"
                                                            else c["worker_names"])
            vals = [float(action["amounts"].get(n, 0.0)) for n in names]
            key = "pay" if sc == "formation" else "x"
            state[key] = vals
            return {"allocated": {n: v for n, v in zip(names, vals)}}
        if sc == "learning" and k == "discrete":
            state["me"] = c["option_names"].index(action["choice"])
            return {"play": action["choice"]}
        if sc == "commons":
            if k == "discrete":
                ch = action["choice"]
                state["ground"][i] = -1 if ch == "rest" else c["patch_names"].index(ch)
                return {"ground": ch}
            if k == "levers":
                if "effort" in action["values"]:
                    state["effort"][i] = float(action["values"]["effort"])
                return {"effort": state["effort"][i]}
        if sc == "discovery" and k == "query" and action["target"] == "assay":
            name = str((action.get("args") or {}).get("ingredient", ""))
            mine = c["ingr_a"] if i == 0 else c["ingr_b"]
            if name not in mine:
                raise ActionError("assay needs args {\"ingredient\": <one of your own ingredients>}")
            j = mine.index(name)
            state["pend_cost"][i] += float(c["assay_cost"])
            lab = c["labelA"][c["typeA"][j]] if i == 0 else c["labelB"][c["typeB"][j]]
            return {"ingredient": name, "class": lab, "cost": float(c["assay_cost"])}
        raise ActionError(f"{k} is not available in this world")

    def coordinate(self, state, plan, parties, rng, t):
        c = self.config
        if sorted(parties) != sorted(self.agents):
            raise ActionError("a joint plan needs every participant")
        n = len(self.agents)
        if self.sc == "profile":
            if not state["open"]:
                raise ActionError("there is no open set of options (the last one was decided)")
            k = c["slot_names"].index(plan["option"])
            F = state["F"][k]
            val = float(c["scale"]) * (sum(w * f for w, f in zip(c["w"], F)) + float(c["value_sd"])
                                       * float(rng.standard_normal()))
            state["open"] = 0
            state["decisions"] += 1
            state["pend"] = val
            return {"option": plan["option"], "payoff": round(val, 6)}
        if self.sc == "discovery":
            a, b = c["ingr_a"].index(plan["a"]), c["ingr_b"].index(plan["b"])
            match = c["typeA"][a] == c["typeB"][b]
            y = float(c["base_yield"]) * float(c["qualA"][a]) * float(c["qualB"][b]) if match else 0.0
            y = max(0.0, y + float(c["yield_sd"]) * float(rng.standard_normal()))
            state["tests"] += 1
            for p in range(n):
                state["pend_rev"][p] += float(c["unit_price"]) * y / n
                state["pend_cost"][p] += float(c["try_cost"]) / n
            return {"a": plan["a"], "b": plan["b"], "yield": round(y, 6)}
        raise ActionError("this world has no joint plans")

    # ================================================================== dynamics
    def advance(self, st, rng, t):
        return getattr(self, "_adv_" + self.sc)(st, rng, t)

    # ---------------------------------------------------------------- 8.8.1
    def rival_prices(self, st, u) -> tuple[list[float], list[int]]:
        c = self.config
        last = st["last"]
        prices, modes = [], []
        for r in range(2):
            kind = c["rival_kind"][r]
            rc = float(c["rival_cost"][r])
            floor = rc + 0.5
            mode = 0
            if kind == "propensity":
                d = float(c["temp"]) * (st["V"][r][1] - st["V"][r][0])
                pu = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, d))))
                mode = 1 if u[r] < pu else 0
                p = float(c["hi"][r]) if mode == 0 else max(floor, min(last[0], last[2 - r]) - float(c["cut"]))
            elif kind == "follower":
                p = last[1 + r] + float(c["follow"]) * (last[0] - float(c["gap"]) - last[1 + r])
                p = min(float(c["p_max"]), max(floor, p))
            else:                                               # trigger
                if st["punish_left"][r] > 0:
                    st["punish_left"][r] -= 1
                    p, mode = floor, 1
                elif last[0] < float(c["trigger"]):
                    st["punish_left"][r] = int(c["punish"]) - 1
                    p, mode = floor, 1
                else:
                    p = float(c["hi"][r])
            prices.append(round(p, 2))                       # posted prices are in cents
            modes.append(mode)
        return prices, modes

    def logit_units(self, p, M):
        c = self.config
        e = [math.exp(float(c["appeal"]) - float(c["sens"]) * x) for x in p]
        den = 1.0 + sum(e)
        return [M * x / den for x in e]

    def _adv_price_war(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        u = rng.random(2)
        z = float(rng.standard_normal())
        rp, modes = self.rival_prices(st, u)
        p = [st["p"][0]] + rp
        M = float(c["market"]) * max(0.2, 1.0 + float(c["market_sd"]) * z)
        q = self.logit_units(p, M)
        costs = [float(c["cost"])] + [float(x) for x in c["rival_cost"]]
        prof = [(p[i] - costs[i]) * q[i] for i in range(3)]
        for r in range(2):
            if c["rival_kind"][r] != "propensity":
                continue
            pm = st["prev_mode"][r]
            if pm >= 0:
                cred = 0.5 * (st["prev_profit"][r] + prof[1 + r])
                st["V"][r][pm] += float(c["eta"]) * (cred - st["V"][r][pm])
            st["prev_mode"][r] = modes[r]
            st["prev_profit"][r] = prof[1 + r]
            st["cuts"][r] += modes[r]
        st["modes"] = modes
        st["last"] = list(p)
        st["p"][1], st["p"][2] = rp
        nums = {f"{n}.price": rp[k] for k, n in enumerate(c["rival_names"])}
        nums["my_units"] = q[0]
        nums["my_price"] = p[0]
        return {"obs": {a: {"numbers": nums}}, "revenue": {a: p[0] * q[0]}, "cost": {a: costs[0] * q[0]},
                "info": {}}

    # ---------------------------------------------------------------- 8.8.2
    def fight_prob(self, st, m) -> float:
        c = self.config
        kind = c["inc_kind"][m]
        if kind == "tough":
            return float(c["p_fight_tough"])
        if kind == "soft":
            return float(c["p_fight_soft"])
        return float(c["p_fight_tough"]) if st["since"][m] < int(c["patience"][m]) else float(c["p_fight_soft"])

    def _adv_entry(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        M = len(c["market_names"])
        u = rng.random(M)
        z = rng.standard_normal(M)
        rev = 0.0
        cost = st["pend_cost"]
        st["pend_cost"] = 0.0
        nums = {}
        for m, n in enumerate(c["market_names"]):
            base = float(c["inc_cost"][m]) + float(c["markup"])
            if st["active"][m]:
                fight = u[m] < self.fight_prob(st, m)
                st["fought"][m] += int(fight)
                r_ = float(c["rev_fight"][m]) if fight else float(c["rev_acc"][m])
                cst = float(c["op_cost"][m])
                rev += r_
                cost += cst
                price = base * (0.75 if fight else 1.0)
                st["since"][m] += 1
                nums[f"{n}.my_profit"] = r_ - cst
            else:
                price = base
                nums[f"{n}.my_profit"] = 0.0
            nums[f"{n}.incumbent_price"] = price + float(c["price_sd"]) * float(z[m])
            nums[f"{n}.in_market"] = float(st["active"][m])
        return {"obs": {a: {"numbers": nums}}, "revenue": {a: rev}, "cost": {a: cost}, "info": {}}

    # ---------------------------------------------------------------- 8.8.3
    def cp_aspiration(self, st) -> float:
        c = self.config
        j = st["cp"]
        r0 = float(c["cp_r0"][j])
        if c["cp_kind"][j] == "boulware":
            K = int(c["rounds"])
            return r0 + (1.0 - r0) * (1.0 - (min(st["round"], K) / K) ** (1.0 / float(c["cp_e"][j])))
        return max(r0, st["asp"])

    def _new_deal(self, st):
        c = self.config
        st["cp"] = int(c["schedule"][st["deal"] % len(c["schedule"])])
        st["round"] = 0
        st["asp"] = float(c["asp0"])
        st["last_prop_v"] = None
        st["offer"] = concede_offer(c["cp_v"][st["cp"]], self.cp_aspiration(st))

    def my_utility(self, x) -> float:
        return float(sum(w * v for w, v in zip(self.config["my_w"], x)))

    def cp_utility(self, j, x) -> float:
        return float(sum(v * (1.0 - xx) for v, xx in zip(self.config["cp_v"][j], x)))

    def _adv_bargain(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        z = float(rng.standard_normal())
        rev = 0.0
        cost = float(c["round_cost"])
        resp = st["resp"]
        st["resp"] = None
        outcome = "open"
        j = st["cp"]
        if resp == "accept":
            rev = float(c["deal_value"]) * self.my_utility(st["offer"])
            outcome = "agreed_their_terms"
        elif resp == "propose":
            v = self.cp_utility(j, st["x"])
            if v + float(c["accept_sd"]) * z >= self.cp_aspiration(st):
                rev = float(c["deal_value"]) * self.my_utility(st["x"])
                outcome = "agreed_your_terms"
            else:
                outcome = "rejected"
                if c["cp_kind"][j] == "tit_for_tat":
                    if st["last_prop_v"] is not None and v > st["last_prop_v"]:
                        st["asp"] = st["asp"] - float(c["cp_zeta"][j]) * (v - st["last_prop_v"])
                    st["last_prop_v"] = v if st["last_prop_v"] is None else max(v, st["last_prop_v"])
        elif resp == "walk_away":
            outcome = "walked_away"
        ended = outcome in ("agreed_their_terms", "agreed_your_terms", "walked_away")
        if ended:
            st["deals_made"] += int(outcome != "walked_away")
            st["util"].append(rev / float(c["deal_value"]))
        else:
            st["round"] += 1
            if st["round"] >= int(c["rounds"]):
                outcome = "deadline_passed"
                ended = True
                st["util"].append(0.0)
        if ended:
            st["deal"] += 1
            self._new_deal(st)
        else:
            st["offer"] = concede_offer(c["cp_v"][j], self.cp_aspiration(st))
        nums = {"round": float(st["round"]), "deals_made": float(st["deals_made"]),
                "offer_value_to_me": self.my_utility(st["offer"])}
        for k, n in enumerate(c["issue_names"]):
            nums[f"offer.{n}"] = st["offer"][k]
        ent = [{"kind": "counterpart", "id": c["cp_names"][st["cp"]], "round": st["round"], "last": outcome}]
        return {"obs": {a: {"numbers": nums, "entities": ent,
                            "text": [f"negotiating with {c['cp_names'][st['cp']]}; last: {outcome}"]}},
                "revenue": {a: rev}, "cost": {a: cost}, "info": {}}

    # ---------------------------------------------------------------- 8.8.4
    def opponent_allocation(self, st, zb: float) -> list[float]:
        c = self.config
        vals = [float(p) * float(v) for p, v in zip(c["prize"], c["opp_value"])]
        B = float(c["opp_budget"]) * max(0.3, 1.0 + float(c["budget_sd"]) * zb)
        return contest_br(vals, st["xbar"], B, float(c["decisive"]))

    def _adv_blotto(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        zb = float(rng.standard_normal())
        y = self.opponent_allocation(st, zb)
        r = float(c["decisive"])
        x = st["x"]
        rev = 0.0
        nums = {}
        for k, n in enumerate(c["site_names"]):
            if x[k] <= 0:
                s = 0.0
            elif y[k] <= 0:
                s = 1.0
            else:
                s = x[k] ** r / (x[k] ** r + y[k] ** r)
            rev += float(c["prize"][k]) * s
            nums[f"{n}.rival_effort"] = y[k]
            nums[f"{n}.my_share"] = s
        cost = float(c["unit_cost"]) * sum(x)
        m = float(c["memory"])
        st["xbar"] = [(1 - m) * xb + m * xx for xb, xx in zip(st["xbar"], x)]
        st["y"] = y
        st["x"] = [0.0] * len(x)
        return {"obs": {a: {"numbers": nums}}, "revenue": {a: rev}, "cost": {a: cost}, "info": {}}

    # ---------------------------------------------------------------- 8.8.5
    def attacker_mix(self, st) -> list[float]:
        return _softmax(st["Q"], float(self.config["att_beta"]))

    def _adv_interdiction(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        n = int(c["attempts"])
        ur = rng.random(n)
        uc = rng.random(n)
        p = self.attacker_mix(st)
        N = len(c["route_names"])
        caught, hits = [0] * N, [0] * N
        for k in range(n):
            j = _pick(p, float(ur[k]))
            pc = 1.0 - math.exp(-float(c["kappa"]) * st["x"][j])
            got = uc[k] < pc
            pay = -float(c["penalty"]) if got else float(c["att_util"][j])
            st["Q"][j] += float(c["eta"]) * (pay - st["Q"][j])
            st["tries"][j] += 1
            if got:
                caught[j] += 1
            else:
                hits[j] += 1
        rev = float(c["bounty"]) * sum(caught)
        cost = sum(float(c["damage"][j]) * hits[j] for j in range(N)) + float(c["deploy_cost"]) * sum(st["x"])
        nums = {}
        for j, nm in enumerate(c["route_names"]):
            nums[f"{nm}.caught"] = float(caught[j])
            nums[f"{nm}.got_through"] = float(hits[j])
        return {"obs": {a: {"numbers": nums}}, "revenue": {a: rev}, "cost": {a: cost}, "info": {}}

    # ---------------------------------------------------------------- 8.8.6
    def opponent_mix(self, st) -> list[float]:
        pos = [max(0.0, r) for r in st["R"]]
        s = sum(pos)
        return [x / s for x in pos] if s > 1e-12 else [1.0 / 3] * 3

    def _adv_learning(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        u = float(rng.random())
        p = self.opponent_mix(st)
        j = _pick(p, u)
        i = st["me"]
        B = c["B"]
        f = float(c["forget"])
        st["R"] = [f * st["R"][k] + (float(B[i][k]) - float(B[i][j])) for k in range(3)]
        st["counts"][i][j] += 1
        pay = float(c["A"][i][j]) * float(c["scale"])
        nums = {"opponent_played": float(j), "my_play": float(i), "my_payoff": pay}
        return {"obs": {a: {"numbers": nums, "text": [f"opponent played {c['opp_names'][j]}"]}},
                "revenue": {a: pay}, "cost": {a: float(c["fee"])}, "info": {}}

    # ---------------------------------------------------------------- 8.9.2
    def _adv_profile(self, st, rng, t):
        c = self.config
        ags = self.agents
        K = len(c["slot_names"])
        Z = rng.standard_normal((K, 3))
        pend = float(st.pop("pend", 0.0))
        n = len(ags)
        rev = {x: max(pend, 0.0) / n for x in ags}
        cost = {x: (max(-pend, 0.0) + (float(c["decision_cost"]) if pend != 0.0 else 0.0)) / n for x in ags}
        period = int(c["period"])
        if t % period == 0:
            st["F"] = [[float(Z[k, a_]) for a_ in range(3)] for k in range(K)]
            st["open"] = 1
            st["fresh"] = 1
        else:
            st["fresh"] = 0
        obs = {}
        for i, x in enumerate(ags):
            nums = {"options_open": float(st["open"]), "new_options": float(st["fresh"]),
                    "decisions_made": float(st["decisions"])}
            for k, sn in enumerate(c["slot_names"]):
                nums[f"{sn}.my_attribute"] = st["F"][k][i]
            obs[x] = {"numbers": nums}
        return {"obs": obs, "revenue": rev, "cost": cost, "info": {}}

    # ---------------------------------------------------------------- 8.9.3
    def _adv_commons(self, st, rng, t):
        c = self.config
        ags = self.agents
        P = len(c["patch_names"])
        z = rng.standard_normal(P)
        zs = rng.standard_normal(len(ags))
        catch = [0.0] * len(ags)
        boats = [0] * P
        for i in range(len(ags)):
            if st["ground"][i] >= 0 and st["effort"][i] > 0:
                boats[st["ground"][i]] += 1
        for p in range(P):
            who = [i for i in range(len(ags)) if st["ground"][i] == p and st["effort"][i] > 0]
            if not who:
                continue
            S = st["S"][p]
            want = [float(c["catchability"]) * st["effort"][i] * S / (1.0 + float(c["crowd"]) * (len(who) - 1))
                    for i in who]
            tot = sum(want)
            if tot > 0.9 * S:
                want = [w * 0.9 * S / tot for w in want]
            for i, w in zip(who, want):
                catch[i] = w
            st["S"][p] = S - sum(want)
        for p in range(P):
            S = st["S"][p]
            K, r, Al = float(c["Kp"][p]), float(c["rp"][p]), float(c["allee"][p])
            # logistic regrowth with a strong Allee effect: below the tipping point Al the stock shrinks
            g = r * S * (1.0 - S / K) * ((S - Al) / (S + Al)) * math.exp(float(c["shock_sd"]) * float(z[p])
                                                                        - 0.5 * float(c["shock_sd"]) ** 2)
            st["S"][p] = max(0.0, S + g)
            if st["S"][p] < 0.05 * K:
                st["collapsed"][p] = 1
        rev = {x: float(c["unit_price"]) * catch[i] for i, x in enumerate(ags)}
        cost = {x: float(c["effort_cost"]) * st["effort"][i] * (1.0 if st["ground"][i] >= 0 else 0.0)
                for i, x in enumerate(ags)}
        landed = float(sum(catch))
        obs = {}
        for i, x in enumerate(ags):
            g = st["ground"][i]
            nums = {"my_catch": catch[i], "fleet_landings": landed}
            if g >= 0:
                nums["my_ground_survey"] = max(0.0, st["S"][g] * (1.0 + float(c["survey_sd"]) * float(zs[i])))
                nums["boats_on_my_ground"] = float(boats[g])
            obs[x] = {"numbers": nums, "missing": [] if g >= 0 else ["my_ground_survey", "boats_on_my_ground"]}
        return {"obs": obs, "revenue": rev, "cost": cost, "info": {}}

    # ---------------------------------------------------------------- 8.9.5
    def _adv_discovery(self, st, rng, t):
        c = self.config
        ags = self.agents
        z = rng.standard_normal((2, len(c["ingr_a"])))
        rev = {x: st["pend_rev"][i] for i, x in enumerate(ags)}
        cost = {x: st["pend_cost"][i] for i, x in enumerate(ags)}
        st["pend_rev"], st["pend_cost"] = [0.0, 0.0], [0.0, 0.0]
        obs = {}
        for i, x in enumerate(ags):
            mine = c["ingr_a"] if i == 0 else c["ingr_b"]
            qual = c["qualA"] if i == 0 else c["qualB"]
            nums = {"joint_tests": float(st["tests"])}
            for j, nm in enumerate(mine):
                nums[f"{nm}.purity"] = float(qual[j]) + float(c["purity_sd"]) * float(z[i, j])
            obs[x] = {"numbers": nums}
        return {"obs": obs, "revenue": rev, "cost": cost, "info": {}}

    # ---------------------------------------------------------------- 8.9.6
    def draw_task(self, u) -> list:
        c = self.config
        S = len(c["skill_names"])
        need = [1 if float(u[k]) < float(c["p_need"]) else 0 for k in range(S)]
        if not any(need):
            need[int(float(u[S]) * S) % S] = 1
        val = float(c["task_value"]) * (0.5 + 0.5 * sum(need)) * (0.8 + 0.4 * float(u[S + 1]))
        return need + [val]

    def _adv_formation(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        W, S = len(c["worker_names"]), len(c["skill_names"])
        us = rng.random(W)
        ut = rng.random(S + 2)
        need, val = st["task"][:S], st["task"][S]
        acc, show = [0] * W, [0] * W
        paid = 0.0
        for w in range(W):
            pay = st["pay"][w]
            if pay <= 0:
                continue
            if pay >= float(c["reservation"][w]):
                acc[w] = 1
                if us[w] < float(c["reliability"][w]) * st["trust"][w]:
                    show[w] = 1
                    paid += pay
                ratio = pay / float(c["reservation"][w])
                if ratio >= 1.3:
                    st["trust"][w] = min(1.0, st["trust"][w] + float(c["trust_step"]))
                elif ratio < 1.1:
                    st["trust"][w] = max(0.5, st["trust"][w] - float(c["trust_step"]))
        covered = [any(show[w] and c["skills"][w][k] for w in range(W)) for k in range(S)]
        ok = all(covered[k] or not need[k] for k in range(S))
        rev = val if ok else 0.0
        st["done"] += int(ok)
        st["task"] = self.draw_task(ut)
        st["pay"] = [0.0] * W
        nums = {"task_completed": float(ok), "next_task_value": st["task"][S]}
        for k, sk in enumerate(c["skill_names"]):
            nums[f"next_needs.{sk}"] = float(st["task"][k])
        ents = [{"kind": "worker", "id": n, "accepted": bool(acc[w]), "turned_up": bool(show[w]),
                 "claimed_skills": list(c["claims"][w])} for w, n in enumerate(c["worker_names"])]
        return {"obs": {a: {"numbers": nums, "entities": ents}}, "revenue": {a: rev}, "cost": {a: paid},
                "info": {}}

    def history(self, state) -> dict:
        c = self.config
        if self.sc == "formation":
            S = len(c["skill_names"])
            need = ", ".join(sk for k, sk in enumerate(c["skill_names"]) if state["task"][k])
            return {"text": [f"first task needs: {need}; it pays {state['task'][S]:.0f}"],
                    "arrays": {"first_task_needs": [float(x) for x in state["task"][:S]],
                               "first_task_value": [float(state["task"][S])]}}
        return {}

    # ================================================================== evaluator side
    def truth(self) -> dict:
        c = self.config
        sc = self.sc
        nodes, edges, hidden = [], [], {}
        if sc == "price_war":
            nodes = ["a0", "rival0", "rival1", "market"]
            edges = [{"src": x, "dst": "market", "kind": "price", "lag": 0} for x in ("a0", "rival0", "rival1")]
            edges += [{"src": "a0", "dst": f"rival{r}", "kind": f"observes:{c['rival_kind'][r]}", "lag": 1}
                      for r in range(2)]
            hidden = {"rival_kind": c["rival_kind"], "rival_cost": c["rival_cost"]}
        elif sc == "entry":
            nodes = ["a0"] + [f"incumbent{m}" for m in range(len(c["market_names"]))]
            edges = [{"src": "a0", "dst": f"incumbent{m}", "kind": "entry", "lag": 0}
                     for m in range(len(c["market_names"]))]
            edges += [{"src": f"incumbent{m}", "dst": "a0", "kind": "response", "lag": 0}
                      for m in range(len(c["market_names"]))]
            hidden = {"inc_kind": c["inc_kind"]}
        elif sc == "bargain":
            nodes = ["a0"] + [f"counterpart{j}" for j in range(len(c["cp_names"]))]
            edges = [{"src": "a0", "dst": f"counterpart{j}", "kind": "offer", "lag": 0} for j in range(len(c["cp_names"]))]
            edges += [{"src": f"counterpart{j}", "dst": "a0", "kind": "counteroffer", "lag": 1}
                      for j in range(len(c["cp_names"]))]
            hidden = {"cp_v": c["cp_v"], "cp_kind": c["cp_kind"]}
        elif sc == "blotto":
            nodes = ["a0", "opponent"] + [f"site{k}" for k in range(len(c["site_names"]))]
            edges = [{"src": x, "dst": f"site{k}", "kind": "contest", "lag": 0} for x in ("a0", "opponent")
                     for k in range(len(c["site_names"]))]
            edges += [{"src": "a0", "dst": "opponent", "kind": "belief", "lag": 1}]
            hidden = {"opp_value": c["opp_value"], "opp_budget": c["opp_budget"]}
        elif sc == "interdiction":
            nodes = ["a0", "attacker"] + [f"route{j}" for j in range(len(c["route_names"]))]
            edges = [{"src": "attacker", "dst": f"route{j}", "kind": "attempt", "lag": 0}
                     for j in range(len(c["route_names"]))]
            edges += [{"src": "a0", "dst": f"route{j}", "kind": "inspect", "lag": 0} for j in range(len(c["route_names"]))]
            edges += [{"src": f"route{j}", "dst": "attacker", "kind": "experience", "lag": 1}
                      for j in range(len(c["route_names"]))]
            hidden = {"att_util": c["att_util"]}
        elif sc == "learning":
            nodes = ["a0", "opponent"]
            edges = [{"src": "a0", "dst": "opponent", "kind": "regret", "lag": 1},
                     {"src": "opponent", "dst": "a0", "kind": "play", "lag": 0}]
            hidden = {"B": c["B"]}
        elif sc == "profile":
            nodes = list(self.agents) + ["options", "decision"]
            edges = [{"src": "options", "dst": x, "kind": "private_attribute", "lag": 0} for x in self.agents]
            edges += [{"src": x, "dst": "decision", "kind": "joint", "lag": 0} for x in self.agents]
            hidden = {"w": c["w"]}
        elif sc == "commons":
            nodes = list(self.agents) + [f"patch{p}" for p in range(len(c["patch_names"]))]
            edges = [{"src": x, "dst": f"patch{p}", "kind": "harvest", "lag": 0} for x in self.agents
                     for p in range(len(c["patch_names"]))]
            edges += [{"src": f"patch{p}", "dst": f"patch{p}", "kind": "regrowth", "lag": 1}
                      for p in range(len(c["patch_names"]))]
            hidden = {"Kp": c["Kp"], "rp": c["rp"], "allee": c["allee"]}
        elif sc == "discovery":
            nodes = ["a0", "a1", "test"]
            edges = [{"src": "a0", "dst": "test", "kind": "ingredient", "lag": 0},
                     {"src": "a1", "dst": "test", "kind": "ingredient", "lag": 0}]
            hidden = {"typeA": c["typeA"], "typeB": c["typeB"]}
        else:
            nodes = ["a0"] + [f"worker{w}" for w in range(len(c["worker_names"]))] + ["task"]
            edges = [{"src": "a0", "dst": f"worker{w}", "kind": "pay", "lag": 0} for w in range(len(c["worker_names"]))]
            edges += [{"src": f"worker{w}", "dst": "task", "kind": "skill", "lag": 0}
                      for w in range(len(c["worker_names"]))]
            hidden = {"skills": c["skills"], "reliability": c["reliability"]}
        return {"engine": self.kind, "scenario": sc, "nodes": nodes, "edges": edges, "hidden": hidden,
                "motifs": self.motifs(), "regimes": False}

    def motifs(self) -> list[str]:
        return {
            "price_war": ["strategic_interaction", "adaptive_opponent", "learning_behaviour", "substitution",
                          "feedback"],
            "entry": ["strategic_interaction", "threshold", "selection_effect", "measurement_noise",
                      "learning_behaviour"],
            "bargain": ["strategic_interaction", "partial_observability", "complementarity", "threshold", "delay"],
            "blotto": ["resource_competition", "strategic_interaction", "adaptive_opponent", "saturation"],
            "interdiction": ["substitution", "adaptive_opponent", "learning_behaviour", "strategic_interaction"],
            "learning": ["learning_behaviour", "strategic_interaction", "non_monotonic", "path_dependence"],
            "profile": ["partial_observability", "linear_nonlinear", "complementarity", "measurement_noise"],
            "commons": ["stock", "threshold", "congestion", "resource_competition", "hysteresis"],
            "discovery": ["complementarity", "partial_observability", "multiplicative", "measurement_noise"],
            "formation": ["complementarity", "threshold", "partial_observability", "learning_behaviour",
                          "selection_effect"],
        }[self.sc]

    def latent(self, state) -> np.ndarray:
        """Hidden state for the informativeness check (evaluator side)."""
        c = self.config
        if self.sc == "profile":
            return np.asarray([sum(w * f for w, f in zip(c["w"], F)) for F in state["F"]], dtype=float)
        if self.sc == "commons":
            return np.asarray(state["S"], dtype=float)
        if self.sc == "price_war":
            return np.asarray(state["last"][1:], dtype=float)
        if self.sc == "blotto":
            return np.asarray(state["y"], dtype=float)
        return np.asarray([0.0])

    def outcome(self, state, ledger) -> dict:
        ags = self.agents
        if self.sc in ("profile", "commons", "discovery"):
            rev = sum(ledger["revenue"].values())
            cost = sum(ledger["cost"].values())
        else:
            rev = ledger["revenue"][ags[0]]
            cost = ledger["cost"][ags[0]]
        out = {"net": float(rev - cost), "value": float(rev - cost), "revenue": float(rev), "cost": float(cost)}
        if len(ags) > 1:
            out["per_agent"] = {a: float(ledger["revenue"][a] - ledger["cost"][a]) for a in ags}
        if self.sc == "bargain":
            out["deals_made"] = int(state["deals_made"])
        return out

    def conservation(self, before, after, out) -> list[str]:
        errs = []
        for a in self.agents:
            if out["revenue"].get(a, 0.0) < -1e-9 or out["cost"].get(a, 0.0) < -1e-9:
                errs.append("negative money flow")
        sc = self.sc
        if sc == "commons":
            if min(after["S"]) < -1e-9:
                errs.append("negative stock")
        if sc == "interdiction" and sum(after["tries"]) - sum(before["tries"]) != int(self.config["attempts"]):
            errs.append("attempts not conserved")
        if sc == "blotto":
            B = float(self.config["opp_budget"])
            if sum(after["y"]) > 2.0 * B + 1e-6:
                errs.append("opponent spent more than its budget")
        if sc == "learning":
            n0 = sum(sum(r) for r in before["counts"])
            n1 = sum(sum(r) for r in after["counts"])
            if n1 - n0 != 1:
                errs.append("one play per period")
        return errs
