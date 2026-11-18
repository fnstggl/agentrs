"""Prosus Vending-Bench (https://github.com/ProsusAI/vending-bench, Apache-2.0) as an external business world.

The engine is the benchmark's own seeded, stdlib-only simulation (demand with price elasticity, weekday/season/
weather effects, suppliers with delivery delays and failures, card settlement, daily fees, complaints and
reputation); no LLM is involved (the optional supplier-prose LLM hook is off). We call its ``tool_*`` methods
directly. One agent decision = one tool call; every call consumes simulated minutes and days roll over as in the
benchmark. 30-day horizon (the benchmark's default config).

Adapter assumptions (documented, fixed before any evaluation):
* Observation: the operator's dashboard as structured readings - day/time, balance, card money in transit,
  reputation, unpaid days, yesterday's units, the contents/price/unit cost of every machine slot, depot stock,
  suppliers and quotes the agent has itself discovered (search_web / check_offers results), its own open orders.
  The benchmark makes the agent pay simulated minutes for reading these; here they are always visible (an easier
  observation model, applied identically to every agent and baseline). Hidden demand parameters, supplier
  personas and the RNG are never exposed.
* Actions: a fixed grammar over the benchmark's tools (search, check offers, order, restock, set price as a
  multiple of the slot's unit cost, clear slot, marketing, wait) with discretised arguments; quantities for
  restocking fill the slot from depot stock. Negotiation by e-mail is not available to the agent.
* Reward: change in the benchmark's net worth (balance + card in transit + stock at cost) per call, /100.
  Verified success: run completed (not terminated) with final net worth >= 2,500 (start: 1,500).
"""
from __future__ import annotations

import json
import re
import sys

from agentrs.sl.world import ActionSpace, Consequence, GoalStatus, Obs, Task, World

SIM = "/workspace/ext/vending-bench/src/tasks/vending-bench/environment/sim-server"
TARGET = 2500.0
SEEDS = {"train": 100_000, "val": 50_000, "test": 0}
QUERIES = {"snacks": "wholesale snacks vending supplier", "drinks": "cold drinks soda wholesale supplier",
           "candy": "candy sweets bulk supplier", "office": "office food service wholesaler"}
ORDER_Q = {"q10": 10, "q20": 20, "q40": 40, "q80": 80, "q160": 160}
MULT = {"x1.2": 1.2, "x1.5": 1.5, "x2": 2.0, "x2.5": 2.5, "x3": 3.0, "x4": 4.0}


def _engine():
    if SIM not in sys.path:
        sys.path.insert(0, SIM)
    from vending import config
    from vending.catalog import PRODUCTS
    from vending.engine import Engine
    return Engine, config, PRODUCTS


class VendingWorld(World):
    family = "ext/vending"
    reward_overrides = {"native": 1.0, "shaping": 0.0, "success": 1.0, "violation": 1.0}

    def __init__(self):
        self.Engine, self.config, self.PRODUCTS = _engine()
        self.slots = self.config.all_slots()
        self.locs = sorted({s.split(":")[0] for s in self.slots})
        self.e = None

    def initialize_world(self, task: Task) -> None:
        self.task = task
        self.reset_world()

    def reset_world(self) -> None:
        self.e = self.Engine.new(SEEDS[self.task.split] + self.task.seed, path=None)
        self.known_sup: list = []
        self.offers: dict = {}          # (supplier, product) -> unit price
        self.last_out = ""
        self.nw0 = self.e.net_worth()
        self._dnw = 0.0
        self._calls = 0

    # ------------------------------------------------------------------ observation
    def observe(self, agent="0") -> Obs:
        s, c = self.e.s, self.config
        nums = {"day": s["day"] / c.SIM_DAYS, "minute": s["minute"] / 1440.0, "balance": s["balance"] / 1000.0,
                "card_transit": sum(p["amount"] for p in s["pending_card"]) / 1000.0, "reputation": float(s["reputation"]),
                "unpaid_days": float(s["unpaid_days"]), "units_yesterday": (s["sales_log"][-1]["units"] if s["sales_log"] else 0) / 100.0,
                "net_worth": self.e.net_worth() / 1000.0, "unread_mail": float(sum(1 for m in s["emails"] if m["folder"] == "inbox" and not m["read"]))}
        ents = []
        for sl in self.slots:
            m = s["machine"][sl]
            loc, kind, rc = sl.split(":")
            pid = m.get("product_id") or "empty"
            ents.append((f"slot:{kind}:{self.config.slot_size_class(sl)}:{pid}",
                         [self.locs.index(loc) / 3, "ABCD".index(rc[0]) / 4, int(rc[1]) / 3, (m.get("quantity") or 0) / self.config.slot_capacity(sl),
                          m.get("price") or 0.0, m.get("unit_cost") or 0.0]))
        for pid, st in s["storage"].items():
            if st["quantity"] > 0:
                uc = st["batches"][0]["unit_cost"] if st["batches"] else 0.0
                ents.append((f"stock:{pid}", [st["quantity"] / 100.0, uc]))
        best: dict = {}
        for (sid, pid), price in sorted(self.offers.items()):  # cheapest known quote per product
            if pid not in best or price < best[pid][1]:
                best[pid] = (sid, price)
        for pid, (sid, price) in sorted(best.items()):
            ents.append((f"offer:{sid}:{pid}", [price]))
        for o in s["orders"].values():
            if o["status"] in ("paid", "shipped"):
                ents.append((f"order:{o['supplier_id']}", [max(0, (o.get("delivery_day") or s["day"]) - s["day"]) / 7.0,
                                                          (o.get("amount_paid") or 0) / 100.0]))
        toks = [f"supplier:{x}" for x in self.known_sup] + [f"weather:{s['weather']}"]
        return Obs(tokens=toks, numbers=nums, entities=ents)

    def goal(self, agent="0") -> Obs:
        return Obs(tokens=["goal:net_worth", "horizon:30d"], numbers={"target_net_worth": TARGET / 1000.0})

    # ------------------------------------------------------------------ actions
    def available_actions(self, agent="0") -> ActionSpace:
        return ActionSpace(kind="sequential", allowed=self._allowed, max_len=7)

    def _slot_parts(self):
        return {loc: sorted({sl.split(":")[1] for sl in self.slots if sl.startswith(loc + ":")}) for loc in self.locs}

    def _allowed(self, p: list) -> list:
        if not p:
            acts = ["search", "wait"]
            if self.known_sup:
                acts += ["offers"]
            if self.offers:
                acts += ["order"]
            if any(st["quantity"] > 0 for st in self.e.s["storage"].values()):
                acts += ["restock"]
            if any(self.e.s["machine"][sl].get("product_id") for sl in self.slots):
                acts += ["price", "clear"]
            acts += ["market"]
            return acts
        a = p[0]
        if a == "wait":
            return ["END"]
        if a == "search":
            return list(QUERIES) if len(p) == 1 else ["END"]
        if a == "offers":
            return [f"sup:{x}" for x in self.known_sup] if len(p) == 1 else ["END"]
        if a == "order":
            if len(p) == 1:
                return sorted({f"sup:{sid}" for sid, _ in self.offers})
            if len(p) == 2:
                sid = p[1][4:]
                return [f"prod:{pid}" for (s_, pid) in sorted(self.offers) if s_ == sid]
            return list(ORDER_Q) if len(p) == 3 else ["END"]
        if a == "market":
            if len(p) == 1:
                return [f"loc:{x}" for x in self.locs]
            return [f"ch:{x}" for x in self.config.MARKETING["channels"]] if len(p) == 2 else ["END"]
        # slot-addressed actions: restock / price / clear
        if len(p) == 1:
            return [f"loc:{x}" for x in self.locs]
        if len(p) == 2:
            return [f"kind:{k}" for k in self._slot_parts()[p[1][4:]]]
        if len(p) == 3:
            return ["row:A", "row:B", "row:C", "row:D"]
        if len(p) == 4:
            return ["col:1", "col:2", "col:3"]
        if a == "restock" and len(p) == 5:
            return [f"prod:{pid}" for pid, st in sorted(self.e.s["storage"].items()) if st["quantity"] > 0] or ["END"]
        if a == "price" and len(p) == 5:
            return list(MULT)
        return ["END"]

    def execute(self, agent, action) -> None:
        a = [x for x in action if x != "END"]
        e = self.e
        before = e.net_worth()
        out = ""
        try:
            if a[0] == "wait":
                out = e.tool_wait_for_next_day()
            elif a[0] == "search":
                out = e.tool_search_web(QUERIES[a[1]])
                for sid in re.findall(r"Supplier ID: (\S+)", out):
                    if sid not in self.known_sup:
                        self.known_sup.append(sid)
            elif a[0] == "offers":
                sid = a[1][4:]
                out = e.tool_check_offers(sid, 60)
                try:
                    offers = json.loads(out.split("\n", 1)[0])["offers"]
                    for pid, o in offers.items():
                        self.offers[(sid, pid)] = float(o["unit_price"])
                except (ValueError, KeyError, TypeError):
                    pass
            elif a[0] == "order":
                out = e.tool_order_goods(a[1][4:], {a[2][5:]: ORDER_Q[a[3]]})
            elif a[0] == "market":
                out = e.tool_run_marketing(a[1][4:], a[2][3:], "Fresh snacks and cold drinks, restocked daily.")
            else:
                slot = f"{a[1][4:]}:{a[2][5:]}:{a[3][4:]}{a[4][4:]}"
                if slot not in e.s["machine"]:
                    out = "unknown slot"
                elif a[0] == "clear":
                    out = e.tool_clear_slot(slot)
                elif a[0] == "restock" and len(a) > 5:
                    pid = a[5][5:]
                    m = e.s["machine"][slot]
                    cap = self.config.slot_capacity(slot)
                    q = min(max(0, cap - (m.get("quantity") or 0) if m.get("product_id") in (None, pid) else cap),
                            e.s["storage"].get(pid, {}).get("quantity", 0))
                    fn = e.tool_restock_machine if m.get("product_id") in (None, pid) else e.tool_swap_item
                    out = fn(slot, pid, max(1, q))
                elif a[0] == "price" and len(a) > 5:
                    uc = e.s["machine"][slot].get("unit_cost") or 1.0
                    out = e.tool_set_price(slot, round(max(0.1, uc * MULT[a[5]]), 2))
        except Exception as ex:  # noqa: BLE001 - an invalid call is a wasted (time-consuming) action, not a crash
            out = f"error: {type(ex).__name__}"
        self.last_out = out if isinstance(out, str) else str(out)
        self._dnw = e.net_worth() - before
        self._calls += 1

    def observe_consequences(self, agent="0") -> Consequence:
        ev = ["refused"] if ("refused" in self.last_out.lower() or "error" in self.last_out.lower()) else []
        return Consequence(reward=self._dnw / 100.0, dt=1.0, done=bool(self.e.over), events=ev)

    def evaluate_goal(self, agent="0") -> GoalStatus:
        s = self.e.s
        nw = self.e.net_worth()
        done = bool(self.e.over)
        terminated = bool(s.get("terminated"))
        return GoalStatus(success=done and not terminated and nw >= TARGET,
                          progress=min(1.0, max(0.0, (nw - self.nw0) / (TARGET - self.nw0))),
                          violated="terminated_unpaid_fees" if terminated else None, score=nw / self.nw0,
                          time_left=1.0 - min(1.0, s["day"] / self.config.SIM_DAYS))

    def save_world_state(self):
        import copy
        return copy.deepcopy((self.e.s, self.known_sup, self.offers, self.nw0))

    def restore_world_state(self, state) -> None:
        import copy
        s, self.known_sup, self.offers, self.nw0 = copy.deepcopy(state)
        self.e.s = s


def available() -> bool:
    try:
        _engine()
        return True
    except Exception:  # noqa: BLE001
        return False
