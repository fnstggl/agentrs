"""Reachability under the benchmark's money score with an operator that only places orders the agent's own grammar
offers (v14+ valid orders), restocks every slot it can, prices at a fixed multiple of unit cost, and stops ordering by a
given day so stock sells down. Frozen v15 adapter; validation seeds."""
import sys
import numpy as np
from agentrs.sl.world import Task
from agentrs.sl.worlds.vending import VendingWorld, ORDER_Q


def act(w, a):
    w.execute("0", a + ["END"]); return w.last_out


def run(w, mult, spend_frac, last_day, n_prod, reserve=200.0):
    e = w.e; steps = 0
    for q in ["snacks", "drinks", "candy", "office"]:
        act(w, ["search", q]); steps += 1
    for s in list(w.known_sup):
        act(w, ["offers", f"sup:{s}"]); steps += 1

    def order_round():
        nonlocal steps
        budget = max(0.0, e.s["balance"] - reserve) * spend_frac
        valid = w._valid_orders()
        best = {}
        for sid, pid, q in valid:
            p = w.offers[(sid, pid)]
            if pid not in best or p < best[pid][1]:
                best[pid] = (sid, p)
        for pid, (sid, p) in sorted(best.items(), key=lambda kv: kv[1][1])[:n_prod]:
            qs = sorted([q for s_, p_, q in w._valid_orders() if s_ == sid and p_ == pid], key=lambda q: ORDER_Q[q])
            qs = [q for q in qs if p * ORDER_Q[q] <= budget]
            if not qs:
                continue
            out = act(w, ["order", f"sup:{sid}", f"prod:{pid}", qs[0]]); steps += 1
            if "refus" not in out.lower() and "error" not in out.lower():
                budget -= p * ORDER_Q[qs[0]]

    order_round(); last = e.s["day"]
    while not e.over and steps < 590:
        for pid in [p for p, st in e.s["storage"].items() if st["quantity"] > 0]:
            for sl in w.slots:
                m = e.s["machine"][sl]
                if m.get("product_id") not in (None, pid) or (m.get("quantity") or 0) >= w.config.slot_capacity(sl):
                    continue
                if e.s["storage"].get(pid, {}).get("quantity", 0) <= 0:
                    break
                loc, kind, rc = sl.split(":")
                out = act(w, ["restock", f"loc:{loc}", f"kind:{kind}", f"row:{rc[0]}", f"col:{rc[1]}", f"prod:{pid}"]); steps += 1
                if "error" in out.lower() or "refus" in out.lower() or "cannot" in out.lower():
                    continue
                if e.s["machine"][sl].get("product_id"):
                    act(w, ["price", f"loc:{loc}", f"kind:{kind}", f"row:{rc[0]}", f"col:{rc[1]}", mult]); steps += 1
                if steps >= 580:
                    break
        if e.s["day"] - last >= 4 and e.s["day"] <= last_day and e.s["balance"] > reserve + 100:
            order_round(); last = e.s["day"]
        act(w, ["wait"]); steps += 1


def money(w):
    sc = w.e.score(); s = w.e.s
    stock = w.e.net_worth() - s["balance"] - sum(p["amount"] for p in s["pending_card"])
    spent = sum(float(o.get("amount_paid") or 0) for o in s["orders"].values())
    return sc["reward"], w.e.net_worth(), bool(s.get("terminated")), w._settled(), s["gross_revenue"], s["units_sold"], stock, spent


N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
for mult in ["x1.5", "x2", "x3"]:
    for frac in [0.3, 0.7]:
        for last_day in [10, 20]:
            for n_prod in [3, 8]:
                res = []
                for seed in range(N):
                    w = VendingWorld(); w.initialize_world(Task("ext/vending", seed, "val"))
                    run(w, mult, frac, last_day, n_prod)
                    res.append(money(w))
                mo = np.array([r[0] for r in res])
                print(f"{mult:5s} frac {frac} last {last_day:2d} prod {n_prod}: money {mo.mean():7.1f} (min {mo.min():6.1f} max {mo.max():6.1f})  "
                      f"net worth {np.mean([r[1] for r in res]):7.1f}  revenue {np.mean([r[4] for r in res]):5.0f}  units {np.mean([r[5] for r in res]):4.0f}  "
                      f"stock left {np.mean([r[6] for r in res]):5.0f}  spent {np.mean([r[7] for r in res]):5.0f}  term {np.mean([r[2] for r in res]):.2f}", flush=True)
