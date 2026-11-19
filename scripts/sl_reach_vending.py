"""Amendment 7 evidence (python scripts/sl_reach_vending.py N). Can a simple scripted operator make a profit through the agent's own action grammar? (test fixture only)"""
import sys, json
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from agentrs.sl.world import Task
from agentrs.sl.worlds.vending import VendingWorld

def legal(w, prefix):
    return w._allowed(list(prefix))

def act(w, a):
    w.execute("0", a + ["END"]); return w.last_out

def operator(w, mult="x2", qty="q40", frac=0.7, max_prod=5):
    e = w.e
    steps = 0
    for q in ["search", ]:
        pass
    for q in ["snacks", "drinks", "candy", "office"]:
        act(w, ["search", q]); steps += 1
    for s in list(w.known_sup):
        act(w, ["offers", f"sup:{s}"]); steps += 1
    def order_round():
        nonlocal steps
        best = {}
        for (sid, pid), p in w.offers.items():
            if pid not in best or p < best[pid][1]:
                best[pid] = (sid, p)
        budget = e.s["balance"] * frac
        for pid, (sid, p) in sorted(best.items(), key=lambda kv: kv[1][1])[:max_prod]:
            cost = p * int(qty[1:])
            if cost > budget: continue
            out = act(w, ["order", f"sup:{sid}", f"prod:{pid}", qty]); steps += 1
            if "order_id" in out: budget -= cost
    order_round()
    last_order_day = e.s["day"]
    while not e.over and steps < 590:
        # restock every slot we can from storage
        stock = [pid for pid, st in e.s["storage"].items() if st["quantity"] > 0]
        for pid in stock:
            for sl in w.slots:
                m = e.s["machine"][sl]
                if m.get("product_id") not in (None, pid) or (m.get("quantity") or 0) >= w.config.slot_capacity(sl):
                    continue
                if e.s["storage"].get(pid, {}).get("quantity", 0) <= 0: break
                loc, kind, rc = sl.split(":")
                out = act(w, ["restock", f"loc:{loc}", f"kind:{kind}", f"row:{rc[0]}", f"col:{rc[1]}", f"prod:{pid}"]); steps += 1
                if "error" in out.lower() or "refus" in out.lower() or "cannot" in out.lower():
                    continue
                if (e.s["machine"][sl].get("price") or 0) == 0 or not m.get("product_id"):
                    act(w, ["price", f"loc:{loc}", f"kind:{kind}", f"row:{rc[0]}", f"col:{rc[1]}", mult]); steps += 1
                if steps >= 580: break
        if e.s["day"] - last_order_day >= 6 and e.s["day"] < 22 and e.s["balance"] > 300:
            order_round(); last_order_day = e.s["day"]
        act(w, ["wait"]); steps += 1
    return e.net_worth() / w.nw0, steps, bool(e.s.get("terminated"))

def wait_only(w):
    steps = 0
    while not w.e.over and steps < 600:
        act(w, ["wait"]); steps += 1
    return w.e.net_worth() / w.nw0, steps, bool(w.e.s.get("terminated"))

def _main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    for name, fn in [("wait_only", wait_only)] + [(f"operator_{m}_{q}", (lambda m=m, q=q: (lambda w: operator(w, m, q)))()) for m in ["x1.5", "x2", "x3"] for q in ["q80", "q160"]]:
        res = []
        for seed in range(N):
            w = VendingWorld(); w.initialize_world(Task("ext/vending", seed, "val"))
            res.append(fn(w))
        sc = np.array([r[0] for r in res])
        print(f"{name:20s} score mean {sc.mean():.3f}  min {sc.min():.3f}  max {sc.max():.3f}  success(>=1.667) {np.mean(sc >= 2500/1500):.2f}  steps {np.mean([r[1] for r in res]):.0f}  terminated {np.mean([r[2] for r in res]):.2f}", flush=True)


if __name__ == "__main__":
    _main()
