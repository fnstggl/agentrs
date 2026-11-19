"""Amendment 8 evidence (python scripts/sl_vending_discount.py). Discounted (gamma=0.99 per decision, as the learner) return of the scripted operator vs inaction, under the
adapter's cash-style reward (delta benchmark net worth) and under accrual accounting (paid goods in transit at cost;
overdue orders written off; terminal step pays the benchmark's final net worth). Test fixture only."""
import sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from agentrs.sl.world import Task
from agentrs.sl.worlds.vending import VendingWorld
import sl_reach_vending as VR  # same directory

def in_transit(e):
    d = e.s["day"]
    return sum(o.get("amount_paid") or 0.0 for o in e.s["orders"].values()
               if o["status"] in ("paid", "shipped") and (o.get("delivery_day") is None or o["delivery_day"] >= d))

def trace(policy, seed):
    w = VendingWorld(); w.initialize_world(Task("ext/vending", seed, "val"))
    cash, accr, dts = [], [], []
    orig = w.execute
    def ex(agent, a):
        if w.e.over:  # the runner never steps a finished episode
            return
        b_nw, b_ac = w.e.net_worth(), w.e.net_worth() + in_transit(w.e)
        t0 = w.e.s["day"] * 1440 + w.e.s["minute"]
        orig(agent, a)
        dts.append(max(0.0, w.e.s["day"] * 1440 + w.e.s["minute"] - t0))
        a_nw = w.e.net_worth()
        a_ac = a_nw if w.e.over else a_nw + in_transit(w.e)
        cash.append((a_nw - b_nw) / 100); accr.append((a_ac - b_ac) / 100)
    w.execute = ex
    policy(w)
    return np.array(cash), np.array(accr), w.e.net_worth() / w.nw0, np.array(dts)

def stall(w):  # what the trained agents do: non-advancing calls until the day rolls over, ~20 per day
    n = 0
    while not w.e.over and n < 2000:
        w.execute("0", ["search", "snacks", "END"]); n += 1

def disc(r, g=0.99):
    return float(np.sum(r * g ** np.arange(len(r))))

def tdisc(r, dt, ref=72.0, g=0.99):  # semi-Markov: gamma ** (elapsed world minutes / ref), as runner.learner_discount
    dtn = np.clip(dt / ref, 0.25, 20.0)
    D = np.concatenate([[1.0], np.cumprod(g ** dtn)[:-1]])
    return float(np.sum(r * D))

pols = {"wait_only": VR.wait_only, "stall(search)": stall, "operator_x3_q160": lambda w: VR.operator(w, "x3", "q160")}
for name, p in pols.items():
    rows = [trace(p, s) for s in range(12)]
    print(f"{name:18s} steps {np.mean([len(r[0]) for r in rows]):5.0f}  score {np.mean([r[2] for r in rows]):.3f}  "
          f"undiscounted {np.mean([r[0].sum() for r in rows]):+7.2f} (accrual {np.mean([r[1].sum() for r in rows]):+7.2f})  "
          f"per-call: cash {np.mean([disc(r[0]) for r in rows]):+7.2f} accrual {np.mean([disc(r[1]) for r in rows]):+7.2f} | "
          f"world-time ref 72 min: cash {np.mean([tdisc(r[0], r[3]) for r in rows]):+7.2f} accrual {np.mean([tdisc(r[1], r[3]) for r in rows]):+7.2f} | "
          f"ref 288 min: accrual {np.mean([tdisc(r[1], r[3], 288.0) for r in rows]):+7.2f}  mean dt {np.mean([r[3].mean() for r in rows]):.0f} min", flush=True)
