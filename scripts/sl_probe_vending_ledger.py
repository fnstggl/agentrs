"""Where does a v15 Vending agent lose money? Runs a checkpoint on Vending validation tasks (frozen weights, mode full) and
decomposes each episode's settled net-worth change: gross revenue, cost of goods sold, spoilage, undelivered paid
orders, marketing, fees paid and owed. Usage: probe_vending_ledger.py <run_dir> <ckpt_dir> [n_tasks]"""
import collections, json, sys
from pathlib import Path
import torch
from agentrs.sl import worlds as W
from agentrs.sl.model import SLConfig, SLCore
from agentrs.sl.runner import Runner
from agentrs.sl.world import RewardContract, RewardSpec

run, ckpt = Path(sys.argv[1]), Path(sys.argv[2])
n_tasks = int(sys.argv[3]) if len(sys.argv) > 3 else 24
st = json.loads((run / "state.json").read_text()); c = st["config"]
m = SLCore(SLConfig(**c["model"]))
if ckpt.name != "random":
    m.load_state_dict(torch.load(ckpt / "weights.pt", map_location="cpu"), strict=False)
m.eval()
fam = "ext/vending"
W.resolve(fam)
eps_out, acts = [], collections.Counter()


def summarise(w):
    e, s = w.e, w.e.s
    if s["tool_calls"] == 0 and s["day"] == 1:
        return
    orders = collections.Counter(); paid = collections.Counter()
    for o in s["orders"].values():
        orders[o["status"]] += 1
        paid[o["status"]] += float(o.get("amount_paid") or 0.0)
    eps_out.append({"settled": w._settled(), "nw0": w.nw0, "revenue": s["gross_revenue"], "units": s["units_sold"],
                    "fees_paid": s["fees_paid"], "fees_owed": w._fees_owed(), "marketing": s["marketing_spend"],
                    "spoiled": w._spoiled, "terminated": s["terminated"], "day": s["day"], "calls": s["tool_calls"],
                    "orders": dict(orders), "paid": dict(paid), "stock_value": e.net_worth() - s["balance"] - sum(p["amount"] for p in s["pending_card"]),
                    "balance": s["balance"], "min_balance": w._min_bal, "unpaid_max": w._unpaid_max})


def mk(f, *a, **k):
    w = W.make_world(f, *a, **k)
    ex, rs = w.execute, w.reset_world

    def reset_world():
        if w.e is not None:
            summarise(w)
        rs()
        w._spoiled, w._min_bal, w._unpaid_max = 0.0, w.e.s["balance"], 0
        sp = w.e._spoilage

        def spoil():
            before = w.e.net_worth()
            out = sp()
            w._spoiled += before - w.e.net_worth()
            return out
        w.e._spoilage = spoil

    def execute(agent, act):
        a = [x for x in act if x != "END"]
        acts[a[0] if a[0] not in ("price",) else f"price {a[-1]}"] += 1
        r = ex(agent, act)
        w._min_bal = min(w._min_bal, w.e.s["balance"])
        w._unpaid_max = max(w._unpaid_max, w.e.s["unpaid_days"])
        return r
    w.reset_world, w.execute = reset_world, execute
    mk.all.append(w)
    return w


mk.all = []
tasks = W.eval_tasks(fam, "val", n_tasks)
r = Runner(m, {fam: 1.0}, mk, W.sample_task, n_tasks, trial_episodes=int(c["trial_episodes"]), split="val", seed=12345, mode="full",
           contract=RewardContract(RewardSpec(**c["reward"])), device=torch.device("cpu"), max_episode_steps=c["max_episode_steps"],
           fixed_tasks=list(tasks), dt_ref=dict(st.get("dt_ref", {})))
with torch.no_grad():
    eps = r.run_trials()
for w in mk.all:
    if w.e is not None and w.e.over:
        summarise(w)
r.close()
n = len(eps_out)
avg = lambda k: sum(e[k] for e in eps_out) / n
print(f"{run.name} {ckpt.name}: {n} episodes; settled {avg('settled'):.1f} (start {avg('nw0'):.0f}); score {avg('settled') / avg('nw0'):.3f}")
for k in ("revenue", "units", "fees_paid", "fees_owed", "marketing", "spoiled", "stock_value", "balance", "min_balance", "unpaid_max", "day", "calls"):
    print(f"  {k:12s} {avg(k):10.1f}")
print("  terminated", sum(e["terminated"] for e in eps_out), "of", n)
oc = collections.Counter(); pc = collections.Counter()
for e in eps_out:
    oc.update(e["orders"]); pc.update(e["paid"])
print("  orders/ep by status", {k: round(v / n, 2) for k, v in oc.items()}, "paid/ep", {k: round(v / n, 1) for k, v in pc.items()})
tot = sum(acts.values())
print("  actions", [(a, round(k / tot, 3)) for a, k in acts.most_common(14)])
