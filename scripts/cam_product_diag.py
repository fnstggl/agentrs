"""Usage: PYTHONPATH=src python scripts/cam_product_diag.py [N]

Read-only diagnosis of the CAM product env: can any policy acting through the real interface reach the target?"""
import random, sys, copy
import numpy as np
from agentrs.cam.envs.product import ProductEnv, FEATS, SEGS, CHANNELS, PRICES, SPEND, WEEKS

def run(policy, seed, split="val"):
    e = ProductEnv(); e.reset(seed, split)
    st = {"t": 0}
    while not e.done and st["t"] < 60:
        a = policy(e, st); st["t"] += 1
        e.step(a)
    return e

def rand_pol(rng):
    def p(e, st):
        g = e.grammar(); pre = []
        while not pre or pre[-1] != "END":
            pre.append(rng.choice(g.allowed(pre)))
        return pre[:-1]
    return p

def oracle_iface(spend_lvl=2, onb=2, nfeat=3, channels=1):
    """Knows the hidden market (like _expert_users) but must pay for and sequence actions through the real interface."""
    def p(e, st):
        if "plan" not in st:
            w = (e.needs * e.sizes[:, None]).sum(0)
            feats = [int(f) for f in np.argsort(-w)[:nfeat]]
            score = (e.aff[:, :3] * e.sizes[:, None]).sum(0) * e.eff[:3]
            chs = [int(c) for c in np.argsort(-score)[:channels]]
            plan = [["BUILD", FEATS[f]] for f in feats]
            st["chs"] = chs; st["plan"] = plan; st["onb"] = onb; st["priced"] = False
        if st["plan"]:
            return st["plan"].pop(0)
        for c in st["chs"]:
            if e.spend_left[c] == 0:
                return ["CAMPAIGN", CHANNELS[c], f"B{spend_lvl}"]
        if not st["priced"] and not e.building:
            st["priced"] = True
            ee = copy.deepcopy(e)
            best = max(range(len(PRICES)), key=lambda q: (setattr(ee, "price", q), (ee._accept() * ee.sizes).sum() * max(1.0, PRICES[q]) ** 0.3)[1])
            return ["SETPRICE", f"P{best}"]
        if st["onb"] > 0 and e.onb < 3:
            st["onb"] -= 1
            return ["ONBOARD"]
        return ["WAIT", "W1"]
    return p

def heuristic(spend_lvl=2):
    """No hidden info: research every segment, build the most-requested features, then advertise."""
    def p(e, st):
        if "r" not in st: st["r"] = 0; st["b"] = None; st["onb"] = 2
        if st["r"] < 4:
            st["r"] += 1; return ["RESEARCH", SEGS[st["r"] - 1]]
        if st["b"] is None:
            cnt = {}
            for s, (top, sens) in e.research.items():
                for i, f in enumerate(top): cnt[f] = cnt.get(f, 0) + (2 - i)
            st["b"] = sorted(cnt, key=lambda f: -cnt[f])[:3]
        if st["b"]:
            return ["BUILD", st["b"].pop(0)]
        if e.spend_left[0] == 0:
            return ["CAMPAIGN", "ads", f"B{spend_lvl}"]
        if st["onb"] > 0:
            st["onb"] -= 1; return ["ONBOARD"]
        return ["WAIT", "W1"]
    return p

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
def evalp(name, mk, split="val"):
    succ, prog, bank, users, tgt = [], [], [], [], []
    for s in range(N):
        e = run(mk(), 10_000 + s, split)
        succ.append(bool(e.success)); prog.append(e.progress()); bank.append(e.budget < 0)
        users.append(e.active.sum()); tgt.append(e.target)
    print(f"{name:42s} success {np.mean(succ):.3f}  progress {np.mean(prog):.3f}  bankrupt {np.mean(bank):.3f}  "
          f"users med {np.median(users):7.0f}  target med {np.median(tgt):7.0f}")

rng = random.Random(0)
evalp("random", lambda: rand_pol(rng))
for lvl in (1, 2, 3, 4):
    evalp(f"oracle-through-interface B{lvl}", lambda lvl=lvl: oracle_iface(spend_lvl=lvl))
evalp("oracle-through-interface B2, 2 channels", lambda: oracle_iface(2, channels=2))
evalp("oracle-through-interface B3, 2 channels", lambda: oracle_iface(3, channels=2))
evalp("oracle-through-interface B2, 4 feats", lambda: oracle_iface(2, nfeat=4))
evalp("oracle-through-interface B2, onb 0", lambda: oracle_iface(2, onb=0))
for lvl in (1, 2, 3):
    evalp(f"heuristic(no hidden info) B{lvl}", lambda lvl=lvl: heuristic(lvl))
print("test split:")
evalp("oracle-through-interface B2 (test)", lambda: oracle_iface(2), "test")
