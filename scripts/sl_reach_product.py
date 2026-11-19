"""Amendment 7 evidence (python scripts/sl_reach_product.py N). Is product success reachable by non-privileged policies? (test fixture; never shown to the agent)"""
import sys, itertools, collections
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from agentrs.cam.envs import ENVS
from agentrs.sl.worlds.cam_worlds import SPLIT_BASE

def run(policy, seed, split="val"):
    env = ENVS["product"](); env.reset(SPLIT_BASE[split] + seed, split)
    st = {"t": 0}
    n = 0
    while not env.done and n < 52:
        a = policy(env, st); env.step(a); n += 1; st["t"] += 1
    return dict(success=bool(env.success), bankrupt=env.budget < 0, ratio=env.active.sum() / env.target, steps=n)

def wait_only(env, st): return ["WAIT", "W4"]

def make_generic(n_feat=3, price="P2", chan="ads", spend="B2", onb=1, research=True):
    def pol(env, st):
        q = st.setdefault("q", None)
        if q is None:
            q = []
            if research: q += [["RESEARCH", s] for s in ["S0", "S1", "S2", "S3"]]
            st["q"] = q; st["phase"] = "research"
        if st["phase"] == "research" and q:
            return q.pop(0)
        if st["phase"] == "research":
            # count features named in visible research tokens
            toks = env.obs_tokens(); cnt = collections.Counter()
            for i, t in enumerate(toks):
                if t == "NEEDS": cnt.update(toks[i + 2:i + 4])
            feats = [f for f, _ in cnt.most_common(n_feat)] or ["F0", "F1", "F2"][:n_feat]
            st["q"] = [["BUILD", f] for f in feats] + [["SETPRICE", price]] + [["ONBOARD"]] * onb
            st["phase"] = "setup"
        if st["phase"] == "setup" and st["q"]:
            return st["q"].pop(0)
        st["phase"] = "grow"
        if env.spend_left[["ads", "content", "partners"].index(chan)] == 0 and env.budget > 20000:
            return ["CAMPAIGN", chan, spend]
        return ["WAIT", "W1"]
    return pol

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
pols = {"wait_only": wait_only}
for nf, pr, ch, sp, onb, res in itertools.product([2, 3], ["P1", "P2", "P3"], ["ads", "content", "partners"], ["B1", "B2", "B3"], [0, 1], [True, False]):
    pols[f"gen_f{nf}_{pr}_{ch}_{sp}_o{onb}_{'r' if res else 'nr'}"] = make_generic(nf, pr, ch, sp, onb, res)
rows = []
for name, p in pols.items():
    r = [run(p, s) for s in range(N)]
    rows.append((np.mean([x["success"] for x in r]), np.mean([x["bankrupt"] for x in r]), np.mean([min(1, x["ratio"]) for x in r]), name))
rows.sort(reverse=True)
print(f"val tasks: {N}; target = 0.45-0.7 x hidden-knowledge expert")
print("success  bankrupt  mean_progress  policy")
for s, b, p, n in rows[:12] + [r for r in rows if r[3] == "wait_only"]:
    print(f"{s:7.3f}  {b:8.3f}  {p:13.3f}  {n}")
print("policies with success > 0:", sum(1 for r in rows if r[0] > 0), "of", len(rows))
