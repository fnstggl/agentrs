"""Roll out a CAM checkpoint on product validation tasks and summarise its actions / bankruptcy rate.

    PYTHONPATH=src python scripts/cam_product_policy.py /workspace/agentrs/runs/cam/ckpt/latest
"""
import json, sys, collections
import numpy as np, torch
from agentrs.cam.model import CAM, CAMConfig
from agentrs.cam.runtime import Workers
ck = sys.argv[1]
cfg = CAMConfig(**json.load(open(f"{ck}/config.json")))
m = CAM(cfg); m.load_state_dict(torch.load(f"{ck}/weights.pt", map_location="cpu")); m.eval()
print("ckpt iter", json.load(open(f"{ck}/state.json")).get("iter"))
for greedy in (False, True):
    w = Workers(m, 64, {"product": 1.0}, "val", seed0=1_000_000, gen_seed=3)
    fixed = [("product", 1_000_000 + i) for i in range(200)]
    with torch.no_grad():
        eps = w.collect(fixed=fixed, greedy=greedy, version="diag") if "fixed" in Workers.collect.__code__.co_varnames else w.collect(min_episodes=200, greedy=greedy, version="diag")
    acts = collections.Counter(); first = collections.Counter(); sub = collections.Counter()
    bank = weeks = steps = 0; ret = []
    for ep in eps:
        for i, s in enumerate(ep.steps):
            a = [t for t in s.act_tokens if t != "END"]
            acts[a[0]] += 1; sub[" ".join(a)] += 1
            if i < 3: first[f"{i}:{' '.join(a)}"] += 1
        last = ep.steps[-1].feat_after
        bank += last[4] == 0; weeks += last[0] * 26; steps += len(ep.steps)
        ret.append(sum(s.reward for s in ep.steps))
    n = len(eps)
    print(f"\n{'greedy' if greedy else 'sampled'}: n {n} success {sum(e.success for e in eps)/n:.3f} bankrupt {bank/n:.3f} "
          f"weeks survived {weeks/n:.1f}/26 steps {steps/n:.1f} undiscounted return {np.mean(ret):.3f}")
    tot = sum(acts.values())
    print(" action types:", {k: round(v / tot, 3) for k, v in acts.most_common()})
    print(" top actions:", sub.most_common(8))
    print(" first 3 steps:", first.most_common(10))
