"""How many scored pairs does the validity rule need before it lowers the mix? Simulation with the rule's own test
(``reward.validity.BettingTest`` at alpha/2 = 0.025, cap 0.75): the bootstrap orders every pair at random (as it does at
reward time in LabWorld), R_psi orders an informative pair like the verifier with probability p, 10 % of the pairs have
tied outcomes (score 0). 400 runs of 400 pairs per p.

    PYTHONPATH=cam_live/src .venv/bin/python cam_live/docs/evidence/RW/promotion_power_sim.py > .../promotion_power.json
"""
import json
import random

from cam_live.reward.validity import BettingTest


def sim(p, reps=400, n=400, tie=0.1, seed=3):
    rng = random.Random(seed)
    hits = []
    for _ in range(reps):
        t, h = BettingTest(0.025, +1), None
        for i in range(1, n + 1):
            if rng.random() < tie:
                x = 0.0
            else:
                c0 = 1 if rng.random() < p else -1
                c1 = 1 if rng.random() < 0.5 else -1
                x = 0.5 * (c0 - c1)
            t.update(x)
            if t.rejected and h is None:
                h = i
        hits.append(h)
    hs = sorted(x for x in hits if x)
    return {"p_correct_order": p, "share_promoted_within_400_pairs": len(hs) / reps,
            "median_pairs_to_promotion": hs[len(hs) // 2] if hs else None,
            "q10_q90_pairs": [hs[len(hs) // 10], hs[(9 * len(hs)) // 10]] if len(hs) >= 10 else None}


if __name__ == "__main__":
    print(json.dumps({"alpha_half": 0.025, "cap": 0.75, "tie_share": 0.1, "runs": 400,
                      "results": [sim(p) for p in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]}, indent=1))
