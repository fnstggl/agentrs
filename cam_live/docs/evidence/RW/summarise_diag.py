"""Summarise RW's long-objective diagnostics: per seed, when the validity evidence moved the mix, how the learned
reward's prospective agreement with verified outcomes developed, and — against the frozen arm of the same seed (same
world, same decisions while the mix is 1) — where the decisions diverged and what the ground truth of the chosen
actions was afterwards.

    .venv/bin/python cam_live/docs/evidence/RW/summarise_diag.py --frozen F.json --arm A.json [--arm B.json ...] \\
        --out summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import random


def _boot_ci(xs, reps=4000, seed=0):
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(reps))
    return [ms[int(0.025 * reps)], ms[int(0.975 * reps)]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fz = {r["seed"]: r for r in json.load(open(a.frozen))["runs"]}
    out = {"frozen": a.frozen, "arms": {}}
    for path in a.arm:
        d = json.load(open(path))
        rows = []
        for r in d["runs"]:
            f = fz.get(r["seed"])
            first_change = r["mix_changes"][0][0] if r["mix_changes"] else None
            div = None
            if f is not None:
                div = next((i for i, (x, y) in enumerate(zip(r["rows"], f["rows"])) if x[2] != y[2]), None)
            after = (first_change if first_change is not None else len(r["rows"]))
            t_arm = [x[0] for x in r["rows"][after:]]
            t_fz = [x[0] for x in f["rows"][after:]] if f is not None else []
            diff = [x - y for x, y in zip(t_arm, t_fz)]
            rows.append({
                "seed": r["seed"], "decisions": r["decisions"], "meta_applied": r["meta_applied"],
                "mix_changes": r["mix_changes"], "mix_end": r["mix_end"], "label_end": r["label_end"],
                "first_mix_change_decision": first_change, "first_divergence_from_frozen": div,
                "identical_to_frozen_before_first_mix_change": (div is None or first_change is None or
                                                                div >= first_change),
                "pairs": r["validity"]["pairs"], "informative_pairs": r["validity"]["informative_pairs"],
                "concordance_learned": r["validity"]["concordance_learned"],
                "frozen_concordance_learned": f["validity"]["concordance_learned"] if f is not None else None,
                "agreement_gain_over_frozen_init": (r["validity"]["concordance_learned"]["mean"]
                                                    - f["validity"]["concordance_learned"]["mean"])
                if f is not None else None,
                "concordance_bootstrap": r["validity"]["concordance_bootstrap"],
                "learned_concordance_by_third_of_pairs": r["learned_concordance_by_third_of_pairs"],
                "e_values_at_mix_1": r.get("validity_e", {}).get("1"),
                "truth_chosen_mean_after_first_change": sum(t_arm) / len(t_arm) if t_arm else None,
                "frozen_truth_chosen_mean_same_decisions": sum(t_fz) / len(t_fz) if t_fz else None,
                "truth_difference_after_first_change": sum(diff) / len(diff) if diff else None,
                "truth_difference_ci95_decisions": _boot_ci(diff)})
        gains = [x["agreement_gain_over_frozen_init"] for x in rows if x["agreement_gain_over_frozen_init"] is not None]
        out["arms"][path] = {"arm": d["arm"], "utility": d.get("utility"), "half_life_s": d.get("half_life_s"),
                             "lr_reward": (d["runs"][0].get("lr_reward", 3e-3) if d["runs"] else None), "runs": rows,
                             "agreement_gain_over_frozen_init": gains,
                             "seeds_with_positive_gain": sum(1 for g in gains if g > 0),
                             "seeds_with_a_promotion": sum(1 for x in rows if x["first_mix_change_decision"]
                                                          is not None)}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out["arms"].items():
        print(k, v["arm"], v["utility"], "seeds with promotion:", v["seeds_with_a_promotion"], "agreement gain:",
              [round(g, 3) for g in v["agreement_gain_over_frozen_init"]])
        for x in v["runs"]:
            print("  seed", x["seed"], "applied", x["meta_applied"], "first change", x["first_mix_change_decision"],
                  "mix_end", x["mix_end"], "div", x["first_divergence_from_frozen"], "conc",
                  round(x["concordance_learned"]["mean"], 3), [round(c, 2) for c in
                                                               x["learned_concordance_by_third_of_pairs"]],
                  "dtruth", x["truth_difference_after_first_change"], x["truth_difference_ci95_decisions"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
