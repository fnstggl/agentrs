"""RW round 2 (G3-core D-RW1): promotion rates of unlearned and learned R_psi under the rule with never-trained
control rewards. Reads the verifier's flipped-frozen probe output (both signs of each frozen initialisation) and RW's
long-objective diagnostics (drift = R_psi moving by random steps of the real steps' size, learned arms, frozen).

    python cam_live/docs/evidence/RW/summarise_round2.py --flipped F.json --diag drift.json learned.json ... --out S.json
"""
import argparse
import json


def _row(r):
    v = r["validity"]
    pl = v.get("placebo") or {}
    return {"seed": r["seed"], "arm": r["arm"], "lr_reward": r.get("lr_reward"), "decisions": r["decisions"],
            "meta_applied": r["meta_applied"], "promoted": bool(r["mix_changes"]),
            "first_mix_change": r["mix_changes"][0][0] if r["mix_changes"] else None,
            "concordance": v["concordance_learned"]["mean"], "concordance_cs": [v["concordance_learned"]["lower"],
                                                                              v["concordance_learned"]["upper"]],
            "e_learned_vs_mix1": v["by_mix"]["1"]["learned_minus_mix"]["e"],
            "beats_mixture_at_1": v["by_mix"]["1"]["learned_minus_mix"]["e"] >= 40.0,
            "placebo_tests_rejected": pl.get("rejected"), "placebo_tests": pl.get("of"),
            "best_abs_placebo_concordance": pl.get("best_abs_concordance"),
            "psi0_concordance": pl.get("psi0_concordance"), "min_e_vs_placebo": pl.get("min_e_learned_vs_placebo")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flipped", required=True)
    ap.add_argument("--diag", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"unlearned_frozen_both_signs": [], "arms": {}}
    fl = json.load(open(a.flipped))
    for r in fl["runs"]:
        for arm, key in (("frozen", "frozen_mix_changes"), ("frozen_neg", "neg_mix_changes")):
            out["unlearned_frozen_both_signs"].append({
                "seed": r["seed"], "arm": arm, "promoted": bool(r[key]),
                "concordance": (r["frozen_concordance"] if arm == "frozen" else r["neg_concordance"])["mean"],
                "e_learned_vs_mix1": r["frozen_e_learned_vs_mix1" if arm == "frozen" else "neg_e_learned_vs_mix1"]})
    u = out["unlearned_frozen_both_signs"]
    out["unlearned_promoted"] = f"{sum(x['promoted'] for x in u)} of {len(u)}"
    out["unlearned_beating_the_mixture_at_1"] = f"{sum(x['e_learned_vs_mix1'] >= 40 for x in u)} of {len(u)}"
    for p in a.diag:
        d = json.load(open(p))
        rows = [_row(r) for r in d["runs"]]
        key = f"{d['arm']}|{d.get('utility')}|lr={rows[0]['lr_reward'] if rows else None}"
        out["arms"][key] = {"file": p, "runs": rows, "promoted": f"{sum(x['promoted'] for x in rows)} of {len(rows)}",
                            "beating_the_mixture_at_1": f"{sum(x['beats_mixture_at_1'] for x in rows)} of {len(rows)}"}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("unlearned (frozen, both signs):", out["unlearned_promoted"], "promoted;",
          out["unlearned_beating_the_mixture_at_1"], "beat the mixture at mix 1 (the round-1 rule would have lowered)")
    for k, v in out["arms"].items():
        print(k, "promoted", v["promoted"], "| beat mixture at 1:", v["beating_the_mixture_at_1"])
        for x in v["runs"]:
            print("  seed", x["seed"], "promoted", x["promoted"], x["first_mix_change"], "conc", round(x["concordance"], 3),
                  "e_mix1", round(x["e_learned_vs_mix1"], 1), "placebo rejected", x["placebo_tests_rejected"], "/",
                  x["placebo_tests"], "best |placebo|", None if x["best_abs_placebo_concordance"] is None
                  else round(x["best_abs_placebo_concordance"], 3))


if __name__ == "__main__":
    main()
