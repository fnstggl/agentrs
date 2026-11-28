"""G3-core probe RW-2: the whole validity procedure (reward/validity.py ProspectiveValidity: note -> resolve -> pair
-> fixed-sequence / Bonferroni decision -> mix walk), not only one BettingTest.

(1) Type-I error of LOWERING under the null, with the mix walk driven by decide() after every scored pair (optional
    stopping, continuous monitoring), up to 2000 pairs per run, three nulls:
      iid        r_learned, r_bootstrap, y independent Gaussians (R_psi exactly as good as the bootstrap: none);
      tied_boot  the bootstrap gives every decision the same reward (ties at reward time, as in the RW LabWorld
                 diagnostics where concordance_bootstrap == 0.000 in every run) and R_psi is independent of y;
      boot_anti  the bootstrap is ANTI-informative (concordance < 0) and R_psi independent of y: R_psi alone does
                 agree better than the mixture in force, so lowering is CORRECT under the rule's H0 — reported to show
                 what the guarantee is relative to (not "R_psi is informative").
(2) Where does the mix walk stop when the bootstrap ties at reward time and R_psi is perfectly informative?
    (Claim checked: "the mix can stand at m_{k+1} only after the x+ tests at m_0..m_k were rejected"; with a tied
    bootstrap every mixture m < 1 ranks pairs exactly like R_psi alone, so x+ = 0 at every m < 1.)

    PYTHONPATH=<freeze>/cam_live/src python probe_rw_validity_procedure.py --out out/rw_validity_procedure.json
"""
import argparse
import json
import random


def one_run(kind, n_pairs, rng, p_inf=1.0):
    from cam_live.reward.validity import ProspectiveValidity
    pv = ProspectiveValidity(step=0.25, alpha=0.05)
    mix = 1.0
    lowered_at = None
    mixes = []
    for t in range(2 * n_pairs):
        y = rng.gauss(0, 1)
        if kind == "iid":
            rb, rl = rng.gauss(0, 1), rng.gauss(0, 1)
        elif kind == "tied_boot":
            rb, rl = 0.0, rng.gauss(0, 1)
        elif kind == "boot_anti":
            rb, rl = -y + 0.5 * rng.gauss(0, 1), rng.gauss(0, 1)
        elif kind == "tied_boot_informative":
            rb = 0.0
            rl = y if rng.random() < p_inf else -y
        pv.note(f"e{t}", segment=0, group=f"g{t}", r_bootstrap=rb, r_learned=rl, ts=float(t), version="v0")
        if pv.resolve(f"e{t}", y):
            d = pv.decide(mix)
            if d.action == "lower" and lowered_at is None:
                lowered_at = pv.n_pairs
            mix = d.mix_after
            mixes.append(mix)
    return lowered_at, mix, pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(7)
    out = {"probe": "rw_validity_procedure", "reps": a.reps, "pairs_per_run": a.pairs, "nominal_lowering_error": 0.025,
           "results": {}}
    for kind in ("iid", "tied_boot", "boot_anti"):
        lows = []
        for _ in range(a.reps):
            lowered_at, mix, _ = one_run(kind, a.pairs, rng)
            lows.append(lowered_at)
        hit = [x for x in lows if x is not None]
        out["results"][kind] = {"share_ever_lowered": len(hit) / a.reps,
                                "median_pairs_to_first_lowering": sorted(hit)[len(hit) // 2] if hit else None}
        print(kind, json.dumps(out["results"][kind]), flush=True)
    ends = []
    for _ in range(50):
        _, mix, pv = one_run("tied_boot_informative", 300, rng, p_inf=1.0)
        ends.append(mix)
    out["results"]["tied_boot_perfect_R_psi_300_pairs"] = {"final_mix_values": sorted(set(ends)),
                                                           "share_ending_at_0.75": ends.count(0.75) / len(ends),
                                                           "e_values_at_mix_0.75": pv.summary()["by_mix"]["0.75"]}
    print(json.dumps(out["results"]["tied_boot_perfect_R_psi_300_pairs"]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
