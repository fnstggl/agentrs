"""G3-core probe RW-1: is an UNLEARNED R_psi promoted by the validity evidence?

RW claims (docs/RW.md 2.1, reward/validity.py module doc): "an uninformative one (random init, frozen psi) is not
promoted". A frozen R_psi at its random initialisation is not necessarily uninformative: its output is a random
function of the signals and latents. The output layer of R_psi is initialised N(0, 0.02) with zero bias and the
bound is odd, so negating the output layer gives an EQUALLY LIKELY initialisation whose rewards are exactly negated.
The RW frozen arms show prospective concordances of -0.20 with anytime 95 % CS excluding 0 in some seeds
(docs/evidence/RW/diag_summary_600.json); the negated init then has +0.20.

Arms (debug scale, the RW controlled configuration of docs/evidence/RW/rw_meta_diag.py, 600 decisions, direct utility,
same seeds and worlds): ``frozen`` (as RW) and ``frozen_neg`` (same, output layer negated; reward step 0, so psi never
changes). Reports per seed: concordance, mix changes, and the true expected adopters of the chosen actions after the
first mix change against the frozen arm on the same decisions (the arms are identical until the mix moves).

    OMP_NUM_THREADS=4 nice -n 10 env PYTHONPATH=<freeze>/cam_live/src python probe_rw_flipped_frozen.py \
        --rw-dir <freeze>/cam_live/docs/evidence/RW --seeds 0 1 2 3 4 5 --out out/rw_flipped_frozen.json
"""
import argparse
import json
import random
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rw-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--decisions", type=int, default=600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.rw_dir)
    import torch
    import rw_meta_diag as diag
    from cam_live.reward import diagnose as dg
    orig = dg.make_stack
    state = {"neg": False}

    def patched(*args, **kw):
        model, store, clock, ml = orig(*args, **kw)
        if state["neg"]:
            with torch.no_grad():
                ml.reward.out.weight.neg_()
                ml.reward.out.bias.neg_()
        return model, store, clock, ml
    dg.make_stack = patched
    out = {"probe": "rw_flipped_frozen", "decisions": a.decisions, "runs": []}
    for s in a.seeds:
        res = {}
        for arm, neg in (("frozen", False), ("frozen_neg", True)):
            state["neg"] = neg
            r = diag.run(s, "frozen", a.decisions, 6, 41_000, 0.25, "direct", 86400.0, 0.0)
            res[arm] = r
        f, n = res["frozen"], res["frozen_neg"]
        ch = n["mix_changes"]
        row = {"seed": s,
               "frozen_concordance": f["validity"]["concordance_learned"],
               "neg_concordance": n["validity"]["concordance_learned"],
               "bootstrap_concordance": f["validity"]["concordance_bootstrap"]["mean"],
               "pairs": n["validity"]["pairs"], "informative": n["validity"]["informative_pairs"],
               "frozen_mix_changes": f["mix_changes"], "neg_mix_changes": ch,
               "neg_e_learned_vs_mix1": n["validity_e"]["1"]["e_learned_vs_mix"],
               "frozen_e_learned_vs_mix1": f["validity_e"]["1"]["e_learned_vs_mix"]}
        fr, nr = f["rows"], n["rows"]
        div = next((i for i, (x, y) in enumerate(zip(fr, nr)) if x[2] != y[2]), None)
        row["first_divergence"] = div
        if ch:
            i0 = ch[0][0]
            row["identical_before_first_change"] = all(x[2] == y[2] for x, y in zip(fr[:i0], nr[:i0]))
            d = [y[0] - x[0] for x, y in zip(fr[i0:], nr[i0:])]
            rng = random.Random(0)
            boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
            row["truth_after_first_change"] = {"neg": sum(y[0] for y in nr[i0:]) / len(d),
                                               "frozen": sum(x[0] for x in fr[i0:]) / len(d),
                                               "diff": sum(d) / len(d), "ci95": [boots[50], boots[1949]],
                                               "n_decisions": len(d)}
        out["runs"].append(row)
        print(json.dumps(row), flush=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
