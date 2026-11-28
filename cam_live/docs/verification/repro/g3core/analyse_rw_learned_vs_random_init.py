"""G3-core analysis RW-3 (D-RW1): learned R_psi vs an UNLEARNED R_psi of either sign, same seeds / worlds.

Inputs: RW's 600-decision summary (docs/evidence/RW/diag_summary_600.json: learned arms at reward step 3e-3 and 1e-4
with their frozen comparisons) and this directory's out/rw_flipped_frozen.json (frozen and negated-frozen arms of the
same seeds). A negated output layer is an equally likely initialisation (N(0, 0.02), zero bias, odd bound), so
{frozen, frozen_neg} is a sign-symmetric sample of unlearned rewards.

    python analyse_rw_learned_vs_random_init.py --rw-summary <freeze>/cam_live/docs/evidence/RW/diag_summary_600.json \
        --flipped out/rw_flipped_frozen.json > out/rw_learned_vs_random_init.txt
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rw-summary", required=True)
    ap.add_argument("--flipped", required=True)
    a = ap.parse_args()
    rw = json.load(open(a.rw_summary))
    fl = {r["seed"]: r for r in json.load(open(a.flipped))["runs"]}
    print("seed | unlearned R_psi concordance: init / negated init (promoted?) | learned lr 3e-3 | learned lr 1e-4 (promoted?)")
    learned = {}
    for arm in rw["arms"].values():
        for r in arm["runs"]:
            learned.setdefault(r["seed"], {})[arm["lr_reward"]] = (r["concordance_learned"]["mean"], r["mix_end"] < 1.0)
    unl, l4 = [], []
    for s in sorted(fl):
        f = fl[s]
        fc, nc = f["frozen_concordance"]["mean"], f["neg_concordance"]["mean"]
        np_ = bool(f["neg_mix_changes"])
        unl += [(fc, bool(f["frozen_mix_changes"])), (nc, np_)]
        L = learned.get(s, {})
        hi = L.get(0.003)
        lo = L.get(0.0001)
        if lo:
            l4.append(lo)
        tr = f.get("truth_after_first_change")
        print(f"{s} | {fc:+.3f} / {nc:+.3f} ({'promoted at ' + str(f['neg_mix_changes'][0][0]) if np_ else 'no'}"
              f"{'; truth after: %+.2f [%+.2f, %+.2f] over %d decisions' % (tr['diff'], tr['ci95'][0], tr['ci95'][1], tr['n_decisions']) if tr else ''})"
              f" | {('%+.3f' % hi[0]) + (' promoted' if hi[1] else '') if hi else '-'}"
              f" | {('%+.3f' % lo[0]) + (' promoted' if lo[1] else '') if lo else '-'}")
    m = lambda xs: sum(xs) / len(xs)
    print(f"unlearned (both signs, n={len(unl)}): mean concordance {m([x for x, _ in unl]):+.3f}, "
          f"mean |concordance| {m([abs(x) for x, _ in unl]):.3f}, promoted {sum(p for _, p in unl)} of {len(unl)}")
    print(f"learned at the agent's reward step 1e-4 (n={len(l4)}): mean concordance {m([x for x, _ in l4]):+.3f}, "
          f"promoted {sum(p for _, p in l4)} of {len(l4)}")


if __name__ == "__main__":
    main()
