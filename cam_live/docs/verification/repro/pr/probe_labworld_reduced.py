"""Reduced reproduction of C's LabWorld experiment (MR-02, MR-05, MR-08, MR-09; C.json scientific_failures).

Runs C's own learner rig (reward.diagnose.run_learner) for the 'learned' (R_psi from scratch, mix 0) and 'bootstrap'
arms on seeds 11 and 14 for the first 80 lifetimes (of 320 in the reported run) and checks:
  * the run is deterministic: the per-lifetime verified achievement equals the first 80 entries stored in
    docs/evidence/C/diagnostics.json for the same (kind, seed) (C claims bit-reproduction for 3 of 4 arms);
  * the reported qualitative result holds on these seeds: on seed 11 the learned-reward learner collapses (achieves
    ~0 while bootstrap does not) — i.e. the learned reward hurt, as reported;
  * meta-updates were really applied in the learned arm (psi trained by the outer loop).
Usage: .venv/bin/python cam_live/docs/verification/repro/pr/probe_labworld_reduced.py [--lifetimes 80] [--json PATH]
"""
import json
import multiprocessing as mp
import os
import sys

import _common as C

N = int(sys.argv[sys.argv.index("--lifetimes") + 1]) if "--lifetimes" in sys.argv else 80
STORED = os.path.join(C.HERE, "..", "..", "..", "evidence", "C", "diagnostics.json")


def job(args):
    kind, seed, n = args
    import torch
    torch.set_num_threads(1)
    from cam_live.reward.diagnose import ExperimentConfig, run_learner
    ec = ExperimentConfig(n_lifetimes=n)
    r = run_learner(kind, seed, ec, profile=False)
    return {k: r[k] for k in ("kind", "seed", "achieved", "utility", "n_meta_applied", "n_meta_skipped",
                              "policy_updates", "action_frequency_last_quarter", "seconds")}


if __name__ == "__main__":
    stored = {(r["kind"], r["seed"]): r for r in json.load(open(STORED))["experiment"]["runs"]}
    jobs = [(k, s, N) for s in (11, 14) for k in ("learned", "bootstrap")]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        out = pool.map(job, jobs)
    res = {(r["kind"], r["seed"]): r for r in out}
    for (k, s), r in sorted(res.items()):
        ref = stored[(k, s)]["achieved"][:N]
        same = sum(1 for a, b in zip(r["achieved"], ref) if a == b)
        C.check(f"{k} seed {s}: first {N} lifetimes reproduce the stored run exactly", same == N,
                f"{same}/{N} equal; mean achieved {sum(r['achieved']) / N:.3f} vs stored {sum(ref) / N:.3f}; "
                f"{r['seconds']:.0f} s")
    lq = lambda r: sum(r["achieved"][-N // 4:]) / (N // 4)
    C.info("last-quarter achievement of the reduced runs", {f"{k}/{s}": round(lq(r), 3) for (k, s), r in res.items()})
    C.check("seed 11: the learned-reward learner collapses below bootstrap-only (reported failure reproduced)",
            lq(res[("learned", 11)]) < lq(res[("bootstrap", 11)]) - 0.2,
            f"learned {lq(res[('learned', 11)]):.2f} vs bootstrap {lq(res[('bootstrap', 11)]):.2f}")
    C.check("learned arm: meta-updates were applied (psi trained by the outer loop)",
            all(res[("learned", s)]["n_meta_applied"] > 0 for s in (11, 14)),
            f"{[res[('learned', s)]['n_meta_applied'] for s in (11, 14)]}")
    C.info("learned arm last-quarter action frequencies", {s: res[("learned", s)]["action_frequency_last_quarter"]
                                                          for s in (11, 14)})
    sys.exit(C.finish())
