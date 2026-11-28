"""G2-learn: compare a re-run of A's / D's recorded experiments with the committed evidence files.

    # NA-13 (A): seed 0 of the fast-adaptation comparison, full length (~22 min with 8 threads here)
    cd /workspace/camlive/cam_live && PYTHONPATH=src ../.venv/bin/python -m cam_live.nn.fast_compare --steps 2000 \
        --seeds 0 --threads 8 --out RERUN_A.json
    .venv/bin/python cam_live/docs/verification/repro/g2learn/compare_evidence.py fast RERUN_A.json

    # NA-08 / CL-02 / CL-04 / RC-08 (D): D's debug-scale experiments on the current tree (~7 min)
    cd /workspace/camlive && PYTHONHASHSEED=0 .venv/bin/python -m cam_live.credit.diagnose \
        --only learning,credit,retention --threads 1 --out RERUN_D.json --work WORKDIR
    .venv/bin/python cam_live/docs/verification/repro/g2learn/compare_evidence.py diagnose RERUN_D.json

Prints the largest differences and the conclusions both files lead to (run from /workspace/camlive).
"""
import json
import sys

HERE = "cam_live/docs/evidence"
kind, path = sys.argv[1], sys.argv[2]
new = json.load(open(path))
if kind == "fast":
    old = json.load(open(f"{HERE}/A/fast_adaptation.json"))
    seed = new["targets"]["gaussian"]["per_seed"][0]["seed"]
    n, worst = 0, (0.0, None)
    for t in ("gaussian", "count"):
        pa = next(p for p in old["targets"][t]["per_seed"] if p["seed"] == seed)["methods"]
        pb = new["targets"][t]["per_seed"][0]["methods"]
        for m in pa:
            for split in ("in_distribution", "held_out_families"):
                for k in ("mean_nll_k_ge_1", "mean_rmse_k_ge_1"):
                    d = abs(pa[m][split][k] - pb[m][split][k])
                    n += 1
                    worst = max(worst, (d, (t, m, split, k)), key=lambda x: x[0])
                for ss, v in pa[m][split]["per_support_size"].items():
                    for k in ("nll", "rmse"):
                        d = abs(v[k] - pb[m][split]["per_support_size"][ss][k])
                        n += 1
                        worst = max(worst, (d, (t, m, split, ss, k)), key=lambda x: x[0])
            print(f"{t:8s} {m:20s} held-out NLL recorded {pa[m]['held_out_families']['mean_nll_k_ge_1']:.4f} "
                  f"re-run {pb[m]['held_out_families']['mean_nll_k_ge_1']:.4f} | in-distribution recorded "
                  f"{pa[m]['in_distribution']['mean_nll_k_ge_1']:.4f} re-run {pb[m]['in_distribution']['mean_nll_k_ge_1']:.4f}")
    print(f"seed {seed}: {n} recorded numbers compared, largest absolute difference {worst[0]:.6g} at {worst[1]}")
    print("verdicts (re-run):", json.dumps(new.get("verdict")))
elif kind == "diagnose":
    old = json.load(open(f"{HERE}/D/diagnostics.json"))
    print("recorded from", old["_code"].get("commit"), "| re-run on", new["_code"].get("commit"),
          "uncommitted:", new["_code"].get("uncommitted"))
    lb, la = new["learning"]["heldout_summary_before"], new["learning"]["heldout_summary_after"]
    ob, oa = old["learning"]["heldout_summary_before"], old["learning"]["heldout_summary_after"]
    print(f"learning: held-out NLL {ob['nll']:.6f} -> {oa['nll']:.6f} recorded | {lb['nll']:.6f} -> {la['nll']:.6f} re-run")
    print("learning counters recorded / re-run:", old["learning"]["counters"], new["learning"]["counters"])
    for est in ("rudder", "cocoa"):
        so, sn = old["credit"]["selection"]["scores"][est], new["credit"]["selection"]["scores"][est]
        print(f"credit {est}: mean z^2 {so['mean_sq_z']:.3f} / RMSE {so['rmse']:.3f} recorded | "
              f"{sn['mean_sq_z']:.3f} / {sn['rmse']:.3f} re-run; passes {so['passes']} / {sn['passes']}")
    print("credit default recorded / re-run:", old["credit"]["selection"]["default_after"],
          new["credit"]["selection"]["default_after"])
    print("credit hold-out recorded / re-run:", old["credit"]["holdout"], new["credit"]["holdout"])
    ro, rn = old["retention"]["summary"], new["retention"]["summary"]
    for k in ("nll_increase_on_A_naive", "nll_increase_on_A_retention", "se_naive", "se_retention",
              "decision_agreement_A_naive", "decision_agreement_A_retention"):
        print(f"retention {k}: recorded {ro[k]:.4f} re-run {rn[k]:.4f}")
else:
    sys.exit("kind must be 'fast' or 'diagnose'")
