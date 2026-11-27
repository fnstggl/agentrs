"""cam_live.eval.report — compact text digest of the evidence files in docs/evidence/S (used to write RESULTS.md).

    python -m cam_live.eval.report [--tag main_cpu]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, Optional

EVIDENCE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs",
                                        "evidence", "S"))


def ci(d: Optional[Dict[str, Any]], nd: int = 3) -> str:
    if not isinstance(d, dict) or d.get("mean") is None:
        return "n/a"
    m = d["mean"]
    if d.get("lo") is None:
        return f"{m:.{nd}f} (n={d.get('n', '?')})"
    s = f"{m:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"
    if d.get("p") is not None:
        s += f" p={d['p']:.3g}"
    return s


def load(name: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(EVIDENCE, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def baselines(tag: str) -> None:
    b = load(f"{tag}_baselines.json")
    if not b:
        return
    print(f"## {tag}: baselines (q={b['q']}, window={b['window']})")
    for c, v in b["conditions"].items():
        nq = v["N_q"]
        print(f"{c:13s} n={v['n_seeds']} AULC {ci(v['aulc'])} last2 {ci(v['last_two_lifetimes'])} slope "
              f"{ci(v['learning_slope_second_minus_first_half'])} eff {ci(v['efficiency'])} ach "
              f"{ci(v['verified_achievement_rate'])} reached {nq['reached']['k']}/{nq['n_units']} "
              f"Nq {ci(nq['restricted_mean_n'], 1)} train/dec {ci(v['compute_per_decision']['training_s'], 2)} "
              f"inf/dec {ci(v['compute_per_decision']['inference_s'], 2)} wall {ci(v['elapsed']['wall_s_per_run'], 0)}"
              f" noise {ci(v['action_shares']['noise_share'], 2)} act {ci(v['action_shares']['act_share'], 2)}")
    for c, v in b.get("pairwise_vs_E", {}).items():
        print(f"  E - {c:11s} AULC {ci(v['aulc_ref_minus_c'])} last2 {ci(v['last_two_ref_minus_c'])} slope "
              f"{ci(v['slope_ref_minus_c'])} eff {ci(v['efficiency_ref_minus_c'])}")
    for k, v in b.get("anchors", {}).items():
        print(f"  anchor {k}: progress {ci(v['verified_progress'])} eff {ci(v['quality'])} achieved "
              f"{v['achieved']['k']}/{v['achieved']['n']}")


def other(tag: str) -> None:
    c = load(f"{tag}_calibration.json")
    if c:
        print(f"## {tag}: calibration")
        for cond in ("E", "D", "E_mix0", "C"):
            if cond not in c:
                continue
            o = c[cond]["objective_success_forecasts"]
            a = c[cond]["action_succeeded"]
            g = c[cond]["goal_count_predictions"]
            print(f"{cond}: objective n={o.get('n')} clusters={o.get('clusters')} brier "
                  f"{ci(o.get('brier_ci_clustered'))} base {ci(o.get('brier_base_rate_ci_clustered'))} "
                  f"ece {o.get('ece_10')} auc {o.get('auc')} mean_forecast {o.get('mean_forecast')} rate "
                  f"{o.get('observed_rate')} | action.succeeded n={a.get('n')} brier {ci(a.get('brier_ci_clustered'))}"
                  f" base {ci(a.get('brier_base_rate_ci_clustered'))} | counts z2 {ci(g['mean_z2'])} cov80 "
                  f"{ci(g['coverage_80'])} | nll late-early {ci(c[cond]['nll_first_half_vs_second_half_of_lifetimes'])}"
                  f" excluded {c[cond]['non_prospective_predictions_excluded']}")
    a = load(f"{tag}_action_quality.json")
    if a:
        print(f"## {tag}: action quality")
        for cond in ("E", "D", "E_mix0", "C"):
            if cond in a:
                v = a[cond]
                print(f"{cond}: spearman {ci(v['spearman_predicted_vs_true_value_over_candidates'])} top-vs-mean "
                      f"{ci(v['true_value_gain_of_top_predicted_candidate_over_mean_alternative'])} top/best "
                      f"{ci(v['true_value_ratio_of_top_predicted_to_best'])} chosen/best "
                      f"{ci(v['true_value_ratio_of_chosen_to_best'])} experiments "
                      f"{ {k: v['randomised_experiments'][k] for k in ('n', 'concluded', 'supported', 'refuted')} }")
    d = load(f"{tag}_delayed_credit.json")
    if d:
        print(f"## {tag}: delayed credit")
        print({k: (ci(v) if isinstance(v, dict) and "mean" in v else v) for k, v in d.items()
               if k not in ("executed", "plan", "results_dir", "seeds", "family", "conditions", "scale",
                            "learning_efficiency_E_minus_E_no_credit")})
        le = d.get("learning_efficiency_E_minus_E_no_credit")
        if le:
            print(f"  E - E_no_credit AULC {ci(le['aulc'])} last2 {ci(le['last_two'])} reached E "
                  f"{le['N_q_E']['reached']['k']} no_credit {le['N_q_E_no_credit']['reached']['k']}")
    s = load(f"{tag}_shadow.json")
    if s:
        print(f"## {tag}: shadow (E)")
        for k, v in s.get("E", {}).get("targets", {}).items():
            print(f"  {k:28s} IPS {ci(v['ips_estimate_observed'], 2)} SNIPS {ci(v['snips_estimate_observed'], 2)} "
                  f"model {ci(v['model_prediction'], 2)} truth {ci(v['ground_truth_same_contexts'], 2)} ESS "
                  f"{ci(v['ess_fraction'], 2)}")
    r = load(f"{tag}_reward_vs_progress.json")
    if r:
        print(f"## {tag}: internal reward vs progress")
        for cond, v in r.items():
            if isinstance(v, dict) and "spearman_internal_return_vs_verified_progress" in v:
                print(f"  {cond}: rho {ci(v['spearman_internal_return_vs_verified_progress'])} zero-value reward "
                      f"share {ci(v['share_of_positive_internal_reward_to_zero_value_actions'])} decision share "
                      f"{ci(v['share_of_decisions_that_are_zero_value_actions'])}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main_cpu")
    a = ap.parse_args(argv)
    baselines(a.tag)
    other(a.tag)
    for name in sorted(glob.glob(os.path.join(EVIDENCE, "*.json"))):
        print("evidence:", os.path.basename(name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
