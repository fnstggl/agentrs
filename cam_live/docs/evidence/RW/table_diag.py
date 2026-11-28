"""Markdown table of the long-objective diagnostics (one row per arm and seed) from ``summarise_diag.py`` outputs.

    .venv/bin/python cam_live/docs/evidence/RW/table_diag.py summary.json [summary2.json ...]
"""
import json
import sys


def fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main(paths):
    print("| arm (outer utility, inner half-life, reward lr) | seed | decisions | meta-updates applied | learned agreement, thirds of pairs |"
          " agreement overall [95 % CS] | gain over the frozen R_psi (same pairs) | first mix change (decision) | mix at end | first decision unlike frozen |"
          " truth of chosen actions after the change, minus frozen [95 % bootstrap over decisions] |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in paths:
        d = json.load(open(p))
        for name, arm in d["arms"].items():
            hl = arm.get("half_life_s") or 86400.0
            lr = arm.get("lr_reward")
            label = f"{arm['arm']} ({arm.get('utility')}, half-life {int(hl)} s, lr {lr:g})" if lr is not None else \
                f"{arm['arm']} ({arm.get('utility')}, half-life {int(hl)} s)"
            for r in arm["runs"]:
                c = r["concordance_learned"]
                ci = r["truth_difference_ci95_decisions"]
                print(f"| {label} | {r['seed']} | {r['decisions']} | {r['meta_applied']} | "
                      f"{' / '.join(fmt(x) for x in r['learned_concordance_by_third_of_pairs'])} | "
                      f"{fmt(c['mean'])} [{fmt(c['lower'])}, {fmt(c['upper'])}] | "
                      f"{fmt(r.get('agreement_gain_over_frozen_init'))} | "
                      f"{fmt(r['first_mix_change_decision'])} | {fmt(r['mix_end'])} | "
                      f"{fmt(r['first_divergence_from_frozen'])} | "
                      f"{fmt(r['truth_difference_after_first_change'])}"
                      f"{'' if ci is None else f' [{fmt(ci[0])}, {fmt(ci[1])}]'} |")


if __name__ == "__main__":
    main(sys.argv[1:])
