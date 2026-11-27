"""G-pr D1: does the baseline of an outer row depend on that row's own measured outcome? (docs/C/FIXES.md)

    PYTHONPATH=cam_live/src .venv/bin/python cam_live/docs/evidence/C/fix_D1_baseline.py [--out PATH]

1. The verifier's scenario (repro/pr/probe_metalearner_path.py, section 3): one LabWorld lifetime of 24 decisions,
   policy updates every 8, verifier measurements authenticated, utility="return_to_go". For every measured outer
   row a: d b_a / d m_a with the baseline MetaLearner uses now (``MetaLearner.outer_baselines`` on the episode) and
   with the audited rule (mean over the rows of the other independent-outcome groups).
2. Exact expectation over all 2^T action sequences of a T-row lifetime whose row outcomes depend on their own
   actions: relative bias of the outer policy-gradient estimator under both baselines.
"""
import argparse
import dataclasses
import itertools
import json
import math
import os
import sys

import torch

from cam_live.reward import diagnose as dg
from cam_live.reward.metalearner import MetaLearner, _Episode


def scenario() -> dict:
    ec = dg.ExperimentConfig(min_independent=2)
    auth = dg.LabMeasurementAuth()
    model, store, clock, ml = dg.make_stack(7, ec, mix=0.0, check_measurement=auth)
    ml.new_lifetime()
    dg.lab_lifetime(ml, store, clock, ec, seed=4242, auth=auth, n_decisions=24, update_every=8)
    clock.advance(5 * 86400.0)
    eps = [e for e in ml._collect()[0] if e.ibq is not None]
    beta = ml.inner.cfg.beta
    rows_out = []
    for k, e in enumerate(eps):
        mea = [sum(x.utility for x in ml.ledger.by_exp(i) if x.kind == "measurement") for i in e.exp_ids]
        ts = [store.get(i).decision.ts for i in e.exp_ids]

        def rtg(m):
            return [sum(math.exp(-beta * max(0.0, ts[j] - ts[i])) * m[j] for j in range(i, len(m)))
                    for i in range(len(m))]
        rows = [(0, a) for a in range(len(e.exp_ids)) if e.has_u[a] and not e.held[a]]

        def old_rule(U):
            out = {}
            for _, a in rows:
                rest = [U[j] for _, j in rows if e.indep[j] != e.indep[a]]
                out[(0, a)] = sum(rest) / len(rest) if rest else 0.0
            return out
        new0, old0 = MetaLearner.outer_baselines([e], rows), old_rule(e.utility)
        for a in range(len(mea)):
            if not mea[a] or (0, a) not in new0:
                continue
            m2 = list(mea)
            m2[a] += 1.0
            U2 = rtg(m2)
            new1 = MetaLearner.outer_baselines([dataclasses.replace(e, utility=U2)], rows)
            rows_out.append({"episode": k, "row": a, "n_rows": len(e.exp_ids),
                             "d_baseline_d_own_outcome_now": new1[(0, a)] - new0[(0, a)],
                             "d_baseline_d_own_outcome_audited_rule": old_rule(U2)[(0, a)] - old0[(0, a)],
                             "baseline_group": e.group[a], "independent_group": e.indep[a]})
    return {"rows": rows_out, "max_abs_now": max(abs(r["d_baseline_d_own_outcome_now"]) for r in rows_out),
            "max_abs_audited_rule": max(abs(r["d_baseline_d_own_outcome_audited_rule"]) for r in rows_out)}


def exact_bias(T: int = 4, disc: float = 0.9) -> dict:
    D = torch.float64
    pi = torch.softmax(torch.tensor([0.3, -0.4], dtype=D), 0)
    g = torch.Generator().manual_seed(0)
    mu = [torch.randn(2, dtype=D, generator=g) * 2 for _ in range(T)]
    score = lambda a: torch.eye(2, dtype=D)[a] - pi                                # noqa: E731
    true = sum((torch.diag(pi) - torch.outer(pi, pi)) @ mu[t] for t in range(T)) / T
    est = {"now": torch.zeros(2, dtype=D), "audited_rule": torch.zeros(2, dtype=D), "no_baseline": torch.zeros(2, dtype=D)}
    for seq in itertools.product((0, 1), repeat=T):
        p = math.prod(float(pi[a]) for a in seq)
        m = [float(mu[t][a]) for t, a in enumerate(seq)]
        U = [sum(disc ** (j - t) * m[j] for j in range(t, T)) for t in range(T)]
        ep = _Episode(0, [], "s", "q", None, [f"e{i}" for i in range(T)], U, [True] * T, ["rtg:life=0:held=0"] * T,
                      [False] * T, set(), [], indep=[f"g{i}" for i in range(T)], prefix_base=[0.5] * T)
        now = MetaLearner.outer_baselines([ep], [(0, t) for t in range(T)])
        for t, a in enumerate(seq):
            old = sum(U[j] for j in range(T) if j != t) / (T - 1)
            est["now"] += p * score(a) * (U[t] - now[(0, t)]) / T
            est["audited_rule"] += p * score(a) * (U[t] - old) / T
            est["no_baseline"] += p * score(a) * U[t] / T
    return {"T": T, "true_gradient": true.tolist(),
            **{f"relative_bias_{k}": float((v - true).norm() / true.norm()) for k, v in est.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fix_D1_baseline.json"))
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    res = {"command": "PYTHONPATH=cam_live/src .venv/bin/python cam_live/docs/evidence/C/fix_D1_baseline.py",
           "verifier_scenario": scenario(), "exact_expectation": exact_bias()}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps({"max_abs_d_baseline_now": res["verifier_scenario"]["max_abs_now"],
                      "max_abs_d_baseline_audited_rule": res["verifier_scenario"]["max_abs_audited_rule"],
                      "exact": res["exact_expectation"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
