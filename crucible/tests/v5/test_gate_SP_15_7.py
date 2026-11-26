"""SP-15.7 - Paired world comparisons, prespecified metrics, development and confirmation experiments separated,
independent holdouts, transparent failure reporting; no winner selection on the sealed set (access budget audited)
(MASTER_SPEC Part 15.7; workstream N).

Protocol executed against the independent evaluator process (which alone holds the sealed salt):

1. development: two configurations of the weights learner are trained on training items and compared on validation
   items through the evaluator (crossed seed x world interval); the better one is chosen on these results only;
2. preregistration: the hypothesis, metric, analysis, seeds, threshold and the chosen configuration are registered
   with the evaluator before any confirmation item exists;
3. confirmation: one access to a sealed set of evaluation-family items (budget 1) runs the chosen configuration and
   the cannot-learn control on the same sealed worlds (paired); the result is recorded whatever it is;
4. a second access is refused by the budget (the set is then contaminated and nothing more is served), a
   confirmation item cannot be opened without the preregistration, and a development item cannot be reused for
   confirmation.
"""
from __future__ import annotations

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import distribution as D
from crucible.research.v5 import evaluator as EV
from crucible.research.v5 import hstats as HS
from crucible.research.v5 import instrument as I
from crucible.research.v5 import learners as LR
from crucible.research.v5.episodes import lever_entries, run_episode
from crucible.research.v5.sampler import build_env

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]
SEEDS = (0, 1, 2)


def _trained(cfg, seed, reg, fams):
    L = LR.make("weights", seed, **cfg)
    rng = np.random.default_rng(seed)
    for i in range(18):
        f = fams[i % len(fams)]
        for _ in range(3):
            run_episode(build_env(reg[f], 760_000 + 100 * seed + i, {}, {"mode": "assigned"}), L, learn=True,
                        rng=rng, verify=False)
    L.frozen = True
    return L


def test_SP15_7_dev_confirmation_and_sealed_budget(tmp_path):
    reg = D.build_registry()
    d = I.default_design(reg)
    fams = d.train_families
    configs = {"lr0.05": {"lr": 0.05}, "lr0.2": {"lr": 0.2}}
    ev = EV.Evaluator.start(sealed_dir=str(tmp_path / "sealed"))
    try:
        dev_worlds = [(f, 1_000_501 + k) for f in fams for k in range(2)]
        dev = {}
        for name, cfg in configs.items():
            rows = []
            for s in SEEDS:
                L = _trained(cfg, s, reg, fams)
                row = []
                for f, w in dev_worlds:
                    t = ev.open_episode(f, w, purpose=f"dev:{name}")
                    row.append(EV.run_remote_episode(ev, t, L, np.random.default_rng(s))["score"])
                rows.append(row)
            dev[name] = np.array(rows)
        dd = HS.crossed_diff(dev["lr0.05"], dev["lr0.2"], paired_seeds=True)
        chosen = "lr0.2" if dd["estimate"] > 0 else "lr0.05"
        evals = sorted(w["id"] for w in reg.worlds(tier=1, family_partition="evaluation", source="crucible.rapid")
                       if w["id"] in set(lever_entries(reg, "evaluation")))
        no_prereg = ev.call("open_episode", evals[0], 7, purpose="confirmation")
        prereg = ev.preregister({"hypothesis": f"weights learner ({chosen}) scores above the cannot-learn control",
                                 "metric": "verified normalised score", "analysis": "hstats.crossed_diff, paired seeds",
                                 "seeds": list(SEEDS), "level": 0.95, "min_effect": 0.0, "set": "final",
                                 "chosen_by": "development results on validation items", "chosen": chosen})
        ev.must("declare_sealed", "final", evals, 2, 1)
        conf = ev.must("open_sealed", "final", ack=EV.SEALED_ACK, purpose="confirmation", prereg=prereg,
                       copies=len(SEEDS) + 1)
        items = sorted({t["item"] for t in conf["tickets"]})
        by_item = {it: [t["ticket"] for t in conf["tickets"] if t["item"] == it] for it in items}
        A, B = [], []
        for si, s in enumerate(SEEDS):
            L = _trained(configs[chosen], s, reg, fams)
            B.append([EV.run_remote_episode(ev, by_item[it][si], L, np.random.default_rng(s))["score"]
                      for it in items])
        hold = [EV.run_remote_episode(ev, by_item[it][len(SEEDS)], LR.make("hold", 0), np.random.default_rng(0))["score"]
                for it in items]
        A = [hold for _ in SEEDS]
        res = HS.crossed_diff(np.array(A), np.array(B), paired_seeds=True)
        second = ev.call("open_sealed", "final", ack=EV.SEALED_ACK, purpose="second look", prereg=prereg)
        reuse = ev.call("open_episode", dev_worlds[0][0], dev_worlds[0][1], purpose="confirmation", prereg=prereg)
        status = ev.must("sealed_status")["sets"]["final"]
        log = ev.must("log")["log"]
    finally:
        ev.close()
    dev_parts = {e["partition"] for e in log if e["op"] == "open" and e["purpose"].startswith("dev")}
    conf_parts = {e["partition"] for e in log if e["op"] == "open" and e["purpose"] == "confirmation"}
    separated = dev_parts == {"validation"} and conf_parts == {"sealed"} and not no_prereg.get("ok") and \
        not second.get("ok") and status["contaminated"]
    violations = max(0, status["accesses"] - status["budget"])
    record("SP-15.7", {"confirmation_runs_with_prespecified_metrics": int(conf["access"] == 1 and bool(prereg)),
                       "dev_confirmation_separated": bool(separated), "sealed_access_budget_violations": violations,
                       "confirmation_effect": float(res["estimate"]), "confirmation_ci_low": float(res["lo"]),
                       "dev_effect_lr02_minus_lr005": float(dd["estimate"]), "sealed_items": len(items),
                       "second_access_refused": not second.get("ok"),
                       "dev_item_reuse_refused": not reuse.get("ok")},
           notes=f"Chosen on development results: {chosen} (dev difference {dd}). Confirmation on the sealed set, "
                 f"reported whatever it shows: {res} - verdict {HS.verdict(res)}. Sealed status {status}.")
    assert separated and violations == 0 and not reuse.get("ok")
    assert conf["access"] == 1 and len(items) >= 2
