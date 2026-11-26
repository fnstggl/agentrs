"""SP-9.10 (MASTER_SPEC Part 9.10): Tier 2 domain Education ecosystem. Workstream I-2 (dom2).

Every required mechanic (Learner knowledge state, Educational artifacts, Prerequisites, Participation, Learning
dynamics, Independent assessments, Retention of acquired knowledge) is exercised by recorded events in integrated
reference runs reached through crucible.experience.make(2, ...) in the default observation_only mode
(`base.section_metrics`). Learning is measured as improvement on NEW test problems (the evaluator's delayed test draws
fresh problems and excludes every published one), with content controls on the same world instance (same seed, same
individuals): a careful course (correct content), nothing published (no content) and the same course with the
misconception as every answer (wrong content), each studied by the population for the domain's horizon.

    improvement_on_new_problems_pvalue  one-sided paired t-test: per individual, delayed new-problem score with the
                                        correct course minus with no content (all enrolled individuals)
    content_control_order_holds         mean score correct > none > wrong (and a prose-only document does not beat
                                        no content by more than noise)
"""
import numpy as np
import pytest

from crucible.acceptance.v5 import evidence
from crucible.domains import base
from crucible.domains import education_ecosystem as E

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_9_10.py"
CONFIG, SEED = "numeracy_search", 0


def _final(policy):
    r = base.run(E.NAME, CONFIG, SEED, policy=policy, keep_env=True)
    w = r["env"].world
    L = w.modules["learn"]
    sc = L.final_scores(w, int(w.now + E.DELAY_DAYS * 86400), "sp-9.10-content-control", k=4)
    r["env"].close()
    return r, {i: float(np.mean(list(v.values()))) for i, v in sc.items()}


@pytest.mark.slow
@pytest.mark.timeout(4 * 3600)
def test_SP_9_10_education_ecosystem():
    from scipy import stats
    m, runs = base.section_metrics(E.NAME)
    ref, correct = _final("reference")
    _, none = _final("null")
    _, wrong = _final("wrong_keys")
    _, prose = _final("prose_only")
    ids = sorted(set(correct) & set(none) & set(wrong) & set(prose))
    c, n, wr, pr = (np.array([d[i] for i in ids]) for d in (correct, none, wrong, prose))
    t = stats.ttest_rel(c, n, alternative="greater")
    m["improvement_on_new_problems_pvalue"] = float(t.pvalue)
    m["content_control_order_holds"] = bool(c.mean() > n.mean() > wr.mean() and pr.mean() <= n.mean() + 0.01)
    m["mean_score_correct"] = round(float(c.mean()), 4)
    m["mean_score_none"] = round(float(n.mean()), 4)
    m["mean_score_wrong"] = round(float(wr.mean()), 4)
    m["mean_score_prose"] = round(float(pr.mean()), 4)
    m["individuals_compared"] = len(ids)
    m["reference_readers"] = int(ref["outcome"]["details"].get("readers", 0))
    evidence.record("SP-9.10", m, command=CMD, labels={"scripted": True},
                    notes=f"content controls on {CONFIG} seed {SEED}: correct {m['mean_score_correct']}, none "
                          f"{m['mean_score_none']}, wrong {m['mean_score_wrong']}, prose {m['mean_score_prose']} "
                          f"over {len(ids)} individuals; reference runs {m['reference_runs']} "
                          f"(successes {m['reference_successes']})")
    assert m["required_mechanics_listed"] == 7 and m["required_mechanics_exercised"] == 7, m
    assert m["required_mechanics_missing"] == 0, m
    assert m["observation_only_default"] and m["reachable_through_experience"], m
    assert m["improvement_on_new_problems_pvalue"] <= 0.05, m
    assert m["content_control_order_holds"], m
