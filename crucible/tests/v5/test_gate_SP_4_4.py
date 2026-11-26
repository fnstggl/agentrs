"""SP-4.4 (MASTER_SPEC Part 4.4): participants choose content according to the 12 listed factors (interests,
relevance, previous experience, attention, content characteristics, social relationships, creator familiarity,
platform distribution, novelty, satisfaction, competing content, current needs), learned from empirical data where
possible; continuing, sharing, subscribing and disengaging are actual simulated decisions.

Measured on the integrated world of the one population (tests/v5/pop_world.py: crucible.population with its web
browsing and the social internet, 6 simulated days):

choice_factors_used >= 12: `population.choice.factor_effects` evaluates the content-choice decisions the world uses
(the fitted article-open model article_open_semantic@1 on page slates, the resource choice, the class-1 continue,
evaluate, purchase, abandon, recommend and follow decisions) on 400 real individuals' feature rows built from the
world's state, perturbing one factor's input at a time (an alternative-specific input of the resource choice for one
alternative, the attention ratio by a quarter of the day's budget, other inputs by one cohort standard deviation; see
population.choice); a factor is used when it moves a decision probability by more than 1e-3 on average;
decision_kinds_simulated >= 4: continue (an article opened from a slate or another page read), share (a
recommendation to a contact or a share in the social graph), subscribe (following a creator or a board, a
newsletter subscription, a community join) and disengage (abandoning a resource, unfollowing, leaving a community),
each counted when the world recorded at least one such decision of the population;
fitted_models_in_default_consumption >= 2: fitted models behind counted (organic, runtime-backed) population decisions
of this default world (`calibration.inworld.audit`)."""
import numpy as np
import pytest

from crucible.acceptance.v5 import evidence
from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.calibration import inworld
from crucible.population import analysis as A
from crucible.population import choice
from crucible.substrate import columns
from pop_world import integrated

CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-4.4"
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def _kinds(w) -> dict:
    lg = A.visit_log(w)
    soc = A.social_behaviours(w)
    ev = list(w.log)
    pop_actor = lambda a: isinstance(a, str) and a[:1] == "p" and a[1:].isdigit()  # noqa: E731
    reads = sum(int(e.payload["extra"].get("opened", 0)) for e in ev if e.type == "cols.batch" and
                e.payload["kind"] == "popweb.article_open")
    recommend = sum(int(e.payload["n"]) for e in ev if e.type == "cols.batch" and e.payload["kind"] == "popweb.recommend")
    follows = sum(1 for e in ev if e.type == "forum.follow" and pop_actor(e.payload["follower"]))
    return {"continue": int(reads + int((lg["pages"] > 1).sum() if len(lg["ind"]) else 0)),
            "share": int(recommend + soc["share_content"]),
            "subscribe": int(follows + soc["follow_creators"] + soc["subscribe"] + soc["join_communities"]),
            "disengage": int((lg["abandon"].sum() if len(lg["ind"]) else 0) + soc["leave_resources"])}


def test_SP_4_4_content_choice_factors_and_decisions():
    r = integrated()
    w = r["world"]
    known = columns.col(w, "pop", "known")
    cand = np.flatnonzero((known[:, 0] >= 0) & (columns.col(w, "pop", "class") <= 2))
    idx = cand[np.linspace(0, cand.size - 1, min(400, cand.size)).astype(int)] if cand.size else cand
    fe = choice.factor_effects(w, idx)
    kinds = _kinds(w)
    au = inworld.audit(w, usage=r["usage"])
    fitted = au.fitted_used()
    m = {"choice_factors_used": int(sum(1 for v in fe.values() if v["used"])),
         "decision_kinds_simulated": int(sum(1 for v in kinds.values() if v > 0)),
         "fitted_models_in_default_consumption": len(fitted), "cohort": int(idx.size),
         "fitted_models": ",".join(fitted), **{f"decisions_{k}": v for k, v in kinds.items()}}
    for f, v in fe.items():
        m[f"factor_{f}"] = bool(v["used"])
    evidence.record("SP-4.4", m, command=CMD, notes=(
        f"Factor effects (mean absolute change of the decision probability per perturbed input): "
        f"{ {f: v['inputs'] for f, v in fe.items()} }. Decision counts: {kinds}."))
    assert idx.size > 50
    crit = next(e for e in MF.load()["requirements"] if e["id"] == "SP-4.4")["criteria"]
    fails = [msg for c in crit for ok, msg in [criterion_met(c, m)] if not ok]
    assert not fails, (fails, m)
