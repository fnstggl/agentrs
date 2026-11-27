"""The runtime prediction API on the real (debug-scale) model in LabWorld: prospective registration, combination of
heads and fast layer, immediate belief update without a gradient step, resolution, value of information."""
import dataclasses
import math

import pytest
import torch

from cam_live.contracts import (GoalStatus, OutcomeKind, OutcomeObservation, OutcomeQuery, OutcomeStatus,
                                VerifiedOutcome)
from cam_live.predict import dists as D
from cam_live.predict.evaluator import UnverifiedOutcomeError
from cam_live.predict.imagination import ImaginedDataError
from cam_live.predict.predictor import Predictor, at_horizons
from cam_live.predict.registry import PredictionRegistry, ResolutionBeforePredictionError, verify_log

DAY = 86400.0


def _acts(evs):
    return [e for e in evs if e.action.operation == "lab.act"]


def test_predictions_are_registered_before_the_method_returns(lab_rig):
    rig = lab_rig(n_lifetimes=0)
    cands = rig.candidates()
    now = rig.clock.now()
    assert len(rig.registry) == 0
    evs = rig.evaluate(candidates=cands)
    assert [e.action.action_id for e in evs] == [c.action_id for c in cands]
    acts = _acts(evs)
    n_pred = sum(len(e.predictions) for e in evs)
    assert n_pred == 2 * len(acts) > 0 and len(rig.registry) == n_pred + 1        # + the state-level success forecast
    for e in acts:
        assert sorted(p.query.horizon_s for p in e.predictions) == [6 * 3600.0, 2 * DAY]      # two horizons
        for p in e.predictions:
            stored = rig.registry.get(p.prediction_id)
            assert stored.made_at == now and rig.registry.registered_at(p.prediction_id) == now
            assert stored.resolved_status == OutcomeStatus.PENDING and stored.action_id == e.action.action_id
            assert p.query.metric.endswith(e.action.action_id) and p.query.exposure == e.action.arguments["effort"]
            assert p.dist.family == D.FAMILY_NEG_BINOMIAL and p.dist.mean > 0
            assert 0 < p.dist.epistemic_std < p.dist.std                          # epistemic and total reported apart
        assert 0.0 <= e.p_success <= 1.0 and e.voi is not None and e.voi >= 0 and e.uncertainty > 0
        assert e.cost.money == 2.0 and e.cost.seconds == 3600.0
    others = [e for e in evs if e.action.operation != "lab.act"]
    assert others and all(e.predictions == [] and e.extras["q_goal"] == 0.0 for e in others)
    assert verify_log(rig.registry._log.path) == len(rig.registry)

    # an "outcome" that is not strictly later than the prediction is refused by the registry
    p = acts[0].predictions[0]
    early = OutcomeObservation(p.query.query_id, OutcomeStatus.OBSERVED, 3.0, observed_at=now)
    with pytest.raises(ResolutionBeforePredictionError):
        rig.registry.resolve(p.prediction_id, early)
    assert rig.predictor.observe_outcomes([early])["skipped"] == {"not_after_prediction": 1}

    # the registry survives a restart with the same content
    again = PredictionRegistry(rig.registry._log.path, rig.clock)
    assert len(again) == len(rig.registry) and again.get(p.prediction_id).made_at == now


def test_outcomes_update_the_fast_belief_without_any_gradient_step(lab_rig):
    rig = lab_rig(n_lifetimes=0)
    cands = rig.candidates()
    ctx = rig.context()
    evs = rig.evaluate(ctx, cands)
    target = max(_acts(evs), key=lambda e: e.action.arguments["effort"])
    before = {p.query.horizon_s: p for p in target.predictions}
    hashes = rig.model.group_hashes()
    res = rig.execute(target.action)
    links = rig.predictor.adopt_queries(target.action.action_id, res.expected_outcomes)
    assert list(links) == [res.expected_outcomes[0].query_id]         # the environment's query is one of ours
    assert links[res.expected_outcomes[0].query_id] == before[2 * DAY].query.query_id

    rig.clock.advance(3 * DAY)
    mine = [p.query for p in target.predictions]
    outs = rig.verifier.measure(mine)
    assert all(o.status == OutcomeStatus.OBSERVED and o.verified for o in outs)
    high = [dataclasses.replace(o, value=float(4 * math.ceil(before[q.horizon_s].dist.mean) + 4))
            for q, o in zip(mine, outs)]                              # far above what was predicted
    info = rig.predictor.observe_outcomes(high)
    assert info["n"] == 2 and info["surprise"] > 0 and info["kl"] > 0 and not info["skipped"]
    assert {d["error"] for d in info["details"]} <= {"within_noise", "model_uncertain", "model_wrong",
                                                    "changed_conditions"}
    assert rig.model.group_hashes() == hashes                         # no weight changed: this was not a gradient step
    assert rig.fast.n_updates(OutcomeKind.COUNT) == 2

    again = rig.predictor.observe_outcomes(high + rig.verifier.measure(res.expected_outcomes))
    assert again["n"] == 0 and again["skipped"] == {"duplicate": 3}   # one event counts once, under either id
    pend = OutcomeObservation(mine[0].query_id, OutcomeStatus.PENDING)
    assert rig.predictor.observe_outcomes([pend])["skipped"] == {"not_matured": 1}
    assert rig.predictor.observe_outcomes([OutcomeObservation("q_unknown", OutcomeStatus.OBSERVED, 1.0,
                                                              observed_at=rig.clock.now())])["skipped"] == \
        {"unknown_query": 1}

    # the same candidates in the same context are now predicted differently, in the direction of the evidence
    evs2 = rig.evaluate(ctx, cands)
    after = {p.query.horizon_s: p for e in evs2 if e.action.action_id == target.action.action_id
             for p in e.predictions}
    for h in before:
        assert after[h].dist.mean > 1.3 * before[h].dist.mean
        assert after[h].dist.epistemic_std / after[h].dist.mean < before[h].dist.epistemic_std / before[h].dist.mean
    assert after[2 * DAY].made_at > before[2 * DAY].made_at

    # without a fast state the heads alone answer, and nothing adapts
    plain = Predictor(rig.model, None, rig.registry, rig.clock, heads=rig.heads)
    e0 = _acts(plain.evaluate_candidates(ctx, rig.task.objective, cands, rig.queries))[0]
    assert e0.voi is None and e0.predictions[0].dist.epistemic_std > 0
    assert plain.observe_outcomes(high)["n"] == 0


def test_resolve_fills_the_records_once_and_scores_them(lab_rig):
    rig = lab_rig(n_lifetimes=0)
    evs = rig.evaluate()
    target = _acts(evs)[0]
    res = rig.execute(target.action)
    rig.predictor.adopt_queries(target.action.action_id, res.expected_outcomes)
    assert rig.predictor.resolve(rig.verifier.measure(res.expected_outcomes)) == []        # still pending
    assert rig.registry.get(target.predictions[0].prediction_id).resolved_status == OutcomeStatus.PENDING
    rig.clock.advance(3 * DAY)
    env_out = rig.verifier.measure(res.expected_outcomes)             # arrives under the environment's query id
    done = rig.predictor.resolve(env_out)
    assert len(done) == 1 and done[0].query.horizon_s == 2 * DAY
    assert done[0].resolved_value == env_out[0].value and done[0].nll is not None and math.isfinite(done[0].nll)
    assert rig.predictor.resolve(env_out) == []                       # exactly once
    short = [p.query for p in target.predictions if p.query.horizon_s < DAY]
    assert len(rig.predictor.resolve(rig.verifier.measure(short))) == 1
    unobs = OutcomeObservation(_acts(evs)[1].predictions[0].query.query_id, OutcomeStatus.UNOBSERVABLE,
                               observed_at=rig.clock.now())
    closed = rig.predictor.resolve([unobs])
    assert closed[0].resolved_status == OutcomeStatus.UNOBSERVABLE and closed[0].nll is None
    assert len(rig.registry.scoreable()) == 2                         # unobservable ones are never scored
    assert rig.predictor.commit(target.predictions[0].decision_id, target.action.action_id) > 0


def test_success_forecasts_resolve_only_with_a_signed_resolved_objective_outcome(lab_rig):
    rig = lab_rig(n_lifetimes=0)
    rig.evaluate()
    rig.clock.advance(DAY)
    rig.evaluate()
    oid = rig.task.objective_id
    preds = rig.predictor.success_predictions(oid)
    assert len(preds) == 2 and preds[0].query.kind == OutcomeKind.BINARY
    assert preds[0].query.horizon_s == rig.deadline - preds[0].made_at > preds[1].query.horizon_s
    pending = rig.verifier.verify(rig.task)
    assert pending.status == GoalStatus.PENDING
    assert rig.predictor.resolve_objective(pending, rig.verifier.check_signature) == []
    rig.clock.advance(20 * DAY)
    final = rig.verifier.verify(rig.task)
    assert final.status == GoalStatus.FAILED
    with pytest.raises(UnverifiedOutcomeError):
        rig.predictor.resolve_objective(dataclasses.replace(final, status=GoalStatus.ACHIEVED),
                                        rig.verifier.check_signature)
    with pytest.raises(UnverifiedOutcomeError):
        rig.predictor.resolve_objective(VerifiedOutcome(oid, GoalStatus.ACHIEVED), rig.verifier.check_signature)
    done = rig.predictor.resolve_objective(final, rig.verifier.check_signature)
    assert [d.resolved_value for d in done] == [0.0, 0.0] and all(d.nll > 0 for d in done)
    report = rig.registry.score()
    assert report["n_scored"] == 2 and "binary" in report and report["refused"] == 0
    v = rig.predictor.belief_value(rig.context(), rig.task.objective, deadline=rig.deadline + 30 * DAY)
    assert 0.0 < v["p_success"] < 1.0 and v["b_norm"] > 0


def test_every_outcome_kind_is_predicted_registered_and_absorbed(lab_rig):
    rig = lab_rig(n_lifetimes=0)
    for v in (2.0, 2.5, 3.5, 3.0):
        rig.heads.scaler.update("gauge.level", v)
    tmpl = [OutcomeQuery("", "gauge.level", OutcomeKind.REAL, 600.0, description="level read after the action"),
            OutcomeQuery("", "items.from.{action}", OutcomeKind.COUNT, 3600.0, exposure=2.0, goal_relevant=True),
            OutcomeQuery("", "step.ok", OutcomeKind.BINARY, 60.0, description="the step completed"),
            OutcomeQuery("", "first.reply", OutcomeKind.TIME_TO_EVENT, DAY, description="time to the first reply"),
            OutcomeQuery("", "page.kind", OutcomeKind.CATEGORICAL, 60.0, categories=["form", "list", "error"])]
    cands = rig.candidates()[:3]
    evs = rig.predictor.evaluate_candidates(rig.context(), rig.task.objective, cands, tmpl)
    fams = [p.dist.family for p in evs[0].predictions]
    assert fams == [D.FAMILY_STUDENT_T, D.FAMILY_NEG_BINOMIAL, D.FAMILY_BERNOULLI, D.FAMILY_HAZARD,
                    D.FAMILY_CATEGORICAL]
    real = evs[0].predictions[0].dist
    assert 0.0 < real.mean < 6.0 and real.epistemic_std > 0           # reported in the metric's own units
    assert 0.0 < evs[0].predictions[2].dist.mean < 1.0
    assert 0.0 < evs[0].predictions[3].dist.mean <= DAY
    assert len(evs[0].predictions[4].dist.params["probs"]) == 3
    rig.clock.advance(2 * DAY)
    now = rig.clock.now()
    qs = [p.query for p in evs[0].predictions]
    outs = [OutcomeObservation(qs[0].query_id, OutcomeStatus.OBSERVED, 3.2, observed_at=now),
            OutcomeObservation(qs[1].query_id, OutcomeStatus.OBSERVED, 5.0, observed_at=now),
            OutcomeObservation(qs[2].query_id, OutcomeStatus.OBSERVED, 1.0, observed_at=now),
            OutcomeObservation(qs[3].query_id, OutcomeStatus.CENSORED, DAY, observed_at=now),
            OutcomeObservation(qs[4].query_id, OutcomeStatus.OBSERVED, 2.0, observed_at=now)]
    info = rig.predictor.observe_outcomes(outs)
    assert info["n"] == 4 and info["skipped"] == {"kind_not_in_fast_layer": 1} and info["kl"] > 0
    assert all(rig.fast.n_updates(k) == 1 for k in (OutcomeKind.REAL, OutcomeKind.COUNT, OutcomeKind.BINARY,
                                                    OutcomeKind.TIME_TO_EVENT))
    done = rig.predictor.resolve(outs)
    assert len(done) == 5 and all(math.isfinite(d.nll) for d in done)
    evs2 = rig.predictor.evaluate_candidates(rig.context(), rig.task.objective, cands, tmpl)
    assert evs2[0].predictions[2].dist.mean > evs[0].predictions[2].dist.mean            # binary: shifted towards 1
    assert evs2[0].predictions[3].dist.mean > evs[0].predictions[3].dist.mean            # censored: later than thought

    class Imagined:
        imagined = True

    with pytest.raises(ImaginedDataError):
        rig.predictor.observe_outcomes([Imagined()])
    with pytest.raises(ImaginedDataError):
        rig.predictor.observe_outcomes([dataclasses.replace(outs[0], source="imagination")])
    with pytest.raises(ImaginedDataError):
        rig.predictor.resolve([Imagined()])
    with pytest.raises(TypeError):
        rig.predictor.observe_outcomes([{"query_id": qs[0].query_id, "value": 1.0}])


def test_voi_is_decision_relevant_not_surprise(lab_rig):
    """A candidate whose outcome cannot change any later choice has zero VOI however unpredictable it is; a
    candidate whose outcome informs the goal-relevant comparison has positive VOI; knowing more lowers it."""
    rig = lab_rig(n_lifetimes=0, voi_samples=24)
    gen = torch.Generator().manual_seed(0)
    count = at_horizons(OutcomeQuery("", "lab.adopters.from.{action}", OutcomeKind.COUNT, goal_relevant=True,
                                     description="adopters attributed to this intervention"), [2 * DAY])
    gauge = [OutcomeQuery("", "lab.static", OutcomeKind.REAL, 600.0, description="static gauge reading")]

    def queries(a):
        return count if a.operation == "lab.act" else gauge if a.operation == "lab.noise" else []

    cands = [c for c in rig.candidates() if c.operation in ("lab.act", "lab.noise", "wait")]
    evs = rig.predictor.evaluate_candidates(rig.context(), rig.task.objective, cands, queries, generator=gen)
    noise = next(e for e in evs if e.action.operation == "lab.noise")
    acts = _acts(evs)
    assert noise.predictions[0].dist.std > 0 and noise.voi == 0.0     # unpredictable, and worthless to know
    assert noise.extras["info_gain_nats"] > 0                         # ...although it would change beliefs
    assert max(e.voi for e in acts) > 0
    assert all(e.voi == 0.0 for e in evs if e.action.operation == "wait")
    # after many observations of every intervention the remaining value of information shrinks
    v0 = sum(e.voi for e in acts)
    rig.clock.advance(3 * DAY)
    for _ in range(6):
        outs = []
        for e in acts:
            p = e.predictions[0]
            outs.append(OutcomeObservation(p.query.query_id, OutcomeStatus.OBSERVED, float(round(p.dist.mean)),
                                           observed_at=rig.clock.now(), dedup_key=f"{p.query.query_id}:{_}"))
        rig.predictor._absorbed.clear()                               # repeated measurements of the same question
        rig.predictor.observe_outcomes(outs)
    evs2 = rig.predictor.evaluate_candidates(rig.context(), rig.task.objective, cands, queries, generator=gen)
    assert sum(e.voi for e in _acts(evs2)) < 0.6 * v0
