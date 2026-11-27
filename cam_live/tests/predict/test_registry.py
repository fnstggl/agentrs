"""PredictionRegistry / ExperimentRegistry: prospective, append-only, resolved exactly once, tamper-evident."""
import json
import math
import os

import numpy as np
import pytest
import scipy.stats as st

from cam_live.contracts import (Distribution, ExperimentRecord, OutcomeKind, OutcomeObservation, OutcomeQuery,
                                OutcomeStatus, PredictionRecord, RunFiles, SimClock)
from cam_live.predict import dists as D
from cam_live.predict.calibration import RetrospectivePredictionError
from cam_live.predict.imagination import ImaginedDataError, imagined_prediction
from cam_live.predict.registry import (AlreadyResolvedError, DuplicatePredictionError, ExperimentRegistry,
                                       PredictionRegistry, RegistryError, ResolutionBeforePredictionError,
                                       TamperedRegistryError, UnknownRecordError, verify_log, welch_test)

DAY = 86400.0


def _pred(i, clock, kind=OutcomeKind.BINARY, dist=None, horizon=3600.0, made_at=None, exp=""):
    dist = dist or D.Bernoulli.from_prob(0.7).to_record()
    q = OutcomeQuery(f"q{i}", f"metric.{i}", kind=kind, horizon_s=horizon, issued_at=clock.now(), experiment_id=exp)
    return PredictionRecord(f"p{i}", q, dist, clock.now() if made_at is None else made_at, decision_id=f"d{i}")


def _obs(i, clock, value=1.0, status=OutcomeStatus.OBSERVED, **kw):
    return OutcomeObservation(f"q{i}", status, value, observed_at=clock.now(), **kw)


@pytest.fixture
def reg(tmp_path):
    clock = SimClock()
    return PredictionRegistry(str(tmp_path / RunFiles.PREDICTIONS), clock), clock, tmp_path


def test_register_then_resolve_once_and_score(reg):
    r, clock, _ = reg
    stored = r.register(_pred(0, clock))
    assert stored.resolved_status == OutcomeStatus.PENDING and stored.nll is None and len(r) == 1
    assert [p.prediction_id for p in r.pending()] == ["p0"] and r.pending(now=clock.now()) == []
    clock.advance(3600.0)
    assert [p.prediction_id for p in r.pending(now=clock.now())] == ["p0"]      # now due
    done = r.resolve("p0", _obs(0, clock, 1.0, verified=True))
    np.testing.assert_allclose(done.nll, -math.log(0.7), rtol=1e-9)
    assert done.resolved_at == clock.now() and done.resolved_value == 1.0 and r.pending() == []
    with pytest.raises(AlreadyResolvedError):
        r.resolve("p0", _obs(0, clock, 0.0))
    assert r.get("p0").resolved_value == 1.0                                     # the first resolution stands
    with pytest.raises(UnknownRecordError):
        r.resolve("nope", _obs(0, clock))
    # callers get copies: mutating a returned record changes nothing
    done.dist.params["p"] = 0.0
    done.resolved_value = 0.0
    assert r.get("p0").resolved_value == 1.0 and r.get("p0").dist.params["p"] != 0.0


def test_predictions_cannot_be_edited_or_registered_resolved(reg):
    r, clock, _ = reg
    r.register(_pred(0, clock))
    edited = _pred(0, clock, dist=D.Bernoulli.from_prob(0.01).to_record())
    with pytest.raises(DuplicatePredictionError):
        r.register(edited)
    with pytest.raises(DuplicatePredictionError):
        r.register(_pred(0, clock))                                              # even an identical copy
    assert abs(r.get("p0").dist.params["p"] - 0.7) < 1e-12
    pre = _pred(1, clock)
    pre.resolved_at, pre.resolved_value, pre.resolved_status = clock.now(), 1.0, OutcomeStatus.OBSERVED
    with pytest.raises(RegistryError):
        r.register(pre)
    with pytest.raises(RegistryError):
        r.register(_pred(2, clock, made_at=clock.now() + 10.0))                  # dated in the future
    with pytest.raises(RegistryError):
        r.register(_pred(3, clock, dist=Distribution("cauchy", {})))
    with pytest.raises(RegistryError):
        r.register(_pred(4, clock, kind=OutcomeKind.COUNT))                      # Bernoulli cannot predict a count
    with pytest.raises(RegistryError):
        r.register({"prediction_id": "x"})
    assert len(r) == 1


def test_resolution_must_be_strictly_after_the_prediction(reg):
    r, clock, _ = reg
    r.register(_pred(0, clock))
    with pytest.raises(ResolutionBeforePredictionError):
        r.resolve("p0", _obs(0, clock))                                          # same instant
    early = OutcomeObservation("q0", OutcomeStatus.OBSERVED, 1.0, observed_at=clock.now() - 5.0)
    with pytest.raises(ResolutionBeforePredictionError):
        r.resolve("p0", early)
    # back-dating made_at does not help: the registration time counts as well
    clock.advance(100.0)
    r.register(_pred(1, clock, made_at=clock.now() - 90.0))
    sneaky = OutcomeObservation("q1", OutcomeStatus.OBSERVED, 1.0, observed_at=clock.now() - 50.0)
    with pytest.raises(ResolutionBeforePredictionError):
        r.resolve("p1", sneaky)
    assert r.get("p0").resolved_status == OutcomeStatus.PENDING and r.get("p1").resolved_status == OutcomeStatus.PENDING
    clock.advance(1.0)
    assert r.resolve("p1", _obs(1, clock)).resolved_status == OutcomeStatus.OBSERVED


def test_statuses_values_and_imagined_outcomes(reg):
    r, clock, _ = reg
    hz = D.DiscreteHazard([0.0, 0.0], [0.0, 1800.0, 3600.0]).to_record()
    r.register(_pred(0, clock))
    r.register(_pred(1, clock, kind=OutcomeKind.TIME_TO_EVENT, dist=hz))
    r.register(_pred(2, clock, kind=OutcomeKind.COUNT, dist=D.Poisson(2.0, 3.0).to_record()))
    r.register(_pred(3, clock))
    r.register(_pred(4, clock, kind=OutcomeKind.REAL, dist=D.Normal(0.0, 1.0).to_record()))
    clock.advance(3600.0)
    with pytest.raises(RegistryError):
        r.resolve("p0", _obs(0, clock, None, OutcomeStatus.PENDING))             # pending is not a resolution
    with pytest.raises(RegistryError):
        r.resolve("p0", _obs(1, clock))                                          # answers another query
    with pytest.raises(RegistryError):
        r.resolve("p0", _obs(0, clock, None))                                    # observed without a value
    with pytest.raises(RegistryError):
        r.resolve("p0", _obs(0, clock, float("nan")))
    with pytest.raises(RegistryError):
        r.resolve("p4", _obs(4, clock, 1.0, OutcomeStatus.CENSORED))             # censoring only for time-to-event
    with pytest.raises(RegistryError):
        r.resolve("p0", {"query_id": "q0", "status": "observed", "value": 1.0})

    class Imagined(OutcomeObservation):
        imagined = True

    with pytest.raises(ImaginedDataError):
        r.resolve("p0", Imagined("q0", OutcomeStatus.OBSERVED, 1.0, observed_at=clock.now()))
    assert all(p.resolved_status == OutcomeStatus.PENDING for p in r.records())  # nothing above left a trace
    # censored time-to-event: a real observation, scored with the survival function
    cen = r.resolve("p1", _obs(1, clock, 3600.0, OutcomeStatus.CENSORED))
    np.testing.assert_allclose(cen.nll, -2 * math.log(0.5), rtol=1e-9)
    cnt = r.resolve("p2", _obs(2, clock, 4.0))
    np.testing.assert_allclose(cnt.nll, -st.poisson.logpmf(4, 6.0), rtol=1e-9)   # rate 2 x exposure 3
    # unobservable: closed without a value or a likelihood, and never scoreable
    un = r.resolve("p3", _obs(3, clock, 123.0, OutcomeStatus.UNOBSERVABLE))
    assert un.resolved_value is None and un.nll is None and un.resolved_status == OutcomeStatus.UNOBSERVABLE
    assert sorted(p.prediction_id for p in r.scoreable()) == ["p1", "p2"]
    # an imagined *prediction* may be registered (it is a prediction), keeps its flag, and is scored by reality
    ip = imagined_prediction(OutcomeQuery("q9", "m", kind=OutcomeKind.BINARY, horizon_s=60.0),
                             D.Bernoulli.from_prob(0.5).to_record(), clock.now(), horizon_steps=3)
    r.register(ip)
    assert r.get(ip.prediction_id).imagined_horizon_steps == 3


def test_log_is_append_only_reloadable_and_tamper_evident(reg):
    r, clock, tmp = reg
    path = str(tmp / RunFiles.PREDICTIONS)
    for i in range(4):
        r.register(_pred(i, clock))
    clock.advance(10.0)
    r.resolve("p1", _obs(1, clock, 0.0))
    size_before = os.path.getsize(path)
    with pytest.raises(AlreadyResolvedError):
        r.resolve("p1", _obs(1, clock, 1.0))
    assert os.path.getsize(path) == size_before                                  # a refused operation writes nothing
    assert r.verify_chain() == 5 and verify_log(path) == 5
    again = PredictionRegistry(path, clock)
    assert [p.to_dict() for p in again.records()] == [p.to_dict() for p in r.records()]
    assert again.registered_at("p1") == r.registered_at("p1")
    with pytest.raises(AlreadyResolvedError):
        again.resolve("p1", _obs(1, clock, 1.0))
    again.resolve("p2", _obs(2, clock, 1.0))                                     # appending after reload keeps the chain
    assert verify_log(path) == 6
    lines = open(path).read().splitlines()
    assert [json.loads(l)["seq"] for l in lines] == list(range(1, 7))
    # rewrite history: flip the outcome that was recorded for p1
    forged = [l.replace('"resolved_value": 0.0', '"resolved_value": 1.0') for l in lines]
    assert forged != lines
    bad = str(tmp / "forged.jsonl")
    open(bad, "w").write("\n".join(forged) + "\n")
    with pytest.raises(TamperedRegistryError):
        PredictionRegistry(bad, clock)
    # delete a line / append garbage
    open(bad, "w").write("\n".join(lines[:2] + lines[3:]) + "\n")
    with pytest.raises(TamperedRegistryError):
        verify_log(bad)
    open(bad, "w").write("\n".join(lines) + "\n{not json\n")
    with pytest.raises(TamperedRegistryError):
        verify_log(bad)


def test_registry_score_uses_registration_time(reg):
    r, clock, _ = reg
    rng = np.random.default_rng(0)
    for i in range(300):
        p = float(rng.uniform(0.1, 0.9))
        r.register(_pred(i, clock, dist=D.Bernoulli.from_prob(p).to_record()))
        clock.advance(1.0)
        r.resolve(f"p{i}", _obs(i, clock, float(rng.uniform() < p)))
    rep = r.score(family_of=lambda rec: "fam")
    assert rep["n_scored"] == 300 and rep["refused"] == 0
    cal = rep["binary"]["overall"]["calibration"]
    assert cal["slope_lo"] < 1 < cal["slope_hi"]
    # a record doctored after the fact to look prospective is still refused on its registration time
    r._registered_at["p0"] = r.get("p0").resolved_at
    with pytest.raises(RetrospectivePredictionError):
        r.score()


# ----------------------------------------------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------------------------------------------
@pytest.fixture
def exps(tmp_path):
    clock = SimClock()
    preds = PredictionRegistry(str(tmp_path / RunFiles.PREDICTIONS), clock)
    return ExperimentRegistry(str(tmp_path / RunFiles.EXPERIMENTS), preds, clock, seed=11), preds, clock, tmp_path


def _register(er, clock, **kw):
    args = dict(question="Does variant B change the completion count?", hypothesis="B yields more completions than A",
                predictions=[_pred(100, clock, kind=OutcomeKind.REAL, dist=D.Normal(1.0, 1.0).to_record(), horizon=DAY)],
                arms=["A", "B"], assignment="randomized", intervention="serve variant", comparison="variant A",
                horizon_s=DAY)
    args.update(kw)
    return er.register(**args)


def test_experiment_lifecycle_with_randomised_assignment(exps):
    er, preds, clock, tmp = exps
    rec = _register(er, clock, experiment_id="exp_variant")             # fixed id: reproducible assignment
    assert isinstance(rec, ExperimentRecord) and rec.status == "registered" and rec.supported is None
    assert rec.predicted[0].query.experiment_id == rec.experiment_id
    assert "p100" in preds and preds.get("p100").resolved_status == OutcomeStatus.PENDING   # registered before acting
    eid = rec.experiment_id
    with pytest.raises(RegistryError):
        er.add_outcome(eid, _obs(100, clock))                                    # nothing has been done yet
    with pytest.raises(RegistryError):
        er.conclude(eid, supported=True, test={"test": "welch_t"}, alternative_explanations=[])
    # randomised assignment with recorded probabilities
    assigned = [er.assign(eid, f"unit{i}") for i in range(400)]
    assert er.get(eid).status == "running"
    assert all(a.probability == 0.5 and a.probabilities == [0.5, 0.5] for a in assigned)
    n_b = sum(a.arm == "B" for a in assigned)
    assert st.binomtest(n_b, 400, 0.5).pvalue > 0.01 and 150 < n_b < 250
    with pytest.raises(RegistryError):
        er.assign(eid, "unit0")                                                  # a unit is assigned once
    meta = er.meta(eid)
    assert len(meta["assignments"]) == 400 and meta["arm_probs"] == [0.5, 0.5]
    assert {a["probability"] for a in meta["assignments"]} == {0.5}
    er.link_decision(eid, "dec_1")
    # outcomes per arm: B is truly better by 1.0
    rng = np.random.default_rng(1)
    clock.advance(3600.0)
    for i, a in enumerate(assigned):
        val = float(rng.normal(1.0 if a.arm == "B" else 0.0, 1.0))
        er.add_outcome(eid, OutcomeObservation("q100", OutcomeStatus.OBSERVED, val, observed_at=clock.now(),
                                               dedup_key=f"u{i}"), arm=a.arm)
    er.add_outcome(eid, OutcomeObservation("q100", OutcomeStatus.OBSERVED, 99.0, observed_at=clock.now(),
                                           dedup_key="u0"), arm="A")             # duplicate report: counted once
    with pytest.raises(RegistryError):
        er.add_outcome(eid, _obs(100, clock), arm="C")
    assert len(er.get(eid).actual) == 400
    # maturity
    assert er.mature(eid) is False and er.get(eid).status == "running"
    with pytest.raises(RegistryError):
        er.conclude(eid, supported=True, test={"test": "welch_t"}, alternative_explanations=[])
    clock.advance(DAY)
    assert er.mature(eid) is True and er.get(eid).status == "matured"
    with pytest.raises(RegistryError):
        er.assign(eid, "late_unit")
    # analysis: Welch test agrees with scipy
    res = er.analyse(eid, arm_a="B", arm_b="A")
    groups = er.outcomes_by_arm(eid)
    ref = st.ttest_ind(groups["B"], groups["A"], equal_var=False)
    np.testing.assert_allclose([res["t"], res["p_value"], res["df"]], [ref.statistic, ref.pvalue, ref.df], rtol=1e-6)
    ci = ref.confidence_interval(0.95)
    np.testing.assert_allclose([res["lo"], res["hi"]], [ci.low, ci.high], rtol=1e-6)
    assert res["valid"] and res["randomized"] and res["lo"] > 0.5
    # conclusion needs the test and the alternative-explanations slot
    with pytest.raises(RegistryError):
        er.conclude(eid, supported=True, test={}, alternative_explanations=[])
    with pytest.raises(RegistryError):
        er.conclude(eid, supported=True, test=res, alternative_explanations=None)
    closed = er.conclude(eid, supported=True, test=res,
                         alternative_explanations=["time-of-day difference between arms", "novelty of variant B"],
                         measurement_uncertainty=f"diff {res['diff']:.2f} +- {res['se']:.2f}", next_action="repeat")
    assert closed.status == "closed" and closed.supported is True and len(closed.alternative_explanations) == 2
    assert closed.decision_ids == ["dec_1"] and er.meta(eid)["test"]["test"] == "welch_t"
    with pytest.raises(RegistryError):
        er.conclude(eid, supported=False, test=res, alternative_explanations=[])  # a conclusion is not rewritten
    with pytest.raises(RegistryError):
        er.add_outcome(eid, OutcomeObservation("q100", OutcomeStatus.OBSERVED, 0.0, observed_at=clock.now()))
    assert er.open() == []
    # the experiment's prediction resolves through the prediction registry and shows up in the record
    preds.resolve("p100", OutcomeObservation("q100", OutcomeStatus.OBSERVED, res["diff"], observed_at=clock.now()))
    assert er.get(eid).predicted[0].nll is not None
    # reload from disk: identical state, chain intact
    preds2 = PredictionRegistry(str(tmp / RunFiles.PREDICTIONS), clock)
    er2 = ExperimentRegistry(str(tmp / RunFiles.EXPERIMENTS), preds2, clock, seed=11)
    assert er2.get(eid).to_dict() == er.get(eid).to_dict() and er2.meta(eid) == er.meta(eid)
    assert er2.verify_chain() == er.verify_chain() > 800


def test_experiment_registration_rules(exps):
    er, preds, clock, _ = exps
    with pytest.raises(RegistryError):
        _register(er, clock, hypothesis="  ")
    with pytest.raises(RegistryError):
        _register(er, clock, question="")
    with pytest.raises(RegistryError):
        _register(er, clock, predictions=[])                                     # predictions come first
    with pytest.raises(RegistryError):
        _register(er, clock, arms=["A"])                                         # nothing to randomise over
    with pytest.raises(RegistryError):
        _register(er, clock, arms=["A", "A"])
    with pytest.raises(RegistryError):
        _register(er, clock, arm_probs=[0.9, 0.2])
    with pytest.raises(RegistryError):
        _register(er, clock, arm_probs=[1.0, 0.0])                               # every arm must keep support
    with pytest.raises(RegistryError):
        _register(er, clock, assignment="whatever")
    assert len(preds) == 0 and er.records() == []                                # failed registrations leave nothing
    rec = _register(er, clock, arm_probs=[0.8, 0.2], experiment_id="exp_fixed")
    with pytest.raises(DuplicatePredictionError):
        _register(er, clock, experiment_id="exp_fixed", hypothesis="something else")
    with pytest.raises(DuplicatePredictionError):
        _register(er, clock)                                                     # its prediction id is taken
    assert er.get("exp_fixed").hypothesis == "B yields more completions than A" and len(preds) == 1
    draws = [er.assign("exp_fixed", f"u{i}") for i in range(1000)]
    share_a = np.mean([d.arm == "A" for d in draws])
    assert abs(share_a - 0.8) < 0.04 and {d.probability for d in draws} == {0.8, 0.2}
    # the same seed reproduces the same assignment; a sequential experiment has no random assignment
    assert [d.arm for d in draws[:5]] == [d["arm"] for d in er.meta("exp_fixed")["assignments"][:5]]
    seq = er.register(question="q?", hypothesis="h", predictions=[_pred(200, clock)], assignment="sequential")
    with pytest.raises(RegistryError):
        er.assign(seq.experiment_id, "u")
    assert er.start(seq.experiment_id).status == "running"
    with pytest.raises(RegistryError):
        er.start(seq.experiment_id)
    old = OutcomeObservation("q200", OutcomeStatus.OBSERVED, 1.0, observed_at=clock.now() - 10.0)
    with pytest.raises(ResolutionBeforePredictionError):
        er.add_outcome(seq.experiment_id, old)
    with pytest.raises(UnknownRecordError):
        er.get("missing")


def test_welch_test_edge_cases():
    assert welch_test([1.0], [1.0, 2.0])["valid"] is False
    assert welch_test([1.0, 1.0], [2.0, 2.0])["valid"] is False
    res = welch_test([1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5, 10.0], level=0.9)
    ref = st.ttest_ind([1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5, 10.0], equal_var=False)
    np.testing.assert_allclose([res["t"], res["p_value"]], [ref.statistic, ref.pvalue], rtol=1e-7)
    ci = ref.confidence_interval(0.9)
    np.testing.assert_allclose([res["lo"], res["hi"]], [ci.low, ci.high], rtol=1e-7)
