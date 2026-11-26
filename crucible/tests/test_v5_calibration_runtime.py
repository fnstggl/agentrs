"""The runtime decision-model API (crucible.calibration.runtime): feature contracts in simulator terms, row-wise
transforms (no batch-relative mapping), declared support, provenance, the contenders the E3 review keeps, and that
every registered fitted model is served by a decision model. Held-out evaluation of every kept checkpoint through
this API is done by the Gate F reproduction (tests/v5/test_gate_F.py), which uses the same code path."""
import numpy as np
import pytest

from crucible.calibration import list_models, registry as R, runtime as RT
from crucible.calibration.runtime import ContractError


def _inputs(dm, n=64, seed=0):
    """Inputs inside every declared domain, varied per row."""
    rng = np.random.default_rng(seed)
    d = {}
    for i in dm.spec.inputs:
        lo = 0.0 if i.lo is None else float(i.lo)
        hi = (lo + 30.0) if i.hi is None else float(i.hi)
        v = rng.uniform(lo, hi, n)
        if i.integer or i.unit in ("flag",):
            v = np.floor(rng.uniform(lo, hi + 1, n)).clip(lo, hi)
        if i.unit == "flag":
            v = rng.integers(0, 2, n).astype(float)
        if i.name in ("price", "reference_price"):
            v = rng.uniform(1.0, 5.0, n)
        if i.name == "own_creator_views_today":
            v = rng.integers(1, 4, n).astype(float)
        if i.name == "topic_population_share":
            v = rng.uniform(0.01, 0.3, n)
        d[i.name] = v
    return d


READY = sorted(k for k, s in RT.specs().items() if s["status"] == "ready")


def test_every_registered_model_has_a_decision_model_and_a_contract():
    served = {s["model"] for s in RT.specs().values()}
    assert set(list_models()) <= served
    for sid, s in RT.specs().items():
        assert s["fidelity_id"] == f"runtime:{sid}" and s["inputs"] and len(s["limits"]) > 80
        for i in s["inputs"]:
            assert i["meaning"] and i["unit"] and i["kind"] in ("individual", "item", "pair", "context", "slate",
                                                                   "latent")


@pytest.mark.parametrize("sid", READY)
def test_outputs_are_row_wise_never_batch_relative(sid):
    dm = RT.get(sid)
    d = _inputs(dm)
    full = dm.evaluate(d).values
    for v in (0, 5, 63):                                   # one row alone gives the same output as in the batch
        one = dm.evaluate({k: a[v:v + 1] for k, a in d.items()}).values
        np.testing.assert_allclose(one[0], full[v], rtol=0, atol=1e-12)
    perm = np.random.default_rng(1).permutation(64)       # and the order or company of the other rows is irrelevant
    np.testing.assert_allclose(dm.evaluate({k: a[perm] for k, a in d.items()}).values, full[perm], rtol=0, atol=1e-12)
    assert np.all(np.isfinite(full))


@pytest.mark.parametrize("sid", READY)
def test_contract_violations_raise(sid):
    dm = RT.get(sid)
    d = _inputs(dm, n=4)
    with pytest.raises(ContractError):
        dm.evaluate({k: v for k, v in list(d.items())[1:]})              # a missing input
    bounded = [i for i in dm.spec.inputs if i.lo is not None]
    if bounded:
        bad = dict(d)
        bad[bounded[0].name] = np.full(4, bounded[0].lo - 1.0)
        with pytest.raises(ContractError):
            dm.evaluate(bad)                                               # outside the hard domain
    with pytest.raises(ContractError):
        dm.evaluate({k: np.append(v, v[:1]) if j == 0 else v for j, (k, v) in enumerate(d.items())})


def test_provenance_and_sampling_are_deterministic_given_the_callers_rng():
    for sid in READY:
        dm = RT.get(sid)
        d = _inputs(dm, n=32)
        res = dm.evaluate(d)
        prov = res.provenance()
        assert prov["decision"] == sid and prov["model"] == dm.model_id and prov["fidelity"] == f"runtime:{sid}"
        assert prov["served"] == dm.served and prov["rows"] == 32
        a = dm.sample(d, np.random.default_rng(7))
        b = dm.sample(d, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)


def test_kuairand_contract_reproduces_the_fitted_features():
    """Raw simulator counts -> the model's features with the fitting code's formulas (kuairand.choice_frame)."""
    dm = RT.get("content_open@1")
    prior = float(dm.fitted.card["prior_rate"])
    d = {"own_views_7d": np.array([3.0]), "own_opens_before": np.array([10.0]), "own_impressions_before": np.array([40.0]),
         "item_views_7d": np.array([50.0]), "active_individuals_7d": np.array([2000.0]),
         "item_opens_before": np.array([30.0]), "item_impressions_before": np.array([60.0]),
         "item_age_days": np.array([2.0]), "item_duration_s": np.array([30.0]), "own_topic_views": np.array([4.0]),
         "own_views_before": np.array([10.0]), "topic_population_share": np.array([0.1]),
         "own_creator_views": np.array([1.0]), "own_item_exposures": np.array([0.0]),
         "own_item_consumed": np.array([0.0]), "local_hour": np.array([13.5])}
    X = dm.features(d)[0]
    f = dict(zip(dm.fitted.features, X))
    assert f["user_log_plays_7d"] == pytest.approx(np.log1p(3))
    assert f["user_click_rate"] == pytest.approx((10 + 20 * prior) / (40 + 20))
    assert f["item_log_pop_7d"] == pytest.approx(np.log1p(1000 * 50 / 2000))
    assert f["user_tag_share"] == pytest.approx((4 + 5 * 0.1) / (10 + 5))
    assert f["user_tag_lift"] == pytest.approx(np.log(((4 + 5 * 0.1) / 15) / 0.1))
    assert f["hour_sin"] == pytest.approx(np.sin(2 * np.pi * 13 / 24))
    # absolute effect: a uniformly more interested population opens more (no rank mapping that would cancel it)
    base = _inputs(dm, n=200, seed=3)
    more = dict(base, own_topic_views=base["own_topic_views"] + 5.0)
    assert dm.evaluate(more).values.mean() > dm.evaluate(base).values.mean()


def test_offline_only_and_review_kept_contenders():
    dm = RT.get("article_open@1")                       # news_click@1: LSA cosine, not computable in a world
    with pytest.raises(ContractError):
        dm.evaluate({"text_cos_lsa": np.zeros(2)})
    # the contenders the E3 review keeps are what the runtime serves; the checkpoint stays available
    for mid in ("session_rate@1", "commercial_response@1"):
        rv = RT.review_of(mid)
        assert rv is not None
        dm = RT.decision_for(mid)
        assert dm.served == rv["kept_after_review"]
    es = RT.decision_for("session_rate@1")
    if es.served == "S":
        v = es.evaluate({"extra_sessions_observed": np.array([0.0, 9.0]), "active_days_observed": np.array([0.0, 3.0])})
        assert v.values[0] == v.values[1] == pytest.approx(es.spec.params["served_rate"])
        ck = es.evaluate_features(np.array([[0.0, 0.0], [9.0, 3.0]]), checkpoint=True)
        assert ck.served == "K" and ck.values[1] > ck.values[0]
    ad = RT.decision_for("commercial_response@1")       # ad_response@1: superseded by ad_response@2 (E5)
    assert ad.served == RT.review_of("commercial_response@1")["kept_after_review"]
    assert ad.spec.status == "offline_only" and "ad_response@2" in ad.spec.status_reason
    with pytest.raises(ContractError):
        ad.evaluate({"response_type": np.array([0, 63]), "treated": np.array([1, 1])})


def test_unknown_decision_and_missing_checkpoint_fail_loudly(tmp_path, monkeypatch):
    with pytest.raises(ContractError):
        RT.get("no_such_decision@1")
    monkeypatch.setattr(R, "MODELS_DIR", str(tmp_path))
    with pytest.raises(R.CalibrationError):
        R.load("content_consumption@1")
