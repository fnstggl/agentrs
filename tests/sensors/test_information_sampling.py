"""Data selection: stratification keeps every stratum, novelty avoids near-duplicates, and
learnable-error scoring does not chase inherently random outcomes."""

from agentrs.data.temporal_transition import EventLabel, TemporalExample
from agentrs.sensors.information_sampling import coverage, select


def _ex(i, group, text):
    return TemporalExample(example_id=f"e{i}", action_id=f"a{i}", initial_state=f"space={group}",
                           complete_action=f'submit(create_post, "{text}")', group=group,
                           events={"answered": EventLabel(kind="exact", a_s=60, b_s=60)})


def test_strategies():
    pool = [_ex(i, "big", "same repetitive post") for i in range(90)]
    pool += [_ex(100 + i, f"g{i % 5}", f"distinct topic {i} about thing{i} and widget{i * 7}") for i in range(10)]
    strat = select(pool, 10, "stratified", seed=0)
    assert len({pool[i].group for i in strat}) == 6  # every stratum represented
    uni = select(pool, 10, "uniform", seed=0)
    assert sum(pool[i].group == "big" for i in uni) >= 6  # volume baseline mirrors the dominant source
    nov = select(pool, 10, "novelty", seed=0)
    assert sum(pool[i].group == "big" for i in nov) <= 2  # near-duplicates are not chosen twice
    assert coverage(pool)["big"] == 90


def test_learnable_error_ignores_irreducible_noise():
    from agentrs.temporal.temporal_world_model import HorizonPrediction, TemporalPrediction

    def pred(p):
        hp = HorizonPrediction(horizon_s=3600, event_prob={"answered": p}, event_std={"answered": 0.0},
                               count_mean={}, count_std={}, state_probs={}, continuous={})
        return TemporalPrediction(horizons=[hp], event_ever={}, median_time_s={}, checkpoint="x", cutoff_elapsed_s=0,
                                  source_type="third_party")
    # coin-flip outcome predicted at 0.5 (error is irreducible) vs a confidently wrong prediction
    exs = [_ex(0, "a", "coin"), _ex(1, "b", "learnable")]
    picked = select(exs, 1, "learnable_error", predictions={"preds": [pred(0.5), pred(0.05)], "horizons": [3600]})
    assert 1 in picked
