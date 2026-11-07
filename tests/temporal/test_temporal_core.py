"""Temporal core: time encoding, hazard math, censoring, ledger immutability, durable scheduling,
reward maturation, and real weight updates of the temporal world model."""

import math
import sqlite3

import pytest
import torch

from agentrs.data.temporal_transition import (
    CountIncrement,
    EventLabel,
    StateObs,
    TemporalExample,
)
from agentrs.temporal.delayed_outcome_ledger import (
    ActionRecord,
    DelayedOutcomeLedger,
    EventRecord,
    LedgerError,
    ObservationRecord,
)
from agentrs.temporal.event_time_model import (
    EXACT,
    INTERVAL,
    RIGHT,
    event_log_likelihood,
    make_grid,
    negbin_log_prob,
    piecewise_linear_integral,
)
from agentrs.temporal.observation_scheduler import (
    AdaptivePolicy,
    ObservationWorker,
    SchedulerConfig,
    TemporalObservationScheduler,
)
from agentrs.temporal.outcome_labels import build_examples
from agentrs.temporal.reward_maturation import (
    RewardSpec,
    mature_reward,
    semi_markov_return,
)
from agentrs.temporal.temporal_world_model import (
    TemporalQuery,
    TemporalSpec,
    TemporalWorldModel,
    TemporalWorldModelConfig,
)
from agentrs.temporal.time_encoding import (
    DAY,
    HOUR,
    TimeEncoder,
    describe_duration,
    log_time,
    parse_duration,
)
from agentrs.training.common import (
    GradientTrainer,
    OptimConfig,
    load_model,
    load_trainer_state,
    param_change,
    param_snapshot,
    save_checkpoint,
)

T0 = 1_000_000.0


def test_time_encoding_and_durations():
    assert parse_duration("15m") == 900 and parse_duration("3d") == 3 * DAY and describe_duration(3600) == "1h"
    tau = log_time([0, 1, 60, HOUR, DAY, 30 * DAY])
    assert torch.all(tau[1:] > tau[:-1]) and float(tau[0]) == 0.0 and abs(float(tau[-1]) - 1.0) < 1e-6
    enc = TimeEncoder()
    a, b, c = enc(log_time([HOUR, 1.1 * HOUR, 30 * DAY]))
    assert torch.norm(a - b) < torch.norm(a - c)  # nearby durations have nearby embeddings


def test_piecewise_integral_matches_numeric_and_cdf_is_monotone():
    grid = make_grid(12)
    dt = float(grid[1] - grid[0])
    rates = torch.rand(3, 12) * 3
    q = torch.linspace(0, float(grid[-1]) * 1.2, 50).expand(3, 50).contiguous()
    Lam, lam = piecewise_linear_integral(rates, q, dt)
    fine = torch.linspace(0, float(q[0, 20]), 20001)
    r = torch.stack([torch.from_numpy(__import__("numpy").interp(fine.numpy(), grid.numpy(), rr.numpy())) for rr in rates])
    num = torch.trapezoid(r, fine, dim=1)
    assert torch.allclose(Lam[:, 20].double(), num, atol=1e-3)
    assert torch.all(Lam[:, 1:] >= Lam[:, :-1] - 1e-7)


def test_likelihood_kinds():
    pi = torch.tensor([0.0])
    Lam = torch.tensor([[0.5, 1.5]])
    lam = torch.tensor([[1.0, 2.0]])
    p = 0.5
    ex = event_log_likelihood(pi, Lam, lam, torch.tensor([EXACT]))
    assert torch.allclose(ex, torch.tensor([math.log(p * 2.0) - 1.5]), atol=1e-5)
    iv = event_log_likelihood(pi, Lam, lam, torch.tensor([INTERVAL]))
    assert torch.allclose(iv, torch.tensor([math.log(p * (math.exp(-0.5) - math.exp(-1.5)))]), atol=1e-5)
    rc = event_log_likelihood(pi, Lam, lam, torch.tensor([RIGHT]))
    assert torch.allclose(rc, torch.tensor([math.log(1 - p + p * math.exp(-1.5))]), atol=1e-5)
    nb = negbin_log_prob(torch.tensor([3.0]), torch.tensor([2.0]), torch.tensor([1e4]))
    assert abs(float(nb) - (3 * math.log(2) - 2 - math.log(6))) < 1e-2  # ~Poisson for large r


def _model(scratch_bb, spec=None):
    torch.manual_seed(0)
    return TemporalWorldModel(TemporalWorldModelConfig(backbone=scratch_bb, n_heads=3, spec=spec or TemporalSpec()))


def _train(m, data, steps=80, lr=2e-3):
    tr = GradientTrainer(m, OptimConfig(lr=lr))
    for _ in range(steps):
        loss, _ = m.temporal_loss(data)
        tr.step(loss)
    m.eval()
    return tr


def test_pending_outcomes_are_not_trained_as_failures(scratch_bb):
    """Half the examples of one action saw the reply at 6 h; the other half are still pending at
    1 h. The model must keep P(reply by 1 day) high (not drag it towards 'never')."""
    act = 'submit(create_post, "How do I fix pip when venv hangs? error E1234")'
    data = []
    for i in range(16):
        lab = (EventLabel(kind="exact", a_s=6 * HOUR, b_s=6 * HOUR, status="occurred") if i % 2 else
               EventLabel(kind="right_censored", a_s=HOUR, b_s=HOUR, status="pending"))
        data.append(TemporalExample(example_id=f"x{i}", action_id=f"a{i}", initial_state="space=python-help",
                                    complete_action=act, events={"first_response": lab}))
    m = _model(scratch_bb)
    _train(m, data, steps=120)
    p = m.predict(TemporalQuery(context="space=python-help", action=act), horizons=[HOUR, DAY])
    assert p.horizons[0].event_prob["first_response"] < 0.3
    assert p.horizons[1].event_prob["first_response"] > 0.75  # a pending=failure model would give ~0.5
    assert p.event_ever["first_response"] > 0.75


def test_real_weight_updates_and_checkpoint_reload(scratch_bb, tmp_path):
    m = _model(scratch_bb)
    data = [TemporalExample(example_id=f"e{i}", action_id=f"a{i}", initial_state="space=linux",
                            complete_action=f'submit(create_post, "post {i}")',
                            events={"answered": EventLabel(kind="exact", a_s=600 * (i + 1), b_s=600 * (i + 1))},
                            counts={"responses": [CountIncrement(a_s=0, b_s=3600, delta=i % 3)]},
                            states=[StateObs(h_s=3600, state="visible")]) for i in range(8)]
    for n, p in m.named_parameters():
        p.grad = None
    loss, _ = m.temporal_loss(data)
    assert torch.isfinite(loss)
    loss.backward()
    g_heads = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith("t_") and p.grad is not None)
    g_trunk = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith("backbone.") and p.grad is not None)
    assert g_heads > 0 and g_trunk > 0
    snap = param_snapshot(m)
    tr = GradientTrainer(m, OptimConfig(lr=1e-3))
    for _ in range(3):
        loss, _ = m.temporal_loss(data)
        tr.step(loss)
    assert param_change(m, snap) > 0
    st = tr.optimizer.state_dict()["state"]
    assert st and all(torch.isfinite(v["exp_avg"]).all() for v in st.values())
    q = TemporalQuery(context="space=linux", action=data[0].complete_action)
    before = m.predict(q, horizons=[60, 3600, DAY]).to_dict()
    save_checkpoint(tmp_path / "ck", m, tr)
    m2 = load_model(tmp_path / "ck")
    assert m2.component == "temporal_world_model"
    after = m2.predict(q, horizons=[60, 3600, DAY]).to_dict()
    for a, b in zip(before["horizons"], after["horizons"]):
        for k in a["event_prob"]:
            assert abs(a["event_prob"][k] - b["event_prob"][k]) < 1e-5
    tr2 = GradientTrainer(m2, OptimConfig(lr=1e-3))
    load_trainer_state(tmp_path / "ck", tr2, restore_rng=False)
    assert tr2.global_step == 3


def test_ledger_forecasts_are_immutable_and_governed(tmp_path):
    L = DelayedOutcomeLedger(tmp_path / "l.sqlite")
    a = L.record_action(ActionRecord(source_id="s", source_type="third_party", actor_type="simulated_user",
                                     time_base="virtual:x", action_time=T0, observed_time=T0, action_str="x()"))
    assert L.record_action(ActionRecord(action_id=a, source_id="s", source_type="third_party", actor_type="x",
                                        time_base="virtual:x", action_time=T0, observed_time=T0, action_str="y()")) == a
    fid = L.record_forecast(a, {"p": 0.3}, "ck1", info_cutoff_time=T0, created_time=T0, horizons=[60])
    with pytest.raises(sqlite3.DatabaseError):
        L._x("UPDATE forecasts SET prediction_json='{}' WHERE forecast_id=?", (fid,))
    with pytest.raises(sqlite3.DatabaseError):
        L._x("DELETE FROM forecasts WHERE forecast_id=?", (fid,))
    with pytest.raises(LedgerError):  # cannot use information from after creation
        L.record_forecast(a, {}, "ck", info_cutoff_time=T0 + 100, created_time=T0 + 50, horizons=[60])
    L.record_forecast(a, {"p": 0.9}, "ck2", info_cutoff_time=T0 + 600, created_time=T0 + 600, horizons=[60], kind="updated")
    fs = L.forecasts(a)
    assert [f["prediction"]["p"] for f in fs] == [0.3, 0.9]  # the original survives the update
    o1 = L.record_observation(ObservationRecord(action_id=a, actual_time=T0 + 60, payload={}, idem="k"))
    o2 = L.record_observation(ObservationRecord(action_id=a, actual_time=T0 + 60, payload={}, idem="k"))
    assert o1 == o2 and L.stats()["observations"] == 1
    with pytest.raises(LedgerError):
        L.record_observation(ObservationRecord(action_id=a, actual_time=T0 + 70, payload={}, time_base="real"))
    assert L.record_event(EventRecord(action_id=a, type="t", observed_time=T0 + 5, source_id="s", provider_event_id="1"))
    assert not L.record_event(EventRecord(action_id=a, type="t", observed_time=T0 + 5, source_id="s", provider_event_id="1"))
    counts = L.purge_source("s", "test revocation")
    assert counts["forecasts"] == 2 and L.stats()["actions"] == 0
    assert L.governance_log()[-1]["op"] == "purge"


def _ledger_with_action(tmp_path, **kw):
    L = DelayedOutcomeLedger(tmp_path / "l.sqlite")
    a = L.record_action(ActionRecord(source_id="s", source_type="self", actor_type="agent", time_base="virtual:x",
                                     action_time=T0, observed_time=T0, action_str='submit(create_post, "q")',
                                     window_end=T0 + 7 * DAY, policy_checkpoint="pol-v3", **kw))
    return L, a


def _obs(L, a, t, occurred=None, counts=None, state="visible", at=None):
    L.record_observation(ObservationRecord(action_id=a, actual_time=T0 + t, payload={
        "occurred": occurred or {}, "occurred_at": at or {}, "counts": counts or {}, "state": state}, idem=f"o{t}"))


def test_labels_exact_interval_censored_precluded_and_no_leakage(tmp_path):
    spec = TemporalSpec(events=["first_response", "edited", "removed", "resolved"], counts=["responses"])
    L, a = _ledger_with_action(tmp_path)
    _obs(L, a, 60, {"first_response": False, "edited": False, "removed": False, "resolved": False}, {"responses": 0})
    L.record_event(EventRecord(action_id=a, type="item_edited", env_event_time=T0 + 1800, observed_time=T0 + 1900,
                               by_original_actor=True, source_id="s", provider_event_id="e1"))
    _obs(L, a, 3600, {"first_response": True, "edited": True, "removed": False, "resolved": False}, {"responses": 2})
    _obs(L, a, DAY, {"first_response": True, "edited": True, "removed": False, "resolved": False}, {"responses": 5})
    ex0, ex1 = build_examples(L, a, spec, cutoffs=[0, 60])
    ev = ex0.events
    assert ev["edited"].kind == "exact" and abs(ev["edited"].b_s - 1800) < 1e-6
    assert ev["first_response"].kind == "interval" and ev["first_response"].a_s == 60 and ev["first_response"].b_s == 3600
    assert ev["resolved"].kind == "right_censored" and ev["resolved"].status == "pending"
    assert [i.delta for i in ex0.counts["responses"]] == [0, 2, 3]  # increments, not cumulative totals
    # the cutoff-60 example knows nothing observed after 60 s
    assert ex1.history == [] and ex1.known_now["responses"] == 0 and ex1.truncation_s["first_response"] == 60
    # a competing removal precludes later outcomes
    L.record_event(EventRecord(action_id=a, type="item_removed", env_event_time=T0 + 2 * DAY, observed_time=T0 + 2 * DAY,
                               source_id="s", provider_event_id="e2"))
    ex = build_examples(L, a, spec)[0]
    assert ex.events["resolved"].status == "precluded" and ex.events["resolved"].b_s == 2 * DAY
    L.close_action(a, T0 + 7 * DAY, "window")
    _obs(L, a, 7 * DAY, {"resolved": False, "removed": True}, {"responses": 5}, state="removed")


def test_reward_maturation_increments_and_semi_markov(tmp_path):
    L, a = _ledger_with_action(tmp_path)
    spec = RewardSpec(kind="event_by_horizon", objective="resolved", horizon_s=DAY)
    _obs(L, a, 3600, {"resolved": False})
    assert mature_reward(L, a, spec, as_of=T0 + 3600).status == "pending"  # not a 0
    _obs(L, a, DAY + 10, {"resolved": False})
    r = mature_reward(L, a, spec, as_of=T0 + 2 * DAY)
    assert r.status == "matured" and r.value == 0.0 and r.checkpoint == "pol-v3"
    L2, b = _ledger_with_action(tmp_path / "b")
    for t, c in ((3600, 10), (DAY, 20), (2 * DAY, 18)):
        _obs(L2, b, t, counts={"answers": c})
    s = mature_reward(L2, b, RewardSpec(kind="count_stream", objective="answers"), as_of=T0 + 3 * DAY)
    assert [d for _, d in s.increments] == [10, 10, -2] and s.value == 18  # never 10 + 20
    kinds = [r["kind"] for r in L2.rewards(b)]
    assert kinds.count("revision") == 1
    assert abs(semi_markov_return([(T0 + DAY, 1.0)], T0, 0.5, DAY) - 0.5) < 1e-9


def test_scheduler_durable_idempotent_retries_budgets_adaptive(tmp_path):
    L, a = _ledger_with_action(tmp_path)
    L._x("UPDATE actions SET context_json='{\"object_ref\": \"7\"}'")
    p = tmp_path / "jobs.sqlite"
    S = TemporalObservationScheduler(p, SchedulerConfig(max_attempts=3, backoff_base_s=10, lease_s=0.0))
    ids = S.schedule_horizons(a, "src", T0, [60, 3600], "virtual:x")
    assert len(ids) == 2 and S.schedule_horizons(a, "src", T0, [60, 3600], "virtual:x") == []
    S.close()
    S = TemporalObservationScheduler(p, SchedulerConfig(max_attempts=3, backoff_base_s=10, lease_s=60))  # restart
    assert len(S.jobs(a)) == 2
    # a leased job of a crashed worker is recovered
    j = S.claim_due({"virtual:x": T0 + 100}, "w1")
    assert len(j) == 1
    S._x("UPDATE jobs SET lease_until=0")
    assert S.recover_expired_leases() == 1
    calls = {"n": 0}

    def flaky(job):
        calls["n"] += 1
        raise TimeoutError("source timed out")

    W = ObservationWorker(S, L, {"src": flaky})
    W.run_due({"virtual:x": T0 + 100})
    assert W.stats["retry"] == 1
    W.run_due({"virtual:x": T0 + 200})
    assert W.stats["failed"] == 1 and W.stats["missing"] == 1
    assert L.observations(a)[0]["completeness"] == "missing"
    # budgets: one token per observation in the source clock
    S.set_budget("src", capacity=1, refill_per_s=1 / 600, now=T0)
    S.schedule_horizons(a, "src", T0, [5, 6, 7], "virtual:x")
    W2 = ObservationWorker(S, L, {"src": lambda job: ({"counts": {}}, "complete")})
    assert W2.run_due({"virtual:x": T0 + 10}) == 1
    assert W2.run_due({"virtual:x": T0 + 20}) == 0
    assert W2.run_due({"virtual:x": T0 + 700}) == 1
    assert S.on_event(a, "src", T0 + 1000, "virtual:x") and not S.on_event(a, "src", T0 + 1010, "virtual:x")
    assert S.cancel(a, "deleted") >= 1
    plan = AdaptivePolicy().plan([60, 3600, DAY], {"horizons": [
        {"horizon_s": 60, "event_prob": {"x": 0.1}, "event_std": {"x": 0.01}, "count_mean": {"c": 0}},
        {"horizon_s": 3600, "event_prob": {"x": 0.11}, "event_std": {"x": 0.01}, "count_mean": {"c": 0.1}},
        {"horizon_s": DAY, "event_prob": {"x": 0.6}, "event_std": {"x": 0.2}, "count_mean": {"c": 3}}],
        "median_time_s": {"x": 7200}, "event_ever": {"x": 0.6}})
    assert plan[0] == [60, DAY] and plan[1] == [7200]
