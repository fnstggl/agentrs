"""RW (amendment item 1): when the learned reward is used — prospective, anytime-valid evidence from verified outcomes —
and the pooled first-order outer estimator that lets R_psi learn inside one long objective."""
import copy
import dataclasses
import math
import random

import pytest
import torch

from cam_live.reward import diagnose as dg
from cam_live.reward.assemble import STORED_KEY
from cam_live.reward.bootstrap import RewardMixer
from cam_live.reward.integrity import params_hash
from cam_live.reward.meta import MetaGradient
from cam_live.reward.validity import BettingTest, ConfidenceSequence, ProspectiveValidity, mix_grid, pair_scores


# ------------------------------------------------------------------------------------------ sequential statistics
def _bounded(rng, mean):
    """x in {-1, 1} with E[x] = mean."""
    return 1.0 if rng.random() < 0.5 * (1.0 + mean) else -1.0


@pytest.mark.parametrize("mean", [0.0, 0.4, -0.6])
def test_confidence_sequence_covers_the_mean_at_all_times(mean):
    rng = random.Random(7)
    reps, n, misses = 300, 150, 0
    for _ in range(reps):
        cs = ConfidenceSequence(alpha=0.1)
        miss = False
        for _ in range(n):
            cs.update(_bounded(rng, mean))
            lo, hi = cs.bounds()
            miss |= not (lo <= mean <= hi)
        misses += miss
    assert misses / reps <= 0.1 + 0.05                  # time-uniform: ever excluded, at most alpha (+ MC error)


def test_betting_test_is_valid_under_the_null_and_powerful_otherwise():
    rng = random.Random(3)
    false = 0
    for _ in range(400):                                  # mean 0 (boundary of the null): sup wealth < 1 / alpha
        t = BettingTest(alpha=0.05, sign=+1)
        hit = False
        for _ in range(200):
            t.update(_bounded(rng, 0.0))
            hit |= t.rejected
        false += hit
    assert false / 400 <= 0.05 + 0.02
    found = []
    for _ in range(50):                                   # mean 0.4: rejected, typically within a few dozen
        t = BettingTest(alpha=0.025, sign=+1)
        for i in range(1, 400):
            t.update(_bounded(rng, 0.4))
            if t.rejected:
                found.append(i)
                break
    assert len(found) == 50 and sorted(found)[25] < 80               # measured median 60
    down = BettingTest(alpha=0.025, sign=-1)              # the mirrored test finds a negative mean
    for _ in range(200):
        down.update(_bounded(rng, -0.4))
    assert down.rejected


def test_mix_grid_and_pair_scores():
    assert mix_grid(0.25) == [1.0, 0.75, 0.5, 0.25, 0.0] and mix_grid(0.3)[-1] == 0.0
    g = mix_grid(0.25)
    # learned ranks the pair like the verified outcome, bootstrap the other way: c_1 = -1, c_0 = +1
    c = pair_scores((1.0, 0.0), (0.0, 2.0), (0.0, 5.0), g, 1.0, 1.0)
    assert c[0] == -1 and c[-1] == 1
    assert pair_scores((1.0, 0.0), (0.0, 2.0), (3.0, 3.0), g, 1.0, 1.0) == [0] * 5      # tied outcome: no say
    # scale of either reward drops out (each is divided by its own running scale)
    assert pair_scores((10.0, 0.0), (0.0, 0.2), (0.0, 5.0), g, 10.0, 0.1) == c


# ------------------------------------------------------------------------------------------ prospective validity
def _feed(v, n, rng, f_boot, f_learned, *, seg=0, start=0):
    """n decisions of one segment: rewards noted first, verified progress resolved afterwards (one group each)."""
    rows = []
    for i in range(start, start + n):
        y = float(rng.randint(0, 20))
        v.note(f"e{i}", segment=seg, group=f"dec:{i}", r_bootstrap=f_boot(y, rng), r_learned=f_learned(y, rng))
        rows.append((f"e{i}", y))
    return [v.resolve(e, y) for e, y in rows]


def test_informative_learned_reward_is_promoted_step_by_step_and_affine_transforms_do_not_matter():
    out = []
    for scale, shift in ((1.0, 0.0), (7.0, -3.0)):
        rng = random.Random(11)
        v, mixer = ProspectiveValidity(step=0.25), RewardMixer(step=0.25)
        mixer.note_meta_update()
        mixes = []
        for t in range(60):
            _feed(v, 2, rng, lambda y, r: r.gauss(0, 1), lambda y, r: scale * (y + r.gauss(0, 2)) + shift,
                  start=2 * t)
            mixer.apply(v.decide(mixer.mix))
            mixes.append(mixer.mix)
        out.append(mixes)
        # lowered while R_psi alone agrees better than the mixture in force; never raised
        assert mixes[0] == 1.0 and mixes[-1] <= 0.5 and all(b <= a for a, b in zip(mixes, mixes[1:]))
        assert mixer.promoted and mixer.label == "mixed" and v.n_pairs == 60
        s = v.summary()
        assert s["concordance_learned"]["lower"] > 0 and s["concordance_bootstrap"]["mean"] < 0.4
    assert out[0] == out[1]                               # same decisions under R_psi -> 7 R_psi - 3


def test_uninformative_or_frozen_learned_reward_is_never_promoted():
    rng = random.Random(5)
    v, mixer = ProspectiveValidity(step=0.25), RewardMixer(step=0.25)
    for t in range(150):
        _feed(v, 2, rng, lambda y, r: 0.1 * y + r.gauss(0, 1), lambda y, r: r.gauss(0, 1), start=2 * t)
        dec = v.decide(mixer.mix)
        assert dec.action != "lower"
        mixer.apply(dec)
    assert mixer.mix == 1.0 and not mixer.promoted and mixer.label == "bootstrap"
    assert len(mixer.decisions) == 1 and mixer.decisions[0].action == "hold"      # consecutive holds collapse


def test_harmful_learned_reward_is_reverted():
    rng = random.Random(9)
    v, mixer = ProspectiveValidity(step=0.25), RewardMixer(mix=0.5, step=0.25)
    for t in range(200):
        _feed(v, 2, rng, lambda y, r: y + r.gauss(0, 2), lambda y, r: -y + r.gauss(0, 2), start=2 * t)
        mixer.apply(v.decide(mixer.mix))
    acts = [d.action for d in mixer.decisions if d.action != "hold"]
    assert mixer.mix >= 0.75 and acts and set(acts) == {"raise"}        # moved back toward the bootstrap only


def test_units_are_groups_pairs_stay_within_a_segment_and_scores_are_prospective():
    v = ProspectiveValidity(step=0.5)
    v.note("a1", segment=0, group="exp:A", r_bootstrap=0.0, r_learned=1.0)
    v.note("a2", segment=0, group="exp:A", r_bootstrap=0.0, r_learned=3.0)
    v.note("b", segment=1, group="dec:b", r_bootstrap=0.0, r_learned=0.0)
    assert v.resolve("a1", 1.0) == 0 and v.n_units == 0           # the group is not complete yet
    assert v.resolve("a2", 2.0) == 0 and v.n_units == 1           # one unit: mean reward 2, summed outcome 3
    v.note("a3", segment=0, group="exp:A", r_bootstrap=0.0, r_learned=9.0)
    assert "a3" not in v.rows                                      # a scored group never counts again
    assert v.resolve("b", 0.0) == 0 and v.n_pairs == 0            # another segment: no pair across segments
    v.note("c", segment=0, group="dec:c", r_bootstrap=0.0, r_learned=-1.0)
    assert v.resolve("c", 0.0) == 1 and v.n_pairs == 1
    rec = v.recent[-1]
    assert rec["learned"] == [2.0, -1.0] and rec["y"] == [3.0, 0.0]   # the values noted at reward time
    assert v.resolve("c", 99.0) == 0                               # resolved once
    v.note("d", segment=0, group="dec:d", r_bootstrap=0.0, r_learned=0.0)
    v.note("d2", segment=0, group="dec:d", r_bootstrap=0.0, r_learned=0.0)
    v.resolve("d", 1.0)
    assert v.drop("d2") == 0 and v.n_units == 4                    # dropping the member that never matures


def test_validity_state_round_trip_continues_identically():
    rng = random.Random(2)
    v = ProspectiveValidity(step=0.25)
    _feed(v, 41, rng, lambda y, r: r.gauss(0, 1), lambda y, r: y + r.gauss(0, 3))
    v.note("p", segment=0, group="dec:p", r_bootstrap=0.3, r_learned=0.1)
    twin = ProspectiveValidity(step=0.25)
    twin.load_state_dict(copy.deepcopy(v.state_dict()))
    assert twin.summary() == v.summary() and twin.slot == v.slot
    for x in (v, twin):
        _feed(x, 10, random.Random(4), lambda y, r: r.gauss(0, 1), lambda y, r: y, start=100)
        x.resolve("p", 2.0)
    assert twin.summary() == v.summary() and twin.decide(1.0) == v.decide(1.0)
    with pytest.raises(ValueError):
        ProspectiveValidity(step=0.5).load_state_dict(v.state_dict())


# ------------------------------------------------------------------------------------------ meta-learner wiring
def _annotate(store, recs, y_of):
    for r in recs:
        r.decision = dataclasses.replace(r.decision, diagnostics={**(r.decision.diagnostics or {}),
                                                                   STORED_KEY: {"verified_progress_delta":
                                                                                float(y_of(r.decision.chosen))}})
        store.update(r)


def _y(a):
    return 5.0 * float(a.arguments.get("effort", 0)) if a.operation == "lab.act" else 0.0


def test_meta_learner_notes_rewards_resolves_outcomes_and_moves_the_mix_by_validity_only():
    ec = dg.ExperimentConfig()
    model, store, clock, ml = dg.make_stack(0, ec, mix=1.0, mix_step=0.25)
    assert ml.mc.promotion_rule == "validity" and ml.heldout == 0.0     # prospective evidence needs no holdout
    lt, _, _ = dg.lab_lifetime(ml, store, clock, ec, seed=4242, update_every=0)
    assert len(ml.validity.rows) == len(lt.records) and ml.validity.n_units == 0
    _annotate(store, lt.records, _y)
    dec = ml.refresh_validity()
    assert ml.validity.n_units == len(lt.records) and ml.validity.n_pairs == len(lt.records) // 2
    assert dec is not None and dec.n_pairs == ml.validity.n_pairs
    assert ml.mixer.decisions and ml.mixer.decisions[-1].action == dec.action
    # rule "value": the same evidence is collected but never moves the mix; the old held-out default returns
    model2, store2, clock2, ml2 = dg.make_stack(0, ec, mix=1.0, mix_step=0.25, promotion_rule="value",
                                                heldout=None)
    assert ml2.heldout == 0.25
    lt2, _, _ = dg.lab_lifetime(ml2, store2, clock2, ec, seed=4242, update_every=0)
    _annotate(store2, lt2.records, lambda a: _y(a) * 1e3)
    assert ml2.refresh_validity() is not None and ml2.mixer.decisions == [] and ml2.mixer.mix == 1.0


def test_checkpoint_without_validity_state_loads_and_reports_it_fresh():
    ec = dg.ExperimentConfig()
    model, store, clock, ml = dg.make_stack(0, ec, mix=1.0, mix_step=0.25)
    lt, _, _ = dg.lab_lifetime(ml, store, clock, ec, seed=77, update_every=8)
    sd = ml.state_dict()
    assert "validity" in sd and "pooled_done" in sd
    old = {k: v for k, v in sd.items() if k not in ("validity", "validity_registered", "pooled_done")}
    model2, store2, clock2, ml2 = dg.make_stack(0, ec, mix=1.0, mix_step=0.25)
    ml2.store = store
    ml2.load_state_dict(old)
    assert ml2.restored_fresh == ["validity (not in the checkpoint)"] and ml2.validity.n_units == 0
    ml3 = dg.make_stack(0, ec, mix=1.0, mix_step=0.25)[3]
    ml3.load_state_dict(copy.deepcopy(sd))
    assert ml3.restored_fresh == [] and ml3.validity.summary() == ml.validity.summary()


# ------------------------------------------------------------------------------------------ pooled estimator
def _pooled_stack(seed=0, min_independent=3, estimator="pooled_first_order", **kw):
    auth = dg.LabMeasurementAuth()
    ec = dg.ExperimentConfig(min_independent=min_independent)
    model, store, clock, ml = dg.make_stack(seed, ec, mix=1.0, check_measurement=auth, estimator=estimator, **kw)
    ec.lab.update(horizon_s=10 * 86400.0, budget=300.0)
    lt, world, ups = dg.lab_lifetime(ml, store, clock, ec, seed=11, n_decisions=18, update_every=6, auth=auth)
    return ec, model, store, clock, ml, lt


def test_pooled_first_order_meta_update_moves_only_psi_along_the_first_order_outer_gradient(monkeypatch):
    ec, model, store, clock, ml, lt = _pooled_stack()
    ev = ml.events[-1]
    rows, U, groups, st = ml._pool(ev)
    sup = set(ev.exp_ids)
    assert rows and not (sup & {r.exp_id for r in rows}) and len(set(groups)) >= 3
    captured = {}
    real_apply = MetaGradient.apply

    def spy(opt, reward, result, clip=None):
        captured["grads"] = {k: v.detach().clone() for k, v in result.grads.items()}
        return real_apply(opt, reward, result, clip=clip)
    monkeypatch.setattr(MetaGradient, "apply", staticmethod(spy))
    psi0 = {k: v.detach().clone() for k, v in ml.reward.psi().items()}
    theta = params_hash(ml.subset.live())
    h0 = params_hash(ml.reward)
    rec = ml.meta_update()
    assert rec.applied and rec.estimator == "pooled_first_order_importance" and rec.meta_grad_norm > 0
    assert params_hash(ml.reward) != h0 and params_hash(ml.subset.live()) == theta     # only psi moved
    assert rec.n_query == len(rows) and rec.support_window == f"event:{ev.event_id}"
    again = ml.meta_update()                                       # each event is stepped once
    assert not again.applied and again.skipped_reason == "no_new_policy_update_event"
    # the applied gradient is the derivative of the first-order surrogate <g, theta_K(psi)> (central differences)
    g = captured["grads"]
    ibq = ml.inner.build_batch(store, rows, policy_only=True)
    ibq = ibq.select([ibq.exp_ids.index(r.exp_id) for r in rows])
    th = {n: v.detach().clone().requires_grad_(True) for n, v in ev.theta0.items()}
    lp = ml.inner.logp(th, ibq)
    rho = torch.exp(lp.detach().double() - ibq.behaviour_logp.double()).clamp(max=ml.mc.weight_clip)
    w = (rho / rho.sum()).float()
    Ut = torch.tensor(U)
    n = len(U)
    base = torch.tensor([(sum(U) - sum(u for u, g_ in zip(U, groups) if g_ == gg)) /
                         max(1, n - groups.count(gg)) for gg in groups])
    gv = torch.autograd.grad((w * lp * (Ut - base)).sum(), list(th.values()), allow_unused=True)
    gvec = {k: (g_ if g_ is not None else torch.zeros_like(th[k])) for k, g_ in zip(th, gv)}
    steps, _ = ml._steps_of([ev])

    def S(psi):
        with ml.reward.meta_window():
            tk = ml.engine.inner_update(ev.theta0, psi, steps, ev.inner_lr, len(steps), create_graph=False).detached()
        return float(sum((gvec[k] * tk[k]).sum() for k in gvec))
    gen = torch.Generator().manual_seed(0)
    d = {k: torch.randn(v.shape, generator=gen) for k, v in psi0.items()}
    nd = math.sqrt(sum(float((x ** 2).sum()) for x in d.values()))
    d = {k: x / nd for k, x in d.items()}
    eps = 2e-3
    fd = (S({k: psi0[k] + eps * d[k] for k in d}) - S({k: psi0[k] - eps * d[k] for k in d})) / (2 * eps)
    an = sum(float((g[k] * d[k]).sum()) for k in d)
    assert abs(fd - an) <= 0.05 * abs(an) + 1e-6, (fd, an)
    # ascent: a small step along the gradient raises the surrogate
    gn = math.sqrt(sum(float((x ** 2).sum()) for x in g.values()))
    up = {k: psi0[k] + 1e-3 * g[k] / gn for k in g}
    assert S(up) > S(psi0)


def test_pooled_estimator_refuses_without_enough_independent_verified_groups():
    ec, model, store, clock, ml, lt = _pooled_stack(min_independent=1000)
    h0 = params_hash(ml.reward)
    rec = ml.meta_update()
    assert not rec.applied and rec.skipped_reason.startswith("insufficient_independent_outcomes")
    assert params_hash(ml.reward) == h0 and rec.diagnostics["pool.rows"] > 0


# ------------------------------------------------------------------------------------------ level readings
def test_readings_of_the_objective_level_are_admitted_as_changes_never_twice(tmp_path):
    """A verifier reading of the objective's own level metric (e.g. a member count read after an action) is utility
    by its change since the latest earlier reading, not by the level itself (RW: with the level, every decision
    taken late in an objective was credited with all earlier progress)."""
    from cam_live.contracts import OutcomeObservation, OutcomeStatus, Trust
    from cam_live.reward import integrity as I

    seen = set()

    def check(o):
        return o.to_json() in seen

    def obs(i, value, at):
        o = OutcomeObservation(query_id=f"q{i}", status=OutcomeStatus.OBSERVED, value=float(value),
                               observed_at=float(at), verified=True, source="verifier", dedup_key=f"k{i}")
        seen.add(o.to_json())
        return o
    path = str(tmp_path / "ledger.jsonl")
    led = I.OutcomeLedger(lambda v: False, check, path=path)
    led.register_objective("obj", origin=Trust.OPERATOR)
    key = "obj|members"
    got = [led.admit_measurement(o, objective_id="obj", level_key=key, exp_ids=[f"e{i}"]).entry.utility
           for i, o in enumerate([obs(0, 5, 10), obs(1, 7, 20), obs(2, 7, 30), obs(3, 6, 25), obs(4, 11, 40)])]
    assert got == [0.0, 2.0, 0.0, 0.0, 4.0]          # anchor, +2, flat, late reading inside a credited interval, +4
    assert sum(got) == 11 - 5 and led.verify_chain() == (True, -1)
    m = obs(5, 3, 50)
    assert led.admit_measurement(m, objective_id="obj", utility=3.0, level_key=key).reason == "utility_mismatch"
    re = I.OutcomeLedger(lambda v: False, check, path=path)             # the level state is rebuilt from the file
    assert re.head == led.head and re.derived_level_change(obs(6, 12, 60), key) == 1.0
    assert led.admit_measurement(obs(7, 9, 70), objective_id="obj").entry.utility == 9.0   # other metrics: value


def test_meta_learner_admits_goal_metric_readings_as_level_changes():
    from cam_live.contracts import OutcomeObservation, OutcomeQuery, OutcomeKind, OutcomeStatus
    ec = dg.ExperimentConfig()
    model, store, clock, ml = dg.make_stack(0, ec, check_measurement=lambda o: True)
    lt, _, _ = dg.lab_lifetime(ml, store, clock, ec, seed=4242, update_every=0)
    goal = lt.task.success["metric"] if hasattr(lt, "task") else "lab.adopters.total"
    assert ml._goal_metric.get(lt.records[0].objective_id) == goal
    recs = lt.records[:3]
    for k, (r, level) in enumerate(zip(recs, (4.0, 9.0, 9.0))):
        q = OutcomeQuery(query_id=f"gq{k}", metric=goal, kind=OutcomeKind.COUNT, horizon_s=1.0,
                         issued_at=r.decision.ts, goal_relevant=True)
        r.pending_queries = list(r.pending_queries) + [q]
        r.outcomes = list(r.outcomes) + [OutcomeObservation(query_id=q.query_id, status=OutcomeStatus.OBSERVED,
                                                            value=level, observed_at=r.decision.ts + 100.0 + k,
                                                            verified=True, source="verifier", dedup_key=f"gk{k}")]
        store.update(r)
    ml.admit_measurements(recs)
    util = [[e.utility for e in ml.ledger.by_exp(r.exp_id) if e.kind == "measurement" and e.note] for r in recs]
    assert util == [[0.0], [5.0], [0.0]]


def test_a_decision_whose_verified_progress_was_known_when_it_was_rewarded_is_not_a_prospective_unit():
    ec = dg.ExperimentConfig()
    model, store, clock, ml = dg.make_stack(0, ec, mix=1.0, mix_step=0.25)
    lt, _, _ = dg.lab_lifetime(ml, store, clock, ec, seed=4242, n_decisions=4, update_every=0)
    n0 = len(ml.validity.rows)
    from cam_live.diagnostics.collect import LabCollector
    from cam_live.diagnostics.world import LabConfig, LabWorld, lab_task
    world = LabWorld(LabConfig(seed=5, **ec.lab), clock=clock)
    act = dg.model_policy(ml, store, torch.Generator().manual_seed(1))

    def known(rec):                                     # the outcome is in before the reward step
        _annotate(store, [rec], lambda a: 3.0)
        ml.internal_rewards([store.get(rec.exp_id)])
    LabCollector(store, max_context=ec.max_context).run(world, act, 3, seed=5, on_step=known,
                                                         task=lab_task(world, objective_id="obj_known"))
    assert len(ml.validity.rows) == n0 and ml.validity.n_units == 0


def test_auto_estimator_falls_back_to_the_query_windows_with_one_record():
    ec, model, store, clock, ml, lt = _pooled_stack(estimator="auto", pool_size=0)
    n0 = len(ml.records)
    rec = ml.meta_update()
    assert len(ml.records) == n0 + 1 and ml.records[-1] is rec
    assert rec.applied and rec.estimator != "pooled_first_order_importance"
    assert rec.diagnostics["auto_pooled_skipped"] == 1.0 and rec.skipped_reason == ""
    again = ml.meta_update()                                       # neither has evidence now: one skip says both
    assert not again.applied and "| pooled: " in again.skipped_reason


def test_the_agent_configuration_uses_the_rw_defaults():
    from cam_live.config import CamLiveConfig
    from cam_live.reward.metalearner import MetaConfig
    mc = MetaConfig.from_cfg(CamLiveConfig.load())
    assert (mc.promotion_rule, mc.estimator, mc.utility, mc.heldout_fraction) == ("validity", "auto", "direct", None)
