"""Decisions on the real (debug-scale) model: exact probabilities for every rule, the randomised-experiment path,
dependence of the action distribution on CAM's belief latent, and latent planning with the transition head."""
import dataclasses
import math
import random

import pytest
import torch

from cam_live.contracts import Decision, OutcomeObservation, OutcomeStatus
from cam_live.learning_api import encode_decisions
from cam_live.nn.api import GROUP_ADAPTERS, GROUP_BACKBONE_FROZEN, GROUP_EXPERIENTIAL, GROUP_HEADS
from cam_live.predict import policy as P
from cam_live.predict.imagination import ImaginedDataError, is_imagined
from cam_live.predict.planner import ImaginedEvaluation, LatentPlanner
from cam_live.predict.registry import RegistryError
from cam_live.predict.world import build_targets

DAY = 86400.0


def test_every_rule_records_the_exact_distribution_it_sampled_from(lab_rig):
    rig = lab_rig(n_lifetimes=0)
    dm = P.DecisionMaker(rig.model, heads=rig.heads, clock=rig.clock, seed=1)
    cands = rig.candidates()
    bad = dataclasses.replace(cands[0], action_id="bad", arguments={"channel": 99, "effort": 1})
    evs = rig.evaluate(candidates=cands + [bad])
    assert not evs[-1].feasible and "channel" in evs[-1].infeasible_reason
    n = len(evs)
    for rule in P.RULES[:-1]:
        dec = dm.select(evs, rule, objective_id=rig.task.objective_id, model_version="m1")
        assert isinstance(dec, Decision) and dec.exploration == rule and dec.model_version == "m1"
        p = dec.probabilities
        assert len(p) == n and abs(sum(p) - 1.0) < 1e-12 and p[-1] == 0.0          # infeasible: exactly zero
        assert min(p[:-1]) >= rig.heads.cfg.eps_floor / (n - 1) - 1e-12            # every feasible one stays possible
        assert dec.chosen is evs[dec.chosen_index].action and dec.propensity == p[dec.chosen_index] > 0
        assert dec.candidates == list(evs) and dec.ts == rig.clock.now()
        assert dec.decision_id == evs[1].predictions[0].decision_id or not evs[1].predictions
        d = dec.diagnostics
        assert d["rule"] == rule and d["n_feasible"] == n - 1 and d["b_norm"] > 0 and d["z_norm_chosen"] > 0
        assert len(d["top"]) == 3 and d["top"][0]["value"] >= d["top"][1]["value"] and d["value_source"] == "q_goal"
        assert d["thompson_sampled_from"] == "posterior"
        assert f"probability {p[dec.chosen_index]:.3f}" in dec.rationale and rule in dec.rationale
        assert f"candidate {dec.chosen_index} ({dec.chosen.operation})" in dec.rationale
    # the recorded vector is the sampling distribution: empirical frequencies agree with it
    probs, _ = dm.probabilities(evs, P.RULE_INFO_GAIN)
    g = torch.Generator().manual_seed(0)
    draws = torch.tensor([dm.select(evs, P.RULE_INFO_GAIN, generator=g).chosen_index for _ in range(1500)])
    freq = torch.bincount(draws, minlength=n).double() / 1500
    assert float((freq - probs).abs().max()) < 4 * math.sqrt(0.25 / 1500)
    # greedy / ucb put the non-floor mass on the argmax of the recorded value (+ c * epistemic sd)
    sc = dm.scores(evs)
    q = torch.where(sc["a_mask"][0], sc["q"][0], torch.full_like(sc["q"][0], -1e9))
    assert int(dm.probabilities(evs, P.RULE_GREEDY)[0].argmax()) == int(q.argmax())
    assert int(dm.probabilities(evs, P.RULE_UCB)[0].argmax()) == int((q + rig.heads.cfg.ucb_c * sc["sd"][0]).argmax())
    with pytest.raises(ValueError):
        dm.select(evs, P.RULE_RANDOMIZED)
    with pytest.raises(ValueError):
        dm.select([dataclasses.replace(e, feasible=False) for e in evs], P.RULE_GREEDY)


def test_randomised_experiment_is_registered_before_assignment(lab_rig):
    rig = lab_rig(n_lifetimes=0)
    dm = P.DecisionMaker(rig.model, heads=rig.heads, clock=rig.clock, experiments=rig.experiments)
    evs = rig.evaluate()
    prop = dm.propose_experiment(evs)                                 # the learner chooses what to compare
    assert prop is not None and len(prop["arms"]) == 2 and 0.0 <= prop["expected_flip_probability"] <= 0.5
    i, j = prop["arms"].values()
    assert evs[i].action.operation == evs[j].action.operation == "lab.act"
    n_before = len(rig.registry)
    dec, exp = dm.select_experiment(evs, objective_id=rig.task.objective_id, arm_probs=[0.25, 0.75], **prop)
    assert exp.status == "running" and exp.assignment == "randomized" and exp.arms == list(prop["arms"])
    assert exp.question == prop["question"] and exp.hypothesis == prop["hypothesis"] and exp.horizon_s == 2 * DAY
    assert len(exp.predicted) == 4 and len(rig.registry) == n_before + 4          # the arms' predictions, registered
    assert all(p.query.experiment_id == exp.experiment_id and p.made_at <= exp.created_at for p in exp.predicted)
    meta = rig.experiments.meta(exp.experiment_id)
    assert meta["arm_probs"] == [0.25, 0.75] and len(meta["assignments"]) == 1
    assert dec.exploration == "randomized" and dec.experiment_id == exp.experiment_id == dec.chosen.experiment_id
    want = [0.0] * len(evs)
    want[i], want[j] = 0.25, 0.75
    assert dec.probabilities == want and dec.chosen_index in (i, j)
    assert dec.propensity == meta["assignments"][0]["probability"] and dec.decision_id in exp.decision_ids
    assert dec.diagnostics["arm"] == exp.arms[[i, j].index(dec.chosen_index)]

    # further units of the same experiment: new decisions, same arms; assignment frequencies follow arm_probs
    picks = []
    for k in range(60):
        evs_k = rig.evaluate(compute_voi=False)
        arms = {name: next(n for n, e in enumerate(evs_k) if e.action.arguments == evs[idx].action.arguments
                           and e.action.operation == "lab.act") for name, idx in prop["arms"].items()}
        d_k, _ = dm.select_experiment(evs_k, arms, experiment_id=exp.experiment_id)
        picks.append(list(arms.values()).index(d_k.chosen_index))
    assert 0.55 < sum(picks) / len(picks) < 0.92
    assert len(rig.experiments.get(exp.experiment_id).decision_ids) == 61
    with pytest.raises(ValueError):
        dm.select_experiment(evs, {"a": i, "b": i}, question="q", hypothesis="h")
    with pytest.raises(RegistryError):                                # no prediction, no experiment
        waits = [n for n, e in enumerate(evs) if not e.predictions][:2]
        dm.select_experiment(evs, {"a": waits[0], "b": waits[1]}, question="q", hypothesis="h")

    # outcome of the assigned arm arrives later and is analysed per arm
    res = rig.execute(dec.chosen)
    rig.clock.advance(3 * DAY)
    out = rig.verifier.measure(res.expected_outcomes)[0]
    rig.experiments.add_outcome(exp.experiment_id, out, arm=dec.diagnostics["arm"])
    assert rig.experiments.mature(exp.experiment_id) is True
    assert sum(len(v) for v in rig.experiments.outcomes_by_arm(exp.experiment_id).values()) == 1


def _phase_policy(seed):
    """Behaviour that depends on the history (visible to CAM only through its context): interventions first,
    measurements later."""
    rng = random.Random(seed)
    state = {"n": 0}

    def fn(cands, info):
        want = "lab.act" if state["n"] < 6 else "lab.probe"
        state["n"] += 1
        w = [8.0 if c.operation == want else 0.25 for c in cands]
        p = [x / sum(w) for x in w]
        return rng.choices(range(len(cands)), weights=p)[0], p
    return fn


def test_action_distribution_depends_on_the_belief_latent_on_the_real_model(lab_rig, measure):
    rig = lab_rig(n_lifetimes=0)
    model, heads, L = rig.model, rig.heads, rig.L
    lts = L.collect(rig.store, range(12), 12, policy_fn=_phase_policy)
    recs = [r for lt in lts for r in lt.records]
    probe = [r for lt in lts[:6] for r in (lt.records[1], lt.records[10])]      # early and late contexts, mixed

    def dependence():
        with torch.no_grad():
            b = encode_decisions(model, rig.store, probe, grad=False)
            return P.latent_dependence(heads.policy, b.z, b.a_mask, b.b,
                                       z_fn=lambda bel: model.decision_latent(bel, b.e_a, b.e_g)), b

    def probs():
        with torch.no_grad():
            b = encode_decisions(model, rig.store, probe, grad=False)
            return heads.policy(b.z, b.a_mask, b.b).probs.double()

    init, _ = dependence()
    assert init["kl"] > 1e-7                                          # the belief already reaches the logits at init
    frozen = {g: model.group_hashes()[g] for g in (GROUP_BACKBONE_FROZEN, GROUP_ADAPTERS)}
    groups = model.param_groups()
    opt = torch.optim.Adam(groups[GROUP_EXPERIENTIAL] + list(heads.policy.parameters()), lr=3e-3)
    rng = random.Random(0)
    for _ in range(40):                                               # imitate the history-dependent behaviour
        batch = encode_decisions(model, rig.store, rng.sample(recs, 12))
        loss = -heads.policy(batch.z, batch.a_mask, batch.b).log_prob(batch.chosen).mean()
        opt.zero_grad()
        loss.backward()
        model.clip_gradients({GROUP_EXPERIENTIAL: 1.0, GROUP_HEADS: 1.0})
        opt.step()
    trained, b = dependence()
    measure("policy_kl_under_belief_permutation_real_model_init", init["kl"])
    measure("policy_kl_under_belief_permutation_real_model_trained", trained["kl"])
    measure("policy_tv_under_belief_permutation_real_model_trained", trained["tv"])
    assert trained["kl"] > 0.05 and trained["tv"] > 0.1               # far above numerical noise (4e-9 once)
    after = model.group_hashes()
    assert all(after[g] == frozen[g] for g in frozen)                 # backbone and adapters never moved
    # same candidates, same policy head, same backbone: only the belief was permuted
    p_same = heads.policy(b.z, b.a_mask, b.b).probs
    assert torch.allclose(p_same, heads.policy(b.z, b.a_mask, b.b).probs)

    # changing only experiential weights changes the decisions: a few gradient steps on the experiential group
    # alone (policy head, adapters and backbone untouched) towards different choices
    p0 = probs()
    h0 = model.group_hashes()
    opt_e = torch.optim.Adam(groups[GROUP_EXPERIENTIAL], lr=3e-3)
    for _ in range(15):
        batch = encode_decisions(model, rig.store, rng.sample(recs, 12))
        other = torch.tensor([next(i for i, c in enumerate(r.decision.candidates) if c.action.operation == "wait")
                              for r in batch.records])
        loss = -heads.policy(batch.z, batch.a_mask, batch.b).log_prob(other).mean()
        opt_e.zero_grad()
        loss.backward()
        model.clip_gradients({GROUP_EXPERIENTIAL: 1.0})
        opt_e.step()
    h1 = model.group_hashes()
    assert h1[GROUP_EXPERIENTIAL] != h0[GROUP_EXPERIENTIAL]
    assert all(h1[g] == h0[g] for g in (GROUP_BACKBONE_FROZEN, GROUP_ADAPTERS, GROUP_HEADS))
    p1 = probs()
    tv = float(0.5 * (p1 - p0).abs().sum(-1).mean())
    measure("policy_tv_after_experiential_only_change", tv)
    assert tv > 0.05 and bool((p1.argmax(-1) != p0.argmax(-1)).any())


def test_planner_outputs_are_imagined_and_refused_by_every_outcome_api(lab_rig):
    rig = lab_rig(n_lifetimes=3)
    planner = LatentPlanner(rig.model, heads=rig.heads)
    cands = rig.candidates()
    g = torch.Generator().manual_seed(0)
    plans = planner.plan(rig.context(), rig.task.objective, cands, now=rig.clock.now(), deadline=rig.deadline,
                         validate=rig.world.validate_action, generator=g)
    assert len(plans) == len(cands) and all(isinstance(p, ImaginedEvaluation) and p.imagined for p in plans)
    assert all(p.horizon_steps == 1 and 0.0 <= p.value_mean <= 1.0 and p.latent_std > 0 for p in plans)
    assert all(p.epistemic_std > 0 and p.elapsed_s >= 1.0 for p in plans)          # uncertainty is propagated
    act = next(p for p, c in zip(plans, cands) if c.operation == "lab.act")
    assert act.elapsed_s == 3600.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        plans[0].imagined = False                                     # the flag cannot be cleared

    evs = rig.evaluate(candidates=cands)
    pred = next(e for e in evs if e.predictions).predictions[0]
    rig.clock.advance(3 * DAY)
    from cam_live.predict.providers import EvaluatorLoss
    for bad in (plans[0], {"imagined": True, "value": 1.0}):
        assert is_imagined(bad)
        with pytest.raises(ImaginedDataError):
            rig.registry.resolve(pred.prediction_id, bad)
        with pytest.raises(ImaginedDataError):
            rig.predictor.observe_outcomes([bad])
        with pytest.raises(ImaginedDataError):
            rig.predictor.resolve([bad])
        with pytest.raises(ImaginedDataError):
            build_targets([[(pred.query, bad)]])
        with pytest.raises(ImaginedDataError):
            EvaluatorLoss(rig.heads, rig.check_signature).add_outcome(bad, deadline=rig.deadline)
        with pytest.raises(ImaginedDataError):
            rig.predictor.resolve_objective(bad, rig.verifier.check_signature)
    dm = P.DecisionMaker(rig.model, heads=rig.heads, clock=rig.clock, experiments=rig.experiments)
    _, exp = dm.select_experiment(evs, **dm.propose_experiment(evs))
    with pytest.raises(ImaginedDataError):
        rig.experiments.add_outcome(exp.experiment_id, plans[0])
    # an imagined forecast may be registered (and scored later) but is permanently marked, and is no outcome
    forecast = planner.success_prediction(plans[0], rig.task.objective_id, rig.clock.now(), 5 * DAY)
    assert forecast.imagined_horizon_steps == 1 and is_imagined(forecast)
    stored = rig.registry.register(forecast)
    assert stored.imagined_horizon_steps == 1
    with pytest.raises(ImaginedDataError):
        rig.registry.resolve(pred.prediction_id, forecast)
    assert rig.registry.get(pred.prediction_id).resolved_status == OutcomeStatus.PENDING


def test_imagination_horizon_follows_measured_error_on_real_transitions(lab_rig, measure):
    rig = lab_rig(n_lifetimes=8, n_decisions=10, max_imagination_horizon=3)
    model, heads = rig.model, rig.heads
    train, held = rig.lifetimes[:5], [lt.records for lt in rig.lifetimes[5:]]
    planner = LatentPlanner(model, heads=heads)
    m0 = planner.measure(rig.store, held)
    assert m0["n_sequences"] == 3 and len(m0["errors"]) == 3 and planner.controller.allowed() == 1
    # train the transition head alone on real transitions (beliefs fixed), then measure again on held-out ones
    from cam_live.predict.providers import WorldModelLoss
    wm = WorldModelLoss(heads, fast_weight=0.0)
    opt = torch.optim.Adam(heads.transition.parameters(), lr=3e-3)
    recs = [r for lt in train for r in lt.records]
    rng = random.Random(0)
    for _ in range(30):
        with torch.no_grad():
            batch = encode_decisions(model, rig.store, rng.sample(recs, 16), with_next=True)
        tout = heads.transition(batch.z_chosen, batch.duration_s, batch.b)
        from cam_live.predict.world import transition_nll
        loss, _ = transition_nll(tout, batch.b_next, mask=batch.has_next)
        opt.zero_grad()
        loss.backward()
        opt.step()
    strict = LatentPlanner(model, heads=heads)
    m1 = strict.measure(rig.store, held)
    measure("transition_heldout_error_k1_untrained", m0["errors"][0])
    measure("transition_heldout_error_k1_trained", m1["errors"][0])
    measure("transition_heldout_error_k2_trained", m1["errors"][1])
    assert m1["errors"][0] < m0["errors"][0]                          # learning the real transitions helps
    # the horizon is exactly as long as the measured error allows: a threshold below the measured 1-step error
    # keeps one step, a threshold above the measured k-step errors admits k steps
    from cam_live.predict.imagination import HorizonController
    tight = LatentPlanner(model, heads=heads, controller=HorizonController(3, threshold=0.5 * m1["errors"][0]))
    loose = LatentPlanner(model, heads=heads, controller=HorizonController(3, threshold=1.05 * max(m1["errors"][:2])))
    tight.measure(rig.store, held)
    loose.measure(rig.store, held)
    assert tight.controller.allowed() == 1 and loose.controller.allowed() >= 2
    kw = dict(now=rig.clock.now(), deadline=rig.deadline, generator=torch.Generator().manual_seed(0))
    cands = rig.candidates()
    short = tight.plan(rig.context(), rig.task.objective, cands, horizon=3, **kw)
    long = loose.plan(rig.context(), rig.task.objective, cands, horizon=3, **kw)
    assert {p.horizon_steps for p in short} == {1}                    # asking for more does not grant more
    assert {p.horizon_steps for p in long} == {loose.controller.allowed()}
    assert sum(p.elapsed_s for p in long) > sum(p.elapsed_s for p in short)
    assert wm.eligible(recs[0])
