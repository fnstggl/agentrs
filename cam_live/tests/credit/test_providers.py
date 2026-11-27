"""RUDDER / COCOA providers and the credit assigner on the real model's latents (debug scale), with delayed outcomes
produced by LabWorld and linked by the outcome linker (RC-02, RC-03, RC-05, RC-06, RC-07, RC-08, CU-20)."""
import math
import random

import pytest
import torch

from cam_live.config import CamLiveConfig
from cam_live.contracts import Modality, OutcomeStatus
from cam_live.credit.labkit import (LabRun, door_arms, door_candidates, door_config, door_weight, mk_action,
                                    register_experiment, run_door_experiments, run_door_training, uniform_policy,
                                    weighted_policy)
from cam_live.credit.learner import LearnerSettings
from cam_live.credit.links import OutcomeLink
from cam_live.credit.providers import (COCOA, KEY_CONTRIB, KEY_RTG, RUDDER, U_DIM, outcome_features, top_index,
                                       trajectory_return)
from cam_live.credit.simple import NAME as SIMPLE
from cam_live.diagnostics.world import LabWorld
from cam_live.learning_api import UpdatePlan

BETA = 2e-6            # per second: a two-day delay is discounted by exp(-0.35)


def unlock_seed():
    """A LabRun seed whose randomised first decision falls on the 'unlock' arm."""
    return next(s for s in range(100) if random.Random(s).choices(range(2), weights=[0.5, 0.5])[0] == 0)


def act_last(world, rng):
    return [mk_action("lab.act", channel=world.cfg.n_channels - 1, effort=3), mk_action("lab.act", channel=0, effort=1),
            mk_action("wait", Modality.INTERNAL, seconds=3600.0)]


def door_run(lab, seed=None, n_later=2, policy=None):
    seed = unlock_seed() if seed is None else seed
    world = LabWorld(door_config(seed=5), lab.clock)
    register_experiment(lab.store, "door-x", ["unlock", "wait"], lab.clock.now())
    run = LabRun(lab.store, lab.linker, world, seed=seed)
    r0 = run.decide(None, arms=door_arms("door-x"))
    pol = policy or weighted_policy(lambda a, w: 5.0 if a.arguments.get("channel") == w.cfg.n_channels - 1 else 0.5, seed=1)
    later = [run.decide(pol, candidates_fn=act_last) for _ in range(n_later)]
    return run, r0, later


def test_nothing_is_credited_before_maturation_and_the_unlock_is_credited_retrospectively(lab_factory):
    lab = lab_factory(collect=False, world=False, policy=False, beta=BETA)
    store, assigner = lab.store, lab.assigner
    run, r0, later = door_run(lab, n_later=3)
    assert run.arm == "unlock" and r0.decision.chosen.operation == "lab.unlock"
    acted = [r for r in later if r.decision.chosen.operation == "lab.act"]
    assert acted and all(r.pending_queries for r in acted)
    # --- decisions taken, consequences still in the future: no estimator credits anything
    up0 = assigner.assign()
    assert up0.matured == [] and "open" in up0.withheld[run.task.objective_id]
    now = lab.clock.now()
    for e in run.exp_ids:
        rec = store.get(e)
        assert set(rec.credit) <= {SIMPLE}          # learned credit is withheld until the return is final
        if rec.credit:                              # provisional: adopters that have *already* arrived and are linked
            links = store.links(exp_id=e)
            assert links and all(l["outcome_ts"] <= now and l["evidence"] != "declared_query" for l in links)
        assert not rec.outcomes                     # no declared measurement has matured
    assert store.get(r0.exp_id).credit == {}
    assert all(h["estimator"] == SIMPLE for h in store.credit_history())
    # --- the delayed adopters arrive and are measured; the objective is still open
    run.mature()
    marked = [e for rep in run.reports for e in rep.retrain_marked]
    assert r0.exp_id in marked                      # the earlier decision is queued for a retrospective update
    up1 = assigner.assign(marked, reason="outcome_matured")
    rec0 = store.get(r0.exp_id)
    assert SIMPLE not in rec0.credit                # no evidence links any outcome to the unlock decision ...
    assert all(SIMPLE in store.get(r.exp_id).credit for r in acted)
    assert RUDDER not in rec0.credit and COCOA not in rec0.credit and up1.matured == []      # ... and the return is not final
    # --- the verifier closes the objective: the return is final and is redistributed over the earlier decisions
    v = run.close()
    assert store.objective_outcome(run.task.objective_id)["verified"] is True
    up2 = assigner.assign(run.exp_ids, reason="objective_closed")
    assert up2.matured == [run.task.objective_id] and up2.version == up1.version + 1 == assigner.version
    rec0 = store.get(r0.exp_id)
    tr = trajectory_return(store, store.trajectory(run.task.objective_id), beta=BETA)
    assert tr.matured and tr.G > 0 and len(tr.events) == len(acted)
    assert rec0.credit[KEY_RTG] == pytest.approx(tr.G, rel=1e-9)        # return-to-go of the first decision
    assert rec0.credit[RUDDER] == pytest.approx(tr.G, rel=1e-6)         # minus F(empty) = 0 before any training
    assert COCOA in rec0.credit and math.isfinite(rec0.credit[COCOA])
    assert rec0.credit_uncertainty[RUDDER] >= 0
    # --- the history says when and why each value appeared
    hist = store.credit_history(r0.exp_id, RUDDER)
    assert [h["version"] for h in hist] == [up2.version] and hist[0]["reason"] == "objective_closed"
    simple_hist = [h for r in acted for h in store.credit_history(r.exp_id, SIMPLE)]
    assert any(h["version"] == up1.version and h["reason"] == "outcome_matured" for h in simple_hist)
    assert all(h["version"] <= up1.version for h in simple_hist)       # closing the objective changed no measurement
    # --- verified adopters equal what the verifier says, and each event entered the return once
    assert sum(e.value for e in tr.events) <= v.value + 1e-9
    assert len({e.key for e in tr.events}) == len(tr.events)


def test_return_is_conserved_under_duration_aware_discounting_and_no_outcome_links_to_a_later_decision(lab_factory):
    lab = lab_factory(collect=False, world=False, policy=False, beta=BETA)
    store, assigner = lab.store, lab.assigner
    run, r0, later = door_run(lab, n_later=4)
    run.mature()
    run.close()
    up = assigner.assign(run.exp_ids)
    assert up.conservation_error < 1e-9
    recs = store.trajectory(run.task.objective_id)
    t0 = recs[0].decision.ts
    # recompute the discounted return from the stored measurements, independently of the provider
    G = 0.0
    for r in recs:
        for o in r.outcomes:
            assert o.status == OutcomeStatus.OBSERVED and o.verified
            G += o.value * math.exp(-BETA * (o.observed_at - t0))
    undiscounted = sum(o.value for r in recs for o in r.outcomes)
    assert G < undiscounted and G > 0
    total = sum(math.exp(-BETA * (r.decision.ts - t0)) * r.credit[KEY_CONTRIB] for r in recs)
    assert total == pytest.approx(G, abs=1e-9)
    # return-to-go at each decision is the discounted remainder, expressed at that decision's own time
    togo = [sum(math.exp(-BETA * (q.decision.ts - t0)) * q.credit[KEY_CONTRIB] for q in recs[i:]) /
            math.exp(-BETA * (r.decision.ts - t0)) for i, r in enumerate(recs)]
    assert [r.credit[KEY_RTG] for r in recs] == pytest.approx(togo, abs=1e-9)
    # links: only to the decision that declared the measurement, never to a decision taken after the outcome
    links = [OutcomeLink.from_dict(l) for l in store.links()]
    assert links and all(l.decision_ts <= l.outcome_ts for l in links)
    declared = [l for l in links if l.evidence == "declared_query"]
    by_exp = {r.exp_id: r for r in recs}
    for l in declared:
        assert l.detail["query_id"] in {q.query_id for q in by_exp[l.exp_id].result.expected_outcomes}
    assert r0.exp_id not in {l.exp_id for l in links}           # the unlock gets no link: its credit is inferred
    assert lab.linker.audit_leakage() == []                     # no context item is dated after its decision
    # the simple attribution is discounted with the same convention, at the deciding decision's time
    for r in recs:
        if r.outcomes:
            want = sum(o.value * math.exp(-BETA * (o.observed_at - r.decision.ts)) for o in r.outcomes)
            assert r.credit[SIMPLE] == pytest.approx(want, rel=1e-9)


def test_cocoa_refuses_decisions_without_usable_propensities_and_claims_nothing_before_training(lab_factory):
    lab = lab_factory(collect=False, world=False, policy=False, beta=BETA)
    store, assigner = lab.store, lab.assigner

    def no_probs(cands, info):                  # a behaviour policy that did not record its probabilities
        return 0, []

    def near_zero(cands, info):
        return 0, [1.0 - 1e-6 * (len(cands) - 1)] + [1e-6] * (len(cands) - 1)

    world = LabWorld(door_config(seed=7), lab.clock)
    run = LabRun(store, lab.linker, world, seed=0)
    a = run.decide(no_probs, candidates_fn=act_last)
    b = run.decide(near_zero, candidates_fn=act_last)
    c = run.decide(uniform_policy(0), candidates_fn=act_last)
    d = run.decide(uniform_policy(1), candidates_fn=act_last)
    run.mature()
    run.close()
    up = assigner.assign(run.exp_ids)
    ra, rb, rc = (store.get(x.exp_id) for x in (a, b, c))
    assert COCOA not in ra.credit and up.unsupported[a.exp_id] == ["propensities_missing"]
    assert COCOA not in rb.credit and up.unsupported[b.exp_id] == ["near_zero_propensity"]
    assert RUDDER in ra.credit and RUDDER in rb.credit          # the return decomposition needs no propensities
    assert COCOA in rc.credit
    cc = lab.cocoa.credit([run.task.objective_id])
    assert cc[a.exp_id].status == "unsupported" and cc[a.exp_id].per_candidate is None and cc[a.exp_id].value is None
    row = cc[c.exp_id]
    # untrained hindsight model = the logged policy: contribution coefficients are exactly zero, so the credit of
    # the executed candidate is its own measured outcome and every other candidate gets nothing
    assert row.status == "supported" and row.n_later_outcomes >= 1
    assert row.value == pytest.approx(row.own, abs=1e-5)
    others = [v for i, v in enumerate(row.per_candidate) if i != rc.decision.chosen_index]
    assert others == pytest.approx([0.0] * len(others), abs=1e-5)
    # and its loss starts exactly at the cross-entropy of the logged policy
    lab.learner.refresh_eligibility(full=True)
    from cam_live.credit.learner import records_only_batch
    loss, m = lab.cocoa.loss(lab.model, records_only_batch(lab.model, store.trajectory(run.task.objective_id)), store)
    assert loss is not None and m["ce"] == pytest.approx(m["ce_policy_prior"], abs=1e-5) and m["n_pairs"] >= 1
    assert outcome_features(torch.tensor([0.0, 3.0, 50.0])).shape == (3, U_DIM)


def test_an_outcome_is_counted_once_across_levels_and_duplicate_reports(lab_factory):
    lab = lab_factory(collect=False, world=False, policy=False, beta=0.0)
    store, assigner = lab.store, lab.assigner
    world = LabWorld(door_config(seed=9), lab.clock)
    run = LabRun(store, lab.linker, world, seed=0)
    top = run.decide(uniform_policy(0), candidates_fn=lambda w, r: [mk_action("lab.unlock")])

    def child_act(w, r):                       # a lower-level operation issued under the top-level decision
        a = mk_action("lab.act", channel=w.cfg.n_channels - 1, effort=3)
        a.parent_decision_id = top.decision.decision_id
        return [a]

    child = run.decide(uniform_policy(0), candidates_fn=child_act)
    top2 = run.decide(uniform_policy(0), candidates_fn=lambda w, r: [mk_action("lab.act", channel=0, effort=2)])
    run.mature()
    # the same measurement reported again (same dedup key) is the same event
    again = store.get(child.exp_id).outcomes[0]
    rep = lab.linker.reconcile(outcomes=[again])
    assert rep.duplicates == 1 and rep.outcomes_accepted == 0
    run.close()
    recs = store.trajectory(run.task.objective_id)
    assert top_index(recs) == {top.exp_id: 0, child.exp_id: 0, top2.exp_id: 1}
    tr = trajectory_return(store, recs, beta=0.0)
    values = {r.exp_id: sum(o.value for o in r.outcomes) for r in recs}
    assert tr.exp_ids == [top.exp_id, top2.exp_id]              # the sequence is the top-level decisions
    assert tr.G == pytest.approx(values[child.exp_id] + values[top2.exp_id]) and len(tr.events) == 2
    assert [e.index for e in sorted(tr.events, key=lambda e: e.exp_id != child.exp_id)] == [0, 1]
    up = assigner.assign(run.exp_ids)
    rt, rc_, rt2 = (store.get(x.exp_id) for x in (top, child, top2))
    # redistributed over the two top-level decisions, summing to the return exactly once
    assert rt.credit[KEY_CONTRIB] + rt2.credit[KEY_CONTRIB] == pytest.approx(tr.G, abs=1e-9)
    assert RUDDER not in rc_.credit and COCOA not in rc_.credit
    # the directly linked measurement stays with the record that declared it and is not copied to its parent
    assert rc_.credit[SIMPLE] == pytest.approx(values[child.exp_id]) and SIMPLE not in rt.credit
    # the parent's counterfactual credit contains the child's outcome as its own, once
    assert lab.cocoa.credit([run.task.objective_id])[top.exp_id].own == pytest.approx(values[child.exp_id])


def test_credited_returns_follow_the_evidence_selected_estimator_and_mask_missing_credit(lab_factory):
    lab = lab_factory(collect=False, world=False, policy=False, beta=BETA)
    store, assigner = lab.store, lab.assigner
    done = run_door_experiments(lab, 1, 6, 4, seed=2)
    open_run, r0, _ = door_run(lab, n_later=1)                  # one more lifetime, still open
    assigner.assign()
    recs = store.all()
    assert assigner.default == SIMPLE                           # no randomised evidence yet: the simple attribution
    cr = assigner.credited_returns(recs)
    assert cr.estimator == SIMPLE and cr.values.dtype == torch.float64
    has = torch.tensor([SIMPLE in r.credit for r in recs])
    assert torch.equal(cr.mask.cpu(), has) and 0 < int(has.sum()) < len(recs)
    assert float(cr.values[~cr.mask].abs().sum()) == 0.0        # masked, never a target
    rud = assigner.credited_returns(recs, RUDDER)
    closed = {e for lts in done.values() for lt in lts for e in lt.exp_ids}
    assert {r.exp_id for r, m in zip(recs, rud.mask.tolist()) if m} == closed
    assert r0.exp_id not in closed and not rud.mask[[r.exp_id for r in recs].index(r0.exp_id)]
    # evidence: the measured randomised difference next to each estimator's estimate of it
    ev = assigner.compare_with_experiment("door-2-0", treatment="unlock", control="wait")
    if ev is not None:                                          # both arms have at least two matured lifetimes
        assert ev.randomized and ev.n_treatment + ev.n_control == 6
        assert RUDDER in ev.estimates and COCOA in ev.estimates and SIMPLE not in ev.estimates
    sel = assigner.select()
    assert sel.default_after == SIMPLE and not sel.changed      # one experiment validates nothing
    st = assigner.state_dict()
    assert st["credit_version"] == assigner.version and st["selector"]["default"] == SIMPLE


def test_learner_reassigns_credit_for_the_experience_that_triggered_it_and_trains_the_credit_heads(lab_factory):
    cfg = CamLiveConfig.debug()
    cfg.optim.lr_heads = 3e-3
    lab = lab_factory(collect=False, cfg=cfg, world=False, policy=False, beta=BETA,
                      settings=LearnerSettings.from_cfg(cfg, batch_size=32, val_fraction=0.25, min_independent_outcomes=4))
    learner, store = lab.learner, lab.store
    run_door_training(lab, 2, 8, 4, seed=1)        # registered experiments would be held out (validation only)
    plan = learner.poll()
    assert plan is not None and store.get(plan.exp_ids[0]).credit == {}
    lab.learner.refresh_eligibility(full=True)
    v0 = learner.validate()
    first = learner.learn(UpdatePlan(trigger=plan.trigger, max_steps=1, exp_ids=plan.exp_ids))[0]
    planned = set(store.objective_of(plan.exp_ids).values())
    assert first.credit_changes["n_records_changed"] == sum(len(store.trajectory(o)) for o in planned)
    assert first.credit_changes[RUDDER] > 0 and first.credit_changes[SIMPLE] > 0
    assert RUDDER in store.get(plan.exp_ids[0]).credit
    assert set(first.losses) == {RUDDER, COCOA} and first.param_groups == ["heads"]
    enc = lab.rudder.encoder
    hits0 = enc.stats["hits"]
    for _ in range(25):
        learner.learn(UpdatePlan(trigger="test", max_steps=1))
    v1 = learner.validate()
    assert v1[RUDDER] < v0[RUDDER]                  # held-out trajectories: the return regression generalises
    assert enc.stats["hits"] > hits0 and enc.stats["invalidations"] == 0     # heads-only training: latents cached exactly
    # once trained, a second assignment revises the stored credit and versions it
    up = lab.assigner.assign()
    assert up.changes[RUDDER] > 0 and up.conservation_error < 1e-9
    e = plan.exp_ids[0]
    assert len(store.credit_history(e, RUDDER)) == 2
