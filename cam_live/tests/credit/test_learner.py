"""ContinualLearner on the real CamLiveModel at debug scale, with experience collected in LabWorld through the
durable store and the outcome linker (NA-08, CL-02, CL-03, CL-08)."""
import json

import pytest
import torch

from cam_live.config import CamLiveConfig
from cam_live.contracts import LearningUpdateRecord, OutcomeKind
from cam_live.credit.labkit import LabRun, WorldPredictionProvider, uniform_policy
from cam_live.credit.learner import (LEARNER_GROUPS, TIMESCALE_FAST, VAL_PREFIX, ContinualLearner, LearnerSettings,
                                     take)
from cam_live.credit.providers import BridgeGroundingProvider, ReturnDecompositionProvider
from cam_live.diagnostics.world import LabConfig, LabWorld
from cam_live.learning_api import TIMESCALE_FREQUENT, TIMESCALE_SLOW, UpdatePlan, encode_decisions
from cam_live.nn.api import GROUP_ADAPTERS, GROUP_EXPERIENTIAL, GROUP_HEADS, GROUP_REWARD

MANUAL = UpdatePlan(trigger="test", max_steps=1)


def notes(rec):
    return json.loads(rec.notes)


def test_frequent_learning_is_triggered_by_matured_independent_outcomes_not_by_time_or_rows(lab_factory):
    lab = lab_factory(collect=False, credit=False, settings=LearnerSettings(min_independent_outcomes=3))
    learner, store = lab.learner, lab.store
    world = LabWorld(LabConfig(n_channels=3, noisy_tv=False, key_door=False), lab.clock)
    run = LabRun(store, lab.linker, world, seed=1)
    pol = uniform_policy(1)
    act_only = lambda w, rng: [c for c in __import__("cam_live.diagnostics.collect", fromlist=["x"]).lab_candidates(w, rng)
                               if c.operation == "lab.act"]
    for _ in range(4):
        run.decide(pol, candidates_fn=act_only)
    # decisions were taken and time passed, but nothing has matured: no plan, however often we ask
    assert store.retrain_count() == 0 or all("outcome_matured" not in q["reasons"] for q in store.retrain_queue())
    before = [q for q in store.retrain_queue() if "outcome_matured" in q["reasons"]]
    assert before == [] and learner.poll() is None and learner.poll() is None
    lab.clock.advance(3600.0)                                   # wall-clock alone never triggers learning
    assert learner.poll() is None
    assert learner.trigger_state["threshold"] == 3
    run.mature()                                                # the delayed measurements arrive
    plan = learner.poll()
    assert plan is not None and plan.timescales == (TIMESCALE_FREQUENT,)
    st = learner.trigger_state
    assert st["independent_units"] == 4 and plan.trigger.startswith("matured_outcomes:4_independent")
    # correlated rows: every earlier decision of the objective was queued too ("return_changed"), and an event row
    # and a measurement row of the same decision are one unit, not two
    assert st["queued_rows"] >= st["matured_rows"] >= st["independent_units"]
    rows_per_unit = {}
    for q in store.retrain_queue():
        rows_per_unit[q["exp_id"]] = len(q["reasons"])
    assert max(rows_per_unit.values()) >= 2
    recs = learner.learn(plan)
    assert recs and recs[0].timescale == TIMESCALE_FREQUENT and recs[0].trigger == plan.trigger
    assert store.retrain_count() == 0 and learner.poll() is None          # consumed: the same evidence never re-triggers


def test_frequent_update_changes_experiential_and_heads_and_says_exactly_what_happened(lab_factory):
    cfg = CamLiveConfig.debug()
    lab = lab_factory(n_worlds=12, cfg=cfg, credit=False, settings=LearnerSettings.from_cfg(cfg, val_fraction=0.3))
    learner, model = lab.learner, lab.model
    plan = learner.poll()
    assert plan is not None
    adapters_before = model.group_hashes()[GROUP_ADAPTERS]
    rec = learner.learn(dataclass_replace(plan, max_steps=1))[0]
    assert isinstance(rec, LearningUpdateRecord) and rec.timescale == TIMESCALE_FREQUENT
    assert set(rec.param_groups) == {GROUP_EXPERIENTIAL, GROUP_HEADS}
    for g in (GROUP_EXPERIENTIAL, GROUP_HEADS):
        assert rec.hash_before[g] != rec.hash_after[g]
        assert rec.weight_delta[g] > 0 and rec.grad_norms[g] > 0
    assert rec.hash_before[GROUP_ADAPTERS] == rec.hash_after[GROUP_ADAPTERS] == adapters_before
    assert rec.hash_after == {g: h for g, h in model.group_hashes().items() if g in rec.hash_after}
    assert rec.losses["world"] > 0 and rec.n_experiences == len(rec.exp_ids) > 0
    # behavioural probe: predictions and action probabilities on the fixed probe batch moved
    assert rec.prediction_change["world.pred_mean"] > 0 and rec.prediction_change["policy.action_probs"] > 0
    n = notes(rec)
    assert n["behaviour_changed"] is True and "flag" not in n
    assert abs(sum(n["experiential_share"].values()) - 1.0) < 1e-6
    assert set(n["strata"]) & {"recent", "informative", "success", "failure", "delayed", "family", "fresh"}
    assert LearningUpdateRecord.from_dict(json.loads(rec.to_json())).to_dict() == rec.to_dict()
    # held-out records are never trained on
    val = {r.exp_id for r in lab.store.all() if learner.is_validation(r)}
    assert val and not (val & set(rec.exp_ids))
    assert lab.store.count(loss=VAL_PREFIX + "world") > 0


def dataclass_replace(plan, **kw):
    import dataclasses
    return dataclasses.replace(plan, **kw)


def test_weights_that_change_without_changing_behaviour_are_flagged(lab_factory):
    # only the return decomposer (a head no probe reads) is trained: hashes change, probe behaviour cannot
    lab = lab_factory(n_worlds=8, world=False, policy=False)
    lab.learner.providers = [p for p in lab.learner.providers if p.name == "rudder"]
    lab.learner.refresh_eligibility(full=True)
    rec = lab.learner.learn(MANUAL)[0]
    assert rec.param_groups == [GROUP_HEADS] and rec.weight_delta[GROUP_HEADS] > 0
    assert rec.weight_delta[GROUP_EXPERIENTIAL] == 0.0
    assert rec.prediction_change["max_abs"] == 0.0
    assert notes(rec)["behaviour_changed"] is False and "did not" in notes(rec)["flag"]


def test_the_reward_group_never_receives_a_gradient_or_a_step(lab_factory):
    lab = lab_factory(n_worlds=8, credit=False)
    model, learner = lab.model, lab.learner
    psi = model.register_head("r_psi", torch.nn.Linear(model.d_model, 1), GROUP_REWARD)

    class UsesReward(WorldPredictionProvider):
        """The world loss plus a term through R_psi that is *not* detached: a provider bug the learner must contain."""
        def loss(self, m, batch, store):
            loss, met = super().loss(m, batch, store)
            return loss + psi(batch.b).pow(2).mean(), met

    learner.providers = []
    learner.register(UsesReward(model, head_name="world2", name="world"))
    learner.refresh_eligibility(full=True)
    h0 = model.group_hashes()[GROUP_REWARD]
    rec = learner.learn(MANUAL)[0]
    assert GROUP_EXPERIENTIAL in rec.param_groups and GROUP_REWARD not in rec.param_groups
    assert model.group_hashes()[GROUP_REWARD] == h0
    assert all(p.grad is None for p in psi.parameters())
    reward_ids = {id(p) for p in psi.parameters()}
    for g, opt in learner.optimizers.items():
        assert g in LEARNER_GROUPS
        assert not any(id(p) in reward_ids for pg in opt.param_groups for p in pg["params"])


def test_no_loss_supplies_more_than_the_cap_of_the_experiential_gradient(lab_factory):
    cfg = CamLiveConfig.debug()
    lab = lab_factory(n_worlds=12, cfg=cfg, settings=LearnerSettings.from_cfg(cfg, max_share=0.9, val_fraction=0.0))
    learner, model = lab.learner, lab.model
    # a second loss that reaches the core: the return regression with attached latents, and a dominant world loss
    learner.providers = [p for p in learner.providers if p.name == "world"]
    learner.register(ReturnDecompositionProvider(model, lab.store, detach_latents=False, head_name="rudder_attached",
                                                 name="rudder_attached"))
    learner.refresh_eligibility(full=True)
    learner.learn(UpdatePlan(trigger="test", max_steps=2))      # the decomposer's output layer starts at zero
    learner.provider("world").weight = 500.0
    rec = learner.learn(MANUAL)[0]
    n = notes(rec)
    raw, applied = n["experiential_share"], n["experiential_share_applied"]
    assert raw["world"] > 0.9 and raw["rudder_attached"] > 0
    assert applied["world"] == pytest.approx(0.9, abs=1e-6) and applied["rudder_attached"] == pytest.approx(0.1, abs=1e-6)
    assert n["balance_scale"]["world"] < 1.0 and n["balance_scale"]["rudder_attached"] == 1.0
    assert learner.share_log[-1] == applied
    # without a dominant loss nothing is rescaled
    learner.provider("world").weight = 1.0
    n2 = notes(learner.learn(MANUAL)[0])
    if max(n2["experiential_share"].values()) <= 0.9:
        assert all(v == 1.0 for v in n2["balance_scale"].values())


def test_slow_adapter_update_runs_every_adapter_every_frequent_updates_and_reaches_lora_and_bridge(lab_factory):
    cfg = CamLiveConfig.debug()
    cfg.optim.adapter_every = 2
    lab = lab_factory(n_worlds=10, cfg=cfg, credit=False, bridge=True,
                      settings=LearnerSettings.from_cfg(cfg, slow_guard=False, grad_last=4))
    learner, model = lab.learner, lab.model
    out = learner.learn(UpdatePlan(trigger="test", max_steps=4))
    assert [r.timescale for r in out] == [TIMESCALE_FREQUENT, TIMESCALE_FREQUENT, TIMESCALE_SLOW] * 2
    slow = out[2]
    assert slow.param_groups == [GROUP_ADAPTERS] and "adapter_every=2" in slow.trigger
    assert slow.hash_before[GROUP_ADAPTERS] != slow.hash_after[GROUP_ADAPTERS]
    assert slow.hash_before[GROUP_EXPERIENTIAL] == slow.hash_after[GROUP_EXPERIENTIAL]
    assert slow.hash_before[GROUP_HEADS] == slow.hash_after[GROUP_HEADS]
    n = notes(slow)
    # experience-grounded losses: the world NLL through the semantic embeddings (LoRA) and the grounding loss
    # through generation conditioned on the belief (LoRA and bridge)
    assert set(slow.losses) == {"world", "bridge_grounding"}
    assert n["adapter_grad_norm"]["lora"] > 0 and n["adapter_grad_norm"]["bridge"] > 0
    assert n["adapter_delta_norm"]["lora"] > 0 and n["adapter_delta_norm"]["bridge"] > 0
    assert n["adapter_grad_norm_by_loss"]["world"] > 0 and n["adapter_grad_norm_by_loss"]["bridge_grounding"] > 0
    assert slow.weight_delta[GROUP_ADAPTERS] > 0 and slow.grad_norms[GROUP_ADAPTERS] > 0
    assert slow.prediction_change["max_abs"] > 0          # the adapters feed the latents: behaviour moved
    assert learner.counters["slow"] == 2 and learner.counters["frequent"] == 4
    # an explicit slow plan forces one
    forced = learner.learn(UpdatePlan(trigger="operator", timescales=(TIMESCALE_SLOW,)))
    assert len(forced) == 1 and forced[0].timescale == TIMESCALE_SLOW and forced[0].trigger == "operator"


def test_slow_update_that_hurts_held_out_validation_is_reverted(lab_factory):
    cfg = CamLiveConfig.debug()
    cfg.optim.lr_adapters = 0.05
    lab = lab_factory(n_worlds=12, cfg=cfg, credit=False, bridge=True,
                      settings=LearnerSettings.from_cfg(cfg, slow_guard=True, slow_guard_tol=0.0, grad_last=4,
                                                         val_fraction=0.3))
    learner, model = lab.learner, lab.model
    for p in learner.providers:                  # a harmful step on purpose: ascend the losses; it must be undone
        p.weight = -1.0
    learner.refresh_eligibility(full=True)
    h0 = model.group_hashes()[GROUP_ADAPTERS]
    rec = learner.learn(UpdatePlan(trigger="test", timescales=(TIMESCALE_SLOW,)))[0]
    g = notes(rec)["guard"]
    assert g["validation_after"] > g["validation_before"] and g["reverted"] is True
    assert model.group_hashes()[GROUP_ADAPTERS] == h0 and rec.param_groups == []
    assert learner.counters["slow_reverted"] == 1


def test_fast_timescale_updates_beliefs_without_gradient_or_parameter_change(lab_factory):
    lab = lab_factory(n_worlds=4, credit=False)
    learner, model = lab.learner, lab.model
    recs = [r for r in lab.store.all() if r.outcomes][:3]
    batch = encode_decisions(model, lab.store, recs, grad=False)
    q = [[next(iter(r.result.expected_outcomes))] for r in recs]
    e_q, _, _, _ = model.encode_queries(q)
    phi = model.features(batch.z_chosen[:, None], e_q)[:, 0, 0]
    h0 = model.group_hashes()
    mean0, var0, _ = learner.fast_state.predict(phi[0], OutcomeKind.COUNT, exposure=q[0][0].exposure)
    out = [learner.fast_update(phi[i], float(r.outcomes[0].value), OutcomeKind.COUNT, exposure=q[i][0].exposure,
                               exp_id=r.exp_id) for i, r in enumerate(recs)]
    assert all(o.timescale == TIMESCALE_FAST and o.param_groups == [] and not o.hash_before for o in out)
    assert model.group_hashes() == h0 and all(p.grad is None for p in model.parameters())
    assert learner.fast_state.n_updates(OutcomeKind.COUNT) == 3
    mean1, var1, _ = learner.fast_state.predict(phi[0], OutcomeKind.COUNT, exposure=q[0][0].exposure)
    assert float(mean1) != float(mean0) and float(var1) < float(var0)          # the belief moved and sharpened
    assert json.loads(out[0].notes)["fast"]["n"] == 1
    learner.reset_fast()                                                       # new objective: back to the prior
    assert learner.fast_state.n_updates(OutcomeKind.COUNT) == 0
    assert learner.counters["fast"] == 3 and learner.counters["frequent"] == 0


def test_learning_rate_schedule_clipping_and_optimizers_per_group(lab_factory):
    cfg = CamLiveConfig.debug()
    cfg.optim.clip_experiential, cfg.optim.clip_heads = 1e-3, 1e-2
    lab = lab_factory(n_worlds=8, cfg=cfg, credit=False,
                      settings=LearnerSettings.from_cfg(cfg, warmup_steps=4, decay_steps=8, min_lr_factor=0.1))
    learner = lab.learner
    assert set(learner.optimizers) == set(LEARNER_GROUPS)
    assert [round(learner.lr_factor(s), 4) for s in (0, 1, 3)] == [0.25, 0.5, 1.0]
    assert learner.lr_factor(4) == pytest.approx(1.0) and learner.lr_factor(8) == pytest.approx(0.55)
    assert learner.lr_factor(12) == pytest.approx(0.1) and learner.lr_factor(500) == pytest.approx(0.1)
    r1, r2 = learner.learn(UpdatePlan(trigger="test", max_steps=2))
    assert notes(r1)["lr"][GROUP_EXPERIENTIAL] == pytest.approx(cfg.optim.lr_experiential * 0.25)
    assert notes(r2)["lr"][GROUP_HEADS] == pytest.approx(cfg.optim.lr_heads * 0.5)
    # per-group clipping: the recorded norm is the one before clipping; each group has its own limit
    assert r1.grad_norms[GROUP_EXPERIENTIAL] > 1e-3 and r1.grad_norms[GROUP_HEADS] > 1e-2
    assert learner.sched_step[GROUP_EXPERIENTIAL] == 2 and learner.sched_step[GROUP_ADAPTERS] == 0
    # a head registered later joins the heads optimizer; an unknown settings key is refused
    lab.model.register_head("late", torch.nn.Linear(4, 4))
    learner._sync_optimizers()
    ids = {id(p) for pg in learner.optimizers[GROUP_HEADS].param_groups for p in pg["params"]}
    assert all(id(p) in ids for p in lab.model.head("late").parameters())
    with pytest.raises(KeyError):
        LearnerSettings.from_cfg(cfg, no_such_setting=1)


def test_take_and_bridge_provider_on_real_latents(lab_factory):
    lab = lab_factory(n_worlds=4, credit=False)
    recs = lab.store.all()[:6]
    batch = encode_decisions(lab.model, lab.store, recs, with_next=True)
    sub = take(batch, [4, 1])
    assert [r.exp_id for r in sub.records] == [recs[4].exp_id, recs[1].exp_id]
    assert torch.equal(sub.b, batch.b[[4, 1]]) and torch.equal(sub.b_next, batch.b_next[[4, 1]])
    assert sub.e_a.shape[0] == 2 and torch.equal(sub.chosen, batch.chosen[[4, 1]])
    bp = BridgeGroundingProvider(lab.model, max_rows=3)
    assert all(bp.eligible(r) for r in recs)
    loss, m = bp.loss(lab.model, batch, lab.store)
    assert loss.requires_grad and m["n_rows"] == 3
    bb = lab.model.backbone
    lora = torch.autograd.grad(loss, bb.lora_parameters(), retain_graph=True, allow_unused=True)
    bridge = torch.autograd.grad(loss, list(bb.bridge.parameters()), allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0 for g in lora)
    assert any(g is not None and float(g.abs().sum()) > 0 for g in bridge)
