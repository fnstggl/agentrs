"""DX: CL-10 — a learner checkpoint persists every state item of spec 15.2, atomically and from the CPU.

15.2 list -> where it is in a ``credit.checkpoint`` checkpoint (capture_learner / save / load / restore_learner):
  model weights, semantic adapters, intrinsic reward weights   weights/<group>.safetensors (every trainable group)
  optimizer state                                               optim/<name>.pt (learner groups + extra optimizers)
  fast adaptive state                                           fast_state.pt
  current task                                                  state.json current_task
  replay indices                                                state.json replay_state (sampler stream, head)
  experiment records                                            state.json experiments   (added by DX)
  pending outcomes                                              state.json pending_outcomes + pending_pointer (DX)
  learning schedules                                            state.json schedules (counters, lr schedule steps)
  random seeds                                                  rng.pt (python / numpy / torch generators) + config
  authorization configuration                                   state.json auth_config_hash (+ task auth_scope)
  model versions                                                state.json model_version / reward_version,
                                                                MANIFEST group hashes and architecture signature
"""
import json
import os
import random

import torch

from cam_live.config import CamLiveConfig
from cam_live.contracts import SimClock
from cam_live.credit import checkpoint as C
from cam_live.credit.labkit import (LabRun, door_arms, door_candidates, door_config, door_weight, register_experiment,
                                    weighted_policy)
from cam_live.credit.learner import LearnerSettings
from cam_live.diagnostics.world import LabWorld
from cam_live.learning_api import UpdatePlan
from cam_live.nn.api import GROUP_ADAPTERS, GROUP_EXPERIENTIAL, GROUP_HEADS


def _lab_with_everything(lab_factory):
    cfg = CamLiveConfig.debug()
    lab = lab_factory(n_worlds=4, n_decisions=5, cfg=cfg, credit=True,
                      settings=LearnerSettings.from_cfg(cfg, batch_size=8, val_fraction=0.2, adapter_every=1))
    # intrinsic reward weights: a stand-in reward head in group "reward" (the agent's MetaLearner registers R_psi
    # there); the learner never trains it, the checkpoint must still carry it
    torch.manual_seed(3)
    lab.model.register_head("reward_test", torch.nn.Linear(lab.model.d_model, 1), group="reward")
    register_experiment(lab.store, "door-ck", ["unlock", "wait"], lab.clock.now())
    world = LabWorld(door_config(seed=7), lab.clock)
    world.rng = random.Random(7)
    run = LabRun(lab.store, lab.linker, world, seed=7)
    for t in range(3):                                         # decisions whose outcomes are still pending
        run.decide(weighted_policy(door_weight, seed=7), candidates_fn=door_candidates,
                   arms=door_arms("door-ck") if t == 0 else None)
    L = lab.learner
    L.refresh_eligibility(full=True)
    L.learn(UpdatePlan(trigger="test", max_steps=2, budget_s=1e9, exp_ids=[r.exp_id for r in lab.store.all()]))
    return lab, run


def test_every_15_2_item_is_in_the_checkpoint_and_comes_back(lab_factory, tmp_path):
    lab, run = _lab_with_everything(lab_factory)
    L, st = lab.learner, lab.store
    assert st.pending_all() and st.experiments()
    gen = torch.Generator().manual_seed(5)
    task = run.task.to_dict()
    state = C.capture_learner(L, generators={"agent": gen}, current_task=task, auth_config_hash="sha256:auth-v1",
                              reward_version="r7", lifetime_context=torch.ones(3))
    info = C.save_checkpoint(str(tmp_path / "ck"), state, clock=lab.clock)
    path = info.path
    with open(os.path.join(path, "state.json")) as fh:
        js = json.load(fh)
    manifest = json.load(open(os.path.join(path, C.MANIFEST)))
    # weights: every trainable group, including the adapters and the reward group
    groups = set(state.weights)
    assert {GROUP_EXPERIENTIAL, GROUP_HEADS, GROUP_ADAPTERS, "reward"} <= groups
    rw = {k: v for k, v in state.weights["reward"].items() if "reward_test" in k}
    assert rw and all(torch.equal(v, dict(lab.model.named_parameters())[k]) for k, v in rw.items())
    assert {f"weights/{g}.safetensors" for g in groups} <= set(manifest["files"])
    # optimizer state of every learner group
    assert set(L.optimizers) <= {f[len("optim/"):-3] for f in manifest["files"] if f.startswith("optim/")}
    # fast state, lifetime context, task, replay indices, experiments, pending outcomes, schedules, RNG, auth, versions
    assert "fast_state.pt" in manifest["files"] and "rng.pt" in manifest["files"]
    assert js["current_task"] == json.loads(json.dumps(task, default=str))
    assert js["replay_state"]["head_seq"] == st.head_seq() and "draws" in js["replay_state"]
    assert [e["experiment_id"] for e in js["experiments"]] == ["door-ck"]
    assert {q["query_id"] for q in js["pending_outcomes"]} == {q.query_id for q in st.pending_all()}
    assert js["pending_pointer"]["n_pending"] == len(js["pending_outcomes"])
    assert js["schedules"]["counters"] == L.counters and js["schedules"]["sched_step"] == L.sched_step
    assert js["auth_config_hash"] == "sha256:auth-v1" and manifest["auth_config_hash"] == "sha256:auth-v1"
    assert js["model_version"] == L.model_version and js["reward_version"] == "r7"
    assert manifest["group_hashes"] and manifest["arch_signature"]
    assert js["extra"]["config"]["seed"] == lab.cfg.seed
    # every saved tensor file holds CPU tensors only (nothing lives only on an accelerator)
    for f in manifest["files"]:
        if f.endswith(".pt"):
            obj = torch.load(os.path.join(path, f), map_location=None, weights_only=True)
            stack = [obj]
            while stack:
                x = stack.pop()
                if isinstance(x, torch.Tensor):
                    assert x.device.type == "cpu", f
                elif isinstance(x, dict):
                    stack.extend(x.values())
                elif isinstance(x, (list, tuple)):
                    stack.extend(x)
    # load gives back the same items
    loaded = C.load_checkpoint(path)
    assert loaded.experiments == js["experiments"] and loaded.pending_outcomes == js["pending_outcomes"]
    assert loaded.current_task == js["current_task"] and loaded.auth_config_hash == "sha256:auth-v1"


def test_restore_into_a_store_without_the_experiments_puts_them_back_and_checks_pending(lab_factory, tmp_path):
    lab, run = _lab_with_everything(lab_factory)
    L, st = lab.learner, lab.store
    state = C.capture_learner(L)
    C.save_checkpoint(str(tmp_path / "ck"), state)
    loaded = C.load_checkpoint(str(tmp_path / "ck"))
    # a second learner on a store that lost its experiments table rows (e.g. rebuilt from the records)
    with st._tx() as db:
        db.execute("DELETE FROM experiments")
    assert st.experiments() == []
    rep = C.restore_learner(L, loaded, rng=False)
    sr = rep["store_records"]
    assert sr["experiments_restored"] == ["door-ck"] and [e.experiment_id for e in st.experiments()] == ["door-ck"]
    back = st.experiment("door-ck")
    assert back.arms == ["unlock", "wait"] and back.assignment == "randomized" and back.status == "running"
    assert sr["pending_in_checkpoint"] == len(loaded.pending_outcomes) and not sr["pending_missing_in_store"]
    # an experiment record already in the store is never replaced by the checkpoint's older copy
    e = st.experiment("door-ck")
    e.status = "closed"
    st.put_experiment(e)
    rep = C.restore_learner(L, loaded, rng=False)
    assert rep["store_records"]["experiments_restored"] == [] and st.experiment("door-ck").status == "closed"
    # outcomes that matured after the checkpoint are counted, not resurrected
    run.mature()
    rep = C.restore_learner(L, loaded, rng=False)
    assert rep["store_records"]["pending_resolved_since"] >= 1


def test_a_checkpoint_written_before_dx_still_loads(lab_factory, tmp_path):
    lab, _ = _lab_with_everything(lab_factory)
    state = C.capture_learner(lab.learner)
    info = C.save_checkpoint(str(tmp_path / "ck"), state)
    p = os.path.join(info.path, "state.json")
    js = json.load(open(p))
    js.pop("experiments")
    js.pop("pending_outcomes")
    with open(p, "w") as fh:                                    # an old-format state.json (manifest check off)
        json.dump(js, fh)
    old = C.load_checkpoint(info.path, verify=False)
    assert old.experiments == [] and old.pending_outcomes == []
    rep = C.restore_learner(lab.learner, old, rng=False)
    assert "store_records" not in rep


def test_auth_config_hash_is_the_tasks_authorization_fields_and_a_change_is_reported(lab_factory, tmp_path):
    """CL-10 / DX-2: the hash the agent is asked to pass (change request DX-2)."""
    import dataclasses
    lab, run = _lab_with_everything(lab_factory)
    task = run.task
    h = C.auth_config_hash(task)
    assert h.startswith("sha256:") and h == C.auth_config_hash(task.to_dict()) and C.auth_config_hash(None) == ""
    assert C.auth_config_hash(dataclasses.replace(task, objective="other words")) == h      # not authorization
    changed = dataclasses.replace(task, interfaces=list(task.interfaces) + ["code"])
    assert C.auth_config_hash(changed) != h
    state = C.capture_learner(lab.learner, current_task=task.to_dict(), auth_config_hash=h)
    C.save_checkpoint(str(tmp_path / "ck"), state)
    loaded = C.load_checkpoint(str(tmp_path / "ck"))
    assert loaded.auth_config_hash == h
    assert C.restore_learner(lab.learner, loaded, rng=False, expected_auth_hash=h)["auth_config_changed"] is False
    assert C.restore_learner(lab.learner, loaded, rng=False,
                             expected_auth_hash=C.auth_config_hash(changed))["auth_config_changed"] is True


def test_promotions_keep_optimizer_state_only_in_the_newest_checkpoint(lab_factory, tmp_path):
    """Retention policy of cam_live.storage (G2-integration p03): weights in every kept checkpoint, optimizer state
    only in the newest; every kept checkpoint still verifies and loads; a rollback to a promoted checkpoint without
    optimizer state starts the learner's optimizers fresh."""
    lab, _ = _lab_with_everything(lab_factory)
    L = lab.learner
    mgr = C.CheckpointManager(str(tmp_path / "ck"), keep_last=3, keep_promoted=5)
    ret = {"lab/base/neg_nll": -1.0}
    decs = []
    for k in range(3):
        L.learn(UpdatePlan(trigger="test", max_steps=1, budget_s=1e9, exp_ids=[r.exp_id for r in lab.store.all()]))
        dec, info, _ = C.promote_learner(mgr, L, retention=ret, validation={"v": float(k)})
        decs.append((dec, info))
    assert all(d.promoted for d, _ in decs)
    infos = mgr.list()
    with_optim = [i.checkpoint_id for i in infos if os.listdir(os.path.join(i.path, "optim"))]
    assert with_optim == [infos[-1].checkpoint_id]
    for i in infos:
        assert C.verify_checkpoint(i.path) == []
        st = C.load_checkpoint(i.path)
        assert st.weights and (bool(st.optimizers) == (i.checkpoint_id == infos[-1].checkpoint_id))
        if i.checkpoint_id != infos[-1].checkpoint_id:
            assert json.load(open(os.path.join(i.path, C.MANIFEST)))["optimizer_state_pruned"]
    # a refused promotion is saved first (it becomes the newest checkpoint, so the promoted one it rolls back to keeps
    # no optimizer state): the rollback starts the learner's optimizers fresh
    L.learn(UpdatePlan(trigger="test", max_steps=1, budget_s=1e9, exp_ids=[r.exp_id for r in lab.store.all()]))
    assert any(len(o.state) for o in L.optimizers.values())
    dec, info, rb = C.promote_learner(mgr, L, retention=ret, validation={"v": -100.0})
    assert not dec.promoted and rb is not None and rb["rolled_back_to"] == decs[-1][1].checkpoint_id
    assert set(rb["optimizer_state_reset"]) == set(L.optimizers)
    assert all(len(o.state) == 0 for o in L.optimizers.values())
    newest = mgr.list()[-1]                       # the rollback is recorded as a checkpoint of the restored state
    assert newest.rollback_of and not os.listdir(os.path.join(newest.path, "optim"))      # fresh optimizers: none
    assert all(C.verify_checkpoint(i.path) == [] for i in mgr.list())
