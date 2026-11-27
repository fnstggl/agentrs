"""cam_live.credit.diagnose — measured evidence for Workstream D on the real model.

    python -m cam_live.credit.diagnose [--only learning,credit,retention,restart,export] [--out PATH] [--work DIR]
    python -m cam_live.credit.diagnose --real [--out PATH]        # Qwen3-4B + 50m core on the GPU, under the shared lock

Debug scale (``CamLiveConfig.debug()``: tiny random backbone, debug core) on the CPU, in the LabWorld diagnostic
environment. Every number is computed here, from experience collected here; nothing is asserted — the slow tests in
``tests/credit/test_evidence.py`` assert on what these functions return.

    learning    stored experience -> real gradient updates: experiential + heads change and held-out prediction NLL
                improves; the slow path gives LoRA and bridge a non-zero gradient and weight delta
    credit      delayed dependency (unlock -> last channel): credit before / after maturation, conservation under
                duration-aware discounting, estimators against randomised arm differences, what the selector promotes
    retention   family A, then family B (another mechanism), forgetting on A with and without the retention mechanism
    restart     kill -9 in the middle of an update (and in the middle of a checkpoint write); resume from the last
                atomic checkpoint and compare with an uninterrupted run from that checkpoint
    export      weights-only export loaded by a fresh process without any experience: predictions bit-exact
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..config import CamLiveConfig, PROJECT_ROOT
from ..contracts import ActionSpec, ExperienceItem, OutcomeKind, OutcomeQuery
from ..diagnostics.world import LabConfig, LabWorld
from ..learning_api import TIMESCALE_FREQUENT, TIMESCALE_SLOW, UpdatePlan, encode_decisions, matured_outcomes
from ..nn.api import GROUP_ADAPTERS, GROUP_EXPERIENTIAL, GROUP_HEADS
from . import checkpoint as ckpt
from . import learner as learner_mod
from .labkit import (Lab, LabRun, _canonical, build_lab, collect_family, door_arms, door_candidates, door_config,
                     door_weight, lab_eval_fn, lab_functions, register_experiment, register_lab_heads,
                     run_door_experiments, weighted_policy)
from .learner import LearnerSettings
from .links import OutcomeLink
from .providers import COCOA, KEY_CONTRIB, KEY_RTG, RUDDER, trajectory_return
from .retention import AnchorSet, RetentionProvider, evaluate_families, measure_forgetting, retention_metrics
from .select import EstimatorSelector
from .simple import NAME as SIMPLE

EVIDENCE_DIR = os.path.join(PROJECT_ROOT, "docs", "evidence", "D")
FAM_A = LabConfig(n_channels=4, family="lab/A", target=30)
FAM_B = LabConfig(n_channels=4, family="lab/B", target=5, rate_scale=0.3, w_mean=(-1.0, 0.5, -0.3))
BETA = 2e-6


def _notes(rec: Any) -> Dict[str, Any]:
    return json.loads(rec.notes) if rec.notes else {}


def _paired(before: torch.Tensor, after: torch.Tensor) -> Dict[str, float]:
    ok = torch.isfinite(before) & torch.isfinite(after)
    d = (after - before)[ok]
    n = int(d.numel())
    return {"n": n, "before": float(before[ok].mean()), "after": float(after[ok].mean()), "change": float(d.mean()),
            "se": float(d.std(unbiased=True) / math.sqrt(n)) if n > 1 else float("nan"),
            "median_before": float(before[ok].median()), "median_after": float(after[ok].median()),
            "median_change": float(d.median()), "share_improved": float((d < 0).float().mean())}


def _fresh(work: str, name: str) -> str:
    """Path of a store that does not exist yet (an experiment never continues an earlier run's experience)."""
    path = os.path.join(work, name)
    for suffix in ("", "-wal", "-shm", ".lock"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)
    shutil.rmtree(path + ".blobs", ignore_errors=True)
    return path


def _cfg(lr_exp: float = 1e-4, lr_heads: float = 1e-3, adapter_every: int = 8, seed: int = 0) -> CamLiveConfig:
    cfg = CamLiveConfig.debug()
    cfg.seed = seed
    cfg.optim.lr_experiential, cfg.optim.lr_heads, cfg.optim.adapter_every = lr_exp, lr_heads, adapter_every
    return cfg


# ================================================================================================ (1) + (2) learning
def exp_learning(work: str, *, n_worlds: int = 150, n_decisions: int = 10, steps: int = 160, seed: int = 0) -> Dict[str, Any]:
    cfg = _cfg(seed=seed)
    lab = build_lab(_fresh(work, "learning.sqlite"), cfg, credit=False, bridge=True,
                    settings=LearnerSettings.from_cfg(cfg, val_fraction=0.2, val_batch=256, min_independent_outcomes=8))
    try:
        learner, model, store = lab.learner, lab.model, lab.store
        collect_family(store, lab.linker, FAM_A, n_worlds, n_decisions, seed=seed + 1, clock=lab.clock)
        plan = learner.poll()                                   # the matured outcomes are what triggers learning
        trigger = dict(learner.trigger_state)
        learner.refresh_eligibility(full=True)
        held = [r.exp_id for r in store.all() if learner.is_validation(r)]
        ev = lab_eval_fn(lab.world, lab.policy)
        before = evaluate_families(model, store, {"heldout": held}, ev)["heldout"]
        val0 = learner.validate()
        h0 = model.group_hashes()
        s0 = model.trainable_state()
        # fast timescale: beliefs move on matured evidence, parameters do not
        recs = [r for r in store.get_many(held) if matured_outcomes(r)][:8]
        batch = encode_decisions(model, store, recs, grad=False)
        qs = [[_canonical(matured_outcomes(r)[0][0])] for r in recs]
        phi = model.features(batch.z_chosen[:, None], model.encode_queries(qs)[0])[:, 0, 0]
        fast = [learner.fast_update(phi[i], float(matured_outcomes(r)[0][1].value), OutcomeKind.COUNT,
                                    exposure=qs[i][0].exposure, exp_id=r.exp_id) for i, r in enumerate(recs)]
        fast_ok = model.group_hashes() == h0
        t0 = time.perf_counter()
        first = learner.learn(dataclasses.replace(plan, max_steps=1))
        records = list(first)
        while learner.counters["frequent"] < steps:
            records += learner.learn(UpdatePlan(trigger="replay", max_steps=8, budget_s=1e9))
        dt = time.perf_counter() - t0
        after = evaluate_families(model, store, {"heldout": held}, ev)["heldout"]
        freq = [r for r in records if r.timescale == TIMESCALE_FREQUENT and r.param_groups]
        slow = [r for r in records if r.timescale == TIMESCALE_SLOW]
        slow_applied = [r for r in slow if r.param_groups]
        delta = model.group_delta_norms(s0)
        h1 = model.group_hashes()
        sn = [_notes(r) for r in slow_applied]
        return {
            "config": {"n_worlds": n_worlds, "n_decisions": n_decisions, "frequent_updates": len(freq), "seed": seed,
                       "lr_experiential": cfg.optim.lr_experiential, "lr_heads": cfg.optim.lr_heads,
                       "adapter_every": cfg.optim.adapter_every, "batch_size": learner.settings.batch_size},
            "parameters": model.count_parameters()["by_group"],
            "experience": {"records": store.count(), "train_eligible_world": store.count(loss="world"),
                           "heldout_world": store.count(loss="val:world"), "heldout_objectives_fraction": 0.2},
            "trigger": {"plan": plan.trigger, **trigger},
            "first_update": {"trigger": first[0].trigger, "param_groups": first[0].param_groups,
                             "grad_norms": first[0].grad_norms, "weight_delta": first[0].weight_delta,
                             "prediction_change": first[0].prediction_change, "losses": first[0].losses,
                             "hash_changed": {g: first[0].hash_before[g] != first[0].hash_after[g] for g in first[0].hash_after}},
            "frequent": {"n": len(freq), "seconds": dt, "groups_changed_every_update":
                         all(set(r.param_groups) == {GROUP_EXPERIENTIAL, GROUP_HEADS} for r in freq),
                         "updates_flagged_no_behaviour_change": sum(1 for r in freq if "flag" in _notes(r)),
                         "mean_weight_delta": {g: sum(r.weight_delta[g] for r in freq) / len(freq)
                                               for g in (GROUP_EXPERIENTIAL, GROUP_HEADS)},
                         "mean_grad_norm": {g: sum(r.grad_norms[g] for r in freq) / len(freq)
                                            for g in (GROUP_EXPERIENTIAL, GROUP_HEADS)}},
            "total_weight_delta": delta,
            "hash_changed": {g: h0[g] != h1[g] for g in h1},
            "heldout_prediction_nll": _paired(before.nll, after.nll),
            "heldout_summary_before": before.summary(), "heldout_summary_after": after.summary(),
            "heldout_provider_loss": {"before": val0, "after": learner.validate()},
            "slow_adapters": {
                "n": len(slow), "applied": len(slow_applied), "reverted_by_guard": learner.counters["slow_reverted"],
                "losses": sorted({k for r in slow for k in r.losses}),
                "grad_norm_lora": [n["adapter_grad_norm"]["lora"] for n in sn],
                "grad_norm_bridge": [n["adapter_grad_norm"]["bridge"] for n in sn],
                "delta_norm_lora": [n["adapter_delta_norm"]["lora"] for n in sn],
                "delta_norm_bridge": [n["adapter_delta_norm"]["bridge"] for n in sn],
                "grad_norm_by_loss": [n["adapter_grad_norm_by_loss"] for n in sn],
                "total_weight_delta": delta.get(GROUP_ADAPTERS, 0.0),
                "only_adapters_changed": all(r.param_groups == [GROUP_ADAPTERS] for r in slow_applied)},
            "fast": {"n": len(fast), "parameters_unchanged": fast_ok,
                     "surprise_first_last": [fast[0].losses["surprise"], fast[-1].losses["surprise"]] if fast else []},
            "counters": dict(learner.counters),
        }
    finally:
        lab.close()


# ================================================================================================ (3) delayed credit
def exp_credit(work: str, *, n_experiments: int = 6, n_per: int = 48, n_decisions: int = 5, steps: int = 400,
               seed: int = 0) -> Dict[str, Any]:
    cfg = _cfg(lr_heads=2e-3, seed=seed)
    lab = build_lab(_fresh(work, "credit.sqlite"), cfg, world=False, policy=False, credit=True, beta=BETA,
                    settings=LearnerSettings.from_cfg(cfg, batch_size=32, val_fraction=0.15),
                    selector=EstimatorSelector(min_experiments=5))
    try:
        learner, store, assigner = lab.learner, lab.store, lab.assigner
        lab.rudder.max_trajectories = lab.cocoa.max_trajectories = 32
        # --- one lifetime followed step by step: what is credited when
        register_experiment(store, "door-trace", ["unlock", "wait"], lab.clock.now())
        s = next(i for i in range(100) if random.Random(i).choices(range(2), weights=[0.5, 0.5])[0] == 0)
        run = LabRun(store, lab.linker, LabWorld(door_config(seed=77), lab.clock), seed=s)
        r0 = run.decide(None, arms=door_arms("door-trace"))
        pol = weighted_policy(door_weight, seed=5)
        for _ in range(n_decisions - 1):
            run.decide(pol, candidates_fn=door_candidates)
        assigner.assign()
        stage0 = {e: dict(store.get(e).credit) for e in run.exp_ids}
        run.mature()
        assigner.assign([e for rep in run.reports for e in rep.retrain_marked])
        stage1 = {e: dict(store.get(e).credit) for e in run.exp_ids}
        run.close()
        up = assigner.assign(run.exp_ids, reason="objective_closed")
        stage2 = {e: dict(store.get(e).credit) for e in run.exp_ids}
        tr = trajectory_return(store, store.trajectory(run.task.objective_id), beta=BETA)
        trace = {
            "first_decision": r0.decision.chosen.operation, "decisions": [store.get(e).decision.chosen.operation for e in run.exp_ids],
            "learned_credit_keys_before_any_measurement": sorted({k for c in stage0.values() for k in c if k != SIMPLE}),
            "unlock_credit_before_measurement": stage0[r0.exp_id],
            "unlock_credit_after_measurement_objective_open": stage1[r0.exp_id],
            "unlock_credit_after_objective_closed": stage2[r0.exp_id],
            "return_G_discounted": tr.G, "return_undiscounted": sum(e.value for e in tr.events),
            "conservation_error": up.conservation_error, "credit_version": up.version}
        # --- the randomised experiments
        t0 = time.perf_counter()
        runs = run_door_experiments(lab, n_experiments, n_per, n_decisions, seed=seed)
        t_collect = time.perf_counter() - t0
        learner.refresh_eligibility(full=True)
        val0 = learner.validate()
        t0 = time.perf_counter()
        first = learner.learn(learner.poll() or UpdatePlan(trigger="replay"))
        while learner.counters["frequent"] < steps:
            learner.learn(UpdatePlan(trigger="replay", max_steps=8, budget_s=1e9))
        t_train = time.perf_counter() - t0
        val1 = learner.validate()
        up = assigner.assign(reason="estimators_trained")
        trace["unlock_credit_after_estimators_trained"] = dict(store.get(r0.exp_id).credit)
        trace["unlock_credit_history"] = [{k: h[k] for k in ("estimator", "version", "value", "reason")}
                                          for h in store.credit_history(r0.exp_id)]
        links = [OutcomeLink.from_dict(l) for l in store.links()]
        unlock_first = [lt.exp_ids[0] for lts in runs.values() for lt in lts if lt.arm == "unlock"]
        wait_first = [lt.exp_ids[0] for lts in runs.values() for lt in lts if lt.arm == "wait"]

        def mean_credit(ids: Sequence[str], key: str) -> Optional[float]:
            v = [store.get(e).credit[key] for e in ids if key in store.get(e).credit]
            return sum(v) / len(v) if v else None

        evidence = []
        for xid in runs:
            ev = assigner.compare_with_experiment(xid, treatment="unlock", control="wait")
            if ev is not None:
                evidence.append({"experiment": xid, "measured_difference": ev.diff, "se": ev.se, "ci": [ev.ci_low, ev.ci_high],
                                 "n": [ev.n_treatment, ev.n_control], "randomized": ev.randomized,
                                 "estimates": ev.estimates, "estimate_se": ev.estimate_se})
        sel = assigner.select()
        zero = math.sqrt(sum(e["measured_difference"] ** 2 for e in evidence) / max(1, len(evidence)))
        return {
            "config": {"n_experiments": n_experiments, "lifetimes_per_experiment": n_per, "decisions": n_decisions,
                       "beta_per_second": BETA, "updates": learner.counters["frequent"], "lr_heads": cfg.optim.lr_heads,
                       "seed": seed},
            "single_lifetime_trace": trace,
            "experience": {"records": store.count(), "objectives": len(store.objective_ids(closed=True)),
                           "links": len(links), "collect_seconds": t_collect, "train_seconds": t_train},
            "heldout_loss": {"before": val0, "after": val1},
            "first_update_credit_changes": first[0].credit_changes,
            "assignment": {"version": up.version, "written": up.written, "matured_objectives": len(up.matured),
                           "max_conservation_error": up.conservation_error, "cocoa_unsupported": len(up.unsupported)},
            "links_to_a_later_decision": sum(1 for l in links if l.decision_ts > l.outcome_ts),
            "context_leaks": len(lab.linker.audit_leakage()),
            "first_decision_mean_credit": {
                "unlock": {k: mean_credit(unlock_first, k) for k in (SIMPLE, RUDDER, KEY_RTG, KEY_CONTRIB, COCOA)},
                "wait": {k: mean_credit(wait_first, k) for k in (SIMPLE, RUDDER, KEY_RTG, KEY_CONTRIB, COCOA)}},
            "experiments": evidence,
            "rmse_of_claiming_no_effect": zero,
            "selection": {"default_before": sel.default_before, "default_after": sel.default_after, "reasons": sel.reasons,
                          "scores": {k: {m: v[m] for m in ("n", "rmse", "mean_sq_z", "sign_agreement", "coverage",
                                                           "passes", "reasons")} for k, v in sel.scores.items()}},
        }
    finally:
        lab.close()


# ================================================================================================ (4) retention
def exp_retention(work: str, *, n_worlds: int = 40, n_decisions: int = 10, steps_a: int = 120, steps_b: int = 120,
                  seed: int = 0, weight: float = 20.0) -> Dict[str, Any]:
    cfg = _cfg(lr_exp=3e-4, adapter_every=10 ** 9, seed=seed)
    lab = build_lab(_fresh(work, "retention.sqlite"), cfg, credit=False,
                    settings=LearnerSettings.from_cfg(cfg, val_fraction=0.25, val_batch=256))
    try:
        learner, model, store = lab.learner, lab.model, lab.store
        ev = lab_eval_fn(lab.world, lab.policy)
        collect_family(store, lab.linker, FAM_A, n_worlds, n_decisions, seed=seed + 1, clock=lab.clock)
        learner.refresh_eligibility(full=True)
        while learner.counters["frequent"] < steps_a:
            learner.learn(UpdatePlan(trigger="replay", max_steps=8, budget_s=1e9))
        ids_a = [r.exp_id for r in store.all()]
        held_a = [r.exp_id for r in store.get_many(ids_a) if learner.is_validation(r)]
        train_a = [r for r in store.get_many(ids_a) if not learner.is_validation(r)]
        anchors = AnchorSet(lab_functions(lab.world, lab.policy), max_per_family=128)
        anchors.add(model, store, train_a)
        base_a = evaluate_families(model, store, {"lab/A": held_a}, ev)
        collect_family(store, lab.linker, FAM_B, n_worlds, n_decisions, seed=seed + 2, clock=lab.clock)
        learner.refresh_eligibility(full=True)
        held_b = [r.exp_id for r in store.all() if r.objective_family == "lab/B" and learner.is_validation(r)]
        base_b = evaluate_families(model, store, {"lab/B": held_b}, ev)
        n_b = sum(1 for r in store.all() if r.objective_family == "lab/B" and r.eligible.get("world"))
        start = ckpt.capture_learner(learner)
        arms: Dict[str, Any] = {}
        for name in ("naive_sequential", "retention"):
            ckpt.restore_learner(learner, start)
            if name == "retention":
                learner.settings.strata = None                           # family-balanced replay over the lifetime
                store.recent_window, store.recent_fraction = 256, 0.25
                learner.register(RetentionProvider(anchors, weight=weight, n_anchors=8, seed=seed))
            else:
                learner.settings.strata = {"recent": 1.0}                # only the newest family is replayed
                store.recent_window, store.recent_fraction = n_b, 1.0
            target = learner.counters["frequent"] + steps_b
            recs: List[Any] = []
            while learner.counters["frequent"] < target:
                recs += learner.learn(UpdatePlan(trigger="replay", max_steps=8, budget_s=1e9))
            fams: Dict[str, int] = {}
            for r in recs[-8:]:
                for e in store.get_many(r.exp_ids):
                    fams[e.objective_family] = fams.get(e.objective_family, 0) + 1
            rep_a = measure_forgetting(model, store, base_a, ev)["lab/A"]
            rep_b = measure_forgetting(model, store, base_b, ev)["lab/B"]
            shares = [max(_notes(r).get("experiential_share_applied", {"x": 0.0}).values()) for r in recs if r.param_groups]
            arms[name] = {"forgetting_on_A": rep_a, "learning_on_B": rep_b,
                          "families_in_last_batches": fams, "max_experiential_share": max(shares) if shares else None,
                          "retention_metrics": retention_metrics({"lab/A": rep_a}),
                          "losses_last_update": recs[-1].losses}
        f_n, f_r = arms["naive_sequential"]["forgetting_on_A"], arms["retention"]["forgetting_on_A"]
        return {"config": {"n_worlds_per_family": n_worlds, "steps_a": steps_a, "steps_b": steps_b, "seed": seed,
                           "retention_weight": weight, "anchors": len(anchors), "lr_experiential": cfg.optim.lr_experiential,
                           "family_A": dataclasses.asdict(FAM_A), "family_B": dataclasses.asdict(FAM_B)},
                "heldout_records": {"lab/A": len(held_a), "lab/B": len(held_b)},
                "before_B": {"lab/A": base_a["lab/A"].summary(), "lab/B": base_b["lab/B"].summary()},
                "arms": arms,
                "summary": {"nll_increase_on_A_naive": f_n["nll_change"], "nll_increase_on_A_retention": f_r["nll_change"],
                            "se_naive": f_n["nll_change_se"], "se_retention": f_r["nll_change_se"],
                            "forgetting_reduced_by": f_n["nll_change"] - f_r["nll_change"],
                            "nll_change_on_B_naive": arms["naive_sequential"]["learning_on_B"]["nll_change"],
                            "nll_change_on_B_retention": arms["retention"]["learning_on_B"]["nll_change"],
                            "decision_agreement_A_naive": f_n.get("decision_agreement"),
                            "decision_agreement_A_retention": f_r.get("decision_agreement")}}
    finally:
        lab.close()


# ================================================================================================ (5) restart recovery
def _restart_lab(store_path: str, root: str) -> Lab:
    cfg = _cfg(lr_exp=3e-4, adapter_every=2, seed=0)
    lab = build_lab(store_path, cfg, bridge=True, beta=BETA, durable=True,
                    settings=LearnerSettings.from_cfg(cfg, val_fraction=0.25, batch_size=12, grad_last=4))
    return lab


def _restart_anchors(lab: Lab) -> RetentionProvider:
    anchors = AnchorSet(lab_functions(lab.world, lab.policy), max_per_family=16)
    rp = RetentionProvider(anchors, weight=1.0, n_anchors=4)
    lab.learner.register(rp)
    return rp


def _digest(lab: Lab, rec: Any) -> Dict[str, Any]:
    fs = lab.learner.fast_state.state_dict()["post"]
    return {"timescale": rec.timescale, "losses": rec.losses, "hash_after": rec.hash_after, "exp_ids": rec.exp_ids,
            "grad_norms": rec.grad_norms, "prediction_change": rec.prediction_change,
            "fast": {k: float(v["mu"].double().sum()) for k, v in fs.items()}}


def child_restart(mode: str, work: str, k_before: int, m_after: int, kill_at: str) -> None:
    """One process of the restart experiment. mode: reference | victim | resume."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    name = "ref" if mode == "reference" else "victim"
    root = os.path.join(work, name + "_ckpt")
    lab = _restart_lab(os.path.join(work, name, "store.sqlite"), root)
    learner = lab.learner
    rp = _restart_anchors(lab)
    out: Dict[str, Any] = {"mode": mode, "pid": os.getpid()}
    if mode in ("reference", "victim"):
        rp.anchors.add(lab.model, lab.store, [r for r in lab.store.all()[:16]])
        learner.refresh_eligibility(full=True)
        recs = [r for r in lab.store.all() if matured_outcomes(r)][:3]
        batch = encode_decisions(lab.model, lab.store, recs, grad=False)
        qs = [[_canonical(matured_outcomes(r)[0][0])] for r in recs]
        phi = lab.model.features(batch.z_chosen[:, None], lab.model.encode_queries(qs)[0])[:, 0, 0]
        for i, r in enumerate(recs):
            learner.fast_update(phi[i], float(matured_outcomes(r)[0][1].value), OutcomeKind.COUNT, exposure=qs[i][0].exposure)
        for _ in range(k_before):
            learner.learn(UpdatePlan(trigger="replay", max_steps=1))
        info = ckpt.save_learner(root, learner, notes="before the interruption")
        out["checkpoint"] = {"id": info.checkpoint_id, "group_hashes": info.group_hashes, "counters": dict(learner.counters)}
        with open(os.path.join(work, f"{mode}_checkpoint.json"), "w") as fh:
            json.dump(out, fh)
        if mode == "victim":
            n = {"count": 0}
            if kill_at == "update":                 # SIGKILL with gradients computed and no optimizer step taken
                def hook(timescale: str, phase: str) -> None:
                    n["count"] += 1
                    if n["count"] == 2:
                        os.kill(os.getpid(), signal.SIGKILL)
                learner_mod.FAULT_HOOK = hook
            else:                                   # SIGKILL while the *next* checkpoint is half written
                def chook(point: str) -> None:
                    if point == "files":
                        os.kill(os.getpid(), signal.SIGKILL)
    else:
        cleaned = ckpt.recover(root)                    # what a restart does first: remove debris, find the newest intact one
        rep = ckpt.load_learner(root, learner)
        out["recovery"] = {"checkpoint_id": rep["checkpoint_id"], "weights_clean": rep["weights"].clean,
                           "recover": cleaned,
                           "arch_signature_differences": rep["arch_signature_differences"],
                           "optimizers_fresh": {k: v["fresh"] for k, v in rep["optimizers"].items()},
                           "provider_state": rep["provider_state"], "replay": rep.get("replay"),
                           "debris_left": sorted(x for x in os.listdir(root) if x.startswith(".tmp"))}
        out["counters_after_restore"] = dict(learner.counters)
    steps: List[Dict[str, Any]] = []
    for i in range(m_after):
        for r in learner.learn(UpdatePlan(trigger="replay", max_steps=1)):
            steps.append(_digest(lab, r))
        if mode == "victim" and kill_at == "checkpoint" and i == 0:
            ckpt.FAULT_HOOK = chook
            ckpt.save_learner(root, learner, notes="never completes")
    out["steps"] = steps
    out["final_hashes"] = lab.model.group_hashes()
    with open(os.path.join(work, f"{mode}.json"), "w") as fh:
        json.dump(out, fh)
    lab.close()


def _spawn(args: Sequence[str], timeout: float = 900.0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    src = os.path.join(PROJECT_ROOT, "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    return subprocess.run([sys.executable, "-m", "cam_live.credit.diagnose", *args], env=env, capture_output=True,
                          text=True, timeout=timeout)


def exp_restart(work: str, *, k_before: int = 3, m_after: int = 4, n_worlds: int = 12, n_decisions: int = 6) -> Dict[str, Any]:
    out: Dict[str, Any] = {"config": {"updates_before_checkpoint": k_before, "updates_compared": m_after}}
    for kill_at in ("update", "checkpoint"):
        w = os.path.join(work, f"restart_{kill_at}")
        shutil.rmtree(w, ignore_errors=True)
        os.makedirs(os.path.join(w, "seed"))
        lab = _restart_lab(os.path.join(w, "seed", "store.sqlite"), os.path.join(w, "seed_ckpt"))
        collect_family(lab.store, lab.linker, FAM_A, n_worlds, n_decisions, seed=4, clock=lab.clock)
        lab.store.flush()
        lab.close()
        for name in ("ref", "victim"):                       # two identical copies of the same experience
            shutil.copytree(os.path.join(w, "seed"), os.path.join(w, name))
        common = ["--work", w, "--k", str(k_before), "--m", str(m_after), "--kill-at", kill_at]
        ref = _spawn(["--child", "reference", *common])
        victim = _spawn(["--child", "victim", *common])
        res: Dict[str, Any] = {"reference_returncode": ref.returncode, "victim_returncode": victim.returncode,
                               "victim_killed_by_sigkill": victim.returncode == -signal.SIGKILL}
        if ref.returncode != 0:
            res["reference_stderr"] = ref.stderr[-2000:]
        root = os.path.join(w, "victim_ckpt")
        res["debris_after_kill"] = sorted(x for x in os.listdir(root) if x.startswith(".tmp"))
        resume = _spawn(["--child", "resume", *common])
        res["resume_returncode"] = resume.returncode
        if resume.returncode != 0:
            res["resume_stderr"] = resume.stderr[-2000:]
        if ref.returncode == 0 and resume.returncode == 0:
            a = json.load(open(os.path.join(w, "reference.json")))
            b = json.load(open(os.path.join(w, "resume.json")))
            ca = json.load(open(os.path.join(w, "reference_checkpoint.json")))
            cb = json.load(open(os.path.join(w, "victim_checkpoint.json")))
            same = [x == y for x, y in zip(a["steps"], b["steps"])]
            res.update({
                "checkpoints_identical": ca["checkpoint"]["group_hashes"] == cb["checkpoint"]["group_hashes"],
                "resumed_from": b["recovery"]["checkpoint_id"], "victim_checkpoint": cb["checkpoint"]["id"],
                "recovery": b["recovery"], "updates_compared": len(a["steps"]),
                "timescales": [s["timescale"] for s in a["steps"]],
                "identical_updates": sum(same), "all_identical": all(same) and len(a["steps"]) == len(b["steps"]),
                "final_hashes_identical": a["final_hashes"] == b["final_hashes"],
                "counters_after_restore": b["counters_after_restore"],
                "first_difference": next(({"index": i, "reference": x, "resumed": y} for i, (x, y) in
                                          enumerate(zip(a["steps"], b["steps"])) if x != y), None)})
        out[f"kill_during_{kill_at}"] = res
    return out


# ================================================================================================ (6) weights-only reload
def probe_predictions(model: Any, parts: Dict[str, Any], probe: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Predictions of every head on explicit inputs (no experience store involved)."""
    model.eval()
    with torch.no_grad():
        ctx = [[ExperienceItem.from_dict(d) for d in row["items"]] for row in probe]
        out = model.core_forward(model.tokenize(ctx), None)
        e_a, a_mask = model.encode_actions([[ActionSpec.from_dict(a) for a in row["actions"]] for row in probe])
        e_g = model.encode_goals([row["goal"] for row in probe])
        z = model.decision_latent(out.b, e_a, e_g)
        qs = [[OutcomeQuery.from_dict(row["query"])] for row in probe]
        e_q, _, kind_ids, horizon = model.encode_queries(qs)
        zc = z[:, :1]
        phi = model.features(zc, e_q)
        w = parts["world"].heads(zc, e_q, kind_ids, horizon, phi).predictive()
        pol = parts["policy"](z, a_mask, out.b)
        x = torch.cat([zc[:, 0], out.b], -1)[None]
        rud = parts["rudder"].decomposer(x, torch.ones(1, x.shape[1], dtype=torch.bool), e_g[:1])
        fast = model.new_fast_state()
        fm, fv, _ = fast.predict(phi[:, 0, 0], OutcomeKind.COUNT)
        return {"b": out.b, "z": z, "world_mean": w["mean"], "world_std": w["total_std"], "action_probs": pol.probs,
                "rudder_prefix_prediction": rud, "fast_prior_mean": fm, "fast_prior_var": fv}


def child_export(mode: str, work: str) -> None:
    torch.set_num_threads(1)
    exp_dir, probe_path = os.path.join(work, "export"), os.path.join(work, "probe.json")
    if mode == "write":
        cfg = _cfg(lr_exp=3e-4, adapter_every=2)
        lab = build_lab(os.path.join(work, "private", "store.sqlite"), cfg, bridge=True, beta=BETA,
                        settings=LearnerSettings.from_cfg(cfg, grad_last=4))
        collect_family(lab.store, lab.linker, FAM_A, 10, 6, seed=6, clock=lab.clock)
        lab.learner.learn(UpdatePlan(trigger="replay", max_steps=6))
        recs = [r for r in lab.store.all() if r.result.expected_outcomes][:6]
        probe = [{"items": [i.to_dict() for i in lab.store.items(r.context_item_refs)],
                  "actions": [r.decision.chosen.to_dict()] + [c.action.to_dict() for c in r.decision.candidates
                                                               if c.action.action_id != r.decision.chosen.action_id],
                  "goal": r.goal_text, "query": _canonical(r.result.expected_outcomes[0]).to_dict()} for r in recs]
        json.dump(probe, open(probe_path, "w"))
        parts = {"world": lab.world, "policy": lab.policy, "rudder": lab.rudder}
        meta = ckpt.export_learner_weights(lab.learner, exp_dir)
        torch.save(probe_predictions(lab.model, parts, probe), os.path.join(work, "pred_writer.pt"))
        done = [r for r in lab.store.all() if matured_outcomes(r)][:3]
        b = encode_decisions(lab.model, lab.store, done, grad=False)
        qs = [[_canonical(matured_outcomes(r)[0][0])] for r in done]
        phi = lab.model.features(b.z_chosen[:, None], lab.model.encode_queries(qs)[0])[:, 0, 0]
        for i, r in enumerate(done):                         # the writer's fast beliefs have seen evidence
            lab.learner.fast_update(phi[i], float(matured_outcomes(r)[0][1].value), OutcomeKind.COUNT,
                                    exposure=qs[i][0].exposure)
        json.dump({"counters": dict(lab.learner.counters), "n_tensors": meta["n_tensors"],
                   "group_hashes": lab.model.group_hashes(), "store_records": lab.store.count(),
                   "fast_state_updates": lab.learner.fast_state.n_updates(OutcomeKind.COUNT)},
                  open(os.path.join(work, "writer.json"), "w"))
        lab.close()
    else:
        here = os.getcwd()
        loaded = ckpt.load_exported_model(exp_dir, register_heads=lambda m: register_lab_heads(m, None, beta=BETA))
        heads = register_heads_view(loaded.model)
        probe = json.load(open(probe_path))
        torch.save(probe_predictions(loaded.model, heads, probe), os.path.join(work, "pred_reader.pt"))
        json.dump({"cwd": here, "files_in_cwd": sorted(os.listdir(here)), "load_clean": loaded.report.clean,
                   "loaded_tensors": len(loaded.report.loaded), "fast_state_updates": loaded.fast_state.n_updates(OutcomeKind.COUNT),
                   "group_hashes": {g: h for g, h in loaded.model.group_hashes().items()}},
                  open(os.path.join(work, "reader.json"), "w"))


class _View:
    def __init__(self, heads: Any, decomposer: Any = None):
        self.heads, self.decomposer = heads, decomposer


def register_heads_view(model: Any) -> Dict[str, Any]:
    """The heads of a model whose heads were registered by ``register_lab_heads``, in the shape ``probe_predictions``
    expects (no provider objects, no store)."""
    return {"world": _View(model.head("world")), "policy": model.head("policy"),
            "rudder": _View(None, model.head("credit_rudder"))}


def exp_export(work: str) -> Dict[str, Any]:
    w = os.path.join(work, "export_run")
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(os.path.join(w, "private"))
    os.makedirs(os.path.join(w, "empty_cwd"))
    a = _spawn(["--child", "export-write", "--work", w])
    if a.returncode != 0:
        return {"error": "writer failed", "stderr": a.stderr[-3000:]}
    shutil.rmtree(os.path.join(w, "private"))                 # the experience is gone before the reader starts
    env_cwd = os.path.join(w, "empty_cwd")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    b = subprocess.run([sys.executable, "-m", "cam_live.credit.diagnose", "--child", "export-read", "--work", w],
                       env=env, cwd=env_cwd, capture_output=True, text=True, timeout=900)
    if b.returncode != 0:
        return {"error": "reader failed", "stderr": b.stderr[-3000:]}
    pa, pb = torch.load(os.path.join(w, "pred_writer.pt")), torch.load(os.path.join(w, "pred_reader.pt"))
    wj, rj = json.load(open(os.path.join(w, "writer.json"))), json.load(open(os.path.join(w, "reader.json")))
    diffs = {k: float(torch.nan_to_num(pa[k].double() - pb[k].double()).abs().max()) for k in pa}
    files = sorted(os.listdir(os.path.join(w, "export")))
    return {"export_files": files, "export_bytes": sum(os.path.getsize(os.path.join(w, "export", f)) for f in files),
            "n_tensors": wj["n_tensors"], "writer_updates": wj["counters"], "writer_store_records": wj["store_records"],
            "experience_available_to_reader": os.path.exists(os.path.join(w, "private")),
            "reader_cwd_files": rj["files_in_cwd"], "reader_load_clean": rj["load_clean"],
            "writer_fast_state_updates": wj["fast_state_updates"], "reader_fast_state_updates": rj["fast_state_updates"],
            "group_hashes_identical": {g: wj["group_hashes"][g] == rj["group_hashes"].get(g) for g in wj["group_hashes"]},
            "bit_exact": {k: bool(torch.equal(pa[k], pb[k])) for k in pa}, "max_abs_difference": diffs,
            "all_bit_exact": all(bool(torch.equal(pa[k], pb[k])) for k in pa), "probe_rows": int(pa["b"].shape[0])}


# ================================================================================================ (7) real configuration
def exp_real(work: str, *, n_worlds: int = 6, n_decisions: int = 6, batch_size: int = 8, slow_batch: int = 4,
             grad_last: int = 8) -> Dict[str, Any]:
    """One frequent update and one slow adapter update with Qwen3-4B + the 50m core on a small stored batch.
    The caller holds the shared GPU lock."""
    cfg = CamLiveConfig()
    cfg.device = "cuda"
    cfg.core.max_context = 64
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    lab = build_lab(_fresh(work, "real.sqlite"), cfg, bridge=True, beta=BETA,
                    settings=LearnerSettings.from_cfg(cfg, batch_size=batch_size, slow_batch_size=slow_batch,
                                                      grad_last=grad_last, val_fraction=0.2, slow_guard=True,
                                                      probe_size=4, val_batch=8))
    try:
        learner, model, store = lab.learner, lab.model, lab.store
        t_load = time.perf_counter() - t0
        mem_load = torch.cuda.max_memory_allocated() / 2 ** 30
        collect_family(store, lab.linker, FAM_A, n_worlds, n_decisions, seed=3, clock=lab.clock)
        plan = learner.poll()
        counts = model.count_parameters()
        res: Dict[str, Any] = {
            "backbone": counts["backbone"], "backbone_pretrained": counts["backbone_pretrained"], "scale": counts["scale"],
            "parameters": counts["by_group"], "adapters_lora": counts["adapters_lora"],
            "adapters_bridge": counts["adapters_bridge"], "heads": counts["heads"],
            "experience": {"records": store.count(), "trigger": plan.trigger if plan else None},
            "load_seconds": t_load, "gpu_gb_after_load": mem_load,
            "settings": {"batch_size": batch_size, "slow_batch_size": slow_batch, "grad_last": grad_last,
                         "max_context": cfg.core.max_context, "grad_checkpoint": cfg.backbone.grad_checkpoint}}
        for name, p in (("frequent", dataclasses.replace(plan or UpdatePlan(trigger="replay"), max_steps=1)),
                        ("slow_adapters", UpdatePlan(trigger="replay", timescales=(TIMESCALE_SLOW,)))):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rec = [r for r in learner.learn(p) if r.timescale == name][0]
            torch.cuda.synchronize()
            n = _notes(rec)
            res[name] = {"seconds": time.perf_counter() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                         "trigger": rec.trigger, "losses": rec.losses, "param_groups": rec.param_groups,
                         "grad_norms": rec.grad_norms, "weight_delta": rec.weight_delta,
                         "hash_changed": {g: rec.hash_before.get(g) != rec.hash_after.get(g) for g in rec.hash_after},
                         "prediction_change": rec.prediction_change, "n_experiences": rec.n_experiences,
                         "credit_changes": rec.credit_changes,
                         "notes": {k: n.get(k) for k in ("experiential_share", "experiential_share_applied", "balance_scale",
                                                         "adapter_grad_norm", "adapter_delta_norm",
                                                         "adapter_grad_norm_by_loss", "guard", "skipped", "flag",
                                                         "behaviour_changed", "lr", "rows", "strata")}}
        res["counters"] = dict(learner.counters)
        res["backbone_stats"] = {k: (float(v) if isinstance(v, (int, float)) else str(v))
                                 for k, v in getattr(model.backbone, "stats", {}).items()}
        return res
    finally:
        lab.close()
        del lab
        import gc
        gc.collect()
        torch.cuda.empty_cache()


# ================================================================================================ CLI
EXPERIMENTS = {"learning": exp_learning, "credit": exp_credit, "retention": exp_retention, "restart": exp_restart,
               "export": exp_export}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ",".join(EXPERIMENTS))
    ap.add_argument("--out", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--real", action="store_true", help="real configuration on the GPU (docs/evidence/D/real_update.json)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--child", default="")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--kill-at", default="update")
    a = ap.parse_args(argv)
    if a.child:
        if a.child.startswith("export-"):
            child_export(a.child.split("-", 1)[1], a.work)
        else:
            child_restart(a.child, a.work, a.k, a.m, a.kill_at)
        return 0
    work = a.work or tempfile.mkdtemp(prefix="camlive_D_")
    os.makedirs(work, exist_ok=True)
    if a.real:
        from ..nn.gpu import gpu_lock
        out_path = a.out or os.path.join(EVIDENCE_DIR, "real_update.json")
        with gpu_lock(timeout_s=1800.0, min_free_gb=16.0):
            res = exp_real(work)
        res["_command"] = "cd /workspace/camlive && .venv/bin/python -m cam_live.credit.diagnose --real"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(res, open(out_path, "w"), indent=1, sort_keys=True, default=str)
        print(json.dumps({k: res[k] for k in ("frequent", "slow_adapters")}, indent=1, default=str))
        return 0
    torch.set_num_threads(a.threads)
    out_path = a.out or os.path.join(EVIDENCE_DIR, "diagnostics.json")
    names = [n for n in (a.only.split(",") if a.only else EXPERIMENTS) if n]
    result: Dict[str, Any] = {}
    if os.path.exists(out_path):
        try:
            result = json.load(open(out_path))
        except ValueError:
            result = {}
    result["_command"] = "cd /workspace/camlive && .venv/bin/python -m cam_live.credit.diagnose"
    result["_scale"] = "CamLiveConfig.debug(): tiny random backbone (not a pretrained model), debug core, CPU, LabWorld"
    for n in names:
        t0 = time.perf_counter()
        torch.manual_seed(0)
        result[n] = EXPERIMENTS[n](work)
        result[n]["_seconds"] = time.perf_counter() - t0
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(result, open(out_path, "w"), indent=1, sort_keys=True, default=str)
        print(f"[{n}] done in {result[n]['_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
