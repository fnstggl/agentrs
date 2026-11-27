"""Workstream B on the real configuration (frozen Qwen3-4B + LoRA + bridge, 50m experiential core) — gpu-marked.
One short LabWorld episode through the runtime path and one backward pass of the frequent and the slow world-model
provider. Writes docs/evidence/B/real_config.json. Initialisation-level evidence only: nothing is trained here."""
import json
import os
import time

import pytest
import torch

from cam_live.contracts import OutcomeKind, OutcomeStatus, RunFiles

EVIDENCE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "evidence", "B"))


@pytest.mark.gpu
def test_predictor_decisions_and_providers_on_the_real_configuration(gpu_heavy, tmp_path):
    from cam_live.config import CamLiveConfig
    from cam_live.diagnostics.collect import obs_item
    from cam_live.diagnostics.memstore import MemoryStore
    from cam_live.diagnostics.world import AllowAllPolicy, LabVerifier, LabWorld, lab_task
    from cam_live.learning_api import encode_decisions, matured_outcomes
    from cam_live.nn.api import GROUP_ADAPTERS, GROUP_BACKBONE_FROZEN, GROUP_EXPERIENTIAL, GROUP_HEADS, GROUP_REWARD
    from cam_live.nn.model import build_model
    from cam_live.predict import diagnose as G
    from cam_live.predict import labdata as L
    from cam_live.predict import policy as P
    from cam_live.predict.candidates import schema_candidates
    from cam_live.predict.heads import attach_heads
    from cam_live.predict.planner import LatentPlanner
    from cam_live.predict.predictor import Predictor
    from cam_live.predict.providers import WorldModelAdapterLoss, WorldModelLoss
    from cam_live.predict.registry import PredictionRegistry

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = build_model(CamLiveConfig(device="cuda"))
    try:
        heads = attach_heads(model)
        counts = model.count_parameters()
        report = {"backbone": model.backbone.name, "scale": counts["scale"], "d_model": model.d_model,
                  "n_features": model.n_features, "head_parameters": {k: v for k, v in counts["heads"].items()
                                                                     if k.startswith("predict_")},
                  "heads_total": counts["heads_total"], "load_seconds": round(time.time() - t0, 1)}
        assert counts["frozen"] > 4e9 and report["heads_total"] > 1e5 and counts["reward"] == 0

        # ---- runtime path: predict (registered) -> decide -> act -> outcome -> fast belief update ----------
        world = LabWorld(L.lab_config(seed=11))
        clock, task, verifier = world.clock, lab_task(world), LabVerifier(world)
        registry = PredictionRegistry(str(tmp_path / RunFiles.PREDICTIONS), clock)
        fast = model.new_fast_state()
        pred = Predictor(model, fast, registry, clock, heads=heads)
        dm = P.DecisionMaker(model, heads=heads, clock=clock, seed=0)
        ctx = [obs_item(o) for o in world.observe()]
        cands = schema_candidates(world.discover_tools(), per_tool=6)
        queries = G.act_queries()
        t1 = time.time()
        evs = pred.evaluate_candidates(ctx, task.objective, cands, queries, objective_id=task.objective_id,
                                       deadline=world.t0 + world.cfg.horizon_s, objective_scale=30.0,
                                       validate=world.validate_action)
        report["evaluate_seconds_first"] = round(time.time() - t1, 2)
        t1 = time.time()
        evs = pred.evaluate_candidates(ctx, task.objective, cands, queries, validate=world.validate_action)
        report["evaluate_seconds_cached_embeddings"] = round(time.time() - t1, 2)
        acts = [e for e in evs if e.action.operation == "lab.act"]
        assert acts and all(len(e.predictions) == 2 and e.predictions[0].dist.epistemic_std > 0 for e in acts)
        assert all(registry.get(p.prediction_id).made_at == clock.now() for e in acts for p in e.predictions)
        assert all(e.voi is not None and e.voi >= 0 for e in evs)
        hashes = model.group_hashes()
        dec = dm.select(acts, P.RULE_THOMPSON, objective_id=task.objective_id)
        assert abs(sum(dec.probabilities) - 1.0) < 1e-12
        res = world.execute(dec.chosen, AllowAllPolicy().authorize(dec.chosen, task, {}))
        clock.advance(3 * 86400.0)
        outs = verifier.measure([p.query for p in acts[dec.chosen_index].predictions])
        info = pred.observe_outcomes(outs)
        assert info["n"] == 2 and info["kl"] > 0 and model.group_hashes() == hashes        # no gradient step
        assert len(pred.resolve(outs)) == 2 and fast.n_updates(OutcomeKind.COUNT) == 2
        report["fast_update"] = {"surprise_nats": info["surprise"], "belief_change_kl_nats": info["kl"]}
        report["decision_rationale"] = dec.rationale

        # ---- the action distribution depends on the belief at initialisation --------------------------------
        store = MemoryStore()
        lts = L.collect(store, range(2), 8)
        recs = [r for lt in lts for r in lt.records]
        with torch.no_grad():
            b = encode_decisions(model, store, recs[:8], grad=False)
            dep = P.latent_dependence(heads.policy, b.z, b.a_mask, b.b,
                                      z_fn=lambda bel: model.decision_latent(bel, b.e_a, b.e_g))
        report["policy_kl_under_belief_permutation_init"] = dep["kl"]
        assert dep["kl"] > 1e-7
        plans = LatentPlanner(model, heads=heads).plan(ctx, task.objective, cands, now=clock.now(),
                                                       deadline=world.t0 + world.cfg.horizon_s)
        assert all(p.imagined and p.horizon_steps == 1 for p in plans)

        # ---- one backward pass per timescale ------------------------------------------------------------------
        train = [r for r in recs if matured_outcomes(r)][:4]
        wm = WorldModelLoss(heads)
        loss, m = wm.loss(model, encode_decisions(model, store, train, with_next=True), store)
        model.zero_grad()
        loss.backward()
        g = model.grad_norms()
        assert g[GROUP_EXPERIENTIAL] > 0 and g[GROUP_HEADS] > 0 and g[GROUP_ADAPTERS] == 0 and g[GROUP_REWARD] == 0
        report["frequent"] = {"loss": float(loss.detach()), "grad_norms": g, "n_targets": m["world_n_targets"]}
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        wma = WorldModelAdapterLoss(heads)
        loss, m = wma.loss(model, encode_decisions(model, store, train, grad=False), store)
        model.zero_grad()
        loss.backward()
        g = model.grad_norms()
        assert g[GROUP_ADAPTERS] > 0 and g[GROUP_BACKBONE_FROZEN] == 0                     # LoRA learns from prediction
        report["slow_adapters"] = {"loss": float(loss.detach()), "grad_norms": g, "n_targets": m["world_n_targets"],
                                   "extra_peak_gb": round((torch.cuda.max_memory_allocated() - base) / 2 ** 30, 2),
                                   "adapter_grad_last": heads.cfg.adapter_grad_last,
                                   "adapter_max_records": heads.cfg.adapter_max_records}
        model.zero_grad()
        report["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        report["command"] = ("cd /workspace/camlive && .venv/bin/python -m pytest "
                             "cam_live/tests/predict/test_gpu_real_config.py -m gpu -q --rootdir=cam_live")
        assert report["peak_gpu_gb"] < 24.0
        os.makedirs(EVIDENCE, exist_ok=True)
        with open(os.path.join(EVIDENCE, "real_config.json"), "w") as f:
            json.dump(report, f, indent=1, sort_keys=True)
    finally:
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
