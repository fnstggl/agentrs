"""G2-learn copy of G-cl's probe_real.py (one diagnostic line adapted to the renamed adapter fingerprint, as in
D's docs/evidence/D/probe_real_adapted.py). G-cl probe on the REAL configuration (Qwen3-4B + 50m core, CUDA) under the shared GPU lock.

  --part write   build D's lab on the real model, one frequent update and one forced slow update through D's
                 ContinualLearner; check independently (own parameter snapshots, not D's records) which tensors
                 changed in which group; frozen backbone and reward untouched; LoRA and bridge both moved in the
                 slow step; behaviour probe changed; then weights-only export + reference predictions.
  --part read    fresh process: load_exported_model on CUDA (no store writer, no optimizer), recompute the same
                 predictions and compare bit-exactly (CL-06 on the GPU / real backbone, which D lists as not verified).

  .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_real_cl.py --part write --dir DIR
  .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_real_cl.py --part read  --dir DIR
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, "cam_live/src")
from cam_live.nn.gpu import gpu_lock                                            # noqa: E402

FAILS, OUT = [], {}
BETA = 2e-6


def check(name, ok, detail=""):
    print(("[PASS] " if ok else "[FAIL] ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def snap(model):
    return {g: [p.detach().clone() for p in ps] for g, ps in model.param_groups().items() if g != "backbone_frozen"}


def changed(a, b):
    return {g: sum(not torch.equal(x, y) for x, y in zip(a[g], b[g])) for g in a}


def predictions(model, world_head_provider, policy, store, ids):
    from cam_live.learning_api import encode_decisions
    recs = store.get_many(ids)
    model.eval()
    with torch.no_grad():
        b = encode_decisions(model, store, recs, grad=False)
        pw = world_head_provider.predict(model, b)
        pp = policy(b.z, b.a_mask, b.b).probs
    return {"b": b.b.float().cpu(), "z": b.z_chosen.float().cpu(), "nll": pw["nll"].float().cpu(), "probs": pp.float().cpu()}


def part_write(d):
    from cam_live.config import CamLiveConfig
    from cam_live.credit import checkpoint as ckpt
    from cam_live.credit.labkit import build_lab, collect_family
    from cam_live.credit.learner import LearnerSettings
    from cam_live.diagnostics.world import LabConfig
    from cam_live.learning_api import TIMESCALE_SLOW, UpdatePlan
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    torch.manual_seed(0)
    cfg = CamLiveConfig()
    cfg.device = "cuda"
    cfg.core.max_context = 64
    t0 = time.time()
    with gpu_lock(timeout_s=3600.0, min_free_gb=14.0):
        lab = build_lab(os.path.join(d, "store.sqlite"), cfg, bridge=True, beta=BETA,
                        settings=LearnerSettings.from_cfg(cfg, batch_size=8, slow_batch_size=4, grad_last=8,
                                                          val_fraction=0.2, probe_size=4, val_batch=8))
        L, m, st = lab.learner, lab.model, lab.store
        try:
            counts = m.count_parameters()
            OUT["parameters_by_group"] = counts["by_group"]
            collect_family(st, lab.linker, LabConfig(n_channels=4, family="lab/A", target=30), 5, 6, seed=3, clock=lab.clock)
            frozen0 = m.group_hashes().get("backbone_frozen")
            ids = [r.exp_id for r in st.all() if r.outcomes][:4]
            p0 = predictions(m, lab.world, lab.policy, st, ids)
            s0 = snap(m)
            plan = L.poll()
            check("poll() triggers on matured outcomes of the real-config lab", plan is not None, json.dumps(L.trigger_state))
            import dataclasses
            fr = [r for r in L.learn(dataclasses.replace(plan, max_steps=1)) if r.timescale == "frequent"][0]
            s1 = snap(m)
            c1 = changed(s0, s1)
            OUT["frequent"] = {"tensors_changed": c1, "record_param_groups": fr.param_groups, "grad_norms": fr.grad_norms,
                               "weight_delta": fr.weight_delta, "losses": fr.losses}
            check("real frequent update: experiential and heads tensors changed", c1["experiential"] > 0 and c1["heads"] > 0, str(c1))
            check("real frequent update: no adapter / reward tensor changed", c1.get("semantic_adapters", 0) == 0
                  and c1.get("reward", 0) == 0, str(c1))
            p1 = predictions(m, lab.world, lab.policy, st, ids)
            dpred = {k: float((p1[k] - p0[k]).abs().max()) for k in p0}
            OUT["frequent_prediction_change"] = dpred
            check("real frequent update changed future predictions", dpred["nll"] > 0 and dpred["z"] > 0, str(dpred))
            sl = [r for r in L.learn(UpdatePlan(trigger="probe", timescales=(TIMESCALE_SLOW,))) if r.timescale == TIMESCALE_SLOW][-1]
            s2 = snap(m)
            c2 = changed(s1, s2)
            notes = json.loads(sl.notes)
            lora = {id(p) for p in m.backbone.lora_parameters()}
            ad = m.param_groups()["semantic_adapters"]
            lora_ch = sum(not torch.equal(a, b) for p, a, b in zip(ad, s1["semantic_adapters"], s2["semantic_adapters"]) if id(p) in lora)
            br_ch = sum(not torch.equal(a, b) for p, a, b in zip(ad, s1["semantic_adapters"], s2["semantic_adapters"]) if id(p) not in lora)
            OUT["slow"] = {"tensors_changed": c2, "lora_tensors_changed": lora_ch, "bridge_tensors_changed": br_ch,
                           "n_lora_tensors": len(lora), "n_adapter_tensors": len(ad),
                           "adapter_grad_norm_by_loss": notes.get("adapter_grad_norm_by_loss"), "guard": notes.get("guard"),
                           "adapter_grad_norm": notes.get("adapter_grad_norm"), "adapter_delta_norm": notes.get("adapter_delta_norm")}
            reverted = bool((notes.get("guard") or {}).get("reverted"))
            if reverted:
                print("       slow step was reverted by the guard; checking that nothing changed instead")
                check("real slow update reverted: adapters bit-identical", c2.get("semantic_adapters", 0) == 0, str(c2))
            else:
                check("real slow update: LoRA and bridge tensors both changed", lora_ch > 0 and br_ch > 0,
                      f"lora {lora_ch}/{len(lora)}, bridge {br_ch}/{len(ad) - len(lora)}")
                check("real slow update: experiential / heads / reward unchanged",
                      c2["experiential"] == 0 and c2["heads"] == 0 and c2.get("reward", 0) == 0, str(c2))
            check("frozen backbone hash unchanged by both updates", m.group_hashes().get("backbone_frozen") == frozen0)
            # weights-only export and reference predictions for the fresh-process reload
            ckpt.export_learner_weights(L, os.path.join(d, "export"))
            ref = predictions(m, lab.world, lab.policy, st, ids)
            # stale in-process state? drop every inference cache (embeddings, dtype-cast LoRA) and recompute
            OUT["writer_backbone_state"] = {"mem_is_set": getattr(m.backbone, "_mem", None) is not None,
                                            "fingerprint_matches": (m.backbone._adapter_snapshot is not None and torch.equal(m.backbone._adapter_snapshot, m.backbone._adapter_content())),  # G2-learn: nn renamed the fingerprint (content snapshot since e00d28f)
                                            "cache_items": len(m.backbone._cache), "stats": dict(m.backbone.stats)}
            m.backbone._cache.clear()                                   # 1. embedding cache only
            ref_emb = predictions(m, lab.world, lab.policy, st, ids)
            OUT["writer_after_embedding_cache_clear_max_diff"] = {k: float((ref[k] - ref_emb[k]).abs().max()) for k in ref}
            for mod in m.modules():                                     # 2. + dtype-cast LoRA weights
                if hasattr(mod, "_cast_key"):
                    mod._cast_key, mod._cast = None, (None, None)
            m.backbone._cache.clear()
            ref_cold = predictions(m, lab.world, lab.policy, st, ids)
            OUT["writer_after_lora_cast_reset_max_diff"] = {k: float((ref_emb[k] - ref_cold[k]).abs().max()) for k in ref}
            stale = {k: float((ref[k] - ref_cold[k]).abs().max()) for k in ref}
            OUT["writer_warm_vs_cold_cache_max_diff"] = stale
            check("writer: predictions after learning do not depend on inference caches", all(v == 0.0 for v in stale.values()),
                  str(stale))
            torch.save(ref_cold, os.path.join(d, "ref_cold.pt"))
            torch.save(ref, os.path.join(d, "ref.pt"))
            json.dump({"ids": ids, "hashes": m.group_hashes()}, open(os.path.join(d, "ids.json"), "w"))
            st.flush()
            OUT["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        finally:
            lab.close()
    OUT["seconds"] = time.time() - t0


def part_read(d):
    from cam_live.credit import checkpoint as ckpt
    from cam_live.credit.labkit import register_lab_heads
    from cam_live.credit.replay import SqliteExperienceStore
    meta = json.load(open(os.path.join(d, "ids.json")))
    with gpu_lock(timeout_s=3600.0, min_free_gb=14.0):
        holder = {}
        em = ckpt.load_exported_model(os.path.join(d, "export"), device="cuda",
                                      register_heads=lambda mm: holder.update(register_lab_heads(mm, None, beta=BETA)))
        st = SqliteExperienceStore(os.path.join(d, "store.sqlite"), readonly=True)
        got = predictions(em.model, holder["world"], holder["policy"], st, meta["ids"])
        hashes = em.model.group_hashes()
        st.close()
    ref = torch.load(os.path.join(d, "ref.pt"))
    torch.save(got, os.path.join(d, "reader.pt"))
    if os.path.exists(os.path.join(d, "ref_cold.pt")):
        rc = torch.load(os.path.join(d, "ref_cold.pt"))
        OUT["reader_vs_writer_cold_cache_max_diff"] = {k: float((rc[k] - got[k]).abs().max()) for k in rc}
    diffs = {k: float((ref[k] - got[k]).abs().max()) for k in ref}
    OUT["reload_max_abs_diff"] = diffs
    OUT["load_report_clean"] = bool(em.report.clean)
    same_h = {g: meta["hashes"][g] == hashes.get(g) for g in meta["hashes"]}
    OUT["group_hashes_identical"] = same_h
    check("real model: fresh-process weights-only reload has identical group hashes", all(same_h.values()), str(same_h))
    check("real model: fresh-process reload reproduces predictions bit-exactly (CUDA)",
          all(torch.equal(ref[k], got[k]) for k in ref), str(diffs))


def part_within(d):
    """Same fresh process, same weights: predictions with a cold embedding cache in two record orders. Separates
    export/reload error from batch-composition / cache effects of the backbone (bf16)."""
    from cam_live.credit import checkpoint as ckpt
    from cam_live.credit.labkit import register_lab_heads
    from cam_live.credit.replay import SqliteExperienceStore
    meta = json.load(open(os.path.join(d, "ids.json")))
    with gpu_lock(timeout_s=3600.0, min_free_gb=14.0):
        holder = {}
        em = ckpt.load_exported_model(os.path.join(d, "export"), device="cuda",
                                      register_heads=lambda mm: holder.update(register_lab_heads(mm, None, beta=BETA)))
        st = SqliteExperienceStore(os.path.join(d, "store.sqlite"), readonly=True)
        ids = meta["ids"]
        a = predictions(em.model, holder["world"], holder["policy"], st, ids)
        b2 = predictions(em.model, holder["world"], holder["policy"], st, ids)          # warm cache, same order
        em.model.mark_adapters_updated()                                               # drop the embedding cache
        r = predictions(em.model, holder["world"], holder["policy"], st, ids[::-1])    # cold cache, reversed batch
        r = {k: v.flip(0) for k, v in r.items()}
        single = [predictions(em.model, holder["world"], holder["policy"], st, [i]) for i in ids]
        # cache history: warm the embedding cache with one-record batches, then predict the 4-record batch from it
        em.model.mark_adapters_updated()
        for i in ids:
            predictions(em.model, holder["world"], holder["policy"], st, [i])
        h = predictions(em.model, holder["world"], holder["policy"], st, ids)
        OUT["batch_from_single_record_cache_max_diff"] = {k: float((a[k] - h[k]).abs().max()) for k in a}
        st.close()
    OUT["same_order_warm_cache_max_diff"] = {k: float((a[k] - b2[k]).abs().max()) for k in a}
    OUT["reversed_order_cold_cache_max_diff"] = {k: float((a[k] - r[k]).abs().max()) for k in a}
    OUT["one_record_batches_max_diff"] = {k: float((a[k] - torch.cat([s_[k] for s_ in single])).abs().max()) for k in a}
    print(json.dumps(OUT, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["write", "read", "within"])
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    {"write": part_write, "read": part_read, "within": part_within}[a.part](a.dir)
    print(json.dumps(OUT, indent=1, default=str))
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)


if __name__ == "__main__":
    main()
