"""Shared semantic server: one Qwen3-0.6B copy serving two client processes with cross-client request batching,
frozen-encoder versioning and versioned LoRA reload at batch boundaries (spec 22; manifest X06 compute side)."""
from __future__ import annotations

import json
import os
import time

import pytest
import torch

HF = os.environ.get("HF_HOME", "/workspace/.hf_home")
MODEL = "Qwen/Qwen3-0.6B"
_have_model = os.path.isdir(os.path.join(HF, "hub", "models--Qwen--Qwen3-0.6B"))
pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not (torch.cuda.is_available() and _have_model), reason="needs CUDA + Qwen3-0.6B")]

TEXTS = ["The invoice was paid two days late.", "Revenue rose to 10000 cents this week.",
         "def handler(req):\n    return {'ok': True}", "Customers in segment A complain about onboarding.",
         "TimeoutError: upstream db did not respond in 2000 ms", "Lever 3 pays out when the gauge is high."]


def _client(req_q, resp_q, cid, out_path, n_rounds):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from superlearner.cam_unified.dist.semserve import SemanticClient
    c = SemanticClient(req_q, resp_q, cid, timeout_s=300)
    c.info()
    res = {"hidden": [], "batch_clients": [], "batch_texts": [], "versions": set()}
    for r in range(n_rounds):
        ids = [c.submit("hidden", {"texts": [TEXTS[(r + i) % len(TEXTS)]], "max_tokens": 64}) for i in range(3)]
        for i, rid in enumerate(ids):
            out = c.result(rid)
            res["hidden"].append(((r + i) % len(TEXTS), out["h"][0, out["mask"][0]].float()))
            res["batch_clients"].append(out["batch_clients"])
            res["batch_texts"].append(out["batch_texts"])
            res["versions"].add(out["version"])
    e = c.embed(TEXTS[:2])
    g = c.generate(["The capital of France is"], max_new_tokens=6, greedy=True)
    res["embed"] = e
    res["gen"] = g
    res["versions"] = sorted(res["versions"])
    torch.save(res, out_path)


def test_semserve_batches_two_clients_and_reloads_lora_versions(tmp_path):
    import torch.multiprocessing as tmp
    from superlearner.cam_unified.dist.publish import CheckpointPublisher
    from superlearner.cam_unified.dist.semserve import SemanticClient, semserve_main
    ctx = tmp.get_context("spawn")
    req = ctx.Queue()
    resp = {cid: ctx.Queue() for cid in ("c1", "c2", "main")}
    stop = ctx.Event()
    sem_dir = str(tmp_path / "semserve")
    comp = str(tmp_path / "compute")
    os.makedirs(comp)
    spec = {"model_name": MODEL, "device": "cuda", "max_batch_texts": 16, "max_wait_ms": 40.0, "lora": True,
            "lora_rank": 8, "lora_alpha": 16, "lora_targets": ["q_proj", "v_proj"]}
    srv = ctx.Process(target=semserve_main, args=(spec, req, resp, stop, sem_dir, comp))
    srv.start()
    try:
        main = SemanticClient(req, resp["main"], "main", timeout_s=300)
        info = main.info()                                     # blocks until the model is loaded
        assert info["model"] == MODEL and info["sem_dim"] == 1024
        assert info["encoder_version"] == f"base:{MODEL}" and info["lora_version"] == -1
        clients = [ctx.Process(target=_client, args=(req, resp[c], c, str(tmp_path / f"{c}.pt"), 8))
                   for c in ("c1", "c2")]
        for p in clients:
            p.start()
        for p in clients:
            p.join(300)
            assert p.exitcode == 0
        r1, r2 = (torch.load(tmp_path / f"{c}.pt", weights_only=False) for c in ("c1", "c2"))
        st = main.info()
        s = st["stats"]
        # batching: requests from both processes were fused into shared forward passes
        assert st["max_clients_in_batch"] == 2
        assert max(r1["batch_clients"] + r2["batch_clients"]) == 2
        assert s["batches"] < s["requests"]
        assert max(r1["batch_texts"]) >= 3
        # batched (padded, mixed-client) results equal a genuinely unbatched pass of the same text, within the bf16
        # padding tolerance; max_tokens=63 is a fresh cache key, so this reference is computed alone, unpadded
        h_alone, m_alone = main.hidden([TEXTS[0]], max_tokens=63)
        assert main.call("info", {})["cache"]["misses"] > 0
        ref = h_alone[0, m_alone[0]].float()
        n_checked = 0
        for idx, h in r1["hidden"] + r2["hidden"]:
            if idx == 0:
                assert h.shape == ref.shape
                rel = float((h - ref).norm() / ref.norm())
                assert rel < 0.03, rel
                n_checked += 1
        assert n_checked >= 2
        # embed = mask-mean of the final hidden states
        hh, mm = main.hidden(TEXTS[:2], max_tokens=512)
        emb = (hh.float() * mm[..., None]).sum(1) / mm.sum(1, keepdim=True)
        assert torch.allclose(r1["embed"], emb, atol=0.05, rtol=0.05)
        assert r1["gen"]["text"][0].strip().startswith("Paris")
        assert all(lp <= 0 for lp in r1["gen"]["logprobs"][0])
        assert r1["versions"] == [f"base:{MODEL}"]

        # ---- versioned LoRA reload: the learner publishes adapter version 1 with non-zero B matrices
        base_gen = main.generate(["The capital of France is"], max_new_tokens=6, greedy=True)
        assert base_gen["version"].endswith("+lora:v-1")
        state = main.adapter_state()
        assert state and all(("lora_A" in k) or ("lora_B" in k) for k in state)
        g = torch.Generator().manual_seed(0)
        new = {k: (v if "lora_A" in k else torch.randn(v.shape, generator=g) * 0.5).to(v.dtype)
               for k, v in state.items()}
        CheckpointPublisher(os.path.join(sem_dir, "lora")).publish(new, 1)
        time.sleep(0.05)
        gen1 = main.generate(["The capital of France is"], max_new_tokens=6, greedy=True)
        assert gen1["version"].endswith("+lora:v1")
        assert gen1["logprobs"] != base_gen["logprobs"]        # the decoder path changed with the adapter
        h_after, _ = main.hidden([TEXTS[0]], max_tokens=63)
        assert main.version == f"base:{MODEL}"                  # frozen encoder: version unchanged ...
        assert torch.equal(h_after, h_alone)                    # ... so the same-version cache entry stays valid
        h_fresh, m_fresh = main.hidden([TEXTS[0]], max_tokens=62)   # new key: recomputed alone after the reload
        assert torch.equal(h_fresh[0, m_fresh[0]].float(), ref)  # bit-identical: adapter disabled on encoder path
        dec_hidden = main.call("generate", {"prompts": [TEXTS[0]], "max_new_tokens": 1, "greedy": True})
        assert dec_hidden["version"].endswith("+lora:v1")
        st2 = main.info()
        assert st2["stats"]["lora_reloads"] == 1 and st2["cache"]["hits"] >= 1
        assert st2["stats"]["tokens_prompt"] > 0 and st2["stats"]["tokens_generated"] >= 18
    finally:
        stop.set()
        srv.join(60)
        if srv.is_alive():
            srv.terminate()
    snap = [json.load(open(os.path.join(comp, f))) for f in os.listdir(comp)]
    assert snap and snap[0]["role"] == "semserve" and snap[0]["gpu_s"] > 0 and snap[0]["tokens_prompt"] > 0
    assert 0 < snap[0]["peak_gpu_mb"] < 6000


def test_encoder_lora_versions_invalidate_the_hidden_state_cache(tmp_path):
    """With encoder LoRA enabled (a later, versioned condition), a new adapter version changes encoder_version and
    cached hidden states are never served across versions."""
    from superlearner.cam_unified.dist.publish import CheckpointPublisher
    from superlearner.cam_unified.dist.semserve import SemanticServer
    root = str(tmp_path / "lora")
    srv = SemanticServer(MODEL, device="cuda", encoder_lora=True, lora=True, lora_rank=4, lora_alpha=8,
                         lora_targets=("q_proj", "v_proj"), lora_root=root)
    req = [("hidden", "r1", "c", {"texts": [TEXTS[1]], "max_tokens": 64})]
    (_, _, a), = srv.handle_batch(req)
    (_, _, a2), = srv.handle_batch(req)
    assert srv.cache.hits == 1 and torch.equal(a["h"], a2["h"]) and a["version"] == a2["version"]
    st = srv.adapter_state()
    g = torch.Generator().manual_seed(1)
    CheckpointPublisher(root).publish({k: (v if "lora_A" in k else torch.randn(v.shape, generator=g) * 0.5)
                                       for k, v in st.items()}, 3)
    (_, _, b), = srv.handle_batch(req)                     # reload happens at this batch boundary
    assert b["version"] != a["version"] and b["version"].endswith("+lora:v3")
    assert srv.cache.hits == 1 and srv.cache.misses == 2   # recomputed, not served from the old version's entry
    assert not torch.allclose(b["h"].float(), a["h"].float(), atol=1.0)
    del srv
    torch.cuda.empty_cache()
