"""cam_live.nn.bench — measured parameter counts and computational benchmarks (spec 1.1, 1.3, 16.10).

    python -m cam_live.nn.bench [--skip-gpu] [--out docs/evidence/A/params.json]

1. Every scale preset: the experiential network is instantiated on CPU (with the semantic width of the configured
   backbone, read from its config without loading weights) and its parameters are counted per component.
2. Under the shared GPU lock: the real configuration (default Qwen/Qwen3-4B + LoRA + bridge + the default preset):
   frozen / adapter / experiential counts, load time, GPU memory, embedding throughput, generation tokens per second
   with and without the latent bridge, memory and time of one adapter backward pass (batch 4, 256 tokens) with and
   without gradient checkpointing, per-group gradient norms of a joint loss, and a check that a literal all-zero
   soft prefix is *not* a no-op while the bridge is.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from ..config import PROJECT_ROOT, SCALE_PRESETS, CamLiveConfig, CoreConfig
from ..contracts import ActionSpec, ExperienceItem, ItemKind, OutcomeKind, OutcomeQuery, Quantity
from .api import GROUP_ADAPTERS, GROUP_EXPERIENTIAL, TokenBatch
from .gpu import GpuUnavailable, free_gpu_memory, gpu_lock
from .model import build_experiential, build_model
from .presets import NN_PRESETS, NNConfig, preset_nn, resolve_core

GB = 2 ** 30


def backbone_width(cfg: CamLiveConfig) -> Dict[str, Any]:
    """Hidden width / depth of the configured backbone from its config file (no weights loaded)."""
    from transformers import AutoConfig
    kw: Dict[str, Any] = {"local_files_only": cfg.backbone.local_files_only,
                          "cache_dir": os.path.join(os.environ["HF_HOME"], "hub")}
    if cfg.backbone.revision:
        kw["revision"] = cfg.backbone.revision
    c = AutoConfig.from_pretrained(cfg.backbone.name, **kw)
    return {"hidden": int(c.hidden_size), "n_layers": int(c.num_hidden_layers), "n_taps": len(cfg.backbone.tap_layers)}


def _synthetic_tokens(net: Any, batch: int, T: int, device: torch.device) -> TokenBatch:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(batch, T, net.core.d_model, generator=g).to(device)
    time_ = (torch.arange(T, dtype=torch.float64) * 60.0).expand(batch, T).to(device)
    return TokenBatch(x=x, pad_mask=torch.ones(batch, T, dtype=torch.bool, device=device), time=time_,
                      kind_ids=torch.zeros(batch, T, dtype=torch.long, device=device),
                      values=torch.zeros(batch, T, 4, dtype=torch.float64, device=device),
                      value_mask=torch.zeros(batch, T, 4, dtype=torch.bool, device=device))


def preset_counts(cfg: CamLiveConfig, cpu_forward_up_to: float = 60e6, log=print) -> Dict[str, Any]:
    bw = backbone_width(cfg)
    d_sem = bw["hidden"] * bw["n_taps"]
    out: Dict[str, Any] = {}
    for scale in SCALE_PRESETS:
        if scale == "debug":
            continue
        core = CoreConfig().apply_scale(scale)
        rc = resolve_core(core)
        t0 = time.time()
        net = build_experiential(core, NNConfig(**preset_nn(core)), d_sem, bw["n_taps"])
        n = sum(p.numel() for p in net.parameters())
        entry = {"experiential_params": n, "target_params": int(core.target_params),
                 "ratio_to_target": round(n / core.target_params, 4), "breakdown": net.breakdown(),
                 "core": {k: getattr(rc, k) for k in ("d_model", "n_layers", "n_heads", "d_ff", "n_slots", "n_features")},
                 "nn_overrides": preset_nn(core), "master_preset_d_ff": SCALE_PRESETS[scale]["d_ff"],
                 "cpu_build_s": round(time.time() - t0, 2)}
        if n <= cpu_forward_up_to:
            tb = _synthetic_tokens(net, 4, 64, torch.device("cpu"))
            with torch.no_grad():
                net.core(tb)
                t1 = time.time()
                for _ in range(3):
                    net.core(tb)
            entry["cpu_core_forward_ms_b4_t64"] = round((time.time() - t1) / 3 * 1000, 1)
        out[scale] = entry
        log(f"preset {scale}: {n / 1e6:.3f} M (target {core.target_params / 1e6:.0f} M, ratio {entry['ratio_to_target']})")
        del net
    dbg = build_model(CamLiveConfig.debug())
    c = dbg.count_parameters()
    out["debug"] = {"experiential_params": c["experiential"], "target_params": int(SCALE_PRESETS["debug"]["target_params"]),
                    "ratio_to_target": round(c["experiential"] / SCALE_PRESETS["debug"]["target_params"], 4),
                    "breakdown": c["experiential_breakdown"], "note": "with the tiny test backbone (d_sem 192)",
                    "core": {k: getattr(dbg.core_cfg, k) for k in ("d_model", "n_layers", "n_heads", "d_ff", "n_slots",
                                                                    "n_features")}}
    return {"d_sem": d_sem, "backbone_width": bw, "presets": out}


_OVERALL_PEAK = [0]


def _peak(fn) -> Dict[str, float]:
    torch.cuda.synchronize()
    _OVERALL_PEAK[0] = max(_OVERALL_PEAK[0], torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.time()
    extra = fn() or {}
    torch.cuda.synchronize()
    _OVERALL_PEAK[0] = max(_OVERALL_PEAK[0], torch.cuda.max_memory_allocated())
    return {"seconds": round(time.time() - t0, 3), "peak_extra_gb": round((torch.cuda.max_memory_allocated() - base) / GB, 3),
            "peak_total_gb": round(torch.cuda.max_memory_allocated() / GB, 3), **extra}


def _passage(tok_count: int, bb: Any, seed: int) -> str:
    """A text of exactly ``tok_count`` backbone tokens (neutral filler with a varying number so texts are unique)."""
    words = ("the measured value in row {} of the table was recorded after the page finished loading and the form "
             "had been submitted without any error message being shown to the operator ").format(seed)
    ids = bb._encode(words * (tok_count // 8 + 2))[:tok_count]
    return bb._decode(ids)


def gpu_benchmark(cfg: CamLiveConfig, log=print) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats()
    mem0 = torch.cuda.memory_allocated()
    t0 = time.time()
    model = build_model(cfg)
    torch.cuda.synchronize()
    bb = model.backbone
    res["load_s"] = round(time.time() - t0, 2)
    res["gpu_memory_after_load_gb"] = round((torch.cuda.memory_allocated() - mem0) / GB, 3)
    res["gpu_peak_during_load_gb"] = round(torch.cuda.max_memory_allocated() / GB, 3)
    c = model.count_parameters()
    res["parameters"] = {k: c[k] for k in ("frozen", "experiential", "adapters", "adapters_lora", "adapters_bridge",
                                           "reward", "heads_total", "total_persistent_trainable")}
    res["parameters"]["experiential_breakdown"] = c["experiential_breakdown"]
    res["backbone"] = bb.info()
    res["arch_signature_core"] = model.arch_signature()["core"]
    log(f"loaded {bb.name} in {res['load_s']} s; counts {res['parameters']}")

    # ---- embedding throughput (unique texts, cache cold) -----------------------------------------------------------
    emb = {}
    for n_tok, n_text in ((32, 256), (128, 128), (384, 64)):
        texts = [_passage(n_tok, bb, 1000 * n_tok + i) for i in range(n_text)]
        tok0 = bb.stats["embedded_tokens"]
        r = _peak(lambda: (model.embed_texts(texts), None)[1])
        toks = bb.stats["embedded_tokens"] - tok0
        emb[f"{n_tok}_tokens"] = {"texts": n_text, "tokens": int(toks), "texts_per_s": round(n_text / r["seconds"], 1),
                                  "tokens_per_s": round(toks / r["seconds"], 0), "peak_extra_gb": r["peak_extra_gb"]}
    t1 = time.time()
    model.embed_texts(texts)
    emb["cached_384_tokens_texts_per_s"] = round(len(texts) / max(1e-9, time.time() - t1), 0)
    res["embedding_throughput"] = emb
    log(f"embedding: {emb}")

    # ---- generation speed with and without the bridge ---------------------------------------------------------------
    prompt = bb.chat_prompt("Write the integers from 1 to 300 separated by commas, with no other text.")
    gen = {}
    for bs in (1, 8):
        lat = torch.randn(bs, model.d_model, device=model.device, generator=torch.Generator(model.device).manual_seed(0))
        for name, latent in (("no_bridge", None), ("with_bridge", lat)):
            model.generate([prompt] * bs, latent=latent, max_new_tokens=4, temperature=0.0)        # warm-up
            g0 = bb.stats["generated_tokens"]
            r = _peak(lambda: (model.generate([prompt] * bs, latent=latent, max_new_tokens=128, temperature=0.0), None)[1])
            n = bb.stats["generated_tokens"] - g0
            gen[f"batch{bs}_{name}"] = {"new_tokens": int(n), "tokens_per_s": round(n / r["seconds"], 1),
                                        "seconds": r["seconds"], "peak_extra_gb": r["peak_extra_gb"]}
    res["generation"] = gen
    log(f"generation: {gen}")

    # ---- bridge no-op vs a literal all-zero soft prefix ---------------------------------------------------------------
    with torch.no_grad():
        p2 = [bb.chat_prompt("Name one colour.")]
        lat1 = torch.randn(1, model.d_model, device=model.device)
        base = bb.next_token_logits(p2)
        cond = bb.next_token_logits(p2, latent=lat1)
        ids, attn = bb._pad([bb._prompt_ids(p2[0])])
        e = bb.lm.get_input_embeddings()(ids)
        K = bb.bridge.n_tokens
        e2 = torch.cat([torch.zeros(1, K, e.shape[-1], dtype=e.dtype, device=e.device), e], 1)
        a2 = torch.cat([torch.ones(1, K, dtype=attn.dtype, device=attn.device), attn], 1)
        h2 = bb.lm.model(inputs_embeds=e2, attention_mask=a2, use_cache=False).last_hidden_state
        zp = bb._logits(h2[:, -1]).float()
        pb, pz = torch.softmax(base, -1), torch.softmax(zp, -1)
    res["bridge_noop_check"] = {
        "bridge_max_abs_logit_diff_at_init": float((base - cond).abs().max()),
        "bridge_logits_bit_identical_at_init": bool(torch.equal(base, cond)),
        "literal_zero_prefix_max_abs_logit_diff": round(float((base - zp).abs().max()), 4),
        "literal_zero_prefix_total_variation": round(float(0.5 * (pb - pz).abs().sum()), 5),
        "note": f"{K} all-zero embeddings prepended to the sequence change the next-token distribution (they take "
                "softmax attention mass and shift positions); the cross-attention bridge is exactly neutral at init"}
    log(f"no-op check: {res['bridge_noop_check']}")

    # ---- one adapter backward pass, batch 4 x 256 tokens ---------------------------------------------------------------
    prompts = [_passage(128, bb, 10 + i) for i in range(4)]
    comps = [_passage(128, bb, 50 + i) for i in range(4)]
    lat4 = torch.randn(4, model.d_model, device=model.device)
    adapters = model.param_groups()[GROUP_ADAPTERS]
    back = {}
    for name, ck in (("gradient_checkpointing", True), ("no_checkpointing", False)):
        bb.cfg.grad_checkpoint = ck

        def gen_step():
            model.zero_grad(set_to_none=True)
            lp = model.generation_logprob(prompts, comps, latent=lat4)
            (-lp.mean() / 128.0).backward()                                 # mean negative log-likelihood per token
            gl = torch.stack([p.grad.pow(2).sum() for p in bb.lora_parameters()]).sum().sqrt()
            gb = torch.stack([p.grad.pow(2).sum() for p in bb.bridge.parameters() if p.grad is not None]).sum().sqrt()
            return {"grad_norm_adapters": round(model.grad_norms()[GROUP_ADAPTERS], 5),
                    "grad_norm_lora": round(float(gl), 5), "grad_norm_bridge": round(float(gb), 5)}

        def emb_step():
            model.zero_grad(set_to_none=True)
            e = model.embed_texts([_passage(256, bb, 90 + i) for i in range(4)], with_grad=True)
            e.pow(2).mean().backward()
            return {"grad_norm_adapters": round(model.grad_norms()[GROUP_ADAPTERS], 5)}

        gen_step()                                                         # warm-up (allocator, cudnn)
        back[f"generation_logprob_{name}"] = _peak(gen_step)
        back[f"embedding_path_{name}"] = _peak(emb_step)
    bb.cfg.grad_checkpoint = cfg.backbone.grad_checkpoint
    back["shape"] = "batch 4, 256 tokens per row (generation: 128 prompt + 128 completion tokens)"
    back["adapter_params"] = sum(p.numel() for p in adapters)
    res["adapter_backward"] = back
    log(f"adapter backward: {back}")

    # ---- joint loss: per-group gradient scales (the learner clips per group) ---------------------------------------
    model.zero_grad(set_to_none=True)
    ctx = [[ExperienceItem(kind=ItemKind.OBSERVATION, text=_passage(48, bb, 200 + 10 * b + t), ts=1000.0 + 600.0 * t,
                           quantities=[Quantity("visits", 100.0 + t)]) for t in range(8)] for b in range(4)]
    tb = model.tokenize(ctx, with_grad=True, grad_last=2)
    out = model.core_forward(tb)
    e_a, _ = model.encode_actions([[ActionSpec(f"a{b}{i}", "browser.click", arguments={"ref": i}) for i in range(4)]
                                   for b in range(4)])
    e_g = model.encode_goals(["reach the stated target"] * 4)
    e_q = model.encode_queries([[OutcomeQuery("q", "visits", kind=OutcomeKind.COUNT, horizon_s=3600.0)]] * 4)[0]
    phi = model.features(model.decision_latent(out.b, e_a, e_g), e_q)
    w = torch.randn(phi.shape, device=phi.device, generator=torch.Generator(phi.device).manual_seed(0))
    (phi * w).sum().backward()
    norms = model.grad_norms()
    res["joint_loss_grad_norms"] = {"experiential": norms[GROUP_EXPERIENTIAL], "semantic_adapters": norms[GROUP_ADAPTERS],
                                    "adapters_over_experiential": round(norms[GROUP_ADAPTERS] / max(norms[GROUP_EXPERIENTIAL], 1e-30), 4),
                                    "note": "random-probe loss on phi at initialisation; adapters see the newest 2 of 8 items"}
    model.zero_grad(set_to_none=True)

    # ---- experiential network on GPU: one training step, batch 16 x 256 tokens ---------------------------------------
    tb = _synthetic_tokens(model.exp, 16, 256, model.device)

    def core_step():
        model.zero_grad(set_to_none=True)
        model.core_forward(tb).b.pow(2).mean().backward()

    core_step()
    res["experiential_core_step_b16_t256"] = _peak(core_step)
    with torch.no_grad():
        res["experiential_core_forward_b16_t256"] = _peak(lambda: (model.core_forward(tb), None)[1])
    model.zero_grad(set_to_none=True)
    res["gpu_peak_total_gb"] = round(max(_OVERALL_PEAK[0], torch.cuda.max_memory_allocated()) / GB, 3)
    del model, bb, tb, out, phi, e_a, e_g, e_q
    free_gpu_memory()
    res["gpu_memory_after_release_gb"] = round(torch.cuda.memory_allocated() / GB, 3)
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--config", default="", help="YAML config (default: built-in defaults = Qwen3-4B, 50m)")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "docs", "evidence", "A", "params.json"))
    ap.add_argument("--lock-timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)
    cfg = CamLiveConfig.load(args.config) if args.config else CamLiveConfig()
    res: Dict[str, Any] = {
        "command": "python -m cam_live.nn.bench " + " ".join(sys.argv[1:] if argv is None else argv),
        "written_at_unix": round(time.time(), 1),
        "environment": {"torch": torch.__version__, "python": platform.python_version(),
                        "cuda": torch.cuda.is_available(),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""},
        "backbone": {"name": cfg.backbone.name, "revision": cfg.backbone.revision},
    }
    try:
        import transformers
        res["environment"]["transformers"] = transformers.__version__
    except ImportError:
        res["environment"]["transformers"] = ""
    res.update(preset_counts(cfg))
    res["real_configuration"] = {"status": "skipped (--skip-gpu)"}
    if not args.skip_gpu:
        if not torch.cuda.is_available():
            res["real_configuration"] = {"status": "not measured: no CUDA device"}
        else:
            try:
                with gpu_lock(timeout_s=args.lock_timeout, min_free_gb=14.0) as t_lock:
                    gcfg = dataclasses.replace(cfg, device="cuda")
                    res["real_configuration"] = {"status": "measured", **gpu_benchmark(gcfg)}
                    res["real_configuration"]["gpu_lock_held_s"] = round(time.time() - t_lock, 1)
            except GpuUnavailable as e:
                res["real_configuration"] = {"status": f"not measured: {e}"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
