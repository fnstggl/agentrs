"""G2-learn probe on the REAL configuration (Qwen/Qwen3-4B + LoRA + latent bridge + the 50M experiential network),
one load under the shared GPU lock (``cam_live.nn.gpu.gpu_lock``, < 15 min).

    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_real_learn.py

  NA-01..NA-04  frozen / LoRA / bridge / experiential counts recomputed from the module tree; frozen = no grad, bf16;
  NA-12         bridge is an exact no-op at initialisation (logits bit-identical with and without a latent); one Adam
                step on the bridge alone makes the next-token distribution depend on the latent;
  NA-09 / D10   adapter caches follow the content: LoRA weights replaced through ``p.data = ...`` (new storage, fresh
                version counter) and through ``load_trainable_state`` change the cached embedding and the no-grad
                logits exactly as a cold model would compute them;
  NA-05 / D8    text -> embedding -> belief is bit-identical whatever else is in the call: texts at every bucket edge
                (8 / 64 / 65 / 128 / 129 / 256 / 257 tokens), longer than max_text_tokens, more texts of one bucket than
                one forward holds, in different positions and with different companions, cold cache each time;
                ``embed(with_grad=True)`` returns the canonical value bit-exactly;
  NA-05 / D7    a candidate that differs in one argument moves z (cos < 0.99).
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import check, finish, info  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

from cam_live.config import CamLiveConfig  # noqa: E402
from cam_live.contracts import ActionSpec, ExperienceItem, ItemKind, Quantity  # noqa: E402
from cam_live.nn.api import GROUP_ADAPTERS, GROUP_BACKBONE_FROZEN, GROUP_EXPERIENTIAL  # noqa: E402
from cam_live.nn.gpu import free_gpu_memory, gpu_lock  # noqa: E402
from cam_live.nn.model import build_model  # noqa: E402

CLAIMED = {"frozen": 4022468096, "lora": 11796480, "bridge": 4917376, "experiential": 50953765}
WORDS = ("the page shows a form with fields for name email and a submit button that is disabled until the terms are "
         "accepted then a confirmation message appears and the counter of members increases by one after a delay of "
         "several minutes while the server records the request in its log file and sends an email").split()


def text_of(model, n_tokens, seed):
    """A text whose backbone tokenisation has exactly n_tokens tokens (grown word by word, then trimmed)."""
    bb = model.backbone
    rng = random.Random(seed)
    words = []
    while len(bb._encode(" ".join(words))) < n_tokens:
        words.append(rng.choice(WORDS))
    ids = bb._encode(" ".join(words))[:n_tokens]
    t = bb._decode(ids)
    return t, len(bb._encode(t))


TINY = "--tiny" in sys.argv            # dry run of the probe's own code on the CPU test backbone (no claims checked)
CPU = "--cpu" in sys.argv              # the real Qwen3-4B configuration on the CPU (when the shared GPU is unavailable):
                                       # counts, bridge no-op, cache coherence, z sensitivity; D8 (a GPU-kernel
                                       # property) is skipped
t_wait = time.time()
import contextlib  # noqa: E402
with (contextlib.nullcontext(time.time()) if (TINY or CPU) else gpu_lock(timeout_s=7200.0, min_free_gb=16.0, poll_s=0.2)) as t_lock:
    info("waited for the GPU lock (s)", round(t_lock - t_wait, 1))
    t0 = time.time()
    cfg = CamLiveConfig.debug() if TINY else CamLiveConfig(device="cpu" if CPU else "cuda")
    if CPU:
        torch.set_num_threads(16)
    model = build_model(cfg)
    if not (TINY or CPU):
        torch.cuda.synchronize()
    info("load (s)", round(time.time() - t0, 1))
    bb = model.backbone
    try:
        # ---------------------------------------------------------------- parameter counts
        g = model.param_groups()
        frozen = sum(p.numel() for p in g[GROUP_BACKBONE_FROZEN])
        lora_ids = {id(p) for p in bb.lora_parameters()}
        lora = sum(p.numel() for p in g[GROUP_ADAPTERS] if id(p) in lora_ids)
        bridge = sum(p.numel() for p in g[GROUP_ADAPTERS] if id(p) not in lora_ids)
        exp = sum(p.numel() for p in g[GROUP_EXPERIENTIAL])
        all_p = {id(p) for p in model.parameters()}
        grouped = [id(p) for ps in g.values() for p in ps]
        got = {"frozen": frozen, "lora": lora, "bridge": bridge, "experiential": exp}
        if TINY:
            info("parameter counts (tiny)", got)
        else:
            check("parameter counts recomputed from the module tree equal the claimed ones", got == CLAIMED, f"{got}")
        check("parameter groups are disjoint and cover every parameter", len(grouped) == len(set(grouped)) and
              set(grouped) == all_p, f"{len(grouped)} grouped, {len(all_p)} parameters")
        fr = g[GROUP_BACKBONE_FROZEN]
        check("frozen backbone: no parameter requires grad, bf16 (CPU runs load float32: dtype not judged there)", all(not p.requires_grad for p in fr) and
              (TINY or CPU or all(p.dtype == torch.bfloat16 for p in fr)), f"{sum(p.requires_grad for p in fr)} requiring grad; dtypes "
              f"{sorted({str(p.dtype) for p in fr})}")
        info("LoRA modules / rank / targets", (len(bb.lora_modules), cfg.backbone.lora_rank, list(cfg.backbone.lora_targets)))
        check("LoRA adapters alone are within 10-20 M and the adapter group within 10-20 M", TINY or (10e6 <= lora <= 20e6
              and 10e6 <= lora + bridge <= 20e6), f"lora {lora:,}, adapters {lora + bridge:,}")

        # ---------------------------------------------------------------- bridge no-op at init
        prompts = [bb.chat_prompt("Name one colour."), bb.chat_prompt("What is 17 + 25?")]
        with torch.no_grad():
            l0 = bb.next_token_logits(prompts)
            same = True
            for scale in (1e-3, 1.0, 1e3):
                lat = torch.randn(2, bb.bridge.d_latent if hasattr(bb.bridge, "d_latent") else model.core_cfg.d_model,
                                  device=model.device) * scale
                l1 = bb.next_token_logits(prompts, latent=lat)
                same = same and torch.equal(l0, l1)
        check("latent bridge is an exact no-op at initialisation (logits bit-identical for latent norms 1e-3..1e3)", same)
        # one Adam step on the bridge parameters only
        bpar = [p for p in g[GROUP_ADAPTERS] if id(p) not in lora_ids]
        saved = [p.detach().clone() for p in bpar]
        opt = torch.optim.Adam(bpar, lr=1e-3)
        lat = torch.randn(1, model.core_cfg.d_model, device=model.device)
        lp = bb.generation_logprob(prompts[:1], [" Blue."], latent=lat)
        (-lp.sum()).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            pa = torch.softmax(bb.next_token_logits(prompts[:1], latent=lat), -1)
            pb = torch.softmax(bb.next_token_logits(prompts[:1], latent=-lat), -1)
            tv = float(0.5 * (pa - pb).abs().sum())
        check("after one Adam step on the bridge alone the next-token distribution depends on the latent (TV > 0.01)",
              tv > 0.01, f"TV(latent, -latent) = {tv:.4f}")
        with torch.no_grad():
            for p, s in zip(bpar, saved):
                p.copy_(s)
        del opt

        # ---------------------------------------------------------------- D8: batch-composition invariance
        if not CPU:
            lens = [1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 255, 256, 257, 300]
            maxt = int(cfg.backbone.max_text_tokens)
            lens.append(maxt + 50)
            texts = []
            for i, n in enumerate(lens):
                t, n_real = text_of(model, n, i)
                texts.append(t)
            info("token lengths of the probe texts", [len(bb._encode(t)) for t in texts])
            with torch.no_grad():
                bb._cache.clear()
                alone = {t: bb.embed([t]).clone() for t in texts}
                worst, n_cmp = 0.0, 0
                rng = random.Random(0)
                for trial in range(6):
                    bb._cache.clear()
                    comp = texts[:]
                    rng.shuffle(comp)
                    # extra companions of the same bucket as the first text: more rows than one forward holds
                    b0 = bb.embed_bucket(len(bb._text_ids(comp[0], maxt)))
                    n_rows = bb.embed_rows(b0)
                    extra = [comp[0] + f" variant {k}" for k in range(n_rows + 3)] if trial % 2 == 0 else []
                    call = comp + extra
                    rng.shuffle(call)
                    out = bb.embed(call)
                    for j, t in enumerate(call):
                        if t in alone:
                            d = float((out[j] - alone[t][0]).abs().max())
                            worst = max(worst, d)
                            n_cmp += 1
                bb._cache.clear()
                canon = bb.embed(texts).clone()
            with torch.enable_grad():
                bb._cache.clear()
                wg = bb.embed(texts, with_grad=True)
            check("D8: every text's embedding is bit-identical alone and in any call (bucket edges, > max_text_tokens, "
                  "more rows than one forward)", worst == 0.0, f"max |diff| {worst:.3g} over {n_cmp} comparisons")
            check("D8: embed(with_grad=True) returns the canonical value bit-exactly and carries a graph",
                  torch.equal(wg.detach(), canon) and wg.requires_grad, f"max |diff| {float((wg.detach() - canon).abs().max()):.3g}")
            del wg
            # belief: one context alone vs inside a batch of contexts (cold cache both times)
            ctx = [ExperienceItem(kind=ItemKind.OBSERVATION, text=texts[5], ts=10.0, quantities=[Quantity("fields", 3.0)]),
                   ExperienceItem(kind=ItemKind.OUTCOME, text=texts[9], ts=900.0, quantities=[Quantity("members", 7.0)])]
            others = [[ExperienceItem(kind=ItemKind.DOCUMENT, text=texts[k], ts=5.0)] for k in (10, 12, 14)]
            with torch.no_grad():
                bb._cache.clear()
                b1 = model.core_forward(model.tokenize([ctx])).b[0].clone()
                bb._cache.clear()
                b2 = model.core_forward(model.tokenize(others + [ctx])).b[-1].clone()
            relb = float((b1 - b2).norm() / b1.norm())
            check("D8: belief b of a context alone vs inside a padded batch (cold cache) agrees to 1e-5 relative (the "
                  "embeddings are bit-exact; the float32 core reduces padded batches in another order)", relb < 1e-5,
                  f"relative difference {relb:.3g}, max |diff| {float((b1 - b2).abs().max()):.3g}")
        else:                                   # CPU: short texts only (D8 is a property of the GPU kernels)
            texts = [f"{' '.join(WORDS[i:i + 6])} ({i})" for i in range(15)]
            ctx = [ExperienceItem(kind=ItemKind.OBSERVATION, text=texts[5], ts=10.0, quantities=[Quantity("fields", 3.0)]),
                   ExperienceItem(kind=ItemKind.OUTCOME, text=texts[9], ts=900.0, quantities=[Quantity("members", 7.0)])]
        # ---------------------------------------------------------------- D10: caches follow the adapter content
        probe_t = texts[3:6]
        with torch.no_grad():
            e_before = bb.embed(probe_t).clone()
            lg_before = bb.next_token_logits(prompts[:1]).clone()
            lmods = list(bb.lora_modules)
            old_B = [m.lora_B.data for m in lmods]
            gen = torch.Generator(device="cpu").manual_seed(5)
            for m in lmods:                    # replace the storage: new data_ptr and a fresh version counter
                m.lora_B.data = (torch.randn(m.lora_B.shape, generator=gen) * 1e-2).to(m.lora_B.device)
            e_after = bb.embed(probe_t).clone()
            lg_after = bb.next_token_logits(prompts[:1]).clone()
            bb._cache.clear()
            for m in lmods:
                m.invalidate()
            e_cold = bb.embed(probe_t).clone()
            lg_cold = bb.next_token_logits(prompts[:1]).clone()
        check("D10: after `lora_B.data = new tensor` the embedding equals a cold recomputation (and differs from before)",
              torch.equal(e_after, e_cold) and not torch.equal(e_after, e_before),
              f"|after - cold| {float((e_after - e_cold).abs().max()):.3g}, |after - before| {float((e_after - e_before).abs().max()):.3g}")
        check("D10: after `lora_B.data = new tensor` the no-grad logits equal a cold recomputation",
              torch.equal(lg_after, lg_cold) and not torch.equal(lg_after, lg_before),
              f"|after - cold| {float((lg_after - lg_cold).abs().max()):.3g}")
        # load_trainable_state back to zero-B adapters
        state = model.trainable_state()
        for m, ob in zip(lmods, old_B):
            k = next(n for n, p in model.named_parameters() if p is m.lora_B)
            state[k] = ob.detach().cpu().clone()
        with torch.no_grad():
            model.load_trainable_state(state, strict=True)
            e_back = bb.embed(probe_t).clone()
            lg_back = bb.next_token_logits(prompts[:1]).clone()
        check("D10: after load_trainable_state the embedding and logits return bit-exactly to the original ones",
              torch.equal(e_back, e_before) and torch.equal(lg_back, lg_before),
              f"|emb| {float((e_back - e_before).abs().max()):.3g} |logits| {float((lg_back - lg_before).abs().max()):.3g}")

        # ---------------------------------------------------------------- D7: one argument moves z
        acts = [ActionSpec("a1", "browser.click", arguments={"ref": "e3"}), ActionSpec("a2", "browser.click", arguments={"ref": "e4"}),
                ActionSpec("a3", "api.call", modality="api", arguments={"endpoint": "/export", "rows": 100}),
                ActionSpec("a4", "api.call", modality="api", arguments={"endpoint": "/export", "rows": 101})]
        with torch.no_grad():
            o = model.core_forward(model.tokenize([ctx]))
            e_a, _ = model.encode_actions([acts])
            e_g = model.encode_goals(["Reach the stated target within the time limit."])
            z = model.decision_latent(o.b, e_a, e_g)[0]
        c12 = float(Fn.cosine_similarity(z[0], z[1], 0))
        c34 = float(Fn.cosine_similarity(z[2], z[3], 0))
        info("cos(z): click e3 vs e4 / api rows 100 vs 101", (round(c12, 4), round(c34, 4)))
        check("D7: a candidate that differs only in one argument has a different decision latent (cos(z) < 0.99)",
              c12 < 0.99, f"cos {c12:.4f} (rows 100 vs 101: {c34:.4f})")
    finally:
        if not (TINY or CPU):
            info("peak GPU memory (GB)", round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
        del model
        if not (TINY or CPU):
            free_gpu_memory()
        info("GPU lock held (s)", round(time.time() - t_lock, 1))
sys.exit(finish())
