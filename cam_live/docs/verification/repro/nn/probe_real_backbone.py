"""G-nn probe on the REAL configuration (Qwen/Qwen3-4B + LoRA + latent bridge + the 50M experiential network).

    .venv/bin/python cam_live/docs/verification/repro/nn/probe_real_backbone.py --part params|grads|bridge|values|throughput|misc|phi|cond
    .venv/bin/python cam_live/docs/verification/repro/nn/probe_real_backbone.py --part save --dir DIR
    .venv/bin/python cam_live/docs/verification/repro/nn/probe_real_backbone.py --part load --dir DIR     (fresh process)
    .venv/bin/python cam_live/docs/verification/repro/nn/probe_real_backbone.py --part gradsave --dir DIR  (bf16 gradients / log-probs)
    .venv/bin/python cam_live/docs/verification/repro/nn/probe_real_backbone.py --part fd32 --dir DIR      (backbone in float32: finite differences)

Each part takes the shared GPU lock through ``cam_live.nn.gpu.gpu_lock`` for its own duration only and frees the
memory before releasing it. All weight changes are in memory only.
"""
import math
import os
import sys
import time

import _common  # noqa: F401
from _common import check, finish, info

import torch
import torch.nn.functional as Fn

from cam_live.config import CamLiveConfig
from cam_live.contracts import ActionSpec, ExperienceItem, ItemKind, OutcomeKind, OutcomeQuery, Quantity
from cam_live.nn.api import (GROUP_ADAPTERS, GROUP_BACKBONE_FROZEN, GROUP_EXPERIENTIAL, GROUP_HEADS, GROUP_REWARD, GROUPS)
from cam_live.nn.gpu import free_gpu_memory, gpu_lock
from cam_live.nn.model import build_model

GB = 2 ** 30
part = sys.argv[sys.argv.index("--part") + 1]
DIR = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else ""


def contexts():
    return [
        [ExperienceItem(kind=ItemKind.OBSERVATION, text="The form was submitted and a confirmation appeared.", ts=10.0,
                        quantities=[Quantity("visits", 100.0)]),
         ExperienceItem(kind=ItemKind.ACTION, text="browser.click on the element labelled Export", ts=40.0),
         ExperienceItem(kind=ItemKind.OUTCOME, text="three sign-ups were recorded", ts=3600.0, quantities=[Quantity("signups", 3.0)])],
        [ExperienceItem(kind=ItemKind.DOCUMENT, text="Section 2 of the manual explains how the export job is scheduled.", ts=5e4),
         ExperienceItem(kind=ItemKind.FAILURE, text="timeout transient_network: the request did not complete", ts=5.8e4,
                        quantities=[Quantity("result.duration", 30.0, unit="s")])],
    ]


ACTS = [[ActionSpec("a1", "browser.click", arguments={"ref": "e3"}), ActionSpec("a2", "sandbox.python", modality="code", arguments={"code": "print(1)"}),
         ActionSpec("a3", "internal.wait", arguments={"seconds": 600})],
        [ActionSpec("b1", "api.call", modality="api", arguments={"x": 2})]]
GOALS = ["Reach the stated target within the time limit.", "Find out why the export job fails."]
QUERIES = [[OutcomeQuery("q1", "signups", kind=OutcomeKind.COUNT, horizon_s=86400.0, description="new sign-ups after one day")],
           [OutcomeQuery("q2", "latency", horizon_s=30.0, description="seconds until the job is done")]]


def pipeline(model, grad_sem=False, b_override=None):
    tb = model.tokenize(contexts(), with_grad=grad_sem)
    out = model.core_forward(tb)
    e_a, a_mask = model.encode_actions(ACTS, with_grad=grad_sem)
    e_g = model.encode_goals(GOALS, with_grad=grad_sem)
    e_q, q_mask, _, _ = model.encode_queries(QUERIES)
    z = model.decision_latent(out.b if b_override is None else b_override, e_a, e_g)
    phi = model.features(z, e_q)
    return tb, out, e_a, a_mask, e_g, e_q, z, phi


def probe_w(shape, device, seed):
    return torch.randn(shape, device=device, generator=torch.Generator(device).manual_seed(seed))


def gnorm(params):
    sq = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
    return float(torch.stack(sq).sum().sqrt()) if sq else 0.0


def frozen_ok(model):
    fr = model.param_groups()[GROUP_BACKBONE_FROZEN]
    return all((not p.requires_grad) and p.grad is None for p in fr)


# ====================================================================================================================
def part_params(model):
    bb = model.backbone
    c = model.count_parameters()
    info("count_parameters", {k: c[k] for k in ("frozen", "experiential", "adapters", "adapters_lora", "adapters_bridge", "reward",
                                               "heads_total", "total_persistent_trainable")})
    # independent counts straight from the module tree
    all_params = list(model.parameters())
    n_all = sum(p.numel() for p in all_params)
    lm_named = list(bb.lm.named_parameters())
    n_lm_frozen = sum(p.numel() for n, p in lm_named if "lora_" not in n)
    n_lora = sum(p.numel() for n, p in lm_named if "lora_" in n)
    n_bridge = sum(p.numel() for p in bb.bridge.parameters())
    n_exp = sum(p.numel() for p in model.exp.parameters())
    check("frozen backbone = 4,022,468,096 parameters (independent count over backbone.lm, LoRA excluded)", n_lm_frozen == 4_022_468_096 == c["frozen"],
          f"{n_lm_frozen:,}")
    check("every frozen backbone tensor has requires_grad=False and is bfloat16",
          all((not p.requires_grad) and p.dtype == torch.bfloat16 for n, p in lm_named if "lora_" not in n))
    check("LoRA = 11,796,480 (36 layers x (q_proj 2560->4096 + v_proj 2560->1024) x rank 32), float32, trainable",
          n_lora == 11_796_480 == c["adapters_lora"] == 36 * 32 * ((2560 + 4096) + (2560 + 1024))
          and all(p.requires_grad and p.dtype == torch.float32 for n, p in lm_named if "lora_" in n), f"{n_lora:,}")
    check("adapters (LoRA + bridge) within 10-20 M", 10e6 <= n_lora + n_bridge <= 20e6 and n_lora + n_bridge == c["adapters"],
          f"LoRA {n_lora:,} + bridge {n_bridge:,} = {n_lora + n_bridge:,}")
    check("LoRA alone within 10-20 M", 10e6 <= n_lora <= 20e6, f"{n_lora:,}")
    check("experiential group ~50 M (45-55 M)", 45e6 < n_exp < 55e6 and n_exp == c["experiential"], f"{n_exp:,}")
    check("persistent trainable total 60-70 M", 60e6 < c["total_persistent_trainable"] < 70e6, f"{c['total_persistent_trainable']:,}")
    check("frozen + trainable == every parameter of the module tree", c["frozen"] + c["total_persistent_trainable"] == n_all, f"{n_all:,}")
    groups = model.param_groups()
    ids = [id(p) for ps in groups.values() for p in ps]
    check("param_groups(): keys are exactly nn.api.GROUPS, groups disjoint, union == model.parameters()",
          tuple(groups) == GROUPS and len(ids) == len(set(ids)) and set(ids) == {id(p) for p in all_params})
    trainable_ids = {id(p) for g in (GROUP_ADAPTERS, GROUP_EXPERIENTIAL, GROUP_REWARD, GROUP_HEADS) for p in groups[g]}
    check("every requires_grad=True tensor is in a trainable group and vice versa",
          {id(p) for p in all_params if p.requires_grad} == trainable_ids)
    check("status-file numbers: experiential 50,953,765 / adapters 16,713,856 / bridge 4,917,376 / total 67,667,621",
          (c["experiential"], c["adapters"], c["adapters_bridge"], c["total_persistent_trainable"]) == (50_953_765, 16_713_856, 4_917_376, 67_667_621),
          str((c["experiential"], c["adapters"], c["adapters_bridge"], c["total_persistent_trainable"])))
    info("experiential breakdown", c["experiential_breakdown"])
    check("status-file experiential breakdown", c["experiential_breakdown"] == {"tokenizer": 4263746, "core": 36615648, "encoders": 1977600,
                                                                                "trunk": 4720128, "feature_map": 3343104, "fast_prior": 33539})
    # registered heads land in their group
    import torch.nn as nn
    model.register_head("g_probe_world", nn.Linear(model.d_model, 7))
    model.register_head("g_probe_reward", nn.Linear(model.d_model, 3), group=GROUP_REWARD)
    c2 = model.count_parameters()
    check("registered heads are counted per group (heads / reward)", c2["heads_total"] == 7 * 768 + 7 and c2["reward"] == 3 * 768 + 3
          and c2["total_persistent_trainable"] == c["total_persistent_trainable"] + 7 * 768 + 7 + 3 * 768 + 3)
    # checkpoint payload
    st = model.trainable_state()
    n_state = sum(v.numel() for v in st.values())
    pnames = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen_names = {n for n, p in model.named_parameters() if not p.requires_grad}
    check("trainable_state(): no frozen backbone tensor, every trainable tensor present",
          not (set(st) & frozen_names) and pnames <= set(st), f"{len(st)} tensors, {n_state:,} numbers")
    extra = sorted(set(st) - pnames)
    info("non-parameter tensors in trainable_state (buffers)", extra)
    nbytes = sum(v.numel() * v.element_size() for v in st.values())
    check("trainable_state() size is ~0.27 GB (the 8 GB frozen backbone is not in it)", nbytes < 0.4 * GB, f"{nbytes / GB:.3f} GB")
    check("trainable_state() tensors are detached CPU copies", all((not v.requires_grad) and v.device.type == "cpu" for v in st.values()))
    info("gpu memory allocated after load (GB)", round(torch.cuda.memory_allocated() / GB, 3))
    info("arch signature backbone", model.arch_signature()["backbone"])
    sig = model.arch_signature()
    info("arch signature does not record", [k for k in ("max_text_tokens", "tap_layers", "seed") if k not in str(sig)])


# ====================================================================================================================
def part_grads(model):
    bb = model.backbone
    dev = model.device
    lora = bb.lora_modules
    exp_params = list(model.exp.parameters())
    bridge_params = list(bb.bridge.parameters())
    lora_B = [m.lora_B for m in lora]
    lora_A = [m.lora_A for m in lora]

    # ---- (1) loss on z and on phi through tokenize / encode_actions / encode_goals with_grad=True --------------------------
    for name in ("z", "phi"):
        model.zero_grad(set_to_none=True)
        tb, out, e_a, a_mask, e_g, e_q, z, phi = pipeline(model, grad_sem=True)
        check(f"[{name}] tokenize(with_grad=True).x has a graph", tb.x.requires_grad and tb.x.grad_fn is not None)
        t = z if name == "z" else phi
        (t * probe_w(t.shape, dev, 3)).sum().backward()
        g_exp, g_B, g_A, g_br = gnorm(exp_params), gnorm(lora_B), gnorm(lora_A), gnorm(bridge_params)
        n_live = sum(float(m.lora_B.grad.abs().sum()) > 0 for m in lora)
        check(f"[{name}] gradient reaches the experiential weights (finite, non-zero)", 0 < g_exp < float("inf") and math.isfinite(g_exp), f"|g|={g_exp:.4g}")
        check(f"[{name}] gradient reaches all 72 LoRA modules (lora_B; finite, non-zero)", n_live == 72 and math.isfinite(g_B) and g_B > 0,
              f"{n_live}/72, |g_B|={g_B:.4g}, |g_A|={g_A:.4g} (A gets none while B == 0)")
        info(f"[{name}] bridge gradient norm (the bridge is not on this path)", g_br)
        check(f"[{name}] frozen backbone untouched (no .grad, requires_grad False)", frozen_ok(model))
        if name == "phi":
            dead = [n for n, p in model.exp.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
            info("[phi] experiential tensors with zero gradient from a loss on phi", dead)
    # per-path: which semantic inputs carry adapter gradient
    for label, kw in (("context items", dict(ctx=True, act=False, goal=False)), ("action texts", dict(ctx=False, act=True, goal=False)),
                      ("goal text", dict(ctx=False, act=False, goal=True))):
        model.zero_grad(set_to_none=True)
        tb = model.tokenize(contexts(), with_grad=kw["ctx"])
        out = model.core_forward(tb)
        e_a, _ = model.encode_actions(ACTS, with_grad=kw["act"])
        e_g = model.encode_goals(GOALS, with_grad=kw["goal"])
        z = model.decision_latent(out.b, e_a, e_g)
        (z * probe_w(z.shape, dev, 3)).sum().backward()
        check(f"adapter gradient through {label} alone", gnorm(lora_B) > 0, f"|g_B|={gnorm(lora_B):.4g}")
    # with_grad=False builds no adapter graph (and embed_texts default is detached)
    model.zero_grad(set_to_none=True)
    tb, out, e_a, a_mask, e_g, e_q, z, phi = pipeline(model, grad_sem=False)
    (phi * probe_w(phi.shape, dev, 3)).sum().backward()
    check("with_grad=False: no adapter gradient, experiential gradient present", gnorm(lora_B) == 0.0 and gnorm(exp_params) > 0)
    g_ref = {id(p): p.grad.detach().clone() for p in exp_params if p.grad is not None}
    e = model.embed_texts(["a text"], with_grad=True)
    check("embed_texts(with_grad=True) returns a tensor with a graph; default call is detached",
          e.requires_grad and e.grad_fn is not None and not model.embed_texts(["a text"]).requires_grad)
    # grad_last=1: only the newest item carries an adapter graph
    model.zero_grad(set_to_none=True)
    tb_all = model.tokenize(contexts(), with_grad=True)
    tb_last = model.tokenize(contexts(), with_grad=True, grad_last=1)
    d = float((tb_all.x - tb_last.x).abs().max())
    info("max |x(with_grad=True) - x(with_grad=True, grad_last=1)| (float16 embedding cache vs float32 recomputation)", d)
    tb_ng = model.tokenize(contexts())
    info("max |x(with_grad=True) - x(with_grad=False)|", float((tb_all.x - tb_ng.x).abs().max()))
    info("relative |x(with_grad=True) - x(with_grad=False)| / |x|", float((tb_all.x - tb_ng.x).norm() / tb_ng.x.norm()))

    # ---- (2) finite-difference check of the experiential gradient (float32 path; cached embeddings) --------------------------
    with torch.no_grad():
        direction = [g_ref.get(id(p), torch.zeros_like(p)) for p in exp_params]
        gn = math.sqrt(sum(float(d_.double().pow(2).sum()) for d_ in direction))

        def loss_phi():
            *_, phi_ = pipeline(model, grad_sem=False)
            return float((phi_.double() * probe_w(phi_.shape, dev, 3).double()).sum())

        ratios = {}
        for eps in (1e-3, 3e-3, 1e-2):
            for p, d_ in zip(exp_params, direction):
                p.add_(d_, alpha=eps / gn)
            lp = loss_phi()
            for p, d_ in zip(exp_params, direction):
                p.add_(d_, alpha=-2 * eps / gn)
            lm = loss_phi()
            for p, d_ in zip(exp_params, direction):
                p.add_(d_, alpha=eps / gn)
            ratios[eps] = (lp - lm) / (2 * eps) / gn
    check("finite differences: directional derivative of a phi loss along the experiential gradient == |g| (5 %)",
          sum(abs(r - 1) < 0.05 for r in ratios.values()) >= 2, f"FD/|g| for eps 1e-3,3e-3,1e-2: {[round(r, 4) for r in ratios.values()]}, |g|={gn:.4g}")

    # ---- (3) embedding path: finite-difference check of the LoRA gradient on the real backbone ----------------------------------
    texts = ["The form was submitted and a confirmation appeared.", "def mean(xs):\n    return sum(xs) / len(xs)",
             "HTTP 404 Not Found", "three sign-ups were recorded after the announcement"]

    def emb_loss(with_graph):
        e_ = model.embed_texts(texts, with_grad=True)
        w_ = probe_w(e_.shape, dev, 5)
        l_ = (e_ * w_).sum() / 100.0
        return l_ if with_graph else float(l_.detach().double())

    model.zero_grad(set_to_none=True)
    emb_loss(True).backward()
    gB = [m.lora_B.grad.detach().clone() for m in lora]
    gBn = math.sqrt(sum(float(g_.double().pow(2).sum()) for g_ in gB))
    check("embedding path: all 72 lora_B get finite non-zero gradient", all(torch.isfinite(g_).all() and float(g_.abs().sum()) > 0 for g_ in gB),
          f"|g_B|={gBn:.4g}")
    check("embedding path: frozen backbone untouched", frozen_ok(model))
    with torch.no_grad():
        l0 = emb_loss(False)
        ratios = {}
        for dl in (0.05, 0.15, 0.5, 1.5):                   # target change of the loss (absolute)
            eps = dl / gBn
            for m, g_ in zip(lora, gB):
                m.lora_B.add_(g_, alpha=eps / gBn)
            lp = emb_loss(False)
            for m, g_ in zip(lora, gB):
                m.lora_B.add_(g_, alpha=-2 * eps / gBn)
            lm = emb_loss(False)
            for m, g_ in zip(lora, gB):
                m.lora_B.add_(g_, alpha=eps / gBn)
            ratios[dl] = (lp - lm) / (2 * eps) / gBn
        for m in lora:
            m.lora_B.zero_()
    bb.mark_adapters_updated()
    info("bf16 finite differences (NOT decisive: bf16 rounding of the forward pass; see --part fd32): embedding loss along the LoRA-B gradient, "
         "FD/|g| for dL 0.05,0.15,0.5,1.5", [round(r, 3) for r in ratios.values()])

    # ---- (4) generation log-probability: LoRA + bridge gradient, finite differences, checkpointing on / off -------------------
    prompts = [bb.chat_prompt("Name one colour."), bb.chat_prompt("Which tool should be used next? Answer with one word.")]
    comps = ["Green", "browser"]
    model.zero_grad(set_to_none=True)
    tb, out, *_ = pipeline(model, grad_sem=False)
    b = out.b                                                     # CAM's belief, with a graph into the experiential core
    lp = model.generation_logprob(prompts, comps, latent=b)
    with torch.no_grad():
        lp_ng = model.generation_logprob(prompts, comps, latent=b)
    info("generation_logprob", [round(float(v), 4) for v in lp])
    check("generation_logprob with graph (checkpointing, LM transiently in train mode) == value under no_grad", bool(torch.equal(lp.detach(), lp_ng)),
          f"max diff={float((lp.detach() - lp_ng).abs().max()):.3e}")
    (-lp.sum()).backward()
    g_out = [x.out.weight.grad.detach().clone() for x in bb.bridge.xattn]
    g_outn = math.sqrt(sum(float(g_.double().pow(2).sum()) for g_ in g_out))
    gB = [m.lora_B.grad.detach().clone() for m in lora]
    gBn = math.sqrt(sum(float(g_.double().pow(2).sum()) for g_ in gB))
    check("generation path: all 72 lora_B get finite non-zero gradient", all(torch.isfinite(g_).all() and float(g_.abs().sum()) > 0 for g_ in gB), f"|g_B|={gBn:.4g}")
    check("generation path: every zero-initialised bridge output projection gets finite non-zero gradient at step 0",
          all(torch.isfinite(g_).all() and float(g_.abs().sum()) > 0 for g_ in g_out), f"|g_out|={g_outn:.4g} over {len(g_out)} hooked layers")
    other = [n for n, p in bb.bridge.named_parameters() if ".out." not in n and p.grad is not None and float(p.grad.abs().sum()) > 0]
    info("generation path at init: bridge tensors other than xattn.out with non-zero gradient (expected none while out == 0)", other)
    g_exp0 = gnorm(exp_params)
    info("generation path at init: gradient norm on the experiential weights through latent=b (expected exactly 0 while the bridge output is 0)", g_exp0)
    check("generation path: frozen backbone untouched", frozen_ok(model) and not bb.lm.training)

    def gen_loss(latent):
        with torch.no_grad():
            return float(-model.generation_logprob(prompts, comps, latent=latent).double().sum())

    with torch.no_grad():
        bd = b.detach()
        ratios_out, ratios_B = {}, {}
        for dl in (0.1, 0.3, 1.0):
            eps = dl / g_outn
            for x, g_ in zip(bb.bridge.xattn, g_out):
                x.out.weight.add_(g_, alpha=eps / g_outn)
            lp_ = gen_loss(bd)
            for x, g_ in zip(bb.bridge.xattn, g_out):
                x.out.weight.add_(g_, alpha=-2 * eps / g_outn)
            lm_ = gen_loss(bd)
            for x, g_ in zip(bb.bridge.xattn, g_out):
                x.out.weight.add_(g_, alpha=eps / g_outn)
            ratios_out[dl] = (lp_ - lm_) / (2 * eps) / g_outn
            eps = dl / gBn
            for m, g_ in zip(lora, gB):
                m.lora_B.add_(g_, alpha=eps / gBn)
            lp_ = gen_loss(bd)
            for m, g_ in zip(lora, gB):
                m.lora_B.add_(g_, alpha=-2 * eps / gBn)
            lm_ = gen_loss(bd)
            for m, g_ in zip(lora, gB):
                m.lora_B.add_(g_, alpha=eps / gBn)
            ratios_B[dl] = (lp_ - lm_) / (2 * eps) / gBn
        for m in lora:
            m.lora_B.zero_()
        for x in bb.bridge.xattn:
            x.out.weight.zero_()
    info("bf16 finite differences (NOT decisive: the bf16 LM head quantises logits to 1/8; see --part fd32): generation loss along the "
         "bridge-output / LoRA-B gradient, FD/|g| for dL 0.1,0.3,1.0", ([round(r, 3) for r in ratios_out.values()], [round(r, 3) for r in ratios_B.values()]))

    # gradient checkpointing on / off must give the same adapter gradients (magnitude as well as direction)
    with torch.no_grad():                                         # non-trivial adapters first
        gen = torch.Generator(dev).manual_seed(11)
        for m in lora:
            m.lora_B.copy_(torch.randn(m.lora_B.shape, device=dev, generator=gen) * 0.01)
        for x in bb.bridge.xattn:
            x.out.weight.copy_(torch.randn(x.out.weight.shape, device=dev, generator=gen) * 0.02)
    bb.mark_adapters_updated()
    grads = {}
    lat = probe_w((2, model.d_model), dev, 21).requires_grad_(True)
    for ck in (True, False):
        bb.cfg.grad_checkpoint = ck
        model.zero_grad(set_to_none=True)
        lat.grad = None
        loss = -model.generation_logprob(prompts, comps, latent=lat).sum() + emb_loss(True)
        loss.backward()
        grads[ck] = (torch.cat([p.grad.flatten().double() for p in model.param_groups()[GROUP_ADAPTERS]]), lat.grad.detach().double().clone())
        assert not bb.lm.training
    bb.cfg.grad_checkpoint = True
    ga, gb_ = grads[True][0], grads[False][0]
    cos = float(Fn.cosine_similarity(ga, gb_, 0))
    ratio = float(ga.norm() / gb_.norm())
    rel = float((ga - gb_).norm() / gb_.norm())
    check("gradient checkpointing on vs off: adapter gradients agree in direction AND magnitude (cos > 0.98, |norm ratio - 1| < 0.05)",
          cos > 0.98 and abs(ratio - 1) < 0.05, f"cos={cos:.5f} norm ratio={ratio:.5f} relative difference={rel:.4f}")
    la, lb = grads[True][1], grads[False][1]
    check("gradient w.r.t. the latent: finite, non-zero once the bridge output is non-zero; same with / without checkpointing",
          bool(torch.isfinite(la).all()) and float(la.norm()) > 0 and float(Fn.cosine_similarity(la.flatten(), lb.flatten(), 0)) > 0.98,
          f"|dL/dlatent|={float(la.norm()):.4g}, cos={float(Fn.cosine_similarity(la.flatten(), lb.flatten(), 0)):.5f}, ratio={float(la.norm() / lb.norm()):.4f}")
    # generation log-probability -> bridge -> belief b -> experiential weights
    model.zero_grad(set_to_none=True)
    tb, out, *_ = pipeline(model, grad_sem=False)
    lp = model.generation_logprob(prompts, comps, latent=out.b)
    (-lp.sum()).backward()
    g_exp = gnorm(exp_params)
    dead_core = [n for n, p in model.exp.core.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    check("generation log-probability -> bridge -> b -> experiential weights: finite non-zero gradient (bridge output non-zero)",
          math.isfinite(g_exp) and g_exp > 0, f"|g_exp|={g_exp:.4g}; core tensors without gradient: {dead_core}")
    check("... and the frozen backbone is still untouched", frozen_ok(model))
    # finite difference of that path along the experiential gradient
    with torch.no_grad():
        direction = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in exp_params]

        def gl():
            _, o_, *_ = pipeline(model, grad_sem=False)
            return gen_loss(o_.b)

        ratios = {}
        for dl in (0.05, 0.2, 0.5):
            eps = dl / g_exp
            for p, d_ in zip(exp_params, direction):
                p.add_(d_, alpha=eps / g_exp)
            lp_ = gl()
            for p, d_ in zip(exp_params, direction):
                p.add_(d_, alpha=-2 * eps / g_exp)
            lm_ = gl()
            for p, d_ in zip(exp_params, direction):
                p.add_(d_, alpha=eps / g_exp)
            ratios[dl] = (lp_ - lm_) / (2 * eps) / g_exp
    info("bf16 finite differences (NOT decisive; see --part fd32): generation loss along the experiential gradient, FD/|g| for dL 0.05,0.2,0.5",
         [round(r, 3) for r in ratios.values()])
    # does the experiential core change generation log-probabilities with the backbone fixed?
    with torch.no_grad():
        _, o1, *_ = pipeline(model, grad_sem=False)
        lp1 = model.generation_logprob(prompts, comps, latent=o1.b)
        saved = [p.detach().clone() for p in exp_params]
        gen = torch.Generator(dev).manual_seed(5)
        for p in exp_params:
            p.add_(torch.randn(p.shape, device=dev, generator=gen) * 0.05 * p.std().clamp_min(1e-3) if p.numel() > 1 else torch.zeros_like(p))
        _, o2, *_ = pipeline(model, grad_sem=False)
        lp2 = model.generation_logprob(prompts, comps, latent=o2.b)
        lp_none = model.generation_logprob(prompts, comps)
        for p, s_ in zip(exp_params, saved):
            p.copy_(s_)
    check("changing only experiential weights (5 % noise) changes generation log-probabilities through the bridge",
          float((lp1 - lp2).abs().max()) > 1e-3, f"logprob before {[round(float(v), 3) for v in lp1]} after {[round(float(v), 3) for v in lp2]} "
          f"(no latent: {[round(float(v), 3) for v in lp_none]}); db={float((o1.b - o2.b).norm() / o1.b.norm()):.3f}")


# ====================================================================================================================
def part_bridge(model):
    bb = model.backbone
    dev = model.device
    prompts = [bb.chat_prompt("Name one colour."), "The capital of France is", bb.chat_prompt("List three prime numbers."),
               "def fibonacci(n):\n"]
    gen = torch.Generator(dev).manual_seed(0)
    worst, equal = 0.0, True
    with torch.no_grad():
        base = bb.next_token_logits(prompts)
        for scale in (1.0, 1e3, 1e-3):
            lat = torch.randn(len(prompts), model.d_model, device=dev, generator=gen) * scale
            cond = bb.next_token_logits(prompts, latent=lat)
            worst = max(worst, float((base - cond).abs().max()))
            equal &= bool(torch.equal(base, cond))
        lat = torch.randn(len(prompts), model.d_model, device=dev, generator=gen)
        comps = ["Blue", " Paris", "2, 3, 5", "    if n < 2:\n        return n"]
        lp0, lp1 = model.generation_logprob(prompts, comps), model.generation_logprob(prompts, comps, latent=lat)
        g0 = model.generate(prompts, max_new_tokens=24, temperature=0.0)
        g1 = model.generate(prompts, latent=lat, max_new_tokens=24, temperature=0.0)
        s0 = model.generate(prompts, max_new_tokens=24, temperature=0.9, seed=3, num_return_sequences=2)
        s1 = model.generate(prompts, latent=lat, max_new_tokens=24, temperature=0.9, seed=3, num_return_sequences=2)
    check("bridge at init: next-token logits bit-identical with and without a latent (latent norms 1e-3 .. 1e3)", equal, f"max|dlogit|={worst}")
    check("bridge at init: generation_logprob bit-identical", bool(torch.equal(lp0, lp1)), str([round(float(v), 3) for v in lp0]))
    check("bridge at init: greedy and seeded sampled generations identical (so no text can leak from the bridge)", g0 == g1 and s0 == s1,
          repr(g0[0][0][:60]))
    info("greedy generations of the unconditioned backbone", [t[0][:50] for t in g0])
    check("bridge initialisation contains no text-derived embeddings (to_mem / pos / q / k / v random, out == 0)",
          all(float(x.out.weight.abs().max()) == 0.0 for x in bb.bridge.xattn))
    info("bridge buffers derived from the calibration corpus (statistics only)", [n for n, _ in bb.bridge.named_buffers()])

    # one optimizer step on the adapters
    p1 = [prompts[0]]
    lat1 = torch.randn(1, model.d_model, device=dev, generator=gen)
    with torch.no_grad():
        logit_before = bb.next_token_logits(p1, latent=lat1)
        text_before = model.generate(p1, latent=lat1, max_new_tokens=24, temperature=0.0)
        samp_before = model.generate(p1, latent=lat1, max_new_tokens=24, temperature=1.0, seed=7, num_return_sequences=4)
    opt = torch.optim.Adam(model.param_groups()[GROUP_ADAPTERS], lr=1e-3)
    model.zero_grad(set_to_none=True)
    lp_a = model.generation_logprob(p1, ["Green"], latent=lat1)
    (-lp_a.sum()).backward()
    g_out = math.sqrt(sum(float(x.out.weight.grad.double().pow(2).sum()) for x in bb.bridge.xattn))
    check("bridge output projections get non-zero gradient at step 0", g_out > 0 and math.isfinite(g_out), f"|g|={g_out:.4g}")
    opt.step()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        logit_after = bb.next_token_logits(p1, latent=lat1)
        logit_after_none = bb.next_token_logits(p1)
        logit_after_other = bb.next_token_logits(p1, latent=-lat1)
        text_after = model.generate(p1, latent=lat1, max_new_tokens=24, temperature=0.0)
        samp_after = model.generate(p1, latent=lat1, max_new_tokens=24, temperature=1.0, seed=7, num_return_sequences=4)
        lp_b = model.generation_logprob(p1, ["Green"], latent=lat1)
    tv = lambda a, b_: float(0.5 * (torch.softmax(a, -1) - torch.softmax(b_, -1)).abs().sum())
    info("one Adam step (lr 1e-3) on LoRA + bridge", {"logprob(Green) before": round(float(lp_a), 4), "after": round(float(lp_b), 4),
                                                     "TV(next-token) after vs before": round(tv(logit_after, logit_before), 5),
                                                     "TV with latent vs without": round(tv(logit_after, logit_after_none), 5),
                                                     "TV latent vs -latent": round(tv(logit_after, logit_after_other), 5),
                                                     "greedy text changed": text_before != text_after,
                                                     "seeded samples changed": samp_before != samp_after})
    check("after one optimizer step the next-token distribution depends on the latent (TV(latent, -latent) > 1e-4)",
          tv(logit_after, logit_after_other) > 1e-4 and tv(logit_after, logit_after_none) > 1e-4)
    check("after one optimizer step the generated text changes (greedy or seeded sampling)", text_before != text_after or samp_before != samp_after,
          f"greedy before {text_before[0][0][:40]!r} after {text_after[0][0][:40]!r}")

    # can the bridge alone learn to steer generation by the latent? (fresh model state: reset adapters)
    with torch.no_grad():
        for m in bb.lora_modules:
            m.lora_B.zero_()
    bb.mark_adapters_updated()
    for p in bb.lora_parameters():
        p.requires_grad_(False)
    q = [bb.chat_prompt("Name one colour. Answer with one word.")]
    la, lb = torch.randn(1, model.d_model, device=dev, generator=gen), torch.randn(1, model.d_model, device=dev, generator=gen)
    with torch.no_grad():
        base_text = model.generate(q, max_new_tokens=6, temperature=0.0)[0][0]
        base_lp = model.generation_logprob(q + q, ["Green", "Purple"]).tolist()
    opt = torch.optim.Adam(bb.bridge.parameters(), lr=2e-3)
    curve = []
    for step in range(40):
        opt.zero_grad(set_to_none=True)
        lp_ = bb.generation_logprob(q + q, ["Green", "Purple"], latent=torch.cat([la, lb]))
        loss = -lp_.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bb.bridge.parameters(), 1.0)
        opt.step()
        curve.append(round(float(loss), 3))
    for p in bb.lora_parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        ta = model.generate(q, latent=la, max_new_tokens=6, temperature=0.0)[0][0]
        tb_ = model.generate(q, latent=lb, max_new_tokens=6, temperature=0.0)[0][0]
        tn = model.generate(q, max_new_tokens=6, temperature=0.0)[0][0]
        lp_after = bb.generation_logprob(q + q + q + q, ["Green", "Purple", "Purple", "Green"], latent=torch.cat([la, lb, la, lb])).tolist()
    info("bridge-only training, 40 Adam steps (two latents -> two different one-word answers)",
         {"loss curve (every 8th)": curve[::8] + curve[-1:], "unconditioned answer before": base_text, "unconditioned answer after": tn,
          "answer with latent A": ta, "answer with latent B": tb_,
          "logprob [A->Green, B->Purple, A->Purple, B->Green] after": [round(v, 3) for v in lp_after],
          "logprob [Green, Purple] without latent before": [round(v, 3) for v in base_lp]})
    check("a trained bridge steers the generated text by the latent (A -> 'Green', B -> 'Purple'), unconditioned output unchanged",
          "green" in ta.lower() and "purple" in tb_.lower() and tn == base_text, f"A: {ta!r}  B: {tb_!r}  none: {tn!r}")
    check("the frozen backbone stayed frozen and in eval mode", frozen_ok(model) and not bb.lm.training)


# ====================================================================================================================
def part_values(model):
    from cam_live.nn.bench import _passage, init_diagnostics
    bb = model.backbone
    dev = model.device
    d = init_diagnostics(model)
    claimed = {"token_cos_plus1_minus1": 0.18, "token_cos_100_103": 0.9749, "token_cos_same_value_two_metrics": 0.5591,
               "belief_cos_between_contexts": 0.1356, "z_cos_under_permuted_belief": 0.7213,
               "belief_relative_change_under_permutation": 1.3149, "z_relative_change_under_permuted_belief": 0.7466,
               "z_cos_between_two_actions": 0.8256, "action_embedding_cos_between_two_actions": 0.7628,
               "state_constant_fraction_after_one_call": 0.4006, "slot_gate_mean": 0.1242, "phi_norm": 0.999999}
    worst = max(abs(d[k] - v) for k, v in claimed.items())
    info("init diagnostics recomputed", d)
    check("status-file init diagnostics reproduce (max abs deviation < 0.02)", worst < 0.02, f"max deviation={worst:.4f}")

    # independent recomputation with my own texts / metrics
    long_texts = [_passage(n, bb, 100 + i) for i, n in enumerate((16, 64, 200, 380))]
    short = "ok"
    cos = lambda a, b_: float(Fn.cosine_similarity(a, b_, -1))

    def tok(text, v=None, name="members.count", unit="", kind=ItemKind.OUTCOME):
        qs = [] if v is None else [Quantity(name, v, unit=unit)]
        return ExperienceItem(kind=kind, text=text, ts=10.0, quantities=qs)

    with torch.no_grad():
        rows = {}
        for i, text in enumerate([short] + long_texts):
            x = model.tokenize([[tok(text, 1.0)], [tok(text, -1.0)], [tok(text, 100.0)], [tok(text, 103.0)], [tok(text, 3.0)], [tok(text, 4.0)],
                                [tok(text, 1000.0)], [tok(text, 1001.0)], [tok(text, 0.0)], [tok(text, None)], [tok(text, 1e12)], [tok(text, 1e15)],
                                [tok(text, -1e15)], [tok(text, 5.0, "visits")], [tok(text, 5.0, "signups")], [tok(text, 0.5)], [tok(text, 0.05)]]).x[:, 0]
            rows[len(bb._encode(text))] = {"+1/-1": cos(x[0], x[1]), "100/103": cos(x[2], x[3]), "3/4": cos(x[4], x[5]), "1000/1001": cos(x[6], x[7]),
                                           "0/missing": cos(x[8], x[9]), "1e12/1e15": cos(x[10], x[11]), "1e15/-1e15": cos(x[11], x[12]),
                                           "visits/signups": cos(x[13], x[14]), "0.5/0.05": cos(x[15], x[16])}
        for ntok, r in rows.items():
            info(f"token cosine by value pair, text of {ntok} tokens", {k: round(v, 4) for k, v in r.items()})
        check("value channel not swamped: cos(+1, -1) < 0.35 for every text length", max(r["+1/-1"] for r in rows.values()) < 0.35,
              f"max={max(r['+1/-1'] for r in rows.values()):.4f}")
        check("cos(+1e15, -1e15) < 0.35 (sign of very large values)", max(r["1e15/-1e15"] for r in rows.values()) < 0.35)
        check("equal metric, different value distinguishable: cos(3, 4) < 0.98 and cos(100, 103) < 0.999",
              max(r["3/4"] for r in rows.values()) < 0.98 and max(r["100/103"] for r in rows.values()) < 0.999)
        check("zero is distinguishable from a missing value: cos < 0.9", max(r["0/missing"] for r in rows.values()) < 0.9)
        check("1e12 vs 1e15 distinguishable: cos < 0.99", max(r["1e12/1e15"] for r in rows.values()) < 0.99,
              f"max={max(r['1e12/1e15'] for r in rows.values()):.4f}")
        info("1000 vs 1001 (0.1 % apart) token cosine: effectively identical in the embedding; exact values only in TokenBatch.values",
             round(max(r["1000/1001"] for r in rows.values()), 6))
        # channel norms
        tk = model.exp.tokenizer
        batch = model._to_device(tk.collate([[tok(long_texts[2], 42.0)]]))
        sem = model._semantic(batch, False)
        content = tk.content(sem, batch["has_text"])
        value = tk.value_channel(batch["values"], batch["value_mask"], batch["name_ids"], batch["unit_ids"], batch["qkind"], batch["exposure"])
        info("channel RMS for a 200-token text with one quantity", {"content": round(float(content.pow(2).mean().sqrt()), 4),
                                                                   "value": round(float(value.pow(2).mean().sqrt()), 4),
                                                                   "kind+trust": round(float((tk.kind_emb(batch["kind_ids"]) + tk.trust_emb(batch["trust_ids"])).pow(2).mean().sqrt()), 4)})
        # linear decodability of the value from the token and from the belief, across varied texts
        g = torch.Generator().manual_seed(0)
        n, n_tr = 2400, 1800
        vals = torch.sinh(torch.randn(n, generator=g, dtype=torch.float64) * 3.0)          # |v| from ~0 to ~1e4, both signs
        vals = torch.where(torch.rand(n, generator=g) < 0.5, vals.round(), vals)
        pool = [_passage(int(torch.randint(8, 120, (1,), generator=g)), bb, 5000 + i) for i in range(400)]
        pick = torch.randint(0, 400, (n, 2), generator=g).tolist()
        ctxs = [[ExperienceItem(kind=ItemKind.OBSERVATION, text=pool[pick[i][0]], ts=1.0), tok(pool[pick[i][1]], float(vals[i]))] for i in range(n)]
        X, Bf = [], []
        for s in range(0, n, 100):
            tb = model.tokenize(ctxs[s:s + 100])
            X.append(tb.x[:, 1].double().cpu())
            Bf.append(model.core_forward(tb).b.double().cpu())
        X, Bf = torch.cat(X), torch.cat(Bf)

        def fit(F_, target):
            F_ = torch.cat([F_, torch.ones(n, 1, dtype=torch.float64)], 1)
            tr, te = slice(0, n_tr), slice(n_tr, n)
            A = F_[tr].T @ F_[tr] + 1e-1 * torch.eye(F_.shape[1], dtype=torch.float64)
            w = torch.linalg.solve(A, F_[tr].T @ target[tr])
            return F_[te] @ w, target[te]

        def r2(F_, target):
            pred, t = fit(F_, target)
            return float(1 - (pred - t).pow(2).sum() / (t - t.mean()).pow(2).sum())

        def sign_acc(F_):
            pred, t = fit(F_, torch.sign(vals))
            keep = t != 0
            return float((torch.sign(pred[keep]) == t[keep]).double().mean())

        target = torch.asinh(vals)
        r_tok, r_b = r2(X, target), r2(Bf, target)
        s_tok, s_b = sign_acc(X), sign_acc(Bf)
        info("held-out linear probe (600 of 2400 random (text, value) pairs): R^2 of asinh(value) / sign accuracy",
             {"from token x": (round(r_tok, 4), round(s_tok, 4)), "from belief b (untrained core)": (round(r_b, 4), round(s_b, 4))})
        check("the value is linearly decodable from the token regardless of the text: R^2 > 0.8 and sign accuracy > 0.99", r_tok > 0.8 and s_tok > 0.99,
              f"R^2={r_tok:.4f} sign={s_tok:.4f}")
        check("... and survives the (untrained) core into the belief b: R^2 > 0.6 and sign accuracy > 0.95", r_b > 0.6 and s_b > 0.95,
              f"R^2={r_b:.4f} sign={s_b:.4f}")
        # control: the number appears only in the text (no Quantity) -> what an LM-embedding-only design would give
        ctxs_txt = [[tok(f"{pool[pick[i][1]]} value {float(vals[i]):.3f}", None)] for i in range(n)]
        Xt = torch.cat([model.tokenize(ctxs_txt[s:s + 100]).x[:, 0].double().cpu() for s in range(0, n, 100)])
        info("control: number only in the text (no Quantity): R^2 / sign accuracy from the token", (round(r2(Xt, target), 4), round(sign_acc(Xt), 4)))
        # exact values ride along
        exact = [0.1 + 0.2, 1e15 + 1.0, -123456789.123456789, 5e-324, 1.7976931348623157e308, 3.0]
        tb = model.tokenize([[ExperienceItem(kind=ItemKind.OUTCOME, text="x", ts=1.0, quantities=[Quantity("m", v)]) for v in exact]])
        check("TokenBatch.values carries the exact float64 values (incl. 1e15+1, denormal, max double)", tb.values[0, :, 0].tolist() == exact
              and tb.values.dtype == torch.float64 and bool(torch.isfinite(tb.x).all()), str(tb.values[0, :, 0].tolist()))
        # non-finite and overflow handling
        st0 = dict(tk.stats)
        tb = model.tokenize([[ExperienceItem(kind=ItemKind.OUTCOME, text="x", ts=1.0,
                                             quantities=[Quantity("a", float("nan")), Quantity("b", float("inf")), Quantity("c", 7.0)]
                                             + [Quantity(f"m{i}", float(i)) for i in range(6)])]])
        info("item with nan, inf and 7 finite quantities (V = 4): values kept", tb.values[0, 0].tolist())
        info("tokenizer.stats delta", {k: tk.stats[k] - st0[k] for k in st0})
        check("non-finite quantities are dropped and counted; finite tokens stay finite", bool(torch.isfinite(tb.x).all())
              and tk.stats["nonfinite_values"] - st0["nonfinite_values"] == 2)
        check("quantities beyond max_values (4) passed directly to tokenize are dropped but counted (stats['dropped_values'])",
              tk.stats["dropped_values"] - st0["dropped_values"] == 3, "to_items() chunks them into continuation items instead")
        from cam_live.nn.tokens import to_items
        its = to_items(ExperienceItem(kind=ItemKind.OUTCOME, text="x", ts=1.0, quantities=[Quantity(f"m{i}", float(i)) for i in range(9)]))
        check("to_items() keeps all 9 quantities across continuation items", sum(len(i.quantities) for i in its) == 9 and len(its) == 3)
    # embedding determinism: batch composition / cache
    with torch.no_grad():
        bb.mark_adapters_updated()
        t = "The export job finished after twelve minutes."
        e_alone = model.embed_texts([t]).clone()
        bb.mark_adapters_updated()
        e_batch = model.embed_texts([_passage(300, bb, 1), t, _passage(40, bb, 2)])[1].clone()
        rel = float((e_alone[0] - e_batch).norm() / e_alone[0].norm())
        info("semantic embedding of one text: relative difference alone vs inside a batch with longer texts (bf16; the first-seen "
             "version is what the cache keeps)", round(rel, 5))
        eg = model.embed_texts([t], with_grad=True).detach()
        info("relative difference embed_texts(with_grad=True) vs cached float16 default", round(float((eg[0] - e_alone[0]).norm() / e_alone[0].norm()), 6))
        check("embedding is stable under batch composition to 2 % relative", rel < 0.02, f"relative difference={rel:.5f}")


# ====================================================================================================================
def _reference_outputs(model, lat):
    bb = model.backbone
    with torch.no_grad():
        tb, out, e_a, a_mask, e_g, e_q, z, phi = pipeline(model)
        prompts = [bb.chat_prompt("Name one colour."), "The capital of France is"]
        return {"emb": model.embed_texts(["HTTP 404 Not Found", "three sign-ups were recorded"]).cpu(), "x": tb.x.cpu(), "h": out.h.cpu(),
                "b": out.b.cpu(), "slots": out.state.slots.cpu(), "e_a": e_a.cpu(), "e_g": e_g.cpu(), "e_q": e_q.cpu(), "z": z.cpu(), "phi": phi.cpu(),
                "head": model.head("g_world")(out.b).cpu(),
                "logits": bb.next_token_logits(prompts, latent=lat).cpu(),
                "lp_pos": bb.generation_logprob(prompts, ["Blue", " Paris"], latent=lat).cpu(),
                "text": model.generate(prompts, latent=lat, max_new_tokens=12, temperature=0.0)}


def part_save(model):
    import torch.nn as nn
    torch.manual_seed(0)
    model.register_head("g_world", nn.Linear(model.d_model, 5))
    dev = model.device
    lat = probe_w((2, model.d_model), dev, 9)
    # move every trainable group away from its initial value with two real optimizer steps
    opt = torch.optim.Adam([p for g in (GROUP_ADAPTERS, GROUP_EXPERIENTIAL, GROUP_HEADS) for p in model.param_groups()[g]], lr=1e-3)
    bb = model.backbone
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        tb, out, e_a, a_mask, e_g, e_q, z, phi = pipeline(model, grad_sem=True)
        loss = (phi * probe_w(phi.shape, dev, 3)).sum() + model.head("g_world")(out.b).pow(2).sum() \
            - bb.generation_logprob([bb.chat_prompt("Name one colour.")] * 2, ["Green", "Red"], latent=out.b).sum()
        loss.backward()
        opt.step()
    bb.mark_adapters_updated()
    h = model.group_hashes()
    ref = _reference_outputs(model, lat)
    ref2 = _reference_outputs(model, lat)
    same = all((ref[k] == ref2[k]) if k == "text" else bool(torch.equal(ref[k], ref2[k])) for k in ref)
    check("reference outputs are deterministic within the saving process", same,
          str([k for k in ref if not ((ref[k] == ref2[k]) if k == "text" else bool(torch.equal(ref[k], ref2[k])))]))
    st = model.trainable_state()
    os.makedirs(DIR, exist_ok=True)
    torch.save({"state": st, "ref": ref, "lat": lat.cpu(), "hashes": h, "sig": model.arch_signature()}, os.path.join(DIR, "g_nn_state.pt"))
    size = os.path.getsize(os.path.join(DIR, "g_nn_state.pt"))
    check("checkpoint file (trainable state + reference outputs) is < 0.5 GB", size < 0.5 * GB, f"{size / GB:.3f} GB")
    info("group hashes at save", h)


def part_load(model):
    import torch.nn as nn
    blob = torch.load(os.path.join(DIR, "g_nn_state.pt"), weights_only=False)
    model.register_head("g_world", nn.Linear(model.d_model, 5))
    h_fresh = model.group_hashes()
    check("fresh process starts from different trainable weights than the saved ones",
          all(h_fresh[g] != blob["hashes"][g] for g in (GROUP_ADAPTERS, GROUP_EXPERIENTIAL, GROUP_HEADS)))
    check("frozen-backbone identity hash matches", h_fresh[GROUP_BACKBONE_FROZEN] == blob["hashes"][GROUP_BACKBONE_FROZEN])
    import json
    sig_now = json.loads(json.dumps(model.arch_signature(), default=str))
    sig_saved = json.loads(json.dumps(blob["sig"], default=str))
    check("arch_signature() of the fresh process equals the saved one", sig_now == sig_saved)
    rep = model.load_trainable_state(blob["state"], strict=True)
    check("load_trainable_state(strict=True): nothing missing / unexpected / mismatched", rep == {"missing": [], "unexpected": [], "mismatched": []}, str(rep))
    check("group hashes after loading equal the saved hashes", model.group_hashes() == blob["hashes"])
    out = _reference_outputs(model, blob["lat"].to(model.device))
    diffs = {}
    for k, v in blob["ref"].items():
        if k == "text":
            diffs[k] = 0.0 if out[k] == v else float("nan")
        else:
            diffs[k] = 0.0 if torch.equal(out[k], v) else float((out[k].double() - v.double()).abs().max())
    check("fresh process: every output bit-identical after load (embeddings, tokens, h, b, slots, e_a, e_g, e_q, z, phi, head, "
          "bridged logits, generation log-prob, generated text)", all(d_ == 0.0 for d_ in diffs.values()), str(diffs))
    info("generated text after load", out["text"])
    # mismatches are detected
    bad = dict(blob["state"])
    k0 = next(k for k in bad if k.endswith("lora_A"))
    bad[k0] = bad[k0][:16]
    try:
        model.load_trainable_state(bad, strict=True)
        check("strict load rejects a tensor of the wrong shape (rank-16 LoRA into a rank-32 model)", False)
    except RuntimeError as e:
        check("strict load rejects a tensor of the wrong shape (rank-16 LoRA into a rank-32 model)", "shape mismatch" in str(e))
    bad = {k: v for k, v in blob["state"].items() if not k.startswith("exp.trunk.")}
    try:
        model.load_trainable_state(bad, strict=True)
        check("strict load rejects a state with missing tensors", False)
    except RuntimeError:
        check("strict load rejects a state with missing tensors", True)
    check("after the rejected loads the model still holds the saved weights (no partial load)", model.group_hashes() == blob["hashes"])


# ====================================================================================================================
def part_throughput(model, load_s, mem_after_load):
    from cam_live.nn.bench import _passage, _synthetic_tokens
    bb = model.backbone
    claimed = {"gpu_memory_after_load_gb": 7.804, "embedding_tokens_per_s_128": 10625.0, "generation_tokens_per_s_batch8_no_bridge": 192.7,
               "generation_tokens_per_s_batch1_no_bridge": 24.1, "experiential_core_step_b16_t256_s": 0.088,
               "experiential_core_step_peak_extra_gb": 1.556, "gen_logprob_backward_ckpt_s": 0.378, "gen_logprob_backward_ckpt_peak_extra_gb": 1.057,
               "gen_logprob_backward_nockpt_peak_extra_gb": 5.266}
    got = {"gpu_memory_after_load_gb": round(mem_after_load / GB, 3), "load_s": round(load_s, 2)}

    def timed(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        return time.time() - t0, (torch.cuda.max_memory_allocated() - base) / GB

    texts = [_passage(128, bb, 128000 + i) for i in range(128)]
    tok0 = bb.stats["embedded_tokens"]
    s, _ = timed(lambda: model.embed_texts(texts))
    got["embedding_tokens_per_s_128"] = round((bb.stats["embedded_tokens"] - tok0) / s, 0)
    prompt = bb.chat_prompt("Write the integers from 1 to 300 separated by commas, with no other text.")
    for bs in (1, 8):
        model.generate([prompt] * bs, max_new_tokens=4, temperature=0.0)
        g0 = bb.stats["generated_tokens"]
        s, _ = timed(lambda: model.generate([prompt] * bs, max_new_tokens=128, temperature=0.0))
        got[f"generation_tokens_per_s_batch{bs}_no_bridge"] = round((bb.stats["generated_tokens"] - g0) / s, 1)
    tb = _synthetic_tokens(model.exp, 16, 256, model.device)

    def core_step():
        model.zero_grad(set_to_none=True)
        model.core_forward(tb).b.pow(2).mean().backward()

    core_step()
    s, m = timed(core_step)
    got["experiential_core_step_b16_t256_s"], got["experiential_core_step_peak_extra_gb"] = round(s, 3), round(m, 3)
    prompts = [_passage(128, bb, 10 + i) for i in range(4)]
    comps = [_passage(128, bb, 50 + i) for i in range(4)]
    lat4 = torch.randn(4, model.d_model, device=model.device)

    def gen_step():
        model.zero_grad(set_to_none=True)
        (-model.generation_logprob(prompts, comps, latent=lat4).mean() / 128.0).backward()

    gen_step()
    s, m = timed(gen_step)
    got["gen_logprob_backward_ckpt_s"], got["gen_logprob_backward_ckpt_peak_extra_gb"] = round(s, 3), round(m, 3)
    bb.cfg.grad_checkpoint = False
    gen_step()
    s, m = timed(gen_step)
    got["gen_logprob_backward_nockpt_peak_extra_gb"] = round(m, 3)
    bb.cfg.grad_checkpoint = True
    model.zero_grad(set_to_none=True)
    info("claimed (docs/status/A.json measured)", claimed)
    info("measured now (shared GPU; speeds are lower bounds)", got)
    check("memory after load within 3 % of the claim", abs(got["gpu_memory_after_load_gb"] / claimed["gpu_memory_after_load_gb"] - 1) < 0.03)
    for k in ("experiential_core_step_peak_extra_gb", "gen_logprob_backward_ckpt_peak_extra_gb", "gen_logprob_backward_nockpt_peak_extra_gb"):
        check(f"{k} within 10 % of the claim", abs(got[k] / claimed[k] - 1) < 0.10, f"{got[k]} vs {claimed[k]}")
    for k in ("embedding_tokens_per_s_128", "generation_tokens_per_s_batch8_no_bridge", "generation_tokens_per_s_batch1_no_bridge"):
        check(f"{k} at least 60 % of the claim (shared machine)", got[k] > 0.6 * claimed[k], f"{got[k]} vs {claimed[k]}")
    for k in ("experiential_core_step_b16_t256_s", "gen_logprob_backward_ckpt_s"):
        check(f"{k} at most 1.7x the claim (shared machine)", got[k] < 1.7 * claimed[k], f"{got[k]} vs {claimed[k]}")
    check("peak GPU memory of this part < 24 GB", torch.cuda.max_memory_allocated() < 24 * GB)


# ====================================================================================================================
FD_PROMPTS = ["Name one colour.", "Which tool should be used next? Answer with one word.", "Write a short Python function that adds two numbers."]
FD_COMPS = ["Green", "browser", "def add(a, b):\n    return a + b"]
FD_TEXTS = ["The form was submitted and a confirmation appeared.", "def mean(xs):\n    return sum(xs) / len(xs)", "HTTP 404 Not Found",
            "three sign-ups were recorded after the announcement"]


def _emb_loss(model):
    e_ = model.embed_texts(FD_TEXTS, with_grad=True)
    return (e_ * probe_w(e_.shape, model.device, 5)).sum() / 100.0


def _gen_loss(model, latent):
    bb = model.backbone
    return -bb.generation_logprob([bb.chat_prompt(p_) for p_ in FD_PROMPTS], FD_COMPS, latent=latent).sum()


def _flat(ts):
    return torch.cat([t.detach().double().flatten().cpu() for t in ts])


def init_grads(model):
    """Analytic gradients at initialisation (identical seeds in every process): embedding loss w.r.t. LoRA-B,
    generation loss w.r.t. LoRA-B and the bridge output projections; plus the log-probabilities themselves."""
    bb = model.backbone
    lat = probe_w((3, model.d_model), model.device, 21)
    out = {}
    model.zero_grad(set_to_none=True)
    _emb_loss(model).backward()
    out["emb_B"] = _flat([m.lora_B.grad for m in bb.lora_modules])
    model.zero_grad(set_to_none=True)
    _gen_loss(model, lat).backward()
    out["gen_B"] = _flat([m.lora_B.grad for m in bb.lora_modules])
    out["gen_out"] = _flat([x.out.weight.grad for x in bb.bridge.xattn])
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        prompts = [bb.chat_prompt(p_) for p_ in FD_PROMPTS] + ["The capital of France is", "2 + 2 ="]
        comps = FD_COMPS + [" Paris", " 4"]
        out["lp"] = bb.generation_logprob(prompts, comps).double().cpu()
        out["n_tok"] = torch.tensor([len(bb._encode(c_)) for c_ in comps])
    return out


def part_gradsave(model):
    bb = model.backbone
    g = init_grads(model)
    os.makedirs(DIR, exist_ok=True)
    torch.save(g, os.path.join(DIR, "g_nn_grads_bf16.pt"))
    info("bf16 generation log-probabilities (5 completions)", [round(float(v), 4) for v in g["lp"]])
    info("gradient norms (bf16): emb->LoRA-B, gen->LoRA-B, gen->bridge-out", [round(float(g[k].norm()), 4) for k in ("emb_B", "gen_B", "gen_out")])
    # resolution of generation_logprob: the LM head runs in bf16, logits of magnitude 16..32 are multiples of 1/8
    with torch.no_grad():
        prompts = [bb.chat_prompt(p_) for p_ in FD_PROMPTS]
        P = [bb._prompt_ids(p_) for p_ in prompts]
        C = [bb._encode(c_) for c_ in FD_COMPS]
        ids, attn = bb._pad([p_ + c_ for p_, c_ in zip(P, C)])
        h = bb._hidden(ids, attn)
        row = torch.tensor([i for i, c_ in enumerate(C) for _ in c_], device=model.device)
        pos = torch.tensor([len(P[i]) - 1 + j for i, c_ in enumerate(C) for j in range(len(c_))], device=model.device)
        tgt = torch.tensor([t for c_ in C for t in c_], device=model.device)
        logits16 = bb._logits(h[row, pos])
        logits32 = Fn.linear(h[row, pos].float(), bb.lm.lm_head.weight.float())
        lp16 = torch.log_softmax(logits16.float(), -1).gather(1, tgt[:, None]).squeeze(1)
        lp32 = torch.log_softmax(logits32, -1).gather(1, tgt[:, None]).squeeze(1)
        top = logits16.float().max(-1).values
        frac = float(((logits16.float() * 8) % 1 == 0).float().mean())
        info("LM head dtype / max logit magnitude / fraction of logits that are exact multiples of 1/8",
             (str(logits16.dtype), round(float(top.max()), 2), round(frac, 4)))
        d = (lp16 - lp32).abs()
        info("per-token |log p (bf16 head) - log p (float32 head on the same hidden states)|: mean / max", (round(float(d.mean()), 4), round(float(d.max()), 4)))
        torch.save({"lp_tok16": lp16.cpu(), "lp_tok32head": lp32.cpu()}, os.path.join(DIR, "g_nn_lp_bf16.pt"))
        check("generation_logprob resolves per-token log-probabilities to better than 0.02 nats (bf16 head vs float32 head)", float(d.max()) < 0.02,
              f"max per-token error {float(d.max()):.4f} nats, mean {float(d.mean()):.4f}")


def part_fd32(model):
    """The same model with the backbone loaded in float32: finite-difference checks of every gradient path with real
    precision, and the agreement of the bf16 gradients with them."""
    bb = model.backbone
    dev = model.device
    lora = bb.lora_modules
    info("backbone dtype", str(next(bb.lm.parameters()).dtype))
    g32 = init_grads(model)

    def fd(params, grads, loss_fn, dls):
        gn = math.sqrt(sum(float(g_.double().pow(2).sum()) for g_ in grads))
        ratios = []
        with torch.no_grad():
            for dl in dls:
                eps = dl / gn
                for p, g_ in zip(params, grads):
                    p.add_(g_, alpha=eps / gn)
                lp_ = float(loss_fn().double())
                for p, g_ in zip(params, grads):
                    p.add_(g_, alpha=-2 * eps / gn)
                lm_ = float(loss_fn().double())
                for p, g_ in zip(params, grads):
                    p.add_(g_, alpha=eps / gn)
                ratios.append(round((lp_ - lm_) / (2 * eps) / gn, 4))
        return gn, ratios

    lat = probe_w((3, model.d_model), dev, 21)
    # at initialisation
    model.zero_grad(set_to_none=True)
    _emb_loss(model).backward()
    gB = [m.lora_B.grad.detach().clone() for m in lora]
    gn, r = fd([m.lora_B for m in lora], gB, lambda: _emb_loss(model), (0.01, 0.03, 0.1))
    check("[fp32, init] embedding loss: FD along the LoRA-B gradient == |g| (3 %)", all(abs(x - 1) < 0.03 for x in r[:2]), f"|g|={gn:.4g} FD/|g|={r}")
    model.zero_grad(set_to_none=True)
    _gen_loss(model, lat).backward()
    gB = [m.lora_B.grad.detach().clone() for m in lora]
    gO = [x.out.weight.grad.detach().clone() for x in bb.bridge.xattn]
    gn, r = fd([m.lora_B for m in lora], gB, lambda: _gen_loss(model, lat), (0.01, 0.03, 0.1))
    check("[fp32, init] generation loss: FD along the LoRA-B gradient == |g| (3 %)", all(abs(x - 1) < 0.03 for x in r[:2]), f"|g|={gn:.4g} FD/|g|={r}")
    gn, r = fd([x.out.weight for x in bb.bridge.xattn], gO, lambda: _gen_loss(model, lat), (0.01, 0.03, 0.1))
    check("[fp32, init] generation loss: FD along the zero-initialised bridge-output gradient == |g| (3 %)", all(abs(x - 1) < 0.03 for x in r[:2]),
          f"|g|={gn:.4g} FD/|g|={r}")
    # bf16 gradients against the float32 ones
    path = os.path.join(DIR, "g_nn_grads_bf16.pt")
    if os.path.exists(path):
        g16 = torch.load(path)
        for k in ("emb_B", "gen_B", "gen_out"):
            cos = float(Fn.cosine_similarity(g16[k], g32[k], 0))
            ratio = float(g16[k].norm() / g32[k].norm())
            check(f"bf16 gradient vs float32 gradient [{k}]: cos > 0.9 and norm ratio within 15 %", cos > 0.9 and abs(ratio - 1) < 0.15,
                  f"cos={cos:.4f} |g_bf16|/|g_fp32|={ratio:.4f}")
        d = (g16["lp"] - g32["lp"])
        info("generation_logprob bf16 minus float32 (5 completions; tokens per completion)", ([round(float(v), 4) for v in d], g32["n_tok"].tolist()))
        info("float32 generation log-probabilities", [round(float(v), 4) for v in g32["lp"]])
        per_tok = float((d.abs() / g32["n_tok"]).max())
        check("bf16 generation_logprob within 0.05 nats per token of the float32 model", per_tok < 0.05, f"max |error| per token = {per_tok:.4f}")
    # away from initialisation: every adapter tensor, the latent, and the experiential weights through b
    with torch.no_grad():
        gen = torch.Generator(dev).manual_seed(11)
        for m in lora:
            m.lora_B.copy_(torch.randn(m.lora_B.shape, device=dev, generator=gen) * 0.003)
        for x in bb.bridge.xattn:
            x.out.weight.copy_(torch.randn(x.out.weight.shape, device=dev, generator=gen) * 0.02)
    bb.mark_adapters_updated()
    adapters = model.param_groups()[GROUP_ADAPTERS]
    model.zero_grad(set_to_none=True)
    lat_g = lat.clone().requires_grad_(True)
    (_gen_loss(model, lat_g) + _emb_loss(model)).backward()
    gA = [p.grad.detach().clone() for p in adapters]
    dead = [n for n, p in list(bb.bridge.named_parameters()) + [(f"lora{i}.A", m.lora_A) for i, m in enumerate(lora)]
            if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    check("[fp32, trained-like] every adapter tensor (all LoRA A/B, every bridge tensor) has non-zero gradient", not dead, str(dead[:6]))
    gn, r = fd(adapters, gA, lambda: _gen_loss(model, lat) + _emb_loss(model), (0.01, 0.03, 0.1))
    check("[fp32, trained-like] generation + embedding loss: FD along the full adapter gradient == |g| (3 %)", all(abs(x - 1) < 0.03 for x in r[:2]),
          f"|g|={gn:.4g} FD/|g|={r}")
    gl = lat_g.grad.detach().clone()
    gn, r = fd([lat], [gl], lambda: _gen_loss(model, lat), (0.01, 0.03, 0.1))
    check("[fp32, trained-like] generation loss: FD along d/d latent == |g| (3 %)", all(abs(x - 1) < 0.03 for x in r[:2]), f"|g|={gn:.4g} FD/|g|={r}")
    # checkpointing on / off in float32
    grads = {}
    for ck in (True, False):
        bb.cfg.grad_checkpoint = ck
        model.zero_grad(set_to_none=True)
        (_gen_loss(model, lat) + _emb_loss(model)).backward()
        grads[ck] = _flat([p.grad for p in adapters])
    bb.cfg.grad_checkpoint = True
    rel = float((grads[True] - grads[False]).norm() / grads[False].norm())
    check("[fp32] gradient checkpointing on vs off: relative gradient difference < 1e-4", rel < 1e-4, f"relative difference={rel:.2e}")
    # generation loss -> bridge -> b -> experiential weights
    exp_params = list(model.exp.parameters())
    ctx3 = contexts() + [contexts()[0][:2]]

    def gen_from_core():
        o_ = model.core_forward(model.tokenize(ctx3))
        return _gen_loss(model, o_.b)

    model.zero_grad(set_to_none=True)
    gen_from_core().backward()
    gE = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in exp_params]
    gn, r = fd(exp_params, gE, gen_from_core, (0.003, 0.01, 0.03))
    check("[fp32, trained-like] generation loss -> bridge -> b -> experiential weights: FD along the gradient == |g| (3 %)",
          gn > 0 and all(abs(x - 1) < 0.03 for x in r[:2]), f"|g|={gn:.4g} FD/|g|={r}")
    check("[fp32] frozen backbone untouched", frozen_ok(model) and not bb.lm.training)
    # a loss on z through tokenize(with_grad=True): FD along LoRA-B
    def z_loss():
        tb_ = model.tokenize(contexts(), with_grad=True)
        o_ = model.core_forward(tb_)
        e_a_, _ = model.encode_actions(ACTS, with_grad=True)
        e_g_ = model.encode_goals(GOALS, with_grad=True)
        z_ = model.decision_latent(o_.b, e_a_, e_g_)
        return (z_ * probe_w(z_.shape, dev, 3)).sum() / 30.0

    model.zero_grad(set_to_none=True)
    z_loss().backward()
    gL = [p.grad.detach().clone() for p in bb.lora_parameters()]
    gn, r = fd(bb.lora_parameters(), gL, z_loss, (0.01, 0.03, 0.1))
    check("[fp32, trained-like] loss on z via tokenize / encode_actions / encode_goals(with_grad=True): FD along the LoRA gradient == |g| (3 %)",
          all(abs(x - 1) < 0.03 for x in r[:2]), f"|g|={gn:.4g} FD/|g|={r}")


# ====================================================================================================================
def part_misc(model):
    """Spot-checks of further numbers / statements in docs/status/A.json."""
    from cam_live.nn.bench import _passage
    bb = model.backbone
    dev = model.device
    # (a) joint-loss gradient norms (same construction as nn/bench.py)
    model.zero_grad(set_to_none=True)
    ctx = [[ExperienceItem(kind=ItemKind.OBSERVATION, text=_passage(48, bb, 200 + 10 * b + t), ts=1000.0 + 600.0 * t,
                           quantities=[Quantity("visits", 100.0 + t)]) for t in range(8)] for b in range(4)]
    tb = model.tokenize(ctx, with_grad=True, grad_last=2)
    out = model.core_forward(tb)
    e_a, _ = model.encode_actions([[ActionSpec(f"a{b}{i}", "browser.click", arguments={"ref": i}) for i in range(4)] for b in range(4)])
    e_g = model.encode_goals(["reach the stated target"] * 4)
    e_q = model.encode_queries([[OutcomeQuery("q", "visits", kind=OutcomeKind.COUNT, horizon_s=3600.0)]] * 4)[0]
    phi = model.features(model.decision_latent(out.b, e_a, e_g), e_q)
    w = torch.randn(phi.shape, device=phi.device, generator=torch.Generator(phi.device).manual_seed(0))
    (phi * w).sum().backward()
    norms = model.grad_norms()
    check("joint_loss_grad_norms reproduce (claimed experiential 597.79, adapters 3.581; 3 %)",
          abs(norms[GROUP_EXPERIENTIAL] / 597.7935791015625 - 1) < 0.03 and abs(norms[GROUP_ADAPTERS] / 3.5810763835906982 - 1) < 0.03,
          f"experiential {norms[GROUP_EXPERIENTIAL]:.3f}, adapters {norms[GROUP_ADAPTERS]:.4f}")
    model.zero_grad(set_to_none=True)
    # (b) a literal all-zero soft prefix is not a no-op (the justification of change request A-2)
    with torch.no_grad():
        p2 = [bb.chat_prompt("Name one colour.")]
        base = bb.next_token_logits(p2)
        ids, attn = bb._pad([bb._prompt_ids(p2[0])])
        e = bb.lm.get_input_embeddings()(ids)
        K = bb.bridge.n_tokens
        e2 = torch.cat([torch.zeros(1, K, e.shape[-1], dtype=e.dtype, device=e.device), e], 1)
        a2 = torch.cat([torch.ones(1, K, dtype=attn.dtype, device=attn.device), attn], 1)
        zp = bb._logits(bb.lm.model(inputs_embeds=e2, attention_mask=a2, use_cache=False).last_hidden_state[:, -1]).float()
        d = float((base - zp).abs().max())
    check("literal all-zero soft prefix changes the logits (claimed max |dlogit| 1.0469)", abs(d - 1.0469) < 0.15, f"max |dlogit|={d:.4f}")
    # (c) sampler vs generation_logprob: do the probabilities the sampler used equal generation_logprob of the same tokens?
    with torch.no_grad():
        worst = 0.0
        rows_out = []
        for prompt in (bb.chat_prompt("Name three colours."), "The capital of France is", bb.chat_prompt("Write one line of Python that prints 7.")):
            P = bb._prompt_ids(prompt)
            ids, attn = bb._pad([P], left=True)
            cache, cur, toks, lps = None, ids, [], []
            for _ in range(16):
                logits, cache = bb._step(cur, attn, cache)
                lp = torch.log_softmax(logits, -1)
                nxt = lp.argmax(-1)
                toks.append(int(nxt))
                lps.append(float(lp[0, nxt]))
                attn = torch.cat([attn, torch.ones(1, 1, dtype=attn.dtype, device=dev)], 1)
                cur = nxt.unsqueeze(1)
            full, fattn = bb._pad([P + toks])
            h = bb._hidden(full, fattn)
            pos = torch.arange(len(P) - 1, len(P) - 1 + len(toks), device=dev)
            lp_full = torch.log_softmax(bb._logits(h[0, pos]).float(), -1).gather(1, torch.tensor(toks, device=dev)[:, None]).squeeze(1)
            dtok = (lp_full.cpu().double() - torch.tensor(lps, dtype=torch.float64)).abs()
            rows_out.append((round(sum(lps), 3), round(float(lp_full.sum()), 3), round(float(dtok.max()), 3)))
            worst = max(worst, float(dtok.max()))
        info("greedy 16-token continuations: (sum log p in the KV-cache sampler, sum log p by the generation_logprob computation, max per-token gap)", rows_out)
        check("sampler probabilities and generation_logprob agree to 0.05 nats per token", worst < 0.05, f"max per-token gap={worst:.3f}")
    # (d) generation speed: bare backbone vs non-zero LoRA (statement: 'about 15 % slower (measured)')
    prompt = bb.chat_prompt("Write the integers from 1 to 300 separated by commas, with no other text.")

    def speed(bs):
        model.generate([prompt] * bs, max_new_tokens=4, temperature=0.0)
        best = 0.0
        for _ in range(2):
            g0 = bb.stats["generated_tokens"]
            torch.cuda.synchronize()
            t0 = time.time()
            model.generate([prompt] * bs, max_new_tokens=128, temperature=0.0)
            torch.cuda.synchronize()
            best = max(best, (bb.stats["generated_tokens"] - g0) / (time.time() - t0))
        return round(best, 1)

    s1, s8 = speed(1), speed(8)
    with torch.no_grad():
        for m in bb.lora_modules:
            m.lora_B.normal_(0.0, 1e-9)
    bb.mark_adapters_updated()
    l1, l8 = speed(1), speed(8)
    with torch.no_grad():
        for m in bb.lora_modules:
            m.lora_B.zero_()
    bb.mark_adapters_updated()
    info("generation tokens/s: bare (LoRA B == 0, skipped) batch 1 / batch 8", (s1, s8))
    info("generation tokens/s: non-zero LoRA (unmerged) batch 1 / batch 8", (l1, l8))
    info("slow-down with non-zero adapters (batch 1, batch 8)", (round(1 - l1 / s1, 3), round(1 - l8 / s8, 3)))
    check("status claim: single-stream generation ~24.1 tokens/s (at least 60 % reproduced)", s1 > 0.6 * 24.1, f"{s1} tokens/s")
    check("status claim: batch-8 generation ~192.7 tokens/s (at least 60 % reproduced)", s8 > 0.6 * 192.7, f"{s8} tokens/s")
    check("status claim: 'with trained adapters generation is about 15 % slower' (measured slow-down between 5 % and 30 %)",
          0.05 < 1 - l8 / s8 < 0.30 or 0.05 < 1 - l1 / s1 < 0.30, f"batch 1: {1 - l1 / s1:.1%}, batch 8: {1 - l8 / s8:.1%}")


# ====================================================================================================================
def part_phi(model):
    """Does the decision path of the real configuration separate contexts / actions / queries, and is the whole
    text -> belief pipeline invariant to batch composition?"""
    bb = model.backbone
    cos = lambda a, b_: float(Fn.cosine_similarity(a, b_, -1).mean())
    with torch.no_grad():
        ctxs = [
            [ExperienceItem(kind=ItemKind.OBSERVATION, text="A checkout form with three empty fields and a disabled submit button.", ts=1000.0,
                            quantities=[Quantity("fields", 3.0)]),
             ExperienceItem(kind=ItemKind.ACTION, text="browser.type into the field labelled Name", ts=1004.0)],
            [ExperienceItem(kind=ItemKind.DOCUMENT, text="Section 2 of the manual explains how the export job is scheduled and where its log file is written.", ts=50000.0),
             ExperienceItem(kind=ItemKind.OUTCOME, text="rows exported", ts=58000.0, quantities=[Quantity("rows", 18250.0)])],
            [ExperienceItem(kind=ItemKind.FAILURE, text="timeout transient_network: the request did not complete", ts=200.0)],
            [ExperienceItem(kind=ItemKind.SOFTWARE, text="def mean(xs):\n    return sum(xs) / len(xs)", ts=10.0),
             ExperienceItem(kind=ItemKind.OUTCOME, text="members.count (observed)", ts=260000.0, quantities=[Quantity("members.count", 7.0)])],
            [ExperienceItem(kind=ItemKind.OUTCOME, text="members.count (observed)", ts=260000.0, quantities=[Quantity("members.count", 70.0)])],
            [ExperienceItem(kind=ItemKind.OUTCOME, text="members.count (observed)", ts=260000.0, quantities=[Quantity("members.count", 7.0)])],
        ]
        acts = [ActionSpec("a1", "browser.click", arguments={"ref": "e3"}), ActionSpec("a2", "sandbox.python", modality="code", arguments={"code": "print(1)"}),
                ActionSpec("a3", "internal.wait", arguments={"seconds": 600}), ActionSpec("a4", "browser.click", arguments={"ref": "e4"}),
                ActionSpec("a5", "api.call", modality="api", arguments={"endpoint": "/export", "rows": 100})]
        qs = [OutcomeQuery("q1", "signups", kind=OutcomeKind.COUNT, horizon_s=3600.0), OutcomeQuery("q2", "signups", kind=OutcomeKind.COUNT, horizon_s=30 * 86400.0),
              OutcomeQuery("q3", "latency", horizon_s=30.0, description="seconds until the job is done"),
              OutcomeQuery("q4", "done", kind=OutcomeKind.BINARY, horizon_s=600.0)]
        B = len(ctxs)
        o = model.core_forward(model.tokenize(ctxs))
        e_a, _ = model.encode_actions([acts] * B)
        e_g = model.encode_goals(["Reach the stated target within the time limit."] * B)
        e_g2 = model.encode_goals(["Find out why the export job fails."] * B)
        e_q = model.encode_queries([qs] * B)[0]
        z = model.decision_latent(o.b, e_a, e_g)
        phi = model.features(z, e_q)                                      # [B, 5, 4, F]
        z2 = model.decision_latent(o.b, e_a, e_g2)
        info("cos of b between different contexts (mean over pairs 0-1, 1-2, 2-3)", round(cos(o.b[:3], o.b[1:4]), 4))
        info("cos of b: same single outcome item, value 70 vs 7", round(cos(o.b[4], o.b[5]), 4))
        info("cos of z: two different operations / same operation with another argument", (round(cos(z[:, 0], z[:, 1]), 4), round(cos(z[:, 0], z[:, 3]), 4)))
        info("relative change of z when only the goal changes", round(float((z - z2).norm() / z.norm()), 4))
        info("cos of phi: across actions / across contexts / across queries (same metric, horizon 1 h vs 30 d) / across metrics",
             (round(cos(phi[:, 0], phi[:, 1]), 4), round(cos(phi[0], phi[1]), 4), round(cos(phi[:, :, 0], phi[:, :, 1]), 4), round(cos(phi[:, :, 0], phi[:, :, 2]), 4)))
        P = phi.reshape(-1, model.n_features).double()
        ev = torch.linalg.eigvalsh(P.T @ P / P.shape[0]).flip(0)
        pr = float(ev.sum() ** 2 / ev.pow(2).sum())
        info("phi Gram over 120 (context, action, query) triples: top-5 eigenvalue shares, participation ratio (of 128)",
             ([round(float(v / ev.sum()), 3) for v in ev[:5]], round(pr, 2)))
        check("decision path separates candidates: cos(z) between two different operations < 0.95 and phi across actions < 0.99",
              cos(z[:, 0], z[:, 1]) < 0.95 and cos(phi[:, 0], phi[:, 1]) < 0.99)
        # text -> belief under batch composition, cold cache both times
        bb.mark_adapters_updated()
        b_alone = model.core_forward(model.tokenize([ctxs[0]])).b[0].clone()
        bb.mark_adapters_updated()
        b_batch = model.core_forward(model.tokenize(ctxs)).b[0].clone()
        rel = float((b_alone - b_batch).norm() / b_alone.norm())
        check("text -> belief is invariant to batch composition to 1e-3 relative on the real backbone (cold embedding cache)", rel < 1e-3,
              f"relative difference of b = {rel:.5f} (bf16 backbone: an embedding depends on the batch it was first computed in; the cache then keeps it)")
        # item kind / trust carry a visible part of the token?
        t = "members.count (observed)"
        x = model.tokenize([[ExperienceItem(kind=k, text=t, ts=1.0, trust=tr)] for k, tr in ((ItemKind.OUTCOME, "untrusted"), (ItemKind.PREDICTION, "untrusted"),
                                                                                          (ItemKind.OUTCOME, "verifier"))]).x[:, 0]
        info("token cos: same text, kind outcome vs prediction / trust untrusted vs verifier", (round(cos(x[0], x[1]), 4), round(cos(x[0], x[2]), 4)))


# ====================================================================================================================
def part_cond(model):
    """Conditioning of phi on REAL inputs (the unit test feeds i.i.d. Gaussian z / e_q into the debug feature map)."""
    from cam_live.nn.bench import _passage
    bb = model.backbone
    g = torch.Generator().manual_seed(0)
    kinds = [ItemKind.OBSERVATION, ItemKind.ACTION, ItemKind.OUTCOME, ItemKind.EVENT, ItemKind.DOCUMENT, ItemKind.FAILURE, ItemKind.SOFTWARE]
    words = ["checkout form", "export job", "search results", "login page", "error log", "pricing table", "signup funnel", "api response",
             "unit tests", "database migration", "newsletter draft", "support ticket"]
    ctxs = []
    for i in range(48):
        n = int(torch.randint(1, 7, (1,), generator=g))
        its = []
        for t in range(n):
            w = words[int(torch.randint(0, len(words), (1,), generator=g))]
            v = float(torch.sinh(torch.randn(1, generator=g) * 2.0))
            its.append(ExperienceItem(kind=kinds[int(torch.randint(0, len(kinds), (1,), generator=g))],
                                      text=f"{w}: " + _passage(int(torch.randint(6, 40, (1,), generator=g)), bb, 9000 + 50 * i + t),
                                      ts=1000.0 * i + float(torch.rand(1, generator=g)) * 10 ** float(torch.rand(1, generator=g) * 6) + 10.0 * t,
                                      quantities=[Quantity(f"{w.split()[0]}.count", round(v, 2))] if t % 2 == 0 else []))
        ctxs.append(sorted(its, key=lambda it: it.ts))
    acts = [ActionSpec("a1", "browser.click", arguments={"ref": "e3"}), ActionSpec("a2", "sandbox.python", modality="code", arguments={"code": "print(1)"}),
            ActionSpec("a3", "internal.wait", arguments={"seconds": 600}), ActionSpec("a4", "browser.type", arguments={"ref": "e7", "text": "hello"}),
            ActionSpec("a5", "api.call", modality="api", arguments={"endpoint": "/export", "rows": 100}),
            ActionSpec("a6", "skill.run", modality="skill", level="procedure", arguments={"skill": "publish_page"}),
            ActionSpec("a7", "browser.navigate", arguments={"url": "https://example.org/docs"}),
            ActionSpec("a8", "visual.click", modality="visual", arguments={"x": 310, "y": 88})]
    qs = [OutcomeQuery("q1", "signups", kind=OutcomeKind.COUNT, horizon_s=3600.0), OutcomeQuery("q2", "signups", kind=OutcomeKind.COUNT, horizon_s=30 * 86400.0),
          OutcomeQuery("q3", "latency", horizon_s=30.0, description="seconds until the job is done"), OutcomeQuery("q4", "done", kind=OutcomeKind.BINARY, horizon_s=600.0),
          OutcomeQuery("q5", "time_to_first_reply", kind=OutcomeKind.TIME_TO_EVENT, horizon_s=86400.0), OutcomeQuery("q6", "revenue", horizon_s=7 * 86400.0, description="money received")]
    goals = ["Reach the stated target within the time limit.", "Find out why the export job fails.", "Increase voluntary sign-ups.", "Reduce the page load time."]
    with torch.no_grad():
        B = len(ctxs)
        o = model.core_forward(model.tokenize(ctxs))
        e_a, _ = model.encode_actions([acts] * B)
        e_g = model.encode_goals([goals[i % 4] for i in range(B)])
        e_q = model.encode_queries([qs] * B)[0]
        phi = model.features(model.decision_latent(o.b, e_a, e_g), e_q).reshape(-1, model.n_features).double().cpu()
    n = phi.shape[0]
    ev = torch.linalg.eigvalsh(phi.T @ phi / n).flip(0).clamp_min(0)
    cond = float(ev[0] / ev[-1].clamp_min(1e-300))
    pr = float(ev.sum() ** 2 / ev.pow(2).sum())
    n90 = int((ev.cumsum(0) / ev.sum() < 0.90).sum()) + 1
    info(f"phi Gram on {n} real (context, action, goal, query) combinations: eigenvalue max / min, participation ratio, directions holding 90 % of the mass (of 128)",
         (f"{cond:.3g}", round(pr, 2), n90))
    info("top-8 eigenvalue shares", [round(float(v / ev.sum()), 4) for v in ev[:8]])
    # per query (what one Bayesian posterior over a single metric sees)
    with torch.no_grad():
        phi_q = model.features(model.decision_latent(o.b, e_a, e_g), e_q)[:, :, 0].reshape(-1, model.n_features).double().cpu()
    evq = torch.linalg.eigvalsh(phi_q.T @ phi_q / phi_q.shape[0]).flip(0).clamp_min(0)
    info("same for a single query (384 samples): max / min eigenvalue, participation ratio",
         (f"{float(evq[0] / evq[-1].clamp_min(1e-300)):.3g}", round(float(evq.sum() ** 2 / evq.pow(2).sum()), 2)))
    # the unit-test construction on the real feature map, for comparison
    with torch.no_grad():
        gz = torch.Generator(model.device).manual_seed(0)
        zr = torch.randn(64, 8, model.d_model, device=model.device, generator=gz)
        qr = torch.randn(64, 4, model.d_model, device=model.device, generator=gz)
        pr_ = model.features(zr, qr).reshape(-1, model.n_features).double().cpu()
    evr = torch.linalg.eigvalsh(pr_.T @ pr_ / pr_.shape[0]).flip(0)
    info("unit-test construction (i.i.d. Gaussian z and e_q) on the real feature map: max / min eigenvalue", f"{float(evr[0] / evr[-1]):.3g}")
    check("phi Gram on real inputs meets the unit test's own bound (max / min eigenvalue < 1e4)", cond < 1e4,
          f"max/min={cond:.3g} on real inputs vs {float(evr[0] / evr[-1]):.3g} with the test's white-noise inputs")


# ====================================================================================================================
t_wait = time.time()
with gpu_lock(timeout_s=3000.0, min_free_gb=16.0) as t_lock:
    info("waited for the GPU lock (s)", round(t_lock - t_wait, 1))
    torch.cuda.reset_peak_memory_stats()
    mem0 = torch.cuda.memory_allocated()
    t0 = time.time()
    _cfg = CamLiveConfig(device="cuda")
    if part == "fd32":
        _cfg.backbone.dtype = "float32"
    model = build_model(_cfg)
    torch.cuda.synchronize()
    load_s, mem_after = time.time() - t0, torch.cuda.memory_allocated() - mem0
    try:
        if part == "throughput":
            part_throughput(model, load_s, mem_after)
        else:
            {"params": part_params, "grads": part_grads, "bridge": part_bridge, "values": part_values, "save": part_save,
             "load": part_load, "gradsave": part_gradsave, "fd32": part_fd32, "misc": part_misc, "phi": part_phi, "cond": part_cond}[part](model)
    finally:
        info("peak GPU memory in this part (GB)", round(torch.cuda.max_memory_allocated() / GB, 2))
        del model
        free_gpu_memory()
        info("GPU lock held (s)", round(time.time() - t_lock, 1))
sys.exit(finish())
