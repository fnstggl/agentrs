"""'Not a frozen-LLM wrapper' on the REAL configuration (frozen Qwen3-4B + LoRA + bridge, 50m core) — PI-01, PI-02,
PI-08, PI-09, CU-12. Heavy GPU work under cam_live.nn.gpu.gpu_lock().

1. Predictor + DecisionMaker on a LabWorld context: predictions registered before evaluate_candidates returns
   (registry made_at == clock), probabilities recorded.
2. Change ONLY the experiential core weights (small relative perturbation): the predicted consequences and the
   decision probabilities change; the frozen backbone and the adapters stay bit-identical.
3. Bridge: log p(completion | prompt, latent) of the backbone for two different beliefs — at initialisation the
   bridge is documented as an exact no-op, i.e. the proposer is NOT conditioned on the latent in effect; after a
   perturbation of the bridge's output projection the latent changes the backbone's distribution (the path exists).
4. BackboneProposer.propose with the belief latent: provenance, validity and probabilities of the candidates.
"""
import math
import os
import sys
import tempfile
import time

import torch

import _common as C
from cam_live.config import CamLiveConfig
from cam_live.contracts import RunFiles
from cam_live.diagnostics.collect import obs_item
from cam_live.diagnostics.world import LabWorld, lab_task
from cam_live.nn.api import GROUP_ADAPTERS, GROUP_EXPERIENTIAL
from cam_live.nn.gpu import gpu_lock
from cam_live.nn.model import build_model
from cam_live.predict import diagnose as G
from cam_live.predict import labdata as L
from cam_live.predict import policy as P
from cam_live.predict.candidates import BackboneProposer, schema_candidates
from cam_live.predict.heads import attach_heads
from cam_live.predict.predictor import Predictor
from cam_live.predict.registry import PredictionRegistry

t0 = time.time()
with gpu_lock(timeout_s=3600.0, min_free_gb=12.0):
    model = build_model(CamLiveConfig(device="cuda"))
    C.info("model load s", round(time.time() - t0, 1))
    heads = attach_heads(model)
    tmp = tempfile.mkdtemp()
    world = LabWorld(L.lab_config(seed=21))
    clock, task = world.clock, lab_task(world)
    reg = PredictionRegistry(os.path.join(tmp, RunFiles.PREDICTIONS), clock)
    ctx = [obs_item(o) for o in world.observe()]
    cands = schema_candidates(world.discover_tools(), per_tool=6)
    queries = G.act_queries()

    def decide():
        pred = Predictor(model, model.new_fast_state(), reg, clock, heads=heads)
        evs = pred.evaluate_candidates(ctx, task.objective, cands, queries, validate=world.validate_action,
                                       generator=torch.Generator().manual_seed(0))
        acts = [e for e in evs if e.action.operation == "lab.act"]
        dm = P.DecisionMaker(model, heads=heads, clock=clock, seed=0)
        out = {r: dm.probabilities(acts, r)[0] for r in ("policy", "greedy", "ucb", "thompson")}
        means = torch.tensor([[p.dist.mean for p in e.predictions] for e in acts], dtype=torch.float64)
        return evs, acts, out, means

    evs, acts, probs0, means0 = decide()
    C.check("predictions of every lab.act candidate registered with made_at == clock before return",
            all(reg.get(p.prediction_id).made_at == clock.now() for e in acts for p in e.predictions) and
            all(len(e.predictions) == 2 for e in acts), f"{len(acts)} candidates x 2 horizons")
    C.check("every policy-rule probability vector sums to 1 with the floor", all(abs(float(v.sum()) - 1) < 1e-12 and
                                                                               float(v.min()) > 0 for v in probs0.values()))
    hashes = model.group_hashes()
    e0 = acts[0].predictions[0].dist
    C.check("real-model predictions carry epistemic and aleatoric uncertainty (PI-05)",
            all(p.dist.epistemic_std > 0 for e in acts for p in e.predictions), f"epistemic_std={e0.epistemic_std:.3g}")
    exp = model.param_groups()[GROUP_EXPERIENTIAL]
    C.info("experiential parameters", sum(p.numel() for p in exp))
    orig = [p.detach().clone() for p in exp]
    g = torch.Generator(device="cpu").manual_seed(3)
    noise = [torch.randn(p.shape, generator=g) for p in exp]
    for scale in (0.05, 0.5):
        with torch.no_grad():
            for p, o, nz in zip(exp, orig, noise):
                p.copy_(o)
                if p.dtype.is_floating_point and p.numel() > 1:
                    p.add_(scale * o.float().std().clamp_min(1e-3).to(p.dtype) * nz.to(p.device, p.dtype))
        h1 = model.group_hashes()
        C.check(f"[x{scale}] perturbation touched only the experiential group (adapters / heads / reward unchanged)",
                all(h1[k] == hashes[k] for k in hashes if k != GROUP_EXPERIENTIAL) and h1[GROUP_EXPERIENTIAL] !=
                hashes[GROUP_EXPERIENTIAL], f"changed: {[k for k in hashes if h1[k] != hashes[k]]}")
        _, acts1, probs1, means1 = decide()
        tvs = {r: 0.5 * float((probs1[r] - probs0[r]).abs().sum()) for r in probs0}
        C.info(f"[x{scale}] TV of the decision distribution per rule after an experiential-only change", tvs)
        C.check(f"[x{scale}] rule 'policy': changing only experiential weights changes the decision distribution",
                tvs["policy"] > 0.0, f"TV={tvs['policy']:.3e}")
        rel = float(((means1 - means0).abs() / means0.abs().clamp_min(1e-9)).max())
        C.check(f"[x{scale}] changing only experiential weights changes the predicted consequences", rel > 1e-4,
                f"max rel={rel:.4f}")
        C.info(f"[x{scale}] greedy argmax before / after", (int(probs0["greedy"].argmax()), int(probs1["greedy"].argmax())))
    with torch.no_grad():
        for p, o in zip(exp, orig):
            p.copy_(o)

    # ---- the bridge -----------------------------------------------------------------------------------------
    bb = model.backbone
    prompt = "List one action for the community experiment:"
    comp = ' {"operation": "lab.act", "arguments": {"channel": 1, "effort": 2}}'
    with torch.no_grad():
        b1 = torch.randn(1, model.d_model, generator=torch.Generator().manual_seed(1)).to("cuda")
        b2 = torch.randn(1, model.d_model, generator=torch.Generator().manual_seed(2)).to("cuda") * 3
        lp_none = float(bb.generation_logprob([prompt], [comp]))
        lp1 = float(bb.generation_logprob([prompt], [comp], latent=b1))
        lp2 = float(bb.generation_logprob([prompt], [comp], latent=b2))
    C.info("log p(completion) without latent / with b1 / with b2", (lp_none, lp1, lp2))
    C.check("at initialisation the backbone's distribution depends on the belief latent (proposer conditioned on b)",
            abs(lp1 - lp2) > 1e-3, f"|lp(b1) - lp(b2)| = {abs(lp1 - lp2):.2e} (bridge out-projection zero-initialised)")
    with torch.no_grad():
        saved = [x.out.weight.detach().clone() for x in bb.bridge.xattn]
        for x in bb.bridge.xattn:
            x.out.weight.normal_(0.0, 0.02)
        lp1b = float(bb.generation_logprob([prompt], [comp], latent=b1))
        lp2b = float(bb.generation_logprob([prompt], [comp], latent=b2))
        for x, w in zip(bb.bridge.xattn, saved):
            x.out.weight.copy_(w)
    C.check("the latent path exists: with a non-zero bridge output the belief changes log p(completion)",
            abs(lp1b - lp2b) > 1e-3, f"|lp(b1) - lp(b2)| = {abs(lp1b - lp2b):.3f}")

    # ---- the proposer ---------------------------------------------------------------------------------------
    t1 = time.time()
    prop = BackboneProposer(model, k=4, max_new_tokens=256, seed=0)
    res = prop.propose(world.discover_tools(), "\n".join(i.text for i in ctx)[:2000], objective=task.objective,
                       latent=b1, validate=world.validate_action)
    st = res.stats
    C.info("proposer stats", {k: st[k] for k in ("n_texts", "n_items", "n_backbone_valid", "n_schema", "n_merged",
                                                  "by_provenance", "latent_conditioned")})
    C.info("proposer seconds", round(time.time() - t1, 1))
    C.check("every proposed candidate records its provenance", all(getattr(a, "provenance", "") for a in res.actions),
            f"{sorted({a.provenance for a in res.actions})}")
    C.check("the backbone contributed at least one valid candidate", st["n_backbone_valid"] >= 1,
            f"n_backbone_valid={st['n_backbone_valid']}")
    del model
    torch.cuda.empty_cache()
sys.exit(C.finish())
