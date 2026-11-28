"""G3-core probe IMP-2: is the content-prediction target (predict/content.py) fixed, as documented?

predict/content.py:11-13 says the target uses "the frozen backbone's standardised semantic embedding ... a fixed
target, not one the learner can move". ``ContentTarget.__call__`` calls ``model.embed_texts``, i.e.
``SemanticBackbone.embed`` (nn/backbone.py:605), whose embedding is computed WITH the LoRA adapters (the embedding
cache is invalidated when adapter content changes, nn/backbone.py:424-459). The continual learner's slow update
trains those adapters (CL-03). This probe builds the debug model, computes the content target of fixed texts, then
changes the LoRA weights by a small step (as an accepted slow adapter step would) and recomputes it. It also checks
that no gradient reaches the adapters through the target (the learner cannot collapse it directly).

    PYTHONPATH=<freeze>/cam_live/src python probe_imp_content_target.py --out out/imp_content_target.json
"""
import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from cam_live.config import CamLiveConfig
    from cam_live.contracts import ExperienceItem, ItemKind
    from cam_live.nn.model import build_model
    from cam_live.predict.content import ContentTarget
    torch.manual_seed(0)
    cfg = CamLiveConfig.debug(seed=0)
    model = build_model(cfg)
    bb = model.backbone
    lora = bb.lora_parameters()
    tgt = ContentTarget(model.d_sem if hasattr(model, "d_sem") else bb.d_sem, seed=0)
    texts = ["the server answered with status 200 and three new members",
             "error: permission denied while writing the file",
             "observation: the page lists twelve repositories"]
    groups = [[ExperienceItem(kind=ItemKind.OBSERVATION, text=t, ts=0.0)] for t in texts]
    y0, _ = tgt(model, groups)
    y0b, _ = tgt(model, groups)
    # adapter steps of absolute size eps on every LoRA tensor (B starts at zero; a trained adapter has B != 0)
    rels = {}
    base = [p.detach().clone() for p in lora]
    for eps in (1e-3, 1e-2):
        with torch.no_grad():
            g = torch.Generator().manual_seed(1)
            for p, b0 in zip(lora, base):
                p.copy_(b0 + torch.randn(p.shape, generator=g).to(p.device, p.dtype) * eps)
        bb.mark_adapters_updated()
        y1, _ = tgt(model, groups)
        rels[str(eps)] = float((y1 - y0).norm() / y0.norm().clamp_min(1e-12))
    # gradient path from the target into the adapters
    for p in lora:
        p.grad = None
    y2, _ = tgt(model, groups)
    has_grad = bool(y2.requires_grad)
    rel = max(rels.values())
    out = {"probe": "imp_content_target", "lora_tensors": len(lora), "lora_rank": int(bb.cfg.lora_rank),
           "repeat_identical": bool(torch.equal(y0, y0b)),
           "relative_change_of_target_by_adapter_step_size": rels,
           "target_requires_grad": has_grad,
           "verdict": ("target MOVES with the adapters (not fixed)" if rel > 1e-6 else "target unchanged by the adapters")}
    print(json.dumps(out, indent=1))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
