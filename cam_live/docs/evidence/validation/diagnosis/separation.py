"""Can the policy tell the candidates apart? Action-embedding and decision-latent separation of the LabWorld
interventions (every channel x effort) for a fresh agent, debug or real configuration.

Reports, over the 12 ``lab.act`` settings of an F1 world in its first context: the mean / min pairwise cosine distance of
the action embeddings ``e_a`` (backbone text embedding of the rendered action through the action encoder) and of the
decision latents ``z``; the spread (std) of the policy's logits and of its probabilities; and how much one gradient
step of a fixed size on the logit of one candidate moves the probabilities of the others (how separable they are
for the policy head). ``--real`` takes the GPU lock.

    .venv/bin/python cam_live/docs/evidence/validation/diagnosis/separation.py --out OUT.json [--real]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args(argv)
    import contextlib
    import torch
    from cam_live.agent import CamLiveAgent, observation_item, tools_item
    from cam_live.contracts import ExperienceItem, ItemKind, SimClock, Trust
    from cam_live.eval.envs import make_world
    from cam_live.predict.candidates import schema_candidates
    from cam_live.validation import staging_config
    lock = contextlib.nullcontext()
    if a.real:
        from cam_live.nn.gpu import gpu_lock
        lock = gpu_lock(min_free_gb=14.0)
    with lock:
        cfg = staging_config(None, a.real, 0)
        clock = SimClock()
        agent = CamLiveAgent(cfg, tempfile.mkdtemp(), clock=clock)
        world, _v, task = make_world("F1", 10_000, clock, target=12)
        now = clock.now()
        items = [ExperienceItem(kind=ItemKind.GOAL, text=task.objective, ts=now, ref="g", trust=Trust.OPERATOR),
                 tools_item(world.discover_tools(), now, "x")] + [observation_item(o) for o in world.observe()]
        acts = [c for c in schema_candidates(world.discover_tools(), per_tool=12, id_prefix="s")
                if c.operation == "lab.act"]
        m = agent.model
        with torch.no_grad():
            b = m.core_forward(m.tokenize([items]), None).b
            e_a, _ = m.encode_actions([acts])
            e_g = m.encode_goals([task.objective])
            z = m.decision_latent(b, e_a, e_g)
            mask = torch.ones(1, len(acts), dtype=torch.bool, device=z.device)
            out = agent.heads.policy(z, mask, b)

        def dist(x):
            x = torch.nn.functional.normalize(x[0].float(), dim=-1)
            s = x @ x.T
            n = s.shape[0]
            off = s[~torch.eye(n, dtype=torch.bool, device=s.device)]
            return {"mean_cosine_distance": float((1 - off).mean()), "min_cosine_distance": float((1 - off).min())}
        # separability: one SGD step of size 0.1 on -log pi(candidate 0) w.r.t. the policy head only
        pol = agent.heads.policy
        params = [p for p in pol.parameters()]
        before = out.probs[0].detach().clone()
        lp = pol(z.detach(), mask, b.detach()).log_prob(torch.zeros(1, dtype=torch.long, device=z.device))
        g = torch.autograd.grad(lp.sum(), params, allow_unused=True)
        with torch.no_grad():
            for p, gi in zip(params, g):
                if gi is not None:
                    p.add_(0.1 * gi)
            after = pol(z, mask, b).probs[0]
            for p, gi in zip(params, g):
                if gi is not None:
                    p.sub_(0.1 * gi)
        res = {"real_model": a.real, "backbone": cfg.backbone.name, "n_candidates": len(acts),
               "renders": [c.render()[:80] for c in acts],
               "e_a": dist(e_a), "z": dist(z),
               "policy_logit_std": float(out.raw_logits[0].std()), "policy_prob_max": float(out.probs[0].max()),
               "policy_prob_min": float(out.probs[0].min()),
               "one_step_on_candidate0": {"p0_before": float(before[0]), "p0_after": float(after[0]),
                                          "others_mean_change": float((after[1:] - before[1:]).mean()),
                                          "others_max_abs_change": float((after[1:] - before[1:]).abs().max())}}
        agent.close()
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "renders"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
