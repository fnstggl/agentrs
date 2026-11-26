"""Shared set-up of the Gate Q / SP-12 tests: a civilisation self-play campaign (crucible.multiagent.campaign).

Nothing here measures anything: it builds the participants (scripted demonstrations in CPU training worlds,
behaviour-cloning pre-training, the held-out family), the Tier 3 civilisation, and runs the campaign. Every file runs
its own campaign (official evidence is produced per file, from re-execution). Tier 3 unavailable -> the test fails.
"""
import os

import torch

from crucible.multiagent import campaign as C
from crucible.multiagent.harness import Member, civ_env, is_civilisation
from crucible.multiagent.registry import PolicyRegistry

MODEL = os.environ.get("CRUCIBLE_SEMANTIC_MODEL", "Qwen/Qwen3-0.6B")
PROFILE = "balanced"
TRAIN_SEED = 7

# campaign sizes: "full" (Gate Q, SP-12.0) and "short" (the other SP-12 files)
SIZES = {"full": {"demo_seeds": (21, 22, 23), "demo_hours": 48, "epochs": 8, "civ_hours": 72, "ho_hours": 24,
                  "semantic": 2},
         "short": {"demo_seeds": (21,), "demo_hours": 36, "epochs": 3, "civ_hours": 24, "ho_hours": 12,
                   "semantic": 1}}


def service():
    from crucible.semantic.service import SemanticService
    return SemanticService(MODEL, device="auto", max_batch=8, max_new_tokens=640)


def campaign(root: str, size: str = "full", *, semantic: bool = True, heldout: bool = True, seed: int = TRAIN_SEED,
             historical_at: float | None = 0.5, store: bool = False, heldout_join_at: float | None = None,
             before_run=None, wrap=None) -> dict:
    torch.set_num_threads(1)
    z = SIZES[size]
    reg = PolicyRegistry(root)
    demos = C.demonstrations(z["demo_seeds"], hours=z["demo_hours"], n=6)
    groups = C.pretrain(reg, C.STANDARD_GROUPS, demos, epochs=z["epochs"])
    ho = C.heldout_family(reg, demo_seeds=(31,), hours=z["ho_hours"], epochs=z["epochs"],
                          demo_hours=z["demo_hours"] * 0.75) if heldout else None
    svc = service() if semantic and z["semantic"] else None
    env = civ_env(PROFILE, seed, extra_modules=("semantic_inference",) if svc is not None else ())
    assert is_civilisation(env.world), "the self-play world must be a Tier 3 civilisation"
    extra = {}
    if heldout_join_at:
        g = ho.clone("heldout_eval")
        g.frozen = True
        extra = {"extra_at": heldout_join_at, "extra_groups": {"heldout_eval": g},
                 "extra_members": [Member(f"ho{i}", "heldout", "ctrl:heldout", "heldout_eval") for i in range(2)]}
    run = C.civ_run(env, reg, groups, C.standard_members(semantic=z["semantic"] if svc is not None else 0),
                    hours=z["civ_hours"], seed=seed, semantic_service=svc, historical_at=historical_at,
                    store_root=os.path.join(root, "experience") if store else None, before_run=before_run,
                    wrap=wrap, **extra)
    run.update(registry=reg, groups=groups, heldout=ho, demos=demos, service=svc, size=size)
    return run
