"""Shared set-up of the Gate Q / SP-12 tests: a civilisation self-play campaign (crucible.multiagent.campaign).

Nothing here measures anything: it builds the participants (scripted demonstrations in CPU training worlds,
behaviour-cloning pre-training, the held-out families), the Tier 3 civilisation, and runs the campaign. Every file runs
its own campaign (official evidence is produced per file, from re-execution). Tier 3 unavailable -> the test fails.
`prepare` does everything before the civilisation once, so that one file can run the campaign in several
civilisations (profiles, seeds) from the same pre-trained policies (`campaign(..., prepared=...)`).
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
SIZES = {"full": {"demo_seeds": (21, 22, 23), "demo_hours": 48, "epochs": 12, "civ_hours": 96, "ho_hours": 24,
                  "semantic": 2, "semantic_every": 4},
         "short": {"demo_seeds": (21,), "demo_hours": 36, "epochs": 6, "civ_hours": 24, "ho_hours": 12,
                   "semantic": 1, "semantic_every": 2}}


def service():
    from crucible.semantic.service import SemanticService
    return SemanticService(MODEL, device="auto", max_batch=8, max_new_tokens=640)


def prepare(root: str, size: str = "full", *, heldout: bool = True, semantic: bool = True,
            heldout_rl: bool = True) -> dict:
    """Demonstrations, behaviour-cloning pre-training, the held-out families and the semantic service."""
    torch.set_num_threads(1)
    z = SIZES[size]
    reg = PolicyRegistry(root)
    demos = C.demonstrations(z["demo_seeds"], hours=z["demo_hours"], n=6)
    groups = C.pretrain(reg, C.STANDARD_GROUPS, demos, epochs=z["epochs"])
    ho = C.heldout_family(reg, demo_seeds=(31,), hours=z["ho_hours"], epochs=z["epochs"],
                          demo_hours=z["demo_hours"] * 0.75) if heldout else None
    # a second held-out family that never saw a demonstration (self-play from its random initialisation)
    ho_rl = C.heldout_family(reg, seed=4343, hours=2 * z["ho_hours"], source="self_play",
                             drive="reach") if heldout and heldout_rl else None
    svc = service() if semantic and z["semantic"] else None
    return {"registry": reg, "demos": demos, "pretrained": groups, "heldout": ho, "heldout_rl": ho_rl,
            "service": svc, "size": size, "root": root}


def campaign(root: str, size: str = "full", *, semantic: bool = True, heldout: bool = True, seed: int = TRAIN_SEED,
             historical_at: float | None = 0.5, store: bool = False, heldout_join_at: float | None = None,
             before_run=None, wrap=None, profile: str = PROFILE, prepared: dict | None = None) -> dict:
    torch.set_num_threads(1)
    z = SIZES[size]
    if prepared is None:
        prepared = prepare(root, size, heldout=heldout, semantic=semantic)
        reg, groups = prepared["registry"], prepared["pretrained"]
    else:
        # another civilisation from the same pre-trained policies: fresh copies, recorded in a registry of their own
        # (version ids restart from the pre-trained versions), with the pre-training records copied over
        reg = PolicyRegistry(os.path.join(root, f"{profile}-{seed}"))
        reg.versions.update({k: dict(v) for k, v in prepared["registry"].versions.items()})
        groups = {}
        for gid, g in prepared["pretrained"].items():
            c = g.clone(gid)
            c.frozen = False
            reg.register_group(c, world_id=reg.versions[reg.vid(c)]["world_id"], checkpoint=True)
            groups[gid] = c
    ho, svc = prepared["heldout"], prepared["service"]
    env = civ_env(profile, seed, extra_modules=("semantic_inference",) if svc is not None else ())
    assert is_civilisation(env.world), "the self-play world must be a Tier 3 civilisation"
    extra = {}
    if heldout_join_at:
        g = ho.clone("heldout_eval")
        g.frozen = True
        extra = {"extra_at": heldout_join_at, "extra_groups": {"heldout_eval": g},
                 "extra_members": [Member(f"ho{i}", "heldout", "ctrl:heldout", "heldout_eval") for i in range(2)]}
    run = C.civ_run(env, reg, groups, C.standard_members(semantic=z["semantic"] if svc is not None else 0,
                                                         semantic_every=z.get("semantic_every", 4)),
                    hours=z["civ_hours"], seed=seed, semantic_service=svc, historical_at=historical_at,
                    store_root=os.path.join(root, "experience") if store else None, before_run=before_run,
                    wrap=wrap, **extra)
    run.update(registry=reg, groups=groups, heldout=ho, heldout_rl=prepared["heldout_rl"], demos=prepared["demos"],
               service=svc, size=size, profile=profile, seed=seed)
    return run


def frozen_control(run: dict, *, seed: int | None = None, historical_at: float | None = 0.5) -> dict:
    """The same civilisation (profile, seed, members, semantic participants) played by the campaign's policies frozen
    as they were after pre-training (`run["initial"]`): what behaviour cloning alone brings to the world, to compare
    with what the learning campaign did (verifier: the Part 12 capabilities may come from pre-training, not from
    self-play learning)."""
    torch.set_num_threads(1)
    z = SIZES[run["size"]]
    svc = run.get("service")
    seed = run["seed"] if seed is None else seed
    groups = {}
    for gid, g in run["initial"].items():
        c = g.clone(gid)
        c.frozen = True
        groups[c.gid] = c
    env = civ_env(run["profile"], seed, extra_modules=("semantic_inference",) if svc is not None else ())
    assert is_civilisation(env.world)
    out = C.civ_run(env, PolicyRegistry(), groups,
                    C.standard_members(semantic=z["semantic"] if svc is not None else 0,
                                       semantic_every=z.get("semantic_every", 4)),
                    hours=z["civ_hours"], seed=seed, semantic_service=svc, historical_at=historical_at,
                    controls=False)
    out["groups"] = groups
    return out
