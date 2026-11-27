"""G-nn probe: trainable_state() -> fresh process -> load_trainable_state() is bit-exact (tiny backbone, CPU); the
frozen LM is not in the checkpoint; what a different architecture does on load.

    .venv/bin/python cam_live/docs/verification/repro/nn/probe_saveload_cpu.py save DIR
    .venv/bin/python cam_live/docs/verification/repro/nn/probe_saveload_cpu.py load DIR     (a second, fresh process)
"""
import copy
import os
import sys

import _common  # noqa: F401
from _common import check, finish, info

import torch
from torch import nn

from cam_live.config import CamLiveConfig
from cam_live.contracts import ActionSpec, ExperienceItem, ItemKind, OutcomeKind, OutcomeQuery, Quantity
from cam_live.nn.api import GROUP_ADAPTERS, GROUP_EXPERIENTIAL, GROUP_HEADS
from cam_live.nn.model import build_model

mode, DIR = sys.argv[1], sys.argv[2]
CTX = [[ExperienceItem(kind=ItemKind.OBSERVATION, text="the page loaded", ts=10.0, quantities=[Quantity("visits", 100.0)]),
        ExperienceItem(kind=ItemKind.OUTCOME, text="three sign ups", ts=3600.0, quantities=[Quantity("signups", 3.0)])],
       [ExperienceItem(kind=ItemKind.FAILURE, text="timeout", ts=5.0)]]
ACTS = [[ActionSpec("a1", "browser.click", arguments={"ref": 3}), ActionSpec("a2", "internal.wait")], [ActionSpec("b1", "api.call")]]


def outputs(m, lat):
    with torch.no_grad():
        tb = m.tokenize(CTX)
        o = m.core_forward(tb)
        o2 = m.core_forward(m.tokenize([[CTX[0][1]], [CTX[1][0]]]), o.state)
        e_a, _ = m.encode_actions(ACTS)
        e_g = m.encode_goals(["goal one", "goal two"])
        e_q = m.encode_queries([[OutcomeQuery("q", "signups", kind=OutcomeKind.COUNT, horizon_s=60.0)]] * 2)[0]
        z = m.decision_latent(o.b, e_a, e_g)
        return {"emb": m.embed_texts(["a text", "another"]), "x": tb.x, "h": o.h, "b": o.b, "b2": o2.b, "slots2": o2.state.slots, "z": z,
                "phi": m.features(z, e_q), "head": m.head("w")(o.b), "lp": m.generation_logprob(["a prompt", "b"], ["the completion", "x y"], latent=lat),
                "logits": m.backbone.next_token_logits(["a prompt"], latent=lat[:1]),
                "text": m.generate(["a prompt"], latent=lat[:1], max_new_tokens=6, seed=3)}


m = build_model(CamLiveConfig.debug())
m.register_head("w", nn.Linear(m.d_model, 3))
if mode == "save":
    opt = torch.optim.Adam([p for g in (GROUP_ADAPTERS, GROUP_EXPERIENTIAL, GROUP_HEADS) for p in m.param_groups()[g]], lr=1e-2)
    lat = torch.randn(2, m.d_model, generator=torch.Generator().manual_seed(4))
    for _ in range(3):
        opt.zero_grad()
        o = m.core_forward(m.tokenize(CTX, with_grad=True))
        (o.b.pow(2).sum() + m.head("w")(o.b).sum() - m.generation_logprob(["a prompt", "b"], ["the completion", "x y"], latent=o.b).sum()).backward()
        opt.step()
    m.mark_adapters_updated()
    st = m.trainable_state()
    os.makedirs(DIR, exist_ok=True)
    torch.save({"state": st, "ref": outputs(m, lat), "lat": lat, "hashes": m.group_hashes(), "sig": m.arch_signature()}, os.path.join(DIR, "tiny_state.pt"))
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    check("no frozen LM tensor in trainable_state()", not (set(st) & frozen), f"{len(st)} tensors; frozen tensors in the model: {len(frozen)}")
    check("every trainable parameter in trainable_state()", {n for n, p in m.named_parameters() if p.requires_grad} <= set(st))
else:
    blob = torch.load(os.path.join(DIR, "tiny_state.pt"), weights_only=False)
    check("fresh process differs before loading", m.group_hashes()[GROUP_EXPERIENTIAL] != blob["hashes"][GROUP_EXPERIENTIAL])
    rep = m.load_trainable_state(blob["state"], strict=True)
    out = outputs(m, blob["lat"])
    bad = [k for k, v in blob["ref"].items() if not ((out[k] == v) if k == "text" else torch.equal(out[k], v))]
    check("fresh process: all outputs bit-identical after load_trainable_state(strict=True)", not bad and m.group_hashes() == blob["hashes"], str(bad))
    # architecture mismatches
    cfg2 = CamLiveConfig.debug()
    cfg2.backbone.lora_rank = 4
    m2 = build_model(cfg2)
    m2.register_head("w", nn.Linear(m2.d_model, 3))
    try:
        m2.load_trainable_state(blob["state"], strict=True)
        check("strict load into a model with another LoRA rank raises", False)
    except RuntimeError:
        check("strict load into a model with another LoRA rank raises", True)
    rep = m2.load_trainable_state(blob["state"], strict=False)
    check("non-strict load reports the mismatched tensors and does not load them", len(rep["mismatched"]) > 0, f"{len(rep['mismatched'])} mismatched")
    # same shapes, different function: LoRA alpha, tap layers, text truncation length
    for name, mut in (("lora_alpha 2 -> 8", lambda c: setattr(c.backbone, "lora_alpha", 8.0)),
                      ("tap_layers (0.5, 0.75, 1.0) -> (0.5, 1.0, 1.0)", lambda c: setattr(c.backbone, "tap_layers", (0.5, 1.0, 1.0))),
                      ("max_text_tokens 48 -> 8", lambda c: setattr(c.backbone, "max_text_tokens", 8)),
                      ("seed 0 -> 1 (another frozen tiny LM)", lambda c: setattr(c, "seed", 1))):
        c3 = CamLiveConfig.debug()
        mut(c3)
        m3 = build_model(c3)
        m3.register_head("w", nn.Linear(m3.d_model, 3))
        try:
            m3.load_trainable_state(blob["state"], strict=True)
            loaded = True
        except RuntimeError:
            loaded = False
        o3 = outputs(m3, blob["lat"]) if loaded else None
        same = loaded and all(torch.equal(o3[k], blob["ref"][k]) for k in ("b", "phi", "lp"))
        import json
        sig_differs = json.loads(json.dumps(m3.arch_signature(), default=str)) != json.loads(json.dumps(blob["sig"], default=str))
        check(f"different architecture [{name}]: either the strict load fails, or arch_signature() differs, or outputs are unchanged",
              (not loaded) or sig_differs or same,
              f"strict load accepted={loaded}, outputs identical={same}, arch_signature differs={sig_differs}")
sys.exit(finish())
