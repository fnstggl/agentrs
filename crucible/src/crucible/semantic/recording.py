"""Kernel module `semantic_inference`: records the shared model's outputs for decisions made outside a transaction.

Semantic participants are `crucible.agents.base.Policy` objects: they decide between world transactions and act
through ordinary tools. Their model calls are recorded in the world anyway, so that a run can be replayed and
regenerated without the model and the provenance of every decision is in the hash chain:

    out = record_inference(world, service, "generate" | "choose", requests, principals)

stages the full requests in ``world.runtime`` (never in the event log), then runs the operator-only tool
``admin.semantic_infer`` through ``world.system_call``. Its handler runs the batch through
`crucible.semantic.service.recorded` (``ctx.external``): the recorded outcome holds ``{model_id, model_hash,
inputs_digest, output}``. The command event carries only compact request descriptors (SHA-256 of each request,
generation bounds, seed, the participant it serves). During regeneration the staged requests are absent and the
recorded outputs are returned: the model is never run again.

The handler also emits ``semantic.inference`` (internal visibility: participants never see each other's prompts or
outputs) with the model id and weights hash, the participants served, token counts and a digest of the outputs.
The tool is on host ``admin``: refused for, and hidden from, every in-world actor.
"""
from __future__ import annotations

import hashlib
import json

from ..core import schema as S
from ..core.errors import Forbidden, ProtocolError
from ..core.kernel import SYSTEM, Module, applier, endpoint, register_module

STAGE_KEY = "semantic.staged"
_DESC = S.obj({"digest": S.s_str(maxLength=64), "max_new_tokens": {"type": ["integer", "null"]},
               "temperature": {"type": ["number", "null"]}, "seed": {"type": ["integer", "null"]},
               "n_choices": {"type": ["integer", "null"]}, "principal": S.s_str(maxLength=80),
               "purpose": S.s_str(maxLength=40)}, ["digest"])


@register_module
class SemanticInference(Module):
    """Operator-side recording of shared-model inference (see the module docstring)."""
    name = "semantic_inference"
    host = "admin"
    version = "1"

    def install(self, world):
        world.state.declare_table("semantic_inference", index=("model_hash",))

    @endpoint("semantic_infer", S.obj({
        "model_id": S.s_str(maxLength=120), "model_hash": S.s_str(maxLength=64),
        "kind": S.s_enum(["generate", "choose"]),
        "requests": S.s_arr(_DESC, maxItems=256, minItems=1)}, ["model_id", "model_hash", "kind", "requests"]),
        cost_s=0)
    def ep_infer(self, ctx, model_id, model_hash, kind, requests):
        """Operator only: run (or, during regeneration, return the recorded outcome of) one batch of shared-model
        requests staged by `record_inference`."""
        if ctx.actor != SYSTEM:
            raise Forbidden("semantic inference is recorded by the simulator operator only")
        from .service import recorded, service_for

        def run():
            staged = ctx.world.runtime.get(STAGE_KEY) or {}
            full = []
            for d in requests:
                r = staged.get(d["digest"])
                if r is None:
                    raise ProtocolError("semantic request not staged and no recorded outcome (cannot run the model "
                                        "from the log)")
                full.append(r)
            svc = ctx.world.runtime.get("semantic.services", {}).get(model_id) or service_for(model_id)
            if svc.model_hash() != model_hash:
                raise ProtocolError(f"loaded weights of {model_id} do not match the requested hash")
            return svc.run(kind, full)
        out = recorded(ctx, f"semantic.{kind}", model_id, model_hash, requests, run)
        n_prompt = sum(int(o.get("prompt_tokens", 0)) for o in out)
        n_gen = sum(int(o.get("gen_tokens", 0)) for o in out)
        ctx.emit("semantic.inference", {
            "model_id": model_id, "model_hash": model_hash, "kind": kind, "n": len(requests),
            "principals": sorted({d.get("principal", "") for d in requests} - {""}),
            "prompt_tokens": n_prompt, "gen_tokens": n_gen,
            "outputs_digest": hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()}, vis="internal")
        return {"outputs": out}

    @applier("semantic.inference")
    def ap_inference(self, ev):
        p = ev.payload
        rec = self.world.state.get("semantic_inference", p["model_hash"])
        if rec is None:
            self.world.state.insert("semantic_inference", p["model_hash"], {
                "model_hash": p["model_hash"], "model_id": p["model_id"], "batches": 1, "requests": p["n"],
                "prompt_tokens": p["prompt_tokens"], "gen_tokens": p["gen_tokens"], "principals": list(p["principals"])})
        else:
            self.world.state.patch("semantic_inference", p["model_hash"], batches=rec["batches"] + 1,
                                   requests=rec["requests"] + p["n"],
                                   prompt_tokens=rec["prompt_tokens"] + p["prompt_tokens"],
                                   gen_tokens=rec["gen_tokens"] + p["gen_tokens"],
                                   principals=sorted(set(rec["principals"]) | set(p["principals"])))


def record_inference(world, service, kind: str, requests: list[dict], principals: list[str] | None = None,
                     purpose: str = "") -> list[dict]:
    """Run a batch on `service` as a recorded operator command of `world` (see the module docstring). The world must
    include the `semantic_inference` module."""
    if "semantic_inference" not in world.modules:
        raise ProtocolError("the world has no semantic_inference module (add it with extra_modules)")
    reqs = [service._norm(r, choices=(kind == "choose")) for r in requests]
    desc = service.descriptors(reqs)
    for i, d in enumerate(desc):
        if principals:
            d["principal"] = principals[i]
        if purpose:
            d["purpose"] = purpose[:40]
    stage = world.runtime.setdefault(STAGE_KEY, {})
    world.runtime.setdefault("semantic.services", {})[service.model_id] = service
    for d, r in zip(desc, reqs):
        stage[d["digest"]] = r
    try:
        res = world.system_call("admin.semantic_infer", {"model_id": service.model_id,
                                                         "model_hash": service.model_hash(), "kind": kind,
                                                         "requests": desc})
    finally:
        for d in desc:
            stage.pop(d["digest"], None)
    return res["outputs"]
