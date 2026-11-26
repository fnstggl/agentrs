"""What decides for a semantic participant: the shared small language models (`LMBrain`), or - for the ablation that
shows the behaviour comes from the model and not from the runtime around it (SP-3.2d) - a brain without a model
(`AblatedBrain`).

Both receive the same requests and return the same shapes:

* ``choose(ctx, capability, requests) -> [index]`` - pick one option per request ``{messages, choices}``;
* ``generate(ctx, capability, requests) -> [text]`` - bounded text per request ``{messages, max_new_tokens, ...}``.

`LMBrain` routes every request to the smallest model effective for its capability (`crucible.semantic.routing`,
SP-3.2b) and runs it on the process-wide shared service of that model (one weight copy per model, MASTER_SPEC 13.1),
batched; every call is recorded in the world through the service (`crucible.substrate.inference.recorded`), so
replay and regeneration never run a model. `AblatedBrain` chooses uniformly at random and writes words drawn at
random from the request's own prompt (the same vocabulary, no understanding); its draws come from the transaction's
seeded RNG, so its runs are as reproducible as the model's.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{1,20}")


class LMBrain:
    kind = "lm"

    def __init__(self, routes: dict[str, str] | None = None, max_batch: int = 16):
        self.routes = dict(routes or {})
        self.max_batch = int(max_batch)
        self.calls = {"choose": 0, "generate": 0}

    def model_for(self, capability: str) -> str:
        from .. import routing
        return self.routes.get(capability) or routing.route(capability)

    def service(self, capability: str):
        from .. import routing
        model = self.model_for(capability)
        if model in self.routes.values() and model != routing.route(capability):
            return routing.service_for_model(model, max_batch=self.max_batch)
        return routing.service(capability, max_batch=self.max_batch)

    def choose(self, ctx, capability: str, requests: list[dict]) -> list[int]:
        if not requests:
            return []
        self.calls["choose"] += 1
        out = self.service(capability).choose(ctx, requests)
        return [int(o["choice"]) for o in out]

    def generate(self, ctx, capability: str, requests: list[dict]) -> list[str]:
        if not requests:
            return []
        self.calls["generate"] += 1
        out = self.service(capability).generate(ctx, requests)
        return [str(o.get("text") or "") for o in out]

    def describe(self) -> dict:
        from .. import routing
        return {"kind": self.kind, "routes": {c: self.model_for(c) for c in routing.TABLE}}


class AblatedBrain:
    """No language model: uniform random choices, and words drawn at random from the prompt (see module doc)."""
    kind = "ablated"

    def __init__(self):
        self.calls = {"choose": 0, "generate": 0}

    def choose(self, ctx, capability: str, requests: list[dict]) -> list[int]:
        self.calls["choose"] += 1
        rng = ctx.rng("semp.ablated.choose", capability, len(requests))
        return [int(rng.integers(0, max(1, len(r["choices"])))) for r in requests]

    def generate(self, ctx, capability: str, requests: list[dict]) -> list[str]:
        self.calls["generate"] += 1
        rng = ctx.rng("semp.ablated.generate", capability, len(requests))
        out = []
        for r in requests:
            words = _WORD.findall(" ".join(m.get("content", "") for m in r.get("messages", [])[1:]))
            n = max(1, min(int(r.get("max_new_tokens", 16)) // 2, 60))
            out.append(" ".join(words[int(i)] for i in rng.integers(0, len(words), n)) if words else "")
        return out

    def describe(self) -> dict:
        return {"kind": self.kind}


def make(kind: str = "lm", **kw):
    if kind == "lm":
        return LMBrain(**kw)
    if kind == "ablated":
        return AblatedBrain()
    raise ValueError(f"unknown brain {kind!r} (lm, ablated)")
