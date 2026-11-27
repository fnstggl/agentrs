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

    def __init__(self, routes: dict[str, str] | None = None, max_batch: int = 16, think: dict | None = None):
        self.routes = dict(routes or {})
        self.think = dict(think) if think is not None else None     # {capability: tokens}; None: the decision's
        self.max_batch = int(max_batch)
        # brain calls by kind; "think": choices that reasoned first; "think_generate": the generation batches those
        # choices sent to a service (SP-4.2b attributes every service generation batch to its caller)
        self.calls = {"choose": 0, "generate": 0, "think": 0, "think_generate": 0}
        self.requests: dict[str, int] = {}     # decisions served per capability (H.n1: which models do the work)

    def model_for(self, capability: str) -> str:
        """The model of a decision of `capability` (or of a separately routable use such as ``next_step``): an
        explicit route of this brain (a route given for a capability also covers its uses), else the routing
        decision."""
        from .. import routing
        r = self.routes.get(capability)
        if r is None and capability in routing.USES:
            r = self.routes.get(routing.USES[capability])
        return r or routing.route(capability)

    def service(self, capability: str):
        from .. import routing
        return routing.service_for_model(self.model_for(capability), max_batch=self.max_batch)

    def think_tokens(self, capability: str) -> int:
        """Reasoning budget before a choice of `capability`: only for models with a thinking mode (Qwen3)."""
        from .. import routing
        from ..participants import THINKING_MODELS
        explicit = capability in self.routes or (capability in routing.USES and
                                                 routing.USES[capability] in self.routes)
        k = self.think.get(capability, 0) if self.think is not None else (
            routing.think_tokens(capability) if not explicit else 0)
        return int(k) if k and any(f in self.model_for(capability).lower() for f in THINKING_MODELS) else 0

    def choose(self, ctx, capability: str, requests: list[dict]) -> list[int]:
        if not requests:
            return []
        self.calls["choose"] += 1
        self.requests[capability] = self.requests.get(capability, 0) + len(requests)
        svc = self.service(capability)
        k = self.think_tokens(capability)
        if k:
            # the model reasons in its own thinking mode (bounded), then names an option; when it names none, the
            # options are scored after its closed reasoning (`participants.select_options`, as SP-3.2b scores
            # "think"); both requests are recorded through the service like any other
            from ..participants import select_options
            if k > svc.think_bound:
                # the shared service's bound is fixed (never raised here: a request's recorded form must not depend
                # on what ran before in the process); a budget above it would be cut short silently
                raise ValueError(f"reasoning budget {k} of {capability} exceeds the thinking bound {svc.think_bound} "
                                 f"of the shared {svc.model_id} service")
            self.calls["think"] += 1

            def infer(kind, reqs, principals, purpose):
                if kind == "choose":
                    return svc.choose(ctx, reqs)
                self.calls["think_generate"] += 1
                return svc.generate(ctx, reqs)
            picks, _ = select_options(infer, [dict(r, think_tokens=k) for r in requests], [""] * len(requests))
            return [int(p) for p in picks]
        out = svc.choose(ctx, requests)
        return [int(o["choice"]) for o in out]

    def generation_calls(self) -> int:
        """Generation batches this brain sent to the shared services (writing, goals, next steps, and the reasoning of
        choices that reason first): SP-4.2b counts every other generation batch as an ordinary action's."""
        return int(self.calls["generate"]) + int(self.calls["think_generate"])

    def generate(self, ctx, capability: str, requests: list[dict]) -> list[str]:
        if not requests:
            return []
        self.calls["generate"] += 1
        self.requests[capability] = self.requests.get(capability, 0) + len(requests)
        out = self.service(capability).generate(ctx, requests)
        return [str(o.get("text") or "") for o in out]

    def describe(self) -> dict:
        from .. import routing
        return {"kind": self.kind, "routes": {c: self.model_for(c) for c in routing.capabilities()},
                "think_tokens": {c: self.think_tokens(c) for c in routing.capabilities() if self.think_tokens(c)}}


class AblatedBrain:
    """No language model: uniform random choices, and words drawn at random from the prompt (see module doc)."""
    kind = "ablated"

    def __init__(self):
        self.calls = {"choose": 0, "generate": 0}

    def generation_calls(self) -> int:
        return 0                            # no model: nothing is generated

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
