"""Per-capability model routing and the selective-GPU policy of the shared semantic service.

**Routing** (MASTER_SPEC 3.2 class 4: "use the smallest model or combination of models that meets each capability
requirement"; DECISION_LOG D-10). `route(capability)` names the model for a decision of that capability:

* default: ``Qwen/Qwen3-0.6B`` (smallest effective model for content understanding, situation interpretation, tool
  choice, communication and incentive adaptation in SP-3.2b);
* escalation to ``Qwen/Qwen3-1.7B`` for ``programming``, ``negotiation`` and ``strategic_decisions``, where the
  SP-3.2b held-out results show the 0.6B model is not effective.

When an SP-3.2b report is present (`$CRUCIBLE_DATA/semantic/capbench/sp32b.json`, written by `capbench.run`), its
per-capability recommendation on held-out templates replaces this table entry by entry (`table()` says which
source each entry came from). `service(capability)` returns the process-wide `SemanticService` of the routed model,
so every participant routed to a model shares one weight copy.

**Selective GPU** (13.4). `device_for(model_id)` puts a model on the GPU only when the slot scheduler granted GPU
memory to this process (`CUDA_VISIBLE_DEVICES` non-empty) and the grant (`CRUCIBLE_SLOT_GPU_MB`, set by
`crucible.scale.slots`) holds the weights with room for activations; otherwise CPU (float32, or int8 dynamic
quantization for models of 1B parameters and more, which are too slow on CPU in float32). `limit_gpu_memory()`
caps this process's CUDA allocator at the granted MB, so a run cannot exceed its slot.
"""
from __future__ import annotations

import json
import os

DEFAULT = "Qwen/Qwen3-0.6B"
ESCALATION = "Qwen/Qwen3-1.7B"
TABLE = {"content_understanding": DEFAULT, "situation_interpretation": DEFAULT, "tool_choice": DEFAULT,
         "communication": DEFAULT, "incentive_adaptation": DEFAULT, "negotiation": ESCALATION,
         "programming": ESCALATION, "strategic_decisions": ESCALATION}
PARAMS = {"Qwen/Qwen3-0.6B": 596_049_920, "Qwen/Qwen3-1.7B": 1_720_574_976,
          "HuggingFaceTB/SmolLM2-135M-Instruct": 134_515_008, "HuggingFaceTB/SmolLM2-360M-Instruct": 361_821_120}
HEADROOM = 1.6          # weights x this must fit the GPU grant (activations, KV cache, allocator)


def report_path() -> str:
    return os.path.join(os.environ.get("CRUCIBLE_DATA", "/workspace/data/crucible"), "semantic", "capbench",
                        "sp32b.json")


def table() -> dict[str, dict]:
    """capability -> {model, source}: the SP-3.2b held-out recommendation where a report exists, else D-10."""
    out = {c: {"model": m, "source": "DECISION_LOG D-10 (SP-3.2b held-out results)"} for c, m in TABLE.items()}
    try:
        with open(report_path()) as fh:
            rep = json.load(fh)
        rec = rep.get("recommendation_held_out") or {}
        for cap, r in rec.items():
            model = r.get("model") if isinstance(r, dict) else r
            if cap in out and isinstance(model, str) and "/" in model:
                out[cap] = {"model": model, "source": report_path()}
    except (OSError, ValueError):
        pass
    return out


def route(capability: str) -> str:
    return table().get(capability, {"model": DEFAULT})["model"]


def granted_gpu_mb() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis in ("", "-1"):
        return 0
    try:
        return int(os.environ.get("CRUCIBLE_SLOT_GPU_MB", "0") or 0)
    except ValueError:
        return 0


def device_for(model_id: str, dtype_bytes: int = 2) -> tuple[str, str | None]:
    """(device, quantization) for a model under this process's slot."""
    params = PARAMS.get(model_id, 600_000_000)
    need_mb = params * dtype_bytes * HEADROOM / 2 ** 20
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    gmb = granted_gpu_mb()
    if cuda and gmb and need_mb <= gmb:
        return "cuda", None
    return "cpu", ("int8-dynamic" if params >= 1_000_000_000 else None)


def limit_gpu_memory() -> float | None:
    """Cap the CUDA allocator of this process at the slot's grant; returns the fraction set (None without a grant)."""
    gmb = granted_gpu_mb()
    if not gmb:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        frac = max(0.01, min(1.0, gmb / total))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        return frac
    except Exception:  # noqa: BLE001
        return None


_SERVICES: dict[str, object] = {}


def service(capability: str, **kw):
    """The shared service of the model routed for `capability` (one per model and process)."""
    from .service import SemanticService
    model = route(capability)
    s = _SERVICES.get(model)
    if s is None:
        dev, quant = device_for(model)
        if dev == "cuda":
            limit_gpu_memory()
        s = _SERVICES[model] = SemanticService(model, device=dev, quantization=quant, **kw)
    return s
