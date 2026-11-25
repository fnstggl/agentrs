"""Recorded inference (Crucible 5.0, contract 1.7; MASTER_SPEC Part 2.3).

Neural inference, media rendering and other computations that are not bit-reproducible run through
`recorded(ctx, kind, model_id, model_hash, inputs, fn)`. The first time, `fn()` runs and the outcome is recorded in
the transaction's external-outcome record (`ctx.reserve` / `ctx.fill`, kernel event `ext`) as

    {"model_id", "model_hash", "inputs_digest", "inputs_ref"?, "sampling"?, "output" | "output_ref", "output_kind"}

under kind `infer.<kind>`. Regeneration (`World.regenerate`) hands the recorded outcome back instead of running
`fn`, after checking that the model id, model hash and inputs digest are the ones recorded (a mismatch means the
world diverged: ProtocolError). Replay never runs handlers at all. So a recorded trajectory replays and regenerates
without rerunning inference, and anything that depended on the sampled decision follows exactly.

Array outputs (one forward pass for a whole cohort, embeddings) are not put into the event: they are stored as a
content-addressed array blob (`artifacts.put_array_blob`, needs the `artifacts` module) and the record holds the REF.
`store_inputs=True` stores array inputs the same way ("inputs or their reproducible references").

`weights_hash(module_or_state_dict)` is the SHA-256 over the sorted parameter tensors (name, dtype, shape, bytes):
the model version recorded with every outcome.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterable

import numpy as np

from ..core.canonical import canonical_bytes, to_jsonable
from ..core.errors import ProtocolError
from ..core.kernel import Ctx

PREFIX = "infer."


def _np(x: Any) -> Any:
    """torch tensors -> numpy (recursively); everything else unchanged."""
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        t = x.detach().cpu()
        try:
            return t.numpy()
        except TypeError:       # bfloat16 and other dtypes numpy lacks: hash their raw bytes
            return t.contiguous().reshape(-1).view(__import__("torch").uint8).numpy()
    if isinstance(x, dict):
        return {k: _np(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_np(v) for v in x]
    return x


def inputs_digest(inputs: Any) -> str:
    """SHA-256 of the canonical encoding of the inputs (arrays by dtype, shape and content)."""
    return hashlib.sha256(canonical_bytes(_np(inputs))).hexdigest()


def _arrays_of(x: Any) -> tuple[str, dict] | None:
    if isinstance(x, np.ndarray):
        return "array", {"a": x}
    if isinstance(x, dict) and x and all(isinstance(v, np.ndarray) for v in x.values()) and \
            all(isinstance(k, str) for k in x):
        return "arrays", dict(x)
    return None


def recorded(ctx: Ctx, kind: str, model_id: str, model_hash: str, inputs: Any, fn: Callable[[], Any], *,
             sampling: dict | None = None, store_inputs: bool = False) -> Any:
    """Run `fn()` once and record its outcome with the model identity and the inputs digest; during regeneration
    return the recorded outcome without running `fn`. Returns JSON-native data, or read-only numpy arrays when `fn`
    returned an array / a dict of arrays (stored as an array blob)."""
    from . import artifacts
    if not isinstance(kind, str) or not kind or not isinstance(model_id, str) or not isinstance(model_hash, str):
        raise ProtocolError("recorded() needs kind, model_id and model_hash strings")
    inp = _np(inputs)
    dig = inputs_digest(inp)
    inputs_ref = None
    if store_inputs:
        arrs = _arrays_of(inp)
        if arrs is None:
            raise ProtocolError("store_inputs needs an array or a dict of arrays")
        inputs_ref = artifacts.put_array_blob(ctx, arrs[1])
    samp = to_jsonable(sampling) if sampling is not None else None
    slot, rec = ctx.reserve()
    if rec is not None:
        if rec.get("model_id") != model_id or rec.get("model_hash") != model_hash or rec.get("inputs_digest") != dig:
            raise ProtocolError(f"recorded inference {kind} diverged: recorded {rec.get('model_id')}@"
                                f"{str(rec.get('model_hash'))[:12]} inputs {str(rec.get('inputs_digest'))[:12]}, now "
                                f"{model_id}@{model_hash[:12]} inputs {dig[:12]}")
        if "output_ref" in rec:
            h = artifacts.parse_ref(rec["output_ref"])
            data = artifacts._read_bytes(ctx.world, h)
            if data is None:
                raise ProtocolError(f"recorded inference output {rec['output_ref']} is not available")
            arrays = artifacts.npz_arrays(data)
            ref = artifacts.put_array_blob(ctx, arrays)    # re-records the same artifact.stored event
            if ref != rec["output_ref"]:
                raise ProtocolError("recorded inference output blob does not match its reference")
            ctx.fill(slot, PREFIX + kind, rec)
            return arrays["a"] if rec.get("output_kind") == "array" else arrays
        ctx.fill(slot, PREFIX + kind, rec)
        return to_jsonable(rec["output"])
    out = _np(fn())
    record = {"model_id": model_id, "model_hash": model_hash, "inputs_digest": dig}
    if inputs_ref is not None:
        record["inputs_ref"] = inputs_ref
    if samp is not None:
        record["sampling"] = samp
    arrs = _arrays_of(out)
    if arrs is not None:
        okind, arrays = arrs
        for k, a in arrays.items():
            if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
                raise ProtocolError(f"inference output {k} has non-finite values")
        ref = artifacts.put_array_blob(ctx, arrays)
        record["output_ref"] = ref
        record["output_kind"] = okind
        ctx.fill(slot, PREFIX + kind, record)
        stored = artifacts.get_array_blob(ctx.world, ref)
        return stored["a"] if okind == "array" else stored
    try:
        value = to_jsonable(out)
        canonical_bytes(value)
    except (TypeError, ValueError):
        raise ProtocolError(f"inference output of {kind} is not JSON-native or arrays") from None
    record["output"] = value
    record["output_kind"] = "json"
    ctx.fill(slot, PREFIX + kind, record)
    return to_jsonable(value)


def records(events: Iterable) -> list[dict]:
    """Every recorded inference outcome in an event log (Event objects or dicts), in log order:
    [{seq, key, kind, model_id, model_hash, inputs_digest, ...}]."""
    out = []
    for e in events:
        d = e if isinstance(e, dict) else e.to_dict()
        p = d["payload"]
        if d["type"] == "ext":
            outs, key = p.get("outcomes") or [], p.get("key")
        elif d["type"] in ("cmd.rejected", "proc.failed") and p.get("ext"):
            outs, key = p["ext"], p.get("ext_key")
        else:
            continue
        for o in outs:
            if o and str(o.get("kind", "")).startswith(PREFIX):
                out.append({"seq": d["seq"], "key": key, "kind": o["kind"][len(PREFIX):], **o["out"]})
    return out


def weights_hash(module_or_state_dict) -> str:
    """SHA-256 (64 hex) over the sorted parameter tensors of a torch module / state dict (or a dict of numpy
    arrays): name, dtype, shape and raw bytes of each."""
    sd = module_or_state_dict.state_dict() if hasattr(module_or_state_dict, "state_dict") else module_or_state_dict
    h = hashlib.sha256()
    for name in sorted(sd):
        t = sd[name]
        if hasattr(t, "detach"):
            import torch
            tt = t.detach().cpu().contiguous()
            dtype = str(tt.dtype).replace("torch.", "")
            shape = tuple(tt.shape)
            raw = tt.reshape(-1).view(torch.uint8).numpy().tobytes() if tt.numel() else b""
        else:
            a = np.ascontiguousarray(np.asarray(t))
            dtype, shape, raw = str(a.dtype), a.shape, a.tobytes()
        h.update(f"{name}\0{dtype}\0{list(shape)}\0".encode())
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()
