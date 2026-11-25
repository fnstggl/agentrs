"""Common encoder machinery: the Encoder base class (identity, version, preprocessing hash, model hash) and the
registry of the five modality encoders."""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np

from . import models as M

CODE_VERSION = "1"          # bump when an encoder's computation changes in a way not captured by its preprocessing


def canonical_json(x) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()


class Encoder:
    """A modality encoder: bytes of one artifact -> float32 [dim] vector, computed by pretrained models."""
    modality: str = ""
    id: str = ""
    roles: tuple[str, ...] = ()
    dim: int = 0
    preprocessing: dict = {}
    batch_size: int = 16

    # identity ------------------------------------------------------------------------------------------------------
    def version(self) -> str:
        return "+".join(M.SPECS[r].revision[:12] for r in self.roles) + f".c{CODE_VERSION}"

    def preprocessing_hash(self) -> str:
        return hashlib.sha256(canonical_json({"modality": self.modality, "id": self.id,
                                              "preprocessing": self.preprocessing})).hexdigest()[:16]

    def model_hash(self) -> str:
        """SHA-256 over the weights files of every model this encoder runs (their hub LFS SHA-256s) and the encoder
        version: identifies exactly which weights produced a recorded vector, without loading the models."""
        return hashlib.sha256(canonical_json({"version": self.version(),
                                              "weights": {r: M.weights_file_sha256(r) for r in self.roles}})).hexdigest()

    def params(self) -> int:
        return int(sum(M.load(r).params for r in self.roles))

    def info(self) -> dict:
        return {"id": self.id, "version": self.version(), "dim": int(self.dim), "params": self.params(),
                "preprocessing_hash": self.preprocessing_hash(), "modality": self.modality,
                "models": {r: {"repo": M.SPECS[r].repo, "revision": M.SPECS[r].revision,
                               "params": M.load(r).params, "weights_sha256": M.weights_file_sha256(r)}
                           for r in self.roles},
                "model_hash": self.model_hash(), "preprocessing": self.preprocessing,
                "device": M.device()}

    # computation ---------------------------------------------------------------------------------------------------
    def embed(self, items: Sequence[bytes]) -> np.ndarray:
        """[n, dim] float32 for n artifacts' bytes. Subclasses implement it with real model forward passes."""
        raise NotImplementedError


REGISTRY: dict[str, Encoder] = {}


def register(enc: Encoder) -> Encoder:
    REGISTRY[enc.modality] = enc
    return enc


def recorded_json(ctx, kind: str, model_id: str, model_hash: str, inputs, fn) -> dict:
    """Run `fn() -> JSON dict` once and record it with the model identity and an inputs digest; during regeneration
    the recorded dict comes back without running `fn`. Uses `crucible.substrate.inference.recorded` (INTERFACE_CONTRACTS
    1.7) when the substrate is installed, else the kernel's `ctx.external` with the same record shape under kind
    `infer.<kind>` (what `substrate.inference.records` lists)."""
    try:
        from ...substrate import inference as _inf
        recorded = getattr(_inf, "recorded", None)
    except ImportError:
        recorded = None
    if recorded is not None:
        return recorded(ctx, kind, model_id, model_hash, inputs, fn)
    digest = hashlib.sha256(canonical_json(inputs)).hexdigest()

    def run() -> dict:
        return {"model_id": model_id, "model_hash": model_hash, "inputs_digest": digest, "output_kind": "json",
                "output": fn()}
    rec = ctx.external(f"infer.{kind}", run)
    if rec.get("model_id") != model_id or rec.get("model_hash") != model_hash or rec.get("inputs_digest") != digest:
        from ...core.errors import ProtocolError
        raise ProtocolError(f"recorded {kind} diverged from the current model or inputs")
    return rec["output"]


def block(x: np.ndarray, weight: float) -> np.ndarray:
    """An L2-normalised block scaled by `weight`: concatenating blocks whose squared weights sum to 1 gives a unit
    vector whose cosine similarity is the weighted mean of the blocks' cosines."""
    return (M.l2n(x) * np.float32(weight)).astype(np.float32)
