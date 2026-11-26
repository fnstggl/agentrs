"""Common encoder machinery: the Encoder base class (identity, version, preprocessing hash, model hash) and the
registry of the five modality encoders."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Sequence

import numpy as np

from . import models as M

_HERE = os.path.dirname(__file__)
SOURCES = ("base.py", "models.py", "text.py", "image.py", "audio.py", "video.py", "code.py", "../media.py")
SKIP_ATTRS = ("id", "modality", "roles", "dim", "preprocessing", "batch_size")


def canonical_json(x) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()


def _code_version() -> str:
    """Derived, never bumped by hand: SHA-256 of the source of every module that computes an encoder's output."""
    h = hashlib.sha256()
    for rel in SOURCES:
        with open(os.path.join(_HERE, rel), "rb") as fh:
            h.update(rel.encode() + b"\0" + fh.read().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:12]


CODE_VERSION = _code_version()


def _jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def machine() -> dict:
    """What else determines the numbers: device class and library versions."""
    import PIL
    import torch
    import transformers
    return {"device_class": M.device().split(":")[0], "torch": torch.__version__.split("+")[0],
            "transformers": transformers.__version__, "numpy": np.__version__, "pillow": PIL.__version__}


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

    def live_parameters(self) -> dict:
        """The encoder's parameter values as they are now (class and instance attributes that are plain data, e.g.
        `TextEncoder.window`), so a change made in code or at run time changes the key."""
        out = {}
        for k in sorted(set(dir(type(self))) | set(vars(self))):
            if k.startswith("_") or k in SKIP_ATTRS:
                continue
            v = getattr(self, k, None)
            if not callable(v) and _jsonable(v):
                out[k] = v
        return out

    def params_hash(self) -> str:
        """Derived from the declared preprocessing and the live parameter values; never edited by hand. Part of the
        world key (machine-independent, so a world regenerated on another device finds its records)."""
        return hashlib.sha256(canonical_json({"modality": self.modality, "id": self.id,
                                              "preprocessing": self.preprocessing,
                                              "live": self.live_parameters()})).hexdigest()[:16]

    def preprocessing_hash(self) -> str:
        """`params_hash` plus the machine (device class, library versions): part of the machine-wide cache key."""
        return hashlib.sha256(canonical_json({"params": self.params_hash(), "machine": machine()})).hexdigest()[:16]

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
                "live_parameters": self.live_parameters(), "machine": machine(), "device": M.device()}

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
