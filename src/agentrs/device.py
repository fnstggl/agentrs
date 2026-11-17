"""Device selection and device-safe sampling.

``BackboneConfig.device`` accepts ``cpu``, ``cuda``, ``cuda:N`` or ``auto``. Its default comes from
the ``AGENTRS_DEVICE`` environment variable (``cpu`` when unset), so every existing config can be run
on a GPU with ``AGENTRS_DEVICE=cuda`` without editing YAML. A checkpoint saved with ``device: cuda``
still loads on a machine without CUDA (it falls back to CPU with a warning).
"""
from __future__ import annotations

import os
import warnings
from typing import Optional

import torch


def default_device() -> str:
    return os.environ.get("AGENTRS_DEVICE", "cpu")


def resolve_device(device: str) -> str:
    d = (device or "cpu").strip().lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(f"device {device!r} requested but CUDA is unavailable; using cpu", RuntimeWarning)
        return "cpu"
    return d


def multinomial(probs: torch.Tensor, num_samples: int = 1, replacement: bool = False,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """``torch.multinomial`` that works with CPU generators on any device.

    Samplers in this codebase hold seeded CPU ``torch.Generator`` objects. Sampling is done on the
    generator's device (CPU), so CPU runs are bit-identical to before and GPU runs stay reproducible.
    Half-precision probabilities are upcast to float32.
    The sampled indices are returned on ``probs``'s device.
    """
    gdev = generator.device if generator is not None else probs.device
    p = probs.detach().to(gdev)
    if p.dtype not in (torch.float32, torch.float64):
        p = p.float()
    out = torch.multinomial(p, num_samples, replacement=replacement, generator=generator)
    return out.to(probs.device)
