"""Explicit, continuous representation of elapsed time.

Horizons span 0 s to weeks (seven orders of magnitude), so time is represented on a log scale::

    tau(h) = log(1 + h / 1 s) / log(1 + 30 days / 1 s)        # 0 -> 0, 1 s -> 0.047, 1 h -> 0.55, 30 d -> 1

``TimeEncoder`` maps tau to a vector with polynomial + Fourier features followed by a learned
projection, so any duration (not only the configured horizons) has an embedding and nearby
durations have nearby embeddings (interpolation between trained horizons).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Union

import torch
from torch import nn

SECOND = 1.0
MINUTE = 60.0
HOUR = 3600.0
DAY = 86400.0
WEEK = 7 * DAY
REF_SECONDS = 30 * DAY
_LOG_REF = math.log1p(REF_SECONDS)

Number = Union[int, float]


def log_time(h_seconds) -> torch.Tensor:
    """tau(h) as a tensor (accepts floats, sequences or tensors; negative values clamp to 0)."""
    h = torch.as_tensor(h_seconds, dtype=torch.float32)
    return torch.log1p(h.clamp(min=0.0)) / _LOG_REF


def from_log_time(tau) -> torch.Tensor:
    return torch.expm1(torch.as_tensor(tau, dtype=torch.float32) * _LOG_REF)


class TimeEncoder(nn.Module):
    def __init__(self, n_freq: int = 8, d_out: int = 32):
        super().__init__()
        self.n_freq = n_freq
        self.register_buffer("freqs", torch.arange(1, n_freq + 1, dtype=torch.float32) * math.pi, persistent=False)
        self.proj = nn.Linear(2 + 2 * n_freq, d_out)

    def features(self, tau: torch.Tensor) -> torch.Tensor:
        t = tau.unsqueeze(-1)
        ang = t * self.freqs
        return torch.cat([t, t * t, torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(tau.to(self.proj.weight.device)))


_UNITS = {"ms": 0.001, "s": SECOND, "sec": SECOND, "m": MINUTE, "min": MINUTE, "h": HOUR, "hr": HOUR,
          "d": DAY, "day": DAY, "days": DAY, "w": WEEK, "wk": WEEK}


def parse_duration(x: str | Number) -> float:
    """'15m' -> 900.0, '3d' -> 259200.0, 42 -> 42.0."""
    if isinstance(x, (int, float)):
        return float(x)
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([a-z]*)\s*", x.lower())
    if not m:
        raise ValueError(f"bad duration {x!r}")
    unit = m.group(2) or "s"
    if unit not in _UNITS:
        raise ValueError(f"bad duration unit {unit!r}")
    return float(m.group(1)) * _UNITS[unit]


def describe_duration(s: Number) -> str:
    s = float(s)
    for unit, size in (("d", DAY), ("h", HOUR), ("m", MINUTE)):
        if s >= size and abs(s / size - round(s / size)) < 1e-9:
            return f"{int(round(s / size))}{unit}"
    if s >= DAY:
        return f"{s / DAY:.1f}d"
    if s >= HOUR:
        return f"{s / HOUR:.1f}h"
    if s >= MINUTE:
        return f"{s / MINUTE:.1f}m"
    return f"{s:g}s"


def durations(xs: Sequence[str | Number]) -> list[float]:
    return [parse_duration(x) for x in xs]
