"""cam_live.nn.presets — measured experiential-network presets and the ``nn`` configuration section.

``config.SCALE_PRESETS`` (master-owned) was written before the architecture existed and sizes only the transformer
blocks; with the tokenizer projections, slot cross-attention, decision trunk and feature map its ``d_ff`` values
overshoot the targets. ``NN_PRESETS`` holds the corrected values (measured by ``python -m cam_live.nn.bench``; see
``docs/change_requests/A.md``). :func:`resolve_core` substitutes them whenever ``cfg.core`` is an untouched master
preset and leaves any hand-edited ``cfg.core`` alone.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict

from ..config import SCALE_PRESETS, CamLiveConfig, CoreConfig

#: corrected presets (keys of config.SCALE_PRESETS plus optional ``nn`` overrides). Sized for d_sem = 3 * 2560
#: (Qwen3-4B, three taps). The small presets project the 7680-wide semantic embedding through a low-rank map
#: (``sem_rank``): a full projection alone would take 40 % of a 5 M budget.
NN_PRESETS: Dict[str, Dict[str, Any]] = {
    "debug": dict(d_model=64, n_layers=2, n_heads=4, d_ff=64, n_slots=4, n_features=16, target_params=0.2e6),
    "5m": dict(d_model=256, n_layers=4, n_heads=4, d_ff=320, n_slots=8, n_features=64, target_params=5e6,
               nn=dict(sem_rank=96)),
    "20m": dict(d_model=512, n_layers=5, n_heads=8, d_ff=576, n_slots=8, n_features=96, target_params=20e6,
                nn=dict(sem_rank=192)),
    "50m": dict(d_model=768, n_layers=6, n_heads=12, d_ff=1024, n_slots=8, n_features=128, target_params=50e6,
                nn=dict(sem_rank=384)),
    "150m": dict(d_model=1024, n_layers=10, n_heads=16, d_ff=2048, n_slots=16, n_features=192, target_params=150e6),
    "500m": dict(d_model=1536, n_layers=14, n_heads=24, d_ff=4096, n_slots=16, n_features=256, target_params=500e6),
    "1b": dict(d_model=2048, n_layers=16, n_heads=32, d_ff=5504, n_slots=16, n_features=256, target_params=1.0e9),
}


@dataclass
class NNConfig:
    """Settings of Workstream A, read from ``cfg.section("nn")`` (all optional)."""
    # experience tokens
    value_dim: int = 0               # width of the value channel (0 = max(16, d_model // 4))
    value_buckets: int = 0           # hash buckets for metric names (0 = 2048, or 256 below d_model 128)
    unit_buckets: int = 0            # hash buckets for units (0 = 256, or 64 below d_model 128)
    sem_rank: int = 0                # 0 = full d_sem -> d_model projection; > 0 = low-rank projection
    # core
    state_every: int = 2             # every n-th block cross-attends to the persistent slots
    gate_bias: float = -2.0
    trunk_ff: int = 0                # SwiGLU width of the decision trunk (0 = d_model)
    feature_ff: int = 0              # SwiGLU width of the feature map (0 = d_model // 2)
    # latent bridge into the backbone
    bridge_dim: int = 128
    bridge_layers: int = 6
    bridge_heads: int = 4
    # backbone runtime
    max_prompt_tokens: int = 2048
    embed_batch_tokens: int = 16384
    gen_batch: int = 16
    # fast state
    fast_noise_mode: str = "nig"     # "nig" (noise inferred, Student-t) | "fixed" (learned known noise)
    fast_prior_var: float = 1.0
    fast_noise: float = 0.5
    fast_forgetting: float = 0.999
    fast_change_h: float = 12.0

    @classmethod
    def from_cfg(cls, cfg: CamLiveConfig) -> "NNConfig":
        sec = cfg.section("nn")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(sec) - known)
        if unknown:
            raise KeyError(f"unknown keys in the 'nn' config section: {unknown}")
        return cls(**{**preset_nn(cfg.core), **sec})


_PRESET_KEYS = ("d_model", "n_layers", "n_heads", "d_ff", "n_slots", "n_features", "target_params")


def _is_master_preset(core: CoreConfig) -> bool:
    master = SCALE_PRESETS.get(core.scale)
    return master is not None and all(getattr(core, k) == master[k] for k in _PRESET_KEYS if k in master)


def preset_nn(core: CoreConfig) -> Dict[str, Any]:
    """``nn``-section defaults that belong to a scale preset (e.g. the low-rank semantic projection of the small
    presets). Applied when ``core`` is an untouched master preset or already the corrected one."""
    mine = NN_PRESETS.get(core.scale)
    if mine is None:
        return {}
    if _is_master_preset(core) or all(getattr(core, k) == mine[k] for k in _PRESET_KEYS):
        return dict(mine.get("nn", {}))
    return {}


def resolve_core(core: CoreConfig) -> CoreConfig:
    """The core configuration actually built: the corrected preset when ``core`` is an unmodified master preset."""
    if _is_master_preset(core):
        return dataclasses.replace(core, **{k: NN_PRESETS[core.scale][k] for k in _PRESET_KEYS})
    return core
