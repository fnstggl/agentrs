"""cam_live.config — root configuration. MASTER-OWNED.

One YAML file (``cam_live/configs/cam_live.yaml``) configures everything. The root object holds the settings
shared by several workstreams; each subpackage reads its own section with ``cfg.section("<name>")`` and parses it
into its own dataclass (so a subpackage never has to edit this file to add a field).

Scale presets size the *experiential* network only; the backbone is chosen separately, so preset comparisons keep
the pretrained model, the information available and the evaluation protocol identical (spec 16.10).
"""
from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))            # .../cam_live
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))

#: other tenants on this GPU serialise heavy jobs on this lock; hold it (fcntl.flock) for any job that loads a
#: >= 1.7B backbone or needs > 4 GB of GPU memory, for at most ~15 minutes at a time. Never wrap pytest in it.
#: The default is CAM Live's own path (G2I-10); ``CAM_LIVE_GPU_LOCK`` names another tenant's lock file. On the
#: development machine of 2026-11 ``<repo>/runs/gpu_heavy.lock`` is a symbolic link to the shared tenant's lock
#: (/workspace/agentrs/runs/cam_unified/gpu_heavy.lock), so the lock actually used there is unchanged.
GPU_LOCK_PATH = os.environ.get("CAM_LIVE_GPU_LOCK", os.path.join(REPO_ROOT, "runs", "gpu_heavy.lock"))


@dataclass
class BackboneConfig:
    #: "tiny" = a small randomly initialised causal transformer LM with the same interface (CPU tests and
    #: mathematical diagnostics only; it has no pretrained knowledge and is never reported as the backbone).
    name: str = "Qwen/Qwen3-4B"
    revision: str = "1cfa9a7208912126459214e8b04321603b3df60c"
    local_files_only: bool = True
    dtype: str = "bfloat16"
    tap_layers: tuple = (0.5, 0.75, 1.0)     # fractional depths pooled into the semantic embedding
    max_text_tokens: int = 384
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_targets: tuple = ("q_proj", "v_proj")
    lora_layers: str = "all"                 # "all" or "a:b" slice of transformer layers
    bridge_tokens: int = 8                   # virtual prefix tokens carrying CAM's latent into generation
    grad_checkpoint: bool = True
    embed_cache_items: int = 200_000
    tiny_d: int = 64
    tiny_layers: int = 2
    tiny_vocab: int = 512


#: experiential-core presets. ``target_params`` is the engineering target; the model reports the actual count.
#: ``d_ff`` values are the measured corrections of change request A-1 (the earlier values sized only the transformer
#: blocks and overshot every target: 50m measured 67.8 M). Measured with these values (``python -m cam_live.nn.bench``):
#: debug 202,421 · 5m 5,044,741 · 20m 20,087,333 · 50m 50,953,765 · 150m 151,022,853 · 500m 502,708,549 ·
#: 1b 1,000,547,333 (the small presets also use the low-rank semantic projection of ``nn.presets.NN_PRESETS``).
SCALE_PRESETS: Dict[str, Dict[str, Any]] = {
    "debug": dict(d_model=64, n_layers=2, n_heads=4, d_ff=64, n_slots=4, n_features=16, target_params=0.2e6),
    "5m": dict(d_model=256, n_layers=4, n_heads=4, d_ff=320, n_slots=8, n_features=64, target_params=5e6),
    "20m": dict(d_model=512, n_layers=5, n_heads=8, d_ff=576, n_slots=8, n_features=96, target_params=20e6),
    "50m": dict(d_model=768, n_layers=6, n_heads=12, d_ff=1024, n_slots=8, n_features=128, target_params=50e6),
    "150m": dict(d_model=1024, n_layers=10, n_heads=16, d_ff=2048, n_slots=16, n_features=192, target_params=150e6),
    "500m": dict(d_model=1536, n_layers=14, n_heads=24, d_ff=4096, n_slots=16, n_features=256, target_params=500e6),
    "1b": dict(d_model=2048, n_layers=16, n_heads=32, d_ff=5504, n_slots=16, n_features=256, target_params=1.0e9),
}


@dataclass
class CoreConfig:
    scale: str = "50m"
    d_model: int = 768
    n_layers: int = 6
    n_heads: int = 12
    d_ff: int = 1024                 # corrected 50m preset (change request A-1)
    n_slots: int = 8                 # persistent recurrent state slots
    n_features: int = 128            # F: width of the fast-adaptation feature map
    max_context: int = 256           # tokens of H_t per forward
    max_values: int = 4              # V: exact quantity slots per token
    dropout: float = 0.0
    target_params: float = 50e6

    def apply_scale(self, scale: str) -> "CoreConfig":
        if scale not in SCALE_PRESETS:
            raise KeyError(f"unknown scale {scale!r}; choose from {sorted(SCALE_PRESETS)}")
        out = dataclasses.replace(self, scale=scale, **SCALE_PRESETS[scale])
        return out


@dataclass
class OptimConfig:
    """Shared defaults for the three learning timescales (spec Part 2). Subpackages may refine in their sections."""
    lr_experiential: float = 3e-4
    lr_heads: float = 3e-4
    lr_adapters: float = 5e-5
    lr_reward: float = 1e-4          # outer (meta) learning rate beta
    inner_lr: float = 1e-2           # alpha of the differentiable inner update
    inner_steps: int = 2             # truncated unroll length K
    weight_decay: float = 0.01
    clip_experiential: float = 1.0   # per-group gradient clipping (a single global clip froze the core before)
    clip_heads: float = 1.0
    clip_adapters: float = 0.5
    clip_reward: float = 0.5
    adapter_every: int = 8           # frequent updates between slow adapter updates
    meta_min_independent_outcomes: int = 8
    batch_size: int = 16


@dataclass
class CamLiveConfig:
    seed: int = 0
    device: str = "auto"             # "auto" | "cpu" | "cuda"
    home: str = ""                   # state root; default <repo>/runs/cam_live
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    sections: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # per-subpackage settings

    # ---- helpers -------------------------------------------------------------------------------------------
    def section(self, name: str) -> Dict[str, Any]:
        return copy.deepcopy(self.sections.get(name, {}))

    def resolved_home(self) -> str:
        return self.home or os.environ.get("CAM_LIVE_HOME") or os.path.join(REPO_ROOT, "runs", "cam_live")

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CamLiveConfig":
        d = copy.deepcopy(d or {})
        bb = BackboneConfig(**_tuplify(d.pop("backbone", {}), BackboneConfig))
        core_d = d.pop("core", {})
        core = CoreConfig()
        if "scale" in core_d:
            core = core.apply_scale(core_d["scale"])
        core = dataclasses.replace(core, **core_d)
        optim = OptimConfig(**d.pop("optim", {}))
        known = {f.name for f in dataclasses.fields(cls)} - {"backbone", "core", "optim", "sections"}
        top = {k: d.pop(k) for k in list(d) if k in known}
        sections = d.pop("sections", {})
        sections.update(d)           # any other top-level mapping is a subpackage section
        return cls(backbone=bb, core=core, optim=optim, sections=sections, **top)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "CamLiveConfig":
        path = path or os.path.join(PROJECT_ROOT, "configs", "cam_live.yaml")
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def debug(cls, **over: Any) -> "CamLiveConfig":
        """Tiny CPU configuration for mathematical tests: tiny random backbone + debug core."""
        cfg = cls(device="cpu", backbone=BackboneConfig(name="tiny", revision="", lora_rank=2, bridge_tokens=2,
                                                        max_text_tokens=48),
                  core=CoreConfig().apply_scale("debug"))
        cfg.core.max_context = 48
        for k, v in over.items():
            setattr(cfg, k, v)
        return cfg


def _tuplify(d: Dict[str, Any], cls) -> Dict[str, Any]:
    out = dict(d)
    for f in dataclasses.fields(cls):
        if f.name in out and isinstance(out[f.name], list):
            out[f.name] = tuple(out[f.name])
    return out
