"""Shared training infrastructure: optimizer/scheduler, divergence guards and
checkpointing (weights + optimizer + scheduler + RNG state) with exact resume."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from pydantic import ConfigDict, BaseModel
from safetensors.torch import load_file, save_file


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 0
    total_steps: int = 1000
    schedule: str = "constant"  # constant | cosine | linear
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    max_bad_steps: int = 10


class DivergenceError(RuntimeError):
    pass


def _lr_lambda(cfg: OptimConfig):
    def f(step: int) -> float:
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if cfg.schedule == "constant":
            return 1.0
        prog = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps))
        if cfg.schedule == "cosine":
            return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
        if cfg.schedule == "linear":
            return 1 - (1 - cfg.min_lr_ratio) * prog
        raise ValueError(cfg.schedule)

    return f


class GradientTrainer:
    """Owns the optimizer and scheduler for one model; performs guarded updates."""

    def __init__(self, model: nn.Module, cfg: OptimConfig, params: Optional[list[nn.Parameter]] = None):
        self.model = model
        self.cfg = cfg
        self.params = params if params is not None else [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda(cfg))
        self.global_step = 0
        self.bad_steps = 0

    def step(self, loss: torch.Tensor) -> dict[str, float]:
        """Backprop + clipped update. Non-finite losses/gradients are skipped and counted;
        too many consecutive bad steps raise DivergenceError."""
        self.optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            return self._bad("non-finite loss")
        loss.backward()
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads:
            return {"grad_norm": 0.0, "lr": self.scheduler.get_last_lr()[0], "skipped": 1.0}
        gn = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip if self.cfg.grad_clip > 0 else 1e12)
        if not torch.isfinite(gn):
            self.optimizer.zero_grad(set_to_none=True)
            return self._bad("non-finite gradient")
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        self.bad_steps = 0
        return {"grad_norm": float(gn), "lr": float(self.scheduler.get_last_lr()[0]), "skipped": 0.0}

    def _bad(self, why: str) -> dict[str, float]:
        self.bad_steps += 1
        if self.bad_steps > self.cfg.max_bad_steps:
            raise DivergenceError(f"{self.bad_steps} consecutive bad steps ({why})")
        return {"grad_norm": float("nan"), "lr": float(self.scheduler.get_last_lr()[0]), "skipped": 1.0}

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step, "bad_steps": self.bad_steps}

    def load_state_dict(self, sd: dict) -> None:
        self.optimizer.load_state_dict(sd["optimizer"])
        self.scheduler.load_state_dict(sd["scheduler"])
        self.global_step = sd["global_step"]
        self.bad_steps = sd.get("bad_steps", 0)


# ---------------------------------------------------------------------------- checkpoints
def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Trainable parameters plus buffers that live outside the (frozen) backbone."""
    params = dict(model.named_parameters())
    out = {}
    for k, v in model.state_dict().items():
        trainable = k in params and params[k].requires_grad
        head_buffer = k not in params and not k.startswith("backbone.")
        if trainable or head_buffer:
            out[k] = v.detach().cpu().contiguous()
    return out


def save_checkpoint(path: str | os.PathLike, model: nn.Module, trainer: Optional[GradientTrainer] = None,
                    extra: Optional[dict] = None, keep_optimizer: bool = True) -> Path:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file(trainable_state(model), str(tmp / "weights.safetensors"))
    meta = {"component": getattr(model, "component", type(model).__name__),
            "config": model.cfg.model_dump() if hasattr(model, "cfg") else {},
            "extra": extra or {}}
    (tmp / "model.json").write_text(json.dumps(meta, indent=2, default=str))
    if trainer is not None and keep_optimizer:
        torch.save({"trainer": trainer.state_dict(), "rng": rng_state()}, tmp / "trainer.pt")
    if path.exists():
        shutil.rmtree(path)
    os.replace(tmp, path)
    return path


def read_meta(path: str | os.PathLike) -> dict:
    return json.loads((Path(path) / "model.json").read_text())


def build_model(component: str, config: dict) -> nn.Module:
    if component == "policy":
        from agentrs.models.policy import LMPolicy, PolicyConfig

        return LMPolicy(PolicyConfig.model_validate(config))
    if component == "world_model":
        from agentrs.models.world_model import WorldModel, WorldModelConfig

        return WorldModel(WorldModelConfig.model_validate(config))
    if component == "content_world_model":
        from agentrs.content.world_model import ContentConditionedWorldModel, ContentWorldModelConfig

        return ContentConditionedWorldModel(ContentWorldModelConfig.model_validate(config))
    if component == "critic":
        from agentrs.models.critic import Critic, CriticConfig

        return Critic(CriticConfig.model_validate(config))
    if component == "idm":
        from agentrs.models.idm import InverseDynamicsModel, IDMConfig

        return InverseDynamicsModel(IDMConfig.model_validate(config))
    raise ValueError(f"unknown component {component}")


def load_model(path: str | os.PathLike, config_override: Optional[dict] = None) -> nn.Module:
    meta = read_meta(path)
    cfg = meta["config"]
    if config_override:
        cfg = {**cfg, **config_override}
    model = build_model(meta["component"], cfg)
    load_weights(path, model)
    return model


def load_weights(path: str | os.PathLike, model: nn.Module) -> None:
    sd = load_file(str(Path(path) / "weights.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:5]}")
    trainable_missing = [n for n, p in model.named_parameters() if p.requires_grad and n in missing]
    if trainable_missing:
        raise RuntimeError(f"checkpoint missing trainable params: {trainable_missing[:5]}")


def load_trainer_state(path: str | os.PathLike, trainer: GradientTrainer, restore_rng: bool = True) -> None:
    st = torch.load(Path(path) / "trainer.pt", weights_only=False)
    trainer.load_state_dict(st["trainer"])
    if restore_rng:
        set_rng_state(st["rng"])


def param_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def param_change(model: nn.Module, snap: dict[str, torch.Tensor]) -> float:
    """L2 norm of the parameter change since ``snap`` (proof that weights moved)."""
    tot = 0.0
    for n, p in model.named_parameters():
        if n in snap:
            tot += float((p.detach() - snap[n]).pow(2).sum())
    return math.sqrt(tot)

