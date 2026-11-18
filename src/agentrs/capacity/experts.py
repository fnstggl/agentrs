"""Bounded expert LoRA adapters with learned per-example routing, and consolidation by distillation."""
from __future__ import annotations

import contextlib
import random
from typing import Optional, Sequence

import torch
from torch import nn


class Router(nn.Module):
    """Learned gate: domain/family id -> softmax over experts (index 0 = 'unknown')."""

    def __init__(self, keys: Sequence[str], n_experts: int):
        super().__init__()
        self.keys = ["<unk>"] + list(keys)
        self.logits = nn.Parameter(torch.zeros(len(self.keys), n_experts))

    def ids(self, keys: Sequence[Optional[str]]) -> torch.Tensor:
        idx = {k: i for i, k in enumerate(self.keys)}
        return torch.tensor([idx.get(k, 0) for k in keys], device=self.logits.device)

    def forward(self, keys: Sequence[Optional[str]]) -> torch.Tensor:
        return torch.softmax(self.logits[self.ids(keys)].float(), -1)


class ExpertAdapters:
    """Adds ``n_experts`` LoRA adapters (``expert0..``) on the same target layers as the shared
    ``default`` adapter of a PEFT model. All adapters are active; each expert's contribution is
    scaled per example by the router's gate (the shared adapter is always fully on), so routing is
    learned end-to-end by the same RL / BC losses. The router is registered on ``owner`` (e.g. the
    LMBackbone) so it is saved/restored with the trainable state and included in checkpoints."""

    def __init__(self, owner: nn.Module, peft_model, keys: Sequence[str], n_experts: int = 2, r: int = 8,
                 alpha: Optional[int] = None, max_experts: int = 4):
        from peft import LoraConfig

        if n_experts > max_experts:
            raise ValueError(f"at most {max_experts} experts (bounded capacity)")
        self.peft = peft_model
        base_cfg = peft_model.peft_config["default"]
        self.names = [f"expert{i}" for i in range(n_experts)]
        for nm in self.names:
            peft_model.add_adapter(nm, LoraConfig(r=r, lora_alpha=alpha or 2 * r, lora_dropout=0.0,
                                                  target_modules=base_cfg.target_modules, task_type=base_cfg.task_type))
        peft_model.base_model.set_adapter(["default", *self.names])  # all active (also makes them trainable)
        self.router = Router(keys, n_experts).to(next(peft_model.parameters()).device)
        owner.capacity_router = self.router
        self._base_scaling = {}
        from peft.tuners.lora import LoraLayer

        self.layers = [m for m in peft_model.modules() if isinstance(m, LoraLayer) and self.names[0] in m.lora_A]
        for m in self.layers:
            for nm in self.names:
                self._base_scaling[(id(m), nm)] = m.scaling[nm]
        self.retired: set[str] = set()

    @contextlib.contextmanager
    def route(self, keys: Sequence[Optional[str]]):
        """Per-example routing for a batch: sets each expert's scaling to base_scaling * gate[:, k]
        (shape (B, 1, 1), broadcast over sequence and features)."""
        gate = self.router(keys)
        try:
            for m in self.layers:
                for k, nm in enumerate(self.names):
                    s = 0.0 if nm in self.retired else gate[:, k].view(-1, 1, 1) * self._base_scaling[(id(m), nm)]
                    m.scaling[nm] = s
            yield gate
        finally:
            for m in self.layers:
                for nm in self.names:
                    m.scaling[nm] = self._base_scaling[(id(m), nm)]

    @contextlib.contextmanager
    def shared_only(self):
        try:
            for m in self.layers:
                for nm in self.names:
                    m.scaling[nm] = 0.0
            yield
        finally:
            for m in self.layers:
                for nm in self.names:
                    m.scaling[nm] = self._base_scaling[(id(m), nm)]

    def retire(self, name: str) -> None:
        """Switch an expert off for good (after its knowledge was consolidated / it proved redundant)."""
        self.retired.add(name)
        for n, p in self.peft.named_parameters():
            if f".{name}." in n:
                p.requires_grad_(False)

    def expert_params(self) -> int:
        return sum(p.numel() for n, p in self.peft.named_parameters() if any(f".{e}." in n for e in self.names))


def distill_into_shared(lm, experts: ExpertAdapters, prompts: list[str], keys: list[str], cont_ids: list[list[int]],
                        steps: int = 50, lr: float = 1e-4, batch: int = 4, seed: int = 0) -> list[float]:
    """Consolidation: train only the shared adapter so that the shared-only model matches the routed
    (shared + experts) model's token distributions on replay continuations (forward KL at the
    continuation positions). Returns the KL per step."""
    rng = random.Random(seed)
    shared = [p for n, p in lm.named_parameters() if p.requires_grad and ".default." in n]
    opt = torch.optim.AdamW(shared, lr=lr)
    hist = []
    for _ in range(steps):
        idx = rng.sample(range(len(prompts)), min(batch, len(prompts)))
        P, K, C = [prompts[i] for i in idx], [keys[i] for i in idx], [cont_ids[i] for i in idx]
        with torch.no_grad(), experts.route(K):
            _, _, teacher = lm.token_logprobs_from_ids(P, C, return_logdists=True)
        with experts.shared_only():
            _, mask, student = lm.token_logprobs_from_ids(P, C, return_logdists=True)
        m = mask.float().unsqueeze(-1)
        kl = (teacher.exp() * (teacher - student) * m).sum() / mask.float().sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        kl.backward()
        torch.nn.utils.clip_grad_norm_(shared, 1.0)
        opt.step()
        hist.append(float(kl.detach()))
    return hist
