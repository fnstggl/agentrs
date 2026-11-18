"""Parameter accounting and function-preserving LoRA rank expansion."""
from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path
from typing import Optional

import torch
from torch import nn


def _lora_layers(model: nn.Module, adapter: str = "default"):
    from peft.tuners.lora import LoraLayer

    for name, mod in model.named_modules():
        if isinstance(mod, LoraLayer) and adapter in getattr(mod, "lora_A", {}):
            yield name, mod


def lora_ranks(model: nn.Module, adapter: str = "default") -> dict[str, int]:
    return {n: m.lora_A[adapter].out_features for n, m in _lora_layers(model, adapter)}


def param_report(model: nn.Module) -> dict:
    """Frozen vs trainable parameters (shared tied weights counted once), LoRA rank histogram,
    adapters present and parameters active per forward pass."""
    seen, total, trainable, by_kind = set(), 0, 0, {}
    for n, p in model.named_parameters(remove_duplicate=True):
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            kind = "router" if "router" in n else ("lora_expert" if re.search(r"lora_[AB]\.expert", n) else
                                                   ("lora" if "lora_" in n else "other"))
            by_kind[kind] = by_kind.get(kind, 0) + p.numel()
    ranks = {}
    try:
        for _, r in lora_ranks(model).items():
            ranks[r] = ranks.get(r, 0) + 1
    except ImportError:
        pass
    return {"total": total, "trainable": trainable, "frozen": total - trainable,
            "trainable_frac": trainable / total if total else 0.0, "trainable_by_kind": by_kind,
            "lora_rank_histogram": ranks}


@torch.no_grad()
def expand_lora_rank(model: nn.Module, new_r: int, adapter: str = "default", name_filter: Optional[str] = None,
                     seed: int = 0) -> dict[str, tuple[int, int]]:
    """Grow the LoRA rank of matching layers to ``new_r`` without changing the model's function:
    ΔW = s·B·A with A' = [A; A_new] (A_new random, like LoRA init) and B' = [B, 0], s unchanged, so
    ΔW' = ΔW exactly; the new directions get gradient through B_new immediately. Returns
    {layer: (old_r, new_r)}."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    changed = {}
    for name, mod in list(_lora_layers(model, adapter)):
        if name_filter and not re.search(name_filter, name):
            continue
        A, B = mod.lora_A[adapter], mod.lora_B[adapter]
        r = A.out_features
        if new_r <= r:
            continue
        dev, dt = A.weight.device, A.weight.dtype
        A2 = nn.Linear(A.in_features, new_r, bias=False).to(device=dev, dtype=dt)
        w = torch.empty(new_r - r, A.in_features)
        bound = 1 / math.sqrt(A.in_features)  # kaiming_uniform(a=sqrt(5)) bound, as LoRA initialises A
        w.uniform_(-bound, bound, generator=g)
        A2.weight.copy_(torch.cat([A.weight.detach().cpu().float(), w]).to(dt))
        B2 = nn.Linear(new_r, B.out_features, bias=False).to(device=dev, dtype=dt)
        B2.weight.zero_()
        B2.weight[:, :r] = B.weight
        A2.weight.requires_grad_(A.weight.requires_grad)
        B2.weight.requires_grad_(B.weight.requires_grad)
        scaling = mod.scaling[adapter]
        mod.lora_A[adapter] = A2
        mod.lora_B[adapter] = B2
        mod.r[adapter] = new_r
        mod.scaling[adapter] = scaling                  # keep the learned function's scale
        mod.lora_alpha[adapter] = scaling * new_r
        changed[name] = (r, new_r)
    return changed


def transfer_optimizer_state(old_opt: torch.optim.Optimizer, old_named: dict[str, torch.Tensor],
                             new_opt: torch.optim.Optimizer, new_named: dict[str, torch.Tensor]) -> int:
    """Copy AdamW moments from the old parameters into the (possibly grown) new ones by name; grown
    tensors get the old moments in their leading slice and zeros elsewhere. Returns tensors moved."""
    moved = 0
    old_by_name = {n: old_opt.state.get(p) for n, p in old_named.items()}
    for n, p in new_named.items():
        st = old_by_name.get(n)
        if not st:
            continue
        new = {}
        for k, v in st.items():
            if torch.is_tensor(v) and v.dim() > 0:
                z = torch.zeros_like(p, dtype=v.dtype)
                z[tuple(slice(0, s) for s in v.shape)] = v
                new[k] = z
            else:
                new[k] = v.clone() if torch.is_tensor(v) else v
        new_opt.state[p] = new
        moved += 1
    return moved


def _safetensors_shapes(path: Path) -> dict[str, list[int]]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return {k: v["shape"] for k, v in h.items()}


def match_checkpoint_ranks(model: nn.Module, ckpt_dir: Path, adapter: str = "default") -> dict:
    """Before loading a checkpoint: grow any layer whose saved LoRA rank is larger than the model's
    (the saved weights then overwrite everything, so the restored function is exactly the saved one)."""
    shapes = _safetensors_shapes(Path(ckpt_dir) / "weights.safetensors")
    want: dict[str, int] = {}
    for k, s in shapes.items():
        m = re.match(rf"(.*)\.lora_A\.{adapter}\.weight$", k)
        if m:
            want[m.group(1)] = s[0]
    changed = {}
    sd_names = {n for n, _ in model.named_parameters()}
    for name, mod in list(_lora_layers(model, adapter)):
        # parameter names of this layer may carry a prefix relative to the module name
        key = next((w for w in want if w.endswith(name) or name.endswith(w)), None)
        if key is not None and want[key] > mod.lora_A[adapter].out_features:
            changed |= expand_lora_rank(model, want[key], adapter, name_filter="^" + re.escape(name) + "$")
    del sd_names
    return changed
