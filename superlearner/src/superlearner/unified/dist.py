"""Numerical building blocks: symlog, two-hot regression and straight-through categorical latents."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def make_bins(n: int, low: float, high: float, device=None) -> Tensor:
    return torch.linspace(low, high, n, device=device)


def twohot_encode(y: Tensor, bins: Tensor) -> Tensor:
    """Distribute ``y`` (already in bin space) over its two adjacent bins. Output shape ``y.shape + [n]``."""
    y = y.clamp(bins[0], bins[-1])
    above = torch.searchsorted(bins, y.contiguous(), right=False).clamp(1, len(bins) - 1)
    below = above - 1
    span = bins[above] - bins[below]
    w_above = (y - bins[below]) / span
    target = torch.zeros(*y.shape, len(bins), device=y.device, dtype=y.dtype)
    target.scatter_(-1, below.unsqueeze(-1), (1 - w_above).unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), w_above.unsqueeze(-1))
    return target


def twohot_loss(logits: Tensor, value: Tensor, bins: Tensor) -> Tensor:
    """Cross-entropy against the two-hot encoding of ``symlog(value)``; one loss per element of ``value``."""
    target = twohot_encode(symlog(value), bins)
    return -(target * F.log_softmax(logits, -1)).sum(-1)


def twohot_mean(logits: Tensor, bins: Tensor) -> Tensor:
    """Expected value in the original scale. Symmetric summation returns exactly 0 for uniform logits."""
    probs = F.softmax(logits, -1)
    n = len(bins)
    m = (n - 1) // 2
    p1, p2, p3 = probs[..., :m], probs[..., m:m + 1], probs[..., m + 1:]
    b1, b2, b3 = bins[:m], bins[m:m + 1], bins[m + 1:]
    wavg = (p2 * b2).sum(-1) + ((p1.flip(-1) * b1.flip(-1)) + (p3 * b3)).sum(-1)
    return symexp(wavg)


def unimix_logits(logits: Tensor, unimix: float) -> Tensor:
    if unimix <= 0:
        return logits
    probs = F.softmax(logits, -1)
    probs = (1 - unimix) * probs + unimix / probs.shape[-1]
    return torch.log(probs)


def sample_categorical(logits: Tensor, sample: bool = True) -> Tensor:
    """Straight-through one-hot sample over the last dimension; ``sample=False`` uses the mode."""
    probs = F.softmax(logits, -1)
    if sample:
        flat = probs.reshape(-1, probs.shape[-1])
        idx = torch.multinomial(flat, 1).reshape(probs.shape[:-1])
    else:
        idx = probs.argmax(-1)
    hard = F.one_hot(idx, probs.shape[-1]).to(probs.dtype)
    return hard + probs - probs.detach()


def categorical_kl(p_logits: Tensor, q_logits: Tensor) -> Tensor:
    """KL(p || q), summed over the latent variables; shape drops the last two dimensions."""
    log_p = F.log_softmax(p_logits, -1)
    log_q = F.log_softmax(q_logits, -1)
    return (log_p.exp() * (log_p - log_q)).sum(-1).sum(-1)


def categorical_entropy(logits: Tensor) -> Tensor:
    log_p = F.log_softmax(logits, -1)
    return -(log_p.exp() * log_p).sum(-1).sum(-1)


def balanced_kl(post_logits: Tensor, prior_logits: Tensor, free_nats: float) -> tuple[Tensor, Tensor]:
    """Return per-step ``(L_dyn, L_rep)``, each floored separately at ``free_nats`` as the reference does."""
    dyn = categorical_kl(post_logits.detach(), prior_logits)
    rep = categorical_kl(post_logits, prior_logits.detach())
    return dyn.clamp(min=free_nats), rep.clamp(min=free_nats)
