"""cam_live.nn.core — the experiential core: b_t = F_theta(H_t, m_t) (spec 1.3, 2.1).

Pre-norm transformer over experience tokens:

* RMSNorm, SwiGLU feed-forward;
* causal self-attention with rotary positions over the event index plus a learned per-head function of the elapsed
  time between the attending and the attended event (log-scaled radial + Fourier basis, so 1 second and 30 days are
  both resolved). Time enters only as ``t_i - t_j`` with ``j <= i``, so ``h[:, i]`` depends on items ``<= i`` only;
* cross-attention from the token stream to ``K`` persistent state slots carried between calls, conditioned on the
  time elapsed since the previous call;
* a **gated** slot update after the tokens are read (convex mix of the old slot and a normalised candidate attended
  from the tokens; gate bias negative at init, small init std) — the state stays input-dependent and bounded, it
  neither freezes to a constant nor decays;
* a **direct path** from the newest event's token embedding into ``b`` in addition to the attention-mixed stream;
* a learned per-slot **retention** ``rho`` in [0, 1] (``persist_rho``) of the state carried in from the previous call:
  the call reads ``init + rho * (S - init)`` (and the elapsed-time conditioning mixed the same way), so ``rho = 0`` is
  bit-identically the fresh-state computation and a fresh state (``last_time`` NaN) is unaffected by ``rho``. The
  carried state itself is threaded across decisions and trained by ``cam_live.nn.persist`` (PS, amendment item 3).

Attention is written out explicitly (matmul + softmax) so every operation supports double backward.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as Fn
from torch import Tensor, nn

from .api import CoreOutput, CoreState, TokenBatch

#: basis of the elapsed-time function: radial bumps over u = log1p(seconds) plus Fourier terms of u
_TIME_CENTRES = tuple(float(c) for c in torch.linspace(0.0, 18.0, 16))
_TIME_WIDTH = 1.2
_TIME_FREQS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
N_TIME_FEATS = len(_TIME_CENTRES) + 2 * len(_TIME_FREQS)


def elapsed_features(dt: Tensor) -> Tensor:
    """float32 features [..., N_TIME_FEATS] of a non-negative elapsed time in seconds (any float dtype). Computed
    from u = log1p(dt): u(1 s) = 0.69, u(1 h) = 8.2, u(30 d) = 14.8."""
    u = torch.log1p(dt.clamp_min(0.0)).to(torch.float32).unsqueeze(-1)
    centres = torch.tensor(_TIME_CENTRES, device=u.device)
    freqs = torch.tensor(_TIME_FREQS, device=u.device)
    rbf = torch.exp(-0.5 * ((u - centres) / _TIME_WIDTH) ** 2)
    return torch.cat([rbf, torch.sin(u * freqs), torch.cos(u * freqs)], -1)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d)) if affine else None

    def forward(self, x: Tensor) -> Tensor:
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y if self.weight is None else y * self.weight


class SwiGLU(nn.Module):
    """w3(silu(w1 x) * w2 x), bias-free: 3 * d * d_ff parameters."""

    def __init__(self, d: int, d_ff: int, d_out: Optional[int] = None):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_out or d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(Fn.silu(self.w1(x)) * self.w2(x))


def rotary_tables(T: int, head_dim: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    ang = torch.arange(T, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    return ang.cos(), ang.sin()


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x [B, H, T, hd] with hd even; rotates consecutive pairs by the position angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


def _attend(q: Tensor, k: Tensor, v: Tensor, bias: Optional[Tensor]) -> Tensor:
    """q [B, H, Tq, hd], k/v [B, H, Tk, hd], additive ``bias`` broadcastable to [B, H, Tq, Tk] -> [B, H, Tq, hd]."""
    att = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if bias is not None:
        att = att + bias
    return torch.softmax(att, -1) @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        if d % n_heads or (d // n_heads) % 2:
            raise ValueError("d_model must be divisible by n_heads with an even head dimension (rotary positions)")
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.time_bias = nn.Linear(N_TIME_FEATS, n_heads, bias=False)
        nn.init.zeros_(self.time_bias.weight)

    def forward(self, x: Tensor, neg_mask: Tensor, tfeat: Tensor, rot: Tuple[Tensor, Tensor]) -> Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rotary(q, *rot), apply_rotary(k, *rot)
        bias = neg_mask + self.time_bias(tfeat).permute(0, 3, 1, 2)
        return self.out(_attend(q, k, v, bias).transpose(1, 2).reshape(B, T, D))


class CrossAttention(nn.Module):
    """Queries of width ``d_q`` attend to memory of width ``d_m`` through an inner width ``d_inner``."""

    def __init__(self, d_q: int, d_m: int, d_inner: int, n_heads: int):
        super().__init__()
        if d_inner % n_heads:
            raise ValueError("inner width must be divisible by n_heads")
        self.h = n_heads
        self.q = nn.Linear(d_q, d_inner, bias=False)
        self.k = nn.Linear(d_m, d_inner, bias=False)
        self.v = nn.Linear(d_m, d_inner, bias=False)
        self.out = nn.Linear(d_inner, d_q, bias=False)

    def forward(self, x: Tensor, mem: Tensor, neg_mask: Optional[Tensor] = None) -> Tensor:
        B, T, _ = x.shape
        M = mem.shape[1]
        hd = self.q.out_features // self.h
        q = self.q(x).view(B, T, self.h, hd).transpose(1, 2)
        k = self.k(mem).view(B, M, self.h, hd).transpose(1, 2)
        v = self.v(mem).view(B, M, self.h, hd).transpose(1, 2)
        return self.out(_attend(q, k, v, neg_mask).transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int, with_state: bool, dropout: float):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.with_state = with_state
        if with_state:
            self.nx = RMSNorm(d)
            self.xattn = CrossAttention(d, d, max(n_heads, (d // 2) // n_heads * n_heads), n_heads)
        self.n2 = RMSNorm(d)
        self.ff = SwiGLU(d, d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: Tensor, neg_mask: Tensor, tfeat: Tensor, rot: Tuple[Tensor, Tensor], slots: Tensor) -> Tensor:
        h = h + self.drop(self.attn(self.n1(h), neg_mask, tfeat, rot))
        if self.with_state:
            h = h + self.drop(self.xattn(self.nx(h), slots))
        return h + self.drop(self.ff(self.n2(h)))


class ExperientialCore(nn.Module):
    """Args: d_model, n_layers, n_heads, d_ff, n_slots; ``state_every``: every n-th block (starting with the first)
    cross-attends to the persistent slots; ``gate_bias``: initial bias of the slot-update gate (negative)."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, n_slots: int, *, state_every: int = 2,
                 dropout: float = 0.0, gate_bias: float = -2.0):
        super().__init__()
        d = d_model
        self.d_model, self.n_layers, self.n_heads, self.n_slots = d, n_layers, n_heads, n_slots
        self.blocks = nn.ModuleList(Block(d, n_heads, d_ff, i % max(1, state_every) == 0, dropout)
                                    for i in range(n_layers))
        self.final_norm = RMSNorm(d)
        # persistent state
        self.slot_init = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.slot_read_norm = RMSNorm(d)
        self.elapsed_proj = nn.Linear(N_TIME_FEATS, d, bias=False)
        self.fresh_emb = nn.Parameter(torch.zeros(d))
        self.write_q_norm = RMSNorm(d)
        self.write_attn = CrossAttention(d, d, d, n_heads)
        self.cand_norm = RMSNorm(d)
        self.gate_slot = nn.Linear(d, d, bias=False)
        self.gate_cand = nn.Linear(d, d, bias=True)
        #: learned retention of the carried-in state per slot (PS); 1 = read the carried state as it is, 0 = read the
        #: learned initial slots (the fresh-state computation). Clamped to [0, 1] in the forward with a straight-through
        #: gradient, so it can always move back into the range. A checkpoint written before PS lacks it: the agent's
        #: persistent-state chain then closes it (0) so the restored model computes exactly what it computed before.
        self.persist_rho = nn.Parameter(torch.ones(n_slots))
        # readout
        self.direct_norm = RMSNorm(d, affine=False)
        self.direct = nn.Linear(d, d, bias=False)
        self.state_read = nn.Linear(d, d, bias=False)
        self.out_norm = RMSNorm(d)
        self._init_weights(gate_bias)

    def _init_weights(self, gate_bias: float) -> None:
        res_std = 0.02 / math.sqrt(2.0 * max(1, self.n_layers))
        for name, p in self.named_parameters():
            if p.dim() < 2 or name == "slot_init":
                continue
            nn.init.normal_(p, std=0.02)
            if name.endswith(("attn.out.weight", "ff.w3.weight", "xattn.out.weight")) and name.startswith("blocks."):
                nn.init.normal_(p, std=res_std)
        for blk in self.blocks:
            nn.init.zeros_(blk.attn.time_bias.weight)
        nn.init.normal_(self.elapsed_proj.weight, std=0.02)
        nn.init.normal_(self.gate_slot.weight, std=0.01)
        nn.init.normal_(self.gate_cand.weight, std=0.01)
        nn.init.constant_(self.gate_cand.bias, gate_bias)
        # the candidate must carry the tokens at init: default-scale attention projections for the write path
        for lin in (self.write_attn.q, self.write_attn.k, self.write_attn.v, self.write_attn.out):
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(lin.in_features))
        with torch.no_grad():
            self.direct.weight.copy_(0.5 * torch.eye(self.d_model) + 0.02 * torch.randn(self.d_model, self.d_model))
            nn.init.normal_(self.state_read.weight, std=1.0 / math.sqrt(self.d_model))

    # ---- state ---------------------------------------------------------------------------------------------------
    def retention(self) -> Tensor:
        """rho [K] in [0, 1]; straight-through clamp (the raw parameter keeps its gradient outside the range)."""
        r = self.persist_rho
        return r + (r.clamp(0.0, 1.0) - r).detach()

    def init_state(self, batch: int, device: Optional[torch.device] = None) -> CoreState:
        """Fresh state: the learned initial slots; ``last_time`` = NaN marks "no previous call"."""
        device = device or self.slot_init.device
        slots = self.slot_init.unsqueeze(0).expand(batch, -1, -1).clone()
        return CoreState(slots=slots, last_time=torch.full((batch,), float("nan"), dtype=torch.float64, device=device),
                         step=0)

    # ---- forward -------------------------------------------------------------------------------------------------
    def forward(self, tokens: TokenBatch, state: Optional[CoreState] = None) -> CoreOutput:
        x, pad = tokens.x, tokens.pad_mask
        B, T, D = x.shape
        dev = x.device
        if state is None:
            state = self.init_state(B, dev)
        time = tokens.time.to(torch.float64)
        ar = torch.arange(T, device=dev)
        has_any = pad.any(1)
        last = (pad.long() * ar).max(1).values                                    # index of the newest real token
        rows = torch.arange(B, device=dev)

        # slots as read by the tokens: previous state + time elapsed since the previous call. The elapsed time runs
        # from the previous call's newest event to this call's newest event (re-encoded context windows overlap, so
        # this call's *first* event usually predates the previous call's last one). A carried-in state is read
        # through the learned retention rho: init + rho * (S - init); a fresh state is init itself, so it is
        # unaffected by rho, and rho = 0 reproduces the fresh-state computation bit for bit.
        fresh = torch.isnan(state.last_time)
        dt_call = torch.where(fresh, torch.zeros_like(state.last_time), time[rows, last] - state.last_time)
        cond_row = self.elapsed_proj(elapsed_features(dt_call)) + fresh.to(x.dtype).unsqueeze(-1) * self.fresh_emb
        # the fresh-state conditioning, computed on the same batch shape (a fresh row's cond_row equals it exactly)
        cond_fresh = self.elapsed_proj(elapsed_features(torch.zeros_like(dt_call))) + self.fresh_emb      # [B, D]
        rho = self.retention().view(1, -1, 1).to(x.dtype)                                 # [1, K, 1]
        init = self.slot_init.unsqueeze(0).to(x.dtype)
        slots_in = init + rho * (state.slots - init)                                      # [B, K, D]
        cond = cond_fresh.unsqueeze(1) + rho * (cond_row - cond_fresh).unsqueeze(1)       # [B, K, D]
        slots_read = self.slot_read_norm(slots_in) + cond

        # causal mask (+ padded keys hidden, self always visible so no row is empty), elapsed-time basis
        causal = ar[None, :] <= ar[:, None]
        allowed = causal[None] & (pad[:, None, :] | torch.eye(T, dtype=torch.bool, device=dev)[None])
        neg_mask = torch.zeros(B, 1, T, T, device=dev, dtype=x.dtype).masked_fill(~allowed[:, None], float("-inf"))
        tfeat = elapsed_features(time[:, :, None] - time[:, None, :])           # [B, T, T, N_TIME_FEATS]
        rot = rotary_tables(T, D // self.n_heads, dev)

        h = x
        for blk in self.blocks:
            h = blk(h, neg_mask, tfeat, rot, slots_read)
        h = self.final_norm(h) * pad.unsqueeze(-1).to(x.dtype)

        # gated slot update from the tokens of this call
        key_mask = torch.zeros(B, 1, 1, T, device=dev, dtype=x.dtype).masked_fill(~pad[:, None, None, :], float("-inf"))
        key_mask = torch.where(has_any[:, None, None, None], key_mask, torch.zeros_like(key_mask))
        u = self.write_attn(self.write_q_norm(slots_in) + cond, h, key_mask)
        cand = self.cand_norm(u)
        gate = torch.sigmoid(self.gate_slot(self.slot_read_norm(slots_in)) + self.gate_cand(cand))
        gate = gate * has_any[:, None, None].to(x.dtype)
        slots = slots_in + gate * (cand - slots_in)
        # an empty call leaves the carried state as it was (not its retention-mixed reading)
        slots = torch.where(has_any[:, None, None], slots, state.slots)
        new_time = torch.where(has_any, time[rows, last], state.last_time)
        new_state = CoreState(slots=slots, last_time=new_time, step=state.step + 1)

        # belief: attention-mixed stream at the newest event + direct path from that event + the updated state
        h_last, x_last = h[rows, last], x[rows, last]
        b = self.out_norm(h_last + self.direct(self.direct_norm(x_last))
                          + self.state_read(self.slot_read_norm(slots).mean(1)))
        aux = {"gate_mean": gate.mean().detach(), "slot_rms": slots.detach().pow(2).mean().sqrt(),
               "last_index": last, "carried_rows": (~fresh).sum().detach()}
        return CoreOutput(h=h, b=b, state=new_state, aux=aux)
