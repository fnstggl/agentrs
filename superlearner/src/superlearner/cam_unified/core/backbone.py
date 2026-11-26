"""The shared trainable backbone F_theta of CAM Unified (workstream B, W3; implements `UnifiedCoreP`).

    Z_t = F_theta(Z_{t-1}, [O_t; A_{t-1}; B_t; M_t; G_t]; S_t)

One computation: joint self-attention over the S persistent state tokens concatenated with the context tokens, for
`cfg.n_layers` pre-norm blocks (RMSNorm -> SDPA attention -> RMSNorm -> SwiGLU), with DOM Mechanism C (gated
cross-attention to the semantic token states S_t, `core/semantic_xattn.py`) after every `cfg.sem_xattn_every`-th
block when it is configured and `sem` is given. The outputs at the context positions are returned as `ctx_out`
(aligned with the input context); the outputs at the state positions are the candidate state Z~_t.

Persistence (`cfg.state_gate`, DOM preset on). The carried state is a learned gated update

    u   = sigmoid(W [Z_{t-1}; Z~_t] + b)      (per token and feature; W = 0, b = -2 at init: u = 0.12)
    Z_t = RMSNorm((1 - u) * Z_{t-1} + u * Z~_t)
    Z~_t = clamp-normalised state residual of the blocks, without the re-added slot/kind embedding
           (`state_candidate_of`; Z_0 is the learned 0.02-scale initial state, so the first decision writes Z)

so information in Z decays only as fast as the gates choose (the ungated core re-added a 0.5-std constant and
re-normalised Z every step: cos(Z, const) 0.995, ~0.42 of the previous state lost per decision, delay-8 recall not
learnable; audit 2026-11-26). With the gate, delay-16 recall is learned (key in the first context, and key after 4
distractor decisions). At initialisation the first decision writes ~16% observation-dependent content into Z and later
decisions write little (~0.2% each) until the gates and block updates learn to write (key-after-4 recall: learned by
iteration 450). Without the gate Z_t = Z~_t = out_norm(...) (pre-DOM behaviour exactly).

Context embedding (per token n):  RMSNorm(x_n) + KindEmb(kind_n) + TimeEnc(time_n) + SlotCode(entity_n)
                                    + SlotCode2(entity2_n)                 (init scale `emb_std`, default 0.5).
State embedding (per slot s):     Z_prev[s] + StatePos[s] + KindEmb(STATE) (init scale `cfg.state_emb_std`; the
                                   REASON row as well). These are inputs to the blocks only; the carried residual is
                                   Z_prev itself (through the gate). At the pre-DOM scale 0.5 they dominated the
                                   state stream (cos(Z_t, const) = 0.995 at init); the dom presets use 0.02.
Context tokens get **no positional index**: the context is a set, so permuting it permutes `ctx_out` and leaves Z
unchanged (up to floating-point summation order). State tokens are ordered memory slots and do carry a learned slot
embedding.

The per-step working budget `cfg.max_ctx_tokens` is enforced by kind priority, then recency (token time closer to
now first); every dropped token is counted in `budget_stats` and the kept mask of the last call is `last_kept`. The
budget is a working-memory budget per step, never a lifetime limit (the archive keeps everything).

The world model's prior pass calls the same `forward` with ACTION / BELIEF / TIME / NOISE context tokens (one shared
transition computation), optionally through `transition_blocks(k)` (the last k blocks, `blocks=`); the Reasoner
reuses the last `cfg.reason_layers` entries of `.blocks`. Every entry point takes the semantic context explicitly
(no module state; see `core/semantic_xattn.py`).
"""
from __future__ import annotations

import contextlib
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..config import CamConfig
from ..interfaces import TokenKind, TokenSet
from .blocks import (Block, EntitySlotCodes, KindEmbedding, RMSNorm, TimeEncoding, count_parameters,
                     init_transformer_weights)
from .semantic_xattn import SemanticContext, SemanticKV, build_semantic_modules

#: initial bias of the state gate: u = sigmoid(-2) = 0.12 (retain ~88% of Z_{t-1} per decision until learned)
STATE_GATE_BIAS = -2.0
#: kinds whose embedding rows belong to the state stream (scaled by cfg.state_emb_std)
STATE_STREAM_KINDS = (TokenKind.STATE, TokenKind.REASON)

#: kind priority for the context budget (higher is kept first; ties broken by recency).
DEFAULT_KIND_PRIORITY = {
    TokenKind.GOAL: 10, TokenKind.ACTION: 9, TokenKind.TIME: 9, TokenKind.NOISE: 9, TokenKind.BUDGET: 8,
    TokenKind.OPTION: 8, TokenKind.BELIEF: 8, TokenKind.QUERY: 7, TokenKind.EVENT: 6, TokenKind.ENTITY: 5,
    TokenKind.RELATION: 5, TokenKind.OBS_NUM: 4, TokenKind.OBS_SEM: 4, TokenKind.MEMORY: 3, TokenKind.REASON: 2,
    TokenKind.STATE: 2, TokenKind.TOOL: 1, TokenKind.PAD: 0,
}


class UnifiedCore(nn.Module):
    """F_theta. forward(Z_prev [B,S,D], ctx TokenSet [B,N,D], code_rows=None, sem=None) -> (Z [B,S,D] float32,
    ctx_out [B,N,D] float32).

    Args:
        cfg: CamConfig (n_layers, d_model, n_heads, n_state, d_ff, dropout, max_ctx_tokens, grad_checkpoint,
             max_entities, n_fourier; DOM: state_gate, state_emb_std, sem_xattn_every, sem_xattn_dim, sem_gate_bias).
        autocast: run the blocks under bf16 autocast (default: True on CUDA, False on CPU).
        kind_priority: override of DEFAULT_KIND_PRIORITY for the context budget.
        qk_norm: per-head RMS normalisation of queries/keys.
        emb_std: initial scale of the additive context embeddings (kind / time / entity code). Token content enters
            RMS-normalised (unit scale per element), so these embeddings start at a comparable scale instead of the
            usual 0.02 — otherwise kind, slot and time information is invisible at initialisation and slow to learn.
        state_emb_std: initial scale of the state-stream embeddings (state slot positions and the STATE / REASON kind
            rows); None = `cfg.state_emb_std` (0.5 by default = emb_std, the pre-DOM initialisation exactly; 0.02 in
            the dom presets, so the observation content, not a constant, dominates the state at initialisation).
    """

    def __init__(self, cfg: CamConfig, autocast: Optional[bool] = None, kind_priority: Optional[dict] = None,
                 qk_norm: bool = True, slot_seed: int = 1234, emb_std: float = 0.5,
                 state_emb_std: Optional[float] = None):
        super().__init__()
        self.cfg = cfg
        D, S = cfg.d_model, cfg.n_state
        self.autocast = autocast
        self.grad_checkpoint = cfg.grad_checkpoint
        s_std = float(getattr(cfg, "state_emb_std", emb_std) if state_emb_std is None else state_emb_std)
        self.emb_std, self.state_emb_std = float(emb_std), s_std
        self.z0 = nn.Parameter(torch.randn(S, D) * 0.02)                    # learned initial state tokens
        self.state_pos = nn.Parameter(torch.randn(S, D) * 0.02)             # identity of each state slot
        self.kind_emb = KindEmbedding(D)
        self.time_enc = TimeEncoding(D, n_freq=cfg.n_fourier)
        self.slot_codes = EntitySlotCodes(D, cfg.max_entities, seed=slot_seed)
        self.ctx_in_norm = RMSNorm(D)
        self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.d_ff, cfg.dropout, qk_norm=qk_norm)
                                     for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(D)
        init_transformer_weights(self.blocks, cfg.n_layers)
        init_transformer_weights(self.time_enc, cfg.n_layers)
        init_transformer_weights(self.slot_codes, cfg.n_layers)
        nn.init.normal_(self.kind_emb.weight, std=emb_std)
        nn.init.normal_(self.state_pos, std=s_std)              # same draws as the pre-DOM init (std scales them)
        nn.init.normal_(self.time_enc.proj.weight, std=emb_std / self.time_enc.proj.in_features ** 0.5)
        for pr in self.slot_codes.proj:
            nn.init.normal_(pr.weight, std=emb_std)
        if s_std != emb_std:                                    # rescale (no extra random draws: same init stream)
            with torch.no_grad():
                for k in STATE_STREAM_KINDS:
                    self.kind_emb.weight[int(k)].mul_(s_std / emb_std)
        prio = dict(DEFAULT_KIND_PRIORITY)
        if kind_priority:
            prio.update({TokenKind(k): v for k, v in kind_priority.items()})
        self.register_buffer("kind_priority", torch.tensor([float(prio.get(TokenKind(k), 0))
                                                            for k in range(int(TokenKind.N_KINDS))],
                                                           dtype=torch.float64), persistent=False)
        self.budget_stats = {"calls": 0, "calls_over_budget": 0, "rows_over_budget": 0, "dropped_tokens": 0,
                             "dropped_by_kind": {}}
        self.last_kept: Optional[Tensor] = None
        # ---- DOM additions, created after every pre-DOM parameter (the pre-DOM initialisation stream is unchanged)
        # learned state gate u = sigmoid(W [Z_prev; Z~] + b), W = 0, b = -2 (cfg.state_gate)
        self.state_gate: Optional[nn.Linear] = None
        self.state_candidate = "clamp"            # candidate normalisation of the gated update (state_candidate_of)
        self.state_unit = True                    # RMS-normalise the carried state after the gated update
        if bool(getattr(cfg, "state_gate", False)):
            self.state_gate = nn.Linear(2 * D, D)
            nn.init.zeros_(self.state_gate.weight)
            nn.init.constant_(self.state_gate.bias, STATE_GATE_BIAS)
        # Mechanism C: gated semantic cross-attention after every cfg.sem_xattn_every-th block
        self.xattn_layers, self.sem_proj, self.xattn, self.cond_proj = build_semantic_modules(
            cfg, D, cfg.n_heads, cfg.n_layers)
        self.last_gates: dict = {}                 # {block index: mean gate (detached 0-d tensor)} of the last call
        self.last_state_gate: Optional[Tensor] = None   # mean state-gate value u of the last call (detached)
        self.sem_stats = {"calls_with_sem": 0, "calls_sem_ignored": 0}

    @property
    def has_semantic_xattn(self) -> bool:
        return self.sem_proj is not None and len(self.xattn) > 0

    # ------------------------------------------------------------------ state
    def init_state(self, B: int, device=None) -> Tensor:
        """Learned initial state tokens Z_0 [B,S,D]."""
        z = self.z0.unsqueeze(0).expand(B, -1, -1)
        return (z.to(device) if device is not None else z).clone()

    def new_episode(self, B: int, rows=None, generator: Optional[torch.Generator] = None) -> Tensor:
        """Fresh random per-episode entity-slot code permutations for all rows or the given rows (a partial reset
        with a batch layout of a different size raises ValueError)."""
        return self.slot_codes.new_episode(B, rows=rows, generator=generator)

    def episode_state(self) -> dict:
        """Per-episode (non-learned) state of the backbone: the entity slot-code permutations. Save it with the
        per-environment fast/latent state; it is also part of `state_dict()` (persistent buffer)."""
        return self.slot_codes.episode_state()

    def load_episode_state(self, state: dict) -> None:
        self.slot_codes.load_episode_state(state)

    def _autocast_ctx(self, device: torch.device):
        use = self.autocast if self.autocast is not None else device.type == "cuda"
        if not use:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    # ------------------------------------------------------------------ context budget
    def select_budget(self, ctx: TokenSet, budget: Optional[int] = None) -> tuple:
        """Enforce the working budget. Returns (ctx' TokenSet [B,k,D], idx [B,k] long | None, kept [B,N] bool).
        idx is None when no row exceeds the budget (ctx' is ctx). Selection: valid tokens by kind priority, then
        recency (larger `time` = more recent)."""
        budget = self.cfg.max_ctx_tokens if budget is None else budget
        B, N = ctx.mask.shape
        n_valid = ctx.mask.sum(1)
        self.budget_stats["calls"] += 1
        if N <= budget:
            return ctx, None, ctx.mask
        # over budget, or only padding beyond it: compact to k <= budget positions (dropping only if a row needs it)
        k = min(budget, int(n_valid.max()))
        prio = self.kind_priority.to(ctx.kind.device)[ctx.kind.clamp(0, int(TokenKind.N_KINDS) - 1)]
        score = prio * 1e7 + ctx.time.double().clamp(-1e6, 1e6)
        score = score.masked_fill(~ctx.mask, float("-inf"))
        idx = torch.topk(score, k, dim=1).indices                          # [B,k]
        kept = torch.zeros_like(ctx.mask).scatter_(1, idx, True) & ctx.mask
        over = n_valid > budget
        n_drop = int((n_valid - kept.sum(1)).sum())
        if n_drop:
            self.budget_stats["calls_over_budget"] += 1
            self.budget_stats["rows_over_budget"] += int(over.sum())
            self.budget_stats["dropped_tokens"] += n_drop
            dropped = ctx.mask & ~kept
            kinds, counts = torch.unique(ctx.kind[dropped], return_counts=True)
            for kd, c in zip(kinds.tolist(), counts.tolist()):
                name = TokenKind(kd).name
                self.budget_stats["dropped_by_kind"][name] = self.budget_stats["dropped_by_kind"].get(name, 0) + c

        def g(t):
            return torch.gather(t, 1, idx)
        x = torch.gather(ctx.x, 1, idx.unsqueeze(-1).expand(-1, -1, ctx.x.shape[-1]))
        refs = None if ctx.refs is None else [[ctx.refs[b][i] for i in idx[b].tolist()] for b in range(B)]
        sel = TokenSet(x, g(ctx.mask), g(ctx.kind), g(ctx.time), g(ctx.entity), refs,
                       None if ctx.entity2 is None else g(ctx.entity2))
        return sel, idx, kept

    # ------------------------------------------------------------------ embeddings and blocks
    def embed_ctx(self, ctx: TokenSet, code_rows: Optional[Tensor] = None) -> Tensor:
        """Context token embeddings h [B,N,D] (float32). No positional index (set semantics)."""
        h = self.ctx_in_norm(ctx.x.float())
        h = h + self.kind_emb(ctx.kind) + self.time_enc(ctx.time)
        h = h + self.slot_codes(ctx.entity, 0, code_rows)
        if ctx.entity2 is not None:
            h = h + self.slot_codes(ctx.entity2, 1, code_rows)
        return h * ctx.mask.unsqueeze(-1).to(h.dtype)

    def embed_state(self, Z: Tensor, kind: int = TokenKind.STATE) -> Tensor:
        """State token embeddings [B,S,D]: Z + StatePos + KindEmb(kind)."""
        return Z.float() + self.state_pos + self.kind_emb.weight[int(kind)]

    # ------------------------------------------------------------------ transition blocks / semantic memory
    def transition_blocks(self, k: int) -> nn.ModuleList:
        """The last k blocks of the shared backbone (the same modules, not copies): the world model's transition
        computation (`WorldModel.prior` through `cfg.wm_transition_blocks`) and the Reasoner reuse them. Pass them as
        `forward(..., blocks=core.transition_blocks(k))` or to `run_blocks`; semantic cross-attention after a reused
        block applies as in the full pass (block identity -> index)."""
        n = len(self.blocks)
        k = int(k)
        if not 0 < k <= n:
            raise ValueError(f"transition_blocks({k}): the core has {n} blocks")
        return self.blocks[n - k:]

    def block_ids(self, blocks) -> list:
        """Backbone index of each block in `blocks` (None for a block that is not one of the backbone's)."""
        index = {id(b): i for i, b in enumerate(self.blocks)}
        return [index.get(id(b)) for b in blocks]

    def project_semantic(self, sem, B: int, code_rows: Optional[Tensor] = None,
                         sem_rows: Optional[Tensor] = None) -> Optional[SemanticKV]:
        """SemanticContext -> SemanticKV for a batch of B rows (None when Mechanism C is off or sem is None; a
        SemanticKV passes through, row-selected the same way). Row alignment, explicit and stateless:
        `sem_rows` [B] given -> rows sem_rows of sem; else sem.B == B -> sem as is; else `code_rows` [B] (batch row
        -> episode/environment row, as for the entity slot codes) -> rows code_rows of sem; else ValueError. A
        source-aligned sem whose size happens to equal the batch's must be passed with `sem_rows`."""
        if sem is None:
            return None
        if not self.has_semantic_xattn:
            self.sem_stats["calls_sem_ignored"] += 1
            return None
        if sem_rows is not None:
            sem = sem.select(sem_rows)
        elif sem.B != B:
            if code_rows is None or int(code_rows.numel()) != B:
                raise ValueError(f"semantic context has {sem.B} rows for a batch of {B}; pass sem_rows (batch row -> "
                                 "semantic row) or code_rows, or select the rows first (SemanticContext.select)")
            sem = sem.select(code_rows)
        self.sem_stats["calls_with_sem"] += 1
        if isinstance(sem, SemanticKV):
            return sem
        S = self.sem_proj(sem.h)
        cond = None if sem.cond is None else self.cond_proj(sem.cond.to(self.cond_proj.weight.dtype))
        return SemanticKV(S, sem.mask.to(device=S.device, dtype=torch.bool), cond)

    def gate_stats(self) -> dict:
        """Mean semantic-gate value per cross-attention block of the last call (floats; one host sync here)."""
        return {int(k): float(v) for k, v in self.last_gates.items()}

    def run_blocks(self, h: Tensor, key_mask: Optional[Tensor], blocks=None,
                   attn_bias: Optional[Tensor] = None, checkpoint_blocks: Optional[bool] = None,
                   sem=None, block_ids: Optional[list] = None, gate_log: Optional[dict] = None) -> Tensor:
        """Apply `blocks` (default: all) to h [B,L,D] with key-padding mask [B,L] | None. Activation checkpointing
        per block (and per cross-attention layer) when `grad_checkpoint` is set and gradients are being recorded
        (`checkpoint_blocks` overrides the setting, e.g. False when the caller checkpoints a larger unit around this
        call).

        sem: SemanticKV (or SemanticContext with the batch's rows) -> Mechanism C cross-attention after every block
        whose backbone index (`block_ids`, default by block identity) is a cross-attention layer; None = plain
        blocks. The semantic tensors are explicit inputs of every checkpointed unit. gate_log: dict that receives
        {block index: mean gate (detached 0-d tensor)}."""
        blocks = self.blocks if blocks is None else blocks
        want = self.grad_checkpoint if checkpoint_blocks is None else checkpoint_blocks
        ckpt = want and torch.is_grad_enabled() and self.training
        kv = sem if (sem is None or isinstance(sem, SemanticKV)) else self.project_semantic(sem, h.shape[0])
        if kv is not None and not self.has_semantic_xattn:
            kv = None
        ids = None
        if kv is not None:
            if kv.B != h.shape[0]:
                raise ValueError(f"semantic memory has {kv.B} rows for a batch of {h.shape[0]} (select its rows: "
                                 "project_semantic(sem, B, code_rows, sem_rows))")
            ids = list(block_ids) if block_ids is not None else self.block_ids(blocks)
            if len(ids) != len(blocks):
                raise ValueError(f"block_ids has {len(ids)} entries for {len(blocks)} blocks")
            S = kv.S if kv.S.dtype == h.dtype or torch.is_autocast_enabled(h.device.type) else kv.S.to(h.dtype)
        for pos, blk in enumerate(blocks):
            if ckpt:
                h = checkpoint(blk, h, key_mask, attn_bias, use_reentrant=False)
            else:
                h = blk(h, key_mask, attn_bias)
            if ids is None:
                continue
            li = ids[pos]
            mod = self.xattn[str(li)] if (li is not None and str(li) in self.xattn) else None
            if mod is None:
                continue
            if ckpt:
                delta, g = checkpoint(mod, h, S, kv.mask, kv.cond, use_reentrant=False)
            else:
                delta, g = mod(h, S, kv.mask, kv.cond)
            h = h + delta
            if gate_log is not None:
                gate_log[li] = g.detach().float().mean()
        return h

    # ------------------------------------------------------------------ state update
    def state_candidate_of(self, h_state: Tensor, h_state_normed: Tensor) -> Tensor:
        """Candidate state Z~ from the blocks' state-position residual stream h_state = Z_prev + e + sum of block
        updates (e = StatePos + KindEmb(STATE), the input embedding added by `embed_state`).

        * "clamp" (default with the state gate): Z~ = x / max(rms(x), 1) * g with x = h_state - e, g the out_norm
          gain. The slot/kind embedding is an input of the blocks only and is not carried in the state; the
          near-zero initial state Z_0 (std 0.02) is not amplified (Jacobian <= 1; out_norm's normalisation exactly
          once rms(x) >= 1, i.e. from the second decision on, when Z is unit-RMS).
        * "clamp_keep_emb": the same without removing e; "rms": Z~ = out_norm(h_state) (the audit's literal form;
          ablations only: at state_emb_std 0.02 it re-normalises the near-zero initial state, amplifying distractor
          content ~1/rms(h); delay-16 recall was not learned in 300-800 iterations, 2 seeds).
        Without the state gate the candidate is out_norm(h_state) (pre-DOM behaviour exactly)."""
        mode = self.state_candidate
        if self.state_gate is None or mode == "rms":
            return h_state_normed
        x = h_state.float()
        if mode == "clamp":
            x = x - (self.state_pos + self.kind_emb.weight[int(TokenKind.STATE)]).float()
        elif mode != "clamp_keep_emb":
            raise ValueError(f"unknown state_candidate {mode!r}")
        r = x.pow(2).mean(-1, keepdim=True).add(self.out_norm.eps).sqrt()
        return x / r.clamp_min(1.0) * self.out_norm.weight.float()

    def update_state(self, Z_prev: Tensor, Z_new: Tensor) -> Tensor:
        """The carried state when `cfg.state_gate` (float32, outside autocast):
        Z = RMSNorm(Z_prev + u * (Z_new - Z_prev)), u = sigmoid(W [Z_prev; Z_new] + b) (no gain; `state_unit`), so Z
        is unit-RMS per token from the first decision on, as every consumer of the pre-DOM core assumes (world-model
        heads, latent targets; without it the latent world loss was 171 vs 2.1 on the same tokens); else Z_new."""
        if self.state_gate is None:
            return Z_new
        Zp = Z_prev.float()
        with torch.autocast(device_type=Zp.device.type, enabled=False):
            u = torch.sigmoid(self.state_gate(torch.cat([Zp, Z_new.float()], -1)))
        self.last_state_gate = u.detach().mean()
        Z = Zp + u * (Z_new.float() - Zp)
        if self.state_unit:
            Z = torch.nn.functional.rms_norm(Z, (Z.shape[-1],), eps=1e-6)
        return Z

    # ------------------------------------------------------------------ forward
    def forward(self, Z_prev: Tensor, ctx: TokenSet, code_rows: Optional[Tensor] = None,
                sem: Optional[SemanticContext] = None, *, sem_rows: Optional[Tensor] = None,
                blocks=None) -> tuple:
        """Z_prev [B,S,D], ctx [B,N,D] -> (Z [B,S,D] float32, ctx_out [B,N,D] float32; zero at padded/dropped
        positions). `code_rows` [B] maps batch rows to per-episode slot-code rows (imagination branches).
        `sem`: SemanticContext (Mechanism C; rows as in `project_semantic`, `sem_rows` overrides the mapping) or
        None. `blocks`: a subset of the backbone's blocks, e.g. `transition_blocks(k)` (default: all)."""
        B, S, D = Z_prev.shape
        assert S == self.cfg.n_state and D == self.cfg.d_model, (Z_prev.shape, self.cfg.n_state, self.cfg.d_model)
        assert ctx.B == B, (ctx.B, B)
        N = ctx.N
        sel, idx, kept = self.select_budget(ctx)
        self.last_kept = kept
        log: dict = {}
        with self._autocast_ctx(Z_prev.device):
            kv = self.project_semantic(sem, B, code_rows, sem_rows)
            hs = self.embed_state(Z_prev)
            hc = self.embed_ctx(sel, code_rows)
            h = torch.cat([hs, hc], 1)
            key_mask = torch.cat([torch.ones(B, S, dtype=torch.bool, device=h.device), sel.mask], 1)
            if bool(key_mask.all()):
                key_mask = None                                    # nothing masked: flash kernel eligible
            h_raw = self.run_blocks(h, key_mask, blocks, sem=kv, gate_log=log)
            h = self.out_norm(h_raw).float()
        if kv is not None:
            self.last_gates = log
        Z = self.update_state(Z_prev, self.state_candidate_of(h_raw[:, :S], h[:, :S]))
        out_sel = h[:, S:] * sel.mask.unsqueeze(-1).float()
        if idx is None:
            return Z, out_sel
        ctx_out = torch.zeros(B, N, D, device=h.device, dtype=out_sel.dtype)
        ctx_out = ctx_out.scatter(1, idx.unsqueeze(-1).expand(-1, -1, D), out_sel)
        return Z, ctx_out

    # ------------------------------------------------------------------ accounting
    def count_parameters(self) -> dict:
        """Trainable parameters per component of the backbone."""
        out = {
            "state_init": self.z0.numel() + self.state_pos.numel(),
            "kind_embedding": count_parameters(self.kind_emb),
            "time_encoding": count_parameters(self.time_enc),
            "entity_slot_codes": count_parameters(self.slot_codes),
            "ctx_in_norm": count_parameters(self.ctx_in_norm),
            "blocks": count_parameters(self.blocks),
            "per_block": count_parameters(self.blocks[0]) if len(self.blocks) else 0,
            "out_norm": count_parameters(self.out_norm),
            "state_gate": count_parameters(self.state_gate) if self.state_gate is not None else 0,
            "semantic_projection": count_parameters(self.sem_proj) if self.sem_proj is not None else 0,
            "semantic_xattn": count_parameters(self.xattn),
            "semantic_cond": count_parameters(self.cond_proj) if self.cond_proj is not None else 0,
        }
        out["total"] = count_parameters(self)
        return out


#: Mechanism C is part of the backbone (built from cfg.sem_xattn_every / cfg.sem_xattn_dim); the former subclass
#: name stays importable from here and from core.semantic_xattn.
GatedUnifiedCore = UnifiedCore
