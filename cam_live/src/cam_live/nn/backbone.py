"""cam_live.nn.backbone — the pretrained semantic backbone, its LoRA adapters and the latent bridge (spec 1.2, 1.4).

Two implementations with one interface:

* :class:`SemanticBackbone` — a Hugging Face causal LM (default ``Qwen/Qwen3-4B`` at the pinned local snapshot),
  frozen, bf16 on GPU.
* :class:`TinyBackbone` (``name="tiny"``) — a small *random* causal transformer LM with a deterministic word-hash
  tokenizer, CPU, for mathematical and gradient-path tests only. It has no pretrained knowledge and is never
  reported as the backbone.

Shared machinery (:class:`_Backbone`):

* **LoRA** (:class:`LoRALinear`) on the configured attention projections of the configured layers. The adapter
  weights are kept in float32 (a bf16 master weight would round small optimizer steps away) and cast in the forward.
* **Semantic embedding**: masked mean of the hidden states at several depths (``tap_layers``), concatenated and
  standardised per dimension with statistics of a fixed neutral calibration corpus (mid-layer LM states carry a few
  huge dimensions shared by every text). Results are cached by text hash; ``with_grad=True`` bypasses the cache and
  builds a graph into the adapters; the cache is invalidated automatically when any adapter tensor changes.
* **Latent bridge** (:class:`LatentBridge`): CAM's belief ``b`` is mapped to ``bridge_tokens`` virtual prefix
  tokens that every sequence position reads through cross-attention added to the residual stream of several decoder
  layers. The cross-attention output projection is zero-initialised, so the bridge is an *exact* no-op at
  initialisation yet receives gradient immediately. (Virtual tokens placed literally in the sequence cannot be an
  exact no-op: even an all-zero embedding takes softmax attention mass from the real tokens; and a text-initialised
  prefix leaked into generated strings in an earlier system.)
* ``generate`` (sampling with a KV cache, per-call generator) and ``generation_logprob`` (differentiable w.r.t. the
  adapters, the bridge and the latent), gradient checkpointing for the adapter path.

Provenance: the loading pattern and the calibration idea follow
``superlearner/cam_unified/semantic/encoder.py`` (commit 3968583); the gated cross-attention idea follows
``cam_unified/semantic/decoder_xattn.py``. Rewritten here; nothing is imported from those packages.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import math
import os
import re
import time
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as Fn
import torch.utils.checkpoint as torch_checkpoint
from torch import Tensor, nn

from ..config import BackboneConfig
from .core import CrossAttention, RMSNorm
from .presets import NNConfig

#: fixed, neutral calibration corpus for the per-dimension standardisation of pooled hidden states
CALIBRATION_TEXTS: Tuple[str, ...] = (
    "seconds", "milliseconds", "days", "count", "fraction", "percent", "bytes", "kilograms", "requests per second",
    "42", "-3.5", "1e9", "0", "yes", "no", "true", "false", "null",
    "The request finished without errors.", "The page did not load.", "A file was written to disk.",
    "def add(a, b):\n    return a + b\n", "import numpy as np\nprint(np.arange(3).sum())",
    "SELECT name FROM users WHERE id = 3;", "for (let i = 0; i < n; i++) { total += x[i]; }",
    '{"ok": true, "result": {"value": 1200}}', '{"error": "timeout", "after_ms": 2000}',
    "<html><body><p>Hello</p></body></html>", "<button id=\"submit\" disabled>Send</button>",
    "Traceback (most recent call last):", "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
    "HTTP 404 Not Found", "HTTP 200 OK", "permission denied", "connection reset by peer",
    "a person", "an organisation", "a document", "a repository", "a web page", "a table with three columns",
    "the temperature rose sharply during the afternoon", "the river is wider than the road",
    "seven is a prime number", "the quick brown fox jumps over the lazy dog",
    "click the element with the given reference", "type text into the field", "read the file and return its text",
    "wait for the given number of seconds", "open the address in a new tab",
    "accessibility tree: button 'Next' enabled", "link: Documentation", "heading: Settings", "input: search",
    "objective: reach the stated target within the time limit", "measurement after two days",
    "Zur Lieferung bitte bis Freitag bestaetigen.", "La bibliotheque ouvre a neuf heures.", "数据库连接失败",
    "x = 3 * y + 2", "the integral of x squared is x cubed over three", "[1, 2, 3, 5, 8, 13]",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


class LoRALinear(nn.Module):
    """``base(x) + (alpha / r) * B(A(x))`` with a frozen ``base``. A is float32, uniform(-1/sqrt(in), 1/sqrt(in));
    B is float32 zeros (the adapter starts as an exact no-op)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, generator: Optional[torch.Generator] = None):
        super().__init__()
        self.base = base
        self.rank, self.scale = int(rank), float(alpha) / float(rank)
        dev = base.weight.device
        bound = 1.0 / math.sqrt(base.in_features)
        a = (torch.rand(rank, base.in_features, generator=generator) * 2.0 - 1.0) * bound
        self.lora_A = nn.Parameter(a.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))

        self._cast_key: Any = None
        self._cast: Tuple[Optional[Tensor], Optional[Tensor]] = (None, None)

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        A, B = self.lora_A, self.lora_B
        if torch.is_grad_enabled() and (A.requires_grad or B.requires_grad):
            return out + self.scale * Fn.linear(Fn.linear(x, A.to(x.dtype)), B.to(x.dtype))
        # inference: reuse the dtype-cast weights until an adapter tensor changes (identity, storage and version
        # are all part of the key, so functional parameter substitution is seen too); skip an all-zero adapter
        key = (id(A), A.data_ptr(), A._version, id(B), B.data_ptr(), B._version, x.dtype, x.device)
        if key != self._cast_key:
            zero = not bool(B.detach().any())
            self._cast = (None, None) if zero else (A.detach().to(x.device, x.dtype), B.detach().to(x.device, x.dtype))
            self._cast_key = key
        a, b = self._cast
        if a is None:
            return out
        return out + self.scale * Fn.linear(Fn.linear(x, a), b)


class LatentBridge(nn.Module):
    """belief ``b`` [n, d_latent] -> ``n_tokens`` virtual prefix tokens [n, K, d_x], read by cross-attention from the
    residual stream of the hooked decoder layers. ``xattn[i].out`` is zero-initialised: exact no-op at init."""

    def __init__(self, d_latent: int, hidden: int, n_hooked: int, n_tokens: int, d_x: int, n_heads: int):
        super().__init__()
        self.n_tokens, self.d_x = n_tokens, d_x
        self.in_norm = RMSNorm(d_latent, affine=False)
        self.to_mem = nn.Linear(d_latent, n_tokens * d_x)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_x) * 0.5)
        self.mem_norm = RMSNorm(d_x)
        self.xattn = nn.ModuleList(CrossAttention(hidden, d_x, d_x, n_heads) for _ in range(n_hooked))
        for x in self.xattn:
            nn.init.normal_(x.q.weight, std=1.0 / math.sqrt(hidden))
            nn.init.zeros_(x.out.weight)
        #: per-dimension statistics of the hooked layers' hidden states on the calibration corpus. LM hidden states
        #: carry a few huge dimensions shared by every token: queries are standardised per dimension, and the
        #: injected residual is scaled by the *typical* per-dimension spread (median), not the outlier-dominated RMS.
        self.register_buffer("h_mu", torch.zeros(n_hooked, hidden))
        self.register_buffer("h_sd", torch.ones(n_hooked, hidden))
        self.register_buffer("h_scale", torch.ones(n_hooked))

    def memory(self, latent: Tensor) -> Tensor:
        m = self.to_mem(self.in_norm(latent.to(self.pos.dtype))).view(-1, self.n_tokens, self.d_x)
        return self.mem_norm(m + self.pos)

    def delta(self, slot: int, h: Tensor, mem: Tensor) -> Tensor:
        hq = (h.to(mem.dtype) - self.h_mu[slot]) / self.h_sd[slot]
        return (self.h_scale[slot] * self.xattn[slot](hq, mem)).to(h.dtype)


# --------------------------------------------------------------------------------------------------------------------
# The tiny random LM (tests only)
# --------------------------------------------------------------------------------------------------------------------
class _TinyAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.h = n_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x: Tensor, neg_mask: Tensor) -> Tensor:
        B, T, D = x.shape
        sp = lambda t: t.view(B, T, self.h, D // self.h).transpose(1, 2)
        att = sp(self.q_proj(x)) @ sp(self.k_proj(x)).transpose(-1, -2) / math.sqrt(D // self.h) + neg_mask
        return self.o_proj((torch.softmax(att, -1) @ sp(self.v_proj(x))).transpose(1, 2).reshape(B, T, D))


class _TinyLayer(nn.Module):
    gradient_checkpointing = False

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.input_layernorm = RMSNorm(d)
        self.self_attn = _TinyAttention(d, n_heads)
        self.post_attention_layernorm = RMSNorm(d)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def _block(self, h: Tensor, neg_mask: Tensor) -> Tensor:
        h = h + self.self_attn(self.input_layernorm(h), neg_mask)
        return h + self.down(Fn.silu(self.up(self.post_attention_layernorm(h))))

    def forward(self, h: Tensor, neg_mask: Tensor) -> Tensor:
        # like transformers' GradientCheckpointingLayer: forward is checkpointed, module hooks stay outside
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return torch_checkpoint.checkpoint(self._block, h, neg_mask, use_reentrant=False)
        return self._block(h, neg_mask)


class TinyLM(nn.Module):
    """Small bias-free pre-norm causal transformer with tied output embedding and sinusoidal positions."""

    def __init__(self, vocab: int, d: int, n_layers: int, n_heads: int = 4, seed: int = 0):
        super().__init__()
        self.vocab, self.d = vocab, d
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(_TinyLayer(d, n_heads) for _ in range(n_layers))
        self.norm = RMSNorm(d)
        g = torch.Generator().manual_seed(7919 + int(seed))
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.dim() >= 2:
                    std = 1.0 if name == "embed_tokens.weight" else 1.0 / math.sqrt(p.shape[1])
                    p.copy_(torch.randn(p.shape, generator=g) * std)

    def positions(self, attn: Tensor) -> Tensor:
        pos = (attn.long().cumsum(1) - 1).clamp_min(0).to(torch.float32).unsqueeze(-1)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.d, 2, device=attn.device, dtype=torch.float32) / self.d))
        return torch.cat([torch.sin(pos * inv), torch.cos(pos * inv)], -1)

    def forward(self, ids: Tensor, attn: Tensor) -> Tensor:
        n, L = ids.shape
        h = self.embed_tokens(ids) + self.positions(attn)
        ar = torch.arange(L, device=ids.device)
        ok = (ar[None, :] <= ar[:, None])[None] & (attn.bool()[:, None, :] | torch.eye(L, dtype=torch.bool,
                                                                                         device=ids.device)[None])
        neg = torch.zeros(n, 1, L, L, device=ids.device).masked_fill(~ok[:, None], float("-inf"))
        for layer in self.layers:
            h = layer(h, neg)
        return self.norm(h)

    def logits(self, h: Tensor) -> Tensor:
        return Fn.linear(h, self.embed_tokens.weight) / math.sqrt(self.d)


# --------------------------------------------------------------------------------------------------------------------
# Shared backbone machinery
# --------------------------------------------------------------------------------------------------------------------
class _Backbone(nn.Module):
    name: str
    revision: str
    pretrained: bool
    hidden: int
    n_layers: int

    def __init__(self, cfg: BackboneConfig, nn_cfg: NNConfig, d_latent: int):
        super().__init__()
        self.cfg, self.nn_cfg, self.d_latent = cfg, nn_cfg, int(d_latent)
        self.stats: Dict[str, float] = {"cache_hits": 0, "cache_misses": 0, "embedded_texts": 0,
                                        "embedded_tokens": 0, "embed_s": 0.0, "truncated_tokens": 0,
                                        "generated_tokens": 0, "generate_s": 0.0, "load_s": 0.0,
                                        "cache_invalidations": 0}
        self._cache: "OrderedDict[str, Tensor]" = OrderedDict()
        self._mem: Optional[Tensor] = None
        self._capture: Optional[Dict[int, Tensor]] = None
        self.adapter_version = 0

    # The two backbones implement: _lm_module, _layer_list, _encode, _decode, _forward_hidden (final hidden
    # states; layer hooks fire), _step (one cached decoding step -> last-position logits) and _logits.

    # ---- construction ------------------------------------------------------------------------------------------------
    def _finish_init(self, seed: int = 0) -> None:
        cfg, nn_cfg = self.cfg, self.nn_cfg
        lm = self._lm_module()
        for p in lm.parameters():
            p.requires_grad_(False)
        lm.eval()
        self.n_frozen = sum(p.numel() for p in lm.parameters())
        layers = self._layer_list()
        # LoRA
        if cfg.lora_layers == "all":
            lora_ids = list(range(self.n_layers))
        else:
            a, b = cfg.lora_layers.split(":")
            lora_ids = list(range(self.n_layers))[int(a) if a else None: int(b) if b else None]
        g = torch.Generator().manual_seed(104729 + int(seed))
        self.lora_modules: List[LoRALinear] = []
        if cfg.lora_rank > 0:
            for i in lora_ids:
                for target in cfg.lora_targets:
                    parent = layers[i].self_attn
                    base = getattr(parent, target, None)
                    if not isinstance(base, nn.Linear):
                        raise ValueError(f"LoRA target {target!r} is not a linear projection of the attention block")
                    wrapped = LoRALinear(base, cfg.lora_rank, cfg.lora_alpha, g)
                    setattr(parent, target, wrapped)
                    self.lora_modules.append(wrapped)
        self._lora_params: List[nn.Parameter] = [p for m in self.lora_modules for p in (m.lora_A, m.lora_B)]
        self._fp = self._fingerprint()
        # taps and bridge
        self.tap_ids = [max(0, min(self.n_layers - 1, int(round(f * self.n_layers)) - 1)) for f in cfg.tap_layers]
        self.n_taps = len(self.tap_ids)
        self.d_sem = self.n_taps * self.hidden
        n_hook = max(1, min(nn_cfg.bridge_layers, self.n_layers))
        self.bridge_ids = sorted({int(round(x)) for x in torch.linspace(0, self.n_layers - 1, n_hook).tolist()})
        d_x = max(nn_cfg.bridge_heads, min(nn_cfg.bridge_dim, self.hidden) // nn_cfg.bridge_heads * nn_cfg.bridge_heads)
        with torch.random.fork_rng(devices=[]):                 # deterministic init, global RNG untouched
            torch.manual_seed(15_485_863 + int(seed))
            bridge = LatentBridge(self.d_latent, self.hidden, len(self.bridge_ids), max(1, cfg.bridge_tokens), d_x,
                                  nn_cfg.bridge_heads)
        self.bridge = bridge.to(self.device)
        self._bridge_slot = {i: s for s, i in enumerate(self.bridge_ids)}
        for i in sorted(set(self.tap_ids) | set(self.bridge_ids)):
            layers[i].register_forward_hook(functools.partial(self._layer_hook, i))
        if cfg.grad_checkpoint:
            self._enable_checkpointing()
        self.register_buffer("cal_mu", torch.zeros(self.d_sem, device=self.device))
        self.register_buffer("cal_sd", torch.ones(self.d_sem, device=self.device))
        self._calibrate()

    def _enable_checkpointing(self) -> None:
        for layer in self._layer_list():
            layer.gradient_checkpointing = True

    @property
    def device(self) -> torch.device:
        return next(self._lm_module().parameters()).device

    def train(self, mode: bool = True) -> "_Backbone":
        super().train(mode)
        self._lm_module().eval()         # the frozen LM leaves eval mode only transiently, for checkpointing
        return self

    # ---- parameters ----------------------------------------------------------------------------------------------------
    def adapter_parameters(self) -> List[nn.Parameter]:
        return list(self._lora_params) + list(self.bridge.parameters())

    def lora_parameters(self) -> List[nn.Parameter]:
        return list(self._lora_params)

    def frozen_parameters(self) -> List[nn.Parameter]:
        own = {id(p) for p in self._lora_params}
        return [p for p in self._lm_module().parameters() if id(p) not in own]

    def _fingerprint(self) -> Tuple[int, ...]:
        return tuple(p._version for p in self._lora_params)

    def mark_adapters_updated(self) -> None:
        """Drop cached embeddings (called automatically when an adapter tensor's version counter changed)."""
        self._cache.clear()
        self.adapter_version += 1
        self.stats["cache_invalidations"] += 1
        self._fp = self._fingerprint()

    # ---- hooks ----------------------------------------------------------------------------------------------------------
    def _layer_hook(self, idx: int, _module: nn.Module, _inputs: Any, output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        changed = False
        if self._mem is not None and idx in self._bridge_slot:
            h = h + self.bridge.delta(self._bridge_slot[idx], h, self._mem)
            changed = True
        if self._capture is not None:
            self._capture[idx] = h
        if not changed:
            return None
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    @contextlib.contextmanager
    def _bridged(self, mem: Optional[Tensor]) -> Iterator[None]:
        prev, self._mem = self._mem, mem
        try:
            yield
        finally:
            self._mem = prev

    def _hidden(self, ids: Tensor, attn: Tensor) -> Tensor:
        """Final hidden states [n, L, H]; with gradients enabled the layers run under gradient checkpointing."""
        ckpt = bool(self.cfg.grad_checkpoint) and torch.is_grad_enabled()
        lm = self._lm_module()
        if ckpt:
            lm.train()
        try:
            return self._forward_hidden(ids, attn)
        finally:
            if ckpt:
                lm.eval()

    # ---- tokenisation ---------------------------------------------------------------------------------------------------
    def _text_ids(self, text: str, limit: int) -> List[int]:
        ids = self._encode(text) if text else []
        if not ids:
            ids = [self.pad_id]
        if len(ids) > limit:                       # keep the head and the tail; the cut is counted, never silent
            self.stats["truncated_tokens"] += len(ids) - limit
            tail = limit // 4
            ids = ids[: limit - tail] + (ids[-tail:] if tail else [])
        return ids

    def _prompt_ids(self, prompt: str) -> List[int]:
        ids = self._encode(prompt) if prompt else []
        if not ids:
            ids = [self.pad_id]
        limit = self.nn_cfg.max_prompt_tokens
        if len(ids) > limit:                       # a prompt keeps its end (the part generation continues from)
            self.stats["truncated_tokens"] += len(ids) - limit
            ids = ids[-limit:]
        return ids

    def _pad(self, rows: Sequence[Sequence[int]], left: bool = False) -> Tuple[Tensor, Tensor]:
        L = max(len(r) for r in rows)
        ids = torch.full((len(rows), L), self.pad_id, dtype=torch.long)
        attn = torch.zeros(len(rows), L, dtype=torch.long)
        for i, r in enumerate(rows):
            sl = slice(L - len(r), L) if left else slice(0, len(r))
            ids[i, sl] = torch.tensor(list(r), dtype=torch.long)
            attn[i, sl] = 1
        return ids.to(self.device), attn.to(self.device)

    # ---- semantic embedding ---------------------------------------------------------------------------------------------
    def _pooled(self, rows: Sequence[Sequence[int]], captures: Optional[List[Any]] = None) -> Tensor:
        """Raw (unstandardised) float32 [n, d_sem]: masked mean of the tapped layers' states, taps concatenated.
        ``captures`` (calibration only) receives (hooked layers' hidden states, attention mask) of every batch."""
        order = sorted(range(len(rows)), key=lambda i: -len(rows[i]))
        outs: List[Tensor] = []
        i = 0
        while i < len(order):
            L = len(rows[order[i]])
            grp = order[i: i + max(1, self.nn_cfg.embed_batch_tokens // max(L, 1))]
            ids, attn = self._pad([rows[j] for j in grp])
            prev, self._capture = self._capture, {}
            try:
                self._hidden(ids, attn)
                cap = self._capture
            finally:
                self._capture = prev
            m = attn.to(torch.float32).unsqueeze(-1)
            outs.append(torch.cat([(cap[t].to(torch.float32) * m).sum(1) / m.sum(1) for t in self.tap_ids], -1))
            if captures is not None:
                captures.append((cap, attn))
            i += len(grp)
            self.stats["embedded_tokens"] += int(attn.sum())
        inv = torch.empty(len(order), dtype=torch.long, device=self.device)
        inv[torch.tensor(order, device=self.device)] = torch.arange(len(order), device=self.device)
        return torch.cat(outs, 0).index_select(0, inv)

    @torch.no_grad()
    def _calibrate(self) -> None:
        rows = [self._text_ids(t, self.cfg.max_text_tokens) for t in CALIBRATION_TEXTS]
        toks: Dict[int, List[Tensor]] = {i: [] for i in self.bridge_ids}
        raws = []
        for s in range(0, len(rows), 16):
            caps: List[Any] = []
            raws.append(self._pooled(rows[s: s + 16], caps))
            for cap, attn in caps:
                for i in self.bridge_ids:
                    toks[i].append(cap[i].to(torch.float32)[attn.bool()])        # real tokens only, [n_tok, H]
        raw = torch.cat(raws, 0)
        sd = raw.std(0)
        self.cal_mu.copy_(raw.mean(0))
        self.cal_sd.copy_(sd.clamp_min(max(1e-4, 0.05 * float(sd.median()))))
        for slot, i in enumerate(self.bridge_ids):
            h = torch.cat(toks[i], 0)
            hsd = h.std(0)
            typical = float(hsd.median())
            self.bridge.h_mu[slot].copy_(h.mean(0))
            self.bridge.h_sd[slot].copy_(hsd.clamp_min(max(1e-6, 0.05 * typical)))
            self.bridge.h_scale[slot] = typical
        self.stats["embedded_tokens"] = 0

    def _standardise(self, raw: Tensor) -> Tensor:
        """Per-dimension standardisation with the calibration statistics, winsorised at +-20 so one extreme
        dimension of an unusual text cannot dominate the tokenizer's per-tap normalisation."""
        return ((raw - self.cal_mu) / self.cal_sd).clamp(-20.0, 20.0)

    def embed(self, texts: Sequence[str], *, with_grad: bool = False) -> Tensor:
        """[n, d_sem] float32 standardised multi-depth embeddings. ``with_grad=False``: detached and cached by text
        hash (stored as float16); ``with_grad=True``: recomputed with a graph into the adapters, cache bypassed."""
        texts = list(texts)
        if not texts:
            return torch.zeros(0, self.d_sem, device=self.device)
        t0 = time.time()
        if with_grad:
            uniq = list(dict.fromkeys(texts))
            with torch.enable_grad():
                emb = self._standardise(self._pooled([self._text_ids(t, self.cfg.max_text_tokens) for t in uniq]))
            pos = {t: i for i, t in enumerate(uniq)}
            out = emb.index_select(0, torch.tensor([pos[t] for t in texts], device=self.device))
            self.stats["embedded_texts"] += len(uniq)
            self.stats["embed_s"] += time.time() - t0
            return out
        if self._fingerprint() != self._fp:
            self.mark_adapters_updated()
        keys = [_sha(t) for t in texts]
        todo: "OrderedDict[str, str]" = OrderedDict()
        for k, t in zip(keys, texts):
            if k in self._cache:
                self._cache.move_to_end(k)
                self.stats["cache_hits"] += 1
            elif k not in todo:
                todo[k] = t
        fresh: Dict[str, Tensor] = {}
        if todo:
            self.stats["cache_misses"] += len(todo)
            with torch.no_grad():
                emb = self._standardise(self._pooled([self._text_ids(t, self.cfg.max_text_tokens)
                                                      for t in todo.values()]))
            emb = emb.to(torch.float16).cpu()
            for k, e in zip(todo, emb):
                fresh[k] = e
                if self.cfg.embed_cache_items > 0:
                    self._cache[k] = e
            while len(self._cache) > self.cfg.embed_cache_items:
                self._cache.popitem(last=False)
            self.stats["embedded_texts"] += len(todo)
        out = torch.stack([fresh[k] if k in fresh else self._cache[k] for k in keys], 0)
        self.stats["embed_s"] += time.time() - t0
        return out.to(self.device, torch.float32)

    # ---- generation -------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompts: Sequence[str], *, latent: Optional[Tensor] = None, max_new_tokens: int = 256,
                 temperature: float = 0.8, num_return_sequences: int = 1, seed: Optional[int] = None,
                 top_p: float = 1.0) -> List[List[str]]:
        """Sample continuations; ``latent`` [n, d_latent] conditions them through the bridge. ``temperature <= 0`` is
        greedy. With ``seed`` the samples are reproducible and the global RNG is untouched."""
        prompts = list(prompts)
        r = max(1, int(num_return_sequences))
        rows = [self._prompt_ids(p) for p in prompts for _ in range(r)]
        mem = None
        if latent is not None:
            if latent.shape[0] != len(prompts):
                raise ValueError("latent must have one row per prompt")
            mem = self.bridge.memory(latent.to(self.device)).repeat_interleave(r, 0)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(seed))
        t0 = time.time()
        texts: List[str] = []
        bs = max(1, self.nn_cfg.gen_batch)
        for s in range(0, len(rows), bs):
            texts += self._generate_rows(rows[s: s + bs], None if mem is None else mem[s: s + bs], max_new_tokens,
                                         temperature, top_p, gen)
        self.stats["generate_s"] += time.time() - t0
        return [texts[i * r: (i + 1) * r] for i in range(len(prompts))]

    def _generate_rows(self, rows: Sequence[Sequence[int]], mem: Optional[Tensor], max_new: int, temperature: float,
                       top_p: float, gen: Optional[torch.Generator]) -> List[str]:
        ids, attn = self._pad(rows, left=True)
        n = ids.shape[0]
        eos = torch.tensor(sorted(self.eos_ids), device=self.device)
        done = torch.zeros(n, dtype=torch.bool, device=self.device)
        new: List[Tensor] = []
        cache, cur = None, ids
        with self._bridged(mem):
            for _ in range(int(max_new)):
                logits, cache = self._step(cur, attn, cache)
                if temperature <= 0:
                    nxt = logits.argmax(-1)
                else:
                    probs = torch.softmax(logits / temperature, -1)
                    if top_p < 1.0:
                        sp, si = probs.sort(-1, descending=True)
                        keep = (sp.cumsum(-1) - sp) < top_p
                        probs = torch.zeros_like(probs).scatter(-1, si, sp * keep)
                    nxt = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
                nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
                new.append(nxt)
                done = done | torch.isin(nxt, eos)
                attn = torch.cat([attn, torch.ones(n, 1, dtype=attn.dtype, device=self.device)], 1)
                cur = nxt.unsqueeze(1)
                if bool(done.all()):
                    break
        out = torch.stack(new, 1).tolist() if new else [[] for _ in range(n)]
        texts = []
        for row in out:
            cut = next((j for j, t in enumerate(row) if t in self.eos_ids), len(row))
            self.stats["generated_tokens"] += cut
            texts.append(self._decode(row[:cut]))
        return texts

    def generation_logprob(self, prompts: Sequence[str], completions: Sequence[str], *,
                           latent: Optional[Tensor] = None, append_eos: bool = False) -> Tensor:
        """[n] float32: sum of log p(completion token | prompt, earlier completion tokens[, latent]). Differentiable
        w.r.t. the LoRA adapters, the bridge and ``latent``; frozen weights get no gradient."""
        prompts, completions = list(prompts), list(completions)
        if len(prompts) != len(completions):
            raise ValueError("one completion per prompt")
        P = [self._prompt_ids(p) for p in prompts]
        C = [(self._encode(c) if c else []) + ([self.eos_id] if append_eos else []) for c in completions]
        ids, attn = self._pad([p + c for p, c in zip(P, C)])
        mem = None
        if latent is not None:
            if latent.shape[0] != len(prompts):
                raise ValueError("latent must have one row per prompt")
            mem = self.bridge.memory(latent.to(self.device))
        with self._bridged(mem):
            h = self._hidden(ids, attn)
        row = torch.tensor([i for i, c in enumerate(C) for _ in c], dtype=torch.long, device=self.device)
        pos = torch.tensor([len(P[i]) - 1 + j for i, c in enumerate(C) for j in range(len(c))], dtype=torch.long,
                           device=self.device)
        tgt = torch.tensor([t for c in C for t in c], dtype=torch.long, device=self.device)
        out = torch.zeros(len(prompts), dtype=torch.float32, device=self.device)
        if tgt.numel() == 0:
            return out + 0.0 * h.to(torch.float32).sum()
        logp = torch.log_softmax(self._logits(h[row, pos]).to(torch.float32), -1).gather(1, tgt[:, None]).squeeze(1)
        return out.index_add(0, row, logp)

    def next_token_logits(self, prompts: Sequence[str], *, latent: Optional[Tensor] = None) -> Tensor:
        """[n, vocab] float32 logits of the token following each prompt (used to verify the bridge no-op exactly)."""
        P = [self._prompt_ids(p) for p in prompts]
        ids, attn = self._pad(P)
        mem = None if latent is None else self.bridge.memory(latent.to(self.device))
        with self._bridged(mem):
            h = self._hidden(ids, attn)
        last = torch.tensor([len(p) - 1 for p in P], device=self.device)
        return self._logits(h[torch.arange(len(P), device=self.device), last]).to(torch.float32)

    def chat_prompt(self, user: str, system: str = "", thinking: bool = False) -> str:
        """Wrap a request in the backbone's chat format, ending where the assistant's answer starts."""
        return (f"{system}\n" if system else "") + f"{user}\n"

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "revision": self.revision, "pretrained": self.pretrained, "hidden": self.hidden,
                "n_layers": self.n_layers, "d_sem": self.d_sem, "tap_layer_ids": list(self.tap_ids),
                "lora": {"rank": self.cfg.lora_rank, "alpha": self.cfg.lora_alpha,
                         "targets": list(self.cfg.lora_targets), "layers": self.cfg.lora_layers,
                         "modules": len(self.lora_modules)},
                "bridge": {"tokens": self.bridge.n_tokens, "dim": self.bridge.d_x, "layer_ids": list(self.bridge_ids)},
                "dtype": str(next(self._lm_module().parameters()).dtype).replace("torch.", "")}


class TinyBackbone(_Backbone):
    """``name="tiny"``: random causal LM + word-hash tokenizer. For CPU tests of mathematics and plumbing only."""

    _PIECE = re.compile(r"[A-Za-z_]+|\d|[^\sA-Za-z_\d]")

    def __init__(self, cfg: BackboneConfig, nn_cfg: Optional[NNConfig] = None, d_latent: int = 64,
                 device: Optional[str] = None, seed: int = 0):
        super().__init__(cfg, nn_cfg or NNConfig(), d_latent)
        self.name, self.revision, self.pretrained = "tiny", "", False
        self.hidden, self.n_layers = int(cfg.tiny_d), int(cfg.tiny_layers)
        self.vocab = int(cfg.tiny_vocab)
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2
        self.eos_ids = {self.eos_id}
        self._pieces: Dict[int, str] = {}
        self.lm = TinyLM(self.vocab, self.hidden, self.n_layers, seed=seed).to(torch.device(device or "cpu"))
        self._finish_init(seed)

    def _lm_module(self) -> nn.Module:
        return self.lm

    def _layer_list(self) -> List[nn.Module]:
        return list(self.lm.layers)

    def _encode(self, text: str) -> List[int]:
        ids = []
        for piece in self._PIECE.findall(text):
            i = 3 + int.from_bytes(hashlib.blake2b(piece.encode("utf-8", "surrogatepass"), digest_size=8).digest(),
                                   "little") % (self.vocab - 3)
            self._pieces.setdefault(i, piece)
            ids.append(i)
        return ids

    def _decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._pieces.get(int(i), f"<{int(i)}>") for i in ids if int(i) > 2)

    def _forward_hidden(self, ids: Tensor, attn: Tensor) -> Tensor:
        return self.lm(ids, attn)

    def _step(self, ids: Tensor, attn: Tensor, cache: Any) -> Tuple[Tensor, Any]:
        full = ids if cache is None else torch.cat([cache, ids], 1)        # no KV cache: recompute (tiny model)
        return self.lm.logits(self.lm(full, attn)[:, -1]).to(torch.float32), full

    def _logits(self, h: Tensor) -> Tensor:
        return self.lm.logits(h)


class SemanticBackbone(_Backbone):
    """A Hugging Face causal LM (frozen) with LoRA adapters and the latent bridge. Default: Qwen/Qwen3-4B at the
    pinned local snapshot, bf16 on CUDA (float32 on CPU)."""

    def __init__(self, cfg: BackboneConfig, nn_cfg: Optional[NNConfig] = None, d_latent: int = 768,
                 device: Optional[str] = None, seed: int = 0):
        super().__init__(cfg, nn_cfg or NNConfig(), d_latent)
        hf_home = os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
        if cfg.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = getattr(torch, cfg.dtype) if dev.type == "cuda" else torch.float32
        kw: Dict[str, Any] = {"local_files_only": cfg.local_files_only, "cache_dir": os.path.join(hf_home, "hub")}
        if cfg.revision:
            kw["revision"] = cfg.revision
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(cfg.name, **kw)
        try:
            lm = AutoModelForCausalLM.from_pretrained(cfg.name, dtype=dtype, **kw)
        except TypeError:                                        # transformers < 4.56 names the argument torch_dtype
            lm = AutoModelForCausalLM.from_pretrained(cfg.name, torch_dtype=dtype, **kw)
        self.lm = lm.to(dev).eval()
        self.name, self.pretrained = cfg.name, True
        self.revision = cfg.revision or str(getattr(lm.config, "_commit_hash", "") or "")
        self.hidden, self.n_layers = int(lm.config.hidden_size), int(lm.config.num_hidden_layers)
        self.vocab = int(lm.config.vocab_size)
        self.pad_id = int(self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id)
        eos = {self.tok.eos_token_id}
        for t in ("<|im_end|>", "<|endoftext|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                eos.add(i)
        self.eos_ids = {int(i) for i in eos if i is not None}
        self.eos_id = int(self.tok.eos_token_id)
        params = inspect.signature(self.lm.forward).parameters
        self._cache_kw = "cache" if "cache" in params else "past_key_values"
        self._finish_init(seed)
        self.stats["load_s"] = time.time() - t0

    def _lm_module(self) -> nn.Module:
        return self.lm

    def _layer_list(self) -> List[nn.Module]:
        return list(self.lm.model.layers)

    def _enable_checkpointing(self) -> None:
        fn = functools.partial(torch_checkpoint.checkpoint, use_reentrant=False)
        try:
            self.lm._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=fn)
        except (AttributeError, TypeError):
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def _encode(self, text: str) -> List[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def _decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)

    def _forward_hidden(self, ids: Tensor, attn: Tensor) -> Tensor:
        return self.lm.model(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state

    def _step(self, ids: Tensor, attn: Tensor, cache: Any) -> Tuple[Tensor, Any]:
        out = self.lm(input_ids=ids, attention_mask=attn, use_cache=True, **{self._cache_kw: cache})
        new_cache = getattr(out, "cache", None)
        if new_cache is None:
            new_cache = getattr(out, "past_key_values", None)
        return out.logits[:, -1].to(torch.float32), new_cache

    def _logits(self, h: Tensor) -> Tensor:
        return self.lm.lm_head(h)

    def chat_prompt(self, user: str, system: str = "", thinking: bool = False) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=thinking)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_backbone(cfg: BackboneConfig, nn_cfg: Optional[NNConfig] = None, d_latent: int = 768,
                   device: Optional[str] = None, seed: int = 0) -> _Backbone:
    if cfg.name == "tiny":
        return TinyBackbone(cfg, nn_cfg, d_latent, device, seed)
    return SemanticBackbone(cfg, nn_cfg, d_latent, device, seed)
