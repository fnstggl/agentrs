"""QwenSemantic: the pretrained semantic backbone of CAM Unified (workstream A, spec Part 2.2).

One wrapper around a Hugging Face causal LM (Qwen3-1.7B by default, Qwen3-0.6B in tests) with two paths:

* the ENCODER path (`hidden`, `token_states`, `hidden_chunks`, `embed`) reads token-level hidden states of a
  configurable decoder layer through a forward hook and aborts the forward right after that layer (the layers above
  it are never computed). With `cfg.sem_encoder_lora = False` (Stages 1-3) this path runs with the PEFT adapter
  disabled, i.e. on the frozen base weights, so its content-hash cache stays valid while the decoder's LoRA trains.
* the DECODER path (`decoder_hidden`, `lm_logits`, `step`) runs the full model with the LoRA adapter enabled. It is
  used by the action decoder (`grammar.py` / `actions.py`), which injects CAM's bridge prefix through
  `inputs_embeds` (`input_embeddings()` exposes the embedding matrix).

Cache: hidden states are cached per text under the key (sha256(text), version) in an LRU bounded by bytes. `version`
names everything the cached states depend on (model, revision, layer, dtype, chunking, and - only when the encoder path
uses LoRA - the LoRA update counter), so a cached state is never served across versions. `mark_updated()` must be
called after every optimiser step on the LoRA weights; it bumps `decoder_version` always and `version` only when the
encoder path uses LoRA (then stale entries are purged).

Long texts are split into `sem_chunk_tokens`-token chunks, each encoded independently (no cross-chunk attention);
the per-text state is the concatenation of its chunks ([T_i, H], T_i <= max_text_tokens; tokens beyond that are
dropped and counted in `stats["truncated_tokens"]`, never silently).
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..config import CamConfig


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


# Fixed calibration corpus for the per-dimension standardisation of encoder states (mid-layer LLM states carry a few
# huge outlier dimensions shared by every token; without centring, all pooled embeddings look alike).
CALIBRATION_TEXTS = (
    "cents", "ms", "seconds", "days", "count", "fraction", "USD/month", "cents/core-hour", "requests per second",
    "kg", "percent", "bytes", "The customer asked for a refund.", "Deploy the service to the cloud.",
    "def add(a, b):\n    return a + b\n", '{"ok": true, "result": {"balance": 1200}}', "error: timeout after 2000 ms",
    "SELECT name FROM users WHERE id = 3;", "<html><body><p>Price: $10</p></body></html>", "rig.report tick 7",
    "bank account", "product", "repository", "a person", "an organisation", "revenue", "cost", "profit margin",
    "search the web for pages", "send a message to another participant", "the reactor temperature rose sharply",
    "Hire two engineers for the new team.", "import numpy as np\nprint(np.arange(3).sum())",
    "TypeError: unsupported operand type(s) for +: 'int' and 'str'", "market listing lst_000017 sold 3 units",
    "forum post about onboarding problems", "goal: earn 10000 cents within 8 days", "42", "-3.5", "1e9", "yes", "no",
    "Traceback (most recent call last):", "accessibility tree: button 'Buy' enabled", "unit price", "latency",
    "the quick brown fox jumps over the lazy dog", "Zur Lieferung bitte bis Freitag bestätigen.", "数据库连接失败",
)


class _StopForward(Exception):
    """Raised by the capture hook to abort the forward after the encoder layer."""


@dataclass
class ChunkedHidden:
    """Chunk-level encoder output. h [C, Tc, H]; mask [C, Tc] bool; text_index [C] long (which input text each chunk
    belongs to); chunk_start [C] long (token offset of the chunk inside its text)."""
    h: torch.Tensor
    mask: torch.Tensor
    text_index: torch.Tensor
    chunk_start: torch.Tensor


class QwenSemantic:
    """Implements `interfaces.SemanticEncoder` (sem_dim, version, hidden, embed) plus the decoder-side helpers.

    Not an nn.Module on purpose: the pretrained weights (0.6-4 B parameters) must never end up in a CAM state_dict.
    The trainable LoRA tensors are exposed through `lora_parameters()` / `trainable_state_dict()`; the action
    generator registers them so they are saved with its weights.
    """

    def __init__(self, cfg: Optional[CamConfig] = None, model_name: Optional[str] = None, *, test: bool = False,
                 device: Optional[str] = None, dtype: torch.dtype = torch.bfloat16, layer: float | int = 0.75,
                 lora: bool = True, encoder_lora: Optional[bool] = None, cache_max_bytes: int = 1 << 30,
                 embed_cache_entries: int = 200_000, max_text_tokens: int = 16384, max_batch_tokens: int = 16384,
                 chunk_tokens: Optional[int] = None, decoder_grad_checkpoint: bool = True,
                 local_files_only: bool = True):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = cfg or CamConfig()
        self.cfg = cfg
        self.model_name = model_name or (cfg.sem_test_model if test else cfg.sem_model)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_files_only)
        base = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype, attn_implementation="sdpa",
                                                    local_files_only=local_files_only)
        base.to(self.device).eval()
        for p in base.parameters():
            p.requires_grad_(False)
        self.revision = str(getattr(base.config, "_commit_hash", None) or "local")
        self.sem_dim = int(base.config.hidden_size)
        self.n_layers = int(base.config.num_hidden_layers)
        self.vocab_size = int(base.config.vocab_size)          # logits width (>= len(tokenizer))
        self.n_pretrained = sum(p.numel() for p in base.parameters())
        self.peft = None
        if lora:
            from peft import LoraConfig, get_peft_model
            lcfg = LoraConfig(r=cfg.sem_lora_rank, lora_alpha=cfg.sem_lora_alpha, lora_dropout=0.0, bias="none",
                              target_modules=list(cfg.sem_lora_targets), task_type="CAUSAL_LM")
            self.peft = get_peft_model(base, lcfg)
            self.peft.eval()
            self.lm = self.peft.get_base_model()
        else:
            self.lm = base
        self.encoder_lora = bool(cfg.sem_encoder_lora if encoder_lora is None else encoder_lora) and lora
        if isinstance(layer, float):
            li = int(round(layer * self.n_layers)) - 1
        else:
            li = layer if layer >= 0 else self.n_layers + layer
        self.layer = max(0, min(self.n_layers - 1, li))       # 0-based index into model.layers; state = its output
        self.chunk_tokens = int(chunk_tokens or cfg.sem_chunk_tokens)
        self.max_text_tokens = int(max_text_tokens)
        self.max_batch_tokens = int(max_batch_tokens)
        self.decoder_grad_checkpoint = decoder_grad_checkpoint
        if decoder_grad_checkpoint:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.pad_id = int(self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id)
        self.im_end_id = int(self.tok.convert_tokens_to_ids("<|im_end|>"))
        with torch.no_grad():
            w = self.input_embeddings().weight[: len(self.tok)].float()
            self.embedding_rms = float(w.pow(2).mean().sqrt())          # per-element RMS of real token rows
            self.embedding_row_norm = float(w.norm(dim=1).mean())
        self._updates = 0
        self._cache: OrderedDict = OrderedDict()                     # (sha, version) -> Tensor [T, H]
        self._cache_bytes = 0
        self.cache_max_bytes = int(cache_max_bytes)
        self._ecache: OrderedDict = OrderedDict()                    # (sha, version) -> Tensor [H] float32
        self.embed_cache_entries = int(embed_cache_entries)
        self.stats = {"hits": 0, "misses": 0, "encoded_tokens": 0, "encode_s": 0.0, "forward_calls": 0,
                      "truncated_tokens": 0, "evicted": 0, "load_s": time.time() - t0}

    # ------------------------------------------------------------------ identity / parameters
    @property
    def version(self) -> str:
        """Cache key component: changes whenever anything the encoder-path hidden states depend on changes."""
        enc = f"lora{self._updates}" if self.encoder_lora else "base"
        return (f"{self.model_name}@{self.revision}|L{self.layer}|{str(self.dtype).replace('torch.', '')}"
                f"|c{self.chunk_tokens}|m{self.max_text_tokens}|enc:{enc}")

    @property
    def decoder_version(self) -> str:
        return f"{self.model_name}@{self.revision}|dec:lora{self._updates}"

    def mark_updated(self) -> None:
        """Call after every optimiser step that changed the LoRA weights."""
        self._updates += 1
        self._merged = None                                          # merged acting weights are rebuilt on demand
        if self.encoder_lora:                                        # old entries can never be served again
            self._cache.clear()
            self._cache_bytes = 0
            self._ecache.clear()

    def lora_parameters(self) -> list:
        if self.peft is None:
            return []
        return [p for n, p in self.peft.named_parameters() if "lora_" in n]

    def trainable_state_dict(self) -> dict:
        if self.peft is None:
            return {}
        return {n: p.detach().cpu().clone() for n, p in self.peft.named_parameters() if "lora_" in n}

    def load_trainable_state_dict(self, sd: dict) -> None:
        named = dict(self.peft.named_parameters()) if self.peft is not None else {}
        missing = [k for k in named if "lora_" in k and k not in sd]
        if missing:
            raise KeyError(f"LoRA state is missing {len(missing)} tensors, e.g. {missing[:3]}")
        with torch.no_grad():
            for k, v in sd.items():
                named[k].copy_(v.to(named[k].device, named[k].dtype))
        self.mark_updated()

    def parameter_counts(self) -> dict:
        return {"pretrained": int(self.n_pretrained), "lora": int(sum(p.numel() for p in self.lora_parameters())),
                "model": self.model_name}

    def input_embeddings(self) -> nn.Embedding:
        """The (frozen) input embedding matrix [V, H] used for soft-prefix injection through inputs_embeds."""
        return self.lm.get_input_embeddings()

    def embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.input_embeddings()(ids)

    # ------------------------------------------------------------------ encoder path
    def _encoder_ctx(self):
        if self.peft is not None and not self.encoder_lora:
            return self.peft.disable_adapter()
        return contextlib.nullcontext()

    def _token_ids(self, text: str) -> list:
        ids = self.tok(text, add_special_tokens=False)["input_ids"] if text else []
        if not ids:                                               # empty text -> one end-of-text token
            ids = [self.pad_id]
        if len(ids) > self.max_text_tokens:
            self.stats["truncated_tokens"] += len(ids) - self.max_text_tokens
            ids = ids[: self.max_text_tokens]
        return ids

    @torch.no_grad()
    def _forward_layer(self, ids_rows: list) -> list:
        """Run right-padded rows through the model up to self.layer; returns [len_i, H] states per row."""
        L = max(len(r) for r in ids_rows)
        B = len(ids_rows)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        att = torch.zeros(B, L, dtype=torch.long)
        for i, r in enumerate(ids_rows):
            ids[i, : len(r)] = torch.tensor(r)
            att[i, : len(r)] = 1
        ids, att = ids.to(self.device), att.to(self.device)
        store = {}

        def hook(_m, _inp, out):
            store["h"] = out[0] if isinstance(out, tuple) else out
            raise _StopForward()

        handle = self.lm.model.layers[self.layer].register_forward_hook(hook)
        try:
            with self._encoder_ctx():
                self.lm.model(input_ids=ids, attention_mask=att, use_cache=False)
        except _StopForward:
            pass
        finally:
            handle.remove()
        self.stats["forward_calls"] += 1
        h = store["h"]
        return [h[i, : len(r)] for i, r in enumerate(ids_rows)]

    def _encode_texts(self, texts: list) -> list:
        """Encode (uncached) texts -> list of [T_i, H] tensors (chunk states concatenated)."""
        chunks = []                                   # (text idx, start, ids)
        for ti, t in enumerate(texts):
            ids = self._token_ids(t)
            for s in range(0, len(ids), self.chunk_tokens):
                chunks.append((ti, s, ids[s: s + self.chunk_tokens]))
        order = sorted(range(len(chunks)), key=lambda i: -len(chunks[i][2]))
        outs = [None] * len(chunks)
        i = 0
        while i < len(order):
            L = len(chunks[order[i]][2])
            n = max(1, self.max_batch_tokens // max(L, 1))
            grp = order[i: i + n]
            res = self._forward_layer([chunks[j][2] for j in grp])
            for j, h in zip(grp, res):
                outs[j] = h
            i += n
        per_text: list = [[] for _ in texts]
        for (ti, s, ids), h in zip(chunks, outs):
            per_text[ti].append((s, h))
        self.stats["encoded_tokens"] += sum(len(c[2]) for c in chunks)
        return [torch.cat([h for _, h in sorted(parts, key=lambda x: x[0])], 0) for parts in per_text]

    def _cache_put(self, key, val: torch.Tensor) -> None:
        nbytes = val.numel() * val.element_size()
        if nbytes > self.cache_max_bytes:
            return
        self._cache[key] = val
        self._cache_bytes += nbytes
        while self._cache_bytes > self.cache_max_bytes and self._cache:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.numel() * old.element_size()
            self.stats["evicted"] += 1

    def token_states(self, texts: Sequence[str]) -> list:
        """Cached token-level states of each text: list of [T_i, H] tensors (model dtype, on self.device)."""
        t0 = time.time()
        ver = self.version
        keys = [(_sha(t), ver) for t in texts]
        out: list = [None] * len(texts)
        todo: dict = {}
        for i, k in enumerate(keys):
            v = self._cache.get(k)
            if v is not None:
                self._cache.move_to_end(k)
                self.stats["hits"] += 1
                out[i] = v
            else:
                todo.setdefault(k, []).append(i)
        if todo:
            uniq = list(todo)
            self.stats["misses"] += len(uniq)
            enc = self._encode_texts([texts[todo[k][0]] for k in uniq])
            for k, h in zip(uniq, enc):
                self._cache_put(k, h)
                for i in todo[k]:
                    out[i] = h
        self.stats["encode_s"] += time.time() - t0
        return out

    def hidden(self, texts: Sequence[str], max_tokens: Optional[int] = None) -> tuple:
        """SemanticEncoder.hidden: (h [N, T, H], mask [N, T] bool). Row i is the concatenation of text i's chunk
        states, cut to `max_tokens` (the cut is counted in stats['truncated_tokens']); padding positions are zero."""
        states = self.token_states(list(texts))
        if max_tokens is not None:
            cut = []
            for s in states:
                if s.shape[0] > max_tokens:
                    self.stats["truncated_tokens"] += int(s.shape[0] - max_tokens)
                cut.append(s[:max_tokens])
            states = cut
        T = max(s.shape[0] for s in states)
        h = states[0].new_zeros(len(states), T, self.sem_dim)
        mask = torch.zeros(len(states), T, dtype=torch.bool, device=h.device)
        for i, s in enumerate(states):
            h[i, : s.shape[0]] = s
            mask[i, : s.shape[0]] = True
        return h, mask

    def hidden_chunks(self, texts: Sequence[str]) -> ChunkedHidden:
        """Chunk-level view with the explicit chunk -> text mapping (chunks of `chunk_tokens` tokens)."""
        states = self.token_states(list(texts))
        rows, tix, starts = [], [], []
        for i, s in enumerate(states):
            for st in range(0, s.shape[0], self.chunk_tokens):
                rows.append(s[st: st + self.chunk_tokens])
                tix.append(i)
                starts.append(st)
        Tc = max(r.shape[0] for r in rows)
        h = rows[0].new_zeros(len(rows), Tc, self.sem_dim)
        mask = torch.zeros(len(rows), Tc, dtype=torch.bool, device=h.device)
        for c, r in enumerate(rows):
            h[c, : r.shape[0]] = r
            mask[c, : r.shape[0]] = True
        return ChunkedHidden(h, mask, torch.tensor(tix, device=h.device), torch.tensor(starts, device=h.device))

    def calibration(self) -> dict:
        """Per-dimension statistics of the encoder-layer states over CALIBRATION_TEXTS (cached per version):
        tok_mu/tok_sd over all tokens, emb_mu/emb_sd over the mean-pooled texts. [H] float32 each."""
        ver = self.version
        cal = getattr(self, "_cal", None)
        if cal is None or cal["version"] != ver:
            states = self.token_states(list(CALIBRATION_TEXTS))
            flat = torch.cat([s.float() for s in states], 0)
            pooled = torch.stack([s.float().mean(0) for s in states], 0)
            cal = {"version": ver, "tok_mu": flat.mean(0), "tok_sd": flat.std(0).clamp_min(1e-3),
                   "emb_mu": pooled.mean(0), "emb_sd": pooled.std(0).clamp_min(1e-3)}
            self._cal = cal
        return cal

    def standardize_states(self, h: torch.Tensor) -> torch.Tensor:
        """(h - tok_mu) / tok_sd in float32 (removes the shared outlier dimensions of the encoder layer)."""
        c = self.calibration()
        return (h.float() - c["tok_mu"]) / c["tok_sd"]

    def embed_raw(self, texts: Sequence[str]) -> torch.Tensor:
        """[N, H] float32 masked-mean of the encoder-layer states over all chunks (cached), not standardised."""
        return self._embed_raw(texts)

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """SemanticEncoder.embed: [N, H] float32 pooled document/code embedding = standardised mean of the encoder
        layer states over all chunks ((mean - emb_mu) / emb_sd with the fixed calibration statistics)."""
        c = self.calibration()
        return (self._embed_raw(texts) - c["emb_mu"]) / c["emb_sd"]

    def _embed_raw(self, texts: Sequence[str]) -> torch.Tensor:
        ver = self.version
        keys = [(_sha(t), ver) for t in texts]
        out: list = [None] * len(texts)
        miss = []
        for i, k in enumerate(keys):
            v = self._ecache.get(k)
            if v is not None:
                self._ecache.move_to_end(k)
                out[i] = v
            else:
                miss.append(i)
        if miss:
            states = self.token_states([texts[i] for i in miss])
            for i, s in zip(miss, states):
                v = s.float().mean(0)
                out[i] = v
                self._ecache[keys[i]] = v
                while len(self._ecache) > self.embed_cache_entries:
                    self._ecache.popitem(last=False)
        return torch.stack(out, 0)

    def cache_info(self) -> dict:
        return {"entries": len(self._cache), "bytes": self._cache_bytes, "embed_entries": len(self._ecache),
                **{k: self.stats[k] for k in ("hits", "misses", "evicted")}}

    # ------------------------------------------------------------------ decoder path (LoRA enabled)
    def decoder_hidden(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
                       position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Final (post-norm) hidden states [B, L, H] of the LoRA-enabled decoder. Differentiable w.r.t. the inputs and
        the LoRA weights when grad is enabled (gradient checkpointing is used for the backward if configured)."""
        use_ckpt = self.decoder_grad_checkpoint and torch.is_grad_enabled()
        if use_ckpt:
            self.lm.train()                     # HF checkpoints only in training mode; Qwen3 has no dropout
        try:
            out = self.lm.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids,
                                use_cache=False)
        finally:
            if use_ckpt:
                self.lm.eval()
        return out.last_hidden_state

    def lm_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Vocabulary logits (float32) for final hidden states h [..., H] -> [..., V]."""
        return self.lm.lm_head(h).float()

    @torch.no_grad()
    def step(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor, cache):
        """One incremental decoder call with a KV cache (sampling path). Returns (last-position logits [B, V] float32,
        cache)."""
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids,
                      cache=cache, use_cache=True, logits_to_keep=1)
        return out.logits[:, -1].float(), out.past_key_values

    def new_cache(self):
        from transformers import DynamicCache
        return DynamicCache()

    # ------------------------------------------------------------------ acting with merged LoRA (contract v3 #12)
    def _lora_layers(self) -> list:
        """(module, A, B, scale) of every LoRA-wrapped linear layer of the decoder (default adapter)."""
        out = []
        if self.peft is None:
            return out
        for m in self.peft.modules():
            la, lb = getattr(m, "lora_A", None), getattr(m, "lora_B", None)
            if isinstance(la, nn.ModuleDict) and "default" in la and isinstance(lb, nn.ModuleDict):
                out.append((m, la["default"].weight, lb["default"].weight, float(m.scaling["default"])))
        return out

    def _lora_fingerprint(self) -> tuple:
        """Changes whenever any LoRA tensor is modified (in-place optimiser steps bump the tensor version counter;
        loads replace storage), without a device synchronisation."""
        return tuple((p.data_ptr(), p._version) for _, a, b, _ in self._lora_layers() for p in (a, b))

    def merged_weights(self) -> list:
        """[(base linear layer, merged weight W + scale * B @ A in the model dtype)] for the layers whose LoRA delta is
        non-zero (cached until a LoRA tensor changes; float32 product, one rounding to the model dtype)."""
        fp = self._lora_fingerprint()
        cache = getattr(self, "_merged", None)
        if cache is not None and cache[0] == fp:
            return cache[1]
        out = []
        with torch.no_grad():
            for m, A, Bw, scale in self._lora_layers():
                if not bool(Bw.any()):
                    continue                       # zero delta (e.g. at init): the base weight is the merged weight
                base = m.get_base_layer()
                W = base.weight
                out.append((base, (W.float() + scale * (Bw.float() @ A.float())).to(W.dtype)))
        self._merged = (fp, out)
        self.stats["merged_builds"] = self.stats.get("merged_builds", 0) + 1
        return out

    @contextlib.contextmanager
    def acting(self, merged: bool = True):
        """Acting-path decoder: the LoRA adapter disabled and each adapted linear layer's weight replaced by its merged
        weight W + scale * B A for the duration (fewer kernels per decode step). The logits equal the LoRA path's up to
        the rounding of the merged weight (bit-identical while every LoRA delta is zero); callers that report a
        behaviour log-probability recompute it on the learner path (ActionGenerator.sample rescoring). The frozen base
        weights are restored on exit, so the encoder path and its hidden-state cache are unaffected."""
        if not merged or self.peft is None:
            yield
            return
        swaps = self.merged_weights()
        saved = []
        try:
            with torch.no_grad():
                for base, Wm in swaps:
                    saved.append((base.weight, base.weight.data))
                    base.weight.data = Wm
            with self.peft.disable_adapter():
                yield
        finally:
            with torch.no_grad():
                for p, data in saved:
                    p.data = data
