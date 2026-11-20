"""Frozen pretrained LLM backbone for CAM 2.0.

The LLM's weights are never updated: every parameter has ``requires_grad=False``, the module is in eval mode,
and ``fingerprint()`` hashes every weight so a run can prove bit-identity at the end. There is no LoRA or any
other adaptation of the LLM's weights. Two conditioning mechanisms connect it to CAM:

A. **Semantic features** (``encode_pooled``, ``encode_tokens``): mask-mean pooled hidden states of chosen layers
   (forward hooks, no ``output_hidden_states``), and per-token hidden states for goals. No gradient; cached by
   exact text (the text contains everything the representation can depend on, so cached vectors are never stale).

B. **CAM-controlled generation** (``sample_with_prefix`` / ``logprob_with_prefix``): CAM produces ``P`` continuous
   control vectors in the LLM's input-embedding space. They are spliced into the input sequence between a context
   and an answer stub (``[ctx tokens][P CAM vectors][stub tokens] -> generated tokens``). Generation samples from
   the frozen LLM's temperature-scaled softmax. ``logprob_with_prefix`` recomputes the log-probability of the
   sampled tokens with autograd enabled: gradients flow *through* the frozen LLM's computation into the CAM
   vectors (and from there into CAM's parameters) while the LLM's own parameters receive none. This makes the
   text part of an action an ordinary stochastic policy output: ``grad log p(tokens | CAM vectors, ctx)``.
"""
from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_MODEL = os.environ.get("CAM2_LLM", "Qwen/Qwen3-1.7B")


@dataclass
class GenOut:
    tokens: list[list[int]]       # sampled token ids per row (including the terminating token if one was produced)
    logp: torch.Tensor            # (B,) summed log-prob of ``tokens`` under the temperature-scaled distribution
    texts: list[str]              # decoded answer text (terminator stripped)
    stopped: list[bool]           # True if a stop token ended the row (False = hit max_new)


def _snapshot_revision(name: str) -> Optional[str]:
    """The Hugging Face snapshot hash the weights were loaded from (recorded with every experiment)."""
    try:
        from transformers.utils import cached_file

        parts = Path(cached_file(name, "config.json", local_files_only=True)).parts
        return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
    except Exception:  # noqa: BLE001
        return None


class FrozenLLM:
    """A frozen causal LM. ``device`` defaults to CUDA when available; weights in ``dtype`` (bf16 on GPU)."""

    def __init__(self, name: str = DEFAULT_MODEL, device: Optional[str] = None, dtype: Optional[torch.dtype] = None,
                 layers: Sequence[float] = (0.5, 1.0), max_len: int = 256, batch: int = 128,
                 cache_size: int = 400_000, goal_tokens: int = 40):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self.hidden = cfg.hidden_size
        self.n_layers = cfg.num_hidden_layers
        self.layer_ids = sorted({min(self.n_layers, max(1, int(round(f * self.n_layers)))) for f in layers})
        self.goal_layer = self.layer_ids[0]
        self._captured: dict = {}
        dec = self.model.get_decoder()
        for li in self.layer_ids:
            (dec.norm if li == self.n_layers else dec.layers[li - 1]).register_forward_hook(self._hook(li))
        self._capture_on = False
        self.pooled_dim = self.hidden * len(self.layer_ids)
        self.max_len, self.batch, self.goal_tokens = max_len, batch, goal_tokens
        self.cache: OrderedDict = OrderedDict()
        self.tcache: OrderedDict = OrderedDict()
        self.cache_size = cache_size
        self.stats = {"pooled_texts": 0, "pooled_hits": 0, "encode_s": 0.0, "encode_tokens": 0,
                      "gen_calls": 0, "gen_rows": 0, "gen_tokens": 0, "gen_s": 0.0,
                      "lp_rows": 0, "lp_tokens": 0, "lp_s": 0.0}
        self.revision = getattr(cfg, "_commit_hash", None) or _snapshot_revision(name)
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.eos_ids = {i for i in [self.im_end, self.tok.eos_token_id] if i is not None}
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.im_end

    # ------------------------------------------------------------------ bookkeeping
    def _hook(self, li):
        def fn(_mod, _inp, out):
            if self._capture_on:
                self._captured[li] = out[0] if isinstance(out, tuple) else out
        return fn

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for name, p in sorted(self.model.state_dict().items()):
            h.update(name.encode())
            h.update(p.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def describe(self) -> dict:
        return {"model": self.name, "revision": self.revision, "params": self.n_params(),
                "trainable_params": self.trainable_params(), "hidden": self.hidden, "layers": self.n_layers,
                "feature_layers": self.layer_ids, "dtype": str(self.dtype), "device": self.device, "frozen": True}

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids)

    def token_embedding_rows(self, text: str) -> torch.Tensor:
        """Float32 copy of the input embeddings of ``text`` (used to initialise CAM's own control vectors)."""
        ids = self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        return self.embed(ids)[0].float().detach().cpu()

    # ------------------------------------------------------------------ mechanism A: features
    @torch.no_grad()
    def _forward_capture(self, texts: list[str], max_len: int):
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(self.device)
        self._captured.clear()
        self._capture_on = True
        try:
            self.model.get_decoder()(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        finally:
            self._capture_on = False
        cap = dict(self._captured)
        self._captured.clear()
        return enc, cap

    @torch.no_grad()
    def encode_pooled(self, texts: Sequence[str]) -> np.ndarray:
        """(n, pooled_dim) float16 — concatenated mask-mean pooled hidden states of ``layer_ids``."""
        out = np.zeros((len(texts), self.pooled_dim), dtype=np.float16)
        todo = []
        for i, t in enumerate(texts):
            v = self.cache.get(t)
            if v is None:
                todo.append(i)
            else:
                self.cache.move_to_end(t)
                out[i] = v
                self.stats["pooled_hits"] += 1
        self.stats["pooled_texts"] += len(texts)
        uniq = list(dict.fromkeys(texts[i] for i in todo))
        order = sorted(range(len(uniq)), key=lambda j: len(uniq[j]))
        got = {}
        t0 = time.time()
        for s in range(0, len(order), self.batch):
            chunk = [uniq[j] for j in order[s:s + self.batch]]
            enc, cap = self._forward_capture(chunk, self.max_len)
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = torch.cat([(cap[li].float() * m).sum(1) / m.sum(1).clamp(min=1) for li in self.layer_ids], -1)
            self.stats["encode_tokens"] += int(enc["attention_mask"].sum())
            for t, row in zip(chunk, v.to(torch.float16).cpu().numpy()):
                got[t] = row
        self.stats["encode_s"] += time.time() - t0
        for i in todo:
            v = got[texts[i]]
            out[i] = v
            self.cache[texts[i]] = v
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return out

    @torch.no_grad()
    def encode_pooled_ctx(self, contexts: Sequence[str], texts: Sequence[str]) -> np.ndarray:
        """(n, pooled_dim) float16: each ``text`` encoded *after* its ``context`` (e.g. the goal), mean-pooled over the
        text's own token positions only. The frozen LLM's attention from the text to the context makes relevance to
        the goal available in the features. Cached by (context, text)."""
        out = np.zeros((len(texts), self.pooled_dim), dtype=np.float16)
        keys = [("\x00ctx\x00" + c, t) for c, t in zip(contexts, texts)]
        todo = []
        for i, k in enumerate(keys):
            v = self.cache.get(k)
            if v is None:
                todo.append(i)
            else:
                self.cache.move_to_end(k)
                out[i] = v
                self.stats["pooled_hits"] += 1
        self.stats["pooled_texts"] += len(texts)
        uniq = list(dict.fromkeys(keys[i] for i in todo))
        got = {}
        t0 = time.time()
        for s in range(0, len(uniq), self.batch):
            chunk = uniq[s:s + self.batch]
            rows, spans = [], []
            for c, t in chunk:
                ci = self.tok(c[5:] + "\nElement:", add_special_tokens=False)["input_ids"][-(self.max_len // 2):]
                ti = self.tok(" " + t, add_special_tokens=False)["input_ids"][: self.max_len // 2] or [self.pad_id]
                rows.append(ci + ti)
                spans.append((len(ci), len(ci) + len(ti)))
            L = max(len(r) for r in rows)
            ids = torch.full((len(rows), L), self.pad_id, dtype=torch.long)
            att = torch.zeros((len(rows), L), dtype=torch.long)
            m = torch.zeros((len(rows), L, 1))
            for j, (r, (a, b)) in enumerate(zip(rows, spans)):
                ids[j, : len(r)] = torch.tensor(r)
                att[j, : len(r)] = 1
                m[j, a:b] = 1.0
            ids, att, m = ids.to(self.device), att.to(self.device), m.to(self.device)
            self._captured.clear()
            self._capture_on = True
            try:
                self.model.get_decoder()(input_ids=ids, attention_mask=att)
            finally:
                self._capture_on = False
            v = torch.cat([(self._captured[li].float() * m).sum(1) / m.sum(1).clamp(min=1) for li in self.layer_ids], -1)
            self._captured.clear()
            self.stats["encode_tokens"] += int(att.sum())
            for k, row in zip(chunk, v.to(torch.float16).cpu().numpy()):
                got[k] = row
        self.stats["encode_s"] += time.time() - t0
        for i in todo:
            out[i] = got[keys[i]]
            self.cache[keys[i]] = out[i]
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return out

    @torch.no_grad()
    def encode_tokens(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Per-token hidden states of layer ``goal_layer`` (first ``goal_tokens`` tokens), float16 (T_i, hidden)."""
        res: list[Optional[np.ndarray]] = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            v = self.tcache.get(t)
            if v is None:
                todo.append(i)
            else:
                res[i] = v
        uniq = list(dict.fromkeys(texts[i] for i in todo))
        got = {}
        t0 = time.time()
        for s in range(0, len(uniq), self.batch):
            chunk = uniq[s:s + self.batch]
            enc, cap = self._forward_capture(chunk, self.goal_tokens)
            hs = cap[self.goal_layer].float()
            for j, t in enumerate(chunk):
                n = int(enc["attention_mask"][j].sum())
                got[t] = hs[j, :n].to(torch.float16).cpu().numpy()
        self.stats["encode_s"] += time.time() - t0
        for i in todo:
            res[i] = got[texts[i]]
            self.tcache[texts[i]] = res[i]
            if len(self.tcache) > self.cache_size // 4:
                self.tcache.popitem(last=False)
        return res  # type: ignore[return-value]

    # ------------------------------------------------------------------ mechanism B: CAM-controlled generation
    def chat_ids(self, user: str, max_ctx: int = 768) -> tuple[list[int], list[int]]:
        """Token ids of a Qwen3 chat turn with thinking disabled, split around the point where CAM's control
        vectors are inserted: (context ids ending with the assistant header, answer-stub ids)."""
        ctx = self.tok(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        if len(ctx) > max_ctx:  # keep the head (instructions) and the tail (assistant header)
            ctx = ctx[: max_ctx // 2] + ctx[-(max_ctx - max_ctx // 2):]
        stub = self.tok("<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
        return ctx, stub

    def _assemble(self, prefix: Optional[torch.Tensor], ctx: list[list[int]], stub: list[list[int]],
                  tgt: Optional[list[list[int]]] = None):
        """Left-padded inputs_embeds for rows [ctx][prefix][stub][tgt]. Returns embeds, attention mask, position
        ids, and for each row the index range of tgt positions."""
        B = len(ctx)
        P = 0 if prefix is None else prefix.shape[1]
        tgt = tgt or [[] for _ in range(B)]
        lens = [len(c) + P + len(s) + len(t) for c, s, t in zip(ctx, stub, tgt)]
        L = max(lens)
        E = self.hidden
        dev = self.device
        all_ids = torch.full((B, L), self.pad_id, dtype=torch.long, device=dev)
        is_pre = torch.zeros(B, L, dtype=torch.bool, device=dev)
        att = torch.zeros(B, L, dtype=torch.long, device=dev)
        for i in range(B):
            off = L - lens[i]
            row = ctx[i] + [self.pad_id] * P + stub[i] + tgt[i]
            all_ids[i, off:] = torch.tensor(row, device=dev)
            att[i, off:] = 1
            if P:
                is_pre[i, off + len(ctx[i]): off + len(ctx[i]) + P] = True
        emb = self.embed(all_ids)
        if P:
            pre = prefix.to(emb.dtype)
            # masked_scatter fills masked positions in row-major order; each row's prefix slots are contiguous and
            # rows are in order, so the (B*P, E) source lands exactly in place (differentiable w.r.t. the source)
            emb = emb.masked_scatter(is_pre.unsqueeze(-1).expand(B, L, E), pre.reshape(-1, E))
        pos = (att.cumsum(1) - 1).clamp(min=0)
        tgt_start = [L - len(t) for t in tgt]
        return emb, att, pos, tgt_start

    @torch.no_grad()
    def sample_with_prefix(self, prefix: Optional[torch.Tensor], ctx: list[list[int]], stub: list[list[int]],
                           max_new: int = 16, temperature: float = 1.0, greedy: bool = False,
                           gen: Optional[torch.Generator] = None, stop_ids: Optional[set] = None) -> GenOut:
        """Sample answers from the frozen LLM conditioned on CAM's control vectors (no gradient)."""
        t0 = time.time()
        from transformers import DynamicCache

        stop = set(stop_ids or self.eos_ids)
        B = len(ctx)
        emb, att, pos, _ = self._assemble(prefix, ctx, stub)
        cache = DynamicCache()
        out = self.model(inputs_embeds=emb, attention_mask=att, position_ids=pos, past_key_values=cache, use_cache=True,
                         logits_to_keep=1)
        cache = out.past_key_values
        logits = out.logits[:, -1].float()
        toks: list[list[int]] = [[] for _ in range(B)]
        logp = torch.zeros(B, device=self.device)
        alive = torch.ones(B, dtype=torch.bool, device=self.device)
        stopped = [False] * B
        cur_pos = pos[:, -1]
        for _ in range(max_new):
            lp = torch.log_softmax(logits / max(temperature, 1e-4), -1)
            if greedy:
                nxt = lp.argmax(-1)
            else:
                nxt = torch.multinomial(lp.exp(), 1, generator=gen).squeeze(-1)
            chosen = lp.gather(1, nxt.unsqueeze(1)).squeeze(1)
            logp = logp + torch.where(alive, chosen, torch.zeros_like(chosen))
            nl = nxt.tolist()
            for i in range(B):
                if alive[i]:
                    toks[i].append(nl[i])
                    if nl[i] in stop:
                        alive[i] = False
                        stopped[i] = True
            if not bool(alive.any()):
                break
            att = torch.cat([att, torch.ones_like(att[:, :1])], 1)
            cur_pos = cur_pos + 1
            out = self.model(inputs_embeds=self.embed(nxt.unsqueeze(1)), attention_mask=att,
                             position_ids=cur_pos.unsqueeze(1), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1].float()
        texts = [self.tok.decode([t for t in tk if t not in stop], skip_special_tokens=True).strip() for tk in toks]
        self.stats["gen_calls"] += 1
        self.stats["gen_rows"] += B
        self.stats["gen_tokens"] += sum(len(t) for t in toks)
        self.stats["gen_s"] += time.time() - t0
        return GenOut(tokens=toks, logp=logp, texts=texts, stopped=stopped)

    def logprob_with_prefix(self, prefix: Optional[torch.Tensor], ctx: list[list[int]], stub: list[list[int]],
                            tgt: list[list[int]], temperature: float = 1.0, chunk: int = 16,
                            exact: bool = True) -> torch.Tensor:
        """(B,) summed log-prob of ``tgt`` tokens. Differentiable w.r.t. ``prefix`` (the LLM's parameters have
        requires_grad=False, so they receive no gradient; autograd still propagates through their computation).

        ``exact=True`` evaluates every row on its own, unpadded: in bf16 the result of a padded batched forward
        depends on the batch's padding length (kernel tiling and rounding), which made the behaviour log-probability
        stored at collection differ from the learner's recomputation by up to ~0.06 nats. Row-wise evaluation is
        batch-invariant, so PPO's importance ratio is exactly 1 at the behaviour parameters."""
        if exact and len(ctx) > 1:
            return torch.cat([self.logprob_with_prefix(None if prefix is None else prefix[i:i + 1], ctx[i:i + 1],
                                                       stub[i:i + 1], tgt[i:i + 1], temperature, 1, False)
                              for i in range(len(ctx))])
        t0 = time.time()
        outs = []
        for s in range(0, len(ctx), chunk):
            sl = slice(s, s + chunk)
            pre = None if prefix is None else prefix[sl]
            emb, att, pos, tstart = self._assemble(pre, ctx[sl], stub[sl], tgt[sl])
            h = self.model.get_decoder()(inputs_embeds=emb, attention_mask=att, position_ids=pos).last_hidden_state
            head = self.model.get_output_embeddings()
            L = emb.shape[1]
            rows = []
            for i, t in enumerate(tgt[sl]):
                if not t:
                    rows.append(h.new_zeros((), dtype=torch.float32))
                    continue
                p = torch.arange(tstart[i] - 1, L - 1, device=self.device)
                lg = head(h[i, p]).float() / max(temperature, 1e-4)
                lp = torch.log_softmax(lg, -1)
                rows.append(lp.gather(1, torch.tensor(t, device=self.device).unsqueeze(1)).sum())
                self.stats["lp_tokens"] += len(t)
            outs.append(torch.stack(rows))
        self.stats["lp_rows"] += len(ctx)
        self.stats["lp_s"] += time.time() - t0
        return torch.cat(outs) if outs else torch.zeros(0, device=self.device)

    # ------------------------------------------------------------------ plain generation (baseline B1)
    @torch.no_grad()
    def generate_text(self, prompts: list[str], max_new: int = 48, max_ctx: int = 3000) -> list[str]:
        ctx, stub = [], []
        for p in prompts:
            c, s = self.chat_ids(p, max_ctx=max_ctx)
            ctx.append(c)
            stub.append(s)
        return self.sample_with_prefix(None, ctx, stub, max_new=max_new, greedy=True).texts


class TinyRandomLLM(FrozenLLM):
    """A randomly initialised Qwen3-architecture model with the real Qwen3 tokenizer, for fast plumbing tests only.
    Same code paths as ``FrozenLLM``; it has no pretrained knowledge and must never be used for experiments."""

    def __init__(self, tokenizer_name: str = "Qwen/Qwen3-0.6B", hidden: int = 64, layers: int = 2, device: str = "cpu",
                 seed: int = 0, **kw):
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(seed)
        self.name = f"tiny-random-qwen3-{hidden}x{layers}"
        self.device = device
        self.dtype = torch.float32
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tok.padding_side = "right"
        cfg = Qwen3Config(vocab_size=len(self.tok), hidden_size=hidden, intermediate_size=hidden * 2,
                          num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                          max_position_embeddings=4096, tie_word_embeddings=True)
        self.model = Qwen3ForCausalLM(cfg).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden, self.n_layers = hidden, layers
        self.layer_ids = [1, layers] if layers > 1 else [1]
        self.goal_layer = self.layer_ids[0]
        self._captured, self._capture_on = {}, False
        dec = self.model.get_decoder()
        for li in self.layer_ids:
            (dec.norm if li == self.n_layers else dec.layers[li - 1]).register_forward_hook(self._hook(li))
        self.pooled_dim = hidden * len(self.layer_ids)
        self.max_len, self.batch, self.goal_tokens = kw.get("max_len", 128), 64, kw.get("goal_tokens", 24)
        self.cache, self.tcache, self.cache_size = OrderedDict(), OrderedDict(), 100_000
        self.stats = {k: 0 for k in ["pooled_texts", "pooled_hits", "encode_tokens", "gen_calls", "gen_rows",
                                     "gen_tokens", "lp_rows", "lp_tokens"]}
        self.stats.update({"encode_s": 0.0, "gen_s": 0.0, "lp_s": 0.0})
        self.revision = None
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.eos_ids = {i for i in [self.im_end, self.tok.eos_token_id] if i is not None}
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.im_end


def load_llm(name: str, device: Optional[str] = None, **kw) -> FrozenLLM:
    if name.startswith("tiny-random"):
        return TinyRandomLLM(device=device or "cpu", **kw)
    return FrozenLLM(name, device=device, **kw)
