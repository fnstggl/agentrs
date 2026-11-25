"""Shared small-model inference service (MASTER_SPEC 13.4, INTERFACE_CONTRACTS section 4).

One set of model weights per ``(model_id, revision, device, dtype, quantization)`` is loaded per process and shared by
every `SemanticService` object (and so by every semantic participant) that names it. Participant-specific state never
lives here: the service is stateless apart from caches of shared prompt prefixes and counters.

What it does

* **Batched generation.** `generate(ctx_or_none, requests)` takes many requests ``{messages, max_new_tokens,
  temperature, seed}`` at once, groups compatible ones (same shared prefix, greedy vs sampled), sorts each group by
  length and decodes it in batches of at most ``max_batch`` rows with left padding of the per-request part. Rows that
  finish (end of turn or their own token bound) leave the batch, so short answers do not pay for long ones.
* **Bounded generation.** Every request is capped at ``max_new_tokens`` (its own value, never more than the
  service's), prompts at ``max_context`` tokens (head and tail are kept and the request is flagged ``truncated``).
* **Prefix (KV) caching.** A request may say that its first ``prefix_messages`` messages are shared (for example the
  common system prompt of all participants). The key/value cache of that prefix is computed once, kept in a small
  LRU on the shared weights, and expanded to the batch instead of recomputing the prefix for every row. Groups of
  requests without a hint share their longest common token prefix when it is long enough. `choose` always computes
  a prompt once and scores all its options against that one cache. `measure_prefix_cache` measures the effect.
* **Scoring options.** `choose(ctx_or_none, requests)` returns, for each request ``{messages, choices}``, the log
  probability the model assigns to every option as the reply (mean per token by default) and the best one. This is
  how a participant picks among a short list of tools or moves without free-form parsing.
* **Quantization** (``quantization="int8-dynamic"``): dynamic int8 linear layers for CPU execution (PyTorch's
  ``quantize_dynamic``). On GPU the weights stay in bfloat16; there is no int8 GPU kernel in this environment.
* **Versioning.** `info()` gives the model id, the Hugging Face revision (commit of the local snapshot), the parameter
  count and `weights_hash`: SHA-256 over the sorted parameter tensors (name, dtype, shape, bytes), as defined for
  `crucible.substrate.inference.weights_hash`.
* **Recording.** With a kernel `Ctx` the outputs are recorded as an external outcome of the transaction
  (`crucible.substrate.inference.recorded` when the substrate provides it, else ``ctx.external`` with the same record
  shape ``{model_id, model_hash, inputs_digest, output}``), so replay and regeneration of the world return the
  recorded text and never run the model. Without a ctx (benchmarks, offline evaluation) nothing is recorded.

Models are loaded from the local Hugging Face cache only (``local_files_only=True``); the service never downloads and
makes no network calls.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
REQUEST_KEYS = ("messages", "max_new_tokens", "temperature", "seed", "prefix_messages", "thinking", "assistant_prefix")

_LOCK = threading.RLock()
_LOADED: dict[tuple, "_Weights"] = {}


# ====================================================================== hashing and recording helpers
def _canon(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def request_digest(req: dict) -> str:
    """SHA-256 of the parts of a request that determine its output."""
    keep = {k: req.get(k) for k in REQUEST_KEYS + ("choices", "normalize") if req.get(k) is not None}
    return hashlib.sha256(_canon(keep)).hexdigest()


def weights_hash(module_or_state_dict: Any) -> str:
    """SHA-256 over the sorted parameter tensors (name, dtype, shape, raw bytes): the definition of contract 1.7."""
    try:
        from ..substrate.inference import weights_hash as _wh   # the substrate's implementation once it is merged
        return _wh(module_or_state_dict)
    except ImportError:
        pass
    import torch
    sd = module_or_state_dict.state_dict() if hasattr(module_or_state_dict, "state_dict") else module_or_state_dict
    h = hashlib.sha256()
    for name in sorted(sd):
        t = sd[name]
        if not isinstance(t, torch.Tensor):
            continue
        t = t.detach()
        if t.is_quantized:
            t = t.int_repr()
        t = t.contiguous().cpu()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(repr(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def recorded(ctx, kind: str, model_id: str, model_hash: str, inputs: Any, fn: Callable[[], Any]) -> Any:
    """Contract 1.7 `crucible.substrate.inference.recorded` (used when the substrate is present). The fallback keeps
    the same record ``{model_id, model_hash, inputs_digest, output}`` in ``ctx.external``: during replay and
    regeneration the recorded output is returned and `fn` is not run."""
    try:
        from ..substrate.inference import recorded as _rec
        return _rec(ctx, kind, model_id, model_hash, inputs, fn)
    except ImportError:
        pass
    dig = hashlib.sha256(_canon(inputs)).hexdigest()
    rec = ctx.external(kind, lambda: {"model_id": model_id, "model_hash": model_hash, "inputs_digest": dig,
                                      "output": fn()})
    if rec.get("inputs_digest") != dig or rec.get("model_hash") != model_hash:
        from ..core.errors import ProtocolError
        raise ProtocolError(f"recorded {kind} outcome does not match its inputs or model (replay diverged)")
    return rec["output"]


def local_revision(model_id: str) -> str:
    """Commit hash of the locally cached snapshot of `model_id` (no network)."""
    from huggingface_hub import constants
    base = os.path.join(constants.HF_HUB_CACHE, "models--" + model_id.replace("/", "--"))
    try:
        return open(os.path.join(base, "refs", "main")).read().strip()
    except OSError:
        snaps = sorted(os.listdir(os.path.join(base, "snapshots"))) if os.path.isdir(os.path.join(base, "snapshots")) else []
        if snaps:
            return snaps[-1]
    raise FileNotFoundError(f"{model_id} is not in the local Hugging Face cache ({base})")


def loaded_copies(model_id: str | None = None) -> int:
    """Number of weight copies loaded in this process (for one model id, or all)."""
    with _LOCK:
        return sum(1 for k in _LOADED if model_id is None or k[0] == model_id)


def _resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available (run through the slot scheduler with --gpu-mb)")
    return device


# ====================================================================== shared weights
@dataclass
class _Prefix:
    ids: tuple
    cache: Any            # DynamicCache for batch size 1
    last_logits: Any      # logits after the prefix (for an empty suffix)
    hits: int = 0


@dataclass
class _Weights:
    key: tuple
    model: Any
    tok: Any
    params: int
    eos_ids: list
    pad_id: int
    load_s: float
    hash: str | None = None
    prefixes: "OrderedDict[str, _Prefix]" = field(default_factory=OrderedDict)
    users: int = 0
    lock: Any = field(default_factory=threading.RLock)


def _load(model_id: str, revision: str, device: str, dtype: str, quantization: str | None) -> _Weights:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=tdtype, local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params = sum(p.numel() for p in model.parameters())
    if quantization == "int8-dynamic":
        if device != "cpu":
            raise ValueError("int8-dynamic quantization runs on CPU only")
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    elif quantization not in (None, "none"):
        raise ValueError(f"unknown quantization {quantization!r}")
    model.to(device)
    gc = getattr(model, "generation_config", None)
    eos = getattr(gc, "eos_token_id", None) if gc is not None else None
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos] if eos is not None else []
    if tok.eos_token_id is not None and tok.eos_token_id not in eos:
        eos.append(tok.eos_token_id)
    im_end = tok.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tok.get_vocab() else None
    if im_end is not None and im_end not in eos:
        eos.append(im_end)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    return _Weights((model_id, revision, device, dtype, quantization), model, tok, int(params), eos, int(pad),
                    time.time() - t0)


# ====================================================================== the service
class SemanticService:
    """One shared model per (model_id, revision) serving many participants. See the module docstring."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto", max_batch: int = 16,
                 max_new_tokens: int = 128, *, revision: str | None = None, dtype: str | None = None,
                 quantization: str | None = None, max_context: int = 2048, prefix_cache: bool = True,
                 min_prefix_tokens: int = 32, prefix_cache_entries: int = 8, enable_thinking: bool = False,
                 max_choice_rows: int = 64):
        self.model_id = model_id
        self.device = _resolve_device(device)
        self.revision = revision or local_revision(model_id)
        self.dtype = dtype or ("bfloat16" if self.device.startswith("cuda") else "float32")
        self.quantization = quantization if quantization not in ("none",) else None
        self.max_batch, self.max_new_tokens, self.max_context = int(max_batch), int(max_new_tokens), int(max_context)
        self.prefix_cache, self.min_prefix_tokens = bool(prefix_cache), int(min_prefix_tokens)
        self.prefix_cache_entries, self.enable_thinking = int(prefix_cache_entries), bool(enable_thinking)
        self.max_choice_rows = int(max_choice_rows)
        self.key = (model_id, self.revision, self.device, self.dtype, self.quantization)
        self._w: _Weights | None = None
        self._null_scores: dict[tuple, float] = {}
        self.stats = {"generate_calls": 0, "choose_calls": 0, "requests": 0, "batches": 0, "rows": 0,
                      "prompt_tokens": 0, "gen_tokens": 0, "prefill_s": 0.0, "decode_s": 0.0, "choose_s": 0.0,
                      "prefix_hits": 0, "prefix_misses": 0, "prefix_tokens_reused": 0, "truncated": 0,
                      "padding_tokens": 0, "recorded_calls": 0}
        _register(self)

    # ------------------------------------------------------------------ weights
    @property
    def weights(self) -> _Weights:
        if self._w is None:
            with _LOCK:
                w = _LOADED.get(self.key)
                if w is None:
                    w = _load(*self.key)
                    _LOADED[self.key] = w
                w.users += 1
                self._w = w
        return self._w

    @property
    def model(self):
        return self.weights.model

    @property
    def tokenizer(self):
        return self.weights.tok

    def model_hash(self) -> str:
        w = self.weights
        if w.hash is None:
            w.hash = _cached_hash(self.key, lambda: weights_hash(w.model))
        return w.hash

    def info(self) -> dict:
        w = self.weights
        return {"model_id": self.model_id, "revision": self.revision, "params": w.params,
                "weights_hash": self.model_hash(), "device": self.device, "dtype": self.dtype,
                "quantization": self.quantization, "max_batch": self.max_batch, "max_new_tokens": self.max_new_tokens,
                "max_context": self.max_context, "prefix_cache": self.prefix_cache, "load_s": round(w.load_s, 2)}

    def close(self) -> None:
        """Release this service's reference; the weights are freed when no service uses them any more."""
        if self._w is None:
            return
        with _LOCK:
            self._w.users -= 1
            if self._w.users <= 0:
                _LOADED.pop(self._w.key, None)
                self._w.prefixes.clear()
            self._w = None

    # ------------------------------------------------------------------ public API
    def generate(self, ctx_or_none, requests: list[dict]) -> list[dict]:
        """Generate one reply per request. Request: ``{messages, max_new_tokens?, temperature?, seed?,
        prefix_messages?}``. Reply: ``{text, prompt_tokens, gen_tokens, finish: stop|length, truncated,
        prefix_tokens}``. With a kernel ctx the replies are recorded (replay never runs the model)."""
        reqs = [self._norm(r) for r in requests]
        if ctx_or_none is None:
            return self._generate(reqs)
        self.stats["recorded_calls"] += 1
        return recorded(ctx_or_none, "semantic.generate", self.model_id, self.model_hash(),
                        self.descriptors(reqs), lambda: self._generate(reqs))

    def choose(self, ctx_or_none, requests: list[dict]) -> list[dict]:
        """Score options. Request: ``{messages, choices: [str], normalize: mean|sum, prefix_messages?}``. Reply:
        ``{choice: index, logprobs: [per option], prompt_tokens, option_tokens: [...]}``."""
        reqs = [self._norm(r, choices=True) for r in requests]
        if ctx_or_none is None:
            return self._choose_any(reqs)
        self.stats["recorded_calls"] += 1
        return recorded(ctx_or_none, "semantic.choose", self.model_id, self.model_hash(),
                        self.descriptors(reqs), lambda: self._choose_any(reqs))

    def run(self, kind: str, requests: list[dict]) -> list[dict]:
        """Unrecorded execution of an already normalised batch (used by the recording kernel module)."""
        return self._generate(requests) if kind == "generate" else self._choose_any(requests)

    def _choose_any(self, reqs: list[dict]) -> list[dict]:
        """`_choose`, plus ``normalize="pmi"``: the option's total log probability after the prompt minus its log
        probability after a content-free prompt (the system message and the prompt's last line only), which removes
        the model's prior preference for some option strings (Holtzman et al. 2021, domain-conditional PMI). The
        content-free scores are cached per (prompt, option)."""
        out = self._choose([dict(r, normalize="sum") if r.get("normalize") == "pmi" else r for r in reqs])
        pmi = [i for i, r in enumerate(reqs) if r.get("normalize") == "pmi"]
        if not pmi:
            return out
        cache = self._null_scores
        todo, keys = [], []
        for i in pmi:
            r = reqs[i]
            sysm = [m for m in r["messages"][:1] if m["role"] == "system"]
            last = r["messages"][-1]["content"].strip().rsplit("\n", 1)[-1]
            null = sysm + [{"role": "user", "content": last}]
            k = hashlib.sha256(_canon(null)).hexdigest()
            keys.append(k)
            missing = [c for c in r["choices"] if (k, c) not in cache]
            if missing:
                todo.append({"messages": null, "choices": sorted(set(missing)), "normalize": "sum", "_key": k})
        if todo:
            res = self._choose([{kk: v for kk, v in t.items() if kk != "_key"} for t in todo])
            for t, o in zip(todo, res):
                for c, s in zip(t["choices"], o["logprobs"]):
                    cache[(t["_key"], c)] = s
        for i, k in zip(pmi, keys):
            sc = [s - cache[(k, c)] for s, c in zip(out[i]["logprobs"], reqs[i]["choices"])]
            out[i] = dict(out[i], logprobs=[round(s, 5) for s in sc],
                          choice=max(range(len(sc)), key=lambda j: (sc[j], -j)))
        return out

    def descriptors(self, requests: list[dict]) -> list[dict]:
        """Compact, JSON-native description of a batch (what is hashed and logged, never the prompt text)."""
        return [{"digest": request_digest(r), "max_new_tokens": r.get("max_new_tokens"),
                 "temperature": r.get("temperature", 0.0), "seed": r.get("seed"),
                 "n_choices": len(r["choices"]) if r.get("choices") is not None else None} for r in requests]

    def usage(self) -> dict:
        s = dict(self.stats)
        dec = s["decode_s"] + s["prefill_s"]
        s["tokens_per_s"] = round(s["gen_tokens"] / dec, 2) if dec > 0 else None
        s["peak_vram_mb"] = peak_vram_mb(self.device)
        return s

    # ------------------------------------------------------------------ request handling
    def _norm(self, r: dict, choices: bool = False) -> dict:
        if not isinstance(r, dict) or not isinstance(r.get("messages"), list) or not r["messages"]:
            raise ValueError("a request needs a non-empty 'messages' list")
        out = {"messages": [{"role": str(m["role"]), "content": str(m["content"])} for m in r["messages"]]}
        if choices:
            ch = r.get("choices")
            if not isinstance(ch, list) or not ch or not all(isinstance(c, str) and c for c in ch):
                raise ValueError("choose() needs a non-empty list of non-empty strings in 'choices'")
            out["choices"] = list(ch)
            out["normalize"] = r.get("normalize", "mean")
        else:
            mnt = int(r.get("max_new_tokens") or self.max_new_tokens)
            out["max_new_tokens"] = max(1, min(mnt, self.max_new_tokens))
            out["temperature"] = float(r.get("temperature") or 0.0)
            out["seed"] = None if r.get("seed") is None else int(r["seed"])
        if r.get("prefix_messages"):
            out["prefix_messages"] = int(r["prefix_messages"])
        if r.get("thinking") is not None:
            out["thinking"] = bool(r["thinking"])
        if r.get("assistant_prefix"):
            out["assistant_prefix"] = str(r["assistant_prefix"])
        return out

    def _template(self, messages: list[dict], gen_prompt: bool, thinking: bool | None = None) -> str:
        tok = self.weights.tok
        kw = {"tokenize": False, "add_generation_prompt": gen_prompt}
        if "qwen3" in self.model_id.lower():
            kw["enable_thinking"] = self.enable_thinking if thinking is None else bool(thinking)
        return tok.apply_chat_template(messages, **kw)

    def _encode(self, r: dict, reserve: int) -> tuple[list[int], int, bool]:
        """Token ids of the prompt, the length of its declared shared prefix (0 if none) and the truncation flag."""
        tok = self.weights.tok
        text = self._template(r["messages"], True, r.get("thinking")) + r.get("assistant_prefix", "")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        pfx = 0
        k = r.get("prefix_messages") or 0
        if k and 0 < k < len(r["messages"]):
            ptext = self._template(r["messages"][:k], False)
            if text.startswith(ptext):
                pids = tok(ptext, add_special_tokens=False)["input_ids"]
                if ids[:len(pids)] == pids:
                    pfx = len(pids)
        room = self.max_context - reserve
        truncated = False
        if len(ids) > room:
            head = max(room // 3, pfx)
            ids = ids[:head] + ids[-(room - head):]
            truncated = True
        return ids, pfx, truncated

    # ------------------------------------------------------------------ prefix cache
    def _prefix_key(self, ids: Sequence[int]) -> str:
        return hashlib.sha256(_canon(list(ids))).hexdigest()

    def _lookup_prefix(self, ids: list[int]) -> _Prefix | None:
        w = self.weights
        best = None
        for p in w.prefixes.values():
            n = len(p.ids)
            if n <= len(ids) and (best is None or n > len(best.ids)) and tuple(ids[:n]) == p.ids:
                best = p
        return best

    def _get_prefix(self, ids: list[int]) -> _Prefix:
        import torch
        from transformers import DynamicCache
        w = self.weights
        key = self._prefix_key(ids)
        p = w.prefixes.get(key)
        if p is not None:
            w.prefixes.move_to_end(key)
            return p
        t0 = time.time()
        inp = torch.tensor([ids], device=self.device)
        cache = DynamicCache(config=w.model.config)
        with torch.inference_mode():
            out = w.model(input_ids=inp, past_key_values=cache, use_cache=True, logits_to_keep=1)
        self.stats["prefill_s"] += time.time() - t0
        p = _Prefix(tuple(ids), out.past_key_values, out.logits[:, -1, :].float())
        w.prefixes[key] = p
        while len(w.prefixes) > self.prefix_cache_entries:
            w.prefixes.popitem(last=False)
        return p

    def _plan_prefix(self, encoded: list[tuple[list[int], int, bool]]) -> list[_Prefix | None]:
        """Per request: the cached prefix to start from (declared prefixes first, then cached common prefixes,
        then the longest common prefix of the group when it is long enough)."""
        if not self.prefix_cache:
            return [None] * len(encoded)
        out: list[_Prefix | None] = []
        for ids, pfx, _ in encoded:
            p = None
            if pfx >= self.min_prefix_tokens and pfx < len(ids):
                p = self._get_prefix(ids[:pfx])
            else:
                q = self._lookup_prefix(ids)
                if q is not None and len(q.ids) < len(ids):
                    p = q
            out.append(p)
        if len(encoded) >= 2 and all(p is None for p in out):
            seqs = [e[0] for e in encoded]
            n = _lcp(seqs)
            n = min(n, min(len(s) for s in seqs) - 1)
            if n >= self.min_prefix_tokens:
                p = self._get_prefix(seqs[0][:n])
                out = [p] * len(encoded)
        for p in out:
            if p is None:
                self.stats["prefix_misses"] += 1
            else:
                p.hits += 1
                self.stats["prefix_hits"] += 1
                self.stats["prefix_tokens_reused"] += len(p.ids)
        return out

    # ------------------------------------------------------------------ prefill (shared by generate and choose)
    def _prefill(self, rows: list[list[int]], prefix: _Prefix | None):
        """Run the prompts of one batch (all sharing `prefix` or none) and return (logits_last [B,V] float32,
        cache, attention mask [B, T], last position ids [B])."""
        import torch
        from transformers import DynamicCache
        w = self.weights
        B = len(rows)
        P = len(prefix.ids) if prefix is not None else 0
        suf = [r[P:] for r in rows]
        S = max(len(s) for s in suf)
        pad = w.pad_id
        ids = torch.full((B, S), pad, dtype=torch.long)
        sm = torch.zeros((B, S), dtype=torch.long)
        pos = torch.zeros((B, S), dtype=torch.long)
        for i, s in enumerate(suf):
            L = len(s)
            if L:
                ids[i, S - L:] = torch.tensor(s)
                sm[i, S - L:] = 1
                pos[i, S - L:] = torch.arange(P, P + L)
        self.stats["padding_tokens"] += int(B * S - sm.sum())
        mask = torch.cat([torch.ones((B, P), dtype=torch.long), sm], dim=1).to(self.device)
        ids, pos = ids.to(self.device), pos.to(self.device)
        if prefix is not None:
            cache = _copy_cache(prefix.cache, B, w.model.config)
        else:
            cache = DynamicCache(config=w.model.config)
        t0 = time.time()
        with torch.inference_mode():
            if S == 0:
                logits = prefix.last_logits.expand(B, -1).clone()
            else:
                out = w.model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache,
                              use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                logits = out.logits[:, -1, :].float()
        self.stats["prefill_s"] += time.time() - t0
        self.stats["prompt_tokens"] += int(sum(len(r) for r in rows))
        last = torch.tensor([len(r) - 1 for r in rows], device=self.device)
        return logits, cache, mask, last

    # ------------------------------------------------------------------ generation
    def _generate(self, reqs: list[dict]) -> list[dict]:
        self.stats["generate_calls"] += 1
        self.stats["requests"] += len(reqs)
        if not reqs:
            return []
        enc = [self._encode(r, r["max_new_tokens"]) for r in reqs]
        self.stats["truncated"] += sum(1 for e in enc if e[2])
        pref = self._plan_prefix(enc)
        out: list[dict | None] = [None] * len(reqs)
        groups: dict[tuple, list[int]] = {}
        for i, r in enumerate(reqs):
            k = (self._prefix_key(pref[i].ids) if pref[i] is not None else "", r["temperature"] > 0)
            groups.setdefault(k, []).append(i)
        for _, idx in sorted(groups.items()):
            idx.sort(key=lambda i: len(enc[i][0]))
            for c in range(0, len(idx), self.max_batch):
                chunk = idx[c:c + self.max_batch]
                res = self._decode_batch([reqs[i] for i in chunk], [enc[i][0] for i in chunk], pref[chunk[0]])
                for i, rr in zip(chunk, res):
                    rr["truncated"] = enc[i][2]
                    rr["prefix_tokens"] = len(pref[i].ids) if pref[i] is not None else 0
                    out[i] = rr
        return out  # type: ignore[return-value]

    def _decode_batch(self, reqs: list[dict], rows: list[list[int]], prefix: _Prefix | None) -> list[dict]:
        import torch
        w = self.weights
        B = len(rows)
        self.stats["batches"] += 1
        self.stats["rows"] += B
        logits, cache, mask, last = self._prefill(rows, prefix)
        limits = torch.tensor([r["max_new_tokens"] for r in reqs], device=self.device)
        temps = [r["temperature"] for r in reqs]
        gens = [torch.Generator(device=self.device).manual_seed(int(r["seed"] if r["seed"] is not None else 0))
                if r["temperature"] > 0 else None for r in reqs]
        alive = list(range(B))                      # original row index of each live batch row
        toks: list[list[int]] = [[] for _ in range(B)]
        finish = ["length"] * B
        pos = last.clone()
        steps = 0
        t0 = time.time()
        with torch.inference_mode():
            while alive:
                nxt = torch.argmax(logits, dim=-1)
                for j, i in enumerate(alive):
                    if temps[i] > 0:
                        pr = torch.softmax(logits[j] / temps[i], dim=-1)
                        nxt[j] = torch.multinomial(pr, 1, generator=gens[i])[0]
                steps += 1
                nl = nxt.tolist()
                keep = []
                for j, i in enumerate(alive):
                    t = nl[j]
                    if t in w.eos_ids:
                        finish[i] = "stop"
                        continue
                    toks[i].append(t)
                    if len(toks[i]) >= int(limits[i]):
                        continue
                    keep.append(j)
                if not keep:
                    break
                if len(keep) < len(alive):
                    sel = torch.tensor(keep, device=self.device)
                    cache.batch_select_indices(sel)
                    mask, pos, nxt = mask[sel], pos[sel], nxt[sel]
                    alive = [alive[j] for j in keep]
                pos = pos + 1
                mask = torch.cat([mask, torch.ones((len(alive), 1), dtype=mask.dtype, device=self.device)], dim=1)
                o = w.model(input_ids=nxt.unsqueeze(1), attention_mask=mask, position_ids=pos.unsqueeze(1),
                            past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                logits = o.logits[:, -1, :].float()
        self.stats["decode_s"] += time.time() - t0
        n_gen = sum(len(t) for t in toks) + sum(1 for f in finish if f == "stop")
        self.stats["gen_tokens"] += n_gen
        tok = w.tok
        return [{"text": tok.decode(toks[i], skip_special_tokens=True), "prompt_tokens": len(rows[i]),
                 "gen_tokens": len(toks[i]) + (1 if finish[i] == "stop" else 0), "finish": finish[i]}
                for i in range(B)]

    # ------------------------------------------------------------------ option scoring
    def _choose(self, reqs: list[dict]) -> list[dict]:
        import torch
        self.stats["choose_calls"] += 1
        self.stats["requests"] += len(reqs)
        if not reqs:
            return []
        w = self.weights
        t0 = time.time()
        enc = [self._encode(r, 16) for r in reqs]
        self.stats["truncated"] += sum(1 for e in enc if e[2])
        pref = self._plan_prefix(enc)
        opts = [[w.tok(c, add_special_tokens=False)["input_ids"] or [w.pad_id] for c in r["choices"]] for r in reqs]
        out: list[dict | None] = [None] * len(reqs)
        groups: dict[str, list[int]] = {}
        for i in range(len(reqs)):
            groups.setdefault(self._prefix_key(pref[i].ids) if pref[i] is not None else "", []).append(i)
        for _, idx in sorted(groups.items()):
            idx.sort(key=lambda i: len(enc[i][0]))
            c = 0
            while c < len(idx):
                # as many prompts as fit max_batch, and their options within max_choice_rows
                chunk, n_rows = [], 0
                while c < len(idx) and len(chunk) < self.max_batch and \
                        (not chunk or n_rows + len(opts[idx[c]]) <= self.max_choice_rows):
                    chunk.append(idx[c])
                    n_rows += len(opts[idx[c]])
                    c += 1
                self.stats["batches"] += 1
                self.stats["rows"] += len(chunk)
                logits, cache, mask, last = self._prefill([enc[i][0] for i in chunk], pref[chunk[0]])
                lp0 = torch.log_softmax(logits, dim=-1)
                for j, i in enumerate(chunk):
                    out[i] = self._score_options(reqs[i], opts[i], lp0[j], cache, j, mask[j], last[j])
                    out[i]["prompt_tokens"] = len(enc[i][0])
                    out[i]["truncated"] = enc[i][2]
                del cache
        self.stats["choose_s"] += time.time() - t0
        return out  # type: ignore[return-value]

    def _score_options(self, req, opts, lp0, cache, j, mask_row, last_pos) -> dict:
        """Log probability of every option after prompt row `j` of `cache` (options scored in chunks)."""
        import torch
        w = self.weights
        sums = [float(lp0[o[0]]) for o in opts]
        multi = [k for k, o in enumerate(opts) if len(o) > 1]
        for c in range(0, len(multi), self.max_choice_rows):
            ks = multi[c:c + self.max_choice_rows]
            n = len(ks)
            Lc = max(len(opts[k]) for k in ks) - 1
            inp = torch.full((n, Lc), w.pad_id, dtype=torch.long)
            cm = torch.zeros((n, Lc), dtype=torch.long)
            for r, k in enumerate(ks):
                t = opts[k][:-1]
                inp[r, :len(t)] = torch.tensor(t)
                cm[r, :len(t)] = 1
            sub = _select_cache(cache, [j] * n, w.model.config)
            m = torch.cat([mask_row.unsqueeze(0).expand(n, -1), cm.to(self.device)], dim=1)
            pos = (last_pos + 1 + torch.arange(Lc, device=self.device)).unsqueeze(0).expand(n, -1)
            with torch.inference_mode():
                o = w.model(input_ids=inp.to(self.device), attention_mask=m, position_ids=pos, past_key_values=sub,
                            use_cache=True)
            lp = torch.log_softmax(o.logits.float(), dim=-1)
            for r, k in enumerate(ks):
                tgt = torch.tensor(opts[k][1:], device=self.device)
                sums[k] += float(lp[r, torch.arange(len(tgt), device=self.device), tgt].sum())
            del sub, o, lp
        norm = req.get("normalize", "mean")
        scores = [s / len(o) if norm == "mean" else s for s, o in zip(sums, opts)]
        best = max(range(len(scores)), key=lambda k: (scores[k], -k))
        return {"choice": best, "logprobs": [round(s, 5) for s in scores], "option_tokens": [len(o) for o in opts]}

    # ------------------------------------------------------------------ measurements
    def measure_prefix_cache(self, requests: list[dict]) -> dict:
        """Run the same batch with and without prefix caching (after a warm-up) and report wall times and outputs
        agreement. The cache entries created by the measurement stay (they are ordinary cache entries)."""
        reqs = [self._norm(r) for r in requests]
        self._generate(reqs[:1])                         # warm-up (kernels, allocator)
        res = {}
        for mode in (False, True):
            saved = self.prefix_cache
            self.prefix_cache = mode
            if mode:
                self.weights.prefixes.clear()
                self._generate(reqs[:1])                 # the shared prefix is computed once ...
            _sync(self.device)
            t0 = time.time()
            outs = self._generate(reqs)                  # ... and reused by the measured batch
            _sync(self.device)
            res["with_prefix" if mode else "without_prefix"] = {"wall_s": round(time.time() - t0, 4),
                                                                "texts": [o["text"] for o in outs],
                                                                "prefix_tokens": [o["prefix_tokens"] for o in outs]}
            self.prefix_cache = saved
        a, b = res["without_prefix"], res["with_prefix"]
        same = sum(1 for x, y in zip(a["texts"], b["texts"]) if x == y)
        return {"n": len(reqs), "without_prefix_s": a["wall_s"], "with_prefix_s": b["wall_s"],
                "speedup": round(a["wall_s"] / b["wall_s"], 3) if b["wall_s"] > 0 else None,
                "prefix_tokens": max(b["prefix_tokens"] or [0]), "identical_outputs": same}


# ====================================================================== helpers
def _lcp(seqs: list[list[int]]) -> int:
    n = min(len(s) for s in seqs)
    first = seqs[0]
    for k in range(n):
        t = first[k]
        if any(s[k] != t for s in seqs[1:]):
            return k
    return n


def _cache_tensors(cache) -> list[tuple]:
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    return [(k, v) for k, v in zip(cache.key_cache, cache.value_cache)]


def _copy_cache(cache, B: int, config):
    """A new DynamicCache holding `cache` (batch 1) repeated `B` times (the source is not modified)."""
    from transformers import DynamicCache
    data = [(k.expand(B, *k.shape[1:]).contiguous(), v.expand(B, *v.shape[1:]).contiguous())
            for k, v in _cache_tensors(cache)]
    return DynamicCache(data, config=config)


def _select_cache(cache, rows: list[int], config):
    import torch
    from transformers import DynamicCache
    idx = torch.tensor(rows, device=_cache_tensors(cache)[0][0].device)
    data = [(k.index_select(0, idx), v.index_select(0, idx)) for k, v in _cache_tensors(cache)]
    return DynamicCache(data, config=config)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def peak_vram_mb(device: str = "cuda") -> float | None:
    """Peak GPU memory allocated by this process (MiB), or None on CPU."""
    try:
        import torch
        if device.startswith("cuda") and torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def _cache_dir() -> str:
    d = os.path.join(os.environ.get("CRUCIBLE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "crucible")),
                     "semantic")
    os.makedirs(d, exist_ok=True)
    return d


def _cached_hash(key: tuple, compute: Callable[[], str]) -> str:
    """Weights hashes are cached on disk per (model, revision, device class, dtype, quantization): the files of a
    revision never change, and hashing 1-4 GB of parameters on every start would dominate short runs."""
    name = hashlib.sha256(_canon([key[0], key[1], "cuda" if key[2].startswith("cuda") else key[2], key[3],
                                  key[4]])).hexdigest()[:24]
    path = os.path.join(_cache_dir(), f"weights_hash_{name}.json")
    try:
        with open(path) as fh:
            rec = json.load(fh)
        if rec.get("key") == [key[0], key[1], key[3], key[4]]:
            return rec["weights_hash"]
    except (OSError, ValueError):
        pass
    h = compute()
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump({"key": [key[0], key[1], key[3], key[4]], "weights_hash": h}, fh)
    os.replace(tmp, path)
    return h


# ====================================================================== process-wide registry
_SERVICES: "dict[str, SemanticService]" = {}


def _register(svc: SemanticService) -> None:
    with _LOCK:
        _SERVICES.setdefault(svc.model_id, svc)


def service_for(model_id: str = DEFAULT_MODEL, **kw) -> SemanticService:
    """The process's default service for `model_id` (created on first use)."""
    with _LOCK:
        s = _SERVICES.get(model_id)
        if s is None:
            s = SemanticService(model_id, **kw)
        return s


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


def binom_sf(k: int, n: int, p: float) -> float:
    """P[X >= k] for X ~ Binomial(n, p) (exact, in log space)."""
    if k <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k, n + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def poisson_binom_sf(k: int, ps: Sequence[float]) -> float:
    """P[X >= k] for X a sum of independent Bernoulli(p_i) (chance levels that differ per item)."""
    dist = [1.0]
    for p in ps:
        nxt = [0.0] * (len(dist) + 1)
        for i, q in enumerate(dist):
            nxt[i] += q * (1 - p)
            nxt[i + 1] += q * p
        dist = nxt
    return float(min(1.0, sum(dist[k:]))) if k < len(dist) else 0.0
