"""The frozen language model as a service: text generation and text embeddings, with a persistent cache.

Nothing here is trained. The weights of the language model are loaded with ``requires_grad=False`` and never
change; `fingerprint()` lets an experiment record that. Three interchangeable back ends implement `LLM`:

* `HFModel`      - a Hugging Face causal LM loaded from the local cache (never downloads).
* `RemoteLLM`    - a client of `superlearner.llm.server` (one process owns the GPU model; rollout workers and
                   candidate runs talk to it over a unix socket).
* `ScriptedLLM`  - TEST DOUBLE ONLY: replies come from a function. Never used for a reported result.

Embeddings are 512-dimensional: the mean-pooled last hidden state of the frozen model, passed through a FIXED
seeded random projection and L2-normalised. `HashEmbedder` is a model-free fallback (hashed word and character
n-gram features) used for CPU-only smoke tests; runs record which embedder they used.

Generation is greedy for ``variant == 0`` (deterministic for a fixed model and prompt) and sampled with a seed
derived from the prompt for ``variant > 0``, so every (prompt, variant) pair has one cached answer.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

EMB_DIM = 512


def text_key(text: str) -> int:
    """Stable positive 62-bit key of a text (0 is reserved for "no text")."""
    return (int.from_bytes(hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).digest(), "big") >> 2) or 1


def prompt_key(model: str, messages: list[dict], max_new_tokens: int, variant: int, think: bool = False) -> str:
    blob = json.dumps([model, messages, int(max_new_tokens), int(variant)] + (["think"] if think else []),
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()


@dataclass
class GenResult:
    text: str
    prompt_tokens: int = 0
    gen_tokens: int = 0
    wall_s: float = 0.0
    cached: bool = False
    truncated: bool = False


class LLM:
    """Interface of the frozen model service."""
    name = "abstract"
    is_test_double = False

    def generate(self, messages: list[dict], max_new_tokens: int = 512, variant: int = 0, think: bool = False) -> GenResult:
        """`think=True` lets a model that has a reasoning mode reason before answering (the reasoning is not
        returned; its tokens are counted in `gen_tokens`)."""
        raise NotImplementedError

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def fingerprint(self) -> str:
        return self.name

    def usage(self) -> dict:
        return {}


# ============================================================================================ model-free embedder
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


class HashEmbedder:
    """Deterministic model-free text embedding: signed feature hashing of words, word bigrams and character
    trigrams into `EMB_DIM` dimensions, sqrt term weighting, L2 normalisation."""
    name = "hash-embedder"

    def __init__(self, dim: int = EMB_DIM):
        self.dim = dim

    def _feats(self, text: str) -> Iterable[str]:
        words = [w.lower() for w in _WORD.findall(text)][:400]
        for w in words:
            yield "w:" + w
            for part in w.split("_"):
                if part and part != w:
                    yield "w:" + part
        for a, b in zip(words, words[1:]):
            yield f"b:{a}|{b}"
        s = " ".join(words)[:1200]
        for i in range(len(s) - 2):
            yield "c:" + s[i:i + 3]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            cnt: dict[str, int] = {}
            for f in self._feats(t):
                cnt[f] = cnt.get(f, 0) + 1
            for f, c in cnt.items():
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=5).digest(), "big")
                out[i, h % self.dim] += (1.0 if (h >> 39) & 1 else -1.0) * np.sqrt(c) * (0.5 if f[0] == "c" else 1.0)
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


# ============================================================================================ test double
class ScriptedLLM(LLM):
    """TEST DOUBLE ONLY - not a model. `reply(messages, variant) -> str`. Embeddings come from `HashEmbedder`."""
    name = "scripted-test-double"
    is_test_double = True

    def __init__(self, reply: Callable[[list[dict], int], str] | None = None):
        self.reply = reply or (lambda messages, variant: "{}")
        self.emb = HashEmbedder()
        self.calls: list[list[dict]] = []
        self.stats = {"gen_calls": 0, "prompt_tokens": 0, "gen_tokens": 0, "embed_texts": 0, "gen_cache_hits": 0}

    def generate(self, messages, max_new_tokens=512, variant=0, think=False):
        self.calls.append(messages)
        text = self.reply(messages, variant)
        self.stats["gen_calls"] += 1
        self.stats["prompt_tokens"] += sum(len(m["content"].split()) for m in messages)
        self.stats["gen_tokens"] += len(text.split())
        return GenResult(text, sum(len(m["content"].split()) for m in messages), len(text.split()))

    def embed(self, texts):
        self.stats["embed_texts"] += len(texts)
        return self.emb.embed(texts)

    def usage(self):
        return dict(self.stats)


class NullLLM(ScriptedLLM):
    """No language model: generation returns an empty object. Used for CPU-only smoke tests and for the
    no-LLM ablation of the controller (the controller then acts only through its own decoder)."""
    name = "null-llm"
    is_test_double = False


# ============================================================================================ cache
class Cache:
    """sqlite cache of generations (by prompt key) and embeddings (by text key). Safe for many reader processes and
    one writer at a time (WAL mode; writes retry on a busy database)."""

    def __init__(self, path: str | None, readonly: bool = False):
        self.path = path
        self.readonly = readonly
        self._local = threading.local()
        if path and not readonly:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            c = self._conn()
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS gen (k TEXT PRIMARY KEY, model TEXT, text TEXT, prompt_tokens INT, "
                      "gen_tokens INT, wall_s REAL, t REAL)")
            c.execute("CREATE TABLE IF NOT EXISTS emb (k INTEGER, model TEXT, vec BLOB, text TEXT, PRIMARY KEY (k, model))")
            c.commit()

    def _conn(self):
        c = getattr(self._local, "c", None)
        if c is None and self.path:
            if self.readonly and not os.path.exists(self.path):
                return None
            uri = f"file:{self.path}?mode=ro" if self.readonly else self.path
            c = sqlite3.connect(uri, uri=self.readonly, timeout=30.0, check_same_thread=False)
            self._local.c = c
        return c

    def get_gen(self, key: str) -> GenResult | None:
        c = self._conn()
        if c is None:
            return None
        try:
            row = c.execute("SELECT text, prompt_tokens, gen_tokens, wall_s FROM gen WHERE k=?", (key,)).fetchone()
        except sqlite3.OperationalError:
            return None
        return GenResult(row[0], row[1], row[2], row[3], cached=True) if row else None

    def put_gen(self, key: str, model: str, g: GenResult) -> None:
        c = self._conn()
        if c is None or self.readonly:
            return
        c.execute("INSERT OR REPLACE INTO gen VALUES (?,?,?,?,?,?,?)",
                  (key, model, g.text, g.prompt_tokens, g.gen_tokens, g.wall_s, time.time()))
        c.commit()

    def get_emb(self, model: str, keys: list[int]) -> dict[int, np.ndarray]:
        c = self._conn()
        out: dict[int, np.ndarray] = {}
        if c is None or not keys:
            return out
        try:
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                q = "SELECT k, vec FROM emb WHERE model=? AND k IN (%s)" % ",".join("?" * len(part))
                for k, blob in c.execute(q, [model] + part):
                    out[int(k)] = np.frombuffer(blob, np.float16).astype(np.float32)
        except sqlite3.OperationalError:
            pass
        return out

    def put_emb(self, model: str, items: list[tuple[int, np.ndarray, str]]) -> None:
        c = self._conn()
        if c is None or self.readonly or not items:
            return
        c.executemany("INSERT OR REPLACE INTO emb VALUES (?,?,?,?)",
                      [(int(k), model, v.astype(np.float16).tobytes(), t[:400]) for k, v, t in items])
        c.commit()

    def counts(self) -> dict:
        c = self._conn()
        if c is None:
            return {"gen": 0, "emb": 0}
        try:
            return {"gen": c.execute("SELECT COUNT(*) FROM gen").fetchone()[0],
                    "emb": c.execute("SELECT COUNT(*) FROM emb").fetchone()[0]}
        except sqlite3.OperationalError:
            return {"gen": 0, "emb": 0}


class CachedLLM(LLM):
    """Wraps an `LLM` with the persistent cache and usage accounting. Cache hits cost no model compute and are
    counted separately (`gen_cache_hits`), so compute-matched comparisons can use `gen_tokens` (generated by the
    model in this run) and `served_tokens` (delivered to the agent, cached or not)."""

    def __init__(self, inner: LLM, cache_path: str | None = None):
        self.inner = inner
        self.name = inner.name
        self.is_test_double = getattr(inner, "is_test_double", False)
        self.cache = Cache(cache_path)
        self._mem: dict[int, np.ndarray] = {}
        self.stats = {"gen_calls": 0, "gen_cache_hits": 0, "prompt_tokens": 0, "gen_tokens": 0, "served_tokens": 0,
                      "embed_texts": 0, "embed_cache_hits": 0, "gen_wall_s": 0.0}

    def generate(self, messages, max_new_tokens=512, variant=0, think=False):
        key = prompt_key(self.name, messages, max_new_tokens, variant, think)
        self.stats["gen_calls"] += 1
        g = self.cache.get_gen(key)
        if g is not None:
            self.stats["gen_cache_hits"] += 1
            self.stats["served_tokens"] += g.gen_tokens
            return g
        g = self.inner.generate(messages, max_new_tokens, variant, think)
        self.cache.put_gen(key, self.name, g)
        self.stats["prompt_tokens"] += g.prompt_tokens
        self.stats["gen_tokens"] += g.gen_tokens
        self.stats["served_tokens"] += g.gen_tokens
        self.stats["gen_wall_s"] += g.wall_s
        return g

    def generate_many(self, reqs: list[tuple]) -> list[GenResult]:
        """`reqs` = [(messages, max_new_tokens, variant, think)] sharing the last three values. Cache-aware."""
        out: list = [None] * len(reqs)
        miss = []
        for i, (m, n, v, th) in enumerate(reqs):
            self.stats["gen_calls"] += 1
            g = self.cache.get_gen(prompt_key(self.name, m, n, v, th))
            if g is not None:
                self.stats["gen_cache_hits"] += 1
                self.stats["served_tokens"] += g.gen_tokens
                out[i] = g
            else:
                miss.append(i)
        if miss:
            _, n, v, th = reqs[miss[0]]
            if hasattr(self.inner, "generate_batch") and len(miss) > 1:
                gs = self.inner.generate_batch([reqs[i][0] for i in miss], n, v, th)
            else:
                gs = [self.inner.generate(reqs[i][0], n, v, th) for i in miss]
            for i, g in zip(miss, gs):
                self.cache.put_gen(prompt_key(self.name, reqs[i][0], n, v, th), self.name, g)
                self.stats["prompt_tokens"] += g.prompt_tokens
                self.stats["gen_tokens"] += g.gen_tokens
                self.stats["served_tokens"] += g.gen_tokens
                self.stats["gen_wall_s"] += g.wall_s
                out[i] = g
        return out

    def embed(self, texts):
        keys = [text_key(t) for t in texts]
        self.stats["embed_texts"] += len(texts)
        ns = self.name + "#" + getattr(self.inner, "EMB_TAG", "v1")
        need = [k for k in dict.fromkeys(keys) if k not in self._mem]
        if need:
            self._mem.update(self.cache.get_emb(ns, need))
        miss = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self._mem]
        self.stats["embed_cache_hits"] += len(texts) - len(miss)
        if miss:
            vecs = self.inner.embed([t for _, t in miss])
            self.cache.put_emb(ns, [(k, v, t) for (k, t), v in zip(miss, vecs)])
            for (k, _), v in zip(miss, vecs):
                self._mem[k] = np.asarray(v, np.float32)
        return np.stack([self._mem[k] for k in keys]) if keys else np.zeros((0, EMB_DIM), np.float32)

    def fingerprint(self):
        return self.inner.fingerprint()

    def usage(self):
        return {"model": self.name, **self.stats}


# ============================================================================================ Hugging Face model
class HFModel(LLM):
    """A frozen causal LM from the local Hugging Face cache (``local_files_only=True``; thinking mode off)."""

    def __init__(self, model: str = "Qwen/Qwen3-14B", device: str = "cuda", max_context: int = 16384,
                 embed_tokens: int = 256, temperature: float = 0.8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        self.torch = torch
        self.name, self.device, self.max_context = model, device, max_context
        self.embed_tokens, self.temperature = embed_tokens, temperature
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, local_files_only=True).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        hid = self.model.config.hidden_size
        g = torch.Generator().manual_seed(20261121)
        self.proj = (torch.randn(hid, EMB_DIM, generator=g) / np.sqrt(EMB_DIM)).to(device=device, dtype=torch.float32)
        self._fp: str | None = None
        self.lock = threading.Lock()

    def fingerprint(self) -> str:
        if self._fp is None:
            h = hashlib.sha256(self.name.encode())
            for k, p in sorted(self.model.state_dict().items()):
                h.update(k.encode())
                h.update(p.detach().float().sum().cpu().numpy().tobytes())
            self._fp = h.hexdigest()[:16]
        return self._fp

    def _template(self, messages: list[dict], think: bool = False) -> str:
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        except TypeError:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, messages, max_new_tokens=512, variant=0, think=False):
        torch = self.torch
        t0 = time.time()
        ids = self.tok(self._template(messages, think), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        truncated = False
        if think:
            max_new_tokens = max_new_tokens + 1500          # room for the reasoning, which is discarded
        room = self.max_context - max_new_tokens
        if len(ids) > room:
            ids = torch.cat([ids[: room // 3], ids[-(room - room // 3):]])
            truncated = True
        inp = ids.unsqueeze(0).to(self.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        with self.lock, torch.no_grad():
            if variant:
                seed = int(hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:8], 16) + variant
                torch.manual_seed(seed % (2**31))
                kw.update(do_sample=True, temperature=self.temperature, top_p=0.95, top_k=0)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            out = self.model.generate(inp, attention_mask=torch.ones_like(inp), **kw)
        gen = out[0, inp.shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        if "<think>" in text:                 # reasoning ran out of tokens before an answer
            text = ""
        return GenResult(text.strip(), int(inp.shape[1]), int(gen.shape[0]), time.time() - t0, truncated=truncated)

    def generate_batch(self, batch: list[list[dict]], max_new_tokens=512, variant=0, think=False) -> list[GenResult]:
        """Several prompts in one forward pass (left-padded). Greedy for ``variant == 0``."""
        if len(batch) == 1:
            return [self.generate(batch[0], max_new_tokens, variant, think)]
        torch = self.torch
        t0 = time.time()
        if think:
            max_new_tokens = max_new_tokens + 1500
        room = self.max_context - max_new_tokens
        texts = [self._template(m, think) for m in batch]
        self.tok.padding_side = "left"
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=room)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        with self.lock, torch.no_grad():
            if variant:
                seed = int(hashlib.sha256(json.dumps(batch, sort_keys=True).encode()).hexdigest()[:8], 16) + variant
                torch.manual_seed(seed % (2**31))
                kw.update(do_sample=True, temperature=self.temperature, top_p=0.95, top_k=0)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            out = self.model.generate(**enc, **kw)
        n_in = enc["input_ids"].shape[1]
        res = []
        eos = set(x for x in [self.tok.eos_token_id, self.tok.pad_token_id] if x is not None)
        for i in range(len(batch)):
            gen = out[i, n_in:]
            n_gen = int(sum(1 for t in gen.tolist() if t not in eos)) + 1
            text = self.tok.decode(gen, skip_special_tokens=True)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
            if "<think>" in text:
                text = ""
            res.append(GenResult(text.strip(), int(enc["attention_mask"][i].sum()), n_gen, (time.time() - t0) / len(batch)))
        return res

    # Frozen sentence embeddings. A causal LM's mean-pooled states are poor for retrieval; the prompt-based
    # last-token state ("... means in one word:") from a late layer is much better (measured on tool retrieval,
    # docs/ARCHITECTURE.md). A fixed centre, estimated once from a built-in list of calibration texts, removes
    # the common component shared by all embeddings.
    EMB_TEMPLATE = 'This text: "{t}" means in one word: "'
    EMB_LAYER = -4
    EMB_TAG = "eol-4c"

    def _raw_embed(self, texts):
        torch = self.torch
        out = []
        for i in range(0, len(texts), 16):
            batch = [self.EMB_TEMPLATE.replace("{t}", (t if t else " ")[:1500]) for t in texts[i:i + 16]]
            self.tok.padding_side = "left"
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.embed_tokens + 16)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self.lock, torch.no_grad():
                hs = self.model(**enc, output_hidden_states=True).hidden_states[self.EMB_LAYER]
            out.append(hs[:, -1].float())
        return torch.cat(out) if out else torch.zeros(0, self.model.config.hidden_size, device=self.device)

    def embed(self, texts):
        torch = self.torch
        if not texts:
            return np.zeros((0, EMB_DIM), np.float32)
        if getattr(self, "_centre", None) is None:
            self._centre = self._raw_embed(list(CALIBRATION_TEXTS)).mean(0, keepdim=True)
        v = (self._raw_embed(list(texts)) - self._centre) @ self.proj
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return v.cpu().numpy().astype(np.float32)


CALIBRATION_TEXTS = (
    "Open a checking account for yourself or an organisation.", "Send money from one account to another.",
    "Create an organisation; you become its owner.", "Write files to a repository as a new commit.",
    "Deploy a repository as a service.", "Rent a compute instance billed every hour.",
    "Publish a listing on the market.", "Search the public web of this world.", "Read a document.",
    "Publish a public post on a forum board.", "Send a private message.", "Offer employment to an applicant.",
    "Propose a contract built from clauses.", "Open a support ticket with the operator of a service.",
    "Start a search-ad campaign.", "Schedule an event and invite attendees.", "File an issue in a project.",
    "List your rented instances.", "Operational metrics of a service.", "Current simulated time.",
    "Earn revenue from independent customers.", "Operate a service that serves many requests.",
    "Own a registered company.", "Hire three people into your organisation.", "Stay solvent for a month.",
    "no account acct_9", "args.handle: does not match the pattern", "insufficient funds: needs $2.00",
    "you are not allowed to do this", "unknown tool", "the service is unavailable", "rate limit exceeded",
    "Input: a list of records. Output: the sorted ids.", "Shortest path in a weighted road network.",
    "A listing for an API that sorts records, price 5 cents per call.", "A bounty for software, reward $3,000.",
    "The weather is nice today.", "def solve(args, caller): return {}", "{\"entry\": \"service.py\"}",
    "balance: 300000; owner: p:agent; status: open", "provider: o:cloud1; price per core hour: 200",
)


def extract_json(text: str):
    """First balanced JSON object or array in a model reply (code fences and prose around it are ignored).
    Returns the parsed value or None."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    starts = [m.start() for m in re.finditer(r"[\[{]", text)]
    for s in starts[:40]:
        depth, in_str, esc = 0, False, False
        for i in range(s, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "[{":
                depth += 1
            elif ch in "]}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[s:i + 1])
                    except (ValueError, RecursionError):
                        break
    return None
