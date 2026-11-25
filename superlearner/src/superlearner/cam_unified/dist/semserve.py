"""Shared semantic-model server: one copy of the pretrained language model (Qwen3) serves every actor process
(spec 22 "shared semantic-model serving", "batched actor inference"; spec 14.1 actors hold a *versioned* copy of
the trainable weights, the frozen semantic backbone is shared).

Transport: one request queue (``torch.multiprocessing`` spawn context, so CPU tensors in replies travel through
shared memory, not pickled bytes) and one reply queue per client. Request = ``(op, req_id, client_id, payload)``;
reply = ``(req_id, result | {"error": ...})``.

Ops:

* ``hidden``  {"texts", "max_tokens", "layer"} → {"h": [N, T, H] bf16, "mask": [N, T] bool, "version"} — token
  hidden states of the chosen layer (``-1`` = final post-norm), right-padded per request;
* ``embed``   {"texts", "max_tokens"} → {"e": [N, H] float32, "version"} — mask-mean-pooled final hidden state;
* ``generate`` {"prompts", "max_new_tokens", "greedy", "temperature", "seed", "prefix_embeds"?} → {"tokens": [[id]],
  "text": [str], "logprobs": [[float]], "version"} — the reply's behaviour log-probabilities are those of the
  served decoder version (``decoder_version``), so actors can log exact ``logp`` for off-policy correction;
* ``info`` → model name, hidden size, versions, stats.

**Batching.** The server blocks for the first request, then keeps collecting for up to ``max_wait_ms`` or until
``max_batch_texts`` texts are queued; requests of the same op and parameters (from any client) are concatenated into
one padded forward pass and the results are split back per request. ``stats`` records batches, texts per batch, and
how many distinct clients shared a batch.

**Versions.** ``encoder_version`` names the weights used for ``hidden``/``embed``: with ``encoder_lora=False``
(architecture decision 9: frozen encoder) the adapter is disabled for those ops and the version stays
``base:<model>``, so hidden-state caches remain valid across decoder updates. ``decoder_version`` names the LoRA
adapter used for ``generate``. The learner publishes adapter tensors with `CheckpointPublisher` into
``lora_root``; the server adopts a new validated version only **between batches** (a version boundary), never inside
one. A content-hash cache of hidden states keyed by (sha256(text), encoder_version, layer, max_tokens) is never
served across versions.
"""
from __future__ import annotations

import hashlib
import os
import queue as _queue
import time
from collections import Counter, OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F

from ..observe.accounting import ComputeMeter, atomic_write_json
from .publish import load_weights, read_latest


class _LRU:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.d: OrderedDict = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    def get(self, k):
        v = self.d.get(k)
        if v is None:
            self.misses += 1
            return None
        self.d.move_to_end(k)
        self.hits += 1
        return v

    def put(self, k, v: torch.Tensor):
        n = v.numel() * v.element_size()
        if n > self.max_bytes:
            return
        if k in self.d:
            self.bytes -= self.d[k].numel() * self.d[k].element_size()
        self.d[k] = v
        self.d.move_to_end(k)
        self.bytes += n
        while self.bytes > self.max_bytes and self.d:
            _, old = self.d.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()


class SemanticServer:
    """Holds one model; `handle_batch` serves a list of requests. `serve` is the process loop."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", device: str = "cuda", max_batch_texts: int = 32,
                 max_wait_ms: float = 5.0, max_tokens: int = 512, hidden_layer: int = -1, encoder_lora: bool = False,
                 lora: bool = True, lora_rank: int = 16, lora_alpha: int = 32,
                 lora_targets=("q_proj", "k_proj", "v_proj", "o_proj"), cache_bytes: int = 512 * 2**20,
                 lora_root: Optional[str] = None, dtype=torch.bfloat16, meter: Optional[ComputeMeter] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.max_batch_texts, self.max_wait_ms, self.max_tokens = max_batch_texts, max_wait_ms, max_tokens
        self.hidden_layer = hidden_layer
        self.encoder_lora = encoder_lora
        self.lora_root = lora_root
        self.meter = meter or ComputeMeter("semserve", role="semserve")
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, local_files_only=True)
        base.to(self.device).eval()
        for p in base.parameters():
            p.requires_grad_(False)
        self.lora = lora
        if lora:
            from peft import LoraConfig, get_peft_model
            self.model = get_peft_model(base, LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
                                                         target_modules=list(lora_targets), task_type="CAUSAL_LM"))
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        else:
            self.model = base
        self.load_s = time.time() - t0
        self.sem_dim = int(base.config.hidden_size)
        self.vocab_rows = int(base.get_input_embeddings().weight.shape[0])
        self.lora_version = -1                     # -1 = freshly initialised adapter (B = 0: identity)
        self.base_tag = f"base:{model_name}"
        self.cache = _LRU(cache_bytes)
        self.stats = Counter()
        self.batch_texts_hist = Counter()
        self.batch_clients_hist = Counter()
        self.max_clients_in_batch = 0

    # ---------------------------------------------------------------- versions
    @property
    def decoder_version(self) -> str:
        return f"{self.base_tag}+lora:v{self.lora_version}" if self.lora else self.base_tag

    @property
    def encoder_version(self) -> str:
        return self.decoder_version if (self.lora and self.encoder_lora) else self.base_tag

    def maybe_reload(self) -> bool:
        """Adopt a newer validated adapter version from ``lora_root`` (call only between batches)."""
        if not self.lora or not self.lora_root:
            return False
        lt = read_latest(self.lora_root)
        if lt is None or int(lt["version"]) <= self.lora_version:
            return False
        from peft import set_peft_model_state_dict
        sd = load_weights(os.path.join(self.lora_root, lt["dir"], "weights.pt"), lt["sha256"])["state_dict"]
        res = set_peft_model_state_dict(self.model, {k: v.to(self.device) for k, v in sd.items()})
        unexpected = getattr(res, "unexpected_keys", [])
        if unexpected:
            raise RuntimeError(f"adapter checkpoint has unexpected keys: {unexpected[:4]}")
        self.lora_version = int(lt["version"])
        self.stats["lora_reloads"] += 1
        return True

    def adapter_state(self) -> dict:
        """Current adapter tensors (what a learner would publish)."""
        from peft import get_peft_model_state_dict
        return {k: v.detach().cpu() for k, v in get_peft_model_state_dict(self.model).items()}

    # ---------------------------------------------------------------- model ops
    def _enc_ctx(self):
        if self.lora and not self.encoder_lora:
            return self.model.disable_adapter()
        import contextlib
        return contextlib.nullcontext()

    @torch.no_grad()
    def _hidden_batch(self, texts: list, max_tokens: int, layer: int) -> list:
        """-> list of [T_i, H] bf16 CPU tensors (each text's own tokens only)."""
        out: list = [None] * len(texts)
        keys = [(hashlib.sha256(t.encode()).hexdigest(), self.encoder_version, layer, max_tokens) for t in texts]
        todo = []
        for i, k in enumerate(keys):
            v = self.cache.get(k)
            if v is None:
                todo.append(i)
            else:
                out[i] = v
        if todo:
            enc = self.tok([texts[i] for i in todo], return_tensors="pt", padding=True, truncation=True,
                           max_length=max_tokens, padding_side="right")
            ids = enc["input_ids"].to(self.device)
            att = enc["attention_mask"].to(self.device)
            self.stats["tokens_prompt"] += int(att.sum())
            self.meter.add(tokens_prompt=int(att.sum()))
            with self._enc_ctx(), self.meter.gpu():
                res = self.model(input_ids=ids, attention_mask=att, output_hidden_states=True, logits_to_keep=1)
            hs = res.hidden_states[layer]                                  # [n, T, H]
            lens = att.sum(1).tolist()
            for j, i in enumerate(todo):
                v = hs[j, :int(lens[j])].to("cpu")
                out[i] = v
                self.cache.put(keys[i], v)
        return out

    @torch.no_grad()
    def _generate_batch(self, prompts: list, max_new_tokens: int, greedy: bool, temperature: float,
                        seed: Optional[int], prefixes: Optional[list]) -> list:
        enc = self.tok(prompts, return_tensors="pt", padding=True, padding_side="left")
        ids = enc["input_ids"].to(self.device)
        att = enc["attention_mask"].to(self.device)
        emb = self.model.get_input_embeddings()(ids)
        if prefixes is not None:
            pre = torch.stack([p.to(self.device, emb.dtype) for p in prefixes])          # [n, K, H]
            emb = torch.cat([emb, pre], 1)
            att = torch.cat([att, torch.ones(att.shape[0], pre.shape[1], dtype=att.dtype, device=self.device)], 1)
        if seed is not None:
            torch.manual_seed(seed)
        kw = dict(do_sample=False, temperature=None, top_p=None, top_k=None) if greedy else \
            dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        self.stats["tokens_prompt"] += int(att.sum())
        self.meter.add(tokens_prompt=int(att.sum()))
        with self.meter.gpu():
            gen = self.model.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tok.pad_token_id, return_dict_in_generate=True,
                                      output_scores=True, **kw)
        seqs = gen.sequences                                               # only new tokens (inputs_embeds)
        # behaviour log-probabilities of the sampled tokens under the distribution actually sampled from (the
        # processed scores: temperature applied when sampling); one step at a time to avoid an [n, L, V] tensor
        lp = torch.stack([F.log_softmax(sc.float(), -1).gather(-1, seqs[:, t:t + 1]).squeeze(-1)
                          for t, sc in enumerate(gen.scores)], 1)          # [n, L]
        eos = {self.tok.eos_token_id, self.tok.pad_token_id}
        out = []
        for i in range(seqs.shape[0]):
            toks = seqs[i].tolist()
            n = len(toks)
            for j, t in enumerate(toks):
                if t in eos:
                    n = j + 1
                    break
            out.append({"tokens": toks[:n], "text": self.tok.decode(toks[:n], skip_special_tokens=True),
                        "logprobs": lp[i, :n].tolist()})
            self.stats["tokens_generated"] += n
            self.meter.add(tokens_generated=n)
        return out

    # ---------------------------------------------------------------- batching
    def handle_batch(self, reqs: list) -> list:
        """reqs: [(op, req_id, client_id, payload)] -> [(client_id, req_id, result)]. Requests with equal op and
        parameters are fused into one forward pass."""
        self.maybe_reload()
        groups: dict = {}
        replies = []
        for r in reqs:
            op, rid, cid, pl = r
            if op in ("hidden", "embed"):
                key = (op, int(pl.get("max_tokens", self.max_tokens)), int(pl.get("layer", self.hidden_layer))
                       if op == "hidden" else -1)
            elif op == "generate":
                key = (op, int(pl.get("max_new_tokens", 32)), bool(pl.get("greedy", True)),
                       float(pl.get("temperature", 1.0)), pl.get("seed"), pl.get("prefix_embeds") is not None)
            elif op == "info":
                replies.append((cid, rid, self.info()))
                continue
            else:
                replies.append((cid, rid, {"error": f"unknown op {op!r}"}))
                continue
            groups.setdefault(key, []).append(r)
        for key, rs in groups.items():
            op = key[0]
            texts, spans = [], []
            for _, rid, cid, pl in rs:
                items = pl["texts"] if op != "generate" else pl["prompts"]
                spans.append((cid, rid, len(texts), len(texts) + len(items)))
                texts.extend(items)
            clients = {c for c, _, _, _ in spans}
            self.stats["batches"] += 1
            self.stats["requests"] += len(rs)
            self.stats["texts"] += len(texts)
            self.batch_texts_hist[len(texts)] += 1
            self.batch_clients_hist[len(clients)] += 1
            self.max_clients_in_batch = max(self.max_clients_in_batch, len(clients))
            try:
                if op == "generate":
                    prefixes = None
                    if key[5]:
                        prefixes = [p for _, _, _, pl in rs for p in pl["prefix_embeds"]]
                    res = self._generate_batch(texts, key[1], key[2], key[3], key[4], prefixes)
                    for cid, rid, a, b in spans:
                        replies.append((cid, rid, {"tokens": [x["tokens"] for x in res[a:b]],
                                                   "text": [x["text"] for x in res[a:b]],
                                                   "logprobs": [x["logprobs"] for x in res[a:b]],
                                                   "version": self.decoder_version, "batch_texts": len(texts),
                                                   "batch_clients": len(clients)}))
                else:
                    hs = self._hidden_batch(texts, key[1], key[2] if op == "hidden" else -1)
                    for cid, rid, a, b in spans:
                        part = hs[a:b]
                        T = max(v.shape[0] for v in part)
                        h = torch.zeros(len(part), T, self.sem_dim, dtype=part[0].dtype)
                        m = torch.zeros(len(part), T, dtype=torch.bool)
                        for i, v in enumerate(part):
                            h[i, :v.shape[0]] = v
                            m[i, :v.shape[0]] = True
                        if op == "hidden":
                            res = {"h": h, "mask": m}
                        else:
                            mf = m.float().unsqueeze(-1)
                            res = {"e": (h.float() * mf).sum(1) / mf.sum(1).clamp(min=1)}
                        res.update({"version": self.encoder_version, "batch_texts": len(texts),
                                    "batch_clients": len(clients)})
                        replies.append((cid, rid, res))
            except Exception as e:                      # noqa: BLE001 — report to the clients, keep serving
                for cid, rid, _, _ in spans:
                    replies.append((cid, rid, {"error": f"{type(e).__name__}: {e}"}))
        return replies

    def info(self) -> dict:
        return {"model": self.model_name, "sem_dim": self.sem_dim, "encoder_version": self.encoder_version,
                "decoder_version": self.decoder_version, "lora_version": self.lora_version,
                "stats": dict(self.stats), "batch_texts_hist": {str(k): v for k, v in self.batch_texts_hist.items()},
                "batch_clients_hist": {str(k): v for k, v in self.batch_clients_hist.items()},
                "max_clients_in_batch": self.max_clients_in_batch,
                "cache": {"items": len(self.cache.d), "bytes": self.cache.bytes, "hits": self.cache.hits,
                          "misses": self.cache.misses}, "load_s": self.load_s, "device": str(self.device)}

    def collect(self, req_q, first_timeout_s: float = 0.1) -> list:
        """Block up to ``first_timeout_s`` for one request, then batch more for ``max_wait_ms``."""
        try:
            first = req_q.get(timeout=first_timeout_s)
        except _queue.Empty:
            return []
        reqs = [first]
        n = self._n_texts(first)
        deadline = time.perf_counter() + self.max_wait_ms / 1000.0
        while n < self.max_batch_texts:
            left = deadline - time.perf_counter()
            if left <= 0:
                break
            try:
                r = req_q.get(timeout=left)
            except _queue.Empty:
                break
            reqs.append(r)
            n += self._n_texts(r)
        return reqs

    @staticmethod
    def _n_texts(r) -> int:
        pl = r[3] or {}
        return len(pl.get("texts", pl.get("prompts", [])) or [1])

    def serve(self, req_q, resp_qs: dict, stop_event, status_path: Optional[str] = None,
              compute_path: Optional[str] = None, status_every_s: float = 1.0) -> None:
        last = 0.0
        while not stop_event.is_set():
            reqs = self.collect(req_q)
            if reqs:
                for cid, rid, res in self.handle_batch(reqs):
                    q = resp_qs.get(cid)
                    if q is not None:
                        q.put((rid, res))
            if status_path and time.time() - last > status_every_s:
                self.write_status(status_path, compute_path, "ready")
                last = time.time()
        if status_path:
            self.write_status(status_path, compute_path, "finished")

    def write_status(self, status_path: str, compute_path: Optional[str], state: str) -> None:
        st = self.info()
        st.update({"state": state, "t": time.time(), "pid": os.getpid()})
        atomic_write_json(status_path, st, fsync=False)
        if compute_path:
            self.meter.write(compute_path)


def semserve_main(spec: dict, req_q, resp_qs: dict, stop_event, sem_dir: str, compute_dir: str) -> None:
    """Process entry point (spawn)."""
    spec = dict(spec)
    lora_root = spec.pop("lora_root", None) or os.path.join(sem_dir, "lora")
    os.makedirs(lora_root, exist_ok=True)
    status_path = os.path.join(os.path.dirname(sem_dir.rstrip("/")), "status", "semserve.json")
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    srv = SemanticServer(lora_root=lora_root, **spec)
    compute_path = os.path.join(compute_dir, f"semserve.{os.getpid()}.json")
    srv.write_status(status_path, compute_path, "ready")
    srv.serve(req_q, resp_qs, stop_event, status_path, compute_path)


class SemanticClient:
    """Client side; implements the contract's `SemanticEncoder` protocol (``sem_dim``, ``version``, ``hidden``,
    ``embed``) plus ``generate``. One client per process; requests are synchronous unless `submit`/`result` are
    used to pipeline several."""

    def __init__(self, req_q, resp_q, client_id, timeout_s: float = 300.0):
        self.req_q, self.resp_q, self.client_id = req_q, resp_q, client_id
        self.timeout_s = timeout_s
        self._n = 0
        self._done: dict = {}
        self._info: Optional[dict] = None
        self.version: str = ""
        self.decoder_version: str = ""
        self.sem_dim: int = 0
        self.tokens_prompt = 0

    def submit(self, op: str, payload: dict) -> str:
        self._n += 1
        rid = f"{self.client_id}:{os.getpid()}:{self._n}"
        self.req_q.put((op, rid, self.client_id, payload))
        return rid

    def result(self, rid: str) -> dict:
        t0 = time.time()
        while rid not in self._done:
            left = self.timeout_s - (time.time() - t0)
            if left <= 0:
                raise TimeoutError(f"semantic server did not answer {rid} in {self.timeout_s}s")
            try:
                got, res = self.resp_q.get(timeout=left)
            except _queue.Empty:
                continue
            if not str(got).startswith(f"{self.client_id}:{os.getpid()}:"):
                continue                     # reply addressed to a previous incarnation of this client
            self._done[got] = res
        res = self._done.pop(rid)
        if "error" in res:
            raise RuntimeError(f"semantic server: {res['error']}")
        return res

    def call(self, op: str, payload: dict) -> dict:
        return self.result(self.submit(op, payload))

    def info(self) -> dict:
        self._info = self.call("info", {})
        self.sem_dim = self._info["sem_dim"]
        self.version = self._info["encoder_version"]
        self.decoder_version = self._info["decoder_version"]
        return self._info

    def hidden(self, texts: list, max_tokens: int = 512, layer: int = -1) -> tuple:
        r = self.call("hidden", {"texts": list(texts), "max_tokens": max_tokens, "layer": layer})
        self.version = r["version"]
        return r["h"], r["mask"]

    def embed(self, texts: list, max_tokens: int = 512):
        r = self.call("embed", {"texts": list(texts), "max_tokens": max_tokens})
        self.version = r["version"]
        return r["e"]

    def generate(self, prompts: list, max_new_tokens: int = 32, greedy: bool = True, temperature: float = 1.0,
                 seed: Optional[int] = None, prefix_embeds: Optional[list] = None) -> dict:
        pl = {"prompts": list(prompts), "max_new_tokens": max_new_tokens, "greedy": greedy,
              "temperature": temperature, "seed": seed}
        if prefix_embeds is not None:
            pl["prefix_embeds"] = [p.detach().cpu() for p in prefix_embeds]
        r = self.call("generate", pl)
        self.decoder_version = r["version"]
        return r
