"""One process owns the frozen GPU model; every other process (rollout workers, evaluation, candidate training runs)
uses it through a unix socket.

    python -m superlearner.llm.server --model Qwen/Qwen3-14B --socket runs/llm.sock --cache runs/llm_cache.sqlite

Protocol (`multiprocessing.connection`, pickled dicts):
    {"op": "generate", "messages": [...], "max_new_tokens": n, "variant": v} -> {"text", "prompt_tokens", ...}
    {"op": "embed", "texts": [...]}                                          -> {"vecs": float16 array}
    {"op": "info"}                                                           -> {"model", "fingerprint", "stats"}

Every answer is written to the shared sqlite cache; clients read the cache directly (read-only) and only contact
the server on a miss. The socket is the ONLY channel a sandboxed candidate run has to the model: it can request
text and embeddings and nothing else.
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from multiprocessing.connection import Client, Listener

import numpy as np

from .base import LLM, Cache, CachedLLM, GenResult, HashEmbedder, prompt_key, text_key

AUTHKEY = b"superlearner-llm"


class Server:
    def __init__(self, model: LLM, socket_path: str, cache_path: str):
        self.llm = CachedLLM(model, cache_path)
        self.socket_path = socket_path
        self.jobs: queue.Queue = queue.Queue()
        self.fast: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.t0 = time.time()
        self.batches: list[int] = []

    # A batch is limited both by its size and by its estimated tokens: the attention cache of a 14B model costs
    # about 0.16 MB per token and sequence, so eight 7,000-token prompts would need 9 GB on a GPU that the training
    # runs share. Cached GPU blocks are released after every batch for the same reason.
    MAX_BATCH = int(os.environ.get("SUPERLEARNER_LLM_MAX_BATCH", 8))
    MAX_BATCH_TOKENS = int(os.environ.get("SUPERLEARNER_LLM_MAX_BATCH_TOKENS", 14000))

    @staticmethod
    def _est_tokens(req: dict) -> int:
        chars = sum(len(str(m.get("content", ""))) for m in req.get("messages", []))
        return chars // 3 + int(req.get("max_new_tokens", 512)) + (1000 if req.get("think") else 0)

    @staticmethod
    def _release() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - a test double has no GPU
            pass

    def _generate_group(self, group: list) -> list:
        """Batched generation; if the GPU is out of memory (other processes share it), release and go one by one."""
        try:
            return self.llm.generate_many([(r["messages"],) + self._gkey(r) for r, _ in group])
        except Exception as e:  # noqa: BLE001
            if "out of memory" not in str(e).lower() or len(group) == 1:
                raise
            self._release()
            self.oom_splits = getattr(self, "oom_splits", 0) + 1
            out = []
            for r, _ in group:
                for attempt in range(6):
                    try:
                        out += self.llm.generate_many([(r["messages"],) + self._gkey(r)])
                        break
                    except Exception as e2:  # noqa: BLE001
                        if "out of memory" not in str(e2).lower() or attempt == 5:
                            raise
                        self._release()
                        time.sleep(10)
            return out

    def _gpu_loop(self) -> None:
        """One thread owns the model. Waiting generation requests with the same decoding settings are answered
        in one batched forward pass; everything else is handled one request at a time."""
        while not self.stop.is_set():
            try:
                first = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            group = [first]
            if first[0].get("op") == "generate":
                time.sleep(0.05)                       # let concurrent requests arrive
                key = self._bkey(first[0])
                tokens = self._est_tokens(first[0])
                keep = []
                while len(group) < self.MAX_BATCH:
                    try:
                        j = self.jobs.get_nowait()
                    except queue.Empty:
                        break
                    fits = tokens + self._est_tokens(j[0]) <= self.MAX_BATCH_TOKENS
                    if j[0].get("op") == "generate" and self._bkey(j[0]) == key and fits:
                        group.append(j)
                        tokens += self._est_tokens(j[0])
                    else:
                        keep.append(j)
                for j in keep:
                    self.jobs.put(j)
            try:
                if first[0]["op"] == "generate":
                    gs = self._generate_group(group)
                    for (_, reply), g in zip(group, gs):
                        reply({"text": g.text, "prompt_tokens": g.prompt_tokens, "gen_tokens": g.gen_tokens,
                               "wall_s": g.wall_s, "cached": g.cached})
                    self.batches.append(len(group))
                    self._release()
                    continue
                req, reply = first
                if req["op"] == "embed":       # (normally served by the fast thread; kept for completeness)
                    reply({"vecs": self.llm.embed(list(req["texts"])).astype(np.float16)})
                elif req["op"] == "info":
                    reply({"model": self.llm.name, "fingerprint": self.llm.fingerprint(), "stats": self.llm.usage(),
                           "emb_ns": self.llm.name + "#" + getattr(self.llm.inner, "EMB_TAG", "v1"),
                           "uptime_s": time.time() - self.t0, "test_double": self.llm.is_test_double,
                           "mean_batch": float(np.mean(self.batches)) if self.batches else 0.0})
                else:
                    reply({"error": f"unknown op {req.get('op')!r}"})
            except Exception as e:  # noqa: BLE001 - a bad request must not kill the server
                for _, reply in group:
                    reply({"error": f"{type(e).__name__}: {e}"})

    def _fast_loop(self) -> None:
        """Embeddings and status requests, served while a (long) generation may be running."""
        while not self.stop.is_set():
            try:
                req, reply = self.fast.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if req["op"] == "embed":
                    reply({"vecs": self.llm.embed(list(req["texts"])).astype(np.float16)})
                else:
                    reply({"model": self.llm.name, "fingerprint": self.llm.fingerprint(), "stats": self.llm.usage(),
                           "emb_ns": self.llm.name + "#" + getattr(self.llm.inner, "EMB_TAG", "v1"),
                           "uptime_s": time.time() - self.t0, "test_double": self.llm.is_test_double,
                           "mean_batch": float(np.mean(self.batches)) if self.batches else 0.0})
            except Exception as e:  # noqa: BLE001
                reply({"error": f"{type(e).__name__}: {e}"})

    @staticmethod
    def _gkey(req: dict) -> tuple:
        return int(req.get("max_new_tokens", 512)), int(req.get("variant", 0)), bool(req.get("think", False))

    @classmethod
    def _bkey(cls, req: dict) -> tuple:
        """Which requests may share a forward pass: all greedy ones (each row keeps its own length limit and
        reasoning flag); sampled ones only with identical settings (they share one random stream)."""
        n, v, th = cls._gkey(req)
        if v:
            return (n, v, th)
        # a batch runs as long as its longest row, so short requests (a tool call, a list of file names) are not
        # put behind long ones (a code file, a reasoned manifest): rows are grouped by the size of their limit
        lim = n + (1000 if th else 0)
        return ("greedy", 0 if lim <= 300 else 1 if lim <= 1000 else 2 if lim <= 1500 else 3)

    def _client_loop(self, conn) -> None:
        lock = threading.Lock()

        def reply(msg):
            with lock:
                try:
                    conn.send(msg)
                except (OSError, EOFError, BrokenPipeError):
                    pass
        try:
            while not self.stop.is_set():
                req = conn.recv()
                if not isinstance(req, dict) or req.get("op") == "bye":
                    break
                (self.fast if req.get("op") in ("embed", "info") else self.jobs).put((req, reply))
        except (EOFError, OSError):
            pass
        finally:
            conn.close()

    def serve(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        os.makedirs(os.path.dirname(os.path.abspath(self.socket_path)) or ".", exist_ok=True)
        threading.Thread(target=self._gpu_loop, daemon=True).start()
        threading.Thread(target=self._fast_loop, daemon=True).start()
        with Listener(self.socket_path, family="AF_UNIX", authkey=AUTHKEY) as lst:
            print(f"llm server ready: model={self.llm.name} socket={self.socket_path}", flush=True)
            while not self.stop.is_set():
                try:
                    conn = lst.accept()
                except (OSError, EOFError):
                    continue
                threading.Thread(target=self._client_loop, args=(conn,), daemon=True).start()


class RemoteLLM(LLM):
    """Client: reads the shared cache directly and asks the server only on a miss. Reconnects after `fork`."""

    def __init__(self, socket_path: str, cache_path: str | None = None, name: str | None = None, timeout_s: float = 900.0):
        self.socket_path, self.cache_path, self.timeout_s = socket_path, cache_path, timeout_s
        self._conn = None
        self._pid = None
        self._mem: dict[int, np.ndarray] = {}
        self.cache = Cache(cache_path, readonly=True) if cache_path else None
        self.stats = {"gen_calls": 0, "gen_cache_hits": 0, "prompt_tokens": 0, "gen_tokens": 0, "served_tokens": 0,
                      "embed_texts": 0, "embed_cache_hits": 0, "gen_wall_s": 0.0, "server_requests": 0}
        inf = self.info()
        self.name = name or inf.get("model", "remote")
        self.emb_ns = inf.get("emb_ns", self.name)

    def _c(self):
        if self._conn is None or self._pid != os.getpid():
            self._conn = Client(self.socket_path, family="AF_UNIX", authkey=AUTHKEY)
            self._pid = os.getpid()
            if self.cache_path:
                self.cache = Cache(self.cache_path, readonly=True)
        return self._conn

    RECONNECT_S = float(os.environ.get("SUPERLEARNER_LLM_RECONNECT_S", 600))

    def _ask(self, req: dict) -> dict:
        """One request. Requests are idempotent (answers are cached by content), so when the connection breaks -
        the server was restarted - the client waits for the socket to come back and asks again."""
        self.stats["server_requests"] += 1
        deadline = time.time() + self.RECONNECT_S
        while True:
            try:
                c = self._c()
                c.send(req)
                if not c.poll(self.timeout_s):
                    raise TimeoutError(f"llm server did not answer within {self.timeout_s}s")
                out = c.recv()
                break
            except (EOFError, OSError) as e:                 # includes ConnectionError, BrokenPipeError, FileNotFoundError
                if isinstance(e, TimeoutError) or time.time() > deadline:
                    raise
                try:
                    if self._conn is not None:
                        self._conn.close()
                except Exception:  # noqa: BLE001
                    pass
                self._conn = None
                self.stats["reconnects"] = self.stats.get("reconnects", 0) + 1
                time.sleep(5)
        if "error" in out:
            raise RuntimeError(f"llm server: {out['error']}")
        return out

    def info(self) -> dict:
        return self._ask({"op": "info"})

    def fingerprint(self) -> str:
        return self.info().get("fingerprint", "?")

    def generate(self, messages, max_new_tokens=512, variant=0, think=False):
        self.stats["gen_calls"] += 1
        if self.cache is not None:
            g = self.cache.get_gen(prompt_key(self.name, messages, max_new_tokens, variant, think))
            if g is not None:
                self.stats["gen_cache_hits"] += 1
                self.stats["served_tokens"] += g.gen_tokens
                return g
        o = self._ask({"op": "generate", "messages": messages, "max_new_tokens": max_new_tokens, "variant": variant,
                       "think": bool(think)})
        if o.get("cached"):
            self.stats["gen_cache_hits"] += 1
        else:
            self.stats["prompt_tokens"] += o["prompt_tokens"]
            self.stats["gen_tokens"] += o["gen_tokens"]
            self.stats["gen_wall_s"] += o["wall_s"]
        self.stats["served_tokens"] += o["gen_tokens"]
        return GenResult(o["text"], o["prompt_tokens"], o["gen_tokens"], o["wall_s"], cached=bool(o.get("cached")))

    def embed(self, texts):
        keys = [text_key(t) for t in texts]
        self.stats["embed_texts"] += len(texts)
        need = [k for k in dict.fromkeys(keys) if k not in self._mem]
        if need and self.cache is not None:
            self._mem.update(self.cache.get_emb(self.emb_ns, need))
        miss = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self._mem]
        self.stats["embed_cache_hits"] += len(texts) - len(miss)
        if miss:
            vecs = self._ask({"op": "embed", "texts": [t for _, t in miss]})["vecs"]
            for (k, _), v in zip(miss, vecs):
                self._mem[k] = np.asarray(v, np.float32)
        return np.stack([self._mem[k] for k in keys]) if keys else np.zeros((0, 512), np.float32)

    def usage(self):
        return {"model": self.name, **self.stats}


class EmbedOnly(LLM):
    """Model-free embeddings + no generation (CPU smoke tests)."""
    name = "hash-embedder+null"

    def __init__(self):
        self.emb = HashEmbedder()

    def generate(self, messages, max_new_tokens=512, variant=0, think=False):
        return GenResult("{}", 0, 0)

    def embed(self, texts):
        return self.emb.embed(texts)


class BackboneClient(LLM):
    """Text embeddings from the learner's own (possibly trainable) language backbone, served by the training
    process (`cam/backbone.py:EmbedService`). Vectors are remembered in this process for as long as the backbone's
    weights stay the same: when they change the trainer replaces the worker's embedding store and `clear()` is
    called, so nothing computed with old weights is ever served. Generation, if a run needs it, goes to `gen` (the
    frozen generator's client); without one there is no generation."""

    def __init__(self, socket_path: str, gen: LLM | None = None, trainable: bool = False):
        self.socket_path, self.gen, self.trainable = socket_path, gen, bool(trainable)
        self._conn, self._pid = None, None
        self.name = "backbone"
        self._cache: dict[str, np.ndarray] = {}
        self.stats = {"embed_texts": 0, "embed_requests": 0, "embed_local_hits": 0}

    def clear(self) -> None:
        self._cache = {}

    def _c(self):
        if self._conn is None or self._pid != os.getpid():
            from ..cam.backbone import EmbedService
            self._conn = Client(self.socket_path, family="AF_UNIX", authkey=EmbedService.AUTH)
            self._pid = os.getpid()
        return self._conn

    def embed(self, texts):
        if not len(texts):
            return np.zeros((0, 512), np.float32)
        texts = list(texts)
        need = [t for t in dict.fromkeys(texts) if t not in self._cache]
        self.stats["embed_texts"] += len(texts)
        self.stats["embed_local_hits"] += len(texts) - len(need)
        if need:
            c = self._c()
            c.send(("embed", need))
            st, out = c.recv()
            if st != "ok":
                raise RuntimeError(f"backbone embedding service: {out}")
            self.stats["embed_requests"] += 1
            if len(self._cache) > 100_000:
                self._cache = {}
            self._cache.update(zip(need, out))
            if len(need) == len(texts):
                return out
            fresh = dict(zip(need, out))
            return np.stack([fresh[t] if t in fresh else self._cache[t] for t in texts])
        return np.stack([self._cache[t] for t in texts])

    def generate(self, messages, max_new_tokens=512, variant=0, think=False):
        if self.gen is None:
            return GenResult("{}", 0, 0)
        return self.gen.generate(messages, max_new_tokens, variant, think)

    def generate_many(self, items):
        if self.gen is not None and hasattr(self.gen, "generate_many"):
            return self.gen.generate_many(items)
        return [self.generate(*it) for it in items]

    def usage(self):
        u = dict(self.gen.usage()) if self.gen is not None else {}
        u.update(self.stats)
        return u


def connect(spec: dict | None) -> LLM:
    """Build the LLM client a run asked for. ``{"kind": "remote", "socket": ..., "cache": ...}``,
    ``{"kind": "hf", "model": ..., "cache": ...}`` (in-process), ``{"kind": "none"}`` (hash embeddings, no generation),
    ``{"kind": "backbone", "socket": ..., "gen": {...}}`` (embeddings from the learner's own backbone)."""
    spec = spec or {"kind": "none"}
    kind = spec.get("kind", "none")
    if kind == "backbone":
        gen = spec.get("gen")
        return BackboneClient(spec["socket"], connect(gen) if gen and gen.get("kind", "none") != "none" else None,
                              bool(spec.get("trainable")))
    if kind == "remote":
        return RemoteLLM(spec["socket"], spec.get("cache"))
    if kind == "hf":
        from .base import HFModel
        return CachedLLM(HFModel(spec.get("model", "Qwen/Qwen3-14B"), spec.get("device", "cuda"),
                                 int(spec.get("max_context", 16384))), spec.get("cache"))
    if kind == "none":
        return EmbedOnly()
    raise ValueError(f"unknown llm kind {kind!r}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--socket", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--max-context", type=int, default=16384)
    a = ap.parse_args(argv)
    from .base import HFModel
    Server(HFModel(a.model, a.device, a.max_context), a.socket, a.cache).serve()


if __name__ == "__main__":
    main()
