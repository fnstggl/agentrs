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

    MAX_BATCH = 8

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
                key = self._gkey(first[0])
                keep = []
                while len(group) < self.MAX_BATCH:
                    try:
                        j = self.jobs.get_nowait()
                    except queue.Empty:
                        break
                    (group if (j[0].get("op") == "generate" and self._gkey(j[0]) == key) else keep).append(j)
                for j in keep:
                    self.jobs.put(j)
            try:
                if first[0]["op"] == "generate":
                    n, v, th = self._gkey(first[0])
                    gs = self.llm.generate_many([(r["messages"], n, v, th) for r, _ in group])
                    for (_, reply), g in zip(group, gs):
                        reply({"text": g.text, "prompt_tokens": g.prompt_tokens, "gen_tokens": g.gen_tokens,
                               "wall_s": g.wall_s, "cached": g.cached})
                    self.batches.append(len(group))
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

    def _ask(self, req: dict) -> dict:
        self.stats["server_requests"] += 1
        c = self._c()
        c.send(req)
        if not c.poll(self.timeout_s):
            raise TimeoutError(f"llm server did not answer within {self.timeout_s}s")
        out = c.recv()
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


def connect(spec: dict | None) -> LLM:
    """Build the LLM client a run asked for. ``{"kind": "remote", "socket": ..., "cache": ...}``,
    ``{"kind": "hf", "model": ..., "cache": ...}`` (in-process), ``{"kind": "none"}`` (hash embeddings, no generation)."""
    spec = spec or {"kind": "none"}
    kind = spec.get("kind", "none")
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
