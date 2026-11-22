"""A continually trainable language backbone for the controller's text representations.  [candidate-editable]

Until now every text the controller reads (tool descriptions, the goal, documents, error feedback) entered it as a
FROZEN vector: an embedding computed once by a language model whose weights never change. `TextBackbone` replaces
that vector by the output of a language model that is part of the learner: a pretrained causal LM (default
Qwen3-0.6B) whose own weights receive the gradients of the policy, critic and world-model losses.

    text --> "This text: "<text>" means in one word: "" --> transformer blocks (trainable) --> state of the last token
         --> minus a fixed centre --> fixed random projection to 512 dims --> L2 normalisation --> item embedding

The embedding function is the one the frozen language-model service uses (`llm/base.py:HFModel.embed`: prompted
last-token state four layers below the output), so a backbone with ``mode: none`` is a frozen embedder and the
other modes start from exactly that function and then move away from it by gradient descent.

Which of its parameters are plastic is a setting, so that the amount of backbone adaptation can be compared:

    none      nothing (frozen baseline; Architecture A)
    lora      rank-r adapters on the attention query/value projections (parameter-efficient baseline)
    top:k     full-parameter updates of the last k transformer blocks that feed the embedding
    full      full-parameter updates of every block that feeds the embedding, and of the token embeddings

Gradients reach the backbone through the embedding table with a gradient cache (`LiveTable`): embeddings of the
texts a minibatch uses are computed without a graph and made a leaf; the controller's loss is back-propagated to
that leaf; then the texts are re-encoded in chunks with a graph and each chunk receives its slice of the leaf's
gradient. The parameter gradients are exact and memory is bounded by the chunk size.

The 14B generator that writes code and free text for the agent is NOT this backbone and stays frozen (see
docs/CONTINUAL.md for why); nothing here claims otherwise.
"""
from __future__ import annotations

import math
import os
import threading

import numpy as np
import torch
import torch.nn as nn

EMB_DIM = 512
EMB_TEMPLATE = 'This text: "{t}" means in one word: "'
EMB_DROP_TOP = 3            # the embedding is read below the top 3 blocks (hidden_states[-4] of the full model)
PROJ_SEED = 20261121


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x with W frozen; B starts at zero, so the adapted layer starts as the original."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features, dtype=torch.float32) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        self.scale = alpha / r

    def forward(self, x):
        y = self.base(x)
        return y + (x.to(self.A.dtype) @ self.A.t() @ self.B.t()).to(y.dtype) * self.scale


class TextBackbone(nn.Module):
    def __init__(self, model: str = "Qwen/Qwen3-0.6B", mode: str = "none", device: str = "cuda", embed_tokens: int = 256,
                 lora_rank: int = 8, chunk_tokens: int = 6000, lm=None, tok=None):
        super().__init__()
        self.name, self.mode, self.device_s = model, mode, device
        self.embed_tokens, self.chunk_tokens = int(embed_tokens), int(chunk_tokens)
        if lm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
            full = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32, local_files_only=True)
            lm = full.model
            del full
        self.tok = tok
        n = len(lm.layers)
        keep = max(1, n - EMB_DROP_TOP)
        lm.layers = nn.ModuleList(list(lm.layers)[:keep])       # blocks above the read-out never influence it
        lm.config.num_hidden_layers = keep
        lm.norm = nn.Identity()                                  # the read-out is the un-normalised block output
        self.lm = lm
        self.n_blocks = keep
        hid = int(lm.config.hidden_size)
        g = torch.Generator().manual_seed(PROJ_SEED)
        self.register_buffer("proj", torch.randn(hid, EMB_DIM, generator=g) / np.sqrt(EMB_DIM))
        self.register_buffer("centre", torch.zeros(1, hid))
        self.version = 0                    # incremented whenever the weights (hence all embeddings) change
        self.lock = threading.Lock()
        self._set_mode(mode, lora_rank)
        self.to(device)
        self.amp = device.startswith("cuda")
        with torch.no_grad():
            from ..llm.base import CALIBRATION_TEXTS
            self.centre.copy_(self._raw(list(CALIBRATION_TEXTS)).mean(0, keepdim=True))
        self.eval()

    # ------------------------------------------------------------------ which parameters learn
    def _set_mode(self, mode: str, lora_rank: int) -> None:
        for p in self.lm.parameters():
            p.requires_grad_(False)
        if mode == "none":
            return
        if mode == "lora":
            for blk in self.lm.layers:
                att = blk.self_attn
                att.q_proj = LoRALinear(att.q_proj, lora_rank)
                att.v_proj = LoRALinear(att.v_proj, lora_rank)
        elif mode.startswith("top:"):
            for blk in list(self.lm.layers)[-int(mode.split(":")[1]):]:
                for p in blk.parameters():
                    p.requires_grad_(True)
        elif mode == "full":
            for p in self.lm.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError(f"unknown backbone mode {mode!r} (none | lora | top:k | full)")

    @property
    def trainable(self) -> bool:
        return self.mode != "none"

    def n_params(self) -> dict:
        return {"model": self.name, "mode": self.mode, "blocks_used": self.n_blocks,
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "total": sum(p.numel() for p in self.lm.parameters())}

    def trainable_state(self) -> dict:
        """Only what learning can change (the rest is the pretrained model, identified by its name)."""
        req = {k for k, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in req}

    def load_trainable_state(self, s: dict) -> None:
        own = self.state_dict()
        for k, v in s.items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))
        self.version += 1

    # ------------------------------------------------------------------ embedding
    def _encode(self, texts: list[str]) -> torch.Tensor:
        batch = [EMB_TEMPLATE.replace("{t}", (t if t else " ")[:1500]) for t in texts]
        enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.embed_tokens + 16,
                       padding_side="left")
        enc = {k: v.to(self.proj.device) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            hs = self.lm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        return hs[:, -1].float()

    def _raw(self, texts: list[str], bs: int = 32) -> torch.Tensor:
        out = [self._encode(texts[i:i + bs]) for i in range(0, len(texts), bs)]
        return torch.cat(out) if out else torch.zeros(0, self.proj.shape[0], device=self.proj.device)

    def _finish(self, raw: torch.Tensor) -> torch.Tensor:
        v = (raw - self.centre) @ self.proj
        return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def chunks(self, texts: list[str]) -> list[list[int]]:
        """Indices grouped so that a chunk holds about `chunk_tokens` (padded) tokens; similar lengths together."""
        n_tok = [min(self.embed_tokens + 16, 12 + len(t[:1500]) // 3) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: n_tok[i])
        out, cur, width = [], [], 0
        for i in order:
            w = max(width, n_tok[i])
            if cur and w * (len(cur) + 1) > self.chunk_tokens:
                out.append(cur)
                cur, w = [], n_tok[i]
            cur.append(i)
            width = w
        if cur:
            out.append(cur)
        return out

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        """Embeddings [n, 512] without a graph (acting, and the forward half of the gradient cache)."""
        if not texts:
            return torch.zeros(0, EMB_DIM, device=self.proj.device)
        out = torch.zeros(len(texts), EMB_DIM, device=self.proj.device)
        with self.lock:
            for idx in self.chunks(texts):
                out[idx] = self._finish(self._encode([texts[i] for i in idx]))
        return out

    def embed_np(self, texts: list[str]) -> np.ndarray:
        return self.embed(list(texts)).cpu().numpy().astype(np.float32)

    def backprop(self, texts: list[str], grad: torch.Tensor) -> None:
        """Second half of the gradient cache: accumulate d(loss)/d(parameters) given d(loss)/d(embeddings)."""
        if not self.trainable or not texts:
            return
        with self.lock:
            for idx in self.chunks(texts):
                e = self._finish(self._encode([texts[i] for i in idx]))
                e.backward(grad[idx].to(e.dtype))


class LiveTable:
    """Makes the controller's text-embedding table a differentiable function of the backbone for one minibatch."""

    ROWS = ("ctx_row", "head_row", "par_row", "opt_row")

    def __init__(self, backbone: TextBackbone, store, texts: dict):
        self.bb, self.store, self.texts = backbone, store, texts
        self._pending = None
        self.n_flush = 0

    def attach(self, model, t: dict) -> None:
        dev = t["valid"].device
        base = self.store.tensor(dev)
        if not self.bb.trainable:
            model.set_table(base)
            return
        rows = torch.unique(torch.cat([t[k].reshape(-1) for k in self.ROWS]))
        rows = rows[rows != 0]
        inv = getattr(self.store, "_row2key", None)
        if inv is None or len(inv) != self.store.n:
            inv = self.store._row2key = {r: k for k, r in self.store.key2row.items()}
        keep = [int(r) for r in rows.tolist() if inv.get(int(r)) in self.texts]
        if not keep:
            model.set_table(base)
            return
        texts = [self.texts[inv[r]] for r in keep]
        E = self.bb.embed(texts).to(base.dtype)
        idx = torch.tensor(keep, device=dev)
        if torch.is_grad_enabled():
            E.requires_grad_(True)
            self._pending = (texts, E)
        model.set_table(base.index_copy(0, idx, E))

    def flush(self) -> None:
        """Call after ``loss.backward()``: pushes the table's gradient into the backbone's parameters."""
        if self._pending is None:
            return
        texts, E = self._pending
        self._pending = None
        if E.grad is not None:
            self.bb.backprop(texts, E.grad)
            self.n_flush += 1

    def refresh_store(self) -> int:
        """After the backbone's weights changed: recompute the stored vector of every known text."""
        self.bb.version += 1
        keys = [k for k in self.store.key2row if k in self.texts]
        if keys:
            vec = self.bb.embed_np([self.texts[k] for k in keys])
            for k, v in zip(keys, vec):
                self.store.mat[self.store.key2row[k]] = v
            self.store._tensor = None
        return len(keys)


# ---------------------------------------------------------------------------------------------- service for workers
class _Req:
    __slots__ = ("texts", "out", "err", "done")

    def __init__(self, texts):
        self.texts, self.out, self.err, self.done = list(texts), None, None, threading.Event()


class EmbedService:
    """Serves embeddings of new texts to rollout workers from the backbone's CURRENT weights (unix socket).

    Requests that arrive together are answered by ONE forward pass (a batcher thread drains the queue), and vectors
    are cached per weight version, so a dozen workers asking for the same tool descriptions cost one computation."""

    AUTH = b"superlearner-backbone"

    def __init__(self, backbone: TextBackbone, path: str, max_wait_s: float = 0.004, max_batch: int = 64):
        import queue
        from multiprocessing.connection import Listener
        if os.path.exists(path):
            os.unlink(path)
        self.bb, self.path = backbone, path
        self.listener = Listener(path, family="AF_UNIX", authkey=self.AUTH)
        self.n_texts = self.n_computed = self.n_batches = 0
        self.max_wait_s, self.max_batch = max_wait_s, max_batch
        self.q: "queue.Queue[_Req]" = queue.Queue()
        self.cache: dict[str, np.ndarray] = {}
        self.cache_version = backbone.version
        self._stop = False
        threading.Thread(target=self._accept, daemon=True).start()
        threading.Thread(target=self._batcher, daemon=True).start()

    def _accept(self) -> None:
        while not self._stop:
            try:
                conn = self.listener.accept()
            except (OSError, EOFError):
                break
            except Exception:  # noqa: BLE001 - a failed handshake must not stop the service
                continue
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _batcher(self) -> None:
        import queue
        import time
        while not self._stop:
            try:
                batch = [self.q.get(timeout=0.5)]
            except queue.Empty:
                continue
            time.sleep(self.max_wait_s)                       # let the requests of other workers arrive
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            try:
                if self.cache_version != self.bb.version or len(self.cache) > 200_000:
                    self.cache, self.cache_version = {}, self.bb.version
                need = list(dict.fromkeys(t for r in batch for t in r.texts if t not in self.cache))
                if need:
                    version = self.bb.version
                    vecs = self.bb.embed_np(need)
                    if version == self.bb.version:
                        self.cache.update(zip(need, vecs))
                    fresh = dict(zip(need, vecs))
                else:
                    fresh = {}
                self.n_computed += len(need)
                self.n_batches += 1
                for r in batch:
                    r.out = np.stack([fresh[t] if t in fresh else self.cache[t] for t in r.texts]) if r.texts \
                        else np.zeros((0, EMB_DIM), np.float32)
            except Exception as e:  # noqa: BLE001
                for r in batch:
                    r.err = f"{type(e).__name__}: {e}"
            for r in batch:
                r.done.set()

    def _serve(self, conn) -> None:
        while True:
            try:
                op, payload = conn.recv()
            except (EOFError, OSError):
                break
            try:
                if op == "embed":
                    self.n_texts += len(payload)
                    r = _Req(payload)
                    self.q.put(r)
                    r.done.wait()
                    conn.send(("err", r.err) if r.err else ("ok", r.out))
                elif op == "info":
                    conn.send(("ok", dict(self.bb.n_params(), version=self.bb.version, texts=self.n_texts,
                                          computed=self.n_computed, batches=self.n_batches)))
                else:
                    conn.send(("err", f"unknown op {op}"))
            except Exception as e:  # noqa: BLE001
                try:
                    conn.send(("err", f"{type(e).__name__}: {e}"))
                except (OSError, BrokenPipeError):
                    break
        conn.close()

    def close(self) -> None:
        self._stop = True
        try:
            self.listener.close()
        except OSError:
            pass
        if os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except OSError:
                pass
