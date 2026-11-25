"""Candidates of the SP-3.2b benchmark and how each one answers an item.

* `LMCandidate`: a causal language model behind a `SemanticService` (the same code path the participants use).
  Multiple-choice items are answered by option scoring (``mean`` or ``sum`` log probability of the option text, or
  ``label``: the options are lettered and the letters are scored); the method is chosen per model and capability on
  the dev split only. Generative items use bounded greedy generation.
* `EncoderCandidate`: a task-specific encoder: the pretrained sentence encoder all-MiniLM-L6-v2 (22.7M parameters,
  frozen) with a small scoring head trained per capability on the train split (early stopping on dev). It can answer
  multiple-choice items only; it cannot generate text or programs.
* `NgramCandidate`: the same kind of head on a randomly initialised hashed n-gram embedding (about 2M parameters)
  trained from scratch on the train split - what a small network learns from the task data alone (MASTER_SPEC 3.2:
  do not assume a randomly initialised network has language understanding).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

LETTERS = "ABCDEFGHIJKLMNOP"


def _relabel(item: dict) -> dict:
    """The item's prompt with lettered options (for ``label`` scoring)."""
    user = item["messages"][1]["content"]
    opts = item["options"]
    if "\nOptions:\n" in user:
        head = user.split("\nOptions:\n")[0]
        lines = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(opts))
        user = f"{head}\nOptions:\n{lines}\nAnswer with the letter of the correct option only."
    else:   # tool-choice prompts list the tools themselves; letter them in the same order
        head, _, tail = user.partition("\n\nTools you can call now:\n")
        tl = tail.split("\n\n")[0].split("\n")
        lines = "\n".join(f"{LETTERS[i]}) {l[2:] if l.startswith('- ') else l}" for i, l in enumerate(tl))
        user = f"{head}\n\nTools you can call now:\n{lines}\n\nWhich tool do you call next? Answer with the letter only."
    return {"messages": [item["messages"][0], {"role": "user", "content": user}],
            "choices": [LETTERS[i] for i in range(len(opts))], "normalize": "sum",
            "prefix_messages": item.get("prefix_messages")}


class LMCandidate:
    kind = "language_model"

    def __init__(self, model_id: str, device: str = "auto", max_batch: int = 16, max_context: int = 3072):
        from ..service import SemanticService
        self.model_id = model_id
        self.svc = SemanticService(model_id, device=device, max_batch=max_batch, max_new_tokens=320,
                                   max_context=max_context)
        self.name = model_id

    def info(self) -> dict:
        i = self.svc.info()
        return {"name": self.name, "kind": self.kind, "params": i["params"], "revision": i["revision"],
                "weights_hash": i["weights_hash"], "device": i["device"], "dtype": i["dtype"]}

    def mc(self, items: list[dict], method: str = "mean") -> tuple[list[int], float]:
        if not items:
            return [], 0.0
        if method == "label":
            reqs = [_relabel(it) for it in items]
        else:
            reqs = [{"messages": it["messages"], "choices": it["options"], "normalize": method,
                     "prefix_messages": it.get("prefix_messages")} for it in items]
        t0 = time.time()
        out = self.svc.choose(None, reqs)
        return [int(o["choice"]) for o in out], time.time() - t0

    def gen(self, items: list[dict]) -> tuple[list[str], float, int]:
        if not items:
            return [], 0.0, 0
        reqs = [{"messages": it["messages"], "max_new_tokens": it.get("max_new_tokens", 128), "temperature": 0.0,
                 "prefix_messages": it.get("prefix_messages")} for it in items]
        t0 = time.time()
        out = self.svc.generate(None, reqs)
        return [o["text"] for o in out], time.time() - t0, sum(o["gen_tokens"] for o in out)

    def close(self) -> None:
        self.svc.close()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


# ====================================================================== trained heads
class _Head:
    """Scores (query, option) pairs from their embeddings; trained with a softmax over each item's options."""

    def __init__(self, dim: int, seed: int, hidden: int = 256):
        import torch
        g = torch.Generator().manual_seed(seed)
        self.torch = torch
        self.net = torch.nn.Sequential(torch.nn.Linear(4 * dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
        for p in self.net.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * (0.02 if p.dim() > 1 else 0.0))

    def params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def logits(self, q, o):
        torch = self.torch
        x = torch.cat([q.expand_as(o), o, q * o, (q - o).abs()], dim=-1)
        return self.net(x).squeeze(-1)


def train_head(embed_q, embed_o, train: list[dict], dev: list[dict], dim: int, seed: int = 0, epochs: int = 40,
               extra_params=None, lr: float = 1e-3):
    """Train a `_Head` (and optionally an embedding module passed as `extra_params`) on train items with early
    stopping on dev accuracy. `embed_q(items)` -> list of [1, dim] tensors, `embed_o(items)` -> list of [k, dim]."""
    import torch
    head = _Head(dim, seed)
    params = list(head.net.parameters()) + (list(extra_params) if extra_params is not None else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    best, best_state, best_ep = -1.0, None, 0

    def acc(items):
        if not items:
            return 0.0
        with torch.no_grad():
            qs, os_ = embed_q(items), embed_o(items)
            return float(np.mean([int(torch.argmax(head.logits(q, o))) == it["answer"]
                                  for q, o, it in zip(qs, os_, items)]))
    for ep in range(epochs):
        order = rng.permutation(len(train))
        for s in range(0, len(order), 32):
            batch = [train[int(i)] for i in order[s:s + 32]]
            qs, os_ = embed_q(batch), embed_o(batch)
            loss = sum(torch.nn.functional.cross_entropy(head.logits(q, o).unsqueeze(0),
                                                         torch.tensor([it["answer"]])) for q, o, it in zip(qs, os_, batch))
            opt.zero_grad()
            (loss / len(batch)).backward()
            opt.step()
        a = acc(dev) if dev else acc(train)
        if a > best:
            best, best_ep = a, ep
            best_state = ([p.detach().clone() for p in head.net.parameters()],
                          [p.detach().clone() for p in extra_params] if extra_params is not None else None)
    with torch.no_grad():
        for p, v in zip(head.net.parameters(), best_state[0]):
            p.copy_(v)
        if extra_params is not None:
            for p, v in zip(extra_params, best_state[1]):
                p.copy_(v)
    return head, {"dev_acc": best, "epoch": best_ep}


class EncoderCandidate:
    kind = "task_specific_encoder"
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device: str = "cpu", seed: int = 0):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from ..service import local_revision
        self.torch, self.device, self.seed = torch, device, seed
        self.revision = local_revision(self.MODEL)
        self.tok = AutoTokenizer.from_pretrained(self.MODEL, revision=self.revision, local_files_only=True)
        self.enc = AutoModel.from_pretrained(self.MODEL, revision=self.revision, local_files_only=True).to(device).eval()
        self.enc_params = sum(p.numel() for p in self.enc.parameters())
        self.heads: dict[str, _Head] = {}
        self.train_info: dict[str, dict] = {}
        self._cache: dict[str, np.ndarray] = {}
        self.name = f"minilm-l6+head"

    def _embed(self, texts: list[str]):
        torch = self.torch
        out = [None] * len(texts)
        todo = [i for i, t in enumerate(texts) if hashlib.sha1(t.encode()).hexdigest() not in self._cache]
        for s in range(0, len(todo), 64):
            idx = todo[s:s + 64]
            b = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=512, return_tensors="pt")
            b = {k: v.to(self.device) for k, v in b.items()}
            with torch.no_grad():
                h = self.enc(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu().numpy()
            for j, i in enumerate(idx):
                self._cache[hashlib.sha1(texts[i].encode()).hexdigest()] = e[j]
        for i, t in enumerate(texts):
            out[i] = self._cache[hashlib.sha1(t.encode()).hexdigest()]
        return np.stack(out)

    @staticmethod
    def _qtext(it: dict) -> str:
        q = it["query"]
        tail = q.rsplit("\n", 1)[-1]
        return f"{tail}\n{q}"          # the question first, so truncation never cuts it

    def _eq(self, items):
        e = self._embed([self._qtext(it) for it in items])
        return [self.torch.tensor(x).unsqueeze(0) for x in e]

    def _eo(self, items):
        out = []
        for it in items:
            texts = it.get("option_texts") or it["options"]
            out.append(self.torch.tensor(self._embed(texts)))
        return out

    def fit(self, cap: str, train: list[dict], dev: list[dict]) -> dict:
        tr = [i for i in train if i["format"] == "mc"]
        dv = [i for i in dev if i["format"] == "mc"]
        if not tr:
            return {"trained": False}
        head, info = train_head(self._eq, self._eo, tr, dv, dim=self.enc.config.hidden_size, seed=self.seed)
        self.heads[cap] = head
        self.train_info[cap] = {"train_items": len(tr), **info, "head_params": head.params()}
        return self.train_info[cap]

    def info(self) -> dict:
        hp = max((h.params() for h in self.heads.values()), default=0)
        return {"name": self.name, "kind": self.kind, "params": self.enc_params + hp, "encoder_params": self.enc_params,
                "head_params_per_capability": hp, "revision": self.revision, "device": self.device}

    def mc(self, items: list[dict], cap: str) -> tuple[list[int], float]:
        head = self.heads.get(cap)
        if head is None or not items:
            return [], 0.0
        t0 = time.time()
        with self.torch.no_grad():
            preds = [int(self.torch.argmax(head.logits(q, o))) for q, o in zip(self._eq(items), self._eo(items))]
        return preds, time.time() - t0

    def close(self) -> None:
        self._cache.clear()


class NgramCandidate:
    """Hashed word uni/bigram embedding bag (2^16 x 32, random initialisation) with the same head, trained from
    scratch on the train split of each capability."""
    kind = "from_scratch_small_network"

    def __init__(self, seed: int = 0, buckets: int = 1 << 16, dim: int = 32):
        import torch
        self.torch, self.seed, self.buckets, self.dim = torch, seed, buckets, dim
        self.models: dict[str, tuple] = {}
        self.train_info: dict[str, dict] = {}
        self.name = "ngram-bag+head (from scratch)"

    def _ids(self, text: str):
        import re
        w = re.findall(r"[a-z0-9]+", text.lower())
        grams = w + [a + "_" + b for a, b in zip(w, w[1:])]
        return [int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % self.buckets for g in grams] or [0]

    def _bag(self, emb, texts):
        torch = self.torch
        ids, offs = [], []
        for t in texts:
            offs.append(len(ids))
            ids.extend(self._ids(t))
        return emb(torch.tensor(ids), torch.tensor(offs))

    def fit(self, cap: str, train: list[dict], dev: list[dict]) -> dict:
        torch = self.torch
        tr = [i for i in train if i["format"] == "mc"]
        dv = [i for i in dev if i["format"] == "mc"]
        if not tr:
            return {"trained": False}
        torch.manual_seed(self.seed)
        emb = torch.nn.EmbeddingBag(self.buckets, self.dim, mode="mean")
        eq = lambda items: [self._bag(emb, [EncoderCandidate._qtext(it)]) for it in items]
        eo = lambda items: [self._bag(emb, it.get("option_texts") or it["options"]) for it in items]
        head, info = train_head(eq, eo, tr, dv, dim=self.dim, seed=self.seed, extra_params=list(emb.parameters()),
                                lr=3e-3)
        self.models[cap] = (emb, head, eq, eo)
        self.train_info[cap] = {"train_items": len(tr), **info,
                                "params": sum(p.numel() for p in emb.parameters()) + head.params()}
        return self.train_info[cap]

    def info(self) -> dict:
        p = max((v["params"] for v in self.train_info.values()), default=self.buckets * self.dim)
        return {"name": self.name, "kind": self.kind, "params": p}

    def mc(self, items: list[dict], cap: str) -> tuple[list[int], float]:
        m = self.models.get(cap)
        if m is None or not items:
            return [], 0.0
        emb, head, eq, eo = m
        t0 = time.time()
        with self.torch.no_grad():
            preds = [int(self.torch.argmax(head.logits(q, o))) for q, o in zip(eq(items), eo(items))]
        return preds, time.time() - t0

    def close(self) -> None:
        self.models.clear()
