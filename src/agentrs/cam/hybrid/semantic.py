"""Frozen pretrained semantic foundation for CAM (no LoRA, no gradients, never updated).

``FrozenSemanticEncoder`` turns natural-language observations / goals into fixed-size vectors from a
frozen pretrained language model's hidden states (chosen layer(s), mask-aware mean pooling). One
instance per training process serves every environment worker in batched calls; an LRU cache keyed by
the exact text avoids recomputing repeated observations (the text already contains everything the
representation may depend on, so cached vectors cannot be stale). ``fingerprint()`` hashes every weight
so a run can prove the pretrained model is bit-identical at the end.

``FrozenLLMActor`` (baseline D) lets the same frozen model act directly: grammar-constrained choice of
every action slot by the model's own log-likelihood of the option text after a prompt containing the
same permissible information (goal + observation history in natural language).
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch


def wait_for_gpu(gb: float, max_wait_s: float = 4 * 3600, poll_s: float = 60.0) -> float:
    """Block until ``gb`` GiB of GPU memory are free (or ``max_wait_s`` passes); returns seconds waited."""
    t0 = time.time()
    while torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2**30 < gb and time.time() - t0 < max_wait_s:
        print(f"waiting for {gb:.0f} GiB free GPU memory before loading a frozen LM "
              f"({torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB free)", flush=True)
        time.sleep(poll_s)
    return time.time() - t0


class FrozenSemanticEncoder:
    def __init__(self, model: str = "Qwen/Qwen3-0.6B", layers: tuple = (0.5, 1.0), device: Optional[str] = None,
                 max_len: int = 384, batch: int = 64, cache_size: int = 200_000, dtype=torch.bfloat16):
        from transformers import AutoModel, AutoTokenizer

        self.name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "right"
        self.model = AutoModel.from_pretrained(model, dtype=dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        n = self.model.config.num_hidden_layers
        self.layer_ids = sorted({min(n, max(1, int(round(f * n)))) for f in layers})  # 1-based decoder layer outputs
        self._captured: dict = {}
        for li in self.layer_ids:  # capture only the needed layers (no output_hidden_states: far less memory)
            # same convention as hidden_states: layer outputs, and the final-norm output for the last layer
            (self.model.norm if li == n else self.model.layers[li - 1]).register_forward_hook(self._hook(li))
        self.dim = self.model.config.hidden_size * len(self.layer_ids)
        self.max_len, self.batch = max_len, batch
        self.cache: OrderedDict = OrderedDict()
        self.cache_size = cache_size
        self.stats = {"texts": 0, "cache_hits": 0, "forward_s": 0.0, "tokens": 0}
        try:
            self.revision = getattr(self.model.config, "_commit_hash", None)
        except Exception:  # noqa: BLE001
            self.revision = None

    def _hook(self, li):
        def fn(_mod, _inp, out):
            self._captured[li] = out[0] if isinstance(out, tuple) else out
        return fn

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for name, p in sorted(self.model.state_dict().items()):
            h.update(name.encode())
            h.update(p.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def describe(self) -> dict:
        return {"model": self.name, "revision": self.revision, "layers": self.layer_ids, "dim": self.dim,
                "params": sum(p.numel() for p in self.model.parameters()), "frozen": True, "pooling": "mask-mean"}

    @torch.no_grad()
    def _forward(self, texts: list[str]) -> np.ndarray:
        t0 = time.time()
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).to(self.device)
        self._captured.clear()
        self.model(**enc)
        m = enc["attention_mask"].unsqueeze(-1).float()
        vecs = [(self._captured[i].float() * m).sum(1) / m.sum(1).clamp(min=1) for i in self.layer_ids]
        self._captured.clear()
        v = torch.cat(vecs, -1)
        self.stats["forward_s"] += time.time() - t0
        self.stats["tokens"] += int(enc["attention_mask"].sum())
        return v.to(torch.float16).cpu().numpy()

    def encode(self, texts: list[str]) -> np.ndarray:
        """(n, dim) float16 array; batched, cached by exact text."""
        out = np.zeros((len(texts), self.dim), dtype=np.float16)
        todo = []
        for i, t in enumerate(texts):
            v = self.cache.get(t)
            if v is not None:
                self.cache.move_to_end(t)
                out[i] = v
                self.stats["cache_hits"] += 1
            else:
                todo.append(i)
        self.stats["texts"] += len(texts)
        uniq = list(dict.fromkeys(texts[i] for i in todo))
        order = sorted(range(len(uniq)), key=lambda j: len(uniq[j]))  # length-bucketed batches: less padding
        got = {}
        for s in range(0, len(order), self.batch):
            chunk = [uniq[j] for j in order[s:s + self.batch]]
            for t, v in zip(chunk, self._forward(chunk)):
                got[t] = v
        for i in todo:
            v = got[texts[i]]
            out[i] = v
            self.cache[texts[i]] = v
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return out


class FrozenLLMActor:
    """Baseline D: the frozen LLM chooses each grammar slot by log-likelihood of the option text."""

    PRELUDE = {
        "negotiation": "You are a buyer negotiating with a supplier. Actions: ASK <issue> (learn how much the supplier cares "
                       "about price, qty, delivery or warranty); PROPOSE <price> <qty> <delivery> <warranty> <argument> (levels "
                       "N0..N10 for each issue, argument NOARG or volume/longterm/market/quality/urgent); ACCEPT (accept the "
                       "supplier's current counter-offer); WALK (leave).",
        "product": "You run an early-stage software product. Actions: RESEARCH <segment S0-S3> (interview customers); BUILD "
                   "<feature F0-F5> (costs 15000, ships in 2-3 weeks); SETPRICE <P0..P5> (free, 5, 10, 20, 40, 80 per month); "
                   "CAMPAIGN <ads|content|partners> <B1..B4> (spend 1000/3000/8000/20000 per week for 4 weeks); ONBOARD "
                   "(improve onboarding, 8000); WAIT <W1..W4> (wait 1-4 weeks).",
        "bizdev": "You are recruiting partners for a joint project. Actions: RESEARCH <partner P0-P4> (learn what they want); "
                  "PITCH <partner> <revenue share N0..N10 in 5% steps> <EXCL|NOEXCL> <FAST|SLOW>; FOLLOWUP <partner> "
                  "(chase a pending reply); WAIT.",
    }

    def __init__(self, model: str = "Qwen/Qwen3-4B", device: Optional[str] = None, dtype=torch.bfloat16, max_ctx: int = 1536):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":  # shared GPU: never squeeze the running training processes
            wait_for_gpu(10.0 if "4B" in model else 4.0)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_ctx = max_ctx
        self.stats = {"decisions": 0, "forward_s": 0.0}

    @torch.no_grad()
    def _score(self, prefix: str, options: list[str]) -> np.ndarray:
        """Total log-prob of each option string as the continuation of ``prefix`` (one batched forward)."""
        t0 = time.time()
        p_ids = self.tok(prefix, add_special_tokens=False)["input_ids"][-self.max_ctx:]
        seqs = [p_ids + self.tok(" " + o, add_special_tokens=False)["input_ids"] for o in options]
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.to(self.device), att.to(self.device)
        # project to the vocabulary only at the option-token positions: full logits for every position of every
        # option sequence (options x ~1.6k tokens x 152k vocab, plus float copies) peaked at >16 GB
        h = self.model.get_decoder()(input_ids=ids, attention_mask=att).last_hidden_state
        head = self.model.get_output_embeddings()
        out = []
        for i, s in enumerate(seqs):
            tgt = torch.tensor(s[len(p_ids):], device=self.device)
            pos = torch.arange(len(p_ids) - 1, len(s) - 1, device=self.device)
            logp = torch.log_softmax(head(h[i, pos]).float(), -1)
            out.append(float(logp[torch.arange(len(tgt), device=self.device), tgt].sum()))
        self.stats["forward_s"] += time.time() - t0
        return np.array(out)

    def act(self, family: str, goal: str, history: list[str], grammar) -> list[str]:
        self.stats["decisions"] += 1
        prompt = (f"{self.PRELUDE[family]}\n{goal}\n" + "\n".join(history[-6:])
                  + "\nChoose the single best next action. Next action:")
        prefix: list[str] = []
        while True:
            opts = grammar.allowed(prefix)
            if opts == ["END"]:
                return prefix
            pre_txt = prompt + ("" if not prefix else " " + " ".join(prefix))
            sc = self._score(pre_txt, opts) if len(opts) > 1 else np.zeros(1)
            prefix.append(opts[int(np.argmax(sc))])


class HashingTextEncoder:
    """Test stand-in with the FrozenSemanticEncoder interface: a fixed random projection of hashed word
    unigrams/bigrams (no pretrained knowledge). Used for fast plumbing tests only."""

    def __init__(self, dim: int = 96, seed: int = 0):
        self.name, self.dim, self.seed = "hashing-bow", dim, seed
        self.stats = {"texts": 0, "cache_hits": 0, "forward_s": 0.0, "tokens": 0}

    def fingerprint(self) -> str:
        return f"hashing-{self.dim}-{self.seed}"

    def describe(self) -> dict:
        return {"model": self.name, "revision": None, "layers": [], "dim": self.dim, "params": 0, "frozen": True}

    def encode(self, texts):
        import zlib

        out = np.zeros((len(texts), self.dim), dtype=np.float16)
        for i, t in enumerate(texts):
            w = t.lower().split()
            for a, b in zip(w, w[1:] + [""]):
                for tok in (a, a + "_" + b):
                    h = zlib.crc32(f"{self.seed}:{tok}".encode())
                    out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        self.stats["texts"] += len(texts)
        return out
