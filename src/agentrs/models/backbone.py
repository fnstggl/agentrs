"""Language-model backbone shared by the policy, world model, critic and IDM.

Wraps a Hugging Face causal LM (any open-weight model: Pythia, SmolLM2, Qwen3,
...) or a small randomly initialised GPT-NeoX ("scratch", for offline tests),
with optional LoRA adapters via PEFT.

Core operations
* ``score(prompts, continuations)`` -> differentiable log p(continuation | prompt)
  summed over tokens. Uses a shared-prefix KV cache so each prompt is encoded
  once regardless of the number of candidate continuations (verified against
  the naive implementation in tests).
* ``pooled(texts)`` -> last-token hidden state (for value / event heads).
* ``generate(prompt, ...)`` -> sampled continuations with their log-probs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import ConfigDict, BaseModel, Field


class LoraSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | str = "all-linear"


class ScratchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    vocab_size: int = 512
    max_positions: int = 1024


class BackboneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "EleutherAI/pythia-31m"  # HF id, local path, or "scratch"
    revision: Optional[str] = None
    dtype: str = "float32"
    device: str = "cpu"
    lora: Optional[LoraSettings] = None
    max_prompt_tokens: int = 768
    head_keep_tokens: int = 64  # when truncating, keep this many leading tokens (goal)
    gradient_checkpointing: bool = False
    scratch: ScratchSettings = Field(default_factory=ScratchSettings)
    seed: int = 0


# ---------------------------------------------------------------------------- tokenizer
_SCRATCH_TOKENIZER_CACHE = Path(os.environ.get("AGENTRS_CACHE", Path.home() / ".cache" / "agentrs"))


def _scratch_corpus() -> list[str]:
    """Deterministic corpus for training the scratch tokenizer (MiniWeb prompts + actions)."""
    from agentrs.agent.runtime import TeacherActor, run_episode
    from agentrs.demonstrations.teacher import ScriptedTeacher
    from agentrs.environments.miniweb.core import MiniWebEnv
    from agentrs.environments.miniweb.tasks import task_set

    env = MiniWebEnv()
    texts = []
    for split in ("train", "test"):
        for t in task_set(split, n_per_family=6):
            ep = run_episode(env, t, TeacherActor(ScriptedTeacher(epsilon=0.3, seed=t.seed)))
            for tr in ep.transitions:
                texts.append((tr.belief_prompt or "") + " " + tr.action + "\n" + tr.diff_text)
    return texts


def build_scratch_tokenizer(vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    path = _SCRATCH_TOKENIZER_CACHE / f"scratch_tok_v1_{vocab_size}.json"
    if path.exists():
        tok = Tokenizer.from_file(str(path))
    else:
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|eos|>", "<|pad|>"],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tok.train_from_iterator(_scratch_corpus(), trainer=trainer)
        path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(path))
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|eos|>", pad_token="<|pad|>")


# ---------------------------------------------------------------------------- backbone
class LMBackbone(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        dtype = getattr(torch, cfg.dtype)
        if cfg.name == "scratch":
            from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

            s = cfg.scratch
            self.tokenizer = build_scratch_tokenizer(s.vocab_size)
            mcfg = GPTNeoXConfig(vocab_size=len(self.tokenizer), hidden_size=s.hidden_size,
                                 num_hidden_layers=s.num_layers, num_attention_heads=s.num_heads,
                                 intermediate_size=4 * s.hidden_size, max_position_embeddings=s.max_positions,
                                 rotary_pct=0.25, use_parallel_residual=True, tie_word_embeddings=False)
            model = GPTNeoXForCausalLM(mcfg)
            self.base_id = f"scratch:{hashlib.sha1(s.model_dump_json().encode()).hexdigest()[:8]}"
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(cfg.name, revision=cfg.revision)
            model = AutoModelForCausalLM.from_pretrained(cfg.name, revision=cfg.revision, dtype=dtype)
            self.base_id = cfg.name + (f"@{cfg.revision}" if cfg.revision else "")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = model.to(dtype)
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        if cfg.lora is not None:
            from peft import LoraConfig, get_peft_model

            lc = LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
                            target_modules=cfg.lora.target_modules, task_type="CAUSAL_LM")
            model = get_peft_model(model, lc)
        self.model = model
        self.hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else \
            model.get_base_model().config.hidden_size
        self.newline_id = self._encode("\n")[-1]
        self.flops_counter = 0  # 2 * params * tokens forward (approximate)
        self._n_params = sum(p.numel() for p in model.parameters())
        self.to(cfg.device)

    # ---------------------------------------------------------------- helpers
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _encode(self, s: str) -> list[int]:
        return self.tokenizer(s, add_special_tokens=False)["input_ids"]

    def encode_prompt(self, s: str) -> list[int]:
        ids = self._encode(s)
        m = self.cfg.max_prompt_tokens
        if len(ids) > m:
            h = self.cfg.head_keep_tokens
            ids = ids[:h] + ids[-(m - h):]
        return ids

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _count(self, n_tokens: int, backward: bool = False):
        self.flops_counter += (6 if backward else 2) * self._n_params * n_tokens

    def _lm(self):
        return self.model

    def _transformer_outputs(self, input_ids, attention_mask, position_ids=None, past_key_values=None,
                             use_cache=False, logits_to_keep=0, output_hidden_states=False):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                          past_key_values=past_key_values, use_cache=use_cache, logits_to_keep=logits_to_keep,
                          output_hidden_states=output_hidden_states)

    # ---------------------------------------------------------------- scoring
    def _out_emb(self) -> torch.Tensor:
        return self.model.get_output_embeddings().weight

    def _check_hidden_logits(self) -> bool:
        """Verify that logits == last_hidden @ W_out^T (needed for vocab-restricted scoring)."""
        if getattr(self, "_hl_ok", None) is None:
            with torch.no_grad():
                ids = torch.tensor([self._encode("check hidden to logits mapping")], device=self.device)
                o = self._transformer_outputs(ids, torch.ones_like(ids), output_hidden_states=True)
                rec = o.hidden_states[-1].float() @ self._out_emb().float().T
                b = getattr(self.model.get_output_embeddings(), "bias", None)
                if b is not None:
                    rec = rec + b.float()
                self._hl_ok = bool(torch.allclose(rec, o.logits.float(), atol=1e-3, rtol=1e-3))
        return self._hl_ok

    def _token_logps(self, hidden: torch.Tensor, targets: torch.Tensor, owners: torch.Tensor,
                     vocab_rows: Optional[list[torch.Tensor]]) -> torch.Tensor:
        """log p(target | position) for hidden states (N, T, d) -> (N, T).

        vocab_rows=None: exact full-vocabulary softmax.
        Otherwise: softmax restricted to the per-state candidate vocabulary vocab_rows[owner]."""
        W = self._out_emb()
        bias = getattr(self.model.get_output_embeddings(), "bias", None)
        if vocab_rows is None:
            logits = hidden.float() @ W.float().T
            if bias is not None:
                logits = logits + bias.float()
            return F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        union = torch.unique(torch.cat(vocab_rows))
        pos_in_union = torch.full((W.shape[0],), -1, dtype=torch.long, device=hidden.device)
        pos_in_union[union] = torch.arange(len(union), device=hidden.device)
        logits = hidden.float() @ W[union].float().T  # (N, T, U)
        if bias is not None:
            logits = logits + bias[union].float()
        allowed = torch.zeros((len(vocab_rows), len(union)), dtype=torch.bool, device=hidden.device)
        for i, rows in enumerate(vocab_rows):
            allowed[i, pos_in_union[rows]] = True
        mask = allowed[owners].unsqueeze(1)  # (N, 1, U)
        logits = logits.masked_fill(~mask, float("-inf"))
        tgt = pos_in_union[targets].clamp(min=0)
        return F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

    def score(self, prompts: Sequence[str], continuations: Sequence[Sequence[str]], prefix_cache: bool = True,
              vocab_restrict: bool = False):
        """log p(cont_ij | prompt_i) for every prompt i and continuation j.

        Returns a list (len = len(prompts)) of 1-D tensors (len = len(continuations[i])),
        differentiable w.r.t. trainable parameters. With ``vocab_restrict`` every token
        distribution is normalised over the tokens that occur in prompt i's candidate set
        (a deterministic function of the state), which removes the full-vocabulary softmax
        cost on CPU.
        """
        cids = [[self._encode(c) for c in cs] for cs in continuations]
        vocab_rows = None
        if vocab_restrict and self._check_hidden_logits():
            vocab_rows = [torch.tensor(sorted({t for c in cs for t in c}) or [0], device=self.device)
                          for cs in cids]
        if prefix_cache:
            return self._score_cached(prompts, cids, vocab_rows)
        return self._score_full(prompts, cids, vocab_rows)

    def _forward_full(self, prompts, cids):
        """Right-padded prompt+continuation forward. Returns hidden states, ids, spans, owners."""
        dev = self.device
        seqs, spans, owners = [], [], []
        for i, (p, cs) in enumerate(zip(prompts, cids)):
            pid = self.encode_prompt(p)
            for cid in cs:
                seqs.append(pid + cid)
                spans.append((len(pid), len(pid) + len(cid)))
                owners.append(i)
        L = max(len(s) for s in seqs)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):
            ids[k, : len(s)] = torch.tensor(s)
            mask[k, : len(s)] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        self._count(int(mask.sum()))
        out = self._transformer_outputs(ids, mask, logits_to_keep=1, output_hidden_states=True)
        return out.hidden_states[-1], ids, spans, owners

    def _continuation_logps(self, hidden, ids, spans, owners, vocab_rows):
        """Token log-probs only at continuation positions -> (N, Cmax) padded with 0, and lengths."""
        dev = hidden.device
        Cmax = max(b - a for a, b in spans)
        pos = torch.zeros((len(spans), Cmax), dtype=torch.long, device=dev)
        valid = torch.zeros((len(spans), Cmax), dtype=torch.bool, device=dev)
        for k, (a, b) in enumerate(spans):
            pos[k, : b - a] = torch.arange(a - 1, b - 1, device=dev)
            valid[k, : b - a] = True
        h = hidden.gather(1, pos.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        tgt = ids.gather(1, (pos + 1).clamp(max=ids.shape[1] - 1))
        own = torch.tensor(owners, device=dev)
        lp = self._token_logps(h, tgt, own, vocab_rows)
        return torch.where(valid, lp, 0.0)

    def _score_full(self, prompts, cids, vocab_rows=None):
        dev = self.device
        if not any(cids):
            return [torch.zeros(0, device=dev) for _ in prompts]
        hidden, ids, spans, owners = self._forward_full(prompts, cids)
        tot = self._continuation_logps(hidden, ids, spans, owners, vocab_rows).sum(-1)
        res: list[list[torch.Tensor]] = [[] for _ in prompts]
        for k, i in enumerate(owners):
            res[i].append(tot[k])
        return [torch.stack(o) if o else torch.zeros(0, device=dev) for o in res]

    def nll_and_prompt_hidden(self, prompts: Sequence[str], targets: Sequence[str]):
        """One forward pass giving (summed target NLL, target token counts, hidden state at the
        last prompt token). Used by the world model to share computation between its
        generative diff loss and its pooled-state heads."""
        cids = [[self._encode(t)] for t in targets]
        hidden, ids, spans, owners = self._forward_full(prompts, cids)
        lp = self._continuation_logps(hidden, ids, spans, owners, None)
        last = torch.tensor([a - 1 for a, _ in spans], device=hidden.device)
        h_prompt = hidden[torch.arange(len(spans), device=hidden.device), last].float()
        counts = torch.tensor([b - a for a, b in spans], device=hidden.device, dtype=torch.float32)
        return -lp.sum(-1), counts, h_prompt

    @staticmethod
    def _expand_cache(cache, idx: torch.Tensor, counts: list[int]) -> None:
        """Replicate each prompt's KV cache once per candidate. Candidates of a prompt are
        contiguous, so repeat_interleave (cheap segment-sum backward) is used when the cache
        exposes per-layer tensors; otherwise fall back to the generic index select."""
        layers = getattr(cache, "layers", None)
        if layers and all(hasattr(l, "keys") and hasattr(l, "values") for l in layers):
            rep = torch.tensor(counts, device=idx.device)
            for l in layers:
                l.keys = torch.repeat_interleave(l.keys, rep, dim=0)
                l.values = torch.repeat_interleave(l.values, rep, dim=0)
        else:  # pragma: no cover - other cache implementations
            cache.batch_select_indices(idx)

    def _score_cached(self, prompts, cids_nested, vocab_rows=None):
        dev = self.device
        B = len(prompts)
        pids = [self.encode_prompt(p) for p in prompts]
        P = max(len(p) for p in pids)
        pad = self.tokenizer.pad_token_id
        # left-pad prompts so the last prompt token is aligned at position P-1
        pin = torch.full((B, P), pad, dtype=torch.long)
        pmask = torch.zeros((B, P), dtype=torch.long)
        for i, p in enumerate(pids):
            pin[i, P - len(p):] = torch.tensor(p)
            pmask[i, P - len(p):] = 1
        pin, pmask = pin.to(dev), pmask.to(dev)
        ppos = (pmask.cumsum(-1) - 1).clamp(min=0)
        self._count(int(pmask.sum()))
        pout = self._transformer_outputs(pin, pmask, position_ids=ppos, use_cache=True, logits_to_keep=1,
                                         output_hidden_states=True)
        h_last = pout.hidden_states[-1][:, -1:]  # (B, 1, d): predicts first continuation token
        cache = pout.past_key_values
        owners, cids = [], []
        for i, cs in enumerate(cids_nested):
            for c in cs:
                owners.append(i)
                cids.append(c)
        if not cids:
            return [torch.zeros(0, device=dev) for _ in prompts]
        idx = torch.tensor(owners, device=dev)
        self._expand_cache(cache, idx, [len(cs) for cs in cids_nested])
        C = max(len(c) for c in cids)
        cin = torch.full((len(cids), C), pad, dtype=torch.long)
        cmask = torch.zeros((len(cids), C), dtype=torch.long)
        for k, c in enumerate(cids):
            cin[k, : len(c)] = torch.tensor(c)
            cmask[k, : len(c)] = 1
        cin, cmask = cin.to(dev), cmask.to(dev)
        plen = pmask.sum(-1)[idx]
        cpos = plen.unsqueeze(-1) + torch.arange(C, device=dev).unsqueeze(0)
        full_mask = torch.cat([pmask[idx], cmask], dim=1)
        self._count(int(cmask.sum()))
        cout = self._transformer_outputs(cin, full_mask, position_ids=cpos, past_key_values=cache,
                                         logits_to_keep=1, output_hidden_states=True)
        # hidden states that predict continuation tokens 0..C-1
        hid = torch.cat([h_last[idx], cout.hidden_states[-1][:, :-1]], dim=1)  # (N, C, d)
        tok_lp = torch.where(cmask.bool(), self._token_logps(hid, cin, idx, vocab_rows), 0.0)
        tot = tok_lp.sum(-1)
        res: list[list[torch.Tensor]] = [[] for _ in prompts]
        for k, i in enumerate(owners):
            res[i].append(tot[k])
        return [torch.stack(o) if o else torch.zeros(0, device=dev) for o in res]

    def token_nll(self, prompts: Sequence[str], targets: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-example summed NLL and token counts of targets given prompts (teacher forcing)."""
        scores = self._score_full(prompts, [[self._encode(t)] for t in targets])
        nll = -torch.stack([s[0] for s in scores])
        counts = torch.tensor([max(1, len(self._encode(t))) for t in targets], device=self.device,
                              dtype=torch.float32)
        return nll, counts

    # ---------------------------------------------------------------- pooled features
    def pooled(self, texts: Sequence[str]) -> torch.Tensor:
        dev = self.device
        ids_list = [self.encode_prompt(t) for t in texts]
        L = max(len(x) for x in ids_list)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(texts), L), pad, dtype=torch.long)
        mask = torch.zeros((len(texts), L), dtype=torch.long)
        for i, x in enumerate(ids_list):
            ids[i, L - len(x):] = torch.tensor(x)
            mask[i, L - len(x):] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        self._count(int(mask.sum()))
        out = self._transformer_outputs(ids, mask, position_ids=pos, logits_to_keep=1, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1].float()  # left padding => last position is the last real token
        return h

    # ---------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, prompt: str, num_samples: int = 1, max_new_tokens: int = 48, temperature: float = 1.0,
                 stop_text: Optional[str] = None, stop_token: Optional[int] = None,
                 greedy: bool = False, generator: Optional[torch.Generator] = None) -> list[tuple[str, float]]:
        """Sample continuations of one prompt. Returns (text, log-prob) pairs. Stops at the
        newline token by default (``stop_token``), or once the text contains ``stop_text``."""
        return self.generate_batch([prompt] * num_samples, max_new_tokens, temperature, stop_text, stop_token,
                                   greedy, generator)

    @torch.no_grad()
    def generate_batch(self, prompts: Sequence[str], max_new_tokens: int = 48, temperature: float = 1.0,
                       stop_text: Optional[str] = None, stop_token: Optional[int] = None, greedy: bool = False,
                       generator: Optional[torch.Generator] = None) -> list[tuple[str, float]]:
        """Batched sampling for several (possibly different) prompts with a shared KV-cache loop."""
        outs = self.generate_tokens(prompts, max_new_tokens, temperature, stop_text, stop_token, greedy, generator)
        return [(o["text"], float(sum(o["logprobs"]))) for o in outs]

    @torch.no_grad()
    def generate_tokens(self, prompts: Sequence[str], max_new_tokens: int = 48, temperature: float = 1.0,
                        stop_text: Optional[str] = None, stop_token: Optional[int] = None, greedy: bool = False,
                        generator: Optional[torch.Generator] = None) -> list[dict]:
        """Sample continuations and return, per prompt: decoded ``text``, the exact sampled token
        ``ids``, the per-token ``logprobs`` under the sampling distribution (softmax(logits / T)),
        and whether a stop condition was reached (``finished``). The ids are what training must
        re-score: re-tokenising the decoded text can produce a different token sequence."""
        dev = self.device
        n = len(prompts)
        pids = [self.encode_prompt(p) for p in prompts]
        P = max(len(p) for p in pids)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((n, P), pad, dtype=torch.long)
        mask = torch.zeros((n, P), dtype=torch.long)
        for i, p in enumerate(pids):
            ids[i, P - len(p):] = torch.tensor(p)
            mask[i, P - len(p):] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out = self._transformer_outputs(ids, mask, position_ids=pos, use_cache=True, logits_to_keep=1)
        self._count(int(mask.sum()))
        cache = out.past_key_values
        logits = out.logits[:, -1].float()
        nxt_pos = mask.sum(-1, keepdim=True)
        stop = self.newline_id if stop_token is None else stop_token
        eos = self.tokenizer.eos_token_id
        gen: list[list[int]] = [[] for _ in range(n)]
        tok_lps: list[list[float]] = [[] for _ in range(n)]
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        for _ in range(max_new_tokens):
            logp = F.log_softmax(logits / max(temperature, 1e-5), dim=-1)
            if greedy:
                nxt = logp.argmax(-1)
            else:
                nxt = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
            chosen = logp.gather(-1, nxt[:, None]).squeeze(-1)
            for i in range(n):
                if not done[i]:
                    gen[i].append(int(nxt[i]))
                    tok_lps[i].append(float(chosen[i]))
            if stop_text is not None:
                newly = torch.tensor([stop_text in self.tokenizer.decode(g[-12:]) for g in gen], device=dev)
            else:
                newly = nxt == stop
            done = done | newly | (nxt == eos)
            if bool(done.all()):
                break
            step_ids = nxt[:, None]
            mask = torch.cat([mask, torch.ones_like(step_ids)], dim=1)
            o = self._transformer_outputs(step_ids, mask, position_ids=nxt_pos.clone(), past_key_values=cache,
                                          use_cache=True)
            self._count(n)
            nxt_pos = nxt_pos + 1
            cache = o.past_key_values
            logits = o.logits[:, -1].float()
        return [{"text": self.tokenizer.decode(g), "ids": g, "logprobs": lp, "finished": bool(d)}
                for g, lp, d in zip(gen, tok_lps, done.tolist())]

    def token_logprobs_from_ids(self, prompts: Sequence[str], cont_ids: Sequence[Sequence[int]],
                                return_logdists: bool = False):
        """Exact per-token log-probabilities (full vocabulary, T = 1) of given continuation token ids
        after each prompt. Differentiable. Returns (logps (N, Cmax), mask (N, Cmax)) and, with
        ``return_logdists``, the full log-distributions (N, Cmax, V) at those positions (for exact KL)."""
        dev = self.device
        hidden, ids, spans, owners = self._forward_full(list(prompts), [[list(c)] for c in cont_ids])
        Cmax = max(1, max(b - a for a, b in spans))
        pos = torch.zeros((len(spans), Cmax), dtype=torch.long, device=dev)
        valid = torch.zeros((len(spans), Cmax), dtype=torch.bool, device=dev)
        for k, (a, b) in enumerate(spans):
            if b > a:
                pos[k, : b - a] = torch.arange(a - 1, b - 1, device=dev)
                valid[k, : b - a] = True
        h = hidden.gather(1, pos.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        W = self._out_emb()
        logits = h.float() @ W.float().T
        bias = getattr(self.model.get_output_embeddings(), "bias", None)
        if bias is not None:
            logits = logits + bias.float()
        logd = F.log_softmax(logits, dim=-1)
        tgt = ids.gather(1, (pos + 1).clamp(max=ids.shape[1] - 1))
        lp = torch.where(valid, logd.gather(-1, tgt.unsqueeze(-1)).squeeze(-1), 0.0)
        if return_logdists:
            return lp, valid, logd
        return lp, valid

    # ---------------------------------------------------------------- persistence
    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    def load_trainable_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")


def config_hash(cfg: BaseModel) -> str:
    return hashlib.sha1(json.dumps(cfg.model_dump(), sort_keys=True, default=str).encode()).hexdigest()[:10]

