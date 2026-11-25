"""Frozen pretrained semantic agents with tools (spec Part 19 baselines 5-7), run through the same
`OpenWorldAdapter` records, SMDP clock and evaluator as CAM Unified.

* `FrozenSemanticAgent(variant="tools")` (baseline 5): Crucible's `research.llm_eval.EvalLLMPolicy` - the v4
  interface (budgeted tool catalogue, structured errors, recent-action history) with both prompt corrections on
  (result budget, no stale "last action" line) - on a frozen Qwen3 model (1.7B / 4B; 0.6B in tests). One model
  copy serves every environment of the batch.
* ``variant="memory"`` (baseline 6): `MemoryLLMPolicy` adds a persistent experience memory that survives episodes
  (and processes, via `save` / `load`): after every action a note (tool, arguments, outcome, error code, verified
  reward) is stored; each turn the top-k notes by lexical relevance to the goal and the current observation are
  shown in the prompt. Nothing about the model changes; only its context.
* ``variant="planning"`` (baseline 7): `BestOfNLLMPolicy` spends matched inference-time compute: N candidate tool
  calls are sampled at temperature > 0, candidates that name unknown tools or violate the tool's JSON schema are
  discarded (public information: the schema), and the survivors are scored by the same frozen model's probability
  of answering "yes" to "does this action make progress toward the goal?" - the best is executed. Generations,
  scoring forwards and tokens are counted (`usage`) so compute can be matched against CAM's planner.

All actions are labelled ``ActionSource.BASELINE``. Weights are never updated (`update` returns zero updates).
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Any, Optional, Sequence

from ..interfaces import ActionSource
from .agent_protocol import PolicyAgent, Transition

_WORD = re.compile(r"[a-z][a-z0-9_.-]{2,}")


def _words(text: str) -> list:
    return _WORD.findall(text.lower())


def _llm_base():
    from crucible.research.llm_eval import EvalLLMPolicy
    return EvalLLMPolicy


def make_backend(model: str = "Qwen/Qwen3-1.7B", device: str = "cuda", max_context: int = 8192):
    """A frozen local Hugging Face model (``local_files_only``; never downloads)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from crucible.agents.llm import HFBackend
    return HFBackend(model, device=device, max_context=max_context)


# ============================================================================================== memory variant
class ExperienceMemory:
    """Persistent notes with lexical (BM25-style) retrieval. JSON on disk."""

    def __init__(self, capacity: int = 4000):
        self.capacity = int(capacity)
        self.notes: list[dict] = []

    def add(self, text: str, meta: Optional[dict] = None) -> None:
        self.notes.append({"text": text, "meta": meta or {}, "n": len(self.notes)})
        if len(self.notes) > self.capacity:
            self.notes = self.notes[-self.capacity:]

    def search(self, query: str, k: int = 6) -> list:
        q = Counter(_words(query))
        if not q or not self.notes:
            return self.notes[-k:]
        docs = [Counter(_words(n["text"])) for n in self.notes]
        N = len(docs)
        df = Counter(w for d in docs for w in d)
        avg = sum(sum(d.values()) for d in docs) / max(1, N)
        scores = []
        for i, d in enumerate(docs):
            L = sum(d.values()) or 1
            s = 0.0
            for w in q:
                if w in d:
                    idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                    s += idf * d[w] * 2.2 / (d[w] + 1.2 * (0.25 + 0.75 * L / avg))
            scores.append((s + 1e-6 * i, i))          # ties -> more recent
        top = [self.notes[i] for s, i in sorted(scores, reverse=True)[:k] if s > 0]
        return top or self.notes[-k:]

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"capacity": self.capacity, "notes": self.notes}, fh)

    def load(self, path: str) -> None:
        with open(path) as fh:
            d = json.load(fh)
        self.capacity, self.notes = int(d["capacity"]), list(d["notes"])


def memory_policy_class():
    Base = _llm_base()

    class MemoryLLMPolicy(Base):
        """EvalLLMPolicy + persistent experience memory shown in the prompt (baseline 6)."""

        def __init__(self, *a, memory: Optional[ExperienceMemory] = None, k_notes: int = 6, note_chars: int = 220,
                     **kw):
            super().__init__(*a, **kw)
            self.memory = memory if memory is not None else ExperienceMemory()
            self.k_notes, self.note_chars = int(k_notes), int(note_chars)
            self.name = f"llm_memory:{self.model_name}"
            self.retrieved: list = []

        def build_messages(self, obs, tools):
            msgs, new = super().build_messages(obs, tools)
            from crucible.agents.llm import render_actor_observation
            query = (self.goal or "") + "\n" + render_actor_observation(obs, None, 10, 120)
            notes = self.memory.search(query, self.k_notes)
            self.retrieved = [n["n"] for n in notes]
            if notes:
                block = "Notes from your earlier experience (most relevant first):\n" + "\n".join(
                    "- " + n["text"][: self.note_chars] for n in notes)
                user = msgs[-1]["content"]
                cut = user.rfind("Tools (")
                user = (user[:cut] + block + "\n\n" + user[cut:]) if cut > 0 else (user + "\n\n" + block)
                msgs = [msgs[0], {"role": "user", "content": user}]
            return msgs, new

        def observe_result(self, action, result):
            super().observe_result(action, result)
            from crucible.api import render as R
            out = ("ok: " + R.compact_json(result.get("result"), 120)) if result.get("ok") else \
                f"error {(result.get('error') or {}).get('code')}: {(result.get('error') or {}).get('message', '')[:100]}"
            self.memory.add(f"{action.get('tool')} {R.compact_json(action.get('args', {}), 100)} -> {out}",
                            {"t": result.get("t"), "ok": bool(result.get("ok"))})

    return MemoryLLMPolicy


# ============================================================================================== planning variant
def best_of_n_policy_class():
    Base = _llm_base()

    class BestOfNLLMPolicy(Base):
        """EvalLLMPolicy with best-of-N candidate actions scored by the frozen model itself (baseline 7)."""

        def __init__(self, *a, n_candidates: int = 4, sample_temperature: float = 0.8, **kw):
            super().__init__(*a, **kw)
            self.n_candidates, self.sample_temperature = int(n_candidates), float(sample_temperature)
            self.name = f"llm_bestof{self.n_candidates}:{self.model_name}"
            self.stats.update(candidates=0, valid_candidates=0, score_forwards=0, score_tokens=0)
            self.last_candidates: list = []

        def _p_yes(self, msgs: list, cand: dict) -> float:
            be = self.backend
            torch = be.torch
            q = (msgs[-1]["content"][-6000:] + "\n\nProposed next action: " + json.dumps(cand)[:1500] +
                 "\nWill this action make progress toward the goal? Answer yes or no.")
            m = [{"role": "system", "content": "You judge proposed actions. Answer with one word: yes or no."},
                 {"role": "user", "content": q}]
            try:
                text = be.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = be.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            ids = be.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][:, -be.max_context:]
            with torch.no_grad():
                logits = be.model(ids.to(be.device)).logits[0, -1].float()
            yes = [be.tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
            no = [be.tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]
            lp = torch.log_softmax(logits, -1)
            py, pn = torch.logsumexp(lp[yes], 0), torch.logsumexp(lp[no], 0)
            self.stats["score_forwards"] += 1
            self.stats["score_tokens"] += int(ids.shape[1])
            return float(torch.sigmoid(py - pn))

        def act(self, obs, tools):
            from crucible.api import render as R
            from crucible.core import schema as S
            msgs, new = self.build_messages(obs, tools)
            by = {t["name"]: t for t in tools}
            cands = []
            for i in range(self.n_candidates):
                g = self.backend.generate(msgs, self.max_new_tokens, self.sample_temperature,
                                          (self.seed or 0) * 100_003 + self.turn * 31 + i)
                self.stats["calls"] += 1
                self.stats["prompt_tokens"] += g.prompt_tokens
                self.stats["gen_tokens"] += g.gen_tokens
                self.stats["gen_s"] += g.wall_s
                self.stats["candidates"] += 1
                a = R.parse_action(g.text)
                if "error" in a or a.get("tool") not in by:
                    continue
                args = a.get("args", {}) or {}
                if S.validate(by[a["tool"]].get("inputSchema") or {"type": "object"}, S.with_defaults(
                        by[a["tool"]].get("inputSchema") or {"type": "object"}, args)):
                    continue
                self.stats["valid_candidates"] += 1
                cands.append({"tool": a["tool"], "args": args})
            self.turn += 1
            uniq = list({json.dumps(c, sort_keys=True): c for c in cands}.values())
            scored = [(self._p_yes(msgs, c), c) for c in uniq]
            self.last_candidates = [{"p_yes": p, "action": c} for p, c in scored]
            rec = {"turn": self.turn - 1, "candidates": self.last_candidates, "new_tools": new}
            if not scored:
                self.stats["parse_failures"] += 1
                self.hist.append({"action": None, "note": "no valid candidate"})
                self.transcript.append(rec)
                self.last_result = None
                return None
            best = max(scored, key=lambda x: x[0])[1]
            action = {"tool": best["tool"], "args": best["args"]}
            rec["action"] = action
            self.hist.append({"action": action, "result": None})
            self.transcript.append(rec)
            return action

    return BestOfNLLMPolicy


# ============================================================================================== the agent
class FrozenSemanticAgent(PolicyAgent):
    """A frozen language model with tools as an `AgentP` (see module docstring). `backend`: a loaded
    `crucible.agents.llm.HFBackend` (shared by every env); otherwise `model` is loaded on `device`."""

    def __init__(self, variant: str = "tools", model: str = "Qwen/Qwen3-1.7B", device: str = "cuda", backend=None,
                 max_new_tokens: int = 256, n_candidates: int = 4, memory: Optional[ExperienceMemory] = None,
                 tool_chars: int = 9000, seed: int = 0, idle_wait_s: int = 3000, **policy_kw):
        if variant not in ("tools", "memory", "planning"):
            raise ValueError("variant must be tools | memory | planning")
        self.variant = variant
        self.backend = backend if backend is not None else make_backend(model, device)
        self.memory = memory if memory is not None else ExperienceMemory()
        kw = dict(backend=self.backend, interface="v4", max_new_tokens=max_new_tokens, tool_chars=tool_chars,
                  history_chars=900, last_action_line=False, seed=seed, **policy_kw)
        if variant == "tools":
            cls, extra = _llm_base(), {}
        elif variant == "memory":
            cls, extra = memory_policy_class(), {"memory": self.memory}
        else:
            cls, extra = best_of_n_policy_class(), {"n_candidates": n_candidates}
        super().__init__(lambda b: cls(**kw, **extra), source=ActionSource.BASELINE, idle_wait_s=idle_wait_s,
                         seed=seed)

    def save(self, path: str) -> None:
        if self.variant == "memory":
            self.memory.save(path)

    def load(self, path: str) -> None:
        if self.variant == "memory" and os.path.exists(path):
            self.memory.load(path)

    def observe(self, transition: Transition) -> None:
        super().observe(transition)
        if self.variant == "memory":                    # verified outcomes become notes too (no hidden state)
            for b, r in enumerate(transition.rewards):
                if r:
                    self.memory.add(f"verified progress {r:+.3f} after "
                                    f"{transition.records[b].action.tool if transition.records[b].action else 'waiting'}",
                                    {"reward": float(r)})
