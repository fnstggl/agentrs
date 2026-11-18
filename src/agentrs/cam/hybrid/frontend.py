"""Text front-ends that let a *structured* CAM act on natural-language observations (arm C).

* ``ParserFrontend`` — the rule-based parser of the training wording (simple text-parser baseline);
  unreadable text yields an empty observation (only the family token) rather than invented content.
* ``LLMFrontend`` — a frozen pretrained LM rewrites each natural-language observation / goal into CAM's
  structured tokens (few-shot prompt with *training-wording* examples only; greedy decoding; tokens
  outside CAM's vocabulary are dropped). Batched across workers and cached by exact text.
Neither front-end sees anything but the rendered text (no hidden state)."""
from __future__ import annotations

import random
import time
from collections import OrderedDict

import torch

from agentrs.cam.envs import ENVS
from agentrs.cam.text import render as R


class ParserFrontend:
    kind = "parser"

    def __init__(self):
        self.stats = {"calls": 0, "failures": 0}

    def obs(self, fam: str, text: str) -> list[str]:
        self.stats["calls"] += 1
        try:
            return R.parse_obs(fam, text)
        except Exception:  # noqa: BLE001 - unreadable wording
            self.stats["failures"] += 1
            return []

    def obs_batch(self, fams, texts):
        return [self.obs(f, t) for f, t in zip(fams, texts)]

    def goal(self, fam: str, text: str) -> list[str]:
        try:
            return R.parse_goal(fam, text)
        except Exception:  # noqa: BLE001
            self.stats["failures"] += 1
            return ["GOAL"]


def _examples(fam: str, n: int = 3, seed: int = 0):
    r = random.Random(seed)
    out = []
    e = ENVS[fam]()
    e.reset(777, "train")
    g = e.grammar()
    while not e.done and len(out) < n:
        pre = []
        while not pre or pre[-1] != "END":
            pre.append(r.choice(g.allowed(pre)))
        e.step(pre[:-1])
        out.append(e.obs_tokens())
    return out, e.goal_tokens()


class LLMFrontend:
    kind = "llm"

    def __init__(self, model: str = "Qwen/Qwen3-4B", device=None, batch: int = 24, cache_size: int = 100_000):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.waited_s = 0.0
        if self.device == "cuda":  # shared GPU: weights + batched generation cache; never squeeze running training
            need, t0 = (14.0 if "4B" in model else 5.0), time.time()
            while torch.cuda.mem_get_info()[0] / 2**30 < need and time.time() - t0 < 4 * 3600:
                print(f"LLM front-end: waiting for {need:.0f} GiB free GPU memory "
                      f"({torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB free)", flush=True)
                time.sleep(60)
            self.waited_s = time.time() - t0
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(self.device).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.batch = batch
        self.cache: OrderedDict = OrderedDict()
        self.cache_size = cache_size
        self.shots = {}
        for fam in ENVS:
            ex, gt = _examples(fam)
            obs_shots = "".join(f"Text: {R.obs_text(fam, t, 'train', key=('shot', i))}\nTokens: {' '.join(t)}\n\n" for i, t in enumerate(ex))
            goal_shot = f"Text: {R.goal_text(fam, gt, 'train', key=('shot', 'g'))}\nTokens: {' '.join(gt)}\n\n"
            self.shots[fam] = (obs_shots, goal_shot)
        self.stats = {"calls": 0, "cache_hits": 0, "gen_s": 0.0}

    @torch.no_grad()
    def _gen(self, prompts: list[str], max_new: int) -> list[str]:
        t0 = time.time()
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        g = self.lm.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
        self.stats["gen_s"] += time.time() - t0
        return [self.tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0] for x in g]

    def _run(self, fam_texts: list[tuple[str, str, str]], max_new: int) -> list[list[str]]:
        out = [None] * len(fam_texts)
        todo = []
        for i, (fam, kind, text) in enumerate(fam_texts):
            self.stats["calls"] += 1
            v = self.cache.get((kind, text))
            if v is not None:
                out[i] = v
                self.stats["cache_hits"] += 1
            else:
                todo.append(i)
        for s in range(0, len(todo), self.batch):
            idx = todo[s:s + self.batch]
            prompts = []
            for i in idx:
                fam, kind, text = fam_texts[i]
                shots = self.shots[fam][0] if kind == "obs" else self.shots[fam][1]
                prompts.append("Rewrite the text into the structured token format.\n\n" + shots + f"Text: {text}\nTokens:")
            for i, line in zip(idx, self._gen(prompts, max_new)):
                out[i] = line.split()
                self.cache[(fam_texts[i][1], fam_texts[i][2])] = out[i]
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return out

    def obs_batch(self, fams, texts):
        return self._run([(f, "obs", t) for f, t in zip(fams, texts)], 160)

    def obs(self, fam, text):
        return self.obs_batch([fam], [text])[0]

    def goal(self, fam, text):
        return self._run([(fam, "goal", text)], 80)[0] or ["GOAL"]
