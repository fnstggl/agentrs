"""Step inputs and batching: environment observations -> agent-visible text -> frozen-LLM features -> tensors.

A ``StepInput`` holds only what the agent may see (goal, element texts, feedback text, previous action, tool docs,
candidate structure). It stores **texts**, not vectors: features are recomputed (and mostly cache-hit) by the frozen
LLM when a batch is built, so replay buffers and checkpoints stay small and resumable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from agentrs.cam2.model import EL_FEATS
from agentrs.cam2.tools import Candidate, Observation, ToolSpec, enumerate_candidates


class ToolRegistry:
    """Tool sets seen so far (key -> docs, gen-arg kinds, specs). Shared by actor and learner; checkpointed."""

    def __init__(self):
        self.sets: dict = {}

    def key_for(self, tools: list) -> str:
        import hashlib

        k = hashlib.sha1("\n".join(t.doc() for t in tools).encode()).hexdigest()[:16]
        if k not in self.sets:
            self.sets[k] = {"docs": [t.doc() for t in tools], "specs": list(tools),
                            "gen": [None if t.gen_arg is None else t.gen_arg.type for t in tools]}
        return k

    def state_dict(self) -> dict:
        return {k: {"specs": [t.to_dict() for t in v["specs"]]} for k, v in self.sets.items()}

    def load_state_dict(self, s: dict) -> None:
        from agentrs.cam2.tools import ArgSpec

        for k, v in s.items():
            specs = [ToolSpec(d["name"], d["description"], tuple(ArgSpec(**a) for a in d["args"]), d["effect"])
                     for d in v["specs"]]
            self.sets[k] = {"docs": [t.doc() for t in specs], "specs": specs,
                            "gen": [None if t.gen_arg is None else t.gen_arg.type for t in specs]}


@dataclass
class StepInput:
    goal: str
    el_texts: list
    el_f: list                      # per element: (targetable, depth/8, pos/N, non-page)
    text: str
    pact: str
    scal: tuple                     # (prev reward, step fraction, first-step flag, previous action invalid)
    tools_key: str
    cands: list                     # [(tool_idx, el_idx, enum_text|None)]
    el_ids: list = field(default_factory=list)   # environment element ids, aligned with el_texts


def make_step_input(obs: Observation, tools: list, reg: ToolRegistry, pact: str, prev_r: float, step: int,
                    max_steps: int, invalid: bool, max_cands: int = 512, max_elements: int = 80) -> StepInput:
    els = obs.elements[:max_elements]
    o2 = Observation(goal=obs.goal, elements=els, text=obs.text, step=obs.step)
    cands = enumerate_candidates(tools, o2, max_cands)
    n = max(1, len(els))
    el_f = [(float(e.targetable), min(e.depth, 16) / 8.0, i / n, float(e.role in ("history", "output")))
            for i, e in enumerate(els)]
    return StepInput(goal=obs.goal, el_texts=[e.text for e in els], el_f=el_f, text=obs.text or "",
                     pact=pact, scal=(float(prev_r), step / max(1, max_steps), float(step == 0), float(invalid)),
                     tools_key=reg.key_for(tools), cands=[(c.tool, c.element, c.enum) for c in cands],
                     el_ids=[e.eid for e in els])


def collate(llm, reg: ToolRegistry, steps: list, mem: Optional[torch.Tensor], mem_m: Optional[torch.Tensor],
            device, goal_cond: bool = False) -> dict:
    """Build the model batch for a list of StepInputs (frozen-LLM features computed/cached here, no grad)."""
    B = len(steps)
    g = llm.encode_tokens([s.goal for s in steps])
    Tg = max(1, max(x.shape[0] for x in g))
    goal = np.zeros((B, Tg, llm.hidden), dtype=np.float16)
    goal_m = np.zeros((B, Tg), dtype=bool)
    for i, x in enumerate(g):
        goal[i, : x.shape[0]] = x
        goal_m[i, : x.shape[0]] = True
    Ne = max(1, max(len(s.el_texts) for s in steps))
    flat = [t for s in steps for t in s.el_texts]
    texts = [s.text for s in steps]
    pacts = [s.pact or "(start of episode)" for s in steps]
    # tool and enum tables shared across the batch
    keys = list(dict.fromkeys(s.tools_key for s in steps))
    off, docs = {}, []
    for k in keys:
        off[k] = len(docs)
        docs += reg.sets[k]["docs"]
    enums = list(dict.fromkeys(e for s in steps for (_, _, e) in s.cands if e is not None))
    if goal_cond:  # elements encoded in the context of the goal (pooled over element tokens only)
        el_vecs = llm.encode_pooled_ctx([f"Goal: {s.goal}" for s in steps for _ in s.el_texts], flat) if flat else \
            np.zeros((0, llm.pooled_dim), dtype=np.float16)
        pooled = np.concatenate([el_vecs, llm.encode_pooled([t for t in texts if t] + pacts + docs + [f"value: {e}" for e in enums])])
    else:
        pooled = llm.encode_pooled(flat + [t for t in texts if t] + pacts + docs + [f"value: {e}" for e in enums])
    p = 0
    el = np.zeros((B, Ne, llm.pooled_dim), dtype=np.float16)
    el_m = np.zeros((B, Ne), dtype=bool)
    el_f = np.zeros((B, Ne, EL_FEATS), dtype=np.float32)
    for i, s in enumerate(steps):
        n = len(s.el_texts)
        el[i, :n] = pooled[p:p + n]
        el_m[i, :n] = True
        if n:
            el_f[i, :n] = np.asarray(s.el_f, dtype=np.float32)
        p += n
    text = np.zeros((B, llm.pooled_dim), dtype=np.float16)
    text_m = np.zeros(B, dtype=bool)
    for i, t in enumerate(texts):
        if t:
            text[i] = pooled[p]
            text_m[i] = True
            p += 1
    pact = pooled[p:p + B]
    p += B
    tool_tab = pooled[p:p + len(docs)]
    p += len(docs)
    enum_tab = pooled[p:p + len(enums)] if enums else np.zeros((1, llm.pooled_dim), dtype=np.float16)
    eidx = {e: i for i, e in enumerate(enums)}
    Nc = max(1, max(len(s.cands) for s in steps))
    c_tool = np.full((B, Nc), -1, dtype=np.int64)
    c_el = np.full((B, Nc), -1, dtype=np.int64)
    c_enum = np.full((B, Nc), -1, dtype=np.int64)
    cmask = np.zeros((B, Nc), dtype=bool)
    for i, s in enumerate(steps):
        o = off[s.tools_key]
        for j, (t, e, en) in enumerate(s.cands):
            c_tool[i, j] = o + t
            c_el[i, j] = e
            c_enum[i, j] = -1 if en is None else eidx[en]
            cmask[i, j] = True
    T = lambda a, dt=torch.float32: torch.as_tensor(a).to(device=device, dtype=dt)  # noqa: E731
    b = {"goal": T(goal), "goal_m": T(goal_m, torch.bool), "el": T(el), "el_m": T(el_m, torch.bool), "el_f": T(el_f),
         "text": T(text), "text_m": T(text_m, torch.bool), "pact": T(pact),
         "scal": T(np.asarray([s.scal for s in steps], dtype=np.float32)),
         "tool_tab": T(tool_tab), "enum_tab": T(enum_tab), "c_tool": T(c_tool, torch.long), "c_el": T(c_el, torch.long),
         "c_enum": T(c_enum, torch.long), "cmask": T(cmask, torch.bool)}
    if mem is not None:
        b["mem"], b["mem_m"] = mem.to(device).float(), mem_m.to(device)
    return b
