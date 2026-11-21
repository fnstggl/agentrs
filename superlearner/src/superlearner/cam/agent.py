"""The CAM agent's decision procedure.  [candidate-editable]

One *decision* is one choice of an action head by the controller, followed - for a tool - by one choice per
argument. Heads are built from the situation, not from a list of strategies:

    every tool the actor can call now      (the world's primitive operations, including services it discovered)
    wait <duration>                        let simulated time pass
    ask the frozen LLM for a program       (a free-form multi-step draft with a plan; costs LLM compute)
    run the draft / skip its next step     (execute the whole draft as one decision, or delete a step)
    define a tool from the last k calls    (the agent's own reusable procedure, see grounding/toolsmith.py)
    use one of the agent's own tools

Arguments are chosen option by option (`features.build_slots`); an option may be an identifier the actor has seen,
a number, an enum value, what the draft suggested, what worked before, or "let the frozen LLM write this value
from this document" (code, text, structured objects). Following the draft's next step with its own arguments,
overriding some of them, inserting other steps in between and skipping steps are all ordinary choices, so the
controller can accept, edit or ignore what the language model proposed.

A `Chooser` supplies the choices: `ModelChooser` (the trainable controller - sampling or greedy), a teacher
(demonstrations), a uniform-random control, or a fixed rule (the "frozen LLM with grounding" baseline). The
decision record (`step`) keeps the items, the choices, log-probabilities and values needed to recompute the
policy's likelihood during training.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .. import ensure_crucible
from ..actions.program import Interpreter, ProgramError, parse_program, strip_internal
from ..grounding.executor import GroundedExecutor, Outcome
from ..grounding.knowledge import CapabilityIndex, Knowledge, alias_kind, tool_doc, tool_full_doc
from ..grounding.toolsmith import ToolLibrary, macro_from_trace
from ..llm import prompts as P
from ..llm.base import extract_json, text_key
from . import features as F
from .memory import EpisodicMemory, HistStep, WorkingMemory
from .model import CAMController, EmbStore

ensure_crucible()
from crucible.integrations import obs_features  # noqa: E402

DAY = 86400
AUX_DIM = 16
_DIGITS = re.compile(r"\d+")


@dataclass
class AgentConfig:
    history: int = 12
    ctx_docs: int = 4
    ctx_draft: int = 6
    ctx_mem: int = 3
    llm_budget: int = 3            # program proposals per episode
    gen_budget: int = 10           # LLM-written argument values per episode
    use_llm: bool = True
    use_memory: bool = True
    use_tools: bool = True         # tool-making and agent tools
    macro_spans: tuple = (2, 3, 4, 6)
    max_lib_heads: int = 12
    propose_tokens: int = 700
    gen_tokens: int = 1200
    think_entries: bool = True     # let the frozen LLM reason before writing a structured document entry
    plan_with_world_model: str = ""   # path of a trained world model: one-step lookahead re-ranks the head choice
    plan_beta_ok: float = 1.0
    plan_beta_gain: float = 1.0


class Chooser:
    name = "abstract"

    def head(self, ag: "CAMAgent", items: F.Items, logits: np.ndarray) -> int:
        raise NotImplementedError

    def option(self, ag: "CAMAgent", tool: str, slot: F.Slot, items: F.Items, logits: np.ndarray) -> int:
        raise NotImplementedError


class ModelChooser(Chooser):
    """Sample from (or take the argmax of) the controller's distribution."""
    name = "model"

    def __init__(self, greedy: bool = False, temperature: float = 1.0, seed: int = 0):
        self.greedy, self.temperature = greedy, temperature
        self.rng = np.random.default_rng(seed)

    def _pick(self, logits: np.ndarray) -> int:
        if self.greedy:
            return int(np.argmax(logits))
        z = logits.astype(np.float64) / max(1e-6, self.temperature)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        return int(self.rng.choice(len(p), p=p))

    def head(self, ag, items, logits):
        return self._pick(logits)

    def option(self, ag, tool, slot, items, logits):
        return self._pick(logits)


class RandomChooser(Chooser):
    """Uniform over the same heads and options (control)."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def head(self, ag, items, logits):
        return int(self.rng.integers(0, len(items)))

    def option(self, ag, tool, slot, items, logits):
        return int(self.rng.integers(0, len(items)))


class LLMOnlyChooser(Chooser):
    """Baseline without neural learning: the frozen LLM plans everything through the grounded interface.
    Rule: if there is a draft, run it; otherwise ask for one; when the proposal budget is spent, wait a day."""
    name = "llm+grounding"

    def head(self, ag, items, logits):
        kinds = [m[0] for m in items.meta]
        if "draft_run" in kinds:
            return kinds.index("draft_run")
        if "propose" in kinds:
            return kinds.index("propose")
        return next(i for i, m in enumerate(items.meta) if m[0] == "wait" and m[1] == DAY)

    def option(self, ag, tool, slot, items, logits):
        return 0


class CAMAgent:
    def __init__(self, model: CAMController | None, store: EmbStore, llm, cfg: AgentConfig | None = None,
                 library: ToolLibrary | None = None, memory: EpisodicMemory | None = None, device: str = "cpu"):
        self.model, self.store, self.llm = model, store, llm
        self.cfg = cfg or AgentConfig()
        self.library = library if library is not None else ToolLibrary()
        self.memory = memory if memory is not None else EpisodicMemory()
        self.device = device
        self._pending: dict[int, str] = {}
        self.state = None
        self.human = False          # also record a readable trace of every decision (for inspection)

    def _world_model(self):
        path = self.cfg.plan_with_world_model
        if not path:
            return None
        if getattr(self, "_wm_path", None) != path:
            from .worldmodel import WorldModel
            ck = torch.load(path, map_location=self.device, weights_only=False)
            wm = WorldModel(ck["d"])
            wm.load_state_dict(ck["wm"])
            self._wm, self._wm_path = wm.to(self.device).eval(), path
        return self._wm

    # ================================================================== embeddings
    def _ek(self, text: str) -> int:
        k = text_key(text)
        if not self.store.has(k):
            self._pending[k] = text
        return k

    def _flush(self) -> None:
        if not self._pending:
            return
        keys, texts = list(self._pending), list(self._pending.values())
        vecs = self.llm.embed(texts)
        for k, v in zip(keys, vecs):
            self.store.put(k, v)
        self._pending = {}

    def _vec(self, key: int) -> np.ndarray:
        self._flush()
        return self.store.mat[self.store.key2row.get(int(key), 0)]

    def _tensors(self, a: dict):
        rows = self.store.rows(a["ek"])
        dev = self.device
        return (torch.from_numpy(rows).to(dev)[None], torch.from_numpy(a["tok"].astype(np.int64)).to(dev)[None],
                torch.from_numpy(a["num"].astype(np.float32)).to(dev)[None],
                torch.from_numpy(a["typ"].astype(np.int64)).to(dev)[None])

    # ================================================================== episode
    def begin(self, core, episode_id: str = "", learn: bool = True) -> None:
        self.core = core
        self.info = core.info()
        self.episode_id, self.learn = episode_id, learn
        self.actor, self.goal = self.info["actor"], self.info["goal"]
        self.know = Knowledge(self.actor)
        self.caps = CapabilityIndex(self.llm)
        self._sync_caps(core.tools(True) or [])
        self.exec = GroundedExecutor(core.execute, self.caps, self.know)
        self.wm = WorkingMemory()
        self.mine: set = {self.actor}
        self.goal_key = self._ek(self.goal)
        self.t0 = self.info["time"]
        self.now = self.t0
        self.last_out: Outcome | None = None
        self.tick = 0.0
        self.n_dec = 0
        self.elapsed_frac = 0.0
        self.scratch_repo: str | None = None
        self.llm_use = {"propose": 0, "gen": 0, "propose_fail": 0, "gen_fail": 0}
        self.decisions: list[dict] = []
        for h in self.info.get("start_history", []):      # what the scripted start state did, as the actor
            res = h["result"] or {}
            self.know.observe(action={"tool": h["tool"], "args": h["args"]}, result=res)
            if res.get("ok"):
                self._claim(res.get("result"))
        self.mem_hits = []
        self.mem_args: dict = {}
        self.mem_tools: set = set()
        if self.cfg.use_memory and len(self.memory):
            self.mem_hits = self.memory.retrieve(self._vec(self.goal_key), self.cfg.ctx_mem)
            for sim, it in self.mem_hits:
                if it["success"] and sim > 0.5:
                    self.mem_tools |= set(it["tools"])
                    for tool, args in it["args"].items():
                        self.mem_args.setdefault(tool, args)
        if self.model is not None:
            self.state = self.model.init_state(1, self.device)
        self.obs = core.observe()
        self._see(self.obs)

    def _sync_caps(self, tools: list) -> None:
        """Refresh the capability index and put the frozen embeddings of new tool descriptions into the store."""
        self.caps.update(tools)
        for name in self.caps.order:
            k = self.caps.keys[name]
            if not self.store.has(k):
                self.store.put(k, self.caps.vec(name))

    def _claim(self, x: Any, depth: int = 0) -> None:
        """Identifiers returned by the actor's own successful state-changing calls are 'mine'."""
        if depth > 3:
            return
        if isinstance(x, str) and len(x) <= 81 and alias_kind(x) is not None:
            self.mine.add(x)
        elif isinstance(x, dict):
            for v in x.values():
                self._claim(v, depth + 1)
        elif isinstance(x, list):
            for v in x[:8]:
                self._claim(v, depth + 1)

    def _see(self, obs: dict) -> None:
        self.know.observe(obs=obs, now=obs.get("time", 0))
        self.now = obs.get("time", self.now)
        for mod in ("identity", "bank", "cloud"):
            m = obs.get(mod) or {}
            for sub in ("orgs", "balances", "services"):
                if isinstance(m.get(sub), dict):
                    self.mine |= set(m[sub])

    # ================================================================== primitive execution
    def _call(self, tool: str, args: dict) -> Outcome:
        out = self.exec.execute(tool, args)
        if out.sent:
            self.know.observe(action={"tool": tool, "args": out.args}, result=out.to_result(), now=self.now)
        if out.ok:
            self.wm.calls.append((tool, out.args, out.result))
            self.wm.since_define += 1
            self.wm.last_ok_args[tool] = out.args
            if not out.read_only:
                self._claim(out.result)
        self.wm.note_use(tool, out.ok)
        return out

    def _run_code(self, source: str, inp: Any) -> Outcome:
        """Run agent-written Python inside Crucible's sandbox through the world's own code tools."""
        if self.scratch_repo is None:
            self.wm.fresh_n += 1
            out = self._call("code.create_repo", {"name": f"scratch-{self.wm.fresh_n}"})
            if not out.ok:
                return out
            self.scratch_repo = out.result["repo"]
        path = f"tools/t{hashlib.sha1(source.encode()).hexdigest()[:10]}.py"
        out = self._call("code.commit", {"repo": self.scratch_repo, "files": {path: source}})
        if not out.ok:
            return out
        out = self._call("code.run", {"repo": self.scratch_repo, "target": f"{path}:main", "input": inp})
        if out.ok and isinstance(out.result, dict) and not out.result.get("ok", True):
            err = out.result.get("error") or {}
            return Outcome(False, "code", {"input": inp}, None, {"code": "sandbox_error", "message": str(err.get("message"))[:300]},
                           "runtime", f"your code raised {err.get('type')}: {str(err.get('message'))[:300]}", out.cost_s, True)
        if out.ok and isinstance(out.result, dict):
            out.result = out.result.get("result")
        return out

    def _interp(self) -> Interpreter:
        return Interpreter(self._call, self.core.wait, gen=self._gen_free, code=self._run_code, library=_Lib(self),
                           max_ops=14)

    # ================================================================== frozen LLM
    def _recent_docs(self, k: int = 4) -> list:
        docs = list(self.know.docs.values())[-k:][::-1]
        for d in docs:
            self._ek(d.text)
        return docs

    def _focus_docs(self, k: int = 5) -> list:
        """Documents a language-model-written value could be based on: the most similar to the goal interleaved
        with the most recently read (embeds every remembered document, so it is only called when needed)."""
        docs = list(self.know.docs.values())
        if not docs:
            return []
        for d in docs:
            self._ek(d.text)
        g = self._vec(self.goal_key)
        rows = self.store.rows(np.array([d.key for d in docs], np.int64))
        sims = self.store.mat[rows] @ g
        # the most similar to the goal first, interleaved with the most recently read ones
        by_sim = list(np.argsort(-sims))
        by_new = list(range(len(docs) - 1, -1, -1))
        out: list = []
        for a, b in zip(by_sim, by_new):
            for i in (a, b):
                if docs[i] not in out:
                    out.append(docs[i])
        return out[:k]

    def _gen_value(self, tool: str, spec: dict, slot: F.Slot, opt: F.Opt, chosen: dict) -> tuple[bool, Any, str]:
        if self.wm.n_gen >= self.cfg.gen_budget:
            return False, None, "the budget of language-model-written values for this episode is spent"
        self.wm.n_gen += 1
        self.llm_use["gen"] += 1
        q = self.goal + " " + tool_doc(spec)
        names: list = []
        if self.caps.has(tool):       # tools that operate on what this call produces, then generally similar ones
            names += [n for n, sc_ in self.caps.search(q, 4, restrict=self.caps.consumers(tool)) if sc_ > -5]
        names += [n for n, _ in self.caps.search(q, 3, exclude=(tool,)) if n not in names]
        related = [tool_doc(self.caps.tools[n], 420) for n in names[:6]]
        repair = None
        prev = self.wm.last_gen.get((tool, slot.param))
        if prev is not None:          # error-specific repair: show what was written before and what went wrong since
            users = set(self.caps.consumers(tool)) | {tool} if self.caps.has(tool) else {tool}
            bad = next((s_ for s_ in reversed(self.wm.steps[-4:]) if not s_.ok and s_.tool in users), None)
            if bad is not None:
                repair = (prev, f"{bad.tool}: {self.know.alias_text(bad.feedback)}")
        msgs = P.gen_messages(self.goal, tool_full_doc(spec), slot.param, slot.schema, related,
                              self.know.aliased(chosen), opt.doc.text if opt.doc is not None else None, repair)
        big = slot.kind in ("object", "list")
        ap = slot.schema.get("additionalProperties")
        ap_t = (ap.get("type") if isinstance(ap, dict) else None) or []
        if F._ptype(slot.schema) == "object" and isinstance(ap, dict) and "string" in (ap_t if isinstance(ap_t, list) else [ap_t]):
            # an object of named texts (files): entry names first, then each entry as plain text. A document inside
            # a document never has to be escaped, and each piece is written knowing the pieces before it.
            okv, val = self._gen_entries(msgs, slot.param, opt.variant)
            if not okv:
                self.llm_use["gen_fail"] += 1
                return False, None, f"the language model did not produce usable entries for `{slot.param}`"
            import json as _json
            self.wm.last_gen[(tool, slot.param)] = _json.dumps(val, indent=1)[:4000]
            return True, self.know.resolve(val), ""
        g = self.llm.generate(msgs, self.cfg.gen_tokens if big else 160, opt.variant)
        txt = g.text.strip()
        self.wm.last_gen[(tool, slot.param)] = txt
        if slot.kind == "text":
            v = extract_json(txt) if txt[:1] in "[{" else None
            if not isinstance(v, str):
                v = txt.strip().strip('"')
            okv = isinstance(v, str) and len(v) > 0
        else:
            v = extract_json(txt)
            want = F._ptype(slot.schema)
            okv = (isinstance(v, dict) if want == "object" else isinstance(v, list) if want == "array" else v is not None)
        if not okv:
            self.llm_use["gen_fail"] += 1
            return False, None, f"the language model did not produce a usable value for `{slot.param}`"
        ap = slot.schema.get("additionalProperties")
        if isinstance(v, dict) and isinstance(ap, dict) and "string" in (ap.get("type") if isinstance(ap.get("type"), list) else [ap.get("type")]):
            import json as _json      # a JSON document given as an object where text is required: serialise it
            v = {k: (_json.dumps(x, indent=1) if isinstance(x, (dict, list)) else x) for k, x in v.items()}
        return True, self.know.resolve(v), ""

    def _gen_entries(self, base: list, param: str, variant: int) -> tuple[bool, dict]:
        g = self.llm.generate(P.gen_keys_messages(base, param), 120, variant)
        names = extract_json(g.text)
        if isinstance(names, dict):
            names = list(names)
        if not isinstance(names, list) or not names or not all(isinstance(n, str) and 0 < len(n) <= 120 for n in names):
            return False, {}
        names = list(dict.fromkeys(names))[:6]
        done: dict = {}
        for n in names:
            # structured documents (a manifest, a config) must match a documented format exactly: the frozen model
            # may reason before writing those. Other entries (source code) are written directly.
            think = self.cfg.think_entries and n.rsplit(".", 1)[-1].lower() in ("json", "yaml", "yml", "toml")
            msgs = P.gen_entry_messages(base, param, names, done, n)
            g = self.llm.generate(msgs, self.cfg.gen_tokens, variant, think)
            if think and not g.text.strip():              # the reasoning used up its budget: write directly
                g = self.llm.generate(msgs, self.cfg.gen_tokens, variant, False)
            txt = g.text.strip()
            if txt.startswith("```"):                     # a fenced block: keep what is inside the fence
                body = txt.split("\n", 1)[1] if "\n" in txt else ""
                txt = body.rsplit("```", 1)[0] if "```" in body else body
            done[n] = txt.strip("\n") + "\n"
        return True, done

    def _gen_free(self, brief: str) -> Any:
        self.wm.n_gen += 1
        self.llm_use["gen"] += 1
        g = self.llm.generate([{"role": "system", "content": P.SYSTEM_GEN},
                               {"role": "user", "content": f"Goal: {self.goal}\n\nWrite: {brief}"}], 300, 0)
        v = extract_json(g.text)
        return self.know.resolve(v if v is not None else g.text.strip())

    def _recent_lines(self, n: int = 8) -> list[str]:
        lines = []
        for s in self.wm.steps[-n:]:
            a = P.compact(self.know.aliased(s.args), 160)
            if s.ok:
                lines.append(f"- {s.tool} {a} -> ok {P.compact(self.know.aliased(s.result), 220)}")
            else:
                lines.append(f"- {s.tool} {a} -> FAILED ({s.fclass}): {self.know.alias_text(s.feedback)[:220]}")
        return lines

    def _propose(self, variant: int) -> Outcome:
        self.wm.n_propose += 1
        self.llm_use["propose"] += 1
        fb = self.know.alias_text(self.last_out.feedback) if self.last_out is not None and not self.last_out.ok else ""
        hits = self.caps.search(self.goal + " " + self.wm.plan + " " + fb, 10)
        full = [tool_full_doc(self.caps.tools[n]) for n, _ in hits[:5]]
        brief = [tool_doc(self.caps.tools[n], 120) for n, _ in hits[5:]]
        state = self.know.aliased({k: v for k, v in self.obs.items() if k not in ("time", "principal", "events", "events_dropped")})
        el = f"day {(self.now - self.t0) / DAY:.1f} of {self.info['horizon_s'] / DAY:.0f}"
        msgs = P.propose_messages(self.goal, el, self.know.entity_summary(), self._recent_lines(), fb, full, brief,
                                  [self.library.describe(n) for n in self.library.names()[:8]], state, self.wm.plan,
                                  [d.text for d in self._focus_docs(3)])
        g = self.llm.generate(msgs, self.cfg.propose_tokens, variant)
        try:
            prog = parse_program(g.text)
        except ProgramError as e:
            self.llm_use["propose_fail"] += 1
            return Outcome(False, "llm.propose", {}, None, {"code": "invalid_action", "message": str(e)}, "syntax",
                           f"the language model's reply was not a valid program: {e}", sent=False)
        self.wm.draft = list(prog["steps"])
        self.wm.plan = prog["plan"] or self.wm.plan
        return Outcome(True, "llm.propose", {"variant": variant},
                       {"plan": prog["plan"], "steps": len(prog["steps"])}, sent=False)

    # ================================================================== items
    def _aux(self) -> np.ndarray:
        wm = self.wm
        bal = (self.obs.get("bank") or {}).get("balances") or {}
        cash = sum(v for v in bal.values() if isinstance(v, (int, float)))
        t = self.now % DAY
        a = [self.elapsed_frac, self.n_dec / max(1, self.info.get("max_decisions", 60)),
             1.0 - wm.n_propose / max(1, self.cfg.llm_budget), 1.0 - wm.n_gen / max(1, self.cfg.gen_budget),
             min(len(wm.draft), 8) / 8.0, min(len(self.library.tools), 8) / 8.0,
             0.0 if self.last_out is None else (1.0 if self.last_out.ok else -1.0),
             min(wm.consecutive_failures(), 5) / 5.0, self.exec.n_ok / max(1, self.exec.n_calls),
             F.symlog(cash) / 15.0, min(len((self.obs.get("identity") or {}).get("orgs") or {}), 4) / 4.0,
             min(len((self.obs.get("cloud") or {}).get("services") or {}), 4) / 4.0,
             math.sin(2 * math.pi * t / DAY), math.cos(2 * math.pi * t / DAY), min(len(self.know.docs), 48) / 48.0, 1.0]
        return np.asarray(a, np.float32)

    def _ctx_items(self) -> F.Items:
        it, wm, cfg = F.Items(), self.wm, self.cfg
        it.add(F.C_GOAL, ["G"], [], self.goal_key)
        steps = wm.steps[-cfg.history:]
        for i, s in enumerate(steps):
            spec = self.caps.tools.get(s.tool)
            ek = self.caps.keys.get(s.tool, 0)
            toks = F.tool_tokens(s.tool, spec)[:5] + [f"OUT:{s.fclass or 'ok'}", f"KIND:{s.kind}", f"TO:{s.tool}:{s.fclass or 'ok'}"]
            it.add(F.C_HIST, toks, [(len(steps) - i) / cfg.history, float(s.ok), F.symlog(s.cost_s) / 12.0, min(s.n_calls, 8) / 8.0], ek)
        lo = self.last_out
        if lo is not None and not lo.ok:
            fb = _DIGITS.sub("0", self.know.alias_text(lo.feedback)[:240])
            it.add(F.C_ERR, [f"FC:{lo.fclass}", f"T:{lo.tool}", f"TF:{lo.tool}:{lo.fclass}",
                             f"EC:{(lo.error or {}).get('code', '')}"], [1.0], self._ek(fb) if fb else 0)
        for r, d in enumerate(self._recent_docs(cfg.ctx_docs)):
            it.add(F.C_DOC, [f"DS:{d.source}"], [r / max(1, cfg.ctx_docs), min(1.0, (self.now - d.t) / DAY)], d.key)
        for i, s in enumerate(wm.draft[:cfg.ctx_draft]):
            name = s.get("call") if s.get("_kind") == "call" else f"<{s.get('_kind')}>"
            it.add(F.C_DRAFT, F.tool_tokens(name, self.caps.tools.get(name))[:6] + [f"DP:{i}", f"DK:{s.get('_kind')}"],
                   [i / cfg.ctx_draft, float(name in self.caps.tools), len(wm.draft) / 8.0], self.caps.keys.get(name, 0))
        if wm.plan:
            it.add(F.C_PLAN, ["PLAN"], [min(len(wm.plan), 400) / 400.0], self._ek(wm.plan[:400]))
        for sim, m in self.mem_hits:
            toks = ["MS:ok" if m["success"] else "MS:fail"] + [f"T:{t}" for t in list(dict.fromkeys(m["tools"]))[:9]]
            it.add(F.C_MEM, toks, [sim, float(m["success"]), m["ret"], min(m["n"], 60) / 60.0], m["goal_key"])
        return it

    def _head_items(self) -> F.Items:
        it, wm, cfg = F.Items(), self.wm, self.cfg
        dnext = wm.draft[0].get("call") if wm.draft and wm.draft[0].get("_kind") == "call" else None
        dany = {s.get("call") for s in wm.draft if s.get("_kind") == "call"}
        for name in self.caps.order:
            spec = self.caps.tools[name]
            u = wm.tool_uses.get(name, (0, 0, 0))
            req = len((spec.get("inputSchema") or {}).get("required") or [])
            ann = spec.get("annotations") or {}
            it.add(F.H_TOOL, F.tool_tokens(name, spec),
                   [math.log1p(u[0]) / 3, float(u[2] > 0), float(u[2] < 0), math.log1p(u[1]) / 3, float(name == dnext),
                    float(name in dany), float(not ann.get("builtin", True)), req / 8.0,
                    F.symlog(ann.get("costSeconds", 0)) / 10.0, float(name in self.mem_tools)],
                   self.caps.keys[name], ("tool", name))
        for d in F.WAITS:
            it.add(F.H_WAIT, ["M:wait", f"D:{d}"], [F.symlog(d) / 12.0], 0, ("wait", d))
        if cfg.use_llm and wm.n_propose < cfg.llm_budget:
            for v in range(2):
                it.add(F.H_PROPOSE, ["M:propose", f"V:{v}"], [wm.n_propose / max(1, cfg.llm_budget), float(v)], 0, ("propose", v))
        if wm.draft:
            it.add(F.H_DRAFT_RUN, ["M:draft_run"], [min(len(wm.draft), 8) / 8.0], 0, ("draft_run",))
            it.add(F.H_DRAFT_SKIP, ["M:draft_skip"], [min(len(wm.draft), 8) / 8.0], 0, ("draft_skip",))
        if cfg.use_tools:
            for k in cfg.macro_spans:
                if wm.since_define >= k and len(wm.calls) >= k:
                    it.add(F.H_DEFINE, ["M:define", f"K:{k}"], [k / 8.0, len(self.library.tools) / 8.0], 0, ("define", k))
            for name in self.library.names()[:cfg.max_lib_heads]:
                t = self.library.tools[name]
                s = t["stats"]
                toks = ["M:use", f"N:{name}", f"LK:{t['kind']}"] + [f"T:{d}" for d in t["deps"][:6]]
                it.add(F.H_USE, toks, [math.log1p(s["calls"]) / 3, s["ok"] / max(1, s["calls"]), float(t["verified"]),
                                       len(t["deps"]) / 8.0], self._ek(t["description"] or name), ("use", name))
        return it

    def _slot_context(self, tool: str) -> F.SlotContext:
        wm = self.wm
        dargs = None
        for s in wm.draft:
            if s.get("_kind") == "call" and s.get("call") == tool:
                dargs = {k: v for k, v in (s.get("args") or {}).items()
                         if not (isinstance(v, str) and v.startswith("$")) and not (isinstance(v, dict) and set(v) & {"expr", "gen"})}
                dargs = self.know.resolve(dargs)
                break
        prev = wm.last_ok_args.get(tool) or self.mem_args.get(tool)
        llm_ok = self.cfg.use_llm and wm.n_gen < self.cfg.gen_budget
        return F.SlotContext(self.actor, self.know, self.mine, dargs, prev, self._focus_docs(5) if llm_ok else [],
                             llm_ok, wm.fresh_n)

    # ================================================================== one decision
    @torch.no_grad()
    def step(self, chooser: Chooser) -> dict:
        m = self.model
        ctx, heads = self._ctx_items(), self._head_items()
        obsf = obs_features(self.obs, self.last_out.to_result() if self.last_out is not None else None, F.OBS_DIM)
        aux = self._aux()
        self._flush()
        ca, ha = ctx.arrays(), heads.arrays()
        rec: dict = {"ctx": ca, "obsf": obsf.astype(np.float16), "aux": aux, "tick": float(self.tick), "head": ha,
                     "slots": [], "t": int(self.now)}
        logp, ent, z, he = 0.0, 0.0, None, None
        if m is not None:
            m.set_table(self.store.tensor(self.device))
            ce = m.embed(*self._tensors(ca))
            c = m.encode(ce, torch.ones(1, len(ctx), dtype=torch.bool, device=self.device),
                         torch.from_numpy(obsf)[None].to(self.device), torch.from_numpy(aux)[None].to(self.device))
            z, self.state = m.recur(c, self.state, torch.tensor([self.tick], device=self.device))
            he = m.embed(*self._tensors(ha))
            hl = m.head_logits(z, he, torch.ones(1, len(heads), dtype=torch.bool, device=self.device))[0]
            lsm = torch.log_softmax(hl, -1)
            hlog = hl.cpu().numpy()
            wm = self._world_model()
            if wm is not None:          # hypothetical one-step rollouts in latent space, before acting
                hlog = (lsm + wm.plan_bonus(z, he, self.cfg.plan_beta_ok, self.cfg.plan_beta_gain)).cpu().numpy()
            vals = m.values(z)[0]
            rec["value"], rec["vstd"] = float(vals.mean()), float(vals.std())
            rec["p_succ"] = float(torch.sigmoid(m.p_success(z))[0])
        else:
            hlog = np.zeros(len(heads), np.float32)
            rec["value"], rec["vstd"], rec["p_succ"] = 0.0, 0.0, 0.0
        h = int(chooser.head(self, heads, hlog))
        rec["h"] = h
        if self.human:
            pr = np.exp(hlog - hlog.max())
            pr = pr / pr.sum()
            top = np.argsort(-pr)[:4]
            rec["human"] = {"obs": P.compact({k: v for k, v in self.obs.items() if k not in ("events", "events_dropped")}, 500),
                            "events": len(self.obs.get("events") or []), "p_head": float(pr[h]),
                            "top_heads": [(":".join(str(x) for x in heads.meta[i]), round(float(pr[i]), 4)) for i in top],
                            "n_heads": len(heads), "plan": self.wm.plan[:200], "draft_len": len(self.wm.draft)}
        if m is not None:
            logp += float(lsm[h])
            ent += float(-(lsm.exp() * lsm).sum())
        meta = heads.meta[h]
        kind = meta[0]
        out, n_calls, args = None, 1, {}
        tool_name = meta[1] if kind in ("tool", "use") else kind
        self.tick = 0.0
        if kind in ("tool", "use"):
            spec = self.caps.tools[tool_name] if kind == "tool" else None
            schema = (spec.get("inputSchema") if kind == "tool" else self.library.tools[tool_name]["params"]) or {}
            sc = self._slot_context(tool_name)
            slots = F.build_slots(tool_name, schema, sc)
            if slots is None:
                out = Outcome(False, tool_name, {}, None, {"code": "invalid_action", "message": "no value available"},
                              "syntax", "a required argument of this tool has no value you could supply", sent=False)
                self.exec.counts["syntax"] += 1
                self.exec.n_calls += 1
            else:
                prev_sum, n_prev, gen_err = None, 0, ""
                for j, slot in enumerate(slots):
                    par, opts = F.slot_items(tool_name, slot, j)
                    self._flush()
                    pa, oa = par.arrays(), opts.arrays()
                    if m is not None:
                        pe = m.embed(*self._tensors(pa))[:, 0]
                        oe = m.embed(*self._tensors(oa))
                        prev = (prev_sum / n_prev) if n_prev else torch.zeros_like(z)
                        al = m.arg_logits(z, he[:, h], pe, oe, torch.ones(1, len(opts), dtype=torch.bool, device=self.device), prev)[0]
                        alsm = torch.log_softmax(al, -1)
                        alog = al.cpu().numpy()
                    else:
                        alog = np.zeros(len(opts), np.float32)
                    c_j = int(chooser.option(self, tool_name, slot, opts, alog))
                    if m is not None:
                        logp += float(alsm[c_j])
                        ent += float(-(alsm.exp() * alsm).sum()) / len(slots)
                        prev_sum = oe[:, c_j] if prev_sum is None else prev_sum + oe[:, c_j]
                        n_prev += 1
                    rec["slots"].append({"par": pa, "opt": oa, "c": c_j})
                    o = slot.options[c_j]
                    if o.source == "omit":
                        continue
                    if o.source == "gen":
                        okv, val, msg = self._gen_value(tool_name, spec or {"name": tool_name, "description": "", "inputSchema": schema},
                                                        slot, o, args)
                        if not okv:
                            gen_err = msg
                            continue
                    else:
                        val = o.value
                        if o.source == "fresh":
                            self.wm.fresh_n += 1
                    if slot.param == F.WHOLE:          # a service endpoint without declared arguments: the whole object
                        if isinstance(val, dict):
                            args.update(val)
                    else:
                        args[slot.param] = val
                if gen_err:
                    out = Outcome(False, tool_name, args, None, {"code": "invalid_action", "message": gen_err}, "syntax",
                                  gen_err, sent=False)
                    self.exec.counts["syntax"] += 1
                    self.exec.n_calls += 1
                elif kind == "tool":
                    out = self._call(tool_name, args)
                    if self.wm.draft and self.wm.draft[0].get("call") == tool_name:
                        self.wm.draft.pop(0)                  # the draft's next step was taken (possibly edited)
                else:
                    tr_steps: list = []
                    interp = self._interp()
                    from ..actions.program import ProgramTrace
                    ptr = ProgramTrace(env={"me": self.actor})
                    interp._ops = 0
                    try:
                        out = interp.use(self.library.tools[tool_name], dict(args, me=self.actor), ptr, 0)
                    except Exception as e:  # noqa: BLE001 - a broken agent tool is a failed action, not a crash
                        out = Outcome(False, tool_name, args, None, {"code": "tool_failed", "message": str(e)[:200]},
                                      "runtime", f"your tool stopped: {str(e)[:200]}", sent=True)
                    tr_steps = ptr.steps
                    n_calls = max(1, len(tr_steps))
                    self.library.record(tool_name, out.ok, out.fclass, self.episode_id)
        elif kind == "wait":
            sec = self.core.wait(int(meta[1]))
            out = Outcome(True, "wait", {"seconds": sec}, {"waited_s": sec}, sent=False)
            self.tick = 1.0 if sec >= 6 * 3600 else 0.0
        elif kind == "propose":
            out = self._propose(int(meta[1]))
        elif kind == "draft_run":
            prog = {"plan": self.wm.plan, "steps": self.wm.draft}
            self.wm.draft = []
            tr = self._interp().run(prog, {"me": self.actor})
            n_calls = max(1, tr.n_calls)
            self.wm.subgoals += tr.subgoals
            self.wm.notes += tr.notes
            bad = next((s for s in tr.steps if not s.ok), None)
            out = Outcome(tr.ok, "draft.run", {"steps": len(prog["steps"])},
                          {"executed": len(tr.steps), "calls_ok": sum(1 for s in tr.steps if s.ok and s.kind == "call")},
                          None if tr.ok else {"code": "program_stopped", "message": tr.stopped},
                          "" if tr.ok else (bad.fclass if bad is not None and bad.fclass else "runtime"),
                          "" if tr.ok else tr.stopped, sum(s.cost_s for s in tr.steps), True)
        elif kind == "draft_skip":
            s0 = self.wm.draft.pop(0)
            out = Outcome(True, "draft.skip", {"skipped": str(s0.get("call") or s0.get("_kind"))}, {"remaining": len(self.wm.draft)}, sent=False)
        elif kind == "define":
            k = int(meta[1])
            calls = self.wm.calls[-k:]
            name = "m_" + hashlib.sha1("|".join(t for t, _, _ in calls).encode()).hexdigest()[:8]
            spec_t = macro_from_trace(calls, self.actor, name)
            okv, msg = self.library.register(spec_t, {"episode": self.episode_id, "mechanism": "macro_from_trace",
                                                      "goal": self.goal[:120], "t": int(self.now)})
            self.wm.since_define = 0
            out = Outcome(okv, "tool.define", {"k": k, "name": name}, {"message": msg} if okv else None,
                          None if okv else {"code": "invalid_action", "message": msg}, "" if okv else "syntax", "" if okv else msg, sent=False)
        self.last_out = out
        self.wm.steps.append(HistStep(kind, tool_name, out.args if kind in ("tool", "use") else dict(out.args), out.ok,
                                      out.fclass, out.feedback, _small(out.result), out.cost_s, n_calls, int(self.now)))
        ed = self.core.end_decision()
        self.n_dec += 1
        self.elapsed_frac = ed.get("elapsed_frac", 0.0)
        day0 = self.now // DAY
        tl = self.core.tools()
        if tl is not None:
            self._sync_caps(tl)
        if not (ed["terminated"] or ed["truncated"]):
            self.obs = self.core.observe()
            self._see(self.obs)
            if self.now // DAY != day0:
                self.tick = 1.0
        rec.update(logp=logp, ent=ent, kind=kind, tool=tool_name, ok=bool(out.ok), fclass=out.fclass, n_calls=n_calls,
                   reward=float(ed["reward"]), done=bool(ed["terminated"]), trunc=bool(ed["truncated"]))
        if self.human:
            rec["human"].update(action=f"{kind}:{tool_name}", args=P.compact(out.args, 500), ok=bool(out.ok),
                                fclass=out.fclass, feedback=out.feedback[:400], result=P.compact(out.result, 400),
                                sent_to_world=bool(out.sent), cost_s=int(out.cost_s), reward=float(ed["reward"]),
                                t_after=int(ed["time"]), llm=dict(self.llm_use))
        self.decisions.append(rec)
        return rec

    # ================================================================== whole episode
    def run(self, core, chooser: Chooser, episode_id: str = "", learn: bool = True, max_decisions: int = 10_000) -> dict:
        mark = self.store.n
        self.begin(core, episode_id, learn)
        while len(self.decisions) < max_decisions:
            r = self.step(chooser)
            if r["done"] or r["trunc"]:
                break
        summ = core.summary()
        if learn and self.cfg.use_memory:
            lit: dict = {}
            for tool, a, _ in self.wm.calls:
                lit[tool] = {k: v for k, v in a.items()
                             if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= 60 and alias_kind(v) is None)}
            self.memory.add(self.goal_key, self._vec(self.goal_key), summ["success"], summ["return"],
                            [t for t, _, _ in self.wm.calls], lit, len(self.decisions), episode_id)
        kinds: dict[str, int] = {}
        for d in self.decisions:
            kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
        summ.update(agent=self.exec.stats(), llm=dict(self.llm_use), heads=kinds, chooser=chooser.name,
                    lib_size=len(self.library.tools), mem_size=len(self.memory), episode=episode_id,
                    tools_used=sorted({d["tool"] for d in self.decisions if d["kind"] == "tool"}))
        return {"summary": summ, "decisions": self.decisions, "emb_new": self.store.items_since(mark)}


class _Lib:
    """The view of the tool library a running program gets (registration records provenance)."""

    def __init__(self, ag: CAMAgent):
        self.ag = ag

    def get(self, name):
        return self.ag.library.get(name)

    def register(self, spec):
        if not self.ag.cfg.use_tools:
            return False, "tool making is disabled"
        return self.ag.library.register(spec, {"episode": self.ag.episode_id, "mechanism": "program define_tool",
                                               "goal": self.ag.goal[:120], "t": int(self.ag.now)})


def _small(x: Any, n: int = 600) -> Any:
    try:
        import json
        s = json.dumps(x, default=str)
    except (TypeError, ValueError):
        return None
    return x if len(s) <= n else {"_truncated": s[:n]}
