"""Frozen-LLM agent: a pretrained language model (default Qwen3-1.7B, weights never updated) that reads the actor's
observation and emits ONE JSON tool call per turn.

What the model sees (all of it the actor's own view; nothing about how to achieve the goal):

* the goal description passed in by the caller (``goal=...`` / `set_goal`), verbatim;
* the rendered observation (time, outcome of its last action, module summaries, events it can see) - see
  `crucible.api.render.render_observation`;
* a compact tool list: one line per tool with argument names and types (``name(arg: type, opt?: type) [cost] -
  description``). Lists longer than ``tool_chars`` are summarised: newly discovered and recently used tools keep
  their full line, the rest are listed by name per host. Tools that become available later (deployed services the
  actor discovered) are announced as new in the turn they appear;
* its own last ``history`` actions with their results.

The reply is parsed with `crucible.api.render.parse_action` (code fences, ``<tool_call>`` tags, ``<think>`` blocks,
OpenAI/MCP shapes...). An unparseable reply means *no action this turn*; it is counted (``parse_failures``) and
recorded. Replying ``{"tool": "wait"}`` is an explicit no-op.

Decoding is greedy by default (``temperature=0``), so a run is deterministic for a fixed model, prompt and
hardware. Every generated text is recorded (`LLMPolicy.transcript`); `ReplayLLM` feeds a transcript back so a
trajectory can be re-executed without the model (and detects prompt divergence). `usage()` reports calls, prompt
tokens and generated tokens (for compute-matched comparisons).

`EchoLLM` is a TEST DOUBLE ONLY (scripted replies for unit tests of the plumbing). It is not an agent and is
never used for reported results.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..api import render as R
from .base import Policy

DEFAULT_SYSTEM_PROMPT = (
    "You are an agent living in a simulated internet. You act only through the tools listed below; each turn you "
    "make at most one tool call, then simulated time passes and you see what happened. "
    + R.action_format_help() +
    ' To do nothing this turn, reply {"tool": "wait", "args": {}}. Reply with the JSON object only.')
SKIP_KEYS = ("time", "principal", "events", "events_dropped")


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    gen_tokens: int
    truncated: bool = False
    wall_s: float = 0.0


# ============================================================================================== backends
class HFBackend:
    """A frozen Hugging Face causal LM loaded from the local cache (``local_files_only=True``: never downloads)."""

    def __init__(self, model: str = "Qwen/Qwen3-1.7B", device: str = "cuda", dtype: Any = None,
                 max_context: int = 8192, enable_thinking: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.torch = torch
        self.name, self.device, self.max_context, self.enable_thinking = model, device, max_context, enable_thinking
        dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, local_files_only=True).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for k, p in sorted(self.model.state_dict().items()):
            h.update(k.encode())
            h.update(p.detach().float().sum().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def generate(self, messages: list[dict], max_new_tokens: int = 256, temperature: float = 0.0,
                 seed: int | None = None) -> GenResult:
        torch = self.torch
        t0 = time.time()
        try:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=self.enable_thinking)
        except TypeError:  # tokenizer without the Qwen3 thinking switch
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        truncated = False
        room = self.max_context - max_new_tokens
        if len(ids) > room:   # keep the head (instructions, goal) and the tail (latest state + generation prompt)
            ids = torch.cat([ids[: room // 3], ids[-(room - room // 3):]])
            truncated = True
        inp = ids.unsqueeze(0).to(self.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        if temperature and temperature > 0:
            if seed is not None:
                torch.manual_seed(seed)
            kw.update(do_sample=True, temperature=float(temperature), top_p=1.0, top_k=0)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            out = self.model.generate(inp, attention_mask=torch.ones_like(inp), **kw)
        gen = out[0, inp.shape[1]:]
        reply = self.tok.decode(gen, skip_special_tokens=True)
        return GenResult(reply, int(inp.shape[1]), int(gen.shape[0]), truncated, time.time() - t0)


class EchoLLM:
    """TEST DOUBLE ONLY - not a model and not an agent. Returns scripted replies (a list consumed in order, cycling
    on the last one, or a function of the messages) so the policy's prompt building, parsing, accounting and
    recording can be unit-tested without a GPU. Token counts are whitespace-token counts."""

    name = "echo-test-double"
    is_test_double = True

    def __init__(self, replies: list[str] | Callable[[list[dict]], str]):
        self.replies = replies
        self.i = 0
        self.seen: list[list[dict]] = []

    def generate(self, messages, max_new_tokens=256, temperature=0.0, seed=None) -> GenResult:
        self.seen.append(messages)
        if callable(self.replies):
            text = self.replies(messages)
        else:
            text = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        n_prompt = sum(len(m["content"].split()) for m in messages)
        return GenResult(text, n_prompt, len(text.split()))


class ReplayLLM:
    """Re-issue the texts recorded in an `LLMPolicy.transcript`, in order (no model). With ``check_prompts`` the
    prompt of every turn must hash to the recorded prompt hash, so a replay that diverges from the original
    trajectory fails loudly instead of silently feeding stale text."""

    def __init__(self, transcript: list[dict], check_prompts: bool = True):
        self.records = [r for r in transcript if "text" in r]
        self.i = 0
        self.check = check_prompts
        self.name = "replay"

    def generate(self, messages, max_new_tokens=256, temperature=0.0, seed=None) -> GenResult:
        if self.i >= len(self.records):
            raise IndexError("replay transcript exhausted")
        r = self.records[self.i]
        self.i += 1
        if self.check and r.get("prompt_hash") and r["prompt_hash"] != prompt_hash(messages):
            raise RuntimeError(f"replay diverged at turn {self.i - 1}: prompt differs from the recorded one")
        return GenResult(r["text"], r.get("prompt_tokens", 0), r.get("gen_tokens", 0))


def prompt_hash(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]


# ============================================================================================== rendering
def render_actor_observation(obs: Any, last_result: dict | None = None, max_events: int = 25,
                             payload_chars: int = 200) -> str:
    """Render either a `Runner` observation dict (``time``, ``principal``, module summaries, ``events``) or a
    `WorldServer` `Observation` (object or dict with ``summary``) as text, without the tool list."""
    if hasattr(obs, "summary") or (isinstance(obs, dict) and "summary" in obs):
        o = obs if isinstance(obs, dict) else obs.to_dict() if hasattr(obs, "to_dict") else obs.__dict__
        o = dict(o)
        if last_result is not None and not o.get("last_result"):
            o["last_result"] = last_result
        o["tools_changed"] = False
        return R.render_observation(o, max_events=max_events, payload_chars=payload_chars, show_tools=False)
    o = {"time": obs.get("time", 0), "principal": obs.get("principal"), "last_result": last_result,
         "summary": {k: v for k, v in obs.items() if k not in SKIP_KEYS},
         "events": obs.get("events") or [], "dropped": obs.get("events_dropped", 0), "tools_changed": False}
    return R.render_observation(o, max_events=max_events, payload_chars=payload_chars, show_tools=False)


def render_tool_list(tools: list[dict], budget_chars: int = 9000, priority: tuple = (), description_chars: int = 70) -> str:
    """Compact tool list within a character budget. Full lines (signature + short description) for `priority`
    tools first, then the others alphabetically; when the budget is reached the remaining tools are listed by
    name only, grouped by host (their signatures can be requested by calling them: errors name the arguments)."""
    by = {t["name"]: t for t in tools}
    order = [n for n in priority if n in by] + sorted(n for n in by if n not in set(priority))
    full, rest, used = [], [], 0
    for n in order:
        line = R.render_tool(by[n], description_chars)
        if used + len(line) + 1 <= budget_chars:
            full.append(line)
            used += len(line) + 1
        else:
            rest.append(n)
    out = sorted(full)
    if rest:
        groups: dict[str, list[str]] = {}
        for n in sorted(rest):
            h, _, e = n.partition(".")
            groups.setdefault(h, []).append(e)
        out.append(f"(+{len(rest)} more tools, names only:)")
        out += [f"{h}.{{{', '.join(es)}}}" for h, es in sorted(groups.items())]
    return "\n".join(out)


# ============================================================================================== policy
class LLMPolicy(Policy):
    name, kind = "llm", "llm"

    def __init__(self, model: str = "Qwen/Qwen3-1.7B", device: str = "cuda", max_new_tokens: int = 256,
                 temperature: float = 0.0, history: int = 6, system_prompt: str | None = None, goal: str = "",
                 backend: Any = None, tool_chars: int = 9000, max_events: int = 25, payload_chars: int = 200,
                 record_prompts: bool = False, seed: int = 0):
        self.model_name = getattr(backend, "name", model)
        self._backend, self._backend_args = backend, (model, device)
        self.max_new_tokens, self.temperature, self.k_hist = max_new_tokens, temperature, history
        self.system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        self.goal = goal
        self.tool_chars, self.max_events, self.payload_chars = tool_chars, max_events, payload_chars
        self.record_prompts, self.seed = record_prompts, seed
        self.name = f"llm:{self.model_name}"
        self.transcript: list[dict] = []
        self.stats = {"calls": 0, "prompt_tokens": 0, "gen_tokens": 0, "parse_failures": 0, "waits": 0,
                      "truncated_prompts": 0, "gen_s": 0.0}
        self._reset_episode()

    @property
    def backend(self):
        if self._backend is None:
            self._backend = HFBackend(*self._backend_args)
        return self._backend

    def _reset_episode(self) -> None:
        self.hist: list[dict] = []
        self.known_tools: set[str] | None = None
        self.last_result: dict | None = None
        self.turn = 0

    def reset(self, actor, rng):
        super().reset(actor, rng)
        self._reset_episode()

    def set_goal(self, text: str) -> None:
        self.goal = text

    # ------------------------------------------------------------------ prompt
    def build_messages(self, obs: Any, tools: list[dict]) -> tuple[list[dict], list[str]]:
        tools = [t for t in tools if not t["name"].startswith("admin.")]
        names = {t["name"] for t in tools}
        new = sorted(names - self.known_tools) if self.known_tools is not None else []
        self.known_tools = names
        recent = [h["action"]["tool"] for h in reversed(self.hist) if h.get("action")]
        parts = []
        if self.goal:
            parts.append(f"Goal: {self.goal}")
        parts.append(f"You are {getattr(self, 'actor', '?')}.")
        if self.hist:
            lines = []
            for h in self.hist[-self.k_hist:]:
                if h.get("action"):
                    a = h["action"]
                    lines.append(f"- {a['tool']} {R.compact_json(a.get('args', {}), 200)} -> "
                                 f"{R.render_result(h.get('result'), 240) if h.get('result') else 'no result'}")
                else:
                    lines.append(f"- (no action: {h.get('note', 'wait')})")
            parts.append("Your recent actions:\n" + "\n".join(lines))
        parts.append("Observation:\n" + render_actor_observation(obs, self.last_result, self.max_events,
                                                                 self.payload_chars))
        if new:
            parts.append("Newly available tools: " + ", ".join(new))
        parts.append(f"Tools ({len(tools)}):\n" + render_tool_list(tools, self.tool_chars, tuple(new) + tuple(recent)))
        parts.append("Your next action (one JSON tool call):")
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": "\n\n".join(parts)}], new

    # ------------------------------------------------------------------ acting
    def act(self, obs, tools):
        msgs, new = self.build_messages(obs, tools)
        g = self.backend.generate(msgs, self.max_new_tokens, self.temperature,
                                  None if self.seed is None else self.seed * 100_003 + self.turn)
        self.turn += 1
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += g.prompt_tokens
        self.stats["gen_tokens"] += g.gen_tokens
        self.stats["truncated_prompts"] += int(g.truncated)
        self.stats["gen_s"] += g.wall_s
        a = R.parse_action(g.text)
        rec = {"turn": self.turn - 1, "t": (obs.get("time") if isinstance(obs, dict) else getattr(obs, "time", None)),
               "prompt_hash": prompt_hash(msgs), "prompt_chars": sum(len(m["content"]) for m in msgs),
               "prompt_tokens": g.prompt_tokens, "gen_tokens": g.gen_tokens, "text": g.text, "new_tools": new}
        if self.record_prompts:
            rec["messages"] = msgs
        action = None
        if "error" in a:
            self.stats["parse_failures"] += 1
            rec["parse_error"] = a["error"]
            self.hist.append({"action": None, "note": f"unparseable reply ({a['error']['message'][:80]})"})
        elif a["tool"] in ("wait", "noop", "none"):
            self.stats["waits"] += 1
            rec["wait"] = True
            self.hist.append({"action": None, "note": "wait"})
        else:
            action = {"tool": a["tool"], "args": a.get("args", {})}
            rec["action"] = action
            self.hist.append({"action": action, "result": None})
        self.transcript.append(rec)
        if action is None:
            self.last_result = None
        return action

    def observe_result(self, action, result):
        self.last_result = result
        if self.hist and self.hist[-1].get("action") is action:
            self.hist[-1]["result"] = result
        if self.transcript and self.transcript[-1].get("action") is action:
            self.transcript[-1]["result"] = {k: result.get(k) for k in ("ok", "error", "tool")}

    def usage(self) -> dict:
        return {"model": self.model_name, **self.stats}
