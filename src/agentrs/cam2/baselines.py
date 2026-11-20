"""Baseline B1: the frozen LLM acting alone with exactly CAM's tool access.

Same environments, same tool schemas, same observation elements (the list CAM sees, in the same order), same
feedback text, same schema validation and step limit. The LLM reads a rendering of all of it and writes one action
as JSON; the action is converted to the same ``ToolCall`` CAM would emit and validated by the environment (an
unparseable or schema-invalid answer is an invalid step, exactly as for CAM). No trainable parameters, no updates.
Token usage is counted so comparisons can account for LLM inference cost.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import numpy as np

from agentrs.cam2.tools import Observation, ToolCall, ToolSpec

MAX_ELEMENTS = 80


def render_tools(tools: list) -> str:
    out = []
    for t in tools:
        args = []
        for a in t.args:
            if a.type == "element":
                args.append(f"target: element number{(' (' + ', '.join(a.roles) + ')') if a.roles else ''}")
            elif a.type == "enum":
                args.append(f"value: one of {list(a.options)}")
            elif a.type == "text":
                args.append(f"value: text ({a.description})")
            else:
                args.append(f"value: JSON object ({a.description[:400]})")
        out.append(f"- {t.name}({'; '.join(args)}): {t.description[:200]}")
    return "\n".join(out)


def render_tools_compact(tools: list) -> str:
    """Large tool sets (AppWorld: 457 APIs): one line per tool with its parameter names; full documentation is
    available to the agent through the environment's own documentation tools (``api_docs.*``), as for CAM."""
    out = []
    for t in tools:
        ga = t.gen_arg
        params = ""
        if ga is not None and ga.json_schema:
            params = ", ".join(json.loads(ga.json_schema).get("properties", {}))
        out.append(f"- {t.name}({params})")
    return "\n".join(out)


def render(obs: Observation, tools: list, history: list, max_tools_chars: int = 6000) -> str:
    els = obs.elements[:MAX_ELEMENTS]
    lines = [f"Task: {obs.goal}", "", "Current state (element number: description):"]
    for i, e in enumerate(els):
        lines.append(f"[{i}] {e.text}{'' if e.targetable else '  (not targetable)'}")
    if obs.text:
        lines += ["", f"Feedback: {obs.text[-1500:]}"]
    if history:
        lines += ["", "Your previous actions: " + " | ".join(history[-5:])]
    tool_text = render_tools(tools) if len(tools) <= 60 else render_tools_compact(tools)
    lines += ["", "Tools:", tool_text[:max_tools_chars], "",
              'Reply with exactly one action as JSON: {"tool": "<name>", "target": <element number or null>, '
              '"value": <text, option or JSON arguments, or null>}']
    return "\n".join(lines)


def parse_action(text: str, obs: Observation, tools: list) -> tuple[ToolCall, Optional[str]]:
    """Convert the LLM's JSON answer into a ToolCall (no repair beyond JSON extraction)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return ToolCall(tool="<unparseable>"), "no JSON object"
    s = m.group(0)
    try:
        d = json.loads(s)
    except Exception:  # noqa: BLE001  - try the shortest balanced prefix (models often append a second object)
        depth, end = 0, None
        for i, ch in enumerate(s):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                end = i + 1
                break
        try:
            d = json.loads(s[:end]) if end else None
        except Exception:  # noqa: BLE001
            d = None
        if d is None:
            return ToolCall(tool="<unparseable>"), "invalid JSON"
    if not isinstance(d, dict):
        return ToolCall(tool="<unparseable>"), "not an object"
    name = str(d.get("tool", ""))
    spec: Optional[ToolSpec] = next((t for t in tools if t.name == name), None)
    if spec is None:
        return ToolCall(tool=name or "<missing>"), "unknown tool"
    els = obs.elements[:MAX_ELEMENTS]
    eid = None
    tgt = d.get("target")
    if spec.element_arg is not None and isinstance(tgt, (int, float)) and 0 <= int(tgt) < len(els):
        eid = els[int(tgt)].eid
    v = d.get("value")
    enum = text = None
    if spec.enum_arg is not None:
        enum = None if v is None else str(v)
    ga = spec.gen_arg
    if ga is not None:
        text = json.dumps(v) if (ga.type == "json" and isinstance(v, (dict, list))) else (None if v is None else str(v))
        if ga.type == "json" and text is None:
            text = "{}"
    return ToolCall(tool=name, element=eid, enum=enum, text=text), None


def run_llm_agent(llm, pool, tasks: list, max_new: int = 96, greedy: bool = True, log: Optional[list] = None,
                  max_ctx: int = 3500, max_tools_chars: int = 6000) -> dict:
    """Evaluate B1 on ``tasks`` with the shared worker pool (all ready workers batched into one generate call)."""
    W = pool.n
    S = [dict(active=False) for _ in range(W)]
    it = iter(tasks)
    eps, steps, invalid, parse_fail, errors = [], 0, 0, 0, 0
    g0 = llm.stats["gen_tokens"]
    t0 = time.time()
    p_tokens = 0

    def launch(i, retry=False):
        t = S[i]["task"] if retry and S[i].get("tries", 0) < 3 else next(it, None)
        if t is None:
            S[i] = dict(active=False)
            return
        S[i] = dict(active=True, wait=False, task=t, hist=[], k=0, tries=(S[i].get("tries", 0) + 1) if retry else 0)
        pool.send(i, "reset", t)

    for i in range(W):
        launch(i)
    while any(s["active"] for s in S):
        ready = [i for i in range(W) if S[i]["active"] and S[i]["wait"]]
        if ready:
            prompts = [render(S[i]["obs"], S[i]["tools"], S[i]["hist"], max_tools_chars) for i in ready]
            for p in prompts:
                p_tokens += len(llm.tok(p, add_special_tokens=False)["input_ids"])
            outs = llm.generate_text(prompts, max_new=max_new, max_ctx=max_ctx)
            for i, txt in zip(ready, outs):
                call, perr = parse_action(txt, S[i]["obs"], S[i]["tools"])
                parse_fail += perr is not None
                S[i]["hist"].append(txt.strip().replace("\n", " ")[:160])
                S[i]["wait"] = False
                pool.send(i, "step", call)
                if log is not None and len(log) < 200:
                    log.append({"task": S[i]["task"].key(), "k": S[i]["k"], "answer": txt[:300], "parse_error": perr})
        for i, cmd, st, v in pool.poll(timeout=0.001 if ready else 2.0):
            s = S[i]
            if st != "ok":
                errors += 1
                launch(i, retry=True)
                continue
            if cmd == "reset":
                s["obs"], s["tools"] = v
                s["wait"] = True
                continue
            steps += 1
            s["k"] += 1
            invalid += int(bool(v.info.get("invalid")))
            s["obs"] = v.obs
            if v.terminated or v.truncated:
                eps.append({"task": s["task"].key(), "family": s["task"].family,
                            "success": bool(v.info.get("success", v.reward > 0)),
                            "score": float(v.info.get("score", float(v.reward > 0))), "len": s["k"]})
                launch(i)
            else:
                s["wait"] = True
    fam: dict = {}
    for e in eps:
        fam.setdefault(e["family"], []).append(float(e["success"]))
    succ = [float(e["success"]) for e in eps]
    from agentrs.cam2.trainer import bootstrap_ci

    m, lo, hi = bootstrap_ci(succ)
    return {"label": "B1_llm_only", "n": len(eps), "success": m, "ci95": [lo, hi],
            "score": float(np.mean([e["score"] for e in eps])) if eps else float("nan"),
            "per_family": {k: float(np.mean(v)) for k, v in sorted(fam.items())},
            "invalid_rate": invalid / max(1, steps), "parse_fail_rate": parse_fail / max(1, steps), "env_steps": steps,
            "errors": errors, "llm_gen_tokens": llm.stats["gen_tokens"] - g0, "llm_prompt_tokens": p_tokens,
            "eval_s": time.time() - t0, "episodes": eps}
