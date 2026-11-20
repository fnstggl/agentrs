"""Optional LLM-guided proposals (task statement section 13).

The frozen backbone LLM reads the bounded search space, the parent's spec and a summary of measured results
(successes and failures) and answers with a small JSON edit ``{"changes": {"arch.d": 384, ...}, "why": "..."}``.
The edit is applied to the parent's spec, repaired, and must pass the same ``spec.validate`` admission checks as
every other mutation. The proposer has no access to evaluation tasks, scores beyond the summary it is shown,
checkpoints, the champion pointer or any executable surface; it returns data only. Within the search it is one
more operator (``llm.propose``) whose usefulness the controller measures like any other.
"""
from __future__ import annotations

import copy
import json
import random
import re
from typing import Optional

from agentrs.cam2.search.rules import Rule, mutate_rule
from agentrs.cam2.search.spec import SPACE, SpecError, _check_value, check_schema, repair

MAX_CHANGES = 3


def describe_space() -> str:
    lines = []
    for sec, dims in SPACE.items():
        for k, d in dims.items():
            if d.kind == "choice":
                lines.append(f"{sec}.{k}: one of {list(d.choices)}")
            elif d.kind == "int":
                lines.append(f"{sec}.{k}: integer {int(d.lo)}..{int(d.hi)}")
            elif d.kind in ("float", "logfloat"):
                lines.append(f"{sec}.{k}: number {d.lo:g}..{d.hi:g}")
            elif d.kind == "bool":
                lines.append(f"{sec}.{k}: true/false")
            elif d.kind == "rule":
                lines.append(f"{sec}.{k}: the string \"mutate\" to request a random valid change of the loss expression")
    return "\n".join(lines)


def summarize(pop, k: int = 6) -> str:
    rows = []
    ev = [c for c in pop.cands.values() if c.rung >= 0]
    for c in sorted(ev, key=lambda c: -(c.score(0) or 0))[:k]:
        rows.append(f"score {c.score(0):.3f}: {c.mutation[:160]}")
    fails = [c for c in pop.cands.values() if c.status == "failed"][-3:]
    for c in fails:
        rows.append(f"FAILED ({(c.failure or '')[:80]}): {c.mutation[:120]}")
    return "\n".join(rows) or "(no results yet)"


def flat_spec(spec: dict) -> str:
    out = {}
    for sec, vals in spec.items():
        for k, v in vals.items():
            if k in ("rule", "meta_target"):
                continue
            out[f"{sec}.{k}"] = v
    return json.dumps(out)


def prompt(parent_spec: dict, pop) -> str:
    return ("You are tuning a trainable recurrent agent network (CAM) that learns web tasks by reinforcement learning "
            "next to a frozen language model. Propose ONE small change (at most 3 fields) to the parent configuration "
            "that you expect to raise validation success at the same training budget.\n\n"
            f"Allowed fields and values:\n{describe_space()}\n\n"
            f"Parent configuration:\n{flat_spec(parent_spec)}\n\n"
            f"Measured results so far (validation success, change from its parent):\n{summarize(pop)}\n\n"
            'Answer with only a JSON object: {"changes": {"<field>": <value>, ...}, "why": "<short reason>"}')


def parse_edit(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise SpecError(f"no JSON object in the proposal: {text[:120]!r}")
    try:
        d = json.loads(m.group(0))
    except Exception as e:  # noqa: BLE001
        raise SpecError(f"proposal is not valid JSON: {e}") from e
    ch = d.get("changes")
    if not isinstance(ch, dict) or not ch:
        raise SpecError("proposal has no 'changes' object")
    if len(ch) > MAX_CHANGES:
        raise SpecError(f"proposal changes {len(ch)} fields (max {MAX_CHANGES})")
    return {"changes": ch, "why": str(d.get("why", ""))[:200]}


def apply_edit(spec: dict, edit: dict, rng: random.Random) -> tuple[dict, str]:
    s = copy.deepcopy(spec)
    desc = []
    for key, v in edit["changes"].items():
        if not isinstance(key, str) or key.count(".") != 1:
            raise SpecError(f"bad field name {key!r}")
        sec, k = key.split(".")
        if sec not in SPACE or k not in SPACE[sec]:
            raise SpecError(f"field {key!r} is not in the search space")
        if k == "meta_target":
            raise SpecError("the proposer cannot set meta_target")
        if k == "rule":
            if v != "mutate":
                raise SpecError("learn.rule only accepts \"mutate\"")
            r, how = mutate_rule(Rule.from_json(s["learn"]["rule"]), rng)
            s["learn"]["rule"] = r.to_json()
            desc.append(f"learn.rule {how}")
            continue
        dim = SPACE[sec][k]
        if isinstance(v, str) and dim.kind in ("choice", "int", "float", "logfloat", "bool"):
            # the LLM often quotes scalars ("2.0", "true"); coerce when the field is not a string field
            lv = v.strip().lower()
            if dim.kind == "bool" and lv in ("true", "false"):
                v = lv == "true"
            elif not (dim.kind == "choice" and v in dim.choices):
                try:
                    v = float(v) if any(ch in lv for ch in ".e") else int(v)
                except ValueError:
                    pass
        if dim.kind == "int" and isinstance(v, float) and v.is_integer():
            v = int(v)
        if dim.kind == "choice" and isinstance(v, (int, float)) and not isinstance(v, bool):
            # numeric choices: snap to the nearest allowed value (the LLM often writes 0.1 for 0.10, 500 for 512)
            nums = [c for c in dim.choices if isinstance(c, (int, float)) and not isinstance(c, bool)]
            if nums and v not in dim.choices:
                v = min(nums, key=lambda c: abs(c - v))
        _check_value(sec, k, dim, v)      # type/range check before any cross-field repair touches the value
        desc.append(f"{key}: {s[sec][k]} -> {v}")
        s[sec][k] = v
    s = repair(s)
    if s == spec:
        raise SpecError("proposal does not change the parent")
    check_schema(s)
    return s, "; ".join(desc) + (f"  (why: {edit['why']})" if edit.get("why") else "")


class LLMProposer:
    """``temperature=0`` decodes greedily (deterministic for a given prompt, so repeated proposals for the same
    parent repeat themselves - the setting used in the reported adaptive search); ``temperature>0`` samples, seeded
    from the search's RNG."""

    def __init__(self, llm, max_new: int = 160, temperature: float = 0.0):
        self.llm, self.max_new, self.temperature = llm, max_new, temperature
        self.calls = 0
        self.tokens = 0

    def propose(self, parent_spec: dict, pop, rng: random.Random) -> tuple[dict, str, str]:
        p = prompt(parent_spec, pop)
        g0 = self.llm.stats["gen_tokens"]
        if self.temperature > 0:
            import torch

            ctx, stub = self.llm.chat_ids(p, max_ctx=3500)
            gen = torch.Generator(device=self.llm.device).manual_seed(rng.randrange(2**31))
            text = self.llm.sample_with_prefix(None, [ctx], [stub], max_new=self.max_new, temperature=self.temperature,
                                               gen=gen).texts[0]
        else:
            text = self.llm.generate_text([p], max_new=self.max_new, max_ctx=3500)[0]
        self.calls += 1
        self.tokens += self.llm.stats["gen_tokens"] - g0
        edit = parse_edit(text)
        s, desc = apply_edit(parent_spec, edit, rng)
        return s, desc, text
