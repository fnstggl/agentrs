"""Prompts for the frozen language model. They describe the interface and the situation; they never say how to
reach a goal. Identifiers are shown as aliases (``ACCT1``) so that the same situation in two worlds gives the same
prompt (and the same cached answer), and so that an invented identifier is caught before it reaches the world.
"""
from __future__ import annotations

import json

PROGRAM_HELP = (
    'A program is JSON: {"plan": "<one or two sentences>", "steps": [STEP, ...]}.\n'
    'STEP is one of:\n'
    '  {"call": "tool.name", "args": {...}, "as": "x"}        call a tool; the result is bound to x\n'
    '  {"wait": seconds}                                      let simulated time pass\n'
    '  {"if": COND, "then": [STEP], "else": [STEP]}\n'
    '  {"repeat": n, "do": [STEP]}   {"foreach": "$x.items", "as": "it", "do": [STEP]}\n'
    '  {"code": "def main(input, caller): ...", "input": ..., "as": "y"}   run your own Python in the sandbox\n'
    '  {"use": "your_tool", "args": {...}}                    call a tool you defined earlier\n'
    '  {"define_tool": {"name": "...", "description": "...", "params": {...}, "steps": [STEP]}}\n'
    '  {"subgoal": "text"}  {"note": "text"}  {"stop": "reason"}\n'
    'Values: JSON literals; "$x.field" refers to a field of an earlier result (for example "$x.account"); '
    '{"expr": ["mul", "$x.balance", 0.5]} computes a number.\n'
    'COND: {"ok": "$x"} | {"exists": "$x.field"} | {"cmp": ["$x.n", ">", 3]} | {"not": COND} | {"all": [COND]} | {"any": [COND]}.')

SYSTEM_PROPOSE = (
    "You are an agent living in a simulated world. You act only through the tools you are shown. "
    "Identifiers such as ACCT1 or ORG2 are the names of things you have already observed; use only those, or "
    "refer to results of earlier steps of your program with \"$x.field\". If a step fails, the program stops and "
    "you will see why.\n" + PROGRAM_HELP + "\nReply with one JSON program and nothing else.")

SYSTEM_GEN = (
    "You write the value of ONE argument of a tool call in a simulated world. Reply with the JSON value only "
    "(a JSON string for text, a JSON object or array where the argument needs one). Follow exactly the formats "
    "and calling conventions documented in the tool descriptions you are shown; the value will be used by those "
    "tools as it is. Write complete, working content, never placeholders. No explanations.")


def compact(x, n: int = 300) -> str:
    s = json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


def propose_messages(goal: str, elapsed: str, entities: dict, recent: list[str], feedback: str, tools_full: list[str],
                     tools_brief: list[str], own_tools: list[str], state: dict, plan: str = "",
                     docs: list[str] | None = None, max_steps: int = 8) -> list[dict]:
    parts = [f"Goal: {goal}", f"Time: {elapsed}"]
    if plan:
        parts.append(f"Your current plan: {plan}")
    parts.append("Your situation: " + compact(state, 700))
    if entities:
        parts.append("Things you have observed (by kind): " + compact(entities, 900))
    if recent:
        parts.append("Your recent actions and their outcomes:\n" + "\n".join(recent))
    if feedback:
        parts.append("Last failure: " + feedback[:500])
    if docs:
        parts.append("Documents you have read:\n" + "\n".join("- " + d[:500] for d in docs))
    parts.append("Relevant tools (full details):\n" + "\n".join(tools_full))
    if tools_brief:
        parts.append("Other tools that may help:\n" + "\n".join(tools_brief))
    if own_tools:
        parts.append("Tools you made:\n" + "\n".join(own_tools))
    parts.append(f"Write the next program (1 to {max_steps} steps).")
    return [{"role": "system", "content": SYSTEM_PROPOSE}, {"role": "user", "content": "\n\n".join(parts)}]


def gen_messages(goal: str, tool_doc: str, param: str, schema: dict, related: list[str], chosen: dict,
                 focus: str | None, repair: tuple | None = None) -> list[dict]:
    frag = {k: schema[k] for k in ("type", "description", "maxLength", "pattern", "items", "additionalProperties",
                                   "enum", "minItems", "maxItems") if k in schema}
    parts = [f"Goal: {goal}", "Tool being called:\n" + tool_doc]
    if related:
        parts.append("Related tools:\n" + "\n".join(related))
    if chosen:
        parts.append("Arguments already chosen: " + compact(chosen, 400))
    if focus:
        parts.append("Material this value should be based on:\n" + focus[:1200])
    if repair:
        prev, failure = repair
        parts.append("Your previous value for this argument:\n" + prev[:1800] + "\n\nWhat went wrong when it was used:\n"
                     + failure[:600] + "\n\nWrite a corrected value.")
    parts.append(f"Write the value of the argument `{param}` (schema: {compact(frag, 400)}).")
    ap = schema.get("additionalProperties")
    if schema.get("type") == "object" and isinstance(ap, dict):
        parts.append('Format: one JSON object. Where an entry\'s content is itself JSON, write it as a nested JSON '
                     'object, not as an escaped string; other content (such as source code) is a JSON string. '
                     'Example of the format only: {"settings.json": {"level": 1}, "main.py": "print(1)\\n"}')
    return [{"role": "system", "content": SYSTEM_GEN}, {"role": "user", "content": "\n\n".join(parts)}]


def gen_keys_messages(base: list[dict], param: str) -> list[dict]:
    """First step of writing an object of texts (for example files): which entries does it need?"""
    u = base[1]["content"].rsplit("\n\nWrite the value of the argument", 1)[0]
    u += (f"\n\nThe argument `{param}` is an object of named text entries. First decide which entries it needs. "
          "Reply with a JSON list of the entry names only, in the order they should be written.")
    return [base[0], {"role": "user", "content": u}]


def gen_entry_messages(base: list[dict], param: str, names: list[str], done: dict, name: str) -> list[dict]:
    """Second step: the content of one entry, as plain text."""
    u = base[1]["content"].rsplit("\n\nWrite the value of the argument", 1)[0]
    u += f"\n\nThe argument `{param}` will have these entries: {', '.join(names)}."
    for k, v in done.items():
        u += f"\n\nEntry `{k}` (already written):\n{v[:1500]}"
    u += (f"\n\nWrite the complete content of the entry `{name}`, consistent with the entries already written and "
          "with the formats and calling conventions documented above. Reply with the content only: plain text "
          "exactly as it should be stored (no surrounding quotes, no code fences, no commentary).")
    sys_ = ("You write the content of one named text entry (for example one file) of a tool-call argument in a "
            "simulated world. Write complete, working content, never placeholders. Reply with the content only.")
    return [{"role": "system", "content": sys_}, {"role": "user", "content": u}]
