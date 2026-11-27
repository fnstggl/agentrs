import json
import sys

from superlearner.cam_unified.semantic.grammar import ToolGrammar, GrammarOptions, encode_value, GrammarError, holes_of
from superlearner.cam_unified.interfaces import ToolSchema

R = "/workspace/agentrs/runs/cam_unified/wave2/A2/demo_init"
for depth in (2, 3, 4):
    opts = GrammarOptions(max_string_tokens=1536, max_action_tokens=3072, untyped_depth=depth)
    ok = bad = 0
    errs = {}
    per_tool = {}
    for split in ("train", "held"):
        try:
            f = open(f"{R}/{split}/episodes.jsonl")
        except FileNotFoundError:
            continue
        for line in f:
            ep = json.loads(line)
            tt = ep["tool_table"]
            tools = {t["name"]: ToolSchema.from_dict(t) for t in list(tt.values())[0]}
            for d in ep["decisions"]:
                a = d["action"]
                if a["tool"] not in tools:
                    continue
                g = ToolGrammar(tools[a["tool"]], opts)
                if not g.generable:
                    continue
                cs, hs, fl = [], [], {}
                try:
                    encode_value(g.root, a["args"], opts, cs, hs, fl, None)
                    ok += 1
                    per_tool.setdefault(a["tool"], [0, 0])[0] += 1
                except GrammarError as e:
                    bad += 1
                    per_tool.setdefault(a["tool"], [0, 0])[1] += 1
                    errs.setdefault(a["tool"], str(e)[:160])
    print(depth, "ok", ok, "unrepresentable", bad, errs, {k: v for k, v in per_tool.items() if v[1]})
    if "code.run" in per_tool:
        print("   code.run", per_tool["code.run"])
