"""S-6b: the backbone proposer's prompt and normalisation (PI-09, CU-12; agent.py AgentProposer / compact_tool /
normalise_proposal, predict/candidates.py parser).

Part A (CPU, no model): for the open-ended computer's real tool descriptors (SW and MECH objectives):
  * every argument name of every tool is shown exactly (no marker in the name), required ones say so;
  * nested object arguments show their properties (one level);
  * schema information the compact form drops is listed (INFO: enum options beyond 12, patterns, length bounds,
    formats, anyOf / oneOf, defaults) and whether the validator would refuse values the prompt cannot reveal;
  * the instruction block is constant (identical for every objective) and names no tool, file, objective word
    or strategy of a task;
  * normalise_proposal: a "name*" marker is undone only for a property the tool has, unknown names are removed
    only under additionalProperties false, numbers written as text are coerced only when the schema says
    number / integer / boolean, nothing else is invented or changed.
Part B (with --real P.json, the output of `python -m cam_live.eval.proposal_probe`): every prompt lists exactly the
task's permitted available tools, fits the backbone window, the instruction block is byte-identical in every
round of every objective, and the backbone's valid actions are reported.
"""
import json
import os
import shutil
import sys
import tempfile

import _g2d as C
from cam_live import agent as A
from cam_live.contracts import SimClock
from cam_live.eval.openended import OpenEndedComputer
from cam_live.predict.candidates import H_TOOLS, parse_actions, validate_schema

work = tempfile.mkdtemp(prefix="g2d_prompt_", dir=os.environ.get("G2D_TMP") or None)
os.chmod(work, 0o755)                   # the sandbox uid must traverse to its workspace (else sandbox.* unavailable)
oc = OpenEndedComputer(os.path.join(work, "computer"), SimClock(1_700_000_000.0))
fam_tools = {}
for fam, prob in (("SW", "rle"), ("MECH", "7")):
    env, policy, verifier, task = oc.open(fam, prob, 1 if fam == "SW" else 700, f"obj_{fam}", max_decisions=10)
    allowed = set(task.interfaces or [])
    tools = [t for t in env.discover_tools() if t.available and (not allowed or t.modality in allowed
                                                                  or t.modality == "internal")]
    fam_tools[fam] = (tools, task)
    C.info(f"{fam}: permitted available tools", [t.name for t in tools])
    C.info(f"{fam}: permitted but unavailable", {t.name: t.unavailable_reason for t in env.discover_tools()
                                                  if not t.available})

# ---- A1 argument names exact, required marked, nested properties shown
bad_names, bad_req, nested_missing, dropped = [], [], [], {}
for fam, (tools, task) in fam_tools.items():
    for t in tools:
        ct = A.compact_tool(t)
        sch = t.arg_schema if isinstance(t.arg_schema, dict) else {}
        props = sch.get("properties") or {}
        req = set(sch.get("required") or [])
        if t.name == "program":
            continue
        if set(ct["args"]) != set(props):
            bad_names.append((fam, t.name, sorted(ct["args"]), sorted(props)))
        for k, txt in ct["args"].items():
            if k.endswith("*"):
                bad_names.append((fam, t.name, k))
            if (k in req) != txt.endswith(", required"):
                bad_req.append((fam, t.name, k, txt))
            s = props.get(k) or {}
            if isinstance(s, dict) and s.get("type") == "object" and isinstance(s.get("properties"), dict):
                for kk in s["properties"]:
                    if kk not in txt:
                        nested_missing.append((fam, t.name, k, kk))
            # schema keywords the compact form does not carry
            def walk(x, path):
                if not isinstance(x, dict):
                    return
                for kw in ("pattern", "minLength", "maxLength", "format", "anyOf", "oneOf", "allOf", "default",
                           "minItems", "maxItems", "multipleOf", "uniqueItems", "$ref", "not"):
                    if kw in x:
                        dropped.setdefault(f"{fam}:{t.name}.{path}", []).append(kw)
                if "enum" in x and len(x["enum"]) > 12:
                    dropped.setdefault(f"{fam}:{t.name}.{path}", []).append(f"enum>{12}({len(x['enum'])})")
                for kk, vv in (x.get("properties") or {}).items():
                    if path.count(".") >= 1:                 # deeper than one nested level: not shown at all
                        dropped.setdefault(f"{fam}:{t.name}.{path}.{kk}", []).append("not_shown_depth>1")
                    walk(vv, f"{path}.{kk}")
                if isinstance(x.get("items"), dict):
                    walk(x["items"], f"{path}[]")
            walk(s, k)
C.check("every argument name is shown exactly (no marker, no missing / extra name) for every permitted tool",
        not bad_names, bad_names[:5])
C.check("required arguments, and only they, say 'required'", not bad_req, bad_req[:5])
C.check("nested object arguments show their properties (one level)", not nested_missing, nested_missing[:5])
C.info("schema keywords the compact form drops (argument: keywords)", dropped)

# ---- A2 constant instruction block, no task content
txts = {}
for fam, (tools, task) in fam_tools.items():
    p = A.AgentProposer(model=None, k=4).prompt(tools, "obs", objective=task.objective)
    txts[fam] = p.split("OBJECTIVE")[0]
words = set()
for fam, (tools, task) in fam_tools.items():
    words |= {t.name for t in tools} | {"solution", "model.py", "solve", "fit", "regression", "discord", "member",
                                         "rle", "hidden", "test"}
leak = [w for w in words if w.lower() in txts["SW"].lower()]
C.check("the instruction block is identical for both objectives", txts["SW"] == txts["MECH"])
C.check("the instruction block names no tool, file or task-specific word", not leak, leak)
C.info("instruction block (verbatim)", txts["SW"])

# ---- A3 normalisation
sw_tools = {t.name: t for t in fam_tools["SW"][0]}
wf = sw_tools.get("sandbox.write_file")
notes = {}
o = A.normalise_proposal({"operation": "sandbox.write_file", "arguments": {"path*": "solution.py",
                                                                          "content*": "def solve(s):\n  return s"}},
                         sw_tools, notes)
C.check("a 'name*' marker on a real property is undone, value byte-identical, and logged",
        o["arguments"] == {"path": "solution.py", "content": "def solve(s):\n  return s"}
        and notes.get("unmarked:sandbox.write_file.path") == 1, (o, notes))
notes = {}
o = A.normalise_proposal({"operation": "sandbox.write_file", "arguments": {"bogus*": 1, "path": "a.py",
                                                                          "content": "x"}}, sw_tools, notes)
addl = (wf.arg_schema or {}).get("additionalProperties", True)
C.check("a marker on a name the tool does not have is not turned into a property (removed only if the schema "
        "forbids extra properties)", ("bogus" not in o["arguments"]) and (("bogus*" in o["arguments"]) == (addl is not False)),
        (o, notes, f"additionalProperties={addl}"))
num_tool = next((t for t in fam_tools["SW"][0] + fam_tools["MECH"][0]
                 if any(isinstance(v, dict) and v.get("type") in ("integer", "number")
                        for v in ((t.arg_schema or {}).get("properties") or {}).values())), None)
if num_tool is not None:
    k_num, s_num = next((k, v) for k, v in (num_tool.arg_schema.get("properties") or {}).items()
                        if isinstance(v, dict) and v.get("type") in ("integer", "number"))
    tm = {num_tool.name: num_tool}
    notes = {}
    o = A.normalise_proposal({"operation": num_tool.name, "arguments": {k_num: " 7 "}}, tm, notes)
    o2 = A.normalise_proposal({"operation": num_tool.name, "arguments": {k_num: "seven"}}, tm, {})
    C.check(f"numbers written as text are coerced only when the schema type is numeric ({num_tool.name}.{k_num})",
            o["arguments"][k_num] == 7 and o2["arguments"][k_num] == "seven", (o, o2))
str_tool = wf
notes = {}
o = A.normalise_proposal({"operation": "sandbox.write_file", "arguments": {"path": "3", "content": "12"}},
                         sw_tools, notes)
C.check("string arguments are never coerced", o["arguments"] == {"path": "3", "content": "12"} and not notes,
        (o, notes))
res = parse_actions(json.dumps(A.normalise_proposal({"operation": "sandbox.write_file", "path": "solution.py",
                                                     "content": "def solve(s): return s"}, sw_tools, {})),
                    fam_tools["SW"][0])
C.check("argument names written at the top level are moved into arguments and the action validates",
        len(res.actions) == 1 and res.actions[0].arguments.get("path") == "solution.py", res.reasons[:3])

# ---- Part B: the real-model probe output
if "--real" in sys.argv:
    d = json.load(open(sys.argv[sys.argv.index("--real") + 1]))
    win = d.get("prompt_window_tokens")
    instr, all_listed, fits, n_rounds, ops, sums = set(), True, True, 0, {}, {}
    for ob in d["objectives"]:
        permitted = [t["name"] for t in ob["tools"] if t["available"] and (not ob["interfaces"] or True)]
        sums[ob["family"]] = ob.get("summary")
        for r in ob["rounds"]:
            n_rounds += 1
            p = r["prompt"]
            body = p.split("OBJECTIVE")[0]
            body = body[body.find("Write up to"):]
            instr.add(body)
            listed = r["prompt_tools"]
            rec = r["agent_recorded"]
            if rec.get("tools_not_listed") or rec.get("tools_unavailable") or len(listed) != rec.get("tools_permitted"):
                all_listed = False
            if win and (r["prompt_tokens"] or 0) > win:
                fits = False
            for a in r["backbone_valid"]:
                ops[a["op"]] = ops.get(a["op"], 0) + 1
    C.check("real model: every prompt listed every permitted available tool (none dropped, none unavailable)",
            all_listed and n_rounds > 0, n_rounds)
    C.check("real model: every prompt fits the backbone's window", fits,
            [(ob["family"], [r["prompt_tokens"] for r in ob["rounds"]]) for ob in d["objectives"]] + [win])
    C.check("real model: the instruction block is byte-identical in every round of every objective", len(instr) == 1,
            len(instr))
    C.info("real model: valid backbone actions by operation", ops)
    # the SW files the backbone wrote, scored offline by the objective's own hidden-test verifier (never shown to it)
    from cam_live.eval.openended import HiddenTestVerifier, SOLUTION
    scores = []
    for ob in d["objectives"]:
        if ob["family"] != "SW":
            continue
        for r in ob["rounds"]:
            best = None
            for text in r["raw"] or []:
                for ln in str(text).splitlines():
                    try:
                        o = json.loads(ln, strict=False)
                    except Exception:  # noqa: BLE001
                        continue
                    for x in (o if isinstance(o, list) else [o]):
                        if not isinstance(x, dict):
                            continue
                        steps = x.get("steps") if isinstance(x.get("steps"), list) else [x]
                        for st_ in steps:
                            args = st_.get("arguments") or {}
                            path = args.get("path") or args.get("path*") or ""
                            if st_.get("operation") == "sandbox.write_file" and str(path).endswith(SOLUTION):
                                ws = os.path.join(work, f"sw_{len(scores)}")
                                os.makedirs(ws, exist_ok=True)
                                os.chmod(ws, 0o755)
                                with open(os.path.join(ws, SOLUTION), "w") as f:
                                    f.write(str(args.get("content") or args.get("content*") or ""))
                                v = HiddenTestVerifier(ws, ob["problem"], 1, SimClock(0.0))
                                sc = v.score()
                                best = sc["pass_fraction"] if best is None else max(best, sc["pass_fraction"])
            scores.append(best)
    C.info("real model: SW solution.py written per round, hidden-test pass fraction (offline, 60 tests)", scores)
    C.info("real model: per-objective summary", sums)
shutil.rmtree(work, ignore_errors=True)
sys.exit(C.finish())
