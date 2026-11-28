"""cam_live.eval.proposal_probe — what the full-size agent's open-ended proposer writes for an objective.

    python -m cam_live.eval.proposal_probe --plan SW:rle,MECH:7 --rounds 4 --out P.json   (GPU lock)
    python -m cam_live.eval.proposal_probe analyse-run RUN_DIR [OUT.json]    (ST: what a run recorded, CPU)
    python -m cam_live.eval.proposal_probe --family SW --problem rle --rounds 4 --out P.json

Builds the real agent (Qwen3-4B + 50m, substantive settings), opens each objective of the plan in the open-ended
computer (one agent, objectives in order, like the runtime's sequence), and calls ``agent.propose`` ``--rounds``
times per objective exactly as the runtime would (same tools, same digest). It executes nothing.

Recorded per round (change requests S-6, S-6b): the prompt the backbone was actually sent (captured at
``model.generate``: text, characters, tokens under the backbone's tokenizer, sha256), the tools the prompt listed
(decoded from its tool line), the backbone's raw text, the parsed actions, the rejection reasons and the merged
candidate set. Recorded per objective: every discovered tool with its availability and the environment's reason,
the task's permitted interfaces, and whether the run root is traversable by the sandbox user.

Summary per objective (evaluation measures, computed here, never shown to the agent): fraction of rounds with a
backbone-valid action that writes the file the objective names (``sandbox.write_file`` on that path, also as a
program step), that writes any file, that executes a program (``sandbox.python`` / ``sandbox.shell`` /
``sandbox.job_start``), validity (valid / written items), prompt size.

Before running, the probe checks that every tool the task permits is available: a permitted tool the environment
reports unavailable (e.g. the sandbox, when the run root lies below a directory its uid cannot traverse) makes the
measurement meaningless, so the probe stops unless ``--allow-unavailable`` (used to reproduce that condition).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List

CODE_RUN_OPS = ("sandbox.python", "sandbox.shell", "sandbox.job_start")


def _flat(a: Any) -> List[Any]:
    return [a] + [x for s in (a.steps or []) for x in _flat(s)]


def target_file(family: str) -> str:
    from .openended import MODEL, SOLUTION
    return SOLUTION if family == "SW" else MODEL


def classify(actions: List[Any], target: str) -> Dict[str, bool]:
    """What a set of (backbone-valid) actions does, counting program steps."""
    ops = [x for a in actions for x in _flat(a)]
    writes = [x for x in ops if x.operation == "sandbox.write_file"]
    tgt = [x for x in writes if os.path.basename(str(x.arguments.get("path", "")).strip()) == target]
    return {"writes_target_file": bool(tgt), "writes_any_file": bool(writes),
            "executes_program": any(x.operation in CODE_RUN_OPS for x in ops),
            "target_with_content": any(len(str(x.arguments.get("content", "")).strip()) > 0 for x in tgt),
            "requests_operator": any(x.operation == "internal.request_operator" for x in ops)}


def tools_of_prompt(prompt: str) -> List[str]:
    from ..predict.candidates import H_TOOLS
    lines = prompt.splitlines()
    try:
        return [t.get("name") for t in json.loads(lines[lines.index(H_TOOLS) + 1])]
    except (ValueError, IndexError, TypeError, AttributeError):
        return []


def traversable(path: str, uid: int, gid: int) -> bool:
    cur = os.path.dirname(os.path.abspath(path))
    while True:
        st = os.stat(cur)
        if not (st.st_mode & 0o001 or (st.st_uid == uid and st.st_mode & 0o100)
                or (st.st_gid == gid and st.st_mode & 0o010)):
            return False
        if cur == "/":
            return True
        cur = os.path.dirname(cur)


# ----------------------------------------------------------------------------------------------------------------
# ST: what a run of the agent recorded about research, hypotheses, experiments, plans and their revision
# ----------------------------------------------------------------------------------------------------------------
#: words of the evaluation's crude content measure (counted in what CAM wrote; never shown to CAM, never a signal)
HYPOTHESIS_WORDS = ("hypothes", "experiment", "predict", "expect", "test ", "if ", "assum", "plan", "strategy",
                    "measure", "evidence", "lesson", "revise", "because")
_STOP = set("the a an and or of to in for on with by is are be this that it as at from will can we i our".split())


def _jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def _words(text: str) -> set:
    import re
    return {w for w in re.findall(r"[a-z][a-z0-9]{2,}", str(text).lower()) if w not in _STOP}


def _strings(x: Any) -> List[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _strings(v)]
    return []


def analyse_run(run_dir: str) -> Dict[str, Any]:
    """Counts over a run directory written by ``LiveRuntime`` (decisions.jsonl, the internal executor's notebook,
    operator requests, experiments.jsonl, action results in events.jsonl). Evaluation measures only:

    * notes / requests: executed, their text length, CAM's own tags, the share containing ``HYPOTHESIS_WORDS``;
    * expectations: chosen actions that carried an ``expected`` text (a prediction written with the action);
    * coherence: a chosen action whose argument strings reuse a token (>= 6 characters, not in the task's resources)
      that first appeared in an EARLIER decision's observed result; repeated executions of an identical action;
    * revisions: a note sharing >= 30 % of its content words (Jaccard) with an earlier note; after a refused /
      failed action, a later chosen action of the same operation with different arguments;
    * generation: items written, valid, per-sample truncation, proposed operations, chosen provenance."""
    import collections
    R = os.path.abspath(run_dir)
    decs = [r.get("data", {}).get("decision", {}) for r in _jsonl(os.path.join(R, "decisions.jsonl"))]
    decs = [d for d in decs if d.get("chosen")]
    events = _jsonl(os.path.join(R, "events.jsonl"))
    results = {e["data"].get("decision_id"): e["data"] for e in events if e.get("type") == "action_result"}
    obs_text = {}
    for e in events:
        if e.get("type") == "observation":
            d = e.get("data", {})
            o = d.get("observation", d)
            obs_text[o.get("obs_id", "")] = str(o.get("text", ""))
    try:
        task = json.load(open(os.path.join(R, "task.json")))
    except (OSError, ValueError):
        task = {}
    resource_strings = set(_strings(task.get("resources", {})))
    out: Dict[str, Any] = {"run_dir": R, "decisions": len(decs)}
    chosen_ops = collections.Counter(d["chosen"]["operation"] for d in decs)
    out["chosen_operations"] = dict(chosen_ops)
    out["chosen_provenance"] = dict(collections.Counter(d["chosen"].get("provenance", "").split("+")[0]
                                                        for d in decs))
    gen = [d.get("diagnostics", {}).get("generation") or {} for d in decs]
    items = sum(int(g.get("n_items") or 0) for g in gen)
    valid = sum(int(g.get("n_backbone_valid") or 0) for g in gen)
    samples = [s for g in gen for s in (g.get("per_sample") or [])]
    proposed = collections.Counter(c["action"]["operation"] for d in decs for c in d.get("candidates", [])
                                   if "backbone" in str(c["action"].get("provenance", "")))
    out["generation"] = {
        "items": items, "valid": valid, "validity": (valid / items) if items else None,
        "mean_valid_per_decision": valid / max(1, len(decs)),
        "samples": len(samples) or None,
        "samples_last_line_rejected": sum(int(s.get("last_line_rejected", 0)) for s in samples) if samples else None,
        "decisions_with_not_json": sum(1 for g in gen if (g.get("rejection_reasons") or {}).get("not_json")),
        "proposed_backbone_operations_offered": dict(proposed),
        "mean_prompt_tokens": (sum(int(g.get("prompt_tokens") or 0) for g in gen) / max(1, len(gen))),
        "mean_memory_chars": (sum(int(g.get("memory_chars") or 0) for g in gen) / max(1, len(gen)))}
    # executed own records
    ok = lambda d: (results.get(d["decision_id"]) or {}).get("status") == "ok"          # noqa: E731
    notes = [d for d in decs if d["chosen"]["operation"] == "internal.note" and ok(d)]
    reqs = [d for d in decs if d["chosen"]["operation"] == "internal.request_operator" and ok(d)]
    note_texts = [str(d["chosen"]["arguments"].get("text", "")) for d in notes]
    hyp = lambda t: any(w in t.lower() for w in HYPOTHESIS_WORDS)                         # noqa: E731
    out["notes"] = {"executed": len(notes), "chars": [len(t) for t in note_texts],
                    "with_hypothesis_words": sum(1 for t in note_texts if hyp(t)),
                    "tags": dict(collections.Counter(t for d in notes for t in d["chosen"]["arguments"].get("tags", [])
                                                     if isinstance(t, str))),
                    "proposed_by_backbone_offered": proposed.get("internal.note", 0),
                    "texts": [t[:600] for t in note_texts]}
    out["operator_requests"] = {"executed": len(reqs), "kinds": dict(collections.Counter(
        d["chosen"]["arguments"].get("kind", "") for d in reqs)),
        "distinct_titles": len({d["chosen"]["arguments"].get("title", "") for d in reqs})}
    exp_txt = [d["chosen"].get("expected", "") for d in decs if d["chosen"].get("expected")]
    out["expectations"] = {"chosen_with_expected": len(exp_txt),
                           "with_hypothesis_words": sum(1 for t in exp_txt if hyp(t)), "examples": exp_txt[:8]}
    # coherence: reuse of tokens first observed in an earlier decision's result
    seen_tokens: Dict[str, int] = {}
    follow_ups, repeats, executed_keys = 0, 0, {}
    revisions_after_failure, failed_ops = 0, {}
    for i, d in enumerate(decs):
        ch = d["chosen"]
        key = json.dumps([ch["operation"], ch.get("arguments")], sort_keys=True, default=str)
        strs = [s for s in _strings(ch.get("arguments", {})) if s not in resource_strings]
        toks = {t for s in strs for t in __import__("re").findall(r"[A-Za-z0-9_./:-]{6,}", s)}
        if any(seen_tokens.get(t, i) < i for t in toks):
            follow_ups += 1
        if key in executed_keys:
            repeats += 1
        res = results.get(d["decision_id"]) or {}
        if ch["operation"] in failed_ops and failed_ops[ch["operation"]] != key:
            revisions_after_failure += 1
            failed_ops.pop(ch["operation"], None)
        if res.get("status") in ("denied", "failed", "error", "timeout"):
            failed_ops[ch["operation"]] = key
        executed_keys[key] = i
        for oid in res.get("observation_ids") or []:
            for t in __import__("re").findall(r"[A-Za-z0-9_./:-]{6,}", obs_text.get(oid, "")):
                seen_tokens.setdefault(t, i)
    rev_notes = 0
    for j in range(1, len(note_texts)):
        wj = _words(note_texts[j])
        if any(wj and _words(t) and len(wj & _words(t)) / len(wj | _words(t)) >= 0.3 for t in note_texts[:j]):
            rev_notes += 1
    out["coherence"] = {"follow_ups_using_earlier_results": follow_ups, "identical_repeats": repeats,
                        "max_run_same_operation": max((len(list(g)) for _, g in __import__("itertools").groupby(
                            d["chosen"]["operation"] for d in decs)), default=0)}
    out["revisions"] = {"notes_building_on_earlier_notes": rev_notes,
                        "changed_retry_after_failure": revisions_after_failure}
    out["experiments_registered"] = len({r.get("data", {}).get("experiment_id") for r in
                                         _jsonl(os.path.join(R, "experiments.jsonl"))} - {None})
    out["result_status"] = dict(collections.Counter((results.get(d["decision_id"]) or {}).get("status", "none")
                                                    for d in decs))
    ts = sorted(float(d.get("ts") or 0.0) for d in decs)
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]))
    out["median_seconds_between_decisions"] = gaps[len(gaps) // 2] if gaps else None
    out["seconds_first_to_last_decision"] = (ts[-1] - ts[0]) if len(ts) > 1 else None
    return out


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "analyse-run":                      # python -m cam_live.eval.proposal_probe analyse-run R
        res = analyse_run(argv[1])
        text = json.dumps(res, indent=1, default=str)
        if len(argv) > 2:
            with open(argv[2], "w") as f:
                f.write(text)
        print(text[:6000])
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="", help="FAMILY:PROBLEM[,FAMILY:PROBLEM...] (overrides --family/--problem)")
    ap.add_argument("--family", default="SW")
    ap.add_argument("--problem", default="rle")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--root", default="", help="run root (default <EVAL_HOME>/oe/proposal_probe)")
    ap.add_argument("--allow-unavailable", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="debug model on the CPU (plumbing check only)")
    ap.add_argument("--lock-timeout", type=float, default=1800.0, help="seconds to wait for the shared GPU lock")
    ap.add_argument("--lock-poll", type=float, default=1.0, help="seconds between attempts to take the GPU lock")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import contextlib

    import torch
    from ..nn.gpu import gpu_lock
    from ..validation import OffsetClock
    from .conditions import _CLASSES
    from .experiments import EVAL_HOME
    from .oe_runner import substantive_config
    from .openended import OpenEndedComputer
    plan = [tuple(p.split(":", 1)) for p in a.plan.split(",") if p] if a.plan else [(a.family, a.problem)]
    root = os.path.abspath(a.root or os.path.join(EVAL_HOME, "oe", "proposal_probe"))
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    out: Dict[str, Any] = {"plan": [f"{f}:{p}" for f, p in plan], "rounds_per_objective": a.rounds, "root": root,
                           "real": not a.cpu, "objectives": []}
    lock = contextlib.nullcontext() if a.cpu else gpu_lock(min_free_gb=12.0, timeout_s=a.lock_timeout,
                                                                poll_s=a.lock_poll)
    with lock:
        t_lock = time.time()
        clock = OffsetClock()
        cfg = substantive_config("E", 0, real=not a.cpu)
        if a.tokens:
            cfg.sections["agent"]["proposer_max_new_tokens"] = a.tokens
        agent = _CLASSES["E"](cfg, os.path.join(root, "agent"), clock=clock)
        out["model_build_s"] = time.time() - t_lock
        out["agent_settings"] = {k: getattr(agent.settings, k) for k in
                                 ("proposer_k", "proposer_max_new_tokens", "proposer_temperature",
                                  "proposer_max_obs_chars", "proposer_tools", "proposer_fit_prompt")
                                 if hasattr(agent.settings, k)}
        bb = getattr(agent.model, "backbone", None)
        enc = getattr(bb, "_encode", None)
        out["prompt_window_tokens"] = getattr(getattr(bb, "nn_cfg", None), "max_prompt_tokens", None)
        sent: List[str] = []
        real_generate = agent.model.generate

        def capture(prompts, **kw):
            sent.extend(prompts)
            return real_generate(prompts, **kw)
        agent.model.generate = capture
        orig = agent.proposer.propose
        captured: Dict[str, Any] = {}

        def spy(*args, **kw):
            r = orig(*args, **kw)
            captured["r"] = r
            return r
        agent.proposer.propose = spy
        oc = OpenEndedComputer(os.path.join(root, "computer"), clock)
        from ..harness.sandbox import probe as sandbox_probe
        sp = sandbox_probe()
        for oi, (fam, prob) in enumerate(plan):
            wseed = int(prob) * 100 if fam == "MECH" else 1
            env, policy, verifier, task = oc.open(fam, prob, wseed, f"obj_probe_{oi}", max_decisions=10)
            allowed = set(task.interfaces or [])
            tools = env.discover_tools()
            permitted = [t for t in tools if not allowed or t.modality in allowed or t.modality == "internal"]
            ob: Dict[str, Any] = {
                "family": fam, "problem": prob, "objective": task.objective, "success": task.success,
                "interfaces": task.interfaces, "n_tools": len(tools), "n_permitted": len(permitted),
                "tools": [{"name": t.name, "modality": t.modality, "available": t.available,
                           "reason": t.unavailable_reason} for t in tools],
                "unavailable_permitted": {t.name: t.unavailable_reason for t in permitted if not t.available},
                "sandbox_workspace": oc.workspace,
                "workspace_traversable_by_sandbox_uid": traversable(oc.workspace, sp["uid"], sp["gid"]),
                "rounds": []}
            out["objectives"].append(ob)
            if ob["unavailable_permitted"] and not a.allow_unavailable:
                ob["stopped"] = ("permitted tools unavailable: " + "; ".join(
                    f"{k}: {v}" for k, v in sorted(ob["unavailable_permitted"].items()))[:600])
                print(ob["stopped"], file=sys.stderr)
                break
            agent.attach_verifier(verifier)
            agent.set_objective(task)
            agent.ingest(env.observe())
            target = target_file(fam)
            for i in range(a.rounds):
                sent.clear()
                captured.clear()
                t0 = time.time()
                cands = agent.propose(tools, a.k)
                r = captured.get("r")
                prompt = sent[0] if sent else ""
                bvalid = list(r.actions) if r is not None else []
                ps = (agent.last.get("proposal") or {}).get("stats") or {}
                ob["rounds"].append({
                    "seconds": time.time() - t0, "prompt": prompt, "prompt_chars": len(prompt),
                    "prompt_tokens": len(enc(prompt)) if callable(enc) and prompt else None,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if prompt else None,
                    "prompt_tools": tools_of_prompt(prompt), "agent_recorded": {
                        k: ps.get(k) for k in ("prompt_tools", "prompt_sha256", "prompt_tokens", "prompt_fit",
                                               "tools_permitted", "tools_not_listed", "tools_unavailable")},
                    "stats": r.stats if r else None, "raw": r.raw if r else None,
                    "rejections": [{"reason": x.reason, "detail": x.detail[:200], "text": x.fragment[:300]}
                                   for x in (r.rejections if r else [])][:20],
                    "backbone_valid": [{"op": c.operation, "steps": [s.operation for s in c.steps],
                                        "args": json.dumps(c.arguments, default=str)[:600]} for c in bvalid],
                    "measures": classify(bvalid, target),
                    "candidates": [{"op": c.operation, "prov": c.provenance,
                                    "args": json.dumps(c.arguments, default=str)[:300]} for c in cands]})
                agent.n_decisions += 1
            rs = ob["rounds"]
            n = max(1, len(rs))
            items = sum(int((x["stats"] or {}).get("n_items") or 0) for x in rs)
            valid = sum(int((x["stats"] or {}).get("n_backbone_valid") or 0) for x in rs)
            ob["summary"] = {
                "rounds": len(rs),
                **{f"rounds_{k}": sum(1 for x in rs if x["measures"][k]) / n for k in
                   ("writes_target_file", "target_with_content", "writes_any_file", "executes_program",
                    "requests_operator")},
                "rounds_write_or_execute": sum(1 for x in rs if x["measures"]["writes_target_file"]
                                               or x["measures"]["executes_program"]) / n,
                "items": items, "valid": valid, "validity": (valid / items) if items else None,
                "mean_prompt_chars": sum(x["prompt_chars"] for x in rs) / n,
                "mean_prompt_tokens": (sum(x["prompt_tokens"] or 0 for x in rs) / n) if rs and rs[0]["prompt_tokens"]
                is not None else None,
                "max_prompt_tokens": max((x["prompt_tokens"] or 0 for x in rs), default=0),
                "mean_tools_listed": sum(len(x["prompt_tools"]) for x in rs) / n,
                "operations_written": sorted({o["op"] if not o["steps"] else "program[" + ",".join(o["steps"]) + "]"
                                              for x in rs for o in x["backbone_valid"]}),
                "mean_round_s": sum(x["seconds"] for x in rs) / n}
        oc.close()
        agent.close()
        out["lock_hold_s"] = time.time() - t_lock
        if torch.cuda.is_available():
            out["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({"root": root, "objectives": [{k: o.get(k) for k in ("family", "problem", "n_permitted",
                                                                           "unavailable_permitted", "stopped",
                                                                           "summary")}
                                                    for o in out["objectives"]]}, default=str)[:4000])
    return 0 if all(not o.get("stopped") for o in out["objectives"]) else 2


if __name__ == "__main__":
    sys.exit(main())
