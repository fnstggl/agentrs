"""Automatic failure diagnosis: was a run lost to the interface, to the agent's beliefs, its code, its resources,
its strategy, a competitor - or was the task too hard?

Input: a kernel world (its event log) and an actor; optionally the agent's own transcript metadata (per-turn
prompt and completion tokens, context limit, parse failures - what `agents.llm.LLMPolicy` records), the outcome
of the goal, and the results of reference policies on the same task. Output (`diagnose`, a JSON-able dict):

* ``actions``  - one record per attempt with exactly one class and the evidence (event sequence numbers);
* ``classes`` / ``families`` - counts and rates, so that runs can be compared (`compare`);
* ``loops``, ``context``, ``code``, ``resources``, ``competitor``, ``strategy``, ``difficulty`` - episode-level
  detectors, each with a ``status`` of ``established``, ``absent`` or ``not_determinable`` and its evidence;
* ``primary`` and ``findings`` - the established causes in a fixed, documented order;
* ``not_determinable`` - everything that could not be decided from what was supplied, and why.

Nothing is guessed: a detector that lacks its input says so. Nothing here is reachable from inside a world.

Per-attempt classes (mutually exclusive, in the order they are tested):

  syntax        the reply contained no tool call (transcript ``parse_error``; never reached the world), or the
                action was not ``{tool: str, args: object}``
  wait          an explicit no-op turn (transcript only)
  no_action     a turn without an action whose reason is not recorded (trajectory files: wait or unparseable)
  unknown_tool  no such tool, or ``world.describe`` of a host that does not exist. ``sub`` is ``near_name`` when
                an existing tool was a near miss (`discovery.near_tools`: typo, host alias, same words) - a
                *discovery* failure - otherwise ``unknown_host`` or ``no_near_name`` (an invented tool)
  permission    forbidden / unauthenticated
  schema        the arguments violated the tool's schema (rejected before the handler ran)
  nonexistent   well-formed arguments that refer to something that does not exist (``not_found``)
  conflict      well-formed arguments that collide with existing state (``conflict``: taken, exists, wrong status)
  rule          well-formed arguments refused by a rule of the tool itself (``invalid_action`` raised by the handler)
  resource      ``insufficient_funds`` / ``rate_limited`` / ``capacity_exceeded`` / ``budget_exhausted``
  service       the callee failed: ``service_unavailable`` / ``timeout`` / ``sandbox_error``, or an accepted call
                to somebody else's deployed service that answered with an error
  code          the agent's own code did not work: failing tests, a program that raised, a deployment that
                failed its import probe or was refused for its manifest, a call to its own service that errored
  simulator     an internal error of the simulator (never attributed to the agent)
  ok_effect     accepted and changed the world (emitted at least one event other than bookkeeping)
  ok_read       accepted without effect (a read, or a write that changed nothing)

``schema`` versus ``nonexistent``/``conflict`` separates *incorrect arguments* from *incorrect beliefs about the
world*; ``unknown_tool/near_name`` versus the other two subs separates *discovery* from *invention*.

Loop detector (`find_loops`), precisely. Attempts are taken in order. Two attempts are *near-same* when they call
the same tool with identical arguments, or with the same argument names and exactly one argument differing whose
two values are texts of 20+ characters with word-set Jaccard similarity >= 0.8 (the same call reworded; a
different id, number, name or file content makes a different call; two unparsed replies: identical text). An
attempt's *outcome key* is: its error code, if it was rejected; the kinds of events it emitted, if it was
accepted and changed the world (its result always carries a fresh identifier and is not compared); the digest
of its result, if it was accepted without effect and results were supplied. An attempt is a **repeat without new
information** when at least `LOOP_MIN_PRIOR` (2) of the `LOOP_WINDOW` (8) attempts before it are near-same to it
and have the same outcome key: the third identical outcome of the same call within nine turns. Accepted calls
without effect whose results are unknown are counted separately (``unverified_reads``): the event log does not
record results, so whether such a poll returned something new is not determinable from the log alone (pass the
run's `trajectory` to settle it).

Order of ``primary`` (the first that holds): goal met; context truncated in at least half of the turns; an
unevenly truncated tool list to whose visible part the agent's calls were confined (`tool_list_exposure`); at
least half of the attempts did not succeed (then the largest failing class); loop (at least half of the attempts);
code failure; resource exhaustion; competitor response; then, for valid and varied activity that missed the
goal, task difficulty if a competent reference also failed, else strategic failure; otherwise not determinable.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable, Mapping

from ..core import discovery as D
from ..core.canonical import digest

CLASSES = ("ok_effect", "ok_read", "wait", "no_action", "syntax", "unknown_tool", "permission", "schema",
           "nonexistent", "conflict", "rule", "resource", "service", "code", "simulator")
FAMILIES = {   # coarser grouping for comparison tables
    "valid": ("ok_effect", "ok_read"),
    "idle": ("wait", "no_action"),
    "form": ("syntax", "schema"),                       # the action itself was not well-formed
    "discovery": (),                                    # unknown_tool with a near name (filled by sub)
    "belief": ("nonexistent", "conflict"),              # plus unknown_tool without a near name
    "permission": ("permission",),
    "rule": ("rule",),
    "resource": ("resource",),
    "service": ("service",),
    "code": ("code",),
    "simulator": ("simulator",),
}
BOOKKEEPING = frozenset({"cmd", "cmd.rejected", "ctr", "idem"})          # never part of an outcome
NO_EFFECT = BOOKKEEPING | {"know", "note", "ext"}                        # information or records, not world changes
RESOURCE = {"insufficient_funds": "funds", "rate_limited": "rate_limit", "capacity_exceeded": "capacity",
            "budget_exhausted": "budget"}
SERVICE = ("service_unavailable", "timeout", "sandbox_error")
LOOP_WINDOW, LOOP_MIN_PRIOR = 8, 2
REWORD_MIN_CHARS, REWORD_JACCARD = 20, 0.8    # a reworded text argument still counts as the same call
TRUNCATION_PRIMARY = 0.5       # share of truncated turns from which truncation is the primary cause
INVALID_PRIMARY = 0.5          # share of unaccepted attempts from which invalid activity is the primary cause
LOOP_ESTABLISHED = 0.2         # share of loop attempts from which a loop is an established finding
LOOP_PRIMARY = 0.5
VALID_MIN = 0.7                # strategic failure needs at least this share of accepted attempts ...
VARIED_MIN_TOOLS = 3           # ... at least this many distinct tools among them ...
VARIED_MAX_TOP = 0.5           # ... and no single call making up more than this share
MIN_CUSTOMERS = 5              # a customer base smaller than this cannot be "lost"
REVERSAL = 0.3                 # loss of this share from the peak counts as a reversal
RIVAL_SHARE = 0.5              # share of the loss that rivals who moved must have gained
EXHAUSTED = 0.1                # funds are exhausted when cash is at most this share of the most the actor held
RESOURCE_SHARE = 0.1           # share of attempts refused for one kind of resource from which it is established
EXPOSURE_FULL = 0.9            # a tool list is "uneven" when fewer than this share of tools had a signature ...
EXPOSURE_CALLS = 0.95          # ... and it confounds the run when this share of calls went to those that had one
_MALFORMED = D._MALFORMED
EVIDENCE_MAX = 12


# ====================================================================== attempts
def _canon(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)


def _attempt(i: int, **kw) -> dict:
    a = {"i": i, "seq": None, "t": None, "tool": None, "args": None, "ok": None, "code": None, "message": "",
         "handler_ran": None, "events": [], "event_seqs": [], "result_digest": None, "turn": None,
         "class": "", "sub": "", "flags": [], "near": [], "evidence": []}
    a.update(kw)
    return a


def _events(world, since_seq: int = 0, to_seq: int | None = None) -> list:
    return list(world.events_range(max(int(since_seq), 0), to_seq))


def attempts_from_events(events: list, actor: str) -> list[dict]:
    """Every command `actor` submitted, accepted or rejected, in log order, with the events it emitted."""
    out: list[dict] = []
    k = 0
    while k < len(events):
        e = events[k]
        if e.type == "cmd" and e.actor == actor and e.payload.get("n") is not None:
            n = e.payload["n"]
            j = k + 1
            tx = []
            while j < len(events) and events[j].cmd == n and events[j].type not in ("cmd", "cmd.rejected"):
                tx.append(events[j])
                j += 1
            out.append(_attempt(len(out), seq=e.seq, t=e.t, tool=e.payload["tool"], args=e.payload.get("args") or {},
                                ok=True, handler_ran=True, events=[x.type for x in tx if x.type not in BOOKKEEPING],
                                event_seqs=[e.seq] + [x.seq for x in tx], _tx=tx))
            k = j
            continue
        if e.type == "cmd.rejected" and e.payload.get("actor") == actor:
            err = e.payload.get("error") or {}
            out.append(_attempt(len(out), seq=e.seq, t=e.t, tool=e.payload.get("tool"), args=e.payload.get("args"),
                                ok=False, code=err.get("code"), message=str(err.get("message", "")),
                                handler_ran=e.payload.get("n") is not None, event_seqs=[e.seq],
                                _details=err.get("details") or {}, _tx=[]))
        k += 1
    return out


def attempts_from_trajectory(records: Iterable[Mapping], actor: str | None = None) -> list[dict]:
    """Attempts from ``crucible-traj/1`` step records (`training.trajectories`). Such a file carries the action,
    the result (with the error of a rejection) and the *numbers* of the events of each step, but not the events:
    whether an accepted action changed the world is inferred from the number of events it emitted (more than
    the command record itself), and a step without an action cannot be told apart as wait or unparseable."""
    out: list[dict] = []
    for r in records:
        if actor is not None and r.get("actor") != actor:
            continue
        act, res = r.get("action"), r.get("result")
        if not isinstance(act, Mapping):
            out.append(_attempt(len(out), t=r.get("t"), turn=r.get("step"), _tx=[], _noaction=True))
            continue
        res = res if isinstance(res, Mapping) else {}
        err = res.get("error") or {}
        ok = bool(res.get("ok"))
        seqs = [int(s) for s in (r.get("events") or [])]
        msg = str(err.get("message", ""))
        details = err.get("details") or {}
        pre = (not ok) and (err.get("code") in ("unknown_tool", "unauthenticated", "rate_limited") and not res.get("cost_s")
                            or (err.get("code") == "invalid_action" and (msg == _MALFORMED or (
                                "tool" in details and msg.startswith("args")))))
        a = _attempt(len(out), seq=seqs[0] if seqs else None, t=r.get("t"), tool=act.get("tool"),
                     args=act.get("args") if isinstance(act.get("args"), Mapping) else None, ok=ok,
                     code=None if ok else err.get("code"), message=msg, handler_ran=not pre,
                     events=["?"] * max(0, len(seqs) - 1) if ok else [], event_seqs=seqs, turn=r.get("step"),
                     result_digest=digest(res.get("result")) if ok else None, _details=details, _tx=[],
                     _result=res.get("result") if ok else None)
        out.append(a)
    return out


def merge_transcript(attempts: list[dict], transcript: Iterable[Mapping] | None) -> tuple[list[dict], list[str]]:
    """Interleave turns that never reached the world (unparseable replies, waits) with the commands in the log,
    using the agent's transcript (records with ``parse_error`` / ``wait`` / ``action``, in turn order). Returns
    the merged attempts and notes about anything that could not be aligned."""
    if not transcript:
        return attempts, []
    notes: list[str] = []
    out: list[dict] = []
    k = 0
    for turn, rec in enumerate(transcript):
        tn = rec.get("turn", turn)
        if rec.get("parse_error") is not None:
            out.append(_attempt(0, t=rec.get("t"), turn=tn, ok=False, code="parse_error",
                                message=str((rec.get("parse_error") or {}).get("message", "")),
                                _text=str(rec.get("text", "")), _tx=[]))
        elif rec.get("wait"):
            out.append(_attempt(0, t=rec.get("t"), turn=tn, _wait=True, _tx=[]))
        elif isinstance(rec.get("action"), Mapping):
            tool = rec["action"].get("tool")
            if k < len(attempts) and attempts[k]["tool"] == tool:
                attempts[k]["turn"] = tn
                out.append(attempts[k])
                k += 1
            else:
                notes.append(f"transcript turn {tn} ({tool}) has no matching command in the log")
    if k < len(attempts):
        notes.append(f"{len(attempts) - k} commands in the log have no transcript turn (appended in log order)")
        out.extend(attempts[k:])
    for i, a in enumerate(out):
        a["i"] = i
    return out, notes


# ====================================================================== per-attempt classification
def message_stem(msg: str, words: int = 4) -> str:
    """The message with quoted strings, identifiers and numbers removed, cut to its first words: groups the
    refusals of a handler by what they say (``unknown need 'review'`` -> ``unknown need``) without interpreting."""
    m = re.sub(r"'[^']*'|\"[^\"]*\"|`[^`]*`", " ", str(msg))
    m = re.sub(r"\$?[\d,.]+|\b\w+_\d+\b|\b[a-z]+:[\w-]+\b", " ", m)
    toks = re.findall(r"[A-Za-z]+", m)
    return " ".join(toks[:words]).lower()


_SCHEMA_KINDS = (("required", "required"), ("unexpected property", "unexpected"), (": expected ", "type"),
                 ("does not match", "pattern"), ("not one of", "enum"), ("below minimum", "range"),
                 ("above maximum", "range"), ("longer than", "length"), ("shorter than", "length"),
                 (" items", "items"), ("NaN", "nan"))


def classify_attempt(a: dict, names: Iterable[str] | None, aliases: Mapping[str, str] | None, own_hosts: set,
                     builtin_hosts: set | None) -> None:
    """Set ``class``, ``sub``, ``flags``, ``near`` and ``evidence`` of one attempt (see the module docstring)."""
    a["evidence"] = list(a["event_seqs"][:EVIDENCE_MAX])
    if a.get("_wait"):
        a["class"] = "wait"
        return
    if a.get("_noaction"):
        a["class"] = "no_action"
        return
    code, msg, tool = a["code"], a["message"], a["tool"]
    if code == "parse_error":
        a["class"], a["sub"] = "syntax", "unparseable"
        return
    if a["ok"]:
        host = str(tool).partition(".")[0]
        tx = a.get("_tx") or []
        ext = [o for x in tx if x.type == "ext" for o in x.payload.get("outcomes", []) if o]
        failed_ext = [o for o in ext if o.get("kind") in ("code.test", "code.run", "cloud.probe", "code.batch")
                      and (not (o.get("out") or {}).get("ok")
                           or (o.get("kind") == "code.test" and ((o.get("out") or {}).get("result") or {}).get("failed")))]
        dep_failed = [x for x in tx if x.type == "cloud.dep_status" and x.payload.get("status") == "failed"]
        res = a.get("_result")
        if res is not None and tool in ("code.test", "code.run") and isinstance(res, Mapping):   # trajectory source
            if res.get("ok") is False or (isinstance(res.get("result"), Mapping) and res["result"].get("failed")):
                failed_ext = failed_ext or [{"kind": tool}]
        if res is not None and tool == "cloud.deploy" and isinstance(res, Mapping) and res.get("status") == "failed":
            dep_failed = dep_failed or [None]
        if failed_ext or dep_failed:
            a["class"] = "code"
            a["sub"] = "deploy_failed" if dep_failed else {"code.test": "tests_failed", "cloud.probe": "deploy_failed"}.get(
                failed_ext[0].get("kind"), "program_error")
            return
        usage_bad = [x for x in tx if x.type == "cloud.usage" and x.payload.get("outcome") not in ("ok", None)]
        status = res.get("status") if isinstance(res, Mapping) else None
        deployed = builtin_hosts is not None and host not in builtin_hosts
        if (usage_bad and deployed) or (deployed and isinstance(status, int) and status >= 400):
            mine = host in own_hosts
            a["class"] = "code" if mine else "service"
            a["sub"] = "own_service_error" if mine else "callee_error"
            a["evidence"] = ([a["seq"]] if a["seq"] is not None else []) + [x.seq for x in usage_bad][:EVIDENCE_MAX]
            return
        a["class"] = "ok_effect" if any(t not in NO_EFFECT for t in a["events"]) else "ok_read"
        return
    if code == "invalid_action" and msg == _MALFORMED:
        a["class"], a["sub"] = "syntax", "malformed_action"
        return
    if code == "unknown_tool":
        target = tool
        if tool == "world.describe" and isinstance(a["args"], Mapping) and a["handler_ran"]:
            h = str(a["args"].get("host", ""))
            a["flags"].append("describe_host")
            if names is None:
                a["class"], a["sub"] = "unknown_tool", "undetermined"
                return
            near = D.near_host_names(h, sorted({n.partition(".")[0] for n in names}), aliases)
            a["class"], a["sub"], a["near"] = "unknown_tool", ("near_name" if near else "unknown_host"), near
            return
        a["class"] = "unknown_tool"
        if names is None:
            a["sub"] = "undetermined"
            return
        near = D.near_tools(str(target), names, aliases)
        hosts = {n.partition(".")[0] for n in names}
        a["near"] = near
        a["sub"] = "near_name" if near else ("no_near_name" if str(target).partition(".")[0] in hosts else "unknown_host")
        return
    if code in ("forbidden", "unauthenticated"):
        a["class"], a["sub"] = "permission", code
        return
    if code == "invalid_action":
        if a["handler_ran"] is False:
            a["class"] = "schema"
            a["sub"] = "+".join(sorted({k for part in msg.split("; ") for frag, k in _SCHEMA_KINDS if frag in part}))
            return
        if tool == "cloud.deploy":
            a["class"], a["sub"] = "code", "manifest_rejected"
            return
        a["class"], a["sub"] = "rule", message_stem(msg)
        return
    if code == "not_found":
        a["class"], a["sub"] = "nonexistent", message_stem(msg)
        return
    if code == "conflict":
        a["class"], a["sub"] = "conflict", message_stem(msg)
        return
    if code in RESOURCE:
        a["class"], a["sub"] = "resource", RESOURCE[code]
        return
    if code in SERVICE:
        a["class"], a["sub"] = "service", code
        return
    a["class"], a["sub"] = "simulator", str(code)


# ====================================================================== loops
def _words(s: str) -> set:
    return set(D._TOKEN.findall(s.lower()))


def near_same(a: dict, b: dict) -> bool:
    """The same call, or the same call reworded: same tool and identical arguments; or the same argument names
    with exactly one argument differing, both of its values being texts of at least `REWORD_MIN_CHARS` characters
    whose word sets have Jaccard similarity of at least `REWORD_JACCARD`. A different id, number, name or file
    content is a different call. Two unparsed replies are near-same when their text is identical."""
    if a.get("_text") is not None or b.get("_text") is not None:
        return a.get("_text") is not None and a.get("_text") == b.get("_text")
    if a["tool"] is None or a["tool"] != b["tool"]:
        return False
    x, y = a["args"] or {}, b["args"] or {}
    if not isinstance(x, Mapping) or not isinstance(y, Mapping):
        return _canon(x) == _canon(y)
    if set(x) != set(y):
        return False
    diff = [k for k in x if _canon(x[k]) != _canon(y[k])]
    if not diff:
        return True
    if len(diff) > 1:
        return False
    u, v = x[diff[0]], y[diff[0]]
    if not (isinstance(u, str) and isinstance(v, str) and min(len(u), len(v)) >= REWORD_MIN_CHARS):
        return False
    wu, wv = _words(u), _words(v)
    return bool(wu | wv) and len(wu & wv) / len(wu | wv) >= REWORD_JACCARD


def outcome_key(a: dict) -> tuple:
    """What the attempt told the agent: the error code of a rejection; for an accepted call that changed the
    world, the kinds of events it emitted (its result always differs - a new id - and is not compared); for an
    accepted call without effect, the digest of its result when results are known."""
    if a["class"] in ("wait", "no_action"):
        return ("idle",)
    if not a["ok"]:
        return ("rejected", a["code"])
    if a["class"] == "ok_read":
        return ("read", a["result_digest"])
    return ("effect", a["class"], a["sub"], tuple(sorted(set(a["events"]))), len(a["events"]) if "?" in a["events"] else 0)


def find_loops(attempts: list[dict]) -> dict:
    """Flag repeats without new information (module docstring) and summarise them."""
    flagged: list[int] = []
    unverified: list[int] = []
    for j, a in enumerate(attempts):
        if a["class"] in ("wait", "no_action"):
            continue
        key = outcome_key(a)
        prior = [b for b in attempts[max(0, j - LOOP_WINDOW): j]
                 if b["class"] not in ("wait", "no_action") and near_same(a, b) and outcome_key(b) == key]
        if len(prior) < LOOP_MIN_PRIOR:
            continue
        if a["ok"] and a["class"] == "ok_read" and a["result_digest"] is None:
            a["flags"].append("repeat_read_unverified")
            unverified.append(j)
            continue
        a["flags"].append("loop")
        a["loop_of"] = [b["seq"] for b in prior[-LOOP_MIN_PRIOR:]]
        flagged.append(j)
    runs: list[dict] = []
    for j in flagged:
        a = attempts[j]
        if runs and runs[-1]["_last"] == j - 1:
            r = runs[-1]
            r["length"] += 1
            r["_last"] = j
            r["last_seq"] = a["seq"]
            r["tools"][a["tool"] or "<unparsed>"] = r["tools"].get(a["tool"] or "<unparsed>", 0) + 1
        else:
            runs.append({"first_i": j, "first_seq": a["seq"], "last_seq": a["seq"], "length": 1, "_last": j,
                         "kind": "repeated_rejection" if not a["ok"] else
                         ("repeated_effect" if a["class"] == "ok_effect" else "repeated_read"),
                         "outcome": a["code"] or "ok", "tools": {a["tool"] or "<unparsed>": 1}})
    for r in runs:
        r.pop("_last")
        r["tool"] = max(sorted(r["tools"]), key=lambda t: r["tools"][t])
    n = sum(1 for a in attempts if a["class"] not in ("wait", "no_action"))
    by_tool: dict[str, int] = {}
    for j in flagged:
        t = attempts[j]["tool"] or "<unparsed>"
        by_tool[t] = by_tool.get(t, 0) + 1
    rate = len(flagged) / n if n else 0.0
    return {"status": "established" if rate >= LOOP_ESTABLISHED else "absent",
            "actions": len(flagged), "rate": round(rate, 4), "runs": len(runs),
            "longest_run": max([r["length"] for r in runs] + [0]),
            "by_tool": dict(sorted(by_tool.items(), key=lambda kv: (-kv[1], kv[0]))),
            "run_list": sorted(runs, key=lambda r: -r["length"])[:10],
            "unverified_reads": len(unverified),
            "evidence": [attempts[j]["seq"] for j in flagged if attempts[j]["seq"] is not None][:EVIDENCE_MAX],
            "definition": f"same tool, identical arguments (or one text argument reworded, Jaccard >= "
                          f"{REWORD_JACCARD}), same outcome, at least {LOOP_MIN_PRIOR} times in the preceding "
                          f"{LOOP_WINDOW} attempts"}


# ====================================================================== context truncation
def context_report(attempts: list[dict], transcript, usage: Mapping | None, context_limit: int | None,
                   max_new_tokens: int | None) -> dict:
    """Truncated prompts. Per turn it needs the transcript (``truncated`` flag, or ``prompt_tokens`` with the
    context limit: the backend truncates to ``context_limit - max_new_tokens`` tokens); for the episode the
    policy's usage counters (``truncated_prompts`` of ``calls``) suffice."""
    turns = list(transcript or [])
    room = None if context_limit is None else int(context_limit) - int(max_new_tokens or 0)
    out: dict[str, Any] = {"context_limit": context_limit, "max_new_tokens": max_new_tokens}
    if turns and (any("truncated" in r for r in turns) or (room is not None and any("prompt_tokens" in r for r in turns))):
        trunc = []
        for i, r in enumerate(turns):
            if r.get("truncated") is True or (r.get("truncated") is None and room is not None
                                              and r.get("prompt_tokens") is not None and r["prompt_tokens"] >= room):
                trunc.append(r.get("turn", i))
        toks = [r["prompt_tokens"] for r in turns if r.get("prompt_tokens") is not None]
        rate = len(trunc) / len(turns)
        out.update(status="established" if trunc else "absent", per_turn=True, turns=len(turns),
                   truncated_turns=len(trunc), rate=round(rate, 4), evidence_turns=trunc[:EVIDENCE_MAX],
                   max_prompt_tokens=max(toks) if toks else None,
                   mean_prompt_tokens=round(sum(toks) / len(toks), 1) if toks else None)
        tset = set(trunc)
        for a in attempts:
            if a["turn"] in tset:
                a["flags"].append("context_truncated")
        return out
    if usage and usage.get("calls") and usage.get("truncated_prompts") is not None:
        n, k = int(usage["calls"]), int(usage["truncated_prompts"])
        out.update(status="established" if k else "absent", per_turn=False, turns=n, truncated_turns=k,
                   rate=round(k / n, 4), mean_prompt_tokens=round(usage["prompt_tokens"] / n, 1)
                   if usage.get("prompt_tokens") else None,
                   note="episode total only: which turns were truncated is not determinable without the transcript")
        return out
    out.update(status="not_determinable", reason="no transcript metadata and no usage counters were supplied")
    return out


def tool_list_exposure(tools: list[Mapping], interface: Mapping, attempts: list[dict]) -> dict:
    """What the prompt showed of the catalogue under its character budget, and how the agent's calls divide
    between tools shown with their signature and tools shown by name only (or not at all).

    ``interface["renderer"]``: ``"tool_list"`` (default) is the 3.0 `agents.llm.render_tool_list` with
    ``tool_chars`` / ``tool_order`` - full lines for as many tools as fit, the rest by name; ``"catalogue"`` is
    `discovery.render_catalogue` with ``budget_chars`` / ``mode`` - the same detail for every tool. Only the
    static list is reconstructed (the per-turn priority of new and recently used tools is not).

    ``status`` is ``established`` when the exposure was *uneven* - fewer than `EXPOSURE_FULL` of the tools had a
    signature while others had one - and at least `EXPOSURE_CALLS` of the agent's calls to existing tools went
    to the tools that had one: the run then cannot separate what the agent chose from what it was shown."""
    tools = [t for t in tools if not str(t.get("name", "")).startswith("admin.")]
    names = {t["name"] for t in tools}
    hosts_all = sorted({n.partition(".")[0] for n in names})
    renderer = interface.get("renderer", "tool_list")
    if renderer == "catalogue":
        budget = int(interface.get("budget_chars", interface.get("tool_chars", 9000)))
        text = D.render_catalogue(tools, budget, interface.get("mode", "auto"))
        last = text.splitlines()[-1] if text else ""
        level = next((lv for lv, mark in (("full", "complete:"), ("descriptions", "descriptions longer than"),
                                          ("signatures", "Omitted: all descriptions"), ("arguments", "Omitted: argument types"),
                                          ("names", "every tool is named"), ("partial", "(+k more per host)"),
                                          ("hosts", "the names of all")) if mark in last), "hosts")
        shown = set(names) if level in ("full", "descriptions", "signatures", "arguments") else set()
        out = {"renderer": "catalogue", "budget_chars": budget, "level": level,
               "first_line_only": False, "hosts_listed": len(hosts_all)}
    else:
        from ..agents.llm import render_tool_list
        budget = int(interface.get("tool_chars", 9000))
        text = render_tool_list(tools, budget, (), int(interface.get("description_chars", 70)),
                                interface.get("tool_order", "alpha"))
        shown = {ln.split("(", 1)[0] for ln in text.splitlines() if "(" in ln and not ln.startswith("(+")} & names
        out = {"renderer": "tool_list", "tool_chars": budget, "tool_order": interface.get("tool_order", "alpha"),
               "first_line_only": True,
               "note": "descriptions were cut to their first line and to description_chars characters"}
    hosts_full = {n.partition(".")[0] for n in shown}
    called = [a["tool"] for a in attempts if isinstance(a["tool"], str)]
    known = [t for t in called if t in names]
    in_shown = sum(1 for t in known if t in shown)
    share = len(shown) / max(1, len(names))
    call_share = in_shown / len(known) if known else None
    uneven = 0 < len(shown) and share < EXPOSURE_FULL
    out.update(tools=len(names), with_signature=len(shown), share_with_signature=round(share, 4), hosts=len(hosts_all),
               hosts_with_no_signature=[h for h in hosts_all if h not in hosts_full],
               calls_to_existing_tools=len(known), calls_to_tools_with_signature=in_shown,
               share_of_calls_to_tools_with_signature=None if call_share is None else round(call_share, 4),
               discovery_calls=sum(1 for t in called if t in ("world.describe", "world.find_tools", "world.hosts")),
               status="established" if uneven and call_share is not None and call_share >= EXPOSURE_CALLS else "absent")
    return out


# ====================================================================== the actor's principals and assets
def _mine(world, actor: str, events: list) -> set[str]:
    mine = {actor}
    st = world.state
    if "members" in st.tables:
        mine |= {m["org"] for m in st.find("members", "person", actor)}
    for e in events:
        if e.type == "id.org" and e.payload.get("founder") == actor:
            mine.add(e.payload["id"])
        elif e.type == "id.member" and e.payload.get("person") == actor:
            mine.add(e.payload["org"])
    return mine


def _own_deployments(world, mine: set, events: list) -> dict[str, dict]:
    deps: dict[str, dict] = {}
    st = world.state
    if "deployments" in st.tables:
        for did, d in st.scan("deployments"):
            if d["owner"] in mine:
                deps[did] = {"host": d["host"], "status": d["status"]}
    for e in events:
        if e.type == "cloud.deployment" and e.payload.get("owner") in mine:
            deps.setdefault(e.payload["id"], {"host": e.payload["host"], "status": "?"})["seq"] = e.seq
    return deps


# ====================================================================== code
def code_report(world, actor: str, mine: set, events: list, attempts: list[dict]) -> dict:
    """Did the agent produce working code? Tests run and failed, programs that raised, deployments that failed or
    were refused, requests to its own services that errored, and the verified quality of what they returned."""
    deps = _own_deployments(world, mine, events)
    dep_ids = set(deps)
    out: dict[str, Any] = {"tests": 0, "tests_failed": 0, "runs": 0, "runs_failed": 0, "deploys": 0,
                           "deploys_failed": 0, "manifest_rejections": 0, "commits": 0,
                           "requests": {"ok": 0, "error": 0, "overload": 0, "unavailable": 0},
                           "quality": {"scored_hours": 0, "mean_q": None}, "evidence": []}
    ev: list[int] = []
    last_test: dict[str, bool] = {}
    for a in attempts:
        if a["tool"] == "code.commit" and a["ok"]:
            out["commits"] += 1
        if a["tool"] == "code.test" and a["ok"]:
            out["tests"] += 1
            failed = a["class"] == "code"
            out["tests_failed"] += failed
            last_test[str((a["args"] or {}).get("repo"))] = not failed
            if failed:
                ev.append(a["seq"])
        if a["tool"] == "code.run" and a["ok"]:
            out["runs"] += 1
            if a["class"] == "code":
                out["runs_failed"] += 1
                ev.append(a["seq"])
        if a["tool"] == "cloud.deploy":
            if a["ok"]:
                out["deploys"] += 1
                if a["class"] == "code":
                    out["deploys_failed"] += 1
                    ev.append(a["seq"])
            elif a["class"] == "code":
                out["manifest_rejections"] += 1
                ev.append(a["seq"])
    qs: list[tuple[int, float, int]] = []
    my_slots = _my_slots(world, mine)
    by_dep: dict[str, dict[str, int]] = {}
    for e in events:
        if e.type == "cloud.usage" and e.payload.get("dep") in dep_ids:       # one event per request
            oc = e.payload.get("outcome")
            oc = oc if oc in out["requests"] else "error"
            out["requests"][oc] += 1
            r = by_dep.setdefault(e.payload["dep"], {"ok": 0, "error": 0})
            r[oc] = r.get(oc, 0) + 1
            if oc == "error" and len(ev) < 4 * EVIDENCE_MAX:
                ev.append(e.seq)
        elif e.type == "cloud.usage_bulk" and e.payload.get("dep") in dep_ids:
            p = e.payload
            out["requests"]["ok"] += int(p.get("ok", 0))
            out["requests"]["error"] += int(p.get("errors", 0))
            out["requests"]["overload"] += int(p.get("overload", 0))
            out["requests"]["unavailable"] += int(p.get("unavailable", 0))
            r = by_dep.setdefault(p["dep"], {"ok": 0, "error": 0})
            r["ok"] += int(p.get("ok", 0))
            r["error"] += int(p.get("errors", 0))
        elif e.type == "consumers.hour" and my_slots:
            for k, sm in e.payload.get("slots", {}).items():
                if int(k) in my_slots and sm.get("ok", 0) > 0:
                    qs.append((e.seq, float(sm.get("q", 0.0)), int(sm["ok"])))
    # what is judged is the code that is deployed now: the live deployments (else the last one), and the
    # verified quality since the last deployment
    live_ids = sorted(d for d, v in deps.items() if v.get("status") == "live")
    current = live_ids or sorted(deps)[-1:]
    last_dep_seq = max([deps[d].get("seq") or 0 for d in current] + [0])
    recent = [(q, n) for seq, q, n in qs if seq > last_dep_seq]
    if recent:
        tot = sum(n for _, n in recent)
        out["quality"] = {"scored_hours": len(recent), "mean_q": round(sum(q * n for q, n in recent) / tot, 4),
                          "since_seq": last_dep_seq}
    cur = {"ok": sum(by_dep.get(d, {}).get("ok", 0) for d in current),
           "error": sum(by_dep.get(d, {}).get("error", 0) for d in current)}
    out["current_deployments"] = {"ids": current, "requests": cur}
    executed = cur["ok"] + cur["error"]
    reasons = []
    live = [d for d in deps.values() if d.get("status") == "live"]
    if (out["deploys_failed"] or out["manifest_rejections"]) and not live:
        reasons.append("deployment failed or was refused and no deployment is live")
    if executed >= 4 and cur["error"] / executed >= 0.5:
        reasons.append(f"its deployed code raised on {cur['error']} of {executed} executed requests")
    if out["quality"]["mean_q"] is not None and out["quality"]["scored_hours"] >= 2 and out["quality"]["mean_q"] < 0.1:
        reasons.append(f"its services' verified output quality is {out['quality']['mean_q']} (near zero)")
    if last_test and not any(last_test.values()):
        reasons.append("the last test run of every tested repository failed")
    if out["runs"] and out["runs_failed"] == out["runs"]:
        reasons.append("every program it ran raised")
    attempted = bool(out["commits"] or out["tests"] or out["runs"] or out["deploys"] or out["manifest_rejections"] or deps)
    out.update(status="established" if reasons else "absent", attempted=attempted, reasons=reasons,
               live_deployments=len(live), deployments=len(deps),
               evidence=[s for s in ev if s is not None][:EVIDENCE_MAX])
    if not attempted:
        out["note"] = "the actor wrote, ran and deployed no code in this window"
    return out


# ====================================================================== resources
def resource_report(world, actor: str, mine: set, events: list, attempts: list[dict]) -> dict:
    """Funds, rate limits, compute. Cash is reconstructed from the ledger events of the window."""
    st = world.state
    n = sum(1 for a in attempts if a["class"] not in ("wait", "no_action")) or 1
    by: dict[str, list[int]] = {v: [] for v in RESOURCE.values()}
    for a in attempts:
        if a["class"] == "resource":
            by[a["sub"]].append(a["seq"])
    out: dict[str, Any] = {k: {"n": len(v), "rate": round(len(v) / n, 4),
                               "evidence": [s for s in v if s is not None][:EVIDENCE_MAX]}
                           for k, v in by.items()}
    cash = None
    if "accounts" in st.tables:
        accts = {aid for p in sorted(mine) for aid in st.find_ids("accounts", "owner", p)}
        now = sum(st.tables["accounts"][a]["balance"] for a in accts)
        deltas = []
        for e in events:
            if e.type == "bank.txn":
                d = sum(x for acct, x in e.payload.get("postings", []) if acct in accts)
                if d:
                    deltas.append((e.seq, d, e.payload.get("kind")))
        start = now - sum(d for _, d, _ in deltas)
        level, lo, hi, lo_seq = start, start, start, None
        spent: dict[str, int] = {}
        for seq, d, kind in deltas:
            level += d
            if level < lo:
                lo, lo_seq = level, seq
            hi = max(hi, level)
            if d < 0:
                spent[str(kind)] = spent.get(str(kind), 0) - d
        cash = {"start": start, "end": now, "min": lo, "max": hi, "min_seq": lo_seq,
                "spent_by_kind": dict(sorted(spent.items(), key=lambda kv: -kv[1]))}
    out["cash"] = cash
    suspended = []
    inst_owner = {iid: i["owner"] for iid, i in st.scan("instances")} if "instances" in st.tables else {}
    for e in events:
        if e.type == "cloud.instance":
            inst_owner[e.payload["id"]] = e.payload["owner"]
        elif e.type == "cloud.instance_status" and inst_owner.get(e.payload.get("id")) in mine \
                and e.payload.get("status") == "suspended" and str(e.payload.get("reason", "")).startswith("payment failed"):
            suspended.append(e.seq)
    out["instances_suspended_unpaid"] = {"n": len(suspended), "evidence": suspended[:EVIDENCE_MAX]}
    reasons = []
    if cash is not None and (by["funds"] or suspended) and cash["end"] <= EXHAUSTED * max(cash["max"], 1):
        reasons.append(f"funds exhausted: {cash['end']} cents left of at most {cash['max']} held, with "
                       f"{len(by['funds'])} payments refused and {len(suspended)} unpaid suspensions")
    for kind, label in (("funds", "funds"), ("rate_limit", "rate limit"), ("capacity", "capacity"), ("budget", "budget")):
        if len(by[kind]) / n >= RESOURCE_SHARE:
            reasons.append(f"{label}: {len(by[kind])} of {n} attempts refused")
    total = sum(len(v) for v in by.values())
    out.update(status="established" if reasons else "absent", reasons=reasons, n=total, rate=round(total / n, 4),
               evidence=sorted(s for v in by.values() for s in v if s is not None)[:EVIDENCE_MAX] + suspended[:4])
    if by["funds"] and not reasons:
        out["note"] = ("payments were refused for lack of funds in the account used, but the actor's funds as a "
                       "whole were not exhausted")
    return out


# ====================================================================== competitor response
def _my_slots(world, mine: set) -> dict[int, dict]:
    st = world.state
    if "cons_offers" not in st.tables or "listings" not in st.tables:
        return {}
    out = {}
    for lid, o in st.scan("cons_offers"):
        l = st.get("listings", lid)
        if l is not None and l["seller"] in mine:
            out[o["slot"]] = {"listing": lid, "need": o["need"], "t": o["t"]}
    return out


def _population_channel(world, mine: set, events: list) -> dict:
    """Consumer population (aggregate): the hourly ``consumers.hour`` events record, per offering, the number of
    customers, the public price and the outcomes of executed requests. Individual consumers are not identified
    in the log, so attribution is aggregate: the part of the actor's loss matched by gains of rivals that moved."""
    st = world.state
    if "cons_offers" not in st.tables or "listings" not in st.tables:
        return {"status": "not_applicable", "reason": "no consumer population in this world"}
    info: dict[int, dict] = {}
    for lid, o in st.scan("cons_offers"):
        l = st.get("listings", lid)
        info[o["slot"]] = {"listing": lid, "need": o["need"], "seller": l["seller"] if l else None, "t": o["t"]}
    mine_slots = sorted(s for s, v in info.items() if v["seller"] in mine)
    hours = [e for e in events if e.type == "consumers.hour"]
    if not mine_slots or not hours:
        return {"status": "absent", "reason": "the actor has no offering served to the consumer population" if not mine_slots
                else "no demand was recorded in this window"}
    launches = {e.payload["slot"]: e.seq for e in events if e.type == "consumers.offer"}
    updates: dict[str, list[tuple[int, int, int]]] = {}     # listing -> [(seq, t, new price)] of public price changes
    for e in events:
        if e.type == "market.listing_update" and "price" in (e.payload.get("changes") or {}):
            updates.setdefault(e.payload["id"], []).append((e.seq, e.t, e.payload["changes"]["price"]))

    def series(slot: int) -> list[dict]:
        rows = []
        for e in hours:
            sm = e.payload.get("slots", {}).get(str(slot))
            pub = e.payload.get("public", {}).get(str(slot))
            rows.append({"seq": e.seq, "t": e.payload.get("t", e.t), "c": int(sm["customers"]) if sm else 0,
                         "price": pub["price"] if pub else None, "listed": pub is not None,
                         "ok": int(sm["ok"]) if sm else 0,
                         "bad": int(sm["errors"] + sm["unavailable"] + sm["overload"]) if sm else 0,
                         "q": float(sm["q"]) if sm and sm.get("ok") else None})
        return rows
    findings = []
    for s in mine_slots:
        mine_rows = series(s)
        cs = [r["c"] for r in mine_rows]
        peak = max(cs)
        ip = len(cs) - 1 - cs[::-1].index(peak)
        tail = cs[-min(6, len(cs)):]
        final = sum(tail) / len(tail)
        f = {"slot": s, "listing": info[s]["listing"], "need": info[s]["need"], "peak_customers": peak,
             "final_customers": round(final, 2), "peak_seq": mine_rows[ip]["seq"], "end_seq": mine_rows[-1]["seq"]}
        if peak < MIN_CUSTOMERS or ip >= len(cs) - 1 or peak - final < REVERSAL * peak:
            f["status"] = "absent"
            f["reason"] = "never had a customer base" if peak < MIN_CUSTOMERS else \
                f"no reversal: customers at the end are within {REVERSAL:.0%} of the peak"
            findings.append(f)
            continue
        loss = peak - final
        entry_t = min(info[s]["t"], next((r["t"] for r in mine_rows if r["c"] > 0), info[s]["t"]))
        rivals = []
        for r_slot, v in sorted(info.items()):
            if v["seller"] in mine or v["need"] != info[s]["need"]:
                continue
            rr = series(r_slot)
            tail_r = [x["c"] for x in rr[-min(6, len(rr)):]]
            gain = sum(tail_r) / len(tail_r) - rr[ip]["c"]
            moves = []
            if v["t"] > entry_t:
                moves.append({"move": "launched", "t": v["t"], "seq": launches.get(r_slot)})
            # price cuts: decreases of the public price between consecutive hours after the actor's entry; the
            # recorded listing update that set the new price (if it is in the window) is the evidence
            listed = [x for x in rr if x["price"] is not None]
            for prev, cur in zip(listed, listed[1:]):
                if cur["t"] > entry_t and cur["price"] < prev["price"]:
                    seq = next((sq for sq, t, new in reversed(updates.get(v["listing"], []))
                                if new == cur["price"] and t <= cur["t"]), None)
                    moves.append({"move": "price_cut", "t": cur["t"], "seq": seq, "from": prev["price"],
                                  "price": cur["price"]})
            rivals.append({"slot": r_slot, "seller": v["seller"], "listing": v["listing"], "gain": round(gain, 2),
                           "moves": moves})
        gain_moved = sum(max(0.0, r["gain"]) for r in rivals if r["moves"])
        after, before = mine_rows[ip + 1:], mine_rows[: ip + 1]

        def bad_share(rows):
            tot = sum(r["ok"] + r["bad"] for r in rows)
            return sum(r["bad"] for r in rows) / tot if tot else 0.0

        def mean_q(rows):
            v = [r["q"] for r in rows if r["q"] is not None]
            return sum(v) / len(v) if v else None
        own = []
        p0 = next((r["price"] for r in reversed(before) if r["price"] is not None), None)
        p1 = next((r["price"] for r in reversed(after) if r["price"] is not None), None)
        if p0 is not None and p1 is not None and p1 > p0:
            own.append(f"the actor raised its price from {p0} to {p1}")
        if bad_share(after) >= 0.2 and bad_share(after) > bad_share(before) + 0.1:
            own.append(f"its failed requests rose from {bad_share(before):.0%} to {bad_share(after):.0%}")
        qb, qa = mean_q(before), mean_q(after)
        if qb is not None and qa is not None and qa < 0.8 * qb:
            own.append(f"its verified quality fell from {qb:.2f} to {qa:.2f}")
        if any(not r["listed"] for r in after):
            own.append("its offering was not listed for part of the period")
        f.update(loss=round(loss, 2), rivals=[r for r in rivals if r["moves"] or r["gain"] > 0],
                 gained_by_rivals_that_moved=round(gain_moved, 2),
                 share_of_loss_matched=round(min(loss, gain_moved) / loss, 4), own_causes=own)
        if gain_moved >= RIVAL_SHARE * loss and not own:
            f["status"] = "established"
        elif gain_moved >= RIVAL_SHARE * loss:
            f["status"] = "not_determinable"
            f["reason"] = "rivals moved and gained, but the actor's own service also degraded: " + "; ".join(own)
        else:
            f["status"] = "absent"
            f["reason"] = ("customers were lost without a matching gain by a rival that launched or cut its price "
                           "after the actor's entry" + (": " + "; ".join(own) if own else ""))
        f["evidence"] = [x for x in [f["peak_seq"], f["end_seq"]] + [m["seq"] for r in rivals for m in r["moves"]]
                         if x is not None][:EVIDENCE_MAX]
        findings.append(f)
    status = "established" if any(f["status"] == "established" for f in findings) else \
        ("not_determinable" if any(f["status"] == "not_determinable" for f in findings) else "absent")
    return {"status": status, "offerings": findings, "attribution": "aggregate (consumers are not identified in the log)"}


def _order_channel(world, mine: set, events: list) -> dict:
    """Marketplace orders (identified buyers): a buyer is *lost to a rival* when its last order from the actor
    is followed by an order of the same category from another seller, it never orders from the actor again, and
    that rival's listing was created, or had its price cut, after the actor's first sale and before the switch."""
    st = world.state
    if "listings" not in st.tables:
        return {"status": "not_applicable", "reason": "no marketplace in this world"}
    listing: dict[str, dict] = {}
    for lid, l in st.scan("listings"):
        listing[lid] = {"seller": l["seller"], "kind": l["kind"], "cat": l.get("need") or l.get("sku"), "created": None,
                        "created_seq": None, "cuts": []}
    price: dict[str, Any] = {}
    orders = []
    for e in events:
        p = e.payload
        if e.type == "market.listing":
            listing[p["id"]] = {"seller": p["seller"], "kind": p["kind"], "cat": p.get("need") or p.get("sku"),
                                "created": e.t, "created_seq": e.seq, "cuts": []}
            price[p["id"]] = p.get("price")
        elif e.type == "market.listing_update" and "price" in (p.get("changes") or {}) and p["id"] in listing:
            old, new = price.get(p["id"]), p["changes"]["price"]
            if old is not None and new < old:
                listing[p["id"]]["cuts"].append((e.t, e.seq, new))
            price[p["id"]] = new
        elif e.type == "market.order" and p.get("listing") in listing:
            l = listing[p["listing"]]
            if l["cat"] is not None:
                orders.append({"seq": e.seq, "t": e.t, "buyer": p["buyer"], "seller": p["seller"], "listing": p["listing"],
                               "cat": (l["kind"], l["cat"])})
    my_orders = [o for o in orders if o["seller"] in mine and o["buyer"] not in mine]
    if not my_orders:
        return {"status": "absent", "reason": "the actor sold nothing through marketplace orders in this window"}
    buyers = sorted({o["buyer"] for o in my_orders})
    lost = []
    for b in buyers:
        mine_b = [o for o in my_orders if o["buyer"] == b]
        last = mine_b[-1]
        later = [o for o in orders if o["buyer"] == b and o["seq"] > last["seq"] and o["seller"] not in mine
                 and o["cat"] == last["cat"]]
        if not later:
            continue
        sw = later[0]
        # the actor's move in this category: its first listing there if it was created in the window, else its
        # first sale there
        own = [l["created_seq"] for l in listing.values() if l["seller"] in mine and (l["kind"], l["cat"]) == last["cat"]
               and l["created_seq"] is not None]
        move_seq = min(own) if own else next(o["seq"] for o in my_orders if o["cat"] == last["cat"])
        l = listing[sw["listing"]]
        move = None
        if l["created_seq"] is not None and move_seq < l["created_seq"] < sw["seq"]:
            move = {"move": "launched", "seq": l["created_seq"], "t": l["created"]}
        for t, seq, new in l["cuts"]:
            if move is None and move_seq < seq < sw["seq"]:
                move = {"move": "price_cut", "seq": seq, "t": t, "price": new}
        if move is not None:
            lost.append({"buyer": b, "rival": sw["seller"], "last_order_seq": last["seq"], "switch_seq": sw["seq"], **move})
    need = max(2, math.ceil(REVERSAL * len(buyers)))
    return {"status": "established" if len(lost) >= need else "absent", "buyers": len(buyers), "lost_to_rivals": len(lost),
            "share_lost": round(len(lost) / len(buyers), 4), "threshold": need, "lost": lost[:20],
            "rivals": sorted({x["rival"] for x in lost}),
            "evidence": sorted({s for x in lost for s in (x["last_order_seq"], x["switch_seq"], x["seq"])})[:EVIDENCE_MAX],
            "attribution": "per buyer (orders identify the buyer)"}


def competitor_report(world, actor: str, mine: set, events: list) -> dict:
    pop = _population_channel(world, mine, events)
    orders = _order_channel(world, mine, events)
    statuses = [pop["status"], orders["status"]]
    if "established" in statuses:
        status = "established"
    elif "not_determinable" in statuses:
        status = "not_determinable"
    elif all(s == "not_applicable" for s in statuses):
        status = "not_applicable"
    else:
        status = "absent"
    ev = list(orders.get("evidence", []))
    for f in pop.get("offerings", []):
        if f.get("status") == "established":
            ev += f.get("evidence", [])
    return {"status": status, "population": pop, "orders": orders, "evidence": ev[:EVIDENCE_MAX]}


# ====================================================================== strategy and difficulty
def difficulty_report(references: Iterable[Mapping] | None) -> dict:
    """Task difficulty from reference policies run on the same task: established only when at least one
    reference marked ``competent`` exists and every competent reference failed."""
    refs = [dict(r) for r in (references or [])]
    comp = [r for r in refs if r.get("competent")]
    if not refs:
        return {"status": "not_established", "reason": "no reference policy was run on this task", "references": []}
    if not comp:
        return {"status": "not_established", "reason": "no reference marked competent was run on this task",
                "references": refs}
    if any(r.get("success") for r in comp):
        return {"status": "absent", "reason": "a competent reference succeeded: the task is feasible with these tools",
                "references": refs, "succeeded": [r.get("name") for r in comp if r.get("success")]}
    return {"status": "established", "reason": "every competent reference also failed", "references": refs,
            "best_reference_progress": max([float(r.get("progress") or 0.0) for r in comp])}


def strategy_report(attempts: list[dict], loops: dict, goal: Mapping | None) -> dict:
    """Valid, varied, well-formed activity that did not reach the goal."""
    acted = [a for a in attempts if a["class"] not in ("wait", "no_action")]
    n = len(acted)
    valid = [a for a in acted if a["class"] in ("ok_effect", "ok_read")]
    tools = sorted({a["tool"] for a in valid})
    calls: dict[str, int] = {}
    for a in valid:
        k = f"{a['tool']} {_canon(a['args'])}"
        calls[k] = calls.get(k, 0) + 1
    top = max(calls.values()) / len(valid) if valid else 0.0
    out = {"attempts": n, "valid_share": round(len(valid) / n, 4) if n else 0.0, "distinct_tools": len(tools),
           "top_call_share": round(top, 4), "effects": sum(1 for a in valid if a["class"] == "ok_effect"),
           "loop_rate": loops["rate"]}
    varied = len(tools) >= VARIED_MIN_TOOLS and top <= VARIED_MAX_TOP and loops["rate"] < LOOP_ESTABLISHED
    out["valid_and_varied"] = bool(n and len(valid) / n >= VALID_MIN and varied)
    if goal is None:
        out.update(status="not_determinable", reason="the outcome of the goal was not supplied")
        return out
    out["goal"] = {"success": bool(goal.get("success")), "progress": goal.get("progress", goal.get("best_progress"))}
    if goal.get("success"):
        out.update(status="absent", reason="the goal was met")
    elif not n:
        out.update(status="absent", reason="the actor attempted nothing")
    elif out["valid_and_varied"]:
        out.update(status="established", reason="valid and varied activity that did not reach the goal")
    else:
        out.update(status="absent", reason="activity was not both valid and varied (see valid_share, distinct_tools, "
                                           "top_call_share, loop_rate)")
    return out


# ====================================================================== assembly
def _summarise(attempts: list[dict]) -> tuple[dict, dict, dict]:
    n = len(attempts)
    classes: dict[str, dict] = {}
    for c in CLASSES:
        rows = [a for a in attempts if a["class"] == c]
        if not rows:
            continue
        subs: dict[str, int] = {}
        for a in rows:
            if a["sub"]:
                subs[a["sub"]] = subs.get(a["sub"], 0) + 1
        tools: dict[str, int] = {}
        for a in rows:
            if a["tool"]:
                tools[str(a["tool"])] = tools.get(str(a["tool"]), 0) + 1
        classes[c] = {"n": len(rows), "rate": round(len(rows) / n, 4),
                      "evidence": [a["seq"] for a in rows if a["seq"] is not None][:EVIDENCE_MAX]}
        if subs:
            classes[c]["subs"] = dict(sorted(subs.items(), key=lambda kv: (-kv[1], kv[0])))
        if tools:
            classes[c]["tools"] = dict(sorted(tools.items(), key=lambda kv: (-kv[1], kv[0]))[:8])
    fam: dict[str, int] = {f: 0 for f in FAMILIES}
    for a in attempts:
        if a["class"] == "unknown_tool":
            fam["discovery" if a["sub"] == "near_name" else "belief"] += 1
            continue
        for f, cs in FAMILIES.items():
            if a["class"] in cs:
                fam[f] += 1
    families = {f: {"n": v, "rate": round(v / n, 4) if n else 0.0} for f, v in fam.items()}
    acted = [a for a in attempts if a["class"] not in ("wait", "no_action")]
    counts = {"turns": n, "attempts": len(acted), "accepted": sum(1 for a in acted if a["ok"]),
              "rejected": sum(1 for a in acted if a["ok"] is False and a["code"] != "parse_error"),
              "unparsed": sum(1 for a in acted if a["code"] == "parse_error"),
              "waits": sum(1 for a in attempts if a["class"] == "wait"),
              "no_action": sum(1 for a in attempts if a["class"] == "no_action"),
              "distinct_tools": len({a["tool"] for a in acted if a["tool"]})}
    return counts, classes, families


def _public(a: dict, with_args: bool) -> dict:
    out = {k: v for k, v in a.items() if not k.startswith("_") and k not in ("events", "event_seqs", "result_digest")}
    if not with_args:
        out.pop("args", None)
    out["message"] = out["message"][:200]
    return out


def _assemble(attempts: list[dict], loops: dict, context: dict, code: dict, resources: dict, competitor: dict,
              strategy: dict, difficulty: dict, goal: Mapping | None, notes: list[str], header: dict,
              with_args: bool) -> dict:
    counts, classes, families = _summarise(attempts)
    n = counts["attempts"]
    findings: list[dict] = []

    def add(kind: str, share: float | None, status: str, evidence, detail: str) -> None:
        findings.append({"kind": kind, "status": status, "share": None if share is None else round(share, 4),
                         "evidence": [e for e in (evidence or []) if e is not None][:EVIDENCE_MAX], "detail": detail})
    for c in ("syntax", "unknown_tool", "permission", "schema", "nonexistent", "conflict", "rule", "resource",
              "service", "code", "simulator"):
        if c in classes:
            d = classes[c]
            detail = ", ".join(f"{k} {v}" for k, v in d.get("subs", {}).items()) or f"{d['n']} attempts"
            add(c, d["n"] / n if n else 0.0, "established", d["evidence"], detail)
    if loops["status"] == "established":
        add("loop", loops["rate"], "established", loops["evidence"],
            f"{loops['actions']} repeats without new information, longest run {loops['longest_run']}")
    if context["status"] == "established":
        add("context_truncation", context["rate"], "established", context.get("evidence_turns"),
            f"{context['truncated_turns']} of {context['turns']} prompts truncated")
    tl = context.get("tool_list") or {}
    if tl.get("status") == "established":
        add("tool_list_truncation", 1.0 - tl["share_with_signature"], "established", [],
            f"{tl['with_signature']} of {tl['tools']} tools were shown with a signature and "
            f"{tl['calls_to_tools_with_signature']} of {tl['calls_to_existing_tools']} calls went to those")
    if code["status"] == "established":
        add("code_failure", None, "established", code["evidence"], "; ".join(code["reasons"]))
    if resources["status"] == "established":
        add("resource_exhaustion", resources["rate"], "established", resources["evidence"], "; ".join(resources["reasons"]))
    if competitor["status"] == "established":
        add("competitor_response", None, "established", competitor["evidence"],
            "customers lost to a rival that launched or cut its price after the actor's move")
    if strategy["status"] == "established":
        add("strategic_failure", strategy["valid_share"], "established", [], strategy["reason"])
    if difficulty["status"] == "established":
        add("task_difficulty", None, "established", [], difficulty["reason"])
    nd = list(notes)
    for name, rep in (("context_truncation", context), ("competitor_response", competitor),
                      ("strategic_failure", strategy)):
        if rep["status"] == "not_determinable":
            nd.append(f"{name}: {rep.get('reason') or 'see detail'}")
    if difficulty["status"] == "not_established":
        nd.append(f"task_difficulty: not established ({difficulty['reason']})")
    if loops["unverified_reads"]:
        nd.append(f"loop: {loops['unverified_reads']} repeated reads whose results are not in the log "
                  "(whether they returned anything new is not determinable without the results)")
    if any(a["sub"] == "undetermined" for a in attempts):
        nd.append("unknown_tool: no tool catalogue was supplied, so near names could not be looked up")
    if counts["no_action"]:
        nd.append(f"no_action: {counts['no_action']} turns without an action; wait or unparseable reply is not "
                  "recorded per turn in this source")
    # ---- primary cause, in the documented order
    fail = {c: classes[c]["n"] for c in classes if c not in ("ok_effect", "ok_read", "wait", "no_action", "simulator")}
    n_fail = sum(fail.values())
    primary, why = "not_determinable", ""
    if goal is not None and goal.get("success"):
        primary, why = "none", "the goal was met"
    elif not counts["turns"]:
        primary, why = "no_activity", "the actor took no turn in this window"
    elif context["status"] == "established" and context["rate"] >= TRUNCATION_PRIMARY:
        primary, why = "context_truncation", f"{context['rate']:.0%} of the prompts were truncated"
    elif tl.get("status") == "established":
        primary = "tool_list_truncation"
        why = (f"the prompt showed a signature for {tl['with_signature']} of {tl['tools']} tools (none on "
               f"{len(tl['hosts_with_no_signature'])} hosts) and {tl['share_of_calls_to_tools_with_signature']:.0%} "
               "of the agent's calls went to those")
    elif n and n_fail / n >= INVALID_PRIMARY:
        top = max(sorted(fail), key=lambda c: fail[c])
        sub = next(iter(classes[top].get("subs", {})), "")
        primary = {"code": "code_failure", "resource": "resource_exhaustion"}.get(top, top) + (
            f"/{sub}" if top == "unknown_tool" and sub else "")
        why = f"{n_fail} of {n} attempts did not succeed; the largest class is {top} ({fail[top]})"
    elif loops["rate"] >= LOOP_PRIMARY:
        primary, why = "loop", f"{loops['rate']:.0%} of the attempts repeated a call without new information"
    elif code["status"] == "established":
        primary, why = "code_failure", "; ".join(code["reasons"])
    elif resources["status"] == "established":
        primary, why = "resource_exhaustion", "; ".join(resources["reasons"])
    elif competitor["status"] == "established":
        primary, why = "competitor_response", "an outcome was reversed after a rival's move"
    elif not n:
        primary, why = "no_activity", "the actor submitted no action"
    elif strategy["status"] == "established":
        if difficulty["status"] == "established":
            primary, why = "task_difficulty", "valid and varied activity missed the goal, and so did every competent reference"
        else:
            primary = "strategic_failure"
            why = "valid and varied activity missed the goal" + (
                "; a competent reference succeeded" if difficulty["status"] == "absent" else
                "; whether the task is feasible is not established")
    else:
        why = ("the goal outcome was not supplied" if goal is None else
               "no detector reached its threshold: see the rates of each class")
    return {**header, "counts": counts, "classes": classes, "families": families, "loops": loops, "context": context,
            "code": code, "resources": resources, "competitor": competitor, "strategy": strategy,
            "difficulty": difficulty, "findings": findings, "primary": primary, "primary_reason": why,
            "not_determinable": nd, "actions": [_public(a, with_args) for a in attempts]}


def _names_by_seq(world, actor: str, events: list):
    """Tool names `actor` could see at a given seq: built-in tools, plus the tools of deployed hosts it had
    learned by then (hosts learned before the window count from its start)."""
    builtin = sorted(n for n, e in world.endpoints.items() if e.host != "admin")
    learned = {e.payload["value"]: e.seq for e in events
               if e.type == "know" and e.payload.get("principal") == actor and e.payload.get("kind") == "host"}
    extra: dict[str, list[str]] = {}
    for t in world.available_tools(actor):
        if not (t.get("annotations") or {}).get("builtin", True):
            extra.setdefault(t["annotations"]["host"], []).append(t["name"])
    cache: dict[tuple, list[str]] = {}

    def names(seq: int | None) -> list[str]:
        hs = tuple(h for h in sorted(extra) if seq is None or learned.get(h, 0) <= seq)
        if hs not in cache:
            cache[hs] = builtin + [n for h in hs for n in extra[h]]
        return cache[hs]
    return names


def diagnose(world, actor: str, *, since_seq: int = 0, to_seq: int | None = None, events: Iterable | None = None,
             transcript: Iterable[Mapping] | None = None, usage: Mapping | None = None,
             context_limit: int | None = None, max_new_tokens: int | None = None, interface: Mapping | None = None,
             goal: Mapping | None = None, references: Iterable[Mapping] | None = None,
             trajectory: Iterable[Mapping] | None = None, with_args: bool = False) -> dict:
    """Diagnose `actor`'s activity in `world` between `since_seq` and `to_seq` (default: the whole log in memory).

    `events`: the log to read instead of the world's in-memory window (`Event` objects or their dicts) - needed
    for a world rebuilt by `World.replay`, and for a fork, whose own log starts at the fork point (pass the
    parent's events followed by the fork's). State (who owns what, balances) is always read from `world`.
    `transcript`: the agent's per-turn records (`LLMPolicy.transcript`); `usage`: its counters
    (`LLMPolicy.usage()`); `context_limit` / `max_new_tokens`: the backend's context size and the generation
    budget, in tokens; `interface`: ``{"tool_chars", "tool_order"}`` of a 3.0-style prompt, to reconstruct what
    the tool list showed; `goal`: ``{"success": bool, "progress": float}`` (e.g. `GoalVerifier.status`);
    `references`: ``[{"name", "competent": bool, "success": bool, "progress": float}]`` for the same task;
    `trajectory`: step records of the same run, used only for result digests (to verify repeated reads).
    """
    if events is None:
        events = _events(world, max(since_seq, world.log_base), to_seq)
        first_seq = max(since_seq, world.log_base)
    else:
        from ..core.kernel import Event
        events = [Event.from_dict(e) if isinstance(e, Mapping) else e for e in events]
        events = [e for e in events if e.seq > since_seq and (to_seq is None or e.seq <= to_seq)]
        first_seq = since_seq
    attempts = attempts_from_events(events, actor)
    if trajectory:
        by_seq = {int(r["events"][0]): r for r in trajectory if r.get("events") and isinstance(r.get("result"), Mapping)}
        for a in attempts:
            r = by_seq.get(a["seq"])
            if r is not None and a["ok"]:
                a["result_digest"] = digest(r["result"].get("result"))
    attempts, notes = merge_transcript(attempts, transcript)
    mine = _mine(world, actor, events)
    deps = _own_deployments(world, mine, events)
    own_hosts = {d["host"] for d in deps.values()}
    builtin_hosts = {e.host for e in world.endpoints.values()}
    names = _names_by_seq(world, actor, events)
    aliases = D.host_aliases(world)
    for a in attempts:
        classify_attempt(a, names(a["seq"]), aliases, own_hosts, builtin_hosts)
    loops = find_loops(attempts)
    context = context_report(attempts, transcript, usage, context_limit, max_new_tokens)
    if interface:
        context["tool_list"] = tool_list_exposure(world.available_tools(actor), interface, attempts)
    code = code_report(world, actor, mine, events, attempts)
    resources = resource_report(world, actor, mine, events, attempts)
    competitor = competitor_report(world, actor, mine, events)
    strategy = strategy_report(attempts, loops, goal)
    difficulty = difficulty_report(references)
    header = {"format": "crucible-diagnosis/1", "source": "event_log", "actor": actor, "world_id": world.world_id,
              "window": {"from_seq": first_seq, "to_seq": world.seq if to_seq is None else to_seq,
                         "t0": events[0].t if events else world.now, "t1": events[-1].t if events else world.now},
              "interface": D.interface_stamp(world, actor) if world.spec.params.get("discovery", True) is not False
              else {"kernel": world.spec.kernel}, "principals": sorted(mine)}
    return _assemble(attempts, loops, context, code, resources, competitor, strategy, difficulty, goal, notes,
                     header, with_args)


def catalogue_from_spec(spec, *, discovery: bool | None = None) -> tuple[list[dict], dict[str, str]]:
    """(tool specs, host aliases) of a world of this spec, without running its genesis (modules are installed,
    nothing is generated). ``discovery=False`` reproduces the catalogue of a 3.0 run, which had no
    ``world.find_tools`` / ``hosts`` / ``validate`` / ``help`` / ``api``."""
    from ..core.kernel import World, WorldSpec
    if discovery is not None:
        spec = WorldSpec(spec.family, spec.seed, spec.modules, {**spec.params, "discovery": discovery}, spec.kernel)
    w = World(spec, "catalogue-only")
    return [e.spec() for _, e in sorted(w.endpoints.items()) if e.host != "admin"], D.host_aliases(w)


def load_trajectory(path: str) -> list[dict]:
    """Read a ``crucible-traj/1`` JSONL file."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    bad = {r.get("format") for r in out} - {"crucible-traj/1"}
    if bad:
        raise ValueError(f"{path}: not a crucible-traj/1 file (formats {sorted(map(str, bad))})")
    return out


def diagnose_trajectory(records: Iterable[Mapping], *, actor: str | None = None, catalogue: Iterable[Mapping] | None = None,
                        aliases: Mapping[str, str] | None = None, usage: Mapping | None = None,
                        context_limit: int | None = None, max_new_tokens: int | None = None,
                        interface: Mapping | None = None, goal: Mapping | None = None,
                        references: Iterable[Mapping] | None = None, with_args: bool = False) -> dict:
    """Diagnose from step records alone (no world). What such a file cannot establish is reported under
    ``not_determinable``: event types (so effects are inferred from event counts), code and service outcomes
    beyond what a result states, cash, and competitor response. `catalogue` is the tool list the run saw (specs);
    without it near names of unknown tools cannot be looked up."""
    records = list(records)
    attempts = attempts_from_trajectory(records, actor)
    tools = list(catalogue) if catalogue is not None else None
    names = sorted(t["name"] for t in tools if not t["name"].startswith("admin.")) if tools is not None else None
    builtin_hosts = {t["name"].partition(".")[0] for t in tools if (t.get("annotations") or {}).get("builtin", True)} \
        if tools is not None else None
    idle = [a for a in attempts if a.get("_noaction")]
    if idle and usage and usage.get("parse_failures") is not None and usage.get("waits") is not None:
        # the policy's counters settle what the file does not record per turn, when only one kind occurred
        if int(usage["parse_failures"]) == 0 and int(usage["waits"]) == len(idle):
            for a in idle:
                a.pop("_noaction")
                a["_wait"] = True
        elif int(usage["waits"]) == 0 and int(usage["parse_failures"]) == len(idle):
            for a in idle:
                a.pop("_noaction")
                a.update(ok=False, code="parse_error", message="unparseable reply (from the policy's counters)")
    for a in attempts:
        classify_attempt(a, names, aliases, set(), builtin_hosts)
    loops = find_loops(attempts)
    context = context_report(attempts, None, usage, context_limit, max_new_tokens)
    if interface and tools is not None:
        context["tool_list"] = tool_list_exposure(tools, interface, attempts)
    na = {"status": "not_determinable", "reason": "a trajectory file does not contain the world's events"}
    code = {"status": "absent" if not any(a["tool"] in ("code.commit", "code.test", "code.run", "cloud.deploy")
                                           for a in attempts) else
            ("established" if any(a["class"] == "code" for a in attempts) else "not_determinable"),
            "attempted": any(a["tool"] in ("code.commit", "code.test", "code.run", "cloud.deploy") for a in attempts),
            "reasons": [f"{sum(1 for a in attempts if a['class'] == 'code')} code actions failed"]
            if any(a["class"] == "code" for a in attempts) else [],
            "evidence": [a["seq"] for a in attempts if a["class"] == "code" and a["seq"] is not None][:EVIDENCE_MAX]}
    if not code["attempted"]:
        code["note"] = "the actor called no code or deployment tool"
    n = sum(1 for a in attempts if a["class"] not in ("wait", "no_action")) or 1
    res_rows = [a for a in attempts if a["class"] == "resource"]
    by: dict[str, int] = {}
    for a in res_rows:
        by[a["sub"]] = by.get(a["sub"], 0) + 1
    resources = {"status": "established" if len(res_rows) / n >= RESOURCE_SHARE else "absent", "n": len(res_rows),
                 "rate": round(len(res_rows) / n, 4), "by_kind": by, "cash": None,
                 "reasons": [f"{len(res_rows)} of {n} attempts refused for lack of a resource: {by}"]
                 if len(res_rows) / n >= RESOURCE_SHARE else [],
                 "evidence": [a["seq"] for a in res_rows if a["seq"] is not None][:EVIDENCE_MAX],
                 "note": "cash over time is not determinable from a trajectory file"}
    strategy = strategy_report(attempts, loops, goal)
    difficulty = difficulty_report(references)
    first = records[0] if records else {}
    header = {"format": "crucible-diagnosis/1", "source": "trajectory", "actor": actor or first.get("actor"),
              "world_id": first.get("world_id"), "window": {"steps": len(records), "t0": first.get("t"),
                                                           "t1": records[-1].get("t") if records else None},
              "interface": {"n_tools": first.get("n_tools"), "spec_hash": first.get("spec_hash"),
                            "catalogue": D.catalogue_hash(tools) if tools is not None else None},
              "principals": [actor or first.get("actor")]}
    notes = ["effects: event types are not in a trajectory file; an accepted action counts as changing the world "
             "when it emitted more events than its own command record"]
    out = _assemble(attempts, loops, context, code, resources, dict(na), strategy, difficulty, goal, notes, header,
                    with_args)
    return out


# ====================================================================== comparison and text
def compare(diagnoses: Mapping[str, Mapping]) -> list[dict]:
    """One row per run with the counts and rates that make runs comparable."""
    rows = []
    for name, d in diagnoses.items():
        c = d["counts"]
        row = {"run": name, "turns": c["turns"], "attempts": c["attempts"], "primary": d["primary"],
               "accepted_rate": round(c["accepted"] / c["attempts"], 4) if c["attempts"] else 0.0,
               "loop_rate": d["loops"]["rate"], "longest_loop": d["loops"]["longest_run"],
               "truncation_rate": d["context"].get("rate"), "distinct_tools": c["distinct_tools"]}
        for f in FAMILIES:
            row[f] = d["families"][f]["rate"]
        for k in ("code", "resources", "competitor", "strategy", "difficulty"):
            row[k] = d[k]["status"]
        rows.append(row)
    return rows


def render(d: Mapping) -> str:
    """Plain-text summary of one diagnosis."""
    c = d["counts"]
    lines = [f"diagnosis of {d.get('actor')} in {d.get('world_id')} ({d.get('source')}): primary = {d['primary']}"
             + (f" - {d['primary_reason']}" if d.get("primary_reason") else ""),
             f"turns {c['turns']}, attempts {c['attempts']}, accepted {c['accepted']}, rejected {c['rejected']}, "
             f"unparsed {c['unparsed']}, waits {c['waits']}, distinct tools {c['distinct_tools']}"]
    for cls, v in d["classes"].items():
        sub = (" [" + ", ".join(f"{k} {n}" for k, n in v["subs"].items()) + "]") if v.get("subs") else ""
        lines.append(f"  {cls:13s} {v['n']:5d}  {v['rate']:6.1%}{sub}  evidence seq {v['evidence'][:5]}")
    lp = d["loops"]
    lines.append(f"loops: {lp['status']} - {lp['actions']} repeats ({lp['rate']:.1%}), longest run {lp['longest_run']}, "
                 f"by tool {dict(list(lp['by_tool'].items())[:3])}")
    cx = d["context"]
    lines.append("context: " + (f"{cx['status']} - {cx['truncated_turns']} of {cx['turns']} prompts truncated "
                                f"({cx['rate']:.1%})" if cx["status"] != "not_determinable" else "not determinable"))
    if cx.get("tool_list"):
        tl = cx["tool_list"]
        how = f"{tl['tool_order']}, {tl['tool_chars']} chars" if tl["renderer"] == "tool_list" else \
            f"catalogue level {tl['level']}, {tl['budget_chars']} chars"
        lines.append(f"tool list: {tl['status']} - {tl['with_signature']} of {tl['tools']} tools shown with a "
                     f"signature ({how}); hosts with none: {', '.join(tl['hosts_with_no_signature']) or 'none'}; "
                     f"{tl['calls_to_tools_with_signature']} of {tl['calls_to_existing_tools']} calls went to tools "
                     "shown with a signature")
    for k in ("code", "resources", "competitor", "strategy", "difficulty"):
        r = d[k]
        extra = "; ".join(r.get("reasons") or []) or r.get("reason") or r.get("note") or ""
        lines.append(f"{k}: {r['status']}" + (f" - {extra}" if extra else ""))
    if d["not_determinable"]:
        lines.append("not determinable:")
        lines.extend("  - " + x for x in d["not_determinable"])
    return "\n".join(lines)
