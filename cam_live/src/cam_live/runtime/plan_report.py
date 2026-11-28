"""cam_live.runtime.plan_report — the operator's plan report of a run (``cam-live plan-report``; OP-11, INT3B).

    cam-live plan-report --run-dir R [--out F] [--json]

An export, not a summary. Everything CAM wrote is reproduced verbatim and unedited, each item with its provenance
(the decision that produced it, the record id, the time): the objective it received, its notes (the plan,
hypotheses, lessons — grouped only by the tags CAM itself gave them, in its own order), the predictions it registered
(with their resolutions where they matured), the experiments of its registry, its requests to the operator with
their justifications and current status, the actions executed / queued for approval / refused, and the learning
updates (counts, weight-delta norms, reward versions). Nothing here is written by anything other than CAM: the only
additions are counts, record identifiers and timestamps. Every text — CAM's own included — passes the redaction the
run's logs apply (``runtime.logs.Redactor`` with the run's salt, data permissions and configured ids): members'
handles, e-mail addresses, platform ids and other personal identifiers become the same keyed hashes / markers as in
the logs, credentials are removed (G2R-7b); the number of redactions is reported per section. Apart from those spans
CAM's texts (notebook, requests) are byte-exact in the JSON form; the Markdown form shows every CAM-written string
literally (code fences / spans) with control characters escaped, so nothing CAM wrote becomes report structure or
terminal control (G2R-7, R5-m2). Fields that come from the run's logs (actions, results) are as the run recorded them.

Sources (all read-only): ``task.json``, ``events.jsonl``, ``decisions.jsonl``, ``predictions.jsonl``,
``experiments.jsonl``, ``learning_updates.jsonl``, ``meta_updates.jsonl``, ``state.json`` (via ``RunReader``), the
internal executor's notebook (``<exec>/workspace/notes.jsonl``: the text as saved) and request file
(``operator_requests.jsonl``: requests and the operator's answers).
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional

from ..contracts import RunFiles
from .logs import RunReader, iter_jsonl, read_json

REPORT_VERSION = 1
NOTE_OP, REQUEST_OP = "internal.note", "internal.request_operator"
#: result kinds of the runtime (``LiveRuntime._complete``): what happened to a chosen action
EXECUTED_KINDS = ("execute", "wait", "inflight_recovered")
QUEUED_KINDS = ("queued",)


def _iso(ts: Any) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))
    except (TypeError, ValueError, OverflowError):
        return ""


def _rows(path: str) -> List[Dict[str, Any]]:
    return [r for r in iter_jsonl(path)] if os.path.exists(path) else []


def _find(run_dir: str, name: str) -> str:
    """A file the computer environment keeps: ``<run>/exec/...`` (in-process and separated runs), else the run."""
    for d in (os.path.join(run_dir, "exec"), run_dir, os.path.join(run_dir, "workspace")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return ""


def _latest_requests(path: str) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    revisions: Dict[str, int] = {}
    for r in _rows(path):
        rid = r.get("request_id", "")
        latest[rid] = dict(latest.get(rid, {}), **r)
        revisions[rid] = revisions.get(rid, 0) + 1
    out = sorted(latest.values(), key=lambda r: r.get("ts", 0.0))
    for r in out:
        r["revisions"] = revisions.get(r.get("request_id", ""), 1)
    return out


def _count(xs: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


def _redactor(run_dir: str, st: Dict[str, Any]) -> Any:
    """The run logs' redactor (G2R-7b): the run's own salt (so a member's hash matches the logs'), the task's data
    permissions and the identifiers the operator configured (guild, channels), which stay readable."""
    from ..harness import safefs
    from .logs import Redactor
    try:
        salt = safefs.read_nofollow(os.path.join(run_dir, ".redaction_salt"), 4096)
    except Exception:  # noqa: BLE001 — no readable salt: a fresh one (hashes then differ from the logs')
        salt = b""
    task = st.get("task") if isinstance(st.get("task"), dict) else {}
    r = task.get("resources") if isinstance(task.get("resources"), dict) else {}
    d = r.get("discord") if isinstance(r.get("discord"), dict) else {}
    keep = [r.get("discord_guild_id"), d.get("guild_id"), d.get("application_id"), d.get("metric_channel_id")]
    keep += list((d.get("channels") or {}).values()) if isinstance(d.get("channels"), dict) else []
    keep = [str(x) for x in keep if x not in (None, "") and not str(x).upper().startswith("REQUIRED")]
    return Redactor.from_config({}, salt or os.urandom(32), task.get("data_permissions") or {}, keep_ids=keep)


def build_report(run_dir: str) -> Dict[str, Any]:
    run_dir = os.path.abspath(run_dir)
    r = RunReader(run_dir)
    st = r.state() or {}
    task = read_json(os.path.join(run_dir, "task.json"), {}) or {}
    view = task.get("learner_view") or {}
    events = r.records(RunFiles.EVENTS)
    decisions = r.records(RunFiles.DECISIONS)
    preds = r.records(RunFiles.PREDICTIONS)
    exps = r.records(RunFiles.EXPERIMENTS)
    learning = r.records(RunFiles.LEARNING)
    metas = r.records(RunFiles.META)

    red = _redactor(run_dir, st)
    redactions: Dict[str, int] = {}
    section = ["objective"]

    def clean(x: Any) -> Any:
        """G2R-7b: the run logs' redaction (personal identifiers, credentials), counted per section; otherwise the
        text is exported byte-exact (G2R-7)."""
        before = sum(red.counts.values())
        out = red.text(x) if isinstance(x, str) else red(x) if isinstance(x, (dict, list)) else x
        n = sum(red.counts.values()) - before
        redactions[section[0]] = redactions.get(section[0], 0) + n
        return out

    dec_by_action: Dict[str, Dict[str, Any]] = {}
    for d in decisions:
        c = (d["data"].get("decision") or {}).get("chosen") or {}
        dec_by_action[c.get("action_id", "")] = d
    results = [e for e in events if e["type"] == "action_result"]
    res_by_action: Dict[str, Dict[str, Any]] = {}
    for e in results:
        res_by_action[e["data"].get("action_id", "")] = e         # the latest result of an action (after approval)

    def prov(d: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if d is None:
            return dict(extra or {})
        dd = d["data"].get("decision") or {}
        return {"decision_id": dd.get("decision_id", ""), "decision_record": d.get("rid", ""),
                "time": d.get("ts"), "time_utc": _iso(d.get("ts")), **(extra or {})}

    # ---- objective ------------------------------------------------------------------------------------------
    started = next((e for e in events if e["type"] == "run_started"), None)
    assigned = [e for e in events if e["type"] == "objective_assigned"]
    objective = {"text": clean(view.get("objective", "")), "objective_id": view.get("objective_id", ""),
                 "mode": view.get("mode", ""), "task_sha256": task.get("task_sha256", ""),
                 "received": [{"record": e.get("rid", ""), "time": e.get("ts"), "time_utc": _iso(e.get("ts")),
                               "event": e["type"], "objective_id": (e["data"].get("task") or {}).get("objective_id"),
                               "text": clean((e["data"].get("task") or {}).get("objective", ""))}
                              for e in ([started] if started else []) + assigned],
                 "withheld_from_learner": task.get("withheld_from_learner", [])}

    # ---- notes (the text as saved by the notebook, joined to the decision that wrote it) --------------------
    section[0] = "notes"
    saved = {n.get("action_id", ""): n for n in _rows(_find(run_dir, os.path.join("workspace", "notes.jsonl")))}
    notes: List[Dict[str, Any]] = []
    for d in decisions:
        c = (d["data"].get("decision") or {}).get("chosen") or {}
        if c.get("operation") != NOTE_OP:
            continue
        a = c.get("arguments") or {}
        res = res_by_action.get(c.get("action_id", ""))
        sv = saved.get(c.get("action_id", ""))
        notes.append({"text": clean((sv or {}).get("text", a.get("text", ""))),
                      "tags": [clean(t) for t in ((sv or a).get("tags") or [])],
                      "saved": sv is not None, "note_id": (sv or {}).get("note_id", ""),
                      "result": None if res is None else {"kind": res["data"].get("kind"),
                                                          "status": res["data"].get("status")},
                      "provenance": prov(d, {"action_id": c.get("action_id", ""),
                                             "candidate_source": c.get("provenance", "")})})
    by_tag: Dict[str, List[int]] = {}
    for i, n in enumerate(notes):
        for t in (n["tags"] or ["(untagged)"]):
            by_tag.setdefault(t, []).append(i)

    # ---- predictions ----------------------------------------------------------------------------------------
    resolutions: Dict[str, Dict[str, Any]] = {}
    for p in preds:
        if p["type"] == "resolution":
            resolutions[p["data"].get("prediction_id", "")] = p
    registered = []
    for p in preds:
        if p["type"] != "prediction":
            continue
        pr = p["data"].get("prediction") or {}
        q = pr.get("query") or {}
        res = resolutions.get(pr.get("prediction_id", ""))
        registered.append({"prediction_id": pr.get("prediction_id", ""), "metric": q.get("metric", ""),
                           "kind": q.get("kind", ""), "horizon_s": q.get("horizon_s"), "action_id": pr.get("action_id"),
                           "for_chosen_action": bool(p["data"].get("for_chosen_action")),
                           "distribution": pr.get("dist") or {}, "made_at": pr.get("made_at"),
                           "model_version": pr.get("model_version", ""),
                           "resolution": None if res is None else {k: res["data"].get(k) for k in
                                                                   ("status", "value", "observed_at", "verified",
                                                                    "outcome_rid")},
                           "provenance": {"decision_id": p["data"].get("decision_id", ""), "record": p.get("rid", ""),
                                          "time": p.get("ts"), "time_utc": _iso(p.get("ts"))}})

    # ---- experiments (every revision of the registry, in order) ---------------------------------------------
    section[0] = "experiments"
    experiments = [{"record": e.get("rid", ""), "time": e.get("ts"), "time_utc": _iso(e.get("ts")), "type": e["type"],
                    "data": clean(e["data"])} for e in exps]

    # ---- operator requests --------------------------------------------------------------------------------
    section[0] = "operator_requests"
    requests = []
    for q in _latest_requests(_find(run_dir, "operator_requests.jsonl")):
        d = dec_by_action.get(q.get("action_id", ""))
        requests.append({k: clean(q.get(k)) if k in ("kind", "title", "description", "justification", "resolution_note")
                         else q.get(k) for k in ("request_id", "kind", "title", "description", "justification",
                                                 "blocking", "status", "resolution_note", "resolved_by", "revisions")}
                        | {"requested_at": q.get("ts"), "requested_at_utc": _iso(q.get("ts")),
                           "resolved_at": q.get("resolved_at"), "provenance": prov(d, {"action_id": q.get("action_id")})})

    # ---- actions --------------------------------------------------------------------------------------------
    actions = []
    for d in decisions:
        dd = d["data"].get("decision") or {}
        c = dd.get("chosen") or {}
        res = res_by_action.get(c.get("action_id", ""))
        rd = (res or {}).get("data") or {}
        kind = rd.get("kind", "")
        group = ("executed" if kind in EXECUTED_KINDS and rd.get("status") in ("ok", "failed", "timeout") else
                 "queued" if kind in QUEUED_KINDS else "no_result" if res is None else "refused")
        section[0] = "actions"
        actions.append({"operation": clean(c.get("operation", "")), "arguments": clean(c.get("arguments", {})),
                        "candidate_source": c.get("provenance", ""), "group": group, "result_kind": kind,
                        "status": rd.get("status", ""), "approval_id": rd.get("approval_id", ""),
                        "failure_class": rd.get("failure_class", ""), "failure_detail": clean(rd.get("failure_detail", "")),
                        "selection": dd.get("exploration", ""), "provenance": prov(d, {"action_id": c.get("action_id")})})
    section[0] = "approvals"
    approvals = [{"approval_id": k, **{x: clean(v.get(x)) if x in ("operation", "reasons") else v.get(x)
                                       for x in ("status", "operation", "queued_at", "resolved_at", "decision_id",
                                                 "reasons", "by")}}
                 for k, v in (st.get("approvals") or {}).items()]

    # ---- learning updates -----------------------------------------------------------------------------------
    norms: Dict[str, Dict[str, float]] = {}
    for u in learning:
        ts = u["data"].get("timescale", "")
        for g, v in (u["data"].get("weight_delta") or {}).items():
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            s = norms.setdefault(f"{ts}:{g}", {"n": 0, "nonzero": 0, "sum": 0.0, "max": 0.0})
            s["n"] += 1
            s["nonzero"] += int(x > 0)
            s["sum"] += x
            s["max"] = max(s["max"], x)
    reward_versions = []
    for m in metas:
        md = m["data"]
        for v in (md.get("reward_version_before"), md.get("reward_version_after")):
            if v and v not in reward_versions:
                reward_versions.append(v)
    learning_summary = {
        "n_updates": len(learning), "by_timescale": _count(u["data"].get("timescale", "") for u in learning),
        "with_behavioural_change": sum(1 for u in learning if (u["data"].get("prediction_change") or {})),
        "weight_delta_norms": norms,
        "first_update": None if not learning else {"record": learning[0].get("rid"), "time_utc": _iso(learning[0].get("ts"))},
        "last_update": None if not learning else {"record": learning[-1].get("rid"), "time_utc": _iso(learning[-1].get("ts")),
                                                  "model_version": learning[-1]["data"].get("model_version", "")},
        "meta_updates": {"n": len(metas), "applied": sum(1 for m in metas if m["data"].get("applied")),
                         "skipped_reasons": _count(str(m["data"].get("skipped_reason", "")).split(":")[0]
                                                   for m in metas if not m["data"].get("applied")),
                         "meta_grad_norms_applied": [m["data"].get("meta_grad_norm") for m in metas
                                                     if m["data"].get("applied")],
                         "reward_versions": reward_versions,
                         "bootstrap_mix_last": metas[-1]["data"].get("bootstrap_mix") if metas else None},
    }
    counters = st.get("counters") or {}
    return {
        "report": "cam-live plan report", "version": REPORT_VERSION, "generated_at": time.time(),
        "generated_at_utc": _iso(time.time()), "run_dir": run_dir, "run_id": st.get("run_id", ""),
        "status": st.get("status", ""), "status_reason": st.get("status_reason", ""),
        "authorship": "Every quoted text was written by CAM (or, for request answers, by the operator) and is "
                      "reproduced unedited from the run's records; this report adds counts, identifiers and times "
                      "only. Notes are grouped by CAM's own tags.",
        "objective": objective,
        "counts": {"decisions": len(decisions), "notes": len(notes), "predictions_registered": len(registered),
                   "predictions_resolved": sum(1 for p in registered if p["resolution"]),
                   "experiment_records": len(experiments), "operator_requests": len(requests),
                   "actions_by_group": _count(a["group"] for a in actions),
                   "actions_by_operation": _count(a["operation"] for a in actions), "counters": counters},
        "redaction": "every text passed the run logs' redaction (personal identifiers -> the run's keyed hashes or "
                     "markers, credentials removed); counts per section",
        "redactions": {k: redactions.get(k, 0) for k in ("objective", "notes", "operator_requests", "actions",
                                                          "experiments", "approvals")},
        "notes": notes, "notes_by_tag": by_tag,
        "predictions": registered, "experiments": experiments, "operator_requests": requests,
        "actions": actions, "approvals": approvals, "learning": learning_summary,
    }


_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u00ad\u061c\u180e\u200b-\u200f\u2028-\u202e\u2060-\u206f\ufeff\ufff9-\ufffb]")


def visible(text: Any) -> str:
    """G2R-7 / R5-m2: control characters (C0 / C1, ANSI escapes, bidi overrides, zero-width and format characters)
    shown as escapes (``\\x1b``, ``U+202E``) so that nothing CAM wrote can hide, reorder or rewrite what the operator
    sees; every other character is kept."""
    s = str(text if text is not None else "")
    return _CONTROL.sub(lambda m: (f"\\x{ord(m.group(0)):02x}" if ord(m.group(0)) < 0x100
                                   else f"U+{ord(m.group(0)):04X}"), s)


def _q(text: Any) -> str:
    """CAM's text as a literal block: control characters escaped, inside a code fence longer than any run of
    backticks in it — nothing in the text becomes a heading, a list item or a table of the report."""
    s = visible(text)
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", s)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{s}\n{fence}"


def _c(text: Any) -> str:
    """CAM-written (or learner-derived) text inline, e.g. a tag or an operation name in a sentence or a table cell:
    an inline code span, control characters and line breaks escaped, table pipes escaped."""
    s = visible(text).replace("\r", "\\r").replace("\n", "\\n").replace("|", "\\|")
    if not s:
        return "``"
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", s)), default=0)
    tick = "`" * (longest + 1)
    pad = " " if s.startswith("`") or s.endswith("`") else ""
    return f"{tick}{pad}{s}{pad}{tick}"


def _redline(rep: Dict[str, Any], section: str) -> str:
    n = (rep.get("redactions") or {}).get(section, 0)
    return (f"Redacted in this section by the run logs' rules (personal identifiers, credentials): {n}.\n" if n else
            "Redacted in this section by the run logs' rules: none.\n")


def render_markdown(rep: Dict[str, Any]) -> str:
    L: List[str] = [f"# Plan report — run {_c(rep.get('run_id', ''))}", "",
                    f"Generated {rep['generated_at_utc']} from {_c(rep['run_dir'])}. Run status: {_c(rep['status'])}"
                    + (f" ({_c(rep['status_reason'])})" if rep.get("status_reason") else "") + ".", "",
                    rep["authorship"], ""]
    o = rep["objective"]
    L += ["## Objective (as CAM received it)", "", _redline(rep, "objective"),
          f"Objective id {_c(o['objective_id'])}, mode {_c(o['mode'])}, task sha256 {_c(o['task_sha256'])}.", "",
          _q(o["text"]), ""]
    for e in o["received"]:
        same = "identical to the text above" if e["text"] == o["text"] else "DIFFERS from the text above"
        L.append(f"* {_c(e['event'])} {_c(e['record'])} at {e['time_utc']}: {same}")
    L += ["", "## Counts", ""]
    for k, v in rep["counts"].items():
        L.append(f"* {k}: {json.dumps(v, sort_keys=True)}")
    L += ["", f"## CAM's notes ({len(rep['notes'])})", "", _redline(rep, "notes")]
    if rep["notes_by_tag"]:
        L.append("Tags (CAM's own): " + ", ".join(f"{_c(t)} ({len(ix)})" for t, ix in rep["notes_by_tag"].items()))
        L.append("")
    for i, n in enumerate(rep["notes"], 1):
        p = n["provenance"]
        L += [f"### Note {i} — decision {p.get('decision_id', '')}, {p.get('time_utc', '')}",
              f"tags: {', '.join(_c(t) for t in n['tags']) or '(none)'}; saved: {n['saved']} {_c(n['note_id'])}; record "
              f"{p.get('decision_record', '')}; action {p.get('action_id', '')}", "", _q(n["text"]), ""]
    L += [f"## Requests to the operator ({len(rep['operator_requests'])})", "", _redline(rep, "operator_requests")]
    for q in rep["operator_requests"]:
        p = q["provenance"]
        L += [f"### {_c(q['request_id'])} — {_c(q['kind'])}, status {_c(q['status'])} (requested {q['requested_at_utc']}, "
              f"decision {p.get('decision_id', '')})", "", "Title:", _q(q["title"]), "", "Description:",
              _q(q["description"]), "", "Justification:", _q(q["justification"]), ""]
        if q.get("resolution_note"):
            L += [f"Operator's answer ({_c(q.get('resolved_by', ''))}):", _q(q["resolution_note"]), ""]
    L += [f"## Actions ({len(rep['actions'])})", "", _redline(rep, "actions"),
          "| time (UTC) | decision | operation | source | outcome | detail |",
          "|---|---|---|---|---|---|"]
    for a in rep["actions"]:
        p = a["provenance"]
        L.append(f"| {p.get('time_utc', '')} | {_c(p.get('decision_id', ''))} | {_c(a['operation'])} | "
                 f"{_c(a['candidate_source'])} | {a['group']} ({_c(a['result_kind'])}:{_c(a['status'])}) | "
                 f"{_c(a['failure_detail'] or '')} |")
    L += ["", "Arguments of every action are in the JSON form of this report (`--json`).", "",
          f"## Approvals ({len(rep['approvals'])})", "", _redline(rep, "approvals")]
    for a in rep["approvals"]:
        L.append(f"* {_c(a['approval_id'])}: {_c(a.get('status'))} — {_c(a.get('operation'))} "
                 f"(decision {_c(a.get('decision_id'))})")
    chosen = [p for p in rep["predictions"] if p["for_chosen_action"]]
    L += ["", f"## Registered predictions ({len(rep['predictions'])}; "
              f"{sum(1 for p in rep['predictions'] if p['resolution'])} resolved)", "",
          f"Below: the {len(chosen)} predictions registered for the chosen actions (those for the other considered "
          "candidates are in the JSON form).", "",
          "| registered (UTC) | decision | metric | kind | mean | std | chosen action | resolution |", "|---|---|---|---|---|---|---|---|"]
    for p in chosen:
        d = p["distribution"] or {}
        res = p["resolution"]
        rs = "" if not res else f"{res.get('status')} {res.get('value')}" + (" (verified)" if res.get("verified") else "")
        L.append(f"| {p['provenance']['time_utc']} | {_c(p['provenance']['decision_id'])} | {_c(p['metric'])} | "
                 f"{_c(p['kind'])} | {d.get('mean')} | {d.get('std')} | {p['for_chosen_action']} | {_c(rs)} |")
    L += ["", f"## Experiments ({len(rep['experiments'])} registry records)", "", _redline(rep, "experiments")]
    for e in rep["experiments"]:
        d = e["data"]
        L.append(f"* {e['time_utc']} {_c(e['type'])} {_c(d.get('experiment_id', ''))}: status {_c(d.get('status', ''))}; "
                 f"question: {_c(d.get('question', ''))}; hypothesis: {_c(d.get('hypothesis', ''))}; "
                 f"arms: {_c(json.dumps(d.get('arms', []), ensure_ascii=False))}; supported: {_c(d.get('supported'))}")
    lr = rep["learning"]
    L += ["", "## Learning updates", "", f"* updates: {lr['n_updates']} {json.dumps(lr['by_timescale'], sort_keys=True)}",
          f"* with a measured behavioural change: {lr['with_behavioural_change']}",
          f"* meta-updates: {json.dumps(lr['meta_updates'], sort_keys=True)}", "",
          "| timescale:group | updates | non-zero | sum of delta norms | largest |", "|---|---|---|---|---|"]
    for k, s in sorted(lr["weight_delta_norms"].items()):
        L.append(f"| {_c(k)} | {s['n']} | {s['nonzero']} | {s['sum']:.6g} | {s['max']:.6g} |")
    return "\n".join(L) + "\n"


def write_report(run_dir: str, out: str = "", as_json: bool = False) -> str:
    rep = build_report(run_dir)
    out = out or os.path.join(os.path.abspath(run_dir), "plan_report." + ("json" if as_json else "md"))
    body = json.dumps(rep, indent=1, sort_keys=True, default=str) if as_json or out.endswith(".json") else \
        render_markdown(rep)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # root writes this report on the operator's behalf, by default into the run directory the learner's group may
    # write: random O_EXCL|O_NOFOLLOW temporary, rename within the directory fd, a planted symlink / non-regular
    # target refused (G2-harness R4-M1; cam_live.harness.safefs)
    from ..harness import safefs
    safefs.write_atomic(out, body.encode("utf-8"), 0o644)
    return out
