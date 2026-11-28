"""cam_live.runtime.plan_report — the operator's plan report of a run (``cam-live plan-report``; OP-11, INT3B).

    cam-live plan-report --run-dir R [--out F] [--json]

An export, not a summary. Everything CAM wrote is reproduced verbatim and unedited, each item with its provenance
(the decision that produced it, the record id, the time): the objective it received, its notes (the plan,
hypotheses, lessons — grouped only by the tags CAM itself gave them, in its own order), the predictions it registered
(with their resolutions where they matured), the experiments of its registry, its requests to the operator with
their justifications and current status, the actions executed / queued for approval / refused, and the learning
updates (counts, weight-delta norms, reward versions). Nothing here is written by anything other than CAM: the only
additions are counts, record identifiers and timestamps. The run's own redaction (credentials, personal data,
message content the task does not permit) applies, because the source is the run's records — nothing else is changed.

Sources (all read-only): ``task.json``, ``events.jsonl``, ``decisions.jsonl``, ``predictions.jsonl``,
``experiments.jsonl``, ``learning_updates.jsonl``, ``meta_updates.jsonl``, ``state.json`` (via ``RunReader``), the
internal executor's notebook (``<exec>/workspace/notes.jsonl``: the text as saved) and request file
(``operator_requests.jsonl``: requests and the operator's answers).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional

from ..contracts import RunFiles
from .logs import Redactor, RunReader, iter_jsonl, load_salt, read_json

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

    # the texts of the executor's own files (notebook, request file) pass the run's redactor too (same salt, data
    # permissions and kept ids as the run's logs): credentials and personal data never leave in the report
    tview = (st.get("task") or {}) if isinstance(st.get("task"), dict) else {}
    keep = [str(x) for x in ((tview.get("resources") or {}).get("discord_guild_id"),
                             ((tview.get("resources") or {}).get("discord") or {}).get("guild_id"),
                             ((tview.get("resources") or {}).get("discord") or {}).get("application_id")) if x]
    red = Redactor.from_config({}, load_salt(run_dir), tview.get("data_permissions") or {}, keep_ids=keep)

    def clean(x: Any) -> Any:
        return red.text(x) if isinstance(x, str) else x

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
    objective = {"text": view.get("objective", ""), "objective_id": view.get("objective_id", ""),
                 "mode": view.get("mode", ""), "task_sha256": task.get("task_sha256", ""),
                 "received": [{"record": e.get("rid", ""), "time": e.get("ts"), "time_utc": _iso(e.get("ts")),
                               "event": e["type"], "objective_id": (e["data"].get("task") or {}).get("objective_id"),
                               "text": (e["data"].get("task") or {}).get("objective", "")}
                              for e in ([started] if started else []) + assigned],
                 "withheld_from_learner": task.get("withheld_from_learner", [])}

    # ---- notes (the text as saved by the notebook, joined to the decision that wrote it) --------------------
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
    experiments = [{"record": e.get("rid", ""), "time": e.get("ts"), "time_utc": _iso(e.get("ts")), "type": e["type"],
                    "data": e["data"]} for e in exps]

    # ---- operator requests --------------------------------------------------------------------------------
    requests = []
    for q in _latest_requests(_find(run_dir, "operator_requests.jsonl")):
        d = dec_by_action.get(q.get("action_id", ""))
        requests.append({k: clean(q.get(k)) for k in ("request_id", "kind", "title", "description", "justification",
                                                       "blocking", "status", "resolution_note", "resolved_by",
                                                       "revisions")}
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
        actions.append({"operation": c.get("operation", ""), "arguments": c.get("arguments", {}),
                        "candidate_source": c.get("provenance", ""), "group": group, "result_kind": kind,
                        "status": rd.get("status", ""), "approval_id": rd.get("approval_id", ""),
                        "failure_class": rd.get("failure_class", ""), "failure_detail": rd.get("failure_detail", ""),
                        "selection": dd.get("exploration", ""), "provenance": prov(d, {"action_id": c.get("action_id")})})
    approvals = [{"approval_id": k, **{x: v.get(x) for x in ("status", "operation", "queued_at", "resolved_at",
                                                             "decision_id", "reasons", "by")}}
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
        "notes": notes, "notes_by_tag": by_tag,
        "predictions": registered, "experiments": experiments, "operator_requests": requests,
        "actions": actions, "approvals": approvals, "learning": learning_summary,
    }


def _q(text: Any) -> str:
    """A verbatim block (Markdown blockquote: the text itself is not altered, each line is prefixed)."""
    s = str(text if text is not None else "")
    return "\n".join("> " + line for line in (s.split("\n") if s else [""]))


def render_markdown(rep: Dict[str, Any]) -> str:
    L: List[str] = [f"# Plan report — run {rep.get('run_id', '')}", "",
                    f"Generated {rep['generated_at_utc']} from `{rep['run_dir']}`. Run status: {rep['status']}"
                    + (f" ({rep['status_reason']})" if rep.get("status_reason") else "") + ".", "",
                    rep["authorship"], ""]
    o = rep["objective"]
    L += ["## Objective (as CAM received it)", "",
          f"Objective id `{o['objective_id']}`, mode {o['mode']}, task sha256 `{o['task_sha256']}`.", "", _q(o["text"]), ""]
    for e in o["received"]:
        same = "identical to the text above" if e["text"] == o["text"] else "DIFFERS from the text above"
        L.append(f"* {e['event']} {e['record']} at {e['time_utc']}: {same}")
    L += ["", "## Counts", ""]
    for k, v in rep["counts"].items():
        L.append(f"* {k}: {json.dumps(v, sort_keys=True)}")
    L += ["", f"## CAM's notes ({len(rep['notes'])})", ""]
    if rep["notes_by_tag"]:
        L.append("Tags (CAM's own): " + ", ".join(f"{t} ({len(ix)})" for t, ix in rep["notes_by_tag"].items()))
        L.append("")
    for i, n in enumerate(rep["notes"], 1):
        p = n["provenance"]
        L += [f"### Note {i} — decision {p.get('decision_id', '')}, {p.get('time_utc', '')}",
              f"tags: {', '.join(n['tags']) or '(none)'}; saved: {n['saved']} {n['note_id']}; record "
              f"{p.get('decision_record', '')}; action {p.get('action_id', '')}", "", _q(n["text"]), ""]
    L += [f"## Requests to the operator ({len(rep['operator_requests'])})", ""]
    for q in rep["operator_requests"]:
        p = q["provenance"]
        L += [f"### {q['request_id']} — {q['kind']}, status {q['status']} (requested {q['requested_at_utc']}, "
              f"decision {p.get('decision_id', '')})", "", "Title:", _q(q["title"]), "", "Description:",
              _q(q["description"]), "", "Justification:", _q(q["justification"]), ""]
        if q.get("resolution_note"):
            L += [f"Operator's answer ({q.get('resolved_by', '')}):", _q(q["resolution_note"]), ""]
    L += [f"## Actions ({len(rep['actions'])})", "", "| time (UTC) | decision | operation | source | outcome | detail |",
          "|---|---|---|---|---|---|"]
    for a in rep["actions"]:
        p = a["provenance"]
        detail = str(a["failure_detail"] or "").replace("|", "\\|").replace("\n", " ")
        L.append(f"| {p.get('time_utc', '')} | {p.get('decision_id', '')} | {a['operation']} | {a['candidate_source']} | "
                 f"{a['group']} ({a['result_kind']}:{a['status']}) | {detail} |")
    L += ["", "Arguments of every action are in the JSON form of this report (`--json`).", "",
          f"## Approvals ({len(rep['approvals'])})", ""]
    for a in rep["approvals"]:
        L.append(f"* {a['approval_id']}: {a.get('status')} — {a.get('operation')} (decision {a.get('decision_id')})")
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
        L.append(f"| {p['provenance']['time_utc']} | {p['provenance']['decision_id']} | {p['metric']} | {p['kind']} | "
                 f"{d.get('mean')} | {d.get('std')} | {p['for_chosen_action']} | {rs} |")
    L += ["", f"## Experiments ({len(rep['experiments'])} registry records)", ""]
    for e in rep["experiments"]:
        d = e["data"]
        L.append(f"* {e['time_utc']} {e['type']} {d.get('experiment_id', '')}: status {d.get('status', '')}; "
                 f"question: {json.dumps(d.get('question', ''))}; hypothesis: {json.dumps(d.get('hypothesis', ''))}; "
                 f"arms: {json.dumps(d.get('arms', []))}; supported: {d.get('supported')}")
    lr = rep["learning"]
    L += ["", "## Learning updates", "", f"* updates: {lr['n_updates']} {json.dumps(lr['by_timescale'], sort_keys=True)}",
          f"* with a measured behavioural change: {lr['with_behavioural_change']}",
          f"* meta-updates: {json.dumps(lr['meta_updates'], sort_keys=True)}", "",
          "| timescale:group | updates | non-zero | sum of delta norms | largest |", "|---|---|---|---|---|"]
    for k, s in sorted(lr["weight_delta_norms"].items()):
        L.append(f"| {k} | {s['n']} | {s['nonzero']} | {s['sum']:.6g} | {s['max']:.6g} |")
    return "\n".join(L) + "\n"


def write_report(run_dir: str, out: str = "", as_json: bool = False) -> str:
    rep = build_report(run_dir)
    out = out or os.path.join(os.path.abspath(run_dir), "plan_report." + ("json" if as_json else "md"))
    body = json.dumps(rep, indent=1, sort_keys=True, default=str) if as_json or out.endswith(".json") else \
        render_markdown(rep)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(body)
    os.replace(tmp, out)
    return out
