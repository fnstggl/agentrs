"""Compact evidence summary of a Stage-1 run directory (everything recomputed from the run's own files and its
archive; no number is copied from a log message).

Sections:
  decisions   consequential CAM decisions by source, executed ok vs each error code, tool histogram, decoded tokens
              per decision (chosen action / everything the decision generated incl. imagined candidates), wall time
              per decision, schema violations of the generator (by construction 0).
  updates     neural updates: per-objective losses and backbone gradient norms, global / group gradient norms, the
              UnifiedCore parameter hash before / after and its L2 change per component; training provenance checked in
              the archive (every replayed and fresh event id of an update resolvable, and its update id present in
              EventRecord.training_provenance["used_in_updates"]).
  versions    model_version of archived decision records over the lifetime, policy probes at fixed contexts per version.
  resume      every new-process resume: restored version / archive count / optimiser steps / replay cursor, the
              abandoned branch and the comparison of resumed decisions with the interrupted process's decisions.
  traces      one DecisionTrace per decision with the spec 18.1 fields filled, update refs linked.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import statistics
from collections import Counter
from typing import Optional


def _rows(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _stats(xs: list) -> Optional[dict]:
    xs = [float(x) for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not xs:
        return None
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]   # noqa: E731
    return {"n": len(s), "mean": sum(s) / len(s), "median": statistics.median(s), "p10": q(0.1), "p90": q(0.9),
            "min": s[0], "max": s[-1], "sum": sum(s)}


def decisions_section(run_dir: str) -> dict:
    rows = _rows(os.path.join(run_dir, "decisions.jsonl"))
    src = Counter(r["source"] for r in rows)
    cam = [r for r in rows if r["source"] in ("cam_decoder", "cam_planner")]
    ok = sum(1 for r in cam if r["ok"])
    errs = Counter(str(r["error"]) for r in cam if not r["ok"])
    tools = Counter(r["tool"] for r in cam)
    return {"n_decisions": len(rows), "by_source": dict(src), "consequential_cam": len(cam),
            "scripted_or_other": len(rows) - len(cam), "executed_ok": ok, "rejected": len(cam) - ok,
            "schema_valid": sum(1 for r in cam if r.get("schema_valid")),
            "schema_invalid": sum(1 for r in cam if r.get("schema_valid") is False),
            "schema_unchecked": sum(1 for r in cam if r.get("schema_valid") is None),
            "error_codes": dict(errs.most_common()), "distinct_tools": len(tools),
            "tool_histogram": dict(tools.most_common()),
            "decoded_tokens_chosen_action": _stats([r.get("n_tokens_chosen") for r in cam]),
            "decoded_tokens_per_decision_total": _stats([r.get("decoded_tokens_total") for r in cam]),
            "qwen_decoder_steps_per_decision": _stats([r.get("qwen_steps") for r in cam]),
            "actions_generated_per_decision": _stats([r.get("actions_generated") for r in cam]),
            "encoded_tokens_per_decision": _stats([r.get("encoded_tokens") for r in cam]),
            "act_wall_s": _stats([r.get("act_s") for r in cam]),
            "observe_wall_s": _stats([r.get("observe_s") for r in cam]),
            "peak_gpu_mb_act": _stats([r.get("peak_gpu_mb_act") for r in cam]),
            "tools_available": _stats([r.get("n_tools") for r in cam]),
            "reward_sum": sum(float(r.get("reward") or 0.0) for r in cam),
            "nonzero_rewards": sum(1 for r in cam if r.get("reward")),
            "sim_days": (cam[-1]["t_sim"] - cam[0]["t_sim"]) if len(cam) > 1 else 0.0,
            "model_versions_seen": sorted({r["model_version"] for r in cam})}


def updates_section(run_dir: str, archive=None, check_events: int = 32) -> dict:
    ups = _rows(os.path.join(run_dir, "learner", "updates.jsonl"))
    out = {"n_updates": len(ups), "versions": [u["version"] for u in ups]}
    per = []
    for u in ups:
        pd = u.get("param_delta") or {}
        per.append({"update_id": u["update_id"], "version": u["version"], "losses": u.get("losses"),
                    "backbone_grad_norm": u.get("backbone_grad_norm"), "backbone_share": u.get("backbone_share"),
                    "grad_norm": u.get("grad_norm"), "group_grad_norm": u.get("group_grad_norm"),
                    "clipped": u.get("clipped"), "core_hash_before": (u.get("core_hash_before") or "")[:16],
                    "core_hash_after": (u.get("core_hash_after") or "")[:16],
                    "core_l2_delta": (pd.get("core") or {}).get("l2_delta"),
                    "core_rel_delta": (pd.get("core") or {}).get("rel_delta"),
                    "components_changed": sorted(c for c, v in pd.items() if v.get("tensors_changed")),
                    "components_unchanged": sorted(c for c, v in pd.items() if not v.get("tensors_changed")),
                    "replay_events": u.get("replay_events"), "consumed_events": u.get("consumed_events"),
                    "update_s": u.get("update_s"), "peak_gpu_mb_update": u.get("peak_gpu_mb_update"),
                    "policy_n_awr": (u.get("policy") or {}).get("n_awr"),
                    "policy_n_ppo": (u.get("policy") or {}).get("n_ppo"),
                    "logp_maxdev_lag0": (u.get("policy") or {}).get("logp_maxdev_lag0")})
    out["per_update"] = per
    out["core_hash_changed_every_update"] = all(p["core_hash_before"] != p["core_hash_after"] for p in per) \
        if per else False
    out["core_hash_chain_consistent"] = all(per[i]["core_hash_after"] == per[i + 1]["core_hash_before"]
                                            for i in range(len(per) - 1)) if len(per) > 1 else None
    out["update_s"] = _stats([u.get("update_s") for u in ups])
    out["loss_keys"] = sorted({k for u in ups for k in (u.get("losses") or {})})
    if archive is not None and ups:
        checked, resolvable, linked = 0, 0, 0
        for u in ups:
            ids = list(dict.fromkeys((u.get("replay_event_ids") or []) + (u.get("consumed_event_ids") or [])))
            ids = ids[:check_events]
            if not ids:
                continue
            try:
                recs = archive.get(ids)
            except KeyError:
                recs = []
                for e in ids:
                    try:
                        recs += archive.get([e])
                    except KeyError:
                        pass
            checked += len(ids)
            resolvable += len(recs)
            linked += sum(1 for r in recs if u["update_id"] in (r.training_provenance or {}).get("used_in_updates", []))
        out["provenance_check"] = {"event_ids_checked": checked, "resolvable_in_archive": resolvable,
                                   "with_update_id_in_used_in_updates": linked}
    return out


def versions_section(run_dir: str, archive=None) -> dict:
    out = {}
    if archive is not None:
        seq = []
        for recs in archive.scan(batch_size=512):
            seq.extend((r.step, r.model_version) for r in recs if r.action is not None)
        seq.sort()
        out["decision_model_versions"] = [v for _, v in seq]
        out["monotone_non_decreasing"] = all(b >= a for (_, a), (_, b) in zip(seq, seq[1:]))
        out["distinct_versions"] = len({v for _, v in seq})
    probes = _rows(os.path.join(run_dir, "probes", "probes.jsonl"))
    rows = []
    for p in probes:
        rows.append({"when": p["when"], "version": p["version"],
                     "logp_total": [r["logp_total"] for r in p["rows"]],
                     "logp_tool": [r["logp"].get("tool") for r in p["rows"]],
                     "entropy_tool": [r["entropy"].get("tool") for r in p["rows"]],
                     "value": [r["value"] for r in p["rows"]]})
    out["probes"] = rows
    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        out["probe_change_first_to_last"] = {
            "versions": [a["version"], b["version"]],
            "max_abs_dlogp_total": max(abs(x - y) for x, y in zip(a["logp_total"], b["logp_total"])),
            "max_abs_dlogp_tool": max(abs((x or 0) - (y or 0)) for x, y in zip(a["logp_tool"], b["logp_tool"])),
            "max_abs_dentropy_tool": max(abs((x or 0) - (y or 0)) for x, y in zip(a["entropy_tool"],
                                                                                    b["entropy_tool"])),
            "max_abs_dvalue": max(abs(x - y) for x, y in zip(a["value"], b["value"]))}
        out["probe_changed_at_every_update"] = all(
            any(abs(x - y) > 1e-6 for x, y in zip(rows[i]["logp_total"], rows[i + 1]["logp_total"]))
            for i in range(len(rows) - 1))
    return out


def _same_decision(a: dict, b: dict, tol: float) -> dict:
    lp = (a.get("behavior_logp"), b.get("behavior_logp"))
    dl = abs(lp[0] - lp[1]) if None not in lp else None
    return {"step": a.get("step"), "tool_equal": a.get("tool") == b.get("tool"),
            "args_equal": a.get("args") == b.get("args"), "behavior_logp_abs_diff": dl,
            "equal": a.get("tool") == b.get("tool") and a.get("args") == b.get("args")
            and (dl is None or dl <= tol)}


def resume_section(run_dir: str, tol: float = 1e-3) -> dict:
    res = _rows(os.path.join(run_dir, "resume.jsonl"))
    dec = {r["step"]: r for r in _rows(os.path.join(run_dir, "decisions.jsonl"))}
    out = []
    for r in res:
        cmp = []
        for ab in r.get("abandoned_decision_rows") or []:
            cur = dec.get(ab.get("step"))
            if cur is not None:
                cmp.append(_same_decision(ab, cur, tol))
        first_div = next((c["step"] for c in cmp if not c["equal"]), None)
        out.append({k: r.get(k) for k in ("checkpoint", "pid", "saved_pid", "version", "archive_count_checkpoint",
                                          "archive_count_now", "abandoned_events", "abandoned_decisions",
                                          "abandoned_traces_marked", "optimizer_steps", "trainer_n_updates",
                                          "replay_cursor", "replay_counts", "consolidator_cursor", "memory",
                                          "n_decisions", "steps", "load_s")}
                   | {"interrupted_vs_resumed": cmp, "first_divergent_step": first_div,
                      "identical_prefix": sum(1 for _ in itertools.takewhile(lambda c: c["equal"], cmp))})
    chunks = _rows(os.path.join(run_dir, "chunks.jsonl"))
    return {"resumes": out, "chunks": [{k: c.get(k) for k in ("pid", "resumed", "stopped_chunk", "calls", "wall_s",
                                                              "decisions_total", "version", "archive_count",
                                                              "peak_gpu_mb", "error")} for c in chunks]}


TRACE_FIELDS = ("context_refs", "candidates", "predicted", "uncertainty", "imagined", "selected", "reasoning",
                "belief_before", "belief_after", "outcome", "update_refs", "objective")


def traces_section(run_dir: str) -> dict:
    from ..observe.trace import load_traces
    root = os.path.join(run_dir, "traces")
    if not os.path.isdir(root):
        return {"n_traces": 0}
    ts = load_traces(root)
    live = [t for t in ts if not (t.outcome or {}).get("abandoned_branch")]
    filled = {f: sum(1 for t in live if getattr(t, f)) for f in TRACE_FIELDS}
    return {"n_traces": len(ts), "n_abandoned_branch": len(ts) - len(live), "n_live": len(live),
            "fields_filled": filled, "policy_versions": sorted({t.policy_version for t in live}),
            "with_update_refs": sum(1 for t in live if t.update_refs),
            "imagined_per_trace": _stats([len(t.imagined) for t in live]),
            "candidates_per_trace": _stats([len(t.candidates) for t in live]),
            "reasoning_iters": _stats([(t.reasoning or {}).get("iters") for t in live]),
            "sources": dict(Counter(t.source for t in live))}


def summarize(run_dir: str) -> dict:
    run_dir = os.path.abspath(run_dir)
    meta = json.load(open(os.path.join(run_dir, "run.json"))) if os.path.exists(os.path.join(run_dir,
                                                                                           "run.json")) else {}
    archive = None
    try:
        from ..memory.archive import EventArchive
        archive = EventArchive(os.path.join(run_dir, "archive"), readonly=True)
    except Exception as e:  # noqa: BLE001 - a summary of a broken run still reports what it can
        meta["archive_error"] = repr(e)
    sysd = meta.get("system", {})
    out = {"run_dir": run_dir, "command": (meta.get("spec") or {}).get("command"),
           "task": meta.get("task"), "semantic_model": sysd.get("semantic_model"),
           "scale": (sysd.get("cam_config") or {}).get("scale"),
           "core_shape": {k: (sysd.get("cam_config") or {}).get(k) for k in ("n_layers", "d_model", "n_heads",
                                                                              "n_state")},
           "max_arg_tokens": (sysd.get("cam_config") or {}).get("max_arg_tokens"),
           "imagination": sysd.get("imagination"), "planner": sysd.get("planner"),
           "update_every": (meta.get("spec") or {}).get("update_every"),
           "parameters": sysd.get("parameters"), "trainer": sysd.get("trainer"),
           "archive_count": archive.count() if archive is not None else None,
           "decisions": decisions_section(run_dir), "updates": updates_section(run_dir, archive),
           "versions": versions_section(run_dir, archive), "resume": resume_section(run_dir),
           "traces": traces_section(run_dir),
           "dashboard": os.path.join(run_dir, "dashboard", "index.html")
           if os.path.exists(os.path.join(run_dir, "dashboard", "index.html")) else None}
    if archive is not None:
        archive.close()
    return out
