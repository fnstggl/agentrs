"""cam_live.runtime.journal — the lifetime research journal (spec 14.4), regenerated from the structured records.

The journal is a *derived* document: every line is computed from records in the run directory and carries a tag.

    [FACT]        something a record states; the record ids follow in parentheses
    [HYPOTHESIS]  an inference drawn from the cited records; it may be wrong

Nothing here is written by a language model, and nothing here is an input to learning. The journal states that
learning happened only where a ``LearningUpdateRecord`` with a measured behavioural change (a non-zero change of
probe predictions or action probabilities) supports it.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..contracts import RunFiles
from .logs import RunReader, atomic_write_text

SHIFT_THRESHOLD = 0.25           # total-variation distance between the operation mixes of the two halves of a run
MIN_FOR_TREND = 8                # resolved predictions needed before an early/late comparison is reported


def _ids(rids: Iterable[str], limit: int = 6) -> str:
    rids = [r for r in rids if r]
    if not rids:
        return ""
    shown = ", ".join(rids[:limit])
    return f"{shown}, … +{len(rids) - limit} more" if len(rids) > limit else shown


def fact(text: str, rids: Iterable[str] = (), source: str = "") -> str:
    ref = _ids(rids) or source
    return f"- [FACT] {text} (records: {ref})"


def hypothesis(text: str, rids: Iterable[str] = (), source: str = "") -> str:
    ref = _ids(rids) or source
    return f"- [HYPOTHESIS] {text} (basis: {ref})"


def behavioural_change(update: Dict[str, Any]) -> bool:
    """True when the update record itself measured a change of predictions or action probabilities."""
    try:
        return any(abs(float(v)) > 1e-12 for v in (update.get("prediction_change") or {}).values())
    except (TypeError, ValueError):
        return False


def _tv(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()) or 1, sum(b.values()) or 1
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


def _mix(c: Counter) -> str:
    n = sum(c.values()) or 1
    return ", ".join(f"{k} {v}/{n}" for k, v in c.most_common(6))


def prediction_stats(resolutions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Accuracy of prospectively registered predictions against their resolved outcomes."""
    errs: List[Tuple[int, float, Optional[float], str]] = []
    for r in resolutions:
        d = r.get("data", {})
        v, m, s = (d.get("outcome") or {}).get("value"), d.get("predicted_mean"), d.get("predicted_std")
        if v is None or m is None:
            continue
        z = abs(float(v) - float(m)) / float(s) if s else None
        errs.append((r["seq"], abs(float(v) - float(m)), z, r["rid"]))
    errs.sort()
    out: Dict[str, Any] = {"n": len(errs), "rids": [e[3] for e in errs]}
    if errs:
        out["mean_abs_error"] = sum(e[1] for e in errs) / len(errs)
        zs = [e[2] for e in errs if e[2] is not None]
        if zs:
            out["within_1_std"] = sum(1 for z in zs if z <= 1.0) / len(zs)
            out["within_2_std"] = sum(1 for z in zs if z <= 2.0) / len(zs)
            out["n_with_std"] = len(zs)
        if len(errs) >= MIN_FOR_TREND:
            h = len(errs) // 2
            out["early_mae"] = sum(e[1] for e in errs[:h]) / h
            out["late_mae"] = sum(e[1] for e in errs[h:]) / (len(errs) - h)
            out["early_rids"], out["late_rids"] = [e[3] for e in errs[:h]], [e[3] for e in errs[h:]]
            out["split_seq"] = errs[h][0]
    return out


def generate_journal(run_dir: str, state: Optional[Dict[str, Any]] = None) -> str:
    r = RunReader(run_dir)
    st = state or r.state()
    events = r.records(RunFiles.EVENTS, types=["run_started", "run_resumed", "objective_assigned", "objective_finished",
                                               "verified_outcome", "verified_outcome_refused", "action_result",
                                               "outcome", "paused", "halted", "operator_command", "checkpoint",
                                               "inflight_recovered"])
    decisions = r.records(RunFiles.DECISIONS)
    experiments = r.records(RunFiles.EXPERIMENTS)
    resolutions = r.records(RunFiles.PREDICTIONS, types=["resolution"])
    learning = r.records(RunFiles.LEARNING)
    meta = r.records(RunFiles.META)
    obstacles = r.obstacles()
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_type[e["type"]].append(e)
    task = st.get("task", {})
    L: List[str] = []
    add = L.append

    add(f"# CAM Live research journal — run {st.get('run_id', os.path.basename(os.path.normpath(run_dir)))}")
    add("")
    add("Derived from the structured records of this run. `[FACT]` lines restate records (ids in parentheses); "
        "`[HYPOTHESIS]` lines are inferences from the cited records and may be wrong. Record ids refer to "
        "`events.jsonl` (ev-), `decisions.jsonl` (dec-), `experiments.jsonl` (exp-), `predictions.jsonl` (pred-), "
        "`learning_updates.jsonl` (lu-), `meta_updates.jsonl` (mu-) and `obstacles.jsonl` (ob-).")
    add("")

    # ------------------------------------------------------------------------------------------ 1 objective
    add("## 1. What CAM is currently trying to accomplish")
    start = (by_type["objective_assigned"] or by_type["run_started"])[-1:] if (by_type["objective_assigned"]
                                                                                 or by_type["run_started"]) else []
    objective = task.get("objective", "")
    add(fact(f"Objective `{task.get('objective_id', '')}` (mode {task.get('mode', '')}, data provenance "
             f"{task.get('mode', '')}): " + (f"“{objective}”" if objective else "none assigned (autonomous learning "
                                                                              "inside a fixed budget)"),
             [e["rid"] for e in start], "state.json"))
    add(fact(f"Run status: {st.get('status', 'unknown')}" + (f" — {st['status_reason']}" if st.get("status_reason") else ""),
             [e["rid"] for e in (by_type["objective_finished"] + by_type["paused"] + by_type["halted"])[-1:]],
             "state.json"))
    ver = by_type["verified_outcome"]
    if ver:
        o = ver[-1]["data"]["outcome"]
        excl = ", ".join(f"{k} {v}" for k, v in (o.get("excluded") or {}).items()) or "none"
        add(fact(f"Latest independently verified result: status {o.get('status')}, value {o.get('value'):g} against "
                 f"target {o.get('target'):g}, uncertainty {o.get('uncertainty'):g}; excluded: {excl}. "
                 f"{len(ver)} signed verifications so far", [ver[-1]["rid"]]))
        for q in o.get("secondary") or []:
            add(fact(f"Secondary measurement (kept apart from the objective): {q.get('name')} = {q.get('value'):g} "
                     f"{q.get('unit', '')}".rstrip(), [ver[-1]["rid"]]))
    else:
        add(fact("No signed verification of the objective has been recorded", source="events.jsonl"))
    lim = st.get("limits") or {}
    if lim:
        parts = [f"{k} {v['used']:.6g}" + (f" of {v['limit']:.6g}" if v["limit"] > 0 else " (no limit set)")
                 for k, v in lim.items()]
        add(fact("Resources used: " + "; ".join(parts), source="state.json"))

    # ------------------------------------------------------------------------------------------ 2 attempts
    add("")
    add("## 2. What it has attempted")
    ops = Counter(d["data"]["decision"]["chosen"]["operation"] for d in decisions)
    if decisions:
        add(fact(f"{len(decisions)} decisions recorded; chosen operations: {_mix(ops)}",
                 [decisions[0]["rid"], decisions[-1]["rid"]] if len(decisions) > 1 else [decisions[0]["rid"]]))
        for op, n in ops.most_common(8):
            rids = [d["rid"] for d in decisions if d["data"]["decision"]["chosen"]["operation"] == op]
            add(fact(f"`{op}` chosen {n} time(s)", rids))
    else:
        add(fact("No decision has been recorded", source="decisions.jsonl"))
    reg = [e for e in experiments if e["type"] == "experiment"]
    exp_latest: Dict[str, Dict[str, Any]] = {}
    for e in reg:
        exp_latest[e["data"].get("experiment_id", "")] = e
    if exp_latest:
        add(fact(f"{len(exp_latest)} experiments registered with a hypothesis and a prediction before their outcome",
                 [e["rid"] for e in exp_latest.values()]))

    # ------------------------------------------------------------------------------------------ 3 succeeded
    add("")
    add("## 3. What succeeded")
    results = by_type["action_result"]
    ok = [e for e in results if e["data"].get("status") == "ok" and e["data"].get("kind") == "execute"]
    if ok:
        add(fact(f"{len(ok)} executed action(s) finished with status ok: "
                 f"{_mix(Counter(e['data'].get('operation', '') for e in ok))}", [e["rid"] for e in ok]))
    else:
        add(fact("No executed action has finished with status ok", source="events.jsonl"))
    op_of_action = {e["data"].get("action_id"): e["data"].get("operation", "") for e in results}
    outs = [e for e in by_type["outcome"] if e["data"]["observation"].get("value") is not None]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in outs:
        grouped[op_of_action.get(e["data"].get("action_id"), "unattributed")].append(e)
    for op, es in sorted(grouped.items()):
        vals = [float(e["data"]["observation"]["value"]) for e in es]
        nv = sum(1 for e in es if e["data"]["observation"].get("verified"))
        add(fact(f"{len(es)} delayed outcome(s) matured for `{op}` actions: total {sum(vals):g}, mean "
                 f"{sum(vals) / len(vals):.3g}, range {min(vals):g}–{max(vals):g} ({nv} measured by the verifier)",
                 [e["rid"] for e in es]))
    sup = [e for e in exp_latest.values() if e["data"].get("supported") is True]
    if sup:
        add(fact(f"{len(sup)} experiment(s) ended with the registered hypothesis supported", [e["rid"] for e in sup]))
    if ver and ver[-1]["data"]["outcome"].get("status") == "achieved":
        add(fact("The objective was verified as achieved", [ver[-1]["rid"]]))

    # ------------------------------------------------------------------------------------------ 4 failed
    add("")
    add("## 4. What failed")
    bad = [e for e in results if e["data"].get("status") not in ("ok", "needs_approval")]
    if bad:
        kinds = Counter(f"{e['data'].get('status')}/{e['data'].get('failure_class')}" for e in bad)
        add(fact(f"{len(bad)} action(s) did not complete: {_mix(kinds)}", [e["rid"] for e in bad]))
    else:
        add(fact("No failed, refused or timed-out action has been recorded", source="events.jsonl"))
    if obstacles:
        cats = Counter(o["data"]["obstacle"].get("category") or "unclassified" for o in obstacles)
        add(fact(f"{len(obstacles)} obstacle record(s): {_mix(cats)}", [o["rid"] for o in obstacles]))
        for o in obstacles[-5:]:
            ob = o["data"]["obstacle"]
            add(fact(f"Obstacle on action {ob.get('action_id')}: {ob.get('description')}"
                     + (f"; next: {ob.get('outcome')}" if ob.get("outcome") else ""), [o["rid"]]))
            for ex in ob.get("explanations") or []:
                add(hypothesis(f"Possible explanation (p={ex.get('p')}): {ex.get('hypothesis')}", [o["rid"]]))
    unsup = [e for e in exp_latest.values() if e["data"].get("supported") is False]
    if unsup:
        add(fact(f"{len(unsup)} experiment(s) ended with the registered hypothesis not supported",
                 [e["rid"] for e in unsup]))
    if by_type["verified_outcome_refused"]:
        add(fact(f"{len(by_type['verified_outcome_refused'])} claimed objective outcome(s) were refused because the "
                 "verifier signature did not check; they were not used", [e["rid"] for e in by_type["verified_outcome_refused"]]))
    if ver and ver[-1]["data"]["outcome"].get("status") == "failed":
        add(fact("The objective was verified as failed", [ver[-1]["rid"]]))

    # ------------------------------------------------------------------------------------------ 5 uncertain
    add("")
    add("## 5. What remains uncertain")
    live = st.get("live") or {}
    n_pending = len((live.get("pending") or {}).get("items", []))
    add(fact(f"{n_pending} outcome measurement(s) are still pending", source="state.json"))
    pend_apr = [a for a in (st.get("approvals") or {}).values() if a.get("status") == "pending"]
    if pend_apr:
        add(fact(f"{len(pend_apr)} action(s) await operator approval: "
                 + ", ".join(f"{a.get('approval_id')} ({a.get('operation')})" for a in pend_apr), source="state.json"))
    if ver:
        o = ver[-1]["data"]["outcome"]
        if o.get("uncertainty"):
            add(fact(f"The verifier reports uncertainty {o['uncertainty']:g} on the objective value"
                     + (f": {o['notes']}" if o.get("notes") else ""), [ver[-1]["rid"]]))
    ps = prediction_stats(resolutions)
    if ps["n"]:
        txt = f"{ps['n']} registered prediction(s) have been resolved; mean absolute error {ps['mean_abs_error']:.3g}"
        if "within_2_std" in ps:
            txt += (f"; {ps['within_1_std']:.0%} within one and {ps['within_2_std']:.0%} within two predictive "
                    f"standard deviations (n={ps['n_with_std']})")
        add(fact(txt, ps["rids"]))
        if ps.get("n_with_std", 0) >= 20 and ps["within_2_std"] < 0.80:
            add(hypothesis("The predictive distributions appear over-confident (two-standard-deviation coverage "
                           f"{ps['within_2_std']:.0%})", ps["rids"]))
    else:
        add(fact("No registered prediction has been resolved yet, so predictive accuracy is unknown",
                 source="predictions.jsonl"))
    open_obs = [o for o in obstacles if not o["data"]["obstacle"].get("outcome")]
    if open_obs:
        add(fact(f"{len(open_obs)} obstacle(s) have no recorded follow-up outcome yet", [o["rid"] for o in open_obs]))
    und = [e for e in exp_latest.values() if e["data"].get("supported") is None]
    if und:
        add(fact(f"{len(und)} experiment(s) have no verdict yet", [e["rid"] for e in und]))

    # ------------------------------------------------------------------------------------------ 6 change
    add("")
    add("## 6. How its approach changed")
    if len(decisions) >= 10:
        h = len(decisions) // 2
        a = Counter(d["data"]["decision"]["chosen"]["operation"] for d in decisions[:h])
        b = Counter(d["data"]["decision"]["chosen"]["operation"] for d in decisions[h:])
        tv = _tv(a, b)
        span = [decisions[0]["rid"], decisions[h - 1]["rid"], decisions[h]["rid"], decisions[-1]["rid"]]
        add(fact(f"Chosen operations, first half of the decisions: {_mix(a)}; second half: {_mix(b)} "
                 f"(total-variation distance {tv:.2f})", span))
        if tv >= SHIFT_THRESHOLD:
            add(hypothesis(f"The mix of chosen operations shifted between the two halves (distance {tv:.2f} ≥ "
                           f"{SHIFT_THRESHOLD}); the records do not say why", span))
    else:
        add(fact(f"Only {len(decisions)} decision(s) so far; no comparison over time is reported",
                 source="decisions.jsonl"))
    changed = [o for o in obstacles if o["data"]["obstacle"].get("strategy_changed")]
    if changed:
        add(fact(f"After {len(changed)} obstacle(s) the next chosen action used a different operation",
                 [o["rid"] for o in changed]))
    if by_type["objective_assigned"]:
        add(fact(f"{len(by_type['objective_assigned'])} subsequent objective(s) were assigned by the operator",
                 [e["rid"] for e in by_type["objective_assigned"]]))

    # ------------------------------------------------------------------------------------------ 7 capabilities
    add("")
    add("## 7. Capabilities that appear to have been acquired, and the evidence")
    behav = [u for u in learning if behavioural_change(u["data"])]
    if not behav:
        add(fact(f"No learning update with a measured behavioural change has been recorded ({len(learning)} update "
                 "record(s) examined). This journal therefore makes no claim that learning occurred",
                 [u["rid"] for u in learning], "learning_updates.jsonl"))
    else:
        groups = Counter(g for u in behav for g in u["data"].get("param_groups") or [])
        add(fact(f"{len(behav)} of {len(learning)} learning update(s) measurably changed probe predictions or action "
                 f"probabilities; parameter groups updated: {_mix(groups)}", [u["rid"] for u in behav]))
        last = behav[-1]["data"]
        add(fact(f"Latest such update ({last.get('timescale')}; trigger: {last.get('trigger')}): prediction change "
                 f"{last.get('prediction_change')}, weight change {last.get('weight_delta')}, losses {last.get('losses')}",
                 [behav[-1]["rid"]]))
        if "early_mae" in ps:
            add(fact(f"Mean absolute error of resolved predictions: {ps['early_mae']:.3g} over the earlier half, "
                     f"{ps['late_mae']:.3g} over the later half", ps["early_rids"][:3] + ps["late_rids"][:3]))
            between = [u for u in behav if u["seq"] <= max(r_["seq"] for r_ in resolutions)]
            if ps["late_mae"] < ps["early_mae"] and between:
                add(hypothesis("Prediction of the registered outcome quantities appears to have improved after the "
                               "recorded weight updates; the comparison is not controlled, so other causes are "
                               "possible", [u["rid"] for u in between][:4] + ps["late_rids"][:2]))
    applied = [m for m in meta if m["data"].get("applied")]
    skipped = [m for m in meta if not m["data"].get("applied")]
    if meta:
        reasons = Counter(m["data"].get("skipped_reason") or "unspecified" for m in skipped)
        add(fact(f"Meta-learning of the reward function: {len(applied)} update(s) applied, {len(skipped)} skipped"
                 + (f" ({_mix(reasons)})" if skipped else "")
                 + f"; latest bootstrap mix {meta[-1]['data'].get('bootstrap_mix')}", [m["rid"] for m in meta]))
    else:
        add(fact("No meta-learning update has been recorded", source="meta_updates.jsonl"))

    # ------------------------------------------------------------------------------------------ 8 index
    add("")
    add("## 8. Record index")
    ck = (st.get("committed") or {}).get("ckpt") or {}
    add(fact(f"Current learner checkpoint: {ck.get('id', 'none')} (generation {ck.get('generation', 0)})",
             [e["rid"] for e in by_type["checkpoint"][-1:]], "state.json"))
    if by_type["run_resumed"]:
        add(fact(f"The process was restarted {len(by_type['run_resumed'])} time(s) and resumed from persisted state",
                 [e["rid"] for e in by_type["run_resumed"]]))
    add(fact(f"Record counts — decisions {len(decisions)}, experiments {len(exp_latest)}, resolved predictions "
             f"{ps['n']}, learning updates {len(learning)}, meta updates {len(meta)}, obstacles {len(obstacles)}",
             source="run directory"))
    add("")
    return "\n".join(L)


def write_journal(run_dir: str, state: Optional[Dict[str, Any]] = None) -> str:
    path = os.path.join(run_dir, RunFiles.JOURNAL)
    atomic_write_text(path, generate_journal(run_dir, state))
    return path
