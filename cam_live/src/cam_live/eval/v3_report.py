"""cam_live.eval.v3_report — prespecified summaries of the Protocol v3 substantive runs (docs/S/PROTOCOLS.md v3.4/3.5).

    CAM_LIVE_EVAL_HOME=... python -m cam_live.eval.v3_report --out-dir cam_live/docs/evidence/S/v3 \
        [--expect-commit SHA] [--markdown TABLES.md]
    python -m cam_live.eval.v3_report --results-dir DIR --out-dir D      (an archived results tree, e.g. the pre-fix
                                                                          batch: DIR/x0/E_s0.json ...)

Per experiment and condition: objectives achieved / attempted (never-reached and interrupted counted), decisions to
verified achievement (censored at the budget), how each objective ended (achieved / decision budget / wall budget of
the lock hold / interrupted), verified value, executable fraction, backbone-written fraction of the chosen actions
and of the candidates, repeats of failed actions, distinct strategies, prediction NLL, meta-updates (applied /
skipped with reasons, gradient norms, reward-version changes), slow (adapter) steps kept / reverted by the guard /
deferred / skipped (read from the run roots' learning logs), compute, wall time, configuration and snapshot per run
(checked against ``--expect-commit``); ranges across seeds; S_q for the transfer pairs; paired seed differences with
exact sign-flip p (Protocol v3.5: with 2-3 seeds the smallest attainable p is 0.25-0.5, so these are reported as
descriptive only).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from .stats import diff_ci, speedup

EXPERIMENTS = ("x0", "x1", "x3", "x4", "x5")


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Complete records of a JSONL file (a torn final line is skipped)."""
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "rb") as f:
            for raw in f:
                if not raw.endswith(b"\n"):
                    break
                raw = raw.strip()
                if raw:
                    try:
                        out.append(json.loads(raw))
                    except ValueError:
                        continue
    except OSError:
        pass
    return out


def _run_file(run_dir: str, name: str) -> str:
    meta = os.path.join(run_dir, "runtime_logs.json")
    if os.path.exists(meta):
        d = _load(meta) or {}
        return os.path.join(run_dir, d.get("log_dir", ""), name)
    return os.path.join(run_dir, name)


def slow_step_outcomes(obj_dir: str) -> Optional[Dict[str, Any]]:
    """What happened to every slow (adapter) step of one objective, from its learning log: kept by the guard,
    reverted by the guard, deferred (no held-out evidence to judge it), skipped (with reason), or kept without a
    guard verdict. None when the log is not there."""
    path = _run_file(obj_dir, "learning_updates.jsonl")
    if not os.path.exists(path):
        return None
    out: Dict[str, Any] = {"total": 0, "kept": 0, "reverted": 0, "deferred": 0, "kept_unguarded": 0,
                           "skipped": {}, "kept_adapter_delta": []}
    for e in read_jsonl(path):
        d = e.get("data", e)
        if d.get("timescale") != "slow_adapters":
            continue
        out["total"] += 1
        try:
            n = json.loads(d.get("notes") or "{}")
        except ValueError:
            n = {}
        g = n.get("guard")
        if n.get("deferred"):
            out["deferred"] += 1
        elif n.get("skipped"):
            why = str(n["skipped"])
            out["skipped"][why] = out["skipped"].get(why, 0) + 1
        elif isinstance(g, dict) and g.get("reverted"):
            out["reverted"] += 1
        elif isinstance(g, dict) and g.get("accepted"):
            out["kept"] += 1
            out["kept_adapter_delta"].append((d.get("weight_delta") or {}).get("semantic_adapters"))
        else:
            out["kept_unguarded"] += 1
    return out


def stop_class(o: Dict[str, Any]) -> str:
    """How an objective ended: achieved, decision budget, wall budget of the lock hold, step cap, interrupted
    (child killed at the lease limit) or the runtime's own status."""
    if o.get("interrupted"):
        return "interrupted"
    if o.get("achieved"):
        return "achieved"
    reason = str(o.get("reason") or "")
    if "wall budget" in reason:
        return "wall_budget"
    if "step cap" in reason:
        return "step_cap"
    if o.get("runtime_status") == "exhausted" or "max_decisions" in reason:
        return "decision_budget"
    return str(o.get("runtime_status") or "unknown")


def candidate_backbone_fraction(o: Dict[str, Any]) -> Optional[float]:
    rows = o.get("decisions_detail") or []
    n = sum(int(r.get("n_candidates") or 0) for r in rows)
    return (sum(int(r.get("n_backbone_candidates") or 0) for r in rows) / n) if n else None


def objective_rows(res: Dict[str, Any], budget: int, *, root: Optional[str] = None) -> List[Dict[str, Any]]:
    root = root if root is not None else res.get("root")
    out = []
    for o in res.get("objectives", []):
        n = o.get("decisions")
        ach = bool(o.get("achieved"))
        meta = o.get("meta_updates") or []
        idx = o.get("index")
        slow = slow_step_outcomes(os.path.join(root, f"obj{int(idx):02d}")) if root and idx is not None and \
            os.path.isdir(os.path.join(root, f"obj{int(idx):02d}")) and not o.get("interrupted") else None
        out.append({
            "index": idx, "family": o.get("family"), "problem": o.get("problem"),
            "achieved": ach, "interrupted": bool(o.get("interrupted")), "verified_value": o.get("verified_value"),
            "verified_status": o.get("verified_status"), "stop": stop_class(o),
            "runtime_reason": str(o.get("reason") or "")[:160],
            "decisions": n, "n_to_achieve": float(n) if ach and n is not None else float(budget),
            "censored": not ach, "executable_fraction": o.get("executable_fraction"),
            "backbone_chosen_fraction": o.get("backbone_chosen_fraction"),
            "backbone_candidate_fraction": candidate_backbone_fraction(o),
            "repeats_of_failed_actions": o.get("repeats_of_failed_actions"),
            "distinct_strategies": o.get("distinct_strategies"), "mean_prediction_nll": o.get("mean_prediction_nll"),
            "n_predictions_resolved": o.get("n_predictions_resolved"),
            "primitive_ops_total": o.get("primitive_ops_total"), "skill_runs": o.get("skill_runs"),
            "skill_runs_ok": o.get("skill_runs_ok"),
            "wall_s": o.get("wall_s"),
            "inference_s": ((o.get("compute") or {}).get("seconds") or {}).get("inference"),
            "training_s": ((o.get("compute") or {}).get("seconds") or {}).get("training"),
            "meta_records": len(meta),
            "meta_applied": sum(1 for m in meta if m.get("applied")),
            "meta_skipped": dict(Counter(str(m.get("skipped_reason", "")).split(":")[0] for m in meta
                                         if not m.get("applied"))),
            "meta_grad_norms": [m.get("meta_grad_norm") for m in meta if m.get("applied")],
            "meta_estimators": sorted({str(m.get("estimator")) for m in meta if m.get("applied")}),
            "reward_changed": bool((o.get("group_changed") or {}).get("reward")),
            "groups_changed": o.get("group_changed"), "mix": o.get("mix"),
            "learning_by_timescale": dict(Counter(str(x.get("timescale")) for x in (o.get("learning_updates") or []))),
            "slow_steps": slow,
            "unavailable_tools": [u.get("tool") for u in (o.get("unavailable_permitted_tools") or [])],
            "lock_hold_s": o.get("lock_hold_s"), "gpu_peak_gb": o.get("gpu_peak_gb"),
            "errors": len(o.get("errors") or []),
            "eval_code": o.get("eval_code")})
    return out


def _sum_slow(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tot: Dict[str, Any] = {"objectives_with_log": 0, "total": 0, "kept": 0, "reverted": 0, "deferred": 0,
                           "kept_unguarded": 0, "skipped": {}}
    for x in rows:
        s = x.get("slow_steps")
        if not s:
            continue
        tot["objectives_with_log"] += 1
        for k in ("total", "kept", "reverted", "deferred", "kept_unguarded"):
            tot[k] += int(s.get(k) or 0)
        for k, v in (s.get("skipped") or {}).items():
            tot["skipped"][k] = tot["skipped"].get(k, 0) + v
    return tot


def summarise_condition(runs: Dict[int, Dict[str, Any]], budget: int, *,
                        expect_commit: str = "") -> Dict[str, Any]:
    per_seed = {s: objective_rows(r, budget) for s, r in sorted(runs.items())}
    allrows = [x for rows in per_seed.values() for x in rows]
    att = [x for x in allrows if not x["interrupted"]]

    def seedwise(key: str, f=lambda rows, k: sum(r[k] for r in rows if r[k] is not None) /
                 max(1, sum(1 for r in rows if r[k] is not None))) -> Dict[str, Any]:
        vals = [f(rows, key) for rows in per_seed.values() if rows and any(r[key] is not None for r in rows)]
        return {"per_seed": vals, "mean": sum(vals) / len(vals) if vals else None,
                "min": min(vals) if vals else None, "max": max(vals) if vals else None}
    commits = {s: (r.get("code") or {}).get("snapshot_commit") for s, r in runs.items()}
    plans = {s: r.get("plan") for s, r in runs.items()}
    n_plan = {s: len(str(r.get("plan") or "").split(",")) if r.get("plan") else None for s, r in runs.items()}
    return {
        "seeds": sorted(runs), "objectives_planned": sum(v or 0 for v in n_plan.values()),
        "objectives_attempted": len(att), "objectives_achieved": sum(x["achieved"] for x in att),
        "objectives_never_reached": sum(1 for x in att if not x["achieved"]),
        "objectives_interrupted": sum(1 for x in allrows if x["interrupted"]),
        "objectives_not_run": sum(v or 0 for v in n_plan.values()) - len(allrows),
        "stops": dict(Counter(x["stop"] for x in allrows)),
        "achieved_by_problem": {p: f"{sum(x['achieved'] for x in att if x['problem'] == p)}/"
                                   f"{sum(1 for x in att if x['problem'] == p)}"
                                for p in sorted({x["problem"] for x in att})},
        "decisions_to_achieve_censored": seedwise("n_to_achieve"),
        "decisions_used": seedwise("decisions"),
        "verified_value": seedwise("verified_value"),
        "executable_fraction": seedwise("executable_fraction"),
        "backbone_chosen_fraction": seedwise("backbone_chosen_fraction"),
        "backbone_candidate_fraction": seedwise("backbone_candidate_fraction"),
        "repeats_of_failed_actions": seedwise("repeats_of_failed_actions"),
        "distinct_strategies": seedwise("distinct_strategies"),
        "mean_prediction_nll": seedwise("mean_prediction_nll"),
        "meta_update_records": sum(x["meta_records"] for x in allrows),
        "meta_updates_applied_total": sum(x["meta_applied"] for x in allrows),
        "meta_skip_reasons": dict(sum((Counter(x["meta_skipped"]) for x in allrows), Counter())),
        "meta_grad_norms": [g for x in allrows for g in x["meta_grad_norms"]][:40],
        "objectives_with_reward_change": sum(x["reward_changed"] for x in allrows),
        "objectives_with_adapter_change": sum(bool((x["groups_changed"] or {}).get("semantic_adapters"))
                                              for x in allrows),
        "bootstrap_mix_values": sorted({x["mix"] for x in allrows if x["mix"] is not None}),
        "slow_steps": _sum_slow(allrows),
        "unavailable_tools_seen": sorted({t for x in allrows for t in x["unavailable_tools"]}),
        "max_lock_hold_s": max((x["lock_hold_s"] or 0.0 for x in allrows), default=None),
        "gpu_peak_gb_max": max((x["gpu_peak_gb"] or 0.0 for x in allrows), default=None),
        "wall_s_per_objective": seedwise("wall_s"),
        "learning_curve_by_index": {i: f"{sum(x['achieved'] for x in att if x['index'] == i)}/"
                                       f"{sum(1 for x in att if x['index'] == i)}"
                                    for i in sorted({x["index"] for x in att})},
        "code_check": {"expected": expect_commit or None, "snapshot_commit_per_seed": commits,
                       "objective_eval_codes": dict(Counter(str(x["eval_code"]) for x in allrows
                                                            if not x["interrupted"])),
                       "all_match": (all(c == expect_commit for c in commits.values()) and
                                     all(x["eval_code"] == expect_commit for x in allrows if not x["interrupted"])
                                     if expect_commit else None)},
        "plans": plans,
        "per_seed": per_seed,
        "configuration": {s: r.get("configuration") for s, r in runs.items()},
        "fresh_process": {s: r.get("fresh_process") for s, r in runs.items() if r.get("fresh_process")},
        "code": {s: r.get("code") for s, r in runs.items()}}


def collect(results_dir: str, exp: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """``results_dir/<exp>/<condition>_s<seed>.json`` -> {condition: {seed: result}}."""
    by: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(results_dir, exp, "*.json"))):
        r = _load(p)
        if not r or "objectives" not in r:
            continue
        name = os.path.basename(p)[:-5]
        cond, _, s = name.rpartition("_s")
        try:
            by[cond][int(s)] = r
        except ValueError:
            continue
    return dict(by)


def transfer(results_dir: str, budget: int, *, expect_commit: str = "") -> Dict[str, Any]:
    by = collect(results_dir, "x2")
    ini, exp_ = by.get("initial", {}), by.get("experienced", {})
    seeds = sorted(set(ini) & set(exp_))
    pairs_i, pairs_e, per = [], [], []
    for s in seeds:
        ri, re_ = objective_rows(ini[s], budget), objective_rows(exp_[s], budget)
        for a, b in zip(ri, re_):
            if a["interrupted"] or b["interrupted"]:
                continue
            pairs_i.append({"reached": a["achieved"], "n": a["n_to_achieve"]})
            pairs_e.append({"reached": b["achieved"], "n": b["n_to_achieve"]})
            per.append({"seed": s, "problem": a["problem"], "initial": a["n_to_achieve"], "initial_achieved":
                        a["achieved"], "experienced": b["n_to_achieve"], "experienced_achieved": b["achieved"],
                        "initial_value": a["verified_value"], "experienced_value": b["verified_value"],
                        "initial_stop": a["stop"], "experienced_stop": b["stop"]})
    by_problem = {}
    for p in sorted({x["problem"] for x in per}):
        xi = [{"reached": x["initial_achieved"], "n": x["initial"]} for x in per if x["problem"] == p]
        xe = [{"reached": x["experienced_achieved"], "n": x["experienced"]} for x in per if x["problem"] == p]
        by_problem[p] = speedup(xi, xe)
    out = {"seeds": seeds, "pairs": per,
           "S_q": speedup(pairs_i, pairs_e) if pairs_i else None,
           "S_q_by_problem": by_problem,
           "value_difference_experienced_minus_initial": diff_ci([p["experienced_value"] or 0.0 for p in per],
                                                                 [p["initial_value"] or 0.0 for p in per],
                                                                 paired=True) if per else None,
           "fresh_process_checks": {f"{v}_s{s}": (r.get("fresh_process") or {}) for v, d in
                                    (("initial", ini), ("experienced", exp_)) for s, r in d.items()},
           "initial": summarise_condition(ini, budget, expect_commit=expect_commit) if ini else None,
           "experienced": summarise_condition(exp_, budget, expect_commit=expect_commit) if exp_ else None,
           "note": "pairs are (seed, held-out objective); with this many pairs no test can reach p < 0.05 "
                   "(Protocol v3.5); S_q is reported with its bootstrap range only"}
    return out


def _mean_n(res: Dict[str, Any], budget: int) -> Optional[float]:
    rows = [x for x in objective_rows(res, budget) if not x["interrupted"]]
    return sum(x["n_to_achieve"] for x in rows) / len(rows) if rows else None


#: G3A-3 (independent verification, 2026-11-28): two fresh agents with the same seed are not the same experiment (the
#: agent id is drawn from os.urandom), so arms are independent runs; comparisons are unpaired by default
UNPAIRED_NOTE = ("unpaired: runs with the same seed are independent (agent id from os.urandom, G3A-3), so the "
                 "difference of the arm means is reported with a bootstrap interval and a label-permutation p over the "
                 "per-seed means (descriptive only, v3.5)")


def paired_vs(a: Dict[int, Dict[str, Any]], b: Dict[int, Dict[str, Any]], budget: int,
              keys: tuple = ("n_to_achieve",), *, paired: bool = False) -> Dict[str, Any]:
    """Differences a - b of per-seed means over the seeds both arms have (descriptive; v3.5). ``paired`` treats equal
    seeds as matched units (sign-flip test); the default does not (G3A-3: UNPAIRED_NOTE)."""
    seeds = sorted(set(a) & set(b))
    out: Dict[str, Any] = {"seeds": seeds, "paired": paired, "pairing_note": "" if paired else UNPAIRED_NOTE}
    for k in keys:
        fa, fb = [], []
        for s in seeds:
            ra = [x for x in objective_rows(a[s], budget) if not x["interrupted"] and x[k] is not None]
            rb = [x for x in objective_rows(b[s], budget) if not x["interrupted"] and x[k] is not None]
            fa.append(sum(float(x[k]) for x in ra) / len(ra) if ra else None)
            fb.append(sum(float(x[k]) for x in rb) / len(rb) if rb else None)
        out[k] = {"a_per_seed": fa, "b_per_seed": fb,
                  "a_mean": (sum(x for x in fa if x is not None) / max(1, sum(x is not None for x in fa))) if seeds else None,
                  "b_mean": (sum(x for x in fb if x is not None) / max(1, sum(x is not None for x in fb))) if seeds else None,
                  "a_minus_b": diff_ci(fa, fb, paired=paired) if seeds else None}
    return out


def build_report(results_dir: str, budget: int = 20, *, expect_commit: str = "",
                 baseline_dir: str = "") -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for exp in EXPERIMENTS:
        by = collect(results_dir, exp)
        if by:
            report[exp] = {c: summarise_condition(r, budget, expect_commit=expect_commit) for c, r in by.items()}
    e = collect(results_dir, "x1").get("E", {})
    x3 = collect(results_dir, "x3")
    if e and x3:
        report["x3_vs_E"] = {c: {"note": "E minus baseline (descriptive only, v3.5)",
                                 **paired_vs(e, runs, budget, ("n_to_achieve", "verified_value", "achieved"))}
                             for c, runs in x3.items()}
    x4 = collect(results_dir, "x4")
    if x4.get("skills") and x4.get("primitive"):
        report["x4_skills_vs_primitive"] = {
            "note": "skills minus primitive (descriptive only, v3.5)",
            **paired_vs(x4["skills"], x4["primitive"], budget,
                        ("n_to_achieve", "achieved", "primitive_ops_total", "wall_s", "inference_s"))}
    x5 = collect(results_dir, "x5")
    if e and x5.get("goalfree"):
        report["x5_vs_E"] = {"note": "goal-free phase first minus without (X1 E), the goal-free phase itself excluded "
                                     "(descriptive only, v3.5)",
                             **paired_vs(_drop_goalfree(x5["goalfree"]), e, budget,
                                         ("n_to_achieve", "verified_value", "achieved"))}
    if collect(results_dir, "x2"):
        report["x2"] = transfer(results_dir, budget, expect_commit=expect_commit)
    x1 = collect(results_dir, "x1")
    abl = {c: r for c, r in x1.items() if c != "E"}
    if e and abl:
        report["x1_ablations"] = {c: {"note": f"E minus {c} (descriptive only, v3.5)",
                                      **paired_vs(e, runs, budget, ("n_to_achieve", "verified_value", "achieved"))}
                                  for c, runs in sorted(abl.items())}
    if baseline_dir:
        report["amended_vs_baseline"] = compare_trees(baseline_dir, results_dir, budget)
    return report


def compare_trees(base_dir: str, new_dir: str, budget: int = 20) -> Dict[str, Any]:
    """The same experiment / condition on two snapshots (e.g. the pre-amendment baseline and the amended code),
    over the same seeds: counts of achieved / never reached / interrupted on each side and unpaired differences
    new minus baseline (descriptive only, v3.5)."""
    out: Dict[str, Any] = {"baseline_dir": os.path.abspath(base_dir), "new_dir": os.path.abspath(new_dir),
                           "note": "new minus baseline; no test can reach p < 0.05 (v3.5); " + UNPAIRED_NOTE}
    for exp in EXPERIMENTS + ("x2",):
        b, n = collect(base_dir, exp), collect(new_dir, exp)
        for c in sorted(set(b) & set(n)):
            sb, sn = summarise_condition(b[c], budget), summarise_condition(n[c], budget)
            side = {k: {"baseline": sb[k], "new": sn[k]} for k in
                    ("objectives_attempted", "objectives_achieved", "objectives_never_reached",
                     "objectives_interrupted", "objectives_not_run", "achieved_by_problem", "stops")}
            side["slow_steps"] = {"baseline": {k: sb["slow_steps"][k] for k in ("total", "kept", "reverted", "deferred")},
                                  "new": {k: sn["slow_steps"][k] for k in ("total", "kept", "reverted", "deferred")}}
            side["meta_updates_applied"] = {"baseline": f"{sb['meta_updates_applied_total']}/{sb['meta_update_records']}",
                                            "new": f"{sn['meta_updates_applied_total']}/{sn['meta_update_records']}"}
            out[f"{exp}/{c}"] = {**side, **paired_vs(n[c], b[c], budget,
                                                    ("n_to_achieve", "verified_value", "achieved",
                                                     "executable_fraction", "backbone_chosen_fraction"))}
    return out


def _drop_goalfree(runs: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {s: {**r, "objectives": [o for o in r.get("objectives", []) if o.get("family") != "GOALFREE"]}
            for s, r in runs.items()}


# ------------------------------------------------------------------------------------------------ tables
def _f(x: Any, nd: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def tables_md(report: Dict[str, Any]) -> str:
    """Per-objective Markdown tables of every experiment / condition / seed (input of RESULTS.md)."""
    lines: List[str] = []
    head = ("| seed | # | problem | status | value | dec. | ended | exec | bb chosen | bb cand. | rep. fail | "
            "strategies | NLL | meta applied / records | slow kept / reverted / deferred / skipped | wall s |")
    sep = "|" + "---|" * 16
    for exp in EXPERIMENTS + ("x2",):
        conds = report.get(exp)
        if not conds:
            continue
        if exp == "x2":
            conds = {k: conds[k] for k in ("initial", "experienced") if conds.get(k)}
        for c, summ in conds.items():
            lines += [f"#### {exp} / {c}", "", head, sep]
            for s, rows in summ["per_seed"].items():
                for x in rows:
                    sl = x.get("slow_steps") or {}
                    slow = (f"{sl.get('kept', 0)} / {sl.get('reverted', 0)} / {sl.get('deferred', 0)} / "
                            f"{sum((sl.get('skipped') or {}).values())}") if sl else "-"
                    lines.append(
                        f"| {s} | {x['index']} | {x['family']}:{x['problem']} | {x['verified_status']} | "
                        f"{_f(x['verified_value'])} | {_f(x['decisions'])} | {x['stop']} | "
                        f"{_f(x['executable_fraction'])} | {_f(x['backbone_chosen_fraction'])} | "
                        f"{_f(x['backbone_candidate_fraction'])} | {_f(x['repeats_of_failed_actions'])} | "
                        f"{_f(x['distinct_strategies'])} | {_f(x['mean_prediction_nll'])} | "
                        f"{x['meta_applied']} / {x['meta_records']} | {slow} | {_f(x['wall_s'], 0)} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--results-dir", dest="results_dir", default="",
                    help="results tree (<dir>/x0/E_s0.json ...); default CAM_LIVE_EVAL_HOME/results/v3")
    ap.add_argument("--expect-commit", dest="expect_commit", default="",
                    help="snapshot commit every result must record (code_check.all_match)")
    ap.add_argument("--markdown", default="", help="also write per-objective tables to this Markdown file")
    ap.add_argument("--baseline-dir", dest="baseline_dir", default="",
                    help="results tree of another snapshot to compare with, unpaired (summary_amended_vs_baseline)")
    a = ap.parse_args(argv)
    results_dir = a.results_dir
    if not results_dir:
        from .experiments import EVAL_HOME
        results_dir = os.path.join(EVAL_HOME, "results", "v3")
    os.makedirs(a.out_dir, exist_ok=True)
    report = build_report(results_dir, a.budget, expect_commit=a.expect_commit, baseline_dir=a.baseline_dir)
    for k, v in report.items():
        with open(os.path.join(a.out_dir, f"summary_{k}.json"), "w") as f:
            json.dump({"executed": True, "experiment": k, "results_dir": os.path.abspath(results_dir),
                       **({"conditions": v} if k in EXPERIMENTS else v)}, f, indent=1, default=str)
    if a.markdown:
        with open(a.markdown, "w") as f:
            f.write(tables_md(report))
    for k, v in report.items():
        if k in EXPERIMENTS:
            for c, s in v.items():
                print(f"{k} {c}: achieved {s['objectives_achieved']}/{s['objectives_attempted']} (interrupted "
                      f"{s['objectives_interrupted']}, not run {s['objectives_not_run']}) by problem "
                      f"{s['achieved_by_problem']} decisions-to-achieve {s['decisions_to_achieve_censored']['per_seed']}"
                      f" stops {s['stops']} exec {s['executable_fraction']['mean']} backbone "
                      f"{s['backbone_chosen_fraction']['mean']} meta applied {s['meta_updates_applied_total']} slow "
                      f"{ {k2: s['slow_steps'][k2] for k2 in ('total', 'kept', 'reverted', 'deferred')} } code "
                      f"{s['code_check']['all_match']}")
        elif k == "x2" and v.get("S_q"):
            print(f"x2: S_q {v['S_q'].get('S_q')} [{v['S_q'].get('lo')}, {v['S_q'].get('hi')}] pairs {len(v['pairs'])}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
