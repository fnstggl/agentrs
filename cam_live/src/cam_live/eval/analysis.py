"""cam_live.eval.analysis — prespecified analyses of executed runs (docs/S/PROTOCOLS.md).

Inputs are result files of ``experiments.run_condition`` (one per condition x seed); the unit of every interval is
the seed (one independent run: its own agent, its own sequence of worlds). Conditions with the same seed act in the
*same* sequence of worlds, so condition differences are paired by seed.

    baselines(results, anchors)     16.1 / 16.9: learning curves, AULC, N_q, learning slope, compute, pairwise tests
    calibration(results)            16.2: objective success forecasts, action.succeeded, count predictions
    action_quality(results)         16.3: evaluation vs ground truth of every candidate; randomised experiments
    delayed_credit(results)         16.4: delayed outcomes updating earlier records; conservation; E vs E_no_credit
    shadow(results)                 11.2: off-policy estimates of alternative policies with support diagnostics
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .stats import (aulc, brier, clustered_ci, diff_ci, ece, mean_ci, n_to_threshold, pearson, reliability,
                    spearman, summarise_nq, wilson)

Q_F1 = 0.75         # prespecified competence threshold on F1 (verified progress, rolling window of 2 lifetimes)
WINDOW = 2


def load(dirpath: str) -> Dict[str, List[Dict[str, Any]]]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        with open(p) as f:
            r = json.load(f)
        if r.get("executed"):
            by[r["condition"]].append(r)
    for v in by.values():
        v.sort(key=lambda r: r["seed"])
    return dict(by)


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


# ------------------------------------------------------------------------------------------------ 16.1 / 16.9
def per_seed(r: Dict[str, Any], q: float = Q_F1, window: int = WINDOW) -> Dict[str, Any]:
    lts = r["lifetimes"]
    curve = [lt["progress"] for lt in lts]                     # primary: verified progress
    eff = [lt["quality"] for lt in lts]                        # secondary: ground-truth money efficiency
    exp_ = []
    n = 0
    for lt in lts:
        n += lt["decisions"]
        exp_.append(float(n))
    h = len(curve) // 2
    return {"seed": r["seed"], "curve": curve, "progress": [lt["progress"] for lt in lts],
            "achieved": [bool(lt["achieved"]) for lt in lts], "aulc": aulc(curve),
            "slope": (_mean(curve[h:]) - _mean(curve[:h])) if h else None,
            "first": curve[0] if curve else None, "last2": _mean(curve[-2:]),
            "efficiency": _mean(eff), "efficiency_curve": eff,
            "efficiency_slope": (_mean(eff[h:]) - _mean(eff[:h])) if h else None,
            "nq": n_to_threshold(curve, q, experience=exp_, window=window),
            "decisions": n, "training_s": sum(lt["training_s"] for lt in lts),
            "inference_s": sum(lt["inference_s"] for lt in lts), "wall_s": r.get("wall_seconds"),
            "training_cpu_s": sum(lt.get("training_cpu_s", 0.0) for lt in lts),
            "inference_cpu_s": sum(lt.get("inference_cpu_s", 0.0) for lt in lts),
            "sim_days": sum(lt["sim_seconds"] for lt in lts) / 86400.0,
            "money": sum(lt["money"] for lt in lts),
            "noise_share": _share(r, "lab.noise"), "wait_share": _share(r, "wait"),
            "act_share": _share(r, "lab.act"), "probe_share": _share(r, "lab.probe")}


def _share(r: Dict[str, Any], op: str) -> Optional[float]:
    rows = [x for lt in r.get("decision_rows", []) for x in lt]
    if not rows:
        return None
    return sum(1 for d in rows if d["candidates"][d["chosen_index"]]["operation"] == op) / len(rows)


def baselines(by: Dict[str, List[Dict[str, Any]]], anchors: Optional[Dict[str, Any]] = None,
              q: float = Q_F1, reference: str = "E") -> Dict[str, Any]:
    out: Dict[str, Any] = {"q": q, "window": WINDOW, "conditions": {}, "pairwise_vs_" + reference: {}}
    ps = {c: [per_seed(r, q) for r in rs] for c, rs in by.items()}
    for c, rows in ps.items():
        n_lt = max((len(x["curve"]) for x in rows), default=0)
        out["conditions"][c] = {
            "n_seeds": len(rows), "real_backbone": by[c][0].get("real_backbone"), "backbone": by[c][0].get("backbone"),
            "scale": by[c][0].get("scale"),
            "aulc": mean_ci([x["aulc"] for x in rows]),
            "first_lifetime": mean_ci([x["first"] for x in rows]),
            "last_two_lifetimes": mean_ci([x["last2"] for x in rows]),
            "learning_slope_second_minus_first_half": mean_ci([x["slope"] for x in rows]),
            "curve_mean_by_lifetime": [mean_ci([x["curve"][i] for x in rows if len(x["curve"]) > i])
                                       for i in range(n_lt)],
            "verified_progress": mean_ci([_mean(x["progress"]) for x in rows]),
            "efficiency": mean_ci([x["efficiency"] for x in rows]),
            "efficiency_slope": mean_ci([x["efficiency_slope"] for x in rows]),
            "verified_achievement_rate": mean_ci([_mean([float(a) for a in x["achieved"]]) for x in rows]),
            "N_q": summarise_nq([x["nq"] for x in rows]),
            "compute_per_decision": {
                "training_s": mean_ci([x["training_s"] / max(1, x["decisions"]) for x in rows]),
                "inference_s": mean_ci([x["inference_s"] / max(1, x["decisions"]) for x in rows]),
                "training_cpu_s": mean_ci([x["training_cpu_s"] / max(1, x["decisions"]) for x in rows]),
                "inference_cpu_s": mean_ci([x["inference_cpu_s"] / max(1, x["decisions"]) for x in rows])},
            "elapsed": {"wall_s_per_run": mean_ci([x["wall_s"] for x in rows]),
                        "simulated_days_per_run": mean_ci([x["sim_days"] for x in rows])},
            "money_per_run": mean_ci([x["money"] for x in rows]),
            "action_shares": {k: mean_ci([x[k] for x in rows]) for k in ("act_share", "probe_share", "noise_share",
                                                                         "wait_share")},
            "per_seed": [{k: x[k] for k in ("seed", "curve", "aulc", "nq")} for x in rows]}
    if reference in ps:
        ref = {x["seed"]: x for x in ps[reference]}
        for c, rows in ps.items():
            if c == reference:
                continue
            seeds = [x["seed"] for x in rows if x["seed"] in ref]
            mine = {x["seed"]: x for x in rows}
            out["pairwise_vs_" + reference][c] = {
                "paired_seeds": len(seeds),
                "aulc_ref_minus_c": diff_ci([ref[s]["aulc"] for s in seeds], [mine[s]["aulc"] for s in seeds],
                                            paired=True),
                "last_two_ref_minus_c": diff_ci([ref[s]["last2"] for s in seeds], [mine[s]["last2"] for s in seeds],
                                                paired=True),
                "slope_ref_minus_c": diff_ci([ref[s]["slope"] for s in seeds], [mine[s]["slope"] for s in seeds],
                                             paired=True),
                "efficiency_ref_minus_c": diff_ci([ref[s]["efficiency"] for s in seeds],
                                                  [mine[s]["efficiency"] for s in seeds], paired=True)}
    if anchors:
        for kind, rows in anchors.get("anchors", {}).items():
            out.setdefault("anchors", {})[kind] = {"quality": mean_ci([r["quality"] for r in rows]),
                                                  "verified_progress": mean_ci([r["progress"] for r in rows]),
                                                  "achieved": wilson(sum(1 for r in rows if r["achieved"]),
                                                                     len(rows)), "n_worlds": len(rows)}
    return out


# ------------------------------------------------------------------------------------------------ 16.2
def _auc(p: Sequence[float], y: Sequence[float]) -> Optional[float]:
    pos = [a for a, b in zip(p, y) if b > 0.5]
    neg = [a for a, b in zip(p, y) if b <= 0.5]
    if not pos or not neg:
        return None
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else 0.5 if a == b else 0.0
    return s / (len(pos) * len(neg))


def calibration(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Prospective predictions only (registered before execution, resolved after): objective success forecasts
    (``objective.<id>``, bernoulli), the immediate ``action.succeeded`` and the goal-relevant count predictions."""
    obj_p, obj_y, obj_cluster = [], [], []
    act_p, act_y, act_cluster = [], [], []
    z_by_cluster: Dict[str, List[float]] = defaultdict(list)
    cover_by_cluster: Dict[str, List[float]] = defaultdict(list)
    nll_early, nll_late = [], []
    prospective_violations = 0
    for r in rs:
        n_lt = len(r["lifetimes"])
        lt_of = {}
        for lt in r["lifetimes"]:
            lt_of[lt["objective_id"]] = lt["index"]
        per_half: Dict[int, List[float]] = {0: [], 1: []}
        for p in r["extract"]["predictions"]:
            if p["resolved_at"] is None or p["made_at"] is None:
                continue
            if p["resolved_at"] <= p["made_at"]:
                prospective_violations += 1
                continue
            m = p["metric"]
            cl = f"{r['condition']}:{r['seed']}:{p['decision_id'][:14]}"
            if m.startswith("objective.") and p["p1"] is not None:
                oid = m[len("objective."):]
                obj_p.append(p["p1"])
                obj_y.append(p["observed"])
                obj_cluster.append(f"{r['seed']}:{oid}")
            elif m == "action.succeeded" and p["p1"] is not None:
                act_p.append(p["p1"])
                act_y.append(p["observed"])
                act_cluster.append(f"{r['seed']}")
            elif p["kind"] == "count" and p["goal_relevant"] and p["std"] and p["std"] > 0:
                z = (p["observed"] - p["mean"]) / p["std"]
                key = f"{r['seed']}"
                z_by_cluster[key].append(z * z)
                cover_by_cluster[key].append(1.0 if abs(z) <= 1.2816 else 0.0)
                if p["nll"] is not None:
                    per_half[0 if _lifetime_of_decision(r, p["decision_id"]) < n_lt // 2 else 1].append(p["nll"])
        if per_half[0] and per_half[1]:
            nll_early.append(_mean(per_half[0]))
            nll_late.append(_mean(per_half[1]))

    def block(p: List[float], y: List[float], cl: List[str]) -> Dict[str, Any]:
        if not p:
            return {"n": 0}
        base = _mean(y)
        byc: Dict[str, List[float]] = defaultdict(list)
        byc_base: Dict[str, List[float]] = defaultdict(list)
        for a, b, c in zip(p, y, cl):
            byc[c].append((a - b) ** 2)
            byc_base[c].append((base - b) ** 2)
        return {"n": len(p), "clusters": len(set(cl)), "brier": brier(p, y),
                "brier_ci_clustered": clustered_ci(byc), "brier_base_rate": brier([base] * len(y), y),
                "brier_base_rate_ci_clustered": clustered_ci(byc_base), "ece_10": ece(p, y),
                "mean_forecast": _mean(p), "observed_rate": base, "auc": _auc(p, y),
                "reliability": reliability(p, y)}
    return {"objective_success_forecasts": block(obj_p, obj_y, obj_cluster),
            "action_succeeded": block(act_p, act_y, act_cluster),
            "goal_count_predictions": {"mean_z2": clustered_ci(z_by_cluster),
                                       "coverage_80": clustered_ci(cover_by_cluster),
                                       "note": "calibrated predictive: mean z^2 ~ 1, central-80% coverage ~ 0.80"},
            "nll_first_half_vs_second_half_of_lifetimes": diff_ci(nll_late, nll_early, paired=True),
            "non_prospective_predictions_excluded": prospective_violations}


def _lifetime_of_decision(r: Dict[str, Any], decision_id: str) -> int:
    for i, rows in enumerate(r.get("decision_rows", [])):
        for d in rows:
            if d["decision_id"] == decision_id:
                return i
    return 0


# ------------------------------------------------------------------------------------------------ 16.3
def action_quality(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Does a higher evaluation mean a better actual outcome? (a) ground truth of all candidates at the decision
    (the environment is held fixed: same moment, same state), per seed; (b) the agent's randomised experiments:
    observed outcomes of the arm it predicted higher vs the other arm."""
    per_seed_rho, per_seed_top, per_seed_chosen, per_seed_ratio = [], [], [], []
    obs_pred, obs_seen = [], []
    for r in rs:
        rhos, top_gain, chosen_ratio, sel_ratio = [], [], [], []
        for lt_rows in r.get("decision_rows", []):
            for d in lt_rows:
                c = [x for x in d["candidates"] if x["goal_mean"] is not None]
                if len(c) < 3:
                    continue
                truth = [x["truth"] for x in c]
                pred = [x["goal_mean"] for x in c]
                if max(truth) <= 0:
                    continue
                rho = spearman(pred, truth)
                if rho is not None:
                    rhos.append(rho)
                i = max(range(len(c)), key=lambda j: pred[j])
                others = [t for j, t in enumerate(truth) if j != i]
                top_gain.append(truth[i] - sum(others) / len(others))
                best = max(truth)
                ch = d["candidates"][d["chosen_index"]]
                chosen_ratio.append(ch["truth"] / best)
                sel_ratio.append(truth[i] / best)
        if rhos:
            per_seed_rho.append(_mean(rhos))
            per_seed_top.append(_mean(top_gain))
            per_seed_chosen.append(_mean(chosen_ratio))
            per_seed_ratio.append(_mean(sel_ratio))
        recs = {x["exp_id"]: x for x in r["extract"].get("records", []) if isinstance(x, dict) and "exp_id" in x}
        del recs
    exps = []
    for r in rs:
        for e in (r["extract"].get("experiments") or []):
            if not isinstance(e, dict) or len(e.get("arms", [])) != 2:
                continue
            exps.append({"seed": r["seed"], **{k: e[k] for k in ("experiment_id", "status", "supported")},
                         "n_units": len(e.get("assignments", [])), "n_actual": len(e.get("actual", []))})
    return {"spearman_predicted_vs_true_value_over_candidates": mean_ci(per_seed_rho),
            "true_value_gain_of_top_predicted_candidate_over_mean_alternative": mean_ci(per_seed_top),
            "true_value_ratio_of_top_predicted_to_best": mean_ci(per_seed_ratio),
            "true_value_ratio_of_chosen_to_best": mean_ci(per_seed_chosen),
            "randomised_experiments": {"n": len(exps), "concluded": sum(1 for e in exps if e["supported"] is not None),
                                       "supported": sum(1 for e in exps if e["supported"] is True),
                                       "refuted": sum(1 for e in exps if e["supported"] is False),
                                       "units": sum(e["n_units"] for e in exps), "items": exps[:40]},
            "note": "truth = expected adopters of the candidate at that moment from the hidden mechanism (0 for "
                    "operations without direct effect); goal_mean = the agent's registered prediction of the "
                    "goal-relevant outcome of that candidate"}


# ------------------------------------------------------------------------------------------------ 16.4
def delayed_credit(rs: Sequence[Dict[str, Any]], ablation: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    per_seed_delayed, per_seed_changed, per_seed_frac = [], [], []
    cons = []
    for r in rs:
        ev = r["extract"].get("credit_events", [])
        att = [e for e in ev if e["kind"] == "outcome_attached" and e["later_decisions"] >= 1]
        ch = [e for e in ev if e["kind"] == "credit_changed" and e["later_decisions"] >= 1]
        att_ids = {e["exp_id"] for e in att}
        ch_ids = {e["exp_id"] for e in ch}
        per_seed_delayed.append(float(len(att_ids)))
        per_seed_changed.append(float(len(ch_ids)))
        per_seed_frac.append(len(att_ids & ch_ids) / len(att_ids) if att_ids else None)
        for lt in r["lifetimes"]:
            c = lt.get("conservation") or {}
            if c.get("conservation_error") is not None:
                cons.append(c["conservation_error"])
    out = {"records_with_delayed_outcome_per_run": mean_ci(per_seed_delayed),
           "records_whose_credit_changed_after_later_decisions_per_run": mean_ci(per_seed_changed),
           "fraction_of_delayed_outcome_records_whose_credit_changed": mean_ci(per_seed_frac),
           "rudder_conservation_error": {"n_objectives": len(cons), "max": max(cons) if cons else None,
                                         "mean": _mean(cons)}}
    if ablation:
        ref = {per_seed(r)["seed"]: per_seed(r) for r in rs}
        abl = {per_seed(r)["seed"]: per_seed(r) for r in ablation}
        seeds = sorted(set(ref) & set(abl))
        out["learning_efficiency_E_minus_E_no_credit"] = {
            "paired_seeds": len(seeds),
            "aulc": diff_ci([ref[s]["aulc"] for s in seeds], [abl[s]["aulc"] for s in seeds], paired=True),
            "last_two": diff_ci([ref[s]["last2"] for s in seeds], [abl[s]["last2"] for s in seeds], paired=True),
            "N_q_E": summarise_nq([ref[s]["nq"] for s in seeds]),
            "N_q_E_no_credit": summarise_nq([abl[s]["nq"] for s in seeds])}
    return out


# ------------------------------------------------------------------------------------------------ 11.2
def shadow(rs: Sequence[Dict[str, Any]], *, eps: float = 1e-12) -> Dict[str, Any]:
    """Off-policy value of alternative decision rules on the recorded decisions of the behaviour policy.

    Reward of a logged decision = the *observed* consequence (adopters measured for the executed intervention,
    0 for operations without direct effect). Target policies are functions of what the agent recorded at
    decision time. Three columns are kept apart: the off-policy estimate from observed outcomes (IPS, SNIPS with
    a bootstrap over seeds), the model-based prediction (sum_a pi(a) * predicted goal mean), and — only possible
    in a diagnostic world — the ground truth on the same contexts (sum_a pi(a) * true expected value)."""
    targets = {
        "behaviour": lambda d: d["probabilities"],
        "uniform": lambda d: [1.0 / len(d["candidates"])] * len(d["candidates"]),
        "greedy_predicted_goal_mean": lambda d: _onehot(d, [x["goal_mean"] for x in d["candidates"]]),
        "greedy_p_success": lambda d: _onehot(d, [x["p_success"] for x in d["candidates"]]),
        "always_act_max_effort": lambda d: _onehot(d, [((x["arguments"] or {}).get("effort", 0) if
                                                        x["operation"] == "lab.act" else -1)
                                                       for x in d["candidates"]]),
    }
    res: Dict[str, Any] = {}
    for name, fn in targets.items():
        ips_s, snips_s, model_s, truth_s, ess_s, support_s = [], [], [], [], [], []
        for r in rs:
            obs = _observed_rewards(r)
            num = den = msum = tsum = 0.0
            ws = []
            supp = []
            n = 0
            for lt_rows in r.get("decision_rows", []):
                for d in lt_rows:
                    pi = fn(d)
                    mu = d["probabilities"]
                    i = d["chosen_index"]
                    if not mu or mu[i] <= eps:
                        continue
                    rew = obs.get(d["decision_id"])
                    if rew is None:
                        continue
                    w = pi[i] / mu[i]
                    ws.append(w)
                    num += w * rew
                    den += w
                    msum += sum(p * (x["goal_mean"] or 0.0) for p, x in zip(pi, d["candidates"]))
                    tsum += sum(p * x["truth"] for p, x in zip(pi, d["candidates"]))
                    supp.append(sum(p for p, m in zip(pi, mu) if m > 0.0))
                    n += 1
            if n == 0:
                continue
            ips_s.append(num / n)
            snips_s.append(num / den if den > 0 else None)
            model_s.append(msum / n)
            truth_s.append(tsum / n)
            ess_s.append((sum(ws) ** 2) / sum(w * w for w in ws) / n if ws else None)
            support_s.append(_mean(supp))
        res[name] = {"ips_estimate_observed": mean_ci(ips_s), "snips_estimate_observed": mean_ci(snips_s),
                     "model_prediction": mean_ci(model_s), "ground_truth_same_contexts": mean_ci(truth_s),
                     "ess_fraction": mean_ci(ess_s), "target_mass_on_logged_support": mean_ci(support_s),
                     "n_runs": len(ips_s)}
    return {"targets": res, "reward": "observed adopters attributed to the executed intervention (0 otherwise)",
            "note": "estimates (columns 1-2) and model predictions (3) are not observations of the alternative "
                    "policy's consequences; column 4 exists only because the world is diagnostic"}


def _onehot(d: Dict[str, Any], score: Sequence[Optional[float]]) -> List[float]:
    vals = [(-math.inf if s is None else float(s)) for s in score]
    i = max(range(len(vals)), key=lambda j: vals[j])
    return [1.0 if j == i else 0.0 for j in range(len(vals))]


def _observed_rewards(r: Dict[str, Any]) -> Dict[str, float]:
    """decision id -> observed attributed adopters of the executed action (records joined through the decision
    rows by order within each objective)."""
    by_obj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for x in r["extract"].get("records", []):
        if isinstance(x, dict) and "exp_id" in x:
            by_obj[x["objective_id"]].append(x)
    out: Dict[str, float] = {}
    for lt, rows in zip(r["lifetimes"], r.get("decision_rows", [])):
        recs = sorted(by_obj.get(lt["objective_id"], []), key=lambda x: x["seq"])
        for d, rec in zip(rows, recs):
            ch = d["candidates"][d["chosen_index"]]
            if rec.get("operation") != ch["operation"]:
                continue
            if ch["operation"] == "lab.act":
                vals = [v for v in rec.get("outcome_values", []) if v is not None]
                out[d["decision_id"]] = float(vals[0]) if vals else 0.0
            else:
                out[d["decision_id"]] = 0.0
    return out
