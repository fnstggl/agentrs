"""Aggregation of the self-play experiment jobs into the result files of `examples/ow_selfplay.py`.

Units. Worlds differ in size by an order of magnitude, so every net income is reported as **percent of the world's
demand income per tick** (`pct`): 100 x net per tick / `Task.income`. That is the reward unit of the learner
interface times 100, and it makes runs on different worlds comparable. A *run* is one world and one set of learner
seeds; every interval is a bootstrap interval over runs (`research.stats`), every comparison between conditions is
paired on the run (same world, same individuals where applicable, same evaluation noise).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Sequence

import numpy as np

from ..research import stats
from .arena import arena_task

_SCALE: dict[tuple, float] = {}


def scale(family: str, task_seed: int) -> float:
    key = (family, int(task_seed))
    if key not in _SCALE:
        _SCALE[key] = float(arena_task(family, task_seed).income)
    return _SCALE[key]


def pct(x, family, ts):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [pct(v, family, ts) for v in x]
    return 100.0 * float(x) / scale(family, ts)


def ci(x: Sequence[float]) -> dict:
    return stats.mean_ci([v for v in x if v is not None])


def pd(a: Sequence[float], b: Sequence[float]) -> dict:
    return stats.paired_diff(list(a), list(b))


def _jobs(res: dict, prefix: str, family: str | None = None) -> list[dict]:
    return [v for k, v in sorted(res.items()) if k.startswith(prefix) and (family is None or v["job"].get("family") == family)]


def _fam(v: dict) -> tuple[str, int]:
    return v["job"]["family"], int(v["job"]["task_seed"])


# ====================================================================== A
def report_A(res: dict, cfg: dict) -> dict:
    out: dict[str, Any] = {"experiment": "A: autonomous competition in abm_duopoly", "unit": "pct of demand income per tick"}
    # ---- joint training: where the pairs end up
    pairs = _jobs(res, "pair_", "abm_duopoly")
    rows = []
    for v in pairs:
        f, ts = _fam(v)
        rows.append({"task": ts, "algos": v["job"]["algos"], "seeds": v["job"]["seeds"],
                     "final_net": pct(v["final"]["net_per_tick"], f, ts), "final_joint": pct(v["final"]["joint_net_per_tick"], f, ts),
                     "default_joint": pct(v["defaults"]["joint_net_per_tick"], f, ts), "view": v["final_view"],
                     "curve_eval_joint": [[r["it"], pct(r["eval_joint"], f, ts)] for r in v["curve"] if "eval_joint" in r],
                     "series": {"what": "per training iteration, the mean settings each individual played (operator-side "
                                        "reading: price as a multiple of the reference price, outreach as a fraction of "
                                        "its range) and its net in pct",
                                "price": [[round(r["view"][s]["price"], 4) for r in v["curve"]] for s in range(2)],
                                "outreach": [[round(r["view"][s]["outreach"], 4) for r in v["curve"]] for s in range(2)],
                                "net": [[round(pct(r["net"][s], f, ts), 3) for r in v["curve"]] for s in range(2)]},
                     "branch_reproduces_main": v.get("branch_reproduces_main")})
    out["joint_training"] = {
        "runs": rows,
        "final_joint": {a: ci([r["final_joint"] for r in rows if r["algos"][0] == a]) for a in ("spsa", "pg")},
        "final_price_multiple": {a: ci([np.mean([x["price"] for x in r["view"]]) for r in rows if r["algos"][0] == a])
                                 for a in ("spsa", "pg")},
        "default_joint": ci([r["default_joint"] for r in rows]),
        "checkpoint_branches_reproduce_the_run": [r["branch_reproduces_main"] for r in rows]}
    # ---- (i) the 2x2
    a1 = _jobs(res, "a1_")
    keys = ["B_aggressive|A_aggressive", "B_passive|A_aggressive", "B_aggressive|A_passive", "B_passive|A_passive"]
    if a1:
        n = {k: [pct(v["net"][k], *_fam(v)) for v in a1] for k in keys}
        own = [x + y for x, y in zip(n[keys[0]], n[keys[3]])]
        oth = [x + y for x, y in zip(n[keys[1]], n[keys[2]])]
        out["i_trained_against_different_opponents"] = {
            "design": "B (seat 1) trained against a frozen aggressive scripted A, and the same learner (same seed) "
                      "trained against a frozen passive scripted A; each B evaluated frozen against both A on the "
                      "same noise. A run = (world, algorithm).",
            "net_of_B": {k: ci(n[k]) for k in keys},
            "vs_aggressive_A: B_trained_on_it minus B_trained_on_passive": pd(n[keys[0]], n[keys[1]]),
            "vs_passive_A: B_trained_on_it minus B_trained_on_aggressive": pd(n[keys[3]], n[keys[2]]),
            "specialisation (own-opponent sum minus other-opponent sum)": pd(own, oth),
            "learned_settings_B_aggressive_minus_B_passive": {
                role: pd([v["view"]["aggressive"][role] for v in a1], [v["view"]["passive"][role] for v in a1])
                for role in ("price", "outreach", "quality")},
            "learned_settings": {t: {role: ci([v["view"][t][role] for v in a1]) for role in ("price", "outreach", "quality")}
                                 for t in ("aggressive", "passive")},
            "runs": [{"task": v["job"]["task_seed"], "algo": v["job"]["algo"],
                      "net": {k: pct(v["net"][k], *_fam(v)) for k in keys}, "view": v["view"]} for v in a1]}
    # ---- (ii) lagged relation and the intervention
    lag_rows, shock = {"price": [], "outreach": []}, {}
    for v in pairs:
        for role in lag_rows:
            xa = [r["view"][0][role] for r in v["curve"]]
            xb = [r["view"][1][role] for r in v["curve"]]
            lr = __import__("crucible.selfplay.modes", fromlist=["lagged_response"]).lagged_response(xa, xb, 4)
            lag_rows[role].append(lr["corr"])
        for prof, sh in (v.get("shock") or {}).items():
            K = sh["at"]
            ctrl = {r["it"]: r["view"][1] for r in v["curve"] if r["it"] >= K}
            its = [r["it"] for r in sh["curve"]]
            for role in ("price", "outreach", "quality"):
                d = [r["view"][1][role] - ctrl[r["it"]][role] for r in sh["curve"]]
                e = shock.setdefault(prof, {}).setdefault(role, {"series": [], "last5_shock": [], "last5_control": [], "algo": []})
                e["series"].append(d)
                e["last5_shock"].append(float(np.mean([r["view"][1][role] for r in sh["curve"][-5:]])))
                e["last5_control"].append(float(np.mean([ctrl[i][role] for i in its[-5:]])))
                e["algo"].append(v["job"]["algos"][1])
            e = shock[prof].setdefault("net_B", {"last5_shock": [], "last5_control": []})
            f, ts = _fam(v)
            main = {r["it"]: r["net"][1] for r in v["curve"]}
            e["last5_shock"].append(pct(float(np.mean([r["net"][1] for r in sh["curve"][-5:]])), f, ts))
            e["last5_control"].append(pct(float(np.mean([main[i] for i in its[-5:]])), f, ts))
    lags = list(range(-4, 5))
    out["ii_one_policy_follows_the_other"] = {
        "lagged_correlation": {
            "what": "correlation between the iteration-to-iteration change of A's setting at iteration t and of "
                    "B's at t + lag, per run, then averaged over runs (positive lag: B after A; negative: A after B)",
            "lags": lags,
            **{role: [ci([c[i] for c in lag_rows[role]]) for i in range(len(lags))] for role in lag_rows}},
        "intervention": {
            "design": "at the middle of joint training, A is replaced by a frozen scripted profile; B continues "
                      "from the identical checkpoint (weights, optimiser, random state, training noise). The control "
                      "is the uninterrupted run. Difference = B after the intervention minus B in the control, "
                      "mean of the last five iterations of the branch.",
            **{prof: {**{role: {"B_under_intervention_minus_control": pd(e[role]["last5_shock"], e[role]["last5_control"]),
                                "by_algo": {a: pd([x for x, g in zip(e[role]["last5_shock"], e[role]["algo"]) if g == a],
                                                  [x for x, g in zip(e[role]["last5_control"], e[role]["algo"]) if g == a])
                                            for a in ("spsa", "pg")},
                                "mean_difference_by_iteration_after": np.mean(np.array(e[role]["series"]), axis=0).tolist(),
                                "first_iteration_weights_still_identical_abs_difference": {
                                    a: ci([abs(x[0]) for x, g in zip(e[role]["series"], e[role]["algo"]) if g == a])
                                    for a in ("spsa", "pg")}}
                         for role in ("price", "outreach", "quality")},
                      "net_B_pct": pd(e["net_B"]["last5_shock"], e["net_B"]["last5_control"])}
               for prof, e in shock.items()}}}
    if "aggressive" in shock and "passive" in shock:
        out["ii_one_policy_follows_the_other"]["intervention"]["aggressive_minus_passive"] = {
            role: pd(shock["aggressive"][role]["last5_shock"], shock["passive"][role]["last5_shock"])
            for role in ("price", "outreach", "quality")}
    # ---- (iii) exploitability
    br = _jobs(res, "br_")
    if br:
        def col(kind, key):
            return [pct(v["responses"][kind][key], "abm_duopoly", v["job"]["task_seed"]) for v in br if kind in v["responses"]]
        partner = [pct(v["with_partner"]["partner_net"], "abm_duopoly", v["job"]["task_seed"]) for v in br]
        target = [pct(v["with_partner"]["target_net"], "abm_duopoly", v["job"]["task_seed"]) for v in br]
        ex = {"design": "the seat-1 individual of a trained pair is frozen; a best response is trained in seat 0 "
                        "against it: fresh (a new spsa learner from the defaults) and warm (a copy of its training "
                        "partner that keeps learning). Gain = what the best response earns against the frozen "
                        "target minus what the training partner earned against it, on the same noise.",
              "partner_net": ci(partner), "target_net_with_partner": ci(target), "responses": {}}
        for kind in sorted({k for v in br for k in v["responses"]}):
            ex["responses"][kind] = {"br_net": ci(col(kind, "br_net")), "gain_over_training_partner": pd(col(kind, "br_net"), partner),
                                     "target_net": ci(col(kind, "target_net")),
                                     "target_change": pd(col(kind, "target_net"), target),
                                     "by_target_algo": {a: pd([x for x, v in zip(col(kind, "br_net"), br) if v["target_arch"] == a],
                                                              [x for x, v in zip(partner, br) if v["target_arch"] == a])
                                                        for a in ("spsa", "pg")}}
        ex["runs"] = [{"task": v["job"]["task_seed"], "target": v["target"], "target_arch": v["target_arch"],
                       "partner_net": p, "target_net": t,
                       **{k: {"br_net": pct(r["br_net"], "abm_duopoly", v["job"]["task_seed"]),
                              "target_net": pct(r["target_net"], "abm_duopoly", v["job"]["task_seed"])}
                          for k, r in v["responses"].items()}} for v, p, t in zip(br, partner, target)]
        out["iii_exploitability"] = ex
    # ---- the audit of the trained pairs
    au = _jobs(res, "audit_", "abm_duopoly")
    if au:
        out["audit_of_trained_pairs"] = _audit_summary(au)
    return out


def _audit_summary(au: Sequence[dict]) -> dict:
    return {"reference": "each member of the pair with a non-cooperating scripted partner (the 'compete' profile)",
            "classes": dict(Counter(v["pair"]["class"] for v in au)),
            "classes_by_algo": {a: dict(Counter(v["pair"]["class"] for v in au if v["arch"][0] == a)) for a in ("spsa", "pg")},
            "calibration_scripted_collusion": dict(Counter(v["calibration_collude_vs_compete"]["class"] for v in au
                                                           if "calibration_collude_vs_compete" in v)),
            "units_relative_change": ci([v["pair"]["value_delivered"]["relative_change"] for v in au]),
            "price_per_unit_relative_change": ci([v["pair"]["price_per_unit"]["relative_change"] for v in au]),
            "runs": [{"task": v["job"]["task_seed"], "arch": v["arch"], "class": v["pair"]["class"], "flags": v["pair"]["flags"],
                      "joint_diff_pct": pct(v["pair"]["joint"]["diff"], *_fam(v)),
                      "units_rel": v["pair"]["value_delivered"]["relative_change"],
                      "price_rel": v["pair"]["price_per_unit"]["relative_change"], "view": v["view"]} for v in au]}


# ====================================================================== B
def report_B(res: dict, cfg: dict) -> dict:
    out: dict[str, Any] = {"experiment": "B: cooperation", "unit": "pct of demand income per tick",
                           "what_cooperation_consists_of": {
                               "tech_handoff": "native objective = the mechanism's own score: every seat is credited half of the "
                                               "facility's revenue minus half of all costs (the joint outcome). Ablation: each seat "
                                               "is trained on its own net only (half the revenue minus its own costs); nothing else "
                                               "changes (same observations, same money flows, same seeds).",
                               "abm_complement": "native objective = own net (own price x bundles sold minus own cost). The reverse "
                                                 "ablation adds the cooperative channel: each seat is trained on the mean of both "
                                                 "nets. In both families no money moves between seats and no observation changes."}}
    for family in ("tech_handoff", "abm_complement"):
        jobs = _jobs(res, "coop_", family)
        if not jobs:
            continue
        conds = sorted({c for v in jobs for c in v["cond"]})

        def col(c, key="joint", seat=None, only=None):
            xs = []
            for v in jobs:
                if c not in v["cond"] or (only is not None and any(o not in v["cond"] for o in only)):
                    continue
                x = v["cond"][c][key]
                xs.append(pct(x if seat is None else x[seat], *_fam(v)))
            return xs

        fam: dict[str, Any] = {"n_runs": len(jobs), "joint": {c: ci(col(c)) for c in conds},
                               "net_seat0": {c: ci(col(c, "net", 0)) for c in conds},
                               "net_seat1": {c: ci(col(c, "net", 1)) for c in conds}, "paired": {}}
        for other in conds:
            if other == "pair_native":
                continue
            pair_only = ("pair_native", other)
            fam["paired"][f"pair_native minus {other} (joint)"] = pd(col("pair_native", only=pair_only), col(other, only=pair_only))
        # the claim: the trained pair reaches what neither member reaches with a non-cooperating partner
        for label, c0, c1 in (("default partner", "A+default", "default+B"), ("self-interested solo partner", "A+soloB", "soloA+B")):
            if c0 in conds and c1 in conds:
                only = ("pair_native", c0, c1)
                best = [max(a, b) for a, b in zip(col(c0, only=only), col(c1, only=only))]
                fam["paired"][f"pair_native minus best of its members with a {label} (joint)"] = pd(col("pair_native", only=only), best)
                fam["paired"][f"seat0: own net in the pair minus with a {label}"] = pd(col("pair_native", "net", 0, only), col(c0, "net", 0, only))
                fam["paired"][f"seat1: own net in the pair minus with a {label}"] = pd(col("pair_native", "net", 1, only), col(c1, "net", 1, only))
        fam["by_algo"] = {a: {c: ci([pct(v["cond"][c]["joint"], *_fam(v)) for v in jobs if v["job"]["algo"] == a and c in v["cond"]])
                              for c in conds} for a in ("spsa", "pg")}
        fam["audit"] = {k: dict(Counter(v["audit"][k]["class"] for v in jobs if k in v["audit"]))
                        for k in sorted({k for v in jobs for k in v["audit"]})}
        key = "completed" if family.startswith("tech") else "units_seats"
        fam["value_delivered"] = {"what": "jobs completed while still wanted, per match" if family.startswith("tech")
                                  else "units customers bought from the two seats, per match",
                                  **{c: ci([v["cond"][c]["private"].get(key) for v in jobs if c in v["cond"]]) for c in conds}}
        fam["runs"] = [{"task": v["job"]["task_seed"], "algo": v["job"]["algo"], "pair_index": v["job"]["pair_index"],
                        "joint": {c: pct(v["cond"][c]["joint"], *_fam(v)) for c in v["cond"]},
                        "net": {c: pct(v["cond"][c]["net"], *_fam(v)) for c in v["cond"]},
                        "audit": {k: a["class"] for k, a in v["audit"].items()}} for v in jobs]
        # learning curves of the native pairs
        pj = [v for v in _jobs(res, "pair_", family) if v["job"]["objective"] == ("joint" if family == "tech_handoff" else "own")]
        fam["curves_eval_joint"] = [{"task": v["job"]["task_seed"], "algos": v["job"]["algos"],
                                     "curve": [[r["it"], pct(r["eval_joint"], *_fam(v))] for r in v["curve"] if "eval_joint" in r]}
                                    for v in pj]
        out[family] = fam
    return out


# ====================================================================== C
def _grab_index(view_pair: dict, view_r: dict, view_g: dict) -> float | None:
    """Where a seat's pool-drawing lever sits between the scripted restraint (0) and over-use (1) settings."""
    keys = [k for k in view_pair if k.startswith("cache")] or [k for k in view_pair if k == "outreach"]
    vals = []
    for k in keys:
        gap = view_g[k] - view_r[k]
        if abs(gap) > 1e-9:
            vals.append((view_pair[k] - view_r[k]) / gap)
    return float(np.mean(vals)) if vals else None


def report_C(res: dict, cfg: dict) -> dict:
    out: dict[str, Any] = {"experiment": "C: mixed motive", "unit": "pct of demand income per tick",
                           "position": "0 = joint net of both scripted over-users, 1 = joint net of both scripted restrainers"}
    for family in ("abm_commons", "tech_shared_pool"):
        jobs = _jobs(res, "commons_", family)
        if not jobs:
            continue
        conds = sorted({c for v in jobs for c in v["cond"]})
        col = lambda c, key="joint", seat=None: [pct(v["cond"][c][key] if seat is None else v["cond"][c][key][seat], *_fam(v)) for v in jobs]  # noqa: E731
        pool = "pool" if family == "abm_commons" else "pool_util"
        gi = [[_grab_index(v["cond"]["pair"]["view"][s], v["cond"]["both_restrain"]["view"][s], v["cond"]["both_grab"]["view"][s])
               for s in range(2)] for v in jobs]
        fam = {"n_runs": len(jobs), "position": ci([v["position"] for v in jobs]),
               "position_by_algo": {a: ci([v["position"] for v in jobs if v["arch"][0] == a]) for a in ("spsa", "pg")},
               "joint": {c: ci(col(c)) for c in conds},
               "net_seat0": {c: ci(col(c, "net", 0)) for c in conds}, "net_seat1": {c: ci(col(c, "net", 1)) for c in conds},
               "paired": {"pair minus both_restrain (joint)": pd(col("pair"), col("both_restrain")),
                          "pair minus both_grab (joint)": pd(col("pair"), col("both_grab"))},
               "shared_pool": {"what": "mean level of the shared attention stock (1 = untouched)" if family == "abm_commons"
                               else "mean utilisation of the shared pool (thrashing above a hidden threshold)",
                               **{c: ci([v["cond"][c]["private"].get(pool) for v in jobs]) for c in conds}},
               "grab_index": {"what": "each seat's pool-drawing lever between scripted restraint (0) and over-use (1)",
                              "more_grabbing_seat": ci([max(x) for x in gi if None not in x]),
                              "less_grabbing_seat": ci([min(x) for x in gi if None not in x]),
                              "mean": ci([float(np.mean(x)) for x in gi if None not in x])},
               "net_asymmetry_abs_diff": ci([abs(a - b) for a, b in zip(col("pair", "net", 0), col("pair", "net", 1))]),
               "audit": {k: dict(Counter(v["audit"][k]["class"] for v in jobs)) for k in jobs[0]["audit"]},
               "runs": [{"task": v["job"]["task_seed"], "arch": v["arch"], "position": v["position"],
                         "joint": {c: pct(v["cond"][c]["joint"], *_fam(v)) for c in v["cond"]},
                         "net_pair": pct(v["cond"]["pair"]["net"], *_fam(v)), "grab_index": g,
                         "pool": v["cond"]["pair"]["private"].get(pool)} for v, g in zip(jobs, gi)]}
        pj = _jobs(res, "pair_", family)
        fam["curves_eval_joint"] = [{"task": v["job"]["task_seed"], "algos": v["job"]["algos"],
                                     "curve": [[r["it"], pct(r["eval_joint"], *_fam(v))] for r in v["curve"] if "eval_joint" in r]}
                                    for v in pj]
        out[family] = fam
    return out


# ====================================================================== D
def _category(h: str) -> str:
    h = h.split(":s")[0] if h.count(":") <= 2 and not h.startswith(("ho:set_aside", "ho:other_algo")) else h
    if h.startswith("ho:set_aside"):
        return "earlier checkpoint set aside"
    if h.startswith("ho:other_algo"):
        return "other algorithm (final)"
    name = h[3:]
    if name.startswith("climber_funds"):
        return "scripted climber, funds x" + name[len("climber_funds"):]
    if name in ("hold", "random", "climber"):
        return {"hold": "idle holder", "random": "random operator", "climber": "scripted climber"}[name]
    return f"profile {name}"


def report_D(res: dict, cfg: dict) -> dict:
    out: dict[str, Any] = {"experiment": "D: population, history and cross-play", "unit": "pct of demand income per tick"}
    # ---- cross-play
    cp_all = _jobs(res, "crossplay_")
    cp: dict[str, Any] = {"design": "four pairs per world trained independently (pairs 0-1 spsa, 2-3 pg; different seeds); "
                                    "entry [i][j] = seat-0 individual of pair i with seat-1 individual of pair j, frozen, "
                                    "same noise. Gap = diagonal mean minus off-diagonal mean."}
    for family in sorted({v["job"]["family"] for v in cp_all}):
        jobs = [v for v in cp_all if v["job"]["family"] == family]
        f = lambda v, key: pct(v["matrix"][key], *_fam(v))  # noqa: E731
        same, cross = [], []
        for v in jobs:
            m = np.array(pct(v["matrix"]["joint"], *_fam(v)))
            al = v["job"]["algos"]
            same.append(float(np.mean([m[i, j] for i in range(4) for j in range(4) if i != j and al[i] == al[j]])))
            cross.append(float(np.mean([m[i, j] for i in range(4) for j in range(4) if al[i] != al[j]])))
        ind = [x for v in jobs for x in pct(v["matrix"]["gap_joint_by_seat0"] + v["matrix"]["gap_joint_by_seat1"], *_fam(v))]
        cp[family] = {"split": jobs[0]["matrix"] and arena_task(family, jobs[0]["job"]["task_seed"]).split, "worlds": len(jobs),
                      "diag_joint": ci([f(v, "diag_joint") for v in jobs]), "offdiag_joint": ci([f(v, "offdiag_joint") for v in jobs]),
                      "gap_joint": ci([f(v, "gap_joint") for v in jobs]),
                      "gap_joint_paired": pd([f(v, "diag_joint") for v in jobs], [f(v, "offdiag_joint") for v in jobs]),
                      "gap_own_net_seat0": ci([f(v, "gap_net0") for v in jobs]), "gap_own_net_seat1": ci([f(v, "gap_net1") for v in jobs]),
                      "gap_joint_per_individual": ci(ind),
                      "offdiag_same_algorithm": ci(same), "offdiag_other_algorithm": ci(cross),
                      "matrices": [{"task": v["job"]["task_seed"], "joint": pct(v["matrix"]["joint"], *_fam(v)),
                                    "net0": pct(v["matrix"]["net0"], *_fam(v)), "net1": pct(v["matrix"]["net1"], *_fam(v))} for v in jobs]}
    out["cross_play"] = cp
    # ---- history
    hist: dict[str, Any] = {"design": "archived stages (every 10 iterations) of each lineage: a frozen round robin of the "
                                      "stages of one lineage (both seatings, same noise) rated with the league's Elo, and "
                                      "every stage against the final stage of the other lineage."}
    for family in ("abm_duopoly", "abm_duopoly_fx"):
        jobs = _jobs(res, "history_", family)
        if not jobs:
            continue
        lin = [(v, pid, h) for v in jobs for pid, h in v["lineage"].items()]
        stages = lin[0][2]["stages"] if lin else []
        vs = [(v, rows) for v in jobs for rows in v["vs_final"].values()]
        hist[family] = {"lineages": len(lin), "stages": stages,
                        "elo_by_stage": [ci([h["elo"][i] for _, _, h in lin if len(h["elo"]) > i]) for i in range(len(stages))],
                        "later_stage_beats_earlier_fraction": ci([h["later_beats_earlier"] for _, _, h in lin]),
                        "last_stage_beats_all_earlier": stats.proportion_ci(sum(bool(h["last_beats_all_earlier"]) for _, _, h in lin), len(lin)),
                        "lineages_with_a_cycle": stats.proportion_ci(sum(1 for _, _, h in lin if h["cycles"]), len(lin)),
                        "by_algo": {a: {"later_stage_beats_earlier_fraction": ci([h["later_beats_earlier"] for v, pid, h in lin
                                                                                  if v["arch"][v["job"]["ids"].index(pid)] == a]),
                                        "elo_by_stage": [ci([h["elo"][i] for v, pid, h in lin if v["arch"][v["job"]["ids"].index(pid)] == a
                                                             and len(h["elo"]) > i]) for i in range(len(stages))]}
                                    for a in ("spsa", "pg")},
                        "stage_net_against_final_opponent": [ci([pct(rows[i]["net"], *_fam(v)) for v, rows in vs if len(rows) > i])
                                                             for i in range(len(stages))],
                        "final_opponent_net_against_stage": [ci([pct(rows[i]["final_opponent_net"], *_fam(v)) for v, rows in vs if len(rows) > i])
                                                             for i in range(len(stages))],
                        "win_matrices": [{"task": v["job"]["task_seed"], "lineage": pid, "win": h["win"], "cycles": h["cycles"]}
                                         for v, pid, h in lin]}
    out["history"] = hist
    # ---- held-out opponents
    ho: dict[str, Any] = {"design": "final individuals (frozen) against opponents that never took part in any training: "
                                    "one batch of matches per opponent on held-out noise; win = higher own net in the match."}
    for family in cfg["heldout_families"]:
        jobs = _jobs(res, "heldout_", family)
        if not jobs:
            continue
        rows = []
        for v in jobs:
            for pid, arch in zip(v["job"]["ids"], v["arch"]):
                ev = v["eval"][pid]
                cats: dict[str, list] = {}
                for h, r in ev["vs"].items():
                    cats.setdefault(_category(h), []).append(r)
                rows.append({"task": v["job"]["task_seed"], "id": pid, "arch": arch,
                             "by": {c: {"net": pct(float(np.mean([r["net_per_tick"] for r in rs])), *_fam(v)),
                                        "opp_net": pct(float(np.mean([r["opponent_net_per_tick"] for r in rs])), *_fam(v)),
                                        "win": float(np.mean([r["win_rate"] for r in rs])),
                                        "opp_idle_ticks": int(sum(r["opponent_ticks_idle"] for r in rs))} for c, rs in cats.items()},
                             "mean_net": pct(ev["mean_net_per_tick"], *_fam(v)), "mean_win": ev["mean_win_rate"],
                             "default_net": pct(float(np.mean(v["default_net"])), *_fam(v)),
                             "climber_mean_net": pct(float(np.mean([x for d in v["climber"].values() for x in d.values()])), *_fam(v))})
        cats = sorted({c for r in rows for c in r["by"]})
        ho[family] = {"split": arena_task(family, jobs[0]["job"]["task_seed"]).split, "individuals": len(rows),
                      "mean_net": {a: ci([r["mean_net"] for r in rows if r["arch"] == a]) for a in ("spsa", "pg")},
                      "mean_win_rate": {a: ci([r["mean_win"] for r in rows if r["arch"] == a]) for a in ("spsa", "pg")},
                      "net_if_everyone_holds_defaults": ci([r["default_net"] for r in rows]),
                      "scripted_climber_in_the_same_seat_mean_net": ci([r["climber_mean_net"] for r in rows]),
                      "by_opponent": {c: {"net": ci([r["by"][c]["net"] for r in rows if c in r["by"]]),
                                          "opponent_net": ci([r["by"][c]["opp_net"] for r in rows if c in r["by"]]),
                                          "win_rate": ci([r["by"][c]["win"] for r in rows if c in r["by"]])} for c in cats},
                      "heldout_never_in_training_records": all(v["training_side_records_with_heldout"] == 0 for v in jobs),
                      "logged_evaluations": int(sum(len(v["log"]) for v in jobs)), "rows": rows}
    out["heldout"] = ho
    # ---- population and historical modes
    pop = _jobs(res, "population_")
    if pop:
        out["population_mode"] = [{"task": v["job"]["task_seed"], "members": v["members"], "arch": v["arch"],
                                   "matchmaking": v["matchmaking"], "ratings": v["ratings"],
                                   "final_round_robin_net_pct": pct(v["head_to_head"]["net"], *_fam(v)),
                                   "final_round_robin_win": v["head_to_head"]["win"], "cycles": v["head_to_head"]["cycles"],
                                   "mean_net_last5_pct": {m: pct(float(np.mean([r["net"][m] for r in v["curve"][-5:] if m in r["net"]])), *_fam(v))
                                                          for m in v["members"]}} for v in pop]
    hm = _jobs(res, "historical_")
    if hm:
        pairs = {(v["job"]["task_seed"], v["job"]["algos"][0]): v for v in _jobs(res, "pair_", "abm_duopoly")
                 if v["job"]["pair_index"] in (0, 2)}
        out["historical_mode"] = [{"task": v["job"]["task_seed"], "algo": v["job"]["algos"][0], "sampling": v["sampling"],
                                   "p_archive": v["p_archive"], "final_net_pct": pct(v["final"]["net_per_tick"], *_fam(v)),
                                   "final_view": v["final_view"],
                                   "pair_trained_final_net_pct": (pct(pairs[(v["job"]["task_seed"], v["job"]["algos"][0])]["final"]["net_per_tick"], *_fam(v))
                                                                  if (v["job"]["task_seed"], v["job"]["algos"][0]) in pairs else None),
                                   "history": {k: v["history"][k] for k in ("stages", "elo", "later_beats_earlier", "last_beats_all_earlier", "cycles", "win")},
                                   "share_of_training_matches_against_archive": float(np.mean(["@" in o for r in v["curve"] for o in r["opponents"].values()]))}
                                  for v in hm]
    reg = _jobs(res, "registry_")
    if reg:
        out["registry_learners"] = [
            {"family": v["job"]["family"], "task": v["job"]["task_seed"], "available": v["available"],
             "registry": v.get("registry"), "default_joint_pct": pct(v["defaults"]["joint_net_per_tick"], *_fam(v)) if v["available"] else None,
             "trained_in_the_arena": {k: {"final_joint_pct": pct(x["final"]["joint_net_per_tick"], *_fam(v)),
                                          "final_net_pct": pct(x["final"]["net_per_tick"], *_fam(v)),
                                          "curve_eval_joint_pct": [[r["it"], pct(r["eval_joint"], *_fam(v))] for r in x["curve"]],
                                          "optimiser_steps": x["updates"],
                                          "reloaded_checkpoint_plays_identically": x["reloaded_checkpoint_plays_identically"]}
                                      for k, x in v.get("trained", {}).items()},
             "finals_against_registry_scripted_learners": {k: {"arch": x["arch"], "net_pct": pct(x["net"], *_fam(v)),
                                                               "opponent_net_pct": pct(x["opponent_net"], *_fam(v))}
                                                           for k, x in v.get("vs_scripted", {}).items()}} for v in reg]
    au = _jobs(res, "audit_", "abm_duopoly_fx")
    if au:
        out["audit_test_family_abm_duopoly_fx"] = _audit_summary(au)
    # ---- final nets of the test-split market
    fx = _jobs(res, "pair_", "abm_duopoly_fx")
    if fx:
        out["test_family_training_abm_duopoly_fx"] = {
            "note": "same learners, same options as on every other family; nothing was tuned on this family",
            "final_joint": {a: ci([pct(v["final"]["joint_net_per_tick"], *_fam(v)) for v in fx if v["job"]["algos"][0] == a]) for a in ("spsa", "pg")},
            "default_joint": ci([pct(v["defaults"]["joint_net_per_tick"], *_fam(v)) for v in fx]),
            "curves_eval_joint": [{"task": v["job"]["task_seed"], "algos": v["job"]["algos"],
                                   "curve": [[r["it"], pct(r["eval_joint"], *_fam(v))] for r in v["curve"] if "eval_joint" in r]} for v in fx]}
    return out


# ====================================================================== everything
def export_checkpoints(res: dict, cfg: dict, dest: str) -> list[str]:
    """Copy the final individuals of the first world of every family (pair 0: spsa, pair 2: pg; frozen, weights
    only) to `dest/<family>_<seed>/<individual>/` (`meta.json` + `learner.ckpt`, loadable with `Individual.load`)."""
    from .experiments import SPEC
    from .league import SelfPlayLeague
    out = []
    for family, seeds in cfg["tasks"].items():
        for v in _jobs(res, "pair_", family):
            j = v["job"]
            native = "joint" if family == "tech_handoff" else "own"
            if j["task_seed"] != seeds[0] or j["pair_index"] not in (0, 2) or j["objective"] != native:
                continue
            if not os.path.isdir(v["root"]):
                continue
            lg = SelfPlayLeague.load(v["root"], SPEC)
            out += lg.export_checkpoints(v["ids"], os.path.join(dest, f"{family}_{j['task_seed']}"))
    return out


def build_report(res: dict, cfg: dict, out_dir: str, work: str, tag: str = "") -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    files = []
    parts = {}
    for name, fn in (("A", report_A), ("B", report_B), ("C", report_C), ("D", report_D)):
        try:
            parts[name] = fn(res, cfg)
        except Exception as e:                          # noqa: BLE001 - a partial run still yields the other parts
            import traceback
            parts[name] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
        path = os.path.join(out_dir, f"ow_selfplay_{name}{tag}.json")
        with open(path, "w") as fh:
            json.dump(parts[name], fh, indent=1, default=_json)
        files.append(path)
    ker = _jobs(res, "kernel_")
    cons = [v["conservation"] for v in _jobs(res, "pair_") if "conservation" in v]
    summary = {"jobs_finished": dict(Counter(k.split("_")[0] for k in res)), "n_slots": cfg["n_slots"], "iters": cfg["iters"],
               "training_wall_s": float(sum(v.get("wall_s", 0.0) for v in res.values())),
               "conservation": {"matches_checked": int(sum(c["matches"] for c in cons)),
                                "violations": int(sum(len(c["violations"]) for c in cons)),
                                "max_seat_revenue_over_demand_income": max([c["max_revenue_over_demand_income"] for c in cons], default=None)},
               "kernel_hosted_replays_of_trained_pairs": {
                   "what": "the frozen final pair of the first world of each family (one spsa pair, one pg pair) played "
                           "once more in a kernel world through rig.set / rig.read; `equal` = the arena's net, the sum of "
                           "the kernel reports and the metric rig_net agree to the cent for both seats",
                   "matches": len(ker), "equal": int(sum(bool(v["equal"]) for v in ker)),
                   "logs_replay": int(sum(bool(v["log_replays"]) for v in ker)),
                   "ledger_conservation_ok": int(sum(bool(v["ledger"]["ok"]) for v in ker)),
                   "invariant_violations": int(sum(len(v["invariants"]) for v in ker)),
                   "tool_calls": int(sum(sum(v["tool_calls"]) for v in ker)),
                   "runs": [{"family": v["job"]["family"], "arch": v["arch"], "equal": v["equal"],
                             "rig_net_cents": v["rig_net"], "arena_net_cents": v["arena_net_cents"]} for v in ker]},
               "errors_file": os.path.join(work, "errors.log") if os.path.exists(os.path.join(work, "errors.log")) else None}
    path = os.path.join(out_dir, f"ow_selfplay_summary{tag}.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1, default=_json)
    files.append(path)
    return files


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)
