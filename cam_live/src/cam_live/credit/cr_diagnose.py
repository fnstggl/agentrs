"""cam_live.credit.cr_diagnose — why the learned credit estimators fail their held-out selection (CR, amendment item 4).

    python -m cam_live.credit.cr_diagnose estimand --out OUT.json [--n-train-per 48] [--seed 0] [--work DIR]
    python -m cam_live.credit.cr_diagnose power --out OUT.json

D's lab (``diagnose.exp_credit``): six randomised experiments, one hidden LabWorld mechanism each (key-door world:
``lab.unlock`` has no immediate effect and multiplies the later yield of the last channel), 48 lifetimes each, the
first decision randomised between ``unlock`` and ``wait``; RUDDER and COCOA trained on independent training lifetimes
of the same six mechanisms. The selector compares, per experiment, the estimator's arm contrast with the measured one
(paired z, gate mean z^2 <= 3).

``estimand`` separates the three candidate explanations of the failure by scoring *reference* estimators through the
same gate, on the same held-out units:

    truth        the mechanism's true effect (``n_truth`` forced-arm lifetimes per arm and mechanism, fresh noise)
    train_ideal  the effect an ideal learner can know from the actual training data: the measured arm difference of
                 the training lifetimes of the mechanism (no pooling across mechanisms)
    marginal     one effect for every mechanism (mean of ``truth``): what an estimator that does not resolve the
                 mechanism from the decision's information state can at best learn

If ``truth`` passes and ``train_ideal`` fails, the gate cannot be passed with training data of the size of the
experiments (data scarcity / gate design); if ``marginal`` fails, an estimator must resolve the mechanism; the
estimators' slope against ``truth`` (1 = resolves the mechanisms, 0 = constant) and their pooled bias show which of
these they are. ``power`` measures the gate's pass rate for exactly calibrated estimators at small unit counts.
Everything is computed here (debug scale: tiny random backbone, debug core, CPU)."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..contracts import Modality, SimClock
from ..diagnostics.world import LabWorld
from ..learning_api import UpdatePlan
from .diagnose import BETA, _cfg, _fresh, code_provenance
from .labkit import (Arms, build_lab, collect_lifetime, door_candidates, door_config, door_weight, mk_action,
                     run_door_experiments, run_door_training, weighted_policy)
from .learner import LearnerSettings
from .links import OutcomeLinker, record_arm
from .providers import COCOA, KEY_CONTRIB, RUDDER, trajectory_return
from .replay import SqliteExperienceStore
from .select import (PER_DECISION, EstimatorSelector, UnitContrast, chi2_ppf, paired_comparison, pooled_unit_score)

T, C = "unlock", "wait"


def _arm_action(arm: str):
    return (lambda: mk_action("lab.unlock")) if arm == T else (lambda: mk_action("wait", Modality.INTERNAL, seconds=3600.0))


def truth_effects(work: str, mech_seeds: Sequence[int], *, n_truth: int = 200, n_decisions: int = 5,
                  seed: int = 0) -> Dict[int, Dict[str, float]]:
    """True arm effect per mechanism: ``n_truth`` lifetimes per arm with the first decision forced and the later
    decisions from D's behaviour policy, noise seeds disjoint from training and experiment lifetimes."""
    store = SqliteExperienceStore(_fresh(work, "truth.sqlite"), durable=False, seed=seed)
    clock = SimClock()
    linker = OutcomeLinker(store, clock=clock, retro_predecessors=64)
    out: Dict[int, Dict[str, float]] = {}
    try:
        for e, ms in enumerate(mech_seeds):
            g: Dict[str, List[float]] = {T: [], C: []}
            for arm in (T, C):
                for i in range(n_truth):
                    s = 9_000_000 + 100_000 * (seed + 1) + 10_000 * e + (0 if arm == T else 5_000) + i
                    world = LabWorld(door_config(seed=ms), clock)
                    world.rng = random.Random(s)
                    lt = collect_lifetime(store, linker, world, weighted_policy(door_weight, seed=s), n_decisions,
                                          seed=s, candidates_fn=door_candidates,
                                          arms=Arms("", {arm: _arm_action(arm)}, {arm: 1.0}),
                                          tag=f"truth_{seed}_{e}_{arm}_{i}")
                    g[arm].append(trajectory_return(store, store.trajectory(lt.task.objective_id), beta=BETA).G)
            mt, mc = sum(g[T]) / len(g[T]), sum(g[C]) / len(g[C])
            vt = sum((x - mt) ** 2 for x in g[T]) / (len(g[T]) - 1)
            vc = sum((x - mc) ** 2 for x in g[C]) / (len(g[C]) - 1)
            out[ms] = {"effect": mt - mc, "se": math.sqrt(vt / len(g[T]) + vc / len(g[C])), "mean_unlock": mt,
                       "mean_wait": mc, "sd_unlock": math.sqrt(vt), "sd_wait": math.sqrt(vc), "n_per_arm": n_truth}
    finally:
        store.close()
    return out


def _effect(values: Dict[str, List[float]]) -> Optional[Dict[str, float]]:
    a, b = values.get(T, []), values.get(C, [])
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    return {"effect": ma - mb, "se": math.sqrt(va / len(a) + vb / len(b)), "n": [len(a), len(b)]}


def gate_scores(per_exp: Dict[str, List[Tuple[str, float, float]]], *, max_mean_sq_z: float = 3.0,
                min_experiments: int = 5) -> Dict[str, Any]:
    """The selector's per-experiment rule on per-unit (arm, estimate, measured) rows: paired z per experiment,
    mean z^2 and the gate verdict (as ``EstimatorSelector.score``)."""
    z2, errs = {}, {}
    for xid, units in per_exp.items():
        pc = paired_comparison(units, T, C)
        if pc is None or not pc[1] > 0:
            continue
        z2[xid] = (pc[0] / pc[1]) ** 2
        errs[xid] = pc[0]
    m = sum(z2.values()) / len(z2) if z2 else float("nan")
    k = len(z2)
    return {"z2": z2, "error": errs, "mean_sq_z": m, "n": k,
            "passes_old_gate": bool(k >= min_experiments and m <= max_mean_sq_z),
            "chi2_sum": sum(z2.values()), "chi2_99": chi2_ppf(0.99, k) if k else float("nan"),
            "passes_chi2_99": bool(k and sum(z2.values()) <= chi2_ppf(0.99, k))}


def _slope(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    syy = sum((b - my) ** 2 for b in y)
    return {"slope": sxy / sxx if sxx > 0 else float("nan"), "intercept": my - (sxy / sxx) * mx if sxx > 0 else float("nan"),
            "corr": sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan"), "mean_x": mx, "mean_y": my}


def exp_estimand(work: str, *, seed: int = 0, n_experiments: int = 6, n_per: int = 48, n_train_per: int = 48,
                 n_decisions: int = 5, steps: int = 400, n_truth: int = 200,
                 truth: Optional[Dict[int, Dict[str, float]]] = None) -> Dict[str, Any]:
    t_start = time.perf_counter()
    cfg = _cfg(lr_heads=2e-3, seed=seed)
    lab = build_lab(_fresh(work, f"cr_estimand_{seed}_{n_train_per}.sqlite"), cfg, world=False, policy=False,
                    credit=True, beta=BETA, settings=LearnerSettings.from_cfg(cfg, batch_size=32, val_fraction=0.15),
                    selector=EstimatorSelector(min_experiments=5))
    try:
        learner, store, assigner = lab.learner, lab.store, lab.assigner
        lab.rudder.max_trajectories = lab.cocoa.max_trajectories = 32
        train_runs = run_door_training(lab, n_experiments, n_train_per, n_decisions, seed=seed)
        runs = run_door_experiments(lab, n_experiments, n_per, n_decisions, seed=seed)
        learner.refresh_eligibility(full=True)
        learner.learn(learner.poll() or UpdatePlan(trigger="replay"))
        while learner.counters["frequent"] < steps:
            learner.learn(UpdatePlan(trigger="replay", max_steps=8, budget_s=1e9))
        assigner.assign(reason="estimators_trained")
        mech = {f"door-{seed}-{e}": 1000 * (seed + 1) + e for e in range(n_experiments)}
        if truth is None:
            truth = truth_effects(work, list(mech.values()), n_truth=n_truth, n_decisions=n_decisions, seed=seed)
        truth = {int(k): v for k, v in truth.items()}
        # ideal learner on the actual training data (per mechanism, no pooling)
        train_ideal: Dict[str, Dict[str, float]] = {}
        for e in range(n_experiments):
            g: Dict[str, List[float]] = {T: [], C: []}
            for lt in train_runs[f"door-{seed}-{e}/train"]:
                if lt.arm in g:
                    g[lt.arm].append(trajectory_return(store, store.trajectory(lt.task.objective_id), beta=BETA).G)
            train_ideal[f"door-{seed}-{e}"] = _effect(g) or {"effect": float("nan"), "se": float("nan")}
        marginal = sum(truth[m]["effect"] for m in mech.values()) / len(mech)
        # held-out units (the same filters as the selector) and the per-experiment evidence of the selector
        units: Dict[str, List[UnitContrast]] = {}
        counts: Dict[str, Dict[str, int]] = {}
        evid: Dict[str, Any] = {}
        for xid in runs:
            units[xid], counts[xid] = assigner.experiment_units(xid, treatment=T, control=C)
            ev = assigner.compare_with_experiment(xid, treatment=T, control=C)
            evid[xid] = None if ev is None else {"measured": ev.diff, "se": ev.se, "n": [ev.n_treatment, ev.n_control],
                                                 "estimates": ev.estimates, "paired_diff": ev.paired_diff,
                                                 "paired_se": ev.paired_se}
        refs = {"truth": {x: truth[mech[x]]["effect"] for x in runs},
                "train_ideal": {x: train_ideal[x]["effect"] for x in runs},
                "marginal": {x: marginal for x in runs},
                "no_effect": {x: 0.0 for x in runs}}
        all_units: List[UnitContrast] = []
        for xid, us in units.items():
            for u in us:
                vals = dict(u.values)
                kinds = dict(u.kinds)
                for name, by_x in refs.items():
                    vals[name] = by_x[xid] if u.arm == T else 0.0
                    kinds[name] = PER_DECISION
                all_units.append(UnitContrast(u.experiment_id, u.exp_id, u.unit, u.treatment, u.control, u.arm,
                                              u.p_treatment, u.y, vals, kinds))
        names = [RUDDER, COCOA] + list(refs)
        # per-experiment gate (old rule) for every estimator, on the same units
        old: Dict[str, Any] = {}
        for name in names:
            per_exp: Dict[str, List[Tuple[str, float, float]]] = {}
            for u in all_units:
                if name not in u.values:
                    continue
                if name == COCOA:          # per-unit contrast: paired with its arm mean (selector's UNIT_MEAN kind)
                    continue
                per_exp.setdefault(u.experiment_id, []).append((u.arm, u.values[name], u.y))
            if per_exp:
                old[name] = gate_scores(per_exp)
        # COCOA as the selector scores it (its evidence from compare_with_experiment, UNIT_MEAN kind)
        cz = {x: (e["paired_diff"][COCOA] / e["paired_se"][COCOA]) ** 2 for x, e in evid.items()
              if e and COCOA in e.get("paired_diff", {}) and e["paired_se"].get(COCOA, 0) > 0}
        if cz:
            m = sum(cz.values()) / len(cz)
            old[COCOA] = {"z2": cz, "mean_sq_z": m, "n": len(cz), "passes_old_gate": bool(len(cz) >= 5 and m <= 3.0),
                          "chi2_sum": sum(cz.values()), "chi2_99": chi2_ppf(0.99, len(cz)),
                          "passes_chi2_99": bool(sum(cz.values()) <= chi2_ppf(0.99, len(cz))),
                          "error": {x: evid[x]["paired_diff"][COCOA] for x in cz}}
        pooled = {name: pooled_unit_score(all_units, name).to_dict() for name in names}
        # what the estimators resolve: their per-experiment estimates against the truth
        est = {RUDDER: {x: (evid[x] or {}).get("estimates", {}).get(RUDDER) for x in runs},
               COCOA: {x: (evid[x] or {}).get("estimates", {}).get(COCOA) for x in runs}}
        resolve = {}
        for name, by_x in est.items():
            xs = [x for x in runs if by_x.get(x) is not None]
            if len(xs) >= 3:
                resolve[name] = _slope([refs["truth"][x] for x in xs], [by_x[x] for x in xs])
        resolve["train_ideal"] = _slope([refs["truth"][x] for x in runs], [refs["train_ideal"][x] for x in runs])
        # information state at the randomised decision: first-decision latents per mechanism and arm (between
        # mechanisms vs within a mechanism, same arm) and RUDDER's prediction after the first decision per arm
        lat: Dict[Tuple[str, str], List[torch.Tensor]] = {}
        f1: Dict[Tuple[str, str], List[float]] = {}
        for xid in runs:
            for r in store.experiment_records(xid):
                enc = lab.rudder.encoder.encode(r.objective_id, full=True)
                if enc is None:
                    continue
                arm = record_arm(r) or ""
                lat.setdefault((xid, arm), []).append(torch.cat([enc.z[0], enc.b[0]]).detach().double())
                if KEY_CONTRIB in r.credit:
                    f1.setdefault((xid, arm), []).append(float(r.credit[KEY_CONTRIB]))
        latent: Dict[str, Any] = {}
        for arm in (T, C):
            groups = {x: torch.stack(v) for (x, a_), v in lat.items() if a_ == arm and v}
            if len(groups) < 2:
                continue
            means = {x: v.mean(0) for x, v in groups.items()}
            within = float(sum(((v - means[x]) ** 2).sum(1).mean() for x, v in groups.items()) / len(groups))
            allm = torch.stack(list(means.values()))
            between = float(((allm - allm.mean(0)) ** 2).sum(1).mean())
            latent[arm] = {"between_mechanism_mean_sq": between, "within_mechanism_mean_sq": within,
                           "ratio": between / within if within > 0 else float("inf"),
                           "rudder_F_after_first": {x: sum(v) / len(v) for (x, a_), v in f1.items() if a_ == arm}}
        # heterogeneity of the measured effects (Cochran's Q, I^2)
        meas = [(evid[x]["measured"], evid[x]["se"]) for x in runs if evid[x]]
        w = [1.0 / s ** 2 for _, s in meas]
        mu = sum(wi * d for wi, (d, _) in zip(w, meas)) / sum(w)
        Q = sum(wi * (d - mu) ** 2 for wi, (d, _) in zip(w, meas))
        het = {"Q": Q, "df": len(meas) - 1, "I2": max(0.0, (Q - (len(meas) - 1)) / Q) if Q > 0 else 0.0,
               "fixed_effect_mean": mu, "chi2_99": chi2_ppf(0.99, len(meas) - 1)}
        truth_het_sd = math.sqrt(sum((refs["truth"][x] - marginal) ** 2 for x in runs) / (len(runs) - 1))
        return {
            "config": {"seed": seed, "n_experiments": n_experiments, "lifetimes_per_experiment": n_per,
                       "training_lifetimes_per_mechanism": n_train_per, "decisions": n_decisions, "updates": steps,
                       "n_truth_per_arm": n_truth, "beta_per_second": BETA},
            "truth": {x: truth[mech[x]] for x in runs}, "marginal_effect": marginal,
            "truth_between_mechanism_sd": truth_het_sd,
            "train_ideal": train_ideal, "evidence": evid, "unit_counts": counts,
            "old_gate": old, "pooled": pooled, "resolve_vs_truth": resolve,
            "first_decision_latent": latent,
            "measured_heterogeneity": het, "_seconds": time.perf_counter() - t_start}
    finally:
        lab.close()


def exp_power(*, reps: int = 2000, n_experiments: int = 6, seed: int = 0,
              effects: Sequence[float] = (3.3, 6.0, 2.8, -0.5, 6.0, 11.8), sd_y: float = 5.5) -> Dict[str, Any]:
    """Pass rate of the per-experiment gate (mean z^2 <= 3 over >= 5 experiments, paired SE) and of the pooled rule
    for estimators that are exactly calibrated (``truth``) or ideal on training data of n_train units per arm
    (``train_ideal``), at n units per arm and experiment. Gaussian outcomes with the lab's spread (``sd_y``) and the
    lab's six measured effects; no model involved."""
    rng = random.Random(seed)
    out: Dict[str, Any] = {"effects": list(effects), "sd_y": sd_y, "reps": reps, "rows": []}
    for n in (2, 3, 5, 10, 24):
        for kind, n_train in (("truth", None), ("train_ideal_same_size", n), ("train_ideal_4x", 4 * n)):
            old_pass = chi_pass = pooled_pass = 0
            for _ in range(reps):
                per_exp, units = {}, []
                for k, eff in enumerate(effects[:n_experiments]):
                    est = eff if n_train is None else eff + rng.gauss(0.0, sd_y * math.sqrt(2.0 / n_train))
                    rows = []
                    for arm, mu in ((T, eff), (C, 0.0)):
                        for _ in range(n):
                            y = mu + rng.gauss(0.0, sd_y)
                            rows.append((arm, est if arm == T else 0.0, y))
                            units.append(UnitContrast(f"x{k}", f"u{len(units)}", "", T, C, arm, 0.5, y,
                                                      {"e": est if arm == T else 0.0}, {"e": PER_DECISION}))
                    per_exp[f"x{k}"] = rows
                g = gate_scores(per_exp)
                old_pass += g["passes_old_gate"]
                chi_pass += g["passes_chi2_99"]
                pooled_pass += pooled_unit_score(units, "e").passes
            out["rows"].append({"units_per_arm": n, "estimator": kind, "n_train_per_arm": n_train,
                                "pass_old_gate": old_pass / reps, "pass_chi2_99": chi_pass / reps,
                                "pass_pooled": pooled_pass / reps})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["estimand", "power"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train-per", type=int, default=48)
    ap.add_argument("--n-truth", type=int, default=200)
    ap.add_argument("--truth", default="", help="JSON file with truth effects from an earlier run (reused)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    if a.what == "power":
        res = exp_power(reps=a.reps, seed=a.seed)
    else:
        work = a.work or tempfile.mkdtemp(prefix="camlive_CR_")
        os.makedirs(work, exist_ok=True)
        truth = None
        if a.truth and os.path.exists(a.truth):
            tj = json.load(open(a.truth))
            truth = {int(v["_mech_seed"]): v for v in tj["truth"].values()} if "truth" in tj else None
        res = exp_estimand(work, seed=a.seed, n_train_per=a.n_train_per, n_truth=a.n_truth, steps=a.steps, truth=truth)
        mech = {f"door-{a.seed}-{e}": 1000 * (a.seed + 1) + e for e in range(6)}
        for x, v in res["truth"].items():
            v["_mech_seed"] = mech[x]
    res["_command"] = "cd /workspace/camlive && .venv/bin/python -m cam_live.credit.cr_diagnose " + " ".join(
        argv if argv is not None else __import__("sys").argv[1:])
    res["_code"] = code_provenance()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True, default=str)
    print(json.dumps({k: res[k] for k in res if k in ("old_gate", "pooled", "resolve_vs_truth", "rows",
                                                       "first_decision_latent", "measured_heterogeneity")},
                     indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
