"""cam_live.predict.diagnose — learning evidence for Workstream B at debug scale in LabWorld.

    python -m cam_live.predict.diagnose [--out docs/evidence/B/diagnostics.json] [--quick] [--only 1a,1b,...]

CPU only: ``CamLiveConfig.debug()`` (tiny *random* backbone, debug core) with a 4096-bucket hash vocabulary. The
numbers say whether the mathematics and the wiring learn from honestly collected experience in a controlled world
with known ground truth. They are ``provenance="diagnostic"``: not evidence about the real backbone, a live task or
any real environment. Every experiment reports what it measured, including where the hypothesis failed.

    1a  world model vs action-independent baseline, held-out lifetimes of one mechanism, two horizons
    1b  the same on held-out *new* mechanisms (generalisation through the context)
    2   fast belief updates in new worlds: prediction error over the first K observations, no gradient step
    3   evaluator calibration on prospectively registered success forecasts, before and after training
    4   value of information: probe vs noisy gauge; VOI-aware vs greedy selection in closed loop
    5   action quality vs raw outcome on lucky / unlucky outcomes
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..contracts import (ExperienceRecord, GoalStatus, OutcomeKind, OutcomeObservation, OutcomeQuery,
                         OutcomeStatus, RunFiles)
from ..diagnostics.collect import (LabCollector, Lifetime, action_item, lab_candidates, obs_item, outcome_item,
                                   tool_item, uniform_policy)
from ..diagnostics.memstore import MemoryStore
from ..diagnostics.world import AllowAllPolicy, LabVerifier, LabWorld, lab_task
from ..contracts import ExperienceItem, ItemKind, Trust
from . import labdata as L
from .calibration import binary_report, studentised_ci
from .evaluator import action_quality, outcome_decomposition
from .policy import DecisionMaker
from .predictor import Predictor, at_horizons
from .providers import EvaluatorLoss, FastPriorLoss, WorldModelLoss
from .registry import PredictionRegistry
from .voi import spearman

DAY = L.DAY
COUNT_TEMPLATE = OutcomeQuery("", "lab.adopters.from.{action}", OutcomeKind.COUNT,
                              description="adopters attributed to this intervention within the attribution window",
                              goal_relevant=True)


def act_queries(horizons: Sequence[float] = (L.SHORT_HORIZON_S, 2 * DAY)) -> Callable[[Any], List[OutcomeQuery]]:
    tmpl = at_horizons(COUNT_TEMPLATE, horizons)

    def queries(a: Any) -> List[OutcomeQuery]:
        if a.operation != "lab.act":
            return []
        return [dataclasses.replace(t, exposure=float(a.arguments["effort"])) for t in tmpl]
    return queries


def _ci(values: Sequence[float]) -> Dict[str, float]:
    v = [float(x) for x in values if x == x]
    if len(v) < 2:
        return {"n": len(v), "mean": float(v[0]) if v else float("nan")}
    r = studentised_ci(v)
    return {k: float(r[k]) for k in ("mean", "lo", "hi", "n") if k in r}


def _registry(clock: Any, tag: str) -> PredictionRegistry:
    d = tempfile.mkdtemp(prefix=f"camlive_b_{tag}_")
    return PredictionRegistry(os.path.join(d, RunFiles.PREDICTIONS), clock)


# ----------------------------------------------------------------------------------------------------------------
# 1. world model against an action-independent baseline
# ----------------------------------------------------------------------------------------------------------------
def score_counts(model: Any, heads: Any, store: Any, train: Sequence[ExperienceRecord],
                 test: Sequence[Tuple[Lifetime, ExperienceRecord]]) -> Dict[str, Any]:
    """Held-out NLL of action-attributed counts per horizon: the head ensemble's predictive mixture, the
    maximum-likelihood action-independent negative binomial fitted on the training outcomes of that horizon, and
    the ground-truth Poisson (oracle)."""
    a = L.count_targets(model, heads, store, train)
    b = L.count_targets(model, heads, store, [r for _, r in test])
    out: Dict[str, Any] = {}
    for name, h in (("6h", L.SHORT_HORIZON_S), ("2d", 2 * DAY)):
        ma, mb = a["horizon_s"] == h, b["horizon_s"] == h
        lr_, lc_ = L.fit_marginal_nb(a["y"][ma], a["exposure"][ma])
        base = L.marginal_nll(b["y"][mb], b["exposure"][mb], lr_, lc_)
        oracle = torch.tensor([L.oracle_count_mean(test[int(i)][0], test[int(i)][1], h) for i in b["record"][mb]],
                              dtype=torch.float64).clamp_min(1e-9)
        y = b["y"][mb]
        onll = -(y * torch.log(oracle) - oracle - torch.lgamma(y + 1.0))
        worlds = [test[int(i)][0].task.objective_id for i in b["record"][mb]]
        out[name] = {"n_heldout": int(mb.sum()), "model_nll": float(b["nll"][mb].mean()),
                     "marginal_nll": float(base.mean()), "oracle_nll": float(onll.mean()),
                     "gain_vs_marginal": _cluster_ci((base - b["nll"][mb]).tolist(), worlds),
                     "mean_count": float(y.mean()),
                     "model_mean_vs_truth_spearman": spearman(b["mean"][mb], oracle),
                     "train_model_nll": float(a["nll"][ma].mean())}
    return out


def _verdict(after: Dict[str, Any]) -> Dict[str, bool]:
    gains = [after[h]["gain_vs_marginal"] for h in ("6h", "2d")]
    return {"beats_marginal_both_horizons": bool(all(g.get("lo", -1.0) > 0 for g in gains)),
            "beats_marginal_mean_both_horizons": bool(all(g["mean"] > 0 for g in gains))}


def world_vs_marginal(*, mechanism_seed: Optional[int], n_train: int, n_test: int, n_decisions: int, steps: int,
                      seed: int = 0, lr: float = 2e-3, log_every: int = 0) -> Dict[str, Any]:
    """Experiment 1a: train on lifetimes of one mechanism, score held-out lifetimes of the same mechanism
    (different behaviour sequences, different sampled consequences)."""
    torch.manual_seed(seed)
    model, heads = L.build(seed)
    store = MemoryStore()
    kw = dict(mechanism_seed=mechanism_seed, immediate=False)          # this experiment is about counts only
    train_l = L.collect(store, range(n_train), n_decisions, **kw)
    test_l = L.collect(store, range(10_000, 10_000 + n_test), n_decisions, **kw)
    wm, fp = WorldModelLoss(heads), FastPriorLoss(heads)
    tr = [r for lt in train_l for r in lt.records]
    te = [(lt, r) for lt in test_l for r in lt.records]
    before = score_counts(model, heads, store, tr, te)
    t0 = time.time()
    hist = L.train(model, [wm, fp], store, L.lifetime_sampler(train_l, 3, 8, seed=seed, eligible=wm.eligible), steps,
                   lr=lr, log_every=log_every)
    after = score_counts(model, heads, store, tr, te)
    return {"setting": "same_mechanism" if mechanism_seed is not None else "new_mechanisms",
            "mechanism_seed": mechanism_seed, "n_train_lifetimes": n_train, "n_heldout_lifetimes": n_test,
            "steps": steps, "train_seconds": round(time.time() - t0, 1), "before_training": before,
            "after_training": after,
            "final_train_loss": float(np.mean([h["loss"] for h in hist[-20:]])) if hist else None, **_verdict(after)}


def family_world_vs_marginal(fam: Dict[str, Any], seeds: Sequence[int]) -> Dict[str, Any]:
    """Experiment 1b: the family model (trained on many mechanisms) on held-out lifetimes of *new* mechanisms.
    Beating the action-independent baseline here needs the action's effect to be read from the context."""
    store = fam["store"]
    test_l = L.collect(store, seeds, fam["n_decisions"], base=fam["base"])
    tr = [r for lt in fam["train_lifetimes"] for r in lt.records]
    te = [(lt, r) for lt in test_l for r in lt.records]
    after = score_counts(fam["model"], fam["heads"], store, tr, te)
    return {"setting": "new_mechanisms", "n_train_lifetimes": len(fam["train_lifetimes"]),
            "n_heldout_lifetimes": len(test_l), "steps": fam["info"]["steps"],
            "before_training": score_counts(fam["twin"], fam["twin_heads"], store, tr, te),
            "after_training": after, **_verdict(after)}


# ----------------------------------------------------------------------------------------------------------------
# family model: trained on many mechanisms; used by experiments 2-5
# ----------------------------------------------------------------------------------------------------------------
def family_config(target: int = 22, **over: Any) -> Any:
    return L.lab_config(target=target, **over)


def train_family(*, n_train: int, n_decisions: int, steps: int, seed: int = 0, lr: float = 2e-3,
                 log_every: int = 0, target: int = 22, n_eval_lifetimes: int = 0) -> Dict[str, Any]:
    """A model trained with all three frequent providers on lifetimes from many different mechanisms, and an
    identically initialised untrained twin for before / after comparisons.

    The world model and the fast prior train on batches of many records from few lifetimes (the fast prior needs
    several outcomes per world). One objective outcome labels a whole lifetime, so the evaluator additionally
    trains on batches of few records from *many* lifetimes (``n_eval_lifetimes`` further lifetimes are collected
    for it): with only tens of independent labels it memorised lifetimes instead of generalising."""
    torch.manual_seed(seed)
    model, heads = L.build(seed)
    twin, twin_heads = L.build(seed)
    store = MemoryStore()
    base = family_config(target)
    lts = L.collect(store, range(n_train), n_decisions, base=base)
    more = L.collect(store, range(50_000, 50_000 + n_eval_lifetimes), n_decisions, base=base)
    labelled = lts + more

    def check(o: Any) -> bool:
        return any(LabVerifier(lt.world).check_signature(o) for lt in labelled)

    wm, fp, ev = WorldModelLoss(heads), FastPriorLoss(heads), EvaluatorLoss(heads, check)
    for lt in labelled:
        ev.add_outcome(lt.verified, deadline=lt.world.t0 + lt.world.cfg.horizon_s, trajectory=lt.records)
    t0 = time.time()
    hist = L.train(model, [wm, fp], store, L.lifetime_sampler(lts, 2, 14, seed=seed), steps, lr=lr,
                   log_every=log_every, extra=[([ev], L.lifetime_sampler(labelled, 12, 2, seed=seed + 1))])
    tail = hist[-30:]
    return {"model": model, "heads": heads, "twin": twin, "twin_heads": twin_heads, "store": store, "base": base,
            "train_lifetimes": lts, "n_decisions": n_decisions,
            "info": {"n_train_lifetimes": n_train, "n_evaluator_lifetimes": len(labelled), "steps": steps,
                     "train_seconds": round(time.time() - t0, 1),
                     "train_success_rate": float(np.mean([lt.verified.status == GoalStatus.ACHIEVED
                                                          for lt in labelled])),
                     "final": {k: float(np.mean([h[k] for h in tail if k in h])) for k in
                               ("world_nll_count", "fast_online_nll_count", "fast_online_nll_first_count",
                                "fast_online_nll_later_count", "evaluator_loss", "eval_brier", "transition_nmse")
                               if any(k in h for h in tail)}}}


def heldout_pass(fam: Dict[str, Any], seeds: Sequence[int]) -> Dict[str, Any]:
    """Held-out worlds (new mechanisms) under the uniform behaviour policy, with three observers that only
    predict: the trained model with fast adaptation, the same model without adaptation, and the untrained twin.
    Every forecast is registered before the action is executed and resolved only by a later real outcome."""
    model, heads, twin, twin_heads = fam["model"], fam["heads"], fam["twin"], fam["twin_heads"]
    queries = act_queries()
    rows_adapt: List[Dict[str, Any]] = []
    quality: List[Dict[str, float]] = []
    success: Dict[str, List[Tuple[float, float, str]]] = {"trained": [], "untrained": []}
    store = MemoryStore()
    for s in seeds:
        world = LabWorld(dataclasses.replace(fam["base"], seed=int(s)))
        clock, verifier = world.clock, LabVerifier(world)
        deadline = world.t0 + world.cfg.horizon_s
        regs = {k: _registry(clock, k) for k in ("adapt", "static", "untrained")}
        fast = model.new_fast_state()
        obs = {"adapt": Predictor(model, fast, regs["adapt"], clock, heads=heads),
               "static": Predictor(model, model.new_fast_state(), regs["static"], clock, heads=heads),
               "untrained": Predictor(twin, None, regs["untrained"], clock, heads=twin_heads)}
        open_q: List[Tuple[str, OutcomeQuery]] = []
        log: List[Dict[str, Any]] = []
        rng = random.Random(int(s))

        def mature() -> None:
            still = []
            for who, q in open_q:
                o = verifier.measure([q])[0]
                if o.status == OutcomeStatus.OBSERVED:
                    if who == "adapt":
                        obs["adapt"].observe_outcomes([o])            # the only place anything adapts
                    obs[who].resolve([o])
                else:
                    still.append((who, q))
            open_q[:] = still

        def policy(cands: List[Any], info: Dict[str, Any]) -> Tuple[int, List[float]]:
            mature()
            ctx = store.items(info["context_refs"])
            task = info["task"]
            kw = dict(objective_id=task.objective_id, deadline=deadline, objective_scale=float(world.cfg.target),
                      compute_voi=False)
            n_upd = fast.n_updates(OutcomeKind.COUNT)
            ev_a = obs["adapt"].evaluate_candidates(ctx, task.objective, cands, queries, **kw)
            ev_s = obs["static"].evaluate_candidates(ctx, task.objective, cands, queries, **kw)
            obs["untrained"].evaluate_candidates(ctx, task.objective, cands, None, **kw)
            idx = rng.randrange(len(cands))
            probs = [1.0 / len(cands)] * len(cands)
            for who, evs in (("adapt", ev_a), ("static", ev_s)):
                open_q.extend((who, p.query) for p in evs[idx].predictions)
                obs[who].commit(evs[idx].predictions[0].decision_id if evs[idx].predictions else "", cands[idx].action_id)
            if cands[idx].operation == "lab.act":
                q = torch.tensor([[e.extras.get("q_goal", 0.0) for e in ev_a]])
                aq = action_quality(q, torch.tensor([probs]), torch.tensor([idx]))
                log.append({"n_updates": n_upd, "idx": idx, "cands": cands,
                            "adapt": [p.prediction_id for p in ev_a[idx].predictions],
                            "static": [p.prediction_id for p in ev_s[idx].predictions],
                            "q_chosen": float(aq["q_chosen"]), "baseline": float(aq["baseline"]),
                            "quality": float(aq["quality"])})
            return idx, probs

        lt = LabCollector(store, max_context=40).run(world, policy, fam["n_decisions"], seed=int(s))
        mature()
        for d in log:                                                 # everything below uses resolved records only
            a = [regs["adapt"].get(p) for p in d["adapt"]]
            st = [regs["static"].get(p) for p in d["static"]]
            for ra, rs in zip(a, st):
                if ra.nll is None or rs.nll is None:
                    continue
                rows_adapt.append({"k": d["n_updates"], "horizon_s": ra.query.horizon_s, "nll_adapt": ra.nll,
                                   "nll_static": rs.nll, "abs_adapt": abs(ra.dist.mean - ra.resolved_value),
                                   "abs_static": abs(rs.dist.mean - rs.resolved_value), "world": int(s)})
            long_rec = max(a, key=lambda r: r.query.horizon_s)
            if long_rec.resolved_value is None:
                continue
            info = world._actions[d["cands"][d["idx"]].action_id]
            true = []
            for c in d["cands"]:
                if c.operation == "lab.act":
                    ch = c.arguments["channel"]
                    true.append(world._rate(ch, info["at"]) * c.arguments["effort"]
                                * (1.0 - math.exp(-2 * DAY / world._delay[ch])))
                else:
                    true.append(0.0)
            dec = outcome_decomposition(torch.tensor(long_rec.resolved_value), torch.tensor(d["q_chosen"]),
                                        torch.tensor(d["baseline"]))
            quality.append({"quality": d["quality"], "luck": float(dec["luck"]), "surprise": float(dec["surprise"]),
                            "y": float(long_rec.resolved_value), "true_advantage": true[d["idx"]] - float(np.mean(true)),
                            "true_mean": true[d["idx"]], "world": int(s)})
        for who in ("trained", "untrained"):
            p = obs["adapt" if who == "trained" else "untrained"]
            for r in p.resolve_objective(lt.verified, verifier.check_signature):
                success[who].append((float(r.dist.mean), float(r.resolved_value), f"w{s}"))
    return {"adaptation_rows": rows_adapt, "quality_rows": quality, "success": success}


def fast_adaptation(rows: List[Dict[str, Any]], k_edges: Sequence[int] = (1, 3, 6)) -> Dict[str, Any]:
    """Experiment 2 from the held-out pass: the adapted and the non-adapted predictor differ only by the fast
    belief state; ``k`` is the number of outcomes absorbed before the prediction was made."""
    def bucket(k: int) -> str:
        lo = 0
        for e in k_edges:
            if k < e:
                return f"k{lo}-{e - 1}" if e - 1 > lo else f"k{lo}"
            lo = e
        return f"k{lo}+"
    out: Dict[str, Any] = {"n": len(rows), "by_k": {}}
    for name in sorted({bucket(r["k"]) for r in rows}, key=lambda s: int(s[1:].split("-")[0].rstrip("+"))):
        sel = [r for r in rows if bucket(r["k"]) == name]
        gain = _cluster_ci([r["nll_static"] - r["nll_adapt"] for r in sel], [r["world"] for r in sel])
        out["by_k"][name] = {"n": len(sel), "nll_adapted": float(np.mean([r["nll_adapt"] for r in sel])),
                             "nll_no_adaptation": float(np.mean([r["nll_static"] for r in sel])),
                             "abs_err_adapted": float(np.mean([r["abs_adapt"] for r in sel])),
                             "abs_err_no_adaptation": float(np.mean([r["abs_static"] for r in sel])),
                             "nll_gain": gain}
    late = [r for r in rows if r["k"] >= k_edges[0]]
    out["after_first_observation"] = {
        "n": len(late), "nll_gain": _cluster_ci([r["nll_static"] - r["nll_adapt"] for r in late],
                                                 [r["world"] for r in late]),
        "abs_err_gain": _cluster_ci([r["abs_static"] - r["abs_adapt"] for r in late], [r["world"] for r in late])}
    g = out["after_first_observation"]["nll_gain"]
    out["adaptation_reduces_error"] = bool(g.get("lo", -1.0) > 0)
    return out


def _cluster_ci(values: Sequence[float], clusters: Sequence[Any]) -> Dict[str, float]:
    """Studentised interval over cluster (world) means: rows of one world are not independent."""
    by: Dict[Any, List[float]] = {}
    for v, c in zip(values, clusters):
        by.setdefault(c, []).append(float(v))
    out = _ci([float(np.mean(v)) for v in by.values()])
    out["n_rows"] = len(values)
    return out


def calibration(success: Dict[str, List[Tuple[float, float, str]]]) -> Dict[str, Any]:
    """Experiment 3: prospectively registered success forecasts of held-out lifetimes, scored after resolution."""
    out: Dict[str, Any] = {}
    for who, rows in success.items():
        if not rows:
            out[who] = {"n": 0}
            continue
        p, y, cl = [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
        rep = binary_report(p, y, clusters=cl)
        base = float(np.mean(y))
        out[who] = {"n": len(rows), "n_lifetimes": len(set(cl)), "base_rate": base,
                    "mean_p": rep["mean_p"], "brier": rep["brier"], "nll": rep["nll"],
                    "calibration": rep["calibration"], "ece": rep["ece"], "brier_of_base_rate": base * (1 - base)}
    if out.get("trained", {}).get("n") and out.get("untrained", {}).get("n"):
        b0, b1 = out["untrained"]["brier"], out["trained"]["brier"]
        m0 = b0["mean"] if isinstance(b0, dict) else b0
        m1 = b1["mean"] if isinstance(b1, dict) else b1
        out["brier_improved"] = bool(m1 < m0)
        out["beats_base_rate"] = bool(m1 < out["trained"]["brier_of_base_rate"])
    return out


def quality_vs_outcome(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    """Experiment 5: does the model's action quality track the true advantage of the decision, where the raw
    outcome mostly tracks luck?"""
    if len(rows) < 8:
        return {"n": len(rows)}
    q = [r["quality"] for r in rows]
    adv = [r["true_advantage"] for r in rows]
    y = [r["y"] for r in rows]
    luck_true = [r["y"] - r["true_mean"] for r in rows]               # what chance added to this outcome
    lucky_bad = [r for r in rows if r["true_advantage"] < 0 and r["y"] > r["true_mean"] + 1]
    unlucky_good = [r for r in rows if r["true_advantage"] > 0 and r["y"] < r["true_mean"] - 1]
    out = {"n": len(rows),
           "spearman_quality_vs_true_advantage": spearman(q, adv),
           "spearman_raw_outcome_vs_true_advantage": spearman(y, adv),
           "spearman_quality_vs_true_luck": spearman(q, luck_true),
           "spearman_raw_outcome_vs_true_luck": spearman(y, luck_true),
           "spearman_estimated_luck_vs_true_luck": spearman([r["luck"] for r in rows], luck_true),
           "lucky_outcomes_of_worse_than_average_decisions": {
               "n": len(lucky_bad), "fraction_quality_negative": float(np.mean([r["quality"] < 0 for r in lucky_bad]))
               if lucky_bad else None,
               "fraction_raw_outcome_above_baseline": float(np.mean([r["surprise"] > 0 for r in lucky_bad]))
               if lucky_bad else None},
           "unlucky_outcomes_of_better_than_average_decisions": {
               "n": len(unlucky_good),
               "fraction_quality_positive": float(np.mean([r["quality"] > 0 for r in unlucky_good]))
               if unlucky_good else None}}
    out["quality_less_luck_driven_than_outcome"] = bool(
        abs(out["spearman_quality_vs_true_luck"]) < abs(out["spearman_raw_outcome_vs_true_luck"]))
    out["quality_tracks_advantage"] = bool(out["spearman_quality_vs_true_advantage"] > 0)
    return out


# ----------------------------------------------------------------------------------------------------------------
# 4. value of information
# ----------------------------------------------------------------------------------------------------------------
def run_agent(fam: Dict[str, Any], seed: int, mode: str, *, max_decisions: int = 30, voi: bool = True,
              ig_weight: Optional[float] = None) -> Dict[str, Any]:
    """Closed loop in a held-out world: propose -> predict (registered) -> select -> execute -> reconcile ->
    fast belief update. Stops when the verified total reaches the target, time runs out or ``max_decisions``."""
    model, heads = fam["model"], fam["heads"]
    world = LabWorld(dataclasses.replace(fam["base"], seed=int(seed)))
    clock, verifier, auth, task = world.clock, LabVerifier(world), AllowAllPolicy(), lab_task(world)
    deadline = world.t0 + world.cfg.horizon_s
    fast = model.new_fast_state()
    pred = Predictor(model, fast, _registry(clock, f"agent_{mode}"), clock, heads=heads)
    pred.cfg = dataclasses.replace(pred.cfg, voi_samples=4, voi_max_candidates=6)
    dm = DecisionMaker(model, heads=heads, clock=clock, seed=seed)
    if ig_weight is not None:
        dm.cfg = dataclasses.replace(dm.cfg, ig_weight=ig_weight)
    queries = act_queries()
    rng = random.Random(seed)
    now = clock.now()
    ctx: List[ExperienceItem] = [ExperienceItem(kind=ItemKind.GOAL, text=task.objective, ts=now, ref="goal",
                                                trust=Trust.OPERATOR)]
    ctx += [tool_item(t, now) for t in world.discover_tools()] + [obs_item(o) for o in world.observe()]
    open_q: List[OutcomeQuery] = []
    cursor, n_act, effort, reached_at, n_dec = "", 0, 0, None, 0
    surprise = kl = 0.0

    def window() -> List[ExperienceItem]:
        return ctx if len(ctx) <= 40 else [ctx[0]] + ctx[-39:]

    def reconcile() -> None:
        nonlocal cursor, surprise, kl
        events, cursor = world.poll_events(cursor)
        ctx.extend(obs_item(e) for e in events)
        still = []
        for q in open_q:
            o = verifier.measure([q])[0]
            if o.status == OutcomeStatus.OBSERVED:
                info = pred.observe_outcomes([o])
                surprise, kl = surprise + info["surprise"], kl + info["kl"]
                pred.resolve([o])
                ctx.append(outcome_item(q, o))
            else:
                still.append(q)
        open_q[:] = still

    while n_dec < max_decisions and clock.now() < deadline:
        cands = [c for c in lab_candidates(world, rng) if not world.validate_action(c)]
        evs = pred.evaluate_candidates(window(), task.objective, cands, queries, objective_id=task.objective_id,
                                       deadline=deadline, objective_scale=float(world.cfg.target), compute_voi=voi)
        dec = dm.select(evs, mode, objective_id=task.objective_id)
        res = world.execute(dec.chosen, auth.authorize(dec.chosen, task, {}))
        n_dec += 1
        ctx.append(action_item(dec.chosen, dec.ts))
        ctx.extend(obs_item(o) for o in res.observations)
        pred.commit(dec.decision_id, dec.chosen.action_id)
        open_q.extend(p.query for p in evs[dec.chosen_index].predictions)
        if dec.chosen.operation == "lab.act":
            n_act += 1
            effort += int(dec.chosen.arguments["effort"])
        reconcile()
        if world.total() >= world.cfg.target:
            reached_at = clock.now() - world.t0
            break
    return {"seed": seed, "mode": mode, "reached": reached_at is not None, "interventions": n_act, "effort": effort,
            "decisions": n_dec, "elapsed_s": clock.now() - world.t0, "adopters": world.total(),
            "adopters_per_effort": world.total() / max(effort, 1), "fast_updates": fast.n_updates(OutcomeKind.COUNT),
            "surprise": surprise, "belief_change_kl": kl}


def voi_probe_vs_gauge(fam: Dict[str, Any], seeds: Sequence[int], samples: int = 8) -> Dict[str, Any]:
    """Computed VOI of measuring a channel (``lab.probe``), reading the static gauge (``lab.noise``) and acting, at
    the first decision of held-out worlds. The fast-layer path updates the count posterior only; the slow path
    also lets the imagined reading enter a copy of the context. ``voi_act_against_current_value_baseline`` is the
    plain E[max Q(updated)] - max Q(current) for comparison: with the approximate count update it is biased upwards."""
    model, heads = fam["model"], fam["heads"]
    counts = act_queries()

    def queries(a: Any) -> List[OutcomeQuery]:
        if a.operation == "lab.probe":
            ch = a.arguments["channel"]
            return [OutcomeQuery("", f"lab.probe.{ch}", OutcomeKind.REAL, 600.0,
                                 description=f"probe reading for {names[ch]}")]
        if a.operation == "lab.noise":
            return [OutcomeQuery("", "lab.static", OutcomeKind.REAL, 600.0, description="static gauge reading")]
        return counts(a)

    names: List[str] = []
    out: Dict[str, Any] = {}
    for path in ("fast", "slow"):
        rows = {"lab.probe": [], "lab.noise": [], "lab.act": []}
        plain_act: List[float] = []
        for s in seeds:
            world = LabWorld(dataclasses.replace(fam["base"], seed=int(s)))
            task = lab_task(world)
            names[:] = world.names
            p = Predictor(model, model.new_fast_state(), _registry(world.clock, "voi"), world.clock, heads=heads)
            p.cfg = dataclasses.replace(p.cfg, voi_slow_path=(path == "slow"), voi_samples=samples)
            ctx = [obs_item(o) for o in world.observe()]
            cands = lab_candidates(world, random.Random(int(s)))
            evs = p.evaluate_candidates(ctx, task.objective, cands, queries, generator=torch.Generator().manual_seed(int(s)))
            for op in rows:
                v = [e.voi for e in evs if e.action.operation == op and e.voi is not None]
                if v:
                    rows[op].append(float(np.mean(v)))
            plain_act.append(float(np.mean([e.extras["voi_vs_current"] for e in evs
                                            if e.action.operation == "lab.act" and "voi_vs_current" in e.extras])))
        d = [a - b for a, b in zip(rows["lab.probe"], rows["lab.noise"])]
        out[path] = {"voi_probe": _ci(rows["lab.probe"]), "voi_noisy_gauge": _ci(rows["lab.noise"]),
                     "voi_act": _ci(rows["lab.act"]), "voi_act_against_current_value_baseline": _ci(plain_act),
                     "probe_minus_gauge": _ci(d),
                     "probe_higher_than_gauge": bool(_ci(d).get("lo", -1.0) > 0)}
    return out


def voi_closed_loop(fam: Dict[str, Any], seeds: Sequence[int], max_decisions: int = 40) -> Dict[str, Any]:
    runs = {m: [run_agent(fam, int(s), m, max_decisions=max_decisions) for s in seeds] for m in ("greedy", "voi")}
    out: Dict[str, Any] = {"n_seeds": len(seeds), "max_decisions": max_decisions}
    for m, rs in runs.items():
        out[m] = {"reached_fraction": float(np.mean([r["reached"] for r in rs])),
                  "interventions": _ci([r["interventions"] for r in rs]),
                  "effort": _ci([r["effort"] for r in rs]),
                  "adopters_per_effort": _ci([r["adopters_per_effort"] for r in rs]),
                  "decisions": _ci([r["decisions"] for r in rs])}
    both = [(g, v) for g, v in zip(runs["greedy"], runs["voi"]) if g["reached"] and v["reached"]]
    out["paired_both_reached"] = {
        "n": len(both),
        "interventions_greedy_minus_voi": _ci([g["interventions"] - v["interventions"] for g, v in both]),
        "effort_greedy_minus_voi": _ci([g["effort"] - v["effort"] for g, v in both])}
    out["paired_all"] = {"adopters_per_effort_voi_minus_greedy": _ci(
        [v["adopters_per_effort"] - g["adopters_per_effort"] for g, v in zip(runs["greedy"], runs["voi"])])}
    d = out["paired_both_reached"]["interventions_greedy_minus_voi"]
    out["voi_needs_fewer_interventions"] = bool(d.get("lo", -1.0) > 0)
    out["voi_needs_fewer_interventions_on_average"] = bool(d.get("mean", -1.0) > 0)
    return out


# ----------------------------------------------------------------------------------------------------------------
def run(quick: bool = False, only: Optional[Sequence[str]] = None, log_every: int = 0) -> Dict[str, Any]:
    want = lambda k: only is None or k in only                                             # noqa: E731
    t0 = time.time()
    sz = dict(n_train=24, n_test=10, n_decisions=16, steps=250, fam_train=24, fam_steps=250, held=8, loop=4,
              max_decisions=20, eval_lifetimes=48) if quick else \
        dict(n_train=40, n_test=20, n_decisions=16, steps=1500, fam_train=60, fam_steps=1200, held=40, loop=12,
             max_decisions=30, eval_lifetimes=340)
    out: Dict[str, Any] = {
        "scale": "debug (tiny random backbone, 4096 hash buckets; debug core)", "provenance": "diagnostic",
        "quick": quick, "command": "cd /workspace/camlive && PYTHONPATH=cam_live/src .venv/bin/python -m "
                                   "cam_live.predict.diagnose" + (" --quick" if quick else ""), "sizes": sz}
    if want("1a"):
        out["1a_world_vs_marginal_same_mechanism"] = world_vs_marginal(
            mechanism_seed=7, n_train=sz["n_train"], n_test=sz["n_test"], n_decisions=sz["n_decisions"],
            steps=sz["steps"], log_every=log_every)
    if any(want(k) for k in ("1b", "2", "3", "4", "5")):
        fam = train_family(n_train=sz["fam_train"], n_decisions=sz["n_decisions"], steps=sz["fam_steps"],
                           log_every=log_every, n_eval_lifetimes=sz["eval_lifetimes"])
        out["family_model"] = fam["info"]
        if want("1b"):
            out["1b_world_vs_marginal_new_mechanisms"] = family_world_vs_marginal(
                fam, range(10_000, 10_000 + sz["n_test"]))
        if any(want(k) for k in ("2", "3", "5")):
            hp = heldout_pass(fam, range(20_000, 20_000 + sz["held"]))
            if want("2"):
                out["2_fast_adaptation_new_worlds"] = fast_adaptation(hp["adaptation_rows"])
            if want("3"):
                out["3_evaluator_calibration_prospective"] = calibration(hp["success"])
            if want("5"):
                out["5_action_quality_vs_raw_outcome"] = quality_vs_outcome(hp["quality_rows"])
        if want("4"):
            out["4_value_of_information"] = {
                "probe_vs_noisy_gauge": voi_probe_vs_gauge(fam, range(30_000, 30_000 + max(4, sz["held"] // 2))),
                "closed_loop": voi_closed_loop(fam, range(40_000, 40_000 + sz["loop"]), sz["max_decisions"])}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "docs", "evidence", "B", "diagnostics.json"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated experiment ids, e.g. 1a or 1b,2,3,4,5")
    ap.add_argument("--update", action="store_true", help="merge into an existing output file")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    res = run(args.quick, [s for s in args.only.split(",") if s] or None, args.log_every)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.update and os.path.exists(args.out):
        with open(args.out) as f:
            old = json.load(f)
        res["seconds"] = round(float(old.get("seconds", 0.0)) + res["seconds"], 1)
        old.update(res)
        res = old
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=str)
    print(json.dumps(res, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
