"""cam_live.eval.experiments — one executed run of one condition, and everything measured from it.

    run_condition(condition, family, seed, ...) -> result dict (also written to ``out``)

A run is a sequence of ``n_lifetimes`` objectives of ``family`` (world seeds ``world_seed_base + i``) taken by one
agent of ``condition`` (``eval.conditions``), with the weights carried over between objectives exactly as in a live
lifetime (``set_objective`` with a new objective id). Optionally the pre-experience and the experienced checkpoints
are saved (full checkpoint and weights-only export) for the transfer and continual-learning experiments.

Measured from the live agent at the end of the run (only for conditions that have the component):

* ``predictions``     every registered prediction that resolved: kind, predicted mean / sd, observed value, NLL,
                      registration and resolution time, lifetime                       (calibration, 16.2)
* ``decisions``       per decision: chosen index, probabilities, each candidate's p_success / value / predicted
                      goal-relevant mean, the ground-truth expected adopters of every candidate      (16.2, 16.3)
* ``experiments``     randomised comparisons the agent ran (arms, assignments, outcomes)                 (16.3)
* ``credit``          delayed outcomes attached to earlier records and the credit before / after, RUDDER return
                      conservation per closed objective                                                   (16.4)
* ``reward``          internal rewards per record, bootstrap mix, meta-update records, reward version     (16.5/6)
* ``learning``        learning-update records (timescale, groups, weight-delta norms)                     (16.7)
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..contracts import RunFiles
from ..runtime.logs import iter_jsonl
from .conditions import Timed, base_config, build_condition
from .runner import run_lifetime

EVAL_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "runs", "cam_live",
                         "eval")
EVAL_HOME = os.path.abspath(EVAL_HOME)
WORLD_BASE = {"F1": 10_000, "T1": 20_000, "T2": 30_000}


def _records(run_dir: str, name: str) -> List[Dict[str, Any]]:
    p = os.path.join(run_dir, name)
    meta = os.path.join(run_dir, "runtime_logs.json")
    if os.path.exists(meta):
        with open(meta) as f:
            p = os.path.join(run_dir, json.load(f).get("log_dir", ""), name)
    return list(iter_jsonl(p)) if os.path.exists(p) else []


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def export_checkpoint(agent: Any, ckpt_dir: str, export_dir: str, *, note: str) -> Dict[str, Any]:
    """Full checkpoint (``agent.save``) and its weights-only export (``credit.checkpoint.export_weights``)."""
    from ..credit.checkpoint import export_weights
    cid = agent.save(ckpt_dir)
    meta = export_weights(ckpt_dir, export_dir, config=agent.cfg.to_dict())
    return {"checkpoint_id": cid, "checkpoint": ckpt_dir, "export": export_dir, "note": note,
            "group_hashes": meta.get("group_hashes"), "n_tensors": meta.get("n_tensors"),
            "model_version": meta.get("model_version"), "reward_version": meta.get("reward_version")}


class CreditWatch:
    """Snapshots of (outcome count, credit) per experience record after every runtime step: a delayed outcome
    that arrives after later decisions must change the earlier record (16.4)."""

    def __init__(self, agent: Any):
        self.agent = agent
        self.first_seen: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []
        self.step = 0

    def __call__(self, rt: Any) -> None:
        self.step += 1
        try:
            recs = self.agent.store.all()
        except Exception:  # noqa: BLE001
            return
        n_dec = int(rt.counters.get("decisions", 0))
        for r in recs:
            st = self.first_seen.get(r.exp_id)
            n_out = len(r.outcomes)
            cred = {k: round(float(v), 8) for k, v in (r.credit or {}).items()}
            if st is None:
                self.first_seen[r.exp_id] = {"decisions_at_record": n_dec, "n_out": n_out, "credit": cred,
                                             "objective": r.objective_id}
                continue
            if n_out > st["n_out"]:
                self.events.append({"exp_id": r.exp_id, "objective": r.objective_id,
                                    "later_decisions": n_dec - st["decisions_at_record"], "kind": "outcome_attached",
                                    "n_out_before": st["n_out"], "n_out_after": n_out})
                st["n_out"] = n_out
            if cred != st["credit"]:
                self.events.append({"exp_id": r.exp_id, "objective": r.objective_id,
                                    "later_decisions": n_dec - st["decisions_at_record"], "kind": "credit_changed",
                                    "before": st["credit"], "after": cred})
                st["credit"] = cred


def run_condition(condition: str, family: str, seed: int, *, n_lifetimes: int, max_decisions: int,
                  real: bool = False, scale: str = "", out: str = "", save_dir: str = "", k: int = 16,
                  world_seed_base: Optional[int] = None, threads: int = 0, watch_credit: bool = True,
                  budget_s: float = 0.0, device: str = "") -> Dict[str, Any]:
    from ..validation import OffsetClock
    if threads:
        torch.set_num_threads(int(threads))
    t_start = time.time()
    base = base_config(real=real, scale=scale, seed=seed)
    if device:
        base.device = device
    if base.device == "cuda" or (base.device == "auto" and torch.cuda.is_available()):
        torch.cuda.reset_peak_memory_stats()
    root = os.path.abspath(save_dir or os.path.join(EVAL_HOME, "runs", f"{condition}_{family}_s{seed}"))
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    clock = OffsetClock()
    agent = build_condition(condition, base, os.path.join(root, "agent"), seed=seed, clock=clock)
    timed = Timed(agent)
    res: Dict[str, Any] = {"executed": False, "condition": condition, "family": family, "seed": seed,
                           "real_backbone": bool(real), "backbone": base.backbone.name, "scale": base.core.scale,
                           "n_lifetimes": n_lifetimes, "max_decisions": max_decisions, "k": k, "root": root,
                           "parameters": {k2: v for k2, v in agent.model.count_parameters().items()
                                          if k2 != "experiential_breakdown"}}
    s0 = agent.model.trainable_state()
    res["checkpoints"] = {}
    if save_dir:
        res["checkpoints"]["pre"] = export_checkpoint(agent, os.path.join(root, "ckpt_pre"),
                                                      os.path.join(root, "export_pre"), note="before any experience")
    wb = WORLD_BASE[family] + 100 * int(seed) if world_seed_base is None else int(world_seed_base)
    watch = CreditWatch(agent) if watch_credit and hasattr(agent, "store") else None
    lifetimes = []
    for i in range(n_lifetimes):
        if budget_s and time.time() - t_start > budget_s:
            res["stopped_early"] = f"time budget {budget_s:.0f} s reached after {i} lifetimes"
            break
        lt = run_lifetime(timed, family, wb + i, os.path.join(root, f"lt{i:02d}"), clock,
                          max_decisions=max_decisions, k=k, on_step=watch)
        lt["index"] = i
        lt["mix"] = float(getattr(agent.meta.mixer, "mix", float("nan")))
        lt["reward_version"] = agent.meta.reward.version
        lt["model_version"] = agent.learner.model_version
        try:
            lt["conservation"] = _conservation(agent, lt["objective_id"])
        except Exception as e:  # noqa: BLE001
            lt["conservation"] = {"error": f"{type(e).__name__}: {e}"}
        lifetimes.append(lt)
    res["lifetimes"] = [{k2: v for k2, v in lt.items() if k2 not in ("truth_rows", "decision_rows")}
                        for lt in lifetimes]
    res["decision_rows"] = [lt["decision_rows"] for lt in lifetimes]
    res["truth_rows"] = [[{k2: r.get(k2) for k2 in ("operation", "arguments", "expected_adopters", "rate", "money",
                                                    "quality", "ratio", "status")} for r in lt["truth_rows"]]
                         for lt in lifetimes]
    res["compute"] = {"seconds": dict(timed.seconds), "cpu_seconds": dict(timed.cpu_seconds)}
    if save_dir:
        res["checkpoints"]["post"] = export_checkpoint(agent, os.path.join(root, "ckpt_post"),
                                                       os.path.join(root, "export_post"),
                                                       note=f"after {len(lifetimes)} lifetimes of {family}")
    try:
        res["weight_delta_norm_by_group"] = agent.model.group_delta_norms(s0)
    except Exception as e:  # noqa: BLE001
        res["weight_delta_norm_by_group"] = {"error": str(e)}
    res["extract"] = extract(agent, [lt["runtime_dir"] for lt in lifetimes], lifetimes, watch)
    res["wall_seconds"] = time.time() - t_start
    res["gpu_peak_gb"] = (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else None
    res["executed"] = True
    try:
        agent.close()
    except Exception:  # noqa: BLE001
        pass
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, default=str)
    return res


def _conservation(agent: Any, objective_id: str) -> Dict[str, Any]:
    rudder = getattr(agent, "rudder", None)
    if rudder is None or agent.store.objective_outcome(objective_id) is None:
        return {"closed": agent.store.objective_outcome(objective_id) is not None}
    red = rudder.redistribute([objective_id])
    rc = red.get(objective_id)
    if rc is None:
        return {"closed": True, "rudder": None}
    return {"closed": True, "conservation_error": _f(rc.conservation_error),
            "n_steps": len(getattr(rc, "exp_ids", []) or [])}


def extract(agent: Any, run_dirs: Sequence[str], lifetimes: Sequence[Dict[str, Any]],
            watch: Optional[CreditWatch]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    obj_of = {lt["objective_id"]: lt for lt in lifetimes}
    # ---- predictions (prospective, resolved later) ----
    preds = []
    for p in agent.predictions.records():
        if p.resolved_status == "pending" or p.resolved_value is None:
            continue
        preds.append({"kind": p.query.kind, "metric": p.query.metric.split(".act_")[0].split(".a")[0][:48],
                      "goal_relevant": bool(p.query.goal_relevant), "family": p.dist.family,
                      "mean": _f(p.dist.mean), "std": _f(p.dist.std), "epistemic_std": _f(p.dist.epistemic_std),
                      "p1": _f(p.dist.params.get("p")) if p.dist.family == "bernoulli" else None,
                      "observed": _f(p.resolved_value), "nll": _f(p.nll), "made_at": p.made_at,
                      "resolved_at": p.resolved_at, "decision_id": p.decision_id,
                      "imagined": int(p.imagined_horizon_steps or 0)})
    out["predictions"] = preds
    out["n_predictions_registered"] = len(agent.predictions)
    # ---- experiments ----
    try:
        out["experiments"] = [{"experiment_id": r.experiment_id, "arms": r.arms, "status": r.status,
                               "assignments": r.assignments, "supported": r.supported, "test": r.test,
                               "intervention": r.intervention, "comparison": r.comparison,
                               "predicted": [{"metric": p.query.metric, "mean": _f(p.dist.mean),
                                              "std": _f(p.dist.std), "action_id": p.action_id}
                                             for p in r.predicted],
                               "actual": [{"query_id": o.query_id, "value": _f(o.value), "status": o.status}
                                          for o in r.actual]}
                              for r in agent.experiments.records()]
    except Exception as e:  # noqa: BLE001
        out["experiments"] = {"error": str(e)}
    # ---- store: records, outcomes, credit, internal reward ----
    recs = []
    try:
        for r in agent.store.all():
            ch = r.decision.chosen if r.decision is not None else None
            recs.append({"exp_id": r.exp_id, "objective_id": r.objective_id, "seq": r.seq,
                         "operation": ch.operation if ch else None, "arguments": ch.arguments if ch else None,
                         "intrinsic_reward": _f(r.intrinsic_reward), "bootstrap_reward": _f(r.bootstrap_reward),
                         "reward_version": r.reward_version, "credit": {k: _f(v) for k, v in (r.credit or {}).items()},
                         "n_outcomes": len(r.outcomes),
                         "outcome_values": [_f(o.value) for o in r.outcomes][:6],
                         "lifetime": obj_of.get(r.objective_id, {}).get("index")})
    except Exception as e:  # noqa: BLE001
        recs = [{"error": str(e)}]
    out["records"] = recs
    if watch is not None:
        out["credit_events"] = watch.events
    # ---- learning and meta-learning ----
    ups, metas = [], []
    for d in run_dirs:
        for e in _records(d, RunFiles.LEARNING):
            u = e.get("data", {})
            ups.append({k: u.get(k) for k in ("timescale", "trigger", "param_groups", "weight_delta",
                                               "prediction_change", "model_version")})
        for e in _records(d, RunFiles.META):
            m = e.get("data", {})
            metas.append({k: m.get(k) for k in ("applied", "skipped_reason", "meta_grad_norm",
                                                 "n_independent_outcomes", "n_query", "n_support", "estimator",
                                                 "reward_version_before", "reward_version_after", "bootstrap_mix",
                                                 "outer_objective")})
    for lt in lifetimes:                       # updates made while closing out a lifetime (runner.close_out)
        for m in ((lt.get("close_out") or {}).get("meta_updates") or []):
            metas.append({**m, "close_out": True})
    out["learning_updates"] = ups
    out["meta_updates"] = metas
    try:
        out["agent_status"] = json.loads(json.dumps(agent.status(), default=str))
    except Exception as e:  # noqa: BLE001
        out["agent_status"] = {"error": str(e)}
    return out
