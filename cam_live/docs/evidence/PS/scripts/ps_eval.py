"""PS evaluation: the integrated agent with the persistent core state carried across decisions (``on``) against the
same agent without it (``off``: every belief from the fresh state, the behaviour before PS), same seeds.

Runs one condition per process through the real runtime (``eval.runner.run_lifetime`` -> ``runtime.loop.LiveRuntime``)
over phases of objectives, e.g. ``F1:3,T1:3,F1:2,T2:2``: an experience family, an interfering family with a
different mechanism, the first family again (retention), and an unfamiliar family (held out).

Per lifetime: verified progress; prospective NLL of every resolved prediction made in it (registered before
execution, scored when the outcome arrived), overall and by outcome kind; for ``on`` also the causal effect of the
carried state at each decision: total variation between the trainable policy's distribution over the offered
candidates with the carried state and with the fresh state (same weights, same context), and the relative change of
the belief b. Retention: world-model NLL of the first phase's own stored records, measured after the first phase and
again after every later phase (the same records, re-encoded with their decision's carried state when ``on``).

    python ps_eval.py --cond on --seed 0 --out DIR [--real] [--backbone] [--phases F1:3,T1:3,F1:2,T2:2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

import torch

torch.set_num_threads(int(os.environ.get("PS_THREADS", "4")))     # a shared machine: never oversubscribe the CPU


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=("on", "off"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real", action="store_true", help="Qwen3-4B + 50m core on the GPU (inside the shared GPU lock)")
    ap.add_argument("--backbone", action="store_true", help="open-ended generation by the backbone on")
    ap.add_argument("--phases", default="F1:3,T1:3,F1:2,T2:2")
    ap.add_argument("--decisions", type=int, default=16)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--unroll", type=int, default=2)
    ap.add_argument("--max-context", type=int, default=0, help="override core.max_context (0 = configuration's)")
    ap.add_argument("--world-offset", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from cam_live.contracts import SimClock
    from cam_live.eval.conditions import Timed, base_config
    from cam_live.eval.runner import run_lifetime
    from cam_live.agent import CamLiveAgent

    cfg = base_config(real=a.real, seed=a.seed)
    if a.max_context:
        cfg.core.max_context = int(a.max_context)
    agent_sec = dict(cfg.sections.get("agent", {}))
    agent_sec["use_backbone"] = bool(a.backbone)
    cfg.sections["agent"] = agent_sec
    cfg.sections["persistent_state"] = {"enabled": a.cond == "on", "unroll": int(a.unroll)}
    clock = SimClock(1.8e9)
    root = os.path.join(a.out, f"{a.cond}_s{a.seed}")
    os.makedirs(root, exist_ok=True)

    def build():
        return CamLiveAgent(cfg, os.path.join(root, "agent"), clock=clock)

    if a.real:
        from cam_live.nn.gpu import gpu_lock
        lock = gpu_lock()
    else:
        import contextlib
        lock = contextlib.nullcontext()
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    retention: Dict[str, Any] = {}
    with lock:
        agent = build()
        timed = Timed(agent)
        probe = CausalProbe(agent) if a.cond == "on" else None
        phases = [(f, int(n)) for f, n in (p.split(":") for p in a.phases.split(","))]
        first_recs: List[str] = []
        idx = 0
        for pi, (fam, n) in enumerate(phases):
            for j in range(n):
                ws = 1000 * (pi + 1) + j + a.world_offset if not (pi > 0 and fam == phases[0][0]) else \
                    5000 + 1000 * pi + j + a.world_offset          # a repeated family gets new worlds of it
                t_start = float(clock.now())
                pred_before = set(p.prediction_id for p in agent.predictions.records())
                if probe is not None:
                    probe.reset()
                tgt = timed if probe is None else probe.wrap(timed)
                lt = run_lifetime(tgt, fam, ws, os.path.join(root, f"lt{idx:02d}_{fam}_{ws}"), clock,
                                  max_decisions=a.decisions, k=a.k)
                new = [p.prediction_id for p in agent.predictions.records() if p.prediction_id not in pred_before]
                row = {k: lt.get(k) for k in ("family", "world_seed", "decisions", "verified_status", "progress",
                                               "achieved", "quality", "wall_seconds", "training_s", "inference_s")}
                row.update(index=idx, phase=pi, t_start=t_start, t_end=float(clock.now()), prediction_ids=new)
                recs_now = {p.prediction_id: p for p in agent.predictions.records()}
                row["nll"] = _nll_summary([recs_now[i] for i in new if i in recs_now])   # all arrived: deadline passed
                if probe is not None:
                    row["causal"] = probe.summary()
                ps = agent.persist.status() if agent.persist is not None else {"enabled": False}
                row["persistent_state"] = {k: ps.get(k) for k in ("positions", "rho_mean", "rho_min", "rho_max")}
                row["persistent_counters"] = ps.get("counters", {})
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("index", "family", "progress", "decisions")}), flush=True)
                _dump(a, {"cond": a.cond, "seed": a.seed, "partial": True, "lifetimes": rows,
                          "retention_first_phase_records": retention})
                idx += 1
            if pi == 0:
                first_recs = [r.exp_id for r in agent.store.all()]
            retention[f"after_phase_{pi}"] = _nll_of(agent, first_recs)
        # prospective NLL per lifetime, re-read at the end (an outcome arriving after its lifetime is included)
        recs = {p.prediction_id: p for p in agent.predictions.records()}
        for row in rows:
            row["nll_final"] = _nll_summary([recs[i] for i in row.pop("prediction_ids") if i in recs])
        status = agent.status()
        out = {"cond": a.cond, "seed": a.seed, "real": a.real, "backbone": a.backbone, "phases": a.phases,
               "decisions": a.decisions, "unroll": a.unroll, "max_context": int(cfg.core.max_context),
               "lifetimes": rows, "retention_first_phase_records": retention,
               "persistent_state": status.get("persistent_state"), "parameters": status.get("parameters"),
               "backbone_name": status.get("backbone"), "scale": status.get("scale"),
               "wall_seconds": time.time() - t0, "commit": _commit()}
        agent.close()
    _dump(a, out)
    print("wrote", os.path.join(a.out, f"{a.cond}_s{a.seed}.json"), round(time.time() - t0, 1), "s")
    return 0


def _dump(a: Any, obj: Dict[str, Any]) -> None:
    path = os.path.join(a.out, f"{a.cond}_s{a.seed}.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(path + ".tmp", path)


def _commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def _nll_of(agent: Any, exp_ids: List[str]) -> Dict[str, Any]:
    recs = [agent.store.get(e) for e in exp_ids]
    recs = [r for r in recs if r is not None and r.result is not None]
    vals = agent._world_nll(recs) if recs else []
    v = [x for x in vals if x is not None and math.isfinite(x)]
    return {"n_records": len(recs), "n_scored": len(v), "mean_nll": (sum(v) / len(v)) if v else None}


def _nll_summary(preds: List[Any]) -> Dict[str, Any]:
    by = defaultdict(list)
    goal = []
    for p in preds:
        if p.nll is None or not math.isfinite(p.nll):
            continue
        by[p.query.kind].append(float(p.nll))
        by["all"].append(float(p.nll))
        if p.query.goal_relevant:
            goal.append(float(p.nll))
    out = {k: {"n": len(v), "mean": sum(v) / len(v)} for k, v in by.items()}
    out["goal_relevant"] = {"n": len(goal), "mean": (sum(goal) / len(goal)) if goal else None}
    out["registered"] = len(preds)
    return out


class CausalProbe:
    """At every ``select``: the policy's distribution over the offered candidates and the belief, with the carried
    state and with the fresh state (the chain switched off for the second computation only)."""

    def __init__(self, agent: Any):
        self.agent = agent
        self.reset()

    def reset(self) -> None:
        self.tv: List[float] = []
        self.db: List[float] = []

    def wrap(self, timed: Any) -> Any:
        probe = self

        class _P:
            def __getattr__(self, name: str) -> Any:
                return getattr(timed, name)

            def __setattr__(self, name: str, value: Any) -> None:
                setattr(timed, name, value)

            def select(self, evaluations: Any) -> Any:
                probe.measure(evaluations)
                return timed.select(evaluations)
        return _P()

    def measure(self, evaluations: Any) -> None:
        ag = self.agent
        cands = [e.action for e in evaluations]
        if len(cands) < 2 or ag.persist is None:
            return
        refs = ag.pending_eval.get("refs") or ag._window()
        items = ag._items(refs)
        with torch.no_grad():
            p_on = ag.policy_distribution(items, cands)
            b_on = ag._belief(items)
            ag.persist.settings.enabled = False
            try:
                p_off = ag.policy_distribution(items, cands)
                b_off = ag._belief(items)
            finally:
                ag.persist.settings.enabled = True
        self.tv.append(float(0.5 * (p_on - p_off).abs().sum()))
        self.db.append(float((b_on - b_off).norm() / b_off.norm().clamp_min(1e-12)))

    def summary(self) -> Dict[str, Any]:
        n = len(self.tv)
        return {"n": n, "mean_tv": (sum(self.tv) / n) if n else None, "max_tv": max(self.tv) if n else None,
                "mean_rel_db": (sum(self.db) / n) if n else None}


if __name__ == "__main__":
    sys.exit(main())
