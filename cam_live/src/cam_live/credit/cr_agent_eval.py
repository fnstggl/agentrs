"""cam_live.credit.cr_agent_eval — does retrospective credit change and improve the integrated agent's learning from
delayed consequences inside one long objective? (CR, amendment item 4; evidence levels (b)-(d))

    python -m cam_live.credit.cr_agent_eval --condition simple|rudder|cocoa --seed S --decisions N --out OUT.json
            [--real] [--world-seed W] [--segment K] [--split segment|record] [--work DIR]

The real ``CamLiveAgent`` driven by the real ``runtime.loop.LiveRuntime`` through ONE long open objective of the
LabWorld key-door mechanism class (``eval.envs`` T1: inverted feature effects, ``lab.unlock`` has no immediate effect
and multiplies the later yield of the last channel; long horizon and budget, a target out of reach so the objective
never closes during the run). Debug scale on the CPU (tiny random backbone, debug core, schema-walker candidates) or,
with ``--real``, Qwen3-4B + 50m core on the GPU inside the shared GPU lock with backbone-written candidates on.

Conditions differ only in the estimator whose credit the policy update consumes (``CamLiveAgent._credit_for`` reads
``CreditAssigner.default``):

    simple    the evidence-selected default (the CR selection rule; it stays ``simple`` unless validated)
    rudder    RUDDER's redistributed return-to-go forced as the training default (ablation only)
    cocoa     COCOA's counterfactual credit forced as the training default (ablation only)

Every condition uses the same learner split (``--split``, default ``segment``: the CR change request) and the same
segment length, so the only difference is the credit signal. Per decision the ground truth of the executed action
(``eval.envs.TruthTap``: expected adopters = hidden rate x effort) is recorded, plus when the door was unlocked,
which credit the policy updates consumed (rows labelled ``credit:<estimator>``) and the selection log.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

DAY = 86400.0


def long_world(world_seed: int, clock: Any):
    from ..diagnostics.world import LabConfig, LabVerifier, LabWorld, lab_task
    cfg = LabConfig(seed=int(world_seed), n_channels=5, target=10 ** 6, budget=1e7, horizon_s=1000 * DAY,
                    outcome_horizon_s=6 * 3600.0, mean_delay_s=(1800.0, 4 * 3600.0), at_most_once=True,
                    w_mean=(-1.0, 0.5, -0.3), rate_scale=1.0, door_boost=6.0, door_locked=0.05, family="lab/door")
    world = LabWorld(cfg, clock)
    return world, LabVerifier(world), lab_task(world, f"obj_crlong_{world_seed}")


def force_estimator(agent: Any, name: str) -> None:
    """Ablation: ``name``'s credit is the training default and no evidence can change it."""
    sel = agent.assigner.selector
    sel.baseline = sel.default = name
    sel.min_units = 10 ** 9
    sel.min_experiments = 10 ** 9


def summarise(rows: List[Dict[str, Any]], n: int) -> Dict[str, Any]:
    ea = [float(r.get("expected_adopters", 0.0) or 0.0) for r in rows]
    unlock = next((i for i, r in enumerate(rows) if r.get("operation") == "lab.unlock" and r.get("status") == "ok"),
                  None)
    halves = {}
    for name, sl in (("first_half", slice(0, len(ea) // 2)), ("second_half", slice(len(ea) // 2, len(ea))),
                     ("last_quarter", slice(3 * len(ea) // 4, len(ea)))):
        part = ea[sl]
        halves[name] = sum(part) / len(part) if part else float("nan")
    k = len(ea) // 4
    halves["quarter_totals"] = [sum(ea[i * k:(i + 1) * k if i < 3 else len(ea)]) for i in range(4)] if k else []
    door = [r for r in rows if r.get("operation") == "lab.act"]
    oracle = [float(r.get("oracle_rate", 0.0) or 0.0) for r in rows]
    return {"decisions": len(rows), "expected_adopters_total": sum(ea), "expected_adopters_per_decision": halves,
            "unlock_index": unlock, "n_unlock": sum(1 for r in rows if r.get("operation") == "lab.unlock"),
            "n_act": len(door), "ops": {op: sum(1 for r in rows if r.get("operation") == op)
                                        for op in sorted({r.get("operation") for r in rows})},
            "oracle_rate_mean": sum(oracle) / len(oracle) if oracle else float("nan")}


def run(condition: str, seed: int, decisions: int, work: str, *, real: bool = False, world_seed: Optional[int] = None,
        segment: int = 16, split: str = "segment", backbone: Optional[bool] = None, replay: bool = True,
        max_wall_s: float = 0.0, learn_budget_s: float = 30.0) -> Dict[str, Any]:
    from ..contracts import SimClock
    from ..diagnostics.world import AllowAllPolicy
    from ..eval.conditions import Timed, base_config
    from ..eval.envs import TruthTap
    from ..eval.runner import DecisionTap, runtime_config
    from ..agent import CamLiveAgent
    from ..runtime.loop import LiveRuntime
    t_start = time.time()
    cfg = base_config(real=real, seed=seed)
    agent_sec = dict(cfg.sections.get("agent", {}))
    agent_sec["use_backbone"] = bool(real if backbone is None else backbone)
    cfg.sections["agent"] = agent_sec
    cred = dict(cfg.sections.get("credit", {}))
    cred["learner"] = {**cred.get("learner", {}), "val_split": split}
    cfg.sections["credit"] = cred
    clock = SimClock()
    os.makedirs(work, exist_ok=True)
    agent = CamLiveAgent(cfg, os.path.join(work, "agent"), clock=clock)
    agent.assigner.units.decisions = int(segment)
    if condition in ("rudder", "cocoa"):
        force_estimator(agent, condition)
    elif condition != "simple":
        raise KeyError(condition)
    has_replay = "pop_revised" in open(__import__("inspect").getsourcefile(type(agent))).read()
    if not replay:                                 # ablation of change request CR-3 (no replay of revised credit)
        agent.assigner.pop_revised = lambda n, estimator=None: []
    world, verifier, task = long_world(seed if world_seed is None else world_seed, clock)
    task.limits.max_decisions = int(decisions)
    tap = TruthTap(world)
    agent.attach_verifier(verifier)
    timed = Timed(agent)
    dtap = DecisionTap(timed, tap)
    consumed: List[Dict[str, Any]] = []
    orig = agent.meta.policy_update

    def logged(recs, *a, **kw):                    # what the real policy update consumed (no change to it)
        out = orig(recs, *a, **kw)
        srcs = [agent._credit_for(r) for r in recs]
        recent = set(agent.rewarded[-int(agent.settings.policy_update_batch):])
        consumed.append({"rows": len(recs), "credit_rows": sum(1 for c in srcs if c is not None),
                         "replayed_rows": sum(1 for r in recs if r.exp_id not in recent),
                         "replayed_credit_rows": sum(1 for r, c in zip(recs, srcs) if c is not None and
                                                     r.exp_id not in recent),
                         "estimators": sorted({c[1] for c in srcs if c is not None}),
                         "rows_by_estimator": {e: sum(1 for c in srcs if c is not None and c[1] == e)
                                               for e in sorted({c[1] for c in srcs if c is not None})},
                         "return_like_rows": sum(1 for c in srcs if c is not None and c[2]),
                         "applied": out is not None, "n_decisions": int(agent.n_decisions)})
        return out
    agent.meta.policy_update = logged
    rcfg = runtime_config(16)
    # exact pairing: the runtime's per-step learning budgets are wall-clock; an unbounded budget keeps two same-seed
    # runs identical under any machine load (ST, G3A-3)
    rcfg.learn_budget_s = rcfg.meta_budget_s = float(learn_budget_s)
    rt = LiveRuntime(dtap, tap, AllowAllPolicy(), verifier, task, os.path.join(work, "rt"), clock, rcfg)
    rt.start()
    steps = 0
    stopped_by_wall = False
    while rt.status == "running" and steps < 6 * decisions:
        if max_wall_s and time.time() - t_start > max_wall_s:      # bounded GPU hold (shared lock)
            stopped_by_wall = True
            break
        rt.step()
        steps += 1
    if rt.status == "running":
        rt._finish("stopped", "evaluation wall-clock bound reached" if stopped_by_wall else "step cap reached")
    status = rt.status
    counters = dict(rt.counters)
    errors = list(rt._errors[-5:])
    rt.close()
    st = agent.store
    recs = st.trajectory(task.objective_id)
    keys: Dict[str, int] = {}
    for r in recs:
        for k in r.credit:
            keys[k] = keys.get(k, 0) + 1
    hist = {}
    for r in recs[: min(len(recs), 64)]:
        h = st.credit_history(r.exp_id)
        if h:
            hist[r.exp_id] = len(h)
    sel_rows = []
    p = os.path.join(agent.dir, "credit_selection.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ev = d.get("evaluation") or {}
            se = d.get("selection") or {}
            sel_rows.append({"ran": ev.get("ran"), "n_units": ev.get("n_units"), "reason": (ev.get("reason") or "")[:300],
                             "default_after": se.get("default_after", ev.get("default_after")),
                             "mode": se.get("mode")})
    out = {"condition": condition, "seed": seed, "world_seed": seed if world_seed is None else world_seed,
           "real": real, "use_backbone": agent_sec["use_backbone"], "segment_decisions": segment, "split": split,
           "replay_cr3": bool(replay and has_replay), "agent_source": __import__("inspect").getsourcefile(type(agent)),
           "runtime_status": status, "steps": steps, "stopped_by_wall": stopped_by_wall,
           "learn_budget_s": float(learn_budget_s), "agent_uid": getattr(agent, "uid", None), "counters": counters, "errors": errors,
           "summary": summarise(tap.rows, decisions), "truth_rows": tap.rows,
           "decision_rows": [{"i": i, "chosen": (row["candidates"][row["chosen_index"]]["operation"],
                                                 json.dumps(row["candidates"][row["chosen_index"]]["arguments"],
                                                            sort_keys=True))
                              if 0 <= row["chosen_index"] < len(row["candidates"]) else None,
                              "probabilities": [round(float(x), 6) for x in row["probabilities"]],
                              "exploration": row["exploration"], "t": row["t"]}
                             for i, row in enumerate(dtap.rows)],
           "credit_keys": keys, "credit_versions_first64": hist,
           "default_at_end": agent.assigner.default, "policy_updates": consumed,
           "selection_log": sel_rows[-20:], "n_selection_rows": len(sel_rows),
           "learner_counters": dict(agent.learner.counters),
           "trained_units": len(agent.learner.trained_units),
           "held_out_records": sum(1 for r in recs if agent.learner.is_validation(r)),
           "wall_seconds": time.time() - t_start, "compute": {"seconds": dict(timed.seconds)}}
    return out


def _run_main(a: Any, work: str) -> Dict[str, Any]:
    return run(a.condition, a.seed, a.decisions, work, real=a.real, world_seed=a.world_seed, segment=a.segment,
               split=a.split, backbone=False if a.no_backbone else None, replay=not a.no_replay,
               max_wall_s=a.max_wall_s, learn_budget_s=a.learn_budget_s)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--condition", required=True, choices=["simple", "rudder", "cocoa"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world-seed", type=int, default=None)
    ap.add_argument("--decisions", type=int, default=120)
    ap.add_argument("--segment", type=int, default=16)
    ap.add_argument("--split", default="segment", choices=["segment", "record"])
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--no-backbone", action="store_true", help="real model without backbone-written candidates")
    ap.add_argument("--no-replay", action="store_true", help="disable CR-3 (replay of revised credit) in a patched agent")
    ap.add_argument("--rudder-key", default="", help="ablation: RUDDER's learning key for this process (e.g. 'rudder', "
                    "the segment-return signal used before CR round 2, return-like)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-wall-s", type=float, default=0.0, help="stop the run after this many seconds (GPU hold)")
    ap.add_argument("--gpu-timeout-s", type=float, default=3600.0, help="how long to wait for the shared GPU lock")
    ap.add_argument("--learn-budget-s", type=float, default=30.0, help="runtime learning budget per step (wall clock); "
                    "1e9 for exactly paired arms")
    ap.add_argument("--work", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    if a.rudder_key:                               # ablation of D-CR2's choice of RUDDER's learning signal
        from . import providers as _p
        _p.LEARNING_KEY[_p.RUDDER] = a.rudder_key
        _p.RETURN_LIKE[_p.RUDDER] = a.rudder_key != _p.KEY_CONTRIB
    work = a.work or os.path.splitext(a.out)[0] + "_work"
    lock = contextlib.nullcontext()
    if a.real:
        from ..nn.gpu import gpu_lock
        lock = gpu_lock(timeout_s=float(a.gpu_timeout_s), min_free_gb=16.0)
    with lock:
        flag = os.path.join(os.path.dirname(os.path.abspath(a.out)), "HOLDING_GPU") if a.real else ""
        if flag:                                   # lets the caller pause its own CPU jobs during the hold
            open(flag, "w").write(str(os.getpid()))
        try:
            res = _run_main(a, work)
        finally:
            if flag and os.path.exists(flag):
                os.remove(flag)
    from .diagnose import code_provenance

    res["rudder_learning_key"] = a.rudder_key or "default"
    res["_code"] = code_provenance()
    res["_command"] = "cd /workspace/camlive && .venv/bin/python -m cam_live.credit.cr_agent_eval " + " ".join(
        argv if argv is not None else __import__("sys").argv[1:])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True, default=str)
    print(json.dumps({k: res[k] for k in ("condition", "seed", "runtime_status", "summary", "credit_keys", "replay_cr3",
                                          "default_at_end", "wall_seconds", "held_out_records", "n_selection_rows")},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def credit_quality(result_json: str) -> Dict[str, Any]:
    """How well each estimator's stored credit tracks the ground truth of the agent's own decisions (diagnosis of an
    ablation run): for every executed ``lab.act`` the truth is its expected adopters (hidden rate x effort, the tap's
    row); Pearson / Spearman correlation and the spread of each credit key over those decisions, and the same for the
    realised return-to-go within the decision's segment."""
    from .replay import SqliteExperienceStore
    res = json.load(open(result_json))
    work = os.path.splitext(result_json)[0] + "_work"
    store = SqliteExperienceStore(os.path.join(work, "agent", "learner", "experience.sqlite"), durable=False)
    try:
        oid = next(o for o in store.objective_ids() if o.startswith("obj_crlong"))
        recs = [r for r in store.trajectory(oid) if not r.decision.chosen.parent_decision_id]
    finally:
        store.close()
    rows = res["truth_rows"]
    pairs: Dict[str, List[Tuple[float, float]]] = {}
    n = min(len(rows), len(recs))
    for r, t in zip(recs[:n], rows[:n]):
        if t.get("operation") != "lab.act" or r.decision.chosen.operation != "lab.act" or "expected_adopters" not in t:
            continue
        for k, v in r.credit.items():
            pairs.setdefault(k, []).append((float(v), float(t["expected_adopters"])))

    def rank(x: List[float]) -> List[float]:
        order = sorted(range(len(x)), key=lambda i: x[i])
        out = [0.0] * len(x)
        for pos, i in enumerate(order):
            out[i] = float(pos)
        return out

    def corr(a: List[float], b: List[float]) -> float:
        if len(a) < 3:
            return float("nan")
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        sa = math.sqrt(sum((x - ma) ** 2 for x in a))
        sb = math.sqrt(sum((x - mb) ** 2 for x in b))
        return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa > 0 and sb > 0 else float("nan")

    out = {}
    for k, ps in pairs.items():
        a, b = [p[0] for p in ps], [p[1] for p in ps]
        m = sum(a) / len(a)
        out[k] = {"n": len(ps), "pearson": corr(a, b), "spearman": corr(rank(a), rank(b)),
                  "sd": math.sqrt(sum((x - m) ** 2 for x in a) / max(1, len(a) - 1))}
    return {"condition": res["condition"], "seed": res["seed"], "keys": out}
