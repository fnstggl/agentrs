"""Experiment 2 (spec Part 17): causal discovery with agent-selected interventions, CAM Unified's learned exploration
against the mathematical knowledge-gradient and Thompson-sampling baselines.

Worlds. Hidden-mechanism families whose mechanisms contain confounders (a hidden common cause of gauges and money),
interacting levers and delayed consequences (catalogue motifs ``confounder``, ``interaction`` / ``gate``, ``delay`` /
``stock``; training families, so the structures are familiar and the instances new). Every world starts at its
resting settings and is presented the same way (a facility brief, neutral lever and gauge names in random order, one
typed tool per lever): before any intervention the stream does not say which mechanism is active.

Learners act with their own experiments (one lever setting = one decision = one tick) for the whole episode:
``cam`` (trained checkpoints, weights-only; learned exploration from its own policy, no KG input -
``cfg.math_explore_assist`` is off), ``cam_untrained``, and `nig_agent.NIGToolAgent` with rules ``kg``,
``thompson``, ``greedy``, ``random`` on the meta-learned NIG prior of Experiment 1's baseline fit (``--prior meta``,
default) or the generic prior.

Measures per (learner, training seed, world):
* intervention-effect accuracy at k = 0..64 own decisions: log score and squared error of each learner's own
  prediction of forked single-lever interventions (evaluator-side forks, never shown to the learner);
* decision-relevant accuracy: normalised regret of the intervention its model ranks best among the queries;
* observations needed: the smallest k in K_SET from which that regret stays <= 0.1 at every later grid point
  (None = not reached by 64);
* cumulative regret: sum over ticks of (reference - net) / (reference - default), over the first 64 ticks and over the
  episode; goal achievement (`Task.success` on the episode's net); compute per decision (wall / CPU / GPU section).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

from . import exp1 as E1
from . import mechrun as M
from . import predict as P
from . import stack as S
from . import stats as ST
from .baselines import NIGPredictor
from .nig_agent import NIGToolAgent

FAMILIES = ("cw1_scm_confounder", "cw1_scm_confounder_interaction", "cw1_scm_interaction", "cw1_scm_gate_interaction",
            "cw1_scm_delay", "cw1_scm_delay_interaction", "cw1_scm_delay_stock", "scm_delay")
MOTIF = {"cw1_scm_confounder": "confounder", "cw1_scm_confounder_interaction": "confounder",
         "cw1_scm_interaction": "interaction", "cw1_scm_gate_interaction": "interaction", "cw1_scm_delay": "delay",
         "cw1_scm_delay_interaction": "delay", "cw1_scm_delay_stock": "delay", "scm_delay": "delay"}
SETS = {"dev": {"offset": 110_000, "n": 2}, "confirm": {"offset": 210_000, "n": 3}}
ROOT = os.path.join(S.RUNS_ROOT, "exp2")


def set_dir(name: str) -> str:
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def cmd_prepare(a) -> None:
    sp = SETS[a.set]
    tasks = M.eval_task_set(FAMILIES, sp["n"], sp["offset"], workers=1)
    with open(os.path.join(set_dir(a.set), "tasks.pkl"), "wb") as fh:
        pickle.dump(tasks, fh)
    print(json.dumps({"tasks": [t.id for t in tasks], "failures": [str(f)[:200] for f in M.eval_task_set.failures]}))


def regret_rows(task, nets) -> dict:
    gap = task.reference_net - task.default_net
    r = [(task.reference_net - n) / gap for n in nets]
    return {"cumregret64": float(np.sum(r[:64])), "cumregret": float(np.sum(r)), "ticks": len(nets)}


def obs_needed(prows: list, task_id: str, thr: float = 0.1):
    """Smallest k in K_SET from which the decision-relevant regret stays <= thr at every later grid point."""
    ks = sorted((r["k"], r["regret"]) for r in prows if r["task"] == task_id and r["regret"] == r["regret"])
    for i, (k, _) in enumerate(ks):
        if all(reg <= thr for _, reg in ks[i:]):
            return k
    return None


def run_nig(rule: str, tasks: list, seed: int, prior: NIGPredictor, ks=P.K_SET) -> tuple:
    rows, prows = [], []
    for i in range(0, len(tasks), 16):
        part = tasks[i:i + 16]
        ag = NIGToolAgent(rule, prior=prior, seed=seed)
        w = M.make_world(part, seed=seed * 1000 + i)
        rec = P.TrajectoryRecorder(w, ks=ks, n_queries=12, seed=seed)
        t0, c0 = time.time(), time.process_time()
        res = M.run_batch(ag, w, learn=False, on_round=rec)
        wall, cpu = time.time() - t0, time.process_time() - c0
        eps = rec.episodes()
        for r, ep, t in zip(res, eps, part):
            pr = [{"task": t.id, "k": k, **P.score_queries(ep, k, *prior.predict(ep, k))} for k in ks if k in ep.queries]
            prows += pr
            rows.append({"task": t.id, "family": t.family, "motif": MOTIF.get(t.family), "score": r.score,
                         "success": r.success, "windows": r.windows, "decisions": r.decisions,
                         "obs_needed": obs_needed(pr, t.id), **regret_rows(t, r.nets),
                         "wall_per_decision": wall / max(1, sum(x.decisions for x in res)),
                         "cpu_per_decision": cpu / max(1, sum(x.decisions for x in res)), "gpu_per_decision": 0.0})
    return rows, prows


def cmd_run(a) -> None:
    d = set_dir(a.set)
    with open(os.path.join(d, "tasks.pkl"), "rb") as fh:
        tasks = pickle.load(fh)
    if a.limit:
        tasks = tasks[:a.limit]
    meta = {"learner": a.learner, "tag": a.tag or a.learner, "seed": a.seed, "set": a.set}
    t0 = time.time()
    if a.learner.startswith("cam"):
        st, info = E1.cam_stack(a.learner, a.ckpt, a.seed)
        meta["load"] = info
        rows = []
        c0 = time.process_time()
        from ..observe.accounting import ComputeMeter
        meter = ComputeMeter("exp2-cam")
        with meter.gpu():
            r, prows = E1.cam_decide(st, tasks, cond="B", seed=a.seed, imagination=a.imagination)
        snap = meter.snapshot()
        wall, cpu = time.time() - t0, time.process_time() - c0
        ndec = max(1, sum(x["decisions"] for x in r))
        for x in r:
            t = next(tt for tt in tasks if tt.id == x["task"])
            rows.append({**x, "motif": MOTIF.get(x["family"]), "obs_needed": obs_needed(prows, x["task"]),
                         **regret_rows(t, x.pop("nets_raw")), "wall_per_decision": wall / ndec,
                         "cpu_per_decision": cpu / ndec, "gpu_per_decision": float(snap.get("gpu_s", 0.0)) / ndec})
    else:
        prior = E1.load_predictor("nig_meta", a.seed) if a.prior == "meta" else NIGPredictor(meta=False)
        rows, prows = run_nig(a.learner, tasks, a.seed, prior)
        meta["prior"] = a.prior
    meta["wall_s"] = time.time() - t0
    E1._rows_out(os.path.join(d, "episodes.jsonl"), [{**meta, **x} for x in rows])
    E1._rows_out(os.path.join(d, "predict.jsonl"), [{**meta, **x, "split": "familiar", "cond": "B",
                                                    "family": x["task"].split(":")[0]} for x in prows])
    E1._rows_out(os.path.join(d, "runs.jsonl"), [{"cmd": "run", **meta}])
    print(f"[exp2] {meta['tag']} seed {a.seed}: {len(rows)} episodes {meta['wall_s']:.0f}s")


def cmd_report(a) -> None:
    d = set_dir(a.set)
    with open(os.path.join(d, "tasks.pkl"), "rb") as fh:
        tasks = [t.id for t in pickle.load(fh)]
    rows = [json.loads(l) for l in open(os.path.join(d, "episodes.jsonl"))]
    prows = [json.loads(l) for l in open(os.path.join(d, "predict.jsonl"))]
    rep = {"set": a.set, "learners": {}}
    tags = sorted({r["tag"] for r in rows})
    mats = {}
    for tag in tags:
        rr = [r for r in rows if r["tag"] == tag]
        for r in rr:
            r["succ"] = None if r.get("success") is None else float(bool(r["success"]))
            r["needed"] = 128.0 if r.get("obs_needed") is None else float(r["obs_needed"])
            r["reached"] = 0.0 if r.get("obs_needed") is None else 1.0
        blk = {m: ST.two_level(E1._matrix(rr, m, tasks)) for m in
               ("score", "succ", "cumregret64", "cumregret", "needed", "reached", "wall_per_decision",
                "cpu_per_decision")}
        pr = [r for r in prows if r["tag"] == tag]
        for metric in ("logscore", "sqerr", "regret"):
            per_k = {k: E1._matrix([r for r in pr if r["k"] == k], metric, tasks) for k in P.K_SET}
            blk[f"effect_{metric}"] = ST.curve(per_k, n_boot=1000)
            blk[f"effect_{metric}_area"] = ST.two_level(ST.area(per_k))
            mats[(tag, metric)] = per_k
        mats[(tag, "cumregret64")] = E1._matrix(rr, "cumregret64", tasks)
        mats[(tag, "score")] = E1._matrix(rr, "score", tasks)
        rep["learners"][tag] = blk
    if a.reference in tags:
        cmp = {}
        for tag in tags:
            if tag == a.reference:
                continue
            cmp[tag] = {"cumregret64": ST.paired(mats[(a.reference, "cumregret64")], mats[(tag, "cumregret64")]),
                        "score": ST.paired(mats[(a.reference, "score")], mats[(tag, "score")]),
                        "effect_logscore_area": ST.paired(ST.area(mats[(a.reference, "logscore")]),
                                                          ST.area(mats[(tag, "logscore")])),
                        "effect_regret_area": ST.paired(ST.area(mats[(a.reference, "regret")]),
                                                        ST.area(mats[(tag, "regret")]))}
        rep[f"{a.reference}-minus"] = cmp
    out = os.path.join(d, f"report_{a.reference}.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    print(f"[exp2] report -> {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--set", default="dev")
    p = sub.add_parser("run"); p.add_argument("--set", default="dev"); p.add_argument("--learner", required=True)
    p.add_argument("--ckpt", default=None); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None); p.add_argument("--prior", default="meta")
    p.add_argument("--limit", type=int, default=0); p.add_argument("--imagination", action="store_true")
    p = sub.add_parser("report"); p.add_argument("--set", default="dev"); p.add_argument("--reference", default="cam")
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "run": cmd_run, "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
