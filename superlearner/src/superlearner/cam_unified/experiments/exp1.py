"""Experiment 1 (spec Part 17): prediction and decision improvement after 0, 1, 2, ..., 64 observations.

Subcommands (every result row is one (learner, training seed, task, condition, k) cell, appended as JSON lines under
``runs/cam_unified/wave2/X/exp1/<set>/``):

    prepare   --set dev|confirm                  validated tasks + identical support episodes (A, C, D) -> pickles
    fit       --seed s                           train the predictive baselines on training-family design episodes
    predict   --set S --learner L [--ckpt C] [--seed s]   prediction rows for conditions A, C, D
    decide    --set S --learner L [--ckpt C] [--seed s]   agent-selected decisions (condition B): window scores,
                                                 episode score, success; CAM also records its own-trajectory
                                                 prediction quality at k (forked queries, evaluator side)
    ntrain    --learner L --seed s --ticks N     train a Crucible numeric learner on the training stream
    report    --set S                            two-level bootstrap summaries -> JSON

Learners: ``cam`` (a checkpoint of `train`, weights-only load in this fresh process; experience access removed),
``cam_untrained`` (fresh weights, seed s), ``nig_exact``, ``nig_meta``, ``seq_gru``, ``seq_attn`` (predictive
baselines, `baselines`), and for decisions the Crucible learners ``model_based``, ``ppo_gru``, ``probe_exploit``,
``random``, ``hold`` and CAM 7.0 ``cam7`` (trained with `ntrain` on the same families and seed ranges).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from typing import Optional

import numpy as np
import torch

from . import baselines as BL
from . import mechrun as M
from . import predict as P
from . import stack as S
from . import stats as ST

FAMILIAR = ("scm_additive", "scm_delay", "scm_gate", "cw1_scm_confounder", "cw1_scm_interaction",
            "cw1_scm_delay_stock", "cw1_scm_feedback_gate", "cw1_scm_regime_threshold")
HELDOUT = ("scm_delay_gate", "scm_invu", "cw1_scm_delay_gate", "cw1_scm_feedback_threshold",
           "cw1_scm_interaction_stock", "cw1_scm_invu", "cw1_scm_invu_delay", "cw1_scm_gate_regime_threshold")
SETS = {"dev": {"offset": 100_000, "n": 2}, "confirm": {"offset": 200_000, "n": 3}}
DESIGN_SEED = {"dev": 11, "confirm": 23}
CONDITIONS = {"A": {}, "C": {"notes_m": 4}, "D": {"condition": "D", "delay": 3, "hide_frac": 0.5}}
ROOT = os.path.join(S.RUNS_ROOT, "exp1")


def set_dir(name: str) -> str:
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def split_of(family: str) -> str:
    return "familiar" if family in FAMILIAR else ("heldout" if family in HELDOUT else "other")


def load_tasks(name: str) -> list:
    with open(os.path.join(set_dir(name), "tasks.pkl"), "rb") as fh:
        return pickle.load(fh)


def load_support(name: str, cond: str) -> list:
    with open(os.path.join(set_dir(name), f"support_{cond}.pkl"), "rb") as fh:
        return pickle.load(fh)


def cmd_prepare(a) -> None:
    sp = SETS[a.set]
    tasks = M.eval_task_set(list(FAMILIAR) + list(HELDOUT), sp["n"], sp["offset"], workers=a.workers)
    d = set_dir(a.set)
    with open(os.path.join(d, "tasks.pkl"), "wb") as fh:
        pickle.dump(tasks, fh)
    info = {"set": a.set, "offset": sp["offset"], "tasks": [t.id for t in tasks],
            "failures": [str(f)[:200] for f in M.eval_task_set.failures]}
    for cond, kw in CONDITIONS.items():
        t0 = time.time()
        eps = []
        for i in range(0, len(tasks), 16):
            eps += P.collect_support(tasks[i:i + 16], k_max=64, n_queries=12, design_seed=DESIGN_SEED[a.set], **kw)
        with open(os.path.join(d, f"support_{cond}.pkl"), "wb") as fh:
            pickle.dump(eps, fh)
        info[f"support_{cond}_s"] = time.time() - t0
    with open(os.path.join(d, "prepare.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info)[:2000])


# ============================================================================ baseline fitting

def training_design_episodes(seed: int, n_episodes: int, domain: str = "scm") -> list:
    """Design-policy episodes on training families from the training seed range (stream start disjoint from CAM's
    streams: 900,000 + 10,000 * seed), plain and condition-D presentations alternating."""
    st = M.TaskStream(M.domain_families(domain), start=900_000 + 10_000 * seed, seed=seed)
    out = []
    while len(out) < n_episodes:
        tasks = [t for t in st.batch(16) if t.horizon > 64]
        if not tasks:
            continue
        cond = "D" if (len(out) // 16) % 2 else "A"
        out += P.collect_support(tasks, k_max=64, n_queries=1, design_seed=seed, condition=cond, ks=())
    return out[:n_episodes]


def cmd_fit(a) -> None:
    d = os.path.join(ROOT, "baselines", f"seed{a.seed}")
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    eps = training_design_episodes(a.seed, a.episodes)
    info = {"seed": a.seed, "episodes": len(eps), "ticks": int(sum(len(e.Y) for e in eps)),
            "collect_s": time.time() - t0, "families": sorted({e.task.family for e in eps})}
    nig = BL.NIGPredictor(meta=True)
    info["nig_meta"] = nig.fit_meta(eps, steps=a.nig_steps, seed=a.seed)
    with open(os.path.join(d, "nig_meta.json"), "w") as fh:
        json.dump(nig.state_dict(), fh)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for kind in ("gru", "attn"):
        sp = BL.SeqPredictor(kind, device=dev, seed=a.seed)
        info[f"seq_{kind}"] = sp.fit(eps, steps=a.seq_steps, seed=a.seed)
        torch.save(sp.state_dict(), os.path.join(d, f"seq_{kind}.pt"))
    with open(os.path.join(d, "fit.json"), "w") as fh:
        json.dump(info, fh, indent=1, default=float)
    print(json.dumps(info, default=float)[:3000])


def load_predictor(name: str, seed: int):
    d = os.path.join(ROOT, "baselines", f"seed{seed}")
    if name == "nig_exact":
        return BL.NIGPredictor(meta=False)
    if name == "nig_meta":
        p = BL.NIGPredictor(meta=True)
        with open(os.path.join(d, "nig_meta.json")) as fh:
            p.load_state_dict(json.load(fh))
        return p
    if name in ("seq_gru", "seq_attn"):
        p = BL.SeqPredictor(name[4:], device="cuda" if torch.cuda.is_available() else "cpu", seed=seed)
        p.load_state_dict(torch.load(os.path.join(d, f"{name}.pt")))
        return p
    raise ValueError(name)


# ============================================================================ CAM loading

def cam_stack(learner: str, ckpt: Optional[str], seed: int, sem=None):
    """Weights-only CAM in this fresh process (stack.load_condition "B": optimiser / fast state / experience access /
    counters reset) or a fresh untrained CAM (seeded init). No archive, replay or memory index is attached."""
    st = S.build_stack(S.experiment_config(), experience=False, train=False, seed=seed, sem=sem)
    info = {"learner": learner, "seed": seed}
    if learner == "cam":
        r = S.load_condition(st, ckpt, "B")
        info.update({"ckpt": ckpt, "load": {k: v for k, v in r.items() if k != "config"}})
    info["state_after_load"] = S.state_report(st)
    return st, info


def _rows_out(path: str, rows: list) -> None:
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=float) + "\n")


def cmd_predict(a) -> None:
    d = set_dir(a.set)
    rows = []
    t0 = time.time()
    meta = {"learner": a.learner, "tag": a.tag or a.learner, "seed": a.seed, "set": a.set}
    if a.learner.startswith("cam"):
        st, info = cam_stack(a.learner, a.ckpt, a.seed)
        meta["load"] = info
    for cond in a.conds.split(","):
        eps = load_support(a.set, cond)
        if a.learner.startswith("cam"):
            preds = P.cam_predict(st, eps)
            get = lambda i, k: (*preds[(i, k)], None)              # noqa: E731
        else:
            pr = load_predictor(a.learner, a.seed)
            get = lambda i, k: pr.predict(eps[i], k)                # noqa: E731
        for i, ep in enumerate(eps):
            for k in P.K_SET:
                m, v, dof = get(i, k)
                sc = P.score_queries(ep, k, m, v, dof)
                rows.append({**meta, "cond": cond, "task": ep.task.id, "family": ep.task.family,
                             "split": split_of(ep.task.family), "k": k, **sc})
            if cond == "C" and a.learner == "nig_exact":
                for k in P.K_SET:
                    m, v, dof = BL.NIGPredictor().predict_with_notes(ep, k)
                    rows.append({**meta, "tag": "nig_exact_parsed_notes", "learner": "nig_exact_parsed_notes",
                                 "cond": cond, "task": ep.task.id, "family": ep.task.family,
                                 "split": split_of(ep.task.family), "k": k, **P.score_queries(ep, k, m, v, dof)})
    meta["wall_s"] = time.time() - t0
    meta["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0
    _rows_out(os.path.join(d, "predict.jsonl"), rows)
    _rows_out(os.path.join(d, "runs.jsonl"), [{"cmd": "predict", **meta}])
    print(f"[exp1] predict {meta['tag']} seed {a.seed}: {len(rows)} rows in {meta['wall_s']:.0f}s")


# ============================================================================ decisions (condition B / D)

def cam_decide(st, tasks: list, *, cond: str, seed: int, ks=P.K_SET, batch: int = 16, imagination: bool = False,
               max_rounds: int = 100_000) -> tuple:
    """Frozen CAM acting (no updates; continual_updates irrelevant since update() is never called) on `tasks`.
    Returns (episode rows, own-trajectory prediction rows)."""
    from dataclasses import replace
    if imagination != st.model.cfg.use_imagination:
        st.model.cfg = replace(st.model.cfg, use_imagination=imagination)
    rows, prows = [], []
    for i in range(0, len(tasks), batch):
        part = tasks[i:i + batch]
        w = M.make_world(part, seed=seed * 1000 + i)
        dp = M.DelayedPartial(w, delay=3, hide_frac=0.5, seed=seed * 1000 + i) if cond == "D" else None
        rec = P.TrajectoryRecorder(w, ks=ks, n_queries=12, seed=seed, delayed=dp, condition=cond)
        t0 = time.time()
        res = M.run_batch(st.agent, w, learn=False, delayed=dp, on_round=rec, max_rounds=max_rounds)
        wall = time.time() - t0
        eps = rec.episodes()
        preds = P.cam_predict(st, eps, ks=[k for k in ks if all(k in e.queries for e in eps)])
        for j, (r, ep) in enumerate(zip(res, eps)):
            rows.append({"task": r.task_id, "family": r.family, "split": split_of(r.family), "cond": cond,
                         "score": r.score, "success": r.success, "windows": r.windows, "decisions": r.decisions,
                         "tools": r.tools, "ticks": len(r.nets), "wall_s_batch": wall, "nets_raw": list(r.nets)})
            for (jj, k), (m, v) in preds.items():
                if jj == j:
                    prows.append({"task": r.task_id, "family": r.family, "split": split_of(r.family), "cond": cond,
                                  "k": k, **P.score_queries(ep, k, m, v, None)})
    return rows, prows


def cmd_decide(a) -> None:
    d = set_dir(a.set)
    tasks = load_tasks(a.set)
    if a.limit:
        tasks = tasks[:a.limit] if a.split == "all" else [t for t in tasks if split_of(t.family) == a.split][:a.limit]
    elif a.split != "all":
        tasks = [t for t in tasks if split_of(t.family) == a.split]
    meta = {"learner": a.learner, "tag": a.tag or a.learner, "seed": a.seed, "set": a.set}
    t0 = time.time()
    rows, prows = [], []
    for cond in a.conds.split(","):
        if a.learner.startswith("cam"):
            st, info = cam_stack(a.learner, a.ckpt, a.seed)
            meta["load"] = info
            r, p = cam_decide(st, tasks, cond=cond, seed=a.seed, imagination=a.imagination)
            rows += [{**meta, **x} for x in r]
            prows += [{**meta, **x} for x in p]
        else:
            lr = numeric_learner(a.learner, a.seed, a.ckpt)
            lr.freeze(True)
            for i in range(0, len(tasks), 16):
                part = tasks[i:i + 16]
                out = BL.run_numeric_episodes(lr, part, learn=False, explore=False,
                                              delay=3 if cond == "D" else 0, hide_frac=0.5 if cond == "D" else 0.0,
                                              seed=a.seed)
                for o, t in zip(out, part):
                    rows.append({**meta, "task": o["task"], "family": o["family"], "split": split_of(o["family"]),
                                 "cond": cond, "score": o["score"], "success": o["success"],
                                 "windows": M.window_scores(t, o["nets"]), "ticks": len(o["nets"]),
                                 "nets_raw": list(o["nets"])})
    meta["wall_s"] = time.time() - t0
    meta["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0
    _rows_out(os.path.join(d, "decide.jsonl"), rows)
    _rows_out(os.path.join(d, "decide_predict.jsonl"), prows)
    _rows_out(os.path.join(d, "runs.jsonl"), [{"cmd": "decide", **meta}])
    print(f"[exp1] decide {meta['tag']} seed {a.seed}: {len(rows)} episodes in {meta['wall_s']:.0f}s")


# ============================================================================ numeric learners

def numeric_learner(name: str, seed: int, ckpt: Optional[str] = None):
    dev = "cuda" if (name.startswith("cam7") and torch.cuda.is_available()) else "cpu"
    lr = BL.decision_learner(name, seed=seed, device=dev)
    if ckpt:
        lr.load(ckpt)
    return lr


def cmd_ntrain(a) -> None:
    """Train a Crucible numeric learner on the training stream (same families as CAM's source domain, its own
    training seed range 600,000 + 10,000 * seed) for `ticks` environment ticks, with Crucible's batch runner."""
    from crucible.research.accounting import Cost, Meter
    from crucible.research.learner import EnvSpec
    from crucible.research.protocols import _run_batch
    d = os.path.join(S.RUNS_ROOT, "numeric", f"{a.learner}_seed{a.seed}")
    os.makedirs(d, exist_ok=True)
    lr = numeric_learner(a.learner, a.seed)
    lr.freeze(False)
    cost = Cost()
    meter = Meter(lr, cost)
    spec = EnvSpec(max_levers=32, max_gauges=32)
    stream = M.TaskStream(M.domain_families(a.domain), start=600_000 + 10_000 * a.seed + (0 if a.domain == "scm"
                                                                                          else 5_000), seed=a.seed)
    t0 = time.time()
    ticks, eps = 0, 0
    try:
        while ticks < a.ticks:
            batch = stream.batch(16)
            _run_batch(lr, batch, spec, meter, learn=True, explore=True, segments=4, update_every=a.update_every)
            ticks += sum(int(t.horizon) for t in batch)
            eps += len(batch)
    finally:
        meter.close()
    lr.save(os.path.join(d, "learner"))
    info = {"learner": a.learner, "seed": a.seed, "ticks": ticks, "episodes": eps, "wall_s": time.time() - t0,
            "usage": lr.usage().to_dict(), "domain": a.domain}
    with open(os.path.join(d, "train.json"), "w") as fh:
        json.dump(info, fh, indent=1, default=float)
    print(json.dumps(info, default=float)[:1000])


# ============================================================================ report

def _matrix(rows: list, metric: str, tasks: list, key=lambda r: r["seed"]) -> np.ndarray:
    seeds = sorted({key(r) for r in rows})
    ti = {t: j for j, t in enumerate(tasks)}
    M_ = np.full((len(seeds), len(tasks)), np.nan)
    for r in rows:
        v = r.get(metric)
        if v is None or r["task"] not in ti:
            continue
        M_[seeds.index(key(r)), ti[r["task"]]] = float(v)
    return M_


def summarise_predict(rows: list, tasks: list, metric: str = "logscore", reference: str = "cam") -> dict:
    out = {}
    tags = sorted({r["tag"] for r in rows})
    for cond in sorted({r["cond"] for r in rows}):
        for split in ("familiar", "heldout"):
            tl = [t for t in tasks if split_of(t.split(":")[0]) == split]
            sub = [r for r in rows if r["cond"] == cond and r["split"] == split]
            block = {}
            mats = {}
            for tag in tags:
                rr = [r for r in sub if r["tag"] == tag]
                if not rr:
                    continue
                per_k = {k: _matrix([r for r in rr if r["k"] == k], metric, tl) for k in P.K_SET}
                mats[tag] = per_k
                block[tag] = {"curve": ST.curve(per_k, n_boot=1000), "area": ST.two_level(ST.area(per_k))}
            if reference in mats:
                for tag, per_k in mats.items():
                    if tag == reference:
                        continue
                    block[f"{reference}-minus-{tag}"] = {
                        "area": ST.paired(ST.area(mats[reference]), ST.area(per_k)),
                        "k": {str(k): ST.paired(mats[reference][k], per_k[k], n_boot=1000) for k in P.K_SET}}
            out[f"{cond}/{split}"] = block
    return out


def cmd_report(a) -> None:
    d = set_dir(a.set)
    tasks = [t.id for t in load_tasks(a.set)]
    rep = {"set": a.set}
    p = os.path.join(d, "predict.jsonl")
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p)]
        for metric in ("logscore", "regret", "sqerr", "cover90"):
            rep[f"predict/{metric}"] = summarise_predict(rows, tasks, metric, reference=a.reference)
    p = os.path.join(d, "decide.jsonl")
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p)]
        dec = {}
        for cond in sorted({r["cond"] for r in rows}):
            for split in ("familiar", "heldout"):
                tl = [t for t in tasks if split_of(t.split(":")[0]) == split]
                sub = [r for r in rows if r["cond"] == cond and r["split"] == split]
                blk = {}
                mats = {}
                for tag in sorted({r["tag"] for r in sub}):
                    rr = [r for r in sub if r["tag"] == tag]
                    wins = {}
                    for w in M.WINDOWS:
                        for r in rr:
                            r[f"w{w}"] = (r.get("windows") or {}).get(str(w))
                        wins[w] = _matrix(rr, f"w{w}", tl)
                    sc = _matrix(rr, "score", tl)
                    for r in rr:
                        r["succ"] = None if r.get("success") is None else float(bool(r["success"]))
                    mats[tag] = sc
                    blk[tag] = {"score": ST.two_level(sc), "success": ST.two_level(_matrix(rr, "succ", tl)),
                                "windows": ST.curve(wins, n_boot=1000), "area": ST.two_level(ST.area(wins)),
                                "n_episodes": len(rr)}
                if a.reference in mats:
                    for tag, m in mats.items():
                        if tag != a.reference:
                            blk[f"{a.reference}-minus-{tag}"] = {"score": ST.paired(mats[a.reference], m)}
                dec[f"{cond}/{split}"] = blk
        rep["decide"] = dec
    p = os.path.join(d, "decide_predict.jsonl")
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p)]
        rep["own_trajectory_predict/logscore"] = summarise_predict(rows, tasks, "logscore", reference=a.reference)
        rep["own_trajectory_predict/regret"] = summarise_predict(rows, tasks, "regret", reference=a.reference)
    out = os.path.join(d, f"report_{a.reference}.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    print(f"[exp1] report -> {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--set", default="dev"); p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("fit"); p.add_argument("--seed", type=int, default=0); p.add_argument("--episodes", type=int, default=320)
    p.add_argument("--nig-steps", type=int, default=150); p.add_argument("--seq-steps", type=int, default=1500)
    for name in ("predict", "decide"):
        p = sub.add_parser(name)
        p.add_argument("--set", default="dev"); p.add_argument("--learner", required=True)
        p.add_argument("--ckpt", default=None); p.add_argument("--seed", type=int, default=0)
        p.add_argument("--tag", default=None)
        p.add_argument("--conds", default="A,C,D" if name == "predict" else "B")
        if name == "decide":
            p.add_argument("--imagination", action="store_true")
            p.add_argument("--split", default="all"); p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("ntrain"); p.add_argument("--learner", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ticks", type=int, default=20000); p.add_argument("--update-every", type=int, default=16)
    p.add_argument("--domain", default="scm")
    p = sub.add_parser("report"); p.add_argument("--set", default="dev"); p.add_argument("--reference", default="cam")
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "fit": cmd_fit, "predict": cmd_predict, "decide": cmd_decide, "ntrain": cmd_ntrain,
     "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
