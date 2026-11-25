"""Chunked, resumable CAM Unified training on a guarded stream of real Crucible mechanism worlds.

    python -m superlearner.cam_unified.experiments.train --run DIR --domain scm --seed 0 \
        --budget-decisions 20000 --chunk-minutes 22 [--init CKPT --condition B|C|D|F]

Every invocation runs until the decision budget or the chunk's wall-clock limit is reached, then writes a checkpoint
(``DIR/ckpt/<decisions>/``: weights-only file + E's full-lifetime directory) and ``DIR/state.json`` (stream position,
counters, the checkpoint list), so the next invocation resumes exactly (full-lifetime restore: weights, optimiser,
replay / archive / memory index, RNG, counters). Per-episode results go to ``DIR/episodes.jsonl`` and per-update
losses to ``DIR/updates.jsonl``; compute (wall, CPU, GPU-section time, peak memory, env steps, decisions, simulated
days, updates) to ``DIR/compute.json``.

Training episodes are drawn per batch from three presentations of the same task distribution: plain (p_plain),
with predecessor notes (condition C, m ~ U{1..4}) and with hidden gauges + delayed money reports (condition D,
delay 3 ticks, half the gauges hidden), so the evaluation conditions of Experiment 1 are not out-of-distribution
presentations for the trained model.

``--init`` starts a run from another run's checkpoint under a spec-13.7 transfer condition (stack.load_condition):
B weights only (fresh optimiser, no experience access), C weights + optimiser, D weights + optimiser + the source's
historical replay (the source archive / index are copied into this run first, so the source run is never mutated),
F the full lifetime. Without ``--init`` the run starts fresh (condition A).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time

import torch

from . import mechrun as M
from . import stack as S


def _jsonl(path: str, row: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")


def _scalars(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, bool) or v is None:
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        elif torch.is_tensor(v) and v.numel() == 1:
            out[k] = float(v)
    return out


def copy_experience(src_ckpt: str, run_dir: str) -> None:
    """Copy the source run's archive and memory index (as recorded by the checkpoint) into this run's directory and
    rewrite a private copy of the lifetime pointers, so condition D trains on the source's history without mutating
    the source run."""
    life_dir = os.path.join(src_ckpt, "lifetime")
    with open(os.path.join(life_dir, "lifetime.json")) as fh:
        life = json.load(fh)
    for key, sub in (("archive", "archive"), ("index", "index")):
        p = life.get(key)
        if not p:
            continue
        dst = os.path.join(run_dir, sub)
        if not os.path.exists(dst):
            shutil.copytree(p["root"], dst)
        p["root"] = dst
    if life.get("replay"):
        life["replay"]["path"] = os.path.join(run_dir, "archive", os.path.basename(life["replay"]["path"]))
    own = os.path.join(run_dir, "init_ckpt", "lifetime")
    if os.path.exists(own):
        shutil.rmtree(own)
    shutil.copytree(life_dir, own)
    # file digests cover weights/optim/state only; lifetime.json itself may be rewritten
    with open(os.path.join(own, "lifetime.json"), "w") as fh:
        json.dump(life, fh, indent=1, default=str)
    for f in ("weights.safetensors", "tools.json", "config.json"):
        if os.path.exists(os.path.join(src_ckpt, f)):
            shutil.copy(os.path.join(src_ckpt, f), os.path.join(run_dir, "init_ckpt", f))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--domain", default="scm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--update-every", type=int, default=8)
    ap.add_argument("--budget-decisions", type=int, default=20000)
    ap.add_argument("--budget-episodes", type=int, default=0, help="stop after this many episodes (0 = no limit)")
    ap.add_argument("--save-every-batch", action="store_true", help="checkpoint after every batch (acquisition curves)")
    ap.add_argument("--chunk-minutes", type=float, default=22.0)
    ap.add_argument("--save-every-minutes", type=float, default=11.0)
    ap.add_argument("--p-plain", type=float, default=0.5)
    ap.add_argument("--p-notes", type=float, default=0.25)
    ap.add_argument("--init", default=None, help="source checkpoint dir (transfer runs)")
    ap.add_argument("--condition", default="A")
    ap.add_argument("--stream-start", type=int, default=None)
    ap.add_argument("--keep", type=int, default=4, help="keep this many most recent checkpoints (+ the milestones)")
    ap.add_argument("--milestones", default="", help="comma-separated decision counts whose checkpoints are kept")
    a = ap.parse_args(argv)
    t_start = time.time()
    run = os.path.abspath(a.run)
    os.makedirs(run, exist_ok=True)
    state_path = os.path.join(run, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else None
    from ..observe.accounting import ComputeMeter
    cfg = S.experiment_config()
    if state is None and a.init and a.condition.upper() in ("D", "F"):
        copy_experience(a.init, run)             # before the stack opens (creates) this run's archive / index
    st = S.build_stack(cfg, run_dir=run, seed=a.seed)
    meter = ComputeMeter(f"train:{os.path.basename(run)}", role="learner")
    prev_compute = {}
    stream = M.TaskStream(M.domain_families(a.domain), start=a.stream_start if a.stream_start is not None
                          else 100_000 * (a.seed + 1) + (0 if a.domain == "scm" else 50_000), seed=a.seed)
    if state is not None:
        info = S.load_condition(st, state["latest"], "F")
        stream.load_state_dict(state["stream"])
        prev_compute = state.get("compute", {})
        print(f"[train] resumed from {state['latest']} ({state['decisions']} decisions)", flush=True)
    else:
        state = {"decisions": 0, "episodes": 0, "updates": 0, "checkpoints": [], "domain": a.domain, "seed": a.seed,
                 "condition": a.condition.upper(), "init": a.init, "config": cfg.to_dict()}
        if a.init:
            c = a.condition.upper()
            src = a.init
            if c in ("D", "F"):
                src = os.path.join(run, "init_ckpt")          # the private copy made by copy_experience
            info = S.load_condition(st, src, c)
            state["init_report"] = {"load": {k: v for k, v in info.items() if k in ("restored", "condition", "version")},
                                    "after": S.state_report(st)}
            print(f"[train] init {c} from {a.init}: {state['init_report']}", flush=True)
            if c == "F":
                # a full-lifetime continuation keeps its counters; the transfer budget counts target decisions only
                state["source_version"] = int(st.agent.version)
    with open(os.path.join(run, "config.json"), "w") as fh:
        json.dump({"cfg": cfg.to_dict(), "args": vars(a), "model_kw": S.MODEL_KW,
                   "params": st.param_counts()}, fh, indent=1, default=list)
    rng = random.Random(a.seed * 7919 + state["episodes"])
    notes = M.PredecessorNotes()
    milestones = {int(x) for x in a.milestones.split(",") if x.strip()}
    last_save = time.time()
    ep_path, up_path = os.path.join(run, "episodes.jsonl"), os.path.join(run, "updates.jsonl")

    def on_update(u):
        state["updates"] += 1
        _jsonl(up_path, {"decisions_at_batch_start": state["decisions"], "update": state["updates"], "t": time.time() - t_start, "version": u.get("version"),
                         "losses": u.get("losses"), **{k: v for k, v in _scalars(u).items()
                                                        if k.startswith(("policy/", "adapt/", "world/", "trainer/"))}})

    def save():
        d = os.path.join(run, "ckpt", f"{state['decisions']:08d}")
        info = S.save_checkpoint(st, d, full=True)
        state["checkpoints"].append(d)
        state["latest"] = d
        state["stream"] = stream.state_dict()
        snap = meter.snapshot()
        tot = dict(prev_compute)
        for k in ("wall_s", "cpu_s", "gpu_s", "env_steps", "decisions", "sim_days", "updates"):
            tot[k] = float(prev_compute.get(k, 0.0)) + float(snap.get(k, 0.0))
        tot["peak_gpu_mb"] = max(float(prev_compute.get("peak_gpu_mb", 0.0)), float(snap.get("peak_gpu_mb", 0.0)))
        tot["peak_rss_mb"] = max(float(prev_compute.get("peak_rss_mb", 0.0)), float(snap.get("peak_rss_mb", 0.0)))
        state["compute"] = tot
        keep = set(state["checkpoints"][-a.keep:]) | {c for c in state["checkpoints"]
                                                      if int(os.path.basename(c)) in milestones}
        for c in list(state["checkpoints"]):
            if c not in keep and os.path.exists(c):
                shutil.rmtree(c, ignore_errors=True)
        state["checkpoints"] = [c for c in state["checkpoints"] if c in keep]
        tmp = state_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=1, default=str)
        os.replace(tmp, state_path)
        with open(os.path.join(run, "compute.json"), "w") as fh:
            json.dump(tot, fh, indent=1)
        print(f"[train] saved {d} decisions={state['decisions']} wall={time.time() - t_start:.0f}s "
              f"weights={info['weights'].get('tensors', '?')}", flush=True)

    deadline = t_start + 60 * a.chunk_minutes
    while state["decisions"] < a.budget_decisions and time.time() < deadline and \
            (not a.budget_episodes or state["episodes"] < a.budget_episodes):
        tasks = stream.batch(a.B)
        u = rng.random()
        cond = "plain" if u < a.p_plain else ("notes" if u < a.p_plain + a.p_notes else "delayed")
        w = M.make_world(tasks, seed=rng.randrange(2 ** 31))
        kw = {}
        if cond == "notes":
            nrng = random.Random(rng.randrange(2 ** 31))
            kw["notes"] = lambda world, nrng=nrng: [notes.notes(world, b, nrng.randint(1, 4), nrng)
                                                    for b in range(world.B)]
        elif cond == "delayed":
            kw["delayed"] = M.DelayedPartial(w, delay=3, hide_frac=0.5, seed=rng.randrange(2 ** 31))
        t0 = time.time()
        with meter.gpu():
            res = M.run_batch(st.agent, w, learn=True, update_every=a.update_every, meter=meter,
                              on_update=on_update, **kw)
        dec = sum(r.decisions for r in res)
        state["decisions"] += dec
        state["episodes"] += len(res)
        for r in res:
            _jsonl(ep_path, {"decisions_total": state["decisions"], "condition": cond, "task": r.task_id,
                             "family": r.family, "net": float(sum(r.nets)), "ticks": len(r.nets),
                             "return": r.returns, "decisions": r.decisions, "tools": r.tools,
                             "wall_s": time.time() - t0})
        print(f"[train] batch {cond} h={tasks[0].horizon} dec={dec} total={state['decisions']} "
              f"mean_return={sum(r.returns for r in res) / len(res):.3f} {time.time() - t0:.0f}s", flush=True)
        if a.save_every_batch or time.time() - last_save > 60 * a.save_every_minutes:
            save()
            last_save = time.time()
    if not state["checkpoints"] or state.get("latest") != os.path.join(run, "ckpt", f"{state['decisions']:08d}"):
        save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
