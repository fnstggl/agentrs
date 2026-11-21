"""Command line for the Superlearner Lab.

    superlearner doctor                              hardware, models, isolation, Crucible, a tiny end-to-end check
    superlearner llm-serve [--model M]               start the frozen-LLM server (one process owns the GPU model)
    superlearner train --config C [--run-dir D]      warm start (optional) + reinforcement learning
    superlearner resume --experiment ID|DIR          continue an interrupted run from its last checkpoint
    superlearner evaluate --checkpoint P [...]       evaluate a checkpoint (or a baseline mode) on a split
    superlearner report --experiment ID|DIR          learning curves, AULC, costs, parameter-change evidence
    superlearner inspect-trajectory --trajectory T   print a recorded trajectory step by step
    superlearner lock-final --out FILE               write the sealed final evaluation task list
    superlearner meta-search --config C              run the Meta-Researcher
    superlearner experiment NAME                     run one of the numbered experiments (A-F)

(`python -m superlearner ...` works the same when the package is not installed.)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))            # superlearner/
RUNS = os.environ.get("SUPERLEARNER_RUNS", os.path.join(ROOT, "runs"))
DEFAULT_LLM = {"kind": "remote", "socket": os.path.join(RUNS, "llm.sock"), "cache": os.path.join(RUNS, "llm_cache.sqlite")}


def load_config(path: str) -> dict:
    import yaml
    cfg = yaml.safe_load(open(path)) or {}
    base = cfg.pop("base", None)
    if base:
        from .train.trainer import merge
        cfg = merge(load_config(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


def resolve_llm(cfg: dict) -> dict:
    llm = dict(cfg.get("llm") or {"kind": "none"})
    if llm.get("kind") == "remote":
        llm.setdefault("socket", DEFAULT_LLM["socket"])
        llm.setdefault("cache", DEFAULT_LLM["cache"])
        if not os.path.exists(llm["socket"]):
            raise SystemExit(f"the config needs the LLM server but {llm['socket']} does not exist; "
                             f"start it with `superlearner llm-serve`")
    cfg["llm"] = llm
    return cfg


def run_dir_of(x: str) -> str:
    return x if os.path.isdir(x) else os.path.join(RUNS, x)


# ------------------------------------------------------------------------------------------------ commands
def cmd_doctor(a) -> int:
    from .hardware import doctor
    rep = doctor(quick=a.quick)
    print(json.dumps(rep, indent=1, default=str))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, default=str)
    return 0 if rep["ok"] else 1


def cmd_llm_serve(a) -> int:
    from .llm.server import main as serve
    os.makedirs(RUNS, exist_ok=True)
    serve(["--model", a.model, "--socket", a.socket or DEFAULT_LLM["socket"], "--cache", a.cache or DEFAULT_LLM["cache"]])
    return 0


def cmd_train(a) -> int:
    from .train.trainer import Trainer
    cfg = resolve_llm(load_config(a.config))
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        d = cfg
        ks = k.split(".")
        for p in ks[:-1]:
            d = d.setdefault(p, {})
        d[ks[-1]] = json.loads(v) if v[:1] in "[{0123456789tfn-\"" else v
    name = a.name or cfg.get("name") or os.path.splitext(os.path.basename(a.config))[0]
    rd = a.run_dir or os.path.join(RUNS, name)
    tr = Trainer(cfg, rd, resume=a.resume)
    try:
        tr.run(a.iters)
    finally:
        tr.close()
    print(f"run directory: {rd}")
    return 0


def cmd_resume(a) -> int:
    from .train.trainer import Trainer
    rd = run_dir_of(a.experiment)
    cfg = json.load(open(os.path.join(rd, "config.json")))
    tr = Trainer(cfg, rd, resume=True)
    try:
        print(f"resumed at iteration {tr.iter} ({tr.env_decisions} decisions)")
        tr.run(a.iters)
    finally:
        tr.close()
    return 0


def cmd_evaluate(a) -> int:
    import torch
    from .train.trainer import Trainer
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
    else:
        cfg = load_config(a.config)
    cfg = resolve_llm(cfg)
    cfg["workers"] = a.workers if a.workers is not None else cfg.get("workers", 8)
    rd = a.run_dir or os.path.join(RUNS, "eval_" + time.strftime("%m%d_%H%M%S"))
    tr = Trainer(cfg, rd)
    try:
        if a.checkpoint:
            tr.load(a.checkpoint, weights=not a.reset_weights, memory=not a.reset_memory)
        if a.final:
            from .world.tasks import load_final
            tasks = load_final(a.final)
        else:
            tasks = tr.eval_tasks(a.episodes, a.split, a.arms.split(",") if a.arms else None)
        res = tr.evaluate(tasks, a.mode, a.tag or "evaluate")
    finally:
        tr.close()
    print(json.dumps(res, indent=1))
    return 0


def cmd_report(a) -> int:
    from .train.report import report
    rep = report(run_dir_of(a.experiment))
    print(json.dumps(rep, indent=1, default=str) if a.json else rep["text"])
    return 0


def cmd_inspect(a) -> int:
    path = a.trajectory
    if not os.path.exists(path):
        hits = sorted(glob.glob(os.path.join(RUNS, "*", "traj", f"*{path}*")))
        if not hits:
            raise SystemExit(f"no trajectory matching {path!r}")
        path = hits[-1]
    recs = json.load(gzip.open(path, "rt"))
    e = recs[min(a.episode, len(recs) - 1)]
    s = e["summary"]
    print(f"file: {path}  (episode {a.episode} of {len(recs)})")
    print(f"task {s['task']}  success={s['success']}  progress={s['best_progress']:.3f}  return={s['return']:.3f}  "
          f"decisions={s['decisions']}  mode={s.get('mode')}  policy={e.get('policy')}")
    print(f"agent: {s['agent']}  llm: {s['llm']}")
    for i, h in enumerate(e["steps"]):
        print(f"\n[{i}] t={h.get('t_after')}  p(head)={h.get('p_head', 0):.3f}  top={h.get('top_heads')}")
        if a.verbose:
            print(f"    obs: {h.get('obs')}")
        print(f"    {h.get('action')} {h.get('args')}")
        print(f"    -> {'ok' if h.get('ok') else 'FAILED (' + str(h.get('fclass')) + ')'} "
              f"{h.get('result') if h.get('ok') else h.get('feedback')}  reward={h.get('reward'):.3f}")
    return 0


def cmd_lock_final(a) -> int:
    from .world.tasks import FINAL_FAMILIES, lock_final, task_list
    arms = [("found", 0, "scratch"), ("fund", 0, "scratch"), ("apicall", 0, "scratch"), ("list", 0, "deployed"),
            ("deploy", 0, "coded"), ("service", 0, "listed"), ("service", 0, "deployed"), ("service", 0, "coded"),
            ("repair", 0, "scratch"), ("code", 0, "scratch"), ("deploy", 0, "scratch"), ("service", 0, "scratch")]
    specs = task_list([f"{f}/{l}/{s}" for f, l, s in arms], list(FINAL_FAMILIES), a.n, "final", 700_000)
    print(f"wrote {len(specs)} tasks to {a.out}; seal {lock_final(a.out, specs)}")
    return 0


def cmd_meta(a) -> int:
    from .meta.researcher import main as meta_main
    return meta_main(a)


def cmd_experiment(a) -> int:
    from .experiments import run_experiment
    return run_experiment(a.name, a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="superlearner", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("doctor"); p.add_argument("--quick", action="store_true"); p.add_argument("--out"); p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("llm-serve"); p.add_argument("--model", default="Qwen/Qwen3-14B"); p.add_argument("--socket"); p.add_argument("--cache")
    p.set_defaults(fn=cmd_llm_serve)
    p = sub.add_parser("train"); p.add_argument("--config", required=True); p.add_argument("--run-dir"); p.add_argument("--name")
    p.add_argument("--iters", type=int); p.add_argument("--resume", action="store_true"); p.add_argument("--set", action="append")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("resume"); p.add_argument("--experiment", required=True); p.add_argument("--iters", type=int); p.set_defaults(fn=cmd_resume)
    p = sub.add_parser("evaluate"); p.add_argument("--checkpoint"); p.add_argument("--config"); p.add_argument("--run-dir")
    p.add_argument("--mode", default="greedy", choices=["greedy", "sample", "random", "llm", "teacher:auto"])
    p.add_argument("--split", default="select"); p.add_argument("--episodes", type=int, default=48); p.add_argument("--arms")
    p.add_argument("--final"); p.add_argument("--tag"); p.add_argument("--workers", type=int)
    p.add_argument("--reset-weights", action="store_true"); p.add_argument("--reset-memory", action="store_true")
    p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("report"); p.add_argument("--experiment", required=True); p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_report)
    p = sub.add_parser("inspect-trajectory"); p.add_argument("--trajectory", required=True); p.add_argument("--episode", type=int, default=0)
    p.add_argument("--verbose", action="store_true"); p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("lock-final"); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=96); p.set_defaults(fn=cmd_lock_final)
    p = sub.add_parser("meta-search"); p.add_argument("--config", required=True); p.add_argument("--run-dir"); p.add_argument("--rounds", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--full-feedback", action="store_true",
                   help="screen every proposal of every pool (Experiment F), then develop and confirm the best")
    p.set_defaults(fn=cmd_meta)
    p = sub.add_parser("experiment"); p.add_argument("name"); p.add_argument("--config"); p.add_argument("--run-dir"); p.add_argument("--checkpoint")
    p.add_argument("--episodes", type=int); p.add_argument("--workers", type=int); p.set_defaults(fn=cmd_experiment)
    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
