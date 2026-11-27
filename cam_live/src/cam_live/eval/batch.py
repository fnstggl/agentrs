"""cam_live.eval.batch — run many independent ``cli run`` jobs as separate processes (CPU, bounded parallelism).

Each job is ``python -m cam_live.eval.cli run ...`` in its own process with its own run directory; results go to
``runs/cam_live/eval/results/<plan>/<condition>_<family>_s<seed>.json``. A job whose result file exists is skipped
(resumable). Plans are defined in ``PLANS`` and mirrored in ``docs/S/PROTOCOLS.md``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

from .experiments import EVAL_HOME

#: name -> list of job specs (condition, family, seeds, lifetimes, decisions, extra args)
PLANS: Dict[str, List[Dict[str, Any]]] = {
    "pilot": [dict(condition=c, family="F1", seeds=[900, 901], lifetimes=6, decisions=16)
              for c in ("E", "C", "D", "A", "B", "E_no_credit")],
    # the prespecified main CPU batch (docs/S/PROTOCOLS.md §2): debug scale, tiny backbone, F1, 8 x 16 decisions
    "main_cpu": ([dict(condition=c, family="F1", seeds=list(range(0, 10)), lifetimes=8, decisions=16, save=True)
                  for c in ("E", "D", "E_mix0", "E_unverified")]
                 + [dict(condition=c, family="F1", seeds=list(range(0, 10)), lifetimes=8, decisions=16)
                    for c in ("C", "E_no_credit")]
                 + [dict(condition=c, family="F1", seeds=list(range(0, 5)), lifetimes=8, decisions=16)
                    for c in ("A", "B")]),
}


def transfer_jobs(plan: str, writer_plan: str, seeds: List[int], *, frozen: bool) -> List[Dict[str, Any]]:
    """Transfer (``frozen=False``: T1, T2) or before/after (``frozen=True``: held-out F1) child runs of the saved
    E checkpoints of ``writer_plan``; evaluation agent seed = writer seed."""
    from .transfer import DEFAULT_PLAN
    jobs = []
    for s in seeds:
        root = os.path.join(EVAL_HOME, "ckpts", writer_plan, f"E_F1_s{s}")
        for d in (("F1",) if frozen else ("T1", "T2")):
            for v, name in (("initial", "export_pre"), ("experienced", "export_post")):
                jobs.append(dict(export=os.path.join(root, name), version=v, domain=d, seed=s,
                                 lifetimes=DEFAULT_PLAN[d]["lifetimes"], decisions=DEFAULT_PLAN[d]["decisions"],
                                 frozen=frozen))
    return jobs


def run_transfer_batch(name: str, writer_plan: str, seeds: List[int], *, frozen: bool, n_jobs: int,
                       threads: int) -> int:
    import json
    from concurrent.futures import ThreadPoolExecutor
    from .transfer import spawn, summarise
    res_dir = os.path.join(EVAL_HOME, "results", name)
    os.makedirs(res_dir, exist_ok=True)
    jobs = transfer_jobs(name, writer_plan, seeds, frozen=frozen)
    t0 = time.time()

    def one(j: Dict[str, Any]) -> Dict[str, Any]:
        out = os.path.join(res_dir, f"{j['version']}_{j['domain']}_s{j['seed']}.json")
        if os.path.exists(out):
            with open(out) as f:
                return json.load(f)
        r = spawn(j["export"], j["version"], j["domain"], j["seed"], lifetimes=j["lifetimes"],
                  decisions=j["decisions"], k=16, frozen=j["frozen"], device="cpu", threads=threads)
        if r.get("executed"):
            with open(out, "w") as f:
                json.dump(r, f, default=str)
        print(f"[{time.time() - t0:7.0f}s] {j['version']} {j['domain']} s{j['seed']}: "
              f"{'ok' if r.get('executed') else 'FAILED ' + str(r.get('stderr', ''))[-300:]}", flush=True)
        return r
    with ThreadPoolExecutor(n_jobs) as ex:
        runs = list(ex.map(one, jobs))
    with open(os.path.join(res_dir, "summary.json"), "w") as f:
        json.dump({"executed": True, "writer_plan": writer_plan, "seeds": seeds, "frozen": frozen,
                   "n_runs": len(runs), "n_failed": sum(1 for r in runs if not r.get("executed")),
                   "summary": summarise([r for r in runs if r.get("executed")])}, f, indent=1, default=str)
    return 0


def cmd_transfer_batch(a: argparse.Namespace) -> int:
    seeds = [int(x) for x in a.seeds.split(",")] if "," in a.seeds or "-" not in a.seeds else \
        list(range(int(a.seeds.split("-")[0]), int(a.seeds.split("-")[1]) + 1))
    return run_transfer_batch(a.name, a.writer_plan, seeds, frozen=a.frozen, n_jobs=a.jobs, threads=a.threads)


def jobs_of(plan: str) -> List[Dict[str, Any]]:
    out = []
    for spec in PLANS[plan]:
        for s in spec["seeds"]:
            out.append({**{k: v for k, v in spec.items() if k != "seeds"}, "seed": s})
    return out


def run_plan(plan: str, n_jobs: int, threads: int) -> int:
    res_dir = os.path.join(EVAL_HOME, "results", plan)
    log_dir = os.path.join(EVAL_HOME, "logs", plan)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    todo = []
    for j in jobs_of(plan):
        name = f"{j['condition']}_{j['family']}_s{j['seed']}"
        out = os.path.join(res_dir, name + ".json")
        if os.path.exists(out):
            continue
        cmd = [sys.executable, "-m", "cam_live.eval.cli", "run", "--condition", j["condition"], "--family",
               j["family"], "--seed", str(j["seed"]), "--lifetimes", str(j["lifetimes"]), "--decisions",
               str(j["decisions"]), "--out", out, "--threads", str(threads)]
        if j.get("save"):
            cmd += ["--save-dir", os.path.join(EVAL_HOME, "ckpts", plan, name)]
        if j.get("scale"):
            cmd += ["--scale", j["scale"]]
        todo.append((name, cmd))
    running: List[Any] = []
    failed = 0
    t0 = time.time()
    while todo or running:
        while todo and len(running) < n_jobs:
            name, cmd = todo.pop(0)
            log = open(os.path.join(log_dir, name + ".log"), "w")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ,
                                                                                    "OMP_NUM_THREADS": str(threads)})
            running.append((name, p, log))
        time.sleep(2.0)
        for item in list(running):
            name, p, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                failed += p.returncode != 0
                print(f"[{time.time() - t0:7.0f}s] {name}: exit {p.returncode} ({len(todo)} queued, "
                      f"{len(running)} running)", flush=True)
    return 1 if failed else 0


def cmd_batch(a: argparse.Namespace) -> int:
    return run_plan(a.plan, a.jobs, a.threads)


def add_commands(sub: Any) -> None:
    b = sub.add_parser("batch", help="run a prespecified plan of independent jobs")
    b.add_argument("--plan", required=True, choices=sorted(PLANS))
    b.add_argument("--jobs", type=int, default=8)
    b.add_argument("--threads", type=int, default=3)
    b.set_defaults(fn=cmd_batch)
    t = sub.add_parser("transfer-batch", help="fresh-process transfer / before-after runs of saved E checkpoints")
    t.add_argument("--name", required=True)
    t.add_argument("--writer-plan", dest="writer_plan", default="main_cpu")
    t.add_argument("--seeds", default="0-9")
    t.add_argument("--frozen", action="store_true")
    t.add_argument("--jobs", type=int, default=10)
    t.add_argument("--threads", type=int, default=3)
    t.set_defaults(fn=cmd_transfer_batch)
