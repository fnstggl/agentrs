"""Command line of the Stage-1 OpenWorld path.

    python -m superlearner.cam_unified.stage1 run --run-dir DIR --task open_economy --scale dev \\
        --sem-model Qwen/Qwen3-1.7B --update-every 8 --max-decisions 240 --max-wall-min 18
    python -m superlearner.cam_unified.stage1 resume --run-dir DIR --max-wall-min 18      (a NEW process)
    python -m superlearner.cam_unified.stage1 resume --run-dir COPY --copy-from DIR --exit-after 3
    python -m superlearner.cam_unified.stage1 summarize --run-dir DIR --out evidence.json
    python -m superlearner.cam_unified.stage1 dashboard --run-dir DIR
    python -m superlearner.cam_unified.stage1 main-scale --out main_scale.json

GPU-heavy invocations (Qwen3-1.7B, main scale) go through the shared lock:
    flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock python -m superlearner.cam_unified.stage1 ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

# deterministic cuBLAS (SystemConfig.deterministic): must be in the environment before CUDA creates a handle
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _overrides(items) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        out[k] = tuple(val) if isinstance(val, list) else val
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m superlearner.cam_unified.stage1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--task", default="open_economy", choices=["open_economy", "rig"])
    r.add_argument("--task-index", type=int, default=0)
    r.add_argument("--days", type=float, default=None)
    r.add_argument("--scale", default="dev", help="base scale (dev, main, large, superlearner, ...) or DOM preset "
                                                  "(dom, dom_dev, dom_150m); build.build_agent")
    r.add_argument("--set", action="append", default=[], help="CamConfig override key=JSON (repeatable)")
    r.add_argument("--sem-model", default=None)
    r.add_argument("--device", default="cuda")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--update-every", type=int, default=8)
    r.add_argument("--checkpoint-every-days", type=float, default=0.25)
    r.add_argument("--think-s", type=float, default=600.0)
    r.add_argument("--imag-candidates", type=int, default=None)
    r.add_argument("--imag-horizon", type=int, default=None)
    r.add_argument("--imag-samples", type=int, default=None)
    r.add_argument("--seq-len", type=int, default=8)
    r.add_argument("--n-seqs", type=int, default=2)
    r.add_argument("--adapt-seq-len", type=int, default=17)
    r.add_argument("--fresh-max-steps", type=int, default=64)
    r.add_argument("--in-process", action="store_true", help="operator in this process (no hidden-state firewall)")
    for p in (r, sub.add_parser("resume")):
        p.add_argument("--max-decisions", type=int, default=None)
        p.add_argument("--stop-after", type=int, default=None)
        p.add_argument("--max-wall-min", type=float, default=None)
        p.add_argument("--kill-after", type=int, default=None)
        p.add_argument("--exit-after", type=int, default=None)
    res = sub.choices["resume"]
    res.add_argument("--run-dir", required=True)
    res.add_argument("--copy-from", default=None, help="copy this run directory to --run-dir first")
    s = sub.add_parser("summarize")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--out", default=None)
    d = sub.add_parser("dashboard")
    d.add_argument("--run-dir", required=True)
    pp = sub.add_parser("planner-probe")
    pp.add_argument("--run-dir", required=True)
    pp.add_argument("--out", required=True)
    m = sub.add_parser("main-scale")
    m.add_argument("--out", required=True)
    m.add_argument("--sem-model", default=None)
    m.add_argument("--work-dir", default=None)
    a = ap.parse_args(argv)
    cmdline = "python -m superlearner.cam_unified.stage1 " + " ".join(argv if argv is not None else sys.argv[1:])

    if a.cmd in ("run", "resume"):
        from .runner import RunSpec, run
        from .system import SystemConfig
        wall = None if a.max_wall_min is None else 60.0 * a.max_wall_min
        if a.cmd == "run":
            sc = SystemConfig(scale=a.scale, overrides=_overrides(a.set), sem_model=a.sem_model, device=a.device,
                              seed=a.seed, seq_len=a.seq_len, n_seqs=a.n_seqs, adapt_seq_len=a.adapt_seq_len,
                              fresh_max_steps=a.fresh_max_steps, imag_candidates=a.imag_candidates,
                              imag_horizon=a.imag_horizon, imag_samples=a.imag_samples)
            spec = RunSpec(run_dir=a.run_dir, task=a.task, task_index=a.task_index, days=a.days,
                           max_decisions=a.max_decisions, stop_after=a.stop_after, max_wall_s=wall,
                           kill_after=a.kill_after, exit_after=a.exit_after, update_every=a.update_every,
                           checkpoint_every_days=a.checkpoint_every_days, think_s=a.think_s,
                           isolated=not a.in_process, system=sc, command=cmdline)
        else:
            if a.copy_from:
                if os.path.exists(a.run_dir):
                    raise FileExistsError(a.run_dir)
                shutil.copytree(a.copy_from, a.run_dir, ignore=shutil.ignore_patterns("dashboard", "*.tmp*"))
            spec = RunSpec(run_dir=a.run_dir, resume=True, max_decisions=a.max_decisions, stop_after=a.stop_after,
                           max_wall_s=wall, kill_after=a.kill_after, exit_after=a.exit_after, command=cmdline)
        out = run(spec)
        brief = {k: out.get(k) for k in ("pid", "resumed", "stopped_chunk", "calls", "wall_s", "decisions_total",
                                         "version", "archive_count", "error", "peak_gpu_mb")}
        if out.get("resume_info"):
            ri = out["resume_info"]
            brief["resume"] = {k: ri.get(k) for k in ("version", "archive_count_checkpoint", "abandoned_events",
                                                       "optimizer_steps", "replay_cursor", "n_decisions")}
        print(json.dumps(brief, default=str))
        return 0
    if a.cmd == "summarize":
        from .evidence import summarize
        out = summarize(a.run_dir)
        txt = json.dumps(out, indent=1, default=str)
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w") as fh:
                fh.write(txt + "\n")
        print(txt[:4000])
        return 0
    if a.cmd == "dashboard":
        from ..observe.dashboard import build_dashboard
        print(build_dashboard(a.run_dir, title="CAM Unified Stage 1 (real OpenWorld)"))
        return 0
    if a.cmd == "planner-probe":
        from .probes import planner_probe_run
        out = planner_probe_run(os.path.abspath(a.run_dir), a.out)
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=str))
        return 0
    if a.cmd == "main-scale":
        from .probes import main_scale_probe
        out = main_scale_probe(sem_model=a.sem_model, work_dir=a.work_dir)
        out["command"] = cmdline
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
        print(json.dumps(out, indent=1, default=str)[:4000])
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
