"""End-to-end self-play throughput benchmark on the real environments and the real policy.

Runs each configuration for a fixed wall-clock budget on fresh training-family tasks (a dedicated
index range, so the curriculum's task indices are not consumed) with T = 1 sampling, as in training:

* ``lockstep N``: the original collector — N episodes advance in lock-step; every step waits for the
  slowest environment, and the batch shrinks as episodes finish.
* ``async N``: N persistent workers sharing one dynamically batched engine.

Reports per configuration: episodes / actions / verified successes per hour, generated tokens/s,
mean engine batch, queue-wait and latency percentiles, GPU utilisation, peak VRAM, CPU utilisation,
environment time share and cost per verified success. Benchmark episodes are stored with provenance
``benchmark`` and are not used for policy updates.
"""
from __future__ import annotations

import itertools
import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from agentrs.econ.core import make_task
from agentrs.econ.rollout import rollout_batch
from agentrs.selfplay.workers import WorkUnit, collect


class Sampler(threading.Thread):
    def __init__(self, period: float = 2.0):
        super().__init__(daemon=True)
        self.period, self.stop_flag = period, False
        self.gpu, self.cpu = [], []

    def run(self):
        import subprocess

        import psutil
        psutil.cpu_percent(None)
        while not self.stop_flag:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                self.gpu.append(float(out.strip().split("\n")[0]))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self.cpu.append(psutil.cpu_percent(None))
            time.sleep(self.period)


def task_source(families: list[str], start: int = 700_000) -> Callable[[], WorkUnit]:
    cyc = itertools.cycle(sorted(families))
    idx = {k: start for k in families}

    def nxt() -> WorkUnit:
        k = next(cyc)
        idx[k] += 1
        return WorkUnit(make_task(k, "train", idx[k]))
    return nxt


def run_bench(policy, version: str, families: list[str], configs: list[tuple[str, int]], seconds: float,
              out: Path, dph: float = 0.0, log=print, store: Optional[Callable] = None) -> dict:
    import torch

    results = []
    for mode, n in configs:
        src = task_source(families, start=700_000 + 10_000 * len(results))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        smp = Sampler()
        smp.start()
        t0 = time.time()
        if mode == "async":
            act = lambda reqs, greedy: policy.act(reqs, greedy=greedy, temperature=1.0)  # noqa: E731
            eps, st = collect(act, version, [], n, max_batch=max(n, 1) if n < policy.gen_batch else policy.gen_batch,
                              unit_source=src, wall_budget_s=seconds, provenance="benchmark")
            wall = st["wall_s"]
            actions = st["actions"]
            extra = {k: st[k] for k in ("mean_batch", "max_batch", "gen_tok_per_s", "engine_busy_frac", "queue_wait_p50_s",
                                        "queue_wait_p95_s", "latency_mean_s", "latency_p95_s", "env_time_s", "active_peak")}
            gen_tokens = st["gen_tokens"]
        else:
            eps, actions, gen_tokens, step_lat, batches = [], 0, 0, [], []

            def on_step(step, n_active):
                batches.append(n_active)
            while time.time() - t0 < seconds:
                tasks = [src().task for _ in range(n)]
                t1 = time.time()
                b = rollout_batch(tasks, lambda r: policy.act(r, greedy=False, temperature=1.0), version,
                                  provenance="benchmark", on_step=on_step)
                step_lat.append((time.time() - t1) / max(1, max(len(e.steps) for e in b)))
                eps += b
                actions += sum(len(e.steps) for e in b)
                gen_tokens += sum(len(s.token_ids) for e in b for s in e.steps)
            wall = time.time() - t0
            env_s = sum(s.elapsed_s for e in eps for s in e.steps)
            extra = {"mean_batch": sum(batches) / len(batches) if batches else 0, "max_batch": max(batches) if batches else 0,
                     "gen_tok_per_s": gen_tokens / wall, "latency_mean_s": sum(step_lat) / len(step_lat) if step_lat else None,
                     "env_time_s": env_s, "active_peak": n}
        smp.stop_flag = True
        smp.join(timeout=5)
        succ = sum(e.success for e in eps)
        rec = {"mode": mode, "workers": n, "wall_s": round(wall, 1), "episodes": len(eps),
               "episodes_per_h": len(eps) / wall * 3600, "actions_per_h": actions / wall * 3600,
               "successes": succ, "successes_per_h": succ / wall * 3600, "success_rate": succ / len(eps) if eps else None,
               "gen_tokens": gen_tokens, "gpu_util_mean": sum(smp.gpu) / len(smp.gpu) if smp.gpu else None,
               "cpu_util_mean": sum(smp.cpu) / len(smp.cpu) if smp.cpu else None,
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
               "env_time_frac": extra["env_time_s"] / max(1e-9, wall * (n if mode == "async" else 1)),
               "cost_per_success_usd": (dph * wall / 3600 / succ) if succ and dph else None, **extra}
        results.append(rec)
        if store:
            store(f"bench_{mode}_{n}", eps)
        log(f"bench {mode:8s} x{n:<3d}: {rec['episodes']} eps, {rec['actions_per_h']:.0f} actions/h, "
            f"{rec['successes_per_h']:.0f} succ/h, {rec['gen_tok_per_s']:.0f} tok/s, batch {rec['mean_batch']:.1f}, "
            f"GPU {rec['gpu_util_mean']}%, peak {rec['peak_vram_gb']:.1f} GB")
    base = next((r for r in results if r["mode"] == "lockstep" and r["workers"] == max(
        x["workers"] for x in results if x["mode"] == "lockstep")), None) if any(r["mode"] == "lockstep" for r in results) else None
    best = max((r for r in results if r["mode"] == "async"), key=lambda r: r["actions_per_h"], default=None)
    rep = {"time": time.time(), "version": version, "seconds_per_config": seconds, "results": results,
           "baseline": base and {k: base[k] for k in ("mode", "workers", "actions_per_h", "successes_per_h")},
           "best_async": best and {k: best[k] for k in ("mode", "workers", "actions_per_h", "successes_per_h")}}
    if base and best:
        rep["speedup_actions_per_h"] = best["actions_per_h"] / max(1e-9, base["actions_per_h"])
    out.write_text(json.dumps(rep, indent=1, default=str))
    return rep
