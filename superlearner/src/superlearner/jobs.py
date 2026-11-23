"""Run a list of jobs under resource limits, in dependency order, skipping what is already done.

    superlearner jobs configs/queue_experiments.yaml [--max-gpu N] [--max-cpu N] [--only a,b] [--dry]

Why this exists: one GPU holds the frozen language model (about 32 GB of a 46 GB card) and every training run needs
a few more GB during its updates. Starting all experiments at once made runs die of out-of-memory or stall. The queue
starts a job only when a GPU-training slot and enough CPU workers are free, and it can be run again after an
interruption: a job whose `done` file exists is skipped, everything else starts over (the experiment scripts
themselves continue from their checkpoints or partial results).

    max_gpu: 2          # jobs with `gpu: 1` in flight at the same time (the language-model server is not a job)
    max_cpu: 40         # sum of `cpu` (rollout workers) of the jobs in flight
    env: {NAME: value}  # added to every job's environment
    jobs:
      - name: b_sil_s1
        cmd: python -m superlearner resume --experiment exp_b_fewdemo_sil_s1
        gpu: 1
        cpu: 14
        done: runs/exp_b_fewdemo_sil_s1/ckpt/iter_0012.pt     # optional: exists <=> the job's result exists
        after: [other_job]                                    # optional: start only after these succeeded
        retries: 1                                            # optional: restarts after a failure (default 1)
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time


def _load(path: str) -> dict:
    import yaml
    return yaml.safe_load(open(path)) or {}


def gpu_free_mb() -> int | None:
    """Free memory of the first GPU as the driver reports it (it counts every process on the card, including ones
    this queue knows nothing about). None if it cannot be read."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=20)
        return int(r.stdout.split()[0]) if r.returncode == 0 else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def run_jobs(spec: dict, max_gpu: int | None = None, max_cpu: int | None = None, log_dir: str = "runs/queue",
             only: list[str] | None = None, dry: bool = False, poll_s: float = 5.0, verbose: bool = True,
             strict: bool | None = None, free_fn=gpu_free_mb) -> dict:
    strict = bool(spec.get("strict_order", False)) if strict is None else bool(strict)
    # `gpu_unit_mb`: what one unit of `gpu` needs. A GPU job then starts only if the card really has that much free
    # (plus `gpu_margin_mb`), after subtracting what jobs started in the last `gpu_settle_s` seconds have yet to allocate.
    unit_mb, margin_mb = int(spec.get("gpu_unit_mb", 0)), int(spec.get("gpu_margin_mb", 1500))
    settle_s = float(spec.get("gpu_settle_s", 180))
    waiting_since = None
    max_gpu = int(max_gpu if max_gpu is not None else spec.get("max_gpu", 1))
    max_cpu = int(max_cpu if max_cpu is not None else spec.get("max_cpu", os.cpu_count() or 8))
    jobs = [dict(j) for j in spec.get("jobs", []) if not only or j["name"] in only]
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise ValueError("job names must be unique")
    os.makedirs(log_dir, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (os.path.abspath("src"), env.get("PYTHONPATH", "")) if p)
    env.update({k: str(v) for k, v in (spec.get("env") or {}).items()})
    st = {j["name"]: {"state": "pending", "attempts": 0, "gpu": int(j.get("gpu", 0)), "cpu": int(j.get("cpu", 1))} for j in jobs}
    for j in jobs:
        if j.get("done") and os.path.exists(j["done"]):
            st[j["name"]]["state"] = "done"
            st[j["name"]]["note"] = "result already existed"
        if st[j["name"]]["cpu"] > max_cpu or st[j["name"]]["gpu"] > max_gpu:
            raise ValueError(f"job {j['name']} can never start under max_cpu={max_cpu}, max_gpu={max_gpu}")
    procs: dict[str, tuple] = {}
    status_path = os.path.join(log_dir, "status.json")

    def save():
        tmp = status_path + ".tmp"
        json.dump({"max_gpu": max_gpu, "max_cpu": max_cpu, "updated": time.strftime("%H:%M:%S"), "jobs": st}, open(tmp, "w"), indent=1)
        os.replace(tmp, status_path)

    def say(msg):
        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def stop_all(*_):
        for name, (p, _) in procs.items():
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except OSError:
                pass
            st[name]["state"] = "stopped"
        save()
        raise SystemExit(130)
    if not dry:
        signal.signal(signal.SIGTERM, stop_all)
        signal.signal(signal.SIGINT, stop_all)
    save()
    while True:
        for name in list(procs):                                     # finished jobs
            p, j = procs[name]
            rc = p.poll()
            if rc is None:
                continue
            del procs[name]
            s = st[name]
            s.update(returncode=rc, ended=time.strftime("%H:%M:%S"), wall_s=round(time.time() - s["t0"]))
            ok = rc == 0 and (not j.get("done") or os.path.exists(j["done"]))
            if ok:
                s["state"] = "done"
            elif s["attempts"] <= int(j.get("retries", 1)):
                s["state"] = "pending"
                s["note"] = f"attempt {s['attempts']} failed (exit {rc}); retrying"
            else:
                s["state"] = "failed"
            say(f"{name}: {s['state']} (exit {rc}, {s['wall_s']} s)")
            save()
        for j in jobs:                                                # blocked by a failed dependency
            s = st[j["name"]]
            if s["state"] == "pending" and any(st.get(d, {}).get("state") in ("failed", "blocked") for d in j.get("after", [])):
                s["state"] = "blocked"
                save()
        gpu = sum(st[n]["gpu"] for n in procs)
        cpu = sum(st[n]["cpu"] for n in procs)
        for j in jobs:                                                # start what fits, in the order given
            s = st[j["name"]]
            if s["state"] != "pending":
                continue
            if any(d in st and st[d]["state"] != "done" for d in j.get("after", [])):
                continue
            if gpu + s["gpu"] > max_gpu or cpu + s["cpu"] > max_cpu:
                if strict:
                    break       # keep the order: a large job must not be starved by smaller ones that always fit
                continue
            if j.get("done") and os.path.exists(j["done"]):           # produced meanwhile (by hand, or by another queue)
                s.update(state="done", note="result appeared before the job was started")
                save()
                continue
            if unit_mb and s["gpu"] and not dry:
                free = free_fn()
                if free is not None:
                    now = time.time()
                    settling = sum(st[n]["gpu"] * unit_mb for n in procs if now - st[n]["t0"] < settle_s)
                    if free - settling < s["gpu"] * unit_mb + margin_mb:
                        if waiting_since is None:
                            waiting_since = now
                            say(f"{j['name']}: waiting for GPU memory ({free} MiB free, {settling} MiB still settling)")
                        if strict:
                            break
                        continue
                waiting_since = None
            cmd = shlex.split(j["cmd"]) if isinstance(j["cmd"], str) else list(j["cmd"])
            if cmd and cmd[0] == "python":
                cmd[0] = sys.executable
            if dry:
                say(f"would start {j['name']}: {' '.join(cmd)}")
                s["state"] = "done"
                continue
            log = open(os.path.join(log_dir, j["name"] + ".log"), "ab")
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {s['attempts'] + 1}: {' '.join(cmd)}\n".encode())
            log.flush()
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            procs[j["name"]] = (p, j)
            s.update(state="running", pid=p.pid, started=time.strftime("%H:%M:%S"), t0=time.time(), attempts=s["attempts"] + 1)
            gpu += s["gpu"]
            cpu += s["cpu"]
            say(f"{j['name']}: started (pid {p.pid}; in flight: gpu {gpu}/{max_gpu}, cpu {cpu}/{max_cpu})")
            save()
        if not procs and not any(s["state"] == "pending" for s in st.values()):
            break
        if not dry:
            time.sleep(poll_s)
    save()
    say("queue finished: " + ", ".join(f"{n}={s['state']}" for n, s in st.items()))
    return st


def main(a) -> int:
    st = run_jobs(_load(a.file), a.max_gpu, a.max_cpu, a.log_dir or "runs/queue", a.only.split(",") if a.only else None, a.dry)
    return 0 if all(s["state"] == "done" for s in st.values()) else 1
