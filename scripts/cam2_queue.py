"""CAM-v2 experiment queue: start controlled GPU experiments only as capacity frees up.

Never touches other runs. A job starts only when (a) fewer than ``max_with_qwen`` (while the Qwen-LoRA
trainer is RUNNING) / ``max_without_qwen`` CAM training processes use the GPU — counting a supervisor
CAM program that is still waiting to start as an occupied slot — (b) at least ``min_free_mib`` GPU
memory is free, (c) its ``after`` dependency has finished, and (d) the start deadline has not passed
and ``<runs>/cam2/QUEUE_STOP`` does not exist. At most one job starts per tick. Every job gets its own
GitHub Releases checkpoint watcher (initial, <= 3 h, new best, final). Jobs run in their own process
groups, so restarting this runner never kills them; it re-adopts them by PID.

    PYTHONPATH=src python scripts/cam2_queue.py --queue configs/cam2/queue.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

PY = "/workspace/agentrs/.venv/bin/python"
MAIN = Path("/workspace/agentrs")


def log(msg: str) -> None:
    print(f"[cam2_queue {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def sh(*a) -> str:
    return subprocess.run(a, capture_output=True, text=True).stdout


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def gpu_cam_procs() -> list[int]:
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            if "agentrs.cam.train" not in cmd and "agentrs.cam.v2.train" not in cmd and "hybrid_llm_baseline" not in cmd:
                continue
            env = (p / "environ").read_bytes()
            if b"CUDA_VISIBLE_DEVICES=\0" in env:  # CPU-only process
                continue
            out.append(int(p.name))
        except OSError:
            continue
    return out


def program_state(name: str) -> str:
    s = sh("supervisorctl", "status", name).split()
    return s[1] if len(s) > 1 else "UNKNOWN"


def free_mib() -> int:
    try:
        return int(sh("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits").split()[0])
    except (IndexError, ValueError):
        return 0


def phase(run_dir: Path) -> str:
    try:
        return json.loads((run_dir / "state.json").read_text()).get("phase", "")
    except (OSError, ValueError):
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--interval", type=int, default=90)
    a = ap.parse_args()
    q = yaml.safe_load(Path(a.queue).read_text())
    src = Path(q.get("src", Path(__file__).resolve().parents[1] / "src"))
    root = Path(q.get("runs_root", "/workspace/agentrs/runs/cam2"))
    root.mkdir(parents=True, exist_ok=True)
    sp = root / "queue_state.json"
    st = json.loads(sp.read_text()) if sp.exists() else {}
    deadline = dt.datetime.fromisoformat(q["start_deadline_utc"]).timestamp() if q.get("start_deadline_utc") else None
    reserved_programs = q.get("reserved_programs", [])
    log(f"queue {a.queue}: {len(q['jobs'])} jobs; limits {q['max_with_qwen']} (Qwen running) / {q['max_without_qwen']}")
    while True:
        jobs = q["jobs"]
        for j in jobs:
            s = st.setdefault(j["name"], {"status": "pending", "starts": 0})
            rd = root / j["name"]
            if s["status"] == "running" and not alive(s.get("pid")):
                if phase(rd) == "done" or (j.get("script") and (rd / "result.json").exists()):
                    s.update(status="done", finished_at=time.time())
                    log(f"{j['name']} finished")
                elif s["starts"] < 3:
                    s["status"] = "pending"
                    log(f"{j['name']} exited before completion (phase {phase(rd)!r}); will resume from its checkpoint")
                else:
                    s["status"] = "failed"
                    log(f"{j['name']} FAILED after {s['starts']} starts; see {rd}/stdout.log")
            if s["status"] == "done" and s.get("watcher_pid") and not alive(s["watcher_pid"]):
                s["watcher_pid"] = None
        procs = gpu_cam_procs()
        qwen = program_state("agentrs_trainer") == "RUNNING"
        reserved = 0
        cmds = []
        for pid in procs:
            try:
                cmds.append((Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore"))
            except OSError:
                pass
        for name, run, marker in reserved_programs:  # a supervisor CAM program that will start soon holds a slot
            if program_state(name) == "RUNNING" and phase(Path(run)) != "done" and not any(marker in c for c in cmds):
                reserved += 1
        limit = q["max_with_qwen"] if qwen else q["max_without_qwen"]
        used = len(procs) + reserved
        fm = free_mib()
        can = (used < limit and fm >= q.get("min_free_mib", 7000) and not (root / "QUEUE_STOP").exists()
               and (deadline is None or time.time() < deadline))
        streak = (st.get("_tick", {}).get("capacity_streak", 0) + 1) if can else 0
        if can and streak >= 2:  # capacity must persist across two ticks (e.g. not a gap between two arms of a run)
            for j in jobs:
                s = st[j["name"]]
                if s["status"] != "pending":
                    continue
                dep = j.get("after")
                if dep and st.get(dep, {}).get("status") != "done":
                    continue
                rd = root / j["name"]
                rd.mkdir(parents=True, exist_ok=True)
                env = {**os.environ, "PYTHONPATH": str(src), "OMP_NUM_THREADS": "6", "MKL_NUM_THREADS": "6",
                       "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HUB_OFFLINE": "1"}
                env.pop("CUDA_VISIBLE_DEVICES", None)
                if j.get("script"):  # evaluation-only script job (e.g. the frozen-LLM baseline); done when it writes its output
                    cmd = [PY, "-u", j["script"], *[str(x).replace("{run_dir}", str(rd)) for x in j.get("args", [])]]
                else:
                    sets = [f"run_dir={rd}"] + [f"{k}={json.dumps(v)}" for k, v in (j.get("set") or {}).items()]
                    cmd = [PY, "-u", "-m", "agentrs.cam.v2.train", "--config", j["config"], "--set", *sets]
                with open(rd / "stdout.log", "a") as lf:
                    p = subprocess.Popen(cmd, cwd=str(src.parent), env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
                s.update(status="running", pid=p.pid, starts=s["starts"] + 1, started_at=time.time(), run_dir=str(rd))
                if not j.get("script") and not (s.get("watcher_pid") and alive(s["watcher_pid"])):
                    wenv = {**os.environ, "PYTHONPATH": str(MAIN / "src"), "PYTHONUNBUFFERED": "1"}
                    with open(rd / "gh_backup.log", "a") as wf:
                        w = subprocess.Popen(["nice", "-n", "5", PY, "-u", "-m", "agentrs.gh_backup", "watch", "--run", str(rd),
                                              "--every-h", "3", "--interval", "300"], cwd=str(MAIN), env=wenv, stdout=wf,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    s["watcher_pid"] = w.pid
                log(f"started {j['name']} (pid {p.pid}); GPU CAM processes {used + 1}/{limit}, free {fm} MiB, Qwen {'on' if qwen else 'off'}")
                break
        st["_tick"] = {"time": time.time(), "gpu_cam_procs": len(procs), "reserved": reserved, "limit": limit, "free_mib": fm,
                       "qwen_running": qwen, "capacity_streak": 0 if (can and streak >= 2) else streak}
        tmp = sp.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        os.replace(tmp, sp)
        if all(st[j["name"]]["status"] in ("done", "failed") for j in q["jobs"]):
            log("all jobs finished")
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
