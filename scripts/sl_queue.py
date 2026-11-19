"""Capacity-gated, crash-safe experiment queue for the superlearner research track.

    python scripts/sl_queue.py --queue configs/sl/queue_phase1.yaml

Before a job's first launch, its *exact* production command is run start-to-finish with budget-only overrides
(``smoke_set``) in a scratch run directory; the job only starts if that smoke run reaches phase "done" with no
diagnostic errors. Smoke results are cached by (code commit, config file, job overrides).

A job starts only when: its dependencies are done, fewer than ``max_jobs`` queue jobs run, the GPU has at least
``min_free_mib`` free, the disk has at least ``min_free_disk_gb`` free, at least ``min_free_pids`` processes/threads remain under the container limit, and
capacity persisted for two ticks. A job also needs its own footprint free on top of that reserve (``pids_need`` per job,
default ``pids_per_job``), because some jobs spawn world servers after they start (sl-ops-v16).

On an unexpected exit the job is restarted from its latest checkpoint, unless the same crash signature has
repeated (``CrashLedger``): then it is stopped with ``CRASH_REPORT.md`` in its run directory.
``wait_for`` (list of checkpoint directories): the job starts only once each exists with its ``DONE`` marker (e.g. a
source lifetime's ``life_1000000``), so transfer runs can start while their source lifetime is still training. If
the run that would write a checkpoint (``<run>/ckpt/<name>``) has finished without it, the job is ``skipped``.
Stop new starts: ``touch <runs_root>/QUEUE_STOP``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentrs.runtime_safety import START_MARKER, CrashLedger, config_hash, disk_free_gb, smoke_run, state_phase_done  # noqa: E402

PY = "/workspace/agentrs/.venv/bin/python"
MAIN = Path("/workspace/agentrs")


def sh(*cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def alive(pid) -> bool:
    try:
        pid = int(pid)
        os.kill(pid, 0)
    except (OSError, TypeError, ValueError):
        return False
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def free_mib() -> int:
    try:
        return int(sh("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits").split()[0])
    except (IndexError, ValueError):
        return 0


def free_pids() -> int:
    """Headroom under the container's process+thread limit (cgroup pids.max = 1280 here). Each Melting Pot scenario
    server holds ~100 threads; at the limit, any thread creation in any process fails (2026-11-19: 49 failures)."""
    for d in ("/sys/fs/cgroup/pids", "/sys/fs/cgroup"):
        try:
            mx = Path(d, "pids.max").read_text().strip()
            cur = int(Path(d, "pids.current").read_text())
            return (10 ** 6 if mx == "max" else int(mx)) - cur
        except (OSError, ValueError):
            continue
    return 10 ** 6


def phase(rd: Path) -> str:
    try:
        return json.loads((rd / "state.json").read_text()).get("phase", "")
    except (OSError, ValueError):
        return ""


def ckpt_ready(p: Path) -> bool:
    return (p / "DONE").exists()


def unreachable(p: Path) -> bool:
    """A checkpoint ``<run>/ckpt/<name>`` that will never appear: its run has finished without writing it."""
    return not ckpt_ready(p) and p.parent.name == "ckpt" and phase(p.parent.parent) == "done"


class Queue:
    def __init__(self, path: str):
        self.qpath = Path(path)
        self.q = yaml.safe_load(self.qpath.read_text())
        self.root = Path(self.q["runs_root"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.sp = self.root / "queue_state.json"
        self.st = json.loads(self.sp.read_text()) if self.sp.exists() else {}
        self.ledger = CrashLedger(repeat_limit=self.q.get("crash_repeat_limit", 2), max_starts=self.q.get("max_starts", 4))
        self.src = Path(self.q.get("src", ROOT / "src"))

    def src_of(self, j: dict) -> Path:
        """A job may pin its own frozen code snapshot (``src``); otherwise the queue's."""
        return Path(j.get("src") or self.src)

    def log(self, msg: str) -> None:
        line = f"[sl_queue {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.root / "queue.log", "a") as f:
            f.write(line + "\n")

    def save(self) -> None:
        tmp = self.sp.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.st, indent=1))
        os.replace(tmp, self.sp)

    def env(self, j: dict) -> dict:
        e = {**os.environ, "PYTHONPATH": str(self.src_of(j)), "OMP_NUM_THREADS": str(self.q.get("threads", 4)),
             "MKL_NUM_THREADS": str(self.q.get("threads", 4)), "PYTHONUNBUFFERED": "1",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HUB_OFFLINE": "1"}
        e.update({k: str(v) for k, v in (self.q.get("env") or {}).items()})
        return e

    def cmd(self, j: dict, run_dir: Path, extra: dict | None = None) -> list:
        sets = {**(j.get("set") or {}), **(extra or {}), "run_dir": str(run_dir)}
        return [PY, "-u", "-m", self.q.get("module", "agentrs.sl.train"), "--config", j["config"], "--set",
                *[f"{k}={json.dumps(v)}" for k, v in sets.items()]]

    def smoke(self, j: dict) -> bool:
        s = self.st[j["name"]]
        src = self.src_of(j)
        commit = sh("git", "-C", str(src.parent), "rev-parse", "HEAD").strip()
        # the random seed changes no code path: jobs that differ only by seed share one smoke test
        shape = {k: v for k, v in (j.get("set") or {}).items() if k != "seed"}
        key = config_hash(commit, (src.parent / j["config"]).read_text(), shape, self.q.get("smoke_set") or {})
        cache = self.st.setdefault("_smoke_cache", {})
        if cache.get(key, {}).get("ok"):
            s["smoke"] = cache[key]
            return True
        rd = self.root / "_smoke" / j["name"]
        self.log(f"smoke test of {j['name']}: exact production command with budget-only overrides")
        res = smoke_run(self.cmd(j, rd, self.q.get("smoke_set")), cwd=str(src.parent), env=self.env(j), run_dir=rd,
                        done_check=state_phase_done, timeout_s=self.q.get("smoke_timeout_s", 1800))
        res["key"] = key
        cache[key] = res
        s["smoke"] = res
        self.log(f"smoke {j['name']}: {'OK' if res['ok'] else 'FAILED'} ({res['detail']}, {res['seconds']} s)")
        return res["ok"]

    def tick(self) -> bool:
        q, st = self.q, self.st
        jobs = q["jobs"]
        for j in jobs:
            s = st.setdefault(j["name"], {"status": "pending", "starts": 0})
            rd = self.root / j["name"]
            if s["status"] == "running" and not alive(s.get("pid")):
                if phase(rd) == "done":
                    s.update(status="done", finished_at=time.time())
                    self.log(f"{j['name']} finished")
                elif self.ledger.record(s, rd, rd / "stdout.log"):
                    s["status"] = "crash_stopped"
                    self.log(f"{j['name']} STOPPED after repeated crashes: see {rd}/CRASH_REPORT.md")
                else:
                    s["status"] = "pending"
                    self.log(f"{j['name']} exited unexpectedly (phase {phase(rd)!r}); will resume from its latest checkpoint")
            if s["status"] == "pending" and s["starts"] == 0:
                gone = [str(p) for p in map(Path, j.get("wait_for") or []) if unreachable(p)]
                if gone:
                    s["status"] = "skipped"
                    self.log(f"{j['name']} skipped: its source finished without {', '.join(gone)}")
            if s["status"] == "done" and s.get("watcher_pid") and not alive(s["watcher_pid"]):
                s["watcher_pid"] = None
        running = sum(1 for j in jobs if st[j["name"]]["status"] == "running")
        fm, fd = free_mib(), disk_free_gb(str(self.root))
        load = os.getloadavg()[0]
        can = (running < q.get("max_jobs", 4) and fm >= q.get("min_free_mib", 8000) and fd >= q.get("min_free_disk_gb", 20)
               and load <= q.get("max_load", 1e9) and free_pids() >= q.get("min_free_pids", 40)
               and not (self.root / "QUEUE_STOP").exists())
        streak = (st.get("_tick", {}).get("streak", 0) + 1) if can else 0
        started = False
        if can and streak >= 2:
            for j in jobs:
                s = st[j["name"]]
                if s["status"] != "pending":
                    continue
                if any(st.get(dep, {}).get("status") != "done" for dep in (j.get("after") or [])):
                    continue
                if not all(ckpt_ready(Path(p)) for p in (j.get("wait_for") or [])):
                    continue
                # the job's own footprint (training threads, world servers it spawns after start, its backup watcher)
                # must fit on top of the reserve: a Melting Pot transfer spawns ~20-30 server processes after launch
                need = q.get("min_free_pids", 40) + int(j.get("pids_need", q.get("pids_per_job", 0)))
                if free_pids() < need:
                    continue
                if s["starts"] == 0 and q.get("smoke", True) and not j.get("no_smoke"):
                    if not self.smoke(j):
                        s["status"] = "smoke_failed"
                        self.save()
                        continue
                rd = self.root / j["name"]
                rd.mkdir(parents=True, exist_ok=True)
                with open(rd / "stdout.log", "a") as lf:
                    lf.write(f"{START_MARKER}{s['starts'] + 1} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    lf.flush()
                    p = subprocess.Popen(self.cmd(j, rd), cwd=str(self.src_of(j).parent), env=self.env(j), stdout=lf,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                s.update(status="running", pid=p.pid, starts=s["starts"] + 1, started_at=time.time(), run_dir=str(rd))
                if q.get("gh_backup", True) and not (s.get("watcher_pid") and alive(s["watcher_pid"])):
                    wenv = {**os.environ, "PYTHONPATH": str(MAIN / "src"), "PYTHONUNBUFFERED": "1"}
                    with open(rd / "gh_backup.log", "a") as wf:
                        w = subprocess.Popen(["nice", "-n", "5", PY, "-u", "-m", "agentrs.gh_backup", "watch", "--run", str(rd),
                                              "--every-h", str(q.get("backup_every_h", 3)), "--interval", "300"], cwd=str(MAIN),
                                             env=wenv, stdout=wf, stderr=subprocess.STDOUT, start_new_session=True)
                    s["watcher_pid"] = w.pid
                self.log(f"started {j['name']} (pid {p.pid}, start {s['starts']}); running {running + 1}/{q.get('max_jobs', 4)}, "
                         f"GPU free {fm} MiB, disk free {fd:.0f} GB")
                started = True
                break
        st["_tick"] = {"time": time.time(), "running": running, "free_mib": fm, "free_disk_gb": round(fd, 1), "load": round(load, 1),
                       "free_pids": free_pids(),
                       "streak": 0 if started else streak}
        self.save()
        return all(st[j["name"]]["status"] in ("done", "crash_stopped", "smoke_failed", "skipped") for j in jobs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--interval", type=int, default=60)
    a = ap.parse_args()
    qu = Queue(a.queue)
    qu.log(f"queue {a.queue}: {len(qu.q['jobs'])} jobs, max {qu.q.get('max_jobs', 4)} concurrent")
    while True:
        if qu.tick():
            qu.log("all jobs finished")
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
