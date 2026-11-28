"""cam_live.eval.oe_batch — run Protocol v3 sequences under ONE shared-GPU lease with up to 3 children at a time.

    python -m cam_live.eval.oe_batch --plan v3 [--concurrency 3] [--max-gpu-gb 36]

The parent is the only lock holder (``runtime.gpu_cycle.GpuLease``, hold <= ``--hold-s``, default 810 s). Per lease
it starts up to ``concurrency`` children, each ``python -m cam_live.eval.oe_runner sequence ... --max-objectives 1
--no-lock --resume`` (exactly one objective of one sequence, then exit), waits for them, kills any child still
running when the hold limit is reached (recorded in that sequence's result as an interrupted objective), releases
the lease, yields, and repeats. Concurrency starts at 1 and is raised only while the measured peak GPU memory of the
children times the next concurrency stays below ``--max-gpu-gb``. ``DiskGuard`` is checked before every launch.

Jobs are prespecified in :data:`PLANS` (mirrors docs/S/PROTOCOLS.md v3.4); a job runs when its dependencies have
finished (X2 needs the X1 writer's exports). Fresh-process jobs (X2) start in an empty working directory with an
empty ``CAM_LIVE_HOME`` and only a copy of one weights-only export.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

SW_TRAIN = "SW:rle,SW:rpn,SW:merge,SW:topk"
X2_PLAN = "SW:unique,SW:base,MECH:11"
CU_PLAN = ",".join(["CU:store_entry"] * 6 + ["CU:record_item"] * 2)


def _jobs_v3() -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for s in (0, 1, 2):
        jobs.append(dict(name=f"x0/E_s{s}", condition="E", seed=s, plan="SW:rle,SW:rpn,MECH:7"))
    for s in (0, 1, 2):
        jobs.append(dict(name=f"x1/E_s{s}", condition="E", seed=s, plan=SW_TRAIN, save_exports=True))
    for s in (0, 1, 2):
        for v, exp in (("initial", "export_pre"), ("experienced", "export_post")):
            jobs.append(dict(name=f"x2/{v}_s{s}", condition="E", seed=s, plan=X2_PLAN, fresh=True,
                             from_export=f"@x1/E_s{s}/{exp}", depends=[f"x1/E_s{s}"]))
    for c in ("A", "B", "D", "C"):
        for s in (0, 1):
            jobs.append(dict(name=f"x3/{c}_s{s}", condition=c, seed=s, plan=SW_TRAIN))
    for s in (0, 1):
        jobs.append(dict(name=f"x4/skills_s{s}", condition="E", seed=s, plan=CU_PLAN))
        jobs.append(dict(name=f"x4/primitive_s{s}", condition="E", seed=s, plan=CU_PLAN, extra=["--no-skills"]))
    return jobs


def _jobs_v3x5() -> List[Dict[str, Any]]:
    """X5 (Protocol v3.4): a goal-free period of 20 decisions on the SW computer, then X1's plan; E, seeds 0-1.
    The comparison arm (E without the goal-free period) is X1 ``x1/E_s{0,1}``."""
    return [dict(name=f"x5/goalfree_s{s}", condition="E", seed=s, plan="GOALFREE:sw," + SW_TRAIN) for s in (0, 1)]


PLANS = {"v3": _jobs_v3, "v3x5": _jobs_v3x5}


def _n_plan(job: Dict[str, Any]) -> int:
    return len(job["plan"].split(","))


class Job:
    def __init__(self, spec: Dict[str, Any], home: str):
        self.spec, self.home = spec, home
        self.name = spec["name"]
        self.out = os.path.join(home, "results", "v3", self.name + ".json")
        self.log = os.path.join(home, "results", "v3", self.name + ".log")
        self.root = os.path.join(home, "oe", "v3", self.name.replace("/", "_"))
        self.failures = 0

    def done_count(self) -> int:
        if not os.path.exists(self.out):
            return 0
        try:
            return len(json.load(open(self.out)).get("objectives", []))
        except (ValueError, OSError):
            return 0

    def finished(self) -> bool:
        return self.done_count() >= _n_plan(self.spec) or self.failures >= 3

    def export_path(self, ref: str) -> str:
        """``@x1/E_s0/export_post`` -> that job's run root / export_post."""
        name, _, leaf = ref[1:].rpartition("/")
        return os.path.join(self.home, "oe", "v3", name.replace("/", "_"), leaf)

    def command(self, hold_s: float, threads: int) -> List[str]:
        s = self.spec
        cmd = [sys.executable, "-m", "cam_live.eval.oe_runner", "sequence", "--condition", s["condition"], "--seed",
               str(s["seed"]), "--plan", s["plan"], "--decisions", str(s.get("decisions", 20)), "--out", self.out,
               "--root", self.root, "--resume", "--max-objectives", "1", "--no-lock", "--hold-s", str(hold_s),
               "--threads", str(threads), "--tag", self.name.replace("/", "_")]
        if s.get("save_exports"):
            cmd.append("--save-exports")
        if s.get("from_export") and self.done_count() == 0:
            cmd += ["--from-export", self._fresh_export()]
        return cmd + list(s.get("extra", []))

    def _fresh_export(self) -> str:
        work = os.path.join(self.home, "fresh", self.name.replace("/", "_"))
        own = os.path.join(work, "export")
        if not os.path.exists(own):
            os.makedirs(work, exist_ok=True)
            os.chmod(work, 0o755)
            shutil.copytree(self.export_path(self.spec["from_export"]), own)
        return own

    def env_cwd(self) -> Any:
        env = dict(os.environ)
        cwd = None
        if self.spec.get("fresh"):
            work = os.path.join(self.home, "fresh", self.name.replace("/", "_"))
            cwd, home = os.path.join(work, "cwd"), os.path.join(work, "home")
            os.makedirs(cwd, exist_ok=True)
            os.makedirs(home, exist_ok=True)
            env = {k: v for k, v in os.environ.items() if not k.startswith("CAM_LIVE_") or
                   k.startswith("CAM_LIVE_EVAL") or k == "CAM_LIVE_GPU_LOCK"}
            env["CAM_LIVE_HOME"] = home
        return env, cwd

    def mark_interrupted(self, wall: float, reason: str) -> None:
        """A child killed at the hold limit: the objective is recorded as interrupted and the sequence moves on from
        its last checkpoint (the attempt is not retried, so no objective is run twice)."""
        res = json.load(open(self.out)) if os.path.exists(self.out) else {
            "executed": False, "condition": self.spec["condition"], "seed": self.spec["seed"],
            "plan": self.spec["plan"], "objectives": []}
        i = len(res["objectives"])
        fam, _, prob = self.spec["plan"].split(",")[i].partition(":")
        res["objectives"].append({"index": i, "family": fam, "problem": prob, "interrupted": True, "reason": reason,
                                  "wall_s": wall, "achieved": False, "verified_status": "interrupted",
                                  "verified_value": None, "decisions": None})
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        json.dump(res, open(self.out, "w"), default=str)


def gpu_used_gb() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return float(out.strip().splitlines()[0]) / 1024.0
    except Exception:  # noqa: BLE001
        return -1.0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cam_live.eval.oe_batch")
    ap.add_argument("--plan", default="v3", choices=sorted(PLANS))
    ap.add_argument("--only", default="", help="comma-separated job-name prefixes (e.g. x0,x1)")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max-gpu-gb", dest="max_gpu_gb", type=float, default=36.0)
    ap.add_argument("--hold-s", dest="hold_s", type=float, default=810.0)
    ap.add_argument("--yield-s", dest="yield_s", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args(argv)
    from ..runtime.gpu_cycle import GpuLease
    from ..storage import DiskGuard
    from .experiments import EVAL_HOME
    home = EVAL_HOME
    jobs = [Job(s, home) for s in PLANS[a.plan]()]
    if a.only:
        pref = tuple(p for p in a.only.split(",") if p)
        jobs = [j for j in jobs if j.name.startswith(pref)]
    guard = DiskGuard(home)
    lease = GpuLease(hold_s=a.hold_s, yield_s=a.yield_s)
    conc, peak_per_child = 1, 0.0
    log = os.path.join(home, "results", "v3", "batch_log.jsonl")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    rr = 0
    while True:
        pending = [j for j in jobs if not j.finished()]
        if not pending:
            break
        names_done = {j.name for j in jobs if j.finished()}
        ready = [j for j in pending if all(d in names_done for d in j.spec.get("depends", []))]
        if not ready:
            time.sleep(30)
            continue
        guard.check(need_bytes=4 * 2 ** 30, what="objective runs")
        waited = lease.acquire()
        t_acq = time.time()
        base_gb = gpu_used_gb()
        k = min(conc, len(ready))
        batch = ready[:k]                                 # protocol order: X0, X1, X2, X3, X4
        procs = []
        child_hold = a.hold_s - 120.0
        for j in batch:
            env, cwd = j.env_cwd()
            os.makedirs(os.path.dirname(j.log), exist_ok=True)
            lf = open(j.log, "a")
            procs.append((j, subprocess.Popen(j.command(child_hold, a.threads), stdout=lf, stderr=subprocess.STDOUT,
                                              env=env, cwd=cwd), lf, j.done_count()))
        peak = base_gb
        while any(p.poll() is None for _, p, _, _ in procs):
            time.sleep(5)
            peak = max(peak, gpu_used_gb())
            if time.time() - t_acq > a.hold_s - 20.0:
                for j, p, _, _ in procs:
                    if p.poll() is None:
                        p.kill()
                break
        for _, p, _, _ in procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
        held = time.time() - t_acq
        lease.release()
        recs = []
        for j, p, lf, before in procs:
            lf.close()
            after = j.done_count()
            rec = {"job": j.name, "rc": p.returncode, "objectives_before": before, "objectives_after": after}
            if after == before:
                if p.returncode in (-9, 137):
                    j.mark_interrupted(held, "killed at the GPU hold limit")
                    rec["interrupted"] = True
                else:
                    j.failures += 1
                    rec["failure"] = j.failures
            recs.append(rec)
        per_child = (peak - max(0.0, base_gb)) / max(1, len(procs)) if peak > 0 else 0.0
        peak_per_child = max(peak_per_child, per_child)
        if peak_per_child > 0 and (conc + 1) * peak_per_child + max(0.0, base_gb) <= a.max_gpu_gb:
            conc = min(a.concurrency, conc + 1)
        with open(log, "a") as f:
            f.write(json.dumps({"t": time.time(), "lease_waited_s": waited, "held_s": held, "concurrency": len(procs),
                                "gpu_base_gb": base_gb, "gpu_peak_gb": peak, "peak_per_child_gb": per_child,
                                "next_concurrency": conc, "children": recs}) + "\n")
        print(f"{time.strftime('%H:%M:%S')} lease held {held:.0f}s, {len(procs)} children, peak {peak:.1f} GB: "
              f"{recs}", flush=True)
        time.sleep(a.yield_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
