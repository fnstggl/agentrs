"""Spawn and supervise one learner, N actors and (optionally) the shared semantic server; shut down cleanly; account
for compute (spec 14.5 and Part 22, manifest A16/F04/X06).

Run directory written by a launch::

    run.json                specs of every process, host facts, start time
    launcher.jsonl          supervision events (start, exit, restart, stop) with exit codes
    series.jsonl            sampled time series (decisions, records, learner version, trained decisions, lag)
    compute_summary.json    merged ComputeMeter snapshots of every process incarnation (+ per-role totals)
    summary.json            final summary (throughput in the steady window, staleness, restarts, scores)
    spool/ ckpt/ archive/ traces/ learner/ status/ compute/ logs/ semserve/   (see actor.run_paths)

Processes use the ``spawn`` start method (CUDA-safe; no inherited state). Each child's stdout/stderr go to
``logs/<name>.log``. Supervision restarts a crashed actor (non-zero exit) under the same actor id — its spool writer
recovers sequence numbers and leftover segments — and a crashed learner (it resumes from LATEST), each up to
``max_restarts`` times. Shutdown order: actors (finish the current decision, close the running episode, flush), then
the learner (drains the queue, final status), then the semantic server.

`benchmark_actors` measures actor throughput for several actor counts on the toy environment (the F04 benchmark).
CLI::

    python -m superlearner.cam_unified.dist.launcher bench --actors 1,4,8 --seconds 20 --out runs/cam_unified/bench
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..observe.accounting import ComputeMeter, atomic_write_json, merge_snapshots, read_snapshots
from .actor import ActorSpec, actor_main, run_paths
from .learner import LearnerSpec, learner_main


@dataclass
class SemServeSpec:
    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = "cuda"
    max_batch_texts: int = 32
    max_wait_ms: float = 5.0
    max_tokens: int = 512
    hidden_layer: int = -1
    encoder_lora: bool = False
    lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    cache_bytes: int = 512 * 2**20

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lora_targets"] = list(self.lora_targets)
        return d


def _entry(target_path: str, log_path: str, *args) -> None:
    """Child-process trampoline: redirect stdout/stderr to the log file, then run the target."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)
    from .actor import resolve
    resolve(target_path)(*args)


class Launcher:
    def __init__(self, run_dir: str, learner: LearnerSpec, actors: list, semserve: Optional[SemServeSpec] = None,
                 max_restarts: int = 3, sample_every_s: float = 0.5, env: Optional[dict] = None):
        import torch.multiprocessing as tmp           # torch reductions: CPU tensors cross queues via shared memory
        self.ctx = tmp.get_context("spawn")
        self.run_dir = os.path.abspath(run_dir)
        self.paths = run_paths(self.run_dir)
        for d in self.paths.values():
            os.makedirs(d, exist_ok=True)
        self.learner_spec = learner
        self.actor_specs = {a.actor_id: a for a in actors}
        self.sem_spec = semserve
        self.max_restarts = max_restarts
        self.sample_every_s = sample_every_s
        self.child_env = env or {}
        self.actor_stop = self.ctx.Event()
        self.learner_stop = self.ctx.Event()
        self.sem_stop = self.ctx.Event()
        self.procs: dict = {}
        self.restarts: dict = {}
        self.finished: set = set()
        self.meter = ComputeMeter("launcher", role="launcher", include_children=False)
        self.series: list = []
        self._last_sample = 0.0
        self.sem_req = None
        self.sem_resp: dict = {}
        self.t_start = None

    # ---------------------------------------------------------------- bookkeeping
    def _event(self, kind: str, name: str, **kw) -> None:
        rec = {"t": time.time(), "event": kind, "name": name, **kw}
        with open(os.path.join(self.run_dir, "launcher.jsonl"), "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def _spawn(self, name: str, target: str, args: tuple):
        for k, v in self.child_env.items():
            os.environ[k] = str(v)
        p = self.ctx.Process(target=_entry, args=(target, os.path.join(self.paths["logs"], f"{name}.log")) + args,
                             name=name, daemon=False)
        p.start()
        self.procs[name] = p
        self._event("start", name, pid=p.pid)
        return p

    def _start_actor(self, aid: str) -> None:
        spec = self.actor_specs[aid]
        sem = None
        if self.sem_spec is not None:
            sem = (self.sem_req, self.sem_resp[aid], aid)
        self._spawn(aid, "superlearner.cam_unified.dist.actor:actor_main", (spec.to_dict(), self.actor_stop, sem))

    def _start_learner(self) -> None:
        self._spawn("learner", "superlearner.cam_unified.dist.learner:learner_main",
                    (self.learner_spec.to_dict(), self.learner_stop))

    def _start_semserve(self) -> None:
        self._spawn("semserve", "superlearner.cam_unified.dist.semserve:semserve_main",
                    (self.sem_spec.to_dict(), self.sem_req, self.sem_resp, self.sem_stop,
                     self.paths["semserve"], self.paths["compute"]))

    # ---------------------------------------------------------------- lifecycle
    def start(self, semserve_timeout_s: float = 300.0) -> None:
        self.t_start = time.time()
        atomic_write_json(os.path.join(self.run_dir, "run.json"), {
            "t_start": self.t_start, "host": platform.node(), "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(), "learner": self.learner_spec.to_dict(),
            "actors": [a.to_dict() for a in self.actor_specs.values()],
            "semserve": self.sem_spec.to_dict() if self.sem_spec else None, "max_restarts": self.max_restarts})
        if self.sem_spec is not None:
            self.sem_req = self.ctx.Queue()
            self.sem_resp = {aid: self.ctx.Queue() for aid in self.actor_specs}
            self._start_semserve()
            self._wait_status("semserve", "ready", semserve_timeout_s)
        self._start_learner()
        for aid in self.actor_specs:
            self._start_actor(aid)

    def _wait_status(self, name: str, state: str, timeout_s: float) -> None:
        path = os.path.join(self.paths["status"], f"{name}.json")
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                with open(path) as fh:
                    if json.load(fh).get("state") == state:
                        return
            except (OSError, json.JSONDecodeError):
                pass
            p = self.procs.get(name)
            if p is not None and not p.is_alive():
                raise RuntimeError(f"{name} exited with code {p.exitcode} before reaching {state!r}; "
                                   f"see {self.paths['logs']}/{name}.log")
            time.sleep(0.1)
        raise TimeoutError(f"{name} did not reach {state!r} in {timeout_s}s")

    def poll(self) -> None:
        """One supervision pass: reap exits, restart crashed processes, sample the time series."""
        for name, p in list(self.procs.items()):
            if p.is_alive() or name in self.finished:
                continue
            code = p.exitcode
            self._event("exit", name, pid=p.pid, exitcode=code)
            stopping = (name in self.actor_specs and self.actor_stop.is_set()) or \
                       (name == "learner" and self.learner_stop.is_set()) or \
                       (name == "semserve" and self.sem_stop.is_set())
            if code == 0 or stopping:
                self.finished.add(name)
                continue
            n = self.restarts.get(name, 0)
            if n >= self.max_restarts:
                self._event("give_up", name, restarts=n)
                self.finished.add(name)
                continue
            self.restarts[name] = n + 1
            self._event("restart", name, attempt=n + 1)
            if name in self.actor_specs:
                self._start_actor(name)
            elif name == "learner":
                self._start_learner()
            elif name == "semserve":
                self._start_semserve()
        self._sample()

    def _read_status(self) -> dict:
        out = {}
        d = self.paths["status"]
        for fn in os.listdir(d):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(d, fn)) as fh:
                        out[fn[:-5]] = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    pass
        return out

    def _sample(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_sample < self.sample_every_s:
            return
        self._last_sample = now
        st = self._read_status()
        acts = [v for k, v in st.items() if k in self.actor_specs]
        ln = st.get("learner", {})
        row = {"t": now, "t_rel": now - (self.t_start or now), "decisions": sum(a.get("decisions", 0) for a in acts),
               "records": sum(a.get("records", 0) for a in acts), "episodes": sum(a.get("episodes", 0) for a in acts),
               "learner_version": ln.get("version"),
               "trained_decisions": ln.get("counts", {}).get("trained_decisions", 0),
               "ingested": ln.get("counts", {}).get("ingested_events", 0),
               "actor_versions": {k: v.get("version") for k, v in st.items() if k in self.actor_specs},
               "recent_scores": {k: v.get("recent_score") for k, v in st.items() if k in self.actor_specs}}
        self.series.append(row)
        with open(os.path.join(self.run_dir, "series.jsonl"), "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def actors_done(self) -> bool:
        return all(a in self.finished for a in self.actor_specs)

    def kill(self, name: str, sig: int = 9) -> int:
        """Fault injection for recovery tests: send a signal to a child process."""
        p = self.procs[name]
        os.kill(p.pid, sig)
        return p.pid

    def run(self, duration_s: Optional[float] = None, poll_s: float = 0.1) -> dict:
        if self.t_start is None:
            self.start()
        t0 = time.time()
        try:
            while True:
                self.poll()
                if self.actors_done():
                    break
                if duration_s is not None and time.time() - t0 >= duration_s:
                    break
                if "learner" in self.finished and self.restarts.get("learner", 0) >= self.max_restarts:
                    break
                time.sleep(poll_s)
        finally:
            summary = self.stop()
        return summary

    def stop(self, timeout_s: float = 60.0) -> dict:
        self.actor_stop.set()
        self._join([n for n in self.procs if n in self.actor_specs], timeout_s)
        self.learner_stop.set()
        self._join(["learner"], timeout_s)
        if "semserve" in self.procs:
            self.sem_stop.set()
            self._join(["semserve"], timeout_s)
        self._sample(force=True)
        return self.summarize()

    def _join(self, names: list, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        for n in names:
            p = self.procs.get(n)
            if p is None:
                continue
            p.join(max(0.1, deadline - time.time()))
            if p.is_alive():
                self._event("terminate", n, pid=p.pid)
                p.terminate()
                p.join(5)
            if p.is_alive():
                p.kill()
                p.join(5)
            if n not in self.finished:
                self._event("exit", n, pid=p.pid, exitcode=p.exitcode)
                self.finished.add(n)

    # ---------------------------------------------------------------- accounting
    def summarize(self) -> dict:
        snaps = read_snapshots(self.paths["compute"])
        snaps.append(self.meter.snapshot())
        compute = merge_snapshots(snaps)
        atomic_write_json(os.path.join(self.run_dir, "compute_summary.json"),
                          {"total": compute, "processes": snaps})
        st = self._read_status()
        ln = st.get("learner", {})
        acts = {k: v for k, v in st.items() if k in self.actor_specs}
        thr = steady_rate(self.series, "decisions")
        summary = {"run_dir": self.run_dir, "wall_s": time.time() - (self.t_start or time.time()),
                   "n_actors": len(self.actor_specs), "restarts": dict(self.restarts),
                   "decisions": sum(a.get("decisions", 0) for a in acts.values()),
                   "records": sum(a.get("records", 0) for a in acts.values()),
                   "episodes": sum(a.get("episodes", 0) for a in acts.values()),
                   "decisions_per_s_steady": thr, "learner_version": ln.get("version"),
                   "learner_counts": ln.get("counts", {}), "lag": ln.get("router", {}),
                   "queue": ln.get("queue", {}), "actor_recent_scores": {k: v.get("recent_score") for k, v in acts.items()},
                   "actor_versions_used": {k: v.get("versions_used") for k, v in acts.items()},
                   "compute": {k: compute.get(k) for k in ("cpu_s", "gpu_s", "wall_s", "env_steps", "decisions",
                                                           "sim_days", "tokens_prompt", "tokens_generated", "records",
                                                           "updates", "samples_trained", "peak_rss_mb_max",
                                                           "peak_rss_mb_sum", "peak_gpu_mb_max", "cost_usd")}}
        atomic_write_json(os.path.join(self.run_dir, "summary.json"), summary)
        return summary


def steady_rate(series: list, key: str, warm_frac: float = 0.3) -> Optional[float]:
    """Slope of ``key`` over the samples after the first ``warm_frac`` of the run (excludes spawn/import time)."""
    pts = [(r["t"], r[key]) for r in series if r.get(key) is not None]
    if len(pts) < 3:
        return None
    t0, t1 = pts[0][0], pts[-1][0]
    start = t0 + warm_frac * (t1 - t0)
    w = [p for p in pts if p[0] >= start]
    first = next((p for p in w if p[1] > 0), None)
    if first is None or w[-1][0] - first[0] <= 0:
        return None
    return (w[-1][1] - first[1]) / (w[-1][0] - first[0])


def benchmark_actors(out_dir: str, n_actors=(1, 4, 8), seconds: float = 20.0, envs_per_actor: int = 4,
                     min_decisions: int = 256, max_batch_decisions: int = 1024, seed: int = 0) -> list:
    """Toy-environment throughput benchmark: learner + N actors for ``seconds`` each; returns one summary per N."""
    results = []
    for n in n_actors:
        run = os.path.join(out_dir, f"actors{n}")
        L = LearnerSpec(run_dir=run, model_kwargs={"seed": seed, "lr": 1e-2}, min_decisions=min_decisions,
                        max_batch_decisions=max_batch_decisions, max_lag=16, torch_threads=2)
        A = [ActorSpec(actor_id=f"actor{i}", run_dir=run, env_kwargs={"B": envs_per_actor, "mechanism_seed": seed},
                       seed=1000 + i, batch_records=512) for i in range(n)]
        launch = Launcher(run, L, A)
        s = launch.run(duration_s=seconds)
        s["envs_per_actor"] = envs_per_actor
        results.append(s)
        atomic_write_json(os.path.join(out_dir, "bench.json"), results)
    return results


def _main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="cam_unified.dist.launcher")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="actor throughput on the toy environment")
    b.add_argument("--actors", default="1,4,8")
    b.add_argument("--seconds", type=float, default=20.0)
    b.add_argument("--envs-per-actor", type=int, default=4)
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = benchmark_actors(a.out, tuple(int(x) for x in a.actors.split(",")), a.seconds, a.envs_per_actor)
    for r in res:
        c = r["compute"]
        print(f"actors={r['n_actors']:2d} decisions/s(steady)={r['decisions_per_s_steady'] or 0:9.1f} "
              f"decisions={r['decisions']:8d} learner_v={r['learner_version']} cpu_s={c['cpu_s']:.1f} "
              f"peak_rss_max={c['peak_rss_mb_max']:.0f}MB lag={r['lag'].get('lag_hist')}")


if __name__ == "__main__":
    _main()
