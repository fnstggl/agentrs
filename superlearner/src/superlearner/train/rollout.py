"""Parallel experience collection: a pool of CPU worker processes, each with its own copy of the controller.

A worker receives the current weights, then jobs. A job names a task (or, for held-out evaluation, an index into
the env server's own list), how choices are made (``sample`` / ``greedy`` from the controller, ``random``,
``llm`` = frozen LLM with grounding and no controller, ``teacher:<name>`` = a demonstration script), and a seed.
It returns the episode: the evaluator's summary, the decision records needed for learning, text embeddings that
were new to the worker, its episodic-memory record and tool-library changes, and wall-clock costs.

Worlds are created locally (`SessionCore`) or, when ``env_socket`` is set, by the trusted env hub (`world/hub.py`)
so that the worker - which may be running candidate learner code - never holds a world or an evaluator.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from multiprocessing.connection import wait as mp_wait

import numpy as np


def _make_core(job: dict, wcfg: dict, hub):
    from ..world.tasks import TaskSpec
    if hub is not None:
        if "eval" in job:
            hub.reset(eval_index=int(job["eval"]))
        else:
            hub.reset(TaskSpec.from_dict(job["spec"]))
        return hub
    from ..world.core import CoreConfig, SessionCore
    return SessionCore(TaskSpec.from_dict(job["spec"]), CoreConfig(**dict(wcfg.get("core") or {}, **(job.get("core") or {}))),
                       handle=str(job.get("handle") or "agent"))


def make_chooser(job: dict):
    from ..cam.agent import LLMOnlyChooser, ModelChooser, RandomChooser
    mode = job.get("mode", "sample")
    seed = int(job.get("seed", 0))
    if mode == "sample":
        return ModelChooser(False, float(job.get("temperature", 1.0)), seed)
    if mode == "greedy":
        return ModelChooser(True, 1.0, seed)
    if mode == "random":
        return RandomChooser(seed)
    if mode == "llm":
        return LLMOnlyChooser()
    if mode.startswith("teacher"):
        from .demos import TeacherChooser
        return TeacherChooser(mode.partition(":")[2] or "auto", seed)
    raise ValueError(f"unknown mode {mode!r}")


def run_job(agent, job: dict, wcfg: dict, hub=None) -> dict:
    from ..cam.agent import AgentConfig
    t0 = time.time()
    core = _make_core(job, wcfg, hub)
    base = dict(wcfg.get("agent") or {})
    agent.cfg = AgentConfig(**dict(base, **(job.get("agent") or {})))
    agent.human = bool(job.get("human"))
    chooser = make_chooser(job)
    use_model = job.get("mode", "sample") in ("sample", "greedy") or job.get("with_model", False)
    saved = agent.model
    if not use_model:
        agent.model = None
    try:
        ep = agent.run(core, chooser, job.get("episode_id", ""), learn=bool(job.get("learn", True)))
    finally:
        agent.model = saved
    ep["summary"]["wall_s"] = round(time.time() - t0, 3)
    ep["summary"]["mode"] = job.get("mode", "sample")
    ep["summary"]["job"] = {k: job[k] for k in ("spec", "eval", "seed", "tag") if k in job}
    if hasattr(chooser, "report"):
        ep["summary"]["teacher"] = chooser.report()
    if not job.get("keep_decisions", True):
        ep["decisions"] = [{"human": d["human"]} for d in ep["decisions"] if "human" in d]
    return ep


def die_with_parent() -> None:
    """Ask the kernel to kill this process when its parent dies, so that stopping a run never leaves workers."""
    try:
        import ctypes
        import signal
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)        # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - not on Linux, or not permitted: workers still exit on a closed pipe
        pass


def worker_main(conn, wid: int, wcfg: dict) -> None:
    die_with_parent()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    import torch
    torch.set_num_threads(1)
    from ..cam.agent import AgentConfig, CAMAgent
    from ..cam.model import CAMController, EmbStore, ModelConfig
    from ..llm.server import connect
    llm = connect(wcfg.get("llm"))
    store = EmbStore()
    model = CAMController(ModelConfig(**(wcfg.get("model") or {})))
    model.eval()
    agent = CAMAgent(model, store, llm, AgentConfig(**(wcfg.get("agent") or {})))
    hub = None
    if wcfg.get("env_socket"):
        from multiprocessing.connection import Client
        from ..world.core import RemoteCore
        from ..world.hub import AUTHKEY
        hub = RemoteCore(Client(wcfg["env_socket"], family="AF_UNIX", authkey=AUTHKEY))
    n_synced = 0
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        op = msg[0]
        try:
            if op == "quit":
                break
            if op == "weights":
                model.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in msg[1].items()})
                conn.send(("ok", None))
            elif op == "wm":
                from ..cam.worldmodel import WorldModel
                if agent.live_wm is None:
                    agent.live_wm = WorldModel(model.cfg.d).eval()
                agent.live_wm.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in msg[1].items()})
                conn.send(("ok", None))
            elif op == "table":
                store.replace(msg[1][0], msg[1][1])           # the backbone changed: every text vector is new
                if hasattr(llm, "clear"):
                    llm.clear()
                conn.send(("ok", None))
            elif op == "sync":
                d = msg[1]
                if d.get("reset"):
                    agent.memory.items, n_synced = [], 0
                agent.memory.items = agent.memory.items[:n_synced] + list(d.get("mem_new") or [])
                agent.memory._mat = None
                n_synced = len(agent.memory.items)
                if d.get("library") is not None:
                    agent.library.load_state_dict(d["library"])
                conn.send(("ok", None))
            elif op == "run":
                job = msg[1]
                n0 = len(agent.memory.items)
                ep = run_job(agent, job, wcfg, hub)
                ep["mem_new"] = agent.memory.items[n0:]
                ep["library"] = agent.library.state_dict() if job.get("learn", True) else None
                ep["llm_usage"] = llm.usage()
                ep["wid"] = wid
                conn.send(("ok", ep))
            else:
                conn.send(("err", f"unknown op {op}"))
        except Exception as e:  # noqa: BLE001
            conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"))
    conn.close()


class RolloutPool:
    def __init__(self, n_workers: int, wcfg: dict, start: str = "spawn"):
        self.ctx = mp.get_context(start)
        self.wcfg = wcfg
        self.conns, self.procs = [], []
        # Workers must come up single-threaded: a freshly spawned interpreter starts one BLAS thread per core
        # while importing numpy (before any code of ours runs), and dozens of workers doing that at once exhaust
        # the container's thread limit. The variables are set for the children and restored for this process.
        keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        saved = {k: os.environ.get(k) for k in keys}
        os.environ.update({k: "1" for k in keys})
        try:
            for i in range(n_workers):
                a, b = self.ctx.Pipe()
                p = self.ctx.Process(target=worker_main, args=(b, i, wcfg), daemon=True)
                p.start()
                b.close()
                self.conns.append(a)
                self.procs.append(p)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.errors: list[str] = []

    def _all(self, msg) -> None:
        for c in self.conns:
            c.send(msg)
        for c in self.conns:
            st, out = c.recv()
            if st != "ok":
                raise RuntimeError(f"rollout worker: {out}")

    def set_weights(self, state_dict: dict) -> None:
        # numpy arrays are pickled by value; torch tensors would go through /dev/shm, which a confined run cannot use
        self._all(("weights", {k: v.detach().cpu().numpy() for k, v in state_dict.items()}))

    def set_table(self, keys, mat) -> None:
        self._all(("table", (np.asarray(keys), np.asarray(mat, np.float32))))

    def set_wm(self, state_dict: dict) -> None:
        self._all(("wm", {k: v.detach().cpu().numpy() for k, v in state_dict.items()}))

    def sync(self, mem_new: list, library: dict | None, reset: bool = False) -> None:
        self._all(("sync", {"mem_new": mem_new, "library": library, "reset": reset}))

    def run(self, jobs: list[dict]) -> list[dict | None]:
        """Run jobs on idle workers as they free up. Returns episodes in job order (None where a job failed)."""
        out: list = [None] * len(jobs)
        pending = list(range(len(jobs)))[::-1]
        busy: dict = {}
        idle = list(self.conns)
        while pending or busy:
            while pending and idle:
                c = idle.pop()
                j = pending.pop()
                c.send(("run", jobs[j]))
                busy[c] = j
            if not busy:                    # every worker has died: fail the remaining jobs instead of waiting forever
                self.errors.append(f"no live workers; {len(pending)} job(s) not run")
                break
            for c in mp_wait(list(busy)):
                j = busy.pop(c)
                try:
                    st, ep = c.recv()
                except (EOFError, OSError) as e:
                    st, ep = "err", f"worker died: {e}"
                if st == "ok":
                    out[j] = ep
                    idle.append(c)
                else:
                    self.errors.append(str(ep)[:2000])
                    if "worker died" not in str(ep):
                        idle.append(c)
        return out

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("quit",))
            except (OSError, BrokenPipeError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


class LocalPool:
    """Same interface, in-process (tests and tiny runs)."""

    def __init__(self, wcfg: dict, model=None, store=None, llm=None):
        from ..cam.agent import AgentConfig, CAMAgent
        from ..cam.model import CAMController, EmbStore, ModelConfig
        from ..llm.server import connect
        self.wcfg = wcfg
        self.llm = llm or connect(wcfg.get("llm"))
        self.model = model or CAMController(ModelConfig(**(wcfg.get("model") or {})))
        self.model.eval()
        self.store = store or EmbStore()
        self.agent = CAMAgent(self.model, self.store, self.llm, AgentConfig(**(wcfg.get("agent") or {})))
        self.errors: list[str] = []
        self._n = 0

    def set_weights(self, state_dict):
        self.model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})

    def set_table(self, keys, mat):
        self.store.replace(keys, mat)
        if hasattr(self.llm, "clear"):
            self.llm.clear()

    def set_wm(self, state_dict):
        from ..cam.worldmodel import WorldModel
        if self.agent.live_wm is None:
            self.agent.live_wm = WorldModel(self.model.cfg.d).eval()
        self.agent.live_wm.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})

    def sync(self, mem_new, library, reset=False):
        if reset:
            self.agent.memory.items, self._n = [], 0
        self.agent.memory.items = self.agent.memory.items[: self._n] + list(mem_new or [])
        self.agent.memory._mat = None
        self._n = len(self.agent.memory.items)
        if library is not None:
            self.agent.library.load_state_dict(library)

    def run(self, jobs):
        out = []
        for job in jobs:
            n0 = len(self.agent.memory.items)
            try:
                ep = run_job(self.agent, job, self.wcfg)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
                out.append(None)
                continue
            ep["mem_new"] = self.agent.memory.items[n0:]
            ep["library"] = self.agent.library.state_dict() if job.get("learn", True) else None
            ep["llm_usage"] = self.llm.usage()
            ep["wid"] = 0
            out.append(ep)
        return out

    def close(self):
        pass
