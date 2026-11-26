"""Stage-1 vertical path on the real Crucible OpenWorld (spec Part 21 Stage 1-2):

    semantic observation -> latent model -> generated action -> real Crucible execution -> verified outcome
    -> persistent experience (archive + memory) -> neural-weight update -> later decisions with the new weights

`run(RunSpec)` builds the full real system (`system.build_system`), opens one task world with workstream G's
process-isolated operator (`open_world_isolated`: worlds, operator and evaluator in a spawned process, only records
cross; the world store is a SQLite file in the run directory so another process can resume it) and drives the
`HarnessAgent` with the canonical loop: SMDP clock (cost_s + think_s, or the agent's own wait), verified
rewards from the operator-side evaluator only, a neural update every `update_every` decisions, full-lifetime
checkpoints every `checkpoint_every_days` simulated days, the JSONL transcript - all through the canonical loop
`runtime.run_lifetime` (labels of a LabelSource evaluator go to the run's LabelStore, learner side only).

Chunks: a lifetime longer than one GPU-lock slot runs as several processes. `max_wall_s` / `stop_after` end a chunk
at the first checkpoint written after the limit (the adapter refuses the next tool listing, before any action, so no
step is executed past the checkpoint); ``resume=True`` continues from the latest checkpoint in a new process.
`kill_after` SIGKILLs this process after that many decisions (the kill-and-resume test: whatever it archived after its
last checkpoint becomes an abandoned branch, see `branch.py`).
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from .harness_agent import HarnessAgent, _atomic_json, _jsonable
from .system import SystemConfig, build_system, describe
from .tasks import make_task, open_task_world


class StopChunk(Exception):
    """Raised at the top of the harness loop right after a checkpoint, when the chunk's limit is reached."""


class ExitNow(Exception):
    """`exit_after`: leave the lifetime after that many decisions of this process, without a checkpoint (used on
    throwaway copies of a run directory to compare resumed decisions)."""


@dataclass
class RunSpec:
    run_dir: str
    task: str = "open_economy"
    task_index: int = 0
    days: Optional[float] = None               # task horizon (default: the task's own)
    max_decisions: Optional[int] = None        # lifetime cap (all chunks)
    stop_after: Optional[int] = None           # this chunk: stop at the first checkpoint after this many decisions
    max_wall_s: Optional[float] = None         # this chunk: stop at the first checkpoint after this wall time
    kill_after: Optional[int] = None           # this process: SIGKILL itself after this many decisions
    exit_after: Optional[int] = None           # this process: leave (no checkpoint) after this many decisions
    update_every: int = 8
    checkpoint_every_days: float = 0.25
    keep_checkpoints: int = 3
    think_s: float = 600.0
    isolated: bool = True
    world_seed: int = 0                        # operator RNG (handles); the world seed comes from the task
    resume: bool = False
    require_cam_eval: Optional[bool] = None     # None = armed by the config (DOM decision path); False = declared legacy
    system: SystemConfig = field(default_factory=SystemConfig)
    command: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["system"] = self.system.to_dict()
        return d


class _ChunkAdapter:
    """The harness-side adapter wrapper that ends a chunk cleanly: `available_tools` (the first call of every harness
    iteration) raises StopChunk when a stop was requested and a checkpoint was written at the current step."""

    def __init__(self, adapter, ctl: "_Control"):
        self._ad = adapter
        self._ctl = ctl

    def __getattr__(self, name):
        return getattr(self._ad, name)

    def available_tools(self):
        if self._ctl.should_stop():
            raise StopChunk()
        return self._ad.available_tools()


class _Control:
    def __init__(self, spec: RunSpec, hagent: HarnessAgent):
        self.spec = spec
        self.h = hagent
        self.t0 = time.time()
        self.calls = 0
        self.saved_at_call = -1
        self.stop_requested = False

    def on_step(self, step: int, tr) -> None:
        self.calls += 1
        s = self.spec
        if s.kill_after is not None and self.calls >= s.kill_after:
            _atomic_json(os.path.join(s.run_dir, "killed.json"),
                         {"pid": os.getpid(), "calls": self.calls, "step": step, "t": time.time(),
                          "version": self.h.model_version, "archive": self.h.agent.archive.count()})
            self.h.sys.recorder.flush()
            os.kill(os.getpid(), signal.SIGKILL)
        if s.exit_after is not None and self.calls >= s.exit_after:
            raise ExitNow()
        if s.stop_after is not None and self.calls >= s.stop_after:
            self.stop_requested = True
        if s.max_wall_s is not None and time.time() - self.t0 >= s.max_wall_s:
            self.stop_requested = True
        _atomic_json(os.path.join(s.run_dir, "progress.json"),
                     {"pid": os.getpid(), "step": step, "calls": self.calls, "decisions": self.h.agent.n_decisions,
                      "version": self.h.model_version, "archive": self.h.agent.archive.count(),
                      "wall_s": time.time() - self.t0, "t": time.time()})

    def saved(self) -> None:
        self.saved_at_call = self.calls

    def should_stop(self) -> bool:
        return self.stop_requested and self.saved_at_call == self.calls


def _prune(ckpt_dir: str, keep: int) -> None:
    if not os.path.isdir(ckpt_dir) or keep <= 0:
        return
    ds = sorted(d for d in os.listdir(ckpt_dir) if d.startswith("ckpt_"))
    for d in ds[:-keep]:
        shutil.rmtree(os.path.join(ckpt_dir, d), ignore_errors=True)


def load_spec(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "run.json")) as fh:
        return json.load(fh)


def run(spec: RunSpec) -> dict:
    run_dir = os.path.abspath(spec.run_dir)
    spec.run_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)
    if spec.resume:
        prev = load_spec(run_dir)
        spec.system = SystemConfig(**prev["spec"]["system"])          # a lifetime keeps its system
        for k in ("task", "task_index", "days", "update_every", "checkpoint_every_days", "think_s", "isolated",
                  "world_seed"):
            setattr(spec, k, prev["spec"][k])
    if spec.resume and not os.path.exists(os.path.join(run_dir, "lifetime_ckpt", "LATEST")):
        raise FileNotFoundError(f"{run_dir}: nothing to resume (no lifetime checkpoint); a run directory with "
                                "experience but no checkpoint is never silently restarted")
    if not spec.resume and os.path.exists(os.path.join(run_dir, "run.json")):
        raise FileExistsError(f"{run_dir} already holds a run; use resume")
    task, fam, seed = make_task(spec.task, spec.task_index, days=spec.days)
    t_build = time.time()
    system = build_system(run_dir, spec.system)
    build_s = time.time() - t_build
    hagent = HarnessAgent(system)
    if not spec.resume:
        _atomic_json(os.path.join(run_dir, "run.json"),
                     {"spec": spec.to_dict(), "task": {"name": spec.task, "world_family": fam, "world_seed": seed,
                                                       "public_goal_text": getattr(task, "description", "")},
                      "system": describe(system), "t_start": time.time(), "build_s": build_s,
                      "torch": torch.__version__, "cuda": torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else None})
    store = f"sqlite:///{os.path.join(run_dir, 'world.db')}"
    ad, rop, rev = open_task_world(task, store_url=store, seed=spec.world_seed, isolated=spec.isolated)
    ctl = _Control(spec, hagent)
    ckpt_dir = os.path.join(run_dir, "lifetime_ckpt")
    save0 = hagent.save

    def save(path):
        meta = save0(path)
        ctl.saved()
        _prune(ckpt_dir, spec.keep_checkpoints)
        return meta
    hagent.save = save
    from ..runtime import CheckpointSpec, legacy_default_mode, run_lifetime
    stopped, res, err = False, None, None
    t0 = time.time()
    # the canonical loop; labels (when the evaluator is a LabelSource) go to the learner-side LabelStore only
    labels = rev if rev is not None and (hasattr(rev, "take_labels") or hasattr(rev, "release_labels")) else None
    try:
        res = run_lifetime(hagent, _ChunkAdapter(ad, ctl), evaluator=rev, labels=labels,
                           label_store=system.label_store if labels is not None else None,
                           reward_mode=legacy_default_mode(hagent, rev), learner=system.learner_for(hagent),
                           max_decisions=spec.max_decisions, think_s=spec.think_s, update_every=spec.update_every,
                           checkpoint=CheckpointSpec(ckpt_dir, every_days=spec.checkpoint_every_days, operator=rop,
                                                     resume=spec.resume),
                           transcript=os.path.join(run_dir, "transcript.jsonl"),
                           model_version=lambda: hagent.model_version, on_step=ctl.on_step, flush_reason=None,
                           require_cam_eval=spec.require_cam_eval)
    except StopChunk:
        stopped = True
    except ExitNow:
        stopped = "exit_after"
    finally:
        system.recorder.flush()
        system.archive.flush()
        hagent._status("stopped" if stopped else "finished")
        try:
            final = [rev.evaluate_goal(0).to_dict()] if rev is not None else []
        except Exception as e:  # noqa: BLE001 - the record of a failed run must still be written
            final, err = [], repr(e)
        ad.close()
    chunk = {"pid": os.getpid(), "resumed": spec.resume, "resume_info": hagent.resume_info, "stopped_chunk": stopped,
             "calls": ctl.calls, "wall_s": time.time() - t0, "build_s": build_s,
             "decisions_total": hagent.agent.n_decisions, "version": hagent.model_version,
             "archive_count": system.archive.count(), "final": final, "error": err,
             "lifetime": None if res is None else {k: v for k, v in res.to_dict().items() if k != "updates"},
             "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None}
    with open(os.path.join(run_dir, "chunks.jsonl"), "a") as fh:
        fh.write(json.dumps(_jsonable(chunk)) + "\n")
    return chunk
