"""Asynchronous rollout workers, for learners that train while worlds run.

    farm = RolloutFarm(4, make_policy="mylearner.policy:build", params=theta0,
                       world_kwargs={"think_s": 600})
    for k in range(4):
        farm.submit({"family": "tiny", "seed": k, "key": f"w{k}"}, steps=64)
    for traj in farm:                                  # arrives out of order, as workers finish
        learner.add(traj.obs, traj.actions, traj.rewards, traj.sim_dt, version=traj.param_version)
        if learner.ready():
            farm.publish(learner.update())             # bytes; workers pick it up at their next task
        farm.submit({"family": "tiny", "seed": ..., "key": traj.world_key}, steps=64)
    farm.close()

Each worker process repeatedly (a) takes a task - a world reference and the identifier of the newest policy
parameters -, (b) runs an episode (``steps=None``) or a fixed number of agent steps in its own world, and (c)
pushes the trajectory to the consumer: observations, actions, rewards, action results, the simulated time each
step consumed (transitions have variable duration), wall time, and the parameter version that produced it.

* **Versions.** `publish(blob)` may be called at any time. A task is stamped with the newest version when it is
  handed to a worker (an episode boundary); the blob travels to each worker once per version. Every trajectory
  carries `param_version`, the hash of the blob its policy was built from (computed in the worker) and
  `latest_version` (newest version when the farm received it), so staleness is measured, not assumed.
* **Backpressure.** Finished trajectories wait in a bounded queue. When the consumer falls behind, the queue
  fills, the farm stops reading from workers and stops handing out tasks: workers block. Nothing grows.
* **Exact accounting.** One task in flight per worker over a private pipe: every submitted task comes back
  exactly once, as a trajectory or as a failure (an exception in the world or policy; a worker that died, with
  exit code and stderr tail). Steps are counted where they are produced and where they are received.
* **No model crosses the process boundary** unless the learner chooses: the policy is built inside each worker
  by ``make_policy(params: bytes, **policy_kwargs)``, named as ``"module:callable"``.
* **Persistent worlds.** With ``steps=k`` a task continues the world named by its key; that world lives in one
  worker, and later tasks for the same key are routed to that worker. If the worker is lost, the next task
  rebuilds the world from its reference and the trajectory says so (`fresh_world`).

Worlds are built by ``make_world(ref, **world_kwargs)``; the default, `make_world_env`, wraps a kernel world as a
semi-Markov decision process for one participant (`WorldEnv`). Any object with ``reset() -> obs``,
``step(action) -> (obs, reward, done, info)`` (``info["sim_dt"]`` = simulated seconds consumed) and ``close()``
works, e.g. a kernel-free mechanism simulator.

Like `pool`, this module imports neither numpy nor the kernel at import time.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from .pool import DAY, Supervisor, ensure_registered, resolve

__all__ = ["RolloutFarm", "Rollout", "WorldEnv", "make_world_env", "class_policy"]


# ====================================================================== the record that crosses the boundary
@dataclass
class Rollout:
    """One finished task: a trajectory (``ok=True``) or a failure report (``ok=False``)."""
    task_id: int
    ok: bool
    world_key: str = ""
    tag: Any = None
    worker: int = -1
    worker_pid: int = 0
    param_version: int = -1       # version of the parameters that produced these actions
    params_hash: str = ""         # blake2b-64 of the parameter blob the worker's policy was built from
    latest_version: int = -1      # newest published version when the farm received this
    fresh_world: bool = False     # the world was built for this task (first use, reset, or after a lost worker)
    done: bool = False            # the environment ended the episode
    truncated: bool = False       # stopped by the per-task step cap, not by the environment
    steps: int = 0
    obs: list = field(default_factory=list)        # observation each action was chosen from
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    results: list = field(default_factory=list)    # the world's answer to each action (ActionResult dict or None)
    sim_dt: list = field(default_factory=list)     # simulated seconds each step consumed
    sim_t: list = field(default_factory=list)      # simulated time at which each action was chosen
    final_obs: Any = None
    sim_seconds: int = 0
    wall_s: float = 0.0           # worker wall-clock seconds for the task
    policy_s: float = 0.0         # ... of which inside policy.act
    env_s: float = 0.0            # ... of which inside env.step
    setup_s: float = 0.0          # ... of which building the world and the policy
    queue_s: float = 0.0          # seconds between submission and hand-over to a worker
    kind: str | None = None       # failure kind: "exception" | "worker_died" | "timeout" | "abandoned"
    error: dict | None = None
    evidence: dict | None = None

    @property
    def staleness(self) -> int:
        """How many parameter versions were published between this task's start and its arrival."""
        return self.latest_version - self.param_version

    def transitions(self) -> Iterator[dict]:
        for i in range(self.steps):
            yield {"obs": self.obs[i] if self.obs else None, "action": self.actions[i], "reward": self.rewards[i],
                   "result": self.results[i], "sim_dt": self.sim_dt[i], "t": self.sim_t[i],
                   "next_obs": (self.obs[i + 1] if i + 1 < len(self.obs) else self.final_obs) if self.obs else None,
                   "param_version": self.param_version}


# ====================================================================== a kernel world as an environment
class WorldEnv:
    """One participant's view of a kernel world, as a semi-Markov decision process.

    A step is: the participant's action executes (or it waits, ``action=None``), then the world runs for the
    action's own duration (the tool's `cost_s`, at least one second) plus `think_s` simulated seconds - the
    schedule `agents.base.AsyncRunner` gives an agent. The step therefore consumes a variable amount of simulated
    time, reported as ``info["sim_dt"]``.

    `ref` is ``{"family", "seed", "overrides", "extra_modules"}`` or ``{"snapshot": <file>}`` (a snapshot written
    by `WorldPool`; each episode forks it, which is how cheap episodic resets are done). Reward is
    ``objective.step_reward(world, actor)`` when an objective factory is given (`goals.contracts.RewardContract`
    protocol: `begin`, `step_reward`); otherwise the change of the participant's own account balances, in cents.

    `reward_mode` (Crucible 5.0, `crucible.experience.modes`; Clarification 2): ``observation_only`` (the default)
    returns 0.0 on every step (the objective, if any, is still stepped so its verifier keeps running; `verify()`
    returns its status) and builds the world with evaluator data hidden from the participant;
    ``legacy_shaped_reward``, selected explicitly, is the 4.0 behaviour above; ``verified_terminal_outcome`` returns 1.0 once, at the first step
    where the objective's status reports success, else 0.0 (0.0 always without an objective).
    """

    def __init__(self, ref: dict, *, handle: str = "learner", endowment: int = 3_000_000, think_s: int = 600,
                 horizon_days: float | None = None, objective: str | None = None, objective_kwargs: dict | None = None,
                 max_events: int = 30, imports: tuple = (), strict: bool = False,
                 reward_mode: str | None = None):
        from ..experience.modes import check_mode, prepare_spec, prepare_world
        self.reward_mode = check_mode(reward_mode)
        self._paid = False
        ensure_registered(imports)
        import pickle
        import zlib

        from ..api.local import ensure_modules
        from ..core.kernel import World
        from ..worldgen import generate
        if int(think_s) < 1:
            raise ValueError("think_s must be at least one simulated second")
        if ref.get("snapshot"):
            with open(ref["snapshot"], "rb") as f:
                blob = f.read()
            head = pickle.loads(zlib.decompress(blob))
            ensure_modules([m["name"] for m in head["spec"]["modules"]])
            self.world = prepare_world(World.from_blob(blob, world_id=f"{head['world_id']}.e{head['seq']}", fork=True,
                                                       strict=strict), self.reward_mode)
        else:
            cfg = {k: ref[k] for k in ("family", "overrides", "extra_modules", "drop_modules") if k in ref}
            spec = prepare_spec(generate(cfg, int(ref.get("seed", 0))), self.reward_mode)
            ensure_modules([m["name"] for m in spec.modules])
            self.world = World.create(spec, strict=strict)
        w = self.world
        self.actor = f"p:{handle}"
        if w.principal(self.actor) is None:
            w.system_call("admin.join", {"handle": handle, "endowment": int(endowment)})
        self.think_s = int(think_s)
        self.max_events = max_events
        self.cursor = w.seq
        self.t_end = None if horizon_days is None else w.now + int(round(float(horizon_days) * DAY))
        self.objective = resolve(objective)(**(objective_kwargs or {})) if objective else None
        if self.objective is not None:
            self.objective.begin(w, self.actor)
        self._cash = self._balance()
        self._tools: tuple[int, list] | None = None

    def _balance(self) -> int:
        st = self.world.state
        if "accounts" not in st.tables:
            return 0
        return sum(st.tables["accounts"][a]["balance"] for a in st.find_ids("accounts", "owner", self.actor))

    def _observe(self) -> dict:
        w = self.world
        obs = w.observe(self.actor)
        evs, dropped = w.visible_events(self.actor, self.cursor, limit=self.max_events)
        self.cursor = w.seq
        obs["events"] = [{"seq": e.seq, "t": e.t, "type": e.type, "actor": e.actor, "payload": e.payload}
                         for e in evs if not (e.type == "cmd" and e.actor == self.actor)]
        obs["events_dropped"] = dropped
        return obs

    def tools(self) -> list[dict]:
        n = len(self.world.state.find_ids("knowledge", "principal", self.actor))
        if self._tools is None or self._tools[0] != n:
            self._tools = (n, self.world.available_tools(self.actor))
        return self._tools[1]

    def reset(self) -> dict:
        return self._observe()

    def step(self, action: dict | None) -> tuple[dict, float, bool, dict]:
        w = self.world
        t0 = w.now
        res = None
        dt = self.think_s
        if action is not None:
            r = w.execute(self.actor, action)
            res = r.to_dict()
            dt += max(1, int(r.cost_s))
        target = t0 + dt if self.t_end is None else min(t0 + dt, max(self.t_end, t0))
        if target > w.now:
            w.advance_time(target)
        if self.objective is not None:
            reward = float(self.objective.step_reward(w, self.actor))
        else:
            cash = self._balance()
            reward, self._cash = float(cash - self._cash), cash
        if self.reward_mode != "legacy_shaped_reward":
            reward = 0.0
            if self.reward_mode == "verified_terminal_outcome" and self.objective is not None and not self._paid \
                    and (self.objective.status(w, self.actor) or {}).get("success"):
                self._paid, reward = True, 1.0
        done = self.t_end is not None and w.now >= self.t_end
        return self._observe(), reward, done, {"sim_dt": w.now - t0, "result": res, "t": w.now}

    def verify(self) -> dict:
        """Evaluator side: the objective's status (empty without an objective)."""
        if self.objective is None or not hasattr(self.objective, "status"):
            return {}
        return dict(self.objective.status(self.world, self.actor) or {})

    def digest(self) -> str:
        return self.world.digest()

    def close(self) -> None:
        self.world.close()


def make_world_env(ref: dict, **kwargs) -> WorldEnv:
    """Default `make_world`: a `WorldEnv`. ``ref["env"]`` (a dict) overrides the farm-wide `world_kwargs`."""
    return WorldEnv(ref, **{**kwargs, **(ref.get("env") or {})})


def class_policy(params: bytes, cls: str = "crucible.agents.scripted:RandomPolicy", kwargs: dict | None = None):
    """Default `make_policy`: instantiate a policy class named as ``"module:Class"``. The parameter blob is handed
    to ``policy.load_params(params)`` if the class has it and is always kept as ``policy.params``."""
    pol = resolve(cls)(**(kwargs or {}))
    if hasattr(pol, "load_params"):
        pol.load_params(params)
    pol.params = params
    return pol


def _world_key(ref: dict) -> str:
    if ref.get("key") is not None:
        return str(ref["key"])
    return json.dumps(ref, sort_keys=True, default=str)


# ====================================================================== worker side
class _RolloutHandler:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        ensure_registered(cfg.get("imports") or (), discover=cfg.get("discover", True))
        self.make_world = resolve(cfg["make_world"])
        self.make_policy = resolve(cfg["make_policy"])
        self.obs_fn = resolve(cfg["obs_fn"]) if cfg.get("obs_fn") else None
        self.envs: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
        self.params: bytes | None = None
        self.shared = None                           # the policy object when policy_scope == "worker"
        self.version: int | None = None
        self.params_hash = ""
        self.steps_total = 0

    def handle(self, task: dict) -> dict:
        import numpy as np
        t_wall = time.perf_counter()
        cfg = self.cfg
        if task["params"] is not None:
            self.params, self.version, self.shared = task["params"], task["version"], None
            self.params_hash = hashlib.blake2b(task["params"], digest_size=8).hexdigest()
        elif self.params is None or self.version != task["version"]:
            raise RuntimeError(f"worker holds parameter version {self.version}, task needs {task['version']} "
                               f"and carries no blob")
        key = task["key"]
        entry = self.envs.get(key)
        fresh = False
        evicted: list[str] = []
        if entry is not None and (task["reset"] or task["episodic"] or entry["done"]):
            self._close(key)
            entry = None
        if entry is None:
            env = self.make_world(task["ref"], **(cfg.get("world_kwargs") or {}))
            entry = {"env": env, "obs": env.reset(), "done": False, "policy_version": None, "policy": None}
            self.envs[key] = entry
            fresh = True
            while len(self.envs) > max(1, int(cfg.get("max_live_worlds", 4))):
                old = next(iter(self.envs))
                self._close(old)
                evicted.append(old)
        self.envs.move_to_end(key)
        env = entry["env"]
        # a policy object per world by default (scripted and other stateful policies keep per-world state);
        # policy_scope="worker" shares one object between the worlds of a worker (one model, many worlds)
        if entry["policy_version"] != self.version or fresh:
            try:
                if cfg.get("policy_scope", "world") == "worker":
                    if self.shared is None:
                        self.shared = self.make_policy(self.params, **(cfg.get("policy_kwargs") or {}))
                    entry["policy"] = self.shared
                else:
                    entry["policy"] = self.make_policy(self.params, **(cfg.get("policy_kwargs") or {}))
                if hasattr(entry["policy"], "reset"):
                    entry["policy"].reset(getattr(env, "actor", key),
                                          np.random.default_rng([int(task["seed"]), int(task["id"])]))
            except BaseException:
                self._close(key)                     # the farm no longer routes this key here: do not keep the world
                raise
            entry["policy_version"] = self.version
        policy = entry["policy"]
        setup_s = time.perf_counter() - t_wall
        record_obs, obs_fn = cfg.get("record_obs", True), self.obs_fn
        has_tools, tell = hasattr(env, "tools"), hasattr(policy, "observe_result")
        out_obs, actions, rewards, results, sim_dt, sim_t = [], [], [], [], [], []
        obs = entry["obs"]
        cap = task["steps"] if task["steps"] is not None else int(cfg.get("max_steps_per_task", 100_000))
        done = False
        policy_s = env_s = 0.0
        try:
            for _ in range(cap):
                t = time.perf_counter()
                action = policy.act(obs, env.tools() if has_tools else None)
                t2 = time.perf_counter()
                nxt, reward, done, info = env.step(action)
                t3 = time.perf_counter()
                policy_s += t2 - t
                env_s += t3 - t2
                if tell and action is not None and info.get("result") is not None:
                    policy.observe_result(action, info["result"])
                if record_obs:
                    out_obs.append(obs_fn(obs) if obs_fn else obs)
                actions.append(action)
                rewards.append(float(reward))
                results.append(info.get("result"))
                dt = int(info.get("sim_dt", 0))
                sim_dt.append(dt)
                sim_t.append(int(info["t"]) - dt if "t" in info else None)
                obs = nxt
                if done:
                    break
        except BaseException:
            self._close(key)                         # a world that raised is not continued
            raise
        entry["obs"], entry["done"] = obs, bool(done)
        closed = bool(done) or task["episodic"]
        if closed:
            self._close(key)
        n = len(actions)
        self.steps_total += n
        return {"key": key, "fresh_world": fresh, "done": bool(done),
                "truncated": task["steps"] is None and not done, "steps": n,
                "obs": out_obs, "actions": actions, "rewards": rewards, "results": results, "sim_dt": sim_dt,
                "sim_t": sim_t, "final_obs": (obs_fn(obs) if obs_fn else obs) if record_obs else None,
                "sim_seconds": int(sum(sim_dt)), "wall_s": time.perf_counter() - t_wall, "policy_s": policy_s,
                "env_s": env_s, "setup_s": setup_s, "param_version": self.version, "params_hash": self.params_hash,
                "closed": closed, "evicted": evicted, "worker_pid": os.getpid(),
                "worker_steps_total": self.steps_total, "live_worlds": len(self.envs)}

    def _close(self, key: str) -> None:
        entry = self.envs.pop(key, None)
        if entry is not None:
            try:
                entry["env"].close()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        for key in list(self.envs):
            self._close(key)


# ====================================================================== consumer side
class RolloutFarm:
    """Asynchronous rollout workers with versioned parameters, backpressure and exact accounting (module docstring).

    Parameters
    ----------
    n_workers : worker processes (default: up to 8).
    make_world : ``"module:callable"``, called in the worker as ``make_world(ref, **world_kwargs)``.
    make_policy : ``"module:callable"``, called in the worker as ``make_policy(params_bytes, **policy_kwargs)``.
        The result needs ``act(obs, tools) -> action``; ``reset(actor, rng)`` and ``observe_result(action,
        result)`` are used when present (the `agents.base.Policy` interface).
    policy_scope : ``"world"`` (default) builds one policy object per world and parameter version, so policies
        with internal state (scripts, memories) never mix worlds; ``"worker"`` builds one per worker and version
        and shares it between that worker's worlds (one model, many worlds; it is `reset` for each new world).
    params : initial parameter blob (version 0).
    queue_size : finished trajectories buffered for the consumer before workers are made to wait
        (default ``2 * n_workers``).
    steps_per_task : default for `submit`; ``None`` means a whole episode.
    max_tasks_per_worker : recycle a worker after this many tasks (its persistent worlds are rebuilt from their
        references on next use).
    task_timeout_s : wall-clock limit per task; the worker is killed and the task reported failed.
    record_obs / obs_fn : whether observations are returned, and an optional ``"module:callable"`` applied to each
        observation in the worker before it is recorded (the policy always sees the raw observation).
    """

    def __init__(self, n_workers: int | None = None, make_world: str = "crucible.scale.rollout:make_world_env",
                 make_policy: str = "crucible.scale.rollout:class_policy", *, world_kwargs: dict | None = None,
                 policy_kwargs: dict | None = None, params: bytes = b"", queue_size: int | None = None,
                 steps_per_task: int | None = None, max_tasks_per_worker: int | None = None,
                 task_timeout_s: float | None = None, start_method: str = "spawn", imports: tuple = (),
                 discover: bool = True, record_obs: bool = True, obs_fn: str | None = None, max_live_worlds: int = 4,
                 max_steps_per_task: int = 100_000, seed: int = 0, policy_scope: str = "world",
                 name: str = "rollout"):
        if policy_scope not in ("world", "worker"):
            raise ValueError("policy_scope is 'world' or 'worker'")
        self.steps_per_task = steps_per_task
        self.task_timeout_s = task_timeout_s
        self.seed = int(seed)
        self._lock = threading.Lock()
        self._todo: collections.deque[dict] = collections.deque()
        self._inflight: dict[int, tuple[Any, dict]] = {}
        self._owner: dict[str, Any] = {}             # world key -> the worker process holding that world
        self._version = 0
        self._params = bytes(params)
        self._published: dict[int, float] = {0: time.monotonic()}
        self._next_id = 1
        self._c = collections.Counter()
        self._by_version: collections.Counter = collections.Counter()
        self._stale_sum = 0
        self._stale_max = 0
        self._worker_steps: dict[tuple[int, int], int] = {}
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._abandoned: list[int] = []
        self._t0 = time.monotonic()
        cfg = {"make_world": make_world, "make_policy": make_policy, "world_kwargs": dict(world_kwargs or {}),
               "policy_kwargs": dict(policy_kwargs or {}), "imports": list(imports), "discover": discover,
               "record_obs": record_obs, "obs_fn": obs_fn, "max_live_worlds": max_live_worlds,
               "max_steps_per_task": max_steps_per_task, "policy_scope": policy_scope}
        import pickle
        pickle.dumps(cfg, protocol=5)                # everything a worker is built from must be declarative
        resolve(make_world)
        resolve(make_policy)
        self._sup = Supervisor(n_workers, "crucible.scale.rollout:_RolloutHandler", cfg, start_method=start_method,
                               max_jobs_per_worker=max_tasks_per_worker, name=name)
        self.n_workers = self._sup.n_workers
        self.queue_size = int(queue_size) if queue_size else 2 * self.n_workers
        self._out: "queue.Queue[Rollout]" = queue.Queue(maxsize=self.queue_size)
        self._wake_r, self._wake_w = os.pipe()
        os.set_blocking(self._wake_r, False)
        os.set_blocking(self._wake_w, False)
        self._closed = False
        self._thread = threading.Thread(target=self._pump, name=f"{name}-pump", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ consumer API
    @property
    def version(self) -> int:
        """Newest published parameter version."""
        return self._version

    def publish(self, params: bytes) -> int:
        """Make `params` the newest policy parameters and return their version. Tasks handed to workers from now
        on use them; tasks already running finish with the version they started with."""
        blob = bytes(params)
        with self._lock:
            self._version += 1
            self._params = blob
            self._published[self._version] = time.monotonic()
            v = self._version
        self._wake()
        return v

    def submit(self, world: dict | str, steps: int | None = None, *, reset: bool = False, tag: Any = None,
               seed: int | None = None) -> int:
        """Queue one task and return its id.

        `world` is a world reference (for the default `make_world`: a family name or ``{"family", "seed",
        "overrides", "extra_modules"}`` or ``{"snapshot": file}``, optionally with ``"key"`` naming the world).
        ``steps=k`` runs k agent steps, continuing the world of that key if a worker holds it (``reset=True``
        rebuilds it first). ``steps=None`` (and no farm default) runs one whole episode in a fresh world, which is
        closed afterwards."""
        if self._closed:
            raise RuntimeError("farm is closed")
        if self._error is not None:
            raise self._error
        ref = {"family": world} if isinstance(world, str) else dict(world)
        steps = self.steps_per_task if steps is None else steps
        if steps is not None and int(steps) < 1:
            raise ValueError("steps must be >= 1 (or None for a whole episode)")
        with self._lock:
            tid = self._next_id
            self._next_id += 1
            self._todo.append({"id": tid, "key": _world_key(ref), "ref": ref,
                               "steps": None if steps is None else int(steps), "episodic": steps is None,
                               "reset": bool(reset), "tag": tag, "seed": self.seed if seed is None else int(seed),
                               "t_submit": time.monotonic()})
            self._c["submitted"] += 1
        self._wake()
        return tid

    @property
    def outstanding(self) -> int:
        """Tasks submitted whose result the consumer has not received yet."""
        with self._lock:
            return self._c["submitted"] - self._c["delivered"]

    def get(self, timeout: float | None = None) -> Rollout:
        """The next finished task (trajectory or failure report), in arrival order. Blocks; raises `queue.Empty`
        after `timeout` seconds and `LookupError` if nothing is outstanding."""
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.outstanding <= 0:
                raise LookupError("no outstanding tasks")
            try:
                r = self._out.get(timeout=0.1 if end is None else max(0.0, min(0.1, end - time.monotonic())))
            except queue.Empty:
                if self._error is not None:
                    raise self._error from None
                if self._closed:
                    raise RuntimeError("farm is closed") from None
                if end is not None and time.monotonic() >= end:
                    raise
                continue
            with self._lock:
                self._c["delivered"] += 1
            return r

    def __iter__(self) -> Iterator[Rollout]:
        """Yield finished tasks as they arrive until nothing is outstanding. Tasks submitted from inside the loop
        keep it going."""
        while self.outstanding > 0:
            yield self.get()

    def wait_ready(self, timeout: float = 120.0) -> float:
        """Block until every worker process has finished importing. Returns the seconds waited."""
        t = time.perf_counter()
        while not self._ready.wait(0.1):
            if self._error is not None:
                raise self._error
            if time.perf_counter() - t > timeout:
                raise TimeoutError(f"rollout workers not ready after {timeout}s")
        return time.perf_counter() - t

    def stats(self) -> dict:
        """Accounting. Invariants: ``submitted == ok + failed + queued + in_flight (+ abandoned after close)``;
        ``steps`` is the sum of `steps` over the ok trajectories received from workers; ``steps_executed`` is the
        workers' own count for the tasks they finished (equal to ``steps``)."""
        with self._lock:
            c = dict(self._c)
            ok = c.get("ok", 0)
            return {"workers": self.n_workers, "submitted": c.get("submitted", 0), "dispatched": c.get("dispatched", 0),
                    "ok": ok, "failed": c.get("failed", 0), "delivered": c.get("delivered", 0),
                    "queued": len(self._todo), "in_flight": len(self._inflight), "buffered": self._out.qsize(),
                    "abandoned": len(self._abandoned), "steps": c.get("steps", 0),
                    "steps_executed": sum(self._worker_steps.values()), "sim_seconds": c.get("sim_seconds", 0),
                    "version": self._version, "by_version": dict(sorted(self._by_version.items())),
                    "staleness_mean": (self._stale_sum / ok) if ok else 0.0, "staleness_max": self._stale_max,
                    "spawned": self._sup.spawned, "died": self._sup.died, "recycled": self._sup.recycled,
                    "queue_size": self.queue_size, "elapsed_s": time.monotonic() - self._t0}

    # ------------------------------------------------------------------ pump thread
    def _wake(self) -> None:
        try:
            os.write(self._wake_w, b"x")
        except (BlockingIOError, OSError):
            pass

    def _pump(self) -> None:
        sup = self._sup
        try:
            while not self._stop.is_set():
                self._dispatch()
                events = sup.poll(0.2, extra=(self._wake_r,))
                try:
                    while os.read(self._wake_r, 4096):
                        pass
                except (BlockingIOError, OSError):
                    pass
                if not self._ready.is_set() and all(w.state != "starting" for w in sup.workers):
                    self._ready.set()
                for ev in events:
                    r = self._record(ev)
                    if r is not None:
                        while not self._stop.is_set():   # the bounded queue: block here, and so do the workers
                            try:
                                self._out.put(r, timeout=0.1)
                                break
                            except queue.Full:
                                continue
        except BaseException as e:  # noqa: BLE001 - PoolBroken or a bug: surface it in the consumer
            self._error = e

    def _dispatch(self) -> None:
        sup = self._sup
        idle = sup.idle()
        if not idle:
            return
        with self._lock:
            if not self._todo:
                return
            version, params = self._version, self._params
            for key, owner in list(self._owner.items()):     # worlds of workers that are gone
                if owner.state not in ("idle", "busy"):
                    del self._owner[key]
            for w in idle:
                task = None
                for i, t in enumerate(self._todo):
                    if i >= 1024:
                        break
                    owner = self._owner.get(t["key"])
                    if owner is None or owner is w:
                        task = t
                        del self._todo[i]
                        break
                if task is None:
                    continue
                msg = {k: task[k] for k in ("id", "key", "ref", "steps", "episodic", "reset", "seed")}
                msg["version"] = version
                msg["params"] = params if w.meta.get("version") != version else None
                if not sup.assign(w, task["id"], msg, self.task_timeout_s):
                    self._todo.appendleft(task)          # dead worker: poll() replaces it, the task was not started
                    continue
                w.meta["version"] = version
                task["version"], task["t_dispatch"] = version, time.monotonic()
                self._inflight[task["id"]] = (w, task)
                if not task["episodic"]:
                    self._owner[task["key"]] = w
                self._c["dispatched"] += 1

    def _record(self, ev: tuple) -> Rollout | None:
        kind = ev[0]
        if kind not in ("done", "died"):
            return None
        w, tid = ev[1], ev[2]
        with self._lock:
            if kind == "died":
                for key in [k for k, o in self._owner.items() if o is w]:
                    del self._owner[key]
            if tid is None or tid not in self._inflight:
                return None
            _, task = self._inflight.pop(tid)
            base = {"task_id": tid, "world_key": task["key"], "tag": task["tag"], "worker": w.slot,
                    "worker_pid": w.pid, "latest_version": self._version,
                    "queue_s": task["t_dispatch"] - task["t_submit"]}
            if kind == "done" and ev[3]:
                p = ev[4]
                if p["closed"]:
                    self._owner.pop(task["key"], None)
                for key in p["evicted"]:
                    if self._owner.get(key) is w:
                        del self._owner[key]
                self._worker_steps[(w.slot, w.gen)] = p["worker_steps_total"]
                r = Rollout(ok=True, **base, **{k: p[k] for k in (
                    "param_version", "params_hash", "fresh_world", "done", "truncated", "steps", "obs", "actions",
                    "rewards", "results", "sim_dt", "sim_t", "final_obs", "sim_seconds", "wall_s", "policy_s",
                    "env_s", "setup_s")})
                self._c["ok"] += 1
                self._c["steps"] += r.steps
                self._c["sim_seconds"] += r.sim_seconds
                self._by_version[r.param_version] += 1
                self._stale_sum += r.staleness
                self._stale_max = max(self._stale_max, r.staleness)
                return r
            self._owner.pop(task["key"], None)           # the world of a failed task is not continued
            w.meta["version"] = None                     # the failure may have been in building the policy: resend
            self._c["failed"] += 1
            if kind == "done":
                return Rollout(ok=False, **base, param_version=task["version"], kind="exception", error=ev[4])
            evidence = ev[3]
            return Rollout(ok=False, **base, param_version=task["version"], kind=evidence["kind"], evidence=evidence,
                           error={"type": "WorkerDied", "message": f"worker {w.slot} (pid {w.pid}) died with exit "
                                  f"code {evidence['exitcode']} (signal {evidence['signal']}) after "
                                  f"{evidence['ran_s']}s"})

    # ------------------------------------------------------------------ shutdown
    def close(self) -> None:
        """Stop the farm and its workers. Tasks not yet returned are abandoned (their ids: `abandoned`)."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wake()
        self._thread.join(15)
        with self._lock:
            self._abandoned = sorted([t["id"] for t in self._todo] + list(self._inflight))
            busy = bool(self._inflight)
            self._todo.clear()
            self._inflight.clear()
        if busy or self._thread.is_alive():
            self._sup.terminate()
        else:
            self._sup.close(timeout=10.0)
        for fd in (self._wake_r, self._wake_w):
            try:
                os.close(fd)
            except OSError:
                pass

    terminate = close

    @property
    def abandoned(self) -> list[int]:
        return list(self._abandoned)

    def __enter__(self) -> "RolloutFarm":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

