"""Melting Pot (google-deepmind/meltingpot 2.4.1, Apache-2.0) worlds, driven through ``mp_server.py`` processes that
run in Melting Pot's own CPU-only interpreter. Server processes are pooled and reused across trials.

Families:
* ``mp/<substrate>``: every player is a learner-controlled stream (self-play with shared parameters; each player
  has its own belief and memory and sees only its own permitted observations).
* ``mpsc/<scenario>``: only the scenario's focal players are learner-controlled; background players are the
  benchmark's held-out bot populations (unfamiliar co-players). Evaluation only.

Verified outcome: the agent's own episode return (``score``); Melting Pot defines no binary success. Reward is the
substrate's native reward (no shaping).
"""
from __future__ import annotations

import atexit
import os
import pickle
import struct
import subprocess
from pathlib import Path

import numpy as np

from agentrs.sl.world import ActionSpace, Consequence, GoalStatus, Obs, Task, World

PY = "/workspace/ext/venv-mp/bin/python"
SERVER = str(Path(__file__).with_name("mp_server.py"))
_POOL: dict = {}


class _Server:
    def __init__(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "CUDA"))}
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="3", PYTHONDONTWRITEBYTECODE="1")
        self.p = subprocess.Popen([PY, "-I", SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, env=env)
        self.key = None
        self.pending = False

    def send(self, msg):
        b = pickle.dumps(msg, protocol=4)
        self.p.stdin.write(struct.pack("<Q", len(b)) + b)
        self.p.stdin.flush()

    def recv(self):
        h = self.p.stdout.read(8)
        if len(h) < 8:
            raise RuntimeError("Melting Pot server died")
        (n,) = struct.unpack("<Q", h)
        st, val = pickle.loads(self.p.stdout.read(n))
        if st != "ok":
            raise RuntimeError(f"Melting Pot server error: {val}")
        return val

    def call(self, msg):
        self.send(msg)
        return self.recv()

    def close(self):
        try:
            self.send(("close", None))
            self.p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.p.kill()


def _acquire(key):
    lst = _POOL.setdefault(key, [])
    return lst.pop() if lst else None


MAX_IDLE = 4  # idle servers kept per (kind, name); extras are shut down


def _release(key, srv):
    lst = _POOL.setdefault(key, [])
    if len(lst) < MAX_IDLE:
        lst.append(srv)
    else:
        srv.close()


@atexit.register
def _shutdown():
    for lst in _POOL.values():
        for s in lst:
            s.close()


class MeltingPotWorld(World):
    reward_overrides = {"native": 1.0, "shaping": 0.0, "success": 0.0, "violation": 0.0}

    def __init__(self, kind: str, name: str):
        self.kind, self.name = kind, name
        self.family = ("mp/" if kind == "substrate" else "mpsc/") + name
        self.srv = None

    def initialize_world(self, task: Task) -> None:
        self.task = task
        key = (self.kind, self.name)
        self.srv = _acquire(key)
        if self.srv is None:
            self.srv = _Server()
            r = self.srv.call(("init", {"kind": self.kind, "name": self.name, "seed": task.seed}))
        else:
            r = self.srv.call(("reset", None))
            r.update(self.srv.spec)
        self.srv.spec = {"n": r["n"], "labels": r["labels"]}
        self.n, self.labels = r["n"], r["labels"]
        self.lab_idx = {l: i for i, l in enumerate(self.labels)}
        self._set(r["obs"], [0.0] * self.n, False)
        self.controlled = [str(i) for i in range(self.n)]

    def _set(self, obs, rewards, last):
        self.obs, self.rew, self.last = obs, rewards, last
        if not hasattr(self, "ret") or rewards is None:
            self.ret = [0.0] * self.n
        self.actions = [0] * self.n

    def agents(self) -> list:
        return [str(i) for i in range(self.n)]

    def reset_world(self) -> None:
        r = self.srv.call(("reset", None))
        self.ret = [0.0] * self.n
        self._set(r["obs"], [0.0] * self.n, False)

    def observe(self, agent) -> Obs:
        self._sync()
        o = self.obs[int(agent)]
        return Obs(numbers=dict(o["nums"]), image=o["cells"])

    def goal(self, agent) -> Obs:
        return Obs(tokens=["goal:own_return", f"world:{self.name}"])

    def available_actions(self, agent) -> ActionSpace:
        return ActionSpace(kind="discrete", labels=self.labels)

    def execute(self, agent, action) -> None:
        self.actions[int(agent)] = self.lab_idx[action] if isinstance(action, str) else int(action)

    def advance_time(self) -> None:
        self.srv.send(("step", self.actions))  # asynchronous: other worlds step in parallel
        self.srv.pending = True

    def _sync(self):
        if self.srv is not None and self.srv.pending:
            self.srv.pending = False
            r = self.srv.recv()
            self.obs, self.rew, self.last = r["obs"], r["rewards"], r["last"]
            self.ret = [a + b for a, b in zip(self.ret, self.rew)]

    def observe_consequences(self, agent) -> Consequence:
        self._sync()
        return Consequence(reward=float(self.rew[int(agent)]), dt=1.0, done=bool(self.last))

    def evaluate_goal(self, agent) -> GoalStatus:
        self._sync()
        return GoalStatus(success=False, progress=0.0, score=float(self.ret[int(agent)]), time_left=1.0)

    def close(self) -> None:
        if self.srv is not None:
            self._sync()
            _release((self.kind, self.name), self.srv)
            self.srv = None


def available() -> bool:
    return Path(PY).exists()
