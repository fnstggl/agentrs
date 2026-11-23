"""The same task through the kernel: a numeric learner operating a facility with `rig.*` tool calls.

`KernelMechEnv(task)` builds the kernel world of a task (`Task.world_config`), admits one participant with the
task's starting balance, and exposes exactly the observation, action and reward of `MechVecEnv` for one slot -
but every reading comes from a `rig.read` tool result, every setting goes through `rig.set`, the balance is read
with `bank.accounts`, and time passes by advancing the world clock one tick. `tests/test_research_envs.py` runs a
learner through both and requires identical observations and rewards: this is the evidence that a learner using
the batched simulator and an agent using tools face the same problem with the same information and permissions.

`ToolLog` records every tool call the learner's actions were translated into, so a run is an ordinary agent
trajectory in the world's event log (replayable, auditable, diagnosable with `research.diagnostics`).
"""
from __future__ import annotations

import numpy as np

from .envs import money_scale, symlog
from .learner import EnvSpec
from .tasks import Task


class KernelMechEnv:
    n = 1

    def __init__(self, task: Task, spec: EnvSpec | None = None, handle: str = "agent", tools: bool = False,
                 store=None, strict: bool = True):
        import crucible
        crucible.load_all_modules()
        from ..core.kernel import World
        from ..worldgen import generate
        self.task, self.spec = task, spec or EnvSpec()
        cfg, seed = task.world_config(tools=tools, holders=(handle,))
        self.world = World.create(generate(cfg, seed), world_id=f"task-{task.family}-{task.seed}-{handle}",
                                  store=store, strict=strict)
        r = self.world.system_call("admin.join", {"handle": handle, "endowment": int(task.endowment)})
        self.actor = r["principal"]
        self.calls: list[dict] = []
        sp = self._call("rig.spec", {})
        self.tick_s = int(sp["tick_s"])
        self.levers, self.gauges = sp["levers"], [g["name"] for g in sp["gauges"]]
        if len(self.levers) > self.spec.max_levers or len(self.gauges) > self.spec.max_gauges:
            raise ValueError("task does not fit the EnvSpec")
        self.horizon = int(task.horizon)
        self.scale = money_scale(task)
        self.t = 0
        self.done = np.ones(1, dtype=bool)

    # ------------------------------------------------------------------ tool calls
    def _call(self, tool: str, args: dict):
        res = self.world.execute(self.actor, {"tool": tool, "args": args})
        self.calls.append({"t": self.world.now, "tool": tool, "args": args, "ok": res.ok,
                           "error": None if res.ok else res.error})
        if not res.ok:
            raise RuntimeError(f"{tool} failed: {res.error}")
        return res.result

    def _balance(self) -> int:
        return int(sum(a["balance"] for a in self._call("bank.accounts", {}) if a["owner"] == self.actor))

    def _obs(self, report: dict | None, settings: dict) -> np.ndarray:
        sp = self.spec
        o = np.zeros((1, sp.obs_dim), dtype=np.float32)
        if report is not None:
            g = np.array([report["readings"][name] for name in self.gauges], dtype=np.float64)
            o[0, sp.gauges][:len(g)] = symlog(g)
        o[0, sp.gauge_mask][:len(self.gauges)] = 1.0
        for j, lv in enumerate(self.levers):
            span = lv["max"] - lv["min"]
            o[0, sp.last_action][j] = 0.0 if span <= 0 else 2.0 * (settings[lv["name"]] - lv["min"]) / span - 1.0
        o[0, sp.lever_mask][:len(self.levers)] = 1.0
        sc = o[0, sp.scalars]
        if report is not None:
            sc[0] = report["net_cents"] / self.scale
            sc[1] = report["revenue_cents"] / self.scale
            sc[2] = report["cost_cents"] / self.scale
        sc[3] = self._balance() / max(1.0, float(self.task.endowment))
        sc[4] = self.t / float(self.horizon)
        return o

    # ------------------------------------------------------------------ gym-like API (one slot)
    def reset(self) -> np.ndarray:
        if self.t != 0:
            raise RuntimeError("a kernel world is not rewound: build a new KernelMechEnv (or fork the world)")
        rd = self._call("rig.read", {})
        self.done[:] = False
        self.net = 0.0
        return self._obs(None, rd["next_settings"])

    def step(self, actions: np.ndarray):
        a = np.clip(np.asarray(actions, dtype=np.float64).reshape(-1), -1.0, 1.0)
        settings = {lv["name"]: float(lv["min"] + (a[j] + 1.0) * 0.5 * (lv["max"] - lv["min"]))
                    for j, lv in enumerate(self.levers)}
        self._call("rig.set", {"settings": settings})
        self.world.advance_time(self.world.now + self.tick_s)
        rd = self._call("rig.read", {})
        rep = rd["report"]
        self.t += 1
        self.net += rep["net_cents"]
        self.done[:] = self.t >= self.horizon
        obs = self._obs(rep, rd["next_settings"])
        return obs, np.array([rep["net_cents"] / self.scale], dtype=np.float32), self.done.copy(), \
            {"net_cents": np.array([float(rep["net_cents"])]), "status": rep["status"]}

    def summary(self) -> dict:
        t = self.task
        return {"task": t.id, "net_cents": float(self.net), "net_per_tick": float(self.net / max(1, self.t)),
                "score": t.score(float(self.net / max(1, self.t))), "success": t.success(float(self.net)),
                "tool_calls": len(self.calls), "world_digest": self.world.digest()}
