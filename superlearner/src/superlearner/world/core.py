"""The trusted side of an episode: the world, the clock and the evaluator. PROTECTED (not candidate-editable).

`SessionCore` owns a Crucible world, the actor's identity and the task's `Evaluator`. A learner - in the same
process (`SessionCore` directly) or in a sandboxed candidate process (`RemoteCore` over a pipe to `serve`) - can
only do what an in-world actor can do, through this protocol:

    info()                     public facts: goal text, actor, horizon, time, what the start state created
    observe()                  the actor's own view (`World.observe` + events visible to it)
    tools()                    tool specs the actor may call (operator-only ``admin.*`` removed)
    execute(tool, args)        one recorded command; the clock then advances by the tool's duration
    wait(seconds)              let simulated time pass
    end_decision()             think-time passes; returns the reward from the evaluator and done flags
    summary()                  evaluator status at the end (success is the task predicate on authoritative state)

The learner never receives the world object, the seed of an evaluation task, the evaluator, or reward code.

Time. Every executed command costs its tool's ``cost_s`` (at least 1 s); every decision costs ``think_s``.
An episode terminates on verified success or when the task horizon has elapsed, and is truncated when the
decision budget is exhausted.

Reward (`Evaluator`). r = shaping * (Phi(s_t) - Phi(s_{t-1})) + bonus * [first verified success], with
Phi = (1 - w) * contract progress + w * ladder, both read from authoritative state. ``w = 0`` (default) uses the
task contract alone. There is no reward for valid calls, activity, code volume or claims.
"""
from __future__ import annotations

import os
import time
import traceback
from dataclasses import asdict, dataclass

from .. import ensure_crucible
from .tasks import TaskSpec, build, check_split

ensure_crucible()
from crucible.goals.contracts import GoalVerifier  # noqa: E402
from crucible.goals.metrics import evaluate_metric  # noqa: E402
from crucible.integrations import ActorView, agent_tools  # noqa: E402

DAY = 86400


@dataclass
class CoreConfig:
    think_s: int = 60
    max_decisions: int = 60
    max_calls_per_decision: int = 16
    max_wait_s: int = 3 * DAY
    shaping: float = 1.0
    success_bonus: float = 1.0
    ladder_weight: float = 0.0
    max_events: int = 30


class Evaluator:
    """Independent verification and reward for one actor and one goal. Lives only on the trusted side."""

    def __init__(self, goal, actor: str, cfg: CoreConfig):
        self.goal, self.actor, self.cfg = goal, actor, cfg
        self.v = GoalVerifier(goal, actor)
        self.phi = 0.0
        self.best = 0.0
        self.paid = False
        self.next_check = 0
        self.ret = 0.0

    def _phi(self, world) -> float:
        p = self.v.progress(world)
        self.best = max(self.best, p)
        w = self.cfg.ladder_weight
        if w > 0:
            p = (1 - w) * p + w * evaluate_metric(world, "sl_ladder", {"principal": self.actor})
        return float(p)

    def begin(self, world) -> None:
        self.v.begin(world)
        self.phi = self._phi(world)
        self.next_check = world.now + 6 * 3600

    def step(self, world) -> float:
        # the predicate is checked every 6 simulated hours (and at every decision for short tasks)
        while world.now >= self.next_check:
            self.v.daily_check(world)
            self.next_check += 6 * 3600
        if self.v.success_t is None and self.goal.horizon_s <= 5 * DAY:
            self.v.daily_check(world)
        phi = self._phi(world)
        r = self.cfg.shaping * (phi - self.phi)
        self.phi = phi
        if self.v.success_t is not None and not self.paid:
            self.paid = True
            r += self.cfg.success_bonus
        self.ret += r
        return float(r)

    def status(self, world) -> dict:
        s = self.v.status(world)
        s["best_progress"] = max(self.best, s["progress"])
        s["return"] = self.ret
        return s


class SessionCore:
    def __init__(self, spec: TaskSpec, cfg: CoreConfig | None = None, handle: str = "agent"):
        self.spec, self.cfg = spec, cfg or CoreConfig()
        t0 = time.time()
        self.goal, self.world, self.actor, self.start_info = build(spec, handle)
        self.view = ActorView(self.world, self.actor, self.cfg.max_events)
        self.evaluator = Evaluator(self.goal, self.actor, self.cfg)
        self.evaluator.begin(self.world)
        self.t_start = self.world.now
        self.n_decisions = 0
        self.calls_this_decision = 0
        self.n_commands = 0
        self.n_ok = 0
        self.done = False
        self.commands: list[tuple] = []       # trusted record: (t, tool, ok, error code)
        self._tools_sig = None
        self.build_s = time.time() - t0
        self.sim_s = 0.0

    # ------------------------------------------------------------------ protocol
    def info(self) -> dict:
        hist = [{"tool": a["tool"], "args": a.get("args", {}), "result": r} for a, r in self.start_info.get("history", [])]
        return {"goal": self.goal.description, "actor": self.actor, "horizon_s": self.goal.horizon_s,
                "time": self.world.now, "start": self.start_info.get("stage", "scratch"), "start_history": hist,
                "max_decisions": self.cfg.max_decisions, "task_family": self.spec.family,
                "world_family": self.spec.world_family, "split": self.spec.split}

    def observe(self) -> dict:
        return self.view.observe()

    def tools(self, force: bool = False):
        tl = agent_tools(self.world, self.actor)
        sig = (len(tl), tl[-1]["name"] if tl else "")
        if not force and sig == self._tools_sig:
            return None
        self._tools_sig = sig
        return tl

    def _advance(self, seconds: int) -> None:
        t0 = time.time()
        self.world.advance_time(self.world.now + max(0, int(seconds)))
        self.sim_s += time.time() - t0

    def execute(self, tool, args) -> dict:
        if self.done:
            return {"ok": False, "tool": str(tool), "cost_s": 0, "error": {"code": "budget_exhausted", "message": "episode is over"}}
        if self.calls_this_decision >= self.cfg.max_calls_per_decision:
            return {"ok": False, "tool": str(tool), "cost_s": 0,
                    "error": {"code": "budget_exhausted", "message": "too many calls in one decision"}}
        self.calls_this_decision += 1
        r = self.world.execute(self.actor, {"tool": tool, "args": args})
        self.n_commands += 1
        self.n_ok += int(r.ok)
        self.commands.append((self.world.now, r.tool, bool(r.ok), None if r.ok else (r.error or {}).get("code")))
        d = r.to_dict()
        self._advance(max(1, int(r.cost_s)))
        return d

    def wait(self, seconds: int) -> int:
        s = int(min(max(1, int(seconds)), self.cfg.max_wait_s))
        if not self.done:
            self._advance(s)
        return s

    def end_decision(self) -> dict:
        if self.done:
            return {"reward": 0.0, "terminated": True, "truncated": False, "time": self.world.now}
        self._advance(self.cfg.think_s)
        self.n_decisions += 1
        self.calls_this_decision = 0
        r = self.evaluator.step(self.world)
        st = self.evaluator.v
        elapsed = self.world.now - self.t_start
        terminated = st.success_t is not None or elapsed >= self.goal.horizon_s
        truncated = (not terminated) and self.n_decisions >= self.cfg.max_decisions
        self.done = terminated or truncated
        return {"reward": r, "terminated": terminated, "truncated": truncated, "time": self.world.now,
                "elapsed_frac": min(1.0, elapsed / max(1, self.goal.horizon_s)), "decisions": self.n_decisions}

    def summary(self) -> dict:
        s = self.evaluator.status(self.world)
        fails: dict[str, int] = {}
        for _, _, okv, code in self.commands:
            if not okv:
                fails[code or "error"] = fails.get(code or "error", 0) + 1
        return {"task": self.spec.key, "arm": list(self.spec.arm), "split": self.spec.split,
                "world_family": self.spec.world_family, "success": bool(s["success"]),
                "progress": float(s["progress"]), "best_progress": float(s["best_progress"]), "return": float(s["return"]),
                "elapsed_s": int(s["elapsed_s"]), "decisions": self.n_decisions, "commands": self.n_commands,
                "commands_ok": self.n_ok, "world_errors": fails, "sim_wall_s": round(self.sim_s, 3),
                "build_wall_s": round(self.build_s, 3), "world_events": self.world.seq}

    def command_log(self) -> list:
        return list(self.commands)


# ================================================================================================ remote form
def _append(path: str, rec: dict) -> None:
    import fcntl
    import json
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def serve(conn, allow_splits: tuple = ("train", "curriculum"), eval_tasks: list | None = None,
          cfg: dict | None = None, eval_cfg: dict | None = None, results_path: str | None = None,
          shared: dict | None = None, budget: dict | None = None) -> None:
    """Run sessions for a learner at the other end of `conn` (a `multiprocessing.connection`). TRUSTED process.

    ``("reset", {"spec": {...}})`` starts a task of an allowed split chosen by the learner (training tasks).
    ``("reset", {"eval": i})`` starts the i-th task of the server's own evaluation list; the learner is never
    told its seed. Every other message is a `SessionCore` method call: ``(name, args)``.

    With `results_path`, every finished episode's evaluator summary is appended to that file by THIS process
    (the learner cannot write there), stamped with the number of training decisions the hub has served so far
    (`shared` counters across all connections). Learning curves used for comparing algorithms are computed from
    that file only. `budget` = {"train_decisions": n} makes the hub refuse new training tasks once n training
    decisions have been served, so no learner can use more experience than its budget.
    """
    core: SessionCore | None = None
    results: list = []
    kind, eval_index = "train", None
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        op, a = msg if isinstance(msg, tuple) and len(msg) == 2 else ("bad", None)
        try:
            if op == "close":
                conn.send(("ok", results))
                break
            if op == "reset":
                if "eval" in a:
                    eval_index = int(a["eval"])
                    spec = eval_tasks[eval_index]
                    core = SessionCore(spec, CoreConfig(**(eval_cfg or cfg or {})))
                    kind = "eval"
                else:
                    spec = TaskSpec.from_dict(a["spec"])
                    check_split(spec)
                    if spec.split not in allow_splits:
                        raise PermissionError(f"split {spec.split!r} cannot be requested by a learner")
                    if budget and shared is not None and shared["train_decisions"].value >= budget.get("train_decisions", 1 << 62):
                        raise PermissionError("training budget exhausted")
                    core = SessionCore(spec, CoreConfig(**(cfg or {})))
                    kind, eval_index = "train", None
                conn.send(("ok", core.info()))
            elif op == "n_eval":
                conn.send(("ok", len(eval_tasks or [])))
            elif op == "budget":
                used = shared["train_decisions"].value if shared is not None else 0
                conn.send(("ok", {"train_decisions_used": used, "train_decisions": (budget or {}).get("train_decisions")}))
            elif op in ("info", "observe", "tools", "end_decision", "summary"):
                out = getattr(core, op)()
                if op == "summary":
                    results.append(out)
                    stamp = 0
                    if shared is not None:
                        with shared["train_decisions"].get_lock():
                            if kind == "train":
                                shared["train_decisions"].value += int(out["decisions"])
                                shared["train_episodes"].value += 1
                            stamp = shared["train_decisions"].value
                    if results_path:
                        _append(results_path, {"kind": kind, "eval_index": eval_index, "stamp": stamp, "t": time.time(),
                                               "summary": out})
                    out = dict(out)
                    if core.spec.split not in allow_splits:      # never reveal held-out task identities
                        out["task"] = f"eval#{eval_index}"
                conn.send(("ok", out))
            elif op == "execute":
                conn.send(("ok", core.execute(a[0], a[1])))
            elif op == "wait":
                conn.send(("ok", core.wait(a)))
            else:
                conn.send(("err", f"unknown op {op!r}"))
        except Exception as e:  # noqa: BLE001 - report, keep serving
            conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"))
    conn.close()


class RemoteCore:
    """Learner-side proxy with the `SessionCore` protocol."""

    def __init__(self, conn):
        self.conn = conn
        self._info: dict = {}

    def _rpc(self, op, a=None):
        self.conn.send((op, a))
        st, out = self.conn.recv()
        if st != "ok":
            raise RuntimeError(f"env server: {out}")
        return out

    def reset(self, spec: TaskSpec | None = None, eval_index: int | None = None) -> dict:
        self._info = self._rpc("reset", {"eval": eval_index} if eval_index is not None else {"spec": spec.to_dict()})
        return self._info

    def n_eval(self) -> int:
        return self._rpc("n_eval")

    def budget(self) -> dict:
        return self._rpc("budget")

    def info(self) -> dict:
        return self._info

    def observe(self) -> dict:
        return self._rpc("observe")

    def tools(self, force: bool = False):
        return self._rpc("tools")

    def execute(self, tool, args) -> dict:
        return self._rpc("execute", (tool, args))

    def wait(self, seconds: int) -> int:
        return self._rpc("wait", int(seconds))

    def end_decision(self) -> dict:
        return self._rpc("end_decision")

    def summary(self) -> dict:
        return self._rpc("summary")

    def close(self):
        try:
            return self._rpc("close")
        except Exception:  # noqa: BLE001
            return []


def spawn_env_server(allow_splits=("train", "curriculum"), eval_tasks=None, cfg=None, eval_cfg=None):
    """Start a trusted env-server process; returns ``(process, RemoteCore)``."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe()
    p = ctx.Process(target=_serve_entry, args=(child, tuple(allow_splits), eval_tasks, cfg, eval_cfg), daemon=True)
    p.start()
    child.close()
    return p, RemoteCore(parent)


def _serve_entry(conn, allow, eval_tasks, cfg, eval_cfg, results_path=None, shared=None, budget=None):
    try:
        import ctypes
        import signal
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)        # die with the hub
    except Exception:  # noqa: BLE001
        pass
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    serve(conn, allow, eval_tasks, cfg, eval_cfg, results_path, shared, budget)
