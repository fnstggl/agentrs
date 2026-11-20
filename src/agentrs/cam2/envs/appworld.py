"""AppWorld adapter (https://appworld.dev): nine simulated apps (Spotify, Amazon, Gmail, Venmo, file system, ...)
with persistent databases, 457 documented APIs and AppWorld's official state-based evaluator.

* Tools: every API of the task's allowed apps, published from AppWorld's own API documentation as a tool with one
  ``json`` argument validated against a JSON schema derived from the documented parameters. CAM chooses the API;
  the frozen LLM, under CAM's control vectors, writes the JSON arguments.
* Execution: a worker process in AppWorld's own virtualenv (``appworld_worker.py``) executes
  ``apis.<app>.<api>(**args)`` inside AppWorld's sandboxed interpreter.
* Episode end: the agent calls ``supervisor.complete_task`` (AppWorld's own completion API) or hits ``max_steps``.
* Reward / success: AppWorld's evaluator (``world.evaluate()``): success = all task tests pass. The training reward
  is ``1`` on success, otherwise ``0.5 * max(0, passes - passes_at_start) / num_tests`` (credit only for tests the
  agent's own actions made pass; tests that already pass on the untouched state, e.g. "no collateral damage", give
  nothing). Computed only at episode end.
* Snapshot/restore: AppWorld's native ``save_state`` / ``load_state``.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from agentrs.cam2.envs.base import DependencyUnavailable, EnvironmentAdapter, EvalResult, StepResult, TaskRef
from agentrs.cam2.tools import ArgSpec, Element, Observation, ToolCall, ToolSpec, validate_call, validate_json_args

APPWORLD_PY = os.environ.get("CAM2_APPWORLD_PYTHON", "/workspace/ext/venv-appworld/bin/python")
APPWORLD_ROOT = os.environ.get("APPWORLD_ROOT", "/workspace/ext/appworld_root")
SETUP = """AppWorld is not available: {why}
Setup (AppWorld gets its own virtualenv; it pins its own dependencies):
  uv venv --python 3.12 /workspace/ext/venv-appworld
  /workspace/ext/venv-appworld/bin/python -m pip install appworld   # or: uv pip install --python ... appworld
  mkdir -p /workspace/ext/appworld_root && cd /workspace/ext/appworld_root
  /workspace/ext/venv-appworld/bin/appworld install
  /workspace/ext/venv-appworld/bin/appworld download data
  export CAM2_APPWORLD_PYTHON=/workspace/ext/venv-appworld/bin/python APPWORLD_ROOT=/workspace/ext/appworld_root
"""
WORKER = str(Path(__file__).with_name("appworld_worker.py"))
SPLITS = {"train": "train", "val": "dev", "test": "test_normal"}


def _jtype(t: str):
    t = (t or "").lower()
    if t.startswith("list"):
        return "array"
    if t in ("string", "integer", "number", "boolean", "object"):
        return t
    if t in ("dict",):
        return "object"
    if t in ("float",):
        return "number"
    return None


def api_tool(d: dict) -> ToolSpec:
    props, req, pdoc = {}, [], []
    for p in d.get("parameters", []):
        jt = _jtype(p.get("type", ""))
        props[p["name"]] = {"type": jt} if jt else {}
        if p.get("required"):
            req.append(p["name"])
        pdoc.append(f"{p['name']} ({p.get('type')}{', required' if p.get('required') else ''}): "
                    f"{(p.get('description') or '')[:100]}")
    schema = json.dumps({"type": "object", "properties": props, "required": req}, sort_keys=True)
    args = (ArgSpec("arguments", "json", "; ".join(pdoc)[:900], json_schema=schema),) if props else ()
    return ToolSpec(f"{d['app']}.{d['api']}", (d.get("description") or "")[:300], args, effect="sandboxed")


def flatten_output(text: str, max_lines: int = 40) -> list:
    try:
        v = json.loads(text)
    except Exception:  # noqa: BLE001
        return [ln[:160] for ln in text.strip().splitlines()[:max_lines] if ln.strip()]
    lines: list = []

    def walk(x, path):
        if len(lines) >= max_lines:
            return
        if isinstance(x, dict):
            for k, y in x.items():
                walk(y, f"{path}.{k}" if path else str(k))
        elif isinstance(x, list):
            if not x:
                lines.append(f"{path}: []")
            for i, y in enumerate(x):
                walk(y, f"{path}[{i}]")
        else:
            lines.append(f"{path}: {str(x)[:120]}")

    walk(v, "")
    return lines


class _Proc:
    def __init__(self, experiment: str):
        if not Path(APPWORLD_PY).exists():
            raise DependencyUnavailable(SETUP.format(why=f"{APPWORLD_PY} not found"))
        if not Path(APPWORLD_ROOT, "data", "tasks").exists():
            raise DependencyUnavailable(SETUP.format(why=f"AppWorld data not found under {APPWORLD_ROOT}"))
        env = dict(os.environ, APPWORLD_ROOT=APPWORLD_ROOT, CAM2_APPWORLD_EXPERIMENT=experiment)
        self.p = subprocess.Popen([APPWORLD_PY, "-I", "-u", WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, cwd=APPWORLD_ROOT, env=env, text=True, bufsize=1)

    def call(self, **msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("AppWorld worker exited")
        r = json.loads(line)
        if not r["ok"]:
            raise RuntimeError(r["value"])
        return r["value"]

    def close(self):
        try:
            self.call(cmd="close")
        except Exception:  # noqa: BLE001
            pass
        self.p.wait(timeout=30)


class AppWorldAdapter(EnvironmentAdapter):
    name = "appworld"

    def __init__(self, max_steps: int = 25, experiment: Optional[str] = None, history: int = 3):
        self.proc = _Proc(experiment or f"cam2_{os.getpid()}")
        self.max_steps, self.k_hist = max_steps, history
        self.task: Optional[TaskRef] = None
        self.tools: list = []
        self.tmap: dict = {}
        self._docs_cache: dict = {}
        self._obs: Optional[Observation] = None
        self.calls: list = []
        self._final: Optional[dict] = None
        self._base_passes = 0
        self._done = False

    def describe(self) -> dict:
        return {"name": self.name, "python": APPWORLD_PY, "root": APPWORLD_ROOT, "max_steps": self.max_steps}

    def task_ids(self, split: str) -> list:
        return self.proc.call(cmd="tasks", split=SPLITS.get(split, split))

    def reset(self, task: TaskRef) -> Observation:
        info = self.proc.call(cmd="reset", task_id=task.instance, seed=task.seed)
        key = tuple(sorted(info["allowed_apps"]))
        if key not in self._docs_cache:
            self._docs_cache[key] = [api_tool(d) for d in self.proc.call(cmd="api_docs")]
        self.tools = self._docs_cache[key]
        self.tmap = {t.name: t for t in self.tools}
        self.task = task
        self.calls = []
        self._final = None
        self._done = False
        self._base_passes = self.proc.call(cmd="evaluate")["passes"]
        s = info["supervisor"]
        self.goal = (f"{info['instruction']} (You act for {s['first_name']} {s['last_name']}, email {s['email']}, "
                     f"phone {s['phone_number']}. Date: {info['datetime']}.)")
        self._obs = self._make_obs()
        return self._obs

    def _make_obs(self, note: str = "") -> Observation:
        els = []
        for i, (name, args, out) in enumerate(self.calls[-self.k_hist:]):
            els.append(Element(f"h{i}", "history", f"called {name} {json.dumps(args)[:200]}", targetable=False))
        if self.calls:
            for j, ln in enumerate(flatten_output(self.calls[-1][2])):
                els.append(Element(f"o{j}", "output", ln, targetable=False))
        recent = "\n".join(f"> {n}({json.dumps(a)[:300]})\n{o[:700]}" for n, a, o in self.calls[-self.k_hist:])
        return Observation(goal=self.goal, elements=els, text=(note + "\n" + recent).strip(), step=len(self.calls))

    def observe(self) -> Observation:
        return self._obs

    def available_tools(self) -> list:
        return list(self.tools)

    def step(self, action: ToolCall) -> StepResult:
        assert not self._done
        err = validate_call(action, self.tmap, self._obs)
        info = {"invalid": err is not None}
        completed = False
        if err is None:
            spec = self.tmap[action.tool]
            args = validate_json_args(action.text, spec.gen_arg.json_schema)[0] if spec.gen_arg else {}
            app, api = action.tool.split(".", 1)
            r = self.proc.call(cmd="call", app=app, api=api, args=args)
            self.calls.append((action.tool, args, r["output"]))
            completed = bool(r["completed"])
            self._obs = self._make_obs()
        else:
            self.calls.append((action.tool, {"invalid": True}, "invalid call: " + err))
            self._obs = self._make_obs()
        n = len(self.calls)
        trunc = (not completed) and n >= self.max_steps
        reward = 0.0
        if completed or trunc:
            self._done = True
            self._final = self.proc.call(cmd="evaluate")
            f = self._final
            gain = max(0, f["passes"] - self._base_passes) / max(1, f["num_tests"])
            reward = 1.0 if f["success"] else 0.5 * gain
            info["eval"] = dict(f, base_passes=self._base_passes)
            info["success"] = bool(f["success"])
            info["score"] = f["passes"] / max(1, f["num_tests"])
        # A truncated AppWorld episode is still scored by the official evaluator on the final state: the task is over
        return StepResult(self._obs, reward, completed, trunc, info)

    def evaluate(self) -> EvalResult:
        f = self._final or self.proc.call(cmd="evaluate")
        return EvalResult(success=bool(f["success"]), score=f["passes"] / max(1, f["num_tests"]),
                          details=dict(f, base_passes=self._base_passes, steps=len(self.calls)))

    def snapshot(self):
        sid = f"snap{len(self.calls)}"
        self.proc.call(cmd="save", state_id=sid)
        return {"state_id": sid, "calls": list(self.calls), "task": self.task}

    def restore(self, snap) -> Observation:
        self.proc.call(cmd="load", state_id=snap["state_id"])
        self.calls = list(snap["calls"])
        self._done = False
        self._obs = self._make_obs()
        return self._obs

    def close(self) -> None:
        self.proc.close()
