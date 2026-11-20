"""CAM 2.0 (AGENTRS ``agentrs.cam2``) EnvironmentAdapter for Crucible worlds.

* **Tasks.** ``TaskRef(env="crucible", family="<goal family>:<world family>:<level>[:<prefix>]", seed=s)`` is the goal
  instance ``generate_goal(goal family, world family, s, level)`` in a world built by ``instantiate`` (optionally
  after a documented scripted start-state prefix, `crucible.integrations.PREFIXES`). Example: ``service:tiny:0``.
* **Tools.** Every tool the actor can currently call (``admin.*`` excluded; agent-deployed services appear once
  discovered) becomes a CAM ``ToolSpec`` whose single ``json`` argument is the tool's argument object, with the
  tool's own JSON schema. CAM's pointer network picks the tool; the frozen LLM writes the JSON under CAM's control
  vectors (as for AppWorld). ``effect = "read"`` for ``readOnlyHint`` tools, else ``"sandboxed"`` (simulated world,
  no external side effects). Tools without arguments have no generated argument.
* **Validation.** The generated text is parsed as JSON; an unparseable argument string is an *invalid step*
  (``info["invalid_kind"] = "json_parse"``) that never reaches the world. Every parseable call is sent to Crucible,
  which validates it against the full schema and the world state; a rejection is surfaced with Crucible's error code
  (``info["error_code"]``: ``invalid_action`` = schema, ``not_found``, ``forbidden``, ``insufficient_funds``, ...).
  Schema rejections also count as invalid steps (CAM's "previous action invalid" flag); in-world rejections are
  ordinary consequences. CAM's own shallow validator (`validate_call`) is evaluated too and its agreement with
  Crucible is recorded. All counters are kept separately from task success (`counters`, ``evaluate().details``).
* **Observation.** ``goal`` = the goal's description; ``elements`` (non-targetable) = status line, module summaries,
  the most recent visible events, the last result and the last few actions; ``text`` = last result + recent history.
* **Dynamics.** One tool call per step, then world time advances by ``dt``. Reward = `RewardContract.step_reward`
  (potential shaping on verified progress + verified success bonus). ``terminated`` on verified success or failure
  (horizon elapsed); ``truncated`` at ``max_steps``. `evaluate()` reports the `GoalVerifier` status.
* **Lifetimes.** ``lifetime="fresh"``: every ``reset`` builds the task's world anew (episodic). ``"continue"``: a
  ``reset`` with the same task continues the same world (persistent actor; only the episode counters and the reward
  contract's baseline restart).
* **Snapshots.** `snapshot`/`restore` use the kernel's world snapshot blob plus the adapter's evaluator state.
"""
from __future__ import annotations

import copy
import json
import pickle
from typing import Optional

from agentrs.cam2.envs.base import EnvironmentAdapter, EvalResult, StepResult, TaskRef
from agentrs.cam2.tools import ArgSpec, Element, Observation, ToolCall, ToolSpec, validate_call, validate_json_args

from ..api import render as R
from . import ActorView, agent_tools, compact_json, make_goal_world, tool_signature

SCHEMA_CODES = ("invalid_action", "unknown_tool")
TEXT_SKIP = ("time", "principal", "events", "events_dropped")


def _type_doc(p: dict) -> str:
    if "enum" in p:
        return "|".join(map(str, p["enum"]))
    t = p.get("type", "any")
    if isinstance(t, list):
        t = "|".join(t)
    if t == "array" and isinstance(p.get("items"), dict):
        return f"{_type_doc(p['items'])}[]"
    return str(t)


def crucible_tool_spec(t: dict) -> ToolSpec:
    """Crucible MCP-shaped tool spec -> CAM 2.0 ``ToolSpec`` with one JSON argument (the tool's argument object)."""
    sc = t.get("inputSchema") or {}
    props = sc.get("properties") or {}
    req = list(sc.get("required") or [])
    ann = t.get("annotations") or {}
    effect = "read" if ann.get("readOnlyHint") else "sandboxed"
    desc = (t.get("description") or "").strip().splitlines()
    d = (desc[0] if desc else "")[:260]
    if ann.get("costSeconds"):
        d += f" [takes {ann['costSeconds']}s]"
    args = ()
    if props:
        doc = "; ".join(f"{k} ({_type_doc(props[k])}{', required' if k in req else ''})"
                        f"{': ' + props[k]['description'][:80] if props[k].get('description') else ''}"
                        for k in sorted(props, key=lambda k: (k not in req, k)))
        schema = json.dumps({"type": "object", "properties": props, "required": req}, sort_keys=True)
        args = (ArgSpec("args", "json", doc[:900], json_schema=schema),)
    return ToolSpec(t["name"], d or t["name"], args, effect=effect)


def parse_family(family: str) -> dict:
    parts = family.split(":")
    if len(parts) < 3:
        raise ValueError(f"crucible task family must be '<goal>:<world family>:<level>[:<prefix>]', got {family!r}")
    return {"goal_family": parts[0], "world_family": parts[1], "level": int(parts[2]),
            "prefix": parts[3] if len(parts) > 3 else None}


def parse_args(text: Optional[str]) -> tuple[Optional[dict], Optional[str]]:
    """JSON-object extraction only (same leniency as CAM's validator: code fences, surrounding text)."""
    if text is None:
        return None, "missing JSON arguments"
    return validate_json_args(text, None)


class CrucibleAdapter(EnvironmentAdapter):
    name = "crucible"

    def __init__(self, max_steps: int = 10, dt: int = 3600, max_events: int = 10, history: int = 3,
                 lifetime: str = "fresh", split: str = "dev", prefix_kw: Optional[dict] = None,
                 handle: str = "agent", overrides: Optional[dict] = None):
        if lifetime not in ("fresh", "continue"):
            raise ValueError("lifetime must be 'fresh' or 'continue'")
        self.max_steps, self.dt, self.max_events, self.k_hist = max_steps, int(dt), max_events, history
        self.lifetime, self.split, self.prefix_kw, self.handle = lifetime, split, prefix_kw or {}, handle
        self.overrides = overrides
        self.task: Optional[TaskRef] = None
        self.world = None
        self.counters = self._zero_counters()
        self.episode_counters = self._zero_counters()
        self.call_log: list = []

    @staticmethod
    def _zero_counters() -> dict:
        return {"steps": 0, "calls_sent": 0, "ok": 0, "json_invalid": 0, "schema_rejected": 0, "world_rejected": 0,
                "unknown_tool": 0, "rejections_by_code": {}, "cam_validator_disagreements": 0}

    def describe(self) -> dict:
        return {"name": self.name, "max_steps": self.max_steps, "dt": self.dt, "lifetime": self.lifetime}

    # ------------------------------------------------------------------ episode
    def reset(self, task: TaskRef) -> Observation:
        from ..goals.contracts import RewardContract
        same = self.task is not None and task.key() == self.task.key() and self.world is not None
        if not (self.lifetime == "continue" and same):
            fam = parse_family(task.family)
            self.goal, self.world, self.actor, self.pinfo = make_goal_world(
                fam["goal_family"], fam["world_family"], int(task.seed), fam["level"], split=self.split,
                prefix=fam["prefix"], prefix_kw=self.prefix_kw, handle=self.handle, overrides=self.overrides)
            self.view = ActorView(self.world, self.actor, max_events=max(40, self.max_events))
            self.hist: list = []
            self.last_result = None
        self.task = task
        self.contract = RewardContract(self.goal)
        self.contract.begin(self.world, self.actor)
        self.k = 0
        self._done = False
        self.episode_counters = self._zero_counters()
        self._refresh_tools()
        self._obs = self._make_obs(self.view.observe())
        return self._obs

    def _refresh_tools(self) -> None:
        tools = agent_tools(self.world, self.actor)
        sig = tool_signature(tools)
        if getattr(self, "_tool_sig", None) != sig:
            self._tool_sig = sig
            self.specs = [crucible_tool_spec(t) for t in tools]
            self.tmap = {s.name: s for s in self.specs}

    def _make_obs(self, o: dict) -> Observation:
        els = [Element("st", "status", f"time {R.fmt_time(o.get('time', 0))} | you are {self.actor} | step {self.k} "
                                       f"of {self.max_steps}", targetable=False)]
        for k in sorted(o):
            if k not in TEXT_SKIP:
                els.append(Element(f"s.{k}", "summary", f"{k}: {compact_json(o[k], 220)}", targetable=False))
        evs = (o.get("events") or [])[-self.max_events:]
        for i, e in enumerate(evs):
            els.append(Element(f"e{i}", "event", R.render_event(e, 180), targetable=False))
        if self.last_result is not None:
            els.append(Element("r", "result", "last result: " + R.render_result(self.last_result, 300), targetable=False))
        for i, (tool, args, res) in enumerate(self.hist[-self.k_hist:]):
            els.append(Element(f"h{i}", "history", f"called {tool} {compact_json(args, 160)} -> "
                                                   f"{R.render_result(res, 160) if res else 'not sent'}",
                               targetable=False))
        recent = "\n".join(f"> {t} {compact_json(a, 200)}\n{R.render_result(r, 400) if r else '(not sent)'}"
                           for t, a, r in self.hist[-self.k_hist:])
        return Observation(goal=self.goal.description, elements=els, text=recent, step=self.k, done=self._done)

    def observe(self) -> Observation:
        return self._obs

    def available_tools(self) -> list:
        return list(self.specs)

    # ------------------------------------------------------------------ step
    def _count(self, key: str) -> None:
        for c in (self.counters, self.episode_counters):
            c[key] += 1

    def _count_code(self, code: str) -> None:
        for c in (self.counters, self.episode_counters):
            c["rejections_by_code"][code] = c["rejections_by_code"].get(code, 0) + 1

    def step(self, action: ToolCall) -> StepResult:
        if self._done:
            raise RuntimeError("episode is over; call reset()")
        self._count("steps")
        info: dict = {"invalid": False, "tool": action.tool}
        spec = self.tmap.get(action.tool)
        res = None
        args = None
        if spec is None:
            self._count("unknown_tool")
            info.update(invalid=True, invalid_kind="unknown_tool", error=f"unknown tool {action.tool!r}")
        else:
            if spec.gen_arg is not None:
                args, perr = parse_args(action.text)
            else:
                args, perr = {}, None
            if perr is not None:
                self._count("json_invalid")
                info.update(invalid=True, invalid_kind="json_parse", error=perr)
            else:
                cam_err = validate_call(action if spec.gen_arg is not None else ToolCall(action.tool), self.tmap,
                                        self._obs)
                r = self.world.execute(self.actor, {"tool": action.tool, "args": args})
                res = r.to_dict()
                self._count("calls_sent")
                if r.ok:
                    self._count("ok")
                else:
                    code = (r.error or {}).get("code", "error")
                    self._count_code(code)
                    info["error_code"] = code
                    info["error"] = (r.error or {}).get("message", "")[:300]
                    if code in SCHEMA_CODES:
                        self._count("schema_rejected")
                        info.update(invalid=True, invalid_kind="schema")
                    else:
                        self._count("world_rejected")
                crucible_schema_ok = r.ok or info.get("error_code") not in SCHEMA_CODES
                info["cam_validator_ok"] = cam_err is None
                if (cam_err is None) != crucible_schema_ok:
                    self._count("cam_validator_disagreements")
        self.hist.append((action.tool, args if args is not None else {"unparsed": (action.text or "")[:120]}, res))
        self.call_log.append({"t": self.world.now, "tool": action.tool, "text": action.text, "args": args,
                              "ok": None if res is None else res["ok"], "error_code": info.get("error_code"),
                              "invalid_kind": info.get("invalid_kind")})
        self.last_result = res
        self.world.advance_time(self.world.now + self.dt)
        reward = float(self.contract.step_reward(self.world, self.actor))
        st = self.contract.status(self.world, self.actor)
        self.k += 1
        terminated = bool(st["success"] or st["failed"])
        truncated = (not terminated) and self.k >= self.max_steps
        self._done = terminated or truncated
        self._refresh_tools()
        self._obs = self._make_obs(self.view.observe())
        if self._done:   # "eval" is the key CAM 2.0's runner keeps on the finished Episode (Episode.info)
            ev = {"success": bool(st["success"]), "progress": float(st["progress"]), "failed": bool(st["failed"]),
                  "counters": copy.deepcopy(self.episode_counters), "goal_id": self.goal.id}
            info.update(success=ev["success"], score=ev["progress"], counters=ev["counters"], eval=ev)
        info["progress"] = float(st["progress"])
        return StepResult(self._obs, reward, terminated, truncated, info)

    def evaluate(self) -> EvalResult:
        st = self.contract.status(self.world, self.actor)
        c = self.episode_counters
        n = max(1, c["steps"])
        return EvalResult(success=bool(st["success"]), score=float(st["progress"]),
                          details={"status": st, "counters": copy.deepcopy(c),
                                   "json_invalid_rate": c["json_invalid"] / n,
                                   "schema_rejected_rate": c["schema_rejected"] / n,
                                   "world_rejected_rate": c["world_rejected"] / n, "goal_id": self.goal.id,
                                   "world_digest": self.world.digest(), "steps": self.k})

    # ------------------------------------------------------------------ snapshots
    def snapshot(self):
        return {"task": self.task, "blob": self.world.snapshot_blob(), "world_id": self.world.world_id,
                "contract": pickle.dumps(self.contract), "k": self.k, "hist": copy.deepcopy(self.hist),
                "cursor": self.view.cursor, "last_result": copy.deepcopy(self.last_result),
                "counters": copy.deepcopy(self.episode_counters), "done": self._done}

    def restore(self, snap) -> Observation:
        from ..core.kernel import World
        self.world = World.from_blob(snap["blob"], world_id=snap["world_id"] + "-r")
        self.view.rebind(self.world)
        self.view.cursor = snap["cursor"]
        self.contract = pickle.loads(snap["contract"])
        self.k, self.hist, self.last_result = snap["k"], copy.deepcopy(snap["hist"]), copy.deepcopy(snap["last_result"])
        self.episode_counters, self._done, self.task = copy.deepcopy(snap["counters"]), snap["done"], snap["task"]
        self._tool_sig = None
        self._refresh_tools()
        self._obs = self._make_obs(self.world.observe(self.actor) | {"events": []})
        return self._obs

    def close(self) -> None:
        if self.world is not None:
            self.world.close()


# ============================================================================================== smoke run
def smoke_run(run_dir: str, llm: str = "Qwen/Qwen3-0.6B", family: str = "revenue:tiny:1:deployed_service",
              workers: int = 4, updates: int = 2, episodes_per_update: int = 4, max_steps: int = 8, seed: int = 0,
              cam: Optional[dict] = None, learn: Optional[dict] = None) -> dict:
    """A short real CAM 2.0 run on Crucible: frozen Qwen3 + untrained CAM, collection through `EnvPool` workers,
    ``updates`` PPO updates with `LearnConfig.prefix_replay_correction`. Returns measured numbers (and writes them to
    ``run_dir/summary.json``): parameter change, LLM fingerprint, replay mismatch, invalid / rejected call counts."""
    import time
    from pathlib import Path

    import numpy as np

    from agentrs.cam2.model import param_checksum
    from agentrs.cam2.trainer import RunConfig, Runner, Trainer

    lc = {"episodes_per_update": episodes_per_update, "mb_episodes": episodes_per_update, "epochs": 1,
          "prefix_replay_correction": True, "warmup_steps": 1, "lr": 1e-4, **(learn or {})}
    rc = RunConfig(run_dir=run_dir, env="crucible", llm=llm, workers=workers, seed=seed, max_steps=max_steps,
                   train_families=[family], eval_every=0, ckpt_every=0, cam=dict(cam or {}), learn=lc,
                   save_replay=False, final_eval=False)
    t0 = time.time()
    tr = Trainer(rc)
    pool = tr.ensure_pool()
    runner = Runner(tr.agent, pool, tr.mem, rc.max_steps, tr.lcfg.gamma, gen=tr.gen)
    c0, fp0 = param_checksum(tr.cam), tr.llm.fingerprint()
    snap0 = {k: v.detach().clone() for k, v in tr.cam.state_dict().items()}
    evs: list = []
    ups: list = []
    texts: list = []
    for _ in range(updates):
        eps = runner.run(tr.sampler, tr.lcfg.episodes_per_update, train=True, on_episode=lambda e: evs.append(e))
        st = tr.learner.update(eps, tr.replay)
        ups.append({k: st.get(k) for k in ("replay_mismatch", "replay_mismatch_max", "mismatch_cand_max",
                                           "mismatch_text_mean", "mismatch_text_max", "prefix_deltas",
                                           "params_changed", "update_rel_norm", "opt_steps", "loss", "approx_kl")})
        for e in eps:
            for x in e.tr:
                if x.gen is not None:
                    texts.append(x.gen["text"])
    c1, fp1 = param_checksum(tr.cam), tr.llm.fingerprint()
    changed = sum(int(not torch_equal(snap0[k], v)) for k, v in tr.cam.state_dict().items())
    agg: dict = {"steps": 0, "calls_sent": 0, "ok": 0, "json_invalid": 0, "schema_rejected": 0, "world_rejected": 0,
                 "unknown_tool": 0, "cam_validator_disagreements": 0, "rejections_by_code": {}}
    for e in evs:
        c = (e.info.get("eval") or {}).get("counters") or {}
        for k, v in c.items():
            if k == "rejections_by_code":
                for code, n in v.items():
                    agg[k][code] = agg[k].get(code, 0) + n
            else:
                agg[k] = agg.get(k, 0) + v
    n = max(1, agg["steps"])
    out = {"llm": llm, "llm_params": tr.llm.n_params(), "cam_params": tr.cam.n_params(), "family": family,
           "workers": workers, "episodes": len(evs), "env_steps": runner.env_steps, "updates": ups,
           "success_rate": float(np.mean([e.success for e in evs])) if evs else 0.0,
           "mean_progress": float(np.mean([e.score for e in evs])) if evs else 0.0,
           "mean_return": float(np.mean([e.ret for e in evs])) if evs else 0.0,
           "counters": agg, "json_invalid_rate": agg["json_invalid"] / n, "schema_rejected_rate": agg["schema_rejected"] / n,
           "world_rejected_rate": agg["world_rejected"] / n, "ok_rate": agg["ok"] / n,
           "runner_invalid_rate": runner.invalid / max(1, runner.env_steps),
           "cam_checksum_changed": c1 != c0, "cam_tensors_changed": changed,
           "cam_tensors_total": len(snap0), "llm_fingerprint_unchanged": fp0 == fp1,
           "llm_trainable_params": tr.llm.trainable_params(), "generated_texts": len(texts),
           "sample_texts": texts[:8], "wall_s": time.time() - t0,
           "llm_gen_tokens": tr.llm.stats.get("gen_tokens", 0)}
    tr.close()
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def torch_equal(a, b) -> bool:
    import torch
    return bool(torch.equal(a, b))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CAM 2.0 x Crucible smoke run")
    ap.add_argument("--run-dir", default="runs/cam2/crucible_smoke")
    ap.add_argument("--llm", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--family", default="revenue:tiny:1:deployed_service")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--updates", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=8)
    a = ap.parse_args()
    r = smoke_run(a.run_dir, a.llm, a.family, a.workers, a.updates, a.episodes, a.max_steps)
    print(json.dumps({k: v for k, v in r.items() if k != "sample_texts"}, indent=1, default=float))
