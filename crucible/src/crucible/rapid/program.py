"""`program` actions: an agent-written controller run in the sandbox (MASTER_SPEC Part 8.3 item 6, Part 8.11).

    {"kind": "program", "source": "def act(observation, caller):\\n    ...\\n    return {...}", "ticks": 20}

The source is executed by `crucible.sandbox` (separate unprivileged process, Landlock + seccomp, rlimits, wall-clock
timeouts, no network, numpy only). Each tick the controller receives the participant's current Observation and
returns one action, which is validated against the kinds the world allows inside programs and carried out like any
other action; then the world advances. The controller runs for `ticks` ticks or until the episode ends, it raises,
times out, or returns an invalid action (the tick passes, the program stops, and the error is reported).

Programs are the participant's own code: their outputs are recorded in the result (`actions`) so a run can be
audited; the world itself never depends on the program's internals.
"""
from __future__ import annotations

import json

from ..sandbox.runner import SandboxConfig, Worker


def _cfg() -> SandboxConfig:
    return SandboxConfig(job_wall_s=2.0, mem_mb=768, cpu_s=120)


def run_controller(env, agent: str, action: dict) -> dict:
    space = env.action_space(agent)
    errs = space.validate(action)
    if errs:
        return _refuse(env, agent, "; ".join(errs[:3]))
    allowed = space.limits.get("program_kinds", ["levers"])
    ticks = int(action["ticks"])
    w = Worker({"controller.py": action["source"]}, _cfg(), "rapid-controller")
    log, err, last = [], None, None
    try:
        w.start()
        for _ in range(ticks):
            if env.done:
                break
            obs = env.observe(agent)
            res = w.run({"kind": "call", "target": "controller.py:act", "args": obs, "caller": {"agent": agent}})
            if not res.get("ok"):
                err = f"controller failed: {(res.get('error') or {}).get('type')}: " \
                      f"{(res.get('error') or {}).get('message', '')[:300]}"
                break
            act = res.get("result")
            if act is not None and (not isinstance(act, dict) or act.get("kind") not in allowed):
                err = f"controller returned {json.dumps(act)[:200]}; allowed kinds inside programs: {allowed}"
                break
            if act is not None and space.validate(act):
                err = f"controller returned an invalid action: {space.validate(act)[:2]}"
                break
            last = env._tick({agent: act} if act is not None else {})
            log.append(act)
    finally:
        w.close()
    if last is None:
        return _refuse(env, agent, err or "the controller did not run")
    res = dict(last["result"])
    res["program"] = {"ticks_run": len(log), "actions": log[-20:], "error": err}
    if err:
        res["ok"], res["error"] = False, err
    return dict(last, result=res)


def _refuse(env, agent: str, why: str) -> dict:
    r = env._tick({})
    res = {"ok": False, "error": why, "results": None}
    return dict(r, result=res)
