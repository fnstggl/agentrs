"""G-rt probe 03 — operator control in real time: pause / resume / stop / emergency stop are honoured within a
bounded time, also while the runtime sleeps for a delayed outcome and while an action is executing.

    .venv/bin/python cam_live/docs/verification/repro/rt/p03_control.py          (about 40 s of real time)
"""
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import Report, StandInAgent, make_slow_run, read_log  # noqa: E402

from cam_live.contracts import RunFiles, WallClock  # noqa: E402
from cam_live.diagnostics.world import AllowAllPolicy  # noqa: E402
from cam_live.runtime.control import ControlChannel, OperatorControl  # noqa: E402
from cam_live.runtime.loop import LiveRuntime, RuntimeConfig  # noqa: E402

R = Report("p03_control")
POLL = 1.0            # RuntimeConfig.control_poll_s default: the documented bound for noticing a command while asleep


def state(run_dir):
    try:
        return json.load(open(os.path.join(run_dir, RunFiles.STATE)))
    except (OSError, ValueError):
        return {}


def wait_for(pred, timeout):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return time.monotonic() - t0
        time.sleep(0.02)
    return None


def start(L):
    th = threading.Thread(target=lambda: L.rt.run(block=True), daemon=True)
    th.start()
    return th


# ---- A. asleep, waiting for a delayed outcome one hour away ------------------------------------------------
L = make_slow_run("p03a", WallClock(), env_kw={"outcome_horizon": 3600.0}, script=("slow.act", "wait"), wait_seconds=1800.0,
                  rt_kw={"control_poll_s": POLL})
th = start(L)
dt = wait_for(lambda: (state(L.run_dir).get("activity") or {}).get("phase") == "sleeping", 20)
st = state(L.run_dir)
R.check("A0 the runtime goes to sleep with an outcome pending and a wait in progress",
        dt is not None and len(st["live"]["pending"]["items"]) >= 1, {"wake": st.get("wake"), "waiting": st.get("waiting")})
calls0 = dict(L.env.calls)
time.sleep(2.5)
R.check("A1 while asleep it calls nothing in the environment (no polling, no activity)", L.env.calls == calls0,
        {"before": calls0, "after": L.env.calls})
op = OperatorControl(L.run_dir)
t0 = time.monotonic()
cid = op.pause("inspection")
dt = wait_for(lambda: state(L.run_dir).get("status") == "paused", 10)
R.check(f"A2 pause while asleep is applied within control_poll_s + 0.5 s ({POLL + 0.5:.1f} s)", dt is not None and dt <= POLL + 0.5,
        {"seconds": dt})
st = state(L.run_dir)
R.check("A3 pause checkpointed the learner", bool((st.get("committed") or {}).get("ckpt")) and
        os.path.exists(os.path.join(st["committed"]["ckpt"]["path"], "agent.json")), (st.get("committed") or {}).get("ckpt"))
n_exec = len(L.env.executed)
time.sleep(2.0)
R.check("A4 paused: nothing executes, nothing is observed", len(L.env.executed) == n_exec and L.env.calls == calls0)
cid = op.resume()
dt = wait_for(lambda: state(L.run_dir).get("status") == "running", 10)
R.check(f"A5 resume is applied within {POLL + 0.5:.1f} s", dt is not None and dt <= POLL + 0.5, {"seconds": dt})
wait_for(lambda: (state(L.run_dir).get("activity") or {}).get("phase") == "sleeping", 10)
cid = op.stop("end of probe")
dt = wait_for(lambda: state(L.run_dir).get("status") == "stopped" and not th.is_alive(), 10)
R.check(f"A6 stop while asleep: status stopped and run() returned within {POLL + 0.5:.1f} s", dt is not None and dt <= POLL + 0.5,
        {"seconds": dt})
# a stopped run cannot be resumed by command, nor by restarting the process
ControlChannel(L.run_dir).send("resume", by="learner")
rt2 = LiveRuntime(StandInAgent(L.clock, script=("slow.act",), durable=False), L.env, L.policy, L.verifier, L.task,
                  L.run_dir, L.clock, RuntimeConfig(fsync=False))
n_exec = len(L.env.executed)
s2 = rt2.run(max_iterations=5, block=False)
R.check("A7 after stop: a resume command and a process restart leave the run stopped; nothing executes",
        s2 == "stopped" and len(L.env.executed) == n_exec, {"status": s2})
rt2.close()

# ---- B. stop requested while an action is executing ---------------------------------------------------------
L = make_slow_run("p03b", WallClock(), env_kw={"exec_seconds": 3.0, "outcome_horizon": 3600.0}, script=("slow.act",),
                  rt_kw={"control_poll_s": POLL})
th = start(L)
wait_for(lambda: len(L.env.executed) == 1, 20)
time.sleep(0.5)                                   # the first action is now in flight
t0 = time.monotonic()
OperatorControl(L.run_dir).stop("stop during an action")
dt = wait_for(lambda: state(L.run_dir).get("status") == "stopped", 15)
ev = read_log(L.run_dir, RunFiles.EVENTS)
res = [e for e in ev if e["type"] == "action_result" and e["data"]["kind"] == "execute"]
R.check("B1 stop during an action: the in-flight action finishes and is recorded, no further action starts",
        dt is not None and len(L.env.executed) == 1 and len(res) == 1 and res[0]["data"]["status"] == "ok",
        {"seconds_until_stopped": dt, "executions": len(L.env.executed), "results": len(res)})
R.check("B2 ... and the stop takes effect within the remaining action time + 1 s", dt is not None and dt <= 2.5 + 1.0, {"seconds": dt})
R.check("B3 the learner received the result of the action that was in flight", [r[1] for r in L.agent.results] == ["ok"],
        L.agent.results)

# ---- C. emergency stop raised at the policy only (no control-file command) -----------------------------------
pol = AllowAllPolicy()
L = make_slow_run("p03c", WallClock(), env_kw={"outcome_horizon": 3600.0}, script=("slow.act", "wait"), wait_seconds=1800.0,
                  rt_kw={"control_poll_s": POLL}, policy=pol)
th = start(L)
wait_for(lambda: (state(L.run_dir).get("activity") or {}).get("phase") == "sleeping", 20)
pol.emergency_stop("operator emergency stop", "operator")
dt = wait_for(lambda: state(L.run_dir).get("status") == "stopped", 20)
R.check(f"C1 an emergency stop engaged at the authorization service is noticed while asleep within 5 x control_poll_s + 1 s",
        dt is not None and dt <= 5 * POLL + 1.0, {"seconds": dt})
# attempts to lift it from the learner's side: resume command, state file edit + restart
ControlChannel(L.run_dir).send("resume", by="learner")
time.sleep(1.5)
st = state(L.run_dir)
st["status"], st["status_reason"] = "running", ""
json.dump(st, open(os.path.join(L.run_dir, RunFiles.STATE), "w"))
rt2 = LiveRuntime(StandInAgent(L.clock, script=("slow.act",), durable=False), L.env, pol, L.verifier, L.task, L.run_dir,
                  L.clock, RuntimeConfig(fsync=False))
n_exec = len(L.env.executed)
rt2.start()
rt2.step()
R.check("C2 with the stop still engaged, a resume command + an edited state file + a restart do not get an action executed",
        rt2.status == "stopped" and len(L.env.executed) == n_exec, {"status": rt2.status})
rt2.close()

# ---- D. emergency stop through OperatorControl during an action ----------------------------------------------
pol = AllowAllPolicy()
L = make_slow_run("p03d", WallClock(), env_kw={"exec_seconds": 2.0, "outcome_horizon": 3600.0}, script=("slow.act",),
                  rt_kw={"control_poll_s": POLL}, policy=pol)
th = start(L)
wait_for(lambda: len(L.env.executed) == 1, 20)
OperatorControl(L.run_dir, policy=pol).stop("emergency", emergency=True)
dt = wait_for(lambda: state(L.run_dir).get("status") == "stopped", 15)
R.check("D1 emergency stop during an action: policy engaged at once, run stopped after the in-flight action, no second action",
        pol.stopped() and dt is not None and len(L.env.executed) == 1, {"seconds": dt, "executions": len(L.env.executed)})

# ---- E. who can lift a pause? ---------------------------------------------------------------------------------
L = make_slow_run("p03e", WallClock(), env_kw={"outcome_horizon": 3600.0}, script=("slow.act", "wait"), wait_seconds=1800.0,
                  rt_kw={"control_poll_s": 0.2})
th = start(L)
wait_for(lambda: (state(L.run_dir).get("activity") or {}).get("phase") == "sleeping", 20)
OperatorControl(L.run_dir).pause("operator pause")
wait_for(lambda: state(L.run_dir).get("status") == "paused", 10)
# the learner's own process appends a resume command to the control file it can write
ControlChannel(L.run_dir).send("resume", by="operator")
dt = wait_for(lambda: state(L.run_dir).get("status") == "running", 5)
cmds = [e["data"] for e in read_log(L.run_dir, RunFiles.EVENTS) if e["type"] == "operator_command"]
R.check("E1 a pause cannot be lifted by a command written from the learner's process (commands are authenticated)",
        dt is None, {"resumed_after_s": dt, "logged_as": cmds[-1] if cmds else None})
OperatorControl(L.run_dir).stop("end")
wait_for(lambda: not th.is_alive(), 10)
sys.exit(R.finish())
