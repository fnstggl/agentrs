"""G2-runtime probe: signed operator commands (runtime.control) attacked by whoever can write the control file.

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_control_signed.py

A LabWorld run (tests/runtime/rt_helpers.make_lab, single-uid in-process mode) is paused by the operator; the attacker
then appends entries to control.json directly (as code that can write the run directory but does not hold
operator/control.key would). Checks: replay of an applied signed command (same id), the same signature on altered
arguments, a command signed with another run's key, a replay after a restart of the runtime, and that brakes still
apply unsigned.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2common import Report, scratch  # noqa: E402

from cam_live.contracts import RunFiles  # noqa: E402
from cam_live.runtime.control import ControlChannel, OperatorControl  # noqa: E402
from cam_live.runtime.logs import atomic_write_json, read_json  # noqa: E402
from rt_helpers import make_lab  # noqa: E402

R = Report("probe_control_signed")
base = scratch("ctl")
run = os.path.join(base, "run")
L = make_lab(run)
L.rt.run(max_iterations=3)
op = OperatorControl(run)
ctl_path = os.path.join(run, RunFiles.CONTROL)


def cmds():
    return (read_json(ctl_path, {}) or {}).get("commands", [])


def append_raw(entry):
    doc = read_json(ctl_path, {}) or {}
    doc.setdefault("commands", []).append(entry)
    atomic_write_json(ctl_path, doc)


def results():
    return L.rt._control_results


def step():
    L.rt.step()
    return L.rt.status


op.pause("operator pause")
step()
R.check("0 an operator pause is applied", L.rt.status == "paused", L.rt.status)
cid = op.resume(wake=False)
step()
R.check("1 the operator's signed resume is applied", L.rt.status == "running" and results()[cid]["ok"], results().get(cid))
signed_resume = copy.deepcopy(next(c for c in cmds() if c["id"] == cid))

# 2 replay of the applied signed command (same id, same signature)
ControlChannel(run).send("pause", by="attacker", wake=False)
step()
append_raw(copy.deepcopy(signed_resume))
step()
R.check("2 a replayed signed resume (same id) does not lift a later pause", L.rt.status == "paused",
        {"status": L.rt.status, "result": results().get(cid)})

# 3 the signature of one command on altered arguments / another command
forged = copy.deepcopy(signed_resume)
forged["id"] = "cmd_forged_1"
append_raw(forged)
step()
R.check("3a a signed resume with a new id (signature copied) is refused", L.rt.status == "paused",
        results().get("cmd_forged_1"))
L.rt._apply_command("resume", {}, "probe")       # set back to running through the runtime itself (test harness)
apr = copy.deepcopy(signed_resume)
apr.update(id="cmd_forged_2", cmd="export_checkpoint", args={"dest": os.path.join(base, "stolen")})
append_raw(apr)
step()
R.check("3b a signature copied onto a different command (export_checkpoint to an attacker path) is refused",
        not (results().get("cmd_forged_2") or {}).get("ok") and not os.path.exists(os.path.join(base, "stolen")),
        results().get("cmd_forged_2"))

# 4 a command signed with another run's operator key
other = os.path.join(base, "other_run")
os.makedirs(other)
foreign = OperatorControl(other)
fid = foreign.export_checkpoint(os.path.join(base, "stolen2"))
fentry = next(c for c in (read_json(os.path.join(other, RunFiles.CONTROL), {}) or {})["commands"] if c["id"] == fid)
append_raw(fentry)
step()
R.check("4 a command signed with another run's key is refused", not (results().get(fid) or {}).get("ok")
        and not os.path.exists(os.path.join(base, "stolen2")), results().get(fid))

# 5 replay after a restart of the runtime (seen ids must survive the restart)
op.pause("pause before restart")
step()
L.rt.close()
L2 = make_lab(run)
L2.agent = L.agent
L.rt = L2.rt
L.rt.start()
append_raw(copy.deepcopy(signed_resume))
step()
R.check("5 after a restart, a replayed signed resume does not lift the pause", L.rt.status == "paused",
        {"status": L.rt.status})

# 6 brakes apply unsigned
L.rt._apply_command("resume", {}, "probe")
ControlChannel(run).send("stop", by="anyone", wake=False)
step()
R.check("6 an unsigned stop is applied (brakes need no signature) and recorded as unauthenticated",
        L.rt.status == "stopped", {"status": L.rt.status})
ev = [json.loads(x) for x in open(os.path.join(L.rt.log.log_dir, RunFiles.EVENTS)) if '"operator_command"' in x]
R.check("7 every refused attacker command is in the event log as unauthenticated / refused",
        all(not e["data"]["ok"] for e in ev if e["data"]["id"] in ("cmd_forged_1", "cmd_forged_2", fid)),
        [(e["data"]["id"], e["data"]["authenticated"], e["data"]["ok"]) for e in ev][-8:])
sys.exit(R.done())
