"""G2-final probe: the verifier follows the operator's signed change of the running objective (G2R-15, INT3B round 3)
— can that path be used to put an operator-signed but *superseded* version of the objective back in force, and are
the replay protection, the verifier's own copy (G2R-18) and its restore sound?

    G2F_SRC=<archive>/cam_live/src .venv/bin/python cam_live/docs/verification/repro/g2final/probe_assign_order.py

LabWorld from tests/runtime/rt_helpers; the verifier is built as the run services build it
(``build_verifier(task, ..., pin=True, operator_dir=<run>)``). The operator signs two amendments of the running
objective in this order: X (target 40, duration 1 day) and then Y (target 100000, duration 10 days). The commands are
taken from the control file and handed to the verifier directly — what the learner's process can do over the
verifier socket (G2-runtime's U1 model) — in the order Y, X.
  1  the verifier keeps the newest signed amendment (Y) in force when the older X arrives after it
  1b what the verifier signs on day 2 (X's 1-day deadline passed, Y's 10-day one not): pending, not failed
  2  honest order through the runtime (X then Y): Y is in force (control)
  3  a command already applied is refused when sent again, also by a restarted verifier
  4  the verifier's copy of an assigned file is 0600 in a 0700 directory; a restart restores the assignment from the
     copy after the file was edited
"""
import json
import os
import stat
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2fcommon import PROJECT, Report, scratch  # noqa: E402

from cam_live.contracts import Quantity, RunFiles  # noqa: E402
from cam_live.diagnostics.world import LabConfig  # noqa: E402
from cam_live.runtime.control import OperatorControl  # noqa: E402
from cam_live.runtime.logs import RunReader  # noqa: E402
from cam_live.runtime.verifier import build_verifier  # noqa: E402
from rt_helpers import make_lab  # noqa: E402

R = Report("probe_assign_order")


class WorldMetric:
    name, live = "lab-world-metric", False

    def __init__(self, world):
        self.world = world

    def read(self):
        return [Quantity("lab.adopters.total", float(self.world.total()))]


def edit(task):
    task.success["verifier"] = "metric_threshold"


def setup(tag):
    d = scratch(tag)
    L = make_lab(os.path.join(d, "run"), lab_cfg=LabConfig(seed=3, target=100000, budget=1000.0, horizon_s=30 * 86400.0),
                 task_edit=edit)
    src = WorldMetric(L.world)

    def mk():
        return build_verifier(L.task, os.path.join(d, "verifier"), L.clock, src, log_path=os.path.join(d, "v.jsonl"),
                              pin=True, operator_dir=L.run_dir)
    v = mk()
    L.rt.verifier = v
    L.rt.run(max_iterations=2)
    assert L.rt.status == "running", (L.rt.status, L.rt.status_reason)
    return d, L, v, mk


def amendment(d, L, name, target, days):
    raw = yaml.safe_load(open(os.path.join(PROJECT, "tasks", "lab_staging.yaml")))
    raw.update(objective_id=L.task.objective_id, objective=L.task.objective)
    raw["success"].update(verifier="metric_threshold", metric="lab.adopters.total", target=target)
    raw["limits"]["duration_s"] = days * 86400
    p = os.path.join(d, f"{name}.yaml")
    yaml.safe_dump(raw, open(p, "w"))
    return p


def signed(L, cid):
    return next(c for c in json.load(open(os.path.join(L.run_dir, RunFiles.CONTROL)))["commands"] if c["id"] == cid)


def in_force(v, L):
    t = v._task(L.task)
    return {"target": t.success.get("target"), "duration_days": t.limits.duration_s / 86400}


# ---- 1 the older signed amendment delivered after the newer one -------------------------------------------------
d, L, v, mk = setup("order")
op = OperatorControl(L.run_dir)
px, py = amendment(d, L, "x_old", 40, 1), amendment(d, L, "y_new", 100000, 10)
cx = signed(L, op.assign_objective(px))
cy = signed(L, op.assign_objective(py))
R.note("issued", {"X": {"id": cx["id"], "issued_at": cx["issued_at"]}, "Y": {"id": cy["id"], "issued_at": cy["issued_at"]}})
ry = v.assign_objective(dict(cy))
after_y = in_force(v, L)
try:
    rx = v.assign_objective(dict(cx))
except Exception as e:  # noqa: BLE001
    rx = f"refused: {type(e).__name__}: {str(e)[:160]}"
after_x = in_force(v, L)
R.check("1 the newest signed amendment (Y: target 100000, 10 days) stays in force when the older one (X: target 40, "
        "1 day) reaches the verifier after it", after_x == after_y,
        {"after Y": after_y, "X result": rx, "after X": after_x,
         "note": "assign_objective checks the signature, the file hash and that this command id was not applied "
                 "before; it does not compare issued_at with the amendment in force"})
# what the verifier then signs on day 2 (X's 1-day deadline has passed, Y's 10-day deadline has not)
L.clock.advance(max(0.0, L.rt.started_at + 2 * 86400.0 - L.clock.now()))
vo = v.verify(L.task)
R.check("1b on day 2 the verifier's signed outcome follows the newest amendment (Y: pending until day 10), not the "
        "superseded X (failed after 1 day)", str(vo.status).lower().endswith("pending"),
        {"status": str(vo.status), "value": vo.value, "target": vo.target,
         "days_after_start": round((L.clock.now() - L.rt.started_at) / 86400, 2)})

# ---- 2 honest order through the runtime --------------------------------------------------------------------------
d2, L2, v2, mk2 = setup("honest")
op2 = OperatorControl(L2.run_dir)
qx, qy = amendment(d2, L2, "x_old", 40, 1), amendment(d2, L2, "y_new", 100000, 10)
k1, k2 = op2.assign_objective(qx), op2.assign_objective(qy)
L2.rt.step()
res = RunReader(L2.run_dir).state()["control_results"]
R.check("2 honest order through the runtime (X then Y): Y is in force in the verifier and the runtime",
        in_force(v2, L2) == {"target": 100000, "duration_days": 10.0} and L2.rt.task.success.get("target") == 100000,
        {"verifier": in_force(v2, L2), "runtime_target": L2.rt.task.success.get("target"),
         "results": {k: res.get(k, {}).get("detail") for k in (k1, k2)}})

# ---- 3 replay of an applied command, also after a restart ----------------------------------------------------------
def tries(vv, cmd):
    try:
        vv.assign_objective(dict(cmd))
        return "accepted"
    except Exception as e:  # noqa: BLE001
        return f"refused: {type(e).__name__}: {str(e)[:120]}"


sy = signed(L2, k2)
r_same = tries(v2, sy)
v2b = mk2()
r_restart = tries(v2b, sy)
R.check("3 an applied signed assignment is refused when sent again (same verifier, and a restarted one)",
        r_same.startswith("refused") and r_restart.startswith("refused"), {"same": r_same, "restarted": r_restart})

# ---- 4 the verifier's copy and its restore ------------------------------------------------------------------------
copies = os.path.join(d2, "verifier", "assigned")
modes = {f: oct(stat.S_IMODE(os.stat(os.path.join(copies, f)).st_mode)) for f in os.listdir(copies)} \
    if os.path.isdir(copies) else {}
R.check("4a the verifier's copies of assigned files are 0600 in a 0700 directory",
        bool(modes) and all(m == "0o600" for m in modes.values())
        and stat.S_IMODE(os.stat(copies).st_mode) == 0o700, {"dir": oct(stat.S_IMODE(os.stat(copies).st_mode))
                                                               if os.path.isdir(copies) else None, "files": modes})
with open(qy, "a") as f:
    f.write("# edited after the assignment\n")
v2c = mk2()
R.check("4b after the file was edited, a restarted verifier restores the assignment from its own copy",
        in_force(v2c, L2) == {"target": 100000, "duration_days": 10.0}, in_force(v2c, L2))
sys.exit(R.done())
