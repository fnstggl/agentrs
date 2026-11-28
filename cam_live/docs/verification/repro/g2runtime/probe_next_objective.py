"""G2-runtime probe: "assign a subsequent objective" (spec 15.3) / "enter a separately authorized next phase after
verified success" (13.4, LX-19) with the verifier as the run services build it.

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_next_objective.py

runtime.loop_service.build_inproc and runtime.verifier_service.serve both build the verifier with
``build_verifier(task, ..., pin=True)``: it judges the operator's first task only. LabWorld world and learner from
tests/runtime/rt_helpers; the verifier is the runtime's own MetricThresholdVerifier, pinned like that, reading the
world's adopter count. Objective A completes; the operator assigns objective B (a valid task file, signed command).
Check: B can be verified (its progress is measured and it can complete) — or the runtime at least refuses the
assignment up front instead of accepting it.
"""
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2common import PROJECT, Report, scratch  # noqa: E402

from cam_live.contracts import Quantity, RunFiles  # noqa: E402
from cam_live.diagnostics.world import LabConfig  # noqa: E402
from cam_live.runtime.control import OperatorControl  # noqa: E402
from cam_live.runtime.logs import RunReader  # noqa: E402
from cam_live.runtime.verifier import build_verifier  # noqa: E402
from rt_helpers import make_lab  # noqa: E402

R = Report("probe_next_objective")
base = scratch("next")


class WorldMetric:
    name, live = "lab-world-metric", False

    def __init__(self):
        self.world = None

    def read(self):
        return [Quantity("lab.adopters.total", float(self.world.total()))]


src = WorldMetric()
holder = {}


def edit(task):
    task.success["verifier"] = "metric_threshold"


L = make_lab(os.path.join(base, "run"), lab_cfg=LabConfig(seed=3, target=3, budget=1000.0), task_edit=edit)
src.world = L.world
def pinned(task, clock, d, run_dir):
    """build_verifier as verifier_service.serve / loop_service.build_inproc call it (operator_dir = the run directory
    at the final freeze; the argument does not exist at a5ab772)."""
    try:
        return build_verifier(task, os.path.join(d, "verifier"), clock, src, log_path=os.path.join(d, "v.jsonl"),
                              pin=True, operator_dir=run_dir)
    except TypeError:
        return build_verifier(task, os.path.join(d, "verifier"), clock, src, log_path=os.path.join(d, "v.jsonl"),
                              pin=True)


v = pinned(L.task, L.clock, base, L.run_dir)
L.rt.verifier = v
status_a = L.rt.run(max_iterations=2000)
R.check("0 objective A completes on the pinned verifier's signed outcome", status_a == "completed", status_a)

raw = yaml.safe_load(open(os.path.join(PROJECT, "tasks", "lab_staging.yaml")))
target_b = int(L.world.total()) + 2
raw.update(objective_id="lab-second", objective=f"Reach {target_b} adopters in the lab world before time runs out.")
raw["success"].update(verifier="metric_threshold", metric="lab.adopters.total", target=target_b)
path_b = os.path.join(base, "b.yaml")
yaml.safe_dump(raw, open(path_b, "w"))
cid = OperatorControl(L.run_dir).assign_objective(path_b)
L.rt.step()
res = RunReader(L.run_dir).state()["control_results"][cid]
R.note("assign_objective result", res)
for _ in range(400):
    if L.rt.status != "running":
        break
    L.rt.step()
ev = RunReader(L.run_dir).records(RunFiles.EVENTS)
errs = [e["data"]["error"] for e in ev if e["type"] == "error" and e["data"].get("what") == "verify"]
verified_b = [e for e in ev if e["type"] == "verified_outcome"
              and e["data"]["outcome"]["objective_id"] == "lab-second"]
R.check("1 the assigned objective B is verified by the run's verifier (at least one signed outcome for B)",
        bool(verified_b) and res["ok"],
        {"assign_ok": res["ok"], "status": L.rt.status, "status_reason": L.rt.status_reason[:200],
         "verify_errors": len(errs), "first_error": errs[0][:200] if errs else "",
         "world_total": L.world.total(), "target_b": target_b, "decisions_for_b": L.rt.counters["decisions"]})
# ---- round 2: can anything but the operator's signed command make the verifier follow an objective? -------------
import copy  # noqa: E402
from cam_live.runtime.control import OperatorKey  # noqa: E402
easy = dict(raw, objective_id="lab-easy")
easy["success"] = dict(raw["success"], target=1)
path_easy = os.path.join(base, "easy.yaml")
yaml.safe_dump(easy, open(path_easy, "w"))
from cam_live.runtime.control import file_sha256  # noqa: E402
unsigned = {"id": "cmd_u1", "cmd": "assign_objective", "by": "operator", "issued_at": 0.0,
            "args": {"path": path_easy, "sha256": file_sha256(path_easy)}}


def tries(label, cmd):
    try:
        out = v.assign_objective(cmd)
        R.check(f"U {label}: refused by the verifier", False, out)
    except Exception as e:  # noqa: BLE001
        R.check(f"U {label}: refused by the verifier", True, f"{type(e).__name__}: {str(e)[:160]}")


if hasattr(v, "assign_objective"):
    tries("an unsigned assign_objective command (what the learner's process could send over the socket)", unsigned)
    signed_b = next(c for c in (json.load(open(os.path.join(L.run_dir, RunFiles.CONTROL)))["commands"]) if c["id"] == cid)
    alt = copy.deepcopy(signed_b)
    alt["args"] = {"path": path_easy, "sha256": file_sha256(path_easy)}
    alt["id"] = "cmd_u2"
    tries("the operator's signature on altered arguments (another, easier task file)", alt)
    other = os.path.join(base, "other_run")
    os.makedirs(other, exist_ok=True)
    foreign = copy.deepcopy(unsigned)
    foreign["id"] = "cmd_u3"
    foreign["sig"] = OperatorKey(other).sign(foreign)
    tries("a command signed with another run's operator key", foreign)
    with open(path_b, "a") as f:
        f.write("# edited after the operator signed\n")
    tries("the operator's signed command after the task file it names was changed", copy.deepcopy(signed_b))

    # E1 same objective, new boundaries: the operator extends the duration of the running objective
    base2 = scratch("next_e1")
    L2 = make_lab(os.path.join(base2, "run"), lab_cfg=LabConfig(seed=3, target=100000, budget=1000.0, horizon_s=86400.0),
                  task_edit=edit)
    src.world = L2.world
    v2 = pinned(L2.task, L2.clock, base2, L2.run_dir)
    L2.rt.verifier = v2
    L2.rt.run(max_iterations=5)
    rawA = yaml.safe_load(open(os.path.join(PROJECT, "tasks", "lab_staging.yaml")))
    rawA.update(objective_id=L2.task.objective_id, objective=L2.task.objective)
    rawA["success"].update(verifier="metric_threshold", metric="lab.adopters.total", target=100000)
    rawA["limits"]["duration_s"] = 10 * 86400
    pa = os.path.join(base2, "a_extended.yaml")
    yaml.safe_dump(rawA, open(pa, "w"))
    c2 = OperatorControl(L2.run_dir).assign_objective(pa)
    L2.rt.step()
    r2 = RunReader(L2.run_dir).state()["control_results"][c2]
    t0 = L2.clock.now()
    for _ in range(3000):
        if L2.rt.status != "running" or L2.clock.now() - L2.rt.started_at > 3 * 86400:
            break
        L2.rt.step()
    R.check("E1 after the operator extends the running objective's duration (signed assignment of the same objective, "
            "accepted by the runtime; old duration 1 day, new 10 days), the verifier does not end it before the new deadline",
            not (L2.rt.status == "failed" and L2.clock.now() - L2.rt.started_at < 10 * 86400),
            {"assign": r2.get("detail"), "status": L2.rt.status, "reason": L2.rt.status_reason[:200],
             "days_after_start": round((L2.clock.now() - L2.rt.started_at) / 86400, 2),
             "runtime_duration_s": L2.rt.task.limits.duration_s})
sys.exit(R.done())
