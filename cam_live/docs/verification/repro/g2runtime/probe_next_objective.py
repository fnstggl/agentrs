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
v = build_verifier(L.task, os.path.join(base, "verifier"), L.clock, src, log_path=os.path.join(base, "v.jsonl"), pin=True)
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
        bool(verified_b) or not res["ok"],
        {"assign_ok": res["ok"], "status": L.rt.status, "status_reason": L.rt.status_reason[:200],
         "verify_errors": len(errs), "first_error": errs[0][:200] if errs else "",
         "world_total": L.world.total(), "target_b": target_b, "decisions_for_b": L.rt.counters["decisions"]})
sys.exit(R.done())
