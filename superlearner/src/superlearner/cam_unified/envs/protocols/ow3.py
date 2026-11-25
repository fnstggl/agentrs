"""OW-3 Technical invention and resource efficiency (spec 16): a substantial, independently measured improvement of
infrastructure efficiency - here the routing controller of a production network (Crucible's engineering lab).

Ground truth is executable measurement: a deterministic flow model of the network under a hidden traffic process
scores whatever routing program is deployed; ``lab_improvement_vs_baseline(window_h)`` re-runs the incumbent
controller on the recorded snapshot seeds of the window (paired), so the improvement is causal, not a dashboard
reading. Dropped traffic costs a timeout, so a controller cannot look fast by losing traffic. The agent holds the
engineer seat (handle ``eng``: member of the operator with ``lab.operate``); it must write a controller (Python,
sandboxed), test it with ``lab.experiment`` and deploy it with ``lab.propose_change``.

Threshold 0.3 (30% lower effective delay than the incumbent, sustained on 3 daily checks within 14 days); the
reference greedy controller reaches about 0.75-0.84 (measured), so the threshold is feasible but not trivial.
Held-out: the larger scale-free ``engineering_lab_heldout`` network with datacenter-hub traffic.
"""
from __future__ import annotations

from .base import COMMON_UNSUPPORTED, Control, TaskProtocol, idle_control

DAY = 86400
THRESHOLDS = {"improvement_vs_baseline": 0.3}


def make_task(world_family: str, seed: int, threshold: float = THRESHOLDS["improvement_vs_baseline"],
              horizon_days: int = 14, sustain: int = 3, endowment: int = 1_000_000):
    from crucible.goals.contracts import GoalContract
    m = {"metric": "lab_improvement_vs_baseline", "args": {"window_h": 24}}
    text = (f"You are the network engineer of the operator of a production network. Make its routing substantially "
            f"more efficient: success is verified when the controller in production lowers the effective mean delay "
            f"of the network's real traffic by at least {int(threshold * 100)}% against the incumbent controller over "
            f"the preceding 24 hours, on {sustain} consecutive daily checks within {horizon_days} days.")
    g = GoalContract("ow3_efficiency", text, {"family": world_family, "seed": int(seed), "overrides": {}}, "$self",
                     dict(m, op=">=", value=float(threshold)), dict(m, target=float(threshold)), horizon_days * DAY,
                     sustain, endowment=endowment, tags=("ow3", "outcome_only"), progress_every_s=3600)
    g.protocol = "ow3"
    return g


def lab_engineer(source_name: str):
    """A scripted engineer that commits one controller source from Crucible's lab module, tests it once, waits a
    day and deploys it. With GREEDY_SOURCE (the lab's own reference design) it is a privileged ceiling; with
    BASELINE_SOURCE or INVALID_SOURCE it is a shortcut that must fail."""
    from crucible.agents.scripted import ScriptPolicy, call, ok
    from crucible.worlds import engineering_lab as LAB
    src = getattr(LAB, source_name)

    class LabEngineer(ScriptPolicy):
        name = f"lab_engineer:{source_name.lower()}"

        def script(self):
            r = yield call("code.create_repo", name="routing")
            if not ok(r):
                return
            repo = r["result"]["repo"]
            yield call("code.commit", repo=repo, files={"ctrl.py": src}, message="controller")
            yield call("lab.experiment", repo=repo, target="ctrl.py:route", scenarios=8)
            yield from self.wait(24)
            yield call("lab.propose_change", repo=repo, target="ctrl.py:route")
            while True:
                yield None
    return LabEngineer()


PROTOCOL = TaskProtocol(
    name="ow3", title="Technical invention: >=30% verified routing-efficiency improvement", spec_ref="16 OW-3",
    make_task=make_task,
    splits={"train": [("engineering_lab", 11), ("engineering_lab", 12)], "test": [("engineering_lab_heldout", 11)]},
    horizon_days=14, thresholds=THRESHOLDS, handles=("eng",),
    controls=[Control("lab_greedy", lambda: lab_engineer("GREEDY_SOURCE"), "ceiling", privileged=True,
                      note="deploys the lab's reference congestion-aware controller"),
              Control("lab_baseline", lambda: lab_engineer("BASELINE_SOURCE"), "shortcut",
                      note="re-deploys the incumbent; must fail"),
              Control("lab_invalid", lambda: lab_engineer("INVALID_SOURCE"), "shortcut",
                      note="plausible but broken controller; must fail"),
              idle_control()],
    unsupported=COMMON_UNSUPPORTED + [
        "No non-privileged scripted engineer exists (the only competent controller is the lab's reference design)."],
    arms=["CAM Unified", "frozen semantic agent with tools", "+ matched planning compute"])
