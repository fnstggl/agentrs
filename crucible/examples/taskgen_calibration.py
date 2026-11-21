"""Parts 17-18 end to end: mine tasks from running worlds, validate every one, lock the held-out set, and
calibrate the accepted tasks against a panel. Writes docs/experiments/taskgen_calibration.json, plus the sealed
held-out file docs/experiments/heldout_tasks.json.

Panel:
* idle;
* random;
* the competent scripted control for the task family;
* optionally a frozen LLM (`--llm Qwen/Qwen3-4B`). It gets the task's description as its goal and decides
  every 3 simulated hours.

Usage: PYTHONPATH=crucible/src python crucible/examples/taskgen_calibration.py [--llm MODEL] [family:seed ...]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.scripted import IdlePolicy, RandomPolicy  # noqa: E402
from crucible.evaluation.calibration import calibrate  # noqa: E402
from crucible.goals import taskgen  # noqa: E402

DAY = 86400
HERE = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=None)
    ap.add_argument("--from-file", default=None, help="skip mining/validation; calibrate tasks from a previous run")
    ap.add_argument("worlds", nargs="*", default=["open_tight_compute:2", "open_tight_compute:3", "open_digital:4"])
    a = ap.parse_args()
    t0 = time.time()
    if a.from_file:
        prev = json.load(open(a.from_file))
        accepted = [taskgen.goal_from_dict(r["goal"]) for r in prev["validation"] if r["verdict"] == "accepted"]
        out = prev
    else:
        goals, reports, mined = [], [], []
        for spec in a.worlds:
            fam, seed = spec.split(":")
            w, actor, world, opps = taskgen.mine_world(fam, int(seed), 3 * DAY)
            gs = taskgen.propose(opps, world)
            mined.append({"world": spec, "opportunities": len(opps), "tasks": len(gs),
                          "families": sorted({g.family for g in gs})})
            for g in gs:
                r = taskgen.TaskValidator().validate(g)
                print(spec, g.family, g.id, r.verdict, r.reasons, r.difficulty, round(time.time() - t0), flush=True)
                goals.append(g)
                reports.append(r)
        accepted = [g for g, r in zip(goals, reports) if r.verdict == "accepted"]
        seal = taskgen.lock_heldout([g for g in accepted if g.split == "heldout"],
                                    [r for r in reports if r.verdict == "accepted"],
                                    os.path.join(HERE, "heldout_tasks.json"))
        out = {"mined": mined, "validation": [r.to_dict() for r in reports], "heldout_seal": seal,
               "summary": {"proposed": len(goals), "accepted": len(accepted),
                           "rejected_by_reason": {}}}
        for r in reports:
            if r.verdict == "rejected":
                k = r.reasons[0].split(":")[0]
                out["summary"]["rejected_by_reason"][k] = out["summary"]["rejected_by_reason"].get(k, 0) + 1
    panel = {"idle": lambda g: IdlePolicy(), "random": lambda g: RandomPolicy()}

    def competent(g):
        ctrls = taskgen.default_controls(g)
        return next(iter(ctrls.values()))() if ctrls else None
    panel["competent"] = competent
    dts = {}
    if a.llm:
        from crucible.agents.llm import HFBackend, LLMPolicy
        shared = {}

        def llm(g):   # same interface as the final challenge: every tool listed in full, 32k context
            if "backend" not in shared:
                shared["backend"] = HFBackend(a.llm, "cuda", max_context=32768)
            return LLMPolicy(model=a.llm, goal=g.description, max_new_tokens=320, backend=shared["backend"],
                             tool_order="interleave", tool_chars=24000)
        panel[f"llm:{a.llm.split('/')[-1]}"] = llm
        dts[f"llm:{a.llm.split('/')[-1]}"] = 3 * 3600
        panel = {k: v for k, v in panel.items() if k.startswith("llm:")}
    cal = calibrate(accepted, panel, dt=dts)
    key = "calibration_llm" if a.llm else "calibration"
    if a.llm:
        out["calibration_llm_interface"] = {"tool_order": "interleave", "tool_chars": 24000, "max_context": 32768,
                                            "decision_every_s": 3 * 3600}
    out[key] = cal
    out[key + "_wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(HERE, "taskgen_calibration.json"), "w"), indent=1, default=str)
    print(json.dumps({"summary": out.get("summary"), key: cal["summary"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
