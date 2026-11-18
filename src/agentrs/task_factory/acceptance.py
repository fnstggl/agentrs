"""Task-factory acceptance test (``python -m agentrs.task_factory.acceptance``), no mocked labels.

For every family: generate several tasks through the factory (admission = env builds, initial state
fails the hidden verifier, the scripted expert solves it in the real environment, no duplicates),
execute real expert actions, check that the verifier verdict flipped because of those actions,
persist the trajectories, and test memorisation: replay task A's successful action sequence verbatim
on a different task B of the same family; a family whose tasks can be solved by replaying a memorised
solution is flagged.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

from agentrs.econ.core import REGISTRY, load_all, make_task
from agentrs.econ.rollout import make_env, run_expert
from agentrs.task_factory.factory import TaskFactory


def replay(task, actions: list[str]) -> bool:
    env = make_env(task)
    for a in actions:
        if getattr(env, "done", False):
            break
        try:
            env.step(a)
        except Exception:  # noqa: BLE001
            return False
    ok, _ = env.verify()
    return bool(ok)


def run(out: Path, per_family: int = 3, base_index: int = 900_000) -> dict:
    load_all()
    out.mkdir(parents=True, exist_ok=True)
    fams = sorted(REGISTRY)
    fac = TaskFactory({}, fams, out_dir=None)
    report = {"time": time.time(), "families": {}, "summary": {}}
    traj = gzip.open(out / "acceptance_trajectories.jsonl.gz", "wt")
    for k in fams:
        rec = {"admitted": 0, "rejected": {}, "expert_success": 0, "verdict_flipped": 0, "replay_solved": 0,
               "replay_pairs": 0, "distinct_goals": 0, "specs": []}
        tasks, plans = [], []
        i = 0
        while len(tasks) < per_family and i < per_family * 4:
            i += 1
            try:
                t = make_task(k, "train", base_index + i - 1)
            except Exception as e:  # noqa: BLE001
                rec["rejected"][f"generator crashed: {type(e).__name__}"] = rec["rejected"].get(f"generator crashed: {type(e).__name__}", 0) + 1
                continue
            adm = fac.admit(t, base_index + i - 1)
            if not adm.admitted:
                rec["rejected"][adm.reason] = rec["rejected"].get(adm.reason, 0) + 1
                continue
            rec["admitted"] += 1
            ep = run_expert(t)
            env0 = make_env(t)
            before, _ = env0.verify()
            rec["expert_success"] += int(ep.success)
            rec["verdict_flipped"] += int((not before) and ep.success)
            tasks.append(t)
            plans.append([s.action for s in ep.steps])
            r = ep.to_record()
            for s in r["steps"]:
                s.pop("messages", None)
            traj.write(json.dumps(r, default=str) + "\n")
            s = adm.spec.public_dict()
            rec["specs"].append({x: s[x] for x in ("goal", "environment", "environment_version", "generator_version",
                                                   "tools", "private_keys", "counterparties", "step_budget",
                                                   "expert_steps", "instance_hash")})
        rec["distinct_goals"] = len({t.goal for t in tasks})
        for a in range(len(tasks)):
            b = (a + 1) % len(tasks)
            if a != b:
                rec["replay_pairs"] += 1
                rec["replay_solved"] += int(replay(tasks[b], plans[a]))
        rec["memorized_replay_rate"] = rec["replay_solved"] / rec["replay_pairs"] if rec["replay_pairs"] else None
        report["families"][k] = rec
        print(f"{k:38s} admitted {rec['admitted']} expert {rec['expert_success']} flipped {rec['verdict_flipped']} "
              f"replay-solved {rec['replay_solved']}/{rec['replay_pairs']} rejected {rec['rejected']}", flush=True)
    traj.close()
    fr = report["families"].values()
    report["summary"] = {
        "families": len(report["families"]), "domains": sorted({k.split('/')[0] for k in report["families"]}),
        "tasks_admitted": sum(r["admitted"] for r in fr), "expert_success": sum(r["expert_success"] for r in fr),
        "verdict_flipped_by_actions": sum(r["verdict_flipped"] for r in fr),
        "replay_pairs": sum(r["replay_pairs"] for r in fr), "replay_solved": sum(r["replay_solved"] for r in fr),
        "families_solvable_by_replay": sorted(k for k, r in report["families"].items() if r["replay_solved"]),
        "factory": fac.coverage(),
    }
    (out / "acceptance.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/overnight/factory")
    ap.add_argument("--per-family", type=int, default=3)
    a = ap.parse_args(argv)
    rep = run(Path(a.out), a.per_family)
    print(json.dumps(rep["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
