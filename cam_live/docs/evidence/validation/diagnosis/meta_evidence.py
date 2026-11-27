"""Why meta-updates are skipped in the integrated runtime test setting (INT2, §9 d/e of docs/INTEGRATION.md).

Runs the configuration of ``tests/integration/test_agent_runtime.py`` (CPU, tiny backbone + debug core, LabWorld
lifetimes through the real runtime) and prints, as JSON: the meta-update skip reasons, the outcome-ledger entry
kinds, how many query episodes / independent groups the meta-learner can use at the end, and how many executed
interventions are resolved (every goal-relevant question answered) for the meta-learner.

Usage: python meta_evidence.py <scratch dir> [--seeds 11,12,13] [--drain] [--short] [--rule R] [--update-every N]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tests",
                                "integration"))

from cam_live.agent import CamLiveAgent  # noqa: E402
from cam_live.contracts import RunFiles, SimClock  # noqa: E402
from cam_live.runtime.logs import iter_jsonl  # noqa: E402
from cam_live.runtime.loop import RuntimeConfig  # noqa: E402
from int_helpers import Recorder, agent_config, run_lifetime  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scratch")
    ap.add_argument("--seeds", default="11,12,13")
    ap.add_argument("--drain", action="store_true")
    ap.add_argument("--short", action="store_true", help="30-min attribution windows instead of 6 h")
    ap.add_argument("--rule", default="")
    ap.add_argument("--update-every", type=int, default=3)
    a = ap.parse_args(argv)
    root = tempfile.mkdtemp(dir=a.scratch)
    kw = {"policy_update_every": a.update_every}
    if a.rule:
        kw["selection_rule"] = a.rule
    clock = SimClock()
    agent = CamLiveAgent(agent_config(**kw), os.path.join(root, "agent"), clock=clock)
    rec = Recorder(agent)
    lab = {"outcome_horizon_s": 1800.0, "mean_delay_s": (300.0, 900.0)} if a.short else None
    seeds = [int(x) for x in a.seeds.split(",")]
    rt = RuntimeConfig(fsync=False, candidates_k=8, keep_checkpoints=2, drain_after_exhaustion=a.drain)
    lts = [run_lifetime(rec, clock, root, s, max_steps=40, max_decisions=22, lab_kw=lab, rt_cfg=rt) for s in seeds]
    m = agent.meta
    metas = [e["data"] for lt in lts for e in iter_jsonl(os.path.join(lt.run_dir, RunFiles.META))]
    eps, used, held = m._collect()
    acts = [r for lt in lts for r in agent.store.trajectory(lt.task.objective_id)
            if r.result is not None and r.result.expected_outcomes]
    out = {"args": vars(a), "lifetimes": [{"status": lt.status, "cycles": len(lt.cycles)} for lt in lts],
           "meta_calls": len(metas), "applied": sum(1 for x in metas if x["applied"]),
           "applied_nonzero": sum(1 for x in metas if x["applied"] and x.get("meta_grad_norm", 0) > 0),
           "skipped": dict(collections.Counter(x["skipped_reason"] for x in metas if not x["applied"])),
           "ledger_kinds": dict(collections.Counter(e.kind for e in m.ledger.entries())),
           "episodes_at_end": len(eps), "independent_groups_used": len(used), "groups_held_out": len(held),
           "interventions": len(acts), "interventions_resolved": sum(1 for r in acts if m._resolved(r)),
           "interventions_with_unanswerable_duplicate": sum(
               1 for r in acts if any(q.goal_relevant and q.metric in {x.metric for x in r.result.expected_outcomes}
                                      and q.query_id not in {x.query_id for x in r.result.expected_outcomes}
                                      for q in r.pending_queries))}
    print(json.dumps(out, indent=1))
    agent.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
