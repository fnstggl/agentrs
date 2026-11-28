"""G2-learn probe: is held-out validation evidence available to the agent's first promotion (CL-05 integrated)?

    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_promotion_evidence.py [--reps 3]

Replicates the two lifetimes of tests/integration/test_int2_checkpoints.py (seeds 31 / 32, 14 / 3 decisions, no
backbone proposals) and prints: learner.validate() after the first lifetime, the first promotion decision (made when the second objective starts), and the
number of records flagged for each validation set (``val:<provider>``). Run it on two commits to compare (PYTHONPATH = an export).
"""
import collections
import os
import shutil
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import TESTS, WORK, check, finish, info  # noqa: E402

sys.path.insert(0, os.path.join(TESTS, "integration"))
from int_helpers import Recorder, agent_config, run_lifetime  # noqa: E402

import torch  # noqa: E402

from cam_live.agent import CamLiveAgent  # noqa: E402
from cam_live.contracts import SimClock  # noqa: E402

reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3
torch.set_num_threads(2)
promoted = 0
for k in range(reps):
    d = os.path.join(WORK, f"promotion_{k}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False, promote_every=10_000, retention_records=8),
                         os.path.join(d, "agent"), clock=SimClock())
    try:
        rec = Recorder(agent)
        run_lifetime(rec, agent.clock, d, 31, max_steps=30, max_decisions=14)
        val = agent.learner.validate()            # what the promotion at the objective change will see
        run_lifetime(rec, agent.clock, d, 32, max_steps=6, max_decisions=3)
        first = agent.promotions[0] if agent.promotions else {}
        db = sqlite3.connect(f"file:{os.path.join(agent.dir, 'experience.sqlite')}?mode=ro", uri=True)
        flags = dict(db.execute("SELECT loss, SUM(flag) FROM eligibility WHERE loss LIKE 'val:%' GROUP BY loss").fetchall())
        n = db.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        db.close()
        info(f"run {k}: records / first promotion (promoted, reasons) / validate() / validation flags",
             (n, (first.get("promoted"), first.get("reasons")), {a: round(b, 4) for a, b in val.items()},
              {a: b for a, b in flags.items() if b}))
        promoted += int(bool(first.get("promoted")))
    finally:
        agent.close()
check(f"the first promotion after the lifetime is granted on held-out evidence in every run", promoted == reps,
      f"{promoted} of {reps}")
sys.exit(finish())
