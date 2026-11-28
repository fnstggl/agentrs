"""G2-runtime probe (round 2): the membership exclusion rules for members who joined AFTER the experiment's start.

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_verifier_rules_after_start.py

At the final freeze members who joined before the start are not counted (G2R-2), so G-rt's p06 scenarios (members with
join times before the verifier's first reading) now report 0 members and 14 of its checks fail by construction. This
probe re-runs the core of those scenarios with every member joining after ``measurement.window_start_ts``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2common import PROJECT, Report, scratch  # noqa: E402

from cam_live.contracts import GoalStatus, SimClock  # noqa: E402
from cam_live.runtime import tasks as T  # noqa: E402
from cam_live.runtime.verifier import MemberRecord, build_verifier  # noqa: E402

R = Report("probe_verifier_rules_after_start")
T0 = 1_800_000_000.0


class Src:
    name, live = "g2-memory", False

    def __init__(self, recs):
        self.recs = recs

    def list_members(self):
        return list(self.recs)


def run(recs, edit=None):
    raw = T.load_task_raw(os.path.join(PROJECT, "tasks", "discord_100.yaml"))
    raw["success"]["exclusions"]["test_account_ids"] = ["t1"]
    raw["resources"]["discord_guild_id"] = "1"
    raw["measurement"]["window_start_ts"] = T0
    raw["limits"] = {"duration_s": 60 * 86400, "money": 1, "compute_s": 1, "max_external_actions": 1, "max_decisions": 1}
    if edit:
        edit(raw)
    task = T.parse_task(raw)
    d = scratch("rules")
    v = build_verifier(task, os.path.join(d, "verifier"), SimClock(T0 + 86400), Src(recs),
                       log_path=os.path.join(d, "v.jsonl"))
    return v.verify(task)


def m(i, **kw):
    return MemberRecord(id=f"u{i}", joined_at=T0 + 3600 + i, **kw)


cases = [
    ("100 unflagged members -> achieved, value 100", [m(i) for i in range(100)],
     lambda o: o.status == GoalStatus.ACHIEVED and o.value == 100),
    ("99 humans + 1 bot -> not achieved", [m(i) for i in range(99)] + [m(99, is_bot=True)],
     lambda o: o.status != GoalStatus.ACHIEVED and o.value == 99),
    ("a declared test account is excluded", [m(i) for i in range(99)] + [MemberRecord(id="t1", joined_at=T0 + 9)],
     lambda o: o.value == 99),
    ("one account listed 150 times counts once", [m(1) for _ in range(150)], lambda o: o.value == 1),
    ("150 members in screening: undetermined", [m(i, pending=True) for i in range(150)],
     lambda o: o.status != GoalStatus.ACHIEVED and o.value == 0),
    ("purchased flag (any case) excluded", [m(i) for i in range(99)] + [m(99, flags=["Purchased"])],
     lambda o: o.value == 99 and o.status != GoalStatus.ACHIEVED),
    ("second account of a test account's person (same link_key) not counted",
     [m(i) for i in range(99)] + [MemberRecord(id="t1", joined_at=T0 + 9, link_key="p"), m(99, link_key="p")],
     lambda o: o.value == 99),
    ("150 accounts of one person count once", [m(i, link_key="same") for i in range(150)], lambda o: o.value == 1),
    ("an id listed as human and as bot is not counted", [m(i) for i in range(99)] + [m(99), m(99, is_bot=True)],
     lambda o: o.value == 99),
]
for label, recs, ok in cases:
    o = run(recs)
    R.check(label, ok(o), {"status": o.status, "value": o.value, "excluded": o.excluded})
sys.exit(R.done())
