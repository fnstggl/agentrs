"""G2-runtime probe: the membership verifier with the evidence source the live objective actually uses.

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_verifier_live_source.py

The Discord task defaults to ``cam_live.harness.discord_membership:membership_source`` (DiscordMembershipSource, the
verifier's own read of ``GET /guilds/{id}/members``). No network: the source is given an in-process fake client that
answers like the Discord REST API (current members only, as the real endpoint does). The verifier is built exactly
as ``verifier_service.serve`` builds it (``build_verifier(task, state_dir, clock, source, log_path, pin=True)``).

Checks:
  A  members who joined the guild before the experiment's window began are not counted as members CAM attracted
  B  seven-day retention reflects members who left (the REST list no longer contains them)
  C  the secondary metrics of spec 13.2 are measured for the live source: voluntary participation, consented
     feedback, usefulness of community resources, operating cost, time
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2common import PROJECT, Report, scratch  # noqa: E402

from cam_live.contracts import GoalStatus, SimClock  # noqa: E402
from cam_live.harness.discord_membership import DiscordMembershipSource  # noqa: E402
from cam_live.runtime import tasks as T  # noqa: E402
from cam_live.runtime.verifier import build_verifier  # noqa: E402

R = Report("probe_verifier_live_source")
DAY = 86400.0
T0 = 1_800_000_000.0                       # experiment start (measurement.window_start_ts)
GUILD, OWNER = "900000000000000001", "900000000000000002"


class FakeDiscord:
    """Answers the two requests DiscordMembershipSource makes, like the REST API: only CURRENT members are listed."""

    def __init__(self):
        self.members = {}                  # uid -> joined_at (unix)
        self.requests_sent = 0

    def _request(self, method, path, *, route, params=None, **_kw):
        self.requests_sent += 1
        if path == f"/guilds/{GUILD}":
            return {"id": GUILD, "owner_id": OWNER}
        if path == f"/guilds/{GUILD}/members":
            after = int((params or {}).get("after", "0"))
            rows = [{"user": {"id": uid, "bot": False},
                     "joined_at": dt.datetime.fromtimestamp(j, dt.timezone.utc).isoformat(), "pending": False, "flags": 0}
                    for uid, j in sorted(self.members.items(), key=lambda kv: int(kv[0])) if int(uid) > after]
            return rows[: int((params or {}).get("limit", "1000"))]
        raise AssertionError(path)


def uid(i):
    return str(1_100_000_000_000_000_000 + i)


def task_with_window():
    raw = T.load_task_raw(os.path.join(PROJECT, "tasks", "discord_100.yaml"))
    raw["success"]["exclusions"]["test_account_ids"] = []
    raw["resources"]["discord_guild_id"] = GUILD
    raw["measurement"]["window_start_ts"] = T0
    raw["limits"] = {"duration_s": 60 * DAY, "money": 10, "compute_s": 3600, "max_external_actions": 100,
                     "max_decisions": 100}
    return T.parse_task(raw)


def verifier(fake, clock, name):
    task = task_with_window()
    src = DiscordMembershipSource(GUILD, lambda: "unused", client=fake)
    d = scratch(name)
    return task, build_verifier(task, os.path.join(d, "verifier"), clock, src, log_path=os.path.join(d, "v.jsonl"))


# ---- A: members who were in the guild before the experiment began ------------------------------------------
clock = SimClock(T0 + 1 * DAY)
fake = FakeDiscord()
fake.members[OWNER] = T0 - 400 * DAY
for i in range(100):
    fake.members[uid(i)] = T0 - 30 * DAY          # all joined a month BEFORE window_start_ts
task, v = verifier(fake, clock, "a")
vo = v.verify(task)
R.check("A members who joined before measurement.window_start_ts are not counted as attracted members "
        "(100 pre-existing members, 0 joined during the experiment)",
        vo.status != GoalStatus.ACHIEVED and vo.value == 0,
        {"status": vo.status, "value": vo.value, "target": vo.target, "notes": vo.notes})

# ---- B: seven-day retention with departures -------------------------------------------------------------------
clock = SimClock(T0)
fake = FakeDiscord()
fake.members[OWNER] = T0 - 400 * DAY
for i in range(10):
    fake.members[uid(i)] = T0 + 0.5 * DAY         # 10 humans join on day 0
task, v = verifier(fake, clock, "b")
clock.advance(2 * DAY)
v.verify(task)                                     # the verifier saw all ten
for i in range(5):
    del fake.members[uid(i)]                      # five leave on day 2 (the REST list stops listing them)
clock.advance(8 * DAY)                             # day 10: every one of the ten joined more than 7 days ago
vo = v.verify(task)
ret = {q.name: q.value for q in vo.secondary}.get("secondary.retention_7d")
R.check("B seven-day retention counts the members who left (10 joined, 5 left within 2 days: true value 0.5)",
        ret is not None and abs(ret - 0.5) < 1e-9, {"secondary.retention_7d": ret, "confirmed": vo.value})

# ---- C: which secondary metrics the live source yields --------------------------------------------------------
names = sorted(q.name for q in vo.secondary)
R.note("C secondary quantities reported with the live Discord source", names)
R.check("C1 voluntary participation is measured", any(n.startswith("secondary.participation") for n in names), names)
R.check("C2 consented participant feedback is measured", any(n.startswith("secondary.feedback") for n in names), names)
use = [n for n in names if n.startswith("secondary.resource_use")]
R.check("C3 usefulness of community resources is measured (not the verifier's own API request count)",
        bool(use) and use != ["secondary.resource_use.discord.verifier.requests"], use)
R.check("C4 operating cost is measured (build_verifier as verifier_service.serve calls it: no cost_source)",
        any(n.startswith("secondary.operating_cost") for n in names), names)
R.check("C5 time expenditure is measured", "secondary.time_elapsed_s" in names, names)
R.check("C6 the primary value is the confirmed member count only (secondary kept separate)",
        vo.value == 5.0, {"value": vo.value})
sys.exit(R.done())
