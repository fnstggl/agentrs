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
        self.messages = []                 # author ids of the latest messages of the one text channel
        self.requests_sent = 0

    def _request(self, method, path, *, route, params=None, **_kw):
        self.requests_sent += 1
        if path == f"/guilds/{GUILD}":
            return {"id": GUILD, "owner_id": OWNER}
        if path == f"/guilds/{GUILD}/channels":
            return [{"id": "800000000000000001", "type": 0}]
        if path == "/channels/800000000000000001/messages":
            return [{"author": {"id": a, "bot": False}} for a in self.messages]
        if path == f"/guilds/{GUILD}/members":
            after = int((params or {}).get("after", "0"))
            rows = [{"user": {"id": uid, "bot": False},
                     "joined_at": dt.datetime.fromtimestamp(j, dt.timezone.utc).isoformat(), "pending": False, "flags": 0}
                    for uid, j in sorted(self.members.items(), key=lambda kv: int(kv[0])) if int(uid) > after]
            return rows[: int((params or {}).get("limit", "1000"))]
        raise AssertionError(path)


def uid(i):
    return str(1_100_000_000_000_000_000 + i)


def task_with_window(window=True):
    raw = T.load_task_raw(os.path.join(PROJECT, "tasks", "discord_100.yaml"))
    raw["success"]["exclusions"]["test_account_ids"] = []
    raw["resources"]["discord_guild_id"] = GUILD
    if window:
        raw["measurement"]["window_start_ts"] = T0
    raw["limits"] = {"duration_s": 60 * DAY, "money": 10, "compute_s": 3600, "max_external_actions": 100,
                     "max_decisions": 100}
    return T.parse_task(raw)


class FakeLedger:
    """The authorization service's limits_state as the verifier service reads it (what it charged for the task)."""
    def limits_state(self):
        return {"tasks": {"discord-100": {"used": {"money": 3.5, "external_actions": 12, "decisions": 40,
                                                    "compute_s": 900.0}}}}


def verifier(fake, clock, name, task=None, d=None):
    task = task or task_with_window()
    src = DiscordMembershipSource(GUILD, lambda: "unused", client=fake)
    d = d or scratch(name)
    kw = {}
    try:                                   # the final freeze wires CAM's ledger and the operator's key directory
        from cam_live.runtime.verifier import authorization_cost_source
        kw = {"cost_source": authorization_cost_source(FakeLedger(), lambda _o: "USD"), "operator_dir": d}
    except ImportError:
        pass
    return task, build_verifier(task, os.path.join(d, "verifier"), clock, src, log_path=os.path.join(d, "v.jsonl"),
                                **kw)


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
fake.messages = [uid(5), uid(5), uid(6)]
vo = v.verify(task)
names = sorted(q.name for q in vo.secondary)
R.note("C secondary quantities reported with the live Discord source", names)
R.check("C1 voluntary participation is measured", any(n.startswith("secondary.participation") for n in names), names)
R.check("C2 consented participant feedback is measured, or reported as not measured (never silently absent)",
        any(n.startswith("secondary.feedback") for n in names) or "feedback" in vo.notes.lower(), vo.notes[-300:])
use = [n for n in names if n.startswith("secondary.resource_use")]
CAM_OWN = {"secondary.resource_use.discord.verifier.requests", "secondary.resource_use.external_actions",
           "secondary.resource_use.decisions", "secondary.resource_use.compute_s"}
R.check("C3 usefulness of community resources (spec 13.2) is measured — not the verifier's request count, not CAM's own "
        "action / decision / compute counts — or reported as not measured",
        (bool(use) and not set(use) <= CAM_OWN) or "usefulness" in vo.notes.lower(), {"resource_use": use,
                                                                                    "notes": vo.notes[-300:]})
R.check("C4 operating cost is measured (build_verifier as verifier_service.serve calls it: no cost_source)",
        any(n.startswith("secondary.operating_cost") for n in names), names)
R.check("C5 time expenditure is measured", "secondary.time_elapsed_s" in names, names)
R.check("C6 the primary value is the confirmed member count only (secondary kept separate)",
        vo.value == 5.0, {"value": vo.value})
# ---- round 2 (final freeze): the start baseline ----------------------------------------------------------------
# A2 rejoin: a member who was in the guild at the start leaves and joins again during the experiment
clock = SimClock(T0 + DAY)
fake = FakeDiscord()
fake.members[OWNER] = T0 - 400 * DAY
for i in range(10):
    fake.members[uid(i)] = T0 - 30 * DAY             # 10 members before the start
task, v = verifier(fake, clock, "a2")
v.verify(task)                                        # baseline: 10 before the start, 0 counted
for i in range(10):
    fake.members[uid(i)] = T0 + 2 * DAY               # all 10 leave and rejoin on day 2 (Discord's new joined_at)
clock.advance(2 * DAY)
vo = v.verify(task)
R.check("A2 members who were in the guild at the start and rejoin during the experiment are not counted as attracted "
        "(10 pre-existing members leave and rejoin; nobody new joined)", vo.value == 0,
        {"value": vo.value, "notes": vo.notes[:300]})
# A3 no window_start_ts: the start is the verifier's first reading; a restarted verifier keeps it
clock = SimClock(T0)
fake = FakeDiscord()
fake.members[OWNER] = T0 - 400 * DAY
for i in range(5):
    fake.members[uid(i)] = T0 - 30 * DAY              # 5 before the start
tw = task_with_window(window=False)
d3 = scratch("a3")
task, v = verifier(fake, clock, "a3", task=tw, d=d3)
v.verify(task)                                         # first reading at T0
for i in range(5, 8):
    fake.members[uid(i)] = T0 + DAY                    # 3 join on day 1
clock.advance(3 * DAY)
task, v2 = verifier(fake, clock, "a3", task=tw, d=d3)  # verifier process restarted (same state directory)
vo = v2.verify(task)
R.check("A3 without window_start_ts the start is the first reading and survives a verifier restart (3 joined after it)",
        vo.value == 3, {"value": vo.value, "notes": vo.notes[:200]})
# A4 a first reading taken late: members attracted between the run's start and the verifier's first reading
clock = SimClock(T0 + 5 * DAY)                         # the verifier is first asked five days after CAM started
fake = FakeDiscord()
fake.members[OWNER] = T0 - 400 * DAY
for i in range(4):
    fake.members[uid(i)] = T0 + 2 * DAY                # 4 attracted on day 2, before the first reading
task, v = verifier(fake, clock, "a4", task=task_with_window(window=False))
vo = v.verify(task)
R.note("A4 no window_start_ts and a late first reading: members attracted before it are not counted (conservative)",
       {"value": vo.value, "joined_before_the_first_reading": 4})
sys.exit(R.done())
