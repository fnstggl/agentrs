"""G2-final probe: members present at the start of the experiment are not counted, also when they leave and rejoin
(G2R-2b, INT3B round 3: keyed hashes of the members present at the start, verifier state only).

    G2F_SRC=<archive>/cam_live/src .venv/bin/python cam_live/docs/verification/repro/g2final/probe_baseline_rejoin.py

The membership verifier with the live task's evidence source (DiscordMembershipSource) on an in-process fake of the
REST API that lists CURRENT members only, with the platform's joined_at (a rejoin resets it), built as
verifier_service.serve builds it. No network.
  1  window_start_ts set, first reading at the start: 10 members present at the start leave and rejoin -> value 0
  2  the same across a verifier restart (new verifier object on the same state directory)
  3  the baseline file is 0600 and the hash-chained log holds counts, not member ids or their hashes
  4  first reading only 5 days after window_start_ts: a member present at the start who left before that reading
     and rejoins later (what the verifier can know about him: nothing)
  5  new members who join after the start still count (no false exclusion)
"""
import datetime as dt
import json
import os
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2fcommon import PROJECT, Report, scratch  # noqa: E402

from cam_live.contracts import SimClock  # noqa: E402
from cam_live.harness.discord_membership import DiscordMembershipSource  # noqa: E402
from cam_live.runtime import tasks as T  # noqa: E402
from cam_live.runtime.verifier import build_verifier  # noqa: E402

R = Report("probe_baseline_rejoin")
DAY = 86400.0
T0 = 1_800_000_000.0
GUILD, OWNER = "900000000000000001", "900000000000000002"


class FakeDiscord:
    def __init__(self):
        self.members = {}
        self.requests_sent = 0

    def _request(self, method, path, *, route, params=None, **_kw):
        self.requests_sent += 1
        if path == f"/guilds/{GUILD}":
            return {"id": GUILD, "owner_id": OWNER}
        if path == f"/guilds/{GUILD}/channels":
            return [{"id": "800000000000000001", "type": 0}]
        if path.startswith("/channels/"):
            return []
        if path == f"/guilds/{GUILD}/members":
            after = int((params or {}).get("after", "0"))
            rows = [{"user": {"id": u, "bot": False},
                     "joined_at": dt.datetime.fromtimestamp(j, dt.timezone.utc).isoformat(), "pending": False,
                     "flags": 0} for u, j in sorted(self.members.items(), key=lambda kv: int(kv[0])) if int(u) > after]
            return rows[: int((params or {}).get("limit", "1000"))]
        raise AssertionError(path)


def uid(i):
    return str(1_100_000_000_000_000_000 + i)


def task():
    raw = T.load_task_raw(os.path.join(PROJECT, "tasks", "discord_100.yaml"))
    raw["success"]["exclusions"]["test_account_ids"] = []
    raw["resources"]["discord_guild_id"] = GUILD
    raw["measurement"]["window_start_ts"] = T0
    raw["limits"] = {"duration_s": 60 * DAY, "money": 10, "compute_s": 3600, "max_external_actions": 100,
                     "max_decisions": 100}
    return T.parse_task(raw)


def mk(fake, clock, d, tk):
    src = DiscordMembershipSource(GUILD, lambda: "unused", client=fake)
    return build_verifier(tk, os.path.join(d, "verifier"), clock, src, log_path=os.path.join(d, "v.jsonl"),
                          operator_dir=d)


# ---- 1 / 2 / 3 / 5 -------------------------------------------------------------------------------------------------
d = scratch("rejoin")
tk = task()
clock = SimClock(T0 + 600.0)
fake = FakeDiscord()
fake.members[OWNER] = T0 - 400 * DAY
present = [uid(i) for i in range(10)]
for u in present:
    fake.members[u] = T0 - 30 * DAY
v = mk(fake, clock, d, tk)
vo0 = v.verify(tk)
clock.advance(1 * DAY)
for u in present:
    del fake.members[u]                      # they leave ...
v.verify(tk)
clock.advance(1 * DAY)
for u in present:
    fake.members[u] = clock.now()            # ... and rejoin: the platform's joined_at is now after the start
vo = v.verify(tk)
R.check("1 ten members present at the start who leave and rejoin are not counted", vo.value == 0,
        {"value_at_start": vo0.value, "value_after_rejoin": vo.value, "notes": vo.notes[:300]})
v2 = mk(fake, clock, d, tk)
vo2 = v2.verify(tk)
R.check("2 a restarted verifier (same state directory) still excludes them", vo2.value == 0, {"value": vo2.value})
bp = os.path.join(d, "verifier", "baseline_members.json")
mode = oct(stat.S_IMODE(os.stat(bp).st_mode)) if os.path.exists(bp) else None
log = open(os.path.join(d, "v.jsonl")).read()
leak = [u for u in present if u in log]
hashes = list(json.load(open(bp)).values())[0] if os.path.exists(bp) else []
hash_in_log = [h for h in hashes if h.split(":")[-1] in log]
R.check("3 the baseline file is 0600 and the hash-chained log carries neither member ids nor their baseline hashes",
        mode == "0o600" and not leak and not hash_in_log,
        {"baseline_file_mode": mode, "ids_in_log": len(leak), "baseline_hashes_in_log": len(hash_in_log),
         "baseline_entries": len(hashes)})
newcomers = [uid(100 + i) for i in range(5)]
for u in newcomers:
    fake.members[u] = clock.now()
clock.advance(60.0)
vo3 = v2.verify(tk)
R.check("5 five new members who join after the start count (no false exclusion)", vo3.value == 5, {"value": vo3.value})

# ---- 4 late first reading -------------------------------------------------------------------------------------------
d4 = scratch("late")
clock4 = SimClock(T0 + 600.0)
fake4 = FakeDiscord()
fake4.members[OWNER] = T0 - 400 * DAY
early_leaver = uid(50)
fake4.members[early_leaver] = T0 - 30 * DAY                 # present at window_start_ts ...
clock4.advance(1 * DAY)
del fake4.members[early_leaver]                             # ... leaves on day 1, before the verifier's first reading
clock4.advance(4 * DAY)
v4 = mk(fake4, clock4, d4, tk)
v4.verify(tk)                                               # first reading on day 5
clock4.advance(1 * DAY)
fake4.members[early_leaver] = clock4.now()                  # rejoins on day 6
vo4 = v4.verify(tk)
R.note("4 first reading 5 days after window_start_ts: a member present at the start who left before that reading "
       "and rejoined later", {"value": vo4.value, "counted": vo4.value == 1,
                              "why": "the baseline is the member list at the verifier's first reading; the REST list "
                                     "has no history, so a member who left before it is unknown to the verifier"})
sys.exit(R.done())
