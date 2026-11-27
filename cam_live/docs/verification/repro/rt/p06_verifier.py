"""G-rt probe 06 — independent verifier: counting rules against adversarial member lists, 'undetermined' instead of
success, signatures, pinned criteria, hash-chained log, what the verifier writes about members.

    .venv/bin/python cam_live/docs/verification/repro/rt/p06_verifier.py
"""
import copy
import json
import math
import os
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import Report, StandInAgent, make_slow_run, read_log, scratch  # noqa: E402

from cam_live.contracts import GoalStatus, Limits, RunFiles, SimClock, TaskSpec, VerifiedOutcome  # noqa: E402
from cam_live.runtime import verifier as V  # noqa: E402

R = Report("p06_verifier")
DAY = 86400.0


class Src:
    name, live = "g-memory-source", False

    def __init__(self, members):
        self.members = members

    def list_members(self):
        return [V.MemberRecord.from_dict(m) if isinstance(m, dict) else m for m in self.members]


def task(target=100, **excl):
    return TaskSpec(objective_id="obj_members", objective="100 members", family="community/membership", mode="staging",
                    success={"verifier": "membership", "metric": "membership.confirmed", "target": target,
                             "confidence": 0.95, "exclusions": excl},
                    limits=Limits(duration_s=30 * DAY, max_decisions=10), measurement={})


def run(members, clock=None, t=None, **excl):
    clock = clock or SimClock()
    t = t or task(**excl)
    d = scratch("p06v")
    v = V.MembershipVerifier(Src(members), clock, os.path.join(d, "verifier"), log_path=os.path.join(d, "verifier.jsonl"),
                             pinned_task=t)
    return v, v.verify(t), d


def humans(n, prefix="u", **kw):
    return [{"id": f"{prefix}{i}", "joined_at": 1_800_000_000.0 - 1000 + i, **kw} for i in range(n)]


# ---- A. counting rules ---------------------------------------------------------------------------------------
_, o, _ = run(humans(100))
R.check("A1 100 unflagged current members -> achieved, value 100", o.status == GoalStatus.ACHIEVED and o.value == 100)
_, o, _ = run(humans(99) + [{"id": "b", "is_bot": True}])
R.check("A2 99 humans + 1 bot -> not achieved, bot excluded", o.status == "pending" and o.value == 99 and o.excluded.get("bots") == 1)
_, o, _ = run(humans(100), test_account_ids=["u0", "u1", "u2"])
R.check("A3 declared test accounts are excluded", o.value == 97 and o.excluded.get("test_accounts") == 3 and o.status == "pending")
_, o, _ = run(humans(1) * 150)
R.check("A4 one account listed 150 times counts once", o.value == 1 and o.excluded.get("duplicates") == 149)
_, o, _ = run(humans(100, left_at=1_799_999_999.0))
R.check("A5 departed members do not count", o.value == 0 and o.excluded.get("departed") == 100)
_, o, _ = run(humans(150, pending=True))
R.check("A6 150 members still in screening: undetermined, success not claimed", o.status == "pending" and o.value == 0
        and o.uncertainty > 0 and "success is not claimed" in o.notes, o.notes)
_, o, _ = run(humans(99) + humans(60, "s", flags=["automation_suspected"]))
R.check("A7 99 confirmed + 60 suspected-automation: undetermined (lower bound 99), not success",
        o.status == "pending" and o.value == 99 and "success is not claimed" in o.notes, o.notes)
_, o, _ = run(humans(60) + humans(60, "p", flags=["purchased"]) + humans(60, "i", invite_code="BULK"), suspect_invite_codes=["BULK"])
R.check("A8 purchased-flag accounts excluded, suspect-invite accounts uncertain", o.value == 60 and o.excluded.get("purchased") == 60
        and o.status == "pending", {"value": o.value, "excluded": o.excluded, "unc": o.uncertainty})
_, o, _ = run(humans(150, link_key="same-person"))
R.check("A9 150 accounts known to belong to one person count once", o.value == 1 and o.excluded.get("duplicate_accounts") == 149)
# deadline behaviour
c = SimClock()
v, o, _ = run(humans(99) + humans(60, "s", flags=["authenticity_unknown"]), clock=c)
c.advance(31 * DAY)
o2 = v.verify(v.pinned_task)
R.check("A10 after the deadline with the target only reachable through uncertain accounts: still undetermined (not success, not failure)",
        o2.status == "pending" and "undetermined" in o2.notes, o2.notes)
c = SimClock()
v, o, _ = run(humans(10), clock=c)
c.advance(31 * DAY)
R.check("A11 after the deadline with 10 of 100: failed", v.verify(v.pinned_task).status == "failed")
# exact binomial quantile against brute force
ok = True
for n, p, lev in ((20, 0.5, 0.05), (60, 0.3, 0.05), (7, 0.9, 0.95), (50, 0.01, 0.5)):
    cum, k = 0.0, 0
    while True:
        cum += math.comb(n, k) * p ** k * (1 - p) ** (n - k)
        if cum >= lev - 1e-12:
            break
        k += 1
    ok &= V.binom_quantile(lev, n, p) == k
R.check("A12 binom_quantile agrees with brute-force summation", ok)

# ---- A'. adversarial lists that slip through --------------------------------------------------------------------
_, o, _ = run(humans(99) + [{"id": "T", "joined_at": 1.0, "link_key": "operator"}, {"id": "T-alt", "joined_at": 2.0, "link_key": "operator"}],
              test_account_ids=["T"])
R.check("A13 a second account of the person behind a declared test account (same link_key) is not counted",
        o.value == 99, {"value": o.value, "status": o.status, "excluded": o.excluded})
_, o, _ = run(humans(99) + [{"id": "B", "is_bot": True, "joined_at": 1.0, "link_key": "farm"}, {"id": "B2", "joined_at": 2.0, "link_key": "farm"}])
R.check("A14 an account sharing its link_key with a bot is not counted as a legitimate human",
        o.value == 99, {"value": o.value, "status": o.status})
_, o, _ = run(humans(99) + [{"id": "P", "joined_at": 1.0, "flags": ["purchased"], "link_key": "buyer"},
                            {"id": "P2", "joined_at": 2.0, "link_key": "buyer"}])
R.check("A15 an account sharing its link_key with a purchased account is not counted", o.value == 99, {"value": o.value, "status": o.status})
_, o, _ = run(humans(99) + [{"id": "x", "joined_at": 1.0}, {"id": "x", "joined_at": 2.0, "is_bot": True}])
R.check("A16 an id listed once as a human and once as a bot is not counted as a confirmed human",
        o.value == 99, {"value": o.value, "status": o.status, "excluded": o.excluded})
_, o, _ = run(humans(60) + humans(60, "p", flags=["Purchased"]))
R.check("A17 exclusion flags match regardless of letter case ('Purchased')", o.value == 60, {"value": o.value, "status": o.status})
_, o, _ = run([{"id": f"n{i}"} for i in range(100)])
R.check("A18 100 records that carry nothing but an id (no join time, no bot field, no screening state) are not reported as "
        "100 confirmed legitimate members / achieved", o.status != GoalStatus.ACHIEVED, {"status": o.status, "value": o.value,
                                                                                         "uncertainty": o.uncertainty})
_, o, _ = run(humans(99) + [{"id": " u5"}, {"id": "u5 "}, {"id": "U5"}])
R.check("A19 ids differing only by surrounding whitespace are not counted as separate members", o.value <= 100 and o.status == "pending",
        {"value": o.value, "status": o.status})
_, o, _ = run(humans(100, account_created_at=0.0), min_account_age_s=30 * DAY)
R.check("A20 with min_account_age_s set, accounts of unknown creation time are not counted as established",
        o.status != GoalStatus.ACHIEVED, {"status": o.status, "value": o.value})

# ---- B. signatures ---------------------------------------------------------------------------------------------
v, o, d = run(humans(100))
R.check("B1 a verifier-issued outcome checks", v.check_signature(o))
bad = []
for field, val in (("status", "failed"), ("value", 101.0), ("target", 1.0), ("uncertainty", 5.0), ("evaluated_at", 1.0),
                   ("notes", "x"), ("excluded", {"bots": 9}), ("evidence_refs", []), ("objective_id", "other"),
                   ("secondary", [])):
    f = copy.deepcopy(o)
    setattr(f, field, val)
    if v.check_signature(f):
        bad.append(field)
R.check("B2 changing any single field invalidates the signature", not bad, bad)
f = copy.deepcopy(o)
f.signature = ""
g = copy.deepcopy(o)
g.signature = "0" * 64
v_other, o_other, _ = run(humans(100))
R.check("B3 empty, constant and other-verifier signatures are rejected",
        not v.check_signature(f) and not v.check_signature(g) and not v.check_signature(o_other))
key = os.path.join(d, "verifier", "verifier.key")
R.check("B4 key file is 0600 in a 0700 directory", stat.S_IMODE(os.stat(key).st_mode) == 0o600
        and stat.S_IMODE(os.stat(os.path.dirname(key)).st_mode) == 0o700)
os.chmod(key, 0o644)
try:
    V.MembershipVerifier(Src([]), SimClock(), os.path.join(d, "verifier"), log_path=os.path.join(d, "verifier.jsonl"))
    refused = False
except PermissionError:
    refused = True
os.chmod(key, 0o600)
R.check("B5 a verifier refuses to start with a key other users can read", refused)

# the runtime: forged / unsigned / wrong-objective outcomes
class Forger:
    """What a learner-side impostor could hand the runtime: 'achieved' with a made-up signature."""

    def __init__(self, mode):
        self.mode = mode

    def verify(self, task):
        o = VerifiedOutcome(objective_id=task.objective_id if self.mode != "other_objective" else "x", status="achieved",
                            value=100.0, target=100.0, evaluated_at=0.0)
        o.signature = "" if self.mode == "unsigned" else "f" * 64
        return o

    def check_signature(self, outcome):
        return self.mode == "other_objective"       # even a "valid" one for another objective must be refused

    def measure(self, queries):
        return []


for mode in ("unsigned", "bad_signature", "other_objective"):
    L = make_slow_run("p06f", SimClock(), limits={"max_decisions": 3})
    L.rt.verifier = Forger(mode)
    s = L.rt.run(max_iterations=50)
    ev = read_log(L.run_dir, RunFiles.EVENTS)
    R.check(f"B6 runtime refuses an 'achieved' outcome that is {mode}: run not completed, nothing given to the learner",
            s != "completed" and not L.agent.verified and any(e["type"] == "verified_outcome_refused" for e in ev)
            and not any(e["type"] == "verified_outcome" for e in ev), {"status": s})

# ---- C. success criteria cannot be supplied by the caller -------------------------------------------------------
v, o, d = run(humans(5))
evil = task(target=1)
evil.success["exclusions"] = {"exclude_bots": False}
o2 = v.verify(evil)
R.check("C1 a caller's TaskSpec with target=1 does not change the judged target (pinned task)", o2.target == 100 and o2.status == "pending")
other = task()
other.objective_id = "another"
try:
    v.verify(other)
    refused = False
except PermissionError:
    refused = True
R.check("C2 a pinned verifier refuses another objective id", refused)

# ---- D. hash-chained log ---------------------------------------------------------------------------------------
v, o, d = run(humans(100))
for _ in range(4):
    v.verify(v.pinned_task)
log, head = os.path.join(d, "verifier.jsonl"), os.path.join(d, "verifier", "head.json")
ok0, n0, _ = V.HashChainLog.verify_chain(log, json.load(open(head)))
lines = open(log).read().splitlines()
R.check("D1 untouched log verifies against its head", ok0 and n0 == len(lines), n0)


def with_log(new_lines, use_head=True):
    p = os.path.join(scratch("p06l"), "v.jsonl")
    open(p, "w").write("\n".join(new_lines) + "\n")
    return V.HashChainLog.verify_chain(p, json.load(open(head)) if use_head else None)


e = json.loads(lines[2])
e["payload"]["outcome"]["value"] = 5.0
R.check("D2 editing an entry is detected", not with_log(lines[:2] + [json.dumps(e, sort_keys=True)] + lines[3:])[0])
R.check("D3 deleting a middle entry is detected", not with_log(lines[:2] + lines[3:])[0])
R.check("D4 deleting the last entry is detected with the recorded head", not with_log(lines[:-1])[0])
R.note("D4b deleting the last entry WITHOUT the head file is not detectable (unkeyed chain)", with_log(lines[:-1], use_head=False)[0])
# rebuild the whole chain after an edit (the hash is unkeyed): only the head file reveals it
rebuilt, prev = [], V.HashChainLog.GENESIS
for i, ln in enumerate(lines):
    e = json.loads(ln)
    body = {k: e[k] for k in ("seq", "ts", "kind", "payload")}
    if i == 2:
        body["payload"]["outcome"]["value"] = 5.0
    h = V.HashChainLog.entry_hash(prev, body)
    rebuilt.append(json.dumps({**body, "prev": prev, "hash": h}, sort_keys=True))
    prev = h
R.check("D5 a fully re-hashed forged log is detected with the recorded head", not with_log(rebuilt)[0])
# does the verifier process itself notice a mid-file edit when it restarts?
e = json.loads(lines[2])
e["payload"]["outcome"]["value"] = 5.0
open(log, "w").write("\n".join(lines[:2] + [json.dumps(e, sort_keys=True)] + lines[3:]) + "\n")
try:
    V.MembershipVerifier(Src(humans(100)), SimClock(), os.path.join(d, "verifier"), log_path=log, pinned_task=task())
    noticed = False
except RuntimeError:
    noticed = True
R.check("D6 a restarted verifier refuses to continue on a log edited before its last entry", noticed)

# ---- E. what the verifier writes about members -------------------------------------------------------------------
ids = [f"9{i:017d}" for i in range(100)]
v, o, d = run([{"id": i, "joined_at": 1.0} for i in ids], test_account_ids=ids[:2], suspect_member_ids=ids[2:4])
blob = open(os.path.join(d, "verifier.jsonl")).read()
R.check("E1 the signed outcome (what reaches the learner and the run logs) holds no member id", not any(i in o.to_json() for i in ids))
leaked = [i for i in ids if i in blob]
R.check("E2 the verifier log holds no raw member id", not leaked, {"raw ids in verifier.jsonl": len(leaked),
                                                                  "which": "test_account_ids / suspect_member_ids via detail.rules"})
sys.exit(R.finish())
