"""Limits: money, external actions, decisions, duration, rate; "a limit nobody set is a denial" in live mode.

DEFECT probed: an action is charged as ONE external action, ONE decision and ONE rate event per destination no
matter how many steps it carries (policy.py _authorize: `want = {... "external_actions": 1.0 if external ...}`;
_rate: one key per (task, destination) per action). A program of N write steps to one destination therefore
multiplies the write-rate limit and the external-action budget by N (N <= structure.max_nodes - 1 = 127, each step
is one primitive, limits.max_primitives_per_action = 500).
"""
import os

from _common import ALLOW, DENY, C, Report, cleanup, fresh_dir
from ea_helpers import act, authorize_live, live_task, ok_result, prim, start, task
from cam_live.harness.auth import PolicyService, is_resource_denial

R = Report("p03_limits")
API = C.Modality.API
SITE = "http://127.0.0.1:8765"


def post(i, body="x"):
    return act("http.post", API, arguments={"url": f"{SITE}/items/{i}", "json": {"text": f"{body} {i}"}})


# ---------------------------------------------------------------- A. plain limits fail closed (staging)
d = fresh_dir("p03a")
clock = C.SimClock()
svc = PolicyService(d, clock=clock, fsync=False)
t = task(limits=C.Limits(money=10.0, max_external_actions=3, max_decisions=6, duration_s=100.0))
for i in range(3):
    a = post(i)
    start(svc, a, t)
    svc.record(a, ok_result(a))
v = svc.authorize(post(99), t)
R.check(v.verdict == DENY and "L-EXTERNAL-ACTIONS" in v.policy_ids and is_resource_denial(v),
        "external-action budget is enforced", ",".join(v.policy_ids))
for i in range(3):
    svc.authorize(act("wait", C.Modality.INTERNAL), t)
v = svc.authorize(act("wait", C.Modality.INTERNAL), t)
R.check(v.verdict == DENY and "L-DECISIONS" in v.policy_ids, "decision budget is enforced", ",".join(v.policy_ids))
t2 = task(oid="obj-money", limits=C.Limits(money=10.0, max_external_actions=100, max_decisions=100, duration_s=100.0))
a = act("payment.send", API, arguments={"payee": f"{SITE}/pay", "amount": 6.0})
start(svc, a, t2)
svc.record(a, ok_result(a, money=6.0))
v = svc.authorize(act("payment.send", API, arguments={"payee": f"{SITE}/pay", "amount": 6.0}), t2)
R.check(v.verdict == DENY and "L-MONEY-TOTAL" in v.policy_ids, "total money limit is enforced", ",".join(v.policy_ids))
big = C.TaskSpec.from_dict(dict(t2.to_dict(), limits=C.Limits(money=1e9, max_external_actions=10**6,
                                                              max_decisions=10**6, duration_s=1e9).to_dict()))
v = svc.authorize(act("wait", C.Modality.INTERNAL), big)
R.check(v.verdict == DENY and "P-TASK-MISMATCH" in v.policy_ids, "the learner cannot raise limits by editing the task")
clock.advance(101.0)
v = svc.authorize(act("wait", C.Modality.INTERNAL), t2)
R.check(v.verdict == DENY and "L-DURATION" in v.policy_ids, "duration limit is enforced", ",".join(v.policy_ids))
svc.close()
cleanup(d)

# ---------------------------------------------------------------- B. live mode: unset limits are denials
d = fresh_dir("p03b")
clock = C.SimClock()
svc = PolicyService(d, clock=clock, fsync=False)
lt = live_task(oid="obj-unset", limits=C.Limits())
authorize_live(svc, d, lt, clock, budgets={})
v = svc.authorize(act("wait", C.Modality.INTERNAL), lt)
R.check(v.verdict == DENY and "L-UNSET" in v.policy_ids, "live mode: a limit nobody set is a denial",
        ",".join(v.policy_ids))
svc.close()
cleanup(d)

# ---------------------------------------------------------------- C. rate limit, single actions
d = fresh_dir("p03c")
clock = C.SimClock()
svc = PolicyService(d, clock=clock, fsync=False)
t = task(oid="obj-rate", limits=C.Limits(max_external_actions=10, max_decisions=1000, duration_s=1e6))
write_cap = svc.policy.rate_write.count
n = 0
while True:
    v = svc.authorize(post(n, "single"), task(oid="obj-rate-single"))
    if v.verdict != ALLOW:
        break
    n += 1
R.check(n == write_cap and "L-RATE" in v.policy_ids, f"write rate limit stops single actions at {write_cap} per window",
        f"{n} allowed, then {v.policy_ids}")

# ---------------------------------------------------------------- D. the same writes batched as steps
clock.advance(svc.policy.rate_write.window_s + 1)
N = 100
allowed_actions = allowed_writes = primitives_ok = 0
for k in range(10):
    steps = [post(f"{k}-{i}", "batched") for i in range(N)]
    prog = act("http.post", API, level=C.Level.PROCEDURE, arguments={"url": f"{SITE}/items/batch{k}", "json": {"text": f"b{k}"}},
               steps=steps)
    v = svc.authorize(prog, t)
    if v.verdict != ALLOW:
        break
    allowed_actions += 1
    assert svc.check_grant(prog, v)
    for s in steps:                                   # what the executor does: one net.request primitive per step
        pv = svc.check_primitive(prim("net.request", API, method="POST", url=s.arguments["url"]), prog)
        if pv.verdict == ALLOW:
            primitives_ok += 1
    allowed_writes += N + 1
    svc.record(prog, C.ActionResult(action_id=prog.action_id, status=C.ResultStatus.OK))
used = svc.limits_state()["tasks"]["obj-rate"]["used"]
R.check(primitives_ok <= write_cap,
        f"write rate limit ({write_cap} per {svc.policy.rate_write.window_s:g} s per destination) bounds batched writes",
        f"{allowed_actions} actions carried {allowed_writes} write nodes; {primitives_ok} write primitives were "
        f"ALLOWED inside one window")
R.check(used["external_actions"] >= primitives_ok or primitives_ok <= 10,
        "max_external_actions=10 bounds the number of external writes",
        f"ledger external_actions={used['external_actions']:g}, decisions={used['decisions']:g} for "
        f"{primitives_ok} executed external write primitives")
svc.close()
cleanup(d)
R.done()
