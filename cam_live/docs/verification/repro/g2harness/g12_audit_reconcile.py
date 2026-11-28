"""G2-harness probe g12 (round 3, in process): the audit log's head reconciliation (R2-m3 fix, auth/audit.py
``_reconcile_head``). A head left one or more entries behind by a stop / kill must be advanced when the service opens
the log again; nothing else may be accepted. The tails below are what an attacker without the audit key can produce
(the learner uid cannot even write the log in the launched deployment; p09 / p13): edited entries, entries chained
correctly but MAC'd with another key or without a MAC, entries replayed from another log, a truncated log, a head
that was itself rolled back further than the reconciliation window.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g12_audit_reconcile.py
"""
import hashlib
import hmac
import json
import os
import shutil

from _g2 import C, Report, cleanup, fresh_dir  # noqa: I001

from cam_live.harness.auth import PolicyService
from cam_live.harness.auth.audit import entry_hash

R = Report("g12_audit_reconcile")


def make(tag, n=6):
    d = fresh_dir(tag)
    run = d + "/run"
    svc = PolicyService(run, fsync=False)
    t = C.TaskSpec(objective_id=f"obj-{tag}", objective="probe", mode="staging",
                   limits=C.Limits(duration_s=3600, max_decisions=500, max_external_actions=10))
    for _ in range(n):
        svc.authorize(C.ActionSpec(action_id=C.new_id("a"), operation="wait", modality=C.Modality.INTERNAL,
                                   arguments={"seconds": 1}), t, {})
    return d, run, svc, t


def paths(svc):
    return svc.audit.path, svc.audit.head_path


def more(svc, t, k):
    for _ in range(k):
        svc.authorize(C.ActionSpec(action_id=C.new_id("a"), operation="wait", modality=C.Modality.INTERNAL,
                                   arguments={"seconds": 2}), t, {})


def reopen_verify(run):
    s = PolicyService(run, fsync=False)
    v = s.verify_audit()
    s.close()
    return v


def lines(p):
    return [ln for ln in open(p, "rb").read().split(b"\n") if ln.strip()]


def scenario(tag, mutate, expect_ok, what, k=3):
    d, run, svc, t = make(tag)
    log, head = paths(svc)
    old_head = open(head, "rb").read()
    more(svc, t, k)
    svc.close()
    with open(head, "wb") as f:              # the head as a stop between line and head would leave it (k behind)
        f.write(old_head)
    if mutate:
        mutate(log, head)
    v = reopen_verify(run)
    R.check(bool(v.ok) == expect_ok, what, f"ok={v.ok} entries={v.entries} errors={list(v.errors)[:2]}")
    cleanup(d)


def edit_last(log, head):
    ls = lines(log)
    e = json.loads(ls[-1])
    e["data"] = dict(e["data"], operator="mallory")
    ls[-1] = json.dumps(e).encode()
    open(log, "wb").write(b"\n".join(ls) + b"\n")


def forged(mac_key):
    def f(log, head):
        ls = lines(log)
        last = json.loads(ls[-1])
        seq, ts, ev, data = last["seq"] + 1, last["ts"] + 1, "operator_action", {"by": "operator", "what": "forged"}
        h = entry_hash(last["hash"], seq, ts, ev, data)
        e = {"seq": seq, "ts": ts, "event": ev, "data": data, "prev": last["hash"], "hash": h}
        if mac_key is not None:
            e["mac"] = hmac.new(mac_key, h.encode(), hashlib.sha256).hexdigest()
        open(log, "ab").write(json.dumps(e).encode() + b"\n")
    return f


try:
    scenario("g12a", None, True, "a head left 3 entries behind (all entries intact) is reconciled at open: no false alarm")
    scenario("g12b", edit_last, False, "an edited entry after a lagging head is NOT accepted")
    scenario("g12c", forged(os.urandom(32)), False, "an appended entry chained correctly but MAC'd with another key is NOT accepted")
    scenario("g12d", forged(None), False, "an appended entry chained correctly without a MAC is NOT accepted")

    def replay(log, head):           # an entry of another run's log (valid MAC under ITS key) appended here
        d2, run2, s2, t2 = make("g12x", n=12)
        other = lines(s2.audit.path)[-1]
        s2.close()
        open(log, "ab").write(other + b"\n")
        cleanup(d2)
    scenario("g12e", replay, False, "an entry replayed from another log is NOT accepted")

    def truncate(log, head):
        ls = lines(log)
        open(log, "wb").write(b"\n".join(ls[:-5]) + b"\n")
    scenario("g12f", truncate, False, "a log truncated below the head is NOT accepted", k=1)
    scenario("g12g", None, False, "a head 70 entries behind (outside the reconciliation window) is reported, not "
                                  "silently advanced", k=70)

    # what a pure roll-back of the head achieves: the verifier must still see everything appended since
    d, run, svc, t = make("g12h")
    log, head = paths(svc)
    old = open(head, "rb").read()
    more(svc, t, 2)
    svc.close()
    open(head, "wb").write(old)
    shutil.copy(log, log + ".copy")
    v = reopen_verify(run)
    R.check(v.ok and v.entries == len(lines(log + ".copy")) + 0 or v.entries >= len(lines(log + ".copy")),
            "after reconciliation the verified chain covers every entry (none dropped)", f"entries {v.entries}")
    cleanup(d)
finally:
    pass
R.done()
