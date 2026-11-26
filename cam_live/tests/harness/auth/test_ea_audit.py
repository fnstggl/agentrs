"""Hash-chained audit log: completeness and detection of edits, deletions, reordering, truncation, forgery."""
import json
import os

import pytest

from cam_live import contracts as C
from cam_live.harness.auth import AuditLog, PolicyService, verify_audit_log
from ea_helpers import LOOP, act, ok_result, reopen, start, task


def filled(tmp_path, n=12, key=b"k" * 32):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path, key=key, fsync=False)
    for i in range(n):
        log.append("event", 1000.0 + i, i=i, text=f"entry {i}")
    log.close()
    return path, key


def lines(path):
    with open(path, "rb") as f:
        return f.read().splitlines(keepends=True)


def write(path, ls):
    with open(path, "wb") as f:
        f.write(b"".join(ls))


def test_intact_log_verifies(tmp_path):
    path, key = filled(tmp_path)
    v = verify_audit_log(path, key=key)
    assert v.ok and v.entries == 12 and v.errors == [] and len(v.head_hash) == 64
    assert verify_audit_log(path).ok                       # the chain alone verifies without the key
    assert verify_audit_log(path, key=b"x" * 32).ok is False


@pytest.mark.parametrize("tamper", ["edit_field", "edit_keep_hash", "delete_middle", "delete_first", "swap", "duplicate",
                                    "insert_forged", "truncate_tail", "append_forged", "blank_out", "garbage"])
def test_tampering_is_detected(tmp_path, tamper):
    path, key = filled(tmp_path)
    ls = lines(path)
    if tamper == "edit_field":
        e = json.loads(ls[5]); e["data"]["text"] = "rewritten"; ls[5] = (json.dumps(e) + "\n").encode()
    elif tamper == "edit_keep_hash":                      # change the timestamp only
        e = json.loads(ls[3]); e["ts"] += 1; ls[3] = (json.dumps(e) + "\n").encode()
    elif tamper == "delete_middle":
        del ls[6]
    elif tamper == "delete_first":
        del ls[0]
    elif tamper == "swap":
        ls[4], ls[5] = ls[5], ls[4]
    elif tamper == "duplicate":
        ls.insert(7, ls[7])
    elif tamper == "insert_forged":                       # a self-consistent entry spliced in, chain re-linked after it
        from cam_live.harness.auth.audit import entry_hash
        prev = json.loads(ls[5])["hash"]
        h = entry_hash(prev, 7, 1.0, "authorize", {"verdict": "allow"})
        ls.insert(6, (json.dumps({"seq": 7, "ts": 1.0, "event": "authorize", "data": {"verdict": "allow"}, "prev": prev,
                                  "hash": h, "mac": "00"}) + "\n").encode())
    elif tamper == "truncate_tail":
        ls = ls[:-3]
    elif tamper == "append_forged":
        from cam_live.harness.auth.audit import entry_hash
        prev = json.loads(ls[-1])["hash"]
        h = entry_hash(prev, 13, 2.0, "approval_resolved", {"status": "approved"})
        ls.append((json.dumps({"seq": 13, "ts": 2.0, "event": "approval_resolved", "data": {"status": "approved"},
                               "prev": prev, "hash": h, "mac": ""}) + "\n").encode())
    elif tamper == "blank_out":
        ls[2] = b"\n"
    elif tamper == "garbage":
        ls[9] = b"{not json}\n"
    write(path, ls)
    v = verify_audit_log(path, key=key)
    assert not v.ok and v.errors, tamper


def test_chain_rebuilt_without_the_key_is_detected(tmp_path):
    """Someone who can rewrite the file but does not hold the key recomputes every hash: the MACs give it away,
    and so does an anchor (sequence number, hash) kept elsewhere."""
    from cam_live.harness.auth.audit import GENESIS, entry_hash
    path, key = filled(tmp_path)
    anchor = verify_audit_log(path, key=key)
    out, prev = [], GENESIS
    for i, raw in enumerate(lines(path)):
        e = json.loads(raw)
        if i == 4:
            e["data"]["text"] = "history, improved"
        e["prev"] = prev
        e["hash"] = prev = entry_hash(prev, e["seq"], e["ts"], e["event"], e["data"])
        out.append((json.dumps(e) + "\n").encode())
    write(path, out)
    os.unlink(path + ".head")
    assert not verify_audit_log(path, key=key).ok                                   # bad MACs
    assert verify_audit_log(path).ok                                                # the bare chain is consistent ...
    assert not verify_audit_log(path, expect_head=(anchor.entries, anchor.head_hash)).ok   # ... but not the anchored one
    assert not verify_audit_log(path, key=key, require_head=True).ok


def test_appending_continues_the_chain_across_instances(tmp_path):
    path, key = filled(tmp_path, n=3)
    a = AuditLog(path, key=key, fsync=False)
    b = AuditLog(path, key=key, fsync=False)             # e.g. a restarted service while the old handle lingers
    a.append("x", 1.0, n=1)
    b.append("y", 2.0, n=2)
    a.append("z", 3.0, n=3)
    assert a.head()["seq"] == 6
    v = verify_audit_log(path, key=key)
    assert v.ok and v.entries == 6
    a.close(); b.close()


def test_service_audits_every_kind_of_event(svc, clock):
    svc.put_secret("site_token", "tok_Zx81kQ0pLm4RvT7yWb2N", ["127.0.0.1:8765"])
    t = task(resources={"owned": ["channel:discord:900000000000000011"]}, limits=C.Limits(max_external_actions=2))
    a = act("http.get", arguments={"url": LOOP + "/a"})
    d = start(svc, a, t)                                                              # authorize + grant_use
    svc.check_primitive(C.PrimitiveOp("net.request", C.Modality.API, {"method": "GET", "url": LOOP + "/a"}), a)
    svc.secret_get("site_token", a, d, LOOP + "/a")                                   # secret access by name
    with pytest.raises(PermissionError):
        svc.secret_get("site_token", a, d, "http://localhost:1/")
    svc.record(a, ok_result(a))
    svc.check_grant(a, d)                                                             # replay -> grant_reject
    need = svc.authorize(act("discord.send_message", arguments={"channel_id": "900000000000000011", "content": "c" * 50}), t, {})
    svc.resolve_approval(need.approval_id, False, "olga", "no")                        # approval request + resolution
    b = act("http.get", arguments={"url": LOOP + "/b"})
    start(svc, b, t); svc.record(b, ok_result(b))
    svc.authorize(act("http.get", arguments={"url": LOOP + "/c"}), t, {})             # limit event
    svc.pause("check", "olga"); svc.resume("olga")
    svc.revoke("secret:site_token", "olga")
    svc.emergency_stop("drill", "olga"); svc.reset_stop("olga")
    svc.reload_policy("olga")
    events = {e["event"] for e in svc.audit_tail(200)}
    assert {"service_start", "task_pinned", "authorize", "grant_use", "grant_reject", "primitive", "secret_stored",
            "secret_access", "secret_denied", "record", "approval_requested", "approval_resolved", "limit_event",
            "pause", "resume", "revocation", "emergency_stop", "emergency_stop_reset", "policy_reload"} <= events
    v = svc.verify_audit()
    assert v.ok and v.entries == svc.audit_head()["seq"]
    raw = open(svc.audit.path).read()
    assert "tok_Zx81kQ0pLm4RvT7yWb2N" not in raw and "site_token" in raw             # names, never values
    auth_entries = [e for e in svc.audit_tail(200) if e["event"] == "authorize"]
    assert all(e["data"]["reasons"] and e["data"]["policy_ids"] and e["data"]["action_hash"] for e in auth_entries)


def test_log_survives_restart_and_detects_edits_made_while_down(svc, clock):
    t = task()
    for i in range(5):
        svc.authorize(act("wait", arguments={"seconds": i + 1}), t, {})
    path, key = svc.audit.path, svc.audit.key
    s2 = reopen(svc, clock)
    s2.authorize(act("wait"), t, {})
    assert s2.verify_audit().ok
    n = s2.audit_head()["seq"]
    s2.close()
    ls = lines(path)
    e = json.loads(ls[3]); e["data"]["verdict"] = "allow-ish"; ls[3] = (json.dumps(e) + "\n").encode()
    write(path, ls)
    s3 = PolicyService(os.path.dirname(path), clock=clock, fsync=False)
    v = s3.verify_audit()
    assert not v.ok and any("line 4" in x for x in v.errors) and v.entries == n + 1   # +1: s3's own start entry
    s3.close()
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"
