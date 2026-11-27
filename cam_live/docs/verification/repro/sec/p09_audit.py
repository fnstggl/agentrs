"""Audit log: hash chain + MAC + head file. Edits, deletion, insertion, reordering, truncation, full rebuild.
"""
import json
import os
import shutil

from _common import C, Report, cleanup, fresh_dir
from ea_helpers import act, task
from cam_live.harness.auth import PolicyService, verify_audit_log
from cam_live.harness.auth.audit import GENESIS, entry_hash

R = Report("p09_audit")
d = fresh_dir("p09")
svc = PolicyService(d, clock=C.SimClock(), fsync=False)
t = task()
for i in range(12):
    svc.authorize(act("wait", C.Modality.INTERNAL), t)
svc.authorize(act("teleport.x", C.Modality.API), t)
key, path, head = svc.audit.key, svc.audit.path, svc.audit.head_path
svc.close()
orig, orig_head = open(path, "rb").read(), open(head, "rb").read()
lines = orig.split(b"\n")[:-1]
R.check(bool(verify_audit_log(path, key=key, head_path=head)), "untouched log verifies", f"{len(lines)} entries")


def tampered(what, new_lines=None, new_head=None, *, key_=key, head_=head, expect_detect=True, **kw):
    open(path, "wb").write(b"\n".join(new_lines if new_lines is not None else lines) + b"\n")
    if new_head is None:
        open(head, "wb").write(orig_head)
    elif new_head == "remove":
        os.unlink(head)
    else:
        open(head, "wb").write(new_head)
    res = verify_audit_log(path, key=key_, head_path=head_, **kw)
    ok = (not res.ok) if expect_detect else res.ok
    R.check(ok, f"{what} is detected" if expect_detect else what, (res.errors[0] if res.errors else "verify ok")[:110])
    return res


e = json.loads(lines[5]); e["data"]["verdict"] = "deny"
tampered("an edited entry", lines[:5] + [json.dumps(e, sort_keys=True, separators=(",", ":")).encode()] + lines[6:])
tampered("a deleted entry", lines[:5] + lines[6:])
tampered("two swapped entries", lines[:5] + [lines[6], lines[5]] + lines[7:])
tampered("a duplicated entry", lines[:6] + [lines[5]] + lines[6:])
tampered("a truncated tail (head file left)", lines[:-3])
tampered("a truncated tail with the head file removed", lines[:-3], "remove")
tampered("a truncated tail, verifier given no key", lines[:-3], key_=None)

# rebuild the whole chain without the MAC key
prev, rebuilt = GENESIS, []
for i, raw in enumerate(lines[:-1], 1):
    e = json.loads(raw)
    if e["event"] == "authorize":
        e["data"]["verdict"] = "deny"
    h = entry_hash(prev, i, e["ts"], e["event"], e["data"])
    e.update(seq=i, prev=prev, hash=h, mac="")
    rebuilt.append(json.dumps(e, sort_keys=True, separators=(",", ":")).encode())
    prev = h
fake_head = json.dumps({"seq": len(rebuilt), "hash": prev, "mac": ""}).encode()
tampered("a chain rebuilt without the key (verifier has the key)", rebuilt, fake_head)
r = tampered("a chain rebuilt without the key passes a verifier that has NO key (expected: keyless verification is "
             "only chain consistency)", rebuilt, fake_head, key_=None, expect_detect=False)

# rollback: an older log together with the older head file both carry valid MACs
d2 = fresh_dir("p09b")
svc = PolicyService(d2, clock=C.SimClock(), fsync=False)
for i in range(5):
    svc.authorize(act("wait", C.Modality.INTERNAL), t)
p2, h2, k2 = svc.audit.path, svc.audit.head_path, svc.audit.key
old_log, old_head = open(p2, "rb").read(), open(h2, "rb").read()
old = svc.audit_head()
for i in range(5):
    svc.authorize(act("teleport.x", C.Modality.API), t)
svc.close()
open(p2, "wb").write(old_log); open(h2, "wb").write(old_head)
res = verify_audit_log(p2, key=k2, head_path=h2)
res_anchor = verify_audit_log(p2, key=k2, head_path=h2, expect_head=(old["seq"] + 5, "f" * 64))
R.note(f"rollback of log + head to an earlier consistent pair: verify_audit_log ok={res.ok} (needs write access as "
       f"the authorization uid; SECURITY.md section 3 says so); with an external anchor ok={res_anchor.ok}")
R.check(not res_anchor.ok, "rollback is detected when an external anchor (expect_head) is supplied")

# emptied log + removed head, default arguments
open(p2, "wb").close(); os.unlink(h2)
res = verify_audit_log(p2, key=k2)
R.check(not res.ok, "an emptied log with the head file removed is detected by verify_audit_log(path, key)",
        f"ok={res.ok} entries={res.entries}")
cleanup(d); cleanup(d2)
R.done()
