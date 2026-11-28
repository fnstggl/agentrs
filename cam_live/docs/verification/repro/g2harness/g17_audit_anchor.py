"""G2-harness probe g17 (round 4; root; real launcher, real service users): the out-of-band audit anchor of the R3-m1
fix (``<run>/logs/audit.anchor``, written by the launcher as root when the authorization service stops, passed to
its next start as ``--audit-anchor``).

  A1  the anchor exists after a stop and neither the learner uid nor the authorization uid can write, replace or
      delete it (or the logs directory);
  A2  a tail removed while the service is down -- the log truncated together with a matching head file, i.e. what
      the authorization uid itself could do -- is reported after the next start (verify_audit not ok);
  A3  control: an ordinary stop / relaunch verifies.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g17_audit_anchor.py
"""
import grp
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile

from _g2 import C, Report  # noqa: I001

R = Report("g17_audit_anchor")
if os.geteuid() != 0:
    R.note("not root: skipped")
    R.done()

from cam_live.harness.auth.client import connect  # noqa: E402
from cam_live.harness.auth.launcher import launch  # noqa: E402

FAST = {"separated": True, "poll_s": 0.05, "restart_backoff_s": 0.05, "stop_grace_s": 2.0}


def run_as(user, code, *args):
    pw = pwd.getpwnam(user)
    return subprocess.run([sys.executable, "-I", "-c", code, *map(str, args)], user=pw.pw_uid, group=pw.pw_gid,
                          extra_groups=[grp.getgrnam("camlive").gr_gid], cwd="/", env={"PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True, timeout=60)


def use(run, n=4):
    c = connect(run)
    t = C.TaskSpec(objective_id="obj-g17", objective="probe", mode="staging",
                   limits=C.Limits(duration_s=3600, max_decisions=100, max_external_actions=10))
    for _ in range(n):
        c.authorize(C.ActionSpec(action_id=C.new_id("a"), operation="wait", modality=C.Modality.INTERNAL,
                                 arguments={"seconds": 1}), t, {})


base = tempfile.mkdtemp(prefix="g2han", dir="/tmp")
os.chmod(base, 0o755)
run = os.path.join(base, "run")
try:
    h = launch(run, ["auth"], dict(FAST))
    try:
        use(run)
    finally:
        h.stop()
    anchor = os.path.join(run, "logs", "audit.anchor")
    R.check(os.path.exists(anchor), "the launcher keeps an anchor when the authorization service stops",
            open(anchor).read().strip()[:80] if os.path.exists(anchor) else "missing")
    attack = ("import os, sys\np = sys.argv[1]\nres = {}\n"
              "for name, fn in (('write', lambda: open(p, 'w').write('1:' + '0' * 64)), ('unlink', lambda: os.unlink(p)),"
              " ('replace', lambda: (open(os.path.join(os.path.dirname(p), '.x'), 'w'), os.replace(os.path.join("
              "os.path.dirname(p), '.x'), p)))):\n"
              "    try:\n        fn(); res[name] = 'ok'\n    except Exception as e:\n        res[name] = type(e).__name__\n"
              "print(res)\n")
    for user in ("camlearn", "camauth"):
        cp = run_as(user, attack, anchor)
        R.check("ok" not in cp.stdout, f"{user} cannot write, delete or replace the anchor", cp.stdout.strip()[:160])
    # A3 control
    h = launch(run, ["auth"], dict(FAST))
    try:
        v = connect(run).verify_audit()
        R.check(bool(v.get("ok")), "control: an ordinary stop / relaunch verifies", str(v.get("errors"))[:160])
        use(run, 3)
    finally:
        h.stop()
    # A2: remove the tail and rewrite the head to match (what the auth uid could do while the service is down)
    log = os.path.join(run, C.RunFiles.AUDIT)
    head = os.path.join(run, "auth", "audit.head")
    lines = [ln for ln in open(log, "rb").read().split(b"\n") if ln.strip()]
    keep = lines[:-3]
    fd = os.open(log, os.O_WRONLY | os.O_TRUNC)          # no O_CREAT: fs.protected_regular=2 in a sticky dir
    os.write(fd, b"\n".join(keep) + b"\n")
    os.close(fd)
    last = json.loads(keep[-1])
    from cam_live.harness.auth.audit import _mac
    key = open(os.path.join(run, "auth", "keys", "audit.key"), "rb").read()
    h_old = json.loads(open(head).read())
    mac = _mac(key, f"{last['seq']}:{last['hash']}")
    body = dict(h_old, seq=last["seq"], hash=last["hash"], mac=mac)
    body.pop("discontinuity", None)
    open(head, "w").write(json.dumps(body, sort_keys=True))
    h = launch(run, ["auth"], dict(FAST))
    try:
        v = connect(run).verify_audit()
        R.check(not v.get("ok"), "a tail removed while the service was down (log and head rewritten to match) is "
                                 "reported after the next start", f"ok={v.get('ok')} errors={str(v.get('errors'))[:200]}")
    finally:
        h.stop()
finally:
    shutil.rmtree(base, ignore_errors=True)
R.done()
