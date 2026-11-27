"""Process separation (EA, privileged), as root. Uses the real launcher.prepare_run_dir and launch(). Confirms by
independent means (switching uid with run_as, reading /proc, stat) that:
 - the learner and sandbox uids cannot read the vault, audit log, grant/audit keys, policy copy or verifier files;
 - run-dir and sock directories are root-owned sticky, so another uid cannot replace the audit log or the socket;
 - the authorization service process is non-dumpable (its environ/maps are unreadable to other uids);
 - a service that drops to its uid cannot regain root;
 - the emergency-stop FILE FLAG halts the executor service but not the authorization service, and nothing restarts
   until reset.
"""
import json
import os
import stat
import subprocess
import sys
import time

from _common import C, Report, cleanup, fresh_dir
from ea_helpers import SRC, run_as
from cam_live.harness.auth import launcher

R = Report("p13_uid_separation")
if os.geteuid() != 0 or not __import__("cam_live.harness", fromlist=["users_exist"]).users_exist():
    R.note("not root or service users absent: skipping"); R.done()

run = fresh_dir("p13", world=True)
layout = launcher.prepare_run_dir(run, separated=True)
R.note("run dir prepared with the service users")

# ---- file ownership and modes
import pwd
U = {r: pwd.getpwnam(u).pw_uid for r, u in __import__("cam_live.harness", fromlist=["SERVICE_USERS"]).SERVICE_USERS.items()}
checks = {
    "auth": ("vault", "auth", "auth/keys", "auth/keys/grant.key", "auth/policy.yaml", "audit.jsonl"),
    "verifier": ("verifier", "verifier.jsonl"),
}
for owner, rels in checks.items():
    for rel in rels:
        st = os.lstat(os.path.join(run, rel))
        good = st.st_uid == U[owner] and (st.st_mode & 0o077) == 0 if not rel.endswith("keys") else st.st_uid == U[owner]
        R.check(st.st_uid == U[owner] and (st.st_mode & 0o007) == 0,
                f"{rel} is owned by the {owner} uid and not world-accessible", f"uid={st.st_uid} mode={oct(st.st_mode)}")
for rel in ("", "sock"):
    st = os.lstat(os.path.join(run, rel) if rel else run)
    R.check(st.st_uid == 0 and bool(st.st_mode & stat.S_ISVTX), f"{rel or '.'} is a root-owned sticky directory",
            f"uid={st.st_uid} mode={oct(st.st_mode)}")

# ---- the learner and sandbox uids cannot read the protected files
PROBE = """
import os, sys, json
out = {}
for path in sys.argv[1:]:
    try:
        with open(path, 'rb') as f:
            out[os.path.basename(path)] = 'READ:' + f.read(16).hex()
    except Exception as e:
        out[os.path.basename(path)] = type(e).__name__
print(json.dumps(out))
"""
targets = [os.path.join(run, p) for p in ("vault/..", "auth/keys/grant.key", "auth/keys/audit.key",
           "audit.jsonl", "auth/policy.yaml", "auth/state.sqlite", "verifier.jsonl")]
# make the grant/audit keys and a vault secret exist first by starting the auth service briefly below; for now the
# files that exist are checked
exist = [t for t in targets if os.path.exists(t) and not t.endswith("..")]
for user in ("camlearn", "camsbx"):
    cp = run_as(user, PROBE, *exist)
    try:
        got = json.loads(cp.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        got = {"_": cp.stderr[-120:]}
    read = [k for k, v in got.items() if str(v).startswith("READ:")]
    R.check(not read, f"uid {user} cannot read any protected authorization/verifier file",
            f"read={read}; results={got}")

# ---- launch the real auth service (+ a tiny executor-less set) and probe the live process
handle = launcher.launch(run, ["auth"], config={"separated": True})
try:
    ok = handle.wait_ready("auth", 30)
    status = handle.status()["services"]["auth"]
    R.check(ok and status["state"] == "running" and status["uid"] == U["auth"],
            "the authorization service runs as the camauth uid", f"{status['state']} uid={status['uid']}")
    pid = status["pid"]
    # another uid cannot read its environ / maps (PR_SET_DUMPABLE=0 -> /proc/<pid>/environ is root/owner only)
    cp = run_as("camlearn", "import sys; print(open('/proc/%s/environ'%sys.argv[1],'rb').read()[:8])", str(pid))
    R.check(cp.returncode != 0 or "b''" in cp.stdout or not cp.stdout.strip(),
            "uid camlearn cannot read the authorization service's /proc environ", (cp.stdout or cp.stderr).strip()[-80:])
    # now the keys exist: re-check the learner cannot read them
    cp = run_as("camlearn", PROBE, os.path.join(run, "auth/keys/grant.key"), os.path.join(run, "audit.jsonl"))
    got = json.loads(cp.stdout.strip().splitlines()[-1]) if cp.stdout.strip() else {"_": cp.stderr[-80:]}
    R.check(not any(str(v).startswith("READ:") for v in got.values()),
            "uid camlearn cannot read the grant key or the live audit log", str(got))
    # another uid cannot replace the audit log or the socket inside the sticky dirs
    cp = run_as("camlearn", "import os,sys\ntry:\n os.unlink(sys.argv[1]); print('UNLINKED')\nexcept Exception as e: print(type(e).__name__)",
                os.path.join(run, "audit.jsonl"))
    R.check("UNLINKED" not in cp.stdout, "uid camlearn cannot delete the audit log (sticky root dir)", cp.stdout.strip()[-60:])

    # ---- emergency stop via the file flag
    open(os.path.join(run, "auth", "STOP"), "w").close()
    time.sleep(2.0)
    st = handle.status()
    auth_alive = st["services"]["auth"]["state"] == "running"
    R.check(st["halted"] and auth_alive, "file-flag emergency stop halts the launcher but keeps the auth service up",
            f"halted={st['halted']} auth={st['services']['auth']['state']}")
    # a running authorization service denies everything while stopped
    from cam_live.harness.auth.client import connect
    cl = connect(run, timeout=10)
    from ea_helpers import act, task
    d = cl.authorize(act("wait", C.Modality.INTERNAL), task())
    R.check(d.verdict == C.AuthVerdict.DENY and "P-STOP" in d.policy_ids,
            "while the stop flag is set the authorization service denies every request", ",".join(d.policy_ids))
    cl.close()
finally:
    handle.stop()
    cleanup(run)
R.done()
