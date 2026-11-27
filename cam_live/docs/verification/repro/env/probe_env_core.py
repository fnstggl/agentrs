"""G-env probe 1: the environment contract (authorization path, at-most-once, restart, RPC uid).

    cd /workspace/camlive && .venv/bin/python cam_live/docs/verification/repro/env/probe_env_core.py

Real components: ComputerEnvironment (EB), PolicyService with the default policy (EA), InternalExecutor (EB),
HttpApiExecutor (ES), the ExecRPC service process. Destination: a local HTTP server that counts requests.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time

from g_common import C, CAM, Drv, HERE, HostileServer, action, brief, real_policy, report, scratch, staging_task, summary

from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.failures import RetryPolicy
from cam_live.harness.http_api import ApiModality, HttpApiExecutor
from cam_live.harness.internal import InternalExecutor

srv = HostileServer()
srv.routes["/write"] = lambda m, p, q, h, b: (200, {"Content-Type": "application/json"}, json.dumps({"ok": True, "n": srv.count("/write")}))
srv.routes["/fast"] = srv.routes["/write"]


def slow(m, p, q, h, b):
    time.sleep(8)
    return 200, {"Content-Type": "application/json"}, "{}"


srv.routes["/slow"] = slow


def build(run_dir, task, **kw):
    pol = real_policy(run_dir, task)
    env = ComputerEnvironment(run_dir, pol, secrets=pol.secrets_handle(), **kw)
    env.register(InternalExecutor(run_dir, clock=env.clock))
    env.register(ApiModality([HttpApiExecutor(retry=RetryPolicy(rules={}))]))
    return env, pol


def post(aid="", path="/write"):
    return action("http.request", C.Modality.API, aid=aid, method="POST", url=srv.url + path, json={"x": 1})


# ---- 1. nothing reaches an executor without a verified, unused grant -------------------------------------------
run = scratch("core")
task = staging_task("g_env_core")
env, pol = build(run, task)
d = Drv(env, pol, task)

a = post()
r = env.execute(a, None)
ok1 = r.status == "denied" and srv.count("/write") == 0
a = post()
r2 = env.execute(a, C.AuthorizationDecision(C.AuthVerdict.ALLOW, a.content_hash(), ["forged"], grant="AAAA.BBBB"))
ok2 = r2.status == "denied" and srv.count("/write") == 0
a, b = post(), post()
dec_a = pol.authorize(a, task, {})
r3 = env.execute(b, dec_a)                                   # a's grant presented for b
ok3 = r3.status == "denied" and srv.count("/write") == 0
dec_a2 = C.AuthorizationDecision.from_dict(dec_a.to_dict()); dec_a2.action_hash = b.content_hash()
r4 = env.execute(b, dec_a2)                                  # a's grant relabelled as b's
ok4 = r4.status == "denied" and srv.count("/write") == 0
na = C.AuthorizationDecision(C.AuthVerdict.NEEDS_APPROVAL, a.content_hash(), ["fake"], approval_id="apr_x")
r5 = env.execute(post(), na)
ok5 = r5.status == "needs_approval" and srv.count("/write") == 0
report("ENV-AUTH-1", "HOLDS" if all((ok1, ok2, ok3, ok4, ok5)) else "DEFECT",
       "no / forged / foreign / relabelled grant and needs_approval never reach the executor",
       results=[brief(x) for x in (r, r2, r3, r4, r5)], requests_sent=srv.count("/write"))

# the genuine grant: executes once; the same grant is refused by a second environment (single use)
r6 = env.execute(a, dec_a)
n_after = srv.count("/write")
run_b = scratch("core_b")
env_b = ComputerEnvironment(run_b, pol)
env_b.register(ApiModality([HttpApiExecutor(retry=RetryPolicy(rules={}))]))
r7 = env_b.execute(a, dec_a)
report("ENV-AUTH-2", "HOLDS" if (r6.status == "ok" and n_after == 1 and r7.status == "denied" and srv.count("/write") == 1) else "DEFECT",
       "a real grant executes once; replaying it through a second environment (fresh journal) is refused by the "
       "authorization service", first=brief(r6), replay=brief(r7), requests_sent=srv.count("/write"))
env_b.close()

# ---- 2. order: submitted -> (grant) -> started -> primitive_check -> primitive_op -> result ---------------------
recs = [x for x in env.records("", 500, a.action_id)]
kinds = [x["type"] for x in recs]
want = ["env.action_submitted", "env.action_started", "env.primitive_check", "env.primitive_op", "env.action_result"]
pos = [kinds.index(k) if k in kinds else -1 for k in want]
audit = [e for e in pol.audit_tail(400) if a.content_hash() in json.dumps(e, default=str)]
aud_kinds = [e.get("event") or e.get("type") or e.get("kind") for e in audit]
report("ENV-ORDER", "HOLDS" if (pos == sorted(pos) and -1 not in pos) else "DEFECT",
       "execution log order for one action; authorization-service audit has decision, grant_use, primitive, record",
       env_log=kinds, audit_events=aud_kinds)

# ---- 3. at most once ------------------------------------------------------------------------------------------
before = srv.count("/write")
dup = env.execute(a, pol.authorize(a, task, {}))             # same action id + content, fresh authorization
same_id = action("http.request", C.Modality.API, aid=a.action_id, method="POST", url=srv.url + "/write", json={"x": 2})
other = env.execute(same_id, pol.authorize(same_id, task, {}))
report("ENV-ONCE-1", "HOLDS" if (dup.status == "ok" and srv.count("/write") == before and other.status == "failed") else "DEFECT",
       "re-issued action id returns the recorded result without a second request; the id with different content is refused",
       duplicate=brief(dup), reused_id=brief(other), requests_before=before, requests_after=srv.count("/write"),
       duplicates_suppressed=env.get_resource_state()["duplicates_suppressed"])


# concurrent duplicate submission with two valid grants while the second caller's grant check is slow
class SlowSecondGrantCheck:
    """The real PolicyService; only adds latency to the FIRST check_grant call (as a slow AuthRPC round trip would)."""

    def __init__(self, inner, delay):
        self._inner, self._delay, self._n, self._lock = inner, delay, 0, threading.Lock()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def check_grant(self, action, decision, *a, **k):
        with self._lock:
            self._n += 1
            first = self._n == 1
        if first:
            time.sleep(self._delay)
        return self._inner.check_grant(action, decision, *a, **k)


run_c = scratch("core_race")
task_c = staging_task("g_env_race")
pol_c = real_policy(run_c, task_c)
slowpol = SlowSecondGrantCheck(pol_c, 1.5)
env_c = ComputerEnvironment(run_c, slowpol)
env_c.register(ApiModality([HttpApiExecutor(retry=RetryPolicy(rules={}))]))
srv.routes["/race"] = lambda m, p, q, h, b: (200, {"Content-Type": "application/json"}, "{}")
ra = action("http.request", C.Modality.API, method="POST", url=srv.url + "/race", json={"create": "one thing"})
d1, d2 = pol_c.authorize(ra, task_c, {}), pol_c.authorize(ra, task_c, {})
out = {}
t2 = threading.Thread(target=lambda: out.__setitem__("slow", env_c.execute(ra, d1)))
t1 = threading.Thread(target=lambda: out.__setitem__("fast", env_c.execute(ra, d2)))
t2.start(); time.sleep(0.2); t1.start(); t1.join(); t2.join()
n_race = srv.count("/race", "POST")
report("ENV-ONCE-2", "DEFECT" if n_race > 1 else "HOLDS",
       "two concurrent submissions of ONE action id, each with its own valid grant (second authorization verdict: "
       f"{d2.verdict}); the external POST was sent {n_race} time(s)",
       fast=brief(out["fast"]), slow=brief(out["slow"]), posts_received=n_race,
       external_actions=env_c.get_resource_state()["external_actions"])
env_c.close()

# ---- 4. emergency stop ---------------------------------------------------------------------------------------
x = post()
dx = pol.authorize(x, task, {})
pol.emergency_stop("g-env probe", "operator")
before = srv.count("/write")
rs = env.execute(x, dx)
report("ENV-STOP", "HOLDS" if rs.status == "denied" and srv.count("/write") == before else "DEFECT",
       "a grant issued before the emergency stop is refused after it", result=brief(rs))
env.close()

# ---- 5. crash / restart of the environment service process ----------------------------------------------------
run_s = scratch("svc")
# AF_UNIX paths are limited to ~107 bytes and other uids must be able to reach the socket for the uid test, so the
# socket alone lives in a short world-traversable directory (removed at the end); all state stays in the scratch dir
import atexit, shutil, tempfile  # noqa: E401,E402
sockdir = tempfile.mkdtemp(prefix="genv")
os.chmod(sockdir, 0o755)
atexit.register(shutil.rmtree, sockdir, True)
sock = os.path.join(sockdir, "exec.sock")
envv = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(CAM, "src"), HERE]))


def start():
    p = subprocess.Popen([sys.executable, "-m", "cam_live.harness.service", "--run-dir", run_s, "--socket", sock,
                          "--diagnostic-allow-all", "--no-browser", "--executor", "g_factories:http_api"],
                         env=envv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    line = p.stdout.readline()
    assert '"ready": true' in line, line
    return p


from cam_live.diagnostics.world import AllowAllPolicy  # noqa: E402
from cam_live.harness.service import RemoteEnvironment  # noqa: E402

allow = AllowAllPolicy()
tk = staging_task("g_env_svc")
p = start()
rem = RemoteEnvironment(sock)
w = action("wait", C.Modality.INTERNAL, seconds=5000, reason="g-env restart probe")
rw = rem.execute(w, allow.authorize(w, tk))
wake_before = rem.get_resource_state()["next_wake_at"]
ev_before, cur_before = rem.poll_events("0")
fastw = action("http.request", C.Modality.API, method="POST", url=srv.url + "/fast", json={"k": 1})
rf = rem.execute(fastw, allow.authorize(fastw, tk))
sloww = action("http.request", C.Modality.API, method="POST", url=srv.url + "/slow", json={"k": 2}, timeout=30)
box = {}
th = threading.Thread(target=lambda: box.__setitem__("r", _try(lambda: rem.execute(sloww, allow.authorize(sloww, tk)))))


def _try(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return e


th.start()
t0 = time.time()
while srv.count("/slow") == 0 and time.time() - t0 < 10:
    time.sleep(0.05)
os.kill(p.pid, signal.SIGKILL)                     # the service dies while the write is in flight
p.wait(); th.join()
p2 = start()
rem2 = RemoteEnvironment(sock)
again = rem2.execute(sloww, allow.authorize(sloww, tk))       # the interrupted write: must not be sent again
dupfast = rem2.execute(fastw, allow.authorize(fastw, tk))     # the finished write: recorded result
wake_after = rem2.get_resource_state()["next_wake_at"]
stale_cursor, cur_after = rem2.poll_events(cur_before)
ok = (again.status == "failed" and "interrupted" in again.failure_detail and srv.count("/slow") == 1
      and dupfast.status == "ok" and srv.count("/fast") == 1 and wake_after == wake_before)
report("ENV-RESTART", "HOLDS" if ok else "DEFECT",
       "SIGKILL during an external write, restart on the same run directory: the interrupted write is reported "
       "failed/unknown and not re-sent, the finished write returns its recorded result, the scheduled wake survives",
       in_flight_call=repr(box.get("r"))[:120], resubmitted=brief(again), slow_posts=srv.count("/slow"),
       fast_posts=srv.count("/fast"), wake_before=wake_before, wake_after=wake_after, cursor=cur_after)

# ---- 6. the RPC service refuses a caller with the wrong uid ---------------------------------------------------
client = r'''
import json, socket, struct, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(5)
try:
    s.connect(sys.argv[1])
except OSError as e:
    print("CONNECT-ERROR", type(e).__name__, e); sys.exit(0)
for m in ("__ping__", "ping", "get_capabilities"):
    d = json.dumps({"id": 1, "method": m, "args": [], "kwargs": {}}).encode()
    s.sendall(struct.pack(">I", len(d)) + d)
    n = struct.unpack(">I", s.recv(4))[0]; buf = b""
    while len(buf) < n: buf += s.recv(n - len(buf))
    r = json.loads(buf)
    print(m, "OK" if r.get("ok") else "REFUSED " + r["error"]["kind"] + ": " + r["error"]["message"][:80])
'''
if os.geteuid() == 0:
    import pwd
    st = os.stat(sock)
    out = {}
    for user in ("camsbx", "camverify", "nobody", "camlearn"):
        try:
            pw = pwd.getpwnam(user)
        except KeyError:
            continue
        cp = subprocess.run(["setpriv", f"--reuid={pw.pw_uid}", f"--regid={pw.pw_gid}", "--init-groups",
                             "/usr/bin/python3", "-c", client, sock], capture_output=True, text=True)
        out[user] = (cp.stdout.strip() or cp.stderr.strip()[-200:]).replace("\n", " | ")
    bad = [u for u in ("camsbx", "camverify", "nobody") if u in out and " OK" in (" " + out[u])]
    good = "camlearn" in out and "REFUSED" not in out["camlearn"] and "ERROR" not in out["camlearn"]
    report("RPC-UID", "HOLDS" if not bad and good else "DEFECT",
           "ExecRPC socket: only the learner uid (and root) may call; other service uids / nobody are refused",
           socket_mode=oct(st.st_mode & 0o777), socket_gid=st.st_gid, replies=out)
else:
    report("RPC-UID", "NOTE", "not root: the uid check could not be exercised")
try:
    rem2.shutdown()
except Exception:  # noqa: BLE001
    pass
p2.wait(timeout=10)
srv.close()
sys.exit(summary("probe_env_core"))
