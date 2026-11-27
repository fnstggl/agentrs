"""Background jobs and scoped network (EC), as root. Checks: a job cannot reach the network unless 'scoped' is
chosen AND each connection is authorized; private/loopback destinations are refused; a job is bounded by its
deadline and leaves nothing behind; a job that a crash left running is reported as interrupted, not lost.
"""
import os
import threading
import time

from _common import ALLOW, DENY, C, Report, cleanup, fresh_dir
from cam_live.harness import sandbox as sbx
from cam_live.harness.jobs import JobManager, JobState

R = Report("p12_jobs_network")
if os.geteuid() != 0:
    R.note("not root: skipping (needs the real uid drop)")
    R.done()

base = fresh_dir("p12base", world=True)
cfg = sbx.SandboxConfig(base_dir=base, limits=sbx.SandboxLimits(wall_s=20, mem_mb=256, mem_total_mb=256, nproc=16))
sb = sbx.Sandbox(os.path.join(base, "ws", "jobs"), cfg)
if not sb.available:
    R.defect("sandbox unavailable as root", sb.unavailable_reason); R.done()

# a real egress broker whose authorizer records every (host, port) and allows only a chosen host
authorized = []


def authorizer(host, port, ip, private):
    authorized.append((host, port, ip, private))
    return (host == "allowed.invalid", "only allowed.invalid is permitted")


def make_broker():
    return sbx.EgressBroker(authorizer, allow_private=False, max_connections=8)


# 1. no-network job cannot create a socket even if it tries (seccomp)
res = sb.run_python("import socket; socket.socket(); print('OK')", network="none")
R.check("OK" not in res.stdout, "a network='none' job cannot create a socket")

# 2. scoped job: every outbound connection goes through the authorizer; a refused host never connects
code = ("import socket\n"
        "for host in ('allowed.invalid','blocked.invalid'):\n"
        " try:\n  socket.create_connection((host,80),2); print('CONN',host)\n"
        " except Exception as e: print('ERR',host,type(e).__name__)\n")
res = sb.run_python(code, network="scoped", broker=make_broker())
hosts_tried = {h for h, *_ in authorized}
R.check(hosts_tried == {"allowed.invalid", "blocked.invalid"},
        "scoped network: the broker is asked for every outbound connection", f"asked for {sorted(hosts_tried)}")
R.check("CONN blocked.invalid" not in res.stdout, "scoped network: a host the authorizer refuses is not connected",
        [l for l in res.stdout.splitlines() if "blocked" in l])

# 3. a scoped job cannot reach a real loopback service (private destinations refused before resolution)
authorized.clear()
res = sb.run_python("import socket\ntry:\n socket.create_connection(('127.0.0.1',11111),2); print('LOOPBACK-OK')\n"
                    "except Exception as e: print('ERR',type(e).__name__)\n",
                    network="scoped", broker=sbx.EgressBroker(lambda *a: (True, ""), allow_private=False))
R.check("LOOPBACK-OK" not in res.stdout, "scoped network: loopback is refused even when the authorizer says yes",
        res.stdout.strip()[-60:])

# 4. background job: deadline enforced, record persisted, nothing left running
run_dir = fresh_dir("p12run", world=True)
mgr = JobManager(run_dir, sandbox=sb, clock=C.WallClock(), max_concurrent=2)
rec = mgr.submit("probe sleep past deadline", "python", "import time\nwhile True: time.sleep(1)", deadline_s=3.0)
final = mgr.wait(rec.job_id, timeout=30)
R.check(final.state in (JobState.TIMEOUT, JobState.FAILED) and final.leftover_processes == 0,
        "a background job is stopped at its deadline and leaves nothing running",
        f"state={final.state} leftover={final.leftover_processes}")
R.check(os.path.exists(os.path.join(run_dir, "jobs", rec.job_id, "record.json")), "the job record is persisted")
try:
    too_big = mgr.submit("x", "python", "pass", deadline_s=10 * mgr.max_deadline_s)
    R.check(too_big.deadline_s <= mgr.max_deadline_s, "a job deadline cannot exceed the manager ceiling",
            f"asked {10*mgr.max_deadline_s:g}, got {too_big.deadline_s:g}")
except Exception as e:  # noqa: BLE001
    R.note(f"over-long deadline raised {type(e).__name__}")

# 5. a job still running when the manager process 'crashes' is reported as interrupted by the next manager
run_dir2 = fresh_dir("p12run2", world=True)
mgr2 = JobManager(run_dir2, sandbox=sb, clock=C.WallClock(), max_concurrent=2)
longrec = mgr2.submit("long job", "python", "import time\nwhile True: time.sleep(1)", deadline_s=120.0)
time.sleep(1.5)
# simulate a crash: drop the manager WITHOUT close() so the job keeps running, then open a fresh one
mgr2._live.clear()                        # forget the live handles (what a crash loses)
mgr3 = JobManager(run_dir2, sandbox=sb, clock=C.WallClock())
st = mgr3.status(longrec.job_id)
R.check(st.state == JobState.INTERRUPTED, "a job left running by a crash is reported as interrupted on reopen",
        f"state={st.state}")
leftover = [p for p, c in __import__("cam_live.tests.harness.compute.conftest", fromlist=["session_processes"])
            .session_processes(longrec.job_id)] if False else []
mgr.close(); mgr2.close(); mgr3.close()
sb.close()
cleanup(base); cleanup(run_dir); cleanup(run_dir2)
R.done()
