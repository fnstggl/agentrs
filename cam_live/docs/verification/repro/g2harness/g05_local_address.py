"""G2-harness probe g05: the local-address rule (instance control plane never reachable, EA fix of G-sec M4 /
G-env C1 / M10). Part 1 tries further spellings of this machine's addresses against the real PolicyService (live
mode, the public web readable as tools/launch_live.sh documents). Part 2 checks that the rule knows this
instance's public address ($PUBLIC_IPADDR, named in localnet.py) also in the environment the launcher gives the
authorization service (auth/launcher.py ENV_PASSTHROUGH). Nothing is connected to; only decisions are taken.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g05_local_address.py
"""
import json
import os
import subprocess
import sys

from _g2 import DENY, SRC, C, Report, cleanup, dec_str, fresh_dir  # noqa: I001

from cam_live.harness.auth import PolicyService
from cam_live.harness.auth.launcher import ENV_PASSTHROUGH
from ea_helpers import act, authorize_live, live_task

R = Report("g05_local_address")
d = fresh_dir("g05")
clock = C.SimClock()
svc = PolicyService(d + "/run", clock=clock, fsync=False)
t = live_task(oid="obj-g05")
authorize_live(svc, d, t, clock, owned=["https://owned.example/*"], read=["https://*", "http://*"])
try:
    spellings = ["http://127.0.0.1:11111/", "http://127.0.0.1.:11111/", "http://[::ffff:127.0.0.1]:11111/",
                 "http://[::ffff:7f00:1]:11111/", "http://[0:0:0:0:0:ffff:7f00:0001]:11111/",
                 "http://[::ffff:0:7f00:1]:11111/", "http://[64:ff9b::7f00:1]:11111/", "http://017700000001:11111/",
                 "http://0x7f.0.0.1:11111/", "http://127.000.000.001:11111/", "http://%31%32%37.0.0.1:11111/",
                 "http://ｌｏｃａｌｈｏｓｔ:11111/", "http://localhost.localdomain:11111/", "http://ip6-localhost:11111/",
                 "http://[fe80::1%25lo]:11111/", "http://169.254.169.254/latest/meta-data/",
                 "http://user:pw@127.0.0.1:11111/", "http://127.0.0.1:11111\\@owned.example/",
                 "http://0:8080/", "http://[::]:8080/", "http://127.255.255.254:8080/"]
    host = os.uname().nodename
    spellings.append(f"http://{host}:8080/")
    for u in spellings:
        dec = svc.authorize(act("http.get", C.Modality.API, arguments={"url": u}), t, {})
        R.check(dec.verdict == DENY, f"this machine under the spelling {u!r} is refused", dec_str(dec))
    pub = os.environ.get("PUBLIC_IPADDR", "")
    if pub:
        dec = svc.authorize(act("http.get", C.Modality.API, arguments={"url": f"http://{pub}:36807/"}), t, {})
        R.check(dec.verdict == DENY, "in-process: this instance's public address ($PUBLIC_IPADDR) is refused",
                dec_str(dec))
        env = {k: os.environ[k] for k in ENV_PASSTHROUGH if k in os.environ}
        env["PYTHONPATH"] = SRC
        code = ("import json,sys; from cam_live.harness.auth import localnet as L; "
                "print(json.dumps({'kind': L.local_kind(sys.argv[1]), 'self': sorted(L.self_addresses())[:6]}))")
        out = subprocess.run([sys.executable, "-c", code, pub], env=env, capture_output=True, text=True, timeout=60)
        got = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else {"error": out.stderr[-300:]}
        R.check(bool(got.get("kind")),
                "with the launcher's environment (ENV_PASSTHROUGH) the authorization service still recognises the "
                "instance's public address as this machine", json.dumps(got))
    else:
        R.note("PUBLIC_IPADDR is not set here: part 2 skipped")
finally:
    svc.close()
    cleanup(d)
R.done()
