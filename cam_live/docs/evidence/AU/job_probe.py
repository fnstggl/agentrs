"""Probe: a background job with network='scoped' under the standing profile (real PolicyService)."""
import os, sys, shutil, tempfile, time, uuid, json
sys.path.insert(0, "/workspace/camlive/cam_live/tests/harness/autonomy"); sys.path.insert(0, "/workspace/camlive/cam_live/tests/harness/auth"); sys.path.insert(0, "/workspace/camlive/cam_live/tests/harness/env")
from cam_live import contracts as C
from cam_live.harness import sandbox as sbx
from cam_live.harness.auth import PolicyService
from cam_live.harness.env import ComputerEnvironment
from au_support import PackageIndex, build_wheel, live_task, standing_record, sign_register, PROFILE
import cam_live.harness.jobs as J; print("jobs from", J.__file__)
idx = PackageIndex([build_wheel("autinypkg", "0.1", "VALUE = 42\n")])
base = tempfile.mkdtemp(prefix="camlive-au-", dir="/tmp"); os.chmod(base, 0o755)
sb = f"/var/tmp/cam-live-sbx-au-{uuid.uuid4().hex[:8]}"; os.makedirs(sb); os.chmod(sb, 0o711)
run = os.path.join(base, "run"); os.makedirs(run)
svc = PolicyService(run, fsync=False); svc.register_local_service(idx.host, by="t")
env = ComputerEnvironment(run, svc, secrets=svc.secrets_handle(), provenance="staging")
env.register(sbx.SandboxExecutor(workspace=os.path.join(run, "sandbox"), cfg=sbx.SandboxConfig(base_dir=sb, allow_private_destinations=True)))
t = live_task(oid="obj-job", resources={})
sign_register(svc, base, standing_record(t, time.time(), read=[idx.host], standing={"profile": PROFILE, "code_network": [idx.host]}))
code = f"import urllib.request; print(urllib.request.urlopen('http://{idx.host}/simple/autinypkg/', timeout=10).status)"
a = C.ActionSpec(action_id=C.new_id("act"), operation="sandbox.job_start", modality=C.Modality.CODE,
                 arguments={"purpose": "fetch the index", "kind": "python", "code": code, "deadline_s": 60, "network": "scoped", "hosts": [idx.host]})
d = svc.authorize(a, t, {}); print("authorize", d.verdict, d.reasons)
res = env.execute(a, d); print("job_start", res.status, res.failure_detail[:300])
jid = (res.artifacts or {}).get("job")
for _ in range(100):
    s = C.ActionSpec(action_id=C.new_id("act"), operation="sandbox.job_result", modality=C.Modality.CODE, arguments={"job_id": jid or ""})
    r = env.execute(s, svc.authorize(s, t, {}))
    data = r.observations[0].data if r.observations else {"status_of_call": r.status, "detail": r.failure_detail}
    rec = data.get("job") if isinstance(data.get("job"), dict) else data
    if rec.get("state") in ("done", "failed", "cancelled", "timeout", "succeeded", "ok") : data = rec; break
    last = (r.status, sorted(data)[:12], r.failure_detail[:200])
    time.sleep(0.2)
print("last", last); print("result", json.dumps({k: data.get(k) for k in ("state", "exit_code", "stdout_tail", "status_of_call", "detail")}))
print("index requests", idx.requests)
env.close(); svc.close(); idx.close(); shutil.rmtree(base, True); shutil.rmtree(sb, True)
