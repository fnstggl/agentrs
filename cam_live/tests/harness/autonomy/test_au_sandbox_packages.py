"""AU (privileged): sandboxed software development with package installation from a registry under the standing
profile -- generated code runs as the sandbox uid (Landlock, seccomp, no sockets of its own), reaches only the
hosts its run declared through the egress broker, and every connection is judged by the real authorization
service. The registry is a local PEP 503 index (no external contact); it is registered as a local service and
named in the signed record exactly as a public registry (pypi.org) would be."""
import json
import os
import shutil
import tempfile
import time
import uuid

import pytest

from cam_live import contracts as C
from cam_live.harness import sandbox as sbx
from cam_live.harness.auth import PolicyService
from cam_live.harness.env import ComputerEnvironment

from au_support import (ALLOW, NEEDS, PROFILE, PackageIndex, build_wheel, live_task, sign_register,
                        standing_record)

pytestmark = pytest.mark.privileged

INSTALL = ("python -m pip install --no-cache-dir --disable-pip-version-check --no-input --quiet "
           "--index-url http://{host}/simple --target lib autinypkg && "
           "python -c \"import sys; sys.path.insert(0, 'lib'); import autinypkg; print('VALUE', autinypkg.VALUE)\"")


@pytest.fixture
def world(tmp_path):
    idx = PackageIndex([build_wheel("autinypkg", "0.1", "VALUE = 42\n")])
    other = PackageIndex([])                               # a second local service the program did not declare
    base = tempfile.mkdtemp(prefix="camlive-au-", dir="/tmp")
    os.chmod(base, 0o755)
    sbx_base = f"/var/tmp/cam-live-sbx-au-{uuid.uuid4().hex[:10]}"
    os.makedirs(sbx_base)
    os.chmod(sbx_base, 0o711)
    run_dir = os.path.join(base, "run")
    os.makedirs(run_dir)
    svc = PolicyService(run_dir, fsync=False)
    for h in (idx.host, other.host):
        svc.register_local_service(h, by="test", note="local package index")
    env = ComputerEnvironment(run_dir, svc, secrets=svc.secrets_handle(), provenance="staging")
    ex = sbx.SandboxExecutor(workspace=os.path.join(run_dir, "sandbox"),
                             cfg=sbx.SandboxConfig(base_dir=sbx_base, allow_private_destinations=True))
    if not ex.sandbox.available:
        pytest.skip(ex.sandbox.unavailable_reason)
    env.register(ex)
    t = live_task(oid="obj-au-pkg", resources={})
    w = {"idx": idx, "other": other, "svc": svc, "env": env, "task": t, "run_dir": run_dir, "tmp": tmp_path}
    yield w
    env.close()
    svc.close()
    idx.close()
    other.close()
    shutil.rmtree(base, ignore_errors=True)
    shutil.rmtree(sbx_base, ignore_errors=True)


def record(w, standing=True):
    t, host = w["task"], w["idx"].host
    rec = standing_record(t, time.time(), read=[host, w["other"].host],
                          standing={"profile": PROFILE, "code_network": [host]})
    if not standing:
        rec.pop("standing")
    return sign_register(w["svc"], w["tmp"], rec)


def shell(cmd, hosts, timeout=180.0):
    return C.ActionSpec(action_id=C.new_id("act"), operation="sandbox.shell", modality=C.Modality.CODE,
                        arguments={"command": cmd, "network": "scoped", "hosts": hosts, "timeout_s": timeout},
                        timeout_s=timeout)


def test_generated_code_installs_a_package_from_the_registry_without_a_per_job_approval(world):
    w = world
    record(w)
    a = shell(INSTALL.format(host=w["idx"].host), [w["idx"].host])
    d = w["svc"].authorize(a, w["task"], {})
    assert d.verdict == ALLOW, d.reasons                   # standing profile: no approval for this host
    res = w["env"].execute(a, d)
    out = res.observations[0].data if res.observations else {}
    assert res.status == C.ResultStatus.OK, (res.failure_detail, out.get("stderr", "")[-1500:])
    assert "VALUE 42" in out["stdout"], out
    assert any(p.startswith("/simple/autinypkg") for p in w["idx"].requests), w["idx"].requests
    assert any(p.startswith("/files/autinypkg-0.1-py3-none-any.whl") for p in w["idx"].requests)
    conns = [o for o in res.ops if o.op == "net.connect"]
    assert conns and all(o.status != C.ResultStatus.DENIED for o in conns), [(o.arguments, o.note) for o in conns]
    import pwd
    iso = out["isolation"]                                 # what the job's first process reported about itself
    assert iso["uid"] == pwd.getpwnam("camsbx").pw_uid and iso["landlock_abi"] >= 1 and iso["seccomp"], iso
    # the package landed in the sandbox workspace and nowhere else
    assert os.path.isdir(os.path.join(w["run_dir"], "sandbox", "lib", "autinypkg"))
    audit = open(os.path.join(w["run_dir"], C.RunFiles.AUDIT)).read()
    assert '"net.connect"' in audit or "net.connect" in audit


def test_a_connection_to_a_host_the_run_did_not_declare_is_refused(world):
    w = world
    record(w)
    code = (f"python - <<'PY'\nimport urllib.request\n"
            f"print('declared', urllib.request.urlopen('http://{w['idx'].host}/simple/autinypkg/', timeout=10).status)\n"
            f"try:\n    urllib.request.urlopen('http://{w['other'].host}/simple/x/', timeout=10)\n    print('REACHED')\n"
            f"except Exception as e:\n    print('refused', type(e).__name__, e)\nPY")
    a = shell(code, [w["idx"].host])
    d = w["svc"].authorize(a, w["task"], {})
    assert d.verdict == ALLOW, d.reasons
    res = w["env"].execute(a, d)
    out = res.observations[0].data
    assert "declared 200" in out["stdout"] and "REACHED" not in out["stdout"], out
    assert "refused" in out["stdout"] and "not authorized" in out["stdout"], out["stdout"]
    assert w["other"].requests == []
    denied = [o for o in res.ops if o.op == "net.connect" and o.status == C.ResultStatus.DENIED]
    assert denied and "declared" in denied[0].note, [(o.arguments, o.note) for o in res.ops]


def test_without_the_standing_profile_the_job_waits_for_the_operator_and_nothing_runs(world):
    w = world
    record(w, standing=False)
    a = shell(INSTALL.format(host=w["idx"].host), [w["idx"].host])
    d = w["svc"].authorize(a, w["task"], {})
    assert d.verdict == NEEDS and "P-CODE-NETWORK" in d.policy_ids, (d.verdict, d.reasons)
    assert w["idx"].requests == []
    # the operator approves: the same job now runs
    w["svc"].resolve_approval(d.approval_id, True, "operator")
    d2 = w["svc"].authorize(a, w["task"], {})
    assert d2.verdict == ALLOW
    res = w["env"].execute(a, d2)
    assert res.status == C.ResultStatus.OK and "VALUE 42" in res.observations[0].data["stdout"]
