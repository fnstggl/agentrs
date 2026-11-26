"""Shared helpers for the EA tests."""
import json
import os
import subprocess
import sys

from cam_live import contracts as C
from cam_live.harness.auth import PolicyService, liveauth

LOOP = "http://127.0.0.1:8765"
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
ALLOW, DENY, NEEDS = C.AuthVerdict.ALLOW, C.AuthVerdict.DENY, C.AuthVerdict.NEEDS_APPROVAL
_MOD = {"browser": C.Modality.STRUCTURED_BROWSER, "visual": C.Modality.VISUAL, "sandbox": C.Modality.CODE,
        "code": C.Modality.CODE, "skill": C.Modality.SKILL, "wait": C.Modality.INTERNAL, "job": C.Modality.CODE,
        "file": C.Modality.CODE}


def act(op, modality=None, **kw):
    return C.ActionSpec(action_id=C.new_id("a"), operation=op,
                        modality=modality or _MOD.get(op.split(".")[0], C.Modality.API), **kw)


def prim(op, modality, **arguments):
    return C.PrimitiveOp(op=op, modality=modality, arguments=arguments)


def task(mode="staging", oid="obj-ea", **kw):
    kw.setdefault("objective", "a test objective")
    return C.TaskSpec(objective_id=oid, mode=mode, **kw)


def start(svc, action, t, context=None):
    """authorize + consume the grant, as runtime and executor would. Returns the decision."""
    dec = svc.authorize(action, t, context or {})
    assert dec.verdict == ALLOW, (dec.verdict, dec.reasons)
    assert svc.check_grant(action, dec) is True
    return dec


def ok_result(action, **cost):
    return C.ActionResult(action_id=action.action_id, status=C.ResultStatus.OK, cost=C.Cost(**cost))


LIVE_INTERFACES = ["structured_browser", "api", "code", "internal", "skill", "visual"]


def live_task(oid="obj-live", **kw):
    kw.setdefault("interfaces", list(LIVE_INTERFACES))
    kw.setdefault("limits", C.Limits(duration_s=3600.0, money=50.0, compute_s=1000.0, max_external_actions=100,
                                     max_decisions=200))
    return task(mode="live", oid=oid, **kw)


def install_operator_key(svc, tmp):
    """Create the operator key pair once per test directory and install the public half in the service."""
    import shutil
    priv, pub_src = os.path.join(str(tmp), "operator.key"), os.path.join(str(tmp), "operator.pub")
    if not os.path.exists(priv):
        liveauth.generate_operator_key(priv, pub_src)
    pub = os.path.join(svc.keys_dir, "operator.pub")
    if not os.path.exists(pub):
        shutil.copy(pub_src, pub)
    return priv


def live_record(t, now, **over):
    rec = {"record_id": "rec-1", "task_id": t.objective_id, "task_hash": t.content_hash(),
           "accounts": ["account:discord:bot-000001"], "owned": ["https://owned.example/*", "channel:discord:*"],
           "read": ["*.docs.example", "docs.example", "owned.example"], "preauthorized": [], "interactive": [],
           "opted_in_recipients": [],
           "budgets": {"money": 20.0, "money_per_action": 5.0, "money_per_day": 10.0, "compute_s": 500.0,
                       "max_external_actions": 50, "max_decisions": 100, "duration_s": 1800.0},
           "not_before": now - 10.0, "not_after": now + 3600.0, "issued_by": "the operator", "issued_at": now}
    rec.update(over)
    return rec


def authorize_live(svc, tmp, t, clock, **over):
    priv = install_operator_key(svc, tmp)
    rec = liveauth.sign_live_authorization(live_record(t, clock.now(), **over), priv)
    svc.register_live_authorization(rec, "operator")
    return rec


def run_as(user, code, *args, extra_groups=("camlive",), timeout=60):
    """Run python ``code`` as another unix user with a clean environment. Returns CompletedProcess."""
    import grp
    import pwd
    pw = pwd.getpwnam(user)
    groups = [grp.getgrnam(g).gr_gid for g in extra_groups]
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], user=pw.pw_uid, group=pw.pw_gid,
                          extra_groups=groups, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": SRC, "HOME": "/nonexistent"},
                          capture_output=True, text=True, timeout=timeout, cwd="/")


def jout(cp):
    assert cp.returncode == 0, cp.stderr[-2000:]
    return json.loads(cp.stdout.strip().splitlines()[-1])


def reopen(svc, clock):
    """A fresh service object on the same run directory: what a restart of the service process gives."""
    run_dir, path = svc.run_dir, svc.policy_path
    svc.close()
    return PolicyService(run_dir, path, clock=clock, fsync=False)
