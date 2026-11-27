"""G-rt probe 09 — preflight and doctor: live readiness is never produced by staging / diagnostic components, every
missing item is listed, doctor reports what is really on this machine.

    .venv/bin/python cam_live/docs/verification/repro/rt/p09_preflight.py
"""
import json
import os
import shutil
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import Report, SlowEnv, scratch  # noqa: E402

os.environ.setdefault("CAM_LIVE_GPU_LOCK", os.path.join(scratch("lock"), "gpu.lock"))   # never touch another project's lock
from cam_live.config import PROJECT_ROOT, CamLiveConfig  # noqa: E402
from cam_live.contracts import SimClock, WallClock  # noqa: E402
from cam_live.diagnostics.world import AllowAllPolicy, LabWorld  # noqa: E402
from cam_live.harness import users_exist  # noqa: E402
from cam_live.runtime import doctor as DOC  # noqa: E402
from cam_live.runtime import preflight as PF  # noqa: E402
from cam_live.runtime import tasks as T  # noqa: E402

R = Report("p09_preflight")
tmp = scratch("p09")
DISCORD = os.path.join(PROJECT_ROOT, "tasks", "discord_100.yaml")
cfg = CamLiveConfig()
cfg.home = os.path.join(tmp, "home")


def filled(mode="live"):
    raw = T.load_task_raw(DISCORD)
    raw["mode"] = mode
    raw["success"]["evidence_source"] = "g_fake_module:live_source"
    raw["success"]["exclusions"]["test_account_ids"] = []
    raw["resources"].update(discord_guild_id="G1", discord_bot_credential_name="discord_bot", owned_site="https://example.invalid",
                            operator_authorization_record="rec-1", authorized_tools=["api"], authorized_destinations=["G1"],
                            currency="USD")
    raw["resources"]["publishing"].update(authorized=True, destinations=["G1"])
    raw["limits"] = {"duration_s": 86400, "money": 10, "compute_s": 3600, "max_external_actions": 100, "max_decisions": 100}
    raw["data_permissions"]["collect_feedback_with_consent"] = False
    raw["interfaces"] = ["api", "internal"]
    p = os.path.join(tmp, f"filled_{mode}.yaml")
    yaml.safe_dump(raw, open(p, "w"))
    return p


# ---- 1. the Discord task as shipped ----
t0 = time.time()
rep = PF.preflight(DISCORD, cfg)
items = rep["live"]["missing"]
unset = T.unset_paths(T.load_task_raw(DISCORD))
named = " | ".join(f"{m['item']}: {m['detail']}" for m in items)
R.check("1a discord_100.yaml as shipped: LIVE blocked", rep["live"]["ready"] is False and rep["live"]["blocked"] is True, len(items))
not_listed = [u for u in unset if not any(u == m["item"] or u.startswith(m["item"] + ".") or u.startswith(m["item"] + "[") for m in items)]
R.check("1b every one of the 17 unset REQUIRED items is named in the report", len(unset) == 17 and not not_listed, not_listed)
kinds = sorted({m["kind"] for m in items})
R.check("1c the report also names the authorization service, the environment, the verifier and system items",
        {"authorization", "integration", "verification", "system"} <= set(kinds), kinds)

# ---- 2. filled live task, components of every kind ----
live = filled("live")
tid = "discord-100"


class LivePolicyStub:
    """Answers like an authorization service that holds a valid live record (what a preflight handle would say)."""

    def limits_state(self):
        return {"policy": {"valid": True}, "stopped": False, "tasks": {tid: {"mode": "live"}}}

    def secret_names(self):
        return [{"name": "discord_bot"}]


class DiagnosticInDisguise(AllowAllPolicy, LivePolicyStub):
    """A subclass of the diagnostic allow-all policy that answers limits_state like a live service."""
    limits_state = LivePolicyStub.limits_state


live_env = SlowEnv(WallClock(), caps={"live": True, "provenance": "live", "name": "g-live-labelled"})
live_env.discover_tools = lambda: [t for t in SlowEnv.discover_tools(live_env)]
vinfo_live = {"objective_id": tid, "source": "g", "source_live": True}


def ready(**kw):
    kw.setdefault("with_doctor", False)
    return PF.preflight(live, cfg, **kw)["live"]


base = dict(policy=LivePolicyStub(), env=live_env, verifier_info=vinfo_live)
r0 = ready(**base)
R.note("2a control: with handles that all *say* live and the doctor switched off, the logic reports",
       {"ready": r0["ready"], "missing": [m["item"] for m in r0["missing"]], "root_and_service_users": os.geteuid() == 0 and users_exist()})
cases = {
    "2b diagnostic AllowAllPolicy": dict(base, policy=AllowAllPolicy()),
    "2c LabWorld as the environment": dict(base, env=LabWorld()),
    "2d environment with staging provenance": dict(base, env=SlowEnv(WallClock(), caps={"live": False, "provenance": "staging"})),
    "2e environment that says live but runs on simulated time": dict(base, env=SlowEnv(SimClock(), caps={"live": True, "provenance": "live", "simulated_time": True})),
    "2f verifier whose evidence source is not live (JSONL file)": dict(base, verifier_info={"objective_id": tid, "source": "jsonl:x", "source_live": False}),
    "2g verifier serving another objective": dict(base, verifier_info={"objective_id": "other", "source_live": True}),
    "2h no authorization service": dict(base, policy=None),
    "2i no environment": dict(base, env=None),
    "2j no verifier": dict(base, verifier_info=None),
    "2k authorization service holding the task in staging mode": dict(base, policy=type("P", (LivePolicyStub,), {
        "limits_state": lambda self: {"policy": {"valid": True}, "tasks": {tid: {"mode": "staging"}}}})()),
    "2l authorization service without a live record": dict(base, policy=type("P", (LivePolicyStub,), {
        "limits_state": lambda self: {"policy": {"valid": True}, "tasks": {tid: {"unavailable": ["no operator authorization record for this live task"]}}}})()),
    "2m emergency stop engaged": dict(base, policy=type("P", (LivePolicyStub,), {
        "limits_state": lambda self: {"policy": {"valid": True}, "stopped": True, "tasks": {tid: {"mode": "live"}}}})()),
    "2n bot credential not in the vault": dict(base, policy=type("P", (LivePolicyStub,), {"secret_names": lambda self: []})()),
}
for name, kw in cases.items():
    r = ready(**kw)
    R.check(f"{name}: LIVE blocked with the item named", r["ready"] is False, [f"{m['item']}: {m['detail'][:70]}" for m in r["missing"]][:2])
r = ready(**dict(base, policy=DiagnosticInDisguise()))
R.check("2o a subclass of the diagnostic AllowAllPolicy is recognised as diagnostic (class-name test)", r["ready"] is False,
        {"ready": r["ready"]})
staged = SlowEnv(WallClock(), caps={"live": True, "provenance": "live", "name": "labelled-live",
                                    "executors": {"api": {"discord": {"environment": "staging", "base_url": "http://127.0.0.1:8099"}}}})
r = ready(**dict(base, env=staged))
R.check("2p an environment labelled live whose Discord executor reports the local staging service is not counted as a live integration",
        r["ready"] is False, {"ready": r["ready"], "note": "preflight reads only the top-level live/provenance label"})
# with the doctor on, this machine cannot be live-ready whatever the handles say
r = ready(**dict(base, with_doctor=True))
R.check("2q with the doctor on, this machine is not live-ready (missing learner module / storage / models are listed)",
        r["ready"] is False, [m["item"] for m in r["missing"]][:8])

# ---- 3. a staging / diagnostic task is never live-ready ----
for mode in ("staging", "diagnostic"):
    rep = PF.preflight(filled(mode), cfg, with_doctor=False, **base)
    R.check(f"3 a task in mode {mode} is never live-ready, even with live-looking handles", rep["live"]["ready"] is False)

# ---- 4. the real authorization service (in-process instance) with a live task and no / a forged record ----
try:
    from cam_live.harness.auth.policy import PolicyService
    pol = PolicyService(os.path.join(tmp, "auth_run"))
    task = T.load_task(live, for_run=True)
    pol.register_task(task, "operator")
    r = ready(**dict(base, policy=pol))
    why = [m for m in r["missing"] if m["kind"] in ("authorization", "credential")]
    R.check("4a real PolicyService, live task registered without an authorization record: blocked and named",
            r["ready"] is False and any("authorization record" in (m["item"] + m["detail"]) for m in why),
            [f"{m['item']}: {m['detail'][:80]}" for m in why])
    forged = {"record_id": "live-x", "task_id": task.objective_id, "task_hash": task.content_hash(), "accounts": [], "owned": ["*"],
              "read": ["*"], "preauthorized": [], "budgets": {k: 1 for k in ("money", "money_per_action", "money_per_day",
                                                                               "compute_s", "max_external_actions",
                                                                               "max_decisions", "duration_s")},
              "not_before": 0.0, "not_after": 4e9, "issued_by": "learner", "issued_at": 0.0, "signature": "AAAA"}
    try:
        pol.register_live_authorization(forged, "operator")
        accepted = True
    except Exception as e:  # noqa: BLE001
        accepted, err = False, f"{type(e).__name__}: {e}"[:120]
    R.check("4b a forged (unsigned / wrongly signed) authorization record is refused by the service", not accepted,
            "" if accepted else err)
    pol.close()
except Exception as e:  # noqa: BLE001
    R.note("4 real PolicyService part could not run", f"{type(e).__name__}: {e}"[:300])

# ---- 5. doctor ----
t0 = time.time()
d = DOC.run_doctor(cfg)
dt = time.time() - t0
flat = {c["name"]: c for s in d["sections"].values() for c in s}
R.note("5a doctor wall time in-process (status file: 3.6 s through the CLI)", round(dt, 2))
import importlib.util  # noqa: E402
truth = {m: importlib.util.find_spec(m) is not None for m in ("cam_live.agent", "cam_live.validation")}
R.check("5b doctor reports the absent modules as absent (cam_live.agent, cam_live.validation) and blocks staging on them",
        all(flat[f"module:{m}"]["ok"] is truth[m] for m in truth)
        and all((f"module:{m}" in d["summary"]["blocking_staging"]) is (not truth[m]) for m in truth),
        {m: flat[f"module:{m}"]["ok"] for m in truth})
du = shutil.disk_usage(tmp)
R.check("5c disk figure matches shutil.disk_usage", f"{du.free / 2**30:.0f} GB free" in flat["disk"]["detail"], flat["disk"]["detail"][:80])
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
    R.check("5d GPU name matches nvidia-smi", smi in flat["gpu"]["detail"], {"doctor": flat["gpu"]["detail"][:60], "nvidia-smi": smi})
except Exception as e:  # noqa: BLE001
    R.note("5d nvidia-smi not available", str(e)[:100])
inst = json.load(open("/etc/vast_capabilities.json")).get("instance", {}) if os.path.exists("/etc/vast_capabilities.json") else {}
R.check("5e persistent storage: reported False when the instance has no volume (and then blocks live)",
        flat["persistent_storage"]["ok"] is (bool(inst.get("workspace_is_volume")) and cfg.resolved_home().startswith("/workspace/"))
        and ("persistent_storage" in d["summary"]["blocking_live"]) is (flat["persistent_storage"]["ok"] is not True),
        {"doctor": flat["persistent_storage"]["ok"], "manifest.workspace_is_volume": inst.get("workspace_is_volume")})
R.check("5f an undetermined check (ok=None) is reported as unknown and blocks, never as a pass",
        all(n in d["summary"]["unknown"] for n, c in flat.items() if c["ok"] is None)
        and all(n in d["summary"]["blocking_live"] for n, c in flat.items() if c["ok"] is None and c["needed_for"]),
        d["summary"]["unknown"])
sys.exit(R.finish())
