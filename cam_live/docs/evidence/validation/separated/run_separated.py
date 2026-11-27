"""Staging validation through the separated services: ``cam-live run`` as root with the service users.

The operator (this script, as root) starts the local staging sites, writes a staging task for the entry-book site
and a debug-scale configuration, and runs ``cam-live run``: the launcher starts the authorization service, the
computer environment (Chromium) and the independent verifier under their own uids and the learner process
(``runtime.loop_service`` with ``cam_live.agent:build_agent``) under ``camlearn``. The script acts as the operator
for approvals (``cam-live approve`` / ``reject``) and finally stops the run (``cam-live stop``). The result is
summarised from the run directory into ``--out``.

    cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/validation/separated/run_separated.py --out X.json
"""
from __future__ import annotations

import argparse
import json
import os
import pwd
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))       # cam_live/
REPO = os.path.dirname(PROJECT)
SRC = os.path.join(PROJECT, "src")
sys.path.insert(0, SRC)


def cli(*args, timeout=600):
    env = dict(os.environ, PYTHONPATH=SRC)
    p = subprocess.run([sys.executable, "-m", "cam_live.runtime.cli", *args], capture_output=True, text=True,
                       env=env, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def uid_of_pid(pid):
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("Uid:"):
                return int(line.split()[1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--decisions", type=int, default=24)
    ap.add_argument("--base-port", type=int, default=18740)
    ap.add_argument("--timeout", type=float, default=2400.0)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    import yaml
    from cam_live.contracts import RunFiles
    from cam_live.harness import SERVICE_USERS, users_exist
    from cam_live.runtime.control import pid_alive, read_pid
    from cam_live.runtime.logs import RunReader, read_json
    from cam_live.validation import StagingHttpMetrics
    if os.geteuid() != 0 or not users_exist():
        print("needs root and the service users", file=sys.stderr)
        return 2
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = os.path.join(REPO, "runs", f"int2_separated_{stamp}")
    inp, run_dir = os.path.join(base, "in"), os.path.join(base, "run")
    os.makedirs(inp)
    for d in (os.path.join(REPO, "runs"), base, inp):
        os.chmod(d, 0o755)
    report = {"kind": "staging_validation_separated", "executed": False, "started_utc": stamp, "run_dir": run_dir}
    sites = subprocess.Popen([sys.executable, "-m", "cam_live.harness.staging", "--base-port", str(a.base_port)],
                             env=dict(os.environ, PYTHONPATH=SRC), stdout=subprocess.PIPE, text=True)
    sup = None
    try:
        caps = json.loads("".join(iter(lambda: sites.stdout.readline(), "}\n")) + "}")
        variants = caps["sites"]["variants"]["url"].rstrip("/")
        host = variants.replace("http://", "")
        before = {q.name: q.value for q in StagingHttpMetrics(variants).read()}["staging.entries.count"]
        target = int(before) + 1
        task = {
            "version": 1, "objective_id": "staging-entry-book", "mode": "staging", "family": "staging/store-entry",
            "objective": f"Increase the number of entries stored in the entry book to {target}.",
            "success": {"verifier": "metric_threshold", "metric": "staging.entries.count", "target": target,
                        "definition": "The entry book's own count of stored entries.",
                        "evidence_source": "cam_live.validation:staging_http_source"},
            "secondary_metrics": ["secondary.time_elapsed_s"],
            "resources": {"staging_site": variants, "owned": [host, variants + "/a/"], "read": [host],
                          "preauthorized": [host], "authorized_tools": ["structured_browser", "api", "internal"],
                          "publishing": {"authorized": False}},
            "interfaces": ["structured_browser", "api", "internal", "skill"],
            "limits": {"duration_s": 7200, "money": 0, "compute_s": 7200, "max_external_actions": 400,
                       "max_decisions": int(a.decisions)},
            "stop_conditions": ["verified_success", "verified_failure", "duration_exhausted", "decisions_exhausted",
                                "operator_stop", "emergency_stop"],
            "measurement": {"primary": "staging.entries.count", "verify_interval_s": 0,
                            "protocol": "The independent verifier reads the entry book's own state over its local "
                                        "state endpoint and signs the result."},
            "data_permissions": {"store_message_content": False, "train_on_message_content": False,
                                 "store_member_identifiers": False, "store_aggregate_metrics": True,
                                 "collect_feedback_with_consent": False}}
        task_path, cfg_path = os.path.join(inp, "task.yaml"), os.path.join(inp, "cfg.yaml")
        yaml.safe_dump(task, open(task_path, "w"), sort_keys=False)
        cfg = {"seed": 7, "device": "cpu", "home": os.path.join(base, "home"),
               "backbone": {"name": "tiny", "revision": "", "lora_rank": 2, "bridge_tokens": 2, "max_text_tokens": 48,
                            "tiny_vocab": 4096},
               "core": {"scale": "debug", "max_context": 48},
               "optim": {"meta_min_independent_outcomes": 4, "adapter_every": 3, "batch_size": 8},
               "agent": {"use_backbone": False, "planner": False, "proposer_per_tool": 2, "durable": True,
                         "policy_update_every": 3, "policy_update_batch": 12, "promote_every": 12,
                         "retention_records": 8},
               "credit": {"learner": {"min_independent_outcomes": 2, "warmup_steps": 0, "slow_batch_size": 3,
                                      "grad_last": 2}},
               "runtime": {"control_poll_s": 0.2, "min_idle_sleep_s": 0.0, "verify_every_s": 0.0,
                           "candidates_k": 10}}
        yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
        common = ("--config", cfg_path, "--run-dir", run_dir)
        t0 = time.time()
        code, out, err = cli("run", "--task", task_path, *common, "--no-doctor", "--start-timeout", "300", "--json")
        report["run_exit"] = code
        if code != 0:
            report["run_error"] = (err + out)[-4000:]
            log = os.path.join(run_dir, "logs", "supervisor.log")
            if os.path.exists(log):
                report["supervisor_log"] = open(log).read()[-4000:]
            return 1
        info = json.loads(out)
        report["start"] = info
        pid = info.get("runtime_pid") or read_pid(run_dir)
        uid = {role: pwd.getpwnam(user).pw_uid for role, user in SERVICE_USERS.items()}
        report["learner_uid"] = uid_of_pid(pid)
        report["service_uids"] = uid
        resolved = []
        while time.time() - t0 < a.timeout:
            st = read_json(os.path.join(run_dir, RunFiles.STATE), {}) or {}
            if st.get("status") in ("completed", "exhausted", "failed", "stopped"):
                break
            for row in st.get("approvals", []) if isinstance(st.get("approvals"), list) else \
                    list((st.get("approvals") or {}).values()):
                if isinstance(row, dict) and row.get("status") == "pending" and row["approval_id"] not in \
                        {r["approval_id"] for r in resolved}:
                    ok = len(resolved) % 3 != 2
                    c2, o2, e2 = cli("approve" if ok else "reject", row["approval_id"], *common, "--json")
                    resolved.append({"approval_id": row["approval_id"], "approved": ok, "exit": c2})
            time.sleep(2.0)
        st = read_json(os.path.join(run_dir, RunFiles.STATE), {}) or {}
        report["final_status"] = st.get("status")
        report["status_reason"] = st.get("status_reason")
        report["approvals_resolved"] = resolved
        code, out, err = cli("stop", *common, "--reason", "validation end", "--json")
        report["stop_exit"] = code
        sup_pid = int(open(os.path.join(run_dir, "supervisor.pid")).read()) if os.path.exists(
            os.path.join(run_dir, "supervisor.pid")) else 0
        end = time.time() + 120
        while time.time() < end and (pid_alive(pid) or (sup_pid and pid_alive(sup_pid))):
            time.sleep(0.5)
        r = RunReader(run_dir)
        st = r.state()
        results = [e["data"] for e in r.records(RunFiles.EVENTS, types=["action_result"])]
        decisions = r.records(RunFiles.DECISIONS)
        ups = [e["data"] for e in r.records(RunFiles.LEARNING)]
        metas = [e["data"] for e in r.records(RunFiles.META)]
        verified = r.records(RunFiles.EVENTS, types=["verified_outcome"])
        report.update(
            executed=True, seconds=time.time() - t0, counters=st.get("counters"),
            decisions=len(decisions),
            chosen=[(d["data"]["decision"]["chosen"]["operation"], d["data"]["decision"]["chosen"]["provenance"])
                    for d in decisions],
            result_status=dict(__import__("collections").Counter(f"{d.get('kind')}:{d.get('status')}" for d in results)),
            grant_refusals=sum(1 for d in results if "grant does not verify" in str(d.get("failure_detail"))),
            learning_updates=dict(__import__("collections").Counter(u.get("timescale") for u in ups)),
            weight_deltas_nonzero=sum(1 for u in ups if any(v > 0 for v in (u.get("weight_delta") or {}).values())),
            meta_updates={"applied": sum(1 for m in metas if m.get("applied")),
                          "skipped": dict(__import__("collections").Counter(
                              (m.get("skipped_reason") or "").split(":")[0] for m in metas if not m.get("applied")))},
            verified_outcomes=[v["data"]["outcome"].get("status") for v in verified][-5:],
            last_verified=verified[-1]["data"]["outcome"] if verified else None,
            refused_verified=st.get("refused_verified"),
            entries_after={q.name: q.value for q in StagingHttpMetrics(variants).read()}["staging.entries.count"],
            entries_before=before,
            errors=[e["data"] for e in r.records(RunFiles.EVENTS, types=["error"])][-5:],
            files_owned_by={"state": os.stat(os.path.join(run_dir, RunFiles.STATE)).st_uid,
                            "verifier_key": os.stat(os.path.join(run_dir, "verifier", "verifier.key")).st_uid},
            journal=os.path.exists(os.path.join(run_dir, "journal.md")))
        report["checks"] = {
            "learner_not_root": report["learner_uid"] == uid["learner"] != 0,
            "executed_by_executor": any(k.startswith("execute:ok") for k in report["result_status"]),
            "no_grant_refusals": report["grant_refusals"] == 0,
            "verified_outcomes_signed_and_accepted": bool(verified) and not st.get("refused_verified"),
            "neural_updates": report["weight_deltas_nonzero"] > 0,
            "stopped_cleanly": report["stop_exit"] == 0,
        }
        return 0
    finally:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1, default=str)
        for name in ("supervisor.pid", "runtime.pid"):
            try:
                p = int(open(os.path.join(run_dir, name)).read())
                if pid_alive(p):
                    os.kill(p, signal.SIGTERM)
            except (OSError, ValueError):
                pass
        sites.terminate()
        if not a.keep:
            time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
