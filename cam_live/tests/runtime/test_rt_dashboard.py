"""Dashboard endpoints and operator controls through a test client."""
import os
import re
import shutil

import pytest
from fastapi.testclient import TestClient

from cam_live.contracts import RunFiles
from cam_live.diagnostics.world import DAY, LabConfig
from cam_live.runtime.dashboard import (APP_JS, CSRF_COOKIE, CSRF_HEADER, OPERATOR_HEADER, PAGE, create_app,
                                        load_or_create_token, serve)
from cam_live.runtime.logs import RunReader
from cam_live.runtime.loop import PAUSED, RUNNING, STOPPED
from rt_helpers import ApprovalPolicy, make_lab

TOKEN = "t" * 32
OPER = "o" * 32          # operator credential: needed in addition for every operator action
TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tasks")


@pytest.fixture
def dash(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=5, target=10000, horizon_s=60 * DAY, budget=2000.0),
                   agent_kw={"only_ops": ["lab.act", "lab.unlock", "wait"], "wait_seconds": DAY},
                   policy=ApprovalPolicy(("lab.unlock",)))
    for _ in range(400):
        lab.rt.step()
        if lab.rt.approvals and len(RunReader(lab.run_dir).records(RunFiles.LEARNING)) >= 2:
            break
    assert lab.rt.status == RUNNING and lab.rt.approvals
    tasks_dir = tmp_path / "tasks"
    shutil.copytree(TASKS, tasks_dir)
    app = create_app(lab.run_dir, TOKEN, tasks_dir=str(tasks_dir), operator_token=OPER)
    client = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
    assert client.get("/").status_code == 200                      # the page sets the CSRF cookie
    client.headers.update({OPERATOR_HEADER: OPER, CSRF_HEADER: client.cookies[CSRF_COOKIE]})
    return lab, client, app, tasks_dir


def test_token_is_required_everywhere(dash, tmp_path):
    lab, client, app, _ = dash
    anon = TestClient(app)
    paths = ["/", "/app.js", "/app.css", "/api/operator", "/api/status", "/api/experiments", "/api/predictions", "/api/calibration", "/api/rewards",
             "/api/learning", "/api/meta", "/api/approvals", "/api/obstacles", "/api/health",
             "/api/logs/events.jsonl", "/api/journal", "/api/tasks"]
    for p in paths:
        assert anon.get(p).status_code == 401, p
        assert anon.get(p, headers={"Authorization": "Bearer wrong"}).status_code == 401, p
        assert client.get(p).status_code == 200, p
    for p in ("/api/control/pause", "/api/control/resume", "/api/control/stop", "/api/control/export-checkpoint",
              "/api/control/assign-objective", "/api/approvals/x/approve", "/api/approvals/x/reject",
              "/api/control/reset-stop"):
        assert anon.post(p, json={}).status_code == 401, p
        # the viewing token alone never performs an operator action (an unknown approval id / task file is a 404)
        viewer = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
        assert viewer.post(p, json={}).status_code == (404 if "/x/" in p or "assign" in p else 401), p
    assert not os.path.exists(os.path.join(lab.run_dir, RunFiles.CONTROL))
    assert anon.get("/api/status", params={"token": TOKEN}).status_code == 200
    with pytest.raises(ValueError):
        create_app(lab.run_dir, "short")
    with pytest.raises(ValueError, match="loopback"):
        serve(lab.run_dir, TOKEN, host="0.0.0.0")
    tok = load_or_create_token(str(tmp_path / "home"))
    assert len(tok) >= 32 and load_or_create_token(str(tmp_path / "home")) == tok
    assert oct(os.stat(tmp_path / "home" / "dashboard.token").st_mode & 0o777) == "0o600"


def test_read_endpoints_show_the_run(dash):
    lab, client, _, _ = dash
    s = client.get("/api/status").json()
    assert s["objective"]["text"] == lab.task.objective and s["status"] == RUNNING
    assert s["verified"]["status"] == "pending" and s["verified"]["target"] == 10000 and s["verified"]["record"]
    assert s["activity"]["phase"] and s["checkpoint"]["id"] == lab.agent.checkpoint_id
    assert s["resources"]["money"]["limit"] == 2000 and 0 < s["resources"]["money"]["used"] <= 2000
    assert s["resources"]["max_decisions"]["used"] == lab.agent.calls["select"]
    assert s["pending_approvals"] == 1
    ex = client.get("/api/experiments", params={"n": 1000}).json()
    assert ex and {"question", "hypothesis", "predicted", "actual", "supported_according_to_learner"} <= set(ex[0])
    assert any(e["actual"] for e in ex)                 # outcomes the runtime measured for the experiment
    assert all(a["record"].startswith("exp-") for e in ex for a in e["actual"])
    pr = client.get("/api/predictions").json()
    assert pr["n_resolved"] >= 1 and pr["n_registered"] > pr["n_resolved"]
    row = pr["resolved"][0]
    assert row["predicted_mean"] is not None and row["actual"] is not None and row["verified"] is True
    cal = client.get("/api/calibration").json()
    assert cal["n"] == pr["n_resolved"] and cal["mean_abs_error"] >= 0 and 0 <= cal["within_2_std"] <= 1
    rw = client.get("/api/rewards").json()
    assert rw["n_reports"] == 0 and "no reward" in rw["note"]          # this test agent has no reward model
    ln = client.get("/api/learning").json()
    assert ln["n_updates"] >= 2 and ln["recent"][0]["weight_delta"] and "behavioural_change" in ln["recent"][0]
    mt = client.get("/api/meta").json()
    assert mt["n_applied"] == 0 and mt["bootstrap_mix"] == 1.0
    assert mt["skipped_reasons"] == {"this agent has no learned reward function": mt["n_updates"]}
    ap = client.get("/api/approvals").json()
    assert len(ap["pending"]) == 1 and "lab.unlock" in ap["pending"][0]["preview"] and ap["pending"][0]["reasons"]
    h = client.get("/api/health").json()
    assert h["process"]["alive"] is True and h["disk_free_gb"] > 0 and h["log_bytes"]["events.jsonl"] > 0
    logs = client.get("/api/logs/decisions.jsonl", params={"n": 5}).json()
    assert len(logs) == 5 and logs[-1]["type"] == "decision"
    older = client.get("/api/logs/decisions.jsonl", params={"n": 5, "before_seq": logs[0]["seq"]}).json()
    assert all(x["seq"] < logs[0]["seq"] for x in older)
    assert client.get("/api/logs/state.json").status_code == 404        # only the run logs can be inspected
    assert client.get("/api/logs/..%2Fcontrol.json").status_code == 404
    assert "discord_100.yaml" in client.get("/api/tasks").json()


def test_controls_pause_resume_approve_reject_export_stop(dash):
    lab, client, _, _ = dash
    rt = lab.rt
    (aid,) = [a["approval_id"] for a in client.get("/api/approvals").json()["pending"]]
    assert client.post(f"/api/approvals/{aid}/approve", json={"note": "fine"}).json()["queued"]
    assert client.post("/api/approvals/nope/approve", json={}).status_code == 404
    lab.agent.only_ops = {"lab.act", "wait"}
    for _ in range(6):
        rt.step()
    ap = client.get("/api/approvals").json()
    assert ap["pending"] == [] and ap["resolved"][0]["status"] == "executed" and lab.world._unlocked_at is not None
    cid = client.post("/api/control/pause", json={"reason": "look"}).json()["command_id"]
    rt.step()
    assert rt.status == PAUSED and client.get("/api/status").json()["status"] == "paused"
    assert client.get(f"/api/control/result/{cid}").json()["result"]["ok"] is True
    cid = client.post("/api/control/export-checkpoint").json()["command_id"]
    rt.step()
    res = client.get(f"/api/control/result/{cid}").json()["result"]
    exports = client.get("/api/status").json()["exports"]
    assert res["ok"] and exports and exports[-1]["path"].startswith(os.path.join(lab.run_dir, "exports"))
    assert os.path.exists(os.path.join(exports[-1]["path"], "agent.json"))
    assert os.path.exists(exports[-1]["path"] + ".export.json")
    client.post("/api/control/resume")
    rt.step()
    assert rt.status == RUNNING
    client.post("/api/control/stop", json={"reason": "end"})
    rt.step()
    assert rt.status == STOPPED and client.get("/api/status").json()["status_reason"] == "end"
    assert "## 1. What CAM is currently trying to accomplish" in client.get("/api/journal").text


def test_reject_and_assign_objective_through_the_dashboard(dash):
    lab, client, _, tasks_dir = dash
    (aid,) = [a["approval_id"] for a in client.get("/api/approvals").json()["pending"]]
    client.post(f"/api/approvals/{aid}/reject", json={"note": "SECRET-OPERATOR-NOTE"})
    lab.agent.only_ops = {"lab.act", "wait"}
    for _ in range(4):
        lab.rt.step()
    assert client.get("/api/approvals").json()["resolved"][0]["status"] == "reported"
    action_id = next(a for a, s in lab.agent.results if s == "needs_approval")
    assert [s for a, s in lab.agent.results if a == action_id] == ["needs_approval", "denied"]
    # the operator's note is in the audit trail but never in what the learner receives
    res = [e for e in RunReader(lab.run_dir).records(RunFiles.EVENTS, types=["action_result"])
           if e["data"]["action_id"] == action_id][-1]
    assert "SECRET-OPERATOR-NOTE" not in res["data"]["failure_detail"]
    cmds = RunReader(lab.run_dir).records(RunFiles.EVENTS, types=["operator_command"])
    assert any("SECRET-OPERATOR-NOTE" in str(c["data"]["args"]) for c in cmds)
    # assigning an objective: only a task file from the tasks directory, validated like any task
    assert client.post("/api/control/assign-objective", json={"task": "../../etc/passwd"}).status_code == 404
    assert client.post("/api/control/assign-objective", json={"task": "discord_100.yaml"}).status_code == 422
    bad = tasks_dir / "bad.yaml"
    bad.write_text(open(tasks_dir / "lab_staging.yaml").read() + "\nplaybook: anything\n")
    r = client.post("/api/control/assign-objective", json={"task": "bad.yaml"})
    assert r.status_code == 422 and "playbook" in str(r.json()["detail"])
    r = client.post("/api/control/assign-objective", json={"task": "lab_staging.yaml"})
    assert r.status_code == 200
    lab.rt.step()
    assert client.get("/api/status").json()["objective"]["queued"] == ["lab-staging"]


def test_page_is_self_contained_and_has_no_strategy_input(dash):
    _, client, _, _ = dash
    html = client.get("/").text
    # one same-origin script and stylesheet (a no-inline-script CSP needs them); nothing from elsewhere
    assert html == PAGE and re.findall(r'<script src="([^"]+)"', html) == ["/app.js"] and "http://" not in html
    assert re.findall(r'<link [^>]*href="([^"]+)"', html) == ["/app.css"] and "https://" not in html
    assert client.get("/app.js").text == APP_JS
    for label in ("Objective and verified progress", "Current activity and health", "Resources against limits",
                  "Pending approvals", "Recent experiments", "Predicted and actual outcomes",
                  "Internal reward statistics and calibration", "Recent learning updates", "Meta-learning",
                  "Assign a subsequent objective", "Logs", "Pause", "Resume", "Stop", "Export checkpoint"):
        assert label in html, label
    assert "<textarea" not in html
    # the only free-text inputs are audit notes that never reach the learner (plus the operator credential field);
    # objectives come from task files
    inputs = re.findall(r"<input[^>]*>", html)
    assert inputs and all("audit note (not shown to CAM)" in i or ('type="password"' in i and 'id="op-cred"' in i)
                          for i in inputs)
    assert sum('type="password"' in i for i in inputs) == 1
    assert re.findall(r'<select id="([^"]+)"', html) == ["task-file", "log-name"]
