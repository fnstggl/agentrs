"""INT3B operator items: the shared-GPU lease cycled inside the running process (master item C), the operator's explicit
acknowledgement of ephemeral storage for a live run (item A), the debug validation never loading the pretrained
backbone (item B), and ``cam-live plan-report`` (verbatim export with provenance, written where the operator says).
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from cam_live.contracts import (ActionSpec, Decision, Distribution, Modality, OutcomeKind, OutcomeQuery,
                                PredictionRecord, RunFiles, SimClock)
from cam_live.diagnostics.world import LabConfig
from cam_live.runtime import cli
from cam_live.runtime import preflight as pf
from cam_live.runtime.gpu_cycle import GpuLease, attach
from cam_live.runtime.logs import RunLog, RunReader, atomic_write_json, read_json
from cam_live.runtime.loop import RuntimeConfig
from rt_helpers import make_lab

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LIVE_TASK = os.path.join(PROJECT, "tasks", "discord_100_planning.yaml")
STAGING_TASK = os.path.join(PROJECT, "tasks", "staging", "discord_100_planning_staging.yaml")

_HOLDER = """
import fcntl, sys, time
f = open(sys.argv[1], "a+")
fcntl.flock(f, fcntl.LOCK_EX)
open(sys.argv[2], "w").write("held")
time.sleep(float(sys.argv[3]))
"""


def _try_lock(path: str) -> bool:
    """True when another holder could take the lock right now (it is not held by anyone)."""
    with open(path, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


def _holder(tmp_path, lock: str, seconds: float, wait: bool = True):
    """Another tenant: a process that takes the lock (blocking) and holds it for ``seconds``."""
    flag = str(tmp_path / f"held_{time.time_ns()}")
    p = subprocess.Popen([sys.executable, "-c", _HOLDER, lock, flag, str(seconds)])
    p.flag = flag
    if wait:
        end = time.time() + 20
        while not os.path.exists(flag) and time.time() < end:
            time.sleep(0.02)
        assert os.path.exists(flag)
    return p


def _gpu_events(run_dir):
    return [e["data"] for e in RunReader(run_dir).records(RunFiles.EVENTS, types=["gpu_lock"])]


# ---------------------------------------------------------------------------------------------------------- item C
def test_the_lease_cycles_the_lock_in_the_running_process_and_waits_for_another_holder(tmp_path):
    lock = str(tmp_path / "gpu.lock")
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=100000),
                   rt_cfg=RuntimeConfig(fsync=False))
    lease = GpuLease(lock, hold_s=0.4, yield_s=0.3, poll_s=0.05, release_sleep_s=1e9)   # never release on a sleep
    assert lease.acquire() == 0.0 or lease.held
    rt = lab.rt
    attach(rt, lease)
    rt.start()
    holder = None
    t0 = time.time()
    while time.time() - t0 < 6.0 and rt.status == "running":
        rt.step()
        # once the runtime has yielded for the first time, a fake tenant queues for the lock (it gets it at the
        # runtime's next release, during the yield) and holds it for 1.5 s
        if holder is None and any(e["what"] == "released" for e in _gpu_events(rt.run_dir)):
            holder = _holder(tmp_path, lock, 1.5, wait=False)
    assert holder is not None
    holder.wait(timeout=30)
    assert os.path.exists(holder.flag)                                 # the other tenant did get its turn
    ev = _gpu_events(rt.run_dir)
    released = [e for e in ev if e["what"] == "released"]
    acquired = [e for e in ev if e["what"] == "acquired"]
    assert len(released) >= 2 and len(acquired) >= 2                  # cycled more than once, same process
    assert all("bounded hold" in e["why"] for e in released)
    assert max(e["waited_s"] for e in acquired) >= 1.0                 # it waited for the other holder
    assert lease.stats["longest_hold_s"] <= 0.4 + 0.5                  # hold bound (+ one short LabWorld cycle)
    # waiting for the lock is not charged to the task's compute limit (only learner calls are)
    assert rt.counters["compute_s"] < lease.stats["waited_s_total"]
    assert rt.status == "running" and rt.counters["decisions"] > 0
    st = read_json(os.path.join(rt.run_dir, RunFiles.STATE))
    assert st["gpu"]["acquisitions"] >= 3 and st["gpu"]["yields"] >= 2
    lease.release()
    rt.close()


def test_waiting_for_the_lock_is_visible_and_a_stop_ends_the_wait(tmp_path):
    lock = str(tmp_path / "gpu.lock")
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=100000), rt_cfg=RuntimeConfig(fsync=False))
    lease = GpuLease(lock, hold_s=0.0, yield_s=0.0, poll_s=0.05)
    rt = lab.rt
    lease.acquire()
    attach(rt, lease)
    rt.start()
    lease.release()                                                     # as after a sleep: not held
    holder = _holder(tmp_path, lock, 3.0)                               # another tenant holds it for 3 s
    seen = {}
    calls = []
    orig = lab.agent.save

    def save(path):                                                     # every checkpoint runs under the lock
        calls.append(lease.held)
        return orig(path)
    lab.agent.save = save

    def watch():
        end = time.time() + 20
        while time.time() < end:
            st = read_json(os.path.join(rt.run_dir, RunFiles.STATE)) or {}
            if (st.get("activity") or {}).get("phase") == "waiting_for_gpu_lock":
                seen["state"] = st
                rt.request_stop("operator stop while waiting for the GPU")
                return
            time.sleep(0.05)
    th = threading.Thread(target=watch)
    th.start()
    t0 = time.time()
    for _ in range(3):                                                  # the wait is visible; the stop is applied
        rt.step()                                                       # and its checkpoint taken under the lock
        if rt.status == "stopped":
            break
    th.join()
    try:
        assert seen and seen["state"]["gpu"]["held"] is False
        assert rt.status == "stopped" and time.time() - t0 < 30
        assert rt.counters["decisions"] == 0
        assert calls and all(calls)                                     # G2R-9: no save without the lock
    finally:
        holder.kill()
        lease.release()
        rt.close()


def test_sleeping_releases_the_lock_and_the_next_cycle_takes_it_again(tmp_path):
    lock = str(tmp_path / "gpu.lock")
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=100000), rt_cfg=RuntimeConfig(fsync=False))
    lease = GpuLease(lock, hold_s=840.0, poll_s=0.05, release_sleep_s=5.0)
    lease.acquire()
    rt = lab.rt
    attach(rt, lease)
    rt.sleeper.sleep_until(lab.clock.now() + 3600.0, lambda: False)   # a sleep of the runtime (simulated hour)
    assert not lease.held and _try_lock(lock)                          # nobody holds it while the runtime sleeps
    for _ in range(3):
        rt.step()
    assert any(e["what"] == "acquired" for e in _gpu_events(rt.run_dir))
    lease.release()
    rt.close()


def test_the_run_service_leases_the_lock_with_a_real_backbone_by_default(tmp_path, monkeypatch):
    from cam_live.config import CamLiveConfig
    from cam_live.runtime import loop_service
    from cam_live.runtime.gpu_cycle import lease_settings
    cfg = CamLiveConfig.load(os.path.join(PROJECT, "configs", "cam_live.yaml"))
    monkeypatch.setattr(type(cfg), "resolved_device", lambda self: "cuda")
    assert loop_service.needs_gpu_lock(cfg)                            # Qwen3-4B on CUDA: the lease is used
    assert lease_settings(cfg)["hold_s"] == 840.0 and lease_settings(cfg)["yield_s"] == 15.0
    assert lease_settings(CamLiveConfig())["hold_s"] == 840.0          # default when the file does not say
    assert not loop_service.needs_gpu_lock(CamLiveConfig.debug())


# ---------------------------------------------------------------------------------------------------------- item A
def _doctor_with(persistent):
    def run_doctor(cfg, *a, **k):
        c = {"name": "persistent_storage", "ok": persistent, "detail": "workspace_is_volume=false (test)",
             "needed_for": "live"}
        return {"sections": {"storage": [c]},
                "summary": {"blocking_staging": [], "blocking_live": [] if persistent else ["persistent_storage"]}}
    return run_doctor


def test_ephemeral_storage_blocks_a_live_run_unless_the_operator_accepts_it(monkeypatch):
    monkeypatch.setattr(pf, "run_doctor", _doctor_with(False))
    plain = pf.preflight(LIVE_TASK)
    assert "persistent_storage" in [m["item"] for m in plain["live"]["missing"]]
    assert plain["live"]["accepted_risks"] == []
    acc = pf.preflight(LIVE_TASK, accept_risks=["persistent_storage"])
    assert "persistent_storage" not in [m["item"] for m in acc["live"]["missing"]]
    (r,) = acc["live"]["accepted_risks"]
    assert r["item"] == "persistent_storage" and "--accept-ephemeral-storage" in r["accepted_by"]
    assert "workspace_is_volume=false" in r["detail"]
    assert "ACCEPTED RISK" in pf.render(acc)
    # nothing else can be accepted, and in staging mode the acknowledgement has no effect
    other = pf.preflight(LIVE_TASK, accept_risks=["persistent_storage", "separated services", "task"])
    assert [m["item"] for m in other["live"]["accepted_risks"]] == ["persistent_storage"]
    st = pf.preflight(STAGING_TASK, accept_risks=["persistent_storage"])
    assert st["live"]["accepted_risks"] == []


def test_cli_records_the_acknowledgement_and_passes_it_to_the_supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "run_doctor", _doctor_with(False))
    monkeypatch.setattr(cli, "users_exist", lambda: True)
    monkeypatch.setattr(cli.os, "geteuid", lambda: 0)
    spawned = []
    monkeypatch.setattr(cli, "_spawn_detached", lambda argv, log, env=None: spawned.append(argv) or 2 ** 22 + 7)
    auth = tmp_path / "authorization.json"
    auth.write_text("{}")
    home = tmp_path / "home"
    monkeypatch.setenv("CAM_LIVE_HOME", str(home))
    for flag, rd in (([], "run_plain"), (["--accept-ephemeral-storage"], "run_acc")):
        run_dir = str(tmp_path / rd)
        code = cli.main(["run", "--task", LIVE_TASK, "--run-dir", run_dir, "--authorization", str(auth),
                         "--start-timeout", "0.2"] + flag)
        assert code == 5                                               # nothing really started (spawn faked)
        meta = read_json(os.path.join(run_dir, "run.json"))
        pre = read_json(os.path.join(run_dir, "preflight.json"))
        if flag:
            assert meta["accepted_risks"] and meta["accepted_risks"][0]["check"] == "persistent_storage"
            assert "--accept-ephemeral-storage" in spawned[-1]
        else:
            assert meta["accepted_risks"] == []
            assert "--accept-ephemeral-storage" not in spawned[-1]
        assert isinstance(pre["live"].get("accepted_risks"), list)
    # the status view shows the accepted risk
    from cam_live.runtime.dashboard import RunView
    atomic_write_json(os.path.join(str(tmp_path / "run_acc"), RunFiles.STATE), {"status": "running", "version": 2})
    assert RunView(str(tmp_path / "run_acc")).status()["accepted_risks"][0]["check"] == "persistent_storage"


def test_without_the_flag_the_real_doctor_check_still_blocks_here():
    from cam_live.runtime.doctor import storage_checks
    from cam_live.config import CamLiveConfig
    c = {x["name"]: x for x in storage_checks(CamLiveConfig())}["persistent_storage"]
    if c["ok"] is True:
        pytest.skip("this machine's state root is on a persistent volume")
    rep = pf.preflight(LIVE_TASK, CamLiveConfig())
    assert "persistent_storage" in [m["item"] for m in rep["live"]["missing"]]


# ---------------------------------------------------------------------------------------------------------- item B
def test_debug_validation_never_loads_the_pretrained_backbone():
    from cam_live.config import CamLiveConfig
    from cam_live.validation import staging_config
    real = CamLiveConfig.load(os.path.join(PROJECT, "configs", "cam_live.yaml"))
    assert real.backbone.name != "tiny"
    cfg = staging_config(real, False, 0)                               # what `cam-live validate --debug` passes
    assert cfg.backbone.name == "tiny" and cfg.resolved_device() == "cpu" and cfg.core.scale == "debug"
    assert staging_config(None, False, 0).backbone.name == "tiny"


def test_validation_takes_the_gpu_lock_whenever_the_backbone_is_on_cuda():
    import ast
    src = open(os.path.join(PROJECT, "src", "cam_live", "validation.py")).read()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "run_staging")
    body = ast.get_source_segment(src, fn)
    # the lock decision is the configuration's (pretrained backbone on CUDA), not the real_model flag
    assert 'cfg.backbone.name != "tiny" and cfg.resolved_device() == "cuda"' in body
    assert "lock = shared_gpu(" in body


# ---------------------------------------------------------------------------------------------------------- plan report
def _note_run(tmp_path):
    """A run directory written by the runtime's own writers (RunLog) and the internal executor's notebook and request
    file, with CAM-like texts (newlines, quotes, unicode) that the report must reproduce exactly."""
    from cam_live.harness.operator_requests import OperatorRequests
    run = str(tmp_path / "run")
    os.makedirs(os.path.join(run, "exec", "workspace"), exist_ok=True)
    clock = SimClock()
    log = RunLog(run, clock, provenance="staging", durable=False)
    objective = "An objective text\nwith a second line."
    log.event("run_started", {"task": {"objective": objective, "objective_id": "o1"}})
    atomic_write_json(os.path.join(run, "task.json"), {"learner_view": {"objective": objective, "objective_id": "o1",
                                                                         "mode": "staging"}, "task_sha256": "ab"})
    texts = ["Note A: first line\n  second line — with “quotes” and | pipes", "Note B"]
    notes_path = os.path.join(run, "exec", "workspace", "notes.jsonl")
    reqs = OperatorRequests(os.path.join(run, "exec"), now=clock.now)
    out = {"decisions": []}
    for i, (op, args) in enumerate([("internal.note", {"text": texts[0], "tags": ["plan"]}),
                                    ("internal.request_operator", {"kind": "resource", "title": "A title",
                                                                   "description": "What is asked.",
                                                                   "justification": "Why it\nwould help."}),
                                    ("internal.note", {"text": texts[1]}),
                                    ("discord.modify_guild", {"guild_id": "1", "name": "x"})]):
        clock.advance(10.0)
        a = ActionSpec(action_id=f"act_{i}", operation=op, modality=Modality.INTERNAL, arguments=args,
                       provenance="backbone")
        d = Decision(decision_id=f"dec_{i}", ts=clock.now(), objective_id="o1", chosen=a)
        log.decision(d, objective=objective, situation={})
        q = OutcomeQuery(query_id=f"q{i}", metric="action.succeeded", kind=OutcomeKind.BINARY, horizon_s=0.0,
                         issued_at=clock.now(), decision_id=d.decision_id)
        log.predictions([PredictionRecord(prediction_id=f"p{i}", made_at=clock.now(), query=q,
                                          dist=Distribution(family="bernoulli", mean=0.5, std=0.5),
                                          action_id=a.action_id, decision_id=d.decision_id)],
                        decision_id=d.decision_id, chosen_action_id=a.action_id)
        kind, status = (("queued", "needs_approval") if op.startswith("discord.") else ("execute", "ok"))
        log.event("action_result", {"decision_id": d.decision_id, "action_id": a.action_id, "kind": kind,
                                    "operation": op, "status": status, "approval_id": "apr_1" if kind == "queued" else ""})
        if op == "internal.note":
            with open(notes_path, "a") as f:
                f.write(json.dumps({"note_id": f"note_{i}", "ts": clock.now(), "text": args["text"],
                                    "tags": args.get("tags", []), "action_id": a.action_id}) + "\n")
        if op == "internal.request_operator":
            r = reqs.request(args["kind"], args["title"], args["description"], args["justification"],
                             action_id=a.action_id)
            out["request_id"] = r["request_id"]
        out["decisions"].append(d.decision_id)
    reqs.resolve(out["request_id"], "denied", "not in this phase")
    log.close()
    atomic_write_json(os.path.join(run, RunFiles.STATE), {"version": 2, "status": "running", "run_id": "r",
                                                          "counters": {"decisions": 4},
                                                          "approvals": {"apr_1": {"status": "pending",
                                                                                  "operation": "discord.modify_guild",
                                                                                  "decision_id": "dec_3"}}})
    out.update(run=run, texts=texts, objective=objective)
    return out


def test_plan_report_reproduces_cam_texts_verbatim_with_provenance(tmp_path):
    from cam_live.runtime.plan_report import build_report, render_markdown
    w = _note_run(tmp_path)
    rep = build_report(w["run"])
    assert rep["objective"]["text"] == w["objective"] and rep["objective"]["received"][0]["text"] == w["objective"]
    assert [n["text"] for n in rep["notes"]] == w["texts"]                 # exact, unedited, in CAM's order
    assert rep["notes"][0]["tags"] == ["plan"] and rep["notes_by_tag"] == {"plan": [0], "(untagged)": [1]}
    assert [n["provenance"]["decision_id"] for n in rep["notes"]] == ["dec_0", "dec_2"]
    (q,) = rep["operator_requests"]
    assert q["justification"] == "Why it\nwould help." and q["status"] == "denied"
    assert q["resolution_note"] == "not in this phase" and q["provenance"]["decision_id"] == "dec_1"
    groups = {a["provenance"]["decision_id"]: a["group"] for a in rep["actions"]}
    assert groups == {"dec_0": "executed", "dec_1": "executed", "dec_2": "executed", "dec_3": "queued"}
    assert rep["counts"]["predictions_registered"] == 4 and rep["approvals"][0]["status"] == "pending"
    md = render_markdown(rep)
    # every line of every CAM text appears verbatim (as a line of a literal block) in the Markdown form
    for t in w["texts"] + [w["objective"], "Why it\nwould help."]:
        for line in t.split("\n"):
            assert line in md.splitlines()


def test_plan_report_is_written_wherever_the_operator_says(tmp_path, monkeypatch):
    w = _note_run(tmp_path)
    dest = tmp_path / "elsewhere" / "deep" / "report.md"
    assert cli.main(["plan-report", "--run-dir", w["run"], "--out", str(dest)]) == 0
    assert dest.exists() and w["texts"][1] in dest.read_text()
    dj = tmp_path / "copy" / "report.json"
    assert cli.main(["plan-report", "--run-dir", w["run"], "--out", str(dj), "--json"]) == 0
    assert json.loads(dj.read_text())["notes"][0]["text"] == w["texts"][0]


_GREEDY = """
import fcntl, sys, time
end = time.time() + float(sys.argv[2])
n = 0
while time.time() < end:
    f = open(sys.argv[1], "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close(); time.sleep(0.01); continue
    n += 1
    if n == 1:
        open(sys.argv[3], "w").write("held")
    time.sleep(0.3)                     # hold, release, and immediately try again
    fcntl.flock(f, fcntl.LOCK_UN); f.close()
"""


def test_a_queued_waiter_gets_its_turn_from_a_tenant_that_re_polls_at_once(tmp_path):
    lock, flag = str(tmp_path / "gpu.lock"), str(tmp_path / "greedy_started")
    p = subprocess.Popen([sys.executable, "-c", _GREEDY, lock, "8", flag])
    try:
        end = time.time() + 20
        while not os.path.exists(flag) and time.time() < end:
            time.sleep(0.01)
        lease = GpuLease(lock, hold_s=840.0, poll_s=0.05)
        t0 = time.time()
        waited = lease.acquire(lambda: time.time() - t0 > 6.0)
        assert waited is not None and waited < 6.0 and lease.held     # the kernel queue gave it to the waiter
        assert lease.release() >= 0.0 and not lease.held
    finally:
        p.kill()


def test_a_resource_limit_counts_only_when_it_is_not_the_decision_budget():
    from cam_live.validation import checks
    def rep(reason, refusals=0):
        return {"phase_computer": {"reason": reason, "resource_limit_refusals": refusals}}
    assert checks(rep("limit compute_s exhausted"))["resource_limit_exercised"]["holds"]
    assert checks(rep("limit max_external_actions exhausted"))["resource_limit_exercised"]["holds"]
    assert checks(rep("limit max_decisions exhausted", refusals=1))["resource_limit_exercised"]["holds"]
    assert not checks(rep("limit max_decisions exhausted"))["resource_limit_exercised"]["holds"]
    assert not checks(rep("validation step budget reached"))["resource_limit_exercised"]["holds"]


def test_validate_passes_the_lock_budget_through(monkeypatch, tmp_path):
    seen = {}

    def fake_run_staging(cfg, *, real_model=True, n_decisions=40, seed=0, out_dir=None, evidence_dir=None,
                         budget_s=780.0):
        seen.update(real_model=real_model, budget_s=budget_s, backbone=cfg.backbone.name)
        return {"ok": False}
    monkeypatch.setattr(cli, "load_hook", lambda spec, what: fake_run_staging)
    monkeypatch.setenv("CAM_LIVE_HOME", str(tmp_path / "home"))
    assert cli.main(["validate", "--mode", "staging", "--debug", "--budget-s", "600", "--out", str(tmp_path / "v")]) == 1
    assert seen["budget_s"] == 600.0 and seen["real_model"] is False


# ---------------------------------------------------------------------------------------------------------- HX-5
def _site_task(mode):
    from cam_live.contracts import Limits, TaskSpec
    return TaskSpec(objective_id=f"o_{mode}", objective="x", mode=mode,
                    limits=Limits(duration_s=600, max_decisions=5, money=1, compute_s=60, max_external_actions=5),
                    resources={"staging_site": "http://127.0.0.1:18999/",
                               "owned_sites": [{"name": "a", "base_url": "http://localhost:18998/x"},
                                               {"name": "b", "base_url": "https://example.org/"}]})


def test_a_staging_tasks_own_local_sites_are_registered_and_never_in_live_mode(tmp_path):
    from cam_live.harness.auth import PolicyService
    from cam_live.runtime.tasks import register_staging_sites, staging_local_services
    assert staging_local_services(_site_task("staging")) == ["127.0.0.1:18999", "127.0.0.1:18998"]
    assert staging_local_services(_site_task("live")) == [] and staging_local_services(_site_task("diagnostic")) == []
    p = PolicyService(str(tmp_path / "run"))
    register_staging_sites(p, _site_task("staging"))
    got = {r["address"] for r in p.local_services() if r["source"] == "registered"}
    assert got == {"127.0.0.1:18999", "127.0.0.1:18998"}
    q = PolicyService(str(tmp_path / "live"))
    assert register_staging_sites(q, _site_task("live")) == []
    assert not [r for r in q.local_services() if r["source"] == "registered"]


def test_the_supervisor_registers_staging_sites_before_the_learner_starts(tmp_path, monkeypatch):
    import yaml
    from cam_live.harness import service as hservice
    from cam_live.harness.auth import client as aclient
    from cam_live.harness.auth import launcher as alauncher
    calls = []

    class Auth:
        def register_task(self, task, by):
            calls.append(("task", task.mode))

        def register_local_service(self, address, by, note=""):
            calls.append(("local", address))
            return {"address": address}

    class Handle:
        def stop(self):
            return {}
    monkeypatch.setattr(alauncher, "launch", lambda run_dir, _x, lc: Handle())
    monkeypatch.setattr(aclient, "connect", lambda run_dir: Auth())

    def no_env(*a, **k):
        raise RuntimeError("no executor in this test")
    monkeypatch.setattr(hservice, "RemoteEnvironment", no_env)
    raw = yaml.safe_load(open(STAGING_TASK))
    raw["resources"]["staging_site"] = "http://127.0.0.1:18999/"
    for mode in ("staging",):
        f = tmp_path / f"task_{mode}.yaml"
        f.write_text(yaml.safe_dump(raw, sort_keys=False))
        run = tmp_path / f"run_{mode}"
        run.mkdir()
        assert cli.main(["_supervise", "--run-dir", str(run), "--task", str(f)]) == 5   # stops at the fake executor
    assert calls[0] == ("task", "staging") and ("local", "127.0.0.1:18999") in calls
    # the live planning task: nothing is registered
    calls.clear()
    run = tmp_path / "run_live"
    run.mkdir()
    assert cli.main(["_supervise", "--run-dir", str(run), "--task", LIVE_TASK]) == 5
    assert calls == [("task", "live")]


# ---------------------------------------------------------------------------------------------------------- item E
def test_floats_are_not_redacted_as_member_ids_but_real_ids_are(tmp_path):
    """G2-learn: digits of a float (0.12345678901234567) were hashed as a Discord snowflake in learning_updates.jsonl.
    A number stays a number; a free-standing 18-19 digit id is still replaced by the run's keyed hash."""
    from cam_live.contracts import LearningUpdateRecord
    from cam_live.runtime.logs import Redactor
    r = Redactor.from_config({}, b"salt")
    for num in ("0.12345678901234567", "1.2345678901234567e-05", "12345678901234567.5", "-0.123456789012345678"):
        assert r.text(f"loss {num} ok") == f"loss {num} ok"
    ids = ("175928847299117063", "1234567890123456789")                    # 18 and 19 digits
    for i in ids:
        out = r.text(f"member {i} joined; mention <@{i}>; end {i}.")
        assert i not in out and out.count(r.hash_member(i)) == 3
    # through the run log: a learning record's notes keep their numbers
    log = RunLog(str(tmp_path / "run"), SimClock(), durable=False)
    note = json.dumps({"grad_norm": 0.12345678901234566, "user": "who 175928847299117063"})   # a repr-exact float
    log.learning_update(LearningUpdateRecord(update_id="u1", ts=0.0, trigger="t", timescale="frequent", notes=note))
    log.close()
    (row,) = RunReader(str(tmp_path / "run")).records(RunFiles.LEARNING)
    assert "0.12345678901234566" in row["data"]["notes"] and "175928847299117063" not in row["data"]["notes"]


def test_plan_report_keeps_cam_numbers_and_ids_byte_exact(tmp_path):
    from cam_live.runtime.plan_report import build_report
    w = _note_run(tmp_path)
    path = os.path.join(w["run"], "exec", "workspace", "notes.jsonl")
    text = "estimate 0.12345678901234567 for member 175928847299117063"
    with open(path, "a") as f:
        f.write(json.dumps({"note_id": "n9", "ts": 1.0, "text": text, "tags": [], "action_id": "act_2"}) + "\n")
    rep = build_report(w["run"])
    got = [n["text"] for n in rep["notes"] if "estimate" in n["text"]]
    assert got == [text]                         # G2R-7: CAM's text byte-exact in the JSON form (nothing redacted)
