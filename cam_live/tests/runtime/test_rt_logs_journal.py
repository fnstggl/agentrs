"""Structured logs (redaction, durability, provenance, timestamps) and the research journal."""
import json
import os
import re

from cam_live.contracts import (ActionSpec, Decision, LearningUpdateRecord, Observation, Provenance, RunFiles, SimClock)
from cam_live.diagnostics.world import DAY, LabConfig
from cam_live.runtime.journal import behavioural_change, generate_journal
from cam_live.runtime.logs import LOG_FILES, Redactor, RunLog, RunReader, iter_jsonl
from rt_helpers import make_lab

SECRET = "hunter2-very-secret-value"
TOKEN = ".".join(["MTIzNDU2Nzg5MDEyMzQ1Njc4OQ", "GabcdE", "abcdefghijklmnopqrstuvwxyz0123456789ab"])  # token-shaped fixture, assembled at run time


def test_redaction_of_secrets_personal_information_and_member_identifiers(tmp_path):
    clock = SimClock()
    red = Redactor(b"salt", secret_values=[SECRET], data_permissions={"store_message_content": False})
    log = RunLog(str(tmp_path), clock, provenance="staging", redactor=red, durable=False)
    obs = Observation(
        obs_id="obs_1", ts=clock.now(), kind="page",
        text=f"login with {SECRET}; contact jane.doe@example.org or +1 415 555 0100 from 203.0.113.7; "
             f"header Authorization: Bearer abcdefghijklmnop1234 ; bot {TOKEN} ; hello <@123456789012345678>",
        provenance=Provenance(source="t"),
        data={"member_id": "123456789012345678", "author": {"id": "42", "name": "Jane", "joined": 5},
              "members": [{"user_id": "777"}, {"user_id": "777"}, {"user_id": "778"}],
              "password": "p4ss", "message_content": "the private text of a message", "count": 3,
              "settings": {"api_key": "sk-abcdefghijklmnopqrstuv"}, "note": "password=letmein&x=1"})
    log.observation(obs)
    log.close()
    raw = open(os.path.join(str(tmp_path), RunFiles.EVENTS)).read()
    for leaked in (SECRET, "jane.doe@example.org", "555 0100", "203.0.113.7", "abcdefghijklmnop1234", TOKEN,
                   "123456789012345678", "p4ss", "the private text", "sk-abcdefghijklmnopqrstuv", "letmein", "Jane"):
        assert leaked not in raw, leaked
    d = json.loads(raw)["data"]["observation"]["data"]
    assert d["member_id"].startswith("member:") and d["author"]["id"].startswith("member:")
    assert d["author"]["joined"] == 5 and d["count"] == 3                      # ordinary values are untouched
    assert d["members"][0]["user_id"] == d["members"][1]["user_id"] != d["members"][2]["user_id"]   # stable hashing
    assert d["message_content"] == "[REDACTED:not_permitted]" and d["password"] == "[REDACTED:secret]"
    assert red.counts["secret"] >= 2 and red.counts["email"] == 1 and red.counts["member"] >= 5
    # a different run (different salt) hashes the same member differently; a granted permission keeps content
    red2 = Redactor(b"other", data_permissions={"store_message_content": True})
    assert red2.hash_member("777") != red.hash_member("777")
    assert red2({"message_content": "kept"})["message_content"] == "kept"
    # patterns are configurable
    red3 = Redactor.from_config({"patterns": {"ticket": r"TICKET-\d+"}, "member_keys": ["handle"]}, b"s")
    out = red3({"text": "see TICKET-991", "handle": "someone"})
    assert out["text"] == "see [REDACTED:ticket]" and out["handle"].startswith("member:")


def test_append_only_envelopes_clock_timestamps_provenance_and_fsync(tmp_path, monkeypatch):
    clock = SimClock()
    synced = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    log = RunLog(str(tmp_path), clock, provenance="staging", run_id="run-x")
    n0 = len(synced)
    log.event("note", {"k": 1})                                   # not consequential: no fsync
    assert len(synced) == n0
    d = Decision("dec_1", clock.now(), "obj", ActionSpec("act_1", "lab.probe"))
    clock.advance(5)
    rid = log.decision(d, objective="o", situation={})
    assert len(synced) == n0 + 1                                  # consequential: fsync before returning
    u = LearningUpdateRecord("upd_1", clock.now(), "t", "frequent")
    log.learning_update(u)
    assert len(synced) == n0 + 2
    log.close()
    recs = [r for name in LOG_FILES for r in iter_jsonl(os.path.join(str(tmp_path), name))]
    assert [r["seq"] for r in sorted(recs, key=lambda r: r["seq"])] == [0, 1, 2]      # one sequence across files
    dec = next(r for r in recs if r["rid"] == rid)
    assert dec["ts"] == clock.now() and dec["provenance"] == "staging" and dec["run_id"] == "run-x"
    assert dec["source"] == "runtime" and rid.startswith("dec-")
    # reopening continues the sequence, and a torn final line left by a crash is repaired, never parsed
    with open(os.path.join(str(tmp_path), RunFiles.EVENTS), "a") as f:
        f.write('{"rid": "ev-torn", "seq": 99, "da')
    log2 = RunLog(str(tmp_path), clock, provenance="staging")
    assert log2.repaired == {RunFiles.EVENTS: len('{"rid": "ev-torn", "seq": 99, "da')}
    assert log2.event("after", {}).endswith("0000000003")
    before = open(os.path.join(str(tmp_path), RunFiles.DECISIONS)).read()
    log2.decision(d, objective="o", situation={})
    assert open(os.path.join(str(tmp_path), RunFiles.DECISIONS)).read().startswith(before)        # append-only
    log2.close()


def test_decision_records_have_the_required_fields(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000))
    for _ in range(12):
        lab.rt.step()
    recs = RunReader(lab.run_dir).records(RunFiles.DECISIONS)
    assert len(recs) == lab.agent.calls["select"] >= 5
    for r in recs:
        d = r["data"]["decision"]
        assert r["ts"] > 0 and r["data"]["objective"] == lab.task.objective                 # timestamp, objective
        sit = r["data"]["situation"]
        assert sit["observation_ids"] and "lab.adopters.total" in sit["metrics"]             # observable situation
        assert d["context_obs_ids"]                                                           # relevant evidence
        assert len(d["candidates"]) >= 2 and len(d["probabilities"]) == len(d["candidates"])   # considered actions
        assert abs(sum(d["probabilities"]) - 1) < 1e-9
        assert d["chosen"]["action_id"] == d["candidates"][d["chosen_index"]]["action"]["action_id"]
        assert d["rationale"] and d["model_version"] and d["exploration"]
        chosen = d["candidates"][d["chosen_index"]]
        if d["chosen"]["operation"] == "lab.act":                                             # estimated consequences
            p = chosen["predictions"][0]
            assert p["dist"]["mean"] is not None and p["dist"]["std"] is not None and chosen["uncertainty"] is not None
    assert all("reward" not in json.dumps(r["data"]["situation"]) for r in recs)


def test_journal_facts_carry_record_ids_and_learning_claims_need_evidence(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=5, target=10000, horizon_s=8 * DAY),
                   agent_kw={"only_ops": ["lab.act", "wait"], "wait_seconds": DAY})
    lab.rt.run(max_iterations=600)
    text = open(os.path.join(lab.run_dir, RunFiles.JOURNAL)).read()
    for heading in ("What CAM is currently trying to accomplish", "What it has attempted", "What succeeded",
                    "What failed", "What remains uncertain", "How its approach changed",
                    "Capabilities that appear to have been acquired"):
        assert heading in text, heading
    assert lab.task.objective in text
    lines = [ln for ln in text.splitlines() if ln.startswith("- ")]
    assert len(lines) > 15
    known = set()
    for name in LOG_FILES:
        known |= {r["rid"] for r in iter_jsonl(os.path.join(lab.run_dir, name))}
    cited = 0
    for ln in lines:
        # every statement is tagged, and ends with the records (or the file) it rests on
        assert ln.startswith("- [FACT] ") or ln.startswith("- [HYPOTHESIS] "), ln
        m = re.search(r"\((records|basis): ([^()]*)\)$", ln)
        assert m, ln
        rids = re.findall(r"\b(?:ev|dec|exp|pred|lu|mu|ob)-\d{10}\b", m.group(2))
        assert rids or re.search(r"state\.json|\.jsonl|run directory", m.group(2)), ln
        for rid in rids:
            assert rid in known, (rid, ln)
        cited += len(rids)
    assert cited > 20
    # this run has learning updates with a measured behavioural change, and the journal cites them
    ups = RunReader(lab.run_dir).records(RunFiles.LEARNING)
    behav = [u for u in ups if behavioural_change(u["data"])]
    assert behav
    cap = text.split("## 7.")[1].split("## 8.")[0]
    assert "measurably changed probe predictions" in cap and behav[0]["rid"] in cap
    assert f"{len(behav)} of {len(ups)} learning update(s)" in cap
    assert "skipped" in cap and "no learned reward function" in cap             # meta-learning status is stated as it is
    # inferences are never written as facts
    for ln in lines:
        if "appears" in ln or "shifted" in ln:
            assert ln.startswith("- [HYPOTHESIS] "), ln


def test_journal_makes_no_learning_claim_without_a_behavioural_change(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=5, target=10000, horizon_s=8 * DAY),
                   agent_kw={"only_ops": ["lab.act", "wait"], "wait_seconds": DAY})
    lab.rt.run(max_iterations=600)
    path = os.path.join(lab.run_dir, RunFiles.LEARNING)
    recs = list(iter_jsonl(path))
    assert recs
    with open(path, "w") as f:                      # the same run, had its updates changed weights but no behaviour
        for r in recs:
            r["data"]["prediction_change"] = {"probe_mean_abs_change": 0.0}
            f.write(json.dumps(r) + "\n")
    text = generate_journal(lab.run_dir)
    cap = text.split("## 7.")[1].split("## 8.")[0]
    assert "No learning update with a measured behavioural change has been recorded" in cap
    assert "makes no claim that learning occurred" in cap
    assert "improved" not in cap and "measurably changed" not in cap
    assert not [ln for ln in cap.splitlines() if ln.startswith("- [HYPOTHESIS]")]
    # and with no learning records at all
    os.remove(path)
    assert "(0 update record(s) examined)" in generate_journal(lab.run_dir)


def test_obstacle_records_link_failure_alternatives_and_learning(tmp_path):
    def edit(task):
        task.limits.money = 9.0
        task.limits.max_decisions = 60
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000, horizon_s=30 * DAY),
                   agent_kw={"only_ops": ["lab.act", "wait"], "wait_seconds": 3 * DAY}, task_edit=edit)
    lab.rt.run(max_iterations=600)
    r = RunReader(lab.run_dir)
    obs = [o["data"]["obstacle"] for o in r.obstacles()]
    assert obs
    decisions = {d["data"]["decision"]["chosen"]["action_id"]: d["data"]["decision"] for d in r.records(RunFiles.DECISIONS)}
    results = {e["data"]["action_id"]: e["data"] for e in r.records(RunFiles.EVENTS, types=["action_result"])}
    update_ids = {u["data"]["update_id"] for u in r.records(RunFiles.LEARNING)}
    used = {u["data"]["update_id"]: set(u["data"].get("exp_ids") or []) for u in r.records(RunFiles.LEARNING)}
    linked = 0
    for o in obs:
        assert o["failure_class"] == "resource_limit" and o["category"] == "resource" and o["description"]
        assert results[o["action_id"]]["status"] == "failed"
        if o["chosen_alternative"]:
            alt = decisions[o["chosen_alternative"]]
            assert o["chosen_alternative"] in o["alternatives"] and len(o["alternatives"]) == len(alt["candidates"])
            assert o["strategy_changed"] == (alt["chosen"]["operation"] != decisions[o["action_id"]]["chosen"]["operation"])
            if o["outcome"]:
                assert results[o["chosen_alternative"]]["status"] in o["outcome"]
        assert set(o["learning_update_ids"]) <= update_ids
        # G2I-9 (INT3B): an update is linked only when it used the obstacle's experience (here the refused action or
        # the alternative chosen after it); a later update that did not is not linked
        mine = {o["action_id"], o["chosen_alternative"]} - {""}
        assert all(used[u] & mine for u in o["learning_update_ids"])
        linked += bool(o["learning_update_ids"])
    assert any(o["strategy_changed"] for o in obs)
    # revisions are appended: the file keeps the history of each record
    revs = [x["data"]["rev"] for x in iter_jsonl(os.path.join(lab.run_dir, RunFiles.OBSTACLES))]
    assert max(revs) >= 1 and len(revs) > len(obs)
