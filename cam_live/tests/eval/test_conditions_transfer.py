"""cam_live.eval: the compared conditions are the integrated agent with named switches; the transfer evaluation
runs in fresh processes on weights only (CPU, tiny backbone)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from cam_live.agent import CamLiveAgent
from cam_live.eval.conditions import (FastOnlyAgent, FrozenBackboneChooser, MemoryChooser, NoCreditAgent, Timed,
                                      UnverifiedAgent, base_config, build_condition, eval_config)
from cam_live.eval.runner import run_lifetime
from cam_live.validation import OffsetClock

torch.set_num_threads(2)


def test_eval_config_switches_only_the_named_pathways():
    base = base_config(real=False, seed=3)
    d = eval_config(base, "D", seed=3)
    assert d.sections["reward"]["meta"]["min_independent"] >= 10 ** 9 and d.sections["reward"]["meta"]["mix"] == 1.0
    assert eval_config(base, "C", seed=3).sections["agent"]["selection_rule"] == "thompson"
    assert eval_config(base, "E_mix0", seed=3).sections["reward"]["meta"]["mix"] == 0.0
    e = eval_config(base, "E", seed=3)
    assert e.sections["agent"]["use_backbone"] is False and "reward" not in e.sections or \
        "min_independent" not in e.sections.get("reward", {}).get("meta", {})
    assert base.sections == base_config(real=False, seed=3).sections        # eval_config never mutates its input


@pytest.mark.parametrize("cond,cls", [("A", FrozenBackboneChooser), ("B", MemoryChooser), ("C", FastOnlyAgent),
                                      ("D", CamLiveAgent), ("E", CamLiveAgent), ("E_no_credit", NoCreditAgent),
                                      ("E_unverified", UnverifiedAgent)])
def test_every_condition_is_the_integrated_agent(tmp_path, cond, cls):
    ag = build_condition(cond, base_config(real=False, seed=1), str(tmp_path / "a"), seed=1, clock=OffsetClock())
    try:
        assert type(ag) is cls and isinstance(ag, CamLiveAgent)
        if cond == "E_no_credit":
            assert ag.learner.credit is None and ag.rudder not in ag.learner.providers
        if cond == "E_unverified":
            assert ag.meta is not None and not ag._check_signature(object())
    finally:
        ag.close()


def _run(cond, tmp_path, n=6, family="F1", ws=10_003):
    clock = OffsetClock()
    ag = build_condition(cond, base_config(real=False, seed=2), str(tmp_path / "a"), seed=2, clock=clock)
    h0 = ag.model.group_hashes()
    lt = run_lifetime(Timed(ag), family, ws, str(tmp_path / "lt"), clock, max_decisions=n)
    return ag, h0, lt


def test_frozen_chooser_learns_nothing_and_logs_exact_probabilities(tmp_path):
    ag, h0, lt = _run("A", tmp_path)
    try:
        assert lt["decisions"] >= 3 and lt["training_s"] < 0.5
        assert ag.model.group_hashes() == h0
        for d in lt["decision_rows"]:
            assert abs(sum(d["probabilities"]) - 1.0) < 1e-9 and min(d["probabilities"]) > 0
        assert ag.learner.fast_state.n_updates("count") == 0
    finally:
        ag.close()


def test_memory_chooser_writes_notes_into_its_prompt(tmp_path):
    ag, h0, lt = _run("B", tmp_path)
    try:
        assert ag.notes and ag.model.group_hashes() == h0
        prompt = ag._prompt([])
        assert "Your notes from earlier experience" in prompt and ag.notes[-1][:40] in prompt
    finally:
        ag.close()


def test_fast_only_baseline_changes_beliefs_not_weights(tmp_path):
    ag, h0, lt = _run("C", tmp_path, n=8)
    try:
        assert ag.model.group_hashes() == h0
        assert sum(ag.learner.fast_state.n_updates(k) for k in ("count", "binary", "real")) > 0
    finally:
        ag.close()


def test_full_agent_changes_weights_and_ground_truth_is_recorded(tmp_path):
    ag, h0, lt = _run("E", tmp_path, n=8)
    try:
        h1 = ag.model.group_hashes()
        assert h1["experiential"] != h0["experiential"]
        rows = lt["decision_rows"]
        assert rows and all("truth" in c for d in rows for c in d["candidates"])
        acts = [c for d in rows for c in d["candidates"] if c["operation"] == "lab.act"]
        assert any(c["truth"] > 0 for c in acts)
        assert lt["verified_signature_ok"] and 0.0 <= lt["progress"] <= 1.0
    finally:
        ag.close()


def test_transfer_runs_in_fresh_processes_on_weights_only(tmp_path):
    from cam_live.eval.experiments import export_checkpoint
    clock = OffsetClock()
    ag = build_condition("E", base_config(real=False, seed=4), str(tmp_path / "w" / "agent"), seed=4, clock=clock)
    pre = export_checkpoint(ag, str(tmp_path / "w" / "ckpt_pre"), str(tmp_path / "w" / "export_pre"), note="pre")
    run_lifetime(Timed(ag), "F1", 10_004, str(tmp_path / "w" / "lt"), clock, max_decisions=6)
    post = export_checkpoint(ag, str(tmp_path / "w" / "ckpt_post"), str(tmp_path / "w" / "export_post"), note="post")
    ag.close()
    assert sorted(os.listdir(post["export"])) == ["config.json", "weights.safetensors"]
    out = tmp_path / "transfer.json"
    p = subprocess.run([sys.executable, "-m", "cam_live.runtime.cli", "evaluate-transfer", "--checkpoint",
                        str(tmp_path / "w" / "ckpt_post"), "--domains", "T2", "--seeds", "0", "--lifetimes", "1",
                        "--decisions", "3", "--threads", "2", "--out", str(out)],
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    rep = json.loads(out.read_text())
    assert rep["executed"] and rep["n_failed"] == 0 and rep["groups_differing_between_versions"]["experiential"]
    pids = {c["pid"] for c in rep["fresh_process_checks"]}
    assert len(pids) == 2 and os.getpid() not in pids
    for c in rep["fresh_process_checks"]:
        assert c["store_records"] == 0 and c["predictions_registered"] == 0 and c["experiments"] == 0
        assert c["cwd_files_at_start"] == [] and c["load_clean"] and c["hashes_match_export"]
        assert all(v == 0 for v in c["fast_state_updates"].values())
    assert rep["baseline"].endswith("export_pre")
    assert "T2" in rep["summary"] and rep["summary"]["T2"]["paired_seeds"] == [0]
