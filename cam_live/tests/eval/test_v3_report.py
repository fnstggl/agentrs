"""cam_live.eval.v3_report and the X6 meta-check entry point: the Protocol v3 summaries count what happened
(achieved / never reached / interrupted / not run, how each objective ended, slow steps kept or reverted, code
identity) and the transfer ratio is computed on the (seed, objective) pairs."""
from __future__ import annotations

import json
import os

import pytest

from cam_live.eval import v3_report as R
from cam_live.eval.meta_checks import oe_result_for_checks

SHA = "a" * 40


def _obj(i, problem, *, achieved, decisions, reason="", status="completed", interrupted=False, code=SHA, **kw):
    o = {"index": i, "family": "SW", "problem": problem, "achieved": achieved, "decisions": decisions,
         "verified_status": "achieved" if achieved else "pending", "verified_value": 1.0 if achieved else 0.2,
         "runtime_status": status, "reason": reason, "executable_fraction": 1.0, "backbone_chosen_fraction": 0.5,
         "repeats_of_failed_actions": 0, "distinct_strategies": 2, "mean_prediction_nll": 1.0,
         "meta_updates": [{"applied": False, "skipped_reason": "insufficient_independent_outcomes:2<4"},
                          {"applied": True, "meta_grad_norm": 0.3, "estimator": "snips"}],
         "group_changed": {"reward": True, "semantic_adapters": True}, "mix": 1.0, "eval_code": code,
         "decisions_detail": [{"n_candidates": 10, "n_backbone_candidates": 4}], "wall_s": 100.0,
         "learning_updates": [{"timescale": "fast"}, {"timescale": "slow_adapters"}]}
    if interrupted:
        o = {"index": i, "family": "SW", "problem": problem, "interrupted": True, "achieved": False,
             "verified_status": "interrupted", "verified_value": None, "decisions": None, "wall_s": 790.0}
    o.update(kw)
    return o


def _res(seed, objs, plan, code=SHA, root=""):
    return {"condition": "E", "seed": seed, "plan": plan, "objectives": objs, "root": root,
            "code": {"snapshot_commit": code}}


def _write_slow_log(obj_dir):
    os.makedirs(obj_dir, exist_ok=True)
    rows = [{"timescale": "slow_adapters", "notes": json.dumps({"guard": {"accepted": True}}),
             "weight_delta": {"semantic_adapters": 0.01}},
            {"timescale": "slow_adapters", "notes": json.dumps({"guard": {"accepted": False, "reverted": True}})},
            {"timescale": "slow_adapters", "notes": json.dumps({"deferred": True, "skipped": "no held-out"})},
            {"timescale": "slow_adapters", "notes": json.dumps({"skipped": "no eligible experience"})},
            {"timescale": "frequent", "notes": json.dumps({"guard": {"accepted": True}})}]
    with open(os.path.join(obj_dir, "learning_updates.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"data": r}) + "\n")
        f.write('{"data": {"timescale": "slow_adap')          # torn final line: ignored


def test_stop_class_names_how_each_objective_ended():
    assert R.stop_class({"achieved": True}) == "achieved"
    assert R.stop_class({"interrupted": True}) == "interrupted"
    assert R.stop_class({"reason": "wall budget 600 s of this lock hold reached", "runtime_status": "stopped"}) \
        == "wall_budget"
    assert R.stop_class({"reason": "limit max_decisions exhausted", "runtime_status": "exhausted"}) == \
        "decision_budget"
    assert R.stop_class({"reason": "step cap reached", "runtime_status": "stopped"}) == "step_cap"


def test_slow_step_outcomes_from_the_learning_log(tmp_path):
    d = str(tmp_path / "obj00")
    _write_slow_log(d)
    s = R.slow_step_outcomes(d)
    assert (s["total"], s["kept"], s["reverted"], s["deferred"]) == (4, 1, 1, 1)
    assert s["skipped"] == {"no eligible experience": 1}
    assert R.slow_step_outcomes(str(tmp_path / "missing")) is None


def test_summary_counts_achieved_censored_interrupted_not_run_and_code(tmp_path):
    root = str(tmp_path / "root0")
    _write_slow_log(os.path.join(root, "obj00"))
    runs = {0: _res(0, [_obj(0, "rle", achieved=True, decisions=3),
                        _obj(1, "rpn", achieved=False, decisions=8, status="stopped",
                             reason="wall budget 600 s of this lock hold reached")], "SW:rle,SW:rpn,SW:merge",
                    root=root),
            1: _res(1, [_obj(0, "rle", achieved=False, decisions=20, status="exhausted",
                             reason="limit max_decisions exhausted"),
                        _obj(1, "rpn", achieved=False, decisions=None, interrupted=True)], "SW:rle,SW:rpn,SW:merge",
                    code="b" * 40)}
    s = R.summarise_condition(runs, 20, expect_commit=SHA)
    assert s["objectives_planned"] == 6 and s["objectives_attempted"] == 3
    assert s["objectives_achieved"] == 1 and s["objectives_never_reached"] == 2
    assert s["objectives_interrupted"] == 1 and s["objectives_not_run"] == 2
    assert s["stops"] == {"achieved": 1, "wall_budget": 1, "decision_budget": 1, "interrupted": 1}
    # censored at the budget: seed 0 (3 + 20) / 2, seed 1 only its attempted objective (20)
    assert s["decisions_to_achieve_censored"]["per_seed"] == [11.5, 20.0]
    assert s["backbone_candidate_fraction"]["mean"] == pytest.approx(0.4)
    assert s["slow_steps"]["kept"] == 1 and s["slow_steps"]["reverted"] == 1 and s["slow_steps"]["deferred"] == 1
    assert s["meta_updates_applied_total"] == 3 and s["meta_skip_reasons"] == {"insufficient_independent_outcomes": 3}
    assert s["code_check"]["all_match"] is False                          # seed 1 recorded another snapshot
    runs[1]["code"]["snapshot_commit"] = SHA
    assert R.summarise_condition(runs, 20, expect_commit=SHA)["code_check"]["all_match"] is True
    runs[1]["objectives"][0]["eval_code"] = "c" * 40                     # one objective ran other code
    assert R.summarise_condition(runs, 20, expect_commit=SHA)["code_check"]["all_match"] is False


def test_transfer_pairs_and_report_files(tmp_path):
    d = tmp_path / "res"
    for exp, name, objs in (
            ("x2", "initial_s0", [_obj(0, "unique", achieved=True, decisions=10),
                                  _obj(1, "base", achieved=False, decisions=20)]),
            ("x2", "experienced_s0", [_obj(0, "unique", achieved=True, decisions=5),
                                      _obj(1, "base", achieved=True, decisions=10)]),
            ("x1", "E_s0", [_obj(0, "rle", achieved=True, decisions=4)]),
            ("x4", "skills_s0", [_obj(0, "store_entry", achieved=True, decisions=4, primitive_ops_total=3)]),
            ("x4", "primitive_s0", [_obj(0, "store_entry", achieved=False, decisions=20, primitive_ops_total=9)])):
        os.makedirs(d / exp, exist_ok=True)
        (d / exp / f"{name}.json").write_text(json.dumps(_res(0, objs, ",".join("SW:" + o["problem"]
                                                                                 for o in objs))))
    rep = R.build_report(str(d), 20, expect_commit=SHA)
    t = rep["x2"]
    assert len(t["pairs"]) == 2
    assert t["S_q"]["S_q"] == pytest.approx((10 + 20) / (5 + 10))
    assert t["S_q_by_problem"]["base"]["S_q"] == pytest.approx(2.0)
    assert rep["x4_skills_vs_primitive"]["primitive_ops_total"]["a_minus_b"]["mean"] == pytest.approx(-6.0)
    md = R.tables_md(rep)
    assert "#### x1 / E" in md and "| 0 | 0 | SW:rle | achieved |" in md
    out = tmp_path / "out"
    assert R.main(["--results-dir", str(d), "--out-dir", str(out), "--expect-commit", SHA,
                   "--markdown", str(out / "t.md")]) == 0
    assert json.loads((out / "summary_x2.json").read_text())["S_q"]["S_q"] == pytest.approx(2.0)
    assert (out / "t.md").exists()


def test_oe_results_reach_the_meta_checks_with_every_meta_record():
    res = _res(0, [_obj(0, "rle", achieved=True, decisions=3), _obj(1, "rpn", achieved=True, decisions=2)], "x")
    r = oe_result_for_checks(res)
    assert r["condition"] == "E" and r["seed"] == 0 and r["decision_rows"] == []
    assert len(r["extract"]["meta_updates"]) == 4 and sum(m["applied"] for m in r["extract"]["meta_updates"]) == 2


def test_meta_checks_cli_on_a_cpu_oe_run(tmp_path, monkeypatch):
    """End to end on the tiny CPU agent (plumbing only): an ``oe_runner sequence --save-exports`` root goes through
    ``python -m cam_live.eval.meta_checks`` without the GPU lock (the export says cpu / tiny)."""
    from cam_live.eval import meta_checks, oe_runner
    monkeypatch.setenv("CAM_LIVE_HOME", str(tmp_path / "home"))
    os.makedirs(tmp_path / "home")
    out = tmp_path / "r.json"
    assert oe_runner.main(["sequence", "--cpu", "--allow-unavailable", "--condition", "E", "--seed", "0",
                           "--plan", "SW:rle", "--decisions", "8", "--out", str(out), "--root", str(tmp_path / "root"),
                           "--save-exports", "--threads", "2"]) == 0
    assert meta_checks.main(["--root", str(tmp_path / "root"), "--result", str(out), "--out",
                             str(tmp_path / "m.json"), "--threads", "2"]) == 0
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["gpu"] is False and m["condition"] == "E"
    res = json.loads(out.read_text())
    if res["objectives"][0]["decisions"] >= 6:              # the checks need >= 6 eligible records
        assert "skipped" not in m, m
        assert m["3_4_meta_updates"]["n_records"] == len(res["objectives"][0]["meta_updates"])
        assert m["1_reward_params_affect_inner_update"]["mix0_differentiated"]["noise_floor_same_psi"] == 0.0
        assert m["3_4_meta_updates"]["applied_with_fewer_independent_outcomes"] == 0
