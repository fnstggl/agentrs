"""Task generation from world opportunities and its executable validation (Part 17)."""
import json
import os

import pytest

import crucible
from crucible.goals import taskgen
from crucible.goals.contracts import GoalContract

crucible.load_all_modules()
DAY = 86400


@pytest.fixture(scope="module")
def mined():
    w, a, world, opps = taskgen.mine_world("open_tight_compute", 2, 3 * DAY)
    return w, a, world, opps, taskgen.propose(opps, world)


def test_mining_reads_reported_losses_and_proposes_outcome_only_goals(mined):
    w, a, world, opps, goals = mined
    kinds = {o["kind"] for o in opps}
    assert {"reported_loss", "shared_problem", "search_topic"} <= kinds
    # one task per (firm, loss kind), sourced from a real open request
    rl = [o for o in opps if o["kind"] == "reported_loss"]
    assert len({(o["by"], o["loss"]) for o in rl}) == len(rl)
    for o in rl:
        assert w.state.tables["ops_requests"][o["request"]]["by"] == o["by"]
    for g in goals:
        assert g.validate() == []
        assert "how" not in g.description.lower()
        assert g.split in ("train", "heldout") and g.split == taskgen.split_for(g.world)


@pytest.mark.slow
def test_visibility_task_passes_every_check_with_a_competent_witness(mined):
    g = next(g for g in mined[4] if g.family == "visibility")
    r = taskgen.TaskValidator().validate(g)
    assert r.verdict == "accepted", r.reasons
    for k in ("well_formed", "prerequisites", "measurable", "not_already_satisfied", "no_shortcut", "feasible",
              "near_miss"):
        assert r.checks[k]["ok"], k
    assert not r.controls["idle"]["success"] and not r.controls["random"]["success"]
    assert r.controls["site_author"]["success"]
    assert all(not t["still_true"] for t in r.checks["near_miss"]["threshold"])
    assert r.checks["near_miss"]["binding"]["still_true"] is False
    assert 0 <= r.difficulty <= 1


def _variant(g, **kw):
    d = g.to_dict()
    d.update(kw)
    return taskgen.goal_from_dict({k: v for k, v in d.items() if k != "id"})


def test_validator_rejects_malformed_satisfied_shortcut_and_unwitnessed_tasks(mined):
    goals = mined[4]
    vis = next(g for g in goals if g.family == "visibility")
    v = taskgen.TaskValidator(run_random=False)
    bad = _variant(vis, success={"metric": "no_such_metric", "op": ">=", "value": 1})
    assert v.validate(bad).reasons[0].startswith("malformed")
    sat = _variant(vis, success={"metric": "search_best_rank", "args": {"principal": "$org", "q": "x"}, "op": "<=",
                                 "value": 99})
    assert v.validate(sat).reasons == ["already satisfied at the start"]
    rl = next(g for g in goals if g.family == "relieve_loss")
    firm = rl.params["payer"]
    now = mined[0]
    from crucible.goals.metrics import evaluate_metric
    cur = evaluate_metric(now, "loss_events", {"org": firm, "kind": "capacity_overload", "window_s": 7 * DAY})
    drift = _variant(rl, horizon_s=3 * DAY, success={"metric": "loss_events", "args": {
        "org": firm, "kind": "capacity_overload", "window_s": 7 * DAY}, "op": ">=", "value": cur + 1})
    r = v.validate(drift)
    assert r.reasons == ["shortcut: the idle baseline succeeds"]       # the firm's losses grow by themselves
    nowit = taskgen.TaskValidator(controls=lambda g: {}, run_random=False).validate(_variant(vis, horizon_s=2 * DAY))
    assert nowit.verdict == "rejected" and nowit.reasons[0].startswith("no feasibility witness")


def test_heldout_set_is_sealed_read_only_and_disjoint(tmp_path, mined):
    goals = mined[4]
    held = [_variant(g, split="heldout") for g in goals[:3]]
    p = str(tmp_path / "heldout.json")
    seal = taskgen.lock_heldout(held, None, p)
    assert len(seal) == 64 and not os.access(p, os.W_OK) or os.geteuid() == 0
    back = taskgen.load_locked(p)
    assert [g.id for g in back] == [g.id for g in held]
    with pytest.raises(ValueError):
        taskgen.assert_disjoint(held[:1], p)
    os.chmod(p, 0o644)
    d = json.load(open(p))
    d["tasks"][0]["success"]["value"] = 0
    json.dump(d, open(p, "w"))
    with pytest.raises((PermissionError, ValueError)):
        taskgen.load_locked(p)


def test_cli_mines_tasks_as_json(capsys):
    from crucible.cli import main
    assert main(["tasks", "mine", "--family", "open_tight_compute", "--seed", "2", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {t["family"] for t in out["tasks"]} >= {"relieve_loss", "adoption", "visibility"}
