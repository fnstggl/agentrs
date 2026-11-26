"""Workstream L (wave 3b): the out-of-process evaluation path (`crucible.experience.remote.evaluate`) and its isolation
probes (`crucible.experience.isolation`, W.2 / SP-16c / OO.4): an evaluated learner holds only the participant pipe;
the world, its grader and the evaluator live in other processes; seeds are sealed."""
import pytest

from crucible.experience import make
from crucible.experience import isolation as I
from crucible.experience.remote import evaluate, sealed_seed

pytestmark = pytest.mark.timeout(900)
RULE = {"tier": 1, "name": "pe_rule_induction"}


def test_evaluate_runs_a_learner_out_of_process_and_verifies_on_the_evaluator_side():
    out = evaluate({"tier": 1, "name": "rapid:net_routing"}, "crucible.experience.remote:reference_learner",
                   {"steps": 40, "update_every": 20})
    assert out["learner_status"] == "ok", out["learner_error"]
    res = out["learner_result"]
    assert res["actions"] >= 20 and res["schema_failures"] == 0 and res["nonzero_rewards"] == 0
    assert res["params_hash_before"] != res["params_hash_after"] and res["params_changed_updates"] >= 1
    assert out["seed_sealed"] and out["ref"]["seed"] >= 2**20
    assert "outcome" in out["verification"] and out["digest"]
    p = out["processes"]
    assert len({p["evaluator"], p["learner"], p["world"]}) == 3


def test_a_failing_learner_is_reported_and_the_world_is_still_verified():
    out = evaluate({"tier": 1, "name": "rapid:net_routing", "seed": 5}, "crucible.experience.isolation:nonexistent")
    assert out["learner_status"] == "error" and "AttributeError" in out["learner_error"][0]
    assert not out["seed_sealed"] and out["ref"]["seed"] == 5 and "outcome" in out["verification"]


def test_adversarial_learner_reaches_nothing_with_a_sealed_seed():
    r = I.probe(dict(RULE), family=dict(RULE), max_seeds=12)
    assert r["examined"] and r["seed_sealed"]
    m = r["metrics"]
    assert m["evaluator_objects_in_learner"] == 0 and m["hidden_like_objects_in_learner"] == 0
    assert m["evaluator_ops_tried"] >= 10 and m["evaluator_ops_accepted"] == 0
    assert m["other_processes_examined"] >= 2 and m["other_process_memory_read"] == 0
    assert m["protocol_channels_opened"] == 0
    assert m["hidden_values_checked"] >= 8 and m["hidden_values_in_learner_material"] == 0
    assert not m["seed_in_learner_material"] and m["rebuild_seeds_tried"] == 12 and m["rebuild_recovered_world"] == 0
    assert r["held"] and set(r["rewards"]) == {0.0}


def test_positive_controls_in_process_object_graph_and_guessable_seed():
    # in process: the object make() returns leads to the grader's hidden rule and holdout
    env = make(1, RULE["name"], 3)
    walk = I.in_process_walk(env)
    env.close()
    assert walk["n_found"] >= 1, walk
    # out of process but with a small seed: the adversary rebuilds the world and reads its hidden configuration
    r = I.probe(dict(RULE, seed=3), family=dict(RULE), max_seeds=8)
    assert r["examined"] and not r["seed_sealed"] and not r["held"]
    assert r["metrics"]["rebuild_recovered_world"] == 1 and r["metrics"]["hidden_values_in_learner_material"] >= 8


def test_sealed_seeds_are_large_and_distinct():
    xs = {sealed_seed() for _ in range(64)}
    assert len(xs) == 64 and min(xs) >= 0 and max(xs) < 2**62


def test_evaluated_kernel_world_learner_reaches_no_evaluator_data():
    r = I.probe({"tier": 0, "name": "tiny", "kw": {"agents": ["learner"]}})
    assert r["examined"] and r["held"], r
    assert r["metrics"]["hidden_values_checked"] >= 2 and r["metrics"]["evaluator_ops_accepted"] == 0


def test_the_participant_pipe_is_closed_before_the_evaluator_verifies():
    """A process a learner leaves behind with a copy of its pipe cannot act during verification: the evaluator stops
    the world serving the participant side first."""
    from crucible.experience import host
    from crucible.experience.remote import RemoteError
    part, ev = host({"tier": 1, "name": "rapid:net_routing", "seed": 4})
    try:
        a = part.agents()[0]
        part.advance(int(part.tick_s))
        ev.close_participant()
        with pytest.raises((EOFError, OSError, RemoteError)):
            part.observe(a)
        assert "outcome" in ev.verify()
    finally:
        ev.close()


def test_an_external_learner_file_is_evaluated_out_of_process():
    """An architecture-neutral external learner (examples/external_learner/curiosity_learner.py, its own reward) is
    named by file and function and evaluated with the world in another process; it updates its own parameters."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "external_learner",
                        "curiosity_learner.py")
    out = evaluate({"tier": 1, "name": "rapid:net_routing"}, f"{path}:evaluated", {"steps": 60})
    assert out["learner_status"] == "ok", out["learner_error"]
    res = out["learner_result"]
    assert res["actions"] >= 30 and res["env_reward_sum"] == 0.0 and res["param_updates_changed"] >= 1
    assert res["distinct_hashes"] >= 2 and res["schema_failures"] == 0 and "outcome" in out["verification"]
