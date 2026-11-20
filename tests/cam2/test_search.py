"""Outer-learner components: typed specs, learning-rule expressions, population/ASHA bookkeeping, controllers."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from agentrs.cam2.model import CAM
from agentrs.cam2.search.controller import AdaptiveController, PBTController, UniformController
from agentrs.cam2.search.population import ASHA, Budget, Population, paired_delta
from agentrs.cam2.search.proposer import apply_edit, parse_edit
from agentrs.cam2.search.rules import (Node, Rule, a2c, check_rule, crossover, equivalent_to_ppo_family, mutate_rule,
                                       ppo_clip, synthetic_env)
from agentrs.cam2.search.spec import (OPERATORS, PBT_OPERATORS, SpecError, cam_config, check_schema, count_params,
                                      default_spec, forward_backward_check, inherit_weights, mutate, validate)

DP, H = 64, 32   # tiny backbone dimensions keep the graph checks fast on CPU
TINY = {"arch": {"d": 256, "integ_layers": 1, "heads": 4, "core_dim": 512, "wm_hidden": 512, "head_hidden": 256,
                 "prefix_rank": 64, "wm_ensemble": 2}}


# ------------------------------------------------------------------------------------------------ specs
def test_default_spec_is_admissible():
    s = default_spec(TINY)
    rep = validate(s, DP, H, device="cpu")
    assert rep["params"]["total"] > 0 and np.isfinite(rep["forward_backward"]["loss"])


def test_schema_rejects_out_of_space_specs():
    s = default_spec(TINY)
    for bad in [lambda x: x["arch"].update(d=300), lambda x: x["arch"].update(heads=8, d=384 + 4),
                lambda x: x["learn"].update(lr=1.0), lambda x: x["arch"].update(exec="import os"),
                lambda x: x["learn"].update(rule=json.dumps({"policy": "ret", "vtarget": "ret"})),
                lambda x: x.update(evaluator={"tasks": []}),
                lambda x: (x["plan"].update(enabled=True), x["learn"].update(wm_coef=0.0))]:
        t = copy.deepcopy(s)
        bad(t)
        with pytest.raises(SpecError):
            check_schema(t)


def test_every_operator_produces_a_valid_different_spec():
    rng = random.Random(0)
    s = default_spec(TINY)
    for op in OPERATORS:
        c, desc = mutate(s, op, rng, {"population_rules": [ppo_clip().to_json(), a2c().to_json()]})
        assert c != s and desc
        check_schema(c)


def test_param_count_matches_built_model_and_budget_is_enforced():
    s = default_spec(TINY)
    n = count_params(s, DP, H)["total"]
    m = CAM(cam_config(s, DP, H))
    assert n == sum(p.numel() for p in m.parameters())
    from agentrs.cam2.search.spec import Limits

    with pytest.raises(SpecError, match="exceed"):
        validate(s, DP, H, Limits(max_params=n - 1), run_graph=False)


def test_forward_backward_check_covers_every_head():
    s = default_spec(TINY)
    s["plan"]["enabled"] = True
    s["arch"]["deliberation"] = True
    s["arch"]["core"] = "lstm"
    rep = forward_backward_check(s, DP, H)
    assert rep["params"] > 0


def test_weight_inheritance_copies_matching_shapes_only():
    s = default_spec(TINY)
    parent = CAM(cam_config(s, DP, H))
    s2 = copy.deepcopy(s)
    s2["arch"]["head_hidden"] = 512           # changes policy/value head shapes only
    child = CAM(cam_config(s2, DP, H))
    rep = inherit_weights(child, parent.state_dict())
    assert 0.5 < rep["frac_params_inherited"] < 1.0 and rep["tensors_fresh"] > 0
    assert torch.equal(child.integ.layers[0].linear1.weight, parent.integ.layers[0].linear1.weight)
    assert child.value[0].weight.shape != parent.value[0].weight.shape


# ------------------------------------------------------------------------------------------------ learning rules
def test_seed_rules_admissible_and_roundtrip():
    for r in (ppo_clip(), a2c()):
        assert check_rule(r) is None
        assert str(Rule.from_json(r.to_json())) == str(r)


def test_inadmissible_rules_are_rejected():
    no_path = Rule(Node.from_obj({"op": "mul", "args": ["adv", "ret"]}), Node("ret"))
    wrong_sign = Rule(Node.from_obj({"op": "mul", "args": ["logp", {"op": "sg", "args": ["adv"]}]}), Node("ret"))
    vanishing = Rule(Node.from_obj({"op": "neg", "args": [{"op": "mul", "args": [1e-9, {"op": "mul", "args": [
        "logp", {"op": "sg", "args": ["adv"]}]}]}]}), Node("ret"))
    bad_target = Rule(a2c().policy, Node.from_obj({"op": "mul", "args": [1e6, "ret"]}))
    assert "no path" in check_rule(no_path)
    assert "policy-improvement" in check_rule(wrong_sign)
    assert "vanishing" in check_rule(vanishing)
    assert "out of range" in check_rule(bad_target)


def test_rule_mutation_finds_valid_non_ppo_structures():
    rng = random.Random(1)
    r = ppo_clip()
    novel = 0
    for _ in range(30):
        r2, how = mutate_rule(r, rng)
        assert check_rule(r2) is None
        novel += not equivalent_to_ppo_family(r2)
    assert novel > 0
    c = crossover(ppo_clip(), a2c(), rng)
    assert check_rule(c) is None


def test_rule_gradient_raises_logp_of_positive_advantage_samples():
    env = synthetic_env(seed=5)
    loss = ppo_clip().policy.eval(env).mean()
    (g,) = torch.autograd.grad(loss, env["logp"])
    pos = env["adv"] > 0.5
    assert float((-g[pos]).mean()) > 0 and float((-g[env["adv"] < -0.5]).mean()) < 0


# ------------------------------------------------------------------------------------------------ population / ASHA
def _fake_pop(tmp_path, scores, rung=0):
    pop = Population(str(tmp_path), Budget(gpu_hours=1.0))
    s = default_spec(TINY)
    for i, sc in enumerate(scores):
        c = pop.add(s, None if i == 0 else pop.order[0], "seed" if i == 0 else "arch.width", f"m{i}", seed=i)
        c.rung, c.status = rung, "ready"
        c.results[str(rung)] = {"val": {"success": sc}, "compute": {"gpu_hours": 0.05},
                                "val_episodes": [{"task": f"t{j}", "success": (j / 10) < sc} for j in range(10)]}
    return pop


def test_asha_promotes_top_fraction_only(tmp_path):
    pop = _fake_pop(tmp_path, [0.1, 0.5, 0.3, 0.2, 0.4, 0.0])
    asha = ASHA(b0=100, eta=3, max_rung=2)
    pro = asha.promotable(pop)
    assert [c.score(0) for _, c in pro] == [0.5, 0.4]
    assert asha.budget(0) == 100 and asha.budget(2) == 900
    c = pro[0][1]
    c.promoted_from.append(0)
    assert [x.score(0) for _, x in asha.promotable(pop)] == [0.4]


def test_population_persists_and_resumes(tmp_path):
    pop = _fake_pop(tmp_path, [0.1, 0.3])
    c = pop.cands[pop.order[1]]
    c.status = "running"
    c.promoted_from.append(0)
    pop.hypotheses.append({"op": "arch.width", "outcome": "evaluated", "delta": 0.2})
    pop.budget.used_gpu_hours = 0.4
    pop.save()
    p2 = Population.load(str(tmp_path))
    assert p2.order == pop.order and p2.cands[c.cid].history == c.history
    assert p2.cands[c.cid].status == "ready" and p2.cands[c.cid].promoted_from == []   # interrupted promotion re-armed
    assert p2.budget.used_gpu_hours == 0.4 and p2.hypotheses == pop.hypotheses
    assert len({c.cid for c in p2.cands.values()}) == 2


def test_budget_stops():
    b = Budget(gpu_hours=1.0, env_steps=100, max_candidates=3)
    assert b.exhausted(0) is None
    b.used_env_steps = 100
    assert "environment-step" in b.exhausted(0)
    assert not b.can_create(3)


def test_paired_delta():
    a = [{"task": f"t{i}", "success": i < 7} for i in range(10)]
    b = [{"task": f"t{i}", "success": i < 4} for i in range(10)] + [{"task": "extra", "success": True}]
    d = paired_delta(a, b)
    assert d["n"] == 10 and abs(d["delta"] - 0.3) < 1e-9 and d["wins"] == 3 and d["losses"] == 0


# ------------------------------------------------------------------------------------------------ controllers
def test_adaptive_controller_learns_from_measured_improvements(tmp_path):
    pop = _fake_pop(tmp_path, [0.3, 0.3, 0.3])
    ctrl = AdaptiveController(["arch.width", "learn.optim", "rule.expr"], seed=0, eps=0.0)
    x = ctrl.context(pop, pop.cands[pop.order[0]]).tolist()
    for _ in range(6):
        pop.hypotheses.append({"op": "learn.optim", "context": x, "outcome": "evaluated", "delta": 0.15, "cost_h": 0.05,
                               "child_score": 0.45})
        pop.hypotheses.append({"op": "arch.width", "context": x, "outcome": "evaluated", "delta": -0.1, "cost_h": 0.05,
                               "child_score": 0.2})
        pop.hypotheses.append({"op": "rule.expr", "context": x, "outcome": "failed"})
    p = ctrl.probs(pop)
    assert p["learn.optim"] > 0.8 and p["arch.width"] < 0.1
    picks = [ctrl.propose(pop)[1] for _ in range(20)]
    assert picks.count("learn.optim") >= 16


def test_baseline_controllers_propose_from_their_operator_sets(tmp_path):
    pop = _fake_pop(tmp_path, [0.1, 0.6, 0.2, 0.3])
    u = UniformController(seed=0)
    assert all(op in OPERATORS for op in (u.propose(pop)[1] for _ in range(20)))
    pbt = PBTController(seed=0)
    for _ in range(20):
        parent, op = pbt.propose(pop)
        assert op in PBT_OPERATORS and parent.score(0) == 0.6   # top quartile of 4 = the best one


# ------------------------------------------------------------------------------------------------ LLM proposals
def test_llm_proposal_is_validated_and_bounded():
    s = default_spec(TINY)
    rng = random.Random(0)
    e = parse_edit('Sure: {"changes": {"arch.integ_layers": 2, "learn.lr": 0.0002}, "why": "deeper"} done')
    s2, desc = apply_edit(s, e, rng)
    assert s2["arch"]["integ_layers"] == 2 and "deeper" in desc
    quoted, _ = apply_edit(s, parse_edit('{"changes": {"learn.max_grad": "2.0", "arch.q_head": "false"}}'), rng)
    assert quoted["learn"]["max_grad"] == 2.0 and quoted["arch"]["q_head"] is False
    snapped, _ = apply_edit(s, parse_edit('{"changes": {"arch.d": 500}}'), rng)
    assert snapped["arch"]["d"] == 512 or snapped["arch"]["d"] in (384, 512)
    for bad in ['{"changes": {"eval.tasks": 1}}', '{"changes": {"arch.d": "big"}}', 'no json',
                '{"changes": {"learn.meta_target": [0]}}', '{"changes": {"a.b": 1, "c.d": 2, "e.f": 3, "g.h": 4}}',
                '{"changes": {"learn.rule": "import os"}}']:
        with pytest.raises(SpecError):
            apply_edit(s, parse_edit(bad), rng)
