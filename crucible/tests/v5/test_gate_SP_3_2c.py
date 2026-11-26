"""SP-3.2c (MASTER_SPEC Part 3.2): class 2 models have 100k-2M parameters and class 3 models 2-20M (measured);
strategic groups compete, cooperate, pursue unrelated objectives and adapt over their lifetime; they are not
identical copies of one policy.

One world of the one population (crucible.population): 400 lightweight individuals browsing the population's web,
32 adaptive individuals (class 2, one shared group) and 32 strategic participants (class 3) in the four default
policy groups of `pop_strategic` (A: MLP/REINFORCE, objective profit; A0: frozen copy of A's initial weights;
B: GRU/A2C, objective reach; C: MLP/PPO, objective profit), 14 simulated days, one decision every 12 hours. The
strategic participants build ventures with ordinary tools (code, cloud, forum, search); their visitors and buyers are
the population's own decisions. Measured from the world (`population.strategic.behaviour`, evaluator side):

class2_params_in_range / class3_params_in_range: parameter counts of every policy network in the world;
strategic_group_behaviour_kinds: compete (two groups' live sites on one market topic at the same time), cooperate
(a venture linked another member's venture site), unrelated objectives (>= 2 distinct objectives among deciding
groups), adapt over the lifetime (a learning group whose weights changed and whose policy moved: mean total
variation distance >= 0.05 between its action distributions on its own recorded observations under its first and its
current weights; the frozen control A0 is reported at 0);
identical_policy_copies_across_groups: groups whose final weights equal another group's."""
import pytest

import crucible
from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.acceptance.v5 import evidence
from crucible.core.kernel import World
from crucible.population import strategic as STR
from crucible.population.worlds import population_world

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-3.2c"
DAY = 86400
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3 * 3600)]


def _criteria(req):
    return next(e for e in MF.load()["requirements"] if e["id"] == req)["criteria"]


def test_SP_3_2c_class_sizes_and_strategic_behaviour():
    import torch
    torch.set_num_threads(1)
    spec = population_world("development", 7, n=400, sites=8, tick_s=1800,
                            classes={"adaptive": 32, "strategic": 32, "semantic": 0},
                            adaptive={"learn_every_s": 6 * 3600, "min_examples": 16})
    w = World.create(spec, strict=True)
    try:
        w.advance_time(14 * DAY)
        ad = w.modules["pop_adaptive"]
        c2 = {g: ad.policy(w, int(g)).params for g in w.state.tables["pop_groups"]}
        stm = w.modules["pop_strategic"]
        c3 = {g: stm.group(w, g).params for g in w.state.tables["pop_strat_groups"]}
        b = STR.behaviour(w)
        m = {"class2_params_in_range": bool(c2 and all(100_000 <= v <= 2_000_000 for v in c2.values())),
             "class3_params_in_range": bool(c3 and all(2_000_000 <= v <= 20_000_000 for v in c3.values())),
             "strategic_group_behaviour_kinds": int(b["kind_count"]),
             "identical_policy_copies_across_groups": int(b["identical_policy_copies"]),
             "class2_params_min": int(min(c2.values())), "class3_params_min": int(min(c3.values())),
             "class3_params_max": int(max(c3.values())), "strategic_decisions": int(b["decisions"]),
             "competing_topics": len(b["competing_topics"]), "link_pairs": int(b["link_pairs"]),
             "mutual_link_pairs": int(b["mutual_link_pairs"]), "cross_group_links": int(b["cross_group_links"]),
             "objectives": ",".join(b["objectives"]),
             "adapting_groups": ",".join(g for g, v in b["adapt"].items() if v["learning"] and v["weights_changed"]
                                         and (v["policy_shift_tv"] or 0) >= 0.05),
             **{f"policy_shift_tv_{g}": round(float(v["policy_shift_tv"] or 0.0), 4) for g, v in b["adapt"].items()}}
        for k, v in b["kinds"].items():
            m[f"kind_{k}"] = bool(v)
        evidence.record("SP-3.2c", m, command=CMD, notes=(
            f"Parameters per class-2 group {c2}, per class-3 group {c3}. Strategic behaviour (evaluator side, "
            f"population.strategic.behaviour): {b['kinds']}; adaptation per learning group (chi-square of action "
            f"distributions, first vs second half, and the policy shift on its own observations): {b['adapt']}; second-half mean reward per group "
            f"{b['second_half_mean_reward']}; action outcomes {b['outcomes']}; successful actions {b['actions_ok']}. "
            "Group A0 is a frozen copy of A's initial weights in the same market (control)."))
    finally:
        w.close()
    assert b["decisions"] > 0
    fails = [msg for c in _criteria("SP-3.2c") for ok, msg in [criterion_met(c, m)] if not ok]
    assert not fails, (fails, m)
