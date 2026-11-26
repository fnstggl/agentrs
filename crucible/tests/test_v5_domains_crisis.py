"""Tier 2 domain 9.14 (crisis recovery, kernel module `crisis`): hazard physics, cascades, operators, the
coordinator's tools and the counterfactual-relative evaluator, on real generated worlds."""
import pytest

from crucible.domains import _dom3 as K
from crucible.domains import base
from crucible.domains import crisis as C


def _env(config="storm_radial", seed=0):
    return K.make_env("crisis", config, seed)


def test_definition_valid_reference_tool_only_and_structures_differ():
    assert base.validate(C.DOMAIN) == []
    assert base.reads_world(C.DOMAIN.reference_policy("p:agent", {})) == []
    fps = {c.name: C.structure(base.world("crisis", c.name, 0)) for c in C.DOMAIN.configs}
    assert fps["storm_radial"]["grid_has_loops"] is False and fps["quake_mesh"]["grid_has_loops"] is True
    assert fps["flood_volunteers"]["volunteers"] is True and fps["flood_volunteers"]["hazard"] == "flood"


def test_the_disruption_is_exogenous_and_cascades_through_dependencies():
    e1, e2 = _env(seed=1), _env(seed=1)
    a = e2.agents()[0]
    e2.execute(a, {"tool": "crisis.join", "args": {}})          # a participant's action does not change the event
    for e in (e1, e2):
        e.advance(8 * 3600)
    dmg = [sorted(ev.payload["id"] for ev in e.world.events_range(0) if ev.type == "crisis.damage") for e in (e1, e2)]
    assert dmg[0] and dmg[0] == dmg[1]
    e1.advance(2 * 86400)
    led = C._ledger(e1.world)
    assert led["cascaded"] > 0                                  # undamaged facilities went down with a dependency
    assert not e1.world.check_invariants()


def test_tools_the_coordinator_has_and_what_it_cannot_do():
    env = _env()
    a = env.agents()[0]
    r = env.execute(a, {"tool": "crisis.join", "args": {}})
    assert r["ok"] and r["result"]["crews"]
    sit = env.execute(a, {"tool": "crisis.situation", "args": {}})["result"]
    assert "damage_h" not in str(sit) and all(x["status"] in ("ok", "damaged", "unknown") for x in sit["assets"])
    crew = r["result"]["crews"][0]
    other_crew = next(c["crew"] for c in sit["crews"] if c["owner"] != a)
    assert not env.execute(a, {"tool": "crisis.dispatch", "args": {"crew": other_crew, "asset": "sub0",
                                                                   "task": "inspect"}})["ok"]
    assert env.execute(a, {"tool": "crisis.dispatch", "args": {"crew": crew, "asset": "sub0", "task": "inspect"}})["ok"]
    ag = env.execute(a, {"tool": "crisis.agree", "args": {"operator": "o:powerco"}})
    assert ag["ok"] and ag["result"]["agreed"] is False          # it wants help first
    assert env.reward(a) == 0.0 and env.reward_mode == "observation_only"


def test_household_decisions_come_from_population_sessions():
    env = _env()
    env.advance(2 * 86400)
    prov = K.session_provenance(env.world, C.TABLE, ["crisis.household"])
    assert prov["batches"] > 10 and prov["forced_rows"] == 0, prov


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_coordinator_beats_uncoordinated_response_idle_and_bystander_fail():
    out = {}
    for name, pol in (("ref", C.CrisisCoordinator), ("idle", lambda a, o: K.IdlePolicy()), ("bystander", C.Bystander)):
        env = _env(seed=2)
        a = env.agents()[0]
        K.run_env(env, a, pol(a, C.objective(env.world, a, 1)), days=10, dt=2 * 3600)
        out[name] = env.verify()["evaluation"][a]
    assert out["ref"]["details"]["loss_ratio_vs_uncoordinated"] < 0.9
    assert not out["idle"]["success"] and not out["bystander"]["success"]
    assert out["idle"]["details"]["loss_ratio_vs_uncoordinated"] > 0.95


def test_same_seed_same_log():
    """Determinism (kernel rule): the same seed gives the same state and event hashes."""
    out = []
    for _ in range(2):
        env = K.make_env("crisis", "quake_mesh", 4)
        env.advance(3 * 86400)
        w = env.world
        out.append((w.digest(), w.events_range(w.seq - 1, w.seq)[0].hash))
    assert out[0] == out[1]
