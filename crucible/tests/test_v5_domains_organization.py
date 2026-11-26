"""Tier 2 domain 9.12 (organizational ecosystem, kernel module `orgeco` with `commit`): the shared backbone's
congestion physics, council governance, commitments negotiated with provider organisations, and the evaluator."""
import pytest

from crucible.domains import _dom3 as K
from crucible.domains import base
from crucible.domains import organization as O


def _env(config="four_goals", seed=0):
    return K.make_env("organization", config, seed)


def test_definition_valid_reference_tool_only_and_measured_structure():
    assert base.validate(O.DOMAIN) == []
    assert base.reads_world(O.DOMAIN.reference_policy("p:convener", {})) == []
    fp = O.structure(base.world("organization", "no_mission_tight", 0))
    assert fp["mission_member"] is False


def test_congestion_is_shared_and_quotas_bind_only_through_the_council():
    env = _env()
    a = env.agents()[0]
    assert a == "p:convener"
    env.advance(20 * 86400)
    hours = [ev.payload for ev in env.world.events_range(0) if ev.type == "orgs.hour"]
    assert any(h["congestion_success"] < 1.0 for h in hours)
    congested = [h for h in hours if h["congestion_success"] < 1.0]
    assert all(len({round(v, 4) for v in h["success"].values()}) == 1 for h in congested[:20])   # no quotas yet
    assert not env.execute(a, {"tool": "orgs.propose", "args": {"kind": "expansion"}})["ok"]    # join first
    assert env.execute(a, {"tool": "orgs.join", "args": {}})["ok"]
    r = env.execute(a, {"tool": "orgs.propose", "args": {"kind": "expansion"}})
    assert r["ok"]
    env.advance(4 * 86400)
    votes = [v for _, v in env.world.state.scan("org_votes")]
    assert {v["yes"] for v in votes} == {True, False}                                        # goals differ
    prov = K.session_provenance(env.world, O.TABLE, ["orgs.online"])
    assert prov["forced_rows"] == 0
    assert not env.world.check_invariants()


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_convener_succeeds_where_idle_and_a_solo_funder_fail():
    out = {}
    for name, pol in (("ref", O.CouncilConvener), ("idle", lambda a, o: K.IdlePolicy()), ("solo", O.SoloFunder)):
        env = _env(seed=1)
        a = env.agents()[0]
        K.run_env(env, a, pol(a, O.objective(env.world, a, 1)), days=40, dt=6 * 3600)
        out[name] = env.verify()["evaluation"][a]
    assert out["ref"]["success"], out["ref"]
    assert not out["idle"]["success"] and not out["solo"]["success"]
