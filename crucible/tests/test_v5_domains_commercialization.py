"""Tier 2 domain 9.15 (research commercialization, kernel module `rcomm` over `sci`): the phenomenon is the
population's practice physics, the founder's tools, the investors' public-record rules, and the discovery link."""
import numpy as np
import pytest

from crucible.domains import _dom3 as K
from crucible.domains import base
from crucible.domains import commercialization as R


def test_definition_valid_reference_tool_only_and_measured_structure():
    assert base.validate(R.DOMAIN) == []
    assert base.reads_world(R.DOMAIN.reference_policy("p:founder", {})) == []
    fp = R.structure(base.world("commercialization", "two_conditions", 0))
    assert fp["conditions"] == ["condition", "exposure"] and fp["law_family"] == "two_conditions"


def test_fit_recovers_a_known_optimum_from_clean_readings():
    from crucible.domains.lawlang import compile_expr
    law = compile_expr("1.0 + 4.0 * (setting - (0.2 + 0.05 * condition)) ** 2 + 0.02 * condition",
                       ["setting", "condition"])
    pts = [{"setting": s, "condition": c} for s in np.linspace(0, 1, 7) for c in np.linspace(0, 10, 4)]
    rows = list(zip(pts, law.at(pts)))
    _, advice, info = R.fit_optimum(rows, ["condition"])
    g = compile_expr(advice, ["condition"])
    assert np.allclose(g.at([{"condition": 0.0}, {"condition": 10.0}]), [0.2, 0.7], atol=0.02)
    assert info["sse"] < 1e-4


def test_practice_bills_follow_the_law_and_the_founder_seat_has_only_its_money():
    env = K.make_env("commercialization", "linear_optimum", 0)
    a = env.agents()[0]
    assert a == "p:founder" and env.reward_mode == "observation_only"
    bank = env.world.modules["bank"]
    assert bank.balance(env.world.state, bank.primary_account(env.world.state, a)) == R.DEFAULTS["founder_cash"]
    env.advance(8 * 86400)
    weeks = K.batch_records(env.world, "rc.practice", "weeks")
    assert len(weeks) > 300
    prov = K.session_provenance(env.world, R.TABLE, ["rc.practice"])
    assert prov["forced_rows"] == 0
    st = env.execute(a, {"tool": "rc.investors", "args": {}})["result"]
    assert st["investors"] and "rule" not in str(st)
    bad = env.execute(a, {"tool": "rc.prototype", "args": {"name": "x", "model": "secret_variable + 1"}})
    assert not bad["ok"]


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_the_discovery_founder_succeeds_and_the_common_practice_founder_does_not():
    out = {}
    for disc in (True, False):
        env = K.make_env("commercialization", "linear_optimum", 1)
        a = env.agents()[0]
        common = env.world.modules["rcomm"].c["own_setting"][0]
        pol = R.ScienceFounder(a, R.objective(env.world, a, 1), discovery=disc, common_practice=common)
        K.run_env(env, a, pol, days=100, dt=12 * 3600)
        out[disc] = env.verify()["evaluation"][a]
    assert out[True]["success"], out[True]
    assert not out[False]["success"]
    assert out[True]["details"]["adopters_retained"] > 2 * out[False]["details"]["adopters_retained"]


def test_same_seed_same_log():
    """Determinism (kernel rule): the same seed gives the same state and event hashes (practice weeks, labs'
    replication runs)."""
    out = []
    for _ in range(2):
        env = K.make_env("commercialization", "linear_optimum", 6)
        env.advance(10 * 86400)
        w = env.world
        out.append((w.digest(), w.events_range(w.seq - 1, w.seq)[0].hash))
    assert out[0] == out[1]


def test_evaluator_gives_the_same_verdict_on_a_world_rebuilt_from_its_log():
    """The evaluator reads recorded world state only: on a world rebuilt from the event log alone (World.replay,
    no in-memory history) it gives the same verdict (framework evaluator_report)."""
    rep = base.evaluator_report(R.DOMAIN, "linear_optimum", 0, 10)
    assert rep["ok"] and rep["replay_equal"] and rep["replay_digest_equal"], rep

