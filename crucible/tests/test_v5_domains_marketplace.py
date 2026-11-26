"""Tier 2 domain 9.13 (digital marketplace, kernel module `mktplace` over `xchg`/`products`/`market`): goods physics,
traders' decisions from public statistics, population buyers, disputes and the evaluator, on real generated
worlds."""
import pytest

from crucible.domains import _dom3 as K
from crucible.domains import base
from crucible.domains import marketplace_domain as M


def _env(config="two_categories", seed=0):
    return K.make_env("marketplace", config, seed)


def test_definition_valid_reference_tool_only_and_measured_structure():
    assert base.validate(M.DOMAIN) == []
    assert base.reads_world(M.DOMAIN.reference_policy("p:agent", {})) == []
    fp = M.structure(base.world("marketplace", "two_incumbents", 0))
    assert fp["incumbents"] == "several" and fp["incumbent_rules"] == ["lenient", "strict"]


def test_grades_and_units_are_measured_from_the_sellers_process():
    assert M.grade_of(0.8) == "A" and M.grade_of(0.5) == "B" and M.grade_of(0.1) == "C"
    env = _env()
    env.advance(10 * 86400)
    st = env.world.state
    units = [u for _, u in st.scan("mkt_units")]
    assert len(units) > 50
    by_seller = {}
    for u in units:
        by_seller.setdefault(u["seller"], []).append(u["quality"])
    for seller, qs in by_seller.items():
        spec = env.world.modules["mktplace"].specs[st.tables["mkt_sellers"][seller]["handle"]]
        assert abs(sum(qs) / len(qs) - spec["q"]) < 0.25
    disputed = [o for _, o in st.scan("xchg_orders") if o["disputed"]]
    assert disputed, "no unit below its claimed grade was disputed"
    prov = K.session_provenance(env.world, M.TABLE, ["mkt.shop"])
    assert prov["forced_rows"] == 0 and prov["batches"] > 10
    assert not env.world.check_invariants()


def test_operator_tools_need_the_exchange_operator():
    env = _env()
    a = env.agents()[0]
    ex = env.execute(a, {"tool": "xchg.create", "args": {"name": "Probe", "take_rate_bps": 300}})["result"]["exchange"]
    assert env.execute(a, {"tool": "mkt.register", "args": {"exchange": ex}})["ok"]
    other = env.add_agent("other")
    assert not env.execute(other, {"tool": "mkt.promote", "args": {"exchange": ex, "cents": 1000}})["ok"]
    r = env.execute(a, {"tool": "mkt.promote", "args": {"exchange": ex, "cents": 4000}})
    assert r["ok"] and r["result"]["reached"] == 100
    d = env.execute(a, {"tool": "mkt.directory", "args": {}})["result"]
    assert ex in {x["exchange"] for x in d["exchanges"]} and "q" not in str(d["exchanges"][0].keys())


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_reference_exchange_succeeds_where_idle_and_a_free_exchange_fail():
    out = {}
    for name, pol in (("ref", M.ExchangeOperator), ("idle", lambda a, o: K.IdlePolicy()), ("free", M.FreeExchange)):
        env = _env(seed=1)
        a = env.agents()[0]
        K.run_env(env, a, pol(a, M.objective(env.world, a, 1)), days=45, dt=6 * 3600)
        out[name] = env.verify()["evaluation"][a]
    assert out["ref"]["success"], out["ref"]
    assert not out["idle"]["success"] and not out["free"]["success"]
    assert out["free"]["details"]["operating_result_15d_cents"] <= 0


def test_same_seed_same_log():
    """Determinism (kernel rule): the same seed gives the same state and event hashes."""
    out = []
    for _ in range(2):
        env = _env("two_categories", 5)
        env.advance(6 * 86400)
        w = env.world
        out.append((w.digest(), w.events_range(w.seq - 1, w.seq)[0].hash))
    assert out[0] == out[1]
