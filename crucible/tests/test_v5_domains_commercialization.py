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



def test_a_verdict_asked_after_a_same_instant_action_sees_the_action():
    """The evaluator cache (`_dom3.cached_evaluator`, used by all five dom3 domains) must not return the verdict of
    before an action taken at the same simulated instant: the founder moves all its cash out and `env.verify()` at
    the same instant reports the empty account (it reported the old cash and solvent=True when the cache was keyed by
    the instant alone). Read-only tool calls (command bookkeeping only) keep the cached verdict."""
    env = K.make_env("commercialization", "linear_optimum", 0)
    a = env.agents()[0]
    w = env.world
    env.advance(3600)
    before = env.verify()["evaluation"][a]["details"]
    bank = w.modules["bank"]
    acct = bank.primary_account(w.state, a)
    cash = int(bank.available(w.state, acct))
    assert before["cash_cents"] == cash > 0 and before["checks"]["solvent"]
    calls = []
    inner = R.evaluate.__wrapped__
    counted = K.cached_evaluator(lambda ww, p, o: calls.append(ww.seq) or inner(ww, p, o))
    obj = R.objective(w, a, 1)
    counted(w, a, obj)
    for _ in range(3):
        assert env.execute(a, {"tool": "rc.status", "args": {}})["ok"]
        counted(w, a, obj)
    assert len(calls) == 1                       # read-only calls left nothing but command bookkeeping
    t0 = w.now
    r = env.execute(a, {"tool": "bank.transfer", "args": {"from_account": acct, "to": "o:hosting", "amount": cash}})
    assert r["ok"] and w.now == t0
    after = env.verify()["evaluation"][a]["details"]
    assert after["cash_cents"] == 0 and not after["checks"]["solvent"]
    assert counted(w, a, obj)["details"]["cash_cents"] == 0 and len(calls) == 2
    assert after == inner(w, a, R.objective(w, a, 1))["details"]
    env.close()


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_subscribers_of_a_suspended_product_lapse_and_are_not_counted_as_retained():
    """A product whose hosting goes unpaid is suspended; its users fall back on their own practice and, once the
    month they paid for has run out, their subscriptions lapse. The evaluator (and the investors' public record)
    count a retained adopter only while its last payment covers the present - it used to count a suspended
    product's subscribers as paying forever. The test builds an advisor from the hidden law's optimum (the test is
    evaluator side), deploys and promotes it, lets subscribers renew, then empties the founder's account."""
    env = K.make_env("commercialization", "linear_optimum", 2)
    a = env.agents()[0]
    w = env.world
    mod = w.modules["rcomm"]
    c0 = mod.c["conditions"][0]
    xs = np.linspace(c0["lo"], c0["hi"], 6)
    grid = np.linspace(0, 1, 201)
    best = [grid[int(np.argmin(mod.law([{"setting": float(s), c0["name"]: float(x)} for s in grid])))] for x in xs]
    b1, b0 = np.polyfit(xs, best, 1)
    model = f"clip({b0:.4f} + {b1:.4f} * {c0['name']}, 0, 1)"
    ex = lambda tool, **args: env.execute(a, {"tool": tool, "args": args})  # noqa: E731
    proto = ex("rc.prototype", name="advisor", model=model)["result"]["prototype"]
    env.advance(5 * 86400)
    prod = ex("rc.launch", prototype=proto, name="Advisor", price_cents_month=1200)["result"]["product"]
    assert ex("rc.advertise", product=prod, cents=40_000)["ok"]
    env.advance(50 * 86400)
    obj = R.objective(w, a, 1)
    mid = R.evaluate(w, a, obj)["details"]
    assert mid["adopters_retained"] > 0, mid
    bank = w.modules["bank"]
    acct = bank.primary_account(w.state, a)
    for _ in range(3):                                   # empty the account before the next monthly hosting bill
        cash = int(bank.available(w.state, acct))
        if cash:
            assert ex("bank.transfer", from_account=acct, to="o:devshop", amount=cash)["ok"]
        env.advance(4 * 86400)
    assert w.state.tables["rc_products"][prod]["status"] == "suspended"
    env.advance(42 * 86400)
    st = w.state
    subs = [s for _, s in st.scan("rc_subs") if s["product"] == prod]
    status = {s["status"] for s in subs}
    assert "lapsed" in status, status
    late = [s for s in subs if s["status"] == "paying" and s["paid_until"] + 7 * 86400 < w.now]
    assert late == []
    end = R.evaluate(w, a, obj)["details"]
    assert end["adopters_retained"] == sum(1 for s in subs if R.paying_now(s, w.now)) == 0, end
    assert mod.evidence(st, a)["renewed_subscribers"] == 0
    env.close()
