"""Tier 2 domain 9.12 (organizational ecosystem, kernel module `orgeco` with `commit`): the shared backbone's
congestion physics, council governance through the institutions module (inst), commitments negotiated with provider organisations, and the evaluator."""
import pytest

from crucible.core.errors import CrucibleError

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
    # the council's acts run only as an approved decision of o:commons (crucible.commerce.institutions)
    assert not env.execute(a, {"tool": "orgs.set_quotas", "args": {"shares": {"o:provider0": 1.0}}})["ok"]
    assert not env.execute(a, {"tool": "inst.propose", "args": {
        "org": O.COMMONS, "action": {"kind": "tool", "tool": "orgs.expand", "args": {}}}})["ok"]   # join first
    assert env.execute(a, {"tool": "inst.join", "args": {"org": O.COMMONS}})["ok"]
    r = env.execute(a, {"tool": "inst.propose", "args": {
        "org": O.COMMONS, "action": {"kind": "tool", "tool": "orgs.expand", "args": {}}, "reason": "capacity"}})
    assert r["ok"] and r["result"]["status"] == "open"
    env.advance(4 * 86400)
    st = env.world.state
    ballots = [b for b in st.find("inst_ballots", "decision", r["result"]["decision"])]
    assert {b["choice"] for b in ballots} == {"yes", "no"}                                    # goals differ
    assert st.tables["inst_decisions"][r["result"]["decision"]]["status"] in ("executed", "rejected")
    # nobody - not even the council's founding owner - moves the council's money outside a decision
    bank = env.world.modules["bank"]
    acct = bank.primary_account(st, O.COMMONS)
    env.execute(a, {"tool": "orgs.contribute", "args": {"cents": 10_000}})
    assert bank.balance(st, acct) >= 10_000
    with pytest.raises(CrucibleError, match="charter"):
        env.world.system_call("bank.transfer", {"from_account": acct, "to": "p:commons-chair", "amount": 5_000},
                              as_principal="p:commons-chair")
    prov = K.session_provenance(env.world, O.TABLE, ["orgs.online"])
    assert prov["forced_rows"] == 0
    assert not env.world.check_invariants()


def test_no_manager_approves_an_act_outside_the_councils_three_acts():
    """A self-dealing proposal (move the council's fund to the convener) gets no manager's vote - a mission
    manager supported any council act before its goal was weighed against the proposal's kind."""
    import numpy as np
    rng = np.random.default_rng(0)
    for goal in O.GOALS:
        for convener in (True, False):
            assert O._goal_vote(goal, "other", 0.0, 0.5, convener, rng)[0] is False
    assert O._goal_vote("mission", "levy", 0.0, 0.0, True, rng)[0] is True
    env = _env(seed=12)
    a = env.agents()[0]
    w = env.world
    st = w.state
    bank = w.modules["bank"]
    acct = bank.primary_account(st, O.COMMONS)
    assert env.execute(a, {"tool": "inst.join", "args": {"org": O.COMMONS}})["ok"]
    r = env.execute(a, {"tool": "inst.propose", "args": {"org": O.COMMONS, "action": {
        "kind": "tool", "tool": "bank.transfer", "args": {"from_account": acct, "to": a, "amount": 40_000}},
        "reason": "reimburse the convener"}})
    did = r["result"]["decision"]
    env.advance(5 * 86400)
    assert st.tables["inst_decisions"][did]["status"] == "rejected"
    votes = [b for b in st.find("inst_ballots", "decision", did) if b["voter"] != a]
    assert votes and all(b["choice"] == "no" for b in votes)


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


def test_same_seed_same_log_with_council_decisions():
    """Determinism (kernel rule): the same seed and the same actions (joining the council, a proposal the
    managers vote on through inst) give the same state and event hashes."""
    digests = []
    for _ in range(2):
        env = _env(seed=2)
        a = env.agents()[0]
        assert env.execute(a, {"tool": "inst.join", "args": {"org": O.COMMONS}})["ok"]
        assert env.execute(a, {"tool": "inst.propose", "args": {
            "org": O.COMMONS, "action": {"kind": "tool", "tool": "orgs.levy", "args": {"cents_per_customer": 200}},
            "reason": "fund"}})["ok"]
        env.advance(5 * 86400)
        w = env.world
        digests.append((w.digest(), w.events_range(w.seq - 1, w.seq)[0].hash))
        assert not w.check_invariants()
    assert digests[0] == digests[1]


def test_evaluator_gives_the_same_verdict_on_a_world_rebuilt_from_its_log():
    """The evaluator reads recorded world state only: on a world rebuilt from the event log alone (World.replay,
    no in-memory history) it gives the same verdict (framework evaluator_report)."""
    rep = base.evaluator_report(O.DOMAIN, "four_goals", 0, 6)
    assert rep["ok"] and rep["replay_equal"] and rep["replay_digest_equal"], rep



@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_every_configuration_has_a_feasibility_witness():
    """The hidden scripted reference succeeds on seed 0 of every configuration (measured 2026-11-27: 20/20 on seeds
    0, 1, 2, 12) - growth_heavy and no_mission_tight had no witness before levies were weighed by their amount and
    the convener priced them from the public telemetry."""
    for c in O.DOMAIN.configs:
        r = base.run("organization", c.name, 0, policy="reference")
        assert r["success"], (c.name, (r["outcome"] or {}).get("details"))
        assert r["labels"]["scripted"] is True
