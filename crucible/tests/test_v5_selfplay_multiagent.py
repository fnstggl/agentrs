"""crucible.multiagent on a CPU kernel world (open_digital + the v5 commerce/software modules): the learned
tool-level participants, their learning rules, the registry, the league, and the integrity checks. The Tier 3
civilisation runs of the same code are the Gate Q / SP-12 tests (tests/v5/test_gate_Q.py, test_gate_SP_12_*.py)."""
import json

import numpy as np
import pytest
import torch

from crucible.multiagent import analysis as AN
from crucible.multiagent import grammar as G
from crucible.multiagent import integrity as I
from crucible.multiagent import league as L
from crucible.multiagent.harness import Match, Member, kernel_env
from crucible.multiagent.knowledge import Knowledge
from crucible.multiagent.objectives import Objective
from crucible.multiagent.policy import PolicyGroup, STATE_DIM
from crucible.multiagent.registry import PolicyRegistry

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def run():
    """One short match: two independently updated groups (alpha: 2 identities sharing one REINFORCE policy;
    beta: one PPO identity) and an A2C identity with an unrelated objective."""
    reg = PolicyRegistry()
    env = kernel_env("open_digital", 3)
    groups = {"alpha": PolicyGroup("alpha", "reinforce", seed=1), "beta": PolicyGroup("beta", "ppo", seed=2),
              "gamma": PolicyGroup("gamma", "a2c", seed=3)}
    members = [Member("al1", "current", "ctrl:alpha", "alpha"), Member("al2", "current", "ctrl:alpha", "alpha"),
               Member("be1", "current", "ctrl:beta", "beta"),
               Member("ga1", "independent", "ctrl:gamma", "gamma", drive="reach")]
    h0 = {g: groups[g].hash for g in groups}
    m = Match(env, groups, members, registry=reg, seed=3, think_s=3600)
    m.run(36, chunk_hours=6)
    outs = m.finish()
    yield {"m": m, "reg": reg, "groups": groups, "h0": h0, "outs": outs}
    m.close()


def test_slots_come_from_schema_and_own_knowledge():
    kn = Knowledge("p:ann")
    obs = {"t": 0, "agent": "p:ann", "numbers": {"bank.balances.acct_000001": 50_000}, "events": [
        {"type": "deal.offer", "actor": "p:bob", "payload": {"id": "deal_000007", "from": "p:bob", "to": "p:ann"}}],
        "entities": [{"id": "p:ann", "relations": [{"rel": "member_or_admin_of", "to": "o:annco"}]},
                     {"id": "o:bobco", "kind": "organisation", "relations": []}], "tools": [], "text": []}
    kn.ingest_observation(obs)
    rng = np.random.default_rng(0)
    respond = {"name": "deal.respond", "inputSchema": {"type": "object", "properties": {
        "deal": {"type": "string"}, "action": {"type": "string", "enum": ["accept", "reject", "counter"]}},
        "required": ["deal", "action"]}}
    slots = {s.name: s for s in G.tool_slots(respond, kn, rng)}
    assert [c.value for c in slots["deal"].cands] == ["deal_000007"]          # seen under the `deal` noun
    assert {c.value for c in slots["action"].cands} == {"accept", "reject", "counter"}
    acting = {"name": "x.do", "inputSchema": {"type": "object", "properties": {"as_org": {"type": "string"}},
                                              "required": ["as_org"]}}
    assert [c.value for c in G.tool_slots(acting, kn, rng)[0].cands] == ["o:annco"]   # only organisations it runs
    unknown = {"name": "y.do", "inputSchema": {"type": "object", "properties": {"widget": {"type": "string"}},
                                               "required": ["widget"]}}
    assert G.tool_slots(unknown, kn, rng) is None                              # it cannot name what it never saw
    mk = {"name": "code.create_repo", "inputSchema": {"type": "object", "properties": {
        "name": {"type": "string", "pattern": "[a-z][a-z0-9_-]{0,39}"}}, "required": ["name"]}}
    name = G.tool_slots(mk, kn, rng)[0].cands[0].value
    assert name.startswith("ann") and G._fits(name, mk["inputSchema"]["properties"]["name"])
    money = G.slot_cands("venture.form", "capital_cents", {"type": "integer", "minimum": 0}, True, kn, rng, {})
    assert max(c.value for c in money) <= 25_000                                # never more than half its money


def test_learning_rule_is_evidence_scaled_and_a_no_signal_control_does_not_change():
    from crucible.multiagent.policy import PolicyGroup as PG
    rng = np.random.default_rng(1)
    kn = Knowledge("p:x")
    tools = [{"name": f"h{i}.do", "inputSchema": {"type": "object", "properties": {}}} for i in range(6)]
    obs = {"t": 0, "agent": "p:x", "numbers": {}, "tools": tools, "events": []}
    for algo in ("reinforce", "a2c", "ppo"):
        live, ctl = PG(f"live-{algo}", algo, seed=5), PG(f"ctl-{algo}", algo, seed=5)
        assert live.hash == ctl.hash
        for g, signal in ((live, 1.0), (ctl, 0.0)):
            for k in range(8):
                x = rng.normal(size=STATE_DIM).astype(np.float32)
                act, rec, h = g.decide("p:x", obs, kn, x, None)
                g.record(rec)
                g.add_signal("p:x", signal if (act or {}).get("tool") == "h0.do" else -0.1 * signal)
            g.update()
        assert live.updates == 1 and live.hash != ctl.hash
        assert ctl.updates == 0 and ctl.skipped_no_signal == 1 and ctl.hash == PG(f"c2-{algo}", algo, seed=5).hash


def test_shared_training_and_independent_groups_update_from_their_own_experience(run):
    g, reg, h0 = run["groups"], run["reg"], run["h0"]
    for gid in g:
        assert g[gid].updates >= 1 and g[gid].hash != h0[gid]
    # alpha's updates pooled both of its identities; beta's only its own
    alpha_ids = {i for u in reg.updates if u["group"] == "alpha" and u["changed"] for i in u["identities"]}
    beta_ids = {i for u in reg.updates if u["group"] == "beta" and u["changed"] for i in u["identities"]}
    assert len(alpha_ids) == 2 and len(beta_ids) == 1 and not alpha_ids & beta_ids
    assert I.independence(g, run["m"].agents)["violations"] == []


def test_participants_act_through_ordinary_tools_without_reward(run):
    m = run["m"]
    for p, a in m.agents.items():
        assert a.nonzero_env_rewards == 0
        state_changing = [h for h in a.history if h["ok"] and not h["read_only"]]
        assert len(state_changing) >= 2, p
    rec = m.registry.matches[-1]
    assert rec["roster"] and rec["digest"] == m.world.digest()
    kinds = m.registry.field_kinds()
    assert all(kinds[k] for k in ("versions", "model_hashes", "identities", "provenance", "actions", "outcomes",
                                  "updates", "composition"))
    v = [x for x in m.registry.versions.values() if x["origin"] == "update"]
    assert v and all(x["parent"] in m.registry.versions and x["provenance"]["decisions"] > 0 for x in v)


def test_crossplay_between_versions_replays_exactly(run):
    reg, g = run["reg"], run["groups"]
    for x in g.values():
        reg.checkpoint(x)
    recs = L.crossplay(reg, reg.vid(g["alpha"]), reg.vid(g["beta"]), world="kernel:open_digital", seeds=[11],
                       hours=8)
    again = L.replay(L.MatchSpec.from_dict(recs[0]["spec"]), reg, recs[0])
    assert again["mismatches"] == []
    lr = L.league_ratings(recs, {})
    assert lr["pairs"] == 1 and set(lr["ratings"]) == {"ctrl:A", "ctrl:B"}


def test_collusion_battery_detects_planted_schemes_and_not_clean_controls():
    env = kernel_env("open_digital", 5)
    gs = {"a": PolicyGroup("a", "reinforce", seed=1), "b": PolicyGroup("b", "ppo", seed=2)}
    r = I.collusion_battery(env.world, groups_for_hidden_state=gs)
    assert r["planted"] == 6 and r["undetected"] == 0 and r["clean_controls_flagged"] == 0
    env.close()


def test_shared_reward_audit_flags_a_team_reward_and_not_own_signals(run):
    m = run["m"]
    a = I.shared_reward_audit(m)
    assert a["shared_reward_channels"] == 0 and a["signals_checked"] > 0
    # positive control: one learner given a team objective over two parties' cash
    al = next(x for p, x in m.agents.items() if p.endswith("al1"))
    keep = al.objective
    al.objective = I.TeamObjective(m.world, [p for p in m.agents])
    try:
        b = I.shared_reward_audit(m)
    finally:
        al.objective = keep
    assert b["shared_reward_channels"] >= 1


def test_controlled_identity_transfers_are_not_counted():
    env = kernel_env("open_digital", 6)
    w = env.world
    from crucible.commerce.integrity import _Driver
    d = _Driver(w)
    a1 = d.join("ci1", 1_000_000, controller="ctrl:one")
    a2 = d.join("ci2", 1_000_000, controller="ctrl:one")
    b1 = d.join("ci3", 1_000_000, controller="ctrl:two")
    org, prod, plan, off = I._business(d, a1, "cico", 3_000)
    d.do(a2, "prod.buy", offer=off)          # same controller: a transfer between controlled identities
    d.do(b1, "prod.buy", offer=off)          # an independent customer
    acc = I.controlled_identity_accounting(w, {a1: "ctrl:one", a2: "ctrl:one", b1: "ctrl:two"})
    assert acc["naive_same_controller_cents"] >= 3_000
    assert acc["controlled_identity_transfers_counted"] == 0
    from crucible.multiagent.evaluate import revenue
    assert revenue(w, org)["independent"] == 3_000
    env.close()


def test_interactions_and_consequences_are_measured_between_parties(run):
    m = run["m"]
    ctrl = {p: mm.controller for p, mm in m.members.items()}
    inter = AN.interactions(m.world, ctrl, m.t0)
    for r in inter["cooperative"] + inter["competitive"]:
        assert len(set(r["parties"])) >= 2
    cons = AN.consequences(m.world, ctrl, m.seq0)
    assert cons["events_visible_to_other_parties"] >= 1
    assert all(a != b for k in cons["by_pair"] for a, b in [k.split("->")])


def test_canary_tap_exposes_only_the_protocol_its_participant_implements():
    """A participant without `update` (a semantic generalist) stays without one behind the tap: the driver must not
    call an update the participant does not have (regression: the SP-12.3b civilisation run crashed on it)."""
    class NoUpdate:
        principal = "p:x"
        history: list = []

        def act(self, obs, tools):
            return None

        def observe(self, reward, result, next_obs, done):
            self.seen = next_obs

    class WithUpdate(NoUpdate):
        def update(self):
            return {"updated": True}
    log: list = []
    tap = I.CanaryTap(NoUpdate(), "ctrl:x", {"cnryA": "ctrl:y"}, log)
    assert not hasattr(tap, "update")
    assert tap.act({"t": 0, "events": [{"payload": {"body": "cnryA"}}]}, []) is None
    tap.observe(0.0, {"ok": True}, {"t": 1}, False)
    assert tap.scanned == 3 and [x["canary"] for x in log] == ["cnryA"]
    assert I.CanaryTap(WithUpdate(), "ctrl:x", {}, []).update() == {"updated": True}


def test_semantic_participant_record_tells_read_only_calls_from_actions():
    """A semantic generalist 'acts' (SP-12.1 composition) only with accepted state-changing calls: its record carries
    the read-only flag and the error code from the participant's own record of each call."""
    from crucible.multiagent.agents import SemanticAgent

    class Participant:                       # the SemanticParticipant protocol the agent relies on
        def __init__(self):
            self.history = []

        def reset(self, principal, rng):
            self.principal = principal

        def act(self, obs, tools):
            return {"tool": tools[0]["name"], "args": {}}

        def observe_result(self, action, result):
            ro = action["tool"] == "world.find_tools"
            self.history.append({"tool": action["tool"], "ok": bool(result.get("ok")), "read_only": ro})
    a = SemanticAgent("p:s", Participant(), every=1)
    a.act({"t": 0}, [{"name": "world.find_tools"}])
    a.observe(0.0, {"ok": True, "result": {}}, {"t": 60}, False)
    a.act({"t": 60}, [{"name": "make.build_facility"}])
    a.observe(0.0, {"ok": False, "error": {"code": "invalid_action"}}, {"t": 120}, False)
    assert [(h["tool"], h["ok"], h["read_only"], h["error"]) for h in a.history] == [
        ("world.find_tools", True, True, None), ("make.build_facility", False, False, "invalid_action")]
    assert not any(h["ok"] and not h["read_only"] for h in a.history)


def test_behaviour_regularised_update_pulls_towards_the_pretrained_prior_and_keeps_it_private():
    """After `set_prior` every update adds KL(pi || prior) over the tool distribution: with the same experience and
    signal, a group with a larger KL coefficient ends closer to its pre-trained prior; the prior is serialised with
    the group (clones reproduce it), is a private copy (no tensor shared with a clone) and does not move; a
    no-signal control is still not trained at all; behaviour cloning keeps one optimiser across calls."""
    from crucible.multiagent.policy import PolicyGroup as PG
    kn = Knowledge("p:x")
    tools = [{"name": f"h{i}.do", "inputSchema": {"type": "object", "properties": {}}} for i in range(6)]
    obs = {"t": 0, "agent": "p:x", "numbers": {}, "tools": tools, "events": []}
    base = PG("base", "reinforce", seed=5)
    rng = np.random.default_rng(2)
    demos = []
    for k in range(48):
        x = rng.normal(size=STATE_DIM).astype(np.float32)
        _, rec, _ = base.decide("p:x", obs, kn, x, None)
        rec = dict(rec, ti=1 + (k % 2))          # demonstrated: h0.do / h1.do (index 0 is wait)
        demos.append(rec)
    for _ in range(3):
        base.imitate(demos, epochs=1, batch=16)
    assert len(base._bc_opt) == 1
    steps = [s["step"] for s in next(iter(base._bc_opt.values())).state.values()]
    assert steps and all(int(s) == 9 for s in steps)          # 3 calls x 3 batches, one optimiser
    base.set_prior()
    prior0 = {k: v.clone() for k, v in base.prior.state_dict().items()}
    dist = {}
    for coef in (0.0, 5.0):
        g = base.clone(f"kl{coef}")
        g.hp["kl_prior"] = coef
        assert g.prior is not None and all(torch.equal(g.prior.state_dict()[k], prior0[k]) for k in prior0)
        assert not {p.data_ptr() for p in g.prior.parameters()} & {p.data_ptr() for p in base.prior.parameters()}
        r2 = np.random.default_rng(3)
        situations, kls = [], []
        for _ in range(6):
            for k in range(8):
                x = r2.normal(size=STATE_DIM).astype(np.float32)
                act, rec, _ = g.decide("p:x", obs, kn, x, None)
                situations.append(rec)
                g.record(rec)
                g.add_signal("p:x", 1.0 if (act or {}).get("tool") == "h5.do" else -0.2)
            info = g.update()
            assert info["changed"]
            kls.append(info["kl_prior"])
        assert (max(kls) > 0) == (coef > 0), kls            # 0 at the first update: pi == prior
        assert all(torch.equal(g.prior.state_dict()[k], prior0[k]) for k in prior0)   # the anchor does not move
        dist[coef] = np.mean([AN.tv_distance(base.tool_probs_of(r), g.tool_probs_of(r)) for r in situations])
    assert dist[5.0] < dist[0.0], dist
    ctl = base.clone("ctl")
    for k in range(8):
        _, rec, _ = ctl.decide("p:x", obs, kn, rng.normal(size=STATE_DIM).astype(np.float32), None)
        ctl.record(rec)
        ctl.add_signal("p:x", 0.0)
    h = ctl.hash
    assert not ctl.update()["changed"] and ctl.hash == h
    other = base.frozen_copy("b", seed=99)              # own generators; the prior travels with the checkpoint
    assert other.prior is not None and I.independence({"a": base, "b": other}, {})["violations"] == []
