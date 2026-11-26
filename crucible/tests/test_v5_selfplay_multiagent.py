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
