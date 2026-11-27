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


class TracingGroup(PolicyGroup):
    """A policy group that also lists, per identity, the decisions it recorded and the signal each decision was
    trained with (test instrument; the learning rule is PolicyGroup's)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.recorded: dict[str, list[int]] = {}
        self.trained: list[tuple[str, int, float]] = []

    def record(self, rec):
        if not self.frozen:
            self.recorded.setdefault(rec["identity"], []).append(rec["t"])
        super().record(rec)

    def _split_buffer(self):
        streams, keep = super()._split_buffer()
        self.trained += [(k, r["t"], r["r"]) for k, v in streams.items() for r in v]
        return streams, keep


@pytest.fixture(scope="module")
def run():
    """One short match: two independently updated groups (alpha: 3 identities sharing one REINFORCE policy;
    beta: one PPO identity) and an A2C identity with an unrelated objective."""
    reg = PolicyRegistry()
    env = kernel_env("open_digital", 3)
    groups = {"alpha": TracingGroup("alpha", "reinforce", seed=1), "beta": PolicyGroup("beta", "ppo", seed=2),
              "gamma": PolicyGroup("gamma", "a2c", seed=3)}
    members = [Member("al1", "current", "ctrl:alpha", "alpha"), Member("al2", "current", "ctrl:alpha", "alpha"),
               Member("al3", "current", "ctrl:alpha", "alpha"),
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
    assert len(alpha_ids) == 3 and len(beta_ids) == 1 and not alpha_ids & beta_ids
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


def test_a_shared_training_system_trains_every_decision_with_its_own_signal(run):
    """Verifier probe D (shared-group signal loss): run_learners observes and updates agent by agent, so one
    identity's update can run before the other identities of its group have observed. Every decision of the shared
    group alpha (3 identities) must be trained with the signal its own identity computed for it - never with the
    default 0.0, never another decision's - and no signal may be dropped."""
    m, g = run["m"], run["groups"]["alpha"]
    assert g.signals_dropped == 0 and g.orphans == 0
    assert len(g.identities) == 3 and g.updates >= 2
    checked = nonzero = 0
    for p in g.identities:
        a = m.agents[p]
        own = [e + i for _, e, i in a.signals]            # this identity's k-th signal is for its k-th decision
        idx = {t: k for k, t in enumerate(g.recorded[p])}
        assert len(own) == len(g.recorded[p])
        for ident, t, r in g.trained:
            if ident == p:
                assert r == pytest.approx(own[idx[t]], abs=1e-12), (p, t)
                checked += 1
                nonzero += own[idx[t]] != 0.0
    trained_ids = {(i, t) for i, t, _ in g.trained}
    assert len(trained_ids) == len(g.trained)              # no decision trained twice
    assert checked == len(g.trained) and checked >= 30 and nonzero >= 1
    # every decision is trained once, except each identity's last ones still waiting for the final update
    assert sum(len(v) for v in g.recorded.values()) - checked <= 3 * g.hp["update_every"] + 3
    for gid in ("beta", "gamma"):
        assert run["groups"][gid].signals_dropped == 0 and run["groups"][gid].orphans == 0


def test_signal_waits_for_its_decision_when_another_identity_triggers_the_update():
    """The ordering of run_learners in miniature: three identities of one group decide, then each observes and
    calls maybe_update in turn. The update triggered by the first identity trains only decisions that have their
    signal; the other two identities' latest decisions wait and are trained later with their own signals."""
    rng = np.random.default_rng(4)
    kn = Knowledge("p:x")
    tools = [{"name": f"h{i}.do", "inputSchema": {"type": "object", "properties": {}}} for i in range(4)]
    obs = {"t": 0, "agent": "p:x", "numbers": {}, "tools": tools, "events": []}
    g = TracingGroup("shared", "a2c", seed=8, hp={"update_every": 2})
    ids = ["p:a", "p:b", "p:c"]
    given: dict[tuple, float] = {}
    for step in range(9):
        recs = {}
        for i in ids:
            _, rec, _ = g.decide(i, dict(obs, t=step * 60), kn, rng.normal(size=STATE_DIM).astype(np.float32), None)
            g.record(rec)
            recs[i] = rec
        for k, i in enumerate(ids):
            sig = round(0.1 * (k + 1) + 0.01 * step, 6)          # unique and nonzero per decision
            assert g.add_signal(i, sig, rec=recs[i])
            given[(i, step * 60)] = sig
            g.maybe_update()
    g.update()
    assert g.signals_dropped == 0 and g.orphans == 0 and g.pending() == 0
    assert sorted((i, t) for i, t, _ in g.trained) == sorted(given)
    assert all(r == given[(i, t)] for i, t, r in g.trained)
    # a signal for a decision already trained (or never recorded) is counted, never attached elsewhere
    assert not g.add_signal("p:a", 1.0, rec={"identity": "p:a"}) and g.signals_dropped == 1


def test_learning_controls_train_on_the_same_batches_with_shuffled_and_negated_signals():
    """Q.7 controls (verifier: a no-signal twin cannot change by construction). Twins cloned from the group train on
    exactly its batches with the signals permuted across the batch or negated; their parameters change too. The
    signal-direction measure (expected mean own signal of the chosen tool) rises for the group trained on its real
    signal more than for either twin."""
    from crucible.multiagent import campaign as C
    from crucible.multiagent.policy import PolicyGroup as PG
    kn = Knowledge("p:x")
    tools = [{"name": f"h{i}.do", "inputSchema": {"type": "object", "properties": {}}} for i in range(6)]
    obs = {"t": 0, "agent": "p:x", "numbers": {}, "tools": tools, "events": []}
    for algo in ("reinforce", "a2c", "ppo"):
        g = PG(f"g-{algo}", algo, seed=6)
        init = g.clone("init")
        sh, ng = g.add_control("shuffled"), g.add_control("negated")
        seen_batches = []
        orig_update = sh.update

        def spy(orig=orig_update, c=sh):
            seen_batches.append(sorted(round(r["r"], 9) for v in c.buffer.values() for r in v))
            return orig()
        sh.update = spy
        rng = np.random.default_rng(7)
        hist, recs = [], []
        for _ in range(12):
            batch = []
            for k in range(8):
                x = rng.normal(size=STATE_DIM).astype(np.float32)
                act, rec, _ = g.decide("p:x", obs, kn, x, None)
                g.record(rec)
                tool = (act or {}).get("tool")
                sig = 1.0 if tool == "h0.do" else -0.2
                g.add_signal("p:x", sig, rec=rec)
                batch.append(round(sig, 9))
                hist.append({"tool": tool, "ext": sig, "intr": 0.0})
                recs.append(rec)
            g.update()
            assert seen_batches[-1] == sorted(batch)            # the twin got a permutation of the same signals
        assert sh.hash != init.hash and ng.hash != init.hash and g.hash != init.hash
        means = C.tool_signal_means(hist)
        d0 = C.signal_direction(init, recs, means)
        gains = {k: C.signal_direction(p, recs, means) - d0 for k, p in (("real", g), ("shuffled", sh),
                                                                          ("negated", ng))}
        assert gains["real"] > gains["shuffled"] and gains["real"] > gains["negated"], (algo, gains)
        assert gains["negated"] < 0 < gains["real"], (algo, gains)


def test_capabilities_credit_the_provider_and_only_binding_scarcity():
    """Verifier probe A (SP-12.0 measurement). (1) A learning party that only BUYS from a non-learning seller has
    not created a service used by others; it has once a non-learning party buys from it. (2) Two learning parties
    renting a little compute from a provider far below capacity do not compete for a scarce resource; a learning
    party's rental refused for capacity another party holds is such a competition."""
    from crucible.commerce.integrity import _Driver
    env = kernel_env("open_digital", 3)
    w = env.world
    d = _Driver(w)
    t0 = int(w.now)
    seller = d.join("worldseller", 2_000_000, controller="world:firm")
    alpha = d.join("alpha0", 2_000_000, controller="ctrl:alpha")
    _, _, _, off = I._business(d, seller, "wsell", 4_000)
    d.do(alpha, "prod.buy", offer=off)
    ctrl = {alpha: "ctrl:alpha"}

    def caps():
        inter = AN.interactions(w, ctrl, t0)
        return inter, AN.capabilities(inter, ventures=[], groups_learning={}, strategy_response=None,
                                      parties={"ctrl:alpha"})
    inter, c = caps()
    assert c["trade"]["ok"] and not c["create_services_used_by_others"]["ok"]
    buyer = d.join("worldbuyer", 2_000_000, controller="world:people")
    _, _, _, off2 = I._business(d, alpha, "alphaco", 3_000)
    d.do(buyer, "prod.buy", offer=off2)
    inter, c = caps()
    sv = c["create_services_used_by_others"]
    assert sv["ok"] and sv["providers"] == ["ctrl:alpha"] and sv["cents"] >= 3_000
    env.close()

    env = kernel_env("open_digital", 4)
    w = env.world
    d = _Driver(w)
    t0 = int(w.now)
    a = d.join("alpha0", 50_000_000, controller="ctrl:alpha")
    b = d.join("beta0", 50_000_000, controller="ctrl:beta")
    ctrl = {a: "ctrl:alpha", b: "ctrl:beta"}

    def rent(who, cores):
        return w.execute(who, {"tool": "cloud.rent", "args": {"provider": "o:cloud1", "cores": int(cores),
                                                               "account": d.acct(who)}})
    assert rent(a, 1).ok and rent(b, 1).ok
    inter = AN.interactions(w, ctrl, t0)
    c = AN.capabilities(inter, ventures=[], groups_learning={}, strategy_response=None,
                        parties={"ctrl:alpha", "ctrl:beta"})
    assert not c["compete_for_scarce_resources"]["ok"]
    assert [r["kind"] for r in inter["shared_capacity"]] == ["shared_capacity"]
    free = next(p for p in w.execute(b, {"tool": "cloud.providers", "args": {}}).result
                if p["provider"] == "o:cloud1")["cores_free"]
    while free > 0:                                    # beta takes the rest of the provider's capacity
        assert rent(b, min(64, free)).ok
        free -= min(64, free)
    r = rent(a, 2)
    assert not r.ok and "capacity" in str(r.error).lower()
    inter = AN.interactions(w, ctrl, t0)
    c = AN.capabilities(inter, ventures=[], groups_learning={}, strategy_response=None,
                        parties={"ctrl:alpha", "ctrl:beta"})
    sc = c["compete_for_scarce_resources"]
    assert sc["ok"] and sc["refusals"] == 1 and sc["utilisation_at_refusal_max"] >= 0.98
    assert sc["learning_party_refused"] >= 1 and sc["learning_party_holding"] >= 1 and sc["between_learning_parties"] >= 1
    rec = next(x for x in inter["competitive"] if x["kind"] == "scarce_resource" and x["holder_party"] == "ctrl:beta")
    assert rec["refused_party"] == "ctrl:alpha" and rec["both_learners"]
    env.close()


def test_behaviour_cloning_versions_are_not_counted_as_updates_and_its_order_continues_across_calls():
    """Q.5: versions made by behaviour cloning carry origin 'behaviour_cloning', reinforcement updates 'update'.
    Pre-training epoch by epoch (imitate(epochs=1) n times) is the same training as imitate(epochs=n): one optimiser
    and one minibatch order continued across calls (verifier nit: the order used to restart every call)."""
    from crucible.multiagent.policy import PolicyGroup as PG
    kn = Knowledge("p:x")
    tools = [{"name": f"h{i}.do", "inputSchema": {"type": "object", "properties": {}}} for i in range(5)]
    obs = {"t": 0, "agent": "p:x", "numbers": {}, "tools": tools, "events": []}
    rng = np.random.default_rng(9)
    base = PG("base", "reinforce", seed=3)
    demos = []
    for k in range(40):
        _, rec, _ = base.decide("p:x", obs, kn, rng.normal(size=STATE_DIM).astype(np.float32), None)
        demos.append(dict(rec, ti=1 + (k % 3)))
    a, b = PG("bc", "reinforce", seed=3), PG("bc", "reinforce", seed=3)
    for _ in range(3):
        a.imitate(demos, epochs=1, batch=16)
    b.imitate(demos, epochs=3, batch=16)
    assert a.hash == b.hash
    reg = PolicyRegistry()
    g = PG("g", "reinforce", seed=3)
    reg.register_group(g)
    g.imitate(demos, epochs=1, batch=16)
    for k in range(8):
        act, rec, _ = g.decide("p:x", obs, kn, rng.normal(size=STATE_DIM).astype(np.float32), None)
        g.record(rec)
        g.add_signal("p:x", 1.0 if (act or {}).get("tool") == "h0.do" else -0.3, rec=rec)
    assert g.update()["changed"]
    origins = sorted(v["origin"] for v in reg.versions.values())
    assert origins == ["behaviour_cloning", "initialised", "update"], origins


def test_arguments_from_a_syntax_grammar_are_labelled_scripted_syntax():
    """grammar.py: code files, contract clauses and service inputs come from syntax grammars; a decision that uses
    one records which slots ('scripted_syntax'), and so does the identity's history (the evidence labels say so)."""

    class Pick:
        def choose_tool(self, tools):
            return [t["name"] for t in tools].index("svc.run")

        def choose_value(self, tool, slot, cands, args):
            return 0
    kn = Knowledge("p:x")
    g = PolicyGroup("g", "reinforce", seed=2)
    tool = {"name": "svc.run", "inputSchema": {"type": "object", "properties": {"input": {"type": "object"}},
                                               "required": ["input"]}}
    obs = {"t": 0, "agent": "p:x", "numbers": {}, "tools": [tool], "events": []}
    act, rec, _ = g.decide("p:x", obs, kn, np.zeros(STATE_DIM, np.float32), None, chooser=Pick())
    assert act["tool"] == "svc.run" and rec["scripted_syntax"] == ["input"]
    act, rec, _ = g.decide("p:x", dict(obs, tools=[]), kn, np.zeros(STATE_DIM, np.float32), None)
    assert act is None and rec["scripted_syntax"] == []


def test_a_heldout_family_without_demonstrations_learns_by_self_play_alone_and_its_difference_is_measured(run):
    """Verifier note (Q.6): the behaviour-cloned held-out family shares the training groups' scripted demonstrator.
    The second held-out family never sees a demonstration: gru+a2c from its random initialisation, self-play only,
    no behaviour prior; `heldout_similarity` measures how differently each acts from the training groups."""
    from crucible.multiagent import campaign as C
    reg = PolicyRegistry()
    ho = C.heldout_family(reg, seed=4343, hours=6, source="self_play", drive="reach")
    v = reg.versions[reg.vid(ho)]
    assert (ho.arch, ho.algo, ho.frozen, ho.prior) == ("gru", "a2c", True, None)
    assert v["provenance"]["heldout_source"] == "self_play" and not v["provenance"]["scripted_demonstrator"]
    assert not [x for x in reg.versions.values() if x["origin"] == "behaviour_cloning"]   # never imitated
    assert ho.updates >= 1 and v["origin"].endswith("no demonstrations)")
    with pytest.raises(ValueError):
        C.heldout_family(reg, source="copy")
    m, groups = run["m"], run["groups"]
    recs = [r for a in m.agents.values() if getattr(a, "group", None) is groups["alpha"] for r in a._decisions][:12]
    sim = C.heldout_similarity({"alpha": groups["alpha"], "beta": groups["beta"]},
                               {"heldout_rl": ho, "alpha_copy": groups["alpha"].clone("alpha_copy")}, recs)
    assert sim["situations"] == len(recs) >= 1
    assert sim["heldout_vs_training"]["alpha_copy"]["alpha"] == pytest.approx(0.0, abs=1e-9)
    assert sim["heldout_vs_training"]["heldout_rl"]["alpha"] > 0.0
    assert set(sim["training_vs_training"]) == {"alpha|beta"} and sim["training_pairs_mean"] >= 0.0


def test_strategy_switch_counts_state_changing_responses_and_reports_the_responders_own_noise(run):
    """Respond to changing strategies (SP-12.0): only the switched group's policy differs between the control and
    the switch branch, two unchanged branches agree exactly, a response needs a differing accepted state-changing
    action, and a branch that changes only the responder's own sampling gives the scale of its noise."""
    from crucible.multiagent import campaign as C
    m, groups = run["m"], run["groups"]
    rep = PolicyGroup("replacement", "ppo", "gru", seed=99)
    sw = C.strategy_switch(m, responder="alpha", switched="beta", replacement=rep, hours=6, seed=3)
    assert sw["replicate_divergent_decisions"] == 0
    assert sw["responder_decisions"] >= 1
    assert sw["responded"] == (sw["switch_divergent_state_changing"] > 0)
    assert sw["switch_divergent_state_changing"] <= sw["switch_divergent_decisions"]
    assert sw["own_noise_divergent_decisions"] >= 1          # other sampling generators: other decisions
    for k in ("host_tv_own_noise_vs_control", "tool_tv_switch_vs_control", "tool_tv_own_noise_vs_control"):
        assert 0.0 <= sw[k] <= 1.0
    assert sw["switch_shift_exceeds_own_noise"] == (sw["host_tv_switch_vs_control"] > sw["host_tv_own_noise_vs_control"])
    assert groups["alpha"].frozen is False                   # the original groups are untouched
