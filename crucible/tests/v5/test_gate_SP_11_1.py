"""SP-11.1: the universal interaction contract (observe, execute, observe_since) across tiers, carrying the 13
observation kinds of MASTER_SPEC 11.1 with exact numbers.

Kinds are counted only where real observations carry them (`kinds_present`), never because a field exists. Semantic
representations come from the shared encoders of workstream D (`crucible.semantic.encode`) on v5 artifacts; tiers
count when a world of that tier ran through the interface (Tier 2: crucible.domains, Tier 3: crucible.civ)."""
import json

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.experience import OBSERVATION_KINDS, kinds_present, validate_observation
from crucible.experience.registry import civ_available, domains, make

pytestmark = pytest.mark.timeout(900)
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_11_1.py"


def _exercise_kernel(env) -> list[dict]:
    """Two participants interact: messages (text, artifacts), a self-experiment (delayed, caused consequences),
    payments (exact money); every observation and observe_since page is returned."""
    a, b = env.agents()[:2]
    w = env.world
    acct = w.modules["bank"].primary_account(w.state, a)
    seen = [env.observe(a), env.observe(b)]
    env.execute(a, {"tool": "exp.register", "args": {
        "hypothesis": "paying 12.34 dollars lowers my balance by exactly 1234 cents",
        "measure": {"tool": "bank.accounts", "where": {"owner": "$self"}, "field": "balance", "aggregate": "sum"},
        "prediction": {"op": "decrease", "min_change": 1233}, "deadline_in_s": 7200}})
    env.execute(a, {"tool": "bank.transfer", "args": {"from_account": acct, "to": b, "amount": 1234,
                                                      "memo": "an exact amount"}})
    env.execute(a, {"tool": "mail.send", "args": {"to": [b], "subject": "invoice settled",
                                                  "body": "I paid you 12.34 dollars for the report."}})
    env.execute(a, {"tool": "forum.post", "args": {"board": "general", "title": "market notes",
                                                   "body": "prices moved a little this morning"}})
    for _ in range(4):
        env.advance(3600)
        seen += [env.observe(a), env.observe(b)]
    ev, cur = env.observe_since(a, 0)
    seen.append({"observe_since": ev, "cursor": cur})
    return seen


def test_SP_11_1_universal_contract():
    per_world = {}
    failures = 0
    union: set[str] = set()
    tiers = set()
    exact_ok = True
    # Tier 1: 4.0 hidden-mechanism facility
    env = make(1, "mech:scm_delay", 2, "observation_only", validate=False)
    a = env.agents()[0]
    obs = [env.observe(a)]
    sp = env.execute(a, {"tool": "rig.spec", "args": {}})["result"]
    for k in range(6):
        env.execute(a, {"tool": "rig.set", "args": {"settings": {lv["name"]: lv["min"] + (lv["max"] - lv["min"]) * k / 6
                                                                 for lv in sp["levers"]}}})
        env.advance(env.tick_s)
        obs.append(env.observe(a))
    rep = env.execute(a, {"tool": "rig.read", "args": {}})["result"]["report"]
    exact_ok &= obs[-1]["numbers"]["rig.seats.r1.revenue_cents"] == rep["revenue_cents"] and \
        isinstance(obs[-1]["numbers"]["rig.seats.r1.revenue_cents"], int)
    ev, _ = env.observe_since(a, 0)
    k1 = set().union(*(kinds_present(o) for o in obs))
    failures += sum(len(validate_observation(o)) for o in obs)
    per_world["tier1 mech:scm_delay"] = sorted(k1)
    union |= k1
    tiers.add(1)
    # Tier 1: v5 rapid families, a single participant and a team (the receiver of a message)
    for fam in ("net_routing", "team_commons"):
        env = make(1, f"rapid:{fam}", 2)
        ags = env.agents()
        if len(ags) > 1:
            env.execute(ags[1], {"tool": "rapid.message", "args": {"to": ags[0], "text": "meet at 3"}})
        obs = []
        for _ in range(5):
            env.advance(env.tick_s)
            obs.append(env.observe(ags[0]))
        kr = set().union(*(kinds_present(o) for o in obs))
        failures += sum(len(validate_observation(o)) for o in obs)
        exact_ok &= all(isinstance(o["t"], int) and o["t"] % env.tick_s == 0 for o in obs)
        per_world[f"tier1 rapid:{fam}"] = sorted(kr)
        union |= kr
    # a kernel world with the v5 modules (artifacts, media, encoders): a participant renders a text artifact, the
    # observation carries its semantic representation (not one of the v5 tiers: counted for kinds only)
    from crucible.experience.kernel import KernelExperienceEnv
    from crucible.redteam.v5.worlds import v5_spec
    env = KernelExperienceEnv(v5_spec("all", 3), 3, agents=["writer", "reader"], endowment=2_000_000)
    w5 = env.agents()[0]
    r = env.execute(w5, {"tool": "media.render_text", "args": {"text": "Routing prices moved this morning."}})
    assert r["ok"], r
    env.advance(3600)
    o5 = [env.observe(w5)]
    k5 = kinds_present(o5[0])
    failures += len(validate_observation(o5[0]))
    per_world["kernel v5 modules"] = sorted(k5)
    union |= k5
    # kernel worlds (4.0 families: not one of the v5 tiers, counted for kinds only)
    env = make(0, "open_digital", 3, "observation_only", agents=["ann", "bob"])
    seen = _exercise_kernel(env)
    obs = [o for o in seen if "numbers" in o]
    bal = [v for k, v in obs[-2]["numbers"].items() if k.startswith("bank.balances.")]
    exact_ok &= bool(bal) and all(isinstance(v, int) for v in bal)
    kk = set().union(*(kinds_present(o) for o in obs))
    failures += sum(len(validate_observation(o)) for o in obs)
    per_world["kernel open_digital (4.0)"] = sorted(kk)
    union |= kk
    # v5 tiers through the same make()
    for dn in sorted(domains())[:2]:
        env = make(2, dn, 0, "observation_only", agents=["ann", "bob"])
        obs = [o for o in _exercise_kernel(env) if "numbers" in o]
        k2 = set().union(*(kinds_present(o) for o in obs))
        failures += sum(len(validate_observation(o)) for o in obs)
        per_world[f"tier2 {dn}"] = sorted(k2)
        union |= k2
        tiers.add(2)
    if civ_available():
        import crucible.civ as civ
        env = make(3, sorted(getattr(civ, "PROFILES", {"default": None}))[0], 0, "observation_only",
                   agents=["ann", "bob"])
        obs = [o for o in _exercise_kernel(env) if "numbers" in o]
        k3 = set().union(*(kinds_present(o) for o in obs))
        failures += sum(len(validate_observation(o)) for o in obs)
        per_world["tier3 civ"] = sorted(k3)
        union |= k3
        tiers.add(3)
    missing = sorted(set(OBSERVATION_KINDS) - union)
    metrics = {"observation_kinds": len(union), "tiers_supported": len(tiers), "schema_failures": failures,
               "exact_numbers_checked": bool(exact_ok), "kinds_missing": ",".join(missing) or "none"}
    record("SP-11.1", metrics, command=CMD, notes=f"kinds per world: {json.dumps(per_world)}; missing kinds: {missing}; "
           "v5 tiers that ran: " + str(sorted(tiers)) + " (Tier 2 needs v5 domains, Tier 3 crucible.civ)")
    assert failures == 0 and exact_ok
    assert len(union) >= 13 and len(tiers) >= 3, metrics
