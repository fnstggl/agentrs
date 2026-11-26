"""crucible.civ: generator, profiles, couplings, goods activity, provenance, accounting, Mode C, Tier 3 entry."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import crucible  # noqa: E402

crucible.load_all_modules()

from crucible import civ  # noqa: E402
from crucible.civ import accounting, chains, couplings, integration, profiles, subsystems  # noqa: E402
from crucible.substrate import columns  # noqa: E402


@pytest.fixture(scope="module")
def world():
    import torch
    torch.set_num_threads(1)
    w = civ.generate("balanced", 3)
    w.advance_time(86400)
    yield w
    w.close()


def test_profiles_are_structural_and_cover_part_10_1():
    assert len(profiles.PROFILES) >= 10
    assert set(profiles.PART_10_1) <= {p.part_10_1 for p in profiles.PROFILES.values()}
    keys = [profiles.structural_digest(p) for p in profiles.PROFILES.values()]
    assert len(set(keys)) == len(keys)
    for p in profiles.PROFILES.values():
        assert set(p.couplings) <= set(couplings.CATALOGUE)


def test_recipe_is_one_world_without_second_populations():
    r = civ.recipe("industrial_first", 2)
    mods = set(r["module_list"])
    assert {"pop", "popweb", "civgoods", "civlinks", "civgen", "bank", "identity", "industry", "sci", "cloud"} <= mods
    assert not mods & set(subsystems.EXCLUDED), mods & set(subsystems.EXCLUDED)
    for p in r["pending"]:
        assert p["module"] not in mods and p["waits_for"]
    sp = civ.spec("industrial_first", 2)
    assert sp.hash() == civ.spec("industrial_first", 2).hash()
    order = [m["name"] for m in sp.modules]
    from crucible.core.kernel import MODULES
    for i, m in enumerate(order):
        assert all(d in order[:i] for d in MODULES[m].depends), m
    assert r["modules"]["learn"]["n"] == 0          # no second population of learners


def test_every_profile_composes_one_world():
    """Every profile's spec resolves (no missing dependency), with ONE shared semantic space (the kernel module
    `space` of crucible.semantic.space), the population's social internet and its work activities."""
    from crucible.core.kernel import MODULES
    from crucible.semantic.space import store
    assert MODULES["space"] is store.SpaceRecords
    for name in civ.PROFILES:
        sp = civ.spec(name, 1)
        mods = [m["name"] for m in sp.modules]
        assert {"space", "pop", "aud", "feed", "pop_social", "posts", "blog", "work", "ventures", "rivals"} <= set(mods)
        assert mods.index("civgen") < mods.index("ventures"), name
        cfg = {m["name"]: m["config"] for m in sp.modules}
        assert cfg["pop"]["dims"]["H"] >= 19 and cfg["ventures"]["incumbents"], name


def test_one_day_runs_clean(world):
    assert world.check_invariants() == []
    evs = world.events_range(0)
    types = {e.type for e in evs}
    assert "make.energy" in types and "links.energy" in types       # datacenters draw from the grid
    assert any(e.type == "cols.batch" and e.payload["kind"] == "goods.session" for e in evs)
    acc = accounting.Accountant(world)
    acc.feed(evs)
    assert all(not v for v in acc.check().values()), acc.check()


def test_science_physics_is_one_fact(world):
    binds = world.modules["civlinks"].config["bindings"]
    assert binds
    assert world.modules["civlinks"].check_invariants(world) == []
    ind, sci = world.modules["industry"], world.modules["sci"]
    b = binds[0]
    q = 100 * ind.quality(ind.phys["models"][b["model"]], 0.3, 0.6)
    assert abs(float(sci.truth(b["phenomenon"], [{"speed": 0.3, "temp": 0.6}])[0]) - q) < 1e-6


def test_coupling_removal_and_placebo_are_operator_only(world):
    w = world.fork("t-fork")
    assert couplings.active(w, "goods_wom")
    s0 = w.seq
    couplings.remove(w, "goods_wom")
    assert not couplings.active(w, "goods_wom") and w.seq > s0
    s1 = w.seq
    w.system_call("admin.links_placebo", {"n": 3})
    assert sum(1 for e in w.events_range(s1) if e.type == "links.placebo") == 3
    agent = w.system_call("admin.join", {"handle": "probe", "kind": "agent"})["principal"]
    r = w.execute(agent, {"tool": "admin.links_remove", "args": {"coupling": "goods_supply"}})
    assert not r.ok and couplings.active(w, "goods_supply")


def test_population_buys_goods_with_own_money(world):
    bought = columns.col(world, "goods", "bought_n")
    evs = [e for e in world.events_range(0) if e.type == "make.retail_sold"]
    assert int(bought.sum()) == len(evs)
    from crucible.substrate import ids
    for e in evs[:20]:
        assert ids.index(e.payload["buyer"]) is not None      # buyers are population members
        assert e.actor == e.payload["buyer"]


def test_provenance_reaches_other_subsystems(world):
    pv = chains.build(world, 0)
    dep = next(s for s, e in pv.events.items() if e.type == "cloud.deployment")
    stages = chains.trace(world, dep, pv=pv)
    subs = [s["subsystem"] for s in stages]
    assert "computing_infrastructure" in subs and len(set(subs)) >= 3, subs
    cross = integration.cross_system(pv)
    assert cross["cross_system_event_types"] > 0


def test_fingerprint_differs_between_profiles():
    a = civ.generate("fragmented_social", 4)
    b = civ.generate("interconnected_communities", 4)
    fa, fb = integration.fingerprint(a), integration.fingerprint(b)
    assert fa != fb and integration.structural_distance(fa, fb) > 0
    assert fa["social_locality"] == "fragmented" and fb["social_locality"] == "interconnected"


def test_tier3_entry_defaults_to_observation_only():
    from crucible.civ import experience
    env = experience.make("tier3:digital_first", 1)
    assert env.reward_mode == "observation_only"
    a = env.agents()[0]
    out = env.execute(a, {"tool": "world.clock", "args": {}})
    assert out["ok"] and env.reward(a) == 0.0
    obs = env.observe(a)
    assert "score" not in obs and "success" not in obs


def test_mode_c_provider_registered():
    from crucible.substrate import modes
    modes.discover()
    ps = modes.providers("C")
    assert ps and ps[0].package == "crucible.civ"
    assert "civ:balanced" in ps[0].names()
