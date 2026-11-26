"""Crucible 5.0 substrate: the execution-mode registry (MASTER_SPEC Part 2.1)."""
import os

import pytest

from crucible.substrate import modes


@pytest.fixture
def scratch_providers():
    added = []

    def reg(mode, name, **kw):
        modes.register(mode, name, **kw)
        added.append((mode, name))
    yield reg
    for m, n in added:
        modes.unregister(m, n)


def test_builtin_mode_a_is_checked_for_real():
    s = modes.status()
    a = s["modes"]["A"]
    prov = {p["name"]: p for p in a["providers"]}
    chk = prov["mechanisms-4.0"]["check"]
    assert chk["ok"] and chk["deterministic"] and chk["responsive"] and chk["n_envs"] == 25
    assert a["operational"] and "A" in s["operational"]
    sim = modes.make("A", "scm_additive", seed=4)
    r1 = sim.run(None, 10)["net"]
    r2 = modes.make("A", "scm_additive", seed=4).run(None, 10)["net"]
    assert (r1 == r2).all()
    for m in ("B", "C"):
        if not s["modes"][m]["providers"]:
            assert not s["modes"][m]["operational"] and "no provider" in s["modes"][m]["reason"]
    assert s["execution_modes"] == len(s["operational"])


def test_a_kernel_world_provider_is_operational_only_if_it_runs_deterministically(scratch_providers):
    from crucible.worldgen.generator import generate

    def make_tiny(name, seed, **kw):
        return generate({"family": name, "overrides": {"params": {"cost_model": "deterministic"}}}, seed)
    scratch_providers("B", "test-ow-tiny", package="tests", names=lambda: ["ow_tiny"], make=make_tiny, kind="world")
    chk = modes.check([p for p in modes.providers("B") if p.name == "test-ow-tiny"][0])
    assert chk["ok"] and chk["deterministic"] and chk["replay_exact"] and chk["events"] > 100

    calls = {"n": 0}

    def make_noisy(name, seed, **kw):          # a world whose spec depends on wall-clock entropy
        calls["n"] += 1
        return generate({"family": "ow_tiny", "overrides": {"params": {"cost_model": "deterministic"}}},
                        int.from_bytes(os.urandom(4), "little"))
    scratch_providers("C", "test-noisy", package="tests", names=lambda: ["noisy"], make=make_noisy, kind="world")
    scratch_providers("C", "test-broken", package="tests", names=lambda: ["x"], make=lambda n, s: 1 / 0,
                      kind="world")
    scratch_providers("C", "test-empty", package="tests", names=lambda: [], make=lambda n, s: None, kind="world")
    by = {p.name: modes.check(p) for p in modes.providers("C")}
    assert not by["test-noisy"]["ok"] and calls["n"] == 2
    assert not by["test-broken"]["ok"] and "ZeroDivisionError" in by["test-broken"]["error"]
    assert not by["test-empty"]["ok"] and by["test-empty"]["n_envs"] == 0
    s = modes.status()
    assert s["modes"]["B"]["operational"]                 # the deterministic world provider counts
    test_c = [p for p in s["modes"]["C"]["providers"] if p["package"] == "tests"]
    assert len(test_c) == 3 and not any(p["check"]["ok"] for p in test_c)


def test_register_validates():
    with pytest.raises(Exception):
        modes.register("D", "x", package="p", names=list, make=lambda n, s: None)
    with pytest.raises(Exception):
        modes.register("A", "x", package="p", names=list, make=lambda n, s: None, kind="robot")


def test_rapid_is_a_mode_a_provider_checked_for_real():
    s = modes.status()
    prov = {p["name"]: p for p in s["modes"]["A"]["providers"]}
    chk = prov["rapid"]["check"]
    assert prov["rapid"]["package"] == "crucible.rapid" and prov["rapid"]["kind"] == "env"
    assert chk["ok"] and chk["deterministic"] and chk["engines"] >= 9 and chk["responsive_engines"] >= 5
    assert chk["n_envs"] >= 50 and all(r["deterministic"] for r in chk["families_checked"].values())
    env = modes.make("A", chk["name"], seed=2)
    assert env.reset() == modes.make("A", chk["name"], seed=2).reset()


def _population_world(name, seed, **kw):
    from crucible.population.worlds import population_world
    return population_world("development", seed, n=200, sites=5, adaptive=False, strategic=False)


def test_mode_c_requires_genuine_shared_state_composition(scratch_providers):
    """A deterministic, exactly replayable world that composes the 4.0 internet with the population is still not
    an integrated civilization: it engages 6 of the 17 Part 2.1 domains."""
    scratch_providers("C", "test-pop-only", package="tests", names=lambda: ["pop"], make=_population_world,
                      kind="world")
    chk = modes.check([p for p in modes.providers("C") if p.name == "test-pop-only"][0])
    assert chk["deterministic"] and chk["replay_exact"] and not chk["ok"] and not chk["integrated"]
    comp = chk["composition"]
    assert not comp["criteria"]["domains"] and len(comp["domains_engaged"]) < modes.C_MIN_DOMAINS
    assert comp["criteria"]["facts"] and "money" in comp["cross_module_kinds"]
    assert set(comp["population_domains"]) <= set(comp["domains_engaged"])


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_every_module_loaded_into_one_world_is_not_an_integrated_civilization():
    """Audit J2: loading (nearly) every module into one kernel world runs, but the v5 systems stay inert - the
    Mode C check must refuse it."""
    import crucible
    from crucible.core.kernel import World, WorldSpec
    from crucible.substrate import facts
    crucible.load_all_modules()
    first = ["identity", "bank", "admin", "artifacts", "cols", "code", "search", "forum", "cloud", "net", "appdb",
             "popsites", "pop", "popweb"]
    excl = {"calendar", "crowd", "feed"}      # alias of cal; the stand-in population; needs pop.register_activity
    names = first + [m for m in facts.shipped_modules() if m not in excl and m not in first]
    w = World.create(WorldSpec("allmods", 7, [{"name": m, "config": {}} for m in facts._with_deps(names)],
                               {"cost_model": "deterministic"}), world_id="allmods")
    w.advance_time(86400)
    comp = modes.composition(w)
    assert len(w.modules) >= 85 and not comp["integrated"]
    assert len(comp["domains_not_engaged"]) >= 5 and not comp["criteria"]["domains"]


def test_a_self_attested_or_trivial_provider_cannot_raise_the_mode_count(scratch_providers):
    """Skeptic S6: a custom check returning ok, and a one-module 'civilization', made execution_modes 3."""
    from crucible.core.kernel import WorldSpec

    def one_module(name, seed, **kw):
        return WorldSpec("one", seed, [{"name": "identity", "config": {}}])
    scratch_providers("B", "test-selfattest", package="tests", names=lambda: ["x"], make=one_module, kind="world",
                      check=lambda p: {"ok": True, "deterministic": True})
    scratch_providers("C", "test-one-module", package="tests", names=lambda: ["one"], make=one_module, kind="world")
    scratch_providers("C", "test-env-as-civ", package="tests", names=lambda: ["e"], make=lambda n, s: None,
                      kind="env")
    by = {p.name: modes.check(p) for p in modes.providers() if p.package == "tests"}
    assert by["test-selfattest"]["provider_check_ok"] and not by["test-selfattest"]["registry_check_ok"]
    assert not by["test-selfattest"]["ok"] and not by["test-selfattest"]["standalone_world"]
    assert by["test-one-module"]["deterministic"] and not by["test-one-module"]["ok"]
    assert by["test-one-module"]["composition"]["domains_engaged"] == []
    assert not by["test-env-as-civ"]["ok"] and "kernel worlds" in by["test-env-as-civ"]["error"]
