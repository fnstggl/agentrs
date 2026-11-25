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
