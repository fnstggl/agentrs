"""The Tier 2 domain framework: validation of configurations, registration as world families, mode B provider."""
import pytest

from crucible.domains import base


def _cfg(name, split, structure):
    return base.DomainConfig(name, split, structure, doc=name)


def _domain(configs, name="fw_probe"):
    # a definition only: build reuses the 4.0 'tiny' family recipe, so the framework is tested on a real world
    from crucible.worldgen.generator import FAMILIES

    def build(c, rng, ov):
        return FAMILIES["tiny"].build(rng, ov)
    return base.Domain(name=name, title="probe", spec_part="9.x", required_mechanics=("bank",), configs=tuple(configs),
                       build=build, objective=lambda w, p, d: {"text": "hold"},
                       evaluate=lambda w, p, o: {"success": False, "score": 0.0, "details": {}},
                       reference=lambda w, p, o: None, mechanics=lambda w: {"bank": "bank" in w.modules})


GOOD = [_cfg("a", "train", {"x": 1}), _cfg("b", "train", {"x": 2}), _cfg("c", "train", {"x": 3}),
        _cfg("comp", "eval_compositional", {"x": [1, 2]}), _cfg("struct", "eval_structural", {"y": 1})]


def test_a_valid_domain_registers_world_families_with_v5_splits():
    import crucible
    crucible.load_all_modules()
    d = base.register(_domain(GOOD))
    from crucible.worldgen.generator import FAMILIES
    assert FAMILIES["d5_fw_probe_a"].split == "train"
    assert FAMILIES["d5_fw_probe_comp"].split == "v5eval_compositional"
    assert FAMILIES["d5_fw_probe_struct"].split == "v5eval_structural"
    w = base.world("fw_probe", "b", seed=3)
    assert w.spec.params["domain"] == "fw_probe" and w.spec.params["domain_config"] == "b"
    assert d.mechanics(w) == {"bank": True}


@pytest.mark.parametrize("configs,problem", [
    (GOOD[:2] + GOOD[3:], "three training"),
    (GOOD[:3] + GOOD[4:], "compositionally"),
    (GOOD[:4], "structurally unfamiliar"),
    (GOOD[:2] + [_cfg("c", "train", {"x": 1})] + GOOD[3:], "same structure"),
    (GOOD[:4] + [_cfg("struct", "test", {"z": 1})], "unknown split"),
])
def test_invalid_domains_are_refused(configs, problem):
    assert any(problem in p for p in base.validate(_domain(configs)))
    with pytest.raises(ValueError):
        base.register(_domain(configs, name="fw_bad"))


def test_mode_b_provider_lists_registered_domain_configurations():
    from crucible.substrate import modes
    import crucible.domains.modes  # noqa: F401
    base.register(_domain(GOOD))
    p = [x for x in modes.providers("B") if x.name == "domains"][0]
    assert "fw_probe:a" in p.names()
