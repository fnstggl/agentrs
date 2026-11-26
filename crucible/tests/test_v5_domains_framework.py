"""The Tier 2 domain framework (crucible.domains.base), tested on a real civilisation-backed fixture domain
(tests/v5/_domain_fixture.py: household retail over the supply chain and the one population).

Covers: definition validation (refusals), the one contract through `crucible.experience.make(2, ...)` (seats,
observation_only, reward 0.0, evaluator only in verify(); DEF-I-3), structure measured on built worlds (numbers-only
configurations are refused, compositional / structural unfamiliarity), procedural instances, the tool-only
reference succeeding where the null baseline and degenerate strategies fail, mechanics and autonomy counted from
recorded events, the isolation and dice scanners with positive controls, mode B and module registration."""
import os
import sys
import textwrap
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "v5"))

import _domain_fixture as F  # noqa: E402
from crucible.domains import base  # noqa: E402
from crucible.domains.base import Domain, DomainConfig, Mechanic, Seat  # noqa: E402


@pytest.fixture(scope="module")
def fixture_domain():
    d = F.register()
    yield d
    base.unregister(F.NAME)


def _part9(**kw):
    """A Part 9 definition (definition checks only; nothing is built)."""
    mech = tuple(Mechanic(m, ("bank.txn",)) for m in base.PART9["9.5"]["mechanics"])
    cfgs = (DomainConfig("a", "train", {"x": "a"}, "a"), DomainConfig("b", "train", {"x": "b"}, "b"),
            DomainConfig("c", "train", {"x": "c"}, "c"), DomainConfig("d", "eval_compositional", {"x": "d"}, "d"),
            DomainConfig("e", "eval_structural", {"x": "e"}, "e"))
    args = dict(name="defcheck", title="t", spec_part="9.5", required_mechanics=(), configs=cfgs, build=F.build,
                objective=F.objective, evaluate=F.evaluate, reference=F.RetailReference, mechanics=mech,
                seats=(Seat("s"),), actors=F.DOMAIN.actors, degenerate=F.DOMAIN.degenerate)
    args.update(kw)
    return Domain(**args)


# ---------------------------------------------------------------------------------------------- definitions
def test_part9_lists_are_the_spec_lists():
    assert len(base.PART9) == 15
    assert [len(v["mechanics"]) for v in base.PART9.values()] == [10, 10, 8, 9, 8, 7, 9, 10, 9, 7, 9, 9, 9, 7, 8]
    assert _part9().required_mechanics == base.PART9["9.5"]["mechanics"]
    assert base.validate(_part9()) == []


@pytest.mark.parametrize("change,problem", [
    (lambda d: replace(d, configs=d.configs[1:]), "three training"),
    (lambda d: replace(d, configs=d.configs[:3] + d.configs[4:]), "compositionally"),
    (lambda d: replace(d, configs=d.configs[:4]), "structurally unfamiliar"),
    (lambda d: replace(d, configs=d.configs + (DomainConfig("f", "test", {"x": "f"}, "f"),)), "unknown split"),
    (lambda d: replace(d, configs=d.configs + (DomainConfig("a", "train", {"x": "z"}, "z"),)), "names repeat"),
    (lambda d: replace(d, required_mechanics=("Hidden laws",), mechanics=(Mechanic("Hidden laws", ("x",)),)),
     "differ from MASTER_SPEC"),
    (lambda d: replace(d, mechanics=lambda w: {"Hidden laws": True}), "not a self-report"),
    (lambda d: replace(d, mechanics=d.mechanics[1:]), "no event probe"),
    (lambda d: replace(d, degenerate=()), "degenerate-strategy"),
    (lambda d: replace(d, actors=()), "autonomous participant"),
    (lambda d: replace(d, seats=()), "no participant seat"),
    (lambda d: replace(d, horizon_days=10, slowest_effect_days=150), "horizon"),
])
def test_invalid_definitions_are_refused(change, problem):
    d = change(_part9())
    assert any(problem in p for p in base.validate(d)), base.validate(d)
    with pytest.raises(ValueError):
        base.register(replace(d, name="refused"))
    assert "refused" not in base.REGISTRY


def test_policy_factories_never_receive_the_world():
    seen = []
    base._call_policy_factory(lambda w, p, o: seen.append(w), "p:x", {})      # the earlier 3-argument contract
    base._call_policy_factory(lambda p, o: seen.append(p), "p:x", {})
    assert seen == [None, "p:x"]


# ---------------------------------------------------------------------------------------------- the learner path
def test_learner_occupies_the_seat_in_an_observation_only_civ_world(fixture_domain):
    from crucible.core import evaluator_only as EO
    from crucible.experience import OBSERVATION_ONLY, make
    from crucible.experience.registry import domains
    assert F.NAME in domains()
    env = make(2, f"{F.NAME}:ring_standard", 0)
    a = env.agents()[0]
    assert a == "p:trader" and env.reward_mode == OBSERVATION_ONLY and EO.hidden(env.world)
    assert {"pop", "industry", "civgoods", "seat"} <= set(env.world.modules)
    obs = env.observe(a)
    assert obs["objective"].startswith("Earn at least") and obs["resources"]["balance_cents"] == 600_000
    env.advance(6 * 3600)
    assert env.reward(a) == 0.0
    v = env.verify()
    assert v["evaluation"][a]["success"] is False and "independent_revenue_cents" in v["evaluation"][a]["details"]
    assert "independent_revenue" not in str(env.observe(a)["numbers"])
    from crucible.experience.registry import domain_and_config
    assert domain_and_config(f"tier2:{F.NAME}/line_linear")[1].name == "line_linear"
    assert domain_and_config(f"d5_{F.NAME}_star_diamond")[1].name == "star_diamond"
    # several learners: every seat, or the first n
    two = replace(F.DOMAIN, name="fixture_two", seats=(Seat("trader", 600_000), Seat("rival", 600_000)))
    base.register(two)
    try:
        assert make(2, "fixture_two:ring_standard", 0).agents() == ["p:trader", "p:rival"]
        assert make(2, "fixture_two:ring_standard", 0, agents=1).agents() == ["p:trader"]
    finally:
        base.unregister("fixture_two")


def test_spec_is_a_restricted_civilisation_recipe_with_seats(fixture_domain):
    r = base.recipe(F.NAME, "line_linear", 4)
    assert {"industry", "freight", "pop", "civgoods", "seat"} <= set(r["module_list"])
    assert not {"posts", "feed", "sci", "fleet", "learn"} & set(r["module_list"])
    assert r["params"]["domain"] == F.NAME and r["params"]["family_split"] == "train"
    sp1, sp2 = base.spec(F.NAME, "line_linear", 4), base.spec(F.NAME, "line_linear", 4)
    assert sp1.hash() == sp2.hash() and sp1.hash() != base.spec(F.NAME, "line_linear", 5).hash()
    names = [m["name"] for m in sp1.modules]
    assert names.index("identity") < names.index("bank") < names.index("seat")
    sp = base.spec(F.NAME, "line_linear", 4, {"params": {"probe_flag": 1}, "drop_modules": ["market"]})
    assert sp.params["probe_flag"] == 1 and "market" not in [m["name"] for m in sp.modules]


# ---------------------------------------------------------------------------------------------- structure
def test_discrete_structure_ignores_magnitudes():
    ring4 = base.graph_shape([f"r{i}" for i in range(4)], [(f"r{i}", f"r{(i + 1) % 4}") for i in range(4)], False)
    ring6 = base.graph_shape([f"r{i}" for i in range(6)], [(f"r{i}", f"r{(i + 1) % 6}") for i in range(6)], False)
    line = base.graph_shape([f"r{i}" for i in range(4)], [(f"r{i}", f"r{i + 1}") for i in range(3)], False)
    star = base.graph_shape([f"r{i}" for i in range(5)], [("r0", f"r{i}") for i in range(1, 5)], False)
    assert ring4 == ring6 and len({ring4, line, star}) == 3
    assert base.discrete({"n": 4, "m": 0.3}) == base.discrete({"n": 9, "m": 7.0}) != base.discrete({"n": 0, "m": 1})


def test_structure_is_measured_on_built_worlds(fixture_domain):
    rep = base.structure_report(F.DOMAIN)
    assert rep["errors"] == {} and rep["train_structures"] == 3 and rep["same_structure_pairs"] == []
    assert all(rep["stable_across_seeds"].values()) and min(rep["instances"].values()) == 3
    assert rep["unfamiliarity"]["ring_linear"]["compositional"] is True
    assert rep["unfamiliarity"]["ring_linear"]["novel_atoms"] == []
    assert rep["unfamiliarity"]["complete_wide"]["structural"] is True
    assert rep["compositional_eval"] == 1 and rep["structural_eval"] == 1
    # positive control: a configuration that changes only numbers (6 regions instead of 4 on the same ring and
    # chain) is a different declaration but the same measured structure
    F.SHAPES["ring_standard6"] = {"chain": "standard", "regions": 6, "links": "ring"}
    try:
        planted = replace(F.DOMAIN, name="fixture_numbers", configs=F.DOMAIN.configs + (
            DomainConfig("ring_standard6", "train", dict(F.SHAPES["ring_standard6"]), "six regions"),))
        base.register(planted)
        rep2 = base.structure_report(planted, seeds=(0,))
        assert "ring_standard=ring_standard6" in rep2["same_structure_pairs"] and rep2["train_structures"] == 3
    finally:
        base.unregister("fixture_numbers")
        F.SHAPES.pop("ring_standard6", None)


# ---------------------------------------------------------------------------------------------- runs
@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_reference_succeeds_null_and_degenerate_strategies_fail(fixture_domain):
    ref = base.run(F.NAME, "ring_standard", 0, policy="reference")
    null = base.run(F.NAME, "ring_standard", 0, policy="null")
    selfpay = base.run(F.NAME, "ring_standard", 0, policy="self_funding")
    assert ref["success"] is True and ref["labels"]["scripted"] is True and ref["reward_mode"] == "observation_only"
    assert null["success"] is False and null["outcome"]["details"]["independent_revenue_cents"] == 0
    assert selfpay["success"] is False and selfpay["drive"]["agents"]["p:trader"]["ok"] >= 5
    assert ref["mechanics"]["missing"] == [] and ref["proc_failed"] == 0 and ref["invariant_violations"] == 0
    # autonomy from recorded events while the seat does nothing: households shop, firms produce and trade
    assert null["actors"]["active"] == 2 and null["population"]["activity_kinds"].get("goods.session", 0) > 0
    # the outcome is the world's: same seed, same events
    again = base.run(F.NAME, "ring_standard", 0, policy="reference")
    assert again["events"] == ref["events"] and again["outcome"] == ref["outcome"]


# ---------------------------------------------------------------------------------------------- scanners
def test_reference_isolation_scanner_detects_world_reads(fixture_domain):
    assert base.reads_world(F.RetailReference("p:trader", {})) == []

    class Peeking(F.RetailReference):
        def plan(self, obs):
            self.world.state.tables["accounts"]                       # what a learner cannot do
            return []
    assert any(".world" in h or ".state" in h for h in base.reads_world(Peeking("p:trader", {})))
    assert base.sandbox_can_import_crucible() is False


def test_exposure_scanner_detects_canaries_and_spelled_out_trajectories():
    traj = ["bank.accounts", "id.create_org", "biz.register", "bank.open_account", "make.order"]
    canaries = ["RetailReference"]
    assert base.exposure(["Earn $4,000 from households."], canaries, traj)["exposed"] == 0
    hit = base.exposure(["First bank.accounts, then id.create_org and biz.register."], canaries, traj)
    assert hit["exposed"] == 1 and "ordered reference tool sequence" in hit["hits"][0]
    assert base.exposure(['{"name": "x.RetailReference_run"}'], canaries, traj)["exposed"] == 1


def test_dice_scanner_flags_arbitrary_failure_probabilities(tmp_path):
    p = tmp_path / "dice_mod.py"
    p.write_text(textwrap.dedent('''
        def deploy(ctx, rng, p_fail):
            if rng.random() < p_fail:
                raise RuntimeError("deployment failed")
            ok = rng.random() > 0.3
            return ok

        def weather(rng):
            return rng.random() < 0.2

        def hazard(rng, rate):
            # fidelity: not-a-registered-id
            failed = rng.random() < rate
            return failed
    '''))
    r = base.dice_scan([str(p)])
    assert r["mechanisms"] == 3 and {h["line"] for h in r["hits"]} == {3, 5, 13}, r
    assert base.dice_scan([os.path.join(os.path.dirname(F.__file__), "_domain_fixture.py")])["mechanisms"] == 0


# ---------------------------------------------------------------------------------------------- registration
def test_mode_b_lists_real_domains_not_fixtures(fixture_domain):
    import crucible.domains.modes  # noqa: F401
    from crucible.substrate import modes
    p = [x for x in modes.providers("B") if x.name == "domains"][0]
    assert not any(n.startswith(F.NAME) for n in p.names())
    probe = replace(F.DOMAIN, name="fixture_listed", fixture=False)
    base.register(probe)
    try:
        assert "fixture_listed:ring_standard" in p.names()
        assert base.part9_domains().get("fixture_listed") is None        # spec part "fixture" is not Part 9
    finally:
        base.unregister("fixture_listed")


def test_domain_kernel_modules_exist_in_a_fresh_process():
    import subprocess
    code = ("import crucible; crucible.load_all_modules(); from crucible.core.kernel import MODULES; "
            "print(sorted(m for m in ('seat','sci','fleet','learn','firms') if m in MODULES))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         env=dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(os.path.dirname(
                             os.path.abspath(__file__))), "src")))
    assert out.stdout.strip() == "['firms', 'fleet', 'learn', 'sci', 'seat']", out.stderr[-500:]
