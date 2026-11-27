"""The Tier 2 domain framework (crucible.domains.base), tested on a real civilisation-backed fixture domain
(tests/v5/_domain_fixture.py: household retail over the supply chain and the one population).

Covers: definition validation (refusals); the one contract through `crucible.experience.make(2, ...)` (seats,
observation_only, reward 0.0, evaluator only in verify(); DEF-I-3); objectives from the goal generator
(crucible.goals); structure measured on built worlds (numbers-only configurations, inert modules and structure hooks
that read declarations are refused; compositional / structural unfamiliarity); procedural instances measured on
genesis CONTENT; the tool-only reference in its own process succeeding where the null baseline and degenerate
strategies fail; mechanics and autonomy counted from verified recorded evidence (generic, bookkeeping, constant and
system evidence count nothing); the isolation, exposure, evaluator-tool and dice scanners with positive controls;
the autonomy ablation of the evaluator; reward reading over whole runs; mode B and module registration.

Each test of a repaired measurement carries the verifier's probe (wave 3b repair pass) as its negative control."""
import json
import os
import sys
import textwrap
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "v5"))

import _domain_fixture as F  # noqa: E402
from crucible.domains import base  # noqa: E402
from crucible.domains.base import Actor, Domain, DomainConfig, Mechanic, Seat  # noqa: E402

DAY = 86400


@pytest.fixture(scope="module")
def fixture_domain():
    d = F.register()
    yield d
    base.unregister(F.NAME)


def _part9(**kw):
    """A Part 9 definition (definition checks only; nothing is built)."""
    mech = tuple(Mechanic(m, (f"sci.m{i}",)) for i, m in enumerate(base.PART9["9.5"]["mechanics"]))
    cfgs = (DomainConfig("a", "train", {"x": "a"}, "a"), DomainConfig("b", "train", {"x": "b"}, "b"),
            DomainConfig("c", "train", {"x": "c"}, "c"), DomainConfig("d", "eval_compositional", {"x": "d"}, "d"),
            DomainConfig("e", "eval_structural", {"x": "e"}, "e"))
    args = dict(name="defcheck", title="t", spec_part="9.5", required_mechanics=(), configs=cfgs, build=F.build,
                goal=F.GOAL, reference=F.RetailReference, mechanics=mech, seats=(Seat("s"),),
                actors=F.DOMAIN.actors, degenerate=F.DOMAIN.degenerate)
    args.update(kw)
    return Domain(**args)


def _null_world(days=2):
    r = base.run(F.NAME, "ring_standard", 0, policy="null", days=days, keep_env=True)
    return r, r.pop("env")


# ---------------------------------------------------------------------------------------------- definitions
def test_part9_lists_are_the_spec_lists():
    assert len(base.PART9) == 15
    assert [len(v["mechanics"]) for v in base.PART9.values()] == [10, 10, 8, 9, 8, 7, 9, 10, 9, 7, 9, 9, 9, 7, 8]
    assert _part9().required_mechanics == base.PART9["9.5"]["mechanics"]
    assert base.validate(_part9()) == []


GENERIC_95 = tuple(Mechanic(m, ("bank.txn",)) for m in base.PART9["9.5"]["mechanics"])        # probe p9 (a)


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
    (lambda d: replace(d, mechanics=GENERIC_95), "substrate-generic"),
    (lambda d: replace(d, mechanics=tuple(Mechanic(m.name, ("sched.*", "proc")) for m in d.mechanics)),
     "bookkeeping"),
    (lambda d: replace(d, mechanics=d.mechanics[:-1] + (Mechanic(d.mechanics[-1].name, d.mechanics[0].events),)),
     "same evidence"),
    (lambda d: replace(d, actors=(Actor("clock", ("artifact.stored",)),)), "substrate-generic"),
    (lambda d: replace(d, degenerate=()), "degenerate-strategy"),
    (lambda d: replace(d, actors=()), "autonomous participant"),
    (lambda d: replace(d, seats=()), "no participant seat"),
    (lambda d: replace(d, horizon_days=10, slowest_effect_days=150), "horizon"),
    (lambda d: replace(d, goal=None), "no objective"),
])
def test_invalid_definitions_are_refused(change, problem):
    d = change(_part9())
    assert any(problem in p for p in base.validate(d)), base.validate(d)
    with pytest.raises(ValueError):
        base.register(replace(d, name="refused"))
    assert "refused" not in base.REGISTRY


def test_policy_factories_never_receive_the_world_or_private_objective_fields():
    seen = []
    base._call_policy_factory(lambda w, p, o: seen.append(w), "p:x", {})      # the earlier 3-argument contract
    base._call_policy_factory(lambda p, o: seen.append(p), "p:x", {})
    assert seen == [None, "p:x"]
    # probe p2 (a): a hint of hidden world state in the objective dict never reaches the policy
    leaky = {"text": "Earn money.", "hint": {"m1": {"secret": 1}}, "success": {"metric": "cash"}}
    assert base.public_objective(leaky) == {"text": "Earn money."}
    assert F.DOMAIN.reference_policy("p:trader", leaky).objective == {"text": "Earn money."}


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
    ev = v["evaluation"][a]
    assert ev["success"] is False and ev["details"]["metric"] == "independent_revenue" and ev["details"]["value"] == 0
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


def test_objectives_come_from_the_independent_goal_generator(fixture_domain):
    rep = base.goal_report(F.DOMAIN)
    assert rep["ok"] and rep["generator"] == "crucible.goals" and rep["distinct_texts"] == 3
    w = base.world(F.NAME, "ring_standard", 0)
    try:
        o1 = F.DOMAIN.make_objective(w, "p:trader", 1)
        assert o1["generator"] == "crucible.goals" and o1["success"]["metric"] == "independent_revenue"
        assert o1 == F.DOMAIN.make_objective(w, "p:trader", 1)                 # deterministic
        draws = {F.DOMAIN.make_objective(w, f"p:x{i}", 3)["params"]["target_cents"] for i in range(8)}
        assert len(draws) > 1                     # the generator's own stream: participants get their own instance
        rng0 = w.rng_for("probe", 1).random()
        F.DOMAIN.make_objective(w, "p:trader", 2)
        assert w.rng_for("probe", 1).random() == rng0            # drawing a goal never touches the world's streams
    finally:
        w.close()
    free = replace(F.DOMAIN, name="fixture_free", goal=None, objective=lambda w, p, d: {"text": "Earn $1."},
                   evaluate=F.never_evaluator)
    base.register(free)
    try:
        g = base.goal_report(free)
        assert g["generator"] == "domain" and g["ok"] is False
    finally:
        base.unregister("fixture_free")


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
    assert rep["varying_tables"]["ring_standard"]                  # generated content differs across seeds
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


def _static_build(config, seed, overrides):
    return {"module_list": ["identity", "bank", "admin"], "modules": {}, "params": {}}


def test_instances_are_distinct_generated_content_not_hash_chains(fixture_domain):
    """Verifier probes p5/p6 'static': seeds of a world without procedural content are ONE instance (only the
    event hash chain differed)."""
    static = replace(F.DOMAIN, name="fixture_static", build=_static_build, structure=None)
    base.register(static)
    try:
        rep = base.structure_report(static)
        assert set(rep["instances"].values()) == {1} and rep["varying_tables"]["ring_standard"] == []
        w0, w1 = (base.world("fixture_static", "ring_standard", s, world_id="x") for s in (0, 1))
        assert w0.digest() != w1.digest() and base.instance_digest(w0) == base.instance_digest(w1)
        w0.close()
        w1.close()
    finally:
        base.unregister("fixture_static")


def _declared_build(config, seed, overrides):          # every configuration builds the ring_standard world
    return F.build(replace(config, name="ring_standard"), seed, overrides)


FLAGS = {"ring_standard": "x", "line_linear": "y", "star_diamond": "z", "ring_linear": "y", "complete_wide": "w"}


def _hook_reads_spec(world):
    return {"a": FLAGS[world.spec.params["domain_config"]]}


def _hook_reads_module_config(world):
    return {"a": str(world.modules["industry"].config.get("chain"))}


def _inert_build(config, seed, overrides):
    extra = {"ring_standard": [], "line_linear": ["calendar"], "star_diamond": ["support"],
             "ring_linear": ["calendar", "support"], "complete_wide": ["studio"]}[config.name]
    r = F.build(replace(config, name="ring_standard"), seed, overrides)
    for m in extra:
        r["module_list"].append(m)
        r["modules"].setdefault(m, {})
    return r


@pytest.mark.parametrize("hook", [_hook_reads_spec, _hook_reads_module_config])
def test_structure_hooks_see_built_state_only(fixture_domain, hook):
    """Verifier probe p1 (b): five configurations that build one world, with a hook that reads the declaration
    the framework put into the spec (or the module configuration), are refused, not counted as five structures."""
    d = replace(F.DOMAIN, name="fixture_declared", build=_declared_build, structure=hook)
    base.register(d)
    try:
        rep = base.structure_report(d, seeds=(0,))
        assert rep["errors"] and rep["train_structures"] == 0, rep["errors"]
        assert "structure_hook" in rep["errors"] and rep["hook_reads_declarations"]
        bare = replace(d, structure=None, name="fixture_declared_bare")
        base.register(bare)
        rep2 = base.structure_report(bare, seeds=(0,))
        assert rep2["train_structures"] == 1 and len(rep2["same_structure_pairs"]) == 10
    finally:
        base.unregister("fixture_declared")
        base.unregister("fixture_declared_bare")
    w = base.world(F.NAME, "ring_standard", 0)
    try:
        view = base.BuiltWorld(w)
        with pytest.raises(base.DeclarationAccess):
            view.spec
        with pytest.raises(base.DeclarationAccess):
            view.modules["industry"]
        assert "industry" in view.modules and F.structure(view)["firm_roles"]
    finally:
        w.close()


def test_inert_modules_are_not_structure(fixture_domain):
    """Verifier probe p1 (c): modules nobody uses (calendar, support, studio) do not make configurations
    structurally different."""
    d = replace(F.DOMAIN, name="fixture_inert_mods", build=_inert_build, structure=None)
    base.register(d)
    try:
        rep = base.structure_report(d, seeds=(0,))
        assert rep["train_structures"] == 1 and rep["structural_eval"] == 0, rep["unfamiliarity"]
        assert not any("studio" in a for a in rep["unfamiliarity"]["complete_wide"]["novel_atoms"])
    finally:
        base.unregister("fixture_inert_mods")


def test_cross_domain_duplicate_worlds_are_counted(fixture_domain):
    """Verifier probe p1 (d): two domains with the same build and different objective text."""
    dup = replace(F.DOMAIN, name="fixture_dup", goal=replace(F.GOAL, text=lambda p: "Another prompt, same world."))
    base.register(dup)
    try:
        ra, rb = base.structure_report(F.DOMAIN, seeds=(0,)), base.structure_report(dup, seeds=(0,))
        pairs = base.cross_domain_same_structure({"a": {"structure": ra}, "b": {"structure": rb}})
        assert len(pairs) == 5 and "a:ring_standard=b:ring_standard" in pairs
    finally:
        base.unregister("fixture_dup")


# ---------------------------------------------------------------------------------------------- evidence
def test_generic_bookkeeping_constant_and_unattributed_evidence_counts_nothing(fixture_domain):
    """Verifier probes p9 (b), p4 'count' and p7: in a NULL run (the seat idles for two days) mechanics mapped to
    bank.txn or the scheduler's events, constant evidence functions, evidence outside the run, and actors declared on
    kernel bookkeeping all count 0, while the real probes and a payload-filtered generic pattern do count."""
    r, env = _null_world(2)
    w, s0 = env.world, r["since_seq"]
    try:
        gen = replace(F.DOMAIN, mechanics=tuple(Mechanic(m.name, ("bank.txn",)) for m in F.MECHANICS))
        sch = replace(F.DOMAIN, mechanics=tuple(Mechanic(m.name, ("sched.*", "sched.add", "proc"))
                                                for m in F.MECHANICS))
        assert base.mechanics_exercised(w, gen, s0)["exercised"] == 0
        assert base.mechanics_exercised(w, sch, s0)["exercised"] == 0
        kinds = sorted({str(e.payload.get("kind")) for e in w.events_range(s0, w.seq) if e.type == "bank.txn"})
        flt = replace(F.DOMAIN, mechanics=(Mechanic("Production", (f"bank.txn{{kind={kinds[0]}}}",)),),
                      required_mechanics=("Production",))
        assert base.mechanics_exercised(w, flt, s0)["counts"]["Production"] > 0
        const = replace(F.DOMAIN, mechanics=F.MECHANICS[:-1] + (
            Mechanic("Production", ("make.never_recorded",), count=lambda w_, s: 1),))
        m = base.mechanics_exercised(w, const, s0)
        assert m["counts"]["Production"] == 0 and m["unverified"] == {"Production": 1}
        genesis = replace(F.DOMAIN, mechanics=F.MECHANICS[:-1] + (
            Mechanic("Production", (), count=lambda w_, s: [1, 2, 3]),))       # events of genesis, not of the run
        m = base.mechanics_exercised(w, genesis, s0)
        assert m["counts"]["Production"] == 0 and m["rejected"] == {"Production": 3}
        good = replace(F.DOMAIN, mechanics=F.MECHANICS[:-1] + (
            Mechanic("Production", (), count=lambda w_, s: base.events_where(w_, s, ("make.run",))),))
        assert base.mechanics_exercised(w, good, s0)["counts"]["Production"] > 0
        twins = replace(F.DOMAIN, mechanics=(F.MECHANICS[0], Mechanic(
            "Freight", (), count=lambda w_, s: base.events_where(w_, s, ("make.run",)))) + F.MECHANICS[2:])
        m = base.mechanics_exercised(w, twins, s0)            # 'Freight' evidenced by the production runs
        assert m["counts"]["Freight"] > 0 and m["counts"]["Production"] == 0
        assert m["indistinct"] == {"Production": "Freight"}
        ghosts = replace(F.DOMAIN, actors=F.DOMAIN.actors + (
            Actor("ghosts", ("ghost.acted",), count=lambda w_, s: 3), Actor("clock", ("artifact.stored", "proc")),
            Actor("bookkeepers", ("cols.batch:pop.deposit",)),
            Actor("system", ("goods.wom",))))
        a = base.actors_active(w, ghosts, s0)
        assert a["counts"]["ghosts"] == 0 and a["counts"]["clock"] == 0 and a["counts"]["bookkeepers"] == 0
        assert a["counts"]["system"] == 0                       # recorded by the system, no participant decided it
        assert a["counts"]["households"] > 0 and a["counts"]["firms"] > 0
        assert set(a["processes"]) >= {"pop.tick", "society.firms"}
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------- runs
@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_reference_succeeds_null_and_degenerate_strategies_fail(fixture_domain):
    ref = base.run(F.NAME, "ring_standard", 0, policy="reference")
    null = base.run(F.NAME, "ring_standard", 0, policy="null")
    selfpay = base.run(F.NAME, "ring_standard", 0, policy="self_funding")
    assert ref["success"] is True and ref["labels"]["scripted"] is True and ref["reward_mode"] == "observation_only"
    assert ref["isolated"] is True and ref["labels"]["isolated_process"] is True and ref["nonzero_rewards"] == 0
    assert null["success"] is False and null["outcome"]["details"]["value"] == 0
    assert selfpay["success"] is False and selfpay["drive"]["agents"]["p:trader"]["ok"] >= 5
    assert ref["mechanics"]["missing"] == [] and ref["proc_failed"] == 0 and ref["invariant_violations"] == 0
    # autonomy from recorded decisions while the seat does nothing: households shop, firms produce and trade
    assert null["actors"]["active"] == 2 and null["population"]["activity_kinds"].get("goods.session", 0) > 0
    # the outcome is the world's: same seed, same events (in-process and in its own process alike)
    again = base.run(F.NAME, "ring_standard", 0, policy="reference", isolate=False)
    assert again["events"] == ref["events"] and again["outcome"] == ref["outcome"]


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_section_metrics_are_measured_on_reference_runs(fixture_domain):
    m, runs = base.section_metrics(F.NAME, seeds=(0,))
    assert m["required_mechanics_listed"] == 5 and m["required_mechanics_exercised"] == 5
    assert m["required_mechanics_missing"] == 0 and m["observation_only_default"] is True
    assert m["reachable_through_experience"] is True and m["events_household_demand"] > 0
    assert m["unverified_evidence"] == "none" and m["indistinct_mechanics"] == "none"
    assert len(runs) == 1 and runs[0]["success"] is True


def test_rewards_are_read_over_whole_runs(fixture_domain):
    """Nit of the repair pass: the observation-only verdict reads every reward of every run; a legacy-reward
    environment shows the leak (positive control)."""
    env = base.make_env(F.NAME, "ring_standard", 0, reward_mode="legacy_shaped_reward")
    try:
        obj = F.DOMAIN.make_objective(env.world, "p:trader", 1)
        dr = base.drive(env, {"p:trader": F.RetailReference("p:trader", base.public_objective(obj))}, days=3,
                        dt=F.DOMAIN.dt)
        assert dr["agents"]["p:trader"]["nonzero_rewards"] > 0
    finally:
        env.close()
    oo = base.run(F.NAME, "ring_standard", 0, policy="reference", days=3, isolate=False)
    assert oo["nonzero_rewards"] == 0 and oo["reward_mode"] == "observation_only"


# ---------------------------------------------------------------------------------------------- isolation
def _helper_peek(x):
    return x.world.state


class HelperPeek(F.RetailReference):
    def plan(self, obs):
        return [] if _helper_peek(self) else []


class ModAttrPeek(F.RetailReference):
    def plan(self, obs):
        from crucible.domains import base as B
        return [] if B.REGISTRY else []


class ObfuscatedPeek(F.RetailReference):
    def plan(self, obs):
        import gc
        w = next(o for o in gc.get_objects() if type(o).__name__ == "Wor" + "ld")
        self.peek = len(getattr(getattr(w, "st" + "ate"), "tab" + "les"))
        return super().plan(obs)


def test_reference_isolation_scanner_detects_world_reads(fixture_domain):
    assert base.reads_world(F.RetailReference("p:trader", {})) == []

    class Peeking(F.RetailReference):
        def plan(self, obs):
            self.world.state.tables["accounts"]                       # what a learner cannot do
            return []
    assert any(".world" in h or ".state" in h for h in base.reads_world(Peeking("p:trader", {})))
    # verifier probes p2 (b) and p4 'reads': a module-level helper, a module attribute, gc + string-built getattr
    assert base.reads_world(HelperPeek("p:trader", {}))
    assert any("REGISTRY" in h or "imports" in h for h in base.reads_world(ModAttrPeek("p:trader", {})))
    hits = base.reads_world(ObfuscatedPeek("p:trader", {}))
    assert any("gc" in h for h in hits) and any("getattr" in h for h in hits), hits
    assert base.sandbox_can_import_crucible() is False


def test_the_reference_runs_in_its_own_process_without_the_world(fixture_domain):
    """Dynamic isolation: a reference that looks for a live World through the garbage collector finds one when run
    in the parent's process (the threat is real) and none in its own process."""
    d = replace(F.DOMAIN, name="fixture_peek", reference=F.PeekingReference)
    base.register(d)
    try:
        inproc = base.run("fixture_peek", "ring_standard", 0, policy="reference", days=0.5, isolate=False)
        iso = base.run("fixture_peek", "ring_standard", 0, policy="reference", days=0.5)
        assert inproc["drive"]["agents"]["p:trader"]["tools"][0] == "id.me"
        assert iso["drive"]["agents"]["p:trader"]["tools"][0] == "bank.accounts" and iso["isolated"]
        # a hint of hidden state in the objective dict does not reach the isolated reference either (probe p2 a)
        leak = replace(d, name="fixture_leak", goal=None, objective=F.leaky_objective, evaluate=F.evaluate)
        base.register(leak)
        lr = base.run("fixture_leak", "ring_standard", 0, policy="reference", days=0.5)
        assert lr["objective"]["hint"] and lr["drive"]["agents"]["p:trader"]["tools"][0] == "bank.accounts"
    finally:
        base.unregister("fixture_peek")
        base.unregister("fixture_leak")

    class NotImportable(F.RetailReference):
        pass
    with pytest.raises(base.ReferenceNotIsolated):
        base.IsolatedPolicy(NotImportable, "p:trader", {})


# ---------------------------------------------------------------------------------------------- exposure
def test_exposure_scanner_detects_canaries_and_spelled_out_trajectories():
    traj = ["bank.accounts", "id.create_org", "biz.register", "bank.open_account", "make.order"]
    canaries = ["RetailReference"]
    assert base.exposure(["Earn $4,000 from households."], canaries, traj)["exposed"] == 0
    hit = base.exposure(["First bank.accounts, then id.create_org and biz.register."], canaries, traj)
    assert hit["exposed"] == 1 and "ordered reference tool sequence" in hit["hits"][0]
    assert base.exposure(['{"name": "x.RetailReference_run"}'], canaries, traj)["exposed"] == 1
    tools = traj + [f"misc.tool{i:02d}" for i in range(25)]
    tr = base.ExposureTrace(tools, canaries)
    tr.feed({"notice": "call bank.accounts, then id.create_org, then biz.register"})
    tr.feed({"help": " ".join(tools)})                                  # a catalogue of every tool is not a recipe
    assert base.trajectory_in_sequences(tr.sequences, traj) == [
        "ordered reference tool sequence ['bank.accounts', 'id.create_org', 'biz.register']"]
    assert len(tr.sequences) == 1


def test_exposure_covers_every_observation_field(fixture_domain, monkeypatch):
    """Verifier probe p3 'expose': the reference's ordered tool path planted in an observation field other than
    `objective` (a `notice`) is found in the null run's observations; a clean world shows nothing."""
    from crucible.experience import kernel as K
    ref = base.run(F.NAME, "ring_standard", 0, policy="reference", days=1, isolate=False)
    traj = ref["drive"]["agents"]["p:trader"]["tools"]
    clean = base.run(F.NAME, "ring_standard", 0, policy="null", days=0.5, queries=True)
    assert base.trajectory_in_sequences(clean["exposure"]["sequences"], traj) == []
    assert clean["queried_tools"] and not clean["exposure"]["canary_hits"]
    head = base._first_distinct(traj)
    notice = "Tip of the day: call " + ", then ".join(head[:4]) + "."
    orig = K.KernelExperienceEnv.observe

    def observe(self, agent):
        o = orig(self, agent)
        o["notice"] = notice
        return o
    monkeypatch.setattr(K.KernelExperienceEnv, "observe", observe)
    planted = base.run(F.NAME, "ring_standard", 0, policy="null", days=0.5)
    assert base.trajectory_in_sequences(planted["exposure"]["sequences"], traj)


def test_evaluator_like_participant_tools_are_flagged(fixture_domain):
    env = base.make_env(F.NAME, "ring_standard", 0)
    try:
        tools = env.tools("p:trader")
        assert base.evaluator_like_tools(tools) == []                       # caps.verify is a world service
        planted = tools + [{"name": "shop.score", "description": "your current objective score",
                            "inputSchema": {"type": "object", "properties": {}}}]
        assert base.evaluator_like_tools(planted) == ["shop.score"]
        assert base._calls_evaluator(env.world, F.DOMAIN, "p:trader") == []
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_the_evaluator_verdict_needs_the_autonomous_participants(fixture_domain):
    """Verifier probe p4 'selfreport': a verdict on a participant-controlled fact (the seat founded an
    organisation) holds with every autonomous decision process frozen - not an independent outcome; the fixture's
    independent revenue does not."""
    ok = base.evaluator_report(F.DOMAIN, "ring_standard", 0)
    assert ok["ok"] and ok["needs_autonomous_participants"] is True and ok["ablation_success"] is False
    assert {"pop.tick", "society.firms"} <= set(ok["ablated_processes"]) and ok["replay_equal"]
    selfrep = replace(F.DOMAIN, name="fixture_selfreport", goal=None, objective=F.objective,
                      evaluate=F.org_exists_evaluator)
    base.register(selfrep)
    try:
        bad = base.evaluator_report(selfrep, "ring_standard", 0, days=4)
        assert bad["replay_equal"] and bad["evaluator_hidden"] and bad["reference_success"]
        assert bad["ablation_success"] is True and bad["needs_autonomous_participants"] is False and not bad["ok"]
    finally:
        base.unregister("fixture_selfreport")


# ---------------------------------------------------------------------------------------------- verdicts
def _rep(**kw):
    rep = {"spec_part": "9.1", "static": [], "run_errors": {}, "process_failures": 0, "invariant_violations": 0,
           "feasible": True, "null_fails": True, "probes_fail": True, "reference_tool_only": True,
           "reference_paths_exposed": 0, "autonomous": True, "open_ended": {"open_ended": True},
           "observation_only": {"ok": True}, "tier3": {"composes": True}, "dice": {"mechanisms": 0},
           "goal": {"ok": True}, "evaluator": {"ok": True},
           "mechanics": {"exercised": 10, "listed": 10, "spec_list_matches": True, "self_reported": False,
                         "missing": []},
           "structure": {"digests": {c: c for c in "abcde"}, "stable_across_seeds": {c: True for c in "abcde"},
                         "train_names": ["a", "b", "c"], "instances": {c: 3 for c in "abcde"}, "errors": {},
                         "same_structure_pairs": [], "core_digests": {},
                         "unfamiliarity": {"d": {"compositional": True}, "e": {"structural": True}}},
           "per_config": {"d": {"split": "eval_compositional", "feasible": True},
                          "e": {"split": "eval_structural", "feasible": True}}}
    rep.update(kw)
    return rep


def test_sp916_verdicts_require_feasibility():
    """Verifier probe p6 'infeasible': a null baseline (or a probe) that fails where nothing can succeed shows
    nothing - SP-9.16 counts it only for a feasible domain."""
    assert base.verdicts(_rep())["implemented"] is True
    v = base.verdicts(_rep(feasible=False))
    assert v["null_fails"] is False and v["probes_ok"] is False and v["implemented"] is False
    m = base.gate_metrics({"x": _rep(feasible=False)})
    assert m["domains_where_null_baseline_fails"] == 0 and m["domains_with_degenerate_strategy_probe"] == 0


def test_evaluation_configurations_count_only_when_they_run_and_are_feasible():
    """Verifier probe p3 'evalcfg': a structurally unfamiliar configuration that fails at runtime (or where the
    reference does not succeed) is not an evaluation configuration."""
    broken = _rep(per_config={"d": {"split": "eval_compositional", "feasible": True},
                              "e": {"split": "eval_structural", "feasible": False}})
    v = base.verdicts(broken)
    assert v["structural_eval"] is False and v["compositional_eval"] is True and v["implemented"] is False


def test_generated_objectives_are_required_for_implemented():
    v = base.verdicts(_rep(goal={"ok": False, "generator": "domain"}))
    assert v["objective_generated"] is False and v["implemented"] is False


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_a_broken_evaluation_configuration_fails_at_runtime(fixture_domain):
    def broken_build(config, seed, overrides):
        r = F.build(config, seed, overrides)
        if config.name == "complete_wide":
            r["module_list"] = [m for m in r["module_list"] if m != "market"]
            r["modules"].pop("market", None)
        return r
    d = replace(F.DOMAIN, name="fixture_evalcfg", build=broken_build)
    base.register(d)
    try:
        r = base.run("fixture_evalcfg", "complete_wide", 0, policy="null", days=2)
        assert r["proc_failed"] > 0
    finally:
        base.unregister("fixture_evalcfg")


# ---------------------------------------------------------------------------------------------- dice
DICE_FORMS = {
    "ok_false_return": "def deploy(rng, p):\n    if rng.random() < p:\n        return {'ok': False}\n"
                       "    return {'ok': True}\n",
    "status_down": "def tick(rng, machine):\n    if rng.random() < 0.05:\n        machine['status'] = 'down'\n",
    "choice_with_p": "def deploy(rng):\n    worked = rng.choice([True, False], p=[0.7, 0.3])\n    return worked\n",
    "binomial_n": "def crashes(rng, n, p_crash):\n    return rng.binomial(n, p_crash)\n",
    "integers_roll": "def attempt(rng):\n    if rng.integers(0, 10) < 3:\n        raise RuntimeError('attempt failed')\n",
    "planted_reference_form": "def deploy(rng, p_fail):\n    if rng.random() < p_fail:\n"
                              "        raise RuntimeError('deployment failed')\n",
}


@pytest.mark.parametrize("form", sorted(DICE_FORMS))
def test_dice_scanner_detects_the_common_forms(tmp_path, form):
    """Verifier probe p2 (c): five of six common failure-dice forms went undetected."""
    p = tmp_path / f"{form}.py"
    p.write_text(DICE_FORMS[form])
    assert base.dice_scan([str(p)])["mechanisms"] == 1


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


def test_dice_scan_covers_the_kernel_modules_of_the_built_world(fixture_domain):
    """Verifier probe p2 (c), coverage: a civilisation-backed domain's mechanisms live in civ kernel modules; the
    scan reads their files (the fixture's world includes economy/industry.py, whose breakdown draw is found)."""
    rep = base.structure_report(F.DOMAIN, seeds=(0,))
    files = rep["world_sources"]
    assert len(files) >= 15 and any(f.endswith(os.path.join("economy", "industry.py")) for f in files)
    srcs = base.domain_sources(F.DOMAIN) + files
    scan = base.dice_scan(srcs)
    assert scan["files"] == len(set(srcs))
    assert any(h["file"].endswith("industry.py") for h in scan["hits"] + scan["exempt"])


# ---------------------------------------------------------------------------------------------- registration
def test_mode_b_lists_every_domain_construction_but_not_fixtures(fixture_domain):
    """Execution mode B (Part 2.1: standalone worlds of a subset of civilisation activities) covers both
    constructions: civilisation restrictions (D-17) and standalone semantic worlds; fixtures are not listed."""
    import crucible.domains.modes  # noqa: F401
    from crucible.substrate import modes
    p = [x for x in modes.providers("B") if x.name == "domains"][0]
    assert not any(n.startswith(F.NAME) for n in p.names())
    civ_like = replace(F.DOMAIN, name="fixture_listed", fixture=False)
    standalone = replace(F.DOMAIN, name="fixture_standalone", fixture=False, mode="B")
    base.register(civ_like)
    base.register(standalone)
    try:
        assert "fixture_listed:ring_standard" in p.names() and "fixture_standalone:ring_standard" in p.names()
        s = base.summary()
        assert s["fixture_listed"]["construction"] == "civ_restriction"
        assert s["fixture_standalone"]["construction"] == "standalone"
        assert base.part9_domains().get("fixture_listed") is None        # spec part "fixture" is not Part 9
    finally:
        base.unregister("fixture_listed")
        base.unregister("fixture_standalone")


def test_domain_kernel_modules_exist_in_a_fresh_process():
    import subprocess
    code = ("import crucible; crucible.load_all_modules(); from crucible.core.kernel import MODULES; "
            "print(sorted(m for m in ('seat','sci','fleet','learn','firms') if m in MODULES))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         env=dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(os.path.dirname(
                             os.path.abspath(__file__))), "src")))
    assert out.stdout.strip() == "['firms', 'fleet', 'learn', 'sci', 'seat']", out.stderr[-500:]


def test_api_version_and_public_names():
    import crucible.domains as D
    assert D.API_VERSION == "domains-framework/3"
    for n in ("GoalFamily", "BuiltWorld", "IsolatedPolicy", "events_where", "goal_verdict", "public_objective",
              "instance_digest", "engaged_modules", "evaluator_like_tools"):
        assert hasattr(D, n), n
    assert json.dumps(base.summary())
