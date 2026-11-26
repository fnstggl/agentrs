# Workstream I: Tier 2 domains (`crucible.domains`) - the domain framework and authoring guide

MASTER_SPEC Part 9 (introduction, 9.1-9.16), Part 17 Gate N, SP-9.0, SP-9.16, SP-9.1..SP-9.15; ONE_WORLD_CONTRACTS
section 4 and decisions D-17, D-48, D-49, D-50. Framework API version `domains-framework/2`
(`crucible.domains.base.API_VERSION`), stable since the commit `domains-framework: ready`.

This file is the contract a domain author builds on. The worked example is the test fixture
`tests/v5/_domain_fixture.py` (household retail over a civilisation's supply chain, never counted by the gates).

## 1. What a domain is

A Tier 2 domain is not a task prompt over an unchanged environment. It is a family of worlds for one Part 9 subject:

| part | field of `Domain` | rule |
|---|---|---|
| world recipe | `build(config, seed, overrides) -> recipe` | a civilisation recipe restricted to the subsystems, mechanism variants, couplings, topology and population the configuration needs (`base.civ_recipe`, mode `"C"`, the default), or - where the spec allows a standalone semantic world - any recipe in the same form (mode `"B"`) |
| configurations | `configs: tuple[DomainConfig]` | >= 3 `train`, >= 1 `eval_compositional`, >= 1 `eval_structural`; each with an independent characterisation (`doc`) and a declared `structure` (documentation only: the framework MEASURES structure on the built world) |
| seats | `seats: tuple[Seat]` | the participants a learner (or the reference) occupies; ordinary persons with an endowment created at genesis by module `seat`; several seats make a multi-learner world |
| public objective | `objective(world, principal, difficulty) -> {"text": ..., ...}` | everything in it is public; no hidden parameter, no evaluator data |
| evaluator | `evaluate(world, principal, objective) -> {"success": bool, "score": float, "details": {...}}` | evaluator side only; a function of recorded world state (it must give the same verdict on a world rebuilt from the event log) |
| hidden reference | `reference(principal, objective) -> policy` | a scripted, tool-only policy: it receives observations and its own tool results, NEVER the World; labelled scripted in evidence; never a tool, never named in an observation |
| null baseline | `null_baseline` (default `base.Idle`) | must fail |
| degenerate probes | `degenerate: tuple[Probe]` | >= 1 strategy that games the objective (self-dealing, empty listings, spam, ...); each must fail; the generic `RandomTools` probe runs too |
| mechanics | `mechanics: tuple[Mechanic]` | one probe per required mechanic of the Part 9 section (exact names: `base.PART9[part]["mechanics"]`), naming the recorded event types that exercise it |
| autonomous participants | `actors: tuple[Actor]` | the autonomous participant kinds (population individuals, firms, labs, tenants, ...) and the recorded events their own decisions produce |
| horizon | `horizon_days`, `slowest_effect_days`, `dt` | the evaluation window must exceed the slowest delayed effect of the world (audit I_domains S.3: fleet wear collapses near day 150) |

Everything defaults to `observation_only` (Clarification 2, D-19): the learner gets reward 0.0, observations without
evaluator data, and the evaluator's verdict reaches only `env.verify()`.

## 2. The minimal domain file

`crucible/domains/<name>.py` (cheap to import: definitions only - `crucible.load_all_modules()` imports every file
of the package so that domain kernel modules exist in a fresh process):

```python
from . import base
from .base import Actor, Domain, DomainConfig, Mechanic, Probe, ScriptedReference, Seat, act_

def build(config, seed, overrides):
    return base.civ_recipe(seed, profile=PROFILE_OF[config.name], subsystems=["scientific_research"],
                           modules=["my_module"], population_modules=("pop", "popweb"),
                           module_config={"my_module": {...}})

def objective(world, principal, difficulty):
    return {"text": "Discover ... and demonstrate ...", "difficulty": difficulty, ...public parameters...}

def evaluate(world, principal, obj):            # evaluator side: read authoritative state, never participant claims
    ...
    return {"success": ok, "score": s, "details": {...}}

class ScienceReference(ScriptedReference):      # tool-only: plan(obs) -> [act_(tool, tag, **args)], on_result(...)
    def plan(self, obs): ...
    def on_result(self, tag, result, action): ...

DOMAIN = base.register(Domain(
    name="science", title="Scientific research", spec_part="9.5", required_mechanics=(),   # () = the Part 9 list
    configs=(DomainConfig("lab_grid", "train", {...}, "..."), ...), build=build, objective=objective,
    evaluate=evaluate, reference=ScienceReference,
    mechanics=(Mechanic("Experimental apparatus", ("sci.run", "sci.reading")), ...),
    seats=(Seat("researcher", 2_000_000),),
    actors=(Actor("rival labs", ("sci.paper",)), Actor("households", ("cols.batch:goods.session",))),
    degenerate=(Probe("publish_constant", "publish a constant model", PublishConstant),),
    horizon_days=60, slowest_effect_days=10))
```

`register` validates the definition (refusing it with the list of problems); `base.check(DOMAIN)` builds and runs
it and returns every Gate N / SP-9.0 / SP-9.16 measurement. Run `check` on your domain before you push.

## 3. Building worlds: `civ_recipe`

`civ_recipe(seed, *, subsystems, modules, drop, population, population_modules, profile, module_config, params,
couplings)` takes `crucible.civ.recipe(profile, seed, population=...)` and keeps the substrate (`identity`, `bank`,
`admin`, `artifacts`, `cols`, `space`), the modules of the named Mode C subsystems (`crucible.civ.subsystems
.SUBSYSTEMS`), the population modules, your extra `modules`, and the declared dependencies of all of them. `profile`
may be a profile name or a `crucible.civ.profiles.Profile` (use `dataclasses.replace` on one to vary the supply
chain, freight topology, regimes, couplings - see the fixture). Configurations should differ by such structural
choices, by which subsystems/mechanism variants/participant types exist, and by your module's own discrete
structure; never only by numbers.

Undeclared runtime needs show up as failed processes: e.g. `society` needs the `market` listings index, so the
fixture adds `market`. `check` counts failed processes and invariant violations in every run; a domain is not
"implemented" (N.1) while any occur.

The framework adds the seats (module `seat`) and the params `domain`, `domain_config`, `domain_split` and
`family_split`, and orders the modules by dependencies (`base.ordered_spec`). `base.spec(domain, config, seed,
overrides)` is the WorldSpec of one procedural instance; `overrides` keys `params`, `module_config` and
`drop_modules` are applied after the build, every other key goes to your `build`.

`tier3_spec(domain, config, seed, profile)` merges a configuration into a civilisation (the civilisation's recipe
plus your modules it lacks and the seats). If your modules need different configuration inside a civilisation,
give `Domain.tier3(config, merged_recipe) -> recipe`. SP-9.0 requires that the composed world builds, runs six hours
with no failed process and no invariant error, the seat exists and your evaluator runs on it.

## 4. Structure is measured, not declared (N.2, N.7, SP-9.0; D-49)

`base.fingerprint(world, domain)` is the discrete structure of a built world: module set, participant types
(principal kind with person/org kind), the civilisation structure (`crucible.civ.integration.fingerprint`: couplings,
supply-chain and freight graph shapes, regimes, institutions) and your `Domain.structure(world) -> dict` (measured
on the built world, e.g. the law families of the phenomena, the task-graph topologies). Every value is reduced by
`base.discrete`: a magnitude becomes presence/absence (`"zero"`/`"nonzero"`), a numeric sequence its set of degree
classes, a graph its scale-invariant `graph_shape` (degree classes, cycles, components). So 4 vs 6 regions on a ring
is the same structure; a ring vs a line is not.

* N.2: the number of training configurations with distinct structure digests that are stable across seeds (a
  configuration whose structure changes with the seed is not a configuration).
* N.7: on (path, value) atoms of the fingerprints, an `eval_structural` configuration has at least one atom no
  training configuration has; an `eval_compositional` configuration has no new atom and a combination equal to no
  training configuration.
* SP-9.0 `configurations_differing_only_in_objective_text`: pairs of configurations with equal structure digests
  (plus unstable ones). Different objective texts or difficulty levels over the same structure count here.
* N.6: distinct genesis digests of seeds 0, 1, 2 of every configuration (procedural instances).

## 5. The reference, the null baseline and the probes (N.5, N.n1, SP-9.16)

A policy is any object with `act(obs, tools) -> action | None` and optionally `observe(reward, result, next_obs,
done)` and `burst` (actions per decision point) - the learner protocol of `crucible.experience`. `base.run(domain,
config, seed, policy=...)` builds the world through `crucible.experience.make(2, "<domain>:<config>", seed)` (the
learner's own path, observation_only), drives the first seat with the reference (`"reference"`), the null baseline
(`"null"`), a probe (its name) or `"random_tools"`, other seats idle, for `horizon_days`, and returns the
evaluator's verdict from `env.verify()`.

`ScriptedReference` is the base for references: `plan(obs)` returns actions (`act_(tool, tag, **args)`), results
come back to `on_result(tag, result, action)` (None when refused; the error is in `self.failures`). Reading is done
by calling read-only tools like any learner. Rules, checked by `check`:

* tool-only: `base.reads_world` scans the policy classes' source (AST) for world internals (`.world`, `.state`,
  `.spec`, `.modules`, `.tables`, `system_call`, `rng_for`, `World`, kernel imports, ...). Name your own fields
  differently (`self.stage`, not `self.state`).
* feasible: the reference succeeds on a majority (and at least one) of `reference_seeds` x `reference_configs`
  (default: seeds 0 and 1 of the first training configuration).
* not exposed (N.n1): neither the class/factory names nor the reference's own ordered tool sequence (its first three
  distinct tools in order) may appear in the objective text, the observation's notes or any tool's name, description
  or schema; no tool may be named after the reference; participant sandbox code cannot import `crucible`.
* not guaranteed (SP-9.16): the null baseline fails on every reference instance; every declared degenerate probe
  and `RandomTools` fail.

Typical degenerate probes: paying yourself through a related organisation (independent-revenue evaluators ignore
related payers - `goals.metrics.independent_revenue`), listing without stock, publishing without data, deploying a
no-op controller, firing all staff (when the horizon is shorter than the consequences, the evaluator is wrong:
fix the horizon), keyword-stuffed content.

## 6. Mechanics and autonomous participants (N.3, SP-9.0, SP-9.n)

`Mechanic(name, events, doc, count=None)`: `events` are event-type patterns counted in the event log after genesis:
an exact type (`make.trade`), a prefix (`sci.*`) or a column-batch kind (`cols.batch:goods.session`,
`cols.batch:goods.*`, the population's decisions). `count(world, since_seq) -> int` may add a count measured from
recorded state (rows an applier wrote after `since_seq`). A mechanic is exercised when the union over the reference
runs counts more than zero. Choose events that only happen when the mechanic really operates (a purchase, a
replication report, a failed machine), not set-up events.

`Actor(name, events, doc)` is counted in the null-baseline runs (the seats do nothing), so everything it counts is
autonomous. SP-9.0 also requires decisions of the one population (column batches with decision provenance, D-15)
of an activity other than the population's own bookkeeping (`pop.*`, `attention.*`). A domain without population
activity therefore fails SP-9.0: register a population activity (`crucible.population.register_activity`) or
compose one (`popweb`, `civgoods`, `work`, `pop_social`, ...).

`open_ended` (SP-9.0) is measured on the seat's tool list: >= 10 tools over >= 4 MASTER_SPEC 11.2 categories, an
authoring category (files, content, code, deployments) and a free-form content tool. Civilisation recipes give this
by default (code, artifacts, bank, identity, ...).

## 7. Difficulty without dice (SP-9.16)

Difficulty must come from complexity, scarcity and uncertainty. `base.dice_scan` reads the domain's source files
(`build`/`objective`/`evaluate` files plus `Domain.sources`) and counts Bernoulli draws (`rng.random() < p`,
`rng.binomial(1, p)`) whose expression or assigned name speaks of failure or success (or whose branch raises or
fails). An empirically grounded hazard (a calibrated failure rate) is allowed only with a pragma
`# fidelity: <id>` on the draw's line or the line above, naming a record of docs/v5/FIDELITY_REGISTER.md. List
your kernel module files in `Domain.sources` (relative paths are taken from `crucible/domains/`).

## 8. Per-domain gate test (SP-9.<n>)

Each Part 9 section has its own requirement SP-9.<n> (`tests/v5/test_gate_SP_9_<n>.py`, owned by the domain's
engineer) with metrics `required_mechanics_listed`, `required_mechanics_exercised`, `required_mechanics_missing`,
`observation_only_default`, `reachable_through_experience` (plus section-specific ones: 9.3 difficulty levels toward
10M unique viewers, 9.10 improvement on new problems and the content control order, 9.15 the discovery-to-economy
link by counterfactual removal). The test must measure on integrated runs, never on declarations:

```python
import pytest
from crucible.acceptance.v5 import evidence
from crucible.domains import base
from crucible.domains import science as S          # the domain module (an implementation file)

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_9_5.py"

@pytest.mark.slow
@pytest.mark.timeout(4 * 3600)
def test_SP_9_5_scientific_research_mechanics():
    m, runs = base.section_metrics("science")      # reference runs through experience.make(2, ...), default mode
    # an intervention control: without the replication labs no replication report is recorded
    ablated = base.run("science", "lab_grid", 0, policy="reference",
                       overrides={"module_config": {"sci": {"replicators": {"n": 0}}}})
    m["ablated_replication_events"] = ablated["mechanics"]["counts"]["Independent replication"]
    evidence.record("SP-9.5", m, command=CMD, labels={"scripted": True})
    assert m["ablated_replication_events"] == 0 < m["events_independent_replication"]
    assert m["required_mechanics_listed"] == 8 and m["required_mechanics_missing"] == 0
    assert m["observation_only_default"] and m["reachable_through_experience"]
```

`base.section_metrics(domain, configs=None, seeds=None, days=None) -> (metrics, runs)` returns the five shared
metrics (`required_mechanics_listed`, `required_mechanics_exercised`, `required_mechanics_missing`,
`observation_only_default`, `reachable_through_experience`) plus `events_<mechanic>` counts, measured on the
domain's reference runs (cached in-process, shared with `check`).

For every mechanic, go further than counting where the spec says so: an intervention that removes the mechanic must
change the outcome it is responsible for (e.g. remove replication labs: no replication reports and the published
claims are no longer checked; remove noise: repeated measurements are identical). Put those controls in the same
test (D-26: a measurement that survives replacing your implementation with raising stubs is rejected).

## 9. Gate tests owned by the framework

`tests/v5/test_gate_N.py` (N.1-N.7, N.n1), `tests/v5/test_gate_SP_9_0.py`, `tests/v5/test_gate_SP_9_16.py` iterate
over `base.part9_domains()` with `base.gate_reports()` (one `check` per domain, cached in-process) and
`base.gate_metrics()`: a Part 9 section counts when one of its domains satisfies the criterion; a section without a
domain counts as failing, so these tests FAIL (never skip) while fewer than fifteen sections are covered. Each test
also runs a positive control on the fixture: the planted defect (numbers-only configuration, missing mechanic event,
world-reading reference, spelled-out path, always-succeeding evaluator, inert actor, failure dice) must be detected.

`verdicts(report)["implemented"]` (N.1) requires: definition valid, every configuration builds on three seeds, runs
clean (no run error, failed process or invariant violation), >= 3 distinct stable training structures, >= 3
instances per configuration, a compositional and a structural evaluation configuration, every mechanic exercised,
an independent evaluator (replay-equal, hidden in observation_only, reference succeeds and null fails), a tool-only
feasible reference, no exposure, observation_only by default.

## 10. Reaching a domain

* `crucible.experience.make(2, "<domain>:<config>", seed)` (also `"tier2:<domain>/<config>"`,
  `"d5_<domain>_<config>"`, a bare `"<domain>"` for its first training configuration): learners occupy every seat by
  default (`agents=n`: the first n seats); the objective of each seat is the domain's public objective; `verify()`
  carries the evaluator's verdict.
* execution mode B (`crucible.domains.modes`, SP-2.1): every configuration of every registered non-fixture domain.
* the research distribution registry (`crucible.research.v5.distribution`): one entry per configuration,
  partitioned by split.
* the red-team probes (`crucible.redteam.v5.civ.tier2_probes`, workstream O): one offline and one
  evaluator-isolation probe per evaluation configuration. They still call the earlier contract
  `Domain.reference(world, principal, objective)` and `pol(world, principal)`; the current contract is
  `Domain.reference_policy(principal, objective)` driven by `base.drive(env, {seat: policy}, days=...)` (integration
  note for O in docs/v5/workstreams/I-fw.json). `base._call_policy_factory` drops a world argument for factories of
  the earlier 3-argument form, so no reference ever receives the World.

## 11. Kernel modules in this package

`seats.py` (`seat`), and the domain kernel modules left by earlier waves: `science.py`/`lawlang.py` (`sci`),
`infra_world.py` (`fleet`), `firms.py`/`_kit.py` (`firms`), `learning.py`/`concepts.py`/`common.py` (`learn`).
Their open defects (DEF-I-2, DEF-I-4, DEF-I-5, DEF-I-6) belong to the engineers of the domains that use them.
`_kit.ToolPolicy` reads through `World.query` and is kept for `firms`; new references use `ScriptedReference`.
