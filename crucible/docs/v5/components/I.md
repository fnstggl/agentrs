# Workstream I: Tier 2 domains (`crucible.domains`) - the domain framework and authoring guide

MASTER_SPEC Part 9 (introduction, 9.1-9.16), Part 7 (`IndependentlyGenerateGoal`), Part 17 Gate N, SP-9.0, SP-9.16,
SP-9.1..SP-9.15; ONE_WORLD_CONTRACTS section 4 and decisions D-17, D-48, D-49, D-50. Framework API version
`domains-framework/3` (`crucible.domains.base.API_VERSION`), commit `domains-framework: ready v2` (the wave-3b
repair pass; what changed since `domains-framework/2` and how to migrate is section 12).

This file is the contract a domain author builds on. The worked example is the test fixture
`tests/v5/_domain_fixture.py` (household retail over a civilisation's supply chain, never counted by the gates).

## 1. What a domain is

A Tier 2 domain is not a task prompt over an unchanged environment. It is a family of worlds for one Part 9 subject:

| part | field of `Domain` | rule |
|---|---|---|
| world recipe | `build(config, seed, overrides) -> recipe` | a civilisation recipe restricted to the subsystems, mechanism variants, couplings, topology and population the configuration needs (`base.civ_recipe`, `mode="C"`, construction `civ_restriction`, the default), or - where the spec allows a standalone semantic world - any recipe in the same form (`mode="B"`, construction `standalone`). Both are execution-mode B worlds of Part 2.1 |
| configurations | `configs: tuple[DomainConfig]` | >= 3 `train`, >= 1 `eval_compositional`, >= 1 `eval_structural`; each with an independent characterisation (`doc`) and a declared `structure` (documentation only: the framework MEASURES structure on the built world) |
| seats | `seats: tuple[Seat]` | the participants a learner (or the reference) occupies; ordinary persons with an endowment created at genesis by module `seat`; several seats make a multi-learner world |
| objective | `goal: GoalFamily` | the independent goal generator (`crucible.goals`): parameters drawn per world, participant and difficulty from the generator's own stream; the outcome a `goals.metrics` predicate; only the text is public (section 5) |
| evaluator | `evaluate(world, principal, objective)` (optional with a `goal`) | evaluator side only; a function of recorded world state (same verdict on a world rebuilt from the event log); its verdict must need the autonomous participants (section 5) |
| hidden reference | `reference(principal, public_objective) -> policy` | a scripted, tool-only policy run in its own process; it receives the objective's TEXT, observations and its own tool results, never the World; labelled scripted; never a tool, never named in anything a participant receives |
| null baseline | `null_baseline` (default `base.Idle`) | must fail on every configuration |
| degenerate probes | `degenerate: tuple[Probe]` | >= 1 strategy that games the objective (self-dealing, empty listings, spam, ...); each must fail; the generic `RandomTools` probe runs too |
| mechanics | `mechanics: tuple[Mechanic]` | one probe per required mechanic of the Part 9 section (exact names: `base.PART9[part]["mechanics"]`): its own event types and/or an evidence function returning event sequence numbers (section 6) |
| autonomous participants | `actors: tuple[Actor]` | the autonomous participant kinds and the recorded DECISIONS they make (section 6) |
| horizon | `horizon_days`, `slowest_effect_days`, `dt` | the evaluation window must exceed the slowest delayed effect of the world (audit I_domains S.3: fleet wear collapses near day 150) |

Everything defaults to `observation_only` (Clarification 2, D-19): the learner gets reward 0.0, observations without
evaluator data, and the evaluator's verdict reaches only `env.verify()`.

## 2. The minimal domain file

`crucible/domains/<name>.py` (cheap to import: definitions only - `crucible.load_all_modules()` imports every file
of the package so that domain kernel modules exist in a fresh process):

```python
from . import base
from .base import Actor, Domain, DomainConfig, GoalFamily, Mechanic, Probe, ScriptedReference, Seat, act_

def build(config, seed, overrides):
    return base.civ_recipe(seed, profile=PROFILE_OF[config.name], subsystems=["scientific_research"],
                           modules=["my_module"], population_modules=("pop", "popweb"),
                           module_config={"my_module": {...}})

def _draw(rng, difficulty, world):              # rng: the goal generator's own stream; world: a BuiltWorld
    return {"min_replications": int(2 + difficulty + rng.integers(0, 2)), "window_days": 60.0}

GOAL = GoalFamily(
    "replicated_discovery", draw=_draw,
    text=lambda p: f"Publish a law that at least {p['min_replications']} unrelated labs replicate within ...",
    success=lambda p: {"metric": "independent_replications", "args": {"principal": "$self"}, "op": ">=",
                       "value": p["min_replications"]},
    progress=lambda p: {"metric": "independent_replications", "args": {"principal": "$self"},
                        "target": p["min_replications"]})

def count_replications(world, since):           # evidence: the SEQUENCE NUMBERS of the events, never a number
    return base.events_where(world, since, ("sci.replication",), lambda e: e.payload.get("lab") != e.actor)

class ScienceReference(ScriptedReference):      # tool-only: plan(obs) -> [act_(tool, tag, **args)], on_result(...)
    def plan(self, obs): ...                    # self.objective == {"text": ...} - nothing else
    def on_result(self, tag, result, action): ...

DOMAIN = base.register(Domain(
    name="science", title="Scientific research", spec_part="9.5", required_mechanics=(),   # () = the Part 9 list
    configs=(DomainConfig("lab_grid", "train", {...}, "..."), ...), build=build, goal=GOAL,
    reference=ScienceReference,
    mechanics=(Mechanic("Experimental apparatus", ("sci.run", "sci.reading")),
               Mechanic("Independent replication", (), count=count_replications), ...),
    seats=(Seat("researcher", 2_000_000),),
    actors=(Actor("rival labs", ("sci.paper",)), Actor("households", ("cols.batch:goods.session",))),
    degenerate=(Probe("publish_constant", "publish a constant model", PublishConstant),),
    horizon_days=60, slowest_effect_days=10, sources=("science.py",)))
```

A metric the predicate names must be registered in `crucible.goals.metrics` (`@metric("name")`; evaluator side, a
function of recorded state). `register` validates the definition (refusing it with the list of problems);
`base.check(DOMAIN)` builds and runs it and returns every Gate N / SP-9.0 / SP-9.16 measurement. Run `check` on your
domain before you push, and read `base.verdicts(report)`: every `False` is a failing criterion.

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
fixture adds `market`. `check` runs the reference and the null baseline on EVERY configuration and counts failed
processes and invariant violations in every run; a domain is not "implemented" (N.1) while any occur, and an
evaluation configuration that fails at runtime does not count (N.7).

The framework adds the seats (module `seat`) and the params `domain`, `domain_config`, `domain_split` and
`family_split`, and orders the modules by dependencies (`base.ordered_spec`). `base.spec(domain, config, seed,
overrides)` is the WorldSpec of one procedural instance; `overrides` keys `params`, `module_config` and
`drop_modules` are applied after the build, every other key goes to your `build`.

`tier3_spec(domain, config, seed, profile)` merges a configuration into a civilisation (the civilisation's recipe
plus your modules it lacks and the seats). If your modules need different configuration inside a civilisation,
give `Domain.tier3(config, merged_recipe) -> recipe`. SP-9.0 requires that the composed world builds, runs six hours
with no failed process and no invariant error, the seat exists and your evaluator runs on it.

## 4. Structure is measured, not declared (N.2, N.6, N.7, SP-9.0; D-49)

`base.fingerprint(world, domain, activity)` is the discrete structure of a built world:

* `modules`: the ENGAGED modules (`base.engaged_modules`): a module counts when one of its appliers recorded an
  event, a table it declared holds rows or a column table it declared has rows - at genesis, plus (in `check`) the
  modules the configuration's null run engaged. A module that only adds tools nobody uses, empty tables or a
  process that records nothing (calendar, support, studio in the verifier's probe) is not structure.
* participant types (principal kind with person/org kind), the civilisation structure
  (`crucible.civ.integration.fingerprint`: couplings, supply-chain and freight graph shapes, regimes, institutions);
* your `Domain.structure(world) -> dict`, which receives a `base.BuiltWorld`: the built state only - `state`
  (read-only tables: `get`, `scan`, `find`, `tables[...]`, arrays), `column(table, name)`, `modules` (engaged module
  NAMES; `"pop" in world.modules` works, `world.modules["x"]` raises `DeclarationAccess`), `now`, `seq`. There is no
  spec, no parameters, no module object or configuration: those are declarations. A hook that reaches for one (or
  whose source reads `.spec`, `.config`, `.cfg`, `.params`, uses `getattr` with a computed name, `gc`, ...) is an
  error of the configuration (`structure_report["errors"]["structure_hook"]`). If a structural choice only lives in
  your module's configuration, make the module write it into state at genesis (a row of the thing it configures)
  and read that row - better, read the things it builds (graphs, inventories, the rules in force).

Every value is reduced by `base.discrete`: a magnitude becomes presence/absence (`"zero"`/`"nonzero"`), a numeric
sequence its set of degree classes, a graph its scale-invariant `graph_shape` (degree classes, cycles, components).
So 4 vs 6 regions on a ring is the same structure; a ring vs a line is not.

* N.2: the number of training configurations with distinct structure digests that are stable across seeds (a
  configuration whose structure changes with the seed is not a configuration).
* N.7: on (path, value) atoms of the fingerprints, an `eval_structural` configuration has at least one atom no
  training configuration has; an `eval_compositional` configuration has no new atom and a combination equal to no
  training configuration. Either counts only when the configuration runs clean, its reference succeeds there and its
  null baseline fails there (`check()["per_config"][c]["feasible"]`).
* SP-9.0 `configurations_differing_only_in_objective_text`: pairs of configurations with equal structure digests
  (plus unstable ones), plus pairs of configurations of DIFFERENT domains whose framework-measured (core) structure
  is equal - the same world under another task prompt.
* N.6: distinct genesis CONTENT (`base.instance_digest`: tables and arrays, without the kernel's hash chain,
  sequence numbers, clock and counters) over seeds 0, 1, 2 of every configuration. A world whose build ignores the
  seed has one instance. `structure_report["varying_tables"]` lists what varies.

## 5. Objectives, the evaluator, the reference, the null baseline and the probes (N.4, N.5, N.n1, SP-9.16)

**Objective.** `Domain.make_objective(world, principal, difficulty)` = `base.generate_objective`: for a
`GoalFamily` the parameters come from `draw(rng, difficulty, BuiltWorld)` with `rng = base.goal_rng(...)` (keyed by
the world's seed, the domain, the participant, the level, the family - never one of the world's streams), the
success predicate is checked by `goals.metrics.check_predicate`, and the objective is
`{"text", "generator": "crucible.goals", "goal_family", "params", "success", "progress", "window_days",
"difficulty"}`. The learner's observation carries only `text`. `goal_report` (in `check`) requires a `GoalFamily`
and distinct objective texts across `Domain.difficulties` (`verdicts()["objective_generated"]`, part of N.1). A
domain with only a free `objective` callable is reported `generator: "domain"` and is not implemented.

**Evaluator.** `Domain.judge(world, principal, objective)` = your `evaluate` if given, else `base.goal_verdict` (the
predicate with `$self`, `$org` - each organisation the participant owns - and `$orgs` bound, `goals.contracts`;
score = progress metric / target). `evaluator_report` (N.4) requires: the verdict on a world replayed from the event
log equals the live one; evaluator data hidden from participants; no participant tool named or described like the
evaluator (`base.evaluator_like_tools`: score, grade, verdict, reward, objective, progress, success, "your current
score", ...) and none whose handler calls your `evaluate`; and the verdict NEEDS the autonomous participants: the
reference is run again with the scheduled processes that made the actors' and the population's decisions in the
null run frozen (`base.freeze_processes`), and must then fail. A verdict on a participant-controlled fact (the seat
founded an organisation, published something, paid itself) survives the freeze and is not independent.

**Reference.** A policy is any object with `act(obs, tools) -> action | None` and optionally `observe(reward,
result, next_obs, done)` and `burst` (actions per decision point) - the learner protocol of `crucible.experience`.
`base.run(domain, config, seed, policy=...)` builds the world through `crucible.experience.make(2,
"<domain>:<config>", seed)` (the learner's own path, observation_only), drives the first seat with the reference
(`"reference"`), the null baseline (`"null"`), a probe (its name) or `"random_tools"`, other seats idle, for
`horizon_days`, and returns the evaluator's verdict from `env.verify()`. The reference ALWAYS runs in its own
process (`base.IsolatedPolicy`: JSON observations and results over a pipe; the child holds no World, environment
or event log, so garbage-collector walks or string-built attribute names find nothing). Its factory must therefore
be importable by module path (a module-level class or function of an importable module; not `__main__`, not a
closure). It is called with `(principal, {"text": objective_text})`: exactly what the learner sees.

`ScriptedReference` is the base for references: `plan(obs)` returns actions (`act_(tool, tag, **args)`), results
come back to `on_result(tag, result, action)` (None when refused; the error is in `self.failures`). Reading is done
by calling read-only tools like any learner; numbers the reference needs from the objective are in its text. Rules,
checked by `check`:

* tool-only: `base.reads_world` scans the policy classes' source AND every module-level helper function or class of
  the same module they refer to (transitively), by AST, for world internals (`.world`, `.state`, `.spec`,
  `.modules`, `.tables`, `system_call`, `rng_for`, `REGISTRY`, `__dict__`, frames, `World`), introspection (`gc`,
  `inspect`, `sys`, `importlib`, `globals()`, `vars()`, `eval`, `getattr` with a computed name) and imports of the
  kernel, the framework, `crucible.experience` or `crucible.civ`. Name your own fields differently (`self.stage`,
  not `self.state`). `reference_tool_only` also requires that every reference run was isolated.
* feasible: the reference succeeds on a majority (and at least one) of `reference_seeds` x `reference_configs`
  (default: seeds 0 and 1 of the first training configuration), and on every evaluation configuration you want
  counted (seed 0).
* not exposed (N.n1): `check` scans EVERYTHING the seat receives - every observation field (text, events,
  artifacts, entities, analytics, numbers, last_result, notes, any field a module adds) and every tool result, in
  every run - for the reference's canary identifiers (class, factory and module names), and the null runs'
  observations and results, the results of every argument-free read-only tool (`World.query`), the tool list and
  the objectives for the reference's own ordered tool path (>= 3 of its first distinct tools, in order). A text that
  mentions at least half of the seat's tools is a catalogue (help listing) and is scanned for canaries only. No tool
  may be named after the reference; participant sandbox code cannot import `crucible`.
* not guaranteed (SP-9.16): the null baseline fails on every configuration; every declared degenerate probe and
  `RandomTools` fail - and both count only for a feasible domain (a failure where nothing can succeed shows nothing).

Typical degenerate probes: paying yourself through a related organisation (`goals.metrics.independent_revenue`
ignores related payers), listing without stock, publishing without data, deploying a no-op controller, firing all
staff (when the horizon is shorter than the consequences, the evaluator is wrong: fix the horizon),
keyword-stuffed content.

## 6. Mechanics and autonomous participants (N.3, SP-9.0, SP-9.n)

`Mechanic(name, events, doc, count=None)`:

* `events`: patterns counted in the event log after genesis: an exact type (`make.trade`), a prefix (`sci.*`), a
  column-batch kind (`cols.batch:goods.session`, `cols.batch:goods.*`, the population's decisions), optionally with
  a payload filter (`bank.txn{kind=subscription}`, `pkg.executed{kind=ci}`). Kernel bookkeeping (`cmd`, `proc`,
  `sched.*`, `ctr`, `principal.*`, ...) NEVER counts. A bare pattern of a substrate-generic type (`bank.txn`,
  `bank.account`, `id.*`, `artifact.stored`, `space.recorded`, `cols.grow`, `cols.batch:pop.*`,
  `cols.batch:attention.*`, or a prefix of those namespaces) counts nothing: those happen in every world - filter
  them (`validate` refuses a mechanic evidenced only by such patterns).
* `count(world, since_seq)`: an EVIDENCE FUNCTION returning the sequence numbers of recorded events after
  `since_seq` that exercise the mechanic - use `base.events_where(world, since, types, pred)`; for a mechanic
  measured on state rows, return the seqs of the events whose appliers wrote them. The framework checks each seq
  (recorded in the run, not bookkeeping). A function that returns a NUMBER is a self-report: it counts nothing and
  is listed as `unverified`; seqs outside the run are `rejected`.
* Two mechanics whose evidence is exactly the same set of events are one mechanism: the later one counts nothing
  (`indistinct`; `validate` also refuses identical pattern sets).

A mechanic is exercised when the union over the reference runs counts more than zero. Choose events that only happen
when the mechanic really operates (a purchase, a replication report, a failed machine), not set-up events.

`Actor(name, events, doc, count=None)` is counted in the null-baseline runs (the seats do nothing). An event counts
only when it is a DECISION of a participant: its `Event.actor` is a principal of the world other than the seats and
the organisations they act for, or it is a population column batch carrying decision provenance (D-15). Events the
system records (`sys` actor: bookkeeping, deliveries, word-of-mouth propagation, ...) never count: if your
participants decide inside a scheduled process, emit their events as them (`ctx` acting for the participant's
principal) or record the decision as a population batch with provenance. SP-9.0 also requires decisions of the one
population of an activity other than the population's own bookkeeping (`pop.*`, `attention.*`). A domain without
population activity therefore fails SP-9.0: register a population activity
(`crucible.population.register_activity`) or compose one (`popweb`, `civgoods`, `work`, `pop_social`, ...).

`open_ended` (SP-9.0) is measured on the seat's tool list: >= 10 tools over >= 4 MASTER_SPEC 11.2 categories, an
authoring category (files, content, code, deployments) and a free-form content tool. Civilisation recipes give this
by default (code, artifacts, bank, identity, ...).

## 7. Difficulty without dice (SP-9.16)

Difficulty must come from complexity, scarcity and uncertainty. `base.dice_scan` reads the domain's source files
(`build`/`goal`/`objective`/`evaluate` files plus `Domain.sources`) AND the source files of every kernel module of its
built worlds (`base.world_sources`: where a civilisation-backed domain's mechanisms live), and counts random draws
that decide a failure or success: a draw compared with a threshold (`rng.random() < p`, `rng.uniform() > q`,
`rng.integers(0, 10) < 3`), a binomial draw, a weighted choice between True/False or outcome words, whose
expression, assigned name, enclosing function or branch speaks of failure or success (or whose branch raises,
fails, rejects, marks something down or returns `ok: False`). An empirically grounded hazard (a calibrated failure
rate) is allowed only with a pragma `# fidelity: <id>` on the draw's line or the line above, naming a record of
docs/v5/FIDELITY_REGISTER.md. Known unregistered draws in civilisation modules (economy/industry.py machine
breakdown, services/cloud.py provider reliability) count for every domain whose worlds contain them until their
owners register and mark them.

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
    assert m["unverified_evidence"] == "none" and m["indistinct_mechanics"] == "none"
    assert m["observation_only_default"] and m["reachable_through_experience"]
```

`base.section_metrics(domain, configs=None, seeds=None, days=None) -> (metrics, runs)` returns the five shared
metrics plus `events_<mechanic>` evidence counts, `unverified_evidence` and `indistinct_mechanics`, measured on the
domain's (isolated) reference runs (cached in-process, shared with `check`).

For every mechanic, go further than counting where the spec says so: an intervention that removes the mechanic must
change the outcome it is responsible for (e.g. remove replication labs: no replication reports and the published
claims are no longer checked; remove noise: repeated measurements are identical). `base.run(...,
overrides=...)` builds such an arm, `base.run(..., freeze=[process names])` switches scheduled processes off. Put
those controls in the same test (D-26: a measurement that survives replacing your implementation with raising stubs
is rejected).

## 9. Gate tests owned by the framework

`tests/v5/test_gate_N.py` (N.1-N.7, N.n1), `tests/v5/test_gate_SP_9_0.py`, `tests/v5/test_gate_SP_9_16.py` iterate
over `base.part9_domains()` with `base.gate_reports()` (one `check` per domain, cached in-process) and
`base.gate_metrics()`: a Part 9 section counts when one of its domains satisfies the criterion; a section without a
domain counts as failing, so these tests FAIL (never skip) while fewer than fifteen sections are covered. Each test
also runs positive controls on the fixture (7 of its 21 horizon days): planted defects must be detected - a
numbers-only configuration, a hook reading the spec, inert modules, a world without procedural content, generic /
bookkeeping / constant mechanic evidence, actors on bookkeeping and system events, a self-report evaluator, a
participant tool named like the evaluator, a gc-peeking reference (in-process vs isolated), a path planted in an
observation field, a broken evaluation configuration, an infeasible domain, a second domain over the same world, the
six common dice forms and the scan of the world's kernel module files.

`verdicts(report)["implemented"]` (N.1) requires: definition valid, every configuration builds on three seeds, runs
clean on every configuration (no run error, failed process or invariant violation), >= 3 distinct stable training
structures, >= 3 distinct generated instances per configuration, a clean and feasible compositional and structural
evaluation configuration, every mechanic exercised by verified evidence, an independent evaluator (replay-equal,
hidden, no evaluator-like tool, needs the autonomous participants; reference succeeds and null fails), a feasible
isolated tool-only reference, no exposure, observation_only by default with no nonzero reward in any run, and
objectives from the goal generator.

## 10. Reaching a domain

* `crucible.experience.make(2, "<domain>:<config>", seed)` (also `"tier2:<domain>/<config>"`,
  `"d5_<domain>_<config>"`, a bare `"<domain>"` for its first training configuration): learners occupy every seat by
  default (`agents=n`: the first n seats); each seat's objective is generated for it (`Domain.make_objective`);
  `verify()` carries the verdict (`Domain.judge`).
* execution mode B (`crucible.domains.modes`, SP-2.1): every configuration of every registered non-fixture domain,
  of both constructions (`base.summary()[name]["construction"]`).
* the research distribution registry (`crucible.research.v5.distribution`): one entry per configuration,
  partitioned by split.
* the red-team probes (`crucible.redteam.v5.civ.tier2_probes`, workstream O): one offline and one
  evaluator-isolation probe per evaluation configuration. They still call the earlier contract
  `Domain.reference(world, principal, objective)` and `pol(world, principal)`; the current contract is
  `base.run(domain, config, seed, policy="reference")` (isolated) or `Domain.reference_policy(principal,
  objective)` for inspection (integration note for O in docs/v5/workstreams/I-fw.json).

## 11. Kernel modules in this package

`seats.py` (`seat`), and the domain kernel modules left by earlier waves: `science.py`/`lawlang.py` (`sci`),
`infra_world.py` (`fleet`), `firms.py`/`_kit.py` (`firms`), `learning.py`/`concepts.py`/`common.py` (`learn`).
Their open defects (DEF-I-2, DEF-I-4, DEF-I-5, DEF-I-6) belong to the engineers of the domains that use them.
`_kit.ToolPolicy` reads through `World.query` and is kept for `firms`; new references use `ScriptedReference`.

## 12. Migrating from `domains-framework/2` (the wave-3b repair pass)

An independent verifier showed that several measurements of `/2` could be satisfied without the property they
claim. What changed, and what a domain written against `/2` must do:

| `/2` | `/3` | migrate |
|---|---|---|
| `Mechanic.count`/`Actor.count` returned an int | returns event sequence numbers; an int is `unverified` and counts 0 | `return base.events_where(world, since, ("x.y",), pred)`; for state rows, the seqs of the events that wrote them |
| any event type counted (`bank.txn`, `sched.*`, `artifact.stored`) | bookkeeping never; generic types only with a payload filter; identical evidence counted once | name your mechanism's own event types or filter (`bank.txn{kind=...}`) |
| actors counted any matching event | only decisions of non-seat principals or population batches with provenance | emit participants' decisions as their principals |
| `structure(world)` got the World | gets a `BuiltWorld` (built state only) | read tables/columns, not `world.modules[x].config` / `world.spec` |
| any declared module was structure | only engaged modules | nothing (inert modules no longer count) |
| instances = distinct `World.digest()` | distinct genesis content (`instance_digest`) | make the build use the seed (procedural content) |
| reference got the full objective dict, ran in-process | gets `{"text": ...}`, runs in its own process | parse numbers from your text; keep the factory importable (module level) |
| `objective` free callable | `goal: GoalFamily` (crucible.goals) required for N.1 | wrap the objective: `draw` (targets from `rng`), `text`, `success` predicate over a registered metric (`@goals.metrics.metric`) |
| evaluator checked by replay only | also: no evaluator-like tool, verdict needs the autonomous participants (freeze ablation) | make the verdict an outcome produced by independent participants |
| `check` ran the first training configuration only | reference and null on every configuration; eval configurations must run clean and be feasible | make the reference work on the evaluation configurations |
| exposure scan: objective, notes, tool docs | every observation field, result, read-only query | do not echo solution paths in world text |
| dice scan: domain files, `random() < p` forms | + world module files, thresholded integer/uniform draws, binomial, weighted True/False choices, `ok: False` / status-down branches | register hazards in FIDELITY_REGISTER and mark them `# fidelity: <id>` |
| SP-9.16 null/probe counts ignored feasibility | count only for a feasible domain | - |
