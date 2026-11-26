# Audit G_innovation: open-ended creation, emerging needs, software ecosystem (Gates J, L; SP-6.3 to SP-6.5)

Auditor: independent audit, worktree `feat/crucible-5-complete` @ `a881d18`. No existing files were changed.
Probe scripts and logs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/G_innovation/`.

Scope: MASTER_SPEC Parts 6.3, 6.4 and 6.5, Gates J and L, and Part 10.3 (multi-order chains, which Gate L
operationalises). The binding inputs are Clarification 1 (independence from CAM) and the two new amendments:
observation-only by default, and "do not trust previous completion claims".

Code covered:
- `src/crucible/innovation/*`: caps, needs, pkg, adopt, formats, worlds, reference, scenarios, measure.
- `src/crucible/usefulness/*`.
- The services this area uses: `services/capabilities.py` (4.0 caps), `services/cloud.py`, `services/code.py`,
  `services/issues.py` and `sandbox/*`.
- The 4.0 invention machinery it should connect to: `actors/innovators.py`, `evaluation/open_endedness.py` and
  `evaluation/causal.py`.

## 1. Verdict in one paragraph

Workstream G built a working, carefully written package, and every claimed gate number reproduces. But it is a
*standalone lean-world demonstration*, not a civilisation subsystem:

- **Nothing integrates it.** No module outside `crucible/innovation` uses `caps.declare/try/adopt_service`, `needs`,
  `pkg` or `adopt`, except the red-team world list.
- **Each Gate L chain has its own world.** The chains come from four hand-built scenario worlds. Each has its own
  extractor, hard-coded to host names, and the creators are scripted.
- **Demand is closed.** Participants can hold only the predefined problem formats, which have oracle verifiers, so a
  genuinely novel category cannot be adopted.
- **The software ecosystem's evidence can be forged.** Advisories, adoption-by-execution and fork lineage can all be
  faked by any principal.
- **The compute meter is not compute.** Under the "deterministic" cost model that every Gate J/L scenario uses,
  compute is a count of Python lines.
- **Evaluator-side verdicts leak into observations.** The independence verdict and derived needs both reach in-world
  actors. This conflicts with the observation-only amendment.

No requirement in this workstream can be marked done from the existing evidence.

## 2. What I ran (all from `crucible/`, `PYTHONPATH=src`, `/venv/main/bin/python`)

| # | command | result |
|---|---|---|
| T1 | `slots --cpu 1 -- pytest -q tests/test_v5_innovation_caps.py tests/test_v5_innovation_needs.py tests/test_v5_innovation_pkg.py` | 21 passed, 24.8 s |
| T2 | `slots --cpu 1 -- pytest -q --junitxml=... tests/v5/test_gate_J.py tests/v5/test_gate_L.py tests/v5/test_gate_SP_6_3.py tests/v5/test_gate_SP_6_4.py tests/v5/test_gate_SP_6_5.py` (TMPDIR = scratchpad) | 17 passed, 96 s. Unofficial evidence reproduces the component doc's numbers exactly (table in section 4). |
| T3 | `slots --cpu 1 -- pytest -q tests/test_sandbox.py tests/test_npc_innovation.py` | 9 passed, 80.7 s |
| T4 | `slots --cpu 1 -- pytest -q tests/test_usefulness.py` | 52 passed, 16.4 s |
| P1 | `probe_pkg_integrity.py` (software scenario, seed 3) | advisory, adoption and lineage forgeries all succeed; invariants still clean (section 5.1) |
| P2 | `probe_seed_robustness.py 1,3,4,8` (gate-test chain extractors and L.5 predicates, unchanged) | 14 of 16 chain/L.5 checks hold on the other seeds; batch@8 and social@3 fail (section 5.5) |
| P3 | `probe_registry_openness.py` | novel need is a KeyError; novel service gets 0 trials; needs oracle visible to operator and participants; cost model 3,466x (section 5.2) |
| P4 | `probe_cost_model.py` | 149 ms of real builtin work is metered as 0.043 ms; 1.7 ms of Python loop as 4.04 ms |
| P5 | `probe_needs_generality.py` | data edges for skill, transport and coordination yield no needs; a 5,000,000-cent success bonus labelled `endowment` is invisible to L.n1 |
| P6 | `probe_use_integration.py` | adopters *can* hold a 4.0 generated category (`gen.allocate.*`) when module `use` is installed; no scenario or gate does this |
| P7 | `probe_independence_leak.py` | `caps.tried` and `caps.adopted` payloads seen by participants and providers carry `independent` (the evaluator's `is_related` verdict) |
| P8 | `sandbox.runner.capabilities()` + pool config | Landlock ABI 4 (net), seccomp and `nobody`; `allow_weak=False`; `count_lines=True` |

## 3. Requirements and the spec text they must establish

The 17 workstream-G manifest entries are J.1 to J.7, J.n1, L.1 to L.5, L.n1, SP-6.3, SP-6.4 and SP-6.5. All are
NOT_STARTED.

Every one of them has `implementation_files: []`. `acceptance/v5/checker.py:95-97` reports "no implementation files
listed" for that, so none can reach VERIFIED until the manifest names files. No official evidence exists under
`docs/v5/evidence/gate_{J,L,SP}`.

## 4. Reproduced gate numbers (T2), and what they actually show

| req | metric | value | what it really measures |
|---|---|---|---|
| J.1 | agent_created_executable_products | 4 | 1 route.shortest service and 3 packages. All are built by scripted `ServiceCreator` and `software_history`, which count as "agents" because `admin.join kind=agent` (`measure.py:24-36`). No learner or LM agent ever used the API. |
| J.2 | published_products | 4 | Real public `caps.declared` / `pkg.released` events. |
| J.3 | independent_discoverers | 24 | Any principal whose `caps.search` returned the declaration, or who learned the host (`measure.py:89-105`). In P3 a product nobody wanted also had 24 "discoverers". |
| J.4 | independent_users | 24 | Real executions (trials, use, payments). This is sound. |
| J.5 | measured_behavior_adoption_pvalue | 1.55e-12 (24/24 vs 1/24) | Two forks, tier-1 vs tier-0 code. Adopters score outputs with the format's *hidden ground truth* (`adopt.py:143,186`) against a fixed threshold, so the result is guaranteed by construction. |
| J.6 | independent_transactions | 1,092 (24 payers) | Real ledger `sale` transfers from participants' genesis endowments. This is sound. |
| J.7 | resource_consumption_delta | +57,694 CU | The product's own metered footprint against no product. The meter counts Python lines (section 5.2). |
| J.n1 | product_categories | 2 | `api_service` and `software_package`: artifact kinds hard-coded in `measure.py:64,75`. The package category has 1 scripted user and 0 transactions. |
| L.1 | capability_effects_measured | 5 | Differences between fork arms in G's own scenarios. |
| L.2 | emerged_needs | 8 | Labels produced by G's own rule engine (supply, alternative, provide). |
| L.3 | needs_addressed_by_new_services | 2 | One by a scripted entrepreneur (`on_request`), one by a scripted fixed release. |
| L.4 | independent_chains / min stages | 4 / 4 | Four hand-built worlds with per-scenario extractors (`tests/v5/test_gate_L.py:47-110`). |
| L.5 | counterfactual_removals_confirming | 3/3 | Hand-chosen predicates per chain, single seeds. |
| L.n1 | special_case_bonuses | 0 | The measurement has a hole (section 5.6). |
| SP-6.3 | novel_services_discovered_tested_adopted | 1 | "Novel" means created after genesis *for a predefined format*. |
| SP-6.4 | reusable_need_relations / special-case triggers | 6 / 0 | Static literal check plus rename invariance. Generality is not tested (P5). |
| SP-6.5 | software_features / adoptions_by_execution | 15 / 2 | Each feature happens once in a scripted history. Adoption can be forged (P1). |

## 5. Findings with evidence

### 5.1 Software ecosystem evidence can be forged (P1, `probe_pkg_integrity.log`)

**Advisory forgery.**
- `pkg.py:771-772` sets `verified = res["failed"] > 0`: any failing test in the reporter's repository "verifies" a
  security advisory.
- An unrelated principal `zed` committed `tests/test_x.py: assert False`, which never imports the package, and filed
  a *critical* advisory against `vecmath@1.0.0`. Result: `verified: true`, a public issue, and a needs-graph
  `alternative` need for the dependent `p:ana`.
- After the maintainer released 1.0.1, the advisory added 1.0.1 to `affected`. `fixed_in` stays null forever.
  `pkg.info` keeps showing `open_advisories: 1`.
- The repo's own unit test `tests/test_v5_innovation_pkg.py:108-124` verifies an advisory whose "defect" is
  `norm(['x']) is None`.

So any agent can permanently brand any competitor's package insecure. A learner here would learn griefing, not
security research.

**Adoption-by-execution forgery A.**
- The tracer's counter `_C` is a module global in `_pkgtrace` (`pkg.py:44-66`). It runs in the same interpreter as
  the dependent's code.
- `yan/forge` sets `sys.modules['_pkgtrace']._C['vecmath'] = 1000` without calling vecmath. `pkg.exec` then reports
  `calls {vecmath: 1000}` and records an *independent* adoption (`pkg_adoptions`).

**Adoption forgery B.** `xia/shadow` commits its own `deps/vecmath/fake.py` and calls it. This also records an
independent adoption (`calls: 2`), because the tracer attributes calls by the path substring `/deps/<name>/`.

**Lineage forgery.**
- Fork lineage is parsed from the user-settable repository description (`pkg.py:306-309`), because
  `code.create_repo` takes any description.
- A non-fork registered with the description "fork of eve/vecmath" got `forked_from: vecmath`.
- If its API is a superset, a `substitutes` edge enters the needs graph (`pkg.py:805-809`).

None of these trip `check_invariants()` (result `[]`), and `software_features` stays 15.

**Related defect.** `Packages.remedies` (`pkg.py:828-840`) claims in its docstring to return "substitute packages",
but returns only releases of the same package. A package `alternative` need can therefore never be addressed by
switching to another package.

### 5.2 "Actual world resource consumption" is a Python line count (P3 R5, P4)

`innovation_spec` sets `cost_model: deterministic` (`innovation/worlds.py:89`). Every Gate J/L scenario uses it.

Under it, `sandbox/bridge.py:20,101-102` meters `cpu_ms = executed Python lines x 0.001`:
- A service doing `sorted(range(3e6))` plus `bytes(24 MB)` took 149.4 ms real and was **metered 0.043 ms**.
- A 2,000-iteration Python loop took 1.7 ms real and was **metered 4.04 ms**.
- The same 300-item sort cost 135.2 ms metered as a bubble sort and 0.039 ms with `sorted()`.

Capacity bottlenecks, which drive the supply needs of L.2/L.3 and the J.7 delta, are measured in this unit. The
capacity and batch scenarios also multiply it by `cu_per_ms: 5000` (`scenarios.py:199,228`) to make constraints
bind. An agent can learn "push work into C builtins and compute is free", a shortcut that does not exist on the real
internet.

### 5.3 Demand is closed to predefined formats (P3 R1/R2, P6)

`adopt.genesis` calls `formats.lookup(need)`, which raises `KeyError` for anything outside `economy.needs.NEEDS`
plus the single extra `formats.FORMATS['route.batch']` (`adopt.py:78`, `formats.py:56-73`). So
`innovation_spec(needs={"palette.extract": 1})` fails.

In a world where an agent deployed and declared a novel `palette.extract` service next to `route.shortest`, and
also posted it on the forum:
- The novel service got **24 "discoverers", 0 trials, 0 adopters and 0 transactions**.
- The route service got 24/72/24/487.

Participants try only candidates whose declared input schema accepts a sample of *their own predefined format*
(`adopt.py:126-134`).

Consequences:
- 6.3's "novel product categories", "dynamically discovered uses" and "new workflows" have no demand path.
- 6.4's "genuinely novel combination ... whose adoption changes the future" works only because the developer added
  `route.batch` to `FORMATS` and configured participants to need it (`scenarios.py:227`). The novelty is
  designer-supplied.
- Usefulness is judged by the hidden-truth verifier inside each participant's decision (`adopt.py:143`), which is an
  oracle.
- 6.3's subjective usefulness from "diverse grounded participant responses" is absent from v5. The 4.0 `caps.review`
  exists but v5 declarations are not wired to it.

P6 shows a partial path: with module `use` installed, adopters accept 4.0 procedurally generated categories
(`gen.allocate.ee2534e9`). That widens demand to 10 families but is still closed, and no scenario uses it.

### 5.4 Observation-only amendment: evaluator verdicts and a needs oracle reach in-world actors (P3 R3/R4, P7)

**Needs oracle.**
- `needs.derived` is emitted with `vis=holders` (`needs.py:415-418`). `needs.mine` returns kind, target, rule, path
  and own evidence (`needs.py:461-475`).
- Operator `p:ana` received `needs.derived` with path `svc:anaroute consumes compute:inst_000001 / capacity
  constrains / produces cap:route.shortest`.
- Ordinary participants receive `needs.derived/changed/addressed/resolved`. The derived payload lists all other
  holders, which leaks who else uses the service.
- Participants read the module's internals directly (`adopt.py:211`, `nmod._held(...)`), not even through a tool.
- The scripted entrepreneur builds exactly when `needs.requests` lists `cap:<its format>` (`reference.py:99-109`).

This is an environment-computed opportunity signal that the real internet does not provide. The raw consequences
(`cloud.metrics`: overloads, errors) are already observable and should be the only in-world signal.

**Independence verdict leak.**
- `caps.search` results carry `independent_trials`, `independent_trial_ok_rate` and `independent_adopters`
  (`caps.py:153-165`).
- `caps.tried` and `caps.adopted` payloads carry `independent` (`caps.py:353,457`), which P7 confirmed in the
  participant's and the provider's event streams.
- `pkg.executed` and `pkg.review` do the same.

These are the evaluator's `goals.metrics.is_related` verdicts. An agent can probe them to learn which accounts the
evaluator counts as independent, which is a Sybil oracle. This touches the Gate K and W integrity rules as well.

### 5.5 Gate L chains are scenario-built, not civilisation-emergent (code, P2)

- **Hand-built worlds and extractors.** The four chains come from four separate lean worlds
  (`scenarios.capacity/batch/social/software`), each designed to produce one chain. The extractors in
  `tests/v5/test_gate_L.py:47-110` match literal hosts (`anaroute`, `batchroute`, `anasort`, `geokit@1.0.0`).
  Nothing mines chains generically from the event log.
- **Scripted stages.** The software chain is four of five scripted stages: release, execution by ben, advisory and
  fix are acts of `software_history`. Part 10.3 says scripted narrative events must not count. The capacity chain's
  last stage is a scripted `on_request` builder. In the social chain, forum-channel participants can discover *only*
  via posts that contain the literal format name (`adopt.py:107`), so deleting the posts removes their only channel
  by construction.
- **No integration.** `crucible.civ` and `crucible.multiagent` are stubs, no Tier 2 domain is registered, and no
  domain, population, social or civ module uses the innovation modules.
- **Seed robustness (P2), measured fairly.**
  - capacity: chain and L.5 held on seeds 1, 3, 4 and 8.
  - batch: held on seeds 1, 3 and 4. On seed 8 the chain was out of order and base CU was *lower* with the composite
    (3.54M vs 4.36M), so L.5 failed.
  - social: L.5 failed on seed 3 (forum adopters 3 of 9 after removal, above the 25% threshold).
  - market (J.3 to J.6): held on all 4 seeds (J.5 p between 6e-14 and 1.8e-10).

  The mechanisms are reasonably robust, but the gate uses one hand-picked seed per chain and no replication rate.

### 5.6 Measurement holes (P5, code)

- **L.n1.** `measure.special_case_bonuses` excludes every `endowment`-kind transfer from system accounts
  (`measure.py:312`). P5 paid a creator 5,000,000 cents from `acct_genesis` as "endowment" after 12 adoptions, and the
  bonus count stayed 0. Relabelled `grant`, it was caught (1). The static scan covers only files in
  `crucible/innovation`.
- **J.3.** A search hit counts as discovery (section 5.3: 24 discoverers, 0 trials).
- **J.7.** `> 0` is satisfied by any deployed service's own footprint.
- **J.n1.** The second category need not pass J.3 to J.7.

### 5.7 Needs graph generality (P5, code)

The rules are keyed by relation type, but the observables and dispatch are tied to node-name prefixes:
- Only `compute:` resources have an observable (`needs.py:263`).
- `requires` handles only `cap:` and `pkg:` targets (`needs.py:287,314`).
- `complements` handles only `cap:` nodes (`needs.py:323`).

Relations supplied as world data (`needs_cfg.edges`) for `skill:python`, `transport:route7` with a capacity
constraint, and `coordination:acme` produced **no needs**. Only the `cap:invoice.extract` edge fired.

None of the 6.4 examples is covered: workflow change, transport and market access, coordination, content to
communities, education to skills. Another workstream's domain cannot add a need-generating relation without
editing `needs.py`.

### 5.8 What is solid

- **Sandbox.** Landlock ABI 4 with network denial, seccomp, the unprivileged `nobody` user and `allow_weak=False` are
  all active here (P8). `tests/test_sandbox.py` passes. Agent-created programs really run isolated.
- **Real execution path.** `caps.try` and `caps.verify` execute the live service through `cloud.call_endpoint`, so the
  caller pays and capacity is enforced. Adoption requires the adopter's own successful execution, and invariants
  check it (`caps.py:564-584`).
- **Exact counterfactuals.** Forks from one snapshot (`scenarios.arms`) are exact and invariant-clean.
- **Opt-in surface.** The v5 surface is opt-in, so 4.0 and 3.0 worlds keep their catalogue (unit test passes).
- **PR rules.** PRs need an independent, diff-reading approval of the exact commit. API and semver comparison and the
  upgrade gate on the dependent's tests work as described.
- **Usefulness substrate.** The 4.0 substrate (10 procedural families with exact evaluators) is intact (52 tests).

### 5.9 Duplication with 4.0

The v5 `caps` subclasses the 4.0 registry but builds a second, parallel notion of each concept:

| concept | 4.0 | v5 |
|---|---|---|
| capability | installable capabilities | typed declarations |
| adoption | install kept after a switchback trial | `adopt_service` |
| tests | `caps.test` | `caps.verify` |
| problem signal | `ops.requests` | `needs.requests` |

The 4.0 `caps.categories` and `caps.graph` ignore v5 declarations (`grep cap_decls services/capabilities.py` finds 0
hits). The 4.0 autonomous innovators already do things the v5 work leaves to scripts: they identify problems from
public reports, generate programs from a grammar, run trials and hill-climb versions (`actors/innovators.py`).
`evaluation/open_endedness.py` (census and ablations) and `evaluation/causal.py` (fork-based installation effects)
are also unused by v5. Gate J/L creators remain scripted although autonomous NPC inventors exist.

### 5.10 Fidelity

None of the five v5 behaviour modules (caps, needs, pkg, adopt, infra) has a row in `FIDELITY_REGISTER.md`.
`calibration.fidelity.important_mechanisms()` lists all five, and `v5_rows()` returns none for them, so SP-5.4 fails
on this package.

The adopter parameters are uncalibrated assumptions: threshold 0.85 to 0.97, wtp 4 to 12 cents, 6 to 12 requests a
day. The GH Archive model `software_repo_growth@1` exists but "no simulator mechanism consumes it", including `pkg`.

## 6. Transfer check: what would a learning agent acquire here?

Transferable, if wired into the integrated world:
- Writing and deploying code that passes others' tests.
- Pricing against willingness to pay.
- Running CI and semver discipline.
- Reading diffs before review.

Simulator-specific shortcuts it would also learn:
1. Poll `needs.mine` / `needs.requests`: the world tells you which capability is short, with its exact name.
2. Build only for the handful of predefined formats. Novelty earns zero trials.
3. Win adoption purely on hidden-oracle quality and price thresholds. There is no trust, documentation, UX,
   onboarding or switching cost beyond a fixed window.
4. Push work into C builtins to make compute and capacity limits vanish.
5. Forge package adoption by writing the tracer counter, and knock out competitors with fake advisories.
6. Learn the evaluator's relatedness verdicts from `independent` fields.
7. For forum discovery, the post must contain the literal format name.

## 7. Subsystem classification (detail in `G_innovation.json`)

| subsystem | verdict | effort |
|---|---|---|
| Capability registry v5 (`innovation/caps.py` and 4.0 `services/capabilities.py`) | REPAIR | M |
| Needs graph (`innovation/needs.py`) | EXTEND | L |
| Software ecosystem (`innovation/pkg.py`) | REPAIR | M |
| Demand side: participants and problem formats (`innovation/adopt.py`, `formats.py`) | REDESIGN | L |
| Gate J/L measurement (`innovation/measure.py`) | REPAIR | S |
| Scenario worlds and Gate L chain evidence (`innovation/scenarios.py`, `tests/v5/test_gate_L.py`) | REDESIGN | L |
| Scripted reference agents (`innovation/reference.py`) | RETAIN | S |
| World construction (`innovation/worlds.py`) | EXTEND | M |
| 4.0 usefulness substrate (`usefulness/`) | RETAIN | S |
| Sandbox (`sandbox/runner.py`, `boot.py`) | RETAIN | S |
| Cloud hosting and metering (`services/cloud.py`, `sandbox/bridge.py` cost model) | REPAIR | M |
| Code hosting and issues (`services/code.py`, `services/issues.py`) | REPAIR | S |
| 4.0 autonomous invention, census and causal evaluation (`actors/innovators.py`, `evaluation/*`) | EXTEND | M |

## 8. Requirement classification

| id | verdict | weak criteria | work needed in the integrated system |
|---|---|---|---|
| J.1 | EXTEND | yes | Run in a Tier 2 software or technology-invention domain and a Tier 3 civ. Include at least one non-scripted creator (a 4.0 NPC innovator, a small learner or an LM agent) acting through the tools. List implementation files. |
| J.2 | RETAIN | no | Re-run in the integrated world. |
| J.3 | REPAIR | yes | Count discovery as reading the declaration or docs, or trying it; not a bare search hit. Use workstream-B population principals. |
| J.4 | EXTEND | no | Population-B users with their own tasks, not G adopters. |
| J.5 | EXTEND | no | Adopters that judge with their own imperfect tests, not hidden ground truth. Replicate across at least 5 seeds. |
| J.6 | RETAIN | yes | Require several independent payers net of refunds over a sustained period. |
| J.7 | REPAIR | yes | A cost model that meters real work, including builtins and C calls. Measure the effect on pre-existing resources, not the product's own footprint. |
| J.n1 | EXTEND | yes | At least 2 categories, each passing J.1 to J.7, including one category absent from the predefined formats. |
| L.1 | EXTEND | yes | Effects on pre-existing conditions in a civ world, with CIs across seeds. |
| L.2 | EXTEND | yes | Needs arising from mechanisms in other domains (skills, transport, coordination, content). The labeller stays evaluator-side. |
| L.3 | EXTEND | no | Addressed by autonomous entrepreneurs or learners reacting to observed consequences, not to an oracle request. |
| L.4 | REDESIGN | yes | Generic chain mining over the event log of an integrated civ. Enforce 10.3: an agent artifact, a social chain and a technical chain, with no scripted stages. |
| L.5 | EXTEND | yes | Apply the fork-removal instrument to mined chains, with an effect-size test replicated across seeds. |
| L.n1 | REPAIR | yes | Count all post-admission system-account credits, endowments included, and scan the static paths of every module. |
| SP-6.3 | EXTEND | yes | Open-ended demand: participant-written tests, diverse grounded subjective responses. Show a novel-category service being adopted. |
| SP-6.4 | EXTEND | yes | Generic resource and need observables that other modules register. Test generality with relations supplied as data. Cover the 6.4 examples. |
| SP-6.5 | REPAIR | yes | Fix advisory verification, the tracer's isolation, lineage and substitute remedies. Add an autonomous developer population and maintenance over time. |

## 9. Spec requirements with no manifest entry

- **6.3 (registry).** Subjective usefulness from diverse grounded participant responses (no single LLM judge).
  Adoption of a novel product category. Dynamically discovered uses of existing technologies. Workflows composed of
  independently created services as first-class registry objects.
- **6.4 (needs).** An agent-created *novel combination* whose adoption changes future state, without
  designer-supplied demand. Any of the seven example need mechanisms.
- **6.5 (software).** Resistance to adoption and advisory forgery. Repository stars not counted as adoption, as a
  criterion. Adoption by autonomous independent developers. Maintenance over a long horizon. Agent programs run in
  the sandbox, with no G entry.
- **10.3 / Gate L.** The chain-composition constraints (agent artifact, social, technical), the ban on scripted
  narrative stages, chains in the integrated Tier 3 civilisation, and replication across seeds.
- **Observation-only amendment.** No evaluator verdicts (`independent`) or derived-need oracle in tool responses or
  event metadata by default, with innovation worlds reachable through `experience.make` in `observation_only` mode.
  Note that `experience/registry.py:99` and `api/gym_env.py:71` still default to `legacy_shaped_reward`; that belongs
  to workstream L.
- **SP-5.4 coverage.** Fidelity rows for caps, needs, pkg, adopt and infra.

## Skeptic review

Reviewer: skeptic pass for G_innovation. The worktree head is now `dd745b9`, one commit after `a881d18`. That commit
adds Clarifications 2 and 3 and manifest entries OO.1 to OO.8 and RP.1 to RP.4. `git diff --stat a881d18 dd745b9`
over `src/crucible/{innovation,sandbox,services,usefulness}` is empty, so the code audited above is unchanged. No
existing file was modified except this appended section and the new `skeptic` key in `G_innovation.json`.

Probe scripts and logs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/G_innovation_skeptic/`.

Method:
- I tried to refute every RETAIN: Sandbox, Scripted reference agents, the 4.0 usefulness substrate, J.2 and J.6.
- I tried to refute the generous REPAIR/EXTEND verdicts.
- I checked whether the harsh verdicts hold: Demand REDESIGN, Scenario worlds REDESIGN and L.4 REDESIGN.
- Every overturn below rests on a probe I wrote and ran, not on a unit test or a doc claim.

### S.1 What I ran

| # | probe (from `crucible/`, `PYTHONPATH=src`, via `crucible.scale.slots --cpu 1`) | result |
|---|---|---|
| S1 | `probe_meter_tamper.py` (cost model `deterministic`, as in every Gate J/L world) | One hour, 1 core, 400 calls of the same Python loop. **Honest** service: 49 OK and 351 overloaded (`used_cu` 98,100 of 100,000). **Four tamper variants**, each 1 to 3 lines inside the service's own code, all got **400/400 OK**: reset `__main__._LINES[0]`, `sys.monitoring.set_events(4, 0)`, set `_LINES[0] = -10**9`, or do the work inside a re-import (`sys.modules.pop('heavy'); import heavy`). Metered cpu_ms per request: 0.029 / 0.011 / **-999,999.971** / 0.041 (honest: 40.04). The negative variant drove the instance's `used_cu` to **-2.0e10 CU** and `cu_est` to -5.0e7. `check_invariants() == []`. |
| S2 | `probe_meter_tamper_measured.py` (same, cost model `measured`) | Rebinding `__main__._cpu_ms = lambda: -1e9` gave cpu_ms **-1.0e9** per request and `used_cu` -5.0e10. Invariants `[]`. Both cost models are forgeable by the metered code. |
| S3 | `probe_j_metrics.py` (innovation world, seed 11, 10 participants, about 32 h) | **(A)** All 10 `caps.adopted` events the provider `p:ana` can read carry the note `mean score 1.0`, the adopter's hidden-truth verifier score. 10 public forum events carry it too. **(B)** An always-raising service at 4 cents: J.6 `independent_transactions` = **n 30, 120 cents, 10 payers**, although **all 30 sales were refunded**. J.4 counts 10 users. **(C)** The creator transferred 3,000 cents to a fresh, unrelated principal, which bought 500 calls of the creator's service. All 500 count as independent: anaroute shows n 631 and 11 payers, and 500 of those calls are circular. **(D)** A service undeployed right after declaring still has `published: True` and 10 "discoverers". Invariants `[]`. |
| S4 | `probe_oracle_dependence.py 2` (Gate L chain A, `scenarios.capacity` seed 2, counterfactual off) | **factual**: entrepreneur `bo` builds at h37 after 57 public need requests; 63 adoptions of boroute; L.3 = 1; 82 switches; 29 participants served at the end. **blind** (participants' `_respond_to_needs` is a no-op, so they no longer read `needs._held`): `bo` **never builds**, 0 requests, 0 boroute adoptions, **L.3 = 0**. **blind + bo builds unconditionally at day 2**: participants find and adopt boroute by their own failure-driven re-search (17 adoptions, L.3 = 1, 0 needs-triggered switches); 36 served at the end. |
| S5 | `probe_use_lab_leak.py` (module `use` with generated category `gen.allocate.ee2534e9`) | As the operator, `use.probe` returned the **hidden-evaluator mean score 0.2369** for the worst reference tier and **0.9993** for the best. `use.report` returns `verified_quality`. The world has no `reward_mode` parameter, and nothing in `usefulness/` or `innovation/` reads one. |
| S6 | `probe_planted_need.py` (market world, service created after the first needs update) | The **only** emerged need is `provide cap:route.render` (rule `complements`). It is held by all 12 participants, whose own task is `route.shortest`; nobody produces or calls `route.render`. With `needs_cfg={"interfaces": []}` the need disappears. In the gate's L.2 evidence, 2 of the 8 emerged needs are this need. |
| S7 | `probe_reference_labels.py` | `ServiceCreator.kind == "scripted"`, but world `persons.kind == "agent"`, `measure.is_agent` is True and J.1 counts its product. `reference.py` reads `world.state` / `world.modules` directly at lines 55, 120, 139, 154, 220, 261 and 289. Line 154 (instance cores) is a decision input. It uses the `needs.requests` and `needs.mine` oracles. |
| T1 | `pytest -q -p no:cacheprovider -x tests/test_sandbox.py tests/test_v5_innovation_caps.py` | **17 passed in 8.4 s.** The only meter test (`test_sandbox.py:93-96`) checks determinism, not tamper resistance, so the suite passes on a meter the program can rewrite. |

### S.2 Overturned subsystem verdicts

**Sandbox (`sandbox/runner.py`, `boot.py`): RETAIN -> REPAIR (M).**

Isolation holds: Landlock, seccomp, the `nobody` user, and no exec or sockets. But the sandbox is also the
accounting and attestation boundary, and that part runs inside the user's own interpreter:
- `_LINES`, monitoring tool 4 and `_IMPORTING` (`boot.py:241-296`).
- `_cpu_ms` (`boot.py:227-229`).
- The pkg tracer `_pkgtrace` (auditor 5.1).

The metered program can rewrite all of them (S1, S2), and no invariant notices.

Work needed:
- Produce work accounting and execution or import attestation outside the user's address space (parent side).
- Reject non-finite or negative values.
- Add red-team tests for meter and tracer tampering.

**Cloud hosting and metering (`services/cloud.py`, `sandbox/bridge.py`): REPAIR -> REDESIGN (L).**

The cloud trusts whatever the worker reports:
- `cu = out["cpu_ms"] * work_scale` (`cloud.py:212`), and `ap_usage` folds it, negative or zero, into `used_cu` and
  the `cu_est` EMA (`cloud.py:763-777`).
- The capacity check (`cloud.py:175-181`) uses that EMA, so one forged report disables overload for the deployment.

Clamping is a one-line repair, but it does not fix the problem. A deterministic meter that the metered code cannot
reach, and that also charges builtins and C calls (auditor 5.2), needs a different execution substrate. Examples are
parent-side fuel or instruction metering (a fuel-metered interpreter or WASM runtime) or out-of-process tracing.
`perf` and eBPF are unavailable here.

This is cross-cutting. It affects every world using `cost_model: deterministic`:
- `innovation/worlds.py:89`
- `population/worlds.py:35`
- `training/league.py:248`
- `worldgen/ow_families.py:152`
- `redteam/v5/worlds.py:70`
- the semantic scenarios

Counterfactual forks remain "exact", but they exactly reproduce forged numbers.

**Scripted reference agents (`innovation/reference.py`): RETAIN -> REPAIR (S).**

Evidence from S7 and S4:
- **Labelling.** Part 10.3 says a scripted policy must be "labelled accurately". These creators join as
  `kind="agent"` (`reference.py:34-35`), and the evaluator then counts them as agents (`measure.py:24-36`).
- **Docstring claim is false.** The docstring says they "act exclusively through ordinary tools ... and decide only
  from what those tools return" (`reference.py:3-5`). They read state directly.
- **Oracle dependence.** They depend on `needs.requests` and `needs.mine`, which the observation-only default
  forbids. Without them the entrepreneur never acts (S4, blind arm).

Work needed: join as a distinct kind (`scripted`) that `is_agent` excludes, remove the state reads, and act on
observable consequences only.

**4.0 usefulness substrate (`usefulness/`): RETAIN -> REPAIR (M).**

S5 shows two in-world tool responses that carry hidden-evaluator quality scores:
- `use.probe` (`usefulness/module.py:107-160`) returns `mean_score`, computed with the category's hidden truth.
- `use.report` (`module.py:191-201`; values at `:366` and `:415`) returns `verified_quality`.

That is a dense action-quality signal: 0.24 vs 1.00 between tiers. It is live in six registered families: ow_market,
ow_frontier, ow_composite, ow_heldout, ow_tiny and ow_works (`worldgen/ow_families.py:141-150`, `187-230`). No
`reward_mode` gating exists.

Work needed: make both evaluator-only by default, or available only in an explicitly selected legacy mode. Keep the
real consequences (revenue, retention, errors).

**Needs graph (`innovation/needs.py`): EXTEND -> REDESIGN (L).**

It is not an evaluator-side labeller. It is an in-world oracle that drives the chain it then measures:
- Holders receive `needs.derived`, `needs.addressed` and `needs.resolved` (`needs.py:415-446`).
- Participants publish needs and look for substitutes only when `needs._held` says so. They read module internals,
  not even a tool (`adopt.py:207-238`).
- The entrepreneur builds only on `needs.requests` (`reference.py:99-109`).

S4: hide the oracle and the chain's last three stages vanish (L.3 = 0). Meanwhile S4's third arm shows that
participants can find a substitute from their own experience.

S6: the developer-planted `STANDARD_INTERFACES` entry (`worlds.py:45-49`) creates phantom "participant needs" that no
participant's activity requires.

Redesign:
1. **An evaluator-only labeller.** Invisible in-world by default. It is the instrument for L.2 and SP-6.4.
2. **In-world observable consequences** that actors interpret themselves: complaints and support tickets, status and
   outage pages, error and latency experience, churn and price signals.
3. **Generic observables registered by other modules** (auditor 5.7).

### S.3 Overturned requirement verdicts

| id | auditor | skeptic | why (probe) | work needed |
|---|---|---|---|---|
| J.2 | RETAIN | EXTEND | `published: True` is hard-coded for every service declaration (`measure.py:67`), so the test passes for any `caps.declare`. In S3(D) a withdrawn service still counts. The only publishers are scripted creators labelled `agent` (S7). | Same as J.1: a non-scripted agent publishing through the tools in the integrated world. Published means discoverable while live. |
| J.6 | RETAIN | REPAIR | `measure.py:136-161` counts gross `sale` txns. It ignores the `refund` the cloud posts for every failed call (`cloud.py:161-162`) and where the payer's money came from. S3(B): 30 refunded sales count. S3(C): 500 creator-funded calls count as independent. | Net of refunds; delivered (status 200) only. Apply Gate K circular-flow and related-flow detection to the funding source. Require several payers over a sustained period. |
| J.5 | EXTEND | REDESIGN | The only judge is the format's hidden-truth verifier against a fixed threshold (`adopt.py:139-146`, `158`), so 24/24 vs 1/24 holds by construction. The result comes from the demand side, which the auditor already rated REDESIGN, and the same hidden score is handed to the creator (S3A). | Fallible, participant-owned judgement (own tests, partial checks, subjective responses) in open demand. Replicate across seeds. |
| J.7 | REPAIR | REDESIGN | The CU delta rests on a meter the product can drive to -2e10 (S1, S2). A rent-only metric would be trivially positive. | Depends on the metering redesign. Then measure the effect on pre-existing resources (auditor 8). |
| L.3 | EXTEND | REDESIGN | S4: "need addressed by a new service" exists only through the needs oracle; with the oracle hidden it is 0. No signal observable to third parties reveals the shortage. | Entrepreneurs (autonomous or learner) that react to public consequences. A needs labeller that stays evaluator-side. |
| SP-6.3 | EXTEND | REDESIGN | The auditor's own P3: a novel-category service gets 0 trials, because participants hold only predefined formats and score with hidden truth. Extending the format list does not produce novel categories. | The demand-side redesign: participants whose problems are not drawn from the format registry, own tests, and diverse grounded subjective responses (6.3). |

### S.4 Verdicts checked and upheld

- **Demand side: REDESIGN, upheld and strengthened.** S3A: the hidden-truth score is also written into adoption
  notes visible to the provider and into public forum posts (`adopt.py:162-172`).
- **Scenario worlds and Gate L chain evidence, and L.4: REDESIGN, upheld.** S4 adds that chain A's later stages are
  oracle-mediated.
- **Capability registry: REPAIR, upheld.** Add that `caps.adopted` relays the adopter's free-text note, including
  the hidden score, to the provider (`caps.py:455-459`).
- **Software ecosystem: REPAIR, upheld.** The tracer fix is the same out-of-process attestation as the meter. A robust
  form is ablation: re-run the dependent's tests with the package removed.
- **Gate J/L measurement: REPAIR, upheld, but effort S -> M.** Also fix: refunds and circular funding (J.6), the
  hard-coded published flag (J.2), phantom needs (L.2), discovery of non-live services (J.3), and scripted creators
  counted as agents.
- **L.2: EXTEND, upheld.** The criteria are weak: count only needs whose holders' own activity depends on the target.
  2 of the gate's 8 emerged needs are phantom (S6).
- **J.4: EXTEND, upheld.** Failed trials of an always-failing service count as "testing" (S3B). That is defensible
  for "test or use", but adoption should require successful use.

### S.5 Additional findings and missing requirements

1. **Observation-only.** Beyond the auditor's `independent` fields and the needs events, hidden-truth quality scores
   reach creators through adopters' notes and posts, and reach operators through `use.probe` and `use.report`.
   OO.2 (workstream L) should include tool responses and free-text event payloads among its leak probe kinds.
2. **Missing requirement: metering and attestation integrity.** Work accounting, capacity enforcement and
   adoption-by-execution evidence must be unforgeable by the metered program. This has no manifest entry. J.7,
   SP-6.5, the L.2 supply needs and every exact-fork claim depend on it.
3. **Missing criterion for J.6.** Transactions net of refunds, with funding provenance outside the creator's circle.
4. **Missing criterion for L.2 and SP-6.4.** A derived need must be held only by principals whose own activity
   requires its target. Planted world interfaces must not count as emergence.
