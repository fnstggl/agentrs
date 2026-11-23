# Implementation status ledger

## Crucible 4.0 "OpenWorld" (branch `feat/crucible-4`, based on `feat/crucible-internet` 11e5613)

Baseline at branch start: 305 passed. Last full run on this source: **1,057 passed**, plus **16 of 16 acceptance tests
(A-P)**. Results, negative findings and the full status by category: [docs/OPENWORLD_REPORT.md](docs/OPENWORLD_REPORT.md).
Acceptance records: [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md). Commands: [docs/REPRODUCE_OPENWORLD.md](docs/REPRODUCE_OPENWORLD.md).

| component | status | evidence |
|---|---|---|
| hidden-mechanism engines `scm`, `abm`, `techsys`; 25 families; facility module `rig`; bit-identical simulator | TESTED | `test_mech_*.py`; 288 of 300 first draws valid |
| private evaluator and validity checks | TESTED | `test_mech_evaluator.py`, `ow_validity_acceptance.json` |
| usefulness substrate (10 category families, composition, ledger) | TESTED | `test_usefulness.py`, `test_ow_worlds.py` |
| behaviour models, segments, latent and derived demand, pay-on-acceptance | TESTED | `test_behavior.py`; 3.0 digests unchanged |
| market entry, invariant monitor, long-run harness | TESTED, run to 365 days | `test_entry.py`, `test_invariants.py`, `test_longrun.py`, `ow_longrun*` |
| invention trials by matched forks | TESTED; capacity class validated over 3 seeds, pricing and frontier weak | `test_invention.py`, `ow_invention_*.json` |
| tool discovery, dry-run validation, failure diagnosis, v4 LLM interface | TESTED | `test_discovery.py`, `test_diagnostics.py`, `test_llm_interface4.py` |
| learner contract, baselines, protocols, sealed sets | TESTED | `test_learners.py`, `test_research_protocols.py` |
| self-play arena, six modes, league, cross-play, collusion audit | TESTED, experiments recorded | `test_selfplay.py`, `ow_selfplay_*.json` |
| red team: 81 probes | 79 held, 1 open (store-level log rewrite), 1 not applicable | `test_redteam.py`, `test_hardening.py`, `ow_redteam.json` |
| real-world bridge (refuses by default) | TESTED on loopback only; HTTPS path untested | `test_realworld_gate.py` |
| fidelity register | IMPLEMENTED; 0 of 55 mechanisms calibrated | `test_fidelity.py`, `ow_fidelity.json` |
| parallel worlds, rollout workers, benchmarks | TESTED; quick benchmark only | `test_scale.py`, `benchmarks/openworld.json` |
| baselines on the sealed test | MEASURED: no trained learner beat the scripted experimenter | `ow_benchmark_sealed_final.json` |
| frozen LLM and CAM 2.0 baselines | MEASURED: Qwen3-4B equals idle; CAM 0 of 8 goals | `ow_llm_baseline_summary.json`, `ow_cam_baseline.json` |
| calibration against real data, real-world pilots, VM-grade isolation, full benchmark table | NOT ATTEMPTED | see the report, section 5 |

Tests changed because they encoded behaviour the red team showed to be exploitable or obsolete: `test_internet.py` (own-site
link farm), `test_services_comm.py` (ad-budget drain), `test_web.py` (a colleague rating the seller), `test_diagnostics.py`
(per-turn truncation is now recorded), `test_entry.py` (one test moved to the deterministic cost model because it was flaky).

Status words: **DESIGNED** (documented, not built) · **IMPLEMENTED** (code exists and was exercised manually) ·
**TESTED** (automated tests run and pass) · **BENCHMARKED** (measured numbers in docs/SCALING.md) ·
**INCOMPLETE** (partially built; what is missing is stated) · **BLOCKED** (cannot be done here; why).
Never TESTED without a passing test; never BENCHMARKED without a measurement.

Branch `feat/crucible` (based on `feat/crucible-design`, with `feat/cam-2` merged for the AGENTRS adapter).
Test suite: `cd crucible && python -m pytest -q` — see the bottom of this file for the last full run.

## Crucible 3.0: synthetic open internet (branch `feat/crucible-internet`, based on `feat/crucible-open-world` 58c9b8b)

Baseline at branch start: 268 tests passed. Last full run on this branch: **305 passed** (644 s, nothing deselected). Architecture:
[docs/INTERNET_ARCHITECTURE.md](docs/INTERNET_ARCHITECTURE.md); fidelity: [docs/WORLD_FIDELITY.md](docs/WORLD_FIDELITY.md);
second-order: [docs/SECOND_ORDER_EMERGENCE.md](docs/SECOND_ORDER_EMERGENCE.md); evaluation:
[docs/LONG_HORIZON_EVALUATION.md](docs/LONG_HORIZON_EVALUATION.md); results: docs/EXPERIMENTS.md (Crucible 3.0 section).

| component | status | evidence |
|---|---|---|
| internet fabric: names, `net.fetch` as one recorded command, web apps as sandboxed route handlers, server-side sessions, consent login, per-app database, rate limits, failures | TESTED | `test_internet.py` |
| crawler, link graph, link-ranked index, `noindex`/login pages excluded | TESTED | `test_internet.py` |
| real Chromium with hermetic resolver; browser = API equivalence | TESTED | `test_internet.py::test_real_chromium_*`, `test_internet_adversarial.py::test_hermetic_*` |
| schema-first site generator (12 themes), API and browser site critics, critic rejects a faking site | TESTED | `test_internet.py` |
| AsyncRunner (independent clocks, batch groups) | TESTED, BENCHMARKED | `test_internet.py::test_async_runner_*`; SCALING.md |
| LLM social actors grounded in recorded state | TESTED (test backend) | `test_internet_society.py`; not run with a real model at scale |
| adversarial content and trust-boundary assessment | TESTED | `test_internet_adversarial.py` |
| operating modes A/B/C (Mode C refused by default) | TESTED | `test_internet_adversarial.py::test_operating_modes_*` |
| second-order mechanism (capacity exhaustion, persisted refusals, placement broker) | TESTED (mechanics), IMPLEMENTED (3-seed demonstration) | `demo_second_order.py`; SECOND_ORDER_EMERGENCE.md |
| trial designs: sequential with washout, carryover diagnostics; matched-counterfactual evaluator | TESTED | `test_open_industry.py`; `trial_methods.py` validation (both designs inaccurate for maintenance, documented) |
| industrial calibration fixes (replacement cost, pricing, affordability, B2B accounting) | TESTED (behaviour), INCOMPLETE (seeds 2-3 marginal) | `test_open_industry.py` |
| NPC invention (grammar-generated controllers, deploy, evidence-driven iteration) | TESTED | `test_npc_innovation.py` (incl. exact replay) |
| firm recovery-or-exit after losing its service | TESTED | `test_firm_recovery.py` |
| outcome metrics for open worlds; goal warm-up | TESTED | `test_taskgen.py` |
| task mining, 8-check validation, sealed held-out set, CLI `crucible tasks` | TESTED | `test_taskgen.py` |
| calibration panel with failure categories | TESTED | `test_calibration.py`; `taskgen_calibration.py` |
| Part 21 suite (`evaluation/open_internet`, 15 measures, 4 counterfactual conditions), CLI `crucible openeval` | IMPLEMENTED with recorded results | `open_internet_suite.py`; EXPERIMENTS.md |
| domain demos: A digital (second-order), B industrial (open-world demo 2), C social organisation, D information systems, E infrastructure (placement) | IMPLEMENTED with recorded results | `demo_social_org.py`, `demo_information.py`, `demo_second_order.py` |
| long-run liveness harness | IMPLEMENTED | `long_run.py` |
| minimum-isolation policy (fail closed) | TESTED; VM-grade BLOCKED on this host | `test_open_security.py` |
| internet benchmarks | BENCHMARKED | `docs/benchmarks/internet.json` |
| final challenge (broad and explicit objectives, Qwen3-4B and Qwen3-14B) | IMPLEMENTED with recorded results | `final_challenge.py`; EXPERIMENTS.md |

Bugs found and fixed on this branch:
* **Aliased state broke replay.** The NPC innovator mutated a dictionary after emitting it, and the replay hash
  chain broke. Fixed, and the kernel contract is now documented.
* **The aggregator page failed on missing effect values.** Fixed.
* **Firms kept dead services on open listings for months.** Fixed by recovery-or-exit.
* **Closing the highest-slot offerings crashed consumer demand.** Once the firms that the recovery fix wound up
  closed their listings, the hourly consumer demand process raised an `IndexError` every hour, and the whole
  digital economy stopped being served. Closed slots were still referenced by consumers' provider and
  awareness arrays. Fixed, with a regression test that fails without the fix.
* **Industrial B2B sales were booked as purchases.** Fixed.
* **The visibility task's progress was inverted.** Fixed.
* **The predictive-maintenance test outran sequential trials.** Its horizon was extended.

### Acceptance criteria (Part 33), not redefined

| # | criterion | status | evidence |
|---|---|---|---|
| 1 | A real browser can navigate a functioning synthetic internet | **met** | real Chromium: login, consent, forms, the browser critic, and the hermetic resolver (`test_internet.py`, `test_internet_adversarial.py`) |
| 2 | Interconnected persistent services | **met** | one kernel: sites use identity login, bank invoices, per-app storage; the crawler links sites; everything persists in snapshots and replays |
| 3 | Agents discover websites without a complete site list | **met** | crawler from seeds and mentions, link graph, web search index (`test_crawler_*`); the visibility tasks are scored by search rank |
| 4 | Agents write and deploy entirely new websites | **met** | agent web apps (`test_agent_web_app_*`), SiteAuthor, the EvidenceAggregator live page |
| 5 | Agent-created services expose functional APIs | **met** | capability endpoints, web JSON routes, the NPC innovators' generated controllers |
| 6 | Other agents discover and use these services | **met** | firms find capabilities through requests and search, then trial and adopt them; an LLM agent called an NPC-invented controller it found |
| 7 | Cross-service workflows produce real state changes | **met** | the placement broker rents, redeploys and releases across providers under delegated grants; invoices are paid and reported by sites |
| 8 | Concurrent autonomous actors | **met** (kernel); **not measured** at LLM scale | AsyncRunner test; 200 reasoning-free agents at 2,431 actions/s; firm minds and NPC people in every run; concurrent LLM agents not run |
| 9 | Persistent independent goals | **met** | firm minds' dynamic goals and books; NPC innovators' and entrepreneurs' routines persist in state |
| 10 | New technologies can be invented and adopted | **met** | autoscaler and placement adopted after trials (SECOND_ORDER_EMERGENCE.md); NPC invention implemented, though none of its versions was adopted |
| 11 | New market opportunities emerge from prior actions | **met** | the placement opportunity exists only after autoscaling; imitating entrants in emergent capability markets |
| 12 | At least one second-order problem emerges from an invention through reusable mechanics | **met** (one mechanism family, scripted controls) | 3-seed demonstration plus the Part 21 suite: 3 loss kinds occur only with the invention, in every seed |
| 13 | Technological improvements propagate between independent organisations | **met** | open-source forks with `improves` lineage; propagation-off removes markets and competition in every seed |
| 14 | Social relationships and negotiations affect real decisions | **met** | breach records lead to refusals and trust in memory (open-world demo 3); LLM social actors decide within grounded bounds |
| 15 | Institutions and organisations can evolve | **met** | a governed amendment passed by simulated members' votes (demo C, 30% rule); companies are founded, exit and wind up |
| 16 | Information is partially observable and propagates through working channels | **met**, with weak downstream effect | private books and evidence; sharing ablations change beliefs and trials; demo D changed beliefs and one decision; no consistent effect on adoption |
| 17 | The economy supports multiple viable long-term strategies | **partially met** | several strategies are viable in need-format worlds (v1 strategy matrix); in the open-world runs the successful scripted agents form one strategy cluster; industrial viability depends on the seed |
| 18 | The world remains dynamic across long simulated lifetimes | **partially met** | 4 x 120 days: live every week, structural change every week but one, replay identical. Change thins after about week 9 and output falls to 46-68%, because exited firms' markets are not re-entered |
| 19 | Strong agents can interact without artificial restrictions to narrow strategies | **met** | the agent gets every tool (191) with no strategy constraint; limits are the per-actor action rate and the sandbox. Two interface confounds (tool-list truncation, context truncation) were found and fixed |
| 20 | Complex objectives can be independently verified | **met** | the GoalVerifier reads metrics from state; the 8-check task validation; near-miss and binding mutations |
| 21 | Tasks include genuinely long-horizon, multi-stage problems | **met**, small set | adoption tasks need publish, deploy, independent trials and 3 days kept (competent control: 17 of 30 days); the explicit $100k/month objective; only 4 of 14 mined tasks were accepted |
| 22 | Task difficulty is calibrated against competent agents, not only random ones | **met** | a competent control is required as the feasibility witness; panel calibration; difficulty from controls; the selection effect is documented |
| 23 | The simulator remains model-agnostic | **met** | scripted, RL, CAM 2.0 and LLM policies all use the same tool interface |
| 24 | Existing CAM and superlearner adapters remain functional | **met** | `test_agents_llm.py` (13 passed, including real-Qwen CAM 2.0 replay log-prob and the superlearner world adapter) |
| 25 | Multiple RL policies can participate in one world | **met** | v1 multi-agent learning experiment and RL integration tests (unchanged, passing) |
| 26 | Snapshots, forks and replay remain correct | **met** | every demo regenerates or replays identically; a payload-aliasing bug was caught by the hash chain and fixed |
| 27 | Agent code cannot modify protected simulator infrastructure | **met** (process sandbox); VM-grade **blocked** here | sandbox and security tests; minimum-isolation fails closed |
| 28 | Browser and API interactions preserve authoritative semantics | **met** | the Chromium session's commands, replayed through the API, give an identical digest |
| 29 | Performance and scaling claims are measured | **met** | SCALING.md, `docs/benchmarks/internet.json`; LLM scale not claimed |
| 30 | Critical demonstrations are reproducible from a clean checkout | **met** (short horizon) | a fresh clone of `6d539ce` ran the second-order demo (6 days, same stages, regeneration identical) and 21 tests |
| 31 | The full regression suite passes | **met** | last full run on this branch: **305 passed** (644 s), including the real-Chromium, sandbox, SQLite and real-Qwen GPU tests |
| 32 | Known limitations are honestly documented | **met** | LIMITATIONS.md, WORLD_FIDELITY.md, negative results in EXPERIMENTS.md |

## Open world (branch `feat/crucible-open-world`, based on `feat/crucible` 8b81276)

Baseline before any change: 222 tests passed (226 s). Last full run on this branch: **268 passed** (392 s). The open-world layer is additive: legacy families, tests and
results are unchanged; open families are new (`open_digital`, `open_industry`, `open_mixed` [train],
`open_scarce_energy`, `open_fragile_suppliers` [val], `open_weak_institutions`, `open_perishable` [test, held-out
mechanisms]). Design and audit: `ARCHITECTURE.md` (open-world layer), `docs/OPEN_ENDEDNESS_AUDIT.md`.

| component | status | evidence |
|---|---|---|
| operations journal (P&L by ledger kind, derived in the bank applier) | TESTED | `test_open_substrate.py::test_operating_journal_*` (recomputed from raw ledger; exact replay) |
| loss journal (overload, outage, dependency failure, downtime, defects, stockout, supply shortfall, late delivery, curtailment) | TESTED | substrate, industry and family tests; refusal facts survive rollback (`persist`) |
| problem requests with verified evidence, responses | TESTED | substrate tests, demonstrations 1-2 |
| triggers (schedule/event, act as owner, re-checked authority, pause after failures) | TESTED | `test_triggers_run_as_owner_*` |
| typed flows (references, conditions, foreach, atomic), async jobs, public flows | TESTED | `test_flows_pass_typed_data_*` |
| capability registry: publish/versioning, search, view, install as scoped delegation, uninstall revokes | TESTED | `test_capability_install_creates_scoped_delegation_*`, `test_republishing_*` |
| verification Mode A (tester-written suites in the sandbox, provenance) | TESTED | `test_executable_test_suite_records_mode_a_*` |
| verification Mode B (randomised switchback trials computed from the installer's journal) | TESTED | adoption tests; demonstrations 1, 4 |
| verification Mode C (independent adopters/payers; related parties excluded) | TESTED | `test_related_party_installs_never_count_*` |
| verification Mode D (reviews by users only; attribute preferences of households) | IMPLEMENTED | review endpoint; shoppers |
| observed technology graph, descriptive categories, emergent market registry | TESTED | dynamics test (market with >= 2 independent buyers), census |
| Tier-2 firm minds: dynamic goals, adoption engine, sharing/withholding evidence, monitoring with outage tolerance | TESTED | adoption tests (digital and industrial), demonstrations |
| Tier-2 person minds: reservation wages, greed, patience, trust, memory, contacts, attention | TESTED | `test_open_society.py` |
| negotiation service (offers, counters, threads, binding contracts), reputation from contract outcomes | TESTED | `test_open_society.py` (6 scenarios), demonstration 3 |
| governance (constitutions in authorisation, votes, approvals, delegated budgets, governed amendment, profit sharing, incentive-based effort) | TESTED | `test_open_institutions.py` |
| production economy (hidden physics, designs, machines with wear/telemetry/maintenance, licences, runs, process knowledge, energy grids, freight, B2B trade, standing orders, retail) | TESTED | `test_open_industry.py` (conservation, physics, grid, freight, licences, regeneration) |
| household goods demand over continuous attributes | TESTED | `test_households_buy_by_attributes_*` |
| producer routines (production, maintenance, demand-driven procurement, substitution for shortage and cost, cost-plus pricing, capacity investment, operators) | TESTED (behaviour), INCOMPLETE (calibration) | industry tests; calibration limits in `docs/LIMITATIONS.md` |
| endogenous dynamics: compute pricing/investment, grid expansion, entrepreneurs entering emergent markets by forking open source, forum knowledge | TESTED | `test_open_dynamics.py` |
| open world families with held-out mechanisms | TESTED | `test_open_families_have_partitions_with_held_out_mechanisms`, per-family runs |
| independent census of open-ended structure, ablation runner | IMPLEMENTED | `evaluation/open_endedness.py`; `docs/experiments/open_ablations.json` |
| sandbox backend layer (detect, refuse unavailable; gVisor/bubblewrap/Firecracker wrappers) | TESTED (layer), BLOCKED (stronger backends on this host) | `test_open_security.py`; host has no KVM, runsc, bwrap or user namespaces |
| security of new surfaces (delegation scope, governance not bypassable by delegates, services cannot use operator tools or act for others, minds private) | TESTED | `test_open_security.py`, substrate tests |
| benchmarks of open mechanisms (publish, triggers, flows, deals, worlds, snapshot/fork/replay, minds scaling) | BENCHMARKED | `docs/benchmarks/bench.json` (`open`), `docs/SCALING.md` |
| Tier-3 LLM minds inside the society | DESIGNED | not built on this branch (agents may be LLMs; NPCs are Tier 2) |
| runtime developer-module extension with versioned migration | DESIGNED | not built |
| demonstrations 1-5 | IMPLEMENTED (scripts) with recorded results | `examples/demo_open_*.py`, `docs/experiments/open_*.json`, `docs/EXPERIMENTS.md` |

Bugs found and fixed while building (all with tests or demonstrations that exposed them): a sandboxed service's
own RPCs failed for any caller other than its operator (kernel `brokered` calls); the jobs service had no
`operator` role, so industrial hiring silently failed; routines of one mind overwrote each other's updates (stale
snapshot); capability re-publication failed; delegation made firms stand down routines unrelated to what they
delegated (a maintenance capability stopped their procurement) - scoped by delegated capability now; the
entrepreneur routine crashed the people cycle (type error) for most of a 90-day run before it was found.

### Acceptance criteria of the open-world task (Part 39)

| # | criterion | status |
|---|---|---|
| 1 | existing functionality preserved | met: the 222 original tests still pass (see last full run below) |
| 2 | agent-created capabilities not limited to the 8 needs | met: autoscaling, peak pricing, predictive maintenance, process knowledge |
| 3 | new useful product categories arise during a lifetime | met (descriptive categories/markets recorded during runs; demonstration 1) |
| 4 | independent actors discover and adopt them | met (requests, search, contacts; verified trials) |
| 5 | adoption grounded in value, preferences or verified outcomes | met (Mode B trials from the installer's own journal; related parties excluded) |
| 6 | new markets created without developer intervention | met (`caps_markets`; entrants) |
| 7 | new software interacts with existing services | met (delegated operation of cloud, market and production resources) |
| 8 | agent technologies alter economic and production outcomes | met (counterfactual forks, demonstrations 1, 2, 4) |
| 9 | technologies composable into new capabilities | partially met: composition supported (service calls, flows, designs); deep observed composition chains not demonstrated |
| 10 | multi-stage production under scarcity | met (mines -> components -> products; grids, freight, machines) |
| 11 | actors negotiate and decide independently | met (demonstration 3; tests) |
| 12 | organisations cooperate/compete through commitments | met (contracts, deals, governance, B2B trade) |
| 13 | relationships and reputation have persistent consequences | met (breach -> refusals; trust in memory) |
| 14 | institutions change through in-world actions | met (governed amendment, proposals) |
| 15 | knowledge and capabilities spread | met (evidence sharing, forking, process knowledge purchases) |
| 16 | inventions create subsequent needs/opportunities | partially met: mechanisms exist (dependency failures, entry, investment); entry and upstream investment observed; no new *problem category* observed in recorded runs |
| 17 | world evolves without developer-authored task sequences | met (no scripted shocks in open demos) |
| 18 | agent behaviour changes future opportunity structure | met (entrants into markets created by an agent's invention; counterfactuals) |
| 19 | reproducible, snapshotable, forkable | met (regeneration identical in every demonstration; replay from store of a 90-day run) |
| 20 | security boundaries enforced | met for the tested surfaces; VM-grade isolation BLOCKED on this host |
| 21 | multiple agent architectures can access the world | met (all new functionality is ordinary tools: MCP/HTTP/Gym/PettingZoo/browser adapters unchanged) |
| 22 | long-running civilisation experiments execute | met (90 simulated days, demonstration 5) |
| 23 | independent counterfactual tests show causal effects | met (demonstrations 1, 2, 4) |
| 24 | new functionality covered by tests | met (open test files listed above) |
| 25 | documented for researchers | met (architecture, audit, experiments, limitations, this ledger) |

## Core

| component | status | evidence |
|---|---|---|
| event-sourced kernel: transactions, journaled rollback, savepoints, hash chain, idempotency, rate limits | TESTED, BENCHMARKED | test_kernel.py, test_kernel_props.py (Hypothesis); 7.7k transfers/s |
| exact replay; regeneration from commands with recorded external outcomes (commands and scheduled processes) | TESTED | test_kernel*.py, test_services_comm.py, test_engineering_lab.py, demos |
| scheduler (discrete events, periodic jobs, failure recording) | TESTED | test_kernel.py |
| snapshots, forks (isolated), load = snapshot + replay, lineage | TESTED, BENCHMARKED | test_storage.py, test_goals.py; fork 23–212 ms |
| deterministic cost model (bit-identical forks) | TESTED | test_goals.py::test_goal_evaluated_independently_in_forks, test_sandbox.py |
| stores: memory, SQLite, PostgreSQL (real local cluster), crash recovery, concurrent writers | TESTED, BENCHMARKED | test_storage.py (41 tests incl. SIGKILL crash and multi-process writers) |
| schema-validated dynamic tools, discovery via knowledge, `world.describe` | TESTED | test_api.py, test_services_comm.py |
| module admission harness (schemas, random session, invariants, replay, regeneration, fork, visibility, protected hooks) | TESTED | test_admission.py (rejects a nondeterministic and a money-minting module) |

## Synthetic internet (17 services, one shared state)

| component | status | evidence |
|---|---|---|
| identity, orgs, roles, invites, grants with spend limits | TESTED | test_kernel*.py, test_goals.py |
| bank: double-entry ledger, invoices, mandates, conservation | TESTED | property tests; invariants in every test |
| mail, forum, search (+ads), docs, calendar, issues, support | TESTED | test_services_comm.py (17 tests) |
| code hosting + sandboxed run/test, deps, licences, compute charges | TESTED | test_economy.py, test_sandbox.py |
| cloud: providers, instances, deployments, metering, overload, billing, suspension, outages, hardware | TESTED | test_economy.py (unpaid bills → suspend → terminate; capacity) |
| market: API listings, code licences, goods, escrow, arbitration, verified ratings | TESTED | test_goals.py (rating spam), demos (licence adoption) |
| company registry, cap tables, insolvency/bankruptcy | TESTED | test_economy.py::test_bankruptcy_procedure (claims → distress → bankrupt: frozen accounts, terminated instance, closed listings, settled/written-off claims) |
| contracts (payment, membership, grant, shares, deliverable, notice), claims | TESTED | test_economy.py (payments, breach, claims) |
| jobs: postings, offers as contracts, verified bounties on hidden inputs | TESTED | test_economy.py, scripted Employer runs |
| research surveys (noisy, paid) | IMPLEMENTED | — |
| self-goals module | TESTED | test_goals.py |
| browser mode (pages for all services, generic pages for agent hosts, Chromium, closed network) | TESTED | test_web.py (13 tests incl. real Chromium) |
| headless ≡ browser equivalence | TESTED | test_web.py: identical commands, digest and hash chain |

## Economy, society, worlds

| component | status | evidence |
|---|---|---|
| need formats with verifiers and tiers (8) | TESTED | test_economy.py |
| consumer population (demand, satisfaction, switching, word of mouth, reviews, tickets, budget) | TESTED, BENCHMARKED | test_economy.py; up to 100k consumers |
| NPC firms, workers, lenders acting through public tools | IMPLEMENTED | exercised in demos and matrix; worker hiring path tested via Employer runs |
| licence adoption of agent-built code by NPC firms | IMPLEMENTED | demo_lifetime v2 (knapsaco2 adopted the Builder's code) |
| provenance-based trusted execution | TESTED | test_economy.py |
| world generator: 7 structural families + tiny, train/val/test splits, shocks | TESTED | test_goals.py (suites), generator used by every test |
| engineering lab world (+ held-out family) | TESTED | test_engineering_lab.py (45 tests) |

## Agents, goals, evaluation

| component | status | evidence |
|---|---|---|
| agent API: Python WorldServer, Gymnasium, PettingZoo, OpenEnv HTTP/WS, MCP stdio/HTTP | TESTED | test_api.py (17 tests, incl. pettingzoo parallel_api_test) |
| policy interface, runner, trajectory recording | TESTED | used by all agent tests |
| scripted controls (Builder, Employer, Composer, BountyHunter), Random, Idle | TESTED | test_goals.py, strategy matrix |
| goal contracts (revenue, recruit, service, solvency, outperform), generator, reward contract | TESTED | test_goals.py |
| reward-hacking probes (self-dealing, circular, rating spam, inactivity, pre-existing success, cycles, operator tools) | TESTED | test_goals.py |
| autocurriculum sampler | TESTED (unit) | test_goals.py; not yet used to train a learner (INCOMPLETE as an experiment) |
| evaluation harness, partitions, locked test with access log, paired comparison | TESTED | test_goals.py |
| candidate-learner (architecture discovery) evaluation protocol | TESTED | test_goals.py::test_candidate_evaluation_protocol_and_paired_compare |
| integrity audit | TESTED | test_goals.py |
| online multi-agent learners (pricing competition, joint-advertising public good) | TESTED (run) | EXPERIMENTS.md §7, 12 runs |

## Tooling

| component | status | evidence |
|---|---|---|
| `crucible` CLI (world create/list/inspect/run/snapshot/fork/replay/verify, actor join/run, experiment, selfplay, evaluate, trace, report, counterfactual, emergence, dashboard, serve) | TESTED | test_tooling.py (incl. SIGINT-resume subprocess test) |
| causal trace / explain, metrics export (Prometheus + JSON) | TESTED | test_tooling.py; `crucible trace` on the 30-day demo (EXPERIMENTS §3) |
| read-only researcher dashboard | TESTED | test_tooling.py (TestClient) |
| trajectory export (JSONL, npz), imitation dataset | TESTED | test_tooling.py round trips |
| league / opponent pools / Elo / held-out population, self-play runs | TESTED | test_tooling.py; EXPERIMENTS §8 |
| counterfactual fork comparison (exact under deterministic cost model) | TESTED | test_tooling.py; EXPERIMENTS §8 |
| emergence metrics (tool reuse, diffusion, HHI, specialisation, strategy diversity, dependency graph) | TESTED | test_tooling.py |
| benchmarks | BENCHMARKED | bench/run.py → docs/benchmarks/bench.json, SCALING.md |

## Learners (details and numbers in docs/RL_INTEGRATION.md)

| component | status | evidence |
|---|---|---|
| frozen-LLM policy (local Qwen3, transcripts, replay, token accounting) | TESTED | test_agents_llm.py (13 tests incl. 2 GPU); Qwen3-1.7B and Qwen3-4B lifetimes (EXPERIMENTS §9) |
| CAM 2.0 EnvironmentAdapter + AGENTRS env kind `crucible` | TESTED | protocol tests (invalid JSON counted separately, schema rejections surfaced, log-prob replay exact with prefix correction); smoke run: 2 PPO updates, 162/167 CAM tensors changed, frozen LLM unchanged |
| AGENTRS superlearner `World` adapter | TESTED | sl `reward_audit` passes on 3 Crucible families; no sl training run |
| recurrent A2C learner over factored primitive tool calls | TESTED (experiment) | learned to avoid destructive tools; did not beat idle (RL_INTEGRATION.md §6) |

## Security

| component | status | evidence |
|---|---|---|
| sandbox: uid drop, Landlock fs+TCP, seccomp (no sockets/exec/fork), rlimits, timeouts | TESTED | test_sandbox.py |
| no admin tools for actors; private data visibility | TESTED | test_kernel.py, test_services_comm.py, test_admission.py |
| web: escaping, CSP, CSRF, network closed for Chromium | TESTED | test_web.py |

## Blocked / not done

| item | status | reason |
|---|---|---|
| VM/gVisor-grade isolation | BLOCKED | unprivileged container: no KVM, no user namespaces |
| Docker-based external benchmarks (TheAgentCompany, WebArena, OSWorld) as worlds | BLOCKED | no Docker daemon; not needed for Crucible's own world |
| hundreds of concurrently active LLM agents | INCOMPLETE | one GPU; measured LLM agent costs in RL_INTEGRATION.md |
| self-generated goals → held-out performance experiment | INCOMPLETE | needs a learner that improves first |

## Acceptance criteria (Part 37 of the task statement)

| # | criterion | status |
|---|---|---|
| 1 | persistent world kernel operates correctly | met (tests, 30-day lifetime, exact replay/regeneration) |
| 2 | multiple real applications share authoritative state | met (17 services, one state) |
| 3 | agents discover and use services | met (dynamic tools via search/listings/describe; tests) |
| 4 | agents write and execute real code | met (sandboxed `code.run`/`code.test`) |
| 5 | agent software deployed and used by other actors | met (consumers, Composer RPCs into agent services, NPC licence adoption) |
| 6 | transactions/resources have real consequences | met (billing → suspension → termination; bankruptcy; capacity) |
| 7 | organisations and actors persist over time | met |
| 8 | multiple independent agents interact in one world | met (lifetime demo, multi-agent learning, league) |
| 9 | meaningful competition and cooperation | met for competition (shared markets, league); cooperation mechanisms exist (contracts, joint campaigns) and were exercised by learners, but no learned stable cooperation was demonstrated |
| 10 | multiple legitimate paths to success | met (Builder and Composer both meet the service goal in 3 families; BountyHunter earns bounty income) |
| 11 | problems/opportunities arise without task scripts | met (demand, shocks, competitors, failures; e.g. the v1 over-scaling collapse) |
| 12 | causally distinct procedural worlds | met (7 structural families + lab families) |
| 13 | snapshot, fork, replay, compare | met |
| 14 | browser and headless equivalent | met (identical commands, digest, hash chain) |
| 15 | not dependent on a model architecture | met |
| 16 | RL trajectory collection | met |
| 17 | self-play and population agents connectable | met (league, PettingZoo, multi-agent runner) |
| 18 | independent evaluation and held-out generators | met (locked test families with access log, evaluator outside the world) |
| 19 | metrics and events inspectable | met (CLI trace, dashboard, metrics export) |
| 20 | unit, integration, property, end-to-end tests pass | met (see last full run below) |
| 21 | resource consumption measured and bounded | met (SCALING.md; sandbox/runtime limits) |
| 22 | an end-to-end simulated lifetime executed | met (30 days with scripted agents; 5 days with a frozen LLM agent) |
| 23 | agent artifacts durably affect subsequent agents | met (licensed code adopted by an NPC firm; agent services used by consumers and other agents) |
| 24 | key demonstrations reproducible from a clean checkout | met for scripts in `crucible/examples` (commands in EXPERIMENTS.md §1); the RL subagent's launch scripts live in gitignored `runs/` (INCOMPLETE) |
| 25 | documented for extension | met (16 docs + this ledger) |

What remains (next steps for a continuation):
1. Learners: no learner yet acquires a broad strategy from experience. Next: imitation warm-start of CAM from the
   scripted controls' trajectories (`training/trajectories.py`), then continual RL on `service`/`revenue` level-0
   goals with the autocurriculum, matched against the frozen LLM.
2. Run the self-generated-goals experiment once a learner improves at all.
3. Conditional cost-sharing contracts in the cooperation experiment (deliverable predicate on ad spend).
4. Version the RL launch scripts (currently in `runs/`), more seeds for every reported comparison.
5. Larger LLM agents and agent populations (GPU-bound here).

Last full test run (2026-11-20, `cd crucible && python -m pytest -q`): **222 passed**, 0 failed, 1 warning, 228 s — including PostgreSQL (throwaway local cluster), real-Chromium browser tests, sandbox isolation tests and 2 GPU tests on Qwen3-0.6B.
