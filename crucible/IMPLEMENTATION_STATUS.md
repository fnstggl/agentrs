# Implementation status ledger

Status words: **DESIGNED** (documented, not built) · **IMPLEMENTED** (code exists and was exercised manually) ·
**TESTED** (automated tests run and pass) · **BENCHMARKED** (measured numbers in docs/SCALING.md) ·
**INCOMPLETE** (partially built; what is missing is stated) · **BLOCKED** (cannot be done here; why).
Never TESTED without a passing test; never BENCHMARKED without a measurement.

Branch `feat/crucible` (based on `feat/crucible-design`, with `feat/cam-2` merged for the AGENTRS adapter).
Test suite: `cd crucible && python -m pytest -q` — see the bottom of this file for the last full run.

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
| counterfactual forks, emergence metrics, trace/explain, CLI, dashboard, trajectories export, league/self-play | see section "Tooling" below |
| frozen-LLM agent, CAM 2.0 adapter, superlearner adapter, neural RL learner | see section "Learners" below |

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
