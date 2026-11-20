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
