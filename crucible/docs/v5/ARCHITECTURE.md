# Crucible 5.0 architecture

Status: design of record for the integration branch `feat/crucible-5-openworld`. Binding interfaces are in
[INTERFACE_CONTRACTS.md](INTERFACE_CONTRACTS.md). Decisions and their reasons are in [DECISION_LOG.md](DECISION_LOG.md).
The requirements are [MASTER_SPEC.md](MASTER_SPEC.md) as amended by [MASTER_SPEC_CLARIFICATIONS.md](MASTER_SPEC_CLARIFICATIONS.md);
Clarification 1 makes Crucible independent of CAM or any other learning architecture.

## 1. Starting point (measured, not assumed)

Crucible 4.0 at `c07098b` (see `docs/v5/map/*.json` for the full subsystem map):

* An event-sourced, single-writer kernel: handlers emit events, appliers write state, every event is hash-chained,
  replay and regeneration are exact, snapshots and forks are pickle copies. 42 modules, 244 endpoints.
* About 4,400 transactional commands per wall second; storage appends 39k-81k events/s in batches; a single world is
  single-threaded. 365-day worlds ran at about 11 s per simulated day with no invariant violations.
* Consumers are a NumPy population (up to 100k benchmarked) with one need each and five behaviour models for one
  decision (which API provider to use). Simulated people do not browse, watch, follow or join anything.
* Content is text only (forum, mail, docs); there is no artifact store, no media, no encoder, no recommendation beyond a
  forum hot score, BM25 and sponsored ads, and no audience accounting.
* 25 hidden-mechanism families on three engines (7 training families), a numeric learner contract
  (`EnvSpec` 61 floats in, 12 out) and a separate dict-based `Policy` contract for kernel worlds. Both leak reward:
  `MechVecEnv` writes the reward into the observation and the Gym/PettingZoo adapters return the objective.
* No subscriptions, MRR/ARR, refunds after settlement, or empirical calibration of any mechanism (0 of 55).
* Regression suite: 1,076 tests, 1,074 passed, 2 skipped (`docs/v5/evidence/gate_A/regression_baseline.json`).

Hardware: 11.5-CPU cgroup quota (48 visible), 188 GB RAM, one RTX A6000 (46 GB, shared), 26 GB free disk,
2,816-process limit, `/workspace` is **not** a persistent volume. Local models: Qwen3-0.6B, 1.7B, 4B, 14B.

## 2. Layers

```
  learners (any architecture)          scripted references, statistical learners, small neural RL, LM agents, CAM later
        |  crucible.experience: observe / execute / observe_since, reward modes, experience store, adapters
  ------+-------------------------------------------------------------------------------------------------------
  Tier 1  crucible.rapid           Tier 2  crucible.domains          Tier 3  crucible.civ
  fast controlled environments     15 semantic digital worlds        integrated persistent civilizations
  (CPU, batched, own engines)      (kernel worlds)                   (one kernel world, every subsystem)
  ------+-------------------------------------------------------------------------------------------------------
  world systems (kernel modules):  population  social  commerce  innovation  semantic participants  multiagent
                                   + 4.0 services: identity bank code cloud web search forum ads market rig use ...
  ------+-------------------------------------------------------------------------------------------------------
  crucible.substrate   artifacts (content-addressed)  columns (batch events)  ids  bus  primitives  attention  inference
  ------+-------------------------------------------------------------------------------------------------------
  crucible.core        event-sourced kernel, state store, scheduler, visibility, replay, snapshots, forks (4.0, unchanged)
```

Off to the side, never reachable from world code: `crucible.calibration` (fitted models are loaded by world code, the
datasets are not), evaluators and oracles, `crucible.acceptance.v5`, sealed sets. The red team's static-reachability
guard enforces this.

## 3. The three execution modes (Part 2.1)

| mode | what it is | package | where it runs |
|---|---|---|---|
| A controlled experiments | Tier 1 rapid environments: small, seeded, batched, scientifically controlled | `crucible.rapid` (+ 4.0 `mechanisms`) | CPU, vectorised, no kernel needed; a kernel twin exists for every family that claims one |
| B standalone semantic worlds | one Tier 2 domain with its real mechanisms and tools | `crucible.domains.<d>` | a kernel world built from the domain's modules |
| C integrated civilizations | every domain in one world with shared state, identities and accounting | `crucible.civ` | one kernel world; population and audience state are columnar |

## 4. How scale is reached without fabricating outcomes

The kernel records every state change as an event. Recording 10 million individual views as dictionary events is out
of reach at about 40k events/s, so population-scale activity uses **columnar state plus batch events**:

* Individuals are integer indices `0..N-1` into column arrays held by the kernel's `StateStore` (dense arrays are
  already part of its snapshot and digest). An individual becomes a kernel principal (`p<i>`) only when it needs one
  (a bank account, a membership record, a message), so 10M identities cost tens of bytes each.
* A handler that decides for a cohort writes the per-individual results (indices and values) into a compressed array
  blob in the content-addressed artifact store and emits one `cols.batch` event that carries the blob's SHA-256. The
  applier loads the blob, checks the hash and applies it. Replay and regeneration stay exact and the hash chain covers
  every individual decision, because the digest is in the chained payload.
* Exact audience accounting keeps a per-content bitset of viewers and of exposed identities (1.25 MB per 10M), so
  unique counts are exact, recountable from the batch events, and never written by a creator.
* Individuals act only when they have something to decide (`next_active_at` column, triggered activations).
* An explicitly labelled estimated-population mode exists for statistics beyond the materialised population; its
  outputs carry `estimated: true` and are never reported as individual-level measurements (Gate I).

## 5. The population (Part 3)

| class | count (standard) | what decides | learns |
|---|---|---|---|
| 1 behavioural | 10,000 | vectorised choice models; fitted to KuaiRand / Criteo / third dataset where possible | per-individual state (preferences, satisfaction, habits) evolves from experience |
| 2 adaptive neural | 256 | one shared small network (0.1-2M parameters) per policy family, per-individual memory columns | online parameter updates from own experience, weight hashes recorded |
| 3 strategic neural | 32 | 2-20M-parameter policy groups acting through ordinary tools via `crucible.experience` | independently updated groups, historical checkpoints |
| 4 semantic generalists | 8 | one shared Qwen3-0.6B (benchmarked against smaller and larger) behind a batched inference service | context only; outputs recorded for replay |

Profiles: Development 1,000/32/8/2; Standard 10,000/256/32/8; Large 100,000/1,000/100/20; Internet 10M+ (columnar).

## 6. The social internet (Part 4)

Platforms are kernel modules over one shared content catalogue, one social graph and one population: short and long
video, posts and images, articles and newsletters, forums (4.0), messaging, communities, creator subscriptions,
developer communities and knowledge sharing, search (4.0, extended to every content kind). A **consumption engine**
runs each platform session: candidates come from the platform's recommenders, individuals choose and watch according
to their state and the content's encoded representation, attention is spent from a finite daily budget, and every
impression, view and share is a recorded decision. A view requires watching at least `min(30 s, 50% of duration)`.

At least ten recommenders (chronological, interest retrieval, collaborative filtering, learned engagement, exploration,
social graph, diversity-aware, reputation-sensitive, search-driven, freshness) are mixed per platform; mixtures can
change on a schedule during a run, and their parameters are evaluator-side.

## 7. Semantics and calibration (Parts 4.3, 5)

Encoders are real pretrained models (text, image, audio, video = frames + audio + transcript), cached by
(artifact hash, encoder id, encoder version, preprocessing). Participants' content choice uses these representations;
no creator-assigned quality exists anywhere. Calibrated behaviour models are fitted on public datasets with held-out
evaluation and baselines; the world loads only the fitted checkpoints. Every mechanism has a fidelity record.

## 8. Economy, innovation, institutions (Part 6)

Subscriptions bill independent customers' own accounts; MRR normalises plans to a month and ARR = 12 x MRR, with
related-party and circular-flow exclusion. A capability registry lets agent-created services declare what they do,
be discovered, tested and adopted; needs arise from reusable relations between activities, resources, capabilities
and constraints. The 4.0 code host, cloud and sandbox are extended to packages, dependencies, releases and adoption by
execution.

## 9. Tier 1, Tier 2, Tier 3

* **Tier 1** adds a versioned registry of 28 causal primitives, a structural generator, eight new engine kinds, a
  general structured action interface (graph edits, discrete, sequences, queries, experiments, messages, programs,
  allocations, coordination), fingerprints and admission. Target: 60+ admitted training families and 20+ separately
  partitioned evaluation families, without touching 4.0's validation and sealed families.
* **Tier 2**: 15 domains, each a world family with at least three training configurations, one compositional and one
  structural evaluation configuration, independent outcome evaluation and a hidden scripted reference trajectory.
* **Tier 3**: a civilization generator composes every domain into one world; profiles differ in structure (which
  couplings exist, network topology, institutions, resource regimes). Cross-domain consequences use the shared
  substrate and reusable couplings; causal chains are established by counterfactual forks.

## 10. Learners (Part 11, Clarification 1)

`crucible.experience` gives one interface for every tier: `observe`, `execute`, `observe_since`, tool discovery, and
three reward modes (`legacy_shaped_reward`, `verified_terminal_outcome`, `observation_only`). Observation-only mode
removes every reward path, including the reward copy in numeric observations and the objective fields of the Gym and
PettingZoo adapters; outcome verification stays on the evaluator side. A lightweight reference learner proves real
parameter updates through this interface in all three tiers. CAM, or any other learner, connects through the same
interface with no environment change; `docs/v5/LEARNER_GUIDE.md` documents how.

## 11. Compute

Every heavy job runs through `crucible.scale.slots` (CPU threads, GPU memory, job count, usage log). Development
agents share the machine with other tenants; the default budget is 9 CPUs, 24 GB of GPU memory and 8 jobs.
