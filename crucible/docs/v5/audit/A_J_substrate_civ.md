# Audit A + J: shared world substrate and integrated civilizations

Auditor: independent audit of `feat/crucible-5-complete` @ a881d18 (worktree `.worktrees/c5-complete`).
Scope: MASTER_SPEC Part 2.1 and 2.3 (workstream A: SP-2.1, SP-2.3), Part 10 and Gates O and P (workstream J: O.1-O.7,
O.n1, P.1-P.6, P.n1, SP-10.1, SP-10.3), read against Clarification 1 and the two new user amendments (observation-only
by default everywhere; recovery preamble: nothing counts without reproducible evidence).

All paths below are relative to `crucible/`. Probe scripts and logs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/A_J_substrate_civ/`. No existing file was modified. Several probes needed a
**probe-only shim** (`crucible.population.register_activity = crucible.population.core.register_activity`, set inside
the probe process) because, without it, no world can hold the real population and the recommendation engine together
(finding F2). Every result that depends on the shim is marked.

## 1. Verdict in one paragraph

The substrate's mechanisms (columns with batch events, content-addressed blobs, recorded inference, the transactional
kernel with same-transaction subscribers, replay, regeneration, snapshots and forks) **work at runtime**, also with 87
modules loaded together for 30 simulated days: replay and in-process snapshot/resume were exact, regeneration was exact
and re-ran no neural inference, and no invariant was violated. But the substrate does **not** give Crucible one
authoritative set of shared facts. The duplicate-fact check is opt-in and therefore vacuous (only `assets` declares
anything), and in a running all-module world the same kinds of fact sit in separate, unsynchronised places:
relationships (forum vs social graph), communities, products, institutions, contracts, employment, and at least six
independent populations of simulated people. Cross-process resume fails as soon as the adaptive population is loaded.
Turning on typed-event validation changes the simulated trajectory. **Tier 3 does not exist**: `crucible/civ` and
`crucible/multiagent` are 5-line docstrings. No generator, no profiles, no couplings, no chain tracing, no Gate O or P
test, no Mode C provider. When everything is loaded into one kernel world, the v5 social, commerce, innovation,
institution, labour and science modules stay **inert for 30 days**: 0 products, subscriptions, communities, capabilities,
packages, science runs, charters or labour records. That is exactly the "disconnected mini-environments behind a single
dispatcher" that Gate O forbids. Of the 18 requirements audited, none can pass today.

## 2. What was run

| # | command / probe | result |
|---|---|---|
| T1 | `pytest -q tests/test_v5_substrate_*.py` | 127 passed (21 s) |
| T2 | `pytest -q -x tests/v5/test_gate_SP_2_3.py tests/v5/test_gate_SP_2_1.py` | 2 passed. Recorded metrics: SP-2.3 `shared_primitive_kinds=20`, `cross_module_rollback_failures=0` (240 scenarios, **9-module world of test modules**), `inference_replay_without_rerun=true` (**test-only module** `sp23_policy`). SP-2.1 `execution_modes=1` (A only; B "provider lists no environments", C "no provider registered") |
| T3 | `pytest -q -x tests/v5/test_gate_C.py::test_C_3_consumption_produces_analytics` | **FAILED** at world construction: `ProtocolError: feed: the population module does not provide register_activity` (social/recsys/service.py:186). It is the only gate test that combines `pop`, `aud` and `feed` |
| P1 | `probe_allmods.py` (all 89 registered modules in one world, default configs) | fails 4 times before a world can be built: `aud needs the population table 'pop'` (an undeclared ordering dependency); `duplicate applier cal.cancel` (`cal`/`calendar` alias); `crowd and pop both declare the population table`; `feed: the population module does not provide register_activity`. With the shim and `crowd`/`calendar` excluded: 87 modules, created in 2.4 s |
| P2 | `probe_run.py 30` (shim; slot scheduler) | 30 simulated days at 6.8-10.4 s/day (dev population 1,042), 0 invariant errors on all 30 days, about 9.4k events/day from day 2 to day 30. Replay of the 287k-event log: digest equal (40 s). In-process snapshot: digest equal. Resume +1 day: hash and digest equal |
| P3 | `probe_kinds.py 3` (shim) | cohort decisions only from `pop`, `popweb`, `pop_adaptive`, `attention` and `feed.session.*`; `aud_item_info`=0, `communities`=0, `sub_*`=0, `products`=0, `xchg_*`=0, `capabilities`=0, `pkg_*`=0, `sci_*`=0, `inst_*`=0, `labor_*`=0, `ventures`=0, `patrons`=0; mean habit of the feed's `browse` activity -0.025 (registered with prior 0) vs `web` 1.24 |
| P4 | `probe_regen.py 2` (shim) | 26 recorded inferences (`pop.adaptive.0`, `pop.strategic.A/B`), regeneration: same last hash and digest, **0 inference executions** during regeneration and replay; 0 `proc.failed`, 0 rejections |
| P5 | `probe_resume.py` ref/part/resume (SQLite store, crash with `os._exit`, new process `World.load`) | **fails** with `NotFound: bytes of sha256:d622... are not available in this world's store` from `population/adaptive.py:281` during `on_load`, although the blob **is** in the store's `blobs` table. Same probe without `pop_adaptive`/`pop_strategic`: the resumed hash equals the uninterrupted run (`2f4e4a17...`) |
| P6 | `probe_dupfacts.py` (shim, 3 days) | `primitives.problems(world)` = [] while `forum.forum_follows`=292 (all by population individuals) and `social.follow` active edges = 0; `pop.employer` set for 735 individuals while `labor_workers`=0 and `jobs` tables=0; content items 6 vs audience items 0; populations: `pop` 1,042 rows, 4.0 `consumers` 5,000 rows, 51 `npc` persons, 7 society minds, `patrons` (0 by default) |
| P7 | `probe_typed.py` | of 77 module event types emitted in one day only 11 are registered typed events; the 21 schemas for 4.0 owner events (`bank.txn`, `id.person`, ...) are registered only if `crucible.substrate.primitives` happens to be imported |
| P8 | `probe_observer4.py` | same population world with `params.typed_events=True`: 4,371 events vs 4,291 and a different event-type sequence (two untyped runs are identical). With the subscriber RNG bump removed (probe-only monkeypatch) the typed run's sequence equals the untyped one. Root cause: `core/kernel.py:581` |
| P9 | `probe_determinism.py` | same spec + seed twice in one process: different digest and last hash (14 of 4,291 events differ). Cause: sandbox tracebacks with per-process temp paths (`/var/tmp/crucible-sbx/<pid>-<n>-<t>/service.py`) recorded in `ext` outcomes (4.0 `sandbox/boot.py:335`). Behaviour (the event-type sequence) is identical |
| P10 | `probe_writes.py` (shim, 2 days) | applier writes to tables owned by another module: only 4.0 `bank->ops_days` (484) and `shoppers->retail` (4); no handler-side state writes |
| P11 | `probe_identity.py` (shim, 3 days) | 3,451 principal references in all tables, 0 dangling, 0 `p<i>` outside the population rows |
| P12 | `probe_forks.py` (shim) | two forks with no intervention, +2 days: digest and event sequence identical (7,153 events), parent untouched |
| P13 | `probe_run.py 3` with `pop.profile=standard` (shim; slot scheduler) | day 1 15.6 s, day 2 23.8 s, **day 3 79.0 s** (wall time per simulated day grows quickly); principals 4,449; still 0 products, communities and capabilities |
| P14 | `probe_profile.py` (standard profile, cProfile of day 4) | see section 6 |

## 3. Subsystems (workstream A)

### A1 Kernel transactions, replay, regeneration, snapshots, forks (`core/kernel.py`, `core/state.py`) - REPAIR
* Works: transactions with savepoints (`_notify`, kernel.py:558-581), replay with blob store (kernel.py:997-1031),
  regeneration with recorded externals (kernel.py:1043-1091), forks (kernel.py:987). Measured exact for 87 modules
  (P2, P4, P12).
* **Defect K1 (critical for P.3):** `World.replay(..., base_blob=...)` calls `from_blob` (kernel.py:1003-1004), which
  runs every `Module.on_load` (kernel.py:965-966) **before** `w.store = store` (kernel.py:1007-1008). A module that
  rebuilds runtime state from blobs in `on_load` (`pop_adaptive._rebuild`, adaptive.py:264-281) cannot find them in a
  new process. In-process tests pass only because `artifacts._POOL` and `_SOURCES` (artifacts.py:63-77) hold the bytes.
  Fix: give `from_blob` the store before `on_load` (or defer `on_load` until after replay), and add a cross-process
  resume test (P5 is a ready template).
* **Defect K2 (observer effect):** every subscriber call advances the emitter's RNG counter
  (`ctx._rng_n += sub._rng_n + 1000`, kernel.py:581). Adding a pure validator (`bus.enforce`) or a coupling module
  therefore changes unrelated random draws downstream (P8). For counterfactual removal experiments (O.5, O.6, SP-10.3)
  this confounds "removing the coupling" with "reshuffling the randomness". Subscribers need their own RNG stream keyed
  by (event seq, subscriber owner) that does not shift the emitter's stream.
* Pre-existing (4.0) **K3:** recorded sandbox tracebacks carry per-process temporary paths (P9), so identical specs
  give different hashes run to run. That is harmless for replay but breaks hash-based independent replication (10.4)
  and makes "same spec and seed give the same event hashes" (INTERFACE_CONTRACTS 0.3) false. Normalise the paths before
  recording.
* Long horizons: the in-memory log keeps 2M events (`log_retain`, kernel.py:348, 604-609). A 365-day civilization
  (more than 3.5M events at the dev profile) needs a store, or replay from the log becomes impossible.

### A2 Content-addressed artifacts (`substrate/artifacts.py`) - REPAIR
* Hash-verified reads, quotas, permissions and export work (T1, T2).
* Storage grows without bound: every `cols.batch` writes a blob. 1,767 blobs (15.7 MB) for 2 days at the dev profile
  with adaptive participants (`resume.sqlite`), which extrapolates to roughly 2.9 GB per 365-day dev run and much more
  at the standard profile. Store-less worlds keep every byte in the process-global `_POOL` forever (artifacts.py:63).
  There is no retention policy and no compaction tied to snapshots.
* Depends on K1 for resume.

### A3 Columns and batch events (`substrate/columns.py`) - RETAIN
* About 30k `cols.batch` events in 30 days replayed exactly (P2) and regenerated exactly (P4). Row-journalled rollback
  passes the SP-2.3 scenarios (T2), and the column invariants (row counts, dtypes, finiteness, attention) held on every
  day of P2. No module writes columns outside the applier (P10).

### A4 Identity indices (`substrate/ids.py`) - RETAIN
* `p<i>` principals are created on first need and hold only `pop_index` (ids.py:46-75); 0 dangling references in P11.
* The id scheme is sound. The fact that several populations exist (A6) is not a defect of this module.

### A5 Typed event bus (`substrate/bus.py`) - REPAIR
* Same-transaction subscribers with rollback work (T2).
* Coverage is thin: 11 of 77 emitted module event types are typed (P7); validation is off unless a world is strict or
  sets `typed_events`, and turning it on changes the simulation (K2). The 4.0 owner schemas register as an import side
  effect of `primitives.py` (lines 172-179), not when the modules load.
* Real cross-module use: three production subscriptions in the whole code base (`content<-forum.post/delete`,
  social/content.py:208-209; `community<-content.published/comment`, social/communities.py:235-236;
  `firms<-attrition`, domains/firms.py:116). "All systems communicate through typed events" (Part 2.3) is not true
  of the merged system.

### A6 Shared primitives and duplicate-fact check (`substrate/primitives.py`) - REDESIGN
* `check_no_duplicates` only compares declarations: a module is seen only if it sets `shared_facts`, and only
  `assets` does (substrate/assets.py:57). The SP-2.3 duplicate scan runs on the 74 **4.0** world families only.
* Duplicates in a live all-module world (P6), none reported by the check:
  * relationship: `forum.forum_follows` (the owner, primitives.py:59; 292 follows by the population after 3 days) vs
    `social` follow edges (social/graph.py:36-41; 0 edges). The recommenders' social-graph signal reads the empty one;
  * community: `forum.forum_boards` (owner, primitives.py:63) vs `community.communities` (social/communities.py:223);
  * product: `market.listings` (owner, primitives.py:51) vs `products.products` (commerce/products.py:75) vs
    `xchg_listings` (commerce/marketplace.py:41). D-11 is still open;
  * institution: `gov.constitutions` (primitives.py:65) vs `inst.inst_charters`;
  * contract: `contracts.contracts` vs `commit.commitments` (commerce/contracts.py:69) vs subscriptions;
  * employment: 200 abstract integer employers in `pop.employer` with random job loss/finding
    (population/core.py:40, 285-288, 466-482) vs `jobs` and `labor` (commerce/labor.py:75), which never see the
    population;
  * people: `pop` (B), `crowd` (C2's stand-in, social/scale.py:1-10, which refuses to coexist with `pop`, scale.py:158),
    4.0 `consumers` (5,000 anonymous rows, pooled household account), `patrons` (commerce/customers.py:1-10, "when
    the v5 population lands, it can drive the same tools"; it never did), `npcs`, society minds, `labor` workers,
    `sci_npc`;
  * needs: 4.0 `economy.needs` (fixed formats used by `patrons` and `consumers`) vs `innovation.needs` graph;
  * content metadata: `content.content` + `catalogue.*` arrays vs `aud_items` columns + `aud_item_info` (creator,
    platform, kind, duration and title copied at indexing).
* Table write ownership itself is respected (P10). What is missing is *fact* ownership. Needed: a redesign of the owner
  map (D-11 and the same decision for relationship, community, institution, contract and employment), migration of
  the 4.0 owners or explicit adapters, one population that every behavioural class (consumers, patrons, workers,
  audience) is a view of, and a data-level duplicate detector that scans tables for the same facts instead of trusting
  declarations.

### A7 Attention ledger (`substrate/attention.py`) - RETAIN
* Its invariants run inside `Cols.check_invariants` (columns.py:297-313) and held on every day of the 30-day run (P2),
  and in SP-2.3 (T2).

### A8 Recorded inference (`substrate/inference.py`) - RETAIN
* Used by the population (adaptive.py:323, strategic.py:352), recsys fits, encoders and the semantic service. In the
  all-module world: 26 records, regeneration exact with 0 re-executions, replay 0 re-executions (P4). Divergence
  detection by model hash and inputs digest (inference.py:96-101).
* Caveat: learning steps are not recorded. They are re-executed deterministically from logged examples on load
  (adaptive.py:284-292), which assumes bit-exact torch across machines.

### A9 Execution-mode registry (`substrate/modes.py`) - EXTEND
* The checks are honest (modes.py:176-195; SP-2.1 records 1). But Mode A is served only by the 4.0 mechanisms
  (modes.py:246-268) because `crucible.rapid` registers no provider; Mode B lists 0 environments (no domain calls
  `base.register`); Mode C has no provider. The Mode C check (1-day determinism plus replay) would accept any world, so
  it cannot tell an integrated civilization from a single module.

### A10 Generic assets (`substrate/assets.py`) - EXTEND
* Correct in the SP-2.3 scenarios, but no world module issues or trades assets. Only the red-team code uses them.

## 4. Subsystems (workstream J)

### J1 Civilization generator and profiles (`civ/`) - IMPLEMENT (XL)
`civ/__init__.py` is 5 lines. There is no `generate`, `PROFILES`, `chains`, Mode C provider, Gate O/P test or
`docs/v5/components/J.md`. `experience.make(tier=3)` raises `TierUnavailable` (experience/registry.py:136-142), and
when it does work it will default to `legacy_shaped_reward` (registry.py:99; kernel.py of experience:59). That
violates amendment (a).

### J2 Integrated world composition (one kernel world holding every subsystem) - REDESIGN (XL)
* A world holding the real population and the recommendation engine cannot be built: `feed` looks for
  `crucible.population.register_activity` (social/recsys/service.py:179-187), but that function lives in
  `crucible.population.core` (core.py:48) and the package does not export it.
* The social and recommendation tests and gates D and I use the stand-in population `crowd`, which refuses to coexist
  with `pop` (tests/test_v5_social_calibrated.py:21, tests/v5/test_gate_D.py:3, tests/v5/test_gate_I.py:3). The one
  gate test that uses `pop` with `feed` fails (T3).
* `aud` needs `pop` listed before it but declares only `cols` (social/audience.py:424, 452).
* With the shim, 87 modules run for 30 days, but the v5 systems never interact (P2, P3, P6):
  * population forum posts reach the content catalogue through the forum bridge, but never reach distribution or
    audience accounting (the forum is not an `aud` platform);
  * feed sessions run on an empty inventory, and the habit for the feed's `browse` activity starts at prior 0;
  * the population never touches commerce or innovation (the population package has no reference to `prod`, `subs`,
    `caps` or `pkg`);
  * commerce customers are a separate NPC set that is empty by default.
* Only 4.0 economies (consumers/firms/cloud/net/market) and the population's web/search/forum activity produce events.

### J3 Cross-domain couplings - IMPLEMENT (L)
Three subscriptions exist, all inside the social package or one domain (A5). None links science to technology,
technology to industry costs, service usage to infrastructure demand (beyond 4.0 cloud usage), social discovery to
product adoption by the population, education to workforce, or supply chains to v5 product prices. Part 10.2's
consequences have no reusable mechanism in v5 code. 4.0 has some (cloud usage, `consumers` derived demand, industry
supply chain with energy and freight, `shocks`).

### J4 Causal chains and counterfactual removal - IMPLEMENT (M)
No `chains.trace` or `chains.counterfactual`. Reusable: forks are clean (P12); `evaluation/counterfactual.fork_compare`,
`evaluation/causal.installation_effect`, `evaluation/emergence.useful_novelty` (4.0) implement fork-and-remove for
4.0 entities. They must be generalised to v5 entities (content items, communities, products, packages, papers), and
fixed for K2.

### J5 Long-horizon persistence (30/90/180/365 days, snapshot, resume, replay, historical queries, replication) - REPAIR (L)
* Mechanisms: in-process 30-day run, replay, snapshot and resume exact (P2).
* Blocking: K1 (cross-process resume), K3 (hash replication), unbounded blob growth (A2), `log_retain` (A1), and wall
  time per day at the standard profile growing 15.6 -> 23.8 -> 79 s in 3 days (P13). Historical queries are not
  wrapped in an API (snapshot at seq plus `read_events(to_seq)` exists). 4.0's `evaluation/longrun.run_long` and
  `InvariantMonitor` are reusable but measure 4.0 sectors (households, need formats), not v5 social/economic activity.

### J6 Reusable 4.0 pieces (`worldgen/`, `worlds/`, `evaluation/`) - EXTEND (M)
`worldgen/generator.py` (family recipes for needs, consumers, firms, providers, workers, labour, energy;
`generate()` 333-356), `worldgen/admission.py`, `worlds/digital_civilization.py` (genesis of institutions, firms,
compute providers, lenders and bounties through ordinary APIs; `shocks` for exogenous changes; module name `civ`
already taken), `worlds/open_industry.py` (supply-chain graph, grid, freight), `evaluation/{counterfactual,causal,
emergence,longrun,invariants}.py`. They are tested and integrated, but they target the 4.0 economy and its fixed need
formats. A v5 generator should reuse their genesis-through-APIs pattern and fork tooling, not their populations.

## 5. Requirements

| id | verdict | reason | work needed | criteria weak |
|---|---|---|---|---|
| SP-2.1 | EXTEND | registry real; 1 of 3 modes operational (A via 4.0 only) | rapid provider (A), domain registrations (B), civ provider (C); Mode C check must require O.1-level integration | yes |
| SP-2.3 | REDESIGN | rollback and inference replay real; duplicate check vacuous; measured on test-only modules | fact-ownership redesign (A6), data-level duplicate scan of the integrated civ, rollback and inference metrics measured on real v5 modules in the civ, K2 fix | yes |
| O.1 | IMPLEMENT | no generator; all-module world inert for v5 systems | civ generator composing every Tier 2 domain plus population/social/commerce/innovation with seeded entities and active participants | yes |
| O.2 | REDESIGN | ids consistent (0 dangling) but facts duplicated; six or more disjoint populations | A6 redesign; one population; adapters for 4.0 owners | yes |
| O.3 | EXTEND | ledger invariants hold (0 violations, 30 days) but money also lives in `pop.wallet_cents` outside the ledger and in 4.0 pooled households; v5 flows absent | integrated flows through bank for every class; exogenous inflow accounting; related-party checks on the civ | yes |
| O.4 | IMPLEMENT | 3 subscriptions; no cross-system causation | couplings as subscribers/needs relations | yes |
| O.5 | IMPLEMENT | no couplings | 6+ effect types, each shown by counterfactual fork over seeds | yes |
| O.6 | IMPLEMENT | no chains tooling | trace plus counterfactual removal; exclude scripted/shock events | yes |
| O.7 | IMPLEMENT | no profiles | 3+ profiles with a measurable structural distance | yes |
| O.n1 | REDESIGN | today's all-module world is exactly disconnected modules behind one dispatcher | the J2 redesign; metric defined on data flow, not on co-loading | yes |
| P.1 | REPAIR | 30 days in process OK (dev profile, 4.0 activity only); no civ; K1; standard profile slows to 79 s/day by day 3 | civ, K1 fix, performance work, store-backed 4x2 runs | yes |
| P.2 | REPAIR | 0 violations, 0 proc.failed in probes, but no civ-level monitor | `InvariantMonitor` plus proc.failed/rejection accounting on civ runs | no |
| P.3 | REPAIR | in-process exact; cross-process resume fails with `pop_adaptive` | K1 fix; cross-process resume test | yes |
| P.4 | EXTEND | replay exact 30 days, regeneration exact, 0 inference reruns | run on civ with store at 365 days; add regeneration and K3 | yes |
| P.5 | IMPLEMENT | metric undefined, no harness | define event accounting (per-module counts vs state, batch rows vs column deltas, audience recount) | yes |
| P.6 | IMPLEMENT | steady events, but 0 v5 economic/social activity | civ with endogenous v5 activity; ratio per subsystem | yes |
| P.n1 | IMPLEMENT | no runs | 2+ independent seeds per horizon (more advisable) | no |
| SP-10.1 | IMPLEMENT | no profiles | 10 profiles differing in modules, couplings, topology, institutions, resource regimes | yes |
| SP-10.3 | IMPLEMENT | no chains | 3 chain classes confirmed by counterfactual removal | yes |

Why the criteria are weak:
* O.1 counts subsystems in one world, and P1 loads 87 with no interaction.
* O.2 relies on `duplicate_shared_facts`, which today's check reports as 0 regardless.
* O.3 is trivially 0 with no flows.
* O.4 counts event types, not causation.
* O.5, O.6 and SP-10.3 do not require counterfactual confirmation or exclude scripted events as metrics.
* O.7 and SP-10.1 have no structural-distance definition.
* P.1 does not tie runs to the population profiles or to all subsystems.
* P.3 does not require a cross-process resume.
* P.4 does not require regeneration.
* P.6 accepts a 70% decline and does not separate v5 activity.
* SP-2.3 measures rollback and inference with test-only modules.

## 6. Performance profile of a standard-profile day

All modules, `pop.profile=standard`, default configs, one CPU (P14, `probe_profile.log`).

* Wall time per simulated day: 15.0 s, 24.6 s, 77.9 s, then 105.5 s on day 4 (profiled).
* 87 of the 105 s are `pop.pr_tick`, and 78 s of those are `popweb._visit`. Each population page view is a real
  sandboxed program run: `net.fetch` -> `cloud.execute_handler` -> `sandbox.runner`. In one day that meant 2,600
  sandboxed requests and **486 sandbox worker starts** (972 `fork_exec`, 9.3 s). The rest of the 105 s was 33 s in
  epoll waits on workers, 9.3 s in `time.sleep`, and 13.7 s in `realpath` over 198k paths in `_read_paths`.
* The growth comes from worker churn as the population discovers more sites.

At this rate a 365-day standard-profile run costs at least 10 hours of wall time per seed even before any v5
subsystem is active. P.1 needs 8 such runs, plus 90- and 180-day runs, so it needs a warm worker pool or per-site
batching (one sandbox call per site per tick for the cohort), and a cap on worker restarts.

Only the population's web browsing was this costly. The substrate itself (column batches, blobs, events) was a small
share of the profile.

## 7. Missing requirements (spec text with no manifest entry)
1. Part 2.1 Mode C: an invention must be able to change economic outcomes, participant behaviour, resource demand and
   future technological opportunities.
2. Part 2.3: systems communicate through typed events and approved state interfaces (typed coverage, no cross-module
   state writes).
3. Part 2.3: deterministic simulation wherever possible. Same spec and seed should give the same hashes, which K3
   breaks; only replay is in the manifest.
4. Part 10: preserve standalone worlds for experiments; do not train exclusively on one permanent civilization.
5. Part 10.2: consequences arise from reusable interactions, not scripted ones.
6. Part 10.3: scripted narrative events do not count; scripted reference policies are labelled.
7. Part 10.4: world forks, historical queries, independent replication, horizons beyond 365 days, and persistence and
   evolution of participants, organisations, relationships, products, technologies and institutions.
8. Amendment (a) for Tier 3: `observation_only` by default for civ profiles (OO.1/OO.5 mention Tier 3, but the
   experience registry defaults to legacy and no civ exists to check).

## 8. Transfer to the open internet: shortcuts and simulator-specific artefacts
* The only live economy in the merged system is 4.0's fixed need formats (`sort.records`, ...) with a pooled-household
  consumer population. An agent placed in today's "everything" world would learn that economy, not the v5 internet.
* Population income and employment are exogenous random draws against 200 abstract employers, so no agent action
  (hiring, wages, firm failure) changes what the population earns or can spend.
* Two unsynchronised social graphs. The population follows on the forum while recommenders read the empty v5 graph,
  so reach depends on which channel the simulator happens to wire, not on social behaviour.
* Commerce customers (`patrons`) are a separate scripted NPC class with fixed needs. Products are never discovered by
  the population through social activity, so "social discovery produces adoption" cannot emerge.
* `shocks` (4.0) are scheduled exogenous events. If counted as cross-domain effects or chain stages they would be
  scripted narrative.
* K2 means removing or adding any subscriber reshuffles randomness. Naive single-seed counterfactuals would report
  spurious "effects".

## 9. What a real civilization generator needs (work plan for J)
1. Fix integration blockers first: K1, K2, export `register_activity`, one population (retire `crowd`; make
   consumers, patrons, workers and audience views or classes of `pop`), D-11 and the other ownership decisions, a
   data-level duplicate scan.
2. `civ.generate(profile, seed, population_profile) -> WorldSpec`, built like `worldgen.generator`: genesis through
   ordinary APIs (institutions, firms, providers, platforms with `aud` distribution, communities, labs, packages,
   products); all 15 domains' modules (once registered); `reward_mode` default `observation_only`.
3. Profiles as structural variants: which coupling relations exist, graph topology (fragmented vs interconnected:
   community block structure), institution types (`gov` vs `inst` charters), resource regimes (energy grid,
   compute capacity, materials), market structure (entry costs, concentration), and module sets. Add a structural
   fingerprint (module set, coupling graph, topology statistics) to measure "differ structurally".
4. Couplings as reusable mechanisms: subscribers and needs relations with their own RNG streams. Examples: science
   results feed `innovation.needs`/`caps`; adoption of packages and services drives `cloud` and energy demand;
   `industry` costs follow technology adoption; population purchases through `prod`/`subs`, discovered through
   feed/search; education and skills feed `labor`; supply disruptions feed product prices.
5. Chains: `trace(world, from_event)` over provenance (event causes, artifact parents, ledger refs), and
   `counterfactual(world, at_seq, removal)` by fork, removal and multi-seed advance with effect sizes and confidence
   intervals; exclude system-scheduled shock events from emergence counts.
6. Persistence: store-backed runs (SQLite) with periodic snapshots, blob retention/compaction, historical query API,
   cross-process resume tests, a performance budget (the standard profile must stay roughly flat per simulated day),
   and the 4.0 long-run harness extended with v5 activity measures (consumption, subscriptions, communities, packages,
   papers).


## Skeptic review

Reviewer: independent skeptic for area A_J_substrate_civ. Reviewed worktree HEAD `7c29c75`. Outside
`acceptance/v5/` (manifest derivation only), the source is identical to `a881d18`; the commits after it touch only
docs and the manifest. The manifest now has 253 entries, and none of the new OO.* and RP.* entries belongs to
workstream A or J.

The method was to try to refute every RETAIN, every REPAIR or EXTEND that might deserve a harsher verdict, and any
harsh verdict that might be unjustified. Each challenge used a probe written for this review. Probe scripts and logs
are in `/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/A_J_substrate_civ_skeptic/`. The
world construction (`common.py`) is the same as the first auditor's: all modules except calendar and crowd, with the
probe-only `register_activity` shim. No repository file other than this report and its JSON was modified.

### S.1 Outcome

All four RETAIN verdicts fall. Each one rested on evidence that the defect it misses would also pass: in-process
counters, record-table write hooks, or invariants too weak to detect the failure. Every falsifying probe below ended
with `world.check_invariants() == []`, so the corruption went undetected.

| subsystem / requirement | first verdict | skeptic verdict | decisive probe |
|---|---|---|---|
| columns and batch events | RETAIN | **REPAIR (S-M)** | S1: no writer ownership; integer values silently wrap |
| identity indices | RETAIN | **REPAIR (S)** | S4: an agent can take handle `p7` and receive payments meant for individual 7 |
| attention ledger | RETAIN | **REPAIR (S)** | S8: conservation holds in real runs, but minted attention is invisible to its invariant |
| recorded inference | RETAIN | **REPAIR (M)** | S2: replay re-runs training and fails on another CPU instruction set; regeneration cannot run cross-process |
| kernel (transactions/replay/regen/snapshots/forks) | REPAIR (M) | REPAIR, **effort L** | S2/S7/S5: four new defects K4-K7 |
| execution-mode registry | EXTEND | **REPAIR** | S6: a one-module "civilization" and a self-attesting provider raise `execution_modes` from 1 to 3 |
| long-horizon persistence | REPAIR (L) | REPAIR (L); the performance part is REDESIGN, owned by B/C | S9 |
| SP-2.1 | EXTEND | **IMPLEMENT** | S6: Modes B and C have no real provider, and the metric can be satisfied trivially |
| P.1 | REPAIR | **IMPLEMENT** | there is no civilization to run; S9 measures the cost of the Tier 3 default profile |
| P.4 | EXTEND | **REPAIR** | S2: replay depends on re-training and is not machine-independent |
| O.4 | IMPLEMENT | **EXTEND** (the first verdict was harsh) | S5: 25 cross-module event types per day already flow through schema-validated nested tool calls |
| P.2 criteria_weak | false | **true** | S1/S4/S7/S8: four kinds of corruption leave `check_invariants()` empty |

Upheld after a challenge: typed bus REPAIR, artifacts REPAIR, shared primitives REDESIGN, assets EXTEND, civ
IMPLEMENT, integrated composition REDESIGN, couplings IMPLEMENT, chains IMPLEMENT, 4.0 reuse EXTEND, and every
other requirement verdict. The pop+feed failure was reproduced independently (`probe_pop_feed.log`: `ProtocolError:
feed: the population module does not provide register_activity`; `crucible.population.register_activity` is
absent). K1 was confirmed from the first auditor's `probe_resume.log` and `probe_resume_noadaptive.log`.

### S.2 Probes and what they showed

**S1 `probe_cols_owner.py` (all-module world, 1 day): columns have no owner, and integer columns wrap.**
* `columns.batch` (columns.py:114-160) checks only that the table exists (columns.py:123). It does not check who
  writes. The kind label is free text, and the applier is owned by `cols`. As a result, the first auditor's P10
  hook, which wrapped `put/insert/patch/delete` of record tables, could not see column writes at all.
* Attributing each column to its declaring module and each `cols.batch` to its emitting module gives 2,731 of 4,482
  (61%) per-column writes in one day made by a module that did not declare the column. Most of them write `pop.*`
  columns: popweb writes `pop.wallet_cents`, `spent_cents`, `n_purchases`, `interests`, `knowledge`, `skills`,
  `beliefs` and `goals`; `feed` (workstream C) writes `pop.satisfaction` and `pop.next_active_at`; pop_adaptive and
  pop_strategic write `pop.policy_group`. popweb, pop_adaptive and pop_strategic belong to the population package, but `feed` does not, and nothing declares or enforces who may write these columns either way.
* A batch labelled `aud.consumption`, issued from a non-audience context, set `pop.spent_cents[2] = 123456789` and
  was accepted.
* Integer range: `set` of 3.0e9 into the int32 column `pop.n_purchases` stored **-2147483648**. `add` of 2 x
  (2^31-1) stored **-2**. Both were committed silently, and the world invariants were empty afterwards. The comment
  at columns.py:149 ("out-of-range values are refused just below") holds only for floats (columns.py:151, 200).
* The column tests (`tests/test_v5_substrate_columns.py:138-148`) check float16 overflow only. They would pass on
  this broken implementation.
* There are 17 integer columns of 16 bits or less on `pop`, among them `known_n` (u2), `last_activity` (i2) and
  `commit_len_min` (i2). Over 365-day horizons, wraparound is a live risk for counters.
* **Work:** declare per-column writer modules and enforce them in `batch` and the applier; record the writer in the
  event; range-check integer casts and `add` overflow; add tests for both.

**S2 `probe_replay_neural.py` and `probe_regen_diverge.py`: recorded inference is not replay-safe.**
* The probe world is pop, popweb, popsites, pop_adaptive and pop_strategic plus dependencies, with a SQLite store,
  42 simulated hours, and the pop_adaptive config `{min_examples: 4, learn_every_s: 7200}` so that online learning
  actually happens. The run produced 10 learning steps, 5 checkpoints and 54 inference records.
* The first auditor's counter wrapped only `inference.recorded`. This probe instead counts every torch
  `nn.Module.forward` and every `backward()`.
* Cross-process `World.replay` from genesis on the same machine: digest equal, but **180 forward and 32 backward
  passes**, which is 2 re-executed learning steps. `on_load` rebuilds the weights from the last checkpoint
  (`checkpoint_every: 4`) and re-runs `pol.learn` (population/adaptive.py:264-292).
* The same replay under `ATEN_CPU_CAPABILITY=default` (torch's non-vectorised kernels, standing in for another CPU)
  **fails** with `ProtocolError: adaptive group 0 step 9: rebuilt weights differ from the log`.
  `probe_learn_isa.py` confirms the cause: one `learn` call gives weights hash `218806a6...` under AVX512 and AVX2
  but `f38078a1...` under DEFAULT.
* A recorded trajectory with adaptive participants therefore cannot be replayed on a machine with a different
  instruction set, which defeats "replayed without rerunning inference" (Part 2.3) and "independent replication"
  (10.4).
* Cross-process `World.regenerate`: last hash and digest **differ**; recorded inference functions ran **50 times**.
  The first divergence is at seq 287: `ProtocolError: recorded inference output sha256:135f... is not available`.
  `World.regenerate` (kernel.py:1043-1058) has no `store` parameter, so a new process can never read recorded array
  outputs.
* Same-process regeneration is exact (`probe_regen_diverge.log`). The only reason is the process-global
  `artifacts._POOL`, which is exactly the masking the first auditor described for K1.
* `tests/test_v5_substrate_inference.py:91-122` passes because it is in-process and store-less, and its test module
  has no online learning.
* Side finding (P.2, workstream B): in this world, 38 of 42 `pop_strategic.decide` runs failed with
  `IndexError` at population/strategic.py:403 (`sorted(ctx.state.ids("providers"))[0]`, which assumes a 4.0
  compute provider exists). Strategic participants silently stop acting in any composition without 4.0 providers.
  The 87-module world hides this.
* **Work:** checkpoint (or store weight deltas as blobs) at every learning step, so that replay and load only read
  blobs and never re-train. Give `regenerate` a store. Add cross-process and cross-ISA replay and regeneration tests.

**S4 `probe_identity_squat.py`: the identity namespace is not reserved.**
* `ensure_principal` creates principal and handle `p<i>` (ids.py:46-60) without checking the handle namespace. Agents
  may register any handle matching `[a-z][a-z0-9-]{1,30}` (identity.py:34, 55-60), including `p7`.
* Agent A joined as handle `p7` (principal `p:p7`). Agent B ran `bank.transfer(to="p7", 5000)`, and the money went
  to **agent A**, because `resolve_handle` falls back to a handle search (identity.py:217-223).
* Individual 7 was then materialised. Two persons now have handle `p7` (`p:p7` and `p7`), which breaks the handle
  uniqueness that `create_person` and `create_org` enforce (identity.py:60, 75). A second transfer to the same
  string now reached individual 7.
* The same address therefore resolves to different principals over time, an agent can intercept payments and
  messages meant for individuals not yet materialised, and no invariant flags it.
* The first auditor's P11 check (0 dangling references) cannot detect this.
* **Work:** reserve `p\d+` in `create_person` and `create_org`; make `ensure_principal` refuse taken handles; add a
  handle-uniqueness invariant across persons and orgs.

**S8 `probe_attention.py`: the attention ledger.**
* The ledger's arithmetic was reproduced: in a real all-module run, `left + spent == budget` held on every row at
  3-hour checks, with error at most 2.6e-5 min.
* But `attention.check` (attention.py:127-150) tests only sign and upper bounds. A column write that refilled
  `attention_left_min` to the budget for the 17 individuals who had spent attention minted 43.8 minutes. Both
  `attention.check` and `world.check_invariants` stayed `[]`.
* Because of S1, any module can make such a write. "Invariants held for 30 days", the evidence for RETAIN, cannot
  distinguish a correct ledger from a leaking one.
* Context for C and B: the mean daily budget is 98.4 min, but the 1,042 individuals spent about 1,073 minutes in
  total per day, about 1% of the budget. The population is almost idle in the integrated world.
* **Work:** a conservation invariant (left + spent == budget since the last reset), and writer restriction to
  `attention.spend` and `attention.reset`.

**S7 `probe_subscriber_atomicity.py`: an event can commit without its subscribers.**
* `_emit` applies the event before `_notify` takes the subscriber savepoint (kernel.py:539-555 applies and appends the event; the savepoint is at kernel.py:571). A subscriber that
  fails undoes only its own effects. If the emitting code swallows the error, the triggering event commits alone.
* Real call sites do exactly that: 4.0 consumers (economy/consumers.py:612-617) and npcs (actors/npcs.py:446-452)
  wrap `forum.post` in `except Exception: pass`, and `content <- forum.post` is one of the three production
  couplings (social/content.py:208).
* With a real failure (the author's artifact quota exhausted, `CapacityExceeded`), three swallowed posts gave **3
  forum posts and 2 content items**, with no invariant reported. Through the tool path, the same failure correctly
  rejected the whole command.
* SP-2.3's `cross_module_rollback_failures = 0` measures test modules only and cannot see this.
* **Work (K7):** take the savepoint before applying the triggering event, or poison the transaction when a
  subscriber fails so that commit refuses. Audit the `except Exception` call sites.

**S5 `probe_cross.py` (all-module world, 1 day).**
* Grouping events by transaction, **25 event types** are emitted in transactions that another module initiated,
  across 23 initiator-to-emitter pairs (for example pop->forum 97, pop->id 61, pop->search 43, cloud->bank 24,
  pop_strategic->code 15).
* Only 6 of those event types come from v5 emitters. The v5->v5 pairs are pop->content, pop->popweb, pop->pop_adaptive
  and pop_strategic->content: the forum bridge and the population package.
* Two consequences:
  * The O.4 metric (`cross_system_event_types >= 10`) already passes in the disconnected world that O.n1 forbids.
    The criterion is weak, as the first auditor flagged.
  * The O.4 verdict IMPLEMENT and the A5 claim that cross-module communication is essentially absent are too harsh.
    The integration path that works is nested, schema-validated tool calls inside one transaction (`Ctx.call`
    validates arguments, kernel.py:234-252). What is missing is v5 participation and causal measurement, so O.4 is
    EXTEND.
* **K5:** without a store, `World._unflushed` grows without bound (5,046 events after 1 day; it is cleared only by
  `flush`, kernel.py:610 and 917-920), so trimming the log at `log_retain` frees nothing. With a store, trimming
  silently ends log replayability, and the mode check then treats replay as "not checked" rather than failed
  (modes.py:185-189).
* 17 runtime-only keys sit outside snapshots and the digest, among them `search.index`, `popweb.posts`,
  `recsys_private`, `pop.models` and both populations' nets. Resume correctness depends on every `on_load` rebuilding
  them (K1 is one instance).

**S6 `probe_modes_trivial.py`: the mode registry can be satisfied trivially.**
* Before: `execution_modes = 1` (A only).
* Registering, in-process, a Mode C provider whose "civilization" is the single module `identity` (3 events) and a
  Mode B provider with `check=lambda p: {"ok": True}` gave **`execution_modes = 3`**, with operational modes
  `['A', 'B', 'C']`.
* Custom checks are self-attested (modes.py:203), and the default world check accepts any deterministic world. The
  registry cannot serve as acceptance evidence for SP-2.1. Mode C's check must require O.1, O.2 and O.n1 integration
  measured on data flow.

**S9 `probe_standard_days.py`: wall time at the standard population profile.**
* Setup: all modules, `pop.profile = standard` (10,296 rows; this is what `experience.make(tier=3)` defaults to,
  experience/registry.py:136-142), run through the slot scheduler with 1 CPU.
* Per simulated day: 15.9, 24.2, 76.1, 88.9, 78.8, 87.6, 89.0, 94.9 s (days 1-8). Page views (about 2,200-2,300 per day) and known resources (32) plateau from day 3, so the steep rise on days 1-3 is site discovery, not unbounded growth. This refines the first auditor's '105 s on day 4', which was measured under cProfile. From day 3 the mean is 86 s/day, with an upward drift of about 3 s/day (least squares, days 3-8) at constant page volume; this machine is shared, so the drift is noisy. At a flat 86 s/day a 365-day run is about 8.7 h per seed, and P.1's 2 x (30+90+180+365) = 1,330 simulated days are about 32 CPU hours. That is with every v5 subsystem inert and a site set of only 32, and any growth in agent-created sites raises it.
* v5 activity stayed at 0 products and 0 communities on every day.

### S.3 Corrected requirement verdicts and criteria

* **SP-2.1 EXTEND -> IMPLEMENT.** Modes B and C have no provider: no domain registers, and `civ` is a stub. Mode A is
  served only by the 4.0 mechanisms. The check itself must be redesigned (S6) before the metric means anything.
* **P.1 REPAIR -> IMPLEMENT.**
  * The requirement is runs of a generated civilization at documented profiles. No generator exists, so nothing
    runs.
  * At the profile that Tier 3 would use by default, per-day cost is about 86 s of wall time on one CPU (S9; at least about 32 CPU hours for P.1's 1,330 simulated days, drifting upward) while every v5 subsystem is inert.
  * Work: civ (J1), K1/K4/K5, store-backed runs, and a per-day performance budget. The sandboxed page-view
    architecture is a REDESIGN item for B/C (one sandboxed program run per population page view).
* **P.4 EXTEND -> REPAIR.** Replay re-runs neural training and is ISA-dependent; regeneration cannot run in another
  process (S2).
* **O.4 IMPLEMENT -> EXTEND.** The first verdict was harsh: cross-module transactional event flow exists and works
  (S5). The work is v5 couplings, plus a metric based on causal dependence rather than event-type counts.
* **P.2 criteria_weak false -> true.** `unexplained_invariant_violations == 0` is satisfied by invariant sets that do
  not detect integer wraparound, attention minting, duplicate handles or a forum/content split (S1, S4, S7, S8). The
  criterion needs an invariant-coverage list (conservation of money, attention, inventory, content links and handle
  uniqueness) plus fault-injection evidence that each invariant fires.
* All other requirement verdicts are upheld.

### S.4 Additional missing requirements (no manifest entry)

1. Column- and table-level write ownership: approved state interfaces enforced, not by convention (S1).
2. Identity namespace integrity across agents and population principals: reserved handles, uniqueness invariant (S4).
3. Replay that is machine-independent and cross-process, and regeneration that is cross-process. A recorded
   trajectory must replay on another machine or ISA without re-training (S2).
4. Event and subscriber atomicity regardless of how the emitter handles errors (S7).
5. Invariant coverage and fault injection: each declared invariant must be shown to fire on an injected violation.
   This is what makes P.2 meaningful.
6. A realistic activity level for the integrated population, not only the late/early ratio in P.6. In S8 the
   population used about 1% of its attention budget.

### S.5 Transfer to the open internet (additions)

* No column has an owner. In any environment where agent-authored code or plug-in modules can issue column batches,
  population state (wallets, purchases, satisfaction) is writable without any recorded decision of the population.
  That is a simulator-specific lever with no real-internet analogue.
* Handle squatting on `p<digits>` is an exploit an agent can learn: register handles `p0..pN`, then collect payments
  addressed to individuals who have not yet materialised. It works only because of an artefact of how the simulator
  allocates identities.
* With an almost idle population (about 1% attention use) and inert v5 commerce and social systems, any reach or
  revenue an agent finds in the integrated world comes from 4.0 need formats and forum and search plumbing. It does
  not come from internet-like audience behaviour.
