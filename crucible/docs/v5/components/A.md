# Component A: the shared world substrate (`crucible.substrate`)

Workstream A, Crucible 5.0. Contracts: [INTERFACE_CONTRACTS.md](../INTERFACE_CONTRACTS.md) section 1. Requirements:
SP-2.1 (execution modes) and SP-2.3 (shared substrate). Everything here runs on the 4.0 kernel with three small,
tested extension points; with no subscriber registered the kernel is byte-identical to 4.0.

## What exists

| file | what it is |
|---|---|
| `substrate/artifacts.py` | kernel module `artifacts` (host `artifact`): content-addressed immutable artifacts, permissions, quotas, fees, npz array blobs, export |
| `substrate/columns.py` | columnar state (`declare/col/rows/tables/batch/grow`) and kernel module `cols` (the `cols.batch` / `cols.grow` appliers) |
| `substrate/ids.py` | `pid`, `index`, `ensure_principal`, `principal_of`, `materialised` |
| `substrate/bus.py` | typed event registry, validated `emit`, same-transaction `subscribe`, `enforce` |
| `substrate/primitives.py` | `OWNERS` (20 kinds -> one module/table each), `owner`, `kinds`, `implemented`, `problems`, `check_no_duplicates`, `PRIMITIVE_EVENTS` (typed schemas of the 4.0 owners' events) |
| `substrate/assets.py` | kernel module `assets` (host `asset`): generic ownable, transferable assets with atomic sale through `bank` |
| `substrate/attention.py` | attention ledger columns, `spend`, `reset`, `check` (InvariantMonitor-compatible), `violations` |
| `substrate/inference.py` | `recorded(...)` (wraps `ctx.reserve/fill`), `records(events)`, `inputs_digest`, `weights_hash` |
| `substrate/modes.py` | execution-mode registry for Part 2.1 (A/B/C), real provider checks, built-in Mode A provider |
| `substrate/bench.py` | `python -m crucible.substrate.bench`: batch throughput and per-event costs |
| `substrate/modules.py` | imported by `crucible.load_all_modules()`; registers `artifacts`, `cols`, `assets` |

Kernel extension points (outside `substrate/`, all byte-identical for 4.0 worlds):

* `core/kernel.py`: `World.subscribers` and `World._notify` (called by `_emit` only when a subscriber exists for the
  type; not called in the rejection transaction, `Ctx.quiet`); `World.replay(..., store=None)` so replay can read
  content-addressed blobs (`World.load` passes its store); module attribution of table declarations.
* `core/state.py`: `StateStore.array_rows_mut(name, rows)` journals only the touched rows (O(cohort) rollback for
  10M-row columns); `declaring` / `declared_by` (runtime only, never in snapshots or digests).
* `services/identity.py`: `resolve_handle("p<i>")` returns the population principal itself (4.0 ids always contain
  `:`, bare handles never do, so 4.0 resolution is unchanged).
* `sandbox/bridge.py` (a 4.0 determinism bug found by the regression run): the deterministic cost model counted the
  lines of gc callbacks that other code had installed (hypothesis' gc timer) whenever a collection landed inside a
  measured region, so in-process digests depended on which tests had run before (`tests/test_scale.py` failed after
  `tests/test_kernel_props.py`). gc callbacks are set aside while counting; a 4.0 process has none, so every 4.0
  digest is unchanged.

Proof of byte identity: `tests/test_v5_substrate_kernel.py::test_kernel_is_byte_identical_to_crucible_40` replays
four scenarios (an open world, a randomised 300-command session with rejections and idempotency keys, a mechanism
world, an industrial world) and compares events, last hash and state digest with values recorded on the untouched
4.0 tree (`tests/v5_kernel_scenarios.py`, recorded from `git archive c07098b`). The 4.0 legacy-digest tests
(`tests/test_behavior.py`) also pass.

## How to use it

World spec: add `{"name": "artifacts"}` (and `{"name": "cols"}` for columns, `{"name": "assets"}` for assets).

```python
from crucible.substrate import artifacts, columns, attention, ids, bus, inference, primitives

# 1.1 artifacts (handlers)
ref = artifacts.put(ctx, data, "image/png", owner=ctx.actor, parents=[src_ref], meta={"title": "x"}, kind="image")
artifacts.publish(ctx, ref)                       # artifact.published (public)
artifacts.readable(world, "p:bob", ref)           # owner / owning org admins / grantee / holder / published
artifacts.get(world, ref); artifacts.info(world, ref)
blob = artifacts.put_array_blob(ctx, {"idx": idx, "v": values})   # npz without pickle, owner sys, kind batch
artifacts.export(world, other_store)              # copy every referenced blob when the world moves store
# tools: artifact.put {data_b64, mime, kind?, parents?, meta?, owner?}, artifact.get/info {ref}, artifact.list
#        {owner?, kind?, limit?, offset?}, artifact.publish {ref, audience?}, artifact.grant/revoke {ref, grantee},
#        artifact.usage {}
# config: max_bytes (32 MiB), quota_bytes (512 MiB), batch_max_bytes (1 GiB), fee_cents_per_mib (0),
#         max_meta_bytes (4096), max_parents (64), cache_bytes (256 MiB)

# 1.2 columns (in Module.install, then in handlers / processes)
columns.declare(world, "pop", n, {"interests": ("float16", (32,), 0.0), "wallet_cents": ("int64", (), 0)})
columns.col(world, "pop", "interests")            # read-only view
columns.batch(ctx, "pop", "consume.video", idx, {"watch_s": w}, mode="add", extra={"platform": "v"})
columns.grow(ctx, "pop", 1000)
# a module that uses columns depends on "cols" (which depends on "artifacts")

# 1.3 identities
ids.pid(7) == "p7"; ids.index("p7") == 7; ids.ensure_principal(ctx, 7, account=True)  # principal + person + account

# 1.4 typed events
bus.event_type("social.followed", schema, owner="social", doc="...")   # at import time
bus.emit(ctx, "social.followed", payload, vis=...)                     # validated in strict / typed_events worlds
bus.subscribe(world, "artifact.stored", self.on_stored, owner=self.name)   # in Module.install
bus.enforce(world)                                                     # validate every registered type (strict)

# 1.5 primitives
primitives.owner("community") -> {"module": "forum", "table": "forum_boards", ...}
class MyModule(Module): shared_facts = {"asset": "assets"}            # only kinds you own
primitives.check_no_duplicates(world)            # raises InvariantViolation, else a report

# 1.6 attention (population table "pop")
attention.declare(world, n)                       # or include attention.COLUMNS in your columns.declare
granted = attention.spend(ctx, idx, minutes, activity="browse")   # one cols.batch "attention.browse"
attention.reset(ctx)                              # the daily reset the population runs
attention.check(world)                            # [] or InvariantMonitor-style records

# 1.7 recorded inference
out = inference.recorded(ctx, "policy", model_id, inference.weights_hash(net), {"obs": obs}, lambda: run(net, obs),
                         sampling={"temperature": 1.0}, store_inputs=False)
inference.records(world.log)                      # every recorded outcome with model id/hash and inputs digest

# Part 2.1 modes
from crucible.substrate import modes
modes.register("C", "civ", package="crucible.civ", names=lambda: [...], make=make_world, kind="world")
modes.status()["execution_modes"]                 # operational modes, each provider run for real
```

### Shared primitive owners

| kind | owner module.table | | kind | owner module.table |
|---|---|---|---|---|
| identity | identity.persons | | relationship | forum.forum_follows |
| organization | identity.orgs | | community | forum.forum_boards |
| account | bank.accounts | | institution | gov.constitutions |
| money | bank.txns | | contract | contracts.contracts |
| asset | assets.assets (5.0) | | knowledge_artifact | docs.documents |
| resource_inventory | industry.stock | | tool_definition | caps.capabilities |
| content_artifact | artifacts.artifacts (5.0) | | time | kernel sched (+ meta clock) |
| product | market.listings | | provenance | kernel event log (@log) |
| service | cloud.hosts | | permission | identity.grants |
| software_repository | code.repos | | visibility | kernel event vis (@vis) |

`check_no_duplicates` passes on every one of the 74 4.0 world families and on a world holding all 16 owner modules.
One recognised 4.0 overlap is reported as a warning: `market.inventory` (marketplace goods) and `industry.stock`
(facility stock) are separate goods namespaces in the six `open_*` families that have both.

## Measured numbers

All single-threaded, on this shared machine (11.5-CPU quota, other tenants running), `python -m
crucible.substrate.bench --rows 1000000 --table-rows 10000000` through the slot scheduler, SQLite store:

| what | value |
|---|---|
| one `cols.batch` of 1,000,000 rows x 3 columns into a 10,000,000-row table, whole recorded command (random draw of the cohort, npz, SHA-256, SQLite blob write, event, applier with row journal) | 0.449 s (2.23 M rows/s) |
| of which: npz encode 16.0 MB / SHA-256 / npz decode | 0.032 s / 0.044 s / 0.014 s |
| replay from genesis of that log in a fresh world (10M-row columns created, blob read from SQLite, hash verified, decoded, applied) | 0.231 s, exact digest |
| regeneration (handlers re-run) | 0.374 s, identical hash chain |
| state digest / snapshot (16.7 MB) / fork from snapshot | 0.239 s / 0.843 s / 0.585 s, fork exact |
| kernel cost per emitted event (20,000 small events in one command) | 10.1 us |
| subscriber hook when no subscriber exists for the type | +0.0 us (-0.1, noise) |
| a no-op same-transaction subscriber | +0.9 us per event |
| 100k-row batch into a 1M-row table (same pipeline) | 0.018 s transaction, 0.017 s replay |

So one recorded cohort decision costs about 0.45 us per individual, against about 10 us for one kernel event per
individual: population-scale activity is ~20x cheaper as batches while staying recorded and hash-chained (the blob
SHA-256 is in the chained payload).

Gate measurements (tests/v5, recorded by `crucible.acceptance.v5.evidence.record` when run):

* SP-2.3: `shared_primitive_kinds` 20 (of 20 declared), 0 duplicate-fact families of 74, 240 randomised
  cross-module commands (215 failing in 7 real ways, all at the intended point; 25 effective controls touching up to
  9 event hosts), `cross_module_rollback_failures` 0, 20 contained failing sub-calls with 0 failures (each shown
  effective when not failing), replay and regeneration of the whole run exact; recorded inference: 48 cohort
  decisions by a torch MLP with OS-entropy sampling, 0 network calls during regeneration and replay, both exact,
  `inference_replay_without_rerun` true; 32 typed event types registered. About 7 s.
* SP-2.1: `execution_modes` 1 (Mode A via the 4.0 hidden-mechanism families: 25 environments, deterministic,
  responsive to levers). Modes B and C report "no provider registered" until `crucible.domains` and `crucible.civ`
  register providers; the criterion (>= 3) is therefore not met yet, and the test records the number without
  asserting the threshold (the checker decides).

Side channels closed in the artifact tools: an unreadable artifact and a missing one give the same `not_found`;
storing bytes that another principal already stores produces exactly the same visible event, result (`new` is
relative to the caller), quota charge and fee as storing new bytes, and the holder's `artifact.info` shows its own
record, not the first owner's.

Regression: the full 4.0 suite plus the v5 tests (1,238 tests in three single-thread groups through the slot
scheduler, GPU hidden): 1,235 passed, 2 skipped (GPU Qwen tests, as in the 4.0 baseline), 1 failed only because
SIGINT is ignored in background-launched processes (`test_scale.py::test_keyboard_interrupt_leaves_no_process_behind`
passes in the foreground).

## Limitations

* Bytes uploaded through the `artifact.put` tool are also in the recorded `cmd` event (arguments must be recorded
  for regeneration); large media should be produced by module code (`artifacts.put`), whose events hold only the
  REF. A tool result recorded under an idempotency key (`artifact.get` with a key) also lands in the log.
* Worlds without a store keep artifact bytes in this process's content-addressed pool for the life of the process;
  long or large runs should use a store (bytes then go to the store's blob table and caches are bounded).
* A fork or replay without a store finds its parent's blobs only in the same process (pool or registered stores).
  A world moved to another store needs `artifacts.export(world, dest_store)` before it is loaded elsewhere.
* `columns.declare` runs in `install`, so loading a snapshot of a 10M-row world briefly allocates the fill arrays
  before the snapshot's arrays replace them.
* Subscribers do not run in the rejection transaction (`cmd.rejected` and the persisted refusal events).
* Typed events: the 5.0 substrate types (artifact.*, cols.*, asset.*) and 21 events of the 4.0 primitive owners
  (`primitives.PRIMITIVE_EVENTS`: principal.add, id.person/org/member/grant, bank.account/txn, code.repo,
  cloud.deployment, forum.board/follow, docs.create, gov.constitution, market.listing, contracts.proposed/signed/
  status, caps.published, sched.add, make.stock_add/take) have schemas, checked against 4.0 logs of five families.
  The other 4.0 event types are untyped; the 4.0 schemas tolerate extra keys. `sched.add` cannot be enforced at run
  time (kernel types emitted at commit cannot carry subscribers); it is validated by the tests only.
* `relationship` and `community` are owned by the 4.0 forum tables and `product` by `market.listings`; when the
  social (C) or commerce (F) workstreams need their own tables for these facts, ownership must be moved in
  `primitives.OWNERS` (workstream A) rather than declared twice. `market.inventory` remains a recognised 4.0 overlap.
* Attention invariant tolerance: spent <= budget with a relative tolerance of 1e-5 plus 1e-3 minutes (float32).
* SP-2.1 is partial: only Mode A is operational (4.0 mechanisms); Mode A gains `crucible.rapid` and Modes B/C become
  operational only when workstreams H, I and J register providers.

## 5.0-complete repairs (D-21, recovery audit A_J_substrate_civ)

| audit item | repair | where | test |
|---|---|---|---|
| K1 `on_load` before the store was attached | `World.replay` builds from the snapshot with `from_blob(load=False)`, attaches the store, applies the events, then runs every `on_load` once | `core/kernel.py` | `tests/test_v5_substrate_resume.py` (three processes: crash after a snapshot, resume in a new process with identical hashes; replay and regeneration in new processes with 0 inference runs) |
| K2 subscribers advanced the emitter's RNG | same-transaction subscribers run in *side* contexts: their events carry `Event.via` (hashed only when set), their draws come from their own stream `("sub", owner, main-line position, event type, i)`; main-line draws and external-outcome keys use `World.main_seq` (= seq minus side events, `meta["side_events"]`); counters allocated only by side contexts go to a side `ctr` event | `core/kernel.py`, `core/state.py`, stores (`via` column) | `tests/test_v5_substrate_streams.py` (passive observer: identical events; emitting observer: identical main line and state; typed validation: identical simulation) |
| K3 run-to-run hash differences | worker replies no longer contain the per-process sandbox directory (`runner.WORKDIR_ALIAS`); deterministic line counting stops before a traceback is formatted (it depended on the worker's warm linecache, so a resumed world billed a fresh worker differently) | `sandbox/runner.py`, `sandbox/boot.py` | `tests/test_v5_substrate_streams.py::test_same_spec_and_seed_give_the_same_hashes_run_to_run`, resume test |
| A2 unbounded blob growth, `_POOL` leak | `substrate/retention.py`: `pack` (LZMA chunks, pointers, hash-verified reads), `collect` (drops module-declared superseded blobs no retained snapshot roots), `prune_snapshots`, `compact`, `stats`; snapshots record their `blob_roots`; `Module.blob_roots` / `superseded_blobs` hooks (implemented by `pop`); process pool reference-counted per store-less world (released on close or GC); SQLite stores payloads > 2 KiB compressed | `substrate/retention.py`, `substrate/artifacts.py`, `storage/*` | `tests/test_v5_substrate_retention.py` |
| A5 schemas registered as an import side effect; typed run changed the trajectory | `primitives.register_owner_events()` called explicitly from `substrate/modules.py`; `params.typed_events` makes the kernel call `bus.enforce_world` (every registered type of the world's modules, 4.0 owners included); `bus.coverage(world)` | `substrate/bus.py`, `substrate/primitives.py` | streams test (typed vs untyped identical) |
| A6 vacuous duplicate check | `substrate/facts.py`: mandatory scan of every table, column table, column and dense array of every shipped module (`audit()`), classification (owner / reviewed part, extension, private, duplicate / unreviewed), reviewed allow-list `KNOWN_DUPLICATES` with decision, target owner, workstream, migration, expiry; `primitives.problems` runs it on every world; data-level row counts per copy. D-11 resolved (`product` -> `products.products`; new kind `listing` = `market.listings`); new kinds `subscription`, `employment`, `population`, `need` | `substrate/facts.py`, `substrate/primitives.py` | `tests/test_v5_substrate_facts.py` |
| P.* historical queries | `substrate/history.py`: `state_at(t|seq)` (nearest snapshot + verified replay, read-only store), `events_of(entity)` (incl. a population row's cohort decisions), `record_history`, `column_history`, `mainline` / `mainline_digest` | `substrate/history.py` | `tests/test_v5_substrate_history.py` |
| A9 Mode A only via 4.0, Mode C check accepted any world | Mode A provider `rapid` (one family per engine, seeded random policy, determinism + responsiveness); Mode C check adds `composition(world)` (engaged domains >= 12 of 17, population reach >= 6, >= 8 shared kinds written cross-module incl. population and money, >= 12 initiator->owner pairs, no fact problems) | `substrate/modes.py` | `tests/test_v5_substrate_modes.py` (population world and all-module world refused) |

Store growth, 30-day development-profile population world (seed 11, SQLite, 281,747 events both runs):

| | before (a881d18 code) | after (weekly snapshot + `compact(prune=True)`) |
|---|---|---|
| blob bytes in the store | 161.9 MB (36,820 blobs) | 65.8 MB (36,818 blobs: 36,683 packed into 22.7 MB of chunks + 4.3 MB pointers; 38.9 MB unpacked roots/checkpoints) |
| event payload bytes | 101.4 MB | 58.9 MB |
| snapshots | 5, 14.6 MB | 4, 12.4 MB |
| SQLite file | 351.8 MB | 182.5 MB (vacuumed) |
| replay from genesis / resume, store only | - | exact (53.5 s) / exact (6.7 s) |

Cohort blobs compress about 5x; neural checkpoints (3.3-4.7 MB of float32 each) do not, and are kept while the
current state or a retained snapshot roots them. The event log remains the largest linear term.

### Skeptic review (S1-S8) of the same audit

| finding | repair | test |
|---|---|---|
| S1 columns had no writer ownership; integer values wrapped | every column is owned by its declaring module; a batch may write it only from the owner, a module of the owner's package, or a granted module (`columns.grant`); `Ctx.module` names the running module; the writer is recorded in the event (`by`) and re-checked by the applier in live execution (hand-emitted `cols.batch` cannot bypass it); integer casts and `add` sums are range-checked and refused | `tests/test_v5_substrate_columns.py` (ownership, forged writer, wrap tests) |
| S2 regeneration could not run cross-process; replay re-trains | `World.regenerate(..., store=)` reads recorded outputs from the store (fresh-process regeneration exact, 0 inference runs). Re-training on load (adaptive `on_load` rebuild from an older checkpoint plus logged updates) is counted (`backward_passes` in `tests/v5_resume_child.py`) and reported; removing it needs a checkpoint per learning step in `population/adaptive.py` (workstream B), which retention now makes affordable | gate SP-2.3 metrics `population_training_backward_passes_*` |
| S4 handle `p7` could be squatted | `p<i>` handles are reserved for population individuals (create_person/create_org refuse them, resolution never falls back to a squatter); identity invariant: one principal per handle | `tests/test_v5_substrate_columns.py::test_population_handles_are_reserved_and_cannot_be_squatted` |
| S6 a self-attested or trivial provider raised execution_modes | a provider's check only adds requirements (the registry's check always runs); modes B/C must build kernel worlds; Mode B needs >= 2 engaged domains, Mode C the composition criteria | `tests/test_v5_substrate_modes.py` |
| S7 (K7) an event could commit without its subscribers | the savepoint is taken before the triggering event is applied; a failing subscriber undoes the event too, even if the emitter swallows the error | `tests/test_v5_substrate_bus.py` |
| S5 (K5) `_unflushed` grew without a store; trimmed logs counted as "replay not checked" | store-less worlds keep no second copy; the mode check reads a trimmed log from the store or fails | `tests/test_v5_substrate_bus.py`, modes |
| S8 attention could be minted invisibly | conservation invariant (left + spent == budget since the last reset); attention columns written only through `spend`/`reset` (or by their declaring module when it creates rows) | `tests/test_v5_substrate_columns.py` |

## Wave 3 continuation (c5c/w3-kernel, on the integrated tree 87a6c54)

Re-verified the wave-1 repairs after the wave-1/2a merges; what the merges broke and what the adversarial probes
found, and the repairs:

| finding on the integrated tree | repair | test |
|---|---|---|
| D-20 (parent-measured sandbox CPU) made the same spec and seed give **different event hashes in two new processes** (157 of 1,795 events of a half-day population world differed in `cpu_ms`/`cu`, which feed cloud billing), so cross-process resume could not continue with the uninterrupted run's hashes (K1 gate metric false) | the deterministic memo's anchors (first measurement of a computation per world lineage) are shared by every process of the machine: `sandbox.metering.DiskMemo` (SQLite, first writer wins, the 3x + 1 ms tolerance rule unchanged, lineages never share anchors), `SandboxConfig.memo_path`, worlds use `metering.default_memo_path()` (`$CRUCIBLE_METER_MEMO`, `off` = per process) unless `params.meter_memo` is `process` | `tests/test_v5_substrate_memo.py` (two processes bill one anchor; without the file each measures afresh; the same population world in two new processes: identical hash/digest, different with `off`) |
| K2 tests failed: a world with one more module (or `typed_events`) has another spec hash, hence another measurement lineage, so its sandboxed requests were billed by fresh measurements | `params.meter_lineage`: an experiment's arms whose specs differ share a lineage (forks share it anyway); the K2 tests give every arm the same lineage and a new test shows the confound without it (only `ext`/`cloud.usage` differ) | `tests/test_v5_substrate_streams.py` |
| the mandatory scan found 3 unclassified wave-2 tables | `pop_social.cols:pop` (population extension), `space.space_items` (derived representations of artifacts), `space_read.space_read_items` (private) | `tests/test_v5_substrate_facts.py` |
| **blind spot of the scan**: columns of the owner's own column table were never checked - `pop.contacts` (who knows whom: a fourth tie graph), `pop.communities` (memberships of their own) and the audience engine's creator follower sets (a third follow graph) were silent duplicates | every column of every column table is scanned on its own name; the three are duplicates on the allow-list with migrations (B+C, C); `pop.wallet_cents` reviewed (a pre-ledger claim deposited through `bank.move`); the `edge` signature no longer matches `knowledge` | `test_every_column_is_scanned_on_its_own_name` |
| SP-2.3 counted all 25 OWNERS kinds (6 of them refinements the spec does not list), so 19 could be reached while Part 2.3 primitives still had copies | `primitives.SPEC_PRIMITIVES` (the 19 Part 2.3 primitives in the spec's words, checked against MASTER_SPEC.md by the gate); `shared_primitive_kinds` counts only those; refinements reported apart | gate SP-2.3 |
| 4.0 tables (forum follows/boards, gov constitutions, 4.0 populations) can never be retired (4.0 worlds stay byte-identical), so their kinds could never count | facts role `legacy`: a 4.0 table holding a kind's facts only in worlds without the kind's owner; enforced on data in every world (`facts.legacy_split`: facts in it while the owner is installed = problem) and certified by a named test (`verified_by`); a legacy table is not a known duplicate | `test_a_legacy_table_never_holds_facts_where_the_owner_is_installed` |
| the Mode C composition check accepted a hub-and-spoke world (one scripted driver calling into 13 domains plus a population in 6 satisfies domains, reach and pairs) | criterion `initiators`: >= 5 distinct sources (the population counts once) engage domains outside their own | `test_a_scripted_hub_that_calls_into_every_domain_is_not_an_integrated_civilization` |
| SP-2.3 fault injection found only 30 multi-module targets in 18 simulated hours of the merged all-module world (floor 40) | search window two simulated days | gate SP-2.3 |

Blocked by the concurrent integration fix (not A's code): the merged tree registers two kernel modules named
`space` (`population/semantics.py`, `semantic/space/store.py`); the population's `space.recorded` events reach the
other module's applier (`KeyError: 'items'`), so every population tick fails, and `civ.generate` refuses to build
(`missing dependencies ['space']`). Every A test that runs a population world fails on the tree until that lands
(resume, streams, retention, history, the data-level fact report, the pop-only Mode C control, the cross-process
memo test). They were run in a scratch mirror of this branch that re-registers `population.semantics.Space` as
`space` (probe only, never committed): all pass there, as do the gates' measurements below.

Measured in that mirror (single CPU, slot scheduler):

* gate SP-2.3: `shared_primitive_kinds` = 14 of the 19 Part 2.3 primitives (content artifacts, relationships,
  communities, institutions, contracts have known duplicates; all 19 have an owner); `cross_module_rollback_failures`
  = 0 (60 fault-injected real multi-module transactions of a 99-module world: 55 jobs, 5 commands, up to 9
  modules each, 47 touching v5 modules; 240 composed scenarios, 215 failing as intended); `inference_replay_without_rerun`
  = true (139 recorded population inferences; replay and regeneration in new processes: 0 inference runs, exact;
  cross-process resume exact with the shared memo).
* gate SP-2.1: `execution_modes` = 1 (A: rapid 9/9 engines responsive, 63 families, and the 4.0 mechanisms; B: the
  domains provider lists no environments; C: `civ.generate` fails on the `space` dependency). The all-module control
  world (99 modules) engages 8 of 17 domains and is refused.
* 365 simulated days, stored population world (400 individuals, adaptive participants checkpointing daily, 6
  sandboxed sites), weekly snapshot + `retention.compact(prune=True)`: 1,109,544 events; live blob bytes grow at a
  constant 3.9 -> 3.7 MB/week (first vs last quarter; 222,337 of 222,353 blobs packed into 52 chunks, 159.7 MB, +
  25.8 MB pointers, 12.6 MB unpacked roots), event payloads at a constant 5.2 -> 4.8 MB/week (256 MB); snapshots
  15 retained, 119.9 MB, growing 1.4 -> 3.3 MB/week (each snapshot holds the state, which grows linearly with the
  artifact metadata rows: 222,702 at day 364); SQLite file 859 MB (unvacuumed). At the end, in a fresh store
  object: replay from genesis exact (297 s), recount of the population's 271,993 visits from blobs read through the
  packed chunks exact, resume exact (16 s). Wall time per simulated day 2.5 s (first quarter) -> 4.5 s (last).
  Most of that growth was the weekly compaction itself (19.3 s at day 365: `collect` re-read every `cols.batch`
  event and every retained snapshot was unpickled for its roots and time). Now incremental: a warm weekly
  compaction at day 372-379 takes 8.8-9.4 s including packing that week's blobs (cold, first in a process: 12.4 s);
  the remaining O(blobs) term is two blob-size scans (1.8 s at 222k blobs). Not re-run over a full year.
* History on that compacted store: `history.state_at(seq)` (nearest retained snapshot + replay) equals replay from
  genesis at days 45, 150 and 300 (7 / 37 / 95 s against 31 / 109 / 211 s). `state_at(..., runtime=True)` at day 45
  fails with NotFound: the checkpoint current then was superseded and collected (no retained snapshot roots it) -
  historical *state* is exact at any time, historical *runtime* (rebuilt weights) only where its checkpoint is kept.

## Wave 3b (c5c/w3b-kcal, on the integration head 55433e7)

What the wave-3a merges broke, what this wave's adversarial review found, and the repairs (predecessor commits
f660be6..cf11866 plus this continuation):

| finding | repair | test |
|---|---|---|
| snapshot bytes grew superlinearly: every system array blob (cohort decision, checkpoint, training batch, recorded inference) had an `artifacts` state row and an `artifact.stored` event, so each snapshot carried the whole blob history (222,702 rows, 15 snapshots = 120 MB at day 365) and retention iterated those rows | D-78: a system blob is named only by the hash-chained event that uses it (no state row, no event); lineage-scoped reads; a store-less fork takes over its parent's pool lease (`Module.on_fork`) | `test_v5_substrate_retention.py::test_system_array_blobs_are_not_world_state` |
| two O(blobs) scans per compaction (pack and collect listed every blob size) and a log re-read | retention reads only the events written since its previous call, looks up exactly the sizes it needs, keeps its scan position in the store (verified against the event hash there) | `::test_periodic_compaction_reads_only_what_was_written_since_the_last_one` |
| no measurement of "approved state interfaces" (SP-2.3b had no gate) | `crucible.substrate.integrity`: `trace_writes` attributes every state write to the module whose applier performs it (or the handler) and classifies it against the target's declaring owner; `hashes` builds a world kind in a fresh process for determinism; gate `tests/v5/test_gate_SP_2_3b.py` with a positive control | `test_v5_substrate_integrity.py` |
| K2 observer test and K1 cross-process resume failed after the merge: the same page request was billed 0.1 ms or 2.9 ms in two runs of one spec and seed - the program-load CPU (2.9-5.8 ms on the loaded machine) straddled `LOAD_FREE_NS` and the job memo billed the noisy figure | D-80: the deterministic cost model anchors a program's load like a job (first measurement per lineage, program and target; tolerance floored at the free allowance); a platform cold start's load (the pool's first worker for a key in this host process: a resumed process, a fork) joins the job's first measurement, a restart after a death or eviction bills it on top (minimal edits in `sandbox/runner.py`, `sandbox/metering.py`) | `test_v5_substrate_memo.py::test_a_noisy_program_load_is_billed_by_its_anchor_like_a_job`, `::test_a_platform_cold_start_bills_a_request_as_the_warm_run_did`, `test_v5_substrate_streams.py`, `test_v5_substrate_resume.py` |
| approved state interfaces were a string: any module's code could pass `_iface="attention"` to `columns.batch` and refill attention unchecked | an interface is a capability of the Python module that registered it (`register_interface` records it; `batch(_iface=...)` from other code is refused) | `test_v5_substrate_columns.py::test_a_module_cannot_write_columns_it_does_not_own[claimed_iface]` |
| the write trace sees only writes through the state store's methods: a nested value of a returned row mutated in place (or `state.tables` changed directly) left no trace and no rollback entry | `integrity.replay_check`: a traced world must replay from its own log to its state; the gate counts worlds that do not as cross-module writes (second positive control) | `test_v5_substrate_integrity.py::test_a_write_that_bypasses_the_state_store_is_caught_by_replay`, gate SP-2.3b controls |
| the mandatory fact scan would fail on the integration branch as soon as the Tier 2 domain branches merge (unreviewed copies of needs, institutions, products, subscriptions) | D-79: reviewed from those branches' scans before the merge (allow-list with target owner and migration); absent entries are reported (`allow_listed_absent`), stale ones are problems | `test_v5_substrate_facts.py` |

Measured in wave 3b (one CPU, CUDA hidden; unofficial):

* K2 diagnosis on a real population world (240 individuals, 6 sandboxed sites, 6 hours, base vs emitting observer
  with a shared measurement lineage): before D-80 the first differing main-line event was `cloud.usage` (cpu_ms 0.125
  vs 5.735) - base billed one request "exceeded" (load 5.8 ms, billed 2.954 ms), the observer the same request
  "remembered" (load 3.5 ms, billed 0.097 ms); after it all 22 jobs bill "remembered" in both arms, and
  `tests/test_v5_substrate_streams.py` passes (6).
* K1, cross-process: a 1-day adaptive population world (150 individuals, 6 sites) snapshotted at 0.5 days, crashed
  at 0.75 and resumed in a new process: 13 differing events with the first D-80 step (cold starts of the resumed
  process billed their anchored load), 2 with the load tolerance unfloored (the resume itself was not
  reproducible); with D-80 completed the resumed run ends with the uninterrupted run's hash and digest, twice. At the
  resume test's own configuration (300 individuals, 8 sites, adaptive + strategic; snapshot day 1, crash 1.5, resume
  to 2.5) the resumed process ends at seq 9,717 with the hash and digest of an uninterrupted stored run in another
  new process (58 of 192 inferences re-run); replay of the stored log is exact with 0 inferences.
* Mandatory fact scan on a scratch merge of this branch with c5c/w3b-pop, -soc, -econ1, -civ, -dom1, -econ2, -sem and
  -perf: 327 items over 107 modules, 0 problems, 0 unreviewed; 17 of the 19 Part 2.3 primitives have no copy
  (relationship and community remain: forum follows/boards, `pop.contacts`, `pop.communities`, the audience engine's
  follower sets, `social.cols:follow`, `community.communities` on the allow-list).
* Gates SP-2.1, SP-2.3, SP-2.3b, the 365-day store-growth run and the adversarial civilisation probe (K1/K2,
  retention, data-level facts, `history.state_at`) need the slot scheduler (a civilisation runs 12 simulated hours
  in more than 5 minutes on one CPU without the GPU); they were queued and are reported in docs/v5/workstreams/A.json
  when they ran.

### Wave 3b, third session (after the second usage-limit stop)

* Mandatory fact scan on a fresh scratch merge (this branch + c5c/w3b-pop, -soc, -econ1, -civ, -dom1, -dom2, -dom3,
  -econ2, -sem, -perf; 2026-11-27 01:50 UTC): dom3's `orgeco.org_use` was unreviewed and the `orgeco.org_props`
  allow-list entry stale (dom3 now governs its backbone council through an `inst` charter); classified and removed.
  Then 387 items over 115 modules, 0 problems. `measure_primitives` of gate SP-2.3 on that merge (26 s on one CPU):
  shared_primitive_kinds 16 of 19 - product (dom3's `rcomm.rc_products`, allow-listed for I-3), relationship and
  community keep known copies; 74 4.0 families without duplicate facts.
* The sec/kcal conflict in `sandbox/metering.py` and `sandbox/runner.py` (per-lineage memo eviction vs D-80) has a
  tested resolution (A.json integration_notes): sandbox and security sandbox tests 36 passed and the D-80 and disk-memo
  tests passed on a trial merge.
* SP-2.3b's commerce kind on the scratch merge (one CPU): 5,784 traced writes by 13 writers into 74 targets, 0
  cross-module writes, 0 handler writes, exact replay (3,137 events). The population, social and civilisation kinds
  need the GPU slot (a population world of 400 individuals did not finish 24 simulated hours in 280 s on the loaded
  machine).
* Store growth (D-78) has a durable measurement: `PYTHONPATH=src python -m crucible.substrate.bench growth --world
  kernel|population --days 365 [--json out]` (weekly snapshot + `retention.compact(prune=True)`, per-snapshot bytes,
  retained bytes, blobs, `artifacts` rows, payload bytes, compaction seconds; exact replay from genesis and resume at
  the end). Kernel world, 365 days, 4 cohort batches a day (one CPU, 33 s): the snapshot stays at 187-192 KB from week
  1 to week 52 while 1,460 system blobs are written; 0 `artifacts` rows; 15 retained snapshots hold 2.8 MB (before
  D-78 the wave-3 population world's snapshot grew from 1.4 MB to 3.3 MB a week and 15 snapshots held 120 MB);
  compaction 3 ms (1.5-1.8 s on the weeks it packs). Regression test
  `test_v5_substrate_retention.py::test_a_year_of_cohort_batches_leaves_the_snapshot_size_unchanged`. The population
  world needs the GPU slot (the shared space's Qwen text path embeds every new page on one CPU: 6 simulated hours
  took 20 s, the next 6 more than 220 s) and was queued.
