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
