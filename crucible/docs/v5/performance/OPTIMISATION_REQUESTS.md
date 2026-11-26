# Measured optimisation requests for the integrated civilisation world (workstream M, wave 3b)

Measured on the integration head 55433e7 + workstream M's measurement code (branch c5c/w3b-perf), 2026-11-26, world
`crucible.civ.generate("balanced", seed, population=<profile>)`, one CPU thread, the shared space's language model on
the GPU (governed jobs). Unofficial development measurements; the official ones are the gate tests.

How the shares were measured (three instruments, because each alone misleads):

* `cprof` - cProfile with the CPU timer over simulated days 2-3 of the standard profile (day 1 holds one-time
  imports and deployments). Its overhead is large for functions called millions of times (the profiled days cost
  130-142 CPU-s against 27 CPU-s for an unprofiled day), so it ranks hot spots but inflates call-heavy ones.
* `wall` - cProfile with the default (wall) timer over 3 development days and 1 standard day; it counts waiting
  (SQLite locks, sandbox pipes) as cost.
* `light` - `lightprof`: process-time accumulators around ~25 selected functions only (outermost call, inclusive),
  everything else unprofiled; the shares below marked `light` come from it (see MEASUREMENTS.md for the run).

Per-day costs without any profiler (bench5, 30-day runs): development 9-13 CPU-s per simulated day after day 1
(277-330 simulated days per CPU-hour); standard 27 CPU-s for day 1; large 210 CPU-s for day 1.

Each request names the function, its measured cost, why it costs that, and a change that keeps every recorded
outcome identical (pure memoisation, vectorisation or batching of row-independent work) unless stated otherwise.

LIGHT_TABLE_PLACEHOLDER

## Requests

### R1 - services/search.py `Search._organic` (owner: the 4.0 services; caller `economy/work.py` ECON)

* Cost (`cprof`, standard days 2-3): `Work.act_shop` 24% of the world's CPU, almost all of it in 5,554
  `search.query` calls made by `Work._candidates` (3,449 calls: one search per shopping step). `_organic` scores every
  posting of every query token (537 documents per query on average) and calls the `rank_signal` hook chain for every
  scored document: 2.98M hook calls in two days (`market.rank_signal` -> `net.rank` -> `net.page_authority`, which
  walks `web_links` and calls `goals.metrics.is_related` per linking host: 174k calls).
* Why: the rank signal of a document depends on world state (links, listings, page text), not on the query, but is
  recomputed for every (query, document) pair; `_candidates` issues the same query text (`task_text(need)`) for every
  shopper with the same need.
* Proposed change: (a) memoise the rank-signal product per document in `world.runtime`, invalidated by the appliers
  of the tables it reads (`web_links`, `hosts`, `web_pages`, `listings`): a per-table version counter bumped in those
  appliers, the memo keyed by (doc id, versions); (b) memoise `_organic(toks, kind, limit)` per tick on the same
  versions plus `search_docs`'s version (pure: reads state only; the per-searcher side effects - ad impressions,
  `learn` of hosts - stay in `ep_query`). Outputs are identical (pure functions of the same state), so 4.0 regression
  trajectories do not change. Expected: the shopping activity's cost drops by most of its 24%.

### R2 - social/consumption.py `Engine.run` / social/recsys (owner: C, social)

* Cost (`cprof`): `Engine.run` 12% (1,816 calls, 17.5 ms each), of which `recsys.features.hash_normal` 2.2% (2,466
  calls, 2.4 ms each) and `social.scale.splitmix64` (160k calls); `recsys.features.interests` 2.3%.
* Why: `hash_normal(seed, idx, dims=422, salt)` loops over the 422 dimensions in steps of two and calls
  `splitmix64` twice per step: about 3,400 numpy operations per call however few rows it has.
* Proposed change: one counter grid, bitwise identical output:
  `c = base + np.arange(1, 2 * ceil(dims / 2) + 1, dtype=np.uint64); h = splitmix64(idx[:, None] ^ c[None, :])`;
  `h1, h2 = h[:, 0::2], h[:, 1::2]`; then Box-Muller on whole arrays (`out[:, 0::2]`, `out[:, 1::2]`, trimmed to
  `dims`). About 10 numpy operations per call.

### R3 - semantic/space `preprocessing_hashes()` and the text path (owner: D, semantic)

* Cost (`cprof`): `preprocessing_hashes` 3.1% (3,304 calls from `space.reps`, 2.5 ms each): `preprocessing()` reads
  every parameter module's constants, the encoders' attributes and the LM identity and hashes them on every call.
* Why: the key must follow the live preprocessing (the skeptic's stale-key finding), but it is recomputed per
  lookup.
* Proposed change: compute it once per transaction (cache on the `ctx`/`world.runtime` with the world's seq) or keep
  a module-level fingerprint invalidated when any parameter module attribute is assigned (a `__setattr__` hook on
  the encoder objects); the key stays live and correct.
* Also: the LM path runs one forward pass per text (1,448 forward passes for 1,548 texts in a 30-day development
  run; 304 for 316 in a standard day): `space.item_vectors` is called per page/post. Batch the texts a transaction
  asks for (`population/web.py` `_coords` is called per page; `space.text_vectors` per call): one forward pass of
  up to 16-32 texts on the GPU costs about the same as one text. The shared batching service
  (`crucible.substrate.inference_batch`, BATCHING.md) can group these calls across call sites of one tick.
* `space/__init__.py` `_record_entries`: each representation is stored three times (space_items row, the
  `space.recorded` event with base64 vectors and chunks, the inference record; INT limitation 3): about 5 KB per
  copy and the largest event payloads of the world (see storage, R6).

### R4 - sandbox metering memo `DiskMemo.first` (owner: kcal, D-21 / K3)

* Cost (`wall`, standard day 1): 13.6 s of 81 s wall (17%) in `sqlite3.execute` under `sandbox.metering.bill ->
  DiskMemo.first` (1,250 calls, 10.9 ms each); CPU share small (`light`).
* Why: every sandboxed page fetch inserts its first-measurement anchor into ONE machine-wide SQLite file
  (`~/.cache/crucible/meter/memo-v1.sqlite`, 18 MB + WAL) shared by every process of every engineer: lock waits.
* Proposed change: keep the per-world in-process memo authoritative for a transaction and write anchors to the
  machine file in one transaction per world transaction (or per N anchors), `INSERT OR IGNORE` batched with
  `executemany`; read anchors for a batch of keys with one `SELECT ... WHERE key IN (...)`. Same anchors (first
  writer still wins), far fewer lock acquisitions.

### R5 - kernel and substrate per-event costs (owner: kcal, A)

* `core.canonical.canonical_bytes`: 145k calls / 2.6M `_norm` calls in two standard days (`cprof` 9%, about half of
  it the benchmark's own event counter, which no longer hashes events). The kernel hashes every event body once at
  commit; payloads are already JSON-native when emitted. Proposed: a fast path that `json.dumps(..., sort_keys=True,
  separators=...)` the payload directly when it holds only JSON-native types (one isinstance walk instead of
  rebuilding the tree in `_norm`), falling back to `_norm` otherwise; identical bytes.
* `substrate.columns.batch` -> `artifacts.put_array_blob` -> `npz_bytes`: 10,116 blobs in two standard days, about
  1 ms each in `np.savez` (zip container, per-array headers) for blobs of a few hundred bytes (`cprof` 3.6%).
  Proposed: a raw concatenated format (fixed header: names, dtypes, shapes; then bytes) for small batches; or
  aggregate the one-row batches (R7).
* Scheduler bookkeeping: every periodic process firing writes three events (`sched.add`, `proc`, `sched.del`):
  18,625 + 18,618 + 18,618 of 289k events in a 30-day development run. One event per firing would remove two thirds
  of them (storage, R6).

### R6 - storage growth (DEF-M-3; owners: kcal for the event/blob formats, B, C, D for their event volume)

* Measured (30-day development civ runs, 3 seeds): raw 7.3-7.6 KB per individual-day written (events as stored
  2.9 KB + array blobs 4.6 KB), against target T6 of 550 B (after compaction). See MEASUREMENTS.md for the
  per-type breakdown (`storage_breakdown.py`).
* Largest event families (development, 30 days): `artifact.stored` 53k (one per blob), `cols.batch` 49k,
  `bank.txn` 30k, scheduler bookkeeping 56k, `cloud.usage` 15k, `net.hit`/`net.rate` 13k each, `popweb.page` 9k,
  `space.recorded` 1.4k (but each carries base64 vectors and chunks).
* Proposed: fold `artifact.stored` into the `cols.batch` event that references the blob (one event instead of two);
  one scheduler event per firing (R5); `space.recorded` without the vectors (the space_items table and the
  inference record hold them); delta-only `set` batches (the skeptic measured 36% of `set` bytes unchanged).

### R7 - population (owner: B)

* Class-2 decisions are not batched in the world: mean batch 1.29-1.48 (max 4) per recorded call over 30
  development days (134 calls for 198 decisions); the batched-vs-per-participant speedup at those batch sizes is
  1.41 (V.4 needs 2). The adaptive aggregation window (7,200 s) gathers few web decisions. Proposed: decide the web
  resource choice of every class-2 member that is due in the window in one round (`adaptive_window_s` aligned to
  the web activity), or route the choice through `inference_batch.gather` with the other activities of the tick.
* Class 3 is never batched: 120 recorded `pop.strategic.*` calls for 120 member decisions in 30 days (mean batch
  1.0): each member's `next` decision time is its own. Proposed: round `next` to a common decision grid per group
  (e.g. `decide_every_s` boundaries) so all due members of a group decide in one forward pass (SP-13.1).
* Class 4 is still unbound in the integrated tree (no `register_class_runtime(world, 4, ...)` caller on
  55433e7; D's `pop_semantic` runtime is in progress): V.1 and SP-13.1 fail until it acts.
* `money_deposit`: one one-row `pop.deposit` batch (event + blob) per depositing individual (522 in 3 development
  days; the skeptic counted 4,424 per hour at 10M). Proposed: deposit the due individuals of a tick in one batch.
* `population/web.py:786` `[j for j in range(n) if j not in set(own)]` rebuilds `set(own)` per element (quadratic;
  DEF-M-6). Hoist the set.

### R8 - civ world composition (owner: J)

* The population-only world (`population_world`, 4.0 services and generated sites) runs at 145-159 simulated days
  per CPU-hour at development, about half the civ world's rate (277-330): the sandboxed site handlers
  (`services/net.fetch` -> `cloud.execute_handler` -> sandbox worker, 2,676 fetches in two standard days) and their
  metering dominate it. Not a civ request; reported so that the two rows are not read as one.
