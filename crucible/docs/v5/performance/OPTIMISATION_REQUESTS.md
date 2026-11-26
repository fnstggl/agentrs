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

## Light attribution (`light`), civilisation world, standard profile (10,296 individuals), seed 0

Inclusive CPU of the outermost call of each function (nested functions overlap: `_organic` is inside `act_shop`,
`_compute` inside `item_vectors`). Days 2-3 (56.0 CPU-s) had their texts' representations in the machine-wide cache
from earlier runs of this seed; day 4 (57.2 CPU-s) encoded 356 new texts, which is what a fresh run pays every day
(benchmark runs therefore use a cold cache of their own from 96def85 on). Peak RAM 3.1 GB.

| function (owner) | days 2-3: CPU s / share / calls | day 4 (356 new texts): CPU s / share | ms per call |
|---|---:|---:|---:|
| services/search.py `Search._organic` (4.0 service) | 10.77 / 19.2% / 5,554 | 4.21 / 7.4% | 2.059 |
| economy/work.py `Work.act_shop` (ECON) | 12.38 / 22.1% / 160 | 4.49 / 7.8% | 70.586 |
| social/consumption.py `Engine.run` (C) | 16.91 / 30.2% / 1,816 | 8.84 / 15.4% | 9.376 |
| social/recsys/features.py `hash_normal` (C) | 4.47 / 8.0% / 2,466 | 2.16 / 3.8% | 1.781 |
| semantic/space `_compute` - encoders + LM for new texts (D) | 0.00 / 0.0% / 0 | 27.64 / 48.3% | 77.654 |
| semantic/space `item_vectors` (D) | 3.38 / 6.0% / 2,971 | 19.91 / 34.8% | 5.147 |
| semantic/space `preprocessing_hashes` (D) | 3.08 / 5.5% / 3,304 | 1.43 / 2.5% | 0.896 |
| substrate/columns.py `batch` (kcal) | 5.20 / 9.3% / 9,544 | 2.27 / 4.0% | 0.556 |
| substrate/artifacts.py `put_array_blob` (kcal) | 3.25 / 5.8% / 10,116 | 1.43 / 2.5% | 0.328 |
| substrate/artifacts.py `npz_bytes` (kcal) | 2.41 / 4.3% / 10,116 | 1.06 / 1.8% | 0.243 |
| core/canonical.py `canonical_bytes` (kcal) | 2.01 / 3.6% / 91,728 | 0.85 / 1.5% | 0.022 |
| core/kernel.py `World._commit` (kcal) | 1.53 / 2.7% / 1,240 | 0.67 / 1.2% | 1.182 |
| sandbox/bridge.py `run_in_ctx` (O/kcal) | 3.70 / 6.6% / 2,708 | 2.60 / 4.5% | 1.536 |
| sandbox/runner.py `Worker.start` (O) | 1.93 / 3.4% / 64 | 1.52 / 2.7% | 36.253 |
| sandbox/metering.py `DiskMemo.first` (kcal K3) | 0.17 / 0.3% / 2,689 | 0.18 / 0.3% | 0.086 |
| population/strategic.py `pr_decide` (B) | 2.42 / 4.3% / 48 | 2.55 / 4.5% | 69.111 |
| services/net.py `page_authority` (4.0 service) | 2.37 / 4.2% / 263,190 | 0.55 / 1.0% | 0.009 |
| population/semantics.py `text_features` (B/D) | 1.54 / 2.7% / 8,132 | 0.80 / 1.4% | 0.178 |
| population/core.py `money_deposit` (B) | 0.22 / 0.4% / 487 | 0.05 / 0.1% | 0.446 |

What it says: with new content the shared space's text path is about half of a standard day (78 ms per new text, one
text per forward pass); the social consumption engine is 15-30%; shopping searches 7-19% (growing with the index);
the substrate's batch blobs 4-9%; sandboxed site fetches 5-7%; class-3 decisions 4% for 24 decisions a day.

## Requests

### R1 - services/search.py `Search._organic` (owner: the 4.0 services; caller `economy/work.py` ECON) - the large profile's bottleneck

* Cost (`light`, large profile, 101,120 individuals, day 2): the day cost 697.6 CPU-s, of which `Work.act_shop`
  497.7 s (71.3%) and inside it `Search._organic` 486.1 s (69.7%): 33,219 searches from 23,036 `_candidates` calls,
  14.6 ms each, and 2.8M `page_authority` calls. Day 1 cost 75 CPU-s: the index (listings, pages, posts) grows with
  the world, the shoppers grow with the population, and the cost per search grows with the index, so the shopping
  cost grows faster than the population and through the run. At 697.6 CPU-s per day the large world runs 5.2
  simulated days per CPU-hour on day 2, against T3 >= 3.75; with a growing index the 30-day run is at risk.

* Cost (`light`, standard days 2-3): `Work.act_shop` 22.1% of the world's CPU, of which `_organic` 19.2%: 5,554
  `search.query` calls made by `Work._candidates` (3,449 calls: one search per shopping step); 7.4% on day 4. `_organic` scores every
  posting of every query token (537 documents per query on average) and calls the `rank_signal` hook chain for every
  scored document: 2.98M hook calls in two days (`market.rank_signal` -> `net.rank` -> `net.page_authority`, which
  walks `web_links` and calls `goals.metrics.is_related` per linking host: 174k calls).
* Why: the rank signal of a document depends on world state (links, listings, page text), not on the query, but is
  recomputed for every (query, document) pair; `_candidates` issues the same query text (`task_text(need)`) for every
  shopper with the same need. There are 8 needs (`economy.needs.NEEDS`), so the 33,219 searches of a large day ask at
  most 8 distinct questions (with one `search_limit`): about 346 searches per 15-minute tick for at most 8 distinct
  organic result lists. `Search.search` is documented as pure (state and hooks only); the per-searcher effects (ad
  impressions, `learn` of hosts) are in `ep_query`, after it.
* Proposed change: (a) memoise the rank-signal product per document in `world.runtime`, invalidated by the appliers
  of the tables it reads (`web_links`, `hosts`, `web_pages`, `listings`): a per-table version counter bumped in those
  appliers, the memo keyed by (doc id, versions); (b) memoise `_organic(toks, kind, limit)` per tick on the same
  versions plus `search_docs`'s version (pure: reads state only; the per-searcher side effects - ad impressions,
  `learn` of hosts - stay in `ep_query`). Outputs are identical (pure functions of the same state), so 4.0 regression
  trajectories do not change. Expected: the shopping activity's cost drops by most of its 24% at the standard
  profile; at the large profile, if the memo survives a tick (the tables it reads change little between two
  shoppers), `_organic` runs at most 8 times per tick instead of ~346: 486 of 698 CPU-s of day 2 become ~11 CPU-s,
  about 16 simulated days per CPU-hour instead of 5.2 (T3: 3.75).

### R2 - social/consumption.py `Engine.run` / social/recsys (owner: C, social)

* Cost (`light`, standard): `Engine.run` 30.2% on days 2-3 and 15.4% on day 4 (1,816 calls in two days, 9.4 ms each),
  of which `recsys.features.hash_normal` 8.0% (2,466 calls, 1.8 ms each).
* Why: `hash_normal(seed, idx, dims=422, salt)` loops over the 422 dimensions in steps of two and calls
  `splitmix64` twice per step: about 3,400 numpy operations per call however few rows it has.
* Proposed change: one counter grid, bitwise identical output:
  `c = base + np.arange(1, 2 * ceil(dims / 2) + 1, dtype=np.uint64); h = splitmix64(idx[:, None] ^ c[None, :])`;
  `h1, h2 = h[:, 0::2], h[:, 1::2]`; then Box-Muller on whole arrays (`out[:, 0::2]`, `out[:, 1::2]`, trimmed to
  `dims`). About 10 numpy operations per call.

### R3 - semantic/space `preprocessing_hashes()` and the text path (owner: D, semantic)

* Cost (`light`): the text path for new texts (`space._compute`) is 48.3% of a standard day with fresh content (356
  texts, 77.7 ms each, one text per LM forward pass); `preprocessing_hashes` 5.5% / 2.5% (3,304 calls in two days from
  `space.reps`, 0.9 ms each): `preprocessing()` reads every parameter module's constants, the encoders' attributes
  and the LM identity and hashes them on every call.
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
* `space/lm.py` `device()` calls `torch.cuda.is_available()` while `crucible.semantic.space` is imported
  (`SPACE_VERSION` -> `_models` -> `identity` -> `device`): every process that imports the space initialises the
  CUDA driver (a driver thread and open `/dev/nvidia*` files), CPU-only world workers included (measured
  2026-11-26: 2 threads per pool worker instead of 1, which failed tests/test_scale.py; scale/pool.py now sets
  `PYTORCH_NVML_BASED_CUDA_CHECK=1` in its workers). Proposed: read `CUDA_VISIBLE_DEVICES` first and use the
  NVML-based check, or decide the device when the LM is first used.
* `space/__init__.py` `_record_entries`: each representation is stored three times (space_items row, the
  `space.recorded` event with base64 vectors and chunks, the inference record; INT limitation 3): about 5 KB per
  copy and the largest event payloads of the world (see storage, R6).

### R4 - sandbox metering memo `DiskMemo.first` (owner: kcal, D-21 / K3)

* Cost (`wall`, standard day 1): 13.6 s of 81 s wall (17%) in `sqlite3.execute` under `sandbox.metering.bill ->
  DiskMemo.first` (1,250 calls, 10.9 ms each); its CPU share is 0.3% (`light`): the cost is waiting (wall clock,
  so simulated days per wall hour), not CPU.
* Why: every sandboxed page fetch inserts its first-measurement anchor into ONE machine-wide SQLite file
  (`~/.cache/crucible/meter/memo-v1.sqlite`, 18 MB + WAL) shared by every process of every engineer: lock waits.
* Proposed change: keep the per-world in-process memo authoritative for a transaction and write anchors to the
  machine file in one transaction per world transaction (or per N anchors), `INSERT OR IGNORE` batched with
  `executemany`; read anchors for a batch of keys with one `SELECT ... WHERE key IN (...)`. Same anchors (first
  writer still wins), far fewer lock acquisitions.

### R5 - kernel and substrate per-event costs (owner: kcal, A)

* `core.canonical.canonical_bytes`: 3.6% (`light`; 91,728 calls in two standard days, the kernel's event hashing
  and inference input digests). The kernel hashes every event body once at
  commit; payloads are already JSON-native when emitted. Proposed: a fast path that `json.dumps(..., sort_keys=True,
  separators=...)` the payload directly when it holds only JSON-native types (one isinstance walk instead of
  rebuilding the tree in `_norm`), falling back to `_norm` otherwise; identical bytes.
* `substrate.columns.batch` 9.3% (`light`, 9,544 batches in two standard days) -> `artifacts.put_array_blob` 5.8%
  -> `npz_bytes` 4.3% (10,116 blobs, 0.24 ms each in `np.savez`: zip container and per-array headers for blobs of a
  few hundred bytes).
  Proposed: a raw concatenated format (fixed header: names, dtypes, shapes; then bytes) for small batches; or
  aggregate the one-row batches (R7).
* Scheduler bookkeeping: every periodic process firing writes three events (`sched.add`, `proc`, `sched.del`):
  18,625 + 18,618 + 18,618 of 289k events in a 30-day development run. One event per firing would remove two thirds
  of them (storage, R6).

### R6 - storage growth (DEF-M-3; owners: kcal for the event/blob formats, B, C, D for their event volume)

* Measured (`storage_breakdown.py`, store after daily retention compaction, VACUUM and WAL checkpoint):
  development 7,843 B per individual-day (5 days; events 17.0 MB, blobs 15.1 MB resident of 18.0 MB written);
  standard 2,846 B per individual-day (3 days; events 35.8 MB, blobs 36.5 MB resident of 48.0 MB written). Target
  T6: 550 B (missed 5x at standard). Retention compaction keeps 84% (development) and 76% (standard) of the blob
  bytes written.
* Where the event bytes go (development, 5 days, 54,638 events): `cols.batch` 19.6%, `ext` (recorded inference and
  sandbox outcomes, 5.9 KB each) 13.7%, `artifact.stored` 13.1% (8,988 events: one per blob, next to the
  `cols.batch` event that references it), `space.recorded` 9.4% (400 events, 4 KB each: base64 vectors and chunks
  that the space_items table and the `ext` record hold too), `popweb.page` 7.4%, `bank.txn` 6.9%, scheduler
  bookkeeping 6.9% (`sched.add` + `proc` + `sched.del`, three events per process firing), `cloud.usage` 2.8%.
* Batches: `pop.deposit` is the most frequent batch kind (867 one-row batches in 5 development days, R7); the most
  rewritten columns are `pop.attention_left_min`/`attention_spent_min` (1,454 batches each) and the wake columns.
* Proposed: fold `artifact.stored` into the `cols.batch` event that references the blob; one scheduler event per
  firing; `space.recorded` and the `space.items` inference record without the vectors (keep them once, in the
  space_items table or one blob); delta-only `set` batches (the skeptic measured 36% of `set` bytes unchanged);
  deposits batched per tick. Together these address about half of the event bytes; T6 at the standard profile also
  needs smaller blobs (raw formats instead of npz zip containers, R5).

### R7 - population (owner: B)

* Class-2 decisions are not batched in the world: mean batch 1.29-1.48 (max 4) per recorded call over 30
  development days (134 calls for 198 decisions); the batched-vs-per-participant speedup at those batch sizes is
  1.41 (V.4 needs 2). The adaptive aggregation window (7,200 s) gathers few web decisions. Proposed: decide the web
  resource choice of every class-2 member that is due in the window in one round (`adaptive_window_s` aligned to
  the web activity), or route the choice through `inference_batch.gather` with the other activities of the tick.
* Class 3 is never batched: 120 recorded `pop.strategic.*` calls for 120 member decisions in 30 days (mean batch
  1.0): each member's `next` decision time is its own. `pr_decide` costs 69 ms per call (`light`, standard: 4.3% of
  the day for 24 decisions), mostly observation building and carrying the decision out through tools. Proposed: round `next` to a common decision grid per group
  (e.g. `decide_every_s` boundaries) so all due members of a group decide in one forward pass (SP-13.1).
* Class 4 is still unbound in the integrated tree (no `register_class_runtime(world, 4, ...)` caller on
  55433e7; D binds it on c5c/w3b-sem in `semantic/generalists/module.py`): V.1 and SP-13.1 fail until it acts in
  the merged tree. Its runtime already decides the tick's due participants together (one `generate`/`choose` per
  capability and step), but `_next_times` draws each participant's gap independently, so with a few class-4 rows a
  call rarely holds more than one participant: put class-4 activations on the population tick's decision grid.
* `money_deposit`: one one-row `pop.deposit` batch (event + blob) per depositing individual (522 in 3 development
  days; the skeptic counted 4,424 per hour at 10M). Proposed: deposit the due individuals of a tick in one batch.
* `population/web.py:786` `[j for j in range(n) if j not in set(own)]` rebuilds `set(own)` per element (quadratic;
  DEF-M-6). Hoist the set.

### R8 - civ world composition (owner: J)

* The population-only world (`population_world`, 4.0 services and generated sites) runs at 145-159 simulated days
  per CPU-hour at development, about half the civ world's rate (277-330): the sandboxed site handlers
  (`services/net.fetch` -> `cloud.execute_handler` -> sandbox worker, 2,676 fetches in two standard days) and their
  metering dominate it. Not a civ request; reported so that the two rows are not read as one.
