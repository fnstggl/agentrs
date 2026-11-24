# Performance and scale (measured)

All numbers below were measured on this machine (48 vCPUs, RTX A6000 unused by these benchmarks, Python 3.12,
Linux 6.8 container with `pids.max = 2816`) by `python -m crucible.bench.run` (raw output:
`docs/benchmarks/bench.json`), `tests/test_storage.py` and the demo scripts. Other jobs (agents, tests) were
running concurrently, so treat them as conservative. Nothing here is an extrapolation.

## Kernel

| measure | value |
|---|---|
| fully transactional commands (`bank.transfer`: validation, authorisation, journaled transaction, 3–4 events, hash chain) | **7,732 /s** |
| read commands (recorded `bank.accounts`) | 19,709 /s |
| event replay from genesis (exact digest check) | 41,814 events/s |
| storage append, batches of 500 (memory / SQLite / PostgreSQL) | 43k / 81k / 39k events/s |
| storage append, one event per transaction | 41k / 23k / 1.4k events/s |
| storage read | 48k / 67k / 50k events/s |

(PostgreSQL ran with fsync off in a throwaway local cluster.)

## World simulation (family `bazaar`, 4 needs, 8 incumbents, real code execution for sampled requests)

| consumers | wall s per simulated day | events per sim-day | snapshot | fork | digest |
|---|---|---|---|---|---|
| 1,000 | 0.82 | 2,198 | 7 ms, 0.09 MB | 17 ms | 34 ms |
| 6,000 | 0.97 | 2,351 | 15 ms, 0.27 MB | 23 ms | 30 ms |
| 25,000 | 0.99 | 2,626 | 41 ms, 0.88 MB | 55 ms | 25 ms |
| 100,000 | 2.19 | 3,417 | 162 ms, 3.41 MB | 212 ms | 47 ms |

Every fork was digest-equal to its parent. Cost is dominated by executing and verifying sampled real requests
(constant per offering), not by population size, until ~100k consumers.

* **NPC workers:** 50 / 500 / 2,000 simulated workers → 0.32 / 0.36 / 0.34 s per simulated day (they are
  event-driven and cost nothing while idle).
* **Agents:** 3 scripted agents acting every simulated hour for 3 days: **19 ms per agent step including the world
  advancing** (216 steps, 4.1 s).
* **Replicas (independent worlds in processes, 3,000 consumers each, 1 sim-day each, including process start):**
  1 → 0.63, 2 → 1.22, 4 → 2.53, 8 → 5.81, 16 → 6.85 aggregate simulated days per wall second.
* **Lifetime demo** (bazaar, 6,000 consumers, 5–6 agents, 30 simulated days, SQLite store with snapshots every
  5 days): ~3–4.5 s per simulated day while agent-built services are live; 72,078 events; reload from store
  0.5 s; full replay from genesis 4.7 s.

## Sandbox

| measure | value |
|---|---|
| one-shot program (start sandboxed interpreter, apply Landlock+seccomp, run, tear down) | 91 ms |
| call on a persistent service worker | 0.106 ms |

Service deployments use persistent workers (an LRU pool, default 24 per world), so request-level overhead is
~0.1 ms plus the program's own work.

## Optimisations made after profiling

1. Demand samples are scored inside the recorded external call, so the event log stores scores, not full outputs
   (3.6× faster simulated days, much smaller logs).
2. Vectorised request generators and an O(n·C) numpy knapsack optimum in the verifier.
3. Search keeps a runtime inverted index validated against state (never trusted blindly).
4. Fixed thread pools: every worker process runs with BLAS/OpenMP threads = 1; Chromium sessions are capped; the
   sandbox caps processes per uid (`RLIMIT_NPROC`) and blocks `fork`, so agent code cannot exhaust the container's
   pid budget — the failure mode that broke earlier AGENTRS experiments.

## Bottlenecks and limits

* Single-world simulation is single-threaded (the kernel serialises commands by design); throughput scales with
  replicas, not threads. Snapshot/fork are copy-on-snapshot (pickle) — O(state size), ~200 ms at 100k consumers.
* The in-memory event log keeps up to 2M events by default (`log_retain`), older events are read from the store.
* Measured CPU accounting adds timing noise; the deterministic cost model counts executed Python lines (slower
  for Python-heavy code: sys.monitoring LINE events).
* LLM-backed agents cost far more than the world: see RL_INTEGRATION.md for measured tokens and latency.
* Browser mode: one Chromium ≈ 80 threads; at most ~2 concurrent browsers are safe in this container.

## Open-world mechanisms (measured; `python -m crucible.bench.run --only open`)

From `docs/benchmarks/bench.json` (`open` section), this machine:

| measurement | value |
|---|---|
| capability publish | 0.19 ms |
| 20 triggers firing every 10 min for a simulated day (2,880 fires) | 0.60 s |
| 5-step flow runs | ~9,800 / s |
| deal proposals | ~3,900 / s |
| `open_industry`: simulated day / events per day / snapshot / fork / replay | 0.033 s / 1,254 / 671 kB / 41 ms / 39.7k events/s |
| `open_mixed` (2,000 API consumers, 20,000 households): simulated day / snapshot / fork / replay | 1.28 s / 801 kB / 58 ms / 21.3k events/s |
| Tier-2 minds: 56 / 506 / 2,006 minds - simulated day | 0.31 / 0.39 / 0.44 s |

Minds are cheap (about 0.07 ms per mind per day: thousands of active simulated actors are practical); the cost of a
digital-economy day is dominated by executing sampled requests of incumbent services under the deterministic
line-counting cost model (~11 ms per request for the route-finding incumbents), which is why open families sample 2
requests per offering per hour. Long run: 90 simulated days of `open_mixed` (295k events) in 136 s wall, replayed
from SQLite with an identical digest.

## Synthetic internet (measured; `python crucible/examples/bench_internet.py`, docs/benchmarks/internet.json)

Same container, under load from other experiments (LLM inference on the GPU and three simulations).

| Measure | Result |
|---|---|
| `net.fetch` GET of a generated site via the API | 1.59 ms per request, 631 per wall second. Includes advancing the world 20 simulated s per request, which keeps the agent under its 240 actions/hour limit |
| Same page in real Chromium (hermetic resolver, through the front end) | first load 292 ms, then 125 ms mean: about 79x the API cost for the same recorded command |
| 10 / 50 / 200 scripted agents on independent clocks (AsyncRunner, 6 simulated hours) | 1,325 / 2,338 / 2,431 actions per wall second |

* The concurrent agents above are **reasoning-free readers**. They measure the kernel and scheduler, not LLM
  scale.
* LLM decisions cost what the model costs. In the final challenge (A6000, bf16, prompts of about 4.6k tokens),
  Qwen3-4B took about 3.5 s of generation per decision (`usage` in `docs/experiments/final_challenge_*.json`).
* A world with hundreds of concurrently *reasoning* LLM agents was not run.
* Statistical consumers (thousands per world) are reported separately from agents and minds.

Validation and calibration costs (Part 17/18; open_tight_compute, 3-day warm-up):

| Task | Cost |
|---|---|
| Visibility task | about 43 s (idle + random + competent forks, 7 days) |
| Loss-relief task | about 139 s (21 days) |
| Adoption task | about 163 s (30 days) |

## Crucible 4.0 (`docs/benchmarks/openworld.json`, `examples/bench_openworld.py --quick`)

The machine has 48 logical CPUs but an **11.52-core cgroup quota** shared with other jobs, so earlier references to 48 vCPUs
overstate what a process can use and wall-clock numbers are pessimistic. Each record carries its load and quota.

| measure | value |
|---|---|
| ledger transfers | 4,442 per wall-second (7,196 per CPU-second) |
| replay from genesis | 26,345 events per wall-second |
| parallel small worlds (`tiny`, 2 days each) | 1.15 simulated days/s for 1 world, 1.41 for 2, 4.68 for 4, 10.13 for 8 |
| real Chromium | 7.5 navigations/s |
| resident memory of a world | 4 MB (1,000 people), 40 MB (100,000 people) |
| 365-day `ow_market` run | about 1.2M events, 310-360 MB of log, 1.4-1.6 GB peak memory, about 70 minutes under load |

Quick mode covers 15 of the 73 world families. The full table and the 30-day episode costs were not run.

