# Audit M_performance: Part 13, Gate V, SP-13.7 and SP-13.8

Auditor: independent audit of `feat/crucible-5-complete` at `a881d18` (worktree `.worktrees/c5-complete`), 2026-11-26.
Scope: MASTER_SPEC Part 13 (13.1-13.8), Part 3.3 (profiles), Gate V and its mandatory note, SP-13.7 and SP-13.8; code in
`src/crucible/scale`, `src/crucible/bench`, `docs/benchmarks`, `docs/v5/evidence/baseline_perf`, and the performance
mechanisms of other workstreams that Part 13 depends on (substrate columns, population activation, shared-weight
networks, semantic service, embedding cache, social large-scale path).
Binding amendments: Clarification 1 (CAM independence), amendment (a) (observation-only is the universal default),
amendment (b) (prior completion claims do not count without reproducible evidence).

Paths below are relative to `crucible/`. Probe scripts, logs and JSON outputs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/M_performance/`. Every heavy run went through
`crucible.scale.slots` with `--cpu 1` (one run used `--gpu-mb 100`).

## 1. Bottom line

Workstream M never ran. `docs/v5/agent_reports/INDEX.json` has no M agent. There are no `tests/v5/test_gate_V.py`,
`test_gate_SP_13_7.py` or `test_gate_SP_13_8.py`, no v5 benchmark harness, no recorded throughput targets, no
approximate-simulation mode and no v5 performance report. All 12 M requirements are unmet. What Part 13 needs is
spread across other workstreams' code, and some of it works when measured:

* The columnar substrate (ws-A) reproduces its published numbers.
* Shared-weight class-2 policies run batched forward passes (6-8x faster than one call per participant).
* Event-driven activation runs inside the integrated population world.
* The embedding cache works when it is called.
* B's population runs the development, standard and large profiles, and creates a 10M-row internet profile.

The parts Gate V cares most about are missing or wrong:

1. **The resource scheduler does not enforce anything.** `crucible.scale.slots` is a cooperative admission ledger
   that tracks 3 of the 7 SP-13.7 dimensions. None of them is enforced at OS or library level. A job that reserved
   100 MB of GPU memory allocated 1,024 MB, and 13 of the 219 logged jobs used more than 110% of the CPU they reserved.
2. **Event storage grows too fast for long or large runs.** The large profile writes 184 MB of event log and batch
   blobs per simulated day, about 1.8 KB per individual per day. Blobs are stored uncompressed (`ZIP_STORED`). At that
   rate a 30-day 10M run needs about 550 GB, and the machine has 21 GB of free disk.
3. **Nothing approximates large populations, so nothing can be compared against a high-fidelity run.** The
   "estimated population mode" claimed in ARCHITECTURE.md section 4 and DECISION_LOG D-5 does not exist in code.
4. **The 4.0 rollout environment in the owned path breaks amendment (a).** `crucible.scale.rollout.WorldEnv` defaults
   to `legacy_shaped_reward`, which returns the account-balance change as the scalar reward.
5. **Population costs are dominated by fixed overhead, not by decisions.** Making everyone decide on every tick
   (28.6x more activations) costs only 2.18x more wall time. Sandboxed-program process spawns cost more CPU than the
   world process itself.

## 2. Hardware of this machine (measured 2026-11-26)

| item | value |
|---|---|
| CPU | 2 x Intel Xeon Gold 6226 (12 cores / 24 threads each), 48 logical CPUs, AVX-512, 2 NUMA nodes, 38.5 MiB L3 |
| cgroup CPU quota | 1,152,000 / 100,000 us = **11.52 CPUs**, shared by at least 15 auditors and other tenants |
| CPU contention | 666,717 of 3,522,301 scheduler periods throttled (18.9%) since the container started; load average 9-13 during this audit |
| slot budget (`slots.Budget.from_env`) | 9 CPUs (floor(11.52) - 2), 24,000 MB GPU, 8 jobs |
| RAM | host 188 GiB; container cgroup limit 97,142,177,792 B (**90.5 GiB**); about 13 GB RSS plus 63 GB page cache in use during the audit |
| GPU | 1 x NVIDIA RTX A6000, 46,068 MiB, driver 560.35.03; idle at the start of the audit |
| disk | overlay 300 GB, **21.1 GB free (94% used)** |
| pids | cgroup pids.max 2,816 (pool.py limits workers for this reason) |
| software | Python 3.12, torch (CPU and CUDA), **no Numba** (`ModuleNotFoundError`) |

Raw: `.../M_performance/hardware.txt`.

## 3. What I ran and what it showed

All runs used one CPU slot on a shared machine. Wall-clock numbers are pessimistic. CPU-seconds are listed where
they matter.

### 3.1 Population profile benchmark (my probe `profile_bench.py`, integrated `population_world`, seed 0)

`population_world` (`src/crucible/population/worlds.py:16`) composes the 4.0 internet services, the substrate,
generated sites, `pop`, `popweb`, `pop_adaptive` and `pop_strategic`. Each world starts at 00:00, so short windows
include the night.

| profile | rows (classes 1/2/3/4) | sim h | run wall s | sim days / wall h | activations / wall s | events / sim day | event JSON MB / day | blob MB / day | snapshot MB | peak RSS MB | policy updates |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development | 1,042 (1000/32/8/2) | 24 | 36.5 | **98.5** | 72 | 9,587 | 5.2 | 3.4 | 0.68 | 717 | 0 |
| standard | 10,296 (10000/256/32/8) | 24 | 55.5 | **64.9** | 466 | 22,911 | 11.2 | 17.1 | 3.8 | 812 | 2 |
| large | 101,120 (100000/1000/100/20) | 24 | 161.9 | **22.2** | 1,585 | 96,666 | 37.6 | **147.0** | 33.0 | 1,497 | 3 |
| internet, n=1M | 1,001,120 | 1 (08:00-09:00) | 17.0 | (8.8 for that daytime hour) | 16,420 | - | - | - | - | 2,342 | - |
| internet, n=10M | 10,001,120 | 0.5 (00:00-00:30) | 23.7 | - | 904 (night) | - | - | - | - | **8,355** | - |

The 10M internet profile took 36.5 s to create, held 4.9 GB RSS after creation and took 8.2 s for a state digest.

Observations:

* **Class 4 (semantic generalists) never act.** Their rows exist, but no module in `population_world` drives them:
  `next_active_at` is `NEVER` for classes above 2 (`src/crucible/population/core.py:330`), and there is no
  `pop_semantic` module. The ws-D2 semantic participants run in a 4.0 `open_digital` world instead
  (`docs/v5/components/D2.md`, Limitations). No profile has been benchmarked with all four classes active.
* **Adaptive learning barely happens on day 1.** There were 0 adaptive policy updates in the development profile in a
  day; the defaults are `min_examples=64` and `learn_every_s=21600` (`src/crucible/population/adaptive.py:214`).
* **Most of the cost is outside the population.** In a 200-person world over 24 h, the world process used 10.7 CPU-s
  and its child processes 15.6 CPU-s. The children are 81 sandbox Python processes and 83 `ldconfig -p` calls
  (`subproc_probe.log`). `CodeService.run_program` starts a fresh sandbox process per call (`close_after=True`,
  `src/crucible/services/code.py:184-197`). `sandbox.runner.capabilities()` calls `ctypes.util.find_library`, which
  forks `ldconfig`, every time it runs (`src/crucible/sandbox/runner.py:67-70`). Profiling the first 30 minutes of a 2M
  internet world showed 11.0 s of 17.7 s in `epoll.poll` waiting on site-deployment sandboxes and 97 `fork_exec`
  calls (`tick_profile_2M.log`). That is the world-generation cost, roughly 15-20 s per world independent of N.
* **Storage growth is the binding constraint at scale.** The large profile writes about 1.83 KB per individual per
  simulated day. `pop.session` rewrites 11 columns, including the 16-float `habit` row, for every activated
  individual (`core.py:447`). Batch blobs are uncompressed (`ZIP_STORED`, `src/crucible/substrate/artifacts.py:212`),
  although ARCHITECTURE.md section 4 says "compressed array blob". zlib level 6 compresses them only 1.53x
  (`blob_compress_probe.json`), so the real fix is delta/sparse encoding, not deflate. Without a store, every blob of
  every world stays in the process-global `artifacts._POOL` dict for the life of the process
  (`artifacts.py:63,165`).

### 3.2 Event-driven activation vs activating everyone every tick (my probe `event_driven_probe.py`)

Both arms are forks of the same standard-profile snapshot at 09:00 and run 3 simulated hours. In the dense arm,
`pop.tick` makes every class 1-2 participant decide at every tick.

| arm | wall s | activations | sessions | adaptive forward calls / decisions / max batch |
|---|---:|---:|---:|---|
| event-driven (as built) | 8.47 | 4,308 | 3,753 | 2 / 74 / 45 |
| dense every tick | 18.50 | 123,072 | 100,837 | 12 / 1,473 / 142 |

The wall-time speedup is **2.18x** for **28.6x** fewer activations. The extra cost is about 0.1 ms per added session,
so most of the world's cost does not depend on how many participants decide. The due scan is a vectorised O(N)
comparison per tick (`core.py:341-356`): 26.5 ms on 10M-row columns, about 2.5 s per simulated day at a 900-s tick.
That is acceptable. Event aggregation exists only for class 2 (`core.py:351`). Over 3 h, 256 adaptive participants
made 74 decisions in 2 batched calls, so in-world batches are small.

### 3.3 Batched numerical inference, and alternatives (my probe `batch_inference_probe.py`, one thread)

The network is the class-2 `AdaptiveNet` (`src/crucible/population/nets.py`): 187,585 parameters, hidden 256, K=13
candidates.

| B | torch batched s | torch one call per participant s | batched speedup | NumPy re-impl vs torch | int8 dynamic vs fp32 | int8 argmax agreement | decisions / s batched |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.00132 | 0.0103 | 7.8x | 0.34x (slower) | 1.19x | 0.906 | 24,241 |
| 256 | 0.0126 | 0.0803 | 6.4x | 0.38x | 1.23x | 0.941 | 20,353 |
| 1000 | 0.0547 | 0.3245 | 5.9x | 0.45x | 1.40x | 0.932 | 18,281 |

Shared weights are correct: there is one `AdaptiveNet` per policy group and each member's memory is in the
`pop.memory` column (`adaptive.py:1-30`, `adaptive.py:307-338`). Batching is real at the micro level. But nothing in
the tree benchmarks the alternatives Part 13.3 lists (NumPy, compiled kernels, Numba or equivalent, quantized, GPU), and
Numba is not installed. My NumPy re-implementation matches torch to 3e-7 but is 2-3x slower. int8 dynamic quantization
is 1.2-1.4x faster but changes 6-9% of argmax choices, which no one has measured against behaviour. The strategic
networks have 2,162,715 (MLP, h1024) and 2,362,203 (GRU, h576) parameters.

### 3.4 Content-representation cache (my probe `cache_probe.py`, CPU, fresh cache directory)

The workload was 200 distinct text artifacts, each encoded for 20 viewers in 3 worlds (12,000 requests) with
all-MiniLM-L6-v2:

* Hit rate 0.983. There were 200 misses and 200 vectors computed, so each artifact was encoded exactly once.
* A second cache object on the same directory served 50 of 50 requests from disk.
* The key `ref|encoder@version.c1|preprocessing_hash` includes the encoder version and the preprocessing hash.

The mechanism (`src/crucible/semantic/cache.py`, `semantic/encoders/__init__.py:126-156`) is correct. However:

* **The default integrated path does not use it.** Social `aud` defaults to `c2-random-indexing`
  (`src/crucible/social/audience.py:112`), and the population uses its own sparse random-indexing vectors
  (`src/crucible/population/text.py`).
* These are two different lexical encoders: dense Gaussian in `social/textrep.py` and sparse ternary in
  `population/text.py`.
* V.6's 0.9 threshold is met trivially by any reuse-heavy synthetic workload.

### 3.5 Resource scheduler `crucible.scale.slots` (`src/crucible/scale/slots.py`, 202 lines)

* **Dimensions.** It tracks CPU threads, GPU MB and job count (`slots.py:50-60`, `fits` at `102-109`). It does not track
  RAM, GPU time, disk, inference concurrency or active worlds. That is 3 of the 7 SP-13.7 dimensions.
* **No enforcement** (`slots_enforcement_probe.log`). Inside a `--cpu 1 --gpu-mb 100` slot:
  * `sched_getaffinity` gives all 48 CPUs and every rlimit (AS, DATA, RSS, FSIZE, NPROC, CPU) is unlimited.
  * The job allocated and touched 1,024 MB of RAM.
  * `torch.set_num_threads(8)` took effect. The only CPU control is the `OMP_NUM_THREADS`, `OPENBLAS`, `MKL` and
    `NUMEXPR` environment variables (`slots.py:151-152`); `NUMBA_NUM_THREADS` is not set.
  * **The job allocated 1,024 MB on the GPU after reserving 100 MB.** The GPU is hidden only when `gpu_mb == 0`
    (`slots.py:153-154`); nothing caps memory (no `torch.cuda.set_per_process_memory_fraction`, no allocator
    configuration, no nvidia-smi watchdog).
* **Violations already in the usage log.** My analysis of `/tmp/crucible-slots/usage.jsonl` (241 records, 219 with
  wall time of at least 5 s) found 13 jobs using more than 110% of their reserved CPU, up to 1.8x
  (`usage_analysis.log`). The log records no VRAM, so GPU overuse cannot be detected after the fact.
* **No FIFO or fairness.** On a private ledger with a 2-CPU budget, A held 1 CPU, B asked for 2 CPUs first, and C asked
  for 1 CPU 0.3 s later. C was admitted at 0.3 s while B waited (`slots_fairness_probe.log`). Small jobs can starve
  large ones indefinitely. A request larger than the whole budget runs "alone" (`slots.py:103-108`), which is itself an
  unflagged budget overrun.
* **Zombies keep slots.** `_alive` treats a zombie pid as alive (`slots.py:69-76`; verified: a reaped child reads False,
  an unreaped one True). A library caller that does not reap its children keeps slots forever.
* **Inaccurate accounting.** `cpu_s` and `peak_rss_mb` come from `RUSAGE_CHILDREN` of the wrapper (`slots.py:176-179`):
  cumulative across calls when used as a library, and the maximum of one descendant rather than the sum of a
  multi-process job.
* **The timeout signal reaches only the direct child**, not its process group (`slots.py:166-172`).
* **The simulation never consults it.** It is an external CLI wrapper. The only in-tree importers are docstrings and
  `acceptance/v5/evidence.py:69`, which reads `cgroup_cpus`. SP-13.7 asks that "the simulation respect explicit limits".
* **What `WorldPool` already has.** The 4.0 `WorldPool` caps workers (`MAX_WORKERS=8`), pins BLAS threads, checks the
  pid budget and supports an optional per-worker `RLIMIT_AS` (`src/crucible/scale/pool.py:64,107-130,360-363`). That is
  the closest thing to in-simulation enforcement, but nothing in v5 connects it to a global budget.

Tests: `tests/test_v5_slots.py` (5 tests) passes, but only checks the bookkeeping: fits arithmetic, reclaiming dead
holders, waiting, environment variables and logging.

### 3.6 Crucible 4.0 comparison

* **Same 4.0 workloads on the v5 tree.** I ran `examples/bench_openworld.py --quick --only kernel,families` with 1
  worker and output in the scratchpad (`v5tree_openworld_quick.json` and `.log`). Against the baseline in
  `docs/v5/evidence/baseline_perf/bench_openworld_quick.log`:

  | workload | baseline | v5 tree |
  |---|---:|---:|
  | bank.transfer (tx/s) | 7,978 | 8,147 |
  | mixed tool calls (tx/s) | 8,903 | 8,603 |
  | replay (ev/s) | 47,931 | 45,876 |
  | tiny (idle d/s) | 2.82 | 3.08 |
  | bazaar (idle d/s) | 1.10 | 1.15 |
  | open_digital (idle d/s) | 2.85 | 2.69 |
  | open_industry (idle d/s) | 29.4 | 29.9 |
  | mech_scm_additive (idle d/s) | 174 | 171 |

  4.0 workloads have not regressed. Idle peak RSS grew from about 60 to 64-74 MB because more modules load.
* **Population at similar size.** 4.0 `bazaar` with 100,030 aggregate consumers (75 principals) ran 0.508 sim days per
  wall second (1,830 per hour) with 42 MB world RSS (baseline JSON, memory section). The v5 large profile (101,120
  individuals with persistent identity, attention, money, habits and learning) runs 22.2 sim days per wall hour with
  1.5 GB RSS: **about 82x slower and 35x more memory**. The workloads are not of equivalent fidelity, and SP-13.8's
  "equivalent workloads" comparison has never been designed.

### 3.7 Substrate reproduction (ws-A claim)

`python -m crucible.substrate.bench --rows 100000 --table-rows 1000000 --store sqlite --events 5000`
(`substrate_bench_100k_1M.json`):

| measure | measured | claimed in `docs/v5/components/A.md:128-140` |
|---|---:|---:|
| recorded transaction | 0.0202 s | 0.018 s |
| replay | 0.0176 s, exact | 0.017 s |
| regeneration | exact | - |
| fork | exact | - |
| kernel cost per event | 10.5 us | 10.1 us |
| no-op subscriber | +0.88 us | +0.9 us |

Reproduced.

### 3.8 10M-identity social scale

* **Social `crowd` path.** The positive-control run in `tests/v5/test_gate_I.py` uses `social/cascade.py` and
  `social/scale.py`, a separate, simplified population: interests, satisfaction, `next_active_at`, `share_prop` and
  attention only. It refuses to coexist with B's `pop` (`social/scale.py:158`).
* **Unofficial 10M evidence exists, but not from this tree.** Records under `/tmp/crucible-v5-unofficial-evidence/I/`
  dated 2026-11-25, from commit `b30cac86`, an ancestor of `a881d18` (`consumption.py` has changed since), show
  10,000,000 identities, 10.57M qualifying views, 0 recount mismatches and a run wall time of 153 s.
* **I did not rerun the 10M social path.** The C_social auditor's 10M + 2.5M Gate I run was holding a slot during this
  audit, and a second ~15 GB run would have been an unnecessary load.
* **B's full population at 10M.** It creates in 36.5 s and peaks at 8.4 GB RSS over the first 30 simulated minutes
  (night). From the 1M daytime rate of 17 s per simulated hour, I extrapolate about 3 minutes per daytime hour and
  roughly one simulated day per wall hour at 10M. That figure is not measured.
* **The integrated 10M combination (B's identities with C's platforms, D's representations and commerce) has never
  been run**, and the two population modules are mutually exclusive.

### 3.9 Observation-only amendment in the owned path

`src/crucible/scale/rollout.py:128` sets `reward_mode: str = "legacy_shaped_reward"` as the default. My probe
(`rollout_reward_default_probe.log`) created `WorldEnv({'family': 'tiny', 'seed': 0})`, added 12,345 cents to the
learner exogenously through a system call and stepped with no action. The scalar reward was **12345.0**. This breaks
amendment (a): the observation-only default, Gym reward 0.0, and no shaping by default. `scale/bench.py`'s `rollout`
section builds `RolloutFarm` workers with this default. `experience/parallel.py` (ws-L) defaults to
`OBSERVATION_ONLY`, so the two learner paths disagree.

### 3.10 Hot-path defects found while profiling (13.3 "avoid excessive Python object creation")

* **Quadratic set construction.** `src/crucible/population/web.py:775`:
  `groups = {None: [j for j in range(n) if j not in set(own)]}` rebuilds `set(own)` for every `j`, which is
  O(n * |own|). Measured cost: 0.39 s for a 5,000/5,000 cohort against 0.0003 s with the set hoisted
  (`forum_quadratic_probe.log`). It grows as follow edges accumulate, and line 773 builds a Python string per cohort
  member.
* **Per-individual Python loops.** `src/crucible/population/core.py:165` loops in Python for bank balances.
  `money_deposit` writes a one-row `cols.batch` blob and event per individual (`core.py:573-590`): 884 per simulated
  day at the large profile. About 17k principals are materialised per day at the large profile (`principal.add` plus
  `id.person`), each a Python dict in the state tables.
* **A per-member dict of numpy rows** in `src/crucible/population/adaptive.py:338` (`pend["rows"]`).
* **A sandbox process per program call** plus an `ldconfig` fork per capabilities check (section 3.1).

## 4. Subsystem verdicts

| subsystem | paths | verdict | effort |
|---|---|---|---|
| Resource scheduler | `scale/slots.py` | **REDESIGN** | L |
| v5 performance benchmark and report | none | **IMPLEMENT** | L |
| 4.0 benchmark suite (baseline instrument) | `scale/bench.py`, `bench/run.py`, `examples/bench_openworld.py`, `docs/benchmarks/*.json`, `docs/v5/evidence/baseline_perf/` | **RETAIN** (as the 4.0 baseline instrument) | S |
| WorldPool / RolloutFarm | `scale/pool.py`, `scale/rollout.py` | **REPAIR** | S |
| Columnar state and batch events (storage efficiency) | `substrate/columns.py`, `substrate/artifacts.py` | **EXTEND** | M |
| Event-driven activation and aggregation | `population/core.py:341-450`, `social/scale.py:265` | **EXTEND** | M |
| Shared weights and batched numerical inference | `population/adaptive.py`, `population/nets.py`, `population/strategic.py`, `substrate/inference.py` | **EXTEND** | M |
| Compact semantic inference service | `semantic/service.py`, `semantic/participants.py` | **EXTEND** | M |
| Multimodal caching | `semantic/cache.py`, `semantic/encoders/` | **EXTEND** | S-M |
| Large-population techniques | `social/idset.py`, `social/scale.py` (`ContactGraph`), `substrate/ids.py`, `social/recsys/`, `population/profiles.py` | **EXTEND** | L |
| Approximation vs high-fidelity error measurement | none | **IMPLEMENT** | L |
| 10M-identity scale path | `social/cascade.py`, `social/scale.py`, `population/profiles.py` (internet) | **EXTEND** | L |
| Hot-path efficiency of the integrated population world | `population/web.py`, `population/core.py`, `services/code.py`, `sandbox/runner.py` | **REPAIR** | M |

* **Resource scheduler.** The admission ledger with `flock` is sound and can stay as the admission layer. But it
  tracks 3 of 7 dimensions, enforces none, has no FIFO, is fooled by zombies, records no VRAM, and the simulation
  never consults it. SP-13.7 needs a resource governor that the kernel, `WorldPool`, the inference service and the
  encoders consult and that is enforced. Needed:
  * Enforcement:
    * `sched_setaffinity` to the reserved CPU count;
    * `RLIMIT_AS` or a watchdog on RSS;
    * `torch.cuda.set_per_process_memory_fraction`, plus an nvidia-smi per-pid VRAM watchdog that kills violators;
    * GPU-seconds accounting from nvidia-smi samples;
    * a disk quota on the store and cache directories, by watchdog plus refusal at admission.
  * In-process limits: an inference-concurrency semaphore in `SemanticService` and the encoders, and an active-world
    counter in `World.create`/`WorldPool`.
  * A FIFO queue with reservation.
  * Zombie-safe liveness, by reading the `/proc/<pid>/stat` state.
  * Per-job VRAM, disk and GPU-time fields in `usage.jsonl`.
  * An adversarial test in which each dimension is exceeded and stopped.
* **v5 benchmark and report.** Part 13.8, Gate V and SP-13.8 have no harness and no test files. Write
  `crucible/scale/bench5.py` plus `tests/v5/test_gate_V.py`, `test_gate_SP_13_7.py` and `test_gate_SP_13_8.py`:
  * all four profiles with every class active, at 30 or more simulated days and several seeds;
  * the 12 quantities, including inference token counts, storage growth, snapshot size and world-generation cost;
  * peak CPU, RAM and VRAM by sampling;
  * targets recorded before the runs;
  * a designed equivalent-workload 4.0 comparison.
* **4.0 benchmark suite.** I reproduced its kernel and family sections on this tree, within noise of the recorded
  baseline. It is adequate as the instrument for the 4.0 comparison. It does not cover v5 populations, and its
  `rollout` section uses the legacy shaped-reward default.
* **WorldPool / RolloutFarm.** Process isolation, worker caps, BLAS pinning and the optional `RLIMIT_AS` are sound
  (4.0 tests, not rerun here because they need more than one CPU). Needed:
  * change `WorldEnv`'s default to `observation_only` (`rollout.py:128`);
  * make bench's `rollout` section pass the mode explicitly;
  * report the active-world count and memory to the resource governor.
* **Columnar state and batch events.** The mechanism is correct and reproduced (replay, regeneration and fork exact;
  0.02 s per 100k-row cohort). Storage growth is not: blobs are uncompressed, whole rows are rewritten, and the
  process-global `_POOL` is unbounded without a store. That gives 184 MB per simulated day at 100k, which is
  incompatible with long or internet-scale runs on 21 GB of disk. Needed:
  * delta or sparse encoding (only changed columns, quantized, compressed with a deterministic codec);
  * log compaction with checkpoint snapshots;
  * a bounded `_POOL`;
  * a storage-growth target in Gate V.
* **Event-driven activation and aggregation.** It works and is integrated: due-only activation, `wake`, and an
  aggregation window for class 2. Measured 2.18x wall speedup and 28.6x fewer activations. But costs that do not
  depend on decisions dominate, only class 2 is aggregated, and two population modules each have their own tick
  (`pop.tick` and `crowd.tick`). Needed:
  * make the cost proportional to decisions (section 3.10 fixes);
  * aggregation for strategic and semantic decisions;
  * unify crowd and pop activation;
  * measure in Gate V at the large and internet profiles.
* **Shared weights and batched numerical inference.** Shared weights are correct, and batched forward passes are
  5.9-7.8x faster than one call per participant (reproduced). Decisions are recorded through
  `substrate.inference.recorded`. In the world, batches are small (max 45). The alternatives (NumPy, compiled or
  Numba, quantized, GPU) are not benchmarked in the tree. My NumPy re-implementation is 0.4x torch; int8 is 1.2-1.4x
  faster with 6-9% different choices. Needed:
  * a backend benchmark matrix with a measured choice;
  * behavioural-equivalence checks for quantized inference;
  * larger in-world batches through aggregation;
  * a GPU path when B is large.
* **Compact semantic inference service.** All nine 13.4 features exist in code: batching, quantization, short or
  bounded context, BM25 retrieval, prefix KV cache, bounded generation, versioning by weights hash, and CPU or GPU
  selection. Measured by ws-D2, not reproduced by me: 252 generated tokens/s on the A6000, a prefix-cache speedup of
  1.00-1.07, 6.5 tokens/s on CPU, and no paged KV or CUDA graphs. It is not wired into the population profile, so
  class 4 is inert. Needed:
  * a decode loop with static KV and CUDA graphs, or an equivalent engine, for real throughput;
  * a `pop_semantic` module so profiles include class 4;
  * measured token counts and VRAM per profile.
* **Multimodal caching.** The mechanism is correct (my probe: 0.983 hit rate, one compute per artifact, disk reuse
  across caches, versioned key). The integrated social and population path does not use it: both default to lexical
  random indexing, and there are two different implementations. Needed:
  * route the default social and population representations through `semantic.encode` and the cache, or a declared
    numeric fast path;
  * measure the hit rate on an integrated multimodal workload across worlds.
* **Large-population techniques.** Present:
  * exact bitset or sparse sets (`idset.py`, 1.25 MB per 10M);
  * an on-demand hashed contact graph (`ContactGraph`, `social/scale.py:74`);
  * lazy principal materialisation;
  * columnar arrays;
  * candidate pools in recsys;
  * a derived-init internet profile (10M rows created in 36.5 s, 8.4 GB peak).

  Absent: aggregate transitions, hierarchical or adaptive-resolution simulation, and batched cohort behaviour for
  internet-scale routine activity. Principals are still materialised as Python dicts (about 17k per day per 100k
  people). Needed:
  * an adaptive-resolution design (an active core simulated individually, a validated statistical periphery);
  * columnar principal and bank records for population members;
  * a measured memory model per identity.
* **Approximation vs high-fidelity error measurement.** No approximate mode exists, so nothing can be compared. The
  "estimated population mode" in ARCHITECTURE.md section 4 and DECISION_LOG D-5 is not in the code (`grep estimated`
  finds nothing in social, population or substrate). Needed:
  * an aggregate or cohort transition mode, or the crowd model treated explicitly as the low-fidelity tier;
  * paired runs against B's full individual simulation at 100k-1M on the same seeds;
  * errors in adoption trajectories, audience distributions, revenue, network diffusion, retention, intervention
    effects and rare-event frequencies, with confidence intervals and documented limits.
* **10M-identity scale path.** The capability exists in two disjoint forms: the simplified crowd path (an unofficial
  10M run on an ancestor commit) and B's internet profile (10M creation only, measured by me). Neither is the
  integrated population with social, commerce and representations. Needed:
  * one integrated 10M world (B's identities with C's platforms) running at least one simulated day under recorded
    RSS, wall and disk targets;
  * per-identity decisions recounted from batches;
  * an official run on this tree.
* **Hot-path efficiency of the integrated population world.** The sound design is undermined by per-individual
  Python paths (section 3.10). Needed:
  * hoist `set(own)` and vectorise feed grouping;
  * batch `money_deposit` across a cohort;
  * vectorise `spendable`;
  * a persistent sandbox worker per hosted site, or cache the result of `capabilities()`;
  * profile the large profile and add regression benchmarks.

## 5. Requirement verdicts (workstream M)

| id | verdict | reason | work needed | criteria weak? |
|---|---|---|---|---|
| V.1 | IMPLEMENT | No harness, gate test or targets. My ad-hoc runs show the development, standard and large profiles run (98.5 / 64.9 / 22.2 sim days per wall hour), but class 4 is inert and runs were 1 day on 1 seed. | Gate V benchmark of every profile, all four classes active, at least 30 days, at least 3 seeds, with the 13.8 quantities, against targets recorded beforehand. | **yes**: "3 profiles benchmarked" says nothing about class activity, horizon or targets met. |
| V.2 | EXTEND | 10M capability exists only as the simplified crowd path (unofficial run on an ancestor commit) and B's internet-profile creation (my run, 8.4 GB). There is no integrated or official run. | An official integrated 10M run (pop with social) of at least one simulated day, recording identities that decided, exact recounts, RSS, wall and disk against targets. | **yes**: counting identities does not show they act or are integrated. |
| V.3 | REPAIR | Counts exist and are computable (0.19M adaptive, 2.16M and 2.36M strategic, 596M Qwen3-0.6B, D1 encoders; `pop.policy_update.params`) but are not reported in any Gate V record. | Aggregate the counts into the Gate V / SP-13.8 report from the loaded objects. | no |
| V.4 | EXTEND | Micro speedup 5.9-7.8x reproduced. In the world, batches are tiny (2 calls, max 45 in 3 h at the standard profile). Alternatives are not benchmarked; Numba is absent. | Measure in-world batched vs per-participant cost at equal decisions; benchmark NumPy, compiled, quantized and GPU per 13.3 and record the choice. | **yes**: any micro benchmark clears 2x. |
| V.5 | EXTEND | My probe: 2.18x wall, 28.6x fewer activations. No gate test. Fixed costs dominate. | Gate test at the large and internet profiles; show cost proportional to due participants after the hot-path fixes. | **yes**: 2x against a dense tick is a low bar and is profile-dependent. |
| V.6 | EXTEND | Cache correct (0.983 hit rate in my probe), but not on the default integrated path. | Measure the hit rate on an integrated multimodal workload across worlds and participants, with invalidation on encoder or preprocessing change. | **yes**: a synthetic reuse workload trivially exceeds 0.9. |
| V.7 | IMPLEMENT | No integrated measurement. The slots log has max-child RSS and CPU-s only, no VRAM. | Sample CPU, RSS and VRAM (nvidia-smi per pid, `torch.cuda.max_memory_*`) for every profile run. | **yes**: booleans "measured" are trivially satisfiable; require values within budget. |
| V.8 | REDESIGN | Violations exist and go undetected: 13 of 219 logged jobs over their CPU reservation, and my 1,024 MB VRAM use under a 100 MB reservation. Nothing enforces or detects them. | Enforcing governor (SP-13.7) and per-run violation accounting over all official runs. | **yes**: "violations == 0" without defined, measured dimensions is self-reported. |
| V.9 | IMPLEMENT | No approximate mode and no comparison; the claimed estimated mode is absent. | Low-fidelity tier, paired high-fidelity runs, errors for the 7 quantities with CIs and documented limits. | **yes**: counts metrics, no error bounds. |
| V.n1 | IMPLEMENT | No targets recorded anywhere (grep of docs). | Adopt the targets proposed in section 6 through DECISION_LOG before any official Gate V run; never weaken them without change control. | no |
| SP-13.7 | REDESIGN | 3 of 7 dimensions tracked, 0 enforced; no FIFO; zombie leak; not consulted by the simulation. | See the resource-scheduler subsystem in section 4. | **yes**: "limits_enforced >= 7" needs adversarial over-allocation tests per dimension. |
| SP-13.8 | IMPLEMENT | No v5 report. The 4.0 baseline exists. My numbers cover 9 of 12 quantities ad hoc; there is no equivalent-workload design (4.0 bazaar aggregate consumers vs v5 individuals differ 82x in speed). | Report the 12 quantities per profile plus a designed equivalent-workload comparison with 4.0, including experience-per-compute. | **yes**: one comparison row of any workload passes. |

## 6. Proposed machine-specific Gate V throughput targets (V.n1; to be adopted before official runs)

These are a proposal for change control, not adopted targets. They are derived from this machine (11.52-CPU quota with
19% historical throttling, a 9-CPU slot budget, 90.5 GiB container RAM, one A6000 with a 24 GB slot budget, 21 GB free
disk) and from what the research needs: Gate P's 30-365-day runs, and many seeds within a modest budget. Rates are per
world process on 1 CPU, measured as CPU-seconds so tenant noise does not move them, with wall time also reported.
"Now" is what I measured.

| target | proposed | now | met? |
|---|---|---|---|
| development profile throughput | >= 60 sim days per CPU-hour (30-day CI run <= 30 min) | 98.5 / wall h (day 1) | yes (day 1 only) |
| standard profile throughput | >= 46 sim days per CPU-hour (365-day run <= 8 h on 1 CPU) | 64.9 | yes (day 1 only) |
| large profile throughput | >= 7.6 sim days per CPU-hour (365 days <= 48 h on 1 CPU) | 22.2 | yes (day 1 only) |
| internet profile (10M) | >= 0.6 sim days per wall hour on <= 4 CPUs, peak RSS <= 32 GB | not measured (about 1 / h extrapolated; 8.4 GB at night) | unknown |
| storage growth (events plus blobs) | <= 200 B per individual-day; <= 15 GB per official run | 1,830 B per individual-day (large) | **no (9x over)** |
| snapshot | <= 1 GB and <= 60 s at 10M; fork exact | 33 MB / 1.9 s at 100k | unknown at 10M |
| world-generation cost | <= 30 s per world (any profile) | about 15-20 s sandbox site deployment plus 3 s create (large) | yes |
| class-2 batched inference | >= 15,000 decisions per CPU-s at B >= 256; in-world mean batch >= 64 | 20,353 micro; in-world max 45 | in-world no |
| event-driven cost proportionality | cost per sim day grows <= 1.2x per 2x due participants at fixed N; dense/event wall ratio >= 10x at the large profile | 2.18x at standard | no |
| semantic service | >= 500 generated tokens/s at 0.6B, batch >= 16, <= 8 GB VRAM | 252 (ws-D2 claim) | no |
| embedding cache | hit rate >= 0.9 on the integrated multimodal workload over 3 worlds | 0.983 synthetic; not integrated | unknown |
| parallel worlds | aggregate efficiency >= 0.85 at 8 worlds within the 9-CPU budget | 0.85 (4.0 baseline) | not rerun on v5 worlds |
| budget compliance | 0 violations in 7 dimensions over every official run, detected by enforcement | 13 or more CPU overruns logged; VRAM overrun shown | **no** |

## 7. Missing requirements (Part 13 or Part 3.3 sentences with no manifest entry)

1. 13.3 "Benchmark alternatives (NumPy, compiled numerical kernels, Numba or equivalent, PyTorch CPU, quantized neural
   inference, batched GPU inference) ... Choose based on measured workload characteristics."
2. 13.3 "Avoid excessive Python object creation in high-frequency paths." Violations are listed in section 3.10.
3. 13.4 The nine semantic-service features as requirements: batching, quantization, short context, retrieval, prefix
   caching, bounded generation, model versioning, CPU execution, selective GPU. Also "Use semantic generation only
   where necessary", "Ordinary preference decisions should generally be numerical", and "no network-dependent external
   API calls in the default simulation".
4. 13.5 "Reuse representations across participants and simulation worlds where valid." E.4 covers only invalidation.
5. 13.6 "investigate" the 10 techniques; "low-resolution participants must not become a source of fabricated
   outcomes"; "Document the limits of approximation"; Phase 5 "Validate approximate simulation modes".
6. 13.7 "Inspect available hardware"; "Do not interfere with existing CAM experiments" (under Clarification 1, other
   tenants); "Establish performance budgets before conducting major stress tests"; "No hidden external compute
   purchases".
7. 13.8 and the Part 13 preamble: "maximizing meaningful, diverse learner experience within a modest research compute
   budget". There is no experience-per-compute metric.
8. 3.3 "Benchmark every profile", including the internet profile. V.1 names only three profiles and V.2 only "tested".
9. 3.3 "Large profiles must use resource-aware sparse or lazy state and batched inference".
10. 1.2 "Do not start sixteen expensive GPU jobs simultaneously ... coordinate actual compute through a resource
    scheduler".
11. 13.1 "Group compatible decisions for batched inference" across all neural and semantic classes. G.3 covers class 2
    only.
12. Amendment (a) for the owned path: `crucible.scale.rollout.WorldEnv` and the bench `rollout` section must default to
    `observation_only`. OO.1 (ws-L) is meant to cover every environment, but the 4.0 farm path in `scale/` is not on
    its list of environment families.

## 8. Transfer and shortcuts

* **Shaped reward by default.** The default `WorldEnv` reward is the balance delta (`rollout.py:203-209`). A learner
  trained through the 4.0 farm path is trained on a shaped scalar that does not exist on the real internet, and
  exogenous income looks like reward. This must default to observation-only (amendment (a)).
* **Nothing is fabricated for scale, but there is no route to it either.** No population counter is bumped directly;
  everything goes through recorded batches, and distinct viewers are exact bitsets. Avoiding aggregate approximations
  is good for transfer. With no validated approximate tier, the internet profile is limited by storage and Python hot
  paths instead.
* **Population rows that never act.** Class 4 rows exist but never act. Any "simultaneously active" or "population
  size" claim that counts rows overstates what a learner meets.
* **A lexical content model.** The default content representation is lexical random indexing. An agent can learn to
  stuff words that match audience interest vectors, a shortcut real semantic encoders would penalise. This needs the
  ws-D encoders on the default path, together with the 13.5 cache.
* **Costs that do not depend on decisions.** Sandboxes, per-individual tool calls and storage dominate. Learner
  experience per unit of compute is therefore low; Part 13's stated objective is never measured.

## 9. Tests and probes run

| command | result |
|---|---|
| `slots --cpu 1 -- pytest -q -p no:cacheprovider -x tests/test_v5_slots.py tests/test_v5_substrate_columns.py tests/test_v5_social_idset.py` | 21 passed in 4.9 s |
| `slots --cpu 1 -- python profile_bench.py development 24` | ok: 98.5 sim days per wall hour, 717 MB peak, 0 policy updates |
| `slots --cpu 1 -- python profile_bench.py standard 6` and `standard 24` | ok: 41.6 (night-heavy 6 h), 64.9 (24 h), 812 MB |
| `slots --cpu 1 -- python profile_bench.py large 24` | ok: 22.2 sim days per wall hour, 1,497 MB, 184 MB per sim day |
| `slots --cpu 1 -- python internet_profile_probe.py 1000000 1` | ok: create 5.9 s, 17 s per daytime sim hour, 2.3 GB peak |
| `slots --cpu 1 -- python internet_10M_create_probe.py 10000000` | ok: create 36.5 s, 8.4 GB peak, 23.7 s for the first 30 sim minutes |
| `slots --cpu 1 -- python tick_profile_probe.py 2000000` | 11.0 of 17.7 s in `epoll` waits on sandbox site deployment; 97 `fork_exec` calls |
| `slots --cpu 1 -- python subproc_probe.py 200 24` and `2000 24` | 81 and 107 sandbox Python processes, 83 and 109 `ldconfig` forks; children CPU larger than world CPU |
| `slots --cpu 1 -- python event_driven_probe.py standard 3` | speedup 2.18x wall, 28.6x activations; due scan 26.5 ms at 10M |
| `slots --cpu 1 -- python batch_inference_probe.py 32` / `256` / `1000` | batched 7.8x / 6.4x / 5.9x; NumPy 0.34-0.45x; int8 1.19-1.40x with 0.91-0.94 argmax agreement |
| `slots --cpu 1 -- python cache_probe.py <fresh dir>` | hit rate 0.983; 200 computes for 12,000 requests; 50 of 50 disk hits from a second cache |
| `slots --cpu 1 --gpu-mb 100 -- python slots_enforcement_probe.py 1024 1024` | affinity 48 CPUs, rlimits unlimited, 1,024 MB RAM, torch threads set to 8, **1,024 MB VRAM under a 100 MB reservation** |
| `python slots_fairness_probe.py <private ledger>` | the later 1-CPU request was admitted before the earlier 2-CPU request (no FIFO); the zombie child counted as alive (process killed by `timeout` because the starved thread never finished) |
| `python usage_analysis.py /tmp/crucible-slots/usage.jsonl` | 13 of 219 jobs at more than 110% of reserved CPU (max 1.8x); no VRAM field |
| `slots --cpu 1 -- examples/bench_openworld.py --quick --only kernel,families --families tiny,bazaar,open_digital,open_industry,mech_scm_additive --workers 1 --no-merge --out <scratch>` | no regression against the 4.0 baseline (8,147 vs 7,978 tx/s; families within ±10%) |
| `slots --cpu 1 -- python -m crucible.substrate.bench --rows 100000 --table-rows 1000000 --store sqlite --events 5000 --json <scratch>` | reproduces A.md: 0.020 s transaction, 0.018 s replay (exact), 10.5 us per event |
| `slots --cpu 1 -- python blob_compress_probe.py` | batch blobs compress only 1.53x with zlib-6 (26.5 MB to 17.3 MB) |
| `slots --cpu 1 -- python rollout_reward_default_probe` | `WorldEnv` default `legacy_shaped_reward`; reward 12,345.0 for an exogenous deposit with no action |
| `forum_quadratic_probe` | `set(own)` per element: 0.387 s against 0.00034 s hoisted (5k/5k) |

Not run, with reasons:

* `tests/test_scale.py` (4.0 `WorldPool`): it needs several worker processes.
* Gate I at 10M: the C_social auditor was running it concurrently.
* The semantic service on the GPU: ws-D2's Gate H and SP-3.2b runs belong to the D_semantic auditor, who was running
  them.
* The acceptance checker `run` for V.*: it would write official evidence under `docs/v5/evidence`, and there are no
  test files to run.

## Skeptic review

Reviewer: skeptic for area M_performance, 2026-11-26. Tree: `HEAD` 7c29c75. Its `src/` is identical to the audited
`a881d18` except `acceptance/v5/manifest.py` and `requirements.py` (OO.* and RP.* entries); the rest of the diff is
docs, including DECISION_LOG D-14 to D-21.

Method: I tried to refute every RETAIN and every generous verdict with a probe of my own. Probe scripts, logs and JSON
are in `/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/M_performance_skeptic/`. Heavy runs
went through `crucible.scale.slots`: `--cpu 1`, except `--cpu 2` for the bench sections, `--cpu 4` for
`test_scale.py`, and `--gpu-mb 4000` for the semantic service. I modified no existing file except this report and its
JSON.

### S.1 Verdicts I overturn

| target | auditor | skeptic | probe and evidence |
|---|---|---|---|
| Multimodal caching | EXTEND | **REPAIR** | `cache_invalidation_probe.py`. I set `TextEncoder.window` from 254 to 32, which changes the computation, and `cache_key` stayed identical. The cache then served the old vector (`served_equals_old: true`), with cosine **0.797** to the vector the new setting really produces. The key's "preprocessing" is a hand-written dict (`semantic/encoders/text.py`, `preprocessing = {... "window_wordpieces": 254 ...}`), not derived from the live attributes. `CODE_VERSION` is a manual bump (`encoders/base.py:13`). The auditor's probe only measured reuse (0.983 hit rate), never invalidation. Spec 13.5 "cache by ... relevant preprocessing settings" is therefore violated: a stale representation is a correctness defect, not a missing extension. This agrees with the D_semantic auditor. |
| Event-driven activation and aggregation | EXTEND | **REPAIR** | `event_trigger_probe.py`, standard profile, 24 h, integrated `population_world`. There were **0 `pop.wake` batches and 0 woken individuals in 25,861 activations**, while the world produced 919 forum replies, a `pop.employment` batch and purchases. Every activation comes from the individual's random rhythm (`next_active_at`) found by a 15-minute polling tick (`population/core.py:341-356`). The only in-tree caller of `population.wake` is `social/consumption.py:487`, and the C2 social engine cannot be loaded with `pop` (C_social: C.3 fails at feed install). None of the 13.2 triggers (incoming message, product opportunity, change in employment, world event, experimental result) activates anyone. The measured 2.18x "event-driven speedup" is the speedup of sparse random scheduling over a dense tick, not of event-driven computation. Needed: wire forum, mail, commerce, employment and world-event producers to `wake`, then remeasure. |
| WorldPool / RolloutFarm | REPAIR (S) | **EXTEND (M)** | Three probes. (1) The 4.0 machinery is sound: `tests/test_scale.py` **18 passed** on this tree, which the auditor had not rerun. (2) It cannot host any v5 world. `generate({"family": "population"})` and `WorldEnv({"family": "population"})` raise `KeyError: unknown world family`; all 74 registered families are 4.0 families (`pool_hosts_v5_probe.py`). `population_world` is referenced only by `tests/test_v5_population_core.py`. (3) It leaks memory in long-lived workers. `pool_leak_probe.py` created, ran (14 sim h) and closed 4 development worlds in one process: `artifacts._POOL` grew 12.0, 24.1, 36.1, 48.2 MB and RSS after `close()` grew 714 to 769 MB. Bytes are never released (`substrate/artifacts.py:63,165`), so a RolloutFarm worker running large-profile episodes (147 MB of blobs per sim day) keeps about 4.4 GB per 30-day episode. In addition, `experience.modes.check_mode(None)` returns `legacy_shaped_reward` (`reward_default_probe.py`), contrary to D-19, and `WorldEnv`'s default gave reward 12345.0 for an exogenous deposit (reproduced; 0.0 with `observation_only`). The gap is v5 hosting plus bounded memory, not a one-line default. |
| 10M-identity scale path | EXTEND | **REDESIGN** | Under D-14 and D-15 (one population; gating on a stand-in is ruled out), the only 10M run with acting identities, the social `crowd` path, is inadmissible. The social engine cannot run with `pop`. I ran B's integrated population at 10M through a daytime hour, which the auditor never did (`pop10M_daytime_probe.py`, internet profile, 1 CPU slot); results are in S.3. In short: the identities do decide, but it is neither social nor sustainable. Two disjoint 10M paths must become one bounded-memory population with platforms attached. That is a redesign, consistent with D-14. |

### S.2 Requirement verdicts I overturn

| id | auditor | skeptic | why |
|---|---|---|---|
| V.2 | EXTEND | **IMPLEMENT** | No admissible 10M social run exists. `crowd` is a stand-in (D-14/D-15), and C2 cannot load with `pop`. B's population at 10M decides (S.3) but has no social platform attached and grows memory about 1.8 GB per daytime sim hour. Criteria weak (identities counted, not acting on platforms). |
| V.5 | EXTEND | **REPAIR** | The mechanism exists, but no event triggers anyone in the integrated world (0 wakes in a standard day), so "event-driven computation demonstrated" is not demonstrated. The `event_driven_speedup >= 2` criterion would pass on a pure random-rhythm scheduler; it should require activations caused by recorded trigger events and cost proportional to them. Criteria weak. |
| V.6 | EXTEND | **REPAIR** | The cache key is not derived from the preprocessing actually used, so a stale vector is served (S.1). The default integrated path does not use the cache. Criteria weak (a hit rate passes on a workload that never changes the encoder). |

### S.3 New evidence

**10M integrated population, daytime hour** (internet profile, n = 10,000,000 class 1 + 1,120 others, seed 0, one
loaded CPU slot):

| measure | value |
|---|---|
| create | 35.6 s; 4.6 GB RSS |
| 00:00 to 08:00 | 489 s; RSS 11.4 GB |
| 08:00 to 09:00 | **362 s wall (0.41 sim days per wall hour)** |
| activations / sessions | 2,795,745 / 2,517,694 |
| distinct individuals with a session | about 3.16M (counting the 08:00 tick) |
| class 2 / 3 / 4 with a session | 481 of 1,000 / 0 of 100 (they act only through `pop_strategic`) / **0 of 20** |
| blob bytes | **755.8 MB in the hour** (in RAM: no store) |
| RSS growth | 11.4 to 13.1 GB in the hour |
| principals materialised | 699,373 to 987,226 (Python dicts) |
| batches in the hour | 4,424 one-row `pop.deposit` batches vs 4 `pop.session` batches |

At these rates RSS grows about 25-30 GB per sim day: the 90.5 GiB container would be exhausted within roughly 3 sim
days. With a store, about 10 GB per sim day would be written, against 20 GB free disk. The world-generation target
in section 6 ("<= 30 s, met") is not met at 10M: creation alone takes 35.6-36.5 s.

**Storage composition** (`storage_breakdown_probe.py`, standard, 24 h; 1,664 B per individual-day of blobs):

* **36%** of the bytes written by `set` batches are values identical to what the column already held.
* `popweb.search` rewrites 17 columns and 97% of those bytes are unchanged. `popweb.seen` is 98% unchanged, and
  `popweb.web` (19 columns, including the interest, belief and knowledge vectors) is 44% unchanged.
* Delta encoding (about 1.6x) combined with zlib (1.53x, the auditor's figure) gives about 2.4x, or about 700 B per
  individual-day. That is still about 3.5x over the auditor's proposed 200 B target.
* So only checkpoint plus compaction (D-21) can bound storage. The EXTEND verdict stands, but the delta and compression
  items in the work list cannot reach the target alone.

**Semantic service on the GPU** (`semantic_service_probe.py`, Qwen3-0.6B bf16, 16 requests, 885-token shared
prefix, 32 new tokens):

* Batched generation ran at **303 tok/s**, against 36.6 tok/s one request at a time (8.56x).
* The prefix cache was 1.35x faster.
* Outputs were identical with and without the prefix cache for 11 of 16 requests, and identical between batched and
  single-request runs for 12 of 16. A participant's reply therefore depends on who shares its batch. It is recorded,
  so replay is exact, but this is a cross-participant coupling that ought to be documented.
* Peak VRAM was **4,384 MB inside a 4,000 MB slot**, a second unenforced GPU overrun.

The auditor's EXTEND rested on ws-D2's numbers. It now rests on my measurement and stands.

**Scheduler** (`slots_escape_probe.py`, private ledger):

* **Escape.** A job that backgrounds a busy grandchild and exits releases its slot after 0.11 s. The ledger shows 0
  CPUs used while the grandchild runs, and `usage.jsonl` records `wall_s 0.0, cpu_s 0.0`.
* **Self-declared budget.** With the ledger full (2 of 2 CPUs held), a caller that sets `CRUCIBLE_SLOT_CPUS=1000` is
  admitted with a 4-CPU request after 0.11 s. The budget is not stored in the ledger, so every caller uses its own.
* **Oversize.** A request for 64 CPUs and 999,999 MB of GPU memory is admitted on an idle ledger and is not flagged
  in the log.

REDESIGN stands. The harsh verdict is justified, and the "sound admission ledger" the auditor wanted to keep is not
sound either, because its budget is self-declared.

**D-16 cost risk** (`encoder_throughput_probe.py`): all-MiniLM-L6-v2 on one CPU thread encodes **4.7 page-sized
documents per second** (about 660 words each). D-16 retires lexical random indexing, so every new artifact must be
encoded. At the standard profile's roughly 1,200-1,700 new text artifacts per sim day, that adds roughly 260-370 CPU-s
per sim day, against 55 s per sim day for the whole world today. The section 6 throughput targets assume the lexical
path, and they become unreachable on CPU unless encoding is batched on the GPU or restricted. This is an estimate, not
a measurement; page lengths vary.

**4.0 suite** (`bench_sections_probe.py`; the auditor ran only kernel and families): storage, memory and rollout all
ran on the v5 tree.

* Storage: events and bytes are identical to the 4.0 baseline for 13 of 14 families. `tiny` differs by 2 events,
  and `tiny_determinism_probe.py` shows it varies run to run on both the 4.0 source (`c07098b`, extracted with
  `git archive`) and v5, because of its default `measured` cost model. That is not a regression.
* Memory: world RSS is unchanged (5.3 vs 4.8 MB at 1k consumers, 42.1 vs 41.5 MB at 100k); the base process is 8 MB
  larger.
* Rollout: 119 vs 134 steps/s at 1 worker, under higher load.

**RETAIN is upheld, but only in its stated narrow role** as the 4.0 instrument. It registers no v5 world, so it cannot
supply the v5 side of an equivalent-workload comparison. Its `rollout` section measures the legacy shaped-reward
default.

### S.4 Verdicts I tried to overturn and uphold

| target | verdict | note |
|---|---|---|
| Resource scheduler / V.8 / SP-13.7 | REDESIGN | Strengthened by S.3 (escape, self-declared budget, second VRAM overrun). |
| v5 benchmark harness / V.1 / SP-13.8 | IMPLEMENT | No `test_gate_V.py`, `test_gate_SP_13_7.py` or `test_gate_SP_13_8.py`; nothing in `src/` writes a v5 performance record. |
| 4.0 benchmark suite | RETAIN (narrow) | See S.3. |
| Columnar state and batch events | EXTEND | See S.3; compaction is the necessary fix. |
| Shared weights and batched numerical inference / V.4 | EXTEND | Not re-probed; the auditor's micro-benchmark is their own runtime evidence. In-world batches are tiny (max 45), so V.4's 2x micro criterion is weak. |
| Compact semantic service | EXTEND | Now on my GPU measurement (S.3). Class 4 is still inert: 0 of 20 class-4 rows acted at 10M. |
| Large-population techniques | EXTEND | The 10M run shows lazy materialisation still produces about 1M Python-dict principals in 9 sim hours, and a one-row `pop.deposit` batch per payer (4,424 per hour). The columnar principal and bank work the auditor listed is required, not optional. |
| Approximation vs high fidelity / V.9 | IMPLEMENT | A grep for approximate, aggregate-transition, adaptive-resolution or estimated modes in `src/` finds nothing. |
| Hot paths | REPAIR | Verified `population/web.py:775` (`set(own)` rebuilt per element) and the per-reader principal and follow lookups at 773. Add the unchanged-column rewrites of S.3. |
| V.3 | REPAIR | Counts exist in `pop.policy_update.params` (`core.py:506`), `nets.n_params` and `SemanticService.info()["params"]` (596,049,920 measured). Only the report is missing. |
| V.7 | IMPLEMENT | Harsh but correct: the requirement is the measurement record, which does not exist. The instruments do exist (`bench.cgroup_cpu`, `bench.sample_gpu`, `semantic.service.peak_vram_mb`), so effort is S. |
| V.n1 | IMPLEMENT | Caution: the section 6 targets were proposed after the auditor had seen the measurements. Several sit just below the measured values (development >= 60 vs 98.5, standard >= 46 vs 64.9, class-2 >= 15,000 vs 20,353), and none accounts for D-16. They need justification independent of the observed numbers through change control. They cannot count as "recorded before benchmark" for runs already made. |

### S.5 Additional missing requirements

1. Bounded memory in long-lived workers: `_POOL` release on `World.close()`, or a store for every world. OO.5 ("operate
   indefinitely within configured resource limits") is impossible without it and without storage compaction.
2. v5 worlds (population, Tier 2, Tier 3) must be hostable by the parallel-world and rollout machinery, so that "active
   world count" in 13.7 is meaningful for v5.
3. A performance budget for the pretrained-encoder default path of D-16 (CPU vs GPU, batch size, artifacts per sim
   day).
4. Activations caused by recorded trigger events (13.2 examples), measured in the integrated world.
5. Behavioural equivalence of batched vs single semantic inference (bf16 batch-composition dependence) and of
   quantized numerical inference.
6. A scheduler budget stored with the ledger (not per-caller environment), and accounting for detached descendants.

### S.6 Transfer and shortcut notes

* **Random-rhythm activation.** A learner never sees its message or reply cause a response, because recipients
  activate on random rhythm. Real response dynamics, such as latency after notification and reply chains, are
  therefore absent, and strategies that rely on timing notifications cannot be learned.
* **Batch-dependent semantic outputs.** Semantic replies vary with who shares the batch (4 of 16 differ), so a learner
  probing a semantic participant sees noise that does not come from that participant's state.
* **Shaped reward by default.** `check_mode(None)` is legacy and the `WorldEnv` default is shaped, so any v5 path that
  forgets to pass the mode trains on balance deltas.

### S.7 Tests and probes run by the skeptic

| command | result |
|---|---|
| `slots --cpu 4 -- pytest -q -p no:cacheprovider tests/test_scale.py -k "not throughput_grows"` | 18 passed, 1 deselected, 54.9 s |
| `slots --cpu 2 -- python bench_sections_probe.py` (storage, memory, rollout; 1 worker) | all ok; equal to the 4.0 baseline except `tiny` (nondeterministic in 4.0 too). The log also shows spawn-bootstrap errors from my probe lacking a `__main__` guard (the child re-imported the script); the parent's sections are ok. |
| `python tiny_determinism_probe.py <v5 src>` and `<4.0 src c07098b>` | both nondeterministic run to run (1941/1933 and 1932/1941 events) |
| `python slots_escape_probe.py` (private ledger) | escape after 0.11 s; self-declared budget admitted; 64 CPU / 999,999 MB admitted |
| `slots --cpu 1 -- python pool_leak_probe.py 4 6` | `_POOL` 12 to 48 MB over 4 closed worlds; RSS 714 to 769 MB |
| `slots --cpu 1 -- python event_trigger_probe.py standard 24` | 0 `pop.wake`, 0 woken, 25,861 activations, 919 forum replies |
| `slots --cpu 1 -- python storage_breakdown_probe.py standard 24` | 1,664 B per individual-day; 36% of set bytes unchanged |
| `slots --cpu 1 --gpu-mb 4000 -- python semantic_service_probe.py cuda` | 303 vs 36.6 tok/s; prefix 1.35x; 11/16 and 12/16 identical; 4,384 MB peak |
| `python cache_invalidation_probe.py` (CPU) | key unchanged after `window` change; stale vector, cosine 0.797 |
| `python encoder_throughput_probe.py` (CPU, 1 thread) | 4.7 docs/s |
| `python pool_hosts_v5_probe.py` | `KeyError` for population families in `generate` and `WorldEnv` |
| `python reward_default_probe.py` | `check_mode(None)` is legacy; default reward 12345.0, `observation_only` 0.0 |
| `slots --cpu 1 -- python pop10M_daytime_probe.py 10000000 8 1` | 362 s per daytime hour; 2.8M activations; 755.8 MB blobs; RSS 13.1 GB |
