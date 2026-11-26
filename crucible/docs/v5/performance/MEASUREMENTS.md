# Development measurements of workstream M (wave 3b, 2026-11-26)

Unofficial: measured by workstream M on the integration head 55433e7 plus its own measurement code (branch
c5c/w3b-perf), in governed jobs on the shared machine. Official evidence comes only from the gate tests run by the
acceptance machinery from a clean clone. Everything below is a measurement of the integrated civilisation world
(`crucible.civ.generate("balanced", seed, population=<profile>)`, every subsystem and the one population) unless a
row says otherwise. Targets: docs/v5/performance/targets.json (D-27).

## 1. Civilisation world, development profile (1,042 individuals), 3 seeds x 30 simulated days

Source snapshot 91f6a52 (`bench5 run`, governed sub-jobs, one CPU thread, space LM on the GPU).

| quantity (Part 13.8) | seed 0 | seed 1 | seed 2 | target |
|---|---:|---:|---:|---|
| simulated days per CPU-hour | 330.0 | 297.9 | 277.2 | T1 >= 60: met |
| simulated days per wall hour | 268.2 | 220.7 | 172.1 | - |
| environment transitions per second (events / wall s) | 745 | 634 | 480 | - |
| participant decisions per second (wall) | 249 | 198 | 154 | - |
| CPU utilisation (CPU s / wall s) | 0.81 | 0.74 | 0.62 | - |
| peak RAM (VmHWM, MB) | 2,732 | 2,734 | 2,738 | - |
| peak VRAM (torch reserved, MB) | 2,090 | 2,190 | 2,134 | - |
| world generation (World.create CPU s / + first-day excess) | 3.4 / 23.8 | 3.4 / 23.8 | 3.8 / 21.6 | T5 <= 120: met |
| raw bytes written per individual-day (events as stored + blobs) | 7,322 | 7,598 | 7,528 | - |
| resident bytes per individual-day (this snapshot counted the WAL twice; corrected in 092b79e) | 10,695 | 11,084 | 11,005 | T6 <= 550: MISSED |
| snapshot bytes / seconds; fork exact | 11.6 MB / 0.96 s; yes | 12.7 MB / 0.97 s; yes | 13.6 MB / 1.51 s; yes | (T7 is large only) |
| sessions / decisions recorded | 59,183 / 100,231 | 57,237 / 96,843 | 57,738 / 96,734 | - |
| public events / distinct kinds / acting principals | 4,235 / 24 / 1,079 | 4,030 / 26 / 1,072 | 4,026 / 25 / 1,072 | - |
| shared-space LM tokens / texts encoded / world-record hits | 480k / 1,548 / 55,463 | 559k / 1,757 / 59,117 | 604k / 1,792 / 55,574 | - |
| model parameters (counted on the loaded objects) | class-2 AdaptiveNet 195,777; class-3 StrategicMLP 2,162,715, StrategicGRU 2,362,203; Qwen3-0.6B 596,049,920 | same | same | V.3 |

Steady cost: 9-13 CPU-s per simulated day from day 2 to day 30 (no growth over 30 days); day 1 costs about 30 CPU-s
(lazy imports, model loading, deployments).

Class activity (V.1): class 1 1,000 of 1,000 rows active, class 2 32 of 32, class 3 decides through its own process
(120 member decisions in 30 days; counted as acting from 092b79e on), class 4 0 of 2 (unbound). V.1 cannot pass until
class 4 acts.

Batching in the world (V.4, SP-13.1): class 2 134 recorded calls for 198 decisions (mean batch 1.29-1.48, max 4);
class 3 120 calls for 120 decisions (mean 1.0); class 4 none.

## 2. Backend matrix (V.4, SP-13.3; class-2 AdaptiveNet, 195,777 parameters, one CPU thread)

Decisions per second (paired argmax agreement with torch CPU in brackets where below 1):

| backend | B=1 | B=32 | B=256 | B=1024 |
|---|---:|---:|---:|---:|
| torch, one call per participant | 3,773 | 3,952 | 4,095 | - |
| torch CPU batched | 3,848 | 26,002 | 28,447 | 17,656 |
| NumPy re-implementation | 5,005 | 9,243 | 9,242 | 7,991 |
| TorchScript (JIT, the in-environment Numba equivalent) | 4,108 | 25,799 | 30,340 | 18,393 |
| Inductor (compiled kernels) | 3,048 | 26,023 | 32,710 | 28,472 |
| int8 dynamic quantisation | 2,808 | 30,087 (0.995) | 19,098 (0.993) | 24,129 (0.993) |
| GPU fp32 | 988 | 33,068 | 228,856 | 376,546 |
| GPU fp16 | 342 | 19,171 | 61,678 | 150,940 |

Measured choice: NumPy at B=1, TorchScript at B=8, GPU fp32 from B=32 (best CPU: int8 at 32-64, Inductor from 256).
Batched vs per-participant: 6.6x at B=32, 6.9x at B=256 (micro); at the civ world's own batch sizes (mean 1.5) only
1.41x: V.4 (>= 2 in the world) is not met until class-2 decisions are aggregated (OPTIMISATION_REQUESTS R7).
Strategic: MLP (2,191,385 parameters) 11.9x batched at B=100, GRU (4,203,289) 9.6x. Text encoder: 1.5x batched (16
texts).

## 3. Event-driven computation (V.5), civilisation world, standard profile (10,296 individuals), 24 h

First run (snapshot 91f6a52, two forks): decision-policy CPU for the activated individuals 0.079 s vs 2.219 s for
every eligible individual at every tick: speedup 28.1 (activation ratio 42.0; 244 vs 10,256 rows per tick); 200
`pop.wake` trigger batches (activations caused by recorded events: messages, notifications, shares); world CPU per
activation 1.6 ms. The two forks' hash chains differed (`equal_outcomes` false), so the comparison now runs three
forks (plain, observed, plain) and reports whether forks of this world are reproducible at all; rerun in section 6.

## 4. Representation cache (V.6)

First run (three 12-hour development civ worlds, separate processes, one fresh cache): hit rate 0.670 (651 of 972
lookups), one encoder run per artifact, 156 items served across worlds from the disk cache. Twelve hours measure
almost only first lookups (each item looked up about twice); a 30-day world serves 55-59k lookups from its record
for 1.5-1.8k encodings (0.97). The workload now runs 96-hour worlds; rerun in section 6.

## 5. CPU attribution and optimisation requests

See OPTIMISATION_REQUESTS.md.

## 6. Pending reruns

PENDING_SECTION
