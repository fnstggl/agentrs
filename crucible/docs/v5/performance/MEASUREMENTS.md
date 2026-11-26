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

Governor records of the six runs (V.7, V.8): peak CPU 0.95-1.005 of a 1-CPU grant, summed PSS of the job (world plus
its sandbox workers) 3.4-4.3 GB, GPU memory 1.4-2.2 GB (reported by the interpreter hook), GPU time 406-1,385 s, store
growth 402-437 MB; 0 violations, every dimension declared. The suite (6 runs, 2 at a time) took 47 minutes of wall
time.

Second row, the population-only world (`population.worlds.population_world`, 4.0 services and generated sites), same
seeds and days: 140.3 simulated days per CPU-hour (121.9-153.4), 207 transitions and 43 decisions per wall second,
7,348 raw bytes per individual-day, snapshot 8.6 MB, fork exact. It does not satisfy a target (D-27 rule 3); it is
half as fast as the civ world because sandboxed site handlers dominate it (OPTIMISATION_REQUESTS R8).

Crucible 4.0 equivalent workload (SP-13.8, SP-0.10; `bazaar` family of the c07098b tree, run from that tree's own
source, same days, as many consumers as the profile has individuals; aggregate consumers, so equal population size,
not equal fidelity):

| consumers | 4.0 simulated days per CPU-hour | 4.0 peak RAM MB | v5 civ (development, seed 0) |
|---:|---:|---:|---|
| 1,000 | 2,021 | 130 | 330 days per CPU-hour, 2,732 MB, 12.9 public events per CPU-s (4.0: 11.3) |
| 10,000 | 3,096 | 153 | (standard: section 6) |
| 100,000 | 1,367 | 361 | (large: section 6) |

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

## 5. CPU attribution, storage and optimisation requests

Standard civ world (seed 0), per simulated day without a profiler: day 1 30.7 CPU-s, days 2-3 28.7 and 27.4 CPU-s
with their texts already in the machine cache, day 4 57.2 CPU-s encoding 356 new texts (77.7 ms each): about 63
simulated days per CPU-hour on fresh content (T2 >= 15), half of it in the shared space's text path. Per-function
shares, the store breakdown by event type and the requests to the owners: OPTIMISATION_REQUESTS.md.

Store per individual-day after compaction (T6 <= 550 B): development 7,843 B (5 days), standard 2,846 B (3 days).

Fork reproducibility probe (development, snapshot at 8 h, three 6-hour forks: plain, plain, with the dense observer):
all three hash chains equal (6,383 events each), so on this world the observer does not perturb the trajectory and
forks are exact.

## 6. Reruns of the gate workloads (snapshot dc8f900 / dca1c0c, 2026-11-26 19:00-19:20)

Event-driven computation (V.5; `gatev event_driven`, civ standard, 10,296 individuals, 24 h, three forks plain /
dense observer / plain): all three hash chains equal (`8157a78443dfd5bb`), so `equal_outcomes` and
`forks_reproducible` are both true (the first run's two-fork divergence did not recur; its cause was not
identified, so the three-fork design stays). Decision-policy CPU 0.076 s for the 23,452 activated rows vs 2.144 s for every eligible
row at every tick (984,576 rows): **event-driven speedup 28.2** (activation ratio 42.0; 244 vs 10,256 rows per tick).
200 `pop.wake` trigger batches; world CPU per activation 0.66 ms; the whole world with the dense observer costs 1.10x
the plain world (the population's policy is a small share of a civ world's CPU).

Representation cache (V.6; `gatev cache`, three 96-hour civ development worlds in separate processes on one fresh
cache: two replicas of seed 0 and seed 1): 15,115 lookups, 14,382 hits: **hit rate 0.952**; 733 distinct items, each
encoded exactly once (encoder runs per artifact 1.0); the second replica was served all 353 of its items from the disk
cache (no encoder run) and the other seed 5 of its 385; 358 items reused across worlds. World 1: 4,436 world-record
hits, 353 misses; world 2: 4,436 world-record hits, 353 disk hits, 0 misses; world 3: 5,152 world-record hits, 5 disk
hits, 380 misses.

Estimated cohort mode (V.9; `approx.validate`, fit at 400 individuals on seed 0, predict 1,600 individuals on seeds
1-3 and a shop-share intervention 0.4 -> 0.7, 4 days; 1,099 s): error (mean over the 3 target seeds, 95% interval)

| quantity | error measure | mean | 95% interval |
|---|---|---:|---|
| adoption trajectory | max abs difference of the daily active share | 0.012 | 0.003-0.020 |
| audience distribution | KS distance of per-resource audience shares | 0.906 | 0.813-1.000 |
| revenue | relative error of total revenue | 0.623 | 0.389-0.856 |
| network diffusion | mean abs difference of the top resource's daily reach | 0.379 | 0.321-0.438 |
| retention | abs difference of retention | 0.002 | 0.001-0.003 |
| intervention effect | relative error of the shop-share effect on revenue | 0.992 | 0.389-1.595 |
| rare-event frequency | relative error of rare individual-days | 0.018 | 0.007-0.030 |

Reading: the cohort mode reproduces aggregate activity (adoption, retention, rare-event frequency within 2%) and
nothing that depends on who meets what: audience distributions, revenue, diffusion and above all intervention effects
are wrong by 40-100%. It is therefore labelled `estimated`, is never used inside a world, and must not be used to
answer counterfactual or distributional questions (FIDELITY_REGISTER `scale.approx_cohort`).

Shared batching service (V.4, SP-13.1; `backends.bench_service`, 6 call sites x 4 rows of one shared model each,
recorded as 6 separate calls vs grouped into 1, inside a world transaction):

| model (class) | separate s / round | grouped s / round | speedup | max abs output difference |
|---|---:|---:|---:|---:|
| AdaptiveNet (2) | 0.00520 | 0.00176 | 2.96 | 1.5e-7 |
| StrategicGRU (3) | 0.02576 | 0.00876 | 2.94 | 2.1e-7 |
| StrategicMLP (3) | 0.00855 | 0.00251 | 3.41 | 1.8e-7 |
| Qwen3-0.6B option scoring, bfloat16 on the GPU (4) | 0.1958 | 0.0689 | 2.84 | 0.256 nats |

The class-4 logprobs change with the batch composition in bfloat16 (different GEMM tilings round differently); the
calls are recorded, so replay is exact, and from this version on the benchmark also reports the agreement of the
chosen options (`choice_agreement`), which is what a participant acts on.

## 7. Pending

The 30-day standard (3 seeds) and large (1 seed) civ runs, the population-only standard row and the 10M capacity
run (V.2) are queued as governed jobs; their results replace this paragraph.
