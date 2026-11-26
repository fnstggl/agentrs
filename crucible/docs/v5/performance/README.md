# Crucible 5.0 performance (workstream M)

What is measured, by which code, and how to run it. Results are produced only by official runs of the gate tests
(`python -m crucible.acceptance.v5 run --requirement <id>`); nothing here is a result.

| piece | module | gate tests |
|---|---|---|
| resource governor (7 enforced dimensions, FIFO, stored budget, subreaper, nested ledgers) | `crucible.scale.slots`, `crucible.scale.governor`, `crucible/scale/_slotsite/sitecustomize.py` | SP-13.7 (`crucible.scale.govprobe` attacks), SP-13.7b, SP-1.2, V.8 |
| machine-specific targets, adopted before any official benchmark | `docs/v5/performance/targets.json` (DECISION_LOG D-27) | V.n1, SP-13.7b |
| benchmark harness: 12 quantities per profile, class activity, storage before/after compaction, snapshot/fork, 4.0 equivalent workload | `crucible.scale.bench5` | V.1, V.3, V.7, SP-13.8, SP-13.8b, SP-0.10, SP-3.3b, SP-13.1 |
| inference backends and behavioural equivalence | `crucible.scale.backends` | V.4, SP-13.3 |
| event-driven computation, cross-world representation cache, 10M capacity | `crucible.scale.gatev` | V.5, V.6, V.2 |
| estimated large-population mode and its validation | `crucible.scale.approx` | V.9, SP-13.6 |
| the ten 13.6 techniques | `crucible.scale.techniques` | SP-13.6 |
| shared batched inference across participant classes (compatible requests of every call site grouped into one recorded call per shared model; decision rounds) | `crucible.substrate.inference_batch` ([BATCHING.md](BATCHING.md)) | SP-13.1, V.4 |

## Running

Every heavy job runs through the governor. The gate tests nest their runs in the ledger of the job that runs them, so
give that job the whole suite's budget:

    PYTHONPATH=src python -m crucible.scale.slots --cpu 6 --gpu-mb 14000 --ram-mb 64000 --disk-mb 12000 \
        --label "M: gate V" -- python -m crucible.acceptance.v5 run --requirement V.1

Runs of a clean committed tree are kept per commit (`$CRUCIBLE_BENCH5_CACHE`, default `/tmp/crucible-bench5-cache`),
so the separate gate files of one commit share one measured suite instead of repeating it.

Single pieces:

    python -m crucible.scale.slots --hardware                     # inspected hardware
    python -m crucible.scale.slots --status                       # budget, held slots, FIFO queue
    python -m crucible.scale.govprobe                             # attack every dimension on a private ledger
    python -m crucible.scale.bench5 run --world civ --profile development --days 30 --store-dir D --out r.json
    python -m crucible.scale.bench5 v4 --consumers 1000 --days 30 --work W --out v4.json
    python -m crucible.scale.gatev event_driven --world population --profile standard --hours 24 --out e.json

## Scheduling rules (wave 3b)

* FIFO with EASY backfilling: the head of the queue starts when it fits; a later request starts only if it fits next
  to what is held plus the protected earlier requests - the head's whole request and every earlier request that
  could start now - in the dimensions it uses (`slots.fits_behind`). The head is never delayed by a later request,
  a request that is only waiting for its turn is never overtaken, every request reaches the head in queue order
  (no starvation), and CPU-only jobs do not idle behind GPU jobs waiting for the 4 GPU slots (observed 2026-11-26
  18:00 with the strict-FIFO version: 3 of 9 CPUs idle while 5 CPU-only jobs waited behind a GPU head). A blocked
  earlier request can be delayed by at most the run time of jobs started behind it. Probe:
  `python -m crucible.scale.govprobe backfill`.
* A head that waits only for free disk gives up after 15 minutes (exit status 75) instead of blocking the queue.
* Ledger and queue entries carry their process's start time (a reused pid never keeps a dead holder's slot);
  signals go through a pidfd after the same identity check, so the governor cannot signal a process that took over
  a member's pid.
* The ledger is rewritten only when it changed (dozens of waiters poll it).

## Definitions

* Throughput targets are in simulated days per CPU-hour of the world process and its children (other tenants' load
  does not move them); wall-clock rates are reported next to them.
* Storage growth: raw = event rows as the store holds them (one SQL aggregate per simulated day, outside the measured
  CPU) + array-blob bytes written; resident = store growth after workstream A's retention
  (`substrate.retention.compact`, LZMA-packed blobs), a final VACUUM and a WAL checkpoint (without it the WAL holds a
  second copy of the database and the file sizes count it twice).
* World-generation cost: `World.create` plus the first simulated day's CPU beyond a median day (deployments and
  warm-up the world's first processes do); T5 is checked on the sum.
* Event-driven speedup: CPU of the population's decision policies evaluated for every eligible individual at every
  tick (dense) over the same evaluation for the individuals each tick activated, both as observers of one world whose
  hash chain is unchanged (equal outcomes); activations, triggered wake-ups and world CPU per activation are reported.
* Cache hit rate: served representations (world records, memory and disk cache) over all lookups, over three worlds
  in separate processes sharing one fresh cache.
* The estimated mode's outputs are labelled `estimated` and never enter a world.

## Known limits (see docs/v5/workstreams/M.json)

* GPU memory per process cannot be read from outside this container (nvidia-smi reports host pids): it is capped and
  reported inside each Python process of a GPU job by the interpreter hook; a non-Python CUDA process in a GPU job is
  detected (GPU device open) but not metered.
* Enforcement is by sampling (default 1 s): a job can overshoot its RAM or disk grant by what it allocates or writes
  within one sampling interval before it is stopped.
* Older copies of the scheduler in other worktrees share the ledger but not the FIFO queue or the stored budget.
