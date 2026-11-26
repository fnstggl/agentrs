# Crucible 5.0 performance (workstream M)

What is measured, by which code, and how to run it. Results are produced only by official runs of the gate tests
(`python -m crucible.acceptance.v5 run --requirement <id>`); nothing here is a result.

| piece | module | gate tests |
|---|---|---|
| resource governor (7 enforced dimensions, FIFO, stored budget, subreaper, nested ledgers) | `crucible.scale.slots`, `crucible.scale.governor`, `crucible/scale/_slotsite/sitecustomize.py` | SP-13.7 (`crucible.scale.govprobe` attacks), SP-13.7b, SP-1.2, V.8 |
| machine-specific targets, adopted before any official benchmark | `docs/v5/performance/targets.json` (DECISION_LOG D-26) | V.n1, SP-13.7b |
| benchmark harness: 12 quantities per profile, class activity, storage before/after compaction, snapshot/fork, 4.0 equivalent workload | `crucible.scale.bench5` | V.1, V.3, V.7, SP-13.8, SP-13.8b, SP-0.10, SP-3.3b, SP-13.1 |
| inference backends and behavioural equivalence | `crucible.scale.backends` | V.4, SP-13.3 |
| event-driven computation, cross-world representation cache, 10M capacity | `crucible.scale.gatev` | V.5, V.6, V.2 |
| estimated large-population mode and its validation | `crucible.scale.approx` | V.9, SP-13.6 |
| the ten 13.6 techniques | `crucible.scale.techniques` | SP-13.6 |

## Running

Every heavy job runs through the governor. The gate tests nest their runs in the ledger of the job that runs them, so
give that job the whole suite's budget:

    PYTHONPATH=src python -m crucible.scale.slots --cpu 6 --gpu-mb 2000 --ram-mb 64000 --disk-mb 12000 \
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

## Definitions

* Throughput targets are in simulated days per CPU-hour of the world process and its children (other tenants' load
  does not move them); wall-clock rates are reported next to them.
* Storage growth: raw = event bytes (canonical JSON of every recorded event) + array-blob bytes written; resident =
  store growth after workstream A's retention (`substrate.retention.compact`, LZMA-packed blobs) and a final VACUUM.
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
