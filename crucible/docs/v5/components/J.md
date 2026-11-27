# Workstream J: Tier 3 civilisations (`crucible.civ`)

MASTER_SPEC Part 10, Part 2.1 Mode C; Gates O and P; SP-10.1, SP-10.3; ONE_WORLD_CONTRACTS section 4 (D-17).

## What exists

| file | what |
|---|---|
| `civ/subsystems.py` | the 17 Mode C subsystems and their modules; modules excluded from every civilisation (second populations, scripted shocks) with the reason; `readiness()`: modules whose population hook is not in this checkout (today `feed`) |
| `civ/profiles.py` | 11 profiles (`balanced` + the ten of Part 10.1), each a structural variant: coupling set, supply-chain graph, freight topology, energy and compute regimes, datacenter placement, social placement, boards, institutions, science bindings |
| `civ/generator.py` | `recipe` (inspectable 4.0 family form), `spec`, `generate`; supply-chain and freight generators; science bound to industrial physics (one fact, verified by an invariant) |
| `civ/genesis.py` | module `civgen`: compute providers founded by population members, datacenter regions, constitutions, the R&D routine of producers |
| `civ/goods.py` | module `civgoods`: the population's `goods` activity (households buy the supply chain's retail stock with their own money; awareness by local browsing, search and word of mouth from own contacts) |
| `civ/couplings.py` | module `civlinks` (operator host `admin`): coupling records, removal/restoration/placebo tools, the mechanisms `science_process`, `compute_energy`, `grid_curtailment`, `org_coordination`, `freight_energy`, `lab_energy` (same-transaction subscribers), `energy_cost_pricing`, `datacenter_siting` (providers' daily routines); `education_skills` is recorded as pending |
| `civ/popapi.py` | the population API of the one-world contract in one place (contract export when present, `population.core` otherwise) |
| `civ/chains.py` | provenance graph over the event log (transactions, schedules, entity timelines, column rows, explicit refs), `trace`, generic chain mining and labelling |
| `civ/counterfactual.py` | fork, intervene, rerun; two placebo arms measure the reshuffling of seq-keyed randomness; multi-seed paired statistics |
| `civ/experiments.py` | the effect battery (Part 10.2 categories 1, 2, 3, 4, 5, 7, 8) and the chain experiments (scripted reference agents) |
| `civ/reference.py` | scripted reference agents (labelled `scripted`): a web-shop builder and a researcher |
| `civ/longrun.py` | long-horizon harness: daily invariants and activity, accounting audits, snapshots, segment replay, cross-process resume, wall/RSS/store growth; every snapshot is a checkpoint (`progress.json`), an interrupted run resumes there in a new process; `suite` for Gate P runs in persistent per-run directories keyed by the code digest (`$CRUCIBLE_CIV_RUNS`, default /workspace/data/crucible/runs-c5/civ) and skips completed runs |
| `civ/accounting.py` | event accounting (events vs state) and economic/social activity classification |
| `civ/integration.py` | Gate O measurements on data flow; data-level duplicate scan; identity and ledger checks; structural fingerprint of built worlds |
| `civ/modes.py` | the Mode C provider and its check (determinism, replay, and integration measured, not assumed) |
| `civ/experience.py` | `tier3:<profile>` entry point, `observation_only` by default |
| `civ/gates.py` | the gate measurements (cached by the digest of the whole `crucible` package) used by `tests/v5/test_gate_{O,P,SP_10_0,SP_10_1,SP_10_2,SP_10_3,SP_10_3b,SP_2_1b,SP_17_1,SP_19_3}.py` |
| `civ/events.py` | typed schemas of the civilisation modules' own events (`links.*`, `goods.*`); civilisation worlds are typed (`params.typed_events`) and, by default, strict |
| `civ/history.py` | historical queries on a recorded run (state at a past time, entity timelines, per-day activity, past balances from postings, provenance back from an event) and `verify` (each kind checked against an independent source) |
| `civ/slices.py` | the seven Phase 3 vertical slices (SP-19.3), with `crucible.experience.ReferenceLearner` in place of CAM |
| `civ/redemo.py` | SP-17.1: world gates measured with their own code on civilisations (J/L creation experiment, B/C/K on an autonomous run, Gate D's whole battery, Gate G's adaptation experiment with 256 adaptive individuals), per-requirement manifest criteria; gates without a civilisation measurement fail with the reason |
| `civ/determinism.py` | exact checks under contract 1.7 (D-47): `first_divergence` (where two logs part and whether a recorded non-reproducible outcome or the world's logic is the origin) and `regenerate_continuation` (`World.regenerate` from a loaded snapshot); used by the long-run resume checks, SP-10.0 forks and the interruption test |

Additive changes outside `civ/`: `economy/industry.py` gains `draw_energy` (`make.energy`), `retail_sale`
(`make.retail_sold`) and `retail_close_day` (`make.retail_day`) so datacenters and the population use the 4.0
grid and retail through the industry module's own appliers (no cross-module writes).

## Long runs and interruptions

Usage limits stop processes without warning. `crucible.civ.longrun.run_horizon` writes a checkpoint at every snapshot
(every `snapshot_every` days, default 10): an interrupted run started again in the same directory resumes at the
last checkpoint in a new process (store cut back to the checkpoint, `World.load`, verified against the checkpoint's
hash and digest). `longrun.suite` keeps each (horizon, seed) run in `job_dir(...)` and skips completed ones, so the
Gate P suite can be restarted any number of times. Launch long suites from a frozen copy of a commit (so later edits
never mix into a running horizon), e.g. `git archive <commit> crucible/src | tar -x -C <dir>` and
`PYTHONPATH=<dir>/crucible/src CRUCIBLE_CIV_RUNS=... python -m crucible.civ.gates long`.

## Determinism (D-47, DEF-J-1)

Batched neural inference is not bit-reproducible, so two free runs of one spec and seed can part through one
recorded inference outcome (the shared space's vectors depend on batch composition and on the machine-wide cache).
The checks that count are the exact ones: a fresh process regenerates a recorded run from genesis from the store
alone (`World.regenerate`, SP-10.0 replications), a resumed copy re-executes the live run's recorded continuation
(Gate P.3), and a fork or an interrupted run may part from its reference only through a recorded outcome
(`first_divergence`). Free agreement is reported beside them.

## Chains: what counts as a confirmed stage

A chain's stages are subsystems. The root's subsystem is the root (a labelled scripted reference agent's action);
every later stage counts only when one of the measures standing for that subsystem (`gates.CHAIN_ARMS`, fixed
before a run) is confirmed by removing the root across seeds: sign-consistent in every seed, paired 95% interval
excluding zero, and larger than twice the placebo reshuffling noise. Measures read the consequence where it lands:
the metered compute the agent's own deployment served to others (`m_agent_compute`; the operator's own requests,
`cloud.usage_rel`, are left out), payments by population individuals verified by the ledger, the grid energy per
good unit of the machines a paper is about. Provider-wide or world-wide totals are also reported, but a small
agent's share of them sits below the reshuffling noise (at e77fef3 the provider's whole compute moved -4,103 cu
against 5,902 cu of noise). With several measures per subsystem the chance that one is confirmed by luck is higher
than for one measure; the per-measure verdicts are in the evidence notes so a reader can apply a stricter rule.

## How to run

    PYTHONPATH=src python -m pytest -q tests/test_v5_civ_core.py
    PYTHONPATH=src python -m crucible.civ.gates integration|structure
    PYTHONPATH=src python -m crucible.scale.slots --cpu 3 -- env CRUCIBLE_CIV_WORKERS=3 \
        python -m crucible.civ.gates effects        # or: chains
    PYTHONPATH=src python -m crucible.scale.slots --cpu 4 -- env CRUCIBLE_CIV_WORKERS=4 \
        python -m pytest -q tests/v5/test_gate_P.py

## Known limits (see docs/v5/workstreams/J.json for the measured state)

* Education (`learn`), employment (`labor`/`jobs`/`firms`), institutions (`inst`), marketplaces (`xchg`/`market`)
  and technology invention (`caps`/`pkg` without an agent) are composed but receive no population activity on this
  branch (INT.json), so Gate O's strict subsystem count (12 of 17 at 02c300f) and O.n1 fail honestly until the
  owning workstreams make them live; `crucible.civ.integration.subsystems` counts their population events with no
  civ change.
* Cross-process resume of civilisations with `pop_adaptive` is exact since kernel D-21 (on_load after the store is
  attached): tests/test_v5_civ_determinism.py (DEF-A-5 fixed).
* Gate P suites are memory-bound (~16 MB of state per simulated day; a 365-day run and its verification child need
  ~15 GB at the end): `suite(max_long=1)` runs at most one 365-day run at a time.
* Removing anything that records events reshuffles every later random draw (seq-keyed `Ctx.rng`); the placebo arms
  measure this and effects must exceed it.
