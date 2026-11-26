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
| `civ/couplings.py` | module `civlinks` (operator host `admin`): coupling records, removal/restoration/placebo tools, the mechanisms `science_process`, `compute_energy`, `grid_curtailment`, `org_coordination` |
| `civ/popapi.py` | the population API of the one-world contract in one place (contract export when present, `population.core` otherwise) |
| `civ/chains.py` | provenance graph over the event log (transactions, schedules, entity timelines, column rows, explicit refs), `trace`, generic chain mining and labelling |
| `civ/counterfactual.py` | fork, intervene, rerun; two placebo arms measure the reshuffling of seq-keyed randomness; multi-seed paired statistics |
| `civ/experiments.py` | the effect battery (Part 10.2 categories 1, 2, 3, 4, 5, 7, 8) and the chain experiments (scripted reference agents) |
| `civ/reference.py` | scripted reference agents (labelled `scripted`): a web-shop builder and a researcher |
| `civ/longrun.py` | long-horizon harness: daily invariants and activity, accounting audits, snapshots, segment replay, cross-process resume, wall/RSS/store growth; `suite` for Gate P |
| `civ/accounting.py` | event accounting (events vs state) and economic/social activity classification |
| `civ/integration.py` | Gate O measurements on data flow; data-level duplicate scan; identity and ledger checks; structural fingerprint of built worlds |
| `civ/modes.py` | the Mode C provider and its check (determinism, replay, and integration measured, not assumed) |
| `civ/experience.py` | `tier3:<profile>` entry point, `observation_only` by default |
| `civ/gates.py` | the gate measurements (cached by code digest) used by `tests/v5/test_gate_{O,P,SP_10_1,SP_10_3}.py` |

Additive changes outside `civ/`: `economy/industry.py` gains `draw_energy` (`make.energy`), `retail_sale`
(`make.retail_sold`) and `retail_close_day` (`make.retail_day`) so datacenters and the population use the 4.0
grid and retail through the industry module's own appliers (no cross-module writes).

## How to run

    PYTHONPATH=src python -m pytest -q tests/test_v5_civ_core.py
    PYTHONPATH=src python -m crucible.civ.gates integration|structure
    PYTHONPATH=src python -m crucible.scale.slots --cpu 3 -- env CRUCIBLE_CIV_WORKERS=3 \
        python -m crucible.civ.gates effects        # or: chains
    PYTHONPATH=src python -m crucible.scale.slots --cpu 4 -- env CRUCIBLE_CIV_WORKERS=4 \
        python -m pytest -q tests/v5/test_gate_P.py

## Known limits (see docs/v5/workstreams/J.json for the measured state)

* Until CORE-POP/ECON merge, the v5 social, commerce, innovation, labour and community modules are present but
  receive no population activity, so Gate O's strict subsystem count and O.n1 fail honestly; `feed` is not
  composable yet (population hook not exported).
* Cross-process resume fails while `pop_adaptive` rebuilds from blobs in `on_load` before the store is attached
  (audit K1, workstream A).
* Removing anything that records events reshuffles every later random draw (seq-keyed `Ctx.rng`); the placebo arms
  measure this and effects must exceed it.
