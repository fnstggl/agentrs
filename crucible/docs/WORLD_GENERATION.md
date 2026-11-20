# World generation

Code: `crucible/src/crucible/worldgen/generator.py`, `worlds/digital_civilization.py` (`civ`, `shocks`),
`worlds/engineering_lab.py`. `generate(config, seed) -> WorldSpec`; the spec (module list + configs + params +
seed) and the event log fully determine a world.

## Families: structural, not cosmetic

| family | split | what is causally different |
|---|---|---|
| `bazaar` | train | balanced: 4 needs, two modest incumbents each (tiers 0–1), ordinary compute; a preference drift toward quality on day 20 |
| `cheap_compute_closed_markets` | train | compute cheap and abundant (60–80¢/core-h, cheap hardware) but customers hard to reach (search 0.4 %/h, weak word of mouth, loyal) — distribution, not efficiency, is the bottleneck |
| `expensive_compute_open_distribution` | train | compute scarce and ~4.5× pricier per core-hour with a 2.4× work scale (algorithmic efficiency pays), customers easy to reach and fickle; a datacenter fire (−30 % capacity, ×1.6 price) on day 25 |
| `entrenched_incumbents` | val | incumbents at the best tiers, rich, priced high; loyal customers (high switching costs); an aggressive entrant on day 15 |
| `fragmented_networks` | val | mail takes 6 h and fails 15 % of the time, clustered social graph, little search, many small niches |
| `labor_rich_fragile_infra` | test | many cheap workers, few weak incumbents, unreliable providers and a reliability storm on day 10 |
| `novel_needs` | test | need formats never present in train families (`route.tsp`, `match.stable`); a third (`forecast.series`) emerges as a new market on day 12 |
| `engineering_lab` / `engineering_lab_heldout` | train / test | not an economy: network routing laboratory (see below) |
| `tiny` | dev | small world for tests |

Within a family the seed draws the needs (where the family samples them), incumbents' tiers, prices, cash and
policies, providers' placement, the labour pool (skills, wages), lenders, bounties, the consumer population
(rates, willingness to pay, preferences, loyalty, social graph) and every stochastic process during the world's
life. Generation is reproducible: the same (config, seed) gives the same spec hash and, under the deterministic
cost model, the same trajectory.

## Mechanism plugins

Any `@register_module` module can be added to a family's module list (`extra_modules`), and families can drop
modules (`drop_modules`) — e.g. a world without advertising or without the forum, to measure the causal value of
an institution with paired counterfactual worlds. Shocks are a config list on the `shocks` module.

## Held-out generators

The `test` families withhold entire mechanism settings and problem formats, not only seeds. Train families never
contain `route.tsp`, `match.stable` or `forecast.series`, unreliable infrastructure storms, or abundant cheap
labour. Use `crucible.evaluation.harness.locked_test_goals(ack=..., purpose=...)` to access them for final
evaluation (access-logged).
