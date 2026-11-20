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

## The engineering laboratory (second world)

### Engineering lab worlds (`engineering_lab`, `engineering_lab_heldout`)

Code: `crucible/src/crucible/worlds/engineering_lab.py` (module `lab`, host `lab`). Families: end of
`crucible/src/crucible/worldgen/generator.py`. Tests: `crucible/tests/test_engineering_lab.py`.

This second world has no economy beyond fees and penalties. It shows that the kernel is not hard-wired to markets.
The agent improves a real routing algorithm for a production network. **Executed measurement** decides how good it
is: a deterministic flow model driven by a hidden traffic process. No LLM judge or rubric is involved.

### Mechanics

**Genesis.** The lab sets up the following, seeded through `world.rng_for("lab-genesis")` and emitted as ordinary
events:

* **The production network** (12-40 nodes). It has undirected links, each with an integer capacity and a base
  latency in ms. Three generators exist:
  * `grid`: a near-square lattice plus a few low-latency express links that attract shortest paths;
  * `scale_free`: preferential attachment with `m = 2`, and capacity proportional to `sqrt(deg_u * deg_v)`;
  * `waxman`: geometric k-nearest neighbours.

  A connectivity repair step guarantees that the network is connected.
* **A hidden traffic regime.** It is a fixed set of directed demand pairs `src>dst` with base rates, a diurnal
  profile, hotspot multipliers, a volume scale and per-snapshot lognormal noise.
* **Calibration.** The volume scale is set so that the baseline's worst noise-free hour reaches `peak_util`
  (for example 1.05, slightly over capacity). The SLA is `sla_factor` times the baseline's mean delay over that
  calibrated day. The SLA value is public. How it was derived is not.
* **The operator.** The operator organisation (`o:netops`) has an NPC lead and a funded bank account.
* **The controller repository.** A public, open-licensed repository `netops/controllers` holds `baseline.py`,
  a lowest-latency single-path Dijkstra in real Python. This baseline is deployed in production.
* **A public handbook doc**, if the `docs` module is present.

**Controllers** are repository functions with this signature:

`route(topology, demands) -> {"routes": {demand_id: [[path, fraction], ...]}}`

* `topology` = `{"nodes", "links": [{"id", "u", "v", "capacity", "latency"}]}`.
* `demands` = `[{"id": "src>dst", "src", "dst", "volume"}]`.

How a controller runs:

* Agent controllers run in the sandbox with `world_access=False`. The code service's sandbox path is used:
  `Code.can_read`, `files_with_deps`, `sandbox_pool`, and `bridge.run_in_ctx` for commands.
* A lab harness file (`_crucible_lab_harness.py`) is added next to the repository files. It calls `route` once per
  scenario in a single sandbox process, and seeds `random` (and numpy, if imported) with the scenario seed.
* Exceptions raised during routing are reported as type and location only. Their messages could carry traffic
  data out.
* The lab-owned baseline is trusted code and runs in-process.

**Physics** (`evaluate`, deterministic):

* **Link load** is the sum of `volume × fraction` over every path that uses the link. Capacity is shared by both
  directions.
* **Link delay** is `latency + queue_scale · load/(capacity − load)`, which equals `ρ/(1−ρ)`. `ρ` is capped at
  0.99, so a full queue adds 99 ms.
* **Overload.** Above capacity, a link drops the fraction `(load − capacity)/load` of everything that crosses it.
  A path delivers `Π(1 − drop)`. The model does not solve for a fixed point, so it does not reduce downstream
  load after drops.
* **Validation is strict.** The output must be `{"routes": {...}}`. Each demand must have 1-3 distinct simple paths
  over existing links from src to dst, with positive fractions that sum to 1 ± 1e-6. Any violation drops all
  of that demand. Error messages never echo values the agent supplied.
* **Metrics.** Each run yields:
  * `mean_delay`, the effective mean delay, volume-weighted. Delivered traffic counts at its path delay; dropped
    traffic counts at `drop_penalty_ms` = 250, a timeout.
  * `delivered_delay`;
  * `max_util`;
  * `dropped_fraction`;
  * the number of invalid demands.

  Because dropped traffic is penalised, a controller cannot look fast by losing traffic.

**Processes:**

| process | when | what it does |
|---|---|---|
| `lab.hour` | every hour (prio 40) | Samples the hour's demand snapshot from the hidden process (a seed drawn from `ctx.rng`, which is stored), runs the *deployed* controller and evaluates it. It then emits `lab.hour` (internal: exact metrics, seed, regime id, snapshot digest), publishes the noisy dashboard row for hour `h − dashboard_delay_h` as a public `lab.dashboard`, and charges an SLA penalty when the exact mean delay exceeds the SLA. The penalty goes from the operator account to `acct_external` and is capped by the available balance. |
| `lab.daily` | 00:00 (prio 45) | Emits a new hidden regime. Pair rates drift by a lognormal (`drift_sigma`) and peaks wander. Then any shocks apply: scheduled ones from config, plus a random one with probability `shock_p`. Shock kinds are `hotspot` (a new hot pair), `hotspot_end`, `surge` (volume ×1.08-1.2) and `shift` (peaks move up to ±4 h). |
| `lab.maintenance` | `maintenance_hour` (prio 60) | Deploys the pending change, pinned to its commit. |

### Endpoints

| tool | access | notes |
|---|---|---|
| `lab.topology()` | public | Nodes and links. The network is observable. |
| `lab.status()` | public | Shows the operator, the deployed controller, the next maintenance window, the SLA, prices, total SLA penalties and `you_operate`. Only principals with `lab.operate` see the details of a pending change; everyone else sees only whether one exists. |
| `lab.measure(window_h)` | public | Dashboard rows. They are **delayed** (`dashboard_delay_h`, default 2 h), **sampled** (mean delay and drops are measured on a random probe subset of demands, `probe_fraction`) and **noisy** (lognormal noise `measure_sigma` on delay, utilisation and volume). Also returns a summary and the count of SLA-breach hours. |
| `lab.experiment(repo, target, scenarios \| hours, account, ref)` | read access to the repo | Runs the candidate on *k* fresh snapshots of the **current** hidden process, never on production's. The hours of day are random or given. Returns candidate and baseline metrics on the same snapshots, the paired improvement, per-scenario rows, the first validation errors and the CPU used. It costs `experiment_fee_cents + k · scenario_fee_cents`, paid from `account` (`bank.spend` authorised) to the operator, plus sandbox compute via `Code.charge_compute` after the free daily allowance. |
| `lab.propose_change(repo, target, ref)` | `lab.operate` on the operator org | Pins `ref` (default: head) and schedules deployment at the next maintenance window. A new proposal replaces any pending one. Production runs the controller as the proposer. |

The `lab.operate` capability comes from three sources:

* owners and admins of the operator organisation hold it;
* the configurable **engineer seat** (`engineer_seat`, `"eng"` in both families): when that handle is admitted via
  `admin.join`, it becomes a member of the operator organisation and the lead issues it an ordinary `id.grant`;
* in worlds with contracts, any agreement that makes the operator issue the grant.

### What is hidden, delayed or noisy

* **Hidden:** the regime parameters (`base_rates`, `hotspots`, `diurnal`, `volume_scale`, `noise_sigma`), the
  per-hour snapshot seeds, and the exact production metrics. They appear only in `internal` events (`lab.regime`,
  `lab.hour`, `lab.experiment_samples`, `ext`) and in state that no endpoint returns.
  * The test scans every event visible to the agents for those keys and for the numeric values of the parameters
    and seeds.
  * Agents never see controller inputs. Experiment results return no stdout and no exception messages.
  * The operator-side `WorldSpec` contains the generator config, as it does for every world.
* **Delayed and noisy:** the production dashboard.
* **Exact but private:** experiment results. They are exact on fresh samples, so they carry sampling variance.
* **Not stationary:** daily drift and shocks change the hidden process. A controller tuned to yesterday's hotspot
  can stop working.

### Metrics (`@metric`, evaluator-side, exact)

* `lab_mean_delay(window_h)`: the volume-weighted effective mean delay of production over the last `window_h`
  hours.
* `lab_dropped_fraction(window_h)`: the share of production traffic dropped, through overload or invalid routes.
* `lab_improvement_vs_baseline(window_h)`: `(B − D)/B`, where `D` is the recorded effective delay-volume of the
  deployed controller over the window. `B` comes from re-running the baseline on the **same** demand snapshots,
  re-sampled from each hour's stored seed and regime (the stored snapshot digest is checked), so the comparison is
  paired. While the baseline is deployed, the value is exactly 0.

### Families

| family | split | network | traffic | shocks |
|---|---|---|---|---|
| `engineering_lab` | train | `grid` with 14-20 nodes and 2-4 express links | gravity model (lognormal node masses, 30-45 pairs), one sine diurnal peak (13:00-20:00), 1-2 hotspots ×2.5-4 | drift σ 0.06, random shock p 0.1 |
| `engineering_lab_heldout` | test | `scale_free` with 28-40 nodes | datacenter **hub** structure (2-3 hubs, asymmetric down/up and replication flows, 60-80 pairs), **bimodal** day (morning and evening peaks), 2 flash-crowd hotspots ×3-6 | drift σ 0.1, shock p 0.2 |

Both families use their own module list: `identity, bank, admin, code, search, forum, mail, docs, lab`. The
build result carries it as `"module_list"`. Until `generate()` honours that key, use
`crucible.worlds.engineering_lab.lab_spec(family, seed, overrides)`, which applies the same rules. Overrides
accepted: `n_nodes`, `n_demands`, `shocks`, `peak_util`, `engineer_seat`, `operator`, plus the generic
`module_config` and `params`.

### Reference controllers and measured effect

`BASELINE_SOURCE`, `GREEDY_SOURCE` and `INVALID_SOURCE` are module constants:

* **Greedy** is congestion-aware. It places demands largest first, in three chunks each, on the path with the
  lowest marginal cost (queueing increase plus the overload penalty).
* **Invalid** assumes a full mesh and routes every demand over the direct link.

Measured in the test world (16-node grid, 30 demands, SLA 34.1 ms):

| period | controller | mean delay (ms) | improvement vs baseline | dropped |
|---|---|---|---|---|
| day 0 | baseline | 27.5 | 0 | 0.8% |
| 23 h | greedy | 8.6 | **+0.69** | 0% |
| 23 h | invalid | 156 | **−4.59** | 62% |

SLA penalties paid by the operator: $667 for the baseline's day, $0 under greedy, and $6,756 in the 23
hours of the invalid controller. The offline experiment (8 snapshots) predicted +0.79 for greedy and −2.2 for
the invalid controller.
In family worlds with seed 11, greedy reaches +0.79 in production (train) and +0.63 (held-out scale-free/hub);
offline experiments predicted +0.79 and +0.67.

### Determinism and limits

* **Replay** of the event log reproduces the state digest.
* **Regeneration from commands** reproduces `last_hash`. Two mechanisms make this work:
  * Experiment outcomes go through the bridge, so measured CPU is recorded and injected.
  * The kernel re-executes scheduled processes during regeneration. Production therefore records only the
    deterministic part of the sandbox outcome: routes, ok, error type. It also seeds controller randomness.
* As a consequence, **production compute is not billed**. A controller whose output depends on wall-clock time or
  timing would make regeneration diverge.
* The flow model is a single-pass, static, per-hour model: no TCP dynamics and no fixed point after drops. Link
  capacity is shared by both directions.
* Demand membership is fixed per regime. Only shocks add pairs.
