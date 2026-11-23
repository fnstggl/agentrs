# Crucible 4.0 "OpenWorld" architecture

Crucible 4.0 is built on the unchanged Crucible 3.0 kernel (branch `feat/crucible-internet`, commit `11e5613`,
tag `crucible-3.0-final`). Everything below is additive: legacy families, tests and recorded results are untouched,
and every new piece of world mechanics is an ordinary event-sourced kernel module reached through ordinary tools.

The component contracts in this file are binding for the code: a module that needs something from another module
uses exactly the interface written here.

## 1. What 4.0 adds, and where it lives

| layer | package (under `src/crucible/`) | what it is |
|---|---|---|
| usefulness substrate | `usefulness/` | procedurally generated categories of digital need, each with a public format, a hidden evaluator and reference solvers; a kernel module (`use`) for discovery and the usefulness ledger |
| hidden mechanisms | `mechanisms/` | three independently authored engines (structural causal models, an agent-based market, executable technical systems) behind one contract; a kernel module (`rig`) that hosts a mechanism as a facility; a kernel-free simulator that is bit-identical; private evaluators; validity checks |
| population | `economy/behavior.py`, `economy/consumers.py` | structurally different consumer behaviour models, heterogeneous segments, latent needs and derived demand |
| market dynamics | `actors/entry.py`, `evaluation/invariants.py`, `evaluation/longrun.py` | entry into vacated or newly created markets, a continuous invariant monitor, long-run liveness measures |
| research platform | `research/` | architecture-neutral learner interface, batched environments, learning curves on three timescales, the parametric-internalisation and transfer protocols, cost accounting, statistics, sealed evaluation sets, baseline learners |
| self-play | `selfplay/` | multi-seat arenas with independent learners, the six interaction modes, leagues with archived policies, cross-play, collusion audit |
| interface | `core/discovery.py`, `research/diagnostics.py` | searchable tool documentation, dry-run validation, API versions; automatic classification of failures |
| assurance | `redteam/`, `realworld/`, `evaluation/fidelity.py` | reward-hacking and integrity probes, the authorisation-gated real-world bridge (off by default), the fidelity register |
| scale | `scale/` | process-parallel worlds, asynchronous rollout workers, benchmarks |

## 2. Contracts

### 2.1 Mechanism engines (`mechanisms/base.py`)

`Engine` subclasses implement `generate(rng, params) -> config` (JSON), `levers(seat)`, `gauges(seat)`,
`brief(seat)`, `init_state(rng) -> {name: ndarray}`, `step(state, settings, rng, t) -> StepResult` and
`truth()`. `StepResult` carries the next state, per-seat readings, per-seat revenue and cost in cents, the
per-seat objective quantity, and private diagnostics. Purity, opacity and conservation are required and tested.
Engines register themselves with `@register_engine`; families with `@mech_family(name, engine, split, transfer,
doc, motifs, seats)`. `make_mechanism(family, seed, overrides)` returns `{"engine", "family", "seed", "config"}`.

Randomness for tick `t` of facility `rig_id` is `tick_rng(world_seed, rig_id, t)` everywhere. The initial state
uses `tick_rng(world_seed, rig_id, "init")`.

### 2.2 The facility module (`mechanisms/module.py`, kernel module `rig`, host `rig`)

Config: `{"rigs": [{"id": str, "mech": <make_mechanism output>, "tick_s": int, "income_cents_per_tick": int,
"seats": [{"holder": handle or null}], "lease_cents": int}]}`.

* State is event-sourced. Each tick is one scheduled process run: it calls `engine.step`, emits `rig.tick`
  (internal), pays revenue from the facility's demand account (funded each tick from `acct_external`, never
  overdrawn; a shortfall is recorded, not paid), charges cost to the seat holder, and emits one `rig.report`
  per seat, visible to the holder only.
* Tools: `rig.list`, `rig.take`, `rig.release`, `rig.spec`, `rig.set`, `rig.read`, `rig.history`,
  `rig.hypothesize`. `rig.set` takes effect at the next tick; unknown lever names are an `InvalidAction` with
  the valid names in the message. `rig.hypothesize` files structured claims that only the evaluator scores.
* Metrics (registered with `@metric`): `rig_net` (revenue minus cost in a window), `rig_score` (objective
  quantity in a window), `rig_ticks`.
* World families: every mechanism family `F` is a world family `mech_F` (modules `identity`, `bank`, `admin`,
  `rig`), and `mech_F_tools` adds `code`, `cloud`, `wf` so a participant can deploy a controller program that
  operates the facility through the same tools.

### 2.3 The kernel-free simulator (`mechanisms/sim.py`)

`MechSim(mech, world_seed, rig_id="r1")` exposes `reset()`, `step(settings_per_seat) -> StepResult`, and
`run(policy, ticks)`. It reproduces the readings, revenue, cost and score of the kernel-hosted facility exactly
(tested tick for tick), and is what batched learning uses.

### 2.4 Private evaluation (`mechanisms/evaluator.py`)

Works on any engine through `step` with common random numbers: `effect_matrix` (true average effect of every
lever on every gauge and on the outcome, with lags), `score_claims` (sign, lag and magnitude accuracy of filed
hypotheses, Brier score when probabilities are given), `discriminate` (log score of a posterior over candidate
mechanisms), `oracle` (best constant and best adaptive settings found by search, used only to normalise regret),
`information_efficiency` (identification per tick of experience), `prediction_error` (held-out settings).

### 2.5 Population (`economy/consumers.py` configuration, all defaults reproduce 3.0 exactly)

* `behavior`: `{"model": name, "params": {...}}`, or `segments`: `[{"share": float, "model": name, "params":
  {...}, "name": str}]`. Models are registered in `economy/behavior.py` (`BEHAVIOR_MODELS`); `legacy` is the 3.0
  rule set, byte for byte.
* `latent`: `{need: base_active_fraction}` - consumers of these needs start mostly without the need.
* `derived`: `[{"src": need, "dst": need, "weight": float, "scale": float, "lag_days": int}]` - the active
  fraction of `dst` relaxes daily toward `clip(base + sum_src weight * (1 - exp(-ok_src_per_day / scale)), 0, 1)`
  at `activation_rate` per day, where `ok_src_per_day` is the verified successful volume served in `src`.

### 2.6 Usefulness categories (`usefulness/`)

A `NeedCategory` has a public format and documentation, a seeded generator of request instances, a hidden
evaluator, reference solvers at several quality tiers, a resource scale, a customer population description and
externalities (derived-demand edges, resource use). `CategoryFamily.instantiate(seed)` draws a category whose
hidden rule differs per world. Categories are materialised as `economy.needs.Need` objects under deterministic
names `gen.<family>.<hash>` and registered when the `use` module is installed, so every existing mechanism
(consumers, firms, bounties, goals, the hidden test battery) works with them unchanged. World families that use
them are named `ow_*` and live in `worldgen/ow_families.py`.

### 2.7 Learners and environments (`research/`)

`research.learner.Learner`: `reset_context(batch)`, `act(obs, explore) -> actions`, `observe(transition)`,
`update() -> counters`, `save(path)`, `load(path)`, `freeze(bool)`, `usage()`. Numeric environments
(`research.envs.MechVecEnv`) pad every mechanism to fixed observation and action sizes with masks; text
environments are the kernel world through the existing `agents.base.Policy` interface. `research.bridge` runs a
numeric learner inside a kernel-hosted world through `rig.*` tool calls, which is how equal permissions across
architectures are shown.

## 3. Rules for every new module

1. Handlers read state and emit events; only appliers write. Appliers are deterministic functions of the event.
2. No global randomness: `ctx.rng(...)`, `world.rng_for(...)` or `tick_rng(...)`.
3. Money moves only through `bank.move`; every account has a defined source.
4. Nothing an agent can call returns hidden parameters, evaluator code paths or other participants' private state.
5. Every module passes the admission harness (`crucible.worldgen.admission`) and has invariant checks.

## 4. Documentation map

| subject | document |
|---|---|
| results, status by category, limitations | [OPENWORLD_REPORT.md](OPENWORLD_REPORT.md) |
| setup, exact commands, determinism guarantees, checkpoints | [REPRODUCE_OPENWORLD.md](REPRODUCE_OPENWORLD.md) |
| acceptance tests A-P | [ACCEPTANCE.md](ACCEPTANCE.md) |
| world kernel, events, replay, snapshots | [WORLD_KERNEL.md](WORLD_KERNEL.md) |
| internet fabric, application runtime | [SYNTHETIC_INTERNET.md](SYNTHETIC_INTERNET.md), [INTERNET_ARCHITECTURE.md](INTERNET_ARCHITECTURE.md) |
| economic state, participants | [ECONOMY.md](ECONOMY.md), [ACTOR_SYSTEM.md](ACTOR_SYSTEM.md), `economy/behavior.py`, `actors/entry.py` |
| mechanism generation | section 2.1 above, `mechanisms/base.py` and the docstrings of `scm.py`, `abm.py`, `techsys.py` |
| task generation and evaluation | [EVALUATION.md](EVALUATION.md), `research/tasks.py`, `research/protocols.py`, `research/sealed.py` |
| self-play | [SELF_PLAY.md](SELF_PLAY.md) (3.0 leagues), `selfplay/__init__.py` |
| security and red team | [SECURITY.md](SECURITY.md), `redteam/`, `experiments/ow_redteam.json` |
| fidelity | [WORLD_FIDELITY.md](WORLD_FIDELITY.md), `evaluation/fidelity.py`, `experiments/ow_fidelity.json` |
| performance | [SCALING.md](SCALING.md), `benchmarks/openworld.json` |
