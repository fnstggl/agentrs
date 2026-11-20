# Crucible architecture (as implemented)

The original research brief (branch `feat/crucible-design`) is preserved verbatim in
[docs/DESIGN_BRIEF.md](docs/DESIGN_BRIEF.md). This document describes what was built, how the parts fit, and
where the implementation revised the brief. Status per component: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

```text
        learners & controls (any architecture)                         researchers
  frozen LLM | CAM 2.0+LLM | sl superlearner | torch RL | scripted | human      CLI · dashboard · trace
        │            │              │              │         │       │                 │
        └────────────┴── agent API ─┴──────────────┴─────────┘   web browser          │
   Python WorldServer · Gymnasium · PettingZoo · OpenEnv HTTP/WS · MCP (stdio/HTTP) · Chromium pages
                                         │  (every path calls the same endpoint handlers)
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ WORLD KERNEL  core/kernel.py   commands & scheduled processes → events → deterministic appliers   │
 │   journaled transactions · hash-chained log · discrete-event scheduler · seeded RNG streams       │
 │   principals · dynamic tool registry & discovery · visibility · snapshots/forks · replay/regen    │
 ├───────────────────────────────┬────────────────────────────────┬─────────────────────────────────┤
 │ SYNTHETIC INTERNET services/  │ ECONOMY economy/  actors/      │ OTHER WORLDS worlds/             │
 │ id bank mail forum search docs│ needs (verifiable work)        │ digital_civilization (civ,shocks)│
 │ code issues cal support ads   │ consumers (vectorised demand)  │ engineering_lab (network physics)│
 │ market cloud biz contracts    │ NPC firms · workers · lenders  │                                  │
 │ jobs research goals admin     │                                │                                  │
 ├───────────────────────────────┴────────────────────────────────┴─────────────────────────────────┤
 │ SANDBOX sandbox/  uid nobody · Landlock · seccomp · rlimits · brokered world RPC (recorded)       │
 ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ STORAGE storage/  memory · SQLite (WAL) · PostgreSQL — append-only events, snapshots, lineage    │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
   GOALS goals/ (contracts, metrics, curriculum, self-goals) · EVALUATION evaluation/ (harness, audit,
   partitions, counterfactuals, emergence) · WORLDGEN worldgen/ · TRAINING training/ (trajectories, league)
```

## 1. Package layout

`crucible/src/crucible/`:

| package | contents |
|---|---|
| `core/` | kernel (World, Ctx, Module, events, scheduler, replay/regeneration), state store, schema validation, canonical hashing, errors |
| `storage/` | event/snapshot stores (memory, SQLite, PostgreSQL) |
| `services/` | the synthetic internet (17 service modules + operator tools) |
| `economy/` | needs (verifiable workloads), consumers (demand) |
| `actors/` | NPC society policies |
| `worlds/` | world-content modules: digital civilization genesis + shocks; engineering lab |
| `worldgen/` | procedural families with train/val/test splits |
| `sandbox/` | isolated execution of agent code |
| `goals/` | metrics/predicates, goal contracts, reward contracts, autocurriculum, self-goals |
| `evaluation/` | harness, partitions + locked test, integrity audit, counterfactuals, emergence metrics |
| `agents/` | policy interface, runner, baselines and scripted controls, LLM and RL agents |
| `integrations/` | AGENTRS CAM 2.0 and superlearner adapters |
| `api/` | Python server, Gymnasium, PettingZoo, OpenEnv-compatible HTTP/WS, MCP |
| `web/` | browser mode (FastAPI pages, Playwright sessions) |
| `analytics/`, `dashboard/`, `training/`, `bench/`, `cli.py` | researcher tooling |

Crucible is a separate Python project (`crucible/pyproject.toml`) with no dependency on AGENTRS model code;
AGENTRS reaches it through `integrations/` and `src/agentrs/cam2/envs/crucible_env.py`.

## 2. Key decisions (alternatives considered in docs/RELATED_WORK.md)

1. **Event sourcing with in-memory authoritative state**, durable hash-chained event log and snapshots, instead
   of SQL-resident state. Exact replay (`World.replay`) and *regeneration* (`World.regenerate`: re-run every
   recorded command and clock advance through the handlers; identical event hashes) are tested, including worlds
   with sandboxed agent code, consumer demand and random agents.
2. **Outcomes the kernel cannot recompute are recorded once** (program outputs, measured CPU), keyed by the
   transaction's first sequence number. Two cost models: `measured` (real CPU time; realistic, noisy) and
   `deterministic` (executed Python lines; bit-identical forks for counterfactuals).
3. **No privileged actors.** NPCs, scripted controls and learning agents use the same tools. Operator actions are
   recorded `admin.*` commands that in-world principals cannot call.
4. **Software quality is executable truth.** Needs have verifiers with hidden ground truth; demand executes real
   requests on deployed code; compute cost is measured from that execution.
5. **Populations as arrays.** Consumers are numpy arrays in the state: persistent, forkable, cheap.
6. **Sandbox without namespaces** (unavailable in this container): unprivileged uid + Landlock + seccomp +
   rlimits; world access only through the brokered RPC channel.
7. **Goals are outcome contracts over independent metrics**; rewards are potential-based on verified progress.

## 3. Revisions relative to the brief

* The brief anticipated PostgreSQL as the primary state store; the implementation keeps state in memory
  (event-sourced) and uses PostgreSQL/SQLite as the durable event and snapshot store — this made exact replay,
  cheap forks and high RL throughput possible (SCALING.md).
* WebWorld-style LLM page prediction is rejected as an authority; pages are rendered from authoritative state.
* "Thousands of websites" is replaced by 17 deeply integrated services plus unlimited agent-deployed services.
* OpenEnv is used as a protocol (compatible routes), not as a dependency.

## 4. Lifetimes

World lifetime (until stopped or forked) ≠ agent lifetime (the learner's own checkpoints and memory) ≠ trial
(an evaluation exposure) ≠ episode (a training segmentation; in `reset_mode="continue"` the world persists) ≠
rollout. Agents can leave and return; the world keeps running (AGENT_API.md, RL_INTEGRATION.md).
