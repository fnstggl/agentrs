# Component L: the architecture-independent learner interface (`crucible.experience`)

Owner: workstream L. Contract: INTERFACE_CONTRACTS section 11. How a learner connects: [LEARNER_GUIDE.md](../LEARNER_GUIDE.md).

## What exists

| file | what it is |
|---|---|
| `src/crucible/experience/observation.py` | the `Observation` record (15 fields), `validate_observation`, `validate_action`, `validate_result`, `kinds_present` (the 13 kinds of MASTER_SPEC 11.1) |
| `src/crucible/experience/modes.py` | the three reward modes, `RewardGate`, the verifier/operator keys that must never reach a learner |
| `src/crucible/experience/env.py` | the `ExperienceEnv` protocol and the shared base class |
| `src/crucible/experience/tier1.py` | `MechExperienceEnv` (4.0 `MechVecEnv` / `SelfPlayEnv` with the kernel facility's own `rig.*` / `bank.accounts` tools and principals) and `RapidExperienceEnv` (v5 `crucible.rapid.make`, contract section 8) |
| `src/crucible/experience/kernel.py` | `KernelExperienceEnv`: any `WorldSpec`, several agents, lossless events with `cause` provenance, artifacts, entities, resources, public analytics, `missing`, objectives and verification, semantic representations through `crucible.semantic.encode` when present |
| `src/crucible/experience/features.py` | flattening summaries into exact numbers, text, entities, artifact references, the 13 tool categories |
| `src/crucible/experience/registry.py` | `make(tier, name, seed, reward_mode, **kw)`, `TierUnavailable`, `domains()`, `tiers_available()` |
| `src/crucible/experience/experiments.py` (+ `modules.py`) | kernel module `exp`: self-generated experiments over any read-only measure, evidence only (passes `worldgen.admission.admit`) |
| `src/crucible/experience/store.py` | `ExperienceStore` (append-only shards, per-record SHA-256, hash-chained manifest, content-addressed blobs, crash recovery) and `make_record` (`crucible-exp/1`) |
| `src/crucible/experience/reference.py` | `ReferenceLearner`: recurrent neural learner over hashed structured-observation features, schema-driven argument filling (ids and name tokens it was shown, by argument name: `ArgumentMemory`), PPO-clip + value + entropy + world-model loss, exact checkpoints |
| `src/crucible/experience/statistical.py` | `ToolValueLearner`: statistical reference learner - per-tool value of the observed change of its own signal (default its money) against the median change per step, softmax at max(temperature, MAD), optional multi-step credit, name-independent argument-level arms (`arg_levels`); used by SP-11.6c in every tier (D-68 addendum and amendment) |
| `src/crucible/experience/loop.py` | `run_learner`: the driver loop (schema checks, world-change counting, storage, updates, verification for the evaluator) |
| `src/crucible/experience/parallel.py` | `VecExperience` (in-process) and `FarmRun` + `make_world` / `make_policy` hooks for the unchanged `crucible.scale.RolloutFarm` |
| `src/crucible/experience/scripted.py` | scripted reference agents acting only through the interface (company, service deployment, discovery), `Recording` |
| `src/crucible/experience/leakage.py` | the explicit reward-leakage battery (11 tests, 9 canaries) and the 4.0 legacy-equivalence check |
| `src/crucible/experience/probes.py` | information-flow probes (12 kinds) over ten surfaces incl. a Tier 3 civilisation, positive control on the 4.0 contract |
| `src/crucible/experience/probes_tiers.py` | per-tier probes on five leak channels (observation, info, tool responses, event metadata, free text), the nested-call probe (flows, deferred jobs, triggers), the identity probe (OO.16), the Tier 2 domain-verdict probe (a verdict planted in every domain's `Domain.judge`, plus the real verdict's distinctive numbers within a bounded reference drive) |
| `src/crucible/experience/selfexp.py` | self-generated experiments in the Tier 1 adapters (`exp.register/result/mine` over the agent's own observation numbers) |
| `src/crucible/experience/labels.py` | `LabelChannel`: verified outcome labels through a separate, explicitly authorised evaluator-side channel (OO.12) |
| `src/crucible/experience/__main__.py` | `python -m crucible.experience run ...` (reproducible runs) and `tiers` |
| `examples/experience_external_adapter.py` | an external, CAM-Unified-shaped adapter using only the public interface |

Changes outside the package (all keep the 4.0 behaviour as the default, byte for byte; checked against the 4.0 source
at `c07098b`): `reward_mode=` in `research/envs.py` (`MechVecEnv`, plus `step_values` for exact tool-chosen
settings), `research/bridge.py` (`KernelMechEnv`), `selfplay/arena.py` (`SelfPlayEnv` with `step_values`,
`KernelSelfPlayEnv`), `api/gym_env.py` (`CrucibleEnv`, plus `verify()`), `api/pettingzoo_env.py`
(`CrucibleParallelEnv`, plus `verify()`), `scale/rollout.py` (`WorldEnv`, plus `verify()`) and
`integrations/agentrs_cam2.py` (the 4.0 CAM 2.0 adapter: no `progress`/`score`/`eval` in observation-only infos).

## How to use it

```python
from crucible.experience import make, run_learner, ExperienceStore
from crucible.experience.reference import ReferenceLearner
env = make(1, "mech:scm_additive", 0, "observation_only")          # or make(0, "tiny", 0, ..., agents=[...])
L = ReferenceLearner(seed=0, reward_mode=env.reward_mode)
summary = run_learner(env, L, 300, store=ExperienceStore("/tmp/exp"), update_every=32)
L.save("/tmp/ref.pt")
```

Gate tests: `tests/v5/test_gate_R.py`, `test_gate_S.py`, `test_gate_SP_11_1.py` ... `test_gate_SP_11_6.py`; unit
and integration tests: `tests/test_v5_experience_core.py`. Run with `CRUCIBLE_V5_AUTHOR=ws-L PYTHONPATH=src python
-m pytest tests/v5/test_gate_R.py` etc.; `test_gate_S.py::test_S7_parallel_environments` starts four worker
processes and must run through `crucible.scale.slots --cpu 4`.

## Measured (this branch, 2026-11-25, shared machine, one CPU thread unless stated)

* Leakage battery: 11 of 11 explicit tests pass; every reward path checked (MechVecEnv, KernelMechEnv, SelfPlayEnv,
  KernelSelfPlayEnv, Gymnasium, PettingZoo, WorldEnv, Tier 1 and kernel experience envs and the numeric view):
  0 nonzero reward steps in observation-only mode over 402 steps whose legacy twins had 171 nonzero rewards; 0
  privileged keys; 9 canaries planted (evaluator runtime, recommender-private slot, internal event, another
  participant's private mail, spec evaluator parameter, goal evaluator parameter, task private reference numbers,
  hidden mechanism configuration, internal rig filing), 0 hits; observations byte-identical across reward modes and
  across evaluator-only data.
* 4.0 equivalence: 11 of 11 legacy runs byte-identical to the `c07098b` implementation (MechVecEnv, its explicit
  legacy mode, KernelMechEnv, SelfPlayEnv, KernelSelfPlayEnv, hill_climb / probe_exploit / ppo_gru learners,
  Gymnasium, PettingZoo, WorldEnv).
* Tier 1 adapter = kernel facility: identical `rig.spec`, reports, settings and balances over 25-30 ticks.
* ReferenceLearner: 158,913 parameters; checkpoint 1.9 MB; save/load round trip 0 mismatches (weights, optimiser,
  RNG, memory; identical sampled actions and update afterwards). Updates change parameters in observation-only mode
  on Tier 1 and kernel worlds (gate R.7: 7 parameter-changing updates in one run, reward seen 0.0).
* Throughput (interface only, no learner): Tier 1 4,400 observe+execute+tick per second; kernel `tiny` 640 and
  `open_digital` 450 observe+execute per second. A kernel observation is 88-122 KB, mostly the 145-200 tool specs;
  the store keeps tool lists once (content-addressed), about 6.6 KB per stored record.
* Experience store: 1,140 records from four worlds, reopened from disk, every hash verified, a fresh learner trained
  on it (18 optimiser steps).
* `python -m crucible.experience run ...` twice: identical store digest, parameter hash and world digest.
* SP-11.2: 13 of 13 tool categories available and exercised in `open_digital` + `exp`; one agent-created service
  discovered as a new tool, paid for (5 cents), 3 of 3 permission violations refused.
* SP-11.4: 9 experiment round trips (3 per mode); rewards with and without experiments identical in all 3 modes.
* SP-11.5: 8 of 8 long-horizon features in a 16-day run (durations 1-600 s, consequences up to 14 days after their
  action, all 28 caused events linked to the builder's own `action_seq`).

## Gate status on this branch

Passing: R.1-R.7, R.n1, S.1, S.4, S.6, S.8, SP-11.2, SP-11.3, SP-11.4, SP-11.5, SP-11.6. S.7 needs four worker
processes (run through slots). Failing honestly, because their worlds do not exist yet: S.2 (no v5 Tier 2 domain,
workstream I), S.3 (no civilisation generator, workstream J), S.5 (world types 5 >= 3 and schema failures 0, but
only Tier 1 among the v5 tiers), SP-11.1 (12 of 13 kinds - semantic representations need workstream D's encoders on
v5 artifacts - and 1 of 3 tiers). They pass through the same `make(2|3, ...)` path once those packages land.

## Limitations

* The reference learner proves the interface (real updates, exact checkpoints); it is not a capable agent. In kernel
  worlds with 150-200 tools its random-ish exploration rarely does anything useful; that is an algorithmic
  limitation, not an environment failure.
* `RapidExperienceEnv` follows INTERFACE_CONTRACTS section 8 (`step({agent: action})`); it is untested until
  `crucible.rapid` lands (no stand-in is tested in its place).
* Semantic representations are produced only for v5 `sha256:` artifacts and only when `crucible.semantic.encode`
  exists.
* Kernel observations carry the full tool list every time (exact and lossless, but large); learners that need less
  can read `tools(agent)` on change only.
* The CAM 2.0 adapter (`integrations/agentrs_cam2.py`) has the reward modes too, but it still imports `agentrs` at
  module load (a 4.0 dependency of that file and of `tests/test_agents_llm.py`; the experience interface itself
  imports nothing from any learner).

## Wave 3 (continuation, branch c5c/w3-obs)

Adversarial verification on the integrated tree (87a6c54) found and fixed:

* **nested-call leak** (kernel, `Ctx.call`): a call made *as* a participant from a module process (a deferred
  workflow job `wf.run_later`, a trigger, a simulated participant acting through tools) returned evaluator-only
  fields, because the filter keyed on the chain's origin only. A deferred `rig.read` job returned the mechanism's
  `score` (positive control: 2 leaks on the pre-fix kernel, 0 after). Every call whose principal is not the system
  now gets the participant view.
* **relatedness verdicts in the science world** (`sci.replicate/paper/papers/replications`, `sci.replication`
  events): `related_to_author`, `independent`, `independent_consistent` registered evaluator-only.
* **adoption verdicts of the capability registry** (`services/capabilities.py`, inherited by the v5
  `innovation/caps.py`): `caps.view/list/graph/evidence` returned `independent_adopters`, `independent_payers`,
  `independent_buyers` and per-evidence `independent` flags, and `caps.installed/evidence/payer/market` events carried
  them - a provider could learn which of its own organisations the evaluator links to it by having them install its
  capability; `caps.categories` (markets selected by independent buyers, with a `min_adopters` threshold) is now an
  evaluator tool. Found by reading every relatedness computation that feeds a tool result or event after the
  probes passed: the probes' scenario never created capabilities, so it could not see these (a coverage limit).
* **evaluator metrics missing from the registry**: `scheduled_churn_share` (and the commerce list) were readable
  through self-goals in observation-only worlds; `evaluator_only.evaluator_metrics()` is now the union of the
  registry, `goals.metrics.EVALUATOR_ONLY_METRICS` and `commerce.observables.EVALUATOR_ONLY`.

Added: Tier 3 through `make(3, profile)` with learner-side `goals`/`exp` modules (`registry.civ_world`); Tier 1
self-experiments; the label channel; per-tier, nested-call and identity probes; gate tests for OO.9-OO.12, OO.14,
OO.16, SP-11.1b, SP-11.4b, SP-11.6c, SP-11.6d, SP-2.2; every OO gate counts a tier only where its worlds were really
built and measured (the wave-1 tests counted Tier 3 whenever `crucible.civ.generate` was importable).

Tier 2 has no registered domain (`crucible.domains.base.REGISTRY` empty), so every per-tier gate fails on Tier 2
by design. Tier 3 was verified locally with a stand-in for the pending `space` integration fix (scratch-only plugin
adding the semantic `space` module to the civilisation spec, never committed); the committed tests call
`crucible.civ.generate`/`spec` and fail until the fix is merged.
