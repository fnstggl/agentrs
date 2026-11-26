# Audit: workstream L — universal experience interface, reward modes, observation-only learning

Auditor scope: MASTER_SPEC Part 11 (11.1–11.6), Gate R, Gate S (restated architecture-independently by
Clarification 1), SP-11.1…SP-11.6, and the binding user amendment (a) "observation-only is the universal default".
Code inspected: `src/crucible/experience/*`, `src/crucible/api/*`, `src/crucible/integrations/*`,
`src/crucible/training/*`, the reward paths of `research/envs.py`, `research/bridge.py`, `selfplay/arena.py`,
`scale/rollout.py`, the world modules an observation-only learner can call (`goals/selfgoals.py`,
`usefulness/module.py`, `mechanisms/module.py`), `docs/v5/LEARNER_GUIDE.md`, `examples/experience_external_adapter.py`,
and the CAM interfaces at `../cam7-ref` (read only).

Worktree `feat/crucible-5-complete` @ `a881d18`. Nothing previously marked done was trusted; every verdict below rests on
a test or probe I ran (commands and logs in the last section; probe scripts and raw outputs in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/L_experience/`).

## 1. Bottom line

The experience package is real, integrated and mostly well built: one `make(tier, name, seed, reward_mode)`
constructor, a validated 15-field `Observation`, lossless event delivery with own-action provenance in kernel worlds, a
hash-chained `ExperienceStore`, a reference GRU learner that does real PPO + world-model updates and (I checked)
**actually improves** in a Tier 1 task in observation-only mode, exact checkpoints, a reproducible CLI, an `exp`
self-experiment module that pays nothing, and an external adapter that uses only the public API. Gate R (8/8), S.1,
S.4, S.6, S.8 and SP-11.2…SP-11.6 pass when re-run.

The amendment and the integrity requirements are not met, though:

1. **Observation-only is not the default anywhere.** 13 constructors default to `legacy_shaped_reward`
   (`make`, `check_mode(None)`, all three experience adapters, `MechVecEnv`, `KernelMechEnv`, `SelfPlayEnv`,
   `KernelSelfPlayEnv`, Gymnasium `CrucibleEnv`, PettingZoo `CrucibleParallelEnv`, `scale.rollout.WorldEnv`, the CAM 2.0
   adapter, `ReferenceLearner`). A default-constructed `MechVecEnv` paid a nonzero reward on 30 of 30 steps; the docs
   (`LEARNER_GUIDE.md` §4, `modes.py` docstring) state legacy is the default "everywhere". The AGENTRS superlearner
   adapter (`integrations/agentrs_sl.py:235`) has no reward modes at all and pays a native −0.01 for schema-invalid calls
   (an action-quality judgement).
2. **Privileged goal progress leaks through ordinary tools in observation-only mode, and the Gate R battery cannot see
   it.** `goals.check_now` (read-only, present in 23 of 24 non-mechanism 4.0 world families) evaluates any predicate over
   the evaluator's metrics, including `independent_revenue` — the evaluator's related-party/circular-flow netting over the
   whole payment graph. In my probe an observation-only learner with a revenue objective recovered the verifier's exact
   metric (400,000 cents, after a 500,000-cent circular flow was netted out) and therefore the exact privileged progress
   (0.4) with 21 read-only calls; `leakage.find_keys` reported nothing. `exp.register` accepts `goals.check_now` as a
   measure, so experiments can wrap the oracle. `use.report` returns Crucible's hidden-grader `verified_quality` and
   evaluator-netted independent revenue to any operator in every mode. Tier 1 observations and `rig.read` carry `score`,
   documented as "the objective quantity this tick (what goals count)" (`mechanisms/base.py:89`).
3. **The v5 Tier 1 adapter (`RapidExperienceEnv`) is broken for multi-agent families and leaks private events.**
   `execute(a1, …)` returns agent a0's result (`tier1.py:577`); `observe_since(a1, 0)` delivered a private a0→a2
   message event that the rapid world itself says a1 cannot see (`tier1.py:598-600`, one shared log for all agents);
   every event is duplicated once per agent observation; `t` is a tick index, not seconds; `cost_s` is always 0; no
   `cause`/`action_seq`; `world_changes` is never counted (0 in 4 families), so S.1 measured on v5 families would fail.
4. **Tier 2 and Tier 3 cannot run.** No domain is registered and `crucible.civ` is a 5-line stub; in addition, the
   Tier 2 branch of `make()` is written against INTERFACE_CONTRACTS §9, not against the `crucible.domains.base.Domain`
   that workstream I actually built (configs tuple, `build(config, rng, overrides)` returning a recipe,
   `objective(world, principal, difficulty)`, discovery through `base.REGISTRY` not a module-level `DOMAIN`). With a
   conforming `Domain` patched in, `make(2, …)` raises `AttributeError: 'tuple' object has no attribute 'get'`. So S.2,
   S.3, S.5 and SP-11.1 fail, and would keep failing after I lands.
5. The leakage battery proves **mode-invariance of the adapters**, not absence of leakage: a world tool that hands out
   evaluator data hands it out identically in every mode, so it passes every invariance and canary check.

Transfer question (would a learner here acquire transferable capabilities or simulator shortcuts?): in Tier 1 the
reference learner learns a genuine control policy from its own balance; but in the kernel economies an optimiser
quickly finds that `goals.check_now`/`use.report`/`use.probe` give it the grader's verdict, which no real market offers.
That is the single largest shortcut in this area.

## 2. What I ran and what it showed

| # | Command / probe | Result |
|---|---|---|
| T1 | `pytest tests/v5/test_gate_R.py` | 8 passed (17.7 s) |
| T2 | `pytest tests/v5/test_gate_S.py -k "not S7"` | S.1, S.4, S.6, S.8 passed; S.2, S.3 fail (no Tier 2/3); S.5 fails (`tiers_one_interface=1`) |
| T3 | `pytest tests/v5/test_gate_S.py -k S7` via `crucible.scale.slots --cpu 4` | passed (20.8 s): parallel_environments 4, 4 worker processes, 432 farm steps, 0 failures, 14 learner updates, 2 parameter versions; vectorised 8 envs |
| T4 | `pytest tests/v5/test_gate_SP_11_{1..6}.py` | SP-11.2…11.6 pass; SP-11.1 fails: 12/13 kinds (no semantic), 1 tier |
| T5 | `pytest tests/test_v5_experience_core.py` | 11 passed (incl. CAM 2.0 adapter modes, tier-1 tick-for-tick twin, store tamper detection) |
| P1 | `probe_defaults.py` | 13 constructors default to legacy; default `MechVecEnv` reward nonzero on 30/30 steps; default `make(1,'mech:scm_additive')` paid −8.04 over 20 ticks; `ReferenceLearner()` defaults to legacy, signal `reward` |
| P2 | `probe_goal_oracle.py` | observation-only learner recovers `independent_revenue`=400000 (evaluator 400000.0) and progress 0.4 (verifier 0.4) via 21 `goals.check_now` calls; `exp.register` with `goals.check_now` measure accepted; scanner hits 0 |
| P3 | `probe_use_report.txt` | in `ow_tiny`, observation-only: `use.report` returns `verified_quality`, `revenue_cents` (independent), `retention` … |
| P4 | `probe_keyscan.py` (ReferenceLearner, 100 steps, 8 worlds) | Tier 1 obs carry `numbers.rig.seats.r1.score`; scanner gives 14 false positives on rapid `spec.objective` (public texts) and 1 on a bounty `reward` (money) — key-name scanning is both over- and under-inclusive |
| P5 | `probe_rapid.py`, `probe_rapid_private_event.txt` | 74 rapid families build and validate; `t` after `advance(3600)` = 1 for all; multi-agent: adapter returns a0's `{"waited": true}` for a1's delivered message; a1's `observe_since` returns a0→a2 private message (rapid env: not visible); 4–5 events for a1 vs 2 in the rapid env; ReferenceLearner on 4 rapid families: 70 actions, `world_changes=0`, params changed |
| P6 | `probe_tier2_contract.txt` | real `domains.base.Domain` through `make(2, …)`: `AttributeError` for `toy`, `toy:a`, `d5_toy_a` |
| P7 | `probe_learning.py 3000` (49.6 s, 1 thread) | trained ReferenceLearner on `scm_additive:3`, observation-only: episode score −0.29 → 0.75 (last 3: 0.75, 0.72, 0.61), success 29%, loss 1.19 → 0.78; frozen-at-init: −0.30 → −0.28…−0.48, success 0%; wait-only: −0.006 |
| P8 | `probe_no_objective.py 90` | 90 simulated days, no objective: reward 0 throughout, `observe` 6–10 ms, adapter indices small (1,647 visible seqs), process memory grows ~0.9 MB/day (world event log), `verify()` has `verifier: None` |
| P9 | `probe_v5_modules.py` | KernelExperienceEnv on the v5 module world (`redteam.v5.worlds.v5_spec('all')`, 36 modules incl. products, subs, ventures, xchg, labor, media, embed, semantic_inference): 0 schema failures, **semantic_representations present**, 11 kinds; ReferenceLearner 149 actions, 34 ok, 5 world changes |
| P10 | `probe_tool_categories_v5.txt` | 96 of 307 tools in that world are category `other` (hosts prod, venture, xchg, labor, inst, commit, needs, media, embed, asset unmapped); `browser` and `research` absent |

## 3. Subsystems

### 3.1 Reward modes and defaults — REPAIR (M)
`experience/modes.py` (`check_mode`, `RewardGate`, `info_objective`) is a clean, single point of truth, and the three
modes behave as specified when selected explicitly (SP-11.3 reproduced: kernel legacy rewards nonzero, verified pays
exactly 1.0 once, observation-only pays 0.0 with the verifier still reporting success).
Defects:
* Default is legacy: `modes.py:34`, `registry.py:99`, `kernel.py:59`, `tier1.py:61`, `tier1.py:472`,
  `reference.py:98`, `research/envs.py:47`, `research/bridge.py:31`, `selfplay/arena.py:153,425`,
  `api/gym_env.py:71`, `api/pettingzoo_env.py:42`, `scale/rollout.py:128`, `integrations/agentrs_cam2.py:103`.
  Only `parallel.FarmEnv` defaults to observation-only, so the defaults are also inconsistent.
* `integrations/agentrs_sl.py`: no modes; `observe_consequences` returns `reward=-INVALID_COST` (`:235`), episode ends on
  verified success (`_ended`, `:225-227`), and `goal()` gives the learner the evaluator's progress metric name and target
  (`:107-111`).
* `api/http_server.py` / `api/mcp_server.py` CLIs expose no reward-mode option; the OpenEnv payload puts
  `info["objective"]` (legacy: full status with progress) under `metadata.objective` (`http_server.py:75`).
* `RapidExperienceEnv` verified mode pays when `outcome()["success"]` first becomes true after any `execute`
  (`tier1.py:590-594`), i.e. mid-episode, not at the terminal outcome.
* `legacy_check` (`leakage.py:530+`) compares default-constructed current envs with 4.0; flipping the default will make it
  fail unless every current-side constructor passes `reward_mode=LEGACY` explicitly. CAM 7's
  `superlearner/unified/runtime/rollout.py` constructs `MechVecEnv(tasks)` with the default, so the flip also needs a
  documented migration (CAM must pass `legacy_shaped_reward` explicitly if it wants the 4.0 reward).
Work: flip every default to `observation_only` (and `check_mode(None)`), make legacy/verified explicit opt-ins recorded
in `describe()`/experience records, add modes to the sl adapter (drop the invalid-call penalty and outcome termination
in observation-only), add `--reward-mode` to HTTP/MCP CLIs, make rapid verified mode terminal-only, update
`legacy_check`, 4.0 baseline scripts and tests to pass `legacy_shaped_reward` explicitly.

### 3.2 Reward-leakage defence (`experience/leakage.py`, Gate R tests) — REDESIGN (M)
The battery (11 tests, 9 canaries) is careful about what it checks: zero reward and reward slot, absent
`PRIVILEGED_KEYS`, byte-identical observations across modes and across evaluator-only data, canaries in evaluator
runtime, internal events, other participants' mail, spec params, the hidden mechanism config and internal filings. All
pass when re-run. Its architectural assumption is wrong for the research objective: leakage is detected by key name
(`leakage.py:37-60`) and by **invariance across reward modes** (`:349`, `:416`). A world tool that returns evaluator
data does so identically in every mode and under neutral or canary evaluator parameters, so it passes. Proven
counterexamples: `goals.check_now` over `independent_revenue` (P2), `use.report.verified_quality` (P3), `score` in Tier 1
reports (P4). The key list omits `score`, `success`, `quality`, `verified_quality`, `rating`; it also flags public
objective texts and bounty money (false positives). Coverage omits `RapidExperienceEnv`, multi-agent rapid families,
Tier 2/3, the v5 module worlds, the sl and CAM 2.0 adapters (the CAM 2.0 adapter is covered only by the core test).
Work: (1) an explicit registry of evaluator-side quantities (GoalVerifier metrics incl. `independent_revenue`,
`independent_members`; usefulness grader outputs; rapid `outcome()`; domain `evaluate`) and an observation-only
**tool policy** in the adapters that either hides tools that evaluate them or restricts them to quantities the agent can
compute from its own view; (2) **oracle-recovery tests**: an adversarial scripted prober that tries to reconstruct the
verifier's progress from tool calls and must fail (correlation / exact-recovery bound), run per tier; (3) extend
invariance/canary tests to rapid (incl. multi-agent private events), domains, civilisations, v5 module worlds and all
integration adapters; (4) replace key-name counting by a schema allowlist of legitimate consequence fields where
practical.

### 3.3 Observation contract (`experience/observation.py`, `features.py`) — EXTEND (S)
Exact field set, JSON-nativeness, numeric checks, event/tool/entity shapes. 0 schema failures across all 74 rapid
families (P5), 4.0 kernel worlds (T4) and the v5 module world (P9). `kinds_present` counts content, not fields, and all
13 kinds have now been observed somewhere (semantic representations in the v5 module world, P9), but no single test
world produces all 13 and SP-11.1 does not use a v5 artifact world. Gaps: `t` units differ by tier (seconds in
kernel/mech, ticks in rapid); "uncertain information" is only free text in `missing`; `objective` is text only (CAM's
`GoalSpec.metrics` has numeric targets); execute results carry no monetary cost (CAM `ExecutionResult.cost`; the external
adapter sets it to 0 "not measured"). Tool categories are a static host table (`features.py:224-249`) that maps 31% of
tools in the v5 module world to `other` (P10).

### 3.4 Tier 1 mechanism adapter (`MechExperienceEnv`, `tier1.py:57-466`) — REPAIR (S)
Verified tick for tick against the kernel facility (core test), learner learns through it (P7), provenance via
`settings_in_force`. Defects: `score` (creator-defined objective quantity; for `tech_*` `ontime_value` it is a gated
value, not a physical reading) is in observations (`tier1.py:342`) and reports (`_report_view`, `:231`) in
observation-only mode — needs an explicit fidelity decision (keep only if it is a quantity the operator would see in
reality; otherwise drop in observation-only); no `exp` self-experiment tool in Tier 1; `verify()` is fine.

### 3.5 Tier 1 rapid adapter (`RapidExperienceEnv`, `tier1.py:470-622`) — REPAIR (M)
Builds and validates for all 74 rapid families; single-agent families work. Defects (all reproduced, P5):
* `execute` returns `out["result"]` (agent a0's) for any agent (`:577`); for multi-agent families a1's action results
  are wrong.
* One tick per `execute` (`env.step({agent: action})`), so agents of a team family can never act in the same tick
  (coordination families need joint submission in one tick).
* Shared event log for all agents, no visibility filter (`:514-530`, `:598-600`): private events leak across agents;
  each event is stored once per agent observation (duplicates); `t` of message events missing → non-monotonic.
* `t`/`cost_s` semantics differ from the contract (ticks; always 0); no `action_seq`, no `cause`; `last_changed` never
  set so `run_learner` counts 0 world changes; `digest()` hashes name/seed/steps/cursors, not world state (weak S.8).
Work: delegate `observe_since` to the rapid env's own visibility-filtered `observe_since` with a per-agent seq map;
buffer per-agent actions and step once per tick (joint actions), return `results[agent]`; express `t` in seconds
(tick × tick_s) and real `cost_s`; add cause provenance for own actions; set `last_changed` from state; digest the
engine state.

### 3.6 Kernel adapter (`KernelExperienceEnv`, `kernel.py`) — REPAIR (M)
The strongest part: lossless incremental scanning, visibility through `World.visible`, causes only for the agent's own
commands (direct / scheduled, transitive), truncation reported in `missing`, semantic encodings of v5 artifacts,
analytics from public events only, several agents per world, objectives behind `begin/step_reward/status`. Reproduced in
core tests, SP-11.1/2/4/5 and P8/P9. Defects: default legacy; it "adds nothing to the world's rules" (`kernel.py:5`),
which in observation-only mode means it passes the evaluator oracles of the world modules straight through (P2/P3) —
the adapter is where an observation-only tool policy belongs; without an objective the verifier is `None`
(`kernel.py:401`), so "independent verification" is only digest/invariants; memory of a 1-agent tiny world grows
~0.9 MB/simulated day (world log in memory), which limits "indefinite" runs without a store-backed world.

### 3.7 Tier 2 / Tier 3 paths (`registry.py:99-179`) — REPAIR (S) for the adapter; capability IMPLEMENT (blocked on I/J)
`domains()` looks for a module-level `DOMAIN` (`registry.py:39-58`), `_domain_and_config` treats `configs` as a dict
(`:85`), `dom.build(cfg, seed)` expects a `WorldSpec` (`:125`), `dom.objective(env.world, difficulty)` omits the
principal (`:173`). Workstream I's real framework differs on all four (`domains/base.py:36-60,95-120`), proven by P6.
`crucible.civ` exports nothing. Work: rewrite the Tier 2 branch against `domains.base` (`REGISTRY`, `load_all()`,
`base.spec(domain, config, seed, overrides)`, per-principal objective, `participant_principal`), then run S.2/S.5/SP-11.1
on real domains; Tier 3 waits for J's `generate`.

### 3.8 Tool interface and dynamic affordances — EXTEND (S)
SP-11.2 reproduced: a builder deploys `buildsvc`, a client gains its endpoints after `world.describe` (category
`agent_services`), pays exactly the price for a call, and three permission violations are refused. Extend: categories
for v5 hosts (P10); tool discovery and agent-created services exist only in kernel worlds (Tier 1 tool sets are fixed);
`tool_composition` beyond `wf.*`; the SP-11.2 criterion counts categories *available*, not *exercised*.

### 3.9 Self-generated experiments (`experience/experiments.py`, module `exp`) — REPAIR (S)
Sound design: any read-only tool measured as the agent, baseline/samples/deadline/held, no money moves, private events.
SP-11.4 reproduced (rewards identical with and without experiments in all modes). Defects: `goals.check_now` (and any
other evaluator-backed read-only tool) is accepted as a measure (P2), so the module can package an evaluator oracle;
only kernel worlds carry it (not Tier 1 mech/rapid adapters). Work: deny measures that evaluate evaluator-side metrics
(shared registry from 3.2); offer experiments in Tier 1 adapters.

### 3.10 Long-horizon experience and provenance — EXTEND (S)
SP-11.5 reproduced in a kernel world: ≥3 durations, scheduled delayed events, async workflow 8 days later, live service
used for two weeks, two concurrent experiments, a 3-payment contract, consequences ≥7 days later, every caused event
joined to the agent's own `action_seq`. Missing in the rapid adapter (no causes, no durations) and untested in Tier 2/3.

### 3.11 Experience store (`experience/store.py`) — RETAIN (S for hardening)
Append-only shards, per-record hashes, hash-chained manifest, content-addressed blobs, torn-tail recovery, tamper
detection (core test), 1,140 records written/reopened/verified and trained on by a fresh learner (S.4 reproduced).
Notes: single writer, no lock (fine for `FarmRun`, which writes from the parent; unsafe on a shared volume with
concurrent writers); full observation per record (size).

### 3.12 Reference learner (`experience/reference.py`) — REPAIR (S)
Real, seeded GRU + PPO-clip + value + entropy + world-model loss; exact save/load incl. optimiser and RNG (S.6
reproduced: 0 mismatches and identical future actions). It **learns** in Tier 1 observation-only with its own balance
signal (P7: score −0.29 → 0.75 vs frozen −0.45, success 29% vs 0%). Defects: default `reward_mode` legacy
(`:98`) so `ReferenceLearner()` trains on env reward; GAE bootstrap appends `V(s_T)` of the last stored state instead of
`V(s_{T+1})` (`:287`); `learn_from_records` does not reset the recurrent state between agents/worlds; string arguments
without candidates fall back to `"x"`/the actor id, so it cannot use content/communication tools meaningfully in Tier
2/3; checkpoints exclude the recurrent state; `torch.load(weights_only=False)` (`:401`) loads pickles. Only one task/seed
was shown to learn; learning in Tier 2/3 is untested.

### 3.13 Driver, vectorisation, process parallelism (`loop.py`, `parallel.py`) — EXTEND (S)
`run_learner` drives one agent only (`loop.py:27`) — no multi-agent driver although the adapters host several agents;
world-change counting relies on `last_changed`, which the rapid adapter never sets; `VecExperience` steps environments
sequentially in one process; `FarmRun` reuses `RolloutFarm` unchanged (S.7 reproduced: 4 concurrent environments in 4
worker processes, 0 failures).

### 3.14 CLI / reproducibility (`experience/__main__.py`) — EXTEND (S)
S.8 reproduced (two fresh processes, identical store digest, params hash, world digest, counts) for Tier 1 mech only.
Rapid digests are not world digests; Tier 2/3 absent. The CLI's own `--reward-mode` default is observation-only (good).

### 3.15 External surfaces: Gymnasium, PettingZoo, HTTP/OpenEnv, MCP (`api/`) — REPAIR (M)
Modes implemented and covered by the battery (Gym/PZ). Defects: defaults legacy; observations are rendered text plus a
time box (`gym_env.py:107-110`), so exact numbers reach Gym/HTTP/MCP learners only through text — contrary to 11.1
("do not force precise financial or technical measurements into lossy text-only representations"); there is no
Gym/PettingZoo wrapper over `ExperienceEnv`, so Gym learners cannot reach Tier 1 rapid, Tier 2 or Tier 3 through the
universal contract; HTTP/MCP have no mode switch.

### 3.16 Learner integrations (`integrations/`) — REPAIR (S)
CAM 2.0 adapter: modes implemented and tested, default legacy. AGENTRS sl adapter: no modes, native invalid-call
penalty, outcome termination, progress-metric leakage into `goal()` (3.1). Shared plumbing (`ActorView`, `EntityMemory`,
`ActionTemplates`) is strategy-free. These modules import `agentrs.*` lazily and are optional, consistent with
Clarification 1.

### 3.17 Training exports (`training/trajectories.py`, `training/league.py`) — EXTEND (S)
4.0 trajectory export (`crucible-traj/1`, NPZ) and league with independently measured outcomes; not connected to
`crucible-exp/1` records or reward modes (records a `reward` field from the 4.0 Runner). Unify the formats and record
the reward mode.

### 3.18 Learner guide and CAM compatibility — REPAIR (S)
`LEARNER_GUIDE.md` is clear and the external adapter (SP-11.6) maps one-to-one onto CAM Unified `EnvAdapter`
(`cam_unified/interfaces.py:288-303`): `observe`, `available_tools`, `execute` without clock movement,
`collect_events(since)`, `advance_time(days)`, `goals`, evaluator-side `GoalEvaluator`. But the guide states legacy is
the default (§4), says no evaluator data appears in any result or tool list (§2; refuted by P2/P3), presents the Tier 2
constructor with the contract I's code does not implement (§5), and claims `t` is seconds in every tier. CAM gaps to
document: `ExecutionResult.cost`, `GoalSpec.metrics`, observation `caused_by` (≈ `cause.action_seq`), and the default
flip for `MechVecEnv` consumers.

## 4. Requirement verdicts (workstream L, 22 entries)

| id | verdict | reason | work needed | criteria weak |
|---|---|---|---|---|
| R.1 | REPAIR | 0 nonzero rewards on the covered paths in observation-only (reproduced), but legacy is the default in 13 constructors (default MechVecEnv: 30/30 steps rewarded); sl adapter pays a native penalty with no mode; rapid/Tier 2/3 not covered | flip defaults; add modes to sl adapter; measure on every adapter and tier incl. default construction | yes |
| R.2 | REDESIGN | key-name scan finds 0, but `goals.check_now` yields exact verifier progress (P2), `use.report.verified_quality` (P3), Tier 1 `score` | evaluator-metric registry + observation-only tool policy + oracle-recovery tests per tier | yes |
| R.3 | REPAIR | 9 canaries absent (reproduced); but evaluator functions over other participants' private payment graph are callable as tools; rapid `observe_since` leaks other agents' private events (P5) | close tool oracles; fix rapid visibility; canaries in rapid multi-agent, domains, civs | yes |
| R.4 | EXTEND | legitimate fields equal to legacy on covered envs (reproduced); not checked on rapid/Tier 2/3; must be re-checked after leak closure | extend to every tier; allowlist of legitimate consequence fields | no |
| R.5 | EXTEND | `verify()` runs on every adapter and produced outcomes (reproduced); without an objective it verifies nothing; Tier 2/3 absent | per-tier evaluator logs; domain/civ evaluators; verification of no-objective runs (invariants, outcome measures) | yes |
| R.6 | REPAIR | 11/11 legacy runs byte-identical to 4.0 (reproduced), but only because legacy is the default; after the mandated flip they must be explicit opt-ins | pass `legacy_shaped_reward` explicitly in legacy_check, 4.0 baselines and CAM migration notes; test both explicit legacy and default observation-only | yes |
| R.7 | RETAIN | reference learner changes parameters in observation-only Tier 1 and kernel runs (reproduced) and genuinely improves in Tier 1 (P7) | keep; add the improvement-vs-frozen check to the gate | yes |
| R.n1 | REDESIGN | 11 battery tests pass but cannot detect mode-invariant leakage; no rapid/T2/T3/integration coverage | as R.2 plus per-adapter coverage matrix | yes |
| S.1 | REPAIR | 220-step run on the 4.0 mech facility passes; on v5 rapid families the adapter counts 0 world changes and multi-agent results are misattributed | fix rapid adapter; measure S.1 on v5 rapid families (single and multi-agent) | yes |
| S.2 | IMPLEMENT | no Tier 2 domain registered; Tier 2 branch incompatible with `domains.base.Domain` (P6) | rewrite Tier 2 branch against domains.base; run on ≥3 domains | yes |
| S.3 | IMPLEMENT | `crucible.civ` is a stub | wait for J's `generate`; run on ≥2 profiles | yes |
| S.4 | RETAIN | 1,140 records stored, reopened, hash-verified, a fresh learner updated from them (reproduced) | extend to all tiers when available | no |
| S.5 | REPAIR | fails: 5 world types, 1 v5 tier; rapid not in the list; metric `world_types_one_interface>=3` passes without Tier 2/3 | add rapid, domains, civs; criterion must require tiers 1, 2, 3 | yes |
| S.6 | RETAIN | 0 mismatches incl. optimiser, RNG, identical future actions and update (reproduced) | include recurrent state; `weights_only` loading | no |
| S.7 | RETAIN | reproduced through the slot scheduler: 4 overlapping environments in 4 worker processes, 0 failures, learner updated and parameters republished; 8 vectorised envs | keep; measure on Tier 2/3 worlds too | no |
| S.8 | EXTEND | reproduced for Tier 1 mech; rapid digest is not a world digest; Tier 2/3 absent | real digests in every adapter; reproduce one command per tier | yes |
| SP-11.1 | REPAIR | fails: 12 kinds, 1 tier; semantic kind exists in the v5 module world (P9) but the test never uses it; rapid `t` in ticks | use a v5 artifact world; fix rapid units; Tier 2/3 | yes |
| SP-11.2 | EXTEND | passes in 4.0 open_digital (reproduced); 31% of v5 tools categorised `other`; discovery/agent services only in kernel worlds | category mapping for v5 hosts; criterion on categories exercised | yes |
| SP-11.3 | REPAIR | 3 modes verified (reproduced) but not the universal default; sl adapter has none; rapid verified mode pays mid-episode | as 3.1 | yes |
| SP-11.4 | REPAIR | round trips and zero payment reproduced; `exp` accepts the `goals.check_now` evaluator oracle as a measure (P2); not in Tier 1 adapters | deny evaluator-backed measures; experiments in every tier | no |
| SP-11.5 | EXTEND | 8/8 features reproduced in a kernel world; rapid adapter lacks durations/causes; Tier 2/3 untested | provenance in rapid; run in Tier 2/3 | no |
| SP-11.6 | REPAIR | external adapter imports only public names, changes nothing, runs and stores (reproduced); guide is inaccurate on defaults, leakage, Tier 2 | correct the guide; document CAM migration for the default flip | no |

## 5. Spec requirements with no manifest entry

1. Amendment (a): `observation_only` is the default of every environment, adapter, wrapper, server and CLI in all three
   tiers; legacy and verified modes only when explicitly selected (and recorded in experience records).
2. Amendment (a)/11.3: no evaluator data through world tools, tool responses or event metadata — tested by
   oracle-recovery (can a learner reconstruct verifier progress or grader quality?), not only by key names.
3. Amendment (a): operation without an assigned objective indefinitely (long runs, bounded adapter state, verification
   still meaningful).
4. 11.3: "Authorized verified outcome labels may be supplied separately for training and calibration" — no operator
   label channel exists (no API, no record type).
5. 11.6 / Clarification 1: multiple learners in one world through the interface (multi-agent driver; Tier 1 multi-seat
   and rapid team families; Tier 2/3 multi-participant).
6. 11.1: exact structured observations on the external surfaces (Gym, PettingZoo, HTTP/OpenEnv, MCP), and a Gym
   wrapper over `ExperienceEnv` for every tier.
7. 11.6: separate algorithmic limitations from environment failures (learner-failure vs environment-failure reporting).
8. Amendment (a)/11.4: self-generated experiments without automatic reward in every tier (today kernel worlds only).
9. One unit system across tiers (`t` seconds, `cost_s`, `action_seq`/`cause` in every adapter).
10. Learning-signal sanity: the reference learner improves over a frozen copy in ≥1 world per tier (prevents
    "parameters changed" from passing as learning).

## 6. Shortcuts and integrity notes

* Evaluator oracles callable by the agent: `goals.check_now` (+ `goals.propose` scheduled checks) over
  `independent_revenue`, `independent_members`, `service_requests_ok`; `use.report.verified_quality`; `use.probe`
  `mean_score` (paid, rate-limited acceptance lab backed by the hidden grader). The latter two belong to the economy
  workstreams but surface through this interface in observation-only mode.
* Creator-defined per-tick `score` in Tier 1 reports.
* Native action-quality penalty in the sl adapter.
* No scripted emergence, stub vectors or silent skips found in this area; S.2/S.3 fail loudly rather than substituting
  other worlds (good). `test_cam2_adapter_reward_modes` imports AGENTRS from the parent repo (not a mock).

## 7. Commands (all from `crucible/`, `PYTHONPATH=src`, `OMP_NUM_THREADS=1`)

```
timeout 1500 /venv/main/bin/python -m pytest -q -p no:cacheprovider tests/v5/test_gate_R.py
timeout 1700 /venv/main/bin/python -m pytest -q -p no:cacheprovider tests/v5/test_gate_S.py -k "not S7" -rA
/venv/main/bin/python -m crucible.scale.slots --cpu 4 --gpu-mb 0 --label "audit: L_experience gate S.7 parallel envs (4 workers)" \
    --timeout 1200 -- env OMP_NUM_THREADS=1 PYTHONPATH=src /venv/main/bin/python -m pytest -q -p no:cacheprovider \
    tests/v5/test_gate_S.py -k S7 -rA
timeout 1700 /venv/main/bin/python -m pytest -q -p no:cacheprovider tests/v5/test_gate_SP_11_{1,2,3,4,5,6}.py -rA
timeout 900 /venv/main/bin/python -m pytest -q -p no:cacheprovider tests/test_v5_experience_core.py -rA
/venv/main/bin/python <scratch>/probe_defaults.py | probe_goal_oracle.py | probe_rapid.py | probe_keyscan.py 100 |
    probe_learning.py 3000 | probe_no_objective.py 90 | probe_v5_modules.py   (+ inline probes saved as .txt)
```
S.7 was first queued with `--cpu 5` (not granted while other auditors held the 9-slot budget), then re-run with
`--cpu 4` (4 farm workers; the parent only waits) and passed.

## Skeptic review

Reviewer: independent skeptic for area L_experience. Worktree HEAD is now `dd745b9` (one docs/manifest commit after
the audited `a881d18`: Clarifications 2 and 3 recorded, manifest gains OO.1-OO.8 for workstream L and RP.1-RP.4).
No source file changed between the two commits, so the original findings still apply. Every challenge below rests on a
probe I wrote and ran; scripts, JSON outputs and logs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/L_experience_skeptic/`. All runs were
single-process with `OMP_NUM_THREADS=1`, except the 2-worker farm probe, which went through `crucible.scale.slots --cpu 2`.

### S-1. Verdicts overturned

| target | from | to | evidence (probe) |
|---|---|---|---|
| Experience store (subsystem) | RETAIN | REPAIR (S) | `probe_store.py`, `probe_rapid_msg_schema_store.py` |
| S.4 | RETAIN | REPAIR | `probe_offline_boundary.py`, `probe_store.py`, `probe_rapid_msg_schema_store.py` |
| S.6 | RETAIN | REPAIR | `probe_checkpoint_resume.py` |
| S.7 | RETAIN | REPAIR | `probe_farm_concurrency.py` (1 and 2 workers) |
| Driver loop, vectorisation, parallelism | EXTEND | REPAIR (S) | `probe_farm_concurrency.py`, `probe_offline_boundary.py` |
| Tier 1 rapid adapter | REPAIR (M) | REBUILD (M) | `probe_rapid_spec.py`, `probe_rapid_msg_schema_store.py`, `probe_learning_seeds.py` |
| Tier 2 branch of `make()` | REPAIR (S) | REBUILD (S); Tier 3 IMPLEMENT | grep: no test or caller ever runs it |
| R.3 | REPAIR | REDESIGN | `probe_rapid_spec.py` (evaluation-partition label through the adapter) |

**Experience store, RETAIN to REPAIR.** The original evidence was the happy path plus a torn-tail test. Four defects:
* *Crash window loses data silently.* `_seal_part` renames `NNNNNN.jsonl.part` to `NNNNNN.jsonl` (`store.py:173`)
  before it appends the manifest line (`:178`). After a crash between the two, reopening sets `_shard` from the
  manifest, `_recover` (`:84`) looks only for the `.part`, and the next seal `os.replace`s over the orphaned shard. In
  the probe, records 10-14 were lost and `verify()` still passed.
* *The `$blob` reference is in-band.* `_resolve` treats any dict whose only key is `$blob` as a blob reference
  (`store.py:212`), wherever it occurs in a record. Agent-controlled content of that shape is therefore either rewritten
  on read (round trip not equal; the probe got the tool list back in place of the agent's dict) or makes `iterate()`
  raise `FileNotFoundError`, not `StoreCorrupt`. I reproduced this end to end: in `rapid:coop_split_info`, agent a0 sent
  a message whose `content` was `{"$blob": "fff..."}`. It appeared verbatim in a1's observation entities, and
  `iterate()` over a1's store then crashed. So one participant can make another participant's experience store
  unreadable.
* *Two writers corrupt the store.* With two `ExperienceStore`s on one directory, the second seal raised
  `FileNotFoundError` and reopening raised `StoreCorrupt: record index gap`. There is no lock, and
  `workspace_is_volume` stores may be shared.
* *Tail truncation is undetected.* After I deleted the last sealed shard and its manifest line, `verify()` passed with
  20 of 30 records. The chain has no external anchor; `digest()` is one, but nothing checks it on reopen.

**S.4, RETAIN to REPAIR.** The test passes, but the path it certifies (stored records, then
`learn_from_records`) trains on corrupted transitions. Records have no `done` or episode field (`make_record`,
`store.py:291-300`). `learn_from_records` chains consecutive records of one agent and env (`reference.py:345`), so in
Tier 1 every episode reset becomes a non-terminal transition. With the observation-only signal (change of
`bank.balances`), the reset, where the balance returns to the endowment, dominates. In 300 stored steps of
`scm_additive:21`, the 3 boundary transitions carried |signal| 8.95, 4.80 and 6.59. Inside episodes the mean was 0.07
and the maximum 0.23. Add the store defects above. The criterion (≥1000 records, ≥1 update) passes on this broken
path, so it is weak.

**S.6, RETAIN to REPAIR.** The gate test calls `L.reset()` and `L2.reset()` before comparing futures, which zeroes the
recurrent state in both learners and hides what the checkpoint leaves out. `state()` (`reference.py:387`) excludes `h`,
`h0`, `prev_tokens`, the transition buffer and `_pending`. In a mid-episode resume (the realistic case on recyclable
instances) the recurrent state went from |h|₁ = 33.9 to 0. Seven buffered transitions were dropped (the next update used
11 transitions instead of 18). One greedy action of 11 differed, and the parameters after the next update differed.
The weights, optimiser and RNG round-trip exactly; that part of the original evidence stands.

**S.7, RETAIN to REPAIR.** `FarmRun.concurrency()` builds each environment's interval as `(arrival - wall_s,
arrival)`, where `arrival` is when the **parent dequeued** the result (`parallel.py:131`), not a worker-side
timestamp. I made the parent busy before its first `get` (a sleep, standing in for a learner update or store I/O).
With **1** worker process running the 4 worlds one after another, it reported `max_concurrent_environments = 3`; the
queue size `2*n_workers` capped it. With **2** worker processes it reported **4**, and the gate assertion
(`parallel_environments >= 4 and failures == 0`; distinct pids are not asserted) **would pass** with at most 2-way
parallelism. The original 4-pid run is consistent with real 4-way parallelism, but its number does not establish it.
Work: worker-side monotonic start/end stamps in `Rollout`, and assert distinct pids ≥ 4. FarmRun's
`learn_from_records` also inherits the episode-boundary defect of S.4.

**Driver/parallelism, EXTEND to REPAIR.** The same two defects: a concurrency metric that over-reports, and offline
learning across episode resets. Both are correctness bugs, not missing breadth.

**Tier 1 rapid adapter, REPAIR to REBUILD (M).** Beyond the original list (a0's result returned for other agents,
shared unfiltered log, duplicates, ticks, no causes, `world_changes` never counted, digest not of world state):
* *Schema-invalid observations in every multi-agent family.* In all 6 multi-agent rapid families (coop_split_info,
  team_assembly, team_commons, team_discovery, team_profile, team_schedule), the **receiver's** observation fails
  `validate_observation` after one plain message: 0 errors before, then `entities[k] needs string 'id', 'kind' and a
  'relations' list`. Rapid message entities are `{"kind","from","content"}` (`rapid/env.py:198`), and the adapter passes
  them through. The original "0 schema failures across 74 rapid families" covered only agent a0, which never receives a
  message when it drives alone.
* *Memorisation key and evaluation-partition label.* In **38 of 74** families, `rapid.query {"target": "spec"}` returns
  `family`, `seed`, `category` and `split` through `RapidExperienceEnv` in `observation_only` (`rapid/env.py:64-68,
  235`), and 8 of them say `split: "val"`. The adapter applies no observation-only tool policy, and the Gate R battery
  never runs a rapid world.
* *The learner's own signal is absent.* `ReferenceLearner`'s observation-only signal reads only
  `numbers["bank.balances.*"]` (`reference.py:86,199`). Rapid observations put money in `resources.cash_cents`, so the
  signal is identically 0 in 13 of 13 sampled rapid families. The learner can only fit its world model there.

  With `execute`, `observe_since`, the event model, units, provenance, change tracking and digest all needing
  replacement, almost nothing but `tools()`/`_rapid_action()` survives: that is a rebuild of a ~150-line adapter, not
  a repair.

**Tier 2 branch of `make()`, REPAIR to REBUILD (S); Tier 3 IMPLEMENT.** `registry.py:116-128` and the
`_DomainObjective` and `_attach_domain` code (`:144-175`) have never been executed. No test or caller reaches them:
S.2 and S.5 only hit the `TierUnavailable` path, and `grep DOMAIN tests/` finds nothing else. They are written against
INTERFACE_CONTRACTS §9, which workstream I did not implement (the original auditor's P6 already showed the
`AttributeError`). Code that has never run, built against a contract that does not exist, is a rewrite.

**R.3, REPAIR to REDESIGN.** The same architectural flaw as R.2: canary and invariance tests cannot see evaluator data
that a tool returns identically in every mode. A new counterexample is the rapid `spec.split` partition label, plus
family and seed, which the battery never encounters. R.2 and R.n1 are already REDESIGN; R.3 should be consistent.

### S-2. Verdicts challenged and upheld (with my evidence)

* **R.7 RETAIN, upheld.** Learning reproduces beyond the single original run (`probe_learning_seeds.py`, 3000 steps,
  observation_only, evaluator-side scores). Each line compares the mean score of the first and second halves:
  * scm_additive:3, learner seed 1: 0.103 to 0.515, against a wait-only mean of -0.006.
  * scm_additive:3, learner seed 2: 0.086 to 0.574; success rate in the second half 0.875.
  * scm_delay:5: -0.197 to 0.008, against wait-only -0.021; it only learns to stop doing harm.
  * abm_utility:4: -0.467 to 0.09, against wait-only 0.002.

  Limits: the clear gain over the scripted baseline is on 1 of 3 tasks. The positive result holds for the preserved
  4.0 mechanism facilities only, since the signal is 0 in rapid families, and `ReferenceLearner()` defaults to legacy
  (`reference.py:98`). The criterion is weak (updates ≥ 1); add improvement over a frozen copy, per tier.
* **Kernel adapter REPAIR, upheld** (not raised to RETAIN). Visibility holds in a 3-agent `tiny` world: a canary in a
  mail and in a transfer memo from alice to carol reached carol and not bob, and bob's analytics did not move
  (`probe_kernel_visibility.py`). 146 + 148 read-only calls in `tiny`/`open_digital` were counted as world changes 0
  times (`probe_world_changes.py`), so `last_changed` is honest. A 31-day delayed outcome (exp.register with a 31-day
  deadline) was retrieved after an unobserved 33-day gap (553 events), linked to its own `action_seq` with `via:
  scheduled` and `delay_s` = 31 days (`probe_delayed_30d.py`). The defects that keep it at REPAIR are the original ones
  (legacy default, evaluator oracles passed through).
* **Tier 1 mechanism adapter REPAIR, upheld and sharpened.** `probe_mech_score.py`: in 15 of 25 mechanism families the
  observation's `rig.seats.<r>.score` is not net income. In scm it is output sold, a legitimate physical reading. In
  tech_* it is a creator-defined objective quantity (value delivered "while still wanted", counted only "while the
  operation pays its way", `techsys.py:61,372`). Per amendment (a), the tech_* score is goal progress and must leave
  observation-only observations, or be replaced by its observable ingredients.
* **Reward-leakage battery REDESIGN, R.2 REDESIGN, R.n1 REDESIGN: harsh, but justified.** I checked whether the
  `goals.check_now`/`independent_revenue` finding could be a legitimate observation. It is not: the netting uses
  ownership and related-party structure of other participants, which a real operator cannot see. The rapid spec leak is
  a further counterexample that key-name scanning plus mode invariance cannot catch by construction.
* **Long-horizon EXTEND, Self-experiments REPAIR, CLI EXTEND, Observation contract EXTEND: upheld.** The contract
  validator is what caught the rapid receiver errors, so the contract itself is sound. The defect belongs to the rapid
  adapter, and the original "validated in every world tried" claim is withdrawn. The Tier 1 mech digest
  (`tier1.py:434-438`) hashes the engine state, so S.8 is a real check for mech.

### S-3. Requirements added since the audit (OO.1-OO.8, workstream L)

`tests/v5/test_gate_OO.py`, which the manifest names for all eight, **does not exist**, so none can be verified as
written.

| id | verdict | reason / work |
|---|---|---|
| OO.1 | REPAIR (blocked on I/J for tiers 2/3) | 13 constructors default to legacy (original P1); flip every default, write the per-family default-mode check across ≥100 families including rapid, domains and civilisations |
| OO.2 | REDESIGN | Leaks found in observation_only: `goals.check_now` exact progress, `use.report.verified_quality`, rapid `query spec` (family/seed/split), tech_* `score`. Needs an evaluator-quantity registry, an observation-only tool policy in every adapter, and ≥8 oracle-recovery probe kinds |
| OO.3 | EXTEND | Kernel worlds expose transactions, mail, technical readings and delayed results; the rapid adapter drops causes and cost; nothing for tiers 2/3 |
| OO.4 | REDESIGN | The verifier's metric path is learner-callable (`goals.check_now`, `exp` measures over it); `verify()` lives on the same in-process object the learner drives, with no boundary; verifier-vs-ground-truth agreement is never tested |
| OO.5 | REPAIR (blocked on J) | 90 objective-free days run in `tiny` (original P8), but memory grows about 0.9 MB/day with no configured resource limits to violate or enforce; no civilisation exists for the 90-day civ criterion |
| OO.6 | EXTEND | The external adapter and ReferenceLearner supply their own signal in mech and kernel worlds; the reference signal is 0 in rapid (13/13); tiers 2/3 absent |
| OO.7 | EXTEND | Kernel: 31-day delayed result linked to its action (`probe_delayed_30d.py`); rapid adapter has no provenance; tiers 2/3 absent |
| OO.8 | REPAIR | Legacy runs reproduce, but legacy is the default, so `legacy_mode_requires_explicit_selection` is false today; `legacy_check` builds some current envs by default |

### S-4. Criteria too weak (changes to the original flags)

* **S.4:** add an episode boundary in records, and require a learner trained offline from the store to match or improve
  on the online learner, not just update.
* **S.6:** require a mid-episode save/load with an identical continuation, including the recurrent state and pending
  transitions.
* **S.7:** require worker-side timestamps and ≥4 distinct worker pids whose intervals overlap.
* **S.5 / SP-11.1:** must include multi-agent rapid receivers.

### S-5. Probes run (all exit 0)

`probe_store.py`, `probe_checkpoint_resume.py`, `probe_learning_seeds.py 3000`, `probe_world_changes.py`,
`probe_rapid_spec.py`, `probe_rapid_msg_schema_store.py`, `probe_farm_concurrency.py 1`,
`crucible.scale.slots --cpu 2 -- probe_farm_concurrency.py 2`, `probe_kernel_visibility.py`, `probe_delayed_30d.py`,
`probe_mech_score.py`, `probe_offline_boundary.py`, and
`pytest tests/v5/test_gate_S.py -k "S4 or S6"`: 2 passed in 31.7 s, demonstrating that the gates pass on the defective
paths above (log: `gate_S4_S6.log`).
