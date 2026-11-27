# CAM Live — how the integrated agent is wired (M-int, round 2: INT2)

Code: `src/cam_live/agent.py` (`CamLiveAgent`, `build_agent`, `register_agent_heads`), `src/cam_live/validation.py`
(`run_staging`, `StagingComputer`, `run_episode`, `MetaStepProbe`), the runtime (`src/cam_live/runtime/`).
Tests: `tests/integration/` (M-int: `test_agent_runtime.py`, `test_agent_recovery.py`; INT2: `test_int2_wiring.py`,
`test_int2_checkpoints.py`, `test_int2_grounding.py`), `tests/runtime/test_int2_verifier_origin.py`.
Evidence: `docs/evidence/validation/` (staging runs, `learning/` repeated-task experiment, `separated/`).
Change requests: `docs/change_requests/M-int.md` (round 1), `docs/change_requests/INT2.md` (round 2).

## 1. One learner, composed

`CamLiveAgent(cfg, run_dir, clock=..., task=..., verifier=...)` builds, in this order, and owns nothing else:

| object | component | notes |
|---|---|---|
| `model` | `nn.model.build_model(cfg)` | Qwen3-4B (frozen, bf16) + LoRA + latent bridge + 50m core by default; `CamLiveConfig.debug()` = tiny random LM + debug core (CPU tests only) |
| `heads` | `predict.heads.attach_heads(model)` | world model, latent transition, goal evaluator, candidate policy, downstream-VOI head (group `heads`) |
| `store` | `credit.replay.SqliteExperienceStore` | `<run_dir>/learner/experience.sqlite` (WAL, one writer; history append-only since D-14) |
| `predictions`, `experiments` | `predict.registry` | hash-chained, `<run_dir>/learner/{prediction,experiment}_registry.jsonl` |
| `meta` | `reward.metalearner.MetaLearner` | registers `IntrinsicReward` as head `reward` (group `reward`); owns the only optimizer of that group; ledger `<run_dir>/learner/outcome_ledger.jsonl`; every assigned objective is registered with its origin (C-10) |
| `linker` | `credit.links.OutcomeLinker` | outcome / event linking, de-duplication, re-training queue |
| `rudder`, `cocoa` | `credit.providers` | registered with the learner; credit only for closed objectives |
| `assigner` | `credit.providers.CreditAssigner` + `credit.select.EstimatorSelector` | the agent trains with `assigner.default` (currently `simple`: neither RUDDER nor COCOA passed D's held-out gate) |
| `learner` | `credit.learner.ContinualLearner` | B's `default_providers`, RUDDER, COCOA, retention, bridge grounding; `holdout_experiments=False` unless configured (INT2-2) |
| `predictor`, `decider`, `planner`, `proposer`, `goal_free` | predict / reward | as in round 1; the planner's imagined values only with `agent.planner` (no selection rule uses them) |
| `ckpts` | `credit.checkpoint.CheckpointManager` | `<run_dir>/learner/promoted`: promotion gated on independent validation + retention, rollback (§6) |

`build_agent(cfg, run_dir, clock=None, task=None, verifier=None, **settings)` is the factory of
`runtime.loop_service`; the signature and the `AgentAPI` surface are unchanged from round 1. New optional hooks the
runtime calls when present: `set_action_validator(env.validate_action)` and `export_weights(dest)`.

## 2. Runtime wiring (INT2 item 1)

| defect (round 1) | now | where | test |
|---|---|---|---|
| M-int-6: the runtime consumed the single-use grant before `env.execute`; `ComputerEnvironment` then refused every action ("the grant does not verify") unless the runtime was given `LookOnlyGrants` | `runtime.loop.look_grant` asks `check_grant(..., consume=False)` (plain call for a two-argument `PolicyEngine`) before execution, in in-flight recovery and in the approval path; the executor consumes. `LookOnlyGrants` is gone | `runtime/loop.py` | `test_int2_wiring.py::test_look_grant_leaves_the_grant_to_the_executor`, `::test_runtime_executes_in_the_computer_environment_without_grant_refusals` |
| M-int-1: `loop_service` did not pass the verifier | `run_service` calls the factory with `verifier=` (only factories that accept it get it) | `runtime/loop_service.py` | `::test_loop_service_passes_the_verifier_to_the_agent` |
| M-int-3 / F-11 b: verifier measurements unsigned | `SignedVerifier` signs measurements (F); the agent checks them with `check_measurement` before they become verifier-trust items or reach the meta ledger; the agent's own goal-metric question (§4) is answered by the verifier and arrives signed | `agent.py` | `::test_signed_measurements_supervise_and_forged_ones_do_not` |
| M-int-2: `cam-live validate --mode staging` had no options | `--debug/--cpu` or `--real-model`, `--decisions`, `--out`, `--seed`, `--evidence-dir` | `runtime/cli.py` | `::test_cli_validate_passes_its_options` |
| F-11 a: `learner_view` | the agent needs nothing else (`measurement` empty, `success` = metric + target) | — | `::test_agent_receives_the_learner_view` |
| C-12: origin not signed | `VerifiedOutcome.origin` (additive contract field) is set from the verified task and covered by the signature | `contracts.py`, `runtime/verifier.py` | `tests/runtime/test_int2_verifier_origin.py` |
| C-11: goal-free reservations | goals proposed but refused before execution, or left from an earlier proposal, are abandoned; `abandon_pending` after `load` | `agent.py` | goal-free test of round 1 (`test_agent_recovery.py`) |

## 3. Candidates grounded in the observed interface (INT2 item 3; spec Part 5, 12A items 2, 6, 7)

The round-1 agent proposed element-less clicks and fills: element references were walker vocabulary, mixed across
observations, and the task's site addresses were not in `learner_view`'s resources. Now, per decision:

1. **Surface.** The latest observation that lists interactive elements (`data.elements` with `ref`) is the current
   surface: its observation id, tab, elements (role, name, value, options, href), links, tables, tabs. A page without
   elements replaces it too (`agent.surface_of`).
2. **Grounded candidates** (`agent.grounded_candidates`): every available tool with side effects whose schema has a
   `ref` property is instantiated on each element of the surface with `ref` + that observation's `obs_id` + its `tab`,
   plus — only where the element supplies it — one value of a further argument: one of the element's own options
   (first of `label` / `option` / `value`), an observed string for a text field (numbers for a number field), the
   element's link target for a `url`. A tool some element supplies a value for (fill, select, download) is never
   instantiated bare; a tool whose required argument the element cannot supply is not instantiated on it; read-only
   tools stay page-level. Observed strings = table cells, option labels, field values, link texts, element names.
   Nothing site- or task-specific; provenance `grounded`.
3. **Other tools**: the schema walker with vocabulary from the task's authorized resources (URLs) and the latest
   observations (page URLs, link targets, tab ids) — element refs are no longer vocabulary.
4. **Skills** (`_skill_candidates`): `skill.run` for every validated skill the library reported (`skill.list`,
   `skill.induce` results) with arguments from its schema and observed values; `skill.induce` over the agent's own
   executed actions between two *verified* rises of the objective metric (§4), offered once per trace.
5. **Feasibility**: modalities the task does not permit, actions longer than the objective's time left, and what
   `EnvironmentAdapter.validate_action` (runtime hook) refuses for the current interface are dropped.
6. **Backbone**: the backbone writes JSON actions over the tools of the candidates the trainable policy ranks highest
   (at most `proposer_tools`), reading the compact digest of the current surface (`[ref] role "name" = value
   {options} -> href`, observation id) and the latest results, conditioned on the belief `b` through the bridge.
7. **Offer** (`_offer`): C_t = up to `backbone_slots` backbone proposals + up to `skill_slots` skill candidates
   (the initial modality preference of 12A.7) + `grounded_share` of the remaining slots for grounded candidates,
   the rest for the walker's; in each group `policy_share` go to the candidates the trainable policy ranks highest in
   the current context, the others are drawn uniformly at random. The inclusion probability of every candidate is
   recorded as `decision.diagnostics["proposal_probs"]` (untracked when a backbone proposal is in the set, D-14).
   Then the trainable policy selects among C_t as before (rule `policy`).

## 4. Which questions are asked about a candidate before it is executed

(a) templates of the environment's own declared measurements for that operation, (b) `action.succeeded`,
(c) `observed.<metric>` for up to three reported metrics — as in round 1 — and new (d) **the objective metric**
(`success.metric` of the learner view) after the action, `goal_relevant=True`: the runtime queues it, the
independent verifier measures it and signs the reading, the record holds it as a pending query (so the store
attaches the answer). Uses of the signed reading: verifier-trust outcome item in the context; measurement utility
in the meta ledger; the decision's `reward_signals.verified_progress_delta` annotation (signed reading minus the
previous signed reading: the bootstrap's verified term, which was always 0 in a runtime run, INT2-1); detection of
verified progress for skill induction.

## 5. What is learned at which timescale

| timescale | what changes | trigger |
|---|---|---|
| fast | `FastBeliefState` (Bayesian last layer over `phi`) | every matured outcome of a registered prediction |
| frequent | experiential core, encoders, trunk, feature map, fast prior; world / evaluator / credit heads | `learner.poll()` |
| frequent (policy) | policy head + decision trunk | `meta.policy_update` every `policy_update_every` rewarded decisions |
| slow | LoRA adapters + latent bridge | every `adapter_every` frequent updates, held-out guard |
| meta | `R_psi` (reward group) | verified, signature-checked query evidence of >= `meta_min_independent_outcomes` groups |
| checkpoint | promotion of the learner checkpoint, or rollback to the last promoted one | every `promote_every` frequent updates and at the end of every objective |

`AgentSettings.learning=False` (controlled experiments only) turns every weight, belief and reward update off.

## 6. Checkpoints (INT2 item 2; spec 10.3, 15.2, 15.3)

* **Promotion** (`CamLiveAgent.promote`): `credit.checkpoint.promote_learner` saves the learner (all trainable groups,
  learner optimizers + the meta optimizer, fast state, lifetime context, task, reward version) into the manager and
  asks for promotion with (i) held-out validation losses of the providers (`ContinualLearner.validate`, each with its
  declared provenance — a provider trained on the agent's own reward never counts) and (ii) measured retention on
  fixed retention sets: the first `retention_records` executed records of every objective, frozen with the policy's
  action distributions at that moment; metrics `<family>@<objective>/neg_world_loss` and `/decision_agreement`
  (provenance `retention`). Gates: `PromotionGates(max_validation_drop, max_retention_drop)`. Internal reward is
  ignored by the manager whatever it is called.
* **Rollback**: a refused promotion restores the last promoted checkpoint (weights, optimizers, schedules, fast
  state); experience, counters and registries are never rolled back; the refused checkpoint stays on disk.
* Both are `LearningUpdateRecord(timescale="checkpoint")` in the runtime's learning log (weight deltas of a rollback).
* **Export**: `agent.export_weights(dest)` and `cam-live export-checkpoint` (running: operator command → the same
  hook; not running: the committed checkpoint through `export_weights_only` with the configuration stored in it)
  write `weights.safetensors` + `config.json` only; `load_exported_model(dest, register_heads=register_agent_heads)`
  rebuilds the model in a fresh process (policy distribution identical to 1e-6 on the held-out probe).

## 7. Measured

(see the sections below; every number is from a file under `docs/evidence/`)

## 8. Not yet validated / known limits

(see below)
