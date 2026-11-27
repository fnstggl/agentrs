# CAM Live — how the integrated agent is wired (M-int)

Code: `src/cam_live/agent.py` (`CamLiveAgent`, `build_agent`), `src/cam_live/validation.py` (`run_staging`).
Tests: `tests/integration/test_agent_runtime.py`, `tests/integration/test_agent_recovery.py`.
Evidence: `docs/evidence/validation/`. Contract changes: `docs/change_requests/APPLIED.md`.

## 1. One learner, composed

`CamLiveAgent(cfg, run_dir, clock=..., task=..., verifier=...)` builds, in this order, and owns nothing else:

| object | component | notes |
|---|---|---|
| `model` | `nn.model.build_model(cfg)` | Qwen3-4B (frozen, bf16) + LoRA + latent bridge + 50m core by default; `CamLiveConfig.debug()` = tiny random LM + debug core (CPU tests only) |
| `heads` | `predict.heads.attach_heads(model)` | world model, latent transition, goal evaluator, candidate policy, downstream-VOI head (group `heads`) |
| `store` | `credit.replay.SqliteExperienceStore` | `<run_dir>/learner/experience.sqlite` (WAL, one writer) |
| `predictions`, `experiments` | `predict.registry.PredictionRegistry` / `ExperimentRegistry` | hash-chained, `<run_dir>/learner/{prediction,experiment}_registry.jsonl` |
| `meta` | `reward.metalearner.MetaLearner` | registers `IntrinsicReward` as head `reward` (group `reward`); owns the only optimizer of that group; ledger `<run_dir>/learner/outcome_ledger.jsonl` |
| `linker` | `credit.links.OutcomeLinker` | outcome / event linking, de-duplication, re-training queue |
| `rudder`, `cocoa` | `credit.providers.ReturnDecompositionProvider` / `HindsightProvider` | registered with the learner; credit only for closed objectives |
| `assigner` | `credit.providers.CreditAssigner` + `credit.select.EstimatorSelector` | default estimator chosen by randomised evidence (`credit_selection.jsonl`) |
| `learner` | `credit.learner.ContinualLearner` | providers: B's `default_providers` (world model, fast prior, evaluator, world-model adapters, evaluator adapters), RUDDER, COCOA, `RetentionProvider(AnchorSet)`, `BridgeGroundingProvider`; one AdamW per group experiential / heads / semantic_adapters |
| `predictor` | `predict.predictor.Predictor` | given `learner.fast_state` — one fast belief state for prediction, VOI and the fast update |
| `decider` | `predict.policy.DecisionMaker` | exact probabilities of the whole candidate set; randomised experiments through the experiment registry |
| `planner` | `predict.planner.LatentPlanner` | used only when its `HorizonController` allows >= 1 measured-accurate step |
| `proposer` | `predict.candidates.BackboneProposer` | backbone text conditioned on the current belief `b` through the bridge, merged with the schema walker (and skill procedures listed as tools) |
| `goal_free` | `reward.selfgoals.GoalFreeMode` | budgeted self-chosen learning goals when the task has no objective; its own fast state |

`PolicyImprovementLoss` is **not** registered with the continual learner (change request C-8): the agent calls
`MetaLearner.policy_update(records)`, the exact functional SGD that the meta-gradient differentiates, every
`agent.policy_update_every` newly rewarded decisions. `MetaLearner.check_learner_optimizers` asserts at construction
that no learner optimizer holds a reward parameter.

`build_agent(cfg, run_dir, clock=None, task=None, verifier=None)` is the factory of `runtime.loop_service`.

## 2. Per-step data flow (runtime.loop.LiveRuntime, spec 15.1)

| step | runtime | agent |
|---|---|---|
| 1 | task | `set_objective(task)`: same id = new boundaries only; new id = retention anchors from the finished objective, `meta.new_lifetime()`, `learner.reset_fast()` (fast beliefs back to the learned prior), fresh context with a goal item. Weights, reward parameters, learned priors, templates and the store carry over. |
| 2-3 | `env.observe()`, `env.poll_events()` | `ingest`: one experience item per observation (exact quantities kept), events through `linker.reconcile(events=...)` (attribution data in the event), readings that answer the agent's own metric questions |
| 4 | due queries measured by the verifier (goal-relevant) or the environment; signed objective verification | `reconcile`: `linker.reconcile(outcomes)` (store, de-duplicate, link, queue), `predictor.resolve` (registry), outcome items into the context (trust `verifier` only when `check_measurement` authenticates them), experiment outcomes. A `VerifiedOutcome` is used only after `verifier.check_signature` here: `meta.admit_objective`, `EvaluatorLoss.add_outcome`, and for achieved / failed objectives `store.put_objective_outcome`, `predictor.resolve_objective`, re-training of the trajectory |
| 5 | | `update_beliefs`: `predictor.observe_outcomes` on every matured outcome (the fast Bayesian update; a `fast` learning record), current `V(b, g)` |
| 6 | `env.discover_tools()` | `propose(tools, k)`: tools item, belief `b = core(H_t)`, `BackboneProposer.propose(tools, digest, latent=b, vocab=...)`, spread over operations; goal-free: `GoalFreeMode.propose` picks the candidates |
| 7-8 | predictions logged before execution | `evaluate`: `Predictor.evaluate_candidates` with the questions of §3 for every candidate — registered in the prediction registry before anything executes; planner values as extras when validated |
| 9 | | `select`: `DecisionMaker.select(rule)` (default `policy`: the trainable policy's distribution with an exploration floor), or with `experiment_rate` a randomised comparison of the pair whose order the model is least sure of (`propose_experiment` / `select_experiment`) |
| 10-11 | validate, authorize (PolicyService), execute or queue | — |
| 12 | result journalled | `record_result`: action + observation items, `ExperienceRecord` (H_t, H_{t+1}, full candidate set with probabilities, environment queries, the agent's own questions, immediate execution outcome), templates learned from the environment's queries, `adopt_queries` links them to the earlier predictions. Idempotent; accepts re-delivery after a restart (F-6) and the executed result of an approved action |
| 13 | | `internal_rewards`: `MetaLearner.internal_rewards(new records)` — bootstrap and learned parts, written into each record (`intrinsic_reward`, `bootstrap_reward`, `reward_version`) |
| 14 | | `learn`: fast records; `learner.poll()` → `learner.learn(plan)` (frequent experiential + heads, slow adapters every `adapter_every`, credit re-assigned for the triggering records); `meta.policy_update(...)`; periodic `planner.measure(...)` |
| 15 | | `meta_learn`: `MetaLearner.meta_update()` whenever its inputs changed — applied, or a record with the exact skipped reason |
| 16 | commit | `save(path)`: `capture_learner` (all trainable groups incl. reward, the learner's and the meta optimizer, fast state, lifetime context, replay sampler, schedules, RNGs, versions, configuration, credit selector, provider tensors) + meta-learner state and its cached support batches, predictor cache, goal-free state, scaler, planner controller, agent bookkeeping → `credit.checkpoint.save_checkpoint_at` (temp dir → fsync → rename) |
| 17 | sleep unless there is work | `next_wake()` returns `None`: the agent itself schedules nothing; outcomes are scheduled by the runtime |

`drain_experiments()` hands new / updated `ExperimentRecord`s to the runtime (experiments.jsonl);
`explain_obstacle(obstacle)` gives a Dirichlet(1) posterior over the failure classes the operation has shown.

## 3. Which questions are asked about a candidate before it is executed

Nothing task-specific: (a) **templates** — the measurements the environment reported after earlier executions of the
same operation (canonical metric with the action id replaced by `{action}`, kind, horizon, exposure bound to the
argument whose value equalled it); (b) `action.succeeded` (binary, resolved from the result); (c) `observed.<metric>`
— the next reading of up to three metrics the environment reports (the task's success metric first when it is
observed), resolved from later observations. (b) and (c) are unverified environment readings: they train the world
model and the fast beliefs, never the meta objective.

## 4. What is learned at which timescale

| timescale | what changes | trigger |
|---|---|---|
| fast | `FastBeliefState` (Bayesian last layer over `phi`) | every matured outcome of a registered prediction |
| frequent | experiential core, encoders, trunk, feature map, fast prior; world / evaluator / credit heads | `learner.poll()`: >= `min_independent_outcomes` independent matured decisions |
| frequent (policy) | policy head + decision trunk | `meta.policy_update` every `policy_update_every` rewarded decisions |
| slow | LoRA adapters + latent bridge | every `adapter_every` frequent updates, held-out guard |
| meta | `R_psi` (reward group) | verified, signature-checked query evidence of >= `meta_min_independent_outcomes` independent groups after a policy update |

## 5. Measured

Staging validation on the real configuration (`docs/evidence/validation/staging_20261127T052759Z.json`; Qwen3-4B
frozen 4,022,468,096 + experiential 50,953,765 + adapters 16,713,856 + reward 949,121 + heads 6,055,337 parameters;
459 s under the GPU lock, peak 11.35 GB): 26 decisions (10 in the computer environment with Chromium, HTTP and the
staging Discord service, 16 in two LabWorld lifetimes, both verified as achieved); 52 of 262 candidates written by
the backbone through the bridge, the rest by the schema walker; 106 predictions resolved, all strictly after they
were registered; 35 frequent updates (experiential delta norm 0.0036 – 1.58, all non-zero), 9 slow adapter updates
(0.070 – 0.172, all non-zero), 26 fast updates; internal rewards on every record; 9 delayed outcomes linked;
policy distribution on the held-out context moved by TV 0.0256, restoring all trainable groups gives TV 0.0 to the
old distribution, restoring only the experiential group leaves TV 0.0046 to it (the rest of the change is in heads and
adapters). **No meta-update was applied** (26 x `insufficient_independent_outcomes`: the two lifetimes ended after 8
decisions each when their targets were reached).

Run 2 (`staging_20261127T053639Z.json`, 778 s under the lock, peak 11.21 GB, heavy CPU load from other jobs): 23
decisions (6 computer environment, 17 in one LabWorld lifetime that stopped at its decision budget with the objective
pending; the second lifetime did not start within the time budget). 28 frequent / 7 slow / 22 fast updates, all with
non-zero deltas; TV 0.0177 on the held-out context, 0.0 after restoring all trainable groups. One meta-update is
recorded as *applied*, but with an importance-weighted meta-gradient of exactly 0 (outer objective 0.0) and an
unchanged reward version: it is **not** evidence of a reward update (the file's `meta_update_applied` check predates
the stricter check that now requires a non-zero gradient and a changed reward version).

Run 3 (`staging_20261127T055041Z.json`, 785 s under the lock while another tenant's GPU test and heavy CPU load ran):
the computer environment composed by `harness.factories.all_executors` (internal, browser, code, visual, api, skill:
80 tools), but only 9 decisions in total (2 computer, 7 LabWorld); updates in every group, TV 0.0103, no meta-update.

On the real configuration no meta-update with a non-zero meta-gradient has been observed yet; at debug scale on the
CPU the integration test requires and gets one. CPU test numbers: `docs/status/M-int.json`.

## 6. Not yet validated / known limits

* LabWorld and the staging services only: no live service was contacted; nothing here is evidence about the Discord
  objective or any real task.
* Learning *quality* is not established by these runs: the tests and the staging run show that the pathway works
  (non-zero updates, meta-gradients, changed behaviour that lives in the weights), not that the agent gets better.
* Save → fresh process → load: the first decision after the load is bit-identical, later ones choose the same
  candidates with probability vectors equal to ~1e-7 (not bitwise) and the reward-parameter hash can differ after a
  meta-update — a floating-point summation-order difference on the meta / learning path that was not located.
* `runtime.loop_service` does not pass the verifier to the factory (M-int-1): an agent started by `cam-live run`
  refuses every verified outcome for supervision until `attach_verifier` is called; `runtime.verifier` does not sign
  measurements (M-int-3), so in a real run only signed objective outcomes can supervise the meta objective.
* The planner contributes values only after its multi-step error has been measured below threshold; its values are
  recorded but not used by the selection rules.
* Goal-free mode is wired (proposal, resolution of its goals) but its usefulness for later objectives is not
  measured here.
* Checkpoint promotion / rollback (`CheckpointManager`, `promote_learner`) is not called by the agent; the runtime
  keeps its own checkpoint directory and the agent writes there.
