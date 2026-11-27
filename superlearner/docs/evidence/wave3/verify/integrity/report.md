I verified this area independently on the real dom_dev model through the canonical loop. The checks you asked for hold: actor and trainer channels are separate, labels reach only the trainer, R_psi has no actor-gradient path, and the meta-gradient pathway works end to end. But the canonical-loop meta step trains psi on the wrong episode's verified utility (F8, HIGH), and seven other defects follow (F1-F7).

**Setup.** The real run was `build_agent("dom_dev")` + `IntegratedLearner.from_system` + `MetaWindowDriver`, through `runtime.run_lifetime`, observation_only. It used 4 Crucible rapid lever worlds (horizon 12), update_every=2, meta_every=1, 40 rounds, under gpu_heavy.lock.
- Script: /tmp/claude-0/verify_integrity/real_check.py
- Output: /tmp/claude-0/verify_integrity/real_run.json
- Family leakage scan: /tmp/claude-0/verify_integrity/family_scan.py, results in /tmp/claude-0/verify_integrity/family_scan.json
- CPU acceptance tests: /tmp/claude-0/verify_integrity/pytest_cpu.log (98 passed, 1 deselected)
- Findings log: /tmp/claude-0/verify_integrity/findings.md

## Findings, most severe first

**F8 (HIGH; IR-04, OB-07): the learner gives two episode-id schemes ordinals from one counter, so the meta step pairs query actions with the next episode's label, and the sparse_verified rewards are misjoined the same way.**
- Where:
  - `learn/integrated.py:562-584` (`on_step` → `_register_episode`) registers the runtime's `EventRecord.episode_id` (runtime.py:762/837).
  - `learn/meta_window.py:139-146` (`_utilities`) and `learn/integrated.py:631-632/756` (`_nodes` → sparse_verified) register `FreshStep.episode_ids`, which the agent assigns itself (agent.py:497/515 `new_id("ep_")`).
  - So the agent's id for episode k gets ordinal k+1.
- Check run:
  - The query window was rounds 7-8, inside episode 1.
  - The outer step used query_U = [0.00278, -0.0644, 0.0559, -0.0196]. That is exactly episode 2's labels; episode 1's were [-0.0232, -0.0345, 0.0186, -0.0213].
  - The window stayed open 22 rounds waiting for episode 2, suspending 11 slow updates.
  - 112 of 160 records have `extra["label_episode"]=2` while `learner.episode_ordinal=3`.
- Fix:
  - Key episodes only by the runtime stamp `extra["label_episode"]`. Carry it on FreshStep, or map agent id → stamp in `on_step` via action_id.
  - Use it in `_utilities` and in `_rewards` for sparse_verified.
  - Add a real-agent regression test that query utilities come from the query steps' own episode.

**F1 (medium; OB-03, IR-09): the actor-side ExperienceAdapter can reach the operator, the world kernel and the verifier through closures.**
- Where: `envs/experience.py:289-303`. The `reset` lambda closes over the ExperienceOperator, and `execute`/`observe` take a default argument `c` = [env, agent].
- Check run (CPU): on `make_env("rapid", ["dl_delayed_returns#4"])`:
  - `ad._fns[0].reset.__closure__[0].cell_contents.private_verify(0)` returns family, seed and references.
  - `ad._fns[0].execute.__defaults__[0][0].verify()` returns the verifier record.
- The test `envs/test_reward_integrity.py:159 test_factory_adapters_reach_no_operator_evaluator_or_label_source` passes vacuously: `_walk` follows attributes only, not closures or defaults. The docstring at experience.py:6-9 is false.
- Mitigation that holds: the agent object itself reaches none of these (gc check below).
- Fix:
  - Replace the lambdas with a narrow per-world proxy, or put rapid worlds behind a process boundary.
  - Extend `_walk` to follow `__closure__`, `__defaults__`, `__self__` and `__func__`.

**F2 (medium; OB-01): `CamConfig.reward_mode` defaults to "legacy" (`config.py:174`).**
- `runtime.py:638` takes the agent's configured mode, so CamConfig() and every non-dom scale run with legacy reward delivery when an evaluator is passed.
- `learn/integrated.py:476-483` accepts that undeclared default with a note instead of refusing it.
- Check run: CamConfig(), dev, research and superlearner all give legacy; only the dom presets give observation_only.
- Fix: default to observation_only; allow legacy only when explicitly declared.

**F3 (medium; IR-01/SE-01 wiring): `model.attach_reward_model` (`model.py:1602`) is never called anywhere.**
- `build.py:345` builds the RewardModel but never attaches it. So Phi = 0 in the decision evaluator's Q_k (`decision/evaluator.py:176-181`) and in `policy_from_features` during the meta step, and decision stamps carry no psi version.
- Check run: `model_reward_model_attached` = false.
- Fix: call `model.attach_reward_model(reward_model)` in `build_agent`, and test the stamp.

**F4 (medium; IR-03/IR-04): the inner-update reward is not the actor reward.**
- Where: `selfeval/fast_policy.py:131-137` (`cam_reward_fn`) and `:207-213` (`_reward_inputs`) pass no cost. The slow actor reward (`learn/integrated.py:711-733`) passes the observed cost.
- Effect: lambda_c never receives a meta-gradient, and rho is meta-trained with its cost signal masked.
- Check run: real per-parameter meta-gradient norm for log_lambda_c = 0.0 (log_lambda_ig 0.216).
- Fix: carry the per-decision observed spend into the Window and pass `cost=`; assert the slow and inner rewards agree on a stored window.

**F5 (medium; IR-04): the inner-score constant c includes the current query labels.**
- Where: `learn/meta_window.py:140-144` takes the mean over the whole store after the query labels have arrived.
- Effect: c depends on the current query data. When the store holds only the query labels, u - c = 0 and the bias-correction term vanishes.
- Check run: code trace, plus the real run (c = -0.0107 over 8 labels including the 4 query labels; u = -0.0063).
- Fix: snapshot c when the meta window opens, or use `mrl.u_mean`.

**F6 (medium; IR-04): actions taken with phi_K while waiting for query labels drop out of the outer score function.**
- Where: `learn/meta_window.py:104-130` discards the wait-phase steps. U is the utility of a whole episode, mostly acted before the query window or after it with phi_K.
- Check run: code trace only.
- Fix: append the wait-phase steps to the query batch, or align the query window to a full episode.

**F7 (low; IR-05): V_L is trained but never used.**
- Where: `meta_window.py:139` calls `query_batch` without a baseline, and MetaRewardTrainer gets no `lifetime_value` (`metagrad.py:349-354`). The outer advantage uses a leave-one-out mean of U instead of U_ver - V_L.
- Check run: code trace only.
- Fix: pass V_L as the query baseline and as the bootstrap for truncated lifetimes.

## Confirmed clean on the real model (real_run.json)
- **Reward channel:** `Transition.rewards` is None on all 40 rounds; every `EventRecord.reward` is 0.
- **psi has no actor-gradient path:**
  - Tensor hooks on all 14 psi tensors fired only inside the meta outer gradient (13 fires) and never during slow steps.
  - psi was unchanged by both slow updates; UpdateInfo `actor_psi_grad_norm` = 0.0.
- **Meta pathway works end to end (apart from the F4-F8 defects):**
  - The meta window opened, ran K=2 on-policy inner steps acting with phi_k, and the outer step changed psi (L2 0.0161, version 0 → 1); phi_K was committed.
  - The next update's actor reward ran under psi_version 1 (non-stub, std 0.148).
  - L_pred trained Phi/V_L on joined labels (64 rows, 62 with gradient).
- **Labels reach only the trainer:**
  - The canaries were a label-meta string, evaluator reference values (1234.56789123 / 98765.4321987) and every released utility.
  - None of them appear in 2.76 MB of agent inputs (act batches, tools, records, next_obs), the transcript, the archive (176 records) or the traces (168 lines).
  - Positive control: the label store does hold the label canary.
- **The agent object reaches no operator-side object:** a gc walk from the agent (184k objects) reached none of the operator, label source, label store, learner, world env or adapter.
- **No family, seed or reference values in actor content across all 50 train-split rapid families.** Two kinds of scanner hit were checked and are not leaks:
  - The "spec.objective" flags are the objective text the agent already gets as its GOAL observation.
  - The one numeric hit (des_inventory 9608.051) is the agent's own cash under a do-nothing script, which equals the null-policy value by construction.

## Status the evidence supports (manifest status → supported)
- **OB-01** IMPLEMENTED → IN_PROGRESS: true for experience/factory and the dom presets, false for CamConfig and non-dom scales (F2). Tests: `envs/test_experience_adapter.py::test_observation_only_is_the_default_and_legacy_needs_declaration`, `learn/test_integrated_learner.py::test_observation_only_is_the_default_and_baselines_must_be_declared`.
- **OB-02** IMPLEMENTED → INTEGRATION_TESTED for the rapid tier and ow_tiny. Mech and kernel families are only sampled. Tests: `test_no_family_seed_reward_or_verifier_leakage_across_families`, `envs/test_reward_integrity.py` (all passed), `integration/test_dom_e2e_real.py::test_no_private_world_identity_in_actor_side_content`. Evidence: real_run.json, family_scan.json.
- **OB-03** IMPLEMENTED → keep IMPLEMENTED. Labels are released only after resolution and the agent cannot reach private state, but the adapter boundary claim fails and its test is vacuous (F1).
- **OB-04** NOT_STARTED → NOT_STARTED (run_lifetime has no goal_free; the meta_suite prerequisite is false).
- **OB-05** IMPLEMENTED → IMPLEMENTED (tests pass; label_join drops pending labels).
- **OB-06** NOT_STARTED → IN_PROGRESS: ow_tiny OpenWorld runs observation_only through run_lifetime (docs/evidence/dom/W1/e2e_dom_dev.json, stage1/dom_e2e.py); not the full kernel tier.
- **OB-07** NOT_STARTED → IN_PROGRESS: runtime legacy/sparse_verified modes and IntegratedLearner D1/D2 exist (`test_legacy_condition_uses_the_verified_increment_and_sparse_condition_the_joined_label`), but D2 labels are misjoined in the real loop (F8).
- **OB-08** NOT_STARTED → IMPLEMENTED: the split guard refuses held-out val/test families and seeds (`test_split_guard_refuses_heldout_worlds_in_training_streams`, `test_split_guard_refuses_heldout_openworld_tasks`, `test_evaluation_streams_build_held_out_worlds_and_tag_their_labels`, all passed; 50 train families).
- **SE-01** IMPLEMENTED → IMPLEMENTED. In the real loop L_pred trains on label_join rows (real_run.json). Phi is not attached at acting time (F3).
- **SE-02** IMPLEMENTED → IMPLEMENTED (heads; mechanics only).
- **SE-03** IMPLEMENTED → IMPLEMENTED (tests pass).
- **SE-04** IN_PROGRESS → IN_PROGRESS: held-out ECE is 0.024, but Brier skill is about 0 and the calibration slope is -5.75 (docs/evidence/dom/W8/e2e_dom_dev.json); Exp A has not run.
- **SE-05** NOT_STARTED → NOT_STARTED: `refresh_potential_target` and `frozen_target` exist but have no caller.
- **SE-06** NOT_STARTED → NOT_STARTED.
- **IR-01** IMPLEMENTED → INTEGRATION_TESTED: R_psi now consumes real latents in the canonical loop, is bounded and versioned (real_run.json; `selfeval/test_selfeval_heads.py::test_intrinsic_reward_is_bounded_versioned_and_goal_free_capable`, `selfeval/test_metagrad_cam.py::test_real_dom_dev_meta_step_through_the_lifetime_loop`). The manifest limitation "not yet consuming real latent experience" is stale.
- **IR-02** NOT_STARTED → INTEGRATION_TESTED for the dom presets: the actor is trained from the non-stub R_psi with no environment reward. Evidence: real_run.json; `integration/test_dom_e2e_real.py::test_observation_only_through_the_canonical_loop`, `learn/test_integrated_learner.py::test_observation_only_refuses_environment_rewards_everywhere`. Caveat F2.
- **IR-03** NOT_STARTED → IMPLEMENTED: functional K=2 inner update on real on-policy windows inside run_lifetime (selfeval/metagrad.py, selfeval/fast_policy.py, learn/meta_window.py; `test_meta_window_spans_rounds_suspends_the_slow_step_and_reads_verified_query_utilities`, `test_meta_step_changes_psi_bumps_the_version_and_leaves_the_slow_weights`; real_run.json). K in {1,2,4,8} not swept; inner reward differs from the actor reward (F4).
- **IR-04** NOT_STARTED → IN_PROGRESS: the pathway exists and psi changes, but the verified-utility pairing is wrong (F8) and the estimator is biased (F5, F6).
- **IR-05** NOT_STARTED → IN_PROGRESS: V_L is trained on verified remaining utility but unused (F7).
- **IR-06** IMPLEMENTED → IMPLEMENTED: the three `credit/test_credit_meta_exact.py` tests pass, plus docs/evidence/dom/W8/meta_gradient_unbiased_2000rep.json. On the real model, the psi gradient appears only through the inner update (hook evidence). A detach ablation was not run on the real model.
- **IR-07** NOT_STARTED → NOT_STARTED.
- **IR-08** NOT_STARTED → IN_PROGRESS: versioning and meta_clip exist; no checkpoint promotion on independent validation and no rollback.
- **IR-09** NOT_STARTED → IN_PROGRESS: the agent object cannot reach operator, labels or verifier (gc check), but the adapter can (F1), and nothing tests that the learner cannot modify goals, verifier, budgets or sealed sets.
- **AU-01** NOT_STARTED → NOT_STARTED.
- **AU-02** IMPLEMENTED → IMPLEMENTED: three `autonomy/test_autonomy_goals.py` tests pass; not wired into the loop.
- **AU-03** IN_PROGRESS → IN_PROGRESS (`autonomy/test_curriculum_learning_gain.py` passes on synthetic learners; not in the loop).
- **AU-04** NOT_STARTED → NOT_STARTED.
- **AU-05** NOT_STARTED → NOT_STARTED.
- **AU-06** NOT_STARTED → IN_PROGRESS: unit coverage only (`test_trivial_unverifiable_and_duplicate_goals_earn_nothing`, `learn/test_learn_probe_meta_rl.py::test_lp_trained_agent_ignores_a_noisy_tv_that_its_belief_reacts_to`, `selfeval/test_reward_integrity_exf.py`).
- **AU-07** NOT_STARTED → IN_PROGRESS: autonomy/hindsight.py plus `test_hindsight_relabels_verified_partial_successes_without_touching_the_original_failure` (passes); the benefit has not been measured.