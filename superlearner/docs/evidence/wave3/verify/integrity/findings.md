# H verifier findings (integrity area)

## F1 (medium, OB-03/IR-09) ExperienceAdapter reaches the operator, world kernel and verifier through closures
- superlearner/src/superlearner/cam_unified/envs/experience.py:289-303 (`adapter()` builds `_ActorFns` lambdas: `reset=(lambda i=i: self.reset_world(i))` closes over the ExperienceOperator; `execute/observe/... (lambda ..., c=c: ...)` default arg `c` = [world env, agent]).
- Check run (CPU): make_env("rapid", ["dl_delayed_returns#4"]) -> `ad._fns[0].reset.__closure__[0].cell_contents` is ExperienceOperator; `.private_verify(0)` returns verify/references/family/seed; `ad._fns[0].execute.__defaults__[0][0].verify()` returns the verifier record.
- tests/cam_unified/envs/test_reward_integrity.py:159 `test_factory_adapters_reach_no_operator_evaluator_or_label_source` passes only because `_walk` follows attributes, not `__closure__`/`__defaults__`; the docstring claim (experience.py:6-9 "no reference to the world object, the verifier...") is false.
- Fix: give the adapter a narrow proxy object per world (a small class with only observe/tools/execute/observe_since/advance/done methods holding the env via a private handle owned by the operator, reset routed through a one-way callback registry), or run rapid worlds behind the same process boundary as isolated OpenWorld; extend `_walk` to follow `__closure__`, `__defaults__`, `__self__`, and bound-method `__func__`.

## F2 (medium, OB-01) CamConfig default reward_mode is "legacy"; only dom presets are observation_only
- superlearner/src/superlearner/cam_unified/config.py:174 `reward_mode: str = "legacy"`; runtime.py:638 takes the agent's cfg mode, so any non-dom config (make_config dev/research/superlearner, CamConfig()) runs in legacy delivery when an evaluator is passed; learn/integrated.py:476-483 accepts the undeclared CamConfig "legacy" default with a note instead of refusing it.
- Check run: CamConfig().reward_mode, make_config('dev'|'research'|'superlearner').reward_mode -> 'legacy'; make_preset('dom').reward_mode -> 'observation_only'.
- Fix: CamConfig default "observation_only"; legacy only via explicit declaration (declared_conditions / reward_mode=); keep pre-seam tests on an explicit legacy fixture.

## F6 (medium, IR-04) Canonical-loop meta step drops the score terms of the phi_K actions taken while waiting for query labels
- superlearner/src/superlearner/cam_unified/learn/meta_window.py:104-130: in phase "wait" the new fresh steps (acted with phi_K, which depends on psi) are computed by `_new_steps` and discarded; only the `window_rounds` query steps enter `query_batch`, while U_verified is the utility of the whole episode (rapid: 60 ticks; dom_dev window 8 rounds), most of it acted before the window (slow / phi_0 / phi_1) or after it (phi_K, omitted).
- Check: code trace (on_round: `_steps.extend(new)` only in phases inner/query; `_query_steps = steps` frozen at the end of the query window).
- Fix: keep appending wait-phase steps to `_query_steps` (they are phi_K on-policy), or end the query window at the episode boundary (start the query window at an episode start and run to resolution); record per-row the fraction of the labelled episode acted with phi_K.

## F7 (low, IR-05) V_L is trained on verified remaining utility but never used by the meta objective
- superlearner/src/superlearner/cam_unified/learn/meta_window.py:139 `fp.query_batch(w, U)` (no baseline) and selfeval/metagrad.py:349-354 (MetaRewardTrainer passes no `lifetime_value`), so the outer advantage uses a leave-one-out mean of U, not V_L (plan §2: -log pi (U_ver - V_L)); RewardModel.losses trains `lifetime_value` (reward_model.py:282-284) with no consumer.
- Fix: pass `baseline={env: V_L(lifetime state at the query start)}` to query_batch and bootstrap truncated lifetimes with V_L.

## F8 (HIGH, IR-04 / OB-07) Learner episode ordinals mix two episode-id schemes: the meta step pairs query actions with the NEXT episode's verified utility; sparse_verified (D2) rewards are misjoined the same way
- learn/integrated.py:562-584 `on_step` registers the canonical loop's EventRecord.episode_id (runtime.py:762/837 `new_id("ep_")`) in `_register_episode`; learn/meta_window.py:139-146 `_utilities` and learn/integrated.py:631-632/756 (`_nodes` -> sparse_verified) register FreshStep.episode_ids, which the agent assigns itself (agent.py:497/515 `new_id("ep_")` on reset). Each id scheme takes its own ordinal from the same per-env counter, so the agent's id of episode k gets ordinal k+1 (or later).
- Check run (real dom_dev, build_agent + IntegratedLearner/MetaWindowDriver, run_lifetime, 4 rapid lever worlds horizon 12, update_every 2, meta_every 1, 40 rounds; /tmp/claude-0/verify_integrity/real_check.py -> real_run.json): query window = rounds 7-8 (episode 1), but the outer step used query_U = [0.00278, -0.0644, 0.0559, -0.0196] = the episode-2 labels (episode-1 labels were [-0.0232, -0.0345, 0.0186, -0.0213]); the window stayed open 22 rounds waiting for episode 2 (11 suspended slow updates); 112/160 records have extra["label_episode"]=2 while learner.episode_ordinal=3.
- Fix: register and look up episodes by the runtime stamp only (EventRecord.extra["label_episode"], already copied into agent records via STAMP_KEYS) - store it on FreshStep (or map agent episode id -> label episode in on_step from records whose action_id matches) and use it in `_utilities` and `_rewards(sparse_verified)`; add a regression test with a real CamUnifiedAgent asserting query utilities come from the query steps' own episode.

## F5 (medium, IR-04) Inner-score constant c is computed from a label set that includes the current query episodes
- learn/meta_window.py:140-144: `base = mean(utility of every label in the store)` is taken after waiting for the query labels, so c depends on the current query data (metagrad.py docstring requires c independent of inner and query data); when the store holds only the query episodes' labels, u - c == 0 exactly and the inner-data score term (bias correction for K>=2) vanishes.
- Check: code trace + real run (store had 8 labels incl. the 4 query labels at the outer step; c = -0.0107 vs u = -0.0063).
- Fix: compute c from labels resolved before the meta window opened (snapshot at meta_step), or use the trainer's running u_mean from previous outer steps.

## F4 (medium, IR-03/IR-04) The inner-update reward is not the actor reward: no cost term, so lambda_c (a psi member) never gets a meta-gradient and rho is meta-trained with the cost signal masked
- selfeval/fast_policy.py:131-137 `cam_reward_fn` -> `reward_psi(R, R_next, goal, dt, info_gain, done)` (no `cost=`), _reward_inputs (fast_policy.py:207-213) carries no cost; the slow actor reward (learn/integrated.py:711-733) passes `cost=self.cost_by_event[...]`, which also fills the rho "cost" signal slot.
- Check run (real dom_dev meta step): per-parameter meta-gradient norm log_lambda_c = 0.0 (log_lambda_ig 0.216, intrinsic.mlp.fc2.weight 614.8).
- Fix: carry the per-decision observed spend (FreshStep event ids -> learner.cost_by_event, or ExecutionResult.cost stored on FreshStep) into Window/_reward_inputs and pass `cost=` in cam_reward_fn; assert slow and inner reward functions agree on a stored window.

## F3 (medium, IR-01/SE-01 wiring) The reward model is never attached to the acting model: Phi is absent from the decision evaluator's Q and decision stamps carry psi_version None
- model.py:1602 `attach_reward_model` has no caller anywhere (grep over superlearner/src); build.py:345 creates the RewardModel but never calls `sysm.model.attach_reward_model(reward_model)`; decision/evaluator.py:176-181 then uses potential=None (Phi = 0) in Q_k and in policy_from_features of the meta step.
- Check run: real_run.json "model_reward_model_attached": false.
- Fix: in build_agent call `model.attach_reward_model(reward_model)` (by reference, as designed), and test that decide's stamp carries the psi version.

## Confirmed clean (real dom_dev, canonical loop; real_run.json)
- Transition.rewards is None on all 40 rounds; every EventRecord.reward == 0.0.
- psi isolation: tensor hooks on all 14 psi tensors fired only inside the meta outer gradient (13 fires), never during the slow steps; psi unchanged across both slow updates; UpdateInfo actor_psi_grad_norm = 0.0.
- Meta pathway end to end: MetaWindowDriver opened at update 1, K=2 on-policy inner steps acted with phi_k, outer step changed psi (L2 0.0161), version 0 -> 1, phi_K committed; the next slow update computed the actor reward under psi_version 1 (reward std 0.148, stub False); L_pred trained (Phi/V_L changed; label_join 64 rows, 62 grad rows).
- Canaries (label meta string, evaluator reference values 1234.56789123 / 98765.4321987, every released utility) absent from 2.76 MB of agent inputs (act batches, tools, records, next_obs), transcript, archive (176 records) and traces (168 lines); positive control: label store holds the label canary.
- gc reachability from the agent object (184k objects): operator, label source, label store, learner, world env and adapter all unreachable.
- All 50 train-split rapid families (CPU scan, family_scan.py/json): no family name / seed / reference value in actor-side content; the 25 "spec.objective" privileged-key flags are the objective text already given as the GOAL observation; one numeric hit (des_inventory 9608.051) is the actor's own cash under a do-nothing script, equal to the null-policy value by construction.
- CPU acceptance tests in area: 98 passed (envs adapter/integrity/labels, selfeval heads/metagrad/label_join/exf, credit meta exact, autonomy, integrated learner).
