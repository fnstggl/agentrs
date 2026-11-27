# CAM Live — how the integrated agent is wired (M-int, round 2: INT2)

Code: `src/cam_live/agent.py` (`CamLiveAgent`, `build_agent`, `register_agent_heads`), `src/cam_live/validation.py`
(`run_staging`, `StagingComputer`, `run_episode`, `MetaStepProbe`), the runtime (`src/cam_live/runtime/`).
Tests: `tests/integration/` (M-int: `test_agent_runtime.py`, `test_agent_recovery.py`; INT2: `test_int2_wiring.py`,
`test_int2_checkpoints.py`, `test_int2_grounding.py`, `test_int2_queries.py`, `test_int2_selection.py`,
`test_int2_storage.py`, `test_int2_prompts.py`, `test_int2_planning.py`, `test_int2_policy_s3s4.py`),
`tests/runtime/test_int2_verifier_origin.py`, `tests/runtime/test_int2_storage_drain.py`.
Evidence: `docs/evidence/validation/` (staging runs, `learning/` repeated-task experiment, `separated/`, `meta_real/`,
`diagnosis/`, `s6/`).
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
6. **Backbone** (`agent.AgentProposer`, change request S-6; open-ended generation is on by default, the schema walker
   is a supplement): the backbone is asked for `proposer_k` (8) *different single* actions as JSON Lines (a sequence
   may still be one `{"steps": [...]}` line), over a bounded tool list — half of the `proposer_tools` slots rotate
   over the tools only generation can instantiate (free text, files, programs, notes, operator requests), half are
   the operations the trainable policy ranks highest — each tool shown as a compact argument list
   (`compact_tool`: name, what it does, `arg*: type range|options`), reading the compact digest of the current
   surface and the latest results, conditioned on the belief `b` through the bridge. Before the strict parser
   (`predict.candidates.parse_actions`, unchanged) `normalise_proposal` makes meaning-preserving repairs and counts
   each in the proposal stats: the program tool's own form `{"operation": "program", "arguments": {"steps": ...}}`
   is lifted to `{"steps": ...}`, argument names a closed schema does not have are dropped (e.g. `"method": "GET"` on
   `http.get`), unknown top-level fields are dropped, argument names written at the top level are moved into
   `arguments`. Nothing is invented; the environment still validates and authorizes every action.
7. **Offer** (`_offer`): C_t = the backbone's valid proposals (at least `backbone_share` = 1/2 of C_t, S-6: the
   policy, not dilution among walker candidates, decides about them) + up to `skill_slots` skill candidates (the
   initial modality preference of 12A.7) + `grounded_share` of the remaining slots for grounded candidates, the rest
   for the walker's; in each group `policy_share` go to the candidates the trainable policy ranks highest in the
   current context, the others are drawn uniformly at random. The inclusion probability of every candidate is
   recorded as `decision.diagnostics["proposal_probs"]` (untracked when a backbone proposal is in the set, D-14).
8. **Selection** (S-5): rule `policy` samples from softmax(policy logit + `value_weight` x the world model's predicted
   goal value net of cost, standardised over C_t) with the exploration floor; the probabilities are exact and the
   decision records `selection: policy+value`. `value_weight = 0` gives the policy alone. (Interim agent-side
   mechanism: once LRN's selection rule in `predict/` combines policy and uncertainty-aware value, the agent should
   pass `value_weight = 0` to avoid counting the value twice.)
9. **Schema walker**: `proposer_per_tool = 12` settings per tool, so a small discrete space (LabWorld: 4 channels x 3
   efforts) is enumerated completely — with 4 the walker offered 4 of 12 interventions and the best one was usually
   not in C_t (diagnosis, §9).

## 4. Which questions are asked about a candidate before it is executed

(a) templates of the environment's own declared measurements for that operation, (b) `action.succeeded`,
(c) `observed.<metric>` for up to three reported metrics — as in round 1 — and new (d) **the objective metric**
(`success.metric` of the learner view) after the action, `goal_relevant=True`, *unless the environment attributes
outcomes to decisions* (it declared goal-relevant attributed queries, e.g. LabWorld's adopters per intervention):
then the attributed questions are the goal-relevant ones and the level would mix earlier decisions' consequences
into a candidate's value. The runtime queues the agent's goal-relevant questions; the independent verifier measures
and signs them; the record holds them as pending queries, so the store attaches the answers. A template question
(a) that the environment's own declared measurement covers after execution (same metric, or linked to it by
`Predictor.adopt_queries`) is not kept pending: the runtime never queues a copy of a covered question
(`LiveRuntime._add_learner_queries`), so until this round that copy stayed unanswered and every intervention stayed
unresolved for the meta-learner (`MetaLearner._resolved`) and for the verified-progress annotation (§9 d;
`tests/integration/test_int2_queries.py` fails on the earlier code).

**Verified progress of a decision** (`_annotate_verified_delta`, written once as the decision's
`reward_signals.verified_progress_delta` — the stored-signal path of `reward.assemble`, INT2-1): when every
goal-relevant answer of the decision is in, its own attributed signed outcomes summed; 0 for a decision without
attributed outcomes in an objective whose environment attributes; the signed level change across the decision only
where nothing is attributed (a site whose count changes when the action executes). Round 2's first version credited
the level change to every decision, which paid waits and probes for adopters that earlier interventions brought
(measured, §9).

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

## 5a. Storage, drain, planning phase

* **Storage guard** (`cam_live.storage`, rule of 2026-11-27): the runtime checks `DiskGuard` before every checkpoint
  (`RuntimeConfig.min_free_gb` / `warn_free_gb`, from config section `storage`: 20 / 50 GB); below the minimum it
  writes nothing and pauses the run with a resource obstacle (`obstacle` + `storage_low` events, journal); the
  learner checks before every save and promotion and its refusal reaches the runtime through `learn()` (same pause).
  Runtime checkpoints: the newest `keep_checkpoints = 2`, optimizer state only in the newest; learner promotions:
  `keep_last 2` + the promoted one. Tests: `tests/runtime/test_int2_storage_drain.py`,
  `tests/integration/test_int2_storage.py`.
* **Drain after exhaustion** (S-1, `RuntimeConfig.drain_after_exhaustion`): after the decision budget, with the
  objective pending, the runtime keeps measuring due outcomes, verifying and learning without deciding until the
  verdict is terminal or the deadline passes.
* **Planning phase** (`tasks/discord_100_planning.yaml`, operator amendment; prepared, not run): the operator's
  objective verbatim (confirmed by its hash; objective text limit raised to 2000 characters), interfaces
  structured_browser / code / api / internal, member_list false, publications and Discord writes need approval.
  Notes, hypotheses and operator requests come only from open-ended generation (`internal.note`,
  `internal.request_operator`: the walker never invents text); `tests/integration/test_int2_planning.py` shows a
  backbone request with its justification becoming a valid candidate. `tests/integration/test_int2_prompts.py` scans
  every prompt template the agent sends for strategy content and checks that the sent prompt is constants +
  objective + tool descriptors + observation digest.
* **Harness/policy changes applied by INT2** (owners finished): S-3 — a network flag `none/off/false/...` is not a
  network request (`harness/auth/evaluate.py`); S-4 — `sandbox.list_files` without a path lists the workspace root
  (rule `T-CODE-LIST` in both policy files, byte-identical); `look_grant` (runtime) passes `consume=False` only to a
  `check_grant` that names the parameter.

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
  state); experience, counters and registries are never rolled back; the refused checkpoint stays on disk. What the
  gates do not judge stays (LRN-6): R_psi with the meta optimizer always, and the policy head unless its retention
  metric (`decision_agreement`) is among the reasons of the refusal (`promotions[-1]["kept"]`).
* Both are `LearningUpdateRecord(timescale="checkpoint")` in the runtime's learning log (weight deltas of a rollback).
* **Export**: `agent.export_weights(dest)` and `cam-live export-checkpoint` (running: operator command → the same
  hook; not running: the committed checkpoint through `export_weights_only` with the configuration stored in it)
  write `weights.safetensors` + `config.json` only; `load_exported_model(dest, register_heads=register_agent_heads)`
  rebuilds the model in a fresh process (policy distribution identical to 1e-6 on the held-out probe).

## 7. Measured

Every number is from a file under `docs/evidence/` (named in each line). LabWorld is a diagnostic world; the
staging sites are local services; nothing here is evidence about the Discord objective.

**7.1 Separated services (`cam-live run` as root, staging mode)** — `evidence/validation/separated/`
`separated_20261127T183623Z.json` (run 2; run 1 `...181748Z` refused navigation because the staging site was not a
registered local service, which the run script now does through `opctl local-service add`): learner under uid 999
(`camlearn`), browser 997, sandbox 996, auth 995, verifier 994; grounded browser actions executed by the executor;
**0 grant refusals**; signed outcomes accepted by the agent; 27 non-zero learning updates; journal; clean stop
(`exhausted` after 20 decisions; objective not achieved, debug core).

**7.2 Grounded computer use, repeated staging task (debug scale)** — `evidence/validation/learning/`
`debug_store_entry_20261127/` (4 arms x 4 seeds x 12 objectives): the agent completes an unfamiliar two-step form task
in 9–33 % of objectives with grounded browser actions; it does **not** improve with repetition (learn_primitive
0.42 -> 0.25, change CI [-0.29, -0.04]); no skill was registered (9 induce attempts failed validation), so
primitive-only vs skills shows nothing attributable to skills. Debug scale (tiny random LM): a diagnostic, not a
conclusion about the real configuration.

**7.3 Reward meta-updates, real configuration** — `evidence/validation/meta_real/meta_real_20261127T184014Z.json`
(Qwen3-4B + 50m core, 946 s under the GPU lock, peak 10.4 GB; 5 LabWorld lifetimes): 63 meta-update calls,
**4 applied with non-zero meta-gradients** (norms 0.050, 0.831, 1.837, 2.886), psi deltas 0.06–0.10, importance
weighted (ESS 2.1–2.9), internal reward on the support decisions changed by 0.017–0.067, and recomputing the same
inner update with the new psi changes the policy by TV 0.004–0.027 (the meta-step changes the next inner update);
56 skipped for insufficient independent outcomes. Whether the changed reward makes later decisions better is not
established.

**7.4 LabWorld F1 at debug scale, before / after the wiring fixes** — `evidence/validation/diagnosis/
debug_f1_20261127/` (same world seeds per arm): verified progress 0.31 [0.07, 0.58] (round-2 code before the
diagnosis) -> 0.72 [0.43, 0.96] (complete enumeration of small candidate spaces, attributed verified progress,
selection with the predicted goal value) and 0.79 [0.58, 0.95] with a larger live policy step. Mechanisms in §9.

**7.5 Executable open-ended proposals, real configuration (S-6)** — `evidence/validation/s6/probe_sw_20261127.json`
(`python -m cam_live.eval.proposal_probe --family SW --problem rle --rounds 4`, snapshot of d55986d, Qwen3-4B + 50m):
the backbone wrote 16 actions in 4 rounds (prompt 3.6 k characters), **16 / 16 parsed as valid candidates** (6 needed
the program form lifted, nothing else normalised, 0 rejections; before the S-6 work: no backbone-written candidate in
15 decisions, `docs/change_requests/S.md` S-6). What it wrote: programs of `internal.inspect` steps, notes restating
the task, a `wait`, and in every round a `internal.request_operator` asking for the capability to write
`solution.py`. `sandbox.write_file` / `sandbox.python` are among the task's 16 tools; whether they were among the
12 shown in a round's prompt is not recorded by the probe (half the slots rotate at random over the tools only the
backbone can instantiate). **The software objective was not attempted** by any backbone proposal in this probe.
Through the runtime (`s6/oe_seq_20261127.json`, `eval.oe_runner sequence --plan SW:rle,MECH:7`, same snapshot; each
objective stopped by the runner's ~790 s lock budget at 110–150 s per decision): SW — 7 decisions, all executed, 3
chosen from the backbone (`wait`, `internal.inspect`), `solution.py` never written; MECH — 6 decisions, 5 executed, 5
chosen from the backbone (`http.get` measurements at points it chose, notes of the values, one `program` refused by
the authorization layer because the parser gave it its first step's modality: INT2-9, fixed in the agent since).
Valid-and-executed: 12 of 13 chosen actions, 7 of 8 chosen backbone actions.

**7.6 Tests** — whole default suite of commit 4b46a97, one pytest process per test directory (module-name clashes,
INT2-5), from a frozen copy: **1467 tests, 1377 passed, 90 skipped, 0 failed, 0 errors**, 619 s wall (1629 s summed),
JUnit `evidence/test_results/INT2_full.xml`, summary in `docs/status/M-int.json` (`measured.full_suite`). An earlier run
at 07be9ba under load ~55 had 7 errors (900 s per-test guard on the runtime module's session, now 2400 s) and 1 reward
test failure fixed by LRN since.

## 8. Not yet validated / known limits

* **No live service was contacted.** The read-only live Discord observation run (`evidence/validation/discord_live/
  run_live_observe.py`) is prepared but was **not executed**: its launch was refused by the session's permission
  system, and it needs the operator's explicit permission. The planning-phase task (`tasks/discord_100_planning.yaml`)
  is prepared, not run.
* **Learning quality is not established.** Debug scale: no improvement with repetition on the staging task (§7.2);
  LabWorld gains in §7.4 come from wiring fixes, not from learning over lifetimes. Real configuration: the meta
  pathway works (§7.3) but whether R_psi's changes improve decisions is not measured; no repeated-task learning run
  of the real configuration was made (GPU time).
* **Skills**: induction never produced a validated skill in these runs, so skills-vs-primitive is not evaluated.
* **Meta-update evidence is scarce by design**: every decision of a lifetime with a signed verdict belongs to that
  lifetime's independent group, and the importance-weighted estimator needs ESS >= 2 per lifetime; at test scale a
  meta-update needs ~5 lifetimes with outcomes that mature inside them (§9 e). Outcomes still pending when an
  objective *completes* are not measured (drain covers only an exhausted decision budget).
* **Save / load**: restored update events rebuilt their support batches from the store, so a restored agent's
  meta-step differed (INT2-8, measured §9 f); fixed by LRN in `reward/` (5ad3777: events still to be differentiated
  are checkpointed with their batch; experiment assignment no longer depends on the random experiment id).
* **Selection rule**: the default is `thompson_policy` since LRN-1 (this round). The real-model meta evidence
  (§7.3) was recorded under rule `policy` + value prior, before that change.
* **Bootstrap inputs** `epistemic_after` / `learning_progress` are still not written (LRN-3, not blocking): those two
  bootstrap terms are 0 in runtime runs.
* INT2-6 (verified term saturates at the reward bound) and INT2-7 (live policy step) are addressed in LRN's files
  (trust-region step, unsaturated verified term); not re-measured by INT2 on the real configuration.

## 9. Diagnosis: why LabWorld choices were near random, and wiring defects found (measured)

(a) **Candidate generation** — the schema walker offered 4 of the 12 interventions (4 evenly spaced settings per
tool); complete enumeration of small spaces is the largest single effect (verified progress 0.31 -> 0.65,
`debug_f1_20261127/`). (b) **Verified reward** — the verified term never had an input in runtime runs (INT2-1); the
first fix credited the objective level change to the decision in flight and paid waits (0.57) and probes (0.33)
more than interventions (0.43); crediting attributed outcomes only pays interventions alone (0.43 vs 0.001).
(c) **Live policy step** — one update moves the selection probabilities by < 0.01 (16 candidates, uniform 0.0625;
one SGD step of 0.1 moves a probability from 0.0826 to 0.0852, `sep_debug.json`; INT2-7). (d) **Unanswerable
duplicate questions** (found this round) — each intervention kept, besides the environment's declared measurement,
the agent's own copy of the same question, which the runtime never queues. `diagnosis/meta_evidence_20261127.json`
(`meta_evidence.py`, the runtime test's configuration): before the fix (ecaec2b) 23 of 24 interventions carried such a
copy and **0 of 24** were resolved for the meta-learner; 27 of 29 meta-update calls skipped
`insufficient_independent_outcomes:0<3`. After it (5a43737): no copies; 2 of 17 resolved (the other interventions'
6-h attribution windows had not matured when their lifetimes ended). (e) **Meta-update evidence in the CPU test** —
same file: with 5 lifetimes, drain after the decision budget and 30-min windows, 23 of 28 interventions resolve but no
meta-update is applied (30 skips for fewer than 3 independent groups — every decision of a lifetime with a signed
verdict belongs to that lifetime's group — and 5 `effective_sample_size_too_small` with 3 query decisions per
policy update); with policy updates every 6 decisions, 1 of 39 calls is applied (non-zero meta-gradient). The runtime
test now uses that configuration (passed 2 of 2 runs of the module); the margin is small. (f) **Restored meta events**
(INT2-8): `diagnosis/inner_batch_rebuild.py` — rebuilt support batches differ from the cached ones in features,
signals, bootstrap reward and normaliser state for every event but the newest (`inner_batch_rebuild_20261127.txt`);
fixed by LRN (5ad3777). (g) **Rollbacks undid meta-steps** (LRN-6, LRN's measurement): applied with the policy-retention
exception (§6).
