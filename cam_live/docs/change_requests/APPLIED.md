# Change requests: applied / declined (integration engineer M-int)

Contract version after these changes: `contracts.CONTRACT_VERSION = "1.1.0"`. Every change is additive: new fields
have defaults, new protocol members are documented as optional where older implementations lack them, and no
existing signature lost a parameter. Verified by running the default suite of every test directory against the
committed tree plus these files only (command and result in `docs/evidence/validation/contract_suite.txt`).

| Request | Decision | Where | Note |
|---|---|---|---|
| A-1 | applied | `config.py` | `SCALE_PRESETS` `d_ff` corrected to the measured values (debug 64, 5m 320, 20m 576, 50m 1024, 150m 2048, 500m 4096, 1b 5504); `CoreConfig.d_ff = 1024`. `nn.presets.resolve_core` is now the identity on presets; `NN_PRESETS` keeps the `sem_rank` overrides. Built models are unchanged (they already used the corrected values). |
| A-2 | not master code | — | Architecture wording; A's implementation (zero-initialised cross-attention bridge) is what runs. |
| A-3 | applied | `nn/api.py`, `learning_api.py` | `tokenize(..., grad_last=None)` in the abstract signature; `load_trainable_state` documents `"mismatched"`; `encode_decisions(..., grad_last=None)` passes it through (only with `with_grad_semantic=True`). |
| A-4 | no change requested | — | Wording note only. |
| A-5 | no change requested | — | Recorded limitation (categorical outcomes are head-only). |
| B-CR-1 | applied | `contracts.py` | `PROGRAM_OPERATION = "program"`. |
| B-CR-2 | applied | `contracts.ExperimentRecord` | `arm_probs`, `assignments`, `test`. |
| B-CR-3 | applied | `contracts.PredictionRecord` | `registered_at: Optional[float] = None`. |
| B-CR-4 | applied | `contracts.Decision.exploration` | Comment lists `predict.policy.RULES`. |
| B-CR-5 | no change requested | — | B derives exposure / categories with `build_targets`. |
| B-CR-6 | applied | `contracts.ExperienceRecord.queries`, `learning_api.matured_outcomes` | The agent stores the questions it asked itself there (immediate execution outcome, later metric readings). |
| B-CR-7 | applied | `learning_api` | `DecisionBatch.with_grad_semantic` (set by `encode_decisions`), `grad_last`, `batch_kwargs` documented as an optional provider attribute. |
| B-CR-8 | **declined** | — | A dataclass field `OutcomeObservation.imagined` is set to `False` by `__init__` and shadows the class-level `imagined = True` marker that B's own tests (`test_registry.py::test_statuses_values_and_imagined_outcomes`, `test_world.py::test_build_targets_masks_pending_and_refuses_imagined`) use; both failed with the field. `predict.imagination.is_imagined` (flag, `imagined_horizon_steps`, provenance) plus the `source` check remain the refusal path; the agent uses them. Re-propose together with a change of those two tests. |
| B-CR-9 | **declined** | — | `BackboneConfig.tiny_vocab = 4096` changes the seeded tiny backbone of every workstream: `tests/nn/test_api.py::test_permuted_belief_changes_the_decision_latent` (KL 8.1e-4 < 1e-3) and `tests/nn/test_core.py::test_state_is_gated_input_dependent_and_carried` (RMS 0.45 < 0.5) failed with it. Configurations that need distinct digits set `backbone.tiny_vocab` themselves (B's `labdata.debug_config`, the integration tests). |
| B-CR-10 | not master code | — | Addressed to A (`nn/fast.py`). |
| B-CR-11 | **declined** | — | Copying the descriptors' estimated cost into `lab_candidates` changes the encoded actions (the action encoder reads `est_cost`) and with them the seeded results of D's tests: `tests/credit/test_retention.py::test_promotion_needs_validation_and_retention_and_a_refusal_rolls_the_learner_back` failed with it (forgetting -0.060 instead of > 0.02) and passes without it (both measured on the committed tree). Code that needs costs uses `predict.candidates.schema_candidates`; the integrated agent fills missing estimates from the tool descriptors itself. |
| C-1 / F-5 | applied | `contracts.OutcomeObservation.signature`, `OutcomeVerifier.check_measurement`, `diagnostics/world.LabVerifier` | `LabVerifier.measure` signs its observations (HMAC over the record with `signature=""`, `verified` included) and `check_measurement` checks them. Callers use `getattr`: `runtime.verifier.SignedVerifier` / `RemoteVerifier` do not sign measurements yet (filed as M-int-3). |
| C-2 | applied | `contracts.TaskSpec.origin` | Default `Trust.OPERATOR`. Changes `TaskSpec.content_hash()` (consistently for every process on this code). |
| C-3 | applied | `contracts.MetaUpdateRecord` | `ledger_head`, `window_pair`. |
| C-4 | applied | `contracts.ExperienceRecord.reward_inputs` | |
| C-5 | applied | `contracts.ExperienceStore.iter_records` | `SqliteExperienceStore` and `diagnostics.memstore.MemoryStore` implement it. |
| C-6 | applied as optional | `nn/api.LatentModelAPI.decision_trunk()` | Concrete method returning `None` ("not addressable") so `CamLiveModel` (A) stays instantiable; A can override it with `self.exp.trunk` (filed as M-int-4). C's `AdaptableSubset` keeps reading `model.exp.trunk`. |
| C-7 | applied | `contracts.ExperienceRecord.lifetime` | The agent writes its lifetime index. |
| C-8 | applied in the agent | `agent.py` | The agent calls `MetaLearner.policy_update(records)` (the exact functional SGD the unroll differentiates) and does not register `PolicyImprovementLoss` with the continual learner. |
| C-9 | information | — | |
| D-1 | applied | `contracts.ExperienceStore` | `sample(..., provenance=)`, `add_outcome`, `next_due`, `state_dict`, `load_state_dict`; `MemoryStore` implements all of them. |
| D-2 | applied | `contracts.ExperienceRecord.eligible` | "no entry = eligible". |
| D-3 | applied | `diagnostics/memstore.py` | `put_item` keeps the first version. |
| D-4 | applied | `contracts.PrimitiveOp.step_id` | |
| D-5 | applied | `contracts.Record.from_dict` | Type hints cached per class (`functools.lru_cache`). |
| D-6 | no change requested | — | |
| D-7 | applied (a) / declined (b, c) | `diagnostics/collect.py` | (a) with `seq_start=0` the collector writes the position after the store's `head_seq()` when the store has one; `MemoryStore` positions are unchanged. (b) linker and (c) randomised arms: `credit.labkit` already provides them, and the integrated agent goes through `OutcomeLinker` and `ExperimentRegistry` itself. |
| D-8 | applied | `learning_api.encode_decisions` | Same change as A-3. |
| D-9 | applied | `learning_api.LossProvider` docstring | |
| D-10 | applied | `contracts.ExperienceStore` | `put_objective_outcome`, `objective_outcome`; called by the agent only after the verifier's signature was checked. |
| D-11 | applied | `contracts.LearningUpdateRecord.diagnostics` | |
| D-12 | **declined** | — | The world's query metric keeps the action id: `predict.heads.canonical_query` (`{action}` placeholder) and `credit.labkit._canonical` already canonicalise it, B's diagnostics are seeded on the current metric names, and `OutcomeQuery.decision_id` is a decision id, not an action id. |
| EA-1 / EB-1 / ES-1 | applied | `contracts.ExecContext` | `authorization`, `granted_action`, `dispatch`, `workspace` declared (the environment already sets them). |
| EA-2 | applied | `contracts.PolicyEngine.check_grant` | Comment: called once by the executor, consumes the grant. |
| EA-3 | applied | `harness/__init__.py` | `SERVICE_ROLES` (same mapping as `launcher.DEFAULT_ROLES`). |
| EA-4..8 | out of scope here | — | Not in this integration's list (config file, pyproject — already declares `cryptography` —, test conftest, rpc, executor integration). |
| EB-2 / EC-2 | applied (merged) | `contracts.SecretHandle` | One protocol in the form EA implements: `get(name, action, decision, destination)`, `scrub(text)`; optional `has(name)` and `bind(action, decision)`. EC's `get(name, op, action)` form is not what the vault offers and EC never calls `get`. |
| EB-3 / F-3 | applied | `contracts.PolicyEngine.record` | Comment: called once per executed action by the environment; the runtime does not call it. |
| EB-4..7 | out of scope here | — | |
| EC-3 | applied | `contracts.VerificationCondition` docstring, `VisualGrounder` docstring | Spec keys by modality; optional `heavy` / `close()`. |
| ES-2 | contract part applied | `contracts.SecretHandle` | `has(name)` documented as optional; the implementation is in `harness/auth/vault.py` (auth owner), filed as M-int-5. |
| F-1 | applied | `contracts.AgentAPI` | `drain_experiments()`, `explain_obstacle(obstacle)`; `CamLiveAgent` implements both. |
| F-4 | applied | `contracts.EnvironmentAdapter` comment, `diagnostics/world.py` | `features.at_most_once_by_action_id` documented; `LabConfig(at_most_once=True)` gives LabWorld a result cache by action id and announces it (off by default: diagnostics re-execute one `ActionSpec` to advance time). |
| F-6 | applied | `contracts.AgentAPI.load` / `record_result` comments | `CamLiveAgent.record_result` is idempotent and accepts a later result for a decision that was queued for approval. |
