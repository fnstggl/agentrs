"""Generate docs/CAM_LIVE_COMPLETION_MANIFEST.json: requirement ids, owners, spec references, metadata, coverage.

    cd /workspace/camlive/cam_live && ../.venv/bin/python tools/gen_manifest.py
    cd /workspace/camlive/cam_live && PYTHONPATH=src ../.venv/bin/python -m cam_live.verify_completion --merge-status

The first command (re)writes the requirement list, the per-requirement metadata (``kind``, ``level``,
``needs_executed_evidence``, ``declared``) and the spec coverage table, keeping every requirement's merged fields.
The second folds the workstream status files and the independent verdicts into the manifest and regenerates
``docs/COMPLETION.md``. Requirement ids are stable: an id is never renumbered or reused.

Per-requirement metadata (read by the checker):

* ``kind`` — what "runtime integration point" means for the requirement:
  ``runtime`` (default): at least one integration point is import-reachable from the running system (runtime
  processes, the ``cam-live`` console script, operator CLIs, factories named in task / config files);
  ``evaluation``: reachable from the evaluation programs as well; ``diagnostic``: a controlled diagnostic program
  whose recorded command produced the evidence is enough (algorithm-correctness requirements);
  ``none``: a document or process requirement — no integration point or acceptance test is required, but the
  named files must exist and the independent verdict must be current.
* ``level`` — which of the five spec 0.3 statuses the requirement belongs to (1 engineering, 2 controlled
  experimental validation, 3 authorized live deployment readiness, 4 live experiment executed).
* ``needs_executed_evidence`` — the claim is an executed experiment / run: an evidence file must show it ran.
* ``declared`` — files / tests / integration points / commands / evidence that exist in the repository for a
  requirement that no workstream status file claims (filled from the code, never a state).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "CAM_LIVE_COMPLETION_MANIFEST.json")

# id | owning workstream | spec reference | title
R = """
NA-01|A|1.2, 18|A 4B-8B pretrained semantic backbone is integrated
NA-02|A|1.3, 18|Approximately 50M experiential parameters are implemented
NA-03|A|1.2, 18|Approximately 10M-20M semantic adapter parameters are implemented
NA-04|A|1.1, 18|Actual trainable parameter counts are reported per group
NA-05|A|3, 18|Shared high-dimensional latent representations function
NA-06|M|1.4, 18|The experiential intelligence influences real decisions (not a frozen-LLM wrapper)
NA-07|A|2.1, 18|Fast numerical adaptation functions (Gaussian and non-Gaussian likelihoods)
NA-08|D|2.2, 18|Experiential weights receive real learning gradients
NA-09|A|2.3, 18|Semantic adapters receive real learning gradients
NA-10|A|16.10, 18|Scaling configurations exist (5M to 1B+)
NA-11|A|3.1|Experience tokens for every item kind; exact values preserved alongside embeddings
NA-12|A|1.4|Trainable conditioning from CAM's latent state into semantic action generation
NA-13|A|2.1|Bayesian adaptation compared against recurrent adaptation and simple statistical baselines
PI-01|B|4.1, 18|Action-conditioned consequence prediction works
PI-02|B|4.2, 18|Predictions are recorded before actual outcomes
PI-03|B|4.2, 18|Actual observations supervise predictive learning
PI-04|B|4.1, 18|Multiple prediction horizons are supported
PI-05|B|4.1, 18|Uncertainty is represented (epistemic and aleatoric)
PI-06|B|4.2, 18|Pending outcomes are handled correctly (masked, never invented)
PI-07|B|4.3, 18|Alternative actions can be evaluated through learned-model imagination
PI-08|B|5.1, 18|Experimental action selection is implemented (Thompson, UCB, value, information gain, randomized)
PI-09|B|5|General-purpose dynamic action generation with candidate provenance and selection probabilities
PI-10|B|4.3|Imagined experience stays distinguishable from real and is never reinforced as an outcome
PI-11|B|4.2|Prediction misses are attributed to noise, missing observations, model uncertainty, changed conditions or wrong assumptions
SE-01|B|6.1, 18|Goal-conditioned success prediction exists
SE-02|B|6.1, 18|Eventual verified outcomes correct earlier predictions (prefix training)
SE-03|B|6.2, 18|Distributional outcome prediction exists (magnitude, time, resources; counts and censoring)
SE-04|B|6.4, 18|Prediction calibration is measured on prospectively registered predictions
SE-05|B|6.3, 18|Action quality is distinguished from raw outcome
SE-06|B|8, 18|Information value is distinguished from surprise (VOI and learned downstream-usefulness estimator)
MR-01|C|7.2, 18|CAM generates its own internal learning rewards
MR-02|C|7.2, 18|The reward function is trainable
MR-03|C|7.3, 18|The reward function influences actual inner updates
MR-04|C|7.3, 18|Differentiable inner updates function
MR-05|C|7.4, 18|Outer meta-gradient optimization functions
MR-06|C|7.4, 18|Independently verified outcomes supervise meta-learning
MR-07|C|7.7, 18|Meta-gradient correctness tests pass (finite differences, relative error, K in 1,2,4,8)
MR-08|C|7.6, 18|Cold-start learning functions without Crucible (labelled bootstrap mode)
MR-09|C|7.8, 16.6, 18|Reward-integrity tests exist and execute
MR-10|C|7.5, 18|Real-world meta-learning can begin as evidence accumulates (windows within one lifetime)
MR-11|C|7.2, 7.8|The policy cannot change reward parameters, success criteria or verification records
RC-01|D|9.1, 18|Delayed outcomes are linked to earlier experiences with uncertainty
RC-02|D|9.2, 18|RUDDER-style credit redistribution is implemented
RC-03|D|9.2, 18|Return conservation is verified under the discounting convention
RC-04|D|9.3, 18|COCOA reference implementation is validated
RC-05|D|9.3, 18|Counterfactual credit estimates support uncertainty and refuse unsupported inference
RC-06|D|9.2, 18|Earlier experiences receive retrospective updates when outcomes mature
RC-07|D|9.1, 18|Future information cannot leak into historical decisions
RC-08|D|9.4|Default credit estimator is selected by evidence; RUDDER and COCOA are exercised
CL-01|D|10.1, 18|Persistent experience replay functions
CL-02|D|10.2, 18|Actual neural updates occur and change predictions or behaviour
CL-03|D|10.2, 18|Adapter updates occur
CL-04|D|10.3, 18|Retention tests function
CL-05|D|10.3, 18|Checkpoint promotion and rollback work
CL-06|D|15.2, 18|Complete restart and recovery are verified
CL-07|S|16.8, 18|Fresh-process weights-only transfer evaluation functions
CL-08|D|2|Three learning timescales are scheduled with event-driven triggers
CL-09|M|10.4|Learned weights, reward and priors persist across objectives; fast beliefs are reset
CL-10|D|15.2|Checkpoints persist every 15.2 state item atomically and off the GPU (weights, adapters, reward weights, optimizer, fast state, task, replay indices, experiments, pending outcomes, schedules, seeds, authorization configuration, model versions)
AE-01|B|11.1, 18|The learner can generate its own experiments (registered hypotheses and predictions)
AE-02|M|11.1, 13.3, 18|The learner can revise its approach when evidence contradicts it
AE-03|M|11.1, 18|Self-directed learning functions end to end
AE-04|C|11.4, 18|Goal-free learning mode exists (budgeted)
AE-05|M|10.4, 18|The architecture supports subsequent unfamiliar objectives
AE-06|F|11.3, 18|Future distributed actor-learner operation is supported
AE-07|S|11.2|Shadow policy comparison without public interaction
AE-08|M|12B|Autonomous obstacle resolution through the learned loop, with obstacle records
OB-01|M|obstacle resolution 1-3|A failure is observed and characterised, its possible explanations carry probabilities, and it is classified as technical, informational, strategic, resource-related or permission-related
OB-02|M|obstacle resolution 4-7|Alternatives to a failed approach are generated, their consequences and costs predicted, an authorized alternative is selected and executed and its result observed
OB-03|M|obstacle resolution 8-10|Obstacle experience updates fast beliefs, generates internal learning incentives and is consolidated into persistent weights
OB-04|M|obstacle resolution|No hardcoded failure-recovery decision tree; permission and platform restrictions stay hard constraints that are never retried or routed around
IH-01|EB|12.1, 18|The common environment contract is implemented
IH-02|EB|12.2, 18|Browser integration is available
IH-03|EC|12.3, 18|Sandboxed terminal execution functions
IH-04|ES|12.4, 18|Authorized publishing interface exists (draft, preview, approval, publish)
IH-05|ES|12.5, 18|Discord integration contract exists (official bot interfaces only)
IH-06|ES|12.5, 18|Permitted aggregate analytics ingestion works
IH-07|EA|12.6, 18|Credentials remain outside model observations
IH-08|EA|12.6, 18|Tool permissions are independently enforced
IH-09|EA|12.6, 18|Approval controls function
IH-10|EA|12.6, 18|Spending and resource limits function
IH-11|EA|12.6, 18|Emergency stop functions
IH-12|EB|12.7|Adapter implementation guide for replacing the browser or tool harness
IH-13|EB|12.1, 12A.15, 13.1|The environment returns observations and consequences only: no handcrafted scalar reward or bonus for completing an operation
IH-14|EA|12.6|Platform-compliance checks refuse every prohibited class of 12.6 (spam, deceptive identities, artificial membership, join-for-join, conditional payments, circumvention, credential leakage, unauthorized data collection, profiling, manipulation)
IH-15|EA|12.5, 12.6|Data-use permissions: message content and personal information are neither stored nor trained on without explicit permission; aggregate metrics otherwise
IH-16|EA|12.6|Independent hash-chained audit log of authorization decisions, not writable by the learner
IH-17|EA|12.6|Revocation of access (scopes, destinations, credentials, outstanding grants)
CU-01|EB|12A.1, 12A.17|Persistent Linux computer environment
CU-02|EB|12A.2, 12A.17|Full Chromium browser
CU-03|EB|12A.2, 12A.17|Native Playwright browser control with the structured primitives
CU-04|EB|12A.2, 12A.17|External Chromium (CDP) connection support
CU-05|EB|12A.2, 12A.17|Structured accessibility and DOM observations with stale-reference detection
CU-06|EC|12A.3, 12A.17|Screenshot-based observations
CU-07|EC|12A.3, 12A.17|Lightweight visual-grounding model integration (pinned, reviewed)
CU-08|EC|12A.3, 12A.17|Optional advanced GUI model integration
CU-09|EC|12A.3, 12A.17|Validated mouse and keyboard execution
CU-10|EC|12A.1, 12A.17|Sandboxed Python and shell execution
CU-11|EB|12A.6, 12A.17|General-purpose typed action interface
CU-12|B|12A.6, 12A.17|Dynamic action generation over discovered tools
CU-13|EB|12A.6, 12A.17|Multiple interchangeable execution modalities behind one interface
CU-14|EB|12A.13, 12A.17|Persistent authorized browser sessions
CU-15|ES|12A.8, 12A.17|Automatically generated executable skills
CU-16|ES|12A.8, 12A.17|Independent skill validation
CU-17|ES|12A.8, 12A.17|Skill parameterization and reuse
CU-18|ES|12A.8, 12A.17|Detection, repair or retirement of outdated or failed skills
CU-19|ES|12A.9, 12A.17|Temporally extended actions (primitive, procedure, intervention)
CU-20|D|12A.9, 12A.17|Duration-aware credit assignment; an outcome is counted once across timescales
CU-21|EB|12A.5, 12A.17|Shared observation and experience representation across modalities
CU-22|M|12A.10, 12A.17|Actual neural learning from interaction experience, independent of the skill library
CU-23|EC|12A.11, 12A.17|Concurrent bounded research and computation (background jobs)
CU-24|F|12A.11, 12A.17|Delayed-feedback handling without redundant activity
CU-25|EA|12A.14, 12A.17|Independent permission enforcement across every execution modality
CU-26|EA|12A.14, 12A.17|Prompt-injection resistance tests
CU-27|EA|12A.13, 12A.17|Credential isolation tests
CU-28|EB|12A.15, 12A.17|Complete execution logging
CU-29|EB|12A.17|Restart and session recovery of the computer environment
CU-30|M|12A.17|End-to-end computer-use tasks on unfamiliar authorized interfaces
CU-31|ES|12A.7|Adaptive execution-modality selection calibrated from experience
CU-32|ES|12A.12|Execution-failure taxonomy, bounded retries, no retry or workaround of denials
CU-33|S|12A.17|Primitive-only operation compared with acquired skills (success, operations, cost, latency)
CU-34|EC|12A.3|Visual models benchmarked on representative controlled computer-use tasks
CU-35|EB|12A.17|Controlled computer-use tests cover structured HTML, visual controls, dynamic changes, multi-step workflows, unfamiliar interfaces, repeated tasks, interface changes and delayed consequences
CU-36|EB|12A.11|Operations that modify shared external state are coordinated: a duplicated or concurrent action executes at most once
CU-37|EA|12A.14, 12A.1|Learner, browser, code sandbox, credential service and independent verifier run as separated processes under separate uids
CU-38|EC|12A.3, 12A.4|GUI and visual models only propose low-level operations; the harness validates them and CAM's policy decides
LX-01|F|13, 18|The Discord objective exists as external task configuration
LX-02|F|13.1, 18|No predefined Discord-growth strategy is embedded (checked mechanically)
LX-03|F|13.2, 18|Independent membership verification functions
LX-04|M|13.1, 18|The model can autonomously propose community concepts
LX-05|M|13.3, 18|The model can research and generate original useful resources
LX-06|M|13.3, 18|Trial-and-error learning functions
LX-07|F|14.2, 18|Structured experiment logging functions
LX-08|F|14.3, 18|Neural learning-update logging functions
LX-09|F|14.4, 18|Research journal generation functions
LX-10|F|15.3, 18|Monitoring interface and operator controls function
LX-11|M|19.4, 18|Staging validation passes
LX-12|F|19.5, 18|Live deployment preflight passes or explicitly identifies missing authorization or integrations
LX-13|F|14.1|Decision records are written for each consequential decision
LX-14|F|13.4|Experiment boundaries are explicit and enforced; exhaustion checkpoints and pauses
LX-15|F|15.1|Persistent event-driven runtime loop implements the 17 steps
LX-16|F|12B|CAM Live runs as an independently persistent process
LX-17|F|13.2|Secondary metrics (seven-day retention, voluntary participation, consented feedback, resource usefulness, operating cost, time) are measured separately from the primary objective
LX-18|M|obstacle resolution and live execution, 19.6|After explicit live authorization CAM itself receives the objective and pursues it until verified success, a stop condition or a genuine external blocker; the development agent does not perform the task
LX-19|F|13.4|After verified success the experiment stops with the learner state checkpointed, or enters a separately authorized next phase
LX-20|F|14.4|Run records are durable, timestamped, carry provenance and are redacted of credentials and prohibited personal information
SV-01|S|16.1, 18|Baseline A: frozen pretrained model with no persistent learning
SV-02|S|16.1, 18|Baseline B: frozen pretrained model with written experience memory
SV-03|S|16.1, 18|Baseline C: Bayesian adaptive learner on semantic features
SV-04|S|16.1, 18|Baseline D: continual learning without learned intrinsic rewards
SV-05|S|16.2, 18|Calibration experiments execute
SV-06|S|16.4, 18|Delayed-credit experiments execute
SV-07|S|16.5, 18|Meta-gradient experiments execute
SV-08|S|16.6, 18|Reward-integrity experiments execute
SV-09|S|16.9, 18|Learning-efficiency evaluation executes (N_q, S_q, AULC)
SV-10|S|16.8, 18|Parametric-transfer evaluation executes on available independent tasks
SV-11|M|18|Real-world experimental results are reported only after genuine execution
SV-12|S|16.10|Model-scaling comparison infrastructure exists
SV-13|S|16.1|Baseline E: full CAM Live compared with A-D under matched permissions, information, experience and compute budgets
SV-14|S|16.3|Action-quality evaluation executes (highly evaluated actions versus alternatives, controlled experiments)
SV-15|S|16.7|Continual neural learning evaluation executes (before / after checkpoints, weight changes in core and adapters, changes in prediction, action quality and competence)
CM-01|F|17|cam-live doctor
CM-02|F|17|cam-live init
CM-03|F|17|cam-live test
CM-04|F|17|cam-live validate --mode staging
CM-05|F|17|cam-live inspect-harness
CM-06|F|17|cam-live run
CM-07|F|17|cam-live status
CM-08|F|17|cam-live pause
CM-09|F|17|cam-live resume
CM-10|F|17|cam-live evaluate-transfer
CM-11|F|17|cam-live export-checkpoint
CM-12|F|17|cam-live verify-completion
CM-13|F|17|All dependencies and exact commands are documented in one place
IN-01|M|18|All implementation workstreams are integrated
IN-02|G|18|Independent verification inspects actual code
IN-03|G|18|Independent verification executes required tests
IN-04|M|0.3, 18|Completion checker verifies mandatory engineering requirements
IN-05|M|18|Reproduction commands are documented
IN-06|M|18|Scientific failures and deployment blockers are explicitly recorded
IN-07|M|19.4|End-to-end pathway: observation, latent update, action, consequence, internal reward, neural update, changed behaviour
IN-08|G|12A.16|Independent verification inspects the computer-use implementation and its connection to the neural training loop
IN-09|M|0.1, final directive|CAM Live is standalone: no dependency on CAM Unified, Crucible or other unfinished components (checked mechanically)
DL-02|M|20.2|Documented model architecture
DL-04|M|20.4|Mathematical implementation notes
DL-19|M|20.19|Research report distinguishing implemented, validated, unsuccessful and unverified capabilities
DL-20|M|20.20, 19.5|Precise list of everything the operator must provide before live execution can begin
OP-01|M|operator amendment 2026-11-27 (storage)|Storage guard: checkpoint and artifact writes are refused below a free-space minimum and the run pauses instead of tearing a checkpoint
OP-02|M|operator amendment 2026-11-27 (storage)|Checkpoint retention: newest checkpoints kept, optimizer state only in the newest, protected checkpoints and records never pruned
OP-03|M|operator amendment 2026-11-27 (storage)|Disk-space monitoring for long-running work (doctor, dashboard health, runtime guard)
OP-04|ES|operator amendment 2026-11-27 (Discord)|Official-bot Discord integration verified against the real API on the operator-designated server only, with the server restored
OP-05|M|operator amendment 2026-11-27 (Discord)|Observations obtained through the Discord integration enter CAM's experience pipeline (tokens, replay, learning)
OP-06|M|operator amendment 2026-11-27|CAM can request additional capabilities or resources from the operator; the answer returns as an event and grants nothing by itself
OP-07|M|operator amendment 2026-11-27 (credentials)|Credential entry by the operator, isolated from the learner, training data and logs
OP-08|M|operator amendment 2026-11-26 (push milestones)|Durable off-instance preservation of source (secret-scanned pushes) and an explicit record of what has no durable backup
OP-09|EA|operator amendment 2026-11-27 (Discord), 12.5, 12A.13|No automation of ordinary or owner user accounts and no circumvention of CAPTCHA or human verification (hard platform constraints)
OP-10|S|operator amendment 2026-11-27 (evaluation)|Substantive evaluation on the full-size agent with open-ended backbone action generation and open-ended objective classes; synthetic environments count as diagnostics only
OP-11|M|operator amendment 2026-11-27 (planning)|Planning phase: CAM receives the operator's objective verbatim and produces its own research, plan and operator requests; the development agent does not make the plan
"""

#: ids that existed in the first manifest (2026-11-26); everything else was added by the reconciliation of 2026-11-27
ADDED_20261127 = ["PI-11", "CL-10", "OB-01", "OB-02", "OB-03", "OB-04", "IH-13", "IH-14", "IH-15", "IH-16", "IH-17",
                  "CU-35", "CU-36", "CU-37", "CU-38", "LX-17", "LX-18", "LX-19", "LX-20", "SV-13", "SV-14", "SV-15",
                  "CM-13", "IN-08", "IN-09", "DL-02", "DL-04", "DL-19", "DL-20", "OP-01", "OP-02", "OP-03", "OP-04",
                  "OP-05", "OP-06", "OP-07", "OP-08", "OP-09", "OP-10", "OP-11"]

KIND: Dict[str, str] = {}
for _i in ("NA-13", "MR-07", "RC-03", "RC-04", "CU-34"):
    KIND[_i] = "diagnostic"
for _i in ("SV-01", "SV-02", "SV-03", "SV-04", "SV-05", "SV-06", "SV-07", "SV-08", "SV-09", "SV-10", "SV-12", "SV-13",
           "SV-14", "SV-15", "CU-33", "CL-07", "AE-07", "OP-10"):
    KIND[_i] = "evaluation"
for _i in ("IN-02", "IN-03", "IN-05", "IN-06", "IN-08", "DL-02", "DL-04", "DL-19", "DL-20", "CM-13", "SV-11", "LX-18",
           "OP-08", "OP-11"):
    KIND[_i] = "none"

LEVEL: Dict[str, int] = {}
for _i in ("NA-13", "MR-07", "RC-03", "RC-04", "SV-05", "SV-06", "SV-07", "SV-08", "SV-09", "SV-10", "SV-12", "SV-13",
           "SV-14", "SV-15", "CU-33", "CU-34", "LX-11", "IN-07", "OP-10"):
    LEVEL[_i] = 2
LEVEL["OP-04"] = 3
LEVEL["LX-18"] = 4
#: the claim is an executed experiment or run: an evidence file must show it ran (level 2 and 3 entries, plus these)
EXECUTED = sorted([i for i, lv in LEVEL.items() if lv in (2, 3)])

# Facts about requirements that no workstream status file claims (yet). Only paths and names that exist in the
# repository at the time of writing; the checker reports any that disappear. Never a state.
_RT = "tests/runtime/"
DECLARED: Dict[str, Dict[str, List[str]]] = {
    "PI-11": {"files": ["src/cam_live/predict/errors.py", "src/cam_live/predict/predictor.py"],
              "integration_points": ["cam_live.predict.errors.classify_error",
                                     "cam_live.predict.predictor.Predictor.observe_outcomes"],
              "tests": ["tests/predict/test_imagination.py::test_change_detector_and_error_attribution"]},
    "CL-10": {"files": ["src/cam_live/credit/checkpoint.py"],
              "integration_points": ["cam_live.credit.checkpoint.capture_state",
                                     "cam_live.credit.checkpoint.save_checkpoint_at"],
              "tests": ["tests/credit/test_checkpoint.py::test_round_trip_is_bit_exact_and_training_continues_identically",
                        "tests/credit/test_checkpoint.py::test_replay_indices_and_pending_pointer_travel_with_the_checkpoint",
                        "tests/credit/test_checkpoint.py::test_crash_at_every_step_of_a_save_leaves_a_loadable_checkpoint",
                        "tests/credit/test_checkpoint.py::test_checkpoint_of_a_gpu_model_is_stored_on_cpu"]},
    "OB-01": {"files": ["src/cam_live/runtime/obstacles.py", "src/cam_live/harness/failures.py", "src/cam_live/agent.py"],
              "integration_points": ["cam_live.runtime.obstacles.ObstacleTracker.open",
                                     "cam_live.agent.CamLiveAgent.explain_obstacle", "cam_live.harness.failures.classify"],
              "tests": [_RT + "test_rt_logs_journal.py::test_obstacle_records_link_failure_alternatives_and_learning",
                        "tests/harness/skills/test_es_failures.py::test_every_failure_class_is_reachable_and_categorised"]},
    "OB-02": {"files": ["src/cam_live/runtime/obstacles.py", "src/cam_live/runtime/loop.py"],
              "integration_points": ["cam_live.runtime.obstacles.ObstacleTracker.on_decision",
                                     "cam_live.runtime.obstacles.ObstacleTracker.on_result"],
              "tests": [_RT + "test_rt_logs_journal.py::test_obstacle_records_link_failure_alternatives_and_learning"]},
    "OB-03": {"files": ["src/cam_live/runtime/obstacles.py"],
              "integration_points": ["cam_live.runtime.obstacles.ObstacleTracker.on_learning"],
              "tests": [_RT + "test_rt_logs_journal.py::test_obstacle_records_link_failure_alternatives_and_learning"]},
    "OB-04": {"files": ["src/cam_live/harness/failures.py", "src/cam_live/harness/modality.py"],
              "integration_points": ["cam_live.harness.failures.run_with_retries"],
              "tests": ["tests/harness/skills/test_es_failures.py::test_hard_classes_are_never_retried_and_cannot_be_configured",
                        "tests/harness/skills/test_es_modality.py::test_a_refusal_never_causes_a_modality_switch"]},
    "IH-13": {"files": ["src/cam_live/harness/env.py", "src/cam_live/contracts.py"],
              "integration_points": ["cam_live.harness.env.ComputerEnvironment.execute"],
              "tests": ["tests/harness/env/test_adapter_contract.py::test_adapter_returns_no_reward"]},
    "IH-14": {"files": ["src/cam_live/harness/auth/evaluate.py", "src/cam_live/harness/auth/policy.py",
                        "src/cam_live/harness/auth/default_policy.yaml"],
              "integration_points": ["cam_live.harness.auth.PolicyService.authorize"],
              "tests": ["tests/harness/auth/test_ea_policy_rules.py::" + t for t in (
                  "test_mass_outreach_recipient_count_and_destinations", "test_near_duplicate_content_across_destinations",
                  "test_direct_messages_need_opt_in", "test_deceptive_identity", "test_artificial_membership_and_join_for_join",
                  "test_paying_or_rewarding_for_engagement", "test_profiling_and_vulnerable_audiences",
                  "test_credential_leakage")] +
                       ["tests/harness/auth/test_ea_fix_policy.py::test_paraphrased_prohibited_content_never_passes_silently"]},
    "IH-15": {"files": ["src/cam_live/harness/auth/policy.py", "src/cam_live/harness/auth/default_policy.yaml",
                        "src/cam_live/harness/discord.py"],
              "integration_points": ["cam_live.harness.auth.PolicyService.authorize",
                                     "cam_live.harness.discord.DiscordExecutor.execute"],
              "tests": ["tests/harness/auth/test_ea_policy_rules.py::test_data_use_permissions",
                        "tests/harness/skills/test_es_discord.py::test_message_content_is_dropped_by_default",
                        "tests/harness/auth/test_ea_fix_policy.py::test_discord_member_events_need_the_member_list_permission"]},
    "IH-16": {"files": ["src/cam_live/harness/auth/audit.py"],
              "integration_points": ["cam_live.harness.auth.audit.AuditLog", "cam_live.harness.auth.audit.verify_audit_log"],
              "tests": ["tests/harness/auth/test_ea_audit.py::" + t for t in (
                  "test_intact_log_verifies", "test_tampering_is_detected", "test_chain_rebuilt_without_the_key_is_detected",
                  "test_appending_continues_the_chain_across_instances", "test_service_audits_every_kind_of_event",
                  "test_log_survives_restart_and_detects_edits_made_while_down")] +
                       ["tests/harness/auth/test_ea_privileged.py::test_learner_and_sandbox_uids_cannot_read_vault_audit_or_verifier_files"]},
    "IH-17": {"files": ["src/cam_live/harness/auth/policy.py", "src/cam_live/harness/auth/vault.py"],
              "integration_points": ["cam_live.harness.auth.PolicyService.revoke"],
              "tests": ["tests/harness/auth/test_ea_grants.py::test_revocation_and_stop_void_outstanding_grants",
                        "tests/harness/auth/test_ea_vault.py::test_revocation_by_scope",
                        "tests/harness/auth/test_ea_vault.py::test_revoking_a_destination_denies_it_and_its_secrets",
                        "tests/harness/auth/test_ea_policy_rules.py::test_live_record_time_window_not_started_and_revocation"]},
    "CU-35": {"files": ["src/cam_live/harness/staging/sites.py", "src/cam_live/harness/staging/canvas.py",
                        "src/cam_live/harness/staging/labsite.py", "src/cam_live/harness/staging/server.py"],
              "integration_points": ["cam_live.harness.staging.StagingSites"],
              "tests": ["tests/harness/env/test_browser.py::test_page_observation_has_the_structured_representation",
                        "tests/harness/compute/test_ec_visual.py::test_browser_click_on_a_canvas_button_changes_the_page_and_is_verified",
                        "tests/harness/env/test_browser.py::test_dynamic_changes_are_observed_as_diffs",
                        "tests/harness/env/test_browser.py::test_multi_step_workflow_on_the_structured_site",
                        "tests/harness/env/test_browser.py::test_unfamiliar_variants_and_a_mid_run_interface_change",
                        "tests/harness/skills/test_es_skills.py::test_measured_primitive_only_versus_skill_based_execution",
                        "tests/harness/skills/test_es_skills.py::test_stale_after_interface_change_then_repaired_to_a_new_version",
                        "tests/harness/env/test_browser_sessions.py::test_delayed_consequences_through_the_browser"]},
    "CU-36": {"files": ["src/cam_live/harness/env.py"],
              "integration_points": ["cam_live.harness.env.ComputerEnvironment.execute"],
              "tests": ["tests/harness/env/test_env_core.py::test_duplicate_submission_executes_once",
                        "tests/harness/env/test_env_core.py::test_concurrent_duplicates_execute_once_and_share_the_result",
                        "tests/harness/env/test_e_fix_core.py::test_one_action_id_with_two_valid_grants_executes_once_under_concurrency",
                        "tests/harness/env/test_e_fix_core.py::test_a_duplicate_during_a_long_execution_gets_in_progress_and_later_the_recorded_result",
                        "tests/harness/env/test_browser.py::test_duplicate_submission_of_a_click_executes_once"]},
    "CU-37": {"files": ["src/cam_live/harness/auth/launcher.py", "src/cam_live/harness/auth/landlock.py",
                        "src/cam_live/harness/auth/service.py"],
              "integration_points": ["cam_live.harness.auth.launcher.launch"],
              "tests": ["tests/harness/auth/test_ea_privileged.py::" + t for t in (
                  "test_launcher_starts_supervises_and_stops_services_under_their_own_uids",
                  "test_learner_and_sandbox_uids_cannot_read_vault_audit_or_verifier_files",
                  "test_rpc_peer_uid_authorization_matrix", "test_service_uids_cannot_regain_privileges")] +
                       ["tests/harness/env/test_browser_uid.py::test_chromium_runs_under_the_browser_uid_with_a_private_profile"]},
    "CU-38": {"files": ["src/cam_live/harness/visual.py"],
              "integration_points": ["cam_live.harness.visual.VisualExecutor.execute"],
              "tests": ["tests/harness/compute/test_ec_visual.py::test_grounder_output_is_only_a_proposal",
                        "tests/harness/compute/test_ec_visual.py::test_out_of_bounds_coordinates_are_rejected",
                        "tests/harness/compute/test_ec_visual.py::test_denied_by_policy_stays_denied"]},
    "LX-17": {"files": ["src/cam_live/runtime/verifier.py", "src/cam_live/runtime/tasks.py", "tasks/discord_100.yaml"],
              "integration_points": ["cam_live.runtime.verifier.MembershipVerifier.verify"],
              "tests": [_RT + "test_rt_verifier.py::test_secondary_metrics_are_separate_quantities",
                        _RT + "test_rt_tasks.py::test_discord_task_is_the_spec_objective_with_independent_criteria"]},
    "LX-19": {"files": ["src/cam_live/runtime/loop.py"],
              "integration_points": ["cam_live.runtime.loop.LiveRuntime._finish"],
              "tests": [_RT + "test_rt_loop.py::test_full_loop_reaches_verified_success"]},
    "LX-20": {"files": ["src/cam_live/runtime/logs.py"],
              "integration_points": ["cam_live.runtime.logs.RunLog"],
              "tests": [_RT + "test_rt_logs_journal.py::test_redaction_of_secrets_personal_information_and_member_identifiers",
                        _RT + "test_rt_fix_runtime.py::test_D7_text_shapes_are_redacted",
                        _RT + "test_rt_fix_runtime.py::test_D7_structured_shapes_are_redacted"]},
    "SV-13": {"files": ["src/cam_live/eval/conditions.py", "src/cam_live/eval/experiments.py"],
              "integration_points": ["cam_live.eval.conditions.build_condition"],
              "tests": ["tests/eval/test_conditions_transfer.py::test_every_condition_is_the_integrated_agent"]},
    "SV-14": {"files": ["src/cam_live/eval/analysis.py"],
              "integration_points": ["cam_live.eval.analysis.action_quality"],
              "tests": ["tests/eval/test_analysis.py::test_action_quality_detects_informative_predictions"]},
    "SV-15": {"files": ["src/cam_live/eval/conditions.py"],
              "integration_points": ["cam_live.eval.conditions.FrozenEvalAgent"],
              "tests": ["tests/eval/test_conditions_transfer.py::test_full_agent_changes_weights_and_ground_truth_is_recorded",
                        "tests/eval/test_conditions_transfer.py::test_fast_only_baseline_changes_beliefs_not_weights"]},
    "CM-13": {"files": ["pyproject.toml"]},
    "CU-33": {"files": ["src/cam_live/harness/skills.py"],
              "integration_points": ["cam_live.harness.skills.SkillExecutor.execute"],
              "tests": ["tests/harness/skills/test_es_skills.py::test_measured_primitive_only_versus_skill_based_execution"],
              "evidence": ["docs/evidence/ES/skills_vs_primitives.json"]},
    "IN-02": {"files": ["docs/verification/README.md"] + [f"docs/verification/G_{x}.md" for x in
                                                           ("cl", "env", "nn", "pr", "rt", "sec")]},
    "IN-03": {"files": ["docs/verification/repro"] + [f"docs/verification/verdicts_{x}.json" for x in
                                                       ("cl", "env", "nn", "pr", "rt", "sec")]},
    "IN-04": {"files": ["src/cam_live/verify_completion.py", "tools/gen_manifest.py"],
              "integration_points": ["cam_live.verify_completion.run", "cam_live.verify_completion.main",
                                     "cam-live verify-completion"],
              "tests": ["tests/test_verify_completion.py::" + t for t in (
                  "test_complete_needs_a_fresh_pass_and_a_verdict_issued_after_the_last_change",
                  "test_a_commit_after_the_run_and_the_verdict_makes_both_stale",
                  "test_a_change_to_the_test_file_after_the_run_makes_the_result_stale",
                  "test_an_uncommitted_edit_counts_as_a_change_made_now",
                  "test_a_newer_skipped_result_never_hides_an_earlier_pass",
                  "test_only_skipped_results_are_not_a_pass",
                  "test_a_fresh_failure_is_failed_and_a_failure_before_a_fix_needs_a_rerun",
                  "test_verdict_states_fresh_partial_failed_and_stale_failed",
                  "test_a_verdict_time_is_when_its_entry_last_changed_not_when_the_file_did",
                  "test_a_newer_verdict_supersedes_an_older_lens_only_when_it_says_so",
                  "test_a_verdict_naming_its_commit_is_fresh_while_the_files_are_unchanged",
                  "test_missing_files_placeholders_and_test_directories",
                  "test_integration_points_must_be_reachable_for_the_requirements_kind",
                  "test_console_script_commands_programs_and_symbols_resolve_or_are_reported",
                  "test_executed_evidence_must_exist_succeed_and_be_newer_than_the_code",
                  "test_document_requirements_need_files_a_claim_and_a_current_verdict_only",
                  "test_the_owners_claim_decides_and_suffixed_status_files_belong_to_their_owner",
                  "test_status_levels_are_computed_separately",
                  "test_completion_md_has_every_group_counts_and_the_regeneration_command",
                  "test_an_overridden_base_method_is_not_a_placeholder_but_an_unoverridden_one_is",
                  "test_a_no_op_override_of_a_library_hook_is_not_a_placeholder",
                  "test_markers_count_in_comments_docstrings_and_raises_not_in_string_data",
                  "test_the_manifest_covers_every_enumerated_mandatory_item_of_the_specification")],
              "commands": ["cd /workspace/camlive && .venv/bin/python -m pytest cam_live/tests/test_verify_completion.py "
                           "-q --rootdir=cam_live",
                           "cd /workspace/camlive/cam_live && PYTHONPATH=src ../.venv/bin/python -m "
                           "cam_live.verify_completion --merge-status --json /tmp/completion.json --quiet"]},
    "IN-06": {"files": ["docs/CAM_LIVE_LEDGER.md"]},
    "IN-08": {"files": ["docs/verification/G_env.md", "docs/verification/G_sec.md"]},
    "IN-09": {"files": ["src/cam_live/verify_completion.py"],
              "integration_points": ["cam_live.verify_completion.standalone_findings"],
              "tests": ["tests/test_verify_completion.py::test_the_cam_live_source_is_standalone",
                        "tests/test_verify_completion.py::test_the_standalone_scan_flags_imports_of_unfinished_projects"],
              "commands": ["cd /workspace/camlive/cam_live && PYTHONPATH=src ../.venv/bin/python -m cam_live.verify_completion --quiet"]},
    "DL-02": {"files": ["docs/CAM_LIVE_ARCHITECTURE.md", "docs/A/DESIGN.md", "docs/INTEGRATION.md"]},
    "DL-04": {"files": ["docs/CAM_LIVE_ARCHITECTURE.md", "docs/B/API.md", "docs/C/API.md", "docs/D/API.md"]},
    "DL-20": {"files": ["src/cam_live/runtime/preflight.py"]},
    "OP-01": {"files": ["src/cam_live/storage.py"],
              "integration_points": ["cam_live.storage.DiskGuard.check"],
              "tests": ["tests/storage/test_storage.py::test_guard_refuses_before_writing_and_warns"]},
    "OP-02": {"files": ["src/cam_live/storage.py", "tools/prune_runs.py"],
              "integration_points": ["cam_live.storage.apply_retention"],
              "tests": ["tests/storage/test_storage.py::test_retention_keeps_records_newest_weights_and_protected",
                        "tests/storage/test_storage.py::test_dry_run_and_drop_groups"],
              "evidence": ["docs/evidence/S/raw/prune_report_20261127.json"]},
    "OP-03": {"files": ["src/cam_live/runtime/doctor.py", "src/cam_live/runtime/dashboard.py", "src/cam_live/storage.py"],
              "integration_points": ["cam_live.runtime.doctor.storage_checks", "cam_live.storage.DiskGuard.status"],
              "tests": [_RT + "test_rt_doctor_cli.py::test_doctor_reports_what_is_on_this_machine",
                        _RT + "test_rt_dashboard.py::test_read_endpoints_show_the_run"]},
    "OP-04": {"files": ["tools/discord_live_check.py", "src/cam_live/harness/discord.py"],
              "integration_points": ["cam_live.harness.discord.DiscordExecutor.execute"],
              "tests": ["tests/harness/skills/test_es_discord_live_check.py::test_full_sequence_in_staging_leaves_the_server_as_it_was",
                        "tests/harness/skills/test_es_discord_live_check.py::test_the_documented_live_preparation_yields_grants_and_releases_the_token_only_to_discord"],
              "evidence": ["docs/evidence/discord/live_check_20261127.json", "docs/evidence/discord/README.md"]},
    "OP-05": {"files": ["src/cam_live/harness/discord.py"],
              "integration_points": ["cam_live.harness.discord.DiscordExecutor.execute"]},
    "OP-06": {"files": ["src/cam_live/harness/operator_requests.py", "src/cam_live/harness/internal.py"],
              "integration_points": ["cam_live.harness.operator_requests.OperatorRequests",
                                     "cam_live.harness.operator_requests.main"],
              "tests": ["tests/integration/test_operator_requests.py::test_request_is_recorded_deduplicated_and_answered",
                        "tests/integration/test_operator_requests.py::test_invalid_kind_and_request_flood_are_refused"]},
    "OP-07": {"files": ["src/cam_live/harness/credentials.py"],
              "integration_points": ["cam_live.harness.credentials.store", "cam_live.harness.credentials.main"],
              "tests": ["tests/integration/test_credentials_membership.py::test_store_puts_the_value_only_in_vault_and_verifier_copy",
                        "tests/integration/test_credentials_membership.py::test_bad_values_and_loose_files_are_refused",
                        "tests/integration/test_credentials_membership.py::test_cli_never_prints_the_value_and_revoke_removes_the_copy"]},
    "OP-08": {"files": ["tools/push_milestone.sh", "src/cam_live/verify_completion.py"],
              "tests": ["tests/test_verify_completion.py::test_durability_reports_unpushed_commits_and_uncommitted_files"]},
    "OP-09": {"files": ["src/cam_live/harness/auth/default_policy.yaml"],
              "integration_points": ["cam_live.harness.auth.PolicyService.authorize"],
              "tests": ["tests/harness/auth/test_ea_policy_rules.py::test_platform_rules_are_data_the_operator_edits"]},
    "OP-10": {"files": ["src/cam_live/eval/openended.py"]},
}

# ---------------------------------------------------------------------------------------------------------------
# Spec coverage: every enumerated mandatory item of the specification -> manifest ids. The test
# tests/test_verify_completion.py parses the same lists out of docs/CAM_LIVE_SPEC.md and requires an entry for each.
P18 = (["NA-%02d" % i for i in range(1, 11)] + ["PI-%02d" % i for i in range(1, 9)] + ["SE-%02d" % i for i in range(1, 7)]
       + ["MR-%02d" % i for i in range(1, 11)] + ["RC-%02d" % i for i in range(1, 8)] + ["CL-%02d" % i for i in range(1, 8)]
       + ["AE-%02d" % i for i in range(1, 7)] + ["IH-%02d" % i for i in range(1, 12)] + ["LX-%02d" % i for i in range(1, 13)]
       + ["SV-%02d" % i for i in range(1, 12)] + ["IN-%02d" % i for i in range(1, 7)])
COVERAGE: Dict[str, List[Tuple[str, List[str]]]] = {
    "18": [],                                                    # filled below from the spec text, in order
    "12A.17": [],
    "12A.17 controlled tests": [(t, ["CU-35"] + extra) for t, extra in (
        ("Structured HTML interfaces.", ["CU-03", "CU-05"]), ("Visually represented controls.", ["CU-06", "CU-09"]),
        ("Dynamic page changes.", ["CU-05"]), ("Multi-step workflows.", ["CU-03"]),
        ("Unfamiliar interfaces.", ["CU-30"]), ("Repeated tasks where learned skills should improve efficiency.", ["CU-33"]),
        ("Interface changes requiring skill adaptation.", ["CU-18"]),
        ("Delayed environmental consequences.", ["CU-24"]))],
    "17": [("cam-live doctor", ["CM-01"]), ("cam-live init --config configs/cam_live.yaml", ["CM-02"]),
           ("cam-live test", ["CM-03"]), ("cam-live validate --mode staging", ["CM-04"]),
           ("cam-live inspect-harness", ["CM-05"]), ("cam-live run --task tasks/discord_100.yaml", ["CM-06"]),
           ("cam-live status", ["CM-07"]), ("cam-live pause", ["CM-08"]), ("cam-live resume", ["CM-09"]),
           ("cam-live evaluate-transfer --checkpoint <checkpoint>", ["CM-10"]),
           ("cam-live export-checkpoint", ["CM-11"]), ("cam-live verify-completion", ["CM-12"])],
    "20": [("A fully integrated CAM Live implementation.", ["IN-01"]), ("A documented model architecture.", ["DL-02"]),
           ("Exact trainable parameter counts.", ["NA-04"]), ("Mathematical implementation notes.", ["DL-04"]),
           ("A functioning continual neural training loop.", ["CL-02", "CL-08"]),
           ("A genuine intrinsic reward meta-learning implementation.", ["MR-05", "MR-10"]),
           ("Retrospective credit-assignment mechanisms.", ["RC-02", "RC-04", "RC-05"]),
           ("Persistent experience storage.", ["CL-01"]), ("Detailed experiment and learning logs.", ["LX-07", "LX-08"]),
           ("A general-purpose internet harness contract.", ["IH-01"]),
           ("Authorized external-action controls.", ["IH-08", "IH-09", "IH-10", "IH-11"]),
           ("A persistent live runtime.", ["LX-15", "LX-16"]), ("A monitoring interface.", ["LX-10"]),
           ("The first Discord experiment configuration.", ["LX-01"]),
           ("Reproducible deployment commands.", ["IN-05", "CM-13"]),
           ("Before-and-after checkpoint support.", ["CL-05", "CL-07"]),
           ("Transfer-evaluation infrastructure.", ["CL-07", "SV-10"]),
           ("An independently verified completion manifest.", ["IN-04"]),
           ("A research report distinguishing implemented, validated, unsuccessful and unverified capabilities.", ["DL-19"]),
           ("A precise list of anything the operator must provide before live execution can begin.", ["DL-20", "LX-12"])],
    "obstacle resolution": [
        ("Observe and characterize the failure.", ["OB-01", "CU-32"]),
        ("Estimate possible explanations and their uncertainty.", ["OB-01"]),
        ("Determine whether the obstacle is technical, informational, strategic, resource-related or permission-related.", ["OB-01"]),
        ("Generate feasible alternative approaches.", ["OB-02", "PI-09"]),
        ("Predict their likely consequences and costs.", ["OB-02", "PI-01", "PI-07"]),
        ("Select and execute an authorized alternative.", ["OB-02", "PI-08", "IH-08"]),
        ("Observe the result.", ["OB-02", "LX-15"]),
        ("Update its fast numerical beliefs.", ["OB-03", "NA-07"]),
        ("Generate internal learning incentives.", ["OB-03", "MR-01"]),
        ("Consolidate useful experience into persistent neural weights.", ["OB-03", "CL-02"])],
    "obstacle resolution (rules)": [
        ("abandon strategies and invent different approaches", ["AE-02", "LX-06"]),
        ("no hardcoded failure-recovery decision tree; permission and platform rules are hard constraints", ["OB-04"]),
        ("log obstacles, explanations, attempts, outcomes, strategy changes, neural updates", ["AE-08"]),
        ("independently persistent process", ["LX-16"]),
        ("CAM itself performs the live objective after authorization; operator supplies access, not the solution", ["LX-18"])],
    "13": [("semantic objective, independent success criteria, resources, interfaces, limits, measurement, data permissions; no strategy",
            ["LX-01", "LX-02"]),
           ("no reward bonus for building a website, publishing content or making a tool call", ["IH-13", "LX-02"])],
    "13.2": [(t, ["LX-03"]) for t in ("Bots.", "Test accounts.", "Duplicate or artificial accounts where identifiable.",
                                      "Purchased memberships.", "Fabricated activity.",
                                      "Platform-prohibited engagement practices.")]
            + [(t, ["LX-17"]) for t in ("Seven-day retention.", "Voluntary participation.", "Consented participant feedback.",
                                        "Usefulness of community resources.", "Operating cost.", "Time expenditure.")],
    "13.3": [("research, opportunities, resources, tests, observation, abandonment, improvement, invention, continued learning",
              ["LX-05", "LX-06", "AE-02", "AE-03"])],
    "13.4": [(t, ["LX-14"]) for t in ("Experiment duration.", "Available compute.", "Financial budget.", "Authorized tools.",
                                      "Data permissions.", "Publishing authorization.", "Stop conditions.")]
            + [("after verified success: stop or a separately authorized next phase", ["LX-19"])],
    "12.6": [("Scoped credentials.", ["IH-07"]), ("Explicitly allowed accounts and destinations.", ["IH-08"]),
             ("Platform-compliance checks.", ["IH-14"]), ("Publishing approvals where required.", ["IH-09", "IH-04"]),
             ("Spending limits.", ["IH-10"]), ("Rate limits.", ["IH-10"]), ("Data-use permissions.", ["IH-15"]),
             ("Independent audit logs.", ["IH-16"]), ("A pause and emergency-stop mechanism.", ["IH-11"]),
             ("Revocation of access.", ["IH-17"])]
            + [(t, ["IH-14"]) for t in ("Spam or unsolicited mass outreach.", "Deceptive identities.",
                                        "Artificial membership inflation.", "Join-for-join schemes.",
                                        "Payments or rewards conditional on artificial engagement.",
                                        "Circumvention of platform restrictions.", "Credential leakage.",
                                        "Unauthorized data collection.", "Individual psychological profiling.",
                                        "Manipulation of vulnerable participants.")],
    "14.1": [("decision record fields", ["LX-13"])],
    "14.2": [("experiment record fields; predictions registered before outcomes", ["LX-07", "AE-01", "PI-02"])],
    "14.3": [("learning record fields", ["LX-08"])],
    "14.4": [("journal contents", ["LX-09"]), ("durable storage, redaction, timestamps, provenance", ["LX-20"])],
    "15.1": [("17-step persistent event-driven loop", ["LX-15"]),
             ("waits for delayed outcomes without unnecessary activity", ["CU-24"])],
    "15.2": [(t, ["CL-10", "CL-06"]) for t in (
        "Model weights.", "Semantic adapters.", "Intrinsic reward weights.", "Optimizer state.", "Fast adaptive state.",
        "Current task.", "Replay indices.", "Experiment records.", "Pending outcomes.", "Learning schedules.",
        "Random seeds.", "Authorization configuration.", "Model versions.")],
    "15.3": [("monitoring fields and operator controls", ["LX-10"])],
    "16": [("16.1 baselines A-E", ["SV-01", "SV-02", "SV-03", "SV-04", "SV-13"]), ("16.2", ["SV-05"]),
           ("16.3", ["SV-14"]), ("16.4", ["SV-06"]), ("16.5", ["SV-07"]), ("16.6", ["SV-08", "MR-09"]),
           ("16.7", ["SV-15"]), ("16.8", ["CL-07", "SV-10"]), ("16.9", ["SV-09"]), ("16.10", ["SV-12", "NA-10"])],
    "4.2": [("distinguish prediction errors, randomness, missing observations, wrong assumptions, changed conditions", ["PI-11"])],
    "12A (other mandatory clauses)": [
        ("12A.1 versioned execution interface; environment replaceable; no generated-code access to host, vault, evaluator",
         ["IH-01", "IH-12", "CU-10", "CU-37"]),
        ("12A.4 the GUI model is not a strategic agent; CAM decides", ["CU-38", "NA-06"]),
        ("12A.11 serialise shared-state operations; no duplicated external actions", ["CU-36"]),
        ("12A.14 separate learner, browser, sandbox, credential service and evaluator", ["CU-37"]),
        ("12A.15 every meaningful interaction enters the experience stream; no handcrafted strategic reward",
         ["CU-28", "CU-22", "IH-13"]),
        ("12A.16 independent verification inspects computer use and its connection to training", ["IN-08"])],
    "0.1 / final directive": [("standalone: no dependency on unfinished CAM Unified / Crucible", ["IN-09"])],
    "operator amendments": [
        ("storage guard", ["OP-01"]), ("checkpoint retention (cam_live.storage, tools/prune_runs.py)", ["OP-02"]),
        ("disk monitoring for long-running work", ["OP-03"]),
        ("verified official-bot Discord integration on the designated server (docs/evidence/discord/)", ["OP-04"]),
        ("Discord observations enter CAM's experience pipeline", ["OP-05"]),
        ("CAM requests capabilities / resources from the operator (internal.request_operator)", ["OP-06"]),
        ("credential entry and isolation from learner / logs (harness/credentials.py)", ["OP-07"]),
        ("durable off-instance preservation of source; record of what lacks durable backup", ["OP-08"]),
        ("no owner-account automation, no CAPTCHA circumvention (hard platform constraint)", ["OP-09"]),
        ("evaluation redirection: full-size agent, open-ended generation and objectives; synthetic = diagnostics", ["OP-10"]),
        ("planning phase: CAM gets the objective verbatim and makes its own plan and operator requests", ["OP-11"])],
}


def _spec_lists() -> Dict[str, List[str]]:
    """The enumerated items of Part 18 and of 12A item 17 as they appear in the specification."""
    import re
    spec = os.path.join(HERE, "..", "docs", "CAM_LIVE_SPEC.md")
    t = open(spec, encoding="utf-8").read()
    p18 = t[t.index("# PART 18"):t.index("# PART 19")]
    s12 = t[t.index("17. Mandatory acceptance tests"):t.index("Test general-purpose interaction across")]
    return {"18": re.findall(r"^- \[ \] (.+)$", p18, re.M), "12A.17": re.findall(r"^\* (.+)$", s12, re.M)}


def build(old: Dict[str, Any]) -> Dict[str, Any]:
    lists = _spec_lists()
    if len(lists["18"]) != len(P18) or len(lists["12A.17"]) != 30:
        raise SystemExit(f"spec lists changed: {len(lists['18'])} Part 18 items, {len(lists['12A.17'])} 12A.17 items")
    cov = dict(COVERAGE)
    cov["18"] = [(item, [rid]) for item, rid in zip(lists["18"], P18)]
    cov["12A.17"] = [(item, ["CU-%02d" % (i + 1)]) for i, item in enumerate(lists["12A.17"])]
    old_by = {r["id"]: r for r in old.get("requirements", [])}
    reqs: List[Dict[str, Any]] = []
    seen = set()
    for line in R.strip().splitlines():
        rid, ws, ref, title = line.split("|", 3)
        if rid in seen:
            raise SystemExit(f"duplicate id {rid}")
        seen.add(rid)
        base = {"id": rid, "title": title, "spec_ref": ref, "workstream": ws, "files": [], "integration_points": [],
                "acceptance_tests": [], "commands": [], "test_results": [], "evidence": [],
                "engineering_status": "not_started", "verification": {"status": "unverified", "by": "", "notes": ""},
                "limitations": ""}
        base.update({k: v for k, v in old_by.get(rid, {}).items()
                     if k not in ("id", "title", "spec_ref", "workstream", "kind", "level", "needs_executed_evidence",
                                  "declared", "added")})
        base["kind"] = KIND.get(rid, "runtime")
        base["level"] = LEVEL.get(rid, 1)
        base["needs_executed_evidence"] = rid in EXECUTED
        base["declared"] = DECLARED.get(rid, {})
        base["added"] = "2026-11-27" if rid in ADDED_20261127 else "2026-11-26"
        reqs.append(base)
    for sec, items in cov.items():
        for _item, ids in items:
            for i in ids:
                if i not in seen:
                    raise SystemExit(f"coverage section {sec} names unknown id {i}")
    return {"version": 2, "project": "CAM Live 1.0", "branch": "feat/cam-live", "spec": "docs/CAM_LIVE_SPEC.md",
            "status_levels": old.get("status_levels", {}),
            "engineering_states": ["not_started", "partial", "implemented", "integrated"],
            "verification_states": ["unverified", "verified", "partial", "failed", "rejected"],
            "groups": ["complete", "implemented-needs-reverification", "partial", "missing", "failed"],
            "rules": [
                "Generated by tools/gen_manifest.py; merged and checked by `python -m cam_live.verify_completion "
                "--merge-status`. The rules are stated in full in docs/COMPLETION.md.",
                "A requirement is complete only if: its owner claims it implemented; every named file exists and has no "
                "placeholder; it names acceptance tests that exist and whose newest recorded result in "
                "docs/evidence/test_results/*.xml passed and is newer than the last change to the requirement's files "
                "and to the test file; an integration point resolves and is import-reachable for the requirement's "
                "kind; executed evidence exists where needed and is not older than those files; and an independent "
                "'verified' verdict is newer than the last change to its files.",
                "The five status levels are computed separately; none implies the next.",
                "Only workstream G sets verification verdicts; only status files set claimed states."],
            "coverage": {sec: [{"item": item, "ids": ids} for item, ids in items] for sec, items in cov.items()},
            "requirements": reqs}


def main(argv: List[str]) -> int:
    old: Dict[str, Any] = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            old = json.load(f)
    doc = build(old)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, OUT)
    n_new = sum(1 for r in doc["requirements"] if r["id"] in ADDED_20261127)
    print(f"{len(doc['requirements'])} requirements ({n_new} added 2026-11-27) -> {os.path.normpath(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
