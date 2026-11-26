"""Generate docs/CAM_LIVE_COMPLETION_MANIFEST.json (requirement ids, owners, spec references).

Run once to create the skeleton; afterwards `python -m cam_live.verify_completion --merge-status` folds the
workstream status files into it. Re-running this script keeps existing per-requirement evidence.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "CAM_LIVE_COMPLETION_MANIFEST.json")
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
AE-01|B|11.1, 18|The learner can generate its own experiments (registered hypotheses and predictions)
AE-02|M|11.1, 13.3, 18|The learner can revise its approach when evidence contradicts it
AE-03|M|11.1, 18|Self-directed learning functions end to end
AE-04|C|11.4, 18|Goal-free learning mode exists (budgeted)
AE-05|M|10.4, 18|The architecture supports subsequent unfamiliar objectives
AE-06|F|11.3, 18|Future distributed actor-learner operation is supported
AE-07|S|11.2|Shadow policy comparison without public interaction
AE-08|M|12B|Autonomous obstacle resolution through the learned loop, with obstacle records
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
IN-01|M|18|All implementation workstreams are integrated
IN-02|G|18|Independent verification inspects actual code
IN-03|G|18|Independent verification executes required tests
IN-04|M|0.3, 18|Completion checker verifies mandatory engineering requirements
IN-05|M|18|Reproduction commands are documented
IN-06|M|18|Scientific failures and deployment blockers are explicitly recorded
IN-07|M|19.4|End-to-end pathway: observation, latent update, action, consequence, internal reward, neural update, changed behaviour
"""
old = {}
if os.path.exists(OUT):
    old = {r["id"]: r for r in json.load(open(OUT)).get("requirements", [])}
reqs = []
for line in R.strip().splitlines():
    rid, ws, ref, title = line.split("|", 3)
    base = {"id": rid, "title": title, "spec_ref": ref, "workstream": ws, "files": [], "integration_points": [],
            "acceptance_tests": [], "commands": [], "test_results": [], "evidence": [],
            "engineering_status": "not_started", "verification": {"status": "unverified", "by": "", "notes": ""},
            "limitations": ""}
    base.update({k: v for k, v in old.get(rid, {}).items() if k not in ("id", "title", "spec_ref", "workstream")})
    reqs.append(base)
doc = {"version": 1, "project": "CAM Live 1.0", "branch": "feat/cam-live", "spec": "docs/CAM_LIVE_SPEC.md",
       "status_levels": {"1_engineering_implementation_complete": False,
                         "2_controlled_experimental_validation_complete": False,
                         "3_ready_for_authorized_live_deployment": False,
                         "4_live_experiment_executed": False,
                         "5_scientific_hypothesis_supported": False},
       "engineering_states": ["not_started", "partial", "implemented", "integrated"],
       "verification_states": ["unverified", "verified", "rejected"],
       "rules": ["A requirement is engineering-complete only when: state is 'integrated', every claimed file exists and contains no placeholder marker, every acceptance test resolves and has a recorded passing result, at least one integration point is import-reachable from cam_live.agent or cam_live.runtime, and independent verification status is 'verified'.",
                 "The five status levels are separate; none implies the next.",
                 "Only workstream G may set verification.status."],
       "requirements": reqs}
json.dump(doc, open(OUT, "w"), indent=1)
print(len(reqs), "requirements ->", os.path.normpath(OUT))
