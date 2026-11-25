# CAM 7.0 UNIFIED — FULL CONTINUAL EXPERIENCE SUPERLEARNER

## PRIMARY DIRECTIVE

Build, integrate, train and evaluate the complete Unified Experience Superlearner described in this specification.

The existing CAM 7.0 is not the intended final intelligence. It is a limited numerical controller consisting of a small transformer, Bayesian regression, programmed numerical candidate search and learned corrections.

That architecture has demonstrated useful within-family learning but failed to demonstrate generalizable learning across unfamiliar mechanisms.

We are now building a substantially more general neural intelligence capable of:

- Understanding arbitrary semantic information.
- Representing people, organizations, software, technologies, documents, relationships and events.
- Reasoning internally through high-dimensional numerical representations.
- Discovering unknown environmental mechanisms.
- Selecting informative experiments through learned exploration.
- Imagining multiple future action sequences and their consequences.
- Generating novel executable actions and software.
- Learning from successes, failures and unexpected outcomes.
- Continually updating its neural weights throughout an extended lifetime.
- Retaining useful capabilities while adapting to unfamiliar environments.
- Sharing acquired knowledge across multiple independently acting instances.
- Accomplishing difficult, open-ended goals in the full Crucible OpenWorld 4.0 environment.

The central objective is:

**Maximize the rate at which consequential experience produces persistent, transferable intelligence.**

We are not primarily building a controller that finds good numerical settings.

We are building a general experience-driven intelligence that learns how to understand unfamiliar worlds and achieve difficult objectives through experimentation, reasoning, invention and adaptation.

Do not claim superintelligence merely because the architecture is implemented. That remains an experimental aspiration.

---

# PART 0 — MANDATORY IMPLEMENTATION GOVERNANCE

## 0.1 Previous execution failures

Previous CAM implementation sessions repeatedly failed to complete the specification.

CAM 6.1 and CAM 7.0 spent most of their development time on controlled numerical experiments while leaving the most important architectural capabilities unimplemented.

This must not happen again.

A failed numerical benchmark must not prevent implementation of semantic reasoning, general-purpose actions, lifetime learning or OpenWorld integration.

Maintain two independent standards:

1. **Engineering completion:** Every required capability is implemented, connected and exercised end to end.
2. **Scientific performance:** The implemented capability produces measurable learning or control improvements.

A component can be implemented correctly while failing scientifically.

Do not confuse those statuses.

Do not stop at a final research report while mandatory architecture components remain missing.

## 0.2 Use a coordinated engineering team

Use Claude Opus 5.5 for the master architecture agent and core ML subagents if that model is available in the installed Claude Code environment.

Otherwise, use the strongest available Opus model and record the actual model identifiers.

Inspect whether Claude Code supports agent teams, subagents and worktree isolation.

The master agent must remain responsible for the entire architecture.

Create the following workstreams, using separate worktrees or non-overlapping file ownership where supported:

### Workstream A — Semantic grounding and executable actions

Own semantic encoders, document and website understanding, entity/event representation, action representations, trainable action decoding and tool execution.

### Workstream B — Unified latent intelligence

Own the deep transformer, persistent latent state, adaptive mechanism inference, numerical reasoning and shared representation interfaces.

### Workstream C — Predictive world model

Own stochastic action-conditioned dynamics, uncertainty estimation, counterfactual prediction, delayed outcomes and multistep imagination.

### Workstream D — Learning and decision optimization

Own actor-critic training, learned exploration, goal-conditioned meta-RL, imagined policy learning, hierarchical action learning and credit assignment.

### Workstream E — Lifetime memory and continual learning

Own streaming experience storage, learned memory retrieval, persistent replay, neural plasticity, retention, weights-only evaluation and durable checkpoint state.

### Workstream F — Distributed learning

Own actor processes, shared experience ingestion, centralized optimization, versioned checkpoints, policy synchronization, distributed self-play and compute accounting.

### Workstream G — Full Crucible OpenWorld

Own the real OpenWorld adapter, sandboxed execution, independent goal evaluation, difficult long-horizon tasks, diverse world generation and behavioral traces.

### Workstream H — Independent verification

Own acceptance tests, architecture audits, gradient checks, contamination checks, independent reproduction and completion verification.

The verification agent must not simply accept the implementing agents' summaries.

It must inspect actual code, execute tests and trace the runtime pathways.

The master agent must integrate all completed work, resolve architectural conflicts and prevent useful branches from remaining unmerged.

Do not modify the same core files concurrently without explicitly coordinating ownership.

If agent teams are unavailable, execute the same workstreams through sequential or parallel subagents with persistent task handoffs.

## 0.3 Mandatory persistent specification

Before implementing, create:

- `superlearner/docs/CAM_UNIFIED_SPEC.md`
- `superlearner/docs/CAM_UNIFIED_LEDGER.md`
- `superlearner/docs/CAM_UNIFIED_ARCHITECTURE.md`
- `superlearner/docs/CAM_UNIFIED_EXPERIMENTS.md`
- `superlearner/docs/CAM_UNIFIED_COMPLETION.json`

The completion manifest must include every mandatory architectural requirement.

For every requirement, record:

- Requirement identifier.
- Responsible workstream.
- Implementation files.
- Dependencies.
- Acceptance test identifiers.
- Integration test identifiers.
- Current status.
- Relevant commit.
- Actual measured evidence.
- Remaining limitations.

Allowed statuses:

`NOT_STARTED`, `IN_PROGRESS`, `IMPLEMENTED`, `INTEGRATION_TESTED`, `EMPIRICALLY_DEMONSTRATED`, `FAILED_EMPIRICALLY`, `BLOCKED`.

A placeholder, stub, mock, disconnected class or design document cannot count as implemented.

A unit test alone cannot establish integration.

Do not weaken acceptance tests to make an incomplete implementation appear finished.

At the beginning of every new session, read the complete specification, completion manifest and ledger.

Resume the highest-priority unfinished requirements.

Do not repeat completed experiments unnecessarily.

## 0.4 Repository and persistence

Repository:

https://github.com/fnstggl/agentrs

Known reference commits:

- CAM 7.0: `f3cee12`
- Crucible 4.0: `c07098b`

Verify current local and remote state.

Recover any subsequent work before modifying the architecture.

Preserve existing CAM 6.1, CAM 7.0 and Superlearner 4.0 implementations as reproducible baselines.

Create a new integrated research branch, preferably:

`feat/cam-7.0-unified`

Use a new package:

`superlearner/src/superlearner/cam_unified/`

Do not spend the entire implementation rewriting the legacy lever controller.

Inspect the previous GPU storage issues. Ensure code, important checkpoints, experiment manifests and results are backed up to authorized durable storage.

Commit each coherent milestone.

Merge accepted workstreams into the integration branch.

Do not leave completed work stranded in temporary worktrees.

Do not force-push, discard user changes, expose credentials or push without authorization.

---

# PART 1 — THE REQUIRED COMPLETE ARCHITECTURE

Build one unified trainable intelligence.

It should have specialized computational components but not a collection of independent strategic agents that communicate solely through written summaries.

The shared neural backbone must be central to:

- Semantic understanding.
- World-state representation.
- Fast adaptation.
- Predictive learning.
- Memory retrieval.
- Goal reasoning.
- Exploration.
- Action generation.
- Long-horizon decision-making.

## 1.1 Model scales

Implement three configurations with identical architectural capabilities:

| Scale | Backbone layers | Width | Attention heads | State tokens |
|---|---:|---:|---:|---:|
| Development | 8 | 384 | 6 | 32 |
| Main | 16 | 768 | 12 | 64 |
| Large | 24 | 1024 | 16 | 128 |

Use pre-normalized transformer blocks, RMSNorm, SwiGLU feed-forward layers, appropriate positional encoding and residual connections.

Use efficient attention kernels where supported.

The main configuration should target roughly 120–180 million trainable core parameters, depending on the final architecture.

The large configuration should target approximately 320–450 million parameters.

These are starting engineering targets, not theoretically optimal sizes.

Report actual parameter counts, including every component.

**The development model is for debugging. It is not an acceptable substitute for implementing and testing the full main architecture.**

Instantiate and exercise the main model during integration testing.

Benchmark GPU memory, throughput and training compute.

If hardware cannot support a configuration, use appropriate activation checkpointing, gradient accumulation, mixed precision, optimizer sharding or offloading.

Do not silently downgrade the final architecture.

The pretrained semantic model's parameters must be reported separately.

## 1.2 Unified mathematical state

At time t, maintain:

- `O_t`: grounded observation tokens.
- `Z_t`: persistent latent state tokens.
- `B_t`: uncertain beliefs about environmental mechanisms.
- `M_t`: retrieved experience-memory tokens.
- `G_t`: goal representation.
- `A_t`: actual or contemplated action representation.

The main computation is:

`Z_t = F_θ(Z_(t−1), O_t, A_(t−1), B_t, M_t, G_t)`

Where F is the shared trainable backbone.

For the main configuration:

`Z_t ∈ R^(B × 64 × 768)`

Do not compress the entire world into one 128-dimensional vector.

Retain multiple latent tokens so the model can represent relationships, competing explanations, relevant entities and partially observed processes simultaneously.

Use distinct token types, attention masks and temporal metadata.

The numerical representation is the primary internal computational state.

Text is an input/output modality, not a mandatory internal reasoning format.

---

# PART 2 — SEMANTIC UNDERSTANDING

Semantic understanding is mandatory from the first integrated implementation.

The architecture must process more than structured financial numbers.

## 2.1 Observation representation

Define a versioned observation schema supporting:

- Natural-language text.
- Tool descriptions and schemas.
- Tool responses.
- HTML and accessibility trees.
- Browser observations.
- Source code.
- Documentation.
- File contents.
- Software execution results.
- Error messages.
- Images and screenshots where available.
- Numerical measurements.
- Entity relationships.
- Communications.
- Economic transactions.
- Delayed events.
- Goals and environmental constraints.

A numerical observation must retain its value, unit, timestamp, affected entity and evidence source.

A semantic observation must retain its original content or a reference to it.

Do not flatten every observation into a string and rely exclusively on a frozen LLM to understand it.

## 2.2 Pretrained semantic backbone

Integrate a capable pretrained open-weight language/code model.

Inspect the existing Qwen3 integration and locally available checkpoints.

Initially use a roughly 1.5–4B parameter pretrained model, or the existing larger Qwen model if hardware permits.

Expose:

- Token-level hidden representations.
- Document and code embeddings.
- Trainable projections into CAM's latent width.
- A conditional action decoder.
- Trainable low-rank adaptations or selected trainable backbone blocks.

Avoid training semantic competence entirely from scratch on sparse RL feedback.

A 100M-parameter randomly initialized network will not acquire broad language and programming competence from a few thousand simulated business interactions.

However, the pretrained semantic model must not become the actual strategic intelligence while CAM merely chooses among its suggestions.

CAM's latent state must influence semantic processing and action generation through trainable connections.

Use a trainable latent-to-language bridge:

`P_t = W_θ(Z_t, G_t, B_t)`

Where P consists of conditioning vectors consumed by the action decoder through prefix embeddings, cross-attention or another validated mechanism.

Provide gradients from action generation and grounded learning objectives into this bridge.

## 2.3 Semantic learning objectives

Train on:

- Masked semantic representation prediction.
- Predicting tool-call consequences.
- Grounding references to entities.
- Matching code to execution behavior.
- Predicting relevant future events.
- Retrieving evidence needed for a decision.
- Selecting correct actions based on semantic goals.

Preserve exact code and numerical data where necessary.

Do not assume an embedding alone preserves every fact.

Use direct retrieval of original artifacts for tasks requiring exact information.

## 2.4 Semantic acceptance tests

The full system must:

1. Interpret an unseen tool schema.
2. Associate a natural-language objective with appropriate observable entities and actions.
3. Understand a previously unseen document describing an environmental mechanism.
4. Generate executable code solving a novel task.
5. Use an unfamiliar website through browser or structured tools.
6. Correctly associate software failures with relevant execution feedback.
7. Demonstrably change its decisions when a semantically meaningful observation changes.

Randomize entity identifiers, tool names and superficial wording.

Distinguish semantic understanding from memorized tool names.

---

# PART 3 — GENERALIZED ENTITY AND EVENT REPRESENTATIONS

Remove the architectural dependence on 12 levers, 16 gauges and revenue/cost outcomes.

Numerical mechanism environments should become one possible input format, not the defining structure of intelligence.

## 3.1 Entity tokens

Represent each observed entity using:

`e_i = E_θ(type_i, attributes_i, semantic_i, relationships_i, history_i)`

An entity could be a person, company, product, software repository, application, machine, customer group, financial account or unknown discovered object.

Entity categories must be extensible.

Do not require a hardcoded neural head for every possible business concept.

Store persistent entity identifiers separately from learned embeddings.

Use entity identity for binding references, not as a shortcut for learning specific training-world answers.

## 3.2 Relationship learning

Use attention over entity tokens and observed relationship tokens.

For example:

`r_ij = R_θ(e_i, e_j, relation_observation, time)`

Allow the model to discover representations of relationships that are not manually named.

Train relationships through their contribution to predicting observable consequences.

## 3.3 Event tokens

Represent every consequential event as:

`event_t = Enc_θ(entities, action, observations, outcomes, duration, cost, provenance)`

Preserve:

- Which action was performed.
- What information existed before the action.
- What actually happened afterward.
- Which entities were affected.
- When the outcome occurred.
- What remains uncertain.
- Whether the observation was controlled, observational or inferred.

Support delayed event attribution.

Do not assume every consequence arrives immediately following the action responsible for it.

## 3.4 Variable-size worlds

The number of entities, actions and outcomes must vary.

Use dynamic token sequences, sparse retrieval and attention-based compression.

Do not impose a fixed lifetime limit on how many entities or events CAM can encounter.

A bounded working-memory budget is acceptable.

A fixed total ingestion limit is not.

---

# PART 4 — DEEP LATENT REASONING

## 4.1 Internal numerical deliberation

Implement iterative latent reasoning.

Before a consequential action, CAM should be capable of performing multiple internal transformations of its numerical state:

`R_(j+1) = Reason_θ(R_j, Z_t, B_t, M_t, G_t)`

Initialize R from the current latent state.

Allow a configurable number of internal reasoning iterations, initially 1, 2, 4, 8 and 16.

Reuse backbone blocks where computationally appropriate.

The final reasoning state conditions exploration, action generation and value estimation.

Do not require natural-language chain-of-thought at every iteration.

Text-based reasoning may exist as a comparison or complementary capability.

## 4.2 Adaptive computation

Learn or configure a reasoning-budget policy.

Easy actions should require little deliberation.

Unfamiliar, costly or consequential actions may require more.

Account for compute cost.

Compare different reasoning depths under matched compute and experience budgets.

Do not treat more inference-time computation as improved sample efficiency.

## 4.3 Train reasoning through outcomes

Latent reasoning must receive meaningful gradients from:

- Action generation.
- Value prediction.
- World-model prediction.
- Exploration objectives.
- Future goal achievement.

Do not implement a latent reasoning loop whose output is ignored by actual decisions.

Use intervention and ablation tests to establish that changes to its numerical state affect behavior.

---

# PART 5 — FAST ADAPTATION AND MULTIPLE HYPOTHESES

Retain the successful mathematical evidence-incorporation principle from CAM 7.0, but generalize it.

Do not make a single Bayesian regression over lever settings the entire world model.

## 5.1 Belief representation

Maintain several plausible latent explanations of the current environment.

A starting configuration:

- 8–16 mechanism particles.
- 128–256 dimensions per particle.
- Normalized importance weights.
- Conditional fast adaptation statistics.
- Explicit uncertainty summaries.

Represent:

`B_t = {(m_t^k, w_t^k, Σ_t^k)}_(k=1)^K`

Each mechanism particle can represent a different hypothesis about how the environment behaves.

Do not manually assign particles to named concepts such as demand, trust or pricing.

They should emerge from training.

## 5.2 Bayesian-style updating

After an executed action produces a new observation:

`log w_t^k = log w_(t−1)^k + log p_θ(o_t | Z_(t−1), a_(t−1), m_(t−1)^k)`

Normalize the resulting weights.

Use a learned proposal network to update the particles themselves.

Monitor effective sample size:

`ESS = 1 / Σ_k (w_t^k)^2`

When particle diversity collapses, use a tested resampling or rejuvenation procedure.

Support new hypotheses emerging when existing explanations make consistently incorrect predictions.

This is approximate neural Bayesian inference, not exact inference over the entire real world.

## 5.3 Retain local Bayesian adaptation

Use exact or approximate Bayesian last-layer inference over learned features for suitable predictable outcome channels.

For local features φ and observed numerical outcome y:

`K_t = Σ_t φ_t / (σ² + φ_tᵀ Σ_t φ_t)`

`μ_(t+1) = μ_t + K_t(y_t − φ_tᵀ μ_t)`

`Σ_(t+1) = Σ_t − K_t φ_tᵀ Σ_t`

Use stable numerical implementations.

These fast adaptive parameters should be conditioned on the shared latent representation and relevant mechanism hypotheses.

They supplement the nonlinear dynamics model.

They must not restrict every possible environmental mechanism to a globally additive linear model.

## 5.4 Prevent false certainty

Learn uncertainty in:

- Environmental mechanisms.
- Hidden current state.
- Outcome randomness.
- Model parameters.
- Distribution shifts.
- Delayed outcomes.

A model can be confidently wrong because none of its hypotheses adequately represents reality.

Support explicit misspecification detection and uncertainty expansion.

Test calibration on unfamiliar mechanism families.

Do not claim calibrated uncertainty from ensemble disagreement alone.

---

# PART 6 — GENUINE CAUSAL WORLD MODEL

This is one of the most important implementation requirements.

CAM 7.0 currently predicts the value of holding numerical settings. That is inadequate.

The new world model must predict how the environment evolves under arbitrary semantic actions.

## 6.1 Stochastic action-conditioned dynamics

Implement:

`p_θ(Z_(t+1), O_(t+1), Y_(t+1) | Z_t, A_t, B_t, Δt)`

Where:

- Z is latent world state.
- A is an actual executable intervention.
- B represents uncertain environmental mechanisms.
- Δt is elapsed time.
- O represents future observations.
- Y represents measured consequences.

Use a shared transformer transition computation with action tokens, mechanism tokens and elapsed-time conditioning.

A proposed transition:

`Ẑ_(t+1) = Dynamics_θ(Z_t, EncAction_θ(a_t), B_t, EncTime(Δt), ε_t)`

Where ε represents stochastic environmental variation.

Predict a distribution over next latent states rather than always a deterministic mean.

Use a validated stochastic latent parameterization, such as categorical latent variables or a conditional continuous distribution.

## 6.2 Predict multiple kinds of consequences

The model must predict more than revenue and cost.

Support variable outcome tokens representing:

- Resources.
- Financial changes.
- User adoption.
- Technical performance.
- Reliability.
- Task progress.
- New entities.
- Changes in relationships.
- Software execution outcomes.
- Observed communications.
- Delayed events.
- Goal completion.

Represent outcome types dynamically.

Use generic distributional prediction heads with type conditioning and masks.

Use appropriate likelihoods:

- Continuous: Gaussian or Student-t distributions.
- Binary: Bernoulli.
- Categorical: categorical likelihood.
- Counts: suitable count distributions.
- Time-to-event: censoring-aware survival objectives where needed.

Predict relevant observations, not just externally assigned reward.

## 6.3 Learning latent dynamics

Train using real experience sequences.

Use an objective of the form:

`L_world = λ_z L_latent + λ_o L_observation + λ_y L_outcome + λ_r L_reward + λ_d L_duration`

For latent prediction, use an independently encoded future observation as the target:

`Z_target_(t+1) = stopgrad(EMAEncoder(O_(≤t+1)))`

Train the dynamics model to predict that representation.

Use reconstruction, variance-preserving or other tested objectives to prevent representational collapse.

The target encoder must not expose future information to the current policy.

Retain meaningful semantic and observable-state prediction losses.

## 6.4 Multistep training

Train the model on actual recorded action sequences.

For horizons:

`h ∈ {1, 2, 4, 8, 16, 32}`

Roll predictions forward without access to future actual observations.

Compare against future encoded states and measured outcomes.

Do not merely repeat one-step teacher-forced predictions and label that multistep imagination.

Support irregular elapsed times and long-running actions.

Evaluate prediction errors as a function of horizon.

## 6.5 Causal evidence

Distinguish observational prediction from causal identification.

Record:

- Intervention assignments.
- Known randomization probabilities.
- Simultaneous changes.
- Evidence reliability.
- Confounding possibilities.
- Delayed effects.
- Missing observations.

Train primarily from information actually available to the agent.

Use controlled Crucible interventions to test whether the model can distinguish alternative causal structures.

The independent evaluator may access privileged counterfactual forks.

The agent must not access evaluator-only hidden mechanisms.

Where counterfactual simulation is used for training, explicitly identify that experimental condition and do not conflate it with real-experience-only learning.

## 6.6 Mandatory causal-model tests

Demonstrate that:

1. Different contemplated actions produce different predictions from the same initial latent state.
2. The model distinguishes delayed from immediate consequences.
3. It models interactions between several interventions.
4. It updates predictions when a mechanism changes.
5. Multi-step predictions outperform suitable no-change, persistence and simple regression baselines where supported.
6. Changing predicted intermediate events can change later predicted consequences.
7. An imagined future does not mutate the actual environment.

The implementation must not contain a steady-state lever regression masquerading as the primary causal dynamics model.

---

# PART 7 — MULTISTEP IMAGINATION AND PLANNING

Implement genuine belief-aware numerical imagination.

CAM must be capable of contemplating an action, imagining possible observations, updating its imagined understanding and then considering subsequent actions.

## 7.1 Imagined trajectories

Starting from actual belief state:

`S_t = (Z_t, B_t, G_t)`

Generate a contemplated action:

`â_t ~ π_θ(. | S_t)`

Predict a possible consequence:

`(Ẑ_(t+1), Ô_(t+1), Ŷ_(t+1)) ~ WorldModel_θ(S_t, â_t)`

Update the simulated belief:

`B̂_(t+1) = Update_θ(B_t, â_t, Ô_(t+1))`

Then select a subsequent imagined action conditioned on this revised belief.

Repeat recursively.

Use independent copies of fast state for each imagined branch.

An imagined observation must never enter real-world memory as an actual observed event.

## 7.2 Planning implementation

Implement batched latent planning with:

- Multiple candidate strategies.
- Multiple stochastic future samples.
- Variable rollout horizons.
- Expected goal achievement.
- Uncertainty and model-error estimates.
- Elapsed time and resource costs.
- Belief updates during imagined experiments.

Start with rollout horizons of 4, 8 and 16 abstract decisions.

Test longer horizons when prediction quality supports them.

For extended tasks, use learned temporally extended actions rather than attempting to predict thousands of individual browser clicks in a single rollout.

Use learned value estimates to approximate continuation beyond the imagination horizon.

Compare beam search, sampling-based planning and policy-generated rollouts under matched compute.

## 7.3 Train from imagination

Once the world model is sufficiently validated, use imagined trajectories to improve the policy and value functions.

Keep real and imagined experiences explicitly separate.

Use uncertainty-aware weighting.

Do not let the policy exploit prediction errors in its own model.

Require continual correction using actual environment experience.

When predictions are unreliable, reduce imagined rollout horizon or weight rather than treating simulated outcomes as facts.

## 7.4 Mandatory imagination acceptance

Verify end to end that CAM can:

- Imagine at least two alternative interventions.
- Roll each forward across multiple sequential decisions.
- Predict different outcomes.
- Revise its imagined beliefs.
- Choose subsequent actions conditional on those revised beliefs.
- Use the imagined consequences to change a real action.

Include a diagnostic task where a useful experiment has immediate cost but creates an opportunity for a later action.

A one-step knowledge-gradient calculation is not sufficient.

---

# PART 8 — LEARNED EXPLORATION

The existing CAM 7.0 chooses experiments largely using a programmed knowledge-gradient formula.

We want the neural intelligence to acquire an increasingly effective experimentation strategy through experience.

Retain mathematical value of information as a baseline and optional initial training signal, not the sole permanent exploration mechanism.

## 8.1 Separate uncertainty from usefulness

An unexpected event is not necessarily useful.

An unpredictable event may contain no learnable mechanism.

Do not reward raw prediction error without accounting for irreducible randomness.

Train exploration to discover information that improves subsequent competence.

## 8.2 Learned exploratory policy

Condition the policy on:

`π_θ(a | Z_t, B_t, G_t, M_t, budget_t)`

Train decisions that may improve either:

- Direct goal achievement.
- Future ability to achieve goals.

The exploration policy must be able to generate and choose experiments, not just adjust a scalar exploration coefficient.

Train it to decide which uncertainty matters and what intervention would resolve it.

## 8.3 Learning-progress objective

Define measurable learning progress using independently evaluated predictive or decision improvements.

For example:

`LP = Eval(learner_after_support, query_set) − Eval(learner_before_support, query_set)`

Use held-out query observations or subsequent independently verified goal outcomes.

Do not use the same observation both to update the learner and to claim improved prediction.

Distinguish epistemic information from noise.

Estimate the long-term contribution of exploratory actions using return-based credit assignment.

## 8.4 Meta-exploration training

Construct learning episodes in which an early experiment changes the value of later decisions.

Optimize complete trajectories, including the cost of early exploration and the downstream benefit of acquired information.

A conceptual objective:

`J_explore(θ) = E[Σ_k w_k J_query(U_θ(D_(1:k))) − λ C_experiment]`

Where U is the learned adaptation mechanism.

Implement a tractable estimator using recurrent meta-RL and policy gradients through the sequence of actual exploratory decisions.

Where adaptation is differentiable, allow gradients through the adaptation operation.

Use controlled simulator forks to construct evaluation evidence about useful experiments without exposing hidden evaluator state to the ordinary agent.

A predictive world model can provide additional imagined training trajectories.

## 8.5 Exploration baselines

Compare:

- Random exploration.
- Greedy exploitation.
- Thompson sampling.
- Existing CAM 7.0 knowledge gradient.
- Expected information gain.
- Learned exploratory policy.
- Learned policy with mathematical exploration assistance.

The learned exploration strategy must be capable of operating without the old programmed knowledge-gradient action score.

It need not outperform all baselines immediately.

If it fails, diagnose the learning process instead of claiming that the programmed algorithm constitutes learned exploration.

---

# PART 9 — GENERAL-PURPOSE ACTION GENERATION

Remove the requirement that CAM select only from numerical lever changes.

## 9.1 Action schema

Define an extensible action representation containing:

- Tool or operation.
- Semantic purpose.
- Arguments.
- Relevant entities.
- Executable code where applicable.
- Anticipated duration.
- Observable resource costs.
- Execution constraints.

The action interface must support arbitrary valid tools discovered from the environment.

It must not require modifying the neural architecture whenever a new tool becomes available.

## 9.2 Trainable action decoder

Use the shared latent state to condition a semantic decoder.

A possible factorization:

`π(a | Z,G,B) = π(tool | Z,G,B) × π(arguments | tool,Z,G,B)`

For open-ended arguments, use autoregressive generation with the pretrained language/code decoder conditioned on CAM's latent representations.

Train tool selection, argument generation and action composition.

Use valid schema masks and grammar-constrained decoding where applicable.

The validator should check executability, not dictate the strategy.

## 9.3 Novel action composition

Support the generation of new:

- Python programs.
- Software projects.
- Tool-call sequences.
- Technical experiments.
- Application deployments.
- Research queries.
- Business operations.
- Multi-step workflows.

CAM must be able to produce executable artifacts that were not present in a fixed candidate library.

Use actual sandboxed execution.

Do not evaluate arbitrary code through a mock reward function.

## 9.4 Temporally extended actions

Implement a learned hierarchical action mechanism.

Represent high-level action intents as latent option tokens.

An option may condition a sequence of lower-level tool actions until a learned or externally defined termination condition.

Use semi-Markov credit assignment.

For an option of duration Δt:

`γ_Δ = exp(−κ Δt)`

`target = R_option + γ_Δ V(next_state)`

Distinguish:

- Primitive tool actions.
- Reusable learned skills.
- Longer goal-conditioned strategies.

Do not hardcode a business-building strategy as an option and attribute its competence to neural learning.

Allow useful options to emerge from successful experience.

## 9.5 Avoid hidden LLM substitution

The pretrained language model may initially assist with grammatical tool calls, coding and semantic grounding.

However:

- CAM's latent state must influence action generation.
- The trainable policy must select consequential actions.
- Outcome learning must update future decision behavior.
- A frozen semantic model must not independently decide the entire strategy.

Log whether each action came from CAM's learned decoder, a frozen proposal model, planning or an explicit baseline.

If using candidate proposals, record the proposal distribution and selection probabilities.

Do not claim end-to-end policy-gradient learning of the proposal model if only the candidate selector receives gradients.

Measure learned proposal quality independently from candidate selection.

---

# PART 10 — TRUE LIFETIME MEMORY

This is non-negotiable.

Remove the failure mode represented by:

`T_MAX = 256`

CAM must never stop receiving experience because it reaches a fixed number of observation rows.

A finite working-memory window is acceptable.

A finite lifetime ingestion cutoff is not.

## 10.1 Four complementary forms of memory

Implement:

### A. Persistent working state

A set of latent state tokens updated continuously.

Maintains current understanding of the situation.

### B. Fast adaptive memory

Numerical beliefs and local adaptive parameters.

Immediately incorporates new evidence.

### C. Durable episodic memory

A persistent indexed archive of actual experiences and artifacts.

Supports retrieval of relevant information even after millions of intervening events.

### D. Parametric long-term memory

Neural weights that consolidate transferable knowledge and skills.

This is mandatory. The system must not merely become a retrieval agent.

## 10.2 Streaming event storage

Implement an append-only, chunked event archive.

Use practical durable formats such as Arrow/Parquet, SQLite metadata and content-addressed artifact storage.

Do not require loading the entire lifetime into RAM.

Each record should include:

- Unique event ID.
- Environment and actor ID.
- Model version.
- Timestamp.
- Observation reference.
- Action.
- Actual result.
- Predicted result.
- Uncertainty.
- Goal.
- Cost.
- Relevant entity links.
- Behavior-policy probability.
- Training provenance.

Support compression, indexing and incremental loading.

Use disk-backed storage and verified backups.

Handle storage exhaustion explicitly.

Never silently discard all older experience because a fixed in-memory buffer is full.

Physical storage is finite. Design for scalable retention, archiving and controlled resource limits rather than claiming literal unlimited memory.

## 10.3 Learned memory retrieval

Encode experience segments into retrieval keys and values.

Use a trainable query conditioned on current latent state and goal:

`q_t = Q_θ(Z_t, G_t, B_t)`

Retrieve relevant experiences:

`M_t = Retrieve(q_t, memory_index, top_k)`

Attend to retrieved numerical memory tokens.

Support retrieval of original documents, code and observations when exact details are required.

Initially use approximate nearest-neighbor retrieval with temporal and entity indexing.

Train retrieval usefulness from downstream prediction and decision outcomes.

Hard retrieval selection may require contrastive training, supervised relevance from observable outcomes or a learned retrieval-policy objective.

Do not assume all retrieval operations are differentiable.

## 10.4 Hierarchical memory consolidation

Maintain multiple levels:

- Raw recent events.
- Compressed event segments.
- Longer experience representations.
- Indexed durable artifacts.
- Parametric knowledge.

Train compression to preserve information useful for future queries, prediction and decisions.

Do not assume compressing events into language summaries is sufficient.

Use learned latent memory tokens as the primary compressed representation.

Retain source links to reconstruct details when needed.

## 10.5 Prevent stale memory embeddings

As the neural representation changes, stored embeddings may become incompatible.

Version memory embeddings.

Implement re-encoding, compatible target encoders or migration.

Do not retrieve obsolete vectors as if their meaning were permanently stable.

## 10.6 Memory acceptance tests

Run the actual runtime across:

- More than 256 events.
- More than 10,000 events.
- A streaming stress test of at least one million lightweight event records.

Verify no silent cessation of ingestion.

Test retrieval of relevant information after increasingly long delays and interfering observations.

Test arbitrary delayed cues using the complete CAM architecture.

Test restart and recovery.

Report retention accuracy versus delay, storage consumption, retrieval latency and training costs.

Do not claim perfect memory merely because raw information exists in a database.

---

# PART 11 — CONTINUAL NEURAL LEARNING

The system must continually acquire capabilities in actual neural parameters.

Persistent written or retrieved context alone is insufficient.

## 11.1 Three learning timescales

Implement:

### Immediate

Update latent state and fast adaptive beliefs following new evidence.

### Ongoing

Perform gradient updates to shared neural weights using recent experience and replay.

### Long-term

Optimize transfer, retention, plasticity and the efficiency of adaptation across many tasks and lifetimes.

The model must learn not just environmental facts, but increasingly useful representations and priors for interpreting future evidence.

## 11.2 Persistent replay

Do not replace lifelong replay with a FIFO buffer of 2,048 episodes.

Maintain:

- Recent experiences.
- Uniform historical samples.
- Domain-balanced samples.
- Important failure trajectories.
- Successful innovations.
- Rare but informative outcomes.
- Previously acquired skills.
- Underrepresented mechanisms.

Use the durable event archive as the underlying source.

A bounded active replay cache is acceptable.

Historical experience must remain accessible through the archive subject to explicit storage policies.

Avoid replaying only successes.

Learn from failures and unexpected outcomes.

## 11.3 Optimize plasticity

Instrument:

- Activation diversity.
- Gradient norms.
- Gradient interference.
- Dormant units.
- Parameter-update magnitude.
- Effective representation rank.
- New-task learning speed.
- Historical retention.

Start with strong ordinary training controls: balanced replay, appropriate learning rates and normalization.

Test additional plasticity mechanisms independently:

- Continual backpropagation.
- Selective reinitialization of persistently underused units.
- Adaptive learning-rate schedules.
- Gradient conflict mitigation.
- Replay-based distillation.
- Context-dependent parameter use.

Do not apply all mechanisms simultaneously without understanding their interactions.

Retain modifications only when they improve acquisition or retention under controlled comparison.

## 11.4 Parametric consolidation

Train the semantic projections, shared core, world model, action policy, adaptation priors and selected language-model parameters as appropriate.

Log actual gradient norms and weight changes for every intended trainable parameter group.

Do not freeze the core strategic representation and claim continual intelligence because an output head changes.

Use full core training as the default.

For large pretrained semantic components, use parameter-efficient adaptation when necessary, while measuring which capabilities actually change.

## 11.5 Checkpoint semantics

Implement separate explicit operations:

`save_weights()`

`load_weights_only()`

`save_full_lifetime()`

`load_full_lifetime()`

`reset_fast_state()`

`reset_optimizer()`

`reset_experience_access()`

A weights-only checkpoint must contain only the necessary learned model parameters and fixed model configuration.

It must not restore replay, optimizer momentum, task history, adaptive posterior, random-generator state or previous environment state.

This is required to fix CAM 7.0's contaminated transfer comparison.

---

# PART 12 — THE INTEGRATED LEARNING OBJECTIVE

Implement a coherent training algorithm, not a collection of unrelated losses that never influence one another.

## 12.1 World learning

Train the shared model on real experience:

`L_world = prediction of observed consequences, future latent states and environmental transitions`

## 12.2 Fast adaptation learning

For a support prefix S and later query observations Q:

`B_S = Infer_θ(S)`

`L_adapt = −log p_θ(Q | B_S, query_actions)`

Backpropagate through differentiable adaptation computations.

Train across different support sizes:

`k ∈ {0,1,2,4,8,16,32,64}`

Support and query data must remain correctly separated.

## 12.3 Goal-conditioned policy learning

Train the policy to achieve independent externally specified goals.

Represent the goal semantically and numerically.

Use actual verified returns and duration-aware temporal credit assignment.

A candidate actor-critic objective:

`L_actor = −E[log π_θ(a_t | state_t) A_t]`

`L_critic = E[(V_θ(state_t) − target_t)^2]`

Use PPO for fresh on-policy trajectories or an explicitly validated off-policy method such as V-trace for asynchronous trajectories.

Do not apply ordinary PPO ratios to arbitrarily stale replay without validating the estimator.

Record complete behavior-policy provenance.

## 12.4 Optimize learning efficiency directly

The outer learning objective must reward becoming competent from less experience.

Use meta-training across task distributions:

`J_meta = E_τ[Σ_k w_k J_τ(U_θ(D_(1:k)))]`

Where:

- τ is a task drawn from a training distribution.
- D contains observed support experiences.
- U is the fast adaptation mechanism.
- J measures subsequent goal-directed competence.

Use support/query meta-RL trajectories that include the actual cost of learning.

Train early decisions according to their downstream effect on competence.

For gradient-based inner adaptation, use a tractable differentiable approximation or appropriate meta-gradient method.

For exploratory decisions with non-differentiable real outcomes, use policy-gradient credit assignment.

Do not pretend gradients can flow through the real environment.

## 12.5 Combined training

A candidate total loss:

`L_total = λ_world L_world + λ_adapt L_adapt + λ_actor L_actor + λ_value L_value + λ_action L_action + λ_memory L_memory + λ_retention L_retention + λ_meta L_meta`

These weights must be measured and tuned.

Use separate optimizer schedules or gradient boundaries where needed.

Do not let a large reconstruction loss dominate decision-relevant learning.

Track gradient contributions from every objective into the shared backbone.

The complete architecture must be integrated, but stability may require deliberate staged optimization.

Use the following training progression:

1. Semantic and tool-action grounding.
2. Predictive world-model learning from recorded experience.
3. Fast adaptation and support/query training.
4. Real-environment actor-critic learning.
5. Learned exploration and meta-RL.
6. Calibrated imagination training.
7. Continual consolidation and retention.

These stages are training schedules, not excuses to omit later components.

Every component must be implemented and integration-tested regardless of whether an earlier learning stage reaches its scientific target.

---

# PART 13 — LEARNING GENERALIZABLE PATTERNS

CAM 7.0 trained on six mechanism families and failed to transfer beyond familiar structures.

We must address this at the representation, data-distribution and learning-objective levels.

## 13.1 Diversity of world structure

Do not merely generate more seeds of the same six mechanisms.

Generate diverse training worlds containing different combinations of:

- Delays.
- Nonlinearities.
- Feedback loops.
- Resource constraints.
- Partial observability.
- Hidden state.
- Competing agents.
- Cooperative opportunities.
- Changing mechanisms.
- Network effects.
- Technical failures.
- Dependencies between services.
- Market structures.
- Behavioral populations.

Randomize the actual causal structures.

Preserve disjoint structural holdouts.

## 13.2 Compositional learning

Train on mechanisms that appear individually and in various combinations.

Hold out selected combinations entirely.

The model should learn representations of reusable causal processes rather than memorize complete world configurations.

Use relational attention and factorized or object-centric dynamics.

Allow different parts of the latent state to represent interacting processes without assigning every process a manually named meaning.

Evaluate whether learned components can recombine in unfamiliar circumstances.

## 13.3 Invariant representations

Construct pairs of experiences with different surface appearances but equivalent underlying dynamics.

For example, change:

- Entity identities.
- Tool names.
- Units.
- Observation ordering.
- Wording.
- Interface presentation.

Train appropriate invariance or equivariance in the representations and predictions.

Do not force fundamentally different causal mechanisms into identical embeddings.

## 13.4 Learn intervention-sensitive representations

Two situations should share useful representations when their responses to corresponding interventions are similar.

Train on observed intervention consequences.

Use contrastive or predictive objectives that distinguish similar-looking situations with different controllable dynamics.

Only use hidden causal labels as an explicitly disclosed auxiliary research condition.

## 13.5 Automated curriculum

Implement a curriculum driven by learning progress and capability coverage.

Maintain a distribution over training task families and difficulty levels.

Prioritize tasks that are challenging but learnable.

Avoid spending all experience on tasks the model already masters or tasks that provide no usable learning signal.

Use diversity constraints so the curriculum does not collapse onto one profitable environment family.

Do not use held-out evaluation tasks to select training examples.

## 13.6 Heterogeneous opponents

Use diverse independently controlled participants and competitors.

Include:

- Existing Crucible behavioral actors.
- Independently learned policies.
- Historical CAM checkpoints.
- Current CAM actors.
- Optional language-capable agents.

Avoid training exclusively through interaction with identical copies of the current model.

Evaluate against unfamiliar participant populations.

## 13.7 Correct negative-transfer diagnosis

Rerun the CAM 7.0 transfer comparison with:

A. Fresh model and optimizer.

B. Pretrained weights only.

C. Pretrained weights plus optimizer state.

D. Pretrained weights, optimizer and historical replay.

Match target experience budgets.

Evaluate zero-shot behavior and acquisition curves.

Identify whether negative transfer arises from representations, overly confident priors, optimizer state, replay interference or insufficient task diversity.

Do not infer the cause solely from the negative score.

---

# PART 14 — DISTRIBUTED SHARED-WEIGHT LEARNING

Implement distributed experience collection from the beginning.

The goal is for multiple CAM instances to gain different experiences and improve one shared intelligence.

## 14.1 Actor-learner architecture

Actors maintain:

- Independent environments.
- Independent goals.
- Independent latent state.
- Independent fast adaptation state.
- Independent random seeds.
- A versioned copy of shared model parameters.

A central learner receives their experiences.

The central learner is responsible for neural optimization.

Actors do not independently overwrite the global model.

## 14.2 Experience transport

Implement a durable, versioned experience queue.

Support:

- Batching.
- Deduplication.
- Backpressure.
- Interrupted actor recovery.
- Delayed outcomes.
- Model-version tagging.
- Exactly identifiable event records.
- Replay ingestion.

Use a local multiprocess implementation first.

Provide a clear extension to distributed machines and multiple GPUs.

## 14.3 Policy synchronization

Publish checkpoints atomically.

Actors should refresh to validated model versions at safe decision or trajectory boundaries.

Do not reuse stale latent states blindly after substantial representation changes.

Reconstruct relevant state from version-compatible memory or replay when necessary.

## 14.4 Asynchronous learning

Track policy lag.

Use appropriate off-policy corrections or discard samples outside validated training conditions.

Do not misrepresent asynchronous stale experience as ordinary on-policy PPO data.

## 14.5 Evaluation

Run actual distributed experiments with 1, 4, 8 and, where hardware allows, 16 actors.

Measure:

- Experiences per second.
- Compute cost.
- Sample efficiency.
- Wall-clock learning speed.
- Retention.
- Policy staleness.
- Cross-domain interference.

Demonstrate that an actor's experience can produce a central weight update that changes a different actor's later behavior.

A collection of independent processes without shared learning does not satisfy this requirement.

---

# PART 15 — FULL CRUCIBLE OPENWORLD 4.0 INTEGRATION

This must be implemented concurrently with the core neural architecture.

Do not wait until numerical benchmarks succeed.

## 15.1 Inspect actual Crucible interfaces

Read:

- `crucible/docs/AGENT_API.md`
- `crucible/docs/OPENWORLD_REPORT.md`
- `crucible/docs/SYNTHETIC_INTERNET.md`
- `crucible/src/crucible/api/`
- `crucible/src/crucible/agents/`
- `crucible/src/crucible/services/`
- `crucible/src/crucible/goals/`
- `crucible/src/crucible/worldgen/`
- `crucible/src/crucible/research/`

Use existing functioning interfaces wherever possible.

Support the real generated worlds, not only `MechVecEnv`.

## 15.2 Architecture-neutral environment adapter

Define a shared interface equivalent to:

`observe() -> ObservationBatch`

`available_tools() -> ToolSchemaBatch`

`execute(action) -> ExecutionResult`

`collect_events(since) -> EventBatch`

`advance_time(duration) -> ObservationBatch`

`evaluate_goal() -> IndependentGoalResult`

The adapter must preserve partially observable information.

Do not expose evaluator-only state to CAM.

Use the same neural intelligence for numerical environments and OpenWorld.

Differences in observation schemas and execution mechanisms should be handled by adapters, not by entirely separate strategic architectures.

## 15.3 Real capabilities

CAM must be able to:

- Search the simulated internet.
- Read documents and websites.
- Discover unfamiliar APIs.
- Create and execute code.
- Build and deploy software.
- Operate businesses.
- Observe independent customers.
- Manage resources and costs.
- Respond to software failures.
- Communicate with simulated participants.
- Investigate opportunities.
- Generate original solutions.
- Observe delayed consequences.
- Continue operating over extended simulated lifetimes.

Use actual software execution and existing independent outcome verification.

Do not substitute scripted business trajectories for learned action generation.

## 15.4 Sandbox boundary

All experiments in this assignment must remain inside Crucible and approved local execution sandboxes.

Do not enable uncontrolled access to the actual public internet.

Keep external account creation, public communications, financial transactions and real-world deployment disabled.

Treat website content and tool responses as untrusted data.

The agent must not modify the evaluator, research harness, permission system or its own external authorization.

Use resource limits and isolated execution.

---

# PART 16 — DIFFICULT LONG-HORIZON OPENWORLD TASKS

Implement actual research protocols for difficult outcome-based goals.

Do not give the agent the intended solution.

Use existing Crucible goal contracts and add independently verifiable predicates where necessary.

Calibrate task feasibility using existing scripted builders and evaluator-only diagnostics.

A task must be difficult but achievable under the environment's actual laws.

Do not artificially increase reward or lower success thresholds to make CAM appear successful.

## Experiment OW-1 — Business creation

Objective:

Create and operate a profitable business generating the equivalent of $10,000 per month in independently verified recurring revenue within 90 simulated days.

Require positive operating economics and genuine independent customer activity.

Vary starting capital, customer populations, technological opportunities and competing firms.

The agent chooses what to build and how to operate it.

Scale to a $100,000 annualized recurring revenue objective and other thresholds where task feasibility supports them.

Do not count circular payments, self-purchases or fabricated customers.

## Experiment OW-2 — Novel software creation

Give CAM an environment containing unmet technical needs.

Require it to:

- Discover a useful problem.
- Develop an original executable software solution.
- Deploy it.
- Produce independently measurable utility.
- Achieve genuine adoption by other participants.

Do not supply the intended software architecture.

Evaluate on hidden work instances.

## Experiment OW-3 — Technical invention and resource efficiency

Require CAM to achieve a substantial measurable improvement in infrastructure, production or service efficiency.

Examples:

- Reduce operating cost while preserving service quality.
- Solve a previously unknown allocation problem.
- Improve reliability under changing workloads.
- Discover and deploy a useful scheduling or technical algorithm.

Select thresholds based on verified task feasibility.

## Experiment OW-4 — Long-horizon adaptation

Place CAM in a persistent world containing unexpected changes:

- Competitor entry.
- Infrastructure shocks.
- Demand changes.
- Resource shortages.
- Service failures.
- Changing technical mechanisms.

Evaluate whether CAM detects changes, revises its understanding and recovers performance.

Run over 90–180 simulated days.

## Experiment OW-5 — Cross-domain transfer

Train CAM through digital business experiences.

Then expose it to unfamiliar industrial and technical environments.

Compare:

- Pretrained CAM.
- Weights-only transferred CAM.
- Fresh CAM.
- Frozen pretrained semantic agent.
- Existing CAM baselines where compatible.

Measure whether previous experience accelerates acquisition of genuinely unfamiliar capabilities.

## Experiment OW-6 — Self-directed lifetime learning

Use a persistent Crucible world without a fixed immediate external task.

Allow CAM to select useful projects and experiments within its authorized simulated capabilities.

Implement an internal learning-progress objective grounded in improvements on independently measured prediction and competence.

Prevent the agent from satisfying itself with trivial self-generated goals.

Periodically evaluate it against independently chosen held-out objectives.

Measure whether self-directed experience improves external competence.

## Experiment OW-7 — Cooperation and coordination

Require CAM to achieve a useful outcome that depends on other independently controlled participants.

Measure actual completed work, quality, reliability and jointly produced utility.

Do not reward sheer message volume or superficial agreement.

## Important environment limitation

Crucible 4.0 has limited social-media and general browsing-population mechanics.

Do not claim to test realistic viral-video growth or community dynamics if those mechanisms do not exist.

Record those gaps.

If suitable extensions can be implemented without derailing the main architecture, add independently testable simulated content and community tasks.

Otherwise preserve them as clearly scoped future environment work.

---

# PART 17 — THE FOUR FOUNDATIONAL LEARNING EXPERIMENTS

Run these alongside OpenWorld development.

Use the same shared neural architecture.

## Experiment 1 — One-shot and few-shot adaptation

Test prediction and decision improvement after:

`0,1,2,4,8,16,32,64 observations`

Run:

A. Identical support experience for every learner.

B. Agent-selected experiments.

C. Semantic observations describing potentially relevant world information.

D. Delayed and partially observable evidence.

Compare strong Bayesian, recurrent meta-RL, attention-based, frozen semantic and existing CAM baselines.

Measure proper predictive scores, regret, goal success and learning-curve area.

Do not confuse prediction improvement with control improvement.

## Experiment 2 — Causal discovery

Construct initially indistinguishable worlds with different underlying mechanisms.

Test whether CAM chooses informative interventions and learns which mechanisms are active.

Include confounders, interacting effects and delayed consequences.

Measure:

- Number of observations needed.
- Decision-relevant prediction accuracy.
- Intervention-effect accuracy.
- Cumulative regret.
- Goal achievement.
- Resource and compute efficiency.

Compare learned exploration with the existing mathematical knowledge-gradient and Thompson-sampling baselines.

## Experiment 3 — Parametric transfer and retention

Run fresh-process weights-only transfer.

Evaluate:

- New instances of familiar structures.
- New combinations.
- Unseen causal structures.
- New simulator implementations.
- New semantic interfaces.
- Unfamiliar full OpenWorld tasks.

Remove access to historical replay and task-specific memory for weights-only conditions.

Also evaluate full-lifetime transfer separately.

Measure both new-task acquisition and historical retention.

Run sequential learning across multiple domains.

Test whether prior experience makes future learning faster without catastrophic loss of earlier capabilities.

## Experiment 4 — Consequential long-horizon transfer

Use the full OpenWorld tasks in Part 16.

Begin with the architecture's current competence.

Run repeated lifetimes.

Increase research scale progressively toward:

- 10 lifetimes.
- 30 lifetimes.
- 100 lifetimes.

Use different world instances and objectives.

This is a target research scale, not permission to exceed available compute or spending limits.

At each checkpoint, evaluate on untouched unfamiliar tasks.

The decisive metric is:

**Does accumulated experience make CAM capable of accomplishing unfamiliar, difficult goals with substantially fewer interactions, less elapsed time or less compute?**

---

# PART 18 — DETAILED STRATEGY AND LEARNING OBSERVABILITY

A final aggregate reward score is not sufficient.

We must be able to understand how the agent's strategies develop.

## 18.1 Decision traces

For every consequential decision, record:

- Observable context.
- Current objective.
- Relevant retrieved experiences.
- Candidate actions or generated action sequence.
- Predicted outcomes.
- Predicted uncertainty.
- Selected action.
- Alternative imagined trajectories.
- Actual execution result.
- Fast-belief changes.
- Subsequent observed consequences.
- Policy version.
- Learning-update references.

Store machine-readable traces.

Do not require the agent to generate elaborate natural-language explanations at every step.

Its internal reasoning is numerical.

Generate retrospective readable summaries from actual recorded actions and predictions.

Label them as interpretations, not direct access to the model's thoughts.

## 18.2 Neural learning analysis

Track:

- Parameter changes by component.
- Gradient magnitudes.
- Gradient alignment.
- Representation changes.
- Predictive improvements.
- Changes in action distributions.
- Memory retrieval usefulness.
- Fast-adaptation behavior.
- Long-term retention.

Inspect checkpoints before and after important learning periods.

Use controlled probes to test whether neural changes actually improve decisions.

## 18.3 Strategy evolution

Produce actual case studies of individual OpenWorld lifetimes.

For example:

- What did CAM initially attempt?
- Which attempts failed?
- What did the model predict incorrectly?
- What information changed its behavior?
- Did it invent a better strategy?
- Did repeated experience improve subsequent attempts?
- Did the improvement persist without retrieval of previous experiences?
- Did it transfer to an unrelated goal?

Do not fabricate narrative explanations that are unsupported by the trace.

## 18.4 Learning dashboard

Implement a research dashboard or generated HTML report containing:

- Active worlds and goals.
- Current checkpoint.
- Experience counts.
- Learning curves.
- World-model calibration.
- Retention and transfer metrics.
- Real versus imagined outcomes.
- Action timelines.
- Relevant model updates.
- Compute consumption.

Make it possible to examine a specific CAM lifetime in detail.

---

# PART 19 — MANDATORY BASELINES AND SCIENTIFIC RIGOR

Compare against:

1. Existing CAM 6.1.
2. Existing CAM 7.0.
3. Existing model-based learner.
4. Exact Bayesian regression on appropriate numerical tasks.
5. Frozen pretrained semantic agent with tools.
6. Frozen semantic agent with persistent memory.
7. Frozen semantic agent with matched planning compute.
8. CAM Unified without continual weight updates.
9. CAM Unified without fast adaptation.
10. CAM Unified without imagination.
11. CAM Unified without learned exploration.
12. CAM Unified without historical memory.
13. CAM Unified with target-only training.
14. CAM Unified with cross-domain pretraining.

Use applicable baselines for each environment.

Do not require a numerical controller to operate websites it cannot represent.

Match permissions, task information, relevant pretrained knowledge, environment interaction budgets and compute where possible.

Report unavoidable mismatches.

Use independent training seeds and evaluation worlds.

Prespecify confirmation criteria before evaluating untouched sets.

Do not repeatedly inspect test worlds and continue calling them held-out.

Use confidence intervals that account for training-seed and world variation.

Report null and negative results.

Separate:

- More data.
- More neural training compute.
- More inference-time search.
- Better sample efficiency.
- Better final competence.

A model that improves only because it consumes ten times more computation has not necessarily become a more efficient learner.

---

# PART 20 — HARD ENGINEERING ACCEPTANCE REQUIREMENTS

Create executable integration tests for every requirement.

The full architecture is not complete until these requirements have been exercised.

| ID | Requirement | Required evidence |
|---|---|---|
| A01 | Deep main model | Actual 16-layer main configuration instantiated and run |
| A02 | Semantic understanding | Unseen document, code and tool-schema tasks |
| A03 | Numerical latent reasoning | Reasoning iterations affect actual decisions |
| A04 | Generalized entities | Variable entities and relationships processed |
| A05 | Fast adaptation | New observations alter predictive beliefs |
| A06 | Multiple hypotheses | Competing mechanism beliefs update with evidence |
| A07 | Causal dynamics | Different actions produce different future-state predictions |
| A08 | Multistep imagination | Real multistep belief-aware rollouts affect action choice |
| A09 | Learned exploration | Trainable exploratory decisions receive outcome gradients |
| A10 | General action generation | Novel executable code and valid tools used |
| A11 | Lifetime ingestion | No 256-row or equivalent lifetime cutoff |
| A12 | Durable episodic memory | Old experience survives process restart |
| A13 | Continual neural learning | Verified shared-core weight updates from real experience |
| A14 | Plasticity monitoring | Sequential retention and new-task acquisition measured |
| A15 | Weights-only evaluation | No optimizer, replay or context leakage |
| A16 | Distributed learning | Multiple actors contribute to shared neural updates |
| A17 | OpenWorld adapter | Actual full environment used |
| A18 | OpenWorld goal execution | At least one complete independently evaluated long-horizon run |
| A19 | Cross-domain evaluation | Unfamiliar task families tested |
| A20 | Strategy observability | Recorded actions, predictions, outcomes and updates |
| A21 | Durable recovery | Verified restart and checkpoint recovery |
| A22 | Full integration | Semantic input through neural decision, execution, outcome and subsequent weight update |

The independent verification agent owns these acceptance tests.

The implementing agents cannot mark their own work complete without review.

Tests may use small configurations for speed, but the main configuration must be separately exercised.

Do not count mock environments as proof of full OpenWorld integration.

Do not count code generation by a completely independent frozen LLM as learned CAM action generation.

Do not count successful disk storage as demonstrated neural memory.

Do not count a one-step regression planner as multistep imagination.

Do not count a checkpoint save as evidence of continual learning.

Do not count numerical mechanism tasks as the complete OpenWorld experiment.

---

# PART 21 — EXECUTION SCHEDULE

Implement in coordinated stages, with parallel workstreams within each stage.

## Stage 0: Recovery and architecture contract

- Recover the repository and existing results.
- Establish persistent ledger and completion manifest.
- Inspect available hardware and storage.
- Define shared event, observation, action, belief and latent tensor interfaces.
- Assign workstreams and file ownership.

Do not spend excessive time writing plans instead of implementing.

## Stage 1: Complete vertical OpenWorld learning path

Build a minimal but genuine end-to-end path using the new architecture:

Semantic observation → latent model → generated action → real Crucible execution → actual outcome → persistent experience → neural-weight update.

This must use the actual OpenWorld environment.

In parallel, implement the larger backbone, memory system, dynamics model and distributed infrastructure.

The master agent must ensure early compatibility.

## Stage 2: Integrated architecture

Connect:

- Deep latent backbone.
- Semantic model.
- Fast adaptive beliefs.
- Persistent memory.
- Action-conditioned dynamics.
- General action generation.
- Actor-critic learning.
- Learned exploration.
- Multistep imagination.

Run the complete integration tests.

Do not postpone missing subsystems indefinitely.

## Stage 3: Main-model training

Instantiate the main 16-layer model.

Run semantic grounding and world-model training.

Run real OpenWorld interaction and continual updates.

Collect detailed learning traces.

Benchmark throughput and memory usage.

If resource limits prevent the intended scale, document the exact blocker and preserve the full architecture.

## Stage 4: Controlled learning experiments

Run one-shot adaptation, causal discovery, negative-transfer diagnostics and retention tests.

Run these alongside OpenWorld experiments.

Do not stop OpenWorld development because a controlled numerical experiment fails.

## Stage 5: Difficult OpenWorld lifetimes

Run actual long-horizon goals.

Train across diverse lifetimes.

Observe strategies and changes in behavior.

Run distributed actor learning.

Compare pretrained and fresh models.

## Stage 6: Improvement

Analyze the strongest remaining failures.

Run discriminating ablations.

Improve learning efficiency, semantic grounding, prediction, memory and generalization.

Do not defend the initially proposed architecture if a tested alternative performs better.

Preserve the mandatory capabilities while allowing mathematically justified architectural improvements.

## Stage 7: Final independent audit

The master agent and independent reviewer must inspect every requirement in the completion manifest.

For every incomplete item:

- Identify what is missing.
- Identify why.
- Continue implementation if resources permit.
- Preserve a specific resumable task if genuinely blocked.

Do not write a final report declaring the architecture complete while substantial required components remain unimplemented.

If interrupted by usage, compute, storage or rental limitations, save all progress and clearly state that implementation remains incomplete.

---

# PART 22 — COMPUTE EFFICIENCY

We want substantial neural capacity without unnecessarily expensive experiments.

Use:

- Mixed precision.
- Efficient attention.
- Batched actor inference.
- Shared semantic-model serving.
- Gradient accumulation.
- Cached immutable semantic representations where valid.
- Asynchronous CPU environment simulation.
- Durable replay streaming.
- Selective high-cost reasoning.
- Event-driven processing.
- Latent imagination instead of repeated real simulation where validated.

Do not cache embeddings produced by continuously changing encoders without addressing representation versioning.

Do not invoke a large language model for every trivial numerical observation.

Do not assume GPU parallelism makes the simulator or data pipeline infinitely scalable.

Benchmark before scaling.

Record actual GPU time, CPU time, inference tokens, environment steps, consequential decisions, simulated elapsed time, memory consumption and monetary cost where available.

Respect existing authorized compute and spending limits.

Do not procure additional hardware or paid services without authorization.

---

# PART 23 — COMPLETION DELIVERABLES

Produce:

1. The fully integrated CAM Unified architecture.
2. Actual neural implementation and mathematical tensor interfaces.
3. Semantic text, code and website grounding.
4. Deep high-dimensional latent reasoning.
5. Multiple-hypothesis adaptive inference.
6. Stochastic action-conditioned world-model learning.
7. Genuine multistep belief-aware imagination.
8. Learned exploratory decision-making.
9. General-purpose executable action generation.
10. Scalable lifetime memory.
11. Persistent continual shared-core neural updates.
12. Distributed actor-learner implementation.
13. Full Crucible OpenWorld integration.
14. Difficult long-horizon task protocols.
15. Actual OpenWorld runs.
16. Foundational controlled learning experiments.
17. Corrected weights-only transfer experiments.
18. Sequential retention and plasticity evaluations.
19. Detailed strategy and learning traces.
20. Durable checkpoints and raw results.
21. Independent acceptance-test evidence.
22. Reproducible commands and configurations.
23. A complete final research report.
24. A persistent completion manifest and continuation ledger.

The final report must distinguish:

- Architectural capability implemented.
- Architectural capability integration-tested.
- Scientific benefit demonstrated.
- Scientific benefit inconclusive.
- Scientific benefit failed.
- Experiment not attempted.
- Genuine resource blocker.

The project must not end with another partially implemented numerical controller being presented as the complete Unified Experience Superlearner.

---

# FINAL DIRECTIVE

Our goal is not to establish that CAM can adjust numerical levers more effectively.

Our goal is to investigate whether a sufficiently expressive neural intelligence can continuously transform semantic and environmental experience into deeper, reusable knowledge about how the world works.

It should become increasingly capable of deciding:

- What it does not understand.
- What information matters.
- Which experiment to perform.
- What the consequences of an action might be.
- Which strategy to pursue.
- What to invent.
- How to adapt after failure.
- How to transfer previous learning to unfamiliar problems.

The core neural weights must progressively acquire these capabilities.

Fast adaptive beliefs, episodic memory and pretrained semantic models are complementary tools, not substitutes for persistent neural learning.

No known architecture guarantees the desired general intelligence or learning rate. Treat the proposal as a serious empirical research program.

Build the complete system.

Run it in the actual OpenWorld.

Observe what it does.

Measure what it learns.

Identify what remains broken.

Improve it.

Use coordinated Opus workstreams with a master integration agent and independent verification.

Begin by recovering the latest repository state, establishing the shared architecture interfaces and completion manifest, launching the implementation workstreams, and building the first genuine end-to-end OpenWorld learning path.

Do not stop after planning.

START IMPLEMENTATION NOW.
