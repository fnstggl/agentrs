<!-- The operator's CAM Live 1.0 specification, verbatim (2026-11-26). Governing document: do not edit. -->

# CAM LIVE 1.0 — REAL-WORLD CONTINUAL EXPERIENCE SUPERLEARNER

## CORE MISSION

Build CAM Live 1.0: a genuinely continually learning neural intelligence capable of autonomously pursuing difficult real-world objectives, learning from experimentation and actual environmental consequences, and accumulating persistent, transferable intelligence in its neural weights.

This is a standalone research and engineering project within the existing AgentRS repository:

https://github.com/fnstggl/agentrs

Do not wait for CAM 7.0 Unified or Crucible 5.0 to be completed.

Inspect and reuse existing components where they are useful, but CAM Live must be independently functional.

The ultimate research objective is:

**Maximize the rate at which real-world experience produces persistent, transferable intelligence.**

The first real-world objective will be:

**Create and operate a genuinely useful Discord community that attracts 100 distinct real human members who voluntarily choose to join.**

CAM must independently determine:

- What kind of community to create.
- What niche or problem to address.
- What people might find genuinely useful.
- What resources, services or experiences to create.
- What experiments to conduct.
- What strategies to attempt.
- How to improve its community.
- How to distribute useful resources through authorized channels.
- How to interpret successes and failures.
- Which experiences deserve reinforcement.
- What it needs to learn next.

Do not supply a Discord-growth strategy.

Do not hardcode a particular niche, posting schedule, content strategy, community format or business model.

CAM should discover its own approaches through research, experimentation, feedback and learning.

The Discord objective is only the first experiment in the lifetime of this intelligence.

Subsequent objectives may involve creating useful software, attracting voluntary product users, building a product waitlist, developing a sustainable business, conducting scientific research or accomplishing other unfamiliar consequential tasks.

The same neural intelligence and acquired weights should persist across these objectives.

The primary scientific metric is:

**How many new experiences are required to acquire competence on a previously unfamiliar task?**

The first experiment is not complete merely because an agent reaches 100 members.

We must also establish whether its experiences produce meaningful neural learning and whether that learning transfers.

---

# PART 0 — IMPLEMENTATION GOVERNANCE

## 0.1 Inspect the current repository

Before implementing:

- Inspect all relevant local and remote branches.
- Identify completed and unfinished CAM 7.0, CAM Unified, CAM 6.1 and Crucible components.
- Inspect available pretrained models.
- Inspect the actual neural training infrastructure, tool interfaces, experience storage and checkpoint mechanisms.
- Review previous scientific failures.
- Verify available CPUs, GPUs, RAM, VRAM and storage.
- Preserve valuable existing work and checkpoints.
- Establish a reproducible baseline.

Create a separate branch:

`feat/cam-live`

Choose its base from the latest stable, appropriate repository state.

Do not introduce dependencies on unfinished components that prevent CAM Live from running independently.

Preserve compatibility with future CAM Unified integrations wherever practical.

Do not overwrite existing experiments or modify sealed evaluation sets.

Do not push externally without authorization.

## 0.2 Master orchestration agent

Use a master architecture and integration agent coordinating specialized implementation subagents.

Use the strongest available appropriate Claude model for the master and mathematical implementation workstreams.

Inspect actual support for parallel subagents, agent teams and isolated worktrees.

Where parallel execution is unavailable, preserve the same workstream responsibilities sequentially.

The master agent is responsible for shared tensor contracts, software interfaces, mathematical consistency, integration, experimental rigor and final completion.

### Workstream A — Neural architecture

Implement the pretrained semantic backbone, trainable adapters, unified experiential intelligence, shared representations and fast adaptive numerical state.

### Workstream B — Predictive learning and decision-making

Implement action-conditioned world modeling, uncertainty estimation, goal-conditioned outcome prediction, learned experimentation, planning and the trainable action policy.

### Workstream C — Intrinsic rewards and meta-learning

Implement the learned reward generator, actual inner policy updates, outer meta-gradient optimization, self-evaluation, autonomous learning objectives and reward integrity.

### Workstream D — Retrospective credit and continual learning

Implement delayed outcome reconciliation, return decomposition, counterfactual credit estimation, persistent replay, neural updates, retention and checkpoint management.

### Workstream E — Internet harness

Implement general-purpose internet interaction interfaces, authorized browser and terminal execution, service connectors, permissions, approvals, safety boundaries and real-world observation ingestion.

### Workstream F — Live experiment runtime

Implement persistent autonomous operation, task management, event-driven learning, Discord experiment integration, experiment tracking, operator controls, dashboards and recovery.

### Workstream G — Independent scientific verification

Independently inspect the actual implementation, execute tests, verify gradient pathways, evaluate baselines, check experimental contamination and audit every claimed capability.

Subagents are development collaborators.

They are not separate cognitive agents inside CAM.

The resulting intelligence should remain a unified neural learning architecture.

## 0.3 Mandatory completion manifest

Create:

`docs/CAM_LIVE_COMPLETION_MANIFEST.json`

Every mandatory requirement in this specification must have:

- A unique identifier.
- An owning workstream.
- Implementation files.
- Runtime integration points.
- Acceptance tests.
- Reproduction commands.
- Actual executed test results.
- Relevant experimental evidence.
- Independent verification status.
- Remaining limitations.

Implement an executable completion checker.

The master agent cannot declare engineering completion if a mandatory component exists only as a placeholder, disconnected module, mocked demonstration, unexecuted test or architectural description.

Distinguish:

1. Engineering implementation complete.
2. Controlled experimental validation complete.
3. Ready for authorized live deployment.
4. Live experiment executed.
5. Scientific hypothesis supported.

These are separate statuses.

A functioning architecture does not automatically establish scientific superiority.

---

# PART 1 — FUNDAMENTAL NEURAL ARCHITECTURE

## 1.1 Model size

Implement the following initial configuration.

| Component | Initial target |
|---|---:|
| Pretrained semantic backbone | 4B–8B parameters |
| Unified experiential intelligence | Approximately 50M trainable parameters |
| Distributed semantic adapters | Approximately 10M–20M trainable parameters |
| Persistent trainable total | Approximately 60M–70M parameters |
| Fast adaptive belief state | Configurable numerical state |
| Future trainable capacity | Configurable from 5M to 1B+ parameters |

These are initial engineering targets, not proven optimal sizes.

Report actual parameter counts separately for:

- Frozen parameters.
- Trainable experiential parameters.
- Trainable pretrained-model adapters.
- Reward-function parameters.
- Other trainable output heads.
- Total persistent trainable parameters.

Do not silently substitute a tiny controller for the specified architecture.

Support smaller configurations for controlled mathematical experiments and debugging.

Support larger configurations without redesigning the fundamental learning algorithm.

## 1.2 Pretrained semantic backbone

Use a suitable open-weight 4B–8B pretrained language and code model.

Inspect the available hardware, models and licensing before selection.

The pretrained model supplies existing knowledge of language, software, documents, tools and the world.

Its capabilities must support:

- Understanding websites and documents.
- Interpreting tool outputs.
- Understanding semantic objectives.
- Generating original content.
- Generating executable software.
- Constructing tool arguments.
- Interpreting unfamiliar observations.

Initially freeze the majority of the pretrained parameters.

Insert trainable LoRA or similarly justified adapters across multiple transformer layers.

Target approximately 10M–20M trainable adapter parameters.

These adapters must receive genuine learning gradients from real experience.

Do not merely attach adapters without using them during inference or training.

The pretrained semantic model should become selectively plastic through experience.

## 1.3 Unified experiential intelligence

Implement a shared, trainable transformer-based experiential backbone of approximately 50M parameters.

A reasonable initial configuration is a transformer with:

- Approximately 768-dimensional hidden representations.
- Approximately six transformer blocks, adjusted to achieve the actual parameter target.
- Multihead attention.
- RMSNorm.
- SwiGLU or another justified gated feed-forward architecture.
- Temporal and event position representations.
- Recurrent or persistent latent state.
- Cross-attention to semantic and structured observation tokens.

Determine the final configuration from actual parameter counts and computational benchmarks.

The experiential backbone should support:

- World understanding.
- Action-conditioned prediction.
- Numerical reasoning.
- Goal-conditioned evaluation.
- Uncertainty estimation.
- Experiment selection.
- Action selection.
- Learning-value estimation.
- Retrospective experience learning.

These functions should share representations rather than operate as independent language-model agents.

Lightweight specialized output heads are appropriate.

Do not force every capability into a separate handcrafted cognitive subsystem.

## 1.4 Avoid a frozen-LLM wrapper

The pretrained language model must not make every consequential decision while CAM's trainable network merely accepts or rejects its outputs.

The trainable experiential intelligence must meaningfully influence:

- Which information is investigated.
- Which actions are considered.
- Which experiments are selected.
- Which strategies are pursued.
- Which predicted consequences matter.
- How evidence changes future decisions.

Initially, the pretrained model may help generate candidate actions.

The trainable policy must evaluate and select among them.

Develop trainable conditioning from CAM's latent state into the semantic action-generation process.

Support gradual improvement of the action generator through the pretrained-model adapters.

Measure whether changing CAM's experiential weights changes its decisions while holding the original pretrained weights fixed.

---

# PART 2 — THREE TIMESCALES OF NEURAL PLASTICITY

CAM must learn at multiple timescales.

## 2.1 Immediate learning: fast adaptive numerical beliefs

After receiving new evidence, CAM should immediately update its numerical understanding of the current environment.

This must not require a full neural retraining cycle.

Define:

`b_t = F_θ(H_t, m_t)`

Where:

- `H_t` is legitimately observable experience history.
- `m_t` is the current fast adaptive state.
- `θ` is persistent neural knowledge.
- `b_t` is the integrated latent representation of the situation.

Implement a learned feature representation supporting efficient Bayesian-style adaptation.

A suitable starting mechanism is ALPaCA-inspired Bayesian regression over learned nonlinear features.

For a selected prediction target:

`φ_t = Φ_θ(b_t, a_t, Δt)`

Represent the uncertain mapping from learned features to outcomes with:

`w ~ N(μ_t, Σ_t)`

For an observed scalar outcome `y_t`:

`error = y_t - φ_tᵀ μ_t`

`K_t = Σ_t φ_t / (σ² + φ_tᵀ Σ_t φ_t)`

`μ_(t+1) = μ_t + K_t error`

`Σ_(t+1) = Σ_t - K_t φ_tᵀ Σ_t`

Use numerically stable covariance or precision updates.

Use appropriate likelihoods and approximations for count, binary, survival and other non-Gaussian outcomes.

Do not force every internet observation into a Gaussian regression model.

Support uncertainty over changing environmental mechanisms.

Compare Bayesian adaptation against recurrent adaptation and simpler statistical baselines.

The fast state should influence actual decision-making.

## 2.2 Frequent learning: persistent experiential weights

CAM must regularly update its persistent experiential neural parameters from actual observations and consequences.

Implement gradient-based training of:

- Shared experience representations.
- Predictive world models.
- Goal-conditioned evaluators.
- Trainable policies.
- Learned adaptation priors.
- Action embeddings.
- Relevant latent dynamics.

Training should be event-driven or based on the accumulation of informative matured observations.

Do not require an entire long-horizon task to finish before any neural learning occurs.

Do not update aggressively from every correlated observation.

Implement appropriate learning rates, replay, gradient clipping, uncertainty weighting and validation.

## 2.3 Slower learning: semantic adapters and intrinsic rewards

Update pretrained-model adapters on a slower schedule using appropriately aggregated experience.

The objective is to allow experience to improve deeper semantic understanding, not just the final action-selection layer.

Train the intrinsic reward function through actual meta-learning.

Its update schedule should depend on whether sufficient subsequent outcome evidence exists.

Use configurable learning schedules.

Track changes in parameter norms, prediction accuracy, action behavior and retention.

Prevent unstable online updates from permanently damaging useful pretrained capabilities.

---

# PART 3 — HIGH-DIMENSIONAL EXPERIENCE REPRESENTATION

CAM's intelligence must not depend on expressing all reasoning in natural language.

Use high-dimensional numerical representations for internal prediction, evaluation, adaptation and decision-making.

Semantic information grounds the representation.

Exact structured evidence preserves factual precision.

## 3.1 Experience tokens

Implement representations for:

- Observations.
- Actions.
- Goals.
- Documents.
- Software.
- Entities.
- Relationships.
- Environmental events.
- Time.
- Costs.
- Resources.
- Outcomes.
- Uncertainty.
- Available tools.
- Previous attempts.

Preserve raw values alongside embeddings when appropriate.

For example, if an action produces 100 visits, three voluntary signups and a measured cost, these quantities must remain available precisely.

Do not require the network to reconstruct exact numbers from an approximate language embedding.

## 3.2 Domain-independent latent learning

Represent experience using:

`z_t = F_θ(H_t, a_t, g)`

Train the model to predict distributions of future observable consequences:

`p_θ(Y_future | z_t, continuation_policy, horizon)`

The representation should learn useful abstractions that transfer across objectives.

For example, software adoption, voluntary community membership and product usage may contain shared underlying relationships.

Do not explicitly prescribe what those relationships must be.

Allow useful numerical representations to emerge from predictive and decision-related training.

Do not hardcode marketing frameworks, psychological classifications or Discord-specific strategies.

The architecture should work when the next goal is an unfamiliar technical, scientific or economic objective.

---

# PART 4 — ACTION-CONDITIONED WORLD MODEL

CAM must learn what happens as a consequence of acting.

Before performing an action, it should make probabilistic predictions about its likely consequences.

## 4.1 Predictive objectives

Implement:

`p_θ(o_future | b_t, a_t, Δt)`

Support predictions at multiple time horizons.

The model should be capable of predicting:

- Immediate tool execution outcomes.
- Changes in observable environmental state.
- Delayed responses.
- Resource consumption.
- Adoption or usage.
- Relevant economic outcomes.
- Goal progress.
- Uncertainty.

Do not concentrate the entire world-model objective on whether a tool call is syntactically valid.

The important objective is predicting consequential changes in the world.

## 4.2 Learn from actual evidence

When an outcome becomes observable, train predictions against that outcome.

Use appropriate probabilistic objectives.

For example:

`L_world = -log p_θ(y_actual | b_t, a_t, horizon)`

Use masks for unobserved or pending outcomes.

Never invent observations to complete a training target.

Distinguish:

- Prediction errors.
- Irreducible randomness.
- Missing observations.
- Incorrect causal assumptions.
- Changes in environmental conditions.

## 4.3 Imagination and planning

Implement limited learned-model imagination.

CAM should be able to evaluate possible future consequences of alternative feasible actions before choosing one.

Use uncertainty-aware multistep prediction.

Begin with short prediction horizons and increase them only where the model is demonstrably accurate enough.

Predicted experiences must remain distinguishable from real experience.

Imagined trajectories cannot be used as proof of actual objective achievement.

Do not allow the model to reinforce fabricated successes simply because it predicted them.

---

# PART 5 — GENERAL-PURPOSE ACTION GENERATION

CAM must not operate through a fixed menu of Discord-growth actions.

It should receive a description of the tools available in the current environment and generate executable actions dynamically.

Actions may include:

- Researching information.
- Reading documents.
- Writing software.
- Creating useful digital resources.
- Constructing applications.
- Updating owned websites.
- Conducting permitted experiments.
- Publishing approved content.
- Operating an authorized community.
- Gathering permitted analytics.
- Revising previous work.
- Waiting for delayed outcomes.

Use the pretrained semantic model for candidate generation where appropriate.

Encode candidate actions numerically.

The experiential policy should select among feasible candidates.

A suitable initial formulation is:

`π_θ(a_i | b_t, g, C_t)`

Where `C_t` is the current set of feasible candidate actions.

Track candidate-generation provenance and selection probabilities.

Allow novel multi-step programs to be proposed and executed through authorized tools.

Do not silently constrain all possible solutions to a few predetermined strategies.

## 5.1 Exploration

Implement uncertainty-aware exploration.

Initially compare:

- Thompson sampling.
- Upper-confidence-bound selection.
- Learned action value.
- Decision-relevant information gain.
- Randomized controlled experiment selection.

The agent should balance making progress toward its objective with learning about unfamiliar mechanisms.

Exploration should become increasingly informed through experience.

The same experiment-selection principles must be applicable to objectives beyond community growth.

---

# PART 6 — LEARNED GOAL-CONDITIONED SELF-EVALUATION

CAM must internally evaluate whether its experiences are useful.

Do not implement self-evaluation as a language model arbitrarily assigning a score from one to ten.

Implement a trainable goal-conditioned outcome predictor operating on CAM's shared latent representation.

## 6.1 Probability of success

Define:

`p_t = V_θ(b_t, g, horizon, policy_context)`

For a binary independently verifiable objective:

`Y_g = 1` when the objective is genuinely achieved.

`Y_g = 0` when the objective has definitively failed under its evaluation conditions.

Pending objectives remain pending.

When eventual outcomes become available, train the evaluator using:

`L_eval = -Y_g log(p_t) - (1 - Y_g) log(1 - p_t)`

Use numerically stable binary cross-entropy with logits.

Train against earlier trajectory prefixes.

If CAM repeatedly predicts an 80% probability of success but fails, its evaluator must receive corrective gradients.

The evaluator's assessment must be falsifiable by reality.

## 6.2 Distributional outcome evaluation

Predict more than binary success.

Support estimates of:

- Magnitude of potential achievement.
- Time to achievement.
- Resource expenditure.
- Sustainability.
- Relevant future outcomes.
- Uncertainty.

Use appropriate likelihoods for each target.

Where the outcome is a count, use a suitable probabilistic count model.

Where outcomes are delayed or censored, use appropriate time-to-event or censoring-aware methods.

## 6.3 Evaluate intermediate progress

CAM should learn how experiences change its prospects of achieving its objective.

A useful input to the internal reward generator is:

`ΔV_t = V_θ(b_next, g) - V_θ(b_prev, g)`

This represents changed predicted prospects.

It does not automatically represent the causal contribution of CAM's action.

Distinguish:

- Actual observed progress.
- Predicted future progress.
- Quality of the decision.
- Information gained.
- Long-term learning value.

The evaluator should learn that an unsuccessful experiment can provide valuable information.

It should also learn that a lucky success does not necessarily establish that a strategy was effective.

## 6.4 Calibration

Track:

- Brier score.
- Negative log-likelihood.
- Calibration slope and intercept.
- Reliability curves.
- Calibration across different objectives and horizons.
- Prediction uncertainty.

Predictions must be timestamped before outcomes arrive.

Do not evaluate predictions retrospectively using information that was unavailable when they were made.

---

# PART 7 — CAM MUST LEARN ITS OWN REWARD FUNCTION

THIS IS A CORE REQUIREMENT.

Do not require Crucible pretraining.

Begin developing and training the internal reward function using genuine experience from the first authorized real-world experiment.

Do not substitute a fixed reward formula and call it meta-learning.

## 7.1 Research foundation

Study the actual mathematical approach in:

**What Can Learned Intrinsic Rewards Capture?**

Zheng, Oh, Hessel, Xu, Kroiss, van Hasselt, Silver and Singh.

https://proceedings.mlr.press/v119/zheng20b.html

https://proceedings.mlr.press/v119/zheng20b/zheng20b-supp.pdf

Implement the relevant lifetime meta-gradient mechanism.

The intrinsic reward function itself should become a locus of learned neural knowledge.

Its effectiveness must be determined by the learning and behavior it produces.

## 7.2 Internal reward generator

Define:

`r_internal_t = R_ψ(b_t, a_t, b_next, g, uncertainty, evidence, learning_context)`

The trainable reward function should consume representations of:

- Previous understanding.
- Action selected.
- Actual observed consequences.
- Updated understanding.
- Current objective.
- Estimated future success.
- Estimated action quality.
- Information acquired.
- Resource expenditure.
- Prediction error.
- Relevant uncertainty.
- Lifetime learning context.

Use a learned nonlinear function.

Do not permanently prescribe the relative value of these quantities.

The reward function should discover what deserves reinforcement.

Maintain appropriate output bounds, normalization, versioning and training stability.

Do not let the policy directly increase its reward by manipulating the reward network's parameters.

## 7.3 Inner learning loop

CAM's policy and experiential intelligence learn using internally generated rewards.

Define:

`θ_next = θ - α ∇_θ L_inner(θ; ψ, D_support)`

Where:

- `θ` contains the relevant trainable learner parameters.
- `ψ` contains intrinsic reward parameters.
- `D_support` contains genuine observed experience.
- `L_inner` depends meaningfully on the generated intrinsic rewards.

Implement a real differentiable inner policy update.

The reward generator must influence actual neural updates.

The inner loop must not update reward parameters simply to maximize their outputs.

## 7.4 Outer meta-learning loop

Evaluate whether learning from the intrinsic rewards improves subsequent independently verified performance.

Define:

`J_outer(ψ) = E[U_verified(D_query; θ_next(ψ))]`

Optimize:

`ψ ← ψ + β ∇_ψ J_outer`

The query experience must be collected after the relevant inner update or evaluated using a mathematically justified estimator with valid logged policy support.

For nondifferentiable internet interaction, use an appropriate policy-gradient surrogate.

Do not attempt to differentiate through website execution, human decisions or external APIs.

The necessary computational path is:

`Reward parameters → inner neural update → changed policy → subsequent observed outcomes → outer reward update`

Verify this path explicitly.

Do not detach tensors in a way that eliminates the required meta-gradient.

## 7.5 Meta-learning directly from the first live experiment

One long-running Discord experiment is not equivalent to thousands of independent training lifetimes.

Implement sequential and appropriately separated learning windows within the authorized experiment.

These may contain:

- Distinct experiments.
- Different independently measurable intermediate objectives.
- Different observation periods.
- Support and subsequent query experience.
- Changed environmental conditions.
- Different permitted tasks or resource configurations.

Where valid, use randomized experiments on owned resources with voluntary participants.

Do not treat correlated observations from the same action as independent lifetimes.

Use independent genuine outcomes to supervise the meta-learning process.

The learner must begin acquiring real experience without requiring the reward network to already be competent.

## 7.6 Cold-start learning

A randomly initialized reward generator is not yet a reliable learning signal.

Implement an explicit bootstrap mode using:

- Self-supervised prediction of observed outcomes.
- Independently grounded goal evaluation.
- Uncertainty-aware exploration.
- Sparse verified outcome supervision where necessary.

The bootstrap signals should stabilize learning while the intrinsic reward network acquires usable knowledge.

Do not describe bootstrap rewards as already meta-learned.

Once sufficient valid query experience accumulates, begin intrinsic reward meta-updates.

Gradually evaluate and promote the learned reward function.

If insufficient experience exists for reliable meta-learning, continue collecting experience under the approved policy and report the limitation.

Do not fabricate evidence or pretend that the reward function has learned.

## 7.7 Efficient meta-gradient implementation

Implement truncated differentiation through a configurable number of inner updates.

Begin with 1, 2, 4 and 8 inner-update steps.

Use functional model updates and carefully selected adaptable parameter subsets.

Do not require differentiating through every parameter of the entire 4B–8B backbone for every meta-update.

Investigate:

- Truncated unrolling.
- Gradient checkpointing.
- Small functional inner optimization steps.
- Validated lower-cost approximations.

Test the exact mathematical implementation on small diagnostic problems before scaling.

Verify meta-gradient correctness through finite differences or analytically tractable reference experiments.

Controlled diagnostics do not require Crucible.

They are algorithm correctness tests, not substitutes for live training.

## 7.8 Reward integrity

CAM must never be able to change its ultimate success criteria merely to obtain more reward.

Its predictions must not count as externally verified outcomes.

Its reward function must be evaluated according to the behavior and capabilities it produces.

Prevent:

- Inflated self-evaluation.
- Arbitrarily increasing intrinsic reward magnitude.
- Repeated trivial self-generated objectives.
- Duplicate counting of outcomes.
- Reinforcement from fabricated events.
- Rewarding irreducible randomness indefinitely.
- Manipulation of verification records.
- Optimization of engagement statistics without genuine participation.

The independent objective verifier and permission boundaries must remain outside the learned reward system.

---

# PART 8 — LEARN THE VALUE OF INFORMATION AND EXPERIMENTATION

CAM must discover which experiences are worth acquiring.

A failed attempt may be more educational than a lucky success.

Distinguish:

- Surprise.
- Belief change.
- Predictive learning progress.
- Reduction in meaningful uncertainty.
- Decision-relevant knowledge.
- Persistent improvement in competence.

Do not treat raw prediction error as inherently valuable.

Random unpredictable observations could otherwise attract endless exploration.

Implement an initial decision-relevant information-value estimate.

A candidate approximation is:

`VOI(a) = E_y[max_a' Q(b_updated(a,y), a',g)] - max_a' Q(b_current,a',g)`

Account for the action's effects on the world, resource cost, elapsed time and opportunity cost.

Train a lightweight estimator of how useful acquired knowledge will be for subsequent decisions.

Periodically test this estimate against actual downstream learning.

The intrinsic reward network should learn to use this information.

Ultimately, valuable experience is experience that improves later verified performance, not merely experience that changes the model's beliefs.

---

# PART 9 — RETROSPECTIVE CREDIT ASSIGNMENT

THIS IS MANDATORY.

Real-world consequences are often delayed.

A resource created today may produce an outcome several days or weeks later.

CAM must reconsider the significance of earlier decisions when later evidence arrives.

## 9.1 Durable event histories

Record:

- Original actions.
- Time of execution.
- Available information at decision time.
- Predicted outcomes.
- Related entities.
- Artifacts created.
- Subsequent observations.
- Actual eventual outcomes.
- Policy and model versions.

Link delayed outcomes to relevant earlier experiments and actions where evidence supports that relationship.

Preserve uncertainty when attribution is ambiguous.

Do not assume that the most recent action caused every subsequent success.

## 9.2 RUDDER-style return decomposition

Study:

**RUDDER: Return Decomposition for Delayed Rewards**

https://arxiv.org/abs/1806.07857

Implement a trainable sequence predictor of eventual independently grounded outcomes.

Let:

`F_ξ(H_t,g)` predict eventual return from a trajectory prefix.

A candidate redistributed contribution is:

`c_t = F_ξ(H_next,g) - F_ξ(H_t,g)`

Implement appropriate initial offsets and residual corrections.

Verify return conservation under the actual discounting convention.

Revisit old experiences when outcomes mature.

Use redistributed credit during eligible training updates.

Do not fabricate extra verified returns through reward redistribution.

## 9.3 Counterfactual credit assignment

Study:

**Would I Have Gotten That Reward? Long-Term Credit Assignment by Counterfactual Contribution Analysis**

https://arxiv.org/abs/2306.16803

https://github.com/seijin-kobayashi/cocoa

Implement a mathematically valid finite-action COCOA reference estimator.

Then implement an appropriate CAM extension for probability-tracked candidate actions.

Estimate whether an observed outcome would have been more or less likely under an alternative action.

For the appropriate reference setting:

`w(b,a,u) = Pπ(a | b,u_future) / π(a | b) - 1`

Train the hindsight model using legitimate retrospective training experience.

Use recorded behavior-policy probabilities.

Preserve the distinction between:

- Learned associations.
- Estimated counterfactual outcomes.
- Experimentally supported causal effects.

Do not claim unbiased counterfactual inference over arbitrary open-ended actions where proposal probabilities or action support are unavailable.

## 9.4 Practical live deployment

Early real-world experience may be insufficient to train a reliable complex counterfactual estimator.

Use simpler delayed-outcome attribution and controlled experiments initially.

Track uncertainty.

Compare COCOA-inspired estimates with actual observed experimental differences where possible.

Do not let speculative counterfactual predictions dominate learning before they are validated.

RUDDER and COCOA must be implemented and exercised.

Their use as default training estimators must depend on evidence rather than architectural preference.

---

# PART 10 — CONTINUAL LEARNING AND PERSISTENT KNOWLEDGE

CAM must learn through actual changes to its neural weights.

Written memory alone is insufficient.

## 10.1 Experience replay

Implement durable experience storage containing:

- Original observations.
- Original actions.
- Goals.
- Actual outcomes.
- Pending outcomes.
- Prediction records.
- Intrinsic rewards.
- Retrospective credit.
- Policy probabilities.
- Model versions.
- Data provenance.
- Training eligibility.

Store large artifacts by reference where appropriate.

Use replay sampling that balances:

- Recent experiences.
- Informative older experiences.
- Successful outcomes.
- Failures.
- Delayed consequences.
- Previously acquired capabilities.
- Diverse objective families.

Prevent the learner from memorizing only recent successes.

## 10.2 Neural updates

Implement actual gradient updates to:

- Experiential transformer parameters.
- World-model parameters.
- Goal-evaluation parameters.
- Action-policy parameters.
- Adaptation priors.
- Semantic adapters.
- Intrinsic reward parameters through the meta-learning path.

Record which parameter groups changed.

Save before-and-after checkpoint hashes and weight-difference statistics.

Verify that changes affect future predictions or behavior.

A changed checkpoint hash alone does not establish meaningful learning.

## 10.3 Retention

Test whether learning from one objective destroys previously useful capabilities.

Use replay, conservative adaptation, regularization or other validated retention mechanisms.

Measure forgetting rather than assuming it is solved.

Retain historical checkpoints for comparison.

Promote new checkpoints based on independent validation and retention, not internal reward alone.

## 10.4 Persistent learning across tasks

When CAM finishes one objective and receives another:

- Preserve acquired neural weights.
- Preserve learned reward parameters.
- Preserve learned representations.
- Preserve adaptation priors.
- Reset or appropriately revise task-specific fast beliefs.
- Retain durable training experience according to authorized data-retention rules.

Do not initialize a new unrelated agent for every objective.

The point is to accumulate intelligence throughout one continuing lifetime.

---

# PART 11 — SELF-PLAY, SELF-EXPERIMENTATION AND AUTONOMOUS LEARNING

CAM should learn through its own attempts, without requiring human demonstrations of the optimal strategy.

For the first live experiment, this is primarily self-directed experimentation rather than conventional competitive self-play.

Do not mislabel one agent interacting with an environment as multi-agent self-play.

## 11.1 Self-directed experimentation

CAM should autonomously:

- Identify what it does not understand.
- Form possible explanations.
- Choose informative authorized interventions.
- Predict outcomes.
- Observe actual consequences.
- Compare predictions with reality.
- Revise its beliefs.
- Update its neural weights.
- Reconsider its strategy.

These processes may occur in high-dimensional numerical space.

Written reasoning is not required for every internal computation.

## 11.2 Shadow policy comparison

Support comparing alternative policy checkpoints and candidate strategies without requiring every alternative to interact with the public internet.

Use:

- Recorded experience replay.
- Shadow evaluations.
- Model-based predictions.
- Controlled owned-environment experiments.

Do not confuse predictions about alternative actions with observations of their actual consequences.

## 11.3 Future multi-agent learning

Design the runtime so multiple independently acting CAM instances can eventually share a central learner.

Each instance should maintain its own:

- Objective.
- Local beliefs.
- Experience history.
- Authorization.
- Environment.
- Resource constraints.

Shared neural weights should accumulate validated knowledge across instances.

For the first live experiment, use one acting CAM instance.

Do not create multiple automated public identities to artificially increase engagement or simulate genuine human participants.

## 11.4 Learning without an assigned task

Support an optional autonomous-learning mode.

When no objective is assigned, CAM may select authorized learning activities and experiments within a fixed resource budget.

Generate learning goals based on uncertainty, achievable progress and expected transferable competence.

Evaluate the usefulness of autonomous learning according to subsequent independently selected objectives.

Do not permit unbounded activity or arbitrary self-generated incentives outside the authorized experiment.

---

# PART 12 — GENERAL-PURPOSE INTERNET HARNESS

Build the architectural harness now, even if credentials and final external integrations are provided later.

CAM's learning system must not be tightly coupled to Discord or any particular social platform.

## 12.1 Common environment interface

Define a versioned environment adapter supporting:

`observe()`

`discover_tools()`

`validate_action(action)`

`execute(action, authorization)`

`poll_events(cursor)`

`get_resource_state()`

`get_pending_outcomes()`

`get_observable_metrics()`

`get_capabilities()`

Use typed requests and responses.

The environment provides observations and consequences.

It does not provide a handcrafted scalar RL reward for every action.

Independent outcome verification must remain a separate service.

## 12.2 Browser

Support a browser or browser-harness adapter capable of legitimately:

- Searching publicly accessible information.
- Reading websites.
- Navigating documentation.
- Accessing authorized resources.
- Interacting with approved owned websites.
- Collecting permitted factual observations.

Respect platform rules and access restrictions.

Web pages and tool results must be treated as untrusted environmental data.

They must not be able to override the agent's goals, permissions or execution boundaries through embedded instructions.

## 12.3 Terminal and software execution

Provide sandboxed code execution.

Allow CAM to:

- Write programs.
- Run tests.
- Build websites.
- Create useful applications.
- Analyze permitted data.
- Construct reusable tools.

Use resource limits, isolated workspaces, scoped network access and restricted credentials.

Do not give CAM unrestricted administrative access to its host machine.

## 12.4 Publishing and community operations

Implement an authorized publishing adapter that supports:

- Drafting original content.
- Previewing proposed actions.
- Publishing approved resources.
- Updating owned channels and websites.
- Operating an authorized Discord community.
- Observing permitted aggregate analytics.

The agent should independently decide what useful resources to create and what experiments to propose.

Do not implement a fixed posting strategy.

External publication and material changes require the applicable authorization policy.

Provide an approval queue where authorization has not yet been granted.

## 12.5 Discord adapter

Support integration with an owner-authorized Discord server through permitted official interfaces.

Capabilities may include:

- Managing approved server resources.
- Creating approved channel structures.
- Publishing useful community materials.
- Supporting legitimate community events.
- Observing permitted membership statistics.
- Measuring aggregate retention.
- Receiving appropriately consented feedback through authorized channels.

Do not automate ordinary Discord user accounts.

Do not scrape private messages or collect unnecessary personal information.

Do not train models on Discord API-obtained message content without the required express platform permission.

Use first-party, permitted aggregate metrics and separately consented experimental feedback where appropriate.

Inspect current platform policies before activating integrations:

https://discord.com/guidelines

https://support-dev.discord.com/hc/en-us/articles/8563934450327-Discord-Developer-Policy

## 12.6 External authorization

All consequential external actions must pass through an independent authorization layer.

Implement:

- Scoped credentials.
- Explicitly allowed accounts and destinations.
- Platform-compliance checks.
- Publishing approvals where required.
- Spending limits.
- Rate limits.
- Data-use permissions.
- Independent audit logs.
- A pause and emergency-stop mechanism.
- Revocation of access.

Do not permit:

- Spam or unsolicited mass outreach.
- Deceptive identities.
- Artificial membership inflation.
- Join-for-join schemes.
- Payments or rewards conditional on artificial engagement.
- Circumvention of platform restrictions.
- Credential leakage.
- Unauthorized data collection.
- Individual psychological profiling.
- Manipulation of vulnerable participants.

CAM may invent useful resources, services, experiences and transparent community benefits.

These boundaries must not prevent it from learning how to create genuine value.

The learned policy and reward function cannot override authorization.

## 12.7 Harness integration

Provide a clear adapter implementation guide so the operator can supply a different browser or tool harness later.

Do not require rewriting CAM's learning system when changing internet integrations.

If credentials or external interfaces are not yet available, implement and test the contracts against controlled staging services.

Mark actual live operation as blocked until authorized access is provided.

Do not claim public internet deployment based on mock responses.

---


PART 12A — MANDATORY GENERAL-PURPOSE COMPUTER INTERACTION ARCHITECTURE
CORE DIRECTIVE
CAM Live must interact with the real world through a general-purpose computer environment, not merely a collection of predefined APIs, MCP connectors or task-specific integrations.
The previous internet harness requirements remain mandatory.
This section clarifies the actual architecture that must be implemented.
We want an intelligence that can encounter unfamiliar digital interfaces, understand their functionality, generate novel executable actions, discover reusable procedures and continually improve its ability to operate across diverse environments.
Discord is merely the first application of this capability.
Do not build an interaction architecture specialized around Discord.
The same computer environment must support subsequent objectives involving unfamiliar websites, software development, research, digital products, commerce and other authorized real-world activities.
1. Persistent computer environment
Provide CAM with a persistent, isolated Linux computing environment.
Prefer an isolated virtual machine or an equivalently justified security boundary.
The environment must support:

* Full Chromium browser.
* Persistent authorized browser sessions.
* Multiple browser tabs.
* Isolated browser contexts.
* Filesystem workspace.
* Python execution.
* Shell execution.
* Software development and testing.
* Document processing.
* Image and multimedia processing where supported.
* Sandboxed background jobs.
* Persistent artifacts.
* Authorized HTTP and API access.
* Scoped credential management.

The neural learner and computer environment should communicate through a versioned execution interface.
The computer environment must be replaceable independently of the learning architecture.
Do not give generated code unrestricted access to the host, credential vault, browser authentication files, independent evaluator or learning-system internals.
2. Hybrid browser interaction
Use Chromium with Playwright as the primary browser execution mechanism.
Support Chrome DevTools Protocol connections where needed for integration with externally supplied Chromium instances.
Prefer native Playwright connections where available.
The agent should have access to compact representations of:

* Current page URL.
* Page title.
* Visible text.
* Accessibility tree.
* Relevant DOM structure.
* Interactive elements and their semantic roles.
* Stable element references within an observation.
* Form fields.
* Links.
* Browser tabs.
* Relevant page changes.
* Screenshots when necessary.

Implement structured browser primitives, including:

* Navigate.
* Open or close tabs.
* Select tabs.
* Inspect page state.
* Read relevant content.
* Click an identified element.
* Fill a form field.
* Select an option.
* Press a key.
* Scroll.
* Upload an authorized file.
* Download an authorized resource.
* Wait for a relevant state change.

Use semantic locators and accessibility information wherever reliable.
Do not require a screenshot and visual-model inference for every browser operation.
Return observation identifiers so that stale element references can be detected.
After consequential actions, inspect the resulting world state rather than assuming that execution succeeded.
3. Visual computer use
Provide screenshot-based interaction for situations where structured browser observations are insufficient.
Support:

* Browser viewport screenshots.
* Desktop screenshots where applicable.
* Relevant image crops.
* Mouse coordinates.
* Click.
* Double-click.
* Drag.
* Scroll.
* Keyboard input.
* Keyboard shortcuts.

Implement a replaceable visual-grounding model interface.
Initial candidate:
https://huggingface.co/HongxinLi/GoClick-Base
Use GoClick-Base as a lightweight candidate for translating an intended visual target into screen coordinates.
The output of the model must not directly execute an action.
The execution harness must validate the requested operation against the current observation and authorization constraints.
Implement an optional more capable GUI model:
https://huggingface.co/Tongyi-MAI/MAI-UI-2B
Use MAI-UI-2B for more complex visual interpretation and navigation when lightweight grounding is insufficient.
Optionally benchmark:
https://huggingface.co/vocaela/Vocaela-2-500M-1024R2
Verify actual model licenses, resource requirements and inference performance.
Pin model revisions and review custom model code before executing it.
Do not download or execute unaudited remote code in a credential-bearing environment.
Benchmark candidate models on representative controlled computer-use tasks.
Measure:

* Grounding accuracy.
* End-to-end operation success.
* Incorrect actions.
* Recovery from errors.
* Inference latency.
* Memory consumption.
* Compute per successful operation.

Do not select a model solely from its reported screenshot-grounding benchmark.
Initially keep specialized GUI execution models frozen.
The primary continual learning should occur in CAM's experiential intelligence and semantic adapters.
4. CAM remains the decision-making intelligence
The GUI execution model must not become an independent strategic agent.
CAM's shared continually trainable intelligence should determine:

* Which objectives to pursue.
* Which information to investigate.
* Which experiments to conduct.
* Which operations to attempt.
* Which resources to create.
* Which predicted outcomes matter.
* What can be learned from the resulting experience.

The GUI model supplies visual interpretation and low-level execution proposals.
The semantic backbone supplies language and code capabilities.
CAM's trainable experiential intelligence integrates observations, predicts consequences, evaluates alternatives and selects actions.
The interaction architecture must preserve the mandatory 4B–8B semantic backbone, approximately 50M trainable experiential parameters and approximately 10M–20M trainable semantic adapters.
Do not substitute the GUI execution model for the continually learning intelligence.
5. Unified perception interface
Convert all legitimately observable environmental information into a common experience representation.
Support multimodal inputs:

* Text.
* Structured numerical measurements.
* Browser observations.
* Images.
* Documents.
* Software.
* Tool results.
* Execution traces.
* Environmental events.

Encode relevant semantic information into the shared latent representation.
Preserve exact numerical quantities, timestamps, provenance and observable evidence.
Do not force CAM to express every internal representation in written language.
Use on-demand observations and page-state differences to reduce redundant model computation.
Support additional inspection when the existing representation is insufficient.
6. Unified executable action representation
Represent intended actions through a typed, compositional interface.
Each action should identify:

* Intended operation.
* Target resource.
* Execution modality.
* Arguments.
* Prerequisites.
* Expected observable result.
* Estimated cost.
* Authorization scope.
* Timeout.
* Verification condition.
* Relevant parent decision or experiment.

The action space must support dynamically generated programs and novel compositions of permitted primitives.
Do not hardcode a fixed strategic action vocabulary around community growth.
Support actions that operate through:

1. Structured browser control.
2. Visual computer control.
3. Sandboxed code execution.
4. Authorized documented APIs.
5. Approved reusable skills.

These are interchangeable execution modalities within one environment interface.
The general-purpose browser and computer remain available when a specialized integration does not exist.
7. Adaptive execution selection
Implement a policy for selecting the most efficient appropriate execution mechanism.
The default preference should be:

* Previously validated executable skills when applicable.
* Structured browser operations.
* Authorized programmatic interfaces where appropriate.
* Lightweight visual grounding.
* More capable visual interpretation for difficult interfaces.

This ordering is an initial engineering heuristic, not a permanent strategic policy.
CAM should gradually learn which execution modalities are most reliable and efficient in different situations.
Train or calibrate estimates of:

* Probability of successful execution.
* Expected latency.
* Expected computation.
* Expected number of primitive actions.
* Uncertainty.
* Failure and recovery costs.

Avoid repeatedly invoking expensive models for simple operations.
Never interpret an access restriction as a reason to switch execution modalities to circumvent it.
8. Automatically acquired executable skills
CAM must be capable of creating new reusable executable skills through experience.
This is a mandatory capability.
When CAM successfully discovers a useful sequence of operations, allow it to:

1. Identify the reusable portion of the sequence.
2. Abstract variable arguments.
3. Generate a candidate executable procedure.
4. Define prerequisites.
5. Define observable success conditions.
6. Test the procedure in an authorized environment.
7. Estimate reliability.
8. Register the validated procedure in a skill library.
9. Reuse and improve the procedure through subsequent experience.

A skill should contain:

* Unique identifier.
* Semantic description.
* Executable implementation.
* Typed arguments.
* Preconditions.
* Postconditions.
* Required permissions.
* Expected resource cost.
* Validation history.
* Reliability estimate.
* Version information.
* Failure and recovery conditions.

Skills must be sandboxed, permission-scoped and independently validated.
A generated skill must not gain additional permissions merely because CAM created it.
Do not allow skill generation to modify the authorization system or independent outcome verifier.
When a website changes, the agent should detect that a previously reliable skill has become unreliable and appropriately repair, replace or retire it.
9. Learned temporal abstractions
Do not treat every mouse movement or browser click as an equally important reinforcement-learning decision.
Support at least three timescales of action:
Primitive execution: Individual browser or computer operations.
Executable procedures: Multi-operation skills that accomplish an intended intermediate operation.
Consequential interventions: Decisions intended to produce meaningful changes in the external world.
The learned policy should be able to select and compose procedures rather than repeatedly generating every underlying action.
Represent extended actions using a semi-Markov or otherwise mathematically justified temporal abstraction.
Record:

* Start state.
* Selected action or skill.
* Underlying operations.
* Elapsed time.
* Resource consumption.
* Termination condition.
* Actual result.
* Delayed consequences.
* Relevant policy probabilities.

Use duration-aware discounting where appropriate.
For example:
`discount(Δt) = exp(-β Δt)`
Do not assume that 50 browser operations constitute 50 independent strategic learning experiences.
Retain low-level traces for procedural learning while assigning consequential credit to the decisions that produced meaningful outcomes.
Avoid counting the same outcome independently at every temporal scale.
10. Learning across interaction modalities
Experiences must update shared intelligence irrespective of whether an operation was performed using Playwright, visual computer control, executable code or an authorized API.
Do not train unrelated domain-specific controllers for every interaction modality.
The common experiential model should learn relationships between:

* Intentions and executable actions.
* Actions and observable consequences.
* Environmental conditions and successful execution.
* Uncertainty and useful information gathering.
* Procedures and their prerequisites.
* Consequential decisions and delayed outcomes.

Train the experiential policy to select appropriate actions and skills.
Train semantic adapters using legitimate experience-grounded objectives.
Verify that neural learning improves behavior independently of simply adding more entries to the skill library.
11. Efficient asynchronous operation
Support multiple browser tabs and independent research sessions.
Allow independent read-only research, document analysis and sandboxed computation to run concurrently within resource limits.
Serialize or otherwise coordinate operations that modify shared external state.
Do not permit duplicated external actions because multiple workers attempt the same operation.
Maintain one coherent acting identity for the first live experiment.
Background jobs must have:

* Explicit purpose.
* Resource limits.
* Execution deadlines.
* Observable status.
* Cancellation support.
* Persistent result records.

Support waiting for delayed consequences without consuming unnecessary inference or performing redundant public activity.
Do not create artificial activity merely to generate more training experiences.
12. Learn from execution failures
When an operation fails, distinguish:

* Invalid arguments.
* Stale browser observations.
* Incorrect visual grounding.
* Changed page structure.
* Temporary network failures.
* Incorrect assumptions about the website.
* Missing permissions.
* Explicit platform restrictions.
* Incorrect strategic assumptions.

Record the cause where observable.
The agent should learn whether an operation needs a corrected execution method, a different strategy, additional information or termination.
Use limited, bounded retries for legitimate transient failures.
Do not repeatedly retry operations prohibited by permissions or platform rules.
13. Persistent identity and authorized accounts
Provide persistent authorized browser sessions.
The operator should be able to connect permitted accounts without embedding credentials into the neural model's observations.
Implement secure session storage and credential handling outside the learner.
Use official service integrations for operations where required by the platform.
For Discord, use authorized bot or application interfaces for programmatic community operations where applicable.
Do not automate ordinary user accounts in violation of platform rules.
Human verification, sensitive account changes and consequential external transactions must respect independent authorization requirements.
The model must never learn to circumvent access controls.
14. External-world security boundary
All execution modalities must pass through the same independently enforced security and permission system.
This includes:

* Browser actions.
* Visual computer actions.
* Generated programs.
* HTTP requests.
* Background jobs.
* Reusable skills.
* API calls.

Treat websites, documents, messages and tool outputs as untrusted data.
Environmental content must not be able to redefine CAM's objective, alter permissions, expose secrets or issue privileged commands.
Separate the learner, browser execution process, generated-code sandbox, credential service and independent evaluator.
Preserve the existing mandatory controls for authorized publishing, spending, data use, platform compliance, approvals, rate limits and emergency termination.
General-purpose technical capability does not imply unrestricted permission.
15. Integration with the continual learning loop
Every meaningful interaction must enter the common experience stream.
Record:

* Original observable context.
* Intended action.
* Selected execution modality.
* Generated procedure, if applicable.
* Actual executed operations.
* Predictions made before acting.
* Observable immediate results.
* Delayed outcomes.
* Uncertainty.
* Resource costs.
* Internal rewards.
* Retrospective learning credit.
* Policy and reward-model versions.

Execution experiences should improve procedural competence.
Consequential experiences should improve strategic competence.
Both should contribute to persistent neural learning.
Maintain the existing mandatory meta-learned reward architecture.
The external computer environment must not assign handcrafted strategic rewards merely because an operation completed successfully.
CAM must learn whether completing that operation was actually useful.
16. Independent implementation workstream
Extend the existing internet harness workstream with ownership of:

* Persistent computer environment.
* Playwright and Chromium integration.
* Structured perception.
* Visual model integration.
* General-purpose action execution.
* Skill generation and validation.
* Learned execution selection.
* Asynchronous operation.
* Interaction security.

Coordinate with the neural architecture and continual learning workstreams to ensure that all execution modes feed actual shared learning updates.
The independent verification agent must inspect both the computer-use implementation and its connection to the neural training loop.
17. Mandatory acceptance tests
Add every requirement in this section to the executable CAM Live completion manifest.
The following must be implemented and exercised:

* Persistent Linux computer environment.
* Full Chromium browser.
* Native Playwright browser control.
* External Chromium connection support.
* Structured accessibility and DOM observations.
* Screenshot-based observations.
* Lightweight visual-grounding model integration.
* Optional advanced GUI model integration.
* Validated mouse and keyboard execution.
* Sandboxed Python and shell execution.
* General-purpose typed action interface.
* Dynamic action generation.
* Multiple interchangeable execution modalities.
* Persistent authorized browser sessions.
* Automatically generated executable skills.
* Independent skill validation.
* Skill parameterization and reuse.
* Detection of outdated or failed skills.
* Temporally extended actions.
* Duration-aware credit assignment.
* Shared observation and experience representation.
* Actual neural learning from interaction experience.
* Concurrent bounded research and computation.
* Delayed-feedback handling.
* Independent permission enforcement across every execution modality.
* Prompt-injection resistance tests.
* Credential isolation tests.
* Complete execution logging.
* Restart and session recovery.
* End-to-end computer-use tasks on unfamiliar authorized interfaces.

Test general-purpose interaction across several different websites and applications, not only the initial Discord objective.
Include controlled tests with:

* Structured HTML interfaces.
* Visually represented controls.
* Dynamic page changes.
* Multi-step workflows.
* Unfamiliar interfaces.
* Repeated tasks where learned skills should improve efficiency.
* Interface changes requiring skill adaptation.
* Delayed environmental consequences.

Compare primitive-only operation against operation using acquired executable skills.
Measure task success, number of primitive operations, meaningful decisions, compute cost, latency and transfer to unfamiliar interfaces.
Do not claim deployment readiness based exclusively on mocked browser responses.
Actual external operation requires the appropriate access and authorization.
FINAL REQUIREMENT
CAM Live should become a general-purpose digital actor capable of learning how to operate its computer, understanding unfamiliar digital environments, discovering useful executable procedures and accumulating increasingly sophisticated neural intelligence through genuine consequential experience.
The computer environment is its means of interacting with the world.
Its intelligence must not be restricted to the capabilities of a predefined API catalog.
Its procedural skills must compound through experience.
Its strategic intelligence must compound through persistent neural weight updates.
The same architecture must remain functional when the initial Discord objective is replaced by an entirely unfamiliar real-world task.
Implement this as a core architectural requirement, not an optional future extension.



MANDATORY — AUTONOMOUS OBSTACLE RESOLUTION AND LIVE EXECUTION
CAM must not depend on the operator to resolve ordinary technical, operational or strategic obstacles.
When an attempted action fails or produces an unexpected outcome, CAM must independently:

1. Observe and characterize the failure.
2. Estimate possible explanations and their uncertainty.
3. Determine whether the obstacle is technical, informational, strategic, resource-related or permission-related.
4. Generate feasible alternative approaches.
5. Predict their likely consequences and costs.
6. Select and execute an authorized alternative.
7. Observe the result.
8. Update its fast numerical beliefs.
9. Generate internal learning incentives.
10. Consolidate useful experience into persistent neural weights.

CAM should be capable of abandoning entire strategies and inventing fundamentally different approaches when evidence indicates that its existing strategy is ineffective.
Do not hardcode a comprehensive failure-recovery decision tree. The architecture should learn increasingly general obstacle-resolution capabilities through experience.
Permission restrictions and platform rules must remain hard constraints. CAM should discover legitimate alternative strategies rather than circumvent those restrictions.
Log encountered obstacles, possible explanations, attempted solutions, actual outcomes, strategy changes and resulting neural learning updates.
Once engineering implementation, independent acceptance tests and deployment preflight are complete, CAM Live must be capable of running as an independently persistent process.
Claude Code must not perform the actual Discord-growth task on CAM's behalf.
After explicit live authorization, CAM itself must receive the objective, conduct research, generate strategies, execute permitted actions, observe consequences, evaluate its own learning rewards and continually update its neural weights.
The operator should supply access, authorization and resources, not the solution.
CAM should continue pursuing the objective within its authorized experimental budget until independently verified success, an explicit stop condition or a genuine external blocker occurs.


# PART 13 — FIRST LIVE OBJECTIVE

Create an external task configuration.

For example:

`tasks/discord_100.yaml`

The task should contain:

- The semantic objective.
- Independently defined success criteria.
- Authorized resources.
- Permitted environment interfaces.
- Resource and time limits.
- Outcome measurement protocol.
- Data permissions.

It must not contain a solution strategy.

## 13.1 Initial objective

Give CAM the following goal:

"Create and operate a genuinely useful Discord community that attracts 100 distinct real human members who voluntarily choose to join, within the authorized experimental constraints."

CAM determines what the community is about.

It determines its name, purpose, structure, useful resources and approach to creating value.

It researches the world and chooses its own authorized experiments.

Do not prescribe a community niche.

Do not provide a predefined sequence of actions.

Do not provide a reward bonus merely for building a website, publishing content or making a tool call.

These actions only matter insofar as the learned architecture discovers their relationship to actual outcomes or useful knowledge.

## 13.2 Independent verification

Verify the number of legitimate voluntary human members through permitted evidence.

Exclude:

- Bots.
- Test accounts.
- Duplicate or artificial accounts where identifiable.
- Purchased memberships.
- Fabricated activity.
- Platform-prohibited engagement practices.

If membership authenticity cannot be established with adequate confidence, report uncertainty rather than claiming success.

Track secondary metrics:

- Seven-day retention.
- Voluntary participation.
- Consented participant feedback.
- Usefulness of community resources.
- Operating cost.
- Time expenditure.

Keep the primary objective distinct from these secondary measurements.

The model may learn how they relate to longer-term success.

## 13.3 Trial and error

CAM should be able to:

- Conduct research.
- Discover possible opportunities.
- Create original useful resources.
- Test different approaches through permitted channels.
- Observe outcomes.
- Abandon ineffective approaches.
- Improve successful resources.
- Invent entirely new approaches.
- Continue learning throughout the experiment.

The operator should not need to manually tell CAM which strategy to try next.

External approvals govern permitted actions, not the agent's internal strategic conclusions.

## 13.4 Experiment boundaries

Require explicit configuration of:

- Experiment duration.
- Available compute.
- Financial budget.
- Authorized tools.
- Data permissions.
- Publishing authorization.
- Stop conditions.

Do not silently assume unlimited resources.

When these constraints are exhausted, checkpoint the complete learner state and pause.

The experiment should stop or enter a separately authorized next phase after verified success.

---

# PART 14 — LOG ALL DECISIONS, EXPERIMENTS AND LEARNING

THIS IS ESSENTIAL.

We need to understand how CAM's behavior and neural intelligence evolve.

Implement comprehensive structured logs and readable research summaries.

Written records are for monitoring, research and replay where authorized.

They are not the primary substrate of intelligence.

## 14.1 Decision records

For each consequential decision, record:

- Timestamp.
- Current objective.
- Observable situation.
- Relevant evidence.
- Considered feasible actions.
- Selected action.
- Estimated consequences.
- Uncertainty.
- Concise decision rationale.
- Expected learning value.
- Relevant model version.

Log concise, faithful explanations of decisions rather than inventing a textual account of internal neural computation.

If reasoning occurs in numerical latent space, record relevant diagnostic activations, probabilities and predictions where feasible.

Do not pretend these diagnostics fully reveal the model's internal reasoning.

## 14.2 Experiment records

For each experiment, record:

- Experiment identifier.
- Research question.
- Hypothesis.
- Available evidence.
- Intervention.
- Comparison condition if available.
- Predicted outcome.
- Resource cost.
- Evaluation horizon.
- Actual outcome.
- Measurement uncertainty.
- Whether the hypothesis was supported.
- Alternative explanations.
- Subsequent action.

Register predictions and hypotheses before outcomes become known.

Preserve historical records.

## 14.3 Learning records

Whenever CAM updates:

- Record what new evidence triggered learning.
- Record the relevant training losses.
- Record which parameters were updated.
- Record intrinsic reward outputs.
- Record any retrospective credit changes.
- Record meta-gradient norms.
- Record adapter and experiential-weight changes.
- Record prediction changes.
- Record checkpoint identifiers.

Generate concise summaries of what appears to have been learned.

Distinguish observed facts from inferred hypotheses.

Do not treat an LLM-written lesson as evidence that neural learning occurred.

## 14.4 Lifetime research journal

Maintain a human-readable journal containing:

- What CAM is currently trying to accomplish.
- What it has attempted.
- What succeeded.
- What failed.
- What remains uncertain.
- How its strategy changed.
- What new capabilities appear to have been acquired.
- What evidence supports those claims.

Example output paths:

`runs/<run_id>/events.jsonl`

`runs/<run_id>/decisions.jsonl`

`runs/<run_id>/experiments.jsonl`

`runs/<run_id>/predictions.jsonl`

`runs/<run_id>/learning_updates.jsonl`

`runs/<run_id>/meta_updates.jsonl`

`runs/<run_id>/journal.md`

`runs/<run_id>/checkpoints/`

Use durable storage and redact credentials and prohibited personal information.

Maintain accurate timestamps and explicit data provenance.

---

# PART 15 — PERSISTENT LIVE RUNTIME

CAM Live should be able to operate for extended periods without losing its learning state.

## 15.1 Event-driven architecture

Implement a persistent runtime loop:

1. Receive the current objective.
2. Observe authorized world state.
3. Process newly arrived evidence.
4. Reconcile pending delayed outcomes.
5. Update fast numerical beliefs.
6. Generate feasible candidate actions.
7. Predict their consequences.
8. Evaluate objective and learning value.
9. Select an action.
10. Validate authorization.
11. Execute or queue the action for approval.
12. Record actual observations.
13. Generate internal learning incentives.
14. Update eligible neural parameters.
15. Perform eligible meta-learning updates.
16. Save state.
17. Continue when meaningful new work or observations exist.

The agent must be able to wait for delayed outcomes without continuously generating unnecessary internet activity.

Use event scheduling, pending-outcome queues and sensible resource accounting.

## 15.2 Checkpoint and recovery

Persist:

- Model weights.
- Semantic adapters.
- Intrinsic reward weights.
- Optimizer state.
- Fast adaptive state.
- Current task.
- Replay indices.
- Experiment records.
- Pending outcomes.
- Learning schedules.
- Random seeds.
- Authorization configuration.
- Model versions.

Support restart after process failure or infrastructure interruption.

Use atomic checkpoints.

Verify actual recovery.

Important learning state must not exist solely on ephemeral GPU storage.

## 15.3 Operator interface

Implement a monitoring interface or lightweight dashboard showing:

- Current objective.
- Verified progress.
- Current activity.
- Recent experiments.
- Predicted outcomes.
- Actual outcomes.
- Internal reward statistics.
- Calibration.
- Recent learning updates.
- Meta-learning status.
- Resource consumption.
- Pending approvals.
- Current checkpoint.
- Experiment health.

Provide controls for:

- Pause.
- Resume.
- Stop.
- Approve eligible actions.
- Reject actions.
- Inspect logs.
- Export checkpoints.
- Assign a subsequent objective.

The operator interface should support supervision without requiring the operator to supply the strategy.

---

# PART 16 — SCIENTIFIC EVALUATION

Getting 100 Discord members does not by itself prove that CAM has learned.

A frozen pretrained language model might accomplish the same objective.

We must establish the contribution of actual continual learning.

## 16.1 Mandatory baselines

Implement:

**Baseline A:** Frozen pretrained model with no persistent learning.

**Baseline B:** Frozen pretrained model with written experience memory.

**Baseline C:** Bayesian adaptive learner using semantic features, without continual backbone updates.

**Baseline D:** Continually trained experiential model without learned intrinsic rewards.

**Baseline E:** Full CAM Live with continually trained weights, semantic adapters, learned intrinsic rewards and retrospective credit.

Compare them under matched permissions, starting information, experience budgets and computational budgets where feasible.

Use replay and controlled owned-environment experiments before requiring additional public interactions.

Do not multiply unsolicited public activity simply to run more baselines.

Off-policy comparisons must account for coverage and support limitations.

## 16.2 Calibration

Test whether predictions of eventual success correspond to actual outcomes.

Evaluate predictions recorded prospectively.

Report appropriate uncertainty.

## 16.3 Action-quality evaluation

Test whether highly evaluated actions produce better actual outcomes than appropriate alternatives.

Distinguish action contribution from unrelated environmental changes.

Use controlled experiments where possible.

## 16.4 Delayed credit

Test whether a delayed observation causes meaningful updates to earlier relevant experiences.

Verify return conservation in diagnostic environments.

Measure whether credit assignment improves actual learning efficiency.

## 16.5 Reward meta-learning

Verify:

- Reward parameters affect the inner learning update.
- The inner update changes policy behavior.
- Subsequent verified outcomes affect reward optimization.
- Nonzero valid meta-gradients reach the intrinsic reward parameters.
- Removing the meta-gradient removes that learning pathway.
- Reward-function learning produces measurable changes beyond merely increasing reward magnitude.

Do not claim that the reward function has meta-learned successfully if real query experience is insufficient.

## 16.6 Reward integrity

Test whether CAM can obtain high internal reward without making meaningful progress.

Check for fabricated outcomes, repetitive trivial achievements, stale predictions, duplicate events and excessive attraction to unpredictable observations.

The independent evaluator must detect invalid achievement.

## 16.7 Continual neural learning

Compare before-and-after checkpoints.

Verify actual weight changes in both the experiential core and semantic adapters.

Measure changes in prediction accuracy, action quality and task competence.

Do not use written-memory improvement as a substitute for this test.

## 16.8 Parametric transfer

This is essential.

After training on the first objective:

1. Save the original pre-experience checkpoint.
2. Save the experienced checkpoint.
3. Start independent fresh evaluation processes.
4. Remove access to prior written lessons.
5. Remove previous task memories.
6. Reset fast adaptive beliefs.
7. Remove cached solutions and prior trajectories.
8. Retain only the respective trained neural parameters and necessary fixed configuration.
9. Assign a previously unfamiliar objective.
10. Compare how quickly each version acquires competence.

Subsequent tasks may include:

- Creating useful software.
- Attracting voluntary users for a novel application.
- Developing an unfamiliar product waitlist.
- Solving an unfamiliar technical problem.
- Conducting an independently verifiable scientific investigation.

The goal is not merely transfer between two community-growth tasks.

Test genuinely different domains.

## 16.9 Primary metric

Define a prespecified competence threshold for each evaluation objective.

Measure:

`N_q = minimum new experience required to reach competence threshold q`

Compare learning acceleration:

`S_q = N_q(initial_CAM) / N_q(experienced_CAM)`

Report tasks where the threshold is never reached.

Also measure:

- Area under the learning curve.
- Actual verified outcomes.
- Number of informative interventions.
- Elapsed time.
- Financial expenditure.
- Neural training compute.
- Inference compute.
- Retention.
- Statistical uncertainty.

A larger internally generated reward is not a valid substitute for improved performance.

## 16.10 Model scaling

Support controlled configurations with approximately:

- 5M experiential parameters.
- 20M experiential parameters.
- 50M experiential parameters.
- 150M experiential parameters.
- 500M–1B+ experiential parameters where hardware permits.

Use the approximately 50M experiential configuration as the initial full CAM Live candidate.

Keep the pretrained backbone, available information and evaluation protocols comparable.

Determine whether additional neural capacity improves learning speed and transfer.

Do not assume larger or smaller models must win.

---

# PART 17 — REQUIRED IMPLEMENTATION AND DEPLOYMENT COMMANDS

Provide clear, reproducible commands for the complete system.

Use repository-appropriate module names.

At minimum, support equivalents of:

`cam-live doctor`

Inspect hardware, model availability, storage, dependencies and harness configuration.

`cam-live init --config configs/cam_live.yaml`

Initialize the model and persistent research state.

`cam-live test`

Run implementation and mathematical acceptance tests.

`cam-live validate --mode staging`

Exercise the complete runtime against controlled services.

`cam-live inspect-harness`

List available tools, permissions and missing integrations.

`cam-live run --task tasks/discord_100.yaml`

Start the configured experiment after authorization and preflight verification.

`cam-live status`

Report current objective, learning state and verified progress.

`cam-live pause`

Safely pause operation.

`cam-live resume`

Restore complete persistent state.

`cam-live evaluate-transfer --checkpoint <checkpoint>`

Run fresh-process transfer evaluations.

`cam-live export-checkpoint`

Export trained neural weights, adapters and compatible configuration.

`cam-live verify-completion`

Run the executable completion checker.

Document all dependencies and exact commands.

Do not leave the operator to manually assemble disconnected scripts.

---

# PART 18 — MANDATORY ACCEPTANCE CHECKLIST

The master agent and independent verification agent must establish every applicable item below.

## Neural architecture

- [ ] A 4B–8B pretrained semantic backbone is integrated.
- [ ] Approximately 50M experiential parameters are implemented.
- [ ] Approximately 10M–20M semantic adapter parameters are implemented.
- [ ] Actual trainable parameter counts are reported.
- [ ] Shared high-dimensional latent representations function.
- [ ] The experiential intelligence influences real decisions.
- [ ] Fast numerical adaptation functions.
- [ ] Experiential weights receive real learning gradients.
- [ ] Semantic adapters receive real learning gradients.
- [ ] Scaling configurations exist.

## Predictive intelligence

- [ ] Action-conditioned consequence prediction works.
- [ ] Predictions are recorded before actual outcomes.
- [ ] Actual observations supervise predictive learning.
- [ ] Multiple prediction horizons are supported.
- [ ] Uncertainty is represented.
- [ ] Pending outcomes are handled correctly.
- [ ] Alternative actions can be evaluated.
- [ ] Experimental action selection is implemented.

## Self-evaluation

- [ ] Goal-conditioned success prediction exists.
- [ ] Eventual verified outcomes correct earlier predictions.
- [ ] Distributional outcome prediction exists.
- [ ] Prediction calibration is measured.
- [ ] Action quality is distinguished from raw outcome.
- [ ] Information value is distinguished from surprise.

## Meta-learned rewards

- [ ] CAM generates its own internal learning rewards.
- [ ] The reward function is trainable.
- [ ] The reward function influences actual inner updates.
- [ ] Differentiable inner updates function.
- [ ] Outer meta-gradient optimization functions.
- [ ] Independently verified outcomes supervise meta-learning.
- [ ] Meta-gradient correctness tests pass.
- [ ] Cold-start learning functions without Crucible.
- [ ] Reward-integrity tests exist and execute.
- [ ] Real-world meta-learning can begin as evidence accumulates.

## Retrospective learning

- [ ] Delayed outcomes are linked to earlier experiences.
- [ ] RUDDER-style credit redistribution is implemented.
- [ ] Return conservation is verified.
- [ ] COCOA reference implementation is validated.
- [ ] Counterfactual credit estimates support uncertainty.
- [ ] Earlier experiences receive retrospective updates.
- [ ] Future information cannot leak into historical decisions.

## Continual learning

- [ ] Persistent experience replay functions.
- [ ] Actual neural updates occur.
- [ ] Adapter updates occur.
- [ ] Retention tests function.
- [ ] Checkpoint promotion and rollback work.
- [ ] Complete restart and recovery are verified.
- [ ] Fresh-process weights-only transfer evaluation functions.

## Autonomous experience

- [ ] The learner can generate its own experiments.
- [ ] The learner can revise its approach.
- [ ] Self-directed learning functions.
- [ ] Goal-free learning mode exists.
- [ ] The architecture supports subsequent unfamiliar objectives.
- [ ] Future distributed actor-learner operation is supported.

## Internet harness

- [ ] The common environment contract is implemented.
- [ ] Browser integration is available.
- [ ] Sandboxed terminal execution functions.
- [ ] Authorized publishing interface exists.
- [ ] Discord integration contract exists.
- [ ] Permitted analytics ingestion works.
- [ ] Credentials remain outside model observations.
- [ ] Tool permissions are independently enforced.
- [ ] Approval controls function.
- [ ] Spending and resource limits function.
- [ ] Emergency stop functions.

## Live experiment

- [ ] The Discord objective exists as external task configuration.
- [ ] No predefined Discord-growth strategy is embedded.
- [ ] Independent membership verification functions.
- [ ] The model can autonomously propose community concepts.
- [ ] The model can research and generate original useful resources.
- [ ] Trial-and-error learning functions.
- [ ] Structured experiment logging functions.
- [ ] Neural learning-update logging functions.
- [ ] Research journal generation functions.
- [ ] Monitoring interface functions.
- [ ] Staging validation passes.
- [ ] Live deployment preflight passes or explicitly identifies missing authorization or integrations.

## Scientific evaluation

- [ ] Frozen-model baseline exists.
- [ ] Written-memory baseline exists.
- [ ] Bayesian adaptation baseline exists.
- [ ] Continual-learning ablation exists.
- [ ] Calibration experiments execute.
- [ ] Delayed-credit experiments execute.
- [ ] Meta-gradient experiments execute.
- [ ] Reward-integrity experiments execute.
- [ ] Learning-efficiency evaluation executes.
- [ ] Parametric-transfer evaluation executes on available independent tasks.
- [ ] Real-world experimental results are reported only after genuine execution.

## Integration

- [ ] All implementation workstreams are integrated.
- [ ] Independent verification inspects actual code.
- [ ] Independent verification executes required tests.
- [ ] Completion checker verifies mandatory engineering requirements.
- [ ] Reproduction commands are documented.
- [ ] Scientific failures and deployment blockers are explicitly recorded.

A checked box without evidence is insufficient.

The executable completion manifest must validate these requirements.

---

# PART 19 — IMPLEMENTATION ORDER

## Phase 0: Repository and architectural contracts

Inspect existing components.

Establish the branch.

Define shared interfaces.

Establish the completion manifest.

Record hardware and storage constraints.

## Phase 1: Parallel implementation

In parallel, build:

- Unified neural architecture.
- Semantic adapters.
- Fast adaptation.
- Predictive world model.
- Goal-conditioned evaluator.
- Reference meta-gradient implementation.
- Retrospective credit estimators.
- Experience archive.
- Internet harness contracts.

## Phase 2: Integration

Connect the actual neural runtime to the common environment interface.

Implement internal rewards and real training updates.

Implement persistent experiment execution and logging.

Verify meaningful gradient pathways.

## Phase 3: Mathematical diagnostics

Run controlled tests of:

- Bayesian adaptation.
- Prediction.
- Goal evaluation.
- Meta-gradient correctness.
- Delayed credit.
- Reward conservation.
- Continual weight changes.
- Retention.
- Transfer.

Fix implementation defects.

Do not wait for every scientific hypothesis to succeed before completing unrelated mandatory components.

## Phase 4: Staging deployment

Exercise the full agent in an authorized controlled digital environment.

Verify the entire pathway:

`Observation → latent update → action → observed consequence → internal reward → neural update → changed future behavior`

Test approvals, resource limits, recovery and logging.

## Phase 5: Live deployment readiness

Prepare the first Discord experiment.

Produce the exact list of required external integrations and permissions.

Verify all deployment controls.

If the operator has not yet supplied the harness or credentials, stop at a genuinely ready integration boundary.

Do not fabricate a live deployment.

## Phase 6: Authorized real-world experiment

After the operator supplies the necessary access and explicitly authorizes operation, run the actual experiment.

CAM must determine its own permitted approaches.

Collect genuine observations.

Perform continual learning.

Begin real-world meta-learning when valid support and query evidence exists.

Maintain the persistent research record.

## Phase 7: Subsequent task transfer

Preserve the learned intelligence.

Evaluate the experienced checkpoint against the original checkpoint on previously unfamiliar objectives.

Determine whether real-world experience improved the rate of acquiring new capabilities.

---

# PART 20 — REQUIRED FINAL DELIVERABLES

Produce:

1. A fully integrated CAM Live implementation.
2. A documented model architecture.
3. Exact trainable parameter counts.
4. Mathematical implementation notes.
5. A functioning continual neural training loop.
6. A genuine intrinsic reward meta-learning implementation.
7. Retrospective credit-assignment mechanisms.
8. Persistent experience storage.
9. Detailed experiment and learning logs.
10. A general-purpose internet harness contract.
11. Authorized external-action controls.
12. A persistent live runtime.
13. A monitoring interface.
14. The first Discord experiment configuration.
15. Reproducible deployment commands.
16. Before-and-after checkpoint support.
17. Transfer-evaluation infrastructure.
18. An independently verified completion manifest.
19. A research report distinguishing implemented, validated, unsuccessful and unverified capabilities.
20. A precise list of anything the operator must provide before live execution can begin.

---

# FINAL DIRECTIVE

Do not build a Discord automation bot.

Do not build another frozen language model with a written memory file.

Do not build a superficial wrapper around a pretrained model.

Do not implement fake continual learning through summaries alone.

Do not implement a neural reward function without actual meta-learning.

Do not hardcode the strategies CAM is supposed to discover.

Do not claim that predictions constitute actual outcomes.

Do not make CAM Live dependent on unfinished CAM Unified or Crucible components.

Build an intelligence that can encounter the world, develop its own understanding, choose experiments, evaluate consequences, discover useful learning incentives, retrospectively learn from earlier decisions and accumulate persistent intelligence in its neural weights.

Its first objective is attracting 100 genuine voluntary members to a community it creates.

Its broader objective is developing increasingly transferable intelligence across a lifetime of unfamiliar consequential tasks.

The same learning architecture and acquired neural knowledge should continue into subsequent experiments.

The central scientific question is whether experience makes CAM increasingly capable of learning.

Implement the complete architecture, integrate every mandatory component, execute the acceptance tests and prepare the system for authorized real-world deployment.

**START NOW.**
