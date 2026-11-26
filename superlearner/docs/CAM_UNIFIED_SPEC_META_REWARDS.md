# CAM Unified — mandatory meta-learned rewards, counterfactual credit assignment and autonomous lifetime learning (user, 2026-11-26)

CAM UNIFIED — MANDATORY META-LEARNED REWARDS, COUNTERFACTUAL CREDIT ASSIGNMENT AND AUTONOMOUS LIFETIME LEARNING

## CORE DIRECTIVE

This specification is a mandatory architectural extension of the existing CAM Unified Experience Superlearner.

It supersedes previous assumptions that CAM must receive intermediate reinforcement-learning rewards from Crucible.

It does not replace or cancel the existing requirements for semantic understanding, high-dimensional latent reasoning, predictive world modeling, general-purpose action generation, continual neural learning, distributed experience collection, multistep imagination, or full OpenWorld integration.

All existing mandatory capabilities must remain implemented.

We are introducing a fundamental new capability:

CAM must learn how to evaluate its own experiences, discover its own internal reward function, identify which past decisions contributed to future outcomes, and autonomously determine what is worth learning throughout its lifetime.

We do not want an intelligence that requires an environment developer to specify which individual actions deserve reinforcement.

We want an intelligence that can inhabit an unfamiliar environment, observe what happens, interpret the significance of its experiences, assign learning credit to its decisions, and improve its neural weights.

The same learning architecture should function in Crucible 5.0 and authorized real-world environments.

The world provides evidence and consequences.

CAM generates its internal learning incentives.

Independent outcome verification establishes whether CAM's understanding and learning process are actually effective.

The ultimate research objective remains:

Maximize the acquisition rate of persistent, transferable, consequential intelligence from experience.

Do not implement this as an LLM grading its own actions from 1 to 10.

Do not implement it as a handwritten weighted sum of novelty, revenue, curiosity and task progress and call that a learned reward function.

Do not implement a neural reward head that receives no meaningful training signal.

Do not implement retrospective credit assignment as ordinary temporal discounting with a new name.

Do not implement autonomous learning as randomly selecting tasks from a fixed menu.

Do not implement continual learning exclusively through persistent written summaries.

These capabilities must be real, trainable, integrated and experimentally exercised.

# PART 0 — EXECUTION AND IMPLEMENTATION GOVERNANCE

## 0.1 Recover the current project

Repository:

https://github.com/fnstggl/agentrs

Inspect the actual latest local and remote state of:

* CAM Unified.
* CAM 7.0.
* CAM 6.1.
* Crucible 4.0.
* Crucible 5.0.
* All relevant ongoing experimental branches.

Identify the currently active CAM Unified branch and worktree.

Do not assume a previous version number or reported commit is still current.

Read the existing architecture specification, implementation manifest, research ledger, test results and experimental outputs.

Preserve completed and unfinished work.

Integrate this specification into the existing implementation manifest as additional mandatory requirements.

If the previous CAM Unified build is still underway, coordinate with its existing workstreams rather than restarting the architecture.

If Crucible 5.0 is being developed in a separate branch or process, establish a versioned integration contract and coordinate compatible changes.

Do not modify sealed historical evaluations.

Verify available CPU, GPU, memory, disk and persistent storage.

Preserve important checkpoints and experimental results outside ephemeral instance storage wherever an authorized destination exists.

## 0.2 Establish a master orchestration agent

Use one master architecture and integration agent.

Use the strongest available appropriate Claude Opus model for the master and core machine-learning subagents.

Inspect actual support for agent teams, subagents, parallel execution and isolated worktrees.

If parallel agent teams are unavailable, implement the same ownership and verification structure sequentially.

The master agent is responsible for:

* Preserving the overall research objective.
* Establishing shared mathematical and software contracts.
* Defining tensor interfaces and data schemas.
* Assigning nonoverlapping implementation ownership.
* Resolving dependencies.
* Integrating completed work.
* Running end-to-end experiments.
* Maintaining the completion manifest.
* Preventing incomplete work from being declared complete.

Subagents are engineering collaborators, not separate intelligence modules inside CAM.

The resulting CAM architecture should remain one unified neural intelligence.

## 0.3 Mandatory workstreams

Workstream A — Semantic goal grounding and experience representation

Own:

* Semantic and numerical goal representations.
* Exact observable evidence.
* Event encoding.
* Outcome representations.
* Integration with the shared latent backbone.
* Missingness, timestamps and measurement provenance.

Workstream B — Learned goal-conditioned self-evaluation

Own:

* Goal-conditioned success prediction.
* Distributional outcome prediction.
* Time-to-success prediction.
* Resource-consumption prediction.
* Calibration.
* Learning from actual eventual outcomes.
* Learned progress estimation.

Workstream C — Meta-learned intrinsic reward function

Own:

* Trainable intrinsic reward function.
* Differentiable inner learning updates.
* Lifetime objective.
* Outer meta-gradient optimization.
* Stable reward learning.
* Meta-learned reward transfer.
* Tests demonstrating that learned rewards change downstream learning.

Workstream D — Retrospective and counterfactual credit assignment

Own:

* RUDDER-style return decomposition.
* COCOA.
* Reward Hindsight Credit Assignment, or R-HCA.
* Counterfactual outcome prediction.
* Long-horizon action attribution.
* Delayed-feedback ingestion.
* Credit-assignment baselines and diagnostic tests.

Workstream E — Autonomous curriculum and learning without assigned goals

Own:

* Self-generated learning objectives.
* Learned experiment selection.
* Decision-relevant information seeking.
* Goal-free exploration.
* Estimation of transferable learning progress.
* Meta-training of curriculum selection.

Workstream F — Observation-only Crucible 5.0

Own:

* Observation-only training mode.
* Removal of native rewards and privileged progress signals.
* Independent outcome verification.
* Rich public event streams.
* Goal-free world operation.
* Diverse world families.
* Counterfactual evaluation infrastructure.
* Compatibility with historical Crucible baselines.

Workstream G — Continual and distributed learning integration

Own:

* Durable experience storage.
* Delayed outcome reconciliation.
* Shared neural-weight updates.
* Actor-learner synchronization.
* Reward-model versioning.
* Retrospective replay.
* Retention evaluation.
* Weights-only transfer.
* Resource accounting.

Workstream H — Independent scientific and implementation verification

Own:

* Acceptance-test definitions.
* Independent implementation audits.
* Gradient-path verification.
* Data-leakage checks.
* Reward-integrity tests.
* Experimental controls.
* Statistical analysis.
* Clean-process reproduction.
* Verification of every mandatory requirement.

Workstream H must independently inspect actual implementation and execute tests.

It must not accept a capability as implemented solely because another subagent reports completion.

## 0.4 Mandatory completion manifest

Before implementation, create or extend:

`docs/CAM_UNIFIED_COMPLETION_MANIFEST.json`

Reuse the existing project manifest if one already exists.

Every requirement in this specification must have a unique identifier and a recorded status.

Each entry must include:

* Requirement identifier.
* Exact capability.
* Owning workstream.
* Dependencies.
* Source files.
* Integration entry point.
* Acceptance-test identifiers.
* Reproduction command.
* Actual test result.
* Experimental result, where applicable.
* Raw evidence location.
* Commit identifier.
* Known limitations.
* Independent verification status.

Create an executable completion checker, for example:

`python -m superlearner.verify_completion`

Adapt the exact path to the existing repository conventions.

The checker must fail if:

* A mandatory requirement is missing.
* A referenced implementation file does not exist.
* A required test has not been executed.
* A component has not been integrated into the real learning path.
* A mandatory gradient pathway is absent.
* A required experiment has not been run.
* A component is represented only by a stub, mock or placeholder.
* Independent verification is missing.
* Required evidence is absent or inconsistent with the claimed status.

The checker must distinguish two separate things:

Engineering completion: The required capability exists, is integrated and passes its implementation tests.

Scientific success: The capability produces the intended measurable improvement.

A scientifically unsuccessful experiment must not be falsely reported as successful.

But failure to beat a baseline must not be used as an excuse to omit other mandatory components.

Previous CAM builds repeatedly concentrated on numerical environments and stopped before implementing the more important general-purpose capabilities.

That must not happen here.

Maintain a persistent research ledger and update it after every significant milestone.

If a session is interrupted, resume from that ledger, actual code and verified experiment outputs.

# PART 1 — MANDATORY RESEARCH FOUNDATION

Before implementing the mathematical learning mechanisms, study the following papers and relevant reference implementations.

## 1.1 Learned intrinsic rewards

What Can Learned Intrinsic Rewards Capture?

Zheng, Oh, Hessel, Xu, Kroiss, van Hasselt, Silver and Singh, ICML 2020.

https://proceedings.mlr.press/v119/zheng20b.html

https://proceedings.mlr.press/v119/zheng20b/zheng20b.pdf

https://proceedings.mlr.press/v119/zheng20b/zheng20b-supp.pdf

The attached PDF may also be available locally.

Read the actual algorithm and derivations, particularly:

* Lifetime return objective.
* Intrinsic reward network.
* Inner policy update.
* Outer reward-function meta-gradient.
* Lifetime value function.
* Truncated differentiation through learning.
* Cross-episode exploration.
* Reward-function transfer between agents and environments.

The central idea is that knowledge can be stored not only in a policy or world model, but also in the learned reward function.

The reward function should encode knowledge about what experiences and behaviors are worth reinforcing.

Its quality is determined by the intelligence acquired through training with it.

## 1.2 Long-horizon reward redistribution

RUDDER: Return Decomposition for Delayed Rewards

https://arxiv.org/abs/1806.07857

https://github.com/ml-jku/rudder

Study:

* Return-equivalent reward redistribution.
* Contribution analysis.
* Sequence-based return prediction.
* Redistribution of delayed returns.
* Residual correction.
* Expected remaining reward.
* Limitations under stochastic, partially observable and discounted settings.

## 1.3 Counterfactual credit assignment

Would I Have Gotten That Reward? Long-Term Credit Assignment by Counterfactual Contribution Analysis

Meulemans et al., NeurIPS 2023.

https://arxiv.org/abs/2306.16803

https://arxiv.org/html/2306.16803

https://github.com/seijin-kobayashi/cocoa

Read the main paper and appendices, especially:

* Counterfactual contribution coefficients.
* Hindsight distributions.
* Rewarding outcome representations.
* Policy-gradient estimators.
* Learning contribution coefficients.
* Continuous action spaces.
* Partially observable environments.
* Temporal discounting.
* Bootstrapping.
* Limitations of learned hindsight estimators.

## 1.4 Improved reward hindsight credit assignment

Improving Reward-Based Hindsight Credit Assignment

Ramesh, He, Schmidhuber and White, 2026.

https://rlj.cs.umass.edu/2026/papers/Paper133.html

Study the weaknesses identified in COCOA's treatment of noisy immediate rewards and the proposed R-HCA estimator.

Implement and compare R-HCA where applicable.

If any necessary implementation information is inaccessible, document precisely what is missing. Do not invent an algorithm and label it R-HCA.

## 1.5 Additional relevant foundations

Study:

* Universal Value Function Approximators.
* Hindsight Experience Replay.
* Hindsight Credit Assignment.
* Potential-based reward shaping.
* Decision-relevant Bayesian experimental design.
* Learning-progress-based intrinsic motivation.
* Meta-gradient reinforcement learning.
* Goal-conditioned distributional RL.
* Continual learning and replay.
* Recurrent meta-RL.

Do not combine every technique indiscriminately.

The purpose of research is to determine how these ideas can form a coherent and efficient learning architecture.

Create a technical research note explaining which mathematical components are adopted, which are adapted, and which assumptions are not valid in CAM's open-ended partially observable environments.

# PART 2 — CRUCIBLE 5.0 MUST DEFAULT TO OBSERVATION-ONLY OPERATION

This is mandatory.

CAM must no longer depend on Crucible providing a handcrafted intermediate reward.

The default mode for new CAM training runs must be:

`reward_mode = "observation_only"`

Retain historical reward modes solely for controlled baselines and reproducibility.

## 2.1 What the environment returns

The environment should return:

* Observable world events.
* Tool execution results.
* Public and appropriately accessible private information.
* Actual transaction records.
* Actual application performance.
* Customer behavior.
* Purchases and cancellations.
* Messages and responses.
* Technical measurements.
* Observable competitor behavior.
* Resource consumption.
* Time elapsed.
* Operational changes.
* Failures and errors.
* Other legitimately observable consequences.

The environment must not return a handcrafted scalar describing whether an action was good or bad.

If compatibility with Gymnasium requires a reward field, return zero in observation-only mode.

Do not treat this zero as a failure or as the learner's actual training reward.

The CAM learner must generate its training reward internally.

## 2.2 Remove hidden reward leakage

Simply setting the reward field to zero is insufficient.

Inspect the complete observation pipeline.

Remove access to:

* Native reward scalars.
* Privileged normalized goal-progress scores.
* Hidden potential functions.
* Evaluator-only success statistics.
* Hidden mechanism parameters.
* Future outcomes.
* Private task solutions.
* Private evaluator state.

Preserve ordinary observable evidence.

For example, actual revenue, operating costs and bank balances are legitimate observations.

A hidden normalized score calculated specifically for the benchmark is not.

Audit both the full OpenWorld interfaces and numerical mechanism interfaces.

Previous numerical environments exposed reward values directly in observations.

That must not happen in observation-only training.

## 2.3 Independent outcome verification

Do not remove independent goal verification.

A researcher may assign:

"Build a profitable business generating $100,000 in genuine annual recurring revenue within 180 simulated days."

CAM chooses how to pursue the objective.

Crucible does not prescribe strategies or supply action-by-action reward.

An independent evaluator verifies whether the business genuinely achieved the target.

For this particular example, distinguish:

* Annual recurring revenue.
* Monthly recurring revenue.
* One-time purchases.
* Refunds.
* Churn.
* Artificial circular payments.
* Revenue from independent customers.
* Profitability.
* Sustainability.

Do not confuse rolling 30-day gross revenue with ARR.

For goals lacking an immediately measurable criterion, support an independently specified evaluation protocol.

If success is uncertain or unverified, record that uncertainty.

Do not convert missing evidence into success.

## 2.4 Separate data channels

Maintain separate channels for:

Actor observations: Information CAM legitimately observes while interacting.

Training outcome labels: Independently verified results available after they have actually occurred.

Private evaluation state: Hidden mechanisms, controlled counterfactual experiments, sealed tasks and evaluator-only data.

The actor must not have access to private evaluation state.

The trainer may use appropriately released training-world outcome labels.

Final held-out evaluation outcomes must not be fed back into training.

## 2.5 Goal-free operation

Support persistent environment operation without an assigned external goal.

CAM should still be able to:

* Observe the world.
* Generate learning objectives.
* Execute permitted experiments.
* Receive consequences.
* Learn from experiences.
* Continue operating across extended simulated lifetimes.

The environment must not secretly substitute a default economic objective.

Goal-free learning is a distinct mandatory mode.

# PART 3 — UNIFIED NUMERICAL EXPERIENCE REPRESENTATION

The core intelligence must operate through learned high-dimensional numerical representations.

Semantic information should ground those representations.

Do not force CAM's internal learning process to occur through written natural-language reflections.

Retain exact structured evidence alongside learned representations.

## 3.1 Shared state

Define:

`b_t = F_θ(H_t, m_t)`

Where:

* `H_t` is the legitimately observable experience history.
* `m_t` is CAM's fast adaptive state.
* `θ` contains persistent trainable neural parameters.
* `b_t` is the integrated latent understanding of the current world.

Represent current goals with a shared goal encoder:

`e_g = G_θ(g)`

Represent executable actions with an action encoder:

`e_a = A_θ(a)`

The representations must support:

* Numerical quantities.
* Text.
* Code.
* Documents.
* Websites.
* Entities.
* Relationships.
* Executable programs.
* Temporal events.
* Multiple interacting participants.

## 3.2 Preserve precision and provenance

Do not rely solely on embeddings for exact quantities.

An event containing revenue, cancellation counts or technical measurements should preserve the original values and units.

Maintain:

* Stable entity identifiers.
* Event identifiers.
* Action identifiers.
* Timestamps.
* Source provenance.
* Measurement reliability.
* Missingness masks.
* Outcome maturity.
* Uncertainty.
* Policy version.
* Reward-model version.

The neural model should learn to interpret the relationships among these observations.

Do not impose a fixed human-authored ontology of all useful concepts.

## 3.3 Shared neural computation

Goal evaluation, intrinsic reward learning, counterfactual credit assignment, prediction and action selection should share useful representations.

Use lightweight trainable output functions where appropriate.

They must not become independent LLM agents exchanging written assessments.

Different output functions may require different optimization schedules and deliberate gradient boundaries.

Unification means coordinated shared representations and learning, not indiscriminate gradient propagation through every component.

Document and test every important gradient pathway.

# PART 4 — LEARNED GOAL-CONDITIONED SELF-EVALUATION

This is mandatory and central to the new architecture.

CAM must learn to estimate the significance of its experiences relative to its objectives.

## 4.1 Predict actual future success

Define a learned success predictor:

`p_t = V_success(b_t, e_g, horizon, policy_context)`

For binary goals, the predictor estimates the probability of achieving the goal under an explicitly defined continuation policy and resource budget.

When the actual outcome becomes known:

`Y_g = 1` if independently verified success occurred.

`Y_g = 0` if the objective was definitively unsuccessful at its evaluation deadline.

If the outcome is still pending, do not prematurely label it a failure.

Train with:

`L_success = -Y_g log(p_t) - (1 - Y_g) log(1 - p_t)`

Use numerically stable binary cross-entropy with logits.

Train on different prefixes of completed trajectories, not only the final state.

The model should learn to predict success from incomplete histories at many stages of an objective.

The final outcome provides supervision for earlier predictions.

## 4.2 Multiple future-outcome predictions

Predict more than binary success.

The internal evaluator should learn distributions over:

* Probability of eventual success.
* Time required.
* Magnitude of achievement.
* Resource consumption.
* Potential losses.
* Sustainability.
* Outcome uncertainty.

Use task-appropriate distributions.

For example:

* Bernoulli for binary outcomes.
* Negative binomial or another justified count distribution.
* Quantile or probabilistic regression for continuous outcomes.
* Survival or time-to-event models for delayed and censored outcomes.

Support novel goals through shared goal-conditioned representations.

Do not build a separate handwritten evaluator network for every business or scientific objective.

## 4.3 Accurate calibration

An evaluator predicting 80% success should actually succeed approximately 80% of the time across comparable cases.

Measure:

* Negative log-likelihood.
* Brier score.
* Reliability diagrams.
* Expected calibration error.
* Calibration slope and intercept.
* Calibration by domain, horizon and difficulty.
* Out-of-distribution uncertainty.

Evaluate on predictions recorded before their outcomes were available.

Use independent training and validation trajectories.

Compare against simple prevalence, statistical and frozen-model baselines.

Do not claim calibration merely because training loss decreases.

## 4.4 Learn intermediate significance

The model should learn that an event can be valuable before the ultimate objective is achieved.

Examples:

* Discovering why customers cancel.
* Finding a previously unknown technical constraint.
* Identifying a reliable collaborator.
* Discovering that an apparently attractive strategy is ineffective.
* Removing a bottleneck.
* Acquiring information that changes the optimal future decision.

A candidate internal progress estimate is:

`progress_t = V_success(b_next, g) - V_success(b_prev, g)`

However, this is a change in predicted prospects, not necessarily the causal contribution of the action.

Separate progress from action quality.

Do not use increases in the evaluator's confidence as independently verified achievement.

## 4.5 Stable value evaluation

Success predictions depend on future policy competence.

As CAM learns, previously accurate predictions may become outdated.

Condition evaluation on relevant policy versions or learned policy representations.

Investigate frozen target evaluators, cross-fitting and recalibration.

Prevent a rapidly changing actor from optimizing temporary critic errors.

The self-evaluator must remain falsifiable by actual future outcomes.

# PART 5 — META-LEARNED INTERNAL REWARD FUNCTION

THIS IS A NON-OPTIONAL CORE REQUIREMENT.

CAM must learn its own reward function through actual meta-learning.

A network that outputs rewards but is trained only to imitate Crucible's existing reward is insufficient.

A fixed weighted sum of handcrafted intrinsic bonuses is insufficient.

A reward model trained to predict success is useful, but is not by itself a meta-learned intrinsic reward function.

Implement the following distinct optimization process.

## 5.1 Trainable intrinsic reward generator

Define:

`r_internal_t = R_ψ(b_t, e_a, b_next, e_g, uncertainty, outcome_evidence, learning_context)`

Where `ψ` is a set of trainable parameters belonging to CAM's intrinsic evaluation computation.

It should receive useful representations of:

* Previous latent understanding.
* Actual action.
* Resulting latent understanding.
* Current objective, if any.
* Observed consequences.
* Uncertainty before and after the action.
* Prediction errors.
* Estimated action contribution.
* Expected learning value.
* Resource costs.
* Relevant lifetime history.

Allow the network to learn nonlinear combinations of these signals.

Do not permanently determine their relative weights by hand.

Use bounded or otherwise stabilized reward outputs, with scale controls.

Record the learned reward separately from verified outcomes.

The network must support both assigned-goal and goal-free contexts.

## 5.2 Inner learning loop

CAM acts using its current policy parameters `θ`.

It receives observation-only experience from Crucible.

It generates intrinsic rewards using `R_ψ`.

The policy undergoes a differentiable update based on those rewards:

`θ_(k+1) = θ_k - α ∇_θ L_inner(θ_k; ψ, D_k)`

Here `D_k` contains actual experience collected by the current learner.

The reward generator must meaningfully affect the policy gradient or other parameter-update mechanism.

Do not detach the reward generator in the differentiable inner loop if doing so eliminates the required meta-gradient.

Do not accidentally allow the actor to directly optimize the reward network simply to increase reward outputs.

The reward generator should be updated through the outer learning objective.

Maintain explicit parameter ownership and gradient-routing rules.

## 5.3 Outer lifetime objective

After the inner learning updates, evaluate the updated agent on subsequent independently selected training objectives.

Define:

`J_outer(ψ) = E[U_verified(τ_query; θ_K(ψ))]`

Where:

* `θ_K` is the agent after K intrinsic-reward-driven updates.
* `τ_query` is newly collected query experience.
* `U_verified` is independently verified objective achievement under the research protocol.

The outer objective must measure whether learning with the intrinsic rewards produces a more capable agent.

It must not measure the magnitude of the rewards generated by `R_ψ`.

Update the reward function through:

`ψ ← ψ + β ∇_ψ J_outer`

For nondifferentiable environment interaction, use an appropriate score-function or policy-gradient estimator over the post-update policy's query trajectories.

The environment itself does not need to be differentiable.

Do not attempt to backpropagate through real-world API execution or other nondifferentiable actions.

A valid truncated meta-gradient can use a query policy-gradient surrogate with independently verified returns or advantages and differentiate its dependence on `θ_K` back through the inner updates into `ψ`.

The necessary path is:

`intrinsic reward parameters → inner learning update → changed policy parameters → subsequent verified performance → intrinsic reward update`

Implement and test this pathway explicitly.

## 5.4 Learn from entire lifetimes

The reward generator should be optimized over extended lifetimes, not just the episode in which a particular reward was generated.

An experience that produces no immediate material benefit may improve performance on a later objective.

The outer optimization should be able to favor such experiences.

Support lifetimes containing:

* Multiple tasks.
* Multiple environments.
* Goal-free learning periods.
* Changes in environmental mechanisms.
* New opportunities.
* Previously unfamiliar objectives.
* Retention tests.

Implement a separately grounded lifetime-value prediction function, inspired by Zheng et al.

Its targets must come from subsequent independently verified lifetime performance.

Do not use the intrinsic rewards themselves as the sole targets for the lifetime evaluator.

## 5.5 Practical computation

Full differentiation through thousands of lifetime updates is likely too expensive.

Implement:

* Truncated meta-gradient unrolling.
* Small differentiable update windows.
* Functional model parameter updates.
* Gradient checkpointing where useful.
* Batched meta-training lifetimes.
* First-order or implicit alternatives only where validated.
* Explicit compute accounting.

Start with inner update horizons such as 1, 2, 4 and 8 optimizer updates.

These are development configurations, not permanent architectural limits.

Test longer unrolls where feasible.

Compare truncated estimates against exact differentiation on small diagnostic problems.

Use finite-difference checks to establish that the meta-gradient has the expected direction.

## 5.6 Reward-function transfer

Test whether the learned reward function contains reusable knowledge.

Evaluate:

* Transfer to unfamiliar world mechanisms.
* Transfer to unfamiliar task objectives.
* Transfer to differently initialized learners.
* Transfer to compatible alternative policy architectures.
* Transfer after clearing episodic memories.
* Transfer while keeping the learned reward function fixed.

A reward function that only memorizes the training environments is insufficient.

The learned reward function should help previously unfamiliar learning processes acquire useful behavior more efficiently.

## 5.7 Independent safeguards

Neither CAM nor its reward network may modify:

* Independent goal definitions.
* Verification logic.
* Hard permission boundaries.
* Resource budgets.
* External authorization requirements.
* Sealed evaluation sets.

Learned incentives govern learning and action selection within those constraints.

They do not override them.

# PART 6 — RETROSPECTIVE LEARNING AND DELAYED OUTCOME CREDIT

THIS IS MANDATORY.

CAM must reconsider the importance of previous experiences as later consequences become observable.

A decision made thousands of actions earlier may be responsible for a future outcome.

Conversely, a decision immediately preceding success may have contributed nothing.

Do not assume credit is proportional to temporal proximity.

## 6.1 Persistent outcome-linked experience

Implement durable event histories capable of linking:

* Original actions.
* Executed programs.
* Entities affected.
* Software artifacts created.
* Subsequent dependencies.
* Intermediate environmental changes.
* Delayed outcomes.
* Ultimate objective achievement.

For example:

Day 12: CAM implements a new feature.

Day 45: Customers begin adopting the feature.

Day 90: Retention improves.

Day 150: Revenue substantially increases.

Day 180: The business reaches its verified objective.

The system must be capable of revisiting the Day 12 experience when Day 180's result becomes available.

Do not require all relevant events to fit inside a single transformer context window.

Use persistent event storage, learned retrieval and multiscale trajectory representations.

Retrospective replay should retrieve relevant earlier experiences while preserving their original observable context.

## 6.2 RUDDER-style sequence return decomposition

Implement a learned sequence model predicting eventual independently grounded returns or outcomes from trajectory prefixes.

Let:

`F_ξ(H_t, g)` approximate the eventual return associated with the trajectory.

Train it against completed trajectory outcomes.

A candidate redistributed contribution is:

`c_t = F_ξ(H_(t+1), g) - F_ξ(H_t, g)`

Apply appropriate initial offsets and final residual corrections.

For an undiscounted trajectory, verify that redistributed returns sum to the original realized return.

For discounted or variable-duration objectives, implement the mathematically appropriate discounted conservation condition.

Do not use the undiscounted formula unchanged for incompatible return definitions.

Ensure that:

* Final outcomes are not available to the online policy before they occur.
* Retrospective information is used only during legitimate learning updates.
* Unresolved trajectories are handled separately.
* Redistribution does not create additional objective achievement.
* Prediction errors are accounted for through residual correction.

Test whether the sequence model identifies consequential earlier events.

Do not claim that every change in return prediction is necessarily causal.

That is why counterfactual contribution analysis is also required.

# PART 7 — COCOA AND COUNTERFACTUAL CREDIT ASSIGNMENT

THIS IS A MANDATORY IMPLEMENTATION WORKSTREAM.

The agent should learn to answer the following computational question:

Would this outcome have been more or less likely if I had chosen a different action?

## 7.1 Basic decision-quality estimate

Define:

`A(b, a, g) = Q(b, a, g) - V(b, g)`

This estimates the quality of a particular decision relative to the reference policy's alternatives.

It should not be confused with the raw material outcome.

A competitor's unexpected failure may benefit CAM without CAM having made a particularly good decision.

A carefully designed experiment may fail despite being a good decision.

## 7.2 Implement actual COCOA contribution coefficients

Follow the published COCOA derivation.

For the appropriate undiscounted finite-action formulation:

`w(s, a, u) = [Σ_(k≥1) Pπ(U_(t+k)=u | S_t=s, A_t=a)] / [Σ_(k≥1) Pπ(U_(t+k)=u | S_t=s)] - 1`

Using the corresponding hindsight distribution:

`w(s, a, u) = Pπ(A_t=a | S_t=s, U_future=u) / π(a|s) - 1`

Here:

* `u` is a rewarding outcome representation.
* `π` is the reference policy.
* The numerator reflects the outcome's occurrence under the selected action.
* The denominator reflects occurrence under the policy's distribution over alternatives.

For finite-horizon, discounted and variable-duration settings, derive and implement the corresponding weighting.

Do not treat a probability of an entire successful trajectory as identical to the published outcome-occurrence definition.

## 7.3 Learn a hindsight action model

Train:

`h_κ(a | b_t, u_future, goal, policy_context)`

to predict the observed earlier action given its original context and a subsequently observed outcome.

Use appropriate cross-entropy objectives or validated density-ratio estimation.

Record actual behavior-policy probabilities.

Account for changes in the policy distribution.

The hindsight model is permitted to access future information during retrospective training.

The online action-selection model is not.

Maintain clear separation between these computational pathways.

## 7.4 Implement the policy-gradient estimator

For the reference undiscounted setting, implement the COCOA estimator incorporating:

* Immediate action reward contribution.
* Future outcome rewards.
* Counterfactual contribution coefficients.
* Gradients of the probabilities of feasible alternative actions.

A reference form is:

`G_COCOA = Σ_t [∇ log π(A_t|S_t) R_t + Σ_a ∇π(a|S_t) Σ_(k≥1) w(S_t,a,U_(t+k)) R_(t+k)]`

Use the paper's estimator and assumptions for reference validation.

Then develop the appropriate CAM extension.

Do not simply multiply an ordinary sampled actor-critic advantage by a hindsight score and claim to have reproduced COCOA.

Implement a reference finite-action version first and verify it against exact gradients on small environments.

## 7.5 Rewarding outcome representations

Represent specific outcomes, not entire irrelevant future states.

For example:

* An independently paying customer.
* Verified software correctness.
* Successful service adoption.
* A scientific prediction confirmed by an experiment.
* An independently accepted code contribution.
* A reduction in operational failures.

Distinct outcomes with the same scalar utility may need distinguishable representations.

Train outcome encoders that preserve information relevant to the return.

Test whether they discard irrelevant distinctions that increase gradient variance.

Do not assume that arbitrary high-dimensional embeddings satisfy COCOA's theoretical requirements.

Measure approximation error and bias.

## 7.6 General-purpose action spaces

CAM can generate executable programs, code changes and semantic interventions.

The action space may be too large to enumerate.

Investigate:

* Probability-tracked candidate action selection.
* Sampled alternative actions.
* Hierarchical program actions.
* Learned action embeddings.
* Importance sampling where support exists.
* Contrastive density-ratio estimation.

Account for the actual probability of the proposal and selection process.

If proposal probabilities cannot be computed, do not claim an unbiased estimator over the entire open-ended action space.

Explicitly define the restricted candidate-policy estimator being used.

Measure candidate coverage and action-support limitations.

## 7.7 Partial observability and causal limitations

The real internet contains unobserved causes, changing participants and nonstationary dynamics.

A hindsight probability is not automatically a real-world causal effect.

Use the original observable history or a validated belief representation.

Track uncertainty.

Where appropriate, validate estimated contributions against controlled interventions.

Crucible's private counterfactual forks should provide independent ground-truth evaluation under declared simulator assumptions.

Do not expose those private counterfactual results to the ordinary actor.

For real-world environments, use controlled authorized experiments and observational methods only where their assumptions are defensible.

Avoid unsupported causal claims.

## 7.8 Implement and compare R-HCA

Investigate the 2026 R-HCA method as an alternative to COCOA, especially under noisy immediate feedback.

Implement its actual mathematical estimator.

Compare:

* Standard actor-critic.
* Monte Carlo return.
* TD(λ) or GAE.
* RUDDER.
* COCOA.
* R-HCA.
* Learned model-based counterfactual credit.

Use equal data and compute budgets where feasible.

Measure bias, variance, gradient alignment, sample efficiency and actual objective achievement.

Do not assume any particular method must win.

## 7.9 Integrating the credit mechanisms

RUDDER, COCOA and R-HCA are alternative or complementary mechanisms for assigning credit.

They must not independently duplicate the same final reward and then sum it three times.

Maintain a mathematically explicit accounting of the underlying objective.

Use:

* One selected credit estimator.
* Controlled combinations.
* Or a meta-learned mixture validated against the underlying objective.

If a learned credit mixture is used, evaluate it against independently verified downstream performance.

The best empirically supported implementation should become the default.

# PART 8 — LEARN THE VALUE OF ACQUIRING KNOWLEDGE

CAM must learn not only how valuable an action is for immediate goal achievement, but how valuable an experience is for developing future competence.

## 8.1 Separate distinct forms of information

Distinguish:

* Observation surprise.
* Belief change.
* Epistemic uncertainty reduction.
* Irreducible environmental randomness.
* Predictive learning progress.
* Decision-relevant knowledge acquisition.
* Persistent capability improvement.

A surprising observation is not necessarily useful.

A repeated unpredictable event should not produce unlimited intrinsic reward.

## 8.2 Decision-relevant information value

Estimate how much resolving an uncertainty could improve subsequent decisions.

A candidate approximation is:

`VOI(a) = E_y[max_a' Q(b_updated(a,y), a', g)] - max_a' Q(b_current,a',g)`

Account for:

* Environmental changes caused by the experiment.
* Resource cost.
* Time elapsed.
* Risk.
* Opportunity cost.
* The continuation policy.

Compare against random exploration, posterior sampling, prediction-error curiosity and ordinary information gain.

## 8.3 Learn actual transfer improvement

The strongest signal for knowledge acquisition is improved future capability.

Define an experimental estimate of learning value:

`learning_gain(D) = J_query(U(θ,D)) - J_query(θ_control)`

Where:

* `D` is the newly acquired experience.
* `U` is the actual neural learning procedure.
* `θ_control` is a matched control that has not learned from that experience.
* `J_query` is performance on independent training-query tasks.

Use matched experience and compute controls.

This is expensive to measure directly for every action.

Train a lightweight prediction function to estimate useful learning value from latent experience.

Periodically correct it using actual measured downstream learning gains.

The intrinsic reward generator may consume this learned estimate.

Its usefulness must ultimately be determined by the outer lifetime objective.

# PART 9 — AUTONOMOUS LEARNING WITHOUT AN ASSIGNED OBJECTIVE

THIS IS MANDATORY.

CAM must be capable of spending a portion of its lifetime learning autonomously when nobody provides an immediate objective.

The learner should not become idle merely because an external task has ended.

## 9.1 Autonomous learning mode

Support:

`goal_mode = "autonomous_learning"`

In this mode, CAM receives access to an authorized environment and an exploration resource budget.

It should be capable of:

* Discovering unfamiliar systems.
* Identifying uncertainty.
* Generating its own learning objectives.
* Conducting experiments.
* Attempting unfamiliar skills.
* Solving novel programming problems.
* Investigating scientific and economic relationships.
* Testing competing explanations.
* Practicing capabilities.
* Revising its own learning curriculum.

The autonomous curriculum must not be limited to selecting a predefined task category.

The agent should generate new goals expressed as semantic or structured outcome descriptions.

## 9.2 Ground self-generated goals

A self-generated goal should contain, where applicable:

* Desired measurable outcome.
* Relevant entities.
* Authorized actions.
* Resource limits.
* Time horizon.
* Evidence required to verify achievement.
* Expected learning benefit.
* Uncertainty.

A goal compiler can translate semantic objectives into testable predicates when supported by observable evidence.

The learner must not be able to certify an objective merely by asserting that it succeeded.

Unsupported or unverifiable goals may still motivate investigation, but must not be counted as verified achievements.

## 9.3 Learned curriculum selection

Implement a trainable curriculum policy:

`g_self ~ C_χ(b_t, lifetime_state, uncertainty, available_resources)`

The curriculum policy should learn to select experiences that improve future competence.

Its training signal must be based on actual subsequent learning and transfer.

Do not reward it simply for completing goals it invented.

Otherwise, it may generate indefinitely trivial goals.

Do not reward it simply for choosing the hardest possible goals.

Otherwise, it may waste its lifetime on impossible tasks.

The desired behavior is allocation of experience toward achievable, informative and transferable learning.

## 9.4 Goal-free meta-learning

Construct training lifetimes containing:

1. Initial checkpoint.
2. Goal-free exploration period.
3. Actual neural weight updates.
4. Independently selected subsequent objectives.
5. Measured goal achievement and adaptation efficiency.
6. Meta-update to the internal reward generator and curriculum policy.

The subsequent objectives must not be disclosed during the preceding goal-free exploration period.

Compare against:

* No autonomous learning.
* Random exploration.
* Fixed training curricula.
* Ordinary novelty seeking.
* Hand-designed learning-progress curricula.
* Learned autonomous curricula.

The main question is whether CAM discovers how to spend its own learning time more productively.

## 9.5 Persistent acquisition

Experience from autonomous learning must update actual neural parameters.

This includes persistent representations, predictive models, learned priors, policies, evaluator parameters and intrinsic reward parameters.

Do not substitute written memory or retrieval for neural internalization.

Written records and external memory may supplement neural learning.

They cannot be the only mechanism of learning.

# PART 10 — HINDSIGHT LEARNING FROM PARTIAL SUCCESSES

CAM should extract useful learning from failed attempts.

For example, an agent attempting to achieve $100,000 ARR might fail at the overall goal but learn how to deploy a useful product, acquire independent users or solve an important infrastructure problem.

Implement goal-conditioned hindsight replay.

Allow actual trajectories to be relabeled with alternative goals that were verifiably achieved.

Preserve the original goal and its actual success or failure.

Do not relabel an original failed task as successful.

The relabeled experience should provide additional training for transferable capabilities.

For semantic goals, require grounded achieved-outcome evidence.

Handle changes in policy conditioning and off-policy distributions.

Do not assume standard hindsight replay is unbiased under arbitrary stochastic semantic environments.

Measure whether hindsight relabeling improves acquisition efficiency and retention.

# PART 11 — INTEGRATED TRAINING ALGORITHM

The following computational structure is mandatory.

Do not implement the parts as disconnected research prototypes.

## 11.1 Online interaction

At every decision:

1. Receive legitimately observable events.
2. Encode semantic and exact numerical observations.
3. Update the fast adaptive belief state.
4. Represent the current objective or autonomous-learning context.
5. Generate feasible executable actions.
6. Predict consequences and uncertainty.
7. Estimate objective prospects.
8. Estimate decision-relevant learning value.
9. Select an authorized action.
10. Execute it.
11. Receive actual observations.
12. Generate an internal training reward.
13. Record the full transition and prediction provenance.
14. Update beliefs and perform eligible neural learning.

## 11.2 Delayed feedback

When delayed consequences become observable:

1. Associate the outcome with relevant prior events.
2. Determine whether the outcome is verified, uncertain or pending.
3. Update the world-model predictions.
4. Correct previous success predictions.
5. Recompute eligible retrospective credit.
6. Update the hindsight contribution model.
7. Add or revise replay targets.
8. Perform appropriate persistent weight updates.

Preserve original event records.

Do not overwrite historical predictions with later knowledge.

## 11.3 Inner intrinsic-reward learning

Use actual internally generated rewards to update the policy.

The normal CAM actor must not require native Crucible reward values.

Support an internal-reward-only actor-training condition as the new default research configuration.

Train world prediction and outcome evaluation from actual observed evidence.

Keep independently verified objective utility available to the meta-learning procedure.

## 11.4 Outer meta-learning

Periodically:

1. Sample training lifetimes.
2. Record the initial learner parameters.
3. Train the learner using its intrinsic reward generator.
4. Collect fresh subsequent experiences.
5. Independently verify subsequent achievements.
6. Estimate the meta-gradient through the inner updates.
7. Update the intrinsic reward generator.
8. Update the lifetime evaluator and autonomous curriculum policy.
9. Evaluate against independent validation tasks.
10. Preserve and distribute validated checkpoints.

The implementation must show that changing the intrinsic reward generator changes the inner learning dynamics.

It must then show whether those changed dynamics improve subsequent externally verified performance.

## 11.5 Loss functions

Maintain an explicit decomposition of objectives, potentially including:

`L_world`: probabilistic prediction of actual world consequences.

`L_success`: independently grounded goal-success prediction.

`L_outcome`: magnitude, timing and resource-outcome prediction.

`L_return`: sequence-based verified return prediction.

`L_hindsight`: retrospective action/outcome contribution learning.

`L_policy`: policy improvement using the selected internal learning signals.

`L_meta`: improvement of the intrinsic reward function through downstream lifetime performance.

`L_curriculum`: improvement of autonomous learning through subsequent competence.

`L_retention`: preservation of previously acquired capabilities, where justified.

Do not optimize every parameter indiscriminately using a single uncontrolled sum of these losses.

Document:

* Which losses update which parameters.
* When gradients are stopped.
* Which updates must be differentiable.
* Which networks use target parameters.
* Which training data each objective receives.
* How shared representation gradients are coordinated.
* How reward, evaluator and policy versions remain consistent.

Use gradient tests to verify the implementation.

# PART 12 — REWARD INTEGRITY AND STABILITY

Self-generated rewards introduce a major scientific challenge.

The agent must not become more skilled at generating reward than at achieving genuine outcomes.

## 12.1 Independent grounding

CAM's internally generated predictions and rewards must never count as independent evidence of success.

The reward function is optimized for learning effectiveness.

The outcome evaluator establishes whether objectives were achieved.

These roles must remain separate.

## 12.2 Prevent circular optimization

Test for:

* Artificially inflated success probabilities.
* Reward outputs increasing without improved behavior.
* Repetitive trivial self-generated goals.
* Curiosity toward irreducible randomness.
* Prediction manipulation.
* Exploitation of evaluator defects.
* Fabricated customer activity.
* Circular transactions.
* Repeated counting of identical outcomes.
* Artificially favorable replay selection.
* Leakage of future evidence.
* Memorization of evaluation task identifiers.

## 12.3 Stability

Implement and evaluate:

* Reward magnitude controls.
* Reward-model versioning.
* Stable normalization using legitimately available statistics.
* Target evaluator versions.
* Meta-gradient clipping.
* Independent validation.
* Conservative policy updates.
* Uncertainty-aware decision-making.
* Checkpoint promotion and rollback.

Do not promote a checkpoint solely because it achieves a higher internally generated return.

Use independent validation and retention criteria.

## 12.4 Reward redistribution versus intrinsic learning

Keep separate:

* Verified objective utility.
* Return redistribution.
* Learned intrinsic reward.
* Policy-gradient credit.
* Learning-value prediction.

These quantities have different meanings.

Do not casually add them together.

If potential-based shaping is used as a baseline, preserve the underlying discounted objective and handle terminal potentials appropriately.

The unrestricted meta-learned intrinsic reward is an optimization of learning dynamics, not a theorem guaranteeing policy invariance.

That is why downstream independent evaluation is essential.

# PART 13 — DISTRIBUTED CONTINUAL LEARNING

The system should support multiple independently acting CAM instances sharing persistent neural knowledge.

Each actor maintains its own:

* Environment.
* Identity.
* Current objective.
* Fast adaptive state.
* Local evidence history.
* Resource budget.

A central learner maintains:

* Shared trainable representations.
* Predictive models.
* Goal-conditioned evaluation parameters.
* Intrinsic reward parameters.
* Credit-assignment parameters.
* Curriculum parameters.
* Persistent optimizer state.
* Versioned experience replay.

## 13.1 Versioned learning

Log:

* Actor policy version.
* Reward model version.
* Evaluator version.
* Action proposal distribution.
* Action selection probability.
* Environment version.
* Outcome timestamp.
* Evidence provenance.

Account for stale actor policies and changing reward functions.

Do not silently train on stale intrinsic rewards as though they were generated by the current model.

Support justified recomputation from historical evidence.

## 13.2 Retention

Use balanced replay across domains, environments and capability types.

Test whether learning new objectives damages previously learned evaluation or control competence.

Measure:

* Forward transfer.
* Backward transfer.
* Forgetting.
* Adaptation speed.
* Retention of reward-function knowledge.
* Retention after new domains are learned.

## 13.3 Durable state

Persist:

* Neural parameters.
* Intrinsic reward parameters.
* Optimizer states.
* Experience archives.
* Pending delayed outcomes.
* Curriculum state.
* Evaluation records.
* Replay indices.
* Experimental configurations.
* Random seeds.

Verify restart and resume.

Do not depend on an ephemeral GPU instance as the sole location of lifetime experience.

# PART 14 — TRAINING ENVIRONMENT DIVERSITY

Do not restrict this architecture to the original six numerical mechanism families.

That would severely limit the scientific meaning of the results.

Use the broader Crucible 5.0 training universe.

Include:

* Hidden causal systems.
* Scientific experimentation.
* Delayed consequences.
* Technical systems.
* Network dynamics.
* Resource allocation.
* Strategic competition.
* Cooperation.
* Nonstationary environments.
* Software engineering.
* Digital product creation.
* Business development.
* Community coordination.
* Industrial and infrastructure systems.
* Integrated persistent civilizations.

Preserve untouched held-out mechanism families and independent environment implementations.

Train across structurally different worlds, not merely different seeds of the same six relationships.

Run controlled numerical experiments and full OpenWorld learning concurrently.

Do not postpone all semantic or open-ended training until numerical experiments pass.

Use the rich OpenWorld interfaces for actual executable software, services, markets, organizations and adaptive participants.

Keep CPU-efficient simulations for large-scale experience collection.

Use larger semantic models selectively and account for all inference costs.

The learned reward mechanism must be shared across these environments.

Do not introduce separate handcrafted learning incentives for every domain.

# PART 15 — MANDATORY SCIENTIFIC EXPERIMENTS

All experiments in this section must be implemented and executed.

They are not optional future work.

Distinguish implementation validation from empirical superiority.

Use development experiments to fix issues, followed by independently held-out confirmation.

Prespecify success criteria, practical effect sizes and statistical analysis before decisive runs.

## EXPERIMENT A — Calibration

Question:

Does CAM accurately evaluate its prospects of achieving unfamiliar objectives?

Test:

* Predictions made before eventual outcomes.
* Multiple goal families.
* Different horizons.
* Unseen mechanism structures.
* Novel objectives.
* Different policy versions.

Measure Brier score, negative log-likelihood, calibration curves and uncertainty.

Verify that the evaluator is corrected when confidently wrong.

A decreasing training loss alone does not establish calibration.

## EXPERIMENT B — Action-quality evaluation

Question:

Do actions CAM evaluates highly actually produce better outcomes than plausible alternatives?

Use controlled counterfactual forks.

Select equivalent initial states.

Compare alternative actions under specified continuation policies and appropriately coupled randomness.

Measure action-ranking accuracy, regret, effect prediction error and realized performance.

Include:

* Lucky successes.
* Good decisions with bad stochastic outcomes.
* Bad decisions with lucky outcomes.
* External shocks unrelated to CAM's actions.
* Confounded observational relationships.

Compare raw progress evaluation against counterfactual credit estimation.

## EXPERIMENT C — Delayed credit assignment

Question:

Can CAM identify consequential actions hundreds or thousands of decisions before an outcome?

Create controlled tasks with delays of approximately:

* 16 decisions.
* 64 decisions.
* 256 decisions.
* 1,024 decisions.
* Longer horizons where feasible.

Include distracting actions and unrelated intermediate outcomes.

Run RUDDER, COCOA, R-HCA and standard RL baselines.

On small exact environments, compare against analytically computed contribution coefficients and policy gradients.

On full OpenWorld tasks, use actual event histories and controlled intervention experiments.

Measure:

* Attribution accuracy.
* Gradient signal-to-noise ratio.
* Gradient bias and variance.
* Reward conservation.
* Sample efficiency.
* Final objective achievement.

Test whether a delayed outcome can trigger meaningful learning from previously archived experience.

## EXPERIMENT D — Learning efficiency

Question:

Does the learned reward function produce faster acquisition of new capabilities?

Compare identical architectures under:

1. Historical Crucible native rewards.
2. Sparse verified terminal utility.
3. Learned goal-progress evaluation.
4. Hand-designed intrinsic motivation.
5. Meta-learned intrinsic reward.
6. Meta-learned reward plus retrospective credit.
7. Full architecture with autonomous curriculum learning.

Use observation-only Crucible operation for the new CAM conditions.

The sparse verified terminal baseline receives its task utility only from the independent evaluator.

Match training experience and separately match computation.

Measure:

* Interactions to competence.
* Area under the learning curve.
* Training computation.
* Inference computation.
* Wall-clock cost.
* Generalization.
* Actual verified outcomes.

An increase in internal reward is not evidence of improved learning efficiency.

## EXPERIMENT E — Novel objectives

Question:

Can the learned evaluator and reward function generalize to goals never encountered during training?

Hold out entire objective families.

Use structurally unfamiliar combinations of constraints, entities and desired outcomes.

Evaluate whether CAM can infer which experiences are relevant without a newly handcrafted reward function.

Compare against a frozen pretrained model and simpler learned evaluation baselines.

## EXPERIMENT F — Reward integrity

Question:

Can CAM increase its internal scores without making genuine progress?

Construct adversarial diagnostic worlds containing:

* Easy but useless achievements.
* Fabricated-looking evidence.
* Duplicate outcome events.
* Irreducible unpredictable observations.
* High intermediate activity with low ultimate utility.
* Short-term revenue with poor long-term sustainability.
* Misleading apparent correlations.
* Attractive but ineffective strategies.

Test whether the learned reward system remains grounded.

Independent verification must detect false achievement.

The learner must not receive access to evaluator internals.

## EXPERIMENT G — Parametric transfer

Question:

Does the learned evaluation and learning ability persist in neural weights?

Train CAM across diverse worlds.

Save before and after checkpoints.

Start a fresh process.

Remove:

* Prior written context.
* Retrieved training memories.
* Task-specific fast belief state.
* Training trajectory access.
* Cached solutions.

Retain only the appropriate trained neural parameters and fixed model configuration.

Evaluate on unfamiliar mechanisms and objectives.

Separately test:

* Transfer of the core representation.
* Transfer of the intrinsic reward generator.
* Transfer of the learned evaluator.
* Transfer of the complete architecture.

Measure whether the experienced model acquires competence faster than the inexperienced model.

## EXPERIMENT H — Real-world transfer

Question:

Can the same learned evaluator operate and learn from authorized real-world observations without Crucible-specific rewards?

Implement and exercise the common experience interface against controlled real-world services.

Use authorized repositories, owned staging services, consented user feedback or another bounded independently verifiable environment.

The default live adapter must not permit uncontrolled public actions.

Require explicit authorization for consequential external actions.

Maintain permissions, resource limits and an independent stop mechanism outside the learned reward system.

Test whether actual observations and delayed outcomes can produce:

* Immediate adaptive belief updates.
* Corrected outcome predictions.
* Internally generated learning rewards.
* Retrospective credit assignment.
* Persistent neural weight updates.

Do not substitute a mocked API and claim live-world transfer.

If authorization or access is unavailable, complete the adapter and independent controlled tests, mark the live experiment as blocked, and report that external transfer remains unverified.

Do not claim this experiment passed without actual authorized execution.

## EXPERIMENT I — Learning without an objective

Question:

Does CAM autonomously acquire useful capabilities when no external objective is assigned?

Give the agent a fixed exploration budget in an unfamiliar environment.

Do not reveal the subsequent evaluation objectives.

Allow it to generate its own goals and conduct permitted experiments.

Then evaluate its updated weights on independently selected unfamiliar tasks.

Compare against:

* No exploration.
* Random exploration.
* Fixed curriculum.
* Ordinary curiosity.
* Meta-learned autonomous curriculum.

Measure downstream adaptation speed and competence.

The number of self-generated goals completed is not the primary success metric.

## EXPERIMENT J — Meta-learned reward verification

Question:

Is the reward function genuinely learning through meta-gradients?

Verify:

* Nonzero meaningful gradients into reward parameters.
* Reward parameters influence inner policy updates.
* Inner updates influence query policy behavior.
* Outer verified utility influences reward updates.
* Removing the meta-gradient eliminates the relevant learning pathway.
* The reward network learns differently across different lifetime objectives.

Compare exact and truncated meta-gradients on small problems.

Construct a task where an immediately costly exploration action improves future lifetime performance.

Test whether the intrinsic reward function learns to value that experience.

This is essential.

Do not infer meta-learning from the presence of a neural reward network.

## EXPERIMENT K — Long-horizon OpenWorld achievement

Question:

Can the architecture achieve difficult open-ended goals using its own learned evaluation and incentives?

Use full Crucible 5.0, not merely numerical controls.

Include objectives involving:

* Creating and operating useful software.
* Building sustainable recurring revenue.
* Improving technical infrastructure.
* Coordinating independent participants.
* Discovering unfamiliar scientific mechanisms.
* Building useful communities.
* Responding to changing market conditions.

Include long-horizon tasks of 30, 90 and 180 simulated days where feasible.

Use appropriate world configurations and feasible objectives.

Measure independently verified success, not simulated claims about success.

Compare against frozen language-model agents, memory-only agents and prior CAM checkpoints.

The agent must interact through actual executable world interfaces.

# PART 16 — HARD IMPLEMENTATION COMPLETION CHECKLIST

The master agent must maintain and independently verify every item below.

A checked item requires actual source code, integration and executed acceptance-test evidence.

Environment

* Crucible 5.0 observation-only mode exists.
* Observation-only mode is the default for new CAM runs.
* Native shaped rewards are disabled.
* Native reward leakage through observations is eliminated.
* Independent outcome verification remains functional.
* Goal-free persistent operation works.
* Delayed outcomes are represented correctly.
* Full OpenWorld integration is exercised.
* Historical reward-based baselines remain reproducible.
* Diverse training worlds are supported.

Unified representation

* Semantic observations enter shared numerical representations.
* Exact numerical evidence is preserved.
* Goals are encoded in shared latent space.
* Actions are represented semantically and numerically.
* Fast adaptive beliefs are integrated.
* Persistent backbone weights receive learning updates.
* Important gradient pathways are verified.

Self-evaluation

* Goal-conditioned success prediction works.
* Actual eventual outcomes supervise earlier predictions.
* Outcome magnitude is predicted.
* Time to achievement is predicted.
* Resource consumption is predicted.
* Uncertainty is represented.
* Pending outcomes are not incorrectly treated as failures.
* Calibration is measured on independent data.
* Novel-objective evaluation is tested.

Learned internal rewards

* A trainable intrinsic reward generator exists.
* It consumes actual latent experience representations.
* CAM's inner policy uses internally generated rewards.
* The reward generator influences real neural updates.
* Differentiable inner updates are implemented.
* Outer lifetime optimization is implemented.
* Actual subsequent verified outcomes supervise meta-learning.
* Meta-gradient correctness is tested.
* Reward-model transfer is tested.
* Learned reward stability is tested.
* Reward integrity is independently evaluated.

Retrospective credit assignment

* Persistent event histories exist.
* Delayed outcomes link to earlier experiences.
* RUDDER-style return prediction is implemented.
* Return-equivalent redistribution is verified.
* COCOA contribution coefficients are implemented.
* A hindsight model is trained.
* COCOA policy-gradient estimation is implemented.
* R-HCA is implemented and evaluated.
* Counterfactual action contributions are independently tested.
* Long-horizon attribution is evaluated.
* Policy-distribution changes are accounted for.
* Future-information leakage is prevented.
* Duplicate reward attribution is prevented.

Autonomous learning

* CAM operates without an assigned objective.
* CAM generates new learning objectives.
* Self-generated objectives are grounded in evidence.
* Learned experiment selection exists.
* Decision-relevant learning value is estimated.
* Autonomous curriculum parameters are trainable.
* Goal-free lifetime meta-training exists.
* Autonomous experience changes persistent neural weights.
* Downstream independently selected objectives test learning.
* Trivial self-generated goal exploitation is tested.
* Irreducible-noise exploitation is tested.

Continual learning

* Durable replay is implemented.
* Delayed experience reconciliation works.
* Multiple actors share a central learner.
* Policy and reward versions are recorded.
* Reward recomputation is supported where justified.
* Retention testing is implemented.
* Fresh-process weights-only evaluation works.
* Complete training state can be resumed.
* Important state is durably preserved.

Scientific experiments

* Calibration experiment executed.
* Action-quality experiment executed.
* Delayed-credit experiment executed.
* Learning-efficiency experiment executed.
* Novel-objective experiment executed.
* Reward-integrity experiment executed.
* Parametric-transfer experiment executed.
* Authorized real-world transfer experiment executed or explicitly blocked.
* Goal-free learning experiment executed.
* Meta-learned reward verification experiment executed.
* Full long-horizon OpenWorld experiment executed.

Independent verification

* Independent verifier inspected every mandatory implementation.
* Independent verifier traced actual runtime integration.
* Independent verifier executed acceptance tests.
* Raw experiment evidence is preserved.
* Relevant results are reproducible.
* All unmerged workstreams are resolved.
* Completion checker passes for engineering requirements.
* Scientific failures are accurately reported.

A manually checked Markdown box is not sufficient evidence.

Synchronize this checklist with the executable completion manifest.

The master agent may not claim full completion while mandatory implementation requirements remain unimplemented or unverified.

If resources prevent an experiment from being completed, record the blocker and preserve the reproducible state.

Do not silently waive requirements.

# PART 17 — IMPLEMENTATION AND INTEGRATION ORDER

Use parallel subagents, but establish dependencies explicitly.

Phase 0

Recover the current code and experiment state.

Establish the shared contracts.

Create the completion manifest.

Freeze independent evaluation protocols.

Phase 1

In parallel, implement:

* Observation-only environment.
* Shared event and goal interfaces.
* Outcome-evaluator infrastructure.
* Reference meta-gradient algorithm.
* Reference credit-assignment algorithms.
* Persistent delayed-outcome storage.

Phase 2

Integrate:

* Shared neural representations.
* Learned success evaluation.
* Internally generated rewards.
* Retrospective learning.
* Goal-free interaction.

Run exact mathematical and gradient diagnostics.

Phase 3

Implement and run:

* Lifetime meta-training.
* Learned reward optimization.
* Counterfactual contribution learning.
* Autonomous curriculum meta-training.
* Distributed continual learning.

Phase 4

Run controlled scientific experiments.

Investigate failures.

Improve the actual architecture.

Phase 5

Train and evaluate inside full Crucible 5.0.

Run semantic, executable, long-horizon tasks.

This track should begin as soon as the minimum viable integrated interfaces are operational, rather than waiting for every numerical diagnostic to demonstrate superiority.

Phase 6

Run independent transfer, retention, reward-integrity and authorized live-world evaluations.

Phase 7

Complete independent verification.

Integrate all valid work.

Commit coherent milestones.

Produce the final research report.

Do not leave completed implementations isolated in unused branches.

Do not end after the first successful smoke test.

Do not treat the creation of many neural modules as evidence of intelligence.

# PART 18 — REQUIRED FINAL RESEARCH REPORT

The final report must directly answer:

1. Does CAM generate its own actual training rewards?
2. Is Crucible's native reward completely disabled by default?
3. Does the internal evaluator accurately predict actual future success?
4. Can it evaluate unfamiliar semantic objectives?
5. Does the intrinsic reward function receive real meta-gradients?
6. Does its learned reward function improve subsequent learning?
7. Does that reward function contain transferable knowledge?
8. Can CAM retrospectively identify important earlier actions?
9. Does COCOA, R-HCA or RUDDER improve its learning efficiency?
10. Can it distinguish good decisions from lucky outcomes?
11. Can it distinguish useful information from random surprise?
12. Can CAM learn productively without an assigned objective?
13. Does its autonomous curriculum improve later competence?
14. Do useful capabilities persist in neural weights?
15. Does it retain earlier knowledge while learning new domains?
16. Can it learn from authorized real-world observations?
17. Can it achieve difficult open-ended goals in Crucible 5.0?
18. What is its actual sample and compute efficiency?
19. Which architectural components contributed measurable benefits?
20. What remains unsuccessful or unproven?

For every claim provide:

* Actual results.
* Relevant baselines.
* Training and evaluation budgets.
* Independent seeds.
* Statistical uncertainty.
* Reproduction commands.
* Source and checkpoint identifiers.
* Known limitations.

Separate engineering completion from scientific success.

The purpose is not to defend a particular proposed architecture.

The purpose is to discover the strongest feasible learning process for acquiring transferable intelligence through experience.

If a simpler mathematical method outperforms a more complicated one, preserve that result and investigate why.

But do not omit the mandatory research mechanisms without implementing and evaluating them.

## FINAL DIRECTIVE

Build an intelligence that learns not only what to do, but also what is worth learning and what deserves reinforcement.

Its reward function must itself contain learned knowledge.

Its assessment of progress must be corrected by reality.

Its understanding of past decisions must improve when delayed outcomes arrive.

Its exploration should increasingly acquire valuable information rather than arbitrary novelty.

Its neural weights must accumulate persistent knowledge across diverse worlds.

When no objective is assigned, it should be capable of autonomously developing useful capabilities through authorized experience.

When a difficult objective is assigned, it should determine how to pursue it without needing a handcrafted intermediate reward function.

Crucible 5.0 must provide experience, not prescriptive action judgments.

The same learning machinery must be capable of operating on legitimately observable real-world feedback.

The independent evaluator must remain separate, immutable and authoritative about verified objective achievement.

The master orchestration agent must ensure that every mandatory component in this specification is genuinely implemented, integrated, tested and independently verified.

Do not stop with an architectural proposal.

Do not stop with a numerical controller.

Do not stop with a neural reward head.

Do not stop with an incomplete meta-learning demonstration.

Do not stop with unintegrated workstreams.

Do not claim capabilities that have not been demonstrated.

Implement the full system, conduct the actual experiments, investigate failures, and maintain the persistent completion record.

The ultimate goal is a continual learner whose accumulated experience makes it increasingly effective at deciding what to learn, how to learn, how to evaluate its actions, and how to achieve unfamiliar consequential objectives.

START NOW.
