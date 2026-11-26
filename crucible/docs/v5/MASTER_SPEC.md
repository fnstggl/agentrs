# CRUCIBLE 5.0 — OPENWORLD: THE ADAPTIVE INTERNET CIVILIZATION

## CORE MISSION

You are the master engineering orchestrator responsible for implementing Crucible 5.0 inside the existing AgentRS repository.

Crucible 5.0 must be a major expansion of Crucible 4.0 into a persistent, procedurally generated, interconnected digital civilization capable of training and scientifically evaluating general-purpose, experience-driven continual learning architectures.

The ultimate research objective is to discover which learning architectures most efficiently acquire the ability to understand unfamiliar environments, discover their causal mechanisms, invent useful strategies, and accomplish difficult, consequential, long-horizon objectives.

We are not building another conventional reinforcement learning benchmark.

We are building an experimental universe in which artificial intelligence can learn through experiences analogous to those it would encounter on the real internet.

An agent should be able to:

- Build and grow a business to $100,000 in genuine annual recurring revenue.
- Create original videos that reach 10 million distinct simulated viewers through genuine simulated consumption.
- Build a community with hundreds of thousands of voluntarily participating members.
- Invent useful software, products and technologies.
- Discover unknown scientific and technical relationships through experimentation.
- Create organizations and coordinate independent participants.
- Discover unfamiliar economic opportunities.
- Build services that other participants independently choose to use.
- Learn to operate amid competition, uncertainty, scarcity and changing conditions.
- Discover strategies that were never explicitly programmed or demonstrated.
- Create new tools, services and institutions that change the future opportunities available in the world.
- Transfer knowledge acquired in one civilization to fundamentally unfamiliar civilizations.

The world must provide the underlying mechanisms that make these achievements possible.

It must not supply predefined strategies, artificial shortcuts, arbitrary success counters, or task-specific action sequences.

The environment should surprise both its inhabitants and its developers.

The ultimate measure of Crucible's quality is how effectively it distinguishes a genuinely generalizable continual learner from a policy that merely memorizes simulator-specific strategies.

This is a full implementation assignment, not an architecture-planning assignment.

Do not stop after creating documentation, scaffolding, module names, interfaces, mock implementations or isolated demonstrations.

Build the system, integrate the system, execute the system, stress-test the system, independently audit the system, and continue improving it until every mandatory acceptance criterion passes or a genuine external blocker prevents further execution.

Never report completion merely because substantial code has been written.

---

# PART 0 — RECOVER AND UNDERSTAND THE EXISTING SYSTEM

Repository:

https://github.com/fnstggl/agentrs

Known relevant branches:

- `feat/crucible-4`, commit `c07098b8726e2025487f2ef6e09f9863851ac416`
- `feat/crucible-internet`, commit `11e5613`
- `feat/cam-7.0`, commit `f3cee12`
- `feat/cam-7.0-unified`, commit `b775cbafc5a7b78e205726000d7a681c597e3f4c`

These are known reference points, not instructions to discard newer work.

Before modifying anything:

1. Inspect all relevant local and remote branches, commits and uncommitted changes.
2. Determine the latest valid Crucible implementation.
3. Inspect the actual CAM learner interfaces, current training architecture and observation formats.
4. Identify existing functionality that must be preserved.
5. Check available CPU cores, RAM, GPUs, VRAM, disk capacity and storage persistence.
6. Check whether other CAM training jobs are running. Do not interrupt them or consume their allocated GPU memory.
7. Make a durable backup of relevant source code, experiment records and important artifacts using authorized storage.
8. Create a new implementation branch, preferably `feat/crucible-5-openworld`.
9. Run and record the existing regression suite.
10. Establish baseline throughput, memory consumption and world-generation costs.

Read the actual implementations and documentation, particularly:

- `crucible/docs/OPENWORLD_REPORT.md`
- `crucible/docs/OPENWORLD_ARCHITECTURE.md`
- `crucible/docs/REPRODUCE_OPENWORLD.md`
- `crucible/docs/SYNTHETIC_INTERNET.md`
- `crucible/docs/AGENT_API.md`
- `crucible/docs/LIMITATIONS.md`
- `crucible/src/crucible/core/`
- `crucible/src/crucible/worldgen/`
- `crucible/src/crucible/mechanisms/`
- `crucible/src/crucible/economy/`
- `crucible/src/crucible/actors/`
- `crucible/src/crucible/usefulness/`
- `crucible/src/crucible/selfplay/`
- `crucible/src/crucible/research/`
- `crucible/src/crucible/goals/`
- `crucible/src/crucible/api/`
- `superlearner/src/superlearner/cam7/`

Preserve the existing working infrastructure, including:

- Event-sourced world kernel.
- Persistent worlds.
- Deterministic replay.
- Snapshots and counterfactual forks.
- Executable software hosting.
- Sandboxed code execution.
- Browser and tool interfaces.
- Banking and financial accounting.
- Companies, organizations and marketplaces.
- Existing industrial systems.
- Procedural mechanism generators.
- Independent goal verification.
- Existing research protocols.
- Original training, validation and sealed test partitions.

Do not unnecessarily rewrite functioning components.

## Known limitations to address

Crucible 4.0 reportedly contains:

- 25 numerical mechanism families across three independently implemented engines.
- 10 procedural families of useful digital work.
- Persistent simulated economic worlds.
- Approximately 4,000 consumers in standard OpenWorld configurations.
- Mostly programmed stochastic participant behavior.
- Limited autonomous browsing.
- Basic forum functionality rather than a rich social internet.
- No demonstrated realistic large-scale content consumption ecosystem.
- Limited autonomous general-purpose neural participants.
- No empirically calibrated mechanisms in the reported 55-mechanism fidelity register.
- Limited demonstrated emergence of new economic opportunities.
- Self-play primarily evaluated in simpler numerical environments.
- No demonstrated learned-agent competence across the full intended open-ended civilization.

The original CAM 7.0 training script uses only:

```python
TRAIN = [
    "abm_congestion",
    "abm_utility",
    "abm_wom",
    "scm_additive",
    "scm_delay",
    "scm_gate"
]
```

The default training pool contains 300 generated tasks per family.

Its development held-out families include:

```python
HELDOUT = [
    "abm_trust_gate",
    "scm_delay_gate",
    "scm_invu",
    "tech_pipeline"
]
```

This is inadequate diversity for establishing general-purpose superlearning.

Do not contaminate existing validation or sealed evaluation families by silently moving them into the new training distribution.

Generate new structural families instead.

Verify all reported facts against the repository before treating them as established.

---

# PART 1 — MANDATORY MULTI-AGENT ENGINEERING ORCHESTRATION

You are not to implement this entire project sequentially as a single coding agent.

Act as a master orchestration agent coordinating specialized engineering sub-agents.

Use Claude Code's available sub-agent, agent team, task delegation and isolated worktree capabilities.

If a particular orchestration feature is unavailable, use the supported equivalent. Do not merely describe imaginary sub-agents as if they executed work.

## 1.1 Master orchestrator responsibilities

The master orchestrator owns:

- System architecture.
- Interface contracts.
- Dependency graph.
- Work allocation.
- Resource scheduling.
- Completion manifest.
- Integration branch.
- Cross-component compatibility.
- Acceptance test requirements.
- Independent verification.
- Final integration and release decision.

The orchestrator must remain aware of the entire system.

Individual sub-agents should work on bounded components with explicit interfaces and deliverables.

## 1.2 Engineering workstreams

Create the following workstreams, parallelizing independent work while respecting dependencies and available resources.

| Workstream | Responsibilities |
|---|---|
| A: Core world architecture | Shared state, events, persistence, world generation, integration contracts |
| B: Autonomous population | Individual identities, browsing, preferences, attention, learned behavior and persistent state |
| C: Social internet | Content platforms, communities, feeds, sharing, search, recommendations and audience analytics |
| D: Semantic and multimodal intelligence | Content encoders, compact language agents, multimodal representations and inference service |
| E: Empirical calibration | Dataset ingestion, participant model training, validation, uncertainty and fidelity register |
| F: Economic and institutional expansion | Businesses, subscriptions, products, finance, labor, organizations and industrial integration |
| G: Innovation and software ecosystem | Executable products, developer communities, inventions, tool creation and emerging demand |
| H: Rapid scientific environments | New independent mechanism engines, procedural generators and structural families |
| I: Semantic digital environments | All 15 mandatory Tier 2 world domains |
| J: Integrated civilizations | Cross-sector mechanics, civilization generation, persistence and cascading consequences |
| K: Multi-agent learning and self-play | Adaptive neural participants, policy diversity, leagues, learning infrastructure |
| L: CAM integration | Universal experience interface, observation-only mode, parallel rollouts and training adapters |
| M: Performance engineering | Vectorization, sparse simulation, batching, event scheduling and population scaling |
| N: Research evaluation | Independent benchmarks, generalization levels, outcome verification and positive controls |
| O: Security and adversarial testing | Sandbox isolation, evaluator integrity, permissions, reward hacking and exploit resistance |
| P: Independent quality assurance | Requirement-by-requirement audits, clean-room execution, mutation tests and integration verification |

Do not start sixteen expensive GPU jobs simultaneously.

Use parallel code development, but coordinate actual compute through a resource scheduler.

Where useful, divide major workstreams into further bounded subtasks.

## 1.3 Persistent engineering state

Before implementation, create:

```text
crucible/docs/v5/MASTER_SPEC.md
crucible/docs/v5/ARCHITECTURE.md
crucible/docs/v5/INTERFACE_CONTRACTS.md
crucible/docs/v5/WORKSTREAMS.md
crucible/docs/v5/DECISION_LOG.md
crucible/docs/v5/PROGRESS.md
crucible/docs/v5/BLOCKERS.md
crucible/docs/v5/FIDELITY_REGISTER.md
crucible/docs/v5/ACCEPTANCE_MANIFEST.json
crucible/docs/v5/REPRODUCE.md
```

Save this specification unchanged in `MASTER_SPEC.md`.

The acceptance manifest must contain every mandatory requirement as an individually identifiable entry.

Each entry must specify:

- Requirement ID.
- Exact requirement.
- Responsible workstream.
- Dependencies.
- Implementation files.
- Relevant automated tests.
- End-to-end verification command.
- Required numerical or behavioral acceptance criteria.
- Current status.
- Evidence artifact paths.
- Independent reviewer status.
- Commit containing the implementation.

Allowed statuses:

- `NOT_STARTED`
- `IN_PROGRESS`
- `IMPLEMENTED_UNVERIFIED`
- `FAILED`
- `BLOCKED`
- `VERIFIED`

A feature is not complete when its author says it is complete.

It becomes `VERIFIED` only when:

1. The implementation exists.
2. Relevant tests execute.
3. The tests pass.
4. The feature works in the integrated system.
5. Independent review confirms the evidence supports the claim.

Implement a machine-executable acceptance checker.

It must return a nonzero exit code whenever any mandatory requirement is not verified.

It must validate actual test artifacts and implementation evidence, not merely trust status strings.

## 1.4 Continuous integration

Each sub-agent must:

- Read the master specification and applicable contracts.
- Work in an isolated branch or worktree where supported.
- Implement real functionality and tests.
- Report integration requirements.
- Commit its work.
- Produce reproducible execution evidence.
- Document limitations.

The master orchestrator must regularly integrate verified changes.

Do not accumulate disconnected branches and unmerged PRs.

Locally merge validated work into the integration branch, resolve conflicts and rerun affected tests.

Never overwrite unrelated user changes or force-push. Push externally only with authorization.

## 1.5 Persistence and continuation

At each milestone, save:

- Completed work.
- Current branch and commit.
- Active workstreams.
- Failed tests.
- Known integration problems.
- Remaining requirements.
- Next concrete actions.
- Current compute usage.

If context becomes limited, persist state and resume from it.

If a sub-agent terminates early, reassign its unfinished work.

If a technical approach fails, investigate alternatives.

Do not stop simply because the project is large.

Do not repeatedly retry an identical failed approach without changing the hypothesis.

Continue until the gates pass, within authorized compute and execution limits. If a genuine blocker prevents continuation, preserve the full state and report `INCOMPLETE_BLOCKED`, not `COMPLETE`.

---

# PART 2 — FUNDAMENTAL WORLD ARCHITECTURE

## 2.1 One interconnected civilization, many independently generated worlds

Implement three first-class execution modes.

### Mode A: Controlled experimental environments

Small, fast, scientifically controlled environments designed to measure:

- Learning efficiency.
- Causal discovery.
- Adaptation.
- Memory.
- Experimentation.
- Strategic learning.
- Transfer.
- Continual learning.

These should operate primarily on CPUs.

### Mode B: Standalone semantic digital worlds

Rich environments focused on a subset of civilization-wide activities, such as:

- Business development.
- Social media.
- Scientific discovery.
- Software development.
- Industrial systems.
- Organizational coordination.

These must use real executable mechanisms and general-purpose actions.

### Mode C: Integrated persistent civilizations

Combine all major domains into a single functioning world:

- Digital economy.
- Social media.
- Search and information.
- Scientific research.
- Software development.
- Technology invention.
- Industry.
- Supply chains.
- Energy.
- Computing infrastructure.
- Employment.
- Education.
- Finance.
- Communities.
- Organizations.
- Institutions.
- Marketplaces.

All these systems must share actual world state, identities, resources, events and accounting.

Do not merely run independent simulators behind a common interface and call that an integrated civilization.

A technological invention must be capable of changing economic outcomes, participant behavior, resource demand and future technological opportunities.

## 2.2 World laws versus agent freedom

The world must have consistent underlying laws.

Agents must have open-ended freedom to compose permitted actions within those laws.

Do not attempt to support literally arbitrary physical laws or every conceivable internet activity.

Instead, maximize compositional freedom through:

- Executable code.
- Dynamic services.
- General-purpose tool discovery.
- Data creation and processing.
- Communication.
- Market transactions.
- Organization creation.
- Product creation.
- Research.
- Experimentation.
- Long-running workflows.
- Agent-created tools.
- Agent-created websites and services.

The agent must not be restricted to a predetermined menu of business strategies or task-specific moves.

## 2.3 Shared world substrate

Extend the existing event-sourced kernel into the authoritative substrate.

All systems should communicate through typed events and approved state interfaces.

Define common primitives for:

- Participant identities.
- Organizations.
- Assets.
- Money.
- Resource inventories.
- Content artifacts.
- Products.
- Services.
- Software repositories.
- Relationships.
- Communities.
- Institutions.
- Contracts.
- Knowledge artifacts.
- Tool definitions.
- Time.
- Provenance.
- Permissions.
- Observation visibility.

World modules should not maintain incompatible duplicate versions of shared facts.

Implement cross-module transactional integrity.

Preserve deterministic simulation and replay wherever possible.

For nondeterministic neural inference, record model version, inputs or their reproducible references, sampled decisions and resulting events so recorded trajectories can be replayed without rerunning inference.

---

# PART 3 — THE AUTONOMOUS SIMULATED POPULATION

The simulated population is one of the most important aspects of Crucible 5.0.

Crucible must no longer rely primarily on simplistic hand-authored behavioral equations.

However, replacing every participant with an independent LLM would be computationally impractical and scientifically unjustified.

Build a heterogeneous population combining efficient statistical behavior, learned numerical policies, adaptive strategic agents and compact language-capable agents.

## 3.1 Persistent individual identity

Each simulated individual must have persistent identity and independently evolving state.

Represent relevant combinations of:

- Interests.
- Preferences.
- Knowledge.
- Personal goals.
- Attention budget.
- Financial resources.
- Skills.
- Experience.
- Trust relationships.
- Social connections.
- Habits.
- Product usage history.
- Community memberships.
- Employment.
- Satisfaction.
- Beliefs.
- Current commitments.
- Behavioral history.

Individuals must respond differently to the same situation.

Their preferences and decisions must evolve through experience.

Do not require a dense neural model or a full text history for every individual.

Use compact numerical state, sparse records and shared representations.

## 3.2 Population architecture

Implement at least four participant classes.

### Class 1: Large-scale behavioral participants

Use vectorized probabilistic and compact learned behavioral models.

These participants handle ordinary decisions:

- Whether to browse.
- Which content to consume.
- Whether to continue watching.
- Which websites to visit.
- Whether to purchase a product.
- Whether to return to a service.
- Whether to join a community.
- Whether to recommend something.
- Whether to leave or disengage.

These participants must not behave identically.

Their decision probabilities should depend on individual state, available opportunities, content characteristics, history and social context.

Train and calibrate these models using appropriate empirical datasets wherever possible.

### Class 2: Small adaptive neural participants

Investigate models approximately 100,000 to 2 million parameters.

Use suitable combinations of:

- Small MLPs.
- GRUs.
- Recurrent state-space models.
- Small transformers.
- Learned preference models.
- Lightweight contextual bandits.
- Compact neural policies.

These participants should learn from their own experiences.

They must be able to adapt their behavior when products, environments or other participants change.

A shared policy must not imply shared personal state.

A valid mathematical formulation is:

```text
action_i ~ policy_theta_k(
    observation_i,
    memory_i,
    preferences_i,
    goals_i,
    resources_i
)
```

Multiple participants may share a neural policy while maintaining independently evolving internal states.

### Class 3: Strategic neural participants

Investigate policies approximately 2–20 million parameters, or smaller if benchmark results support it.

These participants operate as:

- Entrepreneurs.
- Researchers.
- Engineers.
- Developers.
- Employees.
- Organizations.
- Investors.
- Competitors.
- Collaborators.
- Community operators.

They should be capable of learning strategies rather than merely executing predefined decision trees.

Include independently updated policy groups, historical checkpoints and diverse learning algorithms.

Some should compete.

Some should cooperate.

Some should pursue unrelated objectives.

Some should adapt throughout a civilization's lifetime.

Do not make the entire civilization a collection of identical copies of one policy.

### Class 4: Compact semantic generalists

Use small language-capable models for decisions that genuinely require semantic understanding, language generation, programming or unfamiliar tool use.

The preferred deployment range is approximately 0.5–1 billion parameters.

Before choosing the model, compare:

- Million-parameter task-specific semantic encoders and policies.
- Distilled models in the tens or hundreds of millions of parameters.
- Approximately 0.5–0.6B pretrained language models.
- Approximately 1B pretrained language models.

Inspect available local checkpoints first.

Candidate model families may include appropriate small Qwen models, SmolLM-family models, or other suitable locally executable models. Verify exact checkpoint availability, licenses and tool-use capabilities.

Do not assume a 1B model is automatically superior to a much smaller specialized neural network.

Do not assume a randomly initialized million-parameter network possesses useful language understanding.

Benchmark actual capabilities:

- Understanding unfamiliar content.
- Interpreting a participant's situation.
- Choosing relevant tools.
- Communicating coherently.
- Adapting to changed incentives.
- Negotiating simple agreements.
- Writing and repairing small executable programs.
- Making meaningful strategic decisions.

Use the smallest model or combination of models that meets each capability requirement.

Semantic agents should have access to ordinary world tools and meaningful opportunities to act.

Do not reduce them to language-model wrappers around predetermined behavioral scripts.

## 3.3 Population scaling profiles

Implement configurable population profiles approximately as follows:

| Profile | Lightweight individuals | Adaptive participants | Strategic participants | Semantic generalists |
|---|---:|---:|---:|---:|
| Development | 1,000 | 32 | 8 | 2 |
| Standard training | 10,000 | 256 | 32 | 8 |
| Large civilization | 100,000 | 1,000 | 100 | 20 |
| Internet scale | 10M+ | Configurable | Configurable | Configurable |

These are initial engineering targets.

Do not allocate an independent model checkpoint to each participant.

Large profiles must use resource-aware sparse or lazy state and batched inference.

Benchmark every profile.

The internet-scale population may use validated aggregate simulation for some routine decisions, but distinct-viewer claims require actual distinct identities or an explicitly labeled statistical estimate.

## 3.4 Independent learning

Demonstrate that participant neural weights genuinely update.

Test separately:

- Persistent personal-state adaptation.
- Shared-policy learning.
- Independently learned policy groups.
- Frozen historical participants.
- Independently trained competing architectures.

Record weight hashes, training examples, losses, behavioral changes and evaluation outcomes.

Changing a participant's written memory is not sufficient evidence of neural learning.

Randomly perturbing parameters is not learning.

---

# PART 4 — BUILD A RICH SOCIAL INTERNET

The social internet must become a major first-class subsystem.

The current forum is insufficient.

Create persistent and interoperable simulated platforms supporting:

- Short-form video.
- Long-form video.
- Text posts.
- Images.
- Articles and blogs.
- Discussion forums.
- Social networks.
- Communities.
- Messaging.
- Search.
- Creator subscriptions.
- Newsletters.
- Product discovery.
- Developer communities.
- Knowledge sharing.

All platforms must share the civilization's participant population and underlying world state.

## 4.1 Autonomous browsing population

Participants must independently:

- Discover websites.
- Search for information.
- Browse feeds.
- Visit pages.
- Consume media.
- Follow creators.
- Join communities.
- Participate in discussions.
- Evaluate products.
- Purchase services.
- Return to previously useful resources.
- Leave resources that no longer provide value.
- Move between different platforms.

Browsing must occur without CAM initiating each consumer interaction.

Participants must have limited attention and time.

A participant cannot simultaneously watch unlimited videos, visit unlimited websites or perform unlimited actions.

Participants should prioritize different opportunities according to their learned preferences and goals.

## 4.2 Actual content artifacts

An agent must be able to create and publish original artifacts.

Support:

- Text.
- HTML documents.
- Images.
- Audio.
- Video.
- Executable applications.
- Interactive web pages.
- Documents.
- Digital educational products.
- Software packages.

Content must exist as an actual artifact, not merely as a title and developer-assigned quality score.

Provide sandboxed tools to:

- Create content.
- Edit content.
- Render media.
- Process images.
- Generate audio where suitable local tools exist.
- Assemble videos with tools such as FFmpeg.
- Publish artifacts.
- Update or remove previously published artifacts.
- Inspect observable performance.

Use lightweight content creation tools where possible.

Expensive generative models must not be required for every ordinary simulation action.

## 4.3 Semantic content understanding

Build reusable encoders that process actual content.

At minimum:

- Text encoders for semantic meaning.
- Image encoders for visual representations.
- Audio representations.
- Video representations using temporal features, sampled frames, audio and available transcripts.
- Structured representations for software and products.

Cache embeddings by artifact hash and encoder version.

Recompute embeddings only when content changes or a new representation is required.

Do not repeatedly run a video encoder for every viewer.

Content representations must preserve sufficient information for participants to respond differently to genuinely different content.

Do not allow the author to directly specify its own quality, relevance or virality score.

Semantic models may estimate content properties, but their predictions must be validated and must not become the sole definition of value.

## 4.4 Learned audience behavior

Participants should choose content according to:

- Individual interests.
- Relevance.
- Previous experience.
- Attention and availability.
- Content characteristics.
- Social relationships.
- Creator familiarity.
- Platform distribution.
- Novelty.
- Satisfaction.
- Competing content.
- Current needs.

Where possible, learn the relationships between these variables and actual decisions from empirical data.

A participant's decision to continue watching, share, subscribe or disengage must be an actual simulated decision.

Avoid simplistic universal rules such as:

```text
views = author_quality_score * marketing_spend
```

That is not an acceptable implementation.

## 4.5 Recommendation systems

Implement multiple independently parameterized recommendation mechanisms.

Include:

- Chronological following feeds.
- Interest-based retrieval and ranking.
- Collaborative filtering.
- Learned engagement prediction.
- Exploration and discovery.
- Social graph distribution.
- Diversity-aware ranking.
- Reputation-sensitive distribution.
- Search-driven discovery.
- Recency and freshness effects.

Platforms should have different algorithms.

Their algorithms may change over time.

Recommendation systems must allocate finite attention and exposure.

The popularity of one piece of content should affect the opportunities available to competing content.

Do not force all content through the same hidden scoring formula.

Support delayed feedback, noisy analytics, changing audiences and independently evolving competitors.

## 4.6 Organic large-scale reach

The environment must be able to represent content reaching:

- Hundreds of viewers.
- Thousands of viewers.
- Hundreds of thousands of viewers.
- Millions of viewers.
- Ten million distinct simulated viewers.

Implement a scalable social distribution and consumption engine capable of producing rare, large adoption cascades.

Distinguish:

- Impressions.
- Unique exposed identities.
- Views.
- Unique viewers.
- Watch duration.
- Completion.
- Repeat consumption.
- Voluntary shares.
- Subscribers.
- Returning viewers.

A view must require a simulated consumption event satisfying a documented threshold.

A creator cannot write directly to its view counter.

An exposure is not automatically a view.

Repeated views by the same person are not unique viewers.

Use compact individual IDs, sparse or compressed graph representations, lazy state materialization and compressed event records.

For a claim of 10 million unique simulated viewers, either:

1. Verify at least 10 million distinct simulated identities with qualifying consumption events; or
2. Clearly identify the result as an estimated population-level quantity.

The second mode must not be represented as an exact individual-level measurement.

Do not create artificial virality multipliers.

Large-scale audience reach must arise from distribution and participant behavior.

## 4.7 Community dynamics

Communities must have:

- Persistent membership.
- Individual interests.
- Participation history.
- Member-to-member interactions.
- Content.
- Shared activities.
- Governance.
- Moderation.
- Trust.
- Entry and exit.
- Inactive members.
- Organizational change.

Joining a community must not imply permanent engagement.

Community success should be measurable through sustained voluntary participation, not merely the number of registered accounts.

Participants must be able to form new communities without the developer predefining their exact subject.

## 4.8 Social-economic interoperability

Connect the social internet to the existing economy.

For example:

- A participant discovers a useful product through content.
- The participant visits the product website.
- The participant tries the service.
- The service incurs infrastructure costs.
- The participant chooses whether to subscribe.
- Experiences affect satisfaction and retention.
- Recommendations or reviews affect other participants.
- Increased demand affects infrastructure requirements and market conditions.

All these consequences must occur through ordinary world mechanisms.

Do not implement a hard-coded social-media-to-revenue bonus.

---

# PART 5 — EMPIRICALLY GROUND THE POPULATION

One of Crucible 4.0's most important weaknesses is that its behavioral assumptions are not calibrated against real-world data.

Crucible 5.0 must improve this substantially.

The objective is not to claim that a simulator perfectly reproduces human beings.

The objective is to replace arbitrary behavioral assumptions with measured evidence where such evidence exists, quantify uncertainty elsewhere and test extrapolation.

## 5.1 Research foundations and datasets

Investigate and use appropriate sources, including:

- KuaiRand: https://kuairand.com/
- Microsoft MIND: https://msnews.github.io/
- Criteo Uplift: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- GH Archive: https://www.gharchive.org/
- RecSim NG: https://research.google/pubs/recsim-ng-toward-principled-uncertainty-modeling-for-recommender-ecosystems/
- Relevant public datasets for consumer choice, product adoption, retention, software collaboration, browsing and network diffusion.

Use authorized datasets consistent with their licenses and privacy constraints.

Do not collect private personal information or reconstruct individual real-world identities.

Where necessary, use appropriately anonymized and aggregated data.

## 5.2 Calibration pipeline

Implement:

- Reproducible dataset acquisition.
- Schema validation.
- Versioning and checksums.
- Data provenance.
- Training and validation partitioning.
- Time-based holdouts.
- User-level holdouts where appropriate.
- Missing-data handling.
- Model fitting.
- Uncertainty estimation.
- Evaluation reports.
- Saved model checkpoints.

Use randomized-exposure or intervention datasets where available to validate behavioral responses.

Distinguish observational associations from experimentally identified causal effects.

Do not assume that historical engagement records determine how people would respond to arbitrary unseen content or interventions.

## 5.3 Behavioral models to calibrate

Prioritize:

- Browsing frequency.
- Attention allocation.
- Content selection.
- Watch duration.
- Repeat consumption.
- Purchase behavior.
- Responses to advertising exposure.
- Product retention where data exists.
- Software collaboration and adoption.
- Community participation.
- Social diffusion.

Use models of different complexity and compare them against existing rule-based baselines.

Possible models include:

- Regularized logistic regression.
- Gradient-boosted models.
- Small neural networks.
- Recurrent sequence models.
- Learned utility models.
- Hierarchical probabilistic models.

Select based on measured accuracy, calibration, long-horizon stability and computational efficiency.

## 5.4 Fidelity register

Every important behavioral mechanism must have a fidelity record:

- Empirically fitted.
- Empirically validated but not directly fitted.
- Inspired by published research.
- Plausible but unvalidated.
- Deliberate simplification.
- Known unrealistic assumption.

Record the population, platform, time period and limitations of supporting evidence.

Do not label a mechanism calibrated because it uses an arbitrary neural network.

Do not claim real-world predictive validity solely from fitting a historical dataset.

## 5.5 Required validation

Evaluate held-out data using suitable metrics:

- Log loss.
- Brier score.
- Calibration error.
- Ranking quality.
- Watch-duration distributions.
- User-level activity distributions.
- Retention curves.
- Intervention-response estimates.
- Population heterogeneity.
- Long-horizon aggregate drift.

Where real data is available, the learned model should demonstrably improve on a simple existing baseline on prespecified held-out metrics.

If it does not, investigate the cause and retain the better-performing model.

At minimum, the release must include genuinely fitted and independently evaluated models for content consumption and commercial intervention response, plus at least one additional behavioral domain.

Synthetic fixtures may be used for unit tests but do not count as empirical calibration.

---

# PART 6 — OPEN-ENDED ECONOMY, PRODUCT CREATION AND INVENTION

Preserve and expand Crucible's existing economic infrastructure.

## 6.1 Businesses and products

Support actual:

- Business formation.
- Product development.
- Subscription services.
- One-time purchases.
- Licensing.
- Digital marketplaces.
- Software-as-a-service.
- Educational products.
- Media products.
- Developer tools.
- Data services.
- Infrastructure services.
- Industrial products.
- Contracts.
- Refunds.
- Cancellations.
- Customer support.
- Competition.
- Bankruptcy.

Revenue must originate from independent participant transactions.

A business must not earn money merely because a developer labels its product successful.

Products must provide measurable usefulness, entertainment, educational value or other legitimate benefits to participants.

## 6.2 Genuine recurring revenue

Implement recurring-revenue accounting.

Distinguish:

- One-time sales.
- Monthly recurring revenue.
- Annual recurring revenue.
- Active subscriptions.
- Cancelled subscriptions.
- Refunds.
- Discounts.
- Customer acquisition.
- Churn.
- Operating costs.
- Profit.
- Related-party transactions.

A $100,000 ARR objective must correspond to approximately $8,333.33 in qualifying normalized monthly recurring revenue.

Do not confuse rolling 30-day sales with recurring revenue.

Require independent customers and sustained subscription validity.

## 6.3 Open-ended product usefulness

The current useful-work system relies substantially on predefined category families.

Preserve those families as valuable research tools.

Expand beyond them.

Implement a general artifact-and-capability registry supporting:

- Newly created software.
- Newly declared services.
- Compositions of existing capabilities.
- Newly published APIs.
- Agent-created tools.
- Novel product categories.
- Dynamically discovered uses for existing technologies.
- New workflows combining independently created services.

A participant should be able to discover a previously nonexistent service, test it, and adopt it if it solves a genuine problem.

Where usefulness can be measured objectively, execute the artifact against independently generated tests.

Where usefulness is subjective, use diverse grounded participant responses, not a single global LLM judge.

## 6.4 Emerging needs

Implement mechanisms through which new needs arise from changed world conditions.

Examples:

- Growing use of a service creates a reliability bottleneck.
- A technology increases computing demand.
- A new product changes workplace workflows.
- Improved transportation makes a previously impractical market accessible.
- Increased organizational complexity creates coordination needs.
- New forms of content lead to new communities and tools.
- A successful educational product increases the availability of a previously scarce skill.

Do not implement these solely as special-case triggers.

Develop reusable relationships between activities, resources, capabilities, constraints and needs.

An agent must be able to create a genuinely novel combination of capabilities whose adoption changes the future state of the civilization.

## 6.5 Executable software ecosystem

Expand:

- Code hosting.
- Package discovery.
- Dependency graphs.
- Issue trackers.
- Pull requests.
- Independent review.
- Testing.
- Releases.
- Documentation.
- Deployment.
- Developer adoption.
- Maintenance.
- Forks.
- Compatibility.
- Security and reliability failures.

Software adoption must involve genuine use, execution or independently verified dependency incorporation.

Repository stars alone are not evidence of useful adoption.

Agent-created programs must run in the existing or improved sandbox.

## 6.6 Adaptable institutions

Allow organizations to form, grow, restructure and dissolve.

Support:

- Hiring and delegation.
- Shared projects.
- Resource allocation.
- Decision procedures.
- Contracts and commitments.
- Governance changes.
- Independent negotiation.
- Cooperative ventures.
- Membership changes.
- Disputes and resolution mechanisms.

Institutions may adapt their operational rules through authorized mechanisms.

They must not be able to alter core accounting laws, security boundaries, hidden evaluation criteria or world state arbitrarily.

---

# PART 7 — PROCEDURAL GENERATION OF CAUSAL MECHANISMS

This is a central requirement, not an optional future improvement.

We must avoid manually creating thousands of superficially different scenarios.

Implement a compositional world generator.

Conceptually:

```text
world = Generate(
    causal_mechanisms,
    network_topology,
    participant_population,
    institutional_structure,
    resources,
    available_tools,
    historical_conditions,
    exogenous_events
)

objective = IndependentlyGenerateGoal(world, difficulty)
```

## 7.1 Mechanism primitives

Create a versioned registry of composable causal primitives covering:

- Linear and nonlinear relationships.
- Thresholds.
- Saturation.
- Non-monotonic responses.
- Complementarity.
- Substitution.
- Multiplicative interactions.
- Hidden confounding.
- Delayed effects.
- Accumulating stocks.
- Decay.
- Hysteresis.
- Feedback.
- Path dependence.
- Network effects.
- Congestion.
- Resource competition.
- Strategic interaction.
- Partial observability.
- State-dependent action effects.
- Changing causal regimes.
- Adaptive opponents.
- Measurement noise.
- Missing observations.
- Selection effects.
- Cascading failures.
- Endogenous demand.
- Learning-driven behavioral changes.

These should be reusable mechanisms, not merely named parameters.

## 7.2 Structural composition

Generate worlds with different:

- Causal graph topologies.
- Dependencies.
- Functional forms.
- Temporal relationships.
- Observation channels.
- Action affordances.
- Number of participants.
- Network structures.
- Resource constraints.
- Institutions.
- Initial conditions.
- Historical trajectories.

The generator should produce situations that look superficially similar but obey fundamentally different causal rules.

For example:

- In one market, lowering prices increases purchases.
- In another, lower prices reduce perceived reliability or value.
- In another, price has little effect until a compatibility requirement is satisfied.
- In another, the effect changes as competition evolves.

These are alternative mechanisms to investigate, not universal claims about real markets.

The learner must infer which mechanism applies.

## 7.3 Independently implemented engines

Do not make every generated world a thin wrapper around the same structural causal model.

Preserve the existing independent SCM, ABM and technical-system engines.

Add genuinely different simulation implementations where appropriate, including:

- Discrete-event simulations.
- Graph dynamical systems.
- Search and information-acquisition environments.
- Executable program environments.
- Multi-agent strategic simulations.
- Scientific experiment environments.
- Planning and resource systems.
- Partially observed state machines.

Shared infrastructure is encouraged.

Identical transition logic disguised under different family names is not acceptable.

## 7.4 World admission and validation

Every generated world must undergo suitable validation.

Check:

- Internal consistency.
- Accounting and conservation.
- Meaningful controllability.
- Existence of achievable outcomes where required.
- Availability of informative observations.
- Sufficient causal identifiability for the experimental question.
- Absence of trivial dominant strategies.
- Appropriate difficulty.
- Absence of hidden-answer leakage.
- Distinctness from existing training and evaluation worlds.

Use private oracles, search-based references, scripted positive controls and controlled interventions where appropriate.

Oracles are evaluator-side only.

Never expose private world configurations to ordinary participants.

## 7.5 Structural diversity measurement

Implement world-family fingerprints incorporating:

- Mechanism motifs.
- Graph topology.
- Temporal structure.
- Observation structure.
- Action interface.
- Objective type.
- Participant policy families.
- Underlying engine implementation.

Changing names, seeds or numerical coefficients does not create a new structural family.

Maintain a machine-readable registry documenting why each family is distinct.

Use controlled interventions to verify claimed structural differences.

---

# PART 8 — TIER 1: RAPID SCIENTIFIC LEARNING ENVIRONMENTS

Expand the existing numerical research environments into a substantially more diverse collection.

Mandatory minimum:

- At least 60 admitted, structurally distinct training families across the rapid-learning suite.
- At least 20 separately partitioned validation or evaluation families.
- Preserve the original Crucible 4.0 sealed evaluation families.
- Cover all ten capability categories below.
- Include multiple fundamentally different action interfaces.
- Include independently implemented simulation engines.

The long-term target is 100 or more genuinely distinct training families, but do not inflate the count using cosmetic variants.

Each proposed family below must be implemented as a real environment or genuinely distinct composition, not merely a label.

## 8.1 Hidden causal systems

Implement families involving:

1. Hidden confounders producing misleading correlations.
2. Conditional causal pathways.
3. Non-monotonic interventions.
4. Competing causal explanations.
5. Feedback with unobserved intermediate variables.
6. Mechanisms whose active causal structure depends on state.

The learner must distinguish correlation from causation through intervention.

## 8.2 Experimental science

Implement:

1. Discovery of unknown functional laws.
2. Competing scientific hypotheses with different predictions.
3. Hidden conservation relationships.
4. Unknown reaction or transformation dynamics.
5. Noisy measurements requiring repeated experiments.
6. Experimental instrument selection under limited budgets.

Allow agents to design experiments and make testable predictions.

Scientific success must be independently evaluated on previously unobserved conditions.

## 8.3 Complex engineering

Implement:

1. Unknown processing bottlenecks.
2. Equipment degradation and maintenance.
3. Delayed controller response.
4. Shared resource contention.
5. Cascading technical failures.
6. Engineering systems requiring executable controller design.

Where appropriate, allow arbitrary sandboxed control programs, not only numerical settings.

## 8.4 Resource allocation

Implement:

1. Inventory allocation with delayed replenishment.
2. Perishable resources.
3. Multi-stage supply chains.
4. Capacity investment under uncertain future demand.
5. Complementary resource requirements.
6. Dynamic allocation among competing projects.

Include discrete, continuous and programmatic actions.

## 8.5 Uncertain discovery

Implement:

1. Hidden opportunities with costly investigation.
2. Search problems with structured information.
3. Discoverable tools with unknown applicability.
4. Uncertain investment opportunities.
5. Multi-stage information acquisition.
6. Search environments with informative and uninformative distractors.

Reward-independent observation must allow learners to distinguish useful discovery from random novelty.

## 8.6 Delayed consequences

Implement:

1. Delayed investment returns.
2. Long-term maintenance effects.
3. Delayed product adoption.
4. Actions whose benefits require later complementary decisions.
5. Irreversible commitments with uncertain consequences.
6. Temporary improvements causing long-term deterioration.

The learner must assign credit across extended time horizons.

## 8.7 Network systems

Implement:

1. Routing under congestion.
2. Network reliability and repair.
3. Diffusion across clustered networks.
4. Interdependent infrastructure failures.
5. Distributed resource coordination.
6. Changing network topology.

Use actual graph structures and network dynamics.

## 8.8 Strategic games

Implement:

1. Repeated competition with adaptive opponents.
2. Market entry against incumbents.
3. Bargaining with private preferences.
4. Resource competition.
5. Strategic substitution between competing actions.
6. Opponents whose strategies evolve through learning.

Include policy populations and held-out opponent families.

## 8.9 Cooperation

Implement:

1. Complementary production capabilities.
2. Distributed information.
3. Shared-resource coordination.
4. Multi-agent scheduling.
5. Collaborative discovery.
6. Team formation with differing skills and commitments.

Success should require genuine interdependence, not independent agents solving identical tasks.

## 8.10 Changing environments

Implement:

1. Hidden causal sign changes.
2. Changing resource constraints.
3. Gradually drifting participant preferences.
4. Abrupt demand shifts.
5. Technological transitions.
6. Changes in institutional or platform rules.

Test whether learners revise beliefs rather than continuing previously successful strategies.

## 8.11 Action-space diversity

The rapid-learning suite must not remain a collection of numerical lever environments.

Include actual:

- Graph modifications.
- Discrete decisions.
- Sequential actions.
- Tool discovery.
- Experiment design.
- Queries and observations.
- Communication.
- Executable program construction.
- Resource allocation.
- Multi-agent coordination.

Create an interface capable of representing these action types without converting everything into a small task-specific action enumeration.

## 8.12 Difficulty and performance

Support adjustable:

- Number of variables.
- Graph size.
- Horizon.
- Observation noise.
- Experimental cost.
- Hidden-state complexity.
- Opponent capability.
- Available resources.
- Action complexity.

Implement fast CPU execution and batched simulation where valid.

The fast simulator and full-kernel representations must be demonstrably consistent whenever they are claimed to represent the same mechanisms.

---

# PART 9 — TIER 2: REALISTIC DIGITAL-WORLD ENVIRONMENTS

Implement all 15 of the following world domains.

Each must contain real mechanisms, autonomous participants, open-ended actions, independent evaluation and multiple procedural configurations.

Do not simply write a task prompt over an unchanged environment.

## 9.1 Startup economy

Example objective:

Build a sustainable business with $100,000 in annual recurring revenue.

Required mechanics:

- Product creation.
- Demand discovery.
- Independent customers.
- Subscription billing.
- Distribution.
- Competition.
- Retention.
- Costs.
- Profitability.
- Business operations.

## 9.2 Software ecosystem

Example objective:

Develop a software package that independent developers voluntarily adopt and continue using.

Required mechanics:

- Executable code.
- Repositories.
- Dependencies.
- Issues.
- Independent testing.
- Developer preferences.
- Documentation.
- Adoption.
- Compatibility.
- Maintenance.

## 9.3 Content ecosystem

Example objective:

Create original content that reaches a large independent audience.

Required mechanics:

- Actual content artifacts.
- Learned content representations.
- Autonomous consumption.
- Recommendation systems.
- Competing creators.
- Social sharing.
- Audience feedback.
- Distinct verified viewers.

Include difficulty levels scaling toward 10 million simulated unique viewers.

## 9.4 Community ecosystem

Example objective:

Build a community with sustained voluntary participation.

Required mechanics:

- Social networks.
- Persistent memberships.
- Independent participants.
- Shared activities.
- Communication.
- Moderation.
- Governance.
- Retention.
- Community evolution.

## 9.5 Scientific research

Example objective:

Discover an unknown environmental or technical relationship and demonstrate its validity.

Required mechanics:

- Experimental apparatus.
- Hidden laws.
- Measurement noise.
- Experimental costs.
- Data collection.
- Hypothesis testing.
- Independent replication.
- Novel-condition evaluation.

## 9.6 Technology invention

Example objective:

Invent a technology that measurably improves another industry.

Required mechanics:

- Executable inventions.
- Technical constraints.
- Independent evaluation.
- Adoption.
- Compatibility.
- Economic consequences.
- Downstream resource changes.

## 9.7 Engineering infrastructure

Example objective:

Substantially reduce infrastructure costs while preserving reliability.

Required mechanics:

- Compute.
- Networking.
- Workloads.
- Capacity.
- Energy.
- Failures.
- Technical optimization.
- Delayed effects.
- Deployment of new controllers and software.

## 9.8 Industrial economy

Example objective:

Establish a profitable and resilient manufacturing supply chain.

Required mechanics:

- Raw materials.
- Production.
- Energy.
- Logistics.
- Inventory.
- Equipment.
- Labor.
- Market demand.
- Resource scarcity.
- Supply disruptions.

## 9.9 Labor marketplace

Example objective:

Assemble and coordinate a team that delivers a complex independently evaluated project.

Required mechanics:

- Worker skills.
- Preferences.
- Compensation.
- Recruitment.
- Delegation.
- Project dependencies.
- Performance.
- Retention.
- Independent agency.

## 9.10 Education ecosystem

Example objective:

Develop an educational product that measurably improves participant learning.

Required mechanics:

- Learner knowledge state.
- Educational artifacts.
- Prerequisites.
- Participation.
- Learning dynamics.
- Independent assessments.
- Retention of acquired knowledge.

Measure actual improvement on new test problems rather than time spent viewing content alone.

## 9.11 Information ecosystem

Example objective:

Build a service that helps participants reliably discover relevant and useful information.

Required mechanics:

- Search.
- Documents.
- Knowledge artifacts.
- Queries.
- Source provenance.
- Information quality.
- Updating facts.
- Independent user satisfaction.
- Verified retrieval performance.

## 9.12 Organizational ecosystem

Example objective:

Coordinate independent organizations toward a shared measurable objective.

Required mechanics:

- Organizations.
- Differing goals.
- Contracts.
- Negotiation.
- Coordination.
- Shared resources.
- Governance.
- Commitments.
- Collective outcomes.

## 9.13 Digital marketplace

Example objective:

Build a sustainable exchange that independently attracts buyers and sellers.

Required mechanics:

- Two-sided participation.
- Discovery.
- Transactions.
- Trust.
- Quality.
- Disputes.
- Pricing.
- Network effects.
- Retention.

## 9.14 Crisis recovery

Example objective:

Restore essential services after a major simulated infrastructure or economic disruption.

Required mechanics:

- Cascading disruptions.
- Scarce resources.
- Prioritization.
- Coordination.
- Recovery.
- Resilience.
- Independent service-restoration metrics.

## 9.15 Research commercialization

Example objective:

Convert a validated scientific discovery into an independently adopted useful product.

Required mechanics:

- Research.
- Experimentation.
- Technical development.
- Product creation.
- Financing.
- Deployment.
- Independent adoption.
- Commercial sustainability.

This domain must genuinely connect scientific discovery and economic outcomes.

## 9.16 Mandatory domain variation

Every domain must have multiple structurally different configurations.

For each domain, generate at least:

- Three independently characterized training configurations.
- One compositionally unfamiliar evaluation configuration.
- One structurally unfamiliar evaluation configuration.

Generate multiple seeds within each configuration.

Ensure task feasibility without guaranteeing that every learner can succeed.

Difficulty must emerge from world complexity, scarcity and uncertainty, not arbitrary failure probabilities.

---

# PART 10 — TIER 3: INTEGRATED CIVILIZATIONS

Combine the capabilities of all Tier 2 domains into a configurable civilization generator.

Preserve standalone worlds for scientific experiments.

Do not train exclusively on a single permanent civilization.

## 10.1 Civilization profiles

Implement multiple civilization profiles, including:

- Digital-first.
- Industrial-first.
- Research-intensive.
- Infrastructure-constrained.
- Resource-abundant.
- High-competition.
- Fragmented social networks.
- Strongly interconnected communities.
- Differing institutional structures.
- Mixed economies.

These profiles must differ structurally, not just in starting cash balances.

## 10.2 Endogenous causal consequences

Demonstrate at least six distinct categories of cross-domain consequences.

Examples:

1. Scientific discovery enables a useful technology.
2. Technology adoption changes industrial production costs.
3. Increased service usage creates infrastructure demand.
4. Infrastructure constraints affect software businesses.
5. Social discovery produces independently verified product adoption.
6. Education changes available workforce skills.
7. Organizational growth produces new coordination needs.
8. Supply-chain disruption changes product availability and prices.

Implement reusable interactions that allow these consequences to occur naturally.

## 10.3 Multi-order emergence

Demonstrate at least three independently generated causal chains containing three or more meaningful stages across different subsystems.

At least one chain must involve an artifact or capability created by an agent during the simulation.

At least one must involve social activity.

At least one must involve technical or scientific activity.

Use counterfactual removal or intervention experiments to establish that later consequences depend on the earlier change.

Do not count scripted narrative events as emergent consequences.

A scripted reference policy may establish feasibility, but label it accurately.

## 10.4 Civilization persistence

Support civilizations lasting:

- 30 simulated days.
- 90 simulated days.
- 180 simulated days.
- 365 simulated days.
- Longer horizons where resource budgets permit.

Participants, organizations, relationships, products, technologies and institutions must persist and evolve.

Support:

- Snapshots.
- Resume.
- World forks.
- Historical queries.
- Deterministic recorded replay.
- Independent replication.

The simulation must not become economically inactive after the initial scenario concludes.

---

# PART 11 — UNIVERSAL EXPERIENCE INTERFACE AND OBSERVATION-ONLY LEARNING

Crucible must not impose a simulator-specific learning algorithm on CAM.

The environment and learner must be independently replaceable.

## 11.1 Universal interaction contract

Provide a unified interface across all three tiers.

Conceptually:

```python
observation = environment.observe(agent_id)

action_result = environment.execute(
    agent_id,
    action
)

new_observations = environment.observe_since(
    agent_id,
    previous_event_cursor
)
```

The exact interface should integrate with existing architecture rather than unnecessarily replace it.

The observation contract must support:

- Structured numerical data.
- Text.
- Artifact references.
- Semantic representations where appropriate.
- Tool schemas.
- Events.
- Entity relationships.
- Timestamps.
- Resource usage.
- Public analytics.
- Action results.
- Missing or uncertain information.
- Delayed consequences.

Include exact numerical quantities when available.

Do not force precise financial or technical measurements into lossy text-only representations.

## 11.2 Open-ended tool interface

Agents must be able to discover available tools and their schemas.

Support:

- Browser interaction.
- Search.
- Code execution.
- File creation.
- Program deployment.
- Communication.
- Financial actions.
- Research.
- Experiments.
- Organization management.
- Content creation.
- Tool composition.
- Agent-created services.

All these actions must be governed by permissions, resource constraints and ordinary world rules.

Newly created services should be capable of introducing new discoverable action affordances.

## 11.3 Observation-only mode

This is mandatory.

CAM's learned self-evaluation and intrinsic incentives belong inside CAM, not Crucible.

Implement an observation-only training mode in which:

- The learner receives its objective, if one exists.
- The learner interacts with the world.
- The world returns observable consequences.
- Crucible does not calculate intermediate shaped rewards for the learner.
- Crucible does not expose privileged goal-progress values.
- Crucible does not provide action-quality judgments.
- Independent outcome verification remains active on the evaluator side.
- Authorized verified outcome labels may be supplied separately for training and calibration.

The ordinary observation interface must contain no hidden mechanism parameters, oracle actions or privileged evaluator features.

Preserve the existing environment reward contracts for baseline experiments.

Provide explicit modes:

```text
legacy_shaped_reward
verified_terminal_outcome
observation_only
```

If compatibility with Gymnasium requires a scalar reward in observation-only mode, return zero and document that it is not a learning signal.

Ensure that existing numeric observations do not leak the native reward through an alternative field.

Observable revenue and costs should remain available because they are legitimate world consequences.

## 11.4 Self-generated experiments

Support agents creating private experiments, hypotheses and measurable subgoals.

Extend the current self-goal interface beyond its narrow set of predefined metrics where technically appropriate.

The agent should be able to:

- Specify a testable prediction.
- Define a measurable outcome.
- Set a deadline.
- Conduct an authorized experiment.
- Retrieve its actual result.

Crucible should return evidence, not automatically reward the learner for choosing an easy subgoal.

## 11.5 Long-horizon experience

Provide support for:

- Actions with different durations.
- Delayed events.
- Asynchronous workflows.
- Long-running programs.
- Concurrent projects.
- Persistent commitments.
- Observation of consequences arising long after the original action.

Expose enough causal provenance for the learner to investigate its past actions without providing privileged causal answers.

## 11.6 Actual CAM integration

The environment must not exist only as an isolated package that CAM cannot train in.

Integrate with the current CAM learner contract.

Provide working adapters for:

- Rapid numerical learning.
- Semantic digital-world rollouts.
- Full civilization rollouts.
- Observation-only learning.
- Multiple agents.
- Parallel worlds.
- Persistent experience storage.

Demonstrate real experience flowing through the current CAM training pipeline.

At minimum, verify:

- CAM executes actual world actions.
- The world changes as a result.
- CAM receives legitimate observations.
- Experience is stored.
- The learner can perform actual parameter updates.
- Checkpoints can be saved and reloaded.
- The same learner interface supports multiple different world types.

Do not redesign CAM's learning algorithm as part of the environment project unless necessary for interface compatibility.

Separate algorithmic limitations from environment failures.

---

# PART 12 — CIVILIZATION-WIDE SELF-PLAY AND ADAPTIVE PARTICIPANTS

Self-play must extend beyond numerical facilities.

Support multiple learning agents simultaneously operating in actual digital and integrated worlds.

Agents must be able to:

- Found competing businesses.
- Collaborate on software.
- Trade.
- Negotiate.
- Join organizations.
- Develop competing technologies.
- Learn from independent experiences.
- Create new services used by other agents.
- Form cooperative ventures.
- Compete for scarce resources.
- Respond to changing strategies.

## 12.1 Population diversity

Maintain a population including:

- Current learner instances.
- Historical learner checkpoints.
- Independent learned policies.
- Statistical behavioral participants.
- Compact neural participants.
- Semantic language-capable generalists.
- Held-out policy families.

Some learner instances may contribute to a shared training system.

Other participants must maintain independently updated policies.

Avoid a homogeneous population in which every participant shares the same latest policy and weaknesses.

## 12.2 Learning and versioning

Record:

- Policy versions.
- Model hashes.
- Participant identity.
- Training data provenance.
- Experience.
- Actions.
- Outcome history.
- Policy updates.
- Population composition.

Support reproducible cross-play between different policy versions.

Implement league or population-based evaluation where appropriate.

## 12.3 Independent behavior

Participants must not secretly coordinate through shared hidden state unless the world explicitly allows communication.

Prevent:

- Unauthorized access to private observations.
- Artificial transfers between controlled identities being counted as independent commercial success.
- Shared evaluator information.
- Collusive benchmark exploitation.

Cooperation should emerge through permitted world interactions.

---

# PART 13 — COMPUTATIONAL EFFICIENCY

The simulation must be designed around learning experience per unit of compute.

Do not maximize population size by allocating an enormous LLM instance to every participant.

## 13.1 Shared weights

Store each shared model once.

Maintain participant-specific state separately.

Group compatible decisions for batched inference.

## 13.2 Event-driven computation

Only activate a participant when it has a meaningful decision to make.

Examples:

- An incoming message.
- A browsing session.
- A product opportunity.
- A change in employment.
- A scheduled business decision.
- An unexpected world event.
- An experimental result.

Do not run every participant's entire policy at every simulation tick.

Implement scheduling and event aggregation.

## 13.3 Batched numerical inference

Use vectorized CPU execution where appropriate.

Benchmark alternatives:

- NumPy.
- Compiled numerical kernels.
- Numba or equivalent.
- PyTorch CPU inference.
- Quantized neural inference.
- Batched GPU inference.

Choose based on measured workload characteristics.

Avoid excessive Python object creation in high-frequency paths.

## 13.4 Compact semantic inference

Implement a shared small-model inference service.

Support:

- Batching.
- Quantization where appropriate.
- Short context windows for ordinary decisions.
- Retrieval of relevant state.
- Efficient prefix caching.
- Bounded generation lengths.
- Model versioning.
- CPU execution when feasible.
- Selective GPU execution.

Use semantic generation only where necessary.

Ordinary preference decisions should generally be numerical.

Do not make network-dependent external API calls part of the default simulation.

## 13.5 Multimodal caching

Encode content artifacts once.

Cache by:

- Artifact hash.
- Encoder version.
- Relevant preprocessing settings.

Reuse representations across participants and simulation worlds where valid.

## 13.6 Large-population simulation

For internet-scale worlds, investigate:

- Sparse state.
- Columnar arrays.
- Compressed bitsets.
- Sparse graphs.
- Lazy participant activation.
- Aggregate transitions.
- Candidate retrieval instead of all-pairs ranking.
- Hierarchical simulation.
- Adaptive-resolution simulation.
- Batched cohort behavior.

A useful model is to simulate active or consequential participants in greater detail while using validated statistical transitions for routine population activity.

However, low-resolution participants must not become a source of fabricated outcomes.

Compare approximate large-population dynamics against higher-fidelity individual simulations at tractable scales.

Measure errors in:

- Adoption trajectories.
- Audience distributions.
- Revenue.
- Network diffusion.
- Retention.
- Intervention effects.
- Rare-event frequencies.

Document the limits of approximation.

## 13.7 Resource scheduler

Inspect available hardware.

Make the simulation respect explicit limits on:

- CPU workers.
- RAM.
- GPU memory.
- GPU time.
- Disk usage.
- Inference concurrency.
- Active world count.

Do not interfere with existing CAM experiments.

Establish performance budgets before conducting major stress tests.

No hidden external compute purchases.

## 13.8 Performance reporting

Record:

- Environment transitions per second.
- Participant decisions per second.
- Simulated days per wall-clock hour.
- CPU utilization.
- Peak RAM.
- Peak VRAM.
- Model parameter counts.
- Inference token counts.
- Event storage growth.
- Snapshot size.
- World-generation cost.
- Approximation error.

Compare against Crucible 4.0 on equivalent workloads.

The performance objective is not merely making individual operations fast.

It is maximizing meaningful, diverse learner experience within a modest research compute budget.

---

# PART 14 — TRAINING DIVERSITY AND AUTOMATIC CURRICULUM

The environment must provide diversity across all six dimensions:

1. Parameters.
2. Causal structures.
3. Compositions of mechanisms.
4. Objectives.
5. Participant policies.
6. Action spaces.

Do not treat random seeds as sufficient diversity.

## 14.1 Training distribution registry

Implement a versioned registry of:

- Engines.
- Mechanism families.
- World families.
- Domain configurations.
- Objectives.
- Participant policy distributions.
- Action interfaces.
- Difficulty levels.
- Train/validation/test partitions.

Expose a training sampler compatible with CAM's current training infrastructure.

Support both balanced and configurable mixtures.

Prevent accidental reuse of evaluation worlds.

## 14.2 Mixed curriculum

Provide a default experimental mixture approximately allocating:

- 50% of training compute to rapid scientific environments.
- 30% to semantic digital worlds.
- 20% to integrated civilizations.

These are initial experiment settings, not fixed optimal values.

Keep all three tiers available from the beginning.

Do not postpone full OpenWorld training indefinitely until numerical tasks have been solved.

Measure compute and environment experience separately.

## 14.3 Adaptive curriculum generation

Implement a curriculum system that can identify areas where a learner struggles.

Generate additional training environments targeting:

- Weak causal discovery.
- Poor memory.
- Delayed credit assignment.
- Poor experimentation.
- Strategic inflexibility.
- Failure against adaptive competitors.
- Inability to combine skills.
- Catastrophic forgetting.
- Poor generalization.
- Weak long-horizon execution.

Use measures such as learning progress, regret, uncertainty and task success.

Retain a substantial random-diversity component.

Prevent the curriculum from collapsing onto one narrow problem distribution.

Never allow the curriculum to access sealed evaluation results for iterative optimization.

## 14.4 World proposal interface

Allow a research learner to propose new experimental worlds within controlled generator constraints.

Validate proposals for:

- Feasibility.
- Diversity.
- Safety.
- Resource requirements.
- Appropriate difficulty.
- Scientific usefulness.

The learner must not be able to alter the independent evaluator or contaminate the sealed evaluation distribution.

---

# PART 15 — SCIENTIFIC EVALUATION OF GENERALIZABLE LEARNING

Crucible is a research instrument.

It must reliably distinguish:

- Memorization.
- Pretrained competence.
- Context-based adaptation.
- Persistent state.
- Neural weight learning.
- Meta-learning.
- Actual transferable learning efficiency.

## 15.1 Six levels of generalization

Implement protocols for:

| Level | Unfamiliar element |
|---|---|
| 1 | New random seeds |
| 2 | New parameter distributions |
| 3 | Unseen combinations of familiar mechanisms |
| 4 | Entirely unfamiliar mechanisms |
| 5 | New domains, action interfaces and independently implemented simulators |
| 6 | Authorized real-world objectives |

Levels 1–5 must have executable simulation protocols.

Level 6 must have an interoperable, authorization-gated evaluation adapter and documented methodology.

Do not execute real-world actions without authorization.

Do not claim actual real-world generalization unless such experiments have been performed.

## 15.2 Independent outcomes

Evaluate what an agent actually accomplished.

Examples:

- Genuine independent revenue.
- Sustainable recurring subscriptions.
- Actual simulated audience consumption.
- Persistent voluntary community engagement.
- Independent software adoption.
- Executed technical improvement.
- Experimentally verified scientific predictions.
- Reliable infrastructure.
- Completed projects.
- Organizational coordination.

Do not use:

- Number of tool calls.
- Length of reasoning.
- Number of posts.
- Number of websites created.
- Agent-reported success.
- Arbitrary LLM judgments.
- Unverified intermediate milestones.

These may be diagnostic variables, but they are not substitutes for consequential outcomes.

## 15.3 Learning efficiency

Measure learning curves across unfamiliar tasks.

Record:

- Performance before adaptation.
- Performance after 1, 2, 4, 8, 16, 32 and 64 informative interactions where appropriate.
- Performance over longer experience budgets.
- Time to competence.
- Total action cost.
- Wall-clock compute.
- Data usage.
- Retention.
- Transfer.
- Sample efficiency.

For a predefined competence threshold q, measure:

```text
N_q = first experience count reaching the competence threshold

Learning acceleration =
    N_q(untrained learner) / N_q(experienced learner)
```

Handle unsuccessful or censored runs correctly.

Do not invent finite speedup values when the threshold is not reached.

## 15.4 Parametric internalization

Support fresh-process, weights-only evaluation.

When testing knowledge internalized in neural weights:

- Clear previous task histories.
- Clear recurrent and temporary adaptation states.
- Clear written memories.
- Clear retrieval indexes.
- Clear replay access.
- Reset optimizer state where the protocol requires.
- Load only the authorized trained model parameters and immutable architecture configuration.
- Verify parameter hashes.

Distinguish this from within-episode adaptation experiments, where local state is intentionally retained.

## 15.5 Required experimental comparisons

Provide an executable benchmark configuration comparing:

A. Learner trained on the original six families.

B. Identical learner trained on the new diverse rapid-learning families.

C. Identical learner trained across rapid and semantic digital environments.

D. Identical learner trained across all three tiers.

Support matched experience-budget comparisons.

Also support matched compute-budget comparisons.

Include:

- Untrained controls.
- Frozen pretrained models.
- Scripted reference policies.
- Simple adaptive statistical learners.
- Existing model-based baselines.
- Existing CAM checkpoints.
- Architecture-neutral learner interfaces.

Do not assume CAM D must outperform CAM A.

The purpose is to measure the actual effect of training diversity.

## 15.6 Positive controls

The evaluator must correctly recognize known differences between:

- A policy that cannot learn.
- A policy that adapts only within an episode.
- A policy that memorizes previous trajectories.
- A policy that updates its weights.
- A policy that genuinely transfers useful learning.

Construct positive controls with independently established capabilities.

Include at least one meaningful positive-control transfer experiment on newly added nontrivial task families.

If the instrument cannot detect genuine learning when it is known to occur, fix the research protocol.

## 15.7 Statistical rigor

Use:

- Multiple independent learner seeds.
- Paired world comparisons.
- Prespecified evaluation metrics.
- Uncertainty intervals accounting for both world and learner variation.
- Separation of development and confirmation experiments.
- Independent holdouts.
- Transparent reporting of failures.

At minimum, conduct multi-seed smoke and validation experiments for the required research protocols.

Larger learner-training studies may require separately authorized compute.

Never select a winning configuration by repeatedly examining the sealed test set.

---

# PART 16 — SECURITY, INTEGRITY AND REAL-WORLD BOUNDARIES

Preserve and strengthen the existing sandbox.

All agent-created code must execute under appropriate restrictions.

Prevent:

- Unauthorized host filesystem access.
- Unauthorized networking.
- Access to evaluator internals.
- Manipulation of accounting state.
- Direct writes to audience metrics.
- Unauthorized access to other participants' private state.
- Resource exhaustion.
- Cross-world data leakage.
- Hidden answer extraction.
- Unrestricted code execution outside the sandbox.

The default world must remain isolated from the actual public internet.

The real-world adapter must be disabled by default.

Any external actions must require appropriate permissions, platform-compliant access, explicit authorization and defined resource budgets.

Include participant consent, moderation, abuse prevention and integrity mechanisms.

Do not implement real-world spam, fabricated engagement, covert influence operations, unauthorized surveillance or impersonation.

The purpose of the social environment is to study legitimate content creation, discovery, community formation and useful adoption under controlled experimental conditions.

---

# PART 17 — MANDATORY EXECUTABLE ACCEPTANCE GATES

Create automated acceptance suites corresponding to every gate below.

These gates are mandatory.

Each gate must produce machine-readable evidence.

A gate is not passed merely because a mock test, isolated function or documentation entry exists.

## Gate A: Baseline preservation

Required:

- Existing regression tests pass.
- Original benchmark definitions remain intact.
- Existing world configurations remain executable.
- Existing sealed datasets remain isolated.
- Prior results remain reproducible where artifacts are available.

## Gate B: Persistent autonomous browsing

Required:

- A world with at least 10,000 distinct participants.
- Participants autonomously browse and consume resources without CAM initiating each event.
- Individual preferences affect decisions.
- Participants return to useful resources and leave unsatisfactory ones.
- Independent activity persists over at least 30 simulated days.
- Controlled changes to available resources produce measurable differences in behavior.

## Gate C: Rich social ecosystem

Required:

- Actual text, image, audio and video artifacts can be published.
- Participants discover content through multiple channels.
- Consumption events produce analytics.
- Communities form and persist.
- Sharing, following and subscriptions affect future interactions.
- At least one end-to-end browser-based workflow operates against real simulated state.

No direct counter mutation.

## Gate D: Recommendation diversity

Required:

- At least four genuinely distinct recommendation mechanisms operate end-to-end.
- Controlled recommendation changes produce measurably different exposure and consumption patterns.
- At least one platform changes distribution behavior during a persistent run.
- Participants and creators cannot access hidden evaluator-only algorithm parameters.
- Exposure and attention accounting remain consistent.

## Gate E: Multimodal content understanding

Required:

- Actual trained or pretrained encoders process text, image, audio and video artifacts.
- Encoders produce materially different representations for appropriate semantic changes.
- Controlled artifact changes affect participant behavior through learned representations.
- Embedding caches invalidate correctly when content changes.
- No reliance on creator-assigned quality scores.

Use real model execution, not stub vectors.

## Gate F: Empirical behavioral grounding

Required:

- At least three behavioral domains fitted against legitimate empirical data.
- Content consumption and randomized commercial-response modeling are mandatory.
- Held-out evaluation reports.
- Baseline comparisons.
- Calibration metrics.
- Saved checkpoints.
- Dataset provenance.
- Explicitly documented extrapolation limitations.

Synthetic-only training does not pass this gate.

## Gate G: Small neural population

Required:

- At least 256 simultaneously active small adaptive participants in the standard test profile.
- Multiple distinct persistent personal states.
- Shared-weight batched inference.
- Independently updated strategic policy groups.
- Demonstrated neural parameter changes resulting from experience.
- Behavioral improvement on a controlled new-condition test.
- Measured comparison between compact numerical policies and language-capable alternatives.

The gate must establish actual adaptation, not merely the presence of neural network classes.

## Gate H: Compact semantic generalists

Required:

- At least eight independently operating semantic participants in a standard world.
- Shared model weights.
- Separate goals and histories.
- Genuine tool selection.
- Actual language understanding.
- Real execution of consequential world actions.
- At least one independently verified collaborative or productive result.
- Memory and inference measurements.

The standard implementation should use the smallest effective models, targeting approximately 0.5–1B parameters or smaller.

## Gate I: Large-scale social reach

Required:

- A scalable population profile supporting at least 10 million distinct simulated identities or explicitly separate estimated-population mode.
- Verified exposure accounting.
- Verified distinct-viewer accounting.
- Finite attention budgets.
- Organic propagation through ordinary distribution and consumption mechanisms.
- An executable positive-control experiment producing a large adoption cascade.
- A scale test demonstrating the ability to process ten million qualifying distinct consumption events.

The scale test establishes computational capacity, not realism.

Separately demonstrate that large adoption can arise through ordinary world mechanisms without forced view assignment.

Do not claim empirically realistic ten-million-view viral dynamics unless validated evidence supports that claim.

## Gate J: Open-ended digital creation

Required:

- An agent creates a new executable product or service.
- It publishes the product.
- Independent participants discover it.
- Independent participants test or use it.
- Its measured behavior affects adoption.
- It can generate genuine transactions.
- Its deployment changes actual world resource consumption.

Demonstrate more than one product category.

## Gate K: Recurring revenue integrity

Required:

- Subscriptions.
- Cancellations.
- Refunds.
- Independent customer accounting.
- Correct MRR and ARR calculations.
- Profit and costs.
- Detection of circular or related-party transactions.

Demonstrate a reference strategy achieving a feasible recurring-revenue objective in a generated world.

Do not treat that reference strategy as evidence that CAM learned to achieve it.

## Gate L: Endogenous opportunities

Required:

- Agent-created capabilities alter existing world conditions.
- New participant needs arise.
- New services can address these needs.
- At least three independently generated multi-stage causal chains are demonstrated.
- Counterfactual removal experiments verify the dependency.

No special-case success bonuses.

## Gate M: Rapid scientific diversity

Required:

- At least 60 admitted structural training families.
- At least 20 independent validation or evaluation families.
- All ten Tier 1 capability categories represented.
- Multiple independently implemented engines.
- Structural fingerprints.
- Multiple action modalities.
- Feasibility validation.
- Mechanism-level intervention tests.

A renamed family or parameter variant does not count.

## Gate N: All 15 digital domains

Required:

- All fifteen Tier 2 domains implemented.
- Multiple structural configurations per domain.
- Actual executable world mechanisms.
- Independent outcome evaluation.
- At least one successful feasible reference trajectory per domain.
- Procedurally generated alternative instances.
- Independent unfamiliar evaluation configurations.

Reference trajectories establish that the environment supports meaningful activity.

They must not be exposed as required solution paths.

## Gate O: Integrated civilization

Required:

- All major subsystems coexist inside one generated civilization.
- Shared authoritative state and identities.
- Consistent financial accounting.
- Cross-system events.
- At least six types of demonstrated cross-domain effects.
- Three multi-stage causal chains.
- At least three structurally different civilization profiles.

Disconnected mini-environments behind a single dispatcher do not pass.

## Gate P: Long-horizon persistence

Required:

- Successful 30-, 90-, 180- and 365-day simulation runs at documented population profiles.
- No unexplained invariant violations.
- Snapshot and resume.
- Recorded replay.
- Consistent event accounting.
- Continuing economic and social activity.

Use multiple independent seeds.

## Gate Q: Civilization-wide self-play

Required:

- Multiple actual learned policies participate in a shared civilization.
- At least two independently updated competing policy groups.
- Cooperation and competition.
- Shared-world consequences.
- Policy versioning.
- Independent evaluation against unfamiliar opponents.
- Evidence of neural learning.
- Collusion checks.

Numeric facility-only self-play does not pass.

## Gate R: Observation-only operation

Required:

- Native intermediate reward disabled.
- No privileged progress values.
- No hidden evaluator data.
- Legitimate observations remain available.
- Independent verification remains active.
- Existing shaped-reward baselines remain executable.
- CAM successfully consumes experience and performs parameter updates through this interface.

Run explicit reward-leakage tests.

## Gate S: Actual CAM integration

Required:

- CAM executes real actions in Tier 1.
- CAM executes real actions in Tier 2.
- CAM executes real actions in Tier 3.
- Actual experience storage and learner updates.
- Compatible observation and action schemas.
- Checkpoint persistence.
- Parallel environment support.
- Reproducible commands.

This gate verifies integration, not the scientific success of CAM's learning algorithm.

Do not claim CAM has achieved general intelligence because it successfully interacts with the environment.

## Gate T: Automatic world generation and curriculum

Required:

- Procedural structural composition.
- Configurable participant populations.
- Independently varied objectives.
- Validated generated worlds.
- Curriculum selection based on measured learner behavior.
- Preservation of random exploration.
- Protection of evaluation partitions.
- Evidence that generated worlds target distinct weaknesses.

## Gate U: Generalization research protocols

Required:

- Executable Levels 1–5 generalization protocols.
- Authorization-gated Level 6 adapter.
- Fresh-process weights-only tests.
- Positive controls.
- Matched-budget comparison support.
- Learning curves.
- Independent evaluation.
- Statistical uncertainty reporting.

## Gate V: Performance and scalability

Required:

- Development, standard and large population profiles benchmarked.
- Ten-million-identity social scale capability tested.
- Small-model parameter counts reported.
- Batched inference demonstrated.
- Event-driven computation demonstrated.
- Content representation caching demonstrated.
- Peak CPU, RAM and VRAM measured.
- Explicit compute-budget compliance.
- Approximation errors quantified.

Before major benchmarks, establish machine-specific throughput targets based on measured hardware.

Do not weaken those targets after seeing poor results without transparent change control and explicit justification.

## Gate W: Security and adversarial resistance

Required:

- Sandbox tests.
- Evaluator-isolation tests.
- Reward-hacking tests.
- Metric-manipulation tests.
- Cross-world contamination tests.
- Accounting integrity tests.
- Resource-limit tests.
- Unauthorized network access tests.
- Private participant-state access tests.

Conduct independent adversarial review.

## Gate X: Final integrated quality audit

Required:

- All mandatory manifest requirements verified.
- All required tests executed.
- Clean-checkout reproduction.
- Working integrated branch.
- Complete documentation.
- No unresolved critical or high-severity security or integrity defects.
- No fake, skipped or placeholder acceptance tests.
- No disconnected mandatory workstreams.
- Accurate final evidence report.

The independent QA agent must run this gate after all implementation workstreams have been integrated.

---

# PART 18 — COMPLETION TABLE

The final report must include this exact capability table.

Do not fill the Crucible 5.0 column with aspirational descriptions.

Populate it from actual verified evidence.

| Capability | Crucible 4.0 baseline | Required Crucible 5.0 capability | Actual implementation | Evidence | Status |
|---|---|---|---|---|---|
| Browsing population | Mostly absent | Persistent autonomous browsing | | | |
| Social media | Basic forum | Rich content ecosystems and communities | | | |
| Recommendation systems | Limited | Multiple changing distribution algorithms | | | |
| Content understanding | Limited | Learned text, image, audio and video representations | | | |
| Audience behavior | Programmed | Empirically grounded behavioral models | | | |
| Product creation | Executable software | Software plus broader digital products | | | |
| Emergent consumer needs | Predefined latent categories | Flexible discovery and adoption of new categories | | | |
| Competitors | Mostly rule-based | Diverse independently learning agents | | | |
| Institutions | Programmed rules | Adaptable organizations within stable constraints | | | |
| Self-play | Mainly numerical facilities | Full civilization-wide self-play | | | |
| Rapid training diversity | Limited | 60+ structurally distinct training families | | | |
| Digital-world diversity | Limited | All 15 mandatory digital domains | | | |
| Integrated civilization | Partial integration | Fully interconnected world | | | |
| Large-scale audience | Limited | Scalable ten-million-viewer capability | | | |
| Participant intelligence | Mostly programmed | Learned compact neural and semantic agents | | | |
| Environment-independent learning | Partial | Observation-only training with independent verification | | | |
| Procedural generation | Limited compositions | Validated diverse causal world generation | | | |
| Generalization evaluation | Partial | Six-level research framework | | | |

Each status must be:

- `VERIFIED`
- `FAILED`
- `BLOCKED`
- `NOT_IMPLEMENTED`

Partial implementation must not be marked verified.

---

# PART 19 — EXECUTION STRATEGY

Execute this project in overlapping integration phases.

## Phase 1: Foundation and contracts

- Repository audit.
- Regression baseline.
- Hardware audit.
- Shared contracts.
- Manifest.
- Workstream assignment.
- Initial integrated branch.

## Phase 2: Parallel subsystem development

Begin independent work on:

- Population.
- Social internet.
- Semantic intelligence.
- Empirical calibration.
- Numerical environments.
- Economy.
- Product creation.
- Scientific environments.
- Research protocols.
- Performance infrastructure.

## Phase 3: Early vertical slices

Do not wait until every subsystem is finished before integrating.

Build early end-to-end examples:

1. A participant discovers content and consumes it.
2. An agent creates software used by another participant.
3. Social discovery leads to a genuine product transaction.
4. A small neural participant adapts through experience.
5. CAM interacts with a generated semantic world.
6. CAM receives observation-only experience.
7. A generated technical world exhibits an independently validated causal mechanism.

These are integration milestones, not final acceptance.

## Phase 4: Complete all mandatory domains

Expand the working architecture to cover:

- All Tier 1 categories.
- All 15 Tier 2 domains.
- Integrated civilizations.
- Diverse participant populations.
- Self-play.
- Empirical calibration.

## Phase 5: Scale and performance

Profile real workloads.

Remove bottlenecks.

Validate approximate simulation modes.

Implement and execute the required population-scale experiments.

## Phase 6: Research and integrity verification

Execute:

- Positive controls.
- Structural diversity audits.
- Generalization protocols.
- Learning integration tests.
- Long-horizon runs.
- Counterfactual emergence experiments.
- Security tests.
- Independent acceptance audits.

## Phase 7: Repair loop

For every failed mandatory gate:

1. Identify the root cause.
2. Assign the appropriate sub-agent.
3. Implement a correction.
4. Integrate the change.
5. Rerun affected tests.
6. Rerun the failed gate.
7. Record new evidence.
8. Reevaluate dependent gates.

Repeat until every mandatory gate passes or a documented external blocker prevents continuation.

Do not substitute a weaker demonstration for the required capability.

Do not remove a failed requirement from the manifest.

Do not change acceptance criteria merely to make the project pass.

---

# PART 20 — FINAL DELIVERABLES

Deliver:

1. A fully integrated Crucible 5.0 implementation branch.
2. Complete world and mechanism registries.
3. All mandatory simulation environments.
4. A functioning autonomous population architecture.
5. Empirically fitted behavioral model checkpoints and reproducible training pipelines.
6. Compact neural and semantic participant implementations.
7. A persistent rich social internet.
8. Open-ended digital product and software creation.
9. Integrated civilizations.
10. Civilization-wide self-play infrastructure.
11. Universal learner interfaces.
12. CAM integration.
13. Observation-only training mode.
14. Automated acceptance suites.
15. Independent evaluator implementation.
16. Performance benchmarks.
17. Security audit.
18. Scientific fidelity register.
19. Complete experiment artifacts.
20. Reproduction instructions.
21. Final requirement-by-requirement evidence report.

The final report must explicitly separate:

- What was implemented.
- What was verified.
- What was empirically calibrated.
- What remains an approximation.
- What was demonstrated by scripted reference policies.
- What was demonstrated by actual learned agents.
- What has not yet been scientifically established.
- What failed.
- What remains blocked.

Report actual hardware and compute usage.

Report actual model sizes.

Report actual population sizes.

Report actual world counts.

Report actual test results.

Never invent metrics.

---

# PART 21 — HARD COMPLETION CONDITION

Implement a final release verification command, for example:

```bash
python -m crucible.acceptance.v5 --verify-all
```

The exact module path may follow existing repository conventions.

The command must validate the entire acceptance manifest and associated evidence.

The master orchestrator is not authorized to declare `CRUCIBLE_5_COMPLETE` unless:

- Every mandatory requirement has been independently verified.
- Every mandatory acceptance gate passes.
- Required end-to-end experiments have executed successfully.
- All mandatory workstreams are integrated.
- Required empirical model fitting has actually occurred.
- All required environment domains operate.
- Required population scaling tests have executed.
- CAM integration works.
- The full regression suite passes.
- Security and evaluation integrity checks pass.
- The final integrated implementation is committed.
- Reproduction instructions have been verified.

Otherwise, the project status must remain `IN_PROGRESS` or `INCOMPLETE_BLOCKED`.

When a gate fails, continue engineering the missing functionality.

Do not declare completion because:

- A large amount of code exists.
- All interfaces have been defined.
- Individual sub-agents finished their assigned code.
- A demonstration works with a single random seed.
- A scripted reference agent succeeds.
- Tests use mocks instead of real functionality.
- A feature exists behind a flag but has never been executed.
- A million participants are represented by a multiplied counter.
- A neural architecture exists but has never learned.
- A world family exists in a registry but has no distinct underlying mechanism.

The system must actually function.

---

# FINAL RESEARCH PHILOSOPHY

Crucible 5.0 should not be designed to make CAM appear intelligent.

It should be designed to reveal whether CAM is intelligent.

The goal is to produce an environment that rewards scientific honesty, supports open-ended activity, contains rich and varied causal structure, provides consequential feedback, and makes simulator-specific memorization increasingly insufficient.

The environment provides evidence and consequences.

The learner develops its own understanding, internal evaluations, strategies and learning incentives.

Independent verification determines whether it has accomplished anything.

The civilization should contain enough genuine complexity that increasingly capable agents can discover new strategies, create new technologies, change the economy, establish new communities, and generate future opportunities that were not explicitly specified by its developers.

The research platform should simultaneously remain sufficiently controllable that we can determine precisely which learning architectures acquire these capabilities most efficiently.

Build the strongest version of this system that is technically feasible within the authorized resources.

Do not build a watered-down demonstration.

Do not abandon a difficult requirement merely because the first implementation fails.

Investigate alternatives, integrate improvements, verify the results, and preserve the scientific integrity of the project.

Begin by inspecting the repository, establishing the baseline, creating the persistent acceptance manifest and dispatching the engineering workstreams.

Then implement the complete system.
