# AGENTRS Crucible: persistent synthetic digital civilization

**Status:** Research and architecture specification, not an implemented RL environment.
**Date:** 2026-11-20
**Branch:** `feat/crucible-design`

## Thesis

A crucible is not a finite benchmark or a scripted sequence of browser tasks. It is a persistent, partially observed, causally grounded world in which many actions can alter future opportunities; independent actors interact; resources have enforceable scarcity; new tools, businesses, communication protocols, and institutions can be invented and persist; and agents can learn across long lifetimes.

For AGENTRS, build a **synthetic internet and digital economy** whose persistent state is implemented in real code and databases. CAM, an unmodified frozen LLM, a pure learner, scripted control agents, and external research agents must all be able to inhabit the same world.

The primary research question: does experience in a sufficiently rich world lead CAM to discover useful, transferable strategies and learn them faster over time, without a predefined hierarchy of approved plans?

A million unrelated tasks are not a crucible. A million attractive HTML pages are not a crucible. The depth must come from a shared causal substrate.

## Design principles / non-negotiables

1. **World, not task collection.** The world continues between interventions. Agents can initiate projects and change what other agents subsequently encounter. External evaluation tasks are optional probes, not the core world dynamics.
2. **Authoritative state.** Conservation of money, inventory, permissions, company assets, projects, files, identity, and contractual obligations is enforced by deterministic services. A generative model may write believable communications but cannot silently invent transactions or retroactively change the ledger.
3. **Arbitrary strategy composition.** The agent can browse, read, write, code, negotiate, hire, build and operate software, form organizations, purchase simulated resources, and discover previously unknown app and API affordances. There must be no menu of hard-coded strategies.
4. **Partial observability.** Actors have private information, preferences, attention constraints, limited budgets and different incentives. Each actor discovers the state of the world by interacting with it.
5. **Persistence and counterfactual control.** The same world can run continuously in canonical mode and be forked by consistent copy-on-write snapshots for independent experiments and fair comparisons.
6. **Endogenous complexity.** New opportunities arise from scarcity, supply chains, competition, asynchronous interactions, software failure, infrastructure constraints, distribution, information asymmetry, and agent-generated artifacts; not just an ever-growing catalog of prompts.
7. **Grounded feedback.** Outcomes reflect state transitions and constraints. Success, progress, and safety are audited independently of what the agent says. No reward for events already true at initial reset.
8. **Multi-agent variety, not self-play only.** Simulated users, firms, workers, markets, and regulators can be backed by rules, goal-driven scripted models, frozen LLMs, or other policies. Keep their capabilities and motivations heterogeneous.
9. **Scientific falsifiability.** Simple heuristics, the frozen-LLM baseline, imitation-trained CAM, RL-trained CAM, and outer-loop discovery are all evaluated under matched tools and budgets, including held-out world-generating rules.
10. **Strong containment.** Every transaction is synthetic. Agents have no access to real banking, live emails, production credentials, or unrestricted external networking. Generated programs run in isolated, resource-limited sandboxes.

## Proposed architecture

```text
                    AGENTRS LEARNERS
      frozen LLM | CAM+LLM | pure CAM | baselines
                            |
                       OpenEnv/MCP
                            |
                  CRUCIBLE ACTION GATEWAY
            browser | terminal | files | APIs | messages
                            |
           +----------------+----------------+
           |                                 |
     SYNTHETIC INTERNET               SOCIAL ACTOR ENGINE
 browser renderers, search,           customers, organizations,
 app fronts, routing, mail,           competitors, workers,
 forums, Git hosting, docs           adversarial / cooperative bots
           |                                 |
           +----------------+----------------+
                            |
                   CANONICAL WORLD KERNEL
   ledger | identity | permissions | files | software artifacts
   inventory | labor | computing | marketplace | contracts
   communication | information propagation | clock / events
                            |
                    EVENT LOG + SNAPSHOTS
                 fork | replay | audit | compare
                            |
                    INDEPENDENT EVALUATOR
         transfer | retained knowledge | novelty | efficiency
```

### Canonical world kernel

Model a connected graph of real entities and causal relationships: people, organizations, accounts, inventory, contracts, services, projects, repositories, advertisements, messages, cloud resources, and actions. Use transactional Postgres (or another database with comparable invariants) and append-only event records. Each public application must read or write the SAME shared state, rather than an unrelated per-app toy fixture.

Enforce balances, ownership, spending constraints, authorization, deadlines, limited machine capacity, market supply and demand, network effects, switching costs, failures, refunds and contractual duties. Avoid fake economic outcomes decided by a judge LLM.

### Synthetic internet

Start with tightly integrated, fully functional categories rather than superficially copying thousands of pages: search; messaging/mail; social forum; identity; code/Git; issue tracking; document storage; cloud/compute service; payments/ledger; marketplace; analytics/ads; customer support; calendar; and business administration. A coherent world with fewer real apps beats thousands of disconnected mocks.

Expose sites through realistic HTML/DOM and accessibility trees, and their own backend APIs where applicable. Support link discovery, search indexing, programmatic software creation and hosting, redirects, permissions, authentication, rate limits, stale data, network faults, and asynchronous changes. Browser traffic stays inside an isolated fake network. A generated site is admitted only if both functionality tests and backend invariant tests pass.

Do not initially reproduce TLS impersonation of real domains. Synthetic internal DNS and isolated names are sufficient. Google Chrome's Mirror Dimension provides an instructive reference for eventual high-fidelity browser evaluation, but its implementation is not publicly released.

### Digital ecology and economics

A bounded economy ties together customers, services, wages, compute, inventory, pricing, contracts, trust and reputation. Simulated buyers have latent preferences and time-varying budgets. Demand depends on actual service quality, price, availability, reputation, distribution and competition. Competitors adapt within bounded rationality. Failure to maintain software, pay workers, deliver goods, or address support tickets changes future economic state.

The economy must admit multiple valid paths: improve a product, automate workflows, reduce costs, acquire customers, build infrastructure, form partnerships, pivot, invent a new product, or discover a better strategy. No action is scored merely because a developer named it "good".

### Artifact creation and open-ended computation

Agents can create real code, documents, APIs, websites, algorithms, and executable tools in a sandbox. Their products can be used by other agents and persist across simulated time. This is essential: invention is not possible if all useful operations are pre-supplied.

Launch code in sandboxed processes with quota limits, allowlisted synthetic network, deterministic input fixtures when appropriate, and snapshotable artifacts. Never let an agent alter the world kernel or evaluator directly.

### Multi-agent society

Use a hierarchy of actors: cheap rule-based background populations, lightweight preference-based agents, selectively active frozen-LLM agents, and trainable CAM-like agents. Actors get unequal information, costs, goals and resources; they can send delayed messages and make commitments. Asynchronous event scheduling permits a world with many sleeping actors without an LLM forward pass for each entity every tick.

Do not treat simulated agent chatter as evidence of economic complexity. Monitor causal effects of communication, trade, coordination, delegation and institutional changes.

### Simulation time and speed

Implement two execution modes:
- **Headless fast mode:** authoritative state transitions and APIs without rendering browsers for every actor, designed for efficient parallel training.
- **Full-browser mode:** real Chromium with working UIs for selected agents or realistic evaluation, designed to test transfer to actual computer use.

Run a continuous persistent canonical world for longitudinal research, and use snapshots plus forked training worlds for RL training and fair counterfactual evaluations. Never contaminate sealed evaluations with current canonical-world state.

### Agent action boundary

Protocol: enumerate dynamic tool schemas, make valid calls, receive partial observations, handle asynchronous jobs, execute sandboxed code, and perform browser interactions. A schema is not a strategy. CAM should decide its plans; LLM assistance for parsing and text is allowed. Separate learned policy decisions from text formatting, so invalid JSON does not dominate sparse task reward.

### Example crucible episode / lifetime (illustrative, not scripted)

The agent starts as a resource-limited participant in a synthetic commercial ecosystem. It can inspect market offers, exchange messages, write software, publish a service, acquire simulated customers, or negotiate access to resources. Over simulated weeks, user needs and competitors change; services fail, bills recur, customer sentiment propagates, employees leave, and new opportunities arise.

One learned strategy might be to build a monitoring application and sell its services; another to acquire an existing product or collaborate on shared infrastructure. The simulator has no special "build-monitoring-app" action or reward bonus. Only consequences in the market and resource system make the strategy useful.

No scripted walkthrough of the "correct" path exists. A legitimate simple heuristic must be able to fail when circumstances change.

## Rewards: simple incentives, independently rich evaluation

Do not reward every desired high-level human behavior (invent, coordinate, reason, build). Start with a small set of **world-level objective contracts** such as sustainable utility, solvency while delivering requested useful services, resource efficiency, and fulfillment of obligations. Enforce hard safety and integrity constraints separately. Objective weights may vary across world seeds and actor roles.

Multiobjective results should be logged as a vector, not collapsed into a single opaque total. The learner sees only its actual feedback; the evaluator sees detailed protected state.

Use objective metrics: survival/solvency subject to real operating costs, quality-adjusted completed transactions, verified functional software, contract compliance, resource use, retained customer value, environmental integrity, and earned reputation. Track exploration and novel capability emergence separately; avoid direct "novelty points" that reward random noise.

Prevent degenerate optima: inactivity, early termination, self-trading, money minting, replaying a prior success, generating fake users, spam, evaluator contamination, exploiting reset boundaries, or manipulating output text to satisfy an LLM judge.

## Research evaluation

Evaluate on two axes:

**Agent competence:** economic outcome and quality, dynamic problem solving, long-horizon adaptation, transfer to changed markets/new applications/other actors, forgetting, sample efficiency and tool cost.

**Self-improvement:** learning-curve AUC across independent worlds; improvements from experience relative to the same frozen LLM and imitation-only CAM; improvements from outer-loop architecture discovery relative to random and fixed searches; independent held-out families/rules and fresh seeds.

Keep a locked **world-generator test partition**, not only held-out tasks in the same world. Withhold entire services, preference families, network structures, market dynamics, and populations. Measure whether knowledge acquired in world A accelerates learning in world B.

Pre-register baselines, sample sizes, metrics and compute budgets. All populations should be paired on seeds and world snapshots. Use statistical uncertainty and distinguish task success from incremental state progress. Require source/target transfer; improvement on a familiar finite benchmark alone does not demonstrate emergent intelligence.

Crucible progress gates:
- Real state changes and invariant tests verified across browser and API actions.
- Nonzero causally grounded success for a competent LLM-only agent on real multi-step goals.
- A random policy / trivial script does not reliably achieve the same outcomes.
- Reproducible snapshots, rollback, fork, and event replay.
- Multiple novel strategies can demonstrably solve the same objective under different world states.
- Agent-created software or arrangements can alter future trajectories of other actors.
- Genuine domain transfer demonstrated across independent generator seeds and structural variations.
- Outer learner candidates cannot see, modify, or exploit the held-out evaluator.
- If no candidate beats the LLM-only baseline, report negative evidence rather than a success claim.

## Development plan (research program, not a claimed implementation)

**Gate A: Core substrate.** Write event-sourced kernel, identity/permissions, money ledger, messaging, basic market, files, and a small number of real integrated software services. Set up deterministic replay, parallel forks, sandboxed agent actions and state audits.

**Gate B: World richness.** Add asynchronously acting heterogeneous populations, persistent organizations, contracts, app ecosystems, product quality, infrastructure reliability, and meaningful agent-built artifacts. Demonstrate multiple viable emergent strategies without hardcoded plans.

**Gate C: Internet fidelity.** Build realistic service frontends, search/link graph, browser access, automatic site generation with static and runtime acceptance tests, and controlled procedural shocks. Connect many applications to one shared state.

**Gate D: Learning science.** Integrate CAM+frozen LLM and pure CAM, imitation warm-start, continual RL, transfer probes and proper baselines. Establish reliable reward and metrics before architecture discovery.

**Gate E: Scaling and discovery.** Optimize simulation throughput; broaden actors, structures and market mechanisms; run population-based architecture search over real multi-world learning performance; study independent discovery of reusable tools and strategies.

## What to reuse vs build

Use OpenEnv as a protocol layer after compatibility review; study Snowflake Agent World Model and RUC Agent-World for executable stateful tool worlds; study VeriEnv for generated sites and backend verifiers; study TheAgentCompany for integrated workplace software; study WebWorld as an approximate browser dynamics model, not an authority; study Concordia/OASIS/Emergence World for social dynamics and longitudinal evaluation; and study Google's closed Mirror Dimension for hermetic synthetic web design.

**Do not import third-party generated executable tool code without sandboxing and a license review.** OpenEnv's protocol does not itself provide a shared living world. Existing environments cannot simply be concatenated: they require shared identities, ledgers, schemas, causal dynamics and state versioning.

## Falsification and limitations

- Synthetic economies may incentivize strategies that fail in actual markets.
- LLM-driven NPCs may be expensive, inconsistent or easy to exploit.
- Richness is not captured by number of simulated agents or websites.
- Market reward can be gamed; unusual agent behavior is not automatically intelligent.
- A trainable CAM controller may not receive enough useful gradient through a frozen LLM decoder.
- Hardware budgets must be profiled; real browser and LLM execution costs are much higher than headless economic steps.
- The architecture is an ambitious research proposal, NOT evidence that a full-world simulator or a superlearner has been built.

## References (checked 2026-11-20)

- Ineffable Intelligence / Silver and Espeholt interview, supplied transcript: https://www.sequoiacap.com/podcast/training-data/
- Google Chrome Mirror Dimension (2026-11-19): https://blog.google/security/mirror-dimension-for-agents/
- RUC Agent-World: https://github.com/RUC-NLPIR/Agent-World
- Snowflake Agent World Model: https://github.com/Snowflake-Labs/agent-world-model
- OpenEnv: https://github.com/openenv/openenv
- VeriEnv: https://github.com/kyle8581/VeriEnv
- TheAgentCompany: https://github.com/TheAgentCompany/TheAgentCompany
- WebWorld: https://github.com/QwenLM/WebWorld
- Emergence World: https://github.com/EmergenceAI/Emergence-World (research-only/noncommercial license)
- DeepMind Concordia: https://github.com/google-deepmind/concordia
- OASIS: https://arxiv.org/abs/2411.11581

