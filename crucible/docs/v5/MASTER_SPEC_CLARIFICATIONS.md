# Clarifications to MASTER_SPEC.md

These are the user's own amendments, saved verbatim in the order received. Where a clarification conflicts with MASTER_SPEC.md, the clarification governs.

## Clarification 1 (2026-11-25): Crucible is independent of CAM

IMPORTANT ARCHITECTURAL CLARIFICATION:
Crucible 5.0 must be completely independent of CAM 7.0, CAM Unified, or any particular learning architecture.
CAM is being developed separately and is not yet finished. Do not wait for it, modify it, merge its unfinished implementation, or make Crucible's completion dependent on its capabilities.
Instead:

* Implement an architecture-agnostic environment and experience interface.
* Validate all environment capabilities using scripted reference agents, small neural RL agents, statistical learners and lightweight language-model agents.
* Support observation-only learning, externally verified outcomes, persistent experience collection and multi-agent interaction.
* Demonstrate actual neural parameter updates using a lightweight reference learner.
* Replace all CAM-specific mandatory acceptance requirements, including Gate S, with architecture-independent learner integration requirements.
* Preserve compatibility with existing learner interfaces wherever practical.
* Document how future CAM versions can connect to Crucible without requiring changes to the environment itself.

You may inspect CAM's interfaces for compatibility, but its implementation is not a dependency.
This clarification supersedes all CAM-specific implementation and acceptance requirements in the original specification. Every other Crucible 5.0 requirement remains mandatory.
Continue building the complete Crucible 5.0 environment independently.

## Clarification 2 (2026-11-26): Observation-only learning as the universal default

MANDATORY AMENDMENT — OBSERVATION-ONLY LEARNING AS THE UNIVERSAL DEFAULT
Crucible 5.0 must fundamentally operate as an environment that provides experiences and observable consequences, not as a reinforcement learning benchmark that grades the learner's actions.
Observation-only operation must be the default across ALL Crucible 5.0 environments, including controlled scientific environments, standalone digital worlds and integrated persistent civilizations.
This supersedes any conflicting reward-related requirements in the original specification.
Mandatory requirements
* Crucible must not provide predefined intermediate rewards, reward shaping, action-quality scores or privileged goal-progress measurements by default.
* The learner receives observable consequences, including transactions, customer behavior, technical measurements, social responses, resource changes, failures and delayed outcomes.
* Remove all native reward leakage through observations, auxiliary fields, tool responses and event metadata.
* Preserve actual observable quantities such as revenue, costs, customer retention and technical performance. These are world observations, not RL rewards.
* Where Gymnasium compatibility requires a scalar reward, return 0.0.
* Support persistent operation both with externally assigned objectives and without any assigned objective.
* Support self-generated experiments, hypotheses and learning objectives without automatically rewarding their completion.
* Preserve independent outcome verification through a separate evaluator-side interface. Verified outcome labels may supervise the learner's evaluation and meta-learning mechanisms, but must not become handcrafted intermediate rewards.
* Preserve historical reward functions as explicitly enabled experimental baselines only.
The default configuration must be:
reward_mode = "observation_only"
Architectural independence
The purpose of this change is to enable learning architectures such as CAM to develop their own learned reward functions, evaluate the significance of their experiences, assign retrospective credit and autonomously determine what is worth learning.
However, Crucible must remain completely independent of CAM. Do not implement CAM's internal reward function inside Crucible or make Crucible dependent on any particular learning architecture.
Mandatory independent verification
Add explicit acceptance requirements demonstrating that:
1. Observation-only mode is enabled by default across every environment family.
2. No native or privileged reward signals reach the learner.
3. Real observable consequences remain accessible.
4. Independent outcome verification remains functional.
5. Agents can operate indefinitely within configured resource limits without an assigned objective.
6. An architecture-neutral external learner can supply its own training rewards.
7. Delayed outcomes remain accessible for retrospective learning.
8. Historical reward-based experiments remain reproducible through explicitly selected alternative modes.
Assign these requirements to a dedicated implementation subagent and an independent verification subagent. Include every requirement in the executable acceptance manifest.
Do not declare Crucible 5.0 complete until these requirements have been independently verified.
Crucible provides experience. The learner determines what deserves reinforcement. Independent evidence establishes whether genuine learning and objective achievement occurred.

## Clarification 3 (2026-11-26): Recovery, critical audit and complete implementation (preamble of the resumed assignment)

Summary of the binding points (the user's full preamble is reproduced in docs/v5/RECOVERY_PREAMBLE.md):

1. Recover the latest implementation (feat/crucible-5-openworld, recovery branches) and continue on a new branch
   `feat/crucible-5-complete`; the old branch stays an immutable historical reference.
2. Conduct a comprehensive independent architectural audit of every subsystem against the full specification, with a
   master orchestrator coordinating independent auditing subagents; classify each subsystem RETAIN, REPAIR, EXTEND,
   REDESIGN, REBUILD or IMPLEMENT on evidence, not on the effort already spent.
3. Do not trust previous completion claims; re-evaluate every verification claim against reproducible evidence; never
   weaken acceptance criteria.
4. Preserve the research objective: the strongest computationally feasible experimental twin of the open internet,
   distinguishing generalisable continual learners from policies that memorise simulator rules.
5. Crucible is architecture-independent (supersedes CAM-specific requirements, including Gate S); observation-only is
   the default for new learner integrations; independent outcome verification stays separate.
6. Resume full implementation through parallel engineering workstreams with independent verification and repair passes.
7. Persistence: commit coherent milestones often, push the integration branch at every meaningful milestone with the
   authorised credential, keep a continuously updated progress ledger, preserve models and artifacts durably, never
   expose secrets, never force-push.
8. Hard completion condition: CRUCIBLE_5_COMPLETE only when every requirement is implemented, integrated, executed,
   independently audited and verified; otherwise IN_PROGRESS or INCOMPLETE_BLOCKED with the blocker documented.
