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
