"""Crucible 5.0 research instrument (workstream N; MASTER_SPEC Parts 14 and 15, Gates T and U).

    distribution   the versioned cross-tier training-distribution registry, partitions and guards (Part 14.1)
    sampler        seeded, resumable mixture sampler emitting JSON episode specs for any learner (14.1, 14.2)
    episodes       driving a learner through `crucible.experience` and measuring what it did (observation-only)
    learners       architecture-neutral control learners (positive and negative controls, Part 15.6)
    evaluator      the independent evaluator: a separate process holding worlds, partitions, salt and verifiers
    fresh          fresh-process weights-only evaluation (Part 15.4)
    curves         learning curves at 0, 1, 2, 4, ..., 64 informative interactions, censored N_q (Part 15.3)
    hstats         crossed world x learner-seed statistics, null calibration, preregistration (Part 15.7)
    instrument     protocols that classify what a learner did (memorisation ... transferable efficiency, Part 15)
    levels         the six generalization levels (Part 15.1) and the authorization-gated Level 6 adapter
    curriculum     weakness profiling and adaptive curriculum generation (Part 14.3)
    proposal       the world-proposal interface (Part 14.4)
    comparison     the A/B/C/D training-diversity benchmark with matched budgets and seven controls (Part 15.5)
    integrity      claim-integrity scanners and outcome-proxy checks (I.n2, K.n2, S.n1, 15.1, 15.2)
    compose        procedural structural composition of Tier 1 worlds (Part 7, Gate T 1-4)

Every environment a module here builds defaults to ``reward_mode="observation_only"`` (Clarification 2); learners
compute their own signals from what they observe.
"""
