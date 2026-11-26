"""cam_live.reward — learned intrinsic reward and its lifetime meta-gradient (Workstream C).

    exact       exact tabular reference of the meta-gradient (diagnostics; ground truth for the estimators)
    meta        MetaGradient: differentiable inner update + verified-outcome outer surrogate at CAM scale
    signals     the named, versioned signal vector the reward generator consumes
    intrinsic   IntrinsicReward R_psi and LifetimeContext
    bootstrap   labelled cold-start signal and RewardMixer (bootstrap -> learned promotion on held-out evidence)
    windows     support / query windows cut from one lifetime, independent-outcome counting
    integrity   guards: psi isolation, scale invariance, de-duplication, signatures, noisy-TV, outcome ledger
    selfgoals   budgeted goal-free learning: goal scoring, budget, usefulness bookkeeping
    bandit_lifetime, evidence   diagnostic problem and the script that reproduces docs/evidence/C

Nothing here returns a reward to the environment or reads one from it; the only supervision of psi is the outer
objective on signed verifier outcomes. See docs/C/API.md.
"""
