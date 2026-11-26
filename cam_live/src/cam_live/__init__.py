"""CAM Live 1.0 — a continually learning neural agent acting through a general-purpose computer environment.

Standalone: nothing in this package imports CAM Unified, Crucible or the older agentrs packages.

    contracts     shared records and service interfaces (master-owned)
    config        root configuration and scale presets (master-owned)
    nn/           semantic backbone + adapters, experience tokens, experiential core, fast Bayesian adaptation
    predict/      action-conditioned world model, goal-conditioned evaluator, policy, exploration, VOI, planning
    reward/       learned intrinsic reward, differentiable inner update, outer meta-gradient, integrity, self-goals
    credit/       delayed-outcome linking, RUDDER, COCOA, replay, continual learner, retention, checkpoints
    harness/      computer environment (browser, visual, sandbox, api, skills), authorization, staging services
    runtime/      persistent event-driven loop, verifier, logging, dashboard, CLI
    eval/         baselines and scientific evaluation
    diagnostics/  controlled worlds with known ground truth (mathematical tests only)
    agent         the integrated learner (implements contracts.AgentAPI)
"""
__version__ = "1.0.0"
