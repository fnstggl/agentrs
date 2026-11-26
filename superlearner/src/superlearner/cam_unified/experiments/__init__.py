"""CAM Unified foundational controlled experiments (spec Part 17 Experiments 1-3, Part 19 rigor; manifest X01-X03).

Everything here drives the *integrated* system: `stack.build_stack` assembles `model.build_model` over the real
workstream components (A semantic tokenizer + action generator over a frozen Qwen3 encoder, B core / entities /
reasoner / beliefs / Kalman layer, C outcome head + world model, D heads + `CombinedTrainer`, E archive / replay /
memory index / consolidator / `CheckpointOps`) and wraps it in `agent.CamUnifiedAgent`; `mechrun` runs that agent on
real Crucible hidden-mechanism worlds through G's `MechWorldAdapter` with verified returns from `MechWorldEvaluator`.

Modules:

    stack       real-component stack, run-directory layout, weights-only / full-lifetime / transfer-condition loads
    mechrun     task streams (SplitGuard-checked), the SMDP episode driver, operator-side condition wrappers
                (predecessor notes = condition C, hidden gauges + delayed money reports = condition D), counterfactual
                forks for evaluation targets (evaluator side only)
    predict     support/query prediction harness (condition A/C/D): identical support for every learner, strict
                query separation, proper scores and decision regret from the same queries
    baselines   predictive baselines on identical support (exact / meta-learned NIG regression, recurrent and
                attention meta-learners) and the Crucible numeric learners for agent-selected decisions
    stats       two-level bootstrap over training seeds and evaluation worlds
    train       chunked, resumable CAM Unified training on a guarded task stream (full-lifetime checkpoints)
    exp1/exp2/exp3   the experiment drivers (CLI); results go to runs/cam_unified/wave2/X/, compact summaries to
                superlearner/docs/experiments/

Nothing here imports `superlearner.cam_unified.testing` or `integration_doubles`.
"""
