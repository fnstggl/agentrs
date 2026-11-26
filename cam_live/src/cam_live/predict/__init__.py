"""cam_live.predict — prediction and decision-making (Workstream B).

Tensor-level, model-independent modules (stage 1). Every module is a pure function or ``nn.Module`` of the tensors
defined in ``cam_live.nn.api``; see ``cam_live/docs/B/API.md`` for the public surface.

    dists         predictive families, exact masked log-likelihoods, contracts.Distribution conversion
    world         WorldHeads (bootstrapped ensemble over outcome kinds), world_nll, LatentTransition
    imagination   latent rollouts flagged imagined=True, horizon controller, guards
    evaluator     GoalEvaluator (success / magnitude / time / resources), prefix targets, delta_v, action quality
    calibration   Brier, NLL, slope / intercept, reliability, ECE, grouped, with studentised intervals
    policy        CandidatePolicy and exploration rules with exact propensities
    voi           decision-relevant value of information, knowledge gradient, DownstreamValueEstimator
    registry      PredictionRegistry and ExperimentRegistry (append-only JSONL)
    candidates    strict ActionSpec parsing, JSON-Schema walker, data-only prompt builder
"""
