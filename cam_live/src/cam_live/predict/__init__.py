"""cam_live.predict — prediction and decision-making (Workstream B).

Stage 1: tensor-level, model-independent modules — pure functions or ``nn.Module``s of the tensors defined in
``cam_live.nn.api``. Stage 2: the same modules wired onto ``CamLiveModel``. Public surface: ``cam_live/docs/B/API.md``.

    dists         predictive families, exact masked log-likelihoods, contracts.Distribution conversion
    world         WorldHeads (bootstrapped ensemble over outcome kinds), world_nll, LatentTransition
    imagination   latent rollouts flagged imagined=True, horizon controller, guards
    evaluator     GoalEvaluator (success / magnitude / time / resources), prefix targets, delta_v, action quality
    calibration   Brier, NLL, slope / intercept, reliability, ECE, grouped, with studentised intervals
    policy        CandidatePolicy, exploration rules with exact propensities, DecisionMaker (contracts.Decision,
                  self-proposed randomised experiments)
    voi           decision-relevant value of information, knowledge gradient, DownstreamValueEstimator
    registry      PredictionRegistry and ExperimentRegistry (append-only, hash-chained JSONL)
    candidates    strict ActionSpec parsing, JSON-Schema walker, data-only prompt builder, BackboneProposer

    heads         PredictConfig, attach_heads (registers B's heads on the model), head + fast-layer combination
    providers     LossProviders: world model, fast prior (ALPaCA-style), evaluator, and their adapter variants
    predictor     Predictor: evaluate_candidates (registered predictions), observe_outcomes, resolve
    planner       LatentPlanner: latent imagination over the measured-accurate horizon
    labdata       diagnostic-scale data collection and training utilities (LabWorld)
    diagnose      python -m cam_live.predict.diagnose -> docs/evidence/B/diagnostics.json
"""
