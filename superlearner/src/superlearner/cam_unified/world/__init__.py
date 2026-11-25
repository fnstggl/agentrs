"""Workstream C: predictive world model, imagination and planning (spec Parts 6, 7, 13.4).

outcomes.py      OutcomeHead (OutcomeHeadP) + MixedOutcomeDist (OutcomeDist) for every Likelihood
dynamics.py      WorldModel (WorldModelP): stochastic action-conditioned transitions on the shared backbone,
                 posterior/prior categorical latents with KL balancing, EMA latent targets, open-loop multistep loss
baselines.py     no-change / trend-persistence / key-mean / linear-regression forecasters + evaluate_multistep
imagination.py   belief-aware multistep imagination with per-branch FastState clones (Provenance.IMAGINED)
planner.py       policy-rollout, beam and CEM planners at a matched transition budget (ActionSource.CAM_PLANNER)
causal.py        causal provenance ledger/features for EVENT tokens; intervention-sensitive contrastive objective
imagine_train.py imagined lambda-return targets with uncertainty weighting / horizon shortening for workstream D
"""
from .baselines import (KeyMeanBaseline, LinearRegressionBaseline, NoChangeBaseline, TrendPersistenceBaseline,
                        WorldModelForecaster, evaluate_multistep)
from .causal import (PROV_FEATURES, CausalLedger, CausalProvenance, InterventionContrastive, provenance_features)
from .dynamics import WorldModel
from .imagination import (Candidates, ImaginationHooks, Imagination, PolicyContext, WorldImaginedStep,
                          WorldImaginedTrajectory, candidate_scores, trace_summaries)
from .imagine_train import (ImaginedTargets, combine_real_and_imagined, imagination_actor_critic_loss,
                            imagined_targets, select_reliable_horizon)
from .outcomes import MixedOutcomeDist, OutcomeHead, default_survival_edges
from .planner import BeamPlanner, PlanResult, PolicyRolloutPlanner, SamplingPlanner, compare_planners

__all__ = [
    "OutcomeHead", "MixedOutcomeDist", "default_survival_edges", "WorldModel", "NoChangeBaseline",
    "TrendPersistenceBaseline", "KeyMeanBaseline", "LinearRegressionBaseline", "WorldModelForecaster",
    "evaluate_multistep", "Imagination", "ImaginationHooks", "Candidates", "PolicyContext", "WorldImaginedStep",
    "WorldImaginedTrajectory", "candidate_scores", "trace_summaries", "PolicyRolloutPlanner", "BeamPlanner",
    "SamplingPlanner", "PlanResult", "compare_planners", "CausalLedger", "CausalProvenance",
    "InterventionContrastive", "PROV_FEATURES", "provenance_features", "imagined_targets", "ImaginedTargets",
    "imagination_actor_critic_loss", "combine_real_and_imagined", "select_reliable_horizon"]
