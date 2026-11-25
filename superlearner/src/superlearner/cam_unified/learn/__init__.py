"""Workstream D — learning and decision optimisation for CAM Unified.

Modules: `heads` (value, options, budget token), `returns` (SMDP GAE / lambda-returns / V-trace / option targets /
percentile normalisation), `ppo` (PPO-clip for fresh data with V-trace routing of stale data), `explore` (hindsight
learning progress with epistemic/aleatoric separation), `math_assist` (CAM 7.0 NIG / KG / EIG / Thompson scores and
baselines; optional inputs only in the declared math-assist condition), `meta` (support/query adaptation loss,
J_meta over multi-trial episodes, the episode harness), `trainer` (staged combined objective with per-objective
backbone gradient tracking), `curriculum` (learning-progress curriculum with diversity floor and split guard),
`diagnostics` (probe bandit, delayed reward, changing mechanism environments), `trajectory` (learner-side batches).
"""
from .curriculum import HeldOutFamilyError, LearningProgressCurriculum, SplitGuard
from .heads import BudgetEncoder, BudgetState, OptionModule, OptionStep, ValueHead, ValueOutput
from .ppo import ActorCriticObjective, StalePolicyError, approx_kl, ppo_clip_loss
from .returns import ReturnNormalizer, gae, lambda_returns, option_targets, semi_markov_discount, vtrace
from .trainer import STAGES, CombinedTrainer, Stage, UpdateInfo, attach_provenance
from .trajectory import Trajectory

__all__ = ["HeldOutFamilyError", "LearningProgressCurriculum", "SplitGuard", "BudgetEncoder", "BudgetState",
           "OptionModule", "OptionStep", "ValueHead", "ValueOutput", "ActorCriticObjective", "StalePolicyError",
           "approx_kl", "ppo_clip_loss", "ReturnNormalizer", "gae", "lambda_returns", "option_targets",
           "semi_markov_discount", "vtrace", "STAGES", "CombinedTrainer", "Stage", "UpdateInfo", "attach_provenance",
           "Trajectory"]
