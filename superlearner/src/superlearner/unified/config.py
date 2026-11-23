"""Configuration for CAM 6.1. Every capacity is independently scalable; none is a hard ceiling."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class UnifiedConfig:
    # Representation widths
    d_model: int = 512
    obs_layers: int = 6
    obs_heads: int = 8
    obs_latent_tokens: int = 32
    deter_dim: int = 1024
    stoch_vars: int = 32
    stoch_classes: int = 32
    memory_slots: int = 32
    memory_dim: int = 512
    action_dim: int = 512
    goal_dim: int = 512

    # Feature switches used by the ablations. Disabled parts are absent from the computation.
    use_recurrence: bool = True
    use_slots: bool = True
    use_world_model_loss: bool = True
    use_imagination: bool = True
    use_generator: bool = True
    train_generator: bool = True
    shared_policy_gradients: bool = False   # ablation: let actor and critic losses reach the belief layers

    # Imagination safety: only used when held-out reward prediction clears the bar
    imagination_min_r2: float = 0.2
    imagination_weight: float = 1.0
    kappa: float = 0.5
    reward_heads: int = 3
    overshoot: int = 3
    overshoot_weight: float = 0.2
    multi_horizons: tuple = (4, 16)
    generator_fraction: float = 0.5
    generator_lr: float = 3e-4
    target_encoder_tau: float = 0.01

    # Distribution constants, checked against danijar/dreamerv3 at e01491fad643 (configs.yaml, rssm.py)
    unimix: float = 0.01
    free_nats: float = 1.0
    beta_dyn: float = 1.0
    beta_rep: float = 0.1
    twohot_bins: int = 255
    twohot_low: float = -20.0
    twohot_high: float = 20.0

    # Control
    gamma_day: float = 0.997
    seconds_per_day: float = 86400.0
    lam: float = 0.95
    entropy_coef: float = 3e-4
    target_tau: float = 0.02
    return_norm_rate: float = 0.01
    return_norm_limit: float = 1.0
    actor_objective: str = "vtrace"         # "vtrace": sampled gradient; "expected": all-candidate gradient on twin Q
    use_effect_planner: bool = True         # steer decisions with the learned action-effect model once it is calibrated
    effect_heads: int = 3
    effect_weight: float = 1.0
    plan_beta: float = 8.0
    plan_min_r2: float = 0.05
    reward_scale: float = 10.0              # constant applied to environment reward before prediction and value
    eval_temperature: float = 1.0           # 0 selects the top-scoring candidate in evaluation
    awr_temperature: float = 1.0
    awr_max_weight: float = 20.0

    # Sequences and optimisation
    batch_size: int = 16
    seq_len: int = 32
    burn_in: int = 8
    imagination_horizon: int = 3
    candidates: int = 16
    lr_world: float = 1e-4
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip: float = 100.0
    updates_per_step: float = 0.25

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> "UnifiedConfig":
        return replace(self, **kw)


def small() -> UnifiedConfig:
    """Diagnostic scale: cheap enough for many seeds and ablations."""
    return UnifiedConfig(d_model=128, obs_layers=2, obs_heads=4, obs_latent_tokens=8, deter_dim=256,
                         stoch_vars=16, stoch_classes=16, memory_slots=8, memory_dim=128,
                         action_dim=128, goal_dim=128, lr_world=3e-4, lr_actor=3e-4, lr_critic=3e-4)


def medium() -> UnifiedConfig:
    """The specification's default research configuration."""
    return UnifiedConfig()


def large() -> UnifiedConfig:
    """Scaling configuration; used only when measurements justify it."""
    return UnifiedConfig(d_model=1024, obs_layers=8, obs_heads=16, obs_latent_tokens=64, deter_dim=4096,
                         stoch_vars=64, stoch_classes=32, memory_slots=128, memory_dim=1024,
                         action_dim=1024, goal_dim=1024)


SCALES = {"small": small, "medium": medium, "large": large}
