"""CAM 7.0 configuration. Every switch an ablation needs is a field here, so a run is fully described by its config."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Cam7Config:
    # ---- shared representation (one transformer, used twice per decision: before and after the posterior)
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    event_window: int = 64          # recent event tokens
    id_dim: int = 16                # per-episode random identities of levers and gauges (no fixed slot meaning)
    dropout: float = 0.0

    # ---- adaptive features phi = F_theta(H, a, dt) and the fast Bayesian layer
    n_learned: int = 8              # learned features per lever (beside the 4 identified fixed ones)
    n_interact: int = 8             # width of the cross-lever interaction summary (non-additive effects)
    feat_hidden: int = 64
    use_fixed_features: bool = True     # [d, d^2, e, e^2]: hand-designed, identified as such; ablation: False
    use_learned_features: bool = True   # ablation: False gives the exact Bayesian baseline in this code path
    use_context_features: bool = True   # learned features see the backbone's lever context h_j; ablation: False
    ema: float = 0.5                # the moving average of settings behind e (delays), as in model_based
    forget_init: float = 0.999      # per-tick forgetting of old evidence (meta-learned)
    learned_prec_init: float = 10.0     # learned-feature prior precision, x that of the fixed linear terms
    prior_lr_mult: float = 10.0

    # ---- decisions
    dwell: int = 2                  # ticks between decisions
    grid: int = 9
    trust: float = 0.3
    n_thompson: int = 4
    n_random: int = 4
    kg_points: int = 32
    explore_rule: str = "kg"        # kg | thompson | greedy | eig | random | fixed_design | learned_only
    use_posterior_tokens: bool = True   # the backbone reads the posterior (integration); ablation: False
    use_residual: bool = True       # learned correction to the Bayesian decision score; ablation: False
    use_bayes_inputs: bool = True   # False with explore_rule learned_only: no posterior quantity reaches the policy
    eig_weight: float = 1.0
    kg_weight: float = 1.0          # weight on remaining decisions x knowledge gradient (1 = the textbook online KG)
    policy_temp: float = 1.0        # softmax temperature over candidate scores while exploring in training
    min_funds: float = 0.15         # same safety rule as model_based: below this, return to defaults and stop
    surprise: float = 4.0           # change-point rule (as model_based): recent squared error > surprise^2 x earlier
    surprise_window: int = 8

    # ---- training
    lr: float = 3e-4
    grad_clip: float = 1.0
    meta_batch: int = 32
    meta_ks: tuple = (0, 1, 2, 4, 8, 16, 32, 64)
    meta_query: int = 32
    meta_steps_per_episode: float = 0.5
    rl_batch: int = 256
    rl_steps_per_episode: float = 0.25
    rl_clip: float = 0.2
    gae_lambda: float = 0.9
    value_coef: float = 0.5
    entropy_coef: float = 0.003
    world_coef: float = 0.1
    rl_into_backbone: bool = True   # policy/value gradients reach the shared transformer (else only the heads)
    min_episodes: int = 48
    replay_episodes: int = 2048
    recent_episodes: int = 96       # decisions used by the actor-critic come from the latest episodes

    def with_(self, **kw) -> "Cam7Config":
        kw = {k: tuple(v) if isinstance(v, list) else v for k, v in kw.items()}
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


def small() -> Cam7Config:
    return Cam7Config(d_model=128, n_layers=2, n_heads=4)


def base() -> Cam7Config:
    return Cam7Config()


SCALES = {"small": small, "base": base}
