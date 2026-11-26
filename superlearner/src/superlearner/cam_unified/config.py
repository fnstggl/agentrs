"""CAM Unified configuration. Owned by the master architect; workstreams read it and request changes, they do not edit it.

Three scales with identical capabilities (spec Part 1.1). Every ablation switch an experiment needs is a field, so a
run is fully described by its config (`asdict(cfg)` is written next to every checkpoint).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

SCALES = {
    #        layers width heads state-tokens
    "dev":   (8, 384, 6, 32),
    "main":  (16, 768, 12, 64),
    "large": (24, 1024, 16, 128),
    # DOM amendment (2026-11-26): the main Superlearner core, ~1B trainable parameters (32 x 1536 x 24, 128 state
    # tokens, SwiGLU d_ff = 4096, gated semantic cross-attention every 2 blocks); "research" is the former main scale
    "superlearner": (32, 1536, 24, 128),
    "research": (16, 768, 12, 64),
    "advanced": (32, 2048, 32, 128),
}


@dataclass(frozen=True)
class CamConfig:
    scale: str = "dev"
    # ---- shared backbone F_theta (pre-norm RMSNorm, SwiGLU, SDPA attention, continuous-time encodings)
    n_layers: int = 8
    d_model: int = 384
    n_heads: int = 6
    n_state: int = 32                 # persistent latent state tokens Z_t: [B, n_state, d_model]
    ffn_mult: float = 8 / 3           # SwiGLU hidden = multiple of 64 nearest ffn_mult * d_model
    dropout: float = 0.0
    max_ctx_tokens: int = 384         # working-memory budget of context tokens per step (O, A, B, M, G); not a lifetime cap
    max_obs_tokens: int = 0           # observation-token share of that budget (0 = max_ctx_tokens // 2), so memory,
                                      # goal and belief tokens keep room
    grad_checkpoint: bool = False

    # ---- semantic backbone (pretrained, separate parameter count)
    sem_model: str = "Qwen/Qwen3-1.7B"
    sem_test_model: str = "Qwen/Qwen3-0.6B"   # unit/integration tests only
    sem_chunk_tokens: int = 512       # max tokens per encoded observation chunk (longer content: several chunks)
    sem_encoder_lora: bool = False    # Stage 1-3: encoder path uses base weights (hidden-state cache stays valid);
                                      # LoRA trains the decoder path only. Enabling it forces re-encoding (versioned).
    think_s: int = 600                # OpenWorld SMDP: simulated seconds charged per decision beyond the tool's cost_s
    max_wait_s: int = 86400           # longest agent-chosen wait (the 'wait' tool) per decision
    sem_tokens_per_obs: int = 4       # learned attention-pooling queries per semantic observation
    sem_lora_rank: int = 16
    sem_lora_alpha: int = 32
    sem_lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    # ---- DOM Mechanism C: CAM-controlled gated semantic cross-attention inside the core (core/semantic_xattn.py)
    sem_xattn_every: int = 0          # 0 = off; k = gated cross-attention to semantic token states after every k-th block
    sem_xattn_dim: int = 0            # hidden size of the semantic model (Qwen3-4B: 2560, 1.7B: 2048, 0.6B: 1024)
    sem_xattn_max_tokens: int = 512   # semantic tokens per step visible to the core's cross-attention
    sem_gate_bias: float = -3.0       # initial gate bias (sigmoid(-3) = 0.047: CAM starts mostly self-driven)
    # ---- DOM Mechanism A: multilayer gated latent conditioning of the semantic decoder (semantic/decoder_xattn.py)
    dec_xattn_every: int = 0          # 0 = off; k = gated cross-attention from decoder layer to CAM latent every k layers
    dec_latent_tokens: int = 32       # CAM latent/intent tokens exposed to the decoder per decision
    n_prefix: int = 16                # latent-to-language bridge prefix embeddings P_t = W(Z,G,B)
    max_evidence: int = 4             # observations CAM's evidence pointer may show the action decoder verbatim
    max_arg_tokens: int = 1536          # code.commit arguments measured at ~1,068 Qwen3 tokens

    # ---- numerical observation encoding
    n_fourier: int = 16
    max_entities: int = 128           # per-step working budget of entity tokens (archive keeps all)

    # ---- latent reasoning R_{j+1} = Reason(R_j, Z, B, M, G)
    reason_layers: int = 4            # last k backbone blocks reused (shared weights) per iteration
    reason_iters: tuple = (1, 2, 4, 8, 16)
    max_reason_iters: int = 16
    adaptive_reasoning: bool = True   # learned halting with compute penalty
    reason_compute_cost: float = 1e-3

    # ---- beliefs B_t = {(m^k, w^k, Sigma^k)}
    n_particles: int = 16
    d_particle: int = 256
    ess_frac: float = 0.5             # resample/rejuvenate when ESS < ess_frac * K
    misspec_z: float = 3.0            # predictive-surprise z-score that triggers uncertainty expansion
    misspec_window: int = 16
    misspec_arl: float = 1e5          # CUSUM target false-alarm average run length (misspec_z = design shift)
    bayes_feat: int = 32              # features of the Kalman/NIG last layer per outcome channel
    bayes_obs_var_init: float = 1.0
    bayes_prior_var_init: float = 1.0

    # ---- world model
    stoch_groups: int = 32            # categorical stochastic latent: groups x classes (straight-through)
    stoch_classes: int = 32
    kl_free_nats: float = 1.0
    kl_balance: float = 0.8
    ema_target: float = 0.995         # EMA target encoder for Z_target
    imag_horizons: tuple = (4, 8, 16)
    train_horizons: tuple = (1, 2, 4, 8, 16, 32)
    imag_branches: int = 8
    imag_samples: int = 4

    # ---- memory
    mem_top_k: int = 16
    mem_key_dim: int = 256
    mem_segment: int = 32             # events per compressed segment
    mem_segment_tokens: int = 4       # latent memory tokens per segment
    archive_chunk_rows: int = 65536
    archive_chunk_bytes: int = 32 * 2**20   # the archive tail is sealed at this many bytes as well as rows
    archive_max_bytes: int = 200 * 2**30   # explicit storage limit; exceeding it raises (never silently drops)

    # ---- policy / learning
    n_options: int = 16
    option_max_steps: int = 32
    kappa: float = 0.05               # semi-Markov discount gamma = exp(-kappa * dt_days)
    lr: float = 3e-4
    lr_sem: float = 1e-4              # LoRA / bridge
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    vtrace_rho: float = 1.0
    vtrace_c: float = 1.0
    entropy_coef: float = 3e-3
    value_coef: float = 0.5           # must equal loss_weights["value"] (learn.trainer raises when they disagree)
    lp_coef: float = 0.1              # learning-progress reward weight (learned exploration)
    loss_weights: dict = field(default_factory=lambda: {
        "world": 1.0, "adapt": 1.0, "actor": 1.0, "value": 0.5, "action": 1.0, "memory": 0.1, "retention": 0.1,
        "meta": 1.0, "semantic": 0.5})

    # ---- ablation switches (spec Part 19 baselines 8-14)
    continual_updates: bool = True
    use_fast_adaptation: bool = True
    use_imagination: bool = True
    use_learned_exploration: bool = True
    use_memory: bool = True
    use_reasoning: bool = True
    math_explore_assist: bool = False  # feed CAM 7.0-style KG/EIG scores as policy inputs (baseline condition)
    declared_conditions: tuple = ()    # experimental conditions in force, e.g. ("demo-init:example_inductor",)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def d_ff(self) -> int:
        return int(round(self.ffn_mult * self.d_model / 64)) * 64

    @property
    def obs_token_budget(self) -> int:
        return self.max_obs_tokens or self.max_ctx_tokens // 2

    @property
    def stoch_dim(self) -> int:
        return self.stoch_groups * self.stoch_classes


def make_config(scale: str = "dev", **over) -> CamConfig:
    layers, width, heads, state = SCALES[scale]
    d_particle = {"dev": 128, "main": 256, "large": 256, "superlearner": 256, "research": 256, "advanced": 256}[scale]
    base = CamConfig(scale=scale, n_layers=layers, d_model=width, n_heads=heads, n_state=state, d_particle=d_particle,
                     mem_key_dim=min(256, width))
    return replace(base, **over)
