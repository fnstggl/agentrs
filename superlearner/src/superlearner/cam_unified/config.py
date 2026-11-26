"""CAM Unified configuration. Owned by the master architect; workstreams read it and request changes, they do not edit it.

Three scales with identical capabilities (spec Part 1.1). Every ablation switch an experiment needs is a field, so a
run is fully described by its config (`asdict(cfg)` is written next to every checkpoint).

`make_preset("dom" | "dom_dev" | "dom_150m")` gives the DOM configurations of the redesign plan 2026-11-26 §2 (the
1B Superlearner core with Qwen3-4B, its 8 x 384 development twin with Qwen3-0.6B, and a 16 x 768 scaling point).
Their fields were pre-declared by the seam commit with defaults that keep the pre-seam behaviour.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Optional

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

    # ---- DOM integrated runtime (redesign plan 2026-11-26 §2), pre-declared by the seam commit. The defaults keep
    # the pre-seam behaviour exactly (old configs and checkpoints describe the same system); the `dom*` presets
    # (`make_preset`) switch them on. [Wn] = the workstream that implements the switch; until it merges, a switch is
    # inert unless noted.
    preset: str = ""                  # name of the make_preset preset this config came from ("" = none)
    # core F_theta / reasoner [W3]
    state_gate: bool = False          # gated persistence Z = u*Z_new + (1-u)*Z_prev, u = sigmoid(W[Z_prev; Z_new] + b)
    state_emb_std: float = 0.5        # init std of the additive kind / state-slot / time embeddings (backbone emb_std)
    reason_iters_fixed: int = 0       # J > 0: fixed reasoning iterations (CamUnifiedModel.reason_iters defaults to it,
                                      # live now); 0 = adaptive halting / the model's reason_iters kwarg
    d_satellite: int = 0              # width of the satellite modules (entities, beliefs, heads); 0 = d_model
    # semantic stream and decoder conditioning [W4]
    value_channel: bool = False       # NUMERIC tokens = proj([norm(value features); norm(description + unit)])
    sem_lora: bool = True             # PEFT LoRA adapters on the semantic model's decoder path (QwenSemantic lora=)
    dec_alpha_init: float = 0.05      # initial gate alpha of the decoder latent cross-attention (tanh(alpha) scale)
    # decision bottleneck (Mechanism B) [W5]
    n_intent: int = 0                 # intent tokens I_t = IntentHead(R_t, ctx_out); 0 = no intent bottleneck
    n_candidates: int = 1             # K candidate actions per decision scored by decision/evaluator
    select_tau: float = 1.0           # selection k ~ softmax(Q / select_tau) over the K candidates
    veto_q: Optional[float] = None    # evaluator veto: candidates with Q_k < veto_q are rejected (None = no veto)
    # world model and beliefs [W6]
    wm_transition_blocks: int = 0     # prior Z_hat = WM.prior(R, A, B, dt) through the last k core blocks
                                      # (0 = the world model's own transition network)
    wm_ema_target: bool = True        # EMA target encoder for the latent targets (False = no EMA target)
    kl_free_bits_group: float = 0.0   # free bits per stochastic group (0 = kl_free_nats over the whole latent)
    wm_consistency: float = 0.0       # weight of L_cons = ||(Z_hat_{t+1} - sg(Z_post)) / sd||^2 (0 = off)
    burn_in: int = 0                  # replay burn-in decisions without gradient before the graded steps
    bptt_steps: int = 0               # graded replay decisions after the burn-in (0 = the agent's seq_len)
    # reward channels, labels and the integrated learner [W1 / W2 / W7 / W8]
    reward_mode: str = "legacy"       # "legacy": the verified return increment is the actor reward (pre-seam);
                                      # "observation_only": actor reward = R_psi (selfeval/reward_model.py), labels
                                      # reach the learner only; "sparse_verified": labelled baseline condition
    reward_in_beliefs: bool = True    # realised reward is an observed outcome of the belief / Kalman updates
                                      # (CamUnifiedModel.reward_in_beliefs defaults to it, live now)
    lambda_ig: float = 0.05           # initial information-gain weight of R_psi (a member of psi)
    lambda_c: float = 0.01            # initial resource-cost weight of R_psi (a member of psi)
    update_every: int = 8             # canonical loop (runtime.run_lifetime): learner.step() every k rounds
    meta_every: int = 0               # learner.meta_step() every k updates (0 = never)
    meta_inner_steps: int = 2         # functional inner steps on the fast set phi per meta step
    ppo_epochs: int = 1               # epochs over the fresh buffer per update
    target_kl: float = 0.0            # stop the policy epochs at this approximate KL (0 = off)
    tool_entropy_coef: float = 0.0    # entropy bonus of the tool-pointer distribution (0 = off)
    lr_core: float = 0.0              # backbone learning rate (0 = lr)
    lr_heads: float = 0.0             # heads, satellites and conditioning adapters (0 = lr)
    lr_psi: float = 1e-3              # reward-model parameters psi (separate AdamW)
    lr_warmup_updates: int = 0        # linear warmup of the core learning rate (0 = none)
    clip_mode: str = "global"         # "global": one norm clipped at grad_clip; "per_group": per parameter group at
                                      # clip_median_mult x its running median norm
    clip_median_mult: float = 5.0
    balance_every: int = 0            # recompute loss weights from the named-group gradient probe every k updates
    world_core_share_max: float = 1.0  # cap on the world objective's share of the core gradient (balancing target)
    microbatch: int = 0               # learner microbatch in decisions (0 = the whole batch)

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


# ============================================================================ DOM presets (plan 2026-11-26 §2)
# The switches every dom preset shares (dimensions and the semantic model are per preset).
DOM_SWITCHES = dict(
    state_gate=True, state_emb_std=0.02,
    sem_xattn_every=4, sem_xattn_max_tokens=256,
    dec_xattn_every=6, dec_alpha_init=0.0,
    n_intent=16, n_candidates=4, select_tau=1.0,
    value_channel=True,
    wm_transition_blocks=6, wm_ema_target=False, kl_free_bits_group=0.02, wm_consistency=1.0, burn_in=16,
    bptt_steps=8, train_horizons=(1, 2, 4, 8),
    reason_iters_fixed=2, max_reason_iters=32,
    use_imagination=False, use_memory=False,
    reward_mode="observation_only", reward_in_beliefs=False, meta_every=16, meta_inner_steps=2,
    sem_lora=False, sem_encoder_lora=False,
    ppo_epochs=2, target_kl=0.03,
    lr_core=3e-5, lr_heads=1e-4, lr_psi=1e-3, lr_warmup_updates=500,
    clip_mode="per_group", clip_median_mult=5.0, balance_every=25, world_core_share_max=0.45,
    microbatch=4, update_every=8,
)

# Per preset: the base scale (SCALES: layers x width x heads, state tokens), the semantic model and its hidden width,
# the context budget and the satellite width. dom_150m keeps dom's Qwen3-4B (a core-scaling point, not a new stack).
PRESETS = {
    "dom":      dict(scale="superlearner", sem_model="Qwen/Qwen3-4B", sem_xattn_dim=2560, max_ctx_tokens=512,
                     d_satellite=768),                    # 32 x 1536 x 24, S=128, SwiGLU d_ff 4096 (~1B core)
    "dom_dev":  dict(scale="dev", sem_model="Qwen/Qwen3-0.6B", sem_xattn_dim=1024, max_ctx_tokens=512,
                     d_satellite=0),                      # 8 x 384 x 6, S=32 (tests and development ablations)
    "dom_150m": dict(scale="research", sem_model="Qwen/Qwen3-4B", sem_xattn_dim=2560, max_ctx_tokens=512,
                     d_satellite=0),                      # 16 x 768 x 12, S=64 (scaling point / Condition H)
}


def make_preset(name: str, **over) -> CamConfig:
    """A named DOM configuration: `make_config(<base scale>)` + DOM_SWITCHES + the preset's own fields, then `over`.
    dom: 32 x 1536 x 24, 128 state tokens, d_ff 4096, Qwen3-4B, max_ctx_tokens 512 (the 1B Superlearner core);
    dom_dev: 8 x 384 with Qwen3-0.6B and the same switches; dom_150m: 16 x 768 with the same switches."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
    p = dict(PRESETS[name])
    scale = p.pop("scale")
    kw = {**DOM_SWITCHES, **p, "preset": name}
    kw.update(over)
    return make_config(scale, **kw)
