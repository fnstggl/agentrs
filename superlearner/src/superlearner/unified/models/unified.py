"""The integrated CAM 6.1 network: one belief representation consumed by prediction, value, scoring and generation."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import UnifiedConfig
from ..core.adaptive import EpisodeRegression
from ..core.events import EventCore
from ..core.rssm import RSSM
from ..dist import make_bins, twohot_mean
from ..encoders.observation import ActionEncoder, GoalEncoder, ObservationEncoder, _encoder, mlp
from ..encoders.programs import BOS, DEFAULT, PAD, Grammar, Language
from ..interfaces import ActionCandidateBatch, BeliefState, GoalBatch, ObservationBatch, masked_log_softmax


ADAPT_FEATURES = 3       # per-candidate: predicted gain, its spread, one Thompson-sampled gain
EFFECT_SCALE = 2.0       # effects are standardised by the episode's reward variability, then spread on the grid


@dataclass(frozen=True)
class ObsLayout:
    n_gauges: int = 16
    n_levers: int = 12
    n_scalars: int = 5

    @property
    def n_ids(self) -> int:
        return self.n_gauges + self.n_levers + self.n_scalars

    @property
    def lever_id0(self) -> int:
        return self.n_gauges


def zero_last(m: nn.Sequential) -> nn.Sequential:
    """Start a distributional head at a uniform prediction, whose symmetric mean is exactly zero."""
    last = m[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return m


class CandidateScorer(nn.Module):
    """``F_policy(u, e_a, g)``: permutation-equivariant over a variable-size candidate set (no position input)."""

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        d = cfg.d_model
        self.pair = mlp(cfg.d_model + cfg.action_dim + cfg.goal_dim, d, d, layers=2)
        self.context = _encoder(d, max(1, min(cfg.obs_heads, 4)), 1)
        self.out = nn.Linear(d, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        # Starting prior: prefer small changes until evidence says otherwise. The strength is trained.
        self.move_cost = nn.Parameter(torch.tensor(2.0))

    def forward(self, u: Tensor, e: Tensor, g: Tensor, mask: Tensor) -> Tensor:
        k = e.shape[1]
        x = torch.cat([u.unsqueeze(1).expand(-1, k, -1), e, g.unsqueeze(1).expand(-1, k, -1)], -1)
        x = self.context(self.pair(x), src_key_padding_mask=~mask)
        return self.out(x).squeeze(-1)


class ProgramGenerator(nn.Module):
    """``P_psi(tokens | u, g)``: a trainable autoregressive decoder. Grammar masks make every sample parse."""

    def __init__(self, cfg: UnifiedConfig, lang: Language = DEFAULT):
        super().__init__()
        self.lang = lang
        d = cfg.d_model
        self.token = nn.Embedding(lang.vocab, d, padding_idx=PAD)
        self.ctx = mlp(cfg.d_model + cfg.goal_dim, d, d, layers=1)
        self.gru = nn.GRU(2 * d, d, batch_first=True)
        self.out = nn.Linear(d, lang.vocab)
        self.grammar = Grammar(lang)

    def _grammar(self, device) -> Grammar:
        if self.grammar.device != device:
            self.grammar = self.grammar.to(device)
        return self.grammar

    def log_prob(self, u: Tensor, g: Tensor, tokens: Tensor, lever_mask: Tensor) -> Tensor:
        """Log-probability of each program ``[B, L]`` under the grammar-constrained decoder."""
        gr = self._grammar(tokens.device)
        c = self.ctx(torch.cat([u, g], -1))
        inp = torch.cat([self.token(tokens[:, :-1]), c.unsqueeze(1).expand(-1, tokens.shape[1] - 1, -1)], -1)
        out, _ = self.gru(inp, c.unsqueeze(0))
        logits = self.out(out)
        state = torch.full((tokens.shape[0],), gr.ST_FIRST, device=tokens.device)
        n = torch.zeros_like(state)
        total = torch.zeros(tokens.shape[0], device=tokens.device, dtype=logits.dtype)
        for i in range(tokens.shape[1] - 1):
            allow = gr.mask(state, n, lever_mask)
            lp = masked_log_softmax(logits[:, i], allow)
            tgt = tokens[:, i + 1]
            total = total + lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            state, n = gr.step(state, n, tgt)
        return total

    @torch.no_grad()
    def sample(self, u: Tensor, g: Tensor, lever_mask: Tensor, temperature: float = 1.0) -> tuple[Tensor, Tensor]:
        """One program per row. Returns ``tokens [B, L]`` and their log-probabilities."""
        gr = self._grammar(u.device)
        b, L = u.shape[0], self.lang.max_len
        c = self.ctx(torch.cat([u, g], -1))
        tokens = torch.zeros(b, L, dtype=torch.long, device=u.device)
        tokens[:, 0] = BOS
        hidden = c.unsqueeze(0)
        state = torch.full((b,), gr.ST_FIRST, device=u.device)
        n = torch.zeros_like(state)
        total = torch.zeros(b, device=u.device, dtype=c.dtype)
        for i in range(L - 1):
            inp = torch.cat([self.token(tokens[:, i]), c], -1).unsqueeze(1)
            out, hidden = self.gru(inp, hidden)
            allow = gr.mask(state, n, lever_mask)
            # The last slot can only close the program.
            if i == L - 2:
                open_rows = state == gr.ST_NEXT
                allow = allow.clone()
                allow[open_rows] = False
                allow[open_rows, 2] = True
            lp = masked_log_softmax(self.out(out[:, 0]) / temperature, allow)
            nxt = torch.multinomial(lp.exp(), 1).squeeze(1)
            tokens[:, i + 1] = nxt
            total = total + lp.gather(1, nxt.unsqueeze(1)).squeeze(1)
            state, n = gr.step(state, n, nxt)
            if (state == gr.ST_DONE).all():
                break
        return tokens, total


class UnifiedModel(nn.Module):
    def __init__(self, cfg: UnifiedConfig, layout: ObsLayout = ObsLayout(), goal_features: int = 4,
                 lang: Language = DEFAULT, text_dim: int | None = None):
        super().__init__()
        self.cfg, self.layout, self.lang = cfg, layout, lang
        d = cfg.d_model
        self.obs_encoder = ObservationEncoder(cfg, layout.n_ids, text_dim)
        self.goal_encoder = GoalEncoder(cfg, goal_features)
        self.action_encoder = ActionEncoder(cfg, self.obs_encoder.id_embed, layout.lever_id0, lang, ADAPT_FEATURES)
        self.adapt = EpisodeRegression(layout.n_levers, cfg.adapt_k)
        self.rssm = RSSM(cfg)
        self.events = EventCore(cfg) if cfg.core == "attention" else None
        if cfg.core not in ("rssm", "attention"):
            raise ValueError(f"unknown core {cfg.core}")
        # Prediction heads
        self.obs_decoder = mlp(2 * d, d, 1, layers=2)
        self.decode_id = nn.Embedding(layout.n_ids, d)
        self.reward_head = zero_last(mlp(d, d, cfg.twohot_bins, layers=2))
        self.reward_ensemble = nn.ModuleList(zero_last(mlp(d, d // 2, cfg.twohot_bins, layers=1))
                                             for _ in range(cfg.reward_heads))
        self.continue_head = mlp(d, d, 1, layers=2)
        # Predicts the next observation's slow embedding from the state before that observation: the RSSM's
        # deterministic state, or for the attention core the previous belief.
        self.latent_pred = mlp(cfg.deter_dim if cfg.core == "rssm" else d, d, d, layers=2)
        self.multi_head = nn.ModuleList(zero_last(mlp(d + cfg.action_dim, d, cfg.twohot_bins, layers=2))
                                        for _ in cfg.multi_horizons)
        # Action-conditioned consequence model: change of reward per tick caused by a candidate program
        self.effect_heads = nn.ModuleList(zero_last(mlp(d + cfg.action_dim, d, cfg.twohot_bins, layers=2))
                                          for _ in range(cfg.effect_heads))
        # Decision heads
        self.policy = CandidateScorer(cfg)
        self.value = zero_last(mlp(d + cfg.goal_dim, d, cfg.twohot_bins, layers=2))
        self.q = zero_last(mlp(d + cfg.action_dim + cfg.goal_dim, d, cfg.twohot_bins, layers=2))
        self.q2 = zero_last(mlp(d + cfg.action_dim + cfg.goal_dim, d, cfg.twohot_bins, layers=2))
        self.generator = ProgramGenerator(cfg, lang)
        # Slow copies: never optimised directly
        self.value_target = copy.deepcopy(self.value).requires_grad_(False)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)
        self.obs_target = copy.deepcopy(self.obs_encoder).requires_grad_(False)
        self.register_buffer("bins", make_bins(cfg.twohot_bins, cfg.twohot_low, cfg.twohot_high), persistent=False)

    # ---------------------------------------------------------------- parameter ownership
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        core = self.events if self.cfg.core == "attention" else self.rssm
        world = [self.obs_encoder, self.goal_encoder, self.action_encoder, core, self.obs_decoder, self.decode_id,
                 self.reward_head, self.reward_ensemble, self.continue_head, self.latent_pred, self.multi_head, self.effect_heads]
        seen, wm = set(), []
        for m in world:
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    wm.append(p)
        return {"world": wm, "actor": list(self.policy.parameters()),
                "critic": list(self.value.parameters()) + list(self.q.parameters()) + list(self.q2.parameters()),
                "generator": list(self.generator.parameters()), "adapt": list(self.adapt.parameters())}

    def n_params(self) -> dict[str, int]:
        return {k: sum(p.numel() for p in v) for k, v in self.param_groups().items()}

    @torch.no_grad()
    def update_targets(self) -> None:
        tau, enc = self.cfg.target_tau, self.cfg.target_encoder_tau
        for src, dst, t in ((self.value, self.value_target, tau), (self.q, self.q_target, tau), (self.q2, self.q2_target, tau),
                            (self.obs_encoder, self.obs_target, enc)):
            for a, b in zip(src.parameters(), dst.parameters()):
                b.lerp_(a, t)

    # ---------------------------------------------------------------- the core forward API
    def encode_observation(self, obs: ObservationBatch) -> tuple[Tensor, Tensor]:
        return self.obs_encoder(obs)

    def encode_goal(self, goal: GoalBatch) -> Tensor:
        return self.goal_encoder(goal)

    def initial_belief(self, batch_size: int, device=None) -> BeliefState:
        return self.rssm.initial(batch_size, device or next(self.parameters()).device)

    def update_belief(self, previous: BeliefState, previous_action: Tensor, observation: ObservationBatch,
                      elapsed: Tensor, goal: Tensor, sample: bool = True) -> BeliefState:
        _, pooled = self.encode_observation(observation)
        return self.rssm.observe(previous, previous_action, elapsed, pooled, goal, sample)

    def predict_future(self, belief: BeliefState, action: Tensor, elapsed: Tensor, goal: Tensor) -> tuple[BeliefState, dict]:
        nxt = self.rssm.imagine(belief, action, elapsed, goal)
        return nxt, self.outcomes(nxt.u)

    def outcomes(self, u: Tensor) -> dict:
        members = torch.stack([twohot_mean(h(u), self.bins) for h in self.reward_ensemble], 0)
        return {"reward": twohot_mean(self.reward_head(u), self.bins),
                "continue": torch.sigmoid(self.continue_head(u)).squeeze(-1),
                "reward_std": members.std(0) if len(self.reward_ensemble) > 1 else torch.zeros_like(u[..., 0])}

    def decode_observation(self, u: Tensor) -> Tensor:
        """Predicted readings ``[..., N]`` for every observation variable."""
        ids = self.decode_id.weight
        x = torch.cat([u.unsqueeze(-2).expand(*u.shape[:-1], ids.shape[0], -1),
                       ids.expand(*u.shape[:-1], -1, -1)], -1)
        return self.obs_decoder(x).squeeze(-1)

    def effect_predictions(self, u: Tensor, e: Tensor) -> Tensor:
        """Predicted standardised change of reward per tick for each candidate: ``[heads, B, K]``."""
        x = torch.cat([u.unsqueeze(1).expand(-1, e.shape[1], -1), e], -1)
        return torch.stack([twohot_mean(h(x), self.bins) for h in self.effect_heads], 0) / EFFECT_SCALE

    def score_candidates(self, u: Tensor, goal: Tensor, e: Tensor, mask: Tensor, features: Tensor | None = None,
                         bonus: Tensor | None = None) -> Tensor:
        """Normalised log-probabilities ``[B, K]`` over valid candidates only."""
        logits = self.policy(u, e, goal, mask)
        if bonus is not None:
            logits = logits + bonus
        if features is not None:
            logits = logits - F.softplus(self.policy.move_cost) * 8.0 * features[..., self.lang.move_index]
        return masked_log_softmax(logits, mask)

    def estimate_value(self, u: Tensor, goal: Tensor, target: bool = False) -> Tensor:
        net = self.value_target if target else self.value
        return twohot_mean(net(torch.cat([u, goal], -1)), self.bins)

    def value_logits(self, u: Tensor, goal: Tensor) -> Tensor:
        return self.value(torch.cat([u, goal], -1))

    def q_logits(self, u: Tensor, e: Tensor, goal: Tensor, target: bool = False) -> Tensor:
        net = self.q_target if target else self.q
        return net(torch.cat([u, e, goal], -1))

    def q_pessimistic(self, u: Tensor, e: Tensor, goal: Tensor) -> Tensor:
        """The lower of the two slow Q estimates, so the actor does not chase one critic's overestimate."""
        x = torch.cat([u, e, goal], -1)
        return torch.minimum(twohot_mean(self.q_target(x), self.bins), twohot_mean(self.q2_target(x), self.bins))

    def generate_candidates(self, u: Tensor, goal: Tensor, lever_mask: Tensor, temperature: float = 1.0):
        return self.generator.sample(u, goal, lever_mask, temperature)

    def encode_actions(self, c: ActionCandidateBatch) -> Tensor:
        return self.action_encoder.encode_candidates(c)
