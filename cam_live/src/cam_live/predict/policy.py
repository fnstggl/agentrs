"""cam_live.predict.policy — candidate-set policy  pi(a_i | b, g, C_t)  and exploration rules (spec Part 5).

``CandidatePolicy`` scores a *variable* set of candidate actions from their decision latents ``z`` [B, N, D]
(belief + action + goal, produced by the shared trunk) **and** directly from the belief ``b`` [B, D] through a
multiplicative interaction, so the action distribution cannot be a function of the candidates alone
(:func:`latent_dependence` measures it; an earlier system had KL 4e-9 here — the language model decided
everything). It also returns a state value V [B] and a per-candidate value Q [B, N].

:func:`select` implements the exploration rules — policy sampling, Thompson sampling, UCB, greedy, information-gain
weighting, the combined learned-value rule ``thompson_policy`` and explicit randomised assignment over experiment
arms. Every rule first builds the full probability vector and then samples from exactly that vector, so the recorded
propensities are exact, and an epsilon floor keeps every feasible candidate in the support (needed by importance
weighting and hindsight credit).

``thompson_policy`` (spec 5.1 "Thompson sampling" + "learned action value", Part 6) is Thompson sampling over the
world model's joint posterior samples of each candidate's goal-relevant value (standardised within the decision),
corrected by the trainable policy's bounded preference with a weight lambda learned from verified outcomes:
:class:`SelectionCalibration` regresses the verified progress of executed decisions on what the agent knew when it
chose them (the world model's standardised value and the policy's preference of the chosen candidate), and one
(beta_q, beta_pi) is drawn from that posterior per Thompson sample, lambda = beta_pi^+ / beta_q. The policy's
influence therefore starts at its prior (a near-uniform, untrained policy has none whatever lambda is) and grows
only as its preferences prove predictive of verified progress beyond the world model's. Nothing in it names an
action or a task.

``functional_forward`` evaluates the policy with an explicit parameter dictionary (``torch.func.functional_call``)
so the reward workstream can differentiate through policy updates.

Nothing here consumes or produces a reward.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from dataclasses import replace as dataclasses_replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

RULE_POLICY = "policy"
RULE_THOMPSON = "thompson"
RULE_UCB = "ucb"
RULE_GREEDY = "greedy"
RULE_INFO_GAIN = "voi"
RULE_THOMPSON_POLICY = "thompson_policy"
RULE_RANDOMIZED = "randomized"
RULES = (RULE_POLICY, RULE_THOMPSON, RULE_UCB, RULE_GREEDY, RULE_INFO_GAIN, RULE_THOMPSON_POLICY, RULE_RANDOMIZED)


@dataclass
class PolicyConfig:
    d_model: int
    hidden: int = 192
    out_std: float = 0.05            # small random output init (never zero)


def masked_log_softmax(logits: Tensor, a_mask: Tensor) -> Tensor:
    if not bool(a_mask.any(-1).all()):
        raise ValueError("every decision needs at least one feasible candidate")
    neg = torch.finfo(logits.dtype).min
    return torch.log_softmax(torch.where(a_mask, logits, torch.full_like(logits, neg)), dim=-1)


class PolicyOutput:
    def __init__(self, logits: Tensor, a_mask: Tensor, value: Tensor, q: Tensor):
        self.a_mask = a_mask
        self.raw_logits = logits                                   # [B, N] unmasked scores
        self.log_probs = masked_log_softmax(logits, a_mask)        # [B, N]; ~ -inf at infeasible candidates
        self.value = value                                         # [B]
        self.q = torch.where(a_mask, q, torch.zeros_like(q))       # [B, N]

    @property
    def logits(self) -> Tensor:
        return torch.where(self.a_mask, self.raw_logits, torch.full_like(self.raw_logits, float("-inf")))

    @property
    def probs(self) -> Tensor:
        return torch.where(self.a_mask, torch.exp(self.log_probs), torch.zeros_like(self.log_probs))

    def log_prob(self, chosen: Tensor) -> Tensor:
        return self.log_probs.gather(-1, chosen.view(-1, 1)).squeeze(-1)

    def entropy(self) -> Tensor:
        p = self.probs
        return -(p * torch.where(self.a_mask, self.log_probs, torch.zeros_like(p))).sum(-1)


class CandidatePolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.in_z = nn.Linear(cfg.d_model, h)
        self.in_b = nn.Linear(cfg.d_model, h)
        self.in_zb = nn.Linear(cfg.d_model, h, bias=False)         # z (x) b interaction: the direct latent path
        self.in_bb = nn.Linear(cfg.d_model, h, bias=False)
        self.pool = nn.Linear(h, h)                                # summary of the alternatives on offer
        self.norm = nn.LayerNorm(h)
        self.mid = nn.Linear(h, h)
        self.logit_head = nn.Linear(h, 1)
        self.q_head = nn.Linear(h, 1)
        self.v_mid = nn.Linear(h, h)
        self.v_head = nn.Linear(h, 1)
        for head in (self.logit_head, self.q_head, self.v_head):
            nn.init.normal_(head.weight, std=cfg.out_std)
            nn.init.zeros_(head.bias)

    def forward(self, z: Tensor, a_mask: Tensor, b: Tensor) -> PolicyOutput:
        """z [B, N, D] decision latents, a_mask [B, N] (True = feasible candidate), b [B, D] belief."""
        hz = self.in_z(z)
        m = a_mask.unsqueeze(-1).to(hz.dtype)
        pooled = self.pool((hz * m).sum(1) / m.sum(1).clamp_min(1.0))                # [B, H], padding-invariant
        hb = self.in_b(b)
        inter = self.in_zb(z) * self.in_bb(b).unsqueeze(1) / math.sqrt(self.cfg.hidden)
        x = F.silu(self.mid(F.silu(self.norm(hz + hb.unsqueeze(1) + pooled.unsqueeze(1) + inter))))
        logits = self.logit_head(x).squeeze(-1)
        q = self.q_head(x).squeeze(-1)
        value = self.v_head(F.silu(self.v_mid(hb + pooled))).squeeze(-1)
        return PolicyOutput(logits, a_mask, value, q)

    def functional_forward(self, params: Dict[str, Tensor], z: Tensor, a_mask: Tensor, b: Tensor) -> PolicyOutput:
        """Same computation with explicit parameters (``dict(policy.named_parameters())``-shaped). Used to
        evaluate ``pi_{theta'}`` after a differentiable inner update without touching the module's own weights."""
        return torch.func.functional_call(self, params, (z, a_mask, b))

    def named_param_dict(self) -> Dict[str, Tensor]:
        return dict(self.named_parameters())

    ACTOR_PREFIXES = ("in_z.", "in_b.", "in_zb.", "in_bb.", "pool.", "norm.", "mid.", "logit_head.")

    def actor_param_dict(self) -> Dict[str, Tensor]:
        """The parameters the action distribution depends on (everything except the V and Q read-outs). A
        differentiable inner update of ``log pi`` touches exactly these; pass ``{**all, **updated_actor}`` to
        :meth:`functional_forward`."""
        return {k: v for k, v in self.named_parameters() if k.startswith(self.ACTOR_PREFIXES)}


# ----------------------------------------------------------------------------------------------------------------
# exploration rules with exact propensities
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Selection:
    index: Tensor                    # [B] chosen candidate
    probs: Tensor                    # [B, N] the distribution the choice was sampled from
    rule: str
    eps_floor: float
    diagnostics: Dict[str, Tensor] = field(default_factory=dict)

    @property
    def propensity(self) -> Tensor:
        return self.probs.gather(-1, self.index.view(-1, 1)).squeeze(-1)

    def probabilities(self, row: int = 0, n: Optional[int] = None) -> List[float]:
        """Plain list for ``contracts.Decision.probabilities`` (first ``n`` candidates of ``row``)."""
        p = self.probs[row].detach().to(torch.float64)
        return [float(x) for x in (p if n is None else p[:n])]


def apply_floor(p: Tensor, support: Tensor, eps: float) -> Tensor:
    """Mix with the uniform distribution over ``support``: every supported candidate keeps probability
    >= eps / |support|; unsupported candidates stay at exactly 0; rows sum to 1."""
    if not 0.0 <= eps < 1.0:
        raise ValueError("eps_floor must be in [0, 1)")
    sup = support.to(p.dtype)
    n = sup.sum(-1, keepdim=True)
    if bool((n == 0).any()):
        raise ValueError("every decision needs at least one candidate in the support")
    p = torch.where(support, p, torch.zeros_like(p))
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-300)
    out = (1.0 - eps) * p + eps * sup / n
    return out / out.sum(-1, keepdim=True)


def _one_hot_argmax(score: Tensor, support: Tensor) -> Tensor:
    s = torch.where(support, score, torch.full_like(score, float("-inf")))
    return F.one_hot(s.argmax(-1), score.shape[-1]).to(torch.float64)


def _tie_share(score: Tensor, support: Tensor) -> Tensor:
    """score [S, B, N] -> [B, N] float64: the mean over samples of the uniform distribution over each sample's
    maximal supported candidates (exact ties share the sample; an index never wins a tie by its position)."""
    sup = support.unsqueeze(0).expand_as(score)
    s = torch.where(sup, score.to(torch.float64), torch.full_like(score, float("-inf"), dtype=torch.float64))
    mx = s.max(-1, keepdim=True).values
    hit = sup & (s >= mx - 1e-12 * (1.0 + mx.abs()))
    return (hit.to(torch.float64) / hit.sum(-1, keepdim=True).clamp_min(1)).mean(0)


def centred_log_policy(logits: Tensor, a_mask: Tensor) -> Tensor:
    """log pi(a_i) - mean over the feasible candidates of log pi (0 at infeasible ones): the policy's preference in
    nats, independent of the size of the candidate set. A uniform policy gives exactly 0 everywhere."""
    lp = masked_log_softmax(logits.to(torch.float64), a_mask)
    m = a_mask.to(torch.float64)
    mean = (torch.where(a_mask, lp, torch.zeros_like(lp)) * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True)
    return torch.where(a_mask, lp - mean, torch.zeros_like(lp))


Z_CLIP, C_CLIP = 4.0, 4.0             # winsorisation of the calibration features (robust to exploding predictions)


def policy_preference(logits: Tensor, a_mask: Tensor, z_ref: Optional[Tensor] = None) -> Tensor:
    """The policy's preference used by ``thompson_policy``: the centred log-probability (:func:`centred_log_policy`)
    — with ``z_ref`` [B, N] (the world model's standardised predicted values of the same candidates) minus its least-
    squares projection on ``z_ref`` over the feasible candidates, i.e. only what the policy prefers *beyond* the world
    model's ranking — divided by max(1 nat, its spread). A near-uniform policy, or one that merely repeats the world
    model, has no say; a very peaked one is bounded like the standardised value, so a small calibration weight cannot
    hand it the decision by sheer peakedness. (Without the projection a policy trained toward the world model's
    choices is collinear with it and the calibration cannot tell the two apart.)"""
    c = centred_log_policy(logits, a_mask)
    m = a_mask.to(torch.float64)
    if z_ref is not None:
        z = torch.where(a_mask, z_ref.to(torch.float64), torch.zeros_like(c))
        zz = (z * z * m).sum(-1, keepdim=True)
        k = torch.where(zz > 1e-12, (c * z * m).sum(-1, keepdim=True) / zz.clamp_min(1e-12), torch.zeros_like(zz))
        c = torch.where(a_mask, c - k * z, torch.zeros_like(c))
    n = m.sum(-1, keepdim=True).clamp_min(1.0)
    sd = torch.sqrt((c ** 2 * m).sum(-1, keepdim=True) / n)
    return c / sd.clamp_min(1.0)


def standardised_values(q_samples: Tensor, q_mean: Tensor, a_mask: Tensor) -> Tensor:
    """[S, B, N]: each posterior sample of a candidate's net goal value, centred and scaled by the mean and spread of
    the *posterior means* over the feasible candidates of the same decision (so the sampled uncertainty is kept and
    the units of the prediction drop out). A set whose predicted values are all equal has spread 1."""
    m = a_mask.to(torch.float64)
    n = m.sum(-1, keepdim=True).clamp_min(1.0)
    qm = torch.where(a_mask, q_mean.to(torch.float64), torch.zeros_like(q_mean, dtype=torch.float64))
    mu = qm.sum(-1, keepdim=True) / n
    sd = torch.sqrt((torch.where(a_mask, (qm - mu) ** 2, torch.zeros_like(qm))).sum(-1, keepdim=True) / n)
    sd = torch.where(sd > 1e-12, sd, torch.ones_like(sd))
    z = (q_samples.to(torch.float64) - mu.unsqueeze(0)) / sd.unsqueeze(0)
    return torch.where(a_mask.unsqueeze(0), z, torch.zeros_like(z))


BETA_Q_MIN, LAMBDA_MAX = 0.05, 10.0
BETA_Q_FLOOR_IM = 1.0                # imagination: never amplified beyond its own coefficient (the prior mean of beta_q)


def policy_weight(coef_samples: Tensor) -> Tensor:
    """[S] the policy's weight relative to the world model in each Thompson sample:
    lambda = clip(beta_pi^+ / max(beta_q, BETA_Q_MIN), 0, LAMBDA_MAX). The world model's standardised value is the
    numeraire of the rule (Thompson sampling over its posterior is the base exploration); the calibration decides how
    far the policy may correct it. A policy whose preference does not predict verified progress (beta_pi <= 0) has
    weight 0; one that predicts it while the world model's ranking does not (beta_q ~ 0) can take over."""
    bq = coef_samples[:, 0].to(torch.float64).clamp_min(BETA_Q_MIN)
    bp = coef_samples[:, 1].to(torch.float64).clamp_min(0.0)
    return (bp / bq).clamp(0.0, LAMBDA_MAX)


def imagination_weight(coef_samples: Tensor) -> Tensor:
    """[S] weight of the imagined preference in each Thompson sample: beta_im^+ / max(beta_q, 1), clipped at
    LAMBDA_MAX. Unlike the policy's weight (:func:`policy_weight`, numeraire floor 0.05) the denominator is floored at
    the prior mean of beta_q: imagination keeps the weight its own coefficient earns and is never amplified because
    the one-step world model also predicts the verified multi-step progress weakly (measured in the first ablation:
    with beta_im ~ 0 and beta_q near the 0.05 floor, mean weights 0.8-1.2 let unsupported imagined differences
    decide). Both coefficients are in units of the verified scale, so beta_im ~ 1 means the imagined preference
    explains the verified progress as well as a fully predictive one-step value would."""
    bq = coef_samples[:, 0].to(torch.float64).clamp_min(BETA_Q_FLOOR_IM)
    bi = coef_samples[:, 1].to(torch.float64).clamp_min(0.0)
    return (bi / bq).clamp(0.0, LAMBDA_MAX)


def thompson_policy_scores(z_samples: Tensor, coef_samples: Tensor, log_policy: Tensor) -> Tensor:
    """[S, B, N] score of every candidate in each Thompson sample s:

        z[s] + lambda[s] * c                 lambda = :func:`policy_weight` of (beta_q[s], beta_pi[s])

    ``z_samples`` [S, B, N] the world model's joint posterior samples of the net goal value, standardised within the
    decision (:func:`standardised_values`), ``coef_samples`` [S, 2] draws of (beta_q, beta_pi) from
    :class:`SelectionCalibration`, ``log_policy`` [B, N] the policy's preference c of :func:`policy_preference`."""
    S = z_samples.shape[0]
    if coef_samples.shape != (S, 2):
        raise ValueError(f"coef_samples must be [{S}, 2]")
    lam = policy_weight(coef_samples).view(S, 1, 1)
    return z_samples.to(torch.float64) + lam * log_policy.to(torch.float64).unsqueeze(0)


class SelectionCalibration:
    """Learned action value of the ``thompson_policy`` rule: how much the world model's predicted goal value and the
    trainable policy's preference are worth in *verified* progress.

    For every executed decision whose verified progress ``y`` (the verifier-signed outcome attributed to it) is
    known, with ``z`` the chosen candidate's predicted net goal value standardised over the feasible candidates of
    that decision and ``c`` its preference under the policy that chose (:func:`policy_preference`; both recorded
    *before* acting, winsorised at +-4), it fits

        y / s = beta_0 + beta_q * z + beta_pi * c + noise,          s = recency-weighted RMS of the observed y

    by Bayesian linear regression (Gaussian prior beta ~ N((0, 1, 0), diag(1, sd_q^2, sd_pi^2)): a candidate the
    world model ranks higher is initially taken to be better and the policy is initially not trusted at all; noise
    variance by a scaled-inverse-chi^2 plug-in) with exponential forgetting, because both predictors keep changing as
    they are trained. ``sample`` draws (beta_q, beta_pi) for Thompson sampling; the rule uses their ratio
    lambda = beta_pi^+ / beta_q (:func:`policy_weight`), so the selection explores over how far the policy may correct
    the world model. Every quantity is relative to the decision's own candidate set, so the units of a
    prediction (and a prediction that explodes in scale) do not decide the weights; only its ranking does.

    No observation is invented: only decisions with a recorded prediction and a verified value enter, each once."""

    PRIOR_MEAN = (0.0, 1.0, 0.0)

    def __init__(self, *, prior_sd_q: float = 0.5, prior_sd_pi: float = 0.5, forget: float = 0.99,
                 max_obs: int = 1024, noise_dof: float = 2.0, obs_weight: float = 1.0):
        self.prior_sd_q, self.prior_sd_pi = float(prior_sd_q), float(prior_sd_pi)
        self.forget, self.max_obs, self.noise_dof = float(forget), int(max_obs), float(noise_dof)
        # weight of one observation in the likelihood (1 for independent observations; the imagination calibration
        # uses 1 / (h + 1) because its windows of h + 1 decisions overlap: an effective-sample-size correction)
        self.obs_weight = float(obs_weight)
        self.obs: List[Tuple[float, float, float]] = []          # (z, c, y) in arrival order
        self.keys: List[str] = []
        self._seen: set = set()
        self._post: Optional[Dict[str, Any]] = None

    # ---- data ---------------------------------------------------------------------------------------------------
    @staticmethod
    def features(decision: Any) -> Optional[Tuple[float, float]]:
        """(z, c) of a recorded ``contracts.Decision``: the chosen candidate's predicted net goal value
        (``extras["q_goal"] - extras["net_cost"]``) standardised over the feasible candidates, and its preference
        (:func:`policy_preference`) under the logged policy logits (``extras["policy_logit"]``, else
        ``CandidateEvaluation.value``). None when a feasible candidate carries no goal prediction."""
        cands = list(getattr(decision, "candidates", []) or [])
        i = int(getattr(decision, "chosen_index", -1))
        if not cands or not 0 <= i < len(cands):
            return None
        feas = [bool(c.feasible) for c in cands]
        feas[i] = True
        ex = [c.extras or {} for c in cands]
        if any(f and "q_goal" not in e for f, e in zip(feas, ex)):
            return None
        q = [float(e.get("q_goal", 0.0)) - float(e.get("net_cost", 0.0) or 0.0) for e in ex]
        lg = [float(e.get("policy_logit", c.value if c.value is not None else 0.0)) for c, e in zip(cands, ex)]
        if not all(math.isfinite(x) for x in q + lg):
            return None
        mask = torch.tensor([feas])
        qt = torch.tensor([q], dtype=torch.float64)
        zs = standardised_values(qt.unsqueeze(0), qt, mask)[0]
        z = zs[0, i]
        c = policy_preference(torch.tensor([lg], dtype=torch.float64), mask, zs)[0, i]
        return (max(-Z_CLIP, min(Z_CLIP, float(z))), max(-C_CLIP, min(C_CLIP, float(c))))

    @staticmethod
    def candidate_values(decision: Any) -> Optional[List[float]]:
        """The predicted net goal value of every candidate of a recorded decision, standardised over the feasible
        ones (0 at infeasible candidates); None when a feasible candidate carries no goal prediction."""
        cands = list(getattr(decision, "candidates", []) or [])
        if not cands:
            return None
        feas = [bool(c.feasible) for c in cands]
        if 0 <= int(getattr(decision, "chosen_index", -1)) < len(cands):
            feas[int(decision.chosen_index)] = True
        ex = [c.extras or {} for c in cands]
        if not any(feas) or any(f and "q_goal" not in e for f, e in zip(feas, ex)):
            return None
        q = [float(e.get("q_goal", 0.0)) - float(e.get("net_cost", 0.0) or 0.0) for e in ex]
        if not all(math.isfinite(x) for x in q):
            return None
        qt = torch.tensor([q], dtype=torch.float64)
        z = standardised_values(qt.unsqueeze(0), qt, torch.tensor([feas]))[0, 0]
        return [max(-Z_CLIP, min(Z_CLIP, float(x))) for x in z]

    def observe(self, key: str, z: float, c: float, y: float) -> bool:
        """Add one executed decision (``key``: its decision / experience id; a repeated key is ignored)."""
        if key in self._seen or not all(math.isfinite(float(v)) for v in (z, c, y)):
            return False
        self._seen.add(key)
        self.obs.append((max(-Z_CLIP, min(Z_CLIP, float(z))), max(-C_CLIP, min(C_CLIP, float(c))), float(y)))
        self.keys.append(key)
        if len(self.obs) > self.max_obs:
            self.obs.pop(0)
            self.keys.pop(0)
        self._post = None
        return True

    def observe_decision(self, key: str, decision: Any, y: float) -> bool:
        f = self.features(decision)
        return False if f is None else self.observe(key, f[0], f[1], y)

    def __len__(self) -> int:
        return len(self.obs)

    def predict(self, z: float) -> Optional[float]:
        """Calibrated expectation of the verified progress of a decision whose chosen candidate had standardised
        predicted value ``z`` (posterior mean, the policy term left out; beta_q clipped at 0), in verified units;
        None before the verified scale is known."""
        p = self.posterior()
        if p["scale"] is None:
            return None
        return p["scale"] * (p["mean"][0] + max(p["mean"][1], 0.0) * max(-Z_CLIP, min(Z_CLIP, float(z))))

    # ---- weights ------------------------------------------------------------------------------------------------
    def _weights(self) -> Tensor:
        n = len(self.obs)
        return torch.tensor([self.obs_weight * self.forget ** (n - 1 - k) for k in range(n)], dtype=torch.float64)

    @property
    def scale(self) -> Optional[float]:
        """Recency-weighted RMS of the observed verified values (None before two observations, one non-zero)."""
        ys = [o[2] for o in self.obs]
        if len(ys) < 2 or not any(y != 0.0 for y in ys):
            return None
        w = self._weights()
        return float(torch.sqrt((w * torch.tensor(ys, dtype=torch.float64) ** 2).sum() / w.sum()))

    # ---- posterior ----------------------------------------------------------------------------------------------
    def posterior(self) -> Dict[str, Any]:
        """{"mean": [beta_0, beta_q, beta_pi], "cov": 3x3, "noise_var", "n", "scale"} (y / scale units)."""
        if self._post is not None:
            return self._post
        m0 = torch.tensor(self.PRIOR_MEAN, dtype=torch.float64)
        P0 = torch.diag(torch.tensor([1.0, self.prior_sd_q ** -2, self.prior_sd_pi ** -2], dtype=torch.float64))
        s = self.scale
        if s is None:
            post = {"mean": m0.tolist(), "cov": torch.linalg.inv(P0).tolist(), "noise_var": 1.0, "n": len(self.obs),
                    "scale": None}
        else:
            o = torch.tensor(self.obs, dtype=torch.float64)
            X = torch.stack([torch.ones(len(o), dtype=torch.float64), o[:, 0], o[:, 1]], 1)
            y = o[:, 2] / s
            w = self._weights()
            nv = 1.0
            for _ in range(3):                                   # noise variance and coefficients, jointly
                A = P0 + (X * w[:, None]).T @ X / nv
                cov = torch.linalg.inv(A)
                mean = cov @ (P0 @ m0 + (X * w[:, None]).T @ y / nv)
                r = y - X @ mean
                nv = float((self.noise_dof * 1.0 + (w * r ** 2).sum()) / (self.noise_dof + w.sum()))
                nv = max(nv, 1e-6)
            post = {"mean": mean.tolist(), "cov": cov.tolist(), "noise_var": nv, "n": len(self.obs), "scale": s}
        self._post = post
        return post

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        """[n, 2] draws of (beta_q, beta_pi) from the joint posterior."""
        p = self.posterior()
        mu = torch.tensor(p["mean"][1:], dtype=torch.float64)
        cov = torch.tensor(p["cov"], dtype=torch.float64)[1:, 1:]
        L = torch.linalg.cholesky(cov + 1e-12 * torch.eye(2, dtype=torch.float64))
        z = torch.randn(n, 2, generator=generator, dtype=torch.float64)
        return mu + z @ L.T

    def summary(self) -> Dict[str, Any]:
        p = self.posterior()
        cov = p["cov"]
        return {"n": p["n"], "scale": p["scale"], "beta_q": p["mean"][1], "beta_q_sd": math.sqrt(cov[1][1]),
                "beta_pi": p["mean"][2], "beta_pi_sd": math.sqrt(cov[2][2]), "noise_var": p["noise_var"]}

    # ---- persistence --------------------------------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {"prior_sd_q": self.prior_sd_q, "prior_sd_pi": self.prior_sd_pi, "forget": self.forget,
                "max_obs": self.max_obs, "noise_dof": self.noise_dof, "obs": [list(o) for o in self.obs],
                "keys": list(self.keys), "seen": sorted(self._seen), "obs_weight": self.obs_weight}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.prior_sd_q, self.prior_sd_pi = float(d["prior_sd_q"]), float(d["prior_sd_pi"])
        self.forget, self.max_obs, self.noise_dof = float(d["forget"]), int(d["max_obs"]), float(d["noise_dof"])
        self.obs = [(float(a), float(b), float(c)) for a, b, c in d["obs"]]
        self.keys = list(d["keys"])
        self._seen = set(d.get("seen", self.keys))
        self.obs_weight = float(d.get("obs_weight", 1.0))
        self._post = None


CALIBRATION_ATTR = "selection_calibration"


def selection_calibration(policy: nn.Module) -> SelectionCalibration:
    """The one :class:`SelectionCalibration` of a policy head (created on first use and kept on the module, the way
    ``WorldHeads._real_scaler`` is shared): the ``DecisionMaker`` that selects with it and the learner that feeds it
    verified outcomes (``reward.metalearner.MetaLearner``) reach the same object through the policy head."""
    cal = getattr(policy, CALIBRATION_ATTR, None)
    if not isinstance(cal, SelectionCalibration):
        cal = SelectionCalibration()
        object.__setattr__(policy, CALIBRATION_ATTR, cal)
    return cal


# ----------------------------------------------------------------------------------------------------------------
# imagination in the selection rule (amendment 2026-11-28 item 2)
# ----------------------------------------------------------------------------------------------------------------
IM_MEAN = "im_cont"                  # extras: imagined continuation value of a candidate (mean over particles)
IM_SAMPLE = "imc_"                   # extras["imc_<p>"]: the same value in imagined particle p
IM_HORIZON = "im_horizon"            # extras: imagined transitions behind it (the horizon controller's allowance)


def imagined_preference(cont: Tensor, q_mean: Tensor, a_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """What imagination adds to the one-step world model, in the units of the selection rule.

    ``cont`` [P, B, N] the imagined continuation value of every candidate in each imagined particle (the value of the
    best imagined continuation after the candidate, :class:`cam_live.predict.planner.LatentPlanner`), ``q_mean``
    [B, N] the one-step predicted net values of the same candidates (the numbers ``z`` of the rule is standardised
    with). Returns ``(m_samples [P, B, N], m_mean [B, N])``: each particle's continuation values centred over the
    feasible candidates and divided by the spread of the *one-step* means — one numeraire for both terms, so that
    with weight 1 the rule ranks by the standardised imagined long-term value (one-step value + continuation) and a
    continuation difference as large as the one-step differences counts as much — winsorised at +-Z_CLIP, 0 at
    infeasible candidates. Equal continuations (an imagination that distinguishes nothing) give exactly 0."""
    m = a_mask.to(torch.float64)
    n = m.sum(-1, keepdim=True).clamp_min(1.0)
    qm = torch.where(a_mask, q_mean.to(torch.float64), torch.zeros_like(q_mean, dtype=torch.float64))
    mu = qm.sum(-1, keepdim=True) / n
    sd = torch.sqrt(torch.where(a_mask, (qm - mu) ** 2, torch.zeros_like(qm)).sum(-1, keepdim=True) / n)
    sd = torch.where(sd > 1e-12, sd, torch.ones_like(sd))
    c = torch.where(a_mask.unsqueeze(0), cont.to(torch.float64), torch.zeros_like(cont, dtype=torch.float64))

    def centre(x: Tensor) -> Tensor:                                                           # [..., B, N]
        mu_x = (x * m).sum(-1, keepdim=True) / n
        return torch.where(a_mask, (x - mu_x) / sd, torch.zeros_like(x)).clamp(-Z_CLIP, Z_CLIP)

    return centre(c), centre(c.mean(0))


def imagination_inputs(evaluations: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """The planner's per-candidate numbers recorded in ``CandidateEvaluation.extras`` (``IM_MEAN``, ``IM_SAMPLE<p>``,
    ``IM_HORIZON``) as tensors: {"cont": [P, 1, N], "horizon": h}; None unless every candidate carries them."""
    if not evaluations:
        return None
    ex0 = evaluations[0].extras or {}
    keys = sorted((k for k in ex0 if k.startswith(IM_SAMPLE)), key=lambda k: int(k[len(IM_SAMPLE):]))
    if not keys or any(IM_MEAN not in (e.extras or {}) or any(k not in e.extras for k in keys) for e in evaluations):
        return None
    try:
        cont = torch.tensor([[float(e.extras[k]) for e in evaluations] for k in keys], dtype=torch.float64)[:, None]
        h = int(round(float(ex0.get(IM_HORIZON, 1))))
    except (TypeError, ValueError):
        return None
    if not bool(torch.isfinite(cont).all()):
        return None
    return {"cont": cont, "horizon": max(1, h)}


class ImaginationCalibration:
    """How much the planner's imagined consequences are worth in *verified* outcomes, per imagined horizon h.

    For an executed decision planned with horizon h, ``z`` is the chosen candidate's standardised one-step value
    (as in :class:`SelectionCalibration`) and ``m`` its imagined preference (:func:`imagined_preference`, mean over
    particles), both recomputed from what the decision recorded *before* acting. The target is what the imagined
    rollout was about: the verified progress of that decision and the h decisions that followed it in the same
    objective, ``y_h = sum_{j=0..h} verified_progress(t + j)`` (signed verifier readings only; a decision without any
    goal-relevant question contributes 0). Per h a Bayesian linear regression (the machinery of
    :class:`SelectionCalibration`: prior beta ~ N((0, 1, 0), diag(1, sd_q^2, sd_im^2)), exponential forgetting)

        y_h / s = beta_0 + beta_q * z + beta_im * m + noise

    gives the imagination weight of the rule, lambda_im = beta_im^+ / max(beta_q, 1) (:func:`imagination_weight`),
    drawn per Thompson sample. Imagination therefore starts with no systematic weight (prior mean 0, the exploratory spread
    of its prior), gains weight only where its imagined differences predicted verified multi-step progress beyond the
    one-step model, and loses it again when they stop doing so. Consecutive windows share h of their h + 1 terms,
    so each observation has weight 1 / (h + 1) in the likelihood (an effective-sample-size correction: the evidence
    of n overlapping windows counts as that of n / (h + 1) independent ones; G3-core D-IMP3 measured a 1.5x too
    small posterior sd without it at h = 4 with an autocorrelated imagined preference). A new horizon starts from the prior: what was learned
    about h-step imagination says nothing about (h+1)-step imagination. No observation is invented: a window enters
    only when every decision in it has its final verified value, each decision once."""

    def __init__(self, *, prior_sd_q: float = 0.5, prior_sd_im: float = 0.5, forget: float = 0.99,
                 max_obs: int = 1024):
        self.prior_sd_q, self.prior_sd_im = float(prior_sd_q), float(prior_sd_im)
        self.forget, self.max_obs = float(forget), int(max_obs)
        self.by_h: Dict[int, SelectionCalibration] = {}
        self.pending: Dict[str, str] = {}            # exp id -> objective id, decisions waiting for their window
        self.closed: List[str] = []                  # objectives no later decision can belong to (newest last)

    def get(self, h: int) -> SelectionCalibration:
        h = max(1, int(h))
        if h not in self.by_h:
            self.by_h[h] = SelectionCalibration(prior_sd_q=self.prior_sd_q, prior_sd_pi=self.prior_sd_im,
                                                forget=self.forget, max_obs=self.max_obs, obs_weight=1.0 / (h + 1))
        return self.by_h[h]

    def __len__(self) -> int:
        return sum(len(c) for c in self.by_h.values())

    @staticmethod
    def features(decision: Any) -> Optional[Tuple[float, float, int]]:
        """(z, m, h) of a recorded ``contracts.Decision``, or None when it carries no imagination or no goal value."""
        zc = SelectionCalibration.features(decision)
        cands = list(getattr(decision, "candidates", []) or [])
        if zc is None or not cands:
            return None
        im = imagination_inputs(cands)
        if im is None:
            return None
        i = int(decision.chosen_index)
        feas = [bool(c.feasible) for c in cands]
        feas[i] = True
        q = [float((c.extras or {}).get("q_goal", 0.0)) - float((c.extras or {}).get("net_cost", 0.0) or 0.0)
             for c in cands]
        _, mm = imagined_preference(im["cont"], torch.tensor([q], dtype=torch.float64), torch.tensor([feas]))
        return zc[0], float(mm[0, i]), int(im["horizon"])

    def observe(self, key: str, h: int, z: float, m: float, y: float) -> bool:
        return self.get(h).observe(key, z, m, y)

    def sample(self, h: int, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        """[n, 2] draws of (beta_q, beta_im) of horizon ``h``."""
        return self.get(h).sample(n, generator)

    def summary(self, h: Optional[int] = None) -> Dict[str, Any]:
        if h is not None:
            s = self.get(h).summary()
            return {"horizon": int(h), "n": s["n"], "scale": s["scale"], "beta_q": s["beta_q"],
                    "beta_q_sd": s["beta_q_sd"], "beta_im": s["beta_pi"], "beta_im_sd": s["beta_pi_sd"],
                    "noise_var": s["noise_var"]}
        return {"by_horizon": {str(k): self.summary(k) for k in sorted(self.by_h)}, "pending": len(self.pending)}

    def state_dict(self) -> Dict[str, Any]:
        return {"prior_sd_q": self.prior_sd_q, "prior_sd_im": self.prior_sd_im, "forget": self.forget,
                "max_obs": self.max_obs, "by_h": {str(k): c.state_dict() for k, c in self.by_h.items()},
                "pending": dict(self.pending), "closed": list(self.closed)}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.prior_sd_q, self.prior_sd_im = float(d["prior_sd_q"]), float(d["prior_sd_im"])
        self.forget, self.max_obs = float(d["forget"]), int(d["max_obs"])
        self.by_h = {}
        for k, s in (d.get("by_h") or {}).items():
            c = SelectionCalibration()
            c.load_state_dict(s)
            c.obs_weight = 1.0 / (int(k) + 1)        # also for states saved before the overlap correction (G3 D-IMP3)
            c._post = None
            self.by_h[int(k)] = c
        self.pending = {str(k): str(v) for k, v in (d.get("pending") or {}).items()}
        self.closed = list(d.get("closed") or [])


IMAGINATION_ATTR = "imagination_calibration"


def imagination_calibration(transition: nn.Module) -> ImaginationCalibration:
    """The one :class:`ImaginationCalibration` of a model's latent transition head (created on first use and kept on
    the module): the ``LatentPlanner`` that feeds it with verified windows and the ``DecisionMaker`` that selects
    with it reach the same object."""
    cal = getattr(transition, IMAGINATION_ATTR, None)
    if not isinstance(cal, ImaginationCalibration):
        cal = ImaginationCalibration()
        object.__setattr__(transition, IMAGINATION_ATTR, cal)
    return cal


def selection_probs(rule: str, *, a_mask: Tensor, logits: Optional[Tensor] = None, q_mean: Optional[Tensor] = None,
                    q_std: Optional[Tensor] = None, q_samples: Optional[Tensor] = None,
                    info_gain: Optional[Tensor] = None, arms: Optional[Tensor] = None,
                    arm_probs: Optional[Tensor] = None, eps_floor: float = 0.02, temperature: float = 1.0,
                    ucb_c: float = 1.0, ig_weight: float = 1.0, coef_samples: Optional[Tensor] = None) -> Tensor:
    """[B, N] float64 probability of selecting each candidate under ``rule``.

    policy      softmax(logits / temperature)
    thompson    share of posterior samples ``q_samples`` [S, B, N] in which the candidate is the best one
    ucb         all mass on argmax(q_mean + ucb_c * q_std)
    greedy      all mass on argmax(q_mean)  (``logits`` when ``q_mean`` is not given)
    voi         softmax((q_mean + ig_weight * info_gain) / temperature): decision-relevant information gain
                (``predict.voi``) traded against the learned value
    thompson_policy
                share of samples s in which the candidate maximises :func:`thompson_policy_scores` — the posterior
                samples ``q_samples`` (net of cost) standardised by the spread of ``q_mean`` over the candidates,
                weighted by beta_q[s], plus beta_pi[s] x the centred log-probability of the policy ``logits``
                (``coef_samples`` [S, 2] from :class:`SelectionCalibration`; exact ties share a sample)
    randomized  explicit assignment over experiment ``arms`` [B, N] bool with ``arm_probs`` (uniform by default);
                candidates that are not arms get probability 0

    Each is then mixed with ``eps_floor`` of the uniform distribution over its support (feasible candidates, or
    the arms for ``randomized``).
    """
    support = a_mask
    if rule == RULE_POLICY:
        if logits is None:
            raise ValueError("rule 'policy' needs logits")
        lg = torch.where(a_mask, logits.to(torch.float64) / temperature, torch.full_like(logits, float("-inf"),
                                                                                         dtype=torch.float64))
        p = torch.softmax(lg, -1)
    elif rule == RULE_THOMPSON:
        if q_samples is None:
            raise ValueError("rule 'thompson' needs q_samples [S, B, N]")
        s = torch.where(a_mask.unsqueeze(0), q_samples, torch.full_like(q_samples, float("-inf")))
        p = F.one_hot(s.argmax(-1), q_samples.shape[-1]).to(torch.float64).mean(0)
    elif rule == RULE_UCB:
        if q_mean is None or q_std is None:
            raise ValueError("rule 'ucb' needs q_mean and q_std")
        p = _one_hot_argmax(q_mean + ucb_c * q_std, a_mask)
    elif rule == RULE_GREEDY:
        score = q_mean if q_mean is not None else logits
        if score is None:
            raise ValueError("rule 'greedy' needs q_mean or logits")
        p = _one_hot_argmax(score, a_mask)
    elif rule == RULE_INFO_GAIN:
        if info_gain is None:
            raise ValueError("rule 'voi' needs info_gain")
        base = torch.zeros_like(info_gain) if q_mean is None else q_mean
        sc = ((base + ig_weight * info_gain) / temperature).to(torch.float64)
        p = torch.softmax(torch.where(a_mask, sc, torch.full_like(sc, float("-inf"))), -1)
    elif rule == RULE_THOMPSON_POLICY:
        if q_samples is None or logits is None or coef_samples is None:
            raise ValueError("rule 'thompson_policy' needs q_samples [S, B, N], logits and coef_samples [S, 2]")
        qm = q_mean if q_mean is not None else q_samples.mean(0)
        score = thompson_policy_scores(standardised_values(q_samples, qm, a_mask), coef_samples,
                                       policy_preference(logits, a_mask, standardised_values(qm[None], qm, a_mask)[0]))
        p = _tie_share(score, a_mask)
    elif rule == RULE_RANDOMIZED:
        if arms is None:
            raise ValueError("rule 'randomized' needs arms")
        support = arms & a_mask
        if bool((arms & ~a_mask).any()):
            raise ValueError("an experiment arm must be a feasible candidate")
        p = support.to(torch.float64) if arm_probs is None else arm_probs.to(torch.float64)
        if arm_probs is not None and bool(((arm_probs > 0) & ~support).any()):
            raise ValueError("arm_probs puts mass outside the arms")
    else:
        raise ValueError(f"unknown selection rule {rule!r}; choose from {RULES}")
    return apply_floor(p.to(torch.float64), support, eps_floor)


def select(rule: str, *, a_mask: Tensor, generator: Optional[torch.Generator] = None, **kw) -> Selection:
    """Choose one candidate per decision and return the exact probability vector it was drawn from."""
    probs = selection_probs(rule, a_mask=a_mask, **kw)
    idx = torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(-1).to(probs.device)
    diag = {"entropy": -(probs * torch.log(probs.clamp_min(1e-300))).sum(-1),
            "n_support": (probs > 0).sum(-1)}
    return Selection(idx, probs, rule, float(kw.get("eps_floor", 0.02)), diag)


# ----------------------------------------------------------------------------------------------------------------
# does the policy depend on CAM's latent?
# ----------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def latent_dependence(policy: CandidatePolicy, z: Tensor, a_mask: Tensor, b: Tensor, *,
                      z_fn: Optional[Callable[[Tensor], Tensor]] = None, perm: Optional[Tensor] = None
                      ) -> Dict[str, float]:
    """Mean KL(pi(.|b) || pi(.|b permuted across the batch)) and total variation, candidates held fixed.

    ``z_fn(b) -> z`` recomputes the decision latents for the permuted beliefs (pass the model's
    ``decision_latent`` closed over e_a and e_g); without it only the policy's direct belief path is tested.
    A value near zero means the choice is decided by the candidates alone."""
    if perm is None:
        perm = torch.roll(torch.arange(b.shape[0], device=b.device), 1)
    base = policy(z if z_fn is None else z_fn(b), a_mask, b)
    bp = b[perm]
    other = policy(z if z_fn is None else z_fn(bp), a_mask, bp)
    p, q = base.probs.double(), other.probs.double()
    kl = (p * (torch.log(p.clamp_min(1e-300)) - torch.log(q.clamp_min(1e-300)))).sum(-1)
    return {"kl": float(kl.mean()), "tv": float(0.5 * (p - q).abs().sum(-1).mean()),
            "argmax_changed": float((p.argmax(-1) != q.argmax(-1)).double().mean())}


# ----------------------------------------------------------------------------------------------------------------
# stage 2: decisions on the real model
# ----------------------------------------------------------------------------------------------------------------
class DecisionMaker:
    """Turns ``CandidateEvaluation``s (from :class:`cam_live.predict.predictor.Predictor`) into a
    ``contracts.Decision`` with the exact probability of every candidate, the exploration rule, a rationale
    assembled from the recorded numbers and diagnostics.

    The numbers used by each rule (``predict.policy.RULES``):

        policy      the candidate policy's logits (``CandidateEvaluation.value``)
        greedy      the learned value: predicted goal-relevant outcome net of weighted cost (``extras["q_goal"]``),
                    or the evaluator's success probability when no goal-relevant outcome was predicted
        ucb         that value plus ``ucb_c`` epistemic standard deviations
        thompson    joint posterior samples of that value (``extras["ts_<i>"]``: one ensemble member and one draw
                    of the fast-layer weights per sample)
        voi         softmax of value + ``ig_weight`` x decision-relevant value of information
        thompson_policy
                    the Thompson samples of that value standardised over the candidate set and the policy's
                    centred log-probabilities (``extras["policy_logit"]``), weighted by draws from
                    :attr:`calibration` (the policy head's :class:`SelectionCalibration`)
        randomized  explicit assignment over experiment arms (:meth:`select_experiment`)
    """

    def __init__(self, model: Any, cfg: Any = None, *, heads: Any = None, clock: Any = None,
                 experiments: Any = None, seed: int = 0):
        from .heads import PredictConfig, attach_heads               # heads imports this module
        self.model = model
        pcfg = cfg if isinstance(cfg, PredictConfig) else PredictConfig.from_cfg(cfg or getattr(model, "cfg", None))
        self.heads = heads or attach_heads(model, pcfg)
        self.cfg = self.heads.cfg
        self.clock = clock
        self.experiments = experiments
        self.generator = torch.Generator().manual_seed(int(seed))
        self.calibration = selection_calibration(self.heads.policy)
        tr = getattr(self.heads, "transition", None)
        self.imagination = imagination_calibration(tr) if tr is not None else ImaginationCalibration()

    # ---- numbers ------------------------------------------------------------------------------------------------
    def scores(self, evaluations: Sequence[Any]) -> Dict[str, Any]:
        n = len(evaluations)
        if n == 0:
            raise ValueError("a decision needs at least one candidate")
        f64 = torch.float64
        a_mask = torch.tensor([[bool(e.feasible) for e in evaluations]])
        if not bool(a_mask.any()):
            raise ValueError("no feasible candidate")
        has_goal = any("q_goal" in e.extras for e in evaluations)
        cost = torch.tensor([[e.extras.get("net_cost", 0.0) for e in evaluations]], dtype=f64)
        if has_goal:
            q = torch.tensor([[e.extras.get("q_goal", 0.0) for e in evaluations]], dtype=f64) - cost
            sd = torch.tensor([[e.extras.get("q_goal_epistemic_std", 0.0) for e in evaluations]], dtype=f64)
        else:
            q = torch.tensor([[e.p_success if e.p_success is not None else 0.0 for e in evaluations]], dtype=f64) - cost
            sd = torch.tensor([[e.uncertainty or 0.0 for e in evaluations]], dtype=f64)
        keys = sorted((k for k in evaluations[0].extras if k.startswith("ts_")), key=lambda k: int(k[3:]))
        if keys and all(all(k in e.extras for k in keys) for e in evaluations):
            samples = torch.tensor([[e.extras[k] for e in evaluations] for k in keys], dtype=f64)[:, None, :] - cost
            sampled_from = "posterior"
        else:
            s = max(1, self.cfg.thompson_samples)
            samples = q[None] + sd[None] * torch.randn(s, 1, n, generator=self.generator, dtype=f64)
            sampled_from = "normal_marginals"
        return {"a_mask": a_mask, "q": q, "sd": sd, "samples": samples, "sampled_from": sampled_from,
                "logits": torch.tensor([[e.value if e.value is not None else 0.0 for e in evaluations]], dtype=f64),
                "policy_logits": torch.tensor([[e.extras.get("policy_logit", e.value if e.value is not None else 0.0)
                                                for e in evaluations]], dtype=f64),
                "cost": cost,
                "voi": torch.tensor([[e.voi if e.voi is not None else 0.0 for e in evaluations]], dtype=f64),
                "value_source": "q_goal" if has_goal else "p_success",
                "imagination": imagination_inputs(evaluations)}

    def thompson_policy(self, sc: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        """[1, N] probabilities of rule ``thompson_policy`` for the numbers ``sc`` of :meth:`scores`, and what was used
        (recorded in the decision's diagnostics): calibration summary, the policy's share of the decision (total
        variation between this distribution and the same rule with beta_pi = 0, before the floor) and the value-only
        distribution itself.

        When the candidates carry the planner's imagined continuation values (:func:`imagination_inputs`), every
        Thompson sample s adds ``lambda_im[s] * m[s mod P]`` — the imagined preference (:func:`imagined_preference`) of
        particle s mod P weighted by a draw from :attr:`imagination` (:class:`ImaginationCalibration` of the planned
        horizon). ``PredictConfig.imagination_select = False`` is the ablation "imagination weight 0": the same numbers
        are drawn, the distribution without imagination is used, and both are recorded (``imagination.share_tv``)."""
        a_mask = sc["a_mask"]
        z = standardised_values(sc["samples"], sc["q"], a_mask)
        S = z.shape[0]
        coef = self.calibration.sample(S, self.generator)
        lp = policy_preference(sc["policy_logits"], a_mask, standardised_values(sc["q"][None], sc["q"], a_mask)[0])
        base = thompson_policy_scores(z, coef, lp)
        full = _tie_share(base, a_mask)
        no_pi = coef.clone()
        no_pi[:, 1] = 0.0
        value_only = _tie_share(thompson_policy_scores(z, no_pi, lp), a_mask)
        lam = policy_weight(coef)
        diag = {"calibration": self.calibration.summary(),
                "policy_share_tv": float(0.5 * (full - value_only).abs().sum()),
                "policy_weight": {"mean": float(lam.mean()), "share_positive": float((lam > 0).double().mean())},
                "coef_positive_share": {"beta_q": float((coef[:, 0] > 0).double().mean()),
                                        "beta_pi": float((coef[:, 1] > 0).double().mean())},
                "value_only_probabilities": [float(x) for x in value_only[0]]}
        im = sc.get("imagination")
        if im is not None:
            # imagined consequences: Thompson sample s uses imagined particle s mod P and a draw (beta_q, beta_im)
            # of the imagination calibration of this horizon; both arms of the ablation draw the same numbers
            m_s, m_mean = imagined_preference(im["cont"], sc["q"], a_mask)
            P = m_s.shape[0]
            im_coef = self.imagination.sample(im["horizon"], S, self.generator)
            lam_im = imagination_weight(im_coef)
            with_im = _tie_share(base + lam_im.view(S, 1, 1) * m_s[torch.arange(S) % P], a_mask)
            on = bool(getattr(self.cfg, "imagination_select", True))
            diag["imagination"] = {
                "enabled": on, "horizon": int(im["horizon"]), "particles": int(P),
                "calibration": self.imagination.summary(im["horizon"]),
                "weight": {"mean": float(lam_im.mean()), "share_positive": float((lam_im > 0).double().mean())},
                "share_tv": float(0.5 * (with_im - full).abs().sum()),
                "preference": [float(x) for x in m_mean[0]],
                "without_probabilities": [float(x) for x in full[0]],
                "with_probabilities": [float(x) for x in with_im[0]]}
            if on:
                full = with_im
        return full, {"diag": diag, "log_policy": lp}

    @staticmethod
    def _decision_id(evaluations: Sequence[Any]) -> str:
        for e in evaluations:
            for p in e.predictions:
                if p.decision_id:
                    return p.decision_id
        from ..contracts import new_id
        return new_id("dec")

    def _now(self, ts: Optional[float]) -> float:
        if ts is not None:
            return float(ts)
        if self.clock is None:
            raise ValueError("pass ts= or construct the DecisionMaker with a clock")
        return float(self.clock.now())

    def _decision(self, evaluations: Sequence[Any], idx: int, probs: List[float], rule: str, sc: Dict[str, Any],
                  *, objective_id: str, decision_id: Optional[str], ts: Optional[float], extra_diag: Dict[str, Any],
                  experiment_id: str = "", **fields: Any) -> Any:
        from ..contracts import Decision
        e = evaluations[idx]
        q, sd = sc["q"][0], sc["sd"][0]
        feas = sc["a_mask"][0]
        best = int(torch.where(feas, q, torch.full_like(q, float("-inf"))).argmax())
        order = sorted((i for i in range(len(evaluations)) if bool(feas[i])), key=lambda i: -float(q[i]))[:3]
        ent = -sum(p * math.log(p) for p in probs if p > 0)
        diag = {"rule": rule, "eps_floor": self.cfg.eps_floor, "entropy": ent, "n_feasible": int(feas.sum()),
                "value_source": sc["value_source"], "thompson_sampled_from": sc["sampled_from"],
                "b_norm": e.extras.get("b_norm"), "z_norm_chosen": e.extras.get("z_norm"),
                "v_state": e.extras.get("v_state"), "value_chosen": float(q[idx]),
                "uncertainty_chosen": float(sd[idx]), "max_prob": max(probs),
                "top": [{"index": i, "operation": evaluations[i].action.operation, "value": float(q[i]),
                         "epistemic_std": float(sd[i]), "probability": probs[i],
                         "p_success": evaluations[i].p_success} for i in order]}
        diag.update(extra_diag)
        ps = "n/a" if e.p_success is None else f"{e.p_success:.3f}"
        voi = "n/a" if e.voi is None else f"{e.voi:.3g}"
        rationale = (f"{rule}: chose candidate {idx} ({e.action.operation}) with probability {probs[idx]:.3f} among "
                     f"{int(feas.sum())} feasible; {sc['value_source']} {float(q[idx]):.3g} (best {float(q[best]):.3g} "
                     f"at candidate {best}), epistemic sd {float(sd[idx]):.3g}, p_success {ps}, voi {voi}, "
                     f"estimated cost {e.cost.money:.3g} / {e.cost.seconds:.0f} s")
        chosen = e.action if not experiment_id else dataclasses_replace(e.action, experiment_id=experiment_id)
        return Decision(decision_id=decision_id or self._decision_id(evaluations), ts=self._now(ts),
                        objective_id=objective_id, chosen=chosen, candidates=list(evaluations),
                        probabilities=list(probs), chosen_index=idx, exploration=rule, rationale=rationale,
                        experiment_id=experiment_id, diagnostics=diag, **fields)

    # ---- rules --------------------------------------------------------------------------------------------------
    def probabilities(self, evaluations: Sequence[Any], mode: str = RULE_THOMPSON) -> Tuple[Tensor, Dict[str, Any]]:
        """[N] float64 probability of every candidate under ``mode`` (exactly what :meth:`select` samples from)."""
        if mode == RULE_RANDOMIZED:
            raise ValueError("randomised assignment goes through select_experiment()")
        sc = self.scores(evaluations)
        cfg = self.cfg
        if mode == RULE_THOMPSON_POLICY:
            raw, extra = self.thompson_policy(sc)
            p = apply_floor(raw, sc["a_mask"], cfg.eps_floor)
            sc["selection"] = extra["diag"]
            sc["log_policy"] = extra["log_policy"]
            return p[0], sc
        p = selection_probs(mode, a_mask=sc["a_mask"], logits=sc["logits"], q_mean=sc["q"], q_std=sc["sd"],
                            q_samples=sc["samples"], info_gain=sc["voi"], eps_floor=cfg.eps_floor,
                            temperature=cfg.temperature, ucb_c=cfg.ucb_c, ig_weight=cfg.ig_weight)
        return p[0], sc

    def select(self, evaluations: Sequence[Any], mode: str = RULE_THOMPSON, *, objective_id: str = "",
               decision_id: Optional[str] = None, ts: Optional[float] = None,
               generator: Optional[torch.Generator] = None, **fields: Any) -> Any:
        """Sample one candidate from the rule's probability vector and record that vector. ``fields`` are further
        ``contracts.Decision`` fields (``context_obs_ids``, ``model_version``, ``checkpoint_id``, ...)."""
        p, sc = self.probabilities(evaluations, mode)
        idx = int(torch.multinomial(p, 1, generator=generator or self.generator))
        extra: Dict[str, Any] = {}
        if "selection" in sc:
            sel = dict(sc["selection"])
            sel["policy_centred_logp_chosen"] = float(sc["log_policy"][0, idx])
            if isinstance(sel.get("imagination"), dict):
                im = dict(sel["imagination"])
                im["preference_chosen"] = im["preference"][idx]
                im["changed_choice_probability"] = float(im["with_probabilities"][idx] - im["without_probabilities"][idx])
                sel["imagination"] = im
            extra = {"selection": {"rule": mode, **sel}}
        return self._decision(evaluations, idx, [float(x) for x in p], mode, sc, objective_id=objective_id,
                              decision_id=decision_id, ts=ts, extra_diag=extra, **fields)

    # ---- experiments ------------------------------------------------------------------------------------------
    def propose_experiment(self, evaluations: Sequence[Any], top_k: int = 6) -> Optional[Dict[str, Any]]:
        """The comparison between two feasible candidates whose order the model is least sure of among the
        ``top_k`` by upper confidence bound: the pair maximising P(order flips) = Phi(-|dq| / sqrt(s_i^2 + s_j^2)).
        Returns keyword arguments for :meth:`select_experiment` (arms, question, hypothesis) or None when fewer
        than two candidates carry predictions. The texts only restate the recorded predictions."""
        sc = self.scores(evaluations)
        q, sd, feas = sc["q"][0], sc["sd"][0], sc["a_mask"][0]
        pool = [i for i in range(len(evaluations)) if bool(feas[i]) and evaluations[i].predictions]
        pool = sorted(pool, key=lambda i: -float(q[i] + self.cfg.ucb_c * sd[i]))[:top_k]
        best = None
        for a in range(len(pool)):
            for b in range(a + 1, len(pool)):
                i, j = pool[a], pool[b]
                s = math.sqrt(float(sd[i]) ** 2 + float(sd[j]) ** 2)
                flip = 0.5 * math.erfc(abs(float(q[i] - q[j])) / (s * math.sqrt(2.0))) if s > 0 else 0.0
                if best is None or flip > best[0]:
                    best = (flip, i, j)
        if best is None:
            return None
        flip, i, j = best
        if float(q[j]) > float(q[i]):
            i, j = j, i
        name = lambda k: f"{evaluations[k].action.operation}({json.dumps(evaluations[k].action.arguments, sort_keys=True)})"  # noqa: E731
        metric = next((p.query.description or p.query.metric for p in evaluations[i].predictions
                       if p.query.goal_relevant), evaluations[i].predictions[0].query.metric)
        return {"arms": {name(i): i, name(j): j},
                "question": f"Which of {name(i)} and {name(j)} gives more of: {metric}?",
                "hypothesis": (f"{name(i)} gives at least as much as {name(j)}: predicted {float(q[i]):.3g} "
                               f"(epistemic sd {float(sd[i]):.3g}) against {float(q[j]):.3g} "
                               f"(epistemic sd {float(sd[j]):.3g}); probability that the order is wrong {flip:.3f}"),
                "expected_flip_probability": flip}

    def select_experiment(self, evaluations: Sequence[Any], arms: Dict[str, int], *, question: str = "",
                          hypothesis: str = "", experiment_id: Optional[str] = None,
                          arm_probs: Optional[Sequence[float]] = None, horizon_s: Optional[float] = None,
                          objective_id: str = "", decision_id: Optional[str] = None, ts: Optional[float] = None,
                          unit_id: Optional[str] = None, evidence_refs: Sequence[str] = (),
                          expected_flip_probability: Optional[float] = None, **fields: Any) -> Tuple[Any, Any]:
        """Randomised controlled assignment between the candidates in ``arms`` (name -> candidate index).

        Without ``experiment_id`` a new ``ExperimentRecord`` is registered first — question, hypothesis, arms,
        assignment probabilities and fresh copies of the arms' predictions — and only then is the unit (this
        decision) assigned. With ``experiment_id`` the decision is a further unit of that running experiment.
        Returns ``(Decision, ExperimentRecord)``; the decision's probabilities are the assignment probabilities
        (zero for candidates that are not arms)."""
        from ..contracts import Cost, new_id
        if self.experiments is None:
            raise ValueError("this DecisionMaker was built without an ExperimentRegistry")
        names = list(arms)
        idxs = [int(arms[k]) for k in names]
        if len(set(idxs)) != len(idxs) or any(not evaluations[i].feasible for i in idxs):
            raise ValueError("experiment arms must be distinct feasible candidates")
        did = decision_id or self._decision_id(evaluations)
        if experiment_id is None:
            preds = []
            for i in idxs:
                for p in evaluations[i].predictions:
                    preds.append(dataclasses_replace(p, prediction_id=new_id("pred")))
            hz = horizon_s if horizon_s is not None else max((p.query.horizon_s for p in preds), default=0.0)
            cost = Cost(seconds=max(evaluations[i].cost.seconds for i in idxs),
                        money=max(evaluations[i].cost.money for i in idxs))
            rec = self.experiments.register(question=question, hypothesis=hypothesis, predictions=preds, arms=names,
                                            assignment="randomized", arm_probs=arm_probs,
                                            intervention=evaluations[idxs[0]].action.render(),
                                            comparison=evaluations[idxs[1]].action.render() if len(idxs) > 1 else "",
                                            horizon_s=float(hz), cost=cost, evidence_refs=list(evidence_refs))
            experiment_id = rec.experiment_id
        else:
            rec = self.experiments.get(experiment_id)
            if list(rec.arms) != names:
                raise ValueError(f"arms {names} do not match the registered arms {rec.arms}")
        asg = self.experiments.assign(experiment_id, unit_id or did)
        probs = [0.0] * len(evaluations)
        for k, i in enumerate(idxs):
            probs[i] = float(asg.probabilities[k])
        sc = self.scores(evaluations)
        dec = self._decision(evaluations, idxs[asg.arm_index], probs, RULE_RANDOMIZED, sc, objective_id=objective_id,
                             decision_id=did, ts=ts, experiment_id=experiment_id,
                             extra_diag={"arm": asg.arm, "arm_names": names, "eps_floor": 0.0,
                                         "expected_flip_probability": expected_flip_probability}, **fields)
        self.experiments.link_decision(experiment_id, dec.decision_id)
        return dec, self.experiments.get(experiment_id)
