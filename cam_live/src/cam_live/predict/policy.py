"""cam_live.predict.policy — candidate-set policy  pi(a_i | b, g, C_t)  and exploration rules (spec Part 5).

``CandidatePolicy`` scores a *variable* set of candidate actions from their decision latents ``z`` [B, N, D]
(belief + action + goal, produced by the shared trunk) **and** directly from the belief ``b`` [B, D] through a
multiplicative interaction, so the action distribution cannot be a function of the candidates alone
(:func:`latent_dependence` measures it; an earlier system had KL 4e-9 here — the language model decided
everything). It also returns a state value V [B] and a per-candidate value Q [B, N].

:func:`select` implements the exploration rules — policy sampling, Thompson sampling, UCB, greedy, information-gain
weighting and explicit randomised assignment over experiment arms. Every rule first builds the full probability
vector and then samples from exactly that vector, so the recorded propensities are exact, and an epsilon floor
keeps every feasible candidate in the support (needed by importance weighting and hindsight credit).

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
RULE_RANDOMIZED = "randomized"
RULES = (RULE_POLICY, RULE_THOMPSON, RULE_UCB, RULE_GREEDY, RULE_INFO_GAIN, RULE_RANDOMIZED)


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


def selection_probs(rule: str, *, a_mask: Tensor, logits: Optional[Tensor] = None, q_mean: Optional[Tensor] = None,
                    q_std: Optional[Tensor] = None, q_samples: Optional[Tensor] = None,
                    info_gain: Optional[Tensor] = None, arms: Optional[Tensor] = None,
                    arm_probs: Optional[Tensor] = None, eps_floor: float = 0.02, temperature: float = 1.0,
                    ucb_c: float = 1.0, ig_weight: float = 1.0) -> Tensor:
    """[B, N] float64 probability of selecting each candidate under ``rule``.

    policy      softmax(logits / temperature)
    thompson    share of posterior samples ``q_samples`` [S, B, N] in which the candidate is the best one
    ucb         all mass on argmax(q_mean + ucb_c * q_std)
    greedy      all mass on argmax(q_mean)  (``logits`` when ``q_mean`` is not given)
    voi         softmax((q_mean + ig_weight * info_gain) / temperature): decision-relevant information gain
                (``predict.voi``) traded against the learned value
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
                "voi": torch.tensor([[e.voi if e.voi is not None else 0.0 for e in evaluations]], dtype=f64),
                "value_source": "q_goal" if has_goal else "p_success"}

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
        return self._decision(evaluations, idx, [float(x) for x in p], mode, sc, objective_id=objective_id,
                              decision_id=decision_id, ts=ts, extra_diag={}, **fields)

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
