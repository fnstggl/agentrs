"""cam_live.reward.inner — the policy-improvement loss on intrinsic rewards (spec 7.3), on the real model.

    L_inner(theta; psi, D) = - mean_i log pi_theta(a_i | b_i, C_i) * A_i
                             + c_v * mean_i 1/2 (V_theta(b_i, C_i) - G_i)^2  -  c_e * mean_i H[pi_theta(. | b_i, C_i)]

    r_i = mix * r_bootstrap,i + (1 - mix) * R_psi(b_i, e_a_i, b'_i, e_g_i, signals_i, life_ctx_i)
    G_i = r_i + exp(-beta * (t_{i+1} - t_i)) * G_{i+1}        duration-aware discounting, within one lifetime
    A_i = standardise(G)_i - V_theta(b_i, C_i).detach()

``theta`` is the *adaptable subset*: the ``CandidatePolicy`` head and the model's decision trunk, passed as a
dictionary (``"policy.<name>"`` / ``"trunk.<name>"``); the latents ``b``, ``e_a``, ``e_g``, ``b_next`` are constants
(cached with ``encode_decisions(grad=False)``), so nothing here ever reaches the experiential core or the backbone
in the meta unroll.

One function, two uses — :meth:`PolicyImprovementLoss.functional`:

* ``loss(model, batch, store)`` (the ``LossProvider`` the continual learner iterates over) calls it with the live
  parameters and ``detach_reward=True``: R_psi is evaluated under ``no_grad``, so this loss has no gradient path
  into psi and can never move it.
* ``reward.meta.MetaGradient`` calls it with ``detach_reward=False`` on dictionaries: that is what the meta unroll
  differentiates.

Returns are standardised within the batch, so multiplying or shifting the reward output cannot change the update.
The per-group norm clip of the inner gradient (:class:`GroupClip`) is part of the differentiated function.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.func import functional_call

from ..contracts import ExperienceRecord, ExperienceStore
from ..learning_api import TIMESCALE_FREQUENT, DecisionBatch, encode_decisions
from .assemble import AssembledSignals, SignalAssembler
from .bootstrap import BootstrapReward, MixedReward, RewardMixer
from .integrity import scale_normalise
from .intrinsic import IntrinsicReward
from .meta import Params, discounted_returns

POLICY_PREFIX, TRUNK_PREFIX = "policy.", "trunk."
NORM_BUFFERS = tuple(f"{n}.{s}" for n in ("norm_b", "norm_a", "norm_g", "norm_s") for s in ("mean", "var"))

__all__ = ["InnerConfig", "AdaptableSubset", "InnerBatch", "GroupClip", "PolicyImprovementLoss", "POLICY_PREFIX",
           "TRUNK_PREFIX"]


@dataclass
class InnerConfig:
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    half_life_s: float = 86400.0     # gamma(dt) = 0.5 ** (dt / half_life_s);  <= 0 disables discounting
    normalise: bool = True
    sequence: bool = True            # recompute the lifetime context in-graph along each lifetime segment
    adapt_trunk: bool = True         # theta = policy head + decision trunk (False: policy head only)
    weight: float = 1.0

    @classmethod
    def from_cfg(cls, cfg: Any) -> "InnerConfig":
        sec = cfg.section("reward").get("inner", {}) if hasattr(cfg, "section") else {}
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in sec.items() if k in names})

    @property
    def beta(self) -> float:
        return math.log(2.0) / self.half_life_s if self.half_life_s > 0 else 0.0


class AdaptableSubset:
    """theta: the parameters the inner update may change — the policy head and (by default) the decision trunk."""

    def __init__(self, model: Any, policy: nn.Module, adapt_trunk: bool = True):
        self.model, self.policy = model, policy
        trunk = getattr(getattr(model, "exp", None), "trunk", None)
        self.trunk: Optional[nn.Module] = trunk if adapt_trunk else None

    def live(self) -> Dict[str, nn.Parameter]:
        out = {POLICY_PREFIX + n: p for n, p in self.policy.named_parameters()}
        if self.trunk is not None:
            out.update({TRUNK_PREFIX + n: p for n, p in self.trunk.named_parameters()})
        return out

    def snapshot(self, dtype: Optional[torch.dtype] = None) -> Params:
        return {n: p.detach().clone().to(dtype or p.dtype) for n, p in self.live().items()}

    @torch.no_grad()
    def write(self, theta: Params) -> None:
        live = self.live()
        if set(theta) != set(live):
            raise ValueError("theta does not match the adaptable subset")
        for n, p in live.items():
            p.copy_(theta[n].to(p.device, p.dtype))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.live().values())

    def groups(self) -> Dict[str, str]:
        """prefix -> parameter group of ``nn.api.GROUPS``."""
        return {POLICY_PREFIX: "heads", TRUNK_PREFIX: "experiential"}

    def forward(self, theta: Params, lat: "InnerBatch") -> Any:
        """``PolicyOutput`` of pi_theta on cached latents."""
        pol = {k[len(POLICY_PREFIX):]: v for k, v in theta.items() if k.startswith(POLICY_PREFIX)}
        if self.trunk is not None:
            tr = {k[len(TRUNK_PREFIX):]: v for k, v in theta.items() if k.startswith(TRUNK_PREFIX)}
            z = functional_call(self.trunk, tr, (lat.b, lat.e_a, lat.e_g))
        else:
            z = lat.z
        return functional_call(self.policy, pol, (z, lat.a_mask, lat.b))


class GroupClip:
    """Per-group norm clip of the inner gradient, kept in the graph: g <- g * min(1, c / ||g||) for the tensors of
    each prefix. Never one global clip."""

    def __init__(self, max_norms: Dict[str, float]):
        self.max_norms = {k: float(v) for k, v in max_norms.items() if v and v > 0}

    def __call__(self, names: Sequence[str], grads: Sequence[Optional[Tensor]]) -> List[Optional[Tensor]]:
        out = list(grads)
        for prefix, mx in self.max_norms.items():
            idx = [i for i, n in enumerate(names) if n.startswith(prefix) and out[i] is not None]
            if not idx:
                continue
            norm = torch.sqrt(sum((out[i] ** 2).sum() for i in idx) + 1e-30)
            scale = torch.clamp(mx / norm, max=1.0)
            for i in idx:
                out[i] = out[i] * scale
        return out


@dataclass
class InnerBatch:
    """Cached latents and reward inputs of a set of stored decisions, rows in lifetime order."""
    exp_ids: List[str]
    b: Tensor                          # [n, D]
    e_a: Tensor                        # [n, N, D]
    a_mask: Tensor                     # [n, N] bool
    e_g: Tensor                        # [n, D]
    chosen: Tensor                     # [n] long
    behaviour_logp: Tensor             # [n] log of the logged propensity of the executed candidate (-inf: missing)
    ts: Tensor                         # [n] float64 decision times
    z: Optional[Tensor] = None         # [n, N, D] trunk output under the live trunk (used when the trunk is fixed)
    b_next: Optional[Tensor] = None    # [n, D]
    e_a_chosen: Optional[Tensor] = None
    has_goal: Optional[Tensor] = None  # [n] bool: False = goal-free step (learned null goal)
    signals: Optional[Tensor] = None   # [n, N_SIGNALS]
    signal_mask: Optional[Tensor] = None
    r_boot: Optional[Tensor] = None    # [n] labelled bootstrap reward
    gamma: Optional[Tensor] = None     # [n] discount from row i to row i+1 (0 at the end of a lifetime segment)
    segments: List[Tuple[int, int]] = field(default_factory=list)     # [start, end) of each lifetime segment
    h0: List[Tensor] = field(default_factory=list)                    # lifetime state before each segment
    norm_state: Dict[str, Tensor] = field(default_factory=dict)       # R_psi input statistics in force at build
    mix: float = 1.0                   # weight of the bootstrap reward
    boot_components: Dict[str, Tensor] = field(default_factory=dict)
    lifetime: Optional[Tensor] = None  # (engine hook; one lifetime per batch here)

    @property
    def n(self) -> int:
        return int(self.b.shape[0])

    _FLOAT = ("b", "e_a", "e_g", "z", "b_next", "e_a_chosen", "signals", "signal_mask", "r_boot", "gamma")

    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> "InnerBatch":
        kw: Dict[str, Any] = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Tensor):
                v = v.to(device=device, dtype=dtype if (f.name in self._FLOAT and dtype is not None) else v.dtype)
            kw[f.name] = v
        kw["h0"] = [h.to(device=device, dtype=dtype or h.dtype) for h in self.h0]
        kw["norm_state"] = {k: v.to(device=device) for k, v in self.norm_state.items()}
        kw["boot_components"] = dict(self.boot_components)
        return InnerBatch(**kw)

    def with_mix(self, mix: float) -> "InnerBatch":
        return dataclasses.replace(self, mix=float(mix))

    def select(self, idx: Sequence[int]) -> "InnerBatch":
        """Policy-side rows only (for scoring logged query actions)."""
        i = torch.as_tensor(list(idx), dtype=torch.long, device=self.b.device)
        return InnerBatch(exp_ids=[self.exp_ids[j] for j in idx], b=self.b[i], e_a=self.e_a[i],
                          a_mask=self.a_mask[i], e_g=self.e_g[i], chosen=self.chosen[i],
                          behaviour_logp=self.behaviour_logp[i], ts=self.ts[i],
                          z=None if self.z is None else self.z[i])


class PolicyImprovementLoss:
    """``learning_api.LossProvider`` (timescale ``frequent``) and the functional inner loss of the meta unroll.

    ``lifetime_of`` maps an experience id to the index of the lifetime it belongs to (default: one lifetime);
    ``life_states`` maps an experience id to the lifetime-context state R_psi had *before* that step (filled by
    :meth:`reward_step`)."""

    name = "policy_improvement"
    timescale = TIMESCALE_FREQUENT

    def __init__(self, model: Any, policy: nn.Module, reward: IntrinsicReward, *,
                 mixer: Optional[RewardMixer] = None, bootstrap: Optional[BootstrapReward] = None,
                 assembler: Optional[SignalAssembler] = None, cfg: Optional[InnerConfig] = None,
                 lifetime_of: Optional[Callable[[str], int]] = None,
                 life_states: Optional[Dict[str, Tensor]] = None):
        self.model, self.policy, self.reward = model, policy, reward
        self.cfg = cfg or InnerConfig()
        self.weight = float(self.cfg.weight)
        self.mixer = mixer or RewardMixer()
        self.bootstrap = bootstrap or BootstrapReward()
        self.assembler = assembler or SignalAssembler()
        self.subset = AdaptableSubset(model, policy, self.cfg.adapt_trunk)
        self.lifetime_of = lifetime_of or (lambda exp_id: 0)
        self.life_states: Dict[str, Tensor] = life_states if life_states is not None else {}

    # --------------------------------------------------------------------------------------------- batches
    def eligible(self, record: ExperienceRecord) -> bool:
        """Executed decisions with a stored context; nothing imagined is ever written as an ExperienceRecord."""
        return (record.result is not None and bool(record.context_item_refs)
                and record.eligible.get(self.name, True) is not False)

    def _order(self, records: Sequence[ExperienceRecord]) -> List[int]:
        return sorted(range(len(records)), key=lambda i: (self.lifetime_of(records[i].exp_id), records[i].decision.ts,
                                                          records[i].seq, records[i].exp_id))

    def _encode_next(self, store: ExperienceStore, recs: Sequence[ExperienceRecord]) -> Tuple[Tensor, Tensor]:
        has = [bool(r.post_item_refs) for r in recs]
        with torch.no_grad():
            tok = self.model.tokenize([store.items(r.post_item_refs if h else r.context_item_refs)
                                       for r, h in zip(recs, has)])
            b_next = self.model.core_forward(tok, None).b
        return b_next, torch.tensor(has, device=b_next.device)

    def _bootstrap(self, asm: AssembledSignals, gate_update: bool) -> Any:
        s = asm.signals
        n = len(asm.keys)

        def col(v: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
            if v is None:
                return torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)
            ok = torch.isfinite(v)
            return torch.where(ok, v, torch.zeros_like(v)), ok.to(torch.float64)
        lp, lp_ok = col(asm.learning_progress)
        dv, _ = col(s.delta_v)
        vd, vd_ok = col(s.verified_progress_delta)
        eb, eb_ok = col(s.epistemic_before)
        ea, ea_ok = col(s.epistemic_after)
        both = (eb_ok * ea_ok) > 0
        eb = torch.where(both, eb, torch.ones_like(eb))
        ea = torch.where(both, ea, torch.ones_like(ea))           # no measured drop: no exploration reward
        keys = [k if bool(ok) else None for k, ok in zip(asm.keys, both)]
        return self.bootstrap(learning_progress=lp * lp_ok, delta_v=dv, epistemic_before=eb, epistemic_after=ea,
                              verified_delta=vd, verified_mask=vd_ok, context_keys=keys, gate_update=gate_update)

    def build_batch(self, store: ExperienceStore, records: Sequence[ExperienceRecord], *,
                    batch: Optional[DecisionBatch] = None, policy_only: bool = False,
                    gate_update: bool = False) -> InnerBatch:
        """Cached latents (and, unless ``policy_only``, reward inputs) of ``records`` in lifetime order. With
        ``batch`` (a ``DecisionBatch`` of the same records, possibly carrying a graph) its tensors are reused."""
        records = list(records)
        order = self._order(records)
        recs = [records[i] for i in order]
        if batch is None:
            enc = encode_decisions(self.model, store, recs, with_next=not policy_only, grad=False)
            pick = lambda t: t
        else:
            if [r.exp_id for r in batch.records] != [r.exp_id for r in records]:
                raise ValueError("batch.records must be the given records")
            enc = batch
            idx = torch.tensor(order, dtype=torch.long, device=batch.b.device)
            pick = lambda t: t.index_select(0, idx)
        dev = enc.b.device
        b, e_a, a_mask, e_g, chosen = pick(enc.b), pick(enc.e_a), pick(enc.a_mask), pick(enc.e_g), pick(enc.chosen)
        prop = pick(enc.propensity).detach().to(torch.float64)
        has_p = torch.tensor([bool(r.decision.probabilities) for r in recs], device=dev)
        blp = torch.where(has_p & (prop > 0), torch.log(prop.clamp_min(1e-300)),
                          torch.full_like(prop, float("-inf")))
        ts = torch.tensor([r.decision.ts for r in recs], dtype=torch.float64, device=dev)
        ib = InnerBatch(exp_ids=[r.exp_id for r in recs], b=b, e_a=e_a, a_mask=a_mask, e_g=e_g, chosen=chosen,
                        behaviour_logp=blp, ts=ts, z=pick(enc.z), mix=float(self.mixer.mix))
        if policy_only:
            return ib
        if enc.b_next is not None:
            b_next, has_next = pick(enc.b_next).detach(), pick(enc.has_next)
        else:
            b_next, has_next = self._encode_next(store, recs)
        bd = b.detach()
        b_next = torch.where(has_next[:, None], b_next, bd)
        ib.b_next = b_next
        ib.e_a_chosen = e_a.detach().gather(1, chosen.view(-1, 1, 1).expand(-1, 1, e_a.shape[-1])).squeeze(1)
        ib.has_goal = torch.tensor([bool(r.goal_text) for r in recs], device=dev)
        asm = self.assembler.assemble(recs, store, b=bd, b_next=b_next, e_g=e_g.detach(), has_next=has_next,
                                      device=dev)
        ib.signals, ib.signal_mask = asm.tensor, asm.mask
        boot = self._bootstrap(asm, gate_update)
        ib.r_boot = boot.total.to(dev, torch.float32)
        ib.boot_components = {k: v.detach() for k, v in boot.components.items()}
        life = [self.lifetime_of(r.exp_id) for r in recs]
        beta = self.cfg.beta
        gam = [math.exp(-beta * max(0.0, recs[i + 1].decision.ts - recs[i].decision.ts))
               if i + 1 < len(recs) and life[i + 1] == life[i] else 0.0 for i in range(len(recs))]
        ib.gamma = torch.tensor(gam, dtype=torch.float32, device=dev)
        segs, s = [], 0
        for i in range(1, len(recs) + 1):
            if i == len(recs) or life[i] != life[s]:
                segs.append((s, i))
                s = i
        ib.segments = segs
        d_life = self.reward.lifetime.d_life
        ib.h0 = [self.life_states[recs[a].exp_id].to(dev) if recs[a].exp_id in self.life_states
                 else torch.zeros(d_life, device=dev) for a, _ in segs]
        ib.norm_state = {k: v.detach().clone() for k, v in self.reward.named_buffers() if k in NORM_BUFFERS}
        return ib

    # --------------------------------------------------------------------------------------------- reward
    def learned_reward(self, psi: Params, ib: InnerBatch) -> Tensor:
        """R_psi on the rows of ``ib`` [n], graph-connected to ``psi``. Uses the input statistics stored in the
        batch (the reward that is differentiated is the reward the inner update used) and the learned null goal
        on goal-free rows."""
        R = self.reward
        dt = ib.b.dtype
        e_g = ib.e_g
        if ib.has_goal is not None and not bool(ib.has_goal.all()):
            e_g = torch.where(ib.has_goal[:, None], e_g, psi["null_goal"].to(dt).expand_as(e_g))
        args = {**psi, **ib.norm_state}
        sm = ib.signal_mask
        if not self.cfg.sequence:
            ctx = torch.cat([h.to(dt).expand(e - s, -1) for (s, e), h in zip(ib.segments, ib.h0)])
            return R.functional(args, b_t=ib.b, e_a=ib.e_a_chosen, b_next=ib.b_next, e_g=e_g, signals=ib.signals,
                                life_ctx=ctx, signal_mask=sm).reshape(-1)
        outs = []
        for (s, e), h0 in zip(ib.segments, ib.h0):
            outs.append(R.functional(args, sequence=True, b_t=ib.b[s:e], e_a=ib.e_a_chosen[s:e],
                                     b_next=ib.b_next[s:e], e_g=e_g[s:e], signals=ib.signals[s:e],
                                     signal_mask=sm[s:e], h0=h0.to(dt)).reshape(-1))
        return torch.cat(outs)

    def intrinsic(self, psi: Params, ib: InnerBatch, *, detach_reward: bool = False) -> Tuple[Tensor, Tensor]:
        """(r [n] used by the learner, learned part [n]). ``detach_reward`` evaluates R_psi under ``no_grad``."""
        if ib.r_boot is None:
            raise ValueError("this batch carries no reward inputs (policy_only)")
        if detach_reward:
            with torch.no_grad():
                rl = self.learned_reward({k: v.detach() for k, v in psi.items()}, ib)
            rl = rl.detach()
        else:
            rl = self.learned_reward(psi, ib)
        rb = ib.r_boot.to(rl.dtype)
        return ib.mix * rb + (1.0 - ib.mix) * rl, rl

    # --------------------------------------------------------------------------------------------- the loss
    def functional(self, theta: Params, psi: Params, ib: InnerBatch, *, detach_reward: bool = False,
                   return_aux: bool = False, theta_sg: Optional[Params] = None) -> Any:
        """``theta_sg`` (meta unroll only): tensors with the values of ``theta`` that are not the variables of
        the inner gradient. The value baseline of the advantage is evaluated on them, so it is held fixed for
        the inner derivative and still differentiated by the unroll (``v.detach()`` would drop that path for
        K >= 2). Without it the baseline is ``v.detach()``: the same update, first-order use."""
        out = self.subset.forward(theta, ib)
        lp = out.log_prob(ib.chosen)
        r, rl = self.intrinsic(psi, ib, detach_reward=detach_reward)
        G = discounted_returns(r, gamma=ib.gamma.to(r.dtype))
        Gn = scale_normalise(G) if self.cfg.normalise else G
        v = out.value
        adv = Gn - (v.detach() if theta_sg is None else self.subset.forward(theta_sg, ib).value)
        ent = out.entropy()
        loss = -(lp * adv).mean() + self.cfg.value_coef * (0.5 * (v - Gn) ** 2).mean() \
            - self.cfg.entropy_coef * ent.mean()
        if not return_aux:
            return loss
        return loss, {"reward": r.detach(), "learned": rl.detach(), "returns": G.detach(), "logp": lp.detach(),
                      "entropy": ent.detach(), "value": v.detach()}

    def inner_loss(self, theta: Params, psi: Params, ib: InnerBatch, theta_sg: Optional[Params] = None) -> Tensor:
        """``reward.meta.InnerLoss`` (stop-gradient aware): what the meta unroll differentiates."""
        return self.functional(theta, psi, ib, theta_sg=theta_sg)
    inner_loss.stopgrad_aware = True                    # type: ignore[attr-defined]

    def logp(self, theta: Params, ib: InnerBatch) -> Tensor:
        """``reward.meta.LogProbFn``: log pi_theta of the logged actions [n]."""
        return self.subset.forward(theta, ib).log_prob(ib.chosen)

    def probs(self, theta: Params, ib: InnerBatch) -> Tensor:
        return self.subset.forward(theta, ib).probs

    def loss(self, model: Any, batch: DecisionBatch, store: ExperienceStore) -> Tuple[Optional[Tensor], Dict[str, float]]:
        """LossProvider: the same function on the live parameters with R_psi detached."""
        if not batch.records:
            return None, {}
        if model is not self.model:
            raise ValueError("this provider was built for another model instance")
        ib = self.build_batch(store, batch.records, batch=batch)
        L, aux = self.functional(self.subset.live(), self.reward.psi(), ib, detach_reward=True, return_aux=True)
        return L, {"loss": float(L.detach()), "n": float(ib.n), "mix": float(ib.mix),
                   "reward_mean": float(aux["reward"].mean()), "learned_mean": float(aux["learned"].mean()),
                   "bootstrap_mean": float(ib.r_boot.mean()), "entropy": float(aux["entropy"].mean())}

    # --------------------------------------------------------------------------------------------- real-time step
    @torch.no_grad()
    def reward_step(self, store: ExperienceStore, records: Sequence[ExperienceRecord], *, write: bool = True
                    ) -> List[MixedReward]:
        """Reward each new real decision once, in order: remember the lifetime state it saw, update the input
        statistics (unless frozen), advance the lifetime context and the noisy-TV gate, mix with the labelled
        bootstrap reward and (``write``) store ``intrinsic_reward`` / ``bootstrap_reward`` / ``reward_version``."""
        out = []
        for rec in records:
            self.life_states[rec.exp_id] = self.reward.lifetime.state.detach().clone()
            ib = self.build_batch(store, [rec], gate_update=True)
            e_g = ib.e_g if bool(ib.has_goal[0]) else None
            self.reward.observe(ib.b, ib.e_a_chosen, ib.b_next, e_g, ib.signals, ib.signal_mask)
            rl, _, _ = self.reward.reward_sequence(ib.b, ib.e_a_chosen, ib.b_next, e_g, ib.signals, ib.signal_mask,
                                                   advance=True)
            mixed = self.mixer.combine(ib.r_boot, rl, reward_version=self.reward.version,
                                       bootstrap_components=ib.boot_components, exp_ids=[rec.exp_id])
            if write:
                for k, v in mixed.record_fields(0).items():
                    setattr(rec, k, v)
                store.update(rec)
            out.append(mixed)
        return out
