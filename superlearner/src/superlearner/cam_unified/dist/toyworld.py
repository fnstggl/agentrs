"""A small real environment, a tiny recurrent torch policy and its learner, for the distributed stack's mechanism
tests and throughput benchmarks (manifest A16/F04 before CAM integration).

None of this is a CAM component. It exists so the actor → queue → learner → checkpoint → actor loop can be shown to
work, and measured, with a learner whose updates genuinely change behaviour, independently of the CAM model:

* `ToyWorld` implements the contract's `EnvAdapter` for B independent worlds: at each step a context c is drawn from
  the actor's seed and shown as F noisy NUMERIC measurements (prototype[c] + noise); the agent picks one of K tools
  (``lever_k``); the payoff (1 if the tool is the hidden best one for the context, else 0, optionally noisy) arrives
  ``delay`` steps later as an EVENT observation whose ``caused_by`` is the action's id (a delayed, attributed outcome).
  The hidden mechanism (prototypes and best tools) is drawn from ``mechanism_seed`` and is evaluator-only.
* `ToyEvaluator` implements `GoalEvaluator` (score = delivered payoff / episode length; ``details`` hold hidden truth
  and never reach the learner).
* `TinyRecurrentPolicy` implements the actor-side `ActorPolicyP` (dist/actor.py): GRU latent state over
  [measurements, previous action, previous payoff], softmax over tools, a payoff head p(payoff | h, tool) used for
  predictions and for a one-step lookahead over every candidate (recorded as imagined outcomes), weights swapped
  only by `load_weights`, latent state rebuilt from raw episode history by `rebuild_state`.
* `TinyLearnerModel` implements `LearnerModelP` (dist/learner.py): recomputes the policy over each trajectory with
  the current weights, V-trace actor-critic with clipped importance weights from the recorded behaviour
  log-probabilities (lag > 0) or exact on-policy weights (lag 0), plus BCE for the payoff head on attributed outcomes.
"""
from __future__ import annotations

import math
import random
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import (Action, ActionSource, EvidenceKind, ExecutionResult, GoalSpec, IndependentGoalResult,
                          Observation, ObservationBatch, ObsKind, ToolSchema)
from .offpolicy import importance_log_ratio, semi_markov_discount, vtrace


# ============================================================================ environment

class ToyWorld:
    """EnvAdapter over B independent contextual-decision worlds with delayed attributed outcomes."""

    def __init__(self, B: int = 1, n_contexts: int = 4, n_tools: int = 4, n_feat: int = 8, episode_len: int = 16,
                 delay: int = 2, noise: float = 0.3, payoff_noise: float = 0.0, mechanism_seed: int = 0,
                 seed: int = 0, dt_days: float = 1.0 / 24.0, env_prefix: str = "toy"):
        self.B, self.C, self.K, self.F = B, n_contexts, n_tools, n_feat
        self.T, self.delay, self.noise, self.payoff_noise = episode_len, delay, noise, payoff_noise
        self.dt_days = dt_days
        self.env_prefix = env_prefix
        mg = random.Random(mechanism_seed)
        # evaluator-only mechanism
        self._proto = [[mg.gauss(0.0, 1.0) for _ in range(n_feat)] for _ in range(n_contexts)]
        self._best = [mg.randrange(n_tools) for _ in range(n_contexts)]
        self._rng = [random.Random((seed * 1_000_003 + b) & 0xFFFFFFFF) for b in range(B)]
        self._tools = [ToolSchema(f"lever_{k}", f"Pull lever {k} of the panel; the panel pays out later.",
                                  {"type": "object", "properties": {}, "required": []},
                                  returns="acknowledgement; the payout arrives as a later event",
                                  duration_hint=dt_days) for k in range(n_tools)]
        self.env_ids = [f"{env_prefix}{seed}.{b}" for b in range(B)]     # unique per actor seed (one lifetime each)
        self._episode = [0] * B
        self._reset_state()

    def _reset_state(self):
        B = self.B
        self.t = [0.0] * B
        self.step_i = [0] * B
        self.ctx = [0] * B
        self.done = [False] * B
        self.pending = [[] for _ in range(B)]        # (due_step, payoff, action_id, tool)
        self.events = [[] for _ in range(B)]         # delivered EVENT observations (seq ordered)
        self.seq = [0] * B
        self.delivered = [0.0] * B
        self._new_obs = [[] for _ in range(B)]

    def _context_obs(self, b: int) -> list:
        r = self._rng[b]
        c = r.randrange(self.C)
        self.ctx[b] = c
        return [Observation(ObsKind.NUMERIC, value=self._proto[c][i] + r.gauss(0.0, self.noise), unit="signal",
                            entity="panel", entity_type="instrument", t=self.t[b], source="panel.gauges",
                            meta={"i": i}) for i in range(self.F)]

    def _batch(self, take_new: bool = True) -> ObservationBatch:
        obs = [list(o) for o in self._new_obs]
        if take_new:
            self._new_obs = [[] for _ in range(self.B)]
        return ObservationBatch(list(self.env_ids), obs, list(self.t), list(self.done))

    # ---------------------------------------------------------------- EnvAdapter
    def reset(self, seeds: Optional[Sequence[int]] = None) -> ObservationBatch:
        if seeds is not None:
            self._rng = [random.Random(int(s)) for s in seeds]
        self._reset_state()
        for b in range(self.B):
            self._episode[b] += 1
            self._new_obs[b] = self._context_obs(b)
        return self._batch()

    def observe(self) -> ObservationBatch:
        return self._batch()

    def available_tools(self) -> list:
        return [list(self._tools) for _ in range(self.B)]

    def goals(self) -> list:
        return [GoalSpec(goal_id=f"{self.env_ids[b]}:ep{self._episode[b]}",
                         text="Earn as many panel payouts as possible by choosing the right lever for each reading.",
                         metrics={"payout_rate": 1.0}, deadline_days=self.T * self.dt_days, family="toy_panel")
                for b in range(self.B)]

    def execute(self, actions: Sequence[Optional[Action]]) -> list:
        out = []
        for b, a in enumerate(actions):
            if a is None or self.done[b]:
                out.append(ExecutionResult(ok=False, error="skipped", started_t=self.t[b], finished_t=self.t[b]))
                continue
            names = [t.name for t in self._tools]
            if a.tool not in names:
                out.append(ExecutionResult(ok=False, error=f"unknown tool {a.tool}", started_t=self.t[b],
                                           finished_t=self.t[b],
                                           observations=[Observation(ObsKind.ERROR, f"unknown tool {a.tool}",
                                                                     t=self.t[b], source="panel")]))
                continue
            k = names.index(a.tool)
            pay = 1.0 if k == self._best[self.ctx[b]] else 0.0
            if self.payoff_noise > 0:
                pay = pay + self._rng[b].gauss(0.0, self.payoff_noise)
            self.pending[b].append((self.step_i[b] + self.delay, pay, a.action_id, a.tool))
            out.append(ExecutionResult(ok=True, observations=[Observation(
                ObsKind.TOOL_RESULT, '{"ok": true, "status": "pulled"}', t=self.t[b], source=a.tool,
                evidence=EvidenceKind.CONTROLLED, caused_by=a.action_id)], cost=0.0, duration=self.dt_days,
                started_t=self.t[b], finished_t=self.t[b]))
        return out

    def collect_events(self, since: Sequence[int]) -> tuple:
        obs, cur = [], []
        for b in range(self.B):
            s = int(since[b]) if since[b] is not None else 0
            obs.append([o for o in self.events[b] if o.seq is not None and o.seq > s])
            cur.append(self.seq[b])
        return obs, cur

    def advance_time(self, duration: Sequence[float]) -> ObservationBatch:
        for b in range(self.B):
            if self.done[b]:
                continue
            self.t[b] += float(duration[b])
            self.step_i[b] += 1
            final = self.step_i[b] >= self.T
            due = [p for p in self.pending[b] if p[0] <= self.step_i[b] or final]
            self.pending[b] = [p for p in self.pending[b] if not (p[0] <= self.step_i[b] or final)]
            for _, pay, aid, tool in due:
                self.seq[b] += 1
                self.delivered[b] += pay
                self.events[b].append(Observation(ObsKind.EVENT, f"payout {pay:g} for {tool}", value=pay,
                                                  unit="points", entity="panel", t=self.t[b], seq=self.seq[b],
                                                  source="panel.payout", evidence=EvidenceKind.CONTROLLED,
                                                  caused_by=aid, meta={"key": "payoff"}))
            if final:
                self.done[b] = True
            else:
                self._new_obs[b] = self._context_obs(b)
        return self._batch()

    def close(self) -> None:
        pass

    # evaluator-only access (used by ToyEvaluator, never by the agent)
    def _truth(self, b: int) -> dict:
        return {"best": list(self._best), "context": self.ctx[b]}


class ToyEvaluator:
    """GoalEvaluator: verified payout rate. ``details`` carry hidden truth (never given to the learner)."""

    def __init__(self, env: ToyWorld):
        self.env = env

    def evaluate_goal(self, env_index: int) -> IndependentGoalResult:
        e = self.env
        score = e.delivered[env_index] / max(1, e.T)
        return IndependentGoalResult(goal_id=e.goals()[env_index].goal_id, achieved=score >= 0.9, score=score,
                                     metrics={"payout": e.delivered[env_index]}, t=e.t[env_index],
                                     evaluator="toy_payout", details=e._truth(env_index))


def make_toy_env(seed: int = 0, **kw) -> tuple:
    """Env factory (``superlearner.cam_unified.dist.toyworld:make_toy_env``) -> (EnvAdapter, GoalEvaluator)."""
    env = ToyWorld(seed=seed, **kw)
    return env, ToyEvaluator(env)


def best_tool_names(**world_kw) -> list:
    """Evaluator-side helper for tests/benchmarks: the hidden best tool per context of a world configuration."""
    return [f"lever_{k}" for k in ToyWorld(**world_kw)._best]


# ============================================================================ network

class TinyNet(nn.Module):
    """h_t = GRU(h_{t-1}, W[x_t; onehot(a_{t-1}); r_{t-1}]); logits = π(h_t) [K]; value V(h_t); payoff head
    q(h_t)[k] = logit p(payoff = 1 | h_t, tool k)."""

    def __init__(self, n_feat: int = 8, n_tools: int = 4, hidden: int = 32):
        super().__init__()
        self.n_feat, self.n_tools, self.hidden = n_feat, n_tools, hidden
        self.inp = nn.Linear(n_feat + n_tools + 1, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.pi = nn.Linear(hidden, n_tools)
        self.v = nn.Linear(hidden, 1)
        self.q = nn.Linear(hidden, n_tools)

    def init_h(self, B: int, device=None):
        return torch.zeros(B, self.hidden, device=device)

    def step(self, h, x, a_prev, r_prev):
        """h [B,H], x [B,F], a_prev [B,K] one-hot, r_prev [B,1] -> (h' [B,H], logits [B,K], value [B], q [B,K])."""
        z = torch.tanh(self.inp(torch.cat([x, a_prev, r_prev], -1)))
        h = self.gru(z, h)
        return h, self.pi(h), self.v(h).squeeze(-1), self.q(h)


def featurize(obs: list, n_feat: int) -> tuple:
    """Agent-visible observations -> (x [F] measurements, payoff observed in this view)."""
    x = [0.0] * n_feat
    r = 0.0
    for o in obs:
        if o.kind == ObsKind.NUMERIC and o.value is not None and "i" in (o.meta or {}):
            i = int(o.meta["i"])
            if 0 <= i < n_feat:
                x[i] = float(o.value)
        elif o.kind == ObsKind.EVENT and o.value is not None and (o.meta or {}).get("key") == "payoff":
            r += float(o.value)
    return x, r


# ============================================================================ actor-side policy

class TinyRecurrentPolicy:
    """ActorPolicyP over ToyWorld. Batched over the actor's B environments."""

    def __init__(self, seed: int = 0, n_feat: int = 8, n_tools: int = 4, hidden: int = 32, greedy: bool = False,
                 temperature: float = 1.0, **_):
        torch.manual_seed(seed)
        self.net = TinyNet(n_feat, n_tools, hidden).eval()
        self.gen = torch.Generator().manual_seed(seed)
        self.greedy, self.temperature = greedy, temperature
        self.version = -1
        self.h = None
        self.a_prev = None
        self.B = 0

    def load_weights(self, state_dict: dict, version: int) -> None:
        self.net.load_state_dict(state_dict)
        self.version = int(version)

    def reset(self, obs: ObservationBatch, goals: list, tools: list) -> None:
        self.B = len(obs.env_ids)
        self.h = self.net.init_h(self.B)
        self.a_prev = torch.zeros(self.B, self.net.n_tools)

    def _inputs(self, obs_lists: list):
        xs, rs = zip(*(featurize(o, self.net.n_feat) for o in obs_lists))
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(rs, dtype=torch.float32)[:, None]

    @torch.no_grad()
    def act(self, obs_lists: list, tools: list, goals: list, active: Optional[list] = None) -> tuple:
        x, r = self._inputs(obs_lists)
        h_before = self.h
        self.h, logits, value, q = self.net.step(self.h, x, self.a_prev, r)
        logp_all = F.log_softmax(logits / self.temperature, -1)
        if self.greedy:
            a = logp_all.argmax(-1)
        else:
            a = torch.multinomial(logp_all.exp(), 1, generator=self.gen).squeeze(-1)
        p_pay = torch.sigmoid(q)
        actions, infos = [], []
        for b in range(len(obs_lists)):
            if active is not None and not active[b]:
                actions.append(None)
                infos.append({})
                continue
            k = int(a[b])
            lp = float(logp_all[b, k])
            names = [t.name for t in tools[b]]
            act = Action(tool=names[k], logp=lp, logp_parts={"tool": lp}, source=ActionSource.BASELINE)
            actions.append(act)
            pk = float(p_pay[b, k])
            probs = logp_all[b].exp().tolist()
            infos.append({
                "predicted": {"payoff": pk, "value": float(value[b])},
                "uncertainty": {"payoff": math.sqrt(max(pk * (1 - pk), 1e-12))},
                "candidates": [{"action": {"tool": names[j]}, "score": probs[j], "source": "toy_policy"}
                               for j in range(len(names))],
                "imagined": [{"candidate": j, "horizon": 1, "value": float(p_pay[b, j]),
                              "outcomes": {"payoff": float(p_pay[b, j])}} for j in range(len(names))],
                "belief_before": {"h_norm": float(h_before[b].norm())},
                "belief_after": {"h_norm": float(self.h[b].norm())},
                "reasoning": {"iters": 1},
            })
        self.a_prev = F.one_hot(a, self.net.n_tools).float()
        return actions, infos

    @torch.no_grad()
    def rebuild_state(self, history: list) -> None:
        """Recompute the latent state of every env from its raw episode records under the current weights.
        history: [B] list[EventRecord] (decision records of the running episode, in order)."""
        B = len(history)
        self.h = self.net.init_h(B)
        self.a_prev = torch.zeros(B, self.net.n_tools)
        T = max((len(h) for h in history), default=0)
        for t in range(T):
            rows = [b for b in range(B) if t < len(history[b])]
            obs = [history[b][t].observations for b in rows]
            x, r = self._inputs(obs)
            idx = torch.tensor(rows)
            h, _, _, _ = self.net.step(self.h[idx], x, self.a_prev[idx], r)
            self.h[idx] = h
            a = torch.tensor([int(history[b][t].action.tool.split("_")[-1]) for b in rows])
            self.a_prev[idx] = F.one_hot(a, self.net.n_tools).float()


def make_tiny_policy(seed: int = 0, **kw) -> TinyRecurrentPolicy:
    return TinyRecurrentPolicy(seed=seed, **kw)


# ============================================================================ learner-side model

class TinyLearnerModel:
    """LearnerModelP for TinyNet: one optimiser step per `update` call.

    Reward of decision t = sum of the payoff observations attributed to it (delayed outcomes, ``attributed_to``);
    discount γ_t = exp(−κ Δt_t) with Δt from the record (``extra["dt"]``), 0 after the final step."""

    def __init__(self, n_feat: int = 8, n_tools: int = 4, hidden: int = 32, lr: float = 3e-3, kappa: float = 16.6,
                 value_coef: float = 0.5, entropy_coef: float = 0.01, payoff_coef: float = 0.5, rho_bar: float = 1.0,
                 c_bar: float = 1.0, grad_clip: float = 1.0, seed: int = 0, train: bool = True):
        torch.manual_seed(seed)
        self.net = TinyNet(n_feat, n_tools, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.kappa, self.value_coef, self.entropy_coef = kappa, value_coef, entropy_coef
        self.payoff_coef, self.rho_bar, self.c_bar, self.grad_clip = payoff_coef, rho_bar, c_bar, grad_clip
        self.train_enabled = train

    def state_dict(self) -> dict:
        return {k: v.detach().clone() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, sd: dict) -> None:
        self.net.load_state_dict(sd)

    def config(self) -> dict:
        return {"n_feat": self.net.n_feat, "n_tools": self.net.n_tools, "hidden": self.net.hidden}

    def _tensors(self, trajs: list):
        """Trajectories -> padded [B, T(, F)] tensors (built from Python lists, one tensor construction each)."""
        Fd = self.net.n_feat
        B, T = len(trajs), max(len(t.events) for t in trajs)
        xs, rins, acts, mus, rews, pys, pms, dts, masks, lags = ([] for _ in range(10))
        zf = [0.0] * Fd
        for tr in trajs:
            row = [[], [], [], [], [], [], [], [], []]
            for e in tr.events:
                f, rs = featurize(e.observations, Fd)
                vals = [o.value for rec in tr.outcomes.get(e.event_id, []) for o in rec.observations
                        if o.value is not None and (o.meta or {}).get("key") == "payoff"]
                tot = float(sum(vals))
                for lst, v in zip(row, (f, rs, int(e.action.tool.split("_")[-1]), float(e.behavior_logp), tot,
                                        float(tot > 0.5), bool(vals), float(e.extra.get("dt", 0.0)), True)):
                    lst.append(v)
            pad = T - len(tr.events)
            for dst, lst, z in zip((xs, rins, acts, mus, rews, pys, pms, dts, masks), row,
                                   (zf, 0.0, 0, 0.0, 0.0, 0.0, False, 0.0, False)):
                dst.append(lst + [z] * pad)
            lags.append([tr.lag] * T)
        return (torch.tensor(xs, dtype=torch.float32), torch.tensor(rins, dtype=torch.float32)[..., None],
                torch.tensor(acts, dtype=torch.long), torch.tensor(mus, dtype=torch.float32),
                torch.tensor(rews, dtype=torch.float32), torch.tensor(pys, dtype=torch.float32),
                torch.tensor(pms, dtype=torch.bool), torch.tensor(dts, dtype=torch.float32),
                torch.tensor(masks, dtype=torch.bool), torch.tensor(lags, dtype=torch.long))

    def update(self, batch) -> dict:
        # truncated trajectories (actor stopped or crashed mid-episode) lack their final delayed outcomes; training
        # on them would treat missing payouts as zeros, so they are left out (they stay in the archive)
        trajs = [t for t in batch.trajectories if not t.truncated]
        if not trajs:
            return {"skipped": "all trajectories truncated", "n_steps": 0, "n_truncated": len(batch.trajectories)}
        x, r_in, a, mu, rew, pay_y, pay_m, dt, mask, lag = self._tensors(trajs)
        B, T = a.shape
        K = self.net.n_tools
        h = self.net.init_h(B)
        a_prev = torch.zeros(B, K)
        logps, ents, vals, qs = [], [], [], []
        for t in range(T):
            h, logits, v, q = self.net.step(h, x[:, t], a_prev, r_in[:, t])
            lp = F.log_softmax(logits, -1)
            logps.append(lp.gather(1, a[:, t:t + 1]).squeeze(1))
            ents.append(-(lp.exp() * lp).sum(-1))
            vals.append(v)
            qs.append(q.gather(1, a[:, t:t + 1]).squeeze(1))
            a_prev = F.one_hot(a[:, t], K).float()
        logp = torch.stack(logps, 1)
        ent = torch.stack(ents, 1)
        val = torch.stack(vals, 1)
        q = torch.stack(qs, 1)
        last = mask & ~torch.cat([mask[:, 1:], torch.zeros(B, 1, dtype=torch.bool)], 1)
        disc = semi_markov_discount(dt, self.kappa) * (~last).float() * mask.float()
        log_rho = importance_log_ratio(logp.detach(), mu, lag)
        log_rho = torch.where(mask, log_rho, torch.zeros_like(log_rho))
        vs, pg_adv, rho = vtrace(rew, val.detach(), torch.zeros(B), disc, log_rho, self.rho_bar, self.c_bar)
        m = mask.float()
        n = m.sum().clamp(min=1)
        pg_loss = -(pg_adv * logp * m).sum() / n
        v_loss = ((vs - val).pow(2) * m).sum() / n
        ent_mean = (ent * m).sum() / n
        pm = (pay_m & mask).float()
        pay_loss = (F.binary_cross_entropy_with_logits(q, pay_y, reduction="none") * pm).sum() / pm.sum().clamp(min=1)
        loss = pg_loss + self.value_coef * v_loss - self.entropy_coef * ent_mean + self.payoff_coef * pay_loss
        gn = 0.0
        if self.train_enabled:
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip))
            self.opt.step()
        off = (lag > 0) & mask
        return {"loss": loss.item(), "pg_loss": pg_loss.item(), "v_loss": v_loss.item(), "entropy": ent_mean.item(),
                "payoff_loss": pay_loss.item(), "mean_reward": float((rew * m).sum() / n), "grad_norm": gn,
                "rho_mean_offpolicy": float(rho[off].mean()) if off.any() else None,
                "rho_clipped_frac": float(((log_rho.exp() > self.rho_bar) & off).float().sum() / off.float().sum())
                if off.any() else None, "n_steps": int(m.sum()),
                "n_truncated": len(batch.trajectories) - len(trajs)}


def make_tiny_learner(seed: int = 0, **kw) -> TinyLearnerModel:
    return TinyLearnerModel(seed=seed, **kw)
