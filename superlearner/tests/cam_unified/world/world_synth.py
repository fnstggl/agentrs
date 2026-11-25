"""Small synthetic sequence worlds for workstream-C learning tests (test-only helpers).

Every world is simulated exactly and tokenised with fixed random embeddings standing in for workstream A's
tokenizer: ACTION tokens per executed action, OBS_NUM tokens for measured values, QUERY tokens per outcome key.
The model only ever sees these tokens, the recorded actions and elapsed times — never the hidden state.
"""
from __future__ import annotations

import math
import random
import zlib

import torch

from superlearner.cam_unified.interfaces import (Action, FastState, Likelihood, OutcomeBatch, SequenceBatch,
                                                 StepBatch, TokenKind, TokenSet)


class Emb:
    """Fixed random embedding per name."""

    def __init__(self, D: int, seed: int = 0):
        self.D = D
        self.seed = seed
        self.cache = {}

    def __call__(self, name: str) -> torch.Tensor:
        if name not in self.cache:
            g = torch.Generator().manual_seed(zlib.crc32(f"{self.seed}:{name}".encode()) % (2 ** 31))
            self.cache[name] = torch.randn(self.D, generator=g)
        return self.cache[name]


def _ts(x, kind, mask=None, time=None):
    B, N, _ = x.shape
    d = x.device
    return TokenSet(x, torch.ones(B, N, dtype=torch.bool, device=d) if mask is None else mask,
                    torch.full((B, N), int(kind), dtype=torch.long, device=d),
                    torch.zeros(B, N, device=d) if time is None else time,
                    torch.full((B, N), -1, dtype=torch.long, device=d))


class World:
    """Base: subclasses define actions, outcome keys and `simulate`."""
    actions: list = []
    keys: list = []          # [(name, Likelihood)]
    T: int = 8

    def __init__(self, D: int, seed: int = 0):
        self.D = D
        self.E = Emb(D, seed)

    # ---- tokens
    def act_tokens(self, a_ids, extra=None) -> TokenSet:
        """a_ids: list[int] -> one ACTION token per row (plus optional numeric argument tokens)."""
        x = torch.stack([self.E("act:" + self.actions[a]) for a in a_ids]).unsqueeze(1)
        return _ts(x, TokenKind.ACTION)

    def obs_tokens(self, values: list) -> TokenSet:
        """values: [B] list of dict name->value (missing names masked) -> OBS_NUM tokens."""
        names = ["start"] + [k for k, _ in self.obs_keys()]
        B = len(values)
        x = torch.zeros(B, len(names), self.D)
        m = torch.zeros(B, len(names), dtype=torch.bool)
        for b, vals in enumerate(values):
            for i, n in enumerate(names):
                if n in vals:
                    v = float(vals[n])
                    x[b, i] = self.E("key:" + n) + v * self.E("val:" + n)
                    m[b, i] = True
        return _ts(x, TokenKind.OBS_NUM, m)

    def obs_keys(self):
        return self.keys

    def queries(self, B: int, observed=None, ys=None) -> OutcomeBatch:
        Q = len(self.keys)
        x = torch.stack([self.E("q:" + k) for k, _ in self.keys]).unsqueeze(0).expand(B, -1, -1).clone()
        lik = torch.tensor([int(l) for _, l in self.keys]).repeat(B, 1)
        y = torch.zeros(B, Q) if ys is None else ys
        ym = torch.ones(B, Q, dtype=torch.bool) if observed is None else observed
        return OutcomeBatch(_ts(x, TokenKind.QUERY), lik, torch.zeros(B, Q, dtype=torch.long), y,
                            torch.zeros(B, Q, dtype=torch.bool), ym, torch.full((B, Q), -1, dtype=torch.long),
                            [[k for k, _ in self.keys] for _ in range(B)])

    # ---- data
    def batch(self, B: int, seed: int, plans=None) -> SequenceBatch:
        """Simulate B episodes of length T with a random behaviour policy (or fixed `plans` [B][T] action ids)."""
        rng = random.Random(seed)
        T = self.T
        acts = [[None] * T for _ in range(B)]
        obs = [[None] * T for _ in range(B)]
        ys = [[None] * T for _ in range(B)]
        rew = [[0.0] * T for _ in range(B)]
        dts = [[1.0] * T for _ in range(B)]
        done = [[False] * T for _ in range(B)]
        pad = [[False] * T for _ in range(B)]
        for b in range(B):
            st = self.reset(rng)
            ob = {"start": 1.0}
            ended = False
            for t in range(T):
                if ended:
                    pad[b][t] = True
                    acts[b][t], obs[b][t], ys[b][t] = 0, {}, {}
                    continue
                a = plans[b][t] if plans is not None else self.behaviour(st, rng, t)
                dt = self.duration(rng)
                st, y, r, d, ob_next = self.step(st, a, dt, rng)
                acts[b][t], obs[b][t], ys[b][t], rew[b][t], dts[b][t], done[b][t] = a, ob, y, r, dt, d
                ob = ob_next
                ended = d
        steps = []
        for t in range(T):
            o = self.obs_tokens([obs[b][t] for b in range(B)])
            a = self.act_tokens([acts[b][t] for b in range(B)])
            for b in range(B):
                if pad[b][t]:
                    o.mask[b] = False
                    a.mask[b] = False
            Q = len(self.keys)
            yv = torch.zeros(B, Q)
            ym = torch.zeros(B, Q, dtype=torch.bool)
            for b in range(B):
                for qi, (k, _) in enumerate(self.keys):
                    if k in ys[b][t]:
                        yv[b, qi] = float(ys[b][t][k])
                        ym[b, qi] = True
            steps.append(StepBatch(obs=o, act=a, outcomes=self.queries(B, ym, yv),
                                   dt=torch.tensor([dts[b][t] for b in range(B)]),
                                   reward=torch.tensor([rew[b][t] for b in range(B)]),
                                   done=torch.tensor([done[b][t] for b in range(B)]),
                                   behavior_logp=torch.zeros(B), model_version=torch.zeros(B, dtype=torch.long)))
        sb = SequenceBatch(steps)
        sb.actions = acts
        return sb

    def duration(self, rng):
        return 1.0

    def behaviour(self, st, rng, t):
        return rng.randrange(len(self.actions))


class DelayWorld(World):
    """Action A has an immediate unit effect on y; action B has the same effect two steps later."""
    actions = ["noop", "A", "B"]
    keys = [("y", Likelihood.GAUSSIAN)]
    T = 8

    def reset(self, rng):
        return {"hist": [0, 0]}

    def behaviour(self, st, rng, t):
        return rng.choices([0, 1, 2], [0.5, 0.25, 0.25])[0]

    def step(self, st, a, dt, rng):
        y = (1.0 if a == 1 else 0.0) + (1.0 if st["hist"][-2] == 2 else 0.0) + 0.05 * rng.gauss(0, 1)
        st = {"hist": st["hist"] + [a]}
        return st, {"y": y}, 0.0, False, {"y": y}


class InteractionWorld(World):
    """Two simultaneous interventions (levers x, w in {0,1}); y = 0.5x - 0.5w + 2xw (a pure interaction term)."""
    actions = ["x0w0", "x1w0", "x0w1", "x1w1"]
    keys = [("y", Likelihood.GAUSSIAN)]
    T = 4

    def act_tokens(self, a_ids, extra=None):
        x = torch.zeros(len(a_ids), 2, self.D)
        for b, a in enumerate(a_ids):
            xv, wv = a & 1, (a >> 1) & 1
            x[b, 0] = self.E("lever:x") + xv * self.E("set:on") + (1 - xv) * self.E("set:off")
            x[b, 1] = self.E("lever:w") + wv * self.E("set:on") + (1 - wv) * self.E("set:off")
        return _ts(x, TokenKind.ACTION)

    def reset(self, rng):
        return {}

    def step(self, st, a, dt, rng):
        xv, wv = a & 1, (a >> 1) & 1
        y = 0.5 * xv - 0.5 * wv + 2.0 * xv * wv + 0.05 * rng.gauss(0, 1)
        return st, {"y": y}, 0.0, False, {"y": y}

    @staticmethod
    def truth(a):
        xv, wv = a & 1, (a >> 1) & 1
        return 0.5 * xv - 0.5 * wv + 2.0 * xv * wv


class StockWorld(World):
    """A decaying stock with irregular elapsed times and a nonlinear overload: push adds 1 unless the stock is
    already >= 1.5, in which case it crashes to 0; pull removes 0.5; decay exp(-0.3 dt). y = stock after the
    action (observed). Multistep prediction needs the state, the action sequence and the durations."""
    actions = ["noop", "push", "pull"]
    keys = [("y", Likelihood.GAUSSIAN)]
    T = 12

    def reset(self, rng):
        return {"s": rng.uniform(0, 2)}

    def duration(self, rng):
        return rng.choice([0.5, 1.0, 2.0])

    def behaviour(self, st, rng, t):
        return rng.choices([0, 1, 2], [0.3, 0.45, 0.25])[0]

    def step(self, st, a, dt, rng):
        s = st["s"] * math.exp(-0.3 * dt)
        if a == 1:
            s = s + 1.0 if s < 1.5 else 0.0
        elif a == 2:
            s = max(0.0, s - 0.5)
        y = s + 0.03 * rng.gauss(0, 1)
        return {"s": s}, {"y": y}, 0.0, False, {"y": y}


class ProbeWorld(World):
    """Hidden h in {0,1} per episode. ``probe`` costs 1 and reveals h; ``exploit`` pays +10 if h = 1 else -10;
    ``safe`` pays +1; both end the episode; ``wait`` does nothing. Without information the best immediate action is
    ``safe`` (+1 > E[exploit] = 0 > probe = -1); probing first is worth -1 + (10 + 1)/2 = 4.5: an experiment with an
    immediate cost that creates the opportunity for a later payoff (spec 7.4)."""
    actions = ["wait", "probe", "safe", "exploit"]
    keys = [("probe_result", Likelihood.BERNOULLI)]
    T = 4

    def reset(self, rng):
        return {"h": rng.randrange(2)}

    def behaviour(self, st, rng, t):
        return rng.choices([0, 1, 2, 3], [0.2, 0.35, 0.2, 0.25])[0]

    def obs_keys(self):
        return [("probe_result", None), ("tick", None)]

    def step(self, st, a, dt, rng):
        if a == 1:
            return st, {"probe_result": float(st["h"])}, -1.0, False, {"probe_result": float(st["h"])}
        if a == 2:
            return st, {}, 1.0, True, {"tick": 1.0}
        if a == 3:
            return st, {}, (10.0 if st["h"] == 1 else -10.0), True, {"tick": 1.0}
        return st, {}, 0.0, False, {"tick": 1.0}


class InvestWorld(World):
    """invest costs 1 and builds capacity; sell pays 5 if capacity is built (and uses it up); idle does nothing."""
    actions = ["idle", "invest", "sell"]
    keys = [("built", Likelihood.BERNOULLI)]
    T = 6

    def reset(self, rng):
        return {"built": 0}

    def step(self, st, a, dt, rng):
        built = st["built"]
        r = 0.0
        if a == 1:
            r, built = -1.0, 1
        elif a == 2:
            r, built = (5.0 if built else 0.0), 0
        return {"built": built}, {"built": float(built)}, r, False, {"built": float(built)}


# ------------------------------------------------------------------------------------------------ belief double

class HypothesisBelief:
    """Test double of workstream B's belief module for ProbeWorld: particles are hypotheses h_k in {0,1}
    (half each); the probe result reweights them by a known likelihood (0.95 / 0.05). tokens(): K particle tokens
    + one weighted summary token."""

    def __init__(self, cfg, world: World, key="probe_result", acc=0.95, seed=0):
        self.cfg, self.world, self.key, self.acc = cfg, world, key, acc
        K, D = cfg.n_particles, cfg.d_model
        self.h = torch.tensor([k % 2 for k in range(K)])
        g = torch.Generator().manual_seed(seed + 5)
        self.vec = torch.randn(2, cfg.d_particle, generator=g)
        self.proj = torch.randn(cfg.d_particle, D, generator=g) / math.sqrt(cfg.d_particle)
        self.wdir = torch.randn(D, generator=g)

    def init_fast(self, B, device=None):
        K, Dp = self.cfg.n_particles, self.cfg.d_particle
        d = device
        parts = self.vec.to(d)[self.h.to(d)].unsqueeze(0).expand(B, -1, -1).clone()
        return FastState(particles=parts, logw=torch.full((B, K), -math.log(K), device=d),
                         pscale=torch.ones(B, K, Dp, device=d) * 0.1,
                         bayes_mu=torch.zeros(B, 1, 1, dtype=torch.float64, device=d),
                         bayes_P=torch.ones(B, 1, 1, 1, dtype=torch.float64, device=d),
                         bayes_noise=torch.ones(B, 1, dtype=torch.float64, device=d),
                         bayes_n=torch.zeros(B, 1, device=d), surprise=torch.zeros(B, 4, device=d),
                         steps=torch.zeros(B, dtype=torch.long, device=d), channels=[{} for _ in range(B)])

    def p1(self, fast):
        return (fast.logw.exp() * (self.h == 1).float().to(fast.logw.device)).sum(-1)

    def tokens(self, fast) -> TokenSet:
        d = fast.particles.device
        w = fast.logw.exp()
        wdir = self.wdir.to(d)
        px = fast.particles @ self.proj.to(d)                                            # [B,K,D]
        summ = (w.unsqueeze(-1) * px).sum(1, keepdim=True) + (self.p1(fast) - 0.5).view(-1, 1, 1) * 2 * wdir
        x = torch.cat([px + fast.logw.unsqueeze(-1) * 0.1 * wdir, summ], 1)
        return _ts(x, TokenKind.BELIEF)

    def update(self, fast, Z_prev, act, outcomes: OutcomeBatch):
        qi = [k for k, _ in self.world.keys].index(self.key)
        obs = outcomes.y_mask[:, qi].bool()
        y = outcomes.y[:, qi].round()
        hk = self.h.to(y.device).float().unsqueeze(0)
        ll = torch.where(hk == y.unsqueeze(1), torch.tensor(math.log(self.acc), device=y.device),
                         torch.tensor(math.log(1 - self.acc), device=y.device))
        logw = fast.logw + torch.where(obs.unsqueeze(1), ll, torch.zeros_like(ll))
        logw = logw - torch.logsumexp(logw, 1, keepdim=True)
        return FastState(fast.particles.clone(), logw, fast.pscale.clone(), fast.bayes_mu.clone(),
                         fast.bayes_P.clone(), fast.bayes_noise.clone(), fast.bayes_n.clone(),
                         fast.surprise.clone(), fast.steps + 1, [dict(c) for c in fast.channels])

    def sequence_beliefs(self, batch: SequenceBatch) -> list:
        """BELIEF tokens of B_t for every step of a real sequence."""
        return [self.tokens(f) for f in self.sequence_fast(batch)]

    def sequence_fast(self, batch: SequenceBatch) -> list:
        """FastState B_t for every step of a real sequence (updated with the real outcomes of steps < t)."""
        B = batch.steps[0].obs.B
        dev = batch.steps[0].obs.x.device
        fast = self.init_fast(B, dev)
        out = []
        for t, st in enumerate(batch.steps):
            if t > 0:
                prev = batch.steps[t - 1]
                fast = self.update(fast, None, prev.act, prev.outcomes)
                if bool(prev.done.any()):
                    fresh = self.init_fast(B, dev)
                    d = prev.done.bool()
                    fast = FastState(fast.particles, torch.where(d.unsqueeze(1), fresh.logw, fast.logw),
                                     fast.pscale, fast.bayes_mu, fast.bayes_P, fast.bayes_noise, fast.bayes_n,
                                     fast.surprise, fast.steps, fast.channels)
            out.append(fast)
        return out


# ------------------------------------------------------------------------------------------------ training

def batch_to(batch: SequenceBatch, device) -> SequenceBatch:
    out = []
    for s in batch.steps:
        o = s.outcomes
        out.append(StepBatch(s.obs.to(device), s.act.to(device),
                             OutcomeBatch(o.q.to(device), o.lik.to(device), o.n_classes.to(device), o.y.to(device),
                                          o.censored.to(device), o.y_mask.to(device), o.channel.to(device), o.keys),
                             s.dt.to(device), s.reward.to(device), s.done.to(device), s.behavior_logp.to(device),
                             s.model_version.to(device), None if s.goal is None else s.goal.to(device)))
    sb = SequenceBatch(out)
    if hasattr(batch, "actions"):
        sb.actions = batch.actions
    return sb


def tokens_to(ts, device):
    return None if ts is None else ts.to(device)


def train(wm, batches, steps: int, lr: float = 1e-3, beliefs=None, log_every: int = 0, seed: int = 0, belief=None,
          lr_end_frac: float = 1.0):
    """Adam on WorldModel.loss over a fixed list of batches (cycled); EMA target updated each step; optional linear
    learning-rate decay to ``lr_end_frac`` x lr. With a belief double, open-loop rollouts update each row's belief
    from the model's own sampled outcomes."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(wm.trainable_parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0 - (1.0 - lr_end_frac) * min(1.0, i / max(1, steps)))
    hist = []
    fasts = [belief.sequence_fast(b) for b in batches] if belief is not None else None
    for i in range(steps):
        j = i % len(batches)
        if belief is not None:
            out = wm.loss(batches[j], fast_states=fasts[j], belief_update_fn=belief.update,
                          belief_tokens_fn=belief.tokens)
        else:
            out = wm.loss(batches[j], beliefs=None if beliefs is None else beliefs[j])
        opt.zero_grad(set_to_none=True)
        out["world"].backward()
        torch.nn.utils.clip_grad_norm_(wm.trainable_parameters(), 10.0)
        opt.step()
        sched.step()
        wm.update_target(0.98)
        hist.append(float(out["world"].detach()))
        if log_every and i % log_every == 0:
            print(i, round(hist[-1], 3), {k: round(float(out[k].detach()), 3) for k in ("outcome", "reward", "latent", "kl",
                                                                                "obs", "open_loop")})
    return hist


def device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ------------------------------------------------------------------------------------------------ imagination hooks

def probe_hooks(world: "ProbeWorld", belief: HypothesisBelief, device, update: bool = True, value_fn=None):
    """Hooks for ProbeWorld imagination: the hypothesis belief double, a policy double that acts on the imagined
    belief (P(h=1) > 0.8 -> exploit, < 0.2 -> safe, otherwise wait), and the probe-result outcome query (observable
    only when the action is ``probe``)."""
    from superlearner.cam_unified.world.imagination import ImaginationHooks
    names = world.actions

    def act_tokens(ids):
        return world.act_tokens(ids).to(device)

    def policy(ctx):
        p1 = belief.p1(ctx.fast)
        ids = [3 if p > 0.8 else (2 if p < 0.2 else 0) for p in p1.tolist()]
        return [Action(names[i]) for i in ids], act_tokens(ids)

    def queries(Zp, actions, toks, k):
        q = world.queries(len(actions))
        ym = torch.tensor([[a.tool == "probe"] for a in actions])
        return OutcomeBatch(q.q.to(device), q.lik.to(device), q.n_classes.to(device), q.y.to(device),
                            q.censored.to(device), ym.to(device), q.channel.to(device), q.keys)

    def upd(f, Zp, a, ob):
        return belief.update(f, Zp, a, ob)

    def ident(f, Zp, a, ob):
        return f

    return ImaginationHooks(policy_fn=policy, outcome_queries_fn=queries, belief_update_fn=upd if update else ident,
                            belief_tokens_fn=belief.tokens, value_fn=value_fn), act_tokens


def probe_candidates(world: "ProbeWorld", B: int, device):
    from superlearner.cam_unified.world.imagination import Candidates
    n = len(world.actions)
    return Candidates([[Action(a, logp=-math.log(n)) for a in world.actions] for _ in range(B)],
                      world.act_tokens(list(range(n)) * B).to(device))


def real_start(wm, world: World, belief: HypothesisBelief, B: int, device, seed: int = 500):
    """Filtered real state after the first observation of fresh episodes, and the real FastState."""
    sb = batch_to(world.batch(B, seed=seed, plans=[[0] * world.T] * B), device)
    fast = belief.init_fast(B, device)
    Z = wm.filter(SequenceBatch(sb.steps[:1]), [belief.tokens(fast)])[-1]
    return Z, fast
