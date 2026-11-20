"""A small neural learner that learns from Crucible consequences without any language model.

* **Action space** (`crucible.integrations.ActionTemplates`): one primitive tool call per decision, factored as
  ``tool`` (from the actor's *current* tool list, plus "wait") then one choice per argument slot. Slot options come
  from what the actor has seen (identifiers indexed by the JSON key they appeared under, recent identifiers, numbers),
  the schema (enum values, booleans, "omit" for optional arguments) and a fixed log-spaced integer grid. Nothing in
  the action space names a strategy; tools whose required arguments need free-form content are not offered.
* **Policy** (`FactoredPolicy`): hashed observation features -> MLP -> GRU (recurrent state over the agent's
  lifetime) -> pointer heads. Tools and slot options are embedded from hashed tokens of their names/kinds/values, so
  tools discovered later need no new output units.
* **Learning** (`A2CLearner`): online advantage actor-critic with GAE over ``n_steps`` rollouts, truncated BPTT,
  entropy regularisation, running reward-scale normalisation. The reward is the goal's `RewardContract` signal only
  (potential-based progress shaping on verified state + a verified success bonus).
* **Lifetimes** (`Lifetime`): ``persistent`` - one world, never reset; episodes are bounded stretches used for
  reporting and checkpoints, the recurrent state and the world carry over. ``fork`` - every episode starts in a
  fresh fork of a fixed start snapshot (episodic training; the learner itself persists across episodes).
* **Controls**: ``random`` (uniform over the same factored space) and ``idle`` (always wait).
* **Checkpointing**: weights, optimiser, normaliser, entity memory, counters, RNG, the world snapshot and the reward
  contract's evaluator state; `Lifetime.resume` continues a run exactly where it stopped.

Run ``python -m crucible.agents.rl --help``; the experiment protocol and results are in docs/RL_INTEGRATION.md.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..integrations import (DAY, OMIT, ActionTemplates, ActorView, EntityMemory, id_kind, make_goal_world,
                            obs_features, symlog)
from .base import Policy

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - torch is an optional dependency of crucible
    torch = None
    nn = None

OBS_DIM = 512
N_BUCKETS = 8192


def _bucket(s: str) -> int:
    return int(hashlib.blake2b(s.encode(), digest_size=4).hexdigest(), 16) % (N_BUCKETS - 1) + 1


def tool_tokens(t: dict | None) -> list[str]:
    if t is None:
        return ["T:<wait>", "H:<wait>"]
    name = t["name"]
    host, _, ep = name.partition(".")
    ann = t.get("annotations") or {}
    toks = [f"T:{name}", f"H:{host}", f"E:{ep}", "RO" if ann.get("readOnlyHint") else "RW",
            "B" if ann.get("builtin", True) else "D"]
    toks += [f"W:{w}" for w in ep.split("_") if w]
    return toks


def option_tokens(tool: str, slot, opt) -> list[str]:
    v = opt.value
    toks = [f"P:{slot.param}", f"PT:{tool}.{slot.param}", f"K:{slot.kind}", f"SRC:{opt.source}",
            f"V:{slot.param}={opt.label}"[:120], f"TV:{tool}.{slot.param}={opt.label}"[:140]]
    if isinstance(v, str) and opt.source not in ("omit", "gen"):
        toks.append(f"IK:{id_kind(v)}")
        toks.append(f"PIK:{slot.param}:{id_kind(v)}")
    return toks


def option_numeric(opt) -> list[float]:
    v = opt.value
    num = symlog(float(v)) / 12.0 if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0
    return [min(opt.rank, 8) / 8.0, float(opt.source == "omit"), num, float(opt.source in ("key", "seen"))]


# ============================================================================================== policy network
if nn is not None:
    class FactoredPolicy(nn.Module):
        def __init__(self, obs_dim: int = OBS_DIM, hid: int = 128, emb: int = 64):
            super().__init__()
            self.hid, self.emb_dim = hid, emb
            self.inp = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh())
            self.prev = nn.Linear(emb, hid, bias=False)
            self.gru = nn.GRUCell(hid, hid)
            self.tok = nn.EmbeddingBag(N_BUCKETS, emb, mode="mean", padding_idx=0)
            self.num = nn.Linear(4, emb)
            self.q_tool = nn.Linear(hid, emb)
            self.q_slot = nn.Sequential(nn.Linear(hid + 2 * emb, hid), nn.Tanh(), nn.Linear(hid, emb))
            self.v = nn.Sequential(nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
            nn.init.normal_(self.tok.weight, std=0.3)
            with torch.no_grad():
                self.tok.weight[0].zero_()

        def embed(self, token_lists: list[list[str]], numeric: list[list[float]] | None = None) -> "torch.Tensor":
            ids, offs = [], []
            for tl in token_lists:
                offs.append(len(ids))
                ids.extend(_bucket(t) for t in tl)
            e = self.tok(torch.tensor(ids, dtype=torch.long), torch.tensor(offs, dtype=torch.long))
            if numeric is not None:
                e = e + self.num(torch.tensor(numeric, dtype=torch.float32))
            return e

        def core(self, x: np.ndarray, prev: "torch.Tensor", h: "torch.Tensor") -> "torch.Tensor":
            xi = self.inp(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)) + self.prev(prev.unsqueeze(0))
            return self.gru(xi, h)


def _sample(logits: "torch.Tensor", greedy: bool, gen: "torch.Generator") -> int:
    if greedy:
        return int(logits.argmax())
    p = torch.softmax(logits.detach(), -1)
    return int(torch.multinomial(p, 1, generator=gen))


@dataclass
class Decision:
    action: dict | None
    logp: "torch.Tensor"
    ent: "torch.Tensor"
    value: "torch.Tensor"
    prev_emb: "torch.Tensor"
    tool_idx: int
    n_tools: int


class FactoredActor:
    """Maps an observation to one structured action with the network (or uniformly at random / idle)."""

    def __init__(self, net, actor: str, memory: EntityMemory, mode: str = "a2c", seed: int = 0,
                 max_slots: int = 6, temperature: float = 1.0):
        self.net, self.actor, self.mem, self.mode = net, actor, memory, mode
        self.templates = ActionTemplates(memory, actor, max_slots=max_slots)
        self.gen = torch.Generator().manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature
        self.h = torch.zeros(1, net.hid) if net is not None else None
        self.prev = torch.zeros(net.emb_dim) if net is not None else None

    def reset_state(self) -> None:
        if self.net is not None:
            self.h = torch.zeros(1, self.net.hid)
            self.prev = torch.zeros(self.net.emb_dim)

    def detach(self) -> None:
        if self.net is not None:
            self.h = self.h.detach()
            self.prev = self.prev.detach()

    def candidates(self, tools: list[dict]) -> list[tuple[dict | None, list]]:
        out: list = [(None, [])]
        for t in tools:
            s = self.templates.slots(t)
            if s is not None:
                out.append((t, s))
        return out

    def act(self, obs: dict, last_result: dict | None, tools: list[dict], greedy: bool = False) -> Decision:
        cands = self.candidates(tools)
        if self.mode == "idle":
            z = torch.zeros(())
            return Decision(None, z, z, z, z, 0, len(cands))
        if self.mode == "random":
            i = int(self.rng.integers(0, len(cands)))
            t, slots = cands[i]
            if t is None:
                z = torch.zeros(())
                return Decision(None, z, z, z, z, 0, len(cands))
            choice = [int(self.rng.integers(0, len(s.options))) for s in slots]
            z = torch.zeros(())
            return Decision(ActionTemplates.build(t, slots, choice), z, z, z, z, i, len(cands))
        net = self.net
        x = obs_features(obs, last_result, OBS_DIM)
        self.h = net.core(x, self.prev, self.h)
        h = self.h[0]
        value = net.v(h).squeeze(-1)
        temb = net.embed([tool_tokens(t) for t, _ in cands])
        lt = (temb @ net.q_tool(h)) / math.sqrt(net.emb_dim) / self.temperature
        lpt = torch.log_softmax(lt, -1)
        i = _sample(lt, greedy, self.gen)
        logp = lpt[i]
        ent = -(lpt.exp() * lpt).sum()
        t, slots = cands[i]
        prev_emb = temb[i]
        if t is None:
            self.prev = prev_emb
            return Decision(None, logp, ent, value, prev_emb, i, len(cands))
        choice = []
        acc = torch.zeros(net.emb_dim)
        for s in slots:
            pemb = net.embed([[f"P:{s.param}", f"PT:{t['name']}.{s.param}", f"K:{s.kind}"]])[0]
            q = net.q_slot(torch.cat([h, temb[i], pemb + acc]))
            oemb = net.embed([option_tokens(t["name"], s, o) for o in s.options],
                             [option_numeric(o) for o in s.options])
            lo = (oemb @ q) / math.sqrt(net.emb_dim) / self.temperature
            lpo = torch.log_softmax(lo, -1)
            j = _sample(lo, greedy, self.gen)
            logp = logp + lpo[j]
            ent = ent + (-(lpo.exp() * lpo).sum())
            acc = acc + oemb[j]
            choice.append(j)
        self.prev = prev_emb + 0.5 * acc / max(1, len(slots))
        return Decision(ActionTemplates.build(t, slots, choice), logp, ent, value, prev_emb, i, len(cands))


# ============================================================================================== learner
@dataclass
class A2CConfig:
    lr: float = 3e-4
    gamma: float = 0.98
    lam: float = 0.95
    n_steps: int = 32
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad: float = 0.5
    hid: int = 128
    emb: int = 64
    temperature: float = 1.0
    reward_scale: float | None = None   # fixed reward multiplier; None = running return-std normalisation


class RunningStd:
    """Running standard deviation of discounted returns (reward-scale normalisation, as in PPO implementations)."""

    def __init__(self, gamma: float):
        self.gamma, self.ret, self.n, self.mean, self.m2 = gamma, 0.0, 0, 0.0, 0.0

    def update(self, r: float) -> None:
        self.ret = self.ret * self.gamma + r
        self.n += 1
        d = self.ret - self.mean
        self.mean += d / self.n
        self.m2 += d * (self.ret - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / max(1, self.n - 1)) if self.n > 1 else 1.0


class A2CLearner:
    def __init__(self, net, cfg: A2CConfig):
        self.net, self.cfg = net, cfg
        self.opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
        self.rms = RunningStd(cfg.gamma)
        self.buf: list = []
        self.updates = 0

    def record(self, d: Decision, reward: float) -> None:
        if self.cfg.reward_scale is not None:
            r = reward * self.cfg.reward_scale
        else:
            self.rms.update(reward)
            r = reward / max(self.rms.std, 1e-8)
        self.buf.append((d.logp, d.ent, d.value, r))

    def ready(self) -> bool:
        return len(self.buf) >= self.cfg.n_steps

    def update(self, bootstrap: float) -> dict:
        c = self.cfg
        T = len(self.buf)
        vals = [float(b[2].detach()) for b in self.buf] + [bootstrap]
        adv = np.zeros(T, np.float32)
        g = 0.0
        for t in reversed(range(T)):
            delta = self.buf[t][3] + c.gamma * vals[t + 1] - vals[t]
            g = delta + c.gamma * c.lam * g
            adv[t] = g
        ret = adv + np.array(vals[:T], np.float32)
        logp = torch.stack([b[0] for b in self.buf])
        ent = torch.stack([b[1] for b in self.buf])
        v = torch.stack([b[2] for b in self.buf])
        A = torch.as_tensor(adv)
        pl = -(logp * A).mean()
        vl = ((v - torch.as_tensor(ret)) ** 2).mean()
        loss = pl + c.vf_coef * vl - c.ent_coef * ent.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad))
        self.opt.step()
        self.buf = []
        self.updates += 1
        return {"loss": float(loss.detach()), "pol": float(pl.detach()), "v": float(vl.detach()),
                "ent": float(ent.detach().mean()), "grad": gn,
                "adv_abs": float(np.abs(adv).mean())}


# ============================================================================================== lifetime driver
@dataclass
class LifetimeConfig:
    goal_family: str = "revenue"
    world_family: str = "tiny"
    level: int = 1
    seed: int = 0
    prefix: str = "deployed_service"
    prefix_kw: dict = field(default_factory=lambda: {"price_mult": 4.0})
    mode: str = "persistent"          # persistent | fork
    policy: str = "a2c"               # a2c | random | idle
    dt: int = 3 * 3600
    episode_steps: int = 56           # 7 simulated days at dt = 3 h
    total_steps: int = 6000
    ckpt_every: int = 10              # episodes
    a2c: dict = field(default_factory=dict)
    hosts: list | None = None         # optional action-space restriction to these tool hosts (documented condition)
    out_dir: str = "runs/crucible_rl/dev"
    torch_threads: int = 1


def _metric(world, name, **args) -> float:
    from ..goals.metrics import METRICS
    try:
        return float(METRICS[name](world, **args))
    except Exception:  # noqa: BLE001
        return 0.0


class Lifetime:
    """One agent lifetime in a Crucible world, driven by a `FactoredActor`, learning online (``policy="a2c"``)."""

    def __init__(self, cfg: LifetimeConfig):
        from ..core.kernel import World
        from ..goals.contracts import RewardContract
        if torch is None:
            raise ImportError("crucible.agents.rl needs torch")
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.dir = Path(cfg.out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.goal, self.world, self.actor, self.pinfo = make_goal_world(
            cfg.goal_family, cfg.world_family, cfg.seed, cfg.level, prefix=cfg.prefix, prefix_kw=cfg.prefix_kw)
        if cfg.prefix and cfg.prefix != "none" and not self.pinfo.get("ok"):
            raise RuntimeError(f"start-state prefix failed: {self.pinfo.get('errors')}")
        self.mem0 = EntityMemory()
        for a, r in self.pinfo.get("history", []):     # the actor remembers its own past actions and results
            self.mem0.observe(None, a, r)
        self.mem = copy.deepcopy(self.mem0)
        self.base_blob = self.world.snapshot_blob()
        self.World = World
        acfg = A2CConfig(**cfg.a2c)
        self.acfg = acfg
        self.net = FactoredPolicy(hid=acfg.hid, emb=acfg.emb) if cfg.policy == "a2c" else None
        self.learner = A2CLearner(self.net, acfg) if self.net is not None else None
        self.pi = FactoredActor(self.net, self.actor, self.mem, cfg.policy, seed=cfg.seed, temperature=acfg.temperature)
        self.contract = RewardContract(self.goal)
        self.view = ActorView(self.world, self.actor)
        self.steps = 0
        self.episode = 0
        self.history: list = []
        self.last_result: dict | None = None
        self.train_stats: list = []
        self.wall0 = time.time()
        self._begin_episode()

    # ------------------------------------------------------------------ episodes
    def _begin_episode(self) -> None:
        if self.cfg.mode == "fork":
            self.world = self.World.from_blob(self.base_blob, world_id=f"{self.goal.id}-ep{self.episode}")
            self.view.rebind(self.world)
            self.mem = copy.deepcopy(self.mem0)
            self.pi.mem = self.mem
            self.pi.templates.mem = self.mem
            self.pi.reset_state()
            self.last_result = None
        self.contract.begin(self.world, self.actor)
        self.ep = {"episode": self.episode, "t0": self.world.now, "ret": 0.0, "actions": 0, "rejected": 0,
                   "waits": 0, "tools": {}, "errors": {}, "phi0": self.contract.phi[self.actor]}

    def _end_episode(self) -> None:
        w, a, p = self.world, self.actor, self.pinfo
        st = self.contract.status(w, a)
        lst = w.state.get("listings", p.get("listing")) if p.get("listing") else None
        dep_live = False
        if p.get("host") and "deployments" in w.state.tables:
            dep_live = any(d.get("host") == p["host"] and d.get("status") == "live"
                           for _, d in w.state.scan("deployments"))
        org = p.get("org") or "o:none"
        self.ep.update({"t1": w.now, "sim_day": w.now / DAY, "progress": st["progress"], "success": st["success"],
                        "rev7_cents": _metric(w, "independent_revenue", principal=org, window_s=7 * DAY),
                        "rev30_cents": _metric(w, "independent_revenue", principal=org, window_s=30 * DAY),
                        "cash_org": _metric(w, "cash", principal=org), "cash_self": _metric(w, "cash", principal=a),
                        "listing_price": None if lst is None else lst.get("price"),
                        "listing_status": None if lst is None else lst.get("status"), "service_live": dep_live,
                        "steps": self.steps, "wall_s": time.time() - self.wall0})
        self.history.append(self.ep)
        with open(self.dir / "episodes.jsonl", "a") as f:
            f.write(json.dumps(self.ep, default=float) + "\n")
        self.episode += 1

    # ------------------------------------------------------------------ one decision
    def step(self) -> None:
        cfg = self.cfg
        obs = self.view.observe()
        self.mem.observe(obs)
        tools = self.view.tools()
        if cfg.hosts:
            tools = [t for t in tools if t["name"].split(".", 1)[0] in cfg.hosts]
        if self.learner is not None and self.learner.ready():
            with torch.no_grad():   # bootstrap value of the current observation under the pre-update weights
                h = self.net.core(obs_features(obs, self.last_result, OBS_DIM), self.pi.prev, self.pi.h)
                boot = float(self.net.v(h[0]))
            self.train_stats.append(self.learner.update(boot))
            self.pi.detach()
        d = self.pi.act(obs, self.last_result, tools)
        if d.action is None:
            self.ep["waits"] += 1
            res = None
        else:
            r = self.world.execute(self.actor, d.action)
            res = r.to_dict()
            self.ep["actions"] += 1
            tn = d.action["tool"]
            self.ep["tools"][tn] = self.ep["tools"].get(tn, 0) + 1
            if not res["ok"]:
                self.ep["rejected"] += 1
                code = (res.get("error") or {}).get("code", "error")
                self.ep["errors"][code] = self.ep["errors"].get(code, 0) + 1
            self.mem.observe(None, d.action, res)
        self.last_result = res
        self.world.advance_time(self.world.now + cfg.dt)
        reward = float(self.contract.step_reward(self.world, self.actor))
        self.ep["ret"] += reward
        if self.learner is not None:
            self.learner.record(d, reward)
        self.steps += 1
        if self.steps % cfg.episode_steps == 0:
            if self.cfg.mode == "fork" and self.learner is not None and self.learner.buf:
                # time-limit truncation of a forked episode: bootstrap with the value of the final observation
                with torch.no_grad():
                    o_T = self.view.observe()
                    h = self.net.core(obs_features(o_T, self.last_result, OBS_DIM), self.pi.prev, self.pi.h)
                    boot = float(self.net.v(h[0]))
                self.train_stats.append(self.learner.update(boot))
                self.pi.detach()
            self._end_episode()
            if self.episode % cfg.ckpt_every == 0:
                self.save()
            self._begin_episode()

    def run(self) -> list:
        while self.steps < self.cfg.total_steps:
            self.step()
        self.save()
        return self.history

    # ------------------------------------------------------------------ checkpoints
    def save(self) -> str:
        ck = {"cfg": asdict(self.cfg), "steps": self.steps, "episode": self.episode, "history": self.history,
              "world": self.world.snapshot_blob(), "world_id": self.world.world_id, "base": self.base_blob,
              "contract": pickle.dumps(self.contract), "mem": self.mem.state_dict(), "mem0": self.mem0.state_dict(),
              "cursor": self.view.cursor, "last_result": self.last_result,
              "pi_rng": self.pi.rng.bit_generator.state, "pi_gen": self.pi.gen.get_state(),
              "torch_rng": torch.get_rng_state(), "train_stats": self.train_stats[-200:], "ep": self.ep,
              "actor": self.actor, "pinfo": {k: v for k, v in self.pinfo.items() if k != "history"}}
        if self.net is not None:
            ck.update(net=self.net.state_dict(), opt=self.learner.opt.state_dict(), rms=self.learner.rms.__dict__,
                      h=self.pi.h.detach(), prev=self.pi.prev.detach(), updates=self.learner.updates,
                      buf_len=len(self.learner.buf))
        tmp = self.dir / "ckpt.pt.tmp"
        torch.save(ck, tmp)
        os.replace(tmp, self.dir / "ckpt.pt")
        return str(self.dir / "ckpt.pt")

    @classmethod
    def resume(cls, path: str) -> "Lifetime":
        """Continue a lifetime from a checkpoint (the partial rollout in flight at save time is discarded)."""
        ck = torch.load(path, weights_only=False)
        cfg = LifetimeConfig(**ck["cfg"])
        self = cls.__new__(cls)
        from ..core.kernel import World
        torch.set_num_threads(cfg.torch_threads)
        self.cfg, self.dir, self.World = cfg, Path(cfg.out_dir), World
        import crucible
        crucible.load_all_modules()
        from ..goals.contracts import generate_goal
        self.goal = generate_goal(cfg.goal_family, cfg.world_family, cfg.seed, cfg.level, split="dev")
        self.world = World.from_blob(ck["world"], world_id=ck["world_id"])
        self.actor = ck["actor"]
        self.base_blob = ck["base"]
        self.contract = pickle.loads(ck["contract"])
        self.mem0, self.mem = EntityMemory(), EntityMemory()
        self.mem0.load_state_dict(ck["mem0"])
        self.mem.load_state_dict(ck["mem"])
        self.acfg = A2CConfig(**cfg.a2c)
        self.net = FactoredPolicy(hid=self.acfg.hid, emb=self.acfg.emb) if cfg.policy == "a2c" else None
        self.learner = A2CLearner(self.net, self.acfg) if self.net is not None else None
        self.pi = FactoredActor(self.net, self.actor, self.mem, cfg.policy, seed=cfg.seed, temperature=self.acfg.temperature)
        self.pi.rng.bit_generator.state = ck["pi_rng"]
        self.pi.gen.set_state(ck["pi_gen"])
        torch.set_rng_state(ck["torch_rng"])
        if self.net is not None:
            self.net.load_state_dict(ck["net"])
            self.learner.opt.load_state_dict(ck["opt"])
            self.learner.rms.__dict__.update(ck["rms"])
            self.learner.updates = ck["updates"]
            self.pi.h, self.pi.prev = ck["h"].clone(), ck["prev"].clone()
        self.view = ActorView(self.world, self.actor)
        self.view.cursor = ck["cursor"]
        self.pinfo = {k: v for k, v in (ck.get("pinfo") or {}).items()}
        self.steps, self.episode, self.history = ck["steps"], ck["episode"], ck["history"]
        self.last_result, self.train_stats, self.wall0 = ck["last_result"], ck["train_stats"], time.time()
        self.ep = ck["ep"]
        return self


# ============================================================================================== Policy wrapper
class RLPolicy(Policy):
    """A trained `FactoredPolicy` as an ordinary Crucible `Policy` (no learning; for `Runner` / `run_goal`)."""
    name, kind = "rl-factored", "rl"

    def __init__(self, net=None, greedy: bool = False, checkpoint: str | None = None):
        if checkpoint is not None:
            ck = torch.load(checkpoint, weights_only=False)
            a = A2CConfig(**ck["cfg"].get("a2c", {}))
            net = FactoredPolicy(hid=a.hid, emb=a.emb)
            net.load_state_dict(ck["net"])
        self.net, self.greedy = net, greedy
        self.last = None

    def reset(self, actor, rng):
        super().reset(actor, rng)
        self.mem = EntityMemory()
        self.pi = FactoredActor(self.net, actor, self.mem, "a2c" if self.net is not None else "random",
                                seed=int(rng.integers(0, 2**31)))
        self.last = None

    def act(self, obs, tools):
        self.mem.observe(obs)
        with torch.no_grad():
            d = self.pi.act(obs, self.last, [t for t in tools if not t["name"].startswith("admin.")], greedy=self.greedy)
        self._last_action = d.action
        return d.action

    def observe_result(self, action, result):
        self.last = result
        self.mem.observe(None, action, result)


# ============================================================================================== reporting
POLICY_COLORS = {"a2c": "#2a78d6", "random": "#eb6834", "idle": "#1baf7a"}   # categorical slots 1-3


def load_episodes(run_dir: str) -> list:
    p = Path(run_dir, "episodes.jsonl")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def report(groups: dict, out_json: str, out_png: str | None = None, last: int = 50, smooth: int = 10) -> dict:
    """Learning-curve summary. ``groups``: panel label -> {policy: [run dirs (one per seed)]}. Per run: weekly
    independent revenue (verified metric, 7-day window at each episode end), episode return (RewardContract), share
    of episodes with the service still live, market.update calls per episode; aggregated over seeds."""
    out: dict = {"last_episodes": last, "groups": {}}
    for g, pols in groups.items():
        G = out["groups"][g] = {}
        for pol, dirs in pols.items():
            runs = []
            for d in dirs:
                eps = load_episodes(d)
                if not eps:
                    continue
                rev = [e["rev7_cents"] / 100.0 for e in eps]
                k = min(last, len(eps))
                runs.append({"dir": str(d), "episodes": len(eps), "steps": eps[-1]["steps"],
                             "rev7_usd": rev, "ret": [e["ret"] for e in eps],
                             "rev7_first10": float(np.mean(rev[:10])), "rev7_last": float(np.mean(rev[-k:])),
                             "ret_last": float(np.mean([e["ret"] for e in eps[-k:]])),
                             "live_last": float(np.mean([e["service_live"] for e in eps[-k:]])),
                             "updates_per_ep_last": float(np.mean([e["tools"].get("market.update", 0) for e in eps[-k:]])),
                             "waits_last": float(np.mean([e["waits"] for e in eps[-k:]])),
                             "final_price": eps[-1].get("listing_price")})
            n = min((r["episodes"] for r in runs), default=0)
            G[pol] = {"seeds": len(runs), "episodes_common": n,
                      "rev7_last_per_seed": [r["rev7_last"] for r in runs],
                      "rev7_last_mean": float(np.mean([r["rev7_last"] for r in runs])) if runs else float("nan"),
                      "rev7_first10_mean": float(np.mean([r["rev7_first10"] for r in runs])) if runs else float("nan"),
                      "ret_last_per_seed": [r["ret_last"] for r in runs],
                      "live_last_per_seed": [r["live_last"] for r in runs],
                      "updates_per_ep_last": [r["updates_per_ep_last"] for r in runs],
                      "final_price_per_seed": [r["final_price"] for r in runs],
                      "curve_mean": np.mean([r["rev7_usd"][:n] for r in runs], 0).round(2).tolist() if n else [],
                      "curve_min": np.min([r["rev7_usd"][:n] for r in runs], 0).round(2).tolist() if n else [],
                      "curve_max": np.max([r["rev7_usd"][:n] for r in runs], 0).round(2).tolist() if n else []}
        if "a2c" in G and "random" in G and G["a2c"]["seeds"] == G["random"]["seeds"]:
            d = [a - b for a, b in zip(G["a2c"]["rev7_last_per_seed"], G["random"]["rev7_last_per_seed"])]
            G["a2c_minus_random_rev7_last_per_seed"] = d
            G["a2c_beats_random_all_seeds"] = bool(all(x > 0 for x in d))
        if "a2c" in G and "idle" in G and G["a2c"]["seeds"] == G["idle"]["seeds"]:
            d = [a - b for a, b in zip(G["a2c"]["rev7_last_per_seed"], G["idle"]["rev7_last_per_seed"])]
            G["a2c_minus_idle_rev7_last_per_seed"] = d
            G["a2c_beats_idle_all_seeds"] = bool(all(x > 0 for x in d))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1, default=float))
    if out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(groups)
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.6), squeeze=False)
        for ax, (g, G) in zip(axes[0], out["groups"].items()):
            for pol in ("a2c", "random", "idle"):
                if pol not in G or not G[pol]["curve_mean"]:
                    continue
                c = POLICY_COLORS[pol]
                m = np.convolve(G[pol]["curve_mean"], np.ones(smooth) / smooth, mode="valid")
                lo = np.convolve(G[pol]["curve_min"], np.ones(smooth) / smooth, mode="valid")
                hi = np.convolve(G[pol]["curve_max"], np.ones(smooth) / smooth, mode="valid")
                x = np.arange(len(m)) + smooth
                ax.fill_between(x, lo, hi, color=c, alpha=0.12, linewidth=0)
                ax.plot(x, m, color=c, linewidth=2, label=f"{pol} ({G[pol]['seeds']} seeds)")
                if len(m):
                    ax.annotate(pol, (x[-1], m[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8,
                                color="#52514e", va="center")
            ax.set_title(g, fontsize=10, color="#0b0b0b")
            ax.set_xlabel("episode (= 1 simulated week)", fontsize=9, color="#52514e")
            ax.set_ylabel("independent revenue, last 7 days ($)", fontsize=9, color="#52514e")
            ax.grid(True, color="#e6e5e1", linewidth=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=8, colors="#52514e")
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        fig.suptitle(f"Crucible revenue:tiny:1, deployed-service start state; mean over seeds (band: min-max), "
                     f"{smooth}-episode moving average", fontsize=9, color="#52514e")
        fig.tight_layout()
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
    return out


# ============================================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Online neural RL in a Crucible world (no LLM)")
    ap.add_argument("--policy", default="a2c", choices=["a2c", "random", "idle"])
    ap.add_argument("--mode", default="persistent", choices=["persistent", "fork"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--goal", default="revenue")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--world", default="tiny")
    ap.add_argument("--price-mult", type=float, default=4.0)
    ap.add_argument("--capital", type=int, default=400_000, help="cents moved into the company by the prefix")
    ap.add_argument("--hosts", default="", help="comma-separated tool hosts the learner may use (default: all)")
    ap.add_argument("--dt", type=int, default=3 * 3600)
    ap.add_argument("--episode-steps", type=int, default=56)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="a2c overrides key=value")
    a = ap.parse_args(argv)
    out = a.out or f"runs/crucible_rl/{a.mode}-{a.policy}-s{a.seed}"
    if a.resume and Path(out, "ckpt.pt").exists():
        lt = Lifetime.resume(str(Path(out, "ckpt.pt")))
        lt.cfg.total_steps = a.steps
    else:
        a2c = {}
        for kv in a.set:
            k, v = kv.split("=", 1)
            a2c[k] = json.loads(v)
        cfg = LifetimeConfig(goal_family=a.goal, world_family=a.world, level=a.level, seed=a.seed, mode=a.mode,
                             policy=a.policy, dt=a.dt, episode_steps=a.episode_steps, total_steps=a.steps,
                             prefix_kw={"price_mult": a.price_mult, "capital": a.capital}, a2c=a2c, out_dir=out,
                             hosts=[h for h in a.hosts.split(",") if h] or None)
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "episodes.jsonl").write_text("")
        lt = Lifetime(cfg)
        (Path(out) / "config.json").write_text(json.dumps({"cfg": asdict(cfg), "goal": lt.goal.to_dict(),
                                                           "prefix": {k: v for k, v in lt.pinfo.items() if k != "history"}},
                                                          indent=1, default=str))
    hist = lt.run()
    print(json.dumps({"episodes": len(hist), "last": hist[-1] if hist else None}, default=float)[:2000])


if __name__ == "__main__":
    main()
