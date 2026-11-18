"""Persistent CAM workers: many independent episodes acting through one shared, batched CAM core.

Each worker owns its goal, environment instance (hidden parameters), belief state, active option,
history and step budget. Every round, all active workers' beliefs are updated and their actions are
generated in one batched forward pass of the shared network; environments step on the CPU. Every
decision records exactly what PPO needs: the option-termination sample and its log-prob, the option
choice and its log-prob (when a new option starts), the action token ids, their log-probs under the
grammar-masked sampling distribution, the allowed-token sets, the value estimate and the policy
version. Rewards come only from the environments (verified success + potential-based shaping on the
environment-measured progress).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from agentrs.cam.envs import ENVS
from agentrs.cam.model import CAM

INFO_ACTIONS = {"ASK", "RESEARCH"}


@dataclass
class CamStep:
    obs: list
    prev_act: list
    elapsed_prev: float
    feat_obs: np.ndarray
    prev_opt: int
    new_opt: bool
    term_sample: int
    opt: int
    act_tokens: list
    act_ids: list
    allowed: list
    logp_old: float
    v_old: float
    elapsed_h: float = 0.0
    events: np.ndarray = None
    feat_after: np.ndarray = None
    prog_before: float = 0.0
    prog_after: float = 0.0
    reward: float = 0.0
    valid: bool = True
    done: bool = False


@dataclass
class CamEpisode:
    family: str
    split: str
    seed: int
    goal: list
    version: str
    steps: list = field(default_factory=list)
    success: bool = False
    planned: bool = False

    @property
    def ret(self) -> float:
        return sum(s.reward for s in self.steps)


def pad(x: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    out[:len(x)] = x
    return out


class Workers:
    def __init__(self, model: CAM, n: int, families: dict[str, float], split: str = "train", seed0: int = 0,
                 gamma: float = 0.99, shaping: float = 0.5, step_cost: float = 0.005, gen_seed: int = 0):
        self.model, self.n, self.families, self.split = model, n, families, split
        self.seed_ctr = {f: seed0 for f in families}
        self.rng = random.Random(gen_seed)
        self.gen = torch.Generator(device=model.device).manual_seed(gen_seed)
        self.gamma, self.shaping, self.step_cost = gamma, shaping, step_cost
        self.stats = {"env_s": 0.0, "net_s": 0.0, "steps": 0, "rounds": 0, "batch": []}

    def _new_env(self, family: Optional[str] = None):
        fam = family or self.rng.choices(list(self.families), list(self.families.values()))[0]
        s = self.seed_ctr[fam]
        self.seed_ctr[fam] = s + 1
        env = ENVS[fam]()
        env.reset(s, self.split)
        return fam, s, env

    @torch.no_grad()
    def collect(self, min_episodes: int = 0, min_steps: int = 0, greedy: bool = False, version: str = "v0",
                planning: int = 0, fixed: Optional[list[tuple[str, int]]] = None) -> list[CamEpisode]:
        """Run episodes until ``min_episodes`` / ``min_steps`` are reached (or the ``fixed`` list of
        (family, seed) is exhausted); in-flight episodes are always finished (same policy version)."""
        m = self.model
        m.eval()
        dev = m.device
        cfg = m.cfg
        done_eps: list[CamEpisode] = []
        queue = list(fixed) if fixed is not None else None
        slots: list[Optional[dict]] = [None] * self.n
        total_steps = 0

        def start(i):
            if queue is not None:
                if not queue:
                    return None
                fam, seed = queue.pop(0)
                env = ENVS[fam]()
                env.reset(seed, self.split)
            else:
                fam, seed, env = self._new_env()
            return {"env": env, "ep": CamEpisode(fam, self.split, seed, env.goal_tokens(), version, planned=bool(planning)),
                    "b": torch.zeros(cfg.belief, device=dev), "prev_act": ["<bos>"], "elapsed": 0.0, "opt": 0,
                    "feat": pad(env.features(), cfg.n_feat), "t": 0}

        def want_more():
            if queue is not None:
                return bool(queue)
            return len(done_eps) + sum(s is not None for s in slots) < max(min_episodes, 1) or total_steps < min_steps

        for i in range(self.n):
            if want_more():
                slots[i] = start(i)
        while any(s is not None for s in slots):
            act = [i for i, s in enumerate(slots) if s is not None]
            t0 = time.time()
            S = [slots[i] for i in act]
            fam_tok = [[f"FAM_{s['ep'].family}"] for s in S]
            obs_ids = m.ids([f + s["env"].obs_tokens() for f, s in zip(fam_tok, S)], cfg.max_obs)
            goal_ids = m.ids([f + s["ep"].goal for f, s in zip(fam_tok, S)], 48)
            prev_ids = m.ids([s["prev_act"] for s in S], cfg.max_act + 2)
            el = torch.tensor([s["elapsed"] for s in S], device=dev, dtype=torch.float32)
            g = m.encode_goal(goal_ids)
            b = m.update_belief(torch.stack([s["b"] for s in S]), obs_ids, prev_ids, g, el)
            cur = torch.tensor([s["opt"] for s in S], device=dev)
            logits, beta = m.option_dists(b, g, cur)
            pb = torch.sigmoid(beta)
            if greedy:
                term = (pb > 0.5).long()
            else:
                term = torch.bernoulli(pb, generator=self.gen).long()
            first = torch.tensor([s["t"] == 0 for s in S], device=dev)
            new = first | term.bool()
            olp = torch.log_softmax(logits.float(), -1)
            o_s = olp.argmax(-1) if greedy else torch.multinomial(olp.exp(), 1, generator=self.gen).squeeze(-1)
            opt = torch.where(new, o_s, cur)
            t_lp = torch.where(first, torch.zeros_like(pb), torch.where(term.bool(), torch.log(pb + 1e-8), torch.log(1 - pb + 1e-8)))
            o_lp = torch.where(new, olp.gather(-1, o_s[:, None]).squeeze(-1), torch.zeros_like(pb))
            vals = m.estimate_value(b, g)["ret"]
            grams = [s["env"].grammar() for s in S]
            if planning and not greedy:
                raise ValueError("planning is an evaluation-time mode (it changes the behaviour distribution)")
            toks, ids, lps, allowed = m.generate_action(b, g, opt, grams, greedy=greedy, gen=self.gen)
            if planning:
                toks, ids, lps, allowed = self._plan(b, g, opt, grams, toks, ids, lps, allowed, planning, S)
            self.stats["net_s"] += time.time() - t0
            self.stats["batch"].append(len(act))
            t1 = time.time()
            for j, i in enumerate(act):
                s = slots[i]
                env, ep = s["env"], s["ep"]
                a = [t for t in toks[j] if t != "END"]
                prog0 = env.progress()
                res = env.step(a)
                prog1 = env.progress()
                ev = np.zeros(cfg.n_events, dtype=np.float32)
                ev[:len(env.events)] = [float(res.events.get(k, False)) for k in env.events]
                r = self.shaping * (self.gamma * prog1 - prog0) - self.step_cost
                if res.done:
                    r += 1.0 if res.success else 0.0
                feat_after = pad(env.features(), cfg.n_feat)
                ep.steps.append(CamStep(obs=obs_ids[j][:int((obs_ids[j] != 0).sum())].tolist(),  # ids of obs_t (pre-action)
                                        prev_act=list(s["prev_act"]), elapsed_prev=s["elapsed"], feat_obs=s["feat"],
                                        prev_opt=s["opt"], new_opt=bool(new[j]), term_sample=int(term[j]), opt=int(opt[j]),
                                        act_tokens=toks[j], act_ids=ids[j], allowed=allowed[j],
                                        logp_old=float(t_lp[j] + o_lp[j]) + sum(lps[j]), v_old=float(vals[j]),
                                        elapsed_h=res.elapsed_h, events=ev, feat_after=feat_after, prog_before=prog0,
                                        prog_after=prog1, reward=r, valid=res.valid, done=res.done))
                s["b"] = b[j]
                s["prev_act"] = toks[j]
                s["elapsed"] = res.elapsed_h
                s["opt"] = int(opt[j])
                s["feat"] = feat_after
                s["t"] += 1
                total_steps += 1
                if not res.done and len(ep.steps) >= 2 * env.max_steps:  # hard guard: no environment may run forever
                    res.done, res.success = True, False
                    env.done, env.success = True, False
                    ep.steps[-1].done = True
                    ep.truncated_by_guard = True
                if res.done:
                    ep.success = bool(res.success)
                    done_eps.append(ep)
                    slots[i] = None
                    slots[i] = start(i) if want_more() else None
            self.stats["env_s"] += time.time() - t1
            self.stats["steps"] += len(act)
            self.stats["rounds"] += 1
        return done_eps

    def _plan(self, b, g, opt, grams, toks, ids, lps, allowed, k, S):
        """Selective model-based action choice (evaluation only): sample ``k`` extra candidates and
        execute the one whose imagined end-of-task success probability is highest."""
        m = self.model
        cands = [[(toks[i], ids[i], lps[i], allowed[i])] for i in range(len(toks))]
        for _ in range(k):
            t2, i2, l2, a2 = m.generate_action(b, g, opt, grams, greedy=False, gen=self.gen)
            for i in range(len(toks)):
                cands[i].append((t2[i], i2[i], l2[i], a2[i]))
        flat_i, flat_c = [], []
        for i, cs in enumerate(cands):
            for c in cs:
                flat_i.append(i)
                flat_c.append(c[0])
        idx = torch.tensor(flat_i, device=m.device)
        act_ids = m.ids([["<bos>"] + c for c in flat_c], m.cfg.max_act + 2)
        hz = torch.full((len(flat_i),), m.cfg.horizons.index(0), device=m.device)
        el = torch.full((len(flat_i),), 48.0, device=m.device)
        pred = m.predict_effects(b[idx], act_ids, el, hz)
        score = torch.sigmoid(pred["success"]) * torch.sigmoid(pred["term"]) + 0.1 * pred["prog_mu"]
        out = ([], [], [], [])
        p = 0
        for i, cs in enumerate(cands):
            j = int(score[p:p + len(cs)].argmax())
            p += len(cs)
            for o, v in zip(out, cs[j]):
                o.append(v)
        return out


def summarize(eps: list[CamEpisode]) -> dict:
    out = {}
    for fam in sorted({e.family for e in eps}):
        es = [e for e in eps if e.family == fam]
        steps = [s for e in es for s in e.steps]
        n = len(es)
        k = sum(e.success for e in es)
        p = k / n if n else 0.0
        half = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / max(1, n))
        out[fam] = {"n": n, "success": k, "rate": p, "ci95": [round(max(0, p - half), 3), round(min(1, p + half), 3)],
                    "final_progress": float(np.mean([e.steps[-1].prog_after for e in es])) if es else 0.0,
                    "avg_steps": len(steps) / max(1, n), "valid_rate": float(np.mean([s.valid for s in steps])) if steps else 0.0,
                    "info_action_rate": float(np.mean([s.act_tokens[0] in INFO_ACTIONS for s in steps])) if steps else 0.0,
                    "options_used": len({s.opt for s in steps})}
    return out
