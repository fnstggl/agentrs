"""Persistent batched workers for CAM-v2: many isolated episodes acting through one shared network.

Each round, every active worker's belief is advanced with the RSSM (prior transition on its previous
action, posterior on its new observation and verified status) in one batched pass, and the actor
generates the complete structured decision (option termination, option, grammar-masked action
tokens) with its exact behaviour log-probability. The environment executes the action; the reward
contract converts the verified status change into the training signal (semi-Markov discount, time
cost, failure charge, success, constraint-violation flag). Training workers can start from a
promising archived state (Go-Explore); its recorded history is replayed first as belief burn-in and is
excluded from learning (``Step.learn = False``).
"""
from __future__ import annotations

import random
import time
from typing import Optional

import numpy as np
import torch

from agentrs.cam.contract import RewardContract
from agentrs.cam.envs import ENVS
from agentrs.cam.envs.base import status_vector
from agentrs.cam.v2.data import Archive, Episode, Step
from agentrs.cam.text.render import goal_text, obs_text
from agentrs.cam.v2.model import CAM2


def pad(x: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    out[:len(x)] = x
    return out


class Collector:
    def __init__(self, model: CAM2, n: int, families: dict[str, float], split: str = "train", seed0: int = 0,
                 gen_seed: int = 0, reward: Optional[RewardContract] = None, lam: Optional[dict] = None,
                 difficulty: Optional[dict] = None, archive: Optional[Archive] = None, archive_frac: float = 0.0,
                 env_version: str = "", semantic=None, text_style: str = "train", frontend=None, store_feat: bool = False):
        self.model, self.n, self.families, self.split = model, n, families, split
        # natural-language interface: a frozen semantic encoder (semantic CAM) or a text->token front-end for a
        # structured CAM (rule-based parser or frozen-LLM extraction); texts are rendered from the agent's tokens only
        self.semantic, self.text_style, self.frontend = semantic, text_style, frontend
        self.store_feat = store_feat  # PPO: keep the actor's exact inputs for on-policy ratios
        if model.semantic and semantic is None:
            raise ValueError("a semantic CAM needs the frozen semantic encoder")
        self.seed_ctr = {f: seed0 for f in families}
        self.rng = random.Random(gen_seed)
        self.gen = torch.Generator(device=model.device).manual_seed(gen_seed)
        self.reward = reward or RewardContract()
        self.lam = lam if lam is not None else {}
        self.difficulty = difficulty if difficulty is not None else {}
        self.archive = archive if split == "train" else None
        self.archive_frac = archive_frac
        self.env_version = env_version
        self.stats = {"env_s": 0.0, "net_s": 0.0, "steps": 0, "rounds": 0}

    # ------------------------------------------------------------------ episode starts
    def _make_env(self, fam: str):
        env = ENVS[fam]()
        if self.split == "train":
            env.difficulty = float(self.difficulty.get(fam, 1.0))
        return env

    def _slot(self, fam, seed, env, version, start="scratch", cell=None):
        m = self.model
        h, z = m.initial(1)
        ep = Episode(fam, self.split, seed, env.goal_tokens(), version, env_version=self.env_version, start=start,
                     difficulty=float(getattr(env, "difficulty", 1.0)) if self.split == "train" else 1.0)
        s = {"env": env, "ep": ep, "h": h[0], "z": z[0], "prev_act": ["<bos>"], "elapsed": 0.0, "opt": 0,
             "st": env.task_status(), "c": env.contract(), "t": 0, "prefix": [], "cell": cell, "acted": 0,
             "feat": pad(env.features(), m.cfg.n_feat)}
        if cell is not None:
            ep.n_prefix = len(cell["steps"])
            s["prefix"] = list(cell["steps"])
        gt = goal_text(fam, ep.goal, self.text_style, key=(self.split, seed % 4, "goal"))  # 4 wordings per goal
        if m.semantic:
            s["goal_in"], s["goal_txt"] = None, gt  # encoded in one batch at the next round
        elif self.frontend is not None:
            s["goal_in"] = self.frontend.goal(fam, gt)
        else:
            s["goal_in"] = ep.goal
        return s

    def _text_key(self, s):
        e = s["ep"]
        return (self.split, e.seed, e.start, len(e.steps))

    def _start(self, queue, version):
        if queue is not None:
            if not queue:
                return None
            fam, seed = queue.pop(0)
            env = self._make_env(fam)
            env.reset(seed, self.split)
            return self._slot(fam, seed, env, version)
        fam = self.rng.choices(list(self.families), list(self.families.values()))[0]
        if self.archive is not None and self.rng.random() < self.archive_frac:
            cell = self.archive.choose(fam)
            if cell is not None:
                env = cell["env"].snapshot()
                env.reseed(self.rng.randrange(1 << 30))
                if self.split == "train":
                    env.difficulty = float(self.difficulty.get(fam, 1.0))
                    env.retarget(env.difficulty)
                s = self._slot(fam, cell["seed"], env, version, start="archive", cell=cell)
                return s
        sd = self.seed_ctr[fam]
        self.seed_ctr[fam] = sd + 1
        env = self._make_env(fam)
        env.reset(sd, self.split)
        return self._slot(fam, sd, env, version)

    # ------------------------------------------------------------------ main loop
    @torch.no_grad()
    def collect(self, min_episodes: int = 0, min_steps: int = 0, greedy: bool = False, version: str = "v0",
                fixed: Optional[list] = None) -> list[Episode]:
        m = self.model
        m.eval()
        dev = m.device
        cfg = m.cfg
        done_eps: list[Episode] = []
        queue = list(fixed) if fixed is not None else None
        slots: list[Optional[dict]] = [None] * self.n
        total = 0

        def want_more():
            if queue is not None:
                return bool(queue)
            return len(done_eps) + sum(s is not None for s in slots) < max(min_episodes, 1) or total < min_steps

        for i in range(self.n):
            if want_more():
                slots[i] = self._start(queue, version)
        while any(s is not None for s in slots):
            t0 = time.time()
            act = [i for i, s in enumerate(slots) if s is not None]
            S = [slots[i] for i in act]
            if m.semantic:
                miss = [s for s in S if s["goal_in"] is None]
                if miss:
                    for s, v in zip(miss, self.semantic.encode([s["goal_txt"] for s in miss])):
                        s["ep"].goal_sem = s["goal_in"] = v
            # inputs: from the replayed archive prefix (burn-in) or from the live environment
            obs_rows, prev_rows, el, stv, goals, sems, need, fe = [], [], [], [], [], [], [], []
            for j, s in enumerate(S):
                fam = s["ep"].family
                if s["prefix"]:
                    p = s["prefix"][0]
                    obs_rows.append(list(p.obs))
                    prev_rows.append(list(p.prev_act))
                    el.append(p.elapsed_prev)
                    stv.append(p.status)
                    sems.append(p.sem)
                else:
                    toks = s["env"].obs_tokens()
                    obs_rows.append([m.tok[t] for t in [f"FAM_{fam}"] + toks][:cfg.max_obs])
                    if m.semantic or self.frontend is not None:
                        txt = obs_text(fam, toks, self.text_style, key=self._text_key(s))
                        if m.semantic:
                            need.append((j, txt))
                        else:  # structured CAM reading text through a front-end (parser / LLM extraction)
                            fe.append((j, fam, txt))
                    sems.append(None)
                    prev_rows.append(s["prev_act"])
                    el.append(s["elapsed"])
                    stv.append(status_vector(s["c"], s["st"]))
                goals.append(s["goal_in"])
            if fe:
                for (j, fam, _), toks in zip(fe, self.frontend.obs_batch([f for _, f, _ in fe], [t for _, _, t in fe])):
                    obs_rows[j] = [m.tok[t] for t in [f"FAM_{fam}"] + list(toks) if t in m.tok][:cfg.max_obs]
            prev_ids = m.ids(prev_rows, cfg.max_act + 2)
            elt = torch.tensor(el, device=dev, dtype=torch.float32)
            st_t = torch.tensor(np.stack(stv), device=dev, dtype=torch.float32)
            if m.semantic:
                if need:
                    for (j, _), v in zip(need, self.semantic.encode([t for _, t in need])):
                        sems[j] = v
                obs_in = torch.tensor(np.stack(sems), device=dev)
                g = m.encode_goal(torch.tensor(np.stack(goals), device=dev))
            else:
                obs_in = m.pad_ids(obs_rows, cfg.max_obs)
                g = m.encode_goal(m.ids([[f"FAM_{s['ep'].family}"] + list(gl_) for s, gl_ in zip(S, goals)], 48))
            h_prev = torch.stack([s["h"] for s in S])
            z_prev = torch.stack([s["z"] for s in S])
            h, _ = m.img_step(h_prev, z_prev, m.wm_act_enc(prev_ids))
            e = m.encode_obs(obs_in, elt, st_t)
            z = m.sample_z(m.post_logits(h, e), mode=greedy, gen=self.gen)
            feat = m.feat(h, z)
            for j, s in enumerate(S):
                s["h"], s["z"] = h[j], z[j]
            live = [j for j, s in enumerate(S) if not s["prefix"]]
            # ---- burn-in slots: advance through their recorded prefix without acting
            for j, s in enumerate(S):
                if s["prefix"]:
                    p = s["prefix"].pop(0)
                    q = Step(**{k: getattr(p, k) for k in Step.__dataclass_fields__})
                    q.learn = False
                    s["ep"].steps.append(q)
                    if not s["prefix"]:  # burn-in complete: continue from the archived state
                        cell = s["cell"]
                        s["prev_act"], s["opt"], s["elapsed"] = list(cell["prev_act"]), cell["opt"], cell["elapsed"]
                        s["t"] = len(s["ep"].steps)
            if live:
                L = torch.tensor(live, device=dev)
                fl, gl = feat[L], g[L]
                cur = torch.tensor([S[j]["opt"] for j in live], device=dev)
                logits, beta = m.actor_dists(fl, gl, cur)
                pb = torch.sigmoid(beta)
                term = (pb > 0.5).long() if greedy else torch.bernoulli(pb, generator=self.gen).long()
                first = torch.tensor([S[j]["acted"] == 0 for j in live], device=dev)
                new = first | term.bool()
                olp = torch.log_softmax(logits.float(), -1)
                o_s = olp.argmax(-1) if greedy else torch.multinomial(olp.exp(), 1, generator=self.gen).squeeze(-1)
                opt = torch.where(new, o_s, cur)
                t_lp = torch.where(first, torch.zeros_like(pb),
                                   torch.where(term.bool(), torch.log(pb + 1e-8), torch.log(1 - pb + 1e-8)))
                o_lp = torch.where(new, olp.gather(-1, o_s[:, None]).squeeze(-1), torch.zeros_like(pb))
                grams = [S[j]["env"].grammar() for j in live]
                toks, ids, lps = m.generate_action(fl, gl, opt, grams, greedy=greedy, gen=self.gen)
                u_taken = None
                if self.archive is not None and self.archive.investigate > 0:  # INVESTIGATE: world-model disagreement
                    u_taken = m.ens_pred(fl, m.wm_act_enc(m.ids(toks, cfg.max_act + 2))).var(0).mean(-1).tolist()
                self.stats["net_s"] += time.time() - t0
                t1 = time.time()
                pend: list = []  # semantic mode: terminal observations / hindsight goals to encode in one batch
                for jj, j in enumerate(live):
                    s = S[j]
                    env, ep = s["env"], s["ep"]
                    a = [t for t in toks[jj] if t != "END"]
                    s0 = s["st"]
                    res = env.step(a)
                    s1 = env.task_status()
                    s["acted"] += 1
                    n_steps = len(ep.steps) + 1
                    guard = (not res.done) and n_steps >= 2 * env.max_steps
                    done = bool(res.done or guard)
                    succ = bool(res.success) if res.done else False
                    tr = self.reward.transition(s["c"], s0, s1, done, succ, 0.0, n_steps, env.max_steps)
                    sp = self.reward.spec
                    if sp.mode == "contract":
                        r_shape = sp.shaping * (tr.discount * (0.0 if done else s1.progress) - s0.progress)
                        r_succ = sp.success if (done and succ) else 0.0
                        r_fail = 0.0
                        if done and not succ:
                            r_fail = -sp.time_cost * tr.discount * self.reward._ann(max(0.0, s["c"].horizon - s1.clock))
                        r_time = tr.reward - r_shape - r_succ - r_fail
                    else:
                        r_shape, r_succ, r_fail = 0.0, 0.0, 0.0
                        r_time = tr.reward
                    stp = Step(obs=obs_rows[j], prev_act=list(s["prev_act"]), elapsed_prev=s["elapsed"],
                               status=stv[j], feat_obs=s["feat"],
                               prev_opt=int(cur[jj]), new_opt=bool(new[jj]), term_sample=int(term[jj]), opt=int(opt[jj]),
                               act_tokens=toks[jj], act_ids=ids[jj], logp_b=float(t_lp[jj] + o_lp[jj]) + sum(lps[jj]),
                               r_shape=r_shape, r_time=r_time, r_fail=r_fail, r_succ=r_succ, viol=tr.violation,
                               disc=tr.discount, dt=tr.dt, elapsed_h=res.elapsed_h,
                               events=pad(np.array([float(res.events.get(k, False)) for k in env.events], np.float32), cfg.n_events),
                               feat_after=pad(env.features(), cfg.n_feat), status_after=status_vector(s["c"], s1),
                               prog_before=s0.progress, prog_after=s1.progress, done=done, valid=res.valid,
                               first=bool(first[jj]), sem=sems[j],
                               feat=fl[jj].to(torch.float16).cpu().numpy() if self.store_feat else None)
                    if self.store_feat and ep.goal_feat is None:
                        ep.goal_feat = gl[jj].to(torch.float16).cpu().numpy()
                    ep.steps.append(stp)
                    s["feat"] = stp.feat_after
                    s["st"], s["prev_act"], s["elapsed"], s["opt"] = s1, toks[jj], res.elapsed_h, int(opt[jj])
                    s["t"] += 1
                    total += 1
                    if not done and self.archive is not None:
                        self.archive.offer(ep.family, env, ep, s, u=u_taken[jj] if u_taken else 0.0)
                    if done:
                        if guard:
                            env.done, env.success = True, False
                            ep.truncated_by_guard = True
                        ep.success = succ
                        ep.cause = None if succ else ("guard" if guard else env.failure_cause())
                        ep.outcome = "success" if succ else ("violation" if s1.violated else "failure")
                        ep.milestones = {k: bool(v) for k, v in s1.milestones.items()}
                        ep.final = {"time_left": s1.time_left, "resources": dict(s1.resources), "clock": s1.clock}
                        ep.final_obs = [m.tok[t] for t in [f"FAM_{ep.family}"] + env.obs_tokens()][:cfg.max_obs]
                        ep.final_elapsed = float(res.elapsed_h)
                        if m.semantic:
                            pend.append((ep, "final", obs_text(ep.family, env.obs_tokens(), self.text_style, key=self._text_key(s))))
                        if self.split == "train" and not succ:
                            ep.hindsight = [(list(gl_), u) for gl_, u in env.hindsight_goals()]
                            if m.semantic:
                                for hi, (gl_, _) in enumerate(ep.hindsight):
                                    pend.append((ep, hi, goal_text(ep.family, gl_, self.text_style, key=(self.split, ep.seed % 4, "goal"))))
                        if s["cell"] is not None and self.archive is not None:
                            self.archive.result(s["cell"], succ)
                        done_eps.append(ep)
                        slots[act[j]] = self._start(queue, version) if want_more() else None
                if pend:
                    for (ep_, what, _), v in zip(pend, self.semantic.encode([t for _, _, t in pend])):
                        if what == "final":
                            ep_.final_sem = v
                        else:
                            gl_, u_ = ep_.hindsight[what][:2]
                            ep_.hindsight[what] = (gl_, u_, v)
                self.stats["env_s"] += time.time() - t1
            self.stats["steps"] += len(live)
            self.stats["rounds"] += 1
        return done_eps


def summarize(eps: list[Episode]) -> dict:
    """Verified outcomes per family (only steps the agent took count; archive prefixes are excluded)."""
    import math

    out = {}
    for fam in sorted({e.family for e in eps}):
        es = [e for e in eps if e.family == fam]
        n = len(es)
        k = sum(e.success for e in es)
        p = k / n if n else 0.0
        half = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / max(1, n))
        steps = [s for e in es for s in e.steps if s.learn]
        ms_keys = sorted({kk for e in es for kk in e.milestones})
        out[fam] = {"n": n, "success": k, "rate": p, "ci95": [round(max(0, p - half), 3), round(min(1, p + half), 3)],
                    "final_progress": float(np.mean([e.steps[-1].prog_after for e in es])) if es else 0.0,
                    "violation_rate": float(np.mean([e.outcome == "violation" for e in es])) if es else 0.0,
                    "time_left_at_end": float(np.mean([e.final.get("time_left", 0.0) for e in es])) if es else 0.0,
                    "resources_at_end": float(np.mean([np.mean(list(e.final.get("resources", {}).values()) or [0.0])
                                                       for e in es])) if es else 0.0,
                    "milestones": {kk: float(np.mean([e.milestones.get(kk, False) for e in es])) for kk in ms_keys},
                    "causes": {c: sum(e.cause == c for e in es) for c in sorted({str(e.cause) for e in es if not e.success})},
                    "avg_steps": len(steps) / max(1, n), "valid_rate": float(np.mean([s.valid for s in steps])) if steps else 0.0,
                    "from_archive": sum(e.start == "archive" for e in es)}
    return out
