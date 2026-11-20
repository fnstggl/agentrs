"""Inner learner: recurrent actor-critic with a pluggable learning rule (default PPO-clip), auxiliary world-model
learning, replay-based retention regularisation and optional imagination training.

* On-policy part: whole episodes collected by the current policy (segments = episodes, so truncated BPTT runs over
  each episode from its stored starting state). The behaviour log-probability of every decision (deliberation +
  structured candidate with the stored planner bias + LLM text tokens) is recomputed under the current weights;
  ``replay_mismatch`` logs |logp_new - logp_old| before the first optimiser step (should be ~0).
* Advantages: GAE(lambda) with terminal handling (no bootstrap after a terminal state) and truncation handling
  (bootstrap with the value of the final observation).
* Replay (older episodes, any task family) is used ONLY for world-model training and for a CLEAR-style behaviour
  KL / value anchor (``distill_coef``) — never for an importance-unweighted policy gradient.
* World model: ensemble with bootstrap masks; targets are the next step's (stop-gradient) core features and
  integration summary, the reward and the termination flag.
* Imagination (``imagine_coef`` > 0): actor-critic on H-step rollouts of the world model from real on-policy
  states, scaled by the world model's measured held-out reliability (0 until measured).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from agentrs.cam2.featurize import StepInput, ToolRegistry, collate
from agentrs.cam2.model import CAM
from agentrs.cam2.search.rules import MetaTargetNet, Rule, ppo_clip


@dataclass
class LearnConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
    gamma: float = 0.95
    lam: float = 0.9
    epochs: int = 2
    mb_episodes: int = 32
    vf: float = 0.5
    ent: float = 0.01
    q_coef: float = 0.25
    wm_coef: float = 0.5
    distill_coef: float = 0.0
    distill_v: float = 0.1
    replay_episodes: int = 32          # replay episodes per update (world model + retention terms)
    replay_capacity: int = 4000        # episodes
    imagine_coef: float = 0.0
    imagine_horizon: int = 3
    max_grad: float = 1.0
    target_kl: float = 0.08
    episodes_per_update: int = 64
    temperature: float = 1.0
    rule: str = field(default_factory=lambda: ppo_clip().to_json())
    meta_target: Optional[list] = None  # MetaTargetNet flat parameters (outer-searched), None = off
    carry_state: bool = False          # keep recurrent state across episodes of a stream
    text_chunk: int = 16
    warmup_steps: int = 20             # linear learning-rate warm-up (optimizer steps)
    # Exact text-log-prob replay: shift the learner's recomputed control vectors by (stored bf16 vectors - their
    # recomputation at the behaviour parameters), so the bf16 cast reproduces the behaviour inputs exactly.
    # Off in all reported MiniWoB experiments (they predate it; measured residual mismatch is reported instead).
    prefix_replay_correction: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Transition:
    step: StepInput
    h0: dict                     # recurrent state entering this step (cpu)
    mem: Optional[torch.Tensor]  # (K, raw) fp16 rows retrieved at this step
    mem_m: Optional[torch.Tensor]
    cand: int
    logp_cand: float
    delib: int
    logp_delib: float
    bias: Optional[torch.Tensor]  # (Nc,) planner bias used at collection (fp32)
    logits: torch.Tensor          # (Nc,) policy logits at collection (fp16), for behaviour KL on replay
    gen: Optional[dict]
    logp_text: float
    value: float
    f: torch.Tensor               # (D,) fp16 core features (memory writes)
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    boot_value: float = 0.0
    invalid: bool = False
    delib_used: bool = False


@dataclass
class Episode:
    tr: list
    task: object
    success: bool = False
    score: float = 0.0
    ret: float = 0.0
    phase: str = ""
    version: int = 0
    family: str = ""
    info: dict = field(default_factory=dict)


def gae(ep: Episode, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GAE(lambda). After a terminal step nothing is bootstrapped; after a truncated (time-limit) last step the
    value of the final observation (``boot_value``) is."""
    T = len(ep.tr)
    adv = np.zeros(T, dtype=np.float32)
    td = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        x = ep.tr[t]
        if t + 1 < T:
            nv, nonterm = ep.tr[t + 1].value, 1.0
        else:
            nonterm = 0.0 if x.terminated else 1.0
            nv = x.boot_value if not x.terminated else 0.0
        delta = x.reward + gamma * nv * nonterm - x.value
        td[t] = delta
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
    ret = adv + np.array([x.value for x in ep.tr], dtype=np.float32)
    return adv, ret, td


def _stack_state(eps: list, cam: CAM, device) -> dict:
    keys = list(eps[0].tr[0].h0.keys())
    return {k: torch.stack([e.tr[0].h0[k] for e in eps]).to(device).float() for k in keys}


class Learner:
    def __init__(self, cam: CAM, llm, reg: ToolRegistry, cfg: LearnConfig, device):
        self.cam, self.llm, self.reg, self.cfg, self.device = cam, llm, reg, cfg, device
        self.opt = torch.optim.AdamW(cam.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, eps=1e-6)
        self.rule = Rule.from_json(cfg.rule)
        self.meta = None
        if cfg.meta_target is not None:
            self.meta = MetaTargetNet().to(device)
            self.meta.load_flat(cfg.meta_target)
        self.updates = 0
        self.reliability = 0.0
        self.rng = random.Random(0)

    # ------------------------------------------------------------------ batched episode forward
    def forward_episodes(self, eps: list, need_text: bool, wm_masks: bool = True) -> dict:
        cam, dev = self.cam, self.device
        eps = sorted(eps, key=lambda e: -len(e.tr))
        T = len(eps[0].tr)
        st = _stack_state(eps, cam, dev)
        out = {"eps": eps, "steps": []}
        for t in range(T):
            n = sum(1 for e in eps if len(e.tr) > t)
            trs = [e.tr[t] for e in eps[:n]]
            st = {k: v[:n] for k, v in st.items()}
            mem = mm = None
            if cam.cfg.mem_k > 0:
                mem = torch.stack([x.mem for x in trs]) if trs[0].mem is not None else None
                mm = torch.stack([x.mem_m for x in trs]) if trs[0].mem_m is not None else None
            b = collate(self.llm, self.reg, [x.step for x in trs], mem, mm, dev, goal_cond=cam.cfg.el_goal_cond)
            f, el_ctx, new, xs = cam.integrate(b, st)
            cand = cam.candidates(b, el_ctx)
            logits = cam.policy_logits(f, cand, b["cmask"])
            Nc = logits.shape[1]
            bias = torch.zeros_like(logits)
            for i, x in enumerate(trs):
                if x.bias is not None:
                    bias[i, : x.bias.shape[0]] = x.bias.to(dev)
            a = torch.tensor([x.cand for x in trs], device=dev)
            lp_all = torch.log_softmax(logits + bias, -1)
            logp_c = lp_all.gather(1, a.unsqueeze(1)).squeeze(1)
            p_all = lp_all.exp()
            ent = -(p_all * lp_all.masked_fill(~b["cmask"], 0.0)).sum(-1)
            logp_d = torch.zeros(n, device=dev)
            if cam.cfg.deliberation:
                dl = torch.log_softmax(cam.delib_logits(f), -1)
                lpd = dl.gather(1, torch.tensor([x.delib for x in trs], device=dev).unsqueeze(1)).squeeze(1)
                used = torch.tensor([x.delib_used for x in trs], device=dev)
                logp_d = torch.where(used, lpd, torch.zeros_like(lpd))
            c_sel = cand[torch.arange(n, device=dev), a]
            gen_txt = [x.gen["text"] if x.gen else "" for x in trs]
            gm = torch.tensor([x.gen is not None for x in trs], device=dev)
            gt = torch.as_tensor(self.llm.encode_pooled([g or "(none)" for g in gen_txt])).to(dev).float()
            a_emb = cam.action_embedding(c_sel, gt, gm)
            wm = cam.wm_predict(f, a_emb)
            q = cam.q_values(f, c_sel.unsqueeze(1)).squeeze(1) if cam.cfg.q_head else None
            beh = torch.full_like(logits, -1e9)
            for i, x in enumerate(trs):
                k = x.logits.shape[0]
                beh[i, :k] = x.logits.to(dev).float() + (x.bias.to(dev) if x.bias is not None else 0.0)
            out["steps"].append({"n": n, "trs": trs, "f": f, "x": xs, "logp_c": logp_c, "logp_d": logp_d, "ent": ent,
                                 "v": cam.v(f), "q": q, "wm": wm, "c_sel": c_sel, "lp_all": lp_all, "beh": beh,
                                 "cmask": b["cmask"], "cand": cand})
            st = new
        if need_text:
            # Text log-probabilities through the frozen LLM are evaluated WITHOUT keeping an autograd graph here;
            # ``text_backward`` later recomputes them in bounded chunks with the per-row loss gradient (exact, and
            # memory no longer grows with the number of text-generating steps in a minibatch).
            rows = [(t, i) for t, s in enumerate(out["steps"]) for i, x in enumerate(s["trs"]) if x.gen is not None]
            for t, s in enumerate(out["steps"]):
                s["logp_t"] = torch.zeros(s["n"], device=dev)
            out["text"] = None
            if rows:
                fs = torch.stack([out["steps"][t]["f"][i] for t, i in rows])
                cs = torch.stack([out["steps"][t]["c_sel"][i] for t, i in rows])
                pre = cam.prefix(fs, cs) if cam.cfg.prefix_len > 0 else None
                trs = [out["steps"][t]["trs"][i] for t, i in rows]
                if pre is not None and self.cfg.prefix_replay_correction:
                    pre = pre + torch.stack([x.gen["delta"] if x.gen.get("delta") is not None else torch.zeros_like(pre[0])
                                             for x in trs])
                ctx, stub, tok = [x.gen["ctx"] for x in trs], [x.gen["stub"] for x in trs], [x.gen["tokens"] for x in trs]
                with torch.no_grad():
                    lpt = self.llm.logprob_with_prefix(None if pre is None else pre.detach(), ctx, stub, tok,
                                                       temperature=self.cfg.temperature).float()
                leaf = lpt.clone().requires_grad_(pre is not None)
                for j, (t, i) in enumerate(rows):
                    s = out["steps"][t]
                    s["logp_t"] = s["logp_t"].index_put((torch.tensor([i], device=dev),), leaf[j:j + 1])
                out["text"] = {"leaf": leaf, "pre": pre, "ctx": ctx, "stub": stub, "tokens": tok}
        return out

    @torch.no_grad()
    def prefix_deltas(self, eps: list) -> int:
        """At the behaviour parameters (before any optimiser step of this update) recompute every text row's control
        vectors and store ``delta = stored_bf16 - recomputed`` (fp32, ~1e-7 relative). Adding ``delta`` later makes
        the value cast to bf16 equal the behaviour input regardless of batch composition."""
        n = 0
        for s0 in range(0, len(eps), max(1, self.cfg.mb_episodes)):
            out = self.forward_episodes(eps[s0:s0 + self.cfg.mb_episodes], need_text=False)
            for s in out["steps"]:
                for i, x in enumerate(s["trs"]):
                    if x.gen is not None and x.gen.get("prefix") is not None:
                        pre = self.cam.prefix(s["f"][i:i + 1], s["c_sel"][i:i + 1])[0]
                        x.gen["delta"] = x.gen["prefix"].to(self.device).float() - pre
                        n += 1
        return n

    def text_surrogate(self, loss: torch.Tensor, out: dict) -> torch.Tensor:
        """Exact, memory-bounded gradient of ``loss`` through the frozen LLM's text log-probabilities.

        ``loss`` depends on each text-generating row's log-probability only through a leaf scalar ``l_j``. With
        ``g_j = dloss/dl_j`` and ``G_j = d(g_j * l_j(prefix_j))/dprefix_j`` (recomputed through the LLM in chunks of
        ``text_chunk`` rows), the surrogate ``loss + sum_j <prefix_j, G_j>`` has exactly the gradient of the original
        objective with respect to every CAM parameter (``prefix`` stays in CAM's graph; ``G`` is a constant)."""
        T = out.get("text")
        if not T or T["pre"] is None:
            return loss
        (g,) = torch.autograd.grad(loss, T["leaf"], retain_graph=True, allow_unused=True)
        if g is None:
            return loss
        pre = T["pre"]
        G = torch.zeros_like(pre)
        ch = max(1, self.cfg.text_chunk)
        for s in range(0, pre.shape[0], ch):
            sl = slice(s, s + ch)
            if not bool((g[sl] != 0).any()):
                continue
            pc = pre[sl].detach().requires_grad_(True)
            lp = self.llm.logprob_with_prefix(pc, T["ctx"][sl], T["stub"][sl], T["tokens"][sl],
                                              temperature=self.cfg.temperature).float()
            (gp,) = torch.autograd.grad(lp, pc, grad_outputs=g[sl].detach())
            G[sl] = gp
        return loss + (pre * G.detach()).sum()

    def wm_loss(self, out: dict, masks: bool = True) -> tuple[torch.Tensor, dict]:
        E = len(self.cam.wm)
        steps = out["steps"]
        tot, parts = 0.0, {"wm_f": 0.0, "wm_x": 0.0, "wm_r": 0.0, "wm_d": 0.0}
        cnt = 0
        for t, s in enumerate(steps):
            n = s["n"]
            r = torch.tensor([x.reward for x in s["trs"]], device=self.device)
            d = torch.tensor([float(x.terminated) for x in s["trs"]], device=self.device)
            wm = s["wm"]
            bm = (torch.rand(E, n, device=self.device) < 0.8).float() if masks else torch.ones(E, n, device=self.device)
            lr_ = ((wm["reward"] - r) ** 2 * bm).sum() / bm.sum().clamp(min=1)
            ld = (F.binary_cross_entropy_with_logits(wm["done"], d.expand(E, n), reduction="none") * bm).sum() / bm.sum().clamp(min=1)
            loss = lr_ + ld
            parts["wm_r"] += float(lr_.detach())
            parts["wm_d"] += float(ld.detach())
            if t + 1 < len(steps):
                n2 = steps[t + 1]["n"]
                tf = steps[t + 1]["f"].detach()
                tx = steps[t + 1]["x"].detach()
                m2 = bm[:, :n2]
                lf = (((wm["next_f"][:, :n2] - tf) ** 2).mean(-1) * m2).sum() / m2.sum().clamp(min=1)
                lx = (((wm["next_x"][:, :n2] - tx) ** 2).mean(-1) * m2).sum() / m2.sum().clamp(min=1)
                loss = loss + lf + lx
                parts["wm_f"] += float(lf.detach())
                parts["wm_x"] += float(lx.detach())
            tot = tot + loss
            cnt += 1
        return tot / max(1, cnt), {k: v / max(1, cnt) for k, v in parts.items()}

    @torch.no_grad()
    def wm_eval(self, eps: list, gamma: float) -> dict:
        """Held-out one-step world-model quality on real episodes the world model has not trained on:
        R^2 of next-feature and reward predictions, termination accuracy, and the calibration of the ensemble's
        one-step return disagreement against the actual one-step return error."""
        if not eps:
            return {}
        out = self.forward_episodes(eps, need_text=False)
        steps = out["steps"]
        pf, tf, pr, tr, pdn, tdn, us, errs = [], [], [], [], [], [], [], []
        for t, s in enumerate(steps):
            wm = s["wm"]
            r = torch.tensor([x.reward for x in s["trs"]], device=self.device)
            d = torch.tensor([float(x.terminated) for x in s["trs"]], device=self.device)
            pr.append(wm["reward"].mean(0))
            tr.append(r)
            pdn.append(torch.sigmoid(wm["done"].mean(0)))
            tdn.append(d)
            nv = torch.zeros_like(r)
            if t + 1 < len(steps):
                n2 = steps[t + 1]["n"]
                pf.append(wm["next_f"][:, :n2].mean(0))
                tf.append(steps[t + 1]["f"])
                nv[:n2] = self.cam.v(steps[t + 1]["f"])
            # one-step return: r + gamma (1-d) V(f')
            pv = torch.stack([self.cam.v(wm["next_f"][e]) for e in range(wm["next_f"].shape[0])])  # targets are f itself
            q_e = wm["reward"] + gamma * (1 - torch.sigmoid(wm["done"])) * pv
            us.append(q_e.std(0) if q_e.shape[0] > 1 else torch.zeros_like(r))
            errs.append((q_e.mean(0) - (r + gamma * (1 - d) * nv)).abs())

        def r2(p, y):
            p, y = torch.cat(p), torch.cat(y)
            var = ((y - y.mean(0)) ** 2).mean()
            return float(1 - ((p - y) ** 2).mean() / var.clamp(min=1e-8))

        res = {"wm_r2_next_f": r2(pf, tf) if pf else float("nan"), "wm_r2_reward": r2(pr, tr),
               "wm_done_acc": float(((torch.cat(pdn) > 0.5).float() == torch.cat(tdn)).float().mean())}
        u, er = torch.cat(us), torch.cat(errs)
        if u.numel() > 4 and float(u.std()) > 0:
            res["wm_unc_err_corr"] = float(torch.corrcoef(torch.stack([u, er]))[0, 1])
        order = u.argsort()
        uu, ee = u[order], er[order]
        u_max = float("inf")
        nb = max(1, len(uu) // 8)
        for i in range(0, len(uu), nb):  # largest disagreement level whose bin error stays below 0.25 return units
            if float(ee[i:i + nb].mean()) <= 0.25:
                u_max = float(uu[min(len(uu) - 1, i + nb - 1)])
            else:
                break
        res["wm_u_max"] = u_max if math.isfinite(u_max) else 1e9
        fr2 = res["wm_r2_next_f"] if math.isfinite(res["wm_r2_next_f"]) else 0.0
        res["wm_reliability"] = float(max(0.0, min(1.0, 0.5 * max(0.0, fr2) + 0.5 * max(0.0, res["wm_r2_reward"]))))
        return res

    def imagination_loss(self, out: dict) -> tuple[torch.Tensor, dict]:
        cam, H, g = self.cam, self.cfg.imagine_horizon, self.cfg.gamma
        fs = torch.cat([s["f"].detach() for s in out["steps"]])
        cands = [s["cand"].detach() for s in out["steps"]]
        cms = [s["cmask"] for s in out["steps"]]
        Nc = max(c.shape[1] for c in cands)
        cand = torch.cat([F.pad(c, (0, 0, 0, Nc - c.shape[1])) for c in cands])
        cm = torch.cat([F.pad(m, (0, Nc - m.shape[1]), value=False) for m in cms])
        ft = fs
        logps, rews, conts, vals = [], [], [], []
        for _ in range(H):
            lg = cam.policy_logits(ft, cand, cm)
            lp = torch.log_softmax(lg, -1)
            a = torch.multinomial(lp.exp().detach(), 1).squeeze(1)
            logps.append(lp.gather(1, a.unsqueeze(1)).squeeze(1))
            vals.append(cam.v(ft))
            with torch.no_grad():
                ae = cam.action_embedding(cand[torch.arange(len(a)), a], None, None)
                pred = cam.wm_predict(ft, ae)
                rews.append(pred["reward"].mean(0))
                conts.append(1 - torch.sigmoid(pred["done"].mean(0)))
                ft = pred["next_f"].mean(0)
        with torch.no_grad():
            boot = cam.v(ft)
            R = boot
            rets = []
            for h in reversed(range(H)):
                R = rews[h] + g * conts[h] * ((1 - self.cfg.lam) * vals[h + 1].detach() + self.cfg.lam * R
                                               if h + 1 < H else R)
                rets.append(R)
            rets = rets[::-1]
        actor = sum(-(rets[h] - vals[h].detach()) * logps[h] for h in range(H)).mean() / H
        critic = sum((vals[h] - rets[h]) ** 2 for h in range(H)).mean() / H
        return actor + 0.5 * critic, {"img_actor": float(actor), "img_critic": float(critic),
                                      "img_return": float(torch.stack(rets).mean())}

    # ------------------------------------------------------------------ the update
    def update(self, eps: list, replay: list, holdout: Optional[list] = None) -> dict:
        cfg, cam = self.cfg, self.cam
        t0 = time.time()
        stats: dict = {}
        if holdout and cfg.wm_coef > 0:
            stats.update(self.wm_eval(holdout, cfg.gamma))
            self.reliability = stats.get("wm_reliability", 0.0)
        advs = {}
        for e in eps:
            advs[id(e)] = gae(e, cfg.gamma, cfg.lam)
        all_adv = np.concatenate([a for a, _, _ in advs.values()])
        mu, sd = float(all_adv.mean()), float(all_adv.std() + 1e-8)
        if cfg.prefix_replay_correction and cam.cfg.prefix_len > 0:
            stats["prefix_deltas"] = self.prefix_deltas(eps)
        before = [p.detach().clone() for p in cam.parameters()]
        n_mb = max(1, math.ceil(len(eps) / cfg.mb_episodes))
        agg: dict = {}
        first = True
        stop = False
        steps_done = 0
        for ep_i in range(cfg.epochs):
            order = list(range(len(eps)))
            self.rng.shuffle(order)
            for mbi in range(n_mb):
                mb = [eps[i] for i in order[mbi::n_mb]]
                if not mb:
                    continue
                out = self.forward_episodes(mb, need_text=True)
                keys = ["logp", "logp_old", "adv", "adv_raw", "ret", "value", "td", "ent", "q", "rew", "v_new", "q_new"]
                cols = {k: [] for k in keys}
                for t, s in enumerate(out["steps"]):
                    trs = s["trs"]
                    lp = s["logp_c"] + s["logp_d"] + s["logp_t"]
                    cols["logp"].append(lp)
                    cols["logp_old"].append(torch.tensor([x.logp_cand + x.logp_delib + x.logp_text for x in trs], device=self.device))
                    a_raw = torch.tensor([advs[id(out["eps"][i])][0][t] for i in range(s["n"])], device=self.device)
                    cols["adv_raw"].append(a_raw)
                    cols["adv"].append((a_raw - mu) / sd)
                    cols["ret"].append(torch.tensor([advs[id(out["eps"][i])][1][t] for i in range(s["n"])], device=self.device))
                    cols["td"].append(torch.tensor([advs[id(out["eps"][i])][2][t] for i in range(s["n"])], device=self.device))
                    cols["value"].append(torch.tensor([x.value for x in trs], device=self.device))
                    cols["ent"].append(s["ent"])
                    cols["rew"].append(torch.tensor([x.reward for x in trs], device=self.device))
                    cols["v_new"].append(s["v"])
                    cols["q_new"].append(s["q"] if s["q"] is not None else torch.zeros(s["n"], device=self.device))
                    cols["q"].append((s["q"].detach() if s["q"] is not None else torch.zeros(s["n"], device=self.device)))
                C = {k: torch.cat(v) for k, v in cols.items()}
                logratio = C["logp"] - C["logp_old"]
                env = {"ratio": torch.exp(logratio.clamp(-20, 20)), "logratio": logratio, "logp": C["logp"],
                       "logp_old": C["logp_old"], "adv": C["adv"], "adv_raw": C["adv_raw"], "ret": C["ret"],
                       "value": C["value"], "td": C["td"], "ent": C["ent"].detach(), "kl": C["logp_old"] - C["logp"],
                       "q": C["q"], "rew": C["rew"]}
                if first:
                    stats["replay_mismatch"] = float(logratio.detach().abs().mean())
                    stats["replay_mismatch_max"] = float(logratio.detach().abs().max())
                    dc, dt = [], []
                    for s_ in out["steps"]:
                        dc.append((s_["logp_c"].detach() - torch.tensor([x.logp_cand for x in s_["trs"]], device=self.device)).abs())
                        m_ = torch.tensor([x.gen is not None for x in s_["trs"]], device=self.device)
                        d_ = (s_["logp_t"].detach() - torch.tensor([x.logp_text for x in s_["trs"]], device=self.device)).abs()
                        dt.append(d_[m_])
                    dc, dt = torch.cat(dc), torch.cat(dt)
                    stats["mismatch_cand_max"] = float(dc.max())
                    stats["mismatch_text_mean"] = float(dt.mean()) if dt.numel() else 0.0
                    stats["mismatch_text_max"] = float(dt.max()) if dt.numel() else 0.0
                    first = False
                pl = self.rule.policy.eval(env)
                if self.meta is not None:
                    pl = pl * self.meta(env).detach()
                pol = pl.mean()
                vt = self.rule.vtarget.eval({k: v.detach() for k, v in env.items()}).detach()
                vloss = ((C["v_new"] - vt) ** 2).mean()
                ent = C["ent"].mean()
                loss = pol + cfg.vf * vloss - cfg.ent * ent
                if cam.cfg.q_head:
                    loss = loss + cfg.q_coef * ((C["q_new"] - C["ret"]) ** 2).mean()
                wml = torch.zeros((), device=self.device)
                if cfg.wm_coef > 0:
                    wml, wparts = self.wm_loss(out)
                    for k, v in wparts.items():
                        agg.setdefault(k, []).append(v)
                if cfg.imagine_coef > 0 and self.reliability > 0:
                    il, iparts = self.imagination_loss(out)
                    loss = loss + cfg.imagine_coef * self.reliability * il
                    for k, v in iparts.items():
                        agg.setdefault(k, []).append(v)
                loss = loss + cfg.wm_coef * wml
                # ---- replay: world model + retention (behaviour KL / value anchor), no policy gradient
                if replay and (cfg.wm_coef > 0 or cfg.distill_coef > 0) and mbi == 0:
                    rb = self.rng.sample(replay, min(len(replay), cfg.replay_episodes))
                    rout = self.forward_episodes(rb, need_text=False)
                    if cfg.wm_coef > 0:
                        rw, _ = self.wm_loss(rout)
                        loss = loss + cfg.wm_coef * rw
                    if cfg.distill_coef > 0:
                        kls, vls = [], []
                        for s in rout["steps"]:
                            Nc = s["lp_all"].shape[1]
                            beh = torch.log_softmax(s["beh"][:, :Nc].masked_fill(~s["cmask"], -1e9), -1)
                            kls.append((beh.exp() * (beh - s["lp_all"]).masked_fill(~s["cmask"], 0.0)).sum(-1))
                            vls.append((s["v"] - torch.tensor([x.value for x in s["trs"]], device=self.device)) ** 2)
                        kl = torch.cat(kls).mean()
                        loss = loss + cfg.distill_coef * (kl + cfg.distill_v * torch.cat(vls).mean())
                        agg.setdefault("replay_kl", []).append(float(kl))
                if not torch.isfinite(loss):
                    stats["nonfinite_loss"] = stats.get("nonfinite_loss", 0) + 1
                    continue
                self.opt.zero_grad(set_to_none=True)
                self.text_surrogate(loss, out).backward()
                gn = torch.nn.utils.clip_grad_norm_(cam.parameters(), cfg.max_grad)
                self.opt_steps_total = getattr(self, "opt_steps_total", 0) + 1
                for gr in self.opt.param_groups:
                    gr["lr"] = cfg.lr * min(1.0, self.opt_steps_total / max(1, cfg.warmup_steps))
                self.opt.step()
                steps_done += 1
                approx_kl = float(((env["ratio"] - 1) - logratio).detach().mean())
                for k, v in {"loss": float(loss), "pol_loss": float(pol), "v_loss": float(vloss), "entropy": float(ent),
                             "grad_norm": float(gn), "approx_kl": approx_kl, "wm_loss": float(wml),
                             "clipfrac": float(((env["ratio"] - 1).abs() > 0.2).float().mean())}.items():
                    agg.setdefault(k, []).append(v)
                if approx_kl > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        stats.update({k: float(np.mean(v)) for k, v in agg.items()})
        # ---- measured magnitude of the neural update
        with torch.no_grad():
            num = sum(float(((p - b) ** 2).sum()) for p, b in zip(cam.parameters(), before))
            den = sum(float((b ** 2).sum()) for b in before)
            stats["update_rel_norm"] = math.sqrt(num / max(den, 1e-12))
            stats["params_changed"] = sum(int(not torch.equal(p, b)) for p, b in zip(cam.parameters(), before))
        if steps_done > 0 and stats["params_changed"] == 0:
            raise RuntimeError("optimizer steps were taken but no CAM parameter changed")
        stats["opt_steps"] = steps_done
        stats["update_s"] = time.time() - t0
        self.updates += 1
        return stats

    def state_dict(self) -> dict:
        return {"opt": self.opt.state_dict(), "updates": self.updates, "reliability": self.reliability,
                "rng": self.rng.getstate(), "opt_steps_total": getattr(self, "opt_steps_total", 0)}

    def load_state_dict(self, s: dict) -> None:
        self.opt.load_state_dict(s["opt"])
        self.updates = s["updates"]
        self.reliability = s.get("reliability", 0.0)
        self.opt_steps_total = s.get("opt_steps_total", 0)
        if "rng" in s:
            self.rng.setstate(s["rng"])
