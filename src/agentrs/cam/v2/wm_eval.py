"""Validation of the learned world model against the real simulators (evaluation only).

* ``open_loop``: from real posterior states of held-out episodes, roll the RSSM prior forward with the
  *recorded* actions and compare predicted discounted contract returns, continuation and constraint
  violation with what actually happened, per horizon (1, 2, 4, 8); calibration of violation
  forecasts by horizon; ensemble disagreement vs actual one-step error (Spearman) and its quantiles;
  errors on rare actions and on episodes ending in success / violation.
* ``counterfactual_ranking``: from identical simulator snapshots (hidden state never shown to CAM),
  execute alternative first actions (e.g. large campaign, modest campaign, product improvement,
  research, waiting) followed by CAM's own greedy policy, and test whether the world model ranks the
  alternatives like the actual consequences do. The best *actual* alternative is reported as a
  simulator-oracle reference (it uses exact simulator rollouts: oracle information).
* ``corrupted``: a copy of the model with perturbed dynamics/reward heads, to check that the
  reliability measures detect an untrustworthy world model.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch

from agentrs.cam.contract import RewardContract
from agentrs.cam.envs import ENVS
from agentrs.cam.envs.base import status_vector
from agentrs.cam.v2.data import View
from agentrs.cam.v2.learner import _grammar, observe


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def _ece(p, y, bins=10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    tot = 0.0
    for b in range(bins):
        msk = (p >= b / bins) & ((p < (b + 1) / bins) if b < bins - 1 else (p <= 1))
        if msk.any():
            tot += msk.mean() * abs(p[msk].mean() - y[msk].mean())
    return float(tot)


@torch.no_grad()
def open_loop(m, eps, lam: dict, ks=(1, 2, 4, 8), max_starts: int = 6, seed: int = 0) -> dict:
    m.eval()
    rng = random.Random(seed)
    dev = m.device
    views = [View(e, list(e.goal), len(e.steps)) for e in eps if e.n_learn > 0]
    out = {f"ret_abs_err_k{k}": [] for k in ks}
    viol_p = {k: [] for k in ks}
    viol_y = {k: [] for k in ks}
    cont_acc = []
    u_list, err1, rare_err, rew_err_by_out = [], [], [], {"success": [], "violation": [], "failure": []}
    act_freq: dict = {}
    for v in views:
        for s in v.steps:
            act_freq[s.act_tokens[0]] = act_freq.get(s.act_tokens[0], 0) + 1
    rare = {a for a, c in act_freq.items() if c <= np.percentile(list(act_freq.values()), 25)} if act_freq else set()
    for s0 in range(0, len(views), 32):
        vb = views[s0:s0 + 32]
        ob = observe(m, vb)
        for i, v in enumerate(vb):
            st = v.steps
            T = len(st)
            starts = [t for t in range(v.ep.n_prefix, T)]
            rng.shuffle(starts)
            g = ob["goal"][i:i + 1]
            r_real = v.rewards(lam.get(v.ep.family, 0.0))
            for t in starts[:max_starts]:
                f = ob["feat"][ob["pos"][(i, t)]][None]
                h, z = f[:, :m.cfg.deter], f[:, m.cfg.deter:]
                Kmax = max(ks)
                pred, D, alive, pv_none = 0.0, 1.0, 1.0, 1.0
                real, Dr, ended, viol_real = 0.0, 1.0, False, False
                for k in range(1, Kmax + 1):
                    j = t + k - 1
                    if j < T:
                        a_e = m.wm_act_enc(m.ids([st[j].act_tokens], m.cfg.max_act + 2))
                        if k == 1:
                            u = float(m.ens_pred(m.feat(h, z), a_e).var(0).mean())
                            nxt_e = ob["e"][ob["pos"][(i, t + 1)]]
                            ens_err = float(((m.ens_pred(m.feat(h, z), a_e).mean(0)[0] - nxt_e) ** 2).mean())
                            u_list.append(u)
                            err1.append(ens_err)
                        h, pr = m.img_step(h, z, a_e)
                        z = m.sample_z(pr, mode=True)
                        H = m.heads(m.feat(h, z), g)
                        pviol = float(torch.sigmoid(H["viol"]))
                        pred += D * alive * (float(H["rew"]) - lam.get(v.ep.family, 0.0) * pviol)
                        D *= float((0.99 ** torch.expm1(H["dt"].clamp(-3, 5)).clamp(min=0)))
                        pv_none *= (1 - pviol * alive)
                        c = float(torch.sigmoid(H["cont"]))
                        if k == 1:
                            cont_acc.append(float((c > 0.5) == (not st[j].done)))
                            e1 = abs(float(H["rew"]) - r_real[j])
                            if st[j].act_tokens[0] in rare:
                                rare_err.append(e1)
                            if st[j].done:
                                rew_err_by_out[v.ep.outcome or "failure"].append(e1)
                        alive *= c
                    if not ended and j < T:
                        real += Dr * r_real[j]
                        Dr *= st[j].disc
                        if st[j].done:
                            ended = True
                            viol_real = st[j].viol > 0
                    if k in ks:
                        out[f"ret_abs_err_k{k}"].append(abs(pred - real))
                        viol_p[k].append(1 - pv_none)
                        viol_y[k].append(float(viol_real))
    res = {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}
    for k in ks:
        res[f"viol_ece_k{k}"] = _ece(viol_p[k], viol_y[k])
        res[f"viol_brier_k{k}"] = float(np.mean((np.array(viol_p[k]) - np.array(viol_y[k])) ** 2)) if viol_p[k] else float("nan")
    res["cont_acc_k1"] = float(np.mean(cont_acc)) if cont_acc else float("nan")
    res["u_err_spearman"] = _spearman(u_list, err1)
    for q in (50, 80, 90):
        res[f"u_q{q}"] = float(np.percentile(u_list, q)) if u_list else 0.0
    res["rew_abs_err_rare_actions_k1"] = float(np.mean(rare_err)) if rare_err else float("nan")
    for o, v in rew_err_by_out.items():
        res[f"rew_abs_err_terminal_{o}"] = float(np.mean(v)) if v else float("nan")
    res["n_starts"] = len(out[f"ret_abs_err_k{ks[0]}"])
    return res


CANDIDATES = {
    "product": lambda env: [["CAMPAIGN", "ads", "B4"], ["CAMPAIGN", "ads", "B2"],
                            ["BUILD", next((f"F{i}" for i in range(6) if not env.built[i] and i not in env.building), "F0")],
                            ["RESEARCH", "S0"], ["WAIT", "W1"]],
    "negotiation": lambda env: [["ASK", "price"],
                                ["PROPOSE", *[("N10" if d > 0 else "N0") for d in env.d_a], "NOARG"],
                                ["PROPOSE", "N5", "N5", "N5", "N5", "NOARG"]] + ([["ACCEPT"]] if env.last_cp is not None else [])
                                + [["WALK"]],
    "bizdev": lambda env: [["RESEARCH", "P0"], ["PITCH", "P0", "N4", "NOEXCL", "SLOW"], ["PITCH", "P0", "N8", "EXCL", "FAST"],
                           ["FOLLOWUP", "P0"], ["WAIT"]],
}


class _Agent:
    """Greedy CAM acting in one environment copy (for counterfactual continuation)."""

    def __init__(self, m, fam, goal):
        self.m, self.fam = m, fam
        self.g = m.encode_goal(m.ids([[f"FAM_{fam}"] + goal], 48))
        self.h, self.z = m.initial(1)
        self.prev, self.el, self.opt, self.first = ["<bos>"], 0.0, 0, True

    def observe(self, env):
        m = self.m
        obs = m.pad_ids([[m.tok[t] for t in [f"FAM_{self.fam}"] + env.obs_tokens()][:m.cfg.max_obs]], m.cfg.max_obs)
        st = torch.tensor(status_vector(env.contract(), env.task_status())[None], device=m.device)
        self.h, _ = m.img_step(self.h, self.z, m.wm_act_enc(m.ids([self.prev], m.cfg.max_act + 2)))
        e = m.encode_obs(obs, torch.tensor([self.el], device=m.device, dtype=torch.float32), st)
        self.z = m.sample_z(m.post_logits(self.h, e), mode=True)

    def act(self):
        m = self.m
        f = m.feat(self.h, self.z)
        cur = torch.tensor([self.opt], device=m.device)
        logits, beta = m.actor_dists(f, self.g, cur)
        new = self.first or bool(torch.sigmoid(beta)[0] > 0.5)
        opt = int(logits.argmax(-1)) if new else self.opt
        toks, _, _ = m.generate_action(f, self.g, torch.tensor([opt], device=m.device), [_grammar(self.fam)], greedy=True)
        self.opt, self.first = opt, False
        return toks[0]

    def record(self, toks, el):
        self.prev, self.el = toks, el


@torch.no_grad()
def counterfactual_ranking(m, fam: str, n_states: int = 40, k: int = 4, lam: float = 0.0, seed: int = 0,
                           rc: RewardContract | None = None) -> dict:
    m.eval()
    rc = rc or RewardContract()
    rng = random.Random(seed)
    sp, top1, pair, regret, oracle_gain, chance, n_used = [], [], [], [], [], [], 0
    seed_i = 0
    while n_used < n_states and seed_i < n_states * 4:
        env = ENVS[fam]()
        env.reset(2_000_000 + seed_i, "val")
        seed_i += 1
        ag = _Agent(m, fam, env.goal_tokens())
        stop_at = rng.randrange(0, max(1, env.max_steps // 2))
        t = 0
        while not env.done and t < stop_at:
            ag.observe(env)
            a = ag.act()
            res = env.step([x for x in a if x != "END"])
            ag.record(a, res.elapsed_h)
            t += 1
        if env.done:
            continue
        ag.observe(env)
        cands = CANDIDATES[fam](env)
        chance.append(1 / len(cands))
        real_v, pred_v = [], []
        for cand in cands:
            # actual consequence: execute the alternative, then CAM's own greedy policy, in a simulator copy
            e2 = env.snapshot()
            e2.reseed(seed_i)
            a2 = copy.copy(ag)
            c = e2.contract()
            s0 = e2.task_status()
            ret, D, toks = 0.0, 1.0, cand + ["END"]
            for j in range(k):
                if j > 0:
                    a2.observe(e2)
                    toks = a2.act()
                res = e2.step([x for x in toks if x != "END"])
                s1 = e2.task_status()
                tr = rc.transition(c, s0, s1, res.done, bool(res.success), lam)
                ret += D * tr.reward
                D *= tr.discount
                s0 = s1
                a2.record(toks, res.elapsed_h)
                if res.done:
                    break
            real_v.append(ret)
            # world-model prediction of the same intervention followed by the same (greedy) policy
            h, z = ag.h.clone(), ag.z.clone()
            pv, Dp, alive, cur, first = 0.0, 1.0, 1.0, ag.opt, ag.first
            toks = cand + ["END"]
            for j in range(k):
                if j > 0:
                    f = m.feat(h, z)
                    logits, beta = m.actor_dists(f, ag.g, torch.tensor([cur], device=m.device))
                    new = first or bool(torch.sigmoid(beta)[0] > 0.5)
                    cur = int(logits.argmax(-1)) if new else cur
                    toks = m.generate_action(f, ag.g, torch.tensor([cur], device=m.device), [_grammar(fam)], greedy=True)[0][0]
                first = False
                a_e = m.wm_act_enc(m.ids([toks], m.cfg.max_act + 2))
                h, pr = m.img_step(h, z, a_e)
                z = m.sample_z(pr, mode=True)
                H = m.heads(m.feat(h, z), ag.g)
                pv += Dp * alive * (float(H["rew"]) - lam * float(torch.sigmoid(H["viol"])))
                Dp *= float(0.99 ** torch.expm1(H["dt"].clamp(-3, 5)).clamp(min=0))
                alive *= float(torch.sigmoid(H["cont"]))
            pred_v.append(pv)
        n_used += 1
        r, p = np.array(real_v), np.array(pred_v)
        s = _spearman(p, r)
        if not np.isnan(s):
            sp.append(s)
        top1.append(float(np.argmax(p) == np.argmax(r)))
        pr_ = [float((p[a] > p[b]) == (r[a] > r[b])) for a in range(len(r)) for b in range(a + 1, len(r)) if r[a] != r[b]]
        pair.extend(pr_)
        regret.append(float(r.max() - r[int(np.argmax(p))]))
        oracle_gain.append(float(r.max() - r.mean()))
    return {"family": fam, "states": n_used, "k": k, "spearman": float(np.mean(sp)) if sp else float("nan"),
            "top1_agreement": float(np.mean(top1)) if top1 else float("nan"),
            "pairwise_accuracy": float(np.mean(pair)) if pair else float("nan"),
            "regret_of_model_choice": float(np.mean(regret)) if regret else float("nan"),
            "oracle_best_minus_mean (simulator-oracle reference)": float(np.mean(oracle_gain)) if oracle_gain else float("nan"),
            "chance_top1": float(np.mean(chance)) if chance else float("nan")}


def corrupted(m, sigma: float = 0.5, seed: int = 0):
    """A deliberately miscalibrated copy: noise on the prior transition and the reward/violation heads."""
    c = copy.deepcopy(m)
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for mod in (c.prior_net, c.h_rew, c.h_viol, c.h_cont):
            for p in mod.parameters():
                p.add_(torch.randn(p.shape, generator=g).to(p.device) * sigma * (p.std() + 1e-3))
    return c
