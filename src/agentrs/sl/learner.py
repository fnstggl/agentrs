"""Recurrent PPO over trial segments (the default learner; interchangeable via ``Learner`` subclasses), with the
world model trained as an auxiliary objective on the same real transitions.

The segment is replayed from its stored starting recurrent state under the current weights, with exactly the
reset logic used at collection (``runner.apply_resets``), so action log-probabilities are exact at the first
epoch (checked: ``mismatch`` statistic). Advantages: GAE with the per-step semi-Markov discount; a trial boundary
cuts the bootstrap, an episode boundary inside a trial does not (the meta-RL objective is the return over the
whole trial, which is what makes information gathered in early episodes valuable).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from agentrs.sl.featurize import LABEL_IDS, MAX_CAND, collate
from agentrs.sl.model import SLCore, symlog
from agentrs.sl.runner import apply_resets, uses_memory


def gae(r, v, disc, cut, boot, lam):
    """r, v, disc, cut: (T, N); boot: (N,). cut[t] = trial ended at step t."""
    T = r.shape[0]
    adv = torch.zeros_like(r)
    nxt_v, nxt_a = boot, torch.zeros_like(boot)
    for t in reversed(range(T)):
        keep = (~cut[t]).float()
        delta = r[t] + disc[t] * keep * nxt_v - v[t]
        nxt_a = delta + disc[t] * lam * keep * nxt_a
        adv[t] = nxt_a
        nxt_v = v[t]
    return adv, adv + v


class PPO:
    def __init__(self, model: SLCore, cfg: dict):
        self.m = model
        # anchor (sl-exp-v18): weight of a KL penalty that keeps the policy on shared-head (rehearsed source-world)
        # decisions close to a frozen copy of the source policy (``anchor_model``, set by the trainer). 0 = off.
        self.cfg = {"lr": 3e-4, "epochs": 2, "mb_streams": 64, "clip": 0.2, "vclip": 0.2, "ent": 0.01, "vf": 0.5,
                    "lam": 0.95, "wm": 0.5, "max_grad": 1.0, "target_kl": 0.05, "anchor": 0.0, "anchor_clip": 10.0,
                    **(cfg or {})}
        self.opt = torch.optim.Adam(model.parameters(), lr=self.cfg["lr"], eps=1e-5)
        self.anchor_model = None
        self.anchor_state = None  # the anchor model's own recurrent state per stream, carried across segments

    def state_dict(self):
        return {"opt": self.opt.state_dict(), "cfg": self.cfg}

    def load_state_dict(self, d):
        self.opt.load_state_dict(d["opt"])

    def _replay(self, seg, rows, mode, dev, m=None, start=None, final=None):
        """Recompute the segment for the given stream rows; returns per-step tensors (T, R).
        ``m``: model to replay (default the learner's); ``start``: starting recurrent state of all streams (default the
        segment's, i.e. the collecting model's); ``final``: dict that receives the rows' recurrent state after step T."""
        m = self.m if m is None else m
        steps = seg["steps"]
        T, R = len(steps), len(rows)
        st = {k: (v[rows].clone() if torch.is_tensor(v) else v) for k, v in (seg["start"] if start is None else start).items()}
        st = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in st.items()}
        feats = [steps[t]["feat"][i] for t in range(T) for i in rows]
        gfeats = [steps[t]["gfeat"][i] for t in range(T) for i in rows]
        ob_all = m.encode(collate(feats, dev)).view(T, R, -1)
        ge_all = m.encode_goal(collate(gfeats, dev)).view(T, R, -1)
        use_mem = torch.full((R,), uses_memory(m, mode), dtype=torch.bool, device=dev)
        lps, ents, vals, wm_l = [], [], [], []
        for t in range(T):
            s = steps[t]
            apply_resets(m, st, mode, s["newtrial"][rows].to(dev), s["newep"][rows].to(dev))
            st = m.step_belief(st, ob_all[t], ge_all[t], s["flags"][rows].to(dev), use_mem)
            ctx = m.context(st, ge_all[t])
            z = s["z"][rows].to(dev)
            hix = s["hix"][rows].to(dev) if (m.c.head_bank and "hix" in s) else None  # head set as at collection (v17)
            if m.c.skills > 1:
                hi, term_logit = m.skill_dists(ctx, st["z"], hix)
                term = s["term"][rows].to(dev)
                force = s["newep"][rows].to(dev)
                lsm = torch.log_softmax(hi, -1)
                lp_sk = torch.where(force, torch.zeros_like(term_logit),
                                    torch.where(term, F.logsigmoid(term_logit), F.logsigmoid(-term_logit)))
                lp_sk = lp_sk + torch.where(term, lsm.gather(1, z[:, None]).squeeze(1), torch.zeros_like(lp_sk))
                ent_sk = -(lsm.exp() * lsm).sum(-1) * term.float()
            else:
                lp_sk = torch.zeros(R, device=dev)
                ent_sk = torch.zeros(R, device=dev)
            st["z"] = z
            q = m.policy_query(ctx, z, hix)
            v = m.value_of(ctx, z)
            # slot decisions
            cands = [s["cands"][i] for i in rows]
            chosen = [s["chosen"][i] for i in rows]
            L = max((len(c) for c in chosen), default=0)
            lp_a = torch.zeros(R, device=dev)
            ent_a = torch.zeros(R, device=dev)
            a_emb = torch.zeros(R, m.c.da, device=dev)
            qcur = q
            gate = s["gate"][rows].to(dev) if (m.c.fam_gate_n0 > 0 and "gate" in s) else None  # as at collection (v16)
            for j in range(L):
                rr = [r for r in range(R) if len(chosen[r]) > j]
                ids = np.zeros((len(rr), MAX_CAND, LABEL_IDS), np.int64)
                msk = np.zeros((len(rr), MAX_CAND), bool)
                pick = np.zeros(len(rr), np.int64)
                for k, r in enumerate(rr):
                    c = cands[r][j]
                    ids[k, :len(c)] = c
                    msk[k, :len(c)] = True
                    pick[k] = chosen[r][j]
                rt = torch.as_tensor(rr, device=dev)
                emb = m.labels(torch.as_tensor(ids, device=dev))
                logits = torch.einsum("rcd,rd->rc", emb, qcur[rt]) / (m.c.da ** 0.5)
                if gate is not None:
                    logits = m.gate_logits(logits, torch.as_tensor(msk, device=dev), gate[rt])
                logits = logits.masked_fill(~torch.as_tensor(msk, device=dev), -1e9)
                ls = torch.log_softmax(logits, -1)
                pk = torch.as_tensor(pick, device=dev)
                lp_a = lp_a.index_add(0, rt, ls.gather(1, pk[:, None]).squeeze(1))
                ent_a = ent_a.index_add(0, rt, -(ls.exp() * ls).masked_fill(~torch.as_tensor(msk, device=dev), 0).sum(-1))
                pe = emb[torch.arange(len(rr), device=dev), pk]
                a_emb = a_emb.index_add(0, rt, pe)
                qn = qcur.clone()
                qn[rt] = m.slot_step(pe, qcur[rt], None if hix is None else hix[rt])
                qcur = qn
            # world model on this real transition
            nxt, aux = m.wm(st["h"], st["read"], a_emb)
            r_t = s["r"][rows].to(dev)
            d_t = s["done"][rows].to(dev).float()
            wl = F.mse_loss(aux[:, 0], symlog(r_t), reduction="none") + F.binary_cross_entropy_with_logits(aux[:, 1], d_t, reduction="none")
            if t + 1 < T:
                tgt = ob_all[t + 1].detach()
                valid = (~s["done"][rows].to(dev)).float()
                wl = wl + valid * ((nxt - tgt.unsqueeze(0)) ** 2).mean(-1).mean(0)
            wm_l.append(wl)
            r_feat = torch.stack([symlog(r_t), d_t, torch.ones_like(r_t)], -1)
            st = m.write_memory(st, a_emb, r_feat, use_mem)
            lps.append(lp_a + lp_sk)
            ents.append(ent_a + 0.1 * ent_sk)
            vals.append(v)
        if final is not None:
            final.update({k: (v.detach() if torch.is_tensor(v) else v) for k, v in st.items()})
        return torch.stack(lps), torch.stack(ents), torch.stack(vals), torch.stack(wm_l)

    def _anchor_kl(self, seg, rows, lp, dev, start: dict, advance: bool):
        """KL penalty toward the frozen source policy on the decisions of ``rows`` made with the shared heads (head
        index 0, the rehearsed source worlds when the new world has its own heads). k3 estimator on the sampled
        decisions: exp(d) - d - 1 with d = log pi_src(a) - log pi(a), as in KL-regularised policy optimisation.
        The source model is replayed from its own recurrent state (carried across segments; a trial start clears it,
        as at collection). Returns None when the minibatch has no such decision."""
        steps = seg["steps"]
        if self.anchor_model is None or "hix" not in steps[0]:
            return None
        am = torch.stack([s["hix"][rows] for s in steps]) == 0          # (T, R) shared-head decisions
        sel = am.any(0)
        if not bool(sel.any()):
            return None
        idx = torch.nonzero(sel).flatten()
        rr = rows[idx.cpu()] if torch.is_tensor(rows) else torch.as_tensor(rows)[idx.cpu()]
        fin = {} if advance else None
        with torch.no_grad():
            lp_src = self._replay(seg, rr, seg["mode"], dev, m=self.anchor_model, start=start, final=fin)[0]
        if fin:
            for k, v in fin.items():
                if torch.is_tensor(v):
                    self.anchor_state[k][rr] = v.to(self.anchor_state[k].device)
        d = (lp_src - lp[:, idx]).clamp(-self.cfg["anchor_clip"], self.cfg["anchor_clip"])
        k3 = d.exp() - d - 1.0
        w = am[:, idx].to(dev).float()
        return (k3 * w).sum() / w.sum().clamp(min=1.0)

    def update(self, seg: dict, dev, wm_coef=None) -> dict:
        m, c = self.m, self.cfg
        steps = seg["steps"]
        T, N = len(steps), steps[0]["r"].shape[0]
        r = torch.stack([s["r"] for s in steps])
        v_old = torch.stack([s["v"] for s in steps])
        disc = torch.stack([s["disc"] for s in steps]).float()
        cut = torch.stack([s["trial_end"] for s in steps])
        if c.get("episode_returns"):  # advantages per episode; memory still persists across a trial's episodes
            cut = cut | torch.stack([s["done"] for s in steps])
        lp_old = torch.stack([s["lp"] for s in steps])
        adv, ret = gae(r, v_old, disc, cut, seg["boot"], c["lam"])
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        wm_coef = c["wm"] if wm_coef is None else wm_coef
        stats = {"pg": [], "vf": [], "ent": [], "wm": [], "kl": [], "clipfrac": [], "mismatch": [], "anchor_kl": []}
        a_start = None
        if c["anchor"] > 0 and self.anchor_model is not None:
            if self.anchor_state is None or self.anchor_state["h"].shape[0] != N:
                self.anchor_state = self.anchor_model.init_state(N, "cpu")
            a_start = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.anchor_state.items()}
        first = True
        for ep in range(c["epochs"]):
            perm = np.random.permutation(N)
            for s0 in range(0, N, c["mb_streams"]):
                rows = torch.as_tensor(perm[s0:s0 + c["mb_streams"]])
                lp, ent, val, wml = self._replay(seg, rows, seg["mode"], dev)
                lo, a, R_, vo = (lp_old[:, rows].to(dev), adv_n[:, rows].to(dev), ret[:, rows].to(dev), v_old[:, rows].to(dev))
                ratio = torch.exp(lp - lo)
                if first:  # first minibatch of the first epoch: weights not yet updated, ratios must be exactly 1
                    stats["mismatch"].append(float((lp.detach() - lo).abs().mean()))
                    first = False
                pg = -torch.min(ratio * a, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a).mean()
                vcl = vo + (val - vo).clamp(-c["vclip"], c["vclip"])
                vf = torch.max((val - R_) ** 2, (vcl - R_) ** 2).mean()
                loss = pg + c["vf"] * vf - c["ent"] * ent.mean() + wm_coef * wml.mean()
                if a_start is not None:
                    akl = self._anchor_kl(seg, rows, lp, dev, a_start, advance=(ep == 0))
                    if akl is not None:
                        loss = loss + c["anchor"] * akl
                        stats["anchor_kl"].append(float(akl.detach()))
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), c["max_grad"])
                self.opt.step()
                with torch.no_grad():
                    kl = float(((ratio - 1) - torch.log(ratio)).mean())
                stats["pg"].append(float(pg))
                stats["vf"].append(float(vf))
                stats["ent"].append(float(ent.mean()))
                stats["wm"].append(float(wml.mean()))
                stats["kl"].append(kl)
                stats["clipfrac"].append(float(((ratio - 1).abs() > c["clip"]).float().mean()))
            if np.mean(stats["kl"][-max(1, N // c["mb_streams"]):]) > c["target_kl"]:
                break
        out = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items() if v or k != "anchor_kl"}
        out["adv_std"] = float(adv.std())
        out["ret_mean"] = float(ret.mean())
        return out
