"""Continual learning in the weights: the update rule of a learner that lives through many tasks.  [candidate-editable]

`ContinualPPO` is PPO with replay (`algo/ppo.py`) plus what a lifetime needs. Every piece is a weight update;
nothing here writes text, retrieves episodes at decision time or edits prompts.

    world model     a predictive model of consequences (`cam/worldmodel.py`) is trained ONLINE, on every batch of
                    new experience and on replayed transitions; a share of its gradient (`wm.repr`) shapes the
                    controller's own state, so representation learning continues for life
    observation     `observe(demos)`: supervised updates on verified trajectories the learner was shown, with the
                    same optimiser and state as everything else
    rehearsal       verified-successful episodes are kept per task arm (a bounded reservoir each), across phases;
                    replay draws arms uniformly, so old skills keep receiving gradient while new ones are learned
    consolidation   at the end of a phase the weights are anchored: a quadratic penalty pulls later updates back
                    towards them, uniformly (`l2sp`), by estimated importance (`ewc`, diagonal empirical Fisher,
                    accumulated over phases) or only for the most important weights (`anchor.top_frac`)
    plasticity      `l2init` (a pull towards the initial weights), shrink-and-perturb at phase boundaries, and
                    per-group learning-rate multipliers (`group_lr`) that decide which parameter groups stay plastic
    backbone        if the learner has a trainable language backbone its parameters are one more group

The functions at module level (`param_group`, `replay_arm_weights`, `anchor_penalty`, `fisher_update`) are the rule's
moving parts; the Meta-Researcher can replace any of them (docs/CONTINUAL.md, "search over update rules").
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from ..cam.batch import evaluate
from ..cam.worldmodel import WorldModel, joint_loss
from . import rule as R
from .base import register
from .ppo import PPO, flat_params, grad_norm, make_minibatches
from .sil import PPOSIL

CONTINUAL_DEFAULTS = {
    "wm": {"enabled": True, "coef": 1.0, "repr": 0.1, "replay_steps": 4, "replay_batch": 8, "buffer": 400, "lr_mult": 4.0},
    "rehearsal": {"keep": True, "per_arm": 120},
    "consolidate_steps": 0,
    "anchor": {"kind": "none", "coef": 0.0, "decay": 1.0, "fisher_episodes": 64, "top_frac": 1.0},
    "l2init": 0.0,
    "shrink_perturb": {"shrink": 1.0, "sigma": 0.0},
    "group_lr": {},
    "observe": {"steps": 120, "lr": 3e-4, "rehearse": True, "rehearse_frac": 0.5, "value_coef": 0.5, "warmup": 10},
    "backbone_lr": 1e-5,
}

GROUPS = ("input", "encoder", "recurrent", "policy", "critic", "wm", "backbone")


def param_group(name: str) -> str:
    """Which group a controller parameter belongs to (groups can have different learning rates, or be frozen)."""
    head = name.split(".")[0]
    if head in ("tok", "tok_proj", "ek_proj", "num_proj", "typ_emb", "item_norm", "obs_proj", "aux_proj", "cls"):
        return "input"
    if head in ("encoder", "enc_norm"):
        return "encoder"
    if head in ("fast", "slow", "mix"):
        return "recurrent"
    if head in ("value", "success"):
        return "critic"
    return "policy"


def replay_arm_weights(arms: list, current: set) -> np.ndarray:
    """Probability of drawing each arm of the buffer for one replayed episode (uniform over arms: old and new
    skills receive the same share of replayed gradient)."""
    return np.ones(len(arms)) / max(1, len(arms))


def anchor_penalty(params: list, anchors: list, weights: list | None) -> torch.Tensor:
    """sum_i w_i (theta_i - theta*_i)^2, averaged over parameters (so the coefficient does not depend on model size)."""
    tot, n = 0.0, 0
    for i, (p, a) in enumerate(zip(params, anchors)):
        d2 = (p - a) ** 2
        tot = tot + ((d2 * weights[i]).sum() if weights is not None else d2.sum())
        n += p.numel()
    return tot / max(1, n)


def fisher_update(old: list | None, new: list, decay: float) -> list:
    """Combine the importance estimated at the end of this phase with what earlier phases estimated."""
    mean = sum(float(f.sum()) for f in new) / max(1, sum(f.numel() for f in new))
    new = [f / max(mean, 1e-12) for f in new]                 # mean importance 1: comparable to a uniform anchor
    if old is None:
        return new
    return [decay * o + f for o, f in zip(old, new)]


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@register("continual")
class ContinualPPO(PPOSIL):
    uses_replay = True

    def __init__(self, model, cfg=None, device="cpu"):
        cfg = dict(cfg or {})
        self.cc = _merge(CONTINUAL_DEFAULTS, {k: cfg.pop(k) for k in list(cfg) if k in CONTINUAL_DEFAULTS})
        cfg.setdefault("sil_mode", "filtered")
        cfg.setdefault("replay_episodes", 1_000_000)      # the reservoir bounds the buffer per arm, not in total
        super().__init__(model, cfg, device)
        self.wm = WorldModel(model.cfg.d).to(device) if self.cc["wm"]["enabled"] else None
        self.backbone = None
        self.experience: list[dict] = []       # recent episodes of any outcome (world-model replay)
        self.arms_now: set = set()             # arms of the current phase
        self.phase = -1
        self.anchors: list | None = None
        self.fisher: list | None = None
        self.theta_init = [p.detach().clone() for p in model.parameters()] if self.cc["l2init"] > 0 else None
        self.n_seen = 0
        self._build_opt()

    # ------------------------------------------------------------------ optimiser with parameter groups
    def _build_opt(self, state: dict | None = None) -> None:
        by: dict[str, list] = {g: [] for g in GROUPS}
        for n, p in self.model.named_parameters():
            by[param_group(n)].append(p)
        if self.wm is not None:
            by["wm"] = list(self.wm.parameters())
        if self.backbone is not None and self.backbone.trainable:
            by["backbone"] = [p for p in self.backbone.parameters() if p.requires_grad]
        mult = dict({"wm": float(self.cc["wm"].get("lr_mult", 1.0)),
                     "backbone": float(self.cc["backbone_lr"]) / float(self.c["lr"])}, **self.cc["group_lr"])
        groups = []
        for g in GROUPS:
            m = float(mult.get(g, 1.0))
            for p in by[g]:
                p.requires_grad_(m > 0)                       # a group with multiplier 0 is frozen
            if by[g] and m > 0:
                groups.append({"params": by[g], "lr": self.c["lr"] * m, "mult": m, "name": g})
        self.opt = torch.optim.AdamW(groups, lr=self.c["lr"], weight_decay=self.c["weight_decay"])
        self.extra_params = (list(self.wm.parameters()) if self.wm is not None else []) + by["backbone"]
        if state is not None:
            self.opt.load_state_dict(state)

    def attach_backbone(self, backbone) -> None:
        """Make the language backbone's trainable parameters part of this learner (its own optimiser group)."""
        self.backbone = backbone
        self._build_opt()

    def set_lr(self, base: float | None = None) -> float:
        w = int(self.c["lr_warmup_steps"])
        lr = (base if base is not None else self.c["lr"]) * self.lr_scale * (min(1.0, (self.n_policy_steps + 1) / w) if w > 0 else 1.0)
        for g in self.opt.param_groups:
            g["lr"] = lr * g.get("mult", 1.0)
        return lr

    def group_sizes(self) -> dict:
        return {g["name"]: int(sum(p.numel() for p in g["params"])) for g in self.opt.param_groups}

    # ------------------------------------------------------------------ extra loss terms of every step
    def penalty(self) -> tuple:
        info, tot = {}, 0.0
        a = self.cc["anchor"]
        ps = [p for p in self.model.parameters()]
        if self.anchors is not None and a["kind"] != "none" and a["coef"] > 0:
            pen = anchor_penalty(ps, self.anchors, self.fisher if a["kind"] == "ewc" else None)
            tot = tot + a["coef"] * pen
            info["anchor_pen"] = float(pen.detach())
        if self.theta_init is not None:
            pen = anchor_penalty(ps, self.theta_init, None)
            tot = tot + self.cc["l2init"] * pen
            info["l2init_pen"] = float(pen.detach())
        return tot, info

    def extra_loss(self, out, b):
        tot, info = self.penalty()
        if self.wm is not None:
            wl, wi = joint_loss(self.wm, out, b.t, self.rule.gamma, float(self.cc["wm"]["repr"]))
            tot = tot + float(self.cc["wm"]["coef"]) * wl
            info.update(wi, wm_loss=float(wl.detach()))
        return tot, info

    # ------------------------------------------------------------------ buffers
    def _reservoir(self, episodes: list[dict]) -> None:
        cap = int(self.cc["rehearsal"]["per_arm"])
        by: dict = {}
        for e in self.buffer:
            by.setdefault(tuple(e["summary"].get("arm", ())), []).append(e)
        for e in episodes:
            lst = by.setdefault(tuple(e["summary"].get("arm", ())), [])
            n = e["summary"].setdefault("_n_arm", len(lst))
            if len(lst) < cap:
                lst.append(e)
            else:                                   # reservoir: every success of an arm equally likely to be kept
                j = int(self.rng.integers(0, max(n, cap) + 1))
                if j < cap:
                    lst[j] = e
        self.buffer = [e for lst in by.values() for e in lst]

    def _sample_replay(self, n: int) -> list[dict]:
        by: dict = {}
        for e in self.buffer:
            by.setdefault(tuple(e["summary"].get("arm", ())), []).append(e)
        arms = sorted(by)
        p = replay_arm_weights(arms, self.arms_now)
        p = np.asarray(p, np.float64) / float(np.sum(p))
        out = []
        for _ in range(min(n, len(self.buffer))):
            lst = by[arms[int(self.rng.choice(len(arms), p=p))]]
            out.append(lst[int(self.rng.integers(0, len(lst)))])
        return out

    def update(self, episodes, store, demos=None):
        """Exploratory episodes (high temperature, random) are off-policy: they feed the world model and the
        reservoir of successes, never the policy-gradient term."""
        own = [e for e in episodes if not e["summary"].get("explore")]
        m = self._update_on_policy(own, store, demos) if own else {}
        m.update(self.replay(episodes, store))
        return m

    def replay(self, episodes, store) -> dict:
        """As in PPO, with a per-arm reservoir instead of a first-in-first-out buffer (an old skill's successes are
        never pushed out by a new skill's), followed by world-model replay on stored transitions of any outcome."""
        self._reservoir([e for e in episodes if e["summary"].get("success")])
        m = super().replay([], store)                 # the parent's own first-in-first-out append gets nothing
        m.update(self.wm_replay(episodes, store))
        return m

    def wm_replay(self, episodes, store) -> dict:
        wc = self.cc["wm"]
        self.experience = (self.experience + list(episodes))[-int(wc["buffer"]):]
        if self.wm is None or int(wc["replay_steps"]) <= 0 or len(self.experience) < 2:
            return {}
        self.model.train()
        losses = []
        ex = [e for e in self.experience if e["summary"].get("explore")]
        for _ in range(int(wc["replay_steps"])):
            n = min(int(wc["replay_batch"]), len(self.experience))
            n_ex = min(len(ex), int(round(n * float(wc.get("explore_share", 0.5)))))    # consequences of unusual actions
            part = [ex[int(i)] for i in self.rng.integers(0, max(1, len(ex)), size=n_ex)] if n_ex else []
            part += [self.experience[int(i)] for i in self.rng.integers(0, len(self.experience), size=n - n_ex)]
            b = make_minibatches(part, store, self.device, len(part), self.rng)[0]
            out = evaluate(self.model, b, store)
            wl, _ = joint_loss(self.wm, out, b.t, self.rule.gamma, float(wc["repr"]))
            pen, _ = self.penalty()
            loss = float(wc["coef"]) * wl + pen
            if not torch.isfinite(loss):
                continue
            self.set_lr()
            self.opt.zero_grad(set_to_none=True)
            self.backward(loss)
            self.clip()
            self.opt.step()
            self.n_updates += 1
            losses.append(float(wl.detach()))
        self.model.eval()
        return {"wm_replay_loss": float(np.mean(losses)) if losses else 0.0, "wm_replay_steps": len(losses),
                "wm_buffer": len(self.experience)}

    # ------------------------------------------------------------------ observation (supervised updates)
    def observe(self, demos: list[dict], store, steps: int | None = None) -> list[dict]:
        """Learn from verified trajectories the learner was shown: `steps` supervised optimiser steps. With
        rehearsal, a share (`observe.rehearse_frac`) of every minibatch is drawn from the reservoir of EARLIER arms,
        so imitating the new demonstrations does not overwrite what was learned before."""
        oc = self.cc["observe"]
        steps = int(oc["steps"] if steps is None else steps)
        model, dev = self.model, self.device
        if not demos or steps <= 0:
            return []
        new_arms = {tuple(d["summary"].get("arm", ())) for d in demos}
        old = [e for e in self.buffer if tuple(e["summary"].get("arm", ())) not in new_arms]
        self._reservoir([dict(d, summary=dict(d["summary"], demo=True)) for d in demos])
        mc = R.RuleConfig(**dict(self.rule.to_dict(), lam=1.0))
        nb = int(self.c["minibatch_eps"])
        n_old = int(round(nb * float(oc["rehearse_frac"]))) if (oc["rehearse"] and self.cc["rehearsal"]["keep"] and old) else 0
        by_old: dict = {}
        for e in old:
            by_old.setdefault(tuple(e["summary"].get("arm", ())), []).append(e)
        old_arms = sorted(by_old)
        hist, every = [], max(1, steps // 20)
        model.train()
        before = flat_params(model)
        acc: dict = {}
        for st in range(steps):
            part = [demos[int(i)] for i in self.rng.integers(0, len(demos), size=nb - n_old)]
            for _ in range(n_old):                       # earlier arms equally likely, whatever their buffer size
                lst = by_old[old_arms[int(self.rng.integers(0, len(old_arms)))]]
                part.append(lst[int(self.rng.integers(0, len(lst)))])
            b = make_minibatches(part, store, dev, len(part), self.rng)[0]
            t = b.t
            out = evaluate(model, b, store)
            w = t["valid"].float()
            loss_pi = -(out["logp"] * w).sum() / w.sum().clamp(min=1)
            _, ret = R.advantages(t["reward"], torch.zeros_like(t["reward"]), t["done"], t["trunc"], t["valid"], mc)
            vl = R.value_loss(out["values"], ret, t["value_old"], t["valid"], self.rule)
            al, _ = R.aux_loss(out, b, self.rule)
            xl, xi = self.extra_loss(out, b)
            loss = loss_pi + float(oc["value_coef"]) * vl + self.rule.succ_coef * al + xl
            if not torch.isfinite(loss):
                continue
            for g in self.opt.param_groups:
                g["lr"] = float(oc["lr"]) * g.get("mult", 1.0) * min(1.0, (st + 1) / max(1, int(oc["warmup"])))
            self.opt.zero_grad(set_to_none=True)
            self.backward(loss)
            gn = grad_norm(model)
            self.clip()
            self.opt.step()
            self.n_updates += 1
            for k, v in dict(xi, bc_nll=float(loss_pi.detach()), grad_norm=gn).items():
                acc.setdefault(k, []).append(v)
            if (st + 1) % every == 0 or st == steps - 1:
                now = flat_params(model)
                hist.append(dict({k: float(np.mean(v)) for k, v in acc.items()}, phase="observe", step=st + 1,
                                 demo_episodes=len(demos), rehearsed_per_batch=n_old,
                                 param_delta=float((now - before).norm())))
                acc = {}
        model.eval()
        return hist

    # ------------------------------------------------------------------ phase boundaries
    def begin_phase(self, index: int, arms: list) -> dict:
        self.phase = index
        self.arms_now = {tuple(a) for a in arms}
        info: dict = {"buffer_before": len(self.buffer)}
        if not self.cc["rehearsal"]["keep"]:
            self.buffer = [e for e in self.buffer if tuple(e["summary"].get("arm", ())) in self.arms_now]
        sp = self.cc["shrink_perturb"]
        if index > 0 and (sp["shrink"] < 1.0 or sp["sigma"] > 0.0):
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.requires_grad and p.dim() > 1:
                        p.mul_(float(sp["shrink"])).add_(torch.randn_like(p) * float(sp["sigma"]) * p.std())
            info["shrink_perturb"] = dict(sp)
        info["buffer_after"] = len(self.buffer)
        return info

    def end_phase(self, store) -> dict:
        """Consolidate: remember where the weights are and (for `ewc`) how much each one matters for what the
        learner can do now, estimated from its own successful episodes of every arm it holds."""
        a = self.cc["anchor"]
        info: dict = {}
        n_cons = int(self.cc["consolidate_steps"])
        if n_cons > 0 and self.buffer:
            # replay-based consolidation: optimiser steps on the reservoir alone, every arm equally likely, with no
            # new experience - what was learned in this phase is integrated with everything learned before it
            keep_steps, keep_calls = self.c["sil_steps"], self.n_calls
            self.c["sil_steps"], self.n_calls = n_cons, max(self.n_calls, int(self.c["critic_warmup"]) + 1)
            before = flat_params(self.model)
            try:
                m = PPO.replay(self, [], store)
            finally:
                self.c["sil_steps"], self.n_calls = keep_steps, keep_calls
            info["consolidation"] = {"steps": m.get("replay_steps", 0), "loss": m.get("replay_loss"), "arms": m.get("replay_arms"),
                                     "weights_moved": float((flat_params(self.model) - before).norm())}
        if a["kind"] == "none" or a["coef"] <= 0:
            return info
        ps = list(self.model.parameters())
        info["anchor"] = a["kind"]
        if a["kind"] == "ewc" and self.buffer:
            f = [torch.zeros_like(p) for p in ps]
            eps = self._sample_replay(int(a["fisher_episodes"]))
            self.model.train()
            nb = 0
            for b in make_minibatches(eps, store, self.device, int(self.c["minibatch_eps"]), self.rng):
                out = evaluate(self.model, b, store)
                w = b.t["valid"].float()
                self.model.zero_grad(set_to_none=True)
                (-(out["logp"] * w).sum() / w.sum().clamp(min=1)).backward()
                for fi, p in zip(f, ps):
                    if p.grad is not None:
                        fi += p.grad.detach() ** 2
                nb += 1
            self.model.zero_grad(set_to_none=True)
            self.model.eval()
            self.fisher = fisher_update(self.fisher, [fi / max(1, nb) for fi in f], float(a["decay"]))
            if a["top_frac"] < 1.0:                # selective consolidation: protect only the most important weights
                allf = torch.cat([x.reshape(-1) for x in self.fisher])
                k = max(1, int(math.ceil(a["top_frac"] * allf.numel())))
                thr = torch.topk(allf, k, sorted=False).values.min()
                self.fisher = [x * (x >= thr) for x in self.fisher]
            info["fisher_mean"] = float(torch.cat([x.reshape(-1) for x in self.fisher]).mean())
            info["fisher_nonzero"] = float(torch.cat([(x > 0).float().reshape(-1) for x in self.fisher]).mean())
        self.anchors = [p.detach().clone() for p in ps]
        return info

    # ------------------------------------------------------------------ persistence
    def buffers_state(self) -> dict:
        return {"buffer": list(self.buffer), "experience": list(self.experience)}

    def load_buffers_state(self, s: dict) -> None:
        self.buffer = list(s.get("buffer", []))
        self.experience = list(s.get("experience", []))

    def state_dict(self):
        s = super().state_dict()
        s.update(wm=self.wm.state_dict() if self.wm is not None else None, phase=self.phase,
                 arms_now=sorted(self.arms_now), anchors=self.anchors, fisher=self.fisher, theta_init=self.theta_init,
                 cc=self.cc, n_seen=self.n_seen)
        return s

    def load_state_dict(self, s):
        if self.wm is not None and s.get("wm") is not None:
            self.wm.load_state_dict(s["wm"])
        try:
            super().load_state_dict(s)
        except ValueError:                           # optimiser groups differ (e.g. another backbone mode): fresh optimiser
            self._build_opt()
        self.phase = s.get("phase", -1)
        self.arms_now = {tuple(a) for a in s.get("arms_now", [])}
        dev = self.device
        self.anchors = [a.to(dev) for a in s["anchors"]] if s.get("anchors") is not None else None
        self.fisher = [a.to(dev) for a in s["fisher"]] if s.get("fisher") is not None else None
        if s.get("theta_init") is not None and self.theta_init is not None:
            self.theta_init = [a.to(dev) for a in s["theta_init"]]
        self.n_seen = s.get("n_seen", 0)
