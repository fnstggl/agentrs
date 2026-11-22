"""Recurrent PPO with GAE over whole episodes, and behaviour cloning for warm starts.  [candidate-editable]

`PPO.update(episodes, store)`:
  1. collate the iteration's episodes into minibatches of whole episodes;
  2. compute advantages and returns once from the values recorded while acting (`rule.advantages`);
  3. for several epochs, recompute the policy's likelihood of the recorded choices by unrolling the recurrent
     controller from each episode's start (`cam.batch.evaluate`), and take AdamW steps on `rule.total_loss` with
     gradient-norm clipping;
  4. stop early when the approximate KL divergence from the acting policy exceeds a limit;
  5. refuse non-finite losses and restore the weights if an update produced non-finite parameters.

It returns the loss terms, the gradient norm, the KL, the explained variance of the critic, and how far the
weights moved (L2 norm of the parameter change), which is the direct evidence that learning changed the network.

`bc_update` is supervised imitation of demonstrated choices (negative log-likelihood of the demonstrated head
and arguments), plus the critic regressed on the demonstrations' discounted returns.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..cam.batch import collate, evaluate
from . import rule as R
from .base import Algorithm, register

DEFAULTS = {"lr": 1e-4, "epochs": 3, "minibatch_eps": 32, "max_grad_norm": 1.0, "target_kl": 0.05,
            "weight_decay": 0.0, "boot_p": 0.8, "bc_coef": 0.0, "seed": 0, "critic_warmup": 0, "detach_critic": False,
            "sil_steps": None, "sil_coef": None, "sil_batch": None, "sil_mode": "rule", "replay_episodes": 1500,
            "lr_warmup_steps": 0, "kl_rollback": True}


def flat_params(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def grad_norm(model) -> float:
    g = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    return float(torch.cat(g).norm()) if g else 0.0


def explained_variance(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    p, y = pred[valid], target[valid]
    vy = y.var()
    return float(1.0 - (y - p).var() / vy) if vy > 1e-8 else 0.0


def make_minibatches(episodes: list[dict], store, device, size: int, rng: np.random.Generator) -> list:
    order = rng.permutation(len(episodes))
    out = []
    for i in range(0, len(order), size):
        chunk = [episodes[j] for j in order[i:i + size]]
        out.append(collate([e["decisions"] for e in chunk], store, device,
                           [bool(e["summary"].get("success")) for e in chunk]))
    return out


@register("ppo")
class PPO(Algorithm):
    def __init__(self, model, cfg=None, device="cpu"):
        super().__init__(model, cfg, device)
        c = dict(DEFAULTS, **self.cfg)
        self.c = c
        self.rule = R.RuleConfig(**{k: c[k] for k in R.RuleConfig().to_dict() if k in c})
        self.opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        self.rng = np.random.default_rng(c["seed"])
        model.detach_critic = bool(c["detach_critic"])
        self.n_updates = 0
        self.n_steps = 0
        self.n_calls = 0
        self.buffer: list[dict] = []          # the agent's own verified-successful episodes (used only if the rule replays)
        self.n_policy_steps = 0               # optimiser steps that could move the policy (critic warm-up excluded)
        self.lr_scale = 1.0                   # adapted from the measured KL (halved on a rollback, slowly restored)
        self.rollbacks = 0
        self.extra_params: list = []          # parameters optimised besides the controller's (world model, backbone)

    # ------------------------------------------------------------------ hooks for rules that learn more than a policy
    def trainable_params(self) -> list:
        """Everything an optimiser step can change (and a rollback must restore)."""
        return [p for p in self.model.parameters() if p.requires_grad] + [p for p in self.extra_params if p.requires_grad]

    def extra_loss(self, out, b):
        """Further loss terms computed from the same forward pass (none in the baseline rule)."""
        return 0.0, {}

    def backward(self, loss) -> None:
        loss.backward()
        live = getattr(self.model, "live", None)
        if live is not None:
            live.flush()                      # table gradient -> language-backbone parameters (cam/backbone.py)

    def clip(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.c["max_grad_norm"])
        extra = [p for p in self.extra_params if p.grad is not None]
        if extra:
            torch.nn.utils.clip_grad_norm_(extra, float(self.c.get("extra_max_grad_norm", self.c["max_grad_norm"])))

    # hooks a subclass (or a candidate) can override without touching the loop
    def advantages(self, b):
        t = b.t
        return R.advantages(t["reward"], t["value_old"], t["done"], t["trunc"], t["valid"], self.rule)

    def loss(self, out, b, adv, ret, boot):
        if self.n_calls <= self.c["critic_warmup"]:
            # critic warm-up: the first updates fit the value ensemble and the success predictor to the acting
            # policy's returns and leave the policy alone, so that the first policy updates use real advantages
            vl = R.value_loss(out["values"], ret, b.t["value_old"], b.t["valid"], self.rule, boot)
            al, _ = R.aux_loss(out, b, self.rule)
            with torch.no_grad():
                kl = float(((b.t["logp_old"] - out["logp"]) * b.t["valid"]).sum() / b.t["valid"].sum().clamp(min=1))
            return self.rule.vf_coef * vl + self.rule.succ_coef * al, {"kl": kl, "clip_frac": 0.0, "pol_loss": 0.0,
                                                                         "v_loss": float(vl.detach()), "entropy": 0.0,
                                                                         "aux_loss": float(al.detach())}
        return R.total_loss(out, b, adv, ret, self.rule, boot)

    # ------------------------------------------------------------------ replay of own successes
    def replay_settings(self) -> tuple[int, float, int]:
        c = self.c
        steps = c["sil_steps"] if c["sil_steps"] is not None else getattr(R, "REPLAY_STEPS", 0)
        coef = c["sil_coef"] if c["sil_coef"] is not None else getattr(R, "REPLAY_COEF", 1.0)
        batch = c["sil_batch"] if c["sil_batch"] is not None else getattr(R, "REPLAY_BATCH", c["minibatch_eps"])
        return int(steps), float(coef), int(batch)

    def _sample_replay(self, n: int) -> list[dict]:
        by: dict = {}
        for e in self.buffer:
            by.setdefault(tuple(e["summary"].get("arm", ())), []).append(e)
        arms = sorted(by)
        out = []
        for _ in range(min(n, len(self.buffer))):
            lst = by[arms[int(self.rng.integers(0, len(arms)))]]          # every task arm equally likely
            out.append(lst[int(self.rng.integers(0, len(lst)))])
        return out

    def replay(self, episodes, store) -> dict:
        """Extra gradient steps on episodes from the buffer of the agent's own verified successes. How many, and
        with which loss, is part of the learning rule (`rule.REPLAY_*`, `rule.replay_loss`)."""
        steps, coef, nb = self.replay_settings()
        self.buffer += [e for e in episodes if e["summary"].get("success")]
        self.buffer = self.buffer[-int(self.c["replay_episodes"]):]
        if steps <= 0 or not self.buffer or self.n_calls <= self.c["critic_warmup"]:
            return {"replay_steps": 0, "replay_episodes": len(self.buffer)}
        model, dev = self.model, self.device
        model.train()
        mc = R.RuleConfig(**dict(self.rule.to_dict(), lam=1.0))
        losses = []
        for _ in range(steps):
            part = self._sample_replay(nb)
            b = make_minibatches(part, store, dev, len(part), self.rng)[0]
            t = b.t
            out = evaluate(model, b)
            _, ret = R.advantages(t["reward"], torch.zeros_like(t["reward"]), t["done"], t["trunc"], t["valid"], mc)
            if self.c["sil_mode"] == "advantage":
                w = t["valid"].float()
                v = out["values"].mean(-1)
                gap = (ret - v.detach()).clamp(min=0)
                loss = (-(out["logp"] * gap) * w).sum() / w.sum().clamp(min=1) \
                    + 0.5 * ((((ret - v).clamp(min=0)) ** 2) * w).sum() / w.sum().clamp(min=1)
            else:
                loss = R.replay_loss(out, b, ret, self.rule)
            xl, _ = self.extra_loss(out, b)
            if not torch.isfinite(loss):
                continue
            self.set_lr()
            self.opt.zero_grad(set_to_none=True)
            self.backward(coef * loss + xl)
            self.clip()
            self.opt.step()
            self.n_updates += 1
            self.n_policy_steps += 1
            losses.append(float(loss.detach()))
        model.eval()
        return {"replay_loss": float(np.mean(losses)) if losses else 0.0, "replay_steps": len(losses),
                "replay_episodes": len(self.buffer), "replay_arms": len({tuple(e["summary"].get("arm", ())) for e in self.buffer})}

    def set_lr(self) -> float:
        """Learning rate of the next optimiser step: linear warm-up over the first `lr_warmup_steps` steps (a fresh
        Adam state makes the very first steps move every weight by the full learning rate, whatever the gradient;
        after a warm start that alone can wreck the policy), times the KL-adapted scale."""
        w = int(self.c["lr_warmup_steps"])
        lr = self.c["lr"] * self.lr_scale * (min(1.0, (self.n_policy_steps + 1) / w) if w > 0 else 1.0)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def update(self, episodes, store, demos=None):
        m = self._update_on_policy(episodes, store, demos)
        m.update(self.replay(episodes, store))
        return m

    def _update_on_policy(self, episodes, store, demos=None):
        model, c, dev = self.model, self.c, self.device
        self.n_calls += 1
        model.train()
        model.set_table(store.tensor(dev))
        mbs = make_minibatches(episodes, store, dev, c["minibatch_eps"], self.rng)
        K = model.cfg.n_value
        prep = []
        for b in mbs:
            adv, ret = self.advantages(b)
            boot = (torch.rand(b.B, K, device=dev, generator=None) < c["boot_p"]).float()
            prep.append([b, adv, ret, boot])
        if self.rule.adv_norm:
            allv = torch.cat([p[1][p[0].t["valid"]] for p in prep])
            mu, sd = allv.mean(), allv.std().clamp(min=1e-6) if allv.numel() > 1 else torch.tensor(1.0, device=dev)
            for p in prep:
                p[1] = (p[1] - mu) / sd
        before = flat_params(model)
        infos, gns, stop, nonfinite, rolled = [], [], False, 0, 0
        demo_mbs = make_minibatches(demos, store, dev, c["minibatch_eps"], self.rng) if (demos and c["bc_coef"] > 0) else []
        for ep in range(c["epochs"]):
            for i in self.rng.permutation(len(prep)):
                b, adv, ret, boot = prep[i]
                out = evaluate(model, b)
                loss, info = self.loss(out, b, adv, ret, boot)
                xl, xi = self.extra_loss(out, b)
                loss = loss + xl
                info.update(xi)
                if demo_mbs:
                    db = demo_mbs[int(self.rng.integers(0, len(demo_mbs)))]
                    dout = evaluate(model, db)
                    w = db.t["valid"].float()
                    bc = -(dout["logp"] * w).sum() / w.sum().clamp(min=1)
                    loss = loss + c["bc_coef"] * bc
                    info["bc_loss"] = float(bc.detach())
                if not torch.isfinite(loss):
                    nonfinite += 1
                    continue
                guard = bool(c["target_kl"]) and bool(c["kl_rollback"]) and self.n_calls > c["critic_warmup"]
                tp = self.trainable_params()
                prev = [p_.detach().clone() for p_ in tp] if guard else None
                self.set_lr()
                self.opt.zero_grad(set_to_none=True)
                self.backward(loss)
                gns.append(grad_norm(model))
                self.clip()
                self.opt.step()
                self.n_updates += 1
                if self.n_calls > c["critic_warmup"]:
                    self.n_policy_steps += 1
                info["loss"] = float(loss.detach())
                if guard:
                    # trust region with rollback: measure the divergence this step caused on its own minibatch; a
                    # step that left the region is undone, the learning rate is halved, and the update ends
                    with torch.no_grad():
                        new = evaluate(model, b)["logp"]
                        w_ = b.t["valid"].float()
                        kl_after = float(((b.t["logp_old"] - new) * w_).sum() / w_.sum().clamp(min=1))
                    info["kl"] = kl_after
                    if not np.isfinite(kl_after) or kl_after > 1.5 * c["target_kl"]:
                        with torch.no_grad():
                            for p_, q_ in zip(tp, prev):
                                p_.copy_(q_)
                        self.lr_scale = max(self.lr_scale * 0.5, 1e-3)
                        self.rollbacks += 1
                        rolled += 1
                        stop = True
                        break
                infos.append(info)
                if c["target_kl"] and info["kl"] > 1.5 * c["target_kl"]:
                    stop = True
                    break
            if stop:
                break
        after = flat_params(model)
        if not torch.isfinite(after).all():
            torch.nn.utils.vector_to_parameters(before, model.parameters())
            nonfinite += 1
            after = before
        with torch.no_grad():
            ev = float(np.mean([explained_variance(p[0].t["value_old"], p[2], p[0].t["valid"]) for p in prep]))
        steps = int(sum(int(p[0].t["valid"].sum()) for p in prep))
        self.n_steps += steps
        model.eval()
        m = {k: float(np.mean([i[k] for i in infos if k in i])) for k in (infos[0] if infos else {})}
        if not rolled and infos and c["target_kl"] and max(i["kl"] for i in infos) < 0.5 * c["target_kl"]:
            self.lr_scale = min(1.0, self.lr_scale * 1.25)        # well inside the region: restore the step size
        m.update(grad_norm=float(np.mean(gns)) if gns else 0.0, explained_variance=ev, decisions=steps,
                 grad_steps=len(infos), early_stop=bool(stop), nonfinite=nonfinite, rollbacks=rolled, lr_scale=self.lr_scale,
                 param_delta=float((after - before).norm()), param_norm=float(after.norm()))
        return m

    def state_dict(self):
        return {"opt": self.opt.state_dict(), "n_updates": self.n_updates, "n_steps": self.n_steps, "n_calls": self.n_calls,
                "lr_scale": self.lr_scale, "rollbacks": self.rollbacks, "n_policy_steps": self.n_policy_steps,
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, s):
        self.opt.load_state_dict(s["opt"])
        self.n_updates, self.n_steps = s.get("n_updates", 0), s.get("n_steps", 0)
        self.n_calls = s.get("n_calls", 0)
        self.lr_scale, self.rollbacks = s.get("lr_scale", 1.0), s.get("rollbacks", 0)
        self.n_policy_steps = s.get("n_policy_steps", 0)
        if "rng" in s:
            self.rng.bit_generator.state = s["rng"]


def bc_update(model, opt, demos: list[dict], store, device, minibatch_eps: int = 32, rule: R.RuleConfig | None = None,
              value_coef: float = 0.5, max_grad_norm: float = 1.0, rng: np.random.Generator | None = None) -> dict:
    """One pass of behaviour cloning over demonstration episodes."""
    rule = rule or R.RuleConfig()
    rng = rng or np.random.default_rng(0)
    model.train()
    model.set_table(store.tensor(device))
    before = flat_params(model)
    nll, acc_h, vls, gns = [], [], [], []
    mc = R.RuleConfig(**dict(rule.to_dict(), lam=1.0))
    for b in make_minibatches(demos, store, device, minibatch_eps, rng):
        t = b.t
        out = evaluate(model, b)
        w = t["valid"].float()
        loss_pi = -(out["logp"] * w).sum() / w.sum().clamp(min=1)
        _, ret = R.advantages(t["reward"], torch.zeros_like(t["reward"]), t["done"], t["trunc"], t["valid"], mc)
        vl = R.value_loss(out["values"], ret, t["value_old"], t["valid"], rule)
        al, _ = R.aux_loss(out, b, rule)
        loss = loss_pi + value_coef * vl + rule.succ_coef * al
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gns.append(grad_norm(model))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        nll.append(float(loss_pi.detach()))
        vls.append(float(vl.detach()))
        with torch.no_grad():
            acc_h.append(float(((out["logp_head"] > math.log(0.5)).float() * w).sum() / w.sum().clamp(min=1)))
    model.eval()
    after = flat_params(model)
    return {"bc_nll": float(np.mean(nll)) if nll else float("nan"), "bc_head_p>0.5": float(np.mean(acc_h)) if acc_h else 0.0,
            "bc_v_loss": float(np.mean(vls)) if vls else 0.0, "grad_norm": float(np.mean(gns)) if gns else 0.0,
            "param_delta": float((after - before).norm()), "grad_steps": len(nll)}
