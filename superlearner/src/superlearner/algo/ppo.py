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
            "weight_decay": 0.0, "boot_p": 0.8, "bc_coef": 0.0, "seed": 0}


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
        self.n_updates = 0
        self.n_steps = 0

    # hooks a subclass (or a candidate) can override without touching the loop
    def advantages(self, b):
        t = b.t
        return R.advantages(t["reward"], t["value_old"], t["done"], t["trunc"], t["valid"], self.rule)

    def loss(self, out, b, adv, ret, boot):
        return R.total_loss(out, b, adv, ret, self.rule, boot)

    def update(self, episodes, store, demos=None):
        model, c, dev = self.model, self.c, self.device
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
        infos, gns, stop, nonfinite = [], [], False, 0
        demo_mbs = make_minibatches(demos, store, dev, c["minibatch_eps"], self.rng) if (demos and c["bc_coef"] > 0) else []
        for ep in range(c["epochs"]):
            for i in self.rng.permutation(len(prep)):
                b, adv, ret, boot = prep[i]
                out = evaluate(model, b)
                loss, info = self.loss(out, b, adv, ret, boot)
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
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gns.append(grad_norm(model))
                torch.nn.utils.clip_grad_norm_(model.parameters(), c["max_grad_norm"])
                self.opt.step()
                self.n_updates += 1
                info["loss"] = float(loss.detach())
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
        m.update(grad_norm=float(np.mean(gns)) if gns else 0.0, explained_variance=ev, decisions=steps,
                 grad_steps=len(infos), early_stop=bool(stop), nonfinite=nonfinite,
                 param_delta=float((after - before).norm()), param_norm=float(after.norm()))
        return m

    def state_dict(self):
        return {"opt": self.opt.state_dict(), "n_updates": self.n_updates, "n_steps": self.n_steps,
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, s):
        self.opt.load_state_dict(s["opt"])
        self.n_updates, self.n_steps = s.get("n_updates", 0), s.get("n_steps", 0)
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
