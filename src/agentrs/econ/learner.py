"""Shared generative policy (Qwen3-4B + LoRA), world model and critic (Qwen3-0.6B + LoRA), and their
updates: behaviour cloning on expert demonstrations, token-level PPO with GRPO group baselines on
verified task rewards (+ demonstration replay as a forgetting regulariser), critic regression on
verified returns, and world-model next-outcome prediction.

Exactness: the policy samples full actions token by token at T = 1 with no top-k/top-p, stores
the sampled ids and their log-probs, and the PPO update re-scores exactly those ids under the same
prompt. Samples whose policy_version differs from the current learner version are rejected.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from agentrs.econ.rollout import Episode
from agentrs.models.backbone import BackboneConfig, LMBackbone, LoraSettings
from agentrs.models.critic import Critic, CriticConfig
from agentrs.training.common import GradientTrainer, OptimConfig


def lora(dropout: float = 0.05):
    return LoraSettings(r=16, alpha=32, dropout=dropout, target_modules="all-linear")


def policy_backbone_cfg(name="Qwen/Qwen3-4B", revision=None, max_prompt_tokens=3584) -> BackboneConfig:
    # LoRA dropout is OFF for the policy: PPO ratios must compare the same deterministic function that
    # produced the behaviour log-probs (with dropout the re-scored log-probs were off by ~0.017 nats/token)
    return BackboneConfig(name=name, revision=revision, dtype="bfloat16", device="cuda", lora=lora(0.0),
                          max_prompt_tokens=max_prompt_tokens, head_keep_tokens=1024, gradient_checkpointing=True)


def small_backbone_cfg(name="Qwen/Qwen3-0.6B", revision=None) -> BackboneConfig:
    return BackboneConfig(name=name, revision=revision, dtype="bfloat16", device="cuda", lora=lora(),
                          max_prompt_tokens=1536, head_keep_tokens=256)


# ---------------------------------------------------------------------------- policy
class GenPolicy:
    def __init__(self, cfg: BackboneConfig, max_new_tokens: int = 320, gen_batch: int = 24):
        self.bb = LMBackbone(cfg)
        self.tok = self.bb.tokenizer
        self.max_new_tokens = max_new_tokens
        self.gen_batch = gen_batch
        self.gen = torch.Generator().manual_seed(cfg.seed)

    def chat_text(self, messages, prefill: str) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False) + prefill

    @torch.no_grad()
    def act(self, reqs, greedy: bool = False, temperature: float = 1.0) -> list[dict]:
        """reqs: list of (messages, prefill). Sampling at T=1 (or greedy) with exact ids/log-probs."""
        self.bb.eval()
        prompts = [self.chat_text(m, p) for m, p in reqs]
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))  # similar lengths per batch
        outs: list = [None] * len(prompts)
        for s in range(0, len(order), self.gen_batch):
            idx = order[s:s + self.gen_batch]
            res = self.bb.generate_tokens([prompts[i] for i in idx], max_new_tokens=self.max_new_tokens,
                                          temperature=temperature, stop_text="\n", greedy=greedy, generator=self.gen)
            for i, r in zip(idx, res):
                outs[i] = {"text": self.tok.decode(r["ids"], skip_special_tokens=True).split("\n")[0],
                           "ids": r["ids"], "logprobs": r["logprobs"], "finished": r["finished"], "prompt": prompts[i]}
        return outs

    def token_logps(self, prompts: Sequence[str], ids: Sequence[Sequence[int]]):
        return self.bb.token_logprobs_from_ids(list(prompts), [list(x) for x in ids])


def accumulate_step(trainer: GradientTrainer) -> dict:
    """Clip + step after gradients were accumulated with .backward() (guards like GradientTrainer.step)."""
    grads = [p.grad for p in trainer.params if p.grad is not None]
    if not grads:
        return {"grad_norm": 0.0, "skipped": 1.0}
    gn = torch.nn.utils.clip_grad_norm_(trainer.params, trainer.cfg.grad_clip)
    if not torch.isfinite(gn):
        trainer.optimizer.zero_grad(set_to_none=True)
        return trainer._bad("non-finite gradient")
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.global_step += 1
    trainer.bad_steps = 0
    return {"grad_norm": float(gn), "lr": float(trainer.scheduler.get_last_lr()[0]), "skipped": 0.0}


@dataclass
class BCExample:
    prompt: str        # chat text incl. prefill
    target_ids: list   # action continuation + newline


def bc_examples(policy: GenPolicy, demos: Sequence[Episode]) -> list[BCExample]:
    out = []
    for ep in demos:
        if not ep.success:
            continue  # imitate only verified-successful demonstrations
        for st in ep.steps:
            msgs = getattr(st, "messages", None)
            if msgs is None:
                continue
            prompt = policy.chat_text(msgs, st.prefill)
            tgt = st.action[len(st.prefill):] + "\n"
            out.append(BCExample(prompt, policy.tok(tgt, add_special_tokens=False)["input_ids"]))
    return out


def bc_loss(policy: GenPolicy, batch: Sequence[BCExample]) -> tuple[torch.Tensor, int]:
    lp, mask = policy.token_logps([b.prompt for b in batch], [b.target_ids for b in batch])
    m = mask.float()
    per = -(lp * m).sum(-1) / m.sum(-1).clamp(min=1)
    return per.mean(), int(m.sum())


def train_bc(policy: GenPolicy, trainer: GradientTrainer, examples: list[BCExample], steps: int, batch: int = 8,
             micro: int = 4, seed: int = 0, log=None) -> list[dict]:
    rng = random.Random(seed)
    policy.bb.train()
    hist = []
    for s in range(steps):
        b = rng.sample(examples, min(batch, len(examples)))
        tot = 0.0
        for k in range(0, len(b), micro):
            loss, _ = bc_loss(policy, b[k:k + micro])
            (loss * len(b[k:k + micro]) / len(b)).backward()
            tot += float(loss) * len(b[k:k + micro]) / len(b)
        st = accumulate_step(trainer)
        hist.append({"step": s, "loss": tot, **st})
        if log and (s % 20 == 0 or s == steps - 1):
            log(f"bc step {s + 1}/{steps} loss {tot:.4f} gn {st.get('grad_norm', 0):.3f}")
    policy.bb.eval()
    return hist


@dataclass
class PGSample:
    prompt: str
    ids: list
    old_logps: list
    adv: float
    version: str
    family: str


def group_advantages(eps: Sequence[Episode], invalid_penalty: float = 0.05, adv_clip: float = 3.0) -> dict:
    """GRPO: advantage = (R - mean_group) / (std_group + eps); R = verified success - penalty * invalid actions."""
    groups: dict[str, list[Episode]] = {}
    for e in eps:
        groups.setdefault(e.group or e.episode_id, []).append(e)
    adv = {}
    for g, es in groups.items():
        R = [e.reward - min(0.5, invalid_penalty * e.n_invalid) for e in es]
        mu = sum(R) / len(R)
        sd = math.sqrt(sum((r - mu) ** 2 for r in R) / len(R))
        for e, r in zip(es, R):
            adv[e.episode_id] = 0.0 if sd < 1e-6 else max(-adv_clip, min(adv_clip, (r - mu) / (sd + 1e-4)))
    return adv


def pg_samples(eps: Sequence[Episode], adv: dict) -> list[PGSample]:
    out = []
    for e in eps:
        a = adv.get(e.episode_id, 0.0)
        if a == 0.0:
            continue
        for st in e.steps:
            if st.token_ids:
                out.append(PGSample(st.prompt, st.token_ids, st.behavior_logprobs, a, e.policy_version, e.family))
    return out


def ppo_update(policy: GenPolicy, trainer: GradientTrainer, samples: list[PGSample], version: str,
               demo_examples: Optional[list[BCExample]] = None, bc_coef: float = 0.05, clip: float = 0.2,
               minibatch: int = 16, micro: int = 4, epochs: int = 1, seed: int = 0, log=None) -> dict:
    """Token-level clipped PPO over the given on-policy samples (+ BC replay regulariser)."""
    rng = random.Random(seed)
    stale = [s for s in samples if s.version != version]
    samples = [s for s in samples if s.version == version]
    stats = {"samples": len(samples), "rejected_stale": len(stale), "updates": 0, "clipfrac": 0.0, "mismatch_first": None,
             "kl_approx": 0.0, "pg_loss": 0.0, "bc_loss": 0.0, "grad_norm": 0.0}
    if not samples:
        return stats
    policy.bb.train()
    n_mb = 0
    for ep in range(epochs):
        rng.shuffle(samples)
        for s0 in range(0, len(samples), minibatch):
            mb = samples[s0:s0 + minibatch]
            cf_tot, kl_tot, pg_tot = 0.0, 0.0, 0.0
            for k in range(0, len(mb), micro):
                part = mb[k:k + micro]
                lp, mask = policy.token_logps([p.prompt for p in part], [p.ids for p in part])
                L = lp.shape[1]
                old = torch.zeros_like(lp)
                for i, p in enumerate(part):
                    old[i, :len(p.old_logps)] = torch.tensor(p.old_logps[:L], device=lp.device, dtype=lp.dtype)
                m = mask.float()
                if stats["mismatch_first"] is None:  # before any update: behaviour vs re-scored log-probs
                    stats["mismatch_first"] = float(((lp.detach() - old).abs() * m).sum() / m.sum().clamp(min=1))
                ratio = torch.exp(lp - old)
                A = torch.tensor([p.adv for p in part], device=lp.device, dtype=lp.dtype).unsqueeze(1)
                obj = torch.minimum(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A)
                per = -(obj * m).sum(-1) / m.sum(-1).clamp(min=1)
                loss = per.mean() * len(part) / len(mb)
                loss.backward()
                pg_tot += float(loss)
                with torch.no_grad():
                    cf_tot += float((((ratio - 1).abs() > clip).float() * m).sum() / m.sum().clamp(min=1)) * len(part) / len(mb)
                    kl_tot += float((((ratio - 1) - torch.log(ratio)) * m).sum() / m.sum().clamp(min=1)) * len(part) / len(mb)
            if demo_examples and bc_coef > 0:
                b = rng.sample(demo_examples, min(micro, len(demo_examples)))
                bl, _ = bc_loss(policy, b)
                (bc_coef * bl).backward()
                stats["bc_loss"] += float(bl)
            st = accumulate_step(trainer)
            n_mb += 1
            stats["updates"] += 1
            stats["clipfrac"] += cf_tot
            stats["kl_approx"] += kl_tot
            stats["pg_loss"] += pg_tot
            stats["grad_norm"] += st.get("grad_norm", 0.0) or 0.0
    for k in ("clipfrac", "kl_approx", "pg_loss", "bc_loss", "grad_norm"):
        stats[k] = stats[k] / max(1, n_mb)
    policy.bb.eval()
    if log:
        log(f"ppo: {stats}")
    return stats


# ---------------------------------------------------------------------------- critic & world model
def state_text(ep: Episode, t: int, max_chars: int = 3500) -> str:
    """Compact state for the critic / world model: goal + recent history before step t."""
    hist = [f"ACTION {s.action[:300]}\nRESULT {s.result[:300]}" for s in ep.steps[:t]][-3:]
    s = f"DOMAIN {ep.domain}/{ep.family}\nGOAL {ep.goal[:1200]}\nSTEP {t}\n" + "\n".join(hist)
    return s[-max_chars:]


def critic_targets(eps: Sequence[Episode], gamma: float = 0.97) -> list[tuple[str, float]]:
    """Monte-Carlo targets from verified outcomes. Terminated: discounted final reward. Truncated
    episodes (step limit): treated as failure-at-limit — the task gives no reward after the limit,
    so the true continuation value is 0 (the step limit is part of the task)."""
    out = []
    for e in eps:
        T = len(e.steps)
        for t in range(T):
            out.append((state_text(e, t), (gamma ** (T - 1 - t)) * e.reward))
    return out


def train_critic(critic: Critic, trainer: GradientTrainer, data: list[tuple[str, float]], steps: int, batch: int = 16,
                 seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    critic.train()
    losses = []
    for _ in range(steps):
        b = rng.sample(data, min(batch, len(data)))
        loss = critic.loss([x for x, _ in b], torch.tensor([y for _, y in b], dtype=torch.float32))
        trainer.step(loss)
        losses.append(float(loss))
    critic.eval()
    return losses


@torch.no_grad()
def eval_critic(critic: Critic, data: list[tuple[str, float]], batch: int = 32) -> dict:
    if not data:
        return {}
    critic.eval()
    preds = []
    for s in range(0, len(data), batch):
        preds += critic.value([x for x, _ in data[s:s + batch]]).float().cpu().tolist()
    ys = [y for _, y in data]
    mu = sum(ys) / len(ys)
    mse = sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(ys)
    base = sum((mu - y) ** 2 for y in ys) / len(ys)
    return {"n": len(ys), "mse": mse, "mse_const_baseline": base, "explained_var": 1 - mse / base if base > 0 else None}


def wm_example(ep: Episode, t: int) -> tuple[str, str]:
    st = ep.steps[t]
    last = t == len(ep.steps) - 1
    outcome = ("success" if ep.success else "failure") if (last and ep.terminated) else ("truncated" if last else "continue")
    prompt = f"{state_text(ep, t)}\nNEXT ACTION {st.action[:400]}\nELAPSED_S {st.elapsed_s:.2f}\nPREDICT:"
    target = f" VALID {'yes' if st.valid else 'no'}\nOUTCOME {outcome}\nRESULT {st.result[:240]}"
    return prompt, target


def train_wm(wm: LMBackbone, trainer: GradientTrainer, data: list[tuple[str, str]], steps: int, batch: int = 16,
             seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    wm.train()
    losses = []
    for _ in range(steps):
        b = rng.sample(data, min(batch, len(data)))
        nll, cnt = wm.token_nll([p for p, _ in b], [t for _, t in b])
        loss = (nll / cnt).mean()
        trainer.step(loss)
        losses.append(float(loss))
    wm.eval()
    return losses


@torch.no_grad()
def eval_wm(wm: LMBackbone, data: list[tuple[str, str]], batch: int = 16) -> dict:
    """Per-token NLL of the full outcome text, and accuracy of the VALID / OUTCOME predictions vs majority."""
    if not data:
        return {}
    wm.eval()
    nll_tot, cnt_tot = 0.0, 0.0
    valid_correct = out_correct = 0
    vy, oy = [], []
    for s in range(0, len(data), batch):
        b = data[s:s + batch]
        nll, cnt = wm.token_nll([p for p, _ in b], [t for _, t in b])
        nll_tot += float(nll.sum())
        cnt_tot += float(cnt.sum())
        sc = wm.score([p + " VALID" for p, _ in b], [[" yes", " no"]] * len(b))
        for (p, t), v in zip(b, sc):
            truth = "yes" if t.startswith(" VALID yes") else "no"
            vy.append(truth)
            valid_correct += (["yes", "no"][int(torch.argmax(v))] == truth)
        labs = [" continue", " success", " failure", " truncated"]
        sc2 = wm.score([p + t.split("\nOUTCOME")[0] + "\nOUTCOME" for p, t in b], [labs] * len(b))
        for (p, t), v in zip(b, sc2):
            truth = t.split("OUTCOME ")[1].split("\n")[0]
            oy.append(truth)
            out_correct += (labs[int(torch.argmax(v))].strip() == truth)
    n = len(data)
    maj_v = max(vy.count("yes"), vy.count("no")) / n
    maj_o = max(oy.count(x) for x in set(oy)) / n
    return {"n": n, "nll_per_token": nll_tot / max(1, cnt_tot), "valid_acc": valid_correct / n, "valid_majority": maj_v,
            "outcome_acc": out_correct / n, "outcome_majority": maj_o}


# ---------------------------------------------------------------------------- evaluation summaries
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def summarize(eps: Sequence[Episode]) -> dict:
    def agg(es):
        n = len(es)
        k = sum(e.success for e in es)
        steps = sum(len(e.steps) for e in es)
        inv = sum(e.n_invalid for e in es)
        multi = [e for e in es if len(e.steps) > 1]
        lo, hi = wilson(k, n)
        return {"n": n, "success": k, "rate": k / n if n else 0.0, "ci95": [round(lo, 3), round(hi, 3)],
                "valid_action_rate": 1 - inv / steps if steps else 0.0, "avg_steps": steps / n if n else 0.0,
                "multi_step_success": sum(e.success for e in multi) / len(multi) if multi else None,
                "attempts_per_success": n / k if k else None}
    out = {"overall": agg(list(eps)), "by_domain": {}, "by_family": {}}
    for key in sorted({e.domain for e in eps}):
        out["by_domain"][key] = agg([e for e in eps if e.domain == key])
    for key in sorted({f"{e.domain}/{e.family}" for e in eps}):
        out["by_family"][key] = agg([e for e in eps if f"{e.domain}/{e.family}" == key])
    return out
