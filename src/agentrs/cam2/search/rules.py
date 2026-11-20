"""Typed, differentiable learning-rule expressions (learning-rule discovery, levels B-C of the task statement).

A rule is a pair of expression graphs evaluated per sample on the learner's batch:
  * ``policy``: per-sample loss whose batch mean the learner minimises (gradient reaches CAM through ``logp``/``ratio``);
  * ``vtarget``: the regression target for the value head (``ret`` = GAE return for the default rule).

Leaves (per-sample tensors computed by the learner):
  ratio, logratio, logp, logp_old, adv (normalised GAE advantage), adv_raw, ret (lambda-return), value (V_old),
  td (one-step TD error), ent (policy entropy), kl (logp_old - logp), q (Q(s,a) prediction, no grad), rew (reward)
Operators (all numerically guarded): add sub mul div neg abs square sqrt tanh sigmoid relu exp log1p clip min max sg
(stop-gradient) bnorm (batch standardisation) and constants.

This is *search over compositions of primitives*: it can express PPO-clip, A2C, (approximately) V-MPO / AWR-style
weightings, entropy-advantage couplings and new forms. It is not DiscoRL: there is no meta-network producing
targets (that is the separate ``MetaTargetNet`` below, a small first step, reported separately).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Optional

import torch

LEAVES = ["ratio", "logratio", "logp", "logp_old", "adv", "adv_raw", "ret", "value", "td", "ent", "kl", "q", "rew"]
UNARY = ["neg", "abs", "square", "sqrt", "tanh", "sigmoid", "relu", "exp", "log1p", "sg", "bnorm"]
BINARY = ["add", "sub", "mul", "div", "min", "max"]
SPECIAL = ["clip"]  # clip(x, lo, hi) with constant bounds


def _bnorm(x):
    return (x - x.mean()) / (x.std() + 1e-6) if x.numel() > 1 else x


OPS = {
    "neg": lambda a: -a, "abs": torch.abs, "square": lambda a: a.clamp(-1e3, 1e3) ** 2,
    "sqrt": lambda a: torch.sqrt(a.abs() + 1e-8), "tanh": torch.tanh, "sigmoid": torch.sigmoid, "relu": torch.relu,
    "exp": lambda a: torch.exp(a.clamp(-20, 5)), "log1p": lambda a: torch.log1p(a.abs()) * torch.sign(a),
    "sg": lambda a: a.detach(), "bnorm": _bnorm,
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
    "div": lambda a, b: a / (b.abs() + 1e-3) * torch.sign(b + 1e-12), "min": torch.minimum, "max": torch.maximum,
}


@dataclass(frozen=True)
class Node:
    op: str                        # leaf name, "const", or operator
    args: tuple = ()
    value: float = 0.0             # const value, or (for clip) unused
    lo: float = 0.0                # clip bounds
    hi: float = 0.0

    def to_obj(self) -> Any:
        if self.op in LEAVES:
            return self.op
        if self.op == "const":
            return round(self.value, 6)
        d = {"op": self.op, "args": [a.to_obj() for a in self.args]}
        if self.op == "clip":
            d["lo"], d["hi"] = round(self.lo, 6), round(self.hi, 6)
        return d

    @staticmethod
    def from_obj(o) -> "Node":
        if isinstance(o, str):
            if o not in LEAVES:
                raise ValueError(f"unknown leaf {o!r}")
            return Node(o)
        if isinstance(o, (int, float)):
            return Node("const", value=float(o))
        op = o["op"]
        if op not in OPS and op != "clip":
            raise ValueError(f"unknown operator {op!r}")
        args = tuple(Node.from_obj(a) for a in o["args"])
        n_expected = 1 if op in UNARY or op == "clip" else 2
        if len(args) != n_expected:
            raise ValueError(f"{op} takes {n_expected} argument(s)")
        return Node(op, args, lo=float(o.get("lo", 0.0)), hi=float(o.get("hi", 0.0)))

    def eval(self, env: dict) -> torch.Tensor:
        if self.op in LEAVES:
            return env[self.op]
        if self.op == "const":
            return torch.full_like(env["logp"], self.value)
        if self.op == "clip":
            return self.args[0].eval(env).clamp(self.lo, self.hi)
        return OPS[self.op](*[a.eval(env) for a in self.args])

    def size(self) -> int:
        return 1 + sum(a.size() for a in self.args)

    def depth(self) -> int:
        return 1 + max((a.depth() for a in self.args), default=0)

    def leaves(self) -> set:
        if self.op in LEAVES:
            return {self.op}
        out: set = set()
        for a in self.args:
            out |= a.leaves()
        return out

    def __str__(self) -> str:
        if self.op in LEAVES:
            return self.op
        if self.op == "const":
            return f"{self.value:.3g}"
        if self.op == "clip":
            return f"clip({self.args[0]}, {self.lo:.3g}, {self.hi:.3g})"
        return f"{self.op}({', '.join(map(str, self.args))})"


@dataclass(frozen=True)
class Rule:
    policy: Node
    vtarget: Node
    name: str = ""

    def to_json(self) -> str:
        return json.dumps({"policy": self.policy.to_obj(), "vtarget": self.vtarget.to_obj(), "name": self.name},
                          sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "Rule":
        d = json.loads(s) if isinstance(s, str) else s
        return Rule(Node.from_obj(d["policy"]), Node.from_obj(d["vtarget"]), d.get("name", ""))

    def __str__(self) -> str:
        return f"policy={self.policy} | vtarget={self.vtarget}"


def ppo_clip(eps: float = 0.2) -> Rule:
    p = Node.from_obj({"op": "neg", "args": [{"op": "min", "args": [
        {"op": "mul", "args": ["ratio", "adv"]},
        {"op": "mul", "args": [{"op": "clip", "args": ["ratio"], "lo": 1 - eps, "hi": 1 + eps}, "adv"]}]}]})
    return Rule(p, Node("ret"), "ppo_clip")


def a2c() -> Rule:
    return Rule(Node.from_obj({"op": "neg", "args": [{"op": "mul", "args": ["logp", {"op": "sg", "args": ["adv"]}]}]}),
                Node("ret"), "a2c")


SEED_RULES = {"ppo_clip": ppo_clip, "a2c": a2c}


# ---------------------------------------------------------------------------------------------------- validation
def synthetic_env(n: int = 256, seed: int = 0, requires_grad: bool = True) -> dict:
    g = torch.Generator().manual_seed(seed)
    logp_old = -torch.rand(n, generator=g) * 3
    logp = (logp_old + torch.randn(n, generator=g) * 0.1).requires_grad_(requires_grad)
    adv_raw = torch.randn(n, generator=g)
    value = torch.randn(n, generator=g) * 0.5
    ret = value + adv_raw
    env = {"logp": logp, "logp_old": logp_old, "logratio": logp - logp_old, "ratio": torch.exp(logp - logp_old),
           "adv": _bnorm(adv_raw), "adv_raw": adv_raw, "ret": ret, "value": value, "td": torch.randn(n, generator=g),
           "ent": torch.rand(n, generator=g) * 2, "kl": logp_old - logp, "q": value + torch.randn(n, generator=g) * 0.3,
           "rew": (torch.rand(n, generator=g) < 0.2).float()}
    return env


def check_rule(rule: Rule, max_size: int = 40, max_depth: int = 8) -> Optional[str]:
    """None if the rule is admissible: bounded size, finite loss and gradients on synthetic batches, a policy
    gradient path, and a descent direction that raises the log-probability of positive-advantage samples on average
    (otherwise it is not a policy-improvement rule at all). Returns the failure reason otherwise."""
    for part, node in (("policy", rule.policy), ("vtarget", rule.vtarget)):
        if node.size() > max_size or node.depth() > max_depth:
            return f"{part} expression too large"
    if not ({"logp", "ratio", "logratio", "kl"} & rule.policy.leaves()):
        return "policy loss has no path to the policy's log-probability"
    for seed in range(3):
        env = synthetic_env(seed=seed)
        try:
            loss = rule.policy.eval(env).mean()
            vt = rule.vtarget.eval({k: (v.detach() if torch.is_tensor(v) else v) for k, v in env.items()})
        except Exception as e:  # noqa: BLE001
            return f"evaluation error: {e}"
        if not torch.isfinite(loss) or not torch.isfinite(vt).all():
            return "non-finite loss or value target"
        if vt.abs().max() > 1e3:
            return "value target out of range"
        (g,) = torch.autograd.grad(loss, env["logp"], allow_unused=True)
        if g is None or not torch.isfinite(g).all():
            return "no finite gradient w.r.t. logp"
        if float(g.abs().max()) > 1e3:
            return "exploding gradient"
        if float(g.abs().mean()) < 1e-7:
            return "vanishing gradient"
        # descent step -g: d(logp) = -g; require corr(-g, adv) > 0
        c = torch.corrcoef(torch.stack([-g, env["adv"]]))[0, 1]
        if not torch.isfinite(c) or float(c) <= 0.05:
            return f"not a policy-improvement direction (corr={float(c):.3f})"
    return None


# ---------------------------------------------------------------------------------------------------- mutation
def random_tree(rng: random.Random, depth: int, leaves=LEAVES) -> Node:
    if depth <= 1 or rng.random() < 0.3:
        if rng.random() < 0.15:
            return Node("const", value=rng.choice([0.5, 1.0, 2.0, 0.1, 0.01, -1.0]))
        return Node(rng.choice(leaves))
    r = rng.random()
    if r < 0.45:
        return Node(rng.choice(UNARY), (random_tree(rng, depth - 1, leaves),))
    if r < 0.9:
        return Node(rng.choice(BINARY), (random_tree(rng, depth - 1, leaves), random_tree(rng, depth - 1, leaves)))
    lo = rng.choice([-1.0, 0.0, 0.5, 0.8])
    return Node("clip", (random_tree(rng, depth - 1, leaves),), lo=lo, hi=lo + rng.choice([0.4, 1.0, 2.0]))


def _subtrees(n: Node, path=()):
    yield path, n
    for i, a in enumerate(n.args):
        yield from _subtrees(a, path + (i,))


def _replace(n: Node, path: tuple, new: Node) -> Node:
    if not path:
        return new
    args = list(n.args)
    args[path[0]] = _replace(args[path[0]], path[1:], new)
    return Node(n.op, tuple(args), n.value, n.lo, n.hi)


def mutate_node(n: Node, rng: random.Random) -> tuple[Node, str]:
    subs = list(_subtrees(n))
    path, sub = rng.choice(subs)
    r = rng.random()
    if r < 0.3:
        return _replace(n, path, random_tree(rng, rng.randint(1, 3))), "subtree"
    if r < 0.5 and sub.op in UNARY + BINARY:
        pool = UNARY if sub.op in UNARY else BINARY
        return _replace(n, path, Node(rng.choice(pool), sub.args)), "op_swap"
    if r < 0.65 and sub.op == "const":
        return _replace(n, path, Node("const", value=sub.value * rng.choice([0.5, 2.0, -1.0]))), "const"
    if r < 0.8 and sub.op == "clip":
        d = rng.choice([-0.1, 0.1])
        return _replace(n, path, Node("clip", sub.args, lo=sub.lo + d, hi=max(sub.lo + d + 0.05, sub.hi - d))), "clip_bounds"
    if r < 0.9:
        return _replace(n, path, Node(rng.choice(UNARY), (sub,))), "wrap_unary"
    return _replace(n, path, Node(rng.choice(BINARY), (sub, random_tree(rng, 2)))), "wrap_binary"


def mutate_rule(rule: Rule, rng: random.Random, max_tries: int = 50) -> tuple[Rule, str]:
    for _ in range(max_tries):
        if rng.random() < 0.85:
            p, how = mutate_node(rule.policy, rng)
            cand = Rule(p, rule.vtarget, "")
            how = "policy:" + how
        else:
            v, how = mutate_node(rule.vtarget, rng)
            cand = Rule(rule.policy, v, "")
            how = "vtarget:" + how
        if check_rule(cand) is None and str(cand) != str(rule):
            return cand, how
    return rule, "none"


def crossover(a: Rule, b: Rule, rng: random.Random, max_tries: int = 50) -> Rule:
    for _ in range(max_tries):
        pa, _ = rng.choice(list(_subtrees(a.policy)))
        _, sb = rng.choice(list(_subtrees(b.policy)))
        cand = Rule(_replace(a.policy, pa, sb), rng.choice([a.vtarget, b.vtarget]))
        if check_rule(cand) is None and str(cand) not in (str(a), str(b)):
            return cand
    return a


def equivalent_to_ppo_family(rule: Rule) -> bool:
    """Heuristic flag used in reports: does the rule merely re-tune PPO's / A2C's scalar coefficients? (Same
    structure as a seed rule up to constants.)"""
    def shape(n: Node):
        return (n.op if n.op != "const" else "c", tuple(shape(a) for a in n.args))
    return any(shape(rule.policy) == shape(f().policy) for f in SEED_RULES.values())


class MetaTargetNet(torch.nn.Module):
    """Option D (small first step): a meta-network mapping per-sample learning statistics to a bounded advantage
    re-weighting, w = 2*sigmoid(net([adv, td, ent, kl, ratio-1])), applied as loss_i * w_i (w detached in the inner
    update). Its parameters are searched by the outer loop (evolution strategies over candidates), not by inner
    gradients. Equal to 1 everywhere at initialisation."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(5, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        torch.nn.init.zeros_(self.net[2].weight)
        torch.nn.init.zeros_(self.net[2].bias)

    def forward(self, env: dict) -> torch.Tensor:
        x = torch.stack([env["adv"], env["td"], env["ent"], env["kl"], env["ratio"] - 1], -1).detach()
        return 2 * torch.sigmoid(self.net(x).squeeze(-1))

    def flat(self) -> list:
        return torch.cat([p.detach().flatten() for p in self.parameters()]).tolist()

    def load_flat(self, v: list) -> None:
        t = torch.tensor(v, dtype=torch.float32)
        i = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(t[i:i + n].view_as(p))
            i += n
