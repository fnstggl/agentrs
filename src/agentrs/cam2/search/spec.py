"""Typed, bounded candidate specifications for the outer architecture/learning-rule search.

A candidate is **data**, never code: ``{"arch": {...}, "learn": {...}, "plan": {...}}`` whose keys and values are
restricted to ``SPACE`` (architecture: ``CAMConfig`` fields; learning algorithm: ``LearnConfig`` fields including the
learning-rule expression graph and the optional meta-target network; planning: ``PlanConfig`` fields). The outer
learner can only produce specs inside this space; nothing it proposes is executed as Python.

``validate(spec)`` runs every admission check required before a candidate may consume training budget:
  1. schema validation (known keys, types, ranges, cross-field constraints);
  2. learning-rule admissibility (``rules.check_rule``: finite, gradient path, policy-improvement direction);
  3. graph construction (the CAM module builds with the frozen backbone's dimensions);
  4. resource budget (trainable parameters <= ``max_params``; measured peak memory of a training step);
  5. forward pass, backward pass, finite loss, non-zero gradients into every parameter group that should get one
     (on a synthetic batch with the real tensor shapes; the frozen LLM is not needed for this check);
  6. safety: no field can name code, files, evaluation data or the evaluator.

Mutation operators (``OPERATORS``) are named so the meta-controller can learn which kinds of change pay off.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, fields
from typing import Any, Callable, Optional

import torch

from agentrs.cam2.model import CAM, CAMConfig, EL_FEATS
from agentrs.cam2.planner import PlanConfig
from agentrs.cam2.search.rules import MetaTargetNet, Rule, check_rule, crossover, mutate_rule, ppo_clip


@dataclass(frozen=True)
class Dim:
    kind: str                         # "choice" | "int" | "float" | "logfloat" | "bool"
    choices: tuple = ()
    lo: float = 0.0
    hi: float = 0.0


SPACE: dict = {
    "arch": {
        "d": Dim("choice", (256, 384, 512, 768)),
        "integ_layers": Dim("int", lo=1, hi=6),
        "heads": Dim("choice", (4, 8)),
        "ff_mult": Dim("choice", (2, 4)),
        "core": Dim("choice", ("gru", "lstm", "none")),
        "core_dim": Dim("choice", (512, 768, 1024, 1536)),
        "state_tokens": Dim("int", lo=0, hi=8),
        "mem_k": Dim("choice", (0, 4, 8, 16)),
        "wm_ensemble": Dim("int", lo=1, hi=7),
        "wm_hidden": Dim("choice", (512, 1024, 1536)),
        "wm_layers": Dim("int", lo=1, hi=3),
        "prefix_len": Dim("choice", (0, 4, 8, 16)),
        "prefix_rank": Dim("choice", (64, 128, 256)),
        "head_hidden": Dim("choice", (256, 512, 768)),
        "q_head": Dim("bool"),
        "deliberation": Dim("bool"),
        "el_goal_cond": Dim("bool"),
    },
    "learn": {
        "lr": Dim("logfloat", lo=1e-5, hi=3e-4),
        "gamma": Dim("choice", (0.9, 0.95, 0.98, 0.99)),
        "lam": Dim("choice", (0.8, 0.9, 0.95)),
        "epochs": Dim("int", lo=1, hi=4),
        "ent": Dim("logfloat", lo=1e-3, hi=5e-2),
        "vf": Dim("choice", (0.25, 0.5, 1.0)),
        "q_coef": Dim("choice", (0.0, 0.25, 0.5)),
        "wm_coef": Dim("choice", (0.0, 0.25, 0.5, 1.0)),
        "distill_coef": Dim("choice", (0.0, 0.1, 0.5, 1.0)),
        "imagine_coef": Dim("choice", (0.0, 0.1, 0.5)),
        "imagine_horizon": Dim("int", lo=1, hi=5),
        "replay_episodes": Dim("choice", (16, 32, 64)),
        "target_kl": Dim("choice", (0.04, 0.08, 0.15)),
        "max_grad": Dim("choice", (0.5, 1.0, 2.0)),
        "temperature": Dim("choice", (0.7, 1.0)),
        "rule": Dim("rule"),
        "meta_target": Dim("meta"),
    },
    "plan": {
        "enabled": Dim("bool"),
        "top_k": Dim("choice", (4, 8, 16)),
        "horizon": Dim("int", lo=1, hi=5),
        "kappa": Dim("choice", (1.0, 2.0, 4.0, 8.0)),
        "beta": Dim("choice", (0.0, 0.5, 1.0, 2.0)),
        "min_reliability": Dim("choice", (0.0, 0.1, 0.2, 0.4)),
    },
}

# Learner/run settings that are part of the fixed experimental protocol (identical budgets for every candidate);
# the outer learner cannot change them.
PROTOCOL_FIXED_LEARN = ("episodes_per_update", "mb_episodes", "replay_capacity", "carry_state", "text_chunk",
                        "warmup_steps", "weight_decay", "distill_v")


def default_spec(base: Optional[dict] = None) -> dict:
    """The search's seed candidate: CAMConfig / LearnConfig / PlanConfig defaults restricted to the space,
    overridden by ``base`` (e.g. the hand-designed run config)."""
    from agentrs.cam2.learner import LearnConfig

    cam, lc, pc = CAMConfig(), LearnConfig(), PlanConfig()
    spec = {"arch": {k: getattr(cam, k) for k in SPACE["arch"]},
            "learn": {k: getattr(lc, k) for k in SPACE["learn"]},
            "plan": {k: getattr(pc, k) for k in SPACE["plan"]}}
    for sec, vals in (base or {}).items():
        for k, v in vals.items():
            if k in SPACE.get(sec, {}):
                spec[sec][k] = v
    return spec


def spec_hash(spec: dict) -> str:
    return hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------- validation
class SpecError(ValueError):
    pass


def _check_value(sec: str, k: str, dim: Dim, v: Any) -> None:
    if dim.kind == "choice":
        if v not in dim.choices:
            raise SpecError(f"{sec}.{k}={v!r} not in {dim.choices}")
    elif dim.kind == "int":
        if not isinstance(v, int) or isinstance(v, bool) or not (dim.lo <= v <= dim.hi):
            raise SpecError(f"{sec}.{k}={v!r} must be an int in [{dim.lo}, {dim.hi}]")
    elif dim.kind in ("float", "logfloat"):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not (dim.lo <= v <= dim.hi):
            raise SpecError(f"{sec}.{k}={v!r} must be a number in [{dim.lo}, {dim.hi}]")
    elif dim.kind == "bool":
        if not isinstance(v, bool):
            raise SpecError(f"{sec}.{k}={v!r} must be a bool")
    elif dim.kind == "rule":
        try:
            r = Rule.from_json(v)
        except Exception as e:  # noqa: BLE001
            raise SpecError(f"learn.rule does not parse: {e}") from e
        err = check_rule(r)
        if err:
            raise SpecError(f"learn.rule rejected: {err}")
    elif dim.kind == "meta":
        if v is None:
            return
        n = sum(p.numel() for p in MetaTargetNet().parameters())
        if not isinstance(v, list) or len(v) != n or not all(isinstance(x, (int, float)) and math.isfinite(x) and abs(x) < 50 for x in v):
            raise SpecError(f"learn.meta_target must be None or a list of {n} finite numbers with |x| < 50")
    else:
        raise SpecError(f"unknown dimension kind {dim.kind}")


def check_schema(spec: dict) -> None:
    if not isinstance(spec, dict) or set(spec) != set(SPACE):
        raise SpecError(f"spec must have exactly the sections {sorted(SPACE)}")
    for sec, dims in SPACE.items():
        vals = spec[sec]
        if not isinstance(vals, dict):
            raise SpecError(f"section {sec} must be a dict")
        unknown = set(vals) - set(dims)
        if unknown:
            raise SpecError(f"unknown {sec} keys {sorted(unknown)} (the space is closed)")
        missing = set(dims) - set(vals)
        if missing:
            raise SpecError(f"missing {sec} keys {sorted(missing)}")
        for k, dim in dims.items():
            _check_value(sec, k, dim, vals[k])
    a = spec["arch"]
    if a["d"] % a["heads"]:
        raise SpecError("arch.d must be divisible by arch.heads")
    if spec["plan"]["enabled"] and spec["learn"]["wm_coef"] == 0:
        raise SpecError("planning needs a trained world model (learn.wm_coef > 0)")
    if spec["learn"]["imagine_coef"] > 0 and spec["learn"]["wm_coef"] == 0:
        raise SpecError("imagination training needs a trained world model (learn.wm_coef > 0)")
    if spec["arch"]["deliberation"] and not spec["plan"]["enabled"]:
        raise SpecError("the deliberation head allocates planning effort; it needs plan.enabled")


def cam_config(spec: dict, llm_pooled_dim: int, llm_hidden: int) -> CAMConfig:
    return CAMConfig(**{**spec["arch"], "llm_pooled_dim": llm_pooled_dim, "llm_hidden": llm_hidden})


def count_params(spec: dict, llm_pooled_dim: int, llm_hidden: int) -> dict:
    with torch.device("meta"):
        m = CAM(cam_config(spec, llm_pooled_dim, llm_hidden))
    return m.n_params()


def synthetic_batch(cfg: CAMConfig, B: int = 4, Ne: int = 12, Nc: int = 24, Tg: int = 10, device="cpu",
                    seed: int = 0) -> dict:
    """A batch with exactly the shapes and masks ``featurize.collate`` produces (random feature values)."""
    g = torch.Generator().manual_seed(seed)
    Dp, H = cfg.llm_pooled_dim, cfg.llm_hidden
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    el_m = torch.ones(B, Ne, dtype=torch.bool)
    el_m[0, Ne // 2:] = False
    cm = torch.ones(B, Nc, dtype=torch.bool)
    cm[1, Nc // 2:] = False
    b = {"goal": r(B, Tg, H), "goal_m": torch.ones(B, Tg, dtype=torch.bool), "el": r(B, Ne, Dp), "el_m": el_m,
         "el_f": torch.rand(B, Ne, EL_FEATS, generator=g), "text": r(B, Dp), "text_m": torch.tensor([True, False] * (B // 2) + [True] * (B % 2)),
         "pact": r(B, Dp), "scal": torch.rand(B, 4, generator=g), "tool_tab": r(6, Dp), "enum_tab": r(3, Dp),
         "c_tool": torch.randint(0, 6, (B, Nc), generator=g), "c_el": torch.randint(-1, Ne // 2, (B, Nc), generator=g),
         "c_enum": torch.randint(-1, 3, (B, Nc), generator=g), "cmask": cm}
    if cfg.mem_k > 0:
        b["mem"] = r(B, cfg.mem_k, cfg.core_dim + 4)
        mm = torch.ones(B, cfg.mem_k, dtype=torch.bool)
        mm[0, :] = False
        b["mem_m"] = mm
    return {k: v.to(device) for k, v in b.items()}


def forward_backward_check(spec: dict, llm_pooled_dim: int, llm_hidden: int, device: str = "cpu", T: int = 3) -> dict:
    """Build the candidate's CAM, run a T-step recurrent forward through every head (policy, value, Q, world model,
    control vectors, deliberation), backpropagate a combined loss and check finiteness and gradient coverage."""
    torch.manual_seed(0)
    cfg = cam_config(spec, llm_pooled_dim, llm_hidden)
    t0 = time.time()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
    cam = CAM(cfg).to(device)
    st = cam.init_state(4, device)
    loss = torch.zeros((), device=device)
    for t in range(T):
        b = synthetic_batch(cfg, device=device, seed=t)
        f, el_ctx, st, x = cam.integrate(b, st)
        cand = cam.candidates(b, el_ctx)
        lg = cam.policy_logits(f, cand, b["cmask"])
        lp = torch.log_softmax(lg, -1)
        loss = loss - lp[:, 0].mean() + cam.v(f).pow(2).mean()
        if cfg.q_head:
            loss = loss + cam.q_values(f, cand[:, :2]).pow(2).mean()
        if cfg.deliberation:
            loss = loss - torch.log_softmax(cam.delib_logits(f), -1)[:, 1].mean()
        if cfg.prefix_len > 0:
            loss = loss + cam.prefix(f, cand[:, 0]).pow(2).mean() * 1e-3
        ae = cam.action_embedding(cand[:, 0], b["pact"], torch.ones(4, dtype=torch.bool, device=device))
        wm = cam.wm_predict(f, ae)
        loss = loss + wm["next_f"].pow(2).mean() + wm["reward"].pow(2).mean() + wm["done"].pow(2).mean()
    if not torch.isfinite(loss):
        raise SpecError("non-finite loss in the forward check")
    loss.backward()
    groups = {"integration": ["integ", "el_in", "goal_in"], "core": ["core"], "policy_value": ["pol_q", "value", "cand"],
              "world_model": ["wm"], "llm_control": ["pre_a", "pre_b"] if cfg.prefix_len > 0 else []}
    if cfg.mem_k > 0:
        groups["memory"] = ["mem_v"]
    for gname, prefixes in groups.items():
        gs = [p.grad for n, p in cam.named_parameters() if n.split(".")[0] in prefixes]
        if not gs:
            continue
        if any(x is None for x in gs) and all(x is None for x in gs):
            raise SpecError(f"no gradient reaches {gname}")
        tot = sum(float(x.abs().sum()) for x in gs if x is not None)
        if not math.isfinite(tot):
            raise SpecError(f"non-finite gradient in {gname}")
        if tot == 0.0:
            raise SpecError(f"zero gradient in {gname}")
    res = {"loss": float(loss.detach()), "check_s": time.time() - t0, "params": cam.n_params()["total"]}
    if device.startswith("cuda"):
        res["peak_mem_gb"] = (torch.cuda.max_memory_allocated() - base_mem) / 2**30
    del cam
    return res


@dataclass
class Limits:
    max_params: int = 120_000_000
    max_check_mem_gb: float = 8.0


def validate(spec: dict, llm_pooled_dim: int, llm_hidden: int, limits: Optional[Limits] = None,
             device: str = "cpu", run_graph: bool = True) -> dict:
    """All admission checks. Raises ``SpecError`` with the reason; returns a report on success."""
    limits = limits or Limits()
    check_schema(spec)                      # 1, 2 (rule check runs inside), 6 (closed key set)
    n = count_params(spec, llm_pooled_dim, llm_hidden)   # 3 (graph construction on the meta device)
    if n["total"] > limits.max_params:
        raise SpecError(f"{n['total']:,} trainable parameters exceed the budget of {limits.max_params:,}")
    rep = {"params": n}
    if run_graph:
        fb = forward_backward_check(spec, llm_pooled_dim, llm_hidden, device=device)   # 5
        if fb.get("peak_mem_gb", 0.0) > limits.max_check_mem_gb:
            raise SpecError(f"training-step memory {fb['peak_mem_gb']:.1f} GB exceeds {limits.max_check_mem_gb} GB")
        rep["forward_backward"] = fb
    return rep


# ------------------------------------------------------------------------------------------------- mutation
def _step_choice(dim: Dim, v, rng: random.Random):
    ch = list(dim.choices)
    if v not in ch:
        return rng.choice(ch)
    i = ch.index(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and all(isinstance(c, (int, float)) for c in ch):
        j = max(0, min(len(ch) - 1, i + rng.choice([-1, 1])))
        if j == i:
            j = i - 1 if i > 0 else i + 1
        return ch[j]
    return rng.choice([c for c in ch if c != v])


def perturb(sec: str, k: str, v, rng: random.Random):
    dim = SPACE[sec][k]
    if dim.kind == "choice":
        return _step_choice(dim, v, rng)
    if dim.kind == "int":
        nv = int(v) + rng.choice([-1, 1])
        if nv < dim.lo or nv > dim.hi:
            nv = int(v) - (nv - int(v))
        return int(min(dim.hi, max(dim.lo, nv)))
    if dim.kind == "bool":
        return not v
    if dim.kind == "logfloat":
        nv = float(v) * rng.choice([0.5, 0.8, 1.25, 2.0])
        return float(min(dim.hi, max(dim.lo, nv)))
    if dim.kind == "float":
        nv = float(v) + (dim.hi - dim.lo) * rng.uniform(-0.2, 0.2)
        return float(min(dim.hi, max(dim.lo, nv)))
    raise ValueError(dim.kind)


def _mutate_fields(keys: list, n: int = 1):
    def op(spec: dict, rng: random.Random, ctx: dict) -> tuple[dict, str]:
        s = copy.deepcopy(spec)
        chosen = rng.sample(keys, min(n, len(keys)))
        desc = []
        for sec, k in chosen:
            old = s[sec][k]
            s[sec][k] = perturb(sec, k, old, rng)
            desc.append(f"{sec}.{k}: {old} -> {s[sec][k]}")
        return s, "; ".join(desc)
    return op


def _op_rule(spec, rng, ctx):
    s = copy.deepcopy(spec)
    r, how = mutate_rule(Rule.from_json(s["learn"]["rule"]), rng)
    s["learn"]["rule"] = r.to_json()
    return s, f"rule {how}: {r}"


def _op_rule_crossover(spec, rng, ctx):
    others = [o for o in ctx.get("population_rules", []) if o != spec["learn"]["rule"]]
    if not others:
        return _op_rule(spec, rng, ctx)
    s = copy.deepcopy(spec)
    r = crossover(Rule.from_json(s["learn"]["rule"]), Rule.from_json(rng.choice(others)), rng)
    s["learn"]["rule"] = r.to_json()
    return s, f"rule crossover: {r}"


def _op_meta_target(spec, rng, ctx):
    """Evolution-strategies step on the meta-target network's parameters (option D: the outer loop searches the
    parameters of a network that re-weights the inner policy-gradient targets)."""
    s = copy.deepcopy(spec)
    cur = s["learn"]["meta_target"]
    if cur is None:
        cur = MetaTargetNet().flat()
    sigma = rng.choice([0.1, 0.3, 1.0])
    new = [float(max(-49.0, min(49.0, x + rng.gauss(0.0, sigma)))) for x in cur]
    s["learn"]["meta_target"] = new
    return s, f"meta_target ES step sigma={sigma}"


def _op_plan(spec, rng, ctx):
    s = copy.deepcopy(spec)
    if not s["plan"]["enabled"]:
        s["plan"]["enabled"] = True
        if s["learn"]["wm_coef"] == 0:
            s["learn"]["wm_coef"] = 0.5
        return s, "plan.enabled: False -> True"
    if rng.random() < 0.2:
        s["plan"]["enabled"] = False
        s["arch"]["deliberation"] = False
        return s, "plan.enabled: True -> False"
    k = rng.choice(["top_k", "horizon", "kappa", "beta", "min_reliability"])
    old = s["plan"][k]
    s["plan"][k] = perturb("plan", k, old, rng)
    return s, f"plan.{k}: {old} -> {s['plan'][k]}"


A = lambda *ks: [("arch", k) for k in ks]  # noqa: E731
L = lambda *ks: [("learn", k) for k in ks]  # noqa: E731

OPERATORS: dict[str, Callable] = {
    # architecture
    "arch.width": _mutate_fields(A("d", "core_dim", "head_hidden", "ff_mult")),
    "arch.depth": _mutate_fields(A("integ_layers", "wm_layers")),
    "arch.recurrence": _mutate_fields(A("core", "state_tokens")),
    "arch.memory": _mutate_fields(A("mem_k")),
    "arch.world_model": _mutate_fields(A("wm_ensemble", "wm_hidden", "wm_layers")),
    "arch.llm_interface": _mutate_fields(A("prefix_len", "prefix_rank", "el_goal_cond")),
    "arch.heads": _mutate_fields(A("heads", "q_head")),
    # learning algorithm
    "learn.optim": _mutate_fields(L("lr", "epochs", "target_kl", "max_grad")),
    "learn.return": _mutate_fields(L("gamma", "lam")),
    "learn.aux": _mutate_fields(L("wm_coef", "q_coef", "imagine_coef", "imagine_horizon")),
    "learn.retention": _mutate_fields(L("distill_coef", "replay_episodes")),
    "learn.entropy": _mutate_fields(L("ent", "temperature")),
    # learning-rule discovery
    "rule.expr": _op_rule,
    "rule.crossover": _op_rule_crossover,
    "rule.meta_target": _op_meta_target,
    # planning
    "plan": _op_plan,
}
# classical PBT only perturbs hyper-parameters of a fixed architecture/algorithm
PBT_OPERATORS = ["learn.optim", "learn.entropy", "learn.return"]


def repair(spec: dict) -> dict:
    """Restore cross-field constraints after a mutation (deterministic, documented)."""
    s = copy.deepcopy(spec)
    a = s["arch"]
    if a["d"] % a["heads"]:
        a["heads"] = 4
    if s["plan"]["enabled"] and s["learn"]["wm_coef"] == 0:
        s["learn"]["wm_coef"] = 0.25
    if s["learn"]["imagine_coef"] > 0 and s["learn"]["wm_coef"] == 0:
        s["learn"]["imagine_coef"] = 0.0
    if a["deliberation"] and not s["plan"]["enabled"]:
        a["deliberation"] = False
    return s


def mutate(spec: dict, op: str, rng: random.Random, ctx: Optional[dict] = None, tries: int = 20) -> tuple[dict, str]:
    """Apply operator ``op``; retried until the result differs from the parent and passes the schema checks."""
    last = None
    for _ in range(tries):
        s, desc = OPERATORS[op](spec, rng, ctx or {})
        s = repair(s)
        if s == spec:
            continue
        try:
            check_schema(s)
        except SpecError as e:
            last = str(e)
            continue
        return s, desc
    raise SpecError(f"operator {op} produced no valid change ({last})")


def inherit_weights(child, parent_state: dict) -> dict:
    """Copy every parent tensor whose name and shape match the child's (weight inheritance across a mutation).
    Returns counts. Optimizer state is deliberately NOT inherited: Adam moments belong to the parent's parameter
    shapes and loss landscape; the child starts with fresh moments and a fresh learning-rate warm-up."""
    own = child.state_dict()
    copied, skipped, n_cop, n_tot = [], [], 0, 0
    for k, v in own.items():
        n_tot += v.numel()
        pv = parent_state.get(k)
        if pv is not None and pv.shape == v.shape and pv.dtype == v.dtype:
            own[k] = pv.clone()
            copied.append(k)
            n_cop += v.numel()
        else:
            skipped.append(k)
    child.load_state_dict(own)
    return {"tensors_copied": len(copied), "tensors_fresh": len(skipped), "frac_params_inherited": n_cop / max(1, n_tot)}
