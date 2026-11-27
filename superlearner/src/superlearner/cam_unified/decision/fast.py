"""The fast parameter set phi of the meta-gradient (plan 2026-11-26 §2 'Inner', W5 -> W8).

    phi = intent head + tool / evidence pointers + typed heads + length prior + candidate evaluator + V_int

Core, Qwen, the world model, the beliefs, the tokenizer and the bridge / decoder adapters are never inner-updated.
`fast_params(model)` returns {name: Parameter} with the model's own parameter names (model.named_parameters()), so
the functional inner loop of selfeval/metagrad.py can override exactly these tensors:

    feats = model.policy_features(inp, actions)                  # detached core / reasoner / beliefs / WM prior outputs
    out = functional_policy(model, fast, feats, actions)          # torch.func.functional_call: logp [B], value, ...
    # ... inner step on phi, outer step on psi (W8)

`functional_policy` runs `model.forward(inp, actions, features=feats)` = `policy_from_features` with phi replaced
by `fast` (any subset of fast_params; the rest of the model keeps its own tensors).
"""
from __future__ import annotations

from typing import Optional

import torch

# action-generator parameter name prefixes in the fast set (relative to model.action_generator)
GENERATOR_FAST = ("ctx_norm.", "tool_query", "tool_reader.", "tool_q.", "tool_k.", "tool_a", "ev_", "heads.",
                  "length_head.")
MODULE_FAST = ("intent_head.", "decision_evaluator.", "value_head.")


def is_fast(name: str) -> bool:
    if name.startswith(MODULE_FAST):
        return True
    if name.startswith("action_generator."):
        rest = name[len("action_generator."):]
        return rest.startswith(GENERATOR_FAST)
    return False


def fast_params(model, trainable_only: bool = True) -> dict:
    """{model parameter name: Parameter} of the fast set phi (DOM decision path: requires model.dom_decision)."""
    if not getattr(model, "dom_decision", False):
        raise ValueError("fast_params: the model has no DOM decision bottleneck (cfg.n_intent == 0)")
    out = {}
    for n, p in model.named_parameters():
        if is_fast(n) and (p.requires_grad or not trainable_only):
            out[n] = p
    return out


def slow_params(model) -> dict:
    """Every trainable parameter not in the fast set."""
    fast = fast_params(model)
    return {n: p for n, p in model.named_parameters() if p.requires_grad and n not in fast}


def fast_param_count(model) -> dict:
    """{"fast": n, "by_module": {prefix: n}, "slow": n} parameter counts."""
    fast = fast_params(model)
    by = {}
    for n, p in fast.items():
        key = n.split(".")[0] if not n.startswith("action_generator.") else \
            "action_generator." + n.split(".")[1]
        by[key] = by.get(key, 0) + p.numel()
    return {"fast": sum(p.numel() for p in fast.values()), "by_module": by,
            "slow": sum(p.numel() for p in slow_params(model).values())}


def functional_policy(model, fast: dict, feats: dict, actions: list) -> dict:
    """policy_from_features with phi := `fast` (torch.func.functional_call; gradients flow into the given tensors,
    e.g. inner-updated fast weights that depend on psi)."""
    from torch.func import functional_call
    bad = [n for n in fast if not is_fast(n)]
    if bad:
        raise KeyError(f"not fast-set parameters: {bad[:5]}")
    return functional_call(model, fast, (feats["inp"], actions), {"features": feats}, strict=False)


@torch.no_grad()
def consolidate(model, fast: dict) -> int:
    """Copy inner-updated fast weights into the slow weights (bump the weights epoch). Returns tensors copied."""
    params = dict(model.named_parameters())
    n = 0
    for k, v in fast.items():
        if k not in params or not is_fast(k):
            raise KeyError(f"{k!r} is not a fast-set parameter of the model")
        params[k].copy_(v.detach().to(params[k].dtype))
        n += 1
    if hasattr(model, "bump_weights_epoch"):
        model.bump_weights_epoch()
    return n


def fast_state(model, fast: Optional[dict] = None) -> dict:
    """Detached clones of the fast set (e.g. to act with phi_K and restore afterwards)."""
    src = fast if fast is not None else fast_params(model)
    return {k: v.detach().clone() for k, v in src.items()}
