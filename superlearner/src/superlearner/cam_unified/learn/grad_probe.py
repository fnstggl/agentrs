"""Named-group gradient routing for the integrated learner (plan 2026-11-26 §2 / §3 W7; DOM-08 "verified experiential
gradients throughout the core"; audit 'learning' finding: the per-objective probe measured only the core as one
block and its output was dropped from the logs).

`named_groups(model, reward_model)` assigns every trainable parameter of the CAM model (and of the reward model) to
exactly one DOM-08 group by its qualified name, first matching rule wins:

    core.block0 / core.block_mid / core.block_last   blocks 0, L//2 and L-1 of the shared backbone F_theta
    core.blocks_other                                 the remaining backbone blocks
    core.sem_xattn_gate / core.sem_xattn              Mechanism C gates / projections (gated semantic cross-attention)
    core.state_gate                                   the gated state update (W3)
    core.sem_proj                                     Mechanism C's semantic projection inside the core (W3: a core
                                                      tensor - lr_core, counted in the core share)
    core.embed                                        the rest of the core (state / kind / time embeddings, norms)
    sem_proj                                          tokenizer projections (numeric value channel, pools, unit/desc)
    entities, reasoner, beliefs (+ bayes), world_model, outcome_head (+ query encoder)
    intent, evaluator                                 decision bottleneck (W5: IntentHead, decision evaluator)
    dec_xattn, dec_bridge                             semantic decoder conditioning (W4 decoder cross-attention, bridge)
    exploration                                       the BUDGET / exploration head (learned exploration input)
    policy                                            action generator heads + option policy
    v_int                                             value head V(R, G)
    predictors / psi                                  reward-model parameters: supervised predictors / meta params psi
    other                                             anything unmatched (reported, never dropped)

A group whose rule matches no parameter is reported as ``absent`` (the component is not built), never as zero.

`objective_gradients(weighted, params, groups)` back-propagates each weighted objective separately with
``torch.autograd.grad`` and accumulates the result into ``.grad`` (so the optimiser step is exactly the step of the
summed loss), recording per (objective, group) the L2 norm of that objective's gradient. Because the objectives'
graphs are largely disjoint (replay filter vs. decision recomputation), this costs about one combined backward pass,
not one pass per objective. `GradTable` then gives each objective's share of the core gradient (norm_i / sum_j
norm_j over the core groups, the convention of the CAM 7.0 audit), pairwise cosines on the core (exact up to
`cos_exact_numel` coordinates, else on a fixed random sample), and a JSON-ready nested dict.
"""
from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor

CORE_PREFIX = "core."
# DOM-08 groups the real end-to-end check requires to receive gradient (plan §3 W7)
DOM08_REQUIRED = ("core.block0", "core.block_mid", "core.block_last", "sem_proj", "core.sem_xattn_gate", "reasoner",
                  "beliefs", "world_model", "intent", "dec_xattn", "exploration", "policy", "v_int", "predictors")
GROUP_ORDER = ("core.block0", "core.block_mid", "core.block_last", "core.blocks_other", "core.sem_xattn_gate",
               "core.sem_xattn", "core.state_gate", "core.sem_proj", "core.embed", "sem_proj", "entities", "reasoner", "beliefs",
               "world_model", "outcome_head", "intent", "evaluator", "dec_xattn", "dec_bridge", "exploration",
               "policy", "v_int", "predictors", "psi", "other")
# a DOM-08 entry covered by several groups: "sem projections" are the tokenizer's (sem_proj) and the core's (W3)
DOM08_ALIASES = {"sem_proj": ("sem_proj", "core.sem_proj")}
_XATTN = re.compile(r"(sem_xattn|semantic_xattn|xattn|cross_attn|crossattn)")
_GATE = re.compile(r"(gate|alpha)")


def _block_index(name: str) -> Optional[int]:
    m = re.match(r"core\.(?:[A-Za-z_]+\.)*?blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else None


# module classes whose parameters belong to one group wherever the owner attaches them (W3 Mechanism C, W4 decoder
# conditioning, W5 intent / evaluator); checked before the name rules
TYPE_RULES = {"GatedSemanticXAttn": "core.sem_xattn", "SemanticProjection": "sem_proj",
              "LatentDecoderConditioning": "dec_xattn", "LatentXAttn": "dec_xattn", "IntentHead": "intent",
              "DecisionEvaluator": "evaluator", "CandidateEvaluator": "evaluator"}


def _type_hints(model: nn.Module) -> dict:
    """{parameter qualified name: group} for parameters owned (transitively) by a module of a TYPE_RULES class."""
    out: dict = {}
    for mname, mod in model.named_modules():
        g = TYPE_RULES.get(type(mod).__name__)
        if g is None:
            continue
        for pname, _ in mod.named_parameters():
            full = f"{mname}.{pname}" if mname else pname
            out.setdefault(full, g)
    return out


def classify(name: str, n_layers: int, hint: Optional[str] = None) -> str:
    """DOM-08 group of the model parameter with qualified name `name` (see the module docstring); `hint` is the
    TYPE_RULES group of its owning module, if any."""
    leaf = ".".join(name.split(".")[-2:])            # owning submodule + tensor name, e.g. "gate_in.weight"
    if hint is not None:
        if hint == "core.sem_xattn" and _GATE.search(leaf):
            return "core.sem_xattn_gate"
        if hint == "sem_proj" and name.startswith(CORE_PREFIX):
            return "core.sem_proj"
        return hint
    if name.startswith(CORE_PREFIX):
        if name.startswith("core.sem_proj."):
            return "core.sem_proj"
        if _XATTN.search(name) or name.startswith("core.cond_proj."):
            return "core.sem_xattn_gate" if _GATE.search(leaf) else "core.sem_xattn"
        if "state_gate" in name or re.search(r"core\.gate[._]", name):
            return "core.state_gate"
        i = _block_index(name)
        if i is not None:
            if i == 0:
                return "core.block0"
            if i == n_layers // 2:
                return "core.block_mid"
            if i == n_layers - 1:
                return "core.block_last"
            return "core.blocks_other"
        return "core.embed"
    head = name.split(".", 1)[0]
    if head == "tokenizer":
        return "sem_proj"
    if head == "entity_encoder":
        return "entities"
    if head == "reasoner":
        return "reasoner"
    if head in ("beliefs", "bayes"):
        return "beliefs"
    if head == "world_model":
        return "world_model"
    if head in ("outcome_head", "query_encoder"):
        return "outcome_head"
    if "intent" in name:
        return "intent"
    if head in ("decision_evaluator", "evaluator") or ".evaluator." in name:
        return "evaluator"
    if head == "action_generator" or head == "semantic":
        if _XATTN.search(name) or "decoder_xattn" in name or "dec_cond" in name:
            return "dec_xattn"
        if ".bridge." in name or name.startswith("action_generator.bridge"):
            return "dec_bridge"
        return "policy"
    if head == "option_policy":
        return "policy"
    if head == "exploration_head":
        return "exploration"
    if head == "value_head":
        return "v_int"
    return "other"


def named_groups(model: nn.Module, reward_model: Optional[nn.Module] = None,
                 psi_params: Optional[Iterable] = None) -> "OrderedDict[str, list]":
    """{group: [(qualified name, parameter)]} over every trainable parameter (each exactly once; shared modules are
    counted at their first name). Reward-model parameters go to "psi" when they are among `psi_params` (default:
    `reward_model.meta_parameters()` when defined) and to "predictors" otherwise. Empty groups are kept (absent)."""
    n_layers = int(getattr(getattr(model, "cfg", None), "n_layers", 0) or len(getattr(model.core, "blocks", [])))
    out: "OrderedDict[str, list]" = OrderedDict((g, []) for g in GROUP_ORDER)
    seen: set = set()
    hints = _type_hints(model)
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        out[classify(n, n_layers, hints.get(n))].append((n, p))
    if reward_model is not None:
        if psi_params is None and callable(getattr(reward_model, "meta_parameters", None)):
            psi_params = list(reward_model.meta_parameters())
        psi_ids = {id(p) for p in (psi_params or [])}
        for n, p in reward_model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            out["psi" if id(p) in psi_ids else "predictors"].append((f"reward_model.{n}", p))
    return out


def core_groups(groups: dict) -> list:
    return [g for g in groups if g.startswith(CORE_PREFIX)]


@dataclass
class GradTable:
    """Per (objective, group) gradient norms of one update, plus derived core shares and cosines."""
    norms: dict                          # {objective: {group: float}}
    sizes: dict                          # {group: n parameters} (0 = absent)
    core_norm: dict                      # {objective: float} norm over the union of the core groups
    cosine: dict = field(default_factory=dict)   # {"a|b": float} on the core gradient
    cosine_exact: bool = True
    total: dict = field(default_factory=dict)    # {group: norm of the summed gradient}

    def share(self) -> dict:
        tot = sum(self.core_norm.values())
        return {k: (v / tot if tot > 0 else 0.0) for k, v in self.core_norm.items()}

    def nonzero(self, group: str) -> Optional[bool]:
        """True when some objective sent gradient into `group`, None when the group is absent."""
        if not self.sizes.get(group):
            return None
        return any(self.norms[o].get(group, 0.0) > 0 for o in self.norms)

    def to_dict(self) -> dict:
        return {"norm": {o: {g: v for g, v in d.items()} for o, d in self.norms.items()},
                "core_norm": dict(self.core_norm), "core_share": self.share(), "cosine": dict(self.cosine),
                "cosine_exact": self.cosine_exact, "total": dict(self.total),
                "sizes": dict(self.sizes),
                "absent": sorted(g for g, n in self.sizes.items() if not n),
                "nonzero": {g: self.nonzero(g) for g in self.sizes}}

    def dom08(self, required: Iterable[str] = DOM08_REQUIRED) -> dict:
        """{group: "nonzero" | "zero" | "absent"} for the DOM-08 groups."""
        out = {}
        for g in required:
            nzs = [self.nonzero(a) for a in DOM08_ALIASES.get(g, (g,))]
            nzs = [x for x in nzs if x is not None]
            out[g] = "absent" if not nzs else ("nonzero" if any(nzs) else "zero")
        return out


class _CoreSample:
    """Fixed random coordinates of the core parameters for cosine estimates on large cores (norms stay exact)."""

    def __init__(self, params: list, n: int, seed: int = 1234):
        g = torch.Generator().manual_seed(seed)
        tot = sum(p.numel() for p in params)
        self.idx = []
        for p in params:
            k = min(p.numel(), max(1, int(round(n * p.numel() / max(tot, 1)))))
            self.idx.append(torch.randperm(p.numel(), generator=g)[:k].to(p.device))


def objective_gradients(weighted: "OrderedDict[str, Tensor]", groups: dict, accumulate: bool = True,
                        cos_exact_numel: int = 20_000_000, cos_sample: int = 2_000_000,
                        _sample_cache: Optional[dict] = None, retain_graph: bool = False) -> GradTable:
    """Back-propagate each objective of `weighted` ({name: scalar tensor}, already multiplied by its loss weight)
    separately into the parameters of `groups` ({group: [(name, param)]}), accumulating into ``.grad`` when
    `accumulate` (the caller zeroes the gradients first and steps afterwards). `retain_graph` keeps the graph of the
    last objective too (a correction backward follows: the integrated learner's share guard). Returns the
    `GradTable`."""
    names = list(weighted)
    flat = [(g, p) for g, ps in groups.items() for _, p in ps]
    params = [p for _, p in flat]
    gidx = [g for g, _ in flat]
    cg = set(core_groups(groups))
    core_params = [p for g, p in flat if g in cg]
    exact = sum(p.numel() for p in core_params) <= cos_exact_numel
    samp = None
    if not exact and core_params:
        key = ("core_sample", tuple(id(p) for p in core_params), cos_sample)
        if _sample_cache is not None and key in _sample_cache:
            samp = _sample_cache[key]
        else:
            samp = _CoreSample(core_params, cos_sample)
            if _sample_cache is not None:
                _sample_cache[key] = samp
    norms = {n: {g: 0.0 for g in groups} for n in names}
    core_norm = {}
    vecs = {}
    live = [n for n in names if torch.is_tensor(weighted[n]) and weighted[n].requires_grad]
    gorder = list(groups)
    for k, n in enumerate(names):
        sq: dict = {}                    # group -> squared-norm tensor (one device sync per objective, not per tensor)
        parts = []
        if n in live:
            last = n == live[-1]
            gs = torch.autograd.grad(weighted[n], params, retain_graph=retain_graph or not last, allow_unused=True)
        else:
            gs = [None] * len(params)
        ci = 0
        for (g, p), gr in zip(flat, gs):
            if g in cg:
                if gr is None:
                    k_ = p.numel() if exact else samp.idx[ci].numel()
                    parts.append(torch.zeros(k_, device=p.device))
                else:
                    gf = gr.detach().reshape(-1).float()
                    parts.append(gf if exact else gf[samp.idx[ci]])
                ci += 1
            if gr is None:
                continue
            s = gr.detach().float().pow(2).sum()
            sq[g] = s if g not in sq else sq[g] + s
            if accumulate:
                if p.grad is None:
                    p.grad = gr.detach().clone()
                else:
                    p.grad.add_(gr.detach())
        if sq:
            vals = torch.stack([sq[g].to(params[0].device) if g in sq else torch.zeros((), device=params[0].device)
                                for g in gorder]).cpu().tolist()
        else:
            vals = [0.0] * len(gorder)
        sqf = dict(zip(gorder, vals))
        norms[n] = {g: math.sqrt(max(v, 0.0)) for g, v in sqf.items()}
        core_norm[n] = math.sqrt(max(sum(sqf[g] for g in cg), 0.0))
        vecs[n] = torch.cat(parts) if parts else torch.zeros(0)
        del gs
    cos = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            den = vecs[a].norm() * vecs[b].norm()
            cos[f"{a}|{b}"] = float((vecs[a] @ vecs[b]) / den) if vecs[a].numel() and float(den) > 0 else 0.0
    total = {}
    if accumulate:
        for g, ps in groups.items():
            gs_ = [p.grad.detach().float().pow(2).sum() for _, p in ps if p.grad is not None]
            total[g] = float(torch.stack(gs_).sum().sqrt()) if gs_ else 0.0
    sizes = {g: sum(p.numel() for _, p in ps) for g, ps in groups.items()}
    return GradTable(norms, sizes, core_norm, cos, exact, total)
