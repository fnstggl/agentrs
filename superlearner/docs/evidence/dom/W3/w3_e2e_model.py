"""W3 real end-to-end check (plan 2026-11-26 §3 W3), part 1: the real build_model(make_preset("dom_dev")) with
Qwen3-0.6B (CPU, float32 so gradient comparisons are exact to rounding).

  B. state at initialisation through the real decide (audit metric: obs A vs obs B at the first decision; and a
     6-decision episode): cos(Z, StatePos + KindEmb(STATE)) < 0.9 and observation dependence > 10%;
     the legacy core (no gate, std 0.5) on the same model for contrast.
  C. Mechanism C on the real model's own code paths: decide + policy_evaluate + world_loss with a real semantic
     context (Qwen3-0.6B token states of the observations, standardised as the tokenizer does). model.py and
     world/train_seq.py do not pass `sem` yet (owners W5 / W6; change requests filed), so this harness binds the
     semantic context at the core / reasoner call boundary for calls whose batch rows are the environment rows.
     Every cross-attention parameter must receive a nonzero gradient.
  D. A batch of B*K evaluator rows (K = 4 candidate actions per environment) through core.transition_blocks(6)
     with the semantic rows selected by code_rows; each row equals its environment's single-row result.
  E. Reasoner (fixed J = 2, and adaptive) gradient with iteration checkpointing equals the plain gradient at
     rtol 1e-4 on real tensors, with another decision (other semantic context) encoded between forward and backward.

Run from superlearner/ with PYTHONPATH=src:../crucible/src:
    CUDA_VISIBLE_DEVICES= python docs/evidence/dom/W3/w3_e2e_model.py
Writes docs/evidence/dom/W3/w3_e2e_model.json.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import subprocess
import time
from dataclasses import replace

import torch

from superlearner.cam_unified.config import make_preset
from superlearner.cam_unified.core.probes import relative_change, state_constancy
from superlearner.cam_unified.core.reasoning import Reasoner
from superlearner.cam_unified.core.semantic_xattn import SemanticContext
from superlearner.cam_unified.interfaces import (ExecutionResult, GoalSpec, Observation, ObsKind, SequenceBatch,
                                                 StepBatch, TokenSet, ToolSchema)
from superlearner.cam_unified.model import BudgetState, build_model
from superlearner.cam_unified.semantic.encoder import QwenSemantic
from superlearner.cam_unified.semantic.tokenizer import render_observation

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT: dict = {"script": "superlearner/docs/evidence/dom/W3/w3_e2e_model.py", "device": "cpu", "dtype": "float32"}
try:
    OUT["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
except Exception:                                                     # pragma: no cover
    OUT["commit"] = None
T0 = time.time()


def log(k, v):
    OUT[k] = v
    print(k, json.dumps(v, default=str)[:600], flush=True)


# ------------------------------------------------------------------ A. build
cfg = make_preset("dom_dev")
sem_model = QwenSemantic(cfg, model_name=cfg.sem_model, device="cpu", dtype=torch.float32,
                         decoder_grad_checkpoint=False)
torch.manual_seed(0)
model = build_model(cfg, components={"semantic": sem_model})
core = model.core
log("A_build", {"preset": cfg.preset, "sem_model": cfg.sem_model, "core_class": type(core).__name__,
                "reasoner_class": type(model.reasoner).__name__, "entity_encoder": type(model.entity_encoder).__name__,
                "xattn_layers": core.xattn_layers, "sem_dim": sem_model.sem_dim, "state_gate": core.state_gate is not None,
                "state_candidate": core.state_candidate, "state_emb_std": core.state_emb_std,
                "reason_fixed_iters": model.reasoner.fixed_iters, "reasoner_J_max": model.reasoner.J_max,
                "core_params": core.count_parameters(), "same_core_in_world_model": model.world_model.core is core,
                "reasoner_blocks_are_core_blocks": all(a is b for a, b in zip(model.reasoner.blocks, core.blocks[-4:])),
                "build_s": round(time.time() - T0, 1)})
assert sem_model.sem_dim == cfg.sem_xattn_dim

tools = [ToolSchema("inspect_ledger", "Read the ledger of an account and report its balance.",
                    {"type": "object", "properties": {"account": {"type": "string", "enum": ["a1", "a2", "a3"]}},
                     "required": ["account"]}),
         ToolSchema("set_price", "Set the price of a product in cents.",
                    {"type": "object", "properties": {"product": {"type": "string", "enum": ["p1", "p2"]},
                                                       "cents": {"type": "integer", "minimum": 1, "maximum": 10000}},
                     "required": ["product", "cents"]}),
         ToolSchema("advertise", "Buy an advertisement for a product.",
                    {"type": "object", "properties": {"product": {"type": "string", "enum": ["p1", "p2"]}},
                     "required": ["product"]})]
B = 2
goals = [GoalSpec(f"g{b}", "Increase monthly revenue above 5000 dollars.", {"revenue_usd": 5000}, 30.0) for b in range(B)]
tl = [list(tools) for _ in range(B)]
budget = BudgetState.make(B)


def obs_A(b, t=0.0):
    return [Observation(ObsKind.MESSAGE, f"Customer c{b + 1} complains the delivery was late (day {t:.0f}).",
                        entity=f"c{b + 1}", entity_type="customer", t=t, source="inbox"),
            Observation(ObsKind.NUMERIC, "", value=1234.0 + 100 * b + 10 * t, unit="cents", entity="p1",
                        entity_type="product", t=t, source="price")]


def obs_B(b, t=0.0):
    return [Observation(ObsKind.NUMERIC, "", value=99999.0 - 7 * t, unit="cents", entity="acct", entity_type="account",
                        t=t, source="bank"),
            Observation(ObsKind.TEXT, f"The warehouse burned down; all stock is lost (report {t:.0f}).", t=t,
                        source="news")]


def semantic_context(m, obs_lists, max_tokens: int):
    """Harness producer of the step's SemanticContext from real Qwen token states (stand-in for W4's
    ObservationTokenizer.semantic_context): the semantic observations' rendered texts, newest first, concatenated
    up to max_tokens per row, standardised like the tokenizer's pooled path; cond = mean goal token (a CAM tensor)."""
    enc = m.tokenizer.enc
    rows, masks = [], []
    for obs in obs_lists:
        texts = [render_observation(o) for o in sorted(obs, key=lambda o: -float(o.t))]
        st = enc.token_states(texts) if texts else []
        h = torch.cat([enc.standardize_states(s) for s in st], 0)[:max_tokens] if st else torch.zeros(0, enc.sem_dim)
        rows.append(h)
    T = max(1, max(r.shape[0] for r in rows))
    H = torch.zeros(len(rows), T, enc.sem_dim)
    M = torch.zeros(len(rows), T, dtype=torch.bool)
    for i, r in enumerate(rows):
        H[i, : r.shape[0]] = r
        M[i, : r.shape[0]] = True
    cond = m.tokenizer.goal(goals[: len(obs_lists)]).x.float().mean(1)
    return SemanticContext(H, M, cond)


@contextlib.contextmanager
def bind_sem(m, sem):
    """Pass `sem` to every core / reasoner call that has no semantic argument and whose batch has sem's rows (the
    environment rows). Other calls (none in dom_dev's decide / policy_evaluate / world_loss filter) run without it."""
    c, r = m.core, m.reasoner
    cf, rf = c.forward, r.forward
    n = {"core_bound": 0, "core_unbound": 0, "reasoner_bound": 0}

    def core_fwd(Z, ctx, code_rows=None, sem_=None, **kw):
        if sem_ is None and Z.shape[0] == sem.B:
            n["core_bound"] += 1
            return cf(Z, ctx, code_rows, sem, **kw)
        n["core_unbound"] += 1
        return cf(Z, ctx, code_rows, sem_, **kw)

    def rs_fwd(Z, ctx=None, n_iters=None, code_rows=None, sem_=None, **kw):
        if sem_ is None and Z.shape[0] == sem.B:
            n["reasoner_bound"] += 1
            return rf(Z, ctx, n_iters, code_rows, sem, **kw)
        return rf(Z, ctx, n_iters, code_rows, sem_, **kw)
    c.forward, r.forward = core_fwd, rs_fwd
    try:
        yield n
    finally:
        del c.forward, r.forward


def xattn_params(m):
    return {n: p for n, p in m.core.named_parameters() if n.startswith(("sem_proj.", "xattn.", "cond_proj."))}


def grad_table(m):
    out = {}
    for n, p in xattn_params(m).items():
        out[n] = 0.0 if p.grad is None else float(p.grad.norm())
    return out


def zero(m):
    m.zero_grad(set_to_none=True)
    for p in sem_model.lora_parameters():
        p.grad = None


# ------------------------------------------------------------------ B. state at initialisation (real decide)
def init_diagnostics(m, tag):
    st0 = m.init_state(B, seed=1)
    with torch.no_grad():
        dA = m.decide(copy.deepcopy(st0), [obs_A(b) for b in range(B)], [0.0] * B, tl, goals, budget, greedy=True)
        dB = m.decide(copy.deepcopy(st0), [obs_B(b) for b in range(B)], [0.0] * B, tl, goals, budget, greedy=True)
        sa, sb = m.init_state(B, seed=1), m.init_state(B, seed=1)
        cos, per = [], []
        for t in range(6):
            Zprev = sa.Z.clone()
            alt = copy.deepcopy(sa)
            da = m.decide(sa, [obs_A(b, t) for b in range(B)], [float(t)] * B, tl, goals, budget, greedy=True)
            dalt = m.decide(alt, [obs_B(b, t) for b in range(B)], [float(t)] * B, tl, goals, budget, greedy=True)
            m.decide(sb, [obs_B(b, t) for b in range(B)], [float(t)] * B, tl, goals, budget, greedy=True)
            per.append(round(relative_change(da.Z, dalt.Z), 4))
            cos.append(round(float(state_constancy(m.core, da.Z).max()), 4))
            del Zprev
        hist = relative_change(sa.Z, sb.Z)
    res = {"first_decision_obs_dependence": round(relative_change(dA.Z, dB.Z), 4),
           "first_decision_cos_const": [round(float(x), 4) for x in state_constancy(m.core, dA.Z)],
           "R_obs_dependence": round(relative_change(dA.R, dB.R), 4),
           "per_decision_obs_dependence_t0_5": per, "max_cos_const_t0_5": cos,
           "history_obs_dependence_6_decisions": round(hist, 4),
           "rms_Z_after_6": round(float(sa.Z.pow(2).mean().sqrt()), 4),
           "tools_A": [a.tool for a in dA.actions], "tools_B": [a.tool for a in dB.actions]}
    log(f"B_init_{tag}", res)
    return res


model.eval()
b_dom = init_diagnostics(model, "dom_dev")
cfg_leg = replace(cfg, state_gate=False, state_emb_std=0.5, sem_xattn_every=0)
torch.manual_seed(0)
legacy = build_model(cfg_leg, components={"semantic": sem_model})
legacy.eval()
b_leg = init_diagnostics(legacy, "legacy_core_same_preset")
del legacy
OUT["B_pass"] = bool(max(b_dom["max_cos_const_t0_5"] + b_dom["first_decision_cos_const"]) < 0.9
                     and b_dom["first_decision_obs_dependence"] > 0.10
                     and b_dom["history_obs_dependence_6_decisions"] > 0.10)
print("B_pass", OUT["B_pass"], flush=True)

# ------------------------------------------------------------------ C. Mechanism C gradients on the real paths
obs0 = [obs_A(b) for b in range(B)]
semA = semantic_context(model, obs0, cfg.sem_xattn_max_tokens)
log("C_sem_context", {"tokens_per_row": semA.mask.sum(1).tolist(), "sem_dim": semA.h.shape[-1]})
model.eval()
with bind_sem(model, semA) as nb:
    st = model.init_state(B, seed=2)
    d = model.decide(st, obs0, [0.0] * B, tl, goals, budget, greedy=False, generator=torch.Generator().manual_seed(0))
    gates_decide = core.gate_stats()
    model.train()
    zero(model)
    pe = model.policy_evaluate(d.inputs, d.actions)
    (-pe["logp"].mean() + (pe["value"] - 1.0).pow(2).mean()).backward()
    g_policy = grad_table(model)
    calls_policy = dict(nb)
with torch.no_grad():
    plain_st = model.init_state(B, seed=2)
    model.eval()
    d_plain = model.decide(plain_st, obs0, [0.0] * B, tl, goals, budget, greedy=True)
    with bind_sem(model, semA):
        d_sem = model.decide(model.init_state(B, seed=2), obs0, [0.0] * B, tl, goals, budget, greedy=True)
log("C_sem_changes_Z_rel", round(relative_change(d_sem.Z, d_plain.Z), 5))

# world loss on a real 3-step SequenceBatch built from real decisions and observe_outcomes (as the audit did)
model.eval()
state = model.init_state(B, seed=3)
steps, cur = [], obs0
with bind_sem(model, semA):
    for t in range(3):
        regs_before = copy.deepcopy(state.registries)
        dd = model.decide(state, cur, [float(t)] * B, tl, goals, budget, greedy=False,
                          generator=torch.Generator().manual_seed(t))
        results = [ExecutionResult(True, [Observation(ObsKind.TOOL_RESULT, f"ok step {t} row {b}", t=float(t) + 0.1,
                                                      source=dd.actions[b].tool)], cost=float(10 * (b + 1)),
                                   duration=0.01) for b in range(B)]
        nxt = [obs_A(b, t + 1.0) for b in range(B)]
        oo = model.observe_outcomes(state, dd, results, nxt, [0.0] * B)
        O = model.tokenizer.observations(cur, regs_before, [float(t)] * B)
        if model.entity_encoder is not None:
            O = TokenSet.cat(O, model.entity_encoder(O))
        steps.append(StepBatch(obs=O, act=model.tokenizer.action(dd.actions), outcomes=oo["outcomes"],
                               dt=torch.full((B,), 1.0), reward=torch.zeros(B), done=torch.zeros(B, dtype=torch.bool),
                               behavior_logp=dd.logp, model_version=torch.zeros(B, dtype=torch.long),
                               goal=model.tokenizer.goal(goals)))
        cur = nxt
seq = SequenceBatch(steps)
model.train()
zero(model)
semW = semantic_context(model, obs0, cfg.sem_xattn_max_tokens)      # fresh graph for cond (one backward per graph)
with bind_sem(model, semW) as nw:
    filt = model.filter_sequence(seq)
    wl = model.world_loss(seq, filt)
    tot = wl.get("loss", wl.get("world"))
    tot.backward()
g_world = grad_table(model)
names = list(xattn_params(model))
nz_policy = [n for n in names if g_policy[n] > 0]
nz_world = [n for n in names if g_world[n] > 0]
union = sorted(set(nz_policy) | set(nz_world))
log("C_mechanism_c_gradients", {
    "n_xattn_params": len(names), "nonzero_from_policy_evaluate": len(nz_policy),
    "nonzero_from_world_loss": len(nz_world), "nonzero_union": len(union),
    "zero_in_both": sorted(set(names) - set(union)), "world_loss": float(tot.detach()),
    "bound_calls_decide_policy": calls_policy, "bound_calls_world": dict(nw),
    "world_loss_terms": {k: round(float(v), 4) for k, v in wl.items() if torch.is_tensor(v) and v.numel() == 1},
    "gates_at_decide": gates_decide, "reasoner_gate_iters": sorted(model.reasoner.last_gates),
    "grad_norm_policy_sem_proj": g_policy.get("sem_proj.proj.weight"),
    "grad_norm_world_sem_proj": g_world.get("sem_proj.proj.weight")})
OUT["C_pass"] = len(union) == len(names)
# scale check for W6: the same token sequence through the legacy core's world model (no gate, unit-RMS Z)
torch.manual_seed(0)
legacy = build_model(cfg_leg, components={"semantic": sem_model})
legacy.train()
with torch.no_grad():
    wl_leg = legacy.world_loss(seq, legacy.filter_sequence(seq))
log("C_world_loss_terms_legacy_core_same_tokens",
    {k: round(float(v), 4) for k, v in wl_leg.items() if torch.is_tensor(v) and v.numel() == 1})
del legacy
print("C_pass", OUT["C_pass"], flush=True)

# ------------------------------------------------------------------ D. B*K evaluator rows through transition_blocks(6)
model.eval()
K = 4
with torch.no_grad():
    st = model.init_state(B, seed=4)
    dK = [model.decide(copy.deepcopy(st), obs0, [0.0] * B, tl, goals, budget, greedy=False,
                       generator=torch.Generator().manual_seed(10 + k)) for k in range(K)]
    R = dK[0].R
    acts = [model.tokenizer.action([dK[k].actions[b] for b in range(B)]) for k in range(K)]
    rows = torch.arange(B).repeat_interleave(K)
    cand = [[dK[k].actions[b] for k in range(K)] for b in range(B)]
    flat = [a for b in range(B) for a in cand[b]]
    A_tok = model.tokenizer.action(flat)                                      # [B*K] candidate action tokens
    belief = model.beliefs.tokens(st.fast).select(rows.tolist()) if st.fast is not None else None
    ctx = TokenSet.cat(A_tok, belief) if belief is not None else A_tok
    t1 = time.time()
    Zhat, _ = core(R[rows], ctx, code_rows=rows, sem=semA, blocks=core.transition_blocks(6))
    dt_bk = time.time() - t1
    single = []
    for k in range(K):
        idx = torch.arange(B) * K + k
        cs = TokenSet.cat(A_tok.select(idx.tolist()), None if belief is None else belief.select(idx.tolist())) \
            if belief is not None else A_tok.select(idx.tolist())
        z1, _ = core(R, cs, sem=semA, blocks=core.transition_blocks(6))
        single.append(z1)
    ref = torch.stack(single, 1).reshape(B * K, *Zhat.shape[1:])
log("D_evaluator_rows", {"rows": int(Zhat.shape[0]), "B": B, "K": K, "shape": list(Zhat.shape),
                         "max_abs_diff_vs_per_env": float((Zhat - ref).abs().max()), "seconds": round(dt_bk, 3),
                         "transition_block_ids": core.block_ids(core.transition_blocks(6)),
                         "xattn_gates_last_call": core.gate_stats(), "finite": bool(torch.isfinite(Zhat).all())})
OUT["D_pass"] = bool(Zhat.shape[0] == B * K and torch.isfinite(Zhat).all() and float((Zhat - ref).abs().max()) < 1e-4)

# ------------------------------------------------------------------ E. reasoner checkpointed gradient equality
semB = semantic_context(model, [obs_B(b) for b in range(B)], cfg.sem_xattn_max_tokens)
model.train()
with torch.no_grad():
    st = model.init_state(B, seed=5)
    model.eval()
    dE = model.decide(st, obs0, [0.0] * B, tl, goals, budget, greedy=True)
Zr = dE.Z.detach()
rctx = TokenSet.cat(model.beliefs.tokens(st.fast), model.tokenizer.goal(goals))
rctx = TokenSet(rctx.x.detach(), rctx.mask, rctx.kind, rctx.time, rctx.entity, rctx.refs, rctx.entity2)
res_E = {}
W_E = torch.randn(Zr.shape[1:], generator=torch.Generator().manual_seed(7))   # fixed readout (R is out_norm'd)
for tag, rs in (("fixed_J2", model.reasoner), ("adaptive", Reasoner(replace(cfg, reason_iters_fixed=0), core))):
    rs.train()
    core.train()
    params = [p for p in list(core.parameters()) + list(rs.parameters()) if p.requires_grad]
    pn = [n for n, p in list(core.named_parameters()) + list(rs.named_parameters()) if p.requires_grad]
    outs = {}
    for ck in (False, True):
        rs.checkpoint_iters = ck
        for p in params:
            p.grad = None
        semE = semantic_context(model, obs0, cfg.sem_xattn_max_tokens)  # same values, fresh cond graph
        R, info = rs(Zr, rctx, sem=semE)
        # another decision with another semantic context encoded between forward and backward
        with bind_sem(model, semB):
            model.eval()
            with torch.no_grad():
                model.decide(model.init_state(B, seed=6), [obs_B(b) for b in range(B)], [0.0] * B, tl, goals,
                             budget, greedy=True)
            model.train()
        (R * W_E).sum().div(R.shape[0]).add(info["ponder"].mean()).backward()
        outs[ck] = (R.detach(), [None if p.grad is None else p.grad.clone() for p in params], info["n_computed"])
    worst, n_cmp, bad = 0.0, 0, []
    for n, a, b in zip(pn, outs[False][1], outs[True][1]):
        if (a is None) != (b is None):
            bad.append(n)
            continue
        if a is None:
            continue
        n_cmp += 1
        rel = float((a - b).abs().max() / a.abs().max().clamp_min(1e-12))
        worst = max(worst, rel)
        if not torch.allclose(a, b, rtol=1e-4, atol=1e-8):
            bad.append(n)
    res_E[tag] = {"iters": outs[True][2].tolist(), "R_max_abs_diff": float((outs[False][0] - outs[True][0]).abs().max()),
                  "params_compared": n_cmp, "max_rel_grad_diff": worst, "mismatched": bad[:10],
                  "sem_proj_grad_norm": float(dict(zip(pn, outs[True][1]))["sem_proj.proj.weight"].norm())}
log("E_reasoner_checkpoint", res_E)
OUT["E_pass"] = all(not v["mismatched"] and v["R_max_abs_diff"] < 1e-5 for v in res_E.values())
OUT["seconds"] = round(time.time() - T0, 1)
print("PASS", {k: OUT[k] for k in ("B_pass", "C_pass", "D_pass", "E_pass")}, flush=True)
json.dump(OUT, open(os.path.join(HERE, "w3_e2e_model.json"), "w"), indent=1, default=str)
