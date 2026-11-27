"""Mechanism B acceptance on the REAL model (plan 2026-11-26 §3 W5 "Real end-to-end check"; DOM Condition G).

A two-goal decision task: every row sees the same observations and the same four tools (in a per-episode random
order); the goal text alone says which tool is right ("restock" -> order_stock, "invoice" -> send_invoice), so the
decision must carry the goal from the raw goal tokens through Z_t / R_t / ctx_out / I_t into the tool pointer.

* `imitate`               short behaviour cloning of the goal -> tool mapping (declared condition
                          "demo-init:two_goal_oracle"; the demonstrator reads only the agent-visible goal text) through
                          `policy_evaluate` (generator log-probability in intent mode; core, reasoner, intent head,
                          pointer and keys all trained)
* `zero_intent_check`     I_t = 0 -> the tool distribution is identical for both goals (closed by construction)
* `condition_g`           swap (R_t, I_t) between rows with different goals (observations held constant): the fraction
                          of executed (selected, stamped) actions that follow the DONOR's goal
* `kl_under_permutation`  mean KL(pi_tool(. | I_b) || pi_tool(. | I_perm(b))) over rows (a lower bound on the action
                          KL: the tool is a marginal of the action)
* `replay_check`          the recorded behaviour log-probability (generator + selection + option) equals
                          policy_evaluate's recomputation (same batch and a permuted half batch)
* `stamp_check`           every executed action passes decision.select.authorize; unstamped / tampered are refused

`python -m superlearner.cam_unified.decision.condition_g --out <json>` runs everything on the real dom_dev stack
(Qwen3-0.6B, 8 x 384 core) and writes the measured numbers.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import subprocess
import time
from typing import Optional, Sequence

import torch

from ..capabilities.envs import DiagnosticEnv, tool
from ..interfaces import Action, GoalSpec, Observation, ObsKind

SEL = importlib.import_module(__package__ + ".select")      # the module (the package re-exports the function)

GOALS = (("restock", "The shelves are empty. Get more units of our product into stock."),
         ("invoice", "A customer still owes us for last week's delivery. Get the money we are owed."))
ANSWER = {"restock": "order_stock", "invoice": "send_invoice"}
TOOLS = (("order_stock", "Order more units of the product from the supplier."),
         ("send_invoice", "Send a customer a bill for goods that were already delivered."),
         ("post_advert", "Publish an advertisement for the shop."),
         ("clean_shop", "Clean the shop floor."))


class TwoGoalEnv(DiagnosticEnv):
    """One decision per episode; the goal (one of two) decides the correct tool; observations are identical."""

    name = "twogoal"
    horizon = 1

    def __init__(self, B: int, seed: int = 0, goals: Optional[Sequence[int]] = None):
        self.fixed_goals = None if goals is None else list(goals)
        super().__init__(B, seed)

    def _new(self, rng, b):
        g = self.fixed_goals[b] if self.fixed_goals is not None else rng.randrange(2)
        order = list(range(len(TOOLS)))
        rng.shuffle(order)
        return {"goal": g, "order": order}

    def _tools(self, h):
        return [tool(*TOOLS[i]) for i in h["order"]]

    def _obs0(self, h, b):
        return [Observation(ObsKind.TEXT, "Daily report: the shop is open, staffed and quiet.", source="shop")]

    def _goal(self, h, b):
        k, text = GOALS[h["goal"]]
        return GoalSpec(f"twogoal_{k}", text)

    def _act(self, h, a, rng, b):
        r = 1.0 if a.tool == ANSWER[GOALS[h["goal"]][0]] else 0.0
        return True, [Observation(ObsKind.TEXT, "Done.", source="shop")], r, None

    def answer(self, b: int) -> str:
        return ANSWER[GOALS[self.hidden[b]["goal"]][0]]


def _decide(model, env: TwoGoalEnv, greedy: bool = False, generator=None):
    ob = env.reset(env.seeds)
    state = model.init_state(env.B)
    return model.decide(state, ob.observations, ob.t, env.available_tools(), env.goals(), greedy=greedy,
                        generator=generator)


def _new_env(B: int, seed: int, goals=None) -> TwoGoalEnv:
    e = TwoGoalEnv(B, seed=seed, goals=goals)
    e.reset([seed * 100_003 + b for b in range(B)])
    return e


def make_optimizer(model, lr_core: float = 1e-4, lr_heads: float = 3e-4):
    core_ids = {id(p) for p in model.core.parameters()}
    core, heads, seen = [], [], set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (core if id(p) in core_ids else heads).append(p)
    return torch.optim.AdamW([{"params": core, "lr": lr_core}, {"params": heads, "lr": lr_heads}],
                             weight_decay=0.0)


def imitate(model, steps: int = 120, B: int = 16, seed: int = 0, lr_core: float = 1e-4, lr_heads: float = 3e-4,
            log_every: int = 20) -> dict:
    """Behaviour cloning of the goal -> tool mapping through the DOM policy path (gen log-probability)."""
    opt = make_optimizer(model, lr_core, lr_heads)
    hist = []
    t0 = time.perf_counter()
    for it in range(steps):
        env = _new_env(B, seed * 7919 + it)
        dec = _decide(model, env, greedy=True)
        demo = [Action(tool=env.answer(b), args={}) for b in range(B)]
        model.train()
        out = model.policy_evaluate(dec.inputs, demo)
        loss = -out["gen_logp"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        model.version = model.version + 1
        if it % log_every == 0 or it == steps - 1:
            p = out["gen_logp"].detach().exp().mean()
            hist.append({"step": it, "loss": float(loss.detach()), "p_correct": float(p), "grad_norm": float(gn)})
    model.eval()
    return {"steps": steps, "B": B, "history": hist, "seconds": time.perf_counter() - t0}


def tool_probs(model, dec, intent_fn=None) -> tuple:
    lp, allowed, lists2, I = model.tool_distribution(dec.inputs, intent_fn)
    return lp, lists2, I


def _named(lp_row, names) -> dict:
    return {n: float(lp_row[j].exp()) for j, n in enumerate(names)}


def zero_intent_check(model, B: int = 8, seed: int = 101) -> dict:
    goals = [b % 2 for b in range(B)]
    env = _new_env(B, seed, goals)
    dec = _decide(model, env, greedy=True)
    lp0, lists2, _ = tool_probs(model, dec, lambda I: torch.zeros_like(I))
    lp, _, _ = tool_probs(model, dec)
    by_goal0, by_goal = {0: [], 1: []}, {0: [], 1: []}
    for b in range(B):
        names = [t.name for t in lists2[b]]
        by_goal0[goals[b]].append(_named(lp0[b], names))
        by_goal[goals[b]].append(_named(lp[b], names))
    names = sorted(by_goal0[0][0])

    def max_gap(d):
        return max(abs(r0[n] - r1[n]) for r0 in d[0] for r1 in d[1] for n in names)
    return {"max_prob_gap_between_goals_I_zero": max_gap(by_goal0),
            "max_prob_gap_between_goals_trained_I": max_gap(by_goal),
            "uniform_at_I_zero": max(abs(v - 1.0 / len(names)) for r in by_goal0[0] + by_goal0[1]
                                     for v in r.values()),
            "n_tools_incl_builtins": len(names)}


def condition_g(model, B: int = 16, seed: int = 202, generator=None) -> dict:
    """Swap (R, I) between paired rows of different goals; fraction of executed actions following the donor."""
    goals = [b % 2 for b in range(B)]
    perm = torch.tensor([b + 1 if b % 2 == 0 else b - 1 for b in range(B)])
    env = _new_env(B, seed, goals)
    base = _decide(model, env, generator=generator)
    own = sum(int(base.actions[b].tool == env.answer(b)) for b in range(B))
    env2 = _new_env(B, seed, goals)
    dev = model.device
    p = perm.to(dev)
    with model.latent_intervention(lambda R, I: (R[p], I[p])):
        sw = _decide(model, env2, generator=generator)
    donor = sum(int(sw.actions[b].tool == env.answer(int(perm[b]))) for b in range(B))
    kept_own = sum(int(sw.actions[b].tool == env.answer(b)) for b in range(B))
    stamped = all(SEL.verify_stamp(a)[0] for a in sw.actions)
    # I-only swap (R kept): the generator and the evaluator's intent inputs follow the donor; R feeds the prior
    env3 = _new_env(B, seed, goals)
    with model.latent_intervention(lambda R, I: (R, I[p])):
        si = _decide(model, env3, generator=generator)
    donor_i = sum(int(si.actions[b].tool == env.answer(int(perm[b]))) for b in range(B))
    # R-only swap after the intent (diagnostic: R reaches the action only through I and the evaluator)
    env4 = _new_env(B, seed, goals)
    with model.latent_intervention(lambda R, I: (R[p], I)):
        sr = _decide(model, env4, generator=generator)
    own_r = sum(int(sr.actions[b].tool == env.answer(b)) for b in range(B))
    # R swapped BEFORE the intent head reads it (I = Intent(R_donor, ctx_out_own)): how much of the goal the
    # intent takes from R_t rather than from the core-processed context ctx_out
    env5 = _new_env(B, seed, goals)
    with model.latent_intervention(lambda R: R[p], before_intent=True):
        sp = _decide(model, env5, generator=generator)
    donor_pre = sum(int(sp.actions[b].tool == env.answer(int(perm[b]))) for b in range(B))
    return {"B": B, "no_swap_follow_own_goal": own / B, "swap_RI_follow_donor": donor / B,
            "swap_RI_follow_own": kept_own / B, "swap_I_only_follow_donor": donor_i / B,
            "swap_R_only_after_intent_follow_own": own_r / B,
            "swap_R_before_intent_follow_donor": donor_pre / B,
            "all_swapped_actions_stamped": bool(stamped),
            "swapped_tools": [a.tool for a in sw.actions], "donor_answers": [env.answer(int(perm[b]))
                                                                            for b in range(B)]}


def kl_under_permutation(model, B: int = 16, seed: int = 303) -> dict:
    goals = [b % 2 for b in range(B)]
    perm = torch.tensor([b + 1 if b % 2 == 0 else b - 1 for b in range(B)]).to(model.device)
    env = _new_env(B, seed, goals)
    dec = _decide(model, env, greedy=True)
    lp, lists2, _ = tool_probs(model, dec)
    lpp, _, _ = tool_probs(model, dec, lambda I: I[perm])
    p = lp.exp()
    m = torch.isfinite(lp) & torch.isfinite(lpp)
    kl = (p * (lp - lpp)).masked_fill(~m, 0.0).sum(-1)
    return {"B": B, "kl_tool_mean": float(kl.mean()), "kl_tool_min": float(kl.min()), "kl_tool_max": float(kl.max())}


def replay_check(model, B: int = 8, seed: int = 404, generator=None) -> dict:
    env = _new_env(B, seed)
    dec = _decide(model, env, generator=generator)
    with torch.no_grad():
        out = model.policy_evaluate(dec.inputs, dec.actions)
    rec = dec.logp.float().cpu()
    diff = (out["logp"].float().cpu() - rec).abs()
    gen_rec = torch.tensor([a.logp_parts["gen_logp"] for a in dec.actions])
    sel_rec = torch.tensor([a.logp_parts["selection"] for a in dec.actions])
    js = list(range(B))[::2][::-1]
    sub = dec.inputs.select(js)
    with torch.no_grad():
        out2 = model.policy_evaluate(sub, [dec.actions[j] for j in js])
    diff2 = (out2["logp"].float().cpu() - rec[js]).abs()
    # with gradients (the learner's mode)
    model.train()
    out3 = model.policy_evaluate(dec.inputs, dec.actions)
    model.eval()
    diff3 = (out3["logp"].detach().float().cpu() - rec).abs()
    return {"B": B, "K": int(dec.evaluation[0]["n"]),
            "max_abs_diff_same_batch": float(diff.max()),
            "max_abs_diff_gen": float((out["gen_logp"].float().cpu() - gen_rec).abs().max()),
            "max_abs_diff_sel": float((out["sel_logp"].float().cpu() - sel_rec).abs().max()),
            "max_abs_diff_permuted_half_batch": float(diff2.max()),
            "max_abs_diff_grad_mode": float(diff3.max()),
            "bit_identical_same_batch": bool(torch.equal(out["logp"].float().cpu(), rec)),
            "sel_logp_nonzero": bool((sel_rec.abs() > 0).any())}


def stamp_check(model, B: int = 4, seed: int = 505) -> dict:
    env = _new_env(B, seed)
    dec = _decide(model, env)
    out = {"executed_all_authorized": True, "unstamped_refused": False, "tampered_refused": False,
           "forged_refused": False}
    SEL.authorize(dec.actions)
    env.execute(dec.actions)
    try:
        SEL.authorize([Action(tool=dec.actions[0].tool, args={})])
    except SEL.UnauthorizedAction:
        out["unstamped_refused"] = True
    t = Action.from_dict(dec.actions[0].to_dict())
    t.args = {"injected": 1}
    try:
        SEL.require_stamp(t)
    except SEL.UnauthorizedAction:
        out["tampered_refused"] = True
    f = Action.from_dict(dec.actions[0].to_dict())
    SEL.stamp_of(f)["digest"] = "0" * 64
    try:
        SEL.require_stamp(f)
    except SEL.UnauthorizedAction:
        out["forged_refused"] = True
    return out


def candidate_set_check(model, B: int = 8, seed: int = 515, generator=None) -> dict:
    """The recorded candidate set of DOM decisions (W8 credit): emb [B,K,D], mask, Q, behaviour selection log-probs,
    chosen slot; the stamp in Action.proposal["cam_eval"] (v4 slot); recomputed from archived actions."""
    env = _new_env(B, seed)
    dec = _decide(model, env, generator=generator)
    cs = dec.candidate_set
    sel = torch.tensor([a.logp_parts["selection"] for a in dec.actions], device=cs.sel_logp.device).float()
    at_chosen = cs.sel_logp.gather(1, cs.chosen.view(-1, 1)).squeeze(1)
    acts = [Action.from_dict(a.to_dict()) for a in dec.actions]
    cs2 = model.candidate_set_of(acts, dec.inputs.tools)
    return {"B": B, "emb_shape": list(cs.emb.shape), "mask_all": bool(cs.mask.all()),
            "emb_abs_mean": float(cs.emb.abs().mean()),
            "sel_logp_at_chosen_max_abs_diff": float((at_chosen - sel).abs().max()),
            "sel_prob_row_sum_max_dev": float((cs.sel_logp.exp().sum(1) - 1).abs().max()),
            "stamp_in_proposal_v4_slot": sum(int(isinstance((a.proposal or {}).get(SEL.STAMP_KEY), dict))
                                             for a in dec.actions),
            "stamp_in_logp_parts": sum(int(SEL.STAMP_KEY in (a.logp_parts or {})) for a in dec.actions),
            "recomputed_from_archive": {"chosen_equal": bool(torch.equal(cs2.chosen, cs.chosen)),
                                        "sel_logp_equal": bool(torch.equal(cs2.sel_logp, cs.sel_logp)),
                                        "emb_max_abs_diff": float((cs2.emb - cs.emb).abs().max())}}


def decide_timing(model, B: int = 16, reps: int = 3, seed: int = 707, generator=None) -> dict:
    """Wall seconds of a DOM decide (K = cfg.n_candidates, no-argument tools + the built-ins), median of reps."""
    out = []
    for r in range(reps):
        env = _new_env(B, seed + r)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec = _decide(model, env, generator=generator)
        out.append(dec.timings)
    med = {k: sorted(t[k] for t in out)[len(out) // 2] for k in out[0]}
    return {"B": B, "K": int(model.cfg.n_candidates), "median": med, "all": out}


def accuracy(model, B: int = 32, seed: int = 606, greedy: bool = False, generator=None) -> float:
    env = _new_env(B, seed)
    dec = _decide(model, env, greedy=greedy, generator=generator)
    SEL.authorize(dec.actions)
    return sum(int(dec.actions[b].tool == env.answer(b)) for b in range(B)) / B


def ow_tiny_string_check(model, B: int = 4, trials: int = 4, seed: int = 0) -> dict:
    """DOM decisions over the REAL Crucible ow_tiny tool catalogue (string arguments: Qwen holes decoded from the
    bridge prefix of I_t): every action stamped, string holes decoded, behaviour logp replayed exactly."""
    import crucible
    crucible.load_all_modules()
    from crucible.api import WorldServer
    from crucible.worldgen import generate
    from ..interfaces import ToolSchema
    srv = WorldServer("memory://", lease_ttl_s=None)
    wid = srv.create_world(generate({"family": "ow_tiny"}, 1))
    me = srv.join_world(wid, "nova", endowment=3_000_000)
    tools = [ToolSchema(t["name"], t.get("description", ""), t["inputSchema"], cost_hint=None,
                        duration_hint=(t.get("annotations") or {}).get("costSeconds", 0) / 86400.0)
             for t in srv.available_tools(wid, me)]
    obs = [[Observation(ObsKind.TEXT, "Welcome to the market. You have 30000 dollars of funds.", source="world"),
            Observation(ObsKind.NUMERIC, "funds", value=3_000_000.0, unit="cents", source="bank")] for _ in range(B)]
    goals = [GoalSpec("ow", "Build a profitable business in this world.") for _ in range(B)]
    g = torch.Generator(device=str(model.device)).manual_seed(seed)
    holes, diffs, secs, tool_names, unauthorized = 0, [], [], [], 0
    for _ in range(trials):
        state = model.init_state(B)
        t0 = time.perf_counter()
        dec = model.decide(state, obs, [0.0] * B, [list(tools)] * B, goals, generator=g)
        secs.append(time.perf_counter() - t0)
        try:
            SEL.authorize(dec.actions)
        except SEL.UnauthorizedAction:
            unauthorized += 1
        for a in dec.actions:
            tool_names.append(a.tool)
            for c in SEL.stamp_of(a)["candidates"]:
                holes += len((c["logp_parts"].get("decode") or {}).get("holes") or [])
        with torch.no_grad():
            out = model.policy_evaluate(dec.inputs, dec.actions)
        diffs.append(float((out["logp"].float().cpu() - dec.logp.float().cpu()).abs().max()))
    return {"n_tools": len(tools), "B": B, "K": int(model.cfg.n_candidates), "trials": trials,
            "string_holes_decoded_in_candidates": holes, "replay_max_abs_diff": diffs, "decide_s": secs,
            "executed_tools": tool_names, "unauthorized_batches": unauthorized,
            "decoder_cond_api": {"sample": model.action_generator._cond_sample,
                                 "score": model.action_generator._cond_score}}


def run(out_path: Optional[str] = None, steps: int = 120, seed: int = 0, device: Optional[str] = None) -> dict:
    from ..capabilities.stack import build_stack
    from ..config import make_preset
    torch.manual_seed(seed)
    random.seed(seed)
    cfg = make_preset("dom_dev", declared_conditions=("demo-init:two_goal_oracle",))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    st = build_stack(cfg, device=dev, include_builtins=True, exact_rows=True, seed=seed)
    m = st.model
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    g = torch.Generator(device=dev).manual_seed(seed + 17)
    res = {"preset": "dom_dev", "sem_model": cfg.sem_model, "device": dev, "seed": seed,
           "build_s": time.perf_counter() - t0, "n_intent": cfg.n_intent, "n_candidates": cfg.n_candidates,
           "params": {k: v["trainable"] for k, v in m.count_parameters()["components"].items()},
           "real_components": {k: v for k, v in m.build_report.items()}}
    res["commit"] = _commit()
    res["init"] = {"accuracy_sampled": accuracy(m, generator=g), "zero_intent": zero_intent_check(m),
                   "kl": kl_under_permutation(m), "replay": replay_check(m, generator=g)}
    res["imitation"] = imitate(m, steps=steps, seed=seed)
    res["trained"] = {"accuracy_sampled": accuracy(m, generator=g), "accuracy_greedy": accuracy(m, greedy=True),
                      "zero_intent": zero_intent_check(m), "kl": kl_under_permutation(m),
                      "condition_g": condition_g(m, generator=g), "replay": replay_check(m, generator=g),
                      "stamps": stamp_check(m), "candidate_set": candidate_set_check(m, generator=g)}
    from .fast import fast_param_count
    res["fast_set"] = fast_param_count(m)
    res["decide_timing_B16_K4"] = decide_timing(m, B=16, generator=g)
    if torch.cuda.is_available():
        res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    res["total_s"] = time.perf_counter() - t0
    tr = res["trained"]
    res["checks"] = {
        "zero_intent_goal_independent": tr["zero_intent"]["max_prob_gap_between_goals_I_zero"] < 1e-6,
        "condition_g_follow_donor_ge_0.9": tr["condition_g"]["swap_RI_follow_donor"] >= 0.9,
        "kl_permuted_gt_0.1": tr["kl"]["kl_tool_mean"] > 0.1,
        "replay_exact_same_batch": tr["replay"]["max_abs_diff_same_batch"] == 0.0,
        "no_execution_without_stamp": all(tr["stamps"].values()),
        "candidate_set_recorded_and_recomputable": (
            tr["candidate_set"]["mask_all"] and tr["candidate_set"]["sel_logp_at_chosen_max_abs_diff"] < 1e-5
            and tr["candidate_set"]["recomputed_from_archive"]["chosen_equal"]
            and tr["candidate_set"]["recomputed_from_archive"]["sel_logp_equal"]
            and tr["candidate_set"]["stamp_in_proposal_v4_slot"] == tr["candidate_set"]["B"]),
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1, default=str)
    return res


def agent_loop_check(agent, rounds: int = 24, B: int = 8, seed: int = 3, update_every: int = 8) -> dict:
    """The real CamUnifiedAgent (+ its trainer) acting through runtime.run_steps on the two-goal task, with the
    environment's execute wrapped by the execution gate (decision.select.authorize): counts executed actions and
    unstamped ones, and the learner updates that consumed DOM decisions (policy_evaluate on stamped actions)."""
    from .. import runtime
    from ..capabilities.envs import ScoreEvaluator
    env = TwoGoalEnv(B, seed=seed)
    seen = {"executed": 0, "unstamped": 0, "stamp_in_proposal_v4_slot": 0}
    orig = env.execute

    def gated(actions):
        for a in actions:
            if a is not None:
                seen["executed"] += 1
                seen["unstamped"] += int(not SEL.is_stamped(a))
                seen["stamp_in_proposal_v4_slot"] += int(isinstance((a.proposal or {}).get(SEL.STAMP_KEY), dict))
        SEL.authorize(actions)
        return orig(actions)
    env.execute = gated
    t0 = time.perf_counter()
    rep = runtime.run_steps(agent, env, rounds, evaluator=ScoreEvaluator(env), update_every=update_every)
    ups = [u for u in (rep.updates or []) if isinstance(u, dict)]
    return {"rounds": rounds, "B": B, **seen, "updates": len(ups),
            "updated": sum(int(bool(u.get("updated"))) for u in ups),
            "losses_last_update": ups[-1].get("losses") if ups else None,
            "returns": [r if isinstance(r, (int, float)) else str(r) for r in (rep.returns or [])],
            "seconds": time.perf_counter() - t0}


def _commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(__file__)).stdout.strip() or None
    except OSError:
        return None


def run_ow_tiny(out_path: Optional[str] = None, seed: int = 0, device: Optional[str] = None) -> dict:
    from ..capabilities.stack import build_stack
    from ..config import make_preset
    from ..semantic.grammar import GrammarOptions
    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    st = build_stack(make_preset("dom_dev"), device=dev, include_builtins=True, exact_rows=True, seed=seed,
                     grammar_opts=GrammarOptions(max_string_tokens=24, max_action_tokens=96, max_key_tokens=16))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    res = {"preset": "dom_dev", "commit": _commit(), "device": dev, **ow_tiny_string_check(st.model, seed=seed)}
    try:
        res["agent_loop"] = agent_loop_check(st.agent)
    except Exception as e:                     # reported, never hidden: the canonical loop is W1's
        res["agent_loop"] = {"error": f"{type(e).__name__}: {e}"}
    if torch.cuda.is_available():
        res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1, default=str)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ow-tiny", action="store_true", help="only the ow_tiny string-tool check (untrained dom_dev)")
    a = ap.parse_args(argv)
    if a.ow_tiny:
        res = run_ow_tiny(a.out, a.seed, a.device)
        print(json.dumps({k: v for k, v in res.items() if k != "executed_tools"}, indent=1, default=str))
        return
    res = run(a.out, a.steps, a.seed, a.device)
    print(json.dumps({"checks": res["checks"], "trained": {k: v for k, v in res["trained"].items()
                                                           if k != "condition_g"},
                      "condition_g": {k: v for k, v in res["trained"]["condition_g"].items()
                                      if not isinstance(v, list)}}, indent=1, default=str))


if __name__ == "__main__":
    main()


__all__ = ["TwoGoalEnv", "imitate", "zero_intent_check", "condition_g", "kl_under_permutation", "replay_check",
           "stamp_check", "accuracy", "run"]
