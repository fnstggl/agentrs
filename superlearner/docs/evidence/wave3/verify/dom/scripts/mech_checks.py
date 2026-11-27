"""Independent checks on the REAL build_agent('dom_dev') model: reward-model attach, Mechanism C wiring/dependence and
gate conditioning, Mechanism A wiring + gradient through the frozen decoder, Condition G (incl. R-before-intent and
Z swaps), and whether reasoning iterations change decisions (A03), before and after the Condition-G imitation."""
import os, sys, json, time, copy, traceback
os.environ.setdefault("HF_HOME", "/workspace/.hf_home"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/workspace/agentrs/crucible/src")
import torch
OUT = sys.argv[1]
res = {}
def dump():
    with open(OUT, "w") as f: json.dump(res, f, indent=1, default=str)
def section(name):
    def deco(fn):
        if ONLY and name not in ONLY: return fn
        t = time.time()
        try: res[name] = fn()
        except Exception as e: res[name] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2500:]}
        res[name + "_s"] = time.time() - t; dump()
        return fn
    return deco
import transformers
res["transformers"] = transformers.__version__
from superlearner.cam_unified.build import build_agent
from superlearner.cam_unified.decision import condition_g as cg
import importlib
SEL = importlib.import_module('superlearner.cam_unified.decision.select')
ONLY = set(filter(None, os.environ.get('ONLY', '').split(',')))
torch.manual_seed(0)
S = build_agent("dom_dev", device="cuda", run_dir="/workspace/agentrs/runs/verify_dom/mech_%d" % int(time.time()), seed=0)
m = S.model
res["learner"] = type(S.learner).__name__
dump()

@section("reward_model")
def _():
    env = cg._new_env(4, 9)
    dec = cg._decide(m, env)
    st = SEL.stamp_of(dec.actions[0])
    return {"model_reward_model_attached": m.__dict__.get("reward_model") is not None,
            "learner_reward_model": type(getattr(S.learner, "reward_model", None)).__name__,
            "system_reward_model": type(S.reward_model).__name__,
            "stamp_versions": st.get("versions"), "phi_in_stamp": st.get("phi"),
            "eval_versions": dec.evaluation[0].get("versions")}

def tool_dist(inp, **kw):
    lp, allowed, lists2, I = m.tool_distribution(inp, **kw)
    return lp.float(), I.float()

def kl(lp, lq):
    p = lp.exp(); msk = torch.isfinite(lp) & torch.isfinite(lq)
    return (p * (lp - lq)).masked_fill(~msk, 0.0).sum(-1)

def mech_c(tag):
    env = cg._new_env(8, 77, [b % 2 for b in range(8)])
    calls0 = dict(m.core.sem_stats)
    seen = {}
    orig = m.core.project_semantic
    def spy(sem, *a, **k):
        seen["cond_none"] = getattr(sem, "cond", "kv") is None
        seen["type"] = type(sem).__name__
        return orig(sem, *a, **k)
    m.core.project_semantic = spy
    try:
        dec = cg._decide(m, env, greedy=True)
    finally:
        m.core.__dict__.pop("project_semantic", None)
    calls1 = dict(m.core.sem_stats)
    lp, I = tool_dist(dec.inputs)
    orig_sc = m.semantic_context
    m.semantic_context = lambda inp: None
    try:
        lp0, I0 = tool_dist(dec.inputs)
    finally:
        m.__dict__.pop("semantic_context", None)
    return {"sem_stats_before": calls0, "sem_stats_after": calls1, "spy": seen,
            "gate_stats": m.core.gate_stats() if hasattr(m.core, "gate_stats") else None,
            "kl_tool_sem_vs_nosem_mean": float(kl(lp, lp0).mean()), "kl_tool_sem_vs_nosem_max": float(kl(lp, lp0).max()),
            "I_rel_change_nosem": float((I - I0).norm() / I.norm()),
            "Z_rel": None}

def a03(tag, base_J=None):
    """Reasoning iterations: tool distribution / greedy executed action / R across J on fixed inputs."""
    env = cg._new_env(16, 303, [b % 2 for b in range(16)])
    J0 = m.reason_iters
    out = {"default_reason_iters": J0}
    dec = cg._decide(m, env, greedy=True)
    lp_ref, I_ref = tool_dist(dec.inputs)
    acts_ref = None
    for J in (1, 2, 4, 8, 16):
        m.reason_iters = J
        try:
            lp, I = tool_dist(dec.inputs)
            env2 = cg._new_env(16, 303, [b % 2 for b in range(16)])
            d2 = cg._decide(m, env2, greedy=True)
            tools = [a.tool for a in d2.actions]
            args = [json.dumps(a.args, sort_keys=True) for a in d2.actions]
            if J == (J0 or 2):
                acts_ref = (tools, args)
            out[f"J{J}"] = {"kl_tool_vs_default_mean": float(kl(lp_ref, lp).mean()),
                            "kl_tool_vs_default_max": float(kl(lp_ref, lp).max()),
                            "I_rel_change": float((I - I_ref).norm() / I_ref.norm()),
                            "R_iters_reported": d2.reasoning[0]["iters"], "tools": tools, "args": args,
                            "acc": sum(int(t == env2.answer(b)) for b, t in enumerate(tools)) / 16}
        finally:
            m.reason_iters = J0
    for J in (1, 2, 4, 8, 16):
        o = out[f"J{J}"]
        o["tool_changed_frac_vs_default"] = sum(int(a != b) for a, b in zip(o.pop("tools"), acts_ref[0])) / 16
        o["action_changed_frac_vs_default"] = sum(int(a != b) for a, b in zip(o.pop("args"), acts_ref[1])) / 16
    return out

def cond_g_extra():
    """Swaps of R before the intent (the latent reasoning state) and of Z_prev (state tokens) at the 2nd decision."""
    B = 16
    goals = [b % 2 for b in range(B)]
    perm = torch.tensor([b + 1 if b % 2 == 0 else b - 1 for b in range(B)], device=m.device)
    env = cg._new_env(B, 909, goals)
    g = torch.Generator(device="cuda").manual_seed(3)
    with m.latent_intervention(lambda R: R[perm], before_intent=True):
        sp = cg._decide(m, cg._new_env(B, 909, goals), greedy=True)
    pre = sum(int(sp.actions[b].tool == env.answer(int(perm[b]))) for b in range(B)) / B
    # replace R by the donor's R AND recompute nothing else: also report the intent change
    d = cg._decide(m, cg._new_env(B, 909, goals), greedy=True)
    lp, I = tool_dist(d.inputs)
    return {"swap_R_before_intent_follow_donor_greedy": pre, "own_acc_greedy": sum(
        int(d.actions[b].tool == env.answer(b)) for b in range(B)) / B}

res["mech_c_init"] = None
@section("mech_c_init")
def _(): return mech_c("init")
@section("a03_init")
def _(): return a03("init")

@section("mech_a")
def _():
    out = {}
    cond = m.action_generator.dec_cond
    out["dec_cond_registered"] = cond is not None
    out["alphas_init"] = cond.gates()
    B = 4
    import crucible
    crucible.load_all_modules()
    from crucible.api import WorldServer
    from crucible.worldgen import generate
    from superlearner.cam_unified.interfaces import ToolSchema, Observation, ObsKind, GoalSpec, Action
    srv = WorldServer("memory://", lease_ttl_s=None)
    wid = srv.create_world(generate({"family": "ow_tiny"}, 1))
    me = srv.join_world(wid, "nova", endowment=3_000_000)
    tools = [ToolSchema(t["name"], t.get("description", ""), t["inputSchema"], cost_hint=None,
                        duration_hint=(t.get("annotations") or {}).get("costSeconds", 0) / 86400.0)
             for t in srv.available_tools(wid, me)]
    obs = [[Observation(ObsKind.TEXT, "Welcome to the market. You have 30000 dollars of funds.", source="world")]
           for _ in range(B)]
    goals = [GoalSpec("ow", "Build a profitable business in this world.") for _ in range(B)]
    g = torch.Generator(device="cuda").manual_seed(0)
    cands_with_holes = None
    calls0 = cond.calls
    for trial in range(8):
        state = m.init_state(B)
        dec = m.decide(state, obs, [0.0] * B, [list(tools)] * B, goals, generator=g)
        lc = dict(m.action_generator.last_compute)
        for b, a in enumerate(dec.actions):
            for c in SEL.stamp_of(a)["candidates"]:
                if (c["logp_parts"].get("decode") or {}).get("holes"):
                    cands_with_holes = (dec, b, Action.from_dict(c))
                    break
            if cands_with_holes: break
        if cands_with_holes: break
    out["decide_trials"] = trial + 1
    out["dec_cond_calls_during_decides"] = cond.calls - calls0
    out["last_compute_cond_decoder"] = lc.get("cond_decoder")
    if cands_with_holes is None:
        out["note"] = "no candidate with string holes found"; return out
    dec, b, act = cands_with_holes
    out["hole_action_tool"] = act.tool
    inp = dec.inputs.select([b])
    def token_grad(alpha_val):
        with torch.no_grad():
            for mod in cond.mods.values(): mod.alpha.fill_(alpha_val)
        m.train(); m.zero_grad(set_to_none=True)
        enc, _, _, opt_tok, _ = m._dom_encode(inp)
        I = m.intent(enc, opt_tok, inp.budget)
        I.retain_grad()
        lp, en = m.action_generator.log_prob_parts_from_intent(I, enc.tools, enc.ctx_tokens(), inp.tools,
                                                               inp.obs_lists, [act])
        tok = lp["tokens"].sum()
        tok.backward()
        gi = {}
        for lid, mod in cond.mods.items():
            gi[lid] = {"alpha": float(mod.alpha.grad.abs()) if mod.alpha.grad is not None else None,
                       "kv": float(mod.kv.weight.grad.abs().max()) if mod.kv.weight.grad is not None else None}
        ih = [p for p in m.intent_head.parameters() if p.grad is not None]
        b0 = [p for n, p in m.core.named_parameters() if n.startswith("blocks.0.") and p.grad is not None]
        r = {"tokens_logp": float(tok.detach()), "I_grad_absmax": float(I.grad.abs().max()) if I.grad is not None else None,
             "dec_cond_layers": gi,
             "intent_head_grad_absmax": max([float(p.grad.abs().max()) for p in ih], default=0.0),
             "core_block0_grad_absmax": max([float(p.grad.abs().max()) for p in b0], default=0.0),
             "sem_frozen_grads": sum(int(p.grad is not None) for p in m.action_generator.enc.lm.parameters())}
        # dependence of the string log-prob on I: permute/zero I
        with torch.no_grad():
            lpz, _ = m.action_generator.log_prob_parts_from_intent(I * 0.0 + I.flip(-1), enc.tools, enc.ctx_tokens(),
                                                                   inp.tools, inp.obs_lists, [act])
        r["tokens_logp_with_altered_I"] = float(lpz["tokens"].sum())
        m.eval(); m.zero_grad(set_to_none=True)
        return r
    a0 = {k: v.alpha.detach().clone() for k, v in cond.mods.items()}
    try:
        out["alpha0"] = token_grad(0.0)
        out["alpha0.3"] = token_grad(0.3)
    finally:
        with torch.no_grad():
            for k, v in cond.mods.items(): v.alpha.copy_(a0[k])
    return out

@section("cond_g_before")
def _():
    return {"kl": cg.kl_under_permutation(m, B=8, seed=31), "zero": cg.zero_intent_check(m, B=8, seed=71)}

@section("imitate")
def _():
    log = cg.imitate(m, steps=30, B=16, seed=1)
    return {"last": log["history"][-1], "seconds": log["seconds"]}

@section("cond_g_after")
def _():
    g = torch.Generator(device="cuda").manual_seed(6)
    r = cg.condition_g(m, B=16, seed=51, generator=g)
    r.pop("swapped_tools", None); r.pop("donor_answers", None)
    g2 = torch.Generator(device="cuda").manual_seed(5)
    return {"condition_g": r, "kl": cg.kl_under_permutation(m, B=16, seed=61),
            "zero": cg.zero_intent_check(m, B=8, seed=71), "acc": cg.accuracy(m, B=32, seed=41, generator=g2),
            "extra": cond_g_extra(), "stamps": cg.stamp_check(m)}

@section("mech_c_trained")
def _(): return mech_c("trained")
@section("a03_trained")
def _(): return a03("trained")
print("done")
