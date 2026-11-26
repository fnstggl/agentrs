"""W4 SEM real end-to-end check (plan 2026-11-26 §3 W4), on the real dom_dev stack: build_model(make_preset("dom_dev"))
= 8 x 384 core + Qwen3-0.6B, the real ObservationTokenizer / ActionGenerator / HoleFiller / LatentDecoderConditioning,
and real ow_tiny ExampleInductor transcripts (runs/cam_unified/wave2/A2/demo_init/{train,held}) replayed through the
model's own acting computation (demo_init.teacher_forced_rollout). Nothing here is a test double.

Checks (numbers go to superlearner/docs/evidence/dom/W4/e2e.json):
  A. VALUE channel: cos(+1, -1) of NUMERIC tokens at init < 0.5 (legacy path for comparison).
  B. String closure: >= 90% of >= 50 ow_tiny string slots close by the LM's own choice (greedy, no length prior),
     0 trailing commas; the pre-v4 protocol on the same inputs for comparison.
  C. Decoder conditioning at init: logits / string logp bit-identical with and without the CAM latent.
  D. 100 BC steps on demonstrated strings (conditioning + core): ablating Z (cond=None, or another decision's Z)
     changes the string logp by > 1 nat per action on held-out demonstrations.
  E. Exact replay with the conditioning on: strings sampled with cond rescore bit for bit (twice, and in another
     batch composition).
  F. semantic_context() on real observations -> GatedUnifiedCore consumes it (shape, refs, budget, gradient).
  G. code.run demonstration arguments representable (reported; expected to fail: needs a JSON-value hole).

Run (GPU, under the shared lock):
  flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock env PYTHONPATH=src:../crucible/src \
      /workspace/agentrs/.venv/bin/python tests/cam_unified/semantic/w4_e2e.py --out docs/evidence/dom/W4/e2e.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import replace

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEMO = "/workspace/agentrs/runs/cam_unified/wave2/A2/demo_init"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def part_a(model, cfg) -> dict:
    from superlearner.cam_unified.interfaces import EntityRegistry, Observation, ObsKind
    from superlearner.cam_unified.semantic.tokenizer import ObservationTokenizer
    enc = model.action_generator.enc

    def num(v, entity=None):
        return Observation(ObsKind.NUMERIC, value=float(v), unit="cents", source="bank.accounts",
                           entity=entity, entity_type="account" if entity else None, t=0.0,
                           meta={"metric": "balance"})

    def tok_cos(tk, a, b, entity=None):
        regs = [EntityRegistry(cfg.max_entities, seed=0), EntityRegistry(cfg.max_entities, seed=0)]
        with torch.no_grad():
            ts = tk.observations([[num(a, entity)], [num(b, entity)]], regs, [0.0, 0.0])
        return cos(ts.x[0, 0], ts.x[1, 0])

    torch.manual_seed(0)
    legacy = ObservationTokenizer(replace(cfg, value_channel=False), enc)
    pairs = [(1, -1), (5, -5), (0, 1), (100, 200), (100, 101), (1200, 1250), (0.25, 0.5), (1e6, 2e6)]
    out = {"value_channel": bool(model.tokenizer.value_channel), "pairs": {}}
    for a, b in pairs:
        out["pairs"][f"{a} vs {b}"] = {"value_channel": round(tok_cos(model.tokenizer, a, b), 4),
                                       "value_channel_same_entity": round(tok_cos(model.tokenizer, a, b, "acct_1"), 4),
                                       "legacy": round(tok_cos(legacy, a, b), 4)}
    out["cos_plus1_minus1"] = out["pairs"]["1 vs -1"]["value_channel"]
    out["cos_plus1_minus1_same_entity"] = out["pairs"]["1 vs -1"]["value_channel_same_entity"]
    out["cos_plus1_minus1_legacy"] = out["pairs"]["1 vs -1"]["legacy"]
    out["pass"] = out["cos_plus1_minus1"] < 0.5
    return out


def string_steps(model, episodes):
    """Representable non-wait demonstrated decisions whose action has string holes -> [(ForcedStep, n_holes)]."""
    from superlearner.cam_unified.semantic.demo_init import demo_string_programs, teacher_forced_rollout
    steps = teacher_forced_rollout(model, episodes, batch=8, keep=lambda ep, d: d.action.tool != "wait")
    gen = model.action_generator
    out = []
    for st in steps:
        if st.unrepresentable is not None:
            continue
        tool = next((t for t in st.inputs.tools[0] if t.name == st.action.tool), None)
        if tool is None or not gen.grammar(tool).may_hole:
            continue
        try:
            progs, _ = demo_string_programs(model, [st])
        except ValueError:
            continue                                  # no string holes in this demonstration
        out.append((st, len(progs[0].holes)))
    return out


def _leaves(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from _leaves(x)
    elif isinstance(v, list):
        for x in v:
            yield from _leaves(x)
    else:
        yield v


def _force_mask(gen, tool_lists, names):
    lists = gen.tool_lists_with_builtins(tool_lists)
    T = max(len(l) for l in lists)
    m = torch.zeros(len(lists), T, dtype=torch.bool, device=gen.device)
    for b, (lst, n) in enumerate(zip(lists, names)):
        m[b, [t.name for t in lst].index(n)] = True
    return m


def sample_strings(model, steps, protocol: int, batch: int = 4) -> dict:
    """Greedy CAM-decoder actions (demo tool forced, no length prior) on the steps' real decision contexts."""
    from superlearner.cam_unified.model import StepInputs, acting_mode
    from superlearner.cam_unified.semantic.demo_init import _stack_inputs
    import copy
    gen = model.action_generator
    filler = gen.filler
    captured = []
    orig = filler.sample

    def spy(programs, *a, **k):
        captured.append(programs)
        return orig(programs, *a, **k)

    saved = (gen.length_head, filler.length_head, filler.protocol)
    gen.length_head, filler.length_head, filler.protocol = None, None, protocol
    filler.sample = spy
    holes, actions = [], []
    try:
        for c0 in range(0, len(steps), batch):
            sub = steps[c0: c0 + batch]
            inputs = _stack_inputs(sub)
            live = StepInputs(**{**inputs.__dict__, "registries": copy.deepcopy(inputs.registries)})
            with torch.no_grad(), model._autocast(), acting_mode(model):
                encd = model.encode_step(live)
                opt_tok = model._option_token(inputs.option, inputs.option_steps) \
                    if model.option_policy is not None else None
                goal_in = model._goal_inputs(encd, opt_tok, inputs.budget)
                T = model.tokenizer.tools(inputs.tools)
                acts = gen.sample(encd.R, goal_in, encd.belief, T, encd.obs, inputs.tools, inputs.obs_lists,
                                  greedy=True, tool_mask=_force_mask(gen, inputs.tools, [s.action.tool for s in sub]),
                                  rescore=False)
            actions += acts
            for progs in captured:
                for p in progs:
                    for k, h in enumerate(p.holes):
                        holes.append({"kind": h.kind, "path": h.node.path, "value": h.value,
                                      "n_tokens": len(h.tokens), "cap": int(p.cap[k]), "max_len": int(h.max_len),
                                      "own_close": bool(h.own_close), "consumed": int(h.consumed),
                                      "close_token": filler.tables.texts[h.tokens[-1]]})
            captured.clear()
    finally:
        gen.length_head, filler.length_head, filler.protocol = saved
        filler.sample = orig
    n = len(holes)
    digit_runs = sum(1 for h in holes if re.search(r"\d{12,}", h["value"] or ""))
    frag = sum(1 for h in holes if re.search(r'",\s*"|":\s*"|\{"', h["value"] or ""))
    leak = sum(1 for h in holes if "controller" in (h["value"] or "").lower())
    return {"protocol": protocol, "actions": len(actions), "slots": n,
            "own_close": sum(h["own_close"] for h in holes),
            "own_close_rate": sum(h["own_close"] for h in holes) / max(n, 1),
            "trailing_commas": sum(1 for h in holes if (h["value"] or "").rstrip().endswith(",")),
            "merged_closes": sum(1 for h in holes if h["consumed"] > 0),
            "digit_runs_12plus": digit_runs, "json_fragments": frag, "init_phrase_leaks": leak,
            "tokens_mean": sum(h["n_tokens"] for h in holes) / max(n, 1),
            "tokens_max": max([h["n_tokens"] for h in holes] + [0]),
            "forced_by_kind": {k: sum(1 for h in holes if h["kind"] == k and not h["own_close"])
                               for k in ("str", "pattern", "key")},
            "slots_by_kind": {k: sum(1 for h in holes if h["kind"] == k) for k in ("str", "pattern", "key")},
            "forced": [{"kind": h["kind"], "path": h["path"], "value": h["value"][:60], "n_tokens": h["n_tokens"],
                        "cap": h["cap"], "max_len": h["max_len"]} for h in holes if not h["own_close"]][:20],
            "examples": [{"value": h["value"][:80], "close": h["close_token"], "own": h["own_close"]}
                         for h in holes[:12]]}


def part_c_init(model, held) -> dict:
    """Gates at init: the decoder with the CAM latent equals the unconditioned decoder bit for bit."""
    from superlearner.cam_unified.semantic.demo_init import (decision_conditioning, demo_string_log_prob,
                                                             demo_string_programs)
    gen = model.action_generator
    enc = gen.enc
    with torch.no_grad():
        lp_c, nt = demo_string_log_prob(model, held, cond="latent")
        lp_n, _ = demo_string_log_prob(model, held, cond=None)
        progs, ctxs = demo_string_programs(model, held[:4])
        with model._autocast():
            prefix, c = decision_conditioning(model, held[:4])
            tails = [gen.filler.stub_ids + gen.filler.segment(p, 0) for p in progs]
            emb, att, pos, _ = gen.filler._assemble(prefix, ctxs, tails)
            l_c = enc.lm_logits(enc.decoder_hidden(emb, att, pos, cond=c)[:, -48:])   # last 48 positions (memory)
            l_n = enc.lm_logits(enc.decoder_hidden(emb, att, pos)[:, -48:])
            s_c, _ = enc.step(emb, att, pos, enc.new_cache(), cond=c)
            s_n, _ = enc.step(emb, att, pos, enc.new_cache())
    return {"actions": len(held), "string_logp_bit_identical": bool(torch.equal(lp_c, lp_n)),
            "decoder_hidden_logits_bit_identical": bool(torch.equal(l_c, l_n)),
            "step_logits_bit_identical": bool(torch.equal(s_c, s_n)),
            "gates": enc.conditioning.gates(), "layers": enc.conditioning.layer_ids,
            "conditioning_params": enc.conditioning.n_params(), "tokens": int(nt.sum())}


def ablation(model, steps, batch=8) -> dict:
    from superlearner.cam_unified.semantic.demo_init import demo_string_log_prob
    d_none, d_perm, nll = [], [], []
    with torch.no_grad():
        for c0 in range(0, len(steps), batch):
            sub = steps[c0: c0 + batch]
            if len(sub) < 2:
                continue
            lp, nt = demo_string_log_prob(model, sub, cond="latent")
            ln, _ = demo_string_log_prob(model, sub, cond=None)
            perm = list(range(1, len(sub))) + [0]
            lq, _ = demo_string_log_prob(model, sub, cond="latent", z_perm=perm)
            d_none += (lp - ln).abs().tolist()
            d_perm += (lp - lq).abs().tolist()
            nll += (-lp / nt.float()).tolist()
    m = lambda v: sum(v) / max(len(v), 1)
    med = lambda v: sorted(v)[len(v) // 2] if v else 0.0
    return {"actions": len(d_none), "abs_dlogp_no_latent_mean": m(d_none), "abs_dlogp_no_latent_median": med(d_none),
            "abs_dlogp_other_latent_mean": m(d_perm), "abs_dlogp_other_latent_median": med(d_perm),
            "nll_per_token": m(nll)}


def part_d(model, train, held, n_steps=100, batch=8, seed=0, deadline=None) -> dict:
    from superlearner.cam_unified.semantic.demo_init import demo_string_log_prob
    gen = model.action_generator
    enc = gen.enc
    gates = [m.alpha for m in enc.conditioning.mods.values()]
    gate_ids = {id(p) for p in gates}
    cond_params = [p for p in enc.conditioning.parameters() if id(p) not in gate_ids] + \
        list(gen.bridge.cond_src.parameters())
    core_params = [p for m in (model.core, model.reasoner) if m is not None for p in m.parameters()
                   if p.requires_grad]
    opt = torch.optim.AdamW([{"params": gates, "lr": 1e-2}, {"params": cond_params, "lr": 1e-3},
                             {"params": core_params, "lr": 1e-4}], weight_decay=0.0)
    cond_params = cond_params + gates
    before = ablation(model, held)
    rng = random.Random(seed)
    losses, t0 = [], time.time()
    model.train()
    for k in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        sub = rng.sample(train, min(batch, len(train)))
        lp, nt = demo_string_log_prob(model, sub, cond="latent")
        loss = -(lp / nt.float()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_core = torch.nn.utils.clip_grad_norm_(core_params, 1.0)
        g_cond = torch.nn.utils.clip_grad_norm_(cond_params, 1.0)
        opt.step()
        losses.append(float(loss))
        if k == 0:
            first = {"grad_core": float(g_core), "grad_cond": float(g_cond)}
    model.eval()
    after = ablation(model, held)
    after_train = ablation(model, train[:48])
    return {"bc_steps": len(losses), "bc_steps_planned": n_steps, "batch": batch, "train_actions": len(train),
            "held_actions": len(held),
            "optimizer": "AdamW: gates alpha lr 1e-2, decoder cross-attention + cond source tags lr 1e-3, core + "
                         "reasoner lr 1e-4; clip 1.0 per group; loss = mean per-token NLL of the demonstrated strings",
            "loss_first": losses[0], "loss_last10_mean": sum(losses[-10:]) / 10, "seconds": time.time() - t0,
            "first_step_grad_norms": first, "gates_after": enc.conditioning.gates(),
            "held_before": before, "held_after": after, "train_after": after_train,
            "pass": after["abs_dlogp_no_latent_mean"] > 1.0}


def part_e(model, held) -> dict:
    """Strings sampled with the conditioning on rescore bit for bit (twice; reversed half batch)."""
    from superlearner.cam_unified.semantic.demo_init import decision_conditioning, demo_string_programs
    gen = model.action_generator
    filler = gen.filler
    sub = held[:8]
    with torch.no_grad():
        progs, ctxs = demo_string_programs(model, sub)
        with model._autocast():
            prefix, c = decision_conditioning(model, sub)
            lp_inc, _, ns, _ = filler.sample(progs, prefix, ctxs, 1.0, False,
                                             torch.Generator(device=gen.device).manual_seed(0), cond=c)
    tokens = [[list(h.tokens) for h in p.holes] for p in progs]
    with model._autocast():
        prefix, c = decision_conditioning(model, sub)
        s1, _, _ = filler.score(progs, prefix, ctxs, 1.0, exact_rows=True, cond=c)   # grad-enabled learner path
        s2, _, _ = filler.score(progs, prefix, ctxs, 1.0, exact_rows=True, cond=c)
        perm = list(reversed(range(len(sub))))[: len(sub) // 2]
        idx = torch.tensor(perm, device=prefix.device)
        s3, _, _ = filler.score([progs[i] for i in perm], prefix[idx], [ctxs[i] for i in perm], 1.0, exact_rows=True,
                                cond=(c[0][idx], c[1][idx]))
        s0, _, _ = filler.score(progs, prefix, ctxs, 1.0, exact_rows=True)           # without the latent
    same_tokens = tokens == [[list(h.tokens) for h in p.holes] for p in progs]
    return {"actions": len(sub), "sampled_tokens": int(sum(ns)), "rescore_twice_bit_identical": bool(torch.equal(s1, s2)),
            "rescore_other_batch_bit_identical": all(float(s3[j]) == float(s1[i]) for j, i in enumerate(perm)),
            "record_unchanged": same_tokens, "requires_grad": bool(s1.requires_grad),
            "incremental_vs_rescore_max_abs": float((lp_inc - s1.detach()).abs().max()),
            "latent_changes_logp_mean_abs": float((s1 - s0).abs().mean()),
            "consumed": [p.consumed() for p in progs]}


def part_f(model, cfg, steps) -> dict:
    from superlearner.cam_unified.core.semantic_xattn import GatedUnifiedCore
    tok = model.tokenizer
    sub = steps[:4]
    obs = [s.inputs.obs_lists[0] for s in sub]
    tools = [s.inputs.tools[0] for s in sub]
    goals = [s.inputs.goals[0] for s in sub]
    t0 = time.time()
    sc = tok.semantic_context(obs, tools, goals)
    t1 = time.time()
    sc2 = tok.semantic_context(obs, tools, goals)
    t2 = time.time()
    import copy
    core = GatedUnifiedCore(cfg).to(model.device)
    with torch.no_grad():
        ctx = tok.observations(obs, [copy.deepcopy(s.inputs.registries[0]) for s in sub], [s.inputs.now[0] for s in sub])
    core.set_semantic(sc)
    Z, _ = core(core.init_state(len(sub), device=model.device), ctx)
    Z.float().pow(2).mean().backward()
    g = {k: float(sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None))
         for k, m in (("sem_proj", core.sem_proj), ("xattn", core.xattn))}
    refs0 = sorted({r.split("#")[0] for r in sc.refs[0] if r})
    return {"h_shape": list(sc.h.shape), "tokens_per_row": sc.stats["tokens"], "budget": sc.stats["budget"],
            "kept_obs": sc.stats["kept_obs"], "dropped_obs": sc.stats["dropped_obs"],
            "cut_tokens": sc.stats["cut_tokens"], "row0_refs": refs0[:12], "first_s": t1 - t0, "cached_s": t2 - t1,
            "identical_second_call": bool(torch.equal(sc.h, sc2.h)), "gated_core_grad_abs_sum": g,
            "last_gates": core.last_gates}


def part_g() -> dict:
    from superlearner.cam_unified.interfaces import ToolSchema
    from superlearner.cam_unified.semantic.grammar import GrammarError, GrammarOptions, ToolGrammar, encode_value
    opts = GrammarOptions(max_string_tokens=1536, max_action_tokens=3072)
    ok, bad, err = 0, 0, None
    for split in ("train", "held"):
        for line in open(f"{DEMO}/{split}/episodes.jsonl"):
            ep = json.loads(line)
            tools = {t["name"]: ToolSchema.from_dict(t) for t in list(ep["tool_table"].values())[0]}
            for d in ep["decisions"]:
                if d["action"]["tool"] != "code.run":
                    continue
                g = ToolGrammar(tools["code.run"], opts)
                try:
                    encode_value(g.root, d["action"]["args"], opts, [], [], {}, None)
                    ok += 1
                except GrammarError as e:
                    bad += 1
                    err = str(e)[:160]
    return {"code_run_demos": ok + bad, "representable": ok, "error": err, "pass": bad == 0 and ok > 0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/evidence/dom/W4/e2e.json")
    ap.add_argument("--train-episodes", type=int, default=4)
    ap.add_argument("--held-episodes", type=int, default=2)
    ap.add_argument("--bc-steps", type=int, default=100)
    ap.add_argument("--stages", default="ABCDEFG", help="stages to run; results merge into --out")
    ap.add_argument("--budget-s", type=float, default=840.0, help="wall-clock budget (the GPU-lock hold limit)")
    ap.add_argument("--mem-cap-gb", type=float, default=0.0, help="hard per-process GPU memory cap (0 = none)")
    args = ap.parse_args(argv)
    t_start = time.time()
    return _run(args, argv, t_start)


def _run(args, argv, t_start):
    if args.mem_cap_gb > 0 and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gb / total))
    from superlearner.cam_unified.config import make_preset
    from superlearner.cam_unified.model import build_model
    from superlearner.cam_unified.semantic.demo_init import load_episodes
    torch.manual_seed(0)
    cfg = make_preset("dom_dev", declared_conditions=("demo-init:example_inductor",))
    model = build_model(cfg).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    gen = model.action_generator
    res = {}
    if os.path.exists(args.out):                 # stages merge into an existing evidence file
        with open(args.out) as fh:
            res = json.load(fh)
    run = {"commit": _git_commit(), "preset": "dom_dev", "sem_model": cfg.sem_model,
           "core": f"{cfg.n_layers}x{cfg.d_model}",
           "command": " ".join(["python", "tests/cam_unified/semantic/w4_e2e.py"] + list(argv or sys.argv[1:])),
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "dec_xattn_every": cfg.dec_xattn_every, "bridge_prefix_tokens": gen.bridge.n_prefix,
           "stages": args.stages, "timing_s": {}}
    res.setdefault("runs", []).append(run)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def save(stage, key, value, t0):
        res[key] = value
        run["timing_s"][stage] = round(time.time() - t0, 1)
        run["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        print(stage, json.dumps(value, default=str)[:1500], f"({run['timing_s'][stage]} s)", flush=True)

    S = set(args.stages)
    if "A" in S:
        t0 = time.time()
        save("A", "A_value_channel", part_a(model, cfg), t0)
    if "G" in S:
        t0 = time.time()
        save("G", "G_code_run", part_g(), t0)
    if not S & set("BCDEF"):
        return
    t0 = time.time()
    train_eps = load_episodes(f"{DEMO}/train/episodes.jsonl")[: args.train_episodes]
    held_eps = load_episodes(f"{DEMO}/held/episodes.jsonl")[: args.held_episodes]
    run["families"] = sorted({e.family for e in train_eps + held_eps})
    held_keys = {e.key for e in held_eps}
    both = string_steps(model, held_eps + train_eps)              # one batched rollout (<= 8 episodes)
    held = [s for s, _ in both if s.episode in held_keys]
    train = [s for s, _ in both if s.episode not in held_keys]
    run["timing_s"]["rollout"] = round(time.time() - t0, 1)
    run["string_steps"] = {"train": len(train), "held": len(held)}
    print("steps", len(train), len(held), f"({run['timing_s']['rollout']} s)", flush=True)
    if "B" in S:
        t0 = time.time()
        slots, n_h = [], 0
        for st, nh in sorted(both, key=lambda x: x[0].episode not in held_keys):   # held-out decisions first
            slots.append(st)
            n_h += nh
            if n_h >= 60:
                break
        b = {"v4": sample_strings(model, slots, 4), "v3_pre_fix": sample_strings(model, slots, 3)}
        v4 = b["v4"]
        b["pass"] = v4["slots"] >= 50 and v4["own_close_rate"] >= 0.9 and v4["trailing_commas"] == 0
        save("B", "B_string_closure", b, t0)
    if "C" in S:
        t0 = time.time()
        save("C", "C_conditioning_at_init", part_c_init(model, held), t0)
    if "F" in S:
        t0 = time.time()
        save("F", "F_semantic_context", part_f(model, cfg, held), t0)
    if "D" in S:
        t0 = time.time()
        reserve = 150.0 if "E" in S else 60.0                     # time left for the after-ablations (+ E)
        save("D", "D_bc_latent_dependence", part_d(model, train, held, n_steps=args.bc_steps,
                                                   deadline=t_start + args.budget_s - reserve - 120.0), t0)
    if "E" in S:
        t0 = time.time()
        save("E", "E_replay_with_conditioning", part_e(model, held), t0)
    run["wall_s"] = time.time() - t_start
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print("wrote", args.out, "wall", run["wall_s"], flush=True)


if __name__ == "__main__":
    main()
