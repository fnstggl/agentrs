"""Main-scale instantiation of the complete system (spec 1.1 / Part 20 A01: "actual 16-layer main configuration
instantiated and run").

`main_scale_probe` builds ``make_config("main")`` (16 layers x 768 wide x 12 heads, 64 state tokens) with every
component - Qwen3-1.7B semantic model, tokenizer, action generator with its LoRA and bridge, core, entities, reasoner,
beliefs, Kalman layer, outcome head, world model, retriever, value / option / budget heads, built-in planner - plus
the trainer, archive, replay, consolidator and trace recorder, then lives a few decisions in a real ``ow_tiny`` world
through the lifetime harness and performs one neural update. Reported: parameter counts per component (CAM's own
trainable parameters vs the pretrained semantic model, and the spec's "core + heads" figure), wall time and peak GPU
memory of every act and of the update.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Optional

import torch

from .harness_agent import HarnessAgent, _jsonable
from .system import SystemConfig, build_system, describe
from .tasks import make_task, open_task_world

RUNS = "/workspace/agentrs/runs/cam_unified/wave2/S"
SPEC_TARGET_CORE_HEADS = (120_000_000, 180_000_000)


def _rows(path: str) -> list:
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path) if x.strip()]


def core_and_heads(counts: dict) -> dict:
    """The spec's 'trainable core + heads' figure: every CAM component except the semantic-side adapters
    (tokenizer projections are CAM's own and included; LoRA / pretrained weights are not), and the same without the
    action generator (whose latent-to-language bridge maps into the 2048-wide Qwen embedding space)."""
    comps = counts["components"]
    tot = sum(v["trainable"] for v in comps.values())
    ag = comps.get("action_generator", {}).get("trainable", 0)
    return {"cam_trainable_total": tot, "cam_trainable_without_action_generator": tot - ag,
            "action_generator_trainable": ag, "spec_target": list(SPEC_TARGET_CORE_HEADS),
            "in_target_total": SPEC_TARGET_CORE_HEADS[0] <= tot <= SPEC_TARGET_CORE_HEADS[1]}


def main_scale_probe(sem_model: Optional[str] = None, work_dir: Optional[str] = None, n_decisions: int = 3,
                     imag: tuple = (2, 3, 1), max_arg_tokens: int = 32, keep: bool = False) -> dict:
    os.makedirs(RUNS, exist_ok=True)
    work = work_dir or tempfile.mkdtemp(prefix="main_scale_", dir=RUNS)
    shutil.rmtree(work, ignore_errors=True)
    sc = SystemConfig(scale="main", overrides={"max_arg_tokens": int(max_arg_tokens)}, sem_model=sem_model,
                      imag_candidates=imag[0], imag_horizon=imag[1], imag_samples=imag[2], seq_len=4, n_seqs=2,
                      adapt_seq_len=4, adapt_n_seqs=1)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    system = build_system(work, sc)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    build_s = time.time() - t0
    build_peak = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None
    desc = describe(system)
    h = HarnessAgent(system)
    task, fam, seed = make_task("open_economy", 0)
    ad, rop, rev = open_task_world(task, store_url=f"sqlite:///{os.path.join(work, 'world.db')}")
    from ..envs.harness import run_lifetime
    try:
        res = run_lifetime(h, ad, rev, max_decisions=n_decisions, think_s=600.0, update_every=n_decisions,
                           model_version=lambda: h.model_version)
    finally:
        ad.close()
    dec = _rows(os.path.join(work, "decisions.jsonl"))
    ups = _rows(os.path.join(work, "learner", "updates.jsonl"))
    counts = desc["parameters"]
    out = {"config": {k: desc["cam_config"][k] for k in ("scale", "n_layers", "d_model", "n_heads", "n_state",
                                                          "n_particles", "d_particle", "max_arg_tokens")},
           "d_ff": system.cfg.d_ff, "semantic_model": desc["semantic_model"], "build_report": desc["build_report"],
           "parameters": counts, "action_generator_parts": desc["action_generator_parts"],
           "core_and_heads": core_and_heads(counts), "trainer_groups": desc["trainer"]["groups"],
           "imagination": desc["imagination"], "build_s": build_s, "build_peak_gpu_mb": build_peak,
           "task": {"name": "open_economy", "world_family": fam, "world_seed": seed},
           "acts": [{"step": r["step"], "tool": r["tool"], "source": r["source"], "ok": r["ok"], "error": r["error"],
                     "act_s": r["act_s"], "peak_gpu_mb_act": r["peak_gpu_mb_act"],
                     "decoded_tokens_total": r["decoded_tokens_total"], "n_tools": r["n_tools"]} for r in dec],
           "update": None if not ups else {k: ups[0].get(k) for k in (
               "version", "losses", "update_s", "peak_gpu_mb_update", "grad_norm", "group_grad_norm",
               "backbone_grad_norm", "core_hash_before", "core_hash_after")}
           | {"core_l2_delta": ((ups[0].get("param_delta") or {}).get("core") or {}).get("l2_delta")},
           "lifetime_peak_gpu_mb": h.peak_gpu_mb, "decisions": res.steps,
           "work_dir": work if keep else None}
    system.recorder.close()
    system.archive.close()
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return _jsonable(out)


# ============================================================================ planner measurement (spec 7.2, A08)

@torch.no_grad()
def plan_once(model, inp, horizon: int, belief_aware: bool, n_candidates: int, n_samples: int, cand_seed: int,
              roll_seed: int):
    """Re-plan one stored real decision (StepInputs row) with the model's planner: the candidates are drawn from
    CAM's action generator under `cand_seed` (identical across conditions), the imagined rollouts (stochastic
    latents, sampled outcomes, later imagined actions) under `roll_seed`; `horizon` imagined decisions, with the
    row's own belief (cloned FastState updated with imagined outcomes) or without beliefs. Outcome queries are the
    core keys of a fresh outcome tracker (the per-environment tracked numeric keys are not part of StepInputs)."""
    import copy as _copy
    from ..model import PlanRequest, StepInputs, acting_mode
    from ..outcome_spec import build_outcome_batch

    def seed(s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    regs = _copy.deepcopy(inp.registries)
    live = StepInputs(**{**inp.__dict__, "registries": regs})
    with acting_mode(model), model._autocast():
        enc = model.encode_step(live)
        opt_tok = model._option_token(inp.option, inp.option_steps) if model.option_policy is not None else None
        goal_in = model._goal_inputs(enc, opt_tok, inp.budget)
        T = model.tokenizer.tools(inp.tools)
        specs = [model.new_tracker().prospective() for _ in range(inp.B)]
        prosp = build_outcome_batch(specs, model.query_encoder, regs, [[-1] * len(s) for s in specs], model.device)
        dt0 = torch.full((inp.B,), model.think_days, device=model.device)
        B = inp.B
        rep = torch.arange(B, device=model.device).repeat_interleave(n_candidates)
        seed(cand_seed)
        cands = model.action_generator.sample(enc.R[rep], goal_in.select(rep), enc.belief.select(rep), T.select(rep),
                                              enc.obs.select(rep), [inp.tools[int(i)] for i in rep],
                                              [inp.obs_lists[int(i)] for i in rep], temperature=1.0)
        req = PlanRequest(enc.Z, enc.R, inp.fast if belief_aware else None, goal_in, enc.goal, enc.belief,
                          enc.memory, T, enc.obs, list(inp.tools), list(inp.obs_lists), prosp, dt0, n_candidates,
                          horizon, n_samples, None, cands)
        calls0 = dict(model.calls)
        seed(roll_seed)
        plan = model.planner.plan(model, req)
    used = {k: model.calls.get(k, 0) - calls0.get(k, 0) for k in model.calls
            if k.startswith(("imagination.", "world_model.prior", "action_generator"))}
    return plan, used


def planner_probe(system, inputs, horizons=(1, 3), cand_seeds=(0, 1, 2, 3), roll_seeds=(10, 11),
                  n_candidates: int = 2, n_samples: int = 1) -> dict:
    """For every stored real decision x candidate seed (the SAME CAM candidates in every condition): the planner's
    argmax choice when imagining 1 vs H decisions ahead, belief-aware vs belief-free, each under two rollout seeds.
    The rollout-seed pairs of one condition give the noise floor (how often the choice flips with only the imagined
    samples changed); the condition pairs give the effect of horizon and beliefs."""
    model = system.model
    H = max(horizons)
    conds = [(h, True) for h in horizons] + [(H, False)]
    name = lambda h, bel: f"h{h}_{'belief' if bel else 'nobelief'}"      # noqa: E731
    rows = []
    for b in range(inputs.B):
        one = inputs.select([b])
        for cs in cand_seeds:
            r = {"row": b, "cand_seed": cs, "choices": {}, "scores": {}}
            sig = None
            for h, bel in conds:
                for rs in roll_seeds:
                    plan, used = plan_once(model, one, h, bel, n_candidates, n_samples, 1000 * b + cs, rs)
                    p = plan.actions[0].proposal
                    s_ = [(c["action"]["tool"], json.dumps(c["action"]["args"], sort_keys=True))
                          for c in plan.candidates[0]]
                    if sig is None:
                        sig = s_
                    if s_ != sig:
                        raise RuntimeError("candidates differ across conditions (non-deterministic sampling)")
                    r["choices"][f"{name(h, bel)}@{rs}"] = p["chosen"]
                    r["scores"][f"{name(h, bel)}@{rs}"] = p["scores"]
                    if h == H and bel and rs == roll_seeds[0]:
                        r["model_calls_full"] = used
            r["candidate_tools"] = [t for t, _ in sig]
            r["distinct_candidates"] = len(set(sig)) > 1
            rows.append(r)
    ok = [r for r in rows if r["distinct_candidates"]]

    def flip(a, b):
        if not ok:
            return None
        return sum(1 for r in ok if r["choices"][a] != r["choices"][b]) / len(ok)
    r0, r1 = roll_seeds[0], roll_seeds[1]
    full = name(H, True)
    out = {"planner": type(model.planner).__name__, "horizons": list(horizons), "n_candidates": n_candidates,
           "n_samples": n_samples, "n_contexts": inputs.B, "n_pairs_with_distinct_candidates": len(ok),
           "noise_floor": {name(h, bel): flip(f"{name(h, bel)}@{r0}", f"{name(h, bel)}@{r1}") for h, bel in conds},
           "effect": {f"{full}_vs_{name(h, bel)}": flip(f"{full}@{r0}", f"{name(h, bel)}@{r0}")
                      for h, bel in conds if name(h, bel) != full},
           "model_calls_full": rows[0].get("model_calls_full") if rows else None, "rows": rows}
    return out


def planner_probe_run(run_dir: str, out_path: str) -> dict:
    """Load the run's latest lifetime checkpoint into a fresh system (this process) and measure the planner on the
    run's fixed probe contexts (real OpenWorld decisions)."""
    from .harness_agent import HarnessAgent
    from .runner import load_spec
    from .system import SystemConfig
    import tempfile
    prev = load_spec(run_dir)
    sc = SystemConfig(**prev["spec"]["system"])
    work = tempfile.mkdtemp(prefix="planner_probe_", dir=RUNS)
    shutil.rmtree(work)
    shutil.copytree(run_dir, work, ignore=shutil.ignore_patterns("dashboard", "*.tmp*"))
    try:
        system = build_system(work, sc)
        h = HarnessAgent(system)
        latest = open(os.path.join(work, "lifetime_ckpt", "LATEST")).read().strip()
        info = h.load(os.path.join(work, "lifetime_ckpt", latest, "agent"))
        p = h._probe
        m = system.model
        res = planner_probe(system, p["inputs"], horizons=(1, m.imag_horizon), n_candidates=m.imag_candidates,
                            n_samples=m.imag_samples)
        res.update({"run_dir": run_dir, "checkpoint_version": info["version"], "probe_event_ids": p["event_ids"]})
    finally:
        shutil.rmtree(work, ignore_errors=True)
    with open(out_path, "w") as fh:
        json.dump(_jsonable(res), fh, indent=1)
    return res
