"""Profile decisions of the real configuration through the runtime (INT3A item 2).

    python profile_decision.py --root DIR --out P.json --decisions 4 [--plan SW:rle] [--cpu]

Same construction as ``eval.oe_runner`` (substantive config, condition E, OpenEndedComputer, LiveRuntime with the
runner's RuntimeConfig). Every interesting method of the agent, its components, the environment, the verifier and
the runtime is wrapped by a timer (inclusive wall time, CUDA synchronised before and after), accumulated per runtime
step; nested calls are reported inclusive (the parent includes its children).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import shutil
import sys
import time

import torch

STACK = []


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Prof:
    def __init__(self):
        self.cur = collections.defaultdict(float)
        self.cnt = collections.defaultdict(int)
        self.steps = []

    def wrap(self, obj, name, label):
        fn = getattr(obj, name, None)
        if fn is None or not callable(fn):
            return
        prof = self

        def timed(*a, **k):
            sync()
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                sync()
                prof.cur[label] += time.perf_counter() - t0
                prof.cnt[label] += 1
        setattr(obj, name, timed)

    def end_step(self, info):
        self.steps.append({"info": info, "seconds": dict(self.cur), "calls": dict(self.cnt)})
        self.cur = collections.defaultdict(float)
        self.cnt = collections.defaultdict(int)


def capture_voi(agent, path):
    """Save the inputs and outputs of the first VOI computation (old / new code compared on the same inputs)."""
    pred = agent.predictor
    fn = pred._voi
    done = {}

    def cap(wb, view, gmask, feasible, net_cost, ctx, goal, cands, e_a, e_g, generator, **kw):
        if done:
            return fn(wb, view, gmask, feasible, net_cost, ctx, goal, cands, e_a, e_g, generator, **kw)
        gstate = generator.get_state().clone() if generator is not None else None
        fast0 = pred.fast.clone()
        ref0 = pred._ref.clone() if pred._ref is not None else None
        out = fn(wb, view, gmask, feasible, net_cost, ctx, goal, cands, e_a, e_g, generator, **kw)
        torch.save({"wb": wb, "view": view, "gmask": gmask, "feasible": feasible, "net_cost": net_cost, "ctx": ctx,
                    "goal": goal, "cands": cands, "e_a": e_a, "e_g": e_g, "gstate": gstate, "fast": fast0,
                    "ref": ref0, "out": out}, path)
        done["x"] = True
        return out
    pred._voi = cap


def replay_voi(agent, path):
    d = torch.load(path, weights_only=False)
    pred = agent.predictor
    keep = (pred.fast, pred._ref)
    pred.fast, pred._ref = d["fast"].clone(), d["ref"]
    g = torch.Generator()
    g.set_state(d["gstate"])
    t0 = time.perf_counter()
    voi, info = pred._voi(d["wb"], d["view"], d["gmask"], d["feasible"], d["net_cost"], d["ctx"], d["goal"],
                          d["cands"], d["e_a"], d["e_g"], g)
    dt = time.perf_counter() - t0
    pred.fast, pred._ref = keep
    v0, i0 = d["out"]
    rep = {"seconds_new": dt, "max_abs_diff_voi": float((voi - v0).abs().max()),
           "max_abs_voi": float(v0.abs().max())}
    for k in i0:
        rep[f"max_abs_diff_{k}"] = float((info[k] - i0[k]).abs().max())
        rep[f"max_abs_{k}"] = float(i0[k].abs().max())
    return rep


def install(prof, agent, env, verifier, rt):
    for m in ("ingest", "reconcile", "update_beliefs", "propose", "evaluate", "select", "record_result",
              "internal_rewards", "learn", "meta_learn", "save", "promote", "retention_metrics", "_measure_planner",
              "_lp_before", "_lp_after", "_fitted_digest", "_digest", "_feasible", "policy_distribution", "_belief",
              "_items", "_vocab", "_skill_candidates", "_annotate_verified_delta", "_annotate_epistemic_after"):
        prof.wrap(agent, m, f"agent.{m}")
    prof.wrap(agent.proposer, "propose", "proposer.propose")
    prof.wrap(agent.proposer, "prompt", "proposer.prompt")
    prof.wrap(agent.model, "generate", "model.generate")
    for m in ("tokenize", "core_forward", "encode_actions", "encode_goals", "encode_queries", "decision_latent"):
        prof.wrap(agent.model, m, f"model.{m}")
    bb = getattr(agent.model, "backbone", None)
    if bb is not None:
        for m in ("embed", "_encode", "generate"):
            prof.wrap(bb, m, f"backbone.{m}")
    for m in ("evaluate_candidates", "_voi", "_slow_views", "observe_outcomes", "resolve", "belief_value"):
        prof.wrap(agent.predictor, m, f"predictor.{m}")
    for m in ("select", "propose_experiment", "select_experiment"):
        prof.wrap(agent.decider, m, f"decider.{m}")
    prof.wrap(agent.planner, "plan", "planner.plan")
    for m in ("poll", "learn", "validate"):
        prof.wrap(agent.learner, m, f"learner.{m}")
    for m in ("internal_rewards", "policy_update", "meta_update", "refresh_selection"):
        prof.wrap(agent.meta, m, f"meta.{m}")
    for m in ("execute", "observe", "discover_tools", "validate_action", "poll_events"):
        prof.wrap(env, m, f"env.{m}")
    for m in ("verify", "measure"):
        prof.wrap(verifier, m, f"verifier.{m}")
    for m in ("_save_state", "_decide_and_act", "_measure", "_verify", "_ingest", "_journal", "_commit",
              "_drain_agent_records", "_process_approvals", "_execute", "_complete"):
        prof.wrap(rt, m, f"runtime.{m}")
    for p in agent.learner.providers:
        if hasattr(p, "loss"):
            prof.wrap(p, "loss", f"provider.{p.name}.loss")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", default="SW:rle")
    ap.add_argument("--decisions", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--lock-timeout", type=float, default=7200.0)
    ap.add_argument("--lock-poll", type=float, default=0.002)
    ap.add_argument("--voi-replay", default="")
    ap.add_argument("--lock-held-by-parent", action="store_true", help="the parent process holds the GPU lock")
    ap.add_argument("--hold-s", type=float, default=840.0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    from cam_live.eval.conditions import _CLASSES
    from cam_live.eval.oe_runner import substantive_config
    from cam_live.eval.openended import OpenEndedComputer
    from cam_live.nn.gpu import gpu_lock
    from cam_live.runtime.loop import LiveRuntime, RuntimeConfig
    from cam_live.validation import OffsetClock
    shutil.rmtree(a.root, ignore_errors=True)
    os.makedirs(a.root)
    fam, prob = a.plan.split(":", 1)
    out = {"plan": a.plan, "decisions": a.decisions, "real": not a.cpu, "steps": []}
    lock = contextlib.nullcontext() if (a.cpu or a.lock_held_by_parent) else \
        gpu_lock(min_free_gb=12.0, timeout_s=a.lock_timeout, poll_s=a.lock_poll)
    with lock:
        t_lock = time.time()
        clock = OffsetClock()
        cfg = substantive_config("E", 0, real=not a.cpu)
        agent = _CLASSES["E"](cfg, os.path.join(a.root, "agent"), clock=clock)
        out["build_s"] = time.time() - t_lock
        oc = OpenEndedComputer(os.path.join(a.root, "computer"), clock)
        wseed = int(prob) * 100 if fam == "MECH" else 1000
        env, policy, verifier, task = oc.open(fam, prob, wseed, "obj_profile", max_decisions=a.decisions)
        out["unavailable"] = {t.name: t.unavailable_reason for t in env.discover_tools() if not t.available}
        agent.attach_verifier(verifier)
        rt = LiveRuntime(agent, env, policy, verifier, task, os.path.join(a.root, "obj"), clock,
                         RuntimeConfig(fsync=False, candidates_k=10, keep_checkpoints=1, verify_every_s=0.0,
                                       checkpoint_every_s=1e12, journal_every=10 ** 9, journal_min_interval_s=1e12))
        prof = Prof()
        if a.voi_replay and os.path.exists(a.voi_replay):
            out["voi_replay"] = replay_voi(agent, a.voi_replay)
        capture_voi(agent, os.path.join(a.root, "voi_case.pt"))
        install(prof, agent, env, verifier, rt)
        rt.start()
        n = 0
        t0 = time.time()
        while rt.status == "running" and n < 3 * a.decisions + 6:
            if time.time() - t_lock > a.hold_s:
                rt._finish("stopped", "profile hold budget")
                break
            s0 = time.perf_counter()
            rep = rt.step()
            prof.cur["runtime.step"] += time.perf_counter() - s0
            ch = agent.last.get("proposal") or {}
            prof.end_step({"acted": bool(rep.get("acted")), "cycle": bool(rep.get("cycle")),
                           "n_decisions": agent.n_decisions, "reasons": rep.get("reasons"),
                           "prompt_tokens": (ch.get("stats") or {}).get("prompt_tokens")})
            n += 1
        rt.close()
        out["wall_s"] = time.time() - t0
        out["steps"] = prof.steps
        recs = [r for r in agent.store.trajectory(task.objective_id) if r.decision is not None]
        out["chosen"] = [{"op": r.decision.chosen.operation, "prov": r.decision.chosen.provenance,
                          "status": r.result.status if r.result else None,
                          "args": json.dumps(r.decision.chosen.arguments, default=str)[:300]} for r in recs]
        out["verifier_history"] = getattr(verifier, "history", [])[-10:]
        out["lock_hold_s"] = time.time() - t_lock
        out["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
        oc.close()
        agent.close()
    # per-decision summary: steps that acted
    acted = [s for s in out["steps"] if s["info"]["acted"]]
    keys = sorted({k for s in out["steps"] for k in s["seconds"]})
    out["summary"] = {"acted_steps": len(acted),
                      "mean_per_acted_step": {k: sum(s["seconds"].get(k, 0.0) for s in acted) / max(1, len(acted))
                                              for k in keys},
                      "total": {k: sum(s["seconds"].get(k, 0.0) for s in out["steps"]) for k in keys}}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    top = sorted(out["summary"]["mean_per_acted_step"].items(), key=lambda kv: -kv[1])[:30]
    print(json.dumps({"acted": len(acted), "wall_s": out["wall_s"], "top": top}, indent=0, default=str))


if __name__ == "__main__":
    sys.exit(main())
