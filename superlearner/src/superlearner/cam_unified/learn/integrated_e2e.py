"""Real end-to-end check of the integrated learner (plan 2026-11-26 §3 W7): the real `dom_dev` stack (8 x 384 core,
frozen Qwen3-0.6B, every real component from `experiments.stack.build_stack`), 4 Crucible 5.0 rapid training-family
worlds through the observation-only `envs.experience` adapter, `IntegratedLearner.step()` every cfg.update_every
rounds for `n_updates` updates. No test double anywhere.

The loop is the canonical round of plan §2 (act -> execute -> advance -> merge result observations by seq -> observe
-> learner.on_step -> labels to the LabelStore -> learner.step every update_every rounds); it stands in for W1's
`runtime.run_lifetime` until that lands, and passes the agent zero rewards (its observe() needs floats) and the
learner `rewards=None` - no environment reward exists on this channel (ExperienceAdapter has none).

    PYTHONPATH=src:../crucible/src python -m superlearner.cam_unified.learn.integrated_e2e --updates 20 \\
        --out docs/evidence/dom/W7/w7_e2e_dom_dev.json

Checks (plan §3 W7): actor share of the core gradient >= 0.1 and world share <= 0.7 (every probe after the
warmup probes; with the learner's share guard on (default) every update is probed and the share is that of the
APPLIED core gradient, the pre-guard share is reported as core_shares_measured); DOM-08 table nonzero after one optimiser step for the required groups (absent groups reported as
absent = not built yet); lag>0 decisions routed to V-trace (n_stale > 0) without StalePolicyError; per-group clip
rate < 0.5 after warmup.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch

FAMILIES = ("dl_delayed_returns", "hc_confounded", "net_routing", "mk_drift")
DOM08_REQUIRED = None


def _commit() -> str:
    """HEAD at the start of the run, suffixed "+dirty" when tracked files under learn/ differ from it."""
    here = os.path.dirname(__file__)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "."], cwd=here) != 0
        return sha + ("+dirty" if dirty else "")
    except Exception:                                          # pragma: no cover
        return "unknown"


def run(n_updates: int = 20, seed: int = 0, device=None, run_dir: str = "/tmp/w7_e2e", families=FAMILIES,
        out: str = None, overrides: dict = None, max_rounds: int = None, preset: str = "dom_dev") -> dict:
    from ..config import make_preset
    from ..envs import experience as X
    from ..envs.agent_protocol import Transition
    from ..envs.labels import LabelStore, from_outcome_label
    from ..experiments.stack import build_stack
    from ..interfaces import ObservationBatch
    from ..runtime import _max_seq, _merge
    from ..selfeval.reward_model import RewardModel
    from .grad_probe import DOM08_REQUIRED as REQ
    from .integrated import IntegratedLearner
    t_start = time.time()
    commit = _commit()
    over = dict(microbatch=0, tool_entropy_coef=0.01)
    over.update(overrides or {})
    guard_on = bool(over.pop("share_guard", True))
    mb = int(over["microbatch"])
    over["declared_conditions"] = (
        "w7-e2e:interim-loop(runtime.run_lifetime pending, W1)",
        f"w7-e2e:microbatch={mb}" + ("(full-batch policy recompute: exact probe)" if not mb else
                                     "(probe actor/value norms estimated from microbatch 0)"),
        f"w7-e2e:tool_entropy_coef={over['tool_entropy_coef']}(dom presets leave it 0)")
    if over.get("reward_mode") not in (None, "observation_only"):    # a labelled, non-compliant Exp D baseline
        over["declared_conditions"] += (f"exp-d:{over['reward_mode']}(labelled baseline)",)
    cfg = make_preset(preset, **over)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    # a fresh directory per run: the archive and the label store persist on disk, and another run's experience
    # must never enter this run's replay or label join
    run_dir = os.path.join(run_dir, f"run-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-s{seed}")
    os.makedirs(run_dir, exist_ok=False)
    st = build_stack(cfg, run_dir=os.path.join(run_dir, "stack"), seed=seed, device=dev, train=False)
    agent, model = st.agent, st.model
    specs = [X.WorldSpec(1, f"rapid:{f}", seed + i) for i, f in enumerate(families)]
    op = X.ExperienceOperator(specs, compute_references=True)
    env = op.adapter()
    labels = LabelStore(os.path.join(run_dir, "labels"))
    rm = RewardModel(cfg)
    learner = IntegratedLearner(model, rm, labels, cfg, agent=agent, seed=seed, share_guard=guard_on,
                                log_path=os.path.join(run_dir, "updates.jsonl"), probe_every=5)
    think = cfg.think_s / 86400.0
    B = env.B
    ob = env.reset()
    agent.reset(ob.env_ids)
    agent.set_goals(env.goals())
    seen = [_max_seq(o) for o in ob.observations]
    cursors = list(seen)
    updates, rounds, n_labels, episodes, n_dec = [], 0, 0, 1, 0
    errors = []
    max_rounds = max_rounds or (n_updates * cfg.update_every * 4)
    while len(updates) < n_updates and rounds < max_rounds:
        tools = env.available_tools()
        actions = agent.act(ob, tools)
        results = env.execute(actions)
        dur = [((r.duration if r is not None else 0.0) + think) if a is not None else 0.0
               for a, r in zip(actions, results)]
        adv = env.advance_time(dur)
        events, cursors = env.collect_events(cursors)
        nxt = []
        for b in range(B):
            ro = results[b].observations if results[b] is not None else []
            merged, seen[b] = _merge(seen[b], ro, adv.observations[b], events[b])
            nxt.append(merged)
        next_ob = ObservationBatch(list(ob.env_ids), nxt, list(adv.t), list(adv.done))
        recs = agent.observe(results, [0.0] * B, next_ob)          # observation-only: no reward exists
        by_env = {}
        for r in recs:
            if r.action is not None:
                by_env[r.env_id] = r
        n_dec += sum(a is not None for a in actions)
        learner.on_step(Transition([by_env.get(e) for e in ob.env_ids], next_ob, None, list(adv.done), dur))
        for lab in op.release_labels():
            rec = from_outcome_label(lab, evaluator="ExperienceOperator")      # None while pending
            if rec is not None and labels.append(rec):
                n_labels += 1
        rounds += 1
        if rounds % cfg.update_every == 0:
            info = learner.step()
            if info.get("updated"):
                updates.append(info)
        if all(next_ob.done):
            ob = env.reset()
            agent.reset(ob.env_ids)
            agent.set_goals(env.goals())
            seen = [_max_seq(o) for o in ob.observations]
            cursors = list(seen)
            episodes += 1
        else:
            ob = next_ob
    rep = summarise(updates, REQ, warm=learner.probe_warmup)
    rep["clip_rate_all"] = learner.trainer.clip_rate()
    rep.update({"commit": commit, "preset": preset, "device": str(dev), "families": list(families),
                "seed": seed, "run_dir": run_dir, "rounds": rounds, "decisions": n_dec, "episodes": episodes, "labels": n_labels,
                "n_updates": len(updates), "wall_s": time.time() - t_start, "errors": errors,
                "share_guard_on": learner.share_guard,
                "param_counts": model.count_parameters(), "reward_model": type(rm).__name__ +
                (" (seam stub: reward 0)" if getattr(rm, "stub", False) else ""),
                "cfg": {k: v for k, v in cfg.to_dict().items() if k in (
                    "reward_mode", "ppo_epochs", "target_kl", "clip_mode", "clip_median_mult", "balance_every",
                    "world_core_share_max", "lr_core", "lr_heads", "lr_psi", "lr_warmup_updates", "update_every",
                    "microbatch", "n_layers", "d_model", "sem_model", "declared_conditions")},
                "peak_gpu_gb": (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else None,
                "command": " ".join(sys.argv)})
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(rep, fh, indent=1, default=str)
    return rep


def summarise(updates: list, required, warm: int = 8) -> dict:
    """The W7 checks over the UpdateInfo dicts of a run (`warm` = the learner's probe_warmup: share and clip-rate
    checks use the probes / updates after it; the final probe is reported separately)."""
    probes = [u for u in updates if u.get("grad_table") is not None]
    keys = ("world", "adapt", "actor", "value", "pred")
    # core_share = shares of the APPLIED core gradient (after the learner's per-update share guard, when on);
    # core_share_measured = at the balanced weights before the guard (what the balancer factors alone achieve)
    shares = [{"update": u["update"], **{k: u["core_share"].get(k) for k in keys}} for u in probes]
    measured = [{"update": u["update"], **{k: (u.get("core_share_measured") or u["core_share"]).get(k)
                                           for k in keys}} for u in probes]
    guarded = [u for u in probes if u.get("share_guard")]
    guard = {"updates_probed": len(probes), "updates_guarded": len(guarded),
             "updates_scaled": sum(1 for u in guarded if min(u["share_guard"]["scale"].values()) < 1.0),
             "unsatisfied_updates": [u["update"] for u in guarded if not u["share_guard"]["satisfied"]],
             "min_scale": {k: min(u["share_guard"]["scale"].get(k, 1.0) for u in guarded) for k in keys
                           if any(k in u["share_guard"]["scale"] for u in guarded)} if guarded else {},
             "measured_actor_after_warmup_min": min(((m["actor"] or 0.0) for m in measured if m["update"] > warm),
                                                    default=None),
             "measured_adapt_after_warmup_max": max(((m["adapt"] or 0.0) for m in measured if m["update"] > warm),
                                                    default=None)}
    after = [s for s in shares if s["update"] > warm]
    final = shares[-1] if shares else None
    actor_ok = bool(after) and all((s["actor"] or 0.0) >= 0.1 for s in after)
    world_ok = bool(after) and all((s["world"] or 0.0) <= 0.7 for s in after)
    first_after_step = next((u for u in probes if u["update"] >= 2), None)
    dom08 = first_after_step["dom08"] if first_after_step else {}
    dom08_present_ok = bool(dom08) and all(v == "nonzero" for v in dom08.values() if v != "absent")
    dom08_all_ok = bool(dom08) and all(v == "nonzero" for v in dom08.values())
    stale = sum(int((u.get("policy") or {}).get("n_stale_steps", 0)) for u in updates)
    fresh = sum(int((u.get("policy") or {}).get("n_fresh_steps", 0)) for u in updates)
    pairs = clipped = 0
    for u in updates:
        if u["update"] <= warm:
            continue
        for g, c in (u.get("clip") or {}).items():
            pairs += 1
            clipped += int(bool(c.get("clipped")))
    clip_rate = clipped / pairs if pairs else None
    return {
        "checks": {
            "actor_core_share_ge_0.1": {"passed": actor_ok, "values": [s["actor"] for s in after],
                                        "updates": [s["update"] for s in after],
                                        "final": None if final is None else final["actor"]},
            "world_core_share_le_0.7": {"passed": world_ok, "values": [s["world"] for s in after],
                                        "updates": [s["update"] for s in after],
                                        "final": None if final is None else final["world"]},
            "dom08_nonzero_present_groups": {"passed": dom08_present_ok, "table": dom08,
                                             "update": first_after_step["update"] if first_after_step else None},
            "dom08_nonzero_all_required": {"passed": dom08_all_ok,
                                           "absent": sorted(k for k, v in dom08.items() if v == "absent"),
                                           "zero": sorted(k for k, v in dom08.items() if v == "zero")},
            "stale_routed_to_vtrace": {"passed": stale > 0, "n_stale_steps": stale, "n_fresh_steps": fresh},
            "clip_rate_lt_0.5_after_warmup": {"passed": clip_rate is not None and clip_rate < 0.5,
                                              "rate": clip_rate, "pairs": pairs, "warmup_updates": warm},
        },
        "core_shares": shares,
        "core_shares_measured": measured,
        "share_guard": guard,
        "balance": [{"update": u["update"], **(u.get("balance") or {})} for u in updates if u.get("balance")],
        "per_update": [{"update": u["update"], "losses": u["losses"], "weights": u["weights"],
                        "grad_norm": u["grad_norm"], "clipped": u["clipped"],
                        "policy": {k: (u.get("policy") or {}).get(k) for k in (
                            "n_fresh_steps", "n_stale_steps", "kl", "max_lag", "n_nodes", "n_segments", "entropy",
                            "tool_entropy", "rerouted_by_kl", "vtrace_rho_mean", "clip_frac", "n_fresh_rows",
                            "n_stale_rows", "logp_maxdev_lag0", "logp_meandev_lag0")},
                        "weights_applied": u.get("weights_applied"), "share_guard": u.get("share_guard"),
                        "epochs": u.get("epochs"), "reward": u.get("reward"), "wall_s": u.get("wall_s"),
                        "actor_psi_grad_norm": u.get("actor_psi_grad_norm")} for u in updates],
        "grad_tables": {str(u["update"]): {"norm": u["grad_table"]["norm"], "core_share": u["core_share"],
                                           "dom08": u["dom08"], "absent": u["grad_table"]["absent"]}
                        for u in probes},
        "integrity": updates[-1].get("integrity") if updates else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-dir", default="/tmp/w7_e2e")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preset", default="dom_dev", help="dom_dev (W7 check) | dom (1B, W1 only) | dom_150m")
    ap.add_argument("--microbatch", type=int, default=0, help="cfg.microbatch (0 = full-batch policy recompute)")
    ap.add_argument("--share-guard", default="on", choices=("on", "off"),
                    help="per-update core-share guard of the learner (off = factors-only balancing, run 4)")
    ap.add_argument("--reward-mode", default=None,
                    help="Exp D condition: observation_only (default of the preset) | sparse_verified")
    a = ap.parse_args(argv)
    over = {"microbatch": a.microbatch, "share_guard": a.share_guard == "on"}
    if a.reward_mode:
        over["reward_mode"] = a.reward_mode
    rep = run(a.updates, a.seed, a.device, a.run_dir, out=a.out, preset=a.preset, overrides=over)
    print(json.dumps(rep["checks"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
