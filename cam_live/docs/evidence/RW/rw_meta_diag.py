"""RW engineering diagnostic (debug scale, CPU): does R_psi learn, within ONE long LabWorld objective, to reward what the
verifier later confirms — and does the validity evidence then hand it weight in the live updates?

    cd /workspace/camlive && PYTHONPATH=cam_live/src .venv/bin/python cam_live/docs/evidence/RW/rw_meta_diag.py \\
        --arm pooled --seeds 0 1 2 --out cam_live/docs/evidence/RW/diag_pooled.json

``MetaLearner`` on the debug model (``reward.diagnose.make_stack``: the controlled configuration that isolates R_psi —
fixed inner step, no matured replay, no goal / model term, so the live update learns only from the mixed reward),
acting with its own live policy in one LabWorld objective of ``--decisions`` decisions (target never reached, budget
and deadline far beyond the run). After every decision: the reward step, a live policy update every
``--update-every`` decisions, the verified-progress annotation of every decision whose attributed outcomes are all
final (the sum of the verifier's readings attributed to it; 0 when it has none — what the agent writes), and a
meta-update attempt. Arms: ``pooled`` (estimator ``pooled_first_order``), ``query`` (the post-update query windows),
``frozen`` (pooled, reward step 0: R_psi stays at its initialisation). Same seeds, same worlds.

Ground truth (never visible to the learner): the expected adopters of the chosen action, rate x effort at execution.
Not evidence about any real task; records are ``provenance="diagnostic"``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

DAY = 86400.0


def run(seed: int, arm: str, n_decisions: int, update_every: int, world_seed: int, mix_step: float,
        utility: str = "return_to_go", half_life_s: float = DAY, lr_reward: float = 0.0) -> dict:
    import torch
    from cam_live.diagnostics.collect import LabCollector
    from cam_live.diagnostics.world import LabConfig, LabWorld, lab_task
    from cam_live.eval.envs import TruthTap
    from cam_live.reward import diagnose as dg
    from cam_live.reward.assemble import STORED_KEY
    import dataclasses
    t0 = time.time()
    auth = dg.LabMeasurementAuth()
    ec = dg.ExperimentConfig(min_independent=4)
    if lr_reward > 0:
        ec.lr_reward = float(lr_reward)              # default: the stage-1 reference's 3e-3 (the agent's is 1e-4)
    if arm == "frozen":
        ec.lr_reward = 0.0
    est = "query" if arm == "query" else "pooled_first_order"
    restore = None
    if arm == "drift":
        # an R_psi that moves without learning (G3-core D-RW1 control): every applied outer step keeps its norm but
        # gets a random direction (seeded), so psi changes as much as when it learns, carrying no outcome information
        import cam_live.reward.metalearner as mlm
        from cam_live.reward.meta import MetaGradient, MetaGradResult
        real_apply = MetaGradient.apply
        g_rng = torch.Generator().manual_seed(90_000 + seed)

        def drift_apply(opt, reward, result, clip=None):
            if result.supported and result.grad_norm > 0:
                rnd = {k: torch.randn(v.shape, generator=g_rng).to(v) for k, v in result.grads.items()}
                n = math.sqrt(sum(float((x ** 2).sum()) for x in rnd.values()))
                rnd = {k: x * (result.grad_norm / n) for k, x in rnd.items()}
                result = dataclasses.replace(result, grads=rnd)
            return real_apply(opt, reward, result, clip=clip)
        mlm.MetaGradient.apply = staticmethod(drift_apply)
        restore = (mlm.MetaGradient, real_apply)
    model, store, clock, ml = dg.make_stack(seed, ec, mix=1.0, mix_step=mix_step, check_measurement=auth,
                                            estimator=est, utility=utility, half_life_s=half_life_s)
    lab = dict(ec.lab)
    lab.update(horizon_s=120 * DAY, budget=5000.0, target=100_000)
    world = TruthTap(LabWorld(LabConfig(seed=world_seed + seed, **lab), clock=clock))
    gen = torch.Generator().manual_seed(50_000 + seed)
    act = dg.model_policy(ml, store, gen)
    buf, rows, metas, mixes, pending = [], [], [], [], []
    ver = dg.LabVerifier(world._env)
    ts_list = []

    def annotate() -> None:
        keep = []
        for r in pending:
            rec = store.get(r)
            if rec is None:
                continue
            if rec.pending_queries:
                keep.append(r)
                continue
            y = sum(float(o.value) for o in rec.outcomes if o.verified and o.value is not None
                    and str(o.query_id) in {q.query_id for q in rec.result.expected_outcomes
                                            if q.metric.startswith("lab.adopters.from.")})
            rec.decision = dataclasses.replace(rec.decision, diagnostics={**(rec.decision.diagnostics or {}),
                                                                           STORED_KEY: {"verified_progress_delta": y}})
            store.update(rec)
        pending[:] = keep

    def on_step(rec) -> None:
        auth.register(rec, ver)
        truth = world.rows[-1].get("expected_adopters", 0.0) if world.rows else 0.0
        mix_used = ml.mixer.mix
        ml.internal_rewards([rec])
        pending.append(rec.exp_id)
        buf.append(rec)
        if update_every and len(buf) >= update_every:
            ml.policy_update(list(buf))
            buf.clear()
        annotate()
        m = ml.meta_update()
        metas.append({"i": len(rows), "applied": m.applied, "reason": (m.skipped_reason or "").split(":")[0]})
        a_ = rec.decision.chosen
        rows.append({"i": len(rows), "op": a_.operation, "truth": truth, "mix": mix_used,
                     "act": f"{a_.operation}:{json.dumps(a_.arguments, sort_keys=True, default=str)}"})
        ts_list.append(float(rec.decision.ts))
        mixes.append(ml.mixer.mix)
    task = lab_task(world._env, objective_id=f"obj_rw_diag_{seed}")
    LabCollector(store, max_context=ec.max_context).run(world, act, n_decisions, seed=seed, on_step=on_step,
                                                         task=task, finish=False)
    annotate()
    ml.refresh_validity()
    if restore is not None:
        restore[0].apply = staticmethod(restore[1])
    val = ml.validity.summary()
    rec_pairs = list(ml.validity.recent)
    thirds = []
    k = max(1, len(rows) // 3)
    for j in range(3):
        seg = rows[j * k:(j + 1) * k] if j < 2 else rows[2 * k:]
        thirds.append(sum(r["truth"] for r in seg) / max(1, len(seg)))
    conc = [p["scores"][-1] for p in rec_pairs]
    ck = max(1, len(conc) // 3)
    conc_thirds = [sum(conc[j * ck:(j + 1) * ck] if j < 2 else conc[2 * ck:]) /
                   max(1, len(conc[j * ck:(j + 1) * ck] if j < 2 else conc[2 * ck:])) for j in range(3)]
    from collections import Counter
    gaps = [b_ - a_ for a_, b_ in zip(ts_list, ts_list[1:])]
    return {"seed": seed, "arm": arm, "estimator": est, "utility": utility, "half_life_s": half_life_s,
            "lr_reward": ec.lr_reward,
            "median_decision_gap_s": sorted(gaps)[len(gaps) // 2] if gaps else None, "decisions": len(rows),
            "seconds": time.time() - t0,
            "meta_applied": sum(1 for m in metas if m["applied"]),
            "meta_skips": dict(Counter(m["reason"] for m in metas if not m["applied"])),
            "mix_changes": [(i, a, b) for i, (a, b) in enumerate(zip([1.0] + mixes[:-1], mixes)) if a != b],
            "mix_end": ml.mixer.mix, "label_end": ml.mixer.label, "validity": val,
            "learned_concordance_by_third_of_pairs": conc_thirds,
            "truth_chosen_by_third": thirds, "truth_chosen_mean": sum(r["truth"] for r in rows) / max(1, len(rows)),
            "ops": dict(Counter(r["op"] for r in rows)),
            "rows": [[r["truth"], r["mix"], r["act"]] for r in rows],
            "validity_e": {m: {"e_learned_vs_mix": v["learned_minus_mix"]["e"],
                               "e_bootstrap_vs_mix": v["bootstrap_minus_mix"]["e"]}
                           for m, v in val.get("by_mix", {}).items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("pooled", "query", "frozen", "drift"), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--decisions", type=int, default=120)
    ap.add_argument("--update-every", type=int, default=6)
    ap.add_argument("--world-seed", type=int, default=41_000)
    ap.add_argument("--mix-step", type=float, default=0.25)
    ap.add_argument("--utility", choices=("return_to_go", "direct"), default="return_to_go")
    ap.add_argument("--half-life-s", type=float, default=DAY, help="inner-loss return half-life (InnerConfig)")
    ap.add_argument("--lr-reward", type=float, default=0.0, help="reward optimizer step (0: ExperimentConfig's)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = {"kind": "debug_rw_meta_diag", "arm": a.arm, "utility": a.utility, "half_life_s": a.half_life_s,
           "lr_reward_override": a.lr_reward,
           "decisions": a.decisions,
           "update_every": a.update_every, "runs": []}
    for s in a.seeds:
        r = run(s, a.arm, a.decisions, a.update_every, a.world_seed, a.mix_step, a.utility, a.half_life_s,
                a.lr_reward)
        out["runs"].append(r)
        print(json.dumps({k: r[k] for k in ("seed", "arm", "median_decision_gap_s", "decisions", "meta_applied",
                                             "mix_changes", "mix_end",
                                             "learned_concordance_by_third_of_pairs", "truth_chosen_by_third",
                                             "seconds")}), flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
