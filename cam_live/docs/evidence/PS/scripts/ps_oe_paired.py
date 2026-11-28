"""PS full-size comparison, both arms in the same GPU-lock hold: the substantive protocol of ``eval.oe_runner``
(Qwen3-4B + 50m core, open-ended generation on, condition E, ``substantive_config``, ``run_objective``), with the
persistent core state on and off, objective by objective. One backbone is loaded per hold and shared by the two arms'
models; each arm's agent is built and restored from its own checkpoint (which carries its own LoRA / bridge weights:
the shared backbone's adapters are reset to the initial values before an arm's first objective and loaded from the
arm's checkpoint before later ones), so the arms differ only in ``persistent_state.enabled``. Optionally the first
hold also runs ``ps_compat_live`` on the same backbone (``--compat``).

Per decision of the ``on`` arm, the causal effect of the carried state on that decision is logged (see ``ps_oe``).

    python ps_oe_paired.py --plan SW:rle,MECH:7 --objective 0 --decisions 12 --root ROOT --out OUT.json [--cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Dict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ps_oe import measure  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="SW:rle,MECH:7")
    ap.add_argument("--objective", type=int, required=True, help="index into the plan run in this hold")
    ap.add_argument("--decisions", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unroll", type=int, default=2)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold-s", type=float, default=840.0)
    ap.add_argument("--arms", default="on,off")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-skills", dest="no_skills", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from cam_live.agent import CamLiveAgent
    from cam_live.eval import oe_runner
    from cam_live.nn.model import CamLiveModel, build_model
    from cam_live.validation import OffsetClock
    plan = oe_runner._plan(a.plan)
    i = a.objective
    fam, prob = plan[i]
    root = os.path.abspath(a.root)
    os.makedirs(root, exist_ok=True)
    res: Dict[str, Any] = json.load(open(a.out)) if os.path.exists(a.out) else {"plan": a.plan, "seed": a.seed,
                                                                                   "arms": {}}
    lock = None
    if not a.cpu:
        from cam_live.nn.gpu import gpu_lock
        lock = gpu_lock(min_free_gb=14.0, timeout_s=4 * 3600.0)
        lock.__enter__()
    t_lock = time.time()
    try:
        bb, init_adapters = None, None
        arms = a.arms.split(",")
        for j, arm in enumerate(arms):
            cfg = oe_runner.substantive_config("E", a.seed, real=not a.cpu)
            cfg.sections["persistent_state"] = {"enabled": arm == "on", "unroll": int(a.unroll)}
            if bb is None:
                model = build_model(cfg)
                bb = model.backbone
                init_adapters = {n: p.detach().clone() for n, p in bb.named_parameters() if p.requires_grad}
            else:
                model = CamLiveModel(cfg, backbone=bb)
            with torch.no_grad():                      # every arm starts from the initial adapters (checkpoint below)
                for n, p in bb.named_parameters():
                    if n in init_adapters:
                        p.copy_(init_adapters[n])
            bb.mark_adapters_updated()
            aroot = os.path.join(root, arm)
            clock = OffsetClock()
            cs = os.path.join(aroot, "clock.json")
            if os.path.exists(cs):
                clock.offset = json.load(open(cs))["offset"]
            agent = CamLiveAgent(cfg, os.path.join(aroot, "agent"), clock=clock, model=model)
            ck = os.path.join(aroot, "ckpt_latest")
            if i > 0:
                agent.load(ck)
            log_path = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"ps_causal_{arm}.jsonl")
            if agent.persist is not None:
                orig = agent.select

                def select(evaluations: Any, _orig=orig, _agent=agent, _log=log_path) -> Any:
                    try:
                        rec = measure(_agent, evaluations)
                        if rec is not None:
                            with open(_log, "a") as fh:
                                fh.write(json.dumps(rec) + "\n")
                    except Exception as e:  # noqa: BLE001 — a measurement; never changes the decision
                        with open(_log, "a") as fh:
                            fh.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
                    return _orig(evaluations)
                agent.select = select
            ns = argparse.Namespace(no_skills=a.no_skills)
            computer = oe_runner.make_computer(aroot, clock, ns)
            left = a.hold_s - (time.time() - t_lock) - 60.0
            budget = left / (len(arms) - j)
            k = i - sum(1 for f, _ in plan[:i] if f == "GOALFREE")
            wseed = 1000 * (k + 1) + a.seed
            if fam == "MECH" and prob:
                wseed = int(prob) * 100 + a.seed
            t0 = time.time()
            ob = oe_runner.run_objective(agent, computer, fam, prob, wseed, f"obj_{fam}_{prob}_{i}_s{a.seed}",
                                         os.path.join(aroot, f"obj{i:02d}"), clock, max_decisions=a.decisions,
                                         budget_s=budget)
            ob.update(index=i, arm=arm, budget_s=budget, arm_wall_s=time.time() - t0,
                      persistent_state=agent.status().get("persistent_state"),
                      restore_report_persistent=(agent.restore_report or {}).get("persistent_state")
                      if i > 0 else None)
            tmp = ck + ".new"
            shutil.rmtree(tmp, ignore_errors=True)
            agent.save(tmp)
            shutil.rmtree(ck, ignore_errors=True)
            os.rename(tmp, ck)
            json.dump({"offset": clock.offset}, open(cs, "w"))
            computer.close()
            agent.close()
            del agent, model
            res["arms"].setdefault(arm, {})[str(i)] = ob
            res["hold_s"] = time.time() - t_lock
            with open(a.out, "w") as fh:
                json.dump(res, fh, default=str)
            print(f"[{arm}] objective {i} {fam}:{prob} {ob['verified_status']} value {ob['verified_value']:.3f} "
                  f"decisions {ob['decisions']} nll {ob['mean_prediction_nll']} wall {ob['arm_wall_s']:.0f}s",
                  flush=True)
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
