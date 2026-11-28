"""cam_live.predict.imp_eval — does imagination causally change CAM's decisions, and do they improve? (IMP)

    python -m cam_live.predict.imp_eval run --arm on|off --seed S [--family T1] [--lifetimes 6] [--decisions 16]
                                            [--real] [--backbone] [--threads 2] --out <json>
    python -m cam_live.predict.imp_eval summarize <json> ... --out <json>

One *run* is the integrated agent (``cam_live.agent.CamLiveAgent``, condition E of ``eval.conditions``) driven
through the real persistent runtime (``eval.runner.run_lifetime`` -> ``runtime.loop.LiveRuntime``, the family's
independent verifier, ``eval.envs.TruthTap`` for the ground truth the agent never sees) for a sequence of lifetimes.
The two arms differ in exactly one switch: ``predict.imagination_select`` (on: rule ``thompson_policy`` adds the
planner's imagined preference with the weight drawn from the imagination calibration; off: weight 0). Both arms run
the planner, record its numbers and draw the same random numbers, so with the same seed they are identical until
imagination changes a choice. Per decision the on arm records the distribution with and without imagination
(``selection.imagination``); their total variation is the probability that imagination changed the executed action
under the maximal coupling of the two. ``summarize`` pairs the arms by seed.

Nothing here selects actions, sets weights or knows the task: it only builds the configuration, runs, and reads
what the agent recorded. LabWorld results are diagnostics (provenance "diagnostic"); ``--real`` uses Qwen3-4B with
the 50m core on the GPU inside the shared GPU lock.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

WORLD_BASE = {"T1": 71_000, "F1": 72_000}


def _code() -> Dict[str, Any]:
    root = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        head = ""
    return {"source": root, "git_head_of_source_dir": head}


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def decision_row(agent: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    """What the agent recorded for one decision (``DecisionTap`` row + the stored decision's diagnostics)."""
    rec = agent.store.by_decision(row["decision_id"])
    d = rec.decision if rec is not None else None
    diag = d.diagnostics if d is not None and isinstance(d.diagnostics, dict) else {}
    sel = diag.get("selection") if isinstance(diag.get("selection"), dict) else {}
    im = sel.get("imagination") if isinstance(sel.get("imagination"), dict) else None
    i = int(row["chosen_index"])
    cands = row["candidates"]
    truth = [c.get("truth") for c in cands]
    ch = cands[i] if 0 <= i < len(cands) else {}
    ex = (d.candidates[i].extras or {}) if d is not None and 0 <= i < len(d.candidates) else {}
    out = {"decision_id": row["decision_id"], "exploration": row["exploration"], "operation": ch.get("operation"),
           "arguments": ch.get("arguments"), "probability": row["probabilities"][i] if row["probabilities"] else None,
           "truth_chosen": _f(ch.get("truth")), "truth_best": max((_f(t) or 0.0 for t in truth), default=0.0),
           "planner_horizon": diag.get("planner_horizon"), "im_info_value": _f(ex.get("im_info_value")),
           "im_next": ex.get("im_next"), "im_cont": _f(ex.get("im_cont")),
           "ops_offered": sorted({c.get("operation") for c in cands})}
    if im is not None:
        w, wo = im.get("with_probabilities") or [], im.get("without_probabilities") or []
        out["imagination"] = {
            "enabled": im.get("enabled"), "share_tv": _f(im.get("share_tv")),
            "weight_mean": _f((im.get("weight") or {}).get("mean")),
            "weight_share_positive": _f((im.get("weight") or {}).get("share_positive")),
            "preference_chosen": _f(im.get("preference_chosen")),
            "argmax_with": max(range(len(w)), key=lambda k: w[k]) if w else None,
            "argmax_without": max(range(len(wo)), key=lambda k: wo[k]) if wo else None,
            "calibration": im.get("calibration")}
    return out


def build_config(arm: str, seed: int, *, real: bool, backbone: bool, extra_predict: Dict[str, Any]) -> Any:
    from ..eval.conditions import base_config, eval_config
    cfg = eval_config(base_config(real=real, seed=seed), "E", seed=seed)
    pred = dict(cfg.sections.get("predict", {}))
    pred["imagination_select"] = arm == "on"
    pred.update(extra_predict)
    cfg.sections["predict"] = pred
    if backbone:
        cfg.sections["agent"] = {**cfg.sections["agent"], "use_backbone": True}
    return cfg


def run(arm: str, seed: int, *, family: str, lifetimes: int, decisions: int, real: bool, backbone: bool,
        out: str, workdir: str, threads: int, world_base: Optional[int], extra_predict: Dict[str, Any],
        budget_s: float) -> Dict[str, Any]:
    from ..agent import CamLiveAgent
    from ..eval.runner import run_lifetime
    from ..validation import OffsetClock
    if threads:
        torch.set_num_threads(int(threads))
    t0 = time.time()
    root = os.path.abspath(workdir)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    cfg = build_config(arm, seed, real=real, backbone=backbone, extra_predict=extra_predict)
    clock = OffsetClock()
    agent = CamLiveAgent(cfg, os.path.join(root, "agent"), clock=clock)
    plan_s: List[float] = []
    orig_plan = agent.planner.plan

    def timed_plan(*a: Any, **kw: Any) -> Any:
        s = time.perf_counter()
        try:
            return orig_plan(*a, **kw)
        finally:
            plan_s.append(time.perf_counter() - s)
    agent.planner.plan = timed_plan
    res: Dict[str, Any] = {"arm": arm, "seed": seed, "family": family, "real": real, "backbone": backbone,
                           "decisions_per_lifetime": decisions, "code": _code(), "predict_overrides": extra_predict,
                           "backbone_name": cfg.backbone.name, "scale": cfg.core.scale, "lifetimes": []}
    wb = (WORLD_BASE.get(family, 70_000) + 100 * int(seed)) if world_base is None else int(world_base)
    try:
        for i in range(lifetimes):
            if budget_s and time.time() - t0 > budget_s:
                res["stopped_early"] = f"time budget {budget_s:.0f} s reached after {i} lifetimes"
                break
            n0 = len(plan_s)
            lt = run_lifetime(agent, family, wb + i, os.path.join(root, f"lt{i:02d}"), clock,
                              max_decisions=decisions, k=16)
            rows = [decision_row(agent, r) for r in lt["decision_rows"]]
            res["lifetimes"].append({
                "index": i, "world_seed": wb + i, "decisions": lt["decisions"], "progress": lt["progress"],
                "achieved": lt["achieved"], "verified_status": lt["verified_status"],
                "verified_value": lt["verified_value"], "target": lt["target"],
                "expected_adopters": lt.get("expected_adopters"),
                "oracle_expected_adopters": lt.get("oracle_expected_adopters"), "quality": lt.get("quality"),
                "wall_seconds": lt["wall_seconds"], "planner_seconds": sum(plan_s[n0:]),
                "planner_calls": len(plan_s) - n0, "decision_rows": rows,
                "planner_measure": agent.last.get("planner"), "planner_error": agent.last.get("planner_error")})
            print(f"[imp_eval] arm={arm} seed={seed} lifetime {i}: progress {lt['progress']:.3f} "
                  f"achieved {lt['achieved']} decisions {lt['decisions']} wall {lt['wall_seconds']:.0f} s "
                  f"planner {sum(plan_s[n0:]):.1f} s", flush=True)
        res["imagination_calibration"] = agent.planner.calibration.summary()
        res["horizon_controller"] = agent.planner.controller.state()
        res["selection_calibration"] = agent.meta.selection.summary()
    finally:
        res["wall_seconds"] = time.time() - t0
        res["gpu_peak_gb"] = (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else None
        try:
            agent.close()
        except Exception:  # noqa: BLE001
            pass
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w") as f:
                json.dump(res, f, default=str)
    return res


# ------------------------------------------------------------------------------------------------ summary
def _mean_ci(xs: Sequence[float], n_boot: int = 4000, seed: int = 0) -> Dict[str, Any]:
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs:
        return {"n": 0}
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return {"n": len(xs), "mean": m}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n_boot))
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return {"n": len(xs), "mean": m, "sd": sd, "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot) - 1]}


def _arm_stats(r: Dict[str, Any]) -> Dict[str, Any]:
    lts = r["lifetimes"]
    rows = [row for lt in lts for row in lt["decision_rows"]]
    tvs = [row["imagination"]["share_tv"] for row in rows if row.get("imagination")
           and row["imagination"].get("share_tv") is not None]
    flips = [row for row in rows if row.get("imagination") and row["imagination"]["argmax_with"] is not None
             and row["imagination"]["argmax_with"] != row["imagination"]["argmax_without"]]
    eff = [row["truth_chosen"] / row["truth_best"] for row in rows
           if row.get("truth_chosen") is not None and row.get("truth_best")]
    ops = {}
    for row in rows:
        ops[row["operation"]] = ops.get(row["operation"], 0) + 1
    return {"lifetimes": len(lts), "decisions": len(rows),
            "progress_mean": sum(lt["progress"] for lt in lts) / max(len(lts), 1),
            "achieved": sum(1 for lt in lts if lt["achieved"]),
            "expected_adopters": sum(lt.get("expected_adopters") or 0.0 for lt in lts),
            "oracle_expected_adopters": sum(lt.get("oracle_expected_adopters") or 0.0 for lt in lts),
            "truth_efficiency_mean": sum(eff) / len(eff) if eff else None,
            "imagination_tv_mean": sum(tvs) / len(tvs) if tvs else None,
            "imagination_tv_max": max(tvs) if tvs else None,
            "decisions_with_tv_over_0.1": sum(1 for t in tvs if t > 0.1),
            "argmax_changed_by_imagination": len(flips), "n_with_imagination": len(tvs),
            "operations": ops, "unlock": ops.get("lab.unlock", 0),
            "planner_seconds_per_decision": (sum(lt["planner_seconds"] for lt in lts)
                                             / max(sum(lt["planner_calls"] for lt in lts), 1)),
            "horizon_final": (r.get("horizon_controller") or {}).get("allowed"),
            "imagination_calibration": r.get("imagination_calibration"), "wall_seconds": r.get("wall_seconds")}


def summarize(paths: Sequence[str]) -> Dict[str, Any]:
    runs = [json.load(open(p)) for p in paths]
    by = {}
    for r in runs:
        by.setdefault((r["family"], r["real"], r["seed"]), {})[r["arm"]] = r
    pairs, per_seed = [], []
    for key, arms in sorted(by.items()):
        st = {a: _arm_stats(r) for a, r in arms.items()}
        row = {"family": key[0], "real": key[1], "seed": key[2], **{a: s for a, s in st.items()}}
        if "on" in st and "off" in st:
            n = min(st["on"]["lifetimes"], st["off"]["lifetimes"])
            on_l, off_l = arms["on"]["lifetimes"][:n], arms["off"]["lifetimes"][:n]
            diff_progress = [a["progress"] - b["progress"] for a, b in zip(on_l, off_l)]
            diff_adopters = [(a.get("expected_adopters") or 0.0) - (b.get("expected_adopters") or 0.0)
                             for a, b in zip(on_l, off_l)]
            first_div = None
            for li, (a, b) in enumerate(zip(on_l, off_l)):
                for di, (x, y) in enumerate(zip(a["decision_rows"], b["decision_rows"])):
                    if (x["operation"], x["arguments"]) != (y["operation"], y["arguments"]):
                        first_div = {"lifetime": li, "decision": di}
                        break
                if first_div:
                    break
            row["paired"] = {"lifetimes": n, "progress_on_minus_off": sum(diff_progress) / max(n, 1),
                             "expected_adopters_on_minus_off": sum(diff_adopters) / max(n, 1),
                             "first_divergence": first_div}
            pairs.append(row["paired"])
        per_seed.append(row)
    out = {"runs": len(runs), "per_seed": per_seed,
           "paired_over_seeds": {"progress_on_minus_off": _mean_ci([p["progress_on_minus_off"] for p in pairs]),
                                 "expected_adopters_on_minus_off":
                                     _mean_ci([p["expected_adopters_on_minus_off"] for p in pairs]),
                                 "seeds_diverged": sum(1 for p in pairs if p["first_divergence"] is not None),
                                 "n_pairs": len(pairs)},
           "on_arm": {"imagination_tv_mean": _mean_ci([r["on"]["imagination_tv_mean"] for r in per_seed
                                                       if "on" in r and r["on"]["imagination_tv_mean"] is not None]),
                      "argmax_changed": sum(r["on"]["argmax_changed_by_imagination"] for r in per_seed if "on" in r),
                      "decisions_with_imagination": sum(r["on"]["n_with_imagination"] for r in per_seed if "on" in r)},
           "off_arm_counterfactual": {
               "imagination_tv_mean": _mean_ci([r["off"]["imagination_tv_mean"] for r in per_seed
                                                if "off" in r and r["off"]["imagination_tv_mean"] is not None])}}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cam_live.predict.imp_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=("on", "off"), required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--family", default="T1")
    r.add_argument("--lifetimes", type=int, default=6)
    r.add_argument("--decisions", type=int, default=16)
    r.add_argument("--real", action="store_true")
    r.add_argument("--backbone", action="store_true", help="backbone-written candidates on (open-ended generation)")
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--world-base", type=int, default=None)
    r.add_argument("--budget-s", type=float, default=0.0)
    r.add_argument("--predict", default="{}", help="JSON of further predict-section overrides (both arms alike)")
    r.add_argument("--workdir", required=True)
    r.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("paths", nargs="+")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        kw = dict(family=a.family, lifetimes=a.lifetimes, decisions=a.decisions, real=a.real, backbone=a.backbone,
                  out=a.out, workdir=a.workdir, threads=a.threads, world_base=a.world_base,
                  extra_predict=json.loads(a.predict), budget_s=a.budget_s)
        if a.real:
            from ..nn.gpu import free_gpu_memory, gpu_lock
            with gpu_lock(timeout_s=3 * 3600.0):
                try:
                    run(a.arm, a.seed, **kw)
                finally:
                    free_gpu_memory()
        else:
            run(a.arm, a.seed, **kw)
        return 0
    res = summarize(a.paths)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res["paired_over_seeds"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
