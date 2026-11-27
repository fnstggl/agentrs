"""cam_live.eval.extra — the remaining prespecified experiments as commands (docs/S/PROTOCOLS.md §5, §8).

    python -m cam_live.eval.cli integrity --seeds 0-4            attack battery + trivial world, one process each
    python -m cam_live.eval.cli meta --plan main_cpu             16.5 checks on the saved checkpoints
    python -m cam_live.eval.cli scaling --scales 5m,20m,50m,150m --seeds 0-2 --device cuda
    python -m cam_live.eval.cli large --scale 1b                 instantiate 500m / 1b + one short lifetime (GPU lock)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List


def _seeds(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _evidence(name: str, obj: Dict[str, Any]) -> str:
    from .cli import _write
    return _write(name, obj)


# ------------------------------------------------------------------------------------------------ integrity
def _one_integrity(kind: str, seed: int, out: str) -> None:
    import torch
    torch.set_num_threads(3)
    from .experiments import EVAL_HOME
    from .integrity import attack_battery, trivial_world
    root = os.path.join(EVAL_HOME, "integrity", f"{kind}_s{seed}")
    r = attack_battery(seed, root) if kind == "attacks" else trivial_world(seed, root)
    with open(out, "w") as f:
        json.dump(r, f, default=str)


def cmd_integrity(a: argparse.Namespace) -> int:
    from .experiments import EVAL_HOME
    from .stats import mean_ci, wilson
    d = os.path.join(EVAL_HOME, "results", "integrity")
    os.makedirs(d, exist_ok=True)
    jobs = [(k, s, os.path.join(d, f"{k}_s{s}.json")) for k in ("attacks", "trivial") for s in _seeds(a.seeds)]

    def run(j: Any) -> int:
        k, s, out = j
        if os.path.exists(out):
            return 0
        p = subprocess.run([sys.executable, "-m", "cam_live.eval.extra", "--one-integrity", k, str(s), out],
                           capture_output=True, text=True, timeout=7200)
        if p.returncode != 0:
            print(p.stderr[-2000:], file=sys.stderr)
        return p.returncode
    with ThreadPoolExecutor(a.jobs) as ex:
        codes = list(ex.map(run, jobs))
    att = [json.load(open(o)) for k, s, o in jobs if k == "attacks" and os.path.exists(o)]
    triv = [json.load(open(o)) for k, s, o in jobs if k == "trivial" and os.path.exists(o)]
    names = sorted({n for r in att for n in r["detected"]})
    det = {n: {"detected": sum(1 for r in att if r["detected"].get(n) is True),
               "not_detected": sum(1 for r in att if r["detected"].get(n) is False),
               "not_applicable": sum(1 for r in att if r["detected"].get(n) is None)} for n in names}
    tick = [lt["tick"] for r in triv for lt in r["lifetimes"]]
    per_seed_tick = [sum(lt["tick"] for lt in r["lifetimes"]) / len(r["lifetimes"]) for r in triv if r["lifetimes"]]
    per_seed_last = [r["lifetimes"][-1]["tick"] for r in triv if r["lifetimes"]]
    per_seed_noise = [sum(lt["noise"] for lt in r["lifetimes"]) / len(r["lifetimes"]) for r in triv if r["lifetimes"]]
    rew_share = [r["internal_reward_by_operation"].get("lab.tick", {}).get("share_of_abs_reward") for r in triv]
    out = {"executed": True, "attack_battery": {"n_seeds": len(att), "by_attack": det, "runs": att},
           "trivial_world": {"n_seeds": len(triv), "tick_share_per_seed": mean_ci(per_seed_tick),
                             "tick_share_last_lifetime": mean_ci(per_seed_last),
                             "noise_share_per_seed": mean_ci(per_seed_noise),
                             "uniform_share": 1.0 / 16.0, "tick_share_of_abs_internal_reward": mean_ci(rew_share),
                             "verified_progress": mean_ci([sum(lt["progress"] for lt in r["lifetimes"]) /
                                                           len(r["lifetimes"]) for r in triv if r["lifetimes"]]),
                             "runs": triv},
           "exit_codes": codes}
    print(_evidence("integrity.json", out))
    return 0 if all(c == 0 for c in codes) else 1


# ------------------------------------------------------------------------------------------------ meta
def _one_meta(root: str, result: str, out: str) -> None:
    import torch
    torch.set_num_threads(3)
    from .meta_checks import run_checks
    with open(result) as f:
        r = json.load(f)
    res = run_checks(root, r)
    with open(out, "w") as f:
        json.dump(res, f, default=str)


def cmd_meta(a: argparse.Namespace) -> int:
    from .experiments import EVAL_HOME
    from .meta_checks import pathway_ablation
    from .stats import mean_ci, wilson
    res_dir = os.path.join(EVAL_HOME, "results", a.plan)
    ck = os.path.join(EVAL_HOME, "ckpts", a.plan)
    d = os.path.join(EVAL_HOME, "results", f"meta_{a.plan}")
    os.makedirs(d, exist_ok=True)
    jobs = []
    for cond in ("E", "E_mix0"):
        for s in _seeds(a.seeds):
            name = f"{cond}_F1_s{s}"
            if os.path.exists(os.path.join(res_dir, name + ".json")) and os.path.exists(os.path.join(ck, name)):
                jobs.append((os.path.join(ck, name), os.path.join(res_dir, name + ".json"),
                             os.path.join(d, name + ".json")))

    def run(j: Any) -> int:
        root, result, out = j
        if os.path.exists(out):
            return 0
        p = subprocess.run([sys.executable, "-m", "cam_live.eval.extra", "--one-meta", root, result, out],
                           capture_output=True, text=True, timeout=7200)
        if p.returncode != 0:
            print(p.stderr[-2000:], file=sys.stderr)
        return p.returncode
    with ThreadPoolExecutor(a.jobs) as ex:
        codes = list(ex.map(run, jobs))
    checks = [json.load(open(o)) for _, _, o in jobs if os.path.exists(o)]
    abl = {}
    for cond in ("D", "E_unverified", "E", "E_mix0"):
        rows = []
        for s in _seeds(a.seeds):
            root = os.path.join(ck, f"{cond}_F1_s{s}")
            if os.path.exists(os.path.join(root, "export_post")):
                rows.append(pathway_ablation(root))
        if rows:
            groups = sorted({g for r in rows for g in r})
            abl[cond] = {g: {"changed": sum(1 for r in rows if r.get(g)), "n": len(rows)} for g in groups}

    def agg(cond: str) -> Dict[str, Any]:
        cs = [c for c in checks if c.get("condition") == cond and "skipped" not in c]
        if not cs:
            return {"n": 0}
        f = lambda path: [_get(c, path) for c in cs]  # noqa: E731
        return {
            "n_runs": len(cs),
            "1_delta_theta_psi_post_vs_pre_live_mix": mean_ci(f("1_reward_params_affect_inner_update.live_mix."
                                                               "delta_theta_psi_post_vs_psi_pre")),
            "1_delta_theta_psi_post_vs_pre_mix0": mean_ci(f("1_reward_params_affect_inner_update.mix0_differentiated."
                                                          "delta_theta_psi_post_vs_psi_pre")),
            "1_noise_floor_mix0": mean_ci(f("1_reward_params_affect_inner_update.mix0_differentiated."
                                            "noise_floor_same_psi")),
            "1_runs_with_effect_above_floor_mix0": sum(
                1 for c in cs if _get(c, "1_reward_params_affect_inner_update.mix0_differentiated."
                                         "delta_theta_psi_post_vs_psi_pre") >
                10 * max(1e-12, _get(c, "1_reward_params_affect_inner_update.mix0_differentiated."
                                        "noise_floor_same_psi"))),
            "1_runs_with_effect_above_floor_live_mix": sum(
                1 for c in cs if _get(c, "1_reward_params_affect_inner_update.live_mix."
                                         "delta_theta_psi_post_vs_psi_pre") >
                10 * max(1e-12, _get(c, "1_reward_params_affect_inner_update.live_mix.noise_floor_same_psi"))),
            "2_policy_tv_heldout": mean_ci(f("2_inner_update_changes_policy.tv_mean")),
            "3_4_applied_meta_updates_per_run": mean_ci(f("3_4_meta_updates.applied")),
            "3_4_applied_nonzero_changed_version_per_run": mean_ci(f("3_4_meta_updates.applied_nonzero_changed_version")),
            "3_applied_with_fewer_independent_outcomes_total": sum(
                _get(c, "3_4_meta_updates.applied_with_fewer_independent_outcomes") or 0 for c in cs),
            "4_reward_group_changed_pre_to_post": sum(1 for c in cs if c["group_changed_pre_to_post"].get("reward")),
            "6_spearman_learned_reward_pre_vs_post": mean_ci(f("6_beyond_magnitude.spearman_learned_reward_pre_vs_post")),
            "6_delta_theta_output_scaled_x3": mean_ci(f("6_beyond_magnitude.delta_theta_output_scaled_x3_vs_post")),
            "6_delta_theta_psi_pre_vs_post": mean_ci(f("6_beyond_magnitude.delta_theta_psi_pre_vs_post")),
            "live_mix": mean_ci(f("live_mix"))}
    out = {"executed": True, "plan": a.plan, "checks": {c: agg(c) for c in ("E", "E_mix0")},
           "pathway_ablation_groups_changed": abl, "runs": checks, "exit_codes": codes}
    print(_evidence(f"meta_{a.plan}.json", out))
    return 0 if all(c == 0 for c in codes) else 1


def _get(d: Any, path: str) -> Any:
    for k in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


# ------------------------------------------------------------------------------------------------ scaling
def cmd_scaling(a: argparse.Namespace) -> int:
    from .experiments import EVAL_HOME
    d = os.path.join(EVAL_HOME, "results", "scaling")
    os.makedirs(d, exist_ok=True)
    jobs = []
    for sc in a.scales.split(","):
        for s in _seeds(a.seeds):
            out = os.path.join(d, f"E_{sc}_s{s}.json")
            if os.path.exists(out):
                continue
            jobs.append([sys.executable, "-m", "cam_live.eval.cli", "run", "--condition", "E", "--family", "F1",
                         "--seed", str(s), "--lifetimes", str(a.lifetimes), "--decisions", "16", "--scale", sc,
                         "--device", a.device, "--threads", str(a.threads), "--out", out])

    def run(cmd: List[str]) -> int:
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=4 * 3600)
        print(f"{' '.join(cmd[-12:-8])} exit {p.returncode} in {time.time() - t0:.0f}s", flush=True)
        if p.returncode != 0:
            print(p.stderr[-1500:], file=sys.stderr)
        return p.returncode
    with ThreadPoolExecutor(a.jobs) as ex:
        codes = list(ex.map(run, jobs))
    return 0 if all(c == 0 for c in codes) else 1


def cmd_large(a: argparse.Namespace) -> int:
    """Instantiate the 500m / 1b experiential core with the tiny backbone on the GPU (under the lock) and run one
    short lifetime of E: parameters, memory, seconds per decision. Not a learning result."""
    import torch
    from ..nn.gpu import gpu_lock
    from .experiments import EVAL_HOME, run_condition
    outs = {}
    for sc in a.scales.split(","):
        with gpu_lock(min_free_gb=a.min_free_gb, timeout_s=3600.0):
            t0 = time.time()
            try:
                r = run_condition("E", "F1", 0, n_lifetimes=1, max_decisions=a.decisions, real=False, scale=sc,
                                  device="cuda", save_dir="", out="", budget_s=600.0,
                                  world_seed_base=10_000)
                lt = r["lifetimes"][0] if r["lifetimes"] else {}
                outs[sc] = {"executed": True, "parameters": r["parameters"], "gpu_peak_gb": r.get("gpu_peak_gb"),
                            "decisions": lt.get("decisions"), "wall_s": r["wall_seconds"],
                            "training_s_per_decision": (lt.get("training_s", 0.0) / max(1, lt.get("decisions", 1))),
                            "inference_s_per_decision": (lt.get("inference_s", 0.0) / max(1, lt.get("decisions", 1))),
                            "weight_delta_norm_by_group": r.get("weight_delta_norm_by_group"),
                            "progress": lt.get("progress"), "lock_seconds": time.time() - t0}
            except Exception as e:  # noqa: BLE001 — out of memory etc. is a result too
                outs[sc] = {"executed": False, "error": f"{type(e).__name__}: {str(e)[:400]}",
                            "lock_seconds": time.time() - t0}
            finally:
                torch.cuda.empty_cache()
    print(_evidence("scaling_large.json", {"executed": True, "backbone": "tiny (random)", "runs": outs}))
    return 0


def add_commands(sub: Any) -> None:
    i = sub.add_parser("integrity", help="reward-integrity attack battery and trivial-achievement world")
    i.add_argument("--seeds", default="0-4")
    i.add_argument("--jobs", type=int, default=10)
    i.set_defaults(fn=cmd_integrity)
    m = sub.add_parser("meta", help="16.5 meta-learning checks on saved checkpoints")
    m.add_argument("--plan", default="main_cpu")
    m.add_argument("--seeds", default="0-9")
    m.add_argument("--jobs", type=int, default=10)
    m.set_defaults(fn=cmd_meta)
    s = sub.add_parser("scaling", help="E at several experiential scales, same protocol")
    s.add_argument("--scales", default="5m,20m,50m,150m")
    s.add_argument("--seeds", default="0-2")
    s.add_argument("--lifetimes", type=int, default=6)
    s.add_argument("--device", default="cuda")
    s.add_argument("--jobs", type=int, default=4)
    s.add_argument("--threads", type=int, default=3)
    s.set_defaults(fn=cmd_scaling)
    lg = sub.add_parser("large", help="instantiate 500m / 1b and run one short lifetime (GPU lock)")
    lg.add_argument("--scales", default="500m,1b")
    lg.add_argument("--decisions", type=int, default=8)
    lg.add_argument("--min-free-gb", dest="min_free_gb", type=float, default=20.0)
    lg.set_defaults(fn=cmd_large)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--one-integrity":
        _one_integrity(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--one-meta":
        _one_meta(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("internal entry point; use python -m cam_live.eval.cli", file=sys.stderr)
        sys.exit(2)
