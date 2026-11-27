"""cam_live.eval.transfer — fresh-process, weights-only parametric transfer evaluation (spec 16.8, CL-07, SV-10).

    cam-live evaluate-transfer --checkpoint <experienced> [--baseline <pre-experience>] [--domains T1,T2]
                               [--seeds 0-4] [--out result.json] [--real]
    python -m cam_live.eval.transfer ...            (the same)

The ten steps of 16.8, literally:

1-2. the original pre-experience checkpoint (``--baseline``) and the experienced checkpoint (``--checkpoint``):
     full agent checkpoints (``CamLiveAgent.save``) or weights-only exports; when ``--baseline`` is omitted the
     sibling ``ckpt_pre`` / ``export_pre`` of the experienced checkpoint's run directory is used.
3.   every evaluation (version x domain x seed) runs in its own new Python process (``--child``), started with an
     empty working directory, an empty ``CAM_LIVE_HOME`` and no path to the writer's run directory.
4-7. the child receives only a weights-only export (``credit.checkpoint.export_weights``: ``weights.safetensors`` +
     ``config.json``) copied into a fresh directory: no written lessons, no task memories, no experience store,
     no prediction / experiment registries, no trajectories, no cached solutions, no optimizer state, no fast
     beliefs. The child builds a new agent from the configuration, loads the tensors strictly, and resets the fast
     beliefs to the (loaded) learned prior; it checks and records that its store, registries and fast state are
     empty before the first decision.
8.   retained: the trained neural parameters (experiential core, adapters, heads, intrinsic reward) and the fixed
     configuration only. The frozen pretrained backbone is the same file for both versions.
9.   the objective is previously unfamiliar: ``T1`` (LabWorld with a different mechanism class) and ``T2`` (a
     software-utility tuning task on real compressors), never part of the experience (``F1``).
10.  both versions learn the new objective over the same sequence of new worlds with the same seeds; competence per
     lifetime is measured by the independent evaluator; N_q, S_q, AULC, compute and elapsed time are reported.

``--frozen`` evaluates the versions without any learning (continual-learning before/after test, spec 16.7).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

#: prespecified competence definitions (docs/S/PROTOCOLS.md §T): per-lifetime competence, rolling window, q
COMPETENCE = {"F1": {"metric": "progress", "window": 2, "q": 0.75},
              "T1": {"metric": "progress", "window": 2, "q": 0.75},
              "T2": {"metric": "first_attempt", "window": 3, "q": 2 / 3}}
DEFAULT_PLAN = {"T1": {"lifetimes": 5, "decisions": 16}, "T2": {"lifetimes": 6, "decisions": 12},
                "F1": {"lifetimes": 3, "decisions": 16}}
EVAL_WORLD_BASE = {"F1": 50_000, "T1": 60_000, "T2": 70_000}


def competence_of(domain: str, lt: Dict[str, Any]) -> float:
    m = COMPETENCE[domain]["metric"]
    if m == "progress":
        return float(lt["progress"])
    if m == "quality":
        return float(lt["quality"])
    if m == "first_attempt":
        return 1.0 if lt.get("first_achieving_decision") == 1 else 0.0
    raise KeyError(m)


# ------------------------------------------------------------------------------------------------ exports
def _is_export(path: str) -> bool:
    return os.path.exists(os.path.join(path, "weights.safetensors")) and os.path.exists(os.path.join(path,
                                                                                                    "config.json"))


def make_export(src: str, dest: str) -> Dict[str, Any]:
    """A weights-only export of ``src`` in ``dest`` (nothing else is copied)."""
    from ..credit.checkpoint import export_weights
    if os.path.exists(dest):
        shutil.rmtree(dest)
    if _is_export(src):
        os.makedirs(dest)
        for f in ("weights.safetensors", "config.json"):
            shutil.copy2(os.path.join(src, f), os.path.join(dest, f))
        with open(os.path.join(dest, "config.json")) as fh:
            return json.load(fh)
    cfg = None
    with contextlib.suppress(Exception):
        from ..credit.checkpoint import load_checkpoint
        st = load_checkpoint(src)
        cfg = (st.extra or {}).get("config") if isinstance((st.extra or {}).get("config"), dict) else None
    meta = export_weights(src, dest, config=cfg)
    if not meta.get("config"):
        raise RuntimeError(f"{src}: the checkpoint carries no configuration; export it with its configuration")
    return meta


def _baseline_of(ckpt: str) -> Optional[str]:
    run = os.path.dirname(os.path.abspath(ckpt.rstrip("/")))
    for name in ("export_pre", "ckpt_pre"):
        p = os.path.join(run, name)
        if os.path.exists(p):
            return p
    return None


# ------------------------------------------------------------------------------------------------ child
def child(a: argparse.Namespace) -> int:
    """One fresh evaluation process: weights-only export -> new agent -> unfamiliar objective."""
    import torch
    t0 = time.time()
    if a.threads:
        torch.set_num_threads(int(a.threads))
    from ..agent import CamLiveAgent
    from ..config import CamLiveConfig
    from ..credit.checkpoint import load_weights, load_weights_only
    from ..validation import OffsetClock
    from .conditions import NAMES, Timed, eval_config
    from .runner import run_lifetime
    cwd_files = sorted(os.listdir(os.getcwd()))
    weights, meta = load_weights_only(a.export, map_location="cpu")
    cfg = CamLiveConfig.from_dict(meta["config"])
    cfg = eval_config(cfg, "E", seed=int(a.agent_seed))            # same seed for both versions: only weights differ
    if a.device:
        cfg.device = a.device
    run_dir = os.path.join(os.getcwd(), "run")
    clock = OffsetClock()
    agent = CamLiveAgent(cfg, os.path.join(run_dir, "agent"), clock=clock)
    rep = load_weights(agent.model, weights, strict=True)
    agent.model.mark_adapters_updated()
    agent.learner.reset_fast()
    agent.predictor.set_fast_state(agent.learner.fast_state)
    from ..credit.checkpoint import collect_weights, tensor_hash
    hashes = {g: tensor_hash(d) for g, d in collect_weights(agent.model).items() if d}
    fresh = {"cwd_files_at_start": cwd_files, "store_records": agent.store.count(),
             "predictions_registered": len(agent.predictions), "experiments": len(agent.experiments.records()),
             "fast_state_updates": {k: agent.learner.fast_state.n_updates(k) for k in ("count", "real", "binary")},
             "load_clean": rep.clean, "tensors_loaded": len(rep.loaded),
             "group_hashes": hashes, "export_group_hashes": meta.get("group_hashes", {})}
    if a.frozen:
        from .conditions import FrozenEvalAgent
        agent.__class__ = FrozenEvalAgent
    timed = Timed(agent)
    lts = []
    for i in range(int(a.lifetimes)):
        lt = run_lifetime(timed, a.domain, int(a.world_base) + i, os.path.join(run_dir, f"lt{i:02d}"), clock,
                          max_decisions=int(a.decisions), k=int(a.k))
        lt.pop("truth_rows", None)
        rows = lt.pop("decision_rows", [])
        lt["n_decision_rows"] = len(rows)
        lt["index"] = i
        lt["competence"] = competence_of(a.domain, lt)
        lts.append(lt)
    preds = [{"kind": p.query.kind, "nll": p.nll, "goal_relevant": bool(p.query.goal_relevant)}
             for p in agent.predictions.records() if p.nll is not None]
    from .experiments import code_identity
    out = {"executed": True, "version": a.version, "domain": a.domain, "agent_seed": int(a.agent_seed),
           "code": code_identity(),
           "world_base": int(a.world_base), "frozen": bool(a.frozen), "fresh_process": fresh,
           "pid": os.getpid(), "lifetimes": lts, "prediction_nll": preds,
           "compute": {"seconds": dict(timed.seconds), "cpu_seconds": dict(timed.cpu_seconds)},
           "wall_seconds": time.time() - t0, "condition": NAMES["E"]}
    with open(a.result, "w") as f:
        json.dump(out, f, default=str)
    agent.close()
    return 0


def spawn(export: str, version: str, domain: str, seed: int, *, lifetimes: int, decisions: int, k: int,
          frozen: bool, device: str, threads: int, timeout_s: float = 7200.0) -> Dict[str, Any]:
    work = tempfile.mkdtemp(prefix=f"camlive_transfer_{version}_{domain}_{seed}_")
    own = os.path.join(work, "export")
    shutil.copytree(export, own)
    cwd = os.path.join(work, "cwd")
    home = os.path.join(work, "home")
    os.makedirs(cwd)
    os.makedirs(home)
    res = os.path.join(work, "result.json")
    cmd = [sys.executable, "-m", "cam_live.eval.transfer", "--child", "--export", own, "--version", version,
           "--domain", domain, "--agent-seed", str(seed), "--world-base", str(EVAL_WORLD_BASE[domain] + 100 * seed),
           "--lifetimes", str(lifetimes), "--decisions", str(decisions), "--k", str(k), "--result", res,
           "--threads", str(threads)] + (["--frozen"] if frozen else []) + (["--device", device] if device else [])
    env = {k2: v for k2, v in os.environ.items() if not k2.startswith("CAM_LIVE_") or k2.startswith("CAM_LIVE_EVAL_")}
    env["CAM_LIVE_HOME"] = home
    env["OMP_NUM_THREADS"] = str(threads)
    if "CAM_LIVE_GPU_LOCK" in os.environ:
        env["CAM_LIVE_GPU_LOCK"] = os.environ["CAM_LIVE_GPU_LOCK"]
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout_s)
    if p.returncode != 0 or not os.path.exists(res):
        return {"executed": False, "version": version, "domain": domain, "agent_seed": seed,
                "returncode": p.returncode, "stderr": p.stderr[-4000:]}
    with open(res) as f:
        out = json.load(f)
    out["work_dir"] = work
    shutil.rmtree(os.path.join(work, "cwd", "run"), ignore_errors=True)
    return out


# ------------------------------------------------------------------------------------------------ summary
def summarise(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from .stats import aulc, diff_ci, mean_ci, n_to_threshold, speedup, summarise_nq
    out: Dict[str, Any] = {}
    for domain in sorted({r["domain"] for r in runs if r.get("executed")}):
        comp = COMPETENCE[domain]
        per: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for r in runs:
            if not r.get("executed") or r["domain"] != domain:
                continue
            curve = [lt["competence"] for lt in r["lifetimes"]]
            exp_ = []
            n = 0
            for lt in r["lifetimes"]:
                n += lt["decisions"]
                exp_.append(float(n))
            per.setdefault(r["version"], {})[r["agent_seed"]] = {
                "nq": n_to_threshold(curve, comp["q"], experience=exp_, window=comp["window"]),
                "aulc": aulc(curve), "curve": curve,
                "quality": [lt["quality"] for lt in r["lifetimes"]],
                "progress": [lt["progress"] for lt in r["lifetimes"]],
                "training_s": sum(lt["training_s"] for lt in r["lifetimes"]),
                "inference_s": sum(lt["inference_s"] for lt in r["lifetimes"]),
                "wall_s": r["wall_seconds"], "decisions": n}
        seeds = sorted(set(per.get("initial", {})) & set(per.get("experienced", {})))
        d: Dict[str, Any] = {"competence": comp, "paired_seeds": seeds}
        for v in ("initial", "experienced"):
            pv = per.get(v, {})
            d[v] = {"nq": summarise_nq([pv[s]["nq"] for s in seeds]),
                    "aulc": mean_ci([pv[s]["aulc"] for s in seeds]),
                    "curves": {s: pv[s]["curve"] for s in seeds},
                    "training_s": mean_ci([pv[s]["training_s"] for s in seeds]),
                    "inference_s": mean_ci([pv[s]["inference_s"] for s in seeds]),
                    "wall_s": mean_ci([pv[s]["wall_s"] for s in seeds])}
        if seeds:
            d["S_q"] = speedup([per["initial"][s]["nq"] for s in seeds], [per["experienced"][s]["nq"] for s in seeds])
            d["aulc_difference_experienced_minus_initial"] = diff_ci(
                [per["experienced"][s]["aulc"] for s in seeds], [per["initial"][s]["aulc"] for s in seeds],
                paired=True)
            d["first_lifetime_competence_difference"] = diff_ci(
                [per["experienced"][s]["curve"][0] for s in seeds], [per["initial"][s]["curve"][0] for s in seeds],
                paired=True)
        out[domain] = d
    return out


def _seeds(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        if "-" in part:
            x, y = part.split("-")
            out += list(range(int(x), int(y) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cam-live evaluate-transfer", description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--config", default="", help="accepted for the runtime CLI; the export's configuration is used")
    ap.add_argument("--domains", default="T1,T2")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--lifetimes", type=int, default=0, help="0: the prespecified number per domain")
    ap.add_argument("--decisions", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--frozen", action="store_true", help="no learning during the evaluation (spec 16.7)")
    ap.add_argument("--real", action="store_true", help="GPU child processes (real backbone) under the GPU lock")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=1, help="parallel child processes (CPU only)")
    ap.add_argument("--out", default="")
    # child
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--export", default="")
    ap.add_argument("--version", default="")
    ap.add_argument("--domain", default="")
    ap.add_argument("--agent-seed", dest="agent_seed", type=int, default=0)
    ap.add_argument("--world-base", dest="world_base", type=int, default=0)
    ap.add_argument("--result", default="")
    ap.add_argument("--device", default="")
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    if a.child:
        return child(a)
    if not a.checkpoint or not os.path.exists(a.checkpoint):
        print(f"evaluate-transfer: checkpoint {a.checkpoint!r} does not exist", file=sys.stderr)
        return 2
    base = a.baseline or _baseline_of(a.checkpoint)
    if not base or not os.path.exists(base):
        print("evaluate-transfer: no pre-experience checkpoint (--baseline) found", file=sys.stderr)
        return 2
    t0 = time.time()
    stage = tempfile.mkdtemp(prefix="camlive_transfer_exports_")
    versions = {"initial": make_export(base, os.path.join(stage, "initial")),
                "experienced": make_export(a.checkpoint, os.path.join(stage, "experienced"))}
    domains = [d for d in a.domains.split(",") if d]
    jobs = []
    for d in domains:
        plan = DEFAULT_PLAN[d]
        for s in _seeds(a.seeds):
            for v in ("initial", "experienced"):
                jobs.append((v, d, s, a.lifetimes or plan["lifetimes"], a.decisions or plan["decisions"]))
    runs: List[Dict[str, Any]] = []
    dev = "cuda" if a.real else "cpu"

    def one(job: Any) -> Dict[str, Any]:
        v, d, s, L, D = job
        if a.real:
            from ..nn.gpu import gpu_lock
            with gpu_lock(min_free_gb=12.0):
                return spawn(os.path.join(stage, v), v, d, s, lifetimes=L, decisions=D, k=a.k, frozen=a.frozen,
                             device=dev, threads=a.threads)
        return spawn(os.path.join(stage, v), v, d, s, lifetimes=L, decisions=D, k=a.k, frozen=a.frozen, device=dev,
                     threads=a.threads)
    if a.jobs > 1 and not a.real:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(a.jobs) as ex:
            runs = list(ex.map(one, jobs))
    else:
        runs = [one(j) for j in jobs]
    diff_hash = {g: versions["initial"].get("group_hashes", {}).get(g) != h
                 for g, h in versions["experienced"].get("group_hashes", {}).items()}
    report = {"executed": True, "kind": "parametric_transfer" if not a.frozen else "frozen_before_after",
              "checkpoint": os.path.abspath(a.checkpoint), "baseline": os.path.abspath(base),
              "groups_differing_between_versions": diff_hash, "domains": domains, "seeds": _seeds(a.seeds),
              "n_runs": len(runs), "n_failed": sum(1 for r in runs if not r.get("executed")),
              "failures": [r for r in runs if not r.get("executed")][:5],
              "fresh_process_checks": [{"version": r["version"], "domain": r["domain"], "seed": r["agent_seed"],
                                        "pid": r.get("pid"), **{k: r["fresh_process"][k] for k in
                                                                ("cwd_files_at_start", "store_records",
                                                                 "predictions_registered", "experiments",
                                                                 "fast_state_updates", "load_clean",
                                                                 "tensors_loaded")},
                                        "hashes_match_export": all(r["fresh_process"]["group_hashes"].get(g) == h
                                                                   for g, h in r["fresh_process"]
                                                                   ["export_group_hashes"].items())}
                                       for r in runs if r.get("executed")],
              "summary": summarise(runs), "runs": runs, "wall_seconds": time.time() - t0}
    shutil.rmtree(stage, ignore_errors=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1, default=str)
    s = report["summary"]
    print(json.dumps({d: {"S_q": s[d].get("S_q", {}).get("S_q"), "reached_initial": s[d].get("S_q", {}).get(
        "reached_initial"), "reached_experienced": s[d].get("S_q", {}).get("reached_experienced")}
        for d in s}, default=str))
    return 0 if report["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
