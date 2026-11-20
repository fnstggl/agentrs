"""Scratch reproduction of searched candidates with independent seeds (promotion check, task statement §11-12).

Re-trains the given candidates' specs FROM SCRATCH (no inherited weights, fresh optimiser) with seeds that were
never used in the search, at the same budget, and evaluates every model on the same outer-validation tasks
(greedy, no updates). Reports per-seed results and the paired per-task difference of each candidate against the
reference (the search's seed candidate) pooled over seeds.

  PYTHONPATH=src python scripts/cam2/reproduce.py --search runs/cam2/search/adaptive --cids c000-... c005-... \
      --seeds 101 102 --steps 3600 --out runs/cam2/reproduce
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentrs.cam2.backbone import load_llm  # noqa: E402
from agentrs.cam2.envs.pool import EnvPool  # noqa: E402
from agentrs.cam2.search.population import paired_delta  # noqa: E402
from agentrs.cam2.tasks import val_tasks  # noqa: E402
from agentrs.cam2.trainer import RunConfig, Trainer  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", nargs="+", required=True, help="search roots (population.json); cids are looked up in all")
    ap.add_argument("--cids", nargs="+", required=True, help="first cid = reference (the hand-designed seed)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 102])
    ap.add_argument("--steps", type=int, default=3600)
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="runs/cam2/reproduce")
    a = ap.parse_args(argv)
    cands = {}
    cfg = None
    for root in a.search:
        pop = json.loads((Path(root) / "population.json").read_text())
        cfg = cfg or pop["meta"]["config"]
        for c in pop["candidates"]:
            cands[c["cid"]] = c
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_p = out / "results.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {"cids": a.cids, "seeds": a.seeds, "steps": a.steps, "runs": {}}
    llm = load_llm(cfg["llm"], device="cuda")
    pool = EnvPool("miniwob", a.workers, {"max_steps": cfg["max_steps"]})
    tasks = val_tasks("miniwob", a.per_family)
    try:
        for seed in a.seeds:
            for cid in a.cids:
                key = f"{cid}/s{seed}"
                if key in res["runs"]:
                    continue
                s = cands[cid]["spec"]
                rc = RunConfig(run_dir=str(out / cid / f"s{seed}"), env="miniwob", llm=llm.name, workers=a.workers,
                               seed=seed, env_steps=a.steps, max_minutes=40, max_steps=cfg["max_steps"], eval_every=0,
                               ckpt_every=5, cam=dict(s["arch"]), learn={**cfg["fixed_learn"], **s["learn"]},
                               plan=dict(s["plan"]), phase=f"reproduce:{cid}")
                tr = Trainer(rc, llm=llm, pool=pool, device="cuda")
                summ = tr.train()
                ev = tr.evaluate(tasks, label=key)
                res["runs"][key] = {"cid": cid, "seed": seed, "val": {k: ev[k] for k in ("success", "ci95", "per_family", "n",
                                                                                         "invalid_rate", "dropped")},
                                    "episodes": ev["episodes"], "train_env_steps": tr.counters["env_steps"],
                                    "wall_s": tr.counters["wall_s"], "llm_unchanged": summ["llm_fingerprint_unchanged"],
                                    "params": tr.cam.n_params()["total"]}
                print(f"[reproduce] {key}: val={ev['success']:.3f} ci={ev['ci95']}", flush=True)
                tr.close()
                del tr
                torch.cuda.empty_cache()
                res_p.write_text(json.dumps(res, indent=1, default=float))
        ref = a.cids[0]
        res["comparison"] = {}
        for cid in a.cids[1:]:
            A, B = [], []
            for seed in a.seeds:
                A += [dict(e, task=f"s{seed}:{e['task']}") for e in res["runs"][f"{cid}/s{seed}"]["episodes"]]
                B += [dict(e, task=f"s{seed}:{e['task']}") for e in res["runs"][f"{ref}/s{seed}"]["episodes"]]
            res["comparison"][cid] = paired_delta(A, B)
            print(f"[reproduce] {cid} vs {ref}: {json.dumps(res['comparison'][cid])}", flush=True)
        res_p.write_text(json.dumps(res, indent=1, default=float))
    finally:
        pool.close()


if __name__ == "__main__":
    main()
