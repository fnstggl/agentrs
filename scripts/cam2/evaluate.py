"""Evaluate CAM checkpoints (or the LLM-only baseline B1) without parameter updates.

  # B4 / B5: a trained checkpoint, without and with model-based planning at decision time
  PYTHONPATH=src python scripts/cam2/evaluate.py --ckpt runs/cam2/main_s0/ckpt/final.pt --out runs/cam2/eval/b4.json
  PYTHONPATH=src python scripts/cam2/evaluate.py --ckpt ... --plan '{"enabled": true, "min_reliability": 0.0}' --out b5.json
  # B1: the frozen LLM alone with the same tools / observations / validation
  PYTHONPATH=src python scripts/cam2/evaluate.py --llm-only --out runs/cam2/eval/b1.json
  # LOCKED TEST (final, infrequent; every access is logged; never used for selection)
  PYTHONPATH=src python scripts/cam2/evaluate.py --ckpt ... --locked-test --ack "<tasks.ACK>" --out ...

Several checkpoints can be evaluated in one process (one LLM load, one worker pool): --ckpt a.pt b.pt ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentrs.cam2.backbone import load_llm  # noqa: E402
from agentrs.cam2.envs.pool import EnvPool  # noqa: E402
from agentrs.cam2.tasks import locked_test_tasks, train_eval_tasks, val_tasks  # noqa: E402
from agentrs.cam2.trainer import RunConfig, Trainer  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--llm-only", action="store_true")
    ap.add_argument("--env", default="miniwob")
    ap.add_argument("--llm", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--per-family", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="val", choices=["val", "train_eval"])
    ap.add_argument("--plan", nargs="*", default=None,
                    help="JSON PlanConfig overrides applied at evaluation: one for all checkpoints or one per checkpoint")
    ap.add_argument("--sample", action="store_true", help="sample actions instead of greedy")
    ap.add_argument("--locked-test", action="store_true")
    ap.add_argument("--ack", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.locked_test:
        tasks = locked_test_tasks(a.env, a.ack, per_family=a.per_family, who=" ".join(sys.argv))
    elif a.split == "train_eval":
        tasks = train_eval_tasks(a.env, a.per_family)
    else:
        tasks = val_tasks(a.env, a.per_family, limit=a.limit)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    llm = load_llm(a.llm, device=dev)
    pool = EnvPool(a.env, a.workers, {"max_steps": a.max_steps})
    out = {"split": "LOCKED_TEST" if a.locked_test else a.split, "n_tasks": len(tasks), "results": {}}
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    try:
        if a.llm_only:
            from agentrs.cam2.baselines import run_llm_agent

            log: list = []
            big = a.env != "miniwob"
            r = run_llm_agent(llm, pool, tasks, max_new=256 if big else 96, log=log, max_ctx=9000 if big else 3500,
                              max_tools_chars=24000 if big else 6000)
            r["samples"] = log[:40]
            out["results"]["B1_llm_only"] = r
            print(f"[eval] B1 llm-only success={r['success']:.3f} ci={r['ci95']} invalid={r['invalid_rate']:.3f}", flush=True)
        for i, ck in enumerate(a.ckpt):
            label = a.labels[i] if a.labels else ck
            c = torch.load(ck, map_location="cpu", weights_only=False)
            plan = dict(c.get("plan") or {})
            pov = None
            if a.plan:
                pov = json.loads(a.plan[i] if len(a.plan) == len(a.ckpt) else a.plan[0])
                plan.update(pov)
            rc = RunConfig(run_dir=str(outp.parent / "_eval_tmp" / f"{i}"), env=a.env, llm=llm.name, workers=a.workers,
                           max_steps=a.max_steps, cam={k: v for k, v in c["cam_config"].items()
                                                      if k not in ("llm_pooled_dim", "llm_hidden")},
                           learn=c["learn_config"], plan=plan)
            tr = Trainer(rc, llm=llm, pool=pool, device=dev)
            tr.load(ck)
            if pov:   # the checkpoint's measured reliability / u_max stay unless overridden explicitly
                for k, v in pov.items():
                    setattr(tr.pcfg, k, v)
                tr.agent.plan = tr.pcfg if tr.pcfg.enabled else None
            t0, g0, e0 = time.time(), llm.stats["gen_tokens"], llm.stats["encode_tokens"]
            r = tr.evaluate(tasks, greedy=not a.sample, label=label)
            r["plan"] = {k: getattr(tr.pcfg, k) for k in ("enabled", "horizon", "top_k", "kappa", "beta",
                                                            "min_reliability", "reliability", "u_max")}
            r["ckpt"] = ck
            r["updates"] = tr.counters["updates"]
            r["env_steps_trained"] = tr.counters["env_steps"]
            r["llm_gen_tokens"] = llm.stats["gen_tokens"] - g0
            r["llm_encode_tokens"] = llm.stats["encode_tokens"] - e0
            out["results"][label] = r
            print(f"[eval] {label}: success={r['success']:.3f} ci={r['ci95']} invalid={r['invalid_rate']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            tr.close()
            del tr
            torch.cuda.empty_cache()
            outp.write_text(json.dumps(out, indent=1, default=float))
        out["llm"] = llm.describe()
        outp.write_text(json.dumps(out, indent=1, default=float))
    finally:
        pool.close()


if __name__ == "__main__":
    main()
