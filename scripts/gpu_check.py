#!/usr/bin/env python
"""GPU readiness check for a model profile: inference + a short real training run.

Loads the policy (and optionally the world model) from a model profile, runs inference (free-text
generation and candidate-action scoring on MiniWeb tasks), trains for a few behaviour-cloning /
world-model steps on teacher demonstrations, and verifies that trainable weights changed, that
losses are finite, and that a checkpoint save/reload reproduces predictions. Reports step time and
peak GPU memory. Not an experiment: the numbers are a smoke signal only.

    python scripts/gpu_check.py --model research --steps 20
    python scripts/gpu_check.py --model research --components policy,world_model --out runs/gpu_check
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from agentrs.cli import _model_cfg  # noqa: E402


def demos(fams, n_train, n_val, seed=0):
    from agentrs.agent.runtime import TeacherActor, run_episode
    from agentrs.data.schemas import Provenance, Source
    from agentrs.demonstrations.teacher import ScriptedTeacher
    from agentrs.environments.miniweb.tasks import task_set
    from agentrs.evaluation.benchmark import make_env

    env = make_env("miniweb")
    prov = Provenance(source=Source.TEACHER, source_id="teacher:gpu_check")
    train, val = [], []
    for i, t in enumerate(task_set("train", fams, n_per_family=n_train)):
        train += run_episode(env, t, TeacherActor(ScriptedTeacher(0.15, seed + i)), prov).transitions
    for i, t in enumerate(task_set("val", fams, n_per_family=n_val)):
        val += run_episode(env, t, TeacherActor(ScriptedTeacher(0.0, seed + i)), prov).transitions
    return train, val


def gpu_mem():
    if not torch.cuda.is_available():
        return {}
    return {"peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}


def n_params(m):
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {"total_M": round(tot / 1e6, 1), "trainable_M": round(tr / 1e6, 2)}


def check_policy(cfg, train, val, steps, batch, out):
    from agentrs.models.policy import LMPolicy, PolicyConfig
    from agentrs.training.bc import BCConfig, make_examples, train_bc
    from agentrs.training.common import OptimConfig, load_model, save_checkpoint

    res = {}
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.time()
    pol = LMPolicy(PolicyConfig.model_validate(cfg))
    res["load_s"] = round(time.time() - t0, 1)
    res["device"] = str(pol.backbone.device)
    res["dtype"] = str(next(pol.backbone.model.parameters()).dtype)
    res["params"] = n_params(pol)

    # 1) free-text generation through the backbone
    t0 = time.time()
    out_text = pol.backbone.generate_batch(["Q: What is the capital of France?\nA:"], max_new_tokens=12,
                                           greedy=True)[0][0]
    res["generate"] = {"text": out_text.strip()[:120], "s": round(time.time() - t0, 2)}

    # 2) candidate-action scoring (the policy's actual decision path)
    ex = make_examples(val, BCConfig(), pol.cfg.max_candidates)[:8]
    t0 = time.time()
    with torch.no_grad():
        lps = pol.log_probs([e.prompt for e in ex], [e.candidates for e in ex])
    hits = sum(int(torch.argmax(lp)) == e.target for lp, e in zip(lps, ex))
    res["score"] = {"n": len(ex), "argmax_matches_teacher": hits, "s": round(time.time() - t0, 2),
                    "example_candidates": len(ex[0].candidates) if ex else 0}

    # 3) a short real BC run: weights must change and loss must be finite
    cfg_bc = BCConfig(steps=steps, batch_size=batch, eval_every=steps,
                      optim=OptimConfig(lr=2e-4, total_steps=steps, warmup_steps=min(5, steps // 4)))
    t0 = time.time()
    r = train_bc(pol, train, val, cfg_bc)
    dt = time.time() - t0
    losses = [h["loss"] for h in r["history"]]
    k = max(1, len(losses) // 4)
    res["bc"] = {"steps": steps, "batch": batch, "s_per_step": round(dt / steps, 2),
                 "loss_first_q": round(sum(losses[:k]) / k, 4), "loss_last_q": round(sum(losses[-k:]) / k, 4),
                 "val_nll_before": round(r["before"]["val_nll"], 4), "val_nll_after": round(r["after"]["val_nll"], 4),
                 "val_acc_before": round(r["before"]["val_acc"], 3), "val_acc_after": round(r["after"]["val_acc"], 3),
                 "param_change_l2": r["param_change_l2"], "all_finite": all(map(lambda x: x == x, losses))}
    res["mem"] = gpu_mem()

    # 4) checkpoint round trip
    path = os.path.join(out, "policy")
    save_checkpoint(path, pol, r["trainer"], extra={"stage": "gpu_check"})
    pol.eval()
    with torch.no_grad():
        a = pol.log_probs([ex[0].prompt], [ex[0].candidates])[0].float().cpu()
    del pol, r
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    pol2 = load_model(path)
    pol2.eval()
    with torch.no_grad():
        b = pol2.log_probs([ex[0].prompt], [ex[0].candidates])[0].float().cpu()
    res["checkpoint"] = {"path": path, "max_abs_logprob_diff": float((a - b).abs().max()),
                         "size_mb": round(sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) / 2**20, 1)}
    del pol2
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return res


def check_world_model(cfg, train, val, steps, batch):
    from agentrs.models.world_model import WorldModel, WorldModelConfig
    from agentrs.training.common import OptimConfig
    from agentrs.training.world_model import WMTrainConfig, train_world_model

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    wm = WorldModel(WorldModelConfig.model_validate(cfg))
    res = {"device": str(wm.backbone.device), "params": n_params(wm)}
    t0 = time.time()
    r = train_world_model(wm, train, val, WMTrainConfig(steps=steps, batch_size=batch, eval_every=steps,
                                                       optim=OptimConfig(lr=2e-4, total_steps=steps)))
    dt = time.time() - t0
    losses = [h["loss"] for h in r["history"]]
    k = max(1, len(losses) // 4)
    res.update({"steps": steps, "s_per_step": round(dt / steps, 2),
                "loss_first_q": round(sum(losses[:k]) / k, 4), "loss_last_q": round(sum(losses[-k:]) / k, 4),
                "param_change_l2": r.get("param_change_l2"), "mem": gpu_mem()})
    del wm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="research", help="model preset or YAML path")
    ap.add_argument("--components", default="policy,world_model")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--families", default="click_button,fill_field,menu")
    ap.add_argument("--out", default="runs/gpu_check")
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    report = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    train, val = demos(a.families.split(","), n_train=8, n_val=3)
    report["data"] = {"train_transitions": len(train), "val_transitions": len(val)}
    comps = a.components.split(",")
    if "policy" in comps:
        report["policy"] = check_policy(_model_cfg(a.model, "policy"), train, val, a.steps, a.batch_size, a.out)
        print(json.dumps({"policy": report["policy"]}, indent=1, default=str), flush=True)
    if "world_model" in comps:
        report["world_model"] = check_world_model(_model_cfg(a.model, "world_model"), train, val, a.steps,
                                                  a.batch_size)
        print(json.dumps({"world_model": report["world_model"]}, indent=1, default=str), flush=True)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("report:", os.path.join(a.out, "report.json"))
