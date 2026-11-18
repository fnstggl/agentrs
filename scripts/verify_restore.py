"""Real upload-and-restore check for GitHub checkpoint bundles.

Downloads the latest (or given) checkpoint release, reassembles and verifies it (part + tar + per-file
sha256), then loads the restored policy / world-model / critic weights and optimizer states into
freshly built models and checks that the restored policy reproduces the original checkpoint's
predictions exactly (token log-probs on real stored prompts), and that those predictions differ from
the untrained base model (i.e. the learned LoRA weights really are in the bundle).

    python scripts/verify_restore.py --run runs/overnight [--tag ckpt-...] --dest /tmp/restore
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import torch

from agentrs.gh_backup import restore, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/overnight")
    ap.add_argument("--tag")
    ap.add_argument("--dest", default="/tmp/agentrs_restore_test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    res = restore(Path(a.dest), tag=a.tag)
    b = Path(res["dir"])
    info = json.loads((b / "BUNDLE_INFO.json").read_text())
    ck = info["checkpoint"]
    rck = b / "payload" / "ckpt" / ck
    lck = Path(a.run) / "ckpt" / ck
    report = {"tag": res["tag"], "checkpoint": ck, "download_verify_s": round(time.time() - t0), "files": res["files"]}
    same = {}
    for f in sorted(rck.rglob("*")):
        if f.is_file() and (lck / f.relative_to(rck)).exists():
            same[str(f.relative_to(rck))] = sha256_file(f) == sha256_file(lck / f.relative_to(rck))
    report["byte_identical_to_local_checkpoint"] = all(same.values()) and bool(same)
    report["compared_files"] = len(same)

    from agentrs.econ.learner import GenPolicy, policy_backbone_cfg, small_backbone_cfg
    from agentrs.models.backbone import LMBackbone
    from agentrs.models.critic import Critic, CriticConfig
    from agentrs.training.common import GradientTrainer, OptimConfig, load_trainer_state, load_weights

    models = info["models"]
    pol = GenPolicy(policy_backbone_cfg(models["policy"]), max_new_tokens=64)
    # real stored prompts: BC demonstrations / episodes from the run's replay
    prompts, ids = [], []
    exp = b / "payload" / "run" / "experience"
    for p in sorted(exp.glob("*.jsonl.gz")):
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                for s in r["steps"]:
                    if s.get("prompt") and s.get("token_ids"):
                        prompts.append(s["prompt"])
                        ids.append(s["token_ids"])
                if len(prompts) >= 6:
                    break
        if len(prompts) >= 6:
            break
    if not prompts:  # no agent episodes yet: score expert actions on a rendered task
        from agentrs.econ.core import load_all, make_task
        from agentrs.econ.rollout import make_env, render

        load_all()
        for k in ("support/return_items", "finance/payment_matching", "crm/pipeline_update"):
            t = make_task(k, "val", 0)
            msgs, pre = render(make_env(t), t, [])
            prompts.append(pol.chat_text(msgs, pre))
            ids.append(pol.tok('find_customer", "args": {}}\n', add_special_tokens=False)["input_ids"])

    def lp():
        with torch.no_grad():
            v, m = pol.token_logps(prompts, ids)
        return (v * m).sum(-1).float().cpu()

    base = lp()
    load_weights(lck / "policy", pol.bb)
    orig = lp()
    # reset LoRA to base, then load the restored weights
    for n, p in pol.bb.named_parameters():
        if p.requires_grad and "lora_B" in n:
            p.data.zero_()
    assert torch.allclose(lp(), base), "LoRA reset did not restore base behaviour"
    load_weights(rck / "policy", pol.bb)
    rest = lp()
    pt = GradientTrainer(pol.bb, OptimConfig(lr=1e-5, total_steps=10))
    load_trainer_state(rck / "policy", pt, restore_rng=False)
    report["policy"] = {"n_prompts": len(prompts), "max_abs_diff_restored_vs_original": float((rest - orig).abs().max()),
                        "mean_abs_diff_original_vs_base": float((orig - base).abs().mean()),
                        "optimizer_state_loaded": True, "optimizer_step": pt.global_step,
                        "logp_original": [round(x, 4) for x in orig.tolist()],
                        "logp_restored": [round(x, 4) for x in rest.tolist()],
                        "logp_base": [round(x, 4) for x in base.tolist()]}
    del pol, pt
    torch.cuda.empty_cache()
    wm = LMBackbone(small_backbone_cfg(models["small"]))
    load_weights(rck / "world_model", wm)
    load_trainer_state(rck / "world_model", GradientTrainer(wm, OptimConfig(lr=1e-4, total_steps=10)), restore_rng=False)
    cr = Critic(CriticConfig(backbone=small_backbone_cfg(models["small"])))
    load_weights(rck / "critic", cr)
    load_trainer_state(rck / "critic", GradientTrainer(cr, OptimConfig(lr=1e-4, total_steps=10)), restore_rng=False)
    torch.load(rck / "rng.pt", weights_only=False)
    report["world_model_and_critic_loaded"] = True
    report["state_json"] = {k: json.loads((rck / "state.json").read_text()).get(k) for k in ("phase", "version", "cycle")}
    report["passed"] = (report["byte_identical_to_local_checkpoint"] and report["policy"]["max_abs_diff_restored_vs_original"] == 0.0
                        and report["policy"]["mean_abs_diff_original_vs_base"] > 0)
    report["seconds"] = round(time.time() - t0)
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
