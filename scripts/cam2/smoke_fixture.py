"""Plumbing smoke run on the unit-test fixture with a tiny random LLM (not an experiment)."""
import shutil
import sys

from agentrs.cam2.backbone import load_llm
from agentrs.cam2.envs.base import TaskRef
from agentrs.cam2.trainer import RunConfig, Trainer

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cam2_fixture"
    llm = load_llm("tiny-random", device="cuda")
    rc = RunConfig(run_dir=out, env="fixture", llm=llm.name, workers=4, env_steps=int(sys.argv[2]) if len(sys.argv) > 2 else 600,
                   max_steps=4, eval_every=2, eval_per_family=4, ckpt_every=2,
                   cam={"d": 64, "integ_layers": 1, "heads": 4, "core_dim": 64, "wm_hidden": 64, "head_hidden": 64,
                        "prefix_len": 4, "prefix_rank": 16, "mem_k": 4, "q_head": True},
                   learn={"episodes_per_update": 16, "mb_episodes": 8, "lr": 3e-4}, plan={"enabled": True})
    shutil.rmtree(rc.run_dir, ignore_errors=True)
    t = Trainer(rc, llm=llm)
    print(t.cam.n_params())
    print(t.train())
    ev = t.evaluate([TaskRef("fixture", "type", 5)])
    print(ev["success"], ev["episodes"])
    t.close()
