import os, sys, json
os.environ.setdefault("HF_HOME", "/workspace/.hf_home"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/workspace/agentrs/crucible/src")
import torch
from superlearner.cam_unified.config import make_preset
from superlearner.cam_unified.model import build_model
from superlearner.cam_unified.build import make_semantic
cfg = make_preset("dom")
sem = make_semantic(cfg, None, "cpu")
m = build_model(cfg, components={"semantic": sem})
pref = ["core.cond_proj", "beliefs.act_cond", "world_model.rew_pool", "world_model.rew_head", "world_model.rew_src",
        "bayes.", "reasoner.halt_", "action_generator.bridge"]
out = {p: sum(x.numel() for n, x in m.named_parameters() if n.startswith(p)) for p in pref}
out["total_dead_candidates"] = sum(out.values())
print(json.dumps(out))
