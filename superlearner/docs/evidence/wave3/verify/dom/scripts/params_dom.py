import os, sys, json, time, collections
os.environ.setdefault("HF_HOME", "/workspace/.hf_home"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/workspace/agentrs/crucible/src")
import torch
from superlearner.cam_unified.build import build_agent, describe
from superlearner.cam_unified.learn.grad_probe import classify, _type_hints
preset = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
t0 = time.time()
s = build_agent(preset, device=dev)
m = s.model
d = describe(s)
out = {"preset": preset, "build_s": time.time() - t0, "parameters": d["parameters"], "trainer": d["trainer"],
       "action_generator_parts": d["action_generator_parts"]}
# DOM-08 grouping of every trainable parameter
hints = _type_hints(m)
g = collections.Counter(); gtr = collections.Counter()
for n, p in m.named_parameters():
    grp = classify(n, m.cfg.n_layers, hints.get(n))
    g[grp] += p.numel()
    if p.requires_grad: gtr[grp] += p.numel()
out["dom08_groups_trainable"] = dict(gtr)
out["dom08_groups_total"] = dict(g)
# optimizer coverage
opt_ids = set()
for grp, ps in s.trainer.groups.items():
    for p in ps: opt_ids.add(id(p))
missing = [(n, p.numel()) for n, p in m.named_parameters() if p.requires_grad and id(p) not in opt_ids]
out["trainable_not_in_optimizer"] = {"n": len(missing), "numel": sum(x for _, x in missing), "first": missing[:20]}
# core detail
core = m.core
cc = collections.Counter()
for n, p in core.named_parameters():
    key = n.split(".")[0] if not n.startswith("blocks.") else "blocks"
    if "sem" in n or "xattn" in n: key = "sem_xattn:" + n.split(".")[0]
    cc[key] += p.numel()
out["core_breakdown"] = dict(cc)
out["core_total"] = sum(p.numel() for p in core.parameters())
out["core_trainable"] = sum(p.numel() for p in core.parameters() if p.requires_grad)
out["blocks"] = len(core.blocks) if hasattr(core, "blocks") else None
out["cfg"] = {k: getattr(m.cfg, k) for k in ("n_layers", "d_model", "n_heads", "n_state", "d_ff", "sem_xattn_every",
             "dec_xattn_every", "n_intent", "n_candidates", "reason_iters_fixed", "reason_layers", "d_satellite", "sem_model")}
# semantic objects
sem = s.semantic
out["semantic_class"] = type(sem).__name__
cond = getattr(sem, "conditioning", None)
out["dec_cond"] = None if cond is None else {"numel": sum(p.numel() for p in cond.parameters()),
    "registered_in_action_generator": any(cond is mm for mm in m.action_generator.modules()),
    "layers": getattr(cond, "layers", None)}
print(json.dumps(out, indent=1, default=str))
