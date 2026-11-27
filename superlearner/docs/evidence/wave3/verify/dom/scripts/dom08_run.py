"""Independent DOM-08 table at dom_dev through the canonical build_agent + run_lifetime path (stage1.dom_e2e),
plus per-parameter gradient coverage recorded by post-accumulate hooks (every trainable tensor)."""
import os, sys, json, time, collections
os.environ.setdefault("HF_HOME", "/workspace/.hf_home"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/workspace/agentrs/crucible/src")
import torch
import superlearner.cam_unified.build as BUILD
from superlearner.cam_unified.learn.grad_probe import classify, _type_hints
from superlearner.cam_unified.stage1 import dom_e2e
REC = {"cur": 0, "per": collections.defaultdict(dict), "names": {}, "model": None}
_orig = BUILD.build_agent
def wrapped(*a, **k):
    s = _orig(*a, **k)
    m = s.model; REC["model"] = m
    hints = _type_hints(m)
    for n, p in m.named_parameters():
        if not p.requires_grad: continue
        REC["names"][n] = (classify(n, m.cfg.n_layers, hints.get(n)), p.numel())
        def hook(p, n=n):
            if p.grad is not None:
                v = float(p.grad.detach().abs().max())
                d = REC["per"][REC["cur"]]
                d[n] = max(d.get(n, 0.0), v)
        p.register_post_accumulate_grad_hook(hook)
    # count updates: wrap learner.step
    L = s.learner; st0 = L.step
    def step(*a, **k):
        REC["cur"] += 1
        return st0(*a, **k)
    L.step = step
    return s
BUILD.build_agent = wrapped
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 24
out = dom_e2e.run_dom_e2e("dom_dev", n_rapid=4, n_ow=1, rounds=rounds, update_every=8, seed=0,
                          run_dir=f"/workspace/agentrs/runs/verify_dom/dom08_{int(time.time())}")
res = {"n_updates": out.get("n_updates"), "authorisation": out.get("authorisation"), "actions": out.get("actions"),
       "actions_stamp_verified": out.get("actions_stamp_verified"), "by_tool": out.get("by_tool"),
       "core_hash_changed": out.get("core_hash_changed"), "intent_permutation_kl": out.get("intent_permutation_kl"),
       "parameters": out.get("parameters"), "env_exceptions": out.get("env_exceptions")}
res["dom08_per_update"] = [u.get("dom08") for u in out["updates"] if isinstance(u, dict)]
res["grad_norm_by_objective"] = [u.get("grad_norm_by_objective") for u in out["updates"] if isinstance(u, dict)]
res["core_share"] = [u.get("core_share") for u in out["updates"] if isinstance(u, dict)]
res["update_errors"] = [u.get("error") for u in out["updates"] if isinstance(u, dict)]
# per-parameter coverage per update window
cov = {}
for upd, d in sorted(REC["per"].items()):
    grp_tot = collections.Counter(); grp_nz = collections.Counter(); zero = []
    for n, (g, k) in REC["names"].items():
        grp_tot[g] += k
        if d.get(n, 0.0) > 0: grp_nz[g] += k
        else: zero.append((n, k))
    cov[upd] = {"group_numel_nonzero_frac": {g: grp_nz[g] / grp_tot[g] for g in grp_tot},
                "zero_tensors": len(zero), "zero_numel": sum(k for _, k in zero),
                "zero_examples": sorted(zero, key=lambda x: -x[1])[:40]}
res["param_coverage"] = cov
# union across all updates
allnz = set(n for d in REC["per"].values() for n, v in d.items() if v > 0)
never = [(n, g, k) for n, (g, k) in REC["names"].items() if n not in allnz]
res["never_grad"] = {"n": len(never), "numel": sum(k for *_, k in never),
                     "by_group": dict(collections.Counter(g for _, g, _ in never)),
                     "numel_by_group": {g: sum(k for _, gg, k in never if gg == g) for g in set(g for _, g, _ in never)},
                     "list": sorted(never, key=lambda x: -x[2])[:80]}
print(json.dumps(res, indent=1, default=str))
