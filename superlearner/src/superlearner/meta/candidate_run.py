"""What runs INSIDE the confinement, from a candidate's own source tree. PROTECTED (identical in every candidate).

    python -m superlearner.meta.candidate_run job.json

``stage = "selftest"``  numerical and interface checks of the candidate's learner code on a few real episodes
                        obtained from the env hub: the code imports, an episode runs, the recomputed likelihood
                        equals the acting-time likelihood, losses and advantages are finite, an update changes the
                        weights and keeps them finite, gradients reach the controller's components.
``stage = "train"``     a fresh controller trained by the candidate's algorithm: optional cloning from the shared
                        demonstration file, then reinforcement learning until the iteration count is reached or the
                        hub's experience budget is exhausted, with periodic evaluation episodes started by index on
                        the hub.

Everything this process prints or writes is UNTRUSTED diagnostics. Scores come from the hub's own results file.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import traceback


def selftest(job: dict) -> dict:
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from superlearner.algo import rule as R
    from superlearner.algo.base import make
    from superlearner.algo.ppo import flat_params
    from superlearner.cam.batch import collate, evaluate
    from superlearner.train.rollout import LocalPool
    from superlearner.world.tasks import task
    out: dict = {"checks": {}}
    ck = out["checks"]
    cfg = job["config"]
    wcfg = {"llm": {"kind": "none"}, "model": cfg.get("model") or {}, "agent": dict(cfg.get("agent") or {}, use_llm=False),
            "core": cfg.get("core") or {}, "env_socket": job["env_socket"]}
    pool = LocalPool(wcfg)
    from multiprocessing.connection import Client
    from superlearner.world.core import RemoteCore
    from superlearner.world.hub import AUTHKEY
    pool.agent.core_hub = RemoteCore(Client(job["env_socket"], family="AF_UNIX", authkey=AUTHKEY))
    from superlearner.train import rollout as RO
    eps = []
    for i, fam in enumerate(("found", "fund", "found")):
        ep = RO.run_job(pool.agent, {"spec": task(fam, "tiny", 900_000 + i).to_dict(), "mode": "sample", "seed": i,
                                     "core": {"max_decisions": 6}}, wcfg, pool.agent.core_hub)
        eps.append(ep)
    ck["episodes_run"] = all(len(e["decisions"]) > 0 for e in eps)
    model, store = pool.model, pool.store
    b = collate([e["decisions"] for e in eps], store)
    with torch.no_grad():
        ev = evaluate(model, b, store)
    v = b.t["valid"]
    err = float((ev["logp"][v] - b.t["logp_old"][v]).abs().max())
    ck["likelihood_recomputation_matches_acting"] = bool(err < 2e-3)
    out["max_logp_error"] = err
    algo = make(cfg.get("algo", {}).get("name", "ppo"), model, {k: v_ for k, v_ in cfg.get("algo", {}).items() if k != "name"}, "cpu")
    adv, ret = algo.advantages(b) if hasattr(algo, "advantages") else R.advantages(
        b.t["reward"], b.t["value_old"], b.t["done"], b.t["trunc"], b.t["valid"], algo.rule)
    ck["advantages_finite"] = bool(torch.isfinite(adv).all() and torch.isfinite(ret).all())
    ck["advantages_shape"] = tuple(adv.shape) == tuple(b.t["reward"].shape)
    before = flat_params(model).clone()
    m = algo.update(eps, store)
    after = flat_params(model)
    ck["update_metrics_finite"] = all(np.isfinite(x) for x in m.values() if isinstance(x, (int, float)))
    ck["weights_changed"] = bool((after != before).any())
    ck["weights_finite"] = bool(torch.isfinite(after).all())
    ck["no_nonfinite_losses"] = m.get("nonfinite", 0) == 0
    model.train()
    ev2 = evaluate(model, b, store)
    if hasattr(algo, "n_calls") and isinstance(getattr(algo, "c", None), dict):
        # a critic-only warm-up phase deliberately sends no gradient to the policy; test the loss used after it
        algo.n_calls = max(int(algo.n_calls), int(algo.c.get("critic_warmup", 0) or 0) + 1)
    loss, _ = (algo.loss(ev2, b, adv, ret, None) if hasattr(algo, "loss") else R.total_loss(ev2, b, adv, ret, algo.rule))
    model.zero_grad()
    loss.backward()
    groups: dict = {}
    for n, p in model.named_parameters():
        g = n.split(".")[0]
        groups[g] = groups.get(g, 0.0) + (float(p.grad.abs().sum()) if p.grad is not None else 0.0)
    out["grad_groups"] = {k: round(v_, 6) for k, v_ in groups.items()}
    ck["gradients_reach_policy_and_value"] = all(groups.get(g, 0) > 0 for g in ("encoder", "head_score", "value"))
    out["ok"] = all(ck.values())
    return out


def train(job: dict) -> dict:
    import torch
    torch.set_num_threads(int(job.get("update_threads", 4)))
    from superlearner.train.trainer import Trainer
    cfg = dict(job["config"])
    cfg["env_socket"] = job["env_socket"]
    cfg["llm"] = job.get("llm") or {"kind": "none"}
    cfg["device"] = "cpu"
    cfg["hub_eval_n"] = int(job.get("eval_n", 0))
    tr = Trainer(cfg, os.path.join(job["workdir"], "run"))
    t0 = time.time()
    info = {"iters": 0, "stopped": ""}
    try:
        if job.get("demos_path") and int(cfg.get("demos", {}).get("bc_epochs", 0)) > 0:
            d = pickle.load(open(job["demos_path"], "rb"))
            for k, v in d.get("emb", []):
                tr.store.put(k, v)
            tr.demos = d["episodes"]
            tr.pretrain_bc(int(cfg["demos"]["bc_epochs"]))
        tr.evaluate_hub("eval")
        while tr.iter < int(cfg["iters"]):
            rec = tr.train_iter()
            info["iters"] = tr.iter
            if rec.get("train_episodes", 0) == 0:
                info["stopped"] = "no episodes (experience budget exhausted)"
                break
            if tr.iter % int(cfg.get("eval_every", 5)) == 0:
                tr.evaluate_hub("eval")
        tr.evaluate_hub("eval")
        tr.save()
    finally:
        tr.close()
    info.update(wall_s=round(time.time() - t0, 1), env_decisions=tr.env_decisions, worker_errors=tr.pool.errors[:3],
                update_s=round(tr.wall["update_s"], 1), rollout_s=round(tr.wall["rollout_s"], 1), params=tr.model.n_params())
    return info


def main() -> int:
    job = json.load(open(sys.argv[1]))
    out_path = os.path.join(job["workdir"], f"{job['stage']}_result.json")
    try:
        res = selftest(job) if job["stage"] == "selftest" else train(job)
        res.setdefault("ok", True)
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
    json.dump(res, open(out_path, "w"), indent=1, default=str)
    print("CANDIDATE_RESULT " + json.dumps({k: res[k] for k in res if k not in ("traceback",)}, default=str)[:3000])
    return 0 if res.get("ok") else 3


if __name__ == "__main__":
    sys.exit(main())
