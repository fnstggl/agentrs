"""PS compatibility on the real configuration: a checkpoint of CAM's live planning run (written before PS) restored by
the PS code (agent with the persistent-state chain), on a *copy* of the run's learner directory (the live run directory
is only read: files are copied with plain reads).

Measured:
1. restore report (legacy checkpoint -> retention rho closed), parameters missing from the checkpoint;
2. beliefs of the run's stored records with the chain attached vs detached (must be bit-identical: rho = 0, and no
   stored record has a chain position);
3. counterfactual: the run's stored decisions replayed into a chain in order (anchors by the recurrence with the
   restored weights); belief change and world-model NLL change on the records if the retention were open (rho = 1)
   instead of closed — how much an upgrade that carried the state at once would have moved the live run's beliefs.

    python ps_compat_live.py --run-dir RUN --ckpt CKPT_NAME --work DIR --out OUT.json     (inside the GPU lock)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", type=int, default=24)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    src_learner = os.path.join(a.run_dir, "learner")
    ck_root = os.path.join(a.run_dir, "checkpoints")
    name = a.ckpt or open(os.path.join(ck_root, "LATEST")).read().strip()
    work = os.path.abspath(a.work)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, "learner"))
    # plain reads of the live files: the store (database, write-ahead log, blob directory) and the agent's logs and
    # registries; never the run's keys (runtime.key) and never the shared-memory index (the copy rebuilds it)
    for f in glob.glob(os.path.join(src_learner, "*")):
        base = os.path.basename(f)
        dst = os.path.join(work, "learner", base)
        if base in ("runtime.key", "promoted") or base.endswith(("-shm", ".lock")):
            continue
        if os.path.isdir(f):
            shutil.copytree(f, dst)
        else:
            shutil.copyfile(f, dst)
    shutil.copytree(os.path.join(ck_root, name), os.path.join(work, "ckpt"))
    from cam_live.config import CamLiveConfig
    from cam_live.agent import CamLiveAgent
    from cam_live.learning_api import encode_decisions
    cfg = CamLiveConfig.load()
    cfg.device = "cpu" if a.cpu else "cuda"
    sec = dict(cfg.sections.get("agent", {}))
    sec["durable"] = False
    cfg.sections["agent"] = sec
    lock = None
    if not a.cpu:
        from cam_live.nn.gpu import gpu_lock
        lock = gpu_lock(timeout_s=4 * 3600.0, poll_s=1.0)
        lock.__enter__()
    out = {"checkpoint": name, "run_dir": a.run_dir}
    try:
        agent = CamLiveAgent(cfg, work)
        agent.load(os.path.join(work, "ckpt"))
        rep = agent.restore_report
        out["restore_persistent_state"] = rep.get("persistent_state")
        out["weights_missing"] = list(getattr(rep.get("weights"), "missing", []) or [])
        out["arch_signature_differences"] = rep.get("arch_signature_differences")
        recs = [r for r in agent.store.all() if r.result is not None]
        out["records_in_store"] = len(recs)
        out["persist_rho_after_restore"] = agent.persist.rho()
        sample = recs[-a.records:] if a.records > 0 else []
        model = agent.model
        if not sample:                                    # restore report only (CPU: no backbone forwards)
            out["seconds"] = time.time() - t0
            json.dump(out, open(a.out, "w"), indent=1, default=str)
            print(json.dumps(out, indent=1, default=str))
            agent.close()
            return 0
        with torch.no_grad():
            b_chain = encode_decisions(model, agent.store, sample, with_next=True, grad=False)
            chain = agent.persist
            chain.detach()
            b_plain = encode_decisions(model, agent.store, sample, with_next=True, grad=False)
            chain.attach()
            out["chain_vs_plain_bit_identical"] = bool(torch.equal(b_chain.b, b_plain.b)
                                                         and torch.equal(b_chain.b_next, b_plain.b_next))
            nll_closed = agent._world_nll(sample)
            # counterfactual: the run's decisions replayed into a fresh in-memory chain, in order
            from cam_live.nn.persist import PersistentStateChain, PersistSettings
            replay = PersistentStateChain(model, agent.store, PersistSettings(unroll=2))
            for r in recs:
                replay.commit(r.decision.decision_id, context_refs=r.context_item_refs, post_refs=r.post_item_refs,
                              objective_id=r.objective_id, ts=float(r.decision.ts))
            b_rep_closed = encode_decisions(model, agent.store, sample, grad=False).b
            core = model.exp.core
            saved = core.persist_rho.detach().clone()
            core.persist_rho.fill_(1.0)
            b_rep_open = encode_decisions(model, agent.store, sample, grad=False).b
            nll_open = agent._world_nll(sample)
            core.persist_rho.copy_(saved)
            replay.detach()
            chain.attach()
        rel = ((b_rep_open - b_rep_closed).norm(dim=-1) / b_rep_closed.norm(dim=-1)).tolist()
        cos = torch.nn.functional.cosine_similarity(b_rep_open, b_rep_closed, -1).tolist()
        pairs = [(c, o) for c, o in zip(nll_closed, nll_open) if c is not None and o is not None
                 and math.isfinite(c) and math.isfinite(o)]
        out["counterfactual_open_rho"] = {
            "replayed_positions": replay.n, "records_compared": len(sample),
            "rel_belief_change_mean": sum(rel) / len(rel), "rel_belief_change_max": max(rel),
            "cosine_min": min(cos), "closed_equals_plain": bool(torch.equal(b_rep_closed, b_plain.b)),
            "world_nll_closed_mean": (sum(c for c, _ in pairs) / len(pairs)) if pairs else None,
            "world_nll_open_mean": (sum(o for _, o in pairs) / len(pairs)) if pairs else None,
            "world_nll_pairs": len(pairs)}
        out["rho"] = chain.rho()
        out["status"] = chain.status()
        agent.close()
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)
    out["seconds"] = time.time() - t0
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
