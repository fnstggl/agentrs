"""PS: prequential replay of a FIXED experience stream with and without the persistent core state — the paired,
low-noise version of "prediction NLL over experience" (both arms see exactly the same decisions, observations and
outcomes, in the same order; only ``persistent_state`` differs).

For each stored decision r_t of the source store, in lifetime order:
  1. (on) commit r_t to the chain — its anchor is the state recomputed with the current weights, as at decision time;
  2. score: the world model's NLL of r_t's matured outcomes (``predict.providers.record_world_nll``) BEFORE any
     training on r_t (prequential);
  3. learn: ``--steps`` AdamW steps of the agent's world-model loss provider (``predict.providers.WorldModelLoss``:
     ensemble NLL + fast-prior term + latent transition) on a replay batch (r_t + uniform earlier records), per-group
     clipping as the continual learner (experiential 1.0, heads 1.0), learning rates of ``config.OptimConfig``.
After each family block: retention = NLL of the first block's records under the current weights.

Outcomes are attached as the source store holds them at the end of its run (a delayed outcome of an earlier record
is trained on from its record's replay onward: the same look-ahead in both arms).

    python ps_replay.py --source STORE.sqlite --cond on|off --seed 0 --out OUT.json [--real] [--steps 2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List

import torch

torch.set_num_threads(int(os.environ.get("PS_THREADS", "4")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="experience.sqlite of a finished run (copied first); several "
                                                     "comma-separated: one process, one loaded backbone")
    ap.add_argument("--cond", default="on", help="on | off | on,off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--unroll", type=int, default=2)
    ap.add_argument("--rho-init", type=float, default=None)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--no-lock", action="store_true")
    a = ap.parse_args()
    from cam_live.config import CamLiveConfig
    cfg = CamLiveConfig.load() if a.real else CamLiveConfig.debug()
    cfg.device = "cuda" if a.real else "cpu"
    if not a.real:
        cfg.backbone.tiny_vocab = 4096
    cfg.seed = int(a.seed)
    lock = None
    if a.real and not a.no_lock:
        from cam_live.nn.gpu import gpu_lock
        lock = gpu_lock(min_free_gb=12.0, timeout_s=4 * 3600.0, poll_s=1.0)
        lock.__enter__()
    results = []
    shared = {"bb": None}
    try:
        for source in a.source.split(","):
            for cond in a.cond.split(","):
                results.append(_one(a, cfg, source, cond, shared))
                json.dump(results if len(results) > 1 else results[0], open(a.out, "w"), indent=1, default=str)
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)
    return 0


def _one(a: Any, cfg: Any, source: str, cond: str, shared: Dict[str, Any]) -> Dict[str, Any]:
    from cam_live.credit.replay import SqliteExperienceStore
    from cam_live.learning_api import encode_decisions
    from cam_live.nn.api import GROUP_EXPERIENTIAL, GROUP_HEADS
    from cam_live.nn.model import CamLiveModel, build_model
    from cam_live.nn.persist import PersistentStateChain, PersistSettings
    from cam_live.predict.heads import attach_heads
    from cam_live.predict.providers import WorldModelLoss, record_rows, record_world_nll
    t0 = time.time()
    work = tempfile.mkdtemp(prefix="ps_replay_")
    for suffix in ("", "-wal"):
        if os.path.exists(source + suffix):
            shutil.copyfile(source + suffix, os.path.join(work, "experience.sqlite" + suffix))
    if os.path.isdir(source + ".blobs"):
        shutil.copytree(source + ".blobs", os.path.join(work, "experience.sqlite.blobs"))
    if True:
        torch.manual_seed(a.seed)
        if shared["bb"] is None:                  # one backbone per process; its adapters are not trained here
            model = build_model(cfg)
            shared["bb"] = model.backbone
        else:
            model = CamLiveModel(cfg, backbone=shared["bb"])
        heads = attach_heads(model)
        store = SqliteExperienceStore(os.path.join(work, "experience.sqlite"), durable=False)
        chain = None
        if cond == "on":
            chain = PersistentStateChain(model, store, PersistSettings(enabled=True, unroll=a.unroll))
            if a.rho_init is not None:
                with torch.no_grad():
                    model.exp.core.persist_rho.fill_(float(a.rho_init))
        recs = [r for r in store.all() if r.result is not None and r.decision is not None]
        if a.max_records:
            recs = recs[:a.max_records]
        prov = WorldModelLoss(heads)
        groups = model.param_groups()
        o = cfg.optim
        params = {GROUP_EXPERIENTIAL: [p for p in groups[GROUP_EXPERIENTIAL] if p.requires_grad],
                  GROUP_HEADS: [p for p in groups[GROUP_HEADS] if p.requires_grad]}
        opt = torch.optim.AdamW([{"params": params[GROUP_EXPERIENTIAL], "lr": o.lr_experiential},
                                 {"params": params[GROUP_HEADS], "lr": o.lr_heads}], weight_decay=0.0)
        rng = random.Random(a.seed)
        fam = [r.objective_family or r.objective_id for r in recs]
        blocks: List[List[int]] = []
        for i, f in enumerate(fam):
            if not blocks or fam[blocks[-1][-1]] != f:
                blocks.append([])
            blocks[-1].append(i)
        rows_out: List[Dict[str, Any]] = []
        retention: List[Dict[str, Any]] = []
        trained: List[int] = []
        model.train()
        for bi, blk in enumerate(blocks):
            for i in blk:
                r = recs[i]
                if chain is not None:
                    chain.commit(r.decision.decision_id, context_refs=r.context_item_refs, post_refs=r.post_item_refs,
                                 objective_id=r.objective_id, ts=float(r.decision.ts))
                has_targets = bool(record_rows([r])[0])
                nll = None
                if has_targets:
                    model.eval()
                    with torch.no_grad():
                        nll = record_world_nll(model, heads, store, [r])[0]
                    model.train()
                rows_out.append({"i": i, "block": bi, "family": fam[i], "objective": r.objective_id, "nll": nll})
                if prov.eligible(r):
                    trained.append(i)
                if not trained:
                    continue
                for _ in range(a.steps):
                    idx = [trained[-1]] + [rng.choice(trained) for _ in range(a.batch - 1)]
                    batch_recs = [recs[j] for j in idx]
                    b = encode_decisions(model, store, batch_recs, with_next=True)
                    loss, _m = prov.loss(model, b, store)
                    if loss is None or not torch.isfinite(loss):
                        continue
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params[GROUP_EXPERIENTIAL], 1.0)
                    torch.nn.utils.clip_grad_norm_(params[GROUP_HEADS], 1.0)
                    opt.step()
            first = [recs[i] for i in blocks[0] if record_rows([recs[i]])[0]]
            model.eval()
            with torch.no_grad():
                v = [x for x in record_world_nll(model, heads, store, first) if x is not None and math.isfinite(x)]
            model.train()
            retention.append({"after_block": bi, "family": fam[blk[0]], "first_block_nll": (sum(v) / len(v)) if v else None,
                              "n": len(v)})
            print(json.dumps(retention[-1]), flush=True)
        by_block = []
        for bi, blk in enumerate(blocks):
            v = [x["nll"] for x in rows_out if x["block"] == bi and x["nll"] is not None and math.isfinite(x["nll"])]
            by_block.append({"block": bi, "family": fam[blk[0]], "n_records": len(blk), "n_scored": len(v),
                             "prequential_nll": (sum(v) / len(v)) if v else None})
        allv = [x["nll"] for x in rows_out if x["nll"] is not None and math.isfinite(x["nll"])]
        out = {"source": source, "cond": cond, "seed": a.seed, "real": a.real, "steps": a.steps,
               "batch": a.batch, "unroll": a.unroll, "rho_init": a.rho_init, "n_records": len(recs),
               "prequential_nll": (sum(allv) / len(allv)) if allv else None, "n_scored": len(allv),
               "by_block": by_block, "retention": retention, "records": rows_out,
               "rho": chain.rho() if chain is not None else None,
               "chain_counters": dict(chain.counters) if chain is not None else None,
               "seconds": time.time() - t0}
    if chain is not None:
        chain.close()
    store.close()
    print(json.dumps({k: out[k] for k in ("source", "cond", "seed", "prequential_nll", "n_scored", "seconds")}),
          flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return out


if __name__ == "__main__":
    sys.exit(main())
