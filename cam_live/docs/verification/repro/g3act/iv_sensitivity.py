"""G3-act: is the realised information value (agent._information_value) computed on the counterfactual it claims,
and how sensitive are CAM's belief and world-model predictions to removing one decision's items from the context?

For every record that carries ``reward_signals.information_value`` the measurement of ``_information_value`` is
re-done independently from the store (same "new items" set, same 2 later decisions, same context removal) and, in
addition, three sensitivity numbers are computed with the same weights:

* ``iv_recomputed`` - NLL(later | context without the items) - NLL(later | recorded context), to compare with the
  stored annotation;
* ``rel_db`` - ||b_with - b_without|| / ||b_with|| of the later decisions' belief (how much the removed items move the
  experiential core's belief at all);
* ``rel_db_neighbour`` - ||b(record i) - b(record i+1)|| / ||b(record i)|| for consecutive decisions (the scale on which
  the belief moves from one decision to the next);
* ``nll_drop_all_but_3`` - NLL with only the newest 3 context items minus NLL with the recorded context (an upper
  bound on how much the world model's predictions of these targets use the context at all).

Modes:
  debug: the debug-scale agent of tests/integration/test_st_strategy (CPU, random tiny backbone); three decisions
         as in test_st_iv; checks the plumbing.
  real:  --config <run config.yaml> --checkpoint <COPY of a checkpoint dir> --learner-dir <run>/learner (copied):
         the real configuration (Qwen3-4B + 50m core) under the shared GPU lock; nothing of the run is written.

    PYTHONPATH=<tree>/cam_live/src .venv/bin/python iv_sensitivity.py debug
    CAM_LIVE_GPU_LOCK=... PYTHONPATH=<tree>/cam_live/src .venv/bin/python iv_sensitivity.py real --config C --checkpoint K --store S --out O.json
"""
import argparse
import dataclasses
import json
import math
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))


def measure(agent, store, records, max_n=40, budget_s=1e9):
    import time
    t_start = time.time()
    from cam_live.learning_api import encode_decisions
    from cam_live.predict.providers import record_rows, record_world_nll
    later_n = max(1, int(agent.settings.information_value_later))
    model, heads = agent.model, agent.heads
    was = model.training
    model.eval()
    out = []
    by_obj = {}
    for r in records:
        by_obj.setdefault(r.objective_id, []).append(r)
    try:
        for rec in records:
            d = rec.decision.diagnostics if isinstance(rec.decision.diagnostics, dict) else {}
            if "reward_signals.information_value" not in d or rec.result is None:
                continue
            new = set(rec.post_item_refs or []) - set(rec.context_item_refs or [])
            later = [r for r in sorted(by_obj[rec.objective_id], key=lambda x: x.seq)
                     if r.seq > rec.seq and r.result is not None and new & set(r.context_item_refs or [])
                     and record_rows([r])[0]][:later_n]
            if len(later) < later_n:
                continue
            cf = [dataclasses.replace(r, context_item_refs=[x for x in r.context_item_refs if x not in new]) for r in later]
            few = [dataclasses.replace(r, context_item_refs=list(r.context_item_refs)[-3:]) for r in later]
            with torch.no_grad():
                a = record_world_nll(model, heads, store, later)
                b = record_world_nll(model, heads, store, cf)
                c = record_world_nll(model, heads, store, few)
                bw = encode_decisions(model, store, later, grad=False).b.float()
                bo = encode_decisions(model, store, cf, grad=False).b.float()
            pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            iv = sum(y - x for x, y in pairs) / len(pairs) if pairs else None
            drop = [z - x for x, z in zip(a, c) if x is not None and z is not None]
            rel = float(((bw - bo).norm(dim=-1) / bw.norm(dim=-1).clamp_min(1e-12)).mean())
            out.append({"exp_id": rec.exp_id, "op": rec.decision.chosen.operation, "iv_stored": float(d["reward_signals.information_value"]),
                        "iv_recomputed": iv, "items_removed": len(new),
                        "later_context_items": [len(r.context_item_refs) for r in later],
                        "removed_in_later": [len(set(r.context_item_refs) & new) for r in later],
                        "rel_db": rel, "nll_with": a, "nll_drop_all_but_3": sum(drop) / len(drop) if drop else None})
            if len(out) >= max_n or time.time() - t_start > budget_s:
                break
        # natural scale of belief change between consecutive decisions
        seq = sorted([r for r in records if r.result is not None], key=lambda x: x.seq)[:max_n + 1]
        neigh = []
        for i in range(0, len(seq) - 1, 8):
            if time.time() - t_start > budget_s * 1.2:
                break
            chunk = seq[i:i + 9]
            with torch.no_grad():
                bb = encode_decisions(model, store, chunk, grad=False).b.float()
            for j in range(len(chunk) - 1):
                neigh.append(float((bb[j + 1] - bb[j]).norm() / bb[j].norm().clamp_min(1e-12)))
    finally:
        model.train(was)
    return out, neigh


def summary(rows, neigh):
    def med(xs):
        xs = sorted(x for x in xs if x is not None and math.isfinite(x))
        return xs[len(xs) // 2] if xs else None
    diffs = [abs(r["iv_recomputed"] - r["iv_stored"]) for r in rows if r["iv_recomputed"] is not None]
    return {"n": len(rows), "median_rel_db": med([r["rel_db"] for r in rows]), "max_rel_db": max((r["rel_db"] for r in rows), default=None),
            "median_rel_db_neighbour": med(neigh), "median_abs_iv": med([abs(r["iv_recomputed"]) for r in rows if r["iv_recomputed"] is not None]),
            "max_abs_iv_recomputed_minus_stored": max(diffs) if diffs else None,
            "median_nll_drop_all_but_3": med([r["nll_drop_all_but_3"] for r in rows]),
            "median_nll_with": med([sum(x for x in r["nll_with"] if x is not None) / max(1, sum(x is not None for x in r["nll_with"])) for r in rows])}


def debug_mode():
    sys.path.insert(0, os.path.join(ROOT, "tests", "integration"))
    from test_st_iv import _step
    from test_st_strategy import _agent, _fixed_generator
    from cam_live.contracts import Observation
    tmp = tempfile.mkdtemp(prefix="g3act-iv-")
    import pathlib
    agent = _agent(pathlib.Path(tmp))
    _fixed_generator(agent, [['{"operation": "internal.note", "arguments": {"text": "the page lists 3 groups"}}\n'
                              '{"operation": "web.get", "arguments": {"url": "https://a.example/"}}']])
    try:
        agent.ingest([Observation(obs_id="obs_0", ts=agent.clock.now(), kind="metric", text="start")])
        for op, text in [("internal.note", "internal.note: saved"), ("web.get", "page text: groups A, B and C"),
                         ("web.get", "page text: group A has 40 members"), ("web.get", "page text: group B has 7"),
                         ("internal.note", "internal.note: saved")]:
            _step(agent, op, text)
        agent._information_value()
        recs = sorted(agent.store.all(), key=lambda r: r.seq)
        rows, neigh = measure(agent, agent.store, recs)
        print(json.dumps({"mode": "debug", "summary": summary(rows, neigh), "rows": rows}, indent=1, default=str))
    finally:
        agent.close()


def real_mode(a):
    from cam_live.agent import CamLiveAgent
    from cam_live.config import CamLiveConfig
    from cam_live.contracts import SimClock
    from cam_live.credit.replay import SqliteExperienceStore
    from cam_live.nn.gpu import gpu_lock
    cfg = CamLiveConfig.load(a.config)
    d = cfg.to_dict()
    d["home"] = a.home
    cfg = CamLiveConfig.from_dict(d)
    import shutil
    # the agent restores its meta-learner against the outcome ledger of its own directory: the agent directory is a
    # COPY of the run's learner directory (store, ledger, registries; never the run's own files)
    adir = tempfile.mkdtemp(prefix="g3act-ivr-", dir=a.home)
    src = a.learner_dir
    for f in os.listdir(src):
        if f in ("runtime.key", "promoted") or f.endswith(".lock"):
            continue
        p = os.path.join(src, f)
        (shutil.copytree if os.path.isdir(p) else shutil.copy2)(p, os.path.join(adir, f))
    with gpu_lock(timeout_s=3600.0):
        import threading
        print("gpu lock acquired", flush=True)
        wd = threading.Timer(780.0, lambda: (print("watchdog: 13 min in the lock, exiting", flush=True), os._exit(3)))
        wd.daemon = True
        wd.start()                                       # never hold the shared lock longer than 13 minutes
        agent = CamLiveAgent(cfg, adir, clock=SimClock())
        try:
            agent.load(a.checkpoint)
            recs = sorted(agent.store.all(), key=lambda r: r.seq)
            rows, neigh = measure(agent, agent.store, recs, max_n=a.max_n, budget_s=a.budget_s)
        finally:
            agent.close()
    res = {"mode": "real", "config": a.config, "checkpoint": a.checkpoint, "summary": summary(rows, neigh), "rows": rows}
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["debug", "real"])
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--store")
    ap.add_argument("--learner-dir", help="the run's learner directory (copied, never written)")
    ap.add_argument("--home", default=tempfile.gettempdir())
    ap.add_argument("--out", default="iv_sensitivity_real.json")
    ap.add_argument("--max-n", type=int, default=40)
    ap.add_argument("--budget-s", type=float, default=420.0)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    debug_mode() if args.mode == "debug" else real_mode(args)
