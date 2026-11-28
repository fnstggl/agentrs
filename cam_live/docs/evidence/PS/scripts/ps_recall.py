"""PS engineering diagnostic (debug scale, CPU): can the persistent-state chain be *learned* the way the continual
learner trains it — replayed stored decisions, beliefs resolved through the chain (state recomputed K transitions
from the stored anchor), one optimiser step per new decision — to carry information the context window no longer
holds?

Stream: every ``period`` decisions a fact item ("the code is <k>", k in 0..n_cls-1) is observed; the other decisions
observe distractors. The window holds the newest ``width`` items, so after ``width`` decisions the fact has left it.
Target at every decision: the most recent fact. Prequential scoring (predict at decision time, then commit, then
learn on a replay batch). Accuracy by lag since the fact: lag < width is answerable from the window, lag >= width only
through the carried state. Chance = 1 / n_cls.

    python ps_recall.py --out DIR [--steps 600] [--period 8] [--width 2] [--seeds 0,1,2]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List

import torch
import torch.nn.functional as Fn

torch.set_num_threads(int(os.environ.get("PS_THREADS", "4")))     # a shared machine: never oversubscribe the CPU


def run(cond: str, seed: int, steps: int, period: int, width: int, n_cls: int, batch: int, lr: float) -> Dict[str, Any]:
    from cam_live.config import CamLiveConfig
    from cam_live.contracts import ExperienceItem, ItemKind, Quantity
    from cam_live.credit.replay import SqliteExperienceStore
    from cam_live.nn.api import GROUP_EXPERIENTIAL
    from cam_live.nn.model import build_model
    from cam_live.nn.persist import PersistentStateChain, PersistSettings
    torch.manual_seed(seed)
    rng = random.Random(seed)
    cfg = CamLiveConfig.debug()
    cfg.seed = seed
    model = build_model(cfg)
    head = torch.nn.Linear(model.d_model, n_cls)
    d = tempfile.mkdtemp()
    store = SqliteExperienceStore(os.path.join(d, "s.sqlite"), durable=False)
    chain = None
    if cond != "off":
        chain = PersistentStateChain(model, store, PersistSettings(unroll=int(cond[1:])))
    params = model.param_groups()[GROUP_EXPERIENTIAL] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    words = ["red", "green", "blue", "tall", "round", "quiet", "fast", "late"]
    ctx: List[str] = []
    windows: List[List[str]] = []
    targets: List[int] = []
    fact, since = rng.randrange(n_cls), 0
    t = 1.0e6
    scores = defaultdict(lambda: [0, 0])
    nll_sum = defaultdict(float)
    for step in range(steps):
        t += 60.0
        if step % period == 0:
            fact, since = rng.randrange(n_cls), 0
            it = ExperienceItem(kind=ItemKind.OBSERVATION, text=f"the code is {'abcdefgh'[fact]}", ts=t,
                                ref=f"i{step}", quantities=[Quantity("code", float(fact))])
        else:
            since += 1
            it = ExperienceItem(kind=ItemKind.EVENT, text=f"distractor {rng.choice(words)} {rng.choice(words)}",
                                ts=t, ref=f"i{step}")
        store.put_item(it)
        ctx.append(it.ref)
        w = ctx[-width:]
        # decision time: predict the current fact (prequential: before any learning on it)
        live = chain.live() if chain is not None else _null()
        with torch.no_grad(), live:
            b = model.core_forward(model.tokenize([store.items(w)]), None).b
            logp = Fn.log_softmax(head(b), -1)[0]
        lag = min(since, period - 1)
        if step >= steps // 2:                                     # score the second half (after learning started)
            scores[lag][0] += int(int(logp.argmax()) == fact)
            scores[lag][1] += 1
            nll_sum[lag] += float(-logp[fact])
        # commit, then one learning step on a replay batch (the newest decision + uniform earlier ones)
        if chain is not None:
            chain.commit(f"d{step}", context_refs=w, post_refs=w, objective_id="o", ts=t)
        windows.append(w)
        targets.append(fact)
        idx = [step] + [rng.randrange(step + 1) for _ in range(batch - 1)]
        tok = model.tokenize([store.items(windows[i]) for i in idx])
        logits = head(model.core_forward(tok, None).b)
        loss = Fn.cross_entropy(logits, torch.tensor([targets[i] for i in idx]))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    acc = {int(k): v[0] / v[1] for k, v in sorted(scores.items())}
    nll = {int(k): nll_sum[k] / scores[k][1] for k in sorted(scores)}
    beyond = [k for k in acc if k >= width]
    n_b = sum(scores[k][1] for k in beyond)
    out = {"cond": cond, "seed": seed, "acc_by_lag": acc, "nll_by_lag": nll,
           "acc_in_window": sum(scores[k][0] for k in acc if k < width) / max(1, sum(scores[k][1] for k in acc if k < width)),
           "acc_beyond_window": sum(scores[k][0] for k in beyond) / max(1, n_b),
           "nll_beyond_window": sum(nll_sum[k] for k in beyond) / max(1, n_b), "n_beyond": n_b}
    if chain is not None:
        out["rho"] = chain.rho()
        out["counters"] = dict(chain.counters)
    return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--period", type=int, default=8)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--n-cls", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--conds", default="off,K1,K2,K4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        for cond in a.conds.split(","):
            t0 = time.time()
            r = run(cond, seed, a.steps, a.period, a.width, a.n_cls, a.batch, a.lr)
            r["seconds"] = time.time() - t0
            print(json.dumps({k: r[k] for k in ("cond", "seed", "acc_in_window", "acc_beyond_window",
                                                "nll_beyond_window", "seconds")}), flush=True)
            res.append(r)
            json.dump({"args": vars(a), "results": res}, open(os.path.join(a.out, "recall.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
