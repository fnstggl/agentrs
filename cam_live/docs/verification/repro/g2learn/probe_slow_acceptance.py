"""G2-learn round 2 probe: how often does the fail-closed, per-provider slow-step guard (DX, fix of G2L-4 / G2L-5)
let adapter updates through — does it freeze slow learning? (CL-03, CL-08, NA-09)

    # integrated agent (LiveRuntime + CamLiveAgent + LabWorld + signing verifier), tiny backbone, CPU
    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_slow_acceptance.py --mode agent --seeds 11 12 13 14 15 16
    # the same agent on the REAL configuration (Qwen3-4B + 50m core, CUDA, queued GPU lease, < 14 min of lock)
    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_slow_acceptance.py --mode agent --real --seeds 11 12 13
    # D's lab (labkit) with K rounds of (frequent, frequent, frequent, slow), debug scale (CPU) or --real (CUDA)
    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_slow_acceptance.py --mode lab [--real] --rounds 10

Reads the learner's own log (ContinualLearner log) and reports, in order, every slow step: accepted / reverted /
deferred / skipped, the binding provider, chunks per provider, and the reasons; the net adapter change from the start
(is anything left after reverts?), and the acceptance rate in the first and second half of the run.
The agent mode uses the agent's defaults except: adapter_every 3 (so a test-length run has several slow steps),
use_backbone False (speed; the schema walker and grounded candidates still propose), durable store off.
"""
import argparse
import collections
import contextlib
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import TESTS, WORK, check, finish, info, queued_gpu_lock  # noqa: E402

import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["agent", "lab"], required=True)
ap.add_argument("--real", action="store_true")
ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
ap.add_argument("--rounds", type=int, default=10)
ap.add_argument("--max-decisions", type=int, default=22)
ap.add_argument("--lock-budget-s", type=float, default=780.0)
ap.add_argument("--tag", default="")
a = ap.parse_args()
tag = a.tag or f"{a.mode}{'_real' if a.real else ''}"
root = os.path.join(WORK, "slow_acceptance", tag)
shutil.rmtree(root, ignore_errors=True)
os.makedirs(root)
torch.set_num_threads(4)


def summarise(log_path, adapters_now, adapters_start):
    rows = []
    for line in open(log_path):
        d = json.loads(line)
        d = d.get("data", d)
        if d.get("timescale") != "slow_adapters":
            continue
        n = d.get("notes") or "{}"
        n = json.loads(n) if isinstance(n, str) else n
        g = n.get("guard") or {}
        if n.get("deferred"):
            st = "deferred"
        elif n.get("skipped"):
            st = "skipped"
        elif g.get("reverted"):
            st = "reverted"
        elif g.get("accepted"):
            st = "accepted"
        else:
            st = "applied_without_verdict" if (d.get("weight_delta") or {}).get("semantic_adapters", 0) > 0 else "other"
        bp = g.get("by_provider") or {}
        rows.append({"status": st, "binding": g.get("binding_provider"),
                     "chunks": {k: v.get("n_chunks") for k, v in bp.items()},
                     "reasons": (g.get("reasons") or [n.get("skipped") or ""])[:3],
                     "excluded": n.get("excluded_without_held_out"),
                     "delta": (d.get("weight_delta") or {}).get("semantic_adapters", 0.0)})
    c = collections.Counter(r["status"] for r in rows)
    info(f"{tag}: slow steps in order", [r["status"][0].upper() for r in rows])
    info(f"{tag}: slow steps by outcome", dict(c))
    h = len(rows) // 2
    for name, part in (("first half", rows[:h]), ("second half", rows[h:])):
        taken = [r for r in part if r["status"] in ("accepted", "reverted")]
        info(f"{tag}: {name}: accepted / judged", (sum(r["status"] == "accepted" for r in taken), len(taken)))
    bind = collections.Counter(r["binding"] for r in rows if r["status"] == "reverted")
    info(f"{tag}: binding provider of reverted steps", dict(bind))
    one_chunk = sum(1 for r in rows if r["status"] == "reverted" and any(v == 1 for v in r["chunks"].values()))
    info(f"{tag}: reverted steps where some judged provider had a single validation chunk (float-floor tolerance)",
         (one_chunk, c.get("reverted", 0)))
    rs = collections.Counter(x.split(":")[0] + ":" + x.split(":", 1)[1].split("(")[0][:40] if ":" in x else x[:50]
                             for r in rows if r["status"] == "reverted" for x in r["reasons"])
    info(f"{tag}: reasons of reverted steps (provider: kind)", dict(rs.most_common(8)))
    info(f"{tag}: chunks per provider at the last judged step", next((r["chunks"] for r in reversed(rows) if r["chunks"]), {}))
    net = sum(float((x - y).double().pow(2).sum()) for x, y in zip(adapters_now, adapters_start)) ** 0.5
    info(f"{tag}: net adapter change from the start (L2)", net)
    judged = c.get("accepted", 0) + c.get("reverted", 0)
    return rows, c, judged, net


if a.mode == "agent":
    sys.path.insert(0, os.path.join(TESTS, "integration"))
    from int_helpers import Recorder, agent_config, run_lifetime  # noqa: E402
    from cam_live.agent import CamLiveAgent  # noqa: E402
    from cam_live.config import CamLiveConfig  # noqa: E402
    from cam_live.contracts import SimClock  # noqa: E402
    if a.real:
        cfg = CamLiveConfig(device="cuda", seed=0)
        cfg.optim.adapter_every = 3
        cfg.sections["agent"] = dict(use_backbone=False, durable=False, planner=False)
    else:
        cfg = agent_config(policy_update_every=6)
    lock = queued_gpu_lock(16.0) if a.real else contextlib.nullcontext(time.time())
    with lock as t_lock:
        clock = SimClock()
        agent = CamLiveAgent(cfg, os.path.join(root, "agent"), clock=clock)
        start = [p.detach().clone().cpu() for p in agent.model.param_groups()["semantic_adapters"]]
        rec = Recorder(agent)
        lab = {"outcome_horizon_s": 1800.0, "mean_delay_s": (300.0, 900.0)}
        done = []
        for s in a.seeds:
            if a.real and time.time() - t_lock > a.lock_budget_s:
                break
            lt = run_lifetime(rec, clock, root, s, max_steps=40, max_decisions=a.max_decisions, lab_kw=lab)
            done.append((s, lt.status))
        info(f"{tag}: lifetimes run (seed, status)", done)
        info(f"{tag}: records / learner counters", (agent.store.count(), dict(agent.learner.counters)))
        info(f"{tag}: held-out losses available at the end (learner.validate)",
             {k: round(v, 4) for k, v in agent.learner.validate().items()})
        now = [p.detach().clone().cpu() for p in agent.model.param_groups()["semantic_adapters"]]
        log = os.path.join(agent.dir, "learner_updates.jsonl")
        info(f"{tag}: promotions (promoted, reasons)", [(p.get("promoted"), (p.get("reasons") or [])[:1]) for p in agent.promotions])
        if a.real:
            info(f"{tag}: GPU lock held (s)", round(time.time() - t_lock, 1))
        rows, c, judged, net = summarise(log, now, start)
        agent.close()
else:
    from cam_live.config import CamLiveConfig  # noqa: E402
    from cam_live.credit.labkit import build_lab, collect_family  # noqa: E402
    from cam_live.credit.learner import LearnerSettings  # noqa: E402
    from cam_live.diagnostics.world import LabConfig  # noqa: E402
    from cam_live.learning_api import TIMESCALE_FREQUENT, TIMESCALE_SLOW, UpdatePlan  # noqa: E402
    cfg = CamLiveConfig(device="cuda") if a.real else CamLiveConfig.debug()
    if a.real:
        cfg.core.max_context = 64
    lock = queued_gpu_lock(16.0) if a.real else contextlib.nullcontext(time.time())
    with lock as t_lock:
        torch.manual_seed(0)
        # the agent's split settings (record split, every 1/0.15-th record held out), D's lab otherwise
        st = LearnerSettings.from_cfg(cfg, val_split="record", holdout_experiments=False, batch_size=8,
                                      slow_batch_size=4, grad_last=8)
        lab = build_lab(os.path.join(root, "store.sqlite"), cfg, bridge=True, beta=2e-6, settings=st,
                        log_path=os.path.join(root, "learner_updates.jsonl")) if "log_path" in build_lab.__code__.co_varnames \
            else build_lab(os.path.join(root, "store.sqlite"), cfg, bridge=True, beta=2e-6, settings=st)
        L, m = lab.learner, lab.model
        if not getattr(L, "log_path", None):
            L.log_path = os.path.join(root, "learner_updates.jsonl")
        collect_family(lab.store, lab.linker, LabConfig(n_channels=4, family="lab/A", target=30), 8, 6, seed=3,
                       clock=lab.clock)
        L.refresh_eligibility(full=True)
        start = [p.detach().clone().cpu() for p in m.param_groups()["semantic_adapters"]]
        for k in range(a.rounds):
            if a.real and time.time() - t_lock > a.lock_budget_s:
                break
            for _ in range(3):
                L.learn(UpdatePlan(trigger="probe", timescales=(TIMESCALE_FREQUENT,)))
            L.learn(UpdatePlan(trigger="probe", timescales=(TIMESCALE_SLOW,)))
        now = [p.detach().clone().cpu() for p in m.param_groups()["semantic_adapters"]]
        info(f"{tag}: records / learner counters", (lab.store.count(), dict(L.counters)))
        info(f"{tag}: held-out losses (learner.validate)", {k: round(v, 4) for k, v in L.validate().items()})
        if a.real:
            info(f"{tag}: GPU lock held (s)", round(time.time() - t_lock, 1))
        rows, c, judged, net = summarise(L.log_path, now, start)
        lab.close()

acc = c.get("accepted", 0)
check(f"{tag}: no slow step is applied without a held-out verdict", c.get("applied_without_verdict", 0) == 0,
      f"{dict(c)}")
check(f"{tag}: adapter learning is not frozen: at least a quarter of the judged slow steps are accepted",
      judged > 0 and acc >= 0.25 * judged, f"accepted {acc} of {judged} judged ({c.get('deferred', 0)} deferred, "
      f"{c.get('skipped', 0)} skipped); net adapter change {net:.4g}")
sys.exit(finish())
