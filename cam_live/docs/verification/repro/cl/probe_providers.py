"""G-cl probe: RUDDER / COCOA on the real model's latents (debug scale, CPU) through D's own lab wiring.

  A. RUDDER provider truncation (max_len): decisions outside the window get no credit; the window's discount is
     referenced at the window's first decision while G is referenced at the trajectory's first decision.
  B. COCOA pairing: an outcome declared by an EARLIER decision and observed after decision t never enters t's
     contribution sum (Meulemans et al. Eq. 4 sums over every later reward R_{t+k}).
  C. Estimator-selection evidence is in-sample: the randomised experiments compared with the estimators are the
     same lifetimes the estimators were trained on (fraction of experiment records the learner trains on).
  D. COCOA support for a decision whose candidates were generated open-endedly ('backbone') without the
     diagnostics flag.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/cl/probe_providers.py
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys

import torch

sys.path.insert(0, "cam_live/src")
from cam_live.config import CamLiveConfig                                      # noqa: E402
from cam_live.contracts import Modality                                       # noqa: E402
from cam_live.credit.labkit import (LabRun, build_lab, door_arms, door_config, mk_action,   # noqa: E402
                                    register_experiment, run_door_experiments, weighted_policy)
from cam_live.credit.learner import LearnerSettings                           # noqa: E402
from cam_live.credit.providers import trajectory_return, COCOA, RUDDER, KEY_RTG   # noqa: E402
from cam_live.diagnostics.world import LabWorld                               # noqa: E402

WORK = os.environ.get("CL_WORK", "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/G_cl/work_prov")
FAILS, OUT = [], {}
BETA = 2e-5


def check(name, ok, detail=""):
    print(("[PASS] " if ok else "[FAIL] ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def fresh(name):
    p = os.path.join(WORK, name)
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p)
    return os.path.join(p, "store.sqlite")


def act_last(world, rng):
    return [mk_action("lab.act", channel=world.cfg.n_channels - 1, effort=3), mk_action("lab.act", channel=0, effort=1),
            mk_action("wait", Modality.INTERNAL, seconds=3600.0)]


def door_run(lab, seed, n_later):
    world = LabWorld(door_config(seed=5), lab.clock)
    register_experiment(lab.store, "door-x", ["unlock", "wait"], lab.clock.now())
    run = LabRun(lab.store, lab.linker, world, seed=seed)
    r0 = run.decide(None, arms=door_arms("door-x"))
    pol = weighted_policy(lambda a, w: 5.0 if a.arguments.get("channel") == w.cfg.n_channels - 1 else 0.5, seed=1)
    later = [run.decide(pol, candidates_fn=act_last) for _ in range(n_later)]
    return run, r0, later


def part_a_b_d():
    torch.manual_seed(0)
    lab = build_lab(fresh("ab"), CamLiveConfig.debug(), world=False, policy=False, credit=True, beta=BETA)
    try:
        lab.rudder.encoder.max_len = 3
        run, r0, later = door_run(lab, seed=0, n_later=5)
        run.mature()
        run.close()
        st = lab.store
        oid = run.task.objective_id
        up = lab.assigner.assign(run.exp_ids, reason="objective_closed")
        recs = st.trajectory(oid)
        tr = trajectory_return(st, recs, beta=BETA)
        top = [r for r in recs if not r.decision.chosen.parent_decision_id]
        have = [RUDDER in st.get(r.exp_id).credit for r in top]
        OUT["A_rudder_credit_present_per_decision"] = have
        check("A: with max_len=3 every decision of a matured 6-decision objective gets RUDDER credit", all(have),
              f"present {have}")
        # discount reference: the first window decision's return_to_go should be G re-referenced at its own time
        i0 = have.index(True) if any(have) else None
        if i0 is not None:
            e0 = top[i0].exp_id
            rtg = st.get(e0).credit[KEY_RTG]
            t0, ti = top[0].decision.ts, top[i0].decision.ts
            own = sum(e.value * math.exp(-BETA * (e.observed_at - ti)) for e in tr.events if e.observed_at >= ti)
            via_G = tr.G * math.exp(BETA * (ti - t0))
            OUT["A_first_window_decision"] = {"index": i0, "rudder_rtg": rtg, "G_referenced_at_t0": tr.G,
                                              "G_re_referenced_at_its_time": via_G, "own_return_to_go": own,
                                              "seconds_after_t0": ti - t0}
            check("A: first window decision's rudder_rtg equals G referenced at its own decision time",
                  abs(rtg - via_G) < 1e-6 * max(1, abs(via_G)),
                  f"rudder_rtg {rtg:.6f}, G at t0 {tr.G:.6f}, G at its time {via_G:.6f} ({ti - t0:.0f} s later)")
        # B. pairs that Eq. 4 would include but D's pairing excludes
        enc = lab.cocoa.encoder.encode(oid)
        pairs = set(lab.cocoa._pairs(enc, tr))
        full = {e: i for i, e in enumerate(tr.exp_ids)}
        excluded = []
        for k, ev in enumerate(tr.events):
            for t, e in enumerate(enc.exp_ids):
                if float(enc.ts[t]) < ev.observed_at and full[e] > ev.index >= 0 and (t, k) not in pairs:
                    excluded.append((t, ev.index, ev.value))
        OUT["B_events_after_t_declared_by_earlier_decisions_excluded"] = excluded
        OUT["B_pairs_used"] = sorted(pairs)
        check("B: every outcome observed after decision t enters t's COCOA sum (Eq. 4)", not excluded,
              f"{len(excluded)} (decision, declaring decision, value) pairs left out: {excluded[:6]}")
        # D. open-ended candidates without the diagnostics flag
        world = LabWorld(door_config(seed=6), lab.clock)
        run2 = LabRun(lab.store, lab.linker, world, seed=3)

        def gen(world, rng):
            acts = act_last(world, rng)
            for a in acts:
                a.provenance = "backbone"          # free-form proposals from the backbone
            return acts
        r = run2.decide(weighted_policy(lambda a, w: 1.0, seed=2), candidates_fn=gen)
        run2.decide(weighted_policy(lambda a, w: 1.0, seed=3), candidates_fn=gen)
        run2.mature(); run2.close()
        cc = lab.cocoa.credit([run2.task.objective_id])
        stt = {e: c.status for e, c in cc.items()}
        OUT["D_backbone_candidates_cocoa_status"] = stt
        OUT["D_decision_diagnostics_keys"] = sorted(lab.store.get(r.exp_id).decision.diagnostics)
        print(f"       D: decisions with backbone-generated candidates and no 'open_ended' flag -> COCOA status {stt}")
    finally:
        lab.close()


def part_c():
    torch.manual_seed(0)
    cfg = CamLiveConfig.debug()
    lab = build_lab(fresh("c"), cfg, world=False, policy=False, credit=True, beta=BETA,
                    settings=LearnerSettings.from_cfg(cfg, batch_size=32, val_fraction=0.15))
    try:
        runs = run_door_experiments(lab, 2, 12, 4, seed=0)
        lab.learner.refresh_eligibility(full=True)
        recs = [r for xid in runs for r in lab.store.experiment_records(xid)]
        val = [lab.learner.is_validation(r) for r in recs]
        frac_train = 1 - sum(val) / len(val)
        OUT["C_experiment_records"] = len(recs)
        OUT["C_fraction_trained_on"] = frac_train
        elig = [bool(lab.rudder.eligible(r)) for r in recs]
        OUT["C_fraction_rudder_eligible"] = sum(elig) / len(elig)
        check("C: randomised experiments used to validate the estimators are held out from their training",
              frac_train == 0.0, f"{frac_train:.2f} of the {len(recs)} experiment records are in the learner's training "
              f"split; compare_with_experiment uses all of them")
    finally:
        lab.close()


def main():
    os.makedirs(WORK, exist_ok=True)
    part_a_b_d()
    part_c()
    print(json.dumps(OUT, indent=1, default=str))
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)


if __name__ == "__main__":
    main()
