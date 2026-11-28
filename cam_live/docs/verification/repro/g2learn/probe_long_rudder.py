"""G2-learn probe: RUDDER credit on trajectories longer than the decomposer's window (fix of G-cl m3), RC-02 / RC-03.

    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_long_rudder.py

D's own lab (debug-scale CamLiveModel, LabWorld door mechanism, duration discount beta = 2e-5): a matured objective of
11 decisions with ``max_len`` = 3, the decomposer head given random NON-zero weights (its zero initialisation would
make every contribution trivially 0 except the offset). Checks, computed here from the stored credit and an own
discount Gamma_t = exp(-beta (t_t - t_0)):
  * every top-level decision gets RUDDER credit (also those outside every training window);
  * conservation sum_t Gamma_t c_t = G over the WHOLE trajectory (c_t = stored ``rudder_contribution``);
  * return-to-go at the first decision equals G; stored ``rudder_rtg`` is finite everywhere.
"""
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import WORK, check, finish, info  # noqa: E402

import torch  # noqa: E402

from cam_live.config import CamLiveConfig  # noqa: E402
from cam_live.contracts import Modality  # noqa: E402
from cam_live.credit.labkit import (LabRun, build_lab, door_arms, door_config, mk_action,  # noqa: E402
                                    register_experiment, weighted_policy)
from cam_live.credit.providers import KEY_CONTRIB, KEY_RTG, RUDDER, trajectory_return  # noqa: E402
from cam_live.diagnostics.world import LabWorld  # noqa: E402

BETA = 2e-5
torch.set_num_threads(1)
torch.manual_seed(0)
d = os.path.join(WORK, "long_rudder")
shutil.rmtree(d, ignore_errors=True)
os.makedirs(d)
lab = build_lab(os.path.join(d, "store.sqlite"), CamLiveConfig.debug(), world=False, policy=False, credit=True, beta=BETA)
try:
    lab.rudder.encoder.max_len = 3
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in lab.rudder.decomposer.parameters():
            if p.dtype.is_floating_point:
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    world = LabWorld(door_config(seed=5), lab.clock)
    register_experiment(lab.store, "door-long", ["unlock", "wait"], lab.clock.now())
    run = LabRun(lab.store, lab.linker, world, seed=0)
    run.decide(None, arms=door_arms("door-long"))

    def cands(w, rng):
        return [mk_action("lab.act", channel=w.cfg.n_channels - 1, effort=3), mk_action("lab.act", channel=0, effort=1),
                mk_action("wait", Modality.INTERNAL, seconds=3600.0)]
    pol = weighted_policy(lambda a, w: 5.0 if a.arguments.get("channel") == w.cfg.n_channels - 1 else 0.5, seed=1)
    for _ in range(10):
        run.decide(pol, candidates_fn=cands)
    run.mature()
    run.close()
    st = lab.store
    oid = run.task.objective_id
    lab.assigner.assign(run.exp_ids, reason="objective_closed")
    recs = st.trajectory(oid)
    top = [r for r in recs if not r.decision.chosen.parent_decision_id]
    tr = trajectory_return(st, recs, beta=BETA)
    info("decisions / matured / G", (len(top), tr.matured, tr.G))
    cr = [st.get(r.exp_id).credit for r in top]
    have = [RUDDER in c and KEY_CONTRIB in c for c in cr]
    check("every top-level decision of an 11-decision trajectory gets RUDDER credit with max_len = 3", all(have),
          f"present {have}")
    t0 = top[0].decision.ts
    gam = [math.exp(-BETA * (r.decision.ts - t0)) for r in top]
    s = sum(gm * c.get(KEY_CONTRIB, float("nan")) for gm, c in zip(gam, cr))
    check("conservation over the whole trajectory: sum_t Gamma_t c_t = G (|err| <= 1e-6 max(1, |G|))",
          abs(s - tr.G) <= 1e-6 * max(1.0, abs(tr.G)), f"sum {s:.9f}, G {tr.G:.9f}, span {top[-1].decision.ts - t0:.0f} s")
    contrib = [c.get(KEY_CONTRIB) for c in cr]
    info("contributions (first 3 inside / outside the newest window)", [round(x, 4) for x in contrib])
    check("the random decomposer gives non-trivial contributions (the conservation check is not vacuous)",
          sum(1 for x in contrib[1:] if abs(x) > 1e-6) >= 3, f"{sum(1 for x in contrib[1:] if abs(x) > 1e-6)} non-zero")
    check("return-to-go at the first decision equals G", abs(cr[0][KEY_RTG] - tr.G) <= 1e-6 * max(1.0, abs(tr.G)),
          f"rtg {cr[0][KEY_RTG]:.9f}, G {tr.G:.9f}")
    check("stored rudder_rtg is finite for every decision", all(math.isfinite(c.get(KEY_RTG, float("nan"))) for c in cr))
finally:
    lab.close()
sys.exit(finish())
