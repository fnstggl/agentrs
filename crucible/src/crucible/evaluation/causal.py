"""Operator-side causal evaluation by matched counterfactual worlds.

`installation_effect(world, install_id, days)` forks the world twice at the current moment; in one fork the
installation is removed (triggers deleted, grants revoked), in the other it keeps running; both advance `days`
with no agents acting, and the installer's operating results are compared day by day. Because forks share the
past exactly and the cost model is deterministic, the difference is the effect of keeping the capability over
that window, for that firm, from that state - the ground truth that in-world trials (switchback, washout,
sequential) try to estimate from one world. It is a research instrument: simulated firms cannot fork the world.

Assumptions and limits: the effect is conditional on the state at the fork; agents' future behaviour is held out
(no agents act); stochastic mechanisms are seeded identically in both forks, so differences are causal but
chaotic divergence (e.g. customers switching) is part of the effect.
"""
from __future__ import annotations

import numpy as np

DAY = 86400


def installation_effect(world, install_id: str, days: int = 14) -> dict:
    from ..economy.operations import operating_result
    st = world.state
    inst = st.tables["cap_installs"][install_id]
    org = inst["installer"]
    keep = world.fork(f"{world.world_id}-keep-{install_id}")
    drop = world.fork(f"{world.world_id}-drop-{install_id}")
    caps = drop.modules["caps"]
    drop._run_system("counterfactual: remove installation",
                     lambda ctx: caps.uninstall(ctx, install_id, "counterfactual removal"))
    if keep.state.tables["cap_installs"][install_id]["status"] in ("trial", "evaluated", "paused"):
        kc = keep.modules["caps"]

        def adopt(ctx):
            ctx.emit("caps.install_status", {"id": install_id, "status": "active", "reason": "counterfactual keep"},
                     vis=(org,))
            kc.set_on(ctx, install_id, True)
        keep._run_system("counterfactual: keep installation running", adopt)
    d0 = world.now // DAY + 1
    keep.advance_time((d0 + days) * DAY)
    drop.advance_time((d0 + days) * DAY)
    k = [operating_result(keep.modules["ops"].day(keep.state, org, d), exclude_inventory=True) for d in range(d0, d0 + days)]
    r = [operating_result(drop.modules["ops"].day(drop.state, org, d), exclude_inventory=True) for d in range(d0, d0 + days)]
    diff = np.array(k, float) - np.array(r, float)
    return {"install": install_id, "firm": org, "days": days, "effect_per_day": round(float(diff.mean()), 2),
            "se_over_days": round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 2) if len(diff) > 1 else None,
            "keep_total": int(sum(k)), "drop_total": int(sum(r)), "daily_difference": diff.round(1).tolist()}
