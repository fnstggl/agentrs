"""Red-team harness module (registered only when the red team imports this file, never by `load_all_modules`).

`rt_cohort` (host `cohort`) declares a small column table and lets a participant record a cohort tally through the
substrate's real batch path (`crucible.substrate.columns.batch`: content-addressed npz blob + one `cols.batch`
event + the verifying applier). It exists so the batch-blob tampering probe can attack the substrate's verification
before the population workstream's columns are merged; the probe then tampers with the stored blob (store-level
access, which no participant has) and checks that replay never applies tampered values.
"""
from __future__ import annotations

import numpy as np

from ...core import schema as S
from ...core.kernel import Module, endpoint, register_module

ROWS = 64


@register_module
class RtCohort(Module):
    name = "rt_cohort"
    host = "cohort"
    version = "1"
    depends = ("cols",)

    def install(self, world):
        from ...substrate import columns
        columns.declare(world, "rtc", ROWS, {"tally": ("int64", (), 0), "score": ("float32", (), 0.0)})

    @endpoint("tally", S.obj({"rows": S.s_arr(S.s_int(minimum=0, maximum=ROWS - 1), maxItems=ROWS, minItems=1),
                              "amount": S.s_int(minimum=1, maximum=1000)}, ["rows", "amount"]), cost_s=1)
    def ep_tally(self, ctx, rows, amount):
        """Add `amount` to the tally of each row (one recorded cohort batch)."""
        from ...substrate import columns
        idx = np.asarray(rows, dtype=np.int64)
        ref = columns.batch(ctx, "rtc", "rt.tally", idx, {"tally": np.full(idx.size, int(amount), np.int64),
                                                           "score": np.full(idx.size, 0.5, np.float32)}, mode="add")
        return {"ref": ref}
