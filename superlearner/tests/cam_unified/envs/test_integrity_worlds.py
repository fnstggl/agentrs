"""Exp F diagnostic worlds: genuine objective verified once per unique transaction; traps observable but useless."""
import os
import sys

import pytest

CRUCIBLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../crucible/src"))
if CRUCIBLE not in sys.path:
    sys.path.insert(0, CRUCIBLE)

from superlearner.cam_unified.envs import experience as X  # noqa: E402
from superlearner.cam_unified.envs.integrity import VARIANTS, IntegrityWorld  # noqa: E402
from superlearner.cam_unified.interfaces import Action  # noqa: E402


def _run(op, ad, policy, steps=96):
    for k in range(steps):
        ad.execute([policy(k)])
        ad.advance_time([1 / 24])
        if op.resolved(0):
            break
    return op.release_labels()[0]


@pytest.mark.parametrize("variant", VARIANTS)
def test_trap_does_not_raise_verified_utility_and_genuine_policy_does(variant):
    op = X.ExperienceOperator([X.WorldSpec(-1, f"integrity:{variant}", 0)])
    ad = op.adapter()
    ad.reset()
    trap_tool = [t.name for t in ad.available_tools()[0] if not t.name.startswith("shop.")]
    trap = _run(op, ad, lambda k: Action(tool=trap_tool[0], args={"level": 5.0} if "gauge" in trap_tool[0] else {})
                if trap_tool else Action(tool="shop.status"))
    op2 = X.ExperienceOperator([X.WorldSpec(-1, f"integrity:{variant}", 0)])
    ad2 = op2.adapter()
    ad2.reset()
    good = _run(op2, ad2, lambda k: Action(tool="shop.set_price", args={"price": 300}) if k == 0 else
                Action(tool="shop.produce", args={"units": 1}))
    assert trap.status in ("success", "failure") and good.status in ("success", "failure")
    assert good.value > trap.value
    assert good.utility > 0.2 and trap.utility <= 0.05


def test_duplicate_sale_events_never_count_twice():
    w = IntegrityWorld("duplicate_outcomes", 1)
    w.execute("p:agent", {"tool": "shop.produce", "args": {"units": 10}})
    for _ in range(40):
        w.advance(3600)
    sales = [e for e in w.events if e["type"] == "shop.sale"]
    txs = [e["payload"]["tx"] for e in sales]
    assert len(txs) > len(set(txs))                      # duplicates were emitted
    v = w.verify()
    unique = sum(v for k, v in w.sales.items())
    assert v["outcome"]["value"] == unique and v["trap"]["duplicates_emitted"] > 0
