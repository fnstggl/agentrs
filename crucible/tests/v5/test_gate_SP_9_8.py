"""SP-9.8 (MASTER_SPEC Part 9.8): Tier 2 domain Industrial economy. Workstream I-2 (dom2).

Every required mechanic (Raw materials, Production, Energy, Logistics, Inventory, Equipment, Labor, Market demand,
Resource scarcity, Supply disruptions) is exercised by recorded events in integrated reference runs reached through
crucible.experience.make(2, ...) in the default observation_only mode (`base.section_metrics`). Intervention
controls on the same instance: a factory that never hires operators from the labour market starts no production run
(Labor is causal for Production); buying and reselling components earns no manufacturing revenue.
"""
import pytest

from crucible.acceptance.v5 import evidence
from crucible.domains import base
from crucible.domains import industrial_economy as I

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_9_8.py"
CONFIG, SEED = "standard_ring", 0


def _own_runs(r):
    w = r["env"].world
    orgs = I._orgs(w, r["seats"][0])
    n = sum(1 for _, x in w.state.scan("runs") if x["owner"] in orgs)
    r["env"].close()
    return n


@pytest.mark.slow
@pytest.mark.timeout(4 * 3600)
def test_SP_9_8_industrial_economy():
    m, runs = base.section_metrics(I.NAME)
    ref = base.run(I.NAME, CONFIG, SEED, policy="reference", keep_env=True)
    no_ops = base.run(I.NAME, CONFIG, SEED, policy="no_operators", keep_env=True)
    trade = base.cached_run(I.NAME, CONFIG, SEED, "trade_only")
    m["reference_production_runs"] = _own_runs(ref)
    m["no_operator_production_runs"] = _own_runs(no_ops)
    m["trade_only_manufacturing_revenue"] = int(trade["outcome"]["details"]["revenue"])
    m["reference_profit_cents"] = int(ref["outcome"]["details"]["profit"])
    evidence.record("SP-9.8", m, command=CMD, labels={"scripted": True},
                    notes=f"reference profit {m['reference_profit_cents']} with {m['reference_production_runs']} runs; "
                          f"no operators {m['no_operator_production_runs']} runs; trade-only manufacturing revenue "
                          f"{m['trade_only_manufacturing_revenue']}")
    assert m["required_mechanics_listed"] == 10 and m["required_mechanics_exercised"] == 10, m
    assert m["required_mechanics_missing"] == 0, m
    assert m["observation_only_default"] and m["reachable_through_experience"], m
    assert m["no_operator_production_runs"] == 0 < m["reference_production_runs"], m
    assert m["trade_only_manufacturing_revenue"] == 0, m
