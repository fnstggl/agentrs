"""Industrial economy: factories staffed from the labour market, manufacturing-only revenue (SP-9.8)."""
import crucible
from crucible.domains import base
from crucible.domains import industrial_economy as I

crucible.load_all_modules()


def test_operators_follow_labour_market_employment_and_resale_is_not_manufacturing():
    ref = base.run(I.NAME, "standard_ring", 0, policy="reference", days=20, keep_env=True)
    w = ref["env"].world
    st = w.state
    org = I._orgs(w, ref["seats"][0])[0]
    emp = [p for p in st.find_ids("labor_workers", "employer", org)]
    assert emp and int((st.meta.get("own_operators") or {}).get(org, 0)) == len(emp)
    assert any(x["owner"] == org for _, x in st.scan("runs"))
    acc = I.operating_account(w, [org], 0, w.now)
    assert acc["revenue"] > 0 and acc["operating_costs"] > 0
    ref["env"].close()
    trade = base.run(I.NAME, "standard_ring", 0, policy="trade_only", days=20)
    assert trade["outcome"]["details"]["revenue"] == 0
