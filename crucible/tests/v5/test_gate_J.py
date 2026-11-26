"""Gate J - open-ended digital creation (workstream G).

One world (`crucible.innovation.scenarios.market`, seed 11): a scripted reference agent writes, deploys and declares
a route.shortest service; 24 need-driven participants (crucible.innovation.adopt: rule-based, decisions only from
their own test scores, failures and prices) discover it, test it on inputs they generate themselves, adopt it and
use it; five scripted reference developers then publish two packages that an independent developer's code executes.
Counterfactual arms are forks of one snapshot taken before the service exists: the product as built, the same
product built from a weaker implementation, and no product.

Provenance: creators are scripted reference policies (feasibility, not learning); participants are the workstream-G
adopters until the population of workstream B exists. Every number is read from recorded world state.
"""
import pytest

import crucible

crucible.load_all_modules()
from crucible.acceptance.v5 import evidence  # noqa: E402
from crucible.innovation import measure as M  # noqa: E402
from crucible.innovation import scenarios as SC  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]
PROVENANCE = ("scripted reference creators (ana: service; dana..deve: packages) acting through tools; 24 rule-based "
              "need-driven participants (crucible.innovation.adopt); world crucible.innovation.scenarios.market seed 11")


@pytest.fixture(scope="module")
def market():
    r = SC.market(seed=11, participants=24, days=4.0, contrast=True, software=True)
    yield r
    SC.close(r)


def _products(w):
    return M.agent_products(w)


def _best(w, fn):
    vals = {p["id"]: len(fn(w, p)) for p in _products(w)}
    return max(vals.values(), default=0), vals


def test_J1_agent_creates_executable_products(market):
    w = market["world"]
    prods = _products(w)
    ex = [p for p in prods if p["executable"]]
    m = {"agent_created_executable_products": len(ex), "agent_created_products": len(prods),
         "service_ok_requests": sum(p.get("ok_requests", 0) for p in ex if p["kind"] == "service"),
         "packages_executed_by_dependents": sum(1 for p in ex if p["kind"] == "package"
                                                 and M.adopters(w, p))}
    evidence.record("J.1", m, notes=f"{PROVENANCE}. Products: {[p['id'] for p in ex]}. A service counts as executable "
                                    "when it answered requests with status 200 after it was declared; a package when "
                                    "its release imported in the sandbox or its code ran inside a dependent.")
    assert m["agent_created_executable_products"] >= 2


def test_J2_agent_publishes_products(market):
    w = market["world"]
    pub = [p for p in _products(w) if p["published"]]
    m = {"published_products": len(pub),
         "public_declarations": sum(1 for p in pub if p["kind"] == "service"),
         "public_releases": sum(p.get("releases", 0) for p in pub if p["kind"] == "package")}
    evidence.record("J.2", m, notes=f"{PROVENANCE}. Published = a public caps.declared event (indexed for search) or a "
                                    "public pkg.released event.")
    assert m["published_products"] >= 2


def test_J3_independent_participants_discover_it(market):
    w = market["world"]
    best, per = _best(w, M.discoverers)
    m = {"independent_discoverers": best, **{f"independent_discoverers[{k}]": v for k, v in sorted(per.items())}}
    evidence.record("J.3", m, notes=f"{PROVENANCE}. Discovery = a caps.search result (caps.seen), a learned host "
                                    "(search, forum, describe) or pkg.search/info, by principals unrelated to the "
                                    "creator (goals.metrics.related_parties).")
    assert m["independent_discoverers"] >= 10


def test_J4_independent_participants_test_or_use_it(market):
    w = market["world"]
    best, per = _best(w, M.users)
    m = {"independent_users": best, **{f"independent_users[{k}]": v for k, v in sorted(per.items())}}
    evidence.record("J.4", m, notes=f"{PROVENANCE}. A user is an unrelated principal for whom the product actually "
                                    "executed: caps.try trials, counted use after adoption, paid calls, or a dependent "
                                    "repository whose own sandboxed code ran the package.")
    assert m["independent_users"] >= 10


def test_J5_measured_behaviour_affects_adoption(market):
    arms = market["arms"]["arms"]
    n = arms["product"]["participants"]
    a, b = arms["product"]["adopters"], arms["degraded"]["adopters"]
    p = M.fisher_exact(a, n - a, b, n - b)
    m = {"measured_behavior_adoption_pvalue": p, "adopters_as_built": a, "adopters_weaker_implementation": b,
         "participants": n, "satisfied_requests_as_built": arms["product"]["satisfied_requests"],
         "satisfied_requests_weaker": arms["degraded"]["satisfied_requests"]}
    evidence.record("J.5", m, notes=f"{PROVENANCE}. Two forks of one snapshot taken before the service existed; the "
                                    "only difference is the code behind the same declaration and price (Dijkstra vs "
                                    "the hop-count reference tier). Participants decide from scores of their own "
                                    "generated tests. Two-sided Fisher exact test on adopters / non-adopters.")
    assert m["measured_behavior_adoption_pvalue"] <= 0.05 and a > b


def test_J6_genuine_transactions(market):
    w = market["world"]
    rows = {p["id"]: M.independent_transactions(w, p) for p in _products(w) if p["kind"] == "service"}
    best = max(rows.values(), key=lambda r: r["n"])
    m = {"independent_transactions": best["n"], "independent_revenue_cents": best["cents"],
         "independent_payers": best["payers"]}
    evidence.record("J.6", m, notes=f"{PROVENANCE}. Ledger `sale` transactions referencing the service, paid from the "
                                    "own bank accounts of principals unrelated to the creator (system accounts and "
                                    "related parties excluded). Per product: " + str(rows))
    assert m["independent_transactions"] >= 1


def test_J7_deployment_changes_world_resource_consumption(market):
    arms = market["arms"]["arms"]
    with_p, without = arms["product"]["compute"], arms["absent"]["compute"]
    m = {"resource_consumption_delta": round(with_p["cu"] - without["cu"], 3),
         "compute_units_with_product": with_p["cu"], "compute_units_without_product": without["cu"],
         "compute_rent_delta_cents": with_p["compute_rent_cents"] - without["compute_rent_cents"],
         "cpu_ms_delta": round(with_p["cpu_ms"] - without["cpu_ms"], 3)}
    evidence.record("J.7", m, notes=f"{PROVENANCE}. Forks of one snapshot before creation: with the product vs the "
                                    "agent not creating it, same horizon. Compute units metered by the cloud on every "
                                    "service instance (svc_days) and compute rent paid to providers (ledger kind "
                                    "compute).")
    assert m["resource_consumption_delta"] > 0


def test_Jn1_more_than_one_product_category(market):
    w = market["world"]
    cats = sorted({p["category"] for p in _products(w) if p["executable"] and p["published"]})
    m = {"product_categories": len(cats), "categories": ",".join(cats)}
    evidence.record("J.n1", m, notes=f"{PROVENANCE}. Categories of executable, published agent products: an API "
                                     "service declared in the capability registry and a software package adopted by "
                                     "execution.")
    assert m["product_categories"] >= 2
