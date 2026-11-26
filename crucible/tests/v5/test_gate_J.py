"""Gate J - open-ended digital creation (workstream F+G; D-15, D-18).

Measured in creation worlds whose only demand is the one population (`economy.work`): individuals delegate their
own tasks to providers they discover, judge them by their own downstream outcomes and pay from their own money.
Two scripted agents (LABELLED SCRIPTED; `innovation.creators.NoveltyCreator`, `commerce.reference.ReferenceSaaS`)
create products through the ordinary tools: per-call capability services for tasks people publicly ask about -
including a procedurally generated category that no registry documents - and subscription SaaS products for
published formats. Counterfactual arms of the same spec and seed (`innovation.experiments`): the SaaS products
removed, the generated-category service removed, and both agents shipping input-ignoring programs.

* J.1 agent_created_executable_products - products of admitted agents that executed for independent participants;
* J.2 published_products - discoverable while live (declared and deployed at the end);
* J.3 independent_discoverers - (min over the two categories) individuals who considered the product as a
  candidate for one of their own tasks after the interface check (`work.considered`), not search hits;
* J.4 independent_users - (min) individuals for whom it executed on their own input;
* J.5 measured_behavior_adoption_pvalue - Fisher exact test: share of triers who kept relying on the product,
  good programs vs input-ignoring programs (everything else equal);
* J.6 independent_transactions - (min) payments net of refunds from independent payers, weighted by the share of
  their money not traced back to the creator;
* J.7 resource_consumption_delta - (min over the two categories) minutes of human hand work saved in the
  population: the removal arm's hand-work minutes minus the factual arm's (a pre-existing resource, not the
  product's own footprint); incumbents' compute is reported too;
* J.n1 product_categories - categories (delivery model) whose products each pass J.1-J.7.
"""
import pytest

from _creation_kit import arms
from crucible.acceptance.v5.evidence import record
from crucible.innovation import measure as M

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]


def _category(p) -> str:
    return "saas_subscription" if p["host"].startswith("saasco") else "per_call_capability"


@pytest.fixture(scope="module")
def J():
    A = arms()
    f = A["factual"]
    w = f["world"]
    prods = [p for p in M.agent_products(w) if p["creator_kind"] in ("scripted", "agent")]
    o = f["outcome"]["products"]
    rows = {}
    for p in prods:
        cat = _category(p)
        rm = "no_saas" if cat == "saas_subscription" else "no_novelty"
        removed = A[rm]["outcome"]
        delta = removed["diy_minutes"] - f["outcome"]["diy_minutes"]
        g = A["garbage"]["outcome"]["products"].get(p["id"], {"triers": [], "kept": []})
        rows[p["id"]] = {"category": cat, "published": p["published"], "executable": p["executable"],
                         "discoverers": o[p["id"]]["discoverers"], "users": o[p["id"]]["users"],
                         "tx": o[p["id"]]["tx"]["n"], "tx_cents": o[p["id"]]["tx"]["cents"],
                         "payers": o[p["id"]]["tx"]["payers"], "delta_minutes": delta,
                         "triers": len(o[p["id"]]["triers"]), "kept": len(o[p["id"]]["kept"]),
                         "triers_garbage": len(g["triers"]), "kept_garbage": len(g["kept"]),
                         "creator_kind": p["creator_kind"], "incumbent_cu_delta":
                             removed["incumbent_cu"] - f["outcome"]["incumbent_cu"]}
    by_cat = {}
    for pid, r in rows.items():
        by_cat.setdefault(r["category"], []).append(r)
    best = {c: max(rs, key=lambda r: (r["users"], r["tx"])) for c, rs in by_cat.items()}
    return {"rows": rows, "best": best, "world": w}


def _note(J):
    return "LABELLED SCRIPTED creators. Per product: " + "; ".join(
        f"{pid} [{r['category']}]: discoverers {r['discoverers']}, users {r['users']}, tx {r['tx']} "
        f"({r['tx_cents']}c, {r['payers']} payers), kept {r['kept']}/{r['triers']} vs garbage "
        f"{r['kept_garbage']}/{r['triers_garbage']}, hand-work minutes saved {r['delta_minutes']:.0f}"
        for pid, r in sorted(J["rows"].items()))


def test_J1_agent_creates_executable_products(J):
    n = sum(1 for r in J["rows"].values() if r["executable"] and r["users"] > 0)
    record("J.1", {"agent_created_executable_products": n, "creators_scripted": True}, notes=_note(J))
    assert n >= 2


def test_J2_products_are_published(J):
    n = sum(1 for r in J["rows"].values() if r["published"])
    record("J.2", {"published_products": n}, notes=_note(J))
    assert n >= 2


def test_J3_independent_participants_discover_it(J):
    v = min(r["discoverers"] for r in J["best"].values())
    record("J.3", {"independent_discoverers": v, "categories": len(J["best"])}, notes=_note(J))
    assert v >= 10 and len(J["best"]) >= 2


def test_J4_independent_participants_use_it(J):
    v = min(r["users"] for r in J["best"].values())
    record("J.4", {"independent_users": v}, notes=_note(J))
    assert v >= 10


def test_J5_measured_behavior_affects_adoption(J):
    a = sum(r["kept"] for r in J["best"].values())
    b = sum(r["triers"] - r["kept"] for r in J["best"].values())
    c = sum(r["kept_garbage"] for r in J["best"].values())
    d = sum(r["triers_garbage"] - r["kept_garbage"] for r in J["best"].values())
    p = M.fisher_exact(a, b, c, d)
    per = {cat: M.fisher_exact(r["kept"], r["triers"] - r["kept"], r["kept_garbage"],
                               r["triers_garbage"] - r["kept_garbage"]) for cat, r in J["best"].items()}
    record("J.5", {"measured_behavior_adoption_pvalue": p, "kept_good": a, "dropped_good": b, "kept_garbage": c,
                   "dropped_garbage": d, **{f"pvalue_{k}": v for k, v in per.items()}}, notes=_note(J))
    assert p <= 0.05 and a / max(1, a + b) > c / max(1, c + d)


def test_J6_genuine_transactions(J):
    v = min(r["tx"] for r in J["best"].values())
    record("J.6", {"independent_transactions": v,
                   "independent_payers_min": min(r["payers"] for r in J["best"].values())}, notes=_note(J))
    assert v >= 1


def test_J7_deployment_changes_world_resource_consumption(J):
    v = min(r["delta_minutes"] for r in J["best"].values())
    record("J.7", {"resource_consumption_delta": float(v),
                   "incumbent_compute_delta_cu": float(max(r["incumbent_cu_delta"] for r in J["best"].values()))},
           notes=_note(J) + ". resource_consumption_delta = population hand-work minutes without the product minus "
                            "with it (min over the categories)")
    assert v > 0


def test_Jn1_more_than_one_product_category(J):
    ok = [c for c, r in J["best"].items() if r["executable"] and r["published"] and r["discoverers"] >= 10
          and r["users"] >= 10 and r["tx"] >= 1 and r["delta_minutes"] > 0]
    record("J.n1", {"product_categories": len(ok), "categories": ",".join(sorted(ok))}, notes=_note(J))
    assert len(ok) >= 2
