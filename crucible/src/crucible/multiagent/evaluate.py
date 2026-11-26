"""Evaluator-side outcomes of self-play participants (never shown to any participant).

Outcomes come from authoritative world state - the bank ledger, the related-party and circular-flow analysis of
`crucible.commerce.integrity` (which knows that identities admitted with one `controller` are one party), the
organisation registry - never from a learner's own signal or claims:

* ``cash_cents``                 money in every open account of the identity and the organisations it administers;
* ``cash_delta_cents``           that minus its endowment;
* ``independent_revenue_cents``  payments received by it or its organisations from parties unrelated to it (related
                                 parties, identities of the same controller and money that came back round are
                                 excluded), over the match;
* ``excluded_revenue_cents``     what the analysis excluded;
* ``same_controller_counted_cents``  payments from identities of its own controller that were nevertheless counted
                                 as independent (SP-12.3: must be 0);
* action counts: decisions, accepted, rejected, state-changing accepted actions, distinct tools used successfully.
"""
from __future__ import annotations

from typing import Any


def owned_orgs(world, principal: str) -> list[str]:
    st = world.state
    if "members" not in st.tables:
        return []
    return sorted({m["org"] for m in st.find("members", "person", principal) if m.get("role") in ("owner", "admin")})


def controller_of(world, principal: str) -> str | None:
    st = world.state
    if principal.startswith("o:") and "members" in st.tables:
        owners = [m["person"] for m in st.find("members", "org", principal) if m["role"] == "owner"]
        return controller_of(world, owners[0]) if owners else None
    rec = st.get("persons", principal) if "persons" in st.tables else None
    info = (rec or {}).get("info") or {}
    return info.get("controller") or None


def revenue(world, owner: str, window_s: int | None = None) -> dict:
    from ..commerce.integrity import related_flows
    rf = related_flows(world, owner, window_s=window_s)
    ctrl = controller_of(world, owner)
    same = 0
    for payer, rec in rf["payers"].items():
        if ctrl and controller_of(world, payer) == ctrl:
            same += int(rec["paid"]) - int(rec["circular"])
    return {"independent": int(rf["independent_cents"]), "excluded": int(rf["excluded_cents"]),
            "same_controller_counted": int(same), "payers": len(rf["payers"]), "flagged": len(rf["flagged"])}


def outcome(world, principal: str, endowment: int, t0: int, history: list[dict] | None = None) -> dict:
    from ..goals.metrics import m_cash
    orgs = owned_orgs(world, principal)
    window = max(1, int(world.now - t0) + 1)
    cash = int(m_cash(world, principal) + sum(m_cash(world, o) for o in orgs))
    ind = exc = same = 0
    for who in [principal] + orgs:
        r = revenue(world, who, window)
        ind += r["independent"]
        exc += r["excluded"]
        same += r["same_controller_counted"]
    h = history or []
    acts = [x for x in h if x.get("tool")]
    return {"cash_cents": cash, "cash_delta_cents": cash - int(endowment), "independent_revenue_cents": int(ind),
            "excluded_revenue_cents": int(exc), "same_controller_counted_cents": int(same), "orgs": len(orgs),
            "decisions": len(h), "actions": len(acts), "accepted": sum(1 for x in acts if x.get("ok")),
            "rejected": sum(1 for x in acts if x.get("ok") is False),
            "state_changing_accepted": sum(1 for x in acts if x.get("ok") and not x.get("read_only")),
            "tools_used": len({x["tool"] for x in acts if x.get("ok")})}


def match_outcomes(match) -> dict[str, dict]:
    return {p: outcome(match.world, p, match.endowments[p], match.t0, getattr(a, "history", None))
            for p, a in match.agents.items()}
