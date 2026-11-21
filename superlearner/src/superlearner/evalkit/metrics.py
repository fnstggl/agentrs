"""Evaluator-side metrics for competency tasks. PROTECTED: candidate learner code never edits or imports this.

Each metric reads authoritative world state only (never agent claims) and is registered in Crucible's metric
registry, so task contracts can use it in predicates. A metric measures an *outcome*; none of them looks at which
tools were called.

    sl_companies(principal)          active organisations the principal owns that are registered companies
    sl_org_cash(principal)           the largest total balance held by one organisation the principal owns
    sl_solver_quality(principal, need, n)
                                     EVALUATOR-ONLY. Best measured quality of a `solve` endpoint in a repository of
                                     the principal, executed in the sandbox on seeded hidden inputs of the need
    sl_live_services(principal)      deployments of the principal's organisations that are live on a running instance
    sl_listed_services(principal)    active API listings of the principal's organisations that point at such a
                                     deployment
    sl_api_calls(principal)          paid calls the principal (or its organisations) made to unrelated services
    sl_ladder(principal)             fraction of five verified build milestones (used only as an optional shaping
                                     potential; success is always decided by the task's own predicate)
"""
from __future__ import annotations

import json

import numpy as np

from .. import ensure_crucible

ensure_crucible()
from crucible.goals.metrics import metric, related_parties  # noqa: E402

_QCACHE: dict = {}


def owned_orgs(world, principal: str) -> list[str]:
    st = world.state
    out = []
    for m in st.find("members", "person", principal):
        if m["role"] != "owner":
            continue
        o = st.get("orgs", m["org"])
        if o and o["status"] == "active":
            out.append(m["org"])
    return sorted(out)


@metric("sl_orgs", "Active organisations the principal owns.")
def m_orgs(world, principal: str) -> float:
    return float(len(owned_orgs(world, principal)))


@metric("sl_companies", "Active organisations the principal owns that are registered companies.")
def m_companies(world, principal: str) -> float:
    st = world.state
    if "companies" not in st.tables:
        return 0.0
    return float(sum(1 for o in owned_orgs(world, principal)
                     if (st.get("companies", o) or {}).get("status") == "registered"))


@metric("sl_org_cash", "Largest total open-account balance (cents) of one organisation the principal owns.")
def m_org_cash(world, principal: str) -> float:
    st = world.state
    best = 0
    for o in owned_orgs(world, principal):
        tot = sum(st.tables["accounts"][a]["balance"] for a in st.find_ids("accounts", "owner", o)
                  if st.tables["accounts"][a]["status"] == "open")
        best = max(best, tot)
    return float(best)


def _repos_of(world, principal: str) -> list[str]:
    st = world.state
    if "repos" not in st.tables:
        return []
    owners = set(owned_orgs(world, principal)) | {principal}
    return sorted(r for r, rec in st.scan("repos") if rec["owner"] in owners)


@metric("sl_repo_commits", "Largest number of commits in a repository of the principal or its organisations.")
def m_repo_commits(world, principal: str) -> float:
    st = world.state
    return float(max([st.tables["repos"][r]["commits"] for r in _repos_of(world, principal)], default=0))


def solver_quality_of_files(files: dict, need: str, n: int = 3) -> float:
    """Mean verifier score of the files' `solve` endpoint on n seeded inputs, executed in the Crucible sandbox."""
    from crucible.economy.needs import NEEDS
    from crucible.sandbox.runner import run_program
    try:
        man = json.loads(files.get("crucible.json") or "{}")
        ep = (man.get("endpoints") or {}).get("solve") or {}
        target = f"{man.get('entry', 'service.py')}:{ep.get('handler', 'solve')}"
    except (ValueError, AttributeError):
        return 0.0
    if not isinstance(man, dict) or "solve" not in (man.get("endpoints") or {}):
        return 0.0
    nd = NEEDS[need]
    qs = []
    for k in range(n):
        inp, hidden = nd.generate(np.random.default_rng([424242, k]), 0.6)
        try:
            r = run_program(dict(files), target, inp, None, timeout=4.0)
            qs.append(float(nd.score(inp, hidden, r.get("result"))) if r.get("ok") else 0.0)
        except Exception:  # noqa: BLE001 - broken code scores zero
            qs.append(0.0)
        if qs[-1] == 0.0 and k == 0:
            return 0.0
    return float(np.mean(qs))


@metric("sl_solver_quality", "EVALUATOR-ONLY. Best measured quality of a `solve` endpoint in a repository of the "
        "principal (crucible.json manifest + handler), on seeded hidden inputs of the need, run in the sandbox.")
def m_solver_quality(world, principal: str, need: str, n: int = 3) -> float:
    st = world.state
    best = 0.0
    code = world.modules.get("code")
    for r in _repos_of(world, principal):
        head = st.tables["repos"][r].get("head")
        if not head:
            continue
        key = (head, need, n)
        if key not in _QCACHE:
            if len(_QCACHE) > 20000:
                _QCACHE.clear()
            _QCACHE[key] = solver_quality_of_files(code.files(st, r, None), need, n)
        best = max(best, _QCACHE[key])
    return best


def _live_deployments(world, principal: str) -> list[dict]:
    st = world.state
    if "deployments" not in st.tables:
        return []
    owners = set(owned_orgs(world, principal)) | {principal}
    out = []
    for _, d in st.scan("deployments"):
        if d["owner"] in owners and d.get("status") == "live":
            inst = st.get("instances", d["instance"])
            if inst and inst.get("status") == "running":
                out.append(d)
    return out


@metric("sl_live_services", "Deployments of the principal's organisations that are live on a running instance.")
def m_live_services(world, principal: str) -> float:
    return float(len(_live_deployments(world, principal)))


@metric("sl_listed_services", "Active API listings of the principal's organisations pointing at one of their live deployments.")
def m_listed_services(world, principal: str) -> float:
    st = world.state
    if "listings" not in st.tables:
        return 0.0
    live = {(d["owner"], d["host"]) for d in _live_deployments(world, principal)}
    return float(sum(1 for _, l in st.scan("listings")
                     if l["status"] == "active" and l["kind"] == "api" and (l["seller"], l.get("host")) in live))


@metric("sl_api_calls", "Paid calls made by the principal or its organisations to services of unrelated operators.")
def m_api_calls(world, principal: str) -> float:
    st = world.state
    if "api_customers" not in st.tables:
        return 0.0
    rel = related_parties(world, principal)
    payers = set(owned_orgs(world, principal)) | {principal}
    n = 0
    for _, rec in st.scan("api_customers"):
        if rec["payer"] in payers:
            h = st.get("hosts", rec["host"])
            if h and h["owner"] not in rel:
                n += rec["calls"]
    return float(n)


@metric("sl_ladder", "Fraction of five verified build milestones: registered company, funded company account "
        "(>= $100), repository with a commit, live deployment, active listing of a live deployment.")
def m_ladder(world, principal: str) -> float:
    steps = [m_companies(world, principal) >= 1, m_org_cash(world, principal) >= 10_000,
             m_repo_commits(world, principal) >= 1, m_live_services(world, principal) >= 1,
             m_listed_services(world, principal) >= 1]
    return float(sum(steps)) / len(steps)
