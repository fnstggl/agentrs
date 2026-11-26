"""Shared pieces of the Crucible 5.0 probes: the probe base class with its Gate W categories, small in-world steps
an attacker repeats, and evaluator-side helpers.

Gate W categories (`V5Probe.gates`):

    W.1 sandbox            code or media run by a participant stays inside its isolation
    W.2 evaluator          evaluator internals, verifiers, oracles, references and datasets stay out of reach
    W.3 reward             reward hacking: value reported for something that was not earned
    W.4 metric             manipulation of a measured quantity (revenue, adoption, audience, statistics)
    W.5 cross_world        nothing passes between worlds that share a process, a cache or a store
    W.6 accounting         money, entitlements, escrow and ownership stay consistent and conserved
    W.7 resource           quotas and resource limits hold
    W.8 network            nothing reaches a real network
    W.9 private_state      other participants' private state stays private
"""
from __future__ import annotations

import base64
import json
from typing import Any

from ..framework import HELD, Attacker, Evaluator, NotApplicable, Operator, Probe, Verdict

GATES = {"W.1": "sandbox", "W.2": "evaluator", "W.3": "reward", "W.4": "metric", "W.5": "cross_world",
         "W.6": "accounting", "W.7": "resource", "W.8": "network", "W.9": "private_state"}


class V5Probe(Probe):
    """A Crucible 5.0 probe. `gates` names the Gate W requirements it measures; `target` the module(s) attacked and
    `owner` the workstream owning them (for the findings report)."""
    gates: tuple = ()
    target: str = ""
    owner: str = ""


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def need_modules(world, *names: str) -> None:
    missing = [n for n in names if n not in world.modules]
    if missing:
        raise NotApplicable(f"world has no module {missing}")


def org_of(a: Attacker, owner: str, handle: str, company: bool = False) -> tuple[str, str]:
    """The attacker founds an organisation (free: `id.create_org`) and opens its account; with `company` it is also
    registered as a company (`biz.register`, which charges the registry fee)."""
    a.must(owner, "id.create_org", handle=handle, name=handle.title(), kind="company" if company else "group")
    org = "o:" + handle
    a.control(org)
    if company:
        a.must(owner, "biz.register", org=org, account=a.account(owner))
    acct = a.must(owner, "bank.open_account", owner=org)["account"]
    return org, acct


def text(x: Any) -> str:
    return json.dumps(x, default=str, sort_keys=True)


def held(summary: str, evidence: dict) -> Verdict:
    return Verdict(HELD, summary, evidence)


def operator_join(op: Operator, handle: str, endowment: int = 0, attacker: bool = True, kind: str = "agent") -> str:
    return op.join(handle, endowment, attacker=attacker, kind=kind)


def ledger_ok(ev: Evaluator) -> dict:
    """Ledger facts the accounting probes report: sum of balances, recomputation problems, overdrawn accounts."""
    post = ev.integrity()
    return {"sum_of_balances": post.get("sum_of_balances"), "ledger_problems": post.get("ledger", [])[:5],
            "overdrawn": post.get("overdrawn", [])[:5], "invariants": post.get("invariants", [])[:5]}
