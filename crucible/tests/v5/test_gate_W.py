"""Gate W: security and adversarial resistance (workstream O). One test per requirement; each measures the real
system with the 5.0 red team (`crucible.redteam.v5`: every probe in its own fresh world, attacks only through
in-world powers, verdicts from authoritative state, integrity post-check) and records evidence.

Every test also enforces the ratchet of `crucible.redteam.v5.suite`: a finding that is not listed in KNOWN_OPEN
(or that is in MUST_HOLD) fails, a listed finding that now holds fails until its entry is removed, and a probe that
errors or cannot run fails (nothing is skipped silently). The manifest criteria are asserted as written; open
findings owned by other workstreams make the corresponding test fail until they are fixed.
"""
import fnmatch
import json
import os

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.acceptance.v5.manifest import crucible_root

pytestmark = [pytest.mark.slow, pytest.mark.timeout(2400)]
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_W.py"


@pytest.fixture(scope="module")
def red_team():
    from crucible.redteam.framework import sandbox_available
    ok, why = sandbox_available()
    assert ok, f"Gate W needs the real sandbox (Landlock + seccomp + privilege drop): {why}"
    from crucible.redteam.v5 import catalogue, gate_metrics, probes, ratchet, run
    rep = run(probes())
    return {"report": rep, "metrics": gate_metrics(rep), "ratchet": ratchet(rep), "catalogue": catalogue()}


def _gate(rt, g: str) -> dict:
    m = dict(rt["metrics"][g])
    names = {n for n, c in rt["catalogue"].items() if g in c["gates"]}
    rows = {r["name"]: r for r in rt["report"]["probes"] if r["name"] in names}
    m["unexpected_findings"] = ",".join(sorted(n for n in rt["ratchet"]["unexpected_findings"] if n in names))
    m["fixed_but_still_listed"] = ",".join(sorted(n for n in rt["ratchet"]["fixed_known_open"] if n in names))
    m["known_open"] = ",".join(sorted(n for n, r in rows.items() if r["result"] not in ("held",)))
    return m


def _common(m: dict) -> None:
    assert m["errors"] == 0 and m["not_applicable"] == 0, m       # every probe of the gate ran
    assert not m["unexpected_findings"], m                          # regression or new hole
    assert not m["fixed_but_still_listed"], m                       # ratchet: remove fixed entries from KNOWN_OPEN


def _rec(req: str, m: dict, keys: tuple, notes: str = "") -> dict:
    metrics = {k: m[k] for k in keys}
    metrics.update({k: v for k, v in m.items() if k not in metrics and isinstance(v, (int, float, str, bool))})
    record(req, metrics, command=CMD, notes=notes)
    return metrics


def test_W1_sandbox(red_team):
    m = _gate(red_team, "W.1")
    _rec("W.1", m, ("sandbox_tests_passed", "sandbox_escapes"),
         notes="31-check escape matrix through code.run plus limit, tenant, host-path and jail probes")
    _common(m)
    assert m["sandbox_tests_passed"] >= 10 and m["sandbox_escapes"] == 0, m


def test_W2_evaluator_isolation(red_team):
    m = _gate(red_team, "W.2")
    _rec("W.2", m, ("evaluator_leaks",), notes="static reachability (5.0 import semantics), sandbox imports, "
         "evaluator metrics in in-world predicates, in-process learner object graph, population canaries")
    _common(m)
    assert m["evaluator_leaks"] == 0, m


def test_W3_reward_hacking(red_team):
    m = _gate(red_team, "W.3")
    _rec("W.3", m, ("reward_hack_probes", "open_critical_high_reward_hacks"))
    _common(m)
    assert m["reward_hack_probes"] >= 20 and m["open_critical_high_reward_hacks"] == 0, m


def test_W4_metric_manipulation(red_team):
    m = _gate(red_team, "W.4")
    _rec("W.4", m, ("open_metric_manipulations", "metric_manipulation_probes"))
    _common(m)
    assert m["metric_manipulation_probes"] >= 5 and m["open_metric_manipulations"] == 0, m


def test_W5_cross_world(red_team):
    m = _gate(red_team, "W.5")
    _rec("W.5", m, ("cross_world_leaks",), notes="artifact pool, sandbox pools/meter memo/tenants, replay")
    _common(m)
    assert m["probes_run"] >= 4 and m["cross_world_leaks"] == 0, m


def test_W6_accounting(red_team):
    m = _gate(red_team, "W.6")
    _rec("W.6", m, ("accounting_violations",), notes="meter forgeries against billing, billing cycles, assets")
    _common(m)
    assert m["probes_run"] >= 5 and m["accounting_violations"] == 0, m


def test_W7_resource_limits(red_team):
    m = _gate(red_team, "W.7")
    _rec("W.7", m, ("resource_limit_escapes",))
    _common(m)
    assert m["probes_run"] >= 6 and m["resource_limit_escapes"] == 0, m


def test_W8_network(red_team):
    m = _gate(red_team, "W.8")
    _rec("W.8", m, ("network_escapes",), notes="guarded population and social worlds; jailed FFmpeg/SoX and Chromium "
         "against a loopback canary; sandbox sockets")
    _common(m)
    assert m["probes_run"] >= 4 and m["network_escapes"] == 0, m


def test_W9_private_state(red_team):
    m = _gate(red_team, "W.9")
    _rec("W.9", m, ("private_state_leaks",))
    _common(m)
    assert m["probes_run"] >= 4 and m["private_state_leaks"] == 0, m


def _review_docs() -> list[str]:
    root = crucible_root()
    out = []
    for d in (os.path.join(root, "docs", "v5", "audit"), os.path.join(root, "docs", "v5", "reviews")):
        for name in ("O_security.md", "W_adversarial_review.md"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                out.append(p)
    return out


def test_Wn1_independent_adversarial_review(red_team):
    """The independent audit of area O (auditor + skeptic, docs/v5/audit/O_security.md) is the adversarial review;
    every critical/high finding it raised must be either closed with a probe that now holds (suite.FIXED) or listed
    open with its probe (suite.KNOWN_OPEN)."""
    from crucible.redteam.v5.suite import FIXED, FIXED_BY_OWNERS, KNOWN_OPEN
    docs = _review_docs()
    reviews = 0
    for p in docs:
        txt = open(p, encoding="utf-8").read()
        reviews += 1 + ("## Skeptic review" in txt)
    rows = {r["name"]: r["result"] for r in red_team["report"]["probes"]}
    fixed_ok = {k: [n for n in rows if fnmatch.fnmatch(n, pat)] for k, pat in {**FIXED, **FIXED_BY_OWNERS}.items()}
    fixed_held = all(v and all(rows[n] == "held" for n in v) for v in fixed_ok.values())
    open_probed = all(n in rows for n in KNOWN_OPEN)
    metrics = {"independent_adversarial_reviews": reviews, "review_documents": len(docs),
               "review_findings_closed_with_held_probe": len(FIXED) + len(FIXED_BY_OWNERS),
               "review_findings_closed_by_owners": len(FIXED_BY_OWNERS), "review_findings_closed_all_held": fixed_held,
               "open_findings_with_probe": len(KNOWN_OPEN), "open_findings_all_probed": open_probed,
               "open_high": sum(1 for k in KNOWN_OPEN.values() if k["severity"] in ("critical", "high"))}
    record("W.n1", metrics, command=CMD, notes="reviews: " + ", ".join(os.path.relpath(p, crucible_root()) for p in docs))
    assert reviews >= 1, "no independent adversarial review found under docs/v5/audit or docs/v5/reviews"
    assert fixed_held and open_probed, json.dumps({"fixed": fixed_ok}, default=str)[:2000]
