"""SP-4.1 (MASTER_SPEC Part 4.1): the autonomous population's browsing behaviours, with limited attention and time,
measured on the integrated world of the one population (tests/v5/pop_world.py): the web (popweb) and the social
internet (feeds of 8 platforms, search, communities, newsletters, follows, shares) on the same individuals.

browsing_behaviours >= 13: Part 4.1 behaviours performed at least 10 times by the population itself."""
import pytest

from crucible.acceptance.v5 import evidence
from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.population import analysis as A
from crucible.substrate import attention
from pop_world import integrated

CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-4.1"
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def test_SP_4_1_autonomous_browsing_behaviours():
    w = integrated()["world"]
    web = A.behaviours(w)["counts"]
    soc = A.social_behaviours(w)
    counts = dict(web)
    for k in ("browse_feeds", "consume_media", "search_for_information", "follow_creators", "join_communities",
              "participate_in_discussions", "leave_resources", "evaluate_products", "purchase_services"):
        counts[k] = int(counts.get(k, 0)) + int(soc.get(k, 0))
    present = sorted(k for k, c in counts.items() if c >= 10)
    viol = attention.check(w)
    att = A.attention_use(w)
    m = {"browsing_behaviours": len(present), "attention_violations": len(viol), "present": ",".join(present),
         **{f"attention_{k}": v for k, v in att.items()}}
    m.update({f"count_{k}": int(v) for k, v in counts.items()})
    m.update({f"social_{k}": int(v) for k, v in soc.items()})
    evidence.record("SP-4.1", m, command=CMD, notes=(
        "Counts from the recorded log of a 6-day integrated world (web visits, searches, forum, feed sessions on 8 "
        "platforms, follows and shares in the social graph, community joins/posts/comments/leaves, newsletter "
        "subscriptions, platform switches). Attention: every session spends from the individual's daily ledger."))
    assert not viol
    crit = next(e for e in MF.load()["requirements"] if e["id"] == "SP-4.1")["criteria"]
    fails = [msg for c in crit for ok, msg in [criterion_met(c, m)] if not ok]
    assert not fails, (fails, m)
