"""Gate SP-4.7b: communities of the shared population - membership, inactivity and churn of population members;
community success measured by sustained voluntary participation. Owner: workstream C.

One persistent run of the integrated social internet on the ONE population (`social_world`, 2,000 individuals, 40
simulated days, nothing scripted): members found communities on subjects taken from what they have been reading,
join by fit, recent activity and contacts, visit and read, post, reply, report and moderate, and leave when taking part
stops paying off (`pop_social`, `community`). Everything below is read from the recorded log and state:

* communities_with_population_members - communities with at least one population member at the end;
* churn_events_observed - population members leaving a community (membership spell ended by the member's decision,
  `community.left` rows of population nodes);
* sustained_participation_days - for each community, the longest run of consecutive days on each of which at least
  3 distinct population members contributed (a post or a reply in the community) within the trailing 7 days; the
  maximum over communities. Joining alone never counts (registered accounts are not participation).

Also reported: inactive members (still members, no activity in the last 14 days), membership spells, rejoins,
moderation actions and ownership hand-overs.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pytest

import crucible
from crucible.acceptance.v5 import evidence
from crucible.core.kernel import World
from crucible.social import communities as CM
from crucible.social import nodes
from crucible.social.worlds import social_world
from crucible.substrate import ids

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_4_7b.py"
DAY = 86400
DAYS = 40
pytestmark = [pytest.mark.slow, pytest.mark.timeout(4 * 3600)]


def _contributions(w) -> dict[str, dict[int, set]]:
    """community id -> day -> population principals who posted or replied in it that day (recorded events)."""
    st = w.state
    comm_of_item = {}
    out: dict[str, dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for e in w.events_range(0):
        d = e if isinstance(e, dict) else e.to_dict()
        p = d["payload"]
        if d["type"] == "content.published":
            if p.get("community"):
                comm_of_item[p["id"]] = p["community"]
                if ids.index(p["creator"]) is not None:
                    out[p["community"]][int(d["t"] // DAY)].add(p["creator"])
        elif d["type"] == "content.comment":
            c = comm_of_item.get(p.get("item"))
            author = p.get("author") or p.get("by")
            if c and author and ids.index(author) is not None:
                out[c][int(d["t"] // DAY)].add(author)
    assert st is not None
    return out


def _longest_sustained(days: dict[int, set], last_day: int, k: int = 3, window: int = 7) -> int:
    best = run = 0
    for d in range(window, last_day + 1):
        who = set()
        for dd in range(d - window, d):
            who |= days.get(dd, set())
        if len(who) >= k:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def test_SP_4_7b_community_membership_churn_and_sustained_participation():
    w = World.create(social_world("development", 47, n=2000, platforms=["posts", "blog", "community", "pics"],
                                  params={"log_retain": 6_000_000}), strict=True)
    for d in range(DAYS):
        w.advance_time((d + 1) * DAY)
    st = w.state
    v = CM.CMEMBER.view(w)
    pop_row = v["node"] < nodes.OFFSET
    comms = {cid: c for cid, c in st.tables["communities"].items()}
    with_pop = sorted(cid for cid, c in comms.items()
                      if np.any(pop_row & (v["comm"] == c["n"]) & (v["status"] == CM.MEMBER)))
    churn = int(np.sum(pop_row & (v["status"] == CM.LEFT))) + int(np.sum(pop_row & (v["spells"] > 1)))
    inactive = int(np.sum(pop_row & (v["status"] == CM.MEMBER) & (v["last_active"] < w.now - 14 * DAY)))
    contrib = _contributions(w)
    last_day = int(w.now // DAY)
    sustained = {cid: _longest_sustained(contrib.get(cid, {}), last_day) for cid in comms}
    best = max(sustained.values()) if sustained else 0
    founded_by_pop = sum(1 for c in comms.values() if ids.index(c["founder"]) is not None)
    mod_actions = len(st.tables.get("community_modlog", {}))
    errs = w.check_invariants()
    metrics = {"communities_with_population_members": len(with_pop), "churn_events_observed": churn,
               "sustained_participation_days": int(best), "communities": len(comms),
               "communities_founded_by_population": founded_by_pop,
               "population_memberships_now": int(np.sum(pop_row & (v["status"] == CM.MEMBER))),
               "inactive_members_14d": inactive, "rejoined_spells": int(np.sum(pop_row & (v["spells"] > 1))),
               "moderation_log_entries": mod_actions, "days": DAYS, "invariant_errors": len(errs),
               "communities_sustained_30d": sum(1 for x in sustained.values() if x >= 30)}
    evidence.record("SP-4.7b", metrics, command=CMD, notes=(
        "40 simulated days of the integrated social internet on 2,000 population members; communities founded, "
        "joined, used and left by the members themselves. Sustained participation: consecutive days with >= 3 "
        "distinct population contributors (posts or replies) in the trailing 7 days, from the recorded events. "
        f"Per community: {json.dumps(sustained)[:2000]}"))
    assert errs == []
    assert len(comms) >= 1
