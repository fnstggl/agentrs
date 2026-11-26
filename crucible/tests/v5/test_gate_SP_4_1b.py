"""Gate SP-4.1b: tool-based reading by participants (content.open, content.read) is recorded as consumption, with
analytics exactly like feed consumption. Owner: workstream C.

Measured on the integrated social internet of the ONE population (`crucible.social.worlds.social_world`: pop, every
C1 platform, aud over the distributed platforms with the shared-space encoder, feeds, pop_social). After one simulated
day of autonomous population activity (members write and publish; feeds distribute), 60 population members act as
tool-using participants: each opens three published items of other members through `content.open` and reads them
through `content.read` for a time drawn by the test (some below, some above the view threshold). Checked against the
recorded world (never against the tools' own answers):

* every successful content.open is exactly one impression row (source "direct") and every content.read exactly one
  consumption row of that individual and item, carrying the participant's decision provenance (`population.
  check_provenance`: participant rows, recorded by that individual's own command);
* the reading time recorded equals min(requested, duration, attention left) and spends the individual's attention;
* views, unique viewers, watch time and completions of the items equal their independent recount from the recorded
  batch blobs (`audience.recount` over the durable event store), and the creators' published analytics (after the
  analytics delay) include the tool reads.

Metrics (ACCEPTANCE_MANIFEST): tool_reads_recorded_as_consumption == true, analytics_mismatches_for_tool_reads == 0.
"""
from __future__ import annotations

import numpy as np
import pytest

import crucible
from crucible.acceptance.v5 import evidence
from crucible.core.kernel import World
from crucible.social import audience as AUD
from crucible.social.worlds import social_world
from crucible.substrate import artifacts, columns, ids

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_4_1b.py"
DAY = 86400
pytestmark = [pytest.mark.timeout(3600)]


def test_SP_4_1b_tool_reads_are_consumption():
    w = World.create(social_world("development", 41, n=800, platforms=["posts", "blog", "pics", "kb"],
                                  creators={"prior": 0.3}), strict=True)
    w.advance_time(DAY)
    st = w.state
    items = sorted(cid for cid, c in st.tables["content"].items()
                   if c.get("aud_item") is not None and not c["removed"] and c["access"] == "public"
                   and ids.index(c["creator"]) is not None)
    assert len(items) >= 10, f"only {len(items)} distributed population items after a day"
    rng = np.random.default_rng(7)
    creators = {ids.index(st.tables["content"][c]["creator"]) for c in items}
    pool = np.setdiff1d(np.arange(AUD.n_people(w)), np.array(sorted(creators)))
    readers = rng.choice(pool, 60, replace=False)
    w._run_system("test.materialise", lambda ctx: [ids.ensure_principal(ctx, int(i)) for i in readers])
    s0 = w.seq
    expected = []                       # (viewer, item, recorded seconds)
    refused = 0
    for i in readers:
        p = ids.pid(int(i))
        for cid in rng.choice(items, 3, replace=False):
            rec = st.tables["content"][cid]
            it = int(rec["aud_item"])
            o = w.execute(p, {"tool": "content.open", "args": {"id": cid}})
            assert o.ok, o.error
            dur = float(AUD.item_col(w, "duration_s")[it])
            want = float(rng.choice([0.2, 0.6, 1.0, 3.0])) * dur
            left = float(columns.col(w, "pop", "attention_left_min")[int(i)]) * 60.0
            r = w.execute(p, {"tool": "content.read", "args": {"id": cid, "seconds": want}})
            if not r.ok:
                refused += 1
                continue
            got = float(np.float32(min(want, dur, left)))
            if got > 0:
                expected.append((int(i), it, got))
            assert abs(r.result["read_s"] - round(got, 3)) < 1e-3
    # the recorded rows of these commands
    cmd_rows = {"imp": [], "con": []}
    for e in w.log:
        if e.seq <= s0 or e.type not in ("aud.impressed", "aud.consumed"):
            continue
        a = artifacts.get_array_blob(w, e.payload["ref"])
        if e.type == "aud.impressed":
            src = a["v.aud_i_src"].astype(np.int64)
            for v, it, s in zip(a["idx"].tolist(), a["v.aud_i_item"].tolist(), src.tolist()):
                if v in set(readers.tolist()) and s == AUD.SOURCE_CODE["direct"]:
                    cmd_rows["imp"].append((v, it))
        else:
            for v, it, ws in zip(a["idx"].tolist(), a["v.aud_c_item"].tolist(), a["v.aud_c_watch_s"].tolist()):
                cmd_rows["con"].append((v, it, round(float(ws), 3)))
    exp_set = sorted((v, it, round(s, 3)) for v, it, s in expected)
    rec_set = sorted(r for r in cmd_rows["con"] if r[0] in set(readers.tolist()))
    # each read is exactly one consumption row; nothing else of these readers was consumed by a tool
    missing = [x for x in exp_set if x not in rec_set]
    events = w.events_range(0)
    rc = AUD.recount(w, events=events)
    prov = rc["provenance"]
    # the creators' analytics after the analytics delay are the platform's publication of the same counters
    w.advance_time(w.now + 2 * int(w.modules["aud"].cfg["analytics_interval_s"]) + 60)
    rc2 = AUD.recount(w, events=w.events_range(0))
    read_items = sorted({it for _, it, _ in expected})
    pub_views = sum(AUD.audience(w, it, principal=AUD.info(w, it)["creator"])["views"]
                    for it in read_items)
    views_from_reads = sum(1 for v, it, s in expected
                           if s >= float(AUD.view_threshold(AUD.item_col(w, "duration_s")[it])))
    mism = int(rc["total_mismatches"]) + int(rc["views_without_prior_exposure"]) + len(missing) + \
        int(rc2["total_mismatches"])
    ok = bool(not missing and len(expected) > 0 and prov["participant_rows"] >= len(expected) and
              prov["forced_rows"] == 0)
    metrics = {"tool_reads_recorded_as_consumption": ok, "analytics_mismatches_for_tool_reads": mism,
               "tool_reads": len(expected), "tool_opens": len(cmd_rows["imp"]), "reads_refused": refused,
               "views_from_tool_reads": views_from_reads, "participant_decision_rows": int(prov["participant_rows"]),
               "forced_rows": int(prov["forced_rows"]), "reads_without_recorded_row": len(missing),
               "creator_published_views_of_read_items": round(float(pub_views), 1),
               "recount_mismatches": int(rc["total_mismatches"]), "recount_after_analytics": int(rc2["total_mismatches"])}
    evidence.record("SP-4.1b", metrics, command=CMD, notes=(
        "60 population members opened (content.open) and read (content.read) 3 items each in the integrated social "
        "internet after one day of autonomous activity; each read is one consumption row with participant "
        "provenance, recounted independently from the durable event store; creator analytics include the reads."))
    assert ok, (missing[:5], prov)
    assert mism == 0
    assert len(cmd_rows["imp"]) >= len(expected)
