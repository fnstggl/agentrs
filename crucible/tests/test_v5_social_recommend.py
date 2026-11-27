"""Unit test (workstream C, wave 3b): a member who still relies on a service for its work after three uses writes
about it when it creates (social.popcreate `_recommendable`, `_recommendation`, `act_create`).

`economy.work` cancels a provider once the member's own valuation of it turns negative after three outcomes, so
continued reliance after three uses (with at least one accepted task) is the member's revealed judgement that the
service pays off - also when not every task is accepted. Such a member writes, with probability REVIEW_P per
creation, a recommendation naming the service's address in its preferred available text medium instead of its
usual piece (before, only members whose chosen medium happened to be text and whose record showed >= 2/3 accepted
wrote one, and a balanced civilisation produced none in 14 days: SP-4.8b measured 0 chains).

Setup (labelled): a startup world of the population with the social platforms at genesis; up to 24 members with work
subscribe to an incumbent's service through `economy.work`'s own adoption path (`_adopt`: a real subscription from
their own account); their record of three uses (one accepted) is written as `work.outcomes` events through the
work module's applier, standing in for three working sessions. The test sets REVIEW_P to 1 so that each of 12 of
them writes about it in this creation session, whatever medium it would have chosen; members with no such record
never do. (The rate itself, 0.35 per creation, is a hand-set constant: FIDELITY_REGISTER population-creator-model.)"""
from __future__ import annotations

import time

import numpy as np

import crucible
from crucible.commerce.worlds import startup_spec
from crucible.core.kernel import World, WorldSpec
from crucible.social import popcreate as PC
from crucible.social import worlds as SW
from crucible.substrate import columns, ids

crucible.load_all_modules()


def _world():
    base = startup_spec(21, n=260, formats=["schedule.jobs"])
    plats = ["posts", "blog", "launch"]
    mods = []
    for m in base.modules:
        if m["name"] == "work":
            mods += [{"name": x, "config": {}} for x in ("content", "social_ids", "social_rows", "social", "studio",
                                                          "posts", "blog", "launch")]
            mods.append({"name": "aud", "config": {"platforms": plats, "max_items": 4096, "max_creators": 1024}})
            mods.append({"name": "feed", "config": SW.feed_config(plats)})
            mods.append({"name": "pop_social", "config": {}})
        mods.append(m)
        if m["name"] == "cols":
            mods.append({"name": "space", "config": {}})
    return World.create(WorldSpec("startup-social-recommend", 21, mods, dict(base.params)))


def test_a_member_relying_on_a_service_after_three_uses_writes_about_it(monkeypatch):
    t0 = time.time()
    w = _world()
    st = w.state
    work = w.modules["work"]
    services = [(pid, p["need"], p["delivery"]["host"]) for pid, p in st.scan("products")
                if p["status"] == "launched" and (p["delivery"] or {}).get("type") == "service"]
    assert services
    pid, need, host = services[0]
    fam = columns.col(w, "work", "fam")
    act = columns.col(w, "work", "active")
    users = [i for i in range(columns.rows(w, "pop")) if fam[i] >= 0 and act[i]
             and need in {n for _, n in work._steps(i)}][:24]
    others = [i for i in range(columns.rows(w, "pop")) if i not in users][:24]
    assert len(users) >= 20
    adopted = []
    from crucible.economy.work import _needs
    size = columns.col(w, "work", "size")

    def setup(ctx):
        prov = work._provider(st, pid)
        for i in users:
            p = ids.ensure_principal(ctx, i)
            sample, _ = _needs()[need].generate(w.rng_for("work-sample", i, need), float(size[i]))
            if work._adopt(ctx, p, i, need, pid, prov, sample):
                adopted.append(i)
        ctx.emit("work.outcomes", {"rows": [[i, pid, need, 3, 1, 2, 0, 9.0] for i in adopted], "t": int(ctx.now)},
                 vis="internal")
    w._run_system("test.adoption_records", setup)
    assert len(adopted) >= 20
    res = {}

    def check(ctx):
        res["users"] = {i: PC._recommendable(ctx, i) for i in adopted}
        res["others"] = {i: PC._recommendable(ctx, i) for i in others}
    w._run_system("test.recommendable", check)
    assert all(len(v) == 1 and v[0][0] == pid for v in res["users"].values())
    assert not any(res["others"].values())

    # the decision under test: every member with such a service writes about it this time (REVIEW_P = 1 for the
    # test), whatever medium it would have chosen; members without one create as usual
    monkeypatch.setattr(PC, "REVIEW_P", 1.0)
    from crucible.social.popinbox import _hosts_of
    before = set(st.tables.get("content", {}))
    pop_social = w.modules["pop_social"]
    writers_idx = adopted[:12]
    idx = np.array(sorted(writers_idx + others[:6]), np.int64)
    w._run_system("test.create", lambda ctx: pop_social.act_create(ctx, idx, ctx.now))
    new = {cid: st.tables["content"][cid] for cid in st.tables["content"] if cid not in before}
    recs = {cid: c for cid, c in new.items() if host in _hosts_of(w, cid)}
    by = {ids.index(c["creator"]) for c in recs.values()}
    assert by == set(writers_idx), (sorted(by), writers_idx)      # each of them wrote one ...
    assert len(recs) == len(writers_idx)
    assert {c["platform"] for c in recs.values()} <= {"posts", "blog", "launch"}   # ... in text (or as a hunter)
    assert not by & set(others)
    from crucible.social.content import text_of
    body = text_of(w, next(iter(recs)))
    assert f"http://{host}.crucible/" in body and "1 of my last 3 tasks" in body   # its own counts, honestly
    assert time.time() - t0 < 600
    assert not [e for e in w.log if e.type == "proc.failed"]
    assert w.check_invariants() == []
