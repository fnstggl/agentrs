"""Consumer behaviour models, segments, latent and derived demand (economy/behavior.py, economy/consumers.py).

Everything here runs on real worlds: firms are real companies with deployed services whose requests are executed and
verified through the normal demand path; interventions are recorded system calls (a price change by the owner, an
advertising campaign, ratings by paying customers) on one of two forks of the same world. Worlds use
`params.cost_model = "deterministic"` so that every run, fork and regeneration is exactly reproducible.

Legacy equivalence: `LEGACY_DIGESTS` was recorded by running this file's `LEGACY_SCENARIOS` against the untouched
Crucible 3.0 tree:

    PYTHONPATH=<crucible-3.0>/src python tests/test_behavior.py --record
"""
import hashlib
import math
import os
import sys

import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.core.canonical import canonical_bytes  # noqa: E402
from crucible.core.errors import ProtocolError  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy.needs import NEEDS  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

try:
    from crucible.economy.behavior import (BEHAVIOR_MODELS, CALIBRATION, budget_envelopes, build_demand,  # noqa: E402
                                           build_segments, segment_counts)
    from crucible.economy.consumers import _charge, demand_state, seed_adopters, segment_aggregates  # noqa: E402
except ImportError:            # the Crucible 3.0 tree: only `--record` (below) runs there
    BEHAVIOR_MODELS = None

DAY, HOUR = 86400, 3600
MODELS = ["legacy", "contagion", "reputation_gate", "explorer", "bargain"]


# ====================================================================== legacy equivalence (recorded from 3.0)
def _legacy_tiny(seed):
    """A small 3.0 world that exercises every branch of the population update: organic and sponsored search,
    word of mouth, shopping, turnover, reviews, population shocks and an offering that closes."""
    spec = generate({"family": "tiny", "overrides": {
        "n_consumers": 400, "needs": ["route.shortest", "schedule.jobs"], "firms_per_need": 2,
        "params": {"cost_model": "deterministic"},
        "module_config": {"shocks": {"events": [
            {"day": 1, "kind": "demand_surge", "need": "schedule.jobs", "mult": 1.5},
            {"day": 1, "kind": "pref_drift", "need": "route.shortest", "toward": [0.7, 0.1, 0.2], "strength": 0.4},
            {"day": 2, "kind": "new_need", "need": "match.stable", "fraction": 0.2}]}}}}, seed)
    w = World.create(spec)
    st = w.state
    firms = [dict(f, org=org) for org, f in st.scan("npc_firms")]
    adv = firms[0]
    w.system_call("ads.create_campaign", {
        "account": w.modules["bank"].primary_account(st, adv["org"]), "keywords": list(NEEDS[adv["need"]].keywords[:3]),
        "bid_cents": 1, "daily_budget_cents": 50_000, "title": "try us", "hosts": [adv["host"]],
        "ref": adv["listing"]}, as_principal=adv["owner"])
    w.advance_time(36 * HOUR)
    gone = firms[-1]
    w.system_call("market.update", {"listing": gone["listing"], "status": "closed"}, as_principal=gone["owner"])
    w.advance_time(60 * HOUR)
    return w


def _legacy_family(family, seed, n, hours, **cons):
    ov = {"n_consumers": n, "params": {"cost_model": "deterministic"}}
    if cons:
        ov["module_config"] = {"consumers": cons}
    w = World.create(generate({"family": family, "overrides": ov}, seed))
    w.advance_time(hours * HOUR)
    return w


LEGACY_SCENARIOS = {
    "tiny_ads_shocks_closure_seed3": lambda: _legacy_tiny(3),
    "tiny_ads_shocks_closure_seed4": lambda: _legacy_tiny(4),
    "bazaar_seed1": lambda: _legacy_family("bazaar", 1, 600, 30, k_samples=1),
    "open_digital_seed2": lambda: _legacy_family("open_digital", 2, 300, 30),
}


def _consumer_digest(w) -> str:
    """Digest of the consumer population only (its arrays and tables), to tell a divergence of the population
    from a change elsewhere in the world."""
    st = w.state
    h = hashlib.blake2b(digest_size=16)
    for name in sorted(st.arrays):
        if name.startswith("c_"):
            a = np.ascontiguousarray(st.arrays[name])
            h.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    for t in sorted(st.tables):
        if t.startswith("cons_"):
            for rid in sorted(st.tables[t]):
                h.update(t.encode() + rid.encode() + canonical_bytes(st.tables[t][rid]))
    return h.hexdigest()


def _fingerprint(w) -> dict:
    return {"consumers": _consumer_digest(w), "digest": w.digest(), "last_event_hash": w.last_hash, "events": w.seq}


# recorded with the untouched Crucible 3.0 tree (branch feat/crucible-internet, commit 11e5613)
LEGACY_DIGESTS = {
    "bazaar_seed1": {
        "consumers": "d1ba64febef206a1e1e9c214e80863e6",
        "digest": "ce163b4e93bf1c2d2a70a3ee2b4c1645",
        "last_event_hash": "e1ac4f7c274ba006f4b5757cf2714d98",
        "events": 2037,
    },
    "open_digital_seed2": {
        "consumers": "196d19626852fb5b9083503a052bb81d",
        "digest": "3c2bb751a052e5fbfe144f37ea51d152",
        "last_event_hash": "ad781c38bb5e1016763a4dee0e62f637",
        "events": 2615,
    },
    "tiny_ads_shocks_closure_seed3": {
        "consumers": "01d84c3c3d8791fb44c2dac5215b790b",
        "digest": "a4a75925dcd7d672c43faed933be0d0e",
        "last_event_hash": "2919698997c7ce14a4d24ebcc27219f0",
        "events": 3280,
    },
    "tiny_ads_shocks_closure_seed4": {
        "consumers": "e8eae8d3ef58685f9a88db2cb6e9c973",
        "digest": "a583932543b504fcda1517a1e9750908",
        "last_event_hash": "213ff3fae55b67d2523817acfdc245e1",
        "events": 3297,
    },
}

NEW_ARRAYS = {"c_seg", "c_active", "c_thr", "c_quit", "c_quit_h", "c_trust", "c_evid", "c_eps", "c_mslot", "c_mval"}


@pytest.mark.parametrize("name", sorted(LEGACY_SCENARIOS))
def test_unconfigured_worlds_are_byte_identical_to_crucible_3(name):
    w = LEGACY_SCENARIOS[name]()
    got, want = _fingerprint(w), LEGACY_DIGESTS[name]
    assert got["consumers"] == want["consumers"], "the consumer population diverged from Crucible 3.0"
    assert got == want                     # the whole world: state digest, hash of the last event, number of events
    st = w.state
    assert not NEW_ARRAYS & set(st.arrays) and not {"cons_demand", "cons_ok_days"} & set(st.tables)
    assert int((st.arrays["c_prov"] >= 0).sum()) > 0.3 * len(st.arrays["c_prov"])       # the scenario is not idle
    assert w.check_invariants() == []


def test_spelling_out_the_defaults_changes_nothing():
    """`behavior: legacy`, no segments, no latent needs and no edges is the 3.0 population: same arrays, same
    tables, no new state. (The world digest itself differs only because the spec hash is part of the event chain.)"""
    def run(cons):
        ov = {"n_consumers": 300, "needs": ["schedule.jobs", "match.stable"], "firms_per_need": 1,
              "params": {"cost_model": "deterministic"}, "module_config": {"consumers": cons}}
        w = World.create(generate({"family": "tiny", "overrides": ov}, 5))
        w.advance_time(30 * HOUR)
        return w
    a = run({})
    b = run({"behavior": {"model": "legacy"}, "segments": [], "latent": {}, "derived": []})
    assert _consumer_digest(a) == _consumer_digest(b)
    assert a.state.table_digests() == b.state.table_digests() and sorted(a.state.arrays) == sorted(b.state.arrays)
    assert a.seq == b.seq and int((a.state.arrays["c_prov"] >= 0).sum()) > 50


# ====================================================================== small worlds with real firms
PROVIDER = [{"handle": "cloud1", "name": "Cloud1", "cores": 400, "price": 100, "reliability": 1.0, "cash": 50_000_000}]


def firm(handle, need="match.stable", tier=1, price=3, cores=2):
    """An incumbent company: owner, registered company, repository with a real implementation (`tier` of the
    need's reference solvers), rented instance, live deployment and market listing. Its price and code stay as
    configured (the firm's own repricing and R&D are switched off) so that each test controls what changes."""
    return {"handle": handle, "name": handle.title(), "need": need, "tier": tier, "price": price, "host": handle,
            "provider": "o:cloud1", "cores": cores, "cash": 5_000_000,
            "policy": {"price_step": 0.0, "outsource": False}}


def make(cons, firms, n=800, needs=("match.stable",), seed=1):
    ov = {"n_consumers": n, "needs": list(needs), "firms_per_need": 0, "bounties": 0, "workers": 0,
          "params": {"cost_model": "deterministic"},
          "module_config": {"consumers": {"k_samples": 2, **cons},
                            "civ": {"firms": firms, "providers": PROVIDER, "lenders": [], "bounties": []}}}
    return World.create(generate({"family": "tiny", "overrides": ov}, seed), strict=True)


def run(w, hours):
    w.advance_time(w.now + hours * HOUR)
    return w


def listing_of(w, host):
    return next(l for l, r in w.state.scan("listings") if r["host"] == host)


def slot_of(w, host):
    return next(o["slot"] for l, o in w.state.scan("cons_offers") if w.state.get("listings", l)["host"] == host)


def customers(w, host, mask=None) -> int:
    sel = w.state.arrays["c_prov"] == slot_of(w, host)
    return int((sel if mask is None else sel & mask).sum())


def share(w, host) -> float:
    prov = w.state.arrays["c_prov"]
    return customers(w, host) / max(1, int((prov >= 0).sum()))


def aware_of(w, host) -> int:
    return int((w.state.arrays["c_aware"] == slot_of(w, host)).any(axis=1).sum())


def shelve(w, host):
    """The firm takes its listing off the market (the service stays deployed) until `launch`."""
    w.system_call("market.update", {"listing": listing_of(w, host), "status": "closed"},
                  as_principal=f"p:{host}-ceo")


def launch(w, host, need):
    w.system_call("market.update", {"listing": listing_of(w, host), "status": "active",
                                    "description": f"{NEEDS[need].title}. Call {host}.solve. Keywords: "
                                                   f"{' '.join(NEEDS[need].keywords)}"}, as_principal=f"p:{host}-ceo")


def set_price(w, host, cents):
    w.system_call("market.update", {"listing": listing_of(w, host), "price_cents": cents},
                  as_principal=f"p:{host}-ceo")


def advertise(w, host, need):
    """The firm buys search advertising on the keywords consumers of `need` search for."""
    acct = w.modules["bank"].primary_account(w.state, f"o:{host}")
    return w.system_call("ads.create_campaign", {
        "account": acct, "keywords": list(NEEDS[need].keywords[:3]), "bid_cents": 1, "daily_budget_cents": 100_000,
        "title": f"{host}: {need}", "text": "try us", "hosts": [host], "ref": listing_of(w, host)},
        as_principal=f"p:{host}-ceo")["campaign"]


def rate(w, host, need, stars, n=4, tag="rater"):
    """`n` new participants each buy one real request from the service and then rate its listing."""
    for i in range(n):
        h = f"{tag}{i}"
        w.system_call("admin.join", {"handle": h, "endowment": 10_000})
        inp, _ = NEEDS[need].generate(np.random.default_rng(i), 0.5)
        r = w.execute(f"p:{h}", {"tool": f"{host}.solve", "args": inp})
        assert r.ok and r.result["status"] == 200, (r.error, r.result)
        r = w.execute(f"p:{h}", {"tool": "market.rate", "args": {"listing": listing_of(w, host), "stars": stars}})
        assert r.ok, r.error


def entrant_world(model, need="match.stable", tiers=(1, 1), params=None, **cons):
    """`alfa` serves the market for three days; then `beta` (a service that was deployed but not listed) is
    launched at the same price, with no ratings, no customers and no advertising. Returns the world one hour on."""
    w = make({"behavior": {"model": model, "params": params or {}}, **cons},
             [firm("alfa", need, tiers[0], 3), firm("beta", need, tiers[1], 3)], needs=(need,))
    shelve(w, "beta")
    run(w, 72)
    assert customers(w, "beta") == 0
    launch(w, "beta", need)
    run(w, 1)
    assert w.state.get("listings", listing_of(w, "beta"))["rating_n"] == 0
    return w


def replays(w, base_blob=None) -> bool:
    return World.replay(w.spec, [e.to_dict() for e in w.log], base_blob=base_blob).digest() == w.digest()


def healthy(w) -> bool:
    return w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == [] and \
        not [e for e in w.log if e.type == "proc.failed"]


# ====================================================================== the registry and its honesty
def test_registry_and_calibration_record():
    assert sorted(BEHAVIOR_MODELS) == sorted(MODELS)
    for name in MODELS + ["segments", "derived_demand"]:
        rec = CALIBRATION[name]
        assert rec["calibrated"] is False and rec["status"] in ("plausible hypothesis", "design choice")
        assert len(rec["resembles"]) > 40
    assert "No behaviour model" in CALIBRATION["_note"]
    # models differ in what they listen to, not only in constants
    assert BEHAVIOR_MODELS["legacy"].uses_search and BEHAVIOR_MODELS["reputation_gate"].uses_search
    assert not BEHAVIOR_MODELS["explorer"].uses_search and not BEHAVIOR_MODELS["bargain"].uses_search


def test_bad_configuration_is_refused():
    def cfg(**kw):
        return {"search_rate": 0.02, "wom_rate": 0.01, "review_rate": 0.0005, "ticket_rate": 0.01,
                "shop_threshold": 0.35, "personas": 12, **kw}
    with pytest.raises(ProtocolError, match="unknown behaviour model"):
        build_segments(cfg(behavior={"model": "herd"}))
    with pytest.raises(ProtocolError, match="no parameters"):
        build_segments(cfg(behavior={"model": "contagion", "params": {"treshold": [0.1, 0.2]}}))
    with pytest.raises(ProtocolError, match="not both"):
        build_segments(cfg(behavior={"model": "legacy"}, segments=[{"share": 1, "model": "bargain"}]))
    with pytest.raises(ProtocolError, match="more than 1"):
        build_segments(cfg(segments=[{"share": 1, "model": "bargain", "params": {"budget_share": 0.7}},
                                     {"share": 1, "model": "legacy", "params": {"budget_share": 0.6}}]))
    with pytest.raises(ProtocolError):
        build_demand({"derived": [{"src": "a", "dst": "b", "scale": 0}]})
    with pytest.raises(ProtocolError):
        build_demand({"latent": {"a": 1.5}})
    assert build_demand({}) is None and build_demand({"latent": {}, "derived": []}) is None
    with pytest.raises(ProtocolError):             # the same check guards world creation
        make({"behavior": {"model": "herd"}}, [firm("alfa")], n=50)
    segs = build_segments(cfg(segments=[{"name": "a", "share": 3, "model": "bargain"},
                                        {"name": "b", "share": 1, "model": "explorer", "params": {"margin": 0.1}}]))
    assert [round(s.share, 2) for s in segs] == [0.75, 0.25] and segs[1].params["margin"] == 0.1
    assert segs[0].params["search_rate"] == 0.02 and segment_counts(segs, 10) == [8, 2] and budget_envelopes(segs) is None
    assert sum(segment_counts(build_segments(cfg(segments=[{"share": 1, "model": m} for m in MODELS])), 1003)) == 1003


# ====================================================================== causal structure, model by model
def test_advertising_wins_customers_under_legacy_but_only_awareness_under_reputation_gate():
    """One intervention on a matched pair of forks: the entrant buys search advertising (in the control fork it
    is only found organically, so the campaign doubles the consumers exposed to it each hour).
    legacy: exposure -> awareness -> trial. reputation_gate: exposure -> awareness, and nothing else until the
    offering has ratings; once paying customers have rated it, the awareness the campaign bought converts."""
    need = "match.stable"
    arms = {}
    for model in ("legacy", "reputation_gate"):
        w = entrant_world(model, need)
        for arm in ("control", "ads"):
            f = w.fork(f"{model}-{arm}")
            if arm == "ads":
                camp = advertise(f, "beta", need)
            run(f, 24)
            arms[model, arm] = f
        assert arms[model, "ads"].state.get("ads_campaigns", camp)["impressions"] > 100
        assert aware_of(arms[model, "ads"], "beta") > 1.3 * aware_of(arms[model, "control"], "beta")
    c, a = customers(arms["legacy", "control"], "beta"), customers(arms["legacy", "ads"], "beta")
    assert a >= 1.5 * c and a - c >= 20, (c, a)
    gate = {}
    for arm in ("control", "ads"):
        f = arms["reputation_gate", arm]
        needs_ratings = f.state.arrays["c_evid"] > 0          # everyone but the few who try unrated offerings
        assert f.state.get("listings", listing_of(f, "beta"))["rating_n"] < 3
        assert customers(f, "beta", needs_ratings) == 0       # advertising alone never produced a trial
        assert customers(f, "beta") <= 0.06 * len(needs_ratings)
        rate(f, "beta", need, 5)                              # reputation arrives: four paying customers rate it
        run(f, 24)
        gate[arm] = customers(f, "beta", needs_ratings)
        assert healthy(f)
    assert gate["control"] > 10 and gate["ads"] > 1.4 * gate["control"], gate


def test_a_seeded_neighbourhood_cascades_under_contagion_but_not_under_legacy():
    """Intervention: 5% of consumers in one neighbourhood of the social graph are made customers.
    contagion: every later adoption happens next to an existing customer and the seed multiplies; without the
    seed nothing happens at all. legacy: people find the offering through search whatever their neighbours do."""
    out = {}
    for model, params in (("contagion", {"seed_rate": 0.0, "threshold": [0.1, 0.3]}), ("legacy", {})):
        w = make({"behavior": {"model": model, "params": params}}, [firm("acme")], n=1000)
        run(w, 2)
        blob = w.snapshot_blob()
        for arm in ("control", "seeded"):
            f = w.fork(f"{model}-{arm}")
            seeded = seed_adopters(f, "acme", range(300, 350)) if arm == "seeded" else 0
            nbr = f.state.arrays["c_nbr"]
            lonely = new = 0
            for _ in range(48):
                before = f.state.arrays["c_prov"].copy()
                run(f, 1)
                took = np.flatnonzero((before < 0) & (f.state.arrays["c_prov"] >= 0))
                new += len(took)
                lonely += int(((before[nbr[took]] >= 0).sum(axis=1) == 0).sum())
            out[model, arm] = {"seeded": seeded, "customers": customers(f, "acme"), "new": new, "lonely": lonely}
            assert healthy(f) and replays(f, blob)
    con, leg = out["contagion", "seeded"], out["legacy", "seeded"]
    assert con["seeded"] == leg["seeded"] == 50
    assert out["contagion", "control"]["customers"] == 0 and out["contagion", "control"]["new"] == 0
    assert con["customers"] >= 5 * con["seeded"]                       # a cascade
    assert con["new"] > 300 and con["lonely"] == 0                     # always next to an existing customer
    extra = leg["customers"] - out["legacy", "control"]["customers"]
    assert extra <= leg["seeded"]                                      # no multiplier
    assert out["legacy", "control"]["lonely"] >= 0.05 * out["legacy", "control"]["new"] > 10


def test_contagion_needs_reinforcement_and_abandonment_spreads():
    """Two more interventions on the contagion world. (1) With every threshold above one neighbour in six, twelve
    customers of whom nobody sees two persuade no one, while the same number placed in adjacent pairs start a
    cascade. (2) When the service goes down, customers leave faster if their neighbours' defections count
    (`quit_spread`) than if only their own experience does."""
    w = make({"behavior": {"model": "contagion", "params": {"seed_rate": 0.0, "threshold": [0.2, 0.3]}}},
             [firm("acme")], n=1000)
    run(w, 2)
    nbr, wtp = w.state.arrays["c_nbr"], w.state.arrays["c_wtp"]
    watched = np.zeros(1000, int)
    lone = []
    for i in np.flatnonzero(wtp > 4.0):                # consumers who value the service, no two seen by one person
        watchers = np.flatnonzero((nbr == i).any(axis=1))
        if len(lone) < 12 and not watched[watchers].any():
            lone.append(int(i))
            watched[watchers] += 1
    paired = [j for i in lone[:6] for j in (i, (i + 1) % 1000)]
    alone, pairs = w.fork("lone"), w.fork("paired")
    assert seed_adopters(alone, "acme", lone) == 12 and seed_adopters(pairs, "acme", paired) == 12
    run(alone, 24)
    run(pairs, 24)
    assert customers(alone, "acme") <= 12 and customers(pairs, "acme") > 100
    w = make({"behavior": {"model": "contagion", "params": {"seed_rate": 0.0, "threshold": [0.1, 0.3]}}},
             [firm("acme")], n=1000)
    run(w, 2)
    seed_adopters(w, "acme", range(300, 350))
    run(w, 24)
    before = customers(w, "acme")
    assert before > 500
    social, own_only = w.fork("social"), w.fork("own-experience-only")
    own_only.modules["consumers"].segments[0].params["quit_spread"] = 0.0     # ablation on this fork only
    left = {}
    for name, f in (("social", social), ("own_only", own_only)):
        f.system_call("cloud.undeploy", {"host": "acme"})        # an outage: every request now fails
        run(f, 8)
        left[name] = 1.0 - customers(f, "acme") / before
    assert left["social"] > left["own_only"] + 0.1 and left["own_only"] > 0.05, left


def test_a_price_cut_moves_share_fast_under_bargain_and_slowly_under_reputation_gate():
    """Intervention: one of two equal firms cuts its price by a quarter."""
    moved = {}
    for model, days in (("bargain", 3), ("reputation_gate", 6)):
        w = make({"behavior": {"model": model}, "review_rate": 0.002},
                 [firm("alfa", price=4), firm("beta", price=4)])
        run(w, days * 24)
        assert min(customers(w, "alfa"), customers(w, "beta")) > 40
        ctl, cut = w.fork(f"{model}-control"), w.fork(f"{model}-cut")
        set_price(cut, "beta", 3)
        row = {}
        for h in (6, 24):
            for f in (ctl, cut):
                f.advance_time(w.now + h * HOUR)
            row[h] = share(cut, "beta") - share(ctl, "beta")
        moved[model] = row
        assert healthy(cut) and healthy(ctl)
    assert moved["bargain"][6] > 0.3 and moved["bargain"][24] > 0.3, moved
    assert abs(moved["reputation_gate"][6]) < 0.05 and abs(moved["reputation_gate"][24]) < 0.08, moved


def test_explorers_find_an_unrated_better_service_without_advertising_and_ignore_its_ratings():
    """A better service is launched with no ratings and no advertising. explorer: customers find it by trying it
    and stay because of what they experience; public ratings (intervention: four paying customers give it one
    star) change nothing. reputation_gate: the same bad ratings keep it at zero customers."""
    need = "route.shortest"                       # incumbent: hop-count paths (quality ~0.6); entrant: Dijkstra (1.0)
    res = {}
    for model in ("explorer", "reputation_gate"):
        w = entrant_world(model, need, tiers=(0, 1))
        ctl, bad = w.fork(f"{model}-control"), w.fork(f"{model}-bombed")
        rate(bad, "beta", need, 1)
        for f in (ctl, bad):
            run(f, 72 if model == "explorer" else 120)
            assert f.state.count("ads_campaigns") == 0 and healthy(f)
        res[model] = (ctl, bad)
    ctl, bad = res["explorer"]
    assert share(ctl, "beta") > 0.6 and customers(ctl, "beta") > 300
    assert abs(customers(bad, "beta") - customers(ctl, "beta")) < 0.1 * customers(ctl, "beta")
    # the choice rests on private memory: consumers who remember both services value the better one more
    st = ctl.state
    ms, mv = st.arrays["c_mslot"], st.arrays["c_mval"]
    a, b = slot_of(ctl, "alfa"), slot_of(ctl, "beta")
    both = (ms == a).any(axis=1) & (ms == b).any(axis=1)
    va = np.where(ms == a, mv, 0).sum(axis=1)[both]
    vb = np.where(ms == b, mv, 0).sum(axis=1)[both]
    assert both.sum() > 200 and float((vb > va).mean()) > 0.9
    assert (ms[st.arrays["c_prov"] >= 0, 0] == st.arrays["c_prov"][st.arrays["c_prov"] >= 0]).mean() > 0.95
    ctl, bad = res["reputation_gate"]
    assert customers(bad, "beta") == 0 and customers(ctl, "beta") > 100


def test_an_agent_service_wins_explorers_on_quality_alone():
    """The same claim through the ordinary participant path: an agent registers a company, deploys its own
    (sandboxed) implementation and lists it. No advertising, no ratings at launch."""
    from crucible.goals.metrics import m_independent_revenue
    from crucible.worlds.digital_civilization import service_repo_files
    need = "route.shortest"
    w = make({"behavior": {"model": "explorer"}}, [firm("alfa", need, 0, 3)], n=600, needs=(need,))
    run(w, 48)
    w.system_call("admin.join", {"handle": "ada", "endowment": 2_000_000})

    def ex(tool, **args):
        r = w.execute("p:ada", {"tool": tool, "args": args})
        assert r.ok, (tool, r.error)
        return r.result
    me = w.modules["bank"].primary_account(w.state, "p:ada")
    ex("id.create_org", handle="adaco", name="Ada Co")
    ex("biz.register", org="o:adaco", account=me)
    acct = ex("bank.open_account", owner="o:adaco")["account"]
    ex("bank.transfer", from_account=me, to=acct, amount=1_500_000)
    repo = ex("code.create_repo", name="svc", owner="o:adaco")["repo"]
    ex("code.commit", repo=repo, files=service_repo_files(need, 1, 3))
    inst = ex("cloud.rent", provider="o:cloud1", cores=2, account=acct, owner="o:adaco")["instance"]
    assert ex("cloud.deploy", repo=repo, host="adasvc", instance=inst)["status"] == "live"
    ex("market.list", kind="api", title="Ada routing", price_cents=3, seller="o:adaco", host="adasvc",
       endpoint="solve", need=need, description="shortest paths")
    assert not w.modules["cloud"].live_deployment(w.state, "adasvc")["trusted"]
    run(w, 72)
    assert share(w, "adasvc") > 0.55 and customers(w, "adasvc") > 250
    assert m_independent_revenue(w, "o:adaco", 3 * DAY) > 10_000
    assert w.state.count("ads_campaigns") == 0 and healthy(w) and replays(w)


# ====================================================================== segments
def test_two_segments_each_follow_their_own_rule():
    segs = [{"name": "hunters", "share": 0.5, "model": "bargain", "params": {"rate_per_day": [40.0, 50.0],
                                                                             "loyalty": [0.0, 0.1]}},
            {"name": "cautious", "share": 0.5, "model": "reputation_gate", "params": {"wtp_mult": [2.0, 3.0]}}]
    w = make({"segments": segs, "review_rate": 0.002}, [firm("alfa", price=4), firm("beta", price=4)])
    st = w.state
    seg = st.arrays["c_seg"]
    assert np.bincount(seg).tolist() == [400, 400]
    rate_, loyal, wtp = st.arrays["c_rate"], st.arrays["c_loyal"], st.arrays["c_wtp"]
    assert 40 <= rate_[seg == 0].min() and rate_[seg == 0].max() <= 50 and rate_[seg == 1].max() <= 30
    assert loyal[seg == 0].max() <= 0.1 < loyal[seg == 1].max()
    base = NEEDS["match.stable"].base_price_cents
    assert wtp[seg == 1].min() >= 2.0 * base - 1e-3 and wtp[seg == 0].min() < 1.0 * base
    # model traits exist only for the segment that uses them
    assert (st.arrays["c_trust"][seg == 1] > 0).all() and (st.arrays["c_trust"][seg == 0] == 0).all()
    run(w, 6 * 24)
    agg = {a["segment"]: a for a in segment_aggregates(w)}
    assert agg["hunters"]["model"] == "bargain" and agg["cautious"]["model"] == "reputation_gate"
    assert agg["hunters"]["n"] == agg["cautious"]["n"] == 400
    assert agg["hunters"]["served"] > 300 and agg["cautious"]["served"] > 150
    assert sum(agg["hunters"]["customers"].values()) == agg["hunters"]["served"]
    assert segment_aggregates(w, "route.tsp")[0]["n"] == 0
    ctl, cut = w.fork("control"), w.fork("cut")
    set_price(cut, "beta", 3)
    for f in (ctl, cut):
        run(f, 24)
    c = {a["segment"]: a["customers"] for a in segment_aggregates(ctl)}
    t = {a["segment"]: a["customers"] for a in segment_aggregates(cut)}
    hunters = sum(t["hunters"].values())
    assert t["hunters"]["beta"] > 0.9 * hunters and c["hunters"]["beta"] < 0.7 * sum(c["hunters"].values())
    assert abs(t["cautious"].get("beta", 0) - c["cautious"].get("beta", 0)) < 0.08 * 400
    assert healthy(cut) and replays(w)


def test_budget_shares_are_spending_envelopes_of_the_one_household_budget():
    """Two otherwise identical segments under a household budget that binds. Giving one a 5% envelope shifts
    served requests away from it; no envelope ever spends more than its part of the balance, and the household
    account never goes negative."""
    frac = {}
    for bs in (0.5, 0.05):
        segs = [{"name": "poor", "share": 0.5, "model": "legacy", "params": {"budget_share": bs}},
                {"name": "rest", "share": 0.5, "model": "legacy"}]
        w = make({"segments": segs, "income_per_day_cents": 12_000}, [firm("alfa", price=3)])
        low = 10 ** 12
        for _ in range(96):
            run(w, 1)
            low = min(low, w.state.tables["accounts"]["acct_households"]["balance"])
        assert 0 <= low < 300                                   # the budget binds, and is never overdrawn
        evs = [e.payload for e in w.log if e.type == "consumers.hour"]
        parts = [0, 0]
        for p in evs:
            pools = p["pools"]
            assert abs(pools[0] - bs * sum(pools)) <= 1          # the envelope is its share of the balance
            spent = [0, 0]
            for k, sm in p["slots"].items():
                price = p["public"][k]["price"]
                assert sum(sm["parts"]) == sm["requests"] and sm["revenue"] <= sm["requests"] * price
                for e in (0, 1):
                    spent[e] += sm["parts"][e] * price
                    parts[e] += sm["parts"][e]
            assert spent[0] <= pools[0] and spent[1] <= pools[1]
        frac[bs] = parts[0] / sum(parts)
        assert healthy(w)
    assert 0.45 < frac[0.5] < 0.55 and frac[0.05] < frac[0.5] - 0.08, frac
    # a single population can be given an envelope too: at most a quarter of the balance in any hour
    w = make({"behavior": {"model": "legacy", "params": {"budget_share": 0.25}}, "income_per_day_cents": 12_000},
             [firm("alfa", price=3)], n=300)
    for _ in range(30):
        hh = w.state.tables["accounts"]["acct_households"]["balance"]
        run(w, 1)
        p = [e.payload for e in w.log if e.type == "consumers.hour"][-1]
        assert p["pools"] == [hh // 4] or w.now % DAY == 0          # (income lands just before the daily hour)
        assert sum(sm["requests"] * 3 for sm in p["slots"].values()) <= p["pools"][0]
    # the charging rule itself: exact, proportional, never beyond what an envelope reserved
    rng = np.random.default_rng(0)
    for _ in range(300):
        price = int(rng.integers(1, 9))
        parts = [int(x) for x in rng.integers(0, 40, int(rng.integers(1, 5)))]
        R = sum(parts)
        if R == 0:
            continue
        cents = int(rng.integers(0, R + 1)) * price
        pools = [r * price + int(rng.integers(0, 5)) for r in parts]
        before = list(pools)
        _charge(pools, parts, R, cents, price)
        assert sum(before) - sum(pools) == cents and min(pools) >= 0
        assert all(b - a <= r * price for b, a, r in zip(before, pools, parts))


# ====================================================================== latent needs and derived demand
SRC, DST = "schedule.jobs", "match.stable"
EDGE = {"src": SRC, "dst": DST, "weight": 0.8, "scale": 2000.0}


def demand_world(lag=1, edges=True, firms=("sched", "match"), **cons):
    cfg = {"latent": {DST: 0.1}, "activation_rate": 0.3, **cons}
    if edges:
        cfg["derived"] = [{**EDGE, "lag_days": lag}]
    all_firms = {"sched": firm("sched", SRC, 1, 2), "match": firm("match", DST, 1, 3)}
    return make(cfg, [all_firms[f] for f in firms], needs=(SRC, DST))


def _verified_by_day(w, need) -> dict:
    """Verified successful volume per day, recomputed from the hourly demand events: sum of ok x mean score."""
    names = w.state.meta_get("cons_needs")
    out: dict[int, float] = {}
    for e in w.log:
        if e.type == "consumers.hour":
            for k, sm in e.payload["slots"].items():
                if names[e.payload["public"][k]["need"]] == need:
                    out[e.payload["t"] // DAY] = out.get(e.payload["t"] // DAY, 0.0) + sm["ok"] * sm["q"]
    return out


def test_serving_one_need_creates_demand_for_another_with_the_configured_lag():
    w = demand_world(lag=1)
    st = w.state
    need, active = st.arrays["c_need"], st.arrays["c_active"]
    dst = need == st.meta_get("cons_needs").index(DST)
    n_dst = int(dst.sum())
    assert int(active[dst].sum()) == round(0.1 * n_dst) and active[~dst].all()        # latent at genesis
    assert demand_state(w)["needs"][DST]["target"] == 0.1
    on = [int(active[dst].sum())]
    for _ in range(7):
        run(w, 24)
        active = st.arrays["c_active"]
        on.append(int(active[dst].sum()))
        assert (st.arrays["c_prov"][~active] < 0).all()                               # no need, no provider
    vol = _verified_by_day(w, SRC)
    ticks = [e.payload for e in w.log if e.type == "consumers.demand"]
    assert [p["day"] for p in ticks] == list(range(1, 8))
    for p, before, after in zip(ticks, on, on[1:]):
        (inp,) = p["inputs"]
        assert inp["src"] == SRC and inp["day"] == p["day"] - 2                       # last complete day, one day earlier
        assert inp["verified"] == pytest.approx(vol.get(inp["day"], 0.0), abs=0.01)
        target = min(1.0, 0.1 + 0.8 * (1.0 - math.exp(-inp["verified"] / 2000.0)))
        gap = target * n_dst - before
        step = int(round(0.3 * gap)) or (int(math.copysign(1, gap)) if abs(gap) >= 1 else 0)
        assert after == before + step
    assert ticks[0]["inputs"][0]["verified"] == 0 and on[1] == on[0]                  # nothing to react to yet
    assert vol[0] > 500 and on[2] > on[1] + 20                                        # day 0's volume arrives on day 2
    assert on[-1] > 0.6 * n_dst
    ds = demand_state(w)
    d = ds["needs"][DST]
    assert d["active"] == on[-1] and d["n"] == n_dst and d["base"] == 0.1 and d["target"] > 0.8
    assert d["inputs"][0]["verified"] == ticks[-1]["inputs"][0]["verified"] and ds["edges"][0]["lag_days"] == 1
    assert ds["needs"][SRC]["active_fraction"] == 1.0 and ds["needs"][SRC]["target"] == 1.0
    assert customers(w, "match") > 4 * round(0.1 * n_dst)                             # the new demand is served
    assert healthy(w) and replays(w)
    # a shorter lag acts a day earlier on the same history
    w0 = demand_world(lag=0)
    run(w0, 48)
    t0 = [e.payload for e in w0.log if e.type == "consumers.demand"]
    assert t0[0]["inputs"][0]["day"] == 0 and t0[0]["inputs"][0]["verified"] == pytest.approx(vol[0], abs=0.01)
    assert int(w0.state.arrays["c_active"][dst].sum()) > on[2]


def test_derived_demand_decays_when_the_source_service_stops():
    w = demand_world(lag=0)
    run(w, 6 * 24)
    ctl, cut = w.fork("control"), w.fork("stopped")
    cut.system_call("cloud.undeploy", {"host": "sched"})
    rows = {"control": [], "stopped": []}
    for _ in range(5):
        for name, f in (("control", ctl), ("stopped", cut)):
            run(f, 24)
            d = demand_state(f)["needs"][DST]
            rows[name].append((d["active"], d["target"], customers(f, "match")))
    stopped, control = rows["stopped"], rows["control"]
    assert all(t == 0.1 for _, t, _ in stopped[1:]) and all(t > 0.8 for _, t, _ in control)
    assert all(a > b for (a, _, _), (b, _, _) in zip(stopped, stopped[1:]))            # relaxes back down, day by day
    assert stopped[-1][0] < 0.5 * control[-1][0] and stopped[-1][2] < 0.5 * control[-1][2]
    assert control[-1][0] >= control[0][0]
    for f in (ctl, cut):
        active = f.state.arrays["c_active"]
        assert (f.state.arrays["c_prov"][~active] < 0).all() and f.check_invariants() == []


def test_without_edges_or_without_the_source_service_latent_demand_stays_at_its_base():
    for kw in ({"edges": False}, {"firms": ("match",)}):
        w = demand_world(**kw)
        dst = w.state.arrays["c_need"] == w.state.meta_get("cons_needs").index(DST)
        base = round(0.1 * int(dst.sum()))
        for _ in range(4):
            run(w, 24)
            assert int(w.state.arrays["c_active"][dst].sum()) == base
        assert demand_state(w)["needs"][DST]["target"] == 0.1 and healthy(w)
        assert 0 < customers(w, "match") <= base


def test_activation_creates_no_income_and_the_household_budget_is_never_overdrawn():
    """Income is a constant of the configuration. When derived demand grows the population that spends, the
    same balance is shared out: here the budget binds, and spending never exceeds what was paid in."""
    w = demand_world(lag=0, income_per_day_cents=9_000)
    low = 10 ** 12
    for _ in range(6 * 24):
        run(w, 1)
        low = min(low, w.state.tables["accounts"]["acct_households"]["balance"])
    assert 0 <= low < 100                                               # it binds, and is never negative
    ctl, cut = w.fork("control"), w.fork("stopped")
    cut.system_call("cloud.undeploy", {"host": "sched"})
    for f in (ctl, cut):
        for _ in range(3 * 24):
            run(f, 1)
            assert f.state.tables["accounts"]["acct_households"]["balance"] >= 0

    def money(f):
        inc = sum(d for _, t in f.state.scan("txns") if t["kind"] == "income"
                  for a, d in t["postings"] if a == "acct_households")
        out = -sum(d for _, t in f.state.scan("txns") if t["kind"] == "sale"
                   for a, d in t["postings"] if a == "acct_households")
        return inc, out
    (inc_c, out_c), (inc_s, out_s) = money(ctl), money(cut)
    assert inc_c == inc_s == 9_000 * 10                                 # day 0 and nine daily payments
    assert out_c <= inc_c and out_s <= inc_s
    n_c = int(ctl.state.arrays["c_active"].sum())
    n_s = int(cut.state.arrays["c_active"].sum())
    assert n_c > n_s + 50                                               # very different active populations
    assert healthy(ctl) and cut.check_invariants() == []


def test_agents_see_only_the_consequences_of_latent_demand():
    w = demand_world(lag=0)
    w.system_call("admin.join", {"handle": "ada", "endowment": 1_000_000})
    acct = w.modules["bank"].primary_account(w.state, "p:ada")
    legacy = make({}, [firm("match", DST, 1, 3)], needs=(SRC, DST))
    legacy.system_call("admin.join", {"handle": "ada", "endowment": 1_000_000})
    ref = legacy.execute("p:ada", {"tool": "research.survey", "args": {
        "need": DST, "account": legacy.modules["bank"].primary_account(legacy.state, "p:ada")}})
    assert ref.ok, ref.error
    est = []
    for day in (0, 3, 6):
        w.advance_time(day * DAY + HOUR)
        r = w.execute("p:ada", {"tool": "research.survey", "args": {"need": DST, "account": acct}})
        assert r.ok and set(r.result) == set(ref.result)               # the same report as in 3.0
        d = demand_state(w)["needs"][DST]
        assert 0.85 * d["active"] - 1 <= r.result["consumers_est"] <= 1.15 * d["active"] + 1
        est.append(r.result["consumers_est"])
    assert est[-1] > 3 * est[0]                                         # the growth is visible, its cause is not
    # nothing an agent can call or see mentions the hidden structure
    hidden = ("latent", "derived", "target", "segment", "activation", "c_active", "lag_days")
    tools = w.available_tools("p:ada")
    assert not [t["name"] for t in tools if t["name"].startswith("research.") and t["name"] != "research.survey"]
    assert not any(h in str(r.result) for h in hidden)
    assert "consumers" not in w.observe("p:ada")
    evs, _ = w.visible_events("p:ada", 0, limit=10 ** 6)
    assert not [e for e in evs if e.type.startswith("consumers.")]
    assert all(e.vis == "internal" for e in w.log if e.type in ("consumers.demand", "consumers.hour",
                                                                "consumers.genesis"))


# ====================================================================== determinism: every model, segments, demand
DETERMINISM = {
    "legacy": {},
    "contagion": {"behavior": {"model": "contagion", "params": {"seed_rate": 0.5, "threshold": [0.1, 0.4]}}},
    "reputation_gate": {"behavior": {"model": "reputation_gate", "params": {"pioneer_share": 0.3, "min_ratings": 1}},
                        "review_rate": 0.01},
    "explorer": {"behavior": {"model": "explorer"}},
    "bargain": {"behavior": {"model": "bargain"}},
    "five_segments": {"segments": [
        {"name": "a", "share": 0.3, "model": "legacy"},
        {"name": "b", "share": 0.2, "model": "contagion", "params": {"seed_rate": 0.5, "budget_share": 0.3}},
        {"name": "c", "share": 0.2, "model": "reputation_gate", "params": {"pioneer_share": 0.3, "min_ratings": 1}},
        {"name": "d", "share": 0.15, "model": "explorer", "params": {"rate_per_day": [20.0, 40.0]}},
        {"name": "e", "share": 0.15, "model": "bargain", "params": {"budget_share": 0.2}}], "review_rate": 0.01},
    "derived_demand": {"latent": {DST: 0.2}, "derived": [{**EDGE, "scale": 300.0, "lag_days": 0}],
                       "activation_rate": 0.5},
    "derived_demand_segments": {"latent": {DST: 0.2}, "derived": [{**EDGE, "scale": 300.0, "lag_days": 0},
                                                                  {"src": DST, "dst": SRC, "weight": -0.3,
                                                                   "scale": 200.0, "lag_days": 0}],
                                "activation_rate": 0.5, "decay_rate": 0.25,
                                "segments": [{"name": "x", "share": 0.5, "model": "explorer"},
                                             {"name": "y", "share": 0.5, "model": "bargain"}]},
}


def _history(cons):
    """Two days of a small economy with a launch, an advertising campaign and a price cut (all recorded calls)."""
    w = make(cons, [firm("sched", SRC, 1, 2), firm("alfa", DST, 1, 4), firm("beta", DST, 1, 4)], n=400,
             needs=(SRC, DST))
    shelve(w, "beta")
    run(w, 14)
    launch(w, "beta", DST)
    advertise(w, "beta", DST)
    run(w, 16)
    set_price(w, "beta", 3)
    run(w, 18)
    return w


@pytest.mark.parametrize("name", sorted(DETERMINISM))
def test_replay_regeneration_and_forks_are_exact(name):
    w = _history(DETERMINISM[name])
    st = w.state
    assert healthy(w)
    assert customers(w, "sched") > 20 and customers(w, "alfa") + customers(w, "beta") > 10    # not an idle world
    events = [e.to_dict() for e in w.log]
    assert World.replay(w.spec, events).digest() == w.digest()
    g = World.regenerate(w.spec, events)
    assert (g.seq, g.last_hash, g.digest()) == (w.seq, w.last_hash, w.digest())
    a, b = w.fork("a"), w.fork("b")
    for f in (w, a, b):
        run(f, 12)
    assert a.digest() == b.digest() == w.digest() and a.last_hash == w.last_hash
    set_price(b, "alfa", 2)
    run(a, 6)
    run(b, 6)
    assert a.digest() != b.digest() and healthy(a) and healthy(b)
    if "c_active" in st.arrays:
        assert [e for e in w.log if e.type == "consumers.demand"] and len(set(st.arrays["c_active"])) == 2
    if "c_seg" in st.arrays:
        assert all(x["served"] > 0 for x in segment_aggregates(w))


def test_replay_from_stored_events_is_exact_with_ten_or_more_needs():
    """A store returns event payloads with keys sorted as strings ("10" before "2"). The population update visits
    the needs' search results in need order whatever the key order, so crash recovery is exact in worlds with
    many need formats (in 3.0, which never had ten, the visiting order followed the dictionary)."""
    import dataclasses
    import json
    builtin = ["forecast.series", "match.stable", "pack.knapsack", "route.shortest", "route.tsp", "schedule.jobs",
               "sort.records", "text.extract"]
    extra = {f"zz.{c}": dataclasses.replace(NEEDS["schedule.jobs"], name=f"zz.{c}") for c in "abc"}
    NEEDS.update(extra)
    try:
        w = make({}, [firm("pack", "pack.knapsack", 1, 4), firm("zed", "zz.c", 1, 2)], needs=builtin + sorted(extra))
        run(w, 30)
        assert len(w.state.meta_get("cons_needs")) == 11 and customers(w, "pack") > 5 and customers(w, "zed") > 2
        seen = [e.payload["seen"] for e in w.log if e.type == "consumers.hour"]
        assert any(p.get("10") and p.get("2") for p in seen)            # both orders matter in the same hour
        stored = [json.loads(json.dumps(e.to_dict(), sort_keys=True)) for e in w.log]
        assert World.replay(w.spec, stored).digest() == w.digest()
        assert healthy(w)
    finally:
        for k in extra:
            NEEDS.pop(k)


def test_household_income_ignores_need_formats_registered_by_other_worlds():
    """Generated need categories (4.0) are registered process-wide under `gen.*` names. The default household
    income of a world that does not use them is the 3.0 value whatever else has been registered."""
    import dataclasses
    spec = generate({"family": "tiny", "overrides": {"n_consumers": 300}}, 1)
    before = World.create(spec).modules["consumers"].daily_income()
    NEEDS["gen.test.0001"] = dataclasses.replace(NEEDS["route.tsp"], name="gen.test.0001", base_price_cents=500.0)
    try:
        assert World.create(spec).modules["consumers"].daily_income() == before
        own = make({}, [], n=300, needs=("schedule.jobs", "gen.test.0001"))
        assert own.modules["consumers"].daily_income() > 20 * before     # priced on its own (expensive) needs
    finally:
        NEEDS.pop("gen.test.0001")


# ====================================================================== robustness in a standard family
ROBUST = {m: {"behavior": {"model": m}} for m in MODELS if m != "legacy"}
ROBUST["five_segments_latent"] = {
    "segments": [{"name": m, "share": 1, "model": m} for m in MODELS],
    "latent": {"schedule.jobs": 0.3, "match.stable": 0.5},
    "derived": [{"src": "route.shortest", "dst": "schedule.jobs", "weight": 0.6, "scale": 500.0, "lag_days": 0},
                {"src": "route.shortest", "dst": "match.stable", "weight": -0.4, "scale": 500.0, "lag_days": 1}]}


@pytest.mark.parametrize("name", sorted(ROBUST))
def test_every_model_survives_shocks_a_new_need_outages_and_a_closing_offering(name):
    """The stock `tiny` family (self-repricing incumbents, an unreliable provider) with population shocks, a need
    that appears mid-world and the highest-slot offering closing: no process fails, invariants hold, replay is
    exact, and nobody is left holding an offering that no longer exists."""
    shocks = [{"day": 1, "kind": "demand_surge", "need": "schedule.jobs", "mult": 1.5},
              {"day": 1, "kind": "pref_drift", "need": "route.shortest", "toward": [0.7, 0.1, 0.2], "strength": 0.4},
              {"day": 2, "kind": "new_need", "need": "match.stable", "fraction": 0.2},
              {"day": 2, "kind": "reliability", "reliability": 0.6}]
    ov = {"n_consumers": 400, "needs": ["route.shortest", "schedule.jobs"], "firms_per_need": 2,
          "params": {"cost_model": "deterministic"},
          "module_config": {"consumers": {"k_samples": 1, **ROBUST[name]}, "shocks": {"events": shocks}}}
    w = World.create(generate({"family": "tiny", "overrides": ov}, 2), strict=True)
    st = w.state
    run(w, 60)
    top, rec = max(st.scan("cons_offers"), key=lambda kv: kv[1]["slot"])
    owner = next(f["owner"] for _, f in st.scan("npc_firms") if f["listing"] == top)
    w.system_call("market.update", {"listing": top, "status": "closed"}, as_principal=owner)
    run(w, 36)
    assert healthy(w) and replays(w)
    names = st.meta_get("cons_needs")
    assert int((st.arrays["c_need"] == names.index("match.stable")).sum()) > 40
    assert not (st.arrays["c_prov"] == rec["slot"]).any()
    assert int((st.arrays["c_prov"] >= 0).sum()) > 0
    if "latent" in ROBUST[name]:
        active = st.arrays["c_active"]
        assert (st.arrays["c_prov"][~active] < 0).all()
        d = demand_state(w)["needs"]
        assert d["route.shortest"]["active_fraction"] == 1.0
        assert 0.3 <= d["schedule.jobs"]["active_fraction"] <= 0.9          # lifted from its base by the edge
        assert d["match.stable"]["active_fraction"] < 1.0                    # a substitute: pushed below its base
        assert d["match.stable"]["target"] < 0.5
        assert sorted(a["segment"] for a in segment_aggregates(w)) == sorted(MODELS)


if __name__ == "__main__" and "--record" in sys.argv:
    print("# recorded from", os.path.dirname(os.path.dirname(crucible.__file__)))
    print("LEGACY_DIGESTS = {")
    for _name in sorted(LEGACY_SCENARIOS):
        _f = _fingerprint(LEGACY_SCENARIOS[_name]())
        print(f"    {_name!r}: {{")
        for _k in ("consumers", "digest", "last_event_hash", "events"):
            print(f"        {_k!r}: {_f[_k]!r},")
        print("    },")
    print("}")
