"""OpenWorld families (worlds whose valuable work is generated categories) and the end-to-end proof that an
artifact built from public documentation and published examples alone earns its living only if it works.

The end-to-end tests run agent code in the sandbox (Landlock/seccomp required).
"""
import json
import sys
import time

import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy.needs import NEEDS, compile_solver, tier_quality  # noqa: E402
from crucible.usefulness import GENERATED, category_families, unregister_generated  # noqa: E402
from crucible.usefulness.agents import ExampleInductor, GarbageSeller, Shopper, service_repo  # noqa: E402
from crucible.usefulness.module import ledger, usefulness_report  # noqa: E402
from crucible.worldgen import generate, generator  # noqa: E402

DAY = 86400
SPLITS = {"ow_market": "train", "ow_frontier": "train", "ow_composite": "val", "ow_heldout": "test", "ow_tiny": "dev"}
_WORLDS: list = []


@pytest.fixture(scope="module", autouse=True)
def _hermetic():
    """Generated needs and OpenWorld families registered by this file do not leak into other test files (unless
    the families were already wired in before this file ran: then they are left as they were)."""
    from crucible.worldgen import generator
    wired = "crucible.worldgen.ow_families" in sys.modules and any(n.startswith("ow_") for n in generator.FAMILIES)
    from crucible.worldgen import ow_families as owf
    before = set(GENERATED)
    owf.register_ow_families()
    yield
    unregister_generated(sorted(set(GENERATED) - before))
    if not wired:
        owf.unregister_ow_families()


@pytest.fixture(autouse=True)
def _close_worlds():
    yield
    while _WORLDS:
        _WORLDS.pop().close()


def _spec(fam, seed, **ov):
    return generate({"family": fam, "overrides": ov}, seed)


def _world(fam, seed, **ov):
    w = World.create(_spec(fam, seed, **ov))
    _WORLDS.append(w)
    return w


def _company(w, handle, endowment=800_000):
    """An agent with a registered company and a funded operating account, through ordinary tools."""
    w.system_call("admin.join", {"handle": handle, "endowment": endowment})
    ex = lambda t, **k: w.execute(f"p:{handle}", {"tool": t, "args": k})  # noqa: E731
    me = w.modules["bank"].primary_account(w.state, f"p:{handle}")
    org = ex("id.create_org", handle=f"{handle}-co", name=handle, kind="company").result["org"]
    assert ex("biz.register", org=org, account=me).ok
    acct = ex("bank.open_account", owner=org).result["account"]
    assert ex("bank.transfer", from_account=me, to=acct, amount=endowment * 3 // 4).ok
    return ex, org, acct


def _launch(w, handle, ex, org, acct, host, source, need, price):
    """Commit, rent, deploy (sandboxed) and list a service."""
    repo = ex("code.create_repo", name=host, owner=org).result["repo"]
    assert ex("code.commit", repo=repo, files=service_repo(source, host, host, price)).ok
    prov = sorted(p["provider"] for p in w.query(f"p:{handle}", "cloud.providers"))[0]
    inst = ex("cloud.rent", provider=prov, cores=1, account=acct, owner=org).result["instance"]
    assert ex("cloud.deploy", repo=repo, host=host, instance=inst).result["status"] == "live"
    assert ex("market.list", kind="api", title=host, price_cents=price, seller=org, host=host, endpoint="solve",
              need=need, description="Keywords: " + " ".join(NEEDS[need].keywords)).ok


def _cfg(spec, module):
    return next(m["config"] for m in spec.modules if m["name"] == module)


def _cats(spec):
    from crucible.worldgen.ow_families import ow_categories
    return ow_categories(spec)


# ====================================================================== structure of the families
def test_ow_families_are_partitioned_and_test_categories_are_held_out_entirely():
    assert {f: generator.FAMILIES[f].split for f in SPLITS} == SPLITS
    held_out, train, val = (set(category_families(s)) for s in ("test", "train", "val"))
    for seed in range(6):
        for fam in ("ow_market", "ow_frontier", "ow_tiny"):
            assert {c.family for c in _cats(_spec(fam, seed))} <= train, fam
        comp = _cats(_spec("ow_composite", seed))
        assert {c.family for c in comp if not c.parts} <= train | val and any(c.family in val for c in comp)
        assert sum(1 for c in comp if c.parts) == 2 and all(c.split != "train" for c in comp if c.parts)
        assert all(p.name in {c.name for c in comp} for c in comp for p in c.parts)      # its halves are there too
        ho = _cats(_spec("ow_heldout", seed))
        assert len(ho) == 3 and {c.family for c in ho} <= held_out and all(c.split == "test" for c in ho)
    for fam, split in SPLITS.items():
        assert _spec(fam, 0).params["family_split"] == split
        mods = [m["name"] for m in _spec(fam, 0).modules]
        assert "use" in mods and ("society" in mods) == (fam != "ow_tiny")              # firm minds, as open_digital


def test_category_rules_are_drawn_per_world():
    a, b = _spec("ow_frontier", 1), _spec("ow_frontier", 2)          # all five training families in both worlds
    ca, cb = {c.family: c for c in _cats(a)}, {c.family: c for c in _cats(b)}
    assert set(ca) == set(cb) == set(category_families("train"))
    for fam in ca:
        assert ca[fam].name != cb[fam].name and ca[fam].rule != cb[fam].rule, fam
    assert _spec("ow_frontier", 1).hash() == a.hash() and {c.name for c in _cats(a)} <= set(NEEDS)
    # the seed is all that is needed: needs are registered already when a spec is generated (goal feasibility)
    firm = _cfg(a, "civ")["firms"][0]
    assert tier_quality(firm["need"], firm["tier"]) < tier_quality(firm["need"], len(NEEDS[firm["need"]].tiers) - 1)


@pytest.mark.parametrize("fam,n_latent", [("ow_frontier", 2), ("ow_heldout", 1)])
def test_latent_and_derived_demand_follow_the_population_contract(fam, n_latent):
    for seed in range(5):
        spec = _spec(fam, seed, n_consumers=1000)
        cons, civ, cats = _cfg(spec, "consumers"), _cfg(spec, "civ"), {c.name: c for c in _cats(spec)}
        latent, derived = cons["latent"], cons["derived"]
        assert len(latent) == n_latent and set(latent) <= set(cats) and all(0.0 < v <= 0.1 for v in latent.values())
        assert derived and all(set(e) == {"src", "dst", "weight", "scale", "lag_days"} for e in derived)
        total = sum(cons["needs"].values())
        for e in derived:
            assert e["src"] in cats and e["dst"] in latent and e["src"] not in latent
            ext = next(x for x in cats[e["src"]].externalities.derived if x.dst_family == cats[e["dst"]].family)
            volume = 1000 * cons["needs"][e["src"]] / total * sum(cons["rate_per_day"]) / 2.0
            assert e["weight"] == ext.weight and e["lag_days"] == ext.lag_days
            assert e["scale"] == pytest.approx(ext.scale * volume, abs=0.06) and e["scale"] > 0
        assert {e["dst"] for e in derived} == set(latent)                # every latent need can be woken
        served = {f["need"] for f in civ["firms"]}
        assert not served & set(latent) and served == set(cats) - set(latent)   # nobody serves a latent need yet
        assert isinstance(cons["income_per_day_cents"], int) and set(cons["needs"]) == set(cats)
    assert "latent" not in _cfg(_spec("ow_market", 0), "consumers")


def test_demand_edges_are_hidden_from_agents():
    w = _world("ow_frontier", 3, n_consumers=300)
    cons = _cfg(w.spec, "consumers")
    a = w.system_call("admin.join", {"handle": "spy", "endowment": 100_000})["principal"]
    seen = [w.execute(a, {"tool": "use.categories", "args": {}}).result, w.observe(a),
            [t["description"] for t in w.available_tools(a) if t["name"].startswith("use.")]]
    for c in sorted(w.modules["use"].cats):
        seen.append(w.execute(a, {"tool": "use.examples", "args": {"category": c, "n": 1}}).result["relation"])
    seen.append([{"type": e.type, "payload": e.payload} for e in w.visible_events(a, 0, limit=10_000)[0]])
    dump = json.dumps(seen)
    assert set(cons["latent"]) <= {c["category"] for c in seen[0]}       # the format of a latent need is public
    for word in ("latent", "derived", "lag_days", "externalit", "weight\":"):
        assert word not in dump, word
    assert not any(str(e["scale"]) in dump for e in cons["derived"])


def test_latent_demand_wakes_only_when_its_source_category_is_served_and_more_when_served_well():
    """Three arms from one world (forks are controlled counterfactuals): nobody serves the source category; the
    generated incumbents serve it at low tiers; an entrant also serves it with the best reference implementation.
    Uses the population module's evaluator-side view of latent demand."""
    consumers = pytest.importorskip("crucible.economy.consumers")
    if not hasattr(consumers, "demand_state"):
        pytest.skip("latent / derived demand is not implemented in economy.consumers yet")
    w = _world("ow_frontier", 1, n_consumers=600)
    cons = _cfg(w.spec, "consumers")
    edge = min(cons["derived"], key=lambda e: (e["lag_days"], e["src"], e["dst"]))
    src, dst, base = edge["src"], edge["dst"], cons["latent"][edge["dst"]]
    assert abs(consumers.demand_state(w)["needs"][dst]["active_fraction"] - base) < 0.02     # asleep at genesis
    closed, strong = w.fork("closed"), w.fork("strong")
    _WORLDS.extend([closed, strong])
    feeders = {e["src"] for e in cons["derived"] if e["dst"] == dst}
    for lid, l in closed.state.scan("listings"):
        if l["need"] in feeders:
            owner = closed.modules["identity"].org_owner_persons(closed.state, l["seller"])[0]
            closed.system_call("market.update", {"listing": lid, "status": "closed"}, as_principal=owner)
    ex, org, acct = _company(strong, "ace", endowment=2_000_000)
    cheapest = min(l["price"] for _, l in strong.state.scan("listings") if l["need"] == src)
    _launch(strong, "ace", ex, org, acct, "ace-best", NEEDS[src].tiers[-1][1], src, max(1, cheapest))
    days = edge["lag_days"] + 7
    for x in (closed, w, strong):
        x.advance_time(x.now + days * DAY)
        assert x.check_invariants() == [] and not [e for e in x.log if e.type == "proc.failed"]
    s0, s1, s2 = (consumers.demand_state(x)["needs"][dst] for x in (closed, w, strong))
    assert s0["target"] == pytest.approx(base) and s0["active_fraction"] <= base + 0.01     # unserved: still asleep
    assert s1["target"] > base + 0.05 and s1["active_fraction"] > 2 * base                  # served: waking up
    assert s2["target"] > s1["target"] + 0.02 and s2["active_fraction"] >= s1["active_fraction"]   # served well: more
    q_inc = max(ledger(w, l["host"], days * DAY)["verified_quality"] for _, l in w.state.scan("listings") if l["need"] == src)
    assert ledger(strong, "ace-best", days * DAY)["verified_quality"] > q_inc + 0.1
    print(f"\nlatent {dst}: base {base}; after {days} days active fraction / target: unserved "
          f"{s0['active_fraction']:.3f}/{s0['target']:.3f}, incumbents {s1['active_fraction']:.3f}/{s1['target']:.3f}, "
          f"plus best-tier entrant {s2['active_fraction']:.3f}/{s2['target']:.3f}")


# ====================================================================== every family builds and runs
@pytest.mark.parametrize("fam", list(SPLITS))
@pytest.mark.parametrize("seed", [1, 2])
def test_every_ow_family_runs_three_days_with_incumbents_serving_verified_quality(fam, seed):
    w = _world(fam, seed, n_consumers=700)
    t0 = time.perf_counter()
    w.advance_time(3 * DAY)
    rate = 3 / (time.perf_counter() - t0)
    assert not [e.payload for e in w.log if e.type == "proc.failed"]
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []
    assert sum(a["balance"] for a in w.state.tables["accounts"].values()) == 0          # money is conserved
    firms = _cfg(w.spec, "civ")["firms"]
    latent = set(_cfg(w.spec, "consumers").get("latent", {}))
    assert firms and {f["need"] for f in firms} == set(w.modules["use"].cats) - latent
    qs = {}
    for f in firms:
        led = ledger(w, f["host"], 3 * DAY)
        expect = tier_quality(f["need"], f["tier"], n=4)
        assert led["ok"] > 0 and led["customers"] > 0 and led["revenue_cents"] > 0, (f["host"], led)
        assert 0.02 < led["verified_quality"] <= 1.0 and abs(led["verified_quality"] - expect) < 0.25, \
            (f["host"], f["need"], f["tier"], led["verified_quality"], expect)
        assert f["tier"] < len(NEEDS[f["need"]].tiers) - 1                              # low tiers: room to do better
        qs[f["host"]] = round(led["verified_quality"], 3)
    assert not [l for _, l in w.state.scan("listings") if l["need"] in latent]
    print(f"\n{fam} seed {seed}: {rate:.2f} simulated days/s (700 consumers, {len(firms)} firms); verified quality {qs}")


def test_open_ow_world_replays_regenerates_and_forks_exactly():
    """Generated categories, firm minds, latent demand and the usefulness ledger together stay deterministic."""
    w = _world("ow_frontier", 3, n_consumers=500)
    w.advance_time(2 * DAY)
    spec, evs = w.spec, [e.to_dict() for e in w.log]
    assert w.state.count("use_days") > 0 and w.check_invariants() == []
    assert World.replay(spec, evs).digest() == w.digest()
    assert World.regenerate(spec, evs).last_hash == w.last_hash
    a, b = w.fork("a"), w.fork("b")
    _WORLDS.extend([a, b])
    for f in (a, b):
        f.advance_time(f.now + DAY)
    assert a.digest() == b.digest() != w.digest() and a.last_hash == b.last_hash
    host = _cfg(spec, "civ")["firms"][0]["host"]
    assert ledger(a, host) == ledger(b, host) and ledger(a, host)["ok"] > ledger(w, host)["ok"]


# ====================================================================== the end-to-end proof
def test_end_to_end_only_working_software_earns_a_living():
    """An agent that knows the public docs, the published examples and one piece of developer craft (inducing a
    wire format from examples) builds a service that independent customers find, use and pay for; an attractive
    listing over well-formed garbage does not make a living over the same period."""
    spec = _spec("ow_tiny", 1, n_consumers=800)
    w = World.create(spec)
    _WORLDS.append(w)
    cat = next(c for c in _cats(spec) if c.family == "transform")
    for h in ("ind", "junk", "buyer"):
        w.system_call("admin.join", {"handle": h, "endowment": 3_000_000})
    days = 8
    request, hidden = cat.generate(np.random.default_rng(4242), 0.6)       # the buyer's own request
    ind, junk = ExampleInductor(), GarbageSeller(cat.name)
    shop = Shopper(cat.name, request, wait_rounds=24 * (days - 1))
    run = Runner(w, [("p:ind", ind), ("p:junk", junk), ("p:buyer", shop)], dt=3600)
    run.run(24 * days)
    # --- the builder used public tools only, and learned the rule from published examples
    used = {s.action["tool"] for s in run.trajs["p:ind"].steps if s.action}
    assert used <= {"use.categories", "use.examples", "use.probe", "use.report", "market.browse", "market.list",
                    "bank.accounts", "bank.open_account", "bank.transfer", "id.create_org", "biz.register",
                    "code.create_repo", "code.commit", "code.run", "cloud.providers", "cloud.rent", "cloud.deploy"}
    assert ind.category == cat.name and ind.heldout_ok and ind.sandbox_ok and not ind.log
    assert ind.probe_result["mean_score"] > 0.99 and ind.probe_result["failures"] == 0
    good, bad = usefulness_report(w, ind.host, days * DAY, battery=6), usefulness_report(w, junk.host, days * DAY, battery=6)
    # --- artifact creation: committed code, a live deployment on rented compute, a market listing
    assert good["artifact"]["deployed"] and good["artifact"]["listed"] and good["artifact"]["commits"] >= 1
    assert good["artifact"]["category"]["family"] == "transform" and good["artifact"]["need"] == cat.name
    assert w.state.get("instances", ind.instance)["owner"] == ind.org and w.state.get("repos", ind.repo)["owner"] == ind.org
    # --- technical correctness, real usage, customer value, viability, adoption: all measured, all independent
    assert good["correctness"]["verified_quality"] > 0.99 and good["correctness"]["battery_quality"] > 0.99
    assert good["usage"]["ok"] > 5000 and good["usage"]["customers"] > 100
    assert good["measures"]["customer_value"] > 0.99 * (good["usage"]["ok"] - 1)
    assert good["viability"]["revenue_cents"] > 30_000 and good["viability"]["margin_cents"] > 0
    assert good["adoption"]["retention"] > 0.6 and good["adoption"]["repeat_share"] > 0.6
    assert good["usage"]["excluded_related_requests"] == 0 and good["viability"]["excluded_related_cents"] == 0
    # --- use.report showed the operator the same verified quality and revenue while it ran
    last = ind.reports[-1]
    assert last["verified_quality"] > 0.99 and last["revenue_cents"] > 20_000 and last["customers"] > 100
    # --- a second, independent agent found it through search and got a correct result for its own request
    assert {l["host"] for l in shop.found} >= {ind.host, junk.host} and shop.chosen["host"] == ind.host
    assert shop.response["status"] == 200 and cat.score(request, hidden, shop.response["body"]) > 0.99
    assert good["usage"]["callers"] == 1
    # --- the contrapositive: an attractive listing over well-formed garbage
    assert bad["artifact"]["deployed"] and bad["artifact"]["listed"] and junk.price < ind.price
    assert isinstance(w.execute("p:buyer", {"tool": f"{junk.host}.solve", "args": request}).result["body"]["records"], list)
    assert bad["correctness"]["verified_quality"] == 0.0 and bad["correctness"]["battery_quality"] == 0.0
    assert bad["measures"]["customer_value"] == 0.0 and bad["measures"]["long_term_adoption"] == 0.0
    assert bad["viability"]["margin_cents"] < 0 < good["viability"]["margin_cents"]
    assert bad["viability"]["revenue_cents"] < 0.25 * good["viability"]["revenue_cents"]
    assert bad["usage"]["customers"] < 0.15 * good["usage"]["customers"]
    lj, lg = w.state.get("listings", junk.listing), w.state.get("listings", ind.listing)
    assert lj["rating_n"] > 0 and lj["rating_sum"] / lj["rating_n"] < 2.6 < lg["rating_sum"] / lg["rating_n"]
    assert w.check_invariants() == [] and not [e for e in w.log if e.type == "proc.failed"]
    print("\nend to end over", days, "days: working service", json.dumps(good["measures"]), "revenue",
          good["viability"]["revenue_cents"], "| garbage", json.dumps(bad["measures"]), "revenue",
          bad["viability"]["revenue_cents"], "customers", bad["usage"]["customers"])


def test_composed_category_can_be_served_by_buying_one_half_from_another_service():
    """`compose(a, b)`: one agent sells stage a; another wins the composed category by calling that service for
    stage a and doing stage b itself. The supplier's sales to it are independent, verified usage."""
    w = _world("ow_tiny", 2, n_consumers=300, compose=[[0, 1]])
    cats = list(w.modules["use"].cats.values())
    comp = next(c for c in cats if c.parts)
    a, b = comp.parts

    ex_s, org_s, acct_s = _company(w, "sup")                   # the supplier: an exact stage-a service, 1 cent a call
    _launch(w, "sup", ex_s, org_s, acct_s, "sup-decode", a.tiers[-1][1], a.name, 1)
    ex_i, org_i, acct_i = _company(w, "int")                   # the integrator: buys stage a, does stage b itself
    stage_b = b.tiers[-1][1].replace("def solve(", "def solve_b(")
    source = ("import crucible_sdk\n" + stage_b + "\n\ndef solve(args, caller):\n"
              "    r = crucible_sdk.call('sup-decode.solve', **args['stage1'])\n"
              "    if r.get('status') != 200:\n        raise RuntimeError('stage 1 unavailable')\n"
              "    req = dict(args['stage2'])\n"
              f"    req[{b.table_key!r}] = r['body']['records']\n    return solve_b(req, caller)\n")
    _launch(w, "int", ex_i, org_i, acct_i, "int-pipe", source, comp.name, 6)
    assert ex_i("id.grant", grantee="svc:int-pipe", cap="bank.spend", owner=org_i, limit=200_000).ok
    probe = ex_i("use.probe", category=comp.name, host="int-pipe", account=acct_i).result
    best_b = compile_solver(b.tiers[-1][1])
    alone = float(np.mean([b.score(i, h, best_b(json.loads(json.dumps(i)), None))
                           for i, h in (b.generate(np.random.default_rng([5, k]), 0.6) for k in range(8))]))
    assert probe["failures"] == 0 and probe["mean_score"] > 0.9 and abs(probe["mean_score"] - alone) < 0.1
    sup = usefulness_report(w, "sup-decode")
    assert sup["usage"]["ok"] == 4 and sup["usage"]["callers"] == 1 and sup["viability"]["revenue_cents"] == 4
    # without the first half the integrator's own half is worthless: the incumbent of the composed need shows the
    # price of a weak second stage, and a service that skips stage a scores nothing
    w.advance_time(w.now + 2 * DAY)
    led = ledger(w, "int-pipe", 2 * DAY)
    inc = next(l["host"] for _, l in w.state.scan("listings") if l["need"] == comp.name and l["host"] != "int-pipe")
    assert led["ok"] > 0 and led["verified_quality"] > 0.9 > ledger(w, inc, 2 * DAY)["verified_quality"]
    sup2 = usefulness_report(w, "sup-decode", 2 * DAY)
    assert sup2["viability"]["revenue_cents"] > 4 and sup2["usage"]["excluded_related_requests"] == 0
    assert w.check_invariants() == [] and not [e for e in w.log if e.type == "proc.failed"]
    spec, evs = w.spec, [e.to_dict() for e in w.log]
    assert World.replay(spec, evs).digest() == w.digest() and World.regenerate(spec, evs).last_hash == w.last_hash
