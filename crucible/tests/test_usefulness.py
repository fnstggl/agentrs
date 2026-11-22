"""The usefulness substrate: the category extension interface, the built-in families, composition, registration,
and the kernel module `use` (discovery, acceptance lab, usefulness ledger).

Tests that run agent code in the sandbox (anything that deploys a service) need Landlock/seccomp.
"""
import copy
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.core.errors import ProtocolError  # noqa: E402
from crucible.core.kernel import World, WorldSpec  # noqa: E402
from crucible.economy import needs as needs_mod  # noqa: E402
from crucible.economy.needs import NEEDS, compile_solver, tier_quality  # noqa: E402
from crucible.goals.metrics import METRICS, evaluate_metric, m_independent_revenue  # noqa: E402
from crucible.usefulness import (FAMILIES, GENERATED, NeedCategory, Population, category_families, category_family,  # noqa: E402
                                 compose, instantiate, measure_tiers, register_category, scoped_registry,
                                 unregister_generated)
from crucible.usefulness import category as cat_mod  # noqa: E402
from crucible.usefulness.families._common import check_table  # noqa: E402
from crucible.usefulness.module import battery_quality, ledger, usefulness_report  # noqa: E402
from crucible.worldgen import generate  # noqa: E402
from crucible.worldgen.admission import admit  # noqa: E402

DAY = 86400
ALL = category_families()
LEARNED = category_families(relation="exemplified") + category_families(relation="experiential")
# Families whose best reference tier is exactly optimal once the rule is known, by design: decoding a known wire
# format is exact; greedy by marginal return is optimal for concave returns; the most reliable route under known
# hazards is a shortest path. Their difficulty is inducing (transform) or learning (netrel) the rule.
EXACT_BY_DESIGN = {"transform", "allocate", "netrel"}
SEEDS = (11, 12, 13)


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


_WORLDS: list = []


@pytest.fixture(autouse=True)
def _close_worlds():
    """Worlds that ran sandboxed services hold worker processes: release them after every test."""
    yield
    while _WORLDS:
        _WORLDS.pop().close()


def _solve(src, inp):
    return compile_solver(src)(json.loads(json.dumps(inp)), None)


def _cases(cat, n, seed=0, size=0.6):
    return [cat.generate(np.random.default_rng([seed, k]), size) for k in range(n)]


# ====================================================================== families
def test_families_cover_kinds_relations_and_splits():
    kinds = {FAMILIES[f].kind for f in ALL}
    assert len(ALL) >= 8 and {"data transformation", "information retrieval", "scheduling", "resource allocation",
                              "data quality / entity resolution", "network reliability", "content discovery",
                              "organisational coordination"} <= kinds
    assert FAMILIES["transform"].relation == "exemplified" and FAMILIES["netrel"].relation == "experiential"
    for split in ("train", "test"):                        # every relation to the rule is trained on and tested
        assert {FAMILIES[f].relation for f in category_families(split)} == {"specified", "exemplified", "experiential"}
    assert len(category_families("test")) >= 2 and category_families("val")
    assert set(category_families("train")) | set(category_families("val")) | set(category_families("test")) == set(ALL)


@pytest.mark.parametrize("fam", ALL)
def test_category_contract(fam):
    """Scores in [0, 1]; tiers ordered by measured quality; generation and scoring deterministic; inputs are
    modest JSON; one best-tier request costs at most a few milliseconds."""
    best_q = []
    for seed in SEEDS:
        cat = instantiate(fam, seed)
        assert re.fullmatch(rf"gen\.{fam}\.[0-9a-f]{{8}}", cat.name) and len(cat.name) <= 40
        assert 2 <= len(cat.tiers) <= 4 and cat.output_key and cat.doc and len(cat.keywords) >= 3
        m = measure_tiers(cat, n=12, seed=seed, size=0.6)
        q = [x["quality"] for x in m]
        assert all(0.0 <= x <= 1.0 for x in q)
        assert all(a <= b + 1e-9 for a, b in zip(q, q[1:])), (cat.name, q)       # monotone, measured
        assert q[-1] - q[0] >= 0.1 and q[-1] >= 0.8, (cat.name, q)              # clearly different quality
        assert m[-1]["cpu_ms"] < 5.0 and m[-1]["lines"] < 6000, (cat.name, m[-1])
        best_q.append(q[-1])
        # deterministic generation and scoring
        a, b = _cases(cat, 3, seed), _cases(cat, 3, seed)
        assert json.dumps([x[0] for x in a], sort_keys=True) == json.dumps([x[0] for x in b], sort_keys=True)
        out = _solve(cat.tiers[-1][1], a[0][0])
        assert cat.score(a[0][0], a[0][1], out) == cat.score(b[0][0], b[0][1], copy.deepcopy(out))
        for size in (0.3, 1.0):                                                  # JSON, modest in size
            inp, _ = cat.generate(np.random.default_rng([seed, 99]), size)
            assert len(json.dumps(inp, allow_nan=False)) < 16_000
    if fam not in EXACT_BY_DESIGN:
        assert float(np.mean(best_q)) < 0.999, (fam, best_q)                     # not trivially perfect


@pytest.mark.parametrize("fam", ALL)
def test_evaluator_is_total_and_rejects_garbage(fam):
    """Malformed and adversarial outputs score 0; nothing makes the evaluator raise or leave [0, 1]."""
    cat = instantiate(fam, 5)
    rng = np.random.default_rng(5)
    for inp, hidden in _cases(cat, 4, 5):
        good = _solve(cat.tiers[-1][1], inp)
        key = cat.output_key
        assert key in good and cat.score(inp, hidden, good) > 0.2
        nan, inf = float("nan"), float("inf")
        n_items = len(good[key]) if hasattr(good[key], "__len__") else 3
        deep = cur = {}
        for _ in range(60):
            cur["x"] = {}
            cur = cur["x"]
        garbage = [None, {}, [], "x", 1, 3.5, True, {key: None}, {key: "x"}, {key: 5}, {key: nan}, {key: [nan] * n_items},
                   {key: {"a": nan}}, {key: [inf] * n_items}, {key: [None] * n_items}, {key: [[]] * n_items},
                   {key: [{}] * n_items}, {key: ["x"] * n_items}, {key: [-1] * n_items}, {key: [10**12] * n_items},
                   {key: list(range(200_000))}, {key: "y" * 500_000}, {key: deep}, {"wrong_key": good[key]},
                   inp, copy.deepcopy(inp), {key: copy.deepcopy(inp)}, {**copy.deepcopy(inp), key: copy.deepcopy(inp)},
                   {key: list(inp.values())}]
        for g in garbage:
            assert cat.score(inp, hidden, g) == 0.0, (cat.name, repr(g)[:80])
        # fuzz: corrupt a good output at random places; the score stays a number in [0, 1]
        for _ in range(40):
            bad = copy.deepcopy(good)
            node = bad[key]
            for _ in range(int(rng.integers(1, 4))):
                junk = [nan, inf, -1, 10**9, "x", None, True, [], {}, 0.5][int(rng.integers(0, 10))]
                if isinstance(node, list) and node:
                    node[int(rng.integers(0, len(node)))] = junk
                elif isinstance(node, dict) and node:
                    node[sorted(node)[int(rng.integers(0, len(node)))]] = junk
            s = cat.score(inp, hidden, bad)
            assert isinstance(s, float) and 0.0 <= s <= 1.0 and math.isfinite(s)


@pytest.mark.parametrize("fam", ALL)
def test_published_pool_is_seeded_shared_and_disjoint_from_scoring_instances(fam):
    cat = instantiate(fam, 7)
    pool = cat.pool()
    assert len(pool) >= 6 and pool is cat.pool()
    assert cat.examples(3) == pool[:3] and cat.examples(2, offset=2) == pool[2:4]
    keys = {cat_mod.input_key(e["input"]) for e in pool}
    assert len(keys) == len(pool)
    fresh = [cat.generate(np.random.default_rng([3, k]), cat_mod.POOL_INSTANCE_SIZE)[0] for k in range(150)]
    assert not keys & {cat_mod.input_key(i) for i in fresh}
    # even a generator started in the very state that produced the pool never serves a published instance
    twin = np.random.default_rng([int(cat.uid[:12], 16), 0x5EED])
    assert cat_mod.input_key(cat.generate(twin, cat_mod.POOL_INSTANCE_SIZE)[0]) not in keys
    for e in pool:
        if cat.relation == "exemplified":                    # worked examples carry the correct output
            assert set(e) == {"input", "output"}
            hidden = _hidden_of(cat, e["input"])
            assert cat.score(e["input"], hidden, e["output"]) == pytest.approx(1.0)
        else:                                                # the others publish sample requests only
            assert set(e) == {"input"}


def _hidden_of(cat, inp):
    """Regenerate the pool to recover the hidden truth of a published instance (evaluator side)."""
    rng = np.random.default_rng([int(cat.uid[:12], 16), 0x5EED])
    for _ in range(200):
        i, h = cat.generate_raw(rng, cat_mod.POOL_INSTANCE_SIZE)
        if cat_mod.input_key(i) == cat_mod.input_key(inp):
            return h
    raise AssertionError("published instance not found in its own pool stream")


def test_hidden_rule_varies_per_seed_for_exemplified_and_experiential_families():
    """A solver tuned to one instance scores measurably worse on another instance of the same family."""
    gaps = {}
    for fam in LEARNED:
        own, foreign = [], []
        for seed in range(20, 26):
            a, b = instantiate(fam, seed), instantiate(fam, seed + 500)
            assert a.name != b.name and a.rule != b.rule
            tuned_to_a = compile_solver(a.tiers[-1][1])
            for inp, hidden in _cases(b, 6, seed):
                own.append(b.score(inp, hidden, _solve(b.tiers[-1][1], inp)))
                try:
                    foreign.append(b.score(inp, hidden, tuned_to_a(json.loads(json.dumps(inp)), None)))
                except Exception:  # noqa: BLE001 - a solver tuned to another rule may not even run
                    foreign.append(0.0)
        gaps[fam] = (float(np.mean(own)), float(np.mean(foreign)))
        assert gaps[fam][0] - gaps[fam][1] >= 0.05, (fam, gaps[fam])
    print("own vs foreign-tuned best tier:", {k: (round(a, 3), round(b, 3)) for k, (a, b) in gaps.items()})


def test_specified_families_draw_a_documented_mix_per_seed():
    for fam in category_families(relation="specified"):
        docs = {instantiate(fam, s).doc for s in range(30, 42)}
        rules = {json.dumps(instantiate(fam, s).rule, sort_keys=True) for s in range(30, 42)}
        assert len(docs) >= 4 and len(rules) >= 6, fam          # the mix is generated, and the doc states it


def test_experiential_parameters_can_be_learned_from_outcomes_alone():
    """No document or example states how segments respond to price; mean scores of served batches (what
    `use.probe` and `use.report` return) are enough to find out."""
    from crucible.usefulness.families.quote import _model
    cat = instantiate("quote", 3)
    segs = sorted(cat.rule["segments"])

    def outcome(belief, k):          # mean score of a deployed belief on fresh requests: all a learner ever sees
        fn = compile_solver(_model(belief))
        return float(np.mean([cat.score(i, h, fn(json.loads(json.dumps(i)), None)) for i, h in _cases(cat, 6, 1000 + k)]))
    belief = {s: [1.0, 6.0] for s in segs}                   # start from the documented nominal model
    start = outcome(belief, 0)
    n = 0
    for sweep in range(2):
        for s in segs:
            trials = {}
            for m in (0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7):
                n += 1
                trials[m] = outcome({**belief, s: [m, 6.0]}, 1)
            belief[s][0] = max(trials, key=trials.get)
    learned = outcome(belief, 2)
    best = float(np.mean([cat.score(i, h, _solve(cat.tiers[-1][1], i)) for i, h in _cases(cat, 6, 1002)]))
    print(f"quote: nominal {start:.3f} -> learned from {n} outcomes {learned:.3f} (experienced incumbent {best:.3f})")
    assert learned > start + 0.1 and learned > 0.9 * best
    for s in segs:                                          # and what was learned is the hidden truth
        assert abs(belief[s][0] - cat.rule["segments"][s][0]) <= 0.25


# ====================================================================== the extension interface
def test_a_new_family_is_added_without_touching_any_module():
    """The decorator is the whole interface: a developer's own family becomes a need every mechanism understands."""
    @category_family("parity_test", kind="toy", relation="specified", split="train", doc="toy",
                     population=Population(0.5), demand_links=("allocate",))
    def build(rng, params):
        k = int(rng.integers(2, 6))
        return {"title": "Count multiples", "doc": f"Count the multiples of {k}.", "keywords": ("count", "toy", "k"),
                "generate_raw": lambda rng, size: ({"xs": [int(x) for x in rng.integers(0, 99, 8)]}, k),
                "score_raw": lambda inp, k, out: float(out.get("n") == sum(1 for x in inp["xs"] if x % k == 0)),
                "tiers": [("zero", "def solve(args, caller):\n    return {'n': 0}\n"),
                          ("count", f"def solve(args, caller):\n    return {{'n': sum(1 for x in args['xs'] if x % {k} == 0)}}\n")],
                "base_price_cents": 1.0, "rule": {"k": k}, "output_key": "n"}
    try:
        cat = instantiate("parity_test", 1)
        assert cat.externalities.derived[0].dst_family == "allocate" and cat.population.share == 0.5
        nd = register_category(cat)
        assert NEEDS[cat.name] is nd and nd.tiers == cat.tiers and nd.base_price_cents == 1.0
        assert tier_quality(cat.name, 1) == 1.0 and tier_quality(cat.name, 0) < 1.0     # the 3.0 feasibility check
        assert len({instantiate("parity_test", s).rule["k"] for s in range(12)}) >= 3       # the rule varies per seed
        with pytest.raises(ProtocolError):                   # a category must have 2-4 measured tiers
            NeedCategory("parity_test", 0, {}, "toy", "specified", "train", "t", "d", ("a",), None, None, [("x", "")],
                         1.0, 1.0)
    finally:
        FAMILIES.pop("parity_test", None)
        unregister_generated([n for n in list(GENERATED) if n.startswith("gen.parity_test.")])


def test_registration_is_deterministic_idempotent_and_refuses_collisions():
    cat = instantiate("allocate", 77)
    assert instantiate("allocate", 77) is cat and instantiate("allocate", 77, {}) is cat
    assert instantiate("allocate", 78).name != cat.name
    assert instantiate("transform", 1, {"feeds": 3}).name != instantiate("transform", 1).name   # params are hashed
    with scoped_registry():
        nd = register_category(cat)
        assert register_category(cat) is nd and NEEDS[cat.name] is nd          # second world, same instance
        assert tier_quality(cat.name, len(cat.tiers) - 1) > 0.9
        forged = copy.copy(cat)                                                # same name, another rule
        forged.uid = cat.uid[:8] + ("0" if cat.uid[8] != "0" else "1") * 24
        assert forged.name == cat.name and forged.uid != cat.uid
        with pytest.raises(ProtocolError):
            register_category(forged)
        NEEDS.pop(cat.name)                                                    # registry reset under a live world
        again = register_category(cat)
        assert again is not nd and NEEDS[cat.name] is again and (again.name, again.tiers) == (nd.name, nd.tiers)
        assert again.generate(np.random.default_rng(1), 0.5) == nd.generate(np.random.default_rng(1), 0.5)
    assert cat.name not in NEEDS and cat.name not in GENERATED
    assert (cat.name, len(cat.tiers) - 1) not in needs_mod._TQ                 # no stale measured qualities


# ====================================================================== composition
TABLE_FAMILIES = [f for f in ALL if f != "transform"]


def test_every_table_family_can_be_the_second_stage():
    for fam in TABLE_FAMILIES:
        cat = instantiate(fam, 3)
        assert cat.table_key, fam
        for inp, _ in _cases(cat, 3, 3, size=1.0):
            assert check_table(inp[cat.table_key]) == [], (fam, check_table(inp[cat.table_key])[:2])
    t = instantiate("transform", 3)
    assert t.render_table is not None and t.table_key is None
    with pytest.raises(ProtocolError):
        compose(instantiate("allocate", 1), instantiate("dedupe", 1))     # the first stage must produce a table
    with pytest.raises(ProtocolError):
        compose(t, instantiate("transform", 4))                           # the second stage must take one


@pytest.mark.parametrize("fam", TABLE_FAMILIES)
def test_composed_request_needs_both_stages(fam):
    a, b = instantiate("transform", 41), instantiate(fam, 42)
    c = compose(a, b)
    assert compose(a, b) is c and re.fullmatch(r"gen\.compose\.[0-9a-f]{8}", c.name) and c.parts == (a, b)
    assert c.split in ("val", "test") and a.name in c.doc and b.name in c.doc and c.output_key == b.output_key
    cases = _cases(c, 8, 9)
    for inp, hidden in cases[:2]:
        assert set(inp) == {"stage1", "stage2"} and b.table_key not in inp["stage2"]
        # the table appears nowhere in the request: only doing stage 1 produces it
        table = _solve(a.tiers[-1][1], inp["stage1"])["records"]
        assert table == hidden["b_in"][b.table_key]
    q = [x["quality"] for x in measure_tiers(c, n=8, seed=9)]
    assert all(x <= y + 1e-9 for x, y in zip(q, q[1:])), q
    alone = [x["quality"] for x in measure_tiers(b, n=8, seed=9)]
    assert len(q) == len(alone) and q[-1] >= 0.8                           # both halves done: b's own quality
    assert all(abs(x - y) < 0.1 for x, y in zip(q, alone)), (q, alone)
    # skipping stage 1 (handing stage 2 the undecoded rows) does not work
    solve_b = compile_solver(b.tiers[-1][1])
    skipped = []
    for inp, hidden in cases:
        req = dict(json.loads(json.dumps(inp["stage2"])))
        req[b.table_key] = [r["data"] if isinstance(r["data"], dict) else {"raw": r["data"]} for r in inp["stage1"]["rows"]]
        try:
            skipped.append(c.score(inp, hidden, solve_b(req, None)))
        except Exception:  # noqa: BLE001
            skipped.append(0.0)
    assert float(np.mean(skipped)) <= 0.5 * q[-1], (fam, skipped, q)
    # an agent can win with someone else's service for one half: their stage-1 output, its own stage 2
    theirs, mine = compile_solver(a.tiers[-1][1]), compile_solver(b.tiers[-1][1])
    mixed = []
    for inp, hidden in cases:
        req = dict(json.loads(json.dumps(inp["stage2"])))
        req[b.table_key] = theirs(json.loads(json.dumps(inp["stage1"])), None)["records"]
        mixed.append(c.score(inp, hidden, mine(req, None)))
    best = [c.score(inp, hidden, _solve(c.tiers[-1][1], inp)) for inp, hidden in cases]
    assert mixed == best
    e = c.pool()[0]                                      # a worked example shows what stage 1 must produce
    assert e["stage1_output"] == {"records": _solve(a.tiers[-1][1], e["input"]["stage1"])["records"]}
    assert ("output" in e) == (b.truth is not None)


# ====================================================================== the kernel module
USE_CFG = {"categories": [{"family": "transform", "seed": 5}, {"family": "allocate", "seed": 6},
                          {"family": "quote", "seed": 7}], "compose": [[0, 1]], "probes_per_day": 3}


def _bare(seed=0, mods=("identity", "bank", "admin", "use"), cfg=None):
    spec = WorldSpec("use-test", seed, [{"name": m, "config": (cfg or USE_CFG) if m == "use" else {}} for m in mods], {})
    return World.create(spec), spec


def test_discovery_tools_expose_public_fields_only():
    w, spec = _bare()
    for h in ("ann", "bob"):
        w.system_call("admin.join", {"handle": h})
    cats = w.execute("p:ann", {"tool": "use.categories", "args": {}}).result
    assert len(cats) == 4 and sorted(c["category"] for c in cats) == sorted(w.modules["use"].cats)
    assert all(set(c) == {"category", "title", "kind", "relation", "doc", "keywords", "reference_price_cents",
                          "examples"} for c in cats)
    dump = json.dumps(cats)
    for c in w.modules["use"].cats.values():                 # no rule, no reference code, no seed, no split
        assert c.uid not in dump and c.tiers[-1][1][:60] not in dump
        if c.family == "transform":
            assert not any(f"'{tag}'" in dump or f'"{tag}"' in dump for tag in c.rule["feeds"])
    one = w.execute("p:ann", {"tool": "use.categories", "args": {"category": cats[0]["category"]}}).result
    assert one == [cats[0]]
    assert not w.execute("p:ann", {"tool": "use.categories", "args": {"category": "sort.records"}}).ok
    # examples: the same for everyone, paged, worked only for exemplified categories
    for c in cats:
        a = w.execute("p:ann", {"tool": "use.examples", "args": {"category": c["category"], "n": 2}}).result
        b = w.execute("p:bob", {"tool": "use.examples", "args": {"category": c["category"], "n": 2}}).result
        assert a == b and a["total"] == c["examples"] and len(a["examples"]) == 2
        assert all(("output" in e or "stage1_output" in e) == (c["relation"] == "exemplified") for e in a["examples"])
        nxt = w.execute("p:ann", {"tool": "use.examples", "args": {"category": c["category"], "n": 2, "offset": 2}}).result
        assert nxt["examples"] and nxt["examples"][0] != a["examples"][0]
    # the catalog is state, and it must agree with what is registered
    assert sorted(w.state.ids("use_cats")) == sorted(c["category"] for c in cats) and w.check_invariants() == []
    assert World.replay(spec, [e.to_dict() for e in w.log]).digest() == w.digest()


def test_use_module_passes_admission():
    r = admit(["use"], steps=300, configs={"use": USE_CFG})
    assert r.passed, (r.checks, r.errors[:5])
    r = admit(["use"], base=["identity", "bank", "admin", "code", "search", "cloud", "market", "consumers"],
              steps=300, seed=2, configs={"use": USE_CFG, "consumers": {"n": 200, "needs": {
                  n: 1.0 for n in [c.name for c in cat_mod.from_config(USE_CFG)]}, "income_per_day_cents": 100_000}})
    assert r.passed, (r.checks, r.errors[:5])


def _tiny(seed=3, **ov):
    w = World.create(generate({"family": "ow_tiny", "overrides": {"n_consumers": 300, **ov}}, seed))
    _WORLDS.append(w)
    return w


def _deploy(w, handle, need, source, price=2, endowment=600_000, host=None, org=True):
    """An agent's own service through ordinary tools (sandboxed). Returns (ex, owner principal, host, account)."""
    from crucible.usefulness.agents import service_repo
    if not w.state.has("principals", f"p:{handle}"):
        w.system_call("admin.join", {"handle": handle, "endowment": endowment})
    ex = lambda t, **k: w.execute(f"p:{handle}", {"tool": t, "args": k})  # noqa: E731
    me = w.modules["bank"].primary_account(w.state, f"p:{handle}")
    owner, acct = f"p:{handle}", me
    if org:
        owner = f"o:{handle}-co"
        assert ex("id.create_org", handle=f"{handle}-co", name="Co", kind="company").ok
        assert ex("biz.register", org=owner, account=me).ok
        acct = ex("bank.open_account", owner=owner).result["account"]
        assert ex("bank.transfer", from_account=me, to=acct, amount=endowment // 2).ok
    repo = ex("code.create_repo", name=f"{handle}-svc", owner=owner).result["repo"]
    assert ex("code.commit", repo=repo, files=service_repo(source, "svc", "svc", price)).ok
    prov = sorted(p["provider"] for p in w.query(f"p:{handle}", "cloud.providers"))[0]
    inst = ex("cloud.rent", provider=prov, cores=1, account=acct, owner=owner).result["instance"]
    host = host or f"{handle}-svc"
    r = ex("cloud.deploy", repo=repo, host=host, instance=inst)
    assert r.ok and r.result["status"] == "live", r.error or r.result
    assert ex("market.list", kind="api", title=f"{host} service", price_cents=price, seller=owner, host=host,
              endpoint="solve", need=need, description="Keywords: " + " ".join(NEEDS[need].keywords)).ok
    return ex, owner, host, acct


def _cat_of(w, fam):
    return next(c for c in w.modules["use"].cats.values() if c.family == fam)


def test_acceptance_lab_is_paid_metered_rate_limited_and_returns_only_the_score():
    w = _tiny(module_config={"use": {"probes_per_day": 2, "probe_fee_cents": 1500, "probe_n": 4}})
    cat = _cat_of(w, "allocate")
    ex, owner, host, acct = _deploy(w, "dev", cat.name, cat.tiers[-1][1])
    ex2, owner2, host2, acct2 = _deploy(w, "weak", cat.name, cat.tiers[0][1])
    st = w.state
    fees0, cu0 = st.get("accounts", "acct_fees")["balance"], st.get("instances", st.get("deployments", st.get("hosts", host)["deployment"])["instance"])["used_cu"]
    r = ex("use.probe", category=cat.name, host=host, account=acct)
    assert r.ok, r.error
    assert set(r.result) == {"category", "host", "instances", "mean_score", "failures", "fee_cents", "runs_left_today"}
    assert r.result["mean_score"] > 0.95 and r.result["failures"] == 0 and r.result["instances"] == 4
    assert st.get("accounts", "acct_fees")["balance"] - fees0 == 1500              # the fee went through the bank
    dep = st.get("deployments", st.get("hosts", host)["deployment"])
    assert dep["requests"] == 4 and st.get("instances", dep["instance"])["used_cu"] > cu0   # metered like any request
    weak = ex2("use.probe", category=cat.name, host=host2, account=acct2).result
    assert weak["mean_score"] < r.result["mean_score"] - 0.2                      # the lab tells good from weak
    # probes are the operator's own activity: they are not usage
    led = ledger(w, host)
    assert led["requests"] == 0 and led["ok"] == 0 and led["metered_ok"] == 4 and led["revenue_cents"] == 0
    # only the operator may probe a service; unknown categories and dead services are refused
    assert ex2("use.probe", category=cat.name, host=host, account=acct2).error["code"] == "forbidden"
    assert ex("use.probe", category="gen.nope.00000000", host=host, account=acct).error["code"] == "invalid_action"
    assert ex("use.probe", category=cat.name, host="nohost", account=acct).error["code"] == "not_found"
    assert ex("use.probe", category=cat.name, host=host, account=acct).ok
    lim = ex("use.probe", category=cat.name, host=host, account=acct)
    assert lim.error["code"] == "rate_limited" and st.get("accounts", "acct_fees")["balance"] - fees0 == 1500 * 3
    w.advance_time(w.now + DAY)                                                   # a new day, a new allowance
    assert ex("use.probe", category=cat.name, host=host, account=acct).ok
    # a wrong-category probe is scored by that category's evaluator: an allocator is not a decoder
    other = _cat_of(w, "transform")
    assert ex("use.probe", category=other.name, host=host, account=acct).result["mean_score"] == 0.0
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []
    # replay and regeneration reproduce the lab runs (recorded outcomes are never re-executed)
    spec, evs = w.spec, [e.to_dict() for e in w.log]
    assert World.replay(spec, evs).digest() == w.digest()
    assert World.regenerate(spec, evs).last_hash == w.last_hash


def test_ledger_measures_verified_use_and_related_parties_do_not_count():
    w = _tiny(seed=4, n_consumers=500)
    cat = _cat_of(w, "allocate")
    ex, owner, host, acct = _deploy(w, "ada", cat.name, cat.tiers[-1][1], price=3)
    for h in ("friend", "stranger"):
        w.system_call("admin.join", {"handle": h, "endowment": 100_000})
    inv = ex("id.invite", org=owner, person="p:friend", role="member").result["invite"]
    assert w.execute("p:friend", {"tool": "id.respond_invite", "args": {"invite": inv, "accept": True}}).ok
    req, _ = cat.generate(np.random.default_rng(8), 0.5)
    base = usefulness_report(w, host)
    assert base["measures"]["artifact_creation"] == 1.0 and base["measures"]["real_usage"] == 0.0
    money0 = m_independent_revenue(w, owner, 7 * DAY)

    def buy(who):
        r = w.execute(who, {"tool": f"{host}.solve", "args": req})
        assert r.ok and r.result["status"] == 200, r.error or r.result
    # --- self-purchases: the owner, and a member of the owner's organisation
    for _ in range(5):
        buy("p:ada")
        buy("p:friend")
    # --- a purchase by an organisation the owner controls: its relay service pays ours from its own account
    relay = ("import crucible_sdk\n\ndef solve(args, caller):\n"
             f"    return crucible_sdk.call('{host}.solve', **args)\n")
    assert ex("id.create_org", handle="ada-two", name="Two", kind="company").ok
    me = w.modules["bank"].primary_account(w.state, "p:ada")
    assert ex("biz.register", org="o:ada-two", account=me).ok
    acct2 = ex("bank.open_account", owner="o:ada-two").result["account"]
    assert ex("bank.transfer", from_account=me, to=acct2, amount=50_000).ok
    repo2 = ex("code.create_repo", name="relay2", owner="o:ada-two").result["repo"]
    from crucible.usefulness.agents import service_repo
    assert ex("code.commit", repo=repo2, files=service_repo(relay, "relay", "relay", 0)).ok
    prov = sorted(p["provider"] for p in w.query("p:ada", "cloud.providers"))[0]
    inst2 = ex("cloud.rent", provider=prov, cores=1, account=acct2, owner="o:ada-two").result["instance"]
    assert ex("cloud.deploy", repo=repo2, host="two-relay", instance=inst2).result["status"] == "live"
    assert ex("id.grant", grantee="svc:two-relay", cap="bank.spend", owner="o:ada-two", limit=10_000).ok
    bal2 = w.modules["bank"].balance(w.state, acct2)
    for _ in range(4):
        r = w.execute("p:stranger", {"tool": "two-relay.solve", "args": req})
        assert r.ok and r.result["status"] == 200 and r.result["body"]["status"] == 200, r.error or r.result
    assert bal2 - w.modules["bank"].balance(w.state, acct2) == 4 * 3          # o:ada-two really paid o:ada-co
    # --- a transfer dressed up as a sale
    mine = w.modules["bank"].primary_account(w.state, "p:stranger")
    assert w.execute("p:stranger", {"tool": "bank.transfer", "args": {"from_account": mine, "to": acct, "amount": 5000,
                                                                     "ref": f"svc:{host}", "memo": "sale"}}).ok
    after = usefulness_report(w, host)
    assert after["usage"]["excluded_related_requests"] == 14 and after["viability"]["excluded_related_cents"] == 14 * 3
    for k in ("real_usage", "customer_value", "long_term_adoption"):
        assert after["measures"][k] == base["measures"][k] == 0.0, k
    assert after["viability"]["revenue_cents"] == 0 and after["usage"]["callers"] == 0
    assert evaluate_metric(w, "use_independent_revenue", {"host": host}) == 0.0
    # the generic metric and the ledger agree that none of the service money is independent
    assert m_independent_revenue(w, owner, 7 * DAY) - money0 == 5000          # only the stranger's plain transfer
    # --- an independent customer does count
    for _ in range(3):
        buy("p:stranger")
    ind = usefulness_report(w, host)
    assert ind["usage"]["ok"] == 3 and ind["usage"]["callers"] == 1 and ind["viability"]["revenue_cents"] == 9
    assert ind["adoption"]["repeat_callers"] == 1 and ind["viability"]["excluded_related_cents"] == 14 * 3
    # --- and so do households, with verified quality; the tool shows the operator the same ledger
    w.advance_time(w.now + 3 * DAY)
    rep = usefulness_report(w, host, battery=4)
    led = ledger(w, host)
    assert led["requests"] > 50 and led["verified_quality"] > 0.95 and led["customers"] > 0
    assert rep["correctness"]["battery_quality"] > 0.95 and rep["measures"]["customer_value"] > 0.9 * (led["ok"] - 3)
    assert rep["viability"]["revenue_cents"] > 9 and rep["viability"]["compute_cost_cents"] > 0
    assert rep["viability"]["margin_cents"] == rep["viability"]["revenue_cents"] - rep["viability"]["compute_cost_cents"]
    assert 0.0 <= led["retention"] <= 1.0 and led["repeat_customers"] <= led["customers"]
    tool = ex("use.report", host=host, days=7)
    assert tool.ok and all(tool.result[k] == led[k] for k in tool.result) and "owner" not in tool.result
    assert w.execute("p:stranger", {"tool": "use.report", "args": {"host": host}}).error["code"] == "forbidden"
    assert w.execute("p:friend", {"tool": "use.report", "args": {"host": "nohost"}}).error["code"] == "not_found"
    for name in ("use_verified_quality", "use_independent_requests", "use_customer_value", "use_independent_revenue",
                 "use_retained_customers"):
        assert name in METRICS and evaluate_metric(w, name, {"host": host}) >= 0.0
    assert evaluate_metric(w, "use_independent_requests", {"host": host}) == led["ok"]
    # revenue in the ledger is exactly what households and the stranger paid this service
    sales = sum(-d for _, t in w.state.scan("txns") if t["ref"] == f"svc:{host}" and t["kind"] == "sale"
                for a, d in t["postings"] if a in ("acct_households", mine))
    assert led["revenue_cents"] == sales
    # the incumbent of the same category is measured by the same ledger, with its own (lower) verified quality
    inc = next(l["host"] for _, l in w.state.scan("listings") if l["need"] == cat.name and l["host"] != host)
    assert 0.0 < ledger(w, inc)["verified_quality"] < led["verified_quality"]
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []


def test_use_world_replays_regenerates_and_forks_deterministically(tmp_path):
    w = _tiny(seed=6)
    cat = _cat_of(w, "transform")
    ex, owner, host, acct = _deploy(w, "eve", cat.name, cat.tiers[-1][1], price=2)
    w.system_call("admin.join", {"handle": "zed", "endowment": 50_000})
    req, _ = cat.generate(np.random.default_rng(2), 0.4)
    assert ex("use.probe", category=cat.name, host=host, account=acct).ok
    w.advance_time(w.now + DAY)
    assert w.execute("p:zed", {"tool": f"{host}.solve", "args": req}).ok
    assert not w.execute("p:zed", {"tool": "use.probe", "args": {"category": cat.name, "host": host, "account": acct}}).ok
    w.advance_time(w.now + DAY)
    assert w.check_invariants() == [] and not [e for e in w.log if e.type == "proc.failed"]
    spec, evs = w.spec, [e.to_dict() for e in w.log]
    rep = World.replay(spec, evs)
    assert rep.digest() == w.digest() and ledger(rep, host) == ledger(w, host)
    assert World.regenerate(spec, evs).last_hash == w.last_hash
    a, b = w.fork("fa"), w.fork("fb")
    _WORLDS.extend([a, b])
    for f in (a, b):
        assert f.execute("p:zed", {"tool": f"{host}.solve", "args": req}).ok
        f.advance_time(f.now + DAY)
    assert a.digest() == b.digest() != w.digest() and ledger(a, host) == ledger(b, host)
    assert a.check_invariants() == [] and usefulness_report(a, host)["usage"]["callers"] == 1
    # --- a fresh process rebuilds the world from its spec alone: it re-registers identical needs
    (tmp_path / "world.json").write_text(json.dumps({"spec": spec.to_dict(), "events": evs}))
    names = sorted(w.modules["use"].cats)
    code = (
        "import json, sys, hashlib\n"
        "import crucible\ncrucible.load_all_modules()\n"
        "import crucible.usefulness.module\n"
        "from crucible.core.kernel import World, WorldSpec\n"
        "from crucible.economy.needs import NEEDS\n"
        "d = json.load(open(sys.argv[1]))\n"
        "assert not [n for n in NEEDS if n.startswith('gen.')]\n"
        "w = World.replay(WorldSpec.from_dict(d['spec']), d['events'])\n"
        "gen = sorted(n for n in NEEDS if n.startswith('gen.'))\n"
        "sig = hashlib.sha256(json.dumps([[n, NEEDS[n].doc, NEEDS[n].tiers, NEEDS[n].work_scale, "
        "NEEDS[n].base_price_cents] for n in gen]).encode()).hexdigest()\n"
        "print(json.dumps({'digest': w.digest(), 'needs': gen, 'sig': sig, 'inv': w.check_invariants()}))\n")
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "world.json")], capture_output=True, text=True,
                         env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    sig = __import__("hashlib").sha256(json.dumps([[n, NEEDS[n].doc, NEEDS[n].tiers, NEEDS[n].work_scale,
                                                    NEEDS[n].base_price_cents] for n in names]).encode()).hexdigest()
    assert got == {"digest": w.digest(), "needs": names, "sig": sig, "inv": []}


def test_existing_mechanisms_work_unchanged_with_generated_categories():
    """Bounties verified on hidden inputs, the goal feasibility check and the hidden test battery."""
    from crucible.goals.contracts import generate_goal
    w = _tiny(seed=8)
    cat = _cat_of(w, "allocate")
    w.system_call("admin.join", {"handle": "dev", "endowment": 10_000})
    org, firm = next((k, dict(f)) for k, f in w.state.scan("npc_firms") if f["need"] == cat.name)
    assert firm["tier"] == 0 and w.state.get("listings", firm["listing"])["need"] == cat.name
    acct = w.modules["bank"].primary_account(w.state, org)
    b = w.system_call("jobs.post_bounty", {"need": cat.name, "reward_cents": 50_000, "min_quality": 0.9,
                                           "account": acct}, as_principal=firm["owner"])["bounty"]
    ex = lambda t, **k: w.execute("p:dev", {"tool": t, "args": k})  # noqa: E731
    repo = ex("code.create_repo", name="sol").result["repo"]
    assert ex("code.commit", repo=repo, files={"weak.py": cat.tiers[0][1], "good.py": cat.tiers[-1][1]}).ok
    weak = ex("jobs.submit", bounty=b, repo=repo, target="weak:solve").result
    good = ex("jobs.submit", bounty=b, repo=repo, target="good:solve").result
    assert not weak["accepted"] and good["accepted"] and good["paid_cents"] == 50_000 and good["score"] > weak["score"]
    # hidden battery on the incumbent, in an isolated fork: its measured tier quality, and the world is untouched
    d0 = w.digest()
    q = battery_quality(w, firm["host"], n=6)
    assert abs(q - tier_quality(cat.name, 0)) < 0.25 and w.digest() == d0
    # goal feasibility: the 3.0 generator reads measured tier qualities of generated needs through the registry
    g = generate_goal("outperform", "ow_tiny", seed=8, level=1, split="dev", overrides={"n_consumers": 300})
    assert g.params["need"].startswith("gen.") and g.params["headroom"] >= 0.05 * 1.25 and not g.validate()
    assert w.check_invariants() == []
