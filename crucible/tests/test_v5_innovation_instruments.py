"""Negative controls and unit tests of the Gate J/L and SP-6.3b instruments (`crucible.innovation.gates`,
`experiments`, `civexp`, `measure`): each instrument must flag the manipulation or false confirmation it exists to
catch (repair pass of wave 3b; the verifier's probes p3-p6).

* single-judge scan: a language-model or shared-semantic-space call injected into any usefulness decision - directly,
  through `from ..semantic.space import ...`, through `social.audience` / `population.semantics`, or through a
  helper in another crucible module - is flagged; the real tree has none;
* L.5: a chain is confirmed only by the measured world outcome of its removal arm (the arm is chosen by the
  chain's task, not its name): a package chain whose step's hand work also falls without the inventions, a need
  chain the factual arm never relieved, and a competition chain whose incumbents lose and respond anyway are not
  confirmed;
* chain mining: a need is never "relieved" before it first arose;
* J.3/J.4/J.6/J.7: minima over every category, not only the passing ones;
* L.n1: an income credit booked next to a real deposit (same kind and memo, any amount) is counted as a bonus.
"""
import ast
import os

import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.innovation import civexp as CX  # noqa: E402
from crucible.innovation import experiments as X  # noqa: E402
from crucible.innovation import gates as G  # noqa: E402

ROOT = os.path.join(os.path.dirname(G.__file__), "..")
DAY = 86400


# ====================================================================== single-judge scan (SP-6.3b)
def _inject(rel: str, fn: str, stmts: list[str]) -> str:
    txt = open(os.path.join(ROOT, rel)).read()
    tree = ast.parse(txt)
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn)
    first = node.body[0]
    lines = txt.splitlines(True)
    lines.insert(first.lineno - 1, "".join(" " * first.col_offset + s + "\n" for s in stmts))
    return "".join(lines)


INJECTIONS = {
    "space_module": ["from ..semantic import space as _sp", "_sp.text_vector(None, 'is it useful?')"],
    "space_function": ["from ..semantic.space import text_vector as _tv", "_tv(None, 'is it useful?')"],
    "audience": ["from ..social import audience as _au", "_au.encode(None, 'is it useful?')"],
    "population_semantics": ["from ..population import semantics as _ps", "_ps.item_rep(None, 'ref')"],
    "lm_direct": ["from ..semantic.space import lm as _lm", "_lm.analyse(['is it useful?'])"],
    "external": ["import transformers as _tf", "_tf.pipeline('text-classification')"],
}


def test_the_real_tree_has_no_single_judge_usefulness_path():
    assert G.single_judge_paths(ROOT) == []
    # the developer's interest in a task reads the shared space: reported (not a usefulness verdict), and found
    other = G.single_judge_paths(ROOT, decisions=G.OTHER_DECISIONS)
    assert other and "text_vector" in other[0]


@pytest.mark.parametrize("kind", sorted(INJECTIONS))
def test_a_language_model_call_injected_into_a_participant_decision_is_flagged(kind):
    src = _inject("economy/work.py", "_accepted", INJECTIONS[kind])
    paths = G.single_judge_paths(ROOT, overrides={"economy/work.py": src})
    assert any(p.startswith("economy/work.py:_accepted") for p in paths), (kind, paths)


@pytest.mark.parametrize("rel,fn", [(rel, fn) for rel, fns in sorted(G.USEFULNESS_DECISIONS.items()) for fn in fns])
def test_the_scan_reaches_every_listed_decision(rel, fn):
    src = _inject(rel, fn, INJECTIONS["space_module"])
    assert any(p.startswith(f"{rel}:{fn}") for p in G.single_judge_paths(ROOT, overrides={rel: src}))


def test_a_judgement_hidden_in_a_helper_of_another_module_is_flagged():
    helper = "from ..semantic import space\n\n\ndef looks_useful(text):\n    return space.text_vector(None, text)[0] > 0\n"
    src = _inject("economy/work.py", "_accepted", ["from ..innovation import _probe_helper as _h",
                                                   "_h.looks_useful('x')"])
    paths = G.single_judge_paths(ROOT, overrides={"economy/work.py": src, "innovation/_probe_helper.py": helper})
    assert any("_probe_helper.py:looks_useful" in p for p in paths), paths
    # the same helper without the model call is not flagged (no false positive from following calls)
    clean = "def looks_useful(text):\n    return len(text) > 0\n"
    assert not G.single_judge_paths(ROOT, overrides={"economy/work.py": src, "innovation/_probe_helper.py": clean})


# ====================================================================== L.5 confirmations
def _hist(days: int, *, needs=None, rate=None, live=None) -> list[dict]:
    """Synthetic daily snapshots: `needs(d)` -> set of need ids, `rate(d)` -> hand-work minutes that day of an
    activity "x" (cumulative in the snapshot), `live(d)` -> {org: live subscriptions}."""
    out, cum = [], 0.0
    for d in range(days + 1):
        cum += float(rate(d)) if rate else 0.0
        out.append({"t": d * DAY, "needs": {n: {} for n in (needs(d) if needs else ())},
                    "families": {"x": {"diy_minutes": cum}}, "live_subs": dict(live(d) if live else {})})
    return out


def _arm(hist, needs_end=(), needs_ever=(), relying=None, actions=()):
    return {"history": hist, "world": None, "outcome": {"needs": sorted(needs_end), "needs_ever": sorted(needs_ever)},
            "summary": {"relying": dict(relying or {}), "rival_actions": list(actions)}}


def _pkg_chain(need="x"):
    stages = [{"name": "package p released", "subsystem": "software", "t": 2 * DAY},
              {"name": "independent developer's repository d9/r executes it", "subsystem": "software", "t": 3 * DAY},
              {"name": "service h declared", "subsystem": "innovation", "t": 3 * DAY},
              {"name": "individuals adopt the service", "subsystem": "work activities", "t": 5 * DAY},
              {"name": "hand work on the step falls", "subsystem": "work activities", "t": 8 * DAY}]
    return dict(X._chain("package:p->d9/r", stages), need=need, released_t=2 * DAY, adopted_t=5 * DAY)


def test_removal_arm_is_chosen_by_the_chains_task():
    assert G.chain_removal_arm(_pkg_chain("gen.transform.abc")) == "ban_novel"
    assert G.chain_removal_arm(_pkg_chain("schedule.jobs")) == "ban_published"
    assert G.chain_removal_arm(dict(_pkg_chain(), need=None)) is None


def test_a_package_chain_is_confirmed_only_when_hand_work_does_not_fall_without_the_inventions():
    falls = _hist(12, rate=lambda d: 100 if d < 7 else 30)
    flat = _hist(12, rate=lambda d: 100)
    ch = _pkg_chain("x")
    assert CX.hand_work_falls(falls, "x", ch["released_t"], ch["adopted_t"]) is not None
    for removal, expect in ((flat, True), (falls, False)):
        A = {"factual": _arm(falls), "ban_published": _arm(removal), "ban_novel": _arm(removal)}
        assert G.confirmed(A, ch) is expect
    # the removal arm has the very same chain under another name: irrelevant, the outcome decides
    A = {"factual": _arm(falls), "ban_published": dict(_arm(falls), chains=[dict(ch, chain="package:q->d7/r")])}
    assert G.confirmed(A, ch) is False


def _need_chain(nid="provide:capability/x", relieved=True):
    stages = [{"name": f"need {nid}", "subsystem": "work activities", "t": 2 * DAY},
              {"name": "public request for help", "subsystem": "social (forum)", "t": 3 * DAY},
              {"name": "agent artifact s", "subsystem": "innovation", "t": 4 * DAY, "creator_kind": "population"},
              {"name": "adoption by holders", "subsystem": "work activities", "t": 5 * DAY}]
    if relieved:
        stages.append({"name": "need relieved", "subsystem": "work activities", "t": 7 * DAY})
    return dict(X._chain(f"need:{nid}", stages), need="x")


def test_a_need_chain_needs_relief_in_the_factual_arm_and_persistence_without_the_invention():
    nid = "provide:capability/x"
    ch = _need_chain(nid)
    removal_persists = _arm(_hist(10), needs_end=[nid], needs_ever=[nid])
    removal_gone = _arm(_hist(10), needs_end=[], needs_ever=[nid])
    fact_relieved = _arm(_hist(10), needs_end=[], needs_ever=[nid], relying={"x": 12})
    fact_unrelieved = _arm(_hist(10), needs_end=[nid], needs_ever=[nid], relying={"x": 12})
    assert G.confirmed({"factual": fact_relieved, "ban_published": removal_persists}, ch)
    # relieved in the removal arm too (by something else): the relief did not depend on the invention
    assert not G.confirmed({"factual": fact_relieved, "ban_published": removal_gone}, ch)
    # the factual arm never relieved it at the end: nothing to confirm (verifier p4: gen.compose)
    assert not G.confirmed({"factual": fact_unrelieved, "ban_published": removal_persists}, ch)
    assert not G.confirmed({"factual": fact_relieved, "ban_published": removal_persists}, _need_chain(nid, False))


def test_a_competition_chain_is_not_confirmed_when_incumbents_lose_and_respond_anyway():
    stages = [{"name": "agent artifact s", "subsystem": "innovation", "t": 3 * DAY, "creator_kind": "population"},
              {"name": "incumbents lose subscribers", "subsystem": "commerce", "t": 6 * DAY},
              {"name": "incumbent responds (price or engineering)", "subsystem": "rivals", "t": 7 * DAY}]
    ch = dict(X._chain("competition:s", stages), need="schedule.jobs", owners=["o:inc"], artifact_t=3 * DAY)
    losing = _hist(10, live=lambda d: {"o:inc": 10 if d < 5 else 6})
    steady = _hist(10, live=lambda d: {"o:inc": 10})
    fact = _arm(losing, actions=[(7 * DAY, "o:inc", ["price"])])
    assert G.confirmed({"factual": fact, "ban_published": _arm(steady)}, ch)
    assert not G.confirmed({"factual": fact, "ban_published": _arm(losing, actions=[(8 * DAY, "o:inc", ["price"])])},
                           ch)


def test_a_need_is_never_relieved_before_it_arose():
    nid = "alternative:capability/x"
    hist = _hist(20, needs=lambda d: {nid} if 17 <= d <= 18 else set())
    # adoption on day 8: the days 9-16 without the need are not its relief (verifier p5: pack.knapsack)
    assert X.relief_t(hist, nid, 8 * DAY) == 19 * DAY
    stages = [{"name": "adoption of a a provider", "subsystem": "commerce", "t": 1 * DAY},
              {"name": f"need {nid}", "subsystem": "work activities", "t": 17 * DAY},
              {"name": "need relieved", "subsystem": "work activities", "t": 9 * DAY}]
    assert X._chain(f"need:{nid}", stages)["ordered"] is False


# ====================================================================== J minima over all categories
def test_j_minima_cover_every_category_not_only_the_passing_ones():
    def prod(cat, users, disc, tx, kept, tried):
        return {"category": cat, "kind": "service", "creator_kind": "population", "published": True,
                "executable": True, "discoverers": disc, "users": users, "tx": {"n": tx, "cents": 10 * tx, "payers": tx},
                "triers": [f"p{k}" for k in range(tried)], "kept": [f"p{k}" for k in range(kept)], "created_t": DAY}
    fam = {"a": {"diy_minutes": 10.0}, "b": {"diy_minutes": 10.0}}
    fam_removed = {"a": {"diy_minutes": 900.0}, "b": {"diy_minutes": 20.0}}

    def out(products, families):
        return {"products": products, "families": families, "dev_compute": {"cu": 1.0, "cores": 1}}
    good = {"s1": prod("a", 60, 70, 40, 50, 60), "s2": prod("b", 3, 4, 1, 1, 3)}
    garb = {"g1": prod("a", 60, 70, 0, 2, 60), "g2": prod("b", 3, 4, 0, 0, 3)}
    A = {"factual": {"outcome": out(good, fam)}, "garbage": {"outcome": out(garb, fam)},
         "ban_published": {"outcome": out({}, fam_removed)}, "ban_novel": {"outcome": out({}, fam_removed)}}
    J = G.j_measure(A)
    assert J["passing"] == ["a"]
    assert J["J.3"]["independent_discoverers"] == 4 and J["J.4"]["independent_users"] == 3
    assert J["J.6"]["independent_transactions"] == 1 and J["J.n1"]["product_categories"] == 1


# ====================================================================== L.n1 disguised bonus (verifier p3)
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_an_income_credit_booked_next_to_a_real_deposit_is_counted(monkeypatch):
    from crucible.core.kernel import World
    from crucible.innovation import inventworld as IW
    from crucible.innovation import measure as M
    from crucible.population import core
    from crucible.substrate import columns

    def run(plant: bool):
        w = World.create(IW.creation_spec(5, n=300), world_id=f"lnone-{int(plant)}")
        orig = core.money_deposit

        def planted(ctx, i):
            acct = orig(ctx, i)
            if acct and bool(columns.col(ctx.world, "devs", "dev")[int(i)]):
                ctx.world.modules["bank"].move(ctx, "acct_external", acct, 50_000, "income", "income deposited")
            return acct
        if plant:
            monkeypatch.setattr(core, "money_deposit", planted)
        try:
            w.advance_time(6 * DAY)
        finally:
            monkeypatch.setattr(core, "money_deposit", orig)
        devs = [f"p{i}" for i in np.flatnonzero(columns.col(w, "devs", "dev"))]
        r = M.special_case_bonuses(w, devs)
        planted_n = sum(1 for _, t in w.state.scan("txns") if t["kind"] == "income" and
                        any(d == 50_000 for _, d in t["postings"]))
        w.close()
        return r, planted_n
    clean, _ = run(False)
    assert clean["bonuses"] == 0 and clean["income_deposits"] > 0
    dirty, n_planted = run(True)
    assert n_planted > 0 and dirty["bonuses"] >= n_planted


# ====================================================================== designer-supplied demand (SP-6.4b, verifier p2)
def _families(spec):
    return next(m["config"]["families"] for m in spec.modules if m["name"] == "work")


def test_composed_activities_follow_the_world_rule_and_a_hand_picked_pair_is_flagged():
    from crucible.innovation import inventworld as IW
    rule = IW.creation_spec(5, n=300)
    assert [f["compose"] for f in _families(rule) if f.get("compose")]          # the rule gives one here
    assert G.configured_compositions(_families(rule)) == []
    # a world with three generated categories: the rule gives transform -> allocate AND transform -> dedupe
    three = ("allocate", "transform", "dedupe")
    assert len(IW.raw_feed_compositions(three, 5)) == 2
    assert G.configured_compositions(_families(IW.creation_spec(5, n=300, novel=three))) == []
    picked = IW.creation_spec(5, n=300, novel=three, compose=(("transform", "allocate"),))
    flagged = G.configured_compositions(_families(picked))
    assert len(flagged) == 1 and "allocate" in flagged[0]
    # the builder's two-step pipeline is reported as configured (its needs are not counted as SP-6.4b examples)
    assert G.configured_pipelines(_families(rule)) == ["route.shortest+pack.knapsack"]


@pytest.mark.timeout(600)
def test_the_demand_detector_counts_a_configured_combination_in_a_world():
    from crucible.core.kernel import World
    from crucible.innovation import inventworld as IW
    for compose, expect in ((None, 0), ((("transform", "allocate"),), 1)):
        w = World.create(IW.creation_spec(5, n=300, novel=("allocate", "transform", "dedupe"), compose=compose),
                         world_id=f"demand-{expect}")
        try:
            A = {a: {"history": [], "world": None} for a in CX.ARMS}
            A["factual"]["world"] = w
            d = G.designer_supplied_demand(A)
            assert d["designer_supplied_demand_paths"] == expect and d["configured_composed_activities"] == expect
        finally:
            w.close()


def test_a_developer_reads_the_stages_from_the_public_request_text_only():
    from crucible.innovation import devs as D
    from crucible.usefulness import category as C
    a, b = C.instantiate("transform", 501), C.instantiate("allocate", 500)
    comp = C.compose(a, b)
    body = f"I do this by hand and it takes me about 40 minutes a task. {comp.doc[:1500]}\nExample input: {{}}"
    assert D.stages_of(body) == (a.name, b.name, b.table_key)
    assert D.stages_of(f"I do this by hand. {b.doc[:1500]}") is None
