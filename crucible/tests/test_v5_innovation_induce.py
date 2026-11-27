"""Developers learn undocumented rules from worked examples (FR-DEV-1, wave 3b repair pass; verifier finding: "each
program is a designer reference tier; the best tier implements the generated category's hidden rule").

An exemplified category's rule (here: the world's undocumented wire formats of partner feeds) is shown only through
worked examples, which requesters post with their requests (`economy.work.Work.worked_example`). A developer induces
the rule from the examples it read (`innovation.induce`) and writes its program from what it induced: the program
carries no part of the hidden rule the examples did not show, its quality follows the evidence and the conventions
the developer knows, and the developer measures versions on its own examples in the sandbox."""
import ast
import inspect

import numpy as np
import pytest

from crucible.economy.work import Work
from crucible.innovation import induce as I
from crucible.usefulness import category as C
from crucible.usefulness import families  # noqa: F401  (registers the families)

SEEDS = (301, 501, 701)


@pytest.fixture(autouse=True)
def _registered():
    """The categories of these tests registered as needs (as a world does), removed afterwards."""
    names = []
    real = C.instantiate

    def inst(family, seed, params=None):
        cat = real(family, seed, params)
        C.register_category(cat)
        names.append(cat.name)
        return cat
    C.instantiate = inst
    try:
        yield
    finally:
        C.instantiate = real
        C.unregister_generated(names)


def _run(src: str, inp: dict):
    ns = {"__name__": "candidate"}
    exec(compile(src, "<candidate>", "exec"), ns)
    return ns["solve"](inp, None)


def _worked(cat, rng, n):
    out = []
    for _ in range(n):
        inp, hid = cat.generate(rng, float(rng.uniform(0.2, 0.5)))
        out.append({"input": inp, **Work.worked_example(cat.name, inp, hid)})
    return out


def _quality(cat, src, rng, n=30):
    tests = [cat.generate(rng, float(rng.uniform(0.2, 0.5))) for _ in range(n)]
    return float(np.mean([cat.score(inp, hid, _run(src, inp)) for inp, hid in tests]))


ALL = I.unusual_known(np.random.default_rng(0), 1.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_an_induced_decoder_is_as_good_as_the_evidence_and_the_conventions_known(seed):
    cat = C.instantiate("transform", seed)
    rng = np.random.default_rng(seed)
    worked = _worked(cat, rng, 3)
    assert I.induce(cat.name, [], ALL) is None                      # nothing to learn from: no program
    full = I.induce(cat.name, worked, ALL)
    assert full is not None and not full["unknown"]
    assert _quality(cat, full["source"], np.random.default_rng(seed + 1)) >= 0.95
    # knowing no unusual convention: what the examples show in those conventions stays unknown (generic parser,
    # unseen feeds) - the program is worse exactly when the world's formats use them
    plain = I.induce(cat.name, worked, [])
    q_plain = _quality(cat, plain["source"], np.random.default_rng(seed + 1)) if plain else 0.0
    if plain is None or plain["unknown"]:
        assert q_plain < 0.95
    else:
        assert q_plain >= 0.95


def test_the_program_holds_only_what_the_examples_showed():
    cat = C.instantiate("transform", 501)
    rng = np.random.default_rng(5)
    worked = _worked(cat, rng, 2)
    seen = {r["src"] for w in worked for r in w["input"]["rows"]}
    # drop every row of one feed: the developer never saw that feed
    hide = sorted(seen)[0]
    for w in worked:
        keep = [k for k, r in enumerate(w["input"]["rows"]) if r["src"] != hide]
        w["input"] = {"rows": [w["input"]["rows"][k] for k in keep]}
        w["output"] = {"records": [w["output"]["records"][k] for k in keep]}
    r = I.induce(cat.name, worked, ALL)
    assert set(r["rule"]) == seen - {hide}
    inp, _hid = cat.generate(rng, 0.5)
    out = _run(r["source"], inp)["records"]
    assert all(rec == {} for rec, row in zip(out, inp["rows"]) if row["src"] == hide)
    assert any(rec for rec, row in zip(out, inp["rows"]) if row["src"] != hide)
    # it is not one of the reference solvers (which carry the whole hidden rule)
    assert all(r["source"] != src for _, src in cat.tiers)


def test_induction_never_reads_a_categorys_hidden_parts():
    """Static: the induction module imports no category, family, evaluator or needs code and touches no hidden
    field (rule, tiers, truth, scorer, generator) - its only inputs are worked examples and the conventions known."""
    tree = ast.parse(inspect.getsource(I))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
        {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert mods <= {"__future__", "json", "re", "string"}, mods
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"rule", "tiers", "truth", "score_raw", "generate_raw", "score", "generate", "category_of",
                        "render_table", "example_extra"}


def test_requests_show_hand_made_results_only_for_tasks_known_through_worked_examples():
    rng = np.random.default_rng(1)
    t = C.instantiate("transform", 901)
    inp, hid = t.generate(rng, 0.3)
    assert Work.worked_example(t.name, inp, hid) == {"output": t.truth(inp, hid)}
    a = C.instantiate("allocate", 900)                              # documented rule: no worked example
    inp_a, hid_a = a.generate(rng, 0.3)
    assert Work.worked_example(a.name, inp_a, hid_a) is None
    assert Work.worked_example("route.shortest", {"x": 1}, {"y": 2}) is None
    c = C.compose(t, a)                                             # two stages: the first stage's result
    C.register_category(c)
    inp_c, hid_c = c.generate(rng, 0.3)
    wk = Work.worked_example(c.name, inp_c, hid_c)
    assert set(wk) == {"stage1_output"} and wk["stage1_output"]["records"] == hid_c["b_in"][a.table_key]
    # and those pairs are what the developer learns the first stage from
    pairs = I.pairs_of(t.name, [{"input": inp_c, **wk}])
    C.unregister_generated([c.name])
    assert len(pairs) == 1 and pairs[0]["input"] == inp_c["stage1"]


def test_the_bench_scores_a_candidate_against_hand_made_results():
    cat = C.instantiate("transform", 301)
    rng = np.random.default_rng(2)
    worked = _worked(cat, rng, 2)
    pairs = I.pairs_of(cat.name, worked)
    good = I.induce(cat.name, worked, ALL)["source"]
    ns = {"__name__": "bench"}
    exec(compile(I.bench_source(good, pairs), "<bench>", "exec"), ns)
    assert ns["score"]({}, None)["accuracy"] == 1.0
    blank = "def solve(args, caller):\n    return {'records': [{} for _ in args['rows']]}\n"
    ns = {"__name__": "bench"}
    exec(compile(I.bench_source(blank, pairs), "<bench>", "exec"), ns)
    assert ns["score"]({}, None)["accuracy"] == 0.0
    assert I.bench_source(good, []) is None
