"""Education on the population (domains.learning, domains.concepts; SP-9.10, DEF-I-2)."""
import numpy as np
import pytest

import crucible
from crucible.core.kernel import World, WorldSpec
from crucible.domains import concepts as C
from crucible.domains import learning as L
from crucible.substrate import columns

crucible.load_all_modules()


def test_misconceptions_never_grade_correct_and_lowest_terms_are_required():
    rng = np.random.default_rng(0)
    for name, con in C.LIBRARY.items():
        for lvl in (1, 2, 3):
            for _ in range(150):
                p = C.generate(name, rng, lvl)
                assert con.grade(con.solve(p), p), (name, p)
                b = con.misconception(p)
                assert b is None or not con.grade(b, p), (name, p, b)
    fs = C.LIBRARY["frac_simplify"]
    assert not fs.grade("12/18", (12, 18)) and not fs.grade("24/36", (12, 18)) and fs.grade("2/3", (12, 18))
    assert C.LIBRARY["div"].misconception((56, 8)) is not None          # exact division: the bug still bites


def test_reading_model_finds_worked_problems_in_ordinary_writing_and_ignores_prose():
    cur = list(C.LIBRARY)
    body = ("Fractions\n1. 12 + 7 = 19\nQ: What is 3/4 + 1/6?\nA: 11/12\nExercise: simplify 6/8 (answer: 3/4)\n"
            "- gcd(12, 18) is 6\nFractions are parts of a whole; practise every day.\nNext: doc_000004\n")
    r = C.parse_lesson(body, cur)
    assert [(i["concept"], i["correct"]) for i in r["items"]] == [("add", True), ("frac_add", True),
                                                                   ("frac_simplify", True), ("gcd", True)]
    assert r["next"] == "doc_000004"
    assert C.parse_lesson("Read carefully and think about fractions every day.", cur)["items"] == []


def _rec(base_m):
    cur = C.closure(["div"])
    cix = {c: k for k, c in enumerate(cur)}
    m = np.array([[b, 0.75 * b, 0.5 * b] for b in base_m])
    return L.Rec(m.copy(), m * 0.85, np.full(len(cur), 8.0), np.zeros(len(cur), np.int64), np.zeros(len(cur)), cur,
                 cix)


def test_exam_matches_the_learner_model_and_excludes_published_problems():
    rng = np.random.default_rng(1)
    rec = _rec([0.9, 0.7, 0.7, 0.4])
    s = [L.assess(rec, 0, rng, 40, ["div"], set())[0]["div"] for _ in range(3)]
    assert 0.15 < float(np.mean(s)) < 0.8


def test_tests_draw_no_published_problem(monkeypatch):
    """Placement, periodic and certification tests (`assess`) and the evaluator's papers (`exam`) draw fresh problems
    outside the published ones: with every problem of a first paper published, a second paper drawn from the same
    random stream shares none of them (the earlier test always passed an empty exclusion set)."""
    papers = []
    real = L.digest
    monkeypatch.setattr(L, "digest", lambda keys: (papers.append(list(keys)), real(keys))[1])
    rec = _rec([0.9, 0.7, 0.7, 0.4])
    L.assess(rec, 0, np.random.default_rng(5), 12, ["div"], set())
    published = set(papers[-1])
    _, n, dig = L.assess(rec, 0, np.random.default_rng(5), 12, ["div"], published)
    assert n == len(papers[-1]) == 36 and published and not (set(papers[-1]) & published)
    assert dig != real(sorted(published)) and dig != real(papers[-2])
    w = _world(n=40)
    w.advance_time(86400)
    L_ = w.modules["learn"]
    cs = [c for c in L_.cur if c == "div"] or L_.cur[:1]
    L.exam(w, np.arange(10), cs, w.now, np.random.default_rng(9), 6, set(), L_.cix)
    first = set(papers[-1])
    L.exam(w, np.arange(10), cs, w.now, np.random.default_rng(9), 6, first, L_.cix)
    assert first and not (set(papers[-1]) & first)


def _world(n=150, seed=3, **learn):
    mods = [{"name": m, "config": {}} for m in ["identity", "bank", "admin", "artifacts", "cols", "space", "search",
                                                 "forum", "docs"]]
    mods.append({"name": "pop", "config": {"profile": "development", "n": n,
                                           "classes": {"adaptive": 0, "strategic": 0, "semantic": 0}}})
    mods.append({"name": "elsewhere", "config": {}})
    mods.append({"name": "learn", "config": dict(learn)})
    return World.create(WorldSpec("edu-test", seed, mods, {"cost_model": "deterministic"}))


def test_the_population_studies_teaches_and_is_assessed_with_clean_invariants_and_replay():
    w = _world()
    w.advance_time(10 * 86400)
    st = w.state
    assert len(st.tables["learn_sessions"]) > 50
    assert int((columns.col(w, "learn", "state") > 0).sum()) > 20
    assert any(e.type == "learn.summary" for e in w.log)
    assert not [e for e in w.log if e.type == "proc.failed"]
    assert w.check_invariants() == []
    w2 = _world()
    w2.advance_time(10 * 86400)
    assert w2.digest() == w.digest()


def test_learn_refuses_a_second_population():
    with pytest.raises(Exception):
        _world(n=20, n_learners=5)
    from crucible.core.errors import ProtocolError
    with pytest.raises(ProtocolError):
        L.Learning({"n": 40})
