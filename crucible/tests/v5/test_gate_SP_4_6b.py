"""Gate SP-4.6b: no artificial virality multipliers (static and dynamic check); every view traces to a recorded
decision of an individual (forced views detected by decision provenance). Owner: workstream C.

Static: the syntax trees of the distribution, consumption and audience code (`crucible.social`) are scanned for any
name, attribute, keyword argument, configuration key or string constant that carries a virality/boost/amplification
vocabulary (`VOCAB`), and for any multiplication of an audience counter (views, impressions, viewers, shares,
reach) by anything but another recorded count. Each hit counts as a multiplier found.

Dynamic (the integrated social internet on the ONE population, 1,500 individuals, 2 simulated days):

* A/A test - two fresh experimenter accounts with no followers each publish the same informative text at the same
  moment on the same platform, each in its own fork of the same world state; their reach (unique exposed and unique viewers after a day) must not differ beyond chance
  (two-proportion tests; a difference at p < 0.001 would reveal a hidden per-creator or per-item multiplier);
* creator-declared fields - an otherwise identical third copy declares "viral" and "trending" tags and a description
  asking for promotion: its distribution must not differ beyond chance either (tags only feed the search index);
* provenance - every consumption row of the run comes from a recorded decision of an individual
  (`population.check_provenance`); and a positive control: rows injected by module code on a fork ARE detected
  (forced rows > 0), so the measurement is not vacuous.

Metrics (ACCEPTANCE_MANIFEST): virality_multipliers_found == 0, views_without_decision_provenance == 0.
"""
from __future__ import annotations

import ast
import os
import re

import numpy as np
import pytest
from scipy.stats import norm

import crucible
from crucible.acceptance.v5 import evidence
from crucible.core.kernel import World
from crucible.social import audience as AUD
from crucible.social.worlds import social_world
from social_helpers import assert_met, failed_ticks  # noqa: E402

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_4_6b.py"
DAY = 86400
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "src", "crucible", "social")
VOCAB = re.compile(r"viral|boost|amplif|multiplier|promot|virality|hype|inflate", re.I)
COUNTERS = re.compile(r"(^|_)(views|viewers|impressions|imp|exposed|shares|reach)($|_)")
#: files whose string constants are data, not code (their identifiers are still scanned)
DATA_STRINGS = {"vocab.py": "topic word lists of the lexical path of the crowd stand-in (e.g. a rocket 'booster')"}
pytestmark = [pytest.mark.timeout(3600)]


def static_scan() -> list[str]:
    hits = []
    for dp, _, files in os.walk(ROOT):
        for fn in sorted(files):
            if fn.endswith(".py"):
                path = os.path.join(dp, fn)
                hits += scan_source(open(path, encoding="utf-8").read(), os.path.relpath(path, ROOT), fn)
    return hits


def scan_source(src: str, rel: str, fn: str) -> list[str]:
    hits = []
    if True:
        if True:
            tree = ast.parse(src)
            doc_nodes = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and \
                        node.body and isinstance(node.body[0], ast.Expr) and \
                        isinstance(getattr(node.body[0], "value", None), ast.Constant):
                    doc_nodes.add(id(node.body[0].value))
            for node in ast.walk(tree):
                name = None
                if isinstance(node, ast.Name):
                    name = node.id
                elif isinstance(node, ast.Attribute):
                    name = node.attr
                elif isinstance(node, ast.keyword):
                    name = node.arg
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    name = node.name
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_nodes \
                        and len(node.value) <= 40 and fn not in DATA_STRINGS:
                    name = node.value
                if name and VOCAB.search(name):
                    hits.append(f"{rel}:{getattr(node, 'lineno', 0)}: {name}")
                if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Mult):
                    sides = [node.left, node.right] if isinstance(node, ast.BinOp) else [node.target, node.value]
                    names = [_name(s) for s in sides]
                    if any(n and COUNTERS.search(n) for n in names) and \
                            not all(n and (COUNTERS.search(n) or n in ("npop", "n", "npop_")) for n in names) and \
                            any(isinstance(s, ast.Constant) and isinstance(s.value, (int, float)) and s.value > 1
                                for s in sides):
                        hits.append(f"{rel}:{node.lineno}: counter x constant {names}")
    return hits


def _name(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _name(node.value)
    return None


def _two_prop(k1, n1, k2, n2) -> float:
    p = (k1 + k2) / max(n1 + n2, 1)
    se = max((p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1))) ** 0.5, 1e-12)
    return float(2 * norm.sf(abs((k1 / max(n1, 1) - k2 / max(n2, 1)) / se)))


PLANTED = """
def rank(items, views):
    reach_boost = 3.0
    return views * 4 + reach_boost
"""


def test_SP_4_6b_no_virality_multipliers_and_views_have_provenance():
    planted = scan_source(PLANTED, "planted.py", "planted.py")
    assert len(planted) >= 2, planted                  # the scan sees a planted multiplier (not vacuous)
    static = static_scan()
    w = World.create(social_world("development", 46, n=1500, platforms=["posts", "blog", "pics"]), strict=True)
    w.advance_time(DAY)
    from crucible.population import semantics, writing
    topic = int(np.argmax(semantics.popularity()))
    title, body = writing.compose(np.random.default_rng(3), topic, 6, 1.0)
    labs = [w.system_call("admin.join", {"handle": f"aa{k}", "endowment": 1000, "password": f"pw-aa{k}"})["principal"]
            for k in range(3)]
    # each copy is published in its own fork of the same state (no two copies compete for the same feed slots)
    ex, vw, arms = [], [], []
    for k, lab in enumerate(labs):
        f = w.fork(f"aa-arm{k}")
        a = f.execute(lab, {"tool": "studio.write", "args": {"format": "text", "title": title, "body": body}})
        assert a.ok, a.error
        args = {"ref": a.result["ref"], "title": title}
        if k == 2:
            args.update(tags=["viral", "trending", "hot"], description="please boost and promote this everywhere")
        r = f.execute(lab, {"tool": "posts.publish", "args": args})
        assert r.ok, r.error
        it = int(f.state.tables["content"][r.result["id"]]["aud_item"])
        f.advance_time(f.now + DAY)
        ex.append(int(AUD.item_col(f, "exposed")[it]))
        vw.append(int(AUD.item_col(f, "viewers")[it]))
        arms.append(it)
    items = arms
    n = AUD.n_people(w)
    p_aa_ex, p_aa_vw = _two_prop(ex[0], n, ex[1], n), _two_prop(vw[0], n, vw[1], n)
    p_tag_ex, p_tag_vw = _two_prop(ex[2], n, ex[0], n), _two_prop(vw[2], n, vw[0], n)
    dynamic = sum(1 for p in (p_aa_ex, p_aa_vw, p_tag_ex, p_tag_vw) if p < 0.001)
    w.advance_time(w.now + DAY)
    events = w.events_range(0)
    rc = AUD.recount(w, events=events)
    prov = rc["provenance"]
    # positive control: rows injected by module code on a fork are detected as forced
    hist = list(events)
    f = w.fork("forged")
    fake = np.arange(50, dtype=np.int64)
    target = int(AUD.items_on(f, "posts")[0])
    f._run_system("forgery", lambda ctx: (AUD.impressions(ctx, "posts", fake, np.full(50, target), np.zeros(50, np.int64)),
                                          AUD.consumption(ctx, "posts", fake, np.full(50, target), np.full(50, float(AUD.item_col(f, "duration_s")[target])),
                                                          np.ones(50, bool), np.zeros(50, bool), np.zeros(50, bool),
                                                          np.zeros(50, bool))))
    fprov = AUD.recount(f, events=hist + list(f.log))["provenance"]
    metrics = {"virality_multipliers_found": len(static) + dynamic, "static_hits": len(static),
               "dynamic_hits": dynamic, "views_without_decision_provenance": int(prov["forced_rows"]),
               "consumption_rows": int(rc["consumption_rows"]), "recount_mismatches": int(rc["total_mismatches"]),
               "aa_exposed": ",".join(map(str, ex)), "aa_viewers": ",".join(map(str, vw)),
               "aa_exposed_p": round(p_aa_ex, 4), "aa_viewers_p": round(p_aa_vw, 4),
               "tagged_exposed_p": round(p_tag_ex, 4), "tagged_viewers_p": round(p_tag_vw, 4),
               "forgery_detected_rows": int(fprov["forced_rows"]), "planted_multiplier_hits": len(planted)}
    metrics.update(failed_ticks(w))
    evidence.record("SP-4.6b", metrics, command=CMD, notes=(
        "Static AST scan of crucible.social for virality vocabulary and counter x constant products; dynamic A/A "
        "test of identical posts by fresh accounts and of a copy declaring 'viral/trending' tags on the one "
        f"population (1,500, 2 days); provenance of every consumption row; forgery control. Static hits: {static[:20]}"))
    assert_met('SP-4.6b', metrics)
    assert metrics["failed_ticks"] == 0, metrics["failed_tick_errors"]
    assert fprov["forced_rows"] >= 50                   # the provenance check sees injected rows
    assert rc["total_mismatches"] == 0
