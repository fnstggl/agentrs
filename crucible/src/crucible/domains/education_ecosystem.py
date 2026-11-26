"""Tier 2 domain 9.10 Education ecosystem (workstream I-2): develop an educational product that measurably improves
participant learning.

The world (a restriction of the civilisation generator, D-17): the one population (`pop`) with its outside option
(`elsewhere`), the education subsystem (`learn`: knowledge state, study and teaching activities, the Board's
independent tests and certificates) and the information subsystem it runs on (documents, search, the forum). The
learners are population individuals with heterogeneous prior knowledge, goals, motivation, attention, contacts and
search skill; some of them write lessons from their own (fallible) knowledge; the Board of Education publishes a
textbook in some configurations. Nothing about who learns what is scripted: an individual studies when its habit
for studying (learned from its own sessions' outcomes) wins over the rest of its life, finds documents through its
own channels and learns from what they contain (`domains.learning`).

The participant (seat `educator`) acts with the ordinary tools of the world: documents (`docs.*`), the forum, search,
the public education statistics (`learn.stats`, `learn.readers`, `learn.curriculum`). It may publish anything; only
what independent learners choose to study and what it does to their knowledge counts.

Objective (public): independent learners study the participant's material and, on a delayed test of NEW problems,
improve more than comparable learners who did not. Evaluator (evaluator side, never a tool): the Board's placement
score is each learner's baseline; a delayed final test (7 days after the evaluation time, the evaluator's own random
stream, problems never published anywhere) measures retained knowledge; the gain of readers on the concepts the
material covers is compared with the gain of non-readers on the same concepts (one-sided Welch test). Time spent
reading is not learning: a document without worked problems teaches nothing, a document with wrong answers harms.

Configurations (structure measured on the built world: curriculum concepts and the shape of their prerequisite
graph, discovery channels, population authors, the incumbent textbook, staggered cohorts):

    train   numeracy_search     arithmetic to fraction addition (8 concepts), search only, population authors,
                                basic textbook
    train   algebra_forum       arithmetic to linear equations and means (6 concepts, a wide graph), forum and
                                contacts, no population authors, full textbook
    train   numtheory_cohorts   number theory (gcd, lcm, binary, modular powers), search and contacts, population
                                authors, no textbook, a second cohort enrolls on day 14
    eval_compositional  numeracy_cohorts   the numeracy curriculum with forum and contacts, population authors, the
                                basic textbook and staggered cohorts (every part seen in training, never together)
    eval_structural     fractions_percent  fraction multiplication and percentages (concepts and a prerequisite
                                shape no training world has), all channels, population authors, no textbook
"""
from __future__ import annotations

import math

import numpy as np

from ..core.canonical import digest

from . import base
from . import concepts as C
from .base import Actor, Domain, DomainConfig, Mechanic, Probe, ScriptedReference, Seat, act_

NAME = "education"
HORIZON_DAYS = 42.0
DELAY_DAYS = 7.0
N_POP = 600
TARGET = {1: {"readers": 15, "delta": 0.05}, 2: {"readers": 30, "delta": 0.08}, 3: {"readers": 60, "delta": 0.12}}

CONFIGS = {
    "numeracy_search": {"curriculum": ["frac_add"], "channels": ["search"], "authors": True, "textbook": "basic",
                        "cohort": None},
    "algebra_forum": {"curriculum": ["lin_eq", "mean"], "channels": ["forum", "contacts"], "authors": False,
                      "textbook": "full", "cohort": None},
    "numtheory_cohorts": {"curriculum": ["power_mod", "binary", "lcm"], "channels": ["search", "contacts"],
                          "authors": True, "textbook": None, "cohort": {"share": 0.5, "at_day": 14}},
    "numeracy_cohorts": {"curriculum": ["frac_add"], "channels": ["forum", "contacts"], "authors": True,
                         "textbook": "basic", "cohort": {"share": 0.5, "at_day": 14}},
    "fractions_percent": {"curriculum": ["percent", "frac_mul"], "channels": ["search", "forum", "contacts"],
                          "authors": True, "textbook": None, "cohort": None},
}
SPLIT = {"numeracy_search": "train", "algebra_forum": "train", "numtheory_cohorts": "train",
         "numeracy_cohorts": "eval_compositional", "fractions_percent": "eval_structural"}
DOCS = {
    "numeracy_search": "arithmetic to fraction addition (8 concepts, a chain-like prerequisite graph); learners "
                       "find documents only through search; individuals who know a concept write lessons; the "
                       "Board publishes a basic textbook (level-1 worked examples)",
    "algebra_forum": "arithmetic to linear equations and means (6 concepts, a wide prerequisite graph); learners find "
                     "documents through the study board of the forum and their contacts; nobody in the population "
                     "teaches; the Board publishes a full textbook (levels 1-2 with exercises)",
    "numtheory_cohorts": "number theory (gcd, lcm, binary, modular powers); search and contacts; population authors; "
                         "no textbook; half the population becomes able to enroll only on day 14",
    "numeracy_cohorts": "the numeracy curriculum found through the forum and contacts, population authors, the basic "
                        "textbook and a second cohort on day 14 - every structural part seen in training, never this "
                        "combination",
    "fractions_percent": "fraction multiplication and percentages (concepts and a prerequisite graph no training world "
                         "has) with every discovery channel, population authors and no textbook",
}


# ============================================================================================ world
def build(config, seed, overrides):
    cfg = dict(CONFIGS[config.name])
    learn = {"curriculum": list(cfg["curriculum"]), "channels": list(cfg["channels"]), "authors": bool(cfg["authors"]),
             "textbook": cfg["textbook"], "cohort": cfg["cohort"]}
    learn.update(overrides.get("learn") or {})
    pop = {"n": int(overrides.get("n", N_POP)), "dims": {"H": 16}}
    return base.civ_recipe(seed, subsystems=["education"], modules=["docs", "search", "forum", "elsewhere"],
                           population_modules=("pop",), module_config={"learn": learn, "pop": pop},
                           params=overrides.get("params"))


def tier3(config, merged):
    """In a civilisation the education subsystem keeps the configuration's curriculum and channels."""
    mine = build(config, 0, {})
    merged["modules"]["learn"] = {**merged["modules"].get("learn", {}), **mine["modules"]["learn"]}
    return merged


def structure(world):
    from .base import graph_shape
    L = world.modules["learn"]
    cur = list(L.cur)
    edges = [(p, c) for c in cur for p in C.PREREQS[c] if p in cur]
    cfg = L.cfg
    return {"concepts": sorted(cur), "prerequisites": graph_shape(cur, edges), "channels": sorted(cfg["channels"]),
            "population_authors": bool(cfg["authors"]), "textbook": str(cfg["textbook"] or "none"),
            "cohorts": bool(cfg.get("cohort"))}


# ============================================================================================ objective and evaluator
def objective(world, principal, difficulty):
    t = TARGET.get(int(difficulty), TARGET[3])
    return {"text": (f"Develop educational material (public documents) that measurably improves the learning of this "
                     f"world's learners: within {HORIZON_DAYS:.0f} days at least {t['readers']} independent learners "
                     f"must study it, and on a delayed test of new problems their improvement on the concepts it "
                     f"covers must exceed that of comparable learners who did not study it by at least "
                     f"{t['delta']:.2f} (share of problems answered correctly), with statistical significance "
                     f"(one-sided p <= 0.05)."),
            "readers": t["readers"], "delta": t["delta"], "window_days": HORIZON_DAYS, "delay_days": DELAY_DAYS,
            "difficulty": int(difficulty)}


def _welch_one_sided(a, b) -> tuple[float, float]:
    """(mean(a) - mean(b), one-sided p of a > b) by Welch's t-test."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return (float(a.mean() - b.mean()) if a.size and b.size else 0.0), 1.0
    r = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    p = float(r.pvalue) if np.isfinite(r.pvalue) else 1.0
    return float(a.mean() - b.mean()), p


def readers_of(world, principals) -> dict[int, set]:
    """Individuals who studied documents owned by `principals` (at least one item read): {individual: concepts}."""
    st = world.state
    out: dict[int, set] = {}
    for _, s in st.scan("learn_sessions"):
        if s["author"] in principals and s["usable"] > 0:
            out.setdefault(int(s["i"]), set()).update(s["concepts"])
    return out


def evaluate(world, principal, obj):
    from ..substrate import columns
    from .common import independent
    L = world.modules["learn"]
    mine = set(world.principals_of(principal)) | {principal}
    rd = {i: cs for i, cs in readers_of(world, mine).items() if independent(world, principal, f"p{i}")}
    covered = sorted(set().union(*rd.values())) if rd else []
    state = columns.col(world, "learn", "state")
    placed = columns.col(world, "learn", "placed")
    enrolled = np.flatnonzero(state > 0)
    details = {"readers": len(rd), "covered_concepts": covered, "enrolled": int(enrolled.size)}
    if not rd or not covered:
        return {"success": False, "score": 0.0, "details": {**details, "effect": 0.0, "p_value": 1.0}}
    at = int(world.now + DELAY_DAYS * 86400)
    final = L.final_scores(world, at, "education-evaluator", k=4, who=enrolled, concepts=covered)
    kc = [L.cix[c] for c in covered]

    def gain(i):
        base = placed[i, kc]
        ok = base >= 0
        if not ok.any():
            return None
        f = np.array([final[int(i)][c] for c in covered])
        return float(np.mean(f[ok] - base[ok]))
    g_read = [g for i in sorted(rd) if (g := gain(i)) is not None]
    g_ctrl = [g for i in enrolled.tolist() if i not in rd and (g := gain(i)) is not None]
    eff, p = _welch_one_sided(g_read, g_ctrl)
    need = obj.get("readers", TARGET[1]["readers"])
    delta = float(obj.get("delta", TARGET[1]["delta"]))
    ok = len(rd) >= need and eff >= delta and p <= 0.05
    score = min(1.0, len(rd) / need) * 0.5 + 0.5 * max(0.0, min(1.0, eff / delta)) * (1.0 if p <= 0.05 else 0.5)
    return {"success": bool(ok), "score": round(float(score), 4),
            "details": {**details, "effect": round(eff, 4), "p_value": p, "readers_gain": round(float(np.mean(g_read)), 4)
                        if g_read else None, "control_gain": round(float(np.mean(g_ctrl)), 4) if g_ctrl else None,
                        "n_readers_measured": len(g_read), "n_controls": len(g_ctrl)}}


# ============================================================================================ policies
def _order(concepts: list[dict]) -> list[str]:
    """Concepts in prerequisite order (prerequisites first), from the public curriculum."""
    pre = {c["name"]: [p for p in c["prerequisites"]] for c in concepts}
    out: list[str] = []

    def add(n):
        if n in out:
            return
        for p in pre.get(n, []):
            add(p)
        out.append(n)
    for c in sorted(pre):
        add(c)
    return out


def course_text(c: str, rng, *, wrong: bool = False, prose: bool = False, next_doc: str | None = None) -> str:
    """A careful course chapter: worked examples before exercises at each level, spaced over three levels, answers
    from the curriculum's arithmetic (or, for a degenerate probe, the typical misconception / prose only)."""
    con = C.LIBRARY[c]
    lines = [f"{con.title} ({c}) - a complete course chapter", "",
             f"{con.doc} We write problems as `{con.fmt}`; answers: {con.answer_fmt}.", ""]
    if prose:
        lines += [f"{con.title} matters in everyday life. Take your time, think about what each step means and why "
                  f"it works. Practise regularly, review what you learned a few days later, and do not be "
                  f"discouraged by mistakes: everybody makes them when learning {c.replace('_', ' ')}."] * 6
    else:
        plan = [("example", 1), ("example", 1), ("practice", 1), ("example", 2), ("example", 2), ("practice", 2),
                ("practice", 2), ("example", 3), ("practice", 3), ("practice", 3), ("practice", 1), ("practice", 2)]
        for n, (kd, L) in enumerate(plan, 1):
            p = C.generate(c, rng, L)
            a = con.solve(p)
            if wrong:
                b = con.misconception(p)
                a = b if b is not None else con.solve(C.generate(c, rng, L))
            lines += C.write_item(kd, con, p, a, "arrow", n)
    if next_doc:
        lines += ["", f"Next: {next_doc}"]
    return "\n".join(lines)


class CourseReference(ScriptedReference):
    """Tool-only reference (labelled scripted in evidence): read the public curriculum, write one careful chapter per
    concept in prerequisite order (each chapter links to the next), announce the chapters on the study board, then
    watch the public statistics and revise the chapters learners read least. It solves its own problems with the
    curriculum's arithmetic; it reads nothing but tool results."""
    burst = 8
    replan_s = 2 * 86400
    variant = "correct"

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.concepts = None
        self.board = "study"
        self.docs: dict[str, str] = {}
        self.order: list[str] = []
        self.rng = np.random.default_rng(int(digest(str(principal))[:8], 16))
        self.posted = False
        self.round = 0

    def text(self, c, nxt):
        return course_text(c, self.rng, wrong=self.variant == "wrong", prose=self.variant == "prose",
                           next_doc=nxt)

    def plan(self, obs):
        if self.concepts is None:
            return [act_("learn.curriculum", "curriculum")]
        if not self.docs:
            acts = []
            for c in reversed(self.order):          # deepest first, so each chapter can link to the next one
                acts.append(act_("docs.create", f"doc:{c}", title=f"{C.LIBRARY[c].title} ({c}) - complete course: "
                                 f"worked examples and exercises", body=self.text(c, None), visibility="public"))
            return acts
        if not self.posted and len(self.docs) == len(self.order):
            self.posted = True
            acts = []
            for k, c in enumerate(self.order):
                nxt = self.docs.get(self.order[k + 1]) if k + 1 < len(self.order) else None
                acts.append(act_("docs.edit", None, doc=self.docs[c], title=f"{C.LIBRARY[c].title} ({c}) - "
                                 f"complete course: worked examples and exercises", body=self.text(c, nxt)))
                acts.append(act_("forum.post", None, board=self.board, title=f"{C.LIBRARY[c].title} ({c}): a free "
                                 f"course chapter {self.docs[c]}", body=f"Worked examples and exercises for {c} at "
                                 f"every level: {self.docs[c]}. Next chapter linked at the end.", tags=["lesson"]))
            return acts
        self.round += 1
        acts = [act_("learn.readers", "readers")]
        if self.round % 3 == 0:                     # re-announce on the board now and then
            c = self.order[self.round % len(self.order)]
            acts.append(act_("forum.post", None, board=self.board, title=f"{C.LIBRARY[c].title} ({c}) course "
                             f"{self.docs[c]}", body=f"Still free: {self.docs[c]} - worked examples and exercises "
                                                     f"for {c}.", tags=["lesson"]))
        return acts

    def on_result(self, tag, result, action):
        if result is None:
            return
        if tag == "curriculum":
            self.concepts = result["concepts"]
            self.board = result.get("board", "study")
            self.order = _order(self.concepts)
        elif tag and tag.startswith("doc:"):
            self.docs[tag[4:]] = result["doc"]


class WrongKeys(CourseReference):
    """Degenerate: the reference's course with the typical misconception as every stated answer."""
    variant = "wrong"


class ProseOnly(CourseReference):
    """Degenerate: long encouraging prose with no worked problems (time spent reading is not learning)."""
    variant = "prose"


# ============================================================================================ mechanics and actors
def _sessions_since(world, since_seq):
    for e in world.events_range(int(since_seq), world.seq):
        if e.type == "learn.sessions":
            yield from e.payload["sessions"]


def _prereq_count(world, since_seq):
    n = sum(1 for s in _sessions_since(world, since_seq) if s.get("prereq_limited", 0) > 0)
    for e in world.events_range(int(since_seq), world.seq):     # prerequisite-first study decisions
        if e.type == "learn.choices":
            L = world.modules["learn"]
            for i, kind, c in e.payload["choices"]:
                if kind == "study" and c is not None:
                    g = int(_goal_at(world, i))
                    if g >= 0 and L.cur[g] != c:
                        n += 1
    return n


def _goal_at(world, i):
    from ..substrate import columns
    return columns.col(world, "learn", "goal")[int(i)]


def _retention_count(world, since_seq):
    return sum(1 for s in _sessions_since(world, since_seq) if s.get("forgot", 0) > 0.005 or s.get("consolidated", 0))


MECHANICS = (
    Mechanic("Learner knowledge state", ("cols.batch:learn.placed", "cols.batch:learn.assessment"),
             "per-individual mastery, floor, half-life and misconceptions (columns of table learn) measured by tests"),
    Mechanic("Educational artifacts", ("learn.sessions", "learn.authored"),
             "public documents read in study sessions; lessons written by population authors"),
    Mechanic("Prerequisites", (), "study limited by unmastered prerequisites; prerequisite-first concept choices",
             count=_prereq_count),
    Mechanic("Participation", ("learn.choices", "learn.rated"),
             "the population's own study decisions (what, where, whether to rate, take exams)"),
    Mechanic("Learning dynamics", ("cols.batch:learn.study",), "mastery and misconceptions changed by what was read"),
    Mechanic("Independent assessments", ("learn.assessed", "learn.summary", "learn.certified"),
             "the Board's placement, periodic and certification tests on fresh unpublished problems"),
    Mechanic("Retention of acquired knowledge", (), "forgetting measured when a concept is studied again; spaced "
             "retrieval consolidating the stable floor", count=_retention_count),
)

ACTORS = (
    Actor("learners", ("cols.batch:learn.study", "cols.batch:learn.placed"), "population individuals studying"),
    Actor("population_authors", ("learn.authored",), "individuals writing lessons from their own knowledge"),
    Actor("board_of_education", ("learn.summary",), "the Board's periodic independent assessment"),
)


def _configs():
    return tuple(DomainConfig(n, SPLIT[n], {k: (v if not isinstance(v, dict) else sorted(v)) for k, v in
                                            CONFIGS[n].items()}, DOCS[n]) for n in CONFIGS)


DOMAIN = Domain(
    name=NAME, title="Education ecosystem", spec_part="9.10", required_mechanics=(), configs=_configs(),
    build=build, objective=objective, evaluate=evaluate, reference=CourseReference, mechanics=MECHANICS,
    seats=(Seat("educator", 200_000, "Educator", "an author of educational material"),), actors=ACTORS,
    degenerate=(Probe("wrong_keys", "the reference's course with misconceptions as answers", WrongKeys),
                Probe("prose_only", "encouraging prose without worked problems", ProseOnly)),
    structure=structure, tier3=tier3, mode="C", horizon_days=HORIZON_DAYS, slowest_effect_days=28.0, dt=6 * 3600,
    reference_seeds=(0, 1), reference_configs=("numeracy_search",), sources=("learning.py", "concepts.py"),
    doc=__doc__)

base.register(DOMAIN)
