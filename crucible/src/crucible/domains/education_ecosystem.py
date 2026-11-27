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
know measurably more than they would have without it. Evaluator (evaluator side, never a tool): a delayed final test
(7 days after the evaluation time, the evaluator's own random stream, problems never published in either world)
measures retained knowledge; each reader's score on the concepts it studied in the material is compared with the
SAME individual's score in the counterfactual world without the material (the same instance with the seat idle, on
the same paper with the same answer draws), by a one-sided paired t-test - the same individuals, so the comparison
has no self-selection. Time spent reading is not learning: a document without worked problems teaches nothing, a
document with wrong answers harms, material that displaces better material than itself scores below zero.

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
from ..goals.metrics import metric as goal_metric
from .base import Actor, Domain, DomainConfig, GoalFamily, Mechanic, Probe, ScriptedReference, Seat, act_

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
    """Measured on the built world (a `base.BuiltWorld`): the education system's built row (curriculum, channels,
    population authors, textbook, cohorts - `learn.meta` at genesis), the prerequisite graph of its curriculum and the
    Board's textbook chapters actually published."""
    from .base import graph_shape
    st = world.state
    sysrow = st.get("learn_world", "system") or {}
    cur = list(sysrow.get("concepts") or [])
    edges = [(p, c) for c in cur for p in C.PREREQS[c] if p in cur]
    chapters = sum(1 for _, r in st.scan("documents") if "textbook chapter" in str(r.get("title", ""))) \
        if "documents" in st.tables else 0
    return {"concepts": sorted(cur), "prerequisites": graph_shape(cur, edges),
            "channels": sorted(sysrow.get("channels") or []), "population_authors": bool(sysrow.get("authors")),
            "textbook": str(sysrow.get("textbook") or "none"), "textbook_published": chapters > 0,
            "cohorts": bool(sysrow.get("cohort"))}


# ============================================================================================ objective and evaluator
def _draw(rng, difficulty, world):
    """The goal generator's draw: how many independent readers and how large a causal effect on new problems."""
    t = TARGET.get(int(difficulty), TARGET[3])
    return {"readers": int(round(t["readers"] * float(rng.uniform(0.9, 1.1)))), "delta": float(t["delta"]),
            "window_days": HORIZON_DAYS, "delay_days": DELAY_DAYS}


def _text(p):
    return (f"Develop educational material (public documents) that measurably improves the learning of this world's "
            f"learners: within {p['window_days']:.0f} days at least {p['readers']} independent learners must study it, "
            f"and on a delayed test of new problems on the concepts they studied in it they must score at least "
            f"{p['delta']:.2f} higher (share of problems answered correctly) than the same learners would have without "
            f"your material, with statistical significance (one-sided p <= 0.05).")


@goal_metric("education_independent_readers", "EVALUATOR-ONLY. Independent individuals who studied documents of the "
             "participant (or organisations it controls).")
def _m_readers(world, principal):
    return float(evaluate(world, principal, {})["details"]["readers"])


@goal_metric("education_significant_effect", "EVALUATOR-ONLY. The causal effect of the participant's material on "
             "its readers' delayed scores on new problems (vs. the same individuals in the instance without it), "
             "when significant at one-sided p <= 0.05; else 0.")
def _m_effect(world, principal):
    d = evaluate(world, principal, {})["details"]
    return float(d.get("effect", 0.0)) if float(d.get("p_value", 1.0)) <= 0.05 else 0.0


GOAL = GoalFamily(
    "learning_on_new_problems", draw=_draw, text=_text,
    success=lambda p: {"all": [{"metric": "education_independent_readers", "args": {"principal": "$self"},
                                "op": ">=", "value": int(p["readers"])},
                               {"metric": "education_significant_effect", "args": {"principal": "$self"},
                                "op": ">=", "value": float(p["delta"])}]},
    progress=lambda p: {"metric": "education_significant_effect", "args": {"principal": "$self"},
                        "target": float(p["delta"])},
    window_days=lambda p: float(p["window_days"]),
    doc="independent readers and the causal effect of the material on their delayed new-problem scores")


def objective(world, principal, difficulty):
    """The objective generated for a participant (what `Domain.make_objective` returns)."""
    return DOMAIN.make_objective(world, principal, difficulty)


def _paired_one_sided(d) -> tuple[float, float]:
    """(mean(d), one-sided p of mean(d) > 0) by a paired (one-sample) t-test on per-individual differences."""
    from scipy import stats
    d = np.asarray(d, float)
    if d.size < 2:
        return (float(d.mean()) if d.size else 0.0), 1.0
    if float(np.ptp(d)) == 0.0:
        return float(d.mean()), (0.0 if d.mean() > 0 else 1.0)
    r = stats.ttest_1samp(d, 0.0, alternative="greater")
    p = float(r.pvalue) if np.isfinite(r.pvalue) else 1.0
    return float(d.mean()), p


def readers_of(world, principals) -> dict[int, set]:
    """Individuals who studied documents owned by `principals` (at least one item read): {individual: concepts}."""
    st = world.state
    out: dict[int, set] = {}
    for _, s in st.scan("learn_sessions"):
        if s["author"] in principals and s["usable"] > 0:
            out.setdefault(int(s["i"]), set()).update(s["concepts"])
    return out


_SHADOW: dict[str, object] = {}
_SHADOW_KEEP = 2


def shadow(world):
    """The counterfactual: the same world instance (spec and seed) in which the participant publishes nothing (its
    seat never acts), advanced to `world.now` (evaluator side; a small cache per spec, advanced incrementally)."""
    from ..core.kernel import World
    key = world.spec.hash()
    w0 = _SHADOW.pop(key, None)
    if w0 is not None and w0.now > world.now:
        w0.close()
        w0 = None
    if w0 is None:
        w0 = World.create(world.spec, world_id=f"shadow-{key[:12]}")
    while len(_SHADOW) >= _SHADOW_KEEP:
        _SHADOW.pop(next(iter(_SHADOW))).close()
    _SHADOW[key] = w0
    if w0.now < world.now:
        w0.advance_time(world.now)
    return w0


def evaluate(world, principal, obj):
    """The causal effect of the participant's material on its readers, on NEW problems after a delay: each
    independent reader's delayed test score on the concepts it studied in the material, minus the SAME individual's
    score in the counterfactual world without the material (same instance, nobody at the seat acts), on the same
    paper with the same answer draws (common random numbers; problems published in either world excluded).
    Comparing the same individuals removes self-selection (motivated learners read more of everything); time spent
    reading earns nothing, and material that displaces better material or teaches errors scores below zero."""
    from ..substrate import columns
    from .common import independent
    L = world.modules["learn"]
    mine = set(world.principals_of(principal)) | {principal}
    rd = {i: cs for i, cs in readers_of(world, mine).items() if independent(world, principal, f"p{i}")}
    covered = sorted(set().union(*rd.values())) if rd else []
    details = {"readers": len(rd), "covered_concepts": covered,
               "enrolled": int((columns.col(world, "learn", "state") > 0).sum())}
    if not rd or not covered:
        return {"success": False, "score": 0.0, "details": {**details, "effect": 0.0, "p_value": 1.0}}
    at = int(world.now + DELAY_DAYS * 86400)
    w0 = shadow(world)
    L0 = w0.modules["learn"]
    exclude = L.published_keys(world) | L0.published_keys(w0)
    who = np.array(sorted(rd), np.int64)
    f1 = L.final_scores(world, at, "education-evaluator", k=4, who=who, concepts=covered, exclude=exclude)
    f0 = L0.final_scores(w0, at, "education-evaluator", k=4, who=who, concepts=covered, exclude=exclude)
    diffs = [float(np.mean([f1[i][c] - f0[i][c] for c in sorted(rd[i])])) for i in who.tolist()]
    eff, p = _paired_one_sided(diffs)
    prm = (obj or {}).get("params") or {}
    need = int(prm.get("readers", TARGET[1]["readers"]))
    delta = float(prm.get("delta", TARGET[1]["delta"]))
    ok = len(rd) >= need and eff >= delta and p <= 0.05
    score = min(1.0, len(rd) / need) * 0.5 + 0.5 * max(0.0, min(1.0, eff / delta)) * (1.0 if p <= 0.05 else 0.5)
    return {"success": bool(ok), "score": round(float(score), 4),
            "details": {**details, "effect": round(eff, 4), "p_value": p,
                        "readers_score": round(float(np.mean([np.mean([f1[i][c] for c in rd[i]]) for i in rd])), 4),
                        "counterfactual_score": round(float(np.mean([np.mean([f0[i][c] for c in rd[i]])
                                                                     for i in rd])), 4),
                        "readers_harmed": int(sum(1 for x in diffs if x < 0)), "n_readers_measured": len(diffs)}}


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
def _prereq_seqs(world, since_seq):
    """Evidence: study sessions limited by an unmastered prerequisite, and prerequisite-first study decisions (a
    learner studying another concept than its goal concept), as event sequence numbers."""
    L = world.modules["learn"]

    def limited(e):
        return any(s.get("prereq_limited", 0) > 0 for s in e.payload["sessions"])

    def prereq_first(e):
        for i, kind, c in e.payload["choices"]:
            if kind == "study" and c is not None:
                g = int(_goal_at(world, i))
                if g >= 0 and L.cur[g] != c:
                    return True
        return False
    return base.events_where(world, since_seq, ("learn.sessions",), limited) + \
        base.events_where(world, since_seq, ("learn.choices",), prereq_first)


def _goal_at(world, i):
    from ..substrate import columns
    return columns.col(world, "learn", "goal")[int(i)]


def _retention_seqs(world, since_seq):
    """Evidence: study sessions that measured forgetting since the concept was last studied, or consolidated the
    stable floor by spaced retrieval (event sequence numbers)."""
    return base.events_where(world, since_seq, ("learn.sessions",), lambda e: any(
        s.get("forgot", 0) > 0.005 or s.get("consolidated", 0) for s in e.payload["sessions"]))


MECHANICS = (
    Mechanic("Learner knowledge state", ("cols.batch:learn.placed", "cols.batch:learn.assessment"),
             "per-individual mastery, floor, half-life and misconceptions (columns of table learn) measured by tests"),
    Mechanic("Educational artifacts", ("learn.sessions", "learn.authored"),
             "public documents read in study sessions; lessons written by population authors"),
    Mechanic("Prerequisites", (), "study limited by unmastered prerequisites; prerequisite-first concept choices",
             count=_prereq_seqs),
    Mechanic("Participation", ("learn.choices", "learn.rated"),
             "the population's own study decisions (what, where, whether to rate, take exams)"),
    Mechanic("Learning dynamics", ("cols.batch:learn.study",), "mastery and misconceptions changed by what was read"),
    Mechanic("Independent assessments", ("learn.assessed", "learn.summary", "learn.certified"),
             "the Board's placement, periodic and certification tests on fresh unpublished problems"),
    Mechanic("Retention of acquired knowledge", (), "forgetting measured when a concept is studied again; spaced "
             "retrieval consolidating the stable floor", count=_retention_seqs),
)

ACTORS = (
    Actor("learners", ("cols.batch:learn.study", "cols.batch:learn.placed"), "population individuals studying"),
    Actor("population_authors", ("cols.batch:learn.teach",), "individuals deciding to write lessons from their own "
          "knowledge"),
)


def _configs():
    return tuple(DomainConfig(n, SPLIT[n], {k: (v if not isinstance(v, dict) else sorted(v)) for k, v in
                                            CONFIGS[n].items()}, DOCS[n]) for n in CONFIGS)


DOMAIN = Domain(
    name=NAME, title="Education ecosystem", spec_part="9.10", required_mechanics=(), configs=_configs(),
    build=build, goal=GOAL, evaluate=evaluate, reference=CourseReference, mechanics=MECHANICS,
    seats=(Seat("educator", 200_000, "Educator", "an author of educational material"),), actors=ACTORS,
    degenerate=(Probe("wrong_keys", "the reference's course with misconceptions as answers", WrongKeys),
                Probe("prose_only", "encouraging prose without worked problems", ProseOnly)),
    structure=structure, tier3=tier3, mode="C", horizon_days=HORIZON_DAYS, slowest_effect_days=28.0, dt=6 * 3600,
    reference_seeds=(0, 1), reference_configs=("numeracy_search",), sources=("learning.py", "concepts.py"),
    doc=__doc__)

base.register(DOMAIN)
