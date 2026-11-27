"""Education on the ONE population (kernel module `learn`, host `learn`; MASTER_SPEC 9.10, D-15, D-18).

The learners are the individuals of `crucible.population` (no second population: the module keeps no learner
persons of its own). It registers two population activities (ONE_WORLD_CONTRACTS section 1):

* `study` - an individual who spends a session on it works on its own learning goal: a placement test the first
  time, then it chooses what to study (its goal, or the weakest prerequisite it perceives), finds a document
  (a lesson of the Board's textbook, of another individual, of the participant: any public document) through its
  own channels - search with a query it forms itself, the study board of the forum, what its own contacts studied
  last and liked (word of mouth), or the next document a lesson pointed to - reads it within its attention and
  learns from what it contains, rates it now and then, recommends a lesson on the forum after a test showed its
  progress, and takes the Board's certification exam once it believes it masters its goal;
* `teach` - an individual who knows a concept well and writes well enough authors a lesson for it from its OWN
  knowledge (its worked answers come from its own knowledge state, so they can be wrong), in its own style, as a
  public document. What others then learn from it is decided by its content, not by who wrote it.

Per-person state lives in this module's column table `learn` (one row per individual; no population fact copied):
mastery per concept and level, a stable floor of long-term knowledge (permastore), memory half-life, time of last
study, misconception strength, own perceived mastery, own test scores, goal, motivation, trust in authors it has
read, recent documents, certificates.

Mechanisms (fidelity records FR-LEARN-1..6; forms from the learning-science literature, parameters in published
ranges, not fitted in-world):

* **Reading** (`concepts.parse_lesson`): a learner extracts worked problems and exercises written in the
  curriculum's notations from ordinary text; prose alone teaches nothing; whether a stated answer is right is
  decided by the world's solver. Study costs attention minutes (`substrate.attention`).
* **Learning dynamics** (FR-LEARN-1): a worked example raises mastery of its concept and level with readiness
  (prerequisite and lower-level mastery), novelty and massing, more for novices (worked-example effect, expertise
  reversal: Sweller; Kalyuga 2003); an exercise is attempted before its answer is read (testing effect, Roediger &
  Karpicke 2006): a right attempt confirmed by a right key strengthens and lengthens the half-life, more when
  spaced (Cepeda et al. 2006); a wrong attempt corrected by a right key teaches less and weakens a misconception;
  a wrong answer key or a wrong worked example interferes and strengthens a misconception unless the learner knows
  enough to notice (Grosse & Renkl 2007: unflagged erroneous examples harm novices). Per-opportunity gains are in
  the range of Bayesian knowledge-tracing fits (P(T) 0.1-0.4, guess 0.02-0.2, slip 0.05-0.1).
* **Retention** (FR-LEARN-2): mastery above a stable floor (permastore, Bahrick 1984) decays with the concept's
  half-life (half-life regression, Settles & Meeder 2016); spaced successful retrieval raises the floor and the
  half-life, massed study does not.
* **Answering** (FR-LEARN-3): the probability of a right answer is guess + (1 - guess - slip) x effective mastery x
  a prerequisite factor; a misconception answers by its buggy procedure where that differs from the solution.
* **Participation** (FR-LEARN-4): placement, goal choice, concept choice, channels, the lesson-choice logit (query
  match, social proof, ratings, own trust in the author, word of mouth), dropout through falling motivation (the
  population's habit for `study` learns from the session's utility), certification decisions.
* **Independent assessment** (FR-LEARN-5): the Board's placement, periodic and certification tests are drawn fresh
  for each test and exclude every problem published in any public document; only the individual sees its scores;
  aggregates are public. The evaluator's delayed final test (`final_scores`) uses its own random stream.
* **Teaching** (FR-LEARN-6): who teaches, which concept (demand over supply), the plan and style of a lesson.

Certificates are public facts (`learn_certs`): the labour market reads them (domains.firms), so education changes
job matching and, through real mastery, work performance elsewhere.

Handlers read state and emit events; appliers (columns batches and the module's appliers) write. Randomness comes
from `ctx.rng` / `world.rng_for`. Nothing a participant can call returns another individual's knowledge state.
"""
from __future__ import annotations

import functools
import math
import re

import numpy as np

from ..core import schema as S
from ..core.canonical import digest
from ..core.errors import CrucibleError, Forbidden, NotFound, ProtocolError
from ..core.kernel import Module, applier, endpoint, process, register_module
from ..substrate import attention, columns, ids
from . import concepts as C

HOUR, DAY = 3600, 86400
TABLE = "learn"
T_SLOTS, R_SLOTS = 6, 8
LEVELS = (1, 2, 3)
DOC_RE = re.compile(r"doc_(\d{6,})")

DEFAULTS = {
    "curriculum": ["add", "sub", "mul", "div", "gcd", "lcm", "frac_simplify", "frac_add"],
    "half_life_days": [3.0, 8.0], "assess_days": 7, "assess_k": 3, "placement_k": 4, "exam_k": 4,
    "board": "study", "channels": ["search", "forum", "contacts"], "authors": True, "textbook": "basic",
    "prior": 0.2, "teach_prior": -0.4, "cert_pass": 0.8, "cohort": None, "goal_min_depth": 2,
    "recommend_p": 0.35, "rate_p": 0.4, "search_limit": 8, "n": 0,
}
# learning-rate constants (FR-LEARN-1, FR-LEARN-3)
A_EX, A_PR, THETA, GUESS, SLIP, B_WRONG, H_MAX = 0.30, 0.34, 0.6, 0.02, 0.05, 0.22, 180.0
MIN_EX, MIN_PR = 1.5, 2.5          # attention minutes per item and level
NOVEL_REPEAT, MASS, KAPPA = 0.25, 0.35, 0.3
MODEL = {"model_id": "learn.participation", "model_version": "2", "fidelity_id": "FR-LEARN-4"}
TEACH_MODEL = {"model_id": "learn.teaching", "model_version": "1", "fidelity_id": "FR-LEARN-6"}


def _cl(x, lo=0.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


def doc_num(did: str | None) -> int:
    m = DOC_RE.fullmatch(str(did or ""))
    return int(m.group(1)) if m else -1


def doc_id(n: int) -> str | None:
    return f"doc_{int(n):06d}" if int(n) >= 0 else None


# ============================================================================================ the learner model
class Rec:
    """One individual's knowledge state as plain arrays (a working copy; written back through a batch)."""
    __slots__ = ("m", "pm", "h", "tl", "bug", "cur", "cix")

    def __init__(self, m, pm, h, tl, bug, cur, cix):
        self.m, self.pm, self.h, self.tl, self.bug, self.cur, self.cix = m, pm, h, tl, bug, cur, cix

    @classmethod
    def of(cls, world, i: int, cur, cix):
        c = lambda n: np.array(columns.col(world, TABLE, n)[i], dtype=np.float64)  # noqa: E731
        return cls(c("m"), c("pm"), c("h"), np.array(columns.col(world, TABLE, "tl")[i], dtype=np.int64), c("bug"),
                   cur, cix)

    def eff(self, c: str, L: int, t: int) -> float:
        k = self.cix[c]
        dt = max(0.0, (t - int(self.tl[k])) / DAY)
        m, p = self.m[k, L - 1], min(self.pm[k, L - 1], self.m[k, L - 1])
        return float(p + (m - p) * 2.0 ** (-dt / max(1e-6, self.h[k])))

    def readiness(self, c: str, L: int, t: int) -> float:
        pre, low = self.readiness_parts(c, L, t)
        return pre * low

    def readiness_parts(self, c: str, L: int, t: int) -> tuple[float, float]:
        """(prerequisite readiness, lower-level readiness): how far the concept's prerequisites and its own lower
        level are mastered relative to the threshold THETA."""
        pre = [p for p in C.PREREQS[c] if p in self.cix]
        r = min((_cl(self.eff(p, L, t) / THETA) for p in pre), default=1.0)
        low = _cl(self.eff(c, L - 1, t) / THETA) if L > 1 else 1.0
        return r, low

    def p_correct(self, c: str, L: int, t: int) -> float:
        pre = [p for p in C.PREREQS[c] if p in self.cix]
        pf = min((0.4 + 0.6 * self.eff(p, L, t) for p in pre), default=1.0)
        return GUESS + (1.0 - GUESS - SLIP) * self.eff(c, L, t) * pf

    def answer(self, c: str, p, t: int, rng) -> tuple[str, bool]:
        """The learner's answer to problem `p` of concept `c`: its misconception where that bites, a right answer
        with probability p_correct, else a slip."""
        con = C.LIBRARY[c]
        L = int(con.level(p))
        me = self.eff(c, L, t)
        u1, u2 = rng.random(), rng.random()
        bug = con.misconception(p)
        if bug is not None and u1 < self.bug[self.cix[c]] * (1.0 - 0.5 * me):
            a = bug
        elif u2 < self.p_correct(c, L, t):
            a = con.solve(p)
        else:
            a = _slip(con, p, rng)
        return a, con.grade(a, p)


def _slip(con, p, rng) -> str:
    sol = con.solve(p)
    v = con.norm(sol)
    from fractions import Fraction
    if isinstance(v, bool):
        return "?"
    if isinstance(v, int):
        return str(v + int(rng.choice([-10, -2, -1, 1, 2, 10])))
    if isinstance(v, float):
        return str(round(v + float(rng.choice([-1.0, -0.5, 0.5, 1.0])), 2))
    if isinstance(v, tuple):
        return f"{v[0]} r {v[1] + 1}"
    if isinstance(v, str):
        return v + "1"
    if isinstance(v, Fraction):
        return f"{v.numerator + 1}/{v.denominator}"
    return "?"


def assess(rec: Rec, t: int, rng, k: int, concepts, exclude: set) -> tuple[dict, int, str]:
    """Scores on fresh problems (k per concept and level, none of them in `exclude`): the share answered right."""
    scores, keys = {}, []
    for c in concepts:
        ok = 0
        for L in LEVELS:
            for _ in range(k):
                p = C.generate(c, rng, L)
                for _ in range(10):
                    if C.LIBRARY[c].key(p) not in exclude:
                        break
                    p = C.generate(c, rng, L)
                keys.append(C.LIBRARY[c].key(p))
                ok += rec.answer(c, p, t, rng)[1]
        scores[c] = round(ok / (3 * k), 4)
    return scores, len(keys), digest(keys)


def exam(world, idx, concepts, t: int, rng, k: int, exclude: set, cix: dict) -> tuple[np.ndarray, int, str]:
    """A test taken by every individual of `idx` at time `t`: for each concept and level `k` fresh problems (one
    paper for all takers, none of them in `exclude`), answered by each individual from its own knowledge state
    exactly as `Rec.answer` does (misconception where it bites, else right with p_correct, else a slip), vectorised
    over individuals. Returns (share right [n, len(concepts)], problems per taker, digest of the paper)."""
    ix = np.asarray(idx, np.int64)
    n = ix.size
    out = np.zeros((n, len(concepts)), np.float64)
    keys = []
    if n == 0:
        return out, 0, digest(keys)
    m_all = columns.col(world, TABLE, "m")[ix].astype(np.float64)
    pm_all = np.minimum(columns.col(world, TABLE, "pm")[ix].astype(np.float64), m_all)
    h_all = columns.col(world, TABLE, "h")[ix].astype(np.float64)
    tl_all = columns.col(world, TABLE, "tl")[ix].astype(np.int64)
    bug_all = columns.col(world, TABLE, "bug")[ix].astype(np.float64)
    dt = np.maximum(0.0, (int(t) - tl_all) / DAY)
    decay = 2.0 ** (-dt / np.maximum(1e-6, h_all))                        # [n, K]
    eff = pm_all + (m_all - pm_all) * decay[:, :, None]                    # [n, K, 3]
    for j, c in enumerate(concepts):
        kc = cix[c]
        pre = [cix[p] for p in C.PREREQS[c] if p in cix]
        con = C.LIBRARY[c]
        ok = np.zeros(n)
        for L in LEVELS:
            me = eff[:, kc, L - 1]
            pf = np.min(0.4 + 0.6 * eff[:, pre, L - 1], axis=1) if pre else np.ones(n)
            pc = GUESS + (1.0 - GUESS - SLIP) * me * pf
            for _ in range(k):
                p = C.generate(c, rng, L)
                for _ in range(10):
                    if con.key(p) not in exclude:
                        break
                    p = C.generate(c, rng, L)
                keys.append(con.key(p))
                u1, u2 = rng.random(n), rng.random(n)
                bites = con.misconception(p) is not None
                misled = (u1 < bug_all[:, kc] * (1.0 - 0.5 * me)) if bites else np.zeros(n, bool)
                ok += (~misled) & (u2 < pc)
        out[:, j] = ok / (3 * k)
    return np.round(out, 4), len(keys), digest(keys)


def path(goal: str, cur) -> list[str]:
    """The goal and every prerequisite of it in the curriculum (what a course towards the goal covers)."""
    return list(_path(goal, tuple(cur)))


@functools.lru_cache(maxsize=4096)
def _path(goal: str, cur: tuple) -> tuple:
    return tuple(c for c in C.closure([goal]) if c in cur)


# ============================================================================================ the module
@register_module
class Learning(Module):
    name = "learn"
    host = "learn"
    version = "2"
    depends = ("identity", "bank", "docs", "search", "forum", "artifacts", "cols", "pop")

    def __init__(self, config=None):
        super().__init__(config)
        cfg = dict(DEFAULTS)
        for k, v in (self.config or {}).items():
            if k not in DEFAULTS:
                raise ProtocolError(f"learn: unknown config key {k!r}")
            cfg[k] = v
        if int(cfg.get("n") or 0):
            raise ProtocolError("learn: learners are the population's individuals (config 'n' must be 0)")
        for ch in cfg["channels"]:
            if ch not in ("search", "forum", "contacts"):
                raise ProtocolError(f"learn: unknown channel {ch!r}")
        self.cfg = cfg

    def install(self, world):
        from .. import population as P
        self.cur = C.closure(self.cfg["curriculum"])
        if len(self.cur) > 62:
            raise ProtocolError("learn: at most 62 concepts")
        self.cix = {c: k for k, c in enumerate(self.cur)}
        n = P.n_individuals(world)
        K = len(self.cur)
        columns.declare(world, TABLE, n, {
            "m": ("float32", (K, 3), 0.0), "pm": ("float32", (K, 3), 0.0), "h": ("float32", (K,), 3.0),
            "tl": ("int64", (K,), 0), "bug": ("float32", (K,), 0.0), "perc": ("float32", (K,), -1.0),
            "score": ("float32", (K,), -1.0), "score0": ("float32", (K,), -1.0), "placed": ("float32", (K,), -1.0),
            "goal": ("int16", (), -1), "mot": ("float32", (), 0.5), "state": ("uint8", (), 0),
            "sessions": ("int32", (), 0), "minutes": ("float32", (), 25.0), "next_doc": ("int32", (), -1),
            "last_doc": ("int32", (), -1), "last_ok": ("float32", (), 0.0), "ta": ("int32", (T_SLOTS,), -1),
            "tv": ("float32", (T_SLOTS,), 1.0), "rd": ("int32", (R_SLOTS,), -1), "rn": ("uint8", (R_SLOTS,), 0),
            "certs": ("int64", (), 0), "exam_t": ("int64", (), -1), "eligible_at": ("int64", (), 0),
            "taught_t": ("int64", (), -1), "posted": ("int64", (), 0), "lessons_read": ("int32", (), 0)})
        st = world.state
        st.declare_table("learn_lessons", index=("author",))
        st.declare_table("learn_sessions", index=("doc", "author", "i"))
        st.declare_table("learn_assessments", index=("kind",))
        st.declare_table("learn_certs", index=("person", "concept"))
        st.declare_table("learn_authors")
        P.register_activity(world, "study", self.act_study, owner=self.name, prior=float(self.cfg["prior"]),
                            goals=(1,))
        if self.cfg["authors"]:
            P.register_activity(world, "teach", self.act_teach, owner=self.name,
                                prior=float(self.cfg["teach_prior"]), goals=(5,))

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        from .. import population as P
        w = ctx.world
        cfg = self.cfg
        forum = w.modules["forum"]
        if not ctx.state.has("forum_boards", cfg["board"]):
            forum.create_board(ctx, cfg["board"], "Study", "Learning: lessons, courses and study help")
        from .common import make_org
        board, _lead = make_org(ctx, "board-of-education", "Board of Education", "institution")
        docs = w.modules["docs"]
        docs.create_doc(ctx, board, "Curriculum: concepts, prerequisites and problem notations", self.curriculum_text(),
                        "public")
        if cfg["textbook"]:
            rng = w.rng_for("learn-textbook")
            chapters = []
            for c in self.cur:
                chapters.append((c, docs.create_doc(ctx, board, f"{C.LIBRARY[c].title} ({c}) - textbook chapter",
                                                    textbook(c, cfg["textbook"], rng), "public")))
            if "forum" in cfg["channels"]:
                # the Board announces its textbook where this world's learners look for study material (without
                # it, forum-only worlds had nothing to find until somebody posted: nobody ever studied)
                for c, did in chapters:
                    con = C.LIBRARY[c]
                    forum.post(ctx, board, cfg["board"], f"{con.title} ({c}) - textbook chapter {did}",
                               f"The Board of Education's textbook chapter on {con.title.lower()} ({c}): {did}",
                               tags=["textbook"])
        ctx.emit("learn.meta", {"board": board}, vis="public")
        n = P.n_individuals(w)
        if n:
            self._initial_state(ctx, n)
        ctx.schedule(ctx.now + int(cfg["assess_days"]) * DAY, "learn.assess", {}, prio=55,
                     period=int(cfg["assess_days"]) * DAY, key="learn.assess")

    def _initial_state(self, ctx, n: int):
        """Heterogeneous prior knowledge from each individual's background (its reading, technical and planning
        skills in the population) plus individual variation; long-ago learning sits mostly on the stable floor;
        some hold prior misconceptions."""
        from .. import population as P
        from ..population.schema import SKILL_NAMES
        w = ctx.world
        cfg = self.cfg
        rng = w.rng_for("learn-genesis")
        f = P.features(w, np.arange(n))
        sk = f["skills"]
        cols = [SKILL_NAMES.index(s) for s in ("reading", "technical", "planning") if SKILL_NAMES.index(s) < sk.shape[1]]
        bg = np.clip(0.45 * rng.beta(2.0, 2.0, n) + 0.55 * sk[:, cols].mean(axis=1), 0.0, 1.0) if cols else \
            rng.beta(2.0, 2.0, n)
        K = len(self.cur)
        depth = np.array([C.depth(c, self.cur) for c in self.cur], np.float64)
        # adults mostly master the shallow concepts of a curriculum and few master its deep ones
        base = np.clip(1.2 * bg[:, None] + 0.25 - 0.22 * depth[None, :] + rng.normal(0, 0.08, (n, K)), 0.0, 0.97)
        m = np.stack([base, 0.75 * base, 0.5 * base], axis=2)
        pm = m * 0.85
        lo, hi = cfg["half_life_days"]
        h = rng.uniform(float(lo), float(hi), (n, K)) * (1.0 + 3.0 * base)
        # prior misconceptions: prevalence per concept (initial-state heterogeneity, not a failure die)
        bug = np.where(rng.random((n, K)) < 0.25, 0.35 * (1.0 - base), 0.0)  # fidelity: FR-LEARN-7
        prefs = f["prefs"]
        learner = (f["goals"] == 1).any(axis=1)
        mot = np.clip(0.35 + 0.25 * prefs[:, 1] + 0.2 * prefs[:, 3] + 0.15 * learner + rng.normal(0, 0.08, n),
                      0.1, 0.95)
        minutes = np.clip(f["attention_budget_min"] * rng.uniform(0.2, 0.45, n), 10.0, 60.0)
        eligible = np.zeros(n, np.int64)
        co = cfg.get("cohort")
        if co:
            later = rng.random(n) < float(co.get("share", 0.5))
            eligible = np.where(later, int(float(co.get("at_day", 14)) * DAY), 0).astype(np.int64)
        columns.batch(ctx, TABLE, "learn.init", np.arange(n), {
            "m": m.astype(np.float32), "pm": pm.astype(np.float32), "h": h.astype(np.float32),
            "tl": np.zeros((n, K), np.int64), "bug": bug.astype(np.float32), "mot": mot.astype(np.float32),
            "minutes": minutes.astype(np.float32), "eligible_at": eligible},
            extra={"concepts": list(self.cur)})

    def curriculum_text(self) -> str:
        lines = ["Concepts taught and assessed in this world, with prerequisites and the notations problems are "
                 "written in.", ""]
        for c in self.cur:
            con = C.LIBRARY[c]
            pre = ", ".join(p for p in C.PREREQS[c] if p in self.cur) or "none"
            lines.append(f"- {c}: {con.title}. {con.doc} Problems are written `{con.fmt}`; answers: {con.answer_fmt}. "
                         f"Prerequisites: {pre}.")
        lines += ["", "Learners study worked problems and exercises written in these notations. The Board assesses "
                  "learners on new problems of every concept and level and certifies those who pass its exams."]
        return "\n".join(lines)

    # ------------------------------------------------------------------ helpers
    def _call(self, ctx, p, tool, args):
        try:
            return True, ctx.call(tool, args, as_principal=p)
        except CrucibleError as e:
            return False, e.as_dict()

    def _parsed(self, world, did: str, body: str, version: int) -> dict:
        cache = world.runtime.setdefault("learn.parse", {})
        key = (did, int(version))
        if key not in cache:
            cache[key] = C.parse_lesson(body, self.cur)
        return cache[key]

    def published_keys(self, world) -> set:
        """Every problem that appears as a worked problem in any public document (tests never reuse them)."""
        st = world.state
        out = set()
        if "documents" not in st.tables:
            return out
        for did in st.find_ids("documents", "visibility", "public"):
            d = st.tables["documents"][did]
            for it in self._parsed(world, did, d["body"], d["version"])["items"]:
                out.add(it["key"])
        return out

    def author_code(self, st, principal: str) -> int | None:
        r = st.get("learn_authors", principal)
        return None if r is None else int(r["code"])

    # ------------------------------------------------------------------ activity: study
    def act_study(self, ctx, idx, now):
        """A study session of each individual of the cohort (see the module docstring). Returns its utility."""
        w = ctx.world
        idx = np.asarray(idx, np.int64)
        util = np.zeros(idx.size)
        if idx.size == 0:
            return util
        exclude = None
        rows: dict[str, list] = {}
        sessions, rated, recs, certs, choices = [], [], [], [], []
        spend = np.zeros(idx.size)
        for j, i in enumerate(idx.tolist()):
            rng = ctx.rng("learn.study", i)
            row = self._row(w, i)
            if now < row["eligible_at"]:
                util[j] = -0.3                      # nothing open to it yet (a later cohort)
                continue
            rec = Rec.of(w, i, self.cur, self.cix)
            if row["state"] == 0:
                if exclude is None:
                    exclude = self.published_keys(w)
                ch = self._placement(ctx, i, rec, row, rng, exclude, now)
                recs.append({"i": i, "kind": "placement", "scores": ch.pop("score_d"), "n_items": ch.pop("n_items"),
                             "digest": ch.pop("digest"), "t": now})
                rows.setdefault("placed", []).append((i, ch))
                util[j] = 0.1
                spend[j] = 15.0
                choices.append((i, "placement", None))
                continue
            ch: dict = {}
            goal = self.cur[row["goal"]] if row["goal"] >= 0 else None
            self._maybe_recommend(ctx, i, row, rng, ch)       # word of mouth after a test showed progress
            if goal is not None and row["perc"][self.cix[goal]] >= 0.85:
                goal = self._next_goal(row)
                ch["goal"] = self.cix[goal] if goal else -1
                if goal is None:
                    ch["state"] = 3
                    choices.append((i, "done", None))
                    rows.setdefault("decide", []).append((i, ch))
                    util[j] = 0.05
                    continue
                choices.append((i, "goal", goal))
            if goal is None:
                goal = self._next_goal(row)
                if goal is None:
                    util[j] = -0.2
                    continue
                ch["goal"] = self.cix[goal]
            row = {**row, **{k: v for k, v in ch.items() if k in row}}
            c = self.choose_concept(row, goal)
            gk = self.cix[goal]
            if row["perc"][gk] >= 0.8 and not (int(row["certs"]) >> gk) & 1 and \
                    (row["exam_t"] < 0 or now - row["exam_t"] >= 7 * DAY):
                if exclude is None:
                    exclude = self.published_keys(w)
                passed, score = self._exam(ctx, i, rec, goal, rng, exclude, now)
                ch["exam_t"] = now
                perc = np.array(row["perc"], np.float32)
                perc[gk] = 0.5 * perc[gk] + 0.5 * score
                ch["perc"] = perc
                if passed:
                    ch["certs"] = int(row["certs"]) | (1 << gk)
                    certs.append({"person": ids.pid(i), "concept": goal, "score": score, "t": now})
                rows.setdefault("decide", []).append((i, ch))
                util[j] = 0.4 if passed else -0.1
                spend[j] = 20.0
                choices.append((i, "exam", goal))
                continue
            doc, via, post = self._find(ctx, i, row, c, rng, now)
            if doc is None:
                ch["mot"] = float(_cl(row["mot"] - 0.03))
                rows.setdefault("decide", []).append((i, ch))
                choices.append((i, "nothing_found", c))
                util[j] = -0.25
                continue
            p = ids.ensure_principal(ctx, i)
            ok, d = self._call(ctx, p, "docs.get", {"doc": doc})
            if not ok:
                util[j] = -0.25
                continue
            out = self._study(ctx, i, rec, row, doc, d, rng, now, via)
            ch.update(out["changes"])
            rows.setdefault("study", []).append((i, ch))
            sessions.append(out["session"])
            if out["rating"] is not None:
                rated.append(out["rating"])
                if post and out["rating"]["stars"] >= 4:
                    # a lesson that worked: the reader up-votes the board post that led to it (best-voted posts are
                    # what other learners browsing the board by "top" see)
                    self._call(ctx, p, "forum.vote", {"post": post, "value": 1})
            spend[j] = out["session"]["minutes"]
            util[j] = out["utility"]
            choices.append((i, "study", c))
        self._commit(ctx, rows, sessions, rated, recs, certs, choices)
        has = spend > 0
        if has.any():
            attention.spend(ctx, idx[has], spend[has], activity="study")
        return util

    def _row(self, world, i: int) -> dict:
        c = lambda n: columns.col(world, TABLE, n)[i]  # noqa: E731
        return {"state": int(c("state")), "goal": int(c("goal")), "mot": float(c("mot")), "perc": np.array(c("perc")),
                "score": np.array(c("score")), "score0": np.array(c("score0")), "placed": np.array(c("placed")),
                "next_doc": int(c("next_doc")), "last_doc": int(c("last_doc")), "ta": np.array(c("ta")),
                "tv": np.array(c("tv")), "rd": np.array(c("rd")), "rn": np.array(c("rn")), "certs": int(c("certs")),
                "exam_t": int(c("exam_t")), "eligible_at": int(c("eligible_at")), "sessions": int(c("sessions")),
                "minutes": float(c("minutes")), "posted": int(c("posted")), "lessons_read": int(c("lessons_read"))}

    def _placement(self, ctx, i, rec, row, rng, exclude, now) -> dict:
        k = int(self.cfg["placement_k"])
        arr, n, dg = exam(ctx.world, [i], self.cur, now, rng, k, exclude, self.cix)
        scores = {c: float(arr[0, j]) for j, c in enumerate(self.cur)}
        sc = arr[0].astype(np.float32)
        deep = [c for c in self.cur if C.depth(c, self.cur) >= int(self.cfg["goal_min_depth"]) and scores[c] < 0.6] or \
            sorted(self.cur, key=lambda c: (-C.depth(c, self.cur), c))[:1]
        wts = np.array([1.0 + C.depth(c, self.cur) for c in deep])
        goal = deep[int(rng.choice(len(deep), p=wts / wts.sum()))]
        return {"state": 1, "goal": self.cix[goal], "placed": sc, "perc": sc.copy(), "score": sc.copy(),
                "score0": sc.copy(), "score_d": scores, "n_items": n, "digest": dg}

    def _exam(self, ctx, i, rec, goal, rng, exclude, now) -> tuple[bool, float]:
        arr, _, _ = exam(ctx.world, [i], [goal], now, rng, int(self.cfg["exam_k"]), exclude, self.cix)
        s = float(arr[0, 0])
        return s >= float(self.cfg["cert_pass"]), s

    def _next_goal(self, row) -> str | None:
        perc = row["perc"]
        cand = [c for c in self.cur if perc[self.cix[c]] < 0.8 and
                all(perc[self.cix[p]] >= 0.55 for p in C.PREREQS[c] if p in self.cix)]
        if not cand:
            return None
        return sorted(cand, key=lambda c: (-C.depth(c, self.cur), c))[0]

    def choose_concept(self, row, goal: str) -> str:
        c = goal
        perc = row["perc"]
        for _ in range(8):
            weak = [p for p in C.PREREQS[c] if p in self.cix and perc[self.cix[p]] < 0.55]
            if not weak:
                return c
            c = min(weak, key=lambda p: (perc[self.cix[p]], p))
        return c

    def _trust(self, row, st, author: str) -> float:
        code = self.author_code(st, author)
        if code is None:
            return 1.0
        hit = np.flatnonzero(row["ta"] == code)
        return float(row["tv"][hit[0]]) if hit.size else 1.0

    def _times_read(self, row, dn: int) -> int:
        hit = np.flatnonzero(row["rd"] == dn)
        return int(row["rn"][hit[0]]) if hit.size else 0

    # ------------------------------------------------------------------ discovery and choice
    def _find(self, ctx, i, row, c, rng, now) -> tuple[str | None, str, str | None]:
        from .. import population as P
        from .common import bow_cos
        st = ctx.state
        w = ctx.world
        nd = doc_id(row["next_doc"])
        if nd and st.has("documents", nd) and self._times_read(row, row["next_doc"]) < 2 and \
                self._trust(row, st, st.tables["documents"][nd]["owner"]) >= 1.0:
            return nd, "next", None
        con = C.LIBRARY[c]
        cands: dict[str, dict] = {}
        p = ids.ensure_principal(ctx, i)
        chans = self.cfg["channels"]
        f = P.features(w, np.array([i]))
        if "contacts" in chans:
            for j in f["contacts"][0].tolist():
                if j < 0:
                    continue
                ld = int(columns.col(w, TABLE, "last_doc")[j])
                if ld < 0 or float(columns.col(w, TABLE, "last_ok")[j]) < 0.6:
                    continue
                did = doc_id(ld)
                les = st.get("learn_lessons", did)
                if les is not None and c in les["concepts"]:
                    cands.setdefault(did, {"via": "contacts", "cos": 0.6})
        if "forum" in chans:
            # one page of the study board: the newest posts (scrolling up to two pages further back now and then)
            # or the best voted ones
            if rng.random() < 0.3:
                args = {"board": self.cfg["board"], "sort": "top", "limit": 20}
            else:
                args = {"board": self.cfg["board"], "sort": "new", "limit": 20,
                        "offset": 20 * int(rng.choice([0, 0, 1, 2]))}
            ok, posts = self._call(ctx, p, "forum.list", args)
            if ok:
                for post in posts:
                    text = (post.get("title", "") + " " + post.get("snippet", "")).lower()
                    if c not in text and con.title.lower() not in text:
                        continue
                    for mm in DOC_RE.finditer(text):
                        cands.setdefault(f"doc_{mm.group(1)}", {"via": "forum", "cos": 0.5 + 0.05 * min(
                            4, max(0, int(post.get("score") or 0))), "post": post.get("post")})
        if "search" in chans:
            from ..population.schema import SKILL_NAMES
            sk = float(f["skills"][0][SKILL_NAMES.index("searching")])
            words = re.sub(r"[()]", " ", con.title.lower()).split()
            pick = [x for x in words if len(x) > 3 and x not in ("with", "whole", "numbers")]
            parts = []
            if pick:
                parts.append(" ".join(rng.permutation(pick)[: 1 + int(rng.integers(0, len(pick)))].tolist()))
            if sk > 0.5 or not parts:
                parts.append(c.replace("_", " "))
            parts.append(str(rng.choice(["lesson", "exercises", "how to", "examples", "practice", "explained"])))
            q = " ".join(parts)
            ok, res = self._call(ctx, p, "search.query", {"q": q, "kind": "doc", "limit": int(self.cfg["search_limit"])})
            if ok:
                for x in res["results"]:
                    if x.get("sponsored") or not x.get("ref"):
                        continue
                    cands.setdefault(x["ref"], {"via": "search",
                                                "cos": bow_cos(q, x["title"] + " " + x.get("snippet", ""))})
        if not cands:
            return None, "", None
        best, best_u = None, -1e9
        for did in sorted(cands):
            d = st.get("documents", did)
            if d is None or d["visibility"] != "public":
                continue
            dn = doc_num(did)
            n_read = self._times_read(row, dn)
            if n_read >= 3:
                continue
            tr = self._trust(row, st, d["owner"])
            if tr < 0.5:
                continue
            les = st.get("learn_lessons", did)
            readers = les["readers"] if les else 0
            rat = (les["rating_sum"] + 3.0 * 2) / (les["rating_n"] + 2) if les else 3.0
            title = d["title"].lower()
            on_topic = 1.0 if (c in title or con.title.lower() in title) else 0.0
            u = (2.0 * cands[did]["cos"] + 1.2 * on_topic + 0.35 * math.log1p(readers) + 0.5 * (rat - 3.0)
                 + 1.5 * math.log(max(1e-3, tr)) + (0.8 if cands[did]["via"] == "contacts" else 0.0)
                 - 0.8 * n_read + float(rng.gumbel()) * 0.7)
            if u > best_u:
                best, best_u = did, u
        return (best, cands[best]["via"], cands[best].get("post")) if best else (None, "", None)

    # ------------------------------------------------------------------ reading and learning
    def _study(self, ctx, i, rec: Rec, row, doc, d, rng, now, via) -> dict:
        from .. import population as P
        st = ctx.state
        les = self._parsed(ctx.world, doc, d["body"], d["version"])
        dn = doc_num(doc)
        n_read = self._times_read(row, dn)
        left = float(P.features(ctx.world, np.array([i]))["attention_left_min"][0])
        minutes = min(float(row["minutes"]), left)
        used = 0.0
        touched: dict[str, float] = {}
        massed: dict[str, int] = {}
        attempts: list[tuple[str, bool]] = []
        noticed = 0
        items = []
        prereq_limited, forgot, consolidated = 0, 0.0, 0
        for it in les["items"]:
            c, L = it["concept"], it["level"]
            cost = (MIN_EX if it["kind"] == "example" else MIN_PR) * L
            if used + cost > minutes:
                break
            used += cost
            k = self.cix[c]
            if c not in touched:            # consolidate the forgetting curve before studying the concept again
                dt = max(0.0, (now - int(rec.tl[k])) / DAY)
                for lv in LEVELS:
                    e = rec.eff(c, lv, now)
                    forgot += max(0.0, float(rec.m[k, lv - 1]) - e)
                    rec.m[k, lv - 1] = e
                touched[c] = min(1.0, dt / max(1e-6, rec.h[k]))
                rec.tl[k] = now
            spacing = touched[c]
            R_pre, R_low = rec.readiness_parts(c, L, now)
            R = R_pre * R_low
            prereq_limited += int(R_pre < 0.9)
            nov = NOVEL_REPEAT if n_read else 1.0
            kk = massed.get(f"{c}|{L}", 0)
            massed[f"{c}|{L}"] = kk + 1
            mass = 1.0 / (1.0 + MASS * kk)
            mm = rec.m[k, L - 1]
            if it["kind"] == "example":
                if it["correct"]:
                    rec.m[k, L - 1] = mm + A_EX * R * nov * mass * (1 - mm) * (1 - 0.7 * mm)
                    rec.h[k] = min(H_MAX, rec.h[k] * (1 + 0.15 * spacing * R))
                elif rng.random() < mm:     # knows enough to notice the error: no harm, less trust
                    noticed += 1
                else:
                    rec.bug[k] = rec.bug[k] + B_WRONG * (1 - mm) * (1 - rec.bug[k])
                    rec.m[k, L - 1] = mm * 0.96
            else:
                p = tuple(it["params"])
                ans, truly = rec.answer(c, p, now, rng)
                con = C.LIBRARY[c]
                matches = con.norm(ans) is not None and con.norm(ans) == con.norm(it["stated"])
                attempts.append((c, matches))
                if it["correct"]:
                    if truly:
                        rec.m[k, L - 1] = mm + A_PR * R * nov * mass * (1 - mm) * (0.3 + 0.7 * mm)
                        rec.h[k] = min(H_MAX, rec.h[k] * (1 + 0.6 * spacing * R))
                        gain_pm = KAPPA * spacing * max(0.0, rec.m[k, L - 1] - rec.pm[k, L - 1])
                        rec.pm[k, L - 1] = rec.pm[k, L - 1] + gain_pm
                        consolidated += int(gain_pm > 1e-4)
                        rec.bug[k] *= 0.9
                    else:
                        rec.m[k, L - 1] = mm + 0.6 * A_PR * R * nov * mass * (1 - mm) * (0.3 + 0.7 * mm)
                        rec.h[k] = min(H_MAX, rec.h[k] * (1 + 0.1 * spacing * R))
                        rec.bug[k] *= 0.75
                elif truly:                 # a right attempt contradicted by a wrong key: confusion
                    rec.bug[k] = rec.bug[k] + 0.5 * B_WRONG * (1 - rec.bug[k])
                    rec.m[k, L - 1] = mm * 0.97
                    noticed += 1
                else:                       # a wrong attempt confirmed by a wrong key
                    rec.bug[k] = rec.bug[k] + B_WRONG * (1 - rec.bug[k])
                    rec.m[k, L - 1] = mm * 0.98
            rec.m[k, L - 1] = _cl(rec.m[k, L - 1])
            rec.pm[k, L - 1] = _cl(min(rec.pm[k, L - 1], rec.m[k, L - 1]))
            rec.bug[k] = _cl(rec.bug[k])
            items.append([c, L, it["kind"][0], int(it["correct"])])
        usable = len(items)
        # the learner's own reading of the session: did its attempts match the answer key?
        perc = np.array(row["perc"], np.float32)
        by_c: dict[str, list] = {}
        for c, ok in attempts:
            by_c.setdefault(c, []).append(ok)
        for c, oks in by_c.items():
            k = self.cix[c]
            old = perc[k] if perc[k] >= 0 else 0.5
            perc[k] = 0.7 * old + 0.3 * (sum(oks) / len(oks))
        success = (sum(ok for _, ok in attempts) / len(attempts)) if attempts else (0.55 if usable else 0.0)
        if noticed:
            success = max(0.0, success - 0.15 * noticed)
        # trust in the author (own experience)
        author = d["owner"]
        ta, tv = np.array(row["ta"]), np.array(row["tv"], np.float32)
        code = self.author_code(st, author)
        new_author = code is None
        if new_author:
            code = self._new_author_code(ctx, author)
        hit = np.flatnonzero(ta == code)
        slot = int(hit[0]) if hit.size else int(np.argmin(np.where(ta < 0, -1.0, tv)))
        old = float(tv[slot]) if hit.size else 1.0
        ta[slot] = code
        tv[slot] = _cl(old + 0.6 * (success - 0.5) - (0.4 if not usable else 0.0) - 0.15 * noticed, 0.1, 2.5)
        rd, rn = np.array(row["rd"]), np.array(row["rn"])
        hit = np.flatnonzero(rd == dn)
        if hit.size:
            rn[hit[0]] = min(255, int(rn[hit[0]]) + 1)
        else:
            rd = np.roll(rd, 1)
            rn = np.roll(rn, 1)
            rd[0], rn[0] = dn, 1
        mot = _cl(row["mot"] + (0.06 * (success - 0.4) if usable else -0.06) - 0.01)
        nxt = doc_num(les["next"]) if (les["next"] and success >= 0.5 and les["next"] != doc) else -1
        changes = {"m": rec.m.astype(np.float32), "pm": rec.pm.astype(np.float32), "h": rec.h.astype(np.float32),
                   "tl": rec.tl.astype(np.int64), "bug": rec.bug.astype(np.float32), "perc": perc,
                   "mot": float(mot), "ta": ta, "tv": tv, "rd": rd, "rn": rn, "next_doc": nxt, "last_doc": dn,
                   "last_ok": float(success if usable else 0.0), "sessions": row["sessions"] + 1,
                   "lessons_read": row["lessons_read"] + (0 if n_read else 1)}
        rating = None
        if usable and ctx.rng("learn.rate", i, doc).random() < float(self.cfg["rate_p"]):
            rating = {"doc": doc, "stars": int(1 + round(4 * _cl(success)))}
        sess = {"i": int(i), "doc": doc, "version": int(d["version"]), "author": author, "via": via,
                "concepts": sorted(touched), "items": items, "usable": usable, "minutes": round(used, 2),
                "ok": round(float(success), 4), "t": int(now), "new_author": new_author,
                "prereq_limited": int(prereq_limited), "forgot": round(float(forgot), 4),
                "consolidated": int(consolidated)}
        util = (0.8 * success - 0.1) if usable else -0.4
        return {"changes": changes, "session": sess, "rating": rating, "utility": util}

    def _new_author_code(self, ctx, author: str) -> int:
        pend = ctx.world.runtime.setdefault("learn.pending_authors", {})
        if author in pend:
            return pend[author]
        code = len(ctx.state.tables["learn_authors"]) + len(pend)
        pend[author] = code
        return code

    def _maybe_recommend(self, ctx, i, row, rng, ch):
        """Word of mouth: after a test showed a gain of at least 0.25 on a concept, recommend on the study board
        the lesson the individual read most recently for it."""
        sc, s0 = row["score"], row["score0"]
        gain = np.where((sc >= 0) & (s0 >= 0), sc - s0, 0.0)
        k = int(np.argmax(gain))
        if gain[k] < 0.25 or (int(row["posted"]) >> k) & 1 or row["last_doc"] < 0 or \
                rng.random() > float(self.cfg["recommend_p"]):
            return
        did = doc_id(row["last_doc"])
        les = ctx.state.get("learn_lessons", did)
        c = self.cur[k]
        if les is None or c not in les["concepts"]:
            return
        p = ids.ensure_principal(ctx, i)
        con = C.LIBRARY[c]
        ok, _ = self._call(ctx, p, "forum.post", {
            "board": self.cfg["board"], "title": f"{con.title} ({c}): {did} helped me",
            "body": f"I studied {did} for {c} and my {c} test score went from {s0[k]:.2f} to {sc[k]:.2f}.",
            "tags": ["lesson"]})
        if ok:
            ch["posted"] = int(row["posted"]) | (1 << k)
            ch["score0"] = np.array(sc, np.float32)

    # ------------------------------------------------------------------ recording
    def _commit(self, ctx, rows, sessions, rated, recs, certs, choices):
        from .. import population as P
        pend = ctx.world.runtime.pop("learn.pending_authors", {})
        for a, code in sorted(pend.items(), key=lambda kv: kv[1]):
            ctx.emit("learn.author", {"principal": a, "code": int(code)}, vis="internal")
        w = ctx.world
        for kind, lst in sorted(rows.items()):
            lst = [(i, ch) for i, ch in lst if ch]
            if not lst:
                continue
            # one batch per kind: every column any row changed; rows that did not change a column keep its value
            keys = sorted({k for _, ch in lst for k in ch})
            ix = np.array([i for i, _ in lst], np.int64)
            vals = {}
            for k in keys:
                cur = columns.col(w, TABLE, k)
                vals[k] = np.stack([np.asarray(ch[k], dtype=cur.dtype) if k in ch else np.array(cur[i])
                                    for i, ch in lst])
            P.record_decisions(ctx, f"learn.{kind}", ix, vals, table=TABLE, **MODEL)
        if sessions:
            ctx.emit("learn.sessions", {"sessions": sessions}, vis="internal")
        for r in rated:
            ctx.emit("learn.rated", r, vis="public")
        for r in recs:
            ctx.emit("learn.assessed", {"id": ctx.state.new_id("lassess"), **r}, vis="internal")
        for c in certs:
            ctx.emit("learn.certified", c, vis="public")
        if choices:
            ctx.emit("learn.choices", {"choices": [[int(i), k, c] for i, k, c in choices]}, vis="internal")

    # ------------------------------------------------------------------ activity: teach
    def act_teach(self, ctx, idx, now):
        """Individuals who know a concept well and write well enough author a lesson for it from their own
        knowledge (FR-LEARN-6); others get nothing from the session."""
        from .. import population as P
        from ..population.schema import PREF_NAMES, SKILL_NAMES
        w = ctx.world
        idx = np.asarray(idx, np.int64)
        util = np.full(idx.size, -0.4)
        if idx.size == 0:
            return util
        f = P.features(w, idx)
        writing = f["skills"][:, SKILL_NAMES.index("writing")]
        careful = f["prefs"][:, PREF_NAMES.index("quality_sensitivity")]
        taught_t = columns.col(w, TABLE, "taught_t")
        goal = columns.col(w, TABLE, "goal")
        state = columns.col(w, TABLE, "state")
        K = len(self.cur)
        on_path = np.zeros((K, K))
        for gk, gname in enumerate(self.cur):
            for c in path(gname, self.cur):
                on_path[gk, self.cix[c]] = 1.0
        act = goal[(state == 1) & (goal >= 0)].astype(np.int64)
        demand = np.bincount(act, minlength=K).astype(np.float64) @ on_path
        supply = np.zeros(len(self.cur))
        for _, les in ctx.state.scan("learn_lessons"):
            for c in les["concepts"]:
                if c in self.cix:
                    supply[self.cix[c]] += 1
        done, authored = [], []
        for j, i in enumerate(idx.tolist()):
            if writing[j] < 0.4 or (taught_t[i] >= 0 and now - int(taught_t[i]) < 3 * DAY):
                continue
            rec = Rec.of(w, i, self.cur, self.cix)
            know = [c for c in self.cur if rec.eff(c, 1, now) >= 0.75 and rec.eff(c, 2, now) >= 0.55]
            if not know:
                continue
            rng = ctx.rng("learn.teach", i)
            sc = [(demand[self.cix[c]] + 1.0) / (supply[self.cix[c]] + 1.0) * float(rng.uniform(0.7, 1.3))
                  for c in know]
            c = know[int(np.argmax(sc))]
            body = authored_lesson(rec, c, float(careful[j]), rng, now)
            p = ids.ensure_principal(ctx, i)
            con = C.LIBRARY[c]
            title = str(rng.choice([f"{con.title}: worked examples", f"Learn {con.title.lower()} ({c})",
                                    f"{con.title} - practice problems", f"How to do {c.replace('_', ' ')}"]))
            ok, res = self._call(ctx, p, "docs.create", {"title": title, "body": body, "visibility": "public"})
            if ok and "forum" in self.cfg["channels"]:
                # authors share their lesson where learners look for material (the study board)
                self._call(ctx, p, "forum.post", {"board": self.cfg["board"], "title": f"{title} {res['doc']}",
                                                  "body": f"I wrote a lesson on {con.title.lower()} ({c}): "
                                                          f"{res['doc']}", "tags": ["lesson"]})
            if ok:
                done.append(i)
                authored.append({"i": int(i), "doc": res["doc"], "concept": c})
                util[j] = 0.4
        if done:
            P.record_decisions(ctx, "learn.teach", np.array(done, np.int64),
                               {"taught_t": np.full(len(done), now, np.int64)}, table=TABLE, **TEACH_MODEL)
            ctx.emit("learn.authored", {"lessons": authored}, vis="public")
        return util

    # ------------------------------------------------------------------ the Board's periodic assessment
    @process("assess")
    def pr_assess(self, ctx):
        """The Board's periodic test: one fresh paper over the whole curriculum for every enrolled individual (no
        problem published in any public document); scores go to each individual's own row, aggregates are public."""
        w = ctx.world
        st = ctx.state
        state = columns.col(w, TABLE, "state")
        who = np.flatnonzero(state > 0)
        if who.size == 0:
            ctx.emit("learn.summary", {"t": ctx.now, "concepts": {}, "assessed": 0}, vis="public")
            return
        exclude = self.published_keys(w)
        rng = ctx.rng("learn.assess", int(ctx.now))
        arr, n, dg = exam(w, who, self.cur, ctx.now, rng, int(self.cfg["assess_k"]), exclude, self.cix)
        goal = columns.col(w, TABLE, "goal")[who].astype(np.int64)
        sc_old = columns.col(w, TABLE, "score")[who].astype(np.float64)
        pc_old = columns.col(w, TABLE, "perc")[who].astype(np.float64)
        mot = columns.col(w, TABLE, "mot")[who].astype(np.float64)
        pc = 0.4 * np.where(pc_old >= 0, pc_old, arr) + 0.6 * arr
        rows = np.arange(who.size)
        g = np.maximum(goal, 0)
        prev = sc_old[rows, g]
        new_g = arr[rows, g]
        mot = np.where((goal >= 0) & (prev >= 0), np.clip(mot + 0.5 * (new_g - prev), 0.0, 1.0), mot)
        columns.batch(ctx, TABLE, "learn.assessment", who, {
            "score": arr.astype(np.float32), "perc": pc.astype(np.float32), "mot": mot.astype(np.float32)},
            extra={"t": int(ctx.now), "assessed": int(who.size), "items": int(n), "paper": dg})
        ctx.emit("learn.assessed", {"id": st.new_id("lassess"), "kind": "periodic", "t": int(ctx.now),
                                    "n_takers": int(who.size), "n_items": int(n), "digest": dg}, vis="internal")
        summ = {c: {"mean": round(float(arr[:, j].mean()), 4), "n": int(who.size)} for j, c in enumerate(self.cur)}
        ctx.emit("learn.summary", {"t": ctx.now, "concepts": summ, "assessed": int(who.size)}, vis="public")

    # ------------------------------------------------------------------ public tools
    @endpoint("curriculum", S.obj(), read_only=True)
    def ep_curriculum(self, ctx):
        """The concepts of this world: name, title, problem notation, answer format and prerequisites."""
        return {"concepts": [{"name": c, "title": C.LIBRARY[c].title, "doc": C.LIBRARY[c].doc,
                              "format": C.LIBRARY[c].fmt, "answer_format": C.LIBRARY[c].answer_fmt,
                              "prerequisites": [p for p in C.PREREQS[c] if p in self.cur]} for c in self.cur],
                "about": "Learners study worked problems and exercises written in these notations, in documents "
                         "they find through search, the study board of the forum and their contacts. The Board "
                         "assesses learners on new problems and certifies those who pass its exams.",
                "board": self.cfg["board"]}

    @endpoint("stats", S.obj(), read_only=True)
    def ep_stats(self, ctx):
        """Public education statistics: the latest assessment's mean score per concept, learners enrolled,
        study sessions in the last 7 days, certificates issued, and the most-read lessons (readers, sessions,
        mean rating)."""
        st = ctx.state
        w = ctx.world
        summ = st.meta_get("learn:summary", {})
        week = ctx.now - 7 * DAY
        recent = st.meta_get("learn:sessions_by_day", {})
        n7 = sum(v for d, v in recent.items() if int(d) * DAY >= week)
        lessons = sorted(st.scan("learn_lessons"), key=lambda kv: (-kv[1]["readers"], kv[0]))[:20]
        return {"assessment": summ, "enrolled": int((columns.col(w, TABLE, "state") > 0).sum()),
                "sessions_7d": int(n7), "certificates": len(st.tables["learn_certs"]),
                "lessons": [{"doc": k, "author": v["author"], "readers": v["readers"], "sessions": v["sessions"],
                             "rating": round(v["rating_sum"] / v["rating_n"], 2) if v["rating_n"] else None,
                             "ratings": v["rating_n"]} for k, v in lessons]}

    @endpoint("readers", S.obj(), read_only=True)
    def ep_readers(self, ctx):
        """For authors: how many distinct learners studied each of your documents, sessions and ratings."""
        mine = set(ctx.world.principals_of(ctx.actor))
        return [{"doc": k, "readers": v["readers"], "sessions": v["sessions"], "ratings": v["rating_n"],
                 "rating": round(v["rating_sum"] / v["rating_n"], 2) if v["rating_n"] else None}
                for k, v in sorted(ctx.state.scan("learn_lessons")) if v["author"] in mine]

    @endpoint("certificates", S.obj({"person": S.s_str(maxLength=40)}), read_only=True)
    def ep_certificates(self, ctx, person=None):
        """Certificates the Board issued (public): person, concept, exam score, time."""
        st = ctx.state
        keys = st.find_ids("learn_certs", "person", person) if person else st.ids("learn_certs")
        return [dict(st.tables["learn_certs"][k]) for k in keys[-200:]]

    @endpoint("me", S.obj(), read_only=True)
    def ep_me(self, ctx):
        """Individuals of the population only: your goal, your own test scores, certificates and how you rate your
        own progress."""
        i = ids.index(ctx.actor)
        w = ctx.world
        if i is None or i >= columns.rows(w, TABLE):
            raise NotFound("you are not a member of the population")
        row = self._row(w, i)
        sc = lambda a: {c: round(float(a[k]), 3) for c, k in self.cix.items() if a[k] >= 0}  # noqa: E731
        return {"enrolled": row["state"] > 0, "goal": self.cur[row["goal"]] if row["goal"] >= 0 else None,
                "scores": sc(row["score"]), "placement": sc(row["placed"]), "perceived": sc(row["perc"]),
                "certificates": [c for c, k in self.cix.items() if (row["certs"] >> k) & 1],
                "sessions": row["sessions"]}

    # ------------------------------------------------------------------ evaluator side (never a tool)
    def final_scores(self, world, at: int, salt: str, k: int = 6, who=None, concepts=None,
                     exclude: set | None = None) -> dict[int, dict]:
        """A delayed test at time `at` (no further study assumed) on one fresh paper drawn with the evaluator's own
        random stream, excluding every published problem (or `exclude`, e.g. the problems published in either of two
        worlds being compared: the same paper and the same answer draws then go to both, common random numbers).
        Pure: reads state only."""
        exclude = self.published_keys(world) if exclude is None else set(exclude)
        state = columns.col(world, TABLE, "state")
        rows = np.flatnonzero(state > 0) if who is None else np.asarray(who, np.int64)
        cs = list(concepts or self.cur)
        arr, _, _ = exam(world, rows, cs, int(at), world.rng_for("learn-evaluator", salt), k, exclude, self.cix)
        return {int(i): {c: float(arr[j, q]) for q, c in enumerate(cs)} for j, i in enumerate(rows.tolist())}

    def knowledge(self, world, i: int, at: int) -> dict:
        """Evaluator side: effective mastery per concept and level of individual i at time `at`."""
        rec = Rec.of(world, int(i), self.cur, self.cix)
        return {c: [round(rec.eff(c, L, at), 4) for L in LEVELS] for c in self.cur}

    # ------------------------------------------------------------------ appliers
    @applier("learn.meta")
    def ap_meta(self, ev):
        self.world.state.meta_set("learn:board", ev.payload["board"])

    @applier("learn.author")
    def ap_author(self, ev):
        self.world.state.put("learn_authors", ev.payload["principal"], {"code": int(ev.payload["code"])})

    @applier("learn.sessions")
    def ap_sessions(self, ev):
        st = self.world.state
        by_day = dict(st.meta_get("learn:sessions_by_day", {}))
        day = str(ev.t // DAY)
        for s in ev.payload["sessions"]:
            sid = st.new_id("lsess")
            # `seq`: the recorded event of the session (provenance: modules that use what the learner learned, e.g.
            # the labour market's profile update in domains.firms, name it as their cause)
            st.insert("learn_sessions", sid, {**s, "t": ev.t, "seq": ev.seq})
            les = st.get("learn_lessons", s["doc"])
            base = dict(les) if les else {"author": s["author"], "readers": 0, "sessions": 0, "rating_sum": 0,
                                          "rating_n": 0, "concepts": [], "first_t": ev.t, "reader_set": []}
            rs = list(base["reader_set"])
            if s["i"] not in rs:
                rs.append(s["i"])
            base.update(readers=len(rs), reader_set=rs, sessions=base["sessions"] + 1,
                        concepts=sorted(set(base["concepts"]) | set(s["concepts"])))
            st.put("learn_lessons", s["doc"], base)
            by_day[day] = by_day.get(day, 0) + 1
        st.meta_set("learn:sessions_by_day", {k: v for k, v in by_day.items() if int(k) >= ev.t // DAY - 30})

    @applier("learn.rated")
    def ap_rated(self, ev):
        st = self.world.state
        les = st.get("learn_lessons", ev.payload["doc"])
        if les is not None:
            st.patch("learn_lessons", ev.payload["doc"], rating_sum=les["rating_sum"] + int(ev.payload["stars"]),
                     rating_n=les["rating_n"] + 1)

    @applier("learn.assessed")
    def ap_assessed(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("learn_assessments", p.pop("id"), p)

    @applier("learn.certified")
    def ap_certified(self, ev):
        p = ev.payload
        self.world.state.put("learn_certs", f"{p['person']}|{p['concept']}", dict(p) | {"seq": ev.seq})

    @applier("learn.summary")
    def ap_summary(self, ev):
        self.world.state.meta_set("learn:summary", dict(ev.payload))

    @applier("learn.choices")
    def ap_choices(self, ev):
        pass

    @applier("learn.authored")
    def ap_authored(self, ev):
        pass

    # ------------------------------------------------------------------ invariants
    def check_invariants(self, world):
        errs = []
        try:
            m = columns.col(world, TABLE, "m")
        except NotFound:
            return ["learn table missing"]
        if m.size and (float(m.min()) < 0 or float(m.max()) > 1):
            errs.append("mastery out of [0, 1]")
        pm = columns.col(world, TABLE, "pm")
        if pm.size and float((pm - m).max()) > 1e-5:
            errs.append("stable floor above mastery")
        bug = columns.col(world, TABLE, "bug")
        if bug.size and (float(bug.min()) < 0 or float(bug.max()) > 1):
            errs.append("misconception strength out of [0, 1]")
        st = world.state
        n_sess = int(columns.col(world, TABLE, "sessions").sum())
        if n_sess != len(st.tables["learn_sessions"]):
            errs.append(f"{n_sess} sessions counted, {len(st.tables['learn_sessions'])} recorded")
        for did, les in st.scan("learn_lessons"):
            if les["readers"] != len(les["reader_set"]):
                errs.append(f"{did}: readers differ from the reader set")
        return errs


# ============================================================================================ lesson writing
def textbook(c: str, kind: str, rng) -> str:
    """The Board's incumbent textbook chapter: correct worked examples at level 1 only (kind 'basic') or levels 1-2
    with two exercises (kind 'full')."""
    con = C.LIBRARY[c]
    lines = [f"{con.title} ({c})", "", f"{con.doc} Problems are written `{con.fmt}`.", ""]
    plan = [("example", 1)] * 3 if kind == "basic" else [("example", 1), ("example", 1), ("example", 2),
                                                          ("practice", 1), ("practice", 2)]
    for n, (kd, L) in enumerate(plan, 1):
        p = C.generate(c, rng, L)
        lines += C.write_item(kd, con, p, con.solve(p), "arrow", n)
    return "\n".join(lines)


def authored_lesson(rec: Rec, c: str, careful: float, rng, now: int) -> str:
    """A lesson an individual writes from its own knowledge: plan by its carefulness, answers from its own
    knowledge state (they can be wrong), phrasing in a style of its own."""
    con = C.LIBRARY[c]
    style = str(rng.choice(C.STYLES))
    if careful >= 0.6:
        plan = [("example", 1), ("example", 1), ("practice", 1), ("example", 2), ("practice", 2), ("practice", 2)]
    elif careful >= 0.35:
        plan = [("example", 1), ("example", 2), ("practice", 1), ("practice", 2)]
    else:
        plan = [("practice", 1)] * int(rng.integers(4, 8))
    lines = [f"{con.title} ({c})", "", str(rng.choice([
        f"Here is how I do {c.replace('_', ' ')}.", f"Some worked problems on {con.title.lower()}.",
        f"Practice for {con.title.lower()} - check your answers below.", "Notes from my own studying."])), ""]
    for n, (kd, L) in enumerate(plan, 1):
        p = C.generate(c, rng, L)
        a, _ = rec.answer(c, p, now, rng)
        lines += C.write_item(kd, con, p, a, style, n)
    return "\n".join(lines)
